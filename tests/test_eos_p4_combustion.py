"""EOS refactor P4 — combustion on real O2 (docs/eos_refactor_design.md §5,
decisions log #12). Own pass, once per tick, after the EOS solver materializes
P/N/T (cpp/src/combustion.{h,cpp}); re-points FireSimulation's + ignition's O2
gates to the REAL local N_O2 mean (cpp/src/fire_simulation.cpp,
src/simulation/combat.py); applies the per-gas trace `decay` column
(cpp/src/physics_engine.cpp) — decayed mass simply VANISHES since P-T0
(energy-books arc, design §2.6 — the decay->inert_N2 credit is DELETED,
see test_trace_decay_credits_nothing_to_inert_n2 below).

Two tiers of gate, per the patch instructions:

1. **Exact-integer conservation, ISOLATED from the pre-existing, ALREADY-
   blessed non-conservative trace-advection truncation** (design §1: trace
   gas transport is semi-Lagrangian and "NON-CONSERVATIVE by design ... smoke
   decay is the tuning knob" — smoke_dynamics.h; this was true before P4 and
   is explicitly out of scope here). Decision #12's "N_total conserved"
   claim is about the COMBUSTION and DECAY *transactions themselves* never
   fabricating or destroying mass — the tests in tier 1 isolate exactly that,
   the same way tests/test_eos_p1_species_transport.py isolated donor-cell
   flux's own exactness from everything else.
2. **The four emergent payoffs, as full-engine E2E scenarios** (mechanism
   visible; constants are P4's sane defaults, feel-tuned at P5 per the patch
   scope — assertions here are deliberately qualitative/generous, not tight
   numeric pins, matching "mechanism visible" in the patch instructions).

A NOTED, OUT-OF-SCOPE-FOR-P4 finding (not a P4 bug): a strongly-seeded fire
in a small sealed room can drive the unified `temperature` field to the
Q16.16 ceiling BEFORE this patch existed (reproduced here with combustion
fully disabled, H_fuel=burn_rate=0 — see
`test_thermal_spike_is_pre_existing_not_a_p4_regression`). H_fuel scaling
barely moves the observed peak (not a P4 combustion-heat runaway).

**eos-p3fix-thermal-ceiling investigation (branch `eos-p3fix-thermal-
ceiling`, updates this note):** the plume->T shim's self-limiter WAS
confirmed broken — `sat = 1 - atmosphere[i]/p_expand_ref` reads P at the
plume's OWN tile, which the EOS solver force-zeroes for every SOLID cell
(a fire tile is solid), so `sat` never actually engaged. FIXED: the shim
now gates on T against a `T_FLAME_MAX` physical ceiling
(fire_simulation.h) instead, and every temperature write on the shim's and
the compression-work's paths (fixed_point.h `sat_add_q16`) is now
SATURATING — the "occasionally wraps negative" half of the original bug
report is gone (verified: `temperature` never goes negative-garbage in
this scenario post-fix).
HOWEVER — root-cause instrumentation (per-tick T budget at the fire tile
and its open-air neighbours) showed the shim was NEVER the dominant
driver of the climb (disabling it changes peak_disabled by <1%): the
measured driver is a coupling between TemperatureSolver's Pass-1
`ΔT=ΔE/(N·c_v)` heat-deposit reciprocal (temperature_solver.cpp) and
EOSSolver's step-4c compression-work term (eos_solver.cpp) — as the local
pressure spike this pair creates evacuates a cell's bulk N via donor-cell
flux, the SAME (or accumulating) heat deposit divides by an ever-smaller
N, and compression work's `T *= (1 - k)` update (rate-clamped at
±T_WORK_CLAMP, never value-clamped) then compounds that geometrically
(~1.5x/tick at the clamp rail) — reaching the Q16.16 ceiling within
single-digit ticks REGARDLESS of the shim. `temperature` now correctly
SATURATES there instead of wrapping, but still visibly "climbs to and
pins near the ceiling" in this extreme scenario — the deeper fix (an
absolute, not just rate, safety rail on compression work, without
clipping legitimate high-energy blast physics) is a solver-stability
design call, NOT made unilaterally here; flagged for Erik. A related,
separate finding: `wind_x`/`wind_y` were ALSO observed reaching magnitudes
far past the solver's own `c_LOCAL` cap in this same extreme scenario
(velocity wrap/overflow), independent of whether `temperature` wraps —
not investigated further here (out of scope), flagged for its own pass.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_eos_p4_combustion.py -q
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                   # noqa: E402
from level_loader import LevelData                              # noqa: E402
from simulation.gamemap import GameMap                          # noqa: E402
from simulation.physics_runner import PhysicsRunner             # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD, MaterialTable  # noqa: E402
from simulation.gases import O2, INERT_N2, SMOKE, STEAM  # noqa: E402
from simulation import fire_fixed, gas_fixed                    # noqa: E402

SEED_TICK_DT = 1.0 / 24.0
_TBL = MaterialTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------
def _sealed_room(hh=9, wood_at=None, extra_vacuum=None):
    """A hull-walled square room (MAT_HULL border), MAT_AIR interior, an
    optional MAT_WOOD fuel tile, and optional vacuum cell(s) (a breach)."""
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    if wood_at is not None:
        tm[wood_at] = MAT_WOOD
    ld = LevelData(name="p4_test", version="2", path=Path("."),
                   tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    gmap = GameMap(ld)
    if extra_vacuum:
        for (y, x) in extra_vacuum:
            gmap.is_vacuum[y, x] = True
            gmap.atmosphere[y, x] = 0
            gmap.gas[:, y, x] = 0
            gmap.temperature[y, x] = 0
    return gmap


def _runner(**combustion_overrides):
    pr = PhysicsRunner(bp)
    for k, v in combustion_overrides.items():
        setattr(pr.combustion, k, v)
    return pr


def _step_tick(pr, gmap, dt=None):
    """One GAME-FAITHFUL tick: PhysicsRunner.step + the per-tick `heat` clear.

    v2.4 harness-fidelity fix (eos-p3fix-thermal-ceiling): `heat` is a
    per-tick deposit buffer whose clear deliberately lives at the END of
    Simulation.step (after every heat consumer — simulation.py). A bare
    pr.step loop never clears it, so every past tick's fire re-radiates
    FOREVER (a dead fire keeps heating; measured: T pins at the T_MAX_PHYS
    rail and the flood-differentiation ordering inverts under the stale-heat
    artifact). These E2Es assert GAME behavior, so they must step the way
    the game does.

    P-R4 (ruling amendment 5): `rad_net` and `rad_flux` are per-tick planes with
    EXACTLY the same lifetime and exactly the same failure mode — Simulation.step
    wipes all three together at the end of the tick. Measured when they were
    missed here: the burner's radiative LOSS into the sealed room's cold hull
    re-applied every tick without ever being cleared, dragging its temperature
    to -841 game (below ambient, which the antisymmetric exchange alone can
    never do) and killing every arm of the payoff trio at the same ~35 ticks.
    `dem_acc` is deliberately NOT cleared: it is PERSISTENT synced state whose
    whole job is to carry a sub-count oxygen debt ACROSS ticks."""
    burned = pr.step(gmap, SEED_TICK_DT if dt is None else dt)
    gmap.heat.fill(0)
    gmap.rad_net.fill(0)
    gmap.rad_flux.fill(0)
    return burned


def _ignite(gmap, at, intensity=0.6, temp_mult=1.5):
    """Seed a burning tile: fire intensity + a hot temperature (mirrors what
    apply_temperature_ignition + FireSimulation's own feedback would produce
    over a few ticks, done here in one shot for a tight test scenario)."""
    gmap.fire[at] = fire_fixed.quantize_scalar(float(intensity))
    gmap.temperature[at] = int(IGN_WOOD_Q16 * temp_mult)


def _o2n2_total(gmap):
    """O2 + inert_N2 — the CONSERVATIVE bulk pair alone (test_eos_p1_species_
    transport.py's `_isum2` precedent). Deliberately EXCLUDES smoke:
    `gmap.smoke` IS `gas[SMOKE]` (gases.py), and smoke's own semi-Lagrangian
    transport is separately non-conservative by design (smoke_dynamics.h;
    unrelated to combustion) — so smoke's total is not a P4 conservation
    signal regardless of what feeds it. (Historical note: before P-S1,
    2026-08-15, FireSimulation ALSO had its own independent, unbacked
    per-tick smoke emission here — `smoke[nbr] += emission*dt*I`,
    fire_simulation.cpp — which was the additional reason this sum excluded
    smoke; that scatter is now deleted, docs/smoke_single_source_asbuilt_
    2026-08-15.md, but the exclusion still holds for the transport reason
    above.) Combustion can only ever MOVE mass OUT of this pair (the
    soot_yield fraction leaves permanently to smoke). P-T0 (energy-books
    arc, design §2.6 — the 0% ruling) DELETED the decay->inert_N2 credit
    that used to move some mass back in (decisions #12 v2.1, now retired
    doctrine) — decayed trace counts simply vanish, crediting nothing — so
    this sum is now MONOTONICALLY NON-INCREASING (strictly bounded above by
    its starting value; the premise that "smoke decays back in" no longer
    holds)."""
    return (int(gmap.gas[O2].astype(np.int64).sum())
            + int(gmap.gas[INERT_N2].astype(np.int64).sum()))


# ---------------------------------------------------------------------------
# TIER 1 — exact-integer conservation, isolated (the patch's hard gate)
# ---------------------------------------------------------------------------
def test_combustion_pass_conserves_o2_n2_soot_exactly():
    """CombustionSolver.step, called directly and repeatedly: every burn
    transaction moves mass O2 -> (smoke, inert_N2) with ZERO net
    change to the three-plane sum — the literal "N_total conserved" claim
    of decisions.md #12, isolated from FireSimulation's OWN independent
    (pre-existing, unrelated) smoke-emission source and from any lossy
    trace transport (this test never calls SmokeDynamics.step).

    Re-derivation note (fire-family triage, 547fb12, 2026-07-24): burn
    demand became `burn_rate*I*o2f_j*dt` — PER-CLAIMANT, proportional to
    fire intensity `I` (was a uniform gate). A cell with fire==0 now has
    zero demand regardless of wall_hp/temperature/flammable, so this direct
    CombustionSolver.step fixture must seed `fire[]` nonzero at the burning
    cell for the conservation property (still the same "moves mass, never
    fabricates or destroys it" claim) to actually exercise a burn."""
    h = w = 7
    gas = np.zeros((7, h, w), dtype=np.int32)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.zeros((h, w), dtype=bool)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    fire = np.zeros((h, w), dtype=np.int32)
    ignition_temp_q16 = np.zeros((h, w), dtype=np.int32)
    temperature = np.zeros((h, w), dtype=np.int32)

    cy, cx = 3, 3
    solid[cy, cx] = True
    flammable[cy, cx] = True
    wall_hp[cy, cx] = fire_fixed.quantize_scalar(60.0)
    ignition_temp_q16[cy, cx] = IGN_WOOD_Q16
    temperature[cy, cx] = IGN_WOOD_Q16 * 2
    fire[cy, cx] = fire_fixed.quantize_scalar(0.6)
    # Open-air ring around the fuel tile, ambient O2/N2.
    open_air = ~solid
    gas[O2][open_air] = gas_fixed.quantize_scalar(0.21)
    gas[INERT_N2][open_air] = gas_fixed.quantize_scalar(0.79)

    comb = bp.CombustionSolver()
    comb.burn_rate = 2.0
    comb.o2_thresh_burn = 0.02
    comb.H_fuel = 4.0
    comb.soot_yield = 0.3

    total0 = (int(gas[O2].astype(np.int64).sum())
              + int(gas[INERT_N2].astype(np.int64).sum())
              + int(gas[SMOKE].astype(np.int64).sum()))
    assert total0 > 0, "test setup produced no O2/N2 mass — vacuous"

    any_burn = False
    for _ in range(60):
        comb.step(gas, O2, INERT_N2, SMOKE, temperature, wall_hp, fire,
                  flammable, solid, is_vacuum, ignition_temp_q16,
                  SEED_TICK_DT, 1.0, 0.05)
        total = (int(gas[O2].astype(np.int64).sum())
                 + int(gas[INERT_N2].astype(np.int64).sum())
                 + int(gas[SMOKE].astype(np.int64).sum()))
        assert total == total0, (
            f"combustion transaction leaked/fabricated mass: {total} != {total0}")
        if int(gas[SMOKE].sum()) > 0:
            any_burn = True
    assert any_burn, "the scenario never actually burned — the gate is vacuous"


def test_trace_decay_credits_nothing_to_inert_n2():
    """P-T0 (energy-books arc, 2026-08-17, design §2.6 — the trace 0%
    ruling): the v2.1 decay->inert_N2 credit (physics_engine.cpp's
    run_substeps trace loop) is DELETED — "decay is oxidation, not
    deletion" is retired doctrine now that traces carry zero pressure
    weight (nothing to conserve). This is the INVERSE of the pre-P-T0 gate
    it replaces (`test_trace_decay_credits_inert_n2_exactly`, which
    asserted trace+N2 held constant): decay now simply REMOVES trace mass;
    inert_N2 must be unchanged TO THE LSB. Isolated from the trace plane's
    OWN two ALREADY-blessed non-conservative mechanisms (smoke_dynamics.h:
    "NON-CONSERVATIVE by design ... smoke decay is the tuning knob") — its
    diffusion Laplacian (a real truncation loss even at zero wind: disabled
    here via `gases.diffusion[TEARGAS] = 0`) and its SL advection
    displacement (a no-op here since wind stays exactly 0 in this
    quiescent, no-fire, no-vacuum sealed room — verified below). With those
    isolated, decay is the ONLY mechanism touching either plane, so
    inert_N2's exact-LSB stillness proves the credit is truly gone, not
    just small."""
    gmap = _sealed_room(hh=9)
    pr = _runner()
    # Seed a trace gas (teargas — decay 0.010/s per config.toml) at ambient
    # bulk N; disable ITS diffusion (see docstring) so only decay moves mass.
    from simulation.gases import TEARGAS
    gmap.gases.diffusion[TEARGAS] = 0.0
    interior = (~gmap.solid) & (~gmap.is_vacuum)
    gmap.gas[TEARGAS][interior] = gas_fixed.quantize_scalar(0.5)
    assert np.all(gmap.wind_x == 0) and np.all(gmap.wind_y == 0)

    trace0 = int(gmap.gas[TEARGAS].astype(np.int64).sum())
    n2_0 = int(gmap.gas[INERT_N2].astype(np.int64).sum())
    assert trace0 > 0, "test setup produced no trace mass — vacuous"

    for _ in range(20):
        pr.step(gmap, SEED_TICK_DT)
        n2 = int(gmap.gas[INERT_N2].astype(np.int64).sum())
        assert n2 == n2_0, (
            f"trace decay credited inert_N2 — the P-T0 deletion did not "
            f"hold: {n2} != {n2_0}")
    assert trace0 - int(gmap.gas[TEARGAS].astype(np.int64).sum()) > 0, (
        "no decay was observed over 20 ticks at decay=0.010/s — the gate is "
        "vacuous")


def test_thermal_spike_is_pre_existing_not_a_p4_regression():
    """Documents the out-of-scope-for-P4 finding (module docstring): a
    strongly-seeded fire in a small sealed room still drives `temperature`
    to the Q16.16 ceiling with combustion FULLY DISABLED (H_fuel=
    burn_rate=0) — i.e. it predates this patch. Asserts P4 does not make it
    WORSE (H_fuel's effect on this runaway is provably small, not the
    driver) -- UNCHANGED by the eos-p3fix-thermal-ceiling investigation
    (root-cause: TemperatureSolver Pass-1's heat/N reciprocal x EOSSolver's
    step-4c compression work, confirmed independent of the fire-plume->T
    shim by disabling it — see the module docstring for the full writeup).

    eos-p3fix-thermal-ceiling ALSO adds a regression guard for the half of
    the original bug report that IS now fixed: `temperature` must never go
    below a sane floor (the WRAP — "occasionally wraps negative" in the
    original report). The climb-to-ceiling itself remains open (T still
    reaches the format ceiling in this extreme scenario), so this test
    intentionally does NOT assert a tight peak bound — only that reaching
    the ceiling now SATURATES instead of wrapping through it."""
    def _run(H_fuel, burn_rate):
        gmap = _sealed_room(hh=9, wood_at=(4, 4))
        pr = _runner(H_fuel=H_fuel, burn_rate=burn_rate)
        _ignite(gmap, (4, 4), intensity=0.8, temp_mult=3.0)
        peak = 0.0
        trough = 0.0
        for _ in range(60):
            _step_tick(pr, gmap)   # game-faithful tick (v2.4 harness-fidelity fix)
            peak = max(peak, float(np.abs(gmap.temperature).max()) / 65536.0)
            trough = min(trough, float(gmap.temperature.min()) / 65536.0)
        return peak, trough

    peak_disabled, trough_disabled = _run(0.0, 0.0)
    peak_default, trough_default = _run(4.0, 1.0)   # the shipped defaults
    # *** P-R4 RE-ANCHOR (ruling amendment 5 D2) — THE SPIKE IS GONE. ***
    # This test used to ASSERT the runaway reproduced (`peak_disabled > 5000`),
    # documenting it as pre-existing and out of P4's scope. P-R4 removed its
    # DRIVER. The measured root cause (module docstring) was a feedback between
    # TemperatureSolver Pass-1's dT = dE/(N*c_v) reciprocal and EOSSolver's
    # step-4c compression work: as the pressure spike evacuated a cell's bulk N,
    # the SAME heat deposit divided by an ever-smaller N and compounded
    # geometrically. The deposit feeding that loop was the PAINTER's — one-way
    # energy dumped into every AIR cell a fire's rays crossed. Under the net-T^4
    # exchange air has heat_atten == 0, so by Kirchhoff it neither absorbs nor
    # emits and receives NOTHING: the loop has no input left.
    # The test's LIVE intent — a rails/regression guard on `temperature` — is
    # preserved, inverted: it now guards that the runaway does not come BACK,
    # alongside the wrap guard below that was always its other half.
    SPIKE_CEILING = 5000.0
    assert peak_disabled < SPIKE_CEILING, (
        f"the P3 thermal runaway is BACK (combustion-disabled peak="
        f"{peak_disabled} game >= {SPIKE_CEILING}). P-R4 removed its driver by "
        f"retiring the painter's air deposit — a peak this high means something "
        f"is feeding one-way energy into air cells again")
    assert peak_default < SPIKE_CEILING, (
        f"the P3 thermal runaway is BACK with combustion enabled "
        f"(peak={peak_default} game). Combustion's gas-side H_fuel deposit is a "
        f"BURN-SITE term, not a room-wide paint — it must not reopen the loop")
    # eos-p3fix-thermal-ceiling: no wraparound garbage (the fixed half of
    # the bug). T_MIN is -289 (EOSSolver default); a generous floor below
    # that (any legitimate T_MIN-floored cooling stays well above -1000)
    # catches a reintroduced wrap without being a tight numeric pin.
    assert trough_disabled > -1000.0, (
        f"temperature went implausibly negative (trough={trough_disabled}) "
        "-- looks like the int32 WRAP has regressed (disabled-combustion run)")
    assert trough_default > -1000.0, (
        f"temperature went implausibly negative (trough={trough_default}) "
        "-- looks like the int32 WRAP has regressed (default-combustion run)")


# ---------------------------------------------------------------------------
# TIER 2 — the four emergent payoffs (E2E, qualitative)
# ---------------------------------------------------------------------------
def test_e2e_1_sealed_room_fire_self_starves():
    """(1) A sealed-room fire dims/dies as its LOCAL N_O2 depletes, with fuel
    (wall_hp) still remaining — self-starving, not fuel exhaustion. The
    combustion+decay mass triple (O2+N2+smoke) never exceeds its
    starting value (no fake mass creation); room pressure rises above
    ambient during the burn and comes back down off its peak by the end
    (rise-then-settle, not a monotonic runaway).

    v2.4 re-pins (eos-p3fix-thermal-ceiling):
    - FireSimulation's OWN smoke emission (`smoke[nbr] += emission*dt*I` — a
      pre-existing, unbacked source, unrelated to combustion) used to be
      disabled for this test: emitted-from-nothing smoke decayed into
      inert_N2 via the decisions #12 v2.1 credit, so a longer-lived fire
      drifted the O2+N2 pair ABOVE its start through that unrelated channel
      and the exact `<= total0` bound (this test's actual conservation claim
      about the COMBUSTION+DECAY transactions) would false-positive. P-S1
      (2026-08-15, docs/smoke_single_source_asbuilt_2026-08-15.md) DELETED
      that emission outright (Erik's single-source ruling: combustion soot
      is the ONE fire-smoke source), so there is no longer anything to
      isolate here — the per-test zeroing this bullet used to describe is
      gone along with the field it zeroed.
    - Room 9x9 -> 15x15: under the hot-zone-equilibrium O2 gates
      (config.toml [physics.fire], v2.4 second rescale) the 9x9 box reaches
      a SELF-SUSTAINING SMOLDER — its whole gas mass ends up hot enough
      (thousands of K) to conduct the wood tile back above ignition_temp
      forever, and CombustionSolver (which by P4 design consumes NO wall_hp
      — "wall_damage stays the sole fuel-consumption brake", combustion.h)
      then burns O2 fuel-free for thousands of ticks, so pressure never
      comes off its peak within any test horizon. That regime is physically
      coherent (a sealed oven) but it is NOT this test's story; a 15x15 room
      has the gas thermal mass for the burn to actually END (fire dies
      t~=89, P peaks t~=97, then genuinely declines). The fuel-free-smolder
      regime itself is FLAGGED for Erik's P5 pass (design doc §4 v2.4).

    v2.5 re-pins (P5.1 stoichiometric fuel consumption — design §5 v2.5
    amendment, decisions #17; behavioral BY DESIGN):
    - The flame phase lengthens (t~=89 -> t~=426): the ember-scale fuel
      drain weakens F, a weaker flame draws its O2 down more slowly, and
      the marginal flame rides the low-O2 regime longer before a starve dip
      crosses I_min. Horizon re-pinned 220 -> 520 (perturbation-checked:
      the t=426 death is bit-stable under the v2.4 1e-5 dial-perturbation
      probe).
    - The 'pressure settles off its peak' assertion is RETIRED FOR SEALED
      ROOMS: v2.5's designed ember (I=0, T>=ignition, fuel-metered) keeps
      burning O2 and depositing heat after flame death — hot gas conducts
      the wood back over ignition_temp, so a sealed room is now a SEALED
      OVEN whose pressure keeps rising until its O2 (or fuel) exhausts,
      thousands of ticks out (measured: still climbing at t=2400). That is
      canon now (decisions #17: an ember only goes out at hp <= 1 LSB or
      no O2), so this test instead asserts the post-flame climb happens for
      the RIGHT, FUEL-PAID reason: the tile carries the ember signature
      (T >= ignition) and its wall_hp actually paid for the O2 burned. The
      full flame->ember->re-ignite->char-out->quiet lifecycle is gated in
      tests/test_eos_p5_1_stoich.py."""
    gmap = _sealed_room(hh=15, wood_at=(7, 7))
    pr = _runner()
    # smoke_emission zeroing REMOVED at P-S1 — the field/mechanism it
    # isolated is deleted outright now (see the docstring above).
    _ignite(gmap, (7, 7), intensity=0.6, temp_mult=1.5)

    wall_hp0 = float(gmap.wall_hp[7, 7]) / 65536.0
    total0 = _o2n2_total(gmap)
    ambient_p = float(gmap.atmosphere[6, 7]) / 65536.0
    assert ambient_p > 0.9, "the neighbour tile should start near 1 atm"

    fire_hist, p_hist, mass_hist = [], [], []
    for _ in range(520):   # v2.5 re-pin (was 220): the flame lives to t~=426
        _step_tick(pr, gmap)   # game-faithful tick (v2.4 harness-fidelity fix)
        fire_hist.append(float(gmap.fire[7, 7]) / 65536.0)
        p_hist.append(float(gmap.atmosphere[6, 7]) / 65536.0)
        mass_hist.append(_o2n2_total(gmap))

    # The mechanism: fire actually dies (self-starves), not just decays a
    # little.
    assert fire_hist[-1] == 0.0, (
        f"fire never fully extinguished (final intensity {fire_hist[-1]})")
    # ... with fuel remaining (self-STARVING, not burn-through). v2.5: the
    # ember drain is real but small on this horizon (~4 HP of 60) — the >90%
    # bound still holds and still proves starve-not-burn-through.
    wall_hp_final = float(gmap.wall_hp[7, 7]) / 65536.0
    assert wall_hp_final > 0.9 * wall_hp0, (
        f"the wall burned through instead of starving "
        f"(wall_hp {wall_hp0:.2f} -> {wall_hp_final:.2f})")
    # No mass fabrication: the bulk O2+N2 pair never exceeds its start.
    # P-T0 (design §2.6) deleted the decay->inert_N2 credit (decisions #12
    # v2.1, now retired doctrine) — combustion can now only ever REMOVE mass
    # from this pair, never add any back, so the bound is strict.
    assert max(mass_hist) <= total0, (
        f"O2+N2 exceeded its starting total "
        f"(max {max(mass_hist)} > start {total0}) — mass fabricated")
    # A real pressure rise above ambient during the burn (kept from v2.4).
    p_peak = max(p_hist)
    assert p_peak > ambient_p * 1.05, (
        f"no visible pressure rise during the burn (peak {p_peak:.3f} vs "
        f"ambient {ambient_p:.3f})")
    # v2.5 (replaces the retired settle assertion — see docstring): the
    # post-flame pressure climb is the DESIGNED, FUEL-METERED ember, not a
    # leak or fabrication (the mass bound above already excludes the
    # latter): the wood tile still reads ember-hot and its wall_hp PAID
    # for the post-flame O2 it keeps burning.
    assert int(gmap.temperature[7, 7]) >= IGN_WOOD_Q16, (
        "post-flame regime lost the ember signature (tile T fell below "
        "ignition inside the horizon — expected a live sealed-oven ember)")
    assert wall_hp_final < wall_hp0, (
        "ember drew O2 without paying fuel — the v2.5 stoichiometric drain "
        "is not engaging")


def test_e2e_2_breach_vents_o2_and_kills_fire():
    """(2) A breach that vents the room's air puts out an established fire
    FASTER than the same fire left sealed — venting removes O2 wholesale
    (not just local combustion depletion).

    v2.5 re-pin (P5.1, design §5 v2.5 / decisions #17): the sealed control
    now dies t=265 (was 172) — the ember-scale fuel drain weakens F and the
    weaker flame starves its room more slowly (behavioral by design, trio
    re-measured at the P5.1 gate: sealed 265 / vented 48 / flooded 39).
    max_ticks 250 -> 400.

    P6.9a re-measure (EOS P6.9, docs/eos_p6_9_combustion_design.md §5): the
    two-gather reformulation's deltas gamma (contested cells fully drain O2)
    and delta (multi-source cells deposit aggregate heat) nudge the sealed
    self-starve to t=261 (was 265); vented/flooded unchanged (48 / 39). Trio
    now sealed 261 / vented 48 / flooded 39. Ordering + the perturbation gate
    stay green. This test only asserts vented < sealed, so it is unaffected."""
    def _ticks_to_die(vent, max_ticks=400):
        gmap = _sealed_room(hh=9, wood_at=(4, 4),
                            extra_vacuum=[(0, 4)] if vent else None)
        pr = _runner()
        _ignite(gmap, (4, 4), intensity=0.6, temp_mult=1.5)
        if vent:
            # Breach the hull tile itself (open a hole to the vacuum cell).
            gmap.solid[0, 4] = False
            gmap.dyn_permeability[0, 4] = 1.0
            gmap.material[0, 4] = MAT_AIR
        for t in range(max_ticks):
            _step_tick(pr, gmap)   # game-faithful tick (v2.4 harness-fidelity fix)
            if float(gmap.fire[4, 4]) == 0.0:
                return t
        return None

    t_vented = _ticks_to_die(vent=True)
    t_sealed = _ticks_to_die(vent=False)
    assert t_vented is not None, "the vented fire never died"
    assert t_sealed is not None, "the sealed-room control fire never died (bad scenario)"
    assert t_vented < t_sealed, (
        f"venting should kill the fire SOONER than a sealed room "
        f"(vented={t_vented} ticks, sealed={t_sealed} ticks)")


def test_e2e_3_o2_rich_pocket_intensifies_burn():
    """(3) A local N_O2 spike (an O2-tank rupture) produces a VISIBLY more
    intense burn than the same ignition in ambient air — the O2-tank-
    rupture -> fireball payoff (design §5)."""
    def _peak_intensity(o2_boost):
        gmap = _sealed_room(hh=9, wood_at=(4, 4))
        pr = _runner()
        if o2_boost:
            # A tank-rupture spike at the fire's open neighbours: push their
            # N_O2 well above ambient (0.21), range-checked to fit Q16.16
            # headroom (design §8 P1 note), leaving N_total/N2 untouched
            # (the spike is EXTRA O2, not a species swap).
            for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                gmap.gas[O2][4 + dy, 4 + dx] = gas_fixed.quantize_scalar(3.0)
        _ignite(gmap, (4, 4), intensity=0.3, temp_mult=1.05)
        peak = 0.0
        for _ in range(15):
            _step_tick(pr, gmap)   # game-faithful tick (v2.4 harness-fidelity fix)
            peak = max(peak, float(gmap.fire[4, 4]) / 65536.0)
        return peak

    peak_ambient = _peak_intensity(o2_boost=False)
    peak_boosted = _peak_intensity(o2_boost=True)
    assert peak_boosted > peak_ambient, (
        f"an O2-rich pocket should intensify the burn vs ambient "
        f"(boosted peak={peak_boosted:.4f}, ambient peak={peak_ambient:.4f})")


def test_e2e_4_inert_flood_smothers_fire():
    """(4) Flooding the fire's neighbourhood with inert N2 (displacing O2)
    smothers it faster than leaving it alone — the inert/CO2-flood-smothers
    payoff (design §5, decisions.md item B).

    v2.5 re-pin (P5.1, design §5 v2.5 / decisions #17): the unflooded
    control now dies t=265 (was 172) — see test_e2e_2's re-pin note.
    max_ticks 200 -> 400.

    P6.9a re-measure (design §5): the reformulation moves the unflooded
    control to t=261 (deltas gamma/delta); flooded unchanged at 39. This test
    asserts only flooded < control, so it is unaffected."""
    def _ticks_to_die(flood, max_ticks=400):
        gmap = _sealed_room(hh=9, wood_at=(4, 4))
        pr = _runner()
        _ignite(gmap, (4, 4), intensity=0.6, temp_mult=1.5)
        if flood:
            for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                y, x = 4 + dy, 4 + dx
                gmap.gas[O2][y, x] = 0
                gmap.gas[INERT_N2][y, x] = gas_fixed.quantize_scalar(4.0)
        for t in range(max_ticks):
            _step_tick(pr, gmap)   # game-faithful tick (v2.4 harness-fidelity fix)
            if float(gmap.fire[4, 4]) == 0.0:
                return t
        return None

    t_flooded = _ticks_to_die(flood=True)
    t_control = _ticks_to_die(flood=False)
    assert t_flooded is not None, "the flooded fire never died"
    assert t_control is not None, "the control fire never died (bad scenario)"
    assert t_flooded < t_control, (
        f"an inert-N2 flood should smother the fire SOONER than the control "
        f"(flooded={t_flooded} ticks, control={t_control} ticks)")


# ---------------------------------------------------------------------------
# Determinism — two-run digest match
# ---------------------------------------------------------------------------
def _digest_after(ticks):
    gmap = _sealed_room(hh=9, wood_at=(4, 4))
    pr = _runner()
    _ignite(gmap, (4, 4), intensity=0.6, temp_mult=1.5)
    for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        gmap.gas[O2][4 + dy, 4 + dx] = gas_fixed.quantize_scalar(2.0)
    h = hashlib.sha256()
    for _ in range(ticks):
        pr.step(gmap, SEED_TICK_DT)
        for arr in (gmap.gas, gmap.temperature, gmap.fire, gmap.wall_hp,
                    gmap.atmosphere, gmap.wind_x, gmap.wind_y):
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def test_two_run_determinism():
    """The same combustion+decay+fire scenario, run twice from scratch,
    produces a BIT-IDENTICAL per-tick field trajectory (integer Q16.16,
    pure-integer combustion/decay arithmetic — no float, no RNG)."""
    d1 = _digest_after(80)
    d2 = _digest_after(80)
    assert d1 == d2, f"two identical runs diverged: {d1} != {d2}"


# ---------------------------------------------------------------------------
# v2.4 — perturbation robustness of the payoff orderings
# ---------------------------------------------------------------------------
def _payoff_timings(perturb_absorb=None, max_ticks=400):
    """Ticks-to-extinguish for the sealed / vented / flooded arms of the
    e2e-1/2/4 scenario family (game-faithful loop), optionally with one EOS
    dial perturbed. Returns (sealed, vented, flooded).

    v2.5 (P5.1, design §5 v2.5 / decisions #17): trio re-measured at the
    P5.1 gate — sealed 265 / vented 48 / flooded 39 (was 172 / 49 / 39).
    Behavioral by design (the ember-scale fuel drain weakens F; the weaker
    sealed flame starves its room more slowly); ordering preserved and still
    bit-stable under the 1e-5 dial perturbation below. max_ticks 300 -> 400
    for the longer sealed arm.

    P6.9a re-measure (EOS P6.9, docs/eos_p6_9_combustion_design.md §5): the
    two-gather reformulation re-pins the trio to sealed 261 / vented 48 /
    flooded 39 (was 265 / 48 / 39). Only the sealed arm moved (-4 ticks):
    deltas gamma (contested air cells fully drain their O2) and delta (multi-
    source air cells deposit ONE aggregate heat term against the post-burn
    N_total, running marginally hotter) shift the self-starve point; the
    vented/flooded arms die too fast, and with too few multi-source contested
    cells, for the deltas to register. Ordering (flooded < vented < sealed)
    preserved and still bit-stable under the perturbation probe."""
    def _ticks(vent=False, flood=False):
        gmap = _sealed_room(hh=9, wood_at=(4, 4),
                            extra_vacuum=[(0, 4)] if vent else None)
        pr = _runner()
        if perturb_absorb is not None:
            pr.eos.absorb_strength = perturb_absorb
        _ignite(gmap, (4, 4), intensity=0.6, temp_mult=1.5)
        # P-R4 re-anchor (ruling amendment 5 D2): run the burner at the arc's
        # BLESSED cool_shift (9), not the shipped 5. This test is about O2
        # DIFFERENTIATION — sealed vs vented vs flooded must die at different
        # times BECAUSE of oxygen. At the shipped cool_shift the burner's one
        # loss channel is a 1.33 s e-fold, which under the retired painter's
        # 1600-scale free energy did not matter and now does: every arm dies
        # THERMALLY at ~34 ticks before the oxygen difference can register
        # (measured 34/34/35 — the ordering did not break, it was ERASED). The
        # P-R5 joint tune owns that dial (ruling §4: "cool_shift may drift up —
        # radiation is now explicit"); this test owns the O2 axis, so it pins
        # the dial the rest of the arc measures at.
        gmap.cool_shift[4, 4] = 9
        if vent:
            gmap.solid[0, 4] = False
            gmap.dyn_permeability[0, 4] = 1.0
            gmap.material[0, 4] = MAT_AIR
        if flood:
            for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                y, x = 4 + dy, 4 + dx
                gmap.gas[O2][y, x] = 0
                gmap.gas[INERT_N2][y, x] = gas_fixed.quantize_scalar(4.0)
        for t in range(max_ticks):
            _step_tick(pr, gmap)
            if float(gmap.fire[4, 4]) == 0.0:
                return t
        return None

    return _ticks(), _ticks(vent=True), _ticks(flood=True)


def test_payoff_orderings_perturbation_robust():
    """v2.4 (eos-p3fix-thermal-ceiling): the O2-differentiation payoffs must
    be REAL physics, not chaos artifacts. The investigation measured that the
    pre-v2.4 timings rode the chaotic near-format-ceiling regime (a 1e-5
    relative perturbation of ONE dial moved the flamethrower's dist-3
    ignition from t=40 to t=60). This test turns that finding into a gate:
    perturb `absorb_strength` by 1e-5 RELATIVE and assert (a) the payoff
    orderings survive (flooded < vented < sealed — flood suffocates at once,
    venting drains the room, sealed takes longest), and (b) every timing
    stays within a small window of its baseline (a chaotic regime moves
    timings by tens of ticks; a physical one barely at all)."""
    base = _payoff_timings()
    pert = _payoff_timings(perturb_absorb=8.0 * (1.0 + 1e-5))

    for name, (s, v, f) in (("baseline", base), ("perturbed", pert)):
        assert None not in (s, v, f), f"{name}: an arm never extinguished {s, v, f}"
        assert f < v < s, (
            f"{name}: payoff ordering broken (flooded={f}, vented={v}, "
            f"sealed={s}) — expected flooded < vented < sealed")
    for arm, b, p in zip(("sealed", "vented", "flooded"), base, pert):
        window = max(3, int(0.10 * b))   # ±10% (min 3 ticks) — chaos moved
                                          # timings ~50% at the same epsilon
        assert abs(p - b) <= window, (
            f"{arm}: timing chaos-fragile under a 1e-5 dial perturbation "
            f"(baseline {b}, perturbed {p}, window ±{window})")


if __name__ == "__main__":
    test_combustion_pass_conserves_o2_n2_soot_exactly()
    test_trace_decay_credits_nothing_to_inert_n2()
    test_thermal_spike_is_pre_existing_not_a_p4_regression()
    test_e2e_1_sealed_room_fire_self_starves()
    test_e2e_2_breach_vents_o2_and_kills_fire()
    test_e2e_3_o2_rich_pocket_intensifies_burn()
    test_e2e_4_inert_flood_smothers_fire()
    test_two_run_determinism()
    print("OK: EOS P4 combustion tests passed")
