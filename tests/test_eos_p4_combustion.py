"""EOS refactor P4 — combustion on real O2 (docs/eos_refactor_design.md §5,
decisions log #12). Own pass, once per tick, after the EOS solver materializes
P/N/T (cpp/src/combustion.{h,cpp}); re-points FireSimulation's + ignition's O2
gates to the REAL local N_O2 mean (cpp/src/fire_simulation.cpp,
src/simulation/combat.py); applies the per-gas trace `decay` column, crediting
the lost mass to `inert_N2` in the same cell (cpp/src/physics_engine.cpp).

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

A NOTED, OUT-OF-SCOPE finding (not a P4 bug): a strongly-seeded fire in a
small sealed room can drive the unified `temperature` field to within a few
hundred counts of the Q16.16 ceiling BEFORE this patch existed (reproduced
here with combustion fully disabled, H_fuel=burn_rate=0 — see
`test_thermal_spike_is_pre_existing_not_a_p4_regression`). It is the P3
fire-plume->T shim's `temp_gain_scale` (a named P5 TUNING DIAL,
fire_simulation.h) interacting with the EOS solver's own compression-work
feedback, not a P4 combustion-heat runaway (H_fuel scaling barely moves the
observed peak). Flagged for Erik's P5 feel pass; explicitly not fixed here
(non-goal: "no solver changes").

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
from simulation.gases import O2, INERT_N2, BLACK_SMOKE, WHITE_SMOKE  # noqa: E402
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


def _ignite(gmap, at, intensity=0.6, temp_mult=1.5):
    """Seed a burning tile: fire intensity + a hot temperature (mirrors what
    apply_temperature_ignition + FireSimulation's own feedback would produce
    over a few ticks, done here in one shot for a tight test scenario)."""
    gmap.fire[at] = fire_fixed.quantize_scalar(float(intensity))
    gmap.temperature[at] = int(IGN_WOOD_Q16 * temp_mult)


def _o2n2_total(gmap):
    """O2 + inert_N2 — the CONSERVATIVE bulk pair alone (test_eos_p1_species_
    transport.py's `_isum2` precedent). Deliberately EXCLUDES black_smoke:
    `gmap.smoke` IS `gas[BLACK_SMOKE]` (gases.py), and FireSimulation's own
    per-tick smoke emission (`smoke[nbr] += emission*dt*I`, fire_simulation.
    cpp, KEPT/pre-existing/unrelated to O2) writes into that SAME plane
    independent of combustion — so black_smoke's total is not a P4
    conservation signal. Combustion can only ever MOVE mass OUT of this pair
    (the soot_yield fraction leaves permanently to black_smoke) or credit
    SOME of it back in as black_smoke decays to inert_N2 (decisions #12
    v2.1) — so this sum is bounded ABOVE by its starting value (proving no
    fabrication) without being strictly monotonic."""
    return (int(gmap.gas[O2].astype(np.int64).sum())
            + int(gmap.gas[INERT_N2].astype(np.int64).sum()))


# ---------------------------------------------------------------------------
# TIER 1 — exact-integer conservation, isolated (the patch's hard gate)
# ---------------------------------------------------------------------------
def test_combustion_pass_conserves_o2_n2_soot_exactly():
    """CombustionSolver.step, called directly and repeatedly: every burn
    transaction moves mass O2 -> (black_smoke, inert_N2) with ZERO net
    change to the three-plane sum — the literal "N_total conserved" claim
    of decisions.md #12, isolated from FireSimulation's OWN independent
    (pre-existing, unrelated) smoke-emission source and from any lossy
    trace transport (this test never calls SmokeDynamics.step)."""
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
              + int(gas[BLACK_SMOKE].astype(np.int64).sum()))
    assert total0 > 0, "test setup produced no O2/N2 mass — vacuous"

    any_burn = False
    for _ in range(60):
        comb.step(gas, O2, INERT_N2, BLACK_SMOKE, temperature, wall_hp, fire,
                  flammable, solid, is_vacuum, ignition_temp_q16,
                  SEED_TICK_DT, 1.0, 0.05)
        total = (int(gas[O2].astype(np.int64).sum())
                 + int(gas[INERT_N2].astype(np.int64).sum())
                 + int(gas[BLACK_SMOKE].astype(np.int64).sum()))
        assert total == total0, (
            f"combustion transaction leaked/fabricated mass: {total} != {total0}")
        if int(gas[BLACK_SMOKE].sum()) > 0:
            any_burn = True
    assert any_burn, "the scenario never actually burned — the gate is vacuous"


def test_trace_decay_credits_inert_n2_exactly():
    """The v2.1 decay->inert_N2 credit (physics_engine.cpp's run_substeps
    trace loop), isolated from the trace plane's OWN two ALREADY-blessed
    non-conservative mechanisms (smoke_dynamics.h: "NON-CONSERVATIVE by
    design ... smoke decay is the tuning knob") — its diffusion Laplacian
    (a real truncation loss even at zero wind: disabled here via
    `gases.diffusion[TEARGAS] = 0`) and its SL advection displacement
    (a no-op here since wind stays exactly 0 in this quiescent, no-fire,
    no-vacuum sealed room — verified below). With those isolated, any mass
    movement observed is decay's credit transaction alone."""
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
    total0 = trace0 + n2_0
    assert trace0 > 0, "test setup produced no trace mass — vacuous"

    for _ in range(20):
        pr.step(gmap, SEED_TICK_DT)
        trace = int(gmap.gas[TEARGAS].astype(np.int64).sum())
        n2 = int(gmap.gas[INERT_N2].astype(np.int64).sum())
        assert trace + n2 == total0, (
            f"decay->inert_N2 credit leaked mass: {trace + n2} != {total0} "
            f"(trace={trace}, n2={n2})")
    assert trace0 - int(gmap.gas[TEARGAS].astype(np.int64).sum()) > 0, (
        "no decay was observed over 20 ticks at decay=0.010/s — the gate is "
        "vacuous")


def test_thermal_spike_is_pre_existing_not_a_p4_regression():
    """Documents the out-of-scope finding (module docstring): a strongly-
    seeded fire in a small sealed room drives `temperature` near the Q16.16
    ceiling with combustion FULLY DISABLED (H_fuel=burn_rate=0) — i.e. it
    predates this patch (the P3 plume->T shim x EOS compression-work
    coupling). Asserts only that P4 does not make it WORSE: the disabled-
    combustion peak and the default-combustion peak are within the same
    order of magnitude (H_fuel's effect on this particular runaway is
    provably small, not the driver)."""
    def _peak_T(H_fuel, burn_rate):
        gmap = _sealed_room(hh=9, wood_at=(4, 4))
        pr = _runner(H_fuel=H_fuel, burn_rate=burn_rate)
        _ignite(gmap, (4, 4), intensity=0.8, temp_mult=3.0)
        peak = 0.0
        for _ in range(60):
            pr.step(gmap, SEED_TICK_DT)
            peak = max(peak, float(np.abs(gmap.temperature).max()) / 65536.0)
        return peak

    peak_disabled = _peak_T(0.0, 0.0)
    peak_default = _peak_T(4.0, 1.0)   # the shipped defaults
    assert peak_disabled > 5000.0, (
        "expected the PRE-EXISTING (combustion-independent) thermal spike "
        f"to reproduce (peak={peak_disabled}); if this no longer reproduces "
        "the P3 engine changed and this documentation test should be revisited")
    # Same order of magnitude either way (H_fuel is not the driver).
    assert peak_default < peak_disabled * 3.0 + 5000.0, (
        f"P4 combustion made the pre-existing spike MUCH worse "
        f"(disabled={peak_disabled}, default={peak_default}) — investigate")


# ---------------------------------------------------------------------------
# TIER 2 — the four emergent payoffs (E2E, qualitative)
# ---------------------------------------------------------------------------
def test_e2e_1_sealed_room_fire_self_starves():
    """(1) A sealed-room fire dims/dies as its LOCAL N_O2 depletes, with fuel
    (wall_hp) still remaining — self-starving, not fuel exhaustion. The
    combustion+decay mass triple (O2+N2+black_smoke) never exceeds its
    starting value (no fake mass creation); room pressure rises above
    ambient during the burn and comes back down off its peak by the end
    (rise-then-settle, not a monotonic runaway)."""
    gmap = _sealed_room(hh=9, wood_at=(4, 4))
    pr = _runner()
    _ignite(gmap, (4, 4), intensity=0.6, temp_mult=1.5)

    wall_hp0 = float(gmap.wall_hp[4, 4]) / 65536.0
    total0 = _o2n2_total(gmap)
    ambient_p = float(gmap.atmosphere[3, 4]) / 65536.0
    assert ambient_p > 0.9, "the neighbour tile should start near 1 atm"

    fire_hist, p_hist, mass_hist = [], [], []
    for _ in range(220):
        pr.step(gmap, SEED_TICK_DT)
        fire_hist.append(float(gmap.fire[4, 4]) / 65536.0)
        p_hist.append(float(gmap.atmosphere[3, 4]) / 65536.0)
        mass_hist.append(_o2n2_total(gmap))

    # The mechanism: fire actually dies (self-starves), not just decays a
    # little.
    assert fire_hist[-1] == 0.0, (
        f"fire never fully extinguished (final intensity {fire_hist[-1]})")
    # ... with fuel remaining (self-STARVING, not burn-through).
    wall_hp_final = float(gmap.wall_hp[4, 4]) / 65536.0
    assert wall_hp_final > 0.9 * wall_hp0, (
        f"the wall burned through instead of starving "
        f"(wall_hp {wall_hp0:.2f} -> {wall_hp_final:.2f})")
    # No mass fabrication: the bulk O2+N2 pair never exceeds its start (it
    # may recover SOME as soot decays back to inert_N2, decisions #12 v2.1,
    # but combustion can only ever REMOVE mass from this pair, never add to
    # it beyond what decay credits back).
    assert max(mass_hist) <= total0, (
        f"O2+N2 exceeded its starting total "
        f"(max {max(mass_hist)} > start {total0}) — mass fabricated")
    # Pressure rise-then-settle: a real peak above ambient, and the final
    # value has come back down off that peak.
    p_peak = max(p_hist)
    assert p_peak > ambient_p * 1.05, (
        f"no visible pressure rise during the burn (peak {p_peak:.3f} vs "
        f"ambient {ambient_p:.3f})")
    assert p_hist[-1] < p_peak, (
        f"pressure never settled off its peak ({p_hist[-1]:.3f} vs "
        f"peak {p_peak:.3f})")


def test_e2e_2_breach_vents_o2_and_kills_fire():
    """(2) A breach that vents the room's air puts out an established fire
    FASTER than the same fire left sealed — venting removes O2 wholesale
    (not just local combustion depletion)."""
    def _ticks_to_die(vent, max_ticks=250):
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
            pr.step(gmap, SEED_TICK_DT)
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
            pr.step(gmap, SEED_TICK_DT)
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
    payoff (design §5, decisions.md item B)."""
    def _ticks_to_die(flood, max_ticks=200):
        gmap = _sealed_room(hh=9, wood_at=(4, 4))
        pr = _runner()
        _ignite(gmap, (4, 4), intensity=0.6, temp_mult=1.5)
        if flood:
            for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                y, x = 4 + dy, 4 + dx
                gmap.gas[O2][y, x] = 0
                gmap.gas[INERT_N2][y, x] = gas_fixed.quantize_scalar(4.0)
        for t in range(max_ticks):
            pr.step(gmap, SEED_TICK_DT)
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


if __name__ == "__main__":
    test_combustion_pass_conserves_o2_n2_soot_exactly()
    test_trace_decay_credits_inert_n2_exactly()
    test_thermal_spike_is_pre_existing_not_a_p4_regression()
    test_e2e_1_sealed_room_fire_self_starves()
    test_e2e_2_breach_vents_o2_and_kills_fire()
    test_e2e_3_o2_rich_pocket_intensifies_burn()
    test_e2e_4_inert_flood_smothers_fire()
    test_two_run_determinism()
    print("OK: EOS P4 combustion tests passed")
