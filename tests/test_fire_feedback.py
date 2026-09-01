"""Fire intensity FEEDBACK model (fire_design_proposal §2/§3/§5).

The cellular spread is DELETED; spread is now radiation -> heat -> temperature ->
ignition (covered by test_fire_heat_source.py / test_temperature_ignition.py).
This file pins the new per-tile life/death of an already-lit fire:

  T     = temperature[i] / TEMP_SCALE            (the conduction-pass field)
  F     = clamp01(wall_hp[i] / fuel_ref)         (fuel from remaining wall HP)
  O2    = mean REAL n_o2 over open (non-solid,non-vacuum) 4-neighbours
          (EOS refactor P4 — was the atmosphere/P proxy; design §6 item 3)
  W     = |wind| from the shared wind field
  hot   = clamp01((T - fire_T_ext) / fire_T_span)
  o2    = smoothstep(P_min, P_full, O2)          (the REAL local O2 gate)
  avail = F * o2
  grow  = k_grow * avail * hot * I*(1-I) * (1 + k_wind_fan*W)
  die   = k_die*(1 - avail*hot)*I + k_wind_strip*W*(1-I)*I
  I    += dt*(grow - die);  clamp01;  if I < I_min -> 0
  atmosphere[i] += max(fire_pressure_gain*I*(1 - atmosphere[i]/p_expand_ref)*dt, 0)

Cases: a fed fire GROWS; a starved one (cold OR no fuel OR low pressure) DECAYS to
0; a vented room DIES (decompression); a fire whose wall_hp runs out DIES
(burnout); wind FANS a big fire and BLOWS OUT a small one (crossover); the plume
deposit raises the fire's own atmosphere so wind points OUTWARD; spread happens
via radiation->ignition with the cellular stencil gone; determinism (bit-identical
fire field); and a conservative-default fire does NOT firestorm a wood room.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fire_feedback.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402
from simulation.materials import (  # noqa: E402
    MaterialTable, MAT_AIR, MAT_WOOD, MAT_HULL,
)
from simulation import fire_fixed  # noqa: E402  S3a: gmap.fire is int32 Q16.16
from simulation import atmosphere_fixed  # noqa: E402  S2c: atmosphere/wind Q16.16
from simulation import wall_fixed  # noqa: E402  S3b: wall_hp is int32 Q16.16
from simulation import gas_fixed  # noqa: E402  the plume rewrite's O2 refill
from simulation.gases import O2  # noqa: E402  the plume rewrite's O2 refill

# The plume rewrite (below) reuses the P4 module's sealed-room fixture + the
# game-faithful tick helper, the same precedent test_eos_p5_1_stoich.py sets
# for sharing this scenario builder across files (one source of truth).
sys.path.insert(0, str(ROOT / "tests"))
from test_eos_p4_combustion import _sealed_room, _runner, _ignite, _step_tick  # noqa: E402

# S3a: gmap.fire is int32 Q16.16. Helpers for the Simulation-based tests that
# seed / read the field at real intensity (the FireSimulation.step stub tests
# below keep their own float arrays — the C++ step signature is still float).
FIRE_Q = fire_fixed.quantize_scalar          # real [0,1] -> Q16.16 int
FIRE_DEQ = lambda q: float(q) / fire_fixed.FP_ONE_F   # noqa: E731  Q16.16 -> real
FIRE_001_Q = fire_fixed.quantize_scalar(0.01)         # the >0.01 "lit" threshold

TEMP_SCALE = 65536          # Q16.16 (== HEAT_SCALE), shared heat/temperature domain
_TBL = MaterialTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])   # 300 * 65536
DT = 0.1                    # a generous game-tick-sized step for the feedback


def _params_runner():
    """A FireSimulation with the config [physics.fire] defaults bound (exactly
    what the game runs), plus those defaults read back for assertions."""
    r = PhysicsRunner(bp)
    return r.fire


# ---------------------------------------------------------------------------
# Synthetic single-tile feedback harness. A 3x3 grid with a flammable WOOD tile
# in the centre and AIR around it; the air ring supplies the pressure (O2 proxy)
# read. We drive temperature / wall_hp / atmosphere / wind directly and step the
# C++ FireSimulation in isolation (no level, no renderer) so each feedback term
# is exercised cleanly and deterministically.
# ---------------------------------------------------------------------------
class _FeedbackScene:
    # S3b: the C++ FireSimulation.step is now INTEGER end-to-end. fire/atmosphere/
    # wall_hp/wind are int32 Q16.16; the scene quantizes its real-valued inputs and
    # reads the fire intensity back via dequantize. temperature/smoke were already
    # int32. (atmosphere is the S2c Q16.16 domain, wall_hp the S3b one.)
    #
    # EOS refactor P4 (design §6, item 3): the O2 GATE now reads a SEPARATE
    # `n_o2` plane, not `atmosphere` (which still drives the UNRELATED plume
    # saturation term). `o2` defaults to the SAME value as `atm` so every
    # existing atm-only caller keeps testing "non-limiting O2" exactly as
    # before; a test exercising the O2-starvation path overrides `o2` directly.
    def __init__(self, *, I, T=0.0, wall_hp=60.0, atm=1.0, o2=None, wind=0.0):
        m = np.full((3, 3), MAT_AIR, dtype=np.int8)
        m[1, 1] = MAT_WOOD
        self.material = m
        self.flammable = np.ascontiguousarray(_TBL.flammable[m])
        self.solid = np.ascontiguousarray(_TBL.permeability[m] <= 0.0)
        self.is_vacuum = np.zeros((3, 3), dtype=bool)
        # Air ring carries the chosen pressure; the solid centre holds none. (Q16.16)
        self.atmosphere = np.where(
            self.solid, 0, atmosphere_fixed.quantize_scalar(float(atm))
        ).astype(np.int32)
        o2_val = atm if o2 is None else o2
        self.n_o2 = np.where(
            self.solid, 0, atmosphere_fixed.quantize_scalar(float(o2_val))
        ).astype(np.int32)
        # CONTINUOUS-O2 LAW (547fb12, 2026-07-24): the gate stopped reading the
        # ABSOLUTE n_o2 and now reads the MOLE FRACTION X = Sn_o2/Sn_total over
        # the open 4-neighbours, so `step` takes the denominator plane too.
        # `atm` IS this scene's gas density (it is the value `o2` defaults to),
        # which is the same n_o2 = X*density / n_total = density idiom the law's
        # own tests use (test_continuous_o2_law._step_once). Consequences, both
        # of them the pre-547fb12 gate value bit-for-bit:
        #   * atm-only callers (o2 is None) -> X == 1.0 -> o2f == 1, the
        #     NON-LIMITING gate they have always tested;
        #   * the `o2` override reads straight through as the fraction o2/atm,
        #     so the low-O2 seed still lands below the extinction limit.
        self.n_total = np.where(
            self.solid, 0, atmosphere_fixed.quantize_scalar(float(atm))
        ).astype(np.int32)
        self.smoke = np.zeros((3, 3), dtype=np.int32)
        self.wall_hp = np.zeros((3, 3), dtype=np.int32)
        self.wall_hp[1, 1] = wall_fixed.quantize_scalar(float(wall_hp))
        self.temperature = np.zeros((3, 3), dtype=np.int32)
        self.temperature[1, 1] = int(round(float(T) * TEMP_SCALE))
        self.fire = np.zeros((3, 3), dtype=np.int32)
        self.fire[1, 1] = fire_fixed.quantize_scalar(float(I))
        # Uniform wind in +x (magnitude `wind`). The feedback only reads |wind|. (Q16.16)
        self.wind_x = np.full((3, 3), atmosphere_fixed.quantize_scalar(float(wind)),
                              dtype=np.int32)
        self.wind_y = np.zeros((3, 3), dtype=np.int32)

    def step(self, fire_sim, dt=DT):
        fire_sim.step(
            self.fire, self.atmosphere, self.n_o2, self.n_total, self.smoke,
            self.wall_hp, self.temperature, self.wind_x, self.wind_y,
            self.solid, self.is_vacuum, self.flammable,
            dt,
        )
        return float(self.fire[1, 1]) / fire_fixed.FP_ONE_F


# ---------------------------------------------------------------------------
# Fed fire GROWS; starved fires DECAY to 0
# ---------------------------------------------------------------------------
def test_fed_fire_grows():
    # Hot (T=500 >> fire_T_ext 350), fuelled (wall_hp 60 -> F=1), pressured
    # (atm 1.0 -> o2=1), no wind. grow > die -> intensity climbs.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.3, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    after = sc.step(fs)
    assert after > 0.3, f"a fed fire should grow (0.3 -> {after})"


def test_cold_fire_decays_to_zero():
    # No heat (T=0 -> hot=0): grow=0, die = k_die*(1-0)*I. Fire dies out.
    #
    # RESTATE (fire-family triage, 2026-08-30, dial promotion 9016cd7: k_die
    # 2.0->0.008 slows the pure-k_die decay tail by the same ~250x): 200
    # ticks no longer reaches 0 (measured: still at 0.365+ at tick 200).
    # Measured directly: it DOES still reach exactly 0, at tick 2334 — this
    # ODE has no O2/fuel floor to get stuck on, it is a genuine (slow)
    # asymptotic decay to zero. max_ticks re-derived from that measurement
    # with ~30% margin (a real property still holds — it dies — just later).
    fs = _params_runner()
    sc = _FeedbackScene(I=0.5, T=0.0, wall_hp=60.0, atm=1.0, wind=0.0)
    last = 0.5
    for _ in range(3000):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a cold fire must decay to 0, stuck at {last}"


def test_no_fuel_fire_decays_to_zero():
    # Hot + pressured but NO fuel (wall_hp 0 -> F=0 -> avail=0): fire starves.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.5, T=500.0, wall_hp=0.0, atm=1.0, wind=0.0)
    last = 0.5
    for _ in range(200):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a fire with no fuel must die, stuck at {last}"


def test_low_o2_fire_decays_to_zero():
    # Hot + fuelled + FULL pressure but LOW REAL O2 (EOS refactor P4, design
    # §6 item 3: the O2 gate now reads a separate n_o2 field — the seed sits
    # below P_min so o2=0 -> avail=0): oxygen starvation extinguishes the
    # fire even though the room's overall pressure (atm) is nominal.
    # v2.4 re-pin (eos-p3fix-thermal-ceiling): P_min moved 0.126 -> 0.01 (the
    # hot-zone-equilibrium rescale, config.toml [physics.fire]); the low-O2
    # seed moves 0.02 -> 0.005 to stay below the gate it exercises.
    #
    # RESTATE (fire-family triage, 2026-08-30, dial promotion 9016cd7): same
    # slow-decay tail as test_cold_fire_decays_to_zero (avail==0 either way,
    # so it is the same ODE) — measured DIES at tick 2334, just past 200.
    # max_ticks re-derived with margin.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.5, T=500.0, wall_hp=60.0, atm=1.0, o2=0.005, wind=0.0)
    last = 0.5
    for _ in range(3000):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a low-O2 fire must die, stuck at {last}"


def test_vented_room_extinguishes():
    # A fire fully surrounded by VACUUM (a hull breach drained the room) reads
    # P=0 (no open, non-vacuum neighbour) -> o2=0 -> dies. The
    # decompression-extinguishes-fire loop via a READ, not a kill-threshold.
    #
    # RESTATE (fire-family triage, 2026-08-30, dial promotion 9016cd7): same
    # slow-decay tail as the other avail==0 tests above — measured DIES at
    # tick 2400, just past 200. max_ticks re-derived with margin.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.6, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    sc.is_vacuum[:] = True
    sc.is_vacuum[1, 1] = False            # the burning solid itself is not vacuum
    last = 0.6
    for _ in range(3000):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"a vacuum-vented fire must die, stuck at {last}"


# ---------------------------------------------------------------------------
# Burnout: wall_hp runs out -> the fire starves (the fuel brake)
# ---------------------------------------------------------------------------
def test_burnout_when_wall_hp_runs_out():
    # Hot + pressured + a SMALL fuel store. wall_damage drains wall_hp every step;
    # as wall_hp -> 0, F -> 0, the fire starves and dies. (It may also breach the
    # wall at hp<=0 and zero the fire that way — either is a valid burnout.)
    #
    # RESTATE (fire-family triage, 2026-08-30, dial promotion 9016cd7): once
    # F falls low the tail is the same slow k_die-dominated decay as the
    # other tests above — measured DIES at tick 4920 (later than the pure
    # avail==0 cases since F takes time to drain and the fire is only fully
    # starved once it does). max_ticks re-derived with margin; this scene's
    # step is a bare 3x3 FireSimulation.step call (~0.02s for 5000 ticks),
    # so the larger budget costs nothing measurable.
    #
    # RE-DERIVED (R1, fire session #12, 2026-09-01, docs/fire_3c_design_
    # 2026-09-01.md "Ruling R1"): this fixture's `atm=1.0, o2=None` idiom
    # reads X == 1.0 (n_o2 == n_total == atm), which under the renormalized
    # sustain law clamps at the NEW o2f_cap == 5.0 (was o2f == 1.0 pre-R1,
    # since X=1.0 pure-O2-normalized to o2_frac_full==1.0 saturated at
    # exactly 1). That much bigger `avail` — combined with the R1 I_cap_
    # per_avail re-size (14.0 -> 0.95, the closed-form value re-derived from
    # the b1 open-control bench's MEASURED plateau availability, config.toml's
    # own comment) — lets the fire coast on a much smaller F for much longer
    # before dying: measured DIES at tick 10834 (wall_hp still 0.52 — an
    # I_min snap-extinguish, not burn-through). max_ticks re-derived with
    # margin.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.6, T=500.0, wall_hp=3.0, atm=1.0, wind=0.0)
    # Make it a real wall so burn-through can fire if hp hits 0.
    sc.solid[1, 1] = True
    hp0 = float(sc.wall_hp[1, 1])
    last = 0.6
    for _ in range(15000):
        last = sc.step(fs)
        if last == 0.0:
            break
    assert last == 0.0, f"fire should burn out as fuel runs out, stuck at {last}"
    assert float(sc.wall_hp[1, 1]) < hp0, "wall_hp did not deplete (no burn)"


# ---------------------------------------------------------------------------
# Wind: FANS a big fire, BLOWS OUT a small one (the crossover)
# ---------------------------------------------------------------------------
def test_wind_fans_a_big_fire():
    # Same well-fed big fire, with vs without wind. The (1 + k_wind_fan*W) factor
    # means the windy one grows MORE per step (a firestorm forming).
    #
    # RE-DERIVED (R1, fire session #12, 2026-09-01, docs/fire_3c_design_
    # 2026-09-01.md "Ruling R1"): this fixture's `atm=1.0, o2=None` idiom
    # reads X == 1.0, which under the renormalized sustain law clamps at the
    # NEW o2f_cap == 5.0 (was o2f == 1.0 pre-R1) — combined with the R1
    # I_cap_per_avail re-size (14.0 -> 0.95, the closed-form value, config.
    # toml's own comment), growth is now so fast BOTH scenes saturate at
    # I==1.0 within 3-4 ticks, erasing the comparison by tick 5 (measured:
    # calm 0.96/windy 1.0 at tick 3, both 1.0 from tick 4 on). Compare
    # EARLIER in the ramp instead, before saturation swallows the signal — 2
    # ticks, where windy is still measurably ahead (measured: calm 0.827 vs
    # windy 0.951 at tick 2).
    fs = _params_runner()
    calm = _FeedbackScene(I=0.6, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    windy = _FeedbackScene(I=0.6, T=500.0, wall_hp=60.0, atm=1.0, wind=1.0)
    for _ in range(2):
        calm.step(fs)
        windy.step(fs)
    assert windy.fire[1, 1] > calm.fire[1, 1], (
        f"wind should fan a big fire hotter "
        f"(windy {windy.fire[1, 1]:.4f} vs calm {calm.fire[1, 1]:.4f})")


def _ticks_to_die(scene, fire_sim, max_ticks=400):
    for t in range(1, max_ticks + 1):
        if scene.step(fire_sim) == 0.0:
            return t
    return None


def test_wind_blows_out_a_small_fire():
    # The realistic crossover (fire_design_proposal §5): the SAME gust that fans a
    # big blaze SNUFFS a small/marginal one. A small, marginally-fed fire (low
    # drive: T just above extinction -> `hot` small) is in the band where the
    # k_wind_strip*W*(1-I)*I blow-out dominates the k_wind_fan growth, so a strong
    # wind extinguishes it STRICTLY FASTER than calm air does. (A fully-fed big
    # fire under the same wind GROWS — see test_wind_fans_a_big_fire — that is the
    # crossover: big fanned, small snuffed.)
    #
    # RESTATE (fire-family triage, 2026-08-30): `k_wind_strip = 0.0` in the
    # shipped config (P-K0, 9016cd7 — "plume self-blow-out off, 2026-07-23")
    # — the blow-out TERM this test is named for is switched OFF, full stop,
    # not merely slow. Measured directly, this is not just "neither dies":
    # I_cap_per_avail's promotion (2.53->14.0) also means this T=360
    # "marginal" scene is no longer near death at all — BOTH calm and windy
    # GROW to a real equilibrium (calm settles ~0.52, windy ~0.64, both
    # peaking higher along the way), and windy is STRICTLY BIGGER than calm
    # throughout (k_wind_fan, the surviving wind term, still fans it) — the
    # exact OPPOSITE of "wind blows it out". Restated to document the
    # measured fact rather than hide it behind a loosened bound: the
    # mechanism is dormant BY CONFIG, and with it dormant, wind only fans
    # here (matching test_wind_fans_a_big_fire's mechanism, just smaller).
    # FLAGGED FOR ERIK (per the triage brief) — left as a documenting
    # assertion, not a design call: is k_wind_strip meant to be 0 (the
    # blow-out crossover intentionally retired), or is this a dial that
    # should have moved with the rest of the 9016cd7 promotion?
    #
    # RE-DERIVED (R1, fire session #12, 2026-09-01, docs/fire_3c_design_
    # 2026-09-01.md "Ruling R1"): this fixture's `atm=1.0, o2=None` idiom
    # reads X == 1.0, which under the renormalized sustain law clamps at the
    # NEW o2f_cap == 5.0 (was o2f == 1.0 pre-R1) — combined with the R1
    # I_cap_per_avail re-size (14.0 -> 0.95, the closed-form value, config.
    # toml's own comment), BOTH scenes now settle to a much higher,
    # closer-together equilibrium than before (calm ~0.250, windy ~0.257 by
    # tick 1000). The OLD 1.1x margin no longer holds at steady state; windy
    # is still STRICTLY, measurably ahead of calm throughout the run (ratio
    # ~1.03 at tick 1000) — that structural fact (fan, not blow-out) is what
    # this test guards, so the margin is re-derived to the new measured
    # steady-state ratio (~1.03x) rather than loosened arbitrarily.
    fs = _params_runner()
    assert float(fs.params.k_wind_strip) == 0.0, (
        "k_wind_strip is no longer 0 — the blow-out mechanism may be back; "
        "if so, this test's ORIGINAL crossover claim (windy dies sooner) "
        "should be re-tested and restored, not left dormant")
    calm = _FeedbackScene(I=0.15, T=360.0, wall_hp=60.0, atm=1.0, wind=0.0)
    windy = _FeedbackScene(I=0.15, T=360.0, wall_hp=60.0, atm=1.0, wind=5.0)
    for _ in range(1000):
        i_calm = calm.step(fs)
        i_windy = windy.step(fs)
    assert i_windy > i_calm * 1.02, (
        f"with k_wind_strip dormant, wind should FAN this scene (not blow it "
        f"out) — expected windy strictly ahead of calm (windy={i_windy:.4f}, "
        f"calm={i_calm:.4f})")


# ---------------------------------------------------------------------------
# Plume: a burning tile's heat raises its own temperature and pushes wind
# (= -grad p) OUTWARD (smoke pushed away, not pulled in)
# ---------------------------------------------------------------------------
# REWRITE (fire-family triage, 2026-08-30): the plume->T SHIM this test used
# to isolate (`fs.step` taking `atmosphere`/`n_o2`/`n_total` directly, no
# combustion) was deleted at 25a9823 (2026-07-31) — T is owned outright by
# TemperatureSolver now, and there is no more standalone shim to poke. Ruled
# out as redundant: tests/_fire_bench.py (energy-conservation harness, not a
# pytest gate, and not about wind direction) and the P4 e2e tests in
# tests/test_eos_p4_combustion.py (O2-differentiation payoffs, no wind
# assertions at all — confirmed by grep) do NOT already cover "a burning
# tile's own temperature rises and the wind around it points outward", so
# this is a genuine gap, not a redundant test to delete. Replaced with a
# PIPELINE-level assertion: a real PhysicsRunner tick loop (not an isolated
# FireSimulation.step call) on the P4 module's sealed-room fixture.
#
# Measured directly, building this replacement: a REAL burning tile draws
# real local O2, and early on that SUCTION (not the heat push) dominates the
# wind at its immediate neighbours — inward, the opposite of the old shim's
# claim, and genuinely so (not a bug: the fire is consuming oxygen faster
# than it is heating the room). The heat push only visibly wins later, and
# in a SEALED room never settles (measured: net inward on average over
# ticks 30-120, oscillating with the room's pressure reflections). Non-
# limiting O2 (refilled every tick at the fire's 4 neighbours, the same
# "tank-rupture" idiom test_eos_p4_combustion.py's test_e2e_3 uses) removes
# the suction competitor exactly as the old shim's signature (`atmosphere`/
# `n_o2` non-limiting, "unrelated to O2") intended — under that scene,
# temperature climbs monotonically and wind turns and STAYS outward from
# ~tick 35 on (measured: positive at every 5-tick sample through tick 80).
def test_plume_raises_own_atmosphere_wind_points_outward():
    """A burning tile's own temperature rises over the run, and once its
    local O2 is not the limiting factor (refilled each tick, matching the
    old shim's non-limiting-O2 setup), the wind at its 4 immediate
    neighbours nets OUTWARD (smoke pushed away, not drawn in) over the
    back half of a 60-tick run — the property test_eos_p3_gate_
    measurements.md documents as a "clean outward transient" once the
    combustion-side O2 draw is out of the way."""
    gmap = _sealed_room(hh=15, wood_at=(7, 7))
    pr = _runner()
    cy, cx = 7, 7
    t_before = float(gmap.temperature[cy, cx])
    _ignite(gmap, (cy, cx), intensity=0.6, temp_mult=1.5)

    TICKS = 60
    OUTWARD_FROM = 35   # re-derived: measured outward-and-stable from here
    out_sums = []
    for t in range(1, TICKS + 1):
        # Non-limiting O2 at the 4 immediate neighbours (the tank-rupture
        # idiom, test_e2e_3) — isolates the heat-push signal from the real
        # O2-suction transient that otherwise dominates early (see module
        # note above).
        for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            gmap.gas[O2][cy + dy, cx + dx] = gas_fixed.quantize_scalar(2.0)
        _step_tick(pr, gmap)
        if t >= OUTWARD_FROM:
            out_n = -float(gmap.wind_y[cy - 1, cx])
            out_s = float(gmap.wind_y[cy + 1, cx])
            out_w = -float(gmap.wind_x[cy, cx - 1])
            out_e = float(gmap.wind_x[cy, cx + 1])
            out_sums.append(out_n + out_s + out_w + out_e)

    t_after = float(gmap.temperature[cy, cx])
    assert t_after > t_before, (
        f"a burning tile must raise its own temperature over the run "
        f"({t_before / 65536.0:.1f} -> {t_after / 65536.0:.1f} game)")
    assert all(s > 0.0 for s in out_sums), (
        f"wind at the fire's 4 immediate neighbours should net OUTWARD "
        f"(away from the fire) for every sampled tick from {OUTWARD_FROM} "
        f"on, not just on average: {out_sums}")


def test_plume_does_not_subtract_atmosphere_near_fire():
    # Guard against the DELETED backwards O2-consumption: the fire must NOT lower
    # the atmosphere of any neighbour (that sucked smoke IN). After stepping, no
    # open neighbour's atmosphere is below its start value.
    fs = _params_runner()
    sc = _FeedbackScene(I=0.8, T=500.0, wall_hp=60.0, atm=1.0, wind=0.0)
    before = sc.atmosphere.copy()
    for _ in range(5):
        sc.step(fs)
    # Every air tile is >= its starting atmosphere (no suction toward the fire).
    air = ~sc.solid
    assert np.all(sc.atmosphere[air] >= before[air] - 1e-6), (
        "fire must not subtract atmosphere from its neighbours (the old O2 sink)")


# ---------------------------------------------------------------------------
# Spread is now radiation -> ignition (the cellular stencil is GONE)
# ---------------------------------------------------------------------------
def test_spread_is_radiation_only_no_cellular_stencil():
    """Radiation is the ONLY spread path, and it does not leap gaps.

    RE-ANCHORED AT P-F1a. The scenario had the near target FACE-ADJACENT to the
    burner and asserted that it IGNITED. Two P-F1a facts move that:

      1. v7 rule 3 — CONTACT FACES ARE RADIATION-INERT. A face-adjacent plank
         receives nothing radiatively at all; conduction owns contact. The near
         target is therefore AIR-SEPARATED now, which is the geometry "spread by
         radiation" actually means.
      2. At P-F1a's FROZEN dials the near target warms but does NOT reach wood's
         300-game ignition_temp (measured ~183). That is the patch's named,
         expected outcome, and it is P-F1b's recalibration to restore — see the
         strict-xfail twin in tests/test_fire_heat_source.py.

    So this test now gates what is BOTH strictly true and what it was really
    protecting: heat reaches the NEAR tile and reaches the FAR tile NOT AT ALL.
    The gap-leaping cellular stencil is gone, and nothing has quietly replaced
    it — including the long emission rays, which is worth pinning precisely
    BECAUSE v7 rule 4 lengthened them to the grid diagonal.
    """
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=11, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    g.material[50, 14] = MAT_WOOD          # burner
    g.material[50, 15] = MAT_AIR           # the open tile it radiates ACROSS
    g.material[50, 16] = MAT_WOOD          # near target (radiation reaches it)
    g.material[50, 40] = MAT_WOOD          # far target (no heat path)
    g._update_caches()
    # P-R4 re-anchor (ruling amendment 5 D2). Two changes, same scenario, same
    # intent — "radiation is the ONLY spread path, and it does not leap gaps":
    #  1. the burner is held HOT as well as lit. The retired painter deposited
    #     `k_fire_heat * I` regardless of the tile's temperature; the net-T^4
    #     exchange radiates against the emitter's OWN temperature, so a
    #     synthetically-lit ice-cold tile emits E[0] and correctly heats nothing.
    #     In play a burning tile sits at the ~440 game plateau (P-R4 gate f).
    #  2. the pair runs at the arc's blessed cool_shift 9 rather than the shipped
    #     5 — see the same note in tests/test_fire_heat_source.py. P-R5 owns that
    #     dial; this test owns the spread PATH.
    g.cool_shift[50, 14] = 9
    g.cool_shift[50, 16] = 9
    sim.set_paused(False)

    near_peak = 0
    for _ in range(120):
        g.fire[50, 14] = FIRE_Q(0.8)       # hold the burner (S3a: Q16.16)
        g.temperature[50, 14] = FIRE_Q(443.0)   # ...at flame temperature (P-R4)
        sim.step()
        near_peak = max(near_peak, int(g.temperature[50, 16]))
    print(f"\nP-F1a spread - near (air-separated) target reached "
          f"{near_peak / 65536.0:.1f} game; far target fire={int(g.fire[50, 40])}")
    assert near_peak > 0, (
        "the near air-separated target never warmed - radiation is not "
        "spreading heat at all")
    assert near_peak / 65536.0 > 100.0, (
        f"the near target only reached {near_peak / 65536.0:.1f} game - far "
        f"below the ~183 the frozen dials deliver")
    # The far tile, with no heat reaching it, stayed cold and unlit -> the old
    # gap-leaping cellular stencil is gone, and the >= grid-diagonal emission
    # rays (v7 rule 4) have NOT quietly become a new one.
    assert g.fire[50, 40] == 0, "a far flammable tile lit with no heat path"
    # RE-ANCHOR (v7 rule 4). The old assertion was "the far tile is not heated
    # AT ALL" (|T| < 1 K, allowing only the EOS solver's numerical residue).
    # That was only ever true because emission rays expired at ~5 tiles — the
    # very corridor leak rule 4 exists to close. A real fire DOES radiate across
    # an open room, and the far tile is in clear line of sight 24 tiles down the
    # same row, so it correctly receives a little: measured ~1.3 game.
    #
    # What this test is actually guarding is that the fire does not SPREAD by
    # anything other than radiation, and that radiation's reach is governed by
    # 1/r ray DENSITY rather than by a stencil. Both are now stated as a RATIO:
    # the far tile must stay orders of magnitude below the near one, and must
    # never light. A stencil (or a bug that made distance free) would collapse
    # that ratio immediately.
    far_T = abs(int(g.temperature[50, 40]))
    ratio = near_peak / max(far_T, 1)
    print(f"  far target reached {far_T / 65536.0:.2f} game "
          f"({ratio:.0f}x below the near target)")
    assert ratio > 20.0, (
        f"the far tile is within {ratio:.1f}x of the near tile - radiation has "
        f"stopped falling off with distance, or a gap-leaping stencil is back")
    assert far_T < IGN_WOOD_Q16 // 4, (
        "the far tile is being driven toward ignition from 24 tiles away")


# ---------------------------------------------------------------------------
# Determinism: same scene -> bit-identical fire field
# ---------------------------------------------------------------------------
def test_feedback_determinism_bit_identical():
    def run():
        fs = _params_runner()
        sc = _FeedbackScene(I=0.4, T=500.0, wall_hp=60.0, atm=1.0, wind=0.7)
        for _ in range(30):
            sc.step(fs)
        return sc.fire.copy()

    a = run()
    b = run()
    assert np.array_equal(a, b), "fire feedback is not bit-identical across runs"
    assert float(a[1, 1]) > 0.0, "scene produced no fire (nothing to compare)"


# ---------------------------------------------------------------------------
# Conservative defaults do NOT firestorm a wood-filled room
# ---------------------------------------------------------------------------
def test_conservative_default_does_not_firestorm_wood_room():
    # A hollow WOOD room, ignite ONE tile, run a few ticks with the SHIPPED config
    # [physics.fire] defaults. With cellular spread deleted and the heat path
    # gentle, the structure must NOT be engulfed in a handful of ticks.
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=2, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    y0, x0 = 40, 10
    for d in range(8):
        g.material[y0, x0 + d] = MAT_WOOD
        g.material[y0 + 7, x0 + d] = MAT_WOOD
        g.material[y0 + d, x0] = MAT_WOOD
        g.material[y0 + d, x0 + 7] = MAT_WOOD
    g._update_caches()
    n_wood = int((g.material == MAT_WOOD).sum())
    assert n_wood >= 20
    sim.set_paused(False)
    counts = []
    for _ in range(5):
        # hold the seed lit (S3a: integer max on the Q16.16 field)
        g.fire[y0, x0] = max(int(g.fire[y0, x0]), FIRE_Q(0.8))
        sim.step()
        counts.append(int((g.fire > FIRE_001_Q).sum()))
    assert max(counts) < n_wood // 2, (
        f"a lone fire firestormed the wood room too fast: {counts} of {n_wood}")
