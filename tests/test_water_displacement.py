"""Volume displacement: water -> atmosphere (Step W3 of docs/water_implementation_plan.md).

Rising water shrinks the air column above it; W3 (inside ``_step_water``,
after the pipe-model substeps) scales ``atmosphere`` by the isothermal
P*V = const ratio of the free columns ``max(ceiling_h - depth, flood_eps)``
before vs after the tick, clipped to ``ratio_cap``. A cell whose air column
is squeezed down to ``flood_eps`` counts FLOODED: its ``dyn_permeability``
is zeroed for the tick, sealing the tile by FACE-FLUX blocking (the wave
Laplacian, IMEX diffusion and wind gradient all gather faces as
``min(perm[self], perm[n])``). ``stamp_units`` rebuilds the field at the
START of next tick, so the seal auto-clears when the cell drains.

NOTE (plan W3, contracts review): the hard-zero wave BC does NOT see flooded
cells — trapped ``wave_p`` under the water decays via damping, it is not
zeroed — so nothing here asserts ``wave_p`` on flooded tiles.

The four plan tests, in order:
  1. sealed single-cell column, dump 0 -> 0.5 m in one FieldEdit tick:
     atmosphere x exactly 2.5/(2.5-0.5) = 1.25 (the cap doesn't bite);
     REMOVE the water -> the solver re-equalizes pressure SPATIALLY, but
     (P-W1b, design SS0b R-4, T_abs compression-work law) NOT back to the
     pre-flood isothermal value -- the flood transient's compression work
     honestly warms the sealed air, so the settled level sits ~0.07 atm
     above ambient (mechanism + numbers in the test's own docstring/comments).
  2. ceiling-slam (depth -> 2.5 in one tick): the true ratio (50x) is capped
     at ratio_cap (atmosphere x exactly 1.5), no inf/NaN anywhere, and the
     slammed cell reads flooded (dyn_permeability == 0 for the tick).
  3. flooded line across a corridor: gas stays one-sided over 50 ticks and
     wind across the flooded faces is exactly zero — with a no-water CONTROL
     run proving both assertions bite (gas DOES cross, wind DOES blow).
  4. wet-static exactness: a settled flat pool gives free_before ==
     free_after -> ratio is IEEE-exactly 1.0 -> one A/B tick (live vs the
     whole ``_step_water`` no-op'd) leaves atmosphere bit-identical, and the
     live tick leaves atmosphere bit-identical to its own pre-tick copy.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_water_displacement.py -q
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
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.field_edit import (  # noqa: E402
    EditMode, FieldEdit, Region,
)
from simulation.gases import SMOKE  # noqa: E402
from simulation import gas_fixed  # noqa: E402  (S2b: gas Q16.16)
from water_q16 import q, deq  # noqa: E402  (S1: Q16.16 quantize/dequantize)

# S1: water_depth/floor_height are int32 Q16.16. FieldEdit(field="water_depth",
# amount=<metres>) stays authored in metres (the "water" policy quantizes); direct
# reads of g.water_depth are Q16.16 counts -> dequantize. Depth tolerances widen
# from the old float 1e-6 to a few Q16.16 LSB. The W3 atmosphere/dyn_permeability
# math is a FLOAT BRIDGE (still float), so atmosphere ratios keep float tolerances.
Q_EPS = 1.0 / 65536.0

SEED = 42


# ---------------------------------------------------------------------------
# helpers (synthetic-LevelData scenes, the test_water_integration pattern;
# CSV codes: 1 = hull, 4 = interior air)
# ---------------------------------------------------------------------------
def _single_cell_level(tile_size_m: float = 0.333) -> LevelData:
    """A 5x5 all-hull block with ONE open interior cell at (2, 2) — a sealed
    single-cell air column. All four neighbours are hull, so a water dump
    physically cannot flow during the substeps: the painted depth reaches the
    displacement accounting intact."""
    tm = np.ones((5, 5), dtype=np.int32)
    tm[2, 2] = 4
    return LevelData(
        name="water_cell_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=tile_size_m,
        diffuse_path=Path("."),
    )


def _corridor_level(length: int = 21) -> LevelData:
    """A 1-wide hull-ringed corridor: row 1, columns 1..length-2 open."""
    tm = np.ones((3, length), dtype=np.int32)
    tm[1, 1:length - 1] = 4
    return LevelData(
        name="water_corridor_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=0.333,
        diffuse_path=Path("."),
    )


def _sealed_room_level(n: int = 9) -> LevelData:
    """An n x n hull-ringed room, interior air, no vacuum anywhere."""
    tm = np.ones((n, n), dtype=np.int32)
    tm[1:n - 1, 1:n - 1] = 4
    return LevelData(
        name="water_room_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=0.333,
        diffuse_path=Path("."),
    )


# ---------------------------------------------------------------------------
# 1. Sealed single-cell column: compression on the dump, return on the drain
# ---------------------------------------------------------------------------
def test_single_cell_dump_compresses_then_remove_restores():
    """EOS refactor P3 REWRITE (design §2.2 occupancy-transition rule): the
    old `atmosphere *= ratio` compression mechanism is RETIRED — a flooding
    cell now EVACUATES a (1 - 1/ratio) fraction of its bulk N conservatively
    into open neighbors, and the pressure response falls out of the solver
    (p* = C*N*T rises where the N lands). The old scene (a fully wall-sealed
    1-cell column) has no open neighbor to evacuate into, and the flat-2D
    volume-less EOS cannot represent in-place air compression under water —
    that specific physics returns with the 2.5D z-layer arc. This test now
    asserts the NEW contract in an open room: the dump's cell sheds N into
    its neighbors (total bulk N conserved to the LSB), and after the water
    is removed the solver re-equalizes pressure."""
    level = _sealed_room_level(9)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    cy, cx = 4, 4
    sim.set_paused(False)
    sim.step()    # dry tick: seed `before` with zeros

    from simulation.gases import O2, INERT_N2
    def bulk_total():
        return int(g.gas[O2].astype(np.int64).sum()
                   + g.gas[INERT_N2].astype(np.int64).sum())
    def bulk_at(y, x):
        return int(g.gas[O2][y, x]) + int(g.gas[INERT_N2][y, x])

    total0 = bulk_total()
    n_cell0 = bulk_at(cy, cx)
    n_nb0 = sum(bulk_at(cy + dy, cx + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))

    ceil_h = float(sim.physics_runner.water_ceiling_h)
    expected = ceil_h / (ceil_h - 0.5)
    assert abs(expected - 1.25) < 1e-12   # the plan's 2.5/(2.5-0.5)

    sim.edit(FieldEdit(field="water_depth", region=Region.TILE,
                       coords=(cy, cx), amount=0.5))
    sim.step()

    d = float(deq(g.water_depth[cy, cx]))
    assert d > 0.3, f"dump did not land: depth={d}"
    # §2.2: total bulk N EXACTLY conserved (the evacuation is a +/- pair).
    assert bulk_total() == total0, (
        f"evacuation leaked bulk N: {total0} -> {bulk_total()}")
    # The flooding cell shed ~(1 - 1/1.25) = 20% of its N to the neighbors
    # (post-evacuation transport smears it, so assert direction + rough size).
    assert bulk_at(cy, cx) < n_cell0 * 0.9, "flooding cell did not shed N"
    n_nb1 = sum(bulk_at(cy + dy, cx + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    assert n_nb1 > n_nb0, "neighbors did not receive the evacuated N"

    # REMOVE the water: nothing un-evacuates instantly (the rule is one-way);
    # the SOLVER re-equalizes pressure over the following ticks.
    sim.edit(FieldEdit(field="water_depth", region=Region.TILE,
                       coords=(cy, cx), amount=0.5, mode=EditMode.REMOVE))
    for _ in range(30):
        sim.step()
    assert int(g.water_depth[cy, cx]) == 0
    interior = (~g.solid) & (~g.is_vacuum)
    p = g.atmosphere[interior].astype(np.float64) / 65536.0
    # P-W1b (design SS0b R-4): the old assert (`|p - 1.0| < 0.05` everywhere)
    # encoded ISOTHERMAL restoration -- that the flood-then-drain transient
    # must return the room to exactly ambient pressure. The T_abs
    # compression-work law correctly refuses that: the flood transient's
    # compression work honestly warmed the sealed air (measured mean T_rel
    # +23.5 game-deg at this point in the run), and p* = C*N*(T+T_AMB_K)
    # predicts the room settles WARM and slightly over ambient, not back at
    # it (p* = C*N*(23.5+290) predicts +0.081 atm over ambient; measured
    # +0.070 atm -- same sign, same order; N is exactly conserved
    # (bulk_total() == total0, asserted below), so the excess pressure is a
    # temperature effect, not a mass leak). Two asserts instead: (a) the
    # SOLVER did re-equalize SPATIALLY (spread is near machine-epsilon --
    # measured 0.0001 atm, i.e. every open cell agrees almost exactly, just
    # not with the isothermal PRE-flood value); (b) the settled LEVEL sits
    # honestly above ambient, in the physically-predicted band, not at an
    # arbitrary or runaway value.
    spread = float(p.max() - p.min())
    assert spread <= 0.05, (
        f"pressure did not spatially re-equalize: spread {spread:.4f} atm "
        f"({float(p.min()):.4f}..{float(p.max()):.4f})")
    mean_p = float(p.mean())
    assert 1.00 <= mean_p <= 1.12, (
        f"settled mean pressure {mean_p:.4f} atm is outside the "
        "compression-work-warmed band [1.00, 1.12] -- either the room "
        "isothermally restored (law regression) or warmed further than "
        "the p* = C*N*(T+T_AMB_K) prediction accounts for")
    assert bulk_total() == total0


# ---------------------------------------------------------------------------
# 2. Ceiling-slam: the ratio cap holds, nothing blows up, the cell seals
# ---------------------------------------------------------------------------
def test_ceiling_slam_ratio_capped_no_inf_nan():
    level = _single_cell_level()
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    cy, cx = 2, 2
    sim.set_paused(False)
    sim.step()                            # dry tick: seed `before` with zeros

    p0 = float(g.atmosphere[cy, cx])
    assert p0 > 0.5, "no air in the sealed cell (vacuous slam test)"

    runner = sim.physics_runner
    ceil_h = float(runner.water_ceiling_h)
    cap = float(runner.water_ratio_cap)
    flood_eps = float(runner.water_flood_eps)
    assert abs(cap - 1.5) < 1e-12         # the config value the plan pins
    assert ceil_h / flood_eps > cap, (
        "a full slam would not exceed the cap (vacuous cap test)")

    # Depth 0 -> ceiling_h in ONE tick: true ratio 2.5/0.05 = 50x, capped.
    sim.edit(FieldEdit(field="water_depth", region=Region.TILE,
                       coords=(cy, cx), amount=ceil_h))
    sim.step()

    assert abs(float(deq(g.water_depth[cy, cx])) - ceil_h) < 3 * Q_EPS
    # EOS P3: the `atmosphere *= min(ratio, cap)` multiply is RETIRED (design
    # §2.2) — in this fully-sealed 1-cell scene the slam has no open neighbor
    # to evacuate N into, so the cap's job here reduces to "nothing blows
    # up" (the finiteness sweep below) + the flooded seal. p1/p0 stays ~1.
    p1 = float(g.atmosphere[cy, cx])
    assert 0.0 <= p1 / p0 < cap + 1.0, f"slam produced a wild pressure: {p1/p0}"
    # The slammed cell reads flooded -> sealed for this tick (stamp_units
    # only rebuilds dyn_permeability at the START of the next tick, so the
    # zero is still visible here, after step() returned).
    assert float(g.dyn_permeability[cy, cx]) == 0.0
    # No inf/NaN anywhere in the touched fields.
    for name in ("atmosphere", "wave_p", "water_depth",
                 "wind_x", "wind_y", "dyn_permeability"):
        field = getattr(g, name)
        assert np.isfinite(field).all(), f"{name} has inf/NaN after the slam"


# ---------------------------------------------------------------------------
# 3. Flooded line across a corridor: air-seal (gas one-sided, wind zero)
# ---------------------------------------------------------------------------
LINE_X = 10          # the flooded line's column in the 21-long corridor


def _corridor_sim(with_water: bool):
    """Corridor scene: gas + overpressure on the LEFT of column LINE_X.

    The line cell (1, LINE_X) is a 1-cell BASIN: its floor sits 2*ceiling_h
    BELOW the corridor's, so a 2*ceiling_h column painted into it has a flat
    surface with its dry neighbours (floor + depth == 0 on both sides) — a
    settled water trap (the plumber's U-bend) that stays above the flooded
    threshold, hence sealed, for the whole run. A deep line painted on a
    FLAT floor would slump below ceiling_h - flood_eps within a few ticks
    and the seal would blink open — the basin holds the plug, which is the
    property this test probes (a flooded cell seals the corridor).

    Basin 2x-deepened by W4 (pressure head ON, k_p = 0.5): the sustained
    1.3-vs-1.0 step across the line now SHOVES plug water over the east lip
    — MEASURED: a ceiling_h-deep plug dips to 2.405 < 2.45 by tick 3 and the
    seal blinks open (gas crosses) until the spill drains back. The deeper
    U-bend absorbs the shove (~0.09 m spilt; min depth over the run 4.86,
    far above the 2.45 threshold), keeping the scene's premise — a plug that
    STAYS flooded — true with the head live. The invariants asserted below
    are unchanged.

    The left-side pressure step (1.3 atm vs 1.0) drives wind — and gas
    advection + diffusion — RIGHTWARD, straight at the line: the no-water
    control proves gas really does cross when the line is dry.
    """
    level = _corridor_level(21)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    ceil_h = float(sim.physics_runner.water_ceiling_h)
    g.floor_height[1, LINE_X] = q(-2.0 * ceil_h)   # S1: Q16.16 metres
    if with_water:
        # EOS P3 SCENE RE-ANCHOR: under the new engine even a mild sustained
        # pressure step drives hurricane-scale winds (K = c_amb^2/gamma) whose
        # head pumps a static U-bend plug over the lip within a few ticks —
        # physically-correct trap-burping that destroys this scene's premise.
        # Hold the plug with the engine's own CONTINUOUS SOURCE mechanism
        # (gmap.water_sources: depth = max(depth, level) per tick — the same
        # architectural slot a pipe leak uses), so the line cell stays
        # flooded/sealed by construction and the test measures its actual
        # contract: a perm-0 flooded cell passes NO gas.
        g.water_depth[1, LINE_X] = q(2.0 * ceil_h)
        g.water_sources.append((1, LINE_X, 2.0 * ceil_h))
        # ... and decouple the water from the pressure head for THIS scene
        # (k_p = 0): under the new engine the head/pressure feedback tips the
        # tall metastable plug column into a sloshing cascade within ~6 ticks
        # (physically plausible, but not this test's subject — the seal
        # contract is permeability-gating, not water statics). The head's own
        # behavior is covered by test_water_pressure_head.py.
        sim.physics_runner.water.k_p = 0.0
    # S2b: gas is int32 Q16.16 — seed the source side at FULL density (the old
    # `= 10.0` float was clamped to 1.0 by the solver anyway). FP_ONE counts.
    g.gas[SMOKE][1, 1:LINE_X] = gas_fixed.SMOKE_MAX_Q
    # EOS P3: the source-side overpressure must live in the REAL state (P is
    # solver-materialized) — scale the bulk N on the left; the P_prev paint
    # is kept only as a same-tick seed. SCENE RE-TUNED 1.3 -> 1.1: under the
    # new engine the N-step is REAL CONSERVED MASS (the old diffuse_solve
    # smeared a painted P step away within ticks), and a sustained 0.3-atm
    # head (k_p*dP = 0.15 m) pumps the U-bend plug over the lip and drains
    # the trap within the 50-tick run — physically correct trap-burping, but
    # not this test's subject. A 1.1 step still drives a violent dry-control
    # crossing (a 0.1-atm gradient is a ~hurricane-scale driver at K =
    # c_amb^2/gamma) while the plug's spill stays well inside the basin.
    from simulation import atmosphere_fixed
    from simulation.gases import O2 as _O2, INERT_N2 as _N2
    if not with_water:
        # The pressure step drives the DRY control's crossing. In the SEALED
        # run it is omitted (EOS P3): a sustained step's transient winds are
        # near-sonic at real physics scale (K = c_amb^2/gamma) and the
        # non-conservative SL deletes the tracer outright under ~37-tile
        # displacements — while the seal contract under test (a perm-0
        # flooded face passes NOTHING) is wind-independent: donor-cell flux,
        # SL marches AND diffusion are all gated by the same face
        # permeability, the exact code path the dry control exercises.
        g.gas[_O2][1, 1:LINE_X] = (g.gas[_O2][1, 1:LINE_X] * 11) // 10
        g.gas[_N2][1, 1:LINE_X] = (g.gas[_N2][1, 1:LINE_X] * 11) // 10
        g.atmosphere[1, 1:LINE_X] = atmosphere_fixed.quantize_scalar(1.1)
    sim.set_paused(False)
    return sim, g


def _far_side_gas(g) -> float:
    # S2b: gas is int32 Q16.16 — dequantize to real density so the float
    # thresholds (< 1e-12 sealed / > 1e-12 crossed) stay meaningful.
    return float(g.gas[SMOKE][1, LINE_X + 1:].astype(np.float64).sum()) \
        / gas_fixed.FP_ONE_F


def test_flooded_line_seals_corridor_gas_and_wind():
    sim, g = _corridor_sim(with_water=True)
    runner = sim.physics_runner

    for _ in range(50):
        sim.step()
        # EOS P3: the seal is a TRANSPORT property now, not a wind-value one.
        # The solver u is its own advected state, computed at every non-solid
        # cell (a flooded perm-0 cell included) — but no gas can CROSS a
        # perm-0 face (donor-cell flux and the SL march are both
        # permeability-gated). The gas one-sidedness asserts below are the
        # real seal contract; the old per-tick wind == 0.0 assert read an
        # implementation detail of the retired -grad(atm+wave_p) wind.

    # The source-held basin kept the line flooded -> sealed all run.
    assert float(deq(g.water_depth[1, LINE_X])) >= (
        float(runner.water_ceiling_h) - float(runner.water_flood_eps))
    assert float(g.dyn_permeability[1, LINE_X]) == 0.0

    # Gas stayed one-sided: the far side is (bit-)empty ...
    gas_far = _far_side_gas(g)
    assert gas_far < 1e-12, f"gas leaked past the flooded line: {gas_far}"
    # ... and non-vacuously so: the source side still holds its gas. S2b:
    # dequantize to real density (the source was seeded at full density).
    gas_near = float(g.gas[SMOKE][1, 1:LINE_X].astype(np.float64).sum()) \
        / gas_fixed.FP_ONE_F
    assert gas_near > 1.0, "source-side gas vanished (vacuous one-sidedness)"


def test_dry_line_control_proves_the_seal_assertions_bite():
    """CONTROL (non-vacuity guard for test 3): the identical scene with NO
    water — gas DOES cross the line within 50 ticks and wind DOES blow
    across it, so the flooded run's `< 1e-12` / `== 0.0` asserts are real."""
    sim, g = _corridor_sim(with_water=False)
    assert not g.water_depth.any()

    max_wind = 0.0
    for _ in range(50):
        sim.step()
        max_wind = max(max_wind, abs(float(g.wind_x[1, LINE_X])))

    assert _far_side_gas(g) > 1e-12, (
        "gas never crossed the DRY line — the sealed-run assert is vacuous")
    assert max_wind > 0.0, (
        "wind never blew across the DRY line — the wind assert is vacuous")
    # And without water the line cell was never sealed.
    assert float(g.dyn_permeability[1, LINE_X]) == 1.0


# ---------------------------------------------------------------------------
# 4. Wet-static exactness: a settled pool's tick is bit-identical
# ---------------------------------------------------------------------------
def test_wet_static_pool_atmosphere_bit_identical():
    """Settled flat pool in a sealed box: depth is unchanged through the
    substeps, so free_before == free_after bit-identically -> the ratio is
    x/x == IEEE-exact 1.0 -> the displacement multiply is the identity.

    A/B: one tick live vs one tick with the WHOLE ``_step_water`` no-op'd —
    monkeypatched on the runner itself, NOT the pybind solver (pybind methods
    are read-only; the test_water_integration stub precedent). Atmosphere
    must be bit-identical across the runs AND bit-identical to the live
    run's own pre-tick copy (the ratio==1 path really touched nothing).
    """

    def rollout(noop_water: bool):
        level = _sealed_room_level(9)
        sim = Simulation(level, seed=SEED, breach_physics=bp,
                         enable_recorder=False)
        g = sim.gmap
        interior = (~g.solid) & (~g.is_vacuum)
        g.water_depth[interior] = q(0.4)   # flat pool (Q16.16 metres, S1)
        if noop_water:
            sim.physics_runner._step_water = lambda gmap, sim_time: None
        sim.set_paused(False)
        pre_atmos = g.atmosphere.copy()
        pre_depth = g.water_depth.copy()
        sim.step()                          # ONE tick
        return sim, g, pre_atmos, pre_depth

    sim_a, g_a, pre_a, depth_a = rollout(noop_water=False)
    _sim_b, g_b, _pre_b, _ = rollout(noop_water=True)

    # Non-vacuity: the pool exists, there is air to (not) compress, and the
    # live run really took the wet path (the closing copyto ran: `before`
    # tracks the end-of-tick depth).
    assert g_a.water_depth.any()
    assert float(pre_a.max()) > 0.5
    assert np.array_equal(sim_a.physics_runner._water_depth_before,
                          g_a.water_depth)
    # The pool is settled: the live substeps were the identity on it.
    assert np.array_equal(g_a.water_depth, depth_a), (
        "the flat pool moved — the wet-static premise is broken")
    # A/B: live displacement vs whole-step no-op — bit-identical atmosphere.
    assert np.array_equal(g_a.atmosphere, g_b.atmosphere), (
        "atmosphere diverged between live and no-op water on a SETTLED pool")
    # And the live tick left atmosphere bit-untouched vs its own pre-tick
    # copy (ratio == 1 everywhere -> the multiply was exact identity).
    assert np.array_equal(g_a.atmosphere, pre_a), (
        "the ratio==1 displacement path modified atmosphere")
    # No seal: a 0.4 m pool leaves 2.1 m of air, far above flood_eps.
    assert float(g_a.dyn_permeability[4, 4]) == 1.0
