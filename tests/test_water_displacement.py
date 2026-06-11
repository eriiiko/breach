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
     REMOVE the water -> pressure returns within float tolerance.
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
from simulation.gases import BLACK_SMOKE  # noqa: E402

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
    level = _single_cell_level()
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    cy, cx = 2, 2
    assert not g.solid[cy, cx]
    assert (g.solid[cy - 1, cx] and g.solid[cy + 1, cx]
            and g.solid[cy, cx - 1] and g.solid[cy, cx + 1]), (
        "the column must be wall-sealed on all four sides")
    sim.set_paused(False)

    # One DRY tick first: the W2a first-call seed copies the CURRENT depth
    # into the `before` snapshot. Seeding it now (with zeros) means the dump
    # below is counted as a CHANGE — a dump landing on the very first physics
    # tick would be absorbed by the seed (read as pre-existing water, no
    # compression) by design.
    sim.step()
    assert sim.physics_runner._water_depth_before is not None
    assert not sim.physics_runner._water_depth_before.any()

    p0 = float(g.atmosphere[cy, cx])
    assert p0 > 0.5, "no air in the sealed cell (vacuous compression test)"

    ceil_h = float(sim.physics_runner.water_ceiling_h)
    cap = float(sim.physics_runner.water_ratio_cap)
    expected = ceil_h / (ceil_h - 0.5)
    assert abs(expected - 1.25) < 1e-12   # the plan's 2.5/(2.5-0.5)
    assert expected < cap, "the cap would bite — wrong scene for this test"

    # Event-shaped dump through the FieldEdit queue: flushed inside the next
    # step BEFORE physics, so the W3 displacement counts it this same tick.
    sim.edit(FieldEdit(field="water_depth", region=Region.TILE,
                       coords=(cy, cx), amount=0.5))
    sim.step()

    d = float(g.water_depth[cy, cx])
    assert abs(d - 0.5) < 1e-6, f"sealed column lost water: depth={d}"
    p1 = float(g.atmosphere[cy, cx])
    assert abs(p1 / p0 - expected) < 1e-6, (
        f"compression ratio {p1 / p0} != {expected}")

    # REMOVE the water: the air column re-expands -> decompression inrush
    # ratio (2.0/2.5 = 0.8) -> pressure returns within float tolerance.
    sim.edit(FieldEdit(field="water_depth", region=Region.TILE,
                       coords=(cy, cx), amount=0.5, mode=EditMode.REMOVE))
    sim.step()

    assert float(g.water_depth[cy, cx]) == 0.0
    p2 = float(g.atmosphere[cy, cx])
    assert abs(p2 - p0) < 1e-6, f"pressure did not return: {p0} -> {p2}"


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

    assert abs(float(g.water_depth[cy, cx]) - ceil_h) < 1e-6
    p1 = float(g.atmosphere[cy, cx])
    assert abs(p1 / p0 - cap) < 1e-6, (
        f"slam ratio {p1 / p0} != ratio_cap {cap}")
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
    g.floor_height[1, LINE_X] = -2.0 * ceil_h
    if with_water:
        g.water_depth[1, LINE_X] = 2.0 * ceil_h   # stays >= ceiling_h - flood_eps
    g.gas[BLACK_SMOKE][1, 1:LINE_X] = 10.0
    g.atmosphere[1, 1:LINE_X] = 1.3
    sim.set_paused(False)
    return sim, g


def _far_side_gas(g) -> float:
    return float(g.gas[BLACK_SMOKE][1, LINE_X + 1:].sum(dtype=np.float64))


def test_flooded_line_seals_corridor_gas_and_wind():
    sim, g = _corridor_sim(with_water=True)
    runner = sim.physics_runner

    for _ in range(50):
        sim.step()
        # Wind across the flooded faces is EXACTLY zero every tick: the wind
        # gradient pass gathers both of the cell's faces as min(perm) == 0,
        # so the gradient collapses to the mirrored centre value.
        assert float(g.wind_x[1, LINE_X]) == 0.0
        assert float(g.wind_y[1, LINE_X]) == 0.0

    # The basin held the line at full depth -> flooded -> sealed all run.
    assert float(g.water_depth[1, LINE_X]) >= (
        float(runner.water_ceiling_h) - float(runner.water_flood_eps))
    assert float(g.dyn_permeability[1, LINE_X]) == 0.0

    # Gas stayed one-sided: the far side is (bit-)empty ...
    gas_far = _far_side_gas(g)
    assert gas_far < 1e-12, f"gas leaked past the flooded line: {gas_far}"
    # ... and non-vacuously so: the source side still holds its gas.
    gas_near = float(g.gas[BLACK_SMOKE][1, 1:LINE_X].sum(dtype=np.float64))
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
        g.water_depth[interior] = 0.4      # flat pool on a flat floor: settled
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
