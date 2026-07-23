"""Water tick integration (Step W2a of docs/water_implementation_plan.md).

The W1 C++ ``WaterSolver`` is wired into the live tick here: GameMap grows the
water fields (``water_depth`` / ``flow_vx`` / ``flow_vy`` / ``floor_height`` /
tilt / ``tile_size_m`` / ``water_sources``), ``[physics.water]`` binds onto the
solver in PhysicsRunner, and ``_step_water`` runs once per tick right after the
fire heat cast. THE contract of this step is dormancy: with zero water on the
map a full physics tick is behaviour-identical to before the step existed.

The five plan tests, in order:
  1. dormancy trio — (i) dry ticks leave the water fields exactly zero;
     (ii) the early-out is REALLY taken (a raising solver stub never fires);
     (iii) A/B same-seed rollouts (live ``_step_water`` vs no-op) give a
     bit-identical signature (atmosphere, wave_p, gas, fire, temperature,
     dyn_permeability).
  2. source spread — 9x9 sealed room, source (4,4,0.5): holds, spreads to
     >= 90% of open tiles by tick 200, total rises then plateaus.
  3. FieldEdit — queued ADD/REMOVE on ``water_depth`` flush correctly, clamp
     at 0, skip solid.
  4. dx binding — after one step ``water.dx == gmap.tile_size_m``.
  5. runner conservation — sealed room, painted water, no sources: total
     (float64) conserved across 100 ticks.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_water_integration.py -q
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
from level_loader import LevelData, load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.field_edit import (  # noqa: E402
    EditMode, EditQueue, FieldEdit, Region,
)
from simulation.gamemap import GameMap  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from water_q16 import q as wq, deq  # noqa: E402  (S1; `q` is shadowed by EditQueue)

# S1: water_depth/flow_vx/flow_vy are int32 Q16.16. Paint in metres via wq();
# read totals/depths via deq(); FieldEdit amounts stay metres (the "water" policy
# quantizes). Conservation: the integer transport conserves Σdepth to the LSB.
Q_EPS = 1.0 / 65536.0

SEED = 42


# ---------------------------------------------------------------------------
# helpers (deterministic — the only RNG is the seeded sim.rng inside flush)
# ---------------------------------------------------------------------------
def _sealed_room_level(n: int = 9, tile_size_m: float = 0.333) -> LevelData:
    """An n x n hull-ringed room, interior air, NO vacuum anywhere — the
    sealed box for the spread/conservation tests (the test_smoke_sink_pull
    synthetic-LevelData pattern). CSV codes: 1 = hull, 4 = interior air.
    """
    tm = np.ones((n, n), dtype=np.int32)     # hull ring (map border = wall)
    tm[1:n - 1, 1:n - 1] = 4                 # carve interior air
    return LevelData(
        name="water_room_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=tile_size_m,
        diffuse_path=Path("."),
    )


def _lumpy(h: int, w: int, base: float = 0.3, amp: float = 0.2) -> np.ndarray:
    """Deterministic lumpy depth init, Q16.16 int32 (S1)."""
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    return wq(base + amp * np.sin(rows) * np.cos(cols))


def _total(depth: np.ndarray) -> float:
    """Total water in METRES (dequantized) — exact integer sum / 65536."""
    return float(deq(depth).sum(dtype=np.float64))


class _RaisingWaterStub:
    """Stands in for the C++ WaterSolver: ANY solver call is a test failure.

    pybind11 instances reject method monkeypatching (``step`` is a read-only
    attribute), so the whole ``runner.water`` object is swapped — the runner's
    call path (``self.water.step`` / ``.max_dt``) is identical either way.
    The dx lazy-bind (``self.water.dx = ...``) is a plain attribute set and
    must succeed — it happens BEFORE the early-out, by design.
    """

    def max_dt(self):
        raise AssertionError(
            "water.max_dt() called on a dry map (early-out not taken)")

    def step(self, *args, **kwargs):
        raise AssertionError(
            "water.step() called on a dry map (early-out not taken)")


# ---------------------------------------------------------------------------
# 1. Dormancy trio (house pattern: test_temperature_ignition dormancy +
#    test_multigas_structure A/B rollout signature)
# ---------------------------------------------------------------------------
def test_dormant_dry_ticks_leave_water_fields_zero():
    """(i) Full Simulation on the test vessel, 5 dry ticks: the water fields
    stay EXACTLY zero (nothing writes them without water/sources)."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    m = Unit("M1", x=14, y=50, team=0)
    sim.add_unit(m)
    sim.set_paused(False)

    for _ in range(5):
        sim.step()

    g = sim.gmap
    assert not g.water_depth.any(), "water_depth became non-zero on a dry map"
    assert not g.flow_vx.any(), "flow_vx became non-zero on a dry map"
    assert not g.flow_vy.any(), "flow_vy became non-zero on a dry map"


def test_dormant_early_out_never_calls_the_solver():
    """(ii) With the solver swapped for a raiser, 5 dry ticks raise nothing —
    the dormant early-out is REALLY taken (not just harmless)."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    sim.physics_runner.water = _RaisingWaterStub()
    sim.set_paused(False)

    for _ in range(5):
        sim.step()   # must not raise

    # The first-call seed + dx bind DID run (they precede the early-out).
    assert sim.physics_runner._water_depth_before is not None
    assert float(sim.physics_runner.water.dx) == float(sim.gmap.tile_size_m)


def test_dormant_ab_rollout_bit_identical():
    """(iii) Two same-seed 60-tick rollouts — one with ``_step_water``
    monkeypatched to a no-op — produce a bit-identical signature tuple.
    This is the dormancy guarantee in full: on a dry map the live water step
    IS a no-op for every other system."""
    fields = ("atmosphere", "wave_p", "gas", "fire",
              "temperature", "dyn_permeability")

    def rollout(noop_water: bool):
        level = load_level("unhcr_vessel")
        sim = Simulation(level, seed=SEED, breach_physics=bp,
                         enable_recorder=False)
        if noop_water:
            sim.physics_runner._step_water = lambda gmap, sim_time: None
        sim.set_paused(False)
        for _ in range(60):
            sim.step()
        return tuple(getattr(sim.gmap, name).copy() for name in fields)

    live = rollout(noop_water=False)
    noop = rollout(noop_water=True)
    for name, fa, fb in zip(fields, live, noop):
        assert np.array_equal(fa, fb), (
            f"{name} diverged between live and no-op water on a DRY map "
            f"(the water step is not dormant)")


# ---------------------------------------------------------------------------
# 2. Source spread — a continuous hold fills the sealed room and plateaus
# ---------------------------------------------------------------------------
def test_source_spread_fills_sealed_room():
    level = _sealed_room_level(9)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    assert not g.solid[4, 4], "source tile must be interior air"
    g.water_sources.append((4, 4, 0.5))
    sim.set_paused(False)

    # Tick 1: the hold lands (depth = max(depth, 0.5)); the substeps shed a
    # little outward but most of the column is still standing.
    sim.step()
    assert float(deq(g.water_depth[4, 4])) >= 0.4, (
        f"source hold did not land: depth[4,4]={float(deq(g.water_depth[4, 4]))}")

    totals = [_total(g.water_depth)]
    for _ in range(199):                     # ticks 2..200
        sim.step()
        totals.append(_total(g.water_depth))

    # By tick 200 the pool has reached >= 90% of the open tiles.
    open_tiles = (~g.solid) & (~g.is_vacuum)
    eps = float(sim.physics_runner.water.depth_eps)
    wet_frac = float((deq(g.water_depth[open_tiles]) > eps).mean())
    assert wet_frac >= 0.9, f"pool covers only {wet_frac * 100:.0f}% of open tiles"

    # Total water rose (the source fed the room) ...
    assert totals[-1] > totals[0], "total water never rose from the source"

    # ... then plateaus: the pool levels at the hold height, the hold adds ~0.
    # Sampled at tick 320, not the plan's 200 — MEASURED (re-measured for W4,
    # head ON): the 20-tick window first satisfies < 1e-3 at tick 264 (at
    # k_p = 0 it was 215; the fill's displacement compression — the sealed
    # room climbs to 1.251 atm — leaves transient head gradients that slosh
    # the pool ~50 ticks longer); from 300 on it is locked at ~1e-6 noise,
    # total 24.5173. The plan's 200 was an estimate made before W2 was
    # runnable; the mechanism (hard plateau) is exactly as designed. Ticks
    # 201..320 cross the end-of-round auto-pause at 240, hence the unpause
    # guard.
    for _ in range(120):                     # ticks 201..320
        sim.step()
        if sim.paused:                       # end-of-round auto-pause (240)
            sim.set_paused(False)
        totals.append(_total(g.water_depth))
    last20 = totals[-20:]
    assert (max(last20) - min(last20)) < 1e-3, (
        f"total still moving over the last 20 ticks: "
        f"d={max(last20) - min(last20):.2e}")


# ---------------------------------------------------------------------------
# 3. FieldEdit — water_depth goes through the canonical write primitive
# ---------------------------------------------------------------------------
def test_field_edit_add_remove_clamp_skip_solid():
    level = _sealed_room_level(9)
    g = GameMap(level)
    q = EditQueue()
    rng = np.random.default_rng(SEED)

    # ADD then REMOVE on one open tile: same field/source/region -> the seq
    # tie-break keeps enqueue order, so the flush nets 0.3 - 0.1 = 0.2.
    q.enqueue(FieldEdit(field="water_depth", region=Region.TILE,
                        coords=(4, 4), amount=0.3))
    q.enqueue(FieldEdit(field="water_depth", region=Region.TILE,
                        coords=(4, 4), amount=0.1, mode=EditMode.REMOVE))
    # Over-REMOVE on a second tile: the policy clamp (0, inf) floors at 0.
    q.enqueue(FieldEdit(field="water_depth", region=Region.TILE,
                        coords=(2, 2), amount=0.05))
    q.enqueue(FieldEdit(field="water_depth", region=Region.TILE,
                        coords=(2, 2), amount=1.0, mode=EditMode.REMOVE))
    # ADD on a solid (hull) tile: the _skip_solid veto drops it entirely.
    assert bool(g.solid[0, 0])
    q.enqueue(FieldEdit(field="water_depth", region=Region.TILE,
                        coords=(0, 0), amount=0.5))

    q.flush(g, rng)

    # S1: amounts authored in metres, stored Q16.16 -> dequantize to check. The
    # net (0.3-0.1=0.2) carries a couple Q16.16 LSB of round-trip slack.
    assert abs(float(deq(g.water_depth[4, 4])) - 0.2) < 3 * Q_EPS, (
        "ADD/REMOVE did not net")
    assert int(g.water_depth[2, 2]) == 0, "over-REMOVE did not clamp at 0"
    assert int(g.water_depth[0, 0]) == 0, "water written onto a solid tile"
    assert len(q) == 0, "flush did not clear the queue"


# ---------------------------------------------------------------------------
# 4. dx binding — the solver learns the LEVEL's tile size, never a default
# ---------------------------------------------------------------------------
def test_dx_binds_from_level_tile_size():
    # 0.5 is exactly float32-representable (the == below is exact) and is NOT
    # the solver's construction default (0.333) — non-vacuous on both counts.
    level = _sealed_room_level(9, tile_size_m=0.5)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    assert float(sim.physics_runner.water.dx) != 0.5, (
        "dx already 0.5 before any step — vacuous test")
    sim.set_paused(False)

    sim.step()

    assert float(sim.gmap.tile_size_m) == 0.5
    assert float(sim.physics_runner.water.dx) == float(sim.gmap.tile_size_m), (
        "water.dx did not lazy-bind from gmap.tile_size_m on the first step")


# ---------------------------------------------------------------------------
# 5. Runner conservation — a sealed pool keeps its mass through the full tick
# ---------------------------------------------------------------------------
def test_runner_conserves_painted_water():
    level = _sealed_room_level(12)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap

    # Paint deterministic lumpy water on the interior only (the solver zeroes
    # depth on solid, so painting walls would read as a designed mass sink).
    interior = (~g.solid) & (~g.is_vacuum)
    paint = _lumpy(*g.water_depth.shape)
    g.water_depth[interior] = paint[interior]
    total0 = _total(g.water_depth)
    std0 = float(deq(g.water_depth).std())
    assert total0 > 0.0
    sim.set_paused(False)

    for _ in range(100):
        sim.step()

    assert np.isfinite(deq(g.water_depth)).all()
    total1 = _total(g.water_depth)
    # S1: the integer pipe-model transport conserves Σdepth to the LSB in a
    # sealed room (no boil here; the snap doesn't fire on the wet interior), so
    # the drift is ~0 and the float 1e-4 rel bound is trivially met.
    assert abs(total1 - total0) / max(total0, 1e-12) < 1e-4, (
        f"runner leaked water: {total0} -> {total1}")
    # non-vacuity: the lumpy blob actually flowed (it levels, so std drops).
    assert abs(float(deq(g.water_depth).std()) - std0) > 1e-4, (
        "water never moved (vacuous conservation)")
