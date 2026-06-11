"""Pressure head ON: atmosphere -> water (Step W4 of docs/water_implementation_plan.md).

W4 ships exactly ONE value: config ``[physics.water] k_p = 0.5`` (was 0.0).
The head term itself ships since W1 (gated on ``k_p`` inside the C++
``WaterSolver``: ``surface += k_p * (atmosphere + wave_p)``) and its CFL
margin ships in ``max_dt()``; W2a binds ``k_p`` from config. With the head ON,
blasts (``wave_p``) shove water outward and pressure gradients (``atmosphere``)
drag it toward low pressure — the grenade-crater / hull-vent signatures.

The plan's three tests + the config-binding pin, in order:
  1. uniform pressure ~ no-op: a constant head adds a constant to the surface
     potential, which vanishes under the gradient — same pool, k_p=0.5 vs
     k_p=0.0, allclose to atol 1e-5 over 100 steps (NOT bit-exact: float
     (a+c)-(b+c) != a-b; the bit-exact k_p=0 gate is W1 test 7a).
  2. Gaussian ``wave_p`` bump centred on a pool: centre depth drops vs the
     no-bump control, an outward ring rises, total conserved (float64,
     1e-6 rel) — the grenade-crater signature.
  3. sustained low pressure at one end of a corridor pool: net water flux
     toward the low-pressure end vs the uniform-pressure control (centre of
     mass + end-column depths).
  4. shipped-config pin: the live binding (config -> PhysicsRunner.
     _bind_water_params -> solver) yields k_p == 0.5, and the derived substep
     count at 24 tps is 3 (max_dt() = 18.0 ms < 41.7 ms / 2) — pins the
     CONFIG-driven behaviour end-to-end, not just the W1 closed form.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_water_pressure_head.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402

SEED = 42
DT = 0.016  # s — under BOTH CFL bounds (k_p=0 -> 33.6 ms; k_p=0.5 -> 18.0 ms)


# ---------------------------------------------------------------------------
# helpers (deterministic — no RNG anywhere in this file)
# ---------------------------------------------------------------------------
def _solver(**overrides) -> "bp.WaterSolver":
    """A WaterSolver with params bound DIRECTLY on the instance (no config —
    tests 1-3 probe the solver; only test 4 goes through the runner binding)."""
    s = bp.WaterSolver()
    for key, val in overrides.items():
        assert hasattr(s, key), f"unknown WaterSolver param {key!r}"
        setattr(s, key, val)
    return s


def _zeros(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.float32)


def _open(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def _lumpy(h: int, w: int, base: float = 0.3, amp: float = 0.2) -> np.ndarray:
    """Deterministic lumpy depth init (the W1 test-1 pattern — no RNG)."""
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    return (base + amp * np.sin(rows) * np.cos(cols)).astype(np.float32)


def _total(depth: np.ndarray) -> float:
    return float(depth.sum(dtype=np.float64))


def _sealed_room_level(n: int = 9, tile_size_m: float = 0.333) -> LevelData:
    """An n x n hull-ringed room, interior air, no vacuum anywhere (the
    test_water_integration synthetic-LevelData pattern; 1 = hull, 4 = air)."""
    tm = np.ones((n, n), dtype=np.int32)
    tm[1:n - 1, 1:n - 1] = 4
    return LevelData(
        name="water_head_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=tile_size_m,
        diffuse_path=Path("."),
    )


# ---------------------------------------------------------------------------
# 1. Uniform pressure ~ no-op: a constant head vanishes under the gradient
# ---------------------------------------------------------------------------
def _run_lumpy_pool(k_p: float, atm, wp, steps: int = 100) -> np.ndarray:
    h = w = 16
    depth = _lumpy(h, w)
    vx, vy = _zeros(h, w), _zeros(h, w)
    solid = _open(h, w)
    s = _solver(k_p=k_p)
    for _ in range(steps):
        s.step(depth, vx, vy, None, atm, wp, solid, DT, 0.0, 0.0)
    return depth


def test_uniform_pressure_approx_noop():
    """Same lumpy pool, two runs over 100 steps: uniform atmosphere == 1.0
    with k_p = 0.5 vs k_p = 0.0 (which gates the SAME arrays out bit-exactly,
    W1 test 7a). allclose to atol 1e-5, NOT bit-exact — the head adds the
    constant k_p*1.0 to every cell's surface and float (a+c)-(b+c) != a-b,
    so the velocity kicks differ in the last ulps (plan W1 test 7)."""
    h = w = 16
    atm = np.full((h, w), 1.0, dtype=np.float32)
    wp = _zeros(h, w)  # uniform (zero) blast field

    on = _run_lumpy_pool(0.5, atm, wp)
    off = _run_lumpy_pool(0.0, atm, wp)

    assert np.allclose(on, off, atol=1e-5), (
        "uniform pressure changed the flow (a constant head must vanish "
        "under the gradient)")
    # non-vacuity: the lumpy pool really evolved over the 100 steps
    assert not np.array_equal(off, _lumpy(h, w)), "vacuous comparison"


# ---------------------------------------------------------------------------
# 2. Gaussian wave_p bump on a pool — the grenade-crater signature
# ---------------------------------------------------------------------------
def test_gaussian_bump_craters_centre_raises_ring_conserves_mass():
    h = w = 21
    cy = cx = 10
    rows, cols = np.indices((h, w))
    r2 = ((rows - cy) ** 2 + (cols - cx) ** 2).astype(np.float64)
    r = np.sqrt(r2)
    # Peak 2.0 pressure units, sigma = 2 tiles — a sustained grenade-scale
    # bump (held constant across the steps; the solver is called directly).
    bump = (2.0 * np.exp(-r2 / (2.0 * 2.0 ** 2))).astype(np.float32)
    atm = np.ones((h, w), dtype=np.float32)
    solid = _open(h, w)
    init = np.full((h, w), 0.3, dtype=np.float32)  # settled flat pool
    total0 = _total(init)

    def run(wp: np.ndarray) -> np.ndarray:
        depth = init.copy()
        vx, vy = _zeros(h, w), _zeros(h, w)
        # depth_eps = 0: this test pins conservation at 1e-6 rel and the
        # crater floor sweeps through eps-scale depths — the dry snap is a
        # designed eps-scale sink that would eat the budget (the W1 test-9
        # precedent: disable the snap when the test targets mass-exactness).
        s = _solver(k_p=0.5, depth_eps=0.0)
        for _ in range(100):
            s.step(depth, vx, vy, None, atm, wp, solid, DT, 0.0, 0.0)
        return depth

    crater = run(bump)
    ctrl = run(_zeros(h, w))

    # Total water conserved (float64; measured drift ~3e-8 rel).
    assert np.isfinite(crater).all()
    assert float(crater.min()) >= 0.0, "negative depth in the crater run"
    assert abs(_total(crater) - total0) / total0 < 1e-6, (
        f"the bump leaked mass: {total0} -> {_total(crater)}")

    # Centre depth drops vs the no-bump control (measured: 0.000 vs 0.300 —
    # the sustained bump drains the crater floor completely).
    assert float(crater[cy, cx]) < float(ctrl[cy, cx]) - 0.2, (
        f"no crater: centre {float(crater[cy, cx])} vs "
        f"control {float(ctrl[cy, cx])}")

    # An outward ring forms: depth at some radius RISES above the control
    # (measured max rise in the 3 <= r <= 8 band at 100 steps: ~0.044).
    band = (r >= 3.0) & (r <= 8.0)
    ring_rise = float((crater - ctrl)[band].max())
    assert ring_rise > 0.02, f"no displaced ring (max rise {ring_rise})"

    # Non-vacuity of the control: a flat pool under uniform pressure is
    # static (identical surface floats -> exactly zero gradient), so the
    # crater/ring deltas above are pure head-term signal.
    assert float(np.abs(ctrl - init).max()) < 1e-6, (
        "the no-bump control moved — the deltas are not pure bump signal")


# ---------------------------------------------------------------------------
# 3. Sustained low pressure at one end of a corridor pool drags water to it
# ---------------------------------------------------------------------------
def test_low_pressure_end_drags_corridor_water():
    """Corridor pool (5 x 31, uniform 0.5 m) under a sustained atmosphere
    gradient 1.0 -> 0.3 along x (wave_p None — the head term substitutes
    zeros): net flux toward the low-pressure end vs the uniform-pressure
    control. The hull-vent signature: venting drags the water after the air.
    Equilibrium tilts the surface by k_p * (1.0 - 0.3) = 0.35 m end-to-end;
    at 300 steps (4.8 s) the measured centre-of-mass shift is +1.79 tiles
    (equilibrium ~ +1.9) and the end columns sit at 0.66 / 0.34 vs 0.50."""
    h, w = 5, 31
    grad = np.tile(np.linspace(1.0, 0.3, w, dtype=np.float32), (h, 1))
    uni = np.ones((h, w), dtype=np.float32)
    solid = _open(h, w)
    init = np.full((h, w), 0.5, dtype=np.float32)
    total0 = _total(init)

    def run(atm: np.ndarray) -> np.ndarray:
        depth = init.copy()
        vx, vy = _zeros(h, w), _zeros(h, w)
        s = _solver(k_p=0.5)
        for _ in range(300):
            s.step(depth, vx, vy, None, atm, None, solid, DT, 0.0, 0.0)
        return depth

    drag = run(grad)
    ctrl = run(uni)

    xs = np.arange(w, dtype=np.float64)

    def com_x(d: np.ndarray) -> float:
        col = d.sum(axis=0, dtype=np.float64)
        return float((col * xs).sum() / col.sum())

    # Net flux toward the low-pressure (high-x) end: the centre of mass
    # shifts right (measured +1.79 tiles; assert with margin).
    assert com_x(drag) > com_x(ctrl) + 1.0, (
        f"water did not migrate toward low pressure: com {com_x(drag):.3f} "
        f"vs control {com_x(ctrl):.3f}")

    # End columns: the low-pressure end is deeper, the high-pressure end
    # drained (measured 0.660 / 0.344 vs the control's 0.500).
    assert float(drag[:, -3:].mean()) > float(ctrl[:, -3:].mean()) + 0.10
    assert float(drag[:, :3].mean()) < float(ctrl[:, :3].mean()) - 0.10

    # Control non-vacuity: uniform pressure leaves the flat pool centred,
    # so the shift above is pure gradient signal.
    assert abs(com_x(ctrl) - com_x(init)) < 1e-3

    # Mass conserved (float64; no cell nears the eps snap — measured
    # min depth 0.333 at 300 steps, so the snap never fires).
    assert float(drag.min()) > 0.0
    assert abs(_total(drag) - total0) / total0 < 1e-6, (
        f"the gradient leaked mass: {total0} -> {_total(drag)}")


# ---------------------------------------------------------------------------
# 4. Shipped-config pin: k_p = 0.5 binds through the runner; 3 substeps @ 24
# ---------------------------------------------------------------------------
def test_shipped_config_binds_head_on_and_derives_three_substeps():
    """The W1 closed-form test pins max_dt() for a HAND-SET k_p = 0.5; this
    pins the CONFIG-driven behaviour: the shipped [physics.water] k_p reaches
    the live solver through PhysicsRunner._bind_water_params (at construction,
    BEFORE any max_dt() use — max_dt is read per-tick in _step_water), and the
    derived substep count at 24 tps is 3 (max_dt() = 18.0 ms < 41.7 ms / 2)."""
    # Non-vacuity: the C++ construction default is OFF — the 0.5 below can
    # only have come from the config binding.
    assert float(bp.WaterSolver().k_p) == 0.0, (
        "solver default k_p is no longer 0 — the config pin is vacuous")

    level = _sealed_room_level(9)
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    runner = sim.physics_runner
    assert float(runner.water.k_p) == 0.5, (
        "shipped config did not bind k_p = 0.5 (pressure head not ON)")

    # One dry tick: the first-call dx lazy-bind runs (it precedes the
    # dormant early-out), so max_dt() below uses the LEVEL's tile size.
    # (dx is a float32 member: compare through the float32 round-trip —
    # 0.333 is not exactly representable; the binding itself is W2 test 4.)
    sim.set_paused(False)
    sim.step()
    assert float(runner.water.dx) == float(np.float32(sim.gmap.tile_size_m))

    # max_dt under the shipped binding == the W1 closed form at k_p = 0.5
    # (P_ref = 1.0, head_ref = 0.2): 18.0 ms at dx = 0.333, h_ref = 2.5.
    s = runner.water
    wdt = float(s.max_dt())
    assert math.isclose(
        wdt,
        0.5 * float(s.dx) / math.sqrt(
            float(s.g) * float(s.h_ref) * (1.0 + float(s.k_p) * 1.0 / 0.2)),
        rel_tol=1e-6)
    assert abs(wdt - 0.018) < 5e-4, f"max_dt {wdt * 1e3:.2f} ms != ~18.0 ms"

    # Derived substep count, computed EXACTLY as _step_water does. The claim
    # is tick-rate-specific, so pin the tick rate it is made at.
    assert CFG.clock.ticks_per_second == 24
    sim_time = 1.0 / CFG.clock.ticks_per_second
    assert wdt < sim_time / 2.0    # 18.0 ms < 20.8 ms -> n >= 3
    assert wdt >= sim_time / 3.0   # 18.0 ms >= 13.9 ms -> n <= 3
    n = max(1, int(math.ceil(sim_time / wdt)))
    assert n == 3, f"derived substep count {n} != 3 at 24 tps"
