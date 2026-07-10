"""WaterSolver (C++ pipe model) — Step W1 of docs/water_implementation_plan.md.

Solver-only tests: the solver is NOT wired into the game yet (that is W2).
Canon design: docs/architecture/engine/07_fluid_and_water.md §2.

The nine plan tests, in order:
  1. mass conservation (sealed box, lumpy init, 1000 steps, 1e-4 rel)
  2. levelling + wall flatness (dam-break; sloped floor surface; the
     Neumann-mirror property at wall-adjacent columns)
  3. containment (interior wall: empty chamber stays exactly 0; never on solid)
  4. tilt slide (mass migrates low-side, conserved) + settled-pool bit-stability
  5. stability hammer (64x64 checkerboard at 3x the CFL dt: bounded, settles)
  6. null fields (floor_height/atmosphere/wave_p None == explicit zeros)
  7. head gating (k_p=0 never reads pressure: bit-identical; uniform pressure
     at k_p=0.5 ~ no-op to atol 1e-5)
  8. determinism (two identical runs bit-identical)
  9. outflow-limiter mass exactness (worst-case 4-face donor, depth_eps=0)

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_water_solver.py -q
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
from water_q16 import q, deq  # noqa: E402  (S1: Q16.16 quantize/dequantize)

DT = 0.016  # s — game-tick-scale step, under the k_p=0 CFL bound (33.6 ms)

# S1: the water core is int32 Q16.16. These tests author scenarios in metres and
# quantize at the solver boundary; depth/flow arrays passed to step() are int32,
# floor_height is int32 too (atmosphere/wave_p stay float — the FLOAT BRIDGE).
# Conservation/levelling/flatness tolerances stay in metres (the int field
# dequantizes to the same physical quantity, +/- the ~1.5e-5 m granularity).


# ---------------------------------------------------------------------------
# helpers (deterministic — no RNG anywhere in this file)
# ---------------------------------------------------------------------------
def _solver(**overrides) -> "bp.WaterSolver":
    s = bp.WaterSolver()
    for key, val in overrides.items():
        assert hasattr(s, key), f"unknown WaterSolver param {key!r}"
        setattr(s, key, val)
    return s


def _zeros(h: int, w: int) -> np.ndarray:
    """A zeroed Q16.16 int32 field (depth / velocity / floor)."""
    return np.zeros((h, w), dtype=np.int32)


def _open(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def _lumpy(h: int, w: int, base: float = 0.3, amp: float = 0.2) -> np.ndarray:
    """Deterministic lumpy init, Q16.16: base + amp*sin(row)*cos(col)."""
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    return q(base + amp * np.sin(rows) * np.cos(cols))


def _total(depth: np.ndarray) -> float:
    """Total water in METRES (dequantized) — exact integer sum / 65536."""
    return float(deq(depth).sum(dtype=np.float64))


def _run(s, steps, depth, vx, vy, solid, dt=DT,
         floor=None, atm=None, tilt=(0.0, 0.0)):
    # EOS refactor P3 (design §6 "water head"): the FLOAT BRIDGE (separate
    # atmosphere+wave_p float args) is retired — `atm` is now the single
    # INTEGER derived pressure P (Q16.16 int32), read via mul_q16(k_p, P)
    # directly inside the solver. No `wp` arg.
    for _ in range(steps):
        s.step(depth, vx, vy, floor, atm, solid, dt, tilt[0], tilt[1])


# ---------------------------------------------------------------------------
# 1. Mass conservation — sealed box, lumpy init, 1000 steps
# ---------------------------------------------------------------------------
def test_mass_conservation_sealed_box():
    h = w = 32
    solid = _open(h, w)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True  # sealed ring
    depth = _lumpy(h, w)
    depth[solid] = 0  # no water inside walls
    vx, vy = _zeros(h, w), _zeros(h, w)
    s = _solver()

    total0 = _total(depth)
    std0 = float(deq(depth).std())
    _run(s, 1000, depth, vx, vy, solid)

    assert np.isfinite(deq(depth)).all()
    total1 = _total(depth)
    # S1: the integer donor-cell transport conserves mass to the LSB in a sealed
    # box (no snap fires here — all cells stay above depth_eps), so the drift is
    # 0; the 1e-4 rel bound from the float plan is trivially met.
    assert abs(total1 - total0) / total0 < 1e-4, (
        f"mass drifted: {total0} -> {total1}")
    # non-vacuity: the field actually evolved (it levels, so std drops)
    std1 = float(deq(depth).std())
    assert abs(std1 - std0) > 1e-3, "field did not evolve (vacuous run)"


# ---------------------------------------------------------------------------
# 2. Levelling + wall flatness (the Neumann-mirror property)
# ---------------------------------------------------------------------------
def test_levelling_and_wall_flatness():
    h = w = 16
    s = _solver()

    # (a) dam-break in an open box (grid border = wall): settles dead flat,
    #     up to and including the border-adjacent columns. The prototypes'
    #     tall-terrain wall scheme would leave a depressed rim here.
    solid = _open(h, w)
    depth = _zeros(h, w)
    depth[:, : w // 2] = q(0.4)
    vx, vy = _zeros(h, w), _zeros(h, w)
    _run(s, 4000, depth, vx, vy, solid)
    depth_m = deq(depth)
    wet = depth_m > s.depth_eps
    assert wet.all(), "dam-break should wet the whole box"
    assert float(depth_m.max() - depth_m.min()) < 1e-3, (
        "settled pool not flat (incl. wall-adjacent columns)")

    # (b) interior solid wall: the pool is flat up to and including the
    #     wall-adjacent column x=7.
    solid_b = _open(h, w)
    solid_b[:, 8] = True
    depth_b = _zeros(h, w)
    depth_b[:, :8] = _lumpy(h, 8)
    vx_b, vy_b = _zeros(h, w), _zeros(h, w)
    _run(s, 4000, depth_b, vx_b, vy_b, solid_b)
    left = deq(depth_b[:, :8])
    assert float(left.max() - left.min()) < 1e-3, (
        "chamber pool not flat up to the wall-adjacent column")
    assert float(np.abs(deq(depth_b[:, 7]) - left.mean()).max()) < 1e-3, (
        "wall-adjacent column deviates (mirror-BC property violated)")
    assert float(deq(depth_b[:, 8:]).max()) == 0.0

    # (c) sloped floor_height: the settled SURFACE (floor + depth) is flat
    #     over wet cells, while depth itself varies with the slope. floor is
    #     int32 Q16.16 metres (S1).
    floor = q(np.tile(0.01 * np.arange(w, dtype=np.float64), (h, 1)))
    depth_c = q(np.full((h, w), 0.2, dtype=np.float64))
    vx_c, vy_c = _zeros(h, w), _zeros(h, w)
    _run(s, 4000, depth_c, vx_c, vy_c, _open(h, w), floor=floor)
    depth_c_m = deq(depth_c)
    wet_c = depth_c_m > s.depth_eps
    assert wet_c.any()
    surface = deq(floor) + depth_c_m
    assert float(surface[wet_c].std()) < 1e-3, "settled surface not flat"
    assert float(depth_c_m.std()) > 0.01, "depth did not adapt to the slope"
    assert float(depth_c_m[:, 0].mean()) > float(depth_c_m[:, -1].mean())


# ---------------------------------------------------------------------------
# 3. Containment — interior wall between a full and an empty chamber
# ---------------------------------------------------------------------------
def test_containment_interior_wall():
    h = w = 16
    solid = _open(h, w)
    solid[:, 8] = True  # 1-tile wall, full height
    depth = _zeros(h, w)
    depth[:, :8] = _lumpy(h, 8, base=0.5, amp=0.3)  # sloshes the whole run
    init = depth.copy()
    vx, vy = _zeros(h, w), _zeros(h, w)
    s = _solver()

    for _ in range(500):
        s.step(depth, vx, vy, None, None, solid, DT, 0.0, 0.0)
        assert float(np.abs(depth[:, 9:]).max()) == 0.0, (
            "water leaked into the sealed empty chamber")
        assert float(np.abs(depth[solid]).max()) == 0.0, "depth on solid"

    assert not np.array_equal(depth, init), "left chamber never moved (vacuous)"


# ---------------------------------------------------------------------------
# 4. Tilt slide + settled-pool bit-stability at zero tilt
# ---------------------------------------------------------------------------
def test_tilt_slide_and_settled_bit_stability():
    h = w = 16
    s = _solver()

    # (a) constant tilt_x > 0 raises the surface at high x -> water migrates
    #     to the LOW side (low x); total conserved.
    solid = _open(h, w)
    depth = q(np.full((h, w), 0.3, dtype=np.float64))
    vx, vy = _zeros(h, w), _zeros(h, w)
    total0 = _total(depth)
    _run(s, 300, depth, vx, vy, solid, tilt=(0.1, 0.0))
    low = float(deq(depth[:, : w // 2]).sum(dtype=np.float64))
    high = float(deq(depth[:, w // 2:]).sum(dtype=np.float64))
    assert low > 1.5 * high, f"mass did not migrate low-side ({low} vs {high})"
    assert abs(_total(depth) - total0) / total0 < 1e-4

    # (b) zero tilt on a settled flat pool: two more steps change nothing
    #     (beyond depth_eps snaps, which have all already happened).
    depth_b = _lumpy(h, w)
    vx_b, vy_b = _zeros(h, w), _zeros(h, w)
    _run(s, 3000, depth_b, vx_b, vy_b, solid)  # settle
    snap = depth_b.copy()
    _run(s, 2, depth_b, vx_b, vy_b, solid)
    assert np.array_equal(depth_b, snap), "settled pool is not bit-stable"


# ---------------------------------------------------------------------------
# 5. Stability hammer — checkerboard columns at ~3x the CFL-bound dt
# ---------------------------------------------------------------------------
def test_stability_hammer_checkerboard():
    h = w = 64
    solid = _open(h, w)
    rows, cols = np.indices((h, w))
    depth = q(np.where((rows + cols) % 2 == 0, 2.0, 0.0))
    vx, vy = _zeros(h, w), _zeros(h, w)
    s = _solver()

    dt = 0.1  # ~3x max_dt() == 33.6 ms; clamps+limiter must keep it sane
    assert dt > 2.5 * s.max_dt()
    total0 = _total(depth)
    std0 = float(deq(depth).std())

    for _ in range(500):
        s.step(depth, vx, vy, None, None, solid, dt, 0.0, 0.0)
        depth_m = deq(depth)
        assert np.isfinite(depth_m).all(), "depth went non-finite"
        assert float(depth_m.min()) >= 0.0, "negative depth"
        assert float(depth_m.max()) <= 2.2, (
            f"depth blew past the bounded-pile-up slack: {depth_m.max()}")

    assert np.isfinite(deq(vx)).all() and np.isfinite(deq(vy)).all()
    assert abs(_total(depth) - total0) / total0 < 1e-3
    assert float(deq(depth).std()) < std0, "hammer survived but did not settle"


# ---------------------------------------------------------------------------
# 6. Null fields — None is the same run as explicit zero arrays
# ---------------------------------------------------------------------------
def _run_copy(k_p, floor, atm, steps=100, tilt=(0.02, -0.01)):
    h = w = 16
    depth = _lumpy(h, w)
    vx, vy = _zeros(h, w), _zeros(h, w)
    s = _solver(k_p=k_p)
    _run(s, steps, depth, vx, vy, _open(h, w),
         floor=floor, atm=atm, tilt=tilt)
    return depth, vx, vy


def test_null_fields_equal_explicit_zeros():
    h = w = 16
    z_int = _zeros(h, w)                              # Q16.16 zeros (both floor + atm)

    # floor_height: None == flat zero (bit-identical)
    a = _run_copy(0.0, None, None)
    b = _run_copy(0.0, z_int, None)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa, fb), "floor_height=None != explicit zeros"

    # EOS refactor P3: atmosphere is now the single INTEGER P field (the
    # separate wave_p float bridge is retired) — None == explicit Q16.16
    # zeros, bit-identical (the C++ gate substitutes 0 for a null pointer).
    c = _run_copy(0.5, None, None)
    d = _run_copy(0.5, None, z_int)
    for fc, fd in zip(c, d):
        assert np.array_equal(fc, fd), "atmosphere=None != zeros"

    # non-vacuity: the runs actually evolved the field
    assert not np.array_equal(a[0], _lumpy(h, w)), "vacuous comparison"


# ---------------------------------------------------------------------------
# 7. Head gating — k_p == 0 never reads the pressure field
# ---------------------------------------------------------------------------
def test_head_gating():
    h = w = 16
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    wild = q(1e4 * np.sin(3.0 * rows) * np.cos(2.0 * cols))
    assert np.all(np.isfinite(wild))

    # (a) k_p=0 + wild-but-finite atmosphere (P) is BIT-IDENTICAL to
    #     atmosphere=None (the C++ gate makes this exact).
    a = _run_copy(0.0, None, None)
    b = _run_copy(0.0, None, wild)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa, fb), "k_p=0 still read atmosphere (gate broken)"

    # (b) k_p=0.5 + spatially-UNIFORM pressure ~ k_p=0 to atol=1e-5 over 100
    #     steps (NOT bit-exact: integer (a+c)-(b+c) rounding vs a-b).
    # EOS refactor P3: the old atm(0.8)+wp(0.2) SPLIT is now one combined
    # P=1.0 field (the design's merged-field precedent — one pressure, not
    # a bulk/acoustic pair).
    u_p = np.full((h, w), q(1.0), dtype=np.int32)
    e = _run_copy(0.5, None, u_p, steps=100)
    f = _run_copy(0.0, None, None, steps=100)
    assert np.allclose(e[0], f[0], atol=1e-5), (
        "uniform pressure changed the flow (constant head must vanish "
        "under the gradient)")
    # non-vacuity
    assert not np.array_equal(e[0], _lumpy(h, w))


# ---------------------------------------------------------------------------
# 8. Determinism — two identical runs are bit-identical
# ---------------------------------------------------------------------------
def test_determinism_bit_identical():
    h = w = 24

    def run():
        depth = _lumpy(h, w)
        vx, vy = _zeros(h, w), _zeros(h, w)
        solid = _open(h, w)
        solid[10:14, 10:12] = True       # interior obstacle
        depth[solid] = 0
        floor = q(np.tile(0.005 * np.arange(w, dtype=np.float64), (h, 1)))
        # EOS refactor P3: one combined integer P field (design §6) replaces
        # the old atm(non-uniform)+wp(sine ripple) float pair — same shape
        # of non-uniform-plus-ripple scenario, now a single Q16.16 source.
        atm_base = np.full((h, w), 1.0, dtype=np.float64)
        atm_base[:, : w // 2] = 0.6           # non-uniform head
        ripple = 0.1 * np.sin(np.arange(h * w, dtype=np.float64)).reshape(h, w)
        atm = q(atm_base + ripple)
        s = _solver(k_p=0.5)
        _run(s, 200, depth, vx, vy, solid,
             floor=floor, atm=atm, tilt=(0.05, 0.03))
        return depth, vx, vy

    a = run()
    b = run()
    assert np.array_equal(a[0], b[0]), "depth not bit-identical across runs"
    assert np.array_equal(a[1], b[1]), "flow_vx not bit-identical across runs"
    assert np.array_equal(a[2], b[2]), "flow_vy not bit-identical across runs"
    # non-vacuity: the scene actually flowed
    assert not np.array_equal(a[0], _lumpy(h, w)), "nothing happened (vacuous)"
    assert float(np.abs(a[1]).max()) > 0.0


# ---------------------------------------------------------------------------
# 9. Outflow limiter — worst-case 4-face donor conserves mass exactly
# ---------------------------------------------------------------------------
def test_outflow_limiter_conservation():
    s = _solver(depth_eps=0.0)  # the snap is a designed eps-scale sink; this
    #                             test targets the limiter, so disable it
    # pin the CFL bound formula while we are here (P_ref=1.0, head_ref=0.2)
    assert math.isclose(s.max_dt(), 0.5 * s.dx / math.sqrt(s.g * s.h_ref),
                        rel_tol=1e-6)
    s_head = _solver(k_p=0.5)
    assert math.isclose(
        s_head.max_dt(),
        0.5 * s_head.dx / math.sqrt(s_head.g * s_head.h_ref
                                    * (1.0 + 0.5 * 1.0 / 0.2)),
        rel_tol=1e-6)

    h = w = 32
    solid = _open(h, w)
    depth = _zeros(h, w)
    depth[16, 16] = q(2.5)  # single column on a dry plane: 4-face donor
    vx, vy = _zeros(h, w), _zeros(h, w)
    dt = s.max_dt()
    total0 = _total(depth)

    for _ in range(200):
        s.step(depth, vx, vy, None, None, solid, dt, 0.0, 0.0)
        assert int(depth.min()) >= 0, "negative depth"

    assert np.isfinite(deq(depth)).all()
    # S1: with depth_eps=0 (snap off) the integer donor-cell transport conserves
    # to the LSB; the 1e-5 rel bound is trivially met (drift is 0).
    assert abs(_total(depth) - total0) / total0 < 1e-5, (
        "the outflow limiter leaked mass (clamp created/destroyed water)")
    assert int((depth > 0).sum()) > 100, "column never spread (vacuous)"
