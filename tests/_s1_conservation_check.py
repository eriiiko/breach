"""S1 P1 (bit-determinism) + P2 (Sum(water_depth) conservation) standalone check.

P2 is the load-bearing requirement: in a SEALED flood (no boil, no clamp firing,
no sources/sinks), the integer donor-cell transport must conserve Sum(water_depth) to
the LAST BIT across many ticks — the int64 flux gather applies the SAME rounded
per-face delta to both cells, so the >>16 narrow can never leak mass.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    sys.path.insert(0, str(p))

import breach_physics as bp

Q = 65536


def q(arr_m):
    return np.round(np.asarray(arr_m, np.float64) * Q).astype(np.int32)


def box(h, w):
    s = np.zeros((h, w), bool)
    s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = True
    return s


def make_scene(seed=0):
    h = w = 24
    solid = box(h, w)
    rng = np.random.default_rng(seed)
    # lumpy interior fill, all WELL above depth_eps so the snap never fires,
    # and bounded so v_max/limiter keep it sane (no clamp-at-0 mass injection).
    d = np.zeros((h, w), np.float64)
    d[1:23, 1:23] = 0.5 + 0.3 * np.sin(np.arange(1, 23)[:, None]) \
        * np.cos(np.arange(1, 23)[None, :])
    d[1:23, 1:23] = np.clip(d[1:23, 1:23], 0.25, 1.0)
    d[solid] = 0.0
    return solid, q(d)


def run(solid, depth0, n_ticks, dt=0.016, seed_tilt=(0.0, 0.0)):
    depth = depth0.copy()
    vx = np.zeros(depth.shape, np.int32)
    vy = np.zeros(depth.shape, np.int32)
    s = bp.WaterSolver()
    s.dx = 1.0 / 3.0
    totals = []
    for _ in range(n_ticks):
        s.step(depth, vx, vy, None, None, None, solid, dt,
               seed_tilt[0], seed_tilt[1])
        totals.append(int(depth.astype(np.int64).sum()))
    return depth, vx, vy, totals


def main():
    print("=== P2: Sum(water_depth) conservation (sealed flood) ===")
    solid, depth0 = make_scene()
    total0 = int(depth0.astype(np.int64).sum())
    depth, vx, vy, totals = run(solid, depth0, 1000)
    drift = [t - total0 for t in totals]
    max_drift = max(abs(x) for x in drift)
    print(f"  initial Sum (Q16.16 counts) = {total0}")
    print(f"  final   Sum                 = {totals[-1]}")
    print(f"  max |drift| over 1000 ticks = {max_drift} LSB counts "
          f"({max_drift / Q:.3e} m)")
    print(f"  min depth = {depth.min()} (>=0: {depth.min() >= 0}), "
          f"finite always (int -> trivially)")
    p2_ok = (max_drift == 0)
    print(f"  P2 BIT-CONSERVED: {p2_ok}")

    # With a tilt (water slides low-side): still sealed -> still conserved.
    solid2, depth0b = make_scene(seed=1)
    t0b = int(depth0b.astype(np.int64).sum())
    _, _, _, totals_b = run(solid2, depth0b, 500, seed_tilt=(0.1, -0.05))
    drift_b = max(abs(t - t0b) for t in totals_b)
    print(f"  tilt scenario max |drift| (500 ticks) = {drift_b} LSB "
          f"-> conserved: {drift_b == 0}")

    print("\n=== P1: bit-identity (two identical runs) ===")
    a = run(*make_scene(), 200, seed_tilt=(0.05, 0.03))
    b = run(*make_scene(), 200, seed_tilt=(0.05, 0.03))
    p1_depth = np.array_equal(a[0], b[0])
    p1_vx = np.array_equal(a[1], b[1])
    p1_vy = np.array_equal(a[2], b[2])
    print(f"  depth bit-identical: {p1_depth}")
    print(f"  flow_vx bit-identical: {p1_vx}")
    print(f"  flow_vy bit-identical: {p1_vy}")
    # non-vacuity
    moved = not np.array_equal(a[0], make_scene()[1])
    print(f"  non-vacuous (field evolved): {moved}")
    print(f"  max|vx| = {np.abs(a[1]).max()} (counts)")

    print("\nSUMMARY:",
          "P2", "PASS" if p2_ok and drift_b == 0 else "FAIL",
          "| P1", "PASS" if p1_depth and p1_vx and p1_vy and moved else "FAIL")


if __name__ == "__main__":
    main()
