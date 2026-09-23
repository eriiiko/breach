"""Timing bench for the CPU radiation sweep (ray-engine-v2 P1, design v3 §10).

Design §10 estimated ~10 ms per tick at 128x256 on one core, from a GUESSED
~5 ns per cell-update ("the number that is not known is a single
cell-update's cost"). This bench MEASURES it: the C++ sweep
(cpp/src/radiation_sweep.cpp, the binding's RadiationSweep.run) on a random
scene at three grid sizes, S16, shear AND step, and prints ms per tick and
ns per cell-update (one cell-update = one cell in one ordinate).

The scene is a bordered room with random opaque solids (about 12 % of the
interior) at fire temperatures and a few bodies — a busy scene, not an empty
one, so the timing is not flattered by a uniform field.

Run:
    C:/Users/steen/anaconda3/python.exe tools/bench_radiation_sweep.py [--iters N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
import temperature_scale  # noqa: E402

ONE = 65536
SIZES = ((72, 46), (128, 256), (256, 512))


def _table():
    ts = temperature_scale.load(CFG)
    tbl = bp.EmissiveTable()
    # T6 (issue #12): [physics.fire] rad_scale (the old cast's fitted key) is
    # deleted; [physics.radiation] rad_scale_derived is the sweep's own live
    # scale, and this bench is timing the sweep, so it is the right key too.
    tbl.rad_scale = float(CFG.physics.radiation.rad_scale_derived)
    tbl.kelvin_ambient = float(ts.kelvin_ambient)
    tbl.k_temp_to_kelvin = float(ts.k_temp_to_kelvin)
    tbl.bake()
    return tbl, int(round(float(ts.kelvin_ambient) * ONE))


def _scene(h, w, rng):
    a = np.zeros((h, w), dtype=np.int32)
    ts = np.zeros((h, w), dtype=bool)
    T = np.zeros((h, w), dtype=np.int32)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = ONE
    inner = rng.random((h, w)) < 0.12
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    a[inner] = rng.choice([ONE, 32768, 19661], size=int(inner.sum()))
    ts[:] = a > 0
    hot = inner & (rng.random((h, w)) < 0.3)
    T[hot] = rng.choice([300 << 16, 1263 << 16, 5000 << 16], size=int(hot.sum()))
    d = a.copy()
    bodies = (~ts) & (rng.random((h, w)) < 0.01)
    d[bodies] = ONE
    his = np.full((h, w), 3, dtype=np.int32)
    return (np.ascontiguousarray(T), np.ascontiguousarray(a), np.ascontiguousarray(d),
            his, np.ascontiguousarray(ts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args(argv)
    tbl, t_amb_q = _table()
    rng = np.random.default_rng(20260916)
    print(f"radiation sweep CPU timing, S16, {args.iters} iterations per point "
          f"(one cell-update = one cell in one ordinate)")
    print(f"{'grid':>10} {'transport':>10} {'ms/tick':>10} {'ns/cell-update':>16} {'cells':>10}")
    for (h, w) in SIZES:
        T, a, d, his, ts = _scene(h, w, rng)
        for name, transport in (("shear", bp.RadiationSweep.SHEAR), ("step", bp.RadiationSweep.STEP)):
            sweep = bp.RadiationSweep()
            planes = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]
            # warm-up (scratch allocation) outside the timed loop
            sweep.run(T, a, d, his, ts, tbl, None, t_amb_q, 0, transport, 16, *planes)
            best = None
            for _ in range(3):
                for p in planes:
                    p.fill(0)
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    sweep.run(T, a, d, his, ts, tbl, None, t_amb_q, 0, transport, 16, *planes)
                dt = (time.perf_counter() - t0) / args.iters
                best = dt if best is None else min(best, dt)
            ident = int(planes[0].sum()) + int(planes[1].sum()) + int(planes[2].sum())
            assert ident == 0, "the identity failed inside the bench"
            n_updates = h * w * 16
            print(f"{h:>4}x{w:<5} {name:>10} {best * 1e3:10.2f} {best * 1e9 / n_updates:16.1f} {h * w:10d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
