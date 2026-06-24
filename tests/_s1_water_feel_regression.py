"""S1 feel-regression harness — float baseline vs Q16.16 integer water.

Captures three water scenarios at the SOLVER level (WaterSolver.step), driven the
same way the game drives it (dx = 1/3 m, default pipe params), for ~120 ticks.
The float build saves the baseline; the integer build re-runs and reports the
per-field max abs/rel difference (dequantized int -> float vs the float baseline).

Scenarios (each on a 24x24 hull-walled box, dt = 0.016 s, ~120 ticks):
  * pour   : a single source column held at 0.5 m in the centre (a tap)
  * flood  : half the box pre-filled to 0.4 m, dam-break levelling
  * blast  : a uniform 0.2 m sheet, a velocity kick (vx/vy seeded) = a blast wash

Usage:
    # On the FLOAT build (before migration):
    python tests/_s1_water_feel_regression.py --save tests/_s1_feel_float.pkl
    # On the INTEGER build (after migration):
    python tests/_s1_water_feel_regression.py --compare tests/_s1_feel_float.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

DT = 0.016
N_TICKS = 120
DX = 1.0 / 3.0
Q = 65536.0  # Q16.16 scale

# Does this build use the integer (Q16.16) water interface? Detected by trying a
# float step and catching the dtype error; simpler: check a module flag if present,
# else probe. We set the dtype per-build below.
INT_WATER = getattr(bp, "WATER_FIXEDPOINT", False)


def _solver(**ov):
    s = bp.WaterSolver()
    s.dx = DX
    for k, v in ov.items():
        setattr(s, k, v)
    return s


def _depth_dtype():
    return np.int32 if INT_WATER else np.float32


def _q(arr_f):
    """Float metres -> the build's depth dtype (Q16.16 int32 or float32)."""
    if INT_WATER:
        return np.round(arr_f.astype(np.float64) * Q).astype(np.int32)
    return arr_f.astype(np.float32)


def _deq(arr):
    """The build's depth field -> float metres (dequantize if integer)."""
    if INT_WATER:
        return arr.astype(np.float64) / Q
    return arr.astype(np.float64)


def _box(h, w):
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    return solid


def _capture_pour():
    h = w = 24
    solid = _box(h, w)
    depth = _q(np.zeros((h, w), np.float32))
    vx = _q(np.zeros((h, w), np.float32))
    vy = _q(np.zeros((h, w), np.float32))
    s = _solver()
    src_f = 0.5  # metres held at centre
    src_q = _q(np.array([[src_f]], np.float32))[0, 0]
    frames = []
    for _ in range(N_TICKS):
        depth[12, 12] = max(depth[12, 12], src_q) if not INT_WATER else \
            np.int32(max(int(depth[12, 12]), int(src_q)))
        s.step(depth, vx, vy, None, None, None, solid, DT, 0.0, 0.0)
        frames.append((_deq(depth).copy(), _deq(vx).copy(), _deq(vy).copy()))
    return frames


def _capture_flood():
    h = w = 24
    solid = _box(h, w)
    d0 = np.zeros((h, w), np.float32)
    d0[1:23, 1:12] = 0.4
    depth = _q(d0)
    vx = _q(np.zeros((h, w), np.float32))
    vy = _q(np.zeros((h, w), np.float32))
    s = _solver()
    frames = []
    for _ in range(N_TICKS):
        s.step(depth, vx, vy, None, None, None, solid, DT, 0.0, 0.0)
        frames.append((_deq(depth).copy(), _deq(vx).copy(), _deq(vy).copy()))
    return frames


def _capture_blast():
    h = w = 24
    solid = _box(h, w)
    depth = _q(np.full((h, w), 0.2, np.float32) * (~solid))
    # a blast wash: seed a velocity field (a radial-ish kick)
    vx0 = np.zeros((h, w), np.float32)
    vy0 = np.zeros((h, w), np.float32)
    vx0[1:23, 1:23] = 3.0
    vy0[1:23, 1:23] = -2.0
    vx = _q(vx0)
    vy = _q(vy0)
    s = _solver()
    frames = []
    for _ in range(N_TICKS):
        s.step(depth, vx, vy, None, None, None, solid, DT, 0.0, 0.0)
        frames.append((_deq(depth).copy(), _deq(vx).copy(), _deq(vy).copy()))
    return frames


def capture_all():
    return {
        "pour": _capture_pour(),
        "flood": _capture_flood(),
        "blast": _capture_blast(),
    }


def _diff(a_frames, b_frames):
    names = ("water_depth", "flow_vx", "flow_vy")
    out = {}
    for fi, name in enumerate(names):
        max_abs = 0.0
        max_rel = 0.0
        for (a, b) in zip(a_frames, b_frames):
            fa = a[fi].astype(np.float64)
            fb = b[fi].astype(np.float64)
            d = np.abs(fa - fb)
            max_abs = max(max_abs, float(d.max()))
            denom = np.abs(fa)
            mask = denom > 1e-6
            if mask.any():
                max_rel = max(max_rel, float((d[mask] / denom[mask]).max()))
        out[name] = (max_abs, max_rel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    args = ap.parse_args()

    print(f"build: INT_WATER={INT_WATER}")
    data = capture_all()

    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved float baseline -> {args.save}")
        return

    if args.compare:
        with open(args.compare, "rb") as f:
            base = pickle.load(f)
        print("feel-regression: integer (dequantized) vs float baseline")
        print(f"  ({N_TICKS} ticks, dt={DT}, dx={DX:.4f}; Q16.16 granularity "
              f"= {1.0/Q:.3e} m)")
        for scen in ("pour", "flood", "blast"):
            print(f"\n[{scen}]")
            d = _diff(base[scen], data[scen])
            for name, (ma, mr) in d.items():
                print(f"  {name:11s}: max|abs|={ma:.3e}  max|rel|={mr:.3e}")
        return

    # default: just self-check it runs
    for scen, frames in data.items():
        last = frames[-1][0]
        print(f"{scen}: final depth sum={last.sum():.6f}  max={last.max():.6f}")


if __name__ == "__main__":
    main()
