"""Contact sheet of the muzzle flame (#87): the numpy twin of
``shaders/muzzle_flame.fs`` (``renderer/muzzle_flame.py``) rendered for every
``[render.muzzle_flame.kinds.*]`` row of config.toml -- one PNG per kind, a
row per random shot, a column per age -- on a dark floor through the game's
own tone curve, at the deepest zoom (48 world px per 0.333 m tile). The
instrument for tuning a kind's dials before looking in the game: no window,
no GL.

    <breach-py> tools/muzzle_flame_sheet.py OUT_DIR [--shots 6] [--kind rifle]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PX_PER_M = 48.0 / 0.333
AGES = (0.0, 0.4, 0.75)
FLOOR = (0.035, 0.038, 0.045)


def sheet(kind, seed: int, shots: int, noise, ramp) -> np.ndarray:
    from renderer import muzzle_flame as mf
    from renderer.blackbody import aces_curve
    half_w, half_h = 1.3 * max(kind.size, 1.0), 0.8 * max(kind.size, 1.0)
    w, h = int(2 * half_w * PX_PER_M), int(2 * half_h * PX_PER_M)
    xs = (np.arange(w) + 0.5) / PX_PER_M - 0.35 * half_w
    ys = (np.arange(h) + 0.5) / PX_PER_M - half_h
    x, y = np.meshgrid(xs, ys)
    rows = []
    for s in range(shots):
        shot = mf.shot_rows(kind, mf.shot_rng(seed, 1, s))
        cells = [mf.flame_radiance(x, y, shot, a, noise, ramp) for a in AGES]
        rows.append(np.concatenate(cells, axis=1))
    lin = np.concatenate(rows, axis=0) + np.asarray(FLOOR)
    return (np.power(aces_curve(lin), 1 / 2.2) * 255 + 0.5).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--kind", default=None)
    args = ap.parse_args(argv)
    from PIL import Image
    from config import CFG
    from renderer import muzzle_flame as mf
    from renderer.blackbody import BlackbodyRamp
    settings = mf.settings_from_config(CFG)
    ramp = BlackbodyRamp.from_config(CFG)
    noise = mf.noise_field()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, kind in settings.kinds.items():
        if args.kind and name != args.kind:
            continue
        img = sheet(kind, settings.seed, args.shots, noise, ramp)
        path = args.out_dir / f"muzzle_flame_{name}.png"
        Image.fromarray(img).save(path)
        print(f"{name}: {path} {img.shape[1]}x{img.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
