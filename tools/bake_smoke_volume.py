"""tools/bake_smoke_volume.py — bake the looping smoke detail volume (issue #90).

Thin CLI shell over the pyray-free solver in ``renderer/volume_bake.py``
(design: docs/volumetric_smoke_design_2026-10-14.md §2.1). Writes the asset
``assets/volumes/<name>.npz`` that ``renderer/gas_volume.py`` raymarches, plus
optional PNG mosaics for eyeballing the bake before it reaches the GPU.

Run (conda env `data`):
    python tools/bake_smoke_volume.py                       # default 128x128x32, 48 frames
    python tools/bake_smoke_volume.py --quick               # 64x64x16, 24 frames (~1 min)
    python tools/bake_smoke_volume.py --res 256 256 64 --frames 64 --name smoke_hd
    python tools/bake_smoke_volume.py --preview out/vol     # also write out/vol_*.png

The asset folder is gitignored (tens of MB); the game falls back to a
procedural volume when the default asset is missing and prints how to bake.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))     # renderer/__init__ imports simulation.*

import numpy as np

from renderer.volume_bake import (BakeParams, bake, mosaic, mosaic_z,
                                  save_volume)

DEFAULT_DIR = ROOT / "assets" / "volumes"
DEFAULT_NAME = "smoke_v1"


def _write_png(path: Path, img: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(str(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Bake the looping smoke detail volume (issue #90).")
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help=f"asset name under assets/volumes/ (default {DEFAULT_NAME})")
    ap.add_argument("--out", default=None, help="explicit output .npz path")
    ap.add_argument("--res", nargs=3, type=int, metavar=("X", "Y", "Z"),
                    default=None, help="grid resolution (default 128 128 32)")
    ap.add_argument("--frames", type=int, default=None, help="frames per loop")
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--period-tiles", type=float, default=None)
    ap.add_argument("--height-tiles", type=float, default=None)
    ap.add_argument("--loop-seconds", type=float, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="small fast bake (64x64x16, 24 frames, short warm-up)")
    ap.add_argument("--preview", default=None, metavar="PREFIX",
                    help="write PREFIX_smoke.png / _fire.png / _z.png mosaics")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override any BakeParams field, e.g. --set vort_eps=0.5")
    args = ap.parse_args(argv)

    p = BakeParams()
    if args.quick:
        p.nx, p.ny, p.nz = 64, 64, 16
        p.frames, p.warmup_steps = 24, 60
    if args.res:
        p.nx, p.ny, p.nz = args.res
    for name, val in (("frames", args.frames), ("substeps", args.substeps),
                      ("warmup_steps", args.warmup), ("seed", args.seed),
                      ("period_tiles", args.period_tiles),
                      ("height_tiles", args.height_tiles),
                      ("loop_seconds", args.loop_seconds)):
        if val is not None:
            setattr(p, name, val)
    for kv in args.set:
        k, _, v = kv.partition("=")
        if not hasattr(p, k):
            ap.error(f"unknown BakeParams field {k!r}")
        cur = getattr(p, k)
        setattr(p, k, type(cur)(float(v)) if isinstance(cur, (int, float))
                else v)

    out = Path(args.out) if args.out else DEFAULT_DIR / f"{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    res = bake(p, progress=print)
    dt = time.perf_counter() - t0
    save_volume(out, res)
    mb = res.volume.nbytes / 1e6
    print(f"[bake] wrote {out}  ({res.volume.shape}, {mb:.1f} MB raw, "
          f"{dt:.0f} s)")

    if args.preview:
        pre = Path(args.preview)
        _write_png(pre.with_name(pre.name + "_smoke.png"), mosaic(res.volume, 0))
        _write_png(pre.with_name(pre.name + "_fire.png"), mosaic(res.volume, 1))
        _write_png(pre.with_name(pre.name + "_z.png"), mosaic_z(res.volume, 0))
        print(f"[bake] previews -> {pre}_smoke.png / _fire.png / _z.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
