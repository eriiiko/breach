"""ML height + normal maps from a diffuse layer (level-art pipeline §5).

Monocular depth (Depth-Anything-V2) read as a height field, then the normal
map is its slope: n = normalize(-s*dh/dx, s*dh/dy_up, 1). Replaces Laigter's
normal stage (Laigter stays for specular); the height map is also the future
`floor_height` / per-pixel-submersion asset (water chapter §3).

Usage:
    python tools/depth_to_normal.py in.png out_height.png out_normal.png
        [--out-w W --out-h H]   # resize height before slopes (e.g. match 4x art)
        [--strength 6.0]        # slope gain — higher = punchier relief
        [--invert]              # flip height polarity (raised <-> sunken)
        [--flip-g]              # flip green channel if lights come from below
Model: depth-anything/Depth-Anything-V2-Small-hf (downloads to HF cache on
first run). Requires torch + transformers (base anaconda, home desktop).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out_height")
    ap.add_argument("out_normal")
    ap.add_argument("--out-w", type=int, default=0)
    ap.add_argument("--out-h", type=int, default=0)
    ap.add_argument("--strength", type=float, default=6.0)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--flip-g", action="store_true")
    ap.add_argument("--model", default=MODEL,
                    help="HF id; -Small-hf / -Base-hf / -Large-hf variants")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    img = Image.open(args.inp).convert("RGB")
    depth = pipeline("depth-estimation", model=args.model, device=device)(img)
    h = depth["predicted_depth"].squeeze().float().cpu().numpy()
    # Depth-Anything emits relative inverse depth: nearer (raised, top-down)
    # = larger. That is already height polarity; --invert flips it.
    h = (h - h.min()) / max(h.max() - h.min(), 1e-9)
    if args.invert:
        h = 1.0 - h

    target = (args.out_w or img.width, args.out_h or img.height)
    hi = Image.fromarray((h * 65535.0 + 0.5).astype(np.uint16))
    hi = hi.resize(target, Image.BICUBIC)
    hi.save(args.out_height)

    hf = np.asarray(hi, dtype=np.float32) / 65535.0
    gy, gx = np.gradient(hf)              # gy = d/d(row) (image-down)
    up = -gy if not args.flip_g else gy   # OpenGL G channel = +Y is image-up
    nx, ny, nz = -args.strength * gx, args.strength * up, np.ones_like(hf)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    Image.fromarray(((n * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)).save(
        args.out_normal)
    print(f"height {target[0]}x{target[1]} -> {args.out_height}\n"
          f"normal (strength {args.strength}) -> {args.out_normal}")


if __name__ == "__main__":
    main()
