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

Authored material relief (no torch / transformers required):
    python tools/depth_to_normal.py relief.png height.png normal.png \
        --height-input --strength 5 --blur 0.8 --tileable
This path uses make_tileset's canonical row-down normal convention, unlike
the historical ML path's default. --flip-g flips either path's green output.
Strength is full-range relief in output pixels. Height output is grayscale
16-bit PNG, not a calibrated physical displacement. --alpha-from albedo.png
preserves coverage for standalone rugs; omit --tileable for those assets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def smooth_height(height: np.ndarray, sigma: float, tileable: bool) -> np.ndarray:
    """Separable Gaussian smoothing without an intermediate 8-bit image.

    A uint8 PIL blur discarded shallow relief even for 16-bit input. Keep the
    scalar field in float64 until the one final 16-bit PNG quantization.
    """
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    result = height.astype(np.float64)
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="wrap" if tileable else "edge")
        blurred = np.zeros_like(result)
        for index, weight in enumerate(weights):
            sel = [slice(None), slice(None)]
            sel[axis] = slice(index, index + result.shape[axis])
            blurred += weight * padded[tuple(sel)]
        result = blurred
    return result


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
    ap.add_argument("--height-input", action="store_true",
                    help="Input is authored grayscale relief, not a scene photograph; no ML dependencies")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="Height-input smoothing radius in source pixels")
    ap.add_argument("--tileable", action="store_true",
                    help="Height-input gradients wrap at both image boundaries")
    ap.add_argument("--alpha-from", type=Path,
                    help="Height-input: copy coverage from this albedo into the normal PNG")
    args = ap.parse_args()

    if args.height_input:
        # Use the tiled baker's existing normal kit and row-down convention.
        # This path deliberately avoids scene-depth inference for flat materials.
        from make_tileset import normal_from_height, encode_normal

        if args.blur < 0 or not np.isfinite(args.blur):
            ap.error("--blur must be finite and nonnegative")
        if args.strength < 0 or not np.isfinite(args.strength):
            ap.error("--strength must be finite and nonnegative")
        source = Image.open(args.inp)
        if source.mode in ("I;16", "I;16L", "I;16B", "I"):
            hf = np.asarray(source, dtype=np.float32) / 65535.0
        else:
            hf = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
        if args.blur:
            hf = smooth_height(hf, args.blur, args.tileable)
        target = (args.out_w or source.width, args.out_h or source.height)
        hi = Image.fromarray((np.clip(hf, 0, 1) * 65535 + 0.5).astype(np.uint16))
        if hi.size != target:
            hi = hi.resize(target, Image.Resampling.BICUBIC)
        hf = np.asarray(hi, dtype=np.float32) / 65535.0
        if args.invert:
            hf = 1.0 - hf
            hi = Image.fromarray((hf * 65535 + 0.5).astype(np.uint16))
        height_px = hf * args.strength
        if args.tileable:
            normals = normal_from_height(np.pad(height_px, 1, mode="wrap"))[1:-1, 1:-1]
        else:
            normals = normal_from_height(height_px)
        if args.flip_g:
            normals[..., 1] *= -1
        encoded = encode_normal(normals)
        if args.alpha_from:
            albedo = Image.open(args.alpha_from).convert("RGBA")
            if albedo.size != target:
                ap.error("--alpha-from dimensions must match output dimensions")
            alpha = np.asarray(albedo)[..., 3]
            encoded[alpha == 0] = (128, 128, 255)
            encoded = np.dstack((encoded, alpha))
        hi.save(args.out_height)
        Image.fromarray(encoded).save(args.out_normal)
        print(f"authored height {target[0]}x{target[1]} -> {args.out_height}\n"
              f"normal (row-down, relief {args.strength} px) -> {args.out_normal}")
        return

    import torch
    from transformers import pipeline

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
