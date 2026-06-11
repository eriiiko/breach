"""Tiled ESRGAN-family upscaler for .pth weights (UltraSharp, Remacri, ...).

Part of the level-art pipeline (docs/level_editor_and_format_v2_proposal.md §5).
Runs any spandrel-loadable single-image super-resolution model, tiled with
overlap so full ship layers fit in VRAM, on CUDA when available.

Usage:
    python tools/upscale_pth.py <model.pth> <input.png> <output.png>

Requires: torch (CUDA optional), spandrel, pillow, numpy — present in the
base anaconda env on the home desktop (spandrel installed 2026-06-11).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

TILE = 384   # input-pixel tile size processed per model call
PAD = 32     # overlap on each side, cropped away after upscale (kills seams)


def upscale(model_path: str, in_path: str, out_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelLoader().load_from_file(model_path)
    model = model.to(device).eval()
    scale = model.scale

    img = np.asarray(Image.open(in_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = img.shape
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)

    use_half = device == "cuda"  # dropped permanently on first NaN tile —
    # transformer SR (DAT/HAT) overflows fp16; ESRGAN keeps the fast path
    with torch.inference_mode():
        for y0 in range(0, h, TILE):
            for x0 in range(0, w, TILE):
                y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                py0, px0 = max(y0 - PAD, 0), max(x0 - PAD, 0)
                py1, px1 = min(y1 + PAD, h), min(x1 + PAD, w)
                tile = img[py0:py1, px0:px1]
                t = torch.from_numpy(tile.transpose(2, 0, 1))[None].to(device)
                with torch.autocast(device, dtype=torch.float16,
                                    enabled=use_half):
                    r = model(t)
                if use_half and bool(torch.isnan(r).any()):
                    use_half = False
                    print("fp16 NaN — falling back to fp32 for this model")
                    r = model(t)
                r = r.float().clamp_(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
                # crop the padding off in output space, place the core tile
                cy, cx = (y0 - py0) * scale, (x0 - px0) * scale
                out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                    r[cy:cy + (y1 - y0) * scale, cx:cx + (x1 - x0) * scale]

    Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(out_path)
    print(f"{Path(in_path).name} -> {Path(out_path).name}  "
          f"({w}x{h} -> {w*scale}x{h*scale}, {Path(model_path).stem}, {device})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    upscale(sys.argv[1], sys.argv[2], sys.argv[3])
