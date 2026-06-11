"""Compute [art.align] for the v2 art by locating the v1 (grid-aligned) art inside it.

The v1 diffuse is stretched exactly onto the grid (offset 0, art_w/grid_w px/tile),
and the v2 layers are the same geometry uncropped — so finding the v1 view inside
the v2 image (multi-scale template match) gives v2's offset_px + px_per_tile
directly. Writes a QA overlay so the result is judged by eye.

Usage:
    python tools/align_from_v1.py <v1_diffuse.png> <v2_art.png> <grid_w> <grid_h> <qa_out.png>
Prints the [art.align] values to paste into level.toml.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from skimage.feature import match_template


def main() -> None:
    v1_path, v2_path, grid_w, grid_h, qa_out = (
        sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    v1 = Image.open(v1_path).convert("L")
    v2 = Image.open(v2_path).convert("L")

    # search in a downscaled working space for speed; refine afterwards
    WORK_W = 480
    s2 = WORK_W / v2.width
    v2s = np.asarray(v2.resize((WORK_W, round(v2.height * s2)), Image.LANCZOS),
                     dtype=np.float32)

    best = None  # (score, scale_v1_to_v2work, y, x)
    # v1 view occupies an unknown sub-rect of v2; try template widths from 55%
    # to 100% of the v2 working width
    for frac in np.linspace(0.55, 1.0, 19):
        tw = int(WORK_W * frac)
        th = int(tw * v1.height / v1.width)
        if th >= v2s.shape[0] or tw < 40:
            continue
        tpl = np.asarray(v1.resize((tw, th), Image.LANCZOS), dtype=np.float32)
        r = match_template(v2s, tpl)
        y, x = np.unravel_index(np.argmax(r), r.shape)
        sc = float(r[y, x])
        if best is None or sc > best[0]:
            best = (sc, tw / v1.width, int(y), int(x))

    score, s_v1, y, x = best
    # v1 spans the whole grid: px/tile in v2 full-res = (v1_w*s_v1/s2)/grid_w
    ppt_x = (v1.width * s_v1 / s2) / grid_w
    ppt_y = (v1.height * s_v1 / s2) / grid_h
    off_x, off_y = x / s2, y / s2
    print(f"match score {score:.3f} (1.0 = perfect; below ~0.5 = don't trust)")
    print(f"px_per_tile: x {ppt_x:.2f} / y {ppt_y:.2f} (square-ness check)")
    print(f"[art.align]\noffset_px = [{off_x:.1f}, {off_y:.1f}]\n"
          f"px_per_tile = {(ppt_x + ppt_y) / 2:.2f}")

    # QA: v2 darkened, the matched v1 rect blended in at 50%
    qa = np.asarray(v2.convert("RGB"), dtype=np.float32) * 0.35
    v1r = v1.resize((round(v1.width * s_v1 / s2), round(v1.height * s_v1 / s2)),
                    Image.LANCZOS)
    v1a = np.asarray(v1r.convert("RGB"), dtype=np.float32)
    y0, x0 = round(off_y), round(off_x)
    y1 = min(y0 + v1a.shape[0], qa.shape[0]); x1 = min(x0 + v1a.shape[1], qa.shape[1])
    qa[y0:y1, x0:x1] = 0.5 * qa[y0:y1, x0:x1] + 0.6 * v1a[:y1 - y0, :x1 - x0]
    Image.fromarray(np.clip(qa, 0, 255).astype(np.uint8)).save(qa_out)
    print(f"QA overlay -> {qa_out} (v1 ghost should sit EXACTLY on the v2 ship)")


if __name__ == "__main__":
    main()
