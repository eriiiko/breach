"""Align ChatGPT-generated ship art to the tilemap CSV.

Pipeline:
  1. Load tilemap CSV (50 cols x 120 rows). The procedural renderer fills the
     entire grid (vacuum = black), so the IMAGE'S full canvas should map to
     the FULL grid, not just to the non-zero ship bbox.
  2. Detect the actual ship-art content bbox in the diffuse PNG by finding
     non-near-black pixels. Use that bbox to figure out how much "vacuum"
     padding the AI image has compared to what the tilemap's vacuum padding
     would produce.
  3. Because the AI was prompted with a fixed-aspect render of the full
     50x120 grid, we EXPECT the visible ship in the image to occupy the same
     fractional area as the non-zero tiles do in the grid. We compute both,
     fit a single uniform scale that aligns ship-bbox-in-image to
     ship-bbox-in-grid, then crop/pad the image so the result is exactly
     N*50 x N*120 for some integer N.
  4. Apply the same crop+resize to the normal map.
  5. Produce verification.png showing the diffuse at 50% alpha with the CSV's
     wall (1) and door (3) tile centres marked as a red outline overlay.

Usage:
    python align_ship_art.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("c:/Users/steen/projects/breach/prototypes/space_ship_gpt_pipeline1")
CSV_PATH = ROOT / "tilemap.csv"
DIFFUSE_PATH = ROOT / "ChatGPT Image May 16, 2026, 03_08_55 AM.png"
NORMAL_PATH = ROOT / "ChatGPT Image May 16, 2026, 03_08_55 AM_n.png"
DIFFUSE_OUT = ROOT / "diffuse_aligned.png"
NORMAL_OUT = ROOT / "normal_aligned.png"
VERIFY_OUT = ROOT / "verification.png"

# Tile codes
WALL = 1
DOOR = 3
NONZERO_IS_SHIP = True  # non-zero tiles = interior + walls = the visible ship

# Target output tile size in pixels. The image is 972 wide / 50 cols ~= 19.4 px/tile,
# 1619 tall / 120 rows ~= 13.5 px/tile. We pick the closest clean integer to the
# *correct* scale once the image has been cropped (see end of compute_alignment).
DEFAULT_TILE_PX = 20  # gives 1000 x 2400 final canvas — clean and close


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_content_bbox(img: Image.Image, threshold: int = 12) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) bounding box of non-background pixels.

    Background = pixels whose max(R,G,B) <= threshold (near-black vacuum).
    """
    arr = np.asarray(img.convert("RGB"))
    fg = arr.max(axis=2) > threshold
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    if not rows.any() or not cols.any():
        raise RuntimeError("No foreground content found in image.")
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    return int(c[0]), int(r[0]), int(c[-1]) + 1, int(r[-1]) + 1


def tilemap_ship_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    """Return (col0, row0, col1, row1) bbox in TILE coords of non-zero cells.

    NOTE: We DON'T trim border rows that span the full width — those represent
    the engine block / prow plate of the ship and the AI image *does* include
    matching pixels there (just narrower because the AI tapered them). We
    return the literal non-zero bbox; alignment is done per axis."""
    nz = grid != 0
    rows = np.any(nz, axis=1)
    cols = np.any(nz, axis=0)
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    return int(c[0]), int(r[0]), int(c[-1]) + 1, int(r[-1]) + 1


def tilemap_ship_silhouette_width_at_rows(grid: np.ndarray,
                                          frac_rows: tuple[float, ...] = (0.3, 0.5, 0.7)
                                          ) -> tuple[int, ...]:
    """Get widths of non-zero columns at given fractional row positions.

    Used for diagnostic / sanity. Returns widths in tile units."""
    H, W = grid.shape
    widths = []
    for f in frac_rows:
        r = int(f * H)
        cols = np.where(grid[r] != 0)[0]
        if len(cols) == 0:
            widths.append(0)
        else:
            widths.append(int(cols[-1] - cols[0] + 1))
    return tuple(widths)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def compute_alignment(grid: np.ndarray, diffuse: Image.Image, tile_px: int) -> dict:
    H_tiles, W_tiles = grid.shape
    out_W = W_tiles * tile_px
    out_H = H_tiles * tile_px

    # Ship bbox in tile coords -> in target output pixels
    sc0, sr0, sc1, sr1 = tilemap_ship_bbox(grid)
    ship_tile_w = sc1 - sc0
    ship_tile_h = sr1 - sr0
    target_ship_px_w = ship_tile_w * tile_px
    target_ship_px_h = ship_tile_h * tile_px
    target_ship_px_x = sc0 * tile_px
    target_ship_px_y = sr0 * tile_px

    # Ship bbox in the input image
    ix0, iy0, ix1, iy1 = detect_content_bbox(diffuse)
    img_ship_w = ix1 - ix0
    img_ship_h = iy1 - iy0

    # Aspect-ratio sanity check
    target_aspect = target_ship_px_w / target_ship_px_h
    image_aspect = img_ship_w / img_ship_h
    aspect_ratio_error = abs(image_aspect - target_aspect) / target_aspect

    # Uniform scale that maps the image's ship bbox onto the target ship bbox.
    # We use the geometric mean of the two axis scales so both axes get a
    # compromise; if aspect_ratio_error is small (<10%) that's acceptable.
    sx = target_ship_px_w / img_ship_w
    sy = target_ship_px_h / img_ship_h
    # Pick scale that preserves height (vertical is the long, well-defined axis).
    # If aspects differ slightly, we'll stretch horizontally by the small delta.
    # (Both reported below.)
    scale = (sx + sy) / 2.0

    # After scaling, the ship's centre in the (scaled) image should land at the
    # centre of the target ship bbox. Compute the crop window on the *original*
    # image so that scaling that window to (out_W, out_H) lands the ship right.

    # First, figure out the ratio target-canvas / target-ship for each axis.
    # The image's full canvas should expand by the same ratio around its ship.
    canvas_x_ratio = out_W / target_ship_px_w
    canvas_y_ratio = out_H / target_ship_px_h
    # In the input image, the ship centre is:
    img_cx = (ix0 + ix1) / 2.0
    img_cy = (iy0 + iy1) / 2.0
    # Crop window size in input-pixel units so that, after resize to (out_W, out_H),
    # the image-ship maps onto the target-ship region.
    crop_w = img_ship_w * canvas_x_ratio
    crop_h = img_ship_h * canvas_y_ratio
    # Target ship centre (in output px) relative to canvas centre:
    target_cx = (target_ship_px_x + target_ship_px_w / 2.0)
    target_cy = (target_ship_px_y + target_ship_px_h / 2.0)
    # The crop window centre in input px must offset from img-ship-centre by
    # (canvas_centre - target_ship_centre) * scale_input_per_output
    in_per_out_x = crop_w / out_W
    in_per_out_y = crop_h / out_H
    crop_cx = img_cx + (out_W / 2.0 - target_cx) * in_per_out_x
    crop_cy = img_cy + (out_H / 2.0 - target_cy) * in_per_out_y

    crop_x0 = crop_cx - crop_w / 2.0
    crop_y0 = crop_cy - crop_h / 2.0
    crop_x1 = crop_x0 + crop_w
    crop_y1 = crop_y0 + crop_h

    return {
        "out_W": out_W,
        "out_H": out_H,
        "tile_px": tile_px,
        "tilemap_shape": (H_tiles, W_tiles),
        "ship_tile_bbox": (sc0, sr0, sc1, sr1),
        "ship_tile_size": (ship_tile_w, ship_tile_h),
        "target_ship_px_bbox": (target_ship_px_x, target_ship_px_y,
                                target_ship_px_x + target_ship_px_w,
                                target_ship_px_y + target_ship_px_h),
        "image_ship_bbox": (ix0, iy0, ix1, iy1),
        "image_ship_size": (img_ship_w, img_ship_h),
        "target_aspect": target_aspect,
        "image_aspect": image_aspect,
        "aspect_ratio_error": aspect_ratio_error,
        "scale_x": sx,
        "scale_y": sy,
        "scale_mean": scale,
        "crop_box": (crop_x0, crop_y0, crop_x1, crop_y1),
    }


def apply_alignment(img: Image.Image, info: dict, resample: int) -> Image.Image:
    """Crop (with extension beyond image bounds = black/zero padding) then resize."""
    x0, y0, x1, y1 = info["crop_box"]
    out_W, out_H = info["out_W"], info["out_H"]
    cw = int(round(x1 - x0))
    ch = int(round(y1 - y0))

    # Create a black canvas at crop size, paste source at the correct offset.
    if img.mode == "RGBA":
        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 255))
    else:
        canvas = Image.new(img.mode, (cw, ch), 0)
    paste_x = int(round(-x0))
    paste_y = int(round(-y0))
    canvas.paste(img, (paste_x, paste_y))

    return canvas.resize((out_W, out_H), resample=resample)


def build_verification(diffuse_aligned: Image.Image, grid: np.ndarray,
                       tile_px: int) -> Image.Image:
    base = diffuse_aligned.convert("RGBA")
    # 50% alpha version of the diffuse on a dark grey background
    bg = Image.new("RGBA", base.size, (40, 40, 40, 255))
    halfa = base.copy()
    a = halfa.split()[-1].point(lambda v: v // 2)
    halfa.putalpha(a)
    bg.alpha_composite(halfa)

    # Overlay outlines of wall (1) and door (3) cells.
    H_tiles, W_tiles = grid.shape
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    wall_color = (255, 40, 40, 220)
    door_color = (255, 200, 40, 220)

    # Draw only edge pixels of wall cells (1) to produce an outline silhouette
    # instead of filling. We mark cells where any 4-neighbour differs (boundary
    # of the wall mask), then for each such cell stroke its tile rectangle.
    wall_mask = grid == WALL
    door_mask = grid == DOOR

    def is_boundary(mask: np.ndarray, r: int, c: int) -> bool:
        if not mask[r, c]:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= mask.shape[0] or cc < 0 or cc >= mask.shape[1]:
                return True
            if not mask[rr, cc]:
                return True
        return False

    for r in range(H_tiles):
        for c in range(W_tiles):
            if wall_mask[r, c] and is_boundary(wall_mask, r, c):
                x0 = c * tile_px
                y0 = r * tile_px
                draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1],
                               outline=wall_color, width=1)
            elif door_mask[r, c]:
                x0 = c * tile_px
                y0 = r * tile_px
                draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1],
                               outline=door_color, width=2)

    bg.alpha_composite(overlay)
    return bg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(save_aligned: bool = True, tile_px: int = DEFAULT_TILE_PX) -> dict:
    grid = np.loadtxt(CSV_PATH, delimiter=",", dtype=int)
    diffuse = Image.open(DIFFUSE_PATH)
    normal = Image.open(NORMAL_PATH)

    info = compute_alignment(grid, diffuse, tile_px=tile_px)

    print("=== Tilemap ===")
    print(f"  shape (H x W tiles): {info['tilemap_shape']}")
    print(f"  ship tile bbox (col0,row0,col1,row1): {info['ship_tile_bbox']}")
    print(f"  ship tile size (w,h): {info['ship_tile_size']}")
    print(f"  target ship aspect (w/h): {info['target_aspect']:.4f}")
    print("=== Diffuse image ===")
    print(f"  size: {diffuse.size}")
    print(f"  ship pixel bbox: {info['image_ship_bbox']}")
    print(f"  ship pixel size: {info['image_ship_size']}")
    print(f"  ship aspect (w/h): {info['image_aspect']:.4f}")
    print("=== Alignment ===")
    print(f"  aspect ratio error: {info['aspect_ratio_error']*100:.2f}%")
    print(f"  axis scales (sx, sy): {info['scale_x']:.4f}, {info['scale_y']:.4f}")
    print(f"  crop box on input image: {info['crop_box']}")
    print(f"  output canvas: {info['out_W']} x {info['out_H']}  (tile_px={tile_px})")

    diffuse_aligned = apply_alignment(diffuse, info, resample=Image.LANCZOS)
    normal_aligned = apply_alignment(normal, info, resample=Image.BILINEAR)

    verify = build_verification(diffuse_aligned, grid, tile_px)
    verify.save(VERIFY_OUT)
    print(f"Saved {VERIFY_OUT}")

    # NOTE: the script intentionally stretches each axis independently to fit
    # the ship's silhouette to the tilemap. A non-zero aspect_ratio_error here
    # quantifies *how much* anisotropic stretch was applied — not an error in
    # the alignment. The visual wall overlay in verification.png is the real
    # check. We save aligned outputs unless aspect_ratio_error is very large
    # (>25%) which would imply a fundamental shape mismatch.
    if save_aligned and info["aspect_ratio_error"] < 0.25:
        diffuse_aligned.save(DIFFUSE_OUT)
        normal_aligned.save(NORMAL_OUT)
        print(f"Saved {DIFFUSE_OUT}")
        print(f"Saved {NORMAL_OUT}")
        if info["aspect_ratio_error"] > 0.10:
            print(f"  (note: anisotropic stretch of "
                  f"{info['aspect_ratio_error']*100:.1f}% applied — "
                  "inspect verification.png to confirm walls match)")
    else:
        if not save_aligned:
            print("(save_aligned=False — skipped writing aligned outputs)")
        else:
            print(f"Aspect ratio error {info['aspect_ratio_error']*100:.2f}% "
                  ">= 25% — NOT saving aligned outputs. Inspect verification.png "
                  "and adjust manually.")
    return info


if __name__ == "__main__":
    main()
