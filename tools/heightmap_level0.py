"""tools/heightmap_level0.py — level-0 heightmap re-baseline (heightmap tool §6.1).

Standalone numpy/PIL script. Re-baselines a level's per-pixel floor heightmap so
the FLOOR sits at 0 — retiring the water shader's ``height_floor`` render fudge
at the source (the cure baked into the PNG, see the design doc §4 / §8).

The problem (design doc §1.2): Depth-Anything emits *relative*, per-image-
normalized depth, so a level's floor sits at an arbitrary nonzero relief value.
The water depth math (``depth = tileDepth - (relief - height_floor) * scale``)
then has to OVER-FILL before water clears the floor — the "fill-forever" bug,
patched by the ``height_floor`` dial. Re-baselining the map (floor -> 0) means
``height_floor`` can default to 0: the floor is genuinely at relief 0.

Floor detection — the MATERIAL MASK, not a blind histogram (design doc §4,
REVIEW-CORRECTED). A global histogram mode breaks on wall-/furniture-dominated
interiors (the modal surface is the WALL, not the floor) and on sloped relief.
Instead we sample relief ONLY over known interior floor/air tiles (``MAT_AIR``
and NOT vacuum — the walkable interior), mapped tile->art-pixel through the
level's ``[art.align]`` transform, and take a ROBUST statistic (the median) of
that floor-tile relief. Immune to wall/furniture dominance and to slope.

Re-baseline: subtract the detected floor from the whole map (floor -> 0), clamp
negatives (dips below the floor) to 0 — the water only needs relief ABOVE the
floor — and keep the result in 0..1 (relative feature heights preserved; the
range is NOT rescaled). The original PNG is ``.bak``'d first, then overwritten
in place.

Run:
    C:/Users/steen/anaconda3/python.exe tools/heightmap_level0.py <level_name> \
        [--layer bare|furniture|destroyed] [--floor <value>] [--dry-run]

``--layer`` selects which art layer's heightmap to re-baseline (default ``bare``,
which is the heightmap level_loader exposes as ``LevelData.height_path``). The
furniture/destroyed layers are read straight off ``[art.<layer>] height`` in
level.toml. ``--floor`` overrides the detected floor value (skips detection).
``--dry-run`` reports the detected floor + projected new range WITHOUT writing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

# Very large ship art trips PIL's decompression-bomb guard; these maps are
# trusted local assets (3900x6456), so lift the cap.
Image.MAX_IMAGE_PIXELS = None

# Make the repo root importable so this tool can reuse the canon level_loader
# (the [art.align] transform + the material mask), exactly like the other tools.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import level_loader as L  # noqa: E402  (after sys.path setup)
from level_loader import tile_to_art_px, materials_from_tilemap  # noqa: E402


def _layer_height_path(lvl, layer: str) -> Path:
    """Resolve the heightmap PNG path for the requested art layer.

    ``bare`` reuses ``LevelData.height_path`` (the loader-parsed
    ``[art.bare] height`` / flat ``height``). ``furniture`` / ``destroyed`` are
    read straight off ``[art.<layer>] height`` in the raw toml — the loader does
    not currently surface per-layer heightmaps (that is the schema migration of
    a LATER step), so this tool reads them directly from the toml table.
    """
    if layer == "bare":
        if lvl.height_path is None:
            raise SystemExit(
                f"Level '{lvl.name}' has no [art.bare] height map "
                f"(LevelData.height_path is None) — nothing to re-baseline.")
        return Path(lvl.height_path)

    art_tbl = lvl.raw_toml.get("art", {})
    sub = art_tbl.get(layer, {}) if isinstance(art_tbl, dict) else {}
    rel = sub.get("height") if isinstance(sub, dict) else None
    if not rel:
        raise SystemExit(
            f"Level '{lvl.name}' has no [art.{layer}] height key in level.toml "
            f"— nothing to re-baseline for layer '{layer}'.")
    p = (lvl.path / rel).resolve()
    if not p.is_file():
        raise SystemExit(f"[art.{layer}] height declared but missing: {p}")
    return p


def _load_relief(path: Path) -> tuple[np.ndarray, int, str]:
    """Load a greyscale heightmap PNG as float relief normalized 0..1.

    Returns ``(relief01, maxval, mode)`` — ``relief01`` is float64 in [0, 1],
    ``maxval`` is the integer max for the source bit depth (255 for 8-bit,
    65535 for 16-bit ``I;16``), and ``mode`` is the PIL image mode (so the write
    back can round-trip the same bit depth).
    """
    im = Image.open(path)
    mode = im.mode
    arr = np.asarray(im)
    if arr.ndim == 3:
        # An RGB(A) "greyscale" map — collapse to luminance-free single channel
        # (the channels are equal for a true greyscale export; take channel 0).
        arr = arr[..., 0]
    if arr.dtype == np.uint16 or mode in ("I;16", "I;16B", "I", "I;16L"):
        maxval = 65535
    elif arr.dtype == np.uint8:
        maxval = 255
    else:
        # Float or exotic — normalize by the observed max, treat as 16-bit on
        # write (the safest round-trip for relief data).
        maxval = 65535
    relief01 = arr.astype(np.float64) / float(maxval)
    return relief01, maxval, mode


def _floor_tile_mask(lvl) -> np.ndarray:
    """Boolean (rows, cols) mask of interior floor/air tiles.

    Floor = ``MAT_AIR`` (material id 0) AND NOT vacuum — i.e. the walkable
    interior, the design doc's ``MAT_AIR`` / ``mobility > 0`` floor (SPACE
    code 9 is MAT_AIR + vacuum, the outside, and is excluded)."""
    from simulation.materials import MAT_AIR
    material, is_vacuum = materials_from_tilemap(lvl.tilemap, lvl.version)
    return (material == MAT_AIR) & (~is_vacuum)


def _sample_floor_relief(relief01: np.ndarray, lvl,
                         floor_mask: np.ndarray) -> np.ndarray:
    """Sample heightmap relief over every interior-floor tile's art-px region.

    For each floor tile (tx, ty) the ``[art.align]`` transform gives the art-px
    rect [x0,x1) x [y0,y1) that the renderer samples for that tile; we take the
    relief values inside it. Returns a flat float64 array of all sampled relief
    values (clipped to the map bounds; tiles fully off-map contribute nothing).
    """
    offset = lvl.art_offset_px
    ppt = lvl.art_px_per_tile
    if ppt is None:
        raise SystemExit(
            "Level has no resolved px_per_tile (could not read art dims) — "
            "cannot map floor tiles to heightmap pixels.")
    H, W = relief01.shape
    ys, xs = np.where(floor_mask)
    samples = []
    for ty, tx in zip(ys.tolist(), xs.tolist()):
        # Tile-corner art pixels: (tx,ty) top-left, (tx+1,ty+1) bottom-right.
        x0f, y0f = tile_to_art_px(tx, ty, offset, ppt)
        x1f, y1f = tile_to_art_px(tx + 1, ty + 1, offset, ppt)
        x0, x1 = int(np.floor(min(x0f, x1f))), int(np.ceil(max(x0f, x1f)))
        y0, y1 = int(np.floor(min(y0f, y1f))), int(np.ceil(max(y0f, y1f)))
        x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        samples.append(relief01[y0:y1, x0:x1].ravel())
    if not samples:
        raise SystemExit(
            "No floor-tile pixels landed inside the heightmap — check the "
            "[art.align] transform / heightmap dimensions.")
    return np.concatenate(samples)


def _write_relief(path: Path, relief01: np.ndarray, maxval: int,
                  mode: str) -> None:
    """Write a 0..1 relief array back as a greyscale PNG at the source bit depth."""
    q = np.clip(np.rint(relief01 * float(maxval)), 0, maxval)
    if maxval == 65535:
        # uint16 -> PIL infers mode "I;16" (the deprecated `mode=` kwarg is gone
        # in Pillow 13); the array dtype is the contract now.
        out = Image.fromarray(q.astype(np.uint16))
    else:
        out = Image.fromarray(q.astype(np.uint8))   # uint8 -> mode "L"
    out.save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-baseline a level heightmap so the floor sits at 0 "
                    "(retires the water height_floor fudge at the source).")
    ap.add_argument("level_name", help="level folder name under levels/")
    ap.add_argument("--layer", default="bare",
                    choices=["bare", "furniture", "destroyed"],
                    help="which art layer's heightmap to re-baseline "
                         "(default: bare)")
    ap.add_argument("--floor", type=float, default=None,
                    help="manual floor relief override (0..1); skips detection")
    ap.add_argument("--dry-run", action="store_true",
                    help="report detection + projected range, write nothing")
    args = ap.parse_args(argv)

    lvl = L.load(args.level_name)
    height_path = _layer_height_path(lvl, args.layer)

    print(f"Level:     {lvl.name}  ({lvl.width}x{lvl.height} tiles, "
          f"v{lvl.version})")
    print(f"Layer:     {args.layer}")
    print(f"Heightmap: {height_path}")

    relief01, maxval, mode = _load_relief(height_path)
    H, W = relief01.shape
    bits = 16 if maxval == 65535 else 8
    print(f"  loaded {W}x{H} {bits}-bit greyscale (mode {mode}), "
          f"relief min/median/max = "
          f"{relief01.min():.4f} / {np.median(relief01):.4f} / "
          f"{relief01.max():.4f}")

    # ---- floor detection via the material mask -------------------------------
    if args.floor is not None:
        floor = float(args.floor)
        n_floor_tiles = -1
        print(f"Floor:     {floor:.4f} (manual --floor override)")
    else:
        floor_mask = _floor_tile_mask(lvl)
        n_floor_tiles = int(floor_mask.sum())
        floor_samples = _sample_floor_relief(relief01, lvl, floor_mask)
        floor = float(np.median(floor_samples))
        print(f"Floor:     {floor:.4f}  (median of {floor_samples.size} "
              f"heightmap pixels over {n_floor_tiles} interior floor tiles)")
        print(f"           floor-relief percentiles "
              f"[p5 p25 p50 p75 p95] = "
              + " ".join(f"{p:.4f}" for p in np.percentile(
                  floor_samples, [5, 25, 50, 75, 95])))

    # ---- re-baseline: subtract floor, clamp dips to 0, stay in 0..1 ----------
    rebased = np.clip(relief01 - floor, 0.0, 1.0)
    print(f"Re-based:  new relief min/median/max = "
          f"{rebased.min():.4f} / {np.median(rebased):.4f} / "
          f"{rebased.max():.4f}  (floor now at 0)")
    clamped = int((relief01 < floor).sum())
    print(f"           {clamped} pixels clamped to 0 (dips below the floor); "
          f"feature heights above the floor preserved")

    if args.dry_run:
        print("--dry-run: no files written.")
        return 0

    # ---- .bak the original once, then overwrite in place ---------------------
    bak_path = height_path.with_suffix(height_path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy2(height_path, bak_path)
        print(f"Backup:    {bak_path.name} (original preserved)")
    else:
        print(f"Backup:    {bak_path.name} already exists — left as-is "
              f"(first run's original is the true pre-rebaseline map)")

    _write_relief(height_path, rebased, maxval, mode)
    print(f"Wrote:     {height_path.name}  (floor re-baselined to 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
