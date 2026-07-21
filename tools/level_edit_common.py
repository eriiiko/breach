"""tools/level_edit_common.py — pure helpers shared by the two level editors
(engine/15 §0: shared logic, separate UIs).

The painted-path tool (``tools/align_level_art.py``) and the tiled-path map
editor (``tools/map_editor.py``) are separate programs by decision
(2026-07-07), but their paint plumbing is identical and lives here: the
palette generator, the square brush, drag-line interpolation, the
newline-preserving CSV save, the undo ring, and the inverse of the
``level_loader.tile_to_art_px`` align transform. Everything in this module is
pure (numpy + stdlib, no raylib) — unit-tested via
``tests/test_level_editor_tool.py`` (through the align tool's re-exports) and
``tests/test_map_editor_tool.py``.

Palette rule (level format v2 §1.1 / engine/15 §1): **no tool may carry its
own material vocabulary.** :func:`build_palette` reads
``simulation.materials.MATERIAL_NAMES`` at call time, so a new material (one
config row) appears in every editor palette automatically — curated overlay
colours for the shipped set, a deterministic generated colour for anything
newer.
"""
from __future__ import annotations

import colorsys
import sys
from pathlib import Path

# Make project modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from level_loader import SPACE_CODE
from simulation.materials import MATERIAL_NAMES

BRUSH_MIN, BRUSH_MAX = 1, 9
UNDO_CAPACITY = 100

# Curated overlay colours for the shipped materials, keyed by canon NAME (the
# config vocabulary — ids stay in code). AIR maps to None: it is the absence
# of an overlay. Materials added later fall through to _auto_rgb.
_CURATED_RGB = {
    "air": None,
    "hull": (220, 60, 50),       # red
    "wood": (240, 150, 40),      # orange
    "door": (240, 220, 60),      # yellow
    "steel": (70, 130, 180),     # steel blue
    "glass": (80, 220, 230),     # cyan
    "furniture": (160, 110, 60),  # warm brown (crates)
}
_SPACE_RGB = (40, 70, 200)       # deep blue


def _auto_rgb(mat_id: int) -> tuple:
    """Deterministic, well-spread fallback colour for a material id that has
    no curated entry (golden-angle hue walk — consecutive new ids land far
    apart on the wheel)."""
    hue = (int(mat_id) * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def build_palette() -> dict:
    """Editor palette over the canon v2 CSV vocabulary: id -> (display name,
    overlay RGB | None). Ids are exactly ``MATERIAL_NAMES`` (ascending) plus
    ``SPACE_CODE`` last; names come from the material table, uppercased.
    Reads MATERIAL_NAMES at CALL time — a material added to the table shows
    up on the next call, no tool change (engine/15 §1 rule).
    """
    palette = {}
    for mid in sorted(MATERIAL_NAMES):
        name = MATERIAL_NAMES[mid]
        rgb = _CURATED_RGB.get(name, _auto_rgb(mid))
        palette[mid] = (name.upper(), rgb)
    palette[SPACE_CODE] = ("SPACE", _SPACE_RGB)
    return palette


def art_px_to_tile(ax, ay, offset_px, px_per_tile) -> tuple:
    """Inverse of :func:`level_loader.tile_to_art_px`: art pixel ->
    FRACTIONAL tile coordinates (floor() them for the containing tile index).
    ``px_per_tile`` is a scalar or an (x, y) pair, exactly like the forward
    transform."""
    if isinstance(px_per_tile, (list, tuple)):
        ppt_x, ppt_y = float(px_per_tile[0]), float(px_per_tile[1])
    else:
        ppt_x = ppt_y = float(px_per_tile)
    return ((float(ax) - float(offset_px[0])) / ppt_x,
            (float(ay) - float(offset_px[1])) / ppt_y)


def brush_rect(tx: int, ty: int, brush: int,
               grid_w: int, grid_h: int):
    """Inclusive cell rect (x0, y0, x1, y1) of a square brush of side
    ``brush`` centered on tile (tx, ty), clipped to the grid. Even sides
    extend the extra cell right/down of center. Returns None when the center
    tile is outside the grid — the brush paints nothing from out there."""
    if not (0 <= tx < grid_w and 0 <= ty < grid_h):
        return None
    lo = (int(brush) - 1) // 2
    hi = int(brush) // 2
    return (max(0, tx - lo), max(0, ty - lo),
            min(grid_w - 1, tx + hi), min(grid_h - 1, ty + hi))


def paint_tiles(grid: np.ndarray, tx: int, ty: int, mat_id: int,
                brush: int = 1) -> int:
    """Paint ``mat_id`` into ``grid`` IN PLACE with a square brush centered
    on tile (tx, ty) (clipped to the grid; no-op when the center is outside).
    Returns the number of cells whose value actually changed."""
    h, w = grid.shape
    r = brush_rect(int(tx), int(ty), brush, w, h)
    if r is None:
        return 0
    x0, y0, x1, y1 = r
    region = grid[y0:y1 + 1, x0:x1 + 1]
    changed = int(np.count_nonzero(region != mat_id))
    if changed:
        region[...] = mat_id
    return changed


def line_tiles(x0: int, y0: int, x1: int, y1: int) -> list:
    """Integer tile positions along the segment (x0,y0)->(x1,y1), inclusive.
    A mouse drag is painted once per frame; walking this line between the
    previous and current cursor tile keeps fast strokes connected instead of
    leaving gaps where the cursor skipped tiles."""
    steps = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)))
    if steps == 0:
        return [(int(x0), int(y0))]
    return [(round(x0 + (x1 - x0) * i / steps),
             round(y0 + (y1 - y0) * i / steps)) for i in range(steps + 1)]


def save_tilemap_csv(csv_path, grid: np.ndarray, write_bak: bool = True):
    """Rewrite ``csv_path`` from ``grid`` (canon int codes), preserving the
    file's newline convention (CRLF vs LF) and the single trailing newline —
    the same write style as tools/migrate_tilemap_v2.py. When ``write_bak``
    is True the original bytes go to ``<name>.bak`` first (the interactive
    tools pass True exactly once per session, so the .bak keeps the
    pre-session state). Returns the .bak path, or None when not written."""
    csv_path = Path(csv_path)
    original = csv_path.read_bytes()
    newline = "\r\n" if b"\r\n" in original else "\n"
    text = newline.join(
        ",".join(str(int(v)) for v in row) for row in np.asarray(grid).tolist()
    ) + newline
    bak = None
    if write_bak:
        bak = csv_path.with_name(csv_path.name + ".bak")
        bak.write_bytes(original)
    csv_path.write_bytes(text.encode("ascii"))
    return bak


class UndoRing:
    """Fixed-capacity LIFO ring of tilemap snapshots — one per paint stroke.

    ``push`` stores an independent copy; when the ring is full the OLDEST
    snapshot falls off. ``pop`` returns the most recent snapshot (newest
    first) or None when empty."""

    def __init__(self, capacity: int = UNDO_CAPACITY):
        self.capacity = int(capacity)
        self._snaps: list = []

    def __len__(self) -> int:
        return len(self._snaps)

    def push(self, grid: np.ndarray) -> None:
        self._snaps.append(np.array(grid, copy=True))
        if len(self._snaps) > self.capacity:
            self._snaps.pop(0)

    def pop(self):
        return self._snaps.pop() if self._snaps else None
