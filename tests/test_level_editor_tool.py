"""Editor v1 — MATERIAL PAINT + CSV save in tools/align_level_art.py
(standalone slice of F4; proposal §4).

Pins the pure module-level helpers the interactive tool is built from:

  - PALETTE speaks exactly the canon v2 CSV vocabulary (material ids +
    SPACE_CODE) with canon names.
  - paint_tiles: square brush footprints (odd/even sides), edge clipping,
    out-of-grid no-op, changed-cell counting, id values written.
  - art_px_to_tile is the exact inverse of level_loader.tile_to_art_px
    (scalar and per-axis transforms; floor() containment).
  - line_tiles connects a drag segment without gaps.
  - save_tilemap_csv round-trips values and preserves the newline convention
    (CRLF and LF) + single trailing newline, with .bak semantics.
  - UndoRing: LIFO order, capacity eviction, snapshot independence.

The interactive pyray parts are exercised by ``tools/align_level_art.py
<level> --auto`` (paint HUD path + a programmatic paint-stroke check against
a tmp copy — raylib has no input injection, so --auto calls these same
factored functions directly).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_level_editor_tool.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from level_loader import SPACE_CODE, tile_to_art_px  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_FURNITURE,  # noqa: E402
                                  MAT_GLASS, MAT_HULL, MAT_STEEL, MAT_WOOD,
                                  MATERIAL_NAMES)
from align_level_art import (PALETTE, UndoRing, art_px_to_tile, brush_rect,  # noqa: E402
                             line_tiles, paint_tiles, save_tilemap_csv)


# ---------------------------------------------------------------------------
# Palette — canon vocabulary only
# ---------------------------------------------------------------------------

def test_palette_is_canon_v2_vocabulary():
    """The paintable ids are exactly the v2 CSV codes: every canon material
    id + SPACE_CODE, nothing else; names come from MATERIAL_NAMES."""
    assert set(PALETTE) == set(MATERIAL_NAMES) | {SPACE_CODE}
    for mid, name in MATERIAL_NAMES.items():
        assert PALETTE[mid][0] == name.upper()
    assert PALETTE[SPACE_CODE][0] == "SPACE"
    # AIR is the absence of an overlay; everything else has an RGB fill.
    assert PALETTE[MAT_AIR][1] is None
    for mid in (MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_STEEL, MAT_GLASS,
                MAT_FURNITURE, SPACE_CODE):
        rgb = PALETTE[mid][1]
        assert len(rgb) == 3 and all(0 <= v <= 255 for v in rgb)
    # FURNITURE is paintable on key 6 with its own chip colour, visually
    # distinct from WOOD's orange (both are warm — the ids must read apart).
    assert PALETTE[MAT_FURNITURE][0] == "FURNITURE"
    assert PALETTE[MAT_FURNITURE][1] != PALETTE[MAT_WOOD][1]


# ---------------------------------------------------------------------------
# paint_tiles — brush footprints, clamping, ids, changed counts
# ---------------------------------------------------------------------------

def test_paint_single_cell():
    g = np.zeros((8, 10), dtype=np.int32)
    assert paint_tiles(g, 3, 2, MAT_HULL) == 1
    assert g[2, 3] == MAT_HULL
    assert int(np.count_nonzero(g)) == 1          # nothing else touched


def test_paint_brush_3_centered_footprint():
    g = np.zeros((8, 10), dtype=np.int32)
    assert paint_tiles(g, 4, 3, MAT_WOOD, brush=3) == 9
    assert (g[2:5, 3:6] == MAT_WOOD).all()
    assert int(np.count_nonzero(g)) == 9


def test_paint_brush_2_extends_right_down():
    """Even brush sides extend the extra cell right/down of the center."""
    g = np.zeros((8, 10), dtype=np.int32)
    assert paint_tiles(g, 4, 3, MAT_STEEL, brush=2) == 4
    assert (g[3:5, 4:6] == MAT_STEEL).all()
    assert int(np.count_nonzero(g)) == 4


def test_paint_clamps_at_edges():
    g = np.zeros((8, 10), dtype=np.int32)
    # 3x3 at the top-left corner clips to 2x2.
    assert paint_tiles(g, 0, 0, MAT_GLASS, brush=3) == 4
    assert (g[0:2, 0:2] == MAT_GLASS).all()
    # 3x3 at the bottom-right corner clips to 2x2.
    assert paint_tiles(g, 9, 7, MAT_DOOR, brush=3) == 4
    assert (g[6:8, 8:10] == MAT_DOOR).all()
    # A 9x9 brush from the center floods the whole small grid.
    g2 = np.zeros((5, 5), dtype=np.int32)
    assert paint_tiles(g2, 2, 2, SPACE_CODE, brush=9) == 25
    assert (g2 == SPACE_CODE).all()


def test_paint_outside_grid_is_noop():
    g = np.full((8, 10), MAT_HULL, dtype=np.int32)
    before = g.copy()
    for tx, ty in ((-1, 0), (10, 0), (0, -1), (0, 8), (-5, -5)):
        assert paint_tiles(g, tx, ty, MAT_AIR, brush=5) == 0
    assert np.array_equal(g, before)
    # brush_rect agrees: center outside -> None; inside -> clipped rect.
    assert brush_rect(-1, 0, 3, 10, 8) is None
    assert brush_rect(0, 0, 3, 10, 8) == (0, 0, 1, 1)
    assert brush_rect(9, 7, 3, 10, 8) == (8, 6, 9, 7)


def test_paint_changed_count_skips_already_painted():
    g = np.zeros((8, 10), dtype=np.int32)
    g[3, 3:6] = MAT_WOOD                       # one row already wood
    assert paint_tiles(g, 4, 3, MAT_WOOD, brush=3) == 6   # 9 - 3 unchanged
    assert (g[2:5, 3:6] == MAT_WOOD).all()
    assert paint_tiles(g, 4, 3, MAT_WOOD, brush=3) == 0   # repaint = no-op
    # Eraser is just painting AIR.
    assert paint_tiles(g, 4, 3, MAT_AIR, brush=3) == 9
    assert int(np.count_nonzero(g)) == 0


# ---------------------------------------------------------------------------
# art_px_to_tile — exact inverse of tile_to_art_px
# ---------------------------------------------------------------------------

_TRANSFORMS = [
    ((0.0, 0.0), 24.0),                 # scalar, no offset
    ((7.0, 9.0), 78.0),                 # scalar + offset
    ((1.0, -38.4), (77.4, 54.44)),      # the vessel-2 per-axis transform
    ((-10.0, 20.0), (12.5, 31.25)),     # negative offset + per-axis
]


def test_art_px_to_tile_forward_inverse_identity():
    for offset, ppt in _TRANSFORMS:
        for tile in ((0.0, 0.0), (3.0, 5.0), (0.5, 2.25), (49.0, 119.0),
                     (-2.0, -1.5)):
            ax, ay = tile_to_art_px(tile[0], tile[1], offset, ppt)
            assert art_px_to_tile(ax, ay, offset, ppt) == pytest.approx(tile)
    # And the other direction: art px -> tile -> art px.
    for offset, ppt in _TRANSFORMS:
        for ax, ay in ((0.0, 0.0), (123.4, -56.7), (3899.0, 6899.0)):
            t = art_px_to_tile(ax, ay, offset, ppt)
            assert (tile_to_art_px(t[0], t[1], offset, ppt)
                    == pytest.approx((ax, ay)))


def test_art_px_to_tile_scalar_equals_pair():
    assert (art_px_to_tile(100.0, 200.0, (10.0, -8.0), 12.5)
            == art_px_to_tile(100.0, 200.0, (10.0, -8.0), (12.5, 12.5)))


def test_art_px_to_tile_floor_containment():
    """A point strictly inside tile (tx, ty)'s art rect floors back to
    (tx, ty) — the cursor-to-tile resolution the paint brush relies on."""
    for offset, ppt in _TRANSFORMS:
        for tx, ty in ((0, 0), (7, 11), (49, 119)):
            ax, ay = tile_to_art_px(tx + 0.5, ty + 0.5, offset, ppt)
            ftx, fty = art_px_to_tile(ax, ay, offset, ppt)
            assert (int(np.floor(ftx)), int(np.floor(fty))) == (tx, ty)


# ---------------------------------------------------------------------------
# line_tiles — connected drag strokes
# ---------------------------------------------------------------------------

def test_line_tiles_connectivity():
    assert line_tiles(3, 4, 3, 4) == [(3, 4)]
    assert line_tiles(0, 0, 3, 0) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert line_tiles(0, 0, 2, 2) == [(0, 0), (1, 1), (2, 2)]
    # A steep skip: every consecutive pair differs by <= 1 per axis, and the
    # endpoints are exact.
    pts = line_tiles(5, 2, -3, 17)
    assert pts[0] == (5, 2) and pts[-1] == (-3, 17)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert abs(x1 - x0) <= 1 and abs(y1 - y0) <= 1


# ---------------------------------------------------------------------------
# save_tilemap_csv — newline-preserving round-trip + .bak
# ---------------------------------------------------------------------------

def _grid_csv_bytes(grid: np.ndarray, newline: bytes) -> bytes:
    return newline.join(
        b",".join(str(int(v)).encode() for v in row) for row in grid.tolist()
    ) + newline


def test_save_tilemap_csv_crlf_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    grid = rng.choice([0, 1, 2, 3, 4, 5, 9], size=(12, 9)).astype(np.int32)
    csv = tmp_path / "tilemap.csv"
    csv.write_bytes(_grid_csv_bytes(grid, b"\r\n"))   # CRLF like the vessel
    original = csv.read_bytes()

    edited = grid.copy()
    assert paint_tiles(edited, 4, 6, MAT_HULL, brush=3) > 0
    bak = save_tilemap_csv(csv, edited, write_bak=True)

    # .bak carries the pre-save bytes, exactly.
    assert bak == tmp_path / "tilemap.csv.bak"
    assert bak.read_bytes() == original
    # CRLF convention + single trailing newline preserved.
    data = csv.read_bytes()
    assert data.endswith(b"\r\n") and not data.endswith(b"\r\n\r\n")
    assert data.count(b"\r\n") == 12                  # one per row
    assert b"\n" not in data.replace(b"\r\n", b"")    # no stray bare LFs
    # Values round-trip through the same reader the level loader uses.
    back = np.loadtxt(csv, delimiter=",", dtype=np.int32)
    assert np.array_equal(back, edited)


def test_save_tilemap_csv_lf_stays_lf(tmp_path):
    grid = np.arange(20, dtype=np.int32).reshape(4, 5) % 6
    csv = tmp_path / "tilemap.csv"
    csv.write_bytes(_grid_csv_bytes(grid, b"\n"))
    save_tilemap_csv(csv, grid, write_bak=True)
    data = csv.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert np.array_equal(np.loadtxt(csv, delimiter=",", dtype=np.int32), grid)


def test_save_tilemap_csv_bak_once_semantics(tmp_path):
    """write_bak=False writes no .bak; a second save with write_bak=False
    leaves the first .bak intact (the tool's once-per-session contract)."""
    grid = np.zeros((3, 3), dtype=np.int32)
    csv = tmp_path / "tilemap.csv"
    csv.write_bytes(_grid_csv_bytes(grid, b"\r\n"))
    pre_session = csv.read_bytes()

    g1 = grid.copy()
    paint_tiles(g1, 1, 1, MAT_HULL)
    assert save_tilemap_csv(csv, g1, write_bak=False) is None
    assert not (tmp_path / "tilemap.csv.bak").exists()

    bak = save_tilemap_csv(csv, g1, write_bak=True)   # "first save of session"
    g2 = g1.copy()
    paint_tiles(g2, 0, 0, MAT_GLASS)
    assert save_tilemap_csv(csv, g2, write_bak=False) is None
    assert bak.read_bytes() != pre_session            # bak = pre-FIRST-save
    assert np.array_equal(np.loadtxt(bak, delimiter=",", dtype=np.int32), g1)
    assert np.array_equal(np.loadtxt(csv, delimiter=",", dtype=np.int32), g2)


# ---------------------------------------------------------------------------
# UndoRing — one snapshot per stroke
# ---------------------------------------------------------------------------

def test_undo_ring_lifo_and_empty():
    ring = UndoRing(capacity=10)
    assert len(ring) == 0 and ring.pop() is None
    a = np.full((2, 2), 1, dtype=np.int32)
    b = np.full((2, 2), 2, dtype=np.int32)
    ring.push(a)
    ring.push(b)
    assert len(ring) == 2
    assert np.array_equal(ring.pop(), b)
    assert np.array_equal(ring.pop(), a)
    assert ring.pop() is None


def test_undo_ring_capacity_evicts_oldest():
    ring = UndoRing(capacity=5)
    for i in range(8):
        ring.push(np.full((2, 2), i, dtype=np.int32))
    assert len(ring) == 5
    for expect in (7, 6, 5, 4, 3):                 # 0..2 fell off the ring
        assert int(ring.pop()[0, 0]) == expect
    assert ring.pop() is None


def test_undo_ring_snapshots_are_copies():
    """Mutating the live grid after push must not retro-edit the snapshot —
    the stroke flow is push(pre-stroke copy) then paint in place."""
    ring = UndoRing()
    g = np.zeros((4, 4), dtype=np.int32)
    ring.push(g)
    paint_tiles(g, 1, 1, MAT_WOOD, brush=3)        # the stroke
    snap = ring.pop()
    assert int(np.count_nonzero(snap)) == 0        # snapshot is pre-stroke
    g[...] = snap                                  # Ctrl+Z restore
    assert int(np.count_nonzero(g)) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
