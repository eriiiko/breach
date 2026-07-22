"""Pane layout for the map editor — tools/editor_layout.py (Arc C, C0).

Headless: the panes shell is a PURE geometry layer (no raylib window), so it
is fully unit-testable without opening a GL window. The interactive editor
(map_editor.run_editor) needs a real GL context + a baked level and is not
constructible headlessly; its input/draw plumbing is exercised by ``--auto``
and rides on these pure helpers, which carry the contract:

  - the six panes TILE the window exactly (no overlap, no gaps) at any size;
  - the canvas viewport transform round-trips (tile -> screen -> tile);
  - pane hit-testing resolves the canvas first (shared seam edits the map).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from editor_layout import (PANE_KEYS, Rect, compute_panes,  # noqa: E402
                           fit_camera, pane_at, screen_from_world,
                           world_from_screen)
from level_edit_common import art_px_to_tile  # noqa: E402

# A spread of window sizes: default, resized, awkward primes, and windows so
# small the chrome cannot fully fit (clamps must still yield a valid tiling).
WIN_SIZES = [(1280, 920), (1920, 1080), (1024, 768), (800, 600),
             (1001, 733), (400, 300), (200, 150), (120, 90), (60, 60)]


def _overlap(a: Rect, b: Rect) -> bool:
    """True when two rects share positive area."""
    return (a.x < b.x + b.w and b.x < a.x + a.w
            and a.y < b.y + b.h and b.y < a.y + a.h)


@pytest.mark.parametrize("win", WIN_SIZES)
def test_panes_tile_window_exactly(win):
    w, h = win
    panes = compute_panes(w, h)
    assert set(panes) == set(PANE_KEYS)

    # Every rect is non-negative and inside the window.
    for name, rc in panes.items():
        assert rc.w >= 0 and rc.h >= 0, name
        assert rc.x >= 0 and rc.y >= 0, name
        assert rc.x + rc.w <= w and rc.y + rc.h <= h, name

    # No two panes overlap.
    rects = list(panes.values())
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _overlap(rects[i], rects[j]), (i, j)

    # Areas sum to the whole window -> disjoint + in-bounds => exact cover
    # (no gaps).
    assert sum(rc.w * rc.h for rc in rects) == w * h


def test_panes_adjacency_default():
    """The intended arrangement at the default size: rail | canvas | right
    column, banded by the top bar and status bar."""
    panes = compute_panes(1280, 920)
    top, rail = panes["top_bar"], panes["tool_rail"]
    canvas, pal = panes["canvas"], panes["palette"]
    insp, status = panes["inspector"], panes["status_bar"]

    assert top.x == 0 and top.y == 0 and top.w == 1280
    assert status.y + status.h == 920 and status.w == 1280
    # Middle band columns meet edge-to-edge.
    assert rail.x + rail.w == canvas.x
    assert canvas.x + canvas.w == pal.x
    assert pal.x == insp.x and pal.w == insp.w
    # Palette stacks directly on top of the inspector, same band as canvas.
    assert pal.y == canvas.y and pal.y + pal.h == insp.y
    assert insp.y + insp.h == canvas.y + canvas.h


def test_rect_contains_half_open():
    rc = Rect(10, 20, 30, 40)
    assert rc.contains(10, 20)                  # top-left inclusive
    assert not rc.contains(40, 20)              # right edge exclusive
    assert not rc.contains(10, 60)              # bottom edge exclusive
    assert rc.contains(39, 59)
    assert not rc.contains(9, 20) and not rc.contains(10, 19)


def test_pane_at_prefers_canvas():
    panes = compute_panes(1280, 920)
    canvas = panes["canvas"]
    # A point in the canvas resolves to canvas.
    assert pane_at(panes, canvas.x + 5, canvas.y + 5) == "canvas"
    # Points in the chrome resolve to their pane.
    assert pane_at(panes, 5, 5) == "top_bar"
    assert pane_at(panes, 5, panes["tool_rail"].y + 5) == "tool_rail"
    assert pane_at(panes, panes["palette"].x + 5,
                   panes["palette"].y + 5) == "palette"
    assert pane_at(panes, 5, 919) == "status_bar"


def test_canvas_transform_is_invertible():
    """world -> screen -> world is the identity (the two transforms are exact
    inverses); the canvas origin offset cancels."""
    canvas = compute_panes(1280, 920)["canvas"]
    cam_x, cam_y, zoom = 137.5, -42.25, 1.375
    for wx, wy in [(0.0, 0.0), (100.0, 200.0), (-30.0, 512.0), (999.9, 7.1)]:
        sx, sy = screen_from_world(canvas, cam_x, cam_y, zoom, wx, wy)
        bx, by = world_from_screen(canvas, cam_x, cam_y, zoom, sx, sy)
        assert bx == pytest.approx(wx, abs=1e-6)
        assert by == pytest.approx(wy, abs=1e-6)


def test_tile_screen_round_trip_in_canvas():
    """A tile maps to a screen point inside the canvas pane and back to the
    same tile — the editor's cursor pipeline (tile*ppt -> screen; screen ->
    art_px_to_tile) with the canvas offset applied."""
    canvas = compute_panes(1280, 920)["canvas"]
    ppt = 32.0                                   # preview px per tile
    grid_w, grid_h = 48, 32
    # Fit the whole grid in the canvas so every tested tile centre is visible.
    zoom, cam_x, cam_y = fit_camera(canvas, grid_w * ppt, grid_h * ppt)
    offset0 = (0.0, 0.0)
    for tx, ty in [(0, 0), (5, 9), (23, 14), (47, 31)]:
        # Tile centre -> world (art px) -> screen (canvas-relative).
        wx, wy = (tx + 0.5) * ppt, (ty + 0.5) * ppt
        sx, sy = screen_from_world(canvas, cam_x, cam_y, zoom, wx, wy)
        # The centre of an on-screen tile lands inside the canvas pane.
        assert canvas.contains(sx, sy)
        # Screen -> world -> fractional tile -> containing tile index.
        wbx, wby = world_from_screen(canvas, cam_x, cam_y, zoom, sx, sy)
        ftx, fty = art_px_to_tile(wbx, wby, offset0, (ppt, ppt))
        assert int(ftx) == tx and int(fty) == ty


def test_fit_camera_centres_world():
    canvas = compute_panes(1280, 920)["canvas"]
    world_w, world_h = 48 * 32.0, 32 * 32.0
    zoom, cam_x, cam_y = fit_camera(canvas, world_w, world_h)
    assert zoom > 0
    # World centre maps to the canvas centre.
    cx, cy = screen_from_world(canvas, cam_x, cam_y, zoom,
                               world_w / 2, world_h / 2)
    assert cx == pytest.approx(canvas.x + canvas.w / 2, abs=1e-6)
    assert cy == pytest.approx(canvas.y + canvas.h / 2, abs=1e-6)
    # The fit keeps the whole world within the canvas (margin <= 1).
    assert world_w * zoom <= canvas.w + 1e-6
    assert world_h * zoom <= canvas.h + 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
