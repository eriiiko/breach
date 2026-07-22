"""tools/editor_layout.py — pane layout for the map editor (Arc C, editor doc §8).

Pure geometry, no raylib import: given a window size, compute the six pane
rects that TILE THE WINDOW EXACTLY (no overlap, no gaps) —

    top bar      : full width, top
    tool rail    : left column of the middle band
    canvas       : centre of the middle band (where the map draws)
    palette      : upper right column (tabbed palette — §8)
    inspector    : lower right column
    status bar   : full width, bottom

plus the canvas viewport transforms (world<->screen) so the map draws INSIDE
the canvas pane instead of the whole window. The transform is the old
``screen = (world - cam) * zoom`` shifted by the canvas pane origin.

This module owns only WHERE the panes are and the canvas transform; what each
pane *contains* stays in map_editor.py and is extended by later Arc C patches
(C1 fills the palette + inspector from the entity registry). Unit-tested in
tests/test_editor_layout.py — headless, no window needed.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pane thicknesses (px). The canvas takes whatever the fixed chrome leaves;
# every dimension is clamped in compute_panes so a tiny window still tiles.
# ---------------------------------------------------------------------------
TOP_BAR_H = 44
STATUS_BAR_H = 28
TOOL_RAIL_W = 132
RIGHT_COL_W = 240
PALETTE_MIN_H = 132      # palette floor; it takes up to half the middle band

PANE_KEYS = ("top_bar", "tool_rail", "canvas", "palette", "inspector",
             "status_bar")


@dataclass(frozen=True)
class Rect:
    """An axis-aligned integer rectangle (top-left origin, like the screen)."""
    x: int
    y: int
    w: int
    h: int

    def contains(self, px, py) -> bool:
        """True when the point (px, py) lies inside this rect (half-open on
        the right/bottom edges, so adjacent panes never both claim a pixel)."""
        return (self.x <= px < self.x + self.w
                and self.y <= py < self.y + self.h)


def compute_panes(win_w, win_h) -> dict:
    """The six pane rects for a ``win_w`` x ``win_h`` window, keyed by
    :data:`PANE_KEYS`. They tile the window exactly: the union is the whole
    window and no two rects overlap. Every dimension is clamped to be >= 0 so
    a window smaller than the chrome still produces a valid (possibly
    zero-size) tiling instead of negative rects. The canvas is derived by
    subtraction, which is what keeps the tiling exact."""
    W, H = int(win_w), int(win_h)

    top_h = min(TOP_BAR_H, max(0, H))
    status_h = min(STATUS_BAR_H, max(0, H - top_h))
    mid_y = top_h
    mid_h = max(0, H - top_h - status_h)

    rail_w = min(TOOL_RAIL_W, max(0, W))
    right_w = min(RIGHT_COL_W, max(0, W - rail_w))
    canvas_w = max(0, W - rail_w - right_w)

    # Right column splits into palette (top) + inspector (bottom).
    pal_h = min(mid_h, max(PALETTE_MIN_H, mid_h // 2)) if mid_h else 0
    insp_h = mid_h - pal_h

    return {
        "top_bar": Rect(0, 0, W, top_h),
        "tool_rail": Rect(0, mid_y, rail_w, mid_h),
        "canvas": Rect(rail_w, mid_y, canvas_w, mid_h),
        "palette": Rect(W - right_w, mid_y, right_w, pal_h),
        "inspector": Rect(W - right_w, mid_y + pal_h, right_w, insp_h),
        "status_bar": Rect(0, H - status_h, W, status_h),
    }


def pane_at(panes: dict, px, py):
    """Name of the pane containing the point (px, py), or None. Canvas is
    tested first so a click on the shared canvas/chrome seam edits the map."""
    if panes["canvas"].contains(px, py):
        return "canvas"
    for name in PANE_KEYS:
        if name != "canvas" and panes[name].contains(px, py):
            return name
    return None


# ---------------------------------------------------------------------------
# Canvas viewport transform — world (art-pixel) <-> screen, offset by the
# canvas pane origin. ``cam`` is the world coordinate shown at the canvas
# top-left; ``zoom`` is screen-px per world-px. These are the exact inverse of
# each other (round-trip is the layout gate).
# ---------------------------------------------------------------------------

def screen_from_world(canvas: Rect, cam_x, cam_y, zoom, wx, wy) -> tuple:
    """World (art-pixel) point -> screen point, inside the canvas pane."""
    return (canvas.x + (float(wx) - float(cam_x)) * float(zoom),
            canvas.y + (float(wy) - float(cam_y)) * float(zoom))


def world_from_screen(canvas: Rect, cam_x, cam_y, zoom, sx, sy) -> tuple:
    """Screen point -> world (art-pixel) point; inverse of
    :func:`screen_from_world`."""
    z = float(zoom)
    return (float(cam_x) + (float(sx) - canvas.x) / z,
            float(cam_y) + (float(sy) - canvas.y) / z)


def fit_camera(canvas: Rect, world_w, world_h, margin: float = 0.95) -> tuple:
    """Initial (zoom, cam_x, cam_y) that centres a ``world_w`` x ``world_h``
    world inside the canvas pane at ``margin`` of the tight fit. Mirrors the
    old full-window fit, now scoped to the canvas rect."""
    world_w, world_h = float(world_w), float(world_h)
    cw = max(1.0, float(canvas.w))
    ch = max(1.0, float(canvas.h))
    zoom = min(cw / world_w, ch / world_h) * float(margin)
    cam_x = (world_w - cw / zoom) / 2.0
    cam_y = (world_h - ch / zoom) / 2.0
    return zoom, cam_x, cam_y
