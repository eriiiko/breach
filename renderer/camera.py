"""Camera2D — owns viewport position and zoom for the world renderer.

Coordinate spaces this module touches:
  - world tile  : (x_tile, y_tile)  integer tile coordinates in the world
  - world px    : (x_wpx,  y_wpx)   pixels in the world render target
  - screen px   : (x_spx,  y_spx)   pixels on the application window

The camera owns:
  - pos_tile  : top-left of viewport in world tile units (can be sub-tile via float)
  - zoom_px   : screen pixels per world tile (1.0 = world-px-per-tile is 1:1 with screen-px-per-tile)
  - viewport_px : map area pixels on the screen (panel excluded)
  - world_size_tile : the full world bounds (for clamping)

Pyray Camera2D + BeginMode2D were considered but rejected; we want explicit
world-tile units, our own screen<->world transforms, and easy multi-camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Camera2D:
    pos_tile_x: float
    pos_tile_y: float
    zoom_px_per_tile: float         # screen pixels per world tile
    viewport_px_w: int              # map area width in screen pixels
    viewport_px_h: int              # map area height in screen pixels
    world_size_tile_w: int
    world_size_tile_h: int
    # Where the camera's viewport sits on the screen (top-left). Defaults to
    # (0, 0) which is correct when the camera covers the whole map area.
    # Multi-camera (security cam / split screen / inset) sets these to the
    # anchor of the secondary viewport so mouse_to_tile works correctly.
    viewport_screen_x: int = 0
    viewport_screen_y: int = 0

    # ---- Visible region ----

    def visible_tiles(self) -> Tuple[float, float]:
        """How many world tiles fit horizontally / vertically in the viewport."""
        return (self.viewport_px_w / self.zoom_px_per_tile,
                self.viewport_px_h / self.zoom_px_per_tile)

    def contains_screen_px(self, x_spx: float, y_spx: float) -> bool:
        """Is this screen point inside this camera's viewport?"""
        return (self.viewport_screen_x <= x_spx
                < self.viewport_screen_x + self.viewport_px_w
                and self.viewport_screen_y <= y_spx
                < self.viewport_screen_y + self.viewport_px_h)

    def visible_world_rect_tile(self) -> Tuple[float, float, float, float]:
        """Visible region in tile units: (x, y, w, h). Used to compute the
        source rectangle for the final blit from the world RT to the screen."""
        w_tiles, h_tiles = self.visible_tiles()
        return (self.pos_tile_x, self.pos_tile_y, w_tiles, h_tiles)

    def visible_world_rect_world_px(self, world_px_per_tile: float
                                    ) -> Tuple[float, float, float, float]:
        """Visible region in world-pixel units (i.e. inside the world RT)."""
        x, y, w, h = self.visible_world_rect_tile()
        return (x * world_px_per_tile, y * world_px_per_tile,
                w * world_px_per_tile, h * world_px_per_tile)

    # ---- Coordinate conversions ----

    def world_tile_to_screen_px(self, x_tile: float, y_tile: float
                                ) -> Tuple[float, float]:
        """World tile -> screen pixel, accounting for the viewport anchor."""
        return ((x_tile - self.pos_tile_x) * self.zoom_px_per_tile
                + self.viewport_screen_x,
                (y_tile - self.pos_tile_y) * self.zoom_px_per_tile
                + self.viewport_screen_y)

    def screen_px_to_world_tile(self, x_spx: float, y_spx: float
                                ) -> Tuple[float, float]:
        """Screen pixel -> world tile, accounting for the viewport anchor.
        Caller should normally check contains_screen_px first."""
        return (self.pos_tile_x
                + (x_spx - self.viewport_screen_x) / self.zoom_px_per_tile,
                self.pos_tile_y
                + (y_spx - self.viewport_screen_y) / self.zoom_px_per_tile)

    # ---- Camera operations ----

    def pan(self, dx_tile: float, dy_tile: float) -> None:
        self.pos_tile_x += dx_tile
        self.pos_tile_y += dy_tile
        self.clamp_to_world()

    def set_zoom(self, zoom_px_per_tile: float) -> None:
        """Set zoom level. Hard floor of 1 spx/tile so we don't go crazy.
        Below min_zoom_to_fit_width the world becomes narrower than the
        viewport — the renderer should letterbox (black bars) rather than
        stretch the edge."""
        self.zoom_px_per_tile = max(1.0, zoom_px_per_tile)
        self.clamp_to_world()

    def clamp_to_world(self) -> None:
        """Keep the camera within the world bounds so we don't show garbage.

        If the viewport is larger than the world (e.g. tiny ship + huge zoom
        out), the camera locks at (0, 0). The caller is responsible for
        either (a) increasing zoom so the world fills the viewport, or
        (b) letterboxing the empty space. We do not stretch.
        """
        w_tiles, h_tiles = self.visible_tiles()
        max_x = max(0.0, self.world_size_tile_w - w_tiles)
        max_y = max(0.0, self.world_size_tile_h - h_tiles)
        self.pos_tile_x = max(0.0, min(max_x, self.pos_tile_x))
        self.pos_tile_y = max(0.0, min(max_y, self.pos_tile_y))

    def min_zoom_to_fit_width(self) -> float:
        """Smallest zoom that keeps the world wider than the viewport
        horizontally (so the user always scrolls left/right at the edges
        rather than seeing letterbox / edge-stretch)."""
        return self.viewport_px_w / max(1, self.world_size_tile_w)

    def min_zoom_to_fit_height(self) -> float:
        return self.viewport_px_h / max(1, self.world_size_tile_h)
