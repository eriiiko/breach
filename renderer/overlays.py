"""Overlays: smoke, fire, units, orders, debug HUDs.

These are drawn on top of the lit ship layer. Most use simple rectangle/line
draws via pyray. Smoke and fire are uploaded as dynamic RGBA textures at
physics resolution and drawn stretched over the map area.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pyray as rl

from . import core
from .coords import tile_to_world_px


# ----------------------------------------------------------------------------
# Smoke + Fire overlays (physics-resolution textures stretched to map area)
# ----------------------------------------------------------------------------

class FieldOverlay:
    """Holds a dynamic RGBA texture for a scalar physics field.

    Use for smoke (gray semi-transparent) and fire (orange glow).
    """

    def __init__(self, grid_h: int, grid_w: int, tint=(180, 180, 200), max_alpha=200):
        self.h = grid_h
        self.w = grid_w
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        self.tint_r, self.tint_g, self.tint_b = tint
        self.max_alpha = max_alpha

    def update(self, field: np.ndarray, light_modulation: Optional[np.ndarray] = None) -> None:
        """field: (H, W) float in [0,1]. Pack to RGBA, upload.

        If light_modulation is provided (also (H,W) float in [0,1]), alpha is
        multiplied by it — smoke only visible where light reaches it.
        """
        v = np.clip(field, 0.0, 1.0)
        self.packed[..., 0] = self.tint_r
        self.packed[..., 1] = self.tint_g
        self.packed[..., 2] = self.tint_b
        if light_modulation is not None:
            mod = np.clip(light_modulation, 0.0, 1.0)
            # Add a small ambient floor so smoke isn't completely invisible
            mod = 0.15 + 0.85 * mod
            self.packed[..., 3] = (v * mod * self.max_alpha).astype(np.uint8)
        else:
            self.packed[..., 3] = (v * self.max_alpha).astype(np.uint8)
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y), float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)


class FireOverlay(FieldOverlay):
    """Fire-specific: orange/yellow tint, additive blend."""

    def __init__(self, grid_h: int, grid_w: int):
        super().__init__(grid_h, grid_w, tint=(255, 140, 30), max_alpha=220)

    def update(self, fire: np.ndarray, light_modulation: Optional[np.ndarray] = None) -> None:
        # Slight color modulation by intensity (hotter = more white).
        # Fire is its own light source — ignore the light_modulation argument.
        v = np.clip(fire, 0.0, 1.0)
        self.packed[..., 0] = 255
        self.packed[..., 1] = (140 + (255 - 140) * v * 0.5).astype(np.uint8)
        self.packed[..., 2] = (30 + (180 - 30) * v * 0.5).astype(np.uint8)
        self.packed[..., 3] = (v * self.max_alpha).astype(np.uint8)
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
        super().draw(dst_x, dst_y, dst_w, dst_h)
        rl.end_blend_mode()


# ----------------------------------------------------------------------------
# Units, orders, HUD
# ----------------------------------------------------------------------------

def draw_unit(x_tile: float, y_tile: float, world_px_per_tile: float,
              color, label: str = "", radius_tiles: float = 1.5,
              footprint_tiles: int = 3,
              sprite: Optional[rl.Texture] = None) -> None:
    """Draw a unit on its footprint, in world-pixel coordinates.

    x_tile, y_tile = top-left of the unit's footprint in world-tile coords.
    world_px_per_tile = how many world pixels per tile (set by WorldComposite).
    footprint_tiles = side length of the unit's tile footprint (3 for marines).
    radius_tiles = visual radius in tile units (used only for the circle fallback).
    sprite = optional Texture; if provided, draws sprite scaled to the footprint
             instead of the circle placeholder.
    """
    x_wpx = tile_to_world_px(x_tile, world_px_per_tile)
    y_wpx = tile_to_world_px(y_tile, world_px_per_tile)
    size_wpx = footprint_tiles * world_px_per_tile

    if sprite is not None:
        src = rl.Rectangle(0.0, 0.0, float(sprite.width), float(sprite.height))
        dst = rl.Rectangle(x_wpx, y_wpx, size_wpx, size_wpx)
        rl.draw_texture_pro(sprite, src, dst, rl.Vector2(0.0, 0.0), 0.0, rl.WHITE)
    else:
        # Circle fallback — also used when sprite failed to load.
        half = footprint_tiles * 0.5
        cx_wpx = x_wpx + half * world_px_per_tile
        cy_wpx = y_wpx + half * world_px_per_tile
        r_wpx  = radius_tiles * world_px_per_tile
        rl.draw_circle(int(cx_wpx), int(cy_wpx), r_wpx, rl.Color(*color))

    if label:
        r_wpx = radius_tiles * world_px_per_tile
        cx_wpx = x_wpx + (footprint_tiles * 0.5) * world_px_per_tile
        cy_wpx = y_wpx + (footprint_tiles * 0.5) * world_px_per_tile
        rl.draw_text(label, int(cx_wpx - r_wpx),
                     int(cy_wpx - r_wpx - 14), 12, rl.WHITE)


def draw_waypoint_line(p1_tile, p2_tile, world_px_per_tile: float,
                       color=(60, 200, 255, 200), unit_footprint_tiles: int = 3
                       ) -> None:
    """Draw a line between two waypoints in world-pixel coordinates.
    p1, p2 are (x_tile, y_tile). The line is drawn through the centers of
    the unit's footprint at each waypoint."""
    half = unit_footprint_tiles * 0.5
    x1_wpx = tile_to_world_px(p1_tile[0] + half, world_px_per_tile)
    y1_wpx = tile_to_world_px(p1_tile[1] + half, world_px_per_tile)
    x2_wpx = tile_to_world_px(p2_tile[0] + half, world_px_per_tile)
    y2_wpx = tile_to_world_px(p2_tile[1] + half, world_px_per_tile)
    rl.draw_line_ex(rl.Vector2(x1_wpx, y1_wpx),
                    rl.Vector2(x2_wpx, y2_wpx),
                    2.0, rl.Color(*color))


def draw_grid(grid_w_tile: int, grid_h_tile: int, world_px_per_tile: float,
              color=(80, 80, 100, 60), step: int = 3) -> None:
    """Faint grid overlay at every `step` tiles, drawn in world pixels."""
    color_obj = rl.Color(*color)
    px_w = grid_w_tile * world_px_per_tile
    px_h = grid_h_tile * world_px_per_tile
    for x_tile in range(0, grid_w_tile + 1, step):
        xp = tile_to_world_px(x_tile, world_px_per_tile)
        rl.draw_line_ex(rl.Vector2(xp, 0), rl.Vector2(xp, px_h), 1.0, color_obj)
    for y_tile in range(0, grid_h_tile + 1, step):
        yp = tile_to_world_px(y_tile, world_px_per_tile)
        rl.draw_line_ex(rl.Vector2(0, yp), rl.Vector2(px_w, yp), 1.0, color_obj)


def draw_text(text: str, x: int, y: int, size: int = 16, color=(220, 220, 220, 255)) -> None:
    rl.draw_text(text, x, y, size, rl.Color(*color))


def draw_panel_background(x: int, y: int, w: int, h: int, color=(20, 20, 28, 240)) -> None:
    rl.draw_rectangle(x, y, w, h, rl.Color(*color))
    rl.draw_line_ex(rl.Vector2(x, y), rl.Vector2(x, y + h), 2.0, rl.Color(120, 120, 140, 255))


__all__ = [
    "FieldOverlay", "FireOverlay",
    "draw_unit", "draw_waypoint_line",
    "draw_grid", "draw_text", "draw_panel_background",
]
