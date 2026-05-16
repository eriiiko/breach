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

    def update(self, field: np.ndarray) -> None:
        """field: (H, W) float in [0,1]. Pack to RGBA, upload."""
        v = np.clip(field, 0.0, 1.0)
        self.packed[..., 0] = self.tint_r
        self.packed[..., 1] = self.tint_g
        self.packed[..., 2] = self.tint_b
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

    def update(self, fire: np.ndarray) -> None:
        # Slight color modulation by intensity (hotter = more white)
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

def draw_unit(fx: float, fy: float, ft: float, color, label: str = "", radius_tiles: float = 1.5) -> None:
    """Draw a unit at integer tile position (fx, fy) with radius in tile units.
    ft is fine_tile_px (pixels per tile in the rendered map area).
    """
    cx = (fx + 1.5) * ft  # 3x3 footprint, center at +1.5 tiles
    cy = (fy + 1.5) * ft
    r = radius_tiles * ft
    rl.draw_circle(int(cx), int(cy), r, rl.Color(*color))
    if label:
        rl.draw_text(label, int(cx - r), int(cy - r - 14), 12, rl.WHITE)


def draw_waypoint_line(p1, p2, ft: float, color=(60, 200, 255, 200)) -> None:
    """p1, p2 are (fx, fy) tile coords. Draws a line between them in map space."""
    x1 = (p1[0] + 1.5) * ft
    y1 = (p1[1] + 1.5) * ft
    x2 = (p2[0] + 1.5) * ft
    y2 = (p2[1] + 1.5) * ft
    rl.draw_line_ex(rl.Vector2(x1, y1), rl.Vector2(x2, y2), 2.0, rl.Color(*color))


def draw_grid(grid_w: int, grid_h: int, ft: float, color=(80, 80, 100, 60), step: int = 3) -> None:
    """Faint grid overlay at every `step` tiles (default coarse=3)."""
    color_obj = rl.Color(*color)
    px_w = grid_w * ft
    px_h = grid_h * ft
    for x in range(0, grid_w + 1, step):
        xp = x * ft
        rl.draw_line_ex(rl.Vector2(xp, 0), rl.Vector2(xp, px_h), 1.0, color_obj)
    for y in range(0, grid_h + 1, step):
        yp = y * ft
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
