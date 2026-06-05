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

    def __init__(self, grid_h: int, grid_w: int, tint=(180, 180, 200), max_alpha=255):
        self.h = grid_h
        self.w = grid_w
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        self.tint_r, self.tint_g, self.tint_b = tint
        self.max_alpha = max_alpha

    def update(self, field: np.ndarray) -> None:
        """field: (H, W) float in [0,1]. Pack to RGBA, upload.

        Smoke is drawn as a flat grey DENSITY medium: alpha is density-driven,
        the RGB tint is constant. The old ``light_modulation`` parameter (which
        multiplied the smoke colour by the local light to fake lit-smoke tint)
        is RETIRED — the god-ray glow (``GlowOverlay``, fed by the ray march's
        ``smoke_glow`` output) now provides lit-smoke shafts as an additive
        layer, one energy-conserving mechanism with no double-count (ch.03 C16,
        ch.05 §God-rays). Alpha is never modulated by light: smoke as a physical
        medium is always there; the glow overlay adds the colour it scatters.
        """
        # Pack as PREMULTIPLIED alpha so the overlay can be drawn with
        # BLEND_ALPHA_PREMULTIPLY (Porter-Duff "over"). Raylib's default
        # BLEND_ALPHA uses SRC_ALPHA for BOTH the colour AND alpha
        # channels, which means drawing semi-transparent smoke over an
        # opaque ship pixel REDUCES the destination alpha (e.g. ship
        # alpha 1.0 + smoke alpha 0.5 -> result alpha 0.75 instead of
        # 1.0). When the world RT is then blitted to screen, the lower
        # alpha lets the screen-fixed background bleed through what
        # should be opaque ship pixels — exactly the "galaxies through
        # the ship after a grenade" bug. PREMUL gets the correct
        # Porter-Duff alpha math: ship alpha 1.0 stays at 1.0.
        v = np.clip(field, 0.0, 1.0)
        alpha = v * self.max_alpha   # uint range, 0..255
        a_norm = alpha / 255.0       # 0..1 multiplier for premultiplication
        r = self.tint_r * a_norm
        g = self.tint_g * a_norm
        b = self.tint_b * a_norm
        self.packed[..., 0] = r.astype(np.uint8)
        self.packed[..., 1] = g.astype(np.uint8)
        self.packed[..., 2] = b.astype(np.uint8)
        self.packed[..., 3] = alpha.astype(np.uint8)
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
        # Slight color modulation by intensity (hotter = more white).
        # Fire is its own light source.
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


class GlowOverlay:
    """God-ray / lit-smoke glow overlay (ch.05 §God-rays).

    Draws the ray march's ``smoke_glow`` field — the RGB light the smoke
    *absorbed*, per channel — as an ADDITIVE volumetric shaft. This supersedes
    the retired ``light_modulation`` smoke surface-tint: a red beam through
    smoke casts a red shaft, energy-conserving by construction (the energy is
    exactly what the smoke removed from the ray). Additive blend raises RGB
    without touching destination alpha, so (unlike the alpha-blended smoke) it
    is NOT premultiplied (ch.05 §Blend discipline). Drawn before units so they
    occlude it in screen space; the march deposits no glow past opaque tiles,
    so shafts already terminate at walls.
    """

    def __init__(self, grid_h: int, grid_w: int, gain: float = 1.0):
        self.h = grid_h
        self.w = grid_w
        # `gain` scales the glow brightness before the 0..255 quantize — a
        # render-only knob (the deposit is energy-conserving and typically dim).
        self.gain = gain
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)

    def update(self, smoke_glow: np.ndarray) -> None:
        """smoke_glow: (H, W, 3) float — the absorbed-light god-ray field.

        Tone-map by simple clamp (ACES is the final-slice job) and pack into
        an RGBA texture with full alpha. Under BLEND_ADDITIVE (SRC_ALPHA, ONE)
        full alpha passes the RGB straight through as an additive contribution.
        """
        glow = np.clip(smoke_glow * self.gain, 0.0, 1.0)
        self.packed[..., 0] = (glow[..., 0] * 255.0).astype(np.uint8)
        self.packed[..., 1] = (glow[..., 1] * 255.0).astype(np.uint8)
        self.packed[..., 2] = (glow[..., 2] * 255.0).astype(np.uint8)
        self.packed[..., 3] = 255
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y), float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_blend_mode()


# ----------------------------------------------------------------------------
# Units, orders, HUD
# ----------------------------------------------------------------------------

def draw_unit(x_tile: float, y_tile: float, world_px_per_tile: float,
              color, label: str = "", radius_tiles: float = 1.5,
              footprint_tiles: int = 3,
              sprite: Optional[rl.Texture] = None,
              light_intensity: float = 1.0) -> None:
    """Draw a unit on its footprint, in world-pixel coordinates.

    x_tile, y_tile = top-left of the unit's footprint in world-tile coords.
    world_px_per_tile = how many world pixels per tile (set by WorldComposite).
    footprint_tiles = side length of the unit's tile footprint (3 for marines).
    radius_tiles = visual radius in tile units (used only for the circle fallback).
    sprite = optional Texture; if provided, draws sprite scaled to the footprint
             instead of the circle placeholder.
    light_intensity = scalar 0..1+ (clamped) used to tint the sprite. 1.0 = full
             brightness; 0.0 = black silhouette. Lets the unit respond to the
             room's lighting without needing a per-sprite normal map yet.
    """
    x_wpx = tile_to_world_px(x_tile, world_px_per_tile)
    y_wpx = tile_to_world_px(y_tile, world_px_per_tile)
    size_wpx = footprint_tiles * world_px_per_tile

    if sprite is not None:
        # Tint by local light. Clamp to [0, 1] so we never overdrive past white.
        L = max(0.0, min(1.0, float(light_intensity)))
        c = int(L * 255)
        tint = rl.Color(c, c, c, 255)
        src = rl.Rectangle(0.0, 0.0, float(sprite.width), float(sprite.height))
        dst = rl.Rectangle(x_wpx, y_wpx, size_wpx, size_wpx)
        rl.draw_texture_pro(sprite, src, dst, rl.Vector2(0.0, 0.0), 0.0, tint)
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
    "FieldOverlay", "FireOverlay", "GlowOverlay",
    "draw_unit", "draw_waypoint_line",
    "draw_grid", "draw_text", "draw_panel_background",
]
