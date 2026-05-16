"""Coordinate-space naming discipline.

Three coordinate spaces exist in the renderer:
  - WORLD TILE  : (x_tile, y_tile)   — integer or float tile indices in the world
  - WORLD PX    : (x_wpx, y_wpx)     — pixels inside the world render target
  - SCREEN PX   : (x_spx, y_spx)     — pixels on the application window

Use the suffixes `_tile`, `_wpx`, `_spx` on every coordinate variable and
parameter. Mixing them is the source of bugs like "flashlight points at the
wrong tile after camera scrolls" — discipline at the variable-name level
catches the bulk of these.

The functions below are tiny — their value is consistency. Always import
and call them rather than writing the multiplication inline.
"""
from __future__ import annotations

from typing import Tuple


def tile_to_world_px(x_tile: float, world_px_per_tile: float) -> float:
    """Convert a world-tile coordinate to a world-pixel coordinate."""
    return x_tile * world_px_per_tile


def world_px_to_tile(x_wpx: float, world_px_per_tile: float) -> float:
    """Convert a world-pixel coordinate back to world-tile units."""
    return x_wpx / world_px_per_tile


def world_px_to_screen_px(x_wpx: float, camera_pos_tile_x: float,
                          world_px_per_tile: float,
                          zoom_px_per_tile: float) -> float:
    """Convert a world-pixel coordinate to a screen-pixel coordinate via the
    camera. zoom_px_per_tile is screen pixels per world tile.
    """
    x_tile = x_wpx / world_px_per_tile
    return (x_tile - camera_pos_tile_x) * zoom_px_per_tile


def screen_px_to_world_px(x_spx: float, camera_pos_tile_x: float,
                          world_px_per_tile: float,
                          zoom_px_per_tile: float) -> float:
    """Inverse of world_px_to_screen_px."""
    x_tile = camera_pos_tile_x + x_spx / zoom_px_per_tile
    return x_tile * world_px_per_tile


def tile_xy_to_world_px(x_tile: float, y_tile: float,
                        world_px_per_tile: float) -> Tuple[float, float]:
    """Vectorized convenience: (x_tile, y_tile) -> (x_wpx, y_wpx)."""
    return (tile_to_world_px(x_tile, world_px_per_tile),
            tile_to_world_px(y_tile, world_px_per_tile))
