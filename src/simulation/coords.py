"""Coordinate helpers for the physics-tile grid.

There is ONE grid — the physics-tile grid. A position is an (x, y) pair in
tile units. The integer form (row, col) for matrix indexing comes from
flooring; the meter form comes from multiplying by ``level.tile_size_m``.
These helpers exist so the conversions are explicit and centralized — if
the physics resolution (``tile_size_m``) changes, only meter-conversion
call sites are affected. Everything in tile units stays stable.
"""
from __future__ import annotations
from typing import Tuple


def tile_to_index(x: float, y: float) -> Tuple[int, int]:
    """(x, y) in tile units → (row, col) for matrix indexing.

    Returns (row, col) — row first because numpy is row-major. Floors,
    not rounds: tile (3.7, 2.1) belongs to cell (row=2, col=3).
    """
    return int(y), int(x)


def index_to_tile(row: int, col: int) -> Tuple[float, float]:
    """(row, col) matrix index → tile-unit coords at the cell's top-left."""
    return float(col), float(row)


def tile_to_meters(x: float, y: float, tile_size_m: float) -> Tuple[float, float]:
    """Tile-unit coords → meters."""
    return x * tile_size_m, y * tile_size_m


def meters_to_tile(mx: float, my: float, tile_size_m: float) -> Tuple[float, float]:
    """Meters → tile-unit coords."""
    return mx / tile_size_m, my / tile_size_m


def tile_distance_m(x1: float, y1: float, x2: float, y2: float,
                    tile_size_m: float) -> float:
    """Euclidean distance between two tile-unit positions, in meters."""
    dx = (x2 - x1) * tile_size_m
    dy = (y2 - y1) * tile_size_m
    return (dx * dx + dy * dy) ** 0.5
