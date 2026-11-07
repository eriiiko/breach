"""THE blast's expansion source (#31, Erik 2026-11-06): instead of adding gas, a
blast tells the pressure solve that the open tiles around it are expanding. The
solve then finds the wind that fits the room, so the air is delivered around
walls and through doors, and no mass is added. Design:
docs/explosion_design_2026-11-06.md.

The method is a divergence source (Feldman, O'Brien & Arikan 2003; Nielsen et al.
2019; FDS after Rehm & Baum): ``div(u) = S`` in place of ``div(u) = 0``, entered
in EOSSolver::step by subtracting S from ``div_u_`` before the multigrid build.
It conserves N and E by construction, because only the velocity changes and
energy still moves through the face fluxes. docs/research/
explosion_litsearch_2026-11-06.md section 1. Pair it with heat, or the gas springs
back when the source stops.

This module owns three things:

- THE WALK, ``reach_tiles``: what a blast touches, 4-neighbour along open floor on
  the solve's own mask (``open_mask``), so a wall stops it and a gap does not, and
  no share of a fixed total lands where the solve cannot use it. It runs once, at
  detonation, after the blast's wall damage, and never per tick.
- THE SOURCE PLANE, ``GameMap.div_source``: (h, w) int32, Q16 1/s, the pending
  source, written only here and consumed by ``take_source_plane`` at the head of
  the physics step, so a detonation after the solve leaves it for the next tick
  beside its queued heat and smoke. It is synced state (tests/field_digest.py).
- THE [blast] DOOR, ``blast_settings``: what one strength unit means, refused by
  name.

A source accumulates in Python ints and saturates at ``S_MAX_Q``, counted: S
depends on the walk, which is level geometry, so no row door can bound it.

``expansion_m3`` / ``expansion_radius`` / ``expansion_ticks`` rows are the
comparison rows of levels/blast_quad and blast_div. Their first tick goes through
the same plane; a row of more than one tick keeps its later ticks in
``GameMap.expansions``, a plain list, never digested. No shipped row uses it, and
multi-tick delivery is a future special explosive (#174).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from config import CFG
from simulation import gas_fixed

# The largest source a tile may hold, 1/s. Above the largest flat source a row can
# ask for on open floor (strength just under 4 on its one tile, 13 800 1/s at
# 0.333 m tiles), below the int32 Q16 ceiling of 32 768 1/s that 12 strength on
# one tile would wrap. The right-hand side it feeds is int64 and takes it.
S_MAX = 16384.0
S_MAX_Q = gas_fixed.quantize_q16_scalar(S_MAX)


@dataclass(frozen=True)
class BlastSettings:
    """``[blast]``: what one strength unit means. ALL PLACEHOLDERS (Erik tunes)."""
    expansion_m3_per_strength: float
    heat_mj_per_strength: float
    grenade_kg: float


def blast_settings(rows=None) -> BlastSettings:
    """The door: refuses a missing key or a non-positive value by name."""
    rows = getattr(CFG, "blast", None) if rows is None else rows
    if rows is None:
        raise ValueError("config.toml has no [blast] table")
    vals = {}
    for key in ("expansion_m3_per_strength", "heat_mj_per_strength",
                "grenade_kg"):
        v = rows.get(key) if isinstance(rows, dict) else getattr(rows, key, None)
        if v is None:
            raise ValueError(f"[blast] {key} is missing")
        v = float(v)
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"[blast] {key} = {v!r} must be > 0")
        vals[key] = v
    return BlastSettings(**vals)


def source_steps(strength: float) -> int:
    """``r_s``: integer steps, so strength 1 up to just under 4 stays on one tile
    and a 1 % change of strength never moves S 5-fold."""
    return max(1, int(math.floor(math.sqrt(float(strength)))))


def open_mask(gmap):
    """The tiles the pressure solve acts on: never solid, vacuum or the ambient
    ring. EOSSolver::step's own skip (``is_ambient`` is all-False on a space map).
    Passable fuel is in it: the solve reads its partial permeability."""
    return ~(gmap.solid | gmap.is_vacuum | gmap.is_ambient)


def walk_seeds(mask, fy, fx, approach=None):
    """Where a walk from (fy, fx) starts: the centre when it is open. A charge on
    a tile still solid after its wall damage (a round stops ON the wall, a door
    charge on a door that held) delivers on the side it came from: ``approach``
    is that tile, or any tile on that side (the planter): the tile itself when it
    touches the centre and is open, else the open 4-neighbour nearest it. With no
    known side, every open 4-neighbour in sorted order."""
    h, w = mask.shape
    if 0 <= fy < h and 0 <= fx < w and mask[fy, fx]:
        return [(fy, fx)]
    near = sorted((y, x) for y, x in ((fy - 1, fx), (fy, fx - 1), (fy, fx + 1),
                                      (fy + 1, fx))
                  if 0 <= y < h and 0 <= x < w and mask[y, x])
    if approach is None or not near:
        return near
    ay, ax = int(approach[0]), int(approach[1])
    if (max(abs(ay - fy), abs(ax - fx)) == 1 and 0 <= ay < h and 0 <= ax < w
            and mask[ay, ax]):
        return [(ay, ax)]
    return [min(near, key=lambda t: (t[0] - ay) ** 2 + (t[1] - ax) ** 2)]


def _walk(mask, fy, fx, r2, approach=None):
    """The one breadth-first walk behind :func:`reach_tiles` and
    :func:`reach_depth`: ``{(y, x): hops}``, the seeds at 0, each 4-neighbour
    step one more, along ``mask`` and under ``r2`` from (fy, fx). Pure integer."""
    h, w = mask.shape
    seeds = walk_seeds(mask, fy, fx, approach)
    depth = {t: 0 for t in seeds}
    todo = deque(seeds)
    while todo:
        y, x = todo.popleft()
        d = depth[(y, x)] + 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= ny < h and 0 <= nx < w) or (ny, nx) in depth:
                continue
            if not mask[ny, nx] or (ny - fy) ** 2 + (nx - fx) ** 2 >= r2:
                continue
            depth[(ny, nx)] = d
            todo.append((ny, nx))
    return depth


def reach_tiles(mask, fy, fx, r2, approach=None):
    """THE walk: the open tiles with squared distance from (fy, fx) under ``r2``,
    walking 4-neighbour along ``mask``, row-major. A wall stops it; a door gap
    does not. A seed is always in the result, whatever its distance: a charge
    on a wall still delivers beside it."""
    return sorted(_walk(mask, fy, fx, r2, approach))


def reach_depth(mask, fy, fx, r2, approach=None):
    """THE walk with its hop count (G4, #75): ``{(y, x): depth}`` over exactly
    :func:`reach_tiles`' tiles, the seeds at depth 0 and each 4-neighbour step
    one more, so a tile reached through a door gap is as far as the way round.
    Read by the renderer's shock front (the geodesic distance it rides); it
    changes nothing in the sim."""
    return _walk(mask, fy, fx, r2, approach)


def tile_volume_m3(gmap) -> float:
    """One tile of air: the level's tile, the deck's height."""
    return float(gmap.tile_size_m) ** 2 * float(CFG.physics.water.ceiling_h)


def heat_units_per_mj(gmap) -> float:
    """The `heat` plane's units in one MJ on this level. A deposit on air lands
    ``heat / (max(N, n_floor_heat) * c_v)`` degrees and one degree of a tile of
    ambient air is ``c_v * THERMAL_MASS_UNIT`` J/(m3.K) of its volume, so c_v
    cancels and one heat unit is ``THERMAL_MASS_UNIT * V_tile`` joules. Below
    n_floor_heat the solver lands less; that is the solver's law, not this one."""
    from simulation.materials import THERMAL_MASS_UNIT
    return 1.0e6 / (THERMAL_MASS_UNIT * tile_volume_m3(gmap))


def _source_q(gmap, expansion_m3, ticks, n_tiles) -> int:
    """The flat per-tile source of ``expansion_m3`` over ``ticks`` ticks, Q16 1/s,
    as a Python int: the walk can make it wider than int32."""
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return gas_fixed.quantize_q16_scalar(
        expansion_m3 / (ticks * n_tiles * tile_volume_m3(gmap) * dt))


def add_source(gmap, tiles, s_q) -> None:
    """Add ``s_q`` to the pending source on ``tiles``, saturating at ``S_MAX_Q``:
    a cut is counted (``div_source_sat_hits``, and the cut Q16 in
    ``div_source_cut_q``), never a wrap."""
    plane = gmap.div_source
    for y, x in tiles:
        v = int(plane[y, x]) + s_q
        if v > S_MAX_Q:
            gmap.div_source_sat_hits += 1
            gmap.div_source_cut_q += v - S_MAX_Q
            v = S_MAX_Q
        plane[y, x] = v


def start_source(gmap, tiles, expansion_m3) -> None:
    """A strength row's one-tick source, flat over its walked ``tiles``."""
    if tiles:
        add_source(gmap, tiles, _source_q(gmap, expansion_m3, 1, len(tiles)))


def start_expansion(gmap, fy, fx, payload) -> None:
    """A comparison row's source (the module docstring): radius in tiles along
    open floor, first tick on the plane, later ticks in ``gmap.expansions``."""
    r2 = int(math.ceil(float(payload.expansion_radius) ** 2))
    tiles = reach_tiles(open_mask(gmap), fy, fx, r2)
    if not tiles:
        return
    ticks = max(int(payload.expansion_ticks), 1)
    s_q = _source_q(gmap, payload.expansion_m3, ticks, len(tiles))
    add_source(gmap, tiles, s_q)
    if ticks > 1:
        gmap.expansions.append([tiles, s_q, ticks - 1])


def take_source_plane(gmap):
    """This tick's (h, w) int32 Q16 source in 1/s, or None; the plane is zeroed.
    Called once per tick, before the solve. A comparison row's next tick lands on
    the emptied plane here, so it waits as pending state like any other."""
    plane = None
    if gmap.div_source.any():
        plane = gmap.div_source.copy()
        gmap.div_source[...] = 0
    if gmap.expansions:
        for e in gmap.expansions:
            add_source(gmap, e[0], e[1])
            e[2] -= 1
        gmap.expansions[:] = [e for e in gmap.expansions if e[2] > 0]
    return plane


__all__ = ["BlastSettings", "S_MAX", "S_MAX_Q", "add_source", "blast_settings",
           "heat_units_per_mj", "open_mask", "reach_depth", "reach_tiles",
           "source_steps",
           "start_expansion", "start_source", "take_source_plane",
           "tile_volume_m3", "walk_seeds"]
