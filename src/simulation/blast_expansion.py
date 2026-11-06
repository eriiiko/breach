"""THE blast's expansion source (#31 PROTOTYPE, Erik 2026-11-06): instead of
adding gas, a blast tells the pressure solve that the open tiles around it are
expanding. The solve then finds the wind that fits the room, so the air is
delivered around walls and through doors, and no mass is added.

The method is a divergence source (Feldman, O'Brien & Arikan 2003; Nielsen et al.
2019; FDS after Rehm & Baum): ``div(u) = S`` in place of ``div(u) = 0``, entered
in EOSSolver::step by subtracting S from ``div_u_`` before the multigrid build.
It conserves N and E by construction, because only the velocity changes and
energy still moves through the face fluxes. docs/research/
explosion_litsearch_2026-11-06.md section 1.

A payload names the air it pushes out (``expansion_m3``), the reach of the source
(``expansion_radius``, tiles along open floor, so a wall stops it), and how many
ticks it takes (``expansion_ticks``). A row pairs this with heat, or the gas
springs back when the source stops.

PROTOTYPE LIMITS, for Erik's eye and not for merging as is:
- CPU only: the CUDA and resident paths do not read the source.
- The active list is plain Python, never digested. A re-simulation repeats it,
  but a mid-blast snapshot would lose it.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from config import CFG
from simulation import gas_fixed


def reach_tiles(solid, fy, fx, radius):
    """Open tiles within ``radius`` steps of (fy, fx), walking 4-neighbour along
    open floor, in a fixed order. A wall stops it; a door gap does not."""
    h, w = solid.shape
    if solid[fy, fx]:
        return []
    seen = {(fy, fx): 0}
    todo = deque([(fy, fx)])
    r2 = float(radius) * float(radius)
    while todo:
        y, x = todo.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= ny < h and 0 <= nx < w) or (ny, nx) in seen:
                continue
            if solid[ny, nx] or (ny - fy) ** 2 + (nx - fx) ** 2 >= r2:
                continue
            seen[(ny, nx)] = seen[(y, x)] + 1
            todo.append((ny, nx))
    return list(seen)


def start_expansion(gmap, fy, fx, payload):
    """Queue this payload's source on the map; the tick loop takes it from here."""
    tiles = reach_tiles(gmap.solid, fy, fx, payload.expansion_radius)
    if not tiles:
        return
    ticks = max(int(payload.expansion_ticks), 1)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    v_tile = float(gmap.tile_size_m) ** 2 * float(CFG.physics.water.ceiling_h)
    s = payload.expansion_m3 / (ticks * len(tiles) * v_tile * dt)
    gmap.expansions.append([tiles, gas_fixed.quantize_q16_scalar(s), ticks])


def take_source_plane(gmap):
    """This tick's (h, w) int32 Q16 source in 1/s, or None. Called once per tick,
    before the solve."""
    if not gmap.expansions:
        return None
    plane = np.zeros(gmap.solid.shape, dtype=np.int32)
    for e in gmap.expansions:
        tiles, s_q, _left = e
        for y, x in tiles:
            plane[y, x] += s_q
        e[2] -= 1
    gmap.expansions[:] = [e for e in gmap.expansions if e[2] > 0]
    return plane


__all__ = ["reach_tiles", "start_expansion", "take_source_plane"]
