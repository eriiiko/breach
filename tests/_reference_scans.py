"""tests/_reference_scans.py -- the ORACLES of issue #143's parity tests.

Today's two pure-Python hot spots, moved here VERBATIM (names aside) the day
they were rewritten for speed: the pre-#143 ``pathfinding.astar`` and the
pre-#143 ``GameMap.find_burst_walls`` loop (as a free function over a gmap).
They are frozen on purpose -- the optimised code in ``pathfinding.py`` and
``gamemap.py`` must equal them bit for bit (the path list element for element,
the burst list in order), because the game's output may not move by a bit.

Do NOT edit these bodies to follow a change of the real code; a deliberate
behaviour change of either law is a new oracle with its own written rationale.
Harness module (``_`` prefix): never collected.
"""
from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

REF_FINE_W = 120
REF_FINE_H = 75
REF_UNIT_SIZE = 3
REF_NODE_LIMIT = 50_000   # tests monkeypatch this beside pathfinding.NODE_LIMIT


# 8 directions: (dx, dy, is_diagonal)
_REF_DIRECTIONS = [
    ( 1,  0, False),
    (-1,  0, False),
    ( 0,  1, False),
    ( 0, -1, False),
    ( 1,  1, True),
    ( 1, -1, True),
    (-1,  1, True),
    (-1, -1, True),
]


def _ref_in_bounds(x: int, y: int, map_w: int, map_h: int) -> bool:
    """Check that a 3x3 unit block starting at (x, y) fits within the map."""
    return 0 <= x <= map_w - REF_UNIT_SIZE and 0 <= y <= map_h - REF_UNIT_SIZE


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def _ref_heuristic(x0: int, y0: int, x1: int, y1: int) -> float:
    """
    Octile distance using D&D 3.5 alternating diagonal costs.

    Average diagonal cost is 1.5, so:
        h = max(dx, dy) + 0.5 * min(dx, dy)
    which equals straight + 0.5 * diag.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    diag = min(dx, dy)
    straight = max(dx, dy) - diag
    return straight + diag * 1.5


# ---------------------------------------------------------------------------
# Standard A* (used by zombies and player units)
# ---------------------------------------------------------------------------

def astar_reference(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    is_blocked_fn: Callable[[int, int], bool],
    map_w: int = REF_FINE_W,
    map_h: int = REF_FINE_H,
) -> List[Tuple[int, int]]:
    """
    Standard A* pathfinding on the fine tile grid.

    Args:
        start_x, start_y: Top-left corner of the unit's 3x3 block at the start.
        goal_x, goal_y: Top-left corner of the unit's 3x3 block at the goal.
        is_blocked_fn: Callable(x, y) -> bool. Returns True if a 3x3 block
            placed at (x, y) is impassable.
        map_w: Fine grid width (default 120).
        map_h: Fine grid height (default 75).

    Returns:
        List of (x, y) tile positions from start to goal (inclusive), or
        empty list if no path exists.

    Notes:
        Diagonal cost alternates 1-2 (D&D 3.5 style). A boolean flag flips
        each diagonal step: first diagonal costs 1, second costs 2, etc.
        Node expansion is capped at REF_NODE_LIMIT to prevent runaway searches.
    """
    if not _ref_in_bounds(start_x, start_y, map_w, map_h):
        return []
    if not _ref_in_bounds(goal_x, goal_y, map_w, map_h):
        return []
    if is_blocked_fn(start_x, start_y) or is_blocked_fn(goal_x, goal_y):
        return []
    if (start_x, start_y) == (goal_x, goal_y):
        return [(start_x, start_y)]

    # State: (x, y, diag_flag)  — diag_flag True means next diagonal costs 1
    start_state = (start_x, start_y, True)
    h = _ref_heuristic(start_x, start_y, goal_x, goal_y)

    # Priority queue entries: (f, tie_breaker, g, x, y, diag_flag)
    counter = 0
    open_heap: list = []
    heapq.heappush(open_heap, (h, counter, 0.0, start_x, start_y, True))
    counter += 1

    # Best g-cost seen for each (x, y, diag_flag)
    best_g: Dict[Tuple[int, int, bool], float] = {start_state: 0.0}

    # Parent map for path reconstruction
    came_from: Dict[Tuple[int, int, bool], Optional[Tuple[int, int, bool]]] = {
        start_state: None
    }

    expanded = 0

    while open_heap:
        f, _, g, x, y, diag_flag = heapq.heappop(open_heap)

        # Goal check (either diag_flag state)
        if x == goal_x and y == goal_y:
            # Reconstruct path
            path: List[Tuple[int, int]] = []
            state: Optional[Tuple[int, int, bool]] = (x, y, diag_flag)
            while state is not None:
                path.append((state[0], state[1]))
                state = came_from[state]
            path.reverse()
            return path

        # Skip if we already found a better route to this state
        state_key = (x, y, diag_flag)
        if g > best_g.get(state_key, float("inf")):
            continue

        expanded += 1
        if expanded > REF_NODE_LIMIT:
            return []

        for dx, dy, is_diag in _REF_DIRECTIONS:
            nx, ny = x + dx, y + dy

            if not _ref_in_bounds(nx, ny, map_w, map_h):
                continue
            if is_blocked_fn(nx, ny):
                continue

            if is_diag:
                # Alternating diagonal cost: if diag_flag is True, cost=1; else cost=2
                step_cost = 1.0 if diag_flag else 2.0
                new_diag_flag = not diag_flag
            else:
                step_cost = 1.0
                new_diag_flag = diag_flag

            new_g = g + step_cost
            neighbor_state = (nx, ny, new_diag_flag)

            if new_g < best_g.get(neighbor_state, float("inf")):
                best_g[neighbor_state] = new_g
                came_from[neighbor_state] = state_key
                h = _ref_heuristic(nx, ny, goal_x, goal_y)
                heapq.heappush(
                    open_heap, (new_g + h, counter, new_g, nx, ny, new_diag_flag)
                )
                counter += 1

    return []


# ---------------------------------------------------------------------------
# The pre-#143 GameMap.find_burst_walls (self -> gmap), docstring in gamemap.py
# ---------------------------------------------------------------------------

def find_burst_walls_reference(gmap, max_pops: int | None = None):
    h, w = gmap._h, gmap._w
    # S2c: atmosphere is int32 Q16.16 — dequantize to REAL pressure here so
    # the spread (hi-lo) compares against the real-unit burst_threshold `t`.
    from simulation import atmosphere_fixed as _atm_fx
    atm = _atm_fx.dequantize(gmap.atmosphere)
    solid = gmap.solid
    is_vacuum = gmap.is_vacuum
    # BC (boundary_conditions_spec_2026-07-19 §1, audit (c)): an ambient-side
    # neighbour is a real side holding the effective pin P_amb (not 0 like a
    # vacuum side, and not its possibly-stale materialized `atm` value — a
    # just-breached ring tile is seeded by neighbour-mean, not yet re-pinned).
    # So a wall with ambient on BOTH sides holds ~0 spread and never bursts.
    # On a space map is_ambient is empty -> the vacuum/atm branch is unchanged.
    is_ambient = gmap.is_ambient
    p_amb_real = (float(gmap._ambient.pin_q) / _atm_fx.FP_ONE
                  if gmap._ambient is not None else 0.0)
    thresh = gmap.materials.burst_threshold

    failing = []  # (differential, fy, fx)
    ys, xs = np.where(solid)
    for fy, fx in zip(ys.tolist(), xs.tolist()):
        mat_id = int(gmap.material[fy, fx])
        t = float(thresh[mat_id])
        if t <= 0.0:
            continue  # n/a material (e.g. air) never bursts
        lo = None
        hi = None
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = fy + dy, fx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            # A solid neighbour is not a side — it's more wall; skipping
            # it (rather than counting 0) is what makes the spread a true
            # differential: equal pressure on both open sides -> 0, and a
            # tile with fewer than two open/vacuum sides can never burst.
            if solid[ny, nx]:
                continue
            # An exposed-vacuum breach is a real side holding no air; an
            # ambient breach is a real side holding the effective pin P_amb.
            if is_vacuum[ny, nx]:
                p = 0.0
            elif is_ambient[ny, nx]:
                p = p_amb_real
            else:
                p = float(atm[ny, nx])
            lo = p if lo is None or p < lo else lo
            hi = p if hi is None or p > hi else hi
        if lo is None:
            continue
        spread = hi - lo
        if spread > t:
            failing.append((spread, fy, fx))

    if not failing:
        return []
    # Worst differentials first; apply the per-tick cap.
    failing.sort(key=lambda r: r[0], reverse=True)
    if max_pops is not None:
        failing = failing[:max_pops]
    return [(fy, fx) for _, fy, fx in failing]
