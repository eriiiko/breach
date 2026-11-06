"""
Pathfinding systems for Breach tactical game.

Standard A* — used by zombies and player units, operates on the fine tile
grid (120x75).

All coordinates are fine tile positions (top-left corner of 3x3 unit blocks).
Fine grid: 120 wide x 75 tall (coarse 40x25 * 3).
"""

import heapq
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINE_W = 120
FINE_H = 75
UNIT_SIZE = 3
NODE_LIMIT = 50_000

# 8 directions: (dx, dy, is_diagonal)
_DIRECTIONS = [
    ( 1,  0, False),
    (-1,  0, False),
    ( 0,  1, False),
    ( 0, -1, False),
    ( 1,  1, True),
    ( 1, -1, True),
    (-1,  1, True),
    (-1, -1, True),
]


def _in_bounds(x: int, y: int, map_w: int, map_h: int) -> bool:
    """Check that a 3x3 unit block starting at (x, y) fits within the map."""
    return 0 <= x <= map_w - UNIT_SIZE and 0 <= y <= map_h - UNIT_SIZE


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def _alternating_diagonal_heuristic(x0: int, y0: int, x1: int, y1: int) -> float:
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

# The search runs on DOUBLED INTEGER costs (g2 = 2g, h2 = 2h): every float the
# law involves -- a step of 1, a diagonal of 1 or 2, the octile heuristic
# straight + 1.5*diag -- is an exact multiple of 0.5, so the doubled integers
# order exactly as the floats did (#143). The heap's pop order is the total
# order on (f, push counter); the ``g > best_g`` / ``new_g < best_g`` tests and
# the neighbour order of ``_DIRECTIONS`` fix the rest: all kept, so the path
# returned (which of several equal-cost paths included) is the pre-#143 one.
_COST_STRAIGHT = 2        # a straight step, doubled
_COST_DIAG_CHEAP = 2      # a diagonal while the flag is set, doubled
_COST_DIAG_DEAR = 4       # the next diagonal, doubled
_UNKNOWN = 2              # a tile whose blocked-ness the callback has not told yet
_BIG = 1 << 62


def _padded_ok(passable, map_w: int, map_h: int, pw: int, ph: int) -> bytes:
    """The grid fast path's tile table: ``ok[(y + 1) * pw + (x + 1)]`` is 1 when
    a block may stand with its top-left at (x, y) and fits the map
    (``_in_bounds``), else 0. A one-tile border of zeros means a neighbour step
    needs no bounds test. ``passable`` may be SMALLER than the map (the
    footprint grid of ``control_bots``, ``(h - fp + 1, w - fp + 1)``): what it
    does not cover is blocked, which is what indexing it directly meant."""
    import numpy as np
    g = np.asarray(passable, dtype=bool)
    ry = max(0, min(g.shape[0], map_h - UNIT_SIZE + 1))
    rx = max(0, min(g.shape[1], map_w - UNIT_SIZE + 1))
    buf = np.zeros((ph, pw), dtype=np.uint8)
    buf[1:1 + ry, 1:1 + rx] = g[:ry, :rx]
    return buf.tobytes()


def _unknown_ok(map_w: int, map_h: int, pw: int, ph: int) -> bytearray:
    """The callback path's tile table: every in-bounds tile ``_UNKNOWN``,
    resolved the first time a search looks at it (the callback is called at
    most once per tile per search, for the tiles the old search asked about)."""
    ok = bytearray(pw * ph)
    nx = map_w - UNIT_SIZE + 1
    if nx > 0:
        row = bytes([_UNKNOWN]) * nx
        for y in range(map_h - UNIT_SIZE + 1):
            a = (y + 1) * pw + 1
            ok[a:a + nx] = row
    return ok


def astar(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    is_blocked_fn: Optional[Callable[[int, int], bool]],
    map_w: int = FINE_W,
    map_h: int = FINE_H,
    *,
    passable=None,
) -> List[Tuple[int, int]]:
    """
    Standard A* pathfinding on the fine tile grid.

    Args:
        start_x, start_y: Top-left corner of the unit's 3x3 block at the start.
        goal_x, goal_y: Top-left corner of the unit's 3x3 block at the goal.
        is_blocked_fn: Callable(x, y) -> bool. Returns True if a 3x3 block
            placed at (x, y) is impassable. Must be a pure function of (x, y)
            for the duration of the call: the answer is remembered per tile.
            Ignored when ``passable`` is given (then it may be None).
        map_w: Fine grid width (default 120).
        map_h: Fine grid height (default 75).
        passable: Optional grid fast path -- a 2D bool array, ``passable[y, x]``
            True when a block may stand with its top-left at (x, y); outside
            the array is blocked. Equal to ``lambda x, y: not passable[y, x]``
            as ``is_blocked_fn``, without a Python call per neighbour.

    Returns:
        List of (x, y) tile positions from start to goal (inclusive), or
        empty list if no path exists.

    Notes:
        Diagonal cost alternates 1-2 (D&D 3.5 style). A boolean flag flips
        each diagonal step: first diagonal costs 1, second costs 2, etc.
        Node expansion is capped at NODE_LIMIT to prevent runaway searches.
    """
    if passable is None and is_blocked_fn is None:
        raise TypeError("astar needs is_blocked_fn or passable=")
    if not _in_bounds(start_x, start_y, map_w, map_h):
        return []
    if not _in_bounds(goal_x, goal_y, map_w, map_h):
        return []

    pw = map_w + 2                      # padded row stride
    ph = map_h + 2
    start_p = (start_y + 1) * pw + (start_x + 1)
    goal_p = (goal_y + 1) * pw + (goal_x + 1)

    if passable is not None:
        ok = _padded_ok(passable, map_w, map_h, pw, ph)
        if not ok[start_p] or not ok[goal_p]:
            return []
    else:
        if is_blocked_fn(start_x, start_y) or is_blocked_fn(goal_x, goal_y):
            return []
        ok = _unknown_ok(map_w, map_h, pw, ph)
        ok[start_p] = 1
        ok[goal_p] = 1
    if (start_x, start_y) == (goal_x, goal_y):
        return [(start_x, start_y)]

    # |x - goal_x| and |y - goal_y| by padded coordinate, for the heuristic.
    gpx = goal_x + 1
    gpy = goal_y + 1
    adx = [abs(px - gpx) for px in range(pw)]
    ady = [abs(py - gpy) for py in range(ph)]

    # (dx, dy, padded offset, is_diag) in _DIRECTIONS order
    dirs = tuple((dx, dy, dx + dy * pw, is_diag) for dx, dy, is_diag in _DIRECTIONS)

    # State = padded cell * 2 + diag_flag (1: the next diagonal costs 1).
    start_state = start_p * 2 + 1
    a = adx[start_x + 1]
    b = ady[start_y + 1]
    h2 = a + a + b if a > b else b + b + a      # 2*straight + 3*diag
    heap = [(h2, 0, 0, start_state)]            # (f2, push counter, g2, state)
    counter = 1
    best_g = {start_state: 0}
    came_from = {start_state: -1}

    heappush = heapq.heappush
    heappop = heapq.heappop
    node_limit = NODE_LIMIT
    expanded = 0

    while heap:
        _, _, g, state = heappop(heap)
        p = state >> 1

        # Goal check (either diag_flag state)
        if p == goal_p:
            path: List[Tuple[int, int]] = []
            while state >= 0:
                py, px = divmod(state >> 1, pw)
                path.append((px - 1, py - 1))
                state = came_from[state]
            path.reverse()
            return path

        # Skip if we already found a better route to this state
        if g > best_g[state]:
            continue

        expanded += 1
        if expanded > node_limit:
            return []

        flag = state & 1
        py, px = divmod(p, pw)

        for dx, dy, off, is_diag in dirs:
            q = p + off
            v = ok[q]
            if not v:
                continue
            if v == _UNKNOWN:
                if is_blocked_fn(px + dx - 1, py + dy - 1):
                    ok[q] = 0
                    continue
                ok[q] = 1

            if is_diag:
                # Alternating diagonal cost: flag set -> 1 (doubled 2), else 2 (4)
                if flag:
                    new_g = g + _COST_DIAG_CHEAP
                    nstate = q * 2
                else:
                    new_g = g + _COST_DIAG_DEAR
                    nstate = q * 2 + 1
            else:
                new_g = g + _COST_STRAIGHT
                nstate = q * 2 + flag

            if new_g < best_g.get(nstate, _BIG):
                best_g[nstate] = new_g
                came_from[nstate] = state
                a = adx[px + dx]
                b = ady[py + dy]
                h2 = a + a + b if a > b else b + b + a
                heappush(heap, (new_g + h2, counter, new_g, nstate))
                counter += 1

    return []
