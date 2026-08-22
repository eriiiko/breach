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

def astar(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    is_blocked_fn: Callable[[int, int], bool],
    map_w: int = FINE_W,
    map_h: int = FINE_H,
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
        Node expansion is capped at NODE_LIMIT to prevent runaway searches.
    """
    if not _in_bounds(start_x, start_y, map_w, map_h):
        return []
    if not _in_bounds(goal_x, goal_y, map_w, map_h):
        return []
    if is_blocked_fn(start_x, start_y) or is_blocked_fn(goal_x, goal_y):
        return []
    if (start_x, start_y) == (goal_x, goal_y):
        return [(start_x, start_y)]

    # State: (x, y, diag_flag)  — diag_flag True means next diagonal costs 1
    start_state = (start_x, start_y, True)
    h = _alternating_diagonal_heuristic(start_x, start_y, goal_x, goal_y)

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
        if expanded > NODE_LIMIT:
            return []

        for dx, dy, is_diag in _DIRECTIONS:
            nx, ny = x + dx, y + dy

            if not _in_bounds(nx, ny, map_w, map_h):
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
                h = _alternating_diagonal_heuristic(nx, ny, goal_x, goal_y)
                heapq.heappush(
                    open_heap, (new_g + h, counter, new_g, nx, ny, new_diag_flag)
                )
                counter += 1

    return []
