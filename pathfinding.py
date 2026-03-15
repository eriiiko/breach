"""
Pathfinding systems for Breach tactical game.

Two pathfinding algorithms:
1. Standard A* — used by zombies, operates on the fine tile grid (120x75).
2. Temporal A* — used by player units, adds time dimension for collision avoidance.

Plus a ReservationTable class for temporal A* coordination.

All coordinates are fine tile positions (top-left corner of 3x3 unit blocks).
Fine grid: 120 wide x 75 tall (coarse 40x25 * 3).
"""

import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

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
# 1. Standard A* (for zombies)
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


# ---------------------------------------------------------------------------
# 3. Reservation Table
# ---------------------------------------------------------------------------

class ReservationTable:
    """
    Sparse reservation table for temporal A* coordination.

    Stores reservations as {(x, y, tick): unit_id} for individual fine tiles.
    When a unit reserves a path, all tiles in its 3x3 block are marked for
    each tick in the segment's [tick_start, tick_end) range.

    Usage:
        rt = ReservationTable(120, 75, 500)
        rt.reserve("marine_1", [(10, 10, 0, 9), (11, 10, 9, 18)])
        rt.is_reserved(10, 10, 5)           # True
        rt.is_reserved(10, 10, 5, exclude_unit="marine_1")  # False
        rt.clear("marine_1")
    """

    def __init__(self, map_w: int = FINE_W, map_h: int = FINE_H, max_ticks: int = 500):
        self.map_w = map_w
        self.map_h = map_h
        self.max_ticks = max_ticks
        self._table: Dict[Tuple[int, int, int], object] = {}
        # Track which keys belong to each unit for fast clearing
        self._unit_keys: Dict[object, Set[Tuple[int, int, int]]] = {}

    def reserve(
        self,
        unit_id: object,
        path: List[Tuple[int, int, int, int]],
        unit_size: int = UNIT_SIZE,
    ) -> None:
        """
        Reserve tiles along a path for a unit.

        Args:
            unit_id: Identifier for the unit.
            path: List of (x, y, tick_start, tick_end) segments. For each
                segment, the unit's block occupies (x..x+unit_size-1,
                y..y+unit_size-1) for ticks in [tick_start, tick_end).
            unit_size: Side length of the unit block (default 3).
        """
        if unit_id not in self._unit_keys:
            self._unit_keys[unit_id] = set()

        keys = self._unit_keys[unit_id]

        for x, y, tick_start, tick_end in path:
            for t in range(tick_start, tick_end):
                if t >= self.max_ticks:
                    break
                for bx in range(unit_size):
                    for by in range(unit_size):
                        key = (x + bx, y + by, t)
                        self._table[key] = unit_id
                        keys.add(key)

    def is_reserved(
        self,
        x: int,
        y: int,
        tick: int,
        exclude_unit: object = None,
        unit_size: int = UNIT_SIZE,
    ) -> bool:
        """
        Check if any tile in the 3x3 block at (x, y) is reserved at this tick.

        Args:
            x, y: Top-left of the unit block to check.
            tick: The tick to check.
            exclude_unit: If set, ignore reservations by this unit.
            unit_size: Side length of the block (default 3).

        Returns:
            True if any tile in the block is reserved by another unit.
        """
        for bx in range(unit_size):
            for by in range(unit_size):
                key = (x + bx, y + by, tick)
                occupant = self._table.get(key)
                if occupant is not None and occupant != exclude_unit:
                    return True
        return False

    def clear(self, unit_id: object = None) -> None:
        """
        Clear reservations.

        Args:
            unit_id: If provided, clear only this unit's reservations.
                If None, clear the entire table.
        """
        if unit_id is None:
            self._table.clear()
            self._unit_keys.clear()
        else:
            keys = self._unit_keys.pop(unit_id, set())
            for key in keys:
                if self._table.get(key) == unit_id:
                    del self._table[key]


# ---------------------------------------------------------------------------
# 2. Temporal A* (for player unit collision avoidance)
# ---------------------------------------------------------------------------

def temporal_astar(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    ticks_per_tile: int,
    start_tick: int,
    max_ticks: int,
    is_blocked_fn: Callable[[int, int], bool],
    reservation_table: ReservationTable,
    exclude_unit: object = None,
    map_w: int = FINE_W,
    map_h: int = FINE_H,
) -> List[Tuple[int, int, int]]:
    """
    Temporal A* pathfinding — A* in (x, y, tick) space.

    Used for player units that need to avoid colliding with each other.
    Units move 1 tile every `ticks_per_tile` ticks; between moves they wait.

    Args:
        start_x, start_y: Top-left of unit's 3x3 block at start.
        goal_x, goal_y: Top-left of unit's 3x3 block at goal.
        ticks_per_tile: Ticks to traverse one cardinal tile (e.g. 9 for marine).
        start_tick: The current game tick when movement begins.
        max_ticks: Absolute tick limit — do not search beyond this.
        is_blocked_fn: Callable(x, y) -> bool for static walls (3x3 block check).
        reservation_table: ReservationTable instance for dynamic unit avoidance.
        exclude_unit: Unit ID to exclude from reservation checks (self).
        map_w: Fine grid width (default 120).
        map_h: Fine grid height (default 75).

    Returns:
        List of (x, y, tick) waypoints from start to goal, or empty list
        if no path exists within the time budget.

    Notes:
        Neighbors from (x, y, t):
          - Wait:     (x, y, t+1)               cost = 1
          - Cardinal: (x+dx, y, t+ticks)        cost = ticks_per_tile
          - Diagonal: (x+dx, y+dy, t+dc*ticks)  cost = dc*ticks_per_tile
            where dc alternates 1, 2 (D&D 3.5 style).
    """
    if not _in_bounds(start_x, start_y, map_w, map_h):
        return []
    if not _in_bounds(goal_x, goal_y, map_w, map_h):
        return []
    if is_blocked_fn(start_x, start_y) or is_blocked_fn(goal_x, goal_y):
        return []

    if (start_x, start_y) == (goal_x, goal_y):
        return [(start_x, start_y, start_tick)]

    # State: (x, y, tick, diag_flag)
    start_state = (start_x, start_y, start_tick, True)
    h = _alternating_diagonal_heuristic(start_x, start_y, goal_x, goal_y) * ticks_per_tile

    # Priority queue: (f, tie_breaker, g, x, y, tick, diag_flag)
    counter = 0
    open_heap: list = []
    heapq.heappush(open_heap, (h, counter, 0.0, start_x, start_y, start_tick, True))
    counter += 1

    best_g: Dict[Tuple[int, int, int, bool], float] = {start_state: 0.0}
    came_from: Dict[
        Tuple[int, int, int, bool], Optional[Tuple[int, int, int, bool]]
    ] = {start_state: None}

    expanded = 0

    while open_heap:
        f, _, g, x, y, t, diag_flag = heapq.heappop(open_heap)

        # Goal check
        if x == goal_x and y == goal_y:
            path: List[Tuple[int, int, int]] = []
            state: Optional[Tuple[int, int, int, bool]] = (x, y, t, diag_flag)
            while state is not None:
                path.append((state[0], state[1], state[2]))
                state = came_from[state]
            path.reverse()
            return path

        state_key = (x, y, t, diag_flag)
        if g > best_g.get(state_key, float("inf")):
            continue

        expanded += 1
        if expanded > NODE_LIMIT:
            return []

        # --- Wait action: (x, y, t+1) ---
        wait_t = t + 1
        if wait_t < max_ticks:
            if not reservation_table.is_reserved(x, y, wait_t, exclude_unit):
                wait_state = (x, y, wait_t, diag_flag)
                new_g = g + 1.0
                if new_g < best_g.get(wait_state, float("inf")):
                    best_g[wait_state] = new_g
                    came_from[wait_state] = state_key
                    wh = _alternating_diagonal_heuristic(x, y, goal_x, goal_y) * ticks_per_tile
                    heapq.heappush(
                        open_heap, (new_g + wh, counter, new_g, x, y, wait_t, diag_flag)
                    )
                    counter += 1

        # --- Move actions ---
        for dx, dy, is_diag in _DIRECTIONS:
            nx, ny = x + dx, y + dy

            if not _in_bounds(nx, ny, map_w, map_h):
                continue
            if is_blocked_fn(nx, ny):
                continue

            if is_diag:
                diag_cost_mult = 1 if diag_flag else 2
                step_ticks = diag_cost_mult * ticks_per_tile
                new_diag_flag = not diag_flag
            else:
                step_ticks = ticks_per_tile
                new_diag_flag = diag_flag

            arrive_t = t + step_ticks
            if arrive_t >= max_ticks:
                continue

            # Check reservation at arrival tick
            if reservation_table.is_reserved(nx, ny, arrive_t, exclude_unit):
                continue

            # Also check that the destination is clear for all intermediate ticks
            # (the unit occupies the origin during transit, then appears at dest)
            # We check the destination at the arrival tick (already done above)
            # and the origin for all transit ticks
            transit_blocked = False
            for check_t in range(t + 1, arrive_t):
                if check_t >= max_ticks:
                    transit_blocked = True
                    break
                if reservation_table.is_reserved(x, y, check_t, exclude_unit):
                    transit_blocked = True
                    break
            if transit_blocked:
                continue

            new_g = g + float(step_ticks)
            neighbor_state = (nx, ny, arrive_t, new_diag_flag)

            if new_g < best_g.get(neighbor_state, float("inf")):
                best_g[neighbor_state] = new_g
                came_from[neighbor_state] = state_key
                nh = _alternating_diagonal_heuristic(nx, ny, goal_x, goal_y) * ticks_per_tile
                heapq.heappush(
                    open_heap,
                    (new_g + nh, counter, new_g, nx, ny, arrive_t, new_diag_flag),
                )
                counter += 1

    return []
