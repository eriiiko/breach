"""#143 -- ``pathfinding.astar`` is the pre-#143 search, bit for bit.

The optimised A* (doubled integer costs, flat tile tables, a ``passable=`` grid
fast path, a per-search memo of the blocked callback) may not change which path
any bot, zombie or generator gets: the oracle is the frozen pre-#143 body in
``tests/_reference_scans.py::astar_reference``, and every property below is
"the new call returns the SAME LIST (element for element) as the oracle", over
seeded random cases. The path is fixed by the heap's pop order (the total order
on ``(f, push counter)``), the ``g > best_g`` / ``new_g < best_g`` tests and the
neighbour order, so a change to any of those moves a path on open grids with
many equal-cost routes -- which is what the random open-ish grids here are full
of -- and fails these tests.

These tests assert properties, never a count or a set of something designed to
grow; the one real level used (``ship_demo``) is only a source of realistic
grids.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tests", ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _reference_scans as ref  # noqa: E402
import pathfinding  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

SIZES = [(5, 5), (12, 9), (30, 20), (47, 33), (80, 50)]
DENSITIES = [0.0, 0.08, 0.2, 0.32, 0.45]


def _tiles(rng, w, h, density):
    """A random per-tile 'open' grid (True = walkable), with the odd wall run."""
    ok = rng.random((h, w)) >= density
    for _ in range(int(rng.integers(0, 4))):
        if rng.random() < 0.5:
            ok[int(rng.integers(0, h)), :] = rng.random(w) < 0.5
        else:
            ok[:, int(rng.integers(0, w))] = rng.random(h) < 0.5
    return ok


def _footprint_grid(ok, fp):
    """``control_bots._passable_grid``'s product: every tile of the block open."""
    return sliding_window_view(ok, (fp, fp)).all(axis=(2, 3))


def _endpoints(rng, w, h, n):
    """Start/goal pairs: mostly in range, some off the 3x3-block range or the
    map, some equal."""
    out = []
    for _ in range(n):
        sx, sy = int(rng.integers(-2, w + 2)), int(rng.integers(-2, h + 2))
        if rng.random() < 0.85:
            sx, sy = int(rng.integers(0, max(1, w - 2))), int(rng.integers(0, max(1, h - 2)))
        gx, gy = int(rng.integers(-2, w + 2)), int(rng.integers(-2, h + 2))
        if rng.random() < 0.85:
            gx, gy = int(rng.integers(0, max(1, w - 2))), int(rng.integers(0, max(1, h - 2)))
        if rng.random() < 0.05:
            gx, gy = sx, sy
        out.append((sx, sy, gx, gy))
    return out


def _cases(seed, n_per=6):
    rng = np.random.default_rng(seed)
    for (w, h) in SIZES:
        for d in DENSITIES:
            ok = _tiles(rng, w, h, d)
            for e in _endpoints(rng, w, h, n_per):
                yield w, h, ok, e


# ---------------------------------------------------------------------------
# The callback path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_callback_path_returns_the_pre143_path(seed):
    """PROPERTY: for any grid, start and goal (blocked, out of range, equal,
    unreachable), ``astar`` through ``is_blocked_fn`` returns exactly the list
    the pre-#143 A* returns. Breaks if the pop order, the stale-entry or
    relax tests, the neighbour order, the diagonal-flag cost or the early
    returns change."""
    nonempty = 0
    for w, h, ok, (sx, sy, gx, gy) in _cases(1000 + seed):
        def blocked(x, y, _o=ok):
            return not _o[y, x]
        want = ref.astar_reference(sx, sy, gx, gy, blocked, w, h)
        got = pathfinding.astar(sx, sy, gx, gy, blocked, w, h)
        assert got == want, (w, h, sx, sy, gx, gy)
        nonempty += bool(want)
    assert nonempty, "the harness must exercise real paths, not only refusals"


def test_open_grid_ties_pick_the_same_one_of_many_equal_cost_paths():
    """PROPERTY: on an obstacle-free grid (thousands of equal-cost routes) the
    chosen route is the pre-#143 one, for every start/goal pair of a coarse
    sweep. Breaks if the push counter's tie-break (or the neighbour order that
    feeds it) changes."""
    w, h = 24, 18
    never = lambda x, y: False  # noqa: E731
    for sx in range(0, w - 2, 5):
        for sy in range(0, h - 2, 5):
            for gx in range(0, w - 2, 4):
                for gy in range(0, h - 2, 4):
                    assert (pathfinding.astar(sx, sy, gx, gy, never, w, h)
                            == ref.astar_reference(sx, sy, gx, gy, never, w, h))


def test_default_map_size_is_the_fine_grid():
    """PROPERTY: the default ``map_w``/``map_h`` still mean the 120x75 fine
    grid, as before (callers without sizes). Breaks if a default moves."""
    rng = np.random.default_rng(7)
    ok = _tiles(rng, 120, 75, 0.2)

    def blocked(x, y):
        return not ok[y, x]
    for (sx, sy, gx, gy) in _endpoints(rng, 120, 75, 8):
        assert (pathfinding.astar(sx, sy, gx, gy, blocked)
                == ref.astar_reference(sx, sy, gx, gy, blocked))


@pytest.mark.parametrize("limit", [3, 25, 150])
def test_node_limit_returns_empty_exactly_when_the_old_search_did(monkeypatch, limit):
    """PROPERTY: the ``NODE_LIMIT`` cut-off (an empty list after more than
    LIMIT expansions) fires on exactly the searches it fired on before, and
    the paths that finish under it are unchanged. The limit is read at call
    time. Breaks if the expansion count, where it is tested (after the stale
    skip) or the limit's lookup moves."""
    monkeypatch.setattr(pathfinding, "NODE_LIMIT", limit)
    monkeypatch.setattr(ref, "REF_NODE_LIMIT", limit)
    cut = finished = 0
    for w, h, ok, (sx, sy, gx, gy) in _cases(50 + limit, n_per=5):
        def blocked(x, y, _o=ok):
            return not _o[y, x]
        want = ref.astar_reference(sx, sy, gx, gy, blocked, w, h)
        got = pathfinding.astar(sx, sy, gx, gy, blocked, w, h)
        assert got == want
        # classify with the limit lifted: reachable but cut = the limit fired
        monkeypatch.setattr(ref, "REF_NODE_LIMIT", 10 ** 9)
        free = ref.astar_reference(sx, sy, gx, gy, blocked, w, h)
        monkeypatch.setattr(ref, "REF_NODE_LIMIT", limit)
        cut += bool(free) and not want
        finished += bool(want)
    assert cut, "the limit must actually fire in this sweep"
    assert finished, "and some searches must still finish under it"


def test_callback_is_asked_once_per_tile_and_only_about_in_range_tiles():
    """PROPERTY: the per-search memo -- ``is_blocked_fn`` is called at most once
    for any tile, and only for top-lefts a 3x3 block fits at, so a callback
    that is slow (``GameMap.is_passable_block``) is paid once per tile touched,
    never per neighbour visit. Breaks if the memo is dropped or a tile outside
    ``_in_bounds`` is asked about."""
    rng = np.random.default_rng(3)
    w, h = 40, 30
    ok = _tiles(rng, w, h, 0.15)
    ok[0, 0] = ok[h - 3, w - 3] = True
    asked = []

    def blocked(x, y):
        asked.append((x, y))
        return not ok[y, x]
    pathfinding.astar(0, 0, w - 3, h - 3, blocked, w, h)
    assert len(asked) > 10, "the search must have touched tiles"
    assert len(asked) == len(set(asked))
    assert all(0 <= x <= w - 3 and 0 <= y <= h - 3 for x, y in asked)


def test_start_and_goal_are_asked_in_the_old_order_and_short_circuit():
    """PROPERTY: the early checks keep their shape -- the start is asked
    first, and a blocked start means the goal is never asked; an out-of-range
    endpoint asks nothing. Breaks if the early returns reorder."""
    seen = []

    def blocked(x, y):
        seen.append((x, y))
        return (x, y) == (1, 1)
    assert pathfinding.astar(1, 1, 5, 5, blocked, 12, 12) == []
    assert seen == [(1, 1)]
    seen.clear()
    assert pathfinding.astar(-1, 0, 5, 5, blocked, 12, 12) == []
    assert pathfinding.astar(0, 0, 10, 5, blocked, 12, 12) == []   # 10 > 12 - 3
    assert seen == []


# ---------------------------------------------------------------------------
# The grid fast path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fp", [1, 3])
@pytest.mark.parametrize("seed", range(4))
def test_passable_path_equals_the_pre143_callback_on_footprint_grids(fp, seed):
    """PROPERTY: ``passable=`` with a grid built the way
    ``control_bots._passable_grid`` builds it (shape ``(h-fp+1, w-fp+1)``,
    smaller than the map for fp > 1, and for fp = 1 larger than the 3x3-block
    range ``_in_bounds`` allows) returns the list the old callback
    ``not grid[y, x]`` returned. Breaks if the fast path mis-aligns the grid
    with the map, treats the shrunk border wrongly, or drifts from the
    callback law."""
    nonempty = 0
    for w, h, tiles, (sx, sy, gx, gy) in _cases(2000 + seed):
        grid = _footprint_grid(tiles, fp)

        def blocked(x, y, _g=grid):
            return not _g[y, x]
        want = ref.astar_reference(sx, sy, gx, gy, blocked, w, h)
        got = pathfinding.astar(sx, sy, gx, gy, None, w, h, passable=grid)
        assert got == want, (fp, w, h, sx, sy, gx, gy)
        # the callback is ignored when a grid is given
        assert pathfinding.astar(sx, sy, gx, gy, blocked, w, h, passable=grid) == want
        nonempty += bool(want)
    assert nonempty


def test_passable_grid_smaller_than_the_range_is_blocked_outside():
    """PROPERTY: where the grid does not cover a tile a 3x3 block could stand
    at (a footprint-5 grid on a map whose range is wider), the tile is
    blocked -- the path never steps outside the grid, and equals the old
    search with an out-of-grid-is-blocked callback. Breaks if the fast path
    reads outside its table or opens the uncovered strip."""
    rng = np.random.default_rng(11)
    w, h, fp = 36, 26, 5
    tiles = _tiles(rng, w, h, 0.1)
    grid = _footprint_grid(tiles, fp)
    gh, gw = grid.shape

    def blocked(x, y):
        return not (y < gh and x < gw and grid[y, x])
    for (sx, sy, gx, gy) in _endpoints(rng, w, h, 40):
        want = ref.astar_reference(sx, sy, gx, gy, blocked, w, h)
        got = pathfinding.astar(sx, sy, gx, gy, None, w, h, passable=grid)
        assert got == want
        assert all(x < gw and y < gh for x, y in got)


def test_passable_path_node_limit(monkeypatch):
    """PROPERTY: the fast path honours the same ``NODE_LIMIT`` cut as the old
    search (empty list, same searches). Breaks if the fast path forgets the
    limit or reads it at import time."""
    monkeypatch.setattr(pathfinding, "NODE_LIMIT", 40)
    monkeypatch.setattr(ref, "REF_NODE_LIMIT", 40)
    rng = np.random.default_rng(5)
    w, h = 60, 40
    tiles = _tiles(rng, w, h, 0.05)
    grid = _footprint_grid(tiles, 3)
    cut = 0
    for (sx, sy, gx, gy) in _endpoints(rng, w, h, 60):
        def blocked(x, y):
            return not grid[y, x]
        want = ref.astar_reference(sx, sy, gx, gy, blocked, w, h)
        assert pathfinding.astar(sx, sy, gx, gy, None, w, h, passable=grid) == want
        cut += not want
    assert cut


def test_real_level_footprint_grids_plan_the_same_paths():
    """PROPERTY: on the grids the bots actually plan over (the real
    ``ship_demo`` floor, footprint 3, built with ``control_bots``' own
    ``mobility > 0`` law), the fast path returns the pre-#143 path for random
    walkable start/goal pairs. Breaks on any divergence the synthetic grids
    miss (long corridors, doorways, one-tile gaps)."""
    import level_loader
    from simulation.gamemap import GameMap
    gmap = GameMap(level_loader.load("ship_demo"))
    ok = gmap.materials.mobility[gmap.material] > 0
    grid = _footprint_grid(ok, 3)
    h, w = gmap.material.shape
    ys, xs = np.nonzero(grid)
    rng = np.random.default_rng(9)
    nonempty = 0
    for _ in range(12):
        a, b = rng.integers(0, len(ys), 2)
        sx, sy, gx, gy = int(xs[a]), int(ys[a]), int(xs[b]), int(ys[b])

        def blocked(x, y):
            return not grid[y, x]
        want = ref.astar_reference(sx, sy, gx, gy, blocked, w, h)
        assert pathfinding.astar(sx, sy, gx, gy, None, w, h, passable=grid) == want
        assert pathfinding.astar(sx, sy, gx, gy, blocked, w, h) == want
        nonempty += bool(want)
    assert nonempty


def test_astar_without_a_blocked_fn_or_a_grid_is_refused():
    """PROPERTY: ``astar`` needs one of ``is_blocked_fn`` / ``passable=``; a
    call with neither raises by name rather than searching an open map.
    Breaks if the guard is removed."""
    with pytest.raises(TypeError, match="is_blocked_fn or passable"):
        pathfinding.astar(0, 0, 5, 5, None, 12, 12)
