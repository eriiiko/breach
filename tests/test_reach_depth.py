"""``blast_expansion.reach_depth`` (G4 #75, docs/explosion_look_design_2026-11-07.md
§6 B2): THE Blast reach walk with its hop count, which the shock front rides.
Pure and integer, no physics module. Each test names the property it protects and
the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation.blast_expansion import (reach_depth, reach_tiles,  # noqa: E402
                                        walk_seeds)

R2 = 400   # wide enough that the walls, not the radius, bound these walks


def _two_rooms(gap=True):
    """Two rooms split by a wall at column 10, with an optional one-tile door gap
    at row 2 (far from the centre's row 7)."""
    mask = np.zeros((15, 21), dtype=bool)
    mask[1:14, 1:20] = True
    mask[:, 10] = False
    if gap:
        mask[2, 10] = True
    return mask


def test_the_seeds_are_at_depth_zero():
    """Property: every seed of the walk (`walk_seeds`, the centre or the open
    side it delivers on) is at depth 0, open centre and charge-on-a-wall alike.
    BREAKS IF: reach_depth starts counting at 1, or seeds differently from
    reach_tiles' seed rule."""
    mask = _two_rooms()
    for fy, fx, approach in ((7, 5, None), (7, 10, None), (7, 10, (7, 11))):
        depth = reach_depth(mask, fy, fx, R2, approach)
        seeds = walk_seeds(mask, fy, fx, approach)
        assert seeds
        for s in seeds:
            assert depth[s] == 0
        assert {t for t, d in depth.items() if d == 0} == set(seeds)


def test_depth_grows_by_one_per_hop():
    """Property: depth is the BFS hop count along the walk: every reached
    non-seed tile has a reached 4-neighbour exactly one shallower, and no reached
    4-neighbour more than one shallower. BREAKS IF: the walk records squared or
    Euclidean distance, or the order of the queue stops being breadth-first."""
    mask = _two_rooms()
    depth = reach_depth(mask, 7, 5, R2)
    for (y, x), d in depth.items():
        nbrs = [depth[n] for n in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                if n in depth]
        assert all(n >= d - 1 for n in nbrs)
        if d > 0:
            assert d - 1 in nbrs
    # on open floor along the row, depth is the Manhattan step count
    assert [depth[(7, 5 + k)] for k in range(4)] == [0, 1, 2, 3]


def test_a_wall_stops_the_walk_and_a_gap_reaches_round_it():
    """Property: a tile behind a wall is unreached with no gap, and with a door
    gap it is reached at the depth of the way through the gap, deeper than its
    straight-line hop count. BREAKS IF: the walk ignores the mask (a front through
    walls), or counts a hop through the wall."""
    target = (7, 12)
    closed = reach_depth(_two_rooms(gap=False), 7, 8, R2)
    assert target not in closed
    opened = reach_depth(_two_rooms(gap=True), 7, 8, R2)
    gap = (2, 10)
    assert gap in opened and target in opened
    # the way round: centre -> gap, then gap -> target, both Manhattan in open rooms
    via_gap = opened[gap] + abs(target[0] - gap[0]) + abs(target[1] - gap[1])
    assert opened[target] == via_gap
    assert opened[target] > abs(target[1] - 8)


def test_reach_tiles_is_exactly_reach_depths_keys():
    """Property: reach_depth runs THE walk, so its tiles are reach_tiles' tiles on
    every case (open, a radius bound, a charge on a wall with and without a side).
    BREAKS IF: the two walks diverge (a copied walk edited in one place)."""
    rng = np.random.default_rng(7)
    mask = rng.random((24, 24)) > 0.3
    for fy, fx, r2, approach in ((12, 12, 30, None), (12, 12, 400, None),
                                 (0, 0, 50, None), (5, 7, 20, (5, 8)),
                                 (5, 7, 20, None)):
        assert sorted(reach_depth(mask, fy, fx, r2, approach)) == \
            reach_tiles(mask, fy, fx, r2, approach)
    m2 = _two_rooms()
    assert sorted(reach_depth(m2, 7, 10, R2, (7, 9))) == \
        reach_tiles(m2, 7, 10, R2, (7, 9))
