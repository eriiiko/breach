"""The room graph's CHANGING part (#195 1b; RL agent design 5.1 / 5.7).

The fixed part -- the rooms, the openings between them and the tile -> room
map -- is level data (``LevelData.rooms``, a ``level_loader.RoomGraph``,
written by the generator through ``level_lib``). What is walkable NOW is
never stored: it is worked out here from the map as it stands, by two
read-only functions:

- :func:`opening_states`: for each saved opening, whether it is passable
  now and, if not, whether it is a door a unit could open (a living,
  closed door entity that is not a hull door and is not wired);
- :func:`joined_pairs`: the pairs of rooms joined now -- by any connected
  patch of walkable tiles outside every room that touches both (an open
  opening, an airlock's passage with both doors open, or a HOLE blown
  through a wall; a hole has no id: the ACCEPTED GAP, an order cannot name
  it).

Both are cached per map on ``GameMap._tile_serial`` (every topology seam --
``destroy_wall``, ``seal_tiles``, ``unseal_tiles``, a door -- moves it
through ``on_tile_changed``) and, for the openings, the doors' states and
the wired set. Nothing here is synced, hashed or written: host-side reads
only, and nothing in the tick calls them.
"""
from __future__ import annotations

import weakref
from collections import deque
from dataclasses import dataclass

import numpy as np

from simulation.entities.door import DOOR_CLOSED
from simulation.horde import NEIGHBOURS, is_hull_door, walkable_tiles

#: map -> {(function, graph): (cache key, value)}; dies with the map.
_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class OpeningState:
    """One saved opening, now. ``passable``: every tile of it is walkable
    (``mobility > 0``). ``openable``: it is not passable, and every tile of
    it that is not walkable lies in a living, CLOSED door entity that is not
    a hull door and is not wired (``use`` could open it)."""
    id: int
    passable: bool
    openable: bool


def _check_shape(graph, gmap) -> None:
    if tuple(graph.grid.shape) != tuple(gmap.material.shape):
        raise ValueError(
            f"room graph grid {graph.grid.shape} != the map's "
            f"{gmap.material.shape} (the graph is at base resolution; a "
            f"--res-scaled map has none)")


def _cached(gmap, fn: str, graph, key, compute):
    slot = _CACHE.setdefault(gmap, {})
    hit = slot.get((fn, graph))
    if hit is not None and hit[0] == key:
        return hit[1]
    value = compute()
    slot[(fn, graph)] = (key, value)
    return value


def wired_door_ordinals(wires) -> frozenset:
    """The ordinals of the doors a level wire drives (an ``open`` or
    ``close`` input): those doors are the logic's, not a unit's."""
    return frozenset(int(w.target_ordinal) for w in (wires or ())
                     if w.input in ("open", "close"))


def opening_states(graph, gmap, doors=(), wires=()) -> tuple:
    """Per saved opening (id order), an :class:`OpeningState` for the map as
    it stands. ``doors``: the sim's door runtimes (``Simulation._doors``);
    ``wires``: the level's resolved wires (``LevelData.wires``). Cached on
    ``gmap._tile_serial``, the doors' ``(ordinal, alive, state)`` and the
    wired set."""
    _check_shape(graph, gmap)
    doors = tuple(doors or ())
    wired = wired_door_ordinals(wires)
    key = (gmap._tile_serial,
           tuple((int(d.ordinal), bool(d.alive), d.state) for d in doors),
           wired)

    def compute():
        walk = walkable_tiles(gmap)
        door_tile = {}
        for d in doors:
            if (d.alive and d.state == DOOR_CLOSED
                    and int(d.ordinal) not in wired
                    and not is_hull_door(gmap, d.span)):
                for t in d.span:
                    door_tile[(int(t[0]), int(t[1]))] = True
        out = []
        for o in graph.openings:
            blocked = [t for t in o.tiles if not walk[t]]
            passable = not blocked
            openable = (not passable
                        and all(t in door_tile for t in blocked))
            out.append(OpeningState(id=o.id, passable=passable,
                                    openable=openable))
        return tuple(out)

    return _cached(gmap, "openings", graph, key, compute)


def joined_pairs(graph, gmap) -> frozenset:
    """The pairs ``(a, b)``, ``a < b``, of rooms joined NOW: two rooms are
    joined when a 4-connected patch of walkable tiles outside every room
    (grid -1) touches a walkable tile of each, or when walkable tiles of the
    two touch directly. Finds what the saved openings find when they are
    passable AND a hole blown through a wall. Cached on
    ``gmap._tile_serial``."""
    _check_shape(graph, gmap)

    def compute():
        grid = graph.grid
        walk = walkable_tiles(gmap)
        h, w = grid.shape
        free = walk & (grid < 0)
        seen = np.zeros_like(free)
        pairs = set()

        def touch(rooms, ny, nx):
            if 0 <= ny < h and 0 <= nx < w and walk[ny, nx]:
                r = int(grid[ny, nx])
                if r >= 0:
                    rooms.add(r)

        for sy, sx in zip(*np.nonzero(free)):
            if seen[sy, sx]:
                continue
            seen[sy, sx] = True
            rooms = set()
            q = deque([(int(sy), int(sx))])
            while q:
                y, x = q.popleft()
                for dy, dx in NEIGHBOURS:
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if free[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
                    else:
                        touch(rooms, ny, nx)
            rs = sorted(rooms)
            pairs.update((a, b) for i, a in enumerate(rs) for b in rs[i + 1:])
        # Rooms whose walkable tiles touch directly (no generated ship has
        # two rooms without a wall between them; kept for honesty).
        rw = np.where(walk, grid, -1)
        for a_, b_ in ((rw[:, :-1], rw[:, 1:]), (rw[:-1, :], rw[1:, :])):
            m = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
            for a, b in zip(a_[m].tolist(), b_[m].tolist()):
                pairs.add((min(a, b), max(a, b)))
        return frozenset(pairs)

    return _cached(gmap, "joined", graph, gmap._tile_serial, compute)
