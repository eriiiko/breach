"""tools/gen_training_rooms.py -- THE writer of rung 2's small training
levels (#195 2c; design ``docs/rl_agent_rung2_design_2026-11-14.md`` 5.1,
note ``docs/rl_agent_rung2_note_2c_2026-11-17.md`` section 2).

A training room is a few rectangular rooms inside a hull, at the reference
tile (0.333 m), airtight, with a ROOM GRAPH (observation spec v1 reads it),
two spawn rooms (team 0 the learner, team 1 the target; each always on the
same tile) and a match file (``sight = "all_round"``, no zombies, a fixed
clock, elimination ends it). Everything is written through ``level_lib``
(the one writer); the gates are the ship generator's own
(``gen_ship.validate``: the airtight flood ``level_airtight.find_leaks``,
the 3 x 3 floor under each spawn cell, the A* path between the two spawn
rooms) and run BEFORE any byte is written; the level lint
(``level_airtight.check`` on the loaded level) is run after.

THE WRITER FUNCTIONS (:class:`Draft`): a room, an entity door between two
rooms, an open edge, a cover crate (the game's ``cover`` entity, not the
ship generator's crates, which are fuel and burn), the two spawn rooms, the
match file. Patch 2c's rooms are the three fixed layouts of
:data:`HAND_ROOMS` (one per shape of design 5.1); patch 2g adds the seeded
shapes ON THESE FUNCTIONS, never a second tool.

Each layout is written twice, as ``<name>`` and ``<name>_swapped`` (design
5.1 "sides": the two teams' spawn rooms exchanged, nothing else).

The rooms are written OUTSIDE the repository and made again from here;
two calls write the same bytes::

    python tools/gen_training_rooms.py --out <folder>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import gen_ship as G  # noqa: E402
import level_lib  # noqa: E402
from level_loader import EntityInstance, opening_order_key  # noqa: E402
from simulation.match import (  # noqa: E402
    FORMATION_FOOTPRINT, MatchSpec, MatchTeam, SpawnRoom,
)
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

#: The reference tile (the ship generator's, the playground's scale).
TILE_SIZE_M = G.TILE_SIZE_M
#: A door's width along its wall (tiles): the ship generator's.
DOOR_W = G.DOOR_W
#: The match file every training room carries (note 2c, 2; design 5.3).
TIME_LIMIT_S = 45.0
SIGHT = "all_round"
LEARNER_LOADOUT = "rifleman"
TARGET_LOADOUT = "rifleman"
ENV_PROFILE = "default"
#: The controllers written into match.toml: a run file replaces them
#: (``[run] policy``); these make the level playable on its own
#: (``tools/run_match.py``): the layered bots against a standing marine.
LEARNER_CONTROLLER = "bots"
TARGET_CONTROLLER = "idle"
#: The level's own match seed (an episode replaces it).
MATCH_SEED = 1
#: What a level folder holds that this tool writes (cleared first).
DATA_FILES = G.DATA_FILES
SWAPPED_SUFFIX = "_swapped"


def tiles(m: float) -> int:
    """Metres -> whole tiles at :data:`TILE_SIZE_M`."""
    return int(round(float(m) / TILE_SIZE_M))


@dataclass
class Draft:
    """A training level being drawn: the tilemap (all hull to start), the
    room grid, the rooms, the openings, the entities and the spawn rooms.
    Every writer function below adds to it; :func:`write` lands it."""
    name: str
    width: int
    height: int
    tm: np.ndarray = None
    grid: np.ndarray = None
    rooms: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    spawn_rooms: list = field(default_factory=list)
    spec: MatchSpec = None

    def __post_init__(self):
        if self.tm is None:
            self.tm = np.full((self.height, self.width), MAT_HULL,
                              dtype=np.int32)
        if self.grid is None:
            self.grid = np.full((self.height, self.width), -1,
                                dtype=np.int32)

    # -- the writer functions -------------------------------------------
    def room(self, x: int, y: int, w: int, h: int, kind: str = "room") -> int:
        """A room: the ``w x h`` floor rectangle at ``(x, y)`` (tiles), air,
        its tiles in the room grid. Returns its id (rooms are ids 0..n-1 in
        the order drawn)."""
        rid = len(self.rooms)
        if (self.grid[y:y + h, x:x + w] != -1).any():
            raise ValueError(f"{self.name}: room {rid} overlaps a room")
        self.tm[y:y + h, x:x + w] = MAT_AIR
        self.grid[y:y + h, x:x + w] = rid
        self.rooms.append(dict(id=rid, x=x, y=y, w=w, h=h, kind=kind))
        return rid

    def _shared_wall(self, a: int, b: int):
        """``(orient, line, lo, hi)`` of the one-tile wall rooms ``a`` and
        ``b`` share (the ship generator's rule, ``gen_ship.shared_wall``)."""
        ra, rb = (G.Room(id=r["id"], x=r["x"], y=r["y"], w=r["w"], h=r["h"])
                  for r in (self.rooms[a], self.rooms[b]))
        wall = G.shared_wall(ra, rb)
        if wall is None:
            raise ValueError(f"{self.name}: rooms {a} and {b} share no wall")
        return wall

    def _opening(self, a: int, b: int, kind: str, tl) -> None:
        ys = [y for y, _ in tl]
        xs = [x for _, x in tl]
        self.openings.append(dict(a=min(a, b), b=max(a, b), kind=kind,
                                  x=min(xs), y=min(ys),
                                  w=max(xs) - min(xs) + 1,
                                  h=max(ys) - min(ys) + 1))

    def door(self, a: int, b: int, at: int) -> None:
        """An entity door (``gen_ship.EDGE_KINDS["entity_door"]``: emitted
        in its initial state, stamped as that state stamps) of
        :data:`DOOR_W` tiles in the wall rooms ``a`` and ``b`` share,
        starting at tile ``at`` along it."""
        orient, line, lo, hi = self._shared_wall(a, b)
        if at < lo or at + DOOR_W - 1 > hi:
            raise ValueError(f"{self.name}: door {at}+{DOOR_W} leaves the "
                             f"shared wall {lo}..{hi}")
        tl = G._wall_tiles(orient, line, range(at, at + DOOR_W))
        kind = G.EDGE_KINDS["entity_door"]
        for (y, x) in tl:
            self.tm[y, x] = kind.stamps
        n = sum(1 for e in self.entities if e.class_name == "door")
        self.entities.append(G._door_entity(f"door_{n}", tl, orient,
                                            G.ENTITY_DOOR_INITIAL))
        self._opening(a, b, "entity_door", tl)

    def open_edge(self, a: int, b: int) -> None:
        """Rooms ``a`` and ``b`` as ONE open space: their whole shared wall
        is floor (``gen_ship.EDGE_KINDS["open"]``'s stamp), an ``open``
        opening of the graph (design 5.1: "one open space is two graph
        rooms joined by an open edge")."""
        orient, line, lo, hi = self._shared_wall(a, b)
        tl = G._wall_tiles(orient, line, range(lo, hi + 1))
        for (y, x) in tl:
            self.tm[y, x] = G.EDGE_KINDS["open"].stamps
        self._opening(a, b, "open", tl)

    def cover_crate(self, x: int, y: int, size_m: float = 1.0) -> None:
        """A cover crate: the game's ``cover`` entity (a rectangle that eats
        rounds, intangible to the tiles), ``size_m`` square, its top-left
        tile ``(x, y)``."""
        n = sum(1 for e in self.entities if e.class_name == "cover")
        self.entities.append(EntityInstance(
            id=f"crate_{n}", class_name="cover", ordinal=0,
            fields={"x": int(x), "y": int(y), "width_m": float(size_m),
                    "height_m": float(size_m)},
            authored_keys=("x", "y", "width_m", "height_m")))

    def spawns(self, learner, target) -> None:
        """The two spawn rooms: team 0 (the learner) with its footprint's
        top-left on tile ``learner``, team 1 (the target) on ``target``; one
        unit each, so each always stands on that tile."""
        fp = FORMATION_FOOTPRINT
        self.spawn_rooms = [
            SpawnRoom(team=t, x=int(p[0]), y=int(p[1]), w=fp, h=fp,
                      max_per_team=1)
            for t, p in enumerate((learner, target))]

    def match(self) -> None:
        """The match file (note 2c, 2): the learner and the standing target,
        one each, no zombies, :data:`TIME_LIMIT_S`, :data:`SIGHT`;
        elimination ends it (``objective = "eliminate"``, the default)."""
        learner = MatchTeam(n=1, loadout=LEARNER_LOADOUT,
                            env_profile=ENV_PROFILE,
                            controller=LEARNER_CONTROLLER)
        target = MatchTeam(n=1, loadout=TARGET_LOADOUT,
                           env_profile=ENV_PROFILE,
                           controller=TARGET_CONTROLLER)
        self.spec = MatchSpec(teams=(learner, target), seed=MATCH_SEED,
                              zombies=0, time_limit_s=TIME_LIMIT_S,
                              sight=SIGHT)

    # -- derived ------------------------------------------------------------
    def numbered_openings(self) -> list:
        """The openings with their ids: the rank under
        ``level_loader.opening_order_key``."""
        rows = sorted(self.openings, key=lambda r: opening_order_key(
            r["a"], r["b"], r["x"], r["y"]))
        return [dict(id=k, **r) for k, r in enumerate(rows)]

    def swapped(self) -> "Draft":
        """The same level with the two teams' spawn rooms exchanged (design
        5.1 "sides"), named ``<name>_swapped``."""
        a, b = self.spawn_rooms
        d = replace(self, name=self.name + SWAPPED_SUFFIX,
                    rooms=list(self.rooms), openings=list(self.openings),
                    entities=list(self.entities))
        d.spawn_rooms = [replace(b, team=0), replace(a, team=1)]
        return d


# ---------------------------------------------------------------------------
# The three hand-made rooms (note 2c, 2)
# ---------------------------------------------------------------------------
def range_hall() -> Draft:
    """One open space, 12 m x 8 m (two graph rooms joined by an open edge):
    the target in plain view about 9 m away; two cover crates off the line
    of fire."""
    half, h = tiles(6.0), tiles(8.0)                 # 18 x 24 tiles each
    d = Draft("range_hall", width=2 * half + 3, height=h + 2)
    a = d.room(1, 1, half, h)
    b = d.room(half + 2, 1, half, h)
    d.open_edge(a, b)
    y = 1 + (h - FORMATION_FOOTPRINT) // 2           # mid-height, both
    x0 = 3
    d.spawns((x0, y), (x0 + tiles(9.0), y))
    d.cover_crate(x0 + tiles(2.5), 1 + tiles(1.0))
    d.cover_crate(x0 + tiles(7.0), 1 + h - tiles(1.0) - tiles(1.0))
    d.match()
    return d


def range_door() -> Draft:
    """Two rooms of 7 m x 7 m joined by a door low in their shared wall;
    both spawn tiles high in their rooms, so the wall is on the line between
    them and the target is not in view from the learner's tile."""
    s = tiles(7.0)                                   # 21 tiles
    d = Draft("range_door", width=2 * s + 3, height=s + 2)
    a = d.room(1, 1, s, s)
    b = d.room(s + 2, 1, s, s)
    d.door(a, b, at=1 + s - DOOR_W - 2)              # near the floor's end
    d.spawns((3, 2), (2 * s + 3 - 3 - FORMATION_FOOTPRINT, 2))
    d.match()
    return d


def range_corner() -> Draft:
    """Three rooms of 6 m x 6 m in an L (west, east, and south of the
    east): two doors and a turn; the target in the far (south) room, off
    the axis of the second door."""
    s = tiles(6.0)                                   # 18 tiles
    d = Draft("range_corner", width=2 * s + 3, height=2 * s + 3)
    a = d.room(1, 1, s, s)                           # west
    b = d.room(s + 2, 1, s, s)                       # east
    c = d.room(s + 2, s + 2, s, s)                   # south of east
    d.door(a, b, at=1 + (s - DOOR_W) // 2)           # centred
    d.door(b, c, at=s + 3)                           # the west end of b-c
    fp = FORMATION_FOOTPRINT
    d.spawns((3, 1 + (s - fp) // 2),
             (2 * s + 2 - 2 - fp, 2 * s + 2 - 2 - fp))   # c's far corner
    d.match()
    return d


#: The fixed layouts of patch 2c: name -> builder (one per shape of design
#: 5.1). Patch 2g adds the seeded shapes beside them, on the same functions.
HAND_ROOMS = {"range_hall": range_hall, "range_door": range_door,
              "range_corner": range_corner}


def level_names() -> tuple:
    """Every level this tool writes: each hand room and its swapped twin."""
    return tuple(n + s for n in HAND_ROOMS for s in ("", SWAPPED_SUFFIX))


def drafts() -> list:
    """Every level as a :class:`Draft`, in :func:`level_names` order."""
    out = []
    for build in HAND_ROOMS.values():
        d = build()
        out += [d, d.swapped()]
    return out


# ---------------------------------------------------------------------------
# The gates and the write
# ---------------------------------------------------------------------------
class TrainingRoomRefused(ValueError):
    """A training room failed a gate (the message names every reason)."""


def gate(d: Draft) -> list:
    """The ship generator's gates on the draft (``gen_ship.validate``:
    airtight, the spawn cells on floor, an A* path for a 3 x 3 unit from
    the learner's spawn room to the target's; no mirror rule) and the room
    graph through the loader's own door. Returns the reasons (empty =
    pass)."""
    reasons = list(G.validate(d.tm, d.spawn_rooms, mirrored=False))
    try:
        level_lib.room_graph_blocks(d.rooms, d.numbered_openings(), d.grid)
    except ValueError as e:
        reasons.append(f"room graph: {e}")
    return reasons


def write(d: Draft, out_dir) -> Path:
    """Gate ``d`` and write it as ``<out_dir>/<name>`` through
    ``level_lib``; returns the folder. Regenerates the data files (anything
    else there stays)."""
    reasons = gate(d)
    if reasons:
        raise TrainingRoomRefused(f"{d.name}: " + "; ".join(reasons))
    level_dir = Path(out_dir) / d.name
    for fname in DATA_FILES:
        p = level_dir / fname
        if p.exists():
            p.unlink()
    level_dir.mkdir(parents=True, exist_ok=True)
    h, w = d.tm.shape
    toml_path = level_lib.write_level_header(
        level_dir, name=d.name, tile_size_m=TILE_SIZE_M,
        comment_lines=(
            f"TRAINING ROOM {d.name} -- GENERATED by "
            f"tools/gen_training_rooms.py (#195 2c) through level_lib "
            f"({w}x{h} tiles).",
            "Regenerate rather than hand-edit: python "
            "tools/gen_training_rooms.py --out <folder>"))
    level_lib.write_tilemap_csv(level_dir, d.tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(level_dir / "diffuse.png")
    level_lib.write_spawn_rooms(toml_path, d.spawn_rooms)
    blocks = level_lib.room_graph_blocks(d.rooms, d.numbered_openings(),
                                         d.grid)
    if d.entities:
        ents = d.entities
        blocks["entity"] = lambda nl: level_lib.format_entity_lines(ents, nl)
    level_lib.write_managed_blocks(toml_path, blocks, write_bak=False)
    level_lib.write_rooms_npy(level_dir, d.grid, d.rooms,
                              d.numbered_openings())
    level_lib.write_match(level_dir, d.spec)
    return level_dir


def lint(name: str, out_dir) -> bool:
    """The level lint on the written level (``level_airtight.check``: the
    flood over the loaded GameMap). Prints its report; True = airtight."""
    from level_airtight import check
    return check(name, levels_dir=str(out_dir))


def write_all(out_dir) -> list:
    """Write every level of :func:`level_names` into ``out_dir``; returns
    the folders."""
    return [write(d, out_dir) for d in drafts()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="the folder the levels are written into (outside "
                         "the repository)")
    ap.add_argument("--no-lint", action="store_true",
                    help="skip the level lint after writing")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if out == ROOT or ROOT in out.parents:
        print(f"refused: {out} is inside the repository; training rooms "
              f"live outside it")
        return 2
    ok = True
    for p in write_all(out):
        print(f"wrote {p}")
        if not args.no_lint:
            ok &= lint(p.name, out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
