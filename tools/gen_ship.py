r"""tools/gen_ship.py -- levelgen v1: a seeded simple ship for the v1 match
(#120; docs/rl_ready_scope_2026-10-29.md section 5 item 2).

Deliberately simple (Erik: "it can be quite simple"); the graph-grammar
levelgen (docs/breach_levelgen_design_v0.1.md, #27) is the long game, not
this. TWO STAGES, split so #27 can replace the first and keep the rest:

STAGE 1 -- the ROOM GRAPH (:func:`plan_graph` -> :class:`ShipGraph`, plain
data): rooms (nodes) with their rectangle, kind and tags (``spawn_team``,
``zombie``, ``fuel``, ``vent``), door / window edges between rooms, and the
mirror pairing. v1's stage 1 is a mirrored binary split:

  1. A hull rectangle inside a vacuum band (top / bottom insets drawn).
  2. A MIDDLE BAND straddling the mirror axis, cut by rows into 2..5
     sections; the one nearest the vertical centre is tagged ``zombie``.
  3. The LEFT WING: an aft SPAWN room at the far end, sized to seat
     ``max_per_team`` under #119's ``formation`` rule (rooms above / below it
     when there is space), the rest binary-split (BSP) into rooms.
  4. The wing MIRRORED onto the right (each room knows its twin), team 1's
     spawn room the reflection of team 0's.
  5. DOOR edges: a Kruskal spanning tree over the rooms' shared walls (each
     door added with its mirror twin), plus a few loop doors; then a couple
     of WINDOW edges; ``fuel`` and ``vent`` tags on rooms.

STAGE 2 -- the EMBEDDING (:func:`embed` -> :class:`Embedding`): graph ->
tiles. Hull, steel everywhere inside, each room's rectangle air, each edge
cut into its shared wall (door = ``MAT_DOOR``, window = ``MAT_GLASS``),
fuel clusters (furniture / wood) against the walls of tagged rooms clear of
doors (placed in one room of a mirror pair and reflected into its twin),
the vents on inner walls of their tagged rooms, one static lamp per room
(render light only), the spawn rooms from the tags. It reads only the graph (and one seeded generator for the clusters).

GATES (:func:`validate`), before anything is written: airtight
(``level_airtight.find_leaks``), A* (``pathfinding.astar``) between the
spawn rooms and from each to the zombie room, every compartment reachable
by a 3x3 unit and exactly the graph's rooms realised, every formation cell
open floor, the grid mirror-symmetric (when the graph is mirrored). A plan
failing a gate is REJECTED and the next sub-seed is tried (attempt + 1, up
to ``MAX_ATTEMPTS``, then :class:`ShipGenError`). Every draw of attempt
``a`` comes from ONE generator, ``Generator(PCG64(SeedSequence([seed,
a])))``.

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md "Level data layer" / "Level
generators"; precedent ``tools/gen_explosion_studio.py``): header,
boundary, tilemap, ``[[spawn_room]]``, ``[[spawn]]``, ``[[entity]]`` and
match.toml all go through ``level_lib``. ``diffuse.png`` is render-only art
(flat colour-coded tiles); no bake.

Importable: :func:`generate` (pure, returns a :class:`ShipPlan`) and
:func:`write` (plan -> level folder).

Run:
    C:/Users/steen/anaconda3/python.exe tools/gen_ship.py --seed 1 \
        --size medium --out levels/ship_demo --vents
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_lib  # noqa: E402
from level_loader import EntityInstance, LightEntry, SpawnEntry  # noqa: E402
from pathfinding import astar  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import (  # noqa: E402
    FORMATION_FOOTPRINT, FORMATION_GAP, FORMATION_PITCH, MatchSpec,
    MatchTeam, SpawnRoom, formation_capacity, placement_cells,
)
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_DOOR, MAT_FURNITURE, MAT_GLASS, MAT_HULL, MAT_STEEL,
    MAT_WOOD,
)

SPACE_CODE = 9
TILE_SIZE_M = 0.333          # the reference tile (the playground's scale)
PX = 16                      # diffuse px per tile
BAND = 3                     # vacuum band outside the hull (tiles)
DOOR_W = 4                   # door width (even: an axis door is centred)
GLASS_W = 2                  # window pane length
MIN_ROOM = 6                 # a room's least interior side (tiles)
MIN_SECTION = 8              # a middle-band section's least height
MAX_ATTEMPTS = 32            # sub-seeds tried before ShipGenError
FUEL_DOOR_CLEAR = 3          # Chebyshev tiles kept free between fuel and a door
P_FUEL_ROOM = 0.6            # chance a non-spawn room is tagged with fuel
Q_CIRC = 16.0                # the vent studio's circulation rate (untuned)
FILTER = "hepa_basic"        # a [filters.*] row (config.toml)
TIME_LIMIT_S = 300.0
LAMP_RGB = (1.0, 0.84, 0.67)  # the explosion studio's warm static lamp
LAMP_INTENSITY = 1.2


@dataclass(frozen=True)
class SizePreset:
    """One ``--size``: the grid, the middle band's width range (even), its
    section-count range, and the BSP leaf size above which a wing room is
    always split."""
    w: int
    h: int
    band_w: tuple            # (lo, hi) inclusive, even values
    sections: tuple          # (lo, hi) inclusive
    max_room: int


SIZES = {
    "small": SizePreset(60, 40, (10, 14), (2, 3), 14),
    "medium": SizePreset(100, 70, (12, 18), (3, 4), 18),
    "large": SizePreset(160, 110, (14, 22), (3, 5), 24),
}


class ShipGenError(ValueError):
    """No plan passed the gates within ``MAX_ATTEMPTS`` sub-seeds, or the
    request can never be seated (refused before any draw)."""


class _Reject(Exception):
    """A stage could not realise this draw; the attempt is rejected."""


# ===========================================================================
# The room graph -- stage 1's product, stage 2's only input
# ===========================================================================

@dataclass
class Room:
    """A graph node: one compartment. ``x, y, w, h`` is its AIR rectangle
    (the geometry annotation stage 2 embeds); ``kind`` names its role;
    ``tags`` carries ``spawn_team`` (int), ``zombie`` (True), ``fuel``
    (number of clusters), ``vent`` ("supply" / "return"); ``mirror`` is its
    twin's id under the graph's mirror (itself on the axis, None if none)."""
    id: int
    x: int
    y: int
    w: int
    h: int
    kind: str = "room"            # "spawn" | "section" | "room"
    tags: dict = field(default_factory=dict)
    mirror: int | None = None

    @property
    def box(self) -> tuple:
        return self.x, self.y, self.x + self.w - 1, self.y + self.h - 1


@dataclass
class Edge:
    """A graph edge between rooms ``a`` and ``b`` (which share a wall):
    ``kind`` "door" or "window"; ``at`` is the opening's first tile along
    the shared wall (None: stage 2 centres it)."""
    a: int
    b: int
    kind: str = "door"
    at: int | None = None


@dataclass
class ShipGraph:
    """Stage 1's product: the ship as rooms + edges + annotations."""
    width: int
    height: int
    hull: tuple                   # (x0, y0, x1, y1) outer hull faces, inclusive
    rooms: list                   # list[Room], rooms[i].id == i
    edges: list                   # list[Edge]
    mirror_x: bool = False        # left-right mirrored about the grid centre
    max_per_team: int = 6
    zombies: int = 0


def shared_wall(a: Room, b: Room):
    """The 1-tile wall two rooms share: ``("v", col, y_lo, y_hi)`` or
    ``("h", row, x_lo, x_hi)`` over the run where both face it, else None."""
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    for (p, q) in ((a, b), (b, a)):
        px0, py0, px1, py1 = p.box
        qx0, qy0, qx1, qy1 = q.box
        if px1 + 2 == qx0:
            lo, hi = max(py0, qy0), min(py1, qy1)
            if lo <= hi:
                return ("v", px1 + 1, lo, hi)
        if py1 + 2 == qy0:
            lo, hi = max(px0, qx0), min(px1, qx1)
            if lo <= hi:
                return ("h", py1 + 1, lo, hi)
    return None


def _opening_tiles(graph: ShipGraph, e: Edge) -> list:
    """The (y, x) tiles an edge opens in its shared wall."""
    wall = shared_wall(graph.rooms[e.a], graph.rooms[e.b])
    if wall is None:
        raise _Reject(f"edge {e.a}-{e.b}: the rooms share no wall")
    orient, line, lo, hi = wall
    n = DOOR_W if e.kind == "door" else GLASS_W
    at = lo + (hi - lo + 1 - n) // 2 if e.at is None else e.at
    if at < lo or at + n - 1 > hi:
        raise _Reject(f"edge {e.a}-{e.b}: opening {at}+{n} leaves {lo}..{hi}")
    if orient == "v":
        return [(y, line) for y in range(at, at + n)]
    return [(line, x) for x in range(at, at + n)]


# ===========================================================================
# Refusal: what a size can never seat
# ===========================================================================

def _spawn_grid(max_per_team: int, interior_h: int) -> tuple[int, int]:
    """(cols, rows) of the spawn room's formation: as square as the hull's
    height allows."""
    r_full = (interior_h + FORMATION_GAP) // FORMATION_PITCH
    r = max(1, min(r_full, int(np.ceil(np.sqrt(max_per_team)))))
    c = int(np.ceil(max_per_team / r))
    return c, r


def _extent(n: int) -> int:
    """Tiles a formation run of ``n`` footprints needs."""
    return n * FORMATION_PITCH - FORMATION_GAP


def check_request(size: str, max_per_team: int, zombies: int) -> None:
    """Refuse, before any draw, a request the size can never seat."""
    if size not in SIZES:
        raise ShipGenError(f"unknown size {size!r} (known: {sorted(SIZES)})")
    if max_per_team < 1:
        raise ShipGenError(f"max_per_team must be >= 1 (got {max_per_team})")
    if zombies < 0:
        raise ShipGenError(f"zombies must be >= 0 (got {zombies})")
    p = SIZES[size]
    interior_h = p.h - 2 * BAND - 2
    interior_w = p.w - 2 * BAND - 2
    c, r = _spawn_grid(max_per_team, interior_h)
    wing = (interior_w - p.band_w[0]) // 2 - 1          # left wing air width
    if _extent(r) > interior_h or _extent(c) + 1 + MIN_ROOM > wing:
        raise ShipGenError(
            f"size {size!r} cannot seat max_per_team = {max_per_team}: a "
            f"{_extent(c)}x{_extent(r)} spawn room does not fit its "
            f"{wing}-tile wing beside a {MIN_ROOM}-tile room")
    if zombies:
        room = SpawnRoom(team=ZOMBIE_TEAM, x=0, y=0, w=p.band_w[1],
                         h=interior_h - (p.sections[0] - 1) * (MIN_SECTION + 1),
                         max_per_team=1)
        if formation_capacity(room) < zombies:
            raise ShipGenError(
                f"size {size!r} cannot seat zombies = {zombies} in its "
                f"middle band (at most {formation_capacity(room)})")


# ===========================================================================
# STAGE 1 (v1): the mirrored binary split -> ShipGraph
# ===========================================================================

def _split_1d(rng, lo: int, hi: int, k: int, min_len: int) -> list:
    """Wall coordinates cutting ``[lo, hi]`` into ``k`` runs of at least
    ``min_len`` (one wall tile between runs); fewer runs if it is short."""
    length = hi - lo + 1
    k = max(1, min(k, (length + 1) // (min_len + 1)))
    extra = length - k * min_len - (k - 1)
    sizes = np.full(k, min_len) + rng.multinomial(extra, np.full(k, 1.0 / k))
    walls, pos = [], lo
    for s in sizes[:-1]:
        pos += int(s)
        walls.append(pos)
        pos += 1
    return walls


def _bsp(rng, box, max_room, out: list) -> None:
    """Binary-split the air box ``(x0, y0, x1, y1)`` (one wall tile per cut)
    until every leaf is at most ``max_room`` or cannot split; leaves to
    ``out``."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0 + 1, y1 - y0 + 1
    can_v, can_h = w >= 2 * MIN_ROOM + 1, h >= 2 * MIN_ROOM + 1
    if not (can_v or can_h) or (w <= max_room and h <= max_room
                                and rng.random() < 0.35):
        out.append(box)
        return
    if can_v and (not can_h or w >= h):
        cx = int(rng.integers(x0 + MIN_ROOM, x1 - MIN_ROOM + 1))
        _bsp(rng, (x0, y0, cx - 1, y1), max_room, out)
        _bsp(rng, (cx + 1, y0, x1, y1), max_room, out)
    else:
        cy = int(rng.integers(y0 + MIN_ROOM, y1 - MIN_ROOM + 1))
        _bsp(rng, (x0, y0, x1, cy - 1), max_room, out)
        _bsp(rng, (x0, cy + 1, x1, y1), max_room, out)


class _Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b) -> None:
        self.p[self.find(a)] = self.find(b)


def _at(rng, lo: int, hi: int, n: int) -> int:
    """A random opening start in ``[lo, hi]`` keeping one wall tile from
    each end when the run allows."""
    a, b = (lo + 1, hi - n) if hi - lo + 1 >= n + 2 else (lo, hi - n + 1)
    return int(rng.integers(a, b + 1))


def plan_graph(rng, size: str, max_per_team: int, zombies: int,
               vents: bool) -> ShipGraph:
    """v1's stage 1: the mirrored binary split as a :class:`ShipGraph`."""
    p = SIZES[size]
    W, H = p.w, p.h
    inset_top, inset_bot = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    hull = (BAND, BAND + inset_top, W - 1 - BAND, H - 1 - BAND - inset_bot)
    ix0, iy0, iy1 = hull[0] + 1, hull[1] + 1, hull[3] - 1
    interior_h = iy1 - iy0 + 1

    # The middle band and its sections.
    lo, hi = p.band_w
    wm = 2 * int(rng.integers(lo // 2, hi // 2 + 1))
    cx0 = W // 2 - wm // 2
    cx1 = cx0 + wm - 1
    k = int(rng.integers(p.sections[0], p.sections[1] + 1))
    cuts = _split_1d(rng, iy0, iy1, k, MIN_SECTION)
    sec_boxes = [(cx0, a, cx1, b) for a, b in
                 zip([iy0] + [c + 1 for c in cuts], [c - 1 for c in cuts] + [iy1])]

    # The left wing: the aft spawn slab, then BSP.
    wing_x1 = cx0 - 2
    c, r = _spawn_grid(max_per_team, interior_h)
    sw = min(_extent(c) + int(rng.integers(0, 3)),
             wing_x1 - ix0 + 1 - 1 - MIN_ROOM)
    if sw < _extent(c):
        raise _Reject("the band left no room for the spawn slab")
    sh = min(interior_h, _extent(r) + int(rng.integers(0, 4)))
    slab_x1 = ix0 + sw - 1
    sy0 = iy0 + int(rng.integers(0, interior_h - sh + 1))
    left = []                                   # (box, kind)
    if sy0 - iy0 - 1 >= MIN_ROOM:
        left.append(((ix0, iy0, slab_x1, sy0 - 2), "room"))
    else:
        sh, sy0 = sh + (sy0 - iy0), iy0         # absorb the sliver
    if iy1 - (sy0 + sh - 1) - 1 >= MIN_ROOM:
        left.append(((ix0, sy0 + sh + 1, slab_x1, iy1), "room"))
    else:
        sh = iy1 - sy0 + 1
    spawn_box = (ix0, sy0, slab_x1, sy0 + sh - 1)
    left.insert(0, (spawn_box, "spawn"))
    leaves: list = []
    _bsp(rng, (slab_x1 + 2, iy0, wing_x1, iy1), p.max_room, leaves)
    left += [(b, "room") for b in leaves]

    # Nodes: left wing, middle sections (on the axis), right twins.
    rooms = []

    def add(box, kind, mirror=None):
        x0, y0, x1, y1 = box
        rooms.append(Room(id=len(rooms), x=x0, y=y0, w=x1 - x0 + 1,
                          h=y1 - y0 + 1, kind=kind, mirror=mirror))
        return rooms[-1]

    nl = len(left)
    for box, kind in left:
        add(box, kind, mirror=len(rooms) + nl + len(sec_boxes))
    for box in sec_boxes:
        add(box, "section", mirror=len(rooms))
    for i, (box, kind) in enumerate(left):
        x0, y0, x1, y1 = box
        add((W - 1 - x1, y0, W - 1 - x0, y1), kind, mirror=i)
    rooms[0].tags["spawn_team"] = 0
    rooms[rooms[0].mirror].tags["spawn_team"] = 1
    centre = (H - 1) / 2.0
    sections = [rm for rm in rooms if rm.kind == "section"]
    zsec = min(sections, key=lambda s: (abs(s.y + (s.h - 1) / 2.0 - centre), s.id))
    if zombies:
        zsec.tags["zombie"] = True

    # Edges: Kruskal over the shared walls, every door with its twin.
    def twin(a, b):
        return tuple(sorted((rooms[a].mirror, rooms[b].mirror)))

    cands = []
    for a in range(len(rooms)):
        for b in range(a + 1, len(rooms)):
            wall = shared_wall(rooms[a], rooms[b])
            if wall and wall[3] - wall[2] + 1 >= DOOR_W + 2 \
                    and (a, b) <= twin(a, b):
                cands.append((a, b, wall))
    uf = _Union(len(rooms))
    edges, unused = [], []

    def add_edge(a, b, wall, kind, n):
        orient, _line, lo_, hi_ = wall
        if (a, b) == twin(a, b):                       # on the axis
            at = W // 2 - n // 2
        else:
            at = _at(rng, lo_, hi_, n)
        edges.append(Edge(a, b, kind, at))
        ta, tb = twin(a, b)
        if (ta, tb) != (a, b):
            edges.append(Edge(ta, tb, kind,
                              at if orient == "v" else W - at - n))

    for i in rng.permutation(len(cands)):
        a, b, wall = cands[int(i)]
        ta, tb = twin(a, b)
        if uf.find(a) != uf.find(b) or uf.find(ta) != uf.find(tb):
            uf.union(a, b)
            uf.union(ta, tb)
            add_edge(a, b, wall, "door", DOOR_W)
        else:
            unused.append(cands[int(i)])
    n_extra = int(rng.integers(0, max(1, len(rooms) // 4) + 1))
    rest = []
    for (a, b, wall) in unused:
        if n_extra > 0 and rng.random() < 0.5:
            add_edge(a, b, wall, "door", DOOR_W)
            n_extra -= 1
        else:
            rest.append((a, b, wall))
    want = int(rng.integers(1, 3))
    for i in rng.permutation(len(rest))[:want]:
        a, b, wall = rest[int(i)]
        add_edge(a, b, wall, "window", GLASS_W)

    # Tags: fuel (a pair shares the draw), vents.
    for rm in rooms:
        if rm.mirror is not None and rm.mirror < rm.id:
            rm.tags.update({k_: v for k_, v in rooms[rm.mirror].tags.items()
                            if k_ == "fuel"})
            continue
        if rm.kind == "spawn" or rm.tags.get("zombie"):
            continue
        if rng.random() < P_FUEL_ROOM:
            rm.tags["fuel"] = int(rng.integers(1, 3))
    if vents:
        if len(sections) >= 2:
            pair = (sections[0], sections[-1])
        else:
            near = max((rm for rm in rooms[:nl] if rm.kind == "room"),
                       key=lambda rm: (rm.x + rm.w, -rm.id))
            pair = (near, rooms[near.mirror])
        pair[0].tags["vent"] = "supply"
        pair[1].tags["vent"] = "return"
    return ShipGraph(width=W, height=H, hull=hull, rooms=rooms, edges=edges,
                     mirror_x=True, max_per_team=max_per_team,
                     zombies=zombies)


# ===========================================================================
# STAGE 2: the embedding -- ShipGraph -> tiles, spawn rooms, entities
# ===========================================================================

@dataclass
class Embedding:
    """Stage 2's product."""
    tilemap: np.ndarray               # (h, w) int32 v2 codes
    spawn_rooms: list                 # list[SpawnRoom]: team 0, team 1, zombies
    entities: list                    # duct + vents ([] without vents)
    lights: list                      # one ceiling lamp per room (LightEntry)
    doors: int
    windows: int
    fuel_tiles: int


def _reflect(box, w):
    x0, y0, x1, y1 = box
    return (w - 1 - x1, y0, w - 1 - x0, y1)


def _place_fuel(rng, tm, room: Room, graph: ShipGraph) -> None:
    """``room.tags['fuel']`` clusters against its walls, clear of doors; in
    a mirrored graph placed once per pair (an axis room: its left half) and
    reflected into the twin."""
    W = graph.width
    x0, y0, x1, y1 = room.box
    if graph.mirror_x and room.mirror == room.id:
        x1 = min(x1, W // 2 - 1)
    door = tm == MAT_DOOR
    for _ in range(int(room.tags.get("fuel", 0))):
        mat = MAT_FURNITURE if rng.random() < 0.6 else MAT_WOOD
        for _try in range(8):
            cw, ch = int(rng.integers(2, 4)), int(rng.integers(1, 3))
            if rng.random() < 0.5:
                cw, ch = ch, cw
            side = int(rng.integers(0, 4))
            along_x = int(rng.integers(x0, max(x0, x1 - cw + 1) + 1))
            along_y = int(rng.integers(y0, max(y0, y1 - ch + 1) + 1))
            cx, cy = {0: (along_x, y0), 1: (along_x, y1 - ch + 1),
                      2: (x0, along_y), 3: (x1 - cw + 1, along_y)}[side]
            if cx < x0 or cy < y0 or cx + cw - 1 > x1 or cy + ch - 1 > y1:
                continue
            ys, xs = slice(cy, cy + ch), slice(cx, cx + cw)
            if not (tm[ys, xs] == MAT_AIR).all():
                continue
            c = FUEL_DOOR_CLEAR
            if door[max(0, cy - c):cy + ch + c, max(0, cx - c):cx + cw + c].any():
                continue
            boxes = [(cx, cy, cx + cw - 1, cy + ch - 1)]
            if graph.mirror_x and room.mirror is not None:
                boxes.append(_reflect(boxes[0], W))
            for (bx0, by0, bx1, by1) in boxes:
                tm[by0:by1 + 1, bx0:bx1 + 1] = mat
            break


def _vent_site(tm, room: Room, axis2: int):
    """A steel tile on the room's boundary, the room's air in front, no
    door / glass within one tile, closest to the axis (then topmost):
    ``(x, y, facing)`` or None. ``axis2`` is twice the axis coordinate."""
    x0, y0, x1, y1 = room.box
    cands = [(x, y0 - 1, "s") for x in range(x0, x1 + 1)] + \
            [(x, y1 + 1, "n") for x in range(x0, x1 + 1)] + \
            [(x0 - 1, y, "e") for y in range(y0, y1 + 1)] + \
            [(x1 + 1, y, "w") for y in range(y0, y1 + 1)]
    best = None
    for (x, y, facing) in cands:
        if tm[y, x] != MAT_STEEL:
            continue
        dy, dx = {"n": (-1, 0), "s": (1, 0), "e": (0, 1), "w": (0, -1)}[facing]
        if tm[y + dy, x + dx] != MAT_AIR:
            continue
        if np.isin(tm[y - 1:y + 2, x - 1:x + 2], (MAT_DOOR, MAT_GLASS)).any():
            continue
        key = (abs(2 * x - axis2), y, x)
        if best is None or key < best[0]:
            best = (key, (x, y, facing))
    return None if best is None else best[1]


def _build_vents(tm, graph: ShipGraph) -> list:
    """One HEPA duct; a wall vent in each room tagged ``vent``."""
    tagged = sorted((rm for rm in graph.rooms if "vent" in rm.tags),
                    key=lambda rm: (rm.tags["vent"] != "supply", rm.id))
    if not tagged:
        return []
    did = "duct_main"
    ents = [EntityInstance(id=did, class_name="duct", ordinal=0,
                           fields={"filter": FILTER}, authored_keys=("filter",))]
    keys = ("x", "y", "mount", "facing", "role", "duct", "q_circ")
    for i, rm in enumerate(tagged, start=1):
        site = _vent_site(tm, rm, graph.width - 1)
        if site is None:
            raise _Reject(f"vents: no wall site in room {rm.id}")
        x, y, facing = site
        role = rm.tags["vent"]
        ents.append(EntityInstance(
            id=f"vent_{role}", class_name="vent", ordinal=i,
            fields={"x": int(x), "y": int(y), "mount": "wall",
                    "facing": facing, "role": role, "duct": did,
                    "q_circ": Q_CIRC},
            authored_keys=keys))
    return ents


def embed(graph: ShipGraph, rng) -> Embedding:
    """Stage 2: the graph as tiles (+ spawn rooms and vent entities)."""
    W, H = graph.width, graph.height
    tm = np.full((H, W), SPACE_CODE, dtype=np.int32)
    hx0, hy0, hx1, hy1 = graph.hull
    tm[hy0:hy1 + 1, hx0:hx1 + 1] = MAT_HULL
    tm[hy0 + 1:hy1, hx0 + 1:hx1] = MAT_STEEL
    for rm in graph.rooms:
        x0, y0, x1, y1 = rm.box
        if x0 <= hx0 or y0 <= hy0 or x1 >= hx1 or y1 >= hy1:
            raise _Reject(f"room {rm.id} leaves the hull")
        tm[y0:y1 + 1, x0:x1 + 1] = MAT_AIR
    doors = windows = 0
    for e in graph.edges:
        mat = MAT_DOOR if e.kind == "door" else MAT_GLASS
        for (y, x) in _opening_tiles(graph, e):
            tm[y, x] = mat
        doors += e.kind == "door"
        windows += e.kind == "window"
    for rm in graph.rooms:
        if rm.tags.get("fuel") and (rm.mirror is None or rm.mirror >= rm.id
                                    or not graph.mirror_x):
            _place_fuel(rng, tm, rm, graph)
    ents = _build_vents(tm, graph)
    by_team = {rm.tags["spawn_team"]: rm for rm in graph.rooms
               if "spawn_team" in rm.tags}
    rooms = [SpawnRoom(team=t, x=by_team[t].x, y=by_team[t].y, w=by_team[t].w,
                       h=by_team[t].h, max_per_team=graph.max_per_team)
             for t in sorted(by_team)]
    if graph.zombies:
        z = next(rm for rm in graph.rooms if rm.tags.get("zombie"))
        rooms.append(SpawnRoom(team=ZOMBIE_TEAM, x=z.x, y=z.y, w=z.w, h=z.h,
                               max_per_team=graph.zombies))
    fuel = int(np.isin(tm, (MAT_FURNITURE, MAT_WOOD)).sum())
    lights = [LightEntry(x=rm.x + rm.w / 2.0, y=rm.y + rm.h / 2.0,
                         color=LAMP_RGB, intensity=LAMP_INTENSITY,
                         range=16.0, kind="static") for rm in graph.rooms]
    return Embedding(tilemap=tm, spawn_rooms=rooms, entities=ents,
                     lights=lights, doors=doors, windows=windows,
                     fuel_tiles=fuel)


# ===========================================================================
# Gates
# ===========================================================================

_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        from simulation.materials import MaterialTable
        _TABLE = MaterialTable.from_config()
    return _TABLE


def _masks(tm: np.ndarray):
    """``(material, is_vacuum, solid, walkable)`` -- the loader's own CSV
    decode and the material table's columns, so the gates cannot disagree
    with what the sim reads (solid = permeability 0, walkable = mobility >
    0, the GameMap rules)."""
    from level_loader import materials_from_tilemap
    mat, vac = materials_from_tilemap(tm, "2")
    tbl = _table()
    solid = np.asarray(tbl.permeability)[mat] <= 0.0
    walk = np.asarray(tbl.mobility)[mat] > 0
    return mat, np.asarray(vac, dtype=bool), solid, walk


def compartments(tm: np.ndarray):
    """``(labels, n)`` of the ship's compartments: the regions bounded by
    the structure (hull, steel, doors, glass) -- fuel is a room's contents,
    so a pocket of air behind a crate belongs to its room."""
    from scipy import ndimage
    return ndimage.label(np.isin(tm, (MAT_AIR, MAT_FURNITURE, MAT_WOOD)))


def _centre(room: SpawnRoom) -> tuple[int, int]:
    """The footprint top-left nearest the room's centre."""
    fp = FORMATION_FOOTPRINT
    return room.x + (room.w - fp) // 2, room.y + (room.h - fp) // 2


def validate(tm: np.ndarray, rooms: list, *, n_rooms: int | None = None,
             mirrored: bool = True) -> list:
    """The gates; returns the failure reasons (empty = pass). ``rooms``:
    the spawn rooms (team 0, team 1[, zombies]); ``n_rooms``: the graph's
    node count, which the tiles must realise exactly."""
    from level_airtight import find_leaks
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy import ndimage
    reasons = []
    mat, vac, solid, walk = _masks(tm)
    h, w = tm.shape
    _, vac_leak, edge_leak = find_leaks(solid, vac)
    if vac_leak or edge_leak:
        reasons.append(f"airtight: {len(vac_leak)} vacuum / {len(edge_leak)} "
                       f"edge leak cells")
    if mirrored and not np.array_equal(tm, tm[:, ::-1]):
        reasons.append("symmetry: the grid is not left-right mirrored")
    fp = FORMATION_FOOTPRINT
    for r in rooms:
        for (cx, cy) in placement_cells(r, r.max_per_team):
            foot = (slice(cy, cy + fp), slice(cx, cx + fp))
            if (mat[foot] != MAT_AIR).any() or vac[foot].any():
                reasons.append(f"floor: team {r.team} cell ({cx},{cy})")
                break
    # 3x3 footprint passability (GameMap.is_passable_block's rule).
    ok = sliding_window_view(walk & ~vac, (fp, fp)).all(axis=(2, 3))

    def blocked(x, y):
        return not ok[y, x]

    pairs = [(rooms[0], rooms[1])] + [(a, rooms[2]) for a in rooms[:2]
                                      if len(rooms) > 2]
    for a, b in pairs:
        (ax, ay), (bx, by) = _centre(a), _centre(b)
        if not astar(ax, ay, bx, by, blocked, w, h):
            reasons.append(f"reach: no A* path team {a.team} -> {b.team}")
    # Every compartment reachable by a unit: flood the footprint grid from
    # team 0's centre; every compartment must hold a covered tile.
    clab, cn = compartments(tm)
    if n_rooms is not None and cn != n_rooms:
        reasons.append(f"rooms: {cn} compartments for {n_rooms} graph rooms")
    flab, _ = ndimage.label(ok)
    ax, ay = _centre(rooms[0])
    if ok[ay, ax]:
        reach = flab == flab[ay, ax]
        cover = np.zeros((h, w), dtype=bool)     # tiles a reachable foot covers
        rh, rw = reach.shape
        for dy in range(fp):
            for dx in range(fp):
                cover[dy:dy + rh, dx:dx + rw] |= reach
        missing = cn - len(np.unique(clab[cover & (clab > 0)]))
        if missing:
            reasons.append(f"compartments: {missing} not reachable by a unit")
    return reasons


# ===========================================================================
# Public API
# ===========================================================================

@dataclass
class ShipPlan:
    """A generated ship, validated, not yet written."""
    seed: int
    size: str
    attempt: int                      # the sub-seed that passed (0 = first)
    graph: ShipGraph                  # stage 1
    embedding: Embedding              # stage 2
    rejected: list = field(default_factory=list)   # (attempt, reasons)

    @property
    def name(self) -> str:
        return default_name(self.seed, self.size)

    @property
    def tilemap(self) -> np.ndarray:
        return self.embedding.tilemap

    @property
    def spawn_rooms(self) -> list:
        return self.embedding.spawn_rooms

    @property
    def entities(self) -> list:
        return self.embedding.entities

    @property
    def max_per_team(self) -> int:
        return self.graph.max_per_team

    @property
    def zombies(self) -> int:
        return self.graph.zombies

    def match_spec(self) -> MatchSpec:
        team = MatchTeam(n=min(4, self.max_per_team), loadout="rifleman",
                         env_profile="default", controller="bots")
        return MatchSpec(teams=(team, team), seed=self.seed,
                         zombies=self.zombies, time_limit_s=TIME_LIMIT_S,
                         objective="eliminate")


def default_name(seed: int, size: str) -> str:
    return f"ship_s{seed}_{size}"


def generate(seed: int, size: str = "medium", *, max_per_team: int = 6,
             zombies: int = 4, vents: bool = False) -> ShipPlan:
    """A validated ship for ``seed`` (pure: no files). Tries sub-seeds
    ``attempt = 0, 1, ...``: stage 1, stage 2, the gates; the first plan
    that passes is returned."""
    check_request(size, max_per_team, zombies)
    rejected = []
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([int(seed), attempt])))
        try:
            graph = plan_graph(rng, size, max_per_team, zombies, vents)
            emb = embed(graph, rng)
        except _Reject as e:
            rejected.append((attempt, [str(e)]))
            continue
        reasons = validate(emb.tilemap, emb.spawn_rooms,
                           n_rooms=len(graph.rooms), mirrored=graph.mirror_x)
        if not reasons:
            return ShipPlan(seed=int(seed), size=size, attempt=attempt,
                            graph=graph, embedding=emb, rejected=rejected)
        rejected.append((attempt, reasons))
    raise ShipGenError(f"seed {seed} size {size!r}: no plan passed the gates "
                       f"in {MAX_ATTEMPTS} attempts; last: {rejected[-1][1]}")


# ===========================================================================
# Writing
# ===========================================================================

PALETTE = {
    SPACE_CODE:    (6, 6, 12),
    MAT_AIR:       (34, 36, 42),
    MAT_HULL:      (96, 102, 116),
    MAT_STEEL:     (120, 124, 132),
    MAT_DOOR:      (150, 120, 60),
    MAT_GLASS:     (110, 170, 200),
    MAT_WOOD:      (120, 84, 46),
    MAT_FURNITURE: (96, 71, 41),
}
AIR_ALT = (30, 32, 38)
ROOM_TINT = {0: (30, 40, 62), 1: (62, 34, 34), ZOMBIE_TEAM: (34, 56, 36)}


def build_diffuse(plan: ShipPlan) -> Image.Image:
    """Flat colour-coded tiles, spawn rooms tinted, darkened seams.
    RENDER-ONLY; the CSV is the physics truth."""
    tm = plan.tilemap
    h, w = tm.shape
    tile = np.zeros((h, w, 3), dtype=np.uint8)
    for code, c in PALETTE.items():
        tile[tm == code] = c
    yy, xx = np.mgrid[0:h, 0:w]
    tile[(tm == MAT_AIR) & ((xx + yy) % 2 == 1)] = AIR_ALT
    for r in plan.spawn_rooms:
        sub = tm[r.y:r.y + r.h, r.x:r.x + r.w] == MAT_AIR
        tile[r.y:r.y + r.h, r.x:r.x + r.w][sub] = ROOM_TINT[r.team]
    img = np.repeat(np.repeat(tile, PX, axis=0), PX, axis=1)
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.uint8)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.uint8)
    return Image.fromarray(img)


def _authored_spawns(plan: ShipPlan) -> list:
    """``[[spawn]]`` rows so the single-player game (main.py) shows the
    match's start: team 0's squad and the zombies (legacy team 1 at the
    loader door) on their formation cells. A match never reads these."""
    out = []
    spec = plan.match_spec()
    for k, (x, y) in enumerate(placement_cells(plan.spawn_rooms[0],
                                               spec.teams[0].n)):
        out.append(SpawnEntry(name=f"Marine{k + 1}", team=0, x=x, y=y,
                              footprint=FORMATION_FOOTPRINT))
    if plan.zombies:
        for k, (x, y) in enumerate(placement_cells(plan.spawn_rooms[2],
                                                   plan.zombies)):
            out.append(SpawnEntry(name=f"Zomb{k + 1}", team=1, x=x, y=y,
                                  footprint=FORMATION_FOOTPRINT))
    return out


DATA_FILES = ("level.toml", "tilemap.csv", "diffuse.png", "match.toml")


def write(plan: ShipPlan, out_dir) -> Path:
    """Write ``plan`` as a level folder through ``level_lib``; returns the
    folder. Regenerates the data files (anything else there stays)."""
    out_dir = Path(out_dir)
    for name in DATA_FILES:
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = plan.tilemap.shape
    comments = (
        f"SHIP {plan.name} -- GENERATED by tools/gen_ship.py (#120, levelgen",
        f"v1) through level_lib: seed {plan.seed}, size {plan.size} "
        f"({w}x{h}), sub-seed attempt {plan.attempt}.",
        "Regenerate rather than hand-edit:",
        f"  C:/Users/steen/anaconda3/python.exe tools/gen_ship.py --seed "
        f"{plan.seed} --size {plan.size} --max-per-team {plan.max_per_team} "
        f"--zombies {plan.zombies}" + (" --vents" if plan.entities else ""),
        "",
        "v2 codes ARE canon material ids (src/simulation/materials.py):",
        "  0=air 1=hull 2=wood 3=door 4=steel 5=glass 6=furniture 9=SPACE",
    )
    toml_path = level_lib.write_level_header(
        out_dir, name=plan.name, tile_size_m=TILE_SIZE_M,
        comment_lines=comments)
    level_lib.write_boundary_field(toml_path, "space")
    level_lib.write_tilemap_csv(out_dir, plan.tilemap, csv_bak=False)
    build_diffuse(plan).save(out_dir / "diffuse.png")
    level_lib.write_spawn_rooms(toml_path, plan.spawn_rooms)
    spawns = _authored_spawns(plan)
    lights = plan.embedding.lights
    blocks = {"spawn": lambda nl: level_lib.format_spawn_lines(spawns, nl),
              "light": lambda nl: level_lib.format_light_lines(lights, nl)}
    if plan.entities:
        ents = plan.entities
        blocks["entity"] = lambda nl: level_lib.format_entity_lines(ents, nl)
    level_lib.write_managed_blocks(toml_path, blocks, write_bak=False)
    level_lib.write_match(out_dir, plan.match_spec())
    return out_dir


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="gen_ship", description="Generate a seeded v1 ship level.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None,
                    help="level folder (default levels/ship_s<seed>_<size>)")
    ap.add_argument("--size", choices=sorted(SIZES), default="medium")
    ap.add_argument("--max-per-team", type=int, default=6)
    ap.add_argument("--zombies", type=int, default=4)
    ap.add_argument("--vents", action="store_true",
                    help="place one HEPA duct with a supply + return vent")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(sys.argv[1:] if argv is None else argv)
    plan = generate(a.seed, a.size, max_per_team=a.max_per_team,
                    zombies=a.zombies, vents=a.vents)
    out = Path(a.out) if a.out else ROOT / "levels" / plan.name
    write(plan, out)
    h, w = plan.tilemap.shape
    e = plan.embedding
    print(f"wrote {out}  ({w}x{h} @ {TILE_SIZE_M} m, attempt {plan.attempt}, "
          f"{len(plan.graph.rooms)} rooms, {e.doors} doors, {e.windows} "
          f"windows, {e.fuel_tiles} fuel tiles, "
          f"vents {'yes' if plan.entities else 'no'})")
    for r in plan.spawn_rooms:
        print(f"  spawn room team {r.team}: ({r.x},{r.y}) {r.w}x{r.h} "
              f"max {r.max_per_team}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
