r"""tools/ship_layout.py -- the LAYOUT stage (stage 1.5) of family B (#136;
docs/levelgen_layout_stage_2026-10-31.md with its §13 amendments and its
"As built" section, docs/levelgen_archetype_recipe_2026-10-31.md §2, §3, §6
and §7.1).

ONE MODULE IS THE SKELETON'S HOME (archetype doc §7.1). It holds

- the skeleton's GRAPH side, which the grammar (``tools/ship_grammar.py``)
  imports: its quotient rooms by role, its axiom edge kinds, the band's
  section chain, what each archetype ``on`` anchor class may act on;
- the skeleton's GEOMETRY: :func:`layout` turns a ``QuotientGraph`` (one
  wing + the axis band, rooms with size RANGES and declared edges) into
  rectangles inside the hull, reflects the wing and emits the
  ``gen_ship.ShipGraph`` that ``gen_ship.embed`` consumes unchanged;
- the FEASIBILITY arithmetic (:func:`layout_feasibility`,
  :func:`department_feasibility`, :func:`maxima_warning`): the same packing
  sets the layout draws from, asked without a draw -- by the grammar at the
  end of every attempt (layout doc §13 A5), by the archetype loader per
  (size, profile) and by the recipe loader per department;
- the recipe path's :func:`check_request` and :func:`acceptance` (the
  layout's bot-walkable A* and the archetype's ``[validation]`` gates).

Nothing here stamps a tile, adds an edge, picks a room or drops an
application: a miss is ``gen_ship._Reject`` and the attempt rerolls.

THE FREIGHTER SKELETON (layout doc §2, §13 A1 / A2), as a graph on the
QUOTIENT half (one wing + the axis band):

- ``slab`` (``kind = "spawn"``, ``spawn_team`` 0) -- ``passageway`` (ONE
  wide corridor) -- ``lobby`` (ONE wide room at the band end of the wing),
  joined by doors;
- the BAND: ``section`` rooms on the axis, fore to aft the main-cell
  sections then the hub department's BLOCK (its ``fore`` room, its hub, its
  ``aft`` room); consecutive main sections joined by a ``door``, the last
  main section and the block by a ``bulkhead``; the lobby's ``door`` goes to
  the aft-most main section (which carries the horde);
- the hub department's entry: the AIRLOCK ``(a = lobby, b = hub)`` -- the
  vestibule carved on the LOBBY's side (A1).

A 3-cycle all of whose pairs are skeleton contacts (the lobby, the aft-most
main section and the block's first section) is a T-junction the skeleton
DRAWS (the lobby's band wall runs along both stacked sections): the
grammar's no-triangle invariant exempts exactly that, never a triangle a
move or a department makes.

THE GEOMETRY, on the left wing (fore = low y; the right wing is its
reflection)::

     hull
     +---------------------------------------------------------+
     |          hull strip (flank pair, 3 across)        |      |
     |  S  +------------------------------------+  L  +--+      |
     |  L  | row A: departments, stubs up       |  O  | band    |
     |  A  +------------------------------------+  B  | main    |
     |  B  |   passageway (one wide corridor)   |  B  | sections|
     |     +------------------------------------+  Y  | then the|
     |     | row B: departments, stubs down     | [v] | block   |
     +---------------------------------------------------------+

A wing DEPARTMENT is a rectangle of its row (x-extent drawn, depth the
row's) packed by :func:`pack_department` in a local frame ``u`` (along the
row, 0 at the slab end) by ``v`` (0 at the passageway, growing to the hull):
its STUB (a strip of ``across`` running ``along`` v), COLUMNS of side rooms
on one or both sides of the stub (each room touches its stub along its whole
``along``), the END ZONE across the hull end holding the END ROOM (it covers
every stub's end) and the BEHIND rooms: a side room's behind child sits in
the end zone past its parent's hull-end wall (the corner slot beside the end
room; the parent is the last room of an outer column), an end room's behind
child beside it in a free corner slot or stacked behind it against the
hull. A retreat loop's second stub is a second strip with the end room
spanning both (two-stub). Every side room is exactly its column's depth, the
end room reaches the department's edges, a stub with no column on a side
stands against the edge. Slack is steel in FOUR places only, each at most
:data:`VOID_MAX` tiles and none adjacent to another: between two departments
of a row, at a column's far end (none under a corner child), past a slot
room toward the hull, between two stubs with no column between them -- so
two floors are never more than a wall and VOID_MAX apart.

EVERY DRAW IS CONDITIONED (layout doc §13 A4): the feasible sets are
enumerated exactly (integers, closed form per department configuration) and
each draw is uniform over the values that leave a feasible completion, so a
graph :func:`layout_feasibility` accepts always packs; what can still
reject an attempt is stage 2 (a vent or floor site, a fuel cluster in a
doorway) and the gates.

DRAW ORDER of one attempt, ONE stream (``SeedSequence([seed, 1, g, a])``,
which ``embed`` continues): inset top, inset bottom; band width ``wm``;
passageway ``across``; lobby x; slab x; the band (block heights fore to
aft, the main sections' cuts, the ring's side split); row A's then row B's
configuration; row A's depth; per row A, B: the departments' x-extents and
voids; per department in row order: stub ``across`` and column depths, the
end-zone depth, each column's room lengths, the end room's reach and its
stacked child, the slot rooms; then every opening's ``at`` in the quotient
graph's edge order. No set iteration; the right wing is a reflection.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

import gen_ship as G
from gen_ship import (AIRLOCK_W, BAND, DOOR_W, FORMATION_FOOTPRINT,
                      FORMATION_GAP, FORMATION_PITCH, MIN_ROOM, MIN_SECTION,
                      SIZES, VESTIBULE_D)

# ---- the skeleton's quotient rooms, by role (layout doc §2) ---------------
SLAB = "slab"
PASSAGEWAY = "passageway"
LOBBY = "lobby"
SECTION = "section"
ROLES = (SLAB, PASSAGEWAY, LOBBY, SECTION)

# ---- rooms the grammar's cycle realisations and budgets make, by role ----
# (what the layout maps to a primitive, layout doc §4; it fails loudly on an
# unknown role)
STUB2 = "stub2"                # retreat loop: the second stub (two-stub, §4.1)
FLANK_STRIP = "flank_strip"    # flank pair: the narrow hull strip (§4.2, A3)
RING_FORE = "ring_fore"        # maintenance ring: the axis room fore of the hub
RING_AFT = "ring_aft"          # ... aft of the hub
RING_SIDE = "ring_side"        # ... a side room (a twin pair in the band)
POCKET = "pocket"              # pocket: a side room on a stub or a behind room
SECRET = "secret"              # the secret annex, behind its host
MADE_ROLES = (STUB2, FLANK_STRIP, RING_FORE, RING_AFT, RING_SIDE, POCKET,
              SECRET)

# ---- the skeleton's closed-form numbers (graph side) ----------------------
# A1: the engineering airlock's vestibule stands in the lobby: VESTIBULE_D + 1
# deep plus a footprint in front of the inner door.
LOBBY_X_MIN = VESTIBULE_D + 1 + FORMATION_FOOTPRINT          # 9
RING_BAND_W_MIN = 20           # §4.4: ring 6 + 1 + hub + 1 + ring 6
RING_ROOMS_MAX = 6             # §4.4: rooms > 6 are refused in v1
RING_DEPTH = 6                 # §4.4: a ring room's depth
# §4.3: the bypass's wall demand on the hub section's height.
BYPASS_HUB_H_MIN = 1 + AIRLOCK_W + 1 + 1 + DOOR_W + 1         # 13
OWN_STUB_ACROSS_MIN = AIRLOCK_W + 2                           # 7 (§4)
FLANK_STRIP_ACROSS = 3         # §4.2: the narrow leg's floor
POCKET_SIZE = ((MIN_ROOM, 12), (MIN_ROOM, 16))  # (w, h): a one-door room

# ---- the geometry's numbers ------------------------------------------------
VOID_MAX = 3                   # §3: a steel void is at most this thick
CLEAR = 1                      # §1: tiles kept each side of an opening
STRIP_COST = FLANK_STRIP_ACROSS + 1   # the strip and its wall: a row loses 4
INSETS = (0, 1, 2)             # the hull's top / bottom inset (plan_graph's)
SLAB_JITTER = 2                # slab x = _extent(c) + 0..2 (plan_graph's)
# A promoted stub's first PROMOTED_CLEAR tiles hold the entry airlock's
# vestibule and inner wall: a side room's opening is cut beyond them.
PROMOTED_CLEAR = VESTIBULE_D + 1
# An exit room carrying a hazard charge or a heat source has doors in two
# FACING walls (its stub's and the slab's); gen_ship._floor_site keeps
# FUEL_DOOR_CLEAR off both, so the room is at least 2 * FUEL_DOOR_CLEAR + 1
# deep.
EXIT_SITE_W = 2 * G.FUEL_DOOR_CLEAR + 1
ROW_A, ROW_B = "A", "B"
ROW_NAMES = (ROW_A, ROW_B)
_INF = 10 ** 6

# ---- anchor classes (archetype doc §3) ------------------------------------
# What a ship-level [[apply]] entry's `on` may name. `cap`: the closed-form
# realisable maximum the profiles' shifts are capped at (archetype doc §1.1);
# "departments_with_end" = one per wing department instance that has an
# `end` room, "one" = one per (quotient) ship, "slack" = no closed form here
# (bounded by the layout's packing slack: `layout_feasibility` on the graph).
# `exclusive`: a group of classes a ship realises at most `EXCLUSIVE_MAX`
# applications of, together (the ring and the bypass, archetype doc §3).
DEPARTMENT_ENTRY = "department-entry"
SLAB_LOBBY = "slab-lobby"
HUB = "<hub>"                  # spelled "<hub department>.<hub room id>"
LOBBY_HUB = "lobby-<hub>"      # spelled "lobby-<hub room id>"
DEPARTMENT_ROOM = "department-room"
SAME_CELL_ROOMS = "same-cell-rooms"


@dataclass(frozen=True)
class AnchorClass:
    key: str
    move: str                  # the move the class realises (recipe doc §0.1)
    candidates: str            # what it may act on, in graph terms
    realised_as: str           # the layout primitive (layout doc §4)
    cap: str                   # "departments_with_end" | "one" | "slack"
    exclusive: str | None = None


ANCHORS = {
    DEPARTMENT_ENTRY: AnchorClass(
        DEPARTMENT_ENTRY, "second_leg",
        "a wing department instance with an attach = \"end\" room",
        "two-stub (§4.1)", "departments_with_end"),
    SLAB_LOBBY: AnchorClass(
        SLAB_LOBBY, "second_leg", "the wing's slab and lobby",
        "hull strip (§4.2, §13 A3)", "one"),
    HUB: AnchorClass(
        HUB, "ring", "the hub department's hub section",
        "ring in the band (§4.4)", "one", exclusive="hub"),
    LOBBY_HUB: AnchorClass(
        LOBBY_HUB, "parallel", "the lobby's airlock edge into the hub",
        "breakable on the airlock wall (§4.3)", "one", exclusive="hub"),
    DEPARTMENT_ROOM: AnchorClass(
        DEPARTMENT_ROOM, "leaf",
        "a wing department's stub (a new side room) or a side / end room "
        "with no behind child (a new behind room); never a skeleton, ring, "
        "secret, exit or hull-door room",
        "packed room or behind room (§4.5)", "slack"),
    SAME_CELL_ROOMS: AnchorClass(
        SAME_CELL_ROOMS, "duct",
        "two rooms of one cell in different department instances, both in "
        "the quotient wing or both on the axis",
        "none (§4.6)", "slack"),
}
EXCLUSIVE_MAX = {"hub": 1}     # the ring OR the bypass, never both (v1)


@dataclass(frozen=True)
class Skeleton:
    """A skeleton's graph side. ``hub_department``: the REQUIRED axiom
    department, ``placement = "axis"``, whose entry (the hub) the lobbies'
    airlocks land on; it is the only axis department the skeleton realises.
    ``edges``: the axiom edge kinds by what they join."""
    name: str
    hub_department: str
    edges: dict

    def anchor(self, on: str, hub_entry: str) -> AnchorClass | None:
        """The anchor class an archetype's ``on`` names (None: no such
        class on this skeleton). The hub classes are spelled with the hub
        department's names: ``engineering.engine_room`` and
        ``lobby-engine_room`` on the freighter."""
        if on == f"{self.hub_department}.{hub_entry}":
            return ANCHORS[HUB]
        if on == f"{LOBBY}-{hub_entry}":
            return ANCHORS[LOBBY_HUB]
        if on in (HUB, LOBBY_HUB):
            return None
        return ANCHORS.get(on)


FREIGHTER = Skeleton(
    name="freighter",
    hub_department="engineering",
    edges={
        "slab-passageway": "door",
        "passageway-lobby": "door",
        "lobby-section": "door",       # to the aft-most MAIN section
        "section-section": "door",     # consecutive main sections
        "main-block": "bulkhead",      # the last main section to the block
        "lobby-hub": "airlock",        # a = lobby: the vestibule in the lobby
        "entry": "door",               # passageway to a wing department's stub
        "entry-own": "airlock",        # a promoted department's: a = its stub
        "exit": "door",                # slab to a first department's exit room
    })
SKELETONS = {FREIGHTER.name: FREIGHTER}


class LayoutError(ValueError):
    """A quotient graph the layout cannot read (a room whose role or attach
    names no primitive, a behind room whose parent is not a side or end
    room): a programming error in stage 1, never a draw to retry."""


# ===========================================================================
# The department model: what the packer reads of a wing department
# ===========================================================================
# Local frame: u along the row (the room's x extent, `w`), v from the
# passageway to the hull (its y extent, `h`). Every range is an inclusive
# (lo, hi). `need`: the opening + 2 * CLEAR its declared edge asks of the
# shared wall.

@dataclass(frozen=True)
class Kid:
    """A ``behind`` room: its ``w`` (u) and ``h`` (v) ranges."""
    id: str
    w: tuple
    h: tuple
    need: int = DOOR_W + 2 * CLEAR


@dataclass(frozen=True)
class Side:
    """A side room packed along stub ``stub``: ``w`` its depth away from the
    stub, ``h`` its length along it; ``exit``: its far wall must be the
    slab's (a declared spawn exit); ``kid``: its behind child."""
    id: str
    w: tuple
    h: tuple
    stub: int = 0
    need: int = DOOR_W + 2 * CLEAR
    exit: bool = False
    kid: Kid | None = None


@dataclass(frozen=True)
class End:
    """The room across the stubs' far end."""
    id: str
    w: tuple
    h: tuple
    need: int = DOOR_W + 2 * CLEAR
    hull_door: bool = False
    kid: Kid | None = None


@dataclass(frozen=True)
class Stub:
    id: str
    across: tuple
    along: tuple


@dataclass(frozen=True)
class Dept:
    """One wing department instance as the packer sees it. ``stubs``: one,
    or two with a retreat loop (the first nearer the slab); ``promoted``: its
    entry is an airlock whose vestibule stands in the first stub."""
    key: str
    stubs: tuple
    end: End | None = None
    sides: tuple = ()
    promoted: bool = False


@dataclass(frozen=True)
class Zone:
    """What stands on one side of a stub: a COLUMN of side rooms (in order
    from the passageway; a room with a behind child last) or, with no rooms,
    nothing (between two stubs: their wall). ``slot``: what takes the
    end-zone corner past an OUTER
    column -- ``"corner"`` (the last room's behind child), ``"end"`` (the end
    room's behind child, beside it) or nothing (the end room reaches over
    it)."""
    rooms: tuple = ()
    slot: str = ""


@dataclass(frozen=True)
class Config:
    """One way to arrange a department: its zones (left of the first stub,
    between two stubs, right of the last), whether the end room's behind
    child is ``stacked`` behind it, and the exact sets that arrangement can
    take: ``xs`` (bit x set = x-extent x is feasible), the stub length
    ``along``, the end-zone depth ``endz`` (None: no end room) and the
    department depth ``dp``, each an inclusive range."""
    zones: tuple
    stacked: bool
    xs: int
    along: tuple
    endz: tuple | None
    dp: tuple


def _alloc_range(zone: Zone, mid: bool = False) -> tuple:
    """A column's depth range: every room of it is exactly that deep (no
    steel behind a room, so a column's back is the department's edge or the
    other stub's wall, and an exit room reaches the slab's). A zone without
    rooms is nothing at all on an outer side (the stub stands against the
    department's edge) and, between two stubs, a wall up to VOID_MAX
    thicker."""
    if not zone.rooms:
        return (0, VOID_MAX if mid else 0)
    return (max(r.w[0] for r in zone.rooms), min(r.w[1] for r in zone.rooms))


def _slot_kid(zone: Zone, end: End) -> Kid:
    return zone.rooms[-1].kid if zone.slot == "corner" else end.kid


def _contrib(zone: Zone, alloc: int, end: End) -> tuple | None:
    """How far (lo, hi) the end room may reach over an OUTER zone of depth
    ``alloc``, counted from the stub's edge; None when the zone's slot room
    cannot stand there. No column: nothing to reach over. An empty slot: the
    end room covers the column to the department's edge. A slot room: it
    takes the rest of the column's depth, at least its own width (and, for
    a corner child, ``need`` tiles under its parent, which holds since the
    parent is the column's depth)."""
    if not zone.rooms:
        return (0, 0)
    if not zone.slot:
        return (alloc + 1, alloc + 1)
    k = _slot_kid(zone, end)
    lo = max(0, alloc - k.w[1])
    hi = alloc - max(k.w[0], k.need)
    return (lo, hi) if lo <= hi else None


def _core_range(end: End, stacked: bool) -> tuple:
    """The end room's width range (a stacked child is exactly as wide)."""
    lo, hi = end.w
    if stacked:
        lo = max(lo, end.kid.w[0], end.kid.need)
        hi = min(hi, end.kid.w[1])
    return (lo, hi)


def _across_range(dept: Dept) -> tuple:
    lo = max(s.across[0] for s in dept.stubs)
    hi = min(s.across[1] for s in dept.stubs)
    if dept.promoted:
        lo = max(lo, OWN_STUB_ACROSS_MIN)
    return (lo, hi)


def _u_options(dept: Dept, zones: tuple, stacked: bool):
    """Every ``(x-extent, across, left depth, mid depth, right depth)`` the
    arrangement can take: the stubs share one ``across``; an outer column
    adds its depth and a wall; two stubs are a wall (up to VOID_MAX thicker)
    or a walled column apart; the end room's width range must be
    reachable."""
    n = len(dept.stubs)
    a_lo, a_hi = _across_range(dept)
    end = dept.end
    cw = _core_range(end, stacked) if end is not None else None
    outer = []
    for z in (zones[0], zones[-1]):
        lo, hi = _alloc_range(z)
        row = []
        for alloc in range(lo, hi + 1):
            c = _contrib(z, alloc, end) if end is not None else (0, 0)
            if c is not None:
                row.append((alloc, alloc + (1 if z.rooms else 0), c[0], c[1]))
        outer.append(row)
    if n == 2:
        lo, hi = _alloc_range(zones[1], mid=True)
        mids = [(alloc, alloc + (2 if zones[1].rooms else 1))
                for alloc in range(lo, hi + 1)]
    else:
        mids = [(0, 0)]
    for a in range(a_lo, a_hi + 1):
        for (am, em) in mids:
            base = n * a + em
            for (a0, e0, l0, h0) in outer[0]:
                for (a1, e1, l1, h1) in outer[1]:
                    if cw is None or (base + l0 + l1 <= cw[1]
                                      and base + h0 + h1 >= cw[0]):
                        yield (base + e0 + e1, a, a0, am, a1)


def _zone_lengths(zone: Zone, promoted: bool) -> list:
    """The column's rooms' length ranges, in order; on a promoted stub the
    first room is long enough for an opening beyond the vestibule."""
    out = [r.h for r in zone.rooms]
    if promoted and out:
        out[0] = (max(out[0][0], PROMOTED_CLEAR + zone.rooms[0].need),
                  out[0][1])
    return out


def _zone_along(zone: Zone, promoted: bool) -> tuple | None:
    """The stub lengths a column can fill: its rooms and walls, with at
    most VOID_MAX of steel at its far end (none under a corner child)."""
    if not zone.rooms:
        return (0, _INF)
    rs = _zone_lengths(zone, promoted)
    if any(lo > hi for lo, hi in rs):
        return None
    n = len(rs)
    return (sum(lo for lo, _ in rs) + n - 1,
            sum(hi for _, hi in rs) + n - 1
            + (0 if zone.slot == "corner" else VOID_MAX))


def _config(dept: Dept, zones: tuple, stacked: bool) -> Config | None:
    end = dept.end
    lo = max(s.along[0] for s in dept.stubs)
    hi = min(s.along[1] for s in dept.stubs)
    if dept.promoted:
        lo = max(lo, VESTIBULE_D + 1 + FORMATION_FOOTPRINT)
    for z in zones:
        r = _zone_along(z, dept.promoted)
        if r is None:
            return None
        lo, hi = max(lo, r[0]), min(hi, r[1])
    if lo > hi:
        return None
    endz = None
    if end is not None:
        e_lo, e_hi = end.h
        if stacked:
            e_lo += 1 + end.kid.h[0]
            e_hi += 1 + end.kid.h[1]
        for z in (zones[0], zones[-1]):
            if z.slot:
                k = _slot_kid(z, end)
                e_lo, e_hi = max(e_lo, k.h[0]), min(e_hi, k.h[1] + VOID_MAX)
        if e_lo > e_hi:
            return None
        endz = (e_lo, e_hi)
    xs = 0
    for t in _u_options(dept, zones, stacked):
        xs |= 1 << t[0]
    if not xs:
        return None
    dp = (lo, hi) if endz is None else (lo + 1 + endz[0], hi + 1 + endz[1])
    return Config(zones=zones, stacked=stacked, xs=xs, along=(lo, hi),
                  endz=endz, dp=dp)


def _ordered(rooms: list, promoted: bool) -> tuple:
    """A column's rooms from the passageway (given in recipe order): the
    room with a behind child last; on a promoted stub the longest-capable
    room first."""
    hosts = [r for r in rooms if r.kid is not None]
    rest = [r for r in rooms if r.kid is None]
    if promoted and rest:
        first = max(rest, key=lambda r: r.h[1])
        rest = [first] + [r for r in rest if r is not first]
    return tuple(rest + hosts)


def _assignments(dept: Dept) -> list:
    """EVERY way the side rooms are dealt to the zones, up to swapping
    rooms that are alike (same stub, sizes, exit flag and behind child's
    sizes): for each group of alike rooms, how many stand on their stub's
    outer side, the rest on its other side (one stub) or between the stubs
    (two). Kept: a declared exit room stands in the slab-side column, a
    column holds at most one room with a behind child (its corner), and
    none stands between two stubs (there is no corner past the middle)."""
    n = len(dept.stubs)
    order = {r.id: i for i, r in enumerate(dept.sides)}
    groups: dict = {}
    for r in dept.sides:
        sig = (r.stub, r.w, r.h, r.need, r.exit,
               None if r.kid is None else (r.kid.w, r.kid.h, r.kid.need))
        groups.setdefault(sig, []).append(r)
    glist = list(groups.values())
    out = []

    def rec(gi, zones):
        if gi == len(glist):
            out.append(tuple(_ordered(sorted(z, key=lambda r: order[r.id]),
                                      dept.promoted) for z in zones))
            return
        rooms = glist[gi]
        first = 0 if rooms[0].stub == 0 else n
        for k in range(len(rooms), -1, -1):
            if rooms[0].exit and k < len(rooms):
                continue                           # an exit room: slab side
            zs = [list(z) for z in zones]
            zs[first] += rooms[:k]
            zs[1] += rooms[k:]
            if any(sum(r.kid is not None for r in z) > 1 for z in zs):
                continue                           # one corner per column
            if n == 2 and any(r.kid is not None for r in zs[1]):
                continue                           # no corner past the middle
            rec(gi + 1, zs)

    rec(0, [[] for _ in range(n + 1)])
    return out


@functools.lru_cache(maxsize=4096)
def dept_configs(dept: Dept) -> tuple:
    """Every arrangement of ``dept`` that can be packed at all, each with
    its exact feasible sets (:class:`Config`); empty when none can."""
    end = dept.end
    has_kids = any(r.kid is not None for r in dept.sides)
    if end is None and has_kids:
        return ()                                  # no end zone, no corner
    out = []
    for zs in _assignments(dept):
        slots = ["corner" if z and z[-1].kid is not None else ""
                 for z in zs]
        variants = [(tuple(slots), False)]
        if end is not None and end.kid is not None:
            variants = [(tuple(slots), True)]
            for i in (0, len(zs) - 1):
                if zs[i] and not slots[i]:
                    s2 = list(slots)
                    s2[i] = "end"
                    variants.append((tuple(s2), False))
        for sl, stacked in variants:
            zones = tuple(Zone(rooms=z, slot=s) for z, s in zip(zs, sl))
            c = _config(dept, zones, stacked)
            if c is not None:
                out.append(c)
    return tuple(out)


@functools.lru_cache(maxsize=4096)
def dept_front(dept: Dept) -> tuple:
    """The arrangements no other arrangement covers (same or larger
    x-extent set AND depth range): what a row's arithmetic needs. The
    packer still draws among ALL arrangements that fit the rectangle it is
    given."""
    cfgs = dept_configs(dept)
    keep = []
    for i, c in enumerate(cfgs):
        covered = False
        for k, o in enumerate(cfgs):
            if k == i or (o.xs | c.xs) != o.xs or o.dp[0] > c.dp[0] \
                    or o.dp[1] < c.dp[1]:
                continue
            if (o.xs, o.dp) != (c.xs, c.dp) or k < i:
                covered = True
                break
        if not covered:
            keep.append(c)
    return tuple(keep)


# ---- the placement primitives ----------------------------------------------

def pack(rng, length: int, ranges: list, slack: int = 0) -> list:
    """PACK (layout doc §3): sizes, each inside its ``(lo, hi)``, filling
    ``length`` with one wall tile between neighbours and at most ``slack``
    tiles left over at the far end (as little as the maxima allow).
    ``_Reject`` when the run cannot be filled."""
    n = len(ranges)
    total = length - (n - 1)
    lo = sum(r[0] for r in ranges)
    gap = max(0, total - sum(r[1] for r in ranges))
    if gap > slack or total < lo:
        raise G._Reject(f"pack: {n} rooms {list(ranges)} cannot fill "
                        f"{length} tiles (slack {slack})")
    sizes = [r[0] for r in ranges]
    rem = total - gap - lo
    while rem > 0:
        free = [i for i in range(n) if sizes[i] < ranges[i][1]]
        give = rng.multinomial(rem, np.full(len(free), 1.0 / len(free)))
        for i, gv in zip(free, give):
            add = min(int(gv), ranges[i][1] - sizes[i])
            sizes[i] += add
            rem -= add
    return sizes


def _draw(rng, lo: int, hi: int) -> int:
    if lo > hi:
        raise G._Reject(f"layout: an empty draw range {lo}..{hi}")
    return int(rng.integers(lo, hi + 1))


def _pick(rng, options: list):
    if not options:
        raise G._Reject("layout: no feasible option left to draw")
    return options[int(rng.integers(len(options)))]


def pack_department(dept: Dept, x: int, depth: int, rng,
                    cfg: Config | None = None) -> dict:
    """THE DEPARTMENT PACKER (layout doc §4): ``dept`` in a rectangle ``x``
    along the row by ``depth`` deep, as ``{room id: (u0, v0, w, h)}`` in the
    local frame (u from the slab end, v from the passageway). Strips (the
    stubs), packed columns, the end room across every stub's end, behind
    rooms in the end zone, two stubs under one end room. ``cfg``: the
    arrangement (default: drawn among those that fit). ``_Reject`` when
    none fits the rectangle."""
    if cfg is None:
        fits = [c for c in dept_configs(dept)
                if c.xs >> x & 1 and c.dp[0] <= depth <= c.dp[1]]
        cfg = _pick(rng, fits)
    zones, end, n = cfg.zones, dept.end, len(dept.stubs)
    _x, a, a0, am, a1 = _pick(rng, [t for t in _u_options(
        dept, zones, cfg.stacked) if t[0] == x])
    z0, z1 = zones[0], zones[-1]
    stub_u = [a0 + (1 if z0.rooms else 0)]
    mid_u0 = stub_u[0] + a + 1
    if n == 2:
        stub_u.append(stub_u[0] + a + am + (2 if zones[1].rooms else 1))
    last = stub_u[-1] + a - 1                      # the last stub's last tile
    right_u0 = last + 2
    if end is None:
        along, ez = depth, 0
    else:
        ez = _draw(rng, max(cfg.endz[0], depth - 1 - cfg.along[1]),
                   min(cfg.endz[1], depth - 1 - cfg.along[0]))
        along = depth - 1 - ez
    if not cfg.along[0] <= along <= cfg.along[1]:
        raise G._Reject(f"department {dept.key}: a stub of {along} does not "
                        f"fit its columns ({cfg.along})")
    rects = {s.id: (u, 0, a, along) for s, u in zip(dept.stubs, stub_u)}
    allocs = (a0, a1) if n == 1 else (a0, am, a1)
    for zi, (z, alloc) in enumerate(zip(zones, allocs)):
        if not z.rooms:
            continue
        sizes = pack(rng, along, _zone_lengths(z, dept.promoted),
                     0 if z.slot == "corner" else VOID_MAX)
        u0 = (0 if zi == 0 else right_u0 if zi == len(zones) - 1
              else mid_u0)
        v = 0
        for r, s in zip(z.rooms, sizes):
            rects[r.id] = (u0, v, alloc, s)
            v += s + 1
    if end is None:
        return rects
    cw = _core_range(end, cfg.stacked)
    c0, c1 = _contrib(z0, a0, end), _contrib(z1, a1, end)
    base = n * a + (0 if n == 1 else am + (2 if zones[1].rooms else 1))
    i, j = _pick(rng, [(i, j) for i in range(c0[0], c0[1] + 1)
                       for j in range(c1[0], c1[1] + 1)
                       if cw[0] <= base + i + j <= cw[1]])
    core_u0, core_w, v0 = stub_u[0] - i, base + i + j, along + 1
    if cfg.stacked:
        k = end.kid
        eh = _draw(rng, max(end.h[0], ez - 1 - k.h[1]),
                   min(end.h[1], ez - 1 - k.h[0]))
        rects[end.id] = (core_u0, v0, core_w, eh)
        rects[k.id] = (core_u0, v0 + eh + 1, core_w, ez - 1 - eh)
    else:
        rects[end.id] = (core_u0, v0, core_w, ez)
    for z, left in ((z0, True), (z1, False)):
        if not z.slot:
            continue
        k = _slot_kid(z, end)
        span = core_u0 - 1 if left else x - (core_u0 + core_w) - 1
        rects[k.id] = (0 if left else core_u0 + core_w + 1, v0, span,
                       min(k.h[1], ez))
    return rects


# ===========================================================================
# Rows, the band and the skeleton: the exact feasible sets
# ===========================================================================

@functools.lru_cache(maxsize=4096)
def _row_combos(row: tuple) -> tuple:
    """Every choice of one arrangement per department of a row that can
    share a depth: ``(lengths, depth lo, depth hi, picks)``, ``lengths`` the
    bit set of row lengths the choice can fill exactly (x-extents, one wall
    and at most VOID_MAX of steel between neighbours)."""
    out = []

    def rec(i, mask, lo, hi, picks):
        if i == len(row):
            out.append((mask, lo, hi, picks))
            return
        for ci, c in enumerate(dept_front(row[i])):
            l2, h2 = max(lo, c.dp[0]), min(hi, c.dp[1])
            if l2 > h2:
                continue
            if i == 0:
                m2 = c.xs
            else:
                m2, xs, b = 0, c.xs, 0
                while xs:
                    if xs & 1:
                        m2 |= mask << b
                    xs >>= 1
                    b += 1
                gaps = 0
                for v in range(VOID_MAX + 1):
                    gaps |= m2 << (1 + v)
                m2 = gaps
            if m2:
                rec(i + 1, m2, l2, h2, picks + (ci,))

    rec(0, 0, 0, _INF, ())
    return tuple(out)


@functools.lru_cache(maxsize=65536)
def _row_depths(row: tuple, length: int) -> tuple:
    """The depths (merged inclusive ranges) a row of ``length`` can take."""
    spans = sorted((lo, hi) for mask, lo, hi, _p in _row_combos(row)
                   if mask >> length & 1)
    out = []
    for lo, hi in spans:
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return tuple(out)


def _depth_fit(rows: tuple, length: int, total: int) -> bool:
    """Can rows A and B of ``length`` share ``total`` tiles of depth?"""
    da, db = _row_depths(rows[0], length), _row_depths(rows[1], length)
    return any(a0 + b0 <= total <= a1 + b1 for a0, a1 in da for b0, b1 in db)


@dataclass(frozen=True)
class Band:
    """The axis band of a quotient graph, fore to aft: ``mains`` main-cell
    section ids, then the hub department's ``block`` -- ``(id, role, w
    range | None, h range)`` with role ``fore`` / ``hub`` / ``aft`` /
    ``ring_fore`` / ``ring_aft``; ``sides``: the ring's side rooms, aft to
    fore; ``target``: the band room the lobby's airlock lands on; ``door``:
    the main section the lobby's door goes to; ``bypass``: a breakable
    beside that airlock."""
    mains: tuple
    block: tuple
    sides: tuple = ()
    target: str = ""
    door: str = ""
    bypass: bool = False


@dataclass(frozen=True)
class Problem:
    """Everything the skeleton's arithmetic reads: the size preset, rows A
    and B (their departments by position from the slab), whether a hull
    strip is applied, the band, the archetype's passageway and lobby
    ranges and the request."""
    size: str
    rows: tuple
    strip: bool
    band: Band
    passageway: tuple
    lobby: tuple
    max_per_team: int = 6
    zombies: int = 0


def zombie_height(wm: int, zombies: int) -> int:
    """The least height of a band section ``wm`` wide that seats
    ``zombies`` under the ``formation`` rule."""
    if zombies <= 0:
        return 0
    cols = (wm + FORMATION_GAP) // FORMATION_PITCH
    if cols < 1:
        return _INF
    return math.ceil(zombies / cols) * FORMATION_PITCH - FORMATION_GAP


def _side_needs(band: Band, cut: int) -> list:
    """The ring's side rooms' least heights, aft to fore: the lobby's
    airlock lands on the aft-most one."""
    need = [MIN_ROOM] * len(band.sides)
    if band.target == band.sides[0]:
        need[0] = max(need[0], AIRLOCK_W + 2 * CLEAR + cut)
    return need


def _band_cut(n_items: int, i: int, strip_row) -> int:
    """How much of band item ``i``'s height the lobby does NOT face: a hull
    strip cuts the lobby's fore end (row A) or aft end (row B)."""
    return STRIP_COST if ((i == 0 and strip_row == ROW_A)
                          or (i == n_items - 1 and strip_row == ROW_B)) else 0


def _band_items(band: Band, wm: int, zombies: int, strip_row) -> list | None:
    """The band's stacked items fore to aft as ``(id, lo, hi)`` heights, or
    None when ``wm`` cannot take the block (a ring needs RING_BAND_W_MIN, a
    section spans the band, the hub what the ring leaves). The zombie
    section seats its horde; a section the lobby opens onto is tall enough
    for that opening where the lobby still faces it."""
    ring = bool(band.sides)
    if ring and wm < RING_BAND_W_MIN:
        return None
    nm = len(band.mains)
    items = [[m, MIN_SECTION, _INF] for m in band.mains]
    if items:
        items[-1][1] = max(items[-1][1], zombie_height(wm, zombies))
    for (rid, role, w, h) in band.block:
        width = wm - 2 * (RING_DEPTH + 1) if ring and role == "hub" else wm
        if w is not None and not w[0] <= width <= w[1]:
            return None
        items.append([rid, h[0], h[1]])
    for i, it in enumerate(items):
        cut = _band_cut(len(items), i, strip_row)
        need = DOOR_W + 2 * CLEAR + cut if it[0] == band.door else 0
        if i >= nm and band.block[i - nm][1] == "hub":
            if ring:
                sides = _side_needs(band, cut)
                need = max(need, sum(sides) + len(sides) - 1)
            elif band.target == it[0]:
                need = max(need, cut + (BYPASS_HUB_H_MIN if band.bypass
                                        else AIRLOCK_W + 2 * CLEAR))
        it[1] = max(it[1], need)
        if it[1] > it[2]:
            return None
    return [tuple(it) for it in items]


def _band_fit(band: Band, wm: int, height: int, zombies: int,
              strip_row) -> bool:
    items = _band_items(band, wm, zombies, strip_row)
    if items is None or not band.mains:
        return False
    walls = len(items) - 1
    return sum(lo for _i, lo, _h in items) + walls <= height


def _slab_x0(size: str, max_per_team: int) -> int:
    """The slab's least x-extent: plan_graph's formation run."""
    c, _r = G._spawn_grid(max_per_team, SIZES[size].h - 2 * BAND - 2)
    return G._extent(c)


def _wing_width(size: str, wm: int) -> int:
    """The wing's air width beside a band ``wm`` wide."""
    return SIZES[size].w // 2 - wm // 2 - BAND - 2


@functools.lru_cache(maxsize=1024)
def skeleton_options(prob: Problem) -> tuple:
    """EVERY skeleton draw that leaves the problem packable (layout doc §13
    A4): ``(inset top, inset bottom, band width, passageway across, lobby x,
    slab x, strip row)``. The strip takes the first row, in order A, B, that
    has no hull-door end room and still packs (A3). Empty: infeasible."""
    p = SIZES[prob.size]
    sx0 = _slab_x0(prob.size, prob.max_per_team)
    if any(not row for row in prob.rows):
        return ()
    out = []
    ok: dict = {}
    for it in INSETS:
        for ib in INSETS:
            height = p.h - 2 * BAND - 2 - it - ib
            for wm in range(p.band_w[0], p.band_w[1] + 1, 2):
                wing = _wing_width(prob.size, wm)
                for ap in range(prob.passageway[0], prob.passageway[1] + 1):
                    for lx in range(prob.lobby[0], prob.lobby[1] + 1):
                        for sx in range(sx0, sx0 + SLAB_JITTER + 1):
                            key = (height, wm, ap, wing - sx - lx - 2, sx)
                            if key not in ok:
                                ok[key] = _skeleton_row(prob, *key)
                            if ok[key] is not False:
                                out.append((it, ib, wm, ap, lx, sx, ok[key]))
    return tuple(out)


def _skeleton_row(prob: Problem, height: int, wm: int, ap: int, length: int,
                  sx: int):
    """The strip's row (None without a strip) for one skeleton draw, or
    False when the draw cannot be packed."""
    if length < 1:
        return False
    for sr in (ROW_NAMES if prob.strip else (None,)):
        if sr is not None and any(
                d.end is not None and d.end.hull_door
                for d in prob.rows[ROW_NAMES.index(sr)]):
            continue
        slab_h = height - (STRIP_COST if sr else 0)
        cap = ((sx + FORMATION_GAP) // FORMATION_PITCH) \
            * ((slab_h + FORMATION_GAP) // FORMATION_PITCH)
        if cap < prob.max_per_team:
            continue
        if not _depth_fit(prob.rows, length, slab_h - ap - 2):
            continue
        if _band_fit(prob.band, wm, height, prob.zombies, sr):
            return sr
    return False


def _why(prob: Problem) -> str:
    """The reason :func:`skeleton_options` is empty, naming what cannot
    pack: a row, a department, the band or the depth."""
    p = SIZES[prob.size]
    sx0 = _slab_x0(prob.size, prob.max_per_team)
    lengths = sorted({_wing_width(prob.size, wm) - sx - lx - 2
                      for wm in range(p.band_w[0], p.band_w[1] + 1, 2)
                      for sx in range(sx0, sx0 + SLAB_JITTER + 1)
                      for lx in range(prob.lobby[0], prob.lobby[1] + 1)})
    lengths = [n for n in lengths if n >= 1]
    for name, row in zip(ROW_NAMES, prob.rows):
        keys = [d.key for d in row]
        if not row:
            return f"row {name} has no department"
        for d in row:
            if not dept_configs(d):
                return (f"row {name}: department {d.key} cannot be packed "
                        f"at all ({len(d.sides)} side rooms, "
                        f"{sum(r.kid is not None for r in d.sides)} behind a "
                        f"side room: one corner per outer column, a stub "
                        f"long enough for every column)")
        combos = _row_combos(row)
        if not combos:
            return (f"row {name} ({keys}): its departments share no depth "
                    f"({[[c.dp for c in dept_configs(d)] for d in row]})")
        if not any(mask >> n & 1 for mask, *_r in combos for n in lengths):
            fits = sorted({b for mask, *_r in combos
                           for b in range(mask.bit_length())
                           if mask >> b & 1})
            return (f"row {name} ({keys}): its x-extents fill a row of "
                    f"{fits[0]}..{fits[-1]} tiles, size {prob.size!r} offers "
                    f"{lengths[0]}..{lengths[-1]}" if lengths else
                    f"row {name}: size {prob.size!r} leaves no row")
    heights = [p.h - 2 * BAND - 2 - i for i in range(2 * INSETS[-1] + 1)]
    widths = range(p.band_w[0], p.band_w[1] + 1, 2)
    strips = ROW_NAMES if prob.strip else (None,)
    if not any(_band_fit(prob.band, wm, h, prob.zombies, sr)
               for wm in widths for h in heights for sr in strips):
        need = min((sum(lo for _i, lo, _h in it) + len(it) - 1
                    for wm in widths for sr in strips
                    for it in [_band_items(prob.band, wm, prob.zombies, sr)]
                    if it is not None), default=None)
        return (f"the band ({len(prob.band.mains)} main sections + "
                f"{[b[0] for b in prob.band.block]}"
                + (", a ring" if prob.band.sides else "")
                + f", {prob.zombies} zombies) "
                + (f"needs {need} tiles of height, size {prob.size!r} has at "
                   f"most {heights[0]}" if need is not None else
                   f"fits no band width of size {prob.size!r} "
                   f"{list(p.band_w)}"))
    total = heights[0] - prob.passageway[0] - 2 - (STRIP_COST if prob.strip
                                                   else 0)
    return (f"rows A {[d.key for d in prob.rows[0]]} and B "
            f"{[d.key for d in prob.rows[1]]} cannot share the wing: no row "
            f"length in {lengths[0]}..{lengths[-1]} gives both a depth "
            f"within the {total} tiles of height beside the passageway"
            + (" under the hull strip (a hull-door row cannot take it)"
               if prob.strip else ""))


# ===========================================================================
# From a quotient graph, and from an archetype, to a Problem
# ===========================================================================

def _edge_need(kinds) -> int:
    return max((G.EDGE_KINDS[k].width for k in kinds), default=0) + 2 * CLEAR


def problem_of(q) -> Problem:
    """A quotient graph (``ship_grammar.QuotientGraph``) as a
    :class:`Problem`; :class:`LayoutError` for a room no primitive
    places."""
    rooms = {r.id: r for r in q.rooms}
    pairs: dict = {}
    for e in q.edges:
        pairs.setdefault(frozenset((e.a, e.b)), []).append(e.kind)

    def need(a, b):
        return _edge_need(pairs.get(frozenset((a, b)), ()))

    exits = {e.a if e.b == SLAB else e.b for e in q.exit_edges
             if e.role == "exit"}
    placed = set()
    rows: dict = {ROW_A: [], ROW_B: []}
    for d in q.departments:
        if d.placement != "wing":
            continue
        kids, sides, end = {}, [], None
        mine = [rooms[i] for i in d.rooms]
        for r in mine:
            if r.attach == "behind":
                kids[r.parent] = Kid(r.id, r.size.a, r.size.b,
                                     need(r.id, r.parent))
        for r in mine:
            if r.id in d.stubs or r.attach == "behind":
                continue
            if r.attach == "end":
                end = End(r.id, r.size.a, r.size.b,
                          max(need(r.id, s) for s in d.stubs),
                          "hull_door" in r.tags, kids.pop(r.id, None))
            elif r.attach == "side" and r.parent in d.stubs:
                w = r.size.a
                if r.id in exits and ("hazard" in r.tags
                                      or "heat_source" in r.tags):
                    # doors in two facing walls: a floor site clear of both
                    w = (max(w[0], EXIT_SITE_W), w[1])
                sides.append(Side(r.id, w, r.size.b,
                                  d.stubs.index(r.parent),
                                  need(r.id, r.parent), r.id in exits,
                                  kids.pop(r.id, None)))
            else:
                raise LayoutError(f"room {r.id!r}: attach {r.attach!r} "
                                  f"(parent {r.parent!r}) names no placement "
                                  f"primitive of a wing department")
        if kids:
            raise LayoutError(f"department {d.name}#{d.instance}: behind "
                              f"rooms {[k.id for k in kids.values()]} hang "
                              f"off no side or end room")
        promoted = "airlock" in pairs.get(frozenset((d.entry, PASSAGEWAY)),
                                          ())
        rows[d.row].append((d.position, Dept(
            key=f"{d.name}#{d.instance}",
            stubs=tuple(Stub(s, rooms[s].size.a, rooms[s].size.b)
                        for s in d.stubs),
            end=end, sides=tuple(sides), promoted=promoted)))
        placed.update(d.rooms)
    mains = tuple(x for x in q.band if rooms[x].role == SECTION)
    hub_entry = next((d.entry for d in q.departments
                      if d.placement == "axis"), None)
    block = []
    for x in q.band:
        r = rooms[x]
        if r.role == SECTION:
            continue
        if r.role in (RING_FORE, RING_AFT):
            role = r.role
        elif r.attach in ("fore", "aft"):
            role = r.attach
        elif x == hub_entry:
            role = "hub"
        else:
            raise LayoutError(f"band room {x!r}: role {r.role!r}, attach "
                              f"{r.attach!r} is no section of the block")
        block.append((x, role, r.size.a, r.size.b))
    sides = tuple(r.id for r in q.rooms if r.role == RING_SIDE)
    target = next((e.b for e in q.edges if e.kind == "airlock"
                   and e.a == LOBBY), "")
    door = next((e.b if e.a == LOBBY else e.a for e in q.edges
                 if LOBBY in (e.a, e.b) and e.kind != "airlock"
                 and rooms[e.b if e.a == LOBBY else e.a].role == SECTION), "")
    bypass = bool(target) and "breakable" in pairs.get(
        frozenset((LOBBY, target)), ())
    strip = [r.id for r in q.rooms if r.role == FLANK_STRIP]
    placed.update(q.band)
    placed.update(sides)
    placed.update(strip)
    placed.update((SLAB, PASSAGEWAY, LOBBY))
    lost = [r.id for r in q.rooms if r.id not in placed]
    if lost:
        raise LayoutError(f"rooms {lost} have no place in skeleton "
                          f"{q.skeleton!r}: an unknown role or attach")
    return Problem(
        size=q.size,
        rows=tuple(tuple(d for _p, d in sorted(rows[n], key=lambda t: t[0]))
                   for n in ROW_NAMES),
        strip=bool(strip),
        band=Band(mains=mains, block=tuple(block), sides=sides,
                  target=target, door=door, bypass=bypass),
        passageway=rooms[PASSAGEWAY].size.a, lobby=rooms[LOBBY].size.a,
        max_per_team=q.max_per_team, zombies=q.zombies)


def recipe_dept(dept, which: str = "min", *, key: str | None = None,
                promoted: bool = False, exit_used: bool = False,
                two_stub: bool = False) -> Dept:
    """A WING department recipe (``levelgen_recipes.Department``) as the
    packer's :class:`Dept`, at its MINIMA (``which = "min"``: required rooms,
    every ``count`` at its low end, no optional room) or its MAXIMA
    (``"max"``: every ``count`` at its high end, every optional room). A
    behind room is present with its parent. ``two_stub``: with a retreat
    loop's second stub, the side rooms dealt as the grammar deals them."""
    key = key or dept.name
    kinds: dict = {}
    for e in dept.edges:
        kinds.setdefault(frozenset((e.a, e.b)), []).append(e.kind)

    def n_of(r):
        if r.required:
            return 1
        if r.count:
            return r.count[0] if which == "min" else r.count[1]
        return 0 if which == "min" else 1

    count = {r.id: n_of(r) for r in dept.rooms}
    entry = dept.room(dept.entry)
    stubs = [Stub(f"{key}.{entry.id}", entry.size.a, entry.size.b)]
    if two_stub:
        stubs.append(Stub(f"{key}.{STUB2}", entry.size.a, entry.size.b))
    kids = {}
    for r in dept.rooms:
        if r.parent and count[r.id] and count.get(r.parent):
            kids[r.parent] = Kid(f"{key}.{r.id}", r.size.a, r.size.b,
                                 _edge_need(kinds.get(
                                     frozenset((r.id, r.parent)), ())))
    end, sides, moved = None, [], 0
    for r in dept.rooms:
        if r.id == dept.entry or r.parent or not count[r.id]:
            continue
        need = _edge_need(kinds.get(frozenset((r.id, dept.entry)), ()))
        if r.attach == "end":
            end = End(f"{key}.{r.id}", r.size.a, r.size.b, need,
                      r.hull_door, kids.get(r.id))
            continue
        for k in range(count[r.id]):
            ex = r.exit and exit_used
            stub = 0
            if two_stub and not r.exit:
                stub, moved = moved % 2, moved + 1
            sides.append(Side(f"{key}.{r.id}.{k}", r.size.a, r.size.b, stub,
                              need, ex, kids.get(r.id) if k == 0 else None))
    return Dept(key=key, stubs=tuple(stubs), end=end, sides=tuple(sides),
                promoted=promoted)


def _archetype_problems(arch, size: str, profile: str, which: str):
    """The archetype at its MINIMA or MAXIMA on (size, profile) as Problems
    -- alternatives, any of which the grammar may draw: the band's section
    count, and which departments take the retreat loops (archetype doc §6
    items 1-2: departments at minimum / maximum count, the mandatory
    two-stubs and hull strip, the ring and the bypass)."""
    import itertools
    lib = arch.library
    skel = SKELETONS[arch.skeleton]
    hub = lib.departments[arch.hub_department]
    ranges = arch.ranges[(size, profile)]
    pick = 0 if which == "min" else 1
    promoted = set(arch.promoted(profile))
    names = [x.department for x in arch.axiom if x.placement == "wing"]
    n_loops, strip, ring, bypass = 0, False, 0, False
    for e in arch.applies:
        n = ranges[e.index][pick]
        if e.department:
            names += [e.department] * n
        elif n:
            cls = skel.anchor(e.on, hub.entry)
            if cls.key == DEPARTMENT_ENTRY:
                n_loops = n
            elif cls.key == SLAB_LOBBY:
                strip = True
            elif cls.key == HUB:
                ring = 1 if which == "min" else RING_ROOMS_MAX // 2 - 1
            elif cls.key == LOBBY_HUB:
                bypass = True
    if ring:
        bypass = False                             # exclusive (archetype §3)
    order = sorted(range(len(names)), key=lambda i: (
        0 if lib.departments[names[i]].toward == "spawn" else 1, i))
    exits = arch.exits[pick]
    loopable = [j for j, i in enumerate(order) if any(
        r.attach == "end" for r in lib.departments[names[i]].rooms)]
    block_rooms = [r for r in hub.rooms if r.attach in ("fore", "aft")
                   and (r.required or which == "max")]
    hub_room = hub.room(hub.entry)
    block = [(f"{hub.name}.{r.id}", "fore", r.size.a, r.size.b)
             for r in block_rooms if r.attach == "fore"]
    if ring:
        block.append(("ring_fore", RING_FORE, None, (RING_DEPTH, RING_DEPTH)))
    block.append((f"{hub.name}.{hub_room.id}", "hub", hub_room.size.a,
                  hub_room.size.b))
    if ring:
        block.append(("ring_aft", RING_AFT, None, (RING_DEPTH, RING_DEPTH)))
    block += [(f"{hub.name}.{r.id}", "aft", r.size.a, r.size.b)
              for r in block_rooms if r.attach == "aft"]
    sides = tuple(f"ring_side{i}" for i in range(ring))
    hub_id = f"{hub.name}.{hub_room.id}"
    k_lo, k_hi = arch.sections[size]
    n_block = len([b for b in block if b[1] in ("fore", "hub", "aft")])
    for loops in itertools.combinations(loopable, min(n_loops,
                                                      len(loopable))):
        rows = {ROW_A: [], ROW_B: []}
        for j, i in enumerate(order):
            name = names[i]
            row = ROW_NAMES[j % 2]
            first = j // 2 == 0
            rows[row].append(recipe_dept(
                lib.departments[name], which, key=f"{name}#{j}",
                promoted=name in promoted,
                exit_used=first and name not in promoted
                and exits >= 2 + ROW_NAMES.index(row),
                two_stub=j in loops))
        for k in range(k_lo, k_hi + 1):
            m = k - n_block
            if m < 1:
                continue
            mains = tuple(f"{SECTION}_{i}" for i in range(m))
            yield Problem(
                size=size, rows=(tuple(rows[ROW_A]), tuple(rows[ROW_B])),
                strip=strip,
                band=Band(mains=mains, block=tuple(block), sides=sides,
                          target=sides[0] if sides else hub_id,
                          door=mains[-1], bypass=bypass),
                passageway=arch.passageway, lobby=arch.lobby_x)


def layout_feasibility(size: str, *, graph=None, archetype=None,
                       profile: str | None = None) -> str | None:
    """THE LAYOUT FEASIBILITY CALL SITE -- None when the layout can pack
    it, else the reason, naming what cannot (a row and its departments, a
    department, the band, the depth). Called

    1. by the grammar at the end of every attempt with the whole quotient
       ``graph`` (layout doc §13 A5: an infeasible graph costs one grammar
       attempt, not ``MAX_ATTEMPTS`` layout attempts) -- EXACT: the answer is
       whether :func:`skeleton_options` is empty, the very set
       :func:`layout` draws from;
    2. by the archetype loader for every allowed (``size``, ``profile``)
       with the ``archetype`` (archetype doc §6 item 1): the recipe at its
       MINIMA -- every department at its minimum count with only its
       required rooms, the mandatory retreat loops, hull strip, ring and
       bypass; pockets, secrets and duct links are slack-bounded and left to
       (1). Feasible when ANY band section count and any choice of looped
       departments packs.

    Closed form and cheap: per department arrangement the feasible
    x-extents and depths are enumerated once (cached), a row is a sum set,
    the skeleton a product of at most ~1200 draws of table lookups."""
    if graph is not None:
        prob = problem_of(graph)
        if prob.size != size:
            raise LayoutError(f"graph is size {prob.size!r}, asked {size!r}")
        return None if skeleton_options(prob) else _why(prob)
    why = None
    for prob in _archetype_problems(archetype, size, profile, "min"):
        w = _demand(prob)
        if w is None:
            return None
        why = why or w
    return (f"at its minima, "
            f"{why or 'no band section count leaves a main section'}")


def maxima_warning(size: str, archetype, profile: str) -> str | None:
    """Archetype doc §6 item 2: the same arithmetic at the recipe's MAXIMA
    (every count at its high end, every optional room, every cycle the
    profile allows). A recipe that does not pack there still generates --
    the grammar rerolls a draw the layout refuses -- so this is a WARNING,
    naming what overflows; None when even the maxima pack."""
    why = None
    for prob in _archetype_problems(archetype, size, profile, "max"):
        w = _demand(prob)
        if w is None:
            return None
        why = why or w
    return (f"at its maxima, {why}: the grammar will reroll draws near the "
            f"top of its ranges on size {size!r}, profile {profile!r}")


def _demand(prob: Problem) -> str | None:
    """The DEMAND test of the loader's twin (archetype doc §6): does what
    the problem ASKS FOR fit what the size offers -- each row's least
    x-extents (and walls) inside the longest row, the two rows' least depths
    inside the wing's height beside the passageway (less a hull strip), the
    band's least heights inside the hull. It ignores the upper ends (a
    department too small to fill its row is the grammar's draw to fix, not
    the recipe's fault): necessary, not sufficient. None, or what
    overflows."""
    p = SIZES[prob.size]
    sx0 = _slab_x0(prob.size, prob.max_per_team)
    height = p.h - 2 * BAND - 2
    length = _wing_width(prob.size, p.band_w[0]) - sx0 - prob.lobby[0] - 2
    total = height - prob.passageway[0] - 2 - (STRIP_COST if prob.strip
                                               else 0)
    best = []
    for name, row in zip(ROW_NAMES, prob.rows):
        keys = [d.key for d in row]
        if not row:
            return f"row {name} has no department"
        opts = [(0, 0)]
        for d in row:
            cfgs = dept_configs(d)
            if not cfgs:
                return (f"row {name}: department {d.key} cannot be packed "
                        f"at all (its side rooms along its stub, one corner "
                        f"per outer column)")
            mine = sorted({((c.xs & -c.xs).bit_length() - 1, c.dp[0])
                           for c in cfgs})
            opts = sorted({(x + mx + 1, max(dp, mdp)) for x, dp in opts
                           for mx, mdp in mine})
        opts = [(x - 1, dp) for x, dp in opts if x - 1 <= length]
        if not opts:
            return (f"row {name} ({keys}): its least x-extents and walls "
                    f"exceed the longest row of size {prob.size!r} "
                    f"({length} tiles)")
        best.append(opts)
    if min(dp for _x, dp in best[0]) + min(dp for _x, dp in best[1]) > total:
        return (f"rows A {[d.key for d in prob.rows[0]]} and B "
                f"{[d.key for d in prob.rows[1]]} need "
                f"{min(dp for _x, dp in best[0])} + "
                f"{min(dp for _x, dp in best[1])} tiles of depth; size "
                f"{prob.size!r} has {total} beside the passageway"
                + (" under the hull strip" if prob.strip else ""))
    if prob.strip and all(any(d.end is not None and d.end.hull_door
                              for d in row) for row in prob.rows):
        return ("the hull strip has no row: both rows hold a hull-door "
                "room (layout doc §13 A3)")
    strips = ROW_NAMES if prob.strip else (None,)
    if not any(_band_fit(prob.band, wm, height, prob.zombies, sr)
               for wm in range(p.band_w[0], p.band_w[1] + 1, 2)
               for sr in strips):
        return (f"the band ({len(prob.band.mains)} main sections + "
                f"{[b[0] for b in prob.band.block]}"
                + (", a ring" if prob.band.sides else "")
                + f") fits no band of size {prob.size!r}")
    return None


def department_feasibility(dept) -> str | None:
    """THE RECIPE LOADER'S TWIN (layout doc §4 "Loader twin"): a department
    recipe at its MINIMA must pack in SOME size preset -- a wing department
    in a row no longer and no deeper than the largest preset can offer, an
    axis department's block in its band. None, or the reason. (Whether it
    packs on a given size beside its neighbours is the archetype loader's
    question, :func:`layout_feasibility`.)"""
    if dept.placement != "wing":
        hub = dept.room(dept.entry)
        block = [hub] + [r for r in dept.rooms
                         if r.attach in ("fore", "aft") and r.required]
        for name, p in SIZES.items():
            widths = [wm for wm in range(p.band_w[0], p.band_w[1] + 1, 2)
                      if all(r.size.a[0] <= wm <= r.size.a[1] for r in block)]
            need = sum(r.size.b[0] for r in block) + len(block) + MIN_SECTION
            if widths and need <= p.h - 2 * BAND - 2:
                return None
        return (f"its block {[r.id for r in block]} (heights "
                f"{[list(r.size.b) for r in block]}, widths "
                f"{[list(r.size.a) for r in block]}) fits no size preset's "
                f"band beside one main section")
    d = recipe_dept(dept, "min")
    cfgs = dept_configs(d)
    if not cfgs and not d.sides and d.end is not None:
        return (f"with no side room at minimum count its end room (w "
                f"{list(d.end.w)}) must be exactly as wide as its stub "
                f"(across {list(d.stubs[0].across)})"
                + (f", and its behind room (w {list(d.end.kid.w)}) as wide "
                   f"again" if d.end.kid is not None else ""))
    if not cfgs:
        return (f"its {len(d.sides)} side rooms at minimum count cannot be "
                f"packed along its stub (along {list(d.stubs[0].along)}): "
                f"each column needs its rooms' minimum lengths plus a wall "
                f"between them")
    best = None
    for name, p in SIZES.items():
        length = _wing_width(name, p.band_w[0]) - MIN_ROOM - LOBBY_X_MIN - 2
        depth = p.h - 2 * BAND - 2 - MIN_ROOM - 2 - MIN_ROOM
        for c in cfgs:
            x = (c.xs & -c.xs).bit_length() - 1
            if x <= length and c.dp[0] <= depth:
                return None
            best = (x, c.dp[0], length, depth, name)
    x, dp, length, depth, name = best
    return (f"at its minima it needs a row {x} long and {dp} deep; the "
            f"largest size preset ({name!r}) offers at most {length} by "
            f"{depth}")


def check_request(arch, size: str, profile: str, max_per_team: int,
                  zombies: int) -> None:
    """The recipe path's ``gen_ship.check_request``: refuse, before any
    draw, a request this archetype can never seat on ``size`` -- the size or
    profile it does not allow, a team the slab cannot hold, a horde (SCALED
    by the profile, archetype doc §4) the zombie section cannot."""
    if size not in arch.sizes:
        raise G.ShipGenError(f"archetype {arch.name!r} does not allow size "
                             f"{size!r} (allowed: {list(arch.sizes)})")
    if profile not in arch.profiles:
        raise G.ShipGenError(f"archetype {arch.name!r} has no profile "
                             f"{profile!r} (known: {sorted(arch.profiles)})")
    if max_per_team < 1:
        raise G.ShipGenError(f"max_per_team must be >= 1 (got "
                             f"{max_per_team})")
    if zombies < 0:
        raise G.ShipGenError(f"zombies must be >= 0 (got {zombies})")
    p = SIZES[size]
    height = p.h - 2 * BAND - 2
    sx = _slab_x0(size, max_per_team)
    row = _wing_width(size, p.band_w[0]) - sx - arch.lobby_x[0] - 2
    cap = ((sx + SLAB_JITTER + FORMATION_GAP) // FORMATION_PITCH) \
        * ((height + FORMATION_GAP) // FORMATION_PITCH)
    if cap < max_per_team or row < 2 * MIN_ROOM + 1:
        raise G.ShipGenError(
            f"archetype {arch.name!r}, size {size!r} cannot seat "
            f"max_per_team = {max_per_team}: its {sx}..{sx + SLAB_JITTER}-"
            f"tile spawn slab leaves a row of {row} tiles (a department "
            f"needs {2 * MIN_ROOM + 1}) and seats {cap}")
    scaled = arch.scaled_zombies(profile, zombies)
    if scaled:
        hub = arch.library.departments[arch.hub_department]
        block = [hub.room(hub.entry)] + [r for r in hub.rooms
                                         if r.attach in ("fore", "aft")
                                         and r.required]
        k = arch.sections[size][0]
        room = height - sum(r.size.b[0] for r in block) - len(block) \
            - (k - len(block) - 1) * (MIN_SECTION + 1)
        wm = p.band_w[1]
        cap = ((wm + FORMATION_GAP) // FORMATION_PITCH) \
            * max(0, (room + FORMATION_GAP) // FORMATION_PITCH)
        if cap < scaled:
            raise G.ShipGenError(
                f"archetype {arch.name!r}, size {size!r}, profile "
                f"{profile!r} cannot seat zombies = {zombies} (scaled to "
                f"{scaled}): its zombie section holds at most {cap}")


# ===========================================================================
# The layout: draws, rectangles, openings, the reflection
# ===========================================================================

@dataclass
class Layout:
    """:func:`layout`'s product. ``graph``: the ``gen_ship.ShipGraph``
    ``embed`` consumes; ``ids``: quotient room id -> ``(room id, twin id)``
    (the same id twice on the axis); ``params``: the attempt's skeleton
    draws (insets, band width, passageway across and y, lobby and slab x,
    the strip's row, each row's depth, each department's x-extent)."""
    graph: G.ShipGraph
    ids: dict
    params: dict = field(default_factory=dict)


def _mirror_duct(name: str) -> str:
    return f"{name}_m"


def layout(q, rng) -> Layout:
    """THE LAYOUT STAGE (stage 1.5): the quotient graph ``q`` as a mirrored
    ``ShipGraph`` -- every room a rectangle inside the hull and inside its
    size range, every declared edge a shared wall at least its opening + 2
    long with its ``at`` set, the left wing laid out once and reflected,
    each wing given its own copy of every wing duct, ``lamp_span`` on every
    room. Every draw comes from ``rng`` in the module docstring's order.
    ``gen_ship._Reject`` when the draw cannot be realised."""
    prob = problem_of(q)
    opts = list(skeleton_options(prob))
    if not opts:
        raise G._Reject(f"layout: {_why(prob)}")
    for i in range(6):                              # the conditioned draws
        v = _pick(rng, sorted({t[i] for t in opts}))
        opts = [t for t in opts if t[i] == v]
    it, ib, wm, ap, lx, sx, strip_row = opts[0]
    p = SIZES[q.size]
    W, H = p.w, p.h
    hull = (BAND, BAND + it, W - 1 - BAND, H - 1 - BAND - ib)
    ix0, iy0, iy1 = hull[0] + 1, hull[1] + 1, hull[3] - 1
    height = iy1 - iy0 + 1
    cx0 = W // 2 - wm // 2
    wx1 = cx0 - 2
    length = wx1 - ix0 + 1 - sx - lx - 2
    top = iy0 + (STRIP_COST if strip_row == ROW_A else 0)
    bot = iy1 - (STRIP_COST if strip_row == ROW_B else 0)
    rects: dict = {}                                # quotient id -> x, y, w, h

    # ---- the band ----
    _band_rects(prob.band, wm, cx0, iy0, height, q.zombies, strip_row, rng,
                rects)

    # ---- rows: arrangements, the passageway's y, x-extents ----
    total = bot - top + 1 - ap - 2
    ca = [c for c in _row_combos(prob.rows[0]) if c[0] >> length & 1]
    cb = [c for c in _row_combos(prob.rows[1]) if c[0] >> length & 1]
    ca = [a for a in ca if any(a[1] + b[1] <= total <= a[2] + b[2]
                               for b in cb)]
    pa = _pick(rng, ca)
    pb = _pick(rng, [b for b in cb if pa[1] + b[1] <= total <= pa[2] + b[2]])
    depth_a = _draw(rng, max(pa[1], total - pb[2]), min(pa[2], total - pb[1]))
    depths = (depth_a, total - depth_a)
    py0 = top + depth_a + 1
    rx0 = ix0 + sx + 1
    rects[SLAB] = (ix0, top, sx, bot - top + 1)
    rects[PASSAGEWAY] = (rx0, py0, length, ap)
    rects[LOBBY] = (rx0 + length + 1, top, lx, bot - top + 1)
    for r in q.rooms:
        if r.role == FLANK_STRIP:
            rects[r.id] = (ix0, iy0 if strip_row == ROW_A else iy1
                           - FLANK_STRIP_ACROSS + 1, wx1 - ix0 + 1,
                           FLANK_STRIP_ACROSS)
    extents = {}
    for ri, (row, picks) in enumerate(zip(prob.rows, (pa[3], pb[3]))):
        cfgs = [dept_front(d)[ci] for d, ci in zip(row, picks)]
        u = 0
        for d, x, void in zip(row, *_row_extents(rng, cfgs, length)):
            u += void
            local = pack_department(d, x, depths[ri], rng)
            for rid, (u0, v0, w, h) in local.items():
                y0 = (top + depth_a - v0 - h if ri == 0
                      else py0 + ap + 1 + v0)
                rects[rid] = (rx0 + u + u0, y0, w, h)
            extents[d.key] = x
            u += x + 1

    # ---- rooms: the left wing and the band, then the reflection ----
    lost = [r.id for r in q.rooms if r.id not in rects]
    if lost:
        raise LayoutError(f"rooms {lost} were given no rectangle")
    axis_ducts = {dn for r in q.rooms if r.side == "axis" and "vent" in r.tags
                  for dn, _ro, _sh in G.vent_specs(r.tags["vent"])}
    rooms, ids = [], {}
    for r in q.rooms:
        x, y, w, h = rects[r.id]
        tags = dict(r.tags)
        tags["lamp_span"] = G.LAMP_SPAN
        rooms.append(G.Room(id=len(rooms), x=x, y=y, w=w, h=h, kind=r.kind,
                            tags=tags))
        ids[r.id] = (rooms[-1].id, rooms[-1].id)
    for r in q.rooms:
        left = rooms[ids[r.id][0]]
        if r.side == "axis":
            left.mirror = left.id
            if (left.x, left.x + left.w - 1) != (W - left.x - left.w,
                                                 W - 1 - left.x):
                raise G._Reject(f"axis room {r.id!r} is not centred on the "
                                f"mirror axis")
            continue
        tags = dict(left.tags)
        if "spawn_team" in tags:
            tags["spawn_team"] = 1 - tags["spawn_team"]
        if "vent" in tags:
            vt = [{"duct": dn if dn in axis_ducts else _mirror_duct(dn),
                   "roles": list(roles)}
                  for dn, roles, _sh in G.vent_specs(tags["vent"])]
            tags["vent"] = vt[0] if len(vt) == 1 else vt
        twin = G.Room(id=len(rooms), x=W - left.x - left.w, y=left.y,
                      w=left.w, h=left.h, kind=left.kind, tags=tags,
                      mirror=left.id)
        left.mirror = twin.id
        rooms.append(twin)
        ids[r.id] = (left.id, twin.id)
    _check_rooms(rooms, hull, W, H)
    by_q = {r.id: r for r in q.rooms}
    for r in q.rooms:
        _check_size(r, rooms[ids[r.id][0]])

    # ---- openings ----
    airlock_stubs = {e.a for e in q.edges if e.kind == "airlock"
                     and e.b == PASSAGEWAY}
    bypassed = {frozenset((e.a, e.b)) for e in q.edges
                if e.kind == "breakable" and any(
                    f.kind == "airlock" and {f.a, f.b} == {e.a, e.b}
                    for f in q.edges)}
    edges, twins = [], []
    for e in q.edges:
        ra, rb = rooms[ids[e.a][0]], rooms[ids[e.b][0]]
        n = G.EDGE_KINDS[e.kind].width
        wall = G.shared_wall(ra, rb)
        if wall is None or wall[3] - wall[2] + 1 < n + 2 * CLEAR:
            raise G._Reject(
                f"declared edge {e.a}-{e.b} ({e.kind}): contact "
                f"{0 if wall is None else wall[3] - wall[2] + 1} < its "
                f"opening + {2 * CLEAR} = {n + 2 * CLEAR}")
        orient, _line, lo, hi = wall
        axis = by_q[e.a].side == "axis" and by_q[e.b].side == "axis"
        at = _opening_at(rng, q, e, by_q, rooms, ids, wall, W, strip_row,
                         airlock_stubs, bypassed)
        if n and (at < lo or at + n - 1 > hi):
            raise G._Reject(f"declared edge {e.a}-{e.b} ({e.kind}): no "
                            f"place for its opening on {lo}..{hi}")
        edges.append(G.Edge(ra.id, rb.id, e.kind, at if n else None))
        if not axis:
            ta, tb = ids[e.a][1], ids[e.b][1]
            tat = at if (orient == "v" or not n) else W - at - n
            twins.append(G.Edge(ta, tb, e.kind, tat if n else None))
    ducts = {}
    wing_ducts = {dn for r in q.rooms if r.side != "axis" and "vent" in r.tags
                  for dn, _ro, _sh in G.vent_specs(r.tags["vent"])}
    for name, filt in q.ducts:
        ducts[name] = {"filter": filt}
        if name in wing_ducts and name not in axis_ducts:
            ducts[_mirror_duct(name)] = {"filter": filt}
    graph = G.ShipGraph(width=W, height=H, hull=hull, rooms=rooms,
                        edges=edges + twins, mirror_x=True,
                        max_per_team=q.max_per_team, zombies=q.zombies,
                        ducts=ducts)
    return Layout(graph=graph, ids=ids, params={
        "inset_top": it, "inset_bottom": ib, "band_w": wm,
        "passageway_across": ap, "passageway_y": py0, "lobby_x": lx,
        "slab_x": sx, "strip_row": strip_row, "row_length": length,
        "row_depths": depths, "x_extents": extents})


def _row_extents(rng, cfgs: list, length: int) -> tuple:
    """The x-extents of a row's departments and the steel before each (0
    before the first, 0..VOID_MAX between neighbours), filling ``length``
    exactly: one split, drawn uniformly among those with the least steel
    (layout doc §3: slack goes into the rooms first)."""
    sols = []

    def rec(i, used, xs, voids):
        if i == len(cfgs):
            if used == length:
                sols.append((xs, voids))
            return
        for v in ((0,) if i == 0 else range(VOID_MAX + 1)):
            base = used + (0 if i == 0 else 1 + v)
            m, x = cfgs[i].xs, 0
            while m:
                if m & 1 and base + x <= length:
                    rec(i + 1, base + x, xs + (x,), voids + (v,))
                m >>= 1
                x += 1

    rec(0, 0, (), ())
    least = min((sum(v) for _x, v in sols), default=0)
    return _pick(rng, [t for t in sols if sum(t[1]) == least])


def _band_rects(band: Band, wm: int, cx0: int, iy0: int, height: int,
                zombies: int, strip_row, rng, rects: dict) -> None:
    """THE BAND PACKER (layout doc §5, §13 A2, §4.4): the sections stacked
    fore to aft, each spanning the band -- the block's heights drawn inside
    their ranges, the main sections cut by ``gen_ship._split_1d`` (the
    zombie section tall enough for its horde); with a ring, the hub's row
    is a side column, the hub and the mirrored side column, the side column
    cut into the ring's side rooms aft to fore."""
    items = _band_items(band, wm, zombies, strip_row)
    if items is None:
        raise G._Reject("band: the band width cannot take the block")
    nm = len(band.mains)
    mins = [lo for _i, lo, _h in items]
    slack = height - (len(items) - 1) - sum(mins)
    if slack < 0:
        raise G._Reject("band: the sections do not fit the hull's height")
    heights = list(mins)
    for i in range(nm, len(items)):                 # the block, fore to aft
        add = _draw(rng, 0, min(items[i][2] - items[i][1], slack))
        heights[i] += add
        slack -= add
    block_h = sum(heights[nm:]) + (len(items) - nm)
    extra = mins[nm - 1] - MIN_SECTION              # the zombie section's
    y1 = iy0 + height - 1 - block_h
    cuts = G._split_1d(rng, iy0, y1 - extra, nm, MIN_SECTION)
    if len(cuts) != nm - 1:
        raise G._Reject("band: the main sections do not fit")
    starts = [iy0] + [c + 1 for c in cuts]
    ends = [c - 1 for c in cuts] + [y1]
    for m, a, b in zip(band.mains, starts, ends):
        rects[m] = (cx0, a, wm, b - a + 1)
    y = y1 + 2
    for bi, ((rid, role, _w, _h), h) in enumerate(zip(band.block,
                                                      heights[nm:])):
        if role == "hub" and band.sides:
            d = RING_DEPTH
            rects[rid] = (cx0 + d + 1, y, wm - 2 * (d + 1), h)
            need = _side_needs(band, _band_cut(len(items), nm + bi,
                                               strip_row))
            sizes = pack(rng, h, [(n, _INF) for n in need])
            yy = y + h
            for sid, s in zip(band.sides, sizes):   # aft to fore
                rects[sid] = (cx0, yy - s, d, s)
                yy -= s + 1
        else:
            rects[rid] = (cx0, y, wm, h)
        y += h + 1


def _check_rooms(rooms: list, hull: tuple, W: int, H: int) -> None:
    """Every room inside the hull, no two rooms sharing or touching a tile
    without a wall between them (a layout bug, never a draw)."""
    hx0, hy0, hx1, hy1 = hull
    grid = np.zeros((H, W), dtype=np.int32)
    for rm in rooms:
        x0, y0, x1, y1 = rm.box
        if x0 <= hx0 or y0 <= hy0 or x1 >= hx1 or y1 >= hy1:
            raise AssertionError(f"layout: room {rm.id} {rm.box} leaves the "
                                 f"hull {hull}")
        if grid[y0:y1 + 1, x0:x1 + 1].any():
            raise AssertionError(f"layout: room {rm.id} {rm.box} overlaps "
                                 f"another room")
        grid[y0:y1 + 1, x0:x1 + 1] = rm.id + 1
    a, b = grid[:, :-1], grid[:, 1:]
    c, d = grid[:-1, :], grid[1:, :]
    if ((a != b) & (a > 0) & (b > 0)).any() \
            or ((c != d) & (c > 0) & (d > 0)).any():
        raise AssertionError("layout: two rooms touch with no wall between")


def _check_size(r, rm) -> None:
    """A placed room lies inside its quotient size ranges (the layout
    rotates a corridor: its ``across`` is its short side)."""
    s = r.size
    if s is None:
        return
    if s.axes == ("w", "h"):
        got = (rm.w, rm.h)
    else:
        got = (min(rm.w, rm.h), max(rm.w, rm.h))
    for axis, rng_, v in zip(s.axes, (s.a, s.b), got):
        if rng_ is not None and not rng_[0] <= v <= rng_[1]:
            raise AssertionError(f"layout: room {r.id!r} {axis} = {v} leaves "
                                 f"its range {list(rng_)}")


def _opening_at(rng, q, e, by_q, rooms, ids, wall, W, strip_row,
                airlock_stubs, bypassed) -> int:
    """Where an opening starts along its shared wall (layout doc §4 step 5,
    §5, §13 A1): centred on the axis (even width only); an entry on its
    stub; the skeleton's passageway doors on the passageway; the lobby's
    airlock as far aft as its mounts allow and a bypass's breakable at the
    fore end; an opening onto a promoted stub beyond its vestibule; else
    drawn inside the run, one tile clear of each end."""
    orient, _line, lo, hi = wall
    n = G.EDGE_KINDS[e.kind].width
    if not n:
        return lo
    a, b = by_q[e.a], by_q[e.b]
    run = hi - lo + 1
    if a.side == "axis" and b.side == "axis" and orient == "h":
        if n % 2 or (lo + hi + 1) != W:
            raise G._Reject(f"axis edge {e.a}-{e.b} ({e.kind}): an opening "
                            f"between two axis rooms is centred and of even "
                            f"width")
        return W // 2 - n // 2
    pair = frozenset((e.a, e.b))
    if e.kind == "airlock" and e.a == LOBBY:
        return hi - n - CLEAR + 1                   # aft-most, a mount clear
    if e.kind == "breakable" and pair in bypassed:
        return lo + CLEAR
    if e.kind == "airlock":
        return lo + (run - n) // 2                  # a stub's entry airlock
    centred = (e.role in ("entry", "skeleton")
               or STUB2 in (a.role, b.role) and PASSAGEWAY in (e.a, e.b))
    if centred and LOBBY in (e.a, e.b) and SECTION in (a.role, b.role):
        centred = False                             # the lobby's band door
    if centred:
        return lo + (run - n) // 2
    for s, o in ((e.a, e.b), (e.b, e.a)):
        if s in airlock_stubs and orient == "v":    # a promoted stub's side
            st = rooms[ids[s][0]]
            pw = rooms[ids[PASSAGEWAY][0]]
            if st.y > pw.y:                         # row B: vestibule on top
                lo = max(lo, st.y + PROMOTED_CLEAR)
            else:
                hi = min(hi, st.y + st.h - 1 - PROMOTED_CLEAR)
            if hi - lo + 1 < n + 2 * CLEAR:
                raise G._Reject(f"edge {e.a}-{e.b}: no wall beyond the "
                                f"entry airlock's vestibule")
    if FLANK_STRIP in (a.role, b.role) and LOBBY in (e.a, e.b) \
            and strip_row == ROW_B:
        return lo + CLEAR                           # clear of the vestibule
    return G._at(rng, lo, hi, n)


# ===========================================================================
# Acceptance: the layout's own A* and the archetype's [validation] gates
# ===========================================================================

def bot_path(emb) -> list:
    """The bots' walk between the two spawn slabs (``pathfinding.astar`` on
    the 3x3 footprint grid) with every airlock's outer door CLOSED, as the
    tiles are (#142: no bot cycles an airlock); empty when there is none."""
    from numpy.lib.stride_tricks import sliding_window_view
    tm = emb.tilemap
    _mat, vac, _solid, walk = G._masks(tm)
    fp = FORMATION_FOOTPRINT
    ok = sliding_window_view(walk & ~vac, (fp, fp)).all(axis=(2, 3))
    h, w = tm.shape
    (ax, ay), (bx, by) = (G._centre(emb.spawn_rooms[0]),
                          G._centre(emb.spawn_rooms[1]))
    return G.astar(ax, ay, bx, by, lambda x, y: not ok[y, x], w, h)


def acceptance(arch, q, graph, emb) -> list:
    """The recipe path's gates AFTER ``gen_ship.validate`` (archetype doc
    §1 ``[validation]``, layout doc §2): the failure reasons, empty = pass.
    ALWAYS: the two slabs are joined by a bot-walkable path (airlock outer
    doors closed). From the archetype's registry, when it names them:
    ``spawn_distance`` (that path's steps, per size), ``cells`` (pressure
    cells per ship), ``rooms`` (rooms per ship, per size)."""
    reasons = []
    path = bot_path(emb)
    if not path:
        reasons.append("bot_walkable: no path between the spawn slabs with "
                       "the airlocks' outer doors closed")
    val = arch.validation if arch is not None else {}
    if path and "spawn_distance" in val:
        lo, hi = val["spawn_distance"][q.size]
        if not lo <= len(path) - 1 <= hi:
            reasons.append(f"spawn_distance: {len(path) - 1} steps outside "
                           f"{[lo, hi]}")
    if "cells" in val:
        lo, hi = val["cells"]
        n = len({rm.tags.get("cell") for rm in graph.rooms})
        if not lo <= n <= hi:
            reasons.append(f"cells: {n} outside {[lo, hi]}")
    if "rooms" in val:
        lo, hi = val["rooms"][q.size]
        if not lo <= len(graph.rooms) <= hi:
            reasons.append(f"rooms: {len(graph.rooms)} outside {[lo, hi]}")
    return reasons
