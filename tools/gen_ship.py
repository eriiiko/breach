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
cut into its shared wall as its row of the EDGE-TYPE TABLE (:data:`EDGE_KINDS`,
the vocabulary doc's §0.4: door = ``MAT_DOOR``, window = ``MAT_GLASS``, open
= ``MAT_AIR``, breakable = ``MAT_WOOD``, bulkhead / vent_only = nothing cut;
entity_door = an A6 ``door`` entity in the opening; airlock = the L5 recipe,
a vestibule carved on room ``a``'s side with two doors and levels/airlock_demo's
controller and wiring; locked is accepted by the schema and refused here
until #135), per-room tags from :data:`ROOM_TAGS`,
fuel clusters (furniture / wood) against the walls of tagged rooms clear of
doors (placed in one room of a mirror pair and reflected into its twin),
the ducts (one per duct name the ``vent`` tags use, a filter each) and their
vents on inner walls of their tagged rooms, one heat-triggered
``timed_charge`` per ``hazard``-tagged room (the magazine's cook-off, the
hazmat store; vocabulary E.2 row 1), one ``heat_source`` per
``heat_source``-tagged room (the engine, the galley stove; E.2 row 3), both
on the one floor-site rule, the ``temp``-tagged rooms' floor air seeded hot
in ``temp_init.npy`` (hot engineering; E.2 row 4), one static lamp per room
(render light only), the spawn rooms from the tags. It reads only the graph (and one seeded generator for the clusters).

GATES (:func:`validate`), before anything is written: airtight
(``level_airtight.find_leaks``), A* (``pathfinding.astar``) between the
spawn rooms and from each to the zombie room, every compartment reachable
by a 3x3 unit (a ``secret`` room: through a breakable or vent_only edge) and
exactly the graph's COMPARTMENTS realised (rooms an ``open`` edge joins are
one; the structure mask takes the graph's edge kinds), every formation cell
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
from level_loader import (  # noqa: E402
    LEVEL_FORMAT_VERSION, SPACE_CODE, EntityInstance, LightEntry, SpawnEntry,
)
from pathfinding import astar  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gas_fixed import quantize_q16_scalar  # noqa: E402
from simulation.match import (  # noqa: E402
    FORMATION_FOOTPRINT, FORMATION_GAP, FORMATION_PITCH, MatchSpec,
    MatchTeam, SpawnRoom, formation_capacity, placement_cells,
)
from simulation.entities.door import (  # noqa: E402
    quantize_span_tiles, tiles_per_m,
)
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED, MAT_FURNITURE, MAT_GLASS, MAT_HULL,
    MAT_STEEL, MAT_WOOD,
)

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

# Entity doors (A6). The bots (control_bots) never USE a door and a closed
# entity door stamps MAT_DOOR_CLOSED (mobility 0), which no unit crosses --
# so stage 2 emits every entity door OPEN (MAT_AIR in the CSV, the loader's
# own stamp) until a door-opening rule exists (#142). Still a compartment
# boundary in the gates: it can close.
ENTITY_DOOR_INITIAL = "open"

# Airlock pairs (L5): a VESTIBULE carved out of room ``a``'s side of the
# shared wall (the wall itself is one tile: there is nothing to carve
# there), AIRLOCK_W tiles along the wall by VESTIBULE_D deep, its inner wall
# one tile; the doors span the vestibule's full width (levels/airlock_demo's
# shape: a 3x3 unit needs a 5-wide passage for three anchor rows). The
# equipment and its wiring are levels/airlock_demo's exactly
# (docs/arc_b_impl_2026-07-21.md §7): inner door open, outer door closed,
# a pump + a pressure probe in the vestibule's side wall, a presence plate
# at its centre (3 tiles, the demo's radius in tiles), two deciders, the
# controller evacuating toward the far side and repressurising after.
AIRLOCK_W = 5
VESTIBULE_D = 5
AIRLOCK_PUMP = {"rate": 8.0, "target_atm": 0, "hysteresis_band": 13107}
AIRLOCK_PLATE_TILES = 3
AIRLOCK_DEC_FAR = ("le", 6554)       # chamber <= 0.1 atm
AIRLOCK_DEC_NEAR = ("ge", 58982)     # chamber >= 0.9 atm
AIRLOCK_CTRL = {"far_door": "outer", "target_far_atm": 0,
                "target_near_atm": 65536}
# The INTERIOR airlock (#133, Erik's ruling 2026-10-31,
# .claude/skills/level-design-lead/SKILL.md §3): between two pressurised
# cells the airlock only sequences its doors and equalises the vestibule to
# the side being opened -- only a HULL airlock (the far side is space)
# evacuates. :func:`airlock_mode` derives which from the tiles beyond the
# outer door. An interior airlock carries two TRANSFER pumps (one per side,
# mounted in the wall beside each door, moving air between the chamber and
# that side) and three pressure probes (chamber, near side, far side) instead
# of the external pump and the two deciders; the rest is the demo's recipe.
AIRLOCK_MODES = ("hull", "interior")
AIRLOCK_TRANSFER_PUMP = {"rate": 8.0, "target_atm": 65536,
                         "hysteresis_band": 13107, "feed": "transfer"}
AIRLOCK_EQUALIZE_BAND = 3277         # |side - chamber| <= 0.05 atm ends a leg
AIRLOCK_CTRL_INTERIOR = {"far_door": "outer", "mode": "interior",
                         "equalize_band": AIRLOCK_EQUALIZE_BAND}


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


class GraphSchemaError(ValueError):
    """A ShipGraph that breaks the schema (an unknown edge kind or room tag,
    a tag of the wrong shape). A programming error in stage 1, never a
    draw to retry: :func:`generate` lets it through."""


# ===========================================================================
# The edge-type table -- docs/levelgen_zone_vocabulary_2026-10-29.md §0.4
# ===========================================================================

L9_MUST_SHARE_WALL = "must-share-wall"
L9_MAY_BECOME_CORRIDOR = "may-become-corridor"
L9_ROUTED_FREELY = "routed-freely"
L9_STRENGTHS = (L9_MUST_SHARE_WALL, L9_MAY_BECOME_CORRIDOR, L9_ROUTED_FREELY)

@dataclass(frozen=True)
class EdgeKind:
    """One row of the vocabulary's edge-type table (§0.4). ``stamps``: the
    material stage 2 puts in the shared wall's opening (None: nothing is
    cut, the wall stays steel; for an entity kind, what its door's initial
    state stamps there, so the CSV and the loader's door stamp agree);
    ``width``: tiles along the wall; ``bounds``: the edge bounds a
    compartment (False: its two rooms are ONE compartment);
    ``secret_path``: a ``secret`` room reached only through this edge counts
    as reachable (cut the wall / open the duct); ``keep_clear``: a unit
    passes here, so fuel keeps ``FUEL_DOOR_CLEAR`` off it; ``l9``: the L9
    adjacency strength (docs/breach_levelgen_design_v0.1.md); ``entity``:
    what stage 2 emits besides tiles (``"door"``: one A6 door entity in the
    opening; ``"airlock"``: the L5 recipe, :func:`_airlock`); ``pending``:
    None when stage 2 stamps it, else what will (the schema accepts the
    kind, :func:`embed` refuses it naming that)."""
    stamps: int | None
    width: int
    bounds: bool
    secret_path: bool
    keep_clear: bool
    l9: str
    pending: str | None = None
    entity: str | None = None


_DOOR_STAMP = {"open": MAT_AIR, "closed": MAT_DOOR_CLOSED}

# THE table: a new way two rooms join is a new row here, never an inline
# stamp. Same rows, same order as the vocabulary's §0.4.
EDGE_KINDS = {
    "door":        EdgeKind(MAT_DOOR, DOOR_W, True, False, True,
                            L9_MUST_SHARE_WALL),
    "entity_door": EdgeKind(_DOOR_STAMP[ENTITY_DOOR_INITIAL], DOOR_W, True,
                            False, True, L9_MUST_SHARE_WALL, entity="door"),
    "open":        EdgeKind(MAT_AIR, DOOR_W, False, False, True,
                            L9_MUST_SHARE_WALL),
    "window":      EdgeKind(MAT_GLASS, GLASS_W, True, False, False,
                            L9_MUST_SHARE_WALL),
    "bulkhead":    EdgeKind(None, 0, True, False, False,
                            L9_MUST_SHARE_WALL),
    # The opening is the OUTER door (closed); the inner door and the
    # vestibule are carved on room a's side (_airlock).
    "airlock":     EdgeKind(_DOOR_STAMP["closed"], AIRLOCK_W, True, False,
                            True, L9_MUST_SHARE_WALL, entity="airlock"),
    "vent_only":   EdgeKind(None, 0, True, True, False,
                            L9_MUST_SHARE_WALL),
    "breakable":   EdgeKind(MAT_WOOD, DOOR_W, True, True, True,
                            L9_MUST_SHARE_WALL),
    "locked":      EdgeKind(None, DOOR_W, True, False, True,
                            L9_MUST_SHARE_WALL,
                            pending="#135 (terminals: a locked door is an "
                                    "entity door + a terminal)"),
}
# Kinds that may cross a cell boundary (L5): everything else is refused.
CELL_CROSSING_KINDS = ("airlock", "bulkhead")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) \
        and not isinstance(v, bool)


def _check_light(v) -> bool:
    if not (isinstance(v, (tuple, list)) and len(v) == 2):
        return False
    rgb, level = v
    return (isinstance(rgb, (tuple, list)) and len(rgb) == 3
            and all(_is_num(c) and c >= 0 for c in rgb)
            and _is_num(level) and level >= 0)


VENT_ROLES = ("supply", "return")
MAIN_DUCT = "main"


def _check_vent(v) -> bool:
    """v1's shorthand ``"supply"`` / ``"return"``, or ``{"duct": name,
    "roles": [...]}`` with 1-2 distinct roles and a plain id-safe name."""
    if v in VENT_ROLES:
        return True
    if not (isinstance(v, dict) and set(v) == {"duct", "roles"}):
        return False
    name, roles = v["duct"], v["roles"]
    return (isinstance(name, str) and name != ""
            and all(c.isalnum() or c == "_" for c in name)
            and isinstance(roles, (list, tuple)) and 1 <= len(roles) <= 2
            and len(set(roles)) == len(roles)
            and all(r in VENT_ROLES for r in roles))


def _check_hazard(v) -> bool:
    """``{"payload": row, "trigger_temp": T > 0, "fuse_s": s >= 0}``."""
    return (isinstance(v, dict)
            and set(v) == {"payload", "trigger_temp", "fuse_s"}
            and isinstance(v["payload"], str) and v["payload"] != ""
            and _is_num(v["trigger_temp"]) and v["trigger_temp"] > 0
            and _is_num(v["fuse_s"]) and v["fuse_s"] >= 0)


def _check_heat_source(v) -> bool:
    """``{"heat_per_s": > 0, "radius": int >= 0, "max_temp": >= 0,
    "death_payload": row name or ""}``."""
    return (isinstance(v, dict)
            and set(v) == {"heat_per_s", "radius", "max_temp",
                           "death_payload"}
            and _is_num(v["heat_per_s"]) and v["heat_per_s"] > 0
            and _is_int(v["radius"]) and v["radius"] >= 0
            and _is_num(v["max_temp"]) and v["max_temp"] >= 0
            and isinstance(v["death_payload"], str))


def vent_spec(tag):
    """A ``vent`` tag as ``(duct, roles, shorthand)``: roles in
    :data:`VENT_ROLES` order; ``shorthand`` True for v1's string form (on
    the ``main`` duct, keeping v1's vent ids)."""
    if isinstance(tag, str):
        return MAIN_DUCT, (tag,), True
    return tag["duct"], tuple(r for r in VENT_ROLES if r in tag["roles"]), \
        False


# THE per-room tag schema (all keys optional): key -> (what it is, check).
# A department writes these; stage 2 reads ``spawn_team``, ``zombie``,
# ``fuel``, ``vent``, ``hazard``, ``heat_source``, ``temp`` and ``light``, and carries ``dept``, ``role``,
# ``cell`` and ``secret`` into ``Embedding.room_tags`` for the vent builder
# and the grammar. Generator-side data: none of them is a level field.
ROOM_TAGS = {
    "spawn_team": ("int: the team seated here (a spawn room)", _is_int),
    "zombie":     ("bool: the zombies' spawn room",
                   lambda v: isinstance(v, bool)),
    "fuel":       ("int >= 0: fuel clusters against the walls",
                   lambda v: _is_int(v) and v >= 0),
    "vent":       ('"supply" | "return" (a vent on the main duct) or '
                   '{"duct": name, "roles": ["supply"] | ["return"] | '
                   '["supply", "return"]}', _check_vent),
    "dept":       ("str: the department this room belongs to",
                   lambda v: isinstance(v, str)),
    "role":       ("str: the room's role in its department",
                   lambda v: isinstance(v, str)),
    "cell":       ("int: the pressure cell (L5)", _is_int),
    "light":      ("((r, g, b), level): the room's lamp, replacing LAMP_RGB "
                   "/ LAMP_INTENSITY", _check_light),
    "secret":     ("bool: a hidden room, reachable through a breakable or "
                   "vent_only edge", lambda v: isinstance(v, bool)),
    "hazard":     ('{"payload": [payloads.*] row, "trigger_temp": game '
                   'degrees > 0, "fuse_s": seconds >= 0}: one heat-triggered '
                   "timed_charge on a floor tile clear of doors and fuel "
                   "(the magazine's cook-off)", _check_hazard),
    "heat_source": ('{"heat_per_s": > 0, "radius": tiles >= 0, "max_temp": '
                    'game degrees >= 0 (0 = no cap), "death_payload": '
                    '[payloads.*] row or ""}: one heat_source entity on a '
                    "floor tile clear of doors, fuel and other entities "
                    "(the engine, the galley stove)", _check_heat_source),
    "temp":       ("float > 0: the room's floor AIR starts this many game "
                   "degrees above ambient (temp_init.npy; hot engineering)",
                   lambda v: _is_num(v) and v > 0),
}
# The tags carried into Embedding.room_tags (the rest are stage 2's inputs).
ANNOTATION_TAGS = ("dept", "role", "cell", "secret", "light")


# ===========================================================================
# The room graph -- stage 1's product, stage 2's only input
# ===========================================================================

@dataclass
class Room:
    """A graph node. ``x, y, w, h`` is its AIR rectangle (the geometry
    annotation stage 2 embeds); ``kind`` names its role; ``tags`` holds the
    optional keys of :data:`ROOM_TAGS` (any other key is refused when the
    graph is built); ``mirror`` is its twin's id under the graph's mirror
    (itself on the axis, None if none). Rooms joined by ``open`` edges
    form ONE compartment."""
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
    ``kind`` a key of :data:`EDGE_KINDS`; ``at`` is the opening's first
    tile along the shared wall (None: stage 2 centres it)."""
    a: int
    b: int
    kind: str = "door"
    at: int | None = None

    @property
    def spec(self) -> EdgeKind:
        return EDGE_KINDS[self.kind]


@dataclass
class ShipGraph:
    """Stage 1's product: the ship as rooms + edges + annotations. Checked
    against the schema when built (:meth:`check`), and again by
    :func:`embed`."""
    width: int
    height: int
    hull: tuple                   # (x0, y0, x1, y1) outer hull faces, inclusive
    rooms: list                   # list[Room], rooms[i].id == i
    edges: list                   # list[Edge]
    mirror_x: bool = False        # left-right mirrored about the grid centre
    max_per_team: int = 6
    zombies: int = 0
    # Per duct name: {"filter": a [filters.*] row}. A duct not named here
    # takes FILTER (hepa_basic), which is v1's main duct.
    ducts: dict = field(default_factory=dict)

    def __post_init__(self):
        self.check()

    def duct_filter(self, name: str) -> str:
        return self.ducts.get(name, {}).get("filter", FILTER)

    def check(self) -> None:
        """Refuse an unknown edge kind, an unknown room tag key, a tag of
        the wrong shape, a secret room that seats a team, and the duct /
        cell rules (:meth:`_check_ducts_and_cells`), naming the offender."""
        for i, rm in enumerate(self.rooms):
            if rm.id != i:
                raise GraphSchemaError(f"room {i} has id {rm.id} (rooms[i].id "
                                       f"must be i)")
            unknown = sorted(set(rm.tags) - set(ROOM_TAGS))
            if unknown:
                raise GraphSchemaError(
                    f"room {rm.id}: unknown tag key(s) {unknown} (known: "
                    f"{sorted(ROOM_TAGS)})")
            for k, v in rm.tags.items():
                what, ok = ROOM_TAGS[k]
                if not ok(v):
                    raise GraphSchemaError(
                        f"room {rm.id}: tag {k!r} = {v!r} is not {what}")
            if rm.tags.get("secret") and ("spawn_team" in rm.tags
                                          or rm.tags.get("zombie")):
                raise GraphSchemaError(
                    f"room {rm.id}: a secret room cannot be a spawn room")
        n = len(self.rooms)
        for e in self.edges:
            if e.kind not in EDGE_KINDS:
                raise GraphSchemaError(
                    f"edge {e.a}-{e.b}: unknown kind {e.kind!r} (known: "
                    f"{list(EDGE_KINDS)})")
            if not (0 <= e.a < n and 0 <= e.b < n) or e.a == e.b:
                raise GraphSchemaError(
                    f"edge {e.a}-{e.b}: not two distinct rooms of {n}")
        self._check_ducts_and_cells()

    def _check_ducts_and_cells(self) -> None:
        """The vocabulary's §0.5 and L5 rules on the graph:

        - the ``ducts`` annotation names a known ``[filters.*]`` row;
        - a duct's rooms lie in ONE pressure cell (``cell`` tag; rooms
          without one share an implicit cell): a duct never crosses a cell
          boundary;
        - a SEALED room (every edge it has bounds a compartment) carrying a
          ``vent`` tag carries BOTH roles (the vents have no pressure
          feedback: a lone supply pumps it up, a lone return drains it).
          v1's string shorthand is exempt: v1's ``--vents`` pair is exactly
          such a lone supply / lone return (critique finding 2), and
          levels/ship_demo is pinned byte-identical;
        - a ``vent_only`` edge joins two rooms ON ONE DUCT (the air path is
          the duct; nothing is cut in the wall);
        - an edge whose rooms lie in different cells is an airlock or a
          bulkhead, never a plain door (L5)."""
        from config import CFG
        filters = getattr(CFG, "filters", None)
        for name, ann in self.ducts.items():
            if not (isinstance(ann, dict) and set(ann) <= {"filter"}
                    and isinstance(ann.get("filter", FILTER), str)):
                raise GraphSchemaError(
                    f"duct {name!r}: annotation {ann!r} is not "
                    f"{{'filter': <a [filters.*] row>}}")
            fname = self.duct_filter(name)
            if getattr(filters, fname, None) is None:
                raise GraphSchemaError(
                    f"duct {name!r}: filter {fname!r} has no [filters."
                    f"{fname}] row in config.toml")
        specs = {rm.id: vent_spec(rm.tags["vent"]) for rm in self.rooms
                 if "vent" in rm.tags}
        cells: dict = {}
        for rid, (duct, _roles, _sh) in specs.items():
            cells.setdefault(duct, {}).setdefault(
                self.rooms[rid].tags.get("cell"), []).append(rid)
        for duct, by_cell in sorted(cells.items()):
            if len(by_cell) > 1:
                desc = ", ".join(f"cell {c}: rooms {ids}" for c, ids in
                                 sorted(by_cell.items(), key=repr))
                raise GraphSchemaError(
                    f"duct {duct!r} spans {len(by_cell)} cells ({desc}): a "
                    f"duct never crosses a cell boundary (vocabulary §0.5)")
        for rid, (duct, roles, shorthand) in sorted(specs.items()):
            if shorthand or len(roles) == 2:
                continue
            mine = [e for e in self.edges if rid in (e.a, e.b)]
            if all(e.spec.bounds for e in mine):
                raise GraphSchemaError(
                    f"room {rid} is sealed (every edge bounds a compartment) "
                    f"and carries only a {roles[0]!r} vent on duct {duct!r}: "
                    f"a sealed room carries both a supply and a return or "
                    f"neither (vocabulary §0.4 / §0.5)")
        for e in self.edges:
            if e.kind == "vent_only":
                da = specs.get(e.a, (None,))[0]
                db = specs.get(e.b, (None,))[0]
                if da is None or da != db:
                    raise GraphSchemaError(
                        f"edge {e.a}-{e.b} (vent_only): rooms {e.a} and "
                        f"{e.b} are not on one duct (room {e.a}: {da!r}, "
                        f"room {e.b}: {db!r}) -- a vent_only edge's air "
                        f"path IS the duct")
            ca = self.rooms[e.a].tags.get("cell")
            cb = self.rooms[e.b].tags.get("cell")
            if ca != cb and e.kind not in CELL_CROSSING_KINDS:
                raise GraphSchemaError(
                    f"edge {e.a}-{e.b}: kind {e.kind!r} crosses cells {ca} "
                    f"-> {cb}; an edge crossing a cell boundary is an "
                    f"airlock or a bulkhead, never a plain door (L5)")

    def compartment_of(self) -> list:
        """Room id -> compartment index: rooms joined by an edge that does
        not bound a compartment (``open``) are one; indices run 0..k-1 in
        the order of each compartment's lowest room id."""
        uf = _Union(len(self.rooms))
        for e in self.edges:
            if not e.spec.bounds:
                uf.union(e.a, e.b)
        idx, out = {}, []
        for i in range(len(self.rooms)):
            out.append(idx.setdefault(uf.find(i), len(idx)))
        return out


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
    """The (y, x) tiles an edge opens in its shared wall (none for a kind
    that cuts nothing; the rooms must still share a wall: every kind in the
    table is ``must-share-wall``)."""
    wall = shared_wall(graph.rooms[e.a], graph.rooms[e.b])
    if wall is None:
        raise _Reject(f"edge {e.a}-{e.b}: the rooms share no wall")
    orient, line, lo, hi = wall
    n = e.spec.width
    if e.spec.stamps is None:
        return []
    at = lo + (hi - lo + 1 - n) // 2 if e.at is None else e.at
    if at < lo or at + n - 1 > hi:
        raise _Reject(f"edge {e.a}-{e.b}: opening {at}+{n} leaves {lo}..{hi}")
    if orient == "v":
        return [(y, line) for y in range(at, at + n)]
    return [(line, x) for x in range(at, at + n)]


@dataclass(frozen=True)
class Airlock:
    """The L5 recipe of one ``airlock`` edge, as geometry (a pure function
    of the graph, so stage 2 and the gates derive the same one). ``wall``:
    the shared wall's orientation (``"v"``/``"h"``), which is also both
    doors' span orientation; ``outer``: the outer door's tiles (the edge's
    opening, in the shared wall, row-major); ``inner``: the inner door's
    tiles (the vestibule's wall facing room ``a``); ``chamber``: the
    vestibule's air box ``(x0, y0, x1, y1)``; ``walls``: the new wall tiles
    stage 2 sets to steel inside room ``a``; ``carve``: every tile of room
    ``a``'s box the vestibule takes (chamber + walls + inner door);
    ``pump`` / ``probe``: ``(x, y, dx, dy)`` mount tile in the vestibule's
    side wall and the offset of the chamber tile it acts on; ``plate``: the
    presence plate's tile ``(x, y)``, the chamber's centre.

    The interior recipe's equipment (#133): ``beyond``: the tiles one step
    past the outer door on room ``b``'s side (what :func:`airlock_mode`
    reads); ``far_pump`` / ``near_pump``: ``(x, y, port_dx, port_dy,
    source_dx, source_dy)`` -- a transfer pump in the wall beside the outer /
    inner door, its port in the chamber and its source in front of that
    door; ``far_probe`` / ``near_probe``: ``(x, y, dx, dy)``, a pressure
    probe beside the other end of the same door sampling that side."""
    wall: str
    outer: tuple
    inner: tuple
    chamber: tuple
    walls: tuple
    carve: tuple
    pump: tuple
    probe: tuple
    plate: tuple
    beyond: tuple = ()
    far_pump: tuple = ()
    near_pump: tuple = ()
    far_probe: tuple = ()
    near_probe: tuple = ()


def _airlock(graph: ShipGraph, e: Edge) -> Airlock:
    """The vestibule of an ``airlock`` edge, carved on room ``e.a``'s side
    of the shared wall: ``AIRLOCK_W`` along the wall (the outer door's
    span) by ``VESTIBULE_D`` deep, closed by a one-tile inner wall holding
    the inner door (the same span, so the passage is straight) and two
    side walls. Room ``a`` keeps at least a unit's footprint of floor in
    front of the inner door, or the plan is rejected."""
    outer = _opening_tiles(graph, e)
    a = graph.rooms[e.a]
    ax0, ay0, ax1, ay1 = a.box
    orient, line, _lo, _hi = shared_wall(a, graph.rooms[e.b])
    D, fp = VESTIBULE_D, FORMATION_FOOTPRINT
    if orient == "v":
        depth = a.w
        y0, y1 = outer[0][0], outer[-1][0]
        if ax1 + 1 == line:                         # a is left of the wall
            cx0, cx1, iw = line - D, line - 1, line - D - 1
            s = 1                                   # b lies toward +x
        else:
            cx0, cx1, iw = line + 1, line + D, line + D + 1
            s = -1
        chamber = (cx0, y0, cx1, y1)
        inner = tuple((y, iw) for y in range(y0, y1 + 1))
        wx0, wx1 = min(cx0, iw), max(cx1, iw)
        walls = [(y0 - 1, x) for x in range(wx0, wx1 + 1)] + \
                [(y1 + 1, x) for x in range(wx0, wx1 + 1)] + \
                [(y0 - 1, iw), (y1 + 1, iw)]
        mid = (cx0 + cx1) // 2
        pump, probe = (mid, y0 - 1, 0, 1), (mid + 1, y0 - 1, 0, 1)
        plate = (mid, (y0 + y1) // 2)
        beyond = tuple((y, line + s) for y in range(y0, y1 + 1))
        far_pump = (line, y0 - 1, -s, 1, s, 1)
        near_pump = (iw, y0 - 1, s, 1, -s, 1)
        far_probe = (line, y1 + 1, s, -1)
        near_probe = (iw, y1 + 1, -s, -1)
    else:
        depth = a.h
        x0, x1 = outer[0][1], outer[-1][1]
        if ay1 + 1 == line:                         # a is above the wall
            cy0, cy1, iw = line - D, line - 1, line - D - 1
            s = 1                                   # b lies toward +y
        else:
            cy0, cy1, iw = line + 1, line + D, line + D + 1
            s = -1
        chamber = (x0, cy0, x1, cy1)
        inner = tuple((iw, x) for x in range(x0, x1 + 1))
        wy0, wy1 = min(cy0, iw), max(cy1, iw)
        walls = [(y, x0 - 1) for y in range(wy0, wy1 + 1)] + \
                [(y, x1 + 1) for y in range(wy0, wy1 + 1)]
        mid = (cy0 + cy1) // 2
        pump, probe = (x0 - 1, mid, 1, 0), (x0 - 1, mid + 1, 1, 0)
        plate = ((x0 + x1) // 2, mid)
        beyond = tuple((line + s, x) for x in range(x0, x1 + 1))
        far_pump = (x0 - 1, line, 1, -s, 1, s)
        near_pump = (x0 - 1, iw, 1, s, 1, -s)
        far_probe = (x1 + 1, line, -1, s)
        near_probe = (x1 + 1, iw, -1, -s)
    if depth < D + 1 + fp:
        raise _Reject(f"airlock {e.a}-{e.b}: room {e.a} ({depth} tiles deep) "
                      f"cannot hold a {D}-deep vestibule and a unit's "
                      f"footprint in front of it")
    walls = tuple(sorted({(y, x) for (y, x) in walls
                          if ax0 <= x <= ax1 and ay0 <= y <= ay1}))
    cx0, cy0, cx1, cy1 = chamber
    cells = {(y, x) for y in range(cy0, cy1 + 1) for x in range(cx0, cx1 + 1)}
    carve = tuple(sorted(cells | set(walls) | set(inner)))
    return Airlock(wall=orient, outer=tuple(outer), inner=inner,
                   chamber=chamber, walls=walls, carve=carve, pump=pump,
                   probe=probe, plate=plate, beyond=beyond,
                   far_pump=far_pump, near_pump=near_pump,
                   far_probe=far_probe, near_probe=near_probe)


def airlock_mode(tm, al: Airlock) -> str:
    """``"hull"`` when the outer door opens onto SPACE (a tile beyond it is
    the vacuum code), else ``"interior"`` -- the airlock between two
    pressurised cells (#133, Erik's ruling 2026-10-31). Read off the tiles,
    so it is decided once, at generation time, from what stage 2 stamped.
    v1's graphs join rooms only, so every airlock they emit is interior;
    the hull branch waits for an edge that opens onto space."""
    return "hull" if any(int(tm[y, x]) == SPACE_CODE
                         for (y, x) in al.beyond) else "interior"


def _airlock_mounts(al: Airlock, mode: str) -> tuple:
    """``((mount (y, x), (acted-on (y, x), ...)), ...)`` for a mode's
    equipment: what stage 2 reserves and :func:`_carve_airlock` checks.
    Hull: the external pump and the chamber probe. Interior: the chamber
    probe, two transfer pumps (port + source) and two side probes."""
    def acts(m):
        x, y = m[0], m[1]
        out = [(y + m[3], x + m[2])]
        if len(m) == 6:
            out.append((y + m[5], x + m[4]))
        return tuple(out)
    eq = ((al.pump, al.probe) if mode == "hull" else
          (al.probe, al.far_pump, al.near_pump, al.far_probe, al.near_probe))
    return tuple(((m[1], m[0]), acts(m)) for m in eq)


def _airlocks(graph: ShipGraph) -> list:
    """``[(edge index, Airlock)]`` for every airlock edge, in edge order."""
    return [(i, _airlock(graph, e)) for i, e in enumerate(graph.edges)
            if e.spec.entity == "airlock"]


def _carved(graph: ShipGraph) -> dict:
    """Room id -> the set of its box tiles vestibules take."""
    out: dict = {}
    for i, al in _airlocks(graph):
        out.setdefault(graph.edges[i].a, set()).update(al.carve)
    return out


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
    tilemap: np.ndarray               # (h, w) int32 v3 codes
    spawn_rooms: list                 # list[SpawnRoom]: team 0, team 1, zombies
    entities: list                    # ducts + vents, doors, hazard charges, heat sources
    lights: list                      # one ceiling lamp per room (LightEntry)
    doors: int
    windows: int
    fuel_tiles: int
    # Per room id: its compartment index (ShipGraph.compartment_of) plus the
    # annotation tags it carries (dept, role, cell, secret, light) -- the
    # generator-side table the vent builder and the grammar read.
    room_tags: dict = field(default_factory=dict)
    edges_by_kind: dict = field(default_factory=dict)   # kind -> count
    wires: list = field(default_factory=list)   # (from, to): the airlocks'
    # temp_init.npy's grid (int32 Q16.16 game degrees above ambient) from the
    # ``temp`` tags, or None when no room is tagged (no file is written).
    temp_init: np.ndarray | None = None


def _reflect(box, w):
    x0, y0, x1, y1 = box
    return (w - 1 - x1, y0, w - 1 - x0, y1)


def _place_fuel(rng, tm, room: Room, graph: ShipGraph, door) -> None:
    """``room.tags['fuel']`` clusters against its walls, clear of the
    passages in ``door`` (the opening tiles of every ``keep_clear`` edge);
    in a mirrored graph placed once per pair (an axis room: its left half)
    and reflected into the twin."""
    W = graph.width
    x0, y0, x1, y1 = room.box
    if graph.mirror_x and room.mirror == room.id:
        x1 = min(x1, W // 2 - 1)
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


def _vent_site(tm, room: Room, axis2: int, reserved=None):
    """A steel tile on the room's boundary, the room's air in front, no
    door / glass within one tile, closest to the axis (then topmost):
    ``(x, y, facing)`` or None. ``axis2`` is twice the axis coordinate.
    ``reserved`` (bool mask, optional): tiles neither the site nor its
    aperture may take (entity door spans, airlock equipment and vestibules,
    vents already placed)."""
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
        if np.isin(tm[y - 1:y + 2, x - 1:x + 2],
                   (MAT_DOOR, MAT_GLASS, MAT_DOOR_CLOSED)).any():
            continue
        if reserved is not None and (reserved[y - 1:y + 2, x - 1:x + 2].any()
                                     or reserved[y + dy, x + dx]):
            continue
        key = (abs(2 * x - axis2), y, x)
        if best is None or key < best[0]:
            best = (key, (x, y, facing))
    return None if best is None else best[1]


_VENT_KEYS = ("x", "y", "mount", "facing", "role", "duct", "q_circ")
_HAZARD_KEYS = ("x", "y", "payload", "trigger", "trigger_temp", "first_at_s")
_HEAT_SOURCE_KEYS = ("x", "y", "heat_per_s", "radius", "max_temp",
                     "death_payload")


def _floor_site(tm, room: Room, axis2: int, passage, reserved):
    """THE floor-site rule for a room's point entity (a hazard charge, a
    heat source): air, no fuel in its 3x3, no passage within
    :data:`FUEL_DOOR_CLEAR` (Chebyshev), no reserved tile (entity doors,
    airlock equipment, vents, entities already placed) in its 3x3; closest to
    the room's centre (Manhattan), then topmost, then closest to the axis
    -- a key the mirror maps onto itself, so twins get twin sites.
    ``(x, y)`` or None."""
    x0, y0, x1, y1 = room.box
    c = FUEL_DOOR_CLEAR
    best = None
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if tm[y, x] != MAT_AIR:
                continue
            if np.isin(tm[y - 1:y + 2, x - 1:x + 2],
                       (MAT_FURNITURE, MAT_WOOD)).any():
                continue
            if passage[max(0, y - c):y + c + 1, max(0, x - c):x + c + 1].any():
                continue
            if reserved[y - 1:y + 2, x - 1:x + 2].any():
                continue
            key = (abs(2 * x - (x0 + x1)) + abs(2 * y - (y0 + y1)), y,
                   abs(2 * x - axis2))
            if best is None or key < best[0]:
                best = (key, (x, y))
    return None if best is None else best[1]


def _build_hazards(tm, graph: ShipGraph, passage, reserved) -> list:
    """One heat-triggered ``timed_charge`` per ``hazard``-tagged room, by
    room id: id ``hazard_<room>``, its tag's payload, ``trigger = "heat"``,
    ``trigger_temp`` and ``fuse_s`` as ``first_at_s`` (the fuse). The
    payload name is validated when the level loads (the charge's door)."""
    ents = []
    for rm in graph.rooms:
        if "hazard" not in rm.tags:
            continue
        hz = rm.tags["hazard"]
        site = _floor_site(tm, rm, graph.width - 1, passage, reserved)
        if site is None:
            raise _Reject(f"hazard: no clear floor tile in room {rm.id}")
        x, y = site
        ents.append(EntityInstance(
            id=f"hazard_{rm.id}", class_name="timed_charge",
            ordinal=len(ents),
            fields={"x": int(x), "y": int(y), "payload": hz["payload"],
                    "trigger": "heat",
                    "trigger_temp": float(hz["trigger_temp"]),
                    "first_at_s": float(hz["fuse_s"])},
            authored_keys=_HAZARD_KEYS))
    return ents


def _build_heat_sources(tm, graph: ShipGraph, passage, reserved) -> list:
    """One ``heat_source`` per ``heat_source``-tagged room, by room id: id
    ``heat_<room>``, the tag's fields verbatim, on :func:`_floor_site` with
    every already-placed entity tile (``reserved``, which the caller fills
    with the hazard charges' sites) kept out of its 3x3. The death payload
    is validated when the level loads (the source's door)."""
    ents = []
    for rm in graph.rooms:
        if "heat_source" not in rm.tags:
            continue
        hs = rm.tags["heat_source"]
        site = _floor_site(tm, rm, graph.width - 1, passage, reserved)
        if site is None:
            raise _Reject(f"heat_source: no clear floor tile in room {rm.id}")
        x, y = site
        ents.append(EntityInstance(
            id=f"heat_{rm.id}", class_name="heat_source", ordinal=len(ents),
            fields={"x": int(x), "y": int(y),
                    "heat_per_s": float(hs["heat_per_s"]),
                    "radius": int(hs["radius"]),
                    "max_temp": float(hs["max_temp"]),
                    "death_payload": hs["death_payload"]},
            authored_keys=_HEAT_SOURCE_KEYS))
    return ents


def _build_vents(tm, graph: ShipGraph, reserved=None) -> list:
    """One duct entity per duct name the rooms' ``vent`` tags use (id
    ``duct_<name>``, its filter :meth:`ShipGraph.duct_filter`), then that
    duct's wall vents, one per (room, role): supply rooms first, then by
    room id. Ducts in name order. A vent's id is ``vent_<duct>_<room>_<role>``;
    v1's string shorthand keeps v1's ``vent_<role>`` (so v1 ships write the
    same file), and a collision of those is refused by name. A room with
    both roles gets two sites, the second clear of the first."""
    tagged = {}
    for rm in graph.rooms:
        if "vent" in rm.tags:
            duct, roles, shorthand = vent_spec(rm.tags["vent"])
            for role in roles:
                tagged.setdefault(duct, []).append((role, rm, shorthand))
    if not tagged:
        return []
    taken = (np.zeros(tm.shape, dtype=bool) if reserved is None
             else reserved.copy())
    ents, ids = [], set()
    for duct in sorted(tagged):
        did = f"duct_{duct}"
        ents.append(EntityInstance(
            id=did, class_name="duct", ordinal=len(ents),
            fields={"filter": graph.duct_filter(duct)},
            authored_keys=("filter",)))
        for role, rm, shorthand in sorted(
                tagged[duct], key=lambda t: (t[0] != "supply", t[1].id)):
            site = _vent_site(tm, rm, graph.width - 1, taken)
            if site is None:
                raise _Reject(f"vents: no wall site in room {rm.id}")
            x, y, facing = site
            taken[y - 1:y + 2, x - 1:x + 2] = True
            vid = (f"vent_{role}" if shorthand
                   else f"vent_{duct}_{rm.id}_{role}")
            if vid in ids:
                raise GraphSchemaError(
                    f"room {rm.id}: vent id {vid!r} is taken -- v1's string "
                    f"shorthand allows one supply and one return; use "
                    f'{{"duct": ..., "roles": [...]}}')
            ids.add(vid)
            ents.append(EntityInstance(
                id=vid, class_name="vent", ordinal=len(ents),
                fields={"x": int(x), "y": int(y), "mount": "wall",
                        "facing": facing, "role": role, "duct": did,
                        "q_circ": Q_CIRC},
                authored_keys=_VENT_KEYS))
    return ents


def _door_length_m(n: int) -> float:
    """A ``length_m`` the canonical door quantization (A6 §3) maps to
    exactly ``n`` tiles at ``TILE_SIZE_M``."""
    for digits in (3, 4, 6):
        m = round(n / tiles_per_m(TILE_SIZE_M), digits)
        if quantize_span_tiles(m, TILE_SIZE_M) == n:
            return m
    raise ValueError(f"no length_m quantizes to {n} tiles at {TILE_SIZE_M}")


_DOOR_KEYS = ("x", "y", "orientation", "length_m", "initial_state")


def _door_entity(eid: str, tiles, orient: str, state: str) -> EntityInstance:
    """An A6 ``door`` entity spanning ``tiles`` (row-major, in a wall of
    orientation ``orient``: a ``"v"`` wall's door runs along +y)."""
    y, x = tiles[0]
    return EntityInstance(
        id=eid, class_name="door", ordinal=0,
        fields={"x": int(x), "y": int(y), "orientation": orient,
                "length_m": _door_length_m(len(tiles)),
                "initial_state": state},
        authored_keys=_DOOR_KEYS)


def _build_doors(graph: ShipGraph, modes: dict | None = None
                 ) -> tuple[list, list]:
    """The entity half of the edges, ``(entities, wires)``: one ``door``
    entity per ``entity_door`` edge (id ``door_e<edge index>``, initial
    state ``ENTITY_DOOR_INITIAL``), and per ``airlock`` edge its mode's
    recipe (:func:`airlock_entities`; ``modes``: edge index -> mode, from
    :func:`airlock_mode`, absent = hull) under ids
    ``airlock_e<edge index>_<part>``. Ordinals are assigned by the caller
    (file order)."""
    ents, wires = [], []
    for i, e in enumerate(graph.edges):
        if e.spec.entity != "door":
            continue
        orient = shared_wall(graph.rooms[e.a], graph.rooms[e.b])[0]
        ents.append(_door_entity(f"door_e{i}", _opening_tiles(graph, e),
                                 orient, ENTITY_DOOR_INITIAL))
    for i, al in _airlocks(graph):
        a_ents, a_wires = airlock_entities(
            f"airlock_e{i}", al, (modes or {}).get(i, "hull"))
        ents += a_ents
        wires += a_wires
    return ents, wires


def airlock_entities(p: str, al: Airlock, mode: str) -> tuple[list, list]:
    """One airlock's ``(entities, wires)`` under ids ``<p>_<part>``.

    ``"hull"``: the levels/airlock_demo recipe -- inner door (open), outer
    door (closed), an external pump, presence plate, chamber probe, two
    deciders, the controller evacuating toward the far side and refilling
    (15 wires). ``"interior"`` (#133): the same doors, plate and chamber
    probe, plus a transfer pump and a pressure probe per side and the
    controller in ``mode = "interior"`` -- no deciders, no external pump,
    no gas leaves the ship (16 wires)."""
    if mode not in AIRLOCK_MODES:
        raise ValueError(f"airlock {p}: mode {mode!r} is not one of "
                         f"{AIRLOCK_MODES}")
    parts = (("inner", "outer", "pump", "plate", "probe", "dec_far",
              "dec_near", "ctrl") if mode == "hull" else
             ("inner", "outer", "pump_far", "pump_near", "plate", "probe",
              "probe_far", "probe_near", "ctrl"))
    ids = {k: f"{p}_{k}" for k in parts}
    ents = [_door_entity(ids["inner"], al.inner, al.wall, "open"),
            _door_entity(ids["outer"], al.outer, al.wall, "closed")]

    def pump(eid, m, extra):
        f = {"x": m[0], "y": m[1], "port_dx": m[2], "port_dy": m[3], **extra}
        if len(m) == 6:
            f.update(source_dx=m[4], source_dy=m[5])
        ents.append(EntityInstance(id=eid, class_name="pump", ordinal=0,
                                   fields=f, authored_keys=tuple(f)))

    def probe(eid, m):
        ents.append(EntityInstance(
            id=eid, class_name="pressure", ordinal=0,
            fields={"x": m[0], "y": m[1], "sample_dx": m[2],
                    "sample_dy": m[3]},
            authored_keys=("x", "y", "sample_dx", "sample_dy")))

    if mode == "hull":
        pump(ids["pump"], al.pump, AIRLOCK_PUMP)
    else:
        pump(ids["pump_far"], al.far_pump, AIRLOCK_TRANSFER_PUMP)
        pump(ids["pump_near"], al.near_pump, AIRLOCK_TRANSFER_PUMP)
    ents.append(EntityInstance(
        id=ids["plate"], class_name="sensor_motion", ordinal=0,
        fields={"x": al.plate[0], "y": al.plate[1],
                "radius": _door_length_m(AIRLOCK_PLATE_TILES)},
        authored_keys=("x", "y", "radius")))
    probe(ids["probe"], al.probe)
    if mode == "hull":
        for k, (cmp_, thr) in (("dec_far", AIRLOCK_DEC_FAR),
                               ("dec_near", AIRLOCK_DEC_NEAR)):
            ents.append(EntityInstance(
                id=ids[k], class_name="decider", ordinal=0,
                fields={"comparator": cmp_, "threshold": thr},
                authored_keys=("comparator", "threshold")))
        ctrl = AIRLOCK_CTRL
    else:
        probe(ids["probe_far"], al.far_probe)
        probe(ids["probe_near"], al.near_probe)
        ctrl = AIRLOCK_CTRL_INTERIOR
    ents.append(EntityInstance(
        id=ids["ctrl"], class_name="airlock_controller", ordinal=0,
        fields=dict(ctrl), authored_keys=tuple(ctrl)))
    c = ids["ctrl"]
    wires = [
        (f"{ids['plate']}.value", f"{c}.presence"),
        (f"{ids['inner']}.is_open", f"{c}.inner_open"),
        (f"{ids['outer']}.is_open", f"{c}.outer_open"),
        (f"{ids['inner']}.alive", f"{c}.inner_alive"),
        (f"{ids['outer']}.alive", f"{c}.outer_alive"),
    ]
    if mode == "hull":
        wires += [
            (f"{ids['probe']}.value", f"{ids['dec_far']}.in"),
            (f"{ids['probe']}.value", f"{ids['dec_near']}.in"),
            (f"{ids['dec_far']}.out", f"{c}.at_far"),
            (f"{ids['dec_near']}.out", f"{c}.at_near"),
        ]
    else:
        # far_door = "outer": the outer door's side is the far side.
        wires += [
            (f"{ids['probe']}.value", f"{c}.chamber_p"),
            (f"{ids['probe_near']}.value", f"{c}.inner_side_p"),
            (f"{ids['probe_far']}.value", f"{c}.outer_side_p"),
        ]
    wires += [
        (f"{c}.inner_close", f"{ids['inner']}.close"),
        (f"{c}.inner_open_cmd", f"{ids['inner']}.open"),
        (f"{c}.outer_close", f"{ids['outer']}.close"),
        (f"{c}.outer_open_cmd", f"{ids['outer']}.open"),
    ]
    if mode == "hull":
        wires += [(f"{c}.pump_inject", f"{ids['pump']}.inject"),
                  (f"{c}.pump_extract", f"{ids['pump']}.extract")]
    else:
        wires += [(f"{c}.pump_inject", f"{ids['pump_far']}.inject"),
                  (f"{c}.pump_extract", f"{ids['pump_far']}.extract"),
                  (f"{c}.near_pump_inject", f"{ids['pump_near']}.inject"),
                  (f"{c}.near_pump_extract", f"{ids['pump_near']}.extract")]
    return ents, wires


def embed(graph: ShipGraph, rng) -> Embedding:
    """Stage 2: the graph as tiles (+ spawn rooms and vent entities). Each
    edge stamps what its :data:`EDGE_KINDS` row says; a kind the table marks
    ``pending`` raises NotImplementedError naming the patch that will."""
    graph.check()
    for e in graph.edges:
        if e.spec.pending:
            raise NotImplementedError(
                f"edge {e.a}-{e.b}: kind {e.kind!r} is accepted by the schema "
                f"but not yet stamped by stage 2 -- {e.spec.pending}")
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
    passage = np.zeros((H, W), dtype=bool)        # fuel keeps clear of these
    reserved = np.zeros((H, W), dtype=bool)       # vents keep clear of these
    by_kind = {}
    for e in graph.edges:
        spec = e.spec
        for (y, x) in _opening_tiles(graph, e):
            tm[y, x] = spec.stamps
            passage[y, x] |= spec.keep_clear
            reserved[y, x] |= spec.entity is not None
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    modes = {}
    for i, al in _airlocks(graph):
        modes[i] = airlock_mode(tm, al)       # read BEFORE the carve
        _carve_airlock(tm, al, modes[i])
        for (y, x) in al.carve:
            passage[y, x] = reserved[y, x] = True
        for (my, mx), _acted in _airlock_mounts(al, modes[i]):
            reserved[my, mx] = True
    doors, windows = by_kind.get("door", 0), by_kind.get("window", 0)
    for rm in graph.rooms:
        if rm.tags.get("fuel") and (rm.mirror is None or rm.mirror >= rm.id
                                    or not graph.mirror_x):
            _place_fuel(rng, tm, rm, graph, passage)
    ents = _build_vents(tm, graph, reserved if reserved.any() else None)
    door_ents, wires = _build_doors(graph, modes)
    vent_res = np.zeros((H, W), dtype=bool)
    for ent in ents:
        if ent.class_name == "vent":
            vent_res[ent.fields["y"], ent.fields["x"]] = True
    hazards = _build_hazards(tm, graph, passage, reserved | vent_res)
    for ent in hazards:
        vent_res[ent.fields["y"], ent.fields["x"]] = True
    ents = (ents + door_ents + hazards
            + _build_heat_sources(tm, graph, passage, reserved | vent_res))
    for k, ent in enumerate(ents):
        ent.ordinal = k                            # file order (A3 §3a)
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
    lights = []
    for rm in graph.rooms:
        rgb, level = LAMP_RGB, LAMP_INTENSITY
        if "light" in rm.tags:
            rgb = tuple(float(c) for c in rm.tags["light"][0])
            level = float(rm.tags["light"][1])
        lights.append(LightEntry(x=rm.x + rm.w / 2.0, y=rm.y + rm.h / 2.0,
                                 color=rgb, intensity=level, range=16.0,
                                 kind="static"))
    comp = graph.compartment_of()
    room_tags = {rm.id: {"compartment": comp[rm.id],
                         **{k: rm.tags[k] for k in ANNOTATION_TAGS
                            if k in rm.tags}}
                 for rm in graph.rooms}
    return Embedding(tilemap=tm, spawn_rooms=rooms, entities=ents,
                     lights=lights, doors=doors, windows=windows,
                     fuel_tiles=fuel, room_tags=room_tags,
                     edges_by_kind=by_kind, wires=wires,
                     temp_init=_build_temp_init(tm, graph))


def _build_temp_init(tm, graph: ShipGraph):
    """The ``temp`` tags as ``temp_init.npy``'s grid: every tagged room's
    FLOOR AIR (its rectangle's ``MAT_AIR`` tiles, minus any airlock vestibule
    carved into it -- the vestibule is its own compartment) at the tag's
    value, quantized once through the Q16 kit; 0 everywhere else. Gas only:
    a hot WALL is a different thing (a thermal solid holding heat, not a hot
    room), and fuel clusters stay ambient so a hot room does not start its
    own kindling burning. None when no room is tagged -- no file at all."""
    tagged = [rm for rm in graph.rooms if "temp" in rm.tags]
    if not tagged:
        return None
    vest = np.zeros(tm.shape, dtype=bool)
    for _i, al in _airlocks(graph):
        for (y, x) in al.carve:
            vest[y, x] = True
    out = np.zeros(tm.shape, dtype=np.int32)
    for rm in tagged:
        x0, y0, x1, y1 = rm.box
        sub = (tm[y0:y1 + 1, x0:x1 + 1] == MAT_AIR) & ~vest[y0:y1 + 1,
                                                             x0:x1 + 1]
        out[y0:y1 + 1, x0:x1 + 1][sub] = quantize_q16_scalar(
            float(rm.tags["temp"]))
    return out


def _carve_airlock(tm, al: Airlock, mode: str = "hull") -> None:
    """Stamp a vestibule into room ``a``'s air: its new walls steel, the
    inner door's span air (the door spawns open), and refuse a vestibule
    whose enclosure another edge has cut, or whose equipment has no solid
    mount."""
    for (y, x) in al.walls:
        tm[y, x] = MAT_STEEL
    for (y, x) in al.inner:
        tm[y, x] = _DOOR_STAMP["open"]
    x0, y0, x1, y1 = al.chamber
    doors = set(al.inner) | set(al.outer)
    ring = [(y, x) for y in range(y0 - 1, y1 + 2) for x in range(x0 - 1, x1 + 2)
            if not (y0 <= y <= y1 and x0 <= x <= x1)
            and (y0 <= y <= y1 or x0 <= x <= x1)]       # 4-neighbours only
    for (y, x) in ring:
        if (y, x) not in doors and tm[y, x] not in (MAT_STEEL, MAT_HULL):
            raise _Reject(f"airlock: the vestibule's wall at ({x},{y}) is "
                          f"cut ({int(tm[y, x])})")
    for (my, mx), acted in _airlock_mounts(al, mode):
        if tm[my, mx] not in (MAT_STEEL, MAT_HULL):
            raise _Reject(f"airlock: no solid mount at ({mx},{my})")
        for (y, x) in acted:                # a port / source / sample tile
            if tm[y, x] != MAT_AIR:
                raise _Reject(f"airlock: the {mode} equipment at "
                              f"({mx},{my}) acts on a non-air tile "
                              f"({x},{y})")


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
    mat, vac = materials_from_tilemap(tm, LEVEL_FORMAT_VERSION)
    tbl = _table()
    solid = np.asarray(tbl.permeability)[mat] <= 0.0
    walk = np.asarray(tbl.mobility)[mat] > 0
    return mat, np.asarray(vac, dtype=bool), solid, walk


def compartments(tm: np.ndarray, graph: ShipGraph | None = None):
    """``(labels, n)`` of the ship's compartments: the regions bounded by
    the structure (hull, steel, doors, glass) -- fuel is a room's contents,
    so a pocket of air behind a crate belongs to its room. With ``graph``
    the STRUCTURE MASK also takes the graph's edge kinds: the opening tiles
    of every edge that bounds a compartment are structure whatever they are
    made of (a ``breakable`` edge's wood segment is a wall, not a crate),
    while an ``open`` edge's air gap joins its rooms as the table says. An
    entity door's span is structure even while it stands open (it can
    close), and so is an airlock's inner door: each vestibule is its own
    compartment."""
    from scipy import ndimage
    contents = np.isin(tm, (MAT_AIR, MAT_FURNITURE, MAT_WOOD))
    if graph is not None:
        for e in graph.edges:
            if e.spec.bounds and e.spec.pending is None:
                for (y, x) in _opening_tiles(graph, e):
                    contents[y, x] = False
        for _i, al in _airlocks(graph):
            for (y, x) in al.inner:
                contents[y, x] = False
    return ndimage.label(contents)


def _room_label(clab: np.ndarray, rm: Room, carved=None) -> int:
    """The compartment label of a room: its rectangle's top-left tile (air
    or a crate, always contents) -- or, when a vestibule was carved out of
    it (``carved``: room id -> tiles), its first row-major tile outside the
    carve."""
    skip = (carved or {}).get(rm.id)
    if not skip:
        return int(clab[rm.y, rm.x])
    x0, y0, x1, y1 = rm.box
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (y, x) not in skip and clab[y, x] > 0:
                return int(clab[y, x])
    return 0


def _graph_compartment_reasons(clab, cn, graph: ShipGraph) -> list:
    """The tiles realise the graph's compartments exactly: one tile
    compartment per graph compartment, rooms an ``open`` edge joins share
    it, rooms an edge bounds do not; plus one compartment per airlock
    vestibule, shared with nothing."""
    comp = graph.compartment_of()
    airlocks = _airlocks(graph)
    carved = _carved(graph)
    k = (max(comp) + 1 if comp else 0) + len(airlocks)
    reasons = []
    if cn != k:
        reasons.append(f"rooms: {cn} compartments for {k} graph compartments "
                       f"({len(graph.rooms)} rooms, {len(airlocks)} "
                       f"vestibules)")
    labels = {}                                   # graph compartment -> labels
    for rm in graph.rooms:
        labels.setdefault(comp[rm.id], set()).add(
            _room_label(clab, rm, carved))
    for i, al in airlocks:
        x0, y0, _x1, _y1 = al.chamber
        labels[("vestibule", i)] = {int(clab[y0, x0])}
    for c, ls in sorted(labels.items(), key=_comp_key):
        if len(ls) > 1:
            ids = [rm.id for rm in graph.rooms if comp[rm.id] == c]
            reasons.append(f"rooms: rooms {ids} are one graph compartment but "
                           f"{len(ls)} in the tiles")
    owner = {}
    for c, ls in sorted(labels.items(), key=_comp_key):
        for lab in ls:
            if lab in owner and owner[lab] != c:
                reasons.append(f"rooms: graph compartments {owner[lab]} and "
                               f"{c} share one compartment in the tiles")
            owner.setdefault(lab, c)
    return reasons


def _comp_key(kv):
    """Graph compartments (ints) first, in order, then the vestibules."""
    return (isinstance(kv[0], tuple), kv[0])


def _excused_secrets(clab, covered: set, graph: ShipGraph) -> set:
    """The tile labels of compartments a unit does not reach but which are
    SECRET: every room in them tagged ``secret``, and joined to a reached
    (or already excused) compartment by an edge whose kind is a
    ``secret_path`` (breakable, vent_only) -- reachable in principle, by
    cutting the wall or opening the duct."""
    comp = graph.compartment_of()
    carved = _carved(graph)
    lab = {}
    secret = {}
    for rm in graph.rooms:
        lab[comp[rm.id]] = _room_label(clab, rm, carved)
        secret[comp[rm.id]] = secret.get(comp[rm.id], True) and \
            bool(rm.tags.get("secret"))
    reached = {c for c, l in lab.items() if l in covered}
    excused = set()
    grew = True
    while grew:
        grew = False
        for e in graph.edges:
            if not e.spec.secret_path:
                continue
            ca, cb = comp[e.a], comp[e.b]
            for src, dst in ((ca, cb), (cb, ca)):
                if src in reached and dst not in reached and secret[dst]:
                    reached.add(dst)
                    excused.add(lab[dst])
                    grew = True
    return excused


def _centre(room: SpawnRoom) -> tuple[int, int]:
    """The footprint top-left nearest the room's centre."""
    fp = FORMATION_FOOTPRINT
    return room.x + (room.w - fp) // 2, room.y + (room.h - fp) // 2


def validate(tm: np.ndarray, rooms: list, *, n_rooms: int | None = None,
             mirrored: bool = True, graph: ShipGraph | None = None) -> list:
    """The gates; returns the failure reasons (empty = pass). ``rooms``:
    the spawn rooms (team 0, team 1[, zombies]). With ``graph`` (what
    :func:`generate` passes): the compartments are labelled with the
    graph's structure mask and must realise the graph's compartments
    exactly, and a ``secret`` compartment reachable only through a
    breakable / vent_only edge passes the reachability gate. Without it
    (``n_rooms``: a bare node count to match): the tile structure alone,
    every compartment reachable."""
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
    # 3x3 footprint passability (GameMap.is_passable_block's rule). An
    # airlock's outer door spawns closed (mobility 0) but its controller
    # opens it for ANY unit standing in the vestibule (the presence plate
    # counts every team), so the gate walks through it; an entity door
    # stands open in the tiles, exactly as the sim stamps it.
    if graph is not None:
        walk = walk.copy()
        for _i, al in _airlocks(graph):
            for (y, x) in al.outer:
                walk[y, x] = True
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
    clab, cn = compartments(tm, graph)
    if graph is not None:
        reasons += _graph_compartment_reasons(clab, cn, graph)
    elif n_rooms is not None and cn != n_rooms:
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
        covered = {int(v) for v in np.unique(clab[cover & (clab > 0)])}
        excused = (_excused_secrets(clab, covered, graph)
                   if graph is not None else set())
        missing = cn - len(covered) - len(excused - covered)
        if missing:
            reasons.append(f"compartments: {missing} not reachable by a unit"
                           + (" (a secret room needs a breakable or vent_only "
                              "edge to a reached one)" if graph is not None
                              and any(rm.tags.get("secret")
                                      for rm in graph.rooms) else ""))
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
                           mirrored=graph.mirror_x, graph=graph)
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
    MAT_DOOR_CLOSED: (170, 110, 50),
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
# Written only when the plan needs them (``temp_init.npy``: a ``temp`` tag);
# cleared on every regenerate like DATA_FILES, so a stale one never survives.
OPTIONAL_DATA_FILES = (level_lib.TEMP_INIT_FILENAME,)


def write(plan: ShipPlan, out_dir) -> Path:
    """Write ``plan`` as a level folder through ``level_lib``; returns the
    folder. Regenerates the data files (anything else there stays)."""
    out_dir = Path(out_dir)
    for name in DATA_FILES + OPTIONAL_DATA_FILES:
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
        f"--zombies {plan.zombies}" + (" --vents" if any(
            e.class_name == "duct" for e in plan.entities) else ""),
        "",
        f"v{LEVEL_FORMAT_VERSION} codes ARE canon material ids "
        "(src/simulation/materials.py):",
        "  0=air 1=hull 2=wood 3=door 4=steel 5=glass 6=furniture "
        f"{SPACE_CODE}=SPACE",
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
    if plan.embedding.wires:
        wires = plan.embedding.wires
        blocks["wire"] = lambda nl: level_lib.format_wire_lines(wires, nl)
    level_lib.write_managed_blocks(toml_path, blocks, write_bak=False)
    level_lib.write_match(out_dir, plan.match_spec())
    if plan.embedding.temp_init is not None:
        level_lib.write_temp_init_npy(out_dir, plan.embedding.temp_init,
                                      npy_bak=False)
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
    vents = any(x.class_name == "duct" for x in plan.entities)
    print(f"wrote {out}  ({w}x{h} @ {TILE_SIZE_M} m, attempt {plan.attempt}, "
          f"{len(plan.graph.rooms)} rooms, {e.doors} doors, {e.windows} "
          f"windows, {e.fuel_tiles} fuel tiles, "
          f"vents {'yes' if vents else 'no'})")
    for r in plan.spawn_rooms:
        print(f"  spawn room team {r.team}: ({r.x},{r.y}) {r.w}x{r.h} "
              f"max {r.max_per_team}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
