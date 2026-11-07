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
fuel clusters (furniture / wood, or the tag's material mix) against the
walls of tagged rooms clear of
doors (placed in one room of a mirror pair and reflected into its twin),
the ducts (one per duct name the ``vent`` tags use, a filter each) and their
vents on inner walls of their tagged rooms, one heat-triggered
``timed_charge`` per ``hazard``-tagged room (the magazine's cook-off, the
hazmat store; vocabulary E.2 row 1), one ``heat_source`` per
``heat_source``-tagged room (the engine, the galley stove; E.2 row 3), both
on the one floor-site rule, the ``temp``-tagged rooms' floor air seeded hot
in ``temp_init.npy`` (hot engineering; E.2 row 4), a ``hull_door``-tagged
room's A6 door entity cut into its hull wall, CLOSED (the cargo hold's valve
to space, B.4; E.2 row 5), one static lamp per room (or one per
``lamp_span`` of its long side)
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
    MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED, MAT_FURNITURE, MAT_GLASS, MAT_GRILLE,
    MAT_HULL, MAT_KINDLING, MAT_STEEL, MAT_WOOD,
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
LAMP_RANGE = 16.0            # tiles
# Lamps per length (layout doc §11): a room tagged ``lamp_span = s`` gets
# ceil(long side / s) lamps along its long axis, the long side cut into that
# many EQUAL spans, one lamp centred on each (so a room's lamps are symmetric
# about its centre and a mirror twin's are the reflection). LAMP_SPAN is the
# value the layout stage will set; untagged rooms keep the one centred lamp.
LAMP_SPAN = 12
LAMP_SPAN_MIN = 6
# THE LAMP LIGHT BUDGET (#136, found by the first large recipe ship, which
# main.py refused at frame 3): a frame's cone emitters may put at most
# LIGHT_CONE_BUDGET = 2^44 counts per channel = 128 light units at L°'s
# currency into ONE ordinate (the sweep's door, #173), and an omni lamp or
# flash puts 1/16 of itself in each. In one ordinate the muzzle flashes take
# MUZZLE_FLASH_MAX_TOTAL / 16 = 64, and LAMP_RESERVE is held for the beams
# (flashlights, jets: all of a beam can land in one). So the lamps of a ship whose rooms carry
# ``lamp_span`` are held at <= LAMP_LIGHT_BUDGET light units per channel
# (:func:`_ship_lamps`): first fewer lamps per room (a wider span, ending at
# one lamp per room), then -- only if one lamp per room is still too much --
# every lamp dimmed by one common factor. tests/test_levelspec.py holds this
# number against frame_lights' constants (gen_ship stays pyray-free).
CONE_LIGHT_UNITS = 128.0      # == frame_lights.LIGHT_CONE_BUDGET / 2^L_FINE_BITS
OMNI_SHARE = 16               # the sweep's ordinates: an omni lamp's 1/n
MUZZLE_FLASH_LIGHT = 1024.0   # == frame_lights.MUZZLE_FLASH_MAX_TOTAL (all omni)
LAMP_RESERVE = 16.0           # one ordinate: the beams (a squad's flashlights) + slack
LAMP_LIGHT_BUDGET = (CONE_LIGHT_UNITS - MUZZLE_FLASH_LIGHT / OMNI_SHARE
                     - LAMP_RESERVE) * OMNI_SHARE
# The int form of the ``fuel`` tag: v1's furniture / wood 60/40 draw, as a
# mix (:func:`fuel_spec`; one ``rng.random()`` per cluster, the walk in name
# order, so the int form keeps v1's exact draws).
FUEL_MIX_V1 = {"furniture": 0.6, "wood": 0.4}

# Entity doors (A6). The bots (control_bots) never USE a door and a closed
# entity door stamps MAT_DOOR_CLOSED (mobility 0), which no unit crosses --
# so stage 2 emits every entity door OPEN (MAT_AIR in the CSV, the loader's
# own stamp) until a door-opening rule exists (#142). Still a compartment
# boundary in the gates: it can close.
ENTITY_DOOR_INITIAL = "open"

# The CARGO DOOR IN THE HULL (vocabulary B.4 / E.2 row 5): a room's
# ``hull_door`` tag is an A6 ``door`` entity cut into the room's HULL wall
# (the wall whose far side is space), centred on its longest hull run. It is
# emitted CLOSED by default -- the opposite of ENTITY_DOOR_INITIAL, on
# purpose: an interior entity door is a PATH (and no bot opens doors, so it
# must stand open to be one), while a door to space is a VALVE the level or
# a wire opens (the hold's depressurisation valve, D7) -- closed it is a
# wall, open it is a breach. ``state = "open"`` is REFUSED in v1: a hull door
# standing open is a room open to space, the future breach archetype, which
# no gate accepts yet (the airtight gate would reject every such plan).
HULL_DOOR_W = 3                      # a 3x3 unit fits through
HULL_DOOR_STATES = ("closed", "open")
HULL_DOOR_DEFAULT = {"width": HULL_DOOR_W, "state": "closed"}

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
    kind, :func:`embed` refuses it naming that); ``jambs``: the material
    stamped in the ONE wall tile on each side of the opening (None: they stay
    steel; ``vented_door``'s grilles, #134), always inside the shared run;
    ``air``: the opening passes gas from tick 0 and nothing in play closes
    it (``open``, ``grille``, ``vented_door``), so its two rooms are ONE AIR
    VOLUME (:func:`air_volumes`, vocabulary §0.6) -- an entity door can be
    shut, a window or a breakable wall must break first, a door is
    gas-tight.

    WALK, AIR and STRUCTURE are three questions (vocabulary §0.4, the dated
    note of 2026-11-06). A grille tile (``grille``'s opening, a
    ``vented_door``'s jambs) is STRUCTURE for walking and for the
    compartment count :func:`validate` gates, exactly like the steel it
    replaces -- :func:`compartments` never sees it as a gap, and a
    ``vented_door`` bounds a compartment exactly as ``door`` does -- while to
    the AIR it is open (permeability 1.0): the two rooms are one volume."""
    stamps: int | None
    width: int
    bounds: bool
    secret_path: bool
    keep_clear: bool
    l9: str
    pending: str | None = None
    entity: str | None = None
    jambs: int | None = None
    air: bool = False


_DOOR_STAMP = {"open": MAT_AIR, "closed": MAT_DOOR_CLOSED}

# THE table: a new way two rooms join is a new row here, never an inline
# stamp. Same rows, same order as the vocabulary's §0.4.
# A door to SPACE is not a row (it joins a room to the void, not two
# rooms): it is the ``hull_door`` room tag, emitted CLOSED where an
# entity_door is emitted open -- see HULL_DOOR_DEFAULT for why.
EDGE_KINDS = {
    "door":        EdgeKind(MAT_DOOR, DOOR_W, True, False, True,
                            L9_MUST_SHARE_WALL),
    "entity_door": EdgeKind(_DOOR_STAMP[ENTITY_DOOR_INITIAL], DOOR_W, True,
                            False, True, L9_MUST_SHARE_WALL, entity="door"),
    "open":        EdgeKind(MAT_AIR, DOOR_W, False, False, True,
                            L9_MUST_SHARE_WALL, air=True),
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
    # #134 (2026-11-06, vocabulary §0.4 addition): the GRILLE kinds -- air
    # passes, walking does not. `grille` is the window's twin (its width, a
    # steel grating instead of glass): an air-only opening, nobody walks
    # through, not a secret path. `vented_door` is the legacy door with a
    # grille in each jamb: the door's opening and width (so no layout
    # arithmetic moves), door tiles in the opening, the one wall tile on
    # each side grille. Both bound a compartment (a grille is structure for
    # walking and for the count); neither is a CELL_CROSSING_KINDS row.
    "grille":      EdgeKind(MAT_GRILLE, GLASS_W, True, False, False,
                            L9_MUST_SHARE_WALL, air=True),
    "vented_door": EdgeKind(MAT_DOOR, DOOR_W, True, False, True,
                            L9_MUST_SHARE_WALL, jambs=MAT_GRILLE, air=True),
}
# Kinds that may cross a cell boundary (L5 as amended, Erik 2026-10-31: the
# breakable wall is the third, a deliberate weak point): everything else is
# refused -- the grille kinds included (an air opening across a pressure
# cell would make the two cells one).
CELL_CROSSING_KINDS = ("airlock", "bulkhead", "breakable")
# The ONE multi-edge a room pair may carry: the AIRLOCK BYPASS (cycle-types
# doc §1.3, layout doc §4.3) -- an airlock and a breakable on the same shared
# wall, each with an explicit ``at``, the outer door's two wall mounts and a
# clearance tile each side of the breakable kept steel (:meth:`ShipGraph.
# _check_bypass`). Any other pair carrying two edges is refused.
BYPASS_KINDS = ("airlock", "breakable")


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
    "roles": [...]}`` with 1-2 distinct roles and a plain id-safe name, or a
    non-empty LIST of such dicts on distinct ducts (a room on several ducts:
    its department's and a duct link's, cycle-types doc §1.3)."""
    if isinstance(v, (list, tuple)):
        return (len(v) >= 1 and all(_check_vent_dict(d) for d in v)
                and len({d["duct"] for d in v}) == len(v))
    if isinstance(v, str):
        return v in VENT_ROLES
    return _check_vent_dict(v)


def _check_vent_dict(v) -> bool:
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
    """A single-duct ``vent`` tag as ``(duct, roles, shorthand)``: roles in
    :data:`VENT_ROLES` order; ``shorthand`` True for v1's string form (on
    the ``main`` duct, keeping v1's vent ids). A LIST tag has one such
    triple per duct: :func:`vent_specs`."""
    if isinstance(tag, (list, tuple)):
        raise GraphSchemaError(
            f"vent tag {tag!r} is a list (a room on several ducts): read it "
            f"with vent_specs()")
    if isinstance(tag, str):
        return MAIN_DUCT, (tag,), True
    return tag["duct"], tuple(r for r in VENT_ROLES if r in tag["roles"]), \
        False


def vent_specs(tag) -> tuple:
    """Every form of a ``vent`` tag as a tuple of :func:`vent_spec` triples,
    one per duct the room is on (one for the shorthand and the single
    dict), in the tag's order."""
    if isinstance(tag, (list, tuple)):
        return tuple(vent_spec(d) for d in tag)
    return (vent_spec(tag),)


def _id_key(rid):
    """Room ids sort numerically (a ShipGraph's ints) or by name (a recipe's
    and the grammar's strings)."""
    return (isinstance(rid, str), rid)


def air_volumes(rooms, edges) -> dict:
    """Room id -> AIR VOLUME index (vocabulary §0.6 item 1): rooms joined by
    an edge whose :data:`EDGE_KINDS` row says ``air`` (``open``, ``grille``,
    ``vented_door``) are one body of air. ``rooms``: every room id, in
    order; ``edges``: ``(a, b, kind)``. Indices run 0..k-1 in the order of
    each volume's first room. An edge added later can only MERGE volumes."""
    rooms = list(rooms)
    parent = {r: r for r in rooms}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r
    for a, b, kind in edges:
        if EDGE_KINDS[kind].air and a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    idx, out = {}, {}
    for r in rooms:
        out[r] = idx.setdefault(find(r), len(idx))
    return out


@dataclass(frozen=True)
class VentAirViolation:
    """One breach of :func:`vent_air_rule`: on ``duct``, the air volume of
    ``rooms`` has ``have`` vents (in ``have_rooms``) and no ``missing`` one.
    ``str()`` is the refusal message every call site raises."""
    duct: str
    rooms: tuple
    have: str
    have_rooms: tuple
    missing: str

    def __str__(self) -> str:
        return (f"duct {self.duct!r}: the air volume of rooms "
                f"{list(self.rooms)} has {self.have!r} vents (rooms "
                f"{list(self.have_rooms)}) and no {self.missing!r} vent on "
                f"that duct -- every air volume a duct has a vent in carries "
                f"a supply AND a return on it (no pressure feedback: a lone "
                f"supply pumps the volume up, a lone return drains it; "
                f"vocabulary §0.6)")


def vent_air_rule(vents: dict, edges) -> list:
    """THE AIR-VOLUME VENT RULE (vocabulary §0.6 item 2; it replaced the
    sealed-room rule of §0.4): for every duct, every air volume the duct has
    a vent in carries at least one SUPPLY and at least one RETURN on that
    duct -- the vents have no pressure feedback, so a volume a duct only
    feeds is pumped up and one it only drains is emptied. A sealed room is
    the one-room volume, so the old rule is the special case. ONE function:
    :meth:`ShipGraph.check`, the recipe door (``levelgen_recipes``) and the
    grammar's ``air_volume`` invariant (``ship_grammar``) all call it.

    ``vents``: EVERY room id (in order) -> its :func:`vent_specs` triples
    (``()`` for a room without a vent); ``edges``: ``(a, b, kind)``. v1's
    string shorthand is EXEMPT (v1's ``--vents`` pair is a lone supply and a
    lone return in two rooms; levels/ship_demo is pinned byte-identical,
    #151). Returns one :class:`VentAirViolation` per (duct, volume) that
    lacks a role, in (duct, volume) order; ``[]`` when the rule holds."""
    vol = air_volumes(vents, edges)
    members: dict = {}
    for rid, v in vol.items():
        members.setdefault(v, []).append(rid)
    by: dict = {}
    for rid, triples in vents.items():
        for duct, roles, shorthand in triples:
            if shorthand:
                continue
            slot = by.setdefault((duct, vol[rid]),
                                 {r: [] for r in VENT_ROLES})
            for role in roles:
                slot[role].append(rid)
    out = []
    for (duct, v), slot in sorted(by.items(), key=lambda kv: kv[0]):
        for role in VENT_ROLES:
            if slot[role]:
                continue
            have = next(r for r in VENT_ROLES if slot[r])
            out.append(VentAirViolation(
                duct=duct, rooms=tuple(sorted(members[v], key=_id_key)),
                have=have, have_rooms=tuple(sorted(slot[have], key=_id_key)),
                missing=role))
    return out


def _check_fuel(v) -> bool:
    """``int >= 0`` (v1: clusters of :data:`FUEL_MIX_V1`) or ``{"clusters":
    int >= 0, "mix": {material name: fraction > 0, ...}}`` with the
    fractions summing to 1. The shape only: the names are resolved against
    the material table (and a non-fuel material refused by name) in
    :meth:`ShipGraph.check`."""
    if _is_int(v):
        return v >= 0
    if not (isinstance(v, dict) and set(v) == {"clusters", "mix"}):
        return False
    n, mix = v["clusters"], v["mix"]
    return (_is_int(n) and n >= 0 and isinstance(mix, dict) and len(mix) >= 1
            and all(isinstance(k, str) and _is_num(f) and f > 0
                    for k, f in mix.items())
            and abs(sum(float(f) for f in mix.values()) - 1.0) <= 1e-9)


def fuel_materials() -> tuple:
    """THE fuel / contents set: the material ids a fuel class may stamp,
    read off the material table -- the ``flammable`` rows (furniture, wood,
    kindling, foliage today). What a fuel cluster is made of is a room's
    CONTENTS, never its structure: :func:`compartments` labels through it,
    :func:`_floor_site` keeps clear of it, :func:`embed` counts it. A
    structural row (hull, steel, glass, the doors) is not flammable, so a mix
    naming one is refused."""
    global _FUEL_MATS
    if _FUEL_MATS is None:
        tbl = _table()
        _FUEL_MATS = tuple(i for i, f in enumerate(tbl.flammable) if f)
    return _FUEL_MATS


_FUEL_MATS = None


def fuel_spec(tag) -> tuple:
    """A ``fuel`` tag as ``(clusters, ((material id, fraction), ...))``, the
    mix in material-NAME order (so the int form, :data:`FUEL_MIX_V1`, walks
    furniture then wood, v1's draw). Refuses an unknown name or a material
    outside :func:`fuel_materials`, naming it."""
    if _is_int(tag):
        n, mix = int(tag), FUEL_MIX_V1
    else:
        n, mix = int(tag["clusters"]), tag["mix"]
    names = list(_table().names)
    fuel = fuel_materials()
    out = []
    for name in sorted(mix):
        if name not in names:
            raise GraphSchemaError(
                f"fuel mix material {name!r} has no [materials.{name}] row "
                f"(known: {names})")
        mid = names.index(name)
        if mid not in fuel:
            raise GraphSchemaError(
                f"fuel mix material {name!r} is structure, not fuel (its "
                f"[materials.{name}] row is not flammable): a fuel class "
                f"stamps a room's contents only "
                f"({[names[i] for i in fuel]})")
        out.append((mid, float(mix[name])))
    return n, tuple(out)


def _draw_fuel_material(rng, mix) -> int:
    """One ``rng.random()`` walked over the mix's cumulative fractions (the
    last entry takes the rounding remainder)."""
    r = rng.random()
    acc = 0.0
    for mid, f in mix[:-1]:
        acc += f
        if r < acc:
            return mid
    return mix[-1][0]


def _check_hull_door(v) -> bool:
    """``{}`` or a subset of ``{"width": int >= 1, "state": "closed" |
    "open"}`` (defaults :data:`HULL_DOOR_DEFAULT`). The shape only: an
    ``"open"`` state is refused by name in :meth:`ShipGraph.check`."""
    return (isinstance(v, dict) and set(v) <= set(HULL_DOOR_DEFAULT)
            and (_is_int(v.get("width", 1)) and v.get("width", 1) >= 1)
            and v.get("state", "closed") in HULL_DOOR_STATES)


def hull_door_spec(tag) -> tuple:
    """A ``hull_door`` tag as ``(width, state)`` with the defaults."""
    full = {**HULL_DOOR_DEFAULT, **tag}
    return int(full["width"]), full["state"]


# THE per-room tag schema (all keys optional): key -> (what it is, check).
# A department writes these; stage 2 reads ``spawn_team``, ``zombie``,
# ``fuel``, ``vent``, ``hazard``, ``heat_source``, ``temp``, ``hull_door``,
# ``light`` and ``lamp_span``, and carries ``dept``, ``role``,
# ``cell`` and ``secret`` into ``Embedding.room_tags`` for the vent builder
# and the grammar. Generator-side data: none of them is a level field.
ROOM_TAGS = {
    "spawn_team": ("int: the team seated here (a spawn room)", _is_int),
    "zombie":     ("bool: the zombies' spawn room",
                   lambda v: isinstance(v, bool)),
    "fuel":       ("int >= 0 (fuel clusters against the walls, v1's "
                   'furniture / wood 60/40) or {"clusters": int >= 0, "mix": '
                   "{[materials.*] name: fraction > 0, ...} summing to 1} (a "
                   "fuel class)", _check_fuel),
    "vent":       ('"supply" | "return" (a vent on the main duct) or '
                   '{"duct": name, "roles": ["supply"] | ["return"] | '
                   '["supply", "return"]} or a list of such dicts on '
                   "distinct ducts", _check_vent),
    "lamp_span":  (f"int >= {LAMP_SPAN_MIN}: lamps per length -- "
                   "ceil(long side / lamp_span) lamps along the room's long "
                   "axis, one centred on each equal span (absent: one "
                   "centred lamp)",
                   lambda v: _is_int(v) and v >= LAMP_SPAN_MIN),
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
    "hull_door":  ('{"width": tiles >= 1 (default 3), "state": "closed" '
                   '(default; "open" is refused in v1)}: an A6 door entity '
                   "cut into the room's HULL wall, centred on its longest "
                   "hull run -- the cargo hold's door to space, a valve the "
                   "level or a wire opens (B.4)", _check_hull_door),
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
            if "hull_door" in rm.tags:
                self._check_hull_door(rm)
            if "fuel" in rm.tags:
                try:
                    fuel_spec(rm.tags["fuel"])
                except GraphSchemaError as err:
                    raise GraphSchemaError(f"room {rm.id}: {err}") from None
        n = len(self.rooms)
        pairs: dict = {}
        for e in self.edges:
            if e.kind not in EDGE_KINDS:
                raise GraphSchemaError(
                    f"edge {e.a}-{e.b}: unknown kind {e.kind!r} (known: "
                    f"{list(EDGE_KINDS)})")
            if not (0 <= e.a < n and 0 <= e.b < n) or e.a == e.b:
                raise GraphSchemaError(
                    f"edge {e.a}-{e.b}: not two distinct rooms of {n}")
            pairs.setdefault((min(e.a, e.b), max(e.a, e.b)), []).append(e)
        for (a, b), es in sorted(pairs.items()):
            if len(es) == 1:
                continue
            if sorted(e.kind for e in es) != sorted(BYPASS_KINDS):
                raise GraphSchemaError(
                    f"rooms {a}-{b} carry {len(es)} edges "
                    f"({[e.kind for e in es]}): a room pair carries one edge, "
                    f"bar the airlock bypass (exactly one airlock + one "
                    f"breakable on the shared wall)")
            self._check_bypass(es)
        self._check_ducts_and_cells()

    def _check_bypass(self, es: list) -> None:
        """The AIRLOCK BYPASS (layout doc §4.3): the airlock's and the
        breakable's openings on the one shared wall, both ``at`` explicit;
        the outer door keeps a steel wall-mount tile each side (its transfer
        pump and side probe mount there) and the breakable a steel clearance
        tile each side, all inside the shared run -- so the two openings
        never overlap and the run is at least ``1 + AIRLOCK_W + 1 + 1 +
        DOOR_W + 1`` long. Refused by name (the geometry is the graph's)."""
        al = next(e for e in es if e.kind == "airlock")
        br = next(e for e in es if e.kind == "breakable")
        tag = f"airlock bypass {al.a}-{al.b}"
        if al.at is None or br.at is None:
            raise GraphSchemaError(
                f"{tag}: the airlock's and the breakable's `at` must both be "
                f"set (got {al.at!r}, {br.at!r}) -- two openings on one wall "
                f"cannot both be centred")
        wall = shared_wall(self.rooms[al.a], self.rooms[al.b])
        if wall is None:
            raise GraphSchemaError(f"{tag}: the rooms share no wall")
        _o, _line, lo, hi = wall
        na, nb = al.spec.width, br.spec.width
        span_a = (al.at - 1, al.at + na)          # the outer door + its mounts
        span_b = (br.at - 1, br.at + nb)          # the breakable + clearance
        for what, (s0, s1) in (("the airlock's outer door with its two wall "
                                "mounts", span_a),
                               ("the breakable with its two clearance tiles",
                                span_b)):
            if s0 < lo or s1 > hi:
                raise GraphSchemaError(
                    f"{tag}: {what} ({s0}..{s1}) leaves the shared wall "
                    f"{lo}..{hi}")
        if not (span_a[1] < span_b[0] or span_b[1] < span_a[0]):
            raise GraphSchemaError(
                f"{tag}: the outer door's mounts {span_a[0]}..{span_a[1]} and "
                f"the breakable's clearance {span_b[0]}..{span_b[1]} overlap "
                f"-- the wall between the two openings must hold a mount "
                f"tile AND a clearance tile")

    def _check_hull_door(self, rm: Room) -> None:
        """A ``hull_door`` tag: refuse ``state = "open"`` (v1: the breach
        archetype is not a thing the gates accept yet) and a room the door
        cannot be cut into (:func:`hull_door_site` names why)."""
        _n, state = hull_door_spec(rm.tags["hull_door"])
        if state == "open":
            raise GraphSchemaError(
                f"room {rm.id}: hull_door state 'open' is refused in v1 -- a "
                f"hull door standing open is a room open to space, the future "
                f"BREACH archetype (a room whose air path to vacuum is "
                f"designed, not a leak), which no gate accepts yet; emit it "
                f"'closed' and open it in play (a wire, the level)")
        hull_door_site(self, rm)

    def _check_ducts_and_cells(self) -> None:
        """The vocabulary's §0.5 and L5 rules on the graph:

        - the ``ducts`` annotation names a known ``[filters.*]`` row;
        - a duct's rooms lie in ONE pressure cell (``cell`` tag; rooms
          without one share an implicit cell): a duct never crosses a cell
          boundary;
        - THE AIR-VOLUME RULE (:func:`vent_air_rule`, vocabulary §0.6):
          every air volume (:meth:`air_volume_of`) a duct has a vent in
          carries a supply AND a return on that duct. v1's string shorthand
          is exempt: v1's ``--vents`` pair is exactly a lone supply / lone
          return (critique finding 2), and levels/ship_demo is pinned
          byte-identical (#151);
        - a ``vent_only`` edge joins two rooms that share a duct (the air
          path is the duct; nothing is cut in the wall);
        - an edge whose rooms lie in different cells is an airlock, a
          bulkhead or a breakable wall, never a plain door (L5 as
          amended)."""
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
        specs = {rm.id: vent_specs(rm.tags["vent"]) for rm in self.rooms
                 if "vent" in rm.tags}
        cells: dict = {}
        for rid, triples in specs.items():
            for (duct, _roles, _sh) in triples:
                cells.setdefault(duct, {}).setdefault(
                    self.rooms[rid].tags.get("cell"), []).append(rid)
        for duct, by_cell in sorted(cells.items()):
            if len(by_cell) > 1:
                desc = ", ".join(f"cell {c}: rooms {ids}" for c, ids in
                                 sorted(by_cell.items(), key=repr))
                raise GraphSchemaError(
                    f"duct {duct!r} spans {len(by_cell)} cells ({desc}): a "
                    f"duct never crosses a cell boundary (vocabulary §0.5)")
        bad = vent_air_rule({rm.id: specs.get(rm.id, ()) for rm in self.rooms},
                            [(e.a, e.b, e.kind) for e in self.edges])
        if bad:
            raise GraphSchemaError(str(bad[0]))
        for e in self.edges:
            if e.kind == "vent_only":
                da = [t[0] for t in specs.get(e.a, ())]
                db = [t[0] for t in specs.get(e.b, ())]
                if not set(da) & set(db):
                    raise GraphSchemaError(
                        f"edge {e.a}-{e.b} (vent_only): rooms {e.a} and "
                        f"{e.b} are not on one duct (room {e.a}: {da or None}"
                        f", room {e.b}: {db or None}) -- a vent_only edge's "
                        f"air path IS the duct")
            ca = self.rooms[e.a].tags.get("cell")
            cb = self.rooms[e.b].tags.get("cell")
            if ca != cb and e.kind not in CELL_CROSSING_KINDS:
                raise GraphSchemaError(
                    f"edge {e.a}-{e.b}: kind {e.kind!r} crosses cells {ca} "
                    f"-> {cb}; an edge crossing a cell boundary is an "
                    f"airlock, a bulkhead or a breakable wall, never a "
                    f"plain door (L5)")

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

    def air_volume_of(self) -> list:
        """Room id -> AIR VOLUME index (:func:`air_volumes`; vocabulary §0.6
        item 1), the twin of :meth:`compartment_of`: rooms joined by an
        ``air`` edge (``open``, ``grille``, ``vented_door``) are one volume;
        indices run 0..k-1 in the order of each volume's lowest room id. A
        ``vented_door`` bounds a compartment but not a volume."""
        vol = air_volumes(range(len(self.rooms)),
                          [(e.a, e.b, e.kind) for e in self.edges])
        return [vol[i] for i in range(len(self.rooms))]


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


def _opening_span(graph: ShipGraph, e: Edge):
    """``(orient, line, at, n, lo, hi)`` of an edge's opening in its shared
    wall (``at`` None for a kind that cuts nothing; the rooms must still
    share a wall: every kind in the table is ``must-share-wall``)."""
    wall = shared_wall(graph.rooms[e.a], graph.rooms[e.b])
    if wall is None:
        raise _Reject(f"edge {e.a}-{e.b}: the rooms share no wall")
    orient, line, lo, hi = wall
    n = e.spec.width
    if e.spec.stamps is None:
        return orient, line, None, n, lo, hi
    at = lo + (hi - lo + 1 - n) // 2 if e.at is None else e.at
    if at < lo or at + n - 1 > hi:
        raise _Reject(f"edge {e.a}-{e.b}: opening {at}+{n} leaves {lo}..{hi}")
    return orient, line, at, n, lo, hi


def _wall_tiles(orient: str, line: int, ts) -> list:
    if orient == "v":
        return [(t, line) for t in ts]
    return [(line, t) for t in ts]


def _opening_tiles(graph: ShipGraph, e: Edge) -> list:
    """The (y, x) tiles an edge opens in its shared wall (none for a kind
    that cuts nothing). A ``vented_door``'s jambs are NOT opening tiles
    (:func:`_jamb_tiles`)."""
    orient, line, at, n, _lo, _hi = _opening_span(graph, e)
    if at is None:
        return []
    return _wall_tiles(orient, line, range(at, at + n))


def _jamb_tiles(graph: ShipGraph, e: Edge) -> list:
    """The two wall tiles flanking an edge's opening whose kind stamps
    ``jambs`` (none otherwise), refused when one would leave the shared run
    -- a jamb past the run's end is a corner, whose far side may be a third
    room's air."""
    if e.spec.jambs is None:
        return []
    orient, line, at, n, lo, hi = _opening_span(graph, e)
    if at - 1 < lo or at + n > hi:
        raise _Reject(f"edge {e.a}-{e.b} ({e.kind}): a jamb ({at - 1} or "
                      f"{at + n}) leaves the shared wall {lo}..{hi}")
    return _wall_tiles(orient, line, (at - 1, at + n))


def _hull_runs(graph: ShipGraph, rm: Room) -> list:
    """The room's sides that lie against the hull ring, as ``(orient, line,
    lo, hi, s)`` in the order top, bottom, left, right: ``orient`` the
    wall's (and its door's) orientation, ``line`` the hull row / column,
    ``lo..hi`` the run along it, ``s`` the step from the hull tile to the
    space beyond (-1 / +1)."""
    hx0, hy0, hx1, hy1 = graph.hull
    x0, y0, x1, y1 = rm.box
    runs = []
    if y0 - 1 == hy0:
        runs.append(("h", hy0, x0, x1, -1))
    if y1 + 1 == hy1:
        runs.append(("h", hy1, x0, x1, 1))
    if x0 - 1 == hx0:
        runs.append(("v", hx0, y0, y1, -1))
    if x1 + 1 == hx1:
        runs.append(("v", hx1, y0, y1, 1))
    return runs


def hull_door_site(graph: ShipGraph, rm: Room) -> tuple:
    """Where a room's ``hull_door`` goes: ``(orient, tiles, beyond)`` --
    the door's orientation, its span in the hull wall (row-major) and the
    space tiles one step past it. Centred on the room's LONGEST hull run
    (ties: top, bottom, left, right -- an order the mirror maps onto
    itself). In a mirrored graph a wing room whose lower-id twin carries the
    same tag takes the REFLECTION of the twin's site (the fuel clusters'
    convention), so a pair is symmetric whatever the parity; an AXIS room
    uses only its top / bottom hull runs (each mirrors onto itself) and must
    centre its door exactly on the axis. Refuses, by name, a room with no
    hull wall, a door wider than the run, and an axis door that cannot be
    centred (GraphSchemaError: the geometry is the graph's)."""
    n, _state = hull_door_spec(rm.tags["hull_door"])
    W, twin = graph.width, rm.mirror
    if (graph.mirror_x and twin is not None and twin != rm.id and twin < rm.id
            and graph.rooms[twin].tags.get("hull_door")
            == rm.tags["hull_door"]):
        orient, tiles, beyond = hull_door_site(graph, graph.rooms[twin])

        def refl(ts):
            return tuple(sorted((y, W - 1 - x) for (y, x) in ts))
        return orient, refl(tiles), refl(beyond)
    axis = graph.mirror_x and twin == rm.id
    runs = [r for r in _hull_runs(graph, rm) if not (axis and r[0] == "v")]
    if not runs:
        raise GraphSchemaError(
            f"room {rm.id}: tag 'hull_door' but the room has no hull wall "
            f"(no side of its box {rm.box} lies against the hull "
            f"{graph.hull})"
            + ("; an axis room's hull door goes in its top or bottom hull "
               "wall, the ones that mirror onto themselves" if axis else ""))
    orient, line, lo, hi, s = max(runs, key=lambda r: r[3] - r[2])
    length = hi - lo + 1
    if n > length:
        raise GraphSchemaError(
            f"room {rm.id}: a {n}-wide hull_door does not fit its longest "
            f"hull run ({length} tiles)")
    if axis and (length - n) % 2:
        raise GraphSchemaError(
            f"room {rm.id} (on the mirror axis): a {n}-wide hull_door cannot "
            f"be centred on its {length}-tile hull run symmetrically -- the "
            f"door's and the run's widths must share a parity")
    at = lo + (length - n) // 2
    if orient == "h":
        tiles = tuple((line, x) for x in range(at, at + n))
        beyond = tuple((line + s, x) for x in range(at, at + n))
    else:
        tiles = tuple((y, line) for y in range(at, at + n))
        beyond = tuple((y, line + s) for y in range(at, at + n))
    return orient, tiles, beyond


def _hull_doors(graph: ShipGraph) -> list:
    """``[(room, orient, tiles, beyond, state)]`` for every ``hull_door``
    room, in room order."""
    out = []
    for rm in graph.rooms:
        if "hull_door" in rm.tags:
            orient, tiles, beyond = hull_door_site(graph, rm)
            out.append((rm, orient, tiles, beyond,
                        hull_door_spec(rm.tags["hull_door"])[1]))
    return out


def _cut_hull_door(tm, rm: Room, tiles, beyond, state: str) -> None:
    """Stamp a hull door's span with its initial state's material, refusing
    a span that is not hull with space beyond and the room's floor air
    within (a vestibule carved over it, a hull that is not one tile)."""
    for (y, x), (by, bx) in zip(tiles, beyond):
        iy, ix = 2 * y - by, 2 * x - bx           # the tile on the room's side
        if (int(tm[y, x]) != MAT_HULL or int(tm[by, bx]) != SPACE_CODE
                or int(tm[iy, ix]) != MAT_AIR):
            raise _Reject(f"room {rm.id}: hull door at ({x},{y}) is not a "
                          f"hull tile between the room's air and space")
    for (y, x) in tiles:
        tm[y, x] = _DOOR_STAMP[state]


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
    lights: list                      # ceiling lamps, room by room (LightEntry)
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
    """``room.tags['fuel']`` clusters against its walls, each one material
    drawn from the tag's mix (:func:`fuel_spec`; the int form is v1's 60/40
    furniture / wood), clear of the passages in ``door`` (the opening tiles
    of every ``keep_clear`` edge); in a mirrored graph placed once per pair
    (an axis room: its left half) and reflected into the twin."""
    W = graph.width
    x0, y0, x1, y1 = room.box
    if graph.mirror_x and room.mirror == room.id:
        x1 = min(x1, W // 2 - 1)
    n, mix = fuel_spec(room.tags.get("fuel", 0))
    for _ in range(n):
        mat = _draw_fuel_material(rng, mix)
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
                   (MAT_DOOR, MAT_GLASS, MAT_DOOR_CLOSED, MAT_GRILLE)).any():
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
            if np.isin(tm[y - 1:y + 2, x - 1:x + 2], fuel_materials()).any():
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


# THE DESIGN FLOW's rate quantum (vocabulary §0.6 item 3): a vent's
# ``q_circ`` is in TILE-VOLUMES OF AMBIENT AIR PER SECOND -- vent_system.
# build_vents quantizes ``q_circ / (C * T_amb)`` (== q_circ: the P1
# calibration C * T_amb == 1) through ``quantize_bulk_scalar`` into raw
# Q16.16 bulk N per second, and a tile's whole bulk N at 1 atm ambient is
# exactly 1.0 (13763 + 51773 == 65536 raw). So one raw unit per second is
# the runtime's rate quantum, and every designed rate is ``raw / RATE_ONE``
# with ``raw`` an int: exact in a float, written exactly (repr) and
# quantized back to ``raw`` at load.
RATE_ONE = 1 << 16


def vent_ach() -> float:
    """``[levelgen.vents] ach`` -- the air changes per hour a ventilated
    room gets on the recipe path (PLACEHOLDER, Erik tunes). THE door of the
    design flow's one dial: refuses a missing table or key and a value that
    is not a positive number, naming the key."""
    from config import CFG
    tbl = getattr(getattr(CFG, "levelgen", None), "vents", None)
    ach = getattr(tbl, "ach", None)
    if ach is None:
        raise ValueError(
            "config.toml has no [levelgen.vents] ach: the recipe path's vent "
            "design flow (vocabulary §0.6 item 3) has no fallback rate")
    if not _is_num(ach) or not float(ach) > 0:
        raise ValueError(
            f"[levelgen.vents] ach = {ach!r} is not a positive number (air "
            f"changes per hour of a ventilated room)")
    return float(ach)


def _apportion(total: int, groups: list) -> dict:
    """``total`` (int) shared over ``groups`` -- ``(key, weight, members)``,
    ``members`` the room ids that must get EQUAL shares (a mirror pair) --
    in proportion to ``weight`` (the group's summed area) by largest
    remainder in exact integers: room id -> int, summing to ``total``. Each
    group's share is a multiple of its member count; ties go to the lowest
    key. The caller makes ``total`` reachable (even when every group is a
    pair)."""
    wsum = sum(w for _k, w, _m in groups)
    alloc = {k: (total * w // wsum) // len(m) * len(m) for k, w, m in groups}
    rem = total - sum(alloc.values())
    while rem > 0:
        fits = [(k, w, m) for k, w, m in groups if len(m) <= rem]
        if not fits:
            raise ValueError(f"cannot apportion {total} over {groups}")
        k, _w, m = max(fits, key=lambda g: (total * g[1] - alloc[g[0]] * wsum,
                                            [-x for x in sorted(g[2])]))
        alloc[k] += len(m)
        rem -= len(m)
    return {rid: alloc[k] // len(m) for k, _w, m in groups for rid in m}


def vent_rates(graph: ShipGraph, ach: float) -> dict:
    """THE DESIGN FLOW (vocabulary §0.6 item 3): ``(room id, duct, role) ->
    raw rate`` (int, raw Q16.16 bulk N per second, :data:`RATE_ONE` = one
    tile-volume of ambient air per second) for every NON-shorthand vent.

    Per duct D and air volume V (:meth:`ShipGraph.air_volume_of`) it has
    vents in: the served rooms are V's rooms with a vent on D, and ``F =
    ach / 3600 x (their floor area in tiles)`` tile-volumes per second,
    rounded to the quantum. D's supplies in V share F in proportion to their
    rooms' areas; D's returns in V share the SAME integer F the same way --
    so in every volume a duct's supply equals its return, exactly, and the
    runtime (which hands the supplies what the returns collected, in
    proportion to their rates) gives each volume back what it gave. Mirror
    twins get equal rates: twins on mirrored ducts solve mirrored problems
    (room ids keep their order under the mirror); twins on ONE duct in ONE
    volume share as a pair (F made even when every share is a pair). v1's
    shorthand is not here (it keeps :data:`Q_CIRC`)."""
    vol = graph.air_volume_of()
    by: dict = {}
    for rm in graph.rooms:
        if "vent" not in rm.tags:
            continue
        for duct, roles, shorthand in vent_specs(rm.tags["vent"]):
            if shorthand:
                continue
            slot = by.setdefault((duct, vol[rm.id]),
                                 {r: [] for r in VENT_ROLES})
            for role in roles:
                slot[role].append(rm.id)
    per_tile = ach / 3600.0 * RATE_ONE
    out = {}
    for (duct, _v), slot in sorted(by.items(), key=lambda kv: kv[0]):
        served = sorted(set(slot["supply"]) | set(slot["return"]))
        area = {rid: graph.rooms[rid].w * graph.rooms[rid].h
                for rid in served}
        side_groups = {}
        for role in VENT_ROLES:
            groups, seen = [], set()
            for rid in sorted(slot[role]):
                if rid in seen:
                    continue
                tw = graph.rooms[rid].mirror if graph.mirror_x else None
                pair = (tw is not None and tw != rid and tw in slot[role])
                m = (rid, tw) if pair else (rid,)
                seen.update(m)
                groups.append((rid, sum(area[x] for x in m), m))
            side_groups[role] = groups
        total = max(1, round(per_tile * sum(area.values())))
        if any(all(len(m) == 2 for _k, _w, m in g)
               for g in side_groups.values()) and total % 2:
            total += 1
        for role in VENT_ROLES:
            for rid, raw in _apportion(total, side_groups[role]).items():
                out[(rid, duct, role)] = raw
    return out


def _build_vents(tm, graph: ShipGraph, reserved=None,
                 ach: float | None = None) -> list:
    """One duct entity per duct name the rooms' ``vent`` tags use (id
    ``duct_<name>``, its filter :meth:`ShipGraph.duct_filter`), then that
    duct's wall vents, one per (room, role): supply rooms first, then by
    room id. Ducts in name order. A vent's id is ``vent_<duct>_<room>_<role>``;
    v1's string shorthand keeps v1's ``vent_<role>`` (so v1 ships write the
    same file), and a collision of those is refused by name. A room on
    several ducts (a list tag) gets a site per (duct, role); every site is
    clear of every earlier one, whatever its duct.

    A vent's ``q_circ`` is the DESIGN FLOW's (:func:`vent_rates` at ``ach``,
    default :func:`vent_ach`); v1's shorthand keeps :data:`Q_CIRC`."""
    tagged = {}
    for rm in graph.rooms:
        if "vent" in rm.tags:
            for duct, roles, shorthand in vent_specs(rm.tags["vent"]):
                for role in roles:
                    tagged.setdefault(duct, []).append((role, rm, shorthand))
    if not tagged:
        return []
    designed = any(not sh for ts in tagged.values() for _r, _rm, sh in ts)
    rates = (vent_rates(graph, vent_ach() if ach is None else ach)
             if designed else {})
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
                        "q_circ": (Q_CIRC if shorthand else
                                   rates[(rm.id, duct, role)] / RATE_ONE)},
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
    state ``ENTITY_DOOR_INITIAL``), one per ``hull_door`` room (id
    ``hull_door_r<room id>``, the tag's state, closed by default), and per
    ``airlock`` edge its mode's
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
    for rm, orient, tiles, _beyond, state in _hull_doors(graph):
        ents.append(_door_entity(f"hull_door_r{rm.id}", tiles, orient,
                                 state))
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
        for (y, x) in _jamb_tiles(graph, e):
            if tm[y, x] != MAT_STEEL:
                raise _Reject(f"edge {e.a}-{e.b} ({e.kind}): its jamb at "
                              f"({x},{y}) is not steel ({int(tm[y, x])})")
            tm[y, x] = spec.jambs
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    modes = {}
    for i, al in _airlocks(graph):
        modes[i] = airlock_mode(tm, al)       # read BEFORE the carve
        _carve_airlock(tm, al, modes[i])
        for (y, x) in al.carve:
            passage[y, x] = reserved[y, x] = True
        for (my, mx), _acted in _airlock_mounts(al, modes[i]):
            reserved[my, mx] = True
    for rm, _o, tiles, beyond, state in _hull_doors(graph):
        _cut_hull_door(tm, rm, tiles, beyond, state)
        for (y, x) in tiles:
            passage[y, x] = reserved[y, x] = True
    doors = by_kind.get("door", 0) + by_kind.get("vented_door", 0)
    windows = by_kind.get("window", 0)
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
    fuel = int(np.isin(tm, fuel_materials()).sum())
    lights = _ship_lamps(graph)
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


def _lamp_light(lamps) -> float:
    """The most light, in light units, the lamps carry on any one channel
    (``intensity x colour`` summed, the cone door's quantity)."""
    return max((sum(l.intensity * l.color[c] for l in lamps)
                for c in range(3)), default=0.0)


def _ship_lamps(graph: ShipGraph) -> list:
    """Every room's lamps (:func:`_room_lamps`), held within
    :data:`LAMP_LIGHT_BUDGET` when any room carries ``lamp_span`` (the
    recipe path; a graph without it -- v1 -- is returned untouched): the
    spans are widened together (each room's span at least ``s``, ``s``
    growing by one tile) until the lamps fit, and if one lamp per room still
    does not, every lamp is dimmed by the one factor that fits. Pure
    arithmetic on the graph, no draw; a span floor is the same for twins, so
    the mirror holds."""
    lamps = [l for rm in graph.rooms for l in _room_lamps(rm)]
    spanned = [rm for rm in graph.rooms if "lamp_span" in rm.tags]
    if not spanned or _lamp_light(lamps) <= LAMP_LIGHT_BUDGET:
        return lamps
    longest = max(max(rm.w, rm.h) for rm in spanned)
    floor = min(int(rm.tags["lamp_span"]) for rm in spanned)
    while _lamp_light(lamps) > LAMP_LIGHT_BUDGET and floor <= longest:
        floor += 1
        lamps = [l for rm in graph.rooms for l in _room_lamps(rm, floor)]
    total = _lamp_light(lamps)
    if total > LAMP_LIGHT_BUDGET:
        k = LAMP_LIGHT_BUDGET / total
        lamps = [LightEntry(x=l.x, y=l.y, color=l.color,
                            intensity=l.intensity * k, range=l.range,
                            kind=l.kind) for l in lamps]
    return lamps


def _room_lamps(rm: Room, span_floor: int = 0) -> list:
    """A room's static lamps (render light only), in its ``light`` colour
    and level (else LAMP_RGB / LAMP_INTENSITY): ONE at its centre, or --
    with a ``lamp_span`` tag -- ``ceil(long side / lamp_span)`` along its
    long axis (x on a square room), the long side cut into that many equal
    spans with a lamp at each span's centre, the short axis centred. The
    positions are symmetric about the room's centre, so a mirror twin's
    lamps are the reflection of its twin's. A ``light`` tag of level 0 (the
    ``dark`` preset: the fuel bunker) is a room with NO lamp -- the level
    format refuses a zero-intensity ``[[light]]``. ``span_floor``: the
    budget's widened span (:func:`_ship_lamps`); a tagged room uses
    ``max(lamp_span, span_floor)``."""
    rgb, level = LAMP_RGB, LAMP_INTENSITY
    if "light" in rm.tags:
        rgb = tuple(float(c) for c in rm.tags["light"][0])
        level = float(rm.tags["light"][1])
        if level <= 0.0:
            return []
    span = max(int(rm.tags.get("lamp_span", 0)), int(span_floor))
    if "lamp_span" not in rm.tags:
        spots = [(rm.x + rm.w / 2.0, rm.y + rm.h / 2.0)]
    elif rm.w >= rm.h:
        n = -(-rm.w // span)
        spots = [(rm.x + (k + 0.5) * rm.w / n, rm.y + rm.h / 2.0)
                 for k in range(n)]
    else:
        n = -(-rm.h // span)
        spots = [(rm.x + rm.w / 2.0, rm.y + (k + 0.5) * rm.h / n)
                 for k in range(n)]
    return [LightEntry(x=x, y=y, color=rgb, intensity=level,
                       range=LAMP_RANGE, kind="static") for (x, y) in spots]


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
    the structure (hull, steel, doors, glass, grilles -- a grille passes air
    but is structure here, #134) -- fuel is a room's contents,
    so a pocket of air behind a crate belongs to its room. With ``graph``
    the STRUCTURE MASK also takes the graph's edge kinds: the opening tiles
    of every edge that bounds a compartment are structure whatever they are
    made of (a ``breakable`` edge's wood segment is a wall, not a crate),
    while an ``open`` edge's air gap joins its rooms as the table says. An
    entity door's span is structure even while it stands open (it can
    close), and so is an airlock's inner door: each vestibule is its own
    compartment. Contents = air + :func:`fuel_materials` (whatever a fuel
    class may stamp: a kindling cluster never splits its room)."""
    from scipy import ndimage
    contents = np.isin(tm, (MAT_AIR,) + fuel_materials())
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


def centred_spawn_box(room: SpawnRoom) -> SpawnRoom:
    """The RECIPE path's spawn room (#136 LevelSpec patch): the least box
    that seats ``room.max_per_team`` under ``formation`` across the room's
    whole SHORT side, centred along its long side. ``formation`` fills its
    grid row-major from the room's first row, so in a whole 8-wide slab
    (or a tall band section) the squad stood at the fore end; in the box it
    stands in the middle. The box keeps the room's parity along the long
    side, so its :func:`_centre` (the A* endpoint of :func:`validate` and the
    bot path) is the room's own, and a mirrored pair of rooms gives a
    mirrored pair of boxes. A room no longer than the box is returned as
    is."""
    vertical = room.h >= room.w
    short, long_ = (room.w, room.h) if vertical else (room.h, room.w)
    per_row = (short + FORMATION_GAP) // FORMATION_PITCH
    if per_row < 1:
        return room
    need = -(-room.max_per_team // per_row)
    side = need * FORMATION_PITCH - FORMATION_GAP
    if (long_ - side) % 2:
        side += 1                       # same parity: the centre is unmoved
    if side >= long_:
        return room
    off = (long_ - side) // 2
    if vertical:
        return SpawnRoom(team=room.team, x=room.x, y=room.y + off, w=room.w,
                         h=side, max_per_team=room.max_per_team,
                         placement=room.placement)
    return SpawnRoom(team=room.team, x=room.x + off, y=room.y, w=side,
                     h=room.h, max_per_team=room.max_per_team,
                     placement=room.placement)


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
    """A generated ship, validated, not yet written. A RECIPE ship (family
    B: ``generate(recipe=...)``) also carries the archetype's name, the
    profile, the grammar attempt ``g`` (``attempt`` is then the layout
    attempt ``a`` of that graph), the recipe hash, the quotient graph it was
    laid out from and the layout's quotient-id -> room-ids map."""
    seed: int
    size: str
    attempt: int                      # the sub-seed that passed (0 = first)
    graph: ShipGraph                  # stage 1
    embedding: Embedding              # stage 2
    rejected: list = field(default_factory=list)   # (attempt, reasons)
    recipe: str | None = None         # the archetype (None: plan_graph's v1)
    profile: str | None = None
    g: int = 0                        # the grammar attempt that passed
    recipe_hash: str | None = None
    quotient: object | None = None    # ship_grammar.QuotientGraph
    room_ids: dict = field(default_factory=dict)   # quotient id -> (id, twin)
    zombies_base: int | None = None   # the request, before the profile's scale

    @property
    def name(self) -> str:
        return default_name(self.seed, self.size, self.recipe, self.profile)

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


def default_name(seed: int, size: str, recipe: str | None = None,
                 profile: str | None = None) -> str:
    """``ship_s<seed>_<size>``; a recipe ship (archetype doc §5.1)
    ``ship_<archetype>_<profile>_s<seed>_<size>``."""
    if recipe is None:
        return f"ship_s{seed}_{size}"
    return f"ship_{recipe}_{profile}_s{seed}_{size}"


def generate(seed: int, size: str = "medium", *, max_per_team: int = 6,
             zombies: int = 4, vents: bool = False, recipe=None,
             profile: str = "standard") -> ShipPlan:
    """A validated ship for ``seed`` (pure: no files). Tries sub-seeds
    ``attempt = 0, 1, ...``: stage 1, stage 2, the gates; the first plan
    that passes is returned.

    ``recipe`` (an archetype name under ``levels/recipes/archetypes`` or a
    loaded ``ship_grammar.Archetype``) switches stage 1 to FAMILY B
    (:func:`_generate_recipe`): the grammar, then the layout stage, then the
    same :func:`embed` and :func:`validate`; ``profile`` is the archetype's
    difficulty profile. Without it nothing below changes: v1's
    :func:`plan_graph`, its streams and its output."""
    if recipe is not None:
        return _generate_recipe(seed, size, recipe, profile, max_per_team,
                                zombies, vents)
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


def _generate_recipe(seed: int, size: str, recipe, profile: str,
                     max_per_team: int, zombies: int, vents: bool
                     ) -> ShipPlan:
    """THE RECIPE PATH (family B; archetype doc §2): two attempt loops and
    two rng streams. Grammar attempt ``g`` (``ship_grammar.plan_quotient``,
    every draw from ``SeedSequence([seed, 0, g])``) gives a quotient graph
    the layout's arithmetic accepts; then up to ``MAX_ATTEMPTS`` layout
    attempts ``a``, each drawing the layout (``ship_layout.layout``) AND
    stage 2 (:func:`embed`) from ``SeedSequence([seed, 1, g, a])``, through
    :func:`validate` and the archetype's own gates
    (``ship_layout.acceptance``). A graph no attempt realises hands over to
    grammar attempt ``g + 1``; when the grammar's ``G_MAX`` attempts are
    spent: :class:`ShipGenError`. The plan records ``(g, a)``; ``rejected``
    holds ``((g, a), reasons)`` per layout attempt that failed."""
    import ship_grammar
    import ship_layout
    if vents:
        raise ShipGenError("vents = True is v1's single HEPA pair; a recipe "
                           "ship's ducts are its departments'")
    arch = (recipe if isinstance(recipe, ship_grammar.Archetype)
            else ship_grammar.load_archetype(recipe))
    ship_layout.check_request(arch, size, profile, max_per_team, zombies)
    rejected, g = [], 0
    while True:
        try:
            q = ship_grammar.plan_quotient(arch, size, profile, int(seed),
                                           max_per_team=max_per_team,
                                           zombies=zombies, g0=g)
        except ship_grammar.GrammarExhausted as e:
            last = rejected[-1][1] if rejected else [str(e)]
            raise ShipGenError(
                f"archetype {arch.name!r} profile {profile!r} seed {seed} "
                f"size {size!r}: no plan passed the gates in "
                f"{ship_grammar.G_MAX} grammar x {MAX_ATTEMPTS} layout "
                f"attempts; last: {last}") from None
        for a in range(MAX_ATTEMPTS):
            rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([int(seed), 1, q.g, a])))
            try:
                lay = ship_layout.layout(q, rng)
                emb = embed(lay.graph, rng)
                emb.spawn_rooms = [centred_spawn_box(r)
                                   for r in emb.spawn_rooms]
            except _Reject as e:
                rejected.append(((q.g, a), [str(e)]))
                continue
            reasons = validate(emb.tilemap, emb.spawn_rooms, mirrored=True,
                               graph=lay.graph)
            if not reasons:
                reasons = ship_layout.acceptance(arch, q, lay.graph, emb)
            if not reasons:
                return ShipPlan(
                    seed=int(seed), size=size, attempt=a, graph=lay.graph,
                    embedding=emb, rejected=rejected, recipe=arch.name,
                    profile=profile, g=q.g, recipe_hash=q.recipe_hash,
                    quotient=q, room_ids=lay.ids, zombies_base=zombies)
            rejected.append(((q.g, a), reasons))
        g = q.g + 1


# ===========================================================================
# LevelSpec: the [levelgen] record, and a ship from it (archetype doc §5.1)
# ===========================================================================

class StaleRecipeError(ShipGenError):
    """A ``[levelgen]`` record whose ``recipe_hash`` is not the current
    library's: the recipe was retuned since, so the ship cannot be
    reproduced from it."""


def generator_version() -> str:
    """``git describe --always --dirty`` of this checkout (#126's recording
    header ``build``, the one helper: ``match_replay.build_id``), or
    ``"unknown"`` when git is not at hand."""
    from match_replay import build_id
    return build_id() or "unknown"


def levelgen_record(plan: ShipPlan, version: str | None = None):
    """A RECIPE ship's ``[levelgen]`` record (``level_loader.LevelgenRecord``):
    its LevelSpec -- every input of :func:`generate` plus the recipe hash and
    ``version`` (default :func:`generator_version`) -- and the factors it
    realised. ``spawn_distance`` is ``ship_layout.bot_path``'s step count,
    the metric the ``[validation]`` gate measures; ``cells`` and ``rooms`` are
    the gate's counts too. ValueError for a v1 plan (no LevelSpec: v1 writes
    no record)."""
    import ship_grammar
    import ship_layout
    from level_loader import LevelgenRecord
    if plan.recipe is None:
        raise ValueError("a v1 ship (no recipe) has no LevelSpec")
    q = plan.quotient
    arch = ship_grammar.load_archetype(plan.recipe)
    applied = {}
    for c in q.cycles:
        applied[c.type] = applied.get(c.type, 0) + 1
    types = dict.fromkeys(e.cycle for e in arch.applies if e.cycle)
    cycles = {t: applied.get(t, 0) for t in list(types) + sorted(
        set(applied) - set(types))}
    path = ship_layout.bot_path(plan.embedding)
    return LevelgenRecord(
        archetype=plan.recipe, recipe_hash=plan.recipe_hash,
        generator_version=version or generator_version(), size=plan.size,
        profile=plan.profile, max_per_team=plan.max_per_team,
        zombies_base=plan.zombies_base, level_seed=plan.seed, g=plan.g,
        a=plan.attempt, departments=tuple(d.name for d in q.departments),
        cycles=cycles,
        cells=len({rm.tags.get("cell") for rm in plan.graph.rooms}),
        secrets=len(q.secrets), spawn_exits=q.exits, zombies=plan.zombies,
        rooms=len(plan.graph.rooms), spawn_distance=max(0, len(path) - 1))


def generate_from_spec(spec, *, allow_stale: bool = False) -> ShipPlan:
    """THE SHIP OF A LEVELSPEC (archetype doc §5.1: a ship is a pure function
    of it). ``spec``: a ``level_loader.LevelgenRecord``, or a level folder
    whose level.toml carries one. Refuses (:class:`StaleRecipeError`) a
    record whose ``recipe_hash`` is not the current library's, unless
    ``allow_stale``: the recipe was retuned, the ship it names is gone. The
    record's ``generator_version`` is not checked (a different build may
    still give the same ship; compare the regenerated record to know)."""
    import ship_grammar
    from level_loader import LevelgenRecord, _parse_levelgen
    if not isinstance(spec, LevelgenRecord):
        import tomllib
        toml_path = Path(spec) / "level.toml"
        with open(toml_path, "rb") as f:
            spec_ = _parse_levelgen(tomllib.load(f), toml_path)
        if spec_ is None:
            raise ShipGenError(f"{toml_path}: no [levelgen] table (a v1 or "
                               f"hand-made level has no LevelSpec)")
        spec = spec_
    arch = ship_grammar.load_archetype(spec.archetype)
    now = ship_grammar.recipe_hash(arch, spec.profile)
    if now != spec.recipe_hash and not allow_stale:
        raise StaleRecipeError(
            f"[levelgen].recipe_hash {spec.recipe_hash[:12]}... is not the "
            f"current {spec.archetype!r} / {spec.profile!r} recipe's "
            f"({now[:12]}...): the recipe was retuned since this ship was "
            f"generated, so it cannot be reproduced (allow_stale / "
            f"--allow-stale generates today's ship from the same inputs)")
    return generate(spec.level_seed, spec.size,
                    max_per_team=spec.max_per_team, zombies=spec.zombies_base,
                    recipe=arch, profile=spec.profile)


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
    MAT_KINDLING:  (142, 62, 36),    # a recipe's fuel class (no v1 ship has it)
    MAT_GRILLE:    (70, 150, 120),   # #134: a grating -- teal, apart from steel
                                     # and glass (no v1 ship has it)
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


def write(plan: ShipPlan, out_dir, *, version: str | None = None) -> Path:
    """Write ``plan`` as a level folder through ``level_lib``; returns the
    folder. Regenerates the data files (anything else there stays). A recipe
    ship also gets its ``[levelgen]`` record (:func:`levelgen_record`;
    ``version`` overrides the git-described generator version); a v1 ship
    writes none."""
    out_dir = Path(out_dir)
    record = (levelgen_record(plan, version) if plan.recipe is not None
              else None)
    for name in DATA_FILES + OPTIONAL_DATA_FILES:
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = plan.tilemap.shape
    if plan.recipe is None:
        head = (
            f"SHIP {plan.name} -- GENERATED by tools/gen_ship.py (#120, "
            f"levelgen",
            f"v1) through level_lib: seed {plan.seed}, size {plan.size} "
            f"({w}x{h}), sub-seed attempt {plan.attempt}.",
            "Regenerate rather than hand-edit:",
            f"  C:/Users/steen/anaconda3/python.exe tools/gen_ship.py --seed "
            f"{plan.seed} --size {plan.size} --max-per-team "
            f"{plan.max_per_team} "
            f"--zombies {plan.zombies}" + (" --vents" if any(
                e.class_name == "duct" for e in plan.entities) else ""),
        )
    else:
        zb = plan.zombies if plan.zombies_base is None else plan.zombies_base
        head = (
            f"SHIP {plan.name} -- GENERATED by tools/gen_ship.py (#136, "
            f"family B)",
            f"through level_lib: archetype {plan.recipe}, profile "
            f"{plan.profile}, seed {plan.seed}, size {plan.size} ({w}x{h}),",
            f"grammar attempt {plan.g}, layout attempt {plan.attempt}, "
            f"recipe hash {plan.recipe_hash}.",
            "Regenerate rather than hand-edit (or --from-level <this folder>,",
            "which reads [levelgen] below):",
            f"  C:/Users/steen/anaconda3/python.exe tools/gen_ship.py --seed "
            f"{plan.seed} --size {plan.size} --recipe {plan.recipe} "
            f"--profile {plan.profile} --max-per-team {plan.max_per_team} "
            f"--zombies {zb}",
        )
    comments = head + (
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
    if record is not None:
        blocks["levelgen"] = lambda nl: level_lib.format_levelgen_lines(
            record, nl)
    level_lib.write_managed_blocks(toml_path, blocks, write_bak=False)
    level_lib.write_match(out_dir, plan.match_spec())
    if plan.embedding.temp_init is not None:
        level_lib.write_temp_init_npy(out_dir, plan.embedding.temp_init,
                                      npy_bak=False)
    return out_dir


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="gen_ship", description="Generate a seeded v1 ship level.")
    ap.add_argument("--seed", type=int, default=None,
                    help="the level seed (required unless --from-level)")
    ap.add_argument("--from-level", default=None, metavar="FOLDER",
                    help="regenerate a recipe ship from the [levelgen] record "
                         "in FOLDER/level.toml (its LevelSpec); every other "
                         "generator option is then refused")
    ap.add_argument("--allow-stale", action="store_true",
                    help="with --from-level: generate even when the record's "
                         "recipe_hash is not the current library's")
    ap.add_argument("--out", default=None,
                    help="level folder (default levels/ship_s<seed>_<size>)")
    ap.add_argument("--size", choices=sorted(SIZES), default="medium")
    ap.add_argument("--max-per-team", type=int, default=6)
    ap.add_argument("--zombies", type=int, default=4)
    ap.add_argument("--vents", action="store_true",
                    help="place one HEPA duct with a supply + return vent")
    ap.add_argument("--recipe", default=None,
                    help="an archetype recipe (levels/recipes/archetypes): "
                         "family B's grammar + layout instead of v1's split")
    ap.add_argument("--profile", default="standard",
                    help="the archetype's difficulty profile (with --recipe)")
    a = ap.parse_args(argv)
    if a.from_level is None and a.seed is None:
        ap.error("--seed is required (or --from-level FOLDER)")
    if a.from_level is not None:
        given = [f for f, v, d in (
            ("--seed", a.seed, None), ("--size", a.size, "medium"),
            ("--max-per-team", a.max_per_team, 6),
            ("--zombies", a.zombies, 4), ("--vents", a.vents, False),
            ("--recipe", a.recipe, None), ("--profile", a.profile,
                                           "standard")) if v != d]
        if given:
            ap.error(f"--from-level takes its inputs from the record; "
                     f"refusing {given}")
    elif a.allow_stale:
        ap.error("--allow-stale goes with --from-level")
    return a


def main(argv=None) -> int:
    a = _parse_args(sys.argv[1:] if argv is None else argv)
    if a.from_level is not None:
        plan = generate_from_spec(Path(a.from_level),
                                  allow_stale=a.allow_stale)
    else:
        plan = generate(a.seed, a.size, max_per_team=a.max_per_team,
                        zombies=a.zombies, vents=a.vents, recipe=a.recipe,
                        profile=a.profile)
    out = Path(a.out) if a.out else ROOT / "levels" / plan.name
    write(plan, out)
    h, w = plan.tilemap.shape
    e = plan.embedding
    vents = any(x.class_name == "duct" for x in plan.entities)
    print(f"wrote {out}  ({w}x{h} @ {TILE_SIZE_M} m, "
          + (f"grammar attempt {plan.g}, " if plan.recipe else "")
          + f"attempt {plan.attempt}, "
          f"{len(plan.graph.rooms)} rooms, {e.doors} doors, {e.windows} "
          f"windows, {e.fuel_tiles} fuel tiles, "
          f"vents {'yes' if vents else 'no'})")
    for r in plan.spawn_rooms:
        print(f"  spawn room team {r.team}: ({r.x},{r.y}) {r.w}x{r.h} "
              f"max {r.max_per_team}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
