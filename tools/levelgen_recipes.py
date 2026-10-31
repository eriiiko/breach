r"""tools/levelgen_recipes.py -- the RECIPE LIBRARY loader and the RESOLVER
(#136; family B of the levelgen grammar,
docs/levelgen_cycle_types_and_recipes_2026-10-31.md, revision 2).

A ship's pieces are DATA under ``levels/recipes/``:

- ``cycles/<name>.toml``       a cycle type (scale 2, the spec's §2): a move,
  the phases it may be applied in, legs (walk / rooms / width / light / edge
  kinds) or a duct;
- ``departments/<name>.toml``  a department (scale 3, §3): rooms with
  multiplicity and size ranges, edges by kind, ducts, a secret, optional
  internal ``[[cycle]]`` rows, physics by preset name;
- ``fixtures.toml``            the fixtures (scale 1.5, §3.3): a room's physics
  signature as preset NAMES.

THE LOADER (``load_cycle``, ``load_fixtures``, ``load_department``,
``load_recipes``) turns them into frozen dataclasses mirroring §2 / §3.2 and
is THE DOOR: it refuses, naming the file and the field, an unknown field, a
physics NUMBER where a preset NAME belongs (a recipe never carries one:
Erik tunes physics in config.toml), a named preset / config row config.toml
lacks, an edge kind not in ``gen_ship.EDGE_KINDS``, a width class not in
{wide, narrow}, a vent role set a sealed room cannot carry (vocabulary
§0.4, per duct), a room ``count`` / ``p`` out of range, a ``[[cycle]]`` naming
an unknown cycle type, a ``fixture`` not in fixtures.toml, a 3-cycle of
contacts (§1.5), a department whose contact demand cannot fit its hub
(§3.1 feasibility), a stove that could not light its own kindling (the
galley FUSE ruling, §4) -- all at load, before any ship is planned.

THE RESOLVER (``resolve_room``, ``resolve_department``; §3.4) is the ONE door
from recipe names to gen_ship's ``ROOM_TAGS`` values, reading the preset
tables (``[levelgen.lights.*]``, ``[levelgen.fuel.*]``, ``[temps.*]``,
``[pressures.*]``, ``[heat_sources.*]``, ``[hazards.*]``). Every tag it writes
passes gen_ship's own ``ROOM_TAGS`` check. A key whose stage-2 mechanism is
not built yet (§5.2: a kindling fuel class, a vent list, a disabled heat
source, a pressure preset, water) is ACCEPTED and kept as a PENDING entry
naming the missing extension -- never dropped; ``ResolvedRoom.stamp_tags``
refuses it by name.

Not here: expanding a department into room INSTANCES (the grammar's phase 1,
§5.3) and placing them (the layout stage 1.5, §5.1, undesigned) -- both are
later patches that call the resolver.

Home: ``tools/``, beside ``gen_ship.py``, its only consumer: the recipes
are a stage-1 INPUT of the offline generator (never read by the sim, the
level loader or the renderer), and the door's vocabulary (``EDGE_KINDS``,
``ROOM_TAGS``) is gen_ship's. IMPORT-LIGHT: stdlib only at import time; the
door imports ``gen_ship`` and ``config`` lazily when it checks.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = ROOT / "levels" / "recipes"

MOVES = ("subdivide", "edge_to_cycle", "leaf", "ring", "parallel", "duct")
UNUSED_MOVES = {"face": "L4 move 4 is unused in v1: the grammar keeps no "
                        "combinatorial embedding and the `ring` move covers "
                        "the one wanted case (§0.1)"}
WIDTH_CLASSES = ("wide", "narrow")
SHAPES = ("room", "corridor", "vestibule")
PLACEMENTS = ("wing", "axis")
CELL_MODES = ("main", "own")
MAIN_CELL = 0                  # §3.4: main = 0, each own department the next int
TOWARD = ("spawn", "axis")
WATER = ("full", "pool")
VENT_ROLES = ("supply", "return")
# How each gas-passing kind passes gas (a walk = false leg must use one).
GAS_PASSING = {"vent_only": "through a duct",
               "breakable": "once the wall is broken",
               "window": "once the glass bursts"}
AMBIENT = "ambient"
PHASES = (1, 2, 3, 4)
DEPT_CYCLE_PHASE = 1           # a department's own [[cycle]] rows: phase 1 (§3.2)
MIN_WIDE = 6                   # §1.2: every room / corridor floor >= 6 across
MIN_NARROW = 3                 # §1.2: the flank pair's service leg
MAX_COUNT = 12                 # a room's instance count: a bound on authoring slips
ENTRY_RUN = MIN_WIDE           # §3.1: the hub's perimeter less its entry
# The eight departments vocabulary Part B blessed (2026-10-29). A placement
# WISH may name one of these before its recipe file exists (amendment
# 2026-10-31, as built: crew quarters wishes for the medbay and engineering,
# which are not written yet); any other wish must name a library department.
BLESSED_DEPARTMENTS = ("crew_quarters", "engineering", "medbay", "cargo_hold",
                       "armory", "garden_dome", "pool_hall", "doorless_room")
# gen_ship._place_fuel's draw today (60/40 furniture / wood): the one fuel
# mix the ``fuel = n`` tag can express until the material extension (§5.2).
TODAYS_FUEL_MIX = {"furniture": 0.6, "wood": 0.4}

# The §5.2 stage-2 extensions a pending key waits for.
EXT_FUEL = "the fuel tag's material / class extension (§5.2)"
EXT_VENT_LIST = "the vent tag as a list of {duct, roles} (§5.2)"
EXT_HEAT_ENABLED = "the heat_source tag's optional `enabled` (§5.2)"
EXT_PRESSURE = "a per-room pressure composed into air_init.npy (§5.2)"
EXT_WATER = "a per-room water stamp into water_init.npy (§5.2)"


class RecipeError(ValueError):
    """A recipe the door refuses; the message names the file and the field."""


class PendingMechanism(RecipeError):
    """A resolved key whose stage-2 mechanism is not built yet (§5.2),
    refused at stamp time by name -- never silently dropped."""


# ===========================================================================
# The dataclasses (mirror §2 / §3.2 / §3.3 field for field)
# ===========================================================================

@dataclass(frozen=True)
class Leg:
    walk: bool
    edges: tuple
    rooms: tuple | None = None     # (lo, hi) rooms on the leg
    width: str | None = None
    light: str | None = None


@dataclass(frozen=True)
class CycleDuct:
    roles: tuple                   # every room of the shape carries these
    filter: str


@dataclass(frozen=True)
class Cycle:
    name: str
    move: str
    phase: tuple
    tactic: str
    rooms: tuple | None = None     # the ring's room count (lo, hi), even
    crosses_cell: bool = False
    legs: tuple = ()
    ducts: tuple = ()
    path: str = ""


@dataclass(frozen=True)
class Size:
    """A room's ``{w, h}`` (x, y extents) or a corridor's ``{across,
    along}`` (the layout stage rotates), each an inclusive (min, max)."""
    a: tuple
    b: tuple
    axes: tuple = ("w", "h")       # or ("across", "along")

    @property
    def min_side(self) -> int:
        return min(self.a[0], self.b[0])

    @property
    def max_perimeter(self) -> int:
        return 2 * (self.a[1] + self.b[1])


@dataclass(frozen=True)
class RoomVent:
    duct: str
    roles: tuple                   # in VENT_ROLES order


@dataclass(frozen=True)
class RecipeRoom:
    id: str
    size: Size
    shape: str = "room"
    required: bool = False
    count: tuple | None = None     # (lo, hi) instances
    p: float | None = None         # an optional room's probability
    width: str | None = None
    fixture: str | None = None
    fuel: str | None = None
    light: str | None = None
    vents: tuple = ()              # RoomVent per duct (a list is §5.2)
    water: str | None = None

    @property
    def max_count(self) -> int:
        return self.count[1] if self.count else 1


@dataclass(frozen=True)
class RecipeEdge:
    a: str
    b: str
    kind: str


@dataclass(frozen=True)
class RecipeDuct:
    name: str
    filter: str


@dataclass(frozen=True)
class DeptCycle:
    type: str
    a: str
    b: str


@dataclass(frozen=True)
class Secret:
    kind: str
    host: str
    weight: float


@dataclass(frozen=True)
class Physics:
    temp: str = AMBIENT
    pressure: str = AMBIENT


@dataclass(frozen=True)
class Department:
    name: str
    placement: str
    cell: str
    entry: str
    toward: str
    rooms: tuple
    edges: tuple
    wish_near: tuple = ()
    wish_far: tuple = ()
    cycles: tuple = ()             # DeptCycle rows
    ducts: tuple = ()
    secret: Secret | None = None
    physics: Physics = Physics()
    path: str = ""

    def room(self, rid: str) -> RecipeRoom:
        for r in self.rooms:
            if r.id == rid:
                return r
        raise KeyError(rid)


@dataclass(frozen=True)
class Fixture:
    name: str
    heat_source: str | None = None
    heat_source_enabled: bool = True
    fuel: str | None = None
    hazard: str | None = None
    water: str | None = None
    edges_out: tuple = ()


@dataclass(frozen=True)
class RecipeLibrary:
    root: str
    cycles: dict
    departments: dict
    fixtures: dict


# ===========================================================================
# Door helpers
# ===========================================================================

def _err(path, where: str, why: str) -> RecipeError:
    return RecipeError(f"{path}: {where}: {why}")


def _read(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise RecipeError(f"{path}: no such recipe file") from None
    except tomllib.TOMLDecodeError as e:
        raise RecipeError(f"{path}: not valid TOML ({e})") from None


def _keys(path, where: str, table, allowed, required=()) -> None:
    if not isinstance(table, dict):
        raise _err(path, where, f"is {type(table).__name__}, not a table")
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise _err(path, where, f"unknown field(s) {unknown} (known: "
                                f"{sorted(allowed)})")
    for k in required:
        if k not in table:
            raise _err(path, f"{where}.{k}", "is required")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str(path, where: str, v, *, name_of: str | None = None) -> str:
    """A string field. ``name_of``: the field NAMES a preset / config row
    -- a number there is the smuggled physics value §3 forbids."""
    if _is_num(v) and name_of:
        raise _err(path, where, f"is the number {v!r}: a recipe never carries "
                                f"a physics number -- name {name_of} instead")
    if not isinstance(v, str) or v == "":
        raise _err(path, where, f"{v!r} is not a non-empty string"
                                + (f" naming {name_of}" if name_of else ""))
    return v


def _ident(path, where: str, v) -> str:
    v = _str(path, where, v)
    if not all(ch.isalnum() or ch == "_" for ch in v):
        raise _err(path, where, f"{v!r} is not an id (letters, digits, _)")
    return v


def _one_of(path, where: str, v, options) -> str:
    v = _str(path, where, v)
    if v not in options:
        raise _err(path, where, f"{v!r} is not one of {list(options)}")
    return v


def _bool(path, where: str, v) -> bool:
    if not isinstance(v, bool):
        raise _err(path, where, f"{v!r} is not true / false")
    return v


def _range(path, where: str, v, lo_min: int, hi_max: int | None = None
           ) -> tuple:
    if not (isinstance(v, list) and len(v) == 2 and all(_is_int(x) for x in v)):
        raise _err(path, where, f"{v!r} is not an inclusive [min, max] pair "
                                f"of integers")
    lo, hi = v
    if lo < lo_min or hi < lo or (hi_max is not None and hi > hi_max):
        raise _err(path, where, f"{v!r} is out of range (need {lo_min} <= min "
                                f"<= max" + (f" <= {hi_max}" if hi_max
                                             is not None else "") + ")")
    return (lo, hi)


def _prob(path, where: str, v) -> float:
    if not _is_num(v) or not (0.0 < float(v) <= 1.0):
        raise _err(path, where, f"{v!r} is not a probability in (0, 1]")
    return float(v)


def _str_list(path, where: str, v) -> tuple:
    if not isinstance(v, list):
        raise _err(path, where, f"{v!r} is not a list")
    return tuple(_str(path, f"{where}[{i}]", x) for i, x in enumerate(v))


def _edge_kinds():
    from gen_ship import EDGE_KINDS
    return EDGE_KINDS


def _edge_kind(path, where: str, v) -> str:
    kinds = _edge_kinds()
    v = _str(path, where, v)
    if v not in kinds:
        raise _err(path, where, f"edge kind {v!r} is not a row of "
                                f"gen_ship.EDGE_KINDS ({list(kinds)})")
    return v


def _cfg():
    from config import CFG
    return CFG


def _table(dotted: str):
    ns = _cfg()
    for part in dotted.split("."):
        ns = getattr(ns, part, None)
        if ns is None:
            return None
    return ns


def _config_row(path, where: str, v, table: str):
    """``v`` names a ``[<table>.<v>]`` row of config.toml: the row, or a
    refusal naming the missing row."""
    v = _str(path, where, v, name_of=f"a [{table}.*] preset")
    ns = _table(table)
    row = getattr(ns, v, None) if ns is not None else None
    if row is None:
        raise _err(path, where, f"{v!r} has no [{table}.{v}] row in "
                                f"config.toml" + ("" if ns is not None else
                                                  f" (there is no [{table}] "
                                                  f"table at all)"))
    return row


def _vent_roles(path, where: str, v) -> tuple:
    roles = _str_list(path, where, v)
    if not (1 <= len(roles) <= 2 and len(set(roles)) == len(roles)
            and all(r in VENT_ROLES for r in roles)):
        raise _err(path, where, f"{list(roles)} is not one or two distinct "
                                f"roles of {list(VENT_ROLES)}")
    return tuple(r for r in VENT_ROLES if r in roles)


# ===========================================================================
# Presets (config.toml; §3.4). Each returns the RESOLVED value or refuses.
# ===========================================================================

def light_preset(name: str, *, path="config.toml", where="light") -> tuple:
    """``[levelgen.lights.<name>]`` as gen_ship's ``light`` tag value."""
    row = _config_row(path, where, name, "levelgen.lights")
    d = vars(row)
    rgb, level = d.get("rgb"), d.get("level")
    if (set(d) != {"rgb", "level"} or not isinstance(rgb, list)
            or len(rgb) != 3 or not all(_is_num(c) and c >= 0 for c in rgb)
            or not (_is_num(level) and level >= 0)):
        raise _err("config.toml", f"[levelgen.lights.{name}]",
                   "is not exactly rgb = [r, g, b] (>= 0) and level >= 0")
    return (tuple(float(c) for c in rgb), float(level))


def fuel_preset(name: str, *, path="config.toml", where="fuel") -> dict:
    """``[levelgen.fuel.<name>]`` as ``{"clusters": (lo, hi), "mix":
    {material: fraction}}``; every material a ``[materials.*]`` row, the
    fractions summing to 1."""
    row = _config_row(path, where, name, "levelgen.fuel")
    d = vars(row)
    here = f"[levelgen.fuel.{name}]"
    if set(d) != {"clusters", "mix"}:
        raise _err("config.toml", here, f"fields {sorted(d)} are not "
                                        f"['clusters', 'mix']")
    clusters = _range("config.toml", f"{here}.clusters", d["clusters"], 1)
    mix = {k: v for k, v in vars(d["mix"]).items()}
    for mat, frac in mix.items():
        _config_row("config.toml", f"{here}.mix.{mat}", mat, "materials")
        if not (_is_num(frac) and frac > 0):
            raise _err("config.toml", f"{here}.mix.{mat}",
                       f"{frac!r} is not a fraction > 0")
    if not mix or abs(sum(mix.values()) - 1.0) > 1e-9:
        raise _err("config.toml", f"{here}.mix", f"{mix} does not sum to 1")
    return {"clusters": clusters, "mix": mix}


_HEAT_SOURCE_ROW = ("heat_per_s", "radius", "max_temp", "death_payload")


def heat_source_preset(name: str, *, path="config.toml",
                       where="heat_source") -> dict:
    """``[heat_sources.<name>]`` as gen_ship's ``heat_source`` tag fields."""
    row = _config_row(path, where, name, "heat_sources")
    d = dict(vars(row))
    here = f"[heat_sources.{name}]"
    if set(d) != set(_HEAT_SOURCE_ROW):
        raise _err("config.toml", here, f"fields {sorted(d)} are not "
                                        f"{sorted(_HEAT_SOURCE_ROW)}")
    if not (_is_num(d["heat_per_s"]) and d["heat_per_s"] > 0
            and _is_int(d["radius"]) and d["radius"] >= 0
            and _is_num(d["max_temp"]) and d["max_temp"] >= 0
            and isinstance(d["death_payload"], str)):
        raise _err("config.toml", here, f"{d} is not heat_per_s > 0, radius "
                                        f"int >= 0, max_temp >= 0, "
                                        f"death_payload a row name or \"\"")
    if d["death_payload"]:
        _config_row("config.toml", f"{here}.death_payload",
                    d["death_payload"], "payloads")
    return {"heat_per_s": float(d["heat_per_s"]), "radius": int(d["radius"]),
            "max_temp": float(d["max_temp"]),
            "death_payload": d["death_payload"]}


def hazard_preset(name: str, *, path="config.toml", where="hazard") -> dict:
    """``[hazards.<name>]`` as gen_ship's ``hazard`` tag value."""
    row = _config_row(path, where, name, "hazards")
    d = dict(vars(row))
    here = f"[hazards.{name}]"
    if set(d) != {"payload", "trigger_temp", "fuse_s"}:
        raise _err("config.toml", here, f"fields {sorted(d)} are not "
                                        f"['fuse_s', 'payload', "
                                        f"'trigger_temp']")
    _config_row("config.toml", f"{here}.payload", d["payload"], "payloads")
    if not (_is_num(d["trigger_temp"]) and d["trigger_temp"] > 0
            and _is_num(d["fuse_s"]) and d["fuse_s"] >= 0):
        raise _err("config.toml", here, "is not trigger_temp > 0, fuse_s >= 0")
    return {"payload": d["payload"], "trigger_temp": float(d["trigger_temp"]),
            "fuse_s": float(d["fuse_s"])}


def temp_preset(name: str, *, path="config.toml", where="temp") -> float:
    """``[temps.<name>]``'s ``above_ambient`` (game degrees > 0)."""
    row = _config_row(path, where, name, "temps")
    v = getattr(row, "above_ambient", None)
    if set(vars(row)) != {"above_ambient"} or not (_is_num(v) and v > 0):
        raise _err("config.toml", f"[temps.{name}]",
                   "is not exactly above_ambient > 0 (never sub-ambient)")
    return float(v)


def pressure_preset(name: str, *, path="config.toml",
                    where="pressure") -> float:
    """``[pressures.<name>]``'s ``atm`` (> 0)."""
    row = _config_row(path, where, name, "pressures")
    v = getattr(row, "atm", None)
    if set(vars(row)) != {"atm"} or not (_is_num(v) and v > 0):
        raise _err("config.toml", f"[pressures.{name}]",
                   "is not exactly atm > 0")
    return float(v)


def _ignition_temp(material: str) -> float:
    return float(getattr(_table(f"materials.{material}"), "ignition_temp"))


def _check_fuse(path, where: str, hs_name: str, fuel_name: str) -> None:
    """The galley FUSE (Erik, 2026-10-31, §4): a heat source beside a fuel
    class must be able to light it -- its ``max_temp`` cap (0 = none) sits
    ABOVE the easiest material's ``ignition_temp``."""
    hs = heat_source_preset(hs_name, path=path, where=where)
    mix = fuel_preset(fuel_name, path=path, where=where)["mix"]
    ign = min(_ignition_temp(m) for m in mix)
    if hs["max_temp"] and hs["max_temp"] <= ign:
        raise _err(path, where,
                   f"heat source {hs_name!r} caps at max_temp "
                   f"{hs['max_temp']} <= fuel {fuel_name!r}'s ignition_temp "
                   f"{ign}: the stove could never light its fuel, and the "
                   f"galley is a FUSE (Erik's ruling 2026-10-31) -- raise "
                   f"[heat_sources.{hs_name}].max_temp above it")


# ===========================================================================
# Cycle types
# ===========================================================================

_CYCLE_KEYS = ("name", "move", "phase", "rooms", "crosses_cell", "tactic")
_LEG_KEYS = ("walk", "rooms", "width", "light", "edges")
_CDUCT_KEYS = ("roles", "filter")


def load_cycle(path) -> Cycle:
    """One cycle type file (§2) as a :class:`Cycle`, or :class:`RecipeError`."""
    path = Path(path)
    raw = _read(path)
    _keys(path, "file", raw, ("cycle", "leg", "duct"), ("cycle",))
    c = raw["cycle"]
    _keys(path, "[cycle]", c, _CYCLE_KEYS, ("name", "move", "phase", "tactic"))
    name = _ident(path, "cycle.name", c["name"])
    if name != path.stem:
        raise _err(path, "cycle.name", f"{name!r} differs from the file name "
                                       f"{path.stem!r}")
    move = _str(path, "cycle.move", c["move"])
    if move in UNUSED_MOVES:
        raise _err(path, "cycle.move", f"{move!r}: {UNUSED_MOVES[move]}")
    _one_of(path, "cycle.move", move, MOVES)
    phase = c["phase"]
    if not (isinstance(phase, list) and phase
            and all(_is_int(x) and x in PHASES for x in phase)):
        raise _err(path, "cycle.phase", f"{phase!r} is not a non-empty list "
                                        f"of L6 phases in {list(PHASES)}")
    crosses = _bool(path, "cycle.crosses_cell", c.get("crosses_cell", False))
    ring_rooms = None
    if move == "ring":
        if "rooms" not in c:
            raise _err(path, "cycle.rooms", "a ring needs its room count")
        ring_rooms = _range(path, "cycle.rooms", c["rooms"], 4)
        if ring_rooms[0] % 2 or ring_rooms[1] % 2:
            raise _err(path, "cycle.rooms", f"{list(ring_rooms)}: a ring is "
                                            f"EVEN (the hub joins every "
                                            f"other ring room, §0.1)")
    elif "rooms" in c:
        raise _err(path, "cycle.rooms", "a room count on [cycle] is the "
                                        "ring's; the other moves count rooms "
                                        "per [[leg]]")
    legs = []
    for i, lg in enumerate(raw.get("leg", [])):
        legs.append(_leg(path, f"leg[{i}]", lg, crosses))
    if crosses and not all(set(lg.edges) == {"airlock"}
                           for lg in legs if lg.walk):
        raise _err(path, "cycle.crosses_cell", "only a type whose walk legs "
                                               "are all airlock pairs may "
                                               "cross a cell (L5)")
    ducts = []
    for i, d in enumerate(raw.get("duct", [])):
        w = f"duct[{i}]"
        _keys(path, w, d, _CDUCT_KEYS, ("roles", "filter"))
        ducts.append(CycleDuct(
            roles=_vent_roles(path, f"{w}.roles", d["roles"]),
            filter=_filter(path, f"{w}.filter", d["filter"])))
    if move == "duct":
        if legs or not ducts:
            raise _err(path, "cycle.move", "a 'duct' move has [[duct]] and "
                                           "no [[leg]]")
    elif not legs:
        raise _err(path, "file", f"a {move!r} type has at least one [[leg]]")
    return Cycle(name=name, move=move, phase=tuple(phase),
                 tactic=_str(path, "cycle.tactic", c["tactic"]),
                 rooms=ring_rooms, crosses_cell=crosses, legs=tuple(legs),
                 ducts=tuple(ducts), path=str(path))


def _leg(path, w: str, lg, crosses: bool) -> Leg:
    _keys(path, w, lg, _LEG_KEYS, ("walk", "edges"))
    walk = _bool(path, f"{w}.walk", lg["walk"])
    rooms = _range(path, f"{w}.rooms", lg["rooms"], 0) if "rooms" in lg \
        else None
    width = (_one_of(path, f"{w}.width", lg["width"], WIDTH_CLASSES)
             if "width" in lg else None)
    light = None
    if "light" in lg:
        light = _str(path, f"{w}.light", lg["light"],
                     name_of="a [levelgen.lights.*] preset")
        light_preset(light, path=path, where=f"{w}.light")
    edges = _str_list(path, f"{w}.edges", lg["edges"])
    if not edges:
        raise _err(path, f"{w}.edges", "is empty")
    for j, k in enumerate(edges):
        _edge_kind(path, f"{w}.edges[{j}]", k)
    if width == "narrow" and rooms != (1, 1):
        raise _err(path, f"{w}.width", "a narrow leg is exactly one room "
                                       "(rooms = [1, 1]), entered from its "
                                       "long sides (§1.2)")
    if not walk:
        bad = [k for k in edges if k not in GAS_PASSING]
        if bad:
            how = ", ".join(f"{k} ({v})" for k, v in GAS_PASSING.items())
            raise _err(path, f"{w}.edges", f"a walk = false leg uses a kind "
                                           f"that can pass gas -- {how} -- "
                                           f"not {bad}")
        if crosses and "vent_only" in edges:
            raise _err(path, f"{w}.edges", "a vent_only leg across a cell "
                                           "boundary waits for the damper "
                                           "entity (#49) and the archetype's "
                                           "allow_cross_cell_leak (L5)")
        if width is not None:
            raise _err(path, f"{w}.width", "a walk = false leg has no width "
                                           "class")
    return Leg(walk=walk, edges=edges, rooms=rooms, width=width, light=light)


def _filter(path, where: str, v) -> str:
    _config_row(path, where, v, "filters")
    return v


# ===========================================================================
# Fixtures
# ===========================================================================

_FIXTURE_KEYS = ("heat_source", "heat_source_enabled", "fuel", "hazard",
                 "water", "edges_out")


def load_fixtures(path) -> dict:
    """``fixtures.toml`` (§3.3) as ``{name: Fixture}``, or
    :class:`RecipeError`."""
    path = Path(path)
    raw = _read(path)
    out = {}
    for name, t in raw.items():
        w = f"[{name}]"
        _ident(path, w, name)
        _keys(path, w, t, _FIXTURE_KEYS)
        hs = None
        if "heat_source" in t:
            hs = _str(path, f"{w}.heat_source", t["heat_source"],
                      name_of="a [heat_sources.*] preset")
            heat_source_preset(hs, path=path, where=f"{w}.heat_source")
        if "heat_source_enabled" in t and hs is None:
            raise _err(path, f"{w}.heat_source_enabled",
                       "set without a heat_source")
        hz = None
        if "hazard" in t:
            hz = _str(path, f"{w}.hazard", t["hazard"],
                      name_of="a [hazards.*] preset")
            hazard_preset(hz, path=path, where=f"{w}.hazard")
        fuel = None
        if "fuel" in t:
            fuel = _str(path, f"{w}.fuel", t["fuel"],
                        name_of="a [levelgen.fuel.*] class")
            fuel_preset(fuel, path=path, where=f"{w}.fuel")
        if hs and fuel:
            _check_fuse(path, f"{w}.heat_source", hs, fuel)
        edges_out = _str_list(path, f"{w}.edges_out", t.get("edges_out", []))
        for j, k in enumerate(edges_out):
            _edge_kind(path, f"{w}.edges_out[{j}]", k)
        out[name] = Fixture(
            name=name, heat_source=hs,
            heat_source_enabled=_bool(path, f"{w}.heat_source_enabled",
                                      t.get("heat_source_enabled", True)),
            fuel=fuel, hazard=hz,
            water=(_one_of(path, f"{w}.water", t["water"], WATER)
                   if "water" in t else None),
            edges_out=edges_out)
    return out


# ===========================================================================
# Departments
# ===========================================================================

_DEPT_KEYS = ("name", "placement", "cell", "entry", "toward", "wish_near",
              "wish_far")
_ROOM_KEYS = ("id", "shape", "required", "count", "p", "width", "size",
              "fixture", "fuel", "light", "vent", "water")
_EDGE_KEYS = ("a", "b", "kind")
_DUCT_KEYS = ("name", "filter")
_DCYCLE_KEYS = ("type", "a", "b")
_SECRET_KEYS = ("kind", "host", "weight")
_PHYSICS_KEYS = ("temp", "pressure")


def _names(root: Path, sub: str) -> set:
    return {p.stem for p in (root / sub).glob("*.toml")}


def _root_of(path: Path) -> Path:
    """The recipe root a department / cycle file lives under."""
    return path.resolve().parent.parent


def load_department(path, *, root=None, fixtures: dict | None = None,
                    cycles: dict | None = None) -> Department:
    """One department file (§3) as a :class:`Department`, or
    :class:`RecipeError`. Cross-references (cycle types, fixtures, wished
    departments) resolve against the recipe ``root`` (default: the folder
    above the file's ``departments/``)."""
    path = Path(path)
    root = Path(root) if root is not None else _root_of(path)
    if fixtures is None:
        fixtures = load_fixtures(root / "fixtures.toml")
    raw = _read(path)
    _keys(path, "file", raw, ("department", "room", "edge", "duct", "cycle",
                              "secret", "physics"), ("department", "room"))
    d = raw["department"]
    _keys(path, "[department]", d, _DEPT_KEYS,
          ("name", "placement", "cell", "entry", "toward"))
    name = _ident(path, "department.name", d["name"])
    if name != path.stem:
        raise _err(path, "department.name", f"{name!r} differs from the file "
                                            f"name {path.stem!r}")
    library = _names(root, "departments")
    wishes = {}
    for key in ("wish_near", "wish_far"):
        wishes[key] = _str_list(path, f"department.{key}", d.get(key, []))
        for j, w in enumerate(wishes[key]):
            if w not in library and w not in BLESSED_DEPARTMENTS:
                raise _err(path, f"department.{key}[{j}]",
                           f"{w!r} is no department in the library "
                           f"({sorted(library)}) nor a blessed one "
                           f"({list(BLESSED_DEPARTMENTS)})")

    ducts = []
    for i, t in enumerate(raw.get("duct", [])):
        w = f"duct[{i}]"
        _keys(path, w, t, _DUCT_KEYS, ("name", "filter"))
        dn = _ident(path, f"{w}.name", t["name"])
        if any(x.name == dn for x in ducts):
            raise _err(path, f"{w}.name", f"duct {dn!r} declared twice")
        ducts.append(RecipeDuct(name=dn, filter=_filter(path, f"{w}.filter",
                                                        t["filter"])))
    duct_names = {x.name for x in ducts}

    rooms = []
    for i, t in enumerate(raw["room"]):
        rm = _room(path, f"room[{i}]", t, fixtures, duct_names)
        if any(r.id == rm.id for r in rooms):
            raise _err(path, f"room[{i}].id", f"{rm.id!r} declared twice")
        rooms.append(rm)
    ids = {r.id for r in rooms}

    edges = []
    for i, t in enumerate(raw.get("edge", [])):
        w = f"edge[{i}]"
        _keys(path, w, t, _EDGE_KEYS, ("a", "b", "kind"))
        a, b = _room_ref(path, f"{w}.a", t["a"], ids), \
            _room_ref(path, f"{w}.b", t["b"], ids)
        if a == b:
            raise _err(path, w, f"joins room {a!r} to itself")
        if any({e.a, e.b} == {a, b} for e in edges):
            raise _err(path, w, f"rooms {a!r} and {b!r} are joined twice (a "
                                f"second opening on one wall is the "
                                f"`parallel` move's)")
        edges.append(RecipeEdge(a=a, b=b, kind=_edge_kind(path, f"{w}.kind",
                                                          t["kind"])))

    if cycles is None:
        cycles = {}
    dcycles = []
    known_cycles = _names(root, "cycles")
    for i, t in enumerate(raw.get("cycle", [])):
        w = f"cycle[{i}]"
        _keys(path, w, t, _DCYCLE_KEYS, ("type", "a", "b"))
        ty = _str(path, f"{w}.type", t["type"])
        if ty not in known_cycles:
            raise _err(path, f"{w}.type", f"{ty!r} is no cycle type in "
                                          f"{root / 'cycles'} "
                                          f"({sorted(known_cycles)})")
        cyc = cycles.get(ty) or load_cycle(root / "cycles" / f"{ty}.toml")
        if DEPT_CYCLE_PHASE not in cyc.phase:
            raise _err(path, f"{w}.type", f"cycle type {ty!r} may be applied "
                                          f"in phases {list(cyc.phase)}, and "
                                          f"a department's own cycles run in "
                                          f"phase {DEPT_CYCLE_PHASE} (§3.2)")
        a, b = _room_ref(path, f"{w}.a", t["a"], ids), \
            _room_ref(path, f"{w}.b", t["b"], ids)
        if a == b and cyc.move not in ("leaf", "ring"):
            raise _err(path, w, f"a {cyc.move!r} cycle joins two distinct "
                                f"rooms")
        dcycles.append(DeptCycle(type=ty, a=a, b=b))

    secret = None
    if "secret" in raw:
        t = raw["secret"]
        _keys(path, "[secret]", t, _SECRET_KEYS, ("kind", "host", "weight"))
        kind = _edge_kind(path, "secret.kind", t["kind"])
        paths = [k for k, v in _edge_kinds().items() if v.secret_path]
        if kind not in paths:
            raise _err(path, "secret.kind", f"{kind!r} is not a secret-path "
                                            f"edge kind ({paths})")
        host = _room_ref(path, "secret.host", t["host"], ids)
        weight = t["weight"]
        if not (_is_num(weight) and weight > 0):
            raise _err(path, "secret.weight", f"{weight!r} is not a draw "
                                              f"weight > 0")
        if kind == "vent_only" and not next(r for r in rooms
                                            if r.id == host).vents:
            raise _err(path, "secret.host", f"a vent_only secret rides its "
                                            f"host's duct, and room {host!r} "
                                            f"has no vent")
        secret = Secret(kind=kind, host=host, weight=float(weight))

    physics = Physics()
    if "physics" in raw:
        t = raw["physics"]
        _keys(path, "[physics]", t, _PHYSICS_KEYS)
        temp = _str(path, "physics.temp", t.get("temp", AMBIENT),
                    name_of='a [temps.*] preset or "ambient"')
        if temp != AMBIENT:
            temp_preset(temp, path=path, where="physics.temp")
        pres = _str(path, "physics.pressure", t.get("pressure", AMBIENT),
                    name_of='a [pressures.*] preset or "ambient"')
        if pres != AMBIENT:
            pressure_preset(pres, path=path, where="physics.pressure")
        physics = Physics(temp=temp, pressure=pres)

    entry = _room_ref(path, "department.entry", d["entry"], ids)
    if not next(r for r in rooms if r.id == entry).required:
        raise _err(path, "department.entry", f"room {entry!r} is not "
                                             f"required (the entry always "
                                             f"exists)")
    dept = Department(
        name=name,
        placement=_one_of(path, "department.placement", d["placement"],
                          PLACEMENTS),
        cell=_one_of(path, "department.cell", d["cell"], CELL_MODES),
        entry=entry,
        toward=_one_of(path, "department.toward", d["toward"], TOWARD),
        rooms=tuple(rooms), edges=tuple(edges),
        wish_near=wishes["wish_near"], wish_far=wishes["wish_far"],
        cycles=tuple(dcycles), ducts=tuple(ducts), secret=secret,
        physics=physics, path=str(path))
    _check_structure(path, dept, fixtures)
    return dept


def _room_ref(path, where: str, v, ids: set) -> str:
    v = _str(path, where, v)
    if v not in ids:
        raise _err(path, where, f"{v!r} is no room of this department "
                                f"({sorted(ids)})")
    return v


def _room(path, w: str, t, fixtures: dict, duct_names: set) -> RecipeRoom:
    _keys(path, w, t, _ROOM_KEYS, ("id", "size"))
    rid = _ident(path, f"{w}.id", t["id"])
    shape = _one_of(path, f"{w}.shape", t.get("shape", "room"), SHAPES)
    mult = [k for k in ("required", "count", "p") if k in t]
    if len(mult) != 1:
        raise _err(path, w, f"room {rid!r} sets {mult or 'none'} of "
                            f"required / count / p: exactly one")
    required, count, p = False, None, None
    if "required" in t:
        if t["required"] is not True:
            raise _err(path, f"{w}.required", "is only ever true (an optional "
                                              "room sets p, a repeated one "
                                              "count)")
        required = True
    elif "count" in t:
        count = _range(path, f"{w}.count", t["count"], 1, MAX_COUNT)
    else:
        p = _prob(path, f"{w}.p", t["p"])
    width = None
    if "width" in t:
        width = _one_of(path, f"{w}.width", t["width"], WIDTH_CLASSES)
        if shape != "corridor":
            raise _err(path, f"{w}.width", f"a width class is for corridors "
                                           f"only (room {rid!r} is a "
                                           f"{shape})")
    s = t["size"]
    if shape == "corridor":
        _keys(path, f"{w}.size", s, ("across", "along"), ("across", "along"))
        if width == "narrow":
            across = _range(path, f"{w}.size.across", s["across"],
                            MIN_NARROW, MIN_WIDE - 1)
        else:
            across = _range(path, f"{w}.size.across", s["across"], MIN_WIDE)
        size = Size(a=across, b=_range(path, f"{w}.size.along", s["along"],
                                       MIN_WIDE), axes=("across", "along"))
    else:
        _keys(path, f"{w}.size", s, ("w", "h"), ("w", "h"))
        size = Size(a=_range(path, f"{w}.size.w", s["w"], MIN_WIDE),
                    b=_range(path, f"{w}.size.h", s["h"], MIN_WIDE))
    fixture = None
    if "fixture" in t:
        fixture = _str(path, f"{w}.fixture", t["fixture"],
                       name_of="a fixture")
        if fixture not in fixtures:
            raise _err(path, f"{w}.fixture", f"{fixture!r} is no fixture in "
                                             f"fixtures.toml "
                                             f"({sorted(fixtures)})")
    fuel = None
    if "fuel" in t:
        fuel = _str(path, f"{w}.fuel", t["fuel"],
                    name_of="a [levelgen.fuel.*] class")
        fuel_preset(fuel, path=path, where=f"{w}.fuel")
    eff_fuel = fuel or (fixtures[fixture].fuel if fixture else None)
    if width == "narrow" and eff_fuel:
        raise _err(path, f"{w}.fuel", f"a narrow corridor never carries fuel "
                                      f"(a cluster against its wall would "
                                      f"block it, §1.2)")
    if fixture and fixtures[fixture].heat_source and fuel:
        _check_fuse(path, f"{w}.fuel", fixtures[fixture].heat_source, fuel)
    light = None
    if "light" in t:
        light = _str(path, f"{w}.light", t["light"],
                     name_of="a [levelgen.lights.*] preset")
        light_preset(light, path=path, where=f"{w}.light")
    vents = []
    if "vent" in t:
        raw_v = t["vent"] if isinstance(t["vent"], list) else [t["vent"]]
        for j, v in enumerate(raw_v):
            vw = f"{w}.vent" + (f"[{j}]" if isinstance(t["vent"], list)
                                else "")
            _keys(path, vw, v, ("duct", "roles"), ("duct", "roles"))
            dn = _str(path, f"{vw}.duct", v["duct"])
            if dn not in duct_names:
                raise _err(path, f"{vw}.duct", f"duct {dn!r} is not declared "
                                               f"([[duct]]: "
                                               f"{sorted(duct_names)})")
            if any(x.duct == dn for x in vents):
                raise _err(path, f"{vw}.duct", f"room {rid!r} names duct "
                                               f"{dn!r} twice")
            vents.append(RoomVent(duct=dn, roles=_vent_roles(
                path, f"{vw}.roles", v["roles"])))
    return RecipeRoom(
        id=rid, size=size, shape=shape, required=required, count=count, p=p,
        width=width, fixture=fixture, fuel=fuel, light=light,
        vents=tuple(vents),
        water=(_one_of(path, f"{w}.water", t["water"], WATER)
               if "water" in t else None))


def contact_triangle(edges) -> tuple | None:
    """Three rooms joined pairwise by edges of ANY kind -- a 3-cycle of the
    CONTACT graph (§1.5: every EDGE_KINDS row is a shared wall; triangle-free
    keeps a rectangular layout possible) -- or None. ``edges``: (a, b)."""
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for a in sorted(adj, key=str):
        for b in sorted(adj[a], key=str):
            for c in sorted(adj[a] & adj[b], key=str):
                if len({a, b, c}) == 3:
                    return tuple(sorted((a, b, c), key=str))
    return None


def _check_structure(path, dept: Department, fixtures: dict) -> None:
    """Whole-recipe rules: every room but the entry has an edge; no 3-cycle
    of contacts; a fixture's ``edges_out`` holds; the sealed-room vent rule
    per duct (a room every edge of which bounds a compartment -- the entry
    counts its door to the ship, the secret host its secret edge -- carries
    both roles on each of its ducts, or none); the §3.1 feasibility bound."""
    kinds = _edge_kinds()
    tri = contact_triangle((e.a, e.b) for e in dept.edges)
    if tri:
        raise _err(path, "edge", f"rooms {list(tri)} form a 3-cycle of "
                                 f"contacts (§1.5: no triangle in the contact "
                                 f"graph, whatever the edge kinds)")
    for r in dept.rooms:
        mine = [e.kind for e in dept.edges if r.id in (e.a, e.b)]
        fx = fixtures.get(r.fixture) if r.fixture else None
        if fx and fx.edges_out:
            bad = sorted(set(mine) - set(fx.edges_out))
            if bad:
                raise _err(path, f"room {r.id!r}", f"fixture {fx.name!r} "
                                                   f"meets the ship only "
                                                   f"through "
                                                   f"{list(fx.edges_out)}, "
                                                   f"not {bad}")
        if dept.secret and dept.secret.host == r.id:
            mine.append(dept.secret.kind)
        if r.id == dept.entry:
            mine.append("door")                  # the entry's door to the ship
        if not mine:
            raise _err(path, f"room {r.id!r}", "has no edge: nothing reaches "
                                               "it")
        if all(kinds[k].bounds for k in mine):
            for v in r.vents:
                if len(v.roles) == 1:
                    raise _err(path, f"room {r.id!r}.vent",
                               f"the room is sealed (every edge, {mine}, "
                               f"bounds a compartment) and carries only a "
                               f"{v.roles[0]!r} vent on duct {v.duct!r}: a "
                               f"sealed room carries both a supply and a "
                               f"return on each duct, or neither "
                               f"(vocabulary §0.4)")
    _check_feasible(path, dept)


def _check_feasible(path, dept: Department) -> None:
    """§3.1: every edge is a contact, so each neighbour (at its MAXIMUM
    count) needs a run of the room's perimeter of at least its least floor
    side plus a wall; the sum may not exceed the room's MAXIMUM perimeter,
    less one entry run for the department's entry."""
    for r in dept.rooms:
        demand, parts = 0, []
        for e in dept.edges:
            if r.id not in (e.a, e.b):
                continue
            n = dept.room(e.b if e.a == r.id else e.a)
            need = n.max_count * (n.size.min_side + 1)
            demand += need
            parts.append(f"{n.id} {n.max_count}x{n.size.min_side + 1}")
        cap = r.size.max_perimeter - (ENTRY_RUN if r.id == dept.entry else 0)
        if demand > cap:
            raise _err(path, f"room {r.id!r}",
                       f"contact demand {demand} tiles ({', '.join(parts)}) "
                       f"exceeds its maximum perimeter {cap}"
                       + (f" (less its {ENTRY_RUN}-tile entry)"
                          if r.id == dept.entry else "")
                       + " -- no layout can give every edge a wall (§3.1)")


# ===========================================================================
# The library
# ===========================================================================

def load_recipes(root=None) -> RecipeLibrary:
    """Every recipe under ``root`` (default ``levels/recipes``): the cycle
    types, the fixtures and the departments, each through its door."""
    root = Path(root) if root is not None else RECIPES_DIR
    cycles = {p.stem: load_cycle(p)
              for p in sorted((root / "cycles").glob("*.toml"))}
    fixtures = load_fixtures(root / "fixtures.toml")
    depts = {p.stem: load_department(p, root=root, fixtures=fixtures,
                                     cycles=cycles)
             for p in sorted((root / "departments").glob("*.toml"))}
    return RecipeLibrary(root=str(root), cycles=cycles, departments=depts,
                         fixtures=fixtures)


def department(name: str, root=None) -> Department:
    """The department ``name`` from the library, or a refusal naming it and
    what exists."""
    root = Path(root) if root is not None else RECIPES_DIR
    p = root / "departments" / f"{name}.toml"
    if not p.is_file():
        raise RecipeError(f"{root / 'departments'}: no department recipe "
                          f"{name!r} (known: "
                          f"{sorted(_names(root, 'departments'))})")
    return load_department(p, root=root)


# ===========================================================================
# THE RESOLVER (§3.4): recipe names -> ROOM_TAGS values
# ===========================================================================

@dataclass(frozen=True)
class ResolvedRoom:
    """One recipe room's resolved tags (gen_ship ``ROOM_TAGS`` keys and
    values, each already passing ROOM_TAGS' check) plus ``pending``: the
    keys the recipe set whose stage-2 mechanism is not built yet, key ->
    (the resolved value, the extension it waits for). ``stamp_tags`` is
    what a stage 2 may stamp -- it refuses while anything is pending."""
    id: str
    tags: dict
    pending: dict = field(default_factory=dict)

    def stamp_tags(self) -> dict:
        if self.pending:
            k = sorted(self.pending)[0]
            raise PendingMechanism(
                f"room {self.id!r}: {k!r} = {self.pending[k][0]!r} cannot be "
                f"stamped yet -- it waits for {self.pending[k][1]} (pending: "
                f"{sorted(self.pending)})")
        return dict(self.tags)


@dataclass(frozen=True)
class ResolvedDepartment:
    """A department with every name resolved: ``rooms`` (recipe room id ->
    :class:`ResolvedRoom`), ``ducts`` (``ShipGraph.ducts``'s shape), and
    ``digest``: a sha256 over the recipe files AND the resolved values, so
    retuning a preset changes it as it changes the ship (§3.4)."""
    name: str
    rooms: dict
    ducts: dict
    digest: str


def _effective(room: RecipeRoom, fixtures: dict) -> dict:
    """The room's physics keys: its fixture's fields, overridden by any the
    room sets itself (§3.3)."""
    out = {}
    fx = fixtures.get(room.fixture) if room.fixture else None
    if fx:
        for k in ("heat_source", "fuel", "hazard", "water"):
            if getattr(fx, k) is not None:
                out[k] = getattr(fx, k)
        if fx.heat_source:
            out["heat_source_enabled"] = fx.heat_source_enabled
    for k in ("fuel", "water"):
        if getattr(room, k) is not None:
            out[k] = getattr(room, k)
    return out


def resolve_room(dept: Department, room_id: str, fixtures: dict, rng, *,
                 cell: int | None = None) -> ResolvedRoom:
    """The ONE door from a recipe room's names to ``ROOM_TAGS`` values.
    ``rng`` (a numpy Generator) draws what a preset leaves as a range (a fuel
    class's cluster count); ``cell``: the number the grammar gave an ``own``
    department (``main`` is :data:`MAIN_CELL`). The duct names are the
    recipe's: renaming per wing instance is the grammar's."""
    from gen_ship import ROOM_TAGS
    path = dept.path
    room = dept.room(room_id)
    w = f"room {room_id!r}"
    tags: dict = {"dept": dept.name, "role": room.id}
    pending: dict = {}
    if dept.cell == "main":
        tags["cell"] = MAIN_CELL
    elif cell is None or not isinstance(cell, int) or cell == MAIN_CELL:
        raise _err(path, "department.cell", f"an 'own' department's cell "
                                            f"number is the grammar's (§3.4); "
                                            f"got {cell!r}")
    else:
        tags["cell"] = cell
    if room.light:
        tags["light"] = light_preset(room.light, path=path,
                                     where=f"{w}.light")
    eff = _effective(room, fixtures)
    if "fuel" in eff:
        fp = fuel_preset(eff["fuel"], path=path, where=f"{w}.fuel")
        lo, hi = fp["clusters"]
        n = int(rng.integers(lo, hi + 1))
        if fp["mix"] == TODAYS_FUEL_MIX:
            tags["fuel"] = n
        else:
            pending["fuel"] = ({"clusters": n, "mix": fp["mix"]}, EXT_FUEL)
    if room.vents:
        vt = [{"duct": v.duct, "roles": list(v.roles)} for v in room.vents]
        if len(vt) == 1:
            tags["vent"] = vt[0]
        else:
            pending["vent"] = (vt, EXT_VENT_LIST)
    if "heat_source" in eff:
        hs = heat_source_preset(eff["heat_source"], path=path,
                                where=f"{w}.heat_source")
        if eff.get("heat_source_enabled", True):
            tags["heat_source"] = hs
        else:
            pending["heat_source"] = ({**hs, "enabled": False},
                                      EXT_HEAT_ENABLED)
    if "hazard" in eff:
        tags["hazard"] = hazard_preset(eff["hazard"], path=path,
                                       where=f"{w}.hazard")
    if dept.physics.temp != AMBIENT:
        tags["temp"] = temp_preset(dept.physics.temp, path=path,
                                   where="physics.temp")
    if dept.physics.pressure != AMBIENT:
        pending["pressure"] = (pressure_preset(dept.physics.pressure,
                                               path=path,
                                               where="physics.pressure"),
                               EXT_PRESSURE)
    if "water" in eff:
        pending["water"] = (eff["water"], EXT_WATER)
    for k, v in tags.items():
        what, ok = ROOM_TAGS[k]
        if not ok(v):
            raise _err(path, w, f"resolved tag {k!r} = {v!r} is not {what} "
                                f"(gen_ship.ROOM_TAGS)")
    return ResolvedRoom(id=room.id, tags=tags, pending=pending)


def resolve_department(dept: Department, fixtures: dict, rng, *,
                       cell: int | None = None) -> ResolvedDepartment:
    """Every recipe room of ``dept`` through :func:`resolve_room`, in recipe
    order (so the draws are a pure function of the generator's state)."""
    rooms = {r.id: resolve_room(dept, r.id, fixtures, rng, cell=cell)
             for r in dept.rooms}
    ducts = {d.name: {"filter": d.filter} for d in dept.ducts}
    h = hashlib.sha256()
    for p in (dept.path, str(_root_of(Path(dept.path)) / "fixtures.toml")):
        h.update(Path(p).read_bytes())
    h.update(json.dumps(
        {"rooms": {k: [v.tags, {pk: pv[0] for pk, pv in v.pending.items()}]
                   for k, v in rooms.items()}, "ducts": ducts},
        sort_keys=True, default=list).encode())
    return ResolvedDepartment(name=dept.name, rooms=rooms, ducts=ducts,
                              digest=h.hexdigest())
