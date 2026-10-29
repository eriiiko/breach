"""Match config + spawn rooms (#119 P2, rl-ready scope §7).

A MATCH is parameterised by ONE file beside its level,
``levels/<level>/match.toml``, written by ``level_lib.write_match`` and read
by ``level_loader.load`` (``LevelData.match``) — never hand-written, never
argv scans, never per-test dicts::

    objective = "eliminate"      # the only value today
    seed = 1                     # -> Simulation(seed=...); nothing else draws
    time_limit_s = 300.0         # sim seconds (default 300.0); draw at the limit
    zombies = 5                  # placed in the ZOMBIE_TEAM spawn room

    [[team]]                     # row i IS team i (0, 1, 2, ...)
    n = 4
    loadout = "rifleman"         # a [loadouts.*] row
    env_profile = "default"      # an [env_profiles.*] row
    controller = "idle"          # a registered controller (control_source)

WHERE units stand is the LEVEL's decision (Erik's ruling 2026-10-29): a level
carries one ``[[spawn_room]]`` per team (``team, x, y, w, h, max_per_team,
placement``), zombies included (``team = ZOMBIE_TEAM``), written by
``level_lib.write_spawn_rooms``. Spawn-room teams are RUNTIME ints — the
legacy ``team = 1`` zombie mapping of ``[[spawn]]`` does NOT apply here (a
spawn room for team 1 is a marine team's). ``[[spawn]]`` stays for authored
single-player levels; a match never reads it.

v1 has ONE placement rule, ``formation``: slot ``k`` of a team stands at the
``k``-th cell (row-major) of a fixed grid of 3x3 footprints, two tiles apart,
centred in the room — the same offsets relative to its room for every team, so
two equal rooms give symmetric starts, and a slot always starts on the same
cell. A second rule is a new ``placement`` value, refused until it exists.

Import-light (stdlib + ``simulation.factions`` at module level): the level
loader imports it. The door's table checks (loadouts, profiles, controllers)
import their tables lazily, only when a match file is actually read.
"""
from __future__ import annotations

from dataclasses import dataclass

from simulation.factions import ZOMBIE_TEAM

#: The placement rules a spawn room may name (v1: one).
PLACEMENT_RULES = ("formation",)
#: The objectives a match may name (v1: one; bomb-site is v1.1).
OBJECTIVES = ("eliminate",)
DEFAULT_TIME_LIMIT_S = 300.0

#: ``formation``'s geometry: a human's 3x3 footprint, two free tiles between
#: neighbours (the authored playground squad's spacing).
FORMATION_FOOTPRINT = 3
FORMATION_GAP = 2
FORMATION_PITCH = FORMATION_FOOTPRINT + FORMATION_GAP

_ROOM_KEYS = ("team", "x", "y", "w", "h", "max_per_team", "placement")
_MATCH_KEYS = ("objective", "seed", "time_limit_s", "zombies", "team")
_TEAM_KEYS = ("n", "loadout", "env_profile", "controller")


@dataclass(frozen=True)
class SpawnRoom:
    """One ``[[spawn_room]]``: the tile rectangle ``[x, x+w) x [y, y+h)`` a
    team spawns in, its slot budget and its placement rule."""
    team: int
    x: int
    y: int
    w: int
    h: int
    max_per_team: int
    placement: str = "formation"


@dataclass(frozen=True)
class MatchTeam:
    """One ``[[team]]`` row of match.toml (its index is its team id)."""
    n: int
    loadout: str
    env_profile: str
    controller: str


@dataclass(frozen=True)
class MatchSpec:
    """A parsed, validated match.toml."""
    teams: tuple            # tuple[MatchTeam, ...]; teams[i] is team i
    seed: int
    zombies: int = 0
    time_limit_s: float = DEFAULT_TIME_LIMIT_S
    objective: str = "eliminate"


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _grid(room: SpawnRoom) -> tuple[int, int]:
    cols = (room.w + FORMATION_GAP) // FORMATION_PITCH
    rows = (room.h + FORMATION_GAP) // FORMATION_PITCH
    return max(0, cols), max(0, rows)


def formation_capacity(room: SpawnRoom) -> int:
    """How many 3x3 footprints ``formation`` fits in ``room``."""
    cols, rows = _grid(room)
    return cols * rows


def formation_cells(room: SpawnRoom, n: int) -> list:
    """The top-left tiles ``(x, y)`` of slots ``0 .. n-1`` under
    ``formation``: a ``cols x rows`` grid at pitch 5, the block centred in
    the room (integer floor), filled row-major. Pure integer arithmetic, so
    the offsets relative to the room depend on ``(w, h)`` alone."""
    cols, rows = _grid(room)
    if n > cols * rows:
        raise ValueError(
            f"spawn room for team {room.team} ({room.w}x{room.h}) fits "
            f"{cols * rows} footprints under 'formation', asked for {n}")
    ox = (room.w - (cols * FORMATION_PITCH - FORMATION_GAP)) // 2
    oy = (room.h - (rows * FORMATION_PITCH - FORMATION_GAP)) // 2
    return [(room.x + ox + (k % cols) * FORMATION_PITCH,
             room.y + oy + (k // cols) * FORMATION_PITCH) for k in range(n)]


def placement_cells(room: SpawnRoom, n: int) -> list:
    """The ONE dispatch from a room's ``placement`` rule to its cells."""
    if room.placement == "formation":
        return formation_cells(room, n)
    raise ValueError(f"unknown placement {room.placement!r} "
                     f"(known: {list(PLACEMENT_RULES)})")


# ---------------------------------------------------------------------------
# The door — spawn rooms
# ---------------------------------------------------------------------------

def _int(value, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer (got {value!r})")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be >= {minimum} (got {value})")
    return int(value)


def spawn_room_from_dict(entry: dict, where: str) -> SpawnRoom:
    """One raw ``[[spawn_room]]`` table -> a validated :class:`SpawnRoom`
    (schema-local checks: keys, ints, a known placement, a room big enough
    for its ``max_per_team``). The grid checks (in bounds, open floor) are
    the loader's, which holds the tilemap."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a table")
    unknown = sorted(set(entry) - set(_ROOM_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} "
                         f"(a spawn room carries {list(_ROOM_KEYS)})")
    for k in _ROOM_KEYS[:-1]:
        if k not in entry:
            raise ValueError(f"{where}: missing key '{k}'")
    placement = entry.get("placement", "formation")
    if placement not in PLACEMENT_RULES:
        raise ValueError(f"{where}: unknown placement {placement!r} "
                         f"(known: {list(PLACEMENT_RULES)})")
    room = SpawnRoom(
        team=_int(entry["team"], f"{where}.team", minimum=0),
        x=_int(entry["x"], f"{where}.x", minimum=0),
        y=_int(entry["y"], f"{where}.y", minimum=0),
        w=_int(entry["w"], f"{where}.w", minimum=1),
        h=_int(entry["h"], f"{where}.h", minimum=1),
        max_per_team=_int(entry["max_per_team"], f"{where}.max_per_team",
                          minimum=1),
        placement=str(placement))
    cap = formation_capacity(room)
    if room.max_per_team > cap:
        raise ValueError(
            f"{where}: a {room.w}x{room.h} room fits {cap} footprints under "
            f"'{room.placement}', too small for max_per_team = "
            f"{room.max_per_team}")
    return room


def parse_spawn_rooms(raw_list, where: str) -> list:
    """The ``[[spawn_room]]`` array -> a list of rooms, one per team."""
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        raise ValueError(f"[[spawn_room]] in {where} must be an array of "
                         f"tables — spell it [[spawn_room]]")
    rooms = [spawn_room_from_dict(e, f"[[spawn_room]] #{i} in {where}")
             for i, e in enumerate(raw_list)]
    seen = set()
    for r in rooms:
        if r.team in seen:
            raise ValueError(f"{where}: two [[spawn_room]] entries for team "
                             f"{r.team} — one room per team")
        seen.add(r.team)
    return rooms


# ---------------------------------------------------------------------------
# The door — match.toml
# ---------------------------------------------------------------------------

def parse_match(raw: dict, spawn_rooms, where: str) -> MatchSpec:
    """A raw match.toml dict -> a validated :class:`MatchSpec`, checked
    against the level's ``spawn_rooms`` and the live tables: a known
    objective, >= 2 team rows, each ``n`` within its room's
    ``max_per_team``, loadout / env_profile names rows of their tables, a
    registered controller; zombies need a ``ZOMBIE_TEAM`` room."""
    unknown = sorted(set(raw) - set(_MATCH_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} "
                         f"(a match carries {list(_MATCH_KEYS)})")
    objective = raw.get("objective", "eliminate")
    if objective not in OBJECTIVES:
        raise ValueError(f"{where}: unknown objective {objective!r} "
                         f"(known: {list(OBJECTIVES)})")
    if "seed" not in raw:
        raise ValueError(f"{where}: missing key 'seed' (a match is seeded)")
    seed = _int(raw["seed"], f"{where}.seed", minimum=0)
    zombies = _int(raw.get("zombies", 0), f"{where}.zombies", minimum=0)
    tl = raw.get("time_limit_s", DEFAULT_TIME_LIMIT_S)
    if isinstance(tl, bool) or not isinstance(tl, (int, float)) or tl <= 0:
        raise ValueError(f"{where}.time_limit_s must be a positive number "
                         f"(got {tl!r})")
    team_rows = raw.get("team", [])
    if not isinstance(team_rows, list) or len(team_rows) < 2:
        raise ValueError(f"{where}: a match needs at least two [[team]] "
                         f"rows (got {team_rows!r})")

    rooms = {r.team: r for r in spawn_rooms}
    # Lazy: the tables need CFG + the weapon table; only a match file pays.
    from simulation.loadouts import get_tables
    from control_source import registered_controllers
    tables = get_tables()
    controllers = registered_controllers()

    teams = []
    for i, row in enumerate(team_rows):
        w = f"{where} [[team]] #{i}"
        if not isinstance(row, dict):
            raise ValueError(f"{w} must be a table")
        bad = sorted(set(row) - set(_TEAM_KEYS))
        if bad:
            raise ValueError(f"{w}: unknown key(s) {bad} "
                             f"(a team row carries {list(_TEAM_KEYS)})")
        for k in _TEAM_KEYS:
            if k not in row:
                raise ValueError(f"{w}: missing key '{k}'")
        n = _int(row["n"], f"{w}.n", minimum=1)
        if i == ZOMBIE_TEAM:
            raise ValueError(f"{w}: team {i} is ZOMBIE_TEAM")
        room = rooms.get(i)
        if room is None:
            raise ValueError(f"{w}: the level has no [[spawn_room]] for "
                             f"team {i}")
        if n > room.max_per_team:
            raise ValueError(f"{w}: n = {n} exceeds the team's spawn room "
                             f"max_per_team = {room.max_per_team}")
        loadout, profile, ctrl = row["loadout"], row["env_profile"], \
            row["controller"]
        if not isinstance(loadout, str) or \
                loadout not in tables.loadouts.by_name:
            raise ValueError(f"{w}.loadout {loadout!r} names no [loadouts.*] "
                             f"row (known: {sorted(tables.loadouts.by_name)})")
        if not isinstance(profile, str) or \
                profile not in tables.env_profiles.by_name:
            raise ValueError(
                f"{w}.env_profile {profile!r} names no [env_profiles.*] row "
                f"(known: {sorted(tables.env_profiles.by_name)})")
        if not isinstance(ctrl, str) or ctrl not in controllers:
            raise ValueError(f"{w}.controller {ctrl!r} is not a registered "
                             f"controller (known: {sorted(controllers)})")
        teams.append(MatchTeam(n=n, loadout=loadout, env_profile=profile,
                               controller=ctrl))
    if zombies:
        zroom = rooms.get(ZOMBIE_TEAM)
        if zroom is None:
            raise ValueError(f"{where}: zombies = {zombies} but the level has "
                             f"no [[spawn_room]] for ZOMBIE_TEAM "
                             f"({ZOMBIE_TEAM})")
        if zombies > zroom.max_per_team:
            raise ValueError(f"{where}: zombies = {zombies} exceeds the zombie "
                             f"room's max_per_team = {zroom.max_per_team}")
    return MatchSpec(teams=tuple(teams), seed=seed, zombies=zombies,
                     time_limit_s=float(tl), objective=objective)


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------

def unit_name(team: int, slot: int) -> str:
    """``T<team>-<slot>`` for a marine, ``Z-<slot>`` for a zombie."""
    return f"Z-{slot}" if team == ZOMBIE_TEAM else f"T{team}-{slot}"


def spawn_match(sim, spec: MatchSpec) -> list:
    """Spawn ``spec``'s units into a FRESH ``sim`` (no units yet): each team
    row's ``n`` units from its loadout + env_profile, then the zombies, each
    standing on its slot's cell of its room's placement rule. Returns the
    units in spawn order. Draws no randomness (the match seed is the sim's
    own, passed at construction)."""
    from simulation.unit import Unit

    if sim.units:
        raise ValueError("spawn_match needs a fresh Simulation (it already "
                         f"holds {len(sim.units)} units); slots start at 0")
    rooms = {r.team: r for r in sim.level.spawn_rooms}
    spawned = []

    def place(team, n, **kw):
        cells = placement_cells(rooms[team], n)
        for slot, (x, y) in enumerate(cells):
            u = Unit(unit_name(team, slot), x=x, y=y, team=team,
                     footprint=FORMATION_FOOTPRINT, **kw)
            sim.add_unit(u)
            if u.slot != slot:      # add_unit owns slots; they must agree
                raise RuntimeError(f"{u.name}: roster slot {u.slot} != "
                                   f"placement slot {slot}")
            spawned.append(u)

    for team, row in enumerate(spec.teams):
        place(team, row.n, loadout_name=row.loadout,
              env_profile=row.env_profile)
    if spec.zombies:
        place(ZOMBIE_TEAM, spec.zombies)
    return spawned


__all__ = [
    "PLACEMENT_RULES", "OBJECTIVES", "DEFAULT_TIME_LIMIT_S",
    "FORMATION_FOOTPRINT", "FORMATION_GAP", "FORMATION_PITCH",
    "SpawnRoom", "MatchTeam", "MatchSpec",
    "formation_capacity", "formation_cells", "placement_cells",
    "spawn_room_from_dict", "parse_spawn_rooms", "parse_match",
    "unit_name", "spawn_match",
]
