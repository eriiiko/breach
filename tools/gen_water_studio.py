r"""tools/gen_water_studio.py -- generates levels/water_studio/, THE level for
judging water (#97, the water look arc's patch W0;
docs/water_look_design_2026-11-19.md section 2 "W0", designed with Erik on
2026-11-20, rulings in its section 8).

A 52 x 35 grid at the reference 0.333 m tile inside the studios' vacuum band
(boundary "space"), every room sealed by hull, so each room's water levels
out over its own flat floor (``floor_height`` is all zero: a "pool" is a
flooded room). Four rooms:

  * THE POOL HALL (north-west, 12 x 8 interior), flooded to 0.30 m. One lamp
    over its middle (the look shots switch it off with ``lamps = false``). A
    2 x 2 stack of furniture crates stands in the water in its north-east
    part and is lit at 0.5 s by one ``timed_charge`` per crate tile firing
    ``[payloads.studio_ignite]`` (heat at its ignition_temp, the heat-only
    way of the "Starting a fire" rule; the fire_smoke_studio precedent), so
    there is firelight and smoke over the water. A ``timed_charge`` standing
    IN the water fires ``[payloads.engine_death]`` (a generic explosion of
    strength 3, radius 3, wall damage 300) at 8 s and every 20 s after:
    waves now, spray in W3. Measured headless (2026-11-20): of the rows that
    break nothing here it moves the water most (the frag grenade's
    ``frag_standard``: flow 0.02 m/s; this row: 0.05 m/s and the depth
    swinging 0.274..0.293 m), and it breaks nothing -- its wall damage reaches
    the hull's 300 hp only at its own tile, and furniture is not a blast
    wall. THE OCTOPUS CORNER is the hall's
    SOUTH-WEST corner (``OCTOPUS_CORNER``, 4 x 4 tiles): nothing is placed
    there and nothing is coded for it (ruling of 2026-11-20).
  * THE DAM AND THE CORRIDOR (south): a tank (5 x 6 interior, water 1.20 m:
    aquarium_demo's depth, inside its safe band) at the head of a 3 x 25
    corridor. Its front is a glass pane at each end and, between them, a
    WOODEN SLUICE GATE: the dam. A ``timed_charge`` in the corridor against
    the gate fires ``[payloads.breach_focus]`` (the door charge) at 3 s and
    breaks the four gate tiles (wood hp 60; the hull beside it, hp 300, is
    out of the charge's reach for that much damage), and the tank's water
    runs the corridor into a 10 x 11 basin room at its far end. One-shot by
    design: a level has no running water source (ruling of 2026-11-20).
    FINDING (W0): the spec asked for a GLASS dam broken by a charge; a blast
    cannot break glass today (glass is not in
    ``physics._BLAST_WALL_MATERIALS``, and the over-pressure burst is off,
    ``[physics] burst_enabled = false``, #186) -- only rounds chew it. The
    gate is wood until that is ruled; no sim change here.
  * THE WADING ROOM (north, 18 x 11), flooded to 0.15 m. A STOVE: one steel
    tile against the north wall carrying a ``heat_source`` that holds it at
    ``STOVE_MAX_TEMP`` game degrees above ambient (orange heat, steady
    light from the solid's own emission; the source sits ON the steel, radius
    0, so it never rails the air). Beside it a wood wall (west) and a
    furniture fuel pile (east), which can catch from it. The match is fought
    here: a marine on the layered ``bots`` (team 0) and one zombie, each in
    its own spawn room in this room.
  * THE BOOTH (north-east, 5 x 5, dry, sealed): the second marine team's
    room. A match needs two marine teams (``MatchRules`` ends the moment
    fewer than two have a living unit, and ``parse_match`` asks for two
    rows), so team 1 is ONE marine on ``idle`` that nobody can reach: the
    bot tournament's ``--horde`` precedent (X against an idle team, the horde
    in). The match ends when the bot is turned (team 1 wins) or at the clock.

A ``[[spawn]]`` marine ("Wader") stands in the wading room's water by the
stove: the single-player path (``main.py --level water_studio``) and the
look shots (``tools/lighting_demo.py`` spawns ``[[spawn]]`` rows only). A
match never reads ``[[spawn]]``.

Render art: the studios' flat light-grey tiles (diffuse.png) and the
smoke-light studio's seam-groove normal map, so the water is judged over a
floor that reads; no bake.

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md "Level generators"): the header, the
boundary, the [[spawn]] / [[spawn_room]] / [[light]] / [[entity]] blocks, the
CSV, the [water] depth map (water_init.npy) and match.toml all go through
level_lib. DETERMINISTIC (no RNG): run twice -> byte-identical folder.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_water_studio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import gen_explosion_studio as studio  # noqa: E402
import gen_smoke_light_studio as still  # noqa: E402
import level_lib  # noqa: E402
from level_loader import (  # noqa: E402
    SPACE_CODE, EntityInstance, LightEntry, SpawnEntry,
)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_FURNITURE, MAT_GLASS, MAT_HULL, MAT_STEEL, MAT_WOOD,
)

LEVEL_NAME = "water_studio"
DEFAULT_OUT_DIR = ROOT / "levels" / LEVEL_NAME
TILE_SIZE_M = studio.TILE_SIZE_M          # the reference tile, 0.333 m
PX = studio.PX
BAND = studio.BAND
W, H = 52, 35
FP_ONE = 1 << 16

# Interiors, inclusive (x0, y0, x1, y1). Everything inside the hull box that
# is not carved here is hull, so every room is sealed.
POOL = (6, 6, 17, 13)                     # 12 x 8
WADE = (20, 6, 37, 16)                    # 18 x 11
BOOTH = (40, 6, 44, 10)                   # 5 x 5
TANK = (6, 20, 10, 25)                    # 5 x 6
# The tank's front, across the corridor's head: glass panes at its two ends
# and a wooden SLUICE GATE between them (the dam). A blast cannot break glass
# today (physics._BLAST_WALL_MATERIALS has no glass and the over-pressure
# burst is off, burst_enabled = false, #186), so the dam is wood (hp 60).
GLASS_PANES = ((11, 20), (11, 25))
GATE = (11, 21, 11, 24)                   # the dam: wood across the mouth
CORRIDOR = (12, 22, 36, 24)               # 25 x 3
BASIN = (37, 19, 46, 29)                  # 10 x 11

# Standing water at load (metres), per room (the [water] depth map).
DEPTH_M = {"pool": 0.30, "wade": 0.15, "tank": 1.20}

# --- the pool hall ---------------------------------------------------------
CRATES = (14, 7, 15, 8)                   # the 2 x 2 furniture stack
IGNITE_AT_S = 0.5
WAVE_CHARGE = (11, 9)                     # in the water, > 2 tiles from all
WAVE_PAYLOAD = "engine_death"             # strength 3: the biggest generic
                                          # explosion that breaks nothing here
WAVE_FIRST_S = 8.0
WAVE_PERIOD_S = 20.0
# Kept EMPTY for the octopus (design section 8): nothing placed, nothing coded.
OCTOPUS_CORNER = (6, 10, 9, 13)           # the hall's south-west 4 x 4

# --- the dam -----------------------------------------------------------------
DAM_CHARGE = (12, 23)                     # in the corridor, against the pane
DAM_PAYLOAD = "breach_focus"              # the door charge: wood 60 hp goes
                                          # out to ~2.6 tiles, hull 300 to 1.2
DAM_AT_S = 3.0

# --- the wading room ---------------------------------------------------------
STOVE = (28, 6)                           # one steel tile on the north wall
WOOD_WALL = (27, 6, 27, 9)                # west of the stove, touching it
FUEL = (29, 6, 30, 7)                     # east of the stove, touching it
# The heat source ON the stove tile. Steel's thermal_mass is 32 (heat >> 5),
# so HEAT_PER_S raises the bare tile ~100 game degrees a second before
# losses (hot in ~10 s); MAX_TEMP holds it at orange heat (~1290 K on the
# honest map: at 600, ~890 K, its light barely showed in wade_by_firelight),
# well above furniture's ignition_temp (280) and wood's (300), so its
# neighbours can catch (headless: the wood wall at ~38 s). Radius 0: the deposit lands on the steel alone (on an air tile it
# would rail the gas).
STOVE_HEAT_PER_S = 3200.0
STOVE_MAX_TEMP = 1000.0
STOVE_RADIUS = 0
# The [[spawn]] marine for the single-player path and the look shots: in the
# water, a tile clear of the fuel pile, south-east of the stove.
WADER = ("Wader", 29, 9)

# --- the match ---------------------------------------------------------------
# (team, x, y, w, h, max_per_team): one 3 x 3 formation cell each.
SPAWN_ROOMS = (
    (0, 22, 12, 3, 3, 1),                 # the bot marine: wading room, SW
    (1, 41, 7, 3, 3, 1),                  # the idle marine: the booth
    (ZOMBIE_TEAM, 33, 11, 3, 3, 1),       # the zombie: wading room, E
)
MATCH_SEED = 1
MATCH_TIME_LIMIT_S = 60.0
MATCH_LOADOUT = "rifleman"

# Lamps: (x, y) tile centres, warm static. The pool hall's ONE lamp first;
# none in the wading room (the stove and its fire are its light).
LAMPS = (
    (11.5, 9.5),                          # pool hall
    (8.5, 22.5),                          # over the tank
    (18.5, 23.5), (30.5, 23.5),           # the corridor
    (41.5, 24.5),                         # the basin
    (42.5, 8.5),                          # the booth
)


def _carve(tm, box, value):
    x0, y0, x1, y1 = box
    tm[y0:y1 + 1, x0:x1 + 1] = value


def _box_mask(shape, box):
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def build_tilemap() -> np.ndarray:
    tm = np.full((H, W), SPACE_CODE, dtype=np.int32)
    tm[BAND:H - BAND, BAND:W - BAND] = MAT_HULL
    for box in (POOL, WADE, BOOTH, TANK, CORRIDOR, BASIN):
        _carve(tm, box, MAT_AIR)
    for (x, y) in GLASS_PANES:
        tm[y, x] = MAT_GLASS
    _carve(tm, GATE, MAT_WOOD)
    _carve(tm, CRATES, MAT_FURNITURE)
    _carve(tm, WOOD_WALL, MAT_WOOD)
    _carve(tm, FUEL, MAT_FURNITURE)
    tm[STOVE[1], STOVE[0]] = MAT_STEEL
    # the authoring rules the constants must keep
    for (x, y) in (WAVE_CHARGE, DAM_CHARGE):
        assert tm[y, x] == MAT_AIR, f"charge at ({x}, {y}) is not on floor"
    assert (tm[OCTOPUS_CORNER[1]:OCTOPUS_CORNER[3] + 1,
               OCTOPUS_CORNER[0]:OCTOPUS_CORNER[2] + 1] == MAT_AIR).all()
    for (n, x, y) in (WADER,):
        assert (tm[y:y + 3, x:x + 3] == MAT_AIR).all(), f"{n} is not on floor"
    for (_t, x, y, w, h, _m) in SPAWN_ROOMS:
        assert (tm[y:y + h, x:x + w] == MAT_AIR).all(), "spawn room off floor"
    return tm


def build_water(tm: np.ndarray) -> np.ndarray:
    """int32 Q16.16 metres: each flooded room's depth on its OPEN tiles only
    (a solid tile is a mass sink, so the crates, the stove, the wood wall and
    the fuel pile stand dry in the water)."""
    d = np.zeros(tm.shape, dtype=np.int32)
    for box, key in ((POOL, "pool"), (WADE, "wade"), (TANK, "tank")):
        m = _box_mask(tm.shape, box) & (tm == MAT_AIR)
        d[m] = int(round(DEPTH_M[key] * FP_ONE))
    return d


def _charge(cid, ordinal, x, y, payload, first_s, period_s):
    keys = ("x", "y", "payload", "first_at_s", "period_s", "enabled")
    return EntityInstance(
        id=cid, class_name="timed_charge", ordinal=ordinal,
        fields={"x": x, "y": y, "payload": payload,
                "first_at_s": float(first_s), "period_s": float(period_s),
                "enabled": True},
        authored_keys=keys)


def build_entities() -> list:
    ents = []
    x0, y0, x1, y1 = CRATES
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            ents.append(_charge(f"ignite_{x}_{y}", len(ents), x, y,
                                "studio_ignite", IGNITE_AT_S, 0.0))
    ents.append(_charge("wave_charge", len(ents), *WAVE_CHARGE, WAVE_PAYLOAD,
                        WAVE_FIRST_S, WAVE_PERIOD_S))
    ents.append(_charge("dam_charge", len(ents), *DAM_CHARGE, DAM_PAYLOAD,
                        DAM_AT_S, 0.0))
    ents.append(EntityInstance(
        id="stove", class_name="heat_source", ordinal=len(ents),
        fields={"x": STOVE[0], "y": STOVE[1], "heat_per_s": STOVE_HEAT_PER_S,
                "radius": STOVE_RADIUS, "enabled": True, "hp": 200,
                "death_payload": "", "max_temp": STOVE_MAX_TEMP},
        authored_keys=("x", "y", "heat_per_s", "radius", "max_temp")))
    return ents


def build_spawns() -> list:
    n, x, y = WADER
    return [SpawnEntry(name=n, team=0, x=x, y=y, footprint=3)]


def build_spawn_rooms() -> list:
    return [SpawnRoom(team=t, x=x, y=y, w=w, h=h, max_per_team=m)
            for (t, x, y, w, h, m) in SPAWN_ROOMS]


def build_match() -> MatchSpec:
    return MatchSpec(
        teams=(MatchTeam(n=1, loadout=MATCH_LOADOUT, env_profile="default",
                         controller="bots"),
               MatchTeam(n=1, loadout=MATCH_LOADOUT, env_profile="default",
                         controller="idle")),
        seed=MATCH_SEED, zombies=1, time_limit_s=MATCH_TIME_LIMIT_S)


def build_lights() -> list:
    return [LightEntry(x=x, y=y, color=(1.0, 0.84, 0.67), intensity=1.2,
                       range=16.0, kind="static") for (x, y) in LAMPS]


# Render art: the smoke-light studio's light-grey floor (the water is judged
# over a floor that reads), its hull; the materials in the room their own.
PALETTE = dict(still.PALETTE)
PALETTE.update({MAT_GLASS: (150, 180, 190), MAT_STEEL: (60, 62, 68),
                MAT_WOOD: (120, 86, 52), MAT_FURNITURE: (96, 71, 41)})
AIR_ALT = still.AIR_ALT


def build_diffuse(tm: np.ndarray) -> Image.Image:
    img = np.zeros((H * PX, W * PX, 3), dtype=np.uint8)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = PALETTE.get(code, (200, 40, 200))   # loud magenta = bug
            if code == MAT_AIR and (tx + ty) % 2:
                c = AIR_ALT
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.uint8)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.uint8)
    return Image.fromarray(img)


HEADER_COMMENTS = (
    "WATER STUDIO -- THE level for judging water (#97 W0,",
    "docs/water_look_design_2026-11-19.md). GENERATED by",
    "tools/gen_water_studio.py through level_lib -- edit the constants there",
    "and re-run. Pool hall (NW, 0.30 m): one lamp, a crate stack lit at",
    "0.5 s, a blast in the water at 8 s and every 20 s; its SW corner is kept",
    "empty for the octopus. Dam (S): a 1.20 m tank's wooden gate blown at 3 s,",
    "its flood runs a 3 x 25 corridor into the basin. Wading room (N, 0.15 m):",
    "a stove (heat source on steel) beside a wood wall and a fuel pile; the",
    "match (match.toml) is a bot marine against one zombie here, team 1 an",
    "idle marine in the sealed booth (NE). Watch:",
    "  C:/Users/steen/anaconda3/envs/data/python.exe main.py",
    "  --level water_studio --control match",
    "",
    *studio.CODE_LEGEND,
    "  2=wood 4=steel 5=glass",
)


def main(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir = Path(out_dir)
    for name in ("level.toml", "tilemap.csv", "diffuse.png", "normal.png",
                 level_lib.WATER_FILENAME, "match.toml"):
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    toml_path = level_lib.write_level_header(
        out_dir, name="Water Studio", tile_size_m=TILE_SIZE_M,
        normal_rel="normal.png", comment_lines=HEADER_COMMENTS)
    level_lib.write_boundary_field(toml_path, "space")
    tm = build_tilemap()
    level_lib.write_tilemap_csv(out_dir, tm, csv_bak=False)
    build_diffuse(tm).save(out_dir / "diffuse.png")
    still.build_normal(tm).save(out_dir / "normal.png")
    _nbak, has_water = level_lib.write_water_npy(out_dir, build_water(tm),
                                                  npy_bak=False)
    ents, spawns = build_entities(), build_spawns()
    rooms, lights = build_spawn_rooms(), build_lights()
    level_lib.write_managed_blocks(
        toml_path,
        {"spawn": lambda nl: level_lib.format_spawn_lines(spawns, nl),
         "spawn_room": lambda nl: level_lib.format_spawn_room_lines(rooms, nl),
         "light": lambda nl: level_lib.format_light_lines(lights, nl),
         "entity": lambda nl: level_lib.format_entity_lines(ents, nl),
         "water": level_lib.water_block_format(has_water)})
    level_lib.write_match(out_dir, build_match())
    print(f"wrote {out_dir}  ({W}x{H} tiles @ {TILE_SIZE_M} m)")


if __name__ == "__main__":
    main()
