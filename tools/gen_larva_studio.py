r"""tools/gen_larva_studio.py -- generates levels/larva_studio/, the room where
Erik first watches the larvae (arc #63 P4, the swarm units' first look).

Its purpose: one person, in two minutes, sees every behaviour the larva law
has (docs/swarm_P4_impl.md). An 80 x 36 hull at the reference 0.333 m tile
inside the studios' vacuum band (boundary "space"):

  * THE NURSERY (west, 26 x 22 interior) at room temperature, two broods
    (``brood_west``, ``brood_east``, 300 larvae each) painted over its
    middle: they crawl, tumble and turn off its walls.
  * A WIDE DOORWAY (14 tiles) into
  * THE WARM ROOM (east, 26 x 22): its floor seeded warm (``temp_init.npy``,
    a ramp from ambient at the doorway to +14 game degrees, 34 C, at its
    east wall) and held there by a ``heat_source`` in its north-east corner
    (``max_temp`` 14, so the corner never runs away). The comfort band (30 C,
    +10) lies across the room, so the larvae drift in and gather along it.
  * THE FUEL STACK: furniture crates against the warm room's east wall,
    beside the heat source. Light it with the dev key I (ignite under the
    cursor) and watch the larvae at the warm edge die where it passes 50 C.
  * NO COLD ROOM: nothing in the game today makes a room colder than
    ambient honestly (``temp_init.npy`` refuses sub-ambient, a parked physics
    ruling; the ambient temperature is the global ``eos_t_amb_k``). Coma is
    shown by the swarm gate's scene only.
  * One marine (``[[spawn]]``) in the nursery's south-west corner, to walk
    the level in direct control.

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md "Level generators"): the header, the
boundary, the [[spawn]] / [[light]] / [[entity]] blocks, the CSV, zones.npy
and temp_init.npy all go through level_lib; diffuse.png is render art painted
with PIL (the studios' flat tiles). DETERMINISTIC (no RNG): run twice ->
byte-identical folder.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_larva_studio.py
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
import level_lib  # noqa: E402
from level_loader import (  # noqa: E402
    SPACE_CODE, EntityInstance, LightEntry, SpawnEntry,
)
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL  # noqa: E402

LEVEL_NAME = "larva_studio"
DEFAULT_OUT_DIR = ROOT / "levels" / LEVEL_NAME
TILE_SIZE_M = studio.TILE_SIZE_M          # the reference tile, 0.333 m
PX = studio.PX
BAND = studio.BAND
W, H = 80, 36
FP_ONE = 1 << 16

# Interiors, inclusive (x0, y0, x1, y1). Everything inside the hull box that
# is not carved here is hull.
NURSERY = (6, 7, 31, 28)                  # 26 x 22
DOORWAY = (32, 11, 35, 24)                # 4 deep x 14 wide, open
WARM = (36, 7, 61, 28)                    # 26 x 22
# The broods' painted zones (zone_id, box) and their counts.
BROODS = (
    ("brood_west", 1, (9, 11, 17, 24), 300),
    ("brood_east", 2, (20, 11, 28, 24), 300),
)
# The warm room's seed: game degrees above ambient, a ramp in x from 0 at
# its west wall to WARM_MAX at its east wall (30 C = +10 lies at ~70 %).
WARM_MAX = 14.0
# The heat source holding the north-east corner (base tiles).
HEAT_X, HEAT_Y = 59, 9
HEAT_PER_S = 20.0                         # the `heat` plane's unit per second
HEAT_RADIUS = 0
HEAT_MAX_TEMP = 12.0                      # game degrees above ambient (32 C)
# The fuel stack: furniture against the east wall, south of the source.
FUEL = (59, 15, 61, 22)
# The marine: top-left of its 3x3 footprint.
SPAWN = ("Alpha", 7, 25)
# Lamps: (x, y) tile centres, warm static.
LAMPS = ((13.5, 12.5), (24.5, 22.5), (43.5, 12.5), (54.5, 22.5))


def _carve(tm, box, value):
    x0, y0, x1, y1 = box
    tm[y0:y1 + 1, x0:x1 + 1] = value


def build_tilemap() -> np.ndarray:
    tm = np.full((H, W), SPACE_CODE, dtype=np.int32)
    tm[BAND:H - BAND, BAND:W - BAND] = MAT_HULL
    for box in (NURSERY, DOORWAY, WARM):
        _carve(tm, box, MAT_AIR)
    _carve(tm, FUEL, MAT_FURNITURE)
    n, x, y = SPAWN
    assert (tm[y:y + 3, x:x + 3] == MAT_AIR).all(), f"{n} is not on floor"
    assert tm[HEAT_Y, HEAT_X] == MAT_AIR, "the heat source is not on floor"
    return tm


def build_zones() -> np.ndarray:
    z = np.zeros((H, W), dtype=np.uint8)
    for _id, zid, box, _n in BROODS:
        _carve(z, box, zid)
    return z


def build_temp_init(tm: np.ndarray) -> np.ndarray:
    """int32 Q16.16 game degrees above ambient: the warm room's air a ramp
    in x, everything else ambient (0). Air tiles only (the loader refuses a
    non-zero seed on a wall without thermal mass)."""
    t = np.zeros(tm.shape, dtype=np.int32)
    x0, y0, x1, y1 = WARM
    for x in range(x0, x1 + 1):
        deg = WARM_MAX * (x - x0) / float(x1 - x0)
        col = t[y0:y1 + 1, x]
        col[:] = int(round(deg * FP_ONE))
    t[tm != MAT_AIR] = 0
    return t


def build_entities() -> list:
    ents = []
    for i, (bid, zid, _box, count) in enumerate(BROODS):
        ents.append(EntityInstance(
            id=bid, class_name="swarm_brood", ordinal=i,
            fields={"zone_id": zid, "species": "larva", "count": count},
            authored_keys=("zone_id", "species", "count")))
    ents.append(EntityInstance(
        id="warm_corner", class_name="heat_source", ordinal=len(ents),
        fields={"x": HEAT_X, "y": HEAT_Y, "heat_per_s": HEAT_PER_S,
                "radius": HEAT_RADIUS, "enabled": True, "hp": 200,
                "death_payload": "", "max_temp": HEAT_MAX_TEMP},
        authored_keys=("x", "y", "heat_per_s", "radius", "max_temp")))
    return ents


def build_spawns() -> list:
    n, x, y = SPAWN
    return [SpawnEntry(name=n, team=0, x=x, y=y, footprint=3)]


def build_lights() -> list:
    return [LightEntry(x=x, y=y, color=(1.0, 0.84, 0.67), intensity=1.2,
                       range=16.0, kind="static") for (x, y) in LAMPS]


# Render art: the studios' flat tiles; the warm room's floor tinted by its
# seed, the broods' zones faintly marked, the source a red square.
PALETTE = {SPACE_CODE: (6, 6, 12), MAT_AIR: (34, 36, 42),
           MAT_HULL: (96, 102, 116), MAT_FURNITURE: (96, 71, 41)}
AIR_ALT = (30, 32, 38)
WARM_TINT = (22, 4, -6)                   # added at the ramp's top
BROOD_RGB = (40, 44, 38)
SOURCE_RGB = (170, 40, 30)


def build_diffuse(tm: np.ndarray, temp_q: np.ndarray) -> Image.Image:
    img = np.zeros((H * PX, W * PX, 3), dtype=np.int32)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = PALETTE.get(code, (200, 40, 200))   # loud magenta = bug
            if code == MAT_AIR and (tx + ty) % 2:
                c = AIR_ALT
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    for _id, _zid, (x0, y0, x1, y1), _n in BROODS:
        img[y0 * PX:(y1 + 1) * PX, x0 * PX:(x1 + 1) * PX] = BROOD_RGB
    frac = np.repeat(np.repeat(temp_q / (WARM_MAX * FP_ONE), PX, 0), PX, 1)
    img += (frac[..., None] * np.array(WARM_TINT)[None, None, :]).astype(np.int32)
    q = PX // 4
    img[HEAT_Y * PX + q:(HEAT_Y + 1) * PX - q,
        HEAT_X * PX + q:(HEAT_X + 1) * PX - q] = SOURCE_RGB
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.int32)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.int32)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


HEADER_COMMENTS = (
    "LARVA STUDIO -- where the larvae are first watched (arc #63 P4).",
    "GENERATED by tools/gen_larva_studio.py through level_lib -- edit the",
    "constants there and re-run. The nursery (west) holds two broods of 300",
    "larvae; the warm room (east) is seeded warm (temp_init.npy, up to 34 C",
    "at its east wall) and held by the heat source in its NE corner; the",
    "furniture stack beside it is fuel: light it with I under the cursor.",
    "Launch: C:/Users/steen/anaconda3/envs/data/python.exe main.py",
    "--level larva_studio --control direct   (F6 tile inspector: larvae on",
    "the tile; [debug] swarm_dots draws them).",
    "",
    *studio.CODE_LEGEND,
)


def main(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir = Path(out_dir)
    for name in ("level.toml", "tilemap.csv", "diffuse.png",
                 level_lib.ZONES_FILENAME, level_lib.TEMP_INIT_FILENAME):
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    toml_path = level_lib.write_level_header(
        out_dir, name="Larva Studio", tile_size_m=TILE_SIZE_M,
        comment_lines=HEADER_COMMENTS)
    level_lib.write_boundary_field(toml_path, "space")
    tm = build_tilemap()
    temp = build_temp_init(tm)
    level_lib.write_tilemap_csv(out_dir, tm, csv_bak=False)
    level_lib.write_zones_npy(out_dir, build_zones(), npy_bak=False)
    level_lib.write_temp_init_npy(out_dir, temp, npy_bak=False)
    build_diffuse(tm, temp).save(out_dir / "diffuse.png")
    ents, spawns, lights = build_entities(), build_spawns(), build_lights()
    level_lib.write_managed_blocks(
        toml_path,
        {"spawn": lambda nl: level_lib.format_spawn_lines(spawns, nl),
         "light": lambda nl: level_lib.format_light_lines(lights, nl),
         "entity": lambda nl: level_lib.format_entity_lines(ents, nl)})
    print(f"wrote {out_dir}  ({W}x{H} tiles @ {TILE_SIZE_M} m)")


if __name__ == "__main__":
    main()
