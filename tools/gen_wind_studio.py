r"""tools/gen_wind_studio.py -- generates levels/wind_studio/, the HUMAN-TEST
level for wind on units (#140; docs/wind_on_units_design_2026-10-31.md §6.1
scene 1, the #140 scene in a room of its own).

An 80 x 36 hull at the reference 0.333 m tile inside the studios' vacuum
band (boundary "space"):

  * ROOM A (west, 12 x 12 interior) sealed at 1.44 atm (the #140 room's
    overpressure) by ``door_a``, a closed 3-tile door in its east wall;
  * a CORRIDOR (9 wide, 20 long) from that door east to
  * the HALL, a large ambient room (24 x 26);
  * ROOM B (east, 12 x 12) sealed at 1.03 atm by ``door_b`` in the hall's
    east wall -- the slowed band without the knockdown. MEASURED, not the
    brief's 1.15 atm: opening a sealed room is a transient, and at the
    doorway its first ticks carry the most q -- 1.15 atm peaks ~1.6 kPa there
    (knocked down), 1.05 ~480 Pa, 1.03 ~210 Pa (in the band, a few ticks).

The squad (``[[spawn]]``, the authored single-player path): Alpha IN the
doorway of room A (its footprint against the door), Bravo and Charlie
flanking it two tiles out, Delta behind it; Echo against door_b on the hall
side. The door under the cursor toggles with the dev key O
(``debug_keys.debug_toggle_door`` -> ``door.want_open``).

The overpressure is the level's ``air_init.npy`` (int32 Q16.16 atm, 1.0 atm
== 65536; the A9 override, which also seeds the bulk N, so the room holds
under the EOS). Ambient everywhere else.

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md "Level generators"): the header, the
boundary, the [[spawn]]/[[entity]] blocks, the CSV and the air grid all go
through level_lib; diffuse.png is render art painted with PIL. The CSV
carries the door spans as AIR -- the load-time door stamp seals the closed
ones. DETERMINISTIC (no RNG).

Run:
    C:/Users/steen/anaconda3/python.exe tools/gen_wind_studio.py
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
from level_loader import SPACE_CODE, EntityInstance, SpawnEntry  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

LEVEL_NAME = "wind_studio"
DEFAULT_OUT_DIR = ROOT / "levels" / LEVEL_NAME
TILE_SIZE_M = studio.TILE_SIZE_M          # the reference tile, 0.333 m
PX = studio.PX
BAND = studio.BAND
W, H = 80, 36
FP_ONE = 1 << 16

# Interiors, inclusive (x0, y0, x1, y1). Everything inside the hull box that
# is not carved here is hull.
ROOM_A = (5, 10, 16, 21)                  # 12 x 12, west wall = the hull
CORRIDOR = (18, 11, 37, 19)               # 20 long x 9 wide
HALL = (38, 5, 61, 30)                    # 24 x 26
ROOM_B = (63, 12, 74, 23)                 # 12 x 12, east wall = the hull
P_ROOM_A_ATM = 1.44
P_ROOM_B_ATM = 1.03

# Doors: (id, x, y, orientation, length_m) -- 1.0 m = 3 tiles at 0.333 m.
DOORS = (
    ("door_a", 17, 14, "v", 1.0),         # room A <-> corridor, rows 14..16
    ("door_b", 62, 16, "v", 1.0),         # hall <-> room B, rows 16..18
)
DOOR_TILES = 3

# Spawns: (name, x, y) top-left of a 3x3 footprint.
SPAWNS = (
    ("Alpha", 18, 14),                    # in room A's doorway
    ("Bravo", 19, 11),                    # flanking, two tiles out
    ("Charlie", 19, 17),
    ("Delta", 22, 14),                    # behind Alpha, in the jet
    ("Echo", 59, 16),                     # in room B's doorway (hall side)
)


def _carve(tm, box, value):
    x0, y0, x1, y1 = box
    tm[y0:y1 + 1, x0:x1 + 1] = value


def build_tilemap() -> np.ndarray:
    tm = np.full((H, W), SPACE_CODE, dtype=np.int32)
    tm[BAND:H - BAND, BAND:W - BAND] = MAT_HULL
    for box in (ROOM_A, CORRIDOR, HALL, ROOM_B):
        _carve(tm, box, MAT_AIR)
    for (_id, x, y, _o, _l) in DOORS:          # door spans are AIR in the CSV
        tm[y:y + DOOR_TILES, x] = MAT_AIR
    for (n, x, y) in SPAWNS:
        assert (tm[y:y + 3, x:x + 3] == MAT_AIR).all(), f"{n} is not on floor"
    return tm


def build_air_init(tm: np.ndarray) -> np.ndarray:
    """int32 Q16.16 atm: ambient on every tile, the two sealed rooms raised.
    (Values on solid / space tiles are ignored by the seed.)"""
    air = np.full(tm.shape, FP_ONE, dtype=np.int32)
    _carve(air, ROOM_A, int(P_ROOM_A_ATM * FP_ONE + 0.5))
    _carve(air, ROOM_B, int(P_ROOM_B_ATM * FP_ONE + 0.5))
    return air


def build_doors() -> list:
    keys = ("x", "y", "orientation", "length_m", "initial_state")
    return [EntityInstance(id=did, class_name="door", ordinal=i,
                           fields={"x": x, "y": y, "orientation": o,
                                   "length_m": float(length),
                                   "initial_state": "closed"},
                           authored_keys=keys)
            for i, (did, x, y, o, length) in enumerate(DOORS)]


def build_spawns() -> list:
    return [SpawnEntry(name=n, team=0, x=x, y=y, footprint=3)
            for (n, x, y) in SPAWNS]


# Render art: the studios' flat tiles; the pressurised floors tinted warm.
PALETTE = {SPACE_CODE: (6, 6, 12), MAT_AIR: (34, 36, 42),
           MAT_HULL: (96, 102, 116)}
AIR_ALT = (30, 32, 38)
TINT_A = (58, 40, 36)                     # 1.44 atm
TINT_B = (52, 46, 34)                     # 1.03 atm
DOOR_RGB = (150, 120, 60)


def build_diffuse(tm: np.ndarray) -> Image.Image:
    img = np.zeros((H * PX, W * PX, 3), dtype=np.uint8)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = PALETTE.get(code, (200, 40, 200))   # loud magenta = bug
            if code == MAT_AIR and (tx + ty) % 2:
                c = AIR_ALT
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    for box, tint in ((ROOM_A, TINT_A), (ROOM_B, TINT_B)):
        x0, y0, x1, y1 = box
        img[y0 * PX:(y1 + 1) * PX, x0 * PX:(x1 + 1) * PX] = tint
    for (_id, x, y, _o, _l) in DOORS:
        img[y * PX:(y + DOOR_TILES) * PX, x * PX:(x + 1) * PX] = DOOR_RGB
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.uint8)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.uint8)
    return Image.fromarray(img)


HEADER_COMMENTS = (
    "WIND STUDIO -- the HUMAN-TEST level for wind on units (#140). GENERATED",
    "by tools/gen_wind_studio.py through level_lib -- edit the constants there",
    "and re-run. Room A (west) is sealed at 1.44 atm by door_a, room B (east)",
    "at 1.03 atm by door_b (air_init.npy). Alpha stands in room A's doorway,",
    "Bravo/Charlie flank it, Delta behind; Echo at room B's door.",
    "Launch: C:/Users/steen/anaconda3/python.exe main.py --level wind_studio",
    "(the WEGO default: --control gamepad polls no dev keys, so O is dead",
    "there). Keys: O toggles the door under the cursor (the 9e sweep applies",
    "it on the next unpaused tick) | Space unpause | F6 tile inspector (q/W).",
    "",
    *studio.CODE_LEGEND,
)


def main(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir = Path(out_dir)
    for name in ("level.toml", "tilemap.csv", "diffuse.png",
                 level_lib.AIR_INIT_FILENAME):
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    toml_path = level_lib.write_level_header(
        out_dir, name="Wind Studio", tile_size_m=TILE_SIZE_M,
        comment_lines=HEADER_COMMENTS)
    level_lib.write_boundary_field(toml_path, "space")
    tm = build_tilemap()
    level_lib.write_tilemap_csv(out_dir, tm, csv_bak=False)
    level_lib.write_air_init_npy(out_dir, build_air_init(tm), npy_bak=False)
    build_diffuse(tm).save(out_dir / "diffuse.png")
    doors, spawns = build_doors(), build_spawns()
    level_lib.write_managed_blocks(
        toml_path,
        {"spawn": lambda nl: level_lib.format_spawn_lines(spawns, nl),
         "entity": lambda nl: level_lib.format_entity_lines(doors, nl)})
    print(f"wrote {out_dir}  ({W}x{H} tiles @ {TILE_SIZE_M} m)")


if __name__ == "__main__":
    main()
