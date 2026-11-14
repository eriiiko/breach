r"""tools/gen_blast_quad.py -- generates levels/blast_quad/, the #31 side-by-side
blast comparison: four rooms, one candidate delivery each, all firing together
every PERIOD_S seconds on the sim clock. The same layout gives the other levels
of the SETS table; levels/blast_final is the real rows, with a wooden partition
and fuel in the rooms (FEATURES).

  * NW room -- quad_src_tile  one tile, one tick, the standard blast (Erik, 2026-11-06)
  * NE room -- div_40         a disc of radius 3, two ticks
  * SW room -- quad_src_r5    a disc of radius 5, two ticks
  * SE room -- quad_src_slow  radius 3, spread over SLOW_TICKS ticks: SLOW_TICKS
    charges on one tile, one tick apart, each 1/SLOW_TICKS of the blast

All four are the same divergence-source blast with the same total heat, smoke and
expansion, in config.toml (provisional). Each room has a gap in its outer wall to open sky (boundary
"ambient"), so a blast's pressure bleeds off before the next one and repeated
shots stay comparable.

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md "Level data layer"); no RNG, so a re-run
gives a byte-identical folder.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_blast_quad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_lib  # noqa: E402
from gen_explosion_studio import CODE_LEGEND, build_charges  # noqa: E402
from level_loader import SPACE_CODE, LightEntry  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_FURNITURE, MAT_HULL,  # noqa: E402
                                  MAT_WOOD)

LEVEL_NAME = "blast_quad"

BAND = 6                 # open sky outside the hull (tiles)
ROOM_W, ROOM_H = 30, 24  # interior, 10 x 8 m at 0.333 m tiles
TILE_SIZE_M = 0.333
PX = 16                  # diffuse px per tile
GAP = 4                  # the gap to open sky, tiles (1.33 m)

X0, Y0 = BAND, BAND
X_MID = X0 + ROOM_W + 1
Y_MID = Y0 + ROOM_H + 1
X1, Y1 = X_MID + ROOM_W + 1, Y_MID + ROOM_H + 1
W, H = X1 + 1 + BAND, Y1 + 1 + BAND

TPS = 24
FIRST_S = 2.0
PERIOD_S = 3.0
SLOW_TICKS = 6
LAMP_INTENSITY = 0.4     # 36 lamps; 1.6 until 2026-11-07, then 0.8, then 0.4: Erik judged the studio far brighter than a generated ship at the game's exposure (G4 P3)

# The rooms, in order NW, NE, SW, SE: (label, payload row, charges one tick apart).
# A second set of rows is a second level through the same layout.
SETS = {
    "blast_quad": ("Blast Quad", (
        ("1 TILE", "quad_src_tile", 1),
        ("DISC R3", "div_40", 1),
        ("DISC R5", "quad_src_r5", 1),
        (f"R3 OVER {SLOW_TICKS} TICKS", "quad_src_slow", SLOW_TICKS),
    )),
    "blast_div": ("Blast Div", (
        ("HOT MASS", "quad_r3", 1),
        ("SOURCE 40 M3", "div_40", 1),
        ("SOURCE 80 M3", "div_80", 1),
        ("SOURCE 40 + PUFF", "div_40_puff", 1),
    )),
    "blast_smoke": ("Blast Smoke", (
        ("SMOKE 3.5 G", "div_40", 1),
        ("SMOKE 1.8 G", "div_40_s50", 1),
        ("SMOKE 0.9 G", "div_40_s25", 1),
        ("SMOKE 0.4 G", "div_40_s10", 1),
    )),
    "blast_final": ("Blast Final", (
        ("FRAG", "frag_standard", 1),
        ("BREACH ON WOOD", "breach_focus", 1),
        ("MAGAZINE", "magazine_cookoff", 1),
        ("C4", "demolition_c4", 1),
    )),
}
# The seconds between a set's volleys, where the default is too short to see one
# volley's fire and smoke before the next.
PERIODS = {"blast_final": 8.0}
SLOTS = ((X0 + 1, Y0 + 1, "w"), (X_MID + 1, Y0 + 1, "e"),
         (X0 + 1, Y_MID + 1, "w"), (X_MID + 1, Y_MID + 1, "e"))
ROOMS = tuple((label, *slot) for (label, _, _), slot
              in zip(SETS["blast_quad"][1], SLOTS))


def centre(room):
    return room[1] + ROOM_W // 2, room[2] + ROOM_H // 2


def charges_for(rows):
    return tuple((f"{row}_{i}_{k}", *centre(ROOMS[i]), row, FIRST_S + k / TPS)
                 for i, (_label, row, n) in enumerate(rows) for k in range(n))


def final_features(tm: np.ndarray) -> None:
    """blast_final's furniture and wood, all relative to each room's charge."""
    (nw, ne, sw, _se) = (centre(r) for r in ROOMS)
    tm[nw[1] - 1:nw[1] + 2, nw[0] + 1] = MAT_FURNITURE      # crates one tile off
    tm[ROOMS[1][2]:ROOMS[1][2] + ROOM_H, ne[0]] = MAT_WOOD  # the charge's wall
    tm[sw[1] - 1:sw[1] + 2, sw[0] + 3] = MAT_FURNITURE      # crates three tiles off
    tm[sw[1] + 3, sw[0] - 1:sw[0] + 2] = MAT_FURNITURE


FEATURES = {"blast_final": final_features}


def build_tilemap(level: str = LEVEL_NAME) -> np.ndarray:
    tm = np.full((H, W), SPACE_CODE, dtype=np.int32)
    tm[Y0:Y1 + 1, X0:X1 + 1] = MAT_HULL
    for (_label, rx, ry, side) in ROOMS:
        tm[ry:ry + ROOM_H, rx:rx + ROOM_W] = MAT_AIR
        mid = ry + ROOM_H // 2
        col = X0 if side == "w" else X1
        tm[mid - GAP // 2:mid + GAP // 2, col] = MAT_AIR
    if level in FEATURES:
        FEATURES[level](tm)
    return tm


def build_lights() -> list:
    return [LightEntry(x=r[1] + ROOM_W * fx, y=r[2] + ROOM_H * fy,
                       color=(1.0, 0.92, 0.82), intensity=LAMP_INTENSITY,
                       range=16.0, kind="static")
            for r in ROOMS for fy in (1 / 6, 0.5, 5 / 6) for fx in (1 / 6, 0.5, 5 / 6)]


PALETTE = {SPACE_CODE: (58, 70, 88), MAT_AIR: (232, 234, 238),
           MAT_HULL: (96, 102, 116), MAT_WOOD: (150, 104, 62),
           MAT_FURNITURE: (120, 78, 44)}
AIR_ALT = (214, 216, 222)
CHARGE_RGB = (170, 40, 30)


def build_diffuse(tm: np.ndarray, rows, charges) -> Image.Image:
    img = np.zeros((H * PX, W * PX, 3), dtype=np.uint8)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = PALETTE[code]
            if code == MAT_AIR and (tx + ty) % 2:
                c = AIR_ALT
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    for (_id, x, y, _p, _t) in charges:
        q = PX // 4
        img[y * PX + q:(y + 1) * PX - q, x * PX + q:(x + 1) * PX - q] = CHARGE_RGB
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:            # older PIL: fixed-size default font
        font = ImageFont.load_default()
    for (label, _row, _n), (rx, ry, _side) in zip(rows, SLOTS):
        draw.text(((rx + ROOM_W // 2) * PX, (ry + 3) * PX), label,
                  fill=(24, 26, 30), font=font, anchor="mm")
    return pil


HEADER_COMMENTS = (
    "BLAST QUAD -- four candidate blast deliveries side by side (#31).",
    "GENERATED by tools/gen_blast_quad.py through level_lib -- edit the",
    "constants there and re-run. The strengths are the [payloads.quad_src_*] rows",
    "in config.toml; see README.md for the launch line.",
    "",
    *CODE_LEGEND,
)


def main(level: str = LEVEL_NAME) -> None:
    title, rows = SETS[level]
    out_dir = ROOT / "levels" / level
    for name in ("level.toml", "tilemap.csv", "diffuse.png"):
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    toml_path = level_lib.write_level_header(
        out_dir, name=title, tile_size_m=TILE_SIZE_M,
        comment_lines=HEADER_COMMENTS)
    level_lib.write_boundary_field(toml_path, "ambient")
    tm = build_tilemap(level)
    level_lib.write_tilemap_csv(out_dir, tm, csv_bak=False)
    charge_rows = charges_for(rows)
    build_diffuse(tm, rows, charge_rows).save(out_dir / "diffuse.png")
    lights = build_lights()
    charges = build_charges(charge_rows, PERIODS.get(level, PERIOD_S))
    level_lib.write_managed_blocks(
        toml_path,
        {"light": lambda nl: level_lib.format_light_lines(lights, nl),
         "entity": lambda nl: level_lib.format_entity_lines(charges, nl)})
    print(f"wrote {out_dir}  ({W}x{H} tiles @ {TILE_SIZE_M} m)")


if __name__ == "__main__":
    main()
    main("blast_div")
    main("blast_smoke")
    main("blast_final")
