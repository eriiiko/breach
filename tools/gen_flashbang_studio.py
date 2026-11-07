r"""tools/gen_flashbang_studio.py -- generates levels/flashbang_studio/, the G4 P5
look studio for the blast flash (#75, docs/explosion_look_design_2026-11-07.md
§3.5): two identical sealed rooms side by side, a ``flashbang`` timed charge at
each room's centre, both firing at the same tick as levels/blast_final's volley
(2 s = tick 48 at 24 tps, so the ``*_t50`` / ``*_t52`` look shots line up), every
PERIOD_S seconds after.

  * WEST room -- clean: the flash alone on the lamp-lit floor.
  * EAST room -- a standing SOOT column beside the charge (``[payloads.studio_smoke]``,
    a cold trace deposit, no blast and no heat), laid SMOKE_LEAD_S before the
    flash, so the column absorbs the flash and casts its shadow across the floor
    behind it (design §1 item 3: free, the sweep reads every emitter and absorber
    of the same tick).

The flashbang row is a flash and nothing else (no strength, no radius, no heat,
no gas), so the rooms stay as they are between flashes. The lamps are the blast
studios' (9 per room at LAMP_INTENSITY 0.4, tools/gen_blast_quad.py).

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md "Level data layer"); no RNG, so a re-run
gives a byte-identical folder.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_flashbang_studio.py
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
from gen_blast_quad import LAMP_INTENSITY  # noqa: E402
from gen_explosion_studio import CODE_LEGEND  # noqa: E402
from level_loader import SPACE_CODE, EntityInstance, LightEntry  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

LEVEL_NAME = "flashbang_studio"
TITLE = "Flashbang Studio"

BAND = 4                 # space outside the hull (tiles)
ROOM_W, ROOM_H = 30, 24  # interior, 10 x 8 m at 0.333 m tiles (blast_quad's rooms)
TILE_SIZE_M = 0.333
PX = 16                  # diffuse px per tile

X0, Y0 = BAND, BAND
X_MID = X0 + ROOM_W + 1                     # the dividing hull wall
X1, Y1 = X_MID + ROOM_W + 1, Y0 + ROOM_H + 1
W, H = X1 + 1 + BAND, Y1 + 1 + BAND

FIRST_S = 2.0            # = blast_final's volley (tick 48)
PERIOD_S = 8.0
SMOKE_LEAD_S = 1.5       # the soot column is laid this long before the flash
SMOKE_DX = 4             # the column's centre, tiles east of the east charge

ROOMS = (("CLEAN", X0 + 1, Y0 + 1), ("SMOKE COLUMN", X_MID + 1, Y0 + 1))


def centre(room):
    return room[1] + ROOM_W // 2, room[2] + ROOM_H // 2


def build_tilemap() -> np.ndarray:
    tm = np.full((H, W), SPACE_CODE, dtype=np.int32)
    tm[Y0:Y1 + 1, X0:X1 + 1] = MAT_HULL
    for (_label, rx, ry) in ROOMS:
        tm[ry:ry + ROOM_H, rx:rx + ROOM_W] = MAT_AIR
    return tm


def charge_rows() -> tuple:
    """(id, x, y, payload, first_at_s): a flashbang per room, the east room's
    soot column SMOKE_LEAD_S earlier."""
    (wx, wy), (ex, ey) = (centre(r) for r in ROOMS)
    return (("flashbang_clean", wx, wy, "flashbang", FIRST_S),
            ("studio_smoke_column", ex + SMOKE_DX, ey, "studio_smoke",
             FIRST_S - SMOKE_LEAD_S),
            ("flashbang_smoke", ex, ey, "flashbang", FIRST_S))


def build_charges(rows) -> list:
    keys = ("x", "y", "payload", "first_at_s", "period_s", "enabled")
    return [EntityInstance(id=cid, class_name="timed_charge", ordinal=i,
                           fields={"x": x, "y": y, "payload": payload,
                                   "first_at_s": float(first),
                                   "period_s": PERIOD_S, "enabled": True},
                           authored_keys=keys)
            for i, (cid, x, y, payload, first) in enumerate(rows)]


def build_lights() -> list:
    return [LightEntry(x=r[1] + ROOM_W * fx, y=r[2] + ROOM_H * fy,
                       color=(1.0, 0.92, 0.82), intensity=LAMP_INTENSITY,
                       range=16.0, kind="static")
            for r in ROOMS for fy in (1 / 6, 0.5, 5 / 6) for fx in (1 / 6, 0.5, 5 / 6)]


PALETTE = {SPACE_CODE: (6, 6, 12), MAT_AIR: (232, 234, 238),
           MAT_HULL: (96, 102, 116)}
AIR_ALT = (214, 216, 222)
CHARGE_RGB = (170, 40, 30)
SMOKE_RGB = (60, 60, 64)


def build_diffuse(tm: np.ndarray, rows) -> Image.Image:
    img = np.zeros((H * PX, W * PX, 3), dtype=np.uint8)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = PALETTE[code]
            if code == MAT_AIR and (tx + ty) % 2:
                c = AIR_ALT
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    q = PX // 4
    for (_id, x, y, payload, _t) in rows:
        rgb = CHARGE_RGB if payload == "flashbang" else SMOKE_RGB
        img[y * PX + q:(y + 1) * PX - q, x * PX + q:(x + 1) * PX - q] = rgb
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:            # older PIL: fixed-size default font
        font = ImageFont.load_default()
    for (label, rx, ry) in ROOMS:
        draw.text(((rx + ROOM_W // 2) * PX, (ry + 3) * PX), label,
                  fill=(24, 26, 30), font=font, anchor="mm")
    return pil


HEADER_COMMENTS = (
    "FLASHBANG STUDIO -- the blast flash's look studio (G4 P5, #75).",
    "GENERATED by tools/gen_flashbang_studio.py through level_lib -- edit the",
    "constants there and re-run. The rows are [payloads.flashbang] and",
    "[payloads.studio_smoke] in config.toml.",
    "",
    *CODE_LEGEND,
)


def main() -> None:
    out_dir = ROOT / "levels" / LEVEL_NAME
    for name in ("level.toml", "tilemap.csv", "diffuse.png"):
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = level_lib.write_level_header(
        out_dir, name=TITLE, tile_size_m=TILE_SIZE_M,
        comment_lines=HEADER_COMMENTS)
    level_lib.write_boundary_field(toml_path, "space")
    tm = build_tilemap()
    level_lib.write_tilemap_csv(out_dir, tm, csv_bak=False)
    rows = charge_rows()
    build_diffuse(tm, rows).save(out_dir / "diffuse.png")
    lights = build_lights()
    charges = build_charges(rows)
    level_lib.write_managed_blocks(
        toml_path,
        {"light": lambda nl: level_lib.format_light_lines(lights, nl),
         "entity": lambda nl: level_lib.format_entity_lines(charges, nl)})
    print(f"wrote {out_dir}  ({W}x{H} tiles @ {TILE_SIZE_M} m)")


if __name__ == "__main__":
    main()
