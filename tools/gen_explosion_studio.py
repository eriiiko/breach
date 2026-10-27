r"""tools/gen_explosion_studio.py — generates levels/explosion_studio/, the
test level for the "explosion + air" session (#31;
docs/prep_patches_handoff_2026-09-30.md P5).

A 48x48 hull at 0.333 m tiles (16 m across) inside a vacuum band (boundary "space"), cut into
three rooms joined by open doorways, with furniture crates (fuel) beside four
``timed_charge`` entities:

  * NE room  — studio_small  at 2 s, then every 5 s
  * SE room  — studio_medium at 4 s, then every 5 s
  * W hall   — studio_large  at 6 s, then every 5 s
  * NE room, against the north hull — studio_hull at 8 s, then every 5 s:
    breaches the hull so the room vents its smoke to vacuum.

The payload rows are ``[payloads.studio_*]`` in config.toml (provisional);
Erik edits the rows and this level afterwards.

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md "Level data layer"; the "Level
generators" rule). This tool is a client:
  - level.toml header  -> :func:`level_lib.write_level_header`
  - boundary scalar    -> :func:`level_lib.write_boundary_field`
  - [[light]]/[[entity]] -> :func:`level_lib.write_managed_blocks`
  - tilemap.csv        -> :func:`level_lib.write_tilemap_csv` (LF)
  - diffuse.png is RENDER ART (flat colour-coded tiles), painted with PIL —
    the bare-diffuse family fire_studio / fire_tuning ship; no bake needed.
``tools/gen_fire_studio.py``'s own ``np.savetxt`` + hand-written toml is the
anti-pattern this replaces (#67 item 3).

DETERMINISTIC — no RNG (run twice -> byte-identical folder).

Run:
    C:/Users/steen/anaconda3/python.exe tools/gen_explosion_studio.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_lib  # noqa: E402
from level_loader import EntityInstance, LightEntry  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL  # noqa: E402

LEVEL_NAME = "explosion_studio"
DEFAULT_OUT_DIR = ROOT / "levels" / LEVEL_NAME
SPACE_CODE = 9

BAND = 4                 # vacuum band outside the hull (tiles)
HULL = 48                # hull box side (tiles), outer faces included
W = H = HULL + 2 * BAND  # 56 x 56 grid
# THE REFERENCE TILE (Erik, 2026-10-04: tune everything at 0.333 m; was
# 1.0 m). The conduction table, the radiation calibration and soot's derived
# extinctions are built at [physics.thermal] tile_size_ref_m = 0.333 -- a 1 m
# level runs them off-reference (T3 section 8 q9). Values judged here at 1 m
# (k_drag2 = 0.125, the frag grenade) are to be re-judged.
TILE_SIZE_M = 0.333
PX = 16                  # diffuse px per tile

# Hull box (inclusive outer faces) and the interior partitions.
X0 = Y0 = BAND                       # 4
X1 = Y1 = BAND + HULL - 1            # 51
WALL_X = 26                          # west hall | east rooms
WALL_Y = 27                          # NE room | SE room (east half only)
# Open doorways (AIR gaps in the partitions), 3 tiles wide.
DOOR_W_NE = (14, 16)                 # rows in WALL_X, west hall <-> NE room
DOOR_W_SE = (38, 40)                 # rows in WALL_X, west hall <-> SE room
DOOR_NE_SE = (37, 39)                # cols in WALL_Y, NE room <-> SE room

# (id, x, y, payload, first_at_s) — period_s is PERIOD_S for all four.
PERIOD_S = 5.0
CHARGES = (
    ("charge_small", 38, 15, "studio_small", 2.0),
    ("charge_medium", 38, 39, "studio_medium", 4.0),
    ("charge_large", 14, 27, "studio_large", 6.0),
    ("charge_hull", 44, Y0 + 1, "studio_hull", 8.0),
)
# Furniture crates (fuel) beside each interior charge: (x0, y0) of a 2x2.
CRATES = ((41, 15), (41, 39), (18, 27), (9, 26))

# Lamps: (x, y) tile centres, warm static.
LAMPS = ((15.5, 12.5), (38.5, 8.5), (38.5, 45.5), (15.5, 44.5))


def build_tilemap() -> np.ndarray:
    tm = np.full((H, W), SPACE_CODE, dtype=np.int32)
    tm[Y0:Y1 + 1, X0:X1 + 1] = MAT_HULL
    tm[Y0 + 1:Y1, X0 + 1:X1] = MAT_AIR
    tm[Y0 + 1:Y1, WALL_X] = MAT_HULL
    tm[WALL_Y, WALL_X:X1] = MAT_HULL
    tm[DOOR_W_NE[0]:DOOR_W_NE[1] + 1, WALL_X] = MAT_AIR
    tm[DOOR_W_SE[0]:DOOR_W_SE[1] + 1, WALL_X] = MAT_AIR
    tm[WALL_Y, DOOR_NE_SE[0]:DOOR_NE_SE[1] + 1] = MAT_AIR
    for (cx, cy) in CRATES:
        tm[cy:cy + 2, cx:cx + 2] = MAT_FURNITURE
    for (_id, x, y, _p, _t) in CHARGES:
        assert tm[y, x] == MAT_AIR, f"{_id} sits on a non-air tile"
    return tm


def build_charges(charges=CHARGES, period_s: float = PERIOD_S) -> list:
    ents = []
    keys = ("x", "y", "payload", "first_at_s", "period_s", "enabled")
    for i, (cid, x, y, payload, first) in enumerate(charges):
        fields = {"x": x, "y": y, "payload": payload,
                  "first_at_s": float(first), "period_s": float(period_s),
                  "enabled": True}
        ents.append(EntityInstance(id=cid, class_name="timed_charge",
                                   ordinal=i, fields=fields,
                                   authored_keys=keys))
    return ents


def build_lights() -> list:
    return [LightEntry(x=x, y=y, color=(1.0, 0.84, 0.67), intensity=1.2,
                       range=16.0, kind="static") for (x, y) in LAMPS]


# ---------------------------------------------------------------------------
# Diffuse art — flat colour-coded tiles, a faint grid, charge markers.
# RENDER-ONLY; the CSV is the physics truth.
# ---------------------------------------------------------------------------
PALETTE = {
    SPACE_CODE:    (6, 6, 12),
    MAT_AIR:       (34, 36, 42),
    MAT_HULL:      (96, 102, 116),
    MAT_FURNITURE: (96, 71, 41),
}
AIR_ALT = (30, 32, 38)
CHARGE_RGB = (170, 40, 30)
LABELS = ((15, 7, "WEST HALL"), (38, 6, "NE ROOM"), (38, 30, "SE ROOM"))


def build_diffuse(tm: np.ndarray, charges=CHARGES, palette=None,
                  air_alt=None) -> Image.Image:
    palette = PALETTE if palette is None else palette
    air_alt = AIR_ALT if air_alt is None else air_alt
    img = np.zeros((H * PX, W * PX, 3), dtype=np.uint8)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = palette.get(code, (200, 40, 200))     # loud magenta = bug
            if code == MAT_AIR and (tx + ty) % 2:
                c = air_alt
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    for (_id, x, y, _p, _t) in charges:
        q = PX // 4
        img[y * PX + q:(y + 1) * PX - q, x * PX + q:(x + 1) * PX - q] = CHARGE_RGB
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.uint8)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:            # older PIL: fixed-size default font
        font = ImageFont.load_default()
    for (tx, ty, text) in LABELS:
        draw.text((tx * PX, ty * PX), text, fill=(160, 166, 174), font=font,
                  anchor="mm")
    return pil


HEADER_COMMENTS = (
    "EXPLOSION STUDIO — the test level for the #31 \"explosion + air\"",
    "session (docs/prep_patches_handoff_2026-09-30.md P5). GENERATED by",
    "tools/gen_explosion_studio.py through level_lib — edit the constants",
    "there and re-run, or edit the [[entity]] rows here (level_lib-managed).",
    "Four timed_charge entities fire [payloads.studio_*] rows on the sim",
    "clock; see README.md for the launch line.",
    "",
    "v2 codes ARE canon material ids (src/simulation/materials.py):",
    "  0=air 1=hull 6=furniture 9=SPACE",
)


def main(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir = Path(out_dir)
    # Regenerate the level's DATA files; README.md (hand-written) stays.
    for name in ("level.toml", "tilemap.csv", "diffuse.png"):
        p = out_dir / name
        if p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    toml_path = level_lib.write_level_header(
        out_dir, name="Explosion Studio", tile_size_m=TILE_SIZE_M,
        comment_lines=HEADER_COMMENTS)
    level_lib.write_boundary_field(toml_path, "space")
    tm = build_tilemap()
    level_lib.write_tilemap_csv(out_dir, tm, csv_bak=False)
    build_diffuse(tm).save(out_dir / "diffuse.png")
    lights, charges = build_lights(), build_charges()
    level_lib.write_managed_blocks(
        toml_path,
        {"light": lambda nl: level_lib.format_light_lines(lights, nl),
         "entity": lambda nl: level_lib.format_entity_lines(charges, nl)})

    print(f"wrote {out_dir}  ({W}x{H} tiles @ {TILE_SIZE_M} m, codes "
          f"{sorted(np.unique(tm).tolist())})")
    for (cid, x, y, payload, first) in CHARGES:
        print(f"  {cid:14s} ({x},{y})  {payload:14s} first {first} s, "
              f"every {PERIOD_S} s")


if __name__ == "__main__":
    main()
