"""Generate the PLAYGROUND level — the mechanics/06 §8 standard-values sandbox.

Writes ``levels/playground/tilemap.csv`` (v2 codes = canon material ids) and
``levels/playground/diffuse.png`` (a flat, colour-coded, room-labelled map so
the layout reads under a flashlight). ``levels/playground/level.toml`` is
hand-written and committed; re-running this tool only regenerates the two
derived files, so tweak the layout constants below and re-run:

    C:/Users/steen/anaconda3/python.exe tools/gen_playground_level.py

DESIGN (P5, 2026-07-05) — one room per system the combat wave shipped, sized
so a grenade's ~10-tile knockdown ring and a squad-v-horde brawl both fit:

    x ->                                                            100 wide
  y +--------------------------- SPACE (9) ---------------------------+
  | |  hull shell (1)                                                 |
  v |   ARENA (open floor, NW = grenade range,   WOOD ROOM   GLASS   |
    |   furniture cover clusters mid-map)        (fire fuel) GALLERY |
    |                                                                 |
    |   marine squad spawns SW                   SEALED ROOM STEEL   |
    |                                            (pressure)  BUNKER  |
    |   BREACH BAY   POOL BASIN                                       |
    |   (blow the    (steel tub,                 POOL        ZOMBIE  |
    |   south hull)  pour with U)                BASIN       PEN     |
    +-----------------------------------------------------------------+

70 tall. Verified emergent properties this layout leans on (P5 audit):
  - blast STRUCTURAL damage only hits hull/wood/door (physics.py's material
    set) -> glass/steel walls ignore grenade wall_damage entirely;
  - glass (burst_threshold 1.0) pops on ATMOSPHERE differential
    (find_burst_walls) -> a grenade's +10 atm disc opens the zombie pen and
    the sealed room's window; steel (10.0) survives all but a point-blank
    epicentre; hull (0.0) never bursts — it breaches via wall damage only;
  - glass is SOLID (occludes -> permeability 0) -> blocks zombie LOS: the
    penned horde stays dormant until a pane pops;
  - fire fuel = wall_hp -> fires live on wood walls / furniture (hp 60/30),
    die instantly on bare floor (hp 0) — the wood room + crates are the fuel;
  - doors are walk-through but gas/water-SEALED (permeability 0) -> the pool
    basin holds water and the sealed room holds pressure with a door you can
    wade/walk through.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "levels" / "playground"

# ---------------------------------------------------------------------------
# v2 CSV codes ARE canon material ids (simulation.materials) + SPACE.
# ---------------------------------------------------------------------------
AIR, HULL, WOOD, DOOR, STEEL, GLASS, FURN, SPACE = 0, 1, 2, 3, 4, 5, 6, 9

W, H = 100, 70          # tiles (x right, y down); tilemap shape = (H, W)
PX = 16                 # diffuse px per tile -> 1600x1120 png


# ---------------------------------------------------------------------------
# Tilemap
# ---------------------------------------------------------------------------
def _box(tm, x0, y0, x1, y1, code):
    """Rectangle BORDER (inclusive coords) of ``code``; interior untouched."""
    tm[y0, x0:x1 + 1] = code
    tm[y1, x0:x1 + 1] = code
    tm[y0:y1 + 1, x0] = code
    tm[y0:y1 + 1, x1] = code


def _fill(tm, x0, y0, x1, y1, code):
    tm[y0:y1 + 1, x0:x1 + 1] = code


def build_tilemap() -> np.ndarray:
    tm = np.full((H, W), AIR, dtype=np.int32)

    # Outer space band (2 tiles) + the ship hull shell.
    tm[:2, :] = SPACE
    tm[-2:, :] = SPACE
    tm[:, :2] = SPACE
    tm[:, -2:] = SPACE
    _box(tm, 2, 2, W - 3, H - 3, HULL)          # shell: x 2..97, y 2..67

    # --- WOOD ROOM (fire fuel), NE-ish: x 60..77, y 4..20 ------------------
    _box(tm, 60, 4, 77, 20, WOOD)
    tm[10:14, 60] = DOOR                        # west door, y 10..13
    _fill(tm, 66, 8, 67, 9, WOOD)               # 2x2 fuel pillars
    _fill(tm, 71, 15, 72, 16, WOOD)

    # --- GLASS GALLERY (fragile box), far NE: x 82..96, y 4..20 ------------
    _box(tm, 82, 4, 96, 20, GLASS)
    tm[10:14, 82] = DOOR                        # west door, y 10..13

    # --- SEALED ROOM (pressure play), mid-east: x 60..77, y 26..42 ---------
    # Hull border (never bursts) + a walk-through-but-gas-sealed door + one
    # GLASS pane (the designed relief valve, threshold 1.0) + one WOOD
    # segment (flammable failure point, threshold 2.0).
    _box(tm, 60, 26, 77, 42, HULL)
    tm[26, 66:70] = DOOR                        # north door, x 66..69
    tm[32:35, 77] = GLASS                       # east window pane, y 32..34
    tm[42, 66:69] = WOOD                        # south wood segment, x 66..68

    # --- STEEL BUNKER (the tough room), far mid-east: x 82..96, y 26..42 ---
    _box(tm, 82, 26, 96, 42, STEEL)
    tm[32:36, 82] = DOOR                        # west door, y 32..35

    # --- POOL BASIN (steel tub for U-key water), mid-south: x 60..77 -------
    _box(tm, 60, 48, 77, 66, STEEL)
    tm[48, 66:70] = DOOR                        # north door, x 66..69

    # --- ZOMBIE PEN (glass box, NO door), SE: x 82..96, y 48..66 -----------
    # Glass blocks LOS -> the horde is dormant until a pane pops (grenade the
    # wall). hp 15 / burst 1.0: one nearby blast opens it.
    _box(tm, 82, 48, 96, 66, GLASS)

    # --- BREACH BAY, SW: hull-walled anteroom on the outer south hull ------
    # Its south wall IS the ship hull (y 67) with SPACE beyond: the marked
    # breach target. Door explosive (wall_damage 500) opens it in one;
    # grenades (200 vs hull hp 300) need two on the same tile.
    tm[54, 4:21] = HULL                         # north wall, x 4..20
    tm[54, 10:14] = DOOR                        # north door, x 10..13
    tm[55:67, 20] = HULL                        # east wall, y 55..66

    # --- ARENA furniture: cover clusters + a barricade line ----------------
    _fill(tm, 30, 18, 32, 19, FURN)
    _fill(tm, 40, 26, 42, 27, FURN)
    _fill(tm, 26, 30, 28, 31, FURN)
    _fill(tm, 46, 10, 47, 20, FURN)             # barricade wall of crates

    return tm


# ---------------------------------------------------------------------------
# Diffuse art — flat colour-coded tiles + a faint grid + room labels.
# Render-only; the CSV above is the physics truth.
# ---------------------------------------------------------------------------
PALETTE = {
    SPACE: (8, 8, 16),
    AIR:   (52, 56, 62),      # deck plate (checker partner below)
    HULL:  (112, 118, 130),
    WOOD:  (122, 82, 46),
    DOOR:  (150, 126, 62),
    STEEL: (156, 161, 171),
    GLASS: (120, 178, 198),
    FURN:  (96, 71, 41),
}
AIR_ALT = (47, 51, 57)        # checker partner for the deck
FLOOR_TINTS = (               # (x0, y0, x1, y1, rgb) interior floor washes
    (61, 49, 76, 65, (40, 58, 78)),    # pool basin: blue
    (83, 49, 95, 65, (74, 46, 46)),    # zombie pen: red
    (61, 27, 76, 41, (66, 60, 44)),    # sealed room: amber
)
HAZARD_X0, HAZARD_X1, HAZARD_Y = 4, 19, 67    # breach wall stripe tiles

LABELS = (   # (tile_x, tile_y, text) — drawn at tile centres, big + faint
    (13, 8,  "GRENADE RANGE"),
    (30, 44, "ARENA"),
    (68, 12, "WOOD"),
    (89, 12, "GLASS"),
    (68, 34, "SEALED"),
    (89, 34, "STEEL"),
    (68, 57, "POOL"),
    (89, 57, "PEN"),
    (11, 60, "BREACH"),
)


def build_diffuse(tm: np.ndarray) -> Image.Image:
    img = np.zeros((H * PX, W * PX, 3), dtype=np.uint8)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = PALETTE[code]
            if code == AIR and (tx + ty) % 2:
                c = AIR_ALT
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    # Interior floor tints (identify the special rooms at a glance).
    for (x0, y0, x1, y1, tint) in FLOOR_TINTS:
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                if tm[ty, tx] == AIR:
                    img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = tint
    # Hazard stripes on the breach wall (diagonal yellow/black).
    for tx in range(HAZARD_X0, HAZARD_X1 + 1):
        for py in range(PX):
            for px in range(PX):
                gx, gy = tx * PX + px, HAZARD_Y * PX + py
                img[gy, gx] = (208, 172, 32) if ((px + py) // 4) % 2 else (26, 26, 26)
    # Faint tile grid (1 px, darkened) so distances read on the open floor.
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.uint8)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.uint8)

    pil = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=40)
    except TypeError:            # older PIL: fixed-size default font
        font = ImageFont.load_default()
    for (tx, ty, text) in LABELS:
        cx, cy = tx * PX, ty * PX
        draw.text((cx, cy), text, fill=(200, 204, 210), font=font, anchor="mm")
    return pil


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tm = build_tilemap()
    csv_path = OUT_DIR / "tilemap.csv"
    np.savetxt(csv_path, tm, fmt="%d", delimiter=",")
    png_path = OUT_DIR / "diffuse.png"
    build_diffuse(tm).save(png_path)
    codes = sorted(np.unique(tm).tolist())
    print(f"wrote {csv_path}  ({W}x{H} tiles, codes {codes})")
    print(f"wrote {png_path}  ({W * PX}x{H * PX} px, {PX} px/tile)")


if __name__ == "__main__":
    main()
