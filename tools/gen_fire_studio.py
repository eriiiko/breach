"""Generate the FIRE STUDIO level — the B2 "dark room" fire/smoke showcase.

Fire & Heat Beauty arc, beat B2 patch P1 (design
docs/fire_b2_smoke_honesty_design_2026-07-21.md §2). A single hull-sealed box
built to show off the physical gas medium (P2-P4) + the rotating beacon raking
through drifting smoke: a near-dark main hall (two lamps + one sweeping beacon +
a wood crate cluster to burn), a SEALED side room off one door (the
O2-starvation stage — ignite, shut the door, watch it choke and blacken), a
walled corridor with a single lamp at the far end (clean beam-through-haze), a
water pool in a hall corner (steam context + wet-floor look), and three marine
spawns (scale + flashlight carriers + the 3D-marine lighting showcase).

DETERMINISTIC — fixed content, NO RNG (gate: run twice -> byte-identical). This
tool writes the WHOLE level folder (reproducible + reviewable): tilemap.csv,
diffuse.png, water_init.npy, and level.toml. Compose EXISTING level-schema
features only (levels-w1 [[light]] incl. kind="beacon"; the [[entity]] door;
the [water] .npy carrier; unit spawns) — the beacon + door + water blocks are
hand-composed into level.toml here rather than extending the schema.

    C:/Users/steen/anaconda3/python.exe tools/gen_fire_studio.py

Launch (the studio session):
    C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py --level fire_studio
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "levels" / "fire_studio"

# ---------------------------------------------------------------------------
# v2 CSV codes ARE canon material ids (simulation.materials) + SPACE.
# ---------------------------------------------------------------------------
AIR, HULL, WOOD, DOOR, STEEL, GLASS, FURN, SPACE = 0, 1, 2, 3, 4, 5, 6, 9

W, H = 48, 32           # tiles (x right, y down); tilemap shape = (H, W)
PX = 16                 # diffuse px per tile -> 768x512 png
TILE_SIZE_M = 0.333     # same physical scale as the ship

# Water pool: a few tiles in the hall's lower-right corner (metres of standing
# water). int32 Q16.16 carrier, shape == tilemap (level_loader [water]).
FP_ONE = 1 << 16
POOL_DEPTH_M = 0.4
POOL_X0, POOL_X1, POOL_Y0, POOL_Y1 = 32, 35, 26, 28


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

    # Outer SPACE ring (1 tile) + the hull shell (the sealed box in vacuum).
    tm[0, :] = SPACE
    tm[-1, :] = SPACE
    tm[:, 0] = SPACE
    tm[:, -1] = SPACE
    _box(tm, 1, 1, W - 2, H - 2, HULL)          # shell: x 1..46, y 1..30

    # --- SEALED SIDE ROOM (top-right, 8x8 interior) --------------------------
    # Left wall (hull col x=37) + bottom wall (hull row y=10); top + right are
    # the shell. One DOOR gap in the left wall (x=37, y=5..6) — the [[entity]]
    # door seals it closed at load. Wood furniture inside = the O2 fuel.
    tm[2:11, 37] = HULL                          # left wall, y 2..10
    tm[10, 37:47] = HULL                         # bottom wall, x 37..46
    tm[5:7, 37] = AIR                            # doorway (door entity spans it)
    _fill(tm, 41, 5, 42, 6, WOOD)               # 2x2 wood furniture (fuel)

    # --- CORRIDOR (bottom-left, walled tube, 16x3 interior) ------------------
    # Interior floor x=3..18, y=26..28. Walled top/bottom/left/right; ONE
    # entrance gap in the top wall connects it up into the hall. Single lamp at
    # the far (right) end -> clean beam-through-haze compositions.
    tm[25, 2:20] = HULL                          # top wall, x 2..19
    tm[29, 2:20] = HULL                          # bottom wall, x 2..19
    tm[25:30, 2] = HULL                          # left end wall, y 25..29
    tm[25:30, 19] = HULL                         # right end wall, y 25..29
    tm[25, 4:6] = AIR                            # entrance gap to the hall

    # --- MAIN HALL crate cluster (mid-left) — the PRIMARY fire ---------------
    # A 3x3 wood/furniture cluster: solid wood crates (cast beam shadows) +
    # furniture. Mid-left so the open hall + spawns stay clear.
    _fill(tm, 6, 9, 7, 10, WOOD)                # 2x2 wood crates
    tm[9, 8] = FURN
    tm[10, 8] = FURN
    tm[11, 6:9] = FURN                          # furniture skirt

    return tm


def build_water() -> np.ndarray:
    """Standing water in the hall's lower-right corner (int32 Q16.16 metres)."""
    water = np.zeros((H, W), dtype=np.int32)
    water[POOL_Y0:POOL_Y1 + 1, POOL_X0:POOL_X1 + 1] = int(round(POOL_DEPTH_M * FP_ONE))
    return water


# ---------------------------------------------------------------------------
# Diffuse art — flat colour-coded tiles + a faint grid + a few labels.
# Render-only; the CSV above is the physics truth.
# ---------------------------------------------------------------------------
PALETTE = {
    SPACE: (6, 6, 12),
    AIR:   (34, 36, 42),      # near-dark deck plate (checker partner below)
    HULL:  (96, 102, 116),
    WOOD:  (122, 82, 46),
    DOOR:  (150, 126, 62),
    STEEL: (156, 161, 171),
    GLASS: (120, 178, 198),
    FURN:  (96, 71, 41),
}
AIR_ALT = (30, 32, 38)        # checker partner for the deck

FLOOR_TINTS = (               # (x0, y0, x1, y1, rgb) interior floor washes
    (38, 2, 45, 9, (58, 46, 40)),      # side room: warm amber (the burn stage)
    (3, 26, 18, 28, (40, 46, 58)),     # corridor: cool
    (POOL_X0, POOL_Y0, POOL_X1, POOL_Y1, (34, 52, 74)),   # water pool: blue
)

LABELS = (   # (tile_x, tile_y, text) — drawn at tile centres, faint
    (18, 6,  "MAIN HALL"),
    (41, 4,  "SIDE"),
    (10, 27, "CORRIDOR"),
    (33, 27, "POOL"),
    (7, 10,  "FUEL"),
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
    # Faint tile grid (1 px, darkened) so distances read on the open floor.
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.uint8)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.uint8)

    pil = Image.fromarray(img)          # (H, W, 3) uint8 -> RGB (mode inferred)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:            # older PIL: fixed-size default font
        font = ImageFont.load_default()
    for (tx, ty, text) in LABELS:
        cx, cy = tx * PX, ty * PX
        draw.text((cx, cy), text, fill=(150, 156, 164), font=font, anchor="mm")
    return pil


# ---------------------------------------------------------------------------
# level.toml — hand-composed from EXISTING schema features (beacon [[light]],
# [[entity]] door, [water] carrier, unit spawns). Written verbatim (fixed
# content) so the generator stays deterministic.
# ---------------------------------------------------------------------------
LEVEL_TOML = f"""# FIRE STUDIO — the B2 "dark room" fire/smoke showcase (design
# docs/fire_b2_smoke_honesty_design_2026-07-21.md §2). GENERATED by
# tools/gen_fire_studio.py — tweak the constants there and re-run; do NOT
# hand-edit (the tool overwrites the whole folder). Deterministic, no RNG.
#
# A single hull-sealed box: near-dark main hall (two lamps + one rotating
# beacon + a wood crate cluster to burn), a SEALED side room off one door
# (the O2-starvation stage), a walled corridor with a lamp at the far end,
# a water pool in a hall corner, and three marine spawns.
#
# Launch:  python tools/lighting_demo.py --level fire_studio
#   (or set [display] level = "fire_studio" in config.toml)

version = "2"
name = "Fire Studio"

# v2 codes ARE canon material ids (src/simulation/materials.py):
#   0=air 1=hull 2=wood 3=door 4=steel 5=glass 6=furniture 9=SPACE
tilemap = "tilemap.csv"
tile_size_m = {TILE_SIZE_M}

# Flat colour-coded, room-labelled map (generated; render-only).
diffuse = "diffuse.png"

# Standing water in the hall's lower-right corner (steam context + wet floor).
[water]
depth_map = "water_init.npy"

# ---------------------------------------------------------------------------
# Marine squad in the main hall (visual scale + flashlight carriers + the
# 3D-marine lighting showcase). team 0 = marine; (x, y) = top-left of the
# 3x3 footprint on open hall floor.
# ---------------------------------------------------------------------------
[[spawn]]
name = "Alpha"
team = 0
x = 10
y = 16
footprint = 3

[[spawn]]
name = "Bravo"
team = 0
x = 16
y = 16
footprint = 3

[[spawn]]
name = "Cobra"
team = 0
x = 22
y = 16
footprint = 3

# ---------------------------------------------------------------------------
# Lights (levels-w1 [[light]] machinery, src/level_lights.py). Two warm hall
# lamps (the demo-side toggleable group), ONE rotating beacon mounted
# center-hall (its beam sweeps 360 deg through drifting gas), and one cool
# lamp at the far end of the corridor. color = [r, g, b] 0-255 ints.
# RENDER-ONLY (P4 §2.2): these never write the synced heat channel.
# ---------------------------------------------------------------------------
[[light]]
pos = [8.5, 4.5]
color = [255, 214, 170]
intensity = 1.2
range = 16.0
kind = "static"

[[light]]
pos = [30.5, 4.5]
color = [255, 214, 170]
intensity = 1.2
range = 16.0
kind = "static"

[[light]]
pos = [17.5, 27.5]
color = [200, 220, 255]
intensity = 1.0
range = 14.0
kind = "static"

# The rotating beacon — center-hall, sweeping amber beam. Angle is a pure
# function of the sim tick (freezes on pause, replays exactly).
[[light]]
pos = [18.5, 12.5]
color = [255, 92, 40]
intensity = 3.0
range = 20.0
kind = "beacon"
period_s = 3.0
beam_deg = 28.0
phase = 0.0

# ---------------------------------------------------------------------------
# The sealed side room's door ([[entity]] door — src/simulation/door_system.py
# DoorRuntime; the CSV span is AIR, the load stamp seals it closed). Toggle it
# through the door system (the demo's C key), NOT tile paint. orientation "v"
# spans down a column; length_m 0.666 / tile 0.333 = 2 tiles (37, y=5..6).
# ---------------------------------------------------------------------------
[[entity]]
id = "side_door"
class = "door"
x = 37
y = 5
orientation = "v"
length_m = 0.666
initial_state = "closed"
"""


def main(out_dir=OUT_DIR) -> None:
    # out_dir parameter (2026-08-22, issue #47): lets the determinism test
    # regenerate into a tmp dir instead of mutating the committed level.
    out_dir.mkdir(parents=True, exist_ok=True)
    tm = build_tilemap()

    csv_path = out_dir / "tilemap.csv"
    np.savetxt(csv_path, tm, fmt="%d", delimiter=",")

    water_path = out_dir / "water_init.npy"
    np.save(water_path, build_water())

    png_path = out_dir / "diffuse.png"
    build_diffuse(tm).save(png_path)

    toml_path = out_dir / "level.toml"
    toml_path.write_text(LEVEL_TOML, encoding="utf-8")

    codes = sorted(np.unique(tm).tolist())
    print(f"wrote {csv_path}  ({W}x{H} tiles, codes {codes})")
    print(f"wrote {water_path}  (pool {POOL_X0}..{POOL_X1} x {POOL_Y0}..{POOL_Y1}"
          f" @ {POOL_DEPTH_M} m)")
    print(f"wrote {png_path}  ({W * PX}x{H * PX} px, {PX} px/tile)")
    print(f"wrote {toml_path}")


if __name__ == "__main__":
    main()
