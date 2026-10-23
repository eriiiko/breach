r"""tools/make_fire_tuning_level.py — generates levels/fire_tuning/, the
fire-tuning studio (fire session #12 phase 2 part B,
docs/fire_phase2_hud_and_level_2026-09-01.md §B).

Six isolated stations, one hull-sealed hall, so a human can poke each fire
behaviour independently with the F6 tile-inspector HUD
(renderer/hover_readout.py) without one station's radiation (~2-tile reach)
bleeding into the next:

  1. Bonfire stage   — a 2x2 wood-crate cluster (the G1-G4 reference fire).
  2. Spread line     — one wood crate + a kindling fuse line (propagation
                       timing).
  3. Material row    — isolated single-tile wood / furniture / kindling
                       samples, >=6 tiles apart.
  4. Sealed chamber  — a fully-walled box, NO opening at all: one wood
                       crate, O2-starvation must kill its own fire.
  5. Door room       — the same box shape with one door to the hall (the
                       flow/reignition case).
  6. Ambient/space ring boundary (the playground/fire_studio precedent) +
     a small marine roster (2 spawns) so the level is playable.

ONE WRITER, LEVEL_LIB ONLY (CLAUDE.md canonical-systems rule). This tool is
a client, never a second level.toml author:
  - tilemap.csv        -> :func:`level_lib.write_tilemap_csv`
  - [[spawn]]/[[entity]] -> :func:`level_lib.write_managed_blocks` (the
    managed-block family writer every other tool in tree uses — the SAME
    pattern place_playground_vents.py / map_editor.py use)
  - the door entity's span/stamp come from :mod:`door_entity_port` /
    `simulation.entities.door.base_span` (THE canonical span math — never
    re-derived here), exactly like the map editor's own DOOR tool.
  - the level.toml SCALAR HEADER (version/name/tilemap/tile_size_m/diffuse)
    is written once, directly, at folder creation — level_lib does not own
    a "new level" scaffold for these keys (its managed families are
    spawn/light/water/entity/wire/art.bare/art.align/bake; only the
    unrelated `boundary` scalar has a dedicated writer). This mirrors
    tools/map_editor.py's OWN `create_level` bootstrap byte-for-byte in
    spirit (same five keys, same one-time non-managed write) — not a
    parallel invention.
  - diffuse.png is RENDER ART (flat colour-coded tiles, no game-state
    round-trips through it), painted directly with PIL — the "bare
    diffuse" level family levels/fire_studio already ships (level_loader's
    legacy flat `diffuse` key, still first-class; no [bake]/[art.bare]
    block needed). Deliberately NOT the tiled/autotile baker
    (tools/bake_level_art.py): the committed `art/tilesets/greybox`
    manifest predates `MAT_KINDLING` and declares no strip for it
    (materials.py: kindling is "not placed in any shipped level" yet) —
    regenerating that SHARED asset is out of this patch's scope, so this
    level takes the same bare-diffuse path fire_studio already uses rather
    than block station 2/3 on a tileset update.

DETERMINISTIC — no RNG anywhere in this tool (gate: run twice -> byte-
identical outputs).

Run:
    C:/Users/steen/anaconda3/python.exe tools/make_fire_tuning_level.py

Then:
    C:/Users/steen/anaconda3/python.exe tools/lighting_demo.py --level fire_tuning
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_lib  # noqa: E402
from level_loader import EntityInstance, SpawnEntry  # noqa: E402
import door_entity_port  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_DOOR_CLOSED, MAT_FURNITURE,  # noqa: E402
                                  MAT_HULL, MAT_KINDLING, MAT_WOOD)

LEVEL_NAME = "fire_tuning"
DEFAULT_OUT_DIR = ROOT / "levels" / LEVEL_NAME
SPACE_CODE = 9

W, H = 72, 46            # tiles (x right, y down); tilemap shape = (H, W)
PX = 16                  # diffuse px per tile -> 1152x736 png
TILE_SIZE_M = 0.333      # same physical scale as every other shipped level

# ---------------------------------------------------------------------------
# Station coordinates (design doc §B, items 1-5) — every cluster spaced far
# enough apart (>=6 tiles, most much more) that the ~2-tile radiation reach
# can't couple two stations.
# ---------------------------------------------------------------------------
# 1. Bonfire stage: 2x2 wood-crate cluster.
S1_X0, S1_Y0 = 8, 8                        # occupies (8,8)-(9,9)

# 2. Spread line: one wood crate + a kindling fuse line running east.
S2_CRATE_X, S2_CRATE_Y = 22, 8
S2_LINE_X0, S2_LINE_X1, S2_LINE_Y = 23, 28, 8

# 3. Material row: isolated single-tile samples, >=6 tiles apart (design's
# own minimum, hit exactly).
S3_ROW_Y = 8
S3_WOOD_X, S3_FURN_X, S3_KIND_X = 40, 46, 52

# Marine spawns — open hall floor, clear of every fire station (footprint 3,
# the standard marine size — same schema fire_studio's spawns use).
SPAWNS = (("Alpha", 12, 16), ("Bravo", 18, 16))

# 4. Sealed chamber: fully-walled box, NO opening anywhere.
S4_X0, S4_Y0, S4_X1, S4_Y1 = 6, 28, 13, 35
S4_CRATE = (9, 31)

# 5. Door room: same outer box shape, one door in the north wall to the
# hall (the flow/reignition case). Anchor is the door's leftmost tile
# (door.py convention); orientation "h" walks +x.
S5_X0, S5_Y0, S5_X1, S5_Y1 = 24, 28, 31, 35
S5_CRATE = (27, 31)
S5_DOOR_X, S5_DOOR_Y, S5_DOOR_ORIENT, S5_DOOR_LEN_M = 27, 28, "h", 0.666


def scaffold_grid(width: int, height: int) -> np.ndarray:
    """The hull-sealed shell (item 6): a 1-tile SPACE border, a 1-tile
    MAT_HULL ring inside it, MAT_AIR interior — the same shape
    tools/map_editor.py's own `scaffold_grid` and tools/gen_fire_studio.py's
    outer ring use (item 6: "ambient/space ring boundary like the
    playground")."""
    w, h = int(width), int(height)
    g = np.full((h, w), MAT_AIR, dtype=np.int32)
    g[0, :] = g[-1, :] = SPACE_CODE
    g[:, 0] = g[:, -1] = SPACE_CODE
    g[1, 1:w - 1] = MAT_HULL
    g[h - 2, 1:w - 1] = MAT_HULL
    g[1:h - 1, 1] = MAT_HULL
    g[1:h - 1, w - 2] = MAT_HULL
    return g


def _box(tm: np.ndarray, x0: int, y0: int, x1: int, y1: int, code: int) -> None:
    """Rectangle BORDER (inclusive coords) of ``code``; interior untouched."""
    tm[y0, x0:x1 + 1] = code
    tm[y1, x0:x1 + 1] = code
    tm[y0:y1 + 1, x0] = code
    tm[y0:y1 + 1, x1] = code


def build_door_instance() -> EntityInstance:
    """Station 5's door — span/stamp computed through door_entity_port
    (THE canonical span math, `simulation.entities.door.base_span`), never
    re-derived. Closed at load (a human opens it in-game to run the
    flow/reignition case)."""
    fields = {"x": S5_DOOR_X, "y": S5_DOOR_Y, "orientation": S5_DOOR_ORIENT,
             "length_m": S5_DOOR_LEN_M}
    span = door_entity_port.instance_span(fields, TILE_SIZE_M)
    assert len(span) == 2, f"expected a 2-tile door span, got {span}"
    return door_entity_port.build_door_instance(
        S5_DOOR_X, S5_DOOR_Y, S5_DOOR_ORIENT, S5_DOOR_LEN_M, "closed",
        "station5_door")


def build_tilemap(door_instance: EntityInstance) -> np.ndarray:
    """The full fire-tuning tilemap: the scaffold shell + every station."""
    tm = scaffold_grid(W, H)

    # --- 1. Bonfire stage ---------------------------------------------------
    tm[S1_Y0:S1_Y0 + 2, S1_X0:S1_X0 + 2] = MAT_WOOD

    # --- 2. Spread line -------------------------------------------------
    tm[S2_CRATE_Y, S2_CRATE_X] = MAT_WOOD
    tm[S2_LINE_Y, S2_LINE_X0:S2_LINE_X1 + 1] = MAT_KINDLING

    # --- 3. Material row --------------------------------------------------
    tm[S3_ROW_Y, S3_WOOD_X] = MAT_WOOD
    tm[S3_ROW_Y, S3_FURN_X] = MAT_FURNITURE
    tm[S3_ROW_Y, S3_KIND_X] = MAT_KINDLING

    # --- 4. Sealed chamber: fully-walled, no opening ------------------------
    _box(tm, S4_X0, S4_Y0, S4_X1, S4_Y1, MAT_HULL)
    tm[S4_CRATE[1], S4_CRATE[0]] = MAT_WOOD

    # --- 5. Door room: same box shape, door span stamped MAT_DOOR_CLOSED ----
    _box(tm, S5_X0, S5_Y0, S5_X1, S5_Y1, MAT_HULL)
    tm[S5_CRATE[1], S5_CRATE[0]] = MAT_WOOD
    span = door_entity_port.instance_span(door_instance.fields, TILE_SIZE_M)
    stamp = door_entity_port.stamp_value_for(door_instance.fields["initial_state"])
    for (fy, fx) in span:
        tm[fy, fx] = stamp

    return tm


# ---------------------------------------------------------------------------
# Diffuse art — flat colour-coded tiles + a faint grid + station labels.
# RENDER-ONLY; the CSV above is the physics truth (same split as
# tools/gen_fire_studio.py's diffuse painter).
# ---------------------------------------------------------------------------
PALETTE = {
    SPACE_CODE:      (6, 6, 12),
    MAT_AIR:         (34, 36, 42),
    MAT_HULL:        (96, 102, 116),
    MAT_WOOD:        (122, 82, 46),
    MAT_DOOR_CLOSED: (150, 126, 62),
    MAT_FURNITURE:   (96, 71, 41),
    MAT_KINDLING:    (176, 140, 60),
}
AIR_ALT = (30, 32, 38)          # checker partner for the deck

LABELS = (   # (tile_x, tile_y, text) — drawn at tile centres, faint
    (9, 6,   "1 BONFIRE"),
    (25, 6,  "2 SPREAD LINE"),
    (46, 6,  "3 MATERIAL ROW"),
    (15, 14, "SPAWNS"),
    (9, 26,  "4 SEALED"),
    (27, 26, "5 DOOR ROOM"),
)


def build_diffuse(tm: np.ndarray) -> Image.Image:
    img = np.zeros((H * PX, W * PX, 3), dtype=np.uint8)
    for ty in range(H):
        for tx in range(W):
            code = int(tm[ty, tx])
            c = PALETTE.get(code, (200, 40, 200))     # loud magenta = bug
            if code == MAT_AIR and (tx + ty) % 2:
                c = AIR_ALT
            img[ty * PX:(ty + 1) * PX, tx * PX:(tx + 1) * PX] = c
    # Faint tile grid (1 px, darkened) so distances read on the open floor.
    img[::PX, :, :] = (img[::PX, :, :] * 0.82).astype(np.uint8)
    img[:, ::PX, :] = (img[:, ::PX, :] * 0.82).astype(np.uint8)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:            # older PIL: fixed-size default font
        font = ImageFont.load_default()
    for (tx, ty, text) in LABELS:
        cx, cy = tx * PX, ty * PX
        draw.text((cx, cy), text, fill=(160, 166, 174), font=font, anchor="mm")
    return pil


# ---------------------------------------------------------------------------
# level.toml scalar header — written ONCE, directly (see module docstring:
# level_lib owns no "new level" scaffold for these five keys; this mirrors
# tools/map_editor.py::create_level's own one-time bootstrap).
# ---------------------------------------------------------------------------
def _header_toml(display_name: str) -> str:
    return (
        "# GENERATED by tools/make_fire_tuning_level.py — tweak the\n"
        "# constants there and re-run; do NOT hand-edit ([[spawn]]/\n"
        "# [[entity]] below ARE level_lib-managed and will be overwritten).\n"
        "# docs/fire_phase2_hud_and_level_2026-09-01.md SS B. Deterministic,\n"
        "# no RNG.\n"
        'version = "2"\n'
        f'name = "{display_name}"\n'
        "\n"
        "# v2 codes ARE canon material ids (src/simulation/materials.py):\n"
        "#   0=air 1=hull 2=wood 3=door 4=steel 5=glass 6=furniture\n"
        "#   7=door_closed 8=kindling 9=SPACE\n"
        'tilemap = "tilemap.csv"\n'
        f"tile_size_m = {TILE_SIZE_M}\n"
        "\n"
        "# Flat colour-coded, station-labelled map (generated; render-only;\n"
        "# the bare-diffuse family levels/fire_studio also ships — see the\n"
        "# module docstring for why this level skips the tiled/autotile\n"
        "# baker).\n"
        'diffuse = "diffuse.png"\n'
    )


def main(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    # Deterministic full regen (2026-08-22 issue #47 convention: a caller
    # that wants tmp-dir determinism-testing passes a different out_dir —
    # never mutates a level this tool doesn't own).
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "level.toml").write_bytes(
        _header_toml("Fire Tuning").encode("utf-8"))

    door_instance = build_door_instance()
    tm = build_tilemap(door_instance)
    level_lib.write_tilemap_csv(out_dir, tm, csv_bak=False)

    build_diffuse(tm).save(out_dir / "diffuse.png")

    spawns = [SpawnEntry(name=name, team=0, x=x, y=y, footprint=3)
             for (name, x, y) in SPAWNS]
    level_lib.write_managed_blocks(
        out_dir / "level.toml",
        {"spawn": lambda nl: level_lib.format_spawn_lines(spawns, nl),
         "entity": lambda nl: level_lib.format_entity_lines([door_instance], nl)},
        write_bak=False)

    codes = sorted(np.unique(tm).tolist())
    print(f"wrote {out_dir}  ({W}x{H} tiles, codes {codes})")
    print(f"  1 bonfire        : ({S1_X0},{S1_Y0})-({S1_X0 + 1},{S1_Y0 + 1})")
    print(f"  2 spread line    : crate ({S2_CRATE_X},{S2_CRATE_Y}), kindling "
          f"x={S2_LINE_X0}..{S2_LINE_X1} y={S2_LINE_Y}")
    print(f"  3 material row   : y={S3_ROW_Y} wood x={S3_WOOD_X} furniture "
          f"x={S3_FURN_X} kindling x={S3_KIND_X}")
    print(f"  4 sealed chamber : box ({S4_X0},{S4_Y0})-({S4_X1},{S4_Y1}), "
          f"crate {S4_CRATE}")
    print(f"  5 door room      : box ({S5_X0},{S5_Y0})-({S5_X1},{S5_Y1}), "
          f"crate {S5_CRATE}, door span "
          f"{door_entity_port.instance_span(door_instance.fields, TILE_SIZE_M)}")
    print(f"  spawns           : {[(n, x, y) for (n, x, y) in SPAWNS]}")


if __name__ == "__main__":
    main()
