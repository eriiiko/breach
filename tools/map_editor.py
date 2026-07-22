r"""tools/map_editor.py — the tiled-path map editor (engine/15 §5, P3).

Standalone pyray tool for TILED levels (§0: paint materials first, art is
baked from a tileset). It owns `tilemap.csv` + the `[[spawn]]` tables and
shows a LIVE BAKED PREVIEW: the whole grid is baked once on load
(`bake_level_art.bake_full`) and every stroke re-bakes only its dirty
rectangle EXPANDED BY 1 TILE on all sides (a stroke changes the edge16 mask
of every neighbouring wall tile) via `bake_region` — sub-second strokes,
save-time full bakes. Painted levels (no `[bake]` block) are refused: they
belong to tools/align_level_art.py; run tools/bake_level_art.py once to
adopt a bare tilemap into the tiled path.

Run:
    python tools/map_editor.py <level_name> [--auto]
    python tools/map_editor.py new <level_name> --size 48x32 \
        [--tileset art/tilesets/greybox] [--px-per-tile 64] [--seed 0]

`new` scaffolds levels/<name>/ (1-tile SPACE border, MAT_HULL ring, MAT_AIR
interior + minimal v2 level.toml), full-bakes it (so it is immediately
loadable) and opens it. No in-UI text input in v1 — names and sizes come
from the CLI. --auto renders ~90 frames (exercising both view modes), saves
a screenshot PNG into the OS temp dir and exits 0 — raylib has no
input-injection API, so the interactive paths are smoke-only; the pure
helpers are unit-tested in tests/test_map_editor_tool.py.

The window is organised as panes (Arc C, editor doc §8): a top bar, a left
tool rail (the mode list), the central canvas (the live baked map), a right
palette + inspector column, and a bottom status bar. The map draws only
inside the canvas pane; keyboard shortcuts are unchanged (keyboard-first
survives the panes). Pane geometry is pure in tools/editor_layout.py. The
palette pane is tabbed (Arc C1): MATERIAL (unchanged) and ENTITY — the
latter generated from the live entity registry (tools/entity_editor_ui.py,
falling back to the last-good entity_registry.json + a red status-bar
banner on import failure, canon engine/16 §1). Clicking an ENTITY chip
shows that class's default field template in the inspector; LIGHT mode's
hovered marker shows its live instance fields the same way.

Controls (the tool rail + inspector show the active mode's line):

  Any mode:
    TAB / Shift+TAB      - cycle mode PAINT -> ROOM -> CORRIDOR -> DOOR ->
                           SPAWN -> LIGHT -> WATER -> ENTITY -> SELECT ->
                           WAND -> ZONE -> AIR -> PROPS -> WIRE; F1..F12
                           jump straight to a mode EXCEPT F5 (SPAWN, PROPS
                           and WIRE have no F-key left — TAB or a tool-rail
                           click only, Arc C5/C6/C7)
    F5                   - PLAY: save the FULL live state (unsaved edits
                           included) to levels/_editor_scratch/<name>/,
                           reusing the last bake when the material grid is
                           unchanged since it was last written to disk, and
                           launch the game against it via sys.executable
                           (never bare python); the scratch folder is
                           deleted when that process exits AND on editor
                           quit (Arc C7)
    L                    - toggle the LOGIC overlay's show-all (default:
                           only wires touching the SELECT-mode selection,
                           Arc C6)
    0-8, 9               - select material (palette GENERATED from
                           MATERIAL_NAMES at launch — key = material id;
                           9 = SPACE; ids past 8 are eyedropper-only)
    V                    - toggle baked preview <-> material-colour view
    N                    - baked view: toggle diffuse <-> normal map
    G                    - toggle grid lines
    Mouse wheel          - zoom (around cursor)
    Middle-drag / WASD   - pan (arrows pan too)
    Ctrl+Z               - undo · Ctrl+Y / Ctrl+Shift+Z - redo. ONE global
                           transaction log across ALL domains (Arc C2): undo
                           rewinds the most-recent action regardless of the
                           current mode (a compound op — e.g. a door's grid
                           stamp + entity — undoes atomically). Undoing back to
                           the saved state clears the unsaved dot.
    Ctrl+S               - SAVE: tilemap.csv + [[spawn]] + lights (whichever
                           of [[light]] / [[entity]] the level already used —
                           Arc C1, never a forced migration) +
                           [water]/water_init.npy writeback + [art]/[bake]
                           blocks (all .bak once per session) + full bake
                           at the recorded px_per_tile
    Esc                  - quit (pressed twice if there are unsaved edits)

  PAINT:
    Left click/drag      - paint the selected id (right = erase to AIR)
    Shift+click          - line tool: anchor -> cursor (chains a polyline)
    I                    - eyedropper
    [ / ]                - brush size 1..9

  ROOM:   drag LMB       - wall perimeter (selected wall material) + AIR
                           interior; existing wall-family tiles on the
                           perimeter are SHARED (kept), never doubled;
                           RMB cancels the drag
  CORRIDOR: drag LMB     - AIR swath of width w (default 3, +/- to change)
                           cut through solid/space; sides bordering non-AIR
                           get lined with the selected wall material
                           (existing walls shared); RMB cancels
  DOOR (Arc C3):  click LMB - place a `door` entity anchored on a STRAIGHT
                           wall run (corners/T/ends/isolated tiles refused,
                           hover shows green/red); default span 1.0 m,
                           drag to resize (extends toward the cursor,
                           clipped to the wall run — never spills onto
                           AIR/SPACE or into another door); O toggles
                           placing open <-> closed (default closed).
                           MAT_DOOR_CLOSED (closed) / MAT_AIR (open) is
                           stamped across the span IMMEDIATELY, one
                           compound undo transaction with the entity add;
                           a span narrower than the default marine
                           footprint (3 tiles) warns, never blocks.
                           RMB cancels an open drag.
  SPAWN:  LMB            - place a spawn (empty spot) or drag an existing
                           marker; RMB deletes; T toggles the team of the
                           hovered marker (or, off-marker, the placement
                           team); markers draw as team-coloured circles
  LIGHT:  LMB            - place a `light` entity (tile-center snapped) or
                           drag an existing marker; RMB deletes; B toggles
                           static <-> beacon (hovered marker, else the
                           placement kind); C / Shift+C cycles the colour
                           preset (hovered, else placement); on the hovered
                           light, R range, E intensity, and — beacons only —
                           P period, X beam width, H phase (Shift = down).
                           Markers: colour dot + range ring; beacons sweep
                           a beam wedge on the EDITOR's clock (the editor
                           is not the sim — in-game the angle is a pure
                           function of the sim tick, src/level_lights.py)
  WATER:  LMB            - bucket-fill the enclosed region under the cursor
                           to the current depth (default 1.0 m, -/= steps
                           0.1 m); RMB fills the region to dry. The fill is
                           4-connected over tiles that are neither
                           solid-for-water (sim-exact: material
                           permeability <= 0, the mass-sink boundary —
                           NEVER the tileset wall groups) nor SPACE
                           (vacuum bounds a fill exactly like glass);
                           starting on SPACE or a solid is refused. Depth
                           quantizes to int32 Q16.16 at fill time
                           (water_fixed.quantize); wet tiles draw as a blue
                           overlay, alpha by depth. Depths > 1.5 m get the
                           deep-tank hint (drains may flash-boil — by
                           design). Paint a glass box, fill it: that is an
                           aquarium.
  ENTITY (Arc C3):  LMB    - place ONE instance of the class selected in the
                           ENTITY palette tab at the cursor tile. Field
                           sensors (pressure/smoke/water_depth/o2 = AIR
                           family, temperature/fire = BODY family) get a
                           click+drag gesture: the mount is the anchor,
                           dragging sets the AIR family's sample_dx/dy
                           facing (rendered as a small arrow); the sampled
                           tile is refused (no placement, no transaction)
                           if it is solid or SPACE at release — BODY-family
                           sensors always sample their own mount, no drag
                           needed. Everything else (logic nodes, pump,
                           button/terminal, clock, sensor_motion) places
                           via ONE `CollectionOp("entities")` — no grid
                           stamp. DOOR and LIGHT keep their own dedicated
                           modes (F4/F6); selecting either here just points
                           back at that tool. A class needing a field this
                           tool can't fill (a zone's zone_id — C5's paint
                           tool) is refused with a status message.
  SELECT (Arc C4): see entity_selection.py — box/shift-click select, move,
                           delete (Delete; a selected zone instance PROMPTS
                           to also clear its paint — Arc C5), tag-assign,
                           inspector field edit, clump copy/paste.
  WAND (Arc C5):  LMB      - enclosure-fill the region under the cursor
                           (bounded by walls, the WATER fill's own
                           boundary) to the MATERIAL palette's selected id
                           (9/SPACE = vacuum); RMB same-code-selects the
                           contiguous run under the cursor and paints it
                           the same way. One `GridCellsOp("material")`
                           transaction per paint.
  ZONE (Arc C5):  LMB      - enclosure-fill paints the current TARGET zone
                           id into `zones.npy` (a brand-new id also mints
                           its `breach_site`/`extraction_zone` [[entity]]
                           instance, one compound transaction); RMB
                           same-code-selects the id under the cursor and
                           clears its paint; I eyedrops the id/class under
                           the cursor as the new target; [ / ] cycle the
                           target among "new" + every existing id; C
                           cycles the class a NEW id authors. The status
                           bar's validator slot runs the loader's OWN §5
                           binding rules live.
  AIR (Arc C5):  LMB       - enclosure-fill seeds `air_init.npy` to the
                           current pressure over a SEALED region only —
                           the hull-leak validator refuses (paints
                           NOTHING) a fill that reaches the map border;
                           RMB fills to 0 atm (a depressurized start);
                           -/= step the target pressure 0.1 atm.
  PROPS (Arc C5):  B       - toggle the level's `boundary` field
                           space <-> ambient — format + level_lib
                           writeback only (canon §7); no canvas edits, no
                           physics change here.
  WIRE (Arc C6):  LMB      - the two-click wire tool, NEVER a drag: click a
                           source (resolves its output signal — `[`/`]`
                           cycle when it has more than one; default the
                           primary non-alive one), navigate freely (pan/
                           zoom/scroll stay live), click a target (resolves
                           its input the same way; a single-input target
                           commits immediately, multiple need `[`/`]` +
                           Enter). T cycles an alternate TAG target
                           (`tag:name.input`) once a source is picked. Esc
                           cancels the pending wire. With nothing pending,
                           LMB on an existing wire (line or tag badge)
                           selects it — Delete removes it, RMB clears the
                           selection. Positionless entities (logic nodes,
                           zone instances) get a deterministic clickable
                           tile (a zone anchors at its painted centroid;
                           everything else auto-lays-out in a strip past
                           the grid's right edge) — see node markers, drawn
                           every mode. The LOGIC overlay (also every mode)
                           draws every wire touching the current SELECT
                           selection by default; L toggles show-all. A
                           tag-target wire draws as ONE compact badge near
                           its source, never fanned to every member.

Undo — the transaction log (Arc C2, design docs/arc_c_c2_undo_design_
2026-07-22.md): the four per-domain rings are GONE, replaced by ONE global
history of compound operations in tools/undo_log.py. Every mutating gesture
routes through the log's builder seam (begin / snapshot_grid / snapshot_coll /
commit); a grid edit stores a delta (changed cells only), a spawn/light/entity
edit stores a whole-list deep-copy snapshot, and a compound action (C3's door)
stores both in ONE atomic transaction. Ctrl+Z / Ctrl+Y walk a linear cursor;
the unsaved dot is a cursor-vs-saved comparison (undo to the saved position
clears it). Later Arc C patches register their op classes onto the same log.

Spawn/light/water writeback (reported design call, absorbed into level_lib
in Arc A2 — entity doc §3c: level_lib is THE single writer and this editor
is a client): the `[[spawn]]` array-of-tables and the `[water]` table are
MANAGED BLOCKS — on save every existing table is removed and the editor's
state is written back at the position of the first one (or EOF), all
families in ONE atomic temp+rename write (a crash mid-save cannot tear
level.toml). Every byte OUTSIDE the managed tables is preserved (comments
inside individual tables are not). level.toml gets ONE .bak per session,
carrying the pre-session bytes. water_init.npy carries its OWN
once-per-session .bak (pre-session bytes, only when the file predates the
session). On save the water grid is masked against the CURRENT materials
(zeroed on solid/SPACE, count reported) — a wall painted over a pool never
saves hidden depth.

LIGHT is a registry entity now (Arc C1, Amendment A1 — the bespoke
level_lib.write_lights path is DELETED): tools/light_entity_port.py decides
ONCE at load whether a level's lights round-trip through the legacy
`[[light]]` family or the `[[entity]]` family, and never switches a level
between them on save (editor doc §6 / Erik ruling 2 — no forced migration).
A level with no lights at all also saves through `[[entity]]`, since the
old writer no longer exists to fall back to.

Scope limitation on record (P4): the editor refuses levels without a [bake]
block, so the vessel/playground lamp `light` entries are loader-consumed
but hand-edited — LIGHT mode authors tiled-path levels only. WATER mode
inherits the same scope (tiled-path levels only); painted levels take a
hand-authored water_init.npy.
"""
from __future__ import annotations

import argparse
import colorsys
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

# Make project modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import pyray as rl
from PIL import Image
from pyray import ffi

import level_lib
import level_loader
# The write surface moved to level_lib in Arc A2 (entity doc §3c: ONE writer
# implementation, ever); these names stay importable from map_editor for the
# pre-A2 callers/tests.
from level_lib import (WATER_FILENAME, color_255, format_light_lines,  # noqa: F401
                       format_spawn_lines, format_water_lines, write_lights,
                       write_spawns, write_water)
from level_loader import (SPACE_CODE, ZONE_CLASSES, BOUNDARY_MODES,
                          EntityInstance, SpawnEntry)
from level_lights import beacon_angle
from simulation import water_fixed
from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED,
                                  MAT_HULL, MATERIAL_NAMES, MaterialTable)
from bake_level_art import (BIT_E, BIT_N, BIT_S, BIT_W, DEFAULT_PX_PER_TILE,
                            DEFAULT_TILESET, bake_full, bake_level,
                            bake_region, edge16_mask, load_tileset)
from level_edit_common import (BRUSH_MAX, BRUSH_MIN,
                               art_px_to_tile, brush_rect, build_palette,
                               enclosure_fill_region, line_tiles, paint_tiles,
                               region_touches_border, same_code_region,
                               save_tilemap_csv)
from editor_layout import compute_panes, fit_camera, screen_from_world
# Arc C2 — the single transaction-log undo (design
# docs/arc_c_c2_undo_design_2026-07-22.md) REPLACES the four per-domain rings
# (UndoRing/SpawnRing) with ONE global history of compound operations, adds
# redo, and makes the unsaved dot position-accurate. Every op class joins the
# same log through its builder seam; later Arc C patches register theirs.
import undo_log
# Arc C1 — registry-driven palette/inspector + the LIGHT entity port (canon
# engine/16 §1/§2; editor doc §3 pillar 1). LIGHT's bespoke writeback
# (level_lib.write_lights) is DELETED (Amendment A1): light_entity_port
# chooses the level_lib family a level's lights round-trip through.
import entity_editor_ui
import light_entity_port
# Arc C3 — placement tools: DOOR (span drag + immediate MAT_DOOR_CLOSED
# stamp) and sensor placement (sample-tile arrow + solid refusal) each get a
# pure, headless bridge module so the span/family logic is reused, not
# reimplemented, from simulation.entities (read-only imports there).
import door_entity_port
import sensor_entity_port
# Arc C4 — unified entity selection (box/shift/by-class) + move/delete/
# tag-assign/inspector-field-edit/clump-copy-paste. The pure logic (what
# C3 left as "no select/move/delete UX for doors/sensors/generic
# placements") lives in entity_selection.py so it is unit-testable without
# raylib; the SELECT mode below is the thin interactive shell over it.
import entity_selection
# Arc C5 — magic wand (enclosure fill / same-code select, imported above
# from level_edit_common) + zone paint (zone_entity_port: the binding
# validators are level_loader's OWN §5 rules, run live) + air_init paint
# (air_paint: the hull-leak validator) + the level-properties PROPS mode
# (boundary field, format+loader-only — see level_lib.write_boundary_field).
import air_paint
import zone_entity_port
# Arc C6 — the two-click wire tool + LOGIC overlay + tag badges + the
# positionless-entity canvas layout (logic nodes / zone instances have no
# x/y — C3/C4/C5's own punted gap). All pure/headless logic lives in
# wire_tool.py; the WIRE mode below is the thin interactive shell.
import wire_tool
# Arc C7 — play-from-editor (F5): scratch-level save + subprocess launch +
# cleanup, all in the pure/headless tools/play_scratch.py (no raylib import),
# reusing the SAME level_lib/light_entity_port/bake_level_art functions
# Ctrl+S already calls — see its module docstring.
import play_scratch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ("PAINT", "ROOM", "CORRIDOR", "DOOR", "SPAWN", "LIGHT", "WATER",
        "ENTITY", "SELECT", "WAND", "ZONE", "AIR", "PROPS", "WIRE")
# F1..F12 jump straight to the first 12 modes (raylib defines no F13+); the
# 13th/14th (PROPS, Arc C5; WIRE, Arc C6 — no F-key left) are reachable via
# TAB/Shift+TAB cycle and a tool-rail click, same as every other mode.
MAX_MODE_FKEYS = 12
# Arc C7: F5 is reserved GLOBALLY for PLAY (editor doc §8) — it can no
# longer double as a mode-jump key. F5 was SPAWN's slot (MODES index 4,
# since F-key = F1 + index); SPAWN drops to TAB/Shift+TAB cycle + a
# tool-rail click only, joining PROPS/WIRE (which ran out of F-keys first,
# C5/C6) — every OTHER mode's F-key binding is unchanged.
PLAY_FKEY_INDEX = MODES.index("SPAWN")
DEFAULT_CORRIDOR_WIDTH = 3
CORRIDOR_MIN, CORRIDOR_MAX = 1, 9
SCAFFOLD_MIN = 5                 # SPACE border + hull ring + 1 interior tile
DEFAULT_FOOTPRINT = 3            # SpawnEntry default (level_loader)

TEAM_MARINE, TEAM_ZOMBIE = 0, 1  # SpawnEntry.team: 0 = marine, 1 = zombie
TEAM_NAMES = {TEAM_MARINE: "marine", TEAM_ZOMBIE: "zombie"}
TEAM_COLORS = {TEAM_MARINE: (90, 190, 255), TEAM_ZOMBIE: (225, 70, 60)}

# ---- LIGHT mode (P4 §2.4) --------------------------------------------------
# Keys from P3's reported free set (L, B, C, P, R, X, Y, H, E): B = kind
# toggle (pinned by the design doc), C = colour preset, R = range,
# E = intensity, P = period, X = beam width, H = phase; Shift+key nudges
# down. NOT Shift+wheel (wheel zoom is unconditional above); +/- stay
# CORRIDOR-scoped.
LIGHT_PICK_RADIUS = 1.0          # tiles — marker hit-test radius
LIGHT_RANGE_STEP, LIGHT_RANGE_MIN, LIGHT_RANGE_MAX = 1.0, 1.0, 60.0
LIGHT_INTENSITY_STEP, LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX = 0.1, 0.1, 5.0
LIGHT_PERIOD_STEP, LIGHT_PERIOD_MIN, LIGHT_PERIOD_MAX = 0.25, 0.25, 30.0
LIGHT_BEAM_STEP, LIGHT_BEAM_MIN, LIGHT_BEAM_MAX = 5.0, 5.0, 360.0
LIGHT_PHASE_STEP = 0.125         # turns; wraps mod 1 (pair = 0.0 / 0.5)
EDITOR_TICK_DT = 1.0 / 60.0      # editor preview clock (60 fps target) —
                                 # the editor is NOT the sim (P4 §2.4)

# Colour presets cycle on C — 0-255 int triples (the toml schema's units).
# First preset = the ported main.py emergency lamp ((1.0, 0.1, 0.05) * 255);
# "red"/"blue" make the chapter's cop-car beacon pair.
LIGHT_COLOR_PRESETS = (
    ("emergency red", (255, 26, 13)),
    ("warm white", (255, 214, 170)),
    ("cool white", (255, 255, 242)),
    ("amber", (255, 160, 40)),
    ("red", (255, 40, 40)),
    ("blue", (64, 96, 255)),
    ("green", (60, 255, 120)),
)

# ---- WATER mode (P5 §2.4) ---------------------------------------------------
# Bucket-fill an enclosed region to a depth (metres) -> the level's initial
# water field, saved as `water_init.npy` (int32 Q16.16, shape == tilemap) +
# a managed [water] block in level.toml. Depth is quantized AT FILL TIME via
# water_fixed.quantize (the single Python rounding source), UI in metres.
# -/= step the depth (CORRIDOR's +/- are corridor-scoped, so they are free
# here). Depths past WATER_DEEP_HINT_M get the deep-tank status hint: a
# breached deep column dumps a lot of mass fast and drains may flash-boil
# against low pressure — by design (P5 doc §3 drain asymmetry).
# WATER_FILENAME (the .npy carrier name) lives in level_lib with the writer.
WATER_DEPTH_DEFAULT_M = 1.0
WATER_DEPTH_STEP_M = 0.1
WATER_DEPTH_MIN_M = 0.1
WATER_DEPTH_MAX_M = 3.0
WATER_DEEP_HINT_M = 1.5
WATER_OVERLAY_RGB = (60, 140, 255)   # editor overlay tint (not the renderer)

# ---- ZONE mode (Arc C5, editor doc §5) -------------------------------------
ZONE_OVERLAY_ALPHA = 90
ZONE_PAINT_CLASSES = ZONE_CLASSES     # ("breach_site", "extraction_zone")

# ---- AIR mode (Arc C5, editor doc §7) --------------------------------------
AIR_PRESSURE_DEFAULT_ATM = 1.0
AIR_PRESSURE_STEP_ATM = 0.1
AIR_PRESSURE_MIN_ATM = 0.0
AIR_PRESSURE_MAX_ATM = 4.0
AIR_OVERLAY_RGB = (255, 150, 40)      # editor overlay tint for authored air

# ---- WIRE mode / LOGIC overlay (Arc C6, editor doc §8) ---------------------
LOGIC_WIRE_RGBA = (120, 200, 255, 220)     # a plain source->input connector
LOGIC_TAG_RGBA = (255, 200, 90, 235)       # a tag-target badge (never fanned)
LOGIC_NODE_RADIUS_TILES = 0.35             # positionless-node marker chip


def _zone_color(zone_id: int) -> tuple:
    """Deterministic, well-spread colour for a zone paint id (the same
    golden-angle hue-walk `level_edit_common._auto_rgb` uses for materials,
    kept local so this module never reaches into that private helper)."""
    hue = (int(zone_id) * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))

# The +1-tile re-bake margin (engine/15 §4 / P2 edge16 contract): a stroke
# flips the edge masks of its NEIGHBOURS, so the preview re-bake rect must
# grow by one tile on every side or stale wall pieces ring the stroke.
REBAKE_MARGIN = 1

MAX_TEX_PX = 16384               # preview texture cap (common GPU limit)

_STRAIGHT_MASKS = (BIT_N | BIT_S, BIT_E | BIT_W)
_NEIGH8 = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if dx or dy)


# ---------------------------------------------------------------------------
# Pure helpers — NEW-level scaffold
# ---------------------------------------------------------------------------

def parse_size(text) -> tuple:
    """``'48x32' -> (48, 32)`` (tiles, 'x' or 'X'). ValueError with a usable
    message otherwise; both dimensions must be >= SCAFFOLD_MIN."""
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", str(text))
    if not m:
        raise ValueError(
            f"--size wants WxH in tiles (e.g. 48x32), got {text!r}")
    w, h = int(m.group(1)), int(m.group(2))
    if w < SCAFFOLD_MIN or h < SCAFFOLD_MIN:
        raise ValueError(
            f"--size {w}x{h}: the scaffold needs at least "
            f"{SCAFFOLD_MIN}x{SCAFFOLD_MIN} tiles (space border + hull ring "
            f"+ interior)")
    return w, h


def scaffold_grid(width: int, height: int) -> np.ndarray:
    """The NEW-level hull shell (engine/15 §5): a 1-tile SPACE border, a
    1-tile MAT_HULL ring inside it, MAT_AIR interior."""
    w, h = int(width), int(height)
    if w < SCAFFOLD_MIN or h < SCAFFOLD_MIN:
        raise ValueError(
            f"scaffold needs at least {SCAFFOLD_MIN}x{SCAFFOLD_MIN} tiles, "
            f"got {w}x{h}")
    g = np.full((h, w), MAT_AIR, dtype=np.int32)
    g[0, :] = g[-1, :] = SPACE_CODE
    g[:, 0] = g[:, -1] = SPACE_CODE
    g[1, 1:w - 1] = MAT_HULL
    g[h - 2, 1:w - 1] = MAT_HULL
    g[1:h - 1, 1] = MAT_HULL
    g[1:h - 1, w - 2] = MAT_HULL
    return g


def create_level(level_dir, width: int, height: int, *, name=None,
                 tileset=DEFAULT_TILESET,
                 px_per_tile: int = DEFAULT_PX_PER_TILE,
                 seed: int = 0) -> dict:
    """Create a level folder from scratch (engine/15 §5 NEW): scaffold
    tilemap.csv + a minimal v2 level.toml, then run a full bake — the baker
    writes the [art.bare]/[art.align]/[bake] blocks and the PNGs, so the
    fresh folder is immediately loadable by level_loader (which requires
    art) and re-bakeable as recorded. Refuses to clobber an existing
    level.toml. Returns bake_level's summary dict."""
    level_dir = Path(level_dir)
    if (level_dir / "level.toml").exists():
        raise ValueError(
            f"{level_dir} already contains a level.toml — open it instead "
            f"of 'new'")
    grid = scaffold_grid(width, height)
    level_dir.mkdir(parents=True, exist_ok=True)
    csv = "\n".join(",".join(str(int(v)) for v in row)
                    for row in grid.tolist()) + "\n"
    (level_dir / "tilemap.csv").write_bytes(csv.encode("ascii"))
    display = str(name) if name else level_dir.name
    toml = (
        "# created by tools/map_editor.py (engine/15 §5 NEW)\n"
        'version = "2"\n'
        f'name = "{display}"\n'
        "\n"
        "# v2 codes ARE canon material ids (src/simulation/materials.py)\n"
        'tilemap = "tilemap.csv"\n'
        "\n"
        "tile_size_m = 0.333\n")
    (level_dir / "level.toml").write_bytes(toml.encode("utf-8"))
    # No .bak: the file the baker rewrites is seconds old, nothing to keep.
    return bake_level(level_dir, tileset=tileset, px_per_tile=px_per_tile,
                      seed=seed, write_bak=False)


# ---------------------------------------------------------------------------
# Pure helpers — ROOM / CORRIDOR / DOOR geometry
# ---------------------------------------------------------------------------

def wall_family_codes(tileset) -> frozenset:
    """Union of every [groups] entry in the tileset manifest — the codes the
    editor treats as 'a wall stands here' for ROOM sharing, CORRIDOR lining
    and DOOR runs. Manifest-driven (engine/15 §3), never hardcoded."""
    codes = set()
    for members in tileset.group_codes.values():
        codes |= set(members)
    return frozenset(codes)


def normalize_rect(ax, ay, bx, by, grid_w: int, grid_h: int):
    """Inclusive tile rect (x0, y0, x1, y1) from two drag corners (any
    order, possibly outside the grid), clamped to the grid. None when the
    rect lies fully off-grid."""
    x0, x1 = sorted((int(ax), int(bx)))
    y0, y1 = sorted((int(ay), int(by)))
    x0, x1 = max(0, x0), min(int(grid_w) - 1, x1)
    y0, y1 = max(0, y0), min(int(grid_h) - 1, y1)
    if x0 > x1 or y0 > y1:
        return None
    return (x0, y0, x1, y1)


def apply_room(grid: np.ndarray, rect, wall_id: int, wall_codes) -> int:
    """Stamp a room (engine/15 §5 ROOM): wall perimeter + AIR interior.

    ``rect`` is the inclusive tile rect (x0, y0, x1, y1). Interior cells
    become MAT_AIR (old walls/furniture inside are cleared). Perimeter cells
    become ``wall_id`` EXCEPT where a wall-family tile already stands: the
    existing wall is SHARED, not repainted — dragging a room onto an
    existing room's wall reuses that wall instead of doubling it, and a wood
    room drawn against a hull bulkhead keeps the hull (doors in the shared
    run survive too). Degenerate rects (1-2 tiles thin) are all perimeter.
    Returns the number of cells changed."""
    x0, y0, x1, y1 = (int(v) for v in rect)
    changed = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            v = int(grid[ty, tx])
            if tx in (x0, x1) or ty in (y0, y1):
                if v in wall_codes:
                    continue                     # shared wall — keep it
                grid[ty, tx] = wall_id
                changed += 1
            elif v != MAT_AIR:
                grid[ty, tx] = MAT_AIR
                changed += 1
    return changed


def corridor_cells(x0, y0, x1, y1, width: int,
                   grid_w: int, grid_h: int) -> set:
    """The floor swath of a corridor drag: a square brush of side ``width``
    stamped along every tile of the drag line, clipped to the grid (line
    points whose center falls outside contribute nothing — the brush_rect
    contract). Returns a set of (tx, ty) cells."""
    cells = set()
    for px_, py_ in line_tiles(int(x0), int(y0), int(x1), int(y1)):
        r = brush_rect(px_, py_, int(width), int(grid_w), int(grid_h))
        if r is None:
            continue
        a0, b0, a1, b1 = r
        for cy in range(b0, b1 + 1):
            for cx in range(a0, a1 + 1):
                cells.add((cx, cy))
    return cells


def apply_corridor(grid: np.ndarray, x0, y0, x1, y1, *, width: int,
                   wall_id: int, wall_codes) -> int:
    """Cut a corridor (engine/15 §5 CORRIDOR): an AIR floor swath of side
    ``width`` along the drag line, lined with ``wall_id`` on every
    8-neighbour cell still holding non-AIR after the cut — the swath stays
    airtight against SPACE and closed solids, while a side that opens into
    existing AIR (the room it connects) stays open. Existing wall-family
    neighbours are SHARED (kept, not repainted), matching apply_room.
    8-connectivity (not 4) closes the diagonal pinholes a diagonal-ish drag
    would otherwise leave against vacuum. Returns cells changed."""
    h, w = grid.shape
    floor = corridor_cells(x0, y0, x1, y1, width, w, h)
    changed = 0
    for cx, cy in floor:
        if int(grid[cy, cx]) != MAT_AIR:
            grid[cy, cx] = MAT_AIR
            changed += 1
    for cx, cy in floor:
        for dx, dy in _NEIGH8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in floor:
                continue
            v = int(grid[ny, nx])
            if v == MAT_AIR or v in wall_codes:
                continue                         # open join / shared wall
            grid[ny, nx] = wall_id
            changed += 1
    return changed


def door_check(grid: np.ndarray, tx, ty, wall_codes) -> tuple:
    """DOOR-mode gate (engine/15 §5): may tile (tx, ty) become MAT_DOOR?

    Superseded in the interactive tool by
    ``door_entity_port.door_anchor_check`` (Arc C3: the DOOR mode places a
    `door` [[entity]] + stamps MAT_DOOR_CLOSED, not the legacy painted
    MAT_DOOR this function gates) — kept as a pure, tested primitive for
    API stability (tests/test_map_editor_tool.py) and any future legacy-
    MAT_DOOR tooling (e.g. a "convert painted door" helper).

    Returns (ok, why) — ``why`` feeds the status line either way. Accepted:
    a wall-family tile inside a STRAIGHT wall run (edge16 mask exactly N|S
    or E|W). Refused: outside the grid, not a wall, already a door, and
    every non-straight mask (isolated pillar, run end, corner/T/cross — a
    door needs walls on exactly two opposite sides)."""
    h, w = grid.shape
    tx, ty = int(tx), int(ty)
    if not (0 <= tx < w and 0 <= ty < h):
        return False, "outside the grid"
    v = int(grid[ty, tx])
    if v == MAT_DOOR:
        return False, "already a door"
    if v not in wall_codes:
        return False, "not a wall tile"
    mask = edge16_mask(grid, tx, ty, wall_codes)
    if mask == BIT_N | BIT_S:
        return True, "vertical wall run"
    if mask == BIT_E | BIT_W:
        return True, "horizontal wall run"
    if mask == 0:
        return False, "isolated wall tile"
    if mask in (BIT_N, BIT_E, BIT_S, BIT_W):
        return False, "end of a wall run"
    return False, "corner/junction (a door needs a straight run)"


# ---------------------------------------------------------------------------
# Pure helpers — dirty rects for the live preview
# ---------------------------------------------------------------------------

def expand_dirty_rect(rect, grid_w: int, grid_h: int,
                      margin: int = REBAKE_MARGIN) -> tuple:
    """Grow a tile rect (tx0, ty0, tw, th) by ``margin`` tiles on ALL sides,
    clipped to the grid. The preview re-bake MUST use margin >= 1: a stroke
    changes the edge16 mask of every NEIGHBOURING wall tile, so re-baking
    only the changed cells leaves stale wall pieces ringing the stroke
    (engine/15 §4 region re-bake + the P2 mask contract)."""
    x0, y0, tw, th = (int(v) for v in rect)
    nx0, ny0 = max(0, x0 - margin), max(0, y0 - margin)
    nx1 = min(int(grid_w), x0 + tw + margin)
    ny1 = min(int(grid_h), y0 + th + margin)
    return (nx0, ny0, nx1 - nx0, ny1 - ny0)


def diff_rect(a: np.ndarray, b: np.ndarray):
    """Bounding tile rect (tx0, ty0, tw, th) of every cell where the two
    grids differ, or None when identical (drives the undo re-bake)."""
    ys, xs = np.nonzero(np.asarray(a) != np.asarray(b))
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def choose_preview_ppt(ts_px: int, want: int, grid_w: int, grid_h: int,
                       max_px: int = MAX_TEX_PX) -> int:
    """Preview bake resolution: the largest divisor of the tileset's source
    px that is <= the level's bake px_per_tile AND keeps both preview
    texture dimensions under ``max_px`` (GPU texture limit). Falls back to
    the smallest divisor when nothing fits (a huge map previews coarse
    rather than crashing the driver). SAVE always bakes at the recorded
    [bake].px_per_tile — this is view-only."""
    divisors = [d for d in range(1, int(ts_px) + 1) if int(ts_px) % d == 0]
    fitting = [d for d in divisors
               if d <= int(want) and int(grid_w) * d <= max_px
               and int(grid_h) * d <= max_px]
    return max(fitting) if fitting else divisors[0]


# ---------------------------------------------------------------------------
# Pure helpers — SPAWN entries (level.toml writeback lives in level_lib)
# ---------------------------------------------------------------------------

def spawn_at(spawns, ftx: float, fty: float):
    """Index of the topmost spawn whose footprint square contains the
    fractional tile point (ftx, fty), or None."""
    for i in range(len(spawns) - 1, -1, -1):
        s = spawns[i]
        fp = float(s.footprint)
        if s.x <= ftx < s.x + fp and s.y <= fty < s.y + fp:
            return i
    return None


def unique_spawn_name(spawns, team: int) -> str:
    """``marine_1`` / ``zombie_3`` — the first free auto-name for a team."""
    base = TEAM_NAMES.get(int(team), f"team{int(team)}")
    taken = {s.name for s in spawns}
    n = 1
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


# ---------------------------------------------------------------------------
# Pure helpers — LIGHT entries (level.toml writeback lives in level_lib)
# ---------------------------------------------------------------------------

def light_at(lights, ftx: float, fty: float,
             radius: float = LIGHT_PICK_RADIUS):
    """Index of the topmost light whose center lies within ``radius`` tiles
    of the fractional tile point (ftx, fty), or None (spawn_at's pattern —
    lights are points, so the hit-test is a disc, not a footprint)."""
    r2 = float(radius) * float(radius)
    for i in range(len(lights) - 1, -1, -1):
        l = lights[i]
        if (l.x - ftx) ** 2 + (l.y - fty) ** 2 <= r2:
            return i
    return None


def light_color_name(color) -> str:
    """Preset name for a normalized color, or the rgb triple when it is
    not a preset (hand-authored toml values)."""
    ints = color_255(color)
    for name, c in LIGHT_COLOR_PRESETS:
        if c == ints:
            return name
    return f"rgb{ints}"


def next_light_color(color, step: int = 1) -> tuple:
    """The next/previous LIGHT_COLOR_PRESETS entry after ``color``
    (normalized 0-1), matching by 0-255 int triple; colors outside the
    preset list restart at the first preset."""
    ints = color_255(color)
    presets = [c for _, c in LIGHT_COLOR_PRESETS]
    try:
        idx = (presets.index(ints) + int(step)) % len(presets)
    except ValueError:
        idx = 0
    return tuple(v / 255.0 for v in presets[idx])


def next_color_rgb_int(value, step: int = 1) -> list:
    """The next/previous LIGHT_COLOR_PRESETS entry after an ALREADY-0-255
    int triple (Arc C4's SELECT-mode color nudge, generalized to any
    `color_rgb` field a non-light class might carry — `next_light_color`'s
    own matching logic, but never through the 0-1 float LightEntry stores,
    since a generic entity field carries the raw 0-255 ints, per
    `entity_editor_ui.format_field_value`'s `color_rgb` branch)."""
    presets = [c for _, c in LIGHT_COLOR_PRESETS]
    ints = tuple(int(c) for c in value)
    try:
        idx = (presets.index(ints) + int(step)) % len(presets)
    except ValueError:
        idx = 0
    return list(presets[idx])


# ---------------------------------------------------------------------------
# Pure helpers — WATER fill (water_init.npy/[water] writeback in level_lib)
# ---------------------------------------------------------------------------

def water_solid_codes(cfg=None) -> frozenset:
    """Material ids that are solid-for-water — THE seam of P5 critique M1:
    sim-exact, ``MaterialTable.from_config().permeability <= 0.0``, which is
    exactly how gamemap.py derives ``solid`` (the solver's mass-sink
    boundary). NEVER the tileset manifest's ``wall_family_codes`` — that is
    art-connectivity data, equal today by coincidence; a future
    opaque-but-permeable grill would silently diverge the fill boundary
    from the solver's. SPACE_CODE (9) is NOT a material id and is never in
    this set — callers handle it explicitly (see :func:`water_open_mask`)
    before any fancy-indexing of the table."""
    tbl = MaterialTable.from_config(cfg)
    return frozenset(int(i) for i in range(tbl.n)
                     if float(tbl.permeability[i]) <= 0.0)


def water_open_mask(grid: np.ndarray, solid_codes) -> np.ndarray:
    """(H, W) bool: tiles water may occupy — NOT solid-for-water AND NOT
    SPACE (P5 critique M2: vacuum bounds a fill exactly like glass does —
    a breached room floods up to the breach, never into space). Membership
    via np.isin against the id set, so SPACE_CODE never indexes the
    material table."""
    g = np.asarray(grid)
    solid = np.isin(g, np.asarray(sorted(solid_codes), dtype=g.dtype))
    return (~solid) & (g != SPACE_CODE)


def water_fill_region(grid: np.ndarray, tx: int, ty: int, solid_codes):
    """The 4-connected fillable region containing tile (tx, ty).

    Returns ``(set of (tx, ty), why)`` — or ``(None, why)`` when the fill
    must be refused: start outside the grid, on SPACE (P5 §2.4: water
    cannot be authored in vacuum — FieldEdit is the deliberate runtime
    path), or on a solid-for-water tile (the solver zeroes depth there).
    4-neighbour connectivity: diagonal gaps do NOT leak water, matching the
    pipe model's 4-face fluxes."""
    g = np.asarray(grid)
    h, w = g.shape
    tx, ty = int(tx), int(ty)
    if not (0 <= tx < w and 0 <= ty < h):
        return None, "outside the grid"
    v = int(g[ty, tx])
    if v == SPACE_CODE:
        return None, ("started on SPACE — water cannot stand in vacuum "
                      "(it flash-boils; author breach inflow via FieldEdit)")
    if v in solid_codes:
        name = MATERIAL_NAMES.get(v, f"id {v}")
        return None, f"started on solid {name} — water needs an open tile"
    open_ = water_open_mask(g, solid_codes)
    from collections import deque
    region = {(tx, ty)}
    q = deque(region)
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < w and 0 <= ny < h and open_[ny, nx]
                    and (nx, ny) not in region):
                region.add((nx, ny))
                q.append((nx, ny))
    return region, "ok"


def mask_water_to_open(depth_q: np.ndarray, grid: np.ndarray,
                       solid_codes):
    """Zero water depth on every tile water may not occupy (solid-for-water
    / SPACE) — the SAVE-time wall-over-pool guard (P5 critique M3: a wall
    painted over a pool after the fill would otherwise save depth the
    solver immediately destroys as a silent mass sink). Returns
    ``(masked int32 copy, cleared cell count)``; the loader's warn stays as
    the hand-authoring backstop."""
    open_ = water_open_mask(grid, solid_codes)
    d = np.asarray(depth_q)
    masked = np.where(open_, d, 0).astype(np.int32)
    cleared = int(np.count_nonzero(d[~open_]))
    return masked, cleared


# ---------------------------------------------------------------------------
# Interactive editor
# ---------------------------------------------------------------------------

WIN_W, WIN_H = 1280, 920
AUTO_FRAMES = 90          # --auto: close after ~90 frames (smoke test)

COL_BG = (16, 16, 22, 255)
COL_GRID = (255, 255, 255, 40)
COL_PANEL_BG = (26, 26, 34, 255)     # pane fill (top bar / rail / palette / …)
COL_PANEL_EDGE = (60, 60, 72, 255)   # 1-px seam between panes
COL_RAIL_SEL = (60, 62, 96, 255)     # active mode row in the tool rail
COL_CURSOR = (255, 255, 255, 220)
COL_TEXT = (200, 200, 200, 255)
COL_TEXT_DIM = (150, 150, 160, 255)
COL_TEXT_HOT = (255, 230, 120, 255)
COL_OK = (120, 255, 140, 255)
COL_BAD = (255, 90, 80, 255)
COL_SELECT = (255, 220, 60, 255)     # Arc C4 selection highlight

# Arc C4 text-entry (tag-assign / id-rename — the one scoped, deliberate
# exception to "no in-UI text input in v1": tag names and ids are free-form
# slugs no preset/cycle affordance can cover, so SELECT mode's T/I gestures
# open a tiny single-purpose buffer (GetCharPressed + backspace/enter/esc),
# not a general text-editing widget). Mirrors level_loader's own (private)
# id-slug charset (`_ENTITY_ID_RE`) so an id typed here is guaranteed
# loadable; tag names use the same charset for the same reason (wire
# `tag:name.input` targets share the slug rule, engine/16 §8).
_SLUG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]*\Z")
_SLUG_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")

# Tool-rail + palette pane row geometry (shared by click-routing and draw so
# the hit targets match the visuals).
RAIL_PAD_Y = 8
RAIL_ROW_H = 30
PAL_PAD_Y = 30            # room for the tab header above the first chip
PAL_ROW_H = 22
PAL_TABS = ("MATERIAL", "ENTITY")   # the tabbed palette (Arc C1, editor §8)
PAL_TAB_FONT = 14
PAL_TAB_GAP = 20           # px between tab labels' click-hit boxes

MODE_HINTS = {
    "PAINT": "LMB paint  RMB erase  Shift+click line  I eyedrop  [ ] brush",
    "ROOM": "drag LMB: wall perimeter + AIR interior (existing walls "
            "shared)  RMB cancels",
    "CORRIDOR": "drag LMB: AIR swath, walls line the space/solid sides  "
                "+/- width  RMB cancels",
    "DOOR": "click/drag a STRAIGHT wall run -> door entity (default 1.0 m; "
            "drag to resize)  O open/closed  hover shows green/red",
    "SPAWN": "LMB place/drag  RMB delete  T team toggle "
             "(hovered marker, else placement team)",
    "LIGHT": "LMB place/drag  RMB delete  B kind  C color  R range  "
             "E intens  P period  X beam  H phase (Shift = down)",
    "WATER": "LMB fill enclosed region to depth  RMB fill-to-dry  "
             "-/= depth 0.1 m steps (hover shows green/red)",
    "ENTITY": "select a class in the ENTITY palette tab, then LMB place  "
              "(field sensors: drag to set the AIR-facing arrow)",
    "SELECT": "LMB select  shift+LMB toggle  drag empty = box-select  "
              "drag a selected instance = move  Delete removes  "
              "A select-by-class  T tag  I re-id  [ ] focus field  "
              "= / - nudge  Enter cycle  C color  Ctrl+C/V clump copy/paste",
    "WAND": "LMB enclosure-fill  RMB same-code select+paint  "
            "(paints the MATERIAL palette's selected id — 9/SPACE = vacuum)",
    "ZONE": "LMB enclosure-fill paints the target zone  RMB same-code "
            "clears a zone's paint  I eyedrop target  [ ] cycle target "
            "(new/existing)  C zone class (new only)",
    "AIR": "LMB enclosure-fill seeds air_init to the target pressure "
           "(refused if it reaches the map border — hull-leak)  "
           "RMB fill to 0 atm  -/= pressure 0.1 atm steps",
    "PROPS": "level-wide properties (no canvas edits): B toggles "
             "boundary space<->ambient (format only — no physics here)",
    "WIRE": "LMB click a source, navigate, LMB click a target — never a "
            "drag  [ ] cycle signal/input  T tag target  Enter confirms "
            "(multi-input)  Esc cancels  LMB on a wire selects, Delete "
            "removes  L (global) toggles show-all wires",
}


def _pressed(key) -> bool:
    """Key pressed this frame, with OS key-repeat while held."""
    return rl.is_key_pressed(key) or rl.is_key_pressed_repeat(key)


def _pixels_ptr(arr: np.ndarray):
    """cffi ``void *`` view of a contiguous uint8 array for raylib's
    UpdateTexture* (pyray's wrapper only passes pointer cdata through; a
    bare ffi.from_buffer 'char[]' is rejected by its is_cdata check). The
    CALLER must keep ``arr`` alive across the raylib call — the pointer
    does not own the buffer."""
    return ffi.cast("void *", ffi.from_buffer(arr))


def _texture_from_rgba(rgba: np.ndarray):
    """GPU texture (R8G8B8A8) from an (H, W, 4) uint8 array."""
    h, w = rgba.shape[:2]
    img = rl.gen_image_color(int(w), int(h), rl.BLANK)   # RGBA8 by contract
    tex = rl.load_texture_from_image(img)
    rl.unload_image(img)
    arr = np.ascontiguousarray(rgba)
    rl.update_texture(tex, _pixels_ptr(arr))
    return tex


def _load_entity_icon_textures(entity_palette) -> dict:
    """Arc C8 — load each ENTITY-palette class's committed PNG icon
    (``art/entities/icons/<class>.png``, ``entity_editor_ui.icon_png_path``)
    into a GPU texture ONCE at launch, keyed by class name. A class with no
    committed PNG is simply absent from the returned dict — the draw loop's
    palette-chip fallback (colour chip + class initial) is PERMANENT, not
    conditional on this cache being incomplete (editor doc §8). The editor
    RUNTIME only ever reads the already-rasterized PNGs here — it never
    calls ``tools/rasterize_icons.py`` (that is a dev-only regeneration
    step, run by hand when an SVG source changes)."""
    textures = {}
    for pent in entity_palette:
        png_path = entity_editor_ui.icon_png_path(pent.class_name)
        if png_path is None:
            continue
        rgba = np.asarray(Image.open(png_path).convert("RGBA"))
        textures[pent.class_name] = _texture_from_rgba(rgba)
    return textures


def run_editor(level_name: str, *, tileset_override=None, ppt_override=None,
               seed_override=None, auto: bool = False) -> None:
    handle = level_lib.open_level(str(level_name))
    lvl = handle.data
    if lvl.version != "2":
        raise SystemExit(
            f"map_editor speaks level format v2 only; {level_name} is "
            f"v{lvl.version} (migrate first)")
    bake_tbl = lvl.raw_toml.get("bake")
    if not isinstance(bake_tbl, dict) or not bake_tbl:
        raise SystemExit(
            f"{level_name} has no [bake] block — it is a PAINTED level "
            f"(align/paint it with tools/align_level_art.py), or run "
            f"tools/bake_level_art.py {level_name} once to adopt it into "
            f"the tiled path")

    level_dir = lvl.path
    csv_path = level_dir / str(lvl.raw_toml["tilemap"])
    grid = np.array(lvl.tilemap, dtype=np.int32, copy=True)
    level_loader.materials_from_tilemap(grid, lvl.version)  # validate codes
    grid_h, grid_w = grid.shape
    # Arc C3 — the DOOR tool quantizes spans at this level's OWN tile_size_m
    # (the editor never scales resolution; that is --res's job, main.py).
    # Checked ONCE at launch: a non-integer tiles-per-meter level has no
    # defined door quantization (a6 doors design §3 S1) — the DOOR tool
    # refuses cleanly with a flash instead of the quantizer raising mid-drag.
    tile_size_m = float(lvl.tile_size_m)
    door_tool_error = door_entity_port.door_tool_availability(tile_size_m)

    tileset_arg = (tileset_override if tileset_override is not None
                   else bake_tbl.get("tileset", DEFAULT_TILESET))
    bake_ppt = (int(ppt_override) if ppt_override is not None
                else int(bake_tbl.get("px_per_tile", DEFAULT_PX_PER_TILE)))
    bake_seed = (int(seed_override) if seed_override is not None
                 else int(bake_tbl.get("seed", 0)))
    tileset_dir = Path(tileset_arg)
    if not tileset_dir.is_absolute():
        tileset_dir = ROOT / tileset_dir
    ts = load_tileset(tileset_dir)
    wall_codes = wall_family_codes(ts)

    palette = build_palette()
    palette_order = tuple(sorted(MATERIAL_NAMES)) + (SPACE_CODE,)
    mat_fill = {pid: rgb for pid, (_, rgb) in palette.items()
                if rgb is not None}

    # Arc C1 — registry-driven ENTITY palette tab (editor doc §3 pillar 1,
    # canon §1): import-then-fallback once at launch; on success
    # entity_registry.json is rewritten as the new last-good (the registry
    # module's own job — see entity_editor_ui.load_registry). On failure the
    # palette still renders (minus nothing — the whole last-good snapshot is
    # usable) and the status bar shows a red banner (registry_banner below).
    registry_result = entity_editor_ui.load_registry()
    registry_payload = registry_result.payload
    entity_palette = entity_editor_ui.palette_entries(registry_payload)
    # Arc C6 — the same deterministic chip colour the ENTITY palette already
    # generated per class, reused for positionless logic-node markers (one
    # colour per class, no separate recipe).
    entity_chip_rgb = {p.class_name: p.chip_rgb for p in entity_palette}
    registry_banner = (None if registry_result.ok else
                       f"registry fallback ({registry_result.error})")
    palette_tab = "MATERIAL"           # MATERIAL <-> ENTITY (pane click only)
    selected_entity_class = None       # ENTITY tab selection -> inspector template

    preview_ppt = choose_preview_ppt(ts.px, bake_ppt, grid_w, grid_h)
    ppt_pair = (float(preview_ppt), float(preview_ppt))
    offset0 = (0.0, 0.0)

    selected_id = MAT_HULL
    mode_idx = 0
    brush = 1
    corridor_w = DEFAULT_CORRIDOR_WIDTH
    spawn_team = TEAM_MARINE
    spawns = [replace(s) for s in lvl.spawns]
    # LIGHT (Arc C1, Amendment A1): the bespoke LightEntry-list-only state is
    # gone — `lights` now holds EditableLight (LightEntry + entity id), and
    # `light_form`/`entities` (captured ONCE at load) decide which
    # level_lib family a save writes lights through (light_entity_port
    # preserves whichever the level already used — no forced migration).
    light_form = light_entity_port.light_form(lvl.raw_toml)
    # Renamed other_entities -> entities (B4): this is the SAME live list the
    # transaction log mutates in place AND the object Ctrl+S serializes (see
    # the ctx registry below + format_lights_for_save). C2 places nothing into
    # it yet (C3 does), but it is registered + logged so doors/sensors join
    # with no model change.
    entities = [e for e in lvl.entities
                if e.class_name != light_entity_port.LIGHT_CLASS]
    lights = light_entity_port.initial_editable_lights(lvl)
    light_kind = "static"                       # B off-marker toggles this
    light_color = tuple(v / 255.0 for v in LIGHT_COLOR_PRESETS[0][1])

    # Arc C4 — the `[[wire]]` family as DATA only (the interactive wire TOOL
    # is C6's job). Loaded so clump paste can re-id/drop wires and so
    # anything that touches it (delete's referential-integrity cleanup,
    # paste) round-trips through Ctrl+S; registered as an undo collection
    # per C2's reserved "wires" name (as-built §"the extension pattern").
    wires = list(lvl.wire_specs)

    # Arc C4 — unified entity selection (a set of ids, over lights u
    # entities — SPAWN stays out by construction, canon §0/A1). Persists
    # across mode switches; find_instance() re-resolves an id fresh every
    # use, so a stale id (something deleted elsewhere) just quietly misses.
    selection = set()
    select_box = None        # {"anchor": (tx,ty), "add": bool} box-drag
    select_move = None        # {"anchor": (tx,ty)} move-drag
    select_focus_idx = 0      # focused editable field, single-selection nudge
    clump_clipboard = None    # compute_clump_copy() snapshot, or None
    text_entry = None         # {"kind": "tag"|"id", "buf": str} or None

    # Arc C6 — the two-click wire tool. `wire_pending` is the in-progress
    # gesture (a plain dict, matching the entity_drag/select_box convention
    # for un-committed gesture state — see wire_tool.begin_pending_wire);
    # `wire_selected` is an EXISTING wire picked for removal (WIRE mode,
    # Delete key). Neither ever opens a log transaction on its own (§6.2),
    # so a mode switch simply drops them (cancel_transients below).
    # `show_all_wires` (L, global — persists like show_grid/view_baked) is
    # the LOGIC overlay's show-all toggle; its default (False) shows only
    # wires touching the current SELECT-mode selection (editor doc §8).
    wire_pending = None
    wire_selected = None
    show_all_wires = False

    # WATER state (P5 §2.4): a parallel int32 Q16.16 depth grid — loaded
    # from the level's [water] seed when present, else all-dry. Depth is
    # quantized at fill time (water_fixed.quantize); the UI shows metres.
    # The solid-for-water id set is sim-exact (permeability <= 0, the M1
    # seam) and read from config ONCE per session.
    water_q = (np.array(lvl.water_depth_q, dtype=np.int32, copy=True)
               if lvl.water_depth_q is not None
               else np.zeros(grid.shape, dtype=np.int32))
    water_depth_m = WATER_DEPTH_DEFAULT_M
    water_solid = water_solid_codes()

    # Arc C5 — ZONE paint: zones.npy (uint8 paint ids), `None` at load when
    # the level carries no zones.npy at all (zone dormancy, canon §7). The
    # FIRST successful zone paint allocates zeros + registers it with the
    # undo ctx IN PLACE (C2's own forward note: "GridCellsOp('zones') —
    # register those grids in ctx.grids only once allocated" — a GridCellsOp
    # on a None grid crashes). `zone_paint_class` (C key, ZONE mode) picks
    # which class a brand-new id authors; `zone_paint_id` is the paint
    # target — None means "next unused id" ("new"), else an existing id
    # ([ / ] cycles, I eyedrops the id under the cursor).
    zones = (np.array(lvl.zone_grid, dtype=np.uint8, copy=True)
             if lvl.zone_grid is not None else None)
    zone_paint_class = ZONE_PAINT_CLASSES[0]     # "breach_site"
    zone_paint_id = None
    pending_zone_delete = None    # {"ids": [...], "zone_ids": set(...)} or None

    # Arc C5 — AIR paint: air_init.npy (int32 Q16.16 atm), `None` at load
    # when absent (air dormancy, A9). Allocated + registered the SAME way as
    # zones, on the first successful (non-leaky) fill — seeded to FP_ONE
    # everywhere first (air_paint.default_ambient_grid), so painting one
    # room never silently zero-pressures the rest of a previously-unpainted
    # map. `air_pressure_atm` is the UI's target pressure (-/= steps it).
    air = (np.array(lvl.air_init_q, dtype=np.int32, copy=True)
          if lvl.air_init_q is not None else None)
    air_pressure_atm = AIR_PRESSURE_DEFAULT_ATM

    # Arc C5 — PROPS: the level-properties pane's one field so far
    # (`boundary`, format+loader-only — canon §7). B toggles it; Ctrl+S
    # writes it through level_lib.write_boundary_field ONLY when changed
    # from the value the level loaded with (never a no-op rewrite).
    level_boundary = lvl.boundary

    # Arc C2 — ONE global transaction log replaces the four per-domain rings
    # (undo/spawn_undo/light_undo/water_undo) and the four dirty_* bools. The
    # ctx registry maps names to the LIVE state handles; ops write into them in
    # place (grids via fancy indexing, collections via slice-assign), so the
    # closed-over grid/water_q/spawns/lights/entities names stay valid. Handles
    # are refilled in place, NEVER rebound (§7). zones/air join the grids dict
    # ONLY if the level already had them at load — a first in-session paint
    # allocates + adds the key then (never rebinding "material"/"water").
    undo_grids = {"material": grid, "water": water_q}
    if zones is not None:
        undo_grids["zones"] = zones
    if air is not None:
        undo_grids["air"] = air
    undo_ctx = undo_log.UndoContext(
        grids=undo_grids,
        collections={"spawns": spawns, "lights": lights, "entities": entities,
                    "wires": wires})
    log = undo_log.TransactionLog(undo_ctx)
    csv_bak_written = False
    toml_bak_written = False
    water_bak_written = False     # water_init.npy's OWN once-per-session .bak
    zones_bak_written = False     # zones.npy's OWN once-per-session .bak
    air_bak_written = False       # air_init.npy's OWN once-per-session .bak

    # PAINT stroke state (align-tool pattern). The pre-stroke grid snapshot is
    # now the log's transient before-copy (snapshot_grid at stroke start), so
    # stroke_pending is gone; stroke_dirty stays as the live-preview re-bake
    # hint (§2.1: per-gesture rects survive only as a re-bake hint).
    stroke_active = False
    stroke_dirty = None        # inclusive (x0, y0, x1, y1) of the stroke
    last_paint_tile = None
    anchor_tile = None         # Shift+click line-tool anchor

    room_start = None          # ROOM drag corner (tile)
    corr_start = None          # CORRIDOR drag start (tile)
    # spawn/light drag: (index, (orig x, orig y)) — the pre-drag snapshot now
    # lives in the log (snapshot_coll at drag start); we keep orig only to
    # detect whether the drag actually moved (else the commit drops).
    spawn_drag = None
    light_drag = None
    # Arc C3 — DOOR span drag / ENTITY sensor-arrow drag: {"anchor": (tx,ty),
    # ...}. Unlike every drag above, these open NO log transaction until
    # release (§6.2: validate BEFORE the first mutation, so a refused
    # placement never touches the log) — nothing exists yet to move.
    door_drag = None
    entity_drag = None
    door_initial_state = "closed"   # O toggles the DOOR tool's placement
                                    # default (editor doc §6: default closed)

    view_baked = True          # V
    show_normal = False        # N (baked view only)
    show_grid = True           # G
    flash, flash_frames = "", 0
    esc_armed = 0

    # Arc C7 — F5 PLAY: [(subprocess.Popen, scratch_dir), ...] for every
    # launched game still tracked this session. Reaped every frame (below,
    # non-blocking poll) as each process exits; whatever remains at quit is
    # cleaned up unconditionally right after the main loop (both cleanup
    # paths the gate requires).
    active_plays = []

    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(WIN_W, WIN_H,
                   f"Breach map editor — {lvl.name} ({level_name})")
    rl.set_target_fps(60)
    rl.set_exit_key(rl.KeyboardKey.KEY_NULL)   # Esc handled below

    # Live baked preview: full bake once, then dirty-rect re-bakes.
    full = bake_full(grid, ts, px_per_tile=preview_ppt, seed=bake_seed)
    tex_diffuse = _texture_from_rgba(full.diffuse)
    tex_normal = _texture_from_rgba(full.normal)
    world_w, world_h = grid_w * preview_ppt, grid_h * preview_ppt

    # Arc C8 — committed PNG icons, loaded once into GPU textures (NOT per
    # frame; must come after init_window — texture creation needs a live GL
    # context, same reason tex_diffuse/tex_normal are built here and not
    # earlier). Classes with no PNG simply have no entry here; the palette
    # draw falls back to the generated chip + class initial for those,
    # permanently (editor doc §8).
    entity_icon_textures = _load_entity_icon_textures(entity_palette)

    def rebake_rect(rect) -> None:
        """Re-bake one tile rect (ALREADY +1-expanded) into both preview
        textures. Bake errors (e.g. a material with no tileset strip) land
        on the status line instead of killing the session."""
        nonlocal flash, flash_frames
        try:
            patch = bake_region(grid, ts, rect, px_per_tile=preview_ppt,
                                seed=bake_seed)
        except ValueError as e:
            flash, flash_frames = f"preview bake failed: {e}", 300
            return
        x0, y0, tw, th = patch.rect
        dst = rl.Rectangle(x0 * preview_ppt, y0 * preview_ppt,
                           tw * preview_ppt, th * preview_ppt)
        arr_d = np.ascontiguousarray(patch.diffuse)   # keep alive across the
        arr_n = np.ascontiguousarray(patch.normal)    # call (_pixels_ptr)
        rl.update_texture_rec(tex_diffuse, dst, _pixels_ptr(arr_d))
        rl.update_texture_rec(tex_normal, dst, _pixels_ptr(arr_n))

    def rebake_cells(x0, y0, x1, y1) -> None:
        """Re-bake the inclusive cell rect, +1-expanded (edge16 neighbours)."""
        rebake_rect(expand_dirty_rect((x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                                      grid_w, grid_h))

    def rebake_txn(txn) -> None:
        """Re-bake the MATERIAL grid rects a committed/undone/redone
        transaction touched (+1-expanded). Water/spawn/light/entity overlays
        re-derive from live state each frame, so only material needs an
        explicit re-bake (§7 Ctrl+Z block)."""
        if txn is None:
            return
        for gname, r in txn.rebake_rects():
            if gname == "material":
                rebake_rect(expand_dirty_rect(r, grid_w, grid_h))

    def cancel_transients() -> None:
        """Drop drag/stroke state on a mode switch (§2.4): an OPEN transaction
        with a real change commits (a stroke or drag interrupted mid-gesture is
        kept, matching today + the blessed drag-commit improvement); a no-op
        gesture drops (commit() pushes nothing). A stroke interrupted mid-drag
        still gets its dirty rect re-baked."""
        nonlocal room_start, corr_start, spawn_drag, light_drag
        nonlocal stroke_active, stroke_dirty, last_paint_tile
        nonlocal door_drag, entity_drag
        nonlocal select_box, select_move, text_entry, pending_zone_delete
        nonlocal wire_pending, wire_selected
        if log.has_pending:
            log.commit()
        if stroke_dirty is not None:
            rebake_cells(*stroke_dirty)
        room_start = corr_start = spawn_drag = light_drag = None
        # door_drag/entity_drag/select_box/select_move/wire_pending/
        # wire_selected never open a log transaction (§6.2), so dropping
        # them is a pure state reset — nothing to commit or revert.
        door_drag = entity_drag = None
        select_box = select_move = None
        text_entry = None
        pending_zone_delete = None    # a mode switch cancels the confirm
        wire_pending = wire_selected = None
        stroke_active, stroke_dirty = False, None
        last_paint_tile = None

    # View transform (screen = (world_px - cam) * zoom + canvas origin); fit
    # the world inside the CANVAS pane on start (Arc C panes shell, §8).
    zoom, cam_x, cam_y = fit_camera(compute_panes(WIN_W, WIN_H)["canvas"],
                                    world_w, world_h)

    frames = 0
    shot_path = None
    while not rl.window_should_close():
        win_w, win_h = rl.get_screen_width(), rl.get_screen_height()
        panes = compute_panes(win_w, win_h)
        canvas = panes["canvas"]
        dt = rl.get_frame_time()
        K = rl.KeyboardKey
        shift = (rl.is_key_down(K.KEY_LEFT_SHIFT)
                 or rl.is_key_down(K.KEY_RIGHT_SHIFT))
        ctrl = (rl.is_key_down(K.KEY_LEFT_CONTROL)
                or rl.is_key_down(K.KEY_RIGHT_CONTROL))
        mouse = rl.get_mouse_position()
        # Mouse editing is gated to the canvas pane; clicks over any other
        # pane (top bar / rail / palette / inspector / status) never edit the
        # map. ``over_hud`` keeps its old meaning: "over non-canvas chrome".
        over_canvas = canvas.contains(mouse.x, mouse.y)
        over_hud = not over_canvas
        mode = MODES[mode_idx]
        dirty_any = log.dirty      # position-accurate: undo to saved clears it
        # Arc C4 text-entry sub-state (tag-assign / id-rename): while typing,
        # every OTHER single-key global affordance (mode switch, view
        # toggles, material hotkeys, WASD/arrow pan) is suppressed so a
        # letter/digit in the typed name can't also jump modes or pan the
        # camera; only the text-entry block itself (below, in SELECT mode)
        # consumes keys. Esc cancels the entry instead of arming quit.
        typing = text_entry is not None

        # ---- global: quit / mode / view ----------------------------------
        if rl.is_key_pressed(K.KEY_ESCAPE):
            if typing:
                text_entry = None
            elif wire_pending is not None:      # Arc C6: cancel, don't quit
                wire_pending = None
            elif dirty_any and esc_armed <= 0:
                esc_armed = 180
                flash, flash_frames = (
                    "UNSAVED edits — Esc again to quit without saving "
                    "(Ctrl+S saves)", 180)
            else:
                break
        esc_armed = max(0, esc_armed - 1)

        if not typing and rl.is_key_pressed(K.KEY_TAB):
            mode_idx = (mode_idx + (-1 if shift else 1)) % len(MODES)
            cancel_transients()
            mode = MODES[mode_idx]
        if not typing:
            for i in range(min(len(MODES), MAX_MODE_FKEYS)):
                if i == PLAY_FKEY_INDEX:      # F5 — reserved for PLAY (below)
                    continue
                if rl.is_key_pressed(K.KEY_F1 + i):
                    mode_idx = i
                    cancel_transients()
                    mode = MODES[mode_idx]

        if not typing and rl.is_key_pressed(K.KEY_V):
            view_baked = not view_baked
        if not typing and rl.is_key_pressed(K.KEY_N):
            show_normal = not show_normal
        if not typing and rl.is_key_pressed(K.KEY_G):
            show_grid = not show_grid
        if not typing and rl.is_key_pressed(K.KEY_L):
            # Arc C6 — the LOGIC overlay's show-all toggle (editor doc §8):
            # global + persistent, like show_grid/view_baked, so it stays
            # readable across a mode switch instead of resetting.
            show_all_wires = not show_all_wires

        # Material palette keys — GENERATED from the material table: the
        # number key IS the material id (SPACE on 9). Ids past 8 have no key
        # (eyedropper reaches them).
        if not typing:
            for pid in palette_order:
                digit = 9 if pid == SPACE_CODE else int(pid)
                if 0 <= digit <= 9 and (
                        rl.is_key_pressed(K.KEY_ZERO + digit)
                        or rl.is_key_pressed(K.KEY_KP_0 + digit)):
                    selected_id = pid

        # ---- view: wheel zoom around cursor, keys/drag pan ---------------
        # Mouse offsets are relative to the CANVAS pane origin (the world is
        # drawn there, not at the window corner).
        mcx, mcy = mouse.x - canvas.x, mouse.y - canvas.y
        wheel = rl.get_mouse_wheel_move()
        if wheel != 0.0 and over_canvas:
            wx0, wy0 = cam_x + mcx / zoom, cam_y + mcy / zoom
            zoom = max(0.02, min(50.0, zoom * (1.1 ** wheel)))
            cam_x, cam_y = wx0 - mcx / zoom, wy0 - mcy / zoom
        if not ctrl and not typing:         # keep Ctrl+S clear of S-pan
            pan = 900.0 * dt / zoom
            if rl.is_key_down(K.KEY_W) or rl.is_key_down(K.KEY_UP):
                cam_y -= pan
            if rl.is_key_down(K.KEY_A) or rl.is_key_down(K.KEY_LEFT):
                cam_x -= pan
            if rl.is_key_down(K.KEY_D) or rl.is_key_down(K.KEY_RIGHT):
                cam_x += pan
            if rl.is_key_down(K.KEY_S) or rl.is_key_down(K.KEY_DOWN):
                cam_y += pan
        if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_MIDDLE):
            d = rl.get_mouse_delta()
            cam_x -= d.x / zoom
            cam_y -= d.y / zoom

        # Cursor tile (fractional + containing index) — canvas-relative.
        ftx, fty = art_px_to_tile(cam_x + mcx / zoom,
                                  cam_y + mcy / zoom, offset0, ppt_pair)
        cur_tx, cur_ty = int(np.floor(ftx)), int(np.floor(fty))
        cursor_in = (0 <= cur_tx < grid_w and 0 <= cur_ty < grid_h)
        clamp_tx = min(max(cur_tx, 0), grid_w - 1)
        clamp_ty = min(max(cur_ty, 0), grid_h - 1)

        # Arc C6 — every positionless entity's deterministic clickable tile
        # (recomputed fresh each frame from live state: cheap at editor-scale
        # entity counts, and never stale after an undo/redo/paste/delete).
        node_layout = wire_tool.positionless_layout(entities, zones,
                                                    grid.shape)

        lmb = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
        rmb = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT)
        lmb_click = rl.is_mouse_button_pressed(
            rl.MouseButton.MOUSE_BUTTON_LEFT)
        rmb_click = rl.is_mouse_button_pressed(
            rl.MouseButton.MOUSE_BUTTON_RIGHT)
        lmb_up = rl.is_mouse_button_released(
            rl.MouseButton.MOUSE_BUTTON_LEFT)

        # ---- pane routing: clicks outside the canvas act on their pane ----
        # Tool rail row -> select mode (mirrors TAB / F1-F7); palette tab
        # header -> switch MATERIAL <-> ENTITY (Arc C1: the registry-driven
        # tab, editor doc §8); a chip in the active tab -> select material
        # (mirrors 0-9) or select an entity class (drives the inspector
        # template — placement itself is C3's). Keyboard stays the primary
        # path (§5 keyboard-first); the canvas gate (over_hud) already keeps
        # these clicks from also editing the map.
        if lmb_click and panes["tool_rail"].contains(mouse.x, mouse.y):
            ridx = int((mouse.y - panes["tool_rail"].y - RAIL_PAD_Y)
                       // RAIL_ROW_H)
            if 0 <= ridx < len(MODES) and ridx != mode_idx:
                mode_idx = ridx
                cancel_transients()
                mode = MODES[mode_idx]
        elif lmb_click and panes["palette"].contains(mouse.x, mouse.y):
            pal = panes["palette"]
            rel_y = mouse.y - pal.y
            if rel_y < PAL_PAD_Y:
                tx_ = pal.x + 10
                for label in PAL_TABS:
                    w = rl.measure_text(label, PAL_TAB_FONT)
                    if tx_ <= mouse.x <= tx_ + w + 10:
                        palette_tab = label
                        break
                    tx_ += w + PAL_TAB_GAP
            else:
                pidx = int((rel_y - PAL_PAD_Y) // PAL_ROW_H)
                if palette_tab == "MATERIAL":
                    if 0 <= pidx < len(palette_order):
                        selected_id = palette_order[pidx]
                elif 0 <= pidx < len(entity_palette):
                    selected_entity_class = entity_palette[pidx].class_name

        def selected_is_wall() -> bool:
            name = MATERIAL_NAMES.get(selected_id)
            entry = ts.materials.get(name) if name else None
            return entry is not None and entry.mode == "wall"

        # ---- PAINT --------------------------------------------------------
        if mode == "PAINT":
            if rl.is_key_pressed(K.KEY_I) and cursor_in:   # eyedropper
                selected_id = int(grid[cur_ty, cur_tx])
            if _pressed(K.KEY_LEFT_BRACKET):
                brush = max(BRUSH_MIN, brush - 1)
            if _pressed(K.KEY_RIGHT_BRACKET):
                brush = min(BRUSH_MAX, brush + 1)

            def _stamp(px_, py_, pid) -> int:
                nonlocal stroke_dirty
                c = paint_tiles(grid, px_, py_, pid, brush)
                if c:
                    r = brush_rect(px_, py_, brush, grid_w, grid_h)
                    if stroke_dirty is None:
                        stroke_dirty = list(r)
                    else:
                        stroke_dirty[0] = min(stroke_dirty[0], r[0])
                        stroke_dirty[1] = min(stroke_dirty[1], r[1])
                        stroke_dirty[2] = max(stroke_dirty[2], r[2])
                        stroke_dirty[3] = max(stroke_dirty[3], r[3])
                return c

            if shift and (lmb_click or rmb_click) and not over_hud \
                    and cursor_in:
                # LINE TOOL: Shift+click fills anchor -> cursor (RMB erases);
                # chained Shift+clicks draw a polyline.
                if anchor_tile is not None:
                    # One instantaneous transaction (begin -> mutate -> commit);
                    # commit() diffs the whole grid and drops if nothing changed.
                    pid = selected_id if lmb_click else MAT_AIR
                    log.begin("line")
                    log.snapshot_grid("material")
                    for sx_, sy_ in line_tiles(anchor_tile[0], anchor_tile[1],
                                               cur_tx, cur_ty):
                        _stamp(sx_, sy_, pid)
                    log.commit()
                    if stroke_dirty is not None:
                        rebake_cells(*stroke_dirty)
                    stroke_dirty = None
                anchor_tile = (cur_tx, cur_ty)
            elif (lmb or rmb) and not over_hud and not shift:
                # ONE transaction for the whole stroke: open + snapshot at
                # mouse-down, commit at mouse-up (below). Coalescing is
                # automatic — commit stores before=gesture-start, after=commit.
                if not stroke_active:
                    stroke_active = True
                    log.begin("paint")
                    log.snapshot_grid("material")
                    stroke_dirty = None
                    last_paint_tile = None
                pid = selected_id if lmb else MAT_AIR      # right = eraser
                p0 = (last_paint_tile if last_paint_tile is not None
                      else (cur_tx, cur_ty))
                for sx_, sy_ in line_tiles(p0[0], p0[1], cur_tx, cur_ty):
                    _stamp(sx_, sy_, pid)
                last_paint_tile = (cur_tx, cur_ty)
                if cursor_in:
                    anchor_tile = (cur_tx, cur_ty)
            elif not (lmb or rmb):
                if stroke_active:                  # stroke END -> commit the txn
                    log.commit()
                if stroke_dirty is not None:       # + re-bake +1
                    rebake_cells(*stroke_dirty)
                    stroke_dirty = None
                stroke_active = False
                last_paint_tile = None
            else:                  # held but over the HUD: break the line
                last_paint_tile = None

        # ---- ROOM -----------------------------------------------------------
        elif mode == "ROOM":
            if lmb_click and not over_hud:
                if selected_is_wall():
                    room_start = (clamp_tx, clamp_ty)
                else:
                    flash, flash_frames = (
                        f"ROOM wants a wall material — "
                        f"{palette[selected_id][0]} is not one", 180)
            if rmb_click:
                room_start = None
            if room_start is not None and lmb_up:
                rect = normalize_rect(room_start[0], room_start[1],
                                      cur_tx, cur_ty, grid_w, grid_h)
                room_start = None
                if rect is not None:
                    log.begin("room")
                    log.snapshot_grid("material")
                    changed = apply_room(grid, rect, selected_id, wall_codes)
                    log.commit()
                    if changed:
                        rebake_cells(*rect)
                        flash, flash_frames = (
                            f"room {rect[2] - rect[0] + 1}x"
                            f"{rect[3] - rect[1] + 1} ({changed} tiles)", 120)
                    else:
                        flash, flash_frames = "room changed nothing", 120

        # ---- CORRIDOR -------------------------------------------------------
        elif mode == "CORRIDOR":
            if _pressed(K.KEY_KP_ADD) or _pressed(K.KEY_EQUAL):
                corridor_w = min(CORRIDOR_MAX, corridor_w + 1)
            if _pressed(K.KEY_KP_SUBTRACT) or _pressed(K.KEY_MINUS):
                corridor_w = max(CORRIDOR_MIN, corridor_w - 1)
            if lmb_click and not over_hud:
                if selected_is_wall():
                    corr_start = (clamp_tx, clamp_ty)
                else:
                    flash, flash_frames = (
                        f"CORRIDOR lines its sides with a wall material — "
                        f"{palette[selected_id][0]} is not one", 180)
            if rmb_click:
                corr_start = None
            if corr_start is not None and lmb_up:
                x0_, y0_ = corr_start
                corr_start = None
                log.begin("corridor")
                log.snapshot_grid("material")
                changed = apply_corridor(
                    grid, x0_, y0_, clamp_tx, clamp_ty,
                    width=corridor_w, wall_id=selected_id,
                    wall_codes=wall_codes)
                txn = log.commit()
                if changed:
                    rebake_txn(txn)      # the op's whole-grid diff rect
                    flash, flash_frames = (
                        f"corridor w={corridor_w} ({changed} tiles)", 120)
                else:
                    flash, flash_frames = "corridor changed nothing", 120

        # ---- DOOR (Arc C3 — entity door: span drag + immediate stamp) ------
        elif mode == "DOOR":
            if rl.is_key_pressed(K.KEY_O) and door_drag is None:
                door_initial_state = ("open" if door_initial_state == "closed"
                                      else "closed")
                flash, flash_frames = (
                    f"placing {door_initial_state} doors (O)", 120)
            if door_tool_error is not None:
                if lmb_click and not over_hud:
                    flash, flash_frames = (
                        f"DOOR tool unavailable: {door_tool_error}", 240)
            else:
                if (lmb_click and not over_hud and cursor_in
                        and door_drag is None):
                    ok, orientation, why = door_entity_port.door_anchor_check(
                        grid, cur_tx, cur_ty, wall_codes)
                    if ok:
                        door_drag = {"anchor": (cur_tx, cur_ty),
                                     "orientation": orientation}
                    else:
                        flash, flash_frames = f"no door: {why}", 180
                if rmb_click and door_drag is not None:
                    door_drag = None
                if door_drag is not None and lmb_up:
                    orientation = door_drag["orientation"]
                    far = clamp_tx if orientation == "h" else clamp_ty
                    span, length_m = door_entity_port.plan_door_span(
                        grid, door_drag["anchor"], orientation, far,
                        tile_size_m, wall_codes)
                    n = len(span)
                    stamp_val = door_entity_port.stamp_value_for(
                        door_initial_state)
                    ax, ay = door_drag["anchor"]
                    new_id = light_entity_port.unique_entity_id(
                        "door", lights, entities)
                    instance = door_entity_port.build_door_instance(
                        ax, ay, orientation, length_m, door_initial_state,
                        new_id)
                    txn = door_entity_port.commit_door_placement(
                        log, grid, entities, span, stamp_val, instance)
                    rebake_txn(txn)
                    msg = (f"{new_id}: {orientation} x{n} tile"
                          f"{'s' if n != 1 else ''} ({length_m:.3f} m) "
                          f"[{door_initial_state}]")
                    if n < DEFAULT_FOOTPRINT:
                        msg += (f"  — narrower than the {DEFAULT_FOOTPRINT}"
                               f"-tile marine footprint (warning, not "
                               f"blocked)")
                    flash, flash_frames = msg, 220
                    door_drag = None

        # ---- SPAWN ----------------------------------------------------------
        elif mode == "SPAWN":
            hover = spawn_at(spawns, ftx, fty)
            if rl.is_key_pressed(K.KEY_T):
                if hover is not None:
                    log.begin("spawn team")
                    log.snapshot_coll("spawns")
                    s = spawns[hover]
                    new_team = (TEAM_ZOMBIE if int(s.team) == TEAM_MARINE
                                else TEAM_MARINE)
                    spawns[hover] = replace(s, team=new_team)
                    log.commit()
                    flash, flash_frames = (
                        f"{s.name} -> {TEAM_NAMES[new_team]}", 120)
                else:
                    spawn_team = (TEAM_ZOMBIE if spawn_team == TEAM_MARINE
                                  else TEAM_MARINE)
                    flash, flash_frames = (
                        f"placing {TEAM_NAMES[spawn_team]}s", 120)
            if lmb_click and not over_hud:
                if hover is not None:
                    # Drag: open the transaction at mouse-down (snapshot before
                    # any move); commit at release iff the spawn moved (§2.2).
                    spawn_drag = (hover, (spawns[hover].x, spawns[hover].y))
                    log.begin("move spawn")
                    log.snapshot_coll("spawns")
                elif cursor_in:
                    log.begin("place spawn")
                    log.snapshot_coll("spawns")
                    fp = DEFAULT_FOOTPRINT
                    nx = min(max(int(round(ftx - fp / 2.0)), 0), grid_w - fp)
                    ny = min(max(int(round(fty - fp / 2.0)), 0), grid_h - fp)
                    nm = unique_spawn_name(spawns, spawn_team)
                    spawns.append(SpawnEntry(name=nm, team=spawn_team,
                                             x=float(nx), y=float(ny),
                                             footprint=fp))
                    log.commit()
                    flash, flash_frames = f"spawn {nm} at ({nx}, {ny})", 120
            if spawn_drag is not None and lmb:
                di, _ = spawn_drag
                fp = int(spawns[di].footprint)
                nx = min(max(int(round(ftx - fp / 2.0)), 0), grid_w - fp)
                ny = min(max(int(round(fty - fp / 2.0)), 0), grid_h - fp)
                spawns[di] = replace(spawns[di], x=float(nx), y=float(ny))
            if spawn_drag is not None and lmb_up:
                spawn_drag = None
                log.commit()             # pushes iff the spawn actually moved
            if rmb_click and not over_hud and hover is not None:
                log.begin("delete spawn")
                log.snapshot_coll("spawns")
                gone = spawns.pop(hover)
                log.commit()
                flash, flash_frames = f"deleted spawn {gone.name}", 120

        # ---- LIGHT (P4 §2.4 — the SPAWN interaction template) --------------
        elif mode == "LIGHT":
            hover = light_at(lights, ftx, fty)
            if rl.is_key_pressed(K.KEY_B):        # static <-> beacon
                if hover is not None:
                    new_kind = ("beacon" if lights[hover].kind == "static"
                                else "static")
                    log.begin("light kind")
                    log.snapshot_coll("lights")
                    lights[hover] = replace(lights[hover], kind=new_kind)
                    log.commit()
                    flash, flash_frames = f"light -> {new_kind}", 120
                else:
                    light_kind = ("beacon" if light_kind == "static"
                                  else "static")
                    flash, flash_frames = f"placing {light_kind} lights", 120
            if rl.is_key_pressed(K.KEY_C):        # colour preset cycle
                step = -1 if shift else 1
                if hover is not None:
                    nc = next_light_color(lights[hover].color, step)
                    log.begin("light color")
                    log.snapshot_coll("lights")
                    lights[hover] = replace(lights[hover], color=nc)
                    log.commit()
                    flash, flash_frames = (
                        f"light color: {light_color_name(nc)}", 120)
                else:
                    light_color = next_light_color(light_color, step)
                    flash, flash_frames = (
                        f"placing {light_color_name(light_color)} lights",
                        120)
            # Parameter nudges — key = up, Shift+key = down, HOVERED light
            # only (P period / X beam / H phase are beacon parameters).
            for key, attr, step, lo, hi, beacon_only in (
                    (K.KEY_R, "range", LIGHT_RANGE_STEP,
                     LIGHT_RANGE_MIN, LIGHT_RANGE_MAX, False),
                    (K.KEY_E, "intensity", LIGHT_INTENSITY_STEP,
                     LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX, False),
                    (K.KEY_P, "period_s", LIGHT_PERIOD_STEP,
                     LIGHT_PERIOD_MIN, LIGHT_PERIOD_MAX, True),
                    (K.KEY_X, "beam_deg", LIGHT_BEAM_STEP,
                     LIGHT_BEAM_MIN, LIGHT_BEAM_MAX, True)):
                if not rl.is_key_pressed(key):
                    continue
                if hover is None:
                    flash, flash_frames = (
                        f"hover a light to change {attr}", 120)
                elif beacon_only and lights[hover].kind != "beacon":
                    flash, flash_frames = (
                        f"{attr} is a beacon parameter (B toggles kind)", 120)
                else:
                    v = getattr(lights[hover], attr) + (-step if shift
                                                        else step)
                    v = min(hi, max(lo, v))
                    log.begin(attr)
                    log.snapshot_coll("lights")
                    lights[hover] = replace(lights[hover], **{attr: v})
                    log.commit()
                    flash, flash_frames = f"{attr} = {v:g}", 120
            if rl.is_key_pressed(K.KEY_H):        # phase wraps mod 1 turn
                if hover is None:
                    flash, flash_frames = "hover a light to change phase", 120
                elif lights[hover].kind != "beacon":
                    flash, flash_frames = (
                        "phase is a beacon parameter (B toggles kind)", 120)
                else:
                    v = (lights[hover].phase
                         + (-LIGHT_PHASE_STEP if shift
                            else LIGHT_PHASE_STEP)) % 1.0
                    log.begin("phase")
                    log.snapshot_coll("lights")
                    lights[hover] = replace(lights[hover], phase=v)
                    log.commit()
                    flash, flash_frames = f"phase = {v:g} turns", 120
            if lmb_click and not over_hud:
                if hover is not None:
                    # Drag: snapshot at mouse-down, commit at release if moved.
                    light_drag = (hover, (lights[hover].x, lights[hover].y))
                    log.begin("move light")
                    log.snapshot_coll("lights")
                elif cursor_in:
                    log.begin("place light")
                    log.snapshot_coll("lights")
                    # Tile-center snap (engine/15 §2.2: centers at .5).
                    lx, ly = cur_tx + 0.5, cur_ty + 0.5
                    # Mint over the UNION of lights + entities (B3): both
                    # serialize to the ONE [[entity]] family, ids unique or the
                    # saved level is unloadable.
                    new_id = light_entity_port.unique_entity_id(
                        "light", lights, entities)
                    lights.append(light_entity_port.EditableLight(
                        x=lx, y=ly, color=light_color, kind=light_kind,
                        id=new_id))
                    log.commit()
                    flash, flash_frames = (
                        f"{light_kind} light at ({lx:g}, {ly:g}) "
                        f"[{light_color_name(light_color)}]", 120)
            if light_drag is not None and lmb:
                di, _ = light_drag
                if cursor_in:
                    lights[di] = replace(lights[di],
                                         x=cur_tx + 0.5, y=cur_ty + 0.5)
            if light_drag is not None and lmb_up:
                light_drag = None
                log.commit()             # pushes iff the light actually moved
            if rmb_click and not over_hud and hover is not None:
                log.begin("delete light")
                log.snapshot_coll("lights")
                gone = lights.pop(hover)
                log.commit()
                flash, flash_frames = (
                    f"deleted light at ({gone.x:g}, {gone.y:g})", 120)

        # ---- WATER (P5 §2.4 — bucket-fill to depth) -------------------------
        elif mode == "WATER":
            if _pressed(K.KEY_KP_ADD) or _pressed(K.KEY_EQUAL):
                water_depth_m = min(WATER_DEPTH_MAX_M,
                                    round(water_depth_m
                                          + WATER_DEPTH_STEP_M, 1))
            if _pressed(K.KEY_KP_SUBTRACT) or _pressed(K.KEY_MINUS):
                water_depth_m = max(WATER_DEPTH_MIN_M,
                                    round(water_depth_m
                                          - WATER_DEPTH_STEP_M, 1))
            if (lmb_click or rmb_click) and not over_hud:
                region, why = water_fill_region(grid, cur_tx, cur_ty,
                                                water_solid)
                if region is None:
                    flash, flash_frames = f"no fill: {why}", 180
                else:
                    # Quantize AT FILL TIME (P5 §2.4): the stored state is
                    # int32 Q16.16; RMB is fill-to-dry (target 0).
                    target = (0 if rmb_click
                              else int(water_fixed.quantize(water_depth_m)))
                    if all(int(water_q[ty_, tx_]) == target
                           for tx_, ty_ in region):
                        flash, flash_frames = (
                            "region already dry" if rmb_click else
                            f"region already at {water_depth_m:.1f} m", 120)
                    else:
                        log.begin("drain" if rmb_click else "water")
                        log.snapshot_grid("water")
                        for tx_, ty_ in region:
                            water_q[ty_, tx_] = target
                        log.commit()
                        if rmb_click:
                            flash, flash_frames = (
                                f"drained {len(region)} tiles", 120)
                        else:
                            msg = (f"filled {len(region)} tiles to "
                                   f"{water_depth_m:.1f} m")
                            if water_depth_m > WATER_DEEP_HINT_M:
                                msg += ("  — deep tank: drains may "
                                        "flash-boil (by design)")
                            flash, flash_frames = msg, 180

        # ---- ENTITY (Arc C3 — sensor placement + generic place-one) --------
        elif mode == "ENTITY":
            cls_name = selected_entity_class
            cls_payload = (registry_payload.get("classes", {}).get(cls_name)
                          if cls_name else None)
            if cls_name is None:
                pass
            elif cls_payload is None:
                if lmb_click and not over_hud:
                    flash, flash_frames = (
                        f"{cls_name}: not in the loaded registry "
                        f"(stale selection?)", 180)
            elif cls_name == door_entity_port.DOOR_CLASS:
                if lmb_click and not over_hud:
                    flash, flash_frames = "use the DOOR tool (F4) instead", 150
            elif cls_name == light_entity_port.LIGHT_CLASS:
                if lmb_click and not over_hud:
                    flash, flash_frames = "use the LIGHT tool (F6) instead", 150
            elif sensor_entity_port.is_field_sensor(cls_name):
                # Click+drag: the mount is the anchor; dragging sets the
                # AIR-family sample_dx/dy facing (BODY family always samples
                # its own mount — no drag effect). Validation happens ONLY
                # at release, BEFORE any log interaction (§6.2): a refused
                # placement never opens a transaction.
                family = sensor_entity_port.sample_family(cls_name)
                if (lmb_click and not over_hud and cursor_in
                        and entity_drag is None):
                    entity_drag = {"class": cls_name,
                                  "anchor": (cur_tx, cur_ty)}
                if rmb_click and entity_drag is not None:
                    entity_drag = None
                if (entity_drag is not None
                        and entity_drag["class"] == cls_name and lmb_up):
                    ax, ay = entity_drag["anchor"]
                    if family == sensor_entity_port.SAMPLE_AIR:
                        dx, dy = clamp_tx - ax, clamp_ty - ay
                    else:
                        dx, dy = 0, 0
                    sample_tile = sensor_entity_port.resolve_sample_tile(
                        cls_name, {"x": ax, "y": ay,
                                  "sample_dx": dx, "sample_dy": dy})
                    if not sensor_entity_port.tile_is_open(
                            grid, sample_tile, water_solid, SPACE_CODE):
                        flash, flash_frames = (
                            f"refused: {cls_name}'s sample tile "
                            f"{sample_tile} is solid/vacuum", 220)
                    else:
                        new_id = light_entity_port.unique_entity_id(
                            cls_name, lights, entities)
                        instance = sensor_entity_port.build_sensor_instance(
                            cls_name, ax, ay, dx, dy, new_id)
                        sensor_entity_port.commit_sensor_placement(
                            log, entities, instance)
                        flash, flash_frames = (
                            f"{new_id} at ({ax},{ay}) sampling "
                            f"({sample_tile[1]},{sample_tile[0]})", 180)
                    entity_drag = None
            else:
                # Generic place-one: any OTHER registered class (logic
                # nodes, pump, button/terminal, clock, sensor_motion). A
                # class needing a field this tool can't fill (a zone's
                # zone_id — C5's paint tool) is refused rather than
                # authoring an invalid instance.
                field_names = {f["name"] for f in cls_payload["fields"]}
                has_xy = "x" in field_names and "y" in field_names
                required = entity_editor_ui.required_field_names(cls_payload)
                unfillable = [f for f in required if f not in ("x", "y")]
                clickable = not over_hud and (cursor_in if has_xy else True)
                if lmb_click and clickable:
                    if unfillable:
                        flash, flash_frames = (
                            f"{cls_name}: needs {', '.join(unfillable)} — "
                            f"no placement tool for this yet", 220)
                    else:
                        fields = entity_editor_ui.default_instance_fields(
                            cls_payload,
                            x=(cur_tx if has_xy else None),
                            y=(cur_ty if has_xy else None))
                        new_id = light_entity_port.unique_entity_id(
                            cls_name, lights, entities)
                        instance = EntityInstance(
                            id=new_id, class_name=cls_name, ordinal=0,
                            tags=(), fields=fields, authored_keys=required)
                        log.begin(f"place {cls_name}")
                        log.snapshot_coll("entities")
                        entities.append(instance)
                        log.commit()
                        flash, flash_frames = (
                            f"{new_id} placed"
                            + (f" at ({cur_tx},{cur_ty})" if has_xy else ""),
                            180)

        # ---- SELECT (Arc C4 — unified selection over every [[entity]]
        # instance: doors, sensors, generic placements, lights; SPAWN is
        # NEVER folded in, canon §0/A1) + move/delete/tag-assign/inspector-
        # field-edit/clump-copy-paste. The transaction-committing logic
        # lives in entity_selection.py (headless, unit-tested); this block
        # is the thin interactive shell (smoke-tested only, --auto has no
        # input injection).
        elif mode == "SELECT":
            hover = (entity_selection.hit_test(cur_tx, cur_ty, lights,
                                               entities, tile_size_m)
                    if cursor_in else None)
            # Prune ids something else deleted (LIGHT/SPAWN's own bespoke
            # RMB-delete, or a since-undone paste) before acting this frame.
            selection = {i for i in selection
                        if entity_selection.find_instance(
                            i, lights, entities)[0] is not None}

            if text_entry is not None:
                # The ONE scoped free-text gesture (tag name / id rename —
                # see the _SLUG_RE/_SLUG_CHARS comment above): drain this
                # frame's typed codepoints; Enter commits, Backspace erases,
                # Esc (handled in the global block above) cancels.
                ch = rl.get_char_pressed()
                while ch != 0:
                    c = chr(ch)
                    if c in _SLUG_CHARS:
                        text_entry["buf"] += c
                    ch = rl.get_char_pressed()
                if rl.is_key_pressed(K.KEY_BACKSPACE):
                    text_entry["buf"] = text_entry["buf"][:-1]
                if rl.is_key_pressed(K.KEY_ENTER):
                    buf = text_entry["buf"]
                    if not _SLUG_RE.match(buf):
                        flash, flash_frames = (
                            f"{text_entry['kind']} {buf!r}: letters/digits/"
                            f"_/- only", 200)
                    elif text_entry["kind"] == "tag":
                        entity_selection.commit_assign_tag(
                            log, lights, entities, list(selection), buf)
                        flash, flash_frames = (
                            f"tagged {len(selection)} instance"
                            f"{'s' if len(selection) != 1 else ''} '{buf}'",
                            160)
                        text_entry = None
                    else:                                      # "id"
                        sel_id = next(iter(selection))
                        ok, result = entity_selection.commit_field_edit(
                            log, grid, lights, entities, sel_id, "id",
                            buf, tile_size_m)
                        if ok:
                            selection = {buf}
                            flash, flash_frames = f"renamed to '{buf}'", 160
                            text_entry = None
                        else:
                            flash, flash_frames = f"refused: {result}", 200
            else:
                if rl.is_key_pressed(K.KEY_A):        # select-by-class
                    basis = hover
                    if basis is None and len(selection) == 1:
                        basis = next(iter(selection))
                    if basis is not None:
                        coll_name, idx = entity_selection.find_instance(
                            basis, lights, entities)
                        if coll_name is not None:
                            obj = (lights[idx] if coll_name == "lights"
                                  else entities[idx])
                            cls = entity_selection.instance_class_name(
                                coll_name, obj)
                            selection = set(entity_selection.select_by_class(
                                cls, lights, entities))
                            flash, flash_frames = (
                                f"selected {len(selection)} {cls}", 140)
                if rl.is_key_pressed(K.KEY_T) and selection:
                    text_entry = {"kind": "tag", "buf": ""}
                if rl.is_key_pressed(K.KEY_I) and len(selection) == 1:
                    text_entry = {"kind": "id", "buf": next(iter(selection))}
                if rl.is_key_pressed(K.KEY_DELETE) and selection:
                    # Arc C5 (editor design §5): deleting a zone instance
                    # PROMPTS to clear its paint — a second Delete press (the
                    # selection unchanged since arming) confirms "clear it
                    # too"; Backspace confirms "keep it" (orphaned paint,
                    # the loader's own validator just warns). A selection
                    # change while armed restarts the prompt from scratch
                    # rather than acting on stale state.
                    if (pending_zone_delete is not None
                            and set(pending_zone_delete["ids"]) != selection):
                        pending_zone_delete = None
                    if pending_zone_delete is None:
                        zone_ids_here = set()
                        for sid in selection:
                            cn, idx = entity_selection.find_instance(
                                sid, lights, entities)
                            if (cn == "entities"
                                    and entities[idx].class_name
                                    in ZONE_CLASSES):
                                zone_ids_here.add(
                                    int(entities[idx].fields["zone_id"]))
                        if zone_ids_here:
                            pending_zone_delete = {
                                "ids": list(selection),
                                "zone_ids": zone_ids_here}
                            n = len(zone_ids_here)
                            flash, flash_frames = (
                                f"deleting {n} zone"
                                f"{'s' if n != 1 else ''} — Delete again to "
                                f"ALSO clear its zones.npy paint, Backspace "
                                f"to keep it (orphaned, validator warns)",
                                260)
                        else:
                            txn = entity_selection.commit_delete_selection(
                                log, grid, lights, entities, wires,
                                list(selection), tile_size_m, zones=zones,
                                clear_zone_ids=set())
                            rebake_txn(txn)
                            flash, flash_frames = (
                                f"deleted {len(selection)}", 140)
                            selection = set()
                    else:
                        txn = entity_selection.commit_delete_selection(
                            log, grid, lights, entities, wires,
                            pending_zone_delete["ids"], tile_size_m,
                            zones=zones,
                            clear_zone_ids=pending_zone_delete["zone_ids"])
                        rebake_txn(txn)
                        flash, flash_frames = (
                            f"deleted {len(pending_zone_delete['ids'])} "
                            f"(zone paint cleared)", 160)
                        selection = set()
                        pending_zone_delete = None
                if (pending_zone_delete is not None
                        and rl.is_key_pressed(K.KEY_BACKSPACE)):
                    txn = entity_selection.commit_delete_selection(
                        log, grid, lights, entities, wires,
                        pending_zone_delete["ids"], tile_size_m)
                    rebake_txn(txn)
                    flash, flash_frames = (
                        f"deleted {len(pending_zone_delete['ids'])} (paint "
                        f"kept, orphaned)", 160)
                    selection = set()
                    pending_zone_delete = None
                if ctrl and rl.is_key_pressed(K.KEY_C) and selection:
                    clump_clipboard = entity_selection.compute_clump_copy(
                        list(selection), lights, entities, wires)
                    flash, flash_frames = (
                        f"copied {len(selection)} (+ "
                        f"{len(clump_clipboard['wires'])} internal wires)",
                        160)
                if (ctrl and rl.is_key_pressed(K.KEY_V)
                        and clump_clipboard is not None and cursor_in):
                    anchor = entity_selection.clump_anchor_tile(
                        clump_clipboard)
                    dtx, dty = ((cur_tx - anchor[0], cur_ty - anchor[1])
                               if anchor is not None else (0, 0))
                    txn, new_ids = entity_selection.commit_clump_paste(
                        log, grid, lights, entities, wires, clump_clipboard,
                        dtx, dty, tile_size_m)
                    if txn is not None:
                        rebake_txn(txn)
                        selection = set(new_ids)
                        flash, flash_frames = f"pasted {len(new_ids)}", 160

                # Single-selection field focus + nudge/cycle — multi-select
                # editing is tag-assign only (unambiguous over mixed
                # classes; T above covers it regardless of selection size).
                if len(selection) == 1:
                    sel_id = next(iter(selection))
                    coll_name, idx = entity_selection.find_instance(
                        sel_id, lights, entities)
                    if coll_name is not None:
                        obj = (lights[idx] if coll_name == "lights"
                              else entities[idx])
                        cls_name = entity_selection.instance_class_name(
                            coll_name, obj)
                        cls_payload = registry_payload.get(
                            "classes", {}).get(cls_name)
                        if cls_payload is not None:
                            values = (
                                light_entity_port.light_entry_to_fields(obj)
                                if coll_name == "lights" else obj.fields)
                            rows = entity_editor_ui.inspector_rows(
                                cls_payload, values)
                            editable_rows = [r for r in rows if r.editable]
                            if editable_rows:
                                select_focus_idx %= len(editable_rows)
                                if _pressed(K.KEY_LEFT_BRACKET):
                                    select_focus_idx = (
                                        (select_focus_idx - 1)
                                        % len(editable_rows))
                                if _pressed(K.KEY_RIGHT_BRACKET):
                                    select_focus_idx = (
                                        (select_focus_idx + 1)
                                        % len(editable_rows))
                                focus = editable_rows[select_focus_idx]
                                new_value = None
                                eui_ = entity_editor_ui
                                if focus.kind in (eui_.KIND_INT, eui_.KIND_Q16):
                                    if _pressed(K.KEY_EQUAL):
                                        new_value = int(focus.value) + 1
                                    if _pressed(K.KEY_MINUS):
                                        new_value = int(focus.value) - 1
                                elif focus.kind in (eui_.KIND_LENGTH_M,
                                                    eui_.KIND_FLOAT_RENDER):
                                    if _pressed(K.KEY_EQUAL):
                                        new_value = round(
                                            float(focus.value) + 0.1, 3)
                                    if _pressed(K.KEY_MINUS):
                                        new_value = round(
                                            float(focus.value) - 0.1, 3)
                                elif focus.kind == eui_.KIND_BOOL:
                                    if rl.is_key_pressed(K.KEY_ENTER):
                                        new_value = not focus.value
                                elif (focus.kind == eui_.KIND_ENUM
                                      and focus.choices):
                                    if rl.is_key_pressed(K.KEY_ENTER):
                                        choices = list(focus.choices)
                                        ci = (choices.index(focus.value)
                                             if focus.value in choices
                                             else 0)
                                        step = -1 if shift else 1
                                        new_value = choices[
                                            (ci + step) % len(choices)]
                                elif (focus.kind == eui_.KIND_COLOR_RGB
                                      and not ctrl):
                                    if rl.is_key_pressed(K.KEY_C):
                                        step = -1 if shift else 1
                                        new_value = next_color_rgb_int(
                                            focus.value, step)
                                if new_value is not None:
                                    if (focus.minimum is not None
                                            and isinstance(
                                                new_value, (int, float))):
                                        new_value = max(focus.minimum,
                                                        new_value)
                                    if (focus.maximum is not None
                                            and isinstance(
                                                new_value, (int, float))):
                                        new_value = min(focus.maximum,
                                                        new_value)
                                    ok, result = (
                                        entity_selection.commit_field_edit(
                                            log, grid, lights, entities,
                                            sel_id, focus.name, new_value,
                                            tile_size_m))
                                    if ok:
                                        rebake_txn(result)
                                        flash, flash_frames = (
                                            f"{focus.name} = {new_value}",
                                            120)
                                    else:
                                        flash, flash_frames = (
                                            f"refused: {result}", 180)

                # Mouse: click/shift-click select, box-select, move-drag.
                if lmb_click and not over_hud and cursor_in:
                    if hover is not None:
                        if shift:
                            selection = set(selection)
                            if hover in selection:
                                selection.discard(hover)
                            else:
                                selection.add(hover)
                        else:
                            if hover not in selection:
                                selection = {hover}
                            select_move = {"anchor": (cur_tx, cur_ty)}
                    else:
                        select_box = {"anchor": (cur_tx, cur_ty),
                                     "add": shift}
                if rmb_click and not over_hud:
                    if select_box is not None or select_move is not None:
                        select_box = select_move = None
                    else:
                        selection = set()
                if select_move is not None and lmb_up:
                    ax, ay = select_move["anchor"]
                    dtx, dty = cur_tx - ax, cur_ty - ay
                    txn = entity_selection.commit_move_selection(
                        log, grid, lights, entities, list(selection),
                        dtx, dty, tile_size_m)
                    if txn is not None:
                        rebake_txn(txn)
                    select_move = None
                if select_box is not None and lmb_up:
                    ax, ay = select_box["anchor"]
                    ids = entity_selection.box_select(
                        ax, ay, cur_tx, cur_ty, lights, entities,
                        tile_size_m)
                    if select_box["add"]:
                        selection |= set(ids)
                    else:
                        selection = set(ids)
                    select_box = None

        # ---- WAND (Arc C5 — enclosure fill / same-code select, editor doc
        # §7): paints the MATERIAL palette's `selected_id` (9/SPACE included
        # — vacuum is simply that id) over the region either algorithm
        # resolves. One `GridCellsOp("material")` transaction per paint,
        # via the same generic committer AIR/ZONE reuse.
        elif mode == "WAND":
            if (lmb_click or rmb_click) and not over_hud and cursor_in:
                if lmb_click:
                    region = enclosure_fill_region(
                        water_open_mask(grid, water_solid), cur_tx, cur_ty)
                    verb = "enclosure-fill"
                else:
                    region = same_code_region(grid, cur_tx, cur_ty)
                    verb = "same-code select"
                mat_name = palette.get(selected_id, ("?", None))[0]
                if region is None:
                    flash, flash_frames = f"wand {verb}: no region here", 160
                elif all(int(grid[ty_, tx_]) == selected_id
                        for tx_, ty_ in region):
                    flash, flash_frames = (
                        f"wand {verb}: region already {mat_name}", 120)
                else:
                    log.begin(f"wand {verb}")
                    log.snapshot_grid("material")
                    for tx_, ty_ in region:
                        grid[ty_, tx_] = selected_id
                    txn = log.commit()
                    rebake_txn(txn)
                    flash, flash_frames = (
                        f"wand {verb}: painted {len(region)} tiles "
                        f"{mat_name}", 180)

        # ---- ZONE (Arc C5, editor doc §5): paint zones.npy, one instance
        # per painted id. LMB enclosure-fills the current paint TARGET (a
        # brand-new id also authors its [[entity]] instance, in the SAME
        # compound transaction — the C3 DOOR archetype); RMB same-code-
        # selects the id under the cursor and clears its paint; I eyedrops
        # the id/class under the cursor as the new target; [ / ] cycle the
        # target among "new" + every existing id.
        elif mode == "ZONE":
            if rl.is_key_pressed(K.KEY_C):
                idx = ZONE_PAINT_CLASSES.index(zone_paint_class)
                zone_paint_class = ZONE_PAINT_CLASSES[
                    (idx + 1) % len(ZONE_PAINT_CLASSES)]
                flash, flash_frames = (
                    f"new zones paint as {zone_paint_class}", 140)
            if _pressed(K.KEY_LEFT_BRACKET) or _pressed(K.KEY_RIGHT_BRACKET):
                options = [None] + sorted(
                    zone_entity_port.zone_ids_in_use(entities))
                idx = (options.index(zone_paint_id)
                      if zone_paint_id in options else 0)
                step = -1 if _pressed(K.KEY_LEFT_BRACKET) else 1
                zone_paint_id = options[(idx + step) % len(options)]
                if zone_paint_id is not None:
                    claim = zone_entity_port.find_zone_claim(
                        entities, zone_paint_id)
                    if claim is not None:
                        zone_paint_class = claim.class_name
                label = "new" if zone_paint_id is None else str(zone_paint_id)
                flash, flash_frames = f"zone paint target: {label}", 120
            if rl.is_key_pressed(K.KEY_I) and cursor_in and zones is not None:
                under = int(zones[cur_ty, cur_tx])
                if under:
                    zone_paint_id = under
                    claim = zone_entity_port.find_zone_claim(entities, under)
                    if claim is not None:
                        zone_paint_class = claim.class_name
                    flash, flash_frames = f"picked up zone {under}", 120
                else:
                    zone_paint_id = None
                    flash, flash_frames = "target: new zone", 120
            if lmb_click and not over_hud and cursor_in:
                region = enclosure_fill_region(
                    water_open_mask(grid, water_solid), cur_tx, cur_ty)
                if region is None:
                    flash, flash_frames = "zone: no enclosed region here", 160
                else:
                    zid = (zone_paint_id if zone_paint_id is not None
                          else zone_entity_port.next_zone_id(entities))
                    existing_claim = zone_entity_port.find_zone_claim(
                        entities, zid)
                    cls_payload = (None if existing_claim is not None else
                                  registry_payload.get(
                                      "classes", {}).get(zone_paint_class))
                    if existing_claim is None and cls_payload is None:
                        flash, flash_frames = (
                            f"{zone_paint_class}: not in the loaded "
                            f"registry (stale selection?)", 200)
                    else:
                        new_instance = None
                        if existing_claim is None:
                            new_eid = light_entity_port.unique_entity_id(
                                zone_paint_class, lights, entities)
                            new_instance = (
                                zone_entity_port.build_zone_instance(
                                    cls_payload, zone_paint_class, zid,
                                    new_eid))
                        if zones is None:
                            zones = np.zeros(grid.shape, dtype=np.uint8)
                            undo_ctx.grids["zones"] = zones
                        zone_entity_port.commit_zone_paint(
                            log, zones, entities, region, zid, new_instance)
                        zone_paint_id = zid
                        msg = (f"zone {zid} ({zone_paint_class}): painted "
                              f"{len(region)} tiles")
                        if new_instance is not None:
                            msg += f", new instance {new_instance.id}"
                        flash, flash_frames = msg, 200
            if rmb_click and not over_hud and cursor_in and zones is not None:
                under = int(zones[cur_ty, cur_tx])
                if not under:
                    flash, flash_frames = "zone: nothing painted here", 120
                else:
                    region = same_code_region(zones, cur_tx, cur_ty)
                    zone_entity_port.commit_zone_clear(log, zones, region)
                    flash, flash_frames = (
                        f"cleared {len(region)} tiles of zone {under}", 160)

        # ---- AIR (Arc C5, editor doc §7): paint air_init.npy, hull-leak
        # gated. LMB fills the enclosed region to the current pressure (only
        # when the fill is SEALED — a leaky fill paints nothing); RMB fills
        # to 0 atm (a depressurized start, same "drain" convention as WATER).
        elif mode == "AIR":
            if _pressed(K.KEY_KP_ADD) or _pressed(K.KEY_EQUAL):
                air_pressure_atm = min(
                    AIR_PRESSURE_MAX_ATM,
                    round(air_pressure_atm + AIR_PRESSURE_STEP_ATM, 2))
            if _pressed(K.KEY_KP_SUBTRACT) or _pressed(K.KEY_MINUS):
                air_pressure_atm = max(
                    AIR_PRESSURE_MIN_ATM,
                    round(air_pressure_atm - AIR_PRESSURE_STEP_ATM, 2))
            if (lmb_click or rmb_click) and not over_hud and cursor_in:
                region, why = air_paint.plan_air_fill(
                    grid, water_solid, cur_tx, cur_ty)
                if region is None:
                    flash, flash_frames = f"air: {why}", 220
                else:
                    target = (0 if rmb_click
                             else air_paint.quantize_atm(air_pressure_atm))
                    if air is None:
                        air = air_paint.default_ambient_grid(grid.shape)
                        undo_ctx.grids["air"] = air
                    if all(int(air[ty_, tx_]) == target
                          for tx_, ty_ in region):
                        flash, flash_frames = (
                            "air: region already at that pressure", 120)
                    else:
                        log.begin("drain air" if rmb_click else "air")
                        log.snapshot_grid("air")
                        for tx_, ty_ in region:
                            air[ty_, tx_] = target
                        log.commit()
                        if rmb_click:
                            flash, flash_frames = (
                                f"depressurized {len(region)} tiles", 160)
                        else:
                            flash, flash_frames = (
                                f"sealed fill: seeded {len(region)} tiles to "
                                f"{air_pressure_atm:.2f} atm", 180)

        # ---- PROPS (Arc C5, editor doc §7 — a level-properties pane; no
        # canvas edits): B toggles `boundary` — format + level_lib writeback
        # only, the physics it will eventually drive is a later project's
        # (canon §7).
        elif mode == "PROPS":
            if rl.is_key_pressed(K.KEY_B):
                level_boundary = (
                    BOUNDARY_MODES[1] if level_boundary == BOUNDARY_MODES[0]
                    else BOUNDARY_MODES[0])
                flash, flash_frames = (
                    f"boundary = {level_boundary} (Ctrl+S saves; format "
                    f"only — no physics change here)", 200)

        # ---- WIRE (Arc C6, editor doc §8 — the two-click wire tool, NEVER a
        # drag): click 1 resolves a SOURCE + its output signal (multiple
        # signals: `[`/`]` cycle, default the primary non-alive one); pan/
        # zoom/scroll stay live while a wire is pending; click 2 resolves a
        # TARGET + its input (multiple inputs: `[`/`]` cycle, Enter
        # confirms; a single-input target commits immediately — "two clicks,
        # done"). T cycles an alternate TAG target (a `tag:name.input` wire
        # — offered once a source is picked). Esc cancels (global block
        # above). With no pending wire, LMB on an EXISTING wire (line or tag
        # badge) selects it for removal (Delete); RMB clears the selection.
        # All pure logic lives in wire_tool.py — this block only reads
        # input and calls it, matching every prior Arc C tool's split.
        elif mode == "WIRE":
            if wire_pending is None:
                if lmb_click and not over_hud:
                    hit = wire_tool.hit_test_with_layout(
                        cur_tx, cur_ty, lights, entities, tile_size_m,
                        node_layout)
                    if hit is not None:
                        cls_name, cls_payload = wire_tool.class_payload_of(
                            hit, lights, entities, registry_payload)
                        if cls_payload is None:
                            flash, flash_frames = (
                                f"{cls_name}: not in the loaded registry "
                                f"(stale?)", 180)
                        else:
                            wire_pending = wire_tool.begin_pending_wire(
                                hit, cls_payload)
                            sig = wire_pending["signals"][
                                wire_pending["signal_idx"]]
                            flash, flash_frames = (
                                f"wire source {hit}.{sig} — click a target "
                                f"([ ] cycles signal, T = tag target)", 220)
                    else:
                        lines, badges = wire_tool.wire_endpoints_for_draw(
                            wires, lights, entities, tile_size_m, node_layout)
                        wire_selected = wire_tool.hit_test_wire(
                            cur_tx, cur_ty, lines, badges)
                if rmb_click and not over_hud:
                    wire_selected = None
                if (wire_selected is not None
                        and rl.is_key_pressed(K.KEY_DELETE)):
                    txn = wire_tool.commit_remove_wire(
                        log, wires, wire_selected)
                    if txn is not None:
                        flash, flash_frames = "wire removed", 140
                    wire_selected = None
            else:
                if wire_pending["to_spec"] is None:
                    # Phase 2: source known, picking the signal / an
                    # optional tag target, awaiting a target click.
                    if _pressed(K.KEY_LEFT_BRACKET):
                        wire_pending = wire_tool.cycle_pending_signal(
                            wire_pending, -1)
                    if _pressed(K.KEY_RIGHT_BRACKET):
                        wire_pending = wire_tool.cycle_pending_signal(
                            wire_pending, 1)
                    if rl.is_key_pressed(K.KEY_T):
                        tags = wire_tool.all_tags(lights, entities)
                        if not tags:
                            flash, flash_frames = (
                                "no tags in this level yet", 160)
                        else:
                            cur_name = wire_pending.get("tag_name")
                            cur_idx = (tags.index(cur_name)
                                      if cur_name in tags else -1)
                            nxt = tags[(cur_idx + 1) % len(tags)]
                            p2, reason = wire_tool.set_pending_target_tag(
                                wire_pending, nxt, lights, entities,
                                registry_payload)
                            if p2 is None:
                                flash, flash_frames = (
                                    f"tag:{nxt}: {reason}", 200)
                            else:
                                wire_pending = p2
                    if lmb_click and not over_hud:
                        hit = wire_tool.hit_test_with_layout(
                            cur_tx, cur_ty, lights, entities, tile_size_m,
                            node_layout)
                        if hit is not None and hit != wire_pending["from_id"]:
                            _cn, cls_payload = wire_tool.class_payload_of(
                                hit, lights, entities, registry_payload)
                            if cls_payload is None:
                                flash, flash_frames = (
                                    "target: not in the loaded registry",
                                    180)
                            else:
                                p2, reason = wire_tool.set_pending_target_id(
                                    wire_pending, hit, cls_payload)
                                if p2 is None:
                                    flash, flash_frames = (
                                        f"refused: {reason}", 200)
                                else:
                                    wire_pending = p2

                if (wire_pending is not None
                        and wire_pending["to_spec"] is not None):
                    # Phase 3: target known. A single-input target commits
                    # right away (two clicks, done); multiple inputs need an
                    # explicit pick (`[`/`]` + Enter) before committing.
                    if len(wire_pending["inputs"]) <= 1:
                        from_str, to_str = wire_tool.pending_wire_strings(
                            wire_pending)
                        from_id_, signal_ = from_str.split(".", 1)
                        to_spec_, input_ = to_str.split(".", 1)
                        ok, result = wire_tool.commit_add_wire(
                            log, wires, lights, entities, spawns,
                            from_id_, signal_, to_spec_, input_)
                        flash, flash_frames = (
                            (f"wired {from_str} -> {to_str}", 220) if ok
                            else (f"refused: {result}", 240))
                        wire_pending = None
                    else:
                        if _pressed(K.KEY_LEFT_BRACKET):
                            wire_pending = wire_tool.cycle_pending_input(
                                wire_pending, -1)
                        if _pressed(K.KEY_RIGHT_BRACKET):
                            wire_pending = wire_tool.cycle_pending_input(
                                wire_pending, 1)
                        if rl.is_key_pressed(K.KEY_ENTER):
                            from_str, to_str = (
                                wire_tool.pending_wire_strings(wire_pending))
                            from_id_, signal_ = from_str.split(".", 1)
                            to_spec_, input_ = to_str.split(".", 1)
                            ok, result = wire_tool.commit_add_wire(
                                log, wires, lights, entities, spawns,
                                from_id_, signal_, to_spec_, input_)
                            flash, flash_frames = (
                                (f"wired {from_str} -> {to_str}", 220) if ok
                                else (f"refused: {result}", 240))
                            wire_pending = None
                if rmb_click and not over_hud:
                    wire_pending = None

        # ---- undo / redo / save ---------------------------------------------
        # ONE global history across ALL domains (§7.2 — the intended behavior
        # change): Ctrl+Z undoes the most-recent action irrespective of the
        # current mode. Explicit shift guards (§4/C9): Ctrl+Shift+Z is redo, not
        # undo (today's `ctrl and Z` had no shift check). Undo/redo are refused
        # while a gesture is open — commit/force-end it first.
        undo_key = (not typing and ctrl and (not shift)
                   and rl.is_key_pressed(K.KEY_Z))
        redo_key = (not typing and ctrl
                   and (rl.is_key_pressed(K.KEY_Y)
                        or (shift and rl.is_key_pressed(K.KEY_Z))))
        if undo_key:
            if log.has_pending:
                flash, flash_frames = "release the mouse before undo", 120
            else:
                spawn_drag = light_drag = None   # stale drag indices
                door_drag = entity_drag = None    # stale span/arrow drags
                select_box = select_move = None   # stale box/move drags
                txn = log.undo()
                if txn is None:
                    flash, flash_frames = "nothing to undo", 120
                else:
                    rebake_txn(txn)              # material grid rects only
                    flash, flash_frames = (
                        f"undo: {txn.label} (undo {log.undo_count} / "
                        f"redo {log.redo_count})", 120)
        elif redo_key:
            if log.has_pending:
                flash, flash_frames = "release the mouse before redo", 120
            else:
                spawn_drag = light_drag = None
                door_drag = entity_drag = None
                select_box = select_move = None
                txn = log.redo()
                if txn is None:
                    flash, flash_frames = "nothing to redo", 120
                else:
                    rebake_txn(txn)
                    flash, flash_frames = (
                        f"redo: {txn.label} (undo {log.undo_count} / "
                        f"redo {log.redo_count})", 120)

        if ctrl and rl.is_key_pressed(K.KEY_S):
            # level.toml writeback goes through level_lib (entity doc §3c:
            # THE single writer): spawn + light/entity + water managed
            # blocks land as ONE atomic temp+rename write, sharing the
            # session's one toml .bak (pre-session bytes). water_init.npy is
            # written first (its OWN once-per-session .bak — P5 §2.4) so a
            # written [water] block never points at a missing file;
            # bake_level then rewrites the [art]/[bake] blocks with
            # write_bak=False, so nothing can clobber the .bak.
            save_tilemap_csv(csv_path, grid, write_bak=not csv_bak_written)
            csv_bak_written = True
            first = not toml_bak_written
            did_toml_bak = first     # captured before any writer flips `first`
            # Wall-over-pool guard (P5 critique M3): mask the water grid
            # against the CURRENT material grid before writing — a wall or
            # SPACE painted over a pool after the fill zeroes those cells
            # (the solver would destroy them anyway, silently). The editor
            # state follows the save so the overlay matches the file.
            # C1 (design §5): the mask is a real state mutation, so it goes
            # THROUGH the log as a committed GridCellsOp("water") — otherwise a
            # later undo/redo would desync live vs disk while the dot read
            # clean. If the mask changed nothing (the common case) commit()
            # pushes nothing. This commit truncates any redo tail; that is
            # intended (a save is a new committed action, standard linear
            # semantics — the redone-away future is discarded).
            log.begin("save-mask")
            log.snapshot_grid("water")
            masked_water, cleared = mask_water_to_open(water_q, grid,
                                                       water_solid)
            water_q[...] = masked_water
            log.commit()
            _, has_water = level_lib.write_water_npy(
                level_dir, water_q, npy_bak=not water_bak_written)
            water_bak_written = True
            # Arc C5 — zones.npy / air_init.npy: the SAME once-per-session
            # .bak carrier pattern as water_init.npy (level_lib.
            # write_zones_npy/write_air_init_npy — A8/A9's own writers, no
            # level_lib change needed here). `zones`/`air` are `None` only
            # when never painted this session AND absent at load — nothing
            # to write, nothing to delete.
            has_zones = False
            if zones is not None:
                _, has_zones = level_lib.write_zones_npy(
                    level_dir, zones, npy_bak=not zones_bak_written)
                zones_bak_written = True
            has_air = False
            if air is not None:
                _, has_air = level_lib.write_air_init_npy(
                    level_dir, air, npy_bak=not air_bak_written)
                air_bak_written = True
            # Arc C5 — PROPS: `boundary` is a bare top-level scalar, not a
            # managed family — write it through its own tiny writer ONLY
            # when it actually changed from the value the level loaded with
            # (never a no-op rewrite). It must land BEFORE `handle.save`
            # below so the ONE session .bak (`first`) captures the true
            # pre-session bytes; `handle.save` then never re-baks.
            if level_boundary != lvl.boundary:
                level_lib.write_boundary_field(
                    level_dir / "level.toml", level_boundary,
                    write_bak=first)
                toml_bak_written = True
                first = False
            # LIGHT (Arc C1, Amendment A1): write through whichever family
            # this level already used at open time — never a forced
            # migration (editor doc §6 / Erik ruling 2 / canon §2). Arc C3:
            # `entities` now also carries doors/sensors/generic placements,
            # so the "entity" family is written on EVERY save regardless of
            # which family lights use (light_and_entity_replacements' own
            # job — see its docstring for why format_lights_for_save alone
            # is no longer enough).
            replacements = {
                "spawn": lambda nl: format_spawn_lines(spawns, nl),
                "water": level_lib.water_block_format(has_water),
                # Arc C4 — wires as DATA (the wire TOOL is C6's): written on
                # EVERY save, unconditionally, the same reasoning as the
                # entity family below (C3's save-orchestration fix) — once
                # delete/paste can mutate `wires`, an omitted family would
                # mean "leave it byte-preserved" and silently strand a
                # session's wire edits.
                "wire": lambda nl: level_lib.format_wire_lines(wires, nl),
            }
            replacements.update(
                light_entity_port.light_and_entity_replacements(
                    light_form, lights, entities))
            handle.save(replacements, write_bak=first)
            toml_bak_written = True
            summary = bake_level(level_dir, tileset=tileset_arg,
                                 px_per_tile=bake_ppt, seed=bake_seed,
                                 write_bak=False)
            handle.record_disk_state()   # bake rewrote [art]/[bake] blocks
            # Pin the saved marker at the current cursor as the LAST save step
            # (after the save-mask commit above): state == disk, so the dot
            # clears and undoing back here will clear it again (§5). Replaces
            # the four dirty_* bools that could never clear on undo.
            log.mark_saved()
            wet = int(np.count_nonzero(water_q))
            n_zoned = int(np.count_nonzero(zones)) if zones is not None else 0
            n_aired = (int(np.count_nonzero(air != air_paint.FP_ONE))
                      if air is not None else 0)
            flash, flash_frames = (
                f"SAVED tilemap.csv + {len(spawns)} spawns + "
                f"{len(lights)} lights + {len(entities)} entities + "
                f"{len(wires)} wires + "
                f"{f'{wet} water tiles' if has_water else 'no water'} + "
                f"{f'{n_zoned} zoned tiles' if has_zones else 'no zones'} + "
                f"{f'{n_aired} authored air tiles' if has_air else 'no air override'}"
                f"{f' ({cleared} cleared under walls/space)' if cleared else ''}"
                f" + boundary={level_boundary}"
                f" + bake blocks + "
                f"full bake @ {summary['px_per_tile']} px/tile"
                f"{' (.bak written)' if did_toml_bak else ''}", 240)

        # ---- play (F5) ------------------------------------------------------
        # editor doc §8 / C7: save the FULL live state (unsaved edits
        # included — never just what Ctrl+S last wrote) to
        # levels/_editor_scratch/<name>/ and launch the real game against it.
        # Reuses the SAME level_lib/light_entity_port/bake_level_art
        # machinery Ctrl+S calls (tools/play_scratch.py) — never a parallel
        # serializer. Water is masked the same wall-over-pool way Ctrl+S
        # masks it, but as a computed COPY (mask_water_to_open never mutates
        # its input), so F5 never touches the live undo timeline. "Clean"
        # (bake reuse) is `not log.dirty` — the C2 saved-marker: coarser than
        # "only the material grid changed", but a false "dirty" only costs an
        # extra re-bake, never a stale one (§ correctness first).
        if not typing and rl.is_key_pressed(K.KEY_F5):
            try:
                # The scratch subfolder is named after the level_dir's OWN
                # folder name (guaranteed filesystem-safe — it already IS a
                # directory name on disk) rather than the authored `lvl.name`
                # display string, which is free-form TOML text and may carry
                # characters no filesystem accepts.
                play_name = level_dir.name
                water_masked, _ = mask_water_to_open(water_q, grid,
                                                     water_solid)
                play_dir = play_scratch.write_scratch_level(
                    play_name, level_dir=level_dir, grid=grid,
                    water_masked=water_masked, zones=zones, air=air,
                    level_boundary=level_boundary, spawns=spawns,
                    lights=lights, entities=entities, wires=wires,
                    light_form=light_form, bake_clean=not log.dirty,
                    tileset_arg=tileset_arg, bake_ppt=bake_ppt,
                    bake_seed=bake_seed)
                argv = play_scratch.build_launch_argv(play_name)
                proc = subprocess.Popen(argv)
                active_plays.append((proc, play_dir))
                flash, flash_frames = (
                    f"PLAY: launched {play_name} (pid {proc.pid}, "
                    f"{'reused bake' if not log.dirty else 'fresh bake'})",
                    180)
            except Exception as exc:      # noqa: BLE001 — never crash the
                flash, flash_frames = f"PLAY failed: {exc}", 300  # editor

        # Reap finished PLAY subprocesses every frame (non-blocking poll) and
        # clean up their scratch dirs the instant they exit — the OTHER half
        # of the cleanup lives at quit, below the main loop, for whatever is
        # still running when the editor closes.
        if active_plays:
            still_running = []
            for proc, play_dir in active_plays:
                if proc.poll() is None:
                    still_running.append((proc, play_dir))
                else:
                    play_scratch.cleanup_scratch_dir(play_dir)
            active_plays[:] = still_running

        # ---- draw ---------------------------------------------------------
        def to_screen(wx: float, wy: float) -> tuple:
            return screen_from_world(canvas, cam_x, cam_y, zoom, wx, wy)

        rl.begin_drawing()
        rl.clear_background(rl.Color(*COL_BG))

        # The map draws ONLY inside the canvas pane — scissor clips it off the
        # surrounding chrome (top bar / rail / palette / inspector / status).
        rl.begin_scissor_mode(canvas.x, canvas.y,
                              max(0, canvas.w), max(0, canvas.h))

        tw = th = preview_ppt * zoom
        vx0, vy0 = art_px_to_tile(cam_x, cam_y, offset0, ppt_pair)
        vx1, vy1 = art_px_to_tile(cam_x + canvas.w / zoom,
                                  cam_y + canvas.h / zoom, offset0, ppt_pair)
        tx0, ty0 = max(0, int(np.floor(vx0))), max(0, int(np.floor(vy0)))
        tx1 = min(grid_w, int(np.ceil(vx1)) + 1)
        ty1 = min(grid_h, int(np.ceil(vy1)) + 1)

        if view_baked:
            tex = tex_normal if show_normal else tex_diffuse
            dx, dy = to_screen(0.0, 0.0)
            rl.draw_texture_pro(
                tex, rl.Rectangle(0, 0, world_w, world_h),
                rl.Rectangle(dx, dy, world_w * zoom, world_h * zoom),
                rl.Vector2(0, 0), 0.0, rl.WHITE)
        else:
            sub = grid[ty0:ty1, tx0:tx1]
            ys, xs = np.nonzero(sub != MAT_AIR)
            for tx_, ty_, v in zip((xs + tx0).tolist(), (ys + ty0).tolist(),
                                   sub[ys, xs].tolist()):
                c = mat_fill.get(v)
                if c is None:
                    continue
                sx, sy = to_screen(tx_ * preview_ppt, ty_ * preview_ppt)
                if sx + tw < 0 or sy + th < 0 or sx > win_w or sy > win_h:
                    continue
                rl.draw_rectangle(int(sx), int(sy),
                                  max(1, int(tw)), max(1, int(th)),
                                  rl.Color(c[0], c[1], c[2], 255))

        if show_grid:
            for gx in range(tx0, tx1 + 1):
                a0 = to_screen(gx * preview_ppt, ty0 * preview_ppt)
                a1 = to_screen(gx * preview_ppt, ty1 * preview_ppt)
                rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 1.0,
                                COL_GRID)
            for gy in range(ty0, ty1 + 1):
                a0 = to_screen(tx0 * preview_ppt, gy * preview_ppt)
                a1 = to_screen(tx1 * preview_ppt, gy * preview_ppt)
                rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 1.0,
                                COL_GRID)

        # Mode overlays.
        if mode == "PAINT" and not over_hud:
            br = brush_rect(cur_tx, cur_ty, brush, grid_w, grid_h)
            if br is not None:
                s0 = to_screen(br[0] * preview_ppt, br[1] * preview_ppt)
                s1 = to_screen((br[2] + 1) * preview_ppt,
                               (br[3] + 1) * preview_ppt)
                rl.draw_rectangle_lines_ex(
                    rl.Rectangle(s0[0], s0[1], s1[0] - s0[0], s1[1] - s0[1]),
                    2.0, COL_CURSOR)
        elif mode == "ROOM" and room_start is not None:
            rect = normalize_rect(room_start[0], room_start[1],
                                  cur_tx, cur_ty, grid_w, grid_h)
            if rect is not None:
                c = mat_fill.get(selected_id, (255, 255, 255))
                s0 = to_screen(rect[0] * preview_ppt, rect[1] * preview_ppt)
                s1 = to_screen((rect[2] + 1) * preview_ppt,
                               (rect[3] + 1) * preview_ppt)
                rl.draw_rectangle(int(s0[0]), int(s0[1]),
                                  int(s1[0] - s0[0]), int(s1[1] - s0[1]),
                                  rl.Color(c[0], c[1], c[2], 60))
                rl.draw_rectangle_lines_ex(
                    rl.Rectangle(s0[0], s0[1], s1[0] - s0[0], s1[1] - s0[1]),
                    2.0, rl.Color(c[0], c[1], c[2], 230))
        elif mode == "CORRIDOR" and corr_start is not None:
            c = mat_fill.get(selected_id, (255, 255, 255))
            a0 = to_screen((corr_start[0] + 0.5) * preview_ppt,
                           (corr_start[1] + 0.5) * preview_ppt)
            a1 = to_screen((clamp_tx + 0.5) * preview_ppt,
                           (clamp_ty + 0.5) * preview_ppt)
            rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1),
                            max(1.0, corridor_w * preview_ppt * zoom),
                            rl.Color(255, 255, 255, 60))
            rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 2.0,
                            rl.Color(c[0], c[1], c[2], 230))
        elif mode == "DOOR" and not over_hud and door_tool_error is None:
            if door_drag is not None:
                orientation = door_drag["orientation"]
                far = clamp_tx if orientation == "h" else clamp_ty
                span, _length_m = door_entity_port.plan_door_span(
                    grid, door_drag["anchor"], orientation, far,
                    tile_size_m, wall_codes)
                fxs = [fx for (_fy, fx) in span]
                fys = [fy for (fy, _fx) in span]
                s0 = to_screen(min(fxs) * preview_ppt, min(fys) * preview_ppt)
                s1 = to_screen((max(fxs) + 1) * preview_ppt,
                              (max(fys) + 1) * preview_ppt)
                col = (COL_OK if door_initial_state == "closed"
                       else (150, 190, 255, 255))
                rl.draw_rectangle(int(s0[0]), int(s0[1]),
                                  int(s1[0] - s0[0]), int(s1[1] - s0[1]),
                                  rl.Color(col[0], col[1], col[2], 70))
                rl.draw_rectangle_lines_ex(
                    rl.Rectangle(s0[0], s0[1], s1[0] - s0[0], s1[1] - s0[1]),
                    2.0, rl.Color(*col))
            elif cursor_in:
                ok, _orientation, _why = door_entity_port.door_anchor_check(
                    grid, cur_tx, cur_ty, wall_codes)
                col = COL_OK if ok else COL_BAD
                s0 = to_screen(cur_tx * preview_ppt, cur_ty * preview_ppt)
                rl.draw_rectangle_lines_ex(
                    rl.Rectangle(s0[0], s0[1], tw, th), 2.0, rl.Color(*col))
        elif mode == "WATER" and not over_hud and cursor_in:
            # Cheap start-tile validity only (green/red, the DOOR pattern) —
            # never a per-frame BFS; the real region resolves on click.
            v = int(grid[cur_ty, cur_tx])
            ok = (v != SPACE_CODE) and (v not in water_solid)
            col = COL_OK if ok else COL_BAD
            s0 = to_screen(cur_tx * preview_ppt, cur_ty * preview_ppt)
            rl.draw_rectangle_lines_ex(
                rl.Rectangle(s0[0], s0[1], tw, th), 2.0, rl.Color(*col))
        elif mode == "ENTITY" and not over_hud:
            cls_name = selected_entity_class
            if cls_name and sensor_entity_port.is_field_sensor(cls_name):
                family = sensor_entity_port.sample_family(cls_name)
                if entity_drag is not None and entity_drag["class"] == cls_name:
                    ax, ay = entity_drag["anchor"]
                    dx = (clamp_tx - ax) if family == sensor_entity_port.SAMPLE_AIR else 0
                    dy = (clamp_ty - ay) if family == sensor_entity_port.SAMPLE_AIR else 0
                elif cursor_in:
                    ax, ay, dx, dy = cur_tx, cur_ty, 0, 0
                else:
                    ax = ay = None
                if ax is not None:
                    mount_s = to_screen((ax + 0.5) * preview_ppt,
                                        (ay + 0.5) * preview_ppt)
                    rl.draw_circle_lines(
                        int(mount_s[0]), int(mount_s[1]),
                        max(6.0, 0.3 * preview_ppt * zoom), rl.Color(*COL_OK))
                    if dx or dy:
                        tip_s = to_screen((ax + dx + 0.5) * preview_ppt,
                                          (ay + dy + 0.5) * preview_ppt)
                        rl.draw_line_ex(rl.Vector2(*mount_s),
                                       rl.Vector2(*tip_s), 2.0,
                                       rl.Color(*COL_OK))
                        rl.draw_circle_v(rl.Vector2(*tip_s), 4.0,
                                         rl.Color(*COL_OK))

        elif mode in ("WAND", "ZONE", "AIR") and not over_hud and cursor_in:
            # Cheap start-tile validity only (green/red, the DOOR/WATER
            # pattern) — never a per-frame BFS; the real region resolves on
            # click. WAND/ZONE both refuse a start tile that is solid-for-
            # water (the shared enclosure boundary); AIR additionally
            # refuses vacuum/SPACE (a fill cannot start already in vacuum).
            v = int(grid[cur_ty, cur_tx])
            if mode == "AIR":
                ok = v not in water_solid
            else:
                ok = (v != SPACE_CODE) and (v not in water_solid)
            col = COL_OK if ok else COL_BAD
            s0 = to_screen(cur_tx * preview_ppt, cur_ty * preview_ppt)
            rl.draw_rectangle_lines_ex(
                rl.Rectangle(s0[0], s0[1], tw, th), 2.0, rl.Color(*col))

        elif mode == "WIRE" and not over_hud:
            # Live preview: source anchor -> cursor (pan/zoom stay live —
            # editor doc §8's own "navigate freely" spec). Hover highlight on
            # whatever the cursor is over (an entity OR a positionless
            # node's layout slot).
            if wire_pending is not None:
                anchor = wire_tool.anchor_tile(
                    wire_pending["from_id"], lights, entities, tile_size_m,
                    node_layout)
                if anchor is not None:
                    a0 = to_screen((anchor[0] + 0.5) * preview_ppt,
                                  (anchor[1] + 0.5) * preview_ppt)
                    a1 = to_screen((cur_tx + 0.5) * preview_ppt,
                                  (cur_ty + 0.5) * preview_ppt)
                    rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 2.0,
                                    rl.Color(*COL_OK))
            hover = wire_tool.hit_test_with_layout(
                cur_tx, cur_ty, lights, entities, tile_size_m, node_layout)
            if hover is not None:
                s0 = to_screen(cur_tx * preview_ppt, cur_ty * preview_ppt)
                rl.draw_rectangle_lines_ex(
                    rl.Rectangle(s0[0], s0[1], tw, th), 2.0,
                    rl.Color(*COL_OK))

        elif mode == "SELECT" and not over_hud:
            def _instance_box_screen(coll_name, obj, off_tx=0, off_ty=0):
                tiles = entity_selection.instance_tiles(
                    coll_name, obj, tile_size_m)
                if not tiles:
                    return None
                xs = [t[0] for t in tiles]
                ys = [t[1] for t in tiles]
                x0, y0 = min(xs) + off_tx, min(ys) + off_ty
                x1, y1 = max(xs) + off_tx, max(ys) + off_ty
                s0 = to_screen(x0 * preview_ppt, y0 * preview_ppt)
                s1 = to_screen((x1 + 1) * preview_ppt, (y1 + 1) * preview_ppt)
                return rl.Rectangle(s0[0], s0[1], s1[0] - s0[0], s1[1] - s0[1])

            move_delta = None
            if select_move is not None and cursor_in:
                mx, my = select_move["anchor"]
                move_delta = (cur_tx - mx, cur_ty - my)

            # Selected instances: highlight outline (offset live by the
            # in-progress move-drag delta, so the preview shows where the
            # group WILL land — the actual mutation only happens at release).
            for sel_id in selection:
                coll_name, idx = entity_selection.find_instance(
                    sel_id, lights, entities)
                if coll_name is None:
                    continue
                obj = lights[idx] if coll_name == "lights" else entities[idx]
                dtx, dty = move_delta if move_delta is not None else (0, 0)
                rect = _instance_box_screen(coll_name, obj, dtx, dty)
                if rect is not None:
                    rl.draw_rectangle_lines_ex(rect, 2.0,
                                               rl.Color(*COL_SELECT))

            # Hover highlight (dim, unselected instance under the cursor).
            if select_box is None and select_move is None and cursor_in:
                hov = entity_selection.hit_test(cur_tx, cur_ty, lights,
                                                entities, tile_size_m)
                if hov is not None and hov not in selection:
                    coll_name, idx = entity_selection.find_instance(
                        hov, lights, entities)
                    if coll_name is not None:
                        obj = (lights[idx] if coll_name == "lights"
                              else entities[idx])
                        rect = _instance_box_screen(coll_name, obj)
                        if rect is not None:
                            rl.draw_rectangle_lines_ex(
                                rect, 1.5, rl.Color(255, 255, 255, 160))

            # Box-select drag preview.
            if select_box is not None:
                bx, by = select_box["anchor"]
                bx0, bx1 = sorted((bx, clamp_tx))
                by0, by1 = sorted((by, clamp_ty))
                s0 = to_screen(bx0 * preview_ppt, by0 * preview_ppt)
                s1 = to_screen((bx1 + 1) * preview_ppt, (by1 + 1) * preview_ppt)
                rl.draw_rectangle(
                    int(s0[0]), int(s0[1]),
                    int(s1[0] - s0[0]), int(s1[1] - s0[1]),
                    rl.Color(COL_SELECT[0], COL_SELECT[1], COL_SELECT[2], 40))
                rl.draw_rectangle_lines_ex(
                    rl.Rectangle(s0[0], s0[1], s1[0] - s0[0], s1[1] - s0[1]),
                    1.5, rl.Color(*COL_SELECT))

        # Water overlay (every mode — level content): translucent blue per
        # wet tile, alpha scaled by depth (editor view only; in-game the
        # WaterPass renders the field).
        sub_w = water_q[ty0:ty1, tx0:tx1]
        wys, wxs = np.nonzero(sub_w)
        wr, wg, wb = WATER_OVERLAY_RGB
        for tx_, ty_, dq in zip((wxs + tx0).tolist(), (wys + ty0).tolist(),
                                sub_w[wys, wxs].tolist()):
            sx, sy = to_screen(tx_ * preview_ppt, ty_ * preview_ppt)
            if sx + tw < 0 or sy + th < 0 or sx > win_w or sy > win_h:
                continue
            depth_m = dq / water_fixed.FP_ONE_F
            alpha = int(min(190.0, 60.0 + depth_m * 65.0))
            rl.draw_rectangle(int(sx), int(sy),
                              max(1, int(tw)), max(1, int(th)),
                              rl.Color(wr, wg, wb, alpha))

        # Zone overlay (every mode — level content, Arc C5): a deterministic
        # colour per painted id (the same per-id colour ZONE mode's own
        # eyedrop/target reads reuse implicitly, since it's a pure function
        # of the id).
        if zones is not None:
            sub_z = zones[ty0:ty1, tx0:tx1]
            zys, zxs = np.nonzero(sub_z)
            for tx_, ty_, zid in zip((zxs + tx0).tolist(),
                                     (zys + ty0).tolist(),
                                     sub_z[zys, zxs].tolist()):
                sx, sy = to_screen(tx_ * preview_ppt, ty_ * preview_ppt)
                if sx + tw < 0 or sy + th < 0 or sx > win_w or sy > win_h:
                    continue
                zr, zg, zb = _zone_color(int(zid))
                rl.draw_rectangle(int(sx), int(sy),
                                  max(1, int(tw)), max(1, int(th)),
                                  rl.Color(zr, zg, zb, ZONE_OVERLAY_ALPHA))

        # Air overlay (every mode — level content, Arc C5): only the tiles
        # that diverge from the FP_ONE (1.0 atm) default tint — a freshly
        # allocated air_init.npy seeds FP_ONE everywhere (air_paint.
        # default_ambient_grid), so this overlay traces exactly what the
        # user actually painted regardless of mode.
        if air is not None:
            sub_a = air[ty0:ty1, tx0:tx1]
            ays, axs = np.nonzero(sub_a != air_paint.FP_ONE)
            ar, ag, ab = AIR_OVERLAY_RGB
            for tx_, ty_, pq in zip((axs + tx0).tolist(), (ays + ty0).tolist(),
                                    sub_a[ays, axs].tolist()):
                sx, sy = to_screen(tx_ * preview_ppt, ty_ * preview_ppt)
                if sx + tw < 0 or sy + th < 0 or sx > win_w or sy > win_h:
                    continue
                atm = pq / float(air_paint.FP_ONE)
                alpha = int(min(190.0, 40.0 + abs(atm - 1.0) * 90.0))
                rl.draw_rectangle(int(sx), int(sy),
                                  max(1, int(tw)), max(1, int(th)),
                                  rl.Color(ar, ag, ab, alpha))

        # LOGIC overlay (every mode — level content, Arc C6, editor doc §8):
        # defaults to wires touching the current SELECT-mode selection; L
        # (global) toggles show-all. A tag-target wire draws as ONE compact
        # badge near its source (never fanned to every current member — the
        # whole point of the badge is staying readable when a tag has many
        # members); every other wire draws as a plain connector between its
        # resolved endpoints (positionless entities included, via
        # node_layout). Drawn directly over the (small, editor-scale) wire/
        # node lists — no tile-range culling, matching the spawn/light/
        # sensor markers' own pattern below.
        visible_wires = (wires if show_all_wires else
                         wire_tool.wires_touching_selection(wires, selection))
        wire_lines, wire_badges = wire_tool.wire_endpoints_for_draw(
            visible_wires, lights, entities, tile_size_m, node_layout)
        for w, (lx0, ly0), (lx1, ly1) in wire_lines:
            a0 = to_screen((lx0 + 0.5) * preview_ppt, (ly0 + 0.5) * preview_ppt)
            a1 = to_screen((lx1 + 0.5) * preview_ppt, (ly1 + 0.5) * preview_ppt)
            rgba = COL_SELECT if w is wire_selected else LOGIC_WIRE_RGBA
            rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 2.0,
                            rl.Color(*rgba))
            rl.draw_circle_v(rl.Vector2(*a1), 3.0, rl.Color(*rgba))
        for w, (bx, by), tag_name, input_name in wire_badges:
            label = f"#{tag_name}.{input_name}"
            s0 = to_screen((bx + 0.6) * preview_ppt, (by - 0.35) * preview_ppt)
            rgba = COL_SELECT if w is wire_selected else LOGIC_TAG_RGBA
            bw_ = rl.measure_text(label, 12) + 6
            rl.draw_rectangle(int(s0[0]) - 3, int(s0[1]) - 2, bw_, 16,
                              rl.Color(0, 0, 0, 170))
            rl.draw_text(label, int(s0[0]), int(s0[1]), 12, rl.Color(*rgba))

        # Positionless logic-node markers (every mode — level content, Arc
        # C6): a colour chip (the SAME deterministic per-class colour the
        # ENTITY palette shows) + class initial at each node's
        # `node_layout` slot — the wire tool's own "give them a clickable
        # position" answer, made visible.
        for e in entities:
            if not wire_tool.is_positionless(e.fields):
                continue
            pos = node_layout.get(e.id)
            if pos is None:
                continue
            nx, ny = pos
            sx, sy = to_screen((nx + 0.5) * preview_ppt, (ny + 0.5) * preview_ppt)
            chip = entity_chip_rgb.get(e.class_name, (200, 200, 200))
            r = max(6.0, LOGIC_NODE_RADIUS_TILES * preview_ppt * zoom)
            rl.draw_circle_v(rl.Vector2(sx, sy), r,
                             rl.Color(chip[0], chip[1], chip[2], 225))
            rl.draw_text(e.class_name[0].upper(), int(sx - 4), int(sy - 7),
                        14, rl.BLACK)
            if e.id in selection:
                rl.draw_circle_lines(int(sx), int(sy), r + 3.0,
                                     rl.Color(*COL_SELECT))

        # Spawn markers (every mode — they are level content).
        hover_idx = spawn_at(spawns, ftx, fty) if mode == "SPAWN" else None
        for i, s in enumerate(spawns):
            fp = float(s.footprint)
            sx, sy = to_screen((s.x + fp / 2.0) * preview_ppt,
                               (s.y + fp / 2.0) * preview_ppt)
            r = fp / 2.0 * preview_ppt * zoom
            c = TEAM_COLORS.get(int(s.team), (200, 200, 200))
            rl.draw_circle_v(rl.Vector2(sx, sy), r,
                             rl.Color(c[0], c[1], c[2], 130))
            rl.draw_circle_lines(int(sx), int(sy), r,
                                 rl.Color(c[0], c[1], c[2], 255))
            if i == hover_idx:
                rl.draw_circle_lines(int(sx), int(sy), r + 3.0, rl.WHITE)
            if mode == "SPAWN" and r > 8:
                rl.draw_text(str(s.name),
                             int(sx - rl.measure_text(str(s.name), 14) / 2),
                             int(sy - r - 16), 14, rl.WHITE)

        # Light markers (every mode — level content): colour dot + range
        # ring; beacons sweep a beam wedge animated on the EDITOR's clock
        # (the editor is not the sim — in-game the angle is a pure function
        # of the sim tick; same beacon_angle math either way).
        light_hover = light_at(lights, ftx, fty) if mode == "LIGHT" else None
        for i, l in enumerate(lights):
            sx, sy = to_screen(l.x * preview_ppt, l.y * preview_ppt)
            c = color_255(l.color)
            ring = float(l.range) * preview_ppt * zoom
            dot = max(4.0, 0.45 * preview_ppt * zoom)
            if l.kind == "beacon":
                ang = math.degrees(
                    beacon_angle(frames, EDITOR_TICK_DT,
                                 l.period_s, l.phase) % math.tau)
                half = float(l.beam_deg) / 2.0
                rl.draw_circle_sector(
                    rl.Vector2(sx, sy), ring, ang - half, ang + half, 24,
                    rl.Color(c[0], c[1], c[2], 60))
            rl.draw_circle_v(rl.Vector2(sx, sy), dot,
                             rl.Color(c[0], c[1], c[2], 220))
            rl.draw_circle_lines(int(sx), int(sy), ring,
                                 rl.Color(c[0], c[1], c[2], 90))
            if i == light_hover:
                rl.draw_circle_lines(int(sx), int(sy), dot + 3.0, rl.WHITE)

        # Sensor markers (every mode — level content, Arc C3): a small dot
        # on the mount tile + the AIR family's facing arrow (BODY family —
        # temperature/fire — samples its own mount, no arrow: D7/D8).
        for e in entities:
            if not sensor_entity_port.is_field_sensor(e.class_name):
                continue
            ex, ey = e.fields.get("x"), e.fields.get("y")
            if ex is None or ey is None:
                continue
            sx, sy = to_screen((ex + 0.5) * preview_ppt, (ey + 0.5) * preview_ppt)
            rl.draw_circle_v(rl.Vector2(sx, sy),
                             max(3.0, 0.18 * preview_ppt * zoom),
                             rl.Color(255, 210, 90, 230))
            dx, dy = e.fields.get("sample_dx", 0), e.fields.get("sample_dy", 0)
            if (sensor_entity_port.sample_family(e.class_name)
                    == sensor_entity_port.SAMPLE_AIR and (dx or dy)):
                tx_, ty_ = to_screen((ex + dx + 0.5) * preview_ppt,
                                     (ey + dy + 0.5) * preview_ppt)
                rl.draw_line_ex(rl.Vector2(sx, sy), rl.Vector2(tx_, ty_), 2.0,
                                rl.Color(255, 210, 90, 230))

        # ---- panes (chrome) -----------------------------------------------
        # The map is done; stop clipping and paint the surrounding panes on
        # top. Every non-canvas region is a framed pane (§8): top bar, tool
        # rail, palette, inspector, status bar. C1 fills palette + inspector
        # from the entity registry; here they host the old HUD content.
        rl.end_scissor_mode()

        top = panes["top_bar"]
        rail = panes["tool_rail"]
        pal = panes["palette"]
        insp = panes["inspector"]
        status = panes["status_bar"]

        def fill_pane(rc) -> None:
            rl.draw_rectangle(rc.x, rc.y, rc.w, rc.h,
                              rl.Color(*COL_PANEL_BG))
            rl.draw_rectangle_lines(rc.x, rc.y, rc.w, rc.h,
                                    rl.Color(*COL_PANEL_EDGE))

        def wrap_text(text: str, max_w: int, font: int) -> list:
            words, lines, cur = text.split(), [], ""
            for wd in words:
                trial = wd if not cur else f"{cur} {wd}"
                if not cur or rl.measure_text(trial, font) <= max_w:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = wd
            if cur:
                lines.append(cur)
            return lines

        for rc in (top, rail, pal, insp, status):
            fill_pane(rc)

        # Per-mode extra line — verbatim content from the old HUD.
        extra = ""
        if mode == "PAINT":
            extra = f"brush {brush}x{brush} [ ]"
        elif mode == "CORRIDOR":
            extra = f"width {corridor_w} (+/-)"
        elif mode == "SPAWN":
            extra = (f"placing {TEAM_NAMES[spawn_team]} (T)  "
                     f"{len(spawns)} spawns")
        elif mode == "LIGHT":
            extra = (f"placing {light_kind} / "
                     f"{light_color_name(light_color)} (B/C)  "
                     f"{len(lights)} lights")
        elif mode == "WATER":
            extra = (f"depth {water_depth_m:.1f} m (-/=)  "
                     f"{int(np.count_nonzero(water_q))} wet tiles")
            if water_depth_m > WATER_DEEP_HINT_M:
                extra += "  deep tank: drains may flash-boil (by design)"
        elif mode == "WAND":
            extra = f"target: {palette[selected_id][0]} (material palette)"
        elif mode == "ZONE":
            target = "new" if zone_paint_id is None else str(zone_paint_id)
            n_painted = int(np.count_nonzero(zones)) if zones is not None else 0
            extra = (f"target: {target} ({zone_paint_class}, C cycles)  "
                     f"{n_painted} painted tiles")
        elif mode == "AIR":
            n_authored = (int(np.count_nonzero(air != air_paint.FP_ONE))
                         if air is not None else 0)
            extra = (f"pressure {air_pressure_atm:.2f} atm (-/=)  "
                     f"{n_authored} authored tiles")
        elif mode == "WIRE":
            if wire_pending is None:
                pend_txt = (f"selected: {wire_selected.from_} -> "
                           f"{wire_selected.to} (Delete removes)"
                           if wire_selected is not None
                           else "click a source")
            else:
                sig = wire_pending["signals"][wire_pending["signal_idx"]]
                if wire_pending["to_spec"] is None:
                    pend_txt = (f"{wire_pending['from_id']}.{sig} -> click "
                               f"a target")
                else:
                    inp = wire_pending["inputs"][wire_pending["input_idx"]]
                    pend_txt = (f"{wire_pending['from_id']}.{sig} -> "
                               f"{wire_pending['to_spec']}.{inp}"
                               + ("" if len(wire_pending["inputs"]) <= 1
                                  else "  (Enter confirms)"))
            extra = (f"{pend_txt}   {len(wires)} wires   "
                     f"{'ALL' if show_all_wires else 'selection'} (L toggles)")

        # Top bar: level / grid / zoom / view mode.
        preview_note = ("" if preview_ppt == bake_ppt
                        else f"  preview @{preview_ppt}px (bake {bake_ppt})")
        view_name = ("normals" if (view_baked and show_normal)
                     else "baked" if view_baked else "materials")
        rl.draw_text(
            f"{level_dir.name}   grid {grid_w}x{grid_h}   zoom {zoom:.2f}"
            f"   view: {view_name} (V/N){preview_note}",
            top.x + 10, top.y + 12, 18, rl.Color(*COL_TEXT))

        # Tool rail: the mode list (click a row, or TAB / F1-F12 — PROPS has
        # no F-key left, TAB/rail-click only).
        for i, mname in enumerate(MODES):
            ry = rail.y + RAIL_PAD_Y + i * RAIL_ROW_H
            if ry + RAIL_ROW_H > rail.y + rail.h:
                break
            if i == mode_idx:
                rl.draw_rectangle(rail.x + 4, ry, max(0, rail.w - 8),
                                  RAIL_ROW_H - 4, rl.Color(*COL_RAIL_SEL))
            col = COL_TEXT_HOT if i == mode_idx else COL_TEXT_DIM
            fkey = ("   " if i == PLAY_FKEY_INDEX or i >= MAX_MODE_FKEYS
                    else f"F{i + 1} ")
            rl.draw_text(f"{fkey}{mname}", rail.x + 12, ry + 6, 16,
                         rl.Color(*col))

        # Palette pane: tab headers (MATERIAL / ENTITY, click to switch —
        # Arc C1) + the active tab's chips, selected boxed.
        tab_x = pal.x + 10
        for label in PAL_TABS:
            w = rl.measure_text(label, PAL_TAB_FONT)
            active = label == palette_tab
            rl.draw_text(label, tab_x, pal.y + 8, PAL_TAB_FONT,
                         rl.Color(*COL_TEXT_HOT) if active
                         else rl.Color(*COL_TEXT_DIM))
            if active:
                rl.draw_line_ex(rl.Vector2(tab_x, pal.y + 24),
                                rl.Vector2(tab_x + w, pal.y + 24), 2.0,
                                rl.Color(*COL_TEXT_HOT))
            tab_x += w + PAL_TAB_GAP
        if palette_tab == "MATERIAL":
            for i, pid in enumerate(palette_order):
                py = pal.y + PAL_PAD_Y + i * PAL_ROW_H
                if py + PAL_ROW_H > pal.y + pal.h:
                    break
                pname, c = palette[pid]
                chip = (rl.Color(c[0], c[1], c[2], 255) if c is not None
                        else rl.Color(70, 70, 76, 255))
                rl.draw_rectangle(pal.x + 10, py, 14, 14, chip)
                if pid == selected_id:
                    rl.draw_rectangle_lines_ex(
                        rl.Rectangle(pal.x + 8, py - 2, 18, 18), 2.0, rl.WHITE)
                label = f"{9 if pid == SPACE_CODE else pid} {pname}"
                rl.draw_text(label, pal.x + 32, py, 14,
                             rl.WHITE if pid == selected_id
                             else rl.Color(*COL_TEXT_DIM))
        else:
            for i, pent in enumerate(entity_palette):
                py = pal.y + PAL_PAD_Y + i * PAL_ROW_H
                if py + PAL_ROW_H > pal.y + pal.h:
                    break
                selected = pent.class_name == selected_entity_class
                # Arc C8: a committed PNG icon (loaded once into
                # entity_icon_textures at launch) is preferred; the
                # generated colour chip + class initial is the PERMANENT
                # fallback for every class without one (editor doc §8).
                icon_tex = entity_icon_textures.get(pent.class_name)
                if icon_tex is not None:
                    rl.draw_texture_ex(
                        icon_tex, rl.Vector2(pal.x + 10, py),
                        0.0, 14.0 / icon_tex.width, rl.WHITE)
                else:
                    c = pent.chip_rgb
                    rl.draw_rectangle(pal.x + 10, py, 14, 14,
                                      rl.Color(c[0], c[1], c[2], 255))
                    rl.draw_text(pent.initial, pal.x + 13, py - 1, 12,
                                 rl.BLACK)
                if selected:
                    rl.draw_rectangle_lines_ex(
                        rl.Rectangle(pal.x + 8, py - 2, 18, 18), 2.0, rl.WHITE)
                rl.draw_text(pent.class_name, pal.x + 32, py, 14,
                             rl.WHITE if selected else rl.Color(*COL_TEXT_DIM))

        # Inspector pane (Arc C1 — registry-driven): a LIGHT instance under
        # the cursor (LIGHT mode) beats an ENTITY-tab class selection, which
        # beats the C0 mode/material/hint content — the first thing with
        # something concrete to show wins.
        iy = insp.y + 10

        def _draw_rows(rows, start_y: int) -> int:
            y = start_y
            for row in rows:
                if y + 16 > insp.y + insp.h:
                    break
                for ln in wrap_text(f"{row.name}: {row.display}",
                                    max(20, insp.w - 20), 13):
                    rl.draw_text(ln, insp.x + 10, y, 13, rl.Color(*COL_TEXT))
                    y += 16
            return y

        def _resolve_selected(sel_id):
            """(coll_name, obj) for `sel_id`, or None if it's gone stale."""
            coll_name, idx = entity_selection.find_instance(
                sel_id, lights, entities)
            if coll_name is None:
                return None
            return coll_name, (lights[idx] if coll_name == "lights"
                               else entities[idx])

        light_cls_payload = registry_payload.get("classes", {}).get("light")
        if mode == "LIGHT" and light_hover is not None and light_cls_payload:
            l = lights[light_hover]
            rl.draw_text(f"[light] {l.id}", insp.x + 10, iy, 18,
                         rl.Color(*COL_TEXT_HOT))
            iy += 26
            values = light_entity_port.light_entry_to_fields(l)
            rows = entity_editor_ui.inspector_rows(light_cls_payload, values)
            iy = _draw_rows(rows, iy)
        elif mode == "SELECT" and selection:
            # Arc C4 — the unified selection's live inspector: a single
            # selection shows id/tags/class fields (with the FOCUSED editable
            # field marked "> " — [ ] moves it, the value keys nudge/cycle
            # it); a multi-selection shows a class breakdown (tag-assign is
            # the only op that applies uniformly across mixed classes).
            if text_entry is not None:
                rl.draw_text(f"{text_entry['kind']}: {text_entry['buf']}_",
                             insp.x + 10, iy, 16, rl.Color(*COL_TEXT_HOT))
                iy += 22
            if len(selection) == 1:
                sel_id = next(iter(selection))
                resolved = _resolve_selected(sel_id)
                if resolved is None:
                    rl.draw_text("(selection is stale)", insp.x + 10, iy, 16,
                                 rl.Color(*COL_TEXT_DIM))
                    iy += 20
                else:
                    coll_name, obj = resolved
                    cls_name = entity_selection.instance_class_name(
                        coll_name, obj)
                    rl.draw_text(f"[{cls_name}] {sel_id}", insp.x + 10, iy,
                                 18, rl.Color(*COL_TEXT_HOT))
                    iy += 24
                    tags_disp = entity_editor_ui.format_field_value(
                        entity_editor_ui.KIND_STR_LIST,
                        entity_selection.instance_tags(obj))
                    rl.draw_text(f"tags: {tags_disp}", insp.x + 10, iy, 13,
                                 rl.Color(*COL_TEXT))
                    iy += 18
                    cls_payload = registry_payload.get(
                        "classes", {}).get(cls_name)
                    if cls_payload is not None:
                        values = (light_entity_port.light_entry_to_fields(obj)
                                 if coll_name == "lights" else obj.fields)
                        rows = entity_editor_ui.inspector_rows(
                            cls_payload, values)
                        editable_rows = [r for r in rows if r.editable]
                        focus_name = (
                            editable_rows[select_focus_idx
                                         % len(editable_rows)].name
                            if editable_rows else None)
                        for row in rows:
                            if iy + 16 > insp.y + insp.h:
                                break
                            hot = row.name == focus_name
                            marker = "> " if hot else "  "
                            col = COL_TEXT_HOT if hot else COL_TEXT
                            for ln in wrap_text(
                                    f"{marker}{row.name}: {row.display}",
                                    max(20, insp.w - 20), 13):
                                rl.draw_text(ln, insp.x + 10, iy, 13,
                                            rl.Color(*col))
                                iy += 16
            else:
                rl.draw_text(f"[SELECT] {len(selection)} instances",
                             insp.x + 10, iy, 18, rl.Color(*COL_TEXT_HOT))
                iy += 24
                counts: dict = {}
                for sel_id in selection:
                    resolved = _resolve_selected(sel_id)
                    if resolved is None:
                        continue
                    cls_name = entity_selection.instance_class_name(*resolved)
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                summary = ", ".join(f"{n}x {c}" for c, n in sorted(
                    counts.items()))
                for ln in wrap_text(summary or "(none)",
                                    max(20, insp.w - 20), 13):
                    rl.draw_text(ln, insp.x + 10, iy, 13, rl.Color(*COL_TEXT))
                    iy += 16
                iy += 6
                rl.draw_text("T assigns a tag to all selected",
                             insp.x + 10, iy, 13, rl.Color(*COL_TEXT_DIM))
                iy += 16
        elif mode == "PROPS":
            rl.draw_text("[level properties]", insp.x + 10, iy, 18,
                         rl.Color(*COL_TEXT_HOT))
            iy += 26
            rl.draw_text(f"boundary: {level_boundary}  (B toggles)",
                         insp.x + 10, iy, 16, rl.Color(*COL_TEXT))
            iy += 24
            for ln in wrap_text(MODE_HINTS[mode], max(20, insp.w - 20), 13):
                rl.draw_text(ln, insp.x + 10, iy, 13, rl.Color(*COL_TEXT_DIM))
                iy += 16
        elif (selected_entity_class is not None
              and selected_entity_class in registry_payload.get("classes", {})):
            cls_payload = registry_payload["classes"][selected_entity_class]
            rl.draw_text(f"[{selected_entity_class}] template",
                         insp.x + 10, iy, 18, rl.Color(*COL_TEXT_HOT))
            iy += 26
            rows = entity_editor_ui.inspector_rows(cls_payload, {})
            iy = _draw_rows(rows, iy)
        else:
            rl.draw_text(f"[{mode}]", insp.x + 10, iy, 20,
                         rl.Color(*COL_TEXT_HOT))
            iy += 30
            rl.draw_text(f"material: {palette[selected_id][0]}",
                         insp.x + 10, iy, 16, rl.Color(*COL_TEXT_HOT))
            iy += 24
            if extra:
                for ln in wrap_text(extra, max(20, insp.w - 20), 14):
                    rl.draw_text(ln, insp.x + 10, iy, 14, rl.Color(*COL_TEXT))
                    iy += 18
            iy += 6
            for ln in wrap_text(MODE_HINTS[mode], max(20, insp.w - 20), 13):
                rl.draw_text(ln, insp.x + 10, iy, 13, rl.Color(*COL_TEXT_DIM))
                iy += 16
        iy += 6
        rl.draw_text(f"undo {log.undo_count} / redo {log.redo_count}",
                     insp.x + 10, iy, 14, rl.Color(*COL_TEXT_DIM))

        # Status bar: mode | cursor tile | validator summary | registry-
        # import banner (red, Arc C1 — right of the validator slot, left of
        # the unsaved dot: C0's reserved layout) | unsaved dot. A transient
        # action flash (placements, wand paints, ...) takes the slot first;
        # with nothing transient to show, Arc C5's LIVE zone-binding
        # validator (level_loader's own §5 rules, run via zone_entity_port.
        # zone_binding_summary — never re-derived) fills it instead of the
        # old unconditional "ready", whenever the level has any zone content
        # at all.
        if flash_frames > 0:
            flash_frames -= 1
        cur_txt = f"tile ({cur_tx},{cur_ty})" if cursor_in else "tile --"
        if flash_frames > 0:
            valid = flash
        elif zones is not None or any(e.class_name in ZONE_CLASSES
                                      for e in entities):
            valid = zone_entity_port.zone_binding_summary(zones, entities)
        else:
            valid = "ready"
        rl.draw_text(f"{mode}    |    {cur_txt}    |    {valid}",
                     status.x + 10, status.y + 6, 16, rl.Color(*COL_TEXT))
        unsaved_reserve = 0
        if dirty_any:
            tag = "UNSAVED *"
            unsaved_reserve = rl.measure_text(tag, 16) + 12
            rl.draw_text(tag, status.x + status.w - rl.measure_text(tag, 16)
                         - 12, status.y + 6, 16, rl.Color(*COL_BAD))
        if registry_banner:
            bw = rl.measure_text(registry_banner, 14)
            bx = status.x + status.w - unsaved_reserve - bw - 12
            rl.draw_text(registry_banner, bx, status.y + 7, 14,
                         rl.Color(*COL_BAD))

        rl.end_drawing()
        frames += 1

        # ---- --auto smoke plumbing ---------------------------------------
        if auto:
            if frames == AUTO_FRAMES // 3:
                view_baked = False                 # material-view draw path
            if frames == 2 * AUTO_FRAMES // 3:
                view_baked, show_normal = True, False
            if frames == AUTO_FRAMES - 5:
                # raylib's TakeScreenshot writes <cwd>/<basename> — shoot
                # locally, then move into the OS temp dir.
                fname = f"map_editor_auto_{level_dir.name}.png"
                rl.take_screenshot(fname)
                src = Path.cwd() / fname
                shot_path = Path(tempfile.gettempdir()) / fname
                if src.is_file():
                    shutil.move(str(src), str(shot_path))
                else:                              # headless / off-path GL
                    shot_path = None
            if frames >= AUTO_FRAMES:
                break

    # Arc C7 — editor-quit half of the PLAY cleanup contract: whatever is
    # still tracked (a game the user left running, or one that exited this
    # very frame and hasn't been reaped yet) gets its scratch dir removed
    # NOW, unconditionally — best-effort, never raises (crash-safe quit).
    # The launched game already has every level file it needs in memory
    # (level_loader.load reads everything once at startup), so deleting the
    # folder out from under a still-running instance is safe.
    for _proc, _play_dir in active_plays:
        play_scratch.cleanup_scratch_dir(_play_dir)
    active_plays.clear()

    rl.unload_texture(tex_diffuse)
    rl.unload_texture(tex_normal)
    for _tex in entity_icon_textures.values():
        rl.unload_texture(_tex)
    rl.close_window()
    print(f"map_editor: {frames} frames; mode={MODES[mode_idx]} "
          f"unsaved={log.dirty} undo={log.undo_count} redo={log.redo_count} "
          f"spawns={len(spawns)} lights={len(lights)} "
          f"entities={len(entities)} wires={len(wires)} "
          f"water_tiles={int(np.count_nonzero(water_q))} "
          f"zoned_tiles={int(np.count_nonzero(zones)) if zones is not None else 0} "
          f"air_tiles={int(np.count_nonzero(air != air_paint.FP_ONE)) if air is not None else 0} "
          f"boundary={level_boundary}")
    if auto:
        if shot_path is not None and shot_path.is_file():
            print(f"auto screenshot: {shot_path}")
        else:
            print("auto screenshot: NOT captured (headless GL?)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tiled-path map editor (engine/15 §5): paint materials, "
                    "stamp rooms/corridors/doors, place spawns + lights, "
                    "fill water, live baked preview. 'new <name> --size WxH' "
                    "scaffolds a level first.")
    ap.add_argument("level",
                    help="level folder name under levels/ — or the literal "
                         "'new' followed by the name to create")
    ap.add_argument("name", nargs="?", default=None,
                    help="the new level's name (only with 'new')")
    ap.add_argument("--size", default="48x32",
                    help="new-level size WxH in tiles (default 48x32)")
    ap.add_argument("--tileset", default=None,
                    help=f"tileset dir (new: default {DEFAULT_TILESET}; "
                         f"open: the level's [bake].tileset)")
    ap.add_argument("--px-per-tile", type=int, default=None,
                    help="bake resolution (new: default "
                         f"{DEFAULT_PX_PER_TILE}; open: the level's "
                         f"[bake].px_per_tile)")
    ap.add_argument("--seed", type=int, default=None,
                    help="floor-variant seed (new: default 0; open: the "
                         "level's [bake].seed)")
    ap.add_argument("--auto", action="store_true",
                    help="smoke mode: render ~90 frames, save a screenshot "
                         "PNG into the OS temp dir, exit 0")
    return ap


def main(argv=None) -> None:
    # The Windows console defaults to a legacy codepage (cp850/cp1252), which
    # garbles the UTF-8 em-dashes in our messages (mojibake). Reconfigure the
    # streams; errors="replace" keeps output flowing even on a console that
    # can't render a glyph.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.level == "new":
        if not args.name:
            ap.error("'new' needs a level name: map_editor.py new <name> "
                     "--size WxH")
        try:
            w, h = parse_size(args.size)
        except ValueError as e:
            ap.error(str(e))
        level_dir = ROOT / "levels" / args.name
        summary = create_level(
            level_dir, w, h, name=args.name,
            tileset=(args.tileset if args.tileset is not None
                     else DEFAULT_TILESET),
            px_per_tile=(args.px_per_tile if args.px_per_tile is not None
                         else DEFAULT_PX_PER_TILE),
            seed=(args.seed if args.seed is not None else 0))
        print(f"created levels/{args.name}: {w}x{h} tiles, baked @ "
              f"{summary['px_per_tile']} px/tile (tileset "
              f"{summary['tileset']}, seed {summary['seed']})")
        level_name = args.name
    else:
        if args.name is not None:
            ap.error(f"unexpected extra argument {args.name!r} (did you "
                     f"mean: map_editor.py new {args.level}?)")
        level_name = args.level

    run_editor(level_name, tileset_override=args.tileset,
               ppt_override=args.px_per_tile, seed_override=args.seed,
               auto=args.auto)


if __name__ == "__main__":
    main()
