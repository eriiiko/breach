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
survives the panes). Pane geometry is pure in tools/editor_layout.py.

Controls (the tool rail + inspector show the active mode's line):

  Any mode:
    TAB / Shift+TAB      - cycle mode PAINT -> ROOM -> CORRIDOR -> DOOR ->
                           SPAWN -> LIGHT -> WATER; F1..F7 jump straight
                           to a mode
    0-8, 9               - select material (palette GENERATED from
                           MATERIAL_NAMES at launch — key = material id;
                           9 = SPACE; ids past 8 are eyedropper-only)
    V                    - toggle baked preview <-> material-colour view
    N                    - baked view: toggle diffuse <-> normal map
    G                    - toggle grid lines
    Mouse wheel          - zoom (around cursor)
    Middle-drag / WASD   - pan (arrows pan too)
    Ctrl+Z               - undo (tile ring everywhere; SPAWN / LIGHT /
                           WATER modes pop their own rings instead —
                           separate rings, see below)
    Ctrl+S               - SAVE: tilemap.csv + [[spawn]] + [[light]] +
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
  DOOR:   click LMB      - snap MAT_DOOR into a STRAIGHT wall run; corners /
                           T / ends / isolated tiles are refused with a
                           status message (hover shows green/red)
  SPAWN:  LMB            - place a spawn (empty spot) or drag an existing
                           marker; RMB deletes; T toggles the team of the
                           hovered marker (or, off-marker, the placement
                           team); markers draw as team-coloured circles
  LIGHT:  LMB            - place a [[light]] (tile-center snapped) or drag
                           an existing marker; RMB deletes; B toggles
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

Undo rings (reported design call): tile mutations (PAINT/ROOM/CORRIDOR/DOOR)
share ONE UndoRing of grid snapshots; spawn, light and water edits live in
their own rings — Ctrl+Z pops the spawn/light/water ring only while in that
mode. Mixing them into one ring would make Ctrl+Z in PAINT silently rewind
spawn/light/water work (and vice versa).

Spawn/light/water writeback (reported design call, absorbed into level_lib
in Arc A2 — entity doc §3c: level_lib is THE single writer and this editor
is a client): the `[[spawn]]` and `[[light]]` arrays-of-tables and the
`[water]` table are MANAGED BLOCKS — on save every existing table is removed
and the editor's state is written back at the position of the first one (or
EOF), all three families in ONE atomic temp+rename write (a crash mid-save
cannot tear level.toml). Every byte OUTSIDE the managed tables is preserved
(comments inside individual tables are not). level.toml gets ONE .bak per
session, carrying the pre-session bytes. water_init.npy carries its OWN
once-per-session .bak (pre-session bytes, only when the file predates the
session). On save the water grid is masked against the CURRENT materials
(zeroed on solid/SPACE, count reported) — a wall painted over a pool never
saves hidden depth.

Scope limitation on record (P4): the editor refuses levels without a [bake]
block, so the vessel/playground lamp [[light]] entries are loader-consumed
but hand-edited — LIGHT mode authors tiled-path levels only. WATER mode
inherits the same scope (tiled-path levels only); painted levels take a
hand-authored water_init.npy.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
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
from pyray import ffi

import level_lib
import level_loader
# The write surface moved to level_lib in Arc A2 (entity doc §3c: ONE writer
# implementation, ever); these names stay importable from map_editor for the
# pre-A2 callers/tests.
from level_lib import (WATER_FILENAME, color_255, format_light_lines,  # noqa: F401
                       format_spawn_lines, format_water_lines, write_lights,
                       write_spawns, write_water)
from level_loader import SPACE_CODE, LightEntry, SpawnEntry
from level_lights import beacon_angle
from simulation import water_fixed
from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_HULL,
                                  MATERIAL_NAMES, MaterialTable)
from bake_level_art import (BIT_E, BIT_N, BIT_S, BIT_W, DEFAULT_PX_PER_TILE,
                            DEFAULT_TILESET, bake_full, bake_level,
                            bake_region, edge16_mask, load_tileset)
from level_edit_common import (BRUSH_MAX, BRUSH_MIN, UNDO_CAPACITY, UndoRing,
                               art_px_to_tile, brush_rect, build_palette,
                               line_tiles, paint_tiles, save_tilemap_csv)
from editor_layout import compute_panes, fit_camera, screen_from_world

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ("PAINT", "ROOM", "CORRIDOR", "DOOR", "SPAWN", "LIGHT", "WATER")
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


class SpawnRing:
    """UndoRing's little sibling for the spawn list: a fixed-capacity LIFO
    ring of independent SpawnEntry-list snapshots (dataclass copies)."""

    def __init__(self, capacity: int = UNDO_CAPACITY):
        self.capacity = int(capacity)
        self._snaps: list = []

    def __len__(self) -> int:
        return len(self._snaps)

    def push(self, spawns) -> None:
        self._snaps.append([replace(s) for s in spawns])
        if len(self._snaps) > self.capacity:
            self._snaps.pop(0)

    def pop(self):
        return self._snaps.pop() if self._snaps else None


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

# Tool-rail + palette pane row geometry (shared by click-routing and draw so
# the hit targets match the visuals).
RAIL_PAD_Y = 8
RAIL_ROW_H = 30
PAL_PAD_Y = 30            # room for the pane title above the first chip
PAL_ROW_H = 22

MODE_HINTS = {
    "PAINT": "LMB paint  RMB erase  Shift+click line  I eyedrop  [ ] brush",
    "ROOM": "drag LMB: wall perimeter + AIR interior (existing walls "
            "shared)  RMB cancels",
    "CORRIDOR": "drag LMB: AIR swath, walls line the space/solid sides  "
                "+/- width  RMB cancels",
    "DOOR": "click a STRAIGHT wall run -> door (corners/ends refused; "
            "hover shows green/red)",
    "SPAWN": "LMB place/drag  RMB delete  T team toggle "
             "(hovered marker, else placement team)",
    "LIGHT": "LMB place/drag  RMB delete  B kind  C color  R range  "
             "E intens  P period  X beam  H phase (Shift = down)",
    "WATER": "LMB fill enclosed region to depth  RMB fill-to-dry  "
             "-/= depth 0.1 m steps (hover shows green/red)",
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

    preview_ppt = choose_preview_ppt(ts.px, bake_ppt, grid_w, grid_h)
    ppt_pair = (float(preview_ppt), float(preview_ppt))
    offset0 = (0.0, 0.0)

    selected_id = MAT_HULL
    mode_idx = 0
    brush = 1
    corridor_w = DEFAULT_CORRIDOR_WIDTH
    spawn_team = TEAM_MARINE
    spawns = [replace(s) for s in lvl.spawns]
    lights = [replace(l) for l in lvl.lights]
    light_kind = "static"                       # B off-marker toggles this
    light_color = tuple(v / 255.0 for v in LIGHT_COLOR_PRESETS[0][1])

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

    undo = UndoRing()
    spawn_undo = SpawnRing()
    light_undo = SpawnRing()      # copy-generic (dataclass snapshots)
    # Third mode-scoped ring (P5 critique M4, the SpawnRing precedent):
    # the water state is a grid, so the ring IS an UndoRing instance —
    # numpy snapshots, LIFO, capacity-bounded. Popped only in WATER mode.
    water_undo = UndoRing()
    dirty_tiles = False
    dirty_spawns = False
    dirty_lights = False
    dirty_water = False
    csv_bak_written = False
    toml_bak_written = False
    water_bak_written = False     # water_init.npy's OWN once-per-session .bak

    # PAINT stroke state (align-tool pattern).
    stroke_active = False
    stroke_pending = None      # pre-stroke snapshot, pushed on 1st real change
    stroke_dirty = None        # inclusive (x0, y0, x1, y1) of the stroke
    last_paint_tile = None
    anchor_tile = None         # Shift+click line-tool anchor

    room_start = None          # ROOM drag corner (tile)
    corr_start = None          # CORRIDOR drag start (tile)
    spawn_drag = None          # (index, pre-drag list copy, (orig x, orig y))
    light_drag = None          # (index, pre-drag list copy, (orig x, orig y))

    view_baked = True          # V
    show_normal = False        # N (baked view only)
    show_grid = True           # G
    flash, flash_frames = "", 0
    esc_armed = 0

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

    def cancel_transients() -> None:
        """Drop drag/stroke state on a mode switch; a stroke interrupted
        mid-drag still gets its dirty rect re-baked."""
        nonlocal room_start, corr_start, spawn_drag, light_drag
        nonlocal stroke_active, stroke_pending, stroke_dirty, last_paint_tile
        if stroke_dirty is not None:
            rebake_cells(*stroke_dirty)
        room_start = corr_start = spawn_drag = light_drag = None
        stroke_active, stroke_pending, stroke_dirty = False, None, None
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
        dirty_any = (dirty_tiles or dirty_spawns or dirty_lights
                     or dirty_water)

        # ---- global: quit / mode / view ----------------------------------
        if rl.is_key_pressed(K.KEY_ESCAPE):
            if dirty_any and esc_armed <= 0:
                esc_armed = 180
                flash, flash_frames = (
                    "UNSAVED edits — Esc again to quit without saving "
                    "(Ctrl+S saves)", 180)
            else:
                break
        esc_armed = max(0, esc_armed - 1)

        if rl.is_key_pressed(K.KEY_TAB):
            mode_idx = (mode_idx + (-1 if shift else 1)) % len(MODES)
            cancel_transients()
            mode = MODES[mode_idx]
        for i in range(len(MODES)):
            if rl.is_key_pressed(K.KEY_F1 + i):
                mode_idx = i
                cancel_transients()
                mode = MODES[mode_idx]

        if rl.is_key_pressed(K.KEY_V):
            view_baked = not view_baked
        if rl.is_key_pressed(K.KEY_N):
            show_normal = not show_normal
        if rl.is_key_pressed(K.KEY_G):
            show_grid = not show_grid

        # Material palette keys — GENERATED from the material table: the
        # number key IS the material id (SPACE on 9). Ids past 8 have no key
        # (eyedropper reaches them).
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
        if not ctrl:                       # keep Ctrl+S clear of S-pan
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

        lmb = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
        rmb = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT)
        lmb_click = rl.is_mouse_button_pressed(
            rl.MouseButton.MOUSE_BUTTON_LEFT)
        rmb_click = rl.is_mouse_button_pressed(
            rl.MouseButton.MOUSE_BUTTON_RIGHT)
        lmb_up = rl.is_mouse_button_released(
            rl.MouseButton.MOUSE_BUTTON_LEFT)

        # ---- pane routing: clicks outside the canvas act on their pane ----
        # Tool rail row -> select mode (mirrors TAB / F1-F7); palette chip ->
        # select material (mirrors the 0-9 keys). Keyboard stays the primary
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
            pidx = int((mouse.y - panes["palette"].y - PAL_PAD_Y) // PAL_ROW_H)
            if 0 <= pidx < len(palette_order):
                selected_id = palette_order[pidx]

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
                    pid = selected_id if lmb_click else MAT_AIR
                    snap = grid.copy()
                    changed = 0
                    for sx_, sy_ in line_tiles(anchor_tile[0], anchor_tile[1],
                                               cur_tx, cur_ty):
                        changed += _stamp(sx_, sy_, pid)
                    if changed:
                        undo.push(snap)
                        dirty_tiles = True
                        rebake_cells(*stroke_dirty)
                    stroke_dirty = None
                anchor_tile = (cur_tx, cur_ty)
            elif (lmb or rmb) and not over_hud and not shift:
                if not stroke_active:
                    stroke_active = True
                    stroke_pending = grid.copy()
                    stroke_dirty = None
                    last_paint_tile = None
                pid = selected_id if lmb else MAT_AIR      # right = eraser
                p0 = (last_paint_tile if last_paint_tile is not None
                      else (cur_tx, cur_ty))
                changed = 0
                for sx_, sy_ in line_tiles(p0[0], p0[1], cur_tx, cur_ty):
                    changed += _stamp(sx_, sy_, pid)
                last_paint_tile = (cur_tx, cur_ty)
                if cursor_in:
                    anchor_tile = (cur_tx, cur_ty)
                if changed:
                    dirty_tiles = True
                    if stroke_pending is not None:
                        undo.push(stroke_pending)
                        stroke_pending = None
            elif not (lmb or rmb):
                if stroke_dirty is not None:       # stroke END -> re-bake +1
                    rebake_cells(*stroke_dirty)
                    stroke_dirty = None
                stroke_active, stroke_pending = False, None
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
                    snap = grid.copy()
                    changed = apply_room(grid, rect, selected_id, wall_codes)
                    if changed:
                        undo.push(snap)
                        dirty_tiles = True
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
                snap = grid.copy()
                changed = apply_corridor(
                    grid, x0_, y0_, clamp_tx, clamp_ty,
                    width=corridor_w, wall_id=selected_id,
                    wall_codes=wall_codes)
                if changed:
                    undo.push(snap)
                    dirty_tiles = True
                    r = diff_rect(snap, grid)
                    if r is not None:
                        rebake_rect(expand_dirty_rect(r, grid_w, grid_h))
                    flash, flash_frames = (
                        f"corridor w={corridor_w} ({changed} tiles)", 120)
                else:
                    flash, flash_frames = "corridor changed nothing", 120

        # ---- DOOR -----------------------------------------------------------
        elif mode == "DOOR":
            if lmb_click and not over_hud:
                ok, why = door_check(grid, cur_tx, cur_ty, wall_codes)
                if ok:
                    snap = grid.copy()
                    grid[cur_ty, cur_tx] = MAT_DOOR
                    undo.push(snap)
                    dirty_tiles = True
                    rebake_cells(cur_tx, cur_ty, cur_tx, cur_ty)
                    flash, flash_frames = (
                        f"door at ({cur_tx}, {cur_ty}) — {why}", 120)
                else:
                    flash, flash_frames = f"no door: {why}", 180

        # ---- SPAWN ----------------------------------------------------------
        elif mode == "SPAWN":
            hover = spawn_at(spawns, ftx, fty)
            if rl.is_key_pressed(K.KEY_T):
                if hover is not None:
                    spawn_undo.push(spawns)
                    s = spawns[hover]
                    new_team = (TEAM_ZOMBIE if int(s.team) == TEAM_MARINE
                                else TEAM_MARINE)
                    spawns[hover] = replace(s, team=new_team)
                    dirty_spawns = True
                    flash, flash_frames = (
                        f"{s.name} -> {TEAM_NAMES[new_team]}", 120)
                else:
                    spawn_team = (TEAM_ZOMBIE if spawn_team == TEAM_MARINE
                                  else TEAM_MARINE)
                    flash, flash_frames = (
                        f"placing {TEAM_NAMES[spawn_team]}s", 120)
            if lmb_click and not over_hud:
                if hover is not None:
                    spawn_drag = (hover, [replace(s) for s in spawns],
                                  (spawns[hover].x, spawns[hover].y))
                elif cursor_in:
                    spawn_undo.push(spawns)
                    fp = DEFAULT_FOOTPRINT
                    nx = min(max(int(round(ftx - fp / 2.0)), 0), grid_w - fp)
                    ny = min(max(int(round(fty - fp / 2.0)), 0), grid_h - fp)
                    nm = unique_spawn_name(spawns, spawn_team)
                    spawns.append(SpawnEntry(name=nm, team=spawn_team,
                                             x=float(nx), y=float(ny),
                                             footprint=fp))
                    dirty_spawns = True
                    flash, flash_frames = f"spawn {nm} at ({nx}, {ny})", 120
            if spawn_drag is not None and lmb:
                di, _, _ = spawn_drag
                fp = int(spawns[di].footprint)
                nx = min(max(int(round(ftx - fp / 2.0)), 0), grid_w - fp)
                ny = min(max(int(round(fty - fp / 2.0)), 0), grid_h - fp)
                spawns[di] = replace(spawns[di], x=float(nx), y=float(ny))
            if spawn_drag is not None and lmb_up:
                di, pre, orig = spawn_drag
                spawn_drag = None
                if (spawns[di].x, spawns[di].y) != orig:
                    spawn_undo.push(pre)      # pre-drag snapshot
                    dirty_spawns = True
            if rmb_click and not over_hud and hover is not None:
                spawn_undo.push(spawns)
                gone = spawns.pop(hover)
                dirty_spawns = True
                flash, flash_frames = f"deleted spawn {gone.name}", 120

        # ---- LIGHT (P4 §2.4 — the SPAWN interaction template) --------------
        elif mode == "LIGHT":
            hover = light_at(lights, ftx, fty)
            if rl.is_key_pressed(K.KEY_B):        # static <-> beacon
                if hover is not None:
                    new_kind = ("beacon" if lights[hover].kind == "static"
                                else "static")
                    light_undo.push(lights)
                    lights[hover] = replace(lights[hover], kind=new_kind)
                    dirty_lights = True
                    flash, flash_frames = f"light -> {new_kind}", 120
                else:
                    light_kind = ("beacon" if light_kind == "static"
                                  else "static")
                    flash, flash_frames = f"placing {light_kind} lights", 120
            if rl.is_key_pressed(K.KEY_C):        # colour preset cycle
                step = -1 if shift else 1
                if hover is not None:
                    nc = next_light_color(lights[hover].color, step)
                    light_undo.push(lights)
                    lights[hover] = replace(lights[hover], color=nc)
                    dirty_lights = True
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
                    light_undo.push(lights)
                    lights[hover] = replace(lights[hover], **{attr: v})
                    dirty_lights = True
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
                    light_undo.push(lights)
                    lights[hover] = replace(lights[hover], phase=v)
                    dirty_lights = True
                    flash, flash_frames = f"phase = {v:g} turns", 120
            if lmb_click and not over_hud:
                if hover is not None:
                    light_drag = (hover, [replace(l) for l in lights],
                                  (lights[hover].x, lights[hover].y))
                elif cursor_in:
                    light_undo.push(lights)
                    # Tile-center snap (engine/15 §2.2: centers at .5).
                    lx, ly = cur_tx + 0.5, cur_ty + 0.5
                    lights.append(LightEntry(x=lx, y=ly, color=light_color,
                                             kind=light_kind))
                    dirty_lights = True
                    flash, flash_frames = (
                        f"{light_kind} light at ({lx:g}, {ly:g}) "
                        f"[{light_color_name(light_color)}]", 120)
            if light_drag is not None and lmb:
                di, _, _ = light_drag
                if cursor_in:
                    lights[di] = replace(lights[di],
                                         x=cur_tx + 0.5, y=cur_ty + 0.5)
            if light_drag is not None and lmb_up:
                di, pre, orig = light_drag
                light_drag = None
                if (lights[di].x, lights[di].y) != orig:
                    light_undo.push(pre)      # pre-drag snapshot
                    dirty_lights = True
            if rmb_click and not over_hud and hover is not None:
                light_undo.push(lights)
                gone = lights.pop(hover)
                dirty_lights = True
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
                        water_undo.push(water_q)
                        for tx_, ty_ in region:
                            water_q[ty_, tx_] = target
                        dirty_water = True
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

        # ---- undo / save ------------------------------------------------------
        if ctrl and rl.is_key_pressed(K.KEY_Z):
            if mode == "SPAWN":
                spawn_drag = None          # a live drag index would go stale
                snap = spawn_undo.pop()
                if snap is None:
                    flash, flash_frames = "nothing to undo (spawns)", 120
                else:
                    spawns[:] = snap
                    dirty_spawns = True
                    flash, flash_frames = (
                        f"undo spawns ({len(spawn_undo)} left)", 120)
            elif mode == "LIGHT":
                light_drag = None          # a live drag index would go stale
                snap = light_undo.pop()
                if snap is None:
                    flash, flash_frames = "nothing to undo (lights)", 120
                else:
                    lights[:] = snap
                    dirty_lights = True
                    flash, flash_frames = (
                        f"undo lights ({len(light_undo)} left)", 120)
            elif mode == "WATER":
                snap = water_undo.pop()
                if snap is None:
                    flash, flash_frames = "nothing to undo (water)", 120
                else:
                    water_q[...] = snap    # in-place (the live grid persists)
                    dirty_water = True
                    flash, flash_frames = (
                        f"undo water ({len(water_undo)} left)", 120)
            elif stroke_active:
                flash, flash_frames = "release the mouse before undo", 120
            else:
                snap = undo.pop()
                if snap is None:
                    flash, flash_frames = "nothing to undo", 120
                else:
                    r = diff_rect(grid, snap)
                    grid[...] = snap
                    dirty_tiles = True
                    if r is not None:
                        rebake_rect(expand_dirty_rect(r, grid_w, grid_h))
                    flash, flash_frames = f"undo ({len(undo)} left)", 120

        if ctrl and rl.is_key_pressed(K.KEY_S):
            # level.toml writeback goes through level_lib (entity doc §3c:
            # THE single writer): spawn + light + water managed blocks land
            # as ONE atomic temp+rename write, sharing the session's one
            # toml .bak (pre-session bytes). water_init.npy is written
            # first (its OWN once-per-session .bak — P5 §2.4) so a written
            # [water] block never points at a missing file; bake_level then
            # rewrites the [art]/[bake] blocks with write_bak=False, so
            # nothing can clobber the .bak.
            save_tilemap_csv(csv_path, grid, write_bak=not csv_bak_written)
            csv_bak_written = True
            first = not toml_bak_written
            # Wall-over-pool guard (P5 critique M3): mask the water grid
            # against the CURRENT material grid before writing — a wall or
            # SPACE painted over a pool after the fill zeroes those cells
            # (the solver would destroy them anyway, silently). The editor
            # state follows the save so the overlay matches the file.
            masked_water, cleared = mask_water_to_open(water_q, grid,
                                                       water_solid)
            water_q[...] = masked_water
            _, has_water = level_lib.write_water_npy(
                level_dir, water_q, npy_bak=not water_bak_written)
            water_bak_written = True
            handle.save({
                "spawn": lambda nl: format_spawn_lines(spawns, nl),
                "light": lambda nl: format_light_lines(lights, nl),
                "water": level_lib.water_block_format(has_water),
            }, write_bak=first)
            toml_bak_written = True
            summary = bake_level(level_dir, tileset=tileset_arg,
                                 px_per_tile=bake_ppt, seed=bake_seed,
                                 write_bak=False)
            handle.record_disk_state()   # bake rewrote [art]/[bake] blocks
            dirty_tiles = dirty_spawns = dirty_lights = dirty_water = False
            wet = int(np.count_nonzero(water_q))
            flash, flash_frames = (
                f"SAVED tilemap.csv + {len(spawns)} spawns + "
                f"{len(lights)} lights + "
                f"{f'{wet} water tiles' if has_water else 'no water'}"
                f"{f' ({cleared} cleared under walls/space)' if cleared else ''}"
                f" + bake blocks + "
                f"full bake @ {summary['px_per_tile']} px/tile"
                f"{' (.bak written)' if first else ''}", 240)

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
        elif mode == "DOOR" and not over_hud and cursor_in:
            ok, _why = door_check(grid, cur_tx, cur_ty, wall_codes)
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

        # Top bar: level / grid / zoom / view mode.
        preview_note = ("" if preview_ppt == bake_ppt
                        else f"  preview @{preview_ppt}px (bake {bake_ppt})")
        view_name = ("normals" if (view_baked and show_normal)
                     else "baked" if view_baked else "materials")
        rl.draw_text(
            f"{level_dir.name}   grid {grid_w}x{grid_h}   zoom {zoom:.2f}"
            f"   view: {view_name} (V/N){preview_note}",
            top.x + 10, top.y + 12, 18, rl.Color(*COL_TEXT))

        # Tool rail: the mode list (click a row, or TAB / F1-F7).
        for i, mname in enumerate(MODES):
            ry = rail.y + RAIL_PAD_Y + i * RAIL_ROW_H
            if ry + RAIL_ROW_H > rail.y + rail.h:
                break
            if i == mode_idx:
                rl.draw_rectangle(rail.x + 4, ry, max(0, rail.w - 8),
                                  RAIL_ROW_H - 4, rl.Color(*COL_RAIL_SEL))
            col = COL_TEXT_HOT if i == mode_idx else COL_TEXT_DIM
            rl.draw_text(f"F{i + 1} {mname}", rail.x + 12, ry + 6, 16,
                         rl.Color(*col))

        # Palette pane: chips + names, selected boxed (click a chip = 0-9).
        rl.draw_text("MATERIAL", pal.x + 10, pal.y + 8, 14,
                     rl.Color(*COL_TEXT))
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

        # Inspector pane: active tool + material + per-mode extras + hint.
        # (C1 replaces this with the registry-driven inspector.)
        iy = insp.y + 10
        rl.draw_text(f"[{mode}]", insp.x + 10, iy, 20, rl.Color(*COL_TEXT_HOT))
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
        rl.draw_text(f"undo depth {len(undo)}", insp.x + 10, iy, 14,
                     rl.Color(*COL_TEXT_DIM))

        # Status bar: mode | cursor tile | validator summary | unsaved dot |
        # registry-import banner slot (empty in C0 — C1 fills it). The
        # transient status flash rides the validator slot until C5 lands real
        # validators.
        if flash_frames > 0:
            flash_frames -= 1
        cur_txt = f"tile ({cur_tx},{cur_ty})" if cursor_in else "tile --"
        valid = flash if flash_frames > 0 else "ready"
        rl.draw_text(f"{mode}    |    {cur_txt}    |    {valid}",
                     status.x + 10, status.y + 6, 16, rl.Color(*COL_TEXT))
        if dirty_any:
            tag = "UNSAVED *"
            rl.draw_text(tag, status.x + status.w - rl.measure_text(tag, 16)
                         - 12, status.y + 6, 16, rl.Color(*COL_BAD))

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

    rl.unload_texture(tex_diffuse)
    rl.unload_texture(tex_normal)
    rl.close_window()
    print(f"map_editor: {frames} frames; mode={MODES[mode_idx]} "
          f"unsaved_tiles={dirty_tiles} unsaved_spawns={dirty_spawns} "
          f"unsaved_lights={dirty_lights} unsaved_water={dirty_water} "
          f"spawns={len(spawns)} lights={len(lights)} "
          f"water_tiles={int(np.count_nonzero(water_q))}")
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
