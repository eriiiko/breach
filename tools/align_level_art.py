"""tools/align_level_art.py — level editor v1: ALIGN + MATERIAL PAINT + SAVE
(standalone slice of F4).

Loads a level via level_loader, draws one of its diffuse layers (bare /
furniture / destroyed — B cycles whichever exist; the layers are
pixel-registered so they share one transform) in art-pixel space and overlays
the tilemap through the live [art.align] transform (offset_px + per-axis
px_per_tile). What you align/paint here is exactly what the lighting shader
will sample: the overlay rect for tile (tx, ty) is
``tile_to_art_px(tx, ty, offset_px, px_per_tile)`` — the same single source
of truth the renderer's src-rect math uses (renderer/lighting.py
art_src_and_uv_rect). Painting edits the in-memory tilemap live (every
non-air id is filled with its palette colour, so the stroke is WYSIWYG).

The editor proposal (§4, Q5) chose an *in-game* editor — that remains the
target. This standalone tool is the pragmatic v1 slice: the level's stale CSV
has to be repainted against the art now, and everything here (palette, brush,
inverse transform, CSV save) carries over.

Run:
    C:/Users/steen/anaconda3/python.exe tools/align_level_art.py [level_name] [--auto]

level_name defaults to unhcr_vessel_2. --auto closes after ~90 frames
(smoke-test plumbing; flips to PAINT mode halfway to exercise that draw path)
and then runs one programmatic paint stroke + CSV save against a TMP COPY of
the tilemap — raylib has no input-injection API, so the factored paint/save
functions are called directly; the real level files are never written.

Controls — two modes, TAB toggles (HUD shows the active one):

  Both modes:
    TAB                  - toggle ALIGN <-> PAINT
    B                    - cycle backdrop: bare -> furniture -> destroyed
                           (default on launch: furniture if present, else bare)
    Mouse wheel          - zoom (around cursor)
    Middle-drag          - pan
    G                    - toggle SPACE fill in the overlay (on by default)
    L                    - toggle grid lines
    Shift+click          - line tool: fill the segment from the last painted /
                           clicked tile (the anchor) to the cursor with the
                           selected material; Shift+right-click erases the
                           segment; consecutive Shift+clicks chain a polyline
    Ctrl+Z               - undo last paint stroke (ring of ~100)
    Ctrl+S               - SAVE BOTH: tilemap.csv (canon v2 codes, newline
                           style preserved, .bak once per session) + the
                           [art.align] lines of level.toml (.bak per save)
    Esc                  - quit (pressed twice if there are unsaved paint edits)

  ALIGN mode (the original tool — keys unchanged):
    WASD / left-drag     - pan
    Arrow keys           - offset_px +-1 px      (Shift = +-10)
    X / Y                - select the axis that +/- scales
    + / -                - px_per_tile[active axis] +-0.1   (Shift = +-1.0)
                           (main-row =/- and keypad +/- both work)
    R                    - reset align to the values in level.toml (or last save)

  PAINT mode (level format v2 only — CSV codes ARE canon material ids):
    Left click/drag      - paint the selected id under the cursor
    Right click/drag     - erase (paint AIR)
    0 1 2 3 4 5 6        - select AIR / HULL / WOOD / DOOR / STEEL / GLASS /
                           FURNITURE
    9                    - select SPACE (code 9)
    I                    - eyedropper (select the id under the cursor)
    [ / ]                - brush size 1..9 (square; shown in HUD)
    WASD + arrow keys    - pan (SPACE paints via 9 only)
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

# Make project modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyray as rl

from level_loader import (SPACE_CODE, load as load_level,
                          materials_from_tilemap, tile_to_art_px)
from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_FURNITURE, MAT_GLASS,
                                  MAT_HULL, MAT_STEEL, MAT_WOOD,
                                  MATERIAL_NAMES)
# The pure paint plumbing is shared with tools/map_editor.py (engine/15 §0).
# Re-exported here so tests/test_level_editor_tool.py (and any older caller)
# keeps importing them from this module unchanged.
from level_edit_common import (BRUSH_MAX, BRUSH_MIN, UNDO_CAPACITY,  # noqa: F401
                               UndoRing, art_px_to_tile, brush_rect,
                               build_palette, line_tiles, paint_tiles,
                               save_tilemap_csv)


# ---------------------------------------------------------------------------
# [art.align] in-place save — byte-preserving except the align lines
# ---------------------------------------------------------------------------

_ALIGN_HEADER_RE = re.compile(r"^\s*\[art\.align\]\s*(#.*)?$")
_TABLE_HEADER_RE = re.compile(r"^\s*\[")          # any table / array-of-tables
_OFFSET_RE = re.compile(r"^\s*offset_px\s*=")
_PPT_RE = re.compile(r"^\s*px_per_tile\s*=")


def format_align_lines(offset_px, px_per_tile) -> tuple:
    """Canonical [art.align] assignment lines: offset to 1 decimal, px_per_tile
    (always saved as the per-axis pair) to 2 decimals."""
    off = f"offset_px = [{float(offset_px[0]):.1f}, {float(offset_px[1]):.1f}]"
    ppt = (f"px_per_tile = [{float(px_per_tile[0]):.2f}, "
           f"{float(px_per_tile[1]):.2f}]")
    return off, ppt


def save_align(toml_path, offset_px, px_per_tile) -> Path:
    """Rewrite ONLY the [art.align] assignment lines of ``toml_path`` in place.

    Every other byte of the file (comments inside the block included) is
    preserved. The original bytes are written to ``<toml_path>.bak`` first.
    If an assignment line is missing inside an existing [art.align] block it
    is appended at the end of the block; if the block itself is missing a
    fresh one is appended at end of file. Returns the .bak path.
    """
    toml_path = Path(toml_path)
    original = toml_path.read_bytes()
    text = original.decode("utf-8")
    lines = text.splitlines(keepends=True)
    nl = "\r\n" if "\r\n" in text else "\n"   # match the file's newline style
    off_line, ppt_line = format_align_lines(offset_px, px_per_tile)

    def ending(ln: str) -> str:
        e = ln[len(ln.rstrip("\r\n")):]
        return e if e else nl                  # last line may lack a newline

    out = list(lines)
    start = next((i for i, ln in enumerate(out)
                  if _ALIGN_HEADER_RE.match(ln)), None)
    if start is None:
        # No [art.align] block: append a fresh one at EOF (valid TOML — the
        # dotted header reopens under [art] regardless of position).
        if out and not out[-1].endswith(("\n", "\r")):
            out[-1] += nl
        out += [nl, "[art.align]" + nl, off_line + nl, ppt_line + nl]
    else:
        end = next((i for i in range(start + 1, len(out))
                    if _TABLE_HEADER_RE.match(out[i])), len(out))
        replaced_off = replaced_ppt = False
        for i in range(start + 1, end):
            if not replaced_off and _OFFSET_RE.match(out[i]):
                out[i] = off_line + ending(out[i])
                replaced_off = True
            elif not replaced_ppt and _PPT_RE.match(out[i]):
                out[i] = ppt_line + ending(out[i])
                replaced_ppt = True
        inserts = []
        if not replaced_off:
            inserts.append(off_line + nl)
        if not replaced_ppt:
            inserts.append(ppt_line + nl)
        if inserts:
            if end > 0 and not out[end - 1].endswith(("\n", "\r")):
                out[end - 1] += nl
            out[end:end] = inserts

    bak = Path(str(toml_path) + ".bak")
    bak.write_bytes(original)
    toml_path.write_bytes("".join(out).encode("utf-8"))
    return bak


# ---------------------------------------------------------------------------
# MATERIAL PAINT — palette + view constants. The pure paint helpers
# (art_px_to_tile / brush_rect / paint_tiles / line_tiles / save_tilemap_csv /
# UndoRing) live in tools/level_edit_common.py, shared with the map editor,
# and are re-exported above for tests/test_level_editor_tool.py.
# ---------------------------------------------------------------------------

# Editor palette over the canon v2 CSV vocabulary (level format v2 §1.1:
# codes ARE material ids, plus SPACE_CODE). id -> (display name, overlay RGB);
# AIR has no fill (None) — it is the absence of an overlay. Generated from
# MATERIAL_NAMES (§1 rule: no tool-local vocabulary).
PALETTE = build_palette()
PALETTE_ORDER = tuple(sorted(MATERIAL_NAMES)) + (SPACE_CODE,)
OVERLAY_ALPHA = 102            # ~40% — live per-material fill (WYSIWYG paint)
OVERLAY_COLORS = {pid: (c[0], c[1], c[2], OVERLAY_ALPHA)
                  for pid, (_, c) in PALETTE.items() if c is not None}


# ---------------------------------------------------------------------------
# Interactive viewer / editor
# ---------------------------------------------------------------------------

WIN_W, WIN_H = 1280, 920
AUTO_FRAMES = 90          # --auto: close after ~90 frames (smoke test)
HUD_H = 100

COL_GRID = (255, 255, 255, 40)     # thin grid lines
COL_HUD_BG = (0, 0, 0, 170)
COL_CURSOR = (255, 255, 255, 220)  # brush footprint outline
COL_TEXT = (200, 200, 200, 255)
COL_TEXT_DIM = (150, 150, 160, 255)
COL_TEXT_HOT = (255, 230, 120, 255)


def _pressed(key) -> bool:
    """Key pressed this frame, with OS key-repeat while held."""
    return rl.is_key_pressed(key) or rl.is_key_pressed_repeat(key)


def _auto_paint_check(grid: np.ndarray, csv_path: Path) -> None:
    """--auto tail: exercise one programmatic paint stroke + CSV save against
    a TMP COPY of the tilemap. raylib offers no input-injection API, so the
    factored functions (UndoRing/paint_tiles/save_tilemap_csv) are called
    directly — the real level files are never written. SystemExit(1) on
    failure."""
    h, w = grid.shape
    cy, cx = h // 2, w // 2
    with tempfile.TemporaryDirectory(prefix="breach_editor_auto_") as td:
        tmp_csv = Path(td) / "tilemap.csv"
        shutil.copy2(csv_path, tmp_csv)
        original = tmp_csv.read_bytes()

        g = grid.copy()
        ring = UndoRing()
        ring.push(g)                            # one snapshot per stroke
        pid = MAT_HULL if int(g[cy, cx]) != MAT_HULL else MAT_WOOD
        changed = paint_tiles(g, cx, cy, pid, brush=3)
        bak = save_tilemap_csv(tmp_csv, g, write_bak=True)
        new_bytes = tmp_csv.read_bytes()

        ok = (changed > 0
              and new_bytes != original
              and bak is not None and bak.read_bytes() == original
              and (b"\r\n" in new_bytes) == (b"\r\n" in original)
              and np.array_equal(ring.pop(), grid))
        if not ok:
            print("auto paint-stroke check FAILED")
            raise SystemExit(1)
    print("auto paint-stroke check OK (direct function calls - raylib has no "
          "input injection; tmp copy of tilemap.csv, real level untouched)")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    auto = "--auto" in sys.argv
    level_name = args[0] if args else "playground"

    lvl = load_level(level_name)
    grid = np.array(lvl.tilemap, dtype=np.int32, copy=True)  # live paint target
    grid_h, grid_w = grid.shape
    csv_path = lvl.path / str(lvl.raw_toml["tilemap"])

    # PAINT needs the v2 vocabulary (codes ARE material ids); a v1 CSV speaks
    # the retired generator vocabulary and must not be painted with canon ids.
    paintable = (lvl.version == "2")
    if paintable:
        materials_from_tilemap(grid, lvl.version)   # validates codes (raises)
        overlay_codes = grid     # alias: paint edits show in the overlay live
    else:
        mat, vac = materials_from_tilemap(grid, lvl.version)
        overlay_codes = mat.astype(np.int32)        # static per-material view
        overlay_codes[vac] = SPACE_CODE

    # Backdrop cycle (B): the pixel-registered diffuse layers from [art.*].
    backdrops = [("bare", lvl.diffuse_path)]
    if lvl.furniture_diffuse_path is not None:
        backdrops.append(("furniture", lvl.furniture_diffuse_path))
    if lvl.destroyed_diffuse_path is not None:
        backdrops.append(("destroyed", lvl.destroyed_diffuse_path))

    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(WIN_W, WIN_H,
                   f"Breach editor (align+paint) — {lvl.name} ({level_name})")
    rl.set_target_fps(60)
    rl.set_exit_key(rl.KeyboardKey.KEY_NULL)   # Esc handled below (dirty guard)

    tex_cache: dict = {}

    def get_backdrop(idx: int):
        """Texture for backdrops[idx], loaded on first use (None on failure)."""
        name, path = backdrops[idx]
        if name not in tex_cache:
            t = rl.load_texture(str(path))
            if t.id == 0:
                tex_cache[name] = None
            else:
                rl.set_texture_filter(
                    t, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
                tex_cache[name] = t
        return tex_cache[name]

    # Default backdrop: FURNITURE if present, else bare — the CSV gets
    # repainted against the furnished picture; B cycles the others.
    backdrop_idx = next((i for i, (n, _) in enumerate(backdrops)
                         if n == "furniture"), 0)
    tex = get_backdrop(backdrop_idx)
    if tex is None and backdrop_idx != 0:
        backdrop_idx, tex = 0, get_backdrop(0)
    if tex is None:
        rl.close_window()
        raise SystemExit(f"Could not load diffuse texture: "
                         f"{backdrops[backdrop_idx][1]}")
    art_w, art_h = float(tex.width), float(tex.height)

    # The live align state. The loader normalized px_per_tile to a pair; the
    # (diffuse-unreadable) None fallback derives the legacy stretch transform
    # from the loaded texture.
    if lvl.art_px_per_tile is not None:
        ppt = [float(lvl.art_px_per_tile[0]), float(lvl.art_px_per_tile[1])]
    else:
        ppt = [art_w / grid_w, art_h / grid_h]
    offset = [float(lvl.art_offset_px[0]), float(lvl.art_offset_px[1])]
    baseline = (list(offset), list(ppt))    # R resets to this; Ctrl+S rebases
    active_axis = 0                          # 0 = X, 1 = Y (the +/- target)
    show_space = True                        # G — SPACE fill (WYSIWYG default)
    show_grid = True                         # L
    flash, flash_frames = "", 0

    # PAINT state.
    mode_paint = False                       # TAB; start in ALIGN (unsurprising)
    selected_id = MAT_HULL
    brush = 1
    undo = UndoRing()
    stroke_pending = None    # pre-stroke snapshot; pushed on the 1st real change
    stroke_active = False
    last_paint_tile = None   # drag-interpolation anchor
    anchor_tile = None       # line-tool anchor: the last tile painted/clicked;
                             # Shift+click fills the line anchor -> cursor
    dirty = False            # unsaved paint edits
    csv_bak_written = False  # tilemap.csv.bak: once per session
    esc_armed = 0            # frames left of "Esc again to quit" arming

    # View transform (screen = (art_px - cam) * zoom): fit the art on start.
    zoom = min(WIN_W / art_w, WIN_H / art_h)
    cam_x = (art_w - WIN_W / zoom) / 2.0
    cam_y = (art_h - WIN_H / zoom) / 2.0

    frames = 0
    while not rl.window_should_close():
        win_w, win_h = rl.get_screen_width(), rl.get_screen_height()
        dt = rl.get_frame_time()
        shift = (rl.is_key_down(rl.KeyboardKey.KEY_LEFT_SHIFT)
                 or rl.is_key_down(rl.KeyboardKey.KEY_RIGHT_SHIFT))
        ctrl = (rl.is_key_down(rl.KeyboardKey.KEY_LEFT_CONTROL)
                or rl.is_key_down(rl.KeyboardKey.KEY_RIGHT_CONTROL))
        mouse = rl.get_mouse_position()
        over_hud = mouse.y <= HUD_H

        # ---- global: quit / mode / backdrop ------------------------------
        if rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
            if dirty and esc_armed <= 0:
                esc_armed = 180
                flash, flash_frames = (
                    "UNSAVED paint edits — Esc again to quit without saving "
                    "(Ctrl+S saves)", 180)
            else:
                break
        esc_armed = max(0, esc_armed - 1)

        if rl.is_key_pressed(rl.KeyboardKey.KEY_TAB):
            if not mode_paint and not paintable:
                flash, flash_frames = (
                    f"PAINT needs level format v2 — this level is "
                    f"v{lvl.version} (migrate first)", 180)
            else:
                mode_paint = not mode_paint
                stroke_active, stroke_pending = False, None
                last_paint_tile = None

        if rl.is_key_pressed(rl.KeyboardKey.KEY_B) and len(backdrops) > 1:
            nxt = (backdrop_idx + 1) % len(backdrops)
            t = get_backdrop(nxt)
            if t is None:
                flash, flash_frames = (
                    f"could not load '{backdrops[nxt][0]}' layer "
                    f"({backdrops[nxt][1].name})", 180)
            else:
                backdrop_idx, tex = nxt, t
                art_w, art_h = float(tex.width), float(tex.height)

        # ---- view: wheel zoom around cursor, keys/drag pan ---------------
        wheel = rl.get_mouse_wheel_move()
        if wheel != 0.0:
            art_mx0, art_my0 = cam_x + mouse.x / zoom, cam_y + mouse.y / zoom
            zoom = max(0.02, min(50.0, zoom * (1.1 ** wheel)))
            cam_x, cam_y = art_mx0 - mouse.x / zoom, art_my0 - mouse.y / zoom
        if not ctrl:                       # keep Ctrl+S clear of S-pan
            pan = 900.0 * dt / zoom        # screen px/s -> art px
            if rl.is_key_down(rl.KeyboardKey.KEY_W):
                cam_y -= pan
            if rl.is_key_down(rl.KeyboardKey.KEY_A):
                cam_x -= pan
            if rl.is_key_down(rl.KeyboardKey.KEY_D):
                cam_x += pan
            if rl.is_key_down(rl.KeyboardKey.KEY_S):
                cam_y += pan               # S pans in BOTH modes (Erik's call;
                                           # SPACE paints via 9 only)
            if mode_paint:                 # PAINT: arrows pan (no align nudge)
                if rl.is_key_down(rl.KeyboardKey.KEY_UP):
                    cam_y -= pan
                if rl.is_key_down(rl.KeyboardKey.KEY_DOWN):
                    cam_y += pan
                if rl.is_key_down(rl.KeyboardKey.KEY_LEFT):
                    cam_x -= pan
                if rl.is_key_down(rl.KeyboardKey.KEY_RIGHT):
                    cam_x += pan
        drag_pan = (rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_MIDDLE)
                    or (not mode_paint and rl.is_mouse_button_down(
                        rl.MouseButton.MOUSE_BUTTON_LEFT)))
        if drag_pan:                       # PAINT: left button paints, not pans
            d = rl.get_mouse_delta()
            cam_x -= d.x / zoom
            cam_y -= d.y / zoom

        # Cursor tile under the mouse — screen -> art px -> tile via the
        # INVERSE of the live align transform (same math the overlay uses).
        ftx, fty = art_px_to_tile(cam_x + mouse.x / zoom,
                                  cam_y + mouse.y / zoom, offset, ppt)
        cur_tx, cur_ty = int(np.floor(ftx)), int(np.floor(fty))
        cursor_in = (0 <= cur_tx < grid_w and 0 <= cur_ty < grid_h)

        # ---- ALIGN edits --------------------------------------------------
        if not mode_paint:
            step = 10.0 if shift else 1.0
            if _pressed(rl.KeyboardKey.KEY_RIGHT):
                offset[0] += step
            if _pressed(rl.KeyboardKey.KEY_LEFT):
                offset[0] -= step
            if _pressed(rl.KeyboardKey.KEY_DOWN):
                offset[1] += step
            if _pressed(rl.KeyboardKey.KEY_UP):
                offset[1] -= step
            if rl.is_key_pressed(rl.KeyboardKey.KEY_X):
                active_axis = 0
            if rl.is_key_pressed(rl.KeyboardKey.KEY_Y):
                active_axis = 1
            sstep = 1.0 if shift else 0.1
            if (_pressed(rl.KeyboardKey.KEY_KP_ADD)
                    or _pressed(rl.KeyboardKey.KEY_EQUAL)):
                ppt[active_axis] += sstep
            if (_pressed(rl.KeyboardKey.KEY_KP_SUBTRACT)
                    or _pressed(rl.KeyboardKey.KEY_MINUS)):
                ppt[active_axis] = max(0.01, ppt[active_axis] - sstep)
            if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
                offset, ppt = list(baseline[0]), list(baseline[1])
                flash, flash_frames = "reset to level.toml values", 120

        # ---- PAINT input ---------------------------------------------------
        if mode_paint:
            K = rl.KeyboardKey
            for keys, pid in (
                    ((K.KEY_ZERO, K.KEY_KP_0), MAT_AIR),
                    ((K.KEY_ONE, K.KEY_KP_1), MAT_HULL),
                    ((K.KEY_TWO, K.KEY_KP_2), MAT_WOOD),
                    ((K.KEY_THREE, K.KEY_KP_3), MAT_DOOR),
                    ((K.KEY_FOUR, K.KEY_KP_4), MAT_STEEL),
                    ((K.KEY_FIVE, K.KEY_KP_5), MAT_GLASS),
                    ((K.KEY_SIX, K.KEY_KP_6), MAT_FURNITURE),
                    ((K.KEY_NINE, K.KEY_KP_9), SPACE_CODE)):
                if any(rl.is_key_pressed(k) for k in keys):
                    selected_id = pid
            if rl.is_key_pressed(K.KEY_I) and cursor_in:  # eyedropper
                selected_id = int(grid[cur_ty, cur_tx])
            if _pressed(K.KEY_LEFT_BRACKET):
                brush = max(BRUSH_MIN, brush - 1)
            if _pressed(K.KEY_RIGHT_BRACKET):
                brush = min(BRUSH_MAX, brush + 1)

            lmb = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
            rmb = rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT)
            lmb_click = rl.is_mouse_button_pressed(
                rl.MouseButton.MOUSE_BUTTON_LEFT)
            rmb_click = rl.is_mouse_button_pressed(
                rl.MouseButton.MOUSE_BUTTON_RIGHT)
            if shift and (lmb_click or rmb_click) and not over_hud and cursor_in:
                # LINE TOOL: Shift+click fills anchor -> cursor with the brush
                # (Erik's flow: click a tile, release, Shift+click the far end;
                # chaining Shift+clicks draws a polyline). Shift+RMB erases.
                if anchor_tile is not None:
                    pid = selected_id if lmb_click else MAT_AIR
                    snap = grid.copy()             # one undo entry per segment
                    changed = 0
                    for sx_, sy_ in line_tiles(anchor_tile[0], anchor_tile[1],
                                               cur_tx, cur_ty):
                        changed += paint_tiles(grid, sx_, sy_, pid, brush)
                    if changed:
                        undo.push(snap)
                        dirty = True
                anchor_tile = (cur_tx, cur_ty)
            elif (lmb or rmb) and not over_hud and not shift:
                if not stroke_active:
                    stroke_active = True
                    stroke_pending = grid.copy()   # one undo entry per stroke
                    last_paint_tile = None
                pid = selected_id if lmb else MAT_AIR     # right = eraser
                p0 = (last_paint_tile if last_paint_tile is not None
                      else (cur_tx, cur_ty))
                changed = 0
                for sx_, sy_ in line_tiles(p0[0], p0[1], cur_tx, cur_ty):
                    changed += paint_tiles(grid, sx_, sy_, pid, brush)
                last_paint_tile = (cur_tx, cur_ty)
                if cursor_in:
                    anchor_tile = (cur_tx, cur_ty)  # line tool follows the brush
                if changed:
                    dirty = True
                    if stroke_pending is not None:
                        undo.push(stroke_pending)
                        stroke_pending = None
            elif not (lmb or rmb):
                stroke_active, stroke_pending = False, None
                last_paint_tile = None
            else:                  # held but over the HUD: break the line
                last_paint_tile = None

        # ---- undo / save (both modes) ---------------------------------------
        if ctrl and rl.is_key_pressed(rl.KeyboardKey.KEY_Z):
            if stroke_active:
                flash, flash_frames = "release the mouse before undo", 120
            else:
                snap = undo.pop()
                if snap is None:
                    flash, flash_frames = "nothing to undo", 120
                else:
                    grid[...] = snap          # keeps the overlay alias live
                    dirty = True
                    flash, flash_frames = f"undo ({len(undo)} left)", 120
        if ctrl and rl.is_key_pressed(rl.KeyboardKey.KEY_S):
            bak_t = save_align(lvl.path / "level.toml", offset, ppt)
            baseline = (list(offset), list(ppt))
            if paintable:
                save_tilemap_csv(csv_path, grid,
                                 write_bak=not csv_bak_written)
                first = not csv_bak_written
                csv_bak_written = True
                dirty = False
                flash, flash_frames = (
                    f"SAVED tilemap.csv + [art.align]"
                    f"{' (.bak written)' if first else ''}", 180)
            else:
                flash, flash_frames = (
                    f"SAVED [art.align] (backup: {bak_t.name}) — "
                    f"v{lvl.version} CSV is not painted here", 180)

        # ---- draw -----------------------------------------------------------
        def to_screen(ax: float, ay: float) -> tuple:
            return ((ax - cam_x) * zoom, (ay - cam_y) * zoom)

        rl.begin_drawing()
        rl.clear_background(rl.Color(24, 24, 28, 255))

        dx, dy = to_screen(0.0, 0.0)
        rl.draw_texture_pro(
            tex, rl.Rectangle(0, 0, art_w, art_h),
            rl.Rectangle(dx, dy, art_w * zoom, art_h * zoom),
            rl.Vector2(0, 0), 0.0, rl.WHITE)

        # Tile overlays in ART space, mapped through the same view transform —
        # the rect for (tx, ty) sits exactly where the shader samples it.
        # Every non-air id fills with its palette colour from the LIVE grid,
        # so painting is WYSIWYG. Only the visible tile window is walked.
        tw, th = ppt[0] * zoom, ppt[1] * zoom

        def tile_rect(tx: int, ty: int):
            ax, ay = tile_to_art_px(tx, ty, offset, ppt)
            sx, sy = to_screen(ax, ay)
            if sx + tw < 0 or sy + th < 0 or sx > win_w or sy > win_h:
                return None
            return sx, sy

        vx0, vy0 = art_px_to_tile(cam_x, cam_y, offset, ppt)
        vx1, vy1 = art_px_to_tile(cam_x + win_w / zoom,
                                  cam_y + win_h / zoom, offset, ppt)
        tx0, ty0 = max(0, int(np.floor(vx0))), max(0, int(np.floor(vy0)))
        tx1 = min(grid_w, int(np.ceil(vx1)) + 1)
        ty1 = min(grid_h, int(np.ceil(vy1)) + 1)
        sub = overlay_codes[ty0:ty1, tx0:tx1]
        ys, xs = np.nonzero(sub != MAT_AIR)
        for tx_, ty_, v in zip((xs + tx0).tolist(), (ys + ty0).tolist(),
                               sub[ys, xs].tolist()):
            if v == SPACE_CODE and not show_space:
                continue
            col = OVERLAY_COLORS.get(v)
            if col is None:
                continue
            r = tile_rect(tx_, ty_)
            if r:
                rl.draw_rectangle(int(r[0]), int(r[1]),
                                  max(1, int(tw)), max(1, int(th)), col)
        if show_grid:
            for gx in range(grid_w + 1):
                a0 = to_screen(*tile_to_art_px(gx, 0, offset, ppt))
                a1 = to_screen(*tile_to_art_px(gx, grid_h, offset, ppt))
                rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 1.0, COL_GRID)
            for gy in range(grid_h + 1):
                a0 = to_screen(*tile_to_art_px(0, gy, offset, ppt))
                a1 = to_screen(*tile_to_art_px(grid_w, gy, offset, ppt))
                rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 1.0, COL_GRID)

        # Brush footprint outline at the cursor (PAINT mode).
        if mode_paint and not over_hud:
            br = brush_rect(cur_tx, cur_ty, brush, grid_w, grid_h)
            if br is not None:
                ax0, ay0 = tile_to_art_px(br[0], br[1], offset, ppt)
                ax1, ay1 = tile_to_art_px(br[2] + 1, br[3] + 1, offset, ppt)
                s0, s1 = to_screen(ax0, ay0), to_screen(ax1, ay1)
                rl.draw_rectangle_lines_ex(
                    rl.Rectangle(s0[0], s0[1], s1[0] - s0[0], s1[1] - s0[1]),
                    2.0, COL_CURSOR)

        # ---- HUD ------------------------------------------------------------
        rl.draw_rectangle(0, 0, win_w, HUD_H, rl.Color(*COL_HUD_BG))
        rl.draw_text(
            f"{level_name}  art {int(art_w)}x{int(art_h)} px  "
            f"grid {grid_w}x{grid_h} tiles  zoom {zoom:.2f}   |   "
            f"backdrop: {backdrops[backdrop_idx][0]} (B)   |   "
            f"mode: {'PAINT' if mode_paint else 'ALIGN'} (TAB)"
            f"{'   *UNSAVED*' if dirty else ''}", 8, 6, 18, rl.Color(*COL_TEXT))
        if mode_paint:
            x = 8
            for pid in PALETTE_ORDER:
                pname, c = PALETTE[pid]
                label = (f"9 {pname}" if pid == SPACE_CODE
                         else f"{pid} {pname}")
                chip = (rl.Color(c[0], c[1], c[2], 255) if c is not None
                        else rl.Color(70, 70, 76, 255))
                rl.draw_rectangle(x, 30, 16, 16, chip)
                if pid == selected_id:
                    rl.draw_rectangle_lines_ex(
                        rl.Rectangle(x - 2, 28, 20, 20), 2.0, rl.WHITE)
                rl.draw_text(label, x + 21, 30, 16,
                             rl.WHITE if pid == selected_id
                             else rl.Color(*COL_TEXT_DIM))
                x += 21 + rl.measure_text(label, 16) + 14
            rl.draw_text(f"|  brush {brush}x{brush}   undo {len(undo)}",
                         x + 2, 30, 16, rl.Color(*COL_TEXT_HOT))
            line3 = ("LMB paint  RMB erase  Shift+click line  |  0-6 material, "
                     "9 space  |  I eyedrop  |  [ ] brush  |  Ctrl+Z undo")
            line4 = ("Ctrl+S save csv+align | B backdrop | G space fill | "
                     "L grid | TAB align | wheel zoom, MMB drag / "
                     "WAD+arrows pan | Esc quit")
        else:
            rl.draw_text(
                f"offset_px = [{offset[0]:.1f}, {offset[1]:.1f}]   "
                f"px_per_tile = [{ppt[0]:.2f}, {ppt[1]:.2f}]   "
                f"axis: {'X' if active_axis == 0 else 'Y'}", 8, 28, 20,
                rl.Color(*COL_TEXT_HOT))
            line3 = ("arrows offset (Shift x10) | X/Y axis | +/- scale "
                     "(Shift x10) | R reset")
            line4 = ("Ctrl+S save csv+align | B backdrop | G space | L grid | "
                     "TAB paint | wheel zoom, LMB/MMB drag / WASD pan | "
                     "Esc quit")
        rl.draw_text(line3, 8, 56, 16, rl.Color(*COL_TEXT_DIM))
        rl.draw_text(line4, 8, 78, 16, rl.Color(*COL_TEXT_DIM))
        if flash_frames > 0:
            flash_frames -= 1
            rl.draw_text(flash, 8, HUD_H + 6, 20, rl.Color(120, 255, 140, 255))

        rl.end_drawing()
        frames += 1
        if auto and paintable and frames == AUTO_FRAMES // 2:
            mode_paint = True          # exercise the PAINT draw path too
        if auto and frames >= AUTO_FRAMES:
            break

    for t in tex_cache.values():
        if t is not None:
            rl.unload_texture(t)
    rl.close_window()
    print(f"align_level_art: {frames} frames; final "
          f"offset_px=[{offset[0]:.1f}, {offset[1]:.1f}] "
          f"px_per_tile=[{ppt[0]:.2f}, {ppt[1]:.2f}]; "
          f"mode={'PAINT' if mode_paint else 'ALIGN'} "
          f"unsaved_paint={dirty}")

    if auto:
        _auto_paint_check(grid, csv_path)


if __name__ == "__main__":
    main()
