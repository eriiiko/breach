"""tools/align_level_art.py — interactive art<->grid alignment tool (F4 ALIGN
slice, pulled forward as a standalone viewer).

Loads a level via level_loader, draws the BARE diffuse in art-pixel space and
overlays the tilemap through the live [art.align] transform (offset_px +
per-axis px_per_tile). What you align here is exactly what the lighting
shader will sample: the overlay rect for tile (tx, ty) is
``tile_to_art_px(tx, ty, offset_px, px_per_tile)`` — the same single source
of truth the renderer's src-rect math uses (renderer/lighting.py
art_src_and_uv_rect). Ctrl+S rewrites ONLY the [art.align] lines of the
level's level.toml (a .bak of the original bytes is written first).

Run:
    C:/Users/steen/anaconda3/python.exe tools/align_level_art.py [level_name] [--auto]

level_name defaults to unhcr_vessel_2. --auto closes after ~90 frames
(smoke-test plumbing, mirrors tests/test_main_smoke.py).

Controls:
    Mouse wheel          - zoom (around cursor)
    WASD / mouse drag    - pan
    Arrow keys           - offset_px +-1 px      (Shift = +-10)
    X / Y                - select the axis that +/- scales
    + / -                - px_per_tile[active axis] +-0.1   (Shift = +-1.0)
                           (main-row =/- and keypad +/- both work)
    G                    - toggle SPACE (vacuum) blue tint
    L                    - toggle grid lines
    R                    - reset to the values in level.toml (or last save)
    Ctrl+S               - SAVE: rewrite only the [art.align] lines in place
    Esc                  - quit
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Make project modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyray as rl

from level_loader import load as load_level, materials_from_tilemap, tile_to_art_px
from simulation.materials import MAT_DOOR, MAT_HULL


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
# Interactive viewer
# ---------------------------------------------------------------------------

WIN_W, WIN_H = 1280, 920
AUTO_FRAMES = 90          # --auto: close after ~90 frames (smoke test)

COL_WALL = (255, 60, 50, 110)      # hull/door fill — semi-transparent red
COL_SPACE = (60, 120, 255, 55)     # SPACE (vacuum) tint — blue
COL_GRID = (255, 255, 255, 40)     # thin grid lines
COL_HUD_BG = (0, 0, 0, 170)


def _pressed(key) -> bool:
    """Key pressed this frame, with OS key-repeat while held."""
    return rl.is_key_pressed(key) or rl.is_key_pressed_repeat(key)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    auto = "--auto" in sys.argv
    level_name = args[0] if args else "unhcr_vessel_2"

    lvl = load_level(level_name)
    grid_h, grid_w = lvl.tilemap.shape
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    wall_tiles = [(int(x), int(y)) for y, x in
                  np.argwhere((mat == MAT_HULL) | (mat == MAT_DOOR))]
    space_tiles = [(int(x), int(y)) for y, x in np.argwhere(vac)]

    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(WIN_W, WIN_H, f"Breach ALIGN — {lvl.name} ({level_name})")
    rl.set_target_fps(60)

    tex = rl.load_texture(str(lvl.diffuse_path))
    if tex.id == 0:
        rl.close_window()
        raise SystemExit(f"Could not load diffuse texture: {lvl.diffuse_path}")
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
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
    show_space = False                       # G
    show_grid = True                         # L
    flash, flash_frames = "", 0

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

        # ---- view: wheel zoom around cursor, WASD/drag pan ---------------
        wheel = rl.get_mouse_wheel_move()
        if wheel != 0.0:
            m = rl.get_mouse_position()
            art_mx, art_my = cam_x + m.x / zoom, cam_y + m.y / zoom
            zoom = max(0.02, min(50.0, zoom * (1.1 ** wheel)))
            cam_x, cam_y = art_mx - m.x / zoom, art_my - m.y / zoom
        if not ctrl:                       # keep Ctrl+S clear of S-pan
            pan = 900.0 * dt / zoom        # screen px/s -> art px
            if rl.is_key_down(rl.KeyboardKey.KEY_W):
                cam_y -= pan
            if rl.is_key_down(rl.KeyboardKey.KEY_S):
                cam_y += pan
            if rl.is_key_down(rl.KeyboardKey.KEY_A):
                cam_x -= pan
            if rl.is_key_down(rl.KeyboardKey.KEY_D):
                cam_x += pan
        if (rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
                or rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_MIDDLE)):
            d = rl.get_mouse_delta()
            cam_x -= d.x / zoom
            cam_y -= d.y / zoom

        # ---- align edits ---------------------------------------------------
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
        if rl.is_key_pressed(rl.KeyboardKey.KEY_G):
            show_space = not show_space
        if rl.is_key_pressed(rl.KeyboardKey.KEY_L):
            show_grid = not show_grid
        if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
            offset, ppt = list(baseline[0]), list(baseline[1])
            flash, flash_frames = "reset to level.toml values", 120
        if ctrl and rl.is_key_pressed(rl.KeyboardKey.KEY_S):
            bak = save_align(lvl.path / "level.toml", offset, ppt)
            baseline = (list(offset), list(ppt))
            flash, flash_frames = f"SAVED [art.align] (backup: {bak.name})", 180

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
        # the wall rect for (tx, ty) sits exactly where the shader samples it.
        tw, th = ppt[0] * zoom, ppt[1] * zoom

        def tile_rect(tx: int, ty: int):
            ax, ay = tile_to_art_px(tx, ty, offset, ppt)
            sx, sy = to_screen(ax, ay)
            if sx + tw < 0 or sy + th < 0 or sx > win_w or sy > win_h:
                return None
            return sx, sy

        for tx, ty in wall_tiles:
            r = tile_rect(tx, ty)
            if r:
                rl.draw_rectangle(int(r[0]), int(r[1]),
                                  max(1, int(tw)), max(1, int(th)), COL_WALL)
        if show_space:
            for tx, ty in space_tiles:
                r = tile_rect(tx, ty)
                if r:
                    rl.draw_rectangle(int(r[0]), int(r[1]),
                                      max(1, int(tw)), max(1, int(th)),
                                      COL_SPACE)
        if show_grid:
            for tx in range(grid_w + 1):
                a0 = to_screen(*tile_to_art_px(tx, 0, offset, ppt))
                a1 = to_screen(*tile_to_art_px(tx, grid_h, offset, ppt))
                rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 1.0, COL_GRID)
            for ty in range(grid_h + 1):
                a0 = to_screen(*tile_to_art_px(0, ty, offset, ppt))
                a1 = to_screen(*tile_to_art_px(grid_w, ty, offset, ppt))
                rl.draw_line_ex(rl.Vector2(*a0), rl.Vector2(*a1), 1.0, COL_GRID)

        # ---- HUD ------------------------------------------------------------
        rl.draw_rectangle(0, 0, win_w, 78, rl.Color(*COL_HUD_BG))
        rl.draw_text(
            f"{level_name}  art {int(art_w)}x{int(art_h)} px  "
            f"grid {grid_w}x{grid_h} tiles  zoom {zoom:.2f}", 8, 6, 18,
            rl.Color(200, 200, 200, 255))
        rl.draw_text(
            f"offset_px = [{offset[0]:.1f}, {offset[1]:.1f}]   "
            f"px_per_tile = [{ppt[0]:.2f}, {ppt[1]:.2f}]   "
            f"axis: {'X' if active_axis == 0 else 'Y'}", 8, 28, 20,
            rl.Color(255, 230, 120, 255))
        rl.draw_text(
            "arrows offset (Shift x10) | X/Y axis | +/- scale (Shift x10) | "
            "G space | L grid | R reset | Ctrl+S save", 8, 54, 16,
            rl.Color(150, 150, 160, 255))
        if flash_frames > 0:
            flash_frames -= 1
            rl.draw_text(flash, 8, 84, 20, rl.Color(120, 255, 140, 255))

        rl.end_drawing()
        frames += 1
        if auto and frames >= AUTO_FRAMES:
            break

    rl.unload_texture(tex)
    rl.close_window()
    print(f"align_level_art: {frames} frames; final "
          f"offset_px=[{offset[0]:.1f}, {offset[1]:.1f}] "
          f"px_per_tile=[{ppt[0]:.2f}, {ppt[1]:.2f}]")


if __name__ == "__main__":
    main()
