"""tools/view_smoke_volume.py — play a baked smoke volume as a black-and-white
movie (issue #90). The bake's own eyes: no sim, no lighting, no tau — just the
``(T, Z, Y, X, 2)`` uint8 field from ``assets/volumes/<name>.npz``.

Run (conda env `data`):
    python tools/view_smoke_volume.py                 # assets/volumes/smoke_v1.npz
    python tools/view_smoke_volume.py smoke_hd        # another asset by name
    python tools/view_smoke_volume.py --file path.npz
    python tools/view_smoke_volume.py --auto --shot out.png   # 60 frames, screenshot, exit

Keys:
    SPACE      pause / resume the loop
    Z / X      z-slice down / up (slice mode)
    M          mode: slice -> mean-over-z -> max-over-z
    C          channel: smoke <-> fire
    T          tile 2x2 (shows that the wrap is seamless)
    - / +      slower / faster
    1..4       display scale (texels -> screen px)
    ESC        quit

The movie is drawn with bilinear filtering and frames are lerped in time —
the SAME reconstruction the raymarcher does — so what you see here is what
the game samples, minus the sim's macro shape.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyray as rl

from renderer import core
from renderer.volume_bake import load_volume, procedural_volume

ASSET_DIR = ROOT / "assets" / "volumes"
MODES = ("slice", "mean", "max")


def _parse() -> dict:
    a = sys.argv[1:]
    out = {"name": "smoke_v1", "file": None, "auto": "--auto" in a,
           "shot": None, "frames": 60}
    if "--file" in a:
        out["file"] = a[a.index("--file") + 1]
    if "--shot" in a:
        i = a.index("--shot")
        out["shot"] = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else "volume_shot.png"
        out["auto"] = True
    if "--frames" in a:
        out["frames"] = int(a[a.index("--frames") + 1])
    pos = [s for s in a if not s.startswith("--") and s not in
           (out["file"] or "", out["shot"] or "", str(out["frames"]))]
    if pos:
        out["name"] = pos[0]
    return out


def frame_image(vol: np.ndarray, t: float, z: int, mode: str, ch: int,
                tile2: bool) -> np.ndarray:
    """(H, W, 4) uint8 grey image of the movie at fractional frame ``t``
    (lerped between the two neighbouring frames, wrapping), one z-slice or a
    z-projection. Pure numpy (headless-testable)."""
    T = vol.shape[0]
    f0 = int(np.floor(t)) % T
    f1 = (f0 + 1) % T
    w = float(t - np.floor(t))
    a = vol[f0, ..., ch].astype(np.float32)
    b = vol[f1, ..., ch].astype(np.float32)
    v = a * (1.0 - w) + b * w                       # (Z, Y, X)
    if mode == "slice":
        img = v[z % v.shape[0]]
    elif mode == "mean":
        img = v.mean(axis=0)
    else:
        img = v.max(axis=0)
    if tile2:
        img = np.tile(img, (2, 2))
    g = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    out = np.empty(g.shape + (4,), dtype=np.uint8)
    out[..., 0] = g
    out[..., 1] = g
    out[..., 2] = g
    out[..., 3] = 255
    return out


def main() -> int:
    args = _parse()
    if args["file"]:
        path = Path(args["file"])
    else:
        path = ASSET_DIR / f"{args['name']}.npz"
    if path.is_file():
        vol, meta = load_volume(path)
        title = path.name
    else:
        print(f"[view] no asset at {path} — showing the procedural fallback")
        vol = procedural_volume()
        meta = {"loop_seconds": 4.0, "period_tiles": 4.0, "height_tiles": 1.0}
        title = "<procedural>"
    T, Z, Y, X, _ = vol.shape
    loop_s = float(meta["loop_seconds"])
    print(f"[view] {title}: {X}x{Y}x{Z} x {T} frames, loop {loop_s} s, "
          f"period {meta['period_tiles']} tiles, height {meta['height_tiles']} tiles")

    scale = 4 if X <= 128 else 2
    z = Z // 2
    mode_i = 0
    ch = 0
    tile2 = False
    paused = False
    speed = 1.0
    t_movie = 0.0

    core.init_window(min(X * scale, 1600), min(Y * scale, 1000) + 28,
                     title=f"smoke volume — {title}", borderless=False)
    rl.set_target_fps(60)
    tex = core.create_dynamic_rgba_texture(X * 2, Y * 2)   # room for tile 2x2
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    last = time.perf_counter()
    frames = 0
    try:
        while not rl.window_should_close():
            now = time.perf_counter()
            dt = now - last
            last = now
            if not paused:
                t_movie = (t_movie + dt * speed * T / max(loop_s, 1e-3)) % T
            K = rl.KeyboardKey
            if rl.is_key_pressed(K.KEY_SPACE):
                paused = not paused
            if rl.is_key_pressed(K.KEY_Z):
                z = (z - 1) % Z
            if rl.is_key_pressed(K.KEY_X):
                z = (z + 1) % Z
            if rl.is_key_pressed(K.KEY_M):
                mode_i = (mode_i + 1) % len(MODES)
            if rl.is_key_pressed(K.KEY_C):
                ch = 1 - ch
            if rl.is_key_pressed(K.KEY_T):
                tile2 = not tile2
            if rl.is_key_pressed(K.KEY_MINUS) or rl.is_key_pressed(K.KEY_KP_SUBTRACT):
                speed = max(0.125, speed * 0.5)
            if rl.is_key_pressed(K.KEY_EQUAL) or rl.is_key_pressed(K.KEY_KP_ADD):
                speed = min(8.0, speed * 2.0)
            for key, sc in ((K.KEY_ONE, 1), (K.KEY_TWO, 2), (K.KEY_THREE, 3),
                            (K.KEY_FOUR, 4)):
                if rl.is_key_pressed(key):
                    scale = sc

            img = frame_image(vol, t_movie, z, MODES[mode_i], ch, tile2)
            h, w = img.shape[:2]
            full = np.zeros((Y * 2, X * 2, 4), dtype=np.uint8)
            full[:h, :w] = img
            core.update_rgba_texture(tex, full)

            rl.begin_drawing()
            rl.clear_background(rl.Color(12, 12, 14, 255))
            src = rl.Rectangle(0, 0, float(w), float(h))
            dst = rl.Rectangle(0, 28, float(w * scale), float(h * scale))
            rl.draw_texture_pro(tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
            hud = (f"{title}  {X}x{Y}x{Z} x{T}  mode {MODES[mode_i]}"
                   f"{'' if MODES[mode_i] != 'slice' else f' z={z}'}  "
                   f"ch {'smoke' if ch == 0 else 'fire'}  t {t_movie:5.1f}/{T}  "
                   f"x{speed:g}{'  PAUSED' if paused else ''}  [SPACE Z X M C T -/+ 1-4]")
            rl.draw_text(hud, 8, 6, 16, rl.Color(200, 220, 200, 255))
            rl.end_drawing()
            frames += 1
            if args["auto"] and frames >= args["frames"]:
                if args["shot"]:
                    rl.take_screenshot(args["shot"])
                break
    finally:
        rl.unload_texture(tex)
        core.shutdown()
    if args["auto"]:
        print(f"OK — view_smoke_volume rendered {frames} frames (--auto)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
