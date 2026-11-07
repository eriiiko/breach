"""GL checks of the shock front, the resolve's second distortion term
(graphics arc G4, patch P2; docs/explosion_look_design_2026-11-07.md §3.3,
§4 and the §6 critique's S8).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_shockwave.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``SHOCKWAVE_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a subset.

Each check names the PROPERTY it protects and the change that must break it:

  shock_zero_off_the_band -- with no ring the resolve with the term is
      BIT-IDENTICAL to the resolve without it; with a live ring every pixel
      whose offset the twin puts at exactly zero (outside the band, behind
      the wall) is bit-identical too, and the band itself does change.
      Breaks if the term fetches or shifts where it should not (a leak past
      the band or through a wall, a non-zero offset with no ring).
  shock_matches_the_twin -- ONE ring over a synthetic checker scene: the
      resolved image equals the PLAIN resolve of the checker resampled on
      the CPU through the twin (``ring_offset_px``), within 1/255 per
      channel over the band. Breaks if the shader drifts from the spec, the
      y axis is flipped (the critique's S8), the slot/channel mapping or the
      plane upload is off.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_shockwave_check.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pyray as rl  # noqa: E402

RESULTS = []
CHECKS = []

H, W, PPT = 16, 20, 16            # tiles, tiles, world px per tile


def check(name):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                detail = fn() or ""
                RESULTS.append((name, True,
                                f"{detail} [{time.perf_counter() - t0:.1f}s]"))
            except Exception as e:                       # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}\n"
                                + traceback.format_exc()))
        run.__name__ = name
        CHECKS.append((name, run))
        return run
    return deco


def _settings(**kw):
    from renderer.shockwave import ShockwaveSettings
    base = dict(enabled=True, speed_tiles_s=40.0, width_tiles=1.0,
                amp_tiles=0.4, strength_power=0.5, life_s=0.5, max_rings=4,
                zoom_zero=0.0,
                zoom_full=1e-3)
    base.update(kw)
    return ShockwaveSettings(**base)


def _mask():
    """A room with a wall stub at x = 13 (rows 2..13), open elsewhere."""
    m = np.zeros((H, W), bool)
    m[1:-1, 1:-1] = True
    m[2:14, 13] = False
    return m


def _resolve(wc, tex, shock=None):
    """Draw ``tex`` as the scene, resolve (with ``shock``), read back."""
    wc.begin(clear_color=(0, 0, 0, 0))
    rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
    rl.draw_texture(tex, 0, 0, rl.WHITE)
    rl.end_blend_mode()
    wc.resolve(shock=shock)
    wc.end()
    img = rl.load_image_from_texture(wc.rt.texture)
    n = img.width * img.height * 4
    a = np.frombuffer(rl.ffi.buffer(img.data, n), np.uint8).reshape(
        img.height, img.width, 4).copy()
    rl.unload_image(img)
    return a


def _checker():
    """A world-px checker (row 0 = world y 0, north), 5 px cells, two
    linear levels, opaque, plus a slow ramp so no two cells are alike."""
    hp, wp = H * PPT, W * PPT
    yy, xx = np.mgrid[0:hp, 0:wp]
    c = (((yy // 5) + (xx // 5)) % 2).astype(np.float32)
    img = np.zeros((hp, wp, 4), np.float32)
    img[..., 0] = 0.15 + 0.45 * c
    img[..., 1] = 0.10 + 0.30 * c + 0.2 * (yy / hp)
    img[..., 2] = 0.05 + 0.25 * (1 - c) + 0.2 * (xx / wp)
    img[..., 3] = 1.0
    return img


def _upload(img):
    from renderer import core
    tex = core.create_dynamic_rgba32f_texture(img.shape[1], img.shape[0])
    core.update_rgba32f_texture(tex, img)
    return tex


def _start(sw, mask, pos=(7, 5), strength=1.0, tick=2):
    sw.update(effects=[{"kind": "blast", "pos": pos, "strength": strength,
                        "tick0": 0}],
              mask_fn=lambda: mask, sim_tick=tick, alpha=0.0, paused=False,
              tps=24.0, zoom_px_per_tile=64.0)


def _twin_offsets(sw):
    """The twin's offset (world px, y south) at every world pixel centre."""
    from renderer.shockwave import band_box, ring_offset_px
    rings = sw.set.twin_rings(float(PPT), 64.0)
    box = band_box([r[0] for r in rings], [r[3] for r in rings],
                   sw.settings.width_tiles)
    yy, xx = np.mgrid[0:H * PPT, 0:W * PPT]
    return ring_offset_px((xx + 0.5) / PPT, (yy + 0.5) / PPT, rings,
                          sw.settings.width_tiles, box=box)


def _bilinear_px(img, qx, qy):
    """The resolve's ``scene_bilinear`` in world px (centres at .5, CLAMP)."""
    hp, wp = img.shape[:2]
    fx, fy = qx - 0.5, qy - 0.5
    x0, y0 = np.floor(fx), np.floor(fy)
    ax, ay = (fx - x0)[..., None], (fy - y0)[..., None]
    x0i = np.clip(x0.astype(np.int64), 0, wp - 1)
    x1i = np.clip(x0.astype(np.int64) + 1, 0, wp - 1)
    y0i = np.clip(y0.astype(np.int64), 0, hp - 1)
    y1i = np.clip(y0.astype(np.int64) + 1, 0, hp - 1)
    top = img[y0i, x0i] * (1 - ax) + img[y0i, x1i] * ax
    bot = img[y1i, x0i] * (1 - ax) + img[y1i, x1i] * ax
    return top * (1 - ay) + bot * ay


@check("shock_zero_off_the_band")
def check_zero():
    from renderer.shockwave import Shockwave
    from renderer.world_composite import WorldComposite
    wc = WorldComposite(W, H, PPT)
    sw = Shockwave(H, W, _settings())
    img = _checker()
    tex = _upload(img)
    plain = _resolve(wc, tex, None)
    idle = _resolve(wc, tex, sw)
    assert not sw.active(PPT)
    assert np.array_equal(plain, idle), "no ring, yet the resolve changed"
    _start(sw, _mask())
    assert sw.active(PPT)
    ring = _resolve(wc, tex, sw)
    off = _twin_offsets(sw)
    zero = np.linalg.norm(off, axis=-1) == 0.0
    # readback rows: the display RT's GL rows run up (world row = H*PPT-1-r)
    zero_gl = zero[::-1]
    same = np.all(plain == ring, axis=-1)
    assert np.all(same[zero_gl]), (
        f"{int((~same & zero_gl).sum())} px changed where the twin's offset "
        "is exactly zero")
    changed = int((~same).sum())
    assert changed > 200, f"the ring moved only {changed} px"
    sw.unload()
    rl.unload_texture(tex)
    wc.unload()
    return (f"idle bit-identical; ring changed {changed} px, none where the "
            f"twin is zero ({int(zero.sum())} px)")


@check("shock_matches_the_twin")
def check_twin():
    from renderer.shockwave import Shockwave
    from renderer.world_composite import WorldComposite
    wc = WorldComposite(W, H, PPT)
    sw = Shockwave(H, W, _settings())
    img = _checker()
    tex = _upload(img)
    _start(sw, _mask())
    ring = _resolve(wc, tex, sw)
    off = _twin_offsets(sw)
    yy, xx = np.mgrid[0:H * PPT, 0:W * PPT]
    resampled = _bilinear_px(img.astype(np.float64), xx + 0.5 + off[..., 0],
                             yy + 0.5 + off[..., 1]).astype(np.float32)
    resampled[..., 3] = 1.0
    tex2 = _upload(np.ascontiguousarray(resampled))
    expect = _resolve(wc, tex2, None)
    band = (np.linalg.norm(off, axis=-1) > 0.0)[::-1]
    moved = (np.linalg.norm(off, axis=-1) > 1.0)[::-1]
    diff = np.abs(ring.astype(int) - expect.astype(int))[..., :3].max(axis=-1)
    worst = int(diff[band].max())
    bad = int((diff[band] > 1).sum())
    # the twin's resample really differs from the unshifted checker there
    plain = _resolve(wc, tex, None)
    shifted = int((np.abs(plain.astype(int) - expect.astype(int))[..., :3]
                   .max(axis=-1)[moved] > 1).sum())
    assert band.sum() > 500 and shifted > 200, (band.sum(), shifted)
    assert worst <= 1, (f"GL vs twin: {bad} band px differ by more than "
                        f"1/255 (worst {worst}/255)")
    for t in (tex, tex2):
        rl.unload_texture(t)
    sw.unload()
    wc.unload()
    return (f"{int(band.sum())} band px within {worst}/255 of the twin "
            f"({shifted} of them visibly shifted)")


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_shockwave_check")
    try:
        for name, fn in CHECKS:
            if only is None or name in only:
                fn()
    finally:
        if rl.is_window_ready():
            rl.close_window()
    ok = all(r[1] for r in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"[check] {name}: {'PASS' if passed else 'FAIL'} {detail}")
    print(f"SHOCKWAVE_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
