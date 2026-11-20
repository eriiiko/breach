"""GL checks of the water's foam (water look arc #97, patch W2;
docs/water_look_design_2026-11-19.md W2 and its critique §9).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_water_foam.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``WATER_FOAM_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a subset.

Each check names the PROPERTY it protects and the change that must break it:

  foam_dry_owns_nothing -- a DRY map: ``simulate`` issues no pass and no foam
      image is ever allocated (critique minor: lazily on the first wet
      frame); the first wet frame allocates it, once. Breaks if the image is
      allocated eagerly or a dry frame runs a pass.
  foam_step_matches_twin -- the foam image after several ticks of steps
      (``shaders/water_foam.fs``) equals ``water_foam.step`` (the twin, fed
      the same packed fp16 textures, the same step and substep counts) in
      every cell within float32 tolerance, on a scene with walls, a dry
      island, a fast varied flow, an agitated patch and newly arrived water.
      Breaks if the shader drifts from the twin (a source, the trace, the
      clamp, the kill, an axis).
  foam_draw_matches_twin -- debug view 2 writes (amount, cover) per pixel;
      the amount equals the bilinear read of the foam image and the cover
      ``water_foam.draw_cover`` within a small tolerance. Breaks if the
      draw's threshold, its two bubble sizes, the flow map's slides or the
      zoom fade drift from the twin.
  foam_frames_between_ticks_allocate_nothing -- frames that run foam steps
      without a new tick allocate (tracemalloc) under 16 KB each and keep
      the same foam image (the pass allocates nothing per frame). Breaks if
      a frame builds an array or a new PingPong.
  foam_leaving_cells_clear_both_buffers -- when the wet box shrinks, the
      cells that left it are 0 in BOTH ping-pong buffers (the partial-pass
      trap), so old foam never reappears when the water comes back. Breaks
      if the leaving rects are not cleared.
  foam_resets_on_a_backward_tick -- a sim tick that goes back (a replay
      seek) empties the foam image. Breaks if the reset is lost.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_water_foam_check.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
import tracemalloc
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

H, W, PPT = 10, 16, 48          # tiles, tiles, world px per tile
TILE_M = 1.0 / 3.0
TPS = 24.0


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
    from config import CFG
    from renderer.water_surface import water_settings
    return dataclasses.replace(water_settings(CFG), **kw)


def _fields(tick=0, wet=True, dry_east=False):
    """The int planes of a walled pool: 0.3 m, a dry island, a fast flow
    that changes tile by tile, an agitated patch, and (from tick 1 on) a
    strip of newly arrived water in the west."""
    from simulation import water_fixed
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    solid = np.zeros((H, W), bool)
    solid[0, :] = solid[-1, :] = True
    depth = np.where(solid, 0.0, 0.3)
    depth[4:6, 9:11] = 0.0                         # a dry island
    if tick == 0:
        depth[1:-1, 1:3] = 0.0                     # the strip arrives at 1
    if not wet:
        depth[:] = 0.0
    if dry_east:
        depth[:, W // 2:] = 0.0
    vx = 3.0 + 1.2 * np.sin(0.9 * xx + 0.4 * yy)
    vy = 0.9 * np.cos(0.7 * yy - 0.5 * xx)
    vx[depth <= 0] = 0.0
    vy[depth <= 0] = 0.0
    rv = np.where((xx > 11) & (yy > 5), 0.12, 0.004)
    return dict(depth_q=water_fixed.quantize(depth),
                flow_vx_q=water_fixed.quantize(vx),
                flow_vy_q=water_fixed.quantize(vy),
                solid=solid,
                ripple=np.zeros((H, W), np.float32),
                ripple_v=rv.astype(np.float32))


def _pass(**kw):
    from renderer.water import WaterPass
    return WaterPass(H, W, _settings(**kw), tile_size_m=TILE_M)


def _update(wp, tick, alpha=0.0, **fk):
    wp.update(**_fields(tick=tick, **fk), sim_tick=tick, alpha=alpha,
              tps=TPS, zoom_px_per_tile=64.0)


def _read(tex):
    from renderer import core
    return core.read_texture_rgba32f(tex)[..., 0].astype(np.float64)


def _run_ticks(wp, ticks, **fk):
    """Update + simulate tick by tick; returns the twin's image after the
    same steps from the same packed textures."""
    from renderer import water_foam as WF
    s = wp.settings
    m = s.foam_cells_per_tile
    F = np.zeros((H * m, W * m))
    for tick in ticks:
        _update(wp, tick, **fk)
        wp.simulate()
        n = wp.foam_steps_last_frame
        if n:
            inp = WF.cell_inputs(wp.packed, wp.flow_packed, m, s)
            for _ in range(n):
                F = WF.step(F, inp, wp._v_max, m, TILE_M, s)
    return F


@check("foam_dry_owns_nothing")
def check_dry():
    wp = _pass()
    for tick in range(4):
        _update(wp, tick, wet=False)
        assert wp.simulate() == 0, "a dry frame ran a foam pass"
    assert wp.foam is None, "a dry map allocated the foam image"
    _update(wp, 4)
    wp.simulate()
    _update(wp, 5)
    n = wp.simulate()
    assert wp.foam is not None and n > 0, "the wet frame ran no step"
    foam = wp.foam
    _update(wp, 6)
    wp.simulate()
    assert wp.foam is foam, "the foam image was reallocated"
    wp.unload()
    return f"dry: no pass, no image; wet: {n} passes, one image"


@check("foam_step_matches_twin")
def check_step():
    wp = _pass()
    F = _run_ticks(wp, range(0, 7))
    gpu = _read(wp.foam.front)
    err = np.abs(gpu - F)
    p99 = float(np.percentile(err, 99.0))
    assert F.max() > 0.3 and (F > 0.05).mean() > 0.2, \
        f"vacuous: little foam ({F.max():.3f}, {(F > 0.05).mean():.2f})"
    assert wp.foam_passes_last_frame > wp.foam_steps_last_frame, \
        "vacuous: no frame needed substeps"
    assert p99 < 1e-4, f"p99 |GL - twin| {p99:.2e}"
    assert float(err.mean()) < 1e-5, f"mean {err.mean():.2e}"
    assert float(err.max()) < 1e-3, f"max {err.max():.2e}"
    wp.unload()
    return (f"{err.size} cells: mean {err.mean():.1e}, p99 {p99:.1e}, max "
            f"{err.max():.1e}; foam max {F.max():.2f}, "
            f"{(F > 0.05).mean():.0%} > 0.05")


def _textures():
    from renderer import core
    diffuse = core.create_dynamic_rgba_texture(W * 8, H * 8)
    img = np.full((H * 8, W * 8, 4), 120, np.uint8)
    img[..., 3] = 255
    core.update_rgba_texture(diffuse, img)
    la = core.create_dynamic_rgba16f_texture(W, H)
    lb = core.create_dynamic_rgba16f_texture(W, H)
    pa = np.zeros((H, W, 4), np.float16)
    pb = np.zeros((H, W, 4), np.float16)
    pa[..., :3] = 2.0
    pa[..., 3] = 0.6
    pb[..., 3] = -0.3
    core.update_rgba16f_texture(la, pa)
    core.update_rgba16f_texture(lb, pb)
    return diffuse, la, lb


@check("foam_draw_matches_twin")
def check_draw():
    from renderer import core
    from renderer.heat_shimmer import bilinear_tiles
    from renderer.noise_bake import bake_foam_rgba
    from renderer.water import FOAM_TEX_SIZE
    from renderer.water_foam import draw_cover
    # big bubbles: the GPU reads the bake magnified (level 0, as the twin)
    wp = _pass(debug_view=2, foam_bubble_m=(0.6, 1.4), foam_cover=0.8)
    _run_ticks(wp, range(0, 7))
    s = wp.settings
    m = s.foam_cells_per_tile
    amt_img = _read(wp.foam.front)
    tex = _textures()
    rt = core.load_render_texture_rgba16f(W * PPT, H * PPT)
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.Color(0, 0, 0, 0))
    drew = wp.draw(tex[0], tex[1], tex[2], W * PPT, H * PPT)
    rl.end_texture_mode()
    out = core.read_texture_rgba32f(rt.texture)[::-1].astype(np.float64)
    rl.unload_render_texture(rt)
    assert drew
    yy, xx = np.mgrid[0:H * PPT, 0:W * PPT]
    xt = (xx + 0.5) / PPT
    yt = (yy + 0.5) / PPT
    wpk = wp.packed.astype(np.float64)
    fp = wp.flow_packed.astype(np.float64)
    wet = bilinear_tiles(wpk[..., 2], xt, yt) > 0
    amount = bilinear_tiles(amt_img, xt * m, yt * m)
    e_amt = np.abs(out[..., 0] - amount)[wet]
    v = np.stack([bilinear_tiles(fp[..., 0], xt, yt),
                  bilinear_tiles(fp[..., 1], xt, yt)], -1)
    depth = bilinear_tiles(wpk[..., 2], xt, yt)
    twin = draw_cover(xt * TILE_M, yt * TILE_M, v, depth, amount, wp.frame,
                      bake_foam_rgba(FOAM_TEX_SIZE), s)
    e_cov = np.abs(out[..., 1] - twin)[wet]
    lace = ((twin > 0.05) & (twin < 0.95))[wet].mean()
    assert lace > 0.05 and (twin[wet] > 0.95).mean() > 0.05, \
        "vacuous: no lace in the picture"
    assert float(np.percentile(e_amt, 99)) < 0.01, e_amt.max()
    assert float(e_cov.mean()) < 0.01, e_cov.mean()
    assert float(np.percentile(e_cov, 99)) < 0.1, \
        float(np.percentile(e_cov, 99))
    wp.unload()
    for t in tex:
        rl.unload_texture(t)
    return (f"{e_cov.size} wet px: amount p99 {np.percentile(e_amt, 99):.1e}; "
            f"cover mean {e_cov.mean():.1e}, p99 "
            f"{np.percentile(e_cov, 99):.1e}; lace {lace:.0%}")


@check("foam_frames_between_ticks_allocate_nothing")
def check_alloc():
    wp = _pass()
    _run_ticks(wp, range(0, 4))
    foam, ids = wp.foam, tuple(wp.foam._ids)
    fields = _fields(tick=3)
    tracemalloc.start()
    peaks = []
    passes = 0
    for k in range(1, 30):
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        wp.update(**fields, sim_tick=3, alpha=k / 30.0, tps=TPS,
                  zoom_px_per_tile=64.0)
        passes += wp.simulate()
        peaks.append(tracemalloc.get_traced_memory()[1] - base)
    tracemalloc.stop()
    assert passes > 0, "vacuous: no frame stepped"
    assert wp.foam is foam and tuple(wp.foam._ids) == ids
    assert max(peaks) < 16 * 1024, f"a frame allocated {max(peaks)} B"
    wp.unload()
    return f"{passes} passes over 29 frames, peak {max(peaks)} B a frame"


@check("foam_leaving_cells_clear_both_buffers")
def check_leaving():
    wp = _pass()
    _run_ticks(wp, range(0, 6))
    m = wp.settings.foam_cells_per_tile
    assert _read(wp.foam.front)[:, (W // 2 + 1) * m:].max() > 0.05, \
        "vacuous: no foam in the east"
    for tick in (6, 7, 8):
        _update(wp, tick, dry_east=True)
        wp.simulate()
    x0, y0, x1, y1 = wp._foam_box
    for rt in wp.foam._rts:
        img = _read(rt.texture)
        outside = np.ones(img.shape, bool)
        outside[y0:y1, x0:x1] = False
        assert img[outside].max() == 0.0, "a buffer kept foam outside the box"
    wp.unload()
    return f"box {wp._foam_box}: both buffers 0 outside"


@check("foam_resets_on_a_backward_tick")
def check_reset():
    wp = _pass()
    _run_ticks(wp, range(0, 6))
    assert _read(wp.foam.front).max() > 0.05
    _update(wp, 2)
    wp.simulate()
    img = _read(wp.foam.front)
    assert img.max() == 0.0, f"foam survived the seek ({img.max():.3f})"
    wp.unload()
    return "empty after the seek"


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_water_foam_check")
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
    print(f"WATER_FOAM_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
