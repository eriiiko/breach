"""GL checks of the water pass's fine surface that flows (water look arc #97,
patch W1; docs/water_look_design_2026-11-19.md W1 and its critique §9).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_water_flow.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``WATER_FLOW_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a subset.

Each check names the PROPERTY it protects and the change that must break it:

  water_shader_compiles -- ``shaders/water.fs`` with its two splices (lit3d's
      smoke shadow + light curve, advected_noise's crossfade) compiles into
      its own program (not raylib's default fallback). Breaks on a GLSL error
      or a lost splice marker.
  water_dry_draws_nothing -- a DRY map (critique M4): ``update`` dequantizes
      nothing, ``draw`` draws nothing and the world RT is bit-identical to
      the RT before the draw; a wet map then does draw. Breaks if the dry
      fast path is lost (a dequantize before ``.any()``, a full-RT quad).
  water_detail_matches_twin -- debug view 1 writes the fine surface's slope
      per pixel; it equals ``water_surface.detail_slope`` (the numpy twin,
      fed the same packed fp16 textures, the same frame uniforms and the same
      ripple bake) at every pixel centre within a small tolerance, on a flow
      that changes tile by tile. Breaks if the shader drifts from the twin
      (the hex lattice, the hash, the rotation, the slide, the cap, the
      crossfade, the strength) or a y axis is flipped.
  water_picture_is_finite -- the full picture over a lit, flowing pool
      (light curve on, no smoke product) is finite everywhere, with alpha in
      the ``[alpha_min, alpha_max]`` ramp. Breaks on a NaN (a pow of a
      negative weight, a zero-length normalisation).

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_water_flow_check.py
"""
from __future__ import annotations

import dataclasses
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

H, W, PPT = 10, 12, 48          # tiles, tiles, world px per tile
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


def _fields(wet=True):
    """The int planes of a pool: 0.3 m everywhere (no wet/dry edge in the
    compared field), a flow that changes tile by tile."""
    from simulation import water_fixed
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = np.full((H, W), 0.3 if wet else 0.0)
    vx = 0.5 + 0.45 * np.sin(0.9 * xx + 0.4 * yy)
    vy = -0.2 + 0.35 * np.cos(0.7 * yy - 0.5 * xx)
    if not wet:
        vx[:] = 0.0
        vy[:] = 0.0
    rv = 0.02 * np.sin(1.3 * xx) * np.cos(0.8 * yy)
    return dict(depth_q=water_fixed.quantize(depth),
                flow_vx_q=water_fixed.quantize(vx),
                flow_vy_q=water_fixed.quantize(vy),
                solid=np.zeros((H, W), bool),
                ripple=np.zeros((H, W), np.float32),
                ripple_v=rv.astype(np.float32))


def _textures():
    from renderer import core
    diffuse = core.create_dynamic_rgba_texture(W * 8, H * 8)
    yy, xx = np.mgrid[0:H * 8, 0:W * 8]
    c = (((yy // 4) + (xx // 4)) % 2).astype(np.uint8)
    img = np.zeros((H * 8, W * 8, 4), np.uint8)
    img[..., 0] = 60 + 120 * c
    img[..., 1] = 80 + 90 * c
    img[..., 2] = 70 + 60 * (1 - c)
    img[..., 3] = 255
    core.update_rgba_texture(diffuse, img)
    la = core.create_dynamic_rgba16f_texture(W, H)
    lb = core.create_dynamic_rgba16f_texture(W, H)
    pa = np.zeros((H, W, 4), np.float16)
    pb = np.zeros((H, W, 4), np.float16)
    pa[..., :3] = 2.0
    pa[..., 3] = 0.6                      # light from the east-ish
    pb[..., 3] = -0.3
    core.update_rgba16f_texture(la, pa)
    core.update_rgba16f_texture(lb, pb)
    return diffuse, la, lb


def _render(wp, tex, prefill=None, **draw_kw):
    """Draw the pass into a fresh RGBA16F RT (optionally over ``prefill``,
    a texture drawn first) and read it back, world row 0 first."""
    from renderer import core
    rt = core.load_render_texture_rgba16f(W * PPT, H * PPT)
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.Color(0, 0, 0, 0))
    if prefill is not None:
        rl.draw_texture_pro(prefill, rl.Rectangle(0, 0, prefill.width,
                                                  prefill.height),
                            rl.Rectangle(0, 0, W * PPT, H * PPT),
                            rl.Vector2(0, 0), 0.0, rl.WHITE)
    before = None
    rl.end_texture_mode()
    before = core.read_texture_rgba32f(rt.texture)[::-1].copy()
    rl.begin_texture_mode(rt)
    drew = wp.draw(tex[0], tex[1], tex[2], W * PPT, H * PPT, **draw_kw)
    rl.end_texture_mode()
    out = core.read_texture_rgba32f(rt.texture)[::-1].copy()
    rl.unload_render_texture(rt)
    return drew, before, out


@check("water_shader_compiles")
def check_compiles():
    from renderer.water import WaterPass, load_water_fs
    src = load_water_fs()
    assert "// <SMOKE_SHADOW>" not in src and "// <ADVECTION>" not in src
    wp = WaterPass(H, W, _settings(), tile_size_m=TILE_M)
    default_id = rl.rl_get_shader_id_default()
    assert wp.shader.id != default_id, "water.fs fell back to the default"
    assert wp.ripple_tex.mipmaps > 1, "the ripple bake has no mipmaps"
    wp.unload()
    return f"program {wp.shader.id}, ripple mips {wp.ripple_tex.mipmaps}"


@check("water_dry_draws_nothing")
def check_dry():
    import simulation.water_fixed as wf
    from renderer.water import WaterPass
    tex = _textures()
    wp = WaterPass(H, W, _settings(), tile_size_m=TILE_M)
    calls = {"n": 0}
    orig = wf.dequantize_f32

    def counting(q):
        calls["n"] += 1
        return orig(q)
    wf.dequantize_f32 = counting
    try:
        for tick in range(3):
            wp.update(**_fields(wet=False), sim_tick=tick, alpha=0.0,
                      tps=TPS, zoom_px_per_tile=64.0)
    finally:
        wf.dequantize_f32 = orig
    assert calls["n"] == 0, f"a dry map dequantized {calls['n']} planes"
    assert not wp.wet and wp.box is None
    drew, before, out = _render(wp, tex, prefill=tex[0])
    assert not drew, "a dry map drew"
    assert np.array_equal(before, out), "the dry draw changed the RT"
    wp.update(**_fields(wet=True), sim_tick=3, alpha=0.0, tps=TPS,
              zoom_px_per_tile=64.0)
    drew, before, out = _render(wp, tex, prefill=tex[0])
    assert drew and not np.array_equal(before, out), "a wet map drew nothing"
    wp.unload()
    for t in tex:
        rl.unload_texture(t)
    return "dry: 0 dequantizes, no draw, RT bit-identical; wet draws"


@check("water_detail_matches_twin")
def check_twin():
    from renderer.heat_shimmer import bilinear_tiles
    from renderer.noise_bake import bake_ripple_rgba
    from renderer.water import RIPPLE_SIZE, WaterPass
    from renderer.water_surface import detail_slope
    tex = _textures()
    s = _settings(debug_view=1)
    wp = WaterPass(H, W, s, tile_size_m=TILE_M)
    wp.update(**_fields(), sim_tick=1234, alpha=0.37, tps=TPS,
              zoom_px_per_tile=64.0)
    drew, _before, out = _render(wp, tex)
    assert drew
    yy, xx = np.mgrid[0:H * PPT, 0:W * PPT]
    xt = (xx + 0.5) / PPT
    yt = (yy + 0.5) / PPT
    fp = wp.flow_packed.astype(np.float64)
    wpk = wp.packed.astype(np.float64)
    v = np.stack([bilinear_tiles(fp[..., 0], xt, yt),
                  bilinear_tiles(fp[..., 1], xt, yt)], -1)
    depth = bilinear_tiles(wpk[..., 2], xt, yt)
    agit = np.abs(bilinear_tiles(wpk[..., 1], xt, yt))
    twin = detail_slope(xt * TILE_M, yt * TILE_M, v, depth, agit, wp.frame,
                        bake_ripple_rgba(RIPPLE_SIZE), s)
    gpu = out[..., :2].astype(np.float64)
    k = float(np.median(out[..., 2]))
    err = np.sqrt(((gpu - twin) ** 2).sum(-1))
    rms = float(np.sqrt((twin ** 2).sum(-1).mean()))
    p99 = float(np.percentile(err, 99.0))
    bad = float((err > 0.1 * k).mean())
    moving = float(np.sqrt((v ** 2).sum(-1)).max())
    assert rms > 0.3 * k, f"vacuous: the twin's surface is flat ({rms})"
    assert moving > 0.5
    # GPU bilinear weights carry 8 fractional bits: measured p99 ~1.7 % of k
    assert p99 < 0.04 * k, f"GL vs twin: p99 error {p99:.2e} (k {k:.3f})"
    assert bad < 1e-3, f"{bad:.2%} of pixels off by more than 10% of k"
    wp.unload()
    for t in tex:
        rl.unload_texture(t)
    return (f"{err.size} px: p99 |err| {p99:.1e} = {p99 / k:.2%} of k "
            f"(rms slope {rms:.3f}, max err {err.max():.1e}, "
            f"{bad:.3%} > 10% k)")


@check("water_picture_is_finite")
def check_finite():
    from renderer.light_curve import bake_lut, light_curve_settings
    from config import CFG
    from renderer.water import WaterPass
    tex = _textures()
    s = _settings()
    wp = WaterPass(H, W, s, tile_size_m=TILE_M)
    wp.update(**_fields(), sim_tick=99, alpha=0.5, tps=TPS,
              zoom_px_per_tile=64.0)
    curve = light_curve_settings(CFG)
    lut = rl.ffi.new("float[256]", bake_lut(curve.points).tolist())
    uniforms = (lut, (1.0, curve.range, curve.dark_saturation,
                      curve.dark_knee))
    drew, _b, out = _render(wp, tex, light_curve=uniforms, light_gain=1.0)
    assert drew
    assert np.isfinite(out).all(), "non-finite water pixels"
    a = out[..., 3]
    assert a.min() >= s.alpha_min - 1e-3 and a.max() <= s.alpha_max + 1e-3, \
        (a.min(), a.max())
    wp.unload()
    for t in tex:
        rl.unload_texture(t)
    return f"alpha {a.min():.3f}..{a.max():.3f}, rgb max {out[..., :3].max():.2f}"


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_water_flow_check")
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
    print(f"WATER_FLOW_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
