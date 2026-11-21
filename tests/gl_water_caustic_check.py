"""GL checks of the water pass's refraction map -- THE caustic in Wallace's
scatter form, the Snell floor bend -- and the lamps' own glints (water look
arc #97, W4 in its landed form of 2026-11-21; Erik: "I'd like scatter,
Snell, lamp glint"; docs/water_look_design_2026-11-19.md section 9c).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_water_caustic.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``WATER_CAUSTIC_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a
subset. The scene (a 12 x 10 tile pool, 0.3 m, a flow that changes tile by
tile, a lit checker floor) is ``gl_water_flow_check.py``'s, imported, with a
tile ripple added so the slope's both terms are compared.

Each check names the PROPERTY it protects and the change that must break it:

  slope_pass_matches_twin -- the scatter's slope pass writes, per surface
      sample of the light image (``SCATTER_CELLS_PER_TILE`` per tile, at the
      ``SCATTER_OFFSET`` place in its cell), the whole surface slope
      (``water_surface.ripple_slope`` + ``detail_surface``) and the depth;
      they equal the twin's at every sample within a small tolerance.
      Breaks if the pass reads another surface than the draw's (a copy of
      the wave law, the tile ripple left out), samples elsewhere in the
      cell, or loses the depth.
  scatter_matches_twin -- the GL light image equals
      ``water_surface.scatter_light`` of the GL's own slope image (the
      rasteriser's law: the grid, the offset, the anti-diagonal split, the
      area ratio, the bound, additive and unculled) cell for cell, but for
      cells whose centre sits on a moved triangle's edge; its interior mean
      is the incident light. Breaks if a folded triangle is culled, the
      ratio inverted, the split or offset changed, or blending is off.
  floor_read_matches_twin -- debug view 3 writes the floor bend (m), the
      light E read at the bent floor point and the caustic's gain per pixel;
      the bend equals ``water_surface.floor_offset`` of the twin's slope,
      E the light image read bilinear at the GPU's own bent point (held
      inside its written rect), the gain ``caustic_gain(E)``. Breaks if the
      bend takes another sign, factor or slope than the caustic's map, the
      image is read at the unbent point or misplaced, or the gain capped.
  caustic_flat_is_nothing -- with a flat surface (both strengths 0, no tile
      ripple) the light image is exactly 1 at every cell (each centre
      covered once) and the picture with the caustic on is BIT-IDENTICAL to
      the one with it off (``caustic_strength`` 0). Breaks if a flat surface
      leaves any caustic (a sample on a cell centre, a tie double-counted).
  caustic_needs_light -- with no light in the light field (ambient only) a
      rough surface's picture with the caustic on is BIT-IDENTICAL to the one
      with it off. Breaks if the caustic modulates the ambient or adds light
      of its own.
  caustic_dry_does_nothing -- on a dry map the scatter runs nothing:
      ``scatter_caustic`` returns False, allocates no image and compiles no
      program. Breaks if the dry fast path is lost.
  lamp_glint_matches_twin -- the picture with ``lamp_glint`` on minus the
      one with it off is the field's light x ``glint_strength`` x
      ``water_surface.lamp_glint`` (an omni lamp and a beam lamp handed as
      THE assembly's LightSpecs) per pixel; a flat surface is compared
      tightly, a rippled one within the slope's own fp16 tolerance; no
      mounted lamp (only a flashlight) leaves the picture BIT-IDENTICAL to
      the glint off. Breaks if the shader drifts from the twin (the lamp's
      height or position, the sphere widening, the share, the beam's
      wedge or its 2 pi / spread) or a flashlight is taken as a lamp.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_water_caustic_check.py
"""
from __future__ import annotations

import math
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

import gl_water_flow_check as F  # noqa: E402  (the scene, the render)

RESULTS = []
CHECKS = []


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


def _fields(ripple=True, wet=True):
    """The flow check's pool, with a smooth tile ripple (m) unless not."""
    f = F._fields(wet=wet)
    if ripple:
        yy, xx = np.mgrid[0:F.H, 0:F.W].astype(np.float64)
        f["ripple"] = (0.006 * np.sin(0.8 * xx + 0.3)
                       * np.cos(0.6 * yy - 0.2)).astype(np.float32)
    return f


def _pass(s, fields=None, tick=1234, alpha=0.37):
    from renderer.water import WaterPass
    wp = WaterPass(F.H, F.W, s, tile_size_m=F.TILE_M)
    wp.update(**(fields or _fields()), sim_tick=tick, alpha=alpha,
              tps=F.TPS, zoom_px_per_tile=64.0)
    return wp


def _picture(wp, tex):
    wp.scatter_caustic(None)
    drew, _b, out = F._render(wp, tex)
    assert drew
    return out


def _unload(tex):
    for t in tex:
        rl.unload_texture(t)


def _twin_inputs(wp, xt, yt):
    """The twin's per-point inputs from the pass's own packed fp16 planes
    at positions in tiles: flow, depth, agitation, roughness energy."""
    from renderer.heat_shimmer import bilinear_tiles
    fp = wp.flow_packed.astype(np.float64)
    wpk = wp.packed.astype(np.float64)
    v = np.stack([bilinear_tiles(fp[..., 0], xt, yt),
                  bilinear_tiles(fp[..., 1], xt, yt)], -1)
    depth = bilinear_tiles(wpk[..., 2], xt, yt)
    agit = bilinear_tiles(wpk[..., 3], xt, yt)
    energy = (np.abs(bilinear_tiles(wpk[..., 0], xt, yt))
              + np.abs(bilinear_tiles(wpk[..., 1], xt, yt)))
    return fp, wpk, v, depth, agit, energy


def _twin_slope(wp, s, xt, yt):
    """The whole surface slope (the tile ripple's + the fine surface's) of
    the twin at positions in tiles, and the depth there."""
    from renderer.water_surface import (detail_surface, grid_flow_fn,
                                        ripple_slope)
    fp, wpk, v, depth, agit, _e = _twin_inputs(wp, xt, yt)
    det = detail_surface(xt * F.TILE_M, yt * F.TILE_M, v, depth, agit,
                         wp.frame, s, grid_flow_fn(fp, wpk, F.TILE_M, s))
    return ripple_slope(wpk[..., 0], xt, yt, s) + det, depth


def _bilinear_img(img, u, v):
    """GL bilinear (CLAMP) of an (h, w) image at texel coords (u, v)."""
    from renderer.heat_shimmer import bilinear_tiles
    return bilinear_tiles(img, u, v)


def _caustic_images(wp):
    """The scatter's two images, read back over its written rect (cells):
    the light image (R light, G coverage) and the slope image."""
    from renderer import core
    pp = wp.caustic
    x0, y0, x1, y1 = wp.caustic_rect
    from renderer.water_surface import SCATTER_CELLS_PER_TILE as R
    cw, ch = (x1 - x0) * R, (y1 - y0) * R
    light = core.read_texture_rgba32f(pp.front)[:ch, :cw].astype(np.float64)
    slope = core.read_texture_rgba32f(
        pp.back_target.texture)[:ch, :cw].astype(np.float64)
    return light, slope


ROUGH = dict(detail_strength_still=0.08, detail_strength_flow=0.12)


@check("slope_pass_matches_twin")
def check_slope():
    from renderer.water_surface import SCATTER_CELLS_PER_TILE as R
    from renderer.water_surface import SCATTER_OFFSET
    tex = F._textures()
    s = F._settings(caustic_strength=1.0, **ROUGH)
    wp = _pass(s)
    assert wp.scatter_caustic(None)
    _light, slope = _caustic_images(wp)
    x0, y0, _x1, _y1 = wp.caustic_rect
    jj, ii = np.mgrid[0:slope.shape[0], 0:slope.shape[1]]
    xt = x0 + (ii + SCATTER_OFFSET) / R
    yt = y0 + (jj + SCATTER_OFFSET) / R
    twin, depth = _twin_slope(wp, s, xt, yt)
    gpu = slope[..., :2]
    err = np.sqrt(((gpu - twin) ** 2).sum(-1))
    rms = float(np.sqrt((twin ** 2).sum(-1).mean()))
    p99 = float(np.percentile(err, 99.0))
    from renderer.water_surface import ripple_slope
    rip = ripple_slope(wp.packed[..., 0].astype(np.float64), xt, yt, s)
    rip_rms = float(np.sqrt((rip ** 2).sum(-1).mean()))
    assert rms > 0.05 and rip_rms > 0.01, (rms, rip_rms)
    # the flow check's tolerance (8-bit bilinear weights: p99 ~1.7 % of k)
    assert p99 < 0.04 * rms, f"slope GL vs twin: p99 {p99:.2e} (rms {rms})"
    d_err = float(np.abs(slope[..., 2] - depth).max())
    assert d_err < 1e-3, f"depth GL vs twin: max {d_err:.2e}"
    wp.unload()
    _unload(tex)
    return (f"{err.size} samples: p99 |err| {p99:.1e} = {p99 / rms:.2%} of "
            f"rms {rms:.3f} (tile ripple's share rms {rip_rms:.3f}); depth "
            f"max |err| {d_err:.1e} m")


@check("scatter_matches_twin")
def check_scatter():
    from renderer.water_surface import (CAUSTIC_K, SCATTER_CELLS_PER_TILE,
                                        scatter_light)
    tex = F._textures()
    # rough enough to fold here and there at 0.3 m (light lines past 10x)
    s = F._settings(caustic_strength=1.0, detail_strength_still=0.3,
                    detail_strength_flow=0.3)
    wp = _pass(s)
    assert wp.scatter_caustic(None)
    light, slope = _caustic_images(wp)
    cell_m = F.TILE_M / SCATTER_CELLS_PER_TILE
    disp = slope[..., :2] * (slope[..., 2] * CAUSTIC_K / cell_m)[..., None]
    twin = scatter_light(disp)
    gpu = light[..., 0]
    err = np.abs(gpu - twin)
    bad = err > 1e-3 * np.maximum(1.0, twin)
    frac_bad = float(bad.mean())
    # a cell whose centre sits on a moved edge is a tie the two may break
    # differently (fp32 vs fp64 positions): those few cells only
    assert frac_bad < 5e-3, f"{frac_bad:.3%} cells differ"
    assert twin.max() > 10.0, f"vacuous: no fold line ({twin.max():.1f})"
    m = int(np.ceil(np.abs(disp).max())) + 2
    inner = (slice(m, -m), slice(m, -m))
    mean_g = float(gpu[inner].mean())
    assert abs(mean_g - 1.0) < 0.02, f"interior mean {mean_g:.4f}"
    # a sliver at a fold whose edge grazes a cell centre carries its whole
    # ratio into that cell or not (the tie above): the totals agree to the
    # point-sampling noise of those few cells, not to float precision
    assert abs(float(gpu.sum()) - float(twin.sum())) < 1e-2 * twin.sum(), (
        f"sums {gpu.sum():.2f} / {twin.sum():.2f}; diff cells {frac_bad:.3%}; "
        f"max err {err.max():.2f} at E {twin.flat[err.argmax()]:.2f}")
    wp.unload()
    _unload(tex)
    return (f"{gpu.size} cells: {frac_bad:.3%} differ (edge ties), max E "
            f"{gpu.max():.1f}; interior mean {mean_g:.4f}; sums "
            f"{gpu.sum():.1f} / {twin.sum():.1f}")


@check("floor_read_matches_twin")
def check_floor_read():
    from renderer.water_surface import (SCATTER_CELLS_PER_TILE, caustic_gain,
                                        floor_offset)
    tex = F._textures()
    s = F._settings(debug_view=3, caustic_strength=1.0, **ROUGH)
    wp = _pass(s)
    out = _picture(wp, tex)
    light, _slope = _caustic_images(wp)
    yy, xx = np.mgrid[0:F.H * F.PPT, 0:F.W * F.PPT]
    xt = (xx + 0.5) / F.PPT
    yt = (yy + 0.5) / F.PPT
    slope, depth = _twin_slope(wp, s, xt, yt)
    bend = floor_offset(slope, depth)
    gpu_bend = out[..., :2].astype(np.float64)
    b_rms = float(np.sqrt((bend ** 2).sum(-1).mean()))
    b_err = np.sqrt(((gpu_bend - bend) ** 2).sum(-1))
    b_p99 = float(np.percentile(b_err, 99.0))
    assert b_rms > 1e-3, f"vacuous: no bend ({b_rms})"
    assert b_p99 < 0.04 * b_rms, f"bend GL vs twin: p99 {b_p99:.2e}"
    # E read at the GPU's own bent point: the image's placement and filter
    R = SCATTER_CELLS_PER_TILE
    x0, y0, _x1, _y1 = wp.caustic_rect
    ch, cw = light.shape[:2]
    u = (xt + gpu_bend[..., 0] / F.TILE_M - x0) * R
    v = (yt + gpu_bend[..., 1] / F.TILE_M - y0) * R
    u = np.clip(u, 0.5, cw - 0.5)
    v = np.clip(v, 0.5, ch - 0.5)
    e_twin = _bilinear_img(light[..., 0], u, v)
    gpu_e = out[..., 2].astype(np.float64)
    e_err = np.abs(gpu_e - e_twin)
    e_p99 = float(np.percentile(e_err / np.maximum(1.0, e_twin), 99.0))
    assert e_p99 < 0.02, f"E GL vs twin: p99 rel {e_p99:.2e}"
    assert float(e_twin.std()) > 0.05, "vacuous: E is flat"
    # the debug RT is half floats: E and the gain each carry fp16 rounding
    g_rel = np.abs(out[..., 3] - caustic_gain(gpu_e, s)) / np.maximum(
        1.0, np.abs(gpu_e))
    g_err = float(g_rel.max())
    assert g_err < 2.0 ** -9, f"gain != (E - 1) x strength: {g_err:.2e}"
    wp.unload()
    _unload(tex)
    return (f"bend p99 |err| {b_p99:.1e} m = {b_p99 / b_rms:.2%} of rms "
            f"{b_rms * 1000:.2f} mm; E p99 rel err {e_p99:.2e} (E std "
            f"{e_twin.std():.3f}); gain max rel |err| {g_err:.1e} (fp16)")


@check("caustic_flat_is_nothing")
def check_flat():
    tex = F._textures()
    flat = dict(detail_strength_still=0.0, detail_strength_flow=0.0,
                swell_strength_still=0.0, swell_strength_flow=0.0)
    fields = _fields(ripple=False)
    wp = _pass(F._settings(caustic_strength=1.0, **flat), fields)
    on = _picture(wp, tex)
    light, _slope = _caustic_images(wp)
    wp.unload()
    wp = _pass(F._settings(caustic_strength=0.0, **flat), fields)
    off = _picture(wp, tex)
    assert wp.caustic is None, "caustic_strength 0 still ran the scatter"
    wp.unload()
    _unload(tex)
    assert np.all(light[..., 0] == 1.0) and np.all(light[..., 1] == 1.0), (
        f"flat light image: E {light[..., 0].min()}..{light[..., 0].max()}")
    assert np.array_equal(on, off), (
        f"a flat surface left a caustic: max |diff| "
        f"{np.abs(on - off).max():.3e}")
    return (f"flat: {light.shape[0] * light.shape[1]} cells E == 1 and "
            f"covered once; caustic on == off, bit for bit")


@check("caustic_needs_light")
def check_no_light():
    from renderer import core
    tex = F._textures()
    zero = np.zeros((F.H, F.W, 4), np.float16)
    core.update_rgba16f_texture(tex[1], zero)       # no light, no direction
    core.update_rgba16f_texture(tex[2], zero)
    rough = dict(detail_strength_still=0.3, detail_strength_flow=0.3)
    wp = _pass(F._settings(caustic_strength=1.0, **rough))
    on = _picture(wp, tex)
    assert wp.caustic_on, "the scatter did not run"
    wp.unload()
    wp = _pass(F._settings(caustic_strength=0.0, **rough))
    off = _picture(wp, tex)
    wp.unload()
    _unload(tex)
    assert np.array_equal(on, off), (
        f"a caustic with no light: max |diff| {np.abs(on - off).max():.3e}")
    assert on[..., :3].max() > 0.0, "vacuous: the ambient floor is black"
    return "no light: caustic on == off, bit for bit"


@check("caustic_dry_does_nothing")
def check_dry():
    wp = _pass(F._settings(caustic_strength=1.0), _fields(wet=False))
    ran = wp.scatter_caustic(None)
    assert not ran and wp.caustic is None and wp._slope_shader is None, \
        "a dry map ran the scatter"
    wp.unload()
    return "dry: no scatter, no image, no program"


def _glint_case(s, tex, specs, fields):
    """(picture with lamp_glint on, with it off, the pass) for ``specs``."""
    import dataclasses
    wp = _pass(s, fields)
    wp.set_lamp_specs(specs)
    on = _picture(wp, tex)
    wp.refresh_settings(dataclasses.replace(s, lamp_glint=0.0))
    off = _picture(wp, tex)
    return on, off, wp


def _glint_twin(wp, s, specs):
    from renderer.water_surface import lamp_glint, lamp_table
    yy, xx = np.mgrid[0:F.H * F.PPT, 0:F.W * F.PPT]
    xt = (xx + 0.5) / F.PPT
    yt = (yy + 0.5) / F.PPT
    slope, _depth = _twin_slope(wp, s, xt, yt)
    *_r, energy = _twin_inputs(wp, xt, yt)
    N = np.concatenate([-slope, np.ones(slope.shape[:-1] + (1,))], -1)
    N /= np.linalg.norm(N, axis=-1)[..., None]
    rough = np.clip(s.roughness_base + s.roughness_agitation * energy,
                    0.02, 1.0)
    p = np.stack([xt, yt], -1) * F.TILE_M
    return lamp_glint(p, N, lamp_table(specs, F.TILE_M), s, rough)


@check("lamp_glint_matches_twin")
def check_glint():
    from level_lights import LightSpec
    tex = F._textures()
    specs = [LightSpec(x=3.5, y=4.5, color=(1.0, 0.9, 0.8), intensity=1.5,
                       kind="lamp"),
             LightSpec(x=8.5, y=5.5, color=(1.0, 1.0, 1.0), intensity=1.0,
                       angle_center=math.pi, angle_spread=math.radians(90.0),
                       kind="spot")]
    base = dict(lamp_glint=1.0, glint_strength=1.0, roughness_base=0.25,
                caustic_strength=0.0)
    light_rgb = 2.0          # the scene's light field (F._textures), exposure 1
    out = []
    for name, kw, fields, tol in (
            ("flat", dict(detail_strength_still=0.0, detail_strength_flow=0.0,
                          swell_strength_still=0.0, swell_strength_flow=0.0),
             _fields(ripple=False), 0.01),
            ("rippled", ROUGH, _fields(), 0.05)):
        s = F._settings(**base, **kw)
        on, off, wp = _glint_case(s, tex, specs, fields)
        gpu = (on - off)[..., 0].astype(np.float64) / (light_rgb
                                                       * s.glint_strength)
        twin = _glint_twin(wp, s, specs)
        wp.unload()
        gmax = float(twin.max())
        err = np.abs(gpu - twin)
        p99 = float(np.percentile(err, 99.0))
        assert gmax > 0.05, f"vacuous ({name}): no glint ({gmax})"
        assert p99 < tol * gmax, (
            f"{name}: glint GL vs twin p99 {p99:.2e} (max {gmax:.3f})")
        out.append(f"{name} p99 |err| {p99 / gmax:.2%} of max {gmax:.2f}")
    # the beam lights only its wedge: east of the spot (behind it) no glint
    # of it; check on the twin's own numbers that the case is not vacuous
    s = F._settings(**base, detail_strength_still=0.0,
                    detail_strength_flow=0.0, swell_strength_still=0.0,
                    swell_strength_flow=0.0)
    from renderer.water_surface import lamp_glint, lamp_table
    p_behind = np.array([[(8.5 + 1.0) * F.TILE_M, 5.5 * F.TILE_M]])
    p_front = np.array([[(8.5 - 1.0) * F.TILE_M, 5.5 * F.TILE_M]])
    n_up = np.array([[0.0, 0.0, 1.0]])
    beam = lamp_table(specs[1:], F.TILE_M)
    assert lamp_glint(p_behind, n_up, beam, s, 0.25)[0] == 0.0
    assert lamp_glint(p_front, n_up, beam, s, 0.25)[0] > 0.0
    # only a flashlight: not a mounted lamp, no lamp glint at all
    flash = [LightSpec(x=5.5, y=5.5, color=(1.0, 1.0, 1.0), intensity=2.5,
                       angle_spread=0.8, kind="flashlight", source="render")]
    on, off, wp = _glint_case(s, tex, flash, _fields(ripple=False))
    wp.unload()
    _unload(tex)
    assert np.array_equal(on, off), "a flashlight drew a lamp glint"
    out.append("flashlight only: on == off, bit for bit")
    return "; ".join(out)


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_water_caustic_check")
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
    print(f"WATER_CAUSTIC_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
