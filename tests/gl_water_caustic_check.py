"""GL checks of the water pass's sharp caustic (water look arc #97, patch W4;
docs/water_look_design_2026-11-19.md W4 and its critique §9 M5, M14).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_water_caustic.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``WATER_CAUSTIC_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a
subset. The scene (a 12 x 10 tile pool, 0.3 m, a flow that changes tile by
tile, a lit checker floor) is ``gl_water_flow_check.py``'s, imported.

Each check names the PROPERTY it protects and the change that must break it:

  caustic_matches_twin -- debug view 3 writes the fine surface's Hessian
      (hxx, hxy, hyy, 1/m) and the caustic's signed gain per pixel; they
      equal ``water_surface.detail_surface`` + ``caustic_det`` +
      ``caustic_gain`` (the numpy twin, fed the same packed fp16 textures,
      frame uniforms and bakes) at every pixel centre within a small
      tolerance. Breaks if the shader drifts from the twin: the curvature
      read, the tensor's rotation back with its hex tap, the blend weights,
      the k / P scale, the crossfade, the Jacobian, the |det| or the clamp.
  caustic_flat_is_nothing -- with the fine surface flat (both strengths 0)
      the picture with the caustic on is BIT-IDENTICAL to the one with it
      off (``caustic_strength`` 0). Breaks if a flat surface leaves any
      caustic (a bias, a leftover Laplacian of the tile ripple, a time term).
  caustic_needs_light -- with no light in the light field (ambient only) a
      rough surface's picture with the caustic on is BIT-IDENTICAL to the one
      with it off. Breaks if the caustic modulates the ambient or adds light
      of its own (the old Laplacian's floor-independent add did).
  caustic_redistributes -- under light, on a rough surface, the caustic
      both darkens and brightens the floor (the picture minus the caustic-off
      picture has both signs) and never brightens a pixel's floor light by
      more than ``caustic_max - 1`` times. Breaks if the term is clamped to
      focusing only (the old ``max(lap, 0)``) or loses its cap.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_water_caustic_check.py
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


def _picture(s, tex, tick=321, alpha=0.4):
    from renderer.water import WaterPass
    wp = WaterPass(F.H, F.W, s, tile_size_m=F.TILE_M)
    wp.update(**F._fields(), sim_tick=tick, alpha=alpha, tps=F.TPS,
              zoom_px_per_tile=64.0)
    drew, _b, out = F._render(wp, tex)
    assert drew
    wp.unload()
    return out


def _unload(tex):
    for t in tex:
        rl.unload_texture(t)


@check("caustic_matches_twin")
def check_twin():
    from renderer.heat_shimmer import bilinear_tiles
    from renderer.noise_bake import (bake_ripple_curvature_rgba,
                                     bake_ripple_rgba)
    from renderer.water import RIPPLE_SIZE, WaterPass
    from renderer.water_surface import (caustic_det, caustic_gain,
                                        detail_surface)
    tex = F._textures()
    # a moderate surface (focusing up to the cap, barely folding), so the
    # gain is compared where it is not on the knife edge at det = 0 -- the
    # folds' |det| is the twin's property (tests/test_water_caustic.py)
    s = F._settings(debug_view=3, detail_strength_still=0.08,
                    detail_strength_flow=0.12, caustic_strength=1.0,
                    caustic_max=3.0)
    wp = WaterPass(F.H, F.W, s, tile_size_m=F.TILE_M)
    wp.update(**F._fields(), sim_tick=1234, alpha=0.37, tps=F.TPS,
              zoom_px_per_tile=64.0)
    drew, _before, out = F._render(wp, tex)
    assert drew
    yy, xx = np.mgrid[0:F.H * F.PPT, 0:F.W * F.PPT]
    xt = (xx + 0.5) / F.PPT
    yt = (yy + 0.5) / F.PPT
    fp = wp.flow_packed.astype(np.float64)
    wpk = wp.packed.astype(np.float64)
    v = np.stack([bilinear_tiles(fp[..., 0], xt, yt),
                  bilinear_tiles(fp[..., 1], xt, yt)], -1)
    depth = bilinear_tiles(wpk[..., 2], xt, yt)
    agit = bilinear_tiles(wpk[..., 3], xt, yt)
    _slope, hess = detail_surface(
        xt * F.TILE_M, yt * F.TILE_M, v, depth, agit, wp.frame,
        bake_ripple_rgba(RIPPLE_SIZE), bake_ripple_curvature_rgba(RIPPLE_SIZE),
        s)
    det = caustic_det(hess, depth)
    gain = caustic_gain(det, s)
    gpu_h = out[..., :3].astype(np.float64)
    gpu_g = out[..., 3].astype(np.float64)
    h_rms = float(np.sqrt((hess ** 2).sum(-1).mean()))
    h_err = np.sqrt(((gpu_h - hess) ** 2).sum(-1))
    h_p99 = float(np.percentile(h_err, 99.0))
    assert h_rms > 1.0, f"vacuous: the twin's surface is flat ({h_rms})"
    assert (gain < -0.05).mean() > 0.1 and (gain > 0.05).mean() > 0.1, \
        "vacuous: the twin's caustic does not both spread and focus"
    # the same 8-bit bilinear weights as W1's slope check (p99 ~1.7 %)
    assert h_p99 < 0.04 * h_rms, \
        f"Hessian GL vs twin: p99 {h_p99:.3e} (rms {h_rms:.3f})"
    # the gain away from the fold, where 1/|det| is not a knife edge
    calm = np.abs(det) > 0.5
    g_err = np.abs(gpu_g - gain)[calm]
    g_p99 = float(np.percentile(g_err, 99.0))
    assert calm.mean() > 0.5
    assert g_p99 < 0.05, f"gain GL vs twin: p99 {g_p99:.3e}"
    g_max = float(gpu_g.max())
    assert g_max <= (s.caustic_max - 1.0) * s.caustic_strength + 1e-3, g_max
    wp.unload()
    _unload(tex)
    return (f"{h_err.size} px: Hessian p99 |err| {h_p99:.2e} = "
            f"{h_p99 / h_rms:.2%} of rms {h_rms:.2f}/m; gain p99 |err| "
            f"{g_p99:.1e} where |det| > 0.5 ({calm.mean():.0%}); "
            f"folded {(det < 0).mean():.1%}, gpu gain max {g_max:.3f}")


@check("caustic_flat_is_nothing")
def check_flat():
    tex = F._textures()
    flat = dict(detail_strength_still=0.0, detail_strength_flow=0.0)
    on = _picture(F._settings(caustic_strength=1.0, **flat), tex)
    off = _picture(F._settings(caustic_strength=0.0, **flat), tex)
    _unload(tex)
    assert np.array_equal(on, off), (
        f"a flat surface left a caustic: max |diff| "
        f"{np.abs(on - off).max():.3e}")
    return "flat surface: caustic on == off, bit for bit"


@check("caustic_needs_light")
def check_no_light():
    from renderer import core
    tex = F._textures()
    zero = np.zeros((F.H, F.W, 4), np.float16)
    core.update_rgba16f_texture(tex[1], zero)       # no light, no direction
    core.update_rgba16f_texture(tex[2], zero)
    rough = dict(detail_strength_still=0.3, detail_strength_flow=0.3)
    on = _picture(F._settings(caustic_strength=1.0, **rough), tex)
    off = _picture(F._settings(caustic_strength=0.0, **rough), tex)
    _unload(tex)
    assert np.array_equal(on, off), (
        f"a caustic with no light: max |diff| {np.abs(on - off).max():.3e}")
    assert on[..., :3].max() > 0.0, "vacuous: the ambient floor is black"
    return "no light: caustic on == off, bit for bit"


@check("caustic_redistributes")
def check_signs():
    tex = F._textures()
    rough = dict(detail_strength_still=0.15, detail_strength_flow=0.15,
                 glint_strength=0.0, r0=0.0, alpha_min=1.0, alpha_max=1.0,
                 fog_density=0.0)
    s_on = F._settings(caustic_strength=1.0, caustic_max=3.0, **rough)
    on = _picture(s_on, tex)
    off = _picture(F._settings(caustic_strength=0.0, **rough), tex)
    _unload(tex)
    d = (on - off)[..., :3].sum(-1)
    darker, brighter = float((d < -1e-3).mean()), float((d > 1e-3).mean())
    assert darker > 0.1 and brighter > 0.1, (darker, brighter)
    # the floor's own light is the caustic-off picture minus its ambient
    # part; the caustic adds at most (max - 1) x that light, so the whole
    # picture rises by at most (max - 1) x (off) -- a loose, safe bound
    cap = (s_on.caustic_max - 1.0) * s_on.caustic_strength
    over = float(((on - off)[..., :3] > cap * off[..., :3] + 1e-3).mean())
    assert over == 0.0, f"{over:.3%} of pixels brighter than the cap"
    return f"darker {darker:.0%}, brighter {brighter:.0%} of pixels; cap held"


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
