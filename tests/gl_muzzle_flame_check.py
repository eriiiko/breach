"""GL checks of the muzzle flame (#87; docs/muzzle_flame_design_2026-11-12.md).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_muzzle_flame.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``MUZZLE_FLAME_GL_RESULT: PASS`` or ``... FAIL``.

Each check names the PROPERTY it protects and the change that must break it:

  flame_matches_the_twin -- three flashes (three directions, three ages,
      two kinds) drawn into a black float target equal the numpy twin's
      picture (``muzzle_flame.rasterise``): the summed light within 2 %, and
      per pixel within 3 % of the picture's peak. Breaks if the shader
      drifts from the law (a constant, a noise coordinate, a row's layout),
      the quad is rotated or mirrored against the twin's frame, or the y
      axis is flipped.
  flame_is_additive_and_idle_when_empty -- over a lit floor the draw only
      ADDS light and keeps the destination alpha; an empty pack issues no
      draw call and changes nothing; a flash at age >= 1 adds nothing.
      Breaks if the blend state is wrong or a spent flash still draws.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_muzzle_flame_check.py
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
W, H = 384, 320
PPM = 110.0                     # px per metre in the checks


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


def _ramp():
    from renderer.blackbody import BlackbodyRamp
    return BlackbodyRamp(kelvin_ref=1300.0, intensity_exponent=2.0)


def _lut_tex(ramp):
    from renderer import core
    from renderer.fire_vfx import LUT_SIZE, blackbody_lut_data
    lut = core.create_dynamic_rgba32f_texture(LUT_SIZE, 1)
    rl.set_texture_filter(lut, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    core.update_rgba32f_texture(lut, blackbody_lut_data(ramp))
    return lut


def _kind(**kw):
    from renderer.muzzle_flame import FlameKind
    base = dict(size=1.0, length_m=0.95, width_m=0.13, side_jets=2,
                side_len_m=0.55, side_width_m=0.085, side_angle_deg=55.0,
                core_m=0.10, halo_m=0.30, halo_gain=0.035, t_core_k=4200.0,
                t_edge_k=1050.0, gain=1.6, tear=0.8, fireball_chance=0.12,
                duration_s=0.055, sparks=0)
    base.update(kw)
    return FlameKind(**base)


def _pack(flashes):
    """``flashes``: (kind, tick, x px, y px, angle rad, age) -> the pack."""
    import math
    from renderer import muzzle_flame as mf
    out = np.zeros((mf.ROWS, len(flashes), 4), np.float32)
    for i, (kind, tick, x, y, ang, age) in enumerate(flashes):
        shot = mf.shot_rows(kind, mf.shot_rng(5, 1, tick))
        out[0, i] = (x, y, math.cos(ang), math.sin(ang))
        out[1, i] = (age, shot.ofs[0], shot.ofs[1], PPM)
        out[2:, i] = shot.rows
    return out


def _draw(md, pack, lut, ramp, under=(0, 0, 0, 0)):
    from renderer import core
    from renderer.fire_vfx import LUT_SIZE
    rt = core.load_render_texture_rgba16f(W, H)
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.Color(*under))
    n = md.draw(pack, world_px_w=W, world_px_h=H, lut_tex=lut,
                lut_k=(ramp.kelvin_floor, ramp.kelvin_ceil), lut_n=LUT_SIZE)
    rl.end_texture_mode()
    img = core.read_texture_rgba32f(rt.texture).astype(np.float64)
    rl.unload_render_texture(rt)
    return n, img[::-1]             # GL rows run up: flip to world rows


@check("flame_matches_the_twin")
def check_twin():
    from renderer import muzzle_flame as mf
    ramp = _ramp()
    lut = _lut_tex(ramp)
    md = mf.MuzzleFlameDraw(4)
    heavy = _kind(size=1.25, side_jets=4)
    pack = _pack([(_kind(), 3, 70.0, 80.0, 0.0, 0.0),
                  (heavy, 4, 300.0, 60.0, 2.2, 0.45),
                  (_kind(), 6, 150.0, 300.0, -1.2, 0.8)])
    n, img = _draw(md, pack, lut, ramp)
    want = mf.rasterise(pack, W, H, mf.noise_field(), ramp)
    got = img[..., :3]
    assert n == 1, f"{n} draw calls for one batch"
    peak = float(want.max())
    assert peak > 1.0, "the reference picture has no white-hot core"
    tot = abs(got.sum() - want.sum()) / want.sum()
    err = np.abs(got - want).max(axis=-1)
    worst = float(err.max()) / peak
    mean = float(err.mean()) / float(want.max(axis=-1).mean())
    assert tot < 0.02, f"summed light off by {100 * tot:.2f} %"
    assert worst < 0.03, f"worst pixel off by {100 * worst:.2f} % of the peak"
    # a mirrored or rotated quad would move the light: the centres agree
    ys, xs = np.mgrid[0:H, 0:W]
    for a in (got, want):
        w = a.sum(axis=-1)
        a_c = ((w * xs).sum() / w.sum(), (w * ys).sum() / w.sum())
        if a is got:
            c_got = a_c
    assert abs(c_got[0] - a_c[0]) < 0.5 and abs(c_got[1] - a_c[1]) < 0.5
    md.unload()
    rl.unload_texture(lut)
    return (f"sum within {100 * tot:.2f} %, worst pixel {100 * worst:.2f} % "
            f"of the peak, mean error {100 * mean:.2f} % of the mean")


@check("flame_is_additive_and_idle_when_empty")
def check_additive():
    from renderer import muzzle_flame as mf
    ramp = _ramp()
    lut = _lut_tex(ramp)
    md = mf.MuzzleFlameDraw(4)
    under = (40, 60, 30, 200)
    u = np.array(under) / 255.0
    n0, idle = _draw(md, np.empty((mf.ROWS, 0, 4), np.float32), lut, ramp,
                     under)
    assert n0 == 0, "an empty pack issued a draw"
    n1, img = _draw(md, _pack([(_kind(), 3, 120.0, 160.0, 0.3, 0.1)]), lut,
                    ramp, under)
    assert n1 == 1
    assert np.allclose(img[..., 3], idle[..., 3], atol=2e-3), "alpha written"
    assert np.all(img[..., :3] >= idle[..., :3] - 2e-3), "the flame darkened"
    lit = int(((img[..., :3] - idle[..., :3]).sum(axis=-1) > 1e-2).sum())
    assert lit > 400, f"only {lit} px lit"
    n2, spent = _draw(md, _pack([(_kind(), 3, 120.0, 160.0, 0.3, 1.0)]), lut,
                      ramp, under)
    assert np.allclose(spent, idle, atol=2e-3), "a spent flash added light"
    assert np.allclose(idle[..., :3], u[:3], atol=4e-3)
    md.unload()
    rl.unload_texture(lut)
    return f"{lit} px lit, alpha kept, nothing darkened, spent adds nothing"


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_muzzle_flame_check")
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
    print(f"MUZZLE_FLAME_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
