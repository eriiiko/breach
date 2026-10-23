"""GL checks of the flame VFX substrate (graphics arc G3, patch F1;
docs/g3_fire_design_2026-10-23.md §8, §9, §13.3, §13.10, §13.11, §13.13).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_fire_vfx_gl.py`` spawns it and asserts
on its last line (the ``tests/cuda_harness.py`` convention, for GL). Prints
one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``FIRE_VFX_GL_RESULT: PASS`` or ``... FAIL``.

Each check names the PROPERTY it protects and the change that must break it:

  pingpong_replace_and_rows -- a pass REPLACES texels (blending off) and
      writes exactly the GL texel rows it was given (the one Y-flip in
      ``PingPong.draw_cell_rects``). Breaks if blending is left on (raylib's
      alpha blend would keep the old texel under an alpha-0 output) or the
      flip is lost or doubled.
  upload_orientation -- a numpy array uploaded with ``update_rgba32f_texture``
      is read by ``texelFetch(ivec2(gl_FragCoord))`` with array row j at GL
      row j (north = row 0). Breaks if either side flips.
  lut_is_the_ramp -- the blackbody GPU LUT read back equals
      ``BlackbodyRamp.emission_at_kelvin`` at its sample points. Breaks if a
      second T -> colour map appears or the upload drifts.
  stage_parity -- each F1 stage shader (follow_wind, inject, decay_cool)
      matches the numpy twin's function of the same name on one scene with
      permeable and solid fuel, a vacuum tile and a real-shaped wind, to
      float32. Breaks if a shader drifts from the spec (§13.9).
  region_clear -- a tile that leaves the active region is zero in BOTH
      buffers of both fields afterwards, while the live fire's flame is not.
      Breaks if the clear-once rule is lost (a stale flame would reappear).
  no_fire_no_passes -- with no fire, update+simulate issue zero texture-mode
      passes. Breaks if the active-region gate is lost.
  one_tamed_wind_and_render_only -- the real renderer on fire_smoke_studio
      (relief + props + flame all consuming the wind) calls ``tame_wind`` at
      most once per frame, and the field digest of every tick is identical
      with the flame VFX on and off. Breaks if a consumer re-tames the wind
      or any VFX code writes sim state.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_fire_vfx_check.py
(with ``breach_physics`` importable -- PYTHONPATH to a cpp build/Release).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pyray as rl  # noqa: E402

RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                detail = fn() or ""
                RESULTS.append((name, True, detail))
            except Exception as e:                       # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}\n"
                                + traceback.format_exc()))
        run.__name__ = name
        return run
    return deco


_TEST_FS = """#version 330
uniform sampler2D u_tex;
uniform int u_mode;
out vec4 finalColor;
void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    if (u_mode == 0) finalColor = vec4(gl_FragCoord.xy, 0.25, 0.0);
    else finalColor = texelFetch(u_tex, c, 0);
}
"""


def _test_shader():
    from renderer import core
    sh = core.load_shader_from_source(str(ROOT / "shaders" / "lighting.vs"),
                                      _TEST_FS)
    return sh, rl.get_shader_location(sh, "u_tex"), rl.get_shader_location(sh, "u_mode")


def _run_test_pass(pp, sh, loc_tex, loc_mode, mode, rects, tex=None):
    pp.begin_pass()
    rl.begin_shader_mode(sh)
    rl.set_shader_value(sh, loc_mode, rl.ffi.new("int[1]", [mode]),
                        rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
    if tex is not None:
        rl.rl_active_texture_slot(5)
        rl.rl_enable_texture(tex.id)
        rl.set_shader_value(sh, loc_tex, rl.ffi.new("int[1]", [5]),
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
        rl.rl_active_texture_slot(0)
    pp.draw_cell_rects(rects)
    rl.end_shader_mode()
    if tex is not None:
        rl.rl_active_texture_slot(5)
        rl.rl_disable_texture()
        rl.rl_active_texture_slot(0)
    pp.end_pass()


@check("pingpong_replace_and_rows")
def check_pingpong():
    from renderer import core
    pp = core.PingPong(8, 6, fp32=True)
    sh, lt, lm = _test_shader()
    # rows 1..3 (GL texel rows, end-exclusive 4), columns 2..5
    _run_test_pass(pp, sh, lt, lm, 0, [(2, 1, 6, 4)])
    a = core.read_texture_rgba32f(pp.front)
    want = np.zeros((6, 8, 4), np.float32)
    for j in range(1, 4):
        for i in range(2, 6):
            want[j, i] = (i + 0.5, j + 0.5, 0.25, 0.0)
    assert np.array_equal(a, want), f"pass wrote\n{a[..., 0]}\nnot\n{want[..., 0]}"
    rl.unload_shader(sh)
    out = f"fp32={pp.fp32}"
    pp.unload()
    return out


@check("upload_orientation")
def check_upload():
    from renderer import core
    h, w = 5, 7
    arr = np.zeros((h, w, 4), np.float32)
    jj, ii = np.mgrid[0:h, 0:w]
    arr[..., 0] = ii
    arr[..., 1] = jj
    arr[..., 2] = jj * 10 + ii
    tex = core.create_dynamic_rgba32f_texture(w, h)
    core.update_rgba32f_texture(tex, arr)
    pp = core.PingPong(w, h, fp32=True)
    sh, lt, lm = _test_shader()
    _run_test_pass(pp, sh, lt, lm, 1, [(0, 0, w, h)], tex=tex)
    got = core.read_texture_rgba32f(pp.front)
    assert np.array_equal(got, arr), f"copy pass read\n{got[..., 2]}\nnot\n{arr[..., 2]}"
    rl.unload_shader(sh)
    rl.unload_texture(tex)
    pp.unload()


def _scales():
    from renderer import fire_vfx_ref as ref
    return ref.load_scales()


def _t_game_for_kelvin(scales, kelvin):
    return float(scales.tscale.from_kelvin(kelvin))


def _vfx(h, w, **kw):
    from renderer.blackbody import BlackbodyRamp
    from renderer.fire_vfx import FireVFXPass, FireVFXSettings
    from config import CFG
    s = FireVFXSettings(**kw)
    return FireVFXPass(h, w, s, BlackbodyRamp.from_config(CFG))


@check("lut_is_the_ramp")
def check_lut():
    from renderer import core
    from renderer.fire_vfx import LUT_SIZE, blackbody_lut_kelvins
    vfx = _vfx(4, 4)
    got = core.read_texture_rgba32f(vfx.lut_tex)[0]
    chroma, inten = vfx.ramp.emission_at_kelvin(
        blackbody_lut_kelvins(vfx.ramp, LUT_SIZE))
    assert got.shape == (LUT_SIZE, 4)
    assert np.allclose(got[:, :3], chroma, rtol=1e-6, atol=1e-7)
    assert np.allclose(got[:, 3], inten, rtol=1e-6, atol=1e-7)
    vfx.unload()
    return f"{LUT_SIZE} entries, {vfx.ramp.kelvin_floor:.0f}..{vfx.ramp.kelvin_ceil:.0f} K"


def _parity_scene(scales):
    """12 x 10 tiles: a furniture (permeable) fire, a wood (solid) fire, a
    vacuum tile, a solid wall stub, a smooth swirling tamed wind."""
    h, w = 10, 12
    fire = np.zeros((h, w))
    tg = np.full((h, w), _t_game_for_kelvin(scales, 450.0))
    solid = np.zeros((h, w), bool)
    flam = np.zeros((h, w), bool)
    vac = np.zeros((h, w), bool)
    fire[4, 3], flam[4, 3] = 0.8, True                   # furniture
    tg[4, 3] = _t_game_for_kelvin(scales, 1250.0)
    fire[5, 8], flam[5, 8], solid[5, 8] = 1.0, True, True  # wood
    tg[5, 8] = _t_game_for_kelvin(scales, 1100.0)
    solid[2, 5:8] = True
    vac[8, 1] = True
    jj, ii = np.mgrid[0:h, 0:w].astype(np.float64)
    wind = np.zeros((h, w, 2), np.float32)
    wind[..., 0] = 0.3 * np.sin(jj / 3.0) + 0.1
    wind[..., 1] = 0.2 * np.cos(ii / 2.5)
    return fire, tg, solid, flam, vac, wind


@check("stage_parity")
def check_parity():
    from renderer import core
    from renderer import fire_vfx_ref as ref
    from renderer.fire_vfx import (FLAG_DOMAIN, PassItem, pack_tile_inputs)
    scales = _scales()
    fire, tg, solid, flam, vac, wind = _parity_scene(scales)
    h, w = fire.shape
    vfx = _vfx(h, w)
    d = vfx.settings.dials
    m = d.m
    tps = 24.0
    pack = pack_tile_inputs(fire, tg, solid, flam, vac, wind, tps=tps,
                            tile_size_m=0.333, scales=scales, dials=d)
    pack.src_b[..., 3] = pack.src_b[..., 3].astype(np.int32) | FLAG_DOMAIN
    core.update_rgba32f_texture(vfx.src_a_tex, pack.src_a)
    core.update_rgba32f_texture(vfx.src_b_tex, pack.src_b)
    core.update_rgba32f_texture(vfx.wind_tex, pack.wind)
    tick = ref.prepare_tick(fire, tg, solid, flam, vac, wind, tps=tps,
                            tile_size_m=0.333, scales=scales, dials=d)
    rng = np.random.default_rng(7)
    H, W = h * m, w * m
    full = [(0, 0, w, h)]
    notes = []

    # follow_wind (velocity field, m_u == m here)
    u0 = rng.normal(0.0, 1.5, (H, W, 2))
    st = np.zeros((H, W, 4), np.float32)
    st[..., :2] = u0
    st[..., 2] = rng.normal(size=(H, W))                 # p passes through
    core.update_rgba32f_texture(vfx.vel.front, st)
    vfx.run_item(PassItem("stage", "vel", full, stage="follow_wind"))
    got = core.read_texture_rgba32f(vfx.vel.front)
    want = ref.follow_wind(u0.astype(np.float32).astype(np.float64),
                           tick.u_wind, d.k_follow, d.dt_s)
    err = np.abs(got[..., :2] - want).max()
    assert err < 2e-5, f"follow_wind max err {err}"
    assert np.array_equal(got[..., 2], st[..., 2]), "p not passed through"
    notes.append(f"follow_wind {err:.1e}")

    # inject (flame field) at a non-trivial noise time
    R0 = np.abs(rng.normal(0.0, 0.5, (H, W)))
    T0 = rng.uniform(500.0, 1200.0, (H, W))
    fl = np.zeros((H, W, 4), np.float32)
    fl[..., 0] = R0
    fl[..., 1] = R0 * T0
    R0 = fl[..., 0].astype(np.float64)
    RT0 = fl[..., 1].astype(np.float64)
    core.update_rgba32f_texture(vfx.flame.front, fl)
    t = 3.7
    vfx.run_item(PassItem("stage", "flame", full, stage="inject", t=t))
    got = core.read_texture_rgba32f(vfx.flame.front)
    Rw, RTw = ref.inject(R0, RT0, tick, t, d.dt_s, d)
    eR = np.abs(got[..., 0] - Rw).max()
    eT = np.abs(got[..., 1] - RTw).max() / max(1.0, np.abs(RTw).max())
    assert eR < 1e-5 and eT < 1e-6, f"inject err R {eR} RT(rel) {eT}"
    assert (np.abs(got[..., 0] - R0) > 0).sum() == (tick.s > 0).sum(), \
        "injected cells differ from the twin's cover rule"
    notes.append(f"inject {eR:.1e}/{eT:.1e} on {(tick.s > 0).sum()} cells")

    # decay_cool
    fl2 = got.copy()
    core.update_rgba32f_texture(vfx.flame.front, fl2)
    vfx.run_item(PassItem("stage", "flame", full, stage="decay_cool"))
    got2 = core.read_texture_rgba32f(vfx.flame.front)
    Rw, RTw = ref.decay_cool(fl2[..., 0].astype(np.float64),
                             fl2[..., 1].astype(np.float64), tick, d.dt_s, d)
    eR = np.abs(got2[..., 0] - Rw).max()
    eT = np.abs(got2[..., 1] - RTw).max() / max(1.0, np.abs(RTw).max())
    assert eR < 1e-5 and eT < 1e-5, f"decay_cool err R {eR} RT(rel) {eT}"
    assert np.all(got2[..., 0][tick.blocked] == 0.0), "R left in blocked cells"
    notes.append(f"decay_cool {eR:.1e}/{eT:.1e}")
    vfx.unload()
    return "; ".join(notes)


@check("region_clear")
def check_region_clear():
    from renderer import core
    from renderer.fire_vfx import linger_s
    scales = _scales()
    h, w = 24, 40
    vfx = _vfx(h, w)
    d = vfx.settings.dials
    hot = _t_game_for_kelvin(scales, 1250.0)
    z = np.zeros((h, w), bool)
    flam = np.ones((h, w), bool)
    tg = np.zeros((h, w))
    wind = np.zeros((h, w, 2), np.float32)

    def frame(tick, fire):
        tgf = np.where(fire > 0, hot, tg)
        vfx.update(fire=fire, t_game=tgf, solid=z, flammable=flam, vacuum=z,
                   wind_fn=lambda: wind, sim_tick=tick, alpha=0.0,
                   paused=False, tps=24.0, tile_size_m=0.333, scales=scales)
        vfx.simulate()

    fa = np.zeros((h, w))
    fa[5, 5] = 1.0
    tick = 0
    for _ in range(30):
        frame(tick, fa)
        tick += 1
    box_a = list(vfx.region.boxes)
    ra = core.read_texture_rgba32f(vfx.flame.front)[..., 0]
    assert ra.max() > 0.1, f"no flame grew at A (max R {ra.max()})"
    fb = np.zeros((h, w))
    fb[18, 32] = 1.0
    n_after = int(np.ceil(linger_s(d) * 24.0)) + 3
    for _ in range(n_after):
        frame(tick, fb)
        tick += 1
    assert all(b not in vfx.region.boxes for b in box_a), "A's box survived"
    m = d.m
    for f, mm in ((vfx.flame, m), (vfx.vel, vfx.settings.m_u)):
        for rt in f._rts:
            a = core.read_texture_rgba32f(rt.texture)
            for x0, y0, x1, y1 in box_a:
                blk = a[y0 * mm:y1 * mm, x0 * mm:x1 * mm]
                assert not np.any(blk), "a departed tile kept data"
    rb = core.read_texture_rgba32f(vfx.flame.front)[..., 0]
    assert rb[18 * m:19 * m, 32 * m:33 * m].max() > 0.1, "no flame at B"
    out = f"A boxes {box_a} cleared; B boxes {vfx.region.boxes}"
    vfx.unload()
    return out


@check("no_fire_no_passes")
def check_no_fire():
    scales = _scales()
    h, w = 16, 16
    vfx = _vfx(h, w)
    z = np.zeros((h, w), bool)
    calls = []
    orig = rl.begin_texture_mode
    rl.begin_texture_mode = lambda rt: (calls.append(1), orig(rt))[1]
    try:
        for tick in range(20):
            vfx.update(fire=np.zeros((h, w)), t_game=np.zeros((h, w)),
                       solid=z, flammable=z, vacuum=z,
                       wind_fn=lambda: (_ for _ in ()).throw(
                           AssertionError("wind asked for with no fire")),
                       sim_tick=tick, alpha=0.0, paused=False, tps=24.0,
                       tile_size_m=0.333, scales=scales)
            n = vfx.simulate()
            assert n == 0, f"{n} passes planned without fire"
    finally:
        rl.begin_texture_mode = orig
    assert not calls, f"{len(calls)} texture-mode passes without fire"
    vfx.unload()


@check("one_tamed_wind_and_render_only")
def check_one_wind_and_digest():
    import breach_physics as bp
    from config import CFG
    from field_ab_harness import SIM_FIELDS, _snapshot
    from field_digest import field_digest
    from level_loader import load as load_level
    from renderer import game_renderer as gr
    from renderer.game_renderer import GameRenderer, RenderConfig
    from simulation import Simulation

    level = load_level("fire_smoke_studio")
    cfg = RenderConfig(map_px_w=448, map_px_h=448, panel_px_w=0,
                       grid_w=level.width, grid_h=level.height,
                       world_px_per_tile=8.0)
    renderer = GameRenderer(level, bp, cfg)
    renderer._props_seen = True          # the props consumer asks too
    counts = []
    orig = gr.tame_wind

    def counting(*a, **k):
        counts[-1] += 1
        return orig(*a, **k)

    gr.tame_wind = counting
    n_ticks = 40                         # fires light at 0.5 s = tick 12
    digests = {}
    passes = 0
    try:
        for on in (True, False):
            CFG.render.fire_vfx.enabled = on
            sim = Simulation(level, seed=42, breach_physics=bp,
                             enable_recorder=False)
            sim.set_paused(False)
            trail = []
            for tick in range(n_ticks):
                if sim.is_paused():
                    sim.set_paused(False)
                sim.step()
                counts.append(0)
                renderer.upload_state(sim.gmap, sim_tick=tick + 1,
                                      alpha=0.0, paused=False)
                renderer.begin_frame()
                renderer.compose_world()
                renderer.end_frame()
                if on:
                    passes += renderer.fire_vfx.passes_last_frame
                trail.append(field_digest(_snapshot(sim.gmap, SIM_FIELDS),
                                          strict=False))
            digests[on] = trail
    finally:
        gr.tame_wind = orig
        CFG.reload()
        renderer.shutdown()
    assert max(counts) <= 1, f"tame_wind called {max(counts)}x in one frame"
    assert min(counts) == 1, "a frame with consumers computed no wind"
    assert passes > 0, "the flame VFX never ran a pass on a burning studio"
    assert digests[True] == digests[False], "field digests differ VFX on/off"
    return (f"{n_ticks} ticks, tame_wind {min(counts)}..{max(counts)}/frame, "
            f"{passes} VFX passes, digests equal")


def main() -> int:
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_fire_vfx_check")
    try:
        for fn in (check_pingpong, check_upload, check_lut, check_parity,
                   check_region_clear, check_no_fire,
                   check_one_wind_and_digest):
            fn()
    finally:
        if rl.is_window_ready():
            rl.close_window()
    ok = all(r[1] for r in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"[check] {name}: {'PASS' if passed else 'FAIL'} {detail}")
    print(f"FIRE_VFX_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
