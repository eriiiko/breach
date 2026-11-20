"""GL checks of the things in the water (water look arc #97, patch W3;
docs/water_look_design_2026-11-19.md W3 and its critique §9, M12 / M13).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_water_things.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``WATER_THINGS_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a
subset.

Each check names the PROPERTY it protects and the change that must break it:

  waterline_runs_the_twin -- ``lit3d._WATERLINE_GLSL``'s ``waterlineTint``
      over a ramp of fragment heights and four water depths (dry, shallow,
      the wading room's, the pool's) equals ``water_surface.waterline_tint``
      within the RGBA16F target's precision, untouched where dry or above
      the surface. Breaks if the GLSL drifts from the twin (the column's
      sign, the fog, the alpha ramp, the metres-to-px scale).
  dry_body_bit_identical -- a lit-3D body (the units' shader, a box mesh)
      drawn with the waterline (no water on the map; water on the map but a
      dry tile under the body; water under the body but the body above its
      surface) writes the SAME BYTES into the scene's RGBA16F target as the
      same shader with the waterline block and call removed -- and a body
      standing in the water does not (the check is not vacuous). Breaks if
      the waterline touches a fragment it should not, or the block changes
      the compiled dry path.
  foam_splats_match_twin -- W2's foam step (``shaders/water_foam.fs``) with
      a ring, a wake capsule and a landing dab (``WaterPass.set_splats`` /
      ``add_dabs``) equals ``water_foam.step`` fed ``frame_splats``' rows in
      every cell within float32 tolerance; the dab is spent by the frame
      that steps and never again. Breaks if the shader's splat rate drifts
      from ``splat_rate``, the cell / tile frames differ, or a dab is
      applied twice or lost.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_water_things_check.py
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
TILE_M = 1.0 / 3.0


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


def _water_tex(depths: np.ndarray):
    """A per-tile 'u_water' texture (RGBA16F, B = depth), bilinear."""
    from renderer import core
    h, w = depths.shape
    tex = core.create_dynamic_rgba16f_texture(w, h)
    px = np.zeros((h, w, 4), np.float16)
    px[..., 2] = depths
    core.update_rgba16f_texture(tex, px)
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    return tex, px[..., 2].astype(np.float64)


def _waterline_ctx(tex, s):
    return (tex, (s.waterline_strength, s.fog_density, s.height_edge),
            (s.alpha_scale, s.alpha_min, s.alpha_max, 0.0),
            tuple(s.water_color))


# ---------------------------------------------------------------------------
# waterline_runs_the_twin
# ---------------------------------------------------------------------------

_HARNESS_FS = """#version 330
// <COMMON>
uniform vec2 u_size;      // the target, px
uniform float u_hmax_m;   // the ramp's top, metres
out vec4 finalColor;
void main() {
    // x: four depth columns, each a ramp of heights 0 .. u_hmax_m
    int x = int(gl_FragCoord.x);
    int seg = int(u_size.x) / 4;
    int c = x / seg;
    float h_m = (float(x - c * seg) + 0.5) / float(seg) * u_hmax_m;
    vec3 lit = vec3(0.9, 0.55, 0.25);
    vec2 uv = vec2((float(c) + 0.5) / 4.0, 0.5);
    finalColor = vec4(waterlineTint(lit, vec3(0.0, h_m * u_waterline.y, 0.0),
                                    uv), 1.0);
}
"""


@check("waterline_runs_the_twin")
def check_waterline():
    from renderer import core, lit3d
    from renderer.water_surface import waterline_tint
    W, H, PX_PER_M, HMAX = 256, 4, 160.0, 0.4
    s = _settings()
    tex, depth = _water_tex(np.array([[0.0, 0.05, 0.15, 0.3]]))
    fs = _HARNESS_FS.replace("// <COMMON>", lit3d._COMMON_GLSL)
    sh = rl.load_shader_from_memory(rl.ffi.NULL, fs)
    assert sh.id > 3, "harness shader failed"
    rt = core.load_render_texture_rgba16f(W, H)

    class _Ctx:
        waterline = _waterline_ctx(tex, s)
    lit3d.bind_waterline_map(sh, [], _Ctx, PX_PER_M)
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.BLANK)
    rl.begin_shader_mode(sh)
    for name, v, kind in (
            ("u_size", (float(W), float(H)),
             rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2),
            ("u_hmax_m", (HMAX,), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)):
        rl.set_shader_value(sh, rl.get_shader_location(sh, name),
                            rl.ffi.new(f"float[{len(v)}]", list(v)), kind)
    rl.set_shader_value_texture(sh, rl.get_shader_location(
        sh, "u_waterline_tex"), tex)
    rl.draw_rectangle(0, 0, W, H, rl.WHITE)
    rl.end_shader_mode()
    rl.end_texture_mode()
    got = core.read_texture_rgba32f(rt.texture)[H // 2, :, :3].astype(
        np.float64)
    seg = W // 4
    x = np.arange(W)
    c = x // seg
    h_m = ((x - c * seg) + 0.5) / seg * HMAX
    lit = np.tile([0.9, 0.55, 0.25], (W, 1))
    want = waterline_tint(lit, depth[0, c], h_m, s)
    err = np.abs(got - want).max(axis=1)
    above = h_m >= depth[0, c]
    assert np.all(np.abs(got[above] - lit[above]).max(axis=1) <= 1e-3), \
        "a fragment at or above the surface changed"
    moved = np.abs(want - lit).max(axis=1)
    assert moved.max() > 0.2, "vacuous: the water tinted nothing"
    assert err.max() <= 2e-3, f"max |GL - twin| {err.max():.2e}"
    rl.unload_render_texture(rt)
    rl.unload_shader(sh)
    rl.unload_texture(tex)
    return (f"{W} fragments, 4 depths: max err {err.max():.1e}; the deepest "
            f"tint moves the colour by {moved.max():.2f}")


# ---------------------------------------------------------------------------
# dry_body_bit_identical
# ---------------------------------------------------------------------------

_CALL = """    // #97 W3: below the water standing on its tile, the body is seen through
    // the water above it (lit3d's one waterline; no water = untouched).
    lit = waterlineTint(lit, fragWorldPos, world_uv);

"""


def _stripped_marine_fs():
    from renderer import lit3d, marine_shader
    fs = marine_shader.MARINE_FS
    assert fs.count(_CALL) == 1 and fs.count(lit3d._WATERLINE_GLSL) == 1, \
        "the marine shader's waterline call / block moved: update the check"
    return fs.replace(_CALL, "").replace(lit3d._WATERLINE_GLSL, "")


def _draw_box(ms, model, rt, ctx, y0_px, size_px, px_per_m, mats):
    from renderer import lit3d
    wpx, hpx = rt.texture.width, rt.texture.height
    sh = ms.shader
    lit3d.bind_smoke_cover_map(sh, mats, ctx)
    lit3d.set_light_curve_uniforms(sh, None)
    lit3d.bind_waterline_map(sh, mats, ctx, px_per_m)
    ms.set_frame_uniforms((0.1, 0.1, 0.12), 1.0, float(wpx), float(hpx))
    ms.set_body_light(False)
    cam = lit3d.make_camera(wpx, hpx)
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.BLANK)
    rl.begin_mode_3d(cam)
    rl.draw_model(model, rl.Vector3(wpx / 2.0, y0_px + size_px / 2.0,
                                    hpx / 2.0), 1.0, rl.WHITE)
    rl.end_mode_3d()
    rl.end_texture_mode()
    from renderer import core
    return core.read_texture_rgba32f(rt.texture)


@check("dry_body_bit_identical")
def check_dry_body():
    from renderer import core, lit3d, marine_shader
    from renderer.lit3d import LightFieldCtx
    WT, HT, PPT = 16, 10, 48
    wpx, hpx = WT * PPT, HT * PPT
    s = _settings()
    full = marine_shader.load_marine_shader()
    keep = marine_shader.MARINE_FS
    try:
        marine_shader.MARINE_FS = _stripped_marine_fs()
        bare = marine_shader.load_marine_shader()
    finally:
        marine_shader.MARINE_FS = keep
    assert full.shader.id > 3 and bare.shader.id > 3, "a shader failed"
    # the props' shader carries the same block and calls it (linked, live)
    from renderer.static_props import PROP_FS, PROP_VS
    sp = rl.load_shader_from_memory(PROP_VS, PROP_FS)
    assert sp.id > 3 and rl.get_shader_location(sp, "u_waterline") >= 0,         "the props' shader lost the waterline"
    rl.unload_shader(sp)
    # the light: a warm lamp from the north-west over the whole map
    la = core.create_dynamic_rgba16f_texture(WT, HT)
    lb = core.create_dynamic_rgba16f_texture(WT, HT)
    a = np.zeros((HT, WT, 4), np.float16)
    a[..., 0:3] = (0.8, 0.6, 0.4)
    a[..., 3] = -0.7071
    b = np.zeros((HT, WT, 4), np.float16)
    b[..., 3] = -0.7071
    core.update_rgba16f_texture(la, a)
    core.update_rgba16f_texture(lb, b)
    wet = np.full((HT, WT), 0.3)
    dry = np.zeros((HT, WT))
    tex_wet, _ = _water_tex(wet)
    tex_dry, _ = _water_tex(dry)
    PX_PER_M = 100.0
    rt = core.load_render_texture_rgba16f(wpx, hpx)
    outs = {}
    for label, ms in (("full", full), ("bare", bare)):
        mesh = rl.gen_mesh_cube(80.0, 20.0, 80.0)
        model = rl.load_model_from_mesh(mesh)
        mats = [model.materials[i] for i in range(model.materialCount)]
        for m in mats:
            m.shader = ms.shader
            m.maps[rl.MaterialMapIndex.MATERIAL_MAP_METALNESS].texture = la
            m.maps[rl.MaterialMapIndex.MATERIAL_MAP_NORMAL].texture = lb
        for case, wl, y0 in (
                ("no_water", None, 0.0),
                ("dry_tile", _waterline_ctx(tex_dry, s), 0.0),
                ("above", _waterline_ctx(tex_wet, s), 0.35 * PX_PER_M),
                ("in_water", _waterline_ctx(tex_wet, s), 0.0)):
            ctx = LightFieldCtx(tex_a=la, tex_b=lb, world_px_w=float(wpx),
                                world_px_h=float(hpx), ambient=(0.1, 0.1, 0.12),
                                light_gain=1.0, waterline=wl)
            outs[(label, case)] = _draw_box(ms, model, rt, ctx, y0, 20.0,
                                            PX_PER_M, mats)
        rl.unload_model(model)
    for case in ("no_water", "dry_tile", "above"):
        f, b_ = outs[("full", case)], outs[("bare", case)]
        assert (f[..., 3] > 0).sum() > 1000, f"vacuous: no body in {case}"
        assert np.array_equal(f, b_), \
            f"{case}: the waterline changed {(f != b_).any(-1).sum()} px"
    f, b_ = outs[("full", "in_water")], outs[("bare", "in_water")]
    diff = (np.abs(f - b_).max(-1) > 1e-3).sum()
    assert diff > 1000, f"vacuous: a body in 0.3 m of water changed {diff} px"
    for t in (la, lb, tex_wet, tex_dry):
        rl.unload_texture(t)
    rl.unload_render_texture(rt)
    return (f"no water / dry tile / above the surface: byte-identical; in "
            f"0.3 m: {diff} px tinted")


# ---------------------------------------------------------------------------
# foam_splats_match_twin
# ---------------------------------------------------------------------------

@check("foam_splats_match_twin")
def check_splats():
    import gl_water_foam_check as G
    from renderer import water_foam as WF
    from renderer.water_wake import WakeTracker, landing_dabs
    wp = G._pass()
    s = wp.settings
    m = s.foam_cells_per_tile
    F = np.zeros((G.H * m, G.W * m))
    wt = WakeTracker()
    dab = landing_dabs(np.array([[12.5, 3.5]]), G.TILE_M, s)
    spent = 0
    for tick in range(0, 7):
        G._update(wp, tick)
        depth = wp.wet_depth.astype(np.float64)
        rates = wt.splats([(1, 5.0 + 0.25 * tick, 6.0), (2, 13.0, 7.0)],
                          tick, depth, G.TILE_M, G.TPS, s)
        wp.set_splats(rates)
        if tick == 2:
            wp.add_dabs(dab)
        dabs_before = wp._splat_dabs.copy()
        wp.simulate()
        n = wp.foam_steps_last_frame
        if n:
            sp, left = WF.frame_splats(rates, dabs_before,
                                       n * s.foam_step_s)
            spent += dabs_before.shape[0] - left.shape[0]
            inp = WF.cell_inputs(wp.packed, wp.flow_packed, m, s, splats=sp)
            for _ in range(n):
                F = WF.step(F, inp, wp._v_max, m, G.TILE_M, s)
    assert spent == 1 and wp._splat_dabs.shape[0] == 0, \
        f"the dab was spent {spent} times"
    gpu = G._read(wp.foam.front)
    err = np.abs(gpu - F)
    # vacuous unless the splats made foam the sources alone did not
    wp2 = G._pass()
    F0 = G._run_ticks(wp2, range(0, 7))
    gain = float((F - F0).max())
    assert gain > 0.05, f"vacuous: the splats added {gain:.3f}"
    assert float(np.percentile(err, 99.0)) < 1e-4 and err.max() < 1e-3, \
        f"|GL - twin| p99 {np.percentile(err, 99.0):.1e} max {err.max():.1e}"
    wp.unload()
    wp2.unload()
    return (f"{err.size} cells: max err {err.max():.1e}; the splats added up "
            f"to {gain:.2f} of foam; the dab spent once")


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.init_window(320, 200, "water things GL check")
    try:
        for name, run in CHECKS:
            if only is None or name in only:
                run()
    finally:
        rl.close_window()
    ok = all(r[1] for r in RESULTS) and bool(RESULTS)
    for name, passed, detail in RESULTS:
        print(f"[check] {name}: {'PASS' if passed else 'FAIL'} {detail}")
    print(f"WATER_THINGS_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
