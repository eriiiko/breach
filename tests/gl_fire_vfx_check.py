"""GL checks of the flame VFX pass (graphics arc G3, patches F1 + F2;
docs/g3_fire_design_2026-10-23.md §8, §9, §13.1, §13.3, §13.9-§13.11, §13.13).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_fire_vfx_gl.py`` spawns it and asserts
on its last line (the ``tests/cuda_harness.py`` convention, for GL). Prints
one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``FIRE_VFX_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a subset.

Each check names the PROPERTY it protects and the change that must break it:

  pingpong_replace_and_rows -- a direct-GL pass REPLACES texels (blending
      off) and writes exactly the GL texel rows it was given (no Y flip on the
      pass path), and the block restores raylib's blend / framebuffer /
      viewport. Breaks if blending is left on, a flip creeps in, or the block
      leaks GL state into raylib's next draw.
  upload_orientation -- a numpy array uploaded with ``update_rgba32f_texture``
      is read by ``texelFetch(ivec2(gl_FragCoord))`` with array row j at GL
      row j (north = row 0). Breaks if either side flips.
  lut_is_the_ramp -- the blackbody GPU LUT read back equals
      ``BlackbodyRamp.emission_at_kelvin`` at its sample points. Breaks if a
      second T -> colour map appears or the upload drifts.
  pass_parity -- EACH GL pass equals its twin composition (``PASSES``) one
      step from the same random state, on a scene with permeable and solid
      fuel, a vacuum tile, a wall stub and a swirling wind, to float32
      (§13.9). Breaks if a shader drifts from the spec.
  pass_parity_two_boxes -- the same on a frame with two boxes: each box is
      its own domain (the twin run on that box alone). Breaks if a pass reads
      across its box edge or the box lookup is lost.
  long_run_invariants -- over 2.6 s of the full loop on the F0b scenes: R >=
      0, T_vfx <= T_src (max), R = 0 in solid and vacuum, all finite, and the
      total R within 10 % of the twin's on the same scene (§13.9). Breaks if
      a pass makes new extrema, leaks flame into walls or loses mass.
  sealed_room_bounded -- a fire in a closed 6x6 room keeps max|p| bounded
      over 10^4 GL substeps (§13.1). Breaks if the screening is lost in GL.
  ten_minutes_finite -- 72 000 substeps (10 min at 1/120 s) of the studio's
      real fire row stay finite and bounded. Breaks on any slow blow-up.
  region_clear -- a tile that leaves the active region is zero in BOTH
      buffers of both fields afterwards, while the live fire's flame is not.
      Breaks if the clear-once rule is lost (a stale flame would reappear).
  burnout_leaves_nothing -- a fire that goes out leaves R = 0 everywhere in
      both buffers, an empty region and zero passes after its linger. Breaks
      if the decay/clear or the region bookkeeping leaves a stale flame.
  box_growth -- a second fire lighting nearby grows / merges the boxes mid
      run with no NaN and no flame outside the new domain. Breaks if a grown
      box reads stale texels.
  realloc_on_grid_change -- a live m change reallocates zeroed fields;
      disabling empties the plan.
  config_reload_live -- a live dial change (n_jacobi, lifetime_s) reaches
      the next frame's passes without reallocating the fields. Breaks if a
      dial is cached past a reload.
  no_fire_no_passes -- with no fire, update+simulate issue zero passes and
      open no GL block. Breaks if the active-region gate is lost.
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


def check(name):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                detail = fn() or ""
                RESULTS.append((name, True, f"{detail} [{time.perf_counter() - t0:.1f}s]"))
            except Exception as e:                       # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}\n"
                                + traceback.format_exc()))
        run.__name__ = name
        CHECKS.append((name, run))
        return run
    return deco


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _direct(pp, mode, rects, tex=None):
    """One direct-GL pass of the test shader over texel ``rects``."""
    from renderer import core
    sh = core.load_pass_shader(_TEST_FS)
    gl = core.GLDirect.get()
    with core.DirectPasses() as gp:
        gp.use(sh)
        gl.Uniform1i(rl.get_shader_location(sh, "u_mode"), mode)
        flat = [float(v) for r in rects for v in r] + [0.0] * (16 - 4 * len(rects))
        gl.Uniform4fv(rl.get_shader_location(sh, "u_boxes"), 4, gl.floats(flat))
        gl.Uniform2f(rl.get_shader_location(sh, "u_target"),
                     float(pp.width), float(pp.height))
        if tex is not None:
            gp.bind(5, tex.id)
            gl.Uniform1i(rl.get_shader_location(sh, "u_tex"), 5)
        pp.direct_pass(gp, len(rects))
    rl.unload_shader(sh)


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


def _read(tex):
    from renderer import core
    return core.read_texture_rgba32f(tex).astype(np.float64)


def _put(tex, arr):
    from renderer import core
    core.update_rgba32f_texture(tex, np.ascontiguousarray(arr, np.float32))


def _set_boxes(vfx, boxes):
    vfx.region.boxes = list(boxes)
    from renderer.fire_vfx import boxes_mask
    vfx.region.domain = boxes_mask(boxes, (vfx.h, vfx.w))


# ---------------------------------------------------------------------------
# the substrate
# ---------------------------------------------------------------------------

@check("pingpong_replace_and_rows")
def check_pingpong():
    from renderer import core
    pp = core.PingPong(8, 6, fp32=True)
    # a non-zero background in both buffers: an alpha-0 output must REPLACE it
    for _ in range(2):
        _direct(pp, 0, [(0, 0, 8, 6)])
    rt = rl.load_render_texture(16, 16)
    _direct(pp, 0, [(2, 1, 6, 4)])
    a = _read(pp.front)
    want = np.zeros((6, 8, 4))
    for j in range(6):
        for i in range(8):
            want[j, i] = (i + 0.5, j + 0.5, 0.25, 0.0)
    assert np.array_equal(a, want), f"pass wrote\n{a[..., 1]}\nnot\n{want[..., 1]}"
    # raylib carries on with ITS state: an alpha-blended draw still blends
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.Color(0, 0, 255, 255))
    rl.draw_rectangle(0, 0, 16, 16, rl.Color(255, 0, 0, 128))
    rl.end_texture_mode()
    img = rl.load_image_from_texture(rt.texture)
    px = rl.get_image_color(img, 8, 8)
    rl.unload_image(img)
    rl.unload_render_texture(rt)
    assert 100 < px.r < 160 and 100 < px.b < 160, \
        f"blend not restored after a direct block: {px.r},{px.g},{px.b}"
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
    _direct(pp, 1, [(0, 0, w, h)], tex=tex)
    got = _read(pp.front)
    assert np.array_equal(got, arr), f"copy pass read\n{got[..., 2]}\nnot\n{arr[..., 2]}"
    rl.unload_texture(tex)
    pp.unload()


@check("lut_is_the_ramp")
def check_lut():
    from renderer.fire_vfx import LUT_SIZE, blackbody_lut_kelvins
    vfx = _vfx(4, 4)
    got = _read(vfx.lut_tex)[0]
    chroma, inten = vfx.ramp.emission_at_kelvin(
        blackbody_lut_kelvins(vfx.ramp, LUT_SIZE))
    assert got.shape == (LUT_SIZE, 4)
    assert np.allclose(got[:, :3], chroma, rtol=1e-6, atol=1e-7)
    assert np.allclose(got[:, 3], inten, rtol=1e-6, atol=1e-7)
    vfx.unload()
    return f"{LUT_SIZE} entries, {vfx.ramp.kelvin_floor:.0f}..{vfx.ramp.kelvin_ceil:.0f} K"


# ---------------------------------------------------------------------------
# per-pass parity (§13.9)
# ---------------------------------------------------------------------------

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


def _random_state(rng, H, W, blocked):
    u = rng.normal(0.0, 1.5, (H, W, 2))
    p = rng.normal(0.0, 0.05, (H, W))
    rhs = rng.normal(0.0, 5.0, (H, W))
    R = np.abs(rng.normal(0.0, 0.5, (H, W))) * (rng.random((H, W)) < 0.7)
    T = rng.uniform(500.0, 1200.0, (H, W))
    # the twin's own invariants on the input: nothing in blocked cells
    R = np.where(blocked, 0.0, R)
    f32 = lambda a: a.astype(np.float32).astype(np.float64)  # noqa: E731
    return f32(u), f32(p), f32(rhs), f32(R), f32(R * T)


def _vel_tex(u, p, rhs):
    v = np.zeros(u.shape[:2] + (4,))
    v[..., :2], v[..., 2], v[..., 3] = u, p, rhs
    return v


def _flame_tex(R, RT):
    f = np.zeros(R.shape + (4,))
    f[..., 0], f[..., 1] = R, RT
    return f


def _err(got, want, scale=None):
    """Max abs error, normalised by max(1, max|want|) unless ``scale``."""
    s = scale if scale is not None else max(1.0, float(np.abs(want).max()))
    return float(np.abs(got - want).max()) / s


def _pass_parity(vfx, tick, d, modes, H, W, rng, t, t_base, region, label,
                 blocked_full):
    """Every pass vs its twin composition on the cells of ``region`` (a
    (y0, y1, x0, x1) cell slice the twin ran on as its whole grid)."""
    from renderer import fire_vfx_ref as ref
    m = d.m
    notes = []
    u, p, rhs, R, RT = _random_state(rng, H, W, blocked_full)

    def run(stage, vel=None, flame=None):
        if vel is not None:
            _put(vfx.vel.front, vel)
        if flame is not None:
            _put(vfx.flame.front, flame)
        vfx.run_pass(stage, t)
        return _read(vfx.vel.front), _read(vfx.flame.front)

    ys, xs = region
    S = (ys, xs)

    # vel_advect = advect_velocity -> follow_wind
    gv, _ = run("vel_advect", vel=_vel_tex(u, p, rhs))
    want = ref.follow_wind(ref.advect_velocity(u[S], d.dt_s, m),
                           tick.u_wind, d.k_follow, d.dt_s)
    e = _err(gv[S][..., :2], want)
    assert e < 2e-5, f"{label} vel_advect err {e}"
    assert np.array_equal(gv[S][..., 2:], _vel_tex(u, p, rhs)[S][..., 2:].astype(np.float32)), \
        f"{label} vel_advect did not pass p/rhs through"
    notes.append(f"vel_advect {e:.1e}")

    # confine
    gv, _ = run("confine", vel=_vel_tex(u, p, rhs))
    want = ref.confine(u[S], tick.i_local, tick.solid, t, d.dt_s, d, modes)
    e = _err(gv[S][..., :2], want)
    assert e < 5e-5, f"{label} confine err {e}"
    notes.append(f"confine {e:.1e}")

    # divergence = pressure_rhs (into w)
    gv, _ = run("divergence", vel=_vel_tex(u, p, rhs))
    want = ref.pressure_rhs(u[S], tick, t, d)
    e = _err(gv[S][..., 3], want)
    assert e < 2e-5, f"{label} divergence err {e}"
    notes.append(f"divergence {e:.1e}")

    # jacobi = one jacobi_sweep (p in z, rhs in w)
    gv, _ = run("jacobi", vel=_vel_tex(u, p, rhs))
    want = ref.jacobi_sweep(p[S], rhs[S], tick.solid, tick.vacuum, m,
                            d.screen_len_tiles)
    e = _err(gv[S][..., 2], want, scale=max(1e-3, float(np.abs(want).max())))
    assert e < 2e-5, f"{label} jacobi err {e}"
    notes.append(f"jacobi {e:.1e}")

    # project
    gv, _ = run("project", vel=_vel_tex(u, p, rhs))
    want = ref.project(u[S], p[S], tick.solid, tick.vacuum, m)
    e = _err(gv[S][..., :2], want)
    assert e < 2e-5, f"{label} project err {e}"
    notes.append(f"project {e:.1e}")

    # flame = advect_flame -> inject -> decay_cool, on the given velocity
    _, gf = run("flame", vel=_vel_tex(u, p, rhs), flame=_flame_tex(R, RT))
    Rw, RTw = ref.advect_flame(R[S], RT[S], u[S], d.dt_s, d)
    Rw, RTw = ref.inject(Rw, RTw, tick, t, d.dt_s, d)
    Rw, RTw = ref.decay_cool(Rw, RTw, tick, d.dt_s, d)
    dR = np.abs(gf[S][..., 0] - Rw)
    dT = np.abs(gf[S][..., 1] - RTw) / max(1.0, float(np.abs(RTw).max()))
    # MacCormack's revert and the r_cut clear are DISCRETE: a cell whose
    # corrected value sits within float32 of a stencil bound may take the
    # other branch. That is allowed on a vanishing fraction of cells only.
    bad = (dR > 1e-4) | (dT > 1e-5)
    frac = float(bad.mean())
    assert frac < 2e-3, f"{label} flame: {bad.sum()} cells off (R {dR.max():.2e})"
    ok = ~bad
    eR, eT = float(dR[ok].max()), float(dT[ok].max())
    assert eR < 1e-4 and eT < 1e-5, f"{label} flame err R {eR} RT {eT}"
    assert np.all(gf[S][..., 0][tick.blocked] == 0.0), f"{label} R left in blocked cells"
    notes.append(f"flame {eR:.1e}/{eT:.1e} ({bad.sum()} branch cells)")
    return notes


@check("pass_parity")
def check_pass_parity():
    from renderer import fire_vfx_ref as ref
    from renderer.fire_vfx import pack_tile_inputs
    scales = _scales()
    fire, tg, solid, flam, vac, wind = _parity_scene(scales)
    h, w = fire.shape
    vfx = _vfx(h, w)
    d = vfx.settings.dials
    tps, t, t_base = 24.0, 3.7, 3.65
    pack = pack_tile_inputs(fire, tg, solid, flam, vac, wind, tps=tps,
                            tile_size_m=0.333, scales=scales, dials=d,
                            t_base=t_base)
    vfx.upload_pack(pack)
    _set_boxes(vfx, [(0, 0, w, h)])
    tick = ref.prepare_tick(fire, tg, solid, flam, vac, wind, tps=tps,
                            tile_size_m=0.333, scales=scales, dials=d)
    H, W = h * d.m, w * d.m
    notes = _pass_parity(vfx, tick, d, vfx.curl_modes(), H, W,
                         np.random.default_rng(7), t, t_base,
                         (slice(0, H), slice(0, W)), "full", tick.blocked)
    vfx.unload()
    return "; ".join(notes)


@check("pass_parity_two_boxes")
def check_pass_parity_boxes():
    """Two boxes on one map: each must equal the twin run on that box as its
    whole grid (the domain rule). The fires sit in box A; box B is wind only."""
    from renderer import fire_vfx_ref as ref
    from renderer.fire_vfx import pack_tile_inputs
    scales = _scales()
    fire, tg, solid, flam, vac, wind = _parity_scene(scales)
    h, w = fire.shape
    vfx = _vfx(h, w)
    d = vfx.settings.dials
    m = d.m
    tps, t, t_base = 24.0, 1.2, 1.0
    pack = pack_tile_inputs(fire, tg, solid, flam, vac, wind, tps=tps,
                            tile_size_m=0.333, scales=scales, dials=d,
                            t_base=t_base)
    vfx.upload_pack(pack)
    boxes = [(0, 0, 10, 7), (0, 7, 12, 10)]          # A (fires), B (south strip)
    _set_boxes(vfx, boxes)
    notes = []
    for (x0, y0, x1, y1), label in zip(boxes, ("A", "B")):
        sl = (slice(y0, y1), slice(x0, x1))
        # the twin on the box alone: the per-TILE inputs keep their map
        # context (wind bilinear across the box edge is the map's), so the
        # twin's tick is the full-map one cropped to the box's cells
        tick_full = ref.prepare_tick(fire, tg, solid, flam, vac, wind, tps=tps,
                                     tile_size_m=0.333, scales=scales, dials=d)
        cs = (slice(y0 * m, y1 * m), slice(x0 * m, x1 * m))
        tk = ref.TickInputs(**{f: (getattr(tick_full, f)[cs]
                                   if f != "sources" else None)
                               for f in ref.TickInputs.__dataclass_fields__})
        # local I and the noise use MAP positions in GL; the twin's curl
        # noise is grid-relative -> shift its modes' phases to the box origin
        kx, ky, om, ph, amp = vfx.curl_modes()
        modes = (kx, ky, om, ph + kx * x0 + ky * y0, amp)
        H, W = h * m, w * m
        notes += [label + ":" + n for n in _pass_parity(
            vfx, tk, d, modes, H, W, np.random.default_rng(11), t, t_base,
            cs, label, tick_full.blocked)]
    vfx.unload()
    return "; ".join(notes)


# ---------------------------------------------------------------------------
# the loop over time: invariants and statistics only (§13.9)
# ---------------------------------------------------------------------------

def _proto_scenes(scales, which):
    import fire_vfx_proto as proto
    return proto.build_scenes(scales, which)


class _GLScene:
    """One F0b scene run through the GL loop (the whole scene = one box)."""

    def __init__(self, scene, scales, dials=None):
        from renderer.fire_vfx import FireVFXSettings, pack_tile_inputs
        from renderer.blackbody import BlackbodyRamp
        from renderer.fire_vfx import FireVFXPass
        from config import CFG
        from renderer import fire_vfx_ref as ref
        h, w = scene.fire.shape
        s = FireVFXSettings(dials=dials or ref.FireVFXDials())
        self.vfx = FireVFXPass(h, w, s, BlackbodyRamp.from_config(CFG))
        self.d = self.vfx.settings.dials
        self.t = 0.0
        pack = pack_tile_inputs(scene.fire, scene.t_game, scene.solid,
                                scene.flammable, scene.vacuum, scene.wind,
                                tps=scene.tps, tile_size_m=scene.tile_size_m,
                                scales=scales, dials=self.d, t_base=0.0)
        self.vfx.upload_pack(pack)
        _set_boxes(self.vfx, [(0, 0, w, h)])
        self.tick = scene.tick(scales, self.d)

    def run(self, n):
        from renderer.fire_vfx import plan_frame
        k = 0
        while k < n:
            b = min(64, n - k)
            times = [self.t + (j + 1) * self.d.dt_s for j in range(b)]
            self.vfx._run_plan(plan_frame(self.vfx.region.boxes, [], times,
                                          self.d.n_jacobi))
            self.t = times[-1]
            k += b

    def fields(self):
        v = _read(self.vfx.vel.front)
        f = _read(self.vfx.flame.front)
        return v, f

    def close(self):
        self.vfx.unload()


def _invariants(v, f, tick, d, label):
    assert np.isfinite(v).all() and np.isfinite(f).all(), f"{label}: NaN/Inf"
    R, RT = f[..., 0], f[..., 1]
    assert R.min() >= 0.0, f"{label}: R < 0 ({R.min()})"
    assert not R[tick.blocked].any(), f"{label}: flame in solid/vacuum"
    T = np.where(R > d.r_eps, RT / np.maximum(R, d.r_eps), 0.0)
    tmax = float(T[R > 1e-4].max()) if (R > 1e-4).any() else 0.0
    tsrc = float(tick.t_src_k.max())
    assert tmax <= tsrc * (1 + 1e-5) + 1e-3, f"{label}: T {tmax} > T_src {tsrc}"
    return tmax, tsrc


@check("long_run_invariants")
def check_long_run():
    """2.6 s of the full loop on the F0b scenes, GL and twin side by side
    (not in lockstep -- chaotic by design): invariants every 0.25 s, and the
    total flame R averaged over the last second within 10 % of the twin's."""
    from renderer import fire_vfx_ref as ref
    scales = _scales()
    notes = []
    for scene in _proto_scenes(scales, "abcdef"):
        g = _GLScene(scene, scales)
        d, tick = g.d, g.tick
        modes = ref.curl_noise_modes(d)
        st = ref.FireVFXState.zeros(*tick.s.shape)
        n_total = int(round(2.6 / d.dt_s))
        every = 30
        r_gl, r_tw = [], []
        for k in range(0, n_total, every):
            g.run(every)
            for _ in range(every):
                st = ref.substep(st, tick, d, modes)
            v, f = g.fields()
            tmax, tsrc = _invariants(v, f, tick, d, scene.name)
            if g.t >= 1.6:
                r_gl.append(f[..., 0].sum())
                r_tw.append(st.R.sum())
        a, b = float(np.mean(r_gl)), float(np.mean(r_tw))
        rel = abs(a - b) / max(b, 1e-9)
        assert rel < 0.10, f"{scene.name}: total R GL {a:.1f} vs twin {b:.1f} ({rel:.1%})"
        notes.append(f"{scene.name.split('_')[0]} R {rel:+.1%} T<={tmax:.0f}/{tsrc:.0f}")
        g.close()
    return "; ".join(notes)


@check("sealed_room_bounded")
def check_sealed_room():
    """10^4 GL substeps of the sealed 6x6 room: max|p| finite and not growing
    between the middle and the end of the run (the §13.1 F0b gate, in GL)."""
    scales = _scales()
    (scene,) = _proto_scenes(scales, "d")
    g = _GLScene(scene, scales)
    pm = []
    for _ in range(20):
        g.run(500)
        v, f = g.fields()
        _invariants(v, f, g.tick, g.d, "sealed")
        pm.append(float(np.abs(v[..., 2]).max()))
    g.close()
    pm = np.array(pm)
    mid, end = pm[6:12].max(), pm[14:].max()
    assert end <= 1.25 * mid + 1e-6, (mid, end)
    return f"max|p| mid {mid:.3f}, end {end:.3f} over 10^4 substeps"


@check("ten_minutes_finite")
def check_ten_minutes():
    """72 000 substeps (10 min of game at 1/120 s) of the studio's real fire
    row: finite, R >= 0, T <= T_src, p and u bounded throughout."""
    scales = _scales()
    (scene,) = _proto_scenes(scales, "e")
    g = _GLScene(scene, scales)
    n = int(round(600.0 / g.d.dt_s))
    umax = pmax = 0.0
    for _ in range(n // 6000):
        g.run(6000)
        v, f = g.fields()
        _invariants(v, f, g.tick, g.d, f"t={g.t:.0f}s")
        umax = max(umax, float(np.hypot(v[..., 0], v[..., 1]).max()))
        pmax = max(pmax, float(np.abs(v[..., 2]).max()))
    g.close()
    assert umax < 100.0 and pmax < 100.0, (umax, pmax)
    cfl = umax * g.d.dt_s * g.d.m
    return (f"{n} substeps, t_noise {g.t:.0f} s; max|u| {umax:.2f} tiles/s "
            f"(CFL {cfl:.2f} cells/substep), max|p| {pmax:.3f}")


# ---------------------------------------------------------------------------
# the region, the frame plan, reload (through update + simulate)
# ---------------------------------------------------------------------------

class _Frames:
    """Drive a FireVFXPass through update + simulate, one tick per frame."""

    def __init__(self, vfx, scales, h, w):
        self.vfx, self.scales = vfx, scales
        self.hot = _t_game_for_kelvin(scales, 1250.0)
        self.z = np.zeros((h, w), bool)
        self.flam = np.ones((h, w), bool)
        self.wind = np.zeros((h, w, 2), np.float32)
        self.wind[..., 0] = 0.02
        self.tick = 0

    def frame(self, fire):
        tg = np.where(fire > 0, self.hot, 0.0)
        self.vfx.update(fire=fire, t_game=tg, solid=self.z,
                        flammable=self.flam, vacuum=self.z,
                        wind_fn=lambda: self.wind, sim_tick=self.tick,
                        alpha=0.0, paused=False, tps=24.0, tile_size_m=0.333,
                        scales=self.scales)
        n = self.vfx.simulate()
        self.tick += 1
        return n


@check("region_clear")
def check_region_clear():
    from renderer.fire_vfx import linger_s
    scales = _scales()
    h, w = 24, 40
    vfx = _vfx(h, w)
    d = vfx.settings.dials
    fr = _Frames(vfx, scales, h, w)
    fa = np.zeros((h, w))
    fa[5, 5] = 1.0
    for _ in range(30):
        fr.frame(fa)
    box_a = list(vfx.region.boxes)
    ra = _read(vfx.flame.front)[..., 0]
    assert ra.max() > 0.1, f"no flame grew at A (max R {ra.max()})"
    fb = np.zeros((h, w))
    fb[18, 32] = 1.0
    for _ in range(int(np.ceil(linger_s(d) * 24.0)) + 3):
        fr.frame(fb)
    assert all(b not in vfx.region.boxes for b in box_a), "A's box survived"
    m = d.m
    for f in (vfx.flame, vfx.vel):
        for rt in f._rts:
            a = _read(rt.texture)
            for x0, y0, x1, y1 in box_a:
                assert not np.any(a[y0 * m:y1 * m, x0 * m:x1 * m]), \
                    "a departed tile kept data"
    rb = _read(vfx.flame.front)[..., 0]
    assert rb[18 * m:19 * m, 32 * m:33 * m].max() > 0.1, "no flame at B"
    out = f"A boxes {box_a} cleared; B boxes {vfx.region.boxes}"
    vfx.unload()
    return out


@check("burnout_leaves_nothing")
def check_burnout():
    """A fire burns, then goes out: within its linger the flame fades to
    exactly 0 in both buffers of both fields, the region empties and the
    frame plan is empty."""
    from renderer.fire_vfx import linger_s
    scales = _scales()
    h, w = 20, 20
    vfx = _vfx(h, w)
    d = vfx.settings.dials
    fr = _Frames(vfx, scales, h, w)
    f = np.zeros((h, w))
    f[9, 9] = f[9, 10] = 1.0
    for _ in range(40):
        fr.frame(f)
    assert _read(vfx.flame.front)[..., 0].max() > 0.1
    n_out = int(np.ceil(linger_s(d) * 24.0)) + 4
    for _ in range(n_out):
        fr.frame(np.zeros((h, w)))
    assert not vfx.region.active, f"region still {vfx.region.boxes}"
    for fld in (vfx.flame, vfx.vel):
        for rt in fld._rts:
            assert not np.any(_read(rt.texture)), "stale field data after burn-out"
    assert fr.frame(np.zeros((h, w))) == 0, "passes after burn-out"
    vfx.unload()
    return f"flame gone and region empty {n_out} frames after the fire went out"


@check("box_growth")
def check_box_growth():
    """A second fire lights 6 tiles away mid-run: the boxes grow / merge, no
    NaN, and no flame or velocity outside the new domain."""
    scales = _scales()
    h, w = 24, 30
    vfx = _vfx(h, w)
    fr = _Frames(vfx, scales, h, w)
    f = np.zeros((h, w))
    f[10, 8] = 1.0
    for _ in range(30):
        fr.frame(f)
    before = list(vfx.region.boxes)
    f[12, 15] = 1.0
    for _ in range(30):
        fr.frame(f)
    after = list(vfx.region.boxes)
    assert after != before
    m = vfx.settings.m
    dom = np.repeat(np.repeat(vfx.region.domain, m, 0), m, 1)
    for fld in (vfx.flame, vfx.vel):
        a = _read(fld.front)
        assert np.isfinite(a).all()
        assert not a[~dom].any(), "field data outside the domain"
    R = _read(vfx.flame.front)[..., 0]
    assert R[12 * m:13 * m, 15 * m:16 * m].max() > 0.1
    vfx.unload()
    return f"boxes {before} -> {after}"


@check("realloc_on_grid_change")
def check_realloc():
    """Property: changing [render.fire_vfx] m (live config) reallocates the
    fields at the new resolution, zeroed; disabling empties the plan.
    Breaks if a resolution change keeps the old RTs or stale data."""
    from config import CFG
    from renderer.blackbody import BlackbodyRamp
    from renderer.fire_vfx import FireVFXPass
    v = FireVFXPass.from_config(10, 12, CFG, BlackbodyRamp.from_config(CFG))
    try:
        m0 = v.settings.m
        _put(v.flame.front, np.ones((10 * m0, 12 * m0, 4), np.float32))
        CFG.render.fire_vfx.m = max(2, m0 // 2)
        CFG.render.fire_vfx.m_u = max(2, m0 // 2)
        v.refresh_settings(CFG)
        m1 = v.settings.m
        assert (v.flame.width, v.flame.height) == (12 * m1, 10 * m1)
        assert v.vel.width == 12 * m1
        assert not _read(v.flame.front).any()
        CFG.render.fire_vfx.enabled = False
        v.refresh_settings(CFG)
        assert v._plan == [] and v.simulate() == 0
    finally:
        CFG.reload()
        v.unload()
    return f"m {m0} -> {m1}"


@check("config_reload_live")
def check_reload():
    """A live dial change reaches the next frame without reallocating: a
    smaller n_jacobi shrinks the pass count at once; a tiny lifetime_s makes
    the burning flame collapse within a frame -- and the field textures are
    the same objects (no realloc)."""
    from config import CFG
    from renderer.blackbody import BlackbodyRamp
    from renderer.fire_vfx import FireVFXPass, substep_passes
    scales = _scales()
    h, w = 16, 16
    v = FireVFXPass.from_config(h, w, CFG, BlackbodyRamp.from_config(CFG))
    fr = _Frames(v, scales, h, w)
    f = np.zeros((h, w))
    f[8, 8] = 1.0
    try:
        for _ in range(20):
            v.refresh_settings(CFG)
            n0 = fr.frame(f)
        ids = (v.vel.front_id, v.flame.front_id)
        r0 = _read(v.flame.front)[..., 0].sum()
        CFG.render.fire_vfx.n_jacobi = 4
        CFG.render.fire_vfx.lifetime_s = 0.004
        v.refresh_settings(CFG)
        n1 = fr.frame(f)
        r1 = _read(v.flame.front)[..., 0].sum()
        assert n1 == 5 * len(substep_passes(4)), n1
        assert n0 == 5 * len(substep_passes(20)), n0
        assert {v.vel.front_id, v.flame.front_id} <= set(
            i for pp in (v.vel, v.flame) for _, i in pp._ids)
        assert set(ids) <= set(i for pp in (v.vel, v.flame) for _, i in pp._ids)
        assert r1 < 0.2 * r0, (r0, r1)
    finally:
        CFG.reload()
        v.unload()
    return f"passes {n0} -> {n1}, total R {r0:.0f} -> {r1:.1f}"


@check("no_fire_no_passes")
def check_no_fire():
    from renderer import core
    scales = _scales()
    h, w = 16, 16
    vfx = _vfx(h, w)
    z = np.zeros((h, w), bool)
    calls = []
    orig = core.DirectPasses.__enter__
    core.DirectPasses.__enter__ = lambda self: (calls.append(1), orig(self))[1]
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
        core.DirectPasses.__enter__ = orig
    assert not calls, f"{len(calls)} GL blocks without fire"
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
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_fire_vfx_check")
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
    print(f"FIRE_VFX_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
