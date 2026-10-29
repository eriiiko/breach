"""GL checks of the flame VFX pass (graphics arc G3, patches F1 + F2 + F3 + F4a;
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
  flame_draw_cover_and_bound -- (F3) the flame stage of shaders/fire_glow.fs
      on a live field over a dim lit floor: the destination alpha is never
      written; no pixel outshines the feeder's Le(T_src) (round 1's M4);
      the floor is never darkened by more than 10 % (the cover is tied to
      the emission, §13.4); A "mask" and B "thickness" differ by > 5 % of
      the flame's RGB (§8). Breaks
      if the cover decouples from emission (dark tips), the flame
      double-counts over its fuel, a stage writes alpha, or the A/B goes
      meaningless.
  no_fire_no_flame_draws -- (F3) no fire -> no flame draw rects and no
      draw issued; a burnt-out fire stops drawing after its linger. Breaks
      if the draw ignores the active region.
  one_tamed_wind_and_render_only -- the real renderer on fire_smoke_studio
      (relief + props + flame + embers + shimmer all consuming the wind)
      calls ``tame_wind`` at most once per frame, and the field digest of
      every tick is identical with the flame VFX, its embers and the heat
      shimmer on and off; the embers really flew and the shimmer really ran
      in the "on" run. Breaks if a consumer re-tames the wind or any VFX
      code writes sim state.
  ember_draw_additive -- (F4a) THE particle primitive's sprite draw over a
      lit floor: the destination alpha is never written, light is only ever
      ADDED, it lands where the particle is (world px, y down -- the scene
      RT's raylib space) and nowhere else, and its core never outshines
      ``Le(T) x weight`` (the one ramp, through the LUT). Breaks if the
      quad's winding is culled (nothing drawn), the y axis flips, the blend
      writes alpha, or the colour leaves the LUT.
  no_fire_no_ember_draws -- (F4a) no fire -> no ember, no sprite, no draw
      call; a fire makes embers that draw. Breaks if the embers spawn from
      anything but the flame VFX's sources or draw with nothing to draw.
  shimmer_zero_where_cold -- (F4a) the resolve with the heat shimmer on: a
      cold map resolves BIT-IDENTICAL to the shimmer off; one hot tile
      changes pixels only within its 2x2 bilinear footprint (+ the offset),
      and does change some there. Breaks if the shimmer leaks to cold areas,
      shifts the whole screen, or a zero offset stops being the plain fetch.
  smoke_draws_with_fire_layers_on -- (G3 bug fix 2026-10-29, Erik: "I can't
      see any smoke") the real renderer on fire_smoke_studio at 20 s, paused,
      with the flame, embers, shimmer and the room-light flicker all ON (the
      flicker through F4b's paused relight): the ceiling soot is packed and
      darkens the downwind box past the fire row to < 85 % of the same frame
      with the smoke off (F2). Breaks if any fire layer draws over, hides or
      stops the smoke relief (draw order, a cover or compensation pass
      writing over the soot, a key swallowing F2).

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


def _GLScene(scene, scales, dials=None):
    """The proto tool's GL runner (``tools/fire_vfx_proto.GLScene``: the
    whole scene = one box), plus the twin's tick for the invariants."""
    import fire_vfx_proto as proto
    from renderer import fire_vfx_ref as ref
    g = proto.GLScene(scene, scales, dials or ref.FireVFXDials())
    g.tick = scene.tick(scales, g.d)
    return g


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


# ---------------------------------------------------------------------------
# the draw (F3): shaders/fire_glow.fs through FireVFXPass.draw_glow
# ---------------------------------------------------------------------------

_UNDER = (8, 6, 5, 128)          # a dim lit floor (8-bit), half coverage


def _draw_flame(vfx, rt, law, px_per_tile, stage="flame"):
    """Clear ``rt`` to the floor, draw one fire-glow stage under ``law``
    (A "mask" / B "thickness"), read it back (float, GL row order)."""
    from renderer import core
    vfx.emission_override = law
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.Color(*_UNDER))
    n = vfx.draw_glow(stage, vfx.flame_rects(None),
                      world_px_w=vfx.w * px_per_tile,
                      world_px_h=vfx.h * px_per_tile)
    rl.end_texture_mode()
    return n, core.read_texture_rgba32f(rt.texture).astype(np.float64)


def _burning_row(vfx, scales, h, w, frames=48):
    fr = _Frames(vfx, scales, h, w)
    fire = np.zeros((h, w))
    fire[h // 2, w // 2 - 2:w // 2 + 1] = 1.0
    for _ in range(frames):
        fr.frame(fire)
    return fr


@check("flame_draw_cover_and_bound")
def check_flame_draw():
    """The flame stage on a live field over a dim lit floor: it never
    writes the destination alpha, never shows a pixel brighter than its
    feeder's ``Le(T_src)`` (M4), never darkens the floor by more than a
    tolerance (eps tied to emission, §13.4), and A != B."""
    from renderer import core
    scales = _scales()
    h, w, ppt = 16, 16, 8
    vfx = _vfx(h, w)
    fr = _burning_row(vfx, scales, h, w)
    rt = core.load_render_texture_rgba16f(w * ppt, h * ppt)
    under = np.array(_UNDER[:3]) / 255.0
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    t_src = 1250.0
    ch, it = scales.ramp.emission_at_kelvin(np.array([t_src]))
    le_src = (ch * it[:, None])[0]
    out = {}
    for law in ("mask", "thickness"):
        n, img = _draw_flame(vfx, rt, law, ppt)
        assert n > 0, f"{law}: no flame quad drawn with a fire burning"
        a = img[..., 3]
        assert np.allclose(a, _UNDER[3] / 255.0, atol=2e-3),             f"{law}: destination alpha changed ({a.min():.4f}..{a.max():.4f})"
        rgb = img[..., :3]
        cap = np.maximum(under, le_src) * 1.01 + 2e-3
        assert np.all(rgb <= cap), (f"{law}: a pixel outshines its feeder: "
                                    f"max {rgb.max(axis=(0, 1))} > {cap}")
        lum = rgb @ lum_w
        lu = under @ lum_w
        dark = (lu - lum) / lu
        assert dark.max() <= 0.10, (f"{law}: the flame darkens the floor by "
                                    f"{100 * dark.max():.1f} % somewhere")
        lit = lum > lu * 1.5
        assert lit.sum() > 200, f"{law}: almost no flame drawn ({lit.sum()} px)"
        out[law] = (rgb, dark.max(), lit.sum())
    diff = np.abs(out["mask"][0] - out["thickness"][0]).sum(axis=-1)
    shown = (out["mask"][0] @ lum_w > (under @ lum_w) * 1.5) |             (out["thickness"][0] @ lum_w > (under @ lum_w) * 1.5)
    md = float(diff[shown].mean())
    mean_rgb = float(np.maximum(out["mask"][0], out["thickness"][0])
                     .sum(axis=-1)[shown].mean())
    rel = md / max(mean_rgb, 1e-9)
    assert rel > 0.05, (f"A and B look alike on a burning row (mean |dRGB| "
                        f"{md:.3f} = {100 * rel:.1f} % of the flame's RGB)")
    rl.unload_render_texture(rt)
    vfx.unload()
    return (f"A: max darken {100 * out['mask'][1]:.2f} %, {out['mask'][2]} px "
            f"lit; B: {100 * out['thickness'][1]:.2f} %, "
            f"{out['thickness'][2]} px; |A-B| {100 * rel:.1f} % of RGB")


@check("no_fire_no_flame_draws")
def check_no_fire_draws():
    """With no fire the flame has no draw rects and draw_glow issues
    nothing, and a fire that burned out stops drawing after its linger."""
    scales = _scales()
    h, w = 16, 16
    vfx = _vfx(h, w)
    fr = _Frames(vfx, scales, h, w)
    for _ in range(5):
        fr.frame(np.zeros((h, w)))
    assert vfx.flame_rects(None) == [], vfx.flame_rects(None)
    assert vfx.draw_glow("flame", vfx.flame_rects(None), world_px_w=128,
                         world_px_h=128) == 0
    fire = np.zeros((h, w))
    fire[8, 8] = 1.0
    for _ in range(24):
        fr.frame(fire)
    assert vfx.flame_rects(None), "a burning tile drew no flame rects"
    from renderer.fire_vfx import linger_s
    for _ in range(int(24 * (linger_s(vfx.settings.dials) + 1.0))):
        fr.frame(np.zeros((h, w)))
    assert vfx.flame_rects(None) == [], "a burnt-out fire still draws"
    vfx.unload()


# ---------------------------------------------------------------------------
# F4a: the particle primitive (embers) and the heat shimmer
# ---------------------------------------------------------------------------

def _lut_tex(ramp):
    from renderer import core
    from renderer.fire_vfx import LUT_SIZE, blackbody_lut_data
    lut = core.create_dynamic_rgba32f_texture(LUT_SIZE, 1)
    rl.set_texture_filter(lut, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    core.update_rgba32f_texture(lut, blackbody_lut_data(ramp))
    return lut


@check("ember_draw_additive")
def check_ember_draw():
    """Sprites over a lit floor: alpha kept, light only added, at the right
    place, never brighter than Le(T) x weight."""
    from renderer import core
    from renderer.fire_vfx import LUT_SIZE
    from renderer.particles import ParticleDraw, ember_radiance
    scales = _scales()
    ramp = scales.ramp
    lut = _lut_tex(ramp)
    W, H = 256, 128
    rt = core.load_render_texture_rgba16f(W, H)
    pd = ParticleDraw(8)
    spr = np.zeros((2, 2, 4), np.float32)
    spr[0, 0] = (64.0, 32.0, 0.0, 0.0)          # still, upper left
    spr[0, 1] = (180.0, 96.0, -400.0, 0.0)      # fast, moving west
    spr[1, 0] = (1250.0, 5.0, 6.0, 0.06)
    spr[1, 1] = (1000.0, 5.0, 4.0, 0.06)
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.Color(*_UNDER))
    n = pd.draw(spr, world_px_w=W, world_px_h=H, lut_tex=lut,
                lut_k=(ramp.kelvin_floor, ramp.kelvin_ceil), lut_n=LUT_SIZE)
    rl.end_texture_mode()
    img = core.read_texture_rgba32f(rt.texture).astype(np.float64)
    under = np.array(_UNDER) / 255.0
    assert n == 1, f"{n} draw calls for one sprite batch"
    assert np.allclose(img[..., 3], under[3], atol=2e-3), "alpha written"
    assert np.all(img[..., :3] >= under[:3] - 2e-3), "a sprite darkened"
    added = (img[..., :3] - under[:3]).sum(axis=-1)
    # GL rows run up: world y -> GL row H - y
    lit = np.argwhere(added > 1e-3)
    rows_world = H - 1 - lit[:, 0]
    a = (np.abs(lit[:, 1] - 64) <= 7) & (np.abs(rows_world - 32) <= 7)
    # the moving one (west at 400 px/s, streak 0.06 s): its 24 px trail
    # lies BEHIND it, to the east
    b = ((lit[:, 1] >= 180 - 5) & (lit[:, 1] <= 180 + 24 + 5)
         & (np.abs(rows_world - 96) <= 5))
    assert a.any() and b.any(), "a sprite did not land where it is"
    assert np.all(a | b), "light added away from every sprite"
    cols_b = lit[b][:, 1]
    assert cols_b.min() >= 175 and cols_b.max() > 195,         "the motion trail is not behind the moving sprite"
    le = ember_radiance(np.array([1250.0]), np.array([5.0]), ramp)[0]
    assert np.all(img[..., :3] - under[:3] <= le * 1.02 + 2e-3), \
        "a sprite outshines Le(T) x weight"
    rl.unload_render_texture(rt)
    rl.unload_texture(lut)
    pd.unload()
    return f"{len(lit)} px lit, peak {added.max():.2f}"


@check("no_fire_no_ember_draws")
def check_no_fire_embers():
    """No fire -> no ember, no sprite, no draw call; a fire -> embers."""
    from renderer.particles import (EmberSystem, ParticleDraw, pack_sprites,
                                    settings_from_config)
    from renderer.fire_vfx import LUT_SIZE
    from config import CFG
    scales = _scales()
    h, w = 16, 16
    vfx = _vfx(h, w)
    fr = _Frames(vfx, scales, h, w)
    es = EmberSystem(settings_from_config(CFG), t_amb_k=scales.t_amb_k,
                     glow_min_k=scales.glow_min_k)
    pd = ParticleDraw(64)
    lut = _lut_tex(scales.ramp)
    blocked = np.zeros((h, w), bool)

    def step(fire):
        fr.frame(fire)
        pk = vfx.pack
        es.update(t=vfx.clock.t_noise,
                  sources=pk.sources if pk is not None else None,
                  origin=(pk.window[1], pk.window[0]) if pk else (0, 0),
                  s0=vfx.settings.dials.s0,
                  strip_frac=vfx.settings.dials.strip_frac,
                  wind_fn=lambda: fr.wind, tps=24.0, blocked=blocked)
        spr = pack_sprites(es.p, None, 8.0, es.settings)
        return pd.draw(spr, world_px_w=w * 8, world_px_h=h * 8,
                       lut_tex=lut, lut_k=(scales.ramp.kelvin_floor,
                                           scales.ramp.kelvin_ceil),
                       lut_n=LUT_SIZE)

    for _ in range(48):
        assert step(np.zeros((h, w))) == 0
        assert es.p.n == 0
    fire = np.zeros((h, w))
    fire[8, 6:10] = 1.0
    drew = sum(step(fire) for _ in range(72))
    assert es.p.n > 0 and drew > 0, "a burning row made no ember draw"
    rl.unload_texture(lut)
    pd.unload()
    vfx.unload()
    return f"{es.p.n} embers alive, {drew} draws over 3 s of fire"


@check("shimmer_zero_where_cold")
def check_shimmer():
    """The resolve with the shimmer: cold -> bit-identical; one hot tile
    -> changes only inside its footprint, and some there."""
    from renderer import core
    from renderer.heat_shimmer import HeatShimmer, ShimmerSettings
    from renderer.world_composite import WorldComposite
    h, w, ppt = 12, 16, 16
    wc = WorldComposite(w, h, ppt)
    st = ShimmerSettings(amp_tiles=0.2, zoom_zero=0.0, zoom_full=1e-3)
    sh = HeatShimmer(h, w, st)
    rng = np.random.default_rng(3)
    floor = (rng.random((h * ppt, w * ppt, 4)) * 0.8).astype(np.float32)
    floor[..., 3] = 1.0
    tex = core.create_dynamic_rgba32f_texture(w * ppt, h * ppt)
    core.update_rgba32f_texture(tex, floor)
    z = np.zeros((h, w), bool)

    def frame(tk, shimmer):
        sh.update(t_kelvin_fn=lambda: tk, t_game=tk, solid=z, vacuum=z,
                  wind_fn=lambda: np.zeros((h, w, 2), np.float32), tps=24.0,
                  t_noise=1.37, zoom_px_per_tile=64.0)
        wc.begin(clear_color=(0, 0, 0, 0))
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.draw_texture(tex, 0, 0, rl.WHITE)
        rl.end_blend_mode()
        wc.resolve(shimmer=sh if shimmer else None)
        wc.end()
        img = rl.load_image_from_texture(wc.rt.texture)
        n = img.width * img.height * 4
        a = np.frombuffer(rl.ffi.buffer(img.data, n), np.uint8).reshape(
            img.height, img.width, 4).copy()
        rl.unload_image(img)
        return a

    cold = np.full((h, w), 293.0)
    off = frame(cold, False)
    on = frame(cold, True)
    assert sh.active(ppt) is False and not sh.hot
    assert np.array_equal(off, on), "a cold map resolved differently"
    hot = cold.copy()
    hot[6, 9] = 900.0
    on = frame(hot, True)
    assert sh.active(ppt)
    diff = np.abs(on.astype(int) - off.astype(int)).sum(axis=-1) > 0
    ys, xs = np.nonzero(diff)
    assert len(ys) > 50, f"the hot tile shimmered only {len(ys)} px"
    # GL rows run up; the footprint is the 2x2 tile block around the hot
    # tile's centre (8 px/tile..), plus the offset (0.2 tiles)
    wy = h * ppt - 1 - ys
    cx, cy = 9.5 * ppt, 6.5 * ppt
    reach = ppt * 1.0 + st.amp_tiles * ppt + 1.0
    assert np.all(np.abs(xs + 0.5 - cx) <= reach)
    assert np.all(np.abs(wy + 0.5 - cy) <= reach), "shimmer leaked away"
    rl.unload_texture(tex)
    sh.unload()
    wc.unload()
    return f"cold bit-identical; hot tile moved {len(ys)} px, all local"



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
    embers = 0
    shimmer_hot = False
    try:
        for on in (True, False):
            CFG.render.fire_vfx.enabled = on
            CFG.render.embers.enabled = on
            CFG.render.shimmer.enabled = on
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
                    embers = max(embers, renderer.embers.p.n)
                    shimmer_hot |= renderer.shimmer.hot
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
    assert embers > 0, "no ember flew on a burning studio"
    assert shimmer_hot, "the shimmer never saw hot gas on a burning studio"
    assert digests[True] == digests[False], "field digests differ VFX on/off"
    return (f"{n_ticks} ticks, tame_wind {min(counts)}..{max(counts)}/frame, "
            f"{passes} VFX passes, {embers} embers max, digests equal")



@check("smoke_draws_with_fire_layers_on")
def check_smoke_draws_with_fire_layers():
    import breach_physics as bp
    from config import CFG
    from level_loader import load as load_level
    from renderer import core, frame_lights
    from renderer.game_renderer import GameRenderer, RenderConfig
    from simulation import Simulation

    level = load_level("fire_smoke_studio")
    ppt = 8.0
    cfg = RenderConfig(map_px_w=448, map_px_h=448, panel_px_w=0,
                       grid_w=level.width, grid_h=level.height,
                       world_px_per_tile=ppt)
    renderer = GameRenderer(level, bp, cfg)
    for sec in ("fire_vfx", "embers", "shimmer", "flicker"):
        getattr(CFG.render, sec).enabled = True
    sky = frame_lights.sky_for_level(getattr(level, "boundary", "space"),
                                     CFG, bp)
    # The downwind soot box (fire_smoke_studio: the fire row burns at
    # y 22, x 20-25 and the flow carries its smoke east towards the vent;
    # tools/fire_smoke_probe.py) -- past the burning tiles, where the flame
    # does not draw. At 20 s it measures ~67 % of the smoke-off luminance.
    bx0, bx1, by0, by1 = 27, 33, 20, 25
    n_ticks = 480                       # 20 s: a plume, fires still burning
    try:
        sim = Simulation(level, seed=42, breach_physics=bp,
                         enable_recorder=False)
        sim.set_light(True, None, sky, None)
        for tick in range(n_ticks):
            if sim.is_paused():
                sim.set_paused(False)
            sim.step()
        sim.set_paused(True)

        def frame(i, smoke_on):
            renderer.show_smoke = smoke_on
            rows = renderer.fire_light_rows(level.width, level.height)
            sim.set_light(True, None, sky, rows)          # F4b's paused path
            sim.relight()
            renderer.upload_state(sim.gmap, sim_tick=n_ticks,
                                  light_serial=sim.light_serial,
                                  alpha=0.0, paused=True)
            renderer.begin_frame()
            renderer.compose_world()
            renderer.end_frame()
            img = core.read_texture_rgba32f(renderer.world.rt.texture)
            y0 = img.shape[0] - int(by1 * ppt)       # GL row 0 = bottom
            y1 = img.shape[0] - int(by0 * ppt)
            box = img[y0:y1, int(bx0 * ppt):int(bx1 * ppt), :3]
            return float(np.median(box.mean(axis=-1)))

        for i in range(30):               # the flame, embers and shimmer run
            frame(i, True)
        lum_on = frame(30, True)
        has = dict(renderer.gas_relief._has)
        lum_off = frame(31, False)
        flame = renderer.fire_vfx.passes_last_frame
        embers = renderer.embers.p.n
        flick = renderer.fire_flicker.last_n
    finally:
        CFG.reload()
        renderer.shutdown()
    assert flame > 0 and embers > 0 and flick > 0, (
        f"a fire layer did not run (passes {flame}, embers {embers}, "
        f"flickering tiles {flick})")
    assert has.get("ceiling"), "the relief packed no ceiling soot"
    assert lum_on < 0.85 * lum_off, (
        f"the soot does not show downwind: median luminance {lum_on:.4f} "
        f"with the smoke vs {lum_off:.4f} without (needs < 85 %)")
    return (f"downwind luminance {lum_on:.4f} with smoke vs {lum_off:.4f} "
            f"without ({lum_on / max(lum_off, 1e-9):.0%}); passes {flame}, "
            f"embers {embers}, flickering tiles {flick}")


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
