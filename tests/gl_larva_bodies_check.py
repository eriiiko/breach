"""GL checks of the swarm larvae's bodies (arc #63,
``renderer/larva_bodies.py``, ``shaders/larva_body.vs/.fs``).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_gl_larva_bodies.py`` spawns it and
asserts on its last line (the ``tests/cuda_harness.py`` convention, for GL).
Prints one ``[check] NAME: PASS|FAIL detail`` line per check and finally
``LARVA_GL_RESULT: PASS`` or ``... FAIL``. ``--only a,b`` runs a subset;
``--save DIR`` also writes a gamma-encoded close-up of a test brood there
(a look aid, not a gate).

Each check names the PROPERTY it protects and the change that must break it:

  one_draw_whatever_the_count -- 1, 40 and 3 000 larvae are each ONE
      ``glDrawArraysInstanced`` and no other draw, and every larva lands at
      its own place (the last of 3 000, past the instance texture's first
      row, is drawn where its data says). Breaks if the draw loops per larva
      or per row, falls back to ``glDrawArrays``, or the instance texture's
      row layout and the shader's fetch disagree.
  a_larva_is_drawn_and_only_where_it_lies -- one larva covers pixels at its
      position and none a body length away. Breaks if the winding is culled,
      the y axis flips, or the scale (px per metre) is off.
  the_pose_is_a_function_of_the_phase -- the same instance drawn twice is
      bit-identical (no clock in the shader: a still larva is still), and
      another phase changes the picture. Breaks if the shader reads a time
      uniform or ignores the phase.
  lit_by_the_light_field -- no light and no ambient draws black (no
      emission); light in the field's texture brightens the body; twice the
      light, about twice the body. Breaks if the body is emissive, reads a
      constant instead of ``light_tex_a``, or tone-maps itself.
  a_faded_body_leaves_the_floor_untouched -- an instance at alpha 0 leaves
      every pixel of the target bit-identical; a limp, half-faded one darkens
      the floor less than a living one covers it. Breaks if the blend is not
      premultiplied or the alpha is ignored.
  disabled_costs_nothing -- with ``enabled = false`` the renderer-side state
      packs nothing and builds no GL object. Breaks if the off switch still
      reads the store or uploads.

Run alone:
    C:/Users/steen/anaconda3/envs/data/python.exe tests/gl_larva_bodies_check.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pyray as rl  # noqa: E402

RESULTS = []
CHECKS = []

GW, GH, PPT = 16, 8, 64          # grid tiles, world px per tile
TILE_M = 0.333
WPX, HPX = GW * PPT, GH * PPT


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


class _Cfg:
    """A stand-in config: ``render.larvae`` from the real config.toml."""

    def __init__(self, **over):
        import tomllib
        raw = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
        sec = dict(raw["render"]["larvae"])
        sec.update(over)
        self.render = type("R", (), {})()
        self.render.larvae = type("L", (), sec)()


def _settings(**over):
    from renderer.larva_bodies import larva_settings_from_config
    return larva_settings_from_config(_Cfg(**over))


_STATE = {}


def _light(level: float, ambient: float = 0.0):
    """A LightFieldCtx whose field is ``level`` grey light everywhere,
    arriving from straight above."""
    from renderer import core
    from renderer.lit3d import LightFieldCtx
    if "tex" not in _STATE:
        _STATE["tex"] = (core.create_dynamic_rgba16f_texture(GW, GH),
                         core.create_dynamic_rgba16f_texture(GW, GH))
    ta, tb = _STATE["tex"]
    a = np.zeros((GH, GW, 4), np.float16)
    a[..., :3] = level
    a[..., 3] = 0.0
    b = np.zeros((GH, GW, 4), np.float16)
    core.update_rgba16f_texture(ta, a)
    core.update_rgba16f_texture(tb, b)
    return LightFieldCtx(tex_a=ta, tex_b=tb, world_px_w=float(WPX),
                         world_px_h=float(HPX),
                         ambient=(ambient, ambient, ambient), light_gain=1.0)


def _draw():
    from renderer.larva_bodies import LarvaBodyDraw
    if "draw" not in _STATE:
        _STATE["draw"] = LarvaBodyDraw()
    return _STATE["draw"]


def _rt():
    from renderer import core
    if "rt" not in _STATE:
        _STATE["rt"] = core.load_render_texture_rgba16f(WPX, HPX)
    return _STATE["rt"]


def _render(inst, s, light, floor=(0.0, 0.0, 0.0, 0.0), ppt=PPT):
    """Draw ``inst`` over a flat ``floor`` colour into the float RT and read
    it back as (HPX, WPX, 4) float32, row 0 = world y 0."""
    rt = _rt()
    rl.begin_texture_mode(rt)
    from renderer import core
    gl = core.GLDirect.get()
    gl.ClearColor(*floor)
    gl.Clear(0x00004000)
    n = _draw().draw(inst, s, light=light, tile_size_m=TILE_M,
                     px_per_tile=ppt)
    rl.end_texture_mode()
    img = rl.load_image_from_texture(rt.texture)
    if img.format == rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R16G16B16A16:
        raw = rl.ffi.buffer(img.data, WPX * HPX * 8)
        a = np.frombuffer(bytes(raw), np.float16).reshape(HPX, WPX, 4)
    else:
        raw = rl.ffi.buffer(img.data, WPX * HPX * 16)
        a = np.frombuffer(bytes(raw), np.float32).reshape(HPX, WPX, 4)
    rl.unload_image(img)
    # GL row 0 is NDC y = -1, the world's bottom (ndc.y = 1 - 2 y / H)
    return a[::-1].astype(np.float32), n


def _inst(rows, ppt=PPT):
    """Instances from (x_t, y_t, heading, phase, limp, alpha, coma, size)."""
    rows = np.asarray(rows, np.float32).reshape(-1, 8)
    out = np.zeros((2, rows.shape[0], 4), np.float32)
    out[0, :, 0] = rows[:, 0] * ppt
    out[0, :, 1] = rows[:, 1] * ppt
    out[0, :, 2:4] = rows[:, 2:4]
    out[1] = rows[:, 4:8]
    return out


def _covered(img):
    return img[..., 3] > 1e-3


@check("a_larva_is_drawn_and_only_where_it_lies")
def _drawn_where_it_lies():
    s = _settings()
    img, _ = _render(_inst([(4.0, 4.0, 0.0, 0.0, 0, 1, 0, 1)]), s, _light(1.0))
    cov = _covered(img)
    cx, cy = int(4.0 * PPT), int(4.0 * PPT)
    half_px = 0.5 * s.length_m / TILE_M * PPT
    ys, xs = np.nonzero(cov)
    assert cov[cy, cx], "no body at the larva's centre"
    assert xs.size > 50, f"only {xs.size} px covered"
    # heading 0 = +x: the body spans x, not y
    span_x, span_y = xs.max() - xs.min(), ys.max() - ys.min()
    assert abs(span_x - 2 * half_px) <= 0.2 * 2 * half_px + 2, (span_x, half_px)
    assert span_y < span_x * 0.7, (span_x, span_y)
    far = cov[:, int(cx + 3 * half_px):].any() or cov[:, :int(cx - 3 * half_px)].any()
    assert not far, "pixels a body length away"
    # heading +pi/2 = +y (y down): the body turns to span y
    img, _ = _render(_inst([(4.0, 4.0, np.pi / 2, 0.0, 0, 1, 0, 1)]), s,
                     _light(1.0))
    ys, xs = np.nonzero(_covered(img))
    assert ys.max() - ys.min() > xs.max() - xs.min()
    return f"{int(cov.sum())} px, span {span_x} px for {2 * half_px:.1f}"


@check("one_draw_whatever_the_count")
def _one_draw():
    from renderer import core
    s = _settings()
    gl = core.GLDirect.get()
    calls = {"inst": 0, "arrays": 0}
    real_i, real_a = gl.DrawArraysInstanced, gl.DrawArrays

    def count_i(*a):
        calls["inst"] += 1
        return real_i(*a)

    def count_a(*a):
        calls["arrays"] += 1
        return real_a(*a)

    gl.DrawArraysInstanced, gl.DrawArrays = count_i, count_a
    try:
        rng = np.random.default_rng(3)
        report = []
        for k in (1, 40, 3000):
            xy = rng.uniform(0.5, [GW - 0.5, GH - 0.5], size=(k, 2))
            rows = np.zeros((k, 8))
            rows[:, :2] = xy
            rows[:, 2] = rng.uniform(-np.pi, np.pi, k)
            rows[:, 3] = rng.uniform(0, 2 * np.pi, k)
            rows[:, 5] = rows[:, 7] = 1.0
            # the last larva alone in a corner, past the first texture row
            rows[-1, :2] = (GW - 0.6, GH - 0.6)
            calls.update(inst=0, arrays=0)
            img, n = _render(_inst(rows), s, _light(1.0))
            assert n == 1 and calls == {"inst": 1, "arrays": 0}, (k, n, calls)
            px, py = int((GW - 0.6) * PPT), int((GH - 0.6) * PPT)
            assert _covered(img)[py, px], f"larva {k - 1} of {k} not drawn"
            report.append(f"{k}:1")
    finally:
        gl.DrawArraysInstanced, gl.DrawArrays = real_i, real_a
    return ", ".join(report)


@check("the_pose_is_a_function_of_the_phase")
def _pose_phase():
    s = _settings()
    one = _inst([(4.0, 4.0, 0.3, 1.0, 0, 1, 0, 1)])
    a, _ = _render(one, s, _light(1.0))
    b, _ = _render(one, s, _light(1.0))
    assert np.array_equal(a, b), "the same phase drew two pictures"
    c, _ = _render(_inst([(4.0, 4.0, 0.3, 2.5, 0, 1, 0, 1)]), s, _light(1.0))
    d = np.abs(a - c).max(axis=-1)
    assert (d > 1e-3).sum() > 10, "another phase drew the same picture"
    return f"{int((d > 1e-3).sum())} px moved between phases 1.0 and 2.5"


@check("lit_by_the_light_field")
def _lit():
    s = _settings()
    one = _inst([(4.0, 4.0, 0.0, 0.0, 0, 1, 0, 1)])
    dark, _ = _render(one, s, _light(0.0, 0.0))
    cov = _covered(dark)
    assert cov.sum() > 50
    assert dark[cov][:, :3].max() <= 1e-6, "a body glows with no light"
    lo, _ = _render(one, s, _light(0.25))
    hi, _ = _render(one, s, _light(0.5))
    m_lo, m_hi = lo[cov][:, :3].mean(), hi[cov][:, :3].mean()
    assert m_lo > 1e-3 and 1.7 < m_hi / m_lo < 2.3, (m_lo, m_hi)
    return f"mean {m_lo:.4f} -> {m_hi:.4f} for light x2"


@check("a_faded_body_leaves_the_floor_untouched")
def _faded():
    s = _settings()
    floor = (0.2, 0.3, 0.4, 1.0)
    base, _ = _render(np.zeros((2, 0, 4), np.float32), s, _light(1.0), floor)
    gone, _ = _render(_inst([(4.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0, 1)]), s,
                      _light(1.0), floor)
    assert np.array_equal(base, gone), "an alpha-0 body changed the floor"
    live, _ = _render(_inst([(4.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0, 1)]), s,
                      _light(0.0), floor)
    half, _ = _render(_inst([(4.0, 4.0, 0.0, 0.0, 1.0, 0.5, 0, 1)]), s,
                      _light(0.0), floor)
    cov = np.abs(live - base).max(axis=-1) > 1e-3
    # unlit, the living body is black over the floor; half faded, half way
    r = (base[cov][:, :3] - half[cov][:, :3]).mean() / max(
        (base[cov][:, :3] - live[cov][:, :3]).mean(), 1e-6)
    assert 0.3 < r < 0.7, r
    return f"half-faded body darkens {r:.2f} of the living one"


@check("disabled_costs_nothing")
def _disabled():
    from renderer.larva_bodies import LarvaBodies

    class Spy:
        tile_size_m = TILE_M

        def __getattr__(self, name):
            raise AssertionError(f"read gmap.{name} with the view off")

    lb = LarvaBodies()
    lb.refresh_settings(_Cfg(enabled=False))
    lb.upload(Spy(), 1.0, PPT)
    lb.on_deaths([1.0], [1.0], [0.0], [7], 1.0)
    assert lb.instances is None and len(lb.dead) == 0
    return "no store read, nothing packed"


def _save(out_dir: Path):
    """A close-up brood for the eye: living larvae at eight phases, two in
    coma, dead ones limp and fading, under a warm light from the upper left
    over a dark floor; gamma-encoded."""
    s = _settings()
    zp = WPX / 4.0                     # 4 "tiles" across: a larva ~77 px
    rows = []
    for i in range(6):                 # living, six phases
        rows.append((0.35 + i * 0.62, 0.35, 0.15 * i - 0.4, i * np.pi / 3,
                     0, 1, 0, 1))
    rows.append((0.4, 1.0, 2.4, 0.0, 0, 1, 1, 1.0))       # coma
    rows.append((0.9, 1.0, -0.4, 1.0, 0, 1, 1, 0.9))
    for i, (limp, alpha) in enumerate([(0.3, 1.0), (1.0, 1.0), (1.0, 0.6),
                                       (1.0, 0.3)]):        # dying, dead
        rows.append((1.5 + i * 0.6, 1.0, 0.5 * i, 2.0, limp, alpha, 0, 1))
    rng = np.random.default_rng(5)
    for _ in range(14):
        rows.append((rng.uniform(0.2, 3.8), rng.uniform(1.4, 1.9),
                     rng.uniform(-np.pi, np.pi), rng.uniform(0, 6.28), 0, 1,
                     0, rng.uniform(0.88, 1.12)))
    from renderer import core
    light = _light(0.0)
    ta, tb = _STATE["tex"]
    a = np.zeros((GH, GW, 4), np.float16)
    yy, xx = np.mgrid[0:GH, 0:GW]
    lvl = 1.2 * np.exp(-((xx - 3) ** 2 + (yy - 1) ** 2) / 30.0)
    a[..., 0], a[..., 1], a[..., 2] = lvl * 1.0, lvl * 0.84, lvl * 0.67
    d = np.stack([-1.0 + 0 * xx, -0.6 + 0 * yy], -1).astype(np.float64)
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
    a[..., 3] = d[..., 0]
    b = np.zeros((GH, GW, 4), np.float16)
    b[..., 3] = d[..., 1]
    core.update_rgba16f_texture(ta, a)
    core.update_rgba16f_texture(tb, b)
    light = dataclasses.replace(light, ambient=(0.02, 0.02, 0.025))
    img, _ = _render(_inst(rows, zp), s, light,
                     floor=(0.012, 0.012, 0.014, 1.0), ppt=zp)
    rgb = np.clip(img[..., :3], 0, 1) ** (1 / 2.2)
    out = (rgb * 255).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "larva_bodies_closeup.png"
    im = rl.Image()
    buf = rl.ffi.new("unsigned char[]", out.tobytes())
    im.data, im.width, im.height = buf, WPX, HPX
    im.mipmaps, im.format = 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8
    rl.export_image(im, str(path))
    print(f"[save] {path}")


def main() -> int:
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 240, "gl_larva_bodies_check")
    try:
        for name, fn in CHECKS:
            if only is None or name in only:
                fn()
        if "--save" in sys.argv:
            _save(Path(sys.argv[sys.argv.index("--save") + 1]))
    finally:
        if rl.is_window_ready():
            rl.close_window()
    ok = all(r[1] for r in RESULTS)
    for name, passed, detail in RESULTS:
        print(f"[check] {name}: {'PASS' if passed else 'FAIL'} {detail}")
    print(f"LARVA_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
