"""THE particle primitive and its first consumer, the embers (graphics arc G3
"Fire", patch F4a; docs/g3_fire_design_2026-10-23.md §6.1, §8 "Particles").

Pure numpy, no GL context: ``renderer/particles.py`` keeps its GL face in one
class that imports pyray lazily. The GL side (the draw keeps the destination
alpha, lands where the ember is, no ember -> no draw) is
``tests/gl_fire_vfx_check.py`` (``ember_draw_additive`` /
``no_fire_no_ember_draws``).

Every test names the PROPERTY it protects and the change that must break it.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from renderer import fire_vfx_ref as ref  # noqa: E402
from renderer import particles as P  # noqa: E402

SRC = ROOT / "renderer" / "particles.py"


@pytest.fixture(scope="module")
def scales():
    return ref.load_scales()


def _settings(**kw):
    return P.EmberSettings(**kw)


def _scene(scales, *, h=24, w=24, wood=False, intensity=1.0, t_k=1250.0,
           seed=3):
    """A small map with a few burning tiles (permeable fuel, or wood with
    open air around it), through THE source law."""
    rng = np.random.default_rng(seed)
    fire = np.zeros((h, w))
    solid = np.zeros((h, w), bool)
    ys = rng.integers(4, h - 4, 4)
    xs = rng.integers(4, w - 4, 4)
    fire[ys, xs] = intensity
    if wood:
        solid[ys, xs] = True
    tk = np.full((h, w), 293.0)
    tk[fire > 0] = t_k
    vac = np.zeros((h, w), bool)
    src = ref.flame_sources(fire, tk, solid, fire > 0, vac,
                            t_amb_k=scales.t_amb_k,
                            glow_min_k=scales.glow_min_k, tile_size_m=0.333)
    return src, solid | vac


def _run(scales, settings, src, blocked, frames=300, dt=1.0 / 60.0,
         wind=None, seed_check=None):
    """Drive an EmberSystem ``frames`` frames; yields it after each."""
    es = P.EmberSystem(settings, t_amb_k=scales.t_amb_k,
                       glow_min_k=scales.glow_min_k)
    h, w = blocked.shape
    wind = np.zeros((h, w, 2), np.float32) if wind is None else wind
    t = 0.0
    for _ in range(frames):
        t += dt
        es.update(t=t, sources=src, origin=(0, 0), s0=8.0, strip_frac=0.375,
                  wind_fn=lambda: wind, tps=24.0, blocked=blocked)
        yield es


# --- the primitive's bounds ---------------------------------------------------

def test_no_ember_outlives_its_lifetime(scales):
    """Property: at every frame every alive ember is younger than its own
    lifetime, and no lifetime exceeds ``lifetime_s``; with the source gone,
    every ember is dead ``lifetime_s`` later. Breaks if the age check is
    lost, ages stop advancing, or a lifetime is drawn above the dial."""
    s = _settings(rate=40.0, lifetime_s=0.8, cool_s=50.0)
    src, blocked = _scene(scales)
    seen = 0
    es = None
    for es in _run(scales, s, src, blocked, frames=240):
        assert np.all(es.p.age < es.p.life)
        assert np.all(es.p.life <= s.lifetime_s + 1e-12)
        seen = max(seen, es.p.n)
    assert seen > 0, "nothing spawned: the test measured nothing"
    t = 240 / 60.0
    for _ in range(int(60 * s.lifetime_s) + 2):
        t += 1.0 / 60.0
        es.update(t=t, sources=None, origin=(0, 0), s0=8.0, strip_frac=0.375,
                  wind_fn=lambda: np.zeros((24, 24, 2)), tps=24.0,
                  blocked=blocked)
    assert es.p.n == 0, f"{es.p.n} embers outlived their lifetime"


def test_never_more_than_the_cap(scales):
    """Property: however hard the sources spawn, at most ``cap`` embers are
    alive; a live LOWER cap drops the oldest at once. Breaks if a spawn
    path skips the cap or the live cap change is ignored."""
    s = _settings(rate=500.0, cap=37, lifetime_s=2.0, cool_s=50.0)
    src, blocked = _scene(scales)
    full = False
    es = None
    for es in _run(scales, s, src, blocked, frames=120):
        assert es.p.n <= s.cap
        full |= es.p.n == s.cap
    assert full, "the cap was never reached: the test measured nothing"
    young_before = np.sort(es.p.age)[:10]
    es.refresh_settings(_settings(rate=500.0, cap=10, lifetime_s=2.0,
                                  cool_s=50.0))
    assert es.p.n == 10
    assert np.allclose(np.sort(es.p.age), young_before)


def test_an_ember_only_cools_and_only_dims(scales):
    """Property: an ember's temperature never rises along its life, never
    exceeds its birth temperature, and its drawn radiance (THE ramp's Le(T)
    x the fade weight) never rises in any channel -- a spark is born at its
    brightest. Breaks if the cooling re-heats (relaxes toward a hot gas, a
    sign error) or the weight gets a fade-in."""
    t0 = np.array([1350.0, 1100.0, 950.0])
    ages = np.linspace(0.0, 3.0, 200)
    life = np.full(3, 3.0)
    prev_t = None
    prev_rad = None
    for a in ages:
        temp = P.cooled_temperature(t0, np.full(3, a), scales.t_amb_k, 1.2)
        rad = P.ember_radiance(temp, 4.0 * P.fade(np.full(3, a), life),
                               scales.ramp)
        assert np.all(temp <= t0 + 1e-9)
        if prev_t is not None:
            assert np.all(temp <= prev_t + 1e-9)
            assert np.all(rad <= prev_rad + 1e-9)
        prev_t, prev_rad = temp, rad
    # ... and the running system keeps each ember on that curve
    s = _settings(rate=30.0, cool_s=1.5, lifetime_s=3.0)
    src, blocked = _scene(scales)
    for es in _run(scales, s, src, blocked, frames=90):
        want = P.cooled_temperature(es.p.t0, es.p.age, scales.t_amb_k,
                                    s.cool_s)
        assert np.allclose(es.p.temp, want)
        assert np.all(es.p.temp <= es.p.t0 + 1e-9)


@pytest.mark.parametrize("wood", [False, True])
def test_embers_are_born_only_where_a_flame_stands(scales, wood):
    """Property: every ember is born inside a SOURCE tile of THE source law
    (a permeable fuel's own tile; for wood, the strip of open air along the
    burning face), never hotter than that tile's T_src, and never inside
    the wood. Breaks if the spawn reads tiles the source law did not mark,
    detaches its temperature from the feeder, or lands in a wall."""
    src, blocked = _scene(scales, wood=wood)
    tiles = P.source_tiles(src, 8.0)
    s = _settings(rate=50.0)
    rng = np.random.default_rng(7)
    new = P.ember_spawns(tiles, (0, 0), 1.0, s, 0.375, rng)
    n = new["x"].size
    assert n > 50, "almost nothing spawned: the test measured nothing"
    ix = np.floor(new["x"]).astype(int)
    iy = np.floor(new["y"]).astype(int)
    cover = np.asarray(src.cover)
    assert np.all(cover[iy, ix] != 0), "an ember was born off a source tile"
    assert not blocked[iy, ix].any(), "an ember was born inside a wall"
    assert np.all(new["t0"] <= np.asarray(src.t_src_k)[iy, ix] + 1e-9)
    if wood:
        # inside the strip along a face the cover names
        fx, fy = new["x"] - ix, new["y"] - iy
        sf = 0.375
        c = cover[iy, ix]
        ok = (((c & ref.FACE_N) != 0) & (fy <= sf)) \
            | (((c & ref.FACE_S) != 0) & (fy >= 1 - sf)) \
            | (((c & ref.FACE_W) != 0) & (fx <= sf)) \
            | (((c & ref.FACE_E) != 0) & (fx >= 1 - sf))
        assert ok.all(), f"{(~ok).sum()} wood embers outside their strips"


def test_no_fire_no_ember(scales):
    """Property: with no burning tile (no source, or a fire too cool for
    the source law) the system spawns nothing, holds zero particles and
    packs zero sprites -- the draw is skipped. Breaks if embers spawn from
    anything but THE source law's tiles."""
    s = _settings(rate=1000.0)
    _, blocked = _scene(scales)
    cold, _ = _scene(scales, t_k=500.0)          # below the glow floor
    assert P.source_tiles(cold, 8.0) is None
    for src in (None, cold):
        es = None
        for es in _run(scales, s, src, blocked, frames=120):
            assert es.p.n == 0 and es.spawned_last_frame == 0
        assert P.pack_sprites(es.p, None, 48.0, s).shape[1] == 0


def test_spawn_rate_is_proportional_to_intensity(scales):
    """Property: the expected spawn count is rate x I x dt per source tile
    (a fire at a quarter intensity throws a quarter of the sparks). Breaks
    if the rate ignores the fire's intensity."""
    s = _settings(rate=20.0)
    counts = {}
    for i in (0.25, 1.0):
        src, _ = _scene(scales, intensity=i)
        tiles = P.source_tiles(src, 8.0)
        rng = np.random.default_rng(11)
        counts[i] = sum(P.ember_spawns(tiles, (0, 0), 0.05, s, 0.375,
                                       rng)["x"].size for _ in range(4000))
        expect = 20.0 * i * len(tiles.ty) * 0.05 * 4000
        assert abs(counts[i] - expect) < 5.0 * np.sqrt(expect), (i, counts)
    assert 3.4 < counts[1.0] / counts[0.25] < 4.6


def test_seeded_and_reproducible(scales):
    """Property: two systems with the same seed fed the same frames hold
    bit-identical particles (a look shot is reproducible); another seed
    gives other sparks. Breaks if an unseeded RNG or wall time leaks in."""
    src, blocked = _scene(scales)
    s = _settings(rate=30.0)
    rng_w = np.random.default_rng(5)
    wind = (rng_w.standard_normal((24, 24, 2)) * 0.02).astype(np.float32)
    a = list(es.p.d.copy() for es in _run(scales, s, src, blocked,
                                          frames=60, wind=wind))
    b = list(es.p.d.copy() for es in _run(scales, s, src, blocked,
                                          frames=60, wind=wind))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = list(es.p.d.copy() for es in _run(scales, _settings(rate=30.0,
                                                            seed=99),
                                          src, blocked, frames=60, wind=wind))
    assert not all(x.shape == y.shape and np.array_equal(x, y)
                   for x, y in zip(a, c))


def test_embers_ride_the_wind_and_die_in_walls(scales):
    """Property: in a steady wind an ember's velocity relaxes to the wind
    (tamed tiles/tick x tps); one that reaches a blocked tile dies. Breaks
    if the wind's units are lost (tiles/tick taken as tiles/s) or embers
    pass through walls."""
    s = _settings(curl=0.0, drag=4.0, cool_s=100.0, lifetime_s=10.0)
    p = P.Particles(8)
    p.add(x=np.array([2.5]), y=np.array([2.5]), vx=np.array([0.0]),
          vy=np.array([0.0]), t0=np.array([1200.0]), life=np.array([9.0]))
    blocked = np.zeros((6, 12), bool)
    blocked[:, 6] = True
    wind = np.zeros((6, 12, 2))
    wind[..., 0] = 0.05                          # tiles per tick, east
    modes = ref.curl_noise_modes(s.curl_dials())
    for k in range(240):
        P.integrate(p, 1.0 / 60.0, k / 60.0,
                    air_fn=lambda iy, ix: wind[iy, ix] * 24.0,
                    blocked=blocked, settings=s, t_amb_k=scales.t_amb_k,
                    glow_min_k=scales.glow_min_k, modes=modes)
        if p.n and k == 60:
            assert abs(p.vx[0] - 0.05 * 24.0) < 0.05
        assert not p.n or p.x[0] < 6.0
    assert p.n == 0, "the ember crossed into the wall instead of dying"


def test_curl_noise_at_points_is_the_twins_field():
    """Property: the embers' drift is THE twin's curl noise -- the point
    form equals ``curl_noise`` at cell centres. Breaks if the point form
    drifts into a second noise."""
    d = ref.FireVFXDials(curl_modes=8)
    modes = ref.curl_noise_modes(d)
    g = ref.curl_noise(6, 9, 4, 2.7, modes)
    yy, xx = np.meshgrid((np.arange(6) + 0.5) / 4, (np.arange(9) + 0.5) / 4,
                         indexing="ij")
    pt = ref.curl_noise_at(xx.ravel(), yy.ravel(), 2.7, modes).reshape(6, 9, 2)
    assert np.allclose(g, pt, atol=1e-12)


def test_zoomed_out_sparks_keep_their_light(scales):
    """Property: zoomed out, a spark's radius is floored at
    ``min_screen_px`` screen px and its weight scaled so radius^2 x weight
    (its light) is unchanged. Breaks if the floor brightens a spark."""
    s = _settings(size_tiles=0.05, min_screen_px=1.5, brightness=4.0)
    p = P.Particles(4)
    p.add(x=np.array([3.0]), y=np.array([3.0]), vx=np.zeros(1),
          vy=np.zeros(1), t0=np.array([1200.0]), life=np.array([2.0]))
    near = P.pack_sprites(p, None, 48.0, s, screen_px_per_world_px=2.0)
    far = P.pack_sprites(p, None, 48.0, s, screen_px_per_world_px=0.25)
    assert far[1, 0, 2] > near[1, 0, 2]
    assert np.isclose(far[1, 0, 2] ** 2 * far[1, 0, 1],
                      near[1, 0, 2] ** 2 * near[1, 0, 1])


def test_pack_clips_to_the_view(scales):
    """Property: only sprites in the view (plus one tile) are packed --
    the draw is clipped to the view. Breaks if the clip is lost."""
    s = _settings()
    p = P.Particles(8)
    p.add(x=np.array([1.0, 10.0, 30.0]), y=np.array([1.0, 10.0, 30.0]),
          vx=np.zeros(3), vy=np.zeros(3), t0=np.full(3, 1200.0),
          life=np.full(3, 2.0))
    out = P.pack_sprites(p, (5.0, 5.0, 15.0, 15.0), 48.0, s)
    assert out.shape == (2, 1, 4) and out[0, 0, 0] == pytest.approx(480.0)


# --- the config, the order, render-only ----------------------------------------

def test_config_names_every_ember_dial():
    """Property: ``[render.embers]`` names every EmberSettings dial and
    nothing else, and the settings carry the config's values. Breaks if a
    dial is added without a config row (a silent fallback) or misspelled."""
    from config import CFG
    sec = CFG.render.embers
    assert set(vars(sec)) == set(P.config_keys())
    st = P.settings_from_config(CFG)
    for k in P.config_keys():
        assert getattr(st, k) == getattr(sec, k), k


def test_compose_world_draws_the_embers_over_the_flame_under_the_units():
    """Property: ``compose_world`` draws the embers in the linear scene
    after the flame and before the units, their compensation after the
    ceiling soot (with the flame's), and all of it before the one resolve
    (design §6.1, §13.7). Breaks if the embers slip past the resolve into
    display space or out of their height."""
    from renderer.game_renderer import GameRenderer
    src = inspect.getsource(GameRenderer.compose_world)
    order = ['fx.draw_glow("flame"',
             "self._draw_embers(view)",
             "self._draw_units_3d_world(",
             'self.gas_relief.draw("ceiling"',
             'fx.draw_glow("compensation"',
             'self._draw_embers(view, stage="compensation"',
             "self.world.resolve("]
    pos = [src.index(k) for k in order]
    assert pos == sorted(pos), order
    assert "_draw_embers" not in src[src.index("self.world.resolve("):]


def test_the_sprite_shader_colours_only_through_the_lut():
    """Property: the sprite shader reads its colour from the blackbody GPU
    LUT and keeps the destination alpha (writes alpha 0 under an additive
    ONE/ONE, ZERO/ONE blend) -- one ramp, no tone-map. Breaks if a second
    T -> colour map or an alpha write appears."""
    vs, fs = P.sprite_shader_sources()
    assert "texture(u_lut" in fs and "finalColor = vec4(c, 0.0);" in fs
    low = fs.lower()
    for bad in ("planck", "aces", "pow(", "srgb"):
        assert bad not in low, bad
    assert "smokeCoverAt" in fs              # the compensation's one cover
    src = SRC.read_text(encoding="utf-8")
    assert "rl.RL_ONE, rl.RL_ONE, rl.RL_ZERO" in src


def test_particles_are_render_only():
    """Property: ``renderer/particles.py`` never imports the sim nor
    assigns into a gmap / sim / unit object, and never makes a sim view
    writeable. Breaks if the particle system grows a sim write."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            assert not any(n.split(".")[0] in ("simulation", "breach_physics")
                           for n in names), f"sim import: {names}"
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                base = t
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and t is not base:
                    assert base.id not in ("gmap", "sim", "unit", "units",
                                           "sources", "blocked"), \
                        f"line {node.lineno} assigns into {base.id}"
    assert "writeable = True" not in SRC.read_text(encoding="utf-8")


def test_the_effects_queue_is_named_legacy_not_a_second_system():
    """Property: the particle module names the renderer's ``_effects``
    queue as legacy display-space effects (design §6.1, minor 7), so the
    next consumer extends THE primitive instead of the queue. Breaks if the
    rule's text is dropped."""
    doc = P.__doc__ or ""
    assert "_effects" in doc and "LEGACY" in doc
