"""Graphics G4 P4 -- DEBRIS, the particle primitive's third consumer
(``renderer/particles.py`` :class:`DebrisSystem`, :func:`integrate_debris`,
the draw's premultiplied ``"debris"`` stage; design
``docs/explosion_look_design_2026-11-07.md`` §3.4, §6 S5 / N6, §8 P4, and
Erik's ruling at the screen, 2026-11-07: drag toward the air x the tile's
gas density, plus a density-independent ground friction; a chip stops at a
wall).

Properties only (never a snapshot): the count law, the lifetime and the cap,
the births at the walk's seeds, stopping at walls, the density-scaled drag,
the ground friction, the vacuum slide, reproducibility, the VFX clock, the
ember / spark paths' byte-identity, the premultiplied blend, the door.
"""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from renderer import particles as P  # noqa: E402
from simulation.blast_expansion import walk_seeds  # noqa: E402

T_AMB, GLOW_MIN = 293.0, 800.0


def _settings(**kw):
    base = dict(enabled=True, rate=25.0, v0_tiles_s=8.0, v_spread=0.6,
                drag=1.5, friction_tiles_s2=6.0, life_s=3.0, life_jitter=0.3,
                size_tiles=0.08, size_jitter=0.5, albedo=(0.08, 0.07, 0.06),
                alpha=0.9, fade_s=0.8, seed=6271, cap=512)
    base.update(kw)
    return P.DebrisSettings(**base)


def _blast(pos, strength=1.0, tick0=48, approach=None):
    return {"kind": "blast", "pos": pos, "strength": strength,
            "approach": approach, "tick0": tick0, "t": 0.0, "life": 3.0}


def _seeds_fn(mask):
    return lambda fy, fx, approach: walk_seeds(mask, fy, fx, approach)


def _open_room(h=30, w=30):
    mask = np.zeros((h, w), bool)
    mask[1:-1, 1:-1] = True
    return mask


def _density(value):
    return lambda iy, ix: np.full(np.shape(iy), float(value))


def _drive(ds, effects, mask, frames, dt=1.0 / 60.0, t0=0.0, wind=None,
           density=1.0, tps=24.0):
    """Step ``ds`` ``frames`` frames; yields after each."""
    h, w = mask.shape
    wind = np.zeros((h, w, 2), np.float32) if wind is None else wind
    t = t0
    for _ in range(frames):
        t += dt
        ds.update(t=t, effects=effects, seeds_fn=_seeds_fn(mask),
                  sim_tick=48, wind_fn=lambda: wind, tps=tps,
                  density_fn=_density(density), blocked=~mask)
        yield ds


def _one_chip(x, y, vx=0.0, vy=0.0, life=100.0):
    p = P.Particles(8)
    p.add(x=[x], y=[y], vx=[vx], vy=[vy], life=[life], t0=[0.3],
          temp=[1.0])
    return p


def _step(p, settings, mask, seconds, air=(0.0, 0.0), density=1.0,
          h=1.0 / 120.0):
    """Integrate ``p`` for ``seconds`` in still or uniform air (tiles/s)."""
    air = np.asarray(air, np.float64)
    for _ in range(int(round(seconds / h))):
        P.integrate_debris(p, h, air_fn=lambda iy, ix: np.tile(
            air, (np.size(iy), 1)), density_fn=_density(density),
            blocked=~mask, settings=settings)


# --- the count law -------------------------------------------------------------

def test_chip_count_scales_linearly_with_strength():
    """Property: a blast throws ``rate * strength`` chips -- the count ratio
    of two strengths is their ratio (compared by ratio, never a pinned
    number). Breaks if the count goes sqrt (the sparks' law) or ignores the
    strength."""
    mask = _open_room(60, 60)
    counts = []
    for strength in (1.0, 6.0):
        ds = P.DebrisSystem(_settings(rate=40.0, cap=10_000))
        new = ds.consume([_blast((30, 30), strength)], _seeds_fn(mask), 48)
        counts.append(new["x"].size)
    assert counts[0] > 0
    assert counts[1] / counts[0] == pytest.approx(6.0, rel=0.02)
    assert P.debris_count(0.0, 25.0) == 0          # no strength, no chip


# --- the primitive's bounds ----------------------------------------------------

def test_no_chip_outlives_its_life_and_the_cap_holds():
    """Property: at every frame every alive chip is younger than its own
    life, never more than ``cap`` are alive even when a volley offers more,
    and every chip is gone by the longest life. Breaks if the age kill, the
    life draw or the cap's fair subset regresses."""
    mask = _open_room(60, 60)
    s = _settings(cap=120, life_s=1.0, life_jitter=0.3)
    ds = P.DebrisSystem(s)
    effects = [_blast((20, 20), 4.0, tick0=10), _blast((40, 40), 3.0,
                                                         tick0=10)]
    saw = 0
    for k, d in enumerate(_drive(ds, effects, mask, 120, dt=1.0 / 50.0)):
        assert d.p.n <= 120
        if d.p.n:
            saw = max(saw, d.p.n)
            assert np.all(d.p.age < d.p.life)
            assert np.all(d.p.life <= 1.0 * 1.3 + 1e-12)
    assert saw == 120
    assert ds.p.n == 0


# --- births at the walk's seeds ------------------------------------------------

def test_births_are_at_the_walks_seeds_never_in_a_wall():
    """Property: chips are born at the tile CENTRES (+0.5) of the blast
    walk's seeds: the centre when open, beside it for a blast centred on a
    wall (the approach side's one when given), never in the wall. Breaks if
    births ignore the walk's seed rule or lose the tile-centre offset."""
    mask = _open_room()
    mask[:, 15] = False
    cases = [((10, 10), None, {(10, 10)}),
             ((15, 10), None, {(10, 14), (10, 16)}),
             ((15, 10), (10, 16), {(10, 16)})]
    for pos, approach, want in cases:
        ds = P.DebrisSystem(_settings(rate=60.0))
        new = ds.consume([_blast(pos, approach=approach)], _seeds_fn(mask),
                         48)
        assert new["x"].size > 0
        born = set(zip((new["y"] - 0.5).astype(int).tolist(),
                       (new["x"] - 0.5).astype(int).tolist()))
        assert born == want, (pos, approach, born)
        assert np.all(new["x"] % 1.0 == 0.5) and np.all(new["y"] % 1.0 == 0.5)
        assert mask[new["y"].astype(int), new["x"].astype(int)].all()


# --- walls ---------------------------------------------------------------------

def test_a_chip_entering_a_blocked_tile_stops_at_its_last_open_position():
    """Property: a chip flung at a wall STOPS in the open tile before it
    (velocity zero, still alive) and never enters or crosses it, even at a
    speed that would step over it in one frame (the substeps). Breaks if a
    wall kills the chip (the sparks' rule), lets it through, or leaves it
    moving."""
    mask = _open_room()
    mask[:, 18] = False
    s = _settings(friction_tiles_s2=0.0, drag=0.0, rate=200.0, life_s=20.0)
    ds = P.DebrisSystem(s)
    effects = [_blast((16, 15))]
    born = None
    # 10 s: every chip (>= 3.2 tiles/s, no friction) has met some wall
    for k, d in enumerate(_drive(ds, effects, mask, 100, dt=0.1)):
        if k == 0:
            born = d.p.n
            continue
        iy, ix = d.p.y.astype(int), d.p.x.astype(int)
        assert mask[iy, ix].all(), "a chip is inside a wall"
        assert np.all(d.p.x < 18.0), "a chip crossed the wall"
        assert d.p.n == born, "a wall killed a chip"
    east = ds.p.x > 17.0
    assert east.any()
    assert np.all(ds.p.vx[east] == 0.0) and np.all(ds.p.vy[east] == 0.0)


def test_in_vacuum_a_chip_slides_until_the_wall():
    """Property: at zero gas density (no air term) and no friction a chip
    keeps its speed until the wall, then stops there and lives; with
    friction it slides the friction's stopping distance v^2 / 2f, whatever
    the wind. Breaks if vacuum kills the chip, the wind still pushes it, or
    the friction is scaled by the density."""
    mask = _open_room(10, 40)
    s = _settings(friction_tiles_s2=0.0)
    p = _one_chip(5.5, 5.5, vx=10.0)
    _step(p, s, mask, 4.0, air=(-30.0, 0.0), density=0.0)
    assert p.n == 1
    assert 38.0 < p.x[0] < 39.0 and p.vx[0] == 0.0     # at the wall, stopped
    s = _settings(friction_tiles_s2=4.0)
    p = _one_chip(5.5, 5.5, vx=8.0)
    _step(p, s, mask, 3.0, air=(0.0, 30.0), density=0.0)
    assert p.vx[0] == 0.0 and p.vy[0] == 0.0
    assert p.x[0] - 5.5 == pytest.approx(8.0 ** 2 / (2 * 4.0), rel=0.03)
    assert p.y[0] == 5.5


# --- the air term and the friction ----------------------------------------------

def test_drag_scales_with_gas_density():
    """Property: the same resting chip in the same wind is not moved at all
    at zero density, is moved at 1 atm, and is moved further at 2 atm
    (the air term's rate is drag x density). Breaks if the drag ignores the
    density (vacuum would blow a storm) or the density is dropped."""
    mask = _open_room(10, 60)
    s = _settings(friction_tiles_s2=0.0)
    moved = []
    for rho in (0.0, 1.0, 2.0):
        p = _one_chip(5.5, 5.5)
        _step(p, s, mask, 0.5, air=(10.0, 0.0), density=rho)
        moved.append(p.x[0] - 5.5)
    assert moved[0] == 0.0
    assert moved[1] > 0.5
    assert moved[2] > moved[1] * 1.2


def test_ground_friction_rests_a_sliding_chip_and_holds_a_lying_one():
    """Property: in still air a sliding chip comes to rest (exactly zero
    speed) and stays; a lying chip in a draught whose pull drag x |air| is
    below the friction does not move; in a wind above it, it does. Breaks if
    the friction is dropped (a chip glides for ever), reverses the chip, or
    a draught creeps a lying chip across the floor."""
    mask = _open_room(10, 60)
    s = _settings(drag=1.5, friction_tiles_s2=6.0)
    p = _one_chip(5.5, 5.5, vx=8.0)
    _step(p, s, mask, 2.0)
    assert p.vx[0] == 0.0 and p.vy[0] == 0.0
    x_rest = p.x[0]
    assert 5.5 < x_rest < 5.5 + 8.0 ** 2 / (2 * 6.0)   # drag shortens the slide
    _step(p, s, mask, 1.0)
    assert p.x[0] == x_rest
    draught = 0.8 * s.friction_tiles_s2 / s.drag      # pull below friction
    p = _one_chip(5.5, 5.5)
    _step(p, s, mask, 2.0, air=(draught, 0.0))
    assert p.x[0] == 5.5 and p.vx[0] == 0.0
    gale = 3.0 * s.friction_tiles_s2 / s.drag
    p = _one_chip(5.5, 5.5)
    _step(p, s, mask, 0.5, air=(gale, 0.0))
    assert p.x[0] > 6.0


# --- reproducible, on the VFX clock -------------------------------------------

def test_the_debris_is_reproducible_from_the_seed():
    """Property: the same effect (tick0, pos, strength) throws the same
    chips with the same hashed rotations and sizes and they step the same;
    another tick0 throws other chips. Breaks if the spawn or the hash reads
    an unseeded or shared stream (a look shot would not reproduce)."""
    mask = _open_room()
    runs = []
    for tick0 in (48, 48, 49):
        ds = P.DebrisSystem(_settings())
        for _ in _drive(ds, [_blast((12, 12), 2.0, tick0=tick0)], mask, 6):
            pass
        runs.append(ds.p.d.copy())
    assert runs[0].tobytes() == runs[1].tobytes()
    assert runs[0].shape != runs[2].shape or not np.array_equal(runs[0],
                                                                 runs[2])
    rot, size = runs[0][:, P.T0], runs[0][:, P.TEMP]
    assert np.unique(rot).size == rot.size           # a rotation per chip
    assert np.all(size >= 0.5 - 1e-12) and np.all(size <= 1.5 + 1e-12)


def test_the_debris_runs_on_the_vfx_clock_never_wall_clock(monkeypatch):
    """Property: what the chips do is a function of the VFX time handed in,
    never of the wall clock: with the wall clock jumped the state is
    byte-identical, and with ``t`` held still no chip ages or moves. Breaks
    if the spawn or the step reads ``time`` for anything but its timing
    counter."""
    mask = _open_room()

    def run():
        ds = P.DebrisSystem(_settings())
        for _ in _drive(ds, [_blast((12, 12), 3.0)], mask, 5):
            pass
        return ds.p.d.copy()

    a = run()
    real = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real() + 9e4)
    monkeypatch.setattr(time, "time", lambda: 1e9)
    assert a.tobytes() == run().tobytes()
    ds = P.DebrisSystem(_settings())
    effects = [_blast((12, 12))]
    for _ in _drive(ds, effects, mask, 1):
        pass
    before = ds.p.d.copy()
    for _ in range(5):
        ds.update(t=1.0 / 60.0, effects=effects, seeds_fn=_seeds_fn(mask),
                  sim_tick=48, wind_fn=lambda: None, tps=24.0,
                  density_fn=_density(1.0), blocked=~mask)
    assert ds.p.d.tobytes() == before.tobytes()


def test_a_blast_is_consumed_once_and_only_its_own_flag_is_set():
    """Property: a ``"blast"`` effect throws chips exactly once (marked
    ``"debris_seen"``), never re-thrown, no other consumer's flag touched;
    a disabled system marks it seen and throws nothing later. Breaks if the
    debris reads ``sparks_seen`` or re-throws a blast."""
    mask = _open_room()
    fx = _blast((10, 10))
    ds = P.DebrisSystem(_settings())
    assert ds.consume([fx], _seeds_fn(mask), 48)["x"].size > 0
    assert ds.consume([fx], _seeds_fn(mask), 48)["x"].size == 0
    assert fx["debris_seen"] is True
    assert "sparks_seen" not in fx and "shock_seen" not in fx
    off = P.DebrisSystem(_settings(enabled=False))
    fx2 = _blast((10, 10))
    for _ in _drive(off, [fx2], mask, 2):
        pass
    assert fx2["debris_seen"] and off.p.n == 0


# --- the ember / spark paths, the draw -------------------------------------------

def test_ember_and_spark_packs_are_byte_identical_with_the_debris_present():
    """Property: with the debris system present (idle, or holding chips)
    the renderer's ember + spark draw hands the sprite draw exactly the
    embers' and sparks' own packs, and the debris draw is a SEPARATE stage
    that packs only chips. Breaks if the debris joins the additive sprite
    array or perturbs the ember / spark pack."""
    from renderer.game_renderer import GameRenderer
    es = P.EmberSystem(P.EmberSettings(), t_amb_k=T_AMB, glow_min_k=GLOW_MIN)
    rng = np.random.default_rng(1)
    es.p.add(x=rng.uniform(2, 20, 30), y=rng.uniform(2, 20, 30),
             vx=rng.normal(size=30), vy=rng.normal(size=30),
             life=np.full(30, 2.0), t0=np.full(30, 1200.0))
    ss = P.SparkSystem(P.spark_settings_from_config(_cfg()), t_amb_k=T_AMB,
                       glow_min_k=GLOW_MIN)
    ss.p.add(**ss.consume([_blast((10, 10))], _seeds_fn(_open_room()), 48))
    view = (0.0, 0.0, 24.0, 24.0)
    want = np.concatenate(
        [P.pack_sprites(s.p, view, 48.0, s.settings,
                        screen_px_per_world_px=0.5) for s in (es, ss)],
        axis=1)
    got, chips = [], []
    draw = SimpleNamespace(
        draws_last_frame=0, sprites_last_frame=0,
        debris_draws_last_frame=0, debris_sprites_last_frame=0,
        draw=lambda sprites, **kw: got.append(sprites) or 1,
        draw_debris=lambda sprites, **kw: chips.append(sprites) or 1)
    for holding in (False, True):
        ds = P.DebrisSystem(_settings())
        if holding:
            ds.p.add(**ds.consume([_blast((12, 12))],
                                  _seeds_fn(_open_room()), 48))
        fake = SimpleNamespace(
            embers=es, sparks=ss, debris=ds, ember_draw=draw,
            # #87: the muzzle sparks ride the same draw; none here
            muzzle_sparks=SimpleNamespace(enabled=False,
                                          p=SimpleNamespace(n=0)),
            _ember_sprites=None,
            camera=SimpleNamespace(zoom_px_per_tile=24.0),
            world=SimpleNamespace(world_px_per_tile=48.0, world_px_w=1152.0,
                                  world_px_h=1152.0),
            fire_vfx=SimpleNamespace(ramp=SimpleNamespace(
                kelvin_floor=800.0, kelvin_ceil=10000.0), lut_tex=None),
            _smoke_cast=(None,), _light_field_ctx=lambda: None)
        got.clear()
        chips.clear()
        assert GameRenderer._draw_debris(fake, view) == (1 if holding else 0)
        assert GameRenderer._draw_embers(fake, view) == 1
        assert len(got) == 1 and got[0].tobytes() == want.tobytes()
        if holding:
            assert len(chips) == 1
            assert chips[0].tobytes() == P.pack_debris(
                ds.p, view, 48.0, ds.settings).tobytes()
        else:
            assert not chips


def test_the_debris_stage_blends_premultiplied_never_additive():
    """Property: the debris stage sets the premultiplied factors (ONE,
    ONE_MINUS_SRC_ALPHA on RGB and alpha) and writes premultiplied colour
    (rgb x a, a), and the sprite stage keeps its additive factors -- the
    two blend paths, no third. Breaks if a dark chip is drawn additively
    (it could never darken the floor) or the sprite's blend changes."""
    src = inspect.getsource(P.ParticleDraw.draw_debris)
    assert ("rl.RL_ONE, rl.RL_ONE_MINUS_SRC_ALPHA,\n"
            in src.replace("\r\n", "\n"))
    assert src.count("rl.RL_ONE_MINUS_SRC_ALPHA") == 2
    assert "rl.RL_ONE, rl.RL_ONE, rl.RL_ZERO" not in src
    assert "rl.RL_ONE, rl.RL_ONE, rl.RL_ZERO" in inspect.getsource(
        P.ParticleDraw.draw)
    whole = inspect.getsource(P)
    assert whole.count("rl_set_blend_factors_separate") == 2
    vs, fs = P.debris_shader_sources()
    assert "finalColor = vec4(lit * a, a);" in fs
    # lit through the lit-3D block, the props' read: the field sample and
    # the smoke cast shadow, spliced, never re-typed
    from renderer.lit3d import _FIELD_SAMPLE_GLSL, _SMOKE_SHADOW_GLSL
    assert _FIELD_SAMPLE_GLSL in fs and _SMOKE_SHADOW_GLSL in fs
    for bad in ("aces", "dither", "1.0 / 2.2"):
        assert bad not in fs.lower(), bad


def test_compose_world_draws_the_debris_after_the_floor_smoke_under_the_flame():
    """Property: ``compose_world`` draws the debris in the linear scene
    after the floor smoke group and before the flame, the embers, the units
    and the resolve (chips lie on the floor). Breaks if the debris slips
    over the flame / units or past the one tone-map."""
    from renderer.game_renderer import GameRenderer
    src = inspect.getsource(GameRenderer.compose_world)
    order = ['self.gas_relief.draw("floor"',
             "self._draw_debris(",
             'fx.draw_glow("flame"',
             "self._draw_embers(view)",
             "self._draw_units_3d_world(",
             "self.world.resolve("]
    pos = [src.index(k) for k in order]
    assert pos == sorted(pos), order
    assert src.count("self._draw_debris(") == 1


def test_the_debris_density_read_is_the_prop_sways():
    """Property: the debris' density factor is the prop sway's read -- the
    tile's bulk N through the single-tile accessor over the ambient
    reference, 0 in vacuum, 1 at ambient, equal to ``sample_gas_frac`` tile
    for tile. Breaks if the debris grows a second density read."""
    from renderer.static_props import gas_frac_at_tiles
    n = {(1, 1): 0, (1, 2): 50, (2, 1): 100, (2, 2): 200}
    fn = lambda y, x: n[(y, x)]        # noqa: E731
    iy = np.array([1, 1, 2, 2, 2])
    ix = np.array([1, 2, 1, 2, 2])
    got = gas_frac_at_tiles(fn, 100.0, iy, ix)
    assert got.tolist() == [0.0, 0.5, 1.0, 2.0, 2.0]
    assert np.all(gas_frac_at_tiles(None, 100.0, iy, ix) == 1.0)


# --- the door ---------------------------------------------------------------------

def _cfg():
    from config import CFG
    return CFG


_DROP = object()


def test_config_names_every_debris_dial_and_the_door_refuses():
    """Property: ``[render.debris]`` names every DebrisSettings dial and
    nothing else, the settings carry the config's values, and the door
    refuses a missing section, a missing key and a mistyped value by name.
    Breaks if a dial falls back silently or the config drifts from the
    schema."""
    cfg = _cfg()
    sec = cfg.render.debris
    assert set(vars(sec)) == set(P.debris_config_keys())
    st = P.debris_settings_from_config(cfg)
    for k in P.debris_config_keys():
        want = getattr(sec, k)
        got = getattr(st, k)
        assert (tuple(got) == tuple(want) if k == "albedo"
                else got == want), k

    def cfg_with(**over):
        d = dict(vars(sec))
        d.update(over)
        for k, v in list(d.items()):
            if v is _DROP:
                del d[k]
        return SimpleNamespace(render=SimpleNamespace(
            debris=SimpleNamespace(**d)))

    with pytest.raises(ValueError, match="no \\[render.debris\\]"):
        P.debris_settings_from_config(SimpleNamespace(
            render=SimpleNamespace()))
    with pytest.raises(ValueError, match="missing 'friction_tiles_s2'"):
        P.debris_settings_from_config(cfg_with(friction_tiles_s2=_DROP))
    with pytest.raises(ValueError, match="cap has the wrong type"):
        P.debris_settings_from_config(cfg_with(cap=512.0))
    with pytest.raises(ValueError, match="enabled has the wrong type"):
        P.debris_settings_from_config(cfg_with(enabled=1))
    with pytest.raises(ValueError, match="albedo has the wrong type"):
        P.debris_settings_from_config(cfg_with(albedo=[0.1, 0.1]))
    with pytest.raises(ValueError, match="drag has the wrong type"):
        P.debris_settings_from_config(cfg_with(drag="1.5"))
    with pytest.raises(ValueError, match="v_spread"):
        P.debris_settings_from_config(cfg_with(v_spread=1.5))
