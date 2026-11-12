"""Graphics G4 P3 -- SPARKS, the particle primitive's second consumer
(``renderer/particles.py`` :class:`SparkSystem`; design
``docs/explosion_look_design_2026-11-07.md`` §3.4, §4, §6 N2/N5).

Properties only (never a snapshot): the count law, the lifetime and the cap,
the births at the blast walk's seeds, death in walls, reproducibility from
the seed, the VFX clock, the ember path's byte-identity, the refusing door,
and colour through the LUT only.
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

from renderer import fire_vfx_ref as ref  # noqa: E402
from renderer import particles as P  # noqa: E402
from simulation.blast_expansion import walk_seeds  # noqa: E402

T_AMB, GLOW_MIN = 293.0, 800.0


def _settings(**kw):
    base = dict(enabled=True, rate=60.0, v0_tiles_s=12.0, v_spread=0.5,
                t_born_k=2600.0, cool_s=0.5, life_s=0.5, life_jitter=0.5,
                drag=2.0, curl=1.0, size_tiles=0.04, min_screen_px=1.0,
                streak_s=0.04, brightness=1.0, cap=512, seed=9157)
    base.update(kw)
    return P.SparkSettings(**base)


def _modes():
    return ref.curl_noise_modes(P.EmberSettings().curl_dials())


def _blast(pos, strength=1.0, tick0=48, approach=None):
    return {"kind": "blast", "pos": pos, "strength": strength,
            "approach": approach, "tick0": tick0, "t": 0.0, "life": 3.0}


def _seeds_fn(mask):
    return lambda fy, fx, approach: walk_seeds(mask, fy, fx, approach)


def _system(settings=None):
    return P.SparkSystem(settings or _settings(), t_amb_k=T_AMB,
                         glow_min_k=GLOW_MIN)


def _drive(ss, effects, mask, frames, dt=1.0 / 60.0, t0=0.0):
    """Step ``ss`` ``frames`` frames over a still map; yields after each."""
    h, w = mask.shape
    wind = np.zeros((h, w, 2), np.float32)
    t = t0
    for _ in range(frames):
        t += dt
        ss.update(t=t, effects=effects, seeds_fn=_seeds_fn(mask),
                  sim_tick=48, wind_fn=lambda: wind, tps=24.0,
                  blocked=~mask, modes=_modes())
        yield ss


def _open_room(h=30, w=30):
    mask = np.zeros((h, w), bool)
    mask[1:-1, 1:-1] = True
    return mask


# --- the count law -------------------------------------------------------------

def test_spark_count_scales_as_sqrt_strength():
    """Property: a blast throws ``rate * sqrt(strength)`` sparks -- the
    count ratio of two strengths is the square root of their ratio (compared
    by ratio, never a pinned number). Breaks if the count goes linear in
    strength, ignores it, or the spawn drops sparks below the cap."""
    mask = _open_room(60, 60)
    counts = []
    for strength in (1.0, 9.0):
        ss = _system(_settings(rate=40.0, cap=10_000))
        new = ss.consume([_blast((30, 30), strength)], _seeds_fn(mask), 48)
        counts.append(new["x"].size)
    assert counts[0] > 0
    assert counts[1] / counts[0] == pytest.approx(3.0, rel=0.02)
    assert P.spark_count(0.0, 60.0) == 0          # no strength, no spark


# --- the primitive's bounds ----------------------------------------------------

def test_no_spark_outlives_its_lifetime_and_the_cap_holds():
    """Property: at every frame every alive spark is younger than its own
    lifetime, and never more than ``cap`` are alive, even when a volley
    offers more than the cap. Breaks if the kill test, the life draw or the
    cap's fair subset regresses."""
    mask = _open_room(60, 60)
    ss = _system(_settings(cap=150))
    effects = [_blast((20, 20), 12.0, tick0=10), _blast((40, 40), 8.0,
                                                          tick0=10)]
    saw = 0
    for k, s in enumerate(_drive(ss, effects, mask, 120)):
        assert s.p.n <= 150
        if s.p.n:
            saw = max(saw, s.p.n)
            assert np.all(s.p.age < s.p.life)
            assert np.all(s.p.life <= 0.5 * 1.5 + 1e-12)
        if k == 40:
            effects.append(_blast((30, 30), 4.0, tick0=30))
    assert saw == 150                     # the volley filled the cap
    assert ss.p.n == 0                    # and every spark died by its life


def test_a_spark_only_cools():
    """Property: a spark's temperature never rises above its birth
    temperature and is non-increasing in age (Newton, closed form). Breaks
    if a re-heat or a non-monotone cooling law appears."""
    mask = _open_room()
    ss = _system()
    effects = [_blast((15, 15))]
    prev = None
    for s in _drive(ss, effects, mask, 20):
        assert np.all(s.p.temp <= s.p.t0 + 1e-9)
        if prev is not None and s.p.n == prev.shape[0]:
            assert np.all(s.p.temp <= prev + 1e-9)
        prev = s.p.temp.copy()


# --- births at the walk's seeds ------------------------------------------------

def test_births_are_at_the_walks_seeds_never_in_a_wall():
    """Property: sparks are born at the tile CENTRES (+0.5) of the blast
    walk's seeds: the centre when open; for a blast centred on a wall,
    beside it (every open 4-neighbour, or the approach side's one), never in
    the wall. Breaks if births ignore the walk's seed rule (a spark born
    inside a wall) or lose the +0.5 tile-centre offset."""
    mask = _open_room()
    mask[:, 15] = False                       # a wall column at x = 15
    cases = [((10, 10), None, {(10, 10)}),               # open centre
             ((15, 10), None, {(10, 14), (10, 16)}),     # on the wall
             ((15, 10), (10, 16), {(10, 16)})]           # ... from the east
    for pos, approach, want in cases:
        ss = _system()
        new = ss.consume([_blast(pos, approach=approach)], _seeds_fn(mask),
                         48)
        assert new["x"].size > 0
        born = set(zip((new["y"] - 0.5).astype(int).tolist(),
                       (new["x"] - 0.5).astype(int).tolist()))
        assert born == want, (pos, approach, born)
        assert np.all(new["x"] % 1.0 == 0.5) and np.all(new["y"] % 1.0 == 0.5)
        iy, ix = new["y"].astype(int), new["x"].astype(int)
        assert mask[iy, ix].all()


def test_a_spark_entering_a_solid_tile_dies_and_never_crosses_a_wall():
    """Property: no alive spark is ever inside a blocked tile, and none
    gets through a one-tile wall even at a speed that would step over it
    in one frame (the substeps hold a hop under a tile). Breaks if the
    blocked kill is dropped or the integration steps a fast spark past a
    wall."""
    mask = _open_room(30, 30)
    mask[:, 18] = False                       # one-tile wall east of the blast
    ss = _system(_settings(v0_tiles_s=60.0, drag=0.0, curl=0.0, life_s=2.0,
                           cool_s=50.0, rate=200.0))
    effects = [_blast((16, 15))]
    born = None
    for k, s in enumerate(_drive(ss, effects, mask, 30, dt=0.1)):
        if k == 0:
            born = s.p.n        # spawned this frame, not yet stepped
            continue
        if s.p.n:
            iy, ix = s.p.y.astype(int), s.p.x.astype(int)
            assert mask[iy, ix].all()
            assert np.all(s.p.x < 18.0), "a spark crossed the wall"
        if k == 1:
            # 0.1 s at >= 30 tiles/s: every eastbound spark (vx > 15) has
            # reached x = 18 -- so it died there, well inside its life
            assert born and s.p.n < born
            assert not np.any(s.p.vx > 15.0)


# --- reproducible, on the VFX clock -------------------------------------------

def test_the_spawn_is_reproducible_from_the_seed():
    """Property: the same effect (same tick0, pos, strength) spawns the
    same sparks, and they step the same; another tick0 spawns other sparks.
    Breaks if the spawn draws from an unseeded or shared stream (a look shot
    would not reproduce)."""
    mask = _open_room()
    runs = []
    for tick0 in (48, 48, 49):
        ss = _system()
        effects = [_blast((12, 12), 2.0, tick0=tick0)]
        for _ in _drive(ss, effects, mask, 6):
            pass
        runs.append(ss.p.d.copy())
    assert runs[0].tobytes() == runs[1].tobytes()
    assert runs[0].shape != runs[2].shape or not np.array_equal(runs[0],
                                                                 runs[2])


def test_the_spawn_runs_on_the_vfx_clock_never_wall_clock(monkeypatch):
    """Property: what the sparks do is a function of the VFX time ``t``
    handed in, never of the wall clock: with the wall clock jumped by hours
    the state is byte-identical, and with ``t`` held still no spark ages.
    Breaks if the spawn or the step reads ``time.time`` / ``perf_counter``
    for anything but its own timing counter."""
    mask = _open_room()

    def run():
        ss = _system()
        effects = [_blast((12, 12), 3.0)]
        for _ in _drive(ss, effects, mask, 5):
            pass
        return ss.p.d.copy()

    a = run()
    real = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: real() + 9e4)
    monkeypatch.setattr(time, "time", lambda: 1e9)
    b = run()
    assert a.tobytes() == b.tobytes()
    ss = _system()
    effects = [_blast((12, 12))]
    for _ in _drive(ss, effects, mask, 1):
        pass
    n0 = ss.p.n
    for _ in range(5):                        # t held: a paused, idle clock
        ss.update(t=1.0 / 60.0, effects=effects, seeds_fn=_seeds_fn(mask),
                  sim_tick=48, wind_fn=lambda: None, tps=24.0,
                  blocked=~mask, modes=_modes())
    assert ss.p.n == n0 and np.all(ss.p.age == 0.0)


def test_a_blast_is_consumed_once_and_only_its_own_flag_is_set():
    """Property: a ``"blast"`` effect spawns sparks exactly once (marked
    ``"sparks_seen"``), later frames never re-throw it, and no other
    consumer's flag (``shock_seen``) is read or written; a non-blast effect
    is ignored. Breaks if the flag is not set, another consumer's flag is
    used, or a disabled system later throws a stale blast."""
    mask = _open_room()
    fx = _blast((10, 10))
    other = {"kind": "tracer", "t": 0.0, "life": 0.1}
    ss = _system()
    first = ss.consume([fx, other], _seeds_fn(mask), 48)
    again = ss.consume([fx, other], _seeds_fn(mask), 48)
    assert first["x"].size > 0 and again["x"].size == 0
    assert fx["sparks_seen"] is True and "shock_seen" not in fx
    assert "sparks_seen" not in other
    off = _system(_settings(enabled=False))
    fx2 = _blast((10, 10))
    for _ in _drive(off, [fx2], mask, 2):
        pass
    assert fx2["sparks_seen"] and off.p.n == 0


# --- the ember path, the draw ---------------------------------------------------

def test_an_ember_only_frame_draws_exactly_the_embers_sprites():
    """Property: with the spark system present but idle, the renderer's one
    particle draw hands the sprite draw the embers' own pack byte for byte
    (no concatenation, no reorder), and with sparks alive it is ONE array
    holding both systems (one draw per stage, never a second call). Breaks
    if the spark system perturbs the ember path or draws on its own."""
    from renderer.game_renderer import GameRenderer
    es = P.EmberSystem(P.EmberSettings(), t_amb_k=T_AMB, glow_min_k=GLOW_MIN)
    rng = np.random.default_rng(1)
    es.p.add(x=rng.uniform(2, 20, 30), y=rng.uniform(2, 20, 30),
             vx=rng.normal(size=30), vy=rng.normal(size=30),
             life=np.full(30, 2.0), t0=np.full(30, 1200.0))
    ss = _system()
    got = []
    draw = SimpleNamespace(draws_last_frame=0, sprites_last_frame=0,
                           draw=lambda sprites, **kw: got.append(sprites) or 1)
    fake = SimpleNamespace(
        embers=es, sparks=ss, ember_draw=draw, _ember_sprites=None,
        # #87: the muzzle sparks ride the same draw; none here
        muzzle_sparks=SimpleNamespace(enabled=False, p=SimpleNamespace(n=0)),
        camera=SimpleNamespace(zoom_px_per_tile=24.0),
        world=SimpleNamespace(world_px_per_tile=48.0, world_px_w=1152.0,
                              world_px_h=1152.0),
        fire_vfx=SimpleNamespace(ramp=SimpleNamespace(kelvin_floor=800.0,
                                                      kelvin_ceil=10000.0),
                                 lut_tex=None),
        _smoke_cast=(None,))
    view = (0.0, 0.0, 24.0, 24.0)
    want = P.pack_sprites(es.p, view, 48.0, es.settings,
                          screen_px_per_world_px=0.5)
    assert GameRenderer._draw_embers(fake, view) == 1
    assert len(got) == 1 and got[0].tobytes() == want.tobytes()
    # sparks alive: still ONE draw, embers first, both systems in it
    ss.consume([_blast((10, 10))], _seeds_fn(_open_room()), 48)
    new = ss.consume([_blast((12, 12), tick0=5)], _seeds_fn(_open_room()), 5)
    ss.p.add(**new)
    got.clear()
    assert GameRenderer._draw_embers(fake, view) == 1
    assert len(got) == 1
    assert got[0].shape[1] == want.shape[1] + ss.p.n
    assert got[0][:, :want.shape[1]].tobytes() == want.tobytes()


def test_sparks_are_coloured_through_the_lut_only():
    """Property: the spark system writes no colour: its particles carry the
    primitive's eight columns (temperature, never RGB), the packed sprite's
    colour input is the temperature (the sprite shader turns it into colour
    through the blackbody GPU LUT), and the spark code never evaluates the
    ramp. Breaks if a spark RGB column or a second T -> colour map appears."""
    ss = _system()
    new = ss.consume([_blast((10, 10))], _seeds_fn(_open_room()), 48)
    ss.p.add(**new)
    assert ss.p.d.shape[1] == len(P._FIELDS) == 8
    sprites = P.pack_sprites(ss.p, None, 48.0, ss.settings)
    assert np.array_equal(sprites[1, :, 0], ss.p.temp.astype(np.float32))
    src = "".join(inspect.getsource(o) for o in (
        P.SparkSystem, P.spark_spawns, P.spark_count, P.spark_rng))
    for bad in ("emission_at_kelvin", "chroma", "rgb", "colour", "color"):
        assert bad not in src.lower(), bad


# --- the door ---------------------------------------------------------------------

def test_config_names_every_spark_dial_and_the_door_refuses():
    """Property: ``[render.sparks]`` names every SparkSettings dial and
    nothing else, the settings carry the config's values, and the door
    refuses a missing section, a missing key and a mistyped value by name.
    Breaks if a dial falls back silently or the config drifts from the
    schema."""
    from config import CFG
    sec = CFG.render.sparks
    assert set(vars(sec)) == set(P.spark_config_keys())
    st = P.spark_settings_from_config(CFG)
    for k in P.spark_config_keys():
        assert getattr(st, k) == getattr(sec, k), k

    def cfg_with(**over):
        d = dict(vars(sec))
        d.update(over)
        for k, v in list(d.items()):
            if v is _DROP:
                del d[k]
        return SimpleNamespace(render=SimpleNamespace(
            sparks=SimpleNamespace(**d)))

    with pytest.raises(ValueError, match="no \\[render.sparks\\]"):
        P.spark_settings_from_config(SimpleNamespace(
            render=SimpleNamespace()))
    with pytest.raises(ValueError, match="missing 'rate'"):
        P.spark_settings_from_config(cfg_with(rate=_DROP))
    with pytest.raises(ValueError, match="cap has the wrong type"):
        P.spark_settings_from_config(cfg_with(cap=512.0))
    with pytest.raises(ValueError, match="enabled has the wrong type"):
        P.spark_settings_from_config(cfg_with(enabled=1))
    with pytest.raises(ValueError, match="v0_tiles_s has the wrong type"):
        P.spark_settings_from_config(cfg_with(v0_tiles_s="12"))
    with pytest.raises(ValueError, match="v_spread"):
        P.spark_settings_from_config(cfg_with(v_spread=1.5))


_DROP = object()
