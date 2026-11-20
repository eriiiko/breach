"""The tracers (#141): a round in flight as a glowing streak -- the settings
door, the kinematics, the wall impacts and the sprite pack
(``renderer/tracers.py``). Every test names the PROPERTY it protects and the
change that must break it.

    <breach-py> -m pytest tests/test_tracers.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import particles as P  # noqa: E402
from renderer import tracers as T  # noqa: E402

TILE_M, TPS = 0.333, 24.0
KIND = dict(speed_scale=1.0, shutter_s=1 / 60, width_m=0.01, temp_k=3600.0,
            brightness=0.7, halo_gain=0.04, glow_s=0.05, glow_gain=0.05,
            glow_temp_k=1500.0, impact_sparks=5, impact_flash_s=0.03,
            impact_flash_m=0.07)
SPARKS = dict(enabled=True, cone_deg=50.0, v0_mps=6.0, v_spread=0.6,
              t_born_k=2800.0, cool_s=0.2, life_s=0.2, life_jitter=0.5,
              drag=6.0, curl=0.6, size_tiles=0.02, min_screen_px=0.8,
              streak_s=0.03, brightness=6.0, cap=256, seed=1)


def _cfg(kind=None, **top):
    tr = dict(enabled=True, default_kind="bullet", weapons={}, seed=3, cap=8,
              min_screen_px=0.75, kinds={"bullet": {**KIND, **(kind or {})}},
              sparks=dict(SPARKS))
    tr.update(top)
    return {"render": {"tracers": tr},
            "weapons": {"k5_carbine": {"ammo_family": "rifle_556"},
                        "lance_3": {"ammo_family": "cell_laser"}},
            "ammo": {"a": {"family": "rifle_556", "muzzle_mps": 900.0},
                     "b": {"family": "rifle_556", "muzzle_mps": 700.0},
                     "c": {"family": "cell_laser"}}}


def _fx(a=(2.0, 5.0), b=(32.0, 5.0), unit=1, tick0=10, launch=True):
    return {"kind": "tracer", "unit": unit, "tick0": tick0, "launch": launch,
            "from": a, "to": b, "t": 0.0, "life": 0.18}


def _tracers(kind=None, **top) -> T.Tracers:
    return T.Tracers(T.settings_from_config(_cfg(kind, **top)))


def _consume(tr, effects, *, solid=None, start=None, speed=900.0, tick=10):
    tr.consume(effects, weapon_of=lambda u: "k5_carbine",
               speed_of=lambda w: speed, start_of=lambda u: start,
               solid=solid, tile_size_m=TILE_M, tps=TPS, sim_tick=tick)


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def test_the_shipped_config_passes_the_doors():
    """PROPERTY: config.toml's own tables load, and the carbine's honest
    speed is its ammo row's. Breaks on a typo in the shipped tables."""
    from config import CFG
    s = T.settings_from_config(CFG)
    assert s.kind_for("k5_carbine") is not None
    P.muzzle_spark_settings(CFG, "tracers")
    assert T.round_speeds(CFG)["k5_carbine"] == float(
        CFG.ammo.rifle_556_standard.muzzle_mps)


@pytest.mark.parametrize("edit, word", [
    (lambda t: t.pop("min_screen_px"), "min_screen_px"),
    (lambda t: t.update(colour=1), "colour"),
    (lambda t: t["kinds"]["bullet"].pop("glow_s"), "glow_s"),
    (lambda t: t["kinds"]["bullet"].update(shutter_s=0.0), "shutter_s"),
    (lambda t: t["kinds"]["bullet"].update(speed="fast"), "speed"),
    (lambda t: t["weapons"].update(nope="bullet"), "nope"),
    (lambda t: t["weapons"].update(lance_3="beam"), "beam"),
    (lambda t: t.update(default_kind="beam"), "beam"),
])
def test_the_door_refuses_by_name(edit, word):
    """PROPERTY: a missing, unknown or mistyped key, an unknown weapon or
    kind is refused NAMING it. Breaks if the door falls back silently."""
    cfg = _cfg()
    edit(cfg["render"]["tracers"])
    with pytest.raises(ValueError, match=word):
        T.settings_from_config(cfg)


def test_the_speed_is_the_rounds_own():
    """PROPERTY: a weapon's tracer speed is its ammo family's first
    ``muzzle_mps``; a weapon whose rounds have none is absent (its tracer
    takes one tick per segment). Breaks if the speed becomes a render dial
    that can drift from the sim's."""
    assert T.round_speeds(_cfg()) == {"k5_carbine": 900.0}


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def test_honest_speed_crosses_a_room_inside_the_first_frame():
    """PROPERTY: at ``speed_scale`` 1 the streak advances at the round's
    real speed -- 900 m/s covers a 30-tile (10 m) path within the first
    frame's exposure. Breaks if a display speed sneaks back in."""
    tr = _tracers()
    _consume(tr, [_fx()])
    tr.update(1.0)
    t = tr.live[0]
    assert t.span() == (0.0, pytest.approx(30.0))
    assert t.speed == pytest.approx(900.0 / TILE_M)


def test_the_streak_covers_the_whole_path_without_a_gap():
    """PROPERTY: stepped at the exposure time, successive streaks tile the
    path from start to end: every point of the flight is drawn once, none
    skipped. Breaks if the head or the tail uses another clock."""
    tr = _tracers(kind=dict(speed_scale=0.05))          # slow enough to step
    _consume(tr, [_fx()])
    spans, t = [], 2.0
    for _ in range(60):
        tr.update(t)
        if not tr.live:
            break
        tail, head = tr.live[0].span()
        if head > tail:
            spans.append((tail, head))
        t += KIND["shutter_s"]
    assert spans[0][0] == 0.0 and spans[-1][1] == pytest.approx(30.0)
    assert len(spans) > 5
    for (_, h0), (t1, _) in zip(spans, spans[1:]):
        assert t1 == pytest.approx(h0, abs=1e-6)


def test_speed_scale_slows_the_streak_and_nothing_else():
    tr_a, tr_b = _tracers(), _tracers(kind=dict(speed_scale=0.25))
    for tr in (tr_a, tr_b):
        _consume(tr, [_fx()])
        tr.update(0.0)
    assert tr_b.live[0].speed == pytest.approx(0.25 * tr_a.live[0].speed)
    assert tr_b.live[0].a == tr_a.live[0].a and tr_b.live[0].b == tr_a.live[0].b


def test_a_round_with_no_known_speed_takes_one_tick_per_segment():
    """PROPERTY: without a muzzle speed (a plasma bolt) the streak covers
    its segment in exactly one sim tick -- what a round still flying does.
    Breaks if such a round gets no tracer or an arbitrary speed."""
    tr = _tracers()
    _consume(tr, [_fx(b=(8.0, 5.0))], speed=None)
    assert tr.live[0].speed == pytest.approx(6.0 * TPS)


def test_a_tracer_never_outlives_its_flight_and_glow():
    """PROPERTY: once the round has arrived and its afterglow and impact
    flash are below the floor, the tracer is gone; a paused clock holds it.
    Breaks if a tracer lingers or dies while the clock stands."""
    tr = _tracers()
    _consume(tr, [_fx()])
    for _ in range(4):
        tr.update(5.0)
    assert len(tr.live) == 1 and tr.live[0].age == 0.0
    tr.update(5.0 + 0.5 * KIND["glow_s"])
    assert len(tr.live) == 1
    tr.update(5.0 + 5.0 * KIND["glow_s"])
    assert tr.live == []


def test_the_first_segment_starts_at_the_muzzle():
    """PROPERTY: a launched round's streak starts at the shooter's muzzle,
    a later segment where the sim put it, and a round that ended short of
    the muzzle keeps the sim's start (never a streak flying backwards).
    Breaks if the start ignores ``launch`` or the short-path guard goes."""
    tr = _tracers()
    _consume(tr, [_fx(), _fx(a=(40.0, 5.0), b=(70.0, 5.0), launch=False),
                  _fx(a=(2.0, 9.0), b=(2.5, 9.0), tick0=11)],
             start=(3.5, 5.2))
    a = [t.a for t in tr.live]
    assert a[0] == (3.5, 5.2)
    assert a[1] == (40.0, 5.0)
    assert a[2] == (2.0, 9.0)            # the muzzle lies beyond this round's end
    assert tr.live[0].b == (32.0, 5.0)


def test_one_tracer_per_round_and_the_cap_holds():
    """PROPERTY: a re-delivered effect is one tracer; two rounds of one
    tick are two; a shooter that is not drawn has none; never more than
    ``cap`` alive. Breaks if the key or the flag goes."""
    tr = _tracers()
    fx = [_fx(), _fx(), _fx(b=(32.0, 6.0))]
    _consume(tr, fx)
    _consume(tr, [_fx()])
    assert len(tr.live) == 2 and all(e["streak_seen"] for e in fx)
    tr.consume([_fx(unit=7)], weapon_of=lambda u: None,
               speed_of=lambda w: 900.0, start_of=lambda u: None, solid=None,
               tile_size_m=TILE_M, tps=TPS, sim_tick=10)
    assert len(tr.live) == 2
    _consume(tr, [_fx(b=(32.0, 7.0 + i)) for i in range(20)])
    assert len(tr.live) == 8


# ---------------------------------------------------------------------------
# Impacts
# ---------------------------------------------------------------------------

def _room():
    solid = np.zeros((12, 40), bool)
    solid[:, 33:] = True
    return solid


def test_a_wall_stop_is_read_from_the_map():
    """PROPERTY: a round whose path ends at (or one step short of) a solid
    tile stopped on a wall; one that ends in the open did not. Breaks if
    the read looks at the wrong end or off the map."""
    solid = _room()
    assert T.stopped_on_wall(solid, (2.0, 5.0), (33.2, 5.0))
    assert T.stopped_on_wall(solid, (2.0, 5.0), (32.6, 5.0))
    assert not T.stopped_on_wall(solid, (2.0, 5.0), (20.0, 5.0))
    assert not T.stopped_on_wall(solid, (2.0, 5.0), (2.0, 5.0))
    assert not T.stopped_on_wall(None, (2.0, 5.0), (33.2, 5.0))
    assert not T.stopped_on_wall(solid, (2.0, 5.0), (2.0, 200.0))   # off the map


def test_an_impact_is_reported_once_when_the_streak_arrives():
    """PROPERTY: the sparks' cue fires on the frame the streak's head
    reaches a wall, once, backed off the wall, pointing back along the
    flight; a round that stopped in the open gives none. Breaks if the cue
    fires at the sim's tick instead of the streak's arrival, repeats, or
    points into the wall."""
    tr = _tracers(kind=dict(speed_scale=0.05))
    _consume(tr, [_fx(b=(33.2, 5.0)), _fx(a=(2.0, 8.0), b=(20.0, 8.0))],
             solid=_room())
    t, seen = 3.0, []
    for i in range(80):
        tr.update(t)
        seen += [(i, imp) for imp in tr.impacts_last_frame]
        t += KIND["shutter_s"]
    assert len(seen) == 1
    frame, imp = seen[0]
    assert frame > 3                         # not on the first frame: it flew
    assert imp.dir_x == pytest.approx(-1.0) and imp.dir_y == pytest.approx(0.0)
    assert imp.x == pytest.approx(33.2 - T.IMPACT_BACK_TILES) and imp.y == 5.0
    assert not _room()[int(imp.y), int(imp.x)]
    assert imp.sparks == KIND["impact_sparks"]


# ---------------------------------------------------------------------------
# The pack
# ---------------------------------------------------------------------------

def test_the_pack_is_the_particle_draws_layout():
    """PROPERTY: each sprite is a segment in the particle draw's rows --
    head at ``p1``, velocity ``p1 - p0`` with streak 1 s -- and its weight
    undoes the sprite shader's ``r / (r + seg)`` dimming, so the core
    radiance is the kind's whatever the streak's length. Breaks if the
    layout drifts from ``particle_sprite.vs`` or a long streak goes dim."""
    tr = _tracers(kind=dict(halo_gain=0.0, glow_gain=0.0, speed_scale=0.01))
    _consume(tr, [_fx()])
    tr.update(0.0)
    ppt = 48.0
    spr = tr.pack(None, ppt, TILE_M)
    assert spr.shape == (2, 1, 4) and spr.dtype == np.float32
    tail, head = tr.live[0].span()
    assert tuple(spr[0, 0, :2]) == pytest.approx(((2.0 + head) * ppt, 5.0 * ppt))
    assert tuple(spr[0, 0, 2:]) == pytest.approx(((head - tail) * ppt, 0.0))
    temp, weight, r, streak = spr[1, 0]
    seg = 0.5 * (head - tail) * ppt
    assert (temp, streak) == (KIND["temp_k"], 1.0)
    assert r == pytest.approx(KIND["width_m"] * ppt / TILE_M)
    assert weight * r / (r + seg) == pytest.approx(KIND["brightness"], rel=1e-5)


def test_the_parts_of_a_tracer_are_its_sprites():
    """PROPERTY: in flight a tracer is a streak + its halo + the afterglow
    behind it; after arrival at a wall, the afterglow + the impact flash;
    out of view, nothing. Breaks if a part is dropped or drawn when spent."""
    tr = _tracers(kind=dict(speed_scale=0.05))
    _consume(tr, [_fx(b=(33.2, 5.0))], solid=_room())
    tr.update(0.0)
    assert tr.pack(None, 48.0, TILE_M).shape[1] == 2      # nothing behind it yet
    tr.update(0.05)
    assert tr.pack(None, 48.0, TILE_M).shape[1] == 3
    assert tr.pack((50.0, 50.0, 60.0, 60.0), 48.0, TILE_M).shape[1] == 0
    arrive = tr.live[0].length / tr.live[0].speed
    tr.update(arrive + 0.004)
    spr = tr.pack(None, 48.0, TILE_M)
    assert spr.shape[1] == 2
    flash = spr[:, 1]
    assert tuple(flash[0, 2:]) == (0.0, 0.0)             # a point, at the wall
    assert flash[0, 0] == pytest.approx(33.2 * 48.0)
    assert flash[1, 2] == pytest.approx(KIND["impact_flash_m"] * 48.0 / TILE_M)


def test_zoomed_out_a_streak_keeps_its_light():
    """PROPERTY: below the screen-pixel floor the streak is drawn wider and
    proportionally dimmer -- the same light per unit length, so it does not
    vanish between pixels. Breaks if the floor is dropped or adds light."""
    tr = _tracers(kind=dict(halo_gain=0.0, glow_gain=0.0, speed_scale=0.01))
    _consume(tr, [_fx()])
    tr.update(0.0)
    near = tr.pack(None, 48.0, TILE_M, screen_px_per_world_px=1.0)[1, 0]
    far = tr.pack(None, 48.0, TILE_M, screen_px_per_world_px=0.125)[1, 0]
    assert far[2] == pytest.approx(0.75 / 0.125) and far[2] > near[2]
    seg = 0.5 * (tr.live[0].span()[1]) * 48.0

    def per_length(row):          # core radiance x radius
        return row[1] * row[2] / (row[2] + seg) * row[2]

    assert per_length(far) == pytest.approx(per_length(near), rel=1e-4)


def test_impact_sparks_use_the_tracers_own_dials():
    """PROPERTY: the impacts' sparks read ``[render.tracers.sparks]``, the
    muzzle's ``[render.muzzle_flame.sparks]`` -- one door, two tables.
    Breaks if the table argument is ignored."""
    cfg = _cfg()
    cfg["render"]["muzzle_flame"] = {"sparks": {**SPARKS, "cone_deg": 28.0}}
    assert P.muzzle_spark_settings(cfg, "tracers").cone_deg == 50.0
    assert P.muzzle_spark_settings(cfg).cone_deg == 28.0
    del cfg["render"]["tracers"]["sparks"]["drag"]
    with pytest.raises(ValueError, match=r"render\.tracers\.sparks.*drag"):
        P.muzzle_spark_settings(cfg, "tracers")
    assert math.isfinite(P.muzzle_spark_settings(cfg).drag)
