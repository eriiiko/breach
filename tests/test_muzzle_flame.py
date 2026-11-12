"""The muzzle flame (#87): its settings door, its law (the numpy twin of
``shaders/muzzle_flame.fs``), where it sits, and its CPU state.

Design: ``docs/muzzle_flame_design_2026-11-12.md``. Every test names the
PROPERTY it protects and the change that must break it. The shader itself is
held to the twin by ``tests/gl_muzzle_flame_check.py``.

    <breach-py> -m pytest tests/test_muzzle_flame.py -q
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import muzzle_flame as mf  # noqa: E402
from renderer import particles as P  # noqa: E402
from renderer.blackbody import BlackbodyRamp  # noqa: E402

RAMP = BlackbodyRamp(kelvin_ref=1300.0, intensity_exponent=2.0)
NOISE = mf.noise_field()

KIND = dict(size=1.0, length_m=0.95, width_m=0.13, side_jets=2,
            side_len_m=0.55, side_width_m=0.085, side_angle_deg=55.0,
            core_m=0.10, halo_m=0.30, halo_gain=0.035, t_core_k=4200.0,
            t_edge_k=1050.0, gain=1.6, tear=0.8, fireball_chance=0.12,
            duration_s=0.055, sparks=3)
SPARKS = dict(enabled=True, cone_deg=28.0, v0_mps=9.0, v_spread=0.5,
              t_born_k=2600.0, cool_s=0.25, life_s=0.22, life_jitter=0.5,
              drag=5.0, curl=0.6, size_tiles=0.022, min_screen_px=0.8,
              streak_s=0.03, brightness=6.0, cap=256, seed=1)


def _cfg(**top):
    flame = dict(enabled=True, default_kind="rifle",
                 weapons={"k5_carbine": "rifle"}, seed=7, cap=8,
                 kinds={"rifle": dict(KIND)}, sparks=dict(SPARKS))
    flame.update(top)
    return {"render": {"muzzle_flame": flame},
            "weapons": {"k5_carbine": {}, "lance_3": {}}}


def _kind(**kw) -> mf.FlameKind:
    return mf.FlameKind(**{**KIND, **kw})


def _grid(half: float, n: int = 161):
    c = np.linspace(-half, half, n)
    return np.meshgrid(c, c)


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def test_the_shipped_config_passes_both_doors():
    """PROPERTY: config.toml's own tables load, and every weapon it maps
    names a kind row. Breaks on a typo in the shipped tables."""
    from config import CFG
    s = mf.settings_from_config(CFG)
    assert s.kinds and s.kind_for("k5_carbine") is not None
    P.muzzle_spark_settings(CFG)


@pytest.mark.parametrize("edit, word", [
    (lambda f: f.pop("cap"), "cap"),
    (lambda f: f.update(bogus=1), "bogus"),
    (lambda f: f.update(enabled=1), "enabled"),
    (lambda f: f["kinds"]["rifle"].pop("tear"), "tear"),
    (lambda f: f["kinds"]["rifle"].update(tear="x"), "tear"),
    (lambda f: f["kinds"]["rifle"].update(colour=1), "colour"),
    (lambda f: f["kinds"]["rifle"].update(side_jets=5), "side_jets"),
    (lambda f: f["kinds"]["rifle"].update(t_edge_k=5000.0), "t_core_k"),
    (lambda f: f["kinds"]["rifle"].update(size=0.0), "size"),
    (lambda f: f["weapons"].update(no_such_gun="rifle"), "no_such_gun"),
    (lambda f: f["weapons"].update(lance_3="plasma"), "plasma"),
    (lambda f: f.update(default_kind="plasma"), "plasma"),
])
def test_the_door_refuses_by_name(edit, word):
    """PROPERTY: a missing, unknown or mistyped key, an unknown weapon id or
    kind name is refused NAMING it -- never a silent default. Breaks if the
    door starts falling back."""
    cfg = _cfg()
    edit(cfg["render"]["muzzle_flame"])
    with pytest.raises(ValueError, match=word):
        mf.settings_from_config(cfg)


def test_a_weapon_can_be_given_no_flame():
    """PROPERTY: ``""`` means no flame, for a weapon row and for the
    default. Breaks if an empty name is looked up as a row."""
    s = mf.settings_from_config(_cfg(weapons={"lance_3": ""}))
    assert s.kind_for("lance_3") is None
    assert s.kind_for("k5_carbine") is s.kinds["rifle"]     # the default
    s = mf.settings_from_config(_cfg(default_kind="", weapons={}))
    assert s.kind_for("k5_carbine") is None


@pytest.mark.parametrize("edit, word", [
    (lambda d: d.pop("drag"), "drag"),
    (lambda d: d.update(extra=1.0), "extra"),
    (lambda d: d.update(life_s=0.0), "life_s"),
    (lambda d: d.update(v_spread=1.0), "v_spread"),
])
def test_the_spark_door_refuses_by_name(edit, word):
    cfg = _cfg()
    edit(cfg["render"]["muzzle_flame"]["sparks"])
    with pytest.raises(ValueError, match=word):
        P.muzzle_spark_settings(cfg)


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------

def test_a_shot_reproduces_and_two_shots_differ():
    """PROPERTY: the same (seed, unit, tick) draws the same flame bit for
    bit (a look shot reproduces); another tick or another unit draws
    another. Breaks if the shape reads a global generator or ignores a key."""
    k = _kind()
    a = mf.shot_rows(k, mf.shot_rng(7, 3, 100))
    b = mf.shot_rows(k, mf.shot_rng(7, 3, 100))
    assert a.rows.tobytes() == b.rows.tobytes() and a.ofs == b.ofs
    for other in (mf.shot_rng(7, 3, 101), mf.shot_rng(7, 4, 100),
                  mf.shot_rng(8, 3, 100)):
        assert mf.shot_rows(k, other).rows.tobytes() != a.rows.tobytes()


def test_the_flame_is_dark_outside_its_square_and_lit_at_the_muzzle():
    """PROPERTY: the radiance is exactly 0 at and beyond the shot's square
    (the quad the shader draws: nothing may be cut off at its edge) and
    bright at the muzzle. Breaks if a term (the halo) reaches past the quad
    or the window is dropped."""
    for tick in range(12):
        shot = mf.shot_rows(_kind(), mf.shot_rng(7, 1, tick))
        h = shot.half_m
        x, y = _grid(1.5 * h)
        img = mf.flame_radiance(x, y, shot, 0.0, NOISE, RAMP)
        outside = np.maximum(np.abs(x), np.abs(y)) >= h
        assert outside.any() and np.all(img[outside] == 0.0)
        assert np.all(img >= 0.0)
        centre = mf.flame_radiance(np.array(0.0), np.array(0.0), shot, 0.0,
                                   NOISE, RAMP)
        assert centre.max() > 1.0, "the muzzle's core is not above white"
        # what the window removes is nothing one could see
        edge = np.maximum(np.abs(x), np.abs(y)) >= h * mf.EDGE_FROM
        assert img[edge].max() < 0.02


def test_the_flame_fades_to_nothing_with_age():
    """PROPERTY: the total light never grows after the lengths have leapt
    out, ends at exactly nothing at age 1, and the core cools. Breaks if a
    term stops fading (a flash that lingers) or re-heats."""
    shot = mf.shot_rows(_kind(), mf.shot_rng(7, 1, 5))
    x, y = _grid(shot.half_m, 121)
    ages = (0.34, 0.5, 0.7, 0.9, 0.99)
    totals = [mf.flame_radiance(x, y, shot, a, NOISE, RAMP).sum()
              for a in ages]
    assert all(b < a for a, b in zip(totals, totals[1:]))
    assert totals[-1] < 0.02 * totals[0]
    assert mf.flame_radiance(x, y, shot, 1.0, NOISE, RAMP).sum() == 0.0
    core = [mf.flame_radiance(np.array(0.0), np.array(0.0), shot, a, NOISE,
                              RAMP).sum() for a in (0.0, 0.5, 0.9)]
    assert core[0] > core[1] > core[2]


def test_size_scales_the_whole_flame():
    """PROPERTY: ``size`` is THE size dial -- the same shot at size 2 is the
    size-1 flame drawn twice as large (every length doubles, nothing else
    moves). Breaks if a length is left out of the scaling."""
    one = mf.shot_rows(_kind(size=1.0), mf.shot_rng(7, 1, 9))
    two = mf.shot_rows(_kind(size=2.0), mf.shot_rng(7, 1, 9))
    assert two.half_m == pytest.approx(2.0 * one.half_m, rel=1e-6)
    np.testing.assert_allclose(two.rows[0, :2], 2.0 * one.rows[0, :2])
    np.testing.assert_allclose(two.rows[2:, 1:3], 2.0 * one.rows[2:, 1:3],
                               rtol=1e-6)
    np.testing.assert_array_equal(two.rows[2:, 0], one.rows[2:, 0])
    # the densities agree at corresponding points away from the torn edge
    x, y = _grid(0.05, 9)
    d1 = mf.flame_density(x, y, one, 0.0, NOISE)
    d2 = mf.flame_density(2 * x, 2 * y, two, 0.0, NOISE)
    np.testing.assert_allclose(d1, d2, atol=1e-6)


def test_the_tongue_points_along_the_barrel():
    """PROPERTY: the light's centre of mass lies ahead of the muzzle (+x),
    for every shot. Breaks if the frame is mirrored or a jet's sign flips."""
    for tick in range(10):
        shot = mf.shot_rows(_kind(side_jets=0), mf.shot_rng(7, 1, tick))
        x, y = _grid(shot.half_m, 121)
        w = mf.flame_density(x, y, shot, 0.2, NOISE)
        assert (w * x).sum() / w.sum() > 0.05


def test_a_fireball_is_shorter_and_fatter():
    """PROPERTY: a fireball shot's tongue is shorter and wider and its core
    bigger than an ordinary shot's range allows. Breaks if the fireball
    branch stops changing the shape."""
    balls = [mf.shot_rows(_kind(fireball_chance=1.0), mf.shot_rng(7, 1, t))
             for t in range(20)]
    plain = [mf.shot_rows(_kind(fireball_chance=0.0), mf.shot_rng(7, 1, t))
             for t in range(20)]
    assert all(s.fireball for s in balls) and not any(s.fireball for s in plain)
    assert max(s.rows[2, 1] for s in balls) < min(s.rows[2, 1] for s in plain) * 1.1 + 0.2
    assert min(s.rows[2, 2] for s in balls) > max(s.rows[2, 2] for s in plain)
    assert balls[0].rows[0, 0] > plain[0].rows[0, 0]


# ---------------------------------------------------------------------------
# Where it sits
# ---------------------------------------------------------------------------

def test_the_muzzle_follows_the_prop_matrix():
    """PROPERTY: the flame's place is the prop matrix applied to the
    ``muzzle`` point, its direction the matrix's +x axis on the floor plane
    (world x, z), in tiles. Breaks if the point skips the matrix (a flame
    left behind by a turning gun) or the axes are swapped."""
    ppt = 48.0
    yaw = math.radians(30.0)
    m = np.eye(4)
    m[:3, :3] = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
                          [-math.sin(yaw), 0, math.cos(yaw)]]) * 0.7
    m[:3, 3] = (480.0, 40.0, 960.0)
    x, y, dx, dy = mf.muzzle_pose_from_matrix(m, (0.5, 0.01, 0.0), ppt)
    assert (dx, dy) == pytest.approx((math.cos(yaw), -math.sin(yaw)))
    assert x == pytest.approx((480.0 + 0.35 * math.cos(yaw)) / ppt)
    assert y == pytest.approx((960.0 - 0.35 * math.sin(yaw)) / ppt)
    up = np.eye(4)
    up[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # barrel straight up
    assert mf.muzzle_pose_from_matrix(up, (0.5, 0, 0), ppt) is None


def test_the_fallback_is_the_footprint_edge_along_the_facing():
    x, y, dx, dy = mf.fallback_pose(10.0, 20.0, 3.0, 0.0)
    assert (x, y, dx, dy) == pytest.approx((13.0, 21.5, 1.0, 0.0))
    x, y, dx, dy = mf.fallback_pose(10.0, 20.0, 3.0, math.pi / 2)
    assert (x, y) == pytest.approx((11.5, 23.0)) and dy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------

def _tracer(unit=1, tick0=10, launch=True):
    return {"kind": "tracer", "unit": unit, "tick0": tick0, "launch": launch,
            "t": 0.0, "life": 0.18}


def _flames(**top) -> mf.MuzzleFlames:
    return mf.MuzzleFlames(mf.settings_from_config(_cfg(**top)))


POSE = (5.0, 6.0, 1.0, 0.0)


def test_one_flash_per_unit_and_tick():
    """PROPERTY: a shot is one flash however often its effect is seen again
    (#85: effects are re-delivered) and however many rounds a burst
    carries; a round in flight (``launch`` false) and a shooter that is not
    drawn flash nothing. Breaks if the (unit, tick) key or the flag goes."""
    fl = _flames()
    fx = [_tracer(), _tracer(), _tracer(launch=False, tick0=11),
          _tracer(unit=2), _tracer(unit=9)]
    weapon_of = {1: "k5_carbine", 2: "k5_carbine"}.get
    fl.consume(fx, weapon_of, 10)
    fl.consume(fx, weapon_of, 10)
    fl.consume([_tracer()], weapon_of, 10)          # the same shot, a new dict
    assert sorted((f.unit, f.tick0) for f in fl.flashes) == [(1, 10), (2, 10)]
    assert all(e.get("flame_seen") for e in fx)


def test_a_flash_shows_its_first_frame_and_never_outlives_its_duration():
    """PROPERTY: a flash is at age 0 on the first frame it is drawn,
    whenever its tick ran; it ages on the SIM clock (a paused clock holds
    it) and is gone once ``duration_s`` of sim time has passed. Breaks if it
    ages from the tick (a late frame shows a half-spent flash), on the wall
    clock, or lingers."""
    fl = _flames()
    fl.consume([_tracer()], lambda u: "k5_carbine", 10)
    fl.update(3.30, lambda u: POSE)
    assert [f.age for f in fl.flashes] == [0.0]
    assert [f.unit for f in fl.born_last_frame] == [1]
    for _ in range(3):                               # paused: the clock stands
        fl.update(3.30, lambda u: POSE)
        assert fl.flashes[0].age == 0.0 and not fl.born_last_frame
    fl.update(3.30 + 0.5 * KIND["duration_s"], lambda u: POSE)
    assert fl.flashes[0].age == pytest.approx(0.5)
    fl.update(3.30 + KIND["duration_s"], lambda u: POSE)
    assert fl.flashes == []


def test_a_flash_stays_on_its_gun_and_keeps_its_last_place():
    """PROPERTY: each frame the flash is where its shooter's muzzle is NOW;
    when the shooter is no longer drawn it keeps its last place; one that
    never had a place is dropped. Breaks if the place is frozen at birth."""
    fl = _flames()
    fl.consume([_tracer()], lambda u: "k5_carbine", 10)
    fl.update(1.0, lambda u: POSE)
    fl.update(1.01, lambda u: (7.0, 6.0, 0.0, 1.0))
    assert fl.flashes[0].pose == (7.0, 6.0, 0.0, 1.0)
    fl.update(1.02, lambda u: None)
    assert fl.flashes[0].pose == (7.0, 6.0, 0.0, 1.0)
    fl.consume([_tracer(unit=2)], lambda u: "k5_carbine", 10)
    fl.update(1.03, lambda u: None)
    assert [f.unit for f in fl.flashes] == [1]


def test_the_cap_holds_and_off_is_empty():
    """PROPERTY: never more than ``cap`` flashes alive (the newest kept);
    disabled, nothing is ever born but the effects are still marked seen (no
    burst of stale flashes on re-enable)."""
    fl = _flames(cap=3)
    fx = [_tracer(unit=u) for u in range(10)]
    fl.consume(fx, lambda u: "k5_carbine", 10)
    assert [f.unit for f in fl.flashes] == [7, 8, 9]
    off = _flames(enabled=False)
    fx = [_tracer()]
    off.consume(fx, lambda u: "k5_carbine", 10)
    assert off.flashes == [] and fx[0]["flame_seen"]
    off.refresh_settings(mf.settings_from_config(_cfg()))
    off.consume(fx, lambda u: "k5_carbine", 10)
    assert off.flashes == []


def test_the_pack_is_what_the_shader_reads():
    """PROPERTY: the pack's rows are the muzzle in world px, the direction,
    the age, the noise offset, px per metre and the shot's rows; a flash out
    of view is not packed. Breaks if a row's layout drifts from the
    shader's header."""
    fl = _flames()
    fl.consume([_tracer()], lambda u: "k5_carbine", 10)
    fl.update(1.0, lambda u: POSE)
    f = fl.flashes[0]
    a = fl.pack(None, 48.0, 0.333)
    assert a.shape == (mf.ROWS, 1, 4) and a.dtype == np.float32
    assert tuple(a[0, 0]) == pytest.approx((240.0, 288.0, 1.0, 0.0))
    assert tuple(a[1, 0]) == pytest.approx((0.0, *f.shot.ofs, 48.0 / 0.333),
                                           rel=1e-6)
    assert a[2:, 0].tobytes() == f.shot.rows.tobytes()
    assert fl.pack((50.0, 50.0, 60.0, 60.0), 48.0, 0.333).shape[1] == 0
    assert fl.pack((0.0, 0.0, 10.0, 10.0), 48.0, 0.333).shape[1] == 1


def test_the_twins_picture_puts_the_flame_ahead_of_the_muzzle():
    """PROPERTY: ``rasterise`` (the GL check's reference) draws the pack in
    world px with y down along the packed direction. Breaks if its frame
    disagrees with the vertex shader's (dir, normal) convention."""
    fl = _flames()
    fl.consume([_tracer()], lambda u: "k5_carbine", 10)
    fl.update(1.0, lambda u: (2.0, 2.0, 0.0, 1.0))     # barrel pointing south
    img = mf.rasterise(fl.pack(None, 32.0, 0.333), 128, 192, NOISE, RAMP)
    w = img.sum(axis=-1)
    ys, xs = np.mgrid[0:192, 0:128]
    cy, cx = (w * ys).sum() / w.sum(), (w * xs).sum() / w.sum()
    assert cy > 64.0 + 4.0 and abs(cx - 64.0) < 12.0


# ---------------------------------------------------------------------------
# The sparks
# ---------------------------------------------------------------------------

def test_sparks_leave_the_muzzle_forward():
    """PROPERTY: a shot's sparks are born AT the muzzle and fly forward in
    the cone, at ``v0_mps`` converted to tiles/s. Breaks if they spawn at
    the unit, fly every way (the blast's law), or keep metres as tiles."""
    s = P.muzzle_spark_settings(_cfg())
    sp = P.muzzle_spark_spawns(400, (5.0, 6.0, 0.0, 1.0), 0.333, s,
                               np.random.default_rng(1))
    assert np.all(sp["x"] == 5.0) and np.all(sp["y"] == 6.0)
    speed = np.hypot(sp["vx"], sp["vy"])
    v0 = 9.0 / 0.333
    assert speed.min() >= v0 * 0.5 - 1e-9 and speed.max() <= v0 * 1.5 + 1e-9
    assert (sp["vy"] > 0).mean() > 0.97          # forward = +y here
    assert abs(np.mean(sp["vx"])) < 0.15 * v0


def test_the_spark_system_throws_only_what_was_queued():
    """PROPERTY: the muzzle sparks are born from the flame's queue alone
    (never from a blast effect), capped, and not while disabled. Breaks if
    the subclass falls back to the blast sparks' consume."""
    s = P.muzzle_spark_settings(_cfg())
    sys_ = P.MuzzleSparkSystem(s, t_amb_k=293.0, glow_min_k=800.0)
    kw = dict(effects=[{"kind": "blast", "pos": (3, 3), "strength": 4.0}],
              seeds_fn=None, sim_tick=1, wind_fn=lambda: np.zeros((8, 8, 2)),
              tps=24.0, blocked=np.zeros((8, 8), bool), modes=None)
    s0 = copy.copy(s)
    object.__setattr__(s0, "curl", 0.0)
    sys_.refresh_settings(s0)
    sys_.update(t=0.0, **kw)
    assert sys_.p.n == 0 and "sparks_seen" not in kw["effects"][0]
    sys_.throw(5, (4.0, 4.0, 1.0, 0.0), 0.333, np.random.default_rng(2))
    sys_.update(t=0.01, **kw)
    assert sys_.p.n == 5
    sys_.update(t=0.02, **kw)
    assert sys_.p.n == 5 and np.all(sys_.p.x > 4.0)   # no second birth; moved
    sys_.update(t=5.0, **kw)
    sys_.update(t=5.2, **kw)
    assert sys_.p.n == 0                              # all past their lives
    sys_.enabled_override = False
    sys_.throw(5, (4.0, 4.0, 1.0, 0.0), 0.333, np.random.default_rng(2))
    sys_.update(t=5.3, **kw)
    assert sys_.p.n == 0
