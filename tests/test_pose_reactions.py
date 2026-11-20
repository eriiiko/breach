"""#163 -- the pose layer's first reactions: the marine fires while moving
(aim hold + retriggered recoil, the aim-vs-travel twist) and gets hit (the
flinch additive). Properties only.

Spec: docs/animation_milestones_design_2026-11-02.md §4.1, §4.3, §4.5
(corrections 3 and 7) and issue #163. Every test names the property it
protects and the change that must break it. Bone names come from the
config (``[render.pose.skeletons.*]``), never from this file.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_pose_reactions.py -q
"""
from __future__ import annotations

import dataclasses
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from renderer.pose_clips import trs_matrices  # noqa: E402
from renderer.pose_layer import (NO_CLIP, PoseConfigError, PoseLayer,  # noqa: E402
                                 pose_settings)
from renderer.pose_modifiers import steer  # noqa: E402
from renderer.unit_looks import all_look_names, look_path, looks_table  # noqa: E402
from simulation.events import ShotFiredEvent, UnitHitEvent  # noqa: E402

from _pose_glb import CRITTER_BONES, critter_clips, critter_settings, write_glb  # noqa: E402
from _pose_looks import quaternius_looks  # noqa: E402

# The layer-level reaction tests run on the Quaternius looks (the two-sector
# twist, the pack's clips); the renderer stubs register every shipped look
# (#184: the player is a SOMA look; its locomotion: tests/test_pose_locomotion.py)
SHIPPED = quaternius_looks()
ALL_SHIPPED = all_look_names(looks_table(CFG))
SETTINGS = pose_settings(CFG)
LOOK = SHIPPED[0]
DT = 1.0 / 60.0
AT = SETTINGS.aim_travel
FIRE = SETTINGS.fire
HIT = SETTINGS.hit


@pytest.fixture(scope="module")
def layer() -> PoseLayer:
    pl = PoseLayer(SETTINGS)
    for name in SHIPPED:
        pl.add_look(name, look_path(name))
    return pl


def _twist_rule(pl, look=LOOK):
    return pl.settings.twists[pl.rig(look).skeleton_table]


def _bones(pl, look, names):
    sk = pl.rig(look).bank.skeleton
    return [sk.index(n) for n in names]


def _subtree(pl, look, name):
    sk = pl.rig(look).bank.skeleton
    return sk.subtree(sk.index(name))


def _mats(global_trs):
    return trs_matrices(global_trs[..., :3], global_trs[..., 3:7], global_trs[..., 7:])


def _qangle(q1, q2):
    d = np.abs((q1 * q2).sum(-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(d)


def _yaw_between(m_a, m_b):
    """The rotation about the model's vertical (+Y) taking each bone of
    ``m_a`` to ``m_b`` (radians), measured on the bone's x axis."""
    xa = m_a[..., [0, 2], 0]
    xb = m_b[..., [0, 2], 0]
    ang = np.arctan2(xb[..., 0], xb[..., 1]) - np.arctan2(xa[..., 0], xa[..., 1])
    return (ang + np.pi) % (2 * np.pi) - np.pi


def _detached(pl, look, plain_trs, twisted_trs):
    """Largest distance (model units) by which a bone of the twisted pose
    sits off the point its twisted parent's matrix carries its un-twisted
    offset to -- 0 for a skeleton that stays connected."""
    sk = pl.rig(look).bank.skeleton
    kids = np.flatnonzero(sk.parents >= 0)
    par = sk.parents[kids]
    gp, gt = _mats(plain_trs.astype(np.float64)), _mats(twisted_trs.astype(np.float64))
    offset = np.einsum("bij,bj->bi", np.linalg.inv(gp[par]),
                       np.concatenate([plain_trs[kids, :3], np.ones((len(kids), 1))], 1))
    carried = np.einsum("bij,bj->bi", gt[par], offset)[:, :3]
    return float(np.abs(carried - twisted_trs[kids, :3]).max())


def _walk(pl, look=LOOK, t=0.4):
    ps = pl.new_state(look, "walk")
    pl.drive(ps, "walk", t)
    return ps


def _clone(pl, ps):
    out = pl.new_state(ps.look, ps.state)
    for slot in type(ps).__slots__:
        v = getattr(ps, slot)
        setattr(out, slot, v.copy() if isinstance(v, np.ndarray) else v)
    return out


# ---------------------------------------------------------------------------
# aim vs travel: the twist modifier
# ---------------------------------------------------------------------------

def test_travel_along_the_facing_leaves_the_pose_untouched(layer):
    """PROPERTY (#161 / design §1.1): a unit without an aim intent faces its
    travel; then the relative travel is 0 and the twist modifier must do
    nothing -- after any number of frames walking straight (and standing),
    the twist is 0, the legs play forwards and the composed pose equals the
    pose composed with NO modifier, bit for bit.

    BREAKS IF: the modifier twists on a zero relative travel, the filter
    drifts off 0, or the leg rate is flipped without a backpedal.
    """
    ps = _walk(layer)
    for k in range(120):
        layer.steer(ps, 0.0 if k % 3 else None, True, DT)
        layer.drive(ps, "walk", DT)
    for _ in range(30):
        layer.steer(ps, None, False, DT)
        layer.drive(ps, "idle", DT)
    assert ps.twist == 0.0 and (ps.layer_rate == 1.0).all() and not ps.back
    with_mod = layer.compose(LOOK, [ps]).global_trs
    without = layer.compose(LOOK, [ps], modifiers=()).global_trs
    assert np.array_equal(with_mod, without)


@pytest.mark.parametrize("deg", [-85.0, -60.0, -25.0, 10.0, 45.0, 80.0])
def test_torso_and_head_keep_the_facing_while_the_legs_turn(layer, deg):
    """PROPERTY (design §4.5 correction 7): for a travel within +-theta of
    the facing, after steering, the LOWER bone has yawed toward the travel
    (by the twist) while every bone at or below the LAST counter bone (the
    chest, arms, hands, neck, head -- the aim) keeps its model-space
    orientation exactly as without the twist.

    BREAKS IF: the spine counter-twist is dropped or its shares do not sum
    to the twist, the twist turns the wrong subtree, or the counter turns
    about the wrong axis. And the skeleton stays CONNECTED (each bone at the
    point its twisted parent carries it to) -- which breaks if a counter
    bone turns about any vertical but its own joint's.
    """
    assert abs(math.radians(deg)) < AT.theta
    rule = _twist_rule(layer)
    ps = _walk(layer)
    for _ in range(240):
        layer.steer(ps, math.radians(deg), True, DT)
    assert abs(ps.twist - math.radians(deg)) < math.radians(1.0)
    plain = layer.compose(LOOK, [ps], modifiers=()).global_trs[0]
    out = layer.compose(LOOK, [ps]).global_trs[0]
    aim = _subtree(layer, LOOK, rule.counter[-1])
    assert _qangle(out[aim, 3:7], plain[aim, 3:7]).max() < 1e-3
    lower = _bones(layer, LOOK, [rule.lower])
    yaw = _yaw_between(_mats(plain[lower]), _mats(out[lower]))
    assert abs(float(yaw[0]) - ps.twist) < 1e-3
    assert _detached(layer, LOOK, plain, out) < 1e-4
    # and the legs really moved (else this is vacuous)
    legs = _subtree(layer, LOOK, rule.lower) & ~_subtree(layer, LOOK, rule.counter[0])
    assert np.abs(out[legs, :3] - plain[legs, :3]).max() > 0.02


def test_the_lower_body_never_yaws_beyond_theta(layer):
    """PROPERTY (correction 7): whatever the travel does -- sweeps all the
    way round, reversals, noise, stops -- the twist stays within +-theta
    and the lower bone's measured yaw against the un-twisted pose never
    exceeds theta.

    BREAKS IF: the backpedal reflection or the clamp is dropped (a travel
    behind the unit would turn the legs past theta), or the slew overshoots.
    """
    rule = _twist_rule(layer)
    lower = _bones(layer, LOOK, [rule.lower])
    rng = np.random.default_rng(163)
    ps = _walk(layer)
    worst = 0.0
    angles = np.concatenate([np.linspace(-np.pi, np.pi, 200),
                             rng.uniform(-np.pi, np.pi, 200),
                             np.full(60, np.pi), np.full(60, -0.99 * np.pi)])
    for k, a in enumerate(angles):
        travel = None if k % 4 == 3 else float(a + rng.normal(0, 0.2))
        layer.steer(ps, travel, k % 50 != 49, DT)
        layer.drive(ps, "walk", DT)
        worst = max(worst, abs(ps.twist))
        if k % 20 == 0:
            plain = layer.compose(LOOK, [ps], modifiers=()).global_trs[0]
            out = layer.compose(LOOK, [ps]).global_trs[0]
            y = _yaw_between(_mats(plain[lower]), _mats(out[lower]))
            assert abs(float(y[0])) <= AT.theta + 1e-4
    assert worst <= AT.theta + 1e-9
    assert worst > 0.5 * AT.theta, "the sweep never turned the legs (vacuous)"


def test_beyond_theta_the_legs_backpedal_and_nothing_pops(layer):
    """PROPERTY (correction 7): a travel past ``theta + hysteresis`` plays
    the leg layer's clip BACKWARDS (its clock runs down) with the legs
    facing away from the travel (the twist reflected, ``alpha - pi``);
    sweeping the travel slowly through the boundary and back, the twist
    never changes faster than ``twist_rate`` and the legs' model-space
    positions move only a few centimetres per frame (no 180-degree pop),
    while the leg clip's clock stays continuous across the switch.

    BREAKS IF: the backpedal is a hard switch (the twist snaps to its new
    goal), the rate is not negated past theta, or the switch restarts the
    leg clip.
    """
    rule = _twist_rule(layer)
    legs = _subtree(layer, LOOK, rule.lower) & ~_subtree(layer, LOOK, rule.counter[0])
    legs_layer = layer.rig(LOOK).twist_layer
    behind = AT.theta + AT.hysteresis + math.radians(20.0)
    ps = _walk(layer)
    for _ in range(240):
        layer.steer(ps, behind, True, DT)
        layer.drive(ps, "walk", DT)
    assert ps.back and ps.layer_rate[legs_layer] < 0
    want = (behind - math.pi + math.pi) % (2 * math.pi) - math.pi
    assert abs(ps.twist - max(-AT.theta, min(AT.theta, want))) < math.radians(1.0)
    t0 = ps.time[legs_layer]
    layer.drive(ps, "walk", DT)
    dur = layer.rig(LOOK).bank.duration[ps.clip[legs_layer]]
    assert (t0 - ps.time[legs_layer]) % dur < 0.5 * dur          # ran backwards
    # slow sweep 0 -> 180 deg -> 0 over 8 s
    ps = _walk(layer)
    sweep = np.concatenate([np.linspace(0, np.pi * 0.98, 240), np.linspace(np.pi * 0.98, 0, 240)])
    prev_g = layer.compose(LOOK, [ps]).global_trs[0]
    prev_twist, prev_time = ps.twist, ps.time[legs_layer]
    flips = 0
    worst_move = 0.0
    for a in sweep:
        was_back = ps.back
        layer.steer(ps, float(a), True, DT)
        layer.drive(ps, "walk", DT)
        flips += was_back != ps.back
        assert abs(ps.twist - prev_twist) <= AT.twist_rate * DT + 1e-9
        step = SETTINGS.rate * layer.rig(LOOK).bank.rate[ps.clip[legs_layer]] * DT
        dur = layer.rig(LOOK).bank.duration[ps.clip[legs_layer]]
        dt_clip = abs((ps.time[legs_layer] - prev_time + dur / 2) % dur - dur / 2)
        assert dt_clip <= step + 1e-9, "the leg clock jumped"
        g = layer.compose(LOOK, [ps]).global_trs[0]
        worst_move = max(worst_move, float(np.abs(g[legs, :3] - prev_g[legs, :3]).max()))
        prev_g, prev_twist, prev_time = g, ps.twist, ps.time[legs_layer]
    assert flips >= 2, "the sweep never crossed the backpedal boundary"
    # the bound: the plain walk's own largest per-frame move, plus a twist
    # step (twist_rate * DT) swung at the legs' largest reach from the lower
    # bone's vertical axis. A pop (the legs flipped 180 deg in one frame)
    # moves them by up to twice that reach.
    walk = _walk(layer)
    prev = layer.compose(LOOK, [walk]).global_trs[0]
    walk_max, reach = 0.0, 0.0
    lower = _bones(layer, LOOK, [rule.lower])[0]
    for _ in range(120):
        layer.drive(walk, "walk", DT)
        g = layer.compose(LOOK, [walk]).global_trs[0]
        walk_max = max(walk_max, float(np.abs(g[legs, :3] - prev[legs, :3]).max()))
        reach = max(reach, float(np.hypot(*(g[legs][:, [0, 2]] - g[lower, [0, 2]]).T).max()))
        prev = g
    bound = walk_max + AT.twist_rate * DT * reach
    assert worst_move <= bound + 1e-3, (worst_move, bound)
    assert bound < reach, "the bound would not catch a 180-degree pop"


def test_one_frame_of_travel_jitter_barely_moves_the_legs(layer):
    """PROPERTY (#163: the travel is a noisy position delta, low-passed): a
    single frame whose travel reading is off by 80 degrees moves the legs'
    twist by less than a quarter of that, even with the twist allowed to
    turn as fast as it likes.

    BREAKS IF: the travel angle is used raw (unfiltered), or the filter's
    time constant is ignored.
    """
    fast = dataclasses.replace(AT, twist_rate=1e6)
    legs_layer = layer.rig(LOOK).twist_layer
    ps = _walk(layer)
    base = math.radians(30.0)
    for _ in range(300):
        steer(ps, fast, legs_layer, base, True, DT)
    before = ps.twist
    steer(ps, fast, legs_layer, base + math.radians(80.0), True, DT)
    excursion = abs(ps.twist - before)
    for _ in range(3):
        steer(ps, fast, legs_layer, base, True, DT)
        excursion = max(excursion, abs(ps.twist - before))
    assert excursion < 0.25 * math.radians(80.0), math.degrees(excursion)


# ---------------------------------------------------------------------------
# fire: aim hold + retriggered shot
# ---------------------------------------------------------------------------

def _layer_index(name):
    return SETTINGS.layers.index(name)


def test_each_launch_retriggers_the_shot_and_keeps_the_aim(layer):
    """PROPERTY (#163): a launch starts the shot one-shot on the shot layer
    and the aim overlay on its other layers, the legs keep the locomotion
    clip; a second launch within the hold restarts the shot from its start
    (crossfading from where it was) and renews the hold, so the aim never
    drops between two shots.

    BREAKS IF: the shot is not restarted by a later launch, the hold timer
    is not renewed, or the overlay leaks onto a layer it does not name.
    """
    rig = layer.rig(LOOK)
    sl = _layer_index(FIRE.shot_layer)
    ov = rig.overlay_clips[FIRE.overlay]
    walk_ids = rig.state_clips["walk"]
    ps = _walk(layer)
    layer.fire(ps)
    layer.drive(ps, "walk", DT)
    assert ps.clip[sl] == rig.shot_clip
    for l in range(len(SETTINGS.layers)):
        if l != sl:
            assert ps.clip[l] == (ov[l] if ov[l] >= 0 else walk_ids[l])
    t_mid = 0.4 * FIRE.aim_hold_s
    for _ in range(int(t_mid / DT)):
        layer.drive(ps, "walk", DT)
    shot_time = ps.time[sl]
    layer.fire(ps)
    layer.drive(ps, "walk", DT)
    assert ps.clip[sl] == rig.shot_clip and ps.time[sl] < shot_time
    if SETTINGS.crossfade_s > 0:
        assert ps.prev_clip[sl] >= 0 and ps.fade[sl] < 1.0
    assert ps.aim_left > FIRE.aim_hold_s - 2 * DT
    # the aim overlay holds for the whole hold after the LAST shot
    for _ in range(int((FIRE.aim_hold_s - 3 * DT) / DT)):
        layer.drive(ps, "walk", DT)
        for l in range(len(SETTINGS.layers)):
            if ov[l] >= 0 and l != sl:
                assert ps.clip[l] == ov[l]
            if ov[l] < 0 and l != sl:
                assert ps.clip[l] == walk_ids[l]


def test_the_aim_hold_ends_with_a_crossfade_back_to_locomotion(layer):
    """PROPERTY (#163): ``aim_hold_s`` after the last launch the overlay
    layers return to the locomotion clips THROUGH a crossfade (some frame
    shows the ramp part-way), and once settled the pose is the plain
    locomotion pose of a unit that never fired (same clips on every layer).

    BREAKS IF: the hold never expires, the return snaps, or the shot /
    overlay is left on a layer.
    """
    rig = layer.rig(LOOK)
    ps = _walk(layer)
    layer.fire(ps)
    seen_partial = False
    for _ in range(int((FIRE.aim_hold_s + SETTINGS.crossfade_s + 0.5) / DT)):
        layer.drive(ps, "walk", DT)
        if ps.aim_left == 0.0 and ((ps.fade > 0.0) & (ps.fade < 1.0)).any():
            seen_partial = True
    assert ps.aim_left == 0.0
    assert (ps.clip == rig.state_clips["walk"]).all() and (ps.prev_clip == NO_CLIP).all()
    if SETTINGS.crossfade_s > 0:
        assert seen_partial, "the return to locomotion snapped"


# ---------------------------------------------------------------------------
# hit flinch (additive)
# ---------------------------------------------------------------------------

def _twin_hit(pl, damage, cause=None, look=LOOK):
    """Two identical walking units; one is hit. Yields (plain, hit) composed
    rows per frame until well after the flinch ends."""
    a = _walk(pl, look)
    b = _clone(pl, a)
    pl.hit(b, damage, cause or HIT.causes[0])
    rig = pl.rig(look)
    c = rig.hit_clip
    wall = rig.bank.duration[c] / (SETTINGS.rate * rig.bank.rate[c])
    for _ in range(int((wall + 0.3) / DT)):
        pl.drive(a, "walk", DT)
        pl.drive(b, "walk", DT)
        out = pl.compose(look, [a, b])
        yield out, a, b


def test_a_flinch_moves_only_its_masks_and_ends_exactly(layer):
    """PROPERTY (#163, §4.5 correction 3): while a hit flinch plays, every
    bone OUTSIDE the hit masks (the legs) keeps the exact local pose of an
    identical unit that was not hit, and its legs the same model-space pose;
    the flinch visibly moves the upper body; once it has ended the whole
    pose equals the un-hit twin's exactly.

    BREAKS IF: the additive ignores its masks, its reference is the bind
    pose (it would add the clip's stance), or it lingers after the clip.
    """
    rig = layer.rig(LOOK)
    inside = rig.hit_bones > 0
    legs = rig.masks.weights[rig.masks.index(SETTINGS.layers[0])] > 0
    assert not (inside & legs).any()
    moved = 0.0
    last = None
    for out, a, b in _twin_hit(layer, HIT.damage_full):
        assert np.array_equal(out.local_r[0, ~inside], out.local_r[1, ~inside])
        assert np.array_equal(out.local_t[0, ~inside], out.local_t[1, ~inside])
        assert np.abs(out.global_trs[0, legs] - out.global_trs[1, legs]).max() < 1e-6
        moved = max(moved, float(_qangle(out.local_r[0, inside], out.local_r[1, inside]).max()))
        last = (out, b)
    out, b = last
    assert b.flinch_t < 0 and b.add_clip == NO_CLIP
    assert np.array_equal(out.global_trs[0], out.global_trs[1])
    assert moved > math.radians(3.0), "the flinch did not move the upper body"


def test_a_flinch_ramps_in_and_out_without_jumps(layer):
    """PROPERTY (#163): the flinch's weight starts at 0 on the hit, rises
    over ``ramp_in_s`` and falls back to exactly 0 at the clip's end -- it
    is never set in one jump from 0 to its peak or from its peak to 0.

    BREAKS IF: the ramps are dropped (the additive is switched on and off).
    """
    weights = [b.add_weight if b.flinch_t >= 0 else 0.0
               for _, _, b in _twin_hit(layer, HIT.damage_full)]
    peak = max(weights)
    assert peak > 0
    steps = np.abs(np.diff([0.0] + weights))
    if HIT.ramp_in_s > 0 and HIT.ramp_out_s > 0:
        biggest = max(DT / HIT.ramp_in_s, DT / HIT.ramp_out_s) * peak
        assert steps.max() <= biggest + 1e-9
    assert weights[-1] == 0.0


def test_flinch_weight_rises_with_damage_and_is_clamped(layer):
    """PROPERTY (#163): the flinch's peak weight never decreases as the
    damage rises and always lies within ``[weight_min, weight_max]`` -- a
    zero, negative or enormous damage included; a hit of a cause outside
    ``causes`` (a burn's per-tick damage) starts no flinch at all.

    BREAKS IF: the clamp is dropped (a huge hit would exaggerate the clip
    past weight_max), the damage scaling is inverted, or the cause filter
    is ignored.
    """
    peaks = []
    for dmg in (-50, 0, 1, HIT.damage_full / 4, HIT.damage_full, 10 * HIT.damage_full, 1e9):
        ps = _walk(layer)
        layer.hit(ps, dmg, HIT.causes[0])
        peaks.append(ps.flinch_peak)
    assert all(HIT.weight_min - 1e-12 <= p <= HIT.weight_max + 1e-12 for p in peaks)
    assert all(b >= a for a, b in zip(peaks, peaks[1:]))
    other = next(c for c in ("heat", "burning", "poison_gas", "regen") if c not in HIT.causes)
    ps = _walk(layer)
    layer.hit(ps, 1e9, other)
    layer.drive(ps, "walk", DT)
    assert ps.flinch_t < 0 and ps.add_clip == NO_CLIP


# ---------------------------------------------------------------------------
# the renderer wiring: routing, re-delivery, render-only, the off path
# ---------------------------------------------------------------------------

class _Unit:
    """A read-only unit stub: raises on any write."""

    def __init__(self, uid, x, y, facing=0.0):
        object.__setattr__(self, "_d", dict(id=uid, x=float(x), y=float(y), facing=facing,
                                            move_path=[], alive=True, footprint=3,
                                            is_zombie=False))

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AssertionError(f"the renderer wrote {name!r} on a unit")


class _Skin:
    def __init__(self):
        self.rows = {}

    def prepare(self, look, model, template_anim, bone_names):
        pass

    def apply(self, look, model, pose_row):
        self.rows.setdefault(look, []).append(np.array(pose_row))

    def release(self):
        pass


class _Rl:
    def update_model_animation(self, *a):
        raise AssertionError("raylib's clip player called with the pose layer on")

    def __getattr__(self, name):
        return lambda *a, **k: None


def _renderer(monkeypatch, enabled=True):
    from renderer import unit_model_renderer as umr
    from renderer.unit_looks import LookAssigner
    monkeypatch.setattr(umr, "rl", _Rl())
    monkeypatch.setattr(CFG.render.pose, "enabled", enabled)
    r = umr.UnitModelRenderer()
    for name in ALL_SHIPPED:
        model = SimpleNamespace(boneCount=53, name=name)
        anims = [SimpleNamespace(tag=(name, k), frameCount=40) for k in range(2)]
        r._looks[name] = umr._Look(name=name, path=look_path(name), model=model,
                                   anims=anims, n_anims=2, clip_index={}, native_height=1.8)
        r._resolve[name] = r._looks[name]
    r._assigner = LookAssigner(looks_table(CFG))
    r._loaded = True
    sk = _Skin()
    r._setup_pose(skinner=sk)
    return r, sk


def _frames(r, sk, script, n_frames, x0=4.0):
    """Draw ``n_frames`` of two marines (ids 1 and 2) walking north while
    facing east (an aim sideways); ``script(k)`` returns the events handed
    over before frame k. Returns each frame's pose rows per unit id."""
    rows = {1: [], 2: []}
    for k in range(n_frames):
        r.queue_events(script(k))
        units = [_Unit(1, x0, 40 - 0.25 * k), _Unit(2, x0 + 10, 40 - 0.25 * k)]
        before = len(sk.rows.get(ALL_SHIPPED[0], []))
        r.draw_units(units, 8.0, 1.0 + k * DT, camera3d=None, light_view=None)
        new = sk.rows[ALL_SHIPPED[0]][before:]
        rows[1].append(new[0])
        rows[2].append(new[1])
    return rows


def test_events_for_other_units_never_affect_a_unit(monkeypatch):
    """PROPERTY (#163 routing): shots and hits addressed to unit 2 change
    unit 2's pose and leave unit 1's pose -- every frame -- exactly what it
    is with no events at all; drawing never writes a unit.

    BREAKS IF: events are routed by anything but ``unit_id`` (e.g. applied
    to every unit, or to the drawn order), or a reaction leaks across units.
    """
    def none(k):
        return []

    def noisy(k):
        if k % 4 == 1:
            return [ShotFiredEvent(unit_id=2, from_tile=(0, 0), to_tile=(5, 5), launch=True),
                    UnitHitEvent(unit_id=2, damage=60, source="melee", attacker_id=9)]
        return []

    quiet = _frames(*_renderer(monkeypatch), none, 40)
    loud = _frames(*_renderer(monkeypatch), noisy, 40)
    for f in range(40):
        assert np.array_equal(quiet[1][f], loud[1][f])
    assert max(np.abs(q - l).max() for q, l in zip(quiet[2], loud[2])) > 1e-3


def test_a_redelivered_event_list_reacts_once(monkeypatch):
    """PROPERTY (#163): the game hands ``sim.tick_events`` over every frame
    and the list survives until the next sim step, so the same event objects
    arrive on several frames; each is acted on ONCE (one shot, one queued
    entry), while a new event object (the next tick's) is acted on again.

    BREAKS IF: the re-delivery guard is dropped (one bullet would retrigger
    the recoil on every frame between two ticks).
    """
    r, sk = _renderer(monkeypatch)
    ev = ShotFiredEvent(unit_id=1, from_tile=(0, 0), to_tile=(5, 5), launch=True)
    tick = [ev]
    r.queue_events(tick)
    r.queue_events(tick)
    r.queue_events(tick)
    assert len(r._pose.events(1)) == 1
    r.queue_events([ShotFiredEvent(unit_id=1, from_tile=(0, 0), to_tile=(5, 5), launch=True)])
    assert len(r._pose.events(1)) == 2


def test_a_launch_fires_and_a_mid_flight_segment_does_not(monkeypatch):
    """PROPERTY (#163, #161's signal check): a ``ShotFiredEvent`` with
    ``launch=True`` for a unit starts its fire reaction (aim hold + shot);
    one with ``launch=False`` (a slow round's later segment) does not.

    BREAKS IF: ``launch`` is ignored (a plasma bolt would retrigger the
    recoil on every tick of its flight) or the shot is never triggered.
    """
    r, sk = _renderer(monkeypatch)
    _frames(r, sk, lambda k: [ShotFiredEvent(unit_id=1, from_tile=(0, 0), to_tile=(1, 1),
                                             launch=False)], 3)
    assert r._anim[1].pose.aim_left == 0.0
    _frames(r, sk, lambda k: [ShotFiredEvent(unit_id=1, from_tile=(0, 0), to_tile=(1, 1),
                                             launch=True)] if k == 0 else [], 2)
    ps = r._anim[1].pose
    assert ps.aim_left > 0.0
    # the shot one-shot -- or, on a look that declares none (#184: the SOMA
    # marine's Rifle_Shoot is left-handed), the aim overlay's clip
    rig = r._pose.rig(ALL_SHIPPED[0])
    sl = _layer_index(FIRE.shot_layer)
    want = rig.shot_clip if rig.shot_clip != NO_CLIP else rig.overlay_clips[FIRE.overlay][sl]
    assert ps.clip[sl] == want


def test_the_legs_walk_along_the_travel_in_the_world(monkeypatch):
    """PROPERTY (#163): a marine facing east and moving north-east (within
    theta of its facing) is drawn with its LOWER body pointing along its
    travel in the world: the twist plus the renderer's facing yaw give the
    travel direction, under the renderer's own facing -> yaw convention.

    BREAKS IF: the travel angle's sign or the model-space yaw sign is
    flipped (the legs would point away from the motion, mirrored).
    """
    from renderer import unit_model_renderer as umr
    r, sk = _renderer(monkeypatch)
    facing = 0.0                                      # east
    for k in range(120):                              # moving north-east: x up, y down
        u = _Unit(1, 4 + 0.1 * k, 40 - 0.1 * k, facing=facing)
        r.draw_units([u], 8.0, 1.0 + k * DT, camera3d=None, light_view=None)
    twist = r._anim[1].pose.twist
    yaw_deg = umr._YAW_SIGN * math.degrees(facing) + umr._YAW_OFFSET_DEG + math.degrees(twist)
    fwd = (math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg)))   # (X, Z)
    travel = np.array([0.1, -0.1]) / math.hypot(0.1, 0.1)                       # world (x, y)
    assert float(np.dot(fwd, travel)) > math.cos(math.radians(3.0)), (twist, fwd)


def test_the_off_path_ignores_fire_and_hit_events(monkeypatch):
    """PROPERTY (§4.5 correction 9, extended for #163): with the setting off,
    shots and hits change nothing -- the pose layer is never built, and
    every unit is skinned with one of its look's own clips at an integer
    frame, exactly as before the pose layer.

    BREAKS IF: the reactions are wired outside the pose layer (e.g. the
    clip player starts reading events) or anything of the layer is built.
    """
    from renderer import pose_layer as pl_mod
    from renderer import unit_model_renderer as umr

    def _never(*a, **k):
        raise AssertionError("pose layer constructed with the setting off")
    monkeypatch.setattr(pl_mod.PoseLayer, "__init__", _never)
    calls = []

    class _RlRec(_Rl):
        def update_model_animation(self, model, anim, frame):
            calls.append((anim, frame))
    r, _ = _renderer(monkeypatch, enabled=False)
    monkeypatch.setattr(umr, "rl", _RlRec())
    assert r._pose is None
    for k in range(4):
        r.queue_events([ShotFiredEvent(unit_id=1, from_tile=(0, 0), to_tile=(1, 1)),
                        UnitHitEvent(unit_id=1, damage=60, source="melee")])
        r.draw_units([_Unit(1, 4, 40 - k, facing=0.0)], 8.0, 1.0 + k * DT,
                     camera3d=None, light_view=None)
    own = {id(a) for lk in r._looks.values() for a in lk.anims}
    assert len(calls) == 4
    assert all(id(a) in own and type(f) is int for a, f in calls)


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------

def _broken(edit):
    d = critter_settings()
    edit(d["render"]["pose"])
    return d


def _sk(p):
    return p["skeletons"]["critter"]


@pytest.mark.parametrize("edit, needle", [
    (lambda p: p.pop("fire"), "fire"),
    (lambda p: p.pop("aim_travel"), "aim_travel"),
    (lambda p: p["fire"].pop("aim_hold_s"), "aim_hold_s"),
    (lambda p: p["fire"].update(aim_hold_s="long"), "aim_hold_s"),
    (lambda p: p["fire"].update(overlay="crouch"), "overlay"),
    (lambda p: p["fire"].update(shot_layer="tail"), "shot_layer"),
    (lambda p: p["fire"].update(recoil=2), "recoil"),
    (lambda p: p["overlays"].update(aim={"tail": "look"}), "aim"),
    (lambda p: p["hit"].update(weight_max=1.5), "weight_max"),
    (lambda p: p["hit"].update(weight_min=0.9, weight_max=0.5), "weight_max"),
    (lambda p: p["hit"].update(damage_full=0), "damage_full"),
    (lambda p: p["hit"].update(masks=["wings"]), "wings"),
    (lambda p: p["hit"].update(causes="bullet"), "causes"),
    (lambda p: p["hit"].pop("ramp_out_s"), "ramp_out_s"),
    (lambda p: p["aim_travel"].update(theta_deg=0), "theta_deg"),
    (lambda p: p["aim_travel"].update(theta_deg=200), "theta_deg"),
    (lambda p: p["aim_travel"].update(hysteresis_deg=95), "hysteresis_deg"),
    (lambda p: p["aim_travel"].update(twist_rate_deg_s=0), "twist_rate_deg_s"),
    (lambda p: p["aim_travel"].update(smooth="x"), "smooth"),
    (lambda p: _sk(p).pop("twist"), "twist"),
    (lambda p: _sk(p)["twist"].update(shares=[0.5]), "sum to 1"),
    (lambda p: _sk(p)["twist"].update(shares=[1.0, 0.0]), "shares"),
    (lambda p: _sk(p)["twist"].update(lower=3), "lower"),
])
def test_the_new_settings_refuse_missing_unknown_and_mistyped_keys(edit, needle):
    """PROPERTY (#163): the fire, hit, aim-travel, overlay and twist tables
    are refused, naming the key, when a key is missing, unknown, mistyped
    or out of range (a weight above 1, theta outside (0, 180], hysteresis
    not below theta, shares not summing to 1, a mask / layer / overlay that
    does not exist).

    BREAKS IF: any new key falls back to a silent default.
    """
    with pytest.raises(PoseConfigError, match=re.escape(needle)):
        pose_settings(_broken(edit))


@pytest.mark.parametrize("twist, needle", [
    ({"lower": "pelvis_q", "counter": ["back_x"], "shares": [1.0]}, "pelvis_q"),
    ({"lower": "back_x", "counter": ["hindleg_l"], "shares": [1.0]}, "hindleg_l"),
    ({"lower": "pelvis_x", "counter": ["skull_x", "back_x"], "shares": [0.5, 0.5]}, "back_x"),
])
def test_twist_bones_are_checked_on_each_skeleton(tmp_path, twist, needle):
    """PROPERTY (#163, skeleton-agnostic): the twist's bones are resolved on
    each look's skeleton at load; a bone it lacks, a counter bone outside
    the lower bone's subtree, or counter bones that are not one chain
    (root-most first) are refused, naming the bone.

    BREAKS IF: twist bones are assumed (hard-coded) or not checked, so a
    different skeleton would twist the wrong bones silently.
    """
    path = write_glb(tmp_path / "critter.glb", CRITTER_BONES, critter_clips())
    d = critter_settings()
    d["render"]["pose"]["skeletons"]["critter"]["twist"] = twist
    with pytest.raises(PoseConfigError, match=re.escape(needle)):
        PoseLayer(pose_settings(d)).add_look("critter", path)


def test_a_hit_cause_outside_the_sims_vocabulary_is_refused(monkeypatch):
    """PROPERTY (#163): ``[render.pose.hit] causes`` names only the sim's
    damage causes (``simulation.events.Cause``); a typo is refused when the
    renderer builds the layer.

    BREAKS IF: an unknown cause is accepted (that flinch would never fire).
    """
    hit = dataclasses.replace(SETTINGS.hit, causes=("bullet", "bulet"))
    bad = dataclasses.replace(SETTINGS, hit=hit)
    from renderer import pose_layer as pl_mod
    monkeypatch.setattr(pl_mod, "pose_settings", lambda cfg: bad)
    with pytest.raises(PoseConfigError, match="bulet"):
        _renderer(monkeypatch)


def test_the_twist_and_reactions_run_on_a_foreign_skeleton(tmp_path):
    """PROPERTY (#163, skeleton-agnostic): on a five-bone creature with its
    own bone names, masks and twist table, the twist keeps the last counter
    bone's subtree on the facing while the lower bone turns, and the fire
    and hit reactions play its own clips -- through the same code.

    BREAKS IF: any of it assumes the shipped rig.
    """
    path = write_glb(tmp_path / "critter.glb", CRITTER_BONES, critter_clips())
    s = pose_settings(critter_settings())
    pl = PoseLayer(s)
    rig = pl.add_look("critter", path)
    ps = pl.new_state("critter", "walk")
    pl.fire(ps)
    pl.hit(ps, 5, "bullet")
    for _ in range(45):                       # 0.75 s into the 1 s trot: far off the origin
        pl.steer(ps, math.radians(60.0), True, DT)
        pl.drive(ps, "walk", DT)
    assert ps.twist > math.radians(45.0)
    plain = pl.compose("critter", [ps], modifiers=()).global_trs[0]
    out = pl.compose("critter", [ps]).global_trs[0]
    rule = s.twists[rig.skeleton_table]
    aim = rig.bank.skeleton.subtree(rig.bank.skeleton.index(rule.counter[-1]))
    assert _qangle(out[aim, 3:7], plain[aim, 3:7]).max() < 1e-3
    assert _qangle(out[~aim, 3:7], plain[~aim, 3:7]).max() > 0.5
    # the trot carries the creature far off its origin: a counter turned
    # about the wrong vertical would tear the skeleton apart here
    assert np.abs(plain[rig.bank.skeleton.roots][:, [0, 2]]).max() > 0.2
    assert _detached(pl, "critter", plain, out) < 1e-4
