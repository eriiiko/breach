"""#164 -- posture (experiment E4): a zombie's carriage as a profile of the
brief's §3 parameters scaled by an amount a in [0, 1], tempo jitter and
twitch, knees bent without IK, and one melee strike played by the ATTACKER.
Properties only.

Spec: docs/zombie_motion_brief_2026-11-02.md §2-§3, design
docs/animation_milestones_design_2026-11-02.md §4.5 and §5.2-5.3, issue #164.
Every test names the property it protects and the change that must break it.
Bone names come from the config (``[render.pose.skeletons.*.posture]``),
never from this file.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_pose_posture.py -q
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
from renderer.pose_clips import quat_conj, quat_mul, trs_matrices  # noqa: E402
from renderer.pose_layer import (NO_CLIP, PoseConfigError, PoseLayer,  # noqa: E402
                                 pose_settings)
from renderer.pose_posture import PARAMS, smooth_noise, unit_noise  # noqa: E402
from renderer.unit_looks import all_look_names, look_path, looks_table  # noqa: E402
from simulation.events import ShotFiredEvent, UnitHitEvent  # noqa: E402

from _pose_glb import CRITTER_BONES, critter_clips, critter_settings, write_glb  # noqa: E402

SHIPPED = all_look_names(looks_table(CFG))
ZOMBIE_LOOKS = list(looks_table(CFG)["zombie"])
SETTINGS = pose_settings(CFG)
POSTURE = SETTINGS.posture
PROFILE = POSTURE.vector
LOOK = ZOMBIE_LOOKS[0]
DT = 1.0 / 60.0
P = {p: i for i, p in enumerate(PARAMS)}


@pytest.fixture(scope="module")
def layer() -> PoseLayer:
    pl = PoseLayer(SETTINGS)
    for name in SHIPPED:
        pl.add_look(name, look_path(name))
    return pl


def _rule(pl, look=LOOK):
    return pl.settings.postures[pl.rig(look).skeleton_table]


def _bones(pl, look, names):
    sk = pl.rig(look).bank.skeleton
    return [sk.index(n) for n in names]


def _mats(g):
    return trs_matrices(g[..., :3].astype(np.float64), g[..., 3:7].astype(np.float64),
                        g[..., 7:].astype(np.float64))


def _state(pl, state="walk", uid=7, a=1.0, t=0.3, look=LOOK):
    ps = pl.new_state(look, state)
    pl.set_posture(ps, uid, a)
    ps.time[:] = t
    return ps


def _clone(pl, ps):
    out = pl.new_state(ps.look, ps.state)
    for slot in type(ps).__slots__:
        v = getattr(ps, slot)
        setattr(out, slot, v.copy() if isinstance(v, np.ndarray) else v)
    return out


def _lean_deg(pl, g, look=LOOK):
    """Forward lean of the chest (degrees): the last spine_pitch bone's
    up axis (+Y) tipped toward the skeleton's forward."""
    rule = _rule(pl, look)
    last = rule.rotations["spine_pitch"][-1].bones[-1]
    j = _bones(pl, look, [last])[0]
    y = _mats(g[j])[:3, 1]
    up, fwd = np.asarray(rule.up), np.asarray(rule.forward)
    return math.degrees(math.atan2(float(y @ fwd), float(y @ up)))


# ---------------------------------------------------------------------------
# amount: identity at 0, continuous and monotone, lands on the profile
# ---------------------------------------------------------------------------

def test_amount_zero_composes_bit_identically_to_no_posture(layer):
    """PROPERTY (brief line 20, #164): a unit at amount 0 -- even one that
    was posed earlier and carries noise draws, mid-flinch, mid-shot and
    twisted -- composes and advances EXACTLY as a unit the posture never
    touched: same global pose bit for bit, same clip clocks.

    BREAKS IF: any posture term (offset, tempo, twitch, hip drop) is applied
    without being scaled by the amount, or a = 0 rows go through the posture
    arithmetic (float residue).
    """
    plain = layer.new_state(LOOK, "walk")
    layer.drive(plain, "walk", 0.4)
    posed = _clone(layer, plain)
    layer.set_posture(posed, 11, 1.0)
    posed.time[:] = plain.time                       # undo desync_start
    layer.set_posture(posed, 11, 0.0)
    for ps in (plain, posed):
        layer.fire(ps)
        layer.hit(ps, 30, SETTINGS.hit.causes[0])
    for k in range(90):
        for ps in (plain, posed):
            layer.steer(ps, 0.6, True, DT)
            layer.drive(ps, "walk", DT)
        assert np.array_equal(plain.time, posed.time)
        out = layer.compose(LOOK, [plain, posed]).global_trs
        assert np.array_equal(out[0], out[1])


def test_posture_scales_continuously_and_monotonically_with_the_amount(layer):
    """PROPERTY (brief line 20): the chest's forward lean rises
    monotonically as the amount goes 0 -> 1, by small steps (continuous),
    and the hips sink monotonically.

    BREAKS IF: the amount is thresholded (a switch, not a blend), the
    offsets are not proportional to it, or a sign flips part-way.
    """
    rule = _rule(layer)
    hips = _bones(layer, LOOK, [rule.knee.hips])[0]
    leans, hip_y = [], []
    for a in np.linspace(0.0, 1.0, 21):
        ps = _state(layer, "idle", a=float(a), t=0.5)
        g = layer.compose(LOOK, [ps]).global_trs[0]
        leans.append(_lean_deg(layer, g))
        hip_y.append(float(g[hips, 1]))
    steps = np.diff(leans)
    assert (steps > 0).all(), leans
    assert steps.max() < 3.0 * steps.mean()          # no jump anywhere
    assert (np.diff(hip_y) < 0).all()


@pytest.mark.parametrize("look", ZOMBIE_LOOKS)
def test_full_amount_leans_by_the_profiles_spine_pitch(layer, look):
    """PROPERTY (brief §3 stalk column): the profile's spine_pitch lies in
    the brief's 25-35 deg and, at amount 1, the chest leans forward by that
    pitch times the unit's own variation (+- 1.5 deg: the spine bones turn
    about their parents' rest axes, not the clip's), on the idle and the
    walk alike.

    BREAKS IF: the spine shares stop summing to the profile's pitch, the
    pitch turns the wrong way (backwards), or the axis is not the lateral.
    """
    assert 25.0 <= PROFILE[P["spine_pitch"]] <= 35.0
    for state in ("idle", "walk"):
        ps = _state(layer, state, a=1.0, look=look)
        base = _clone(layer, ps)
        base.posture = 0.0
        out = layer.compose(look, [base, ps]).global_trs
        want = PROFILE[P["spine_pitch"]] * ps.noise.scale[P["spine_pitch"]]
        got = _lean_deg(layer, out[1], look) - _lean_deg(layer, out[0], look)
        assert abs(got - want) < 1.5, (state, got, want)


def test_posture_rides_on_any_clip_as_the_same_offset(layer):
    """PROPERTY (design §5.2: posture is an additive over the base): on the
    walk and on the idle, at the same noise time, every bone the spine,
    shoulder, arm and head rows turn gets the SAME local offset (the posed
    local rotation times the inverse of the clip's), so the posture rides
    on whatever plays.

    BREAKS IF: the offset is a pose (an override replacing the clip's
    rotation) instead of an offset, or it depends on the clip.
    """
    rule = _rule(layer)
    names = {b for p in ("spine_pitch", "shoulder_protraction", "head_drop")
             for e in rule.rotations[p] for b in e.bones}
    idx = _bones(layer, LOOK, sorted(names))
    offsets = []
    for state, t in (("walk", 0.37), ("idle", 1.1)):
        ps = _state(layer, state, uid=3, a=1.0, t=t)
        ps.tau = 2.5
        base = _clone(layer, ps)
        base.posture = 0.0
        out = layer.compose(LOOK, [base, ps])
        offsets.append(quat_mul(out.local_r[1, idx], quat_conj(out.local_r[0, idx])))
    a, b = offsets
    d = np.abs((a * b).sum(-1))
    assert (d > 1.0 - 1e-5).all(), d
    angle = 2 * np.arccos(np.clip(np.abs(a[:, 3]), 0, 1))
    assert angle.max() > math.radians(5.0)          # not vacuous


@pytest.mark.parametrize("look", ZOMBIE_LOOKS)
def test_the_arms_are_carried_forward_symmetrically(layer, look):
    """PROPERTY (brief lines 3, 7): at amount 1, averaged over the walk
    cycle, both hands sit well ahead of where the clip puts them (along the
    skeleton's forward, by more than a quarter of the shoulder-to-hand
    reach) and not lower, and each hand stays on its own side of the hips
    (the clips are not symmetric, so no exact mirror is asked).

    BREAKS IF: a model-space axis is applied in the bone's parent frame
    unconverted (the arms would swing sideways or one up, one down), or the
    L / R rows are not mirrored (one hand would cross the body).
    """
    rule = _rule(layer, look)
    arm_bones = [e.bones for e in rule.rotations["arm_forward"]][0]
    sk = layer.rig(look).bank.skeleton
    depth = np.zeros(sk.n_bones, dtype=int)
    for d, lvl in enumerate(sk.levels):
        depth[lvl] = d
    hands = []
    for b in arm_bones:                          # the first deepest bone under each arm root
        sub = np.flatnonzero(sk.subtree(sk.index(b)))
        hands.append(int(sub[np.argmax(depth[sub])]))
    fwd, up = np.asarray(rule.forward), np.asarray(rule.up)
    lat = np.cross(up, fwd)
    roots = [sk.index(b) for b in arm_bones]
    hips = sk.index(rule.knee.hips)
    dur = layer.rig(look).bank.duration[layer.rig(look).state_clips["walk"][0]]
    ahead, rise, reach, side, plain_side = [], [], [], [], []
    for t in np.linspace(0.0, dur, 8, endpoint=False):
        ps = _state(layer, "walk", a=1.0, t=float(t), look=look)
        ps.tau = 0.0
        base = _clone(layer, ps)
        base.posture = 0.0
        out = layer.compose(look, [base, ps]).global_trs
        d = out[1, hands, :3] - out[0, hands, :3]
        ahead.append(d @ fwd)
        rise.append(d @ up)
        reach.append(np.linalg.norm(out[0, hands, :3] - out[0, roots, :3], axis=-1))
        side.append(out[1, hands, :3] @ lat - out[1, hips, :3] @ lat)
        plain_side.append(out[0, hands, :3] @ lat - out[0, hips, :3] @ lat)
    ahead, reach = np.mean(ahead, 0), np.mean(reach, 0)
    assert (ahead > 0.25 * reach).all(), (ahead, reach)
    assert (np.mean(rise, 0) > -0.02).all()
    assert (np.sign(np.mean(side, 0)) == np.sign(np.mean(plain_side, 0))).all()


def test_posture_keeps_the_flinch_and_the_twist_properties(layer):
    """PROPERTY (#164 item 5 / #163): with posture on, a hit flinch still
    moves only its masks (the legs equal an un-hit, equally postured twin's,
    exactly) and the aim-vs-travel twist still keeps the torso and head on
    the facing (their orientations equal the un-twisted pose's).

    BREAKS IF: posture is applied after FK / the modifiers (the twist would
    turn the posture away from the facing), or the flinch and posture
    overwrite one another.
    """
    a = _state(layer, "walk", uid=21)
    b = _clone(layer, a)
    layer.hit(b, SETTINGS.hit.damage_full, SETTINGS.hit.causes[0])
    rig = layer.rig(LOOK)
    inside = rig.hit_bones > 0
    moved = 0.0
    for _ in range(12):
        layer.drive(a, "walk", DT)
        layer.drive(b, "walk", DT)
        out = layer.compose(LOOK, [a, b])
        assert np.array_equal(out.local_r[0, ~inside], out.local_r[1, ~inside])
        moved = max(moved, float(np.abs(out.local_r[0, inside] - out.local_r[1, inside]).max()))
    assert moved > 1e-3
    tw = SETTINGS.twists[rig.skeleton_table]
    ps = _state(layer, "walk", uid=22)
    for _ in range(240):
        layer.steer(ps, math.radians(50.0), True, DT)
    assert ps.twist > math.radians(40.0)
    plain = layer.compose(LOOK, [ps], modifiers=()).global_trs[0]
    out = layer.compose(LOOK, [ps]).global_trs[0]
    sk = rig.bank.skeleton
    aim = sk.subtree(sk.index(tw.counter[-1]))
    d = np.abs((out[aim, 3:7] * plain[aim, 3:7]).sum(-1))
    assert (d > 1 - 1e-6).all()


# ---------------------------------------------------------------------------
# knees without IK
# ---------------------------------------------------------------------------

FOOT_TOL = 0.03            # model units (metres): the documented residual


@pytest.mark.parametrize("look", SHIPPED)
def test_bent_knees_keep_the_feet_near_the_floor(layer, look):
    """PROPERTY (#164 item 3): with the stalk profile's knee bend, over the
    whole walk cycle and the idle, every foot and toe bone stays within
    ``FOOT_TOL`` of its height without posture -- while the hips really
    drop by close to the profile's fraction of the leg.

    BREAKS IF: the hips drop without the legs folding (the feet would sink
    by the whole drop), or the leg fold is applied with the wrong sign
    (the feet would float).
    """
    rule = _rule(layer, look)
    feet = _bones(layer, look, [leg[2] for leg in rule.knee.legs])
    sk = layer.rig(look).bank.skeleton
    toes = [int(np.flatnonzero(sk.parents == f)[0]) for f in feet
            if (sk.parents == f).any()]
    hips = _bones(layer, look, [rule.knee.hips])[0]
    res = layer.rig(look).posture
    drop = PROFILE[P["knee_bend"]] * res.leg_len
    bank = layer.rig(look).bank
    worst, hip_drop = 0.0, []
    for state in ("walk", "idle"):
        dur = bank.duration[layer.rig(look).state_clips[state][0]]
        for t in np.linspace(0.0, dur, 24, endpoint=False):
            ps = _state(layer, state, a=1.0, t=float(t), look=look)
            base = _clone(layer, ps)
            base.posture = 0.0
            out = layer.compose(look, [base, ps]).global_trs
            dy = out[1, feet + toes, 1] - out[0, feet + toes, 1]
            worst = max(worst, float(np.abs(dy).max()))
            hip_drop.append(float(out[0, hips, 1] - out[1, hips, 1]))
    assert worst < FOOT_TOL, worst
    assert min(hip_drop) > 0.8 * drop


# ---------------------------------------------------------------------------
# tempo jitter, twitch, horde variation
# ---------------------------------------------------------------------------

def _clock_run(pl, uid, seconds, a=1.0):
    ps = _state(pl, "walk", uid=uid, a=a, t=0.0)
    ps.time[:] = 0.0
    legs = 0
    total, tempos = 0.0, []
    prev = ps.time[legs]
    dur = pl.rig(LOOK).bank.duration[ps.clip[legs]]
    for _ in range(int(seconds / DT)):
        pl.drive(ps, "walk", DT)
        tempos.append(ps.tempo)
        total += (ps.time[legs] - prev) % dur
        prev = ps.time[legs]
    return total, np.array(tempos)


def test_tempo_jitter_keeps_the_mean_rate_and_is_seeded_per_unit(layer):
    """PROPERTY (brief line 9 / §3 tempo_jitter): a posed unit's walk clock
    over ten minutes advances within 1 % of an un-jittered one (the mean
    rate is kept), its rate really varies (by more than a third of the
    jitter) and never goes negative; the rate sequence is the same for the
    same unit id and differs between ids.

    BREAKS IF: the jitter is biased (not zero-mean), unseeded (global or
    wall-clock randomness), or keyed to anything but the unit id and time.
    """
    nominal = 600.0 * SETTINGS.rate * layer.rig(LOOK).bank.rate[
        layer.rig(LOOK).state_clips["walk"][0]]
    total, tempos = _clock_run(layer, 5, 600.0)
    assert abs(total / nominal - 1.0) < 0.01
    jit = PROFILE[P["tempo_jitter"]]
    assert tempos.min() >= 0.0
    assert tempos.max() - tempos.min() > jit / 3.0
    _, again = _clock_run(layer, 5, 20.0)
    _, other = _clock_run(layer, 6, 20.0)
    assert np.array_equal(again, tempos[:len(again)])
    assert not np.allclose(again, other)


def test_the_noise_never_touches_global_random_state():
    """PROPERTY (#164: render-only randomness): a unit's draws are a pure
    function of (seed, unit id) -- the same whatever numpy's global state
    is -- and drawing them leaves the global state untouched.

    BREAKS IF: the posture draws from ``np.random`` (global) or any shared
    generator (e.g. the sim's).
    """
    np.random.seed(1)
    a = unit_noise(POSTURE, 42, 4)
    state = np.random.get_state()[1].copy()
    np.random.seed(999)
    b = unit_noise(POSTURE, 42, 4)
    np.random.seed(1)
    unit_noise(POSTURE, 43, 4)
    assert np.array_equal(np.random.get_state()[1], state)
    for f in dataclasses.fields(a):
        assert np.array_equal(getattr(a, f.name), getattr(b, f.name))
    c = unit_noise(POSTURE, 43, 4)
    assert not np.array_equal(a.tempo_phase, c.tempo_phase)


def test_twitch_is_bounded_zero_mean_and_moves_its_bones(layer):
    """PROPERTY (brief line 10, §3 twitch): the twitch noise is bounded by
    1 and averages to ~0 over a long window; on a posed unit standing still
    its twitch bones move over time by no more than ``twitch x deg`` (plus
    numerical slack) and by a visible amount.

    BREAKS IF: the twitch is biased (a constant offset), unbounded, or
    scaled by anything but the profile's twitch and the row's degrees.
    """
    nz = unit_noise(POSTURE, 9, 6)
    tau = np.arange(0.0, 600.0, 0.01)
    n = smooth_noise(nz.twitch_phase[:, None, :], nz.twitch_hz[:, None, :], tau[None, :])
    assert np.abs(n).max() <= 1.0 + 1e-12
    assert np.abs(n.mean(-1)).max() < 0.02
    res = layer.rig(LOOK).posture
    ps = _state(layer, "idle", uid=9, a=1.0, t=0.0)
    frames = []
    for k in range(120):
        ps.tau = 0.05 * k
        frames.append(layer.compose(LOOK, [ps]).local_r[0])
    frames = np.array(frames)
    tw_bones = np.flatnonzero((np.abs(res.twitch_pre).sum((0, 2)) + np.abs(res.twitch_post).sum((0, 2))) > 0)
    d = np.abs((frames[:, tw_bones] * frames[0:1, tw_bones]).sum(-1)).clip(0, 1)
    swing = np.degrees(2 * np.arccos(d)).max()
    bound = 2 * PROFILE[P["twitch"]] * ps.noise.scale[P["twitch"]] * res.twitch_deg.max()
    assert 0.5 < swing <= bound + 0.5, (swing, bound)


def test_a_horde_does_not_move_in_unison(layer):
    """PROPERTY (brief line 19): eight posed units drawn at the same moment
    differ -- in walk phase (desync_start), in lean (variation) -- while
    each stays within the variation's bounds of the profile.

    BREAKS IF: the per-unit draws are not keyed to the unit id (every unit
    the same), or the variation exceeds its configured fraction.
    """
    states = []
    for i in range(8):                    # spawned together, at the same moment
        ps = layer.new_state(LOOK, "walk")
        layer.set_posture(ps, 100 + i, 1.0)
        layer.drive(ps, "walk", DT)
        states.append(ps)
    if POSTURE.desync_start:
        times = {round(float(ps.time[0]), 3) for ps in states}
        assert len(times) == 8
    frac = POSTURE.variation.get("spine_pitch", 0.0)
    scales = [ps.noise.scale[P["spine_pitch"]] for ps in states]
    assert all(1 - frac - 1e-12 <= s <= 1 + frac + 1e-12 for s in scales)
    if frac > 0:
        assert max(scales) - min(scales) > 0.2 * frac


# ---------------------------------------------------------------------------
# melee and the renderer's routing
# ---------------------------------------------------------------------------

class _Unit:
    def __init__(self, uid, x, y, facing=math.pi / 2, zombie=False):
        object.__setattr__(self, "_d", dict(id=uid, x=float(x), y=float(y), facing=facing,
                                            move_path=[], alive=True, footprint=3,
                                            is_zombie=zombie))

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AssertionError(f"the renderer wrote {name!r} on a unit")


class _Skin:
    def __init__(self):
        self.rows = []

    def prepare(self, *a):
        pass

    def apply(self, look, model, pose_row):
        self.rows.append(np.array(pose_row))

    def release(self):
        pass


class _Rl:
    def update_model_animation(self, *a):
        raise AssertionError("raylib's clip player called with the pose layer on")

    def __getattr__(self, name):
        return lambda *a, **k: None


def _renderer(monkeypatch, enabled=True, settings=None):
    from renderer import pose_layer as pl_mod
    from renderer import unit_model_renderer as umr
    from renderer.unit_looks import LookAssigner
    monkeypatch.setattr(umr, "rl", _Rl())
    monkeypatch.setattr(CFG.render.pose, "enabled", enabled)
    if settings is not None:
        monkeypatch.setattr(pl_mod, "pose_settings", lambda cfg: settings)
    r = umr.UnitModelRenderer()
    for name in SHIPPED:
        model = SimpleNamespace(boneCount=53, name=name)
        anims = [SimpleNamespace(tag=(name, k), frameCount=40) for k in range(2)]
        r._looks[name] = umr._Look(name=name, path=look_path(name), model=model,
                                   anims=anims, n_anims=2, clip_index={}, native_height=1.8)
        r._resolve[name] = r._looks[name]
    r._assigner = LookAssigner(looks_table(CFG))
    r._loaded = True
    r._setup_pose(skinner=_Skin())
    return r


def _cast(r, events_at, frames=8):
    """Marine 1, zombies 2 and 3 standing; ``events_at(k)`` handed over
    before frame k."""
    units = [_Unit(1, 4, 40), _Unit(2, 14, 40, zombie=True), _Unit(3, 24, 40, zombie=True)]
    for k in range(frames):
        r.queue_events(events_at(k))
        r.draw_units(units, 8.0, 1.0 + k * DT, camera3d=None, light_view=None)
    return {u.id: r._anim[u.id].pose for u in units}


def _melee_layers(r):
    return r._pose.rig(SHIPPED[0]).melee_layers


def _playing_melee(r, ps):
    rig = r._pose.rig(ps.look)
    return bool(np.isin(ps.clip[rig.melee_layers], rig.melee_clips).any())


def test_a_melee_strike_plays_on_the_zombie_attacker_only(monkeypatch):
    """PROPERTY (#164 item 4): a melee hit credited to a zombie makes THAT
    zombie play a melee clip on the melee layers -- not the victim, not a
    bystander; the next strike plays the next clip in turn; a hit of
    another cause, or one credited to a marine, plays no strike.

    BREAKS IF: the strike is routed by the victim's id (today's queue key)
    or to every unit, the cause filter is dropped, or the role gate lets a
    marine (role without the melee reaction) strike.
    """
    r = _renderer(monkeypatch)
    melee = SETTINGS.melee.causes[0]
    ps = _cast(r, lambda k: [UnitHitEvent(unit_id=3, damage=60, source=melee,
                                          attacker_id=2)] if k == 1 else [])
    assert _playing_melee(r, ps[2]) and ps[2].melee_next == 1
    assert not _playing_melee(r, ps[3]) and not _playing_melee(r, ps[1])
    assert ps[3].flinch_t >= 0.0                     # the victim flinched
    first = int(ps[2].clip[_melee_layers(r)[0]])
    _cast(r, lambda k: [UnitHitEvent(unit_id=3, damage=60, source=melee,
                                     attacker_id=2)] if k == 0 else [], frames=2)
    if len(SETTINGS.melee.clips) > 1:
        assert int(ps[2].clip[_melee_layers(r)[0]]) != first
    r = _renderer(monkeypatch)
    other = next(c for c in ("bullet", "explosion") if c not in SETTINGS.melee.causes)
    ps = _cast(r, lambda k: [UnitHitEvent(unit_id=3, damage=60, source=other, attacker_id=2),
                             UnitHitEvent(unit_id=2, damage=60, source=melee, attacker_id=1)]
               if k == 1 else [])
    assert not any(_playing_melee(r, p) for p in ps.values())


def test_a_zombie_never_takes_the_pistol_aim_on_a_shot(monkeypatch):
    """PROPERTY (#164 item 4): a launched shot credited to a zombie (a
    converted marine, say) starts no aim hold and no recoil -- the zombie
    role has no fire reaction -- while the same event for a marine does.

    BREAKS IF: the fire reaction is not gated by role.
    """
    r = _renderer(monkeypatch)
    ps = _cast(r, lambda k: [ShotFiredEvent(unit_id=u, from_tile=(0, 0), to_tile=(1, 1),
                                            launch=True) for u in (1, 2)] if k == 1 else [])
    assert ps[1].aim_left > 0.0
    assert ps[2].aim_left == 0.0
    rig = r._pose.rig(ps[2].look)
    sl = SETTINGS.layers.index(SETTINGS.fire.shot_layer)
    assert ps[2].clip[sl] != rig.shot_clip
    ov = rig.overlay_clips[SETTINGS.fire.overlay]
    assert all(ps[2].clip[l] != ov[l] for l in range(len(ov)) if ov[l] >= 0)


def test_zombies_are_postured_and_marines_are_not_in_the_renderer(monkeypatch):
    """PROPERTY (brief line 24): in the renderer a zombie's amount is its
    role's (1) and a marine's is 0, read from ``is_zombie`` without writing
    the unit (the stub raises on any write).

    BREAKS IF: the amount is not keyed to the role, or the renderer writes
    a unit.
    """
    r = _renderer(monkeypatch)
    ps = _cast(r, lambda k: [])
    assert ps[1].posture == SETTINGS.roles["player"].posture == 0.0
    assert ps[2].posture == ps[3].posture == SETTINGS.roles["zombie"].posture > 0.0


def test_a_strike_fades_the_suppressed_posture_and_restores_it(layer):
    """PROPERTY (#164): while a melee clip plays the ``suppress``ed posture
    parameters fade out by its weight (through a crossfade, never a jump)
    and come back to their full value once it ends.

    BREAKS IF: suppression is a hard switch, or it lingers after the strike.
    """
    ps = _state(layer, "idle", uid=31)
    idx = [P[p] for p in SETTINGS.melee.suppress]
    full = layer._posture_params(ps)[idx].copy()
    layer.melee(ps)
    seen, prev_w = [], 0.0
    for _ in range(int(3.0 / DT)):
        layer.drive(ps, "idle", DT)
        seen.append(ps.melee_w)
        if SETTINGS.crossfade_s > 0:
            assert abs(ps.melee_w - prev_w) <= DT / SETTINGS.crossfade_s + 1e-9
        prev_w = ps.melee_w
        assert np.allclose(layer._posture_params(ps)[idx], full * (1.0 - ps.melee_w))
    assert max(seen) == 1.0 and seen[-1] == 0.0
    assert np.array_equal(layer._posture_params(ps)[idx], full)


def test_the_off_path_ignores_posture_and_melee(monkeypatch):
    """PROPERTY (§4.5 correction 9, extended for #164): with the setting
    off, zombies and melee events change nothing -- the pose layer is never
    built and each unit is skinned by one of its look's own clips at an
    integer frame.

    BREAKS IF: posture or melee is wired outside the pose layer.
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
    r = _renderer(monkeypatch, enabled=False)
    monkeypatch.setattr(umr, "rl", _RlRec())
    assert r._pose is None
    for k in range(3):
        r.queue_events([UnitHitEvent(unit_id=1, damage=60, source="melee", attacker_id=2)])
        r.draw_units([_Unit(1, 4, 40), _Unit(2, 9, 40, zombie=True)], 8.0, 1.0 + k * DT,
                     camera3d=None, light_view=None)
    own = {id(a) for lk in r._looks.values() for a in lk.anims}
    assert len(calls) == 6 and all(id(a) in own and type(f) is int for a, f in calls)


# ---------------------------------------------------------------------------
# the door, the foreign skeleton
# ---------------------------------------------------------------------------

def _broken(edit):
    d = critter_settings()
    edit(d["render"]["pose"])
    return d


def _pr(p):
    return p["skeletons"]["critter"]["posture"]


@pytest.mark.parametrize("edit, needle", [
    (lambda p: p.pop("posture"), "posture"),
    (lambda p: p.pop("melee"), "melee"),
    (lambda p: p.pop("roles"), "roles"),
    (lambda p: p["posture"].pop("seed"), "seed"),
    (lambda p: p["posture"].update(seed=-1), "seed"),
    (lambda p: p["posture"].update(profile="charge"), "charge"),
    (lambda p: p["posture"].update(hz_spread=1.0), "hz_spread"),
    (lambda p: p["posture"].update(tempo_hz=[]), "tempo_hz"),
    (lambda p: p["posture"].update(twitch_hz=[0.0]), "twitch_hz"),
    (lambda p: p["posture"].update(desync_start=1), "desync_start"),
    (lambda p: p["posture"].update(swagger=1), "swagger"),
    (lambda p: p["posture"]["variation"].update(tail=0.1), "tail"),
    (lambda p: p["posture"]["variation"].update(spine_pitch=1.0), "spine_pitch"),
    (lambda p: p["posture"]["profiles"]["prowl"].pop("twitch"), "twitch"),
    (lambda p: p["posture"]["profiles"]["prowl"].update(tempo_jitter=1.5), "tempo_jitter"),
    (lambda p: p["posture"]["profiles"]["prowl"].update(spine_pitch="deep"), "spine_pitch"),
    (lambda p: p["posture"]["profiles"]["prowl"].update(knee_bend=-0.1), "knee_bend"),
    (lambda p: p["posture"]["profiles"]["prowl"].update(wag=1.0), "wag"),
    (lambda p: _pr(p).pop("twitch"), "twitch"),
    (lambda p: _pr(p).pop("up"), "up"),
    (lambda p: _pr(p).update(forward=[0.0, 1.0, 0.0]), "perpendicular"),
    (lambda p: _pr(p).update(up=[0.0, 0.0, 0.0]), "up"),
    (lambda p: _pr(p)["spine_pitch"][0].update(space="world"), "space"),
    (lambda p: _pr(p)["spine_pitch"][0].update(deg="x"), "deg"),
    (lambda p: _pr(p)["spine_pitch"][0].pop("axis"), "axis"),
    (lambda p: _pr(p)["spine_pitch"][0].update(bones="back_x"), "bones"),
    (lambda p: _pr(p)["spine_pitch"][0].update(gain=2), "gain"),
    (lambda p: _pr(p).update(head_drop={"bones": ["skull_x"]}), "head_drop"),
    (lambda p: _pr(p)["knee_bend"].update(legs=[["a", "b"]]), "legs"),
    (lambda p: _pr(p)["knee_bend"].pop("hips"), "hips"),
    (lambda p: p["melee"].update(layers=["tail"]), "tail"),
    (lambda p: p["melee"].update(suppress=["swagger"]), "swagger"),
    (lambda p: p["melee"].update(causes=[]), "causes"),
    (lambda p: p["melee"].pop("clips"), "clips"),
    (lambda p: p["roles"]["zombie"].update(posture=1.5), "posture"),
    (lambda p: p["roles"]["zombie"].update(reactions=["bite"]), "bite"),
    (lambda p: p["roles"]["player"].pop("reactions"), "reactions"),
])
def test_the_posture_melee_and_role_settings_refuse_bad_keys(edit, needle):
    """PROPERTY (#164): the posture, profile, skeleton-posture, melee and
    role tables are refused, naming the key, when a key is missing,
    unknown, mistyped or out of range.

    BREAKS IF: any new key falls back to a silent default.
    """
    with pytest.raises(PoseConfigError, match=re.escape(needle)):
        pose_settings(_broken(edit))


@pytest.mark.parametrize("where, needle", [
    ("spine", "spine_q"), ("twitch", "skull_q"), ("hips", "pelvis_q"),
    ("leg", "not a thigh"), ("melee", "melee"),
])
def test_posture_bones_and_melee_clips_are_checked_on_each_skeleton(tmp_path, where, needle):
    """PROPERTY (#164, skeleton-agnostic): a posture row, twitch row or hip
    naming a bone the skeleton lacks, a leg that is not a thigh -> shin ->
    foot chain, or a melee clip the look lacks is refused at load, naming it.

    BREAKS IF: posture bones are assumed (hard-coded) or not checked.
    """
    path = write_glb(tmp_path / "critter.glb", CRITTER_BONES, critter_clips())
    d = critter_settings()
    p = d["render"]["pose"]
    rule = p["skeletons"]["critter"]["posture"]
    if where == "spine":
        rule["spine_pitch"][0]["bones"] = ["spine_q"]
    elif where == "twitch":
        rule["twitch"][0]["bones"] = ["skull_q"]
    elif where == "hips":
        rule["knee_bend"]["hips"] = "pelvis_q"
    elif where == "leg":
        rule["knee_bend"]["legs"] = [["pelvis_x", "skull_x", "hindleg_l"]]
    else:
        p["melee"]["clips"] = ["haymaker"]
        needle = "haymaker"
    with pytest.raises(PoseConfigError, match=re.escape(needle)):
        PoseLayer(pose_settings(d)).add_look("critter", path)


def test_a_role_table_must_name_exactly_the_renderers_roles(monkeypatch):
    """PROPERTY (#164): ``[render.pose.roles]`` names exactly the look
    table's roles and the melee causes only the sim's damage causes; a
    missing role or a typo'd cause is refused when the renderer builds the
    layer.

    BREAKS IF: a role silently gets no reactions, or a melee cause that can
    never occur is accepted.
    """
    roles = dict(SETTINGS.roles)
    roles.pop("zombie")
    with pytest.raises(PoseConfigError, match="roles"):
        _renderer(monkeypatch, settings=dataclasses.replace(SETTINGS, roles=roles))
    melee = dataclasses.replace(SETTINGS.melee, causes=("mele",))
    with pytest.raises(PoseConfigError, match="mele"):
        _renderer(monkeypatch, settings=dataclasses.replace(SETTINGS, melee=melee))


def test_a_foreign_skeleton_runs_a_posture_profile(tmp_path):
    """PROPERTY (#164, skeleton-agnostic): the five-bone creature with its
    own names and posture table composes a posed unit through the same
    code: its back pitches forward with the amount, its pelvis drops, its
    melee clip plays, and amount 0 is its plain pose bit for bit.

    BREAKS IF: any of it assumes the shipped rig.
    """
    path = write_glb(tmp_path / "critter.glb", CRITTER_BONES, critter_clips())
    pl = PoseLayer(pose_settings(critter_settings()))
    rig = pl.add_look("critter", path)
    sk = rig.bank.skeleton
    back, pelvis = sk.index("back_x"), sk.index("pelvis_x")
    plain = pl.new_state("critter", "walk")
    posed = _clone(pl, plain)
    pl.set_posture(posed, 1, 1.0)
    posed.time[:] = plain.time
    pl.melee(posed)
    pl.drive(posed, "walk", DT)
    pl.drive(plain, "walk", DT)
    assert np.isin(posed.clip[rig.melee_layers], rig.melee_clips).all()
    out = pl.compose("critter", [plain, posed]).global_trs
    assert out[1, pelvis, 1] < out[0, pelvis, 1] - 0.05
    y0, y1 = _mats(out[0, back])[:3, 1], _mats(out[1, back])[:3, 1]
    assert y1[2] - y0[2] > 0.3                        # tipped toward +z (forward)
    zero = _clone(pl, posed)
    zero.posture = 0.0
    ref = _clone(pl, posed)
    ref.noise = None
    ref.posture = 0.0
    o = pl.compose("critter", [zero, ref]).global_trs
    assert np.array_equal(o[0], o[1])
