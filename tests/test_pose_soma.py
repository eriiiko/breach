"""#179 -- the pose layer on a second skeleton: the SOMA look
(``assets/models/soma_worker_male``, 30 joints, Kimodo clips) beside the
Quaternius looks, through the per-skeleton playback of ``[render.pose]``
(``clip_names``, ``rate``, ``clips``, ``roles``). Properties only; every test
names the property it protects and the change that must break it.

The GL half (raylib loads and skins the SOMA look in the same process as
the Quaternius looks) is ``tests/gl_pose_inject_check.py``'s ``EXTRA_LOOKS``.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_pose_soma.py -q
"""
from __future__ import annotations

import copy
import dataclasses
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from renderer.pose_layer import (NO_CLIP, PoseConfigError, PoseLayer,  # noqa: E402
                                 SkeletonPlayback, forward_kinematics, pose_settings)
from renderer.pose_posture import PARAMS, apply  # noqa: E402
from renderer.unit_looks import look_path  # noqa: E402

from _pose_looks import quaternius_looks  # noqa: E402

SOMA = "soma_worker_male"
SOMA_MARINE = "soma_space_marine"                # #184: the player's look
SOMA_TABLE = "soma"
QUAT = quaternius_looks()                        # the Quaternius looks (space_marine first)
SETTINGS = pose_settings(CFG)


@pytest.fixture(scope="module")
def layer() -> PoseLayer:
    pl = PoseLayer(SETTINGS)
    for name in QUAT + [SOMA, SOMA_MARINE]:
        pl.add_look(name, look_path(name))
    return pl


def _raw_pose() -> dict:
    """``[render.pose]`` as a plain dict, to edit for the refusal tests."""
    from renderer.pose_layer import _as_dict
    return {"render": {"pose": copy.deepcopy(_as_dict(CFG.render.pose))}}


# ---------------------------------------------------------------------------
# the skeleton entry
# ---------------------------------------------------------------------------

def test_the_soma_look_takes_the_soma_table_and_its_masks_partition_it(layer):
    """PROPERTY: the SOMA look resolves to ``[render.pose.skeletons.soma]``
    and its masks cover every bone exactly once -- the spine chain's root in
    ``upper``, the neck chain's root in ``head``, the skin root in ``legs``.

    BREAKS IF: a mask rule leaves a bone out or claims one twice (a bone
    driven by no layer, or by two), or the Quaternius table starts matching
    the SOMA skeleton first.
    """
    rig = layer.rig(SOMA)
    assert rig.skeleton_table == SOMA_TABLE
    assert np.array_equal(rig.masks.weights.sum(0), np.ones(rig.bank.skeleton.n_bones))
    sk = rig.bank.skeleton
    owner = {sk.names[j]: rig.masks.names[m] for j, m in enumerate(rig.masks.owner)}
    rules = dict(dict(SETTINGS.skeletons)[SOMA_TABLE])
    for mask, rule in rules.items():
        for root in rule.subtree:
            assert owner[root] == mask, (root, owner[root], mask)
    for name in QUAT:
        assert layer.rig(name).skeleton_table != SOMA_TABLE


def test_every_mapped_clip_name_is_a_clip_of_the_soma_look(layer):
    """PROPERTY: every ``clip_names`` value of the SOMA table that is not
    "absent" names a clip the SOMA file holds, and every state, overlay,
    shot and flinch clip resolves to a real clip of the look.

    BREAKS IF: a clip is renamed in a re-export, or a mapping is typed wrong
    (the door at load would refuse it; this pins it on the shipped files).
    """
    rig = layer.rig(SOMA)
    for canonical, own in SETTINGS.playback[SOMA_TABLE].clip_names.items():
        assert own == "" or rig.bank.has(own), (canonical, own)
    for ids in rig.state_clips.values():
        assert (ids >= 0).all()
    assert rig.shot_clip != NO_CLIP and rig.hit_clip != NO_CLIP


def test_a_mapped_name_the_file_lacks_is_refused_at_load_by_name():
    """PROPERTY: a ``clip_names`` value that is not a clip of the look is
    refused when the look is added, naming the clip; a ``clip_names`` KEY
    that no table uses is refused by the settings door, naming it.

    BREAKS IF: a missing mapped clip silently falls back (the look would
    play the wrong animation with no word), or a typo'd key maps nothing.
    """
    d = _raw_pose()
    d["render"]["pose"]["skeletons"][SOMA_TABLE]["clip_names"]["Walk_Loop"] = "Wakl"
    pl = PoseLayer(pose_settings(d))
    with pytest.raises(PoseConfigError, match="Wakl"):
        pl.add_look(SOMA, look_path(SOMA))
    d = _raw_pose()
    d["render"]["pose"]["skeletons"][SOMA_TABLE]["clip_names"]["Walk_Lop"] = "Walk"
    with pytest.raises(PoseConfigError, match="Walk_Lop"):
        pose_settings(d)
    d = _raw_pose()
    d["render"]["pose"]["skeletons"][SOMA_TABLE]["surprise"] = 1
    with pytest.raises(PoseConfigError, match="surprise"):
        pose_settings(d)


def test_looping_soma_clips_wrap_seamlessly(layer):
    """PROPERTY: every SOMA clip that loops ends on a key equal to its first
    (rotations up to sign, translations to 0.1 mm), so the wrap at its
    duration shows no pop.

    BREAKS IF: a looping clip is re-exported with its travel or a drift left
    in, or a one-shot is marked looping in the SOMA clip rows.
    """
    bank = layer.rig(SOMA).bank
    looped = [c for c in range(bank.n_clips) if bank.loop[c]]
    assert looped
    for c in looped:
        o, k = bank.offset[c], bank.n_keys[c]
        r0, r1 = bank.r[o].astype(float), bank.r[o + k - 1].astype(float)
        assert np.abs(np.abs((r0 * r1).sum(-1)) - 1.0).max() < 1e-5, bank.names[c]
        assert np.abs(bank.t[o] - bank.t[o + k - 1]).max() < 1e-4, bank.names[c]


# ---------------------------------------------------------------------------
# playback speed: the feet do not skate
# ---------------------------------------------------------------------------

def _stance_speed(rig, clip: str) -> float:
    """The clip's in-place stride speed (model m/s of clip time): the
    median backward speed of the feet while they are on the ground (their
    lowest 30 %), the feet being the knee-bend legs' last bones."""
    bank, sk = rig.bank, rig.bank.skeleton
    c = bank.index(clip)
    o, k = bank.offset[c], bank.n_keys[c]
    pos = forward_kinematics(sk, bank.t[o:o + k], bank.r[o:o + k], None)[..., :3, 3]
    rule = SETTINGS.postures[rig.skeleton_table]
    speeds = []
    for leg in rule.knee.legs:
        j = sk.index(leg[2])
        y, z = pos[:, j, 1], pos[:, j, 2]
        vz = np.gradient(z, bank.key_dt[c])
        speeds.append(np.median(-vz[y < np.percentile(y, 30)]))
    return float(np.mean(speeds))


@pytest.mark.parametrize("look", [SOMA, SOMA_MARINE])
@pytest.mark.parametrize("clip", ["Walk_Loop", "Pistol_Idle_Loop", "Zombie_Hunch_Walk"])
def test_the_measured_stride_agrees_with_an_independent_measurement(layer, look, clip):
    """PROPERTY (#184): the pose code's stride measurement
    (``pose_locomotion.measure_gait``, the ONE the derived rates use) of a
    forward walking clip agrees within 5 % with this file's own, simpler
    measurement (the feet's median backward speed while low, the knee-bend
    legs' last bones) -- and says the clip walks forward (within 15 deg).

    BREAKS IF: the measurement is changed into something else (a wrong
    axis, the swing foot instead of the planted one, a sign), which would
    make every derived rate skate.
    """
    from renderer.pose_locomotion import measure_gait
    rig = layer.rig(look)
    own = rig.clip_name(clip)
    sk = rig.bank.skeleton
    feet = [sk.index(leg[2]) for leg in SETTINGS.postures[rig.skeleton_table].knee.legs]
    m = measure_gait(rig.bank, rig.bank.index(own), feet)
    assert m is not None, own
    ref = _stance_speed(rig, own)
    assert abs(m.stride / ref - 1.0) < 0.05, (own, m.stride, ref)
    assert abs(math.degrees(m.travel)) < 15.0, (own, math.degrees(m.travel))


# ---------------------------------------------------------------------------
# the Quaternius path is untouched
# ---------------------------------------------------------------------------

def _scripted(pl: PoseLayer, look: str, frames: int = 150) -> np.ndarray:
    """A run exercising every path: states, the shot, the flinch, posture,
    the melee strike and the twist; every frame's global pose."""
    states = [pl.new_state(look, "idle", role) for role in ("player", "player", "zombie")]
    out, dt = [], 1.0 / 60.0
    for f in range(frames):
        for uid, ps in enumerate(states):
            role = "zombie" if uid == 2 else "player"
            pl.set_posture(ps, uid, pl.role_posture(look, role))
            st = "walk" if (f // 30 + uid) % 2 == 0 else "idle"
            if role == "player" and f % 23 == uid * 5:
                pl.fire(ps)
            if f % 41 == uid * 7:
                pl.hit(ps, 15 + 20 * uid, "bullet")
            if role == "zombie" and f % 50 == 3:
                pl.melee(ps)
            pl.steer(ps, None if st == "idle" else math.radians(25 * uid + f), st == "walk", dt)
            pl.drive(ps, st, dt, role)
        out.append(pl.compose(look, states).global_trs.copy())
    return np.stack(out)


def test_the_quaternius_looks_ignore_every_other_skeletons_table(layer):
    """PROPERTY: a Quaternius look composes BIT FOR BIT the same pose
    through a scripted run (states, shot, flinch, posture, melee, twist)
    whether the SOMA table is there or not, and whatever the SOMA table's
    rate, clip rows and role overrides say. (Measured once more at #179:
    the composed poses equal those of the pre-#179 code, 240 frames x 4
    units x every shipped look, max |diff| 0.0.)

    BREAKS IF: a per-skeleton key leaks into another skeleton's looks
    (a global rate, clip row or role posture read from the wrong table).
    """
    look = QUAT[0]
    ref = _scripted(layer, look)
    without = dataclasses.replace(
        SETTINGS, skeletons=tuple(s for s in SETTINGS.skeletons if s[0] != SOMA_TABLE),
        playback={k: v for k, v in SETTINGS.playback.items() if k != SOMA_TABLE})
    pl = PoseLayer(without)
    pl.add_look(look, look_path(look))
    assert np.array_equal(_scripted(pl, look), ref)
    soma = SETTINGS.playback[SOMA_TABLE]
    odd = dataclasses.replace(SETTINGS, playback={
        **SETTINGS.playback,
        SOMA_TABLE: SkeletonPlayback(clip_names=dict(soma.clip_names), rate=7.0,
                                     clips={}, roles={})})
    pl = PoseLayer(odd)
    pl.add_look(look, look_path(look))
    assert np.array_equal(_scripted(pl, look), ref)


def test_a_skeleton_table_without_playback_plays_the_global_tables(layer):
    """PROPERTY: a skeleton table that carries no playback keys plays at the
    global ``rate`` with the global clip rows and role postures -- the
    Quaternius looks' rig rate IS ``[render.pose] rate``.

    BREAKS IF: an absent per-skeleton key stops meaning "the global one".
    """
    for look in QUAT:
        rig = layer.rig(look)
        assert rig.rate == SETTINGS.rate
        for role, row in SETTINGS.roles.items():
            assert layer.role_posture(look, role) == row.posture
        c = rig.bank.index(SETTINGS.fire.shot_clip)
        assert rig.bank.rate[c] == SETTINGS.clips[SETTINGS.fire.shot_clip].rate


# ---------------------------------------------------------------------------
# roles on the SOMA skeleton
# ---------------------------------------------------------------------------

def test_a_soma_zombie_walks_its_own_clip_and_a_quaternius_zombie_the_shared_one(layer):
    """PROPERTY: the SOMA table's ``roles.zombie`` rows apply to SOMA looks
    only: a SOMA zombie walking plays the row's clip on every layer at the
    row's posture, a SOMA player the global walk; a Quaternius zombie the
    global walk at the global zombie posture.

    BREAKS IF: role rows are read for the wrong skeleton, or the role is
    dropped between the renderer and ``drive``.
    """
    rig = layer.rig(SOMA)
    sr = SETTINGS.playback[SOMA_TABLE].roles["zombie"]
    want = [rig.bank.index(rig.clip_name(sr.states["walk"][lay])) for lay in SETTINGS.layers]
    ps = layer.new_state(SOMA, "walk", "zombie")
    layer.drive(ps, "walk", 0.05, "zombie")
    assert list(ps.clip) == want
    assert layer.role_posture(SOMA, "zombie") == sr.posture
    ps = layer.new_state(SOMA, "walk", "player")
    layer.drive(ps, "walk", 0.05, "player")
    assert list(ps.clip) == list(rig.state_clips["walk"])
    q = layer.rig(QUAT[0])
    ps = layer.new_state(QUAT[0], "walk", "zombie")
    layer.drive(ps, "walk", 0.05, "zombie")
    assert list(ps.clip) == list(q.state_clips["walk"])
    assert layer.role_posture(QUAT[0], "zombie") == SETTINGS.roles["zombie"].posture


def test_an_absent_clip_switches_its_reaction_off_on_that_look(layer):
    """PROPERTY: a reaction whose clip the skeleton maps to "" (SOMA has no
    strike clip) does nothing on that look -- a melee strike leaves every
    layer on its state clip -- while the same reaction still plays on the
    looks that have the clip.

    BREAKS IF: an absent clip raises at load, or plays a wrong clip.
    """
    assert layer.rig(SOMA).melee_clips is None
    ps = layer.new_state(SOMA, "idle", "zombie")
    layer.melee(ps)
    layer.drive(ps, "idle", 0.05, "zombie")
    assert list(ps.clip) == list(layer.rig(SOMA).state_clips["idle"])
    q = layer.rig(QUAT[0])
    ps = layer.new_state(QUAT[0], "idle", "zombie")
    layer.melee(ps)
    layer.drive(ps, "idle", 0.05, "zombie")
    assert any(c in q.melee_clips for c in ps.clip)


# ---------------------------------------------------------------------------
# posture maps: the same carriage on both rigs
# ---------------------------------------------------------------------------

def _point_shift(rig, param: str, bones) -> np.ndarray:
    """Model-space displacement of ``bones`` (mean) when ONLY ``param`` is
    applied (one unit of it; degrees for the degree parameters) on the
    look's idle first key."""
    bank, sk = rig.bank, rig.bank.skeleton
    c = rig.state_clips["idle"][0]
    o = bank.offset[c]
    lt, lr = bank.t[o:o + 1].copy(), bank.r[o:o + 1].copy()
    p = np.zeros((1, len(PARAMS)))
    p[0, PARAMS.index(param)] = 20.0 if param in ("spine_pitch", "arm_forward") else 0.3
    t2, r2 = apply(rig.posture, lt, lr, p, np.zeros((1, rig.posture.n_channels)))
    g0 = forward_kinematics(sk, lt, lr, None)[0, :, :3, 3]
    g1 = forward_kinematics(sk, t2, r2, None)[0, :, :3, 3]
    idx = [sk.index(b) for b in bones]
    return (g1[idx] - g0[idx]).mean(0)


# param -> (SOMA bones, Quaternius bones, the component that must agree in sign)
_CARRIAGE = {
    "spine_pitch": (["Head"], ["DEF-head"], 2),                 # the head goes forward
    "arm_forward": (["LeftHand", "RightHand"], ["DEF-hand.L", "DEF-hand.R"], 2),
    "head_drop": (["Head"], ["DEF-head"], 2),
    "knee_bend": (["Hips"], ["DEF-hips"], 1),                   # the hips go down
}


@pytest.mark.parametrize("param", sorted(_CARRIAGE))
def test_soma_posture_moves_the_body_the_way_the_quaternius_map_does(layer, param):
    """PROPERTY: each SOMA posture parameter moves the same body point the
    same way as the Quaternius map does (the lean and the head thrust take
    the head forward, the arm carry takes the hands forward, the knee bend
    takes the hips down), by a visible amount (> 2 cm) on both rigs.

    BREAKS IF: a SOMA posture row's axis sign or bone is wrong (a zombie
    that leans back, or raises its arms behind it).
    """
    s_bones, q_bones, comp = _CARRIAGE[param]
    soma = _point_shift(layer.rig(SOMA), param, s_bones)[comp]
    quat = _point_shift(layer.rig(QUAT[0]), param, q_bones)[comp]
    assert abs(soma) > 0.02 and abs(quat) > 0.02, (soma, quat)
    assert np.sign(soma) == np.sign(quat), (soma, quat)


def test_no_soma_bone_is_named_in_the_pose_modules(layer):
    """PROPERTY (the skeleton-agnostic rule, §2 rule 1): no SOMA bone name
    appears as a string literal in the pose modules -- SOMA is data.

    BREAKS IF: SOMA support is added as a code branch on a bone name.
    """
    names = set(layer.rig(SOMA).bank.skeleton.names)
    for mod in ("pose_clips", "pose_layer", "pose_modifiers", "pose_posture"):
        src = (ROOT / "renderer" / f"{mod}.py").read_text(encoding="utf-8")
        literals = set(re.findall(r"""["']([^"'\n]+)["']""", src))
        assert not (literals & names), (mod, sorted(literals & names))
