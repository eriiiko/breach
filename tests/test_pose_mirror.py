"""#187 -- the clip mirror and left-handed units (render side only).

A skeleton table's ``mirror_pairs`` pairs every bone with its mirror partner
(a name rule); ``pose_clips.MirrorRig`` turns a pose into its mirror image
``G'_j = M G_{m(j)} A_j``; a ``mirror_of`` clip row is baked from it at
load; a left-handed unit (``[render.unit_looks] left_handed``, dealt by
``unit_looks.HandAssigner``) is drawn as the mirror image of what it would
be, its held prop on the partner hand. Properties only; every test names
the property it protects and the change that must break it.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_pose_mirror.py -q
"""
from __future__ import annotations

import copy
import dataclasses
import math
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
from renderer.held_props import (grip_matrix, held_prop_settings,  # noqa: E402
                                 mirrored_grip, model_extent, prop_world)
from renderer.pose_clips import (MIRROR, ClipRow, PoseDataError,  # noqa: E402
                                 build_clip_bank, mirror_globals, mirror_rig)
from renderer.pose_layer import (PoseConfigError, PoseLayer,  # noqa: E402
                                 forward_kinematics, pose_settings)
from renderer.unit_looks import HandAssigner, handedness_table, look_path  # noqa: E402

from _pose_looks import quaternius_looks  # noqa: E402

SOMA_MARINE = "soma_space_marine"
SOMA_WORKER = "soma_worker_male"
QUAT = quaternius_looks()
SETTINGS = pose_settings(CFG)
DT = 1.0 / 60.0
LOOKS = [SOMA_MARINE, SOMA_WORKER] + QUAT[:1]


@pytest.fixture(scope="module")
def layer() -> PoseLayer:
    pl = PoseLayer(SETTINGS)
    for name in LOOKS:
        pl.add_look(name, look_path(name))
    return pl


def _clip_fk(bank, name):
    c = bank.index(name)
    o, n = bank.offset[c], bank.n_keys[c]
    s = None if bank.s is None else bank.s[o:o + n]
    return forward_kinematics(bank.skeleton, bank.t[o:o + n], bank.r[o:o + n], s
                              ).astype(np.float64)


def _skin(bank, g):
    return np.einsum("kjab,jbc->kjac", g, bank.skeleton.bind_inverse.astype(np.float64))


# ---------------------------------------------------------------------------
# the mirror itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("look", LOOKS)
def test_the_mirror_of_the_mirror_is_the_original(layer, look):
    """PROPERTY: mirroring a pose twice gives the pose back (to float
    tolerance), on every shipped skeleton with ``mirror_pairs`` -- the
    partner map is an involution and ``A_{m(j)} = A_j^-1``.

    BREAKS IF: the partner map is not symmetric (a token swapped one way
    only), or the conjugator is built from the wrong side (``bind_j`` and
    ``bind_{m(j)}`` exchanged), so a second mirror drifts the bones.
    """
    bank = layer.rig(look).bank
    mr = layer.rig(look).mirror
    assert mr is not None
    assert np.array_equal(mr.partner[mr.partner], np.arange(bank.skeleton.n_bones))
    g = _clip_fk(bank, bank.names[0])
    assert np.abs(mirror_globals(mr, mirror_globals(mr, g)) - g).max() < 1e-5


@pytest.mark.parametrize("look", LOOKS)
def test_the_rest_pose_is_a_fixed_point_of_the_mirror(layer, look):
    """PROPERTY: the bind pose -- a symmetric pose by definition of the
    mirror, whatever the rest's own asymmetry (SOMA's joints sit up to
    ~8 mm off each other's reflection) -- maps onto itself.

    BREAKS IF: the mirror reflects the partner's GLOBAL transform without
    conjugating by the rests (``M G_{m(j)} M``), which moves every bone of
    an asymmetric rest even at rest, or a bone is paired with the wrong
    partner.
    """
    sk = layer.rig(look).bank.skeleton
    bind = np.linalg.inv(sk.bind_inverse.astype(np.float64))[None]
    out = mirror_globals(layer.rig(look).mirror, bind)
    assert np.abs(out - bind).max() < 1e-6


def test_a_mirrored_clip_moves_each_bone_as_its_partner_reflected(layer):
    """PROPERTY: in a ``mirror_of`` clip (baked once, in LOCAL space, at
    load) every bone's skin matrix is its partner's in the source clip,
    reflected across the sagittal plane: ``skin'_j = M skin_{m(j)} M`` at
    every key -- so the mesh deforms as the mirror image.

    BREAKS IF: the local bake loses the parent conjugation (a child's
    rotation reflected in the wrong frame), swaps the tracks without
    reflecting them, or the root-motion strip runs before the mirror.
    """
    bank = layer.rig(SOMA_MARINE).bank
    rows = SETTINGS.playback["soma"].clips
    mirrors = [(n, r.mirror_of) for n, r in rows.items() if r.mirror_of]
    assert mirrors
    p = bank.mirror.partner
    M = MIRROR
    for name, src in mirrors:
        s_src = _skin(bank, _clip_fk(bank, src))
        s_mir = _skin(bank, _clip_fk(bank, name))
        want = np.einsum("ab,kjbc,cd->kjad", M, s_src[:, p], M)
        assert np.abs(s_mir - want).max() < 2e-4, name


def test_hit_left_and_its_mirror_lurch_opposite_ways(layer):
    """PROPERTY: the hit from the right (the ``hit_directions`` row at -90
    deg) moves the head to the body's LEFT (+x) and the hit from the left
    to its right -- each away from where the hit came from (#184's
    finding: the file's Hit_Right went the same way as Hit_Left).

    BREAKS IF: the right-side row names a clip that leans the wrong way
    again, or the mirror stops reflecting x.
    """
    rig = layer.rig(SOMA_MARINE)
    sk = rig.bank.skeleton
    rows = {round(math.degrees(f)): c for c, f in SETTINGS.playback["soma"].hit_directions}

    def head_swing(clip):
        g = _clip_fk(rig.bank, clip)
        x = g[:, :, 0, 3] - g[:, :1, 0, 3]       # every bone's x, from its first key
        unpaired = rig.mirror.partner == np.arange(sk.n_bones)
        # the unpaired (midline) bone that swings furthest: the head chain
        j = int(np.argmax(np.where(unpaired, np.abs(x).max(0), -1.0)))
        return float(x[np.argmax(np.abs(x[:, j])), j])

    assert head_swing(rows[90]) < -0.1           # from the left: toward its right
    assert head_swing(rows[-90]) > 0.1           # from the right: toward its left


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def _raw_pose() -> dict:
    from renderer.pose_layer import _as_dict
    return {"render": {"pose": copy.deepcopy(_as_dict(CFG.render.pose))}}


@pytest.mark.parametrize("edit, needle", [
    (lambda c: c.__setitem__("Bad_M", {"mirror_of": "No_Such_Row"}), "No_Such_Row"),
    (lambda c: c.__setitem__("Chain_M", {"mirror_of": "Hit_Left_Mirror"}), "itself a mirror"),
])
def test_a_bad_mirror_row_is_refused_by_name(edit, needle):
    """PROPERTY: the door refuses a ``mirror_of`` naming no row of the
    table, and a mirror of a mirror, naming the row.

    BREAKS IF: a typo bakes nothing silently, or chains are accepted
    (their source is not a clip of the file)."""
    raw = _raw_pose()
    edit(raw["render"]["pose"]["skeletons"]["soma"]["clips"])
    with pytest.raises(PoseConfigError, match=needle):
        pose_settings(raw)


def test_mirror_rows_without_mirror_pairs_are_refused():
    """PROPERTY: a skeleton table with ``mirror_of`` rows but no
    ``mirror_pairs`` is refused at the door.

    BREAKS IF: the bake meets the row with no partner map and fails late
    (or mirrors onto the same bones)."""
    raw = _raw_pose()
    del raw["render"]["pose"]["skeletons"]["soma"]["mirror_pairs"]
    with pytest.raises(PoseConfigError, match="mirror_pairs"):
        pose_settings(raw)


def test_a_pair_rule_naming_a_missing_partner_and_a_shadowing_row_are_refused():
    """PROPERTY: a name rule that maps a bone onto a name the skeleton
    lacks is refused naming both bones; a mirror row named like a clip of
    the file (it would shadow it) is refused naming it.

    BREAKS IF: an unpaired-by-accident bone silently mirrors onto itself,
    or a mirror row quietly replaces a file clip."""
    path = look_path(SOMA_MARINE)
    with pytest.raises(PoseDataError, match="has no partner"):
        build_clip_bank(path, {}, (("Left", "Lefty"),))
    with pytest.raises(PoseDataError, match="shadow"):
        build_clip_bank(path, {"Hit_Right": ClipRow(mirror_of="Hit_Left")},
                        (("Left", "Right"),))


# ---------------------------------------------------------------------------
# left-handed units
# ---------------------------------------------------------------------------

def _drive(pl, look, left, travel, hit_src, role="player", posture=0.0):
    ps = pl.new_state(look, "idle", role, left)
    if posture:
        pl.set_posture(ps, 7, posture)
    for k in range(90):
        state = pl.locomote(ps, "walk", 1.1 * DT, travel, True, DT, role, DT)
        if k == 40:
            pl.hit(ps, 30.0, "bullet", hit_src)
        if k == 50:
            pl.fire(ps)
        if k == 60:
            pl.stagger(ps, hit_src)      # #185's stagger, from the same side
        pl.drive(ps, state, DT, role)
    return ps


@pytest.mark.parametrize("look, role, posture", [
    (SOMA_MARINE, "player", 0.0), (QUAT[0], "player", 0.0), (QUAT[0], "zombie", 1.0)])
def test_a_left_hander_is_the_mirror_image_of_a_right_hander_in_the_mirrored_world(
        layer, look, role, posture):
    """PROPERTY: a left-handed unit moving along ``a`` and hit from ``s``
    is drawn EXACTLY as the mirror image of a right-handed one moving
    along ``-a`` and hit (and staggered) from ``-s`` -- locomotion (gait,
    sector, phase, the twist), the reactions and the posture all included.

    BREAKS IF: a direction input is not reflected for the left-hander (it
    would strafe the wrong way, its legs twisted off its travel, or flinch
    or stagger toward the shooter), or the mirror runs before a modifier (the twist
    applied after it turns the wrong way).
    """
    a, s = math.radians(70.0), math.radians(-60.0)
    left = _drive(layer, look, True, a, s, role, posture)
    right = _drive(layer, look, False, -a, -s, role, posture)
    assert left.mirrored and not right.mirrored
    gl = layer.compose(look, [left]).global_trs
    gr = layer.compose(look, [right])
    # rebuild matrices from the right-hander's rows, mirror, compare rows
    from renderer.pose_layer import global_trs
    from renderer.held_props import trs_matrix
    mats = np.stack([trs_matrix(row) for row in gr.global_trs[0]])[None]
    want = global_trs(mirror_globals(layer.rig(look).mirror, mats).astype(np.float32))
    assert np.abs(gl[..., :3] - want[..., :3]).max() < 2e-4
    dq = np.abs((gl[..., 3:7] * want[..., 3:7]).sum(-1))
    assert (1.0 - dq).max() < 1e-5


def test_a_left_hander_holds_the_carbine_in_its_left_hand(layer):
    """PROPERTY: a left-hander's carbine hangs on the PARTNER of the row's
    socket bone (the ``left_hand`` socket's bone for a right-hand row) and
    is the right-hander's carbine mirrored -- reflected across the body and
    in its own symmetry plane, so a proper transform, not an inside-out one.

    BREAKS IF: the left-hander keeps the right-hand bone, the grip is not
    re-derived (the carbine floats off the left palm), or the prop is
    drawn reflected (determinant -1: inside out, culled).
    """
    rig = layer.rig(SOMA_MARINE)
    prop = held_prop_settings(CFG).props["carbine"]
    grip = grip_matrix(prop, prop.socket, model_extent(look_path(prop.model)))
    right = _drive(layer, SOMA_MARINE, False, -0.3, None)
    left = _drive(layer, SOMA_MARINE, True, 0.3, None)
    bone = layer.socket_bone(left, "carbine", prop.socket)
    held = layer.held_bone(left, bone)
    assert held == rig.sockets["left_hand"] and bone == rig.sockets["right_hand"]
    assert layer.held_bone(right, bone) == bone
    gl = layer.compose(SOMA_MARINE, [left]).global_trs[0]
    gr = layer.compose(SOMA_MARINE, [right]).global_trs[0]
    w_left = prop_world(np.eye(4), gl[held], mirrored_grip(
        layer.mirror_conj(SOMA_MARINE, bone), grip, prop.mirror_axis))
    z = np.diag([1.0, 1.0, -1.0, 1.0])
    w_want = np.einsum("ab,bc,cd->ad", MIRROR, prop_world(np.eye(4), gr[bone], grip), z)
    assert np.abs(w_left - w_want).max() < 1e-3
    assert np.linalg.det(w_left[:3, :3]) > 0.0
    # the grip point sits as far from the left hand bone as the
    # right-hander's from the right one (the palm, ~10 cm from the wrist)
    gp = np.array(list(prop.grip) + [1.0])
    p_left = np.einsum("ab,b->a", w_left, gp)[:3]
    p_right = np.einsum("ab,b->a", prop_world(np.eye(4), gr[bone], grip), gp)[:3]
    d_right = np.linalg.norm(p_right - gr[bone, :3])
    assert abs(np.linalg.norm(p_left - gl[held, :3]) - d_right) < 1e-3


def test_right_handed_units_compose_as_without_any_mirror(layer):
    """PROPERTY: a right-handed unit (every unit at share 0) composes bit
    for bit as on a layer whose skeleton tables have no ``mirror_pairs``
    and no mirror rows (the picture before #187), even batched with
    left-handers.

    BREAKS IF: the mirror touches rows that are not left-handed, or the
    mere presence of ``mirror_pairs`` changes the bake of the file's clips.
    """
    pb = {k: dataclasses.replace(
        v, mirror_pairs=(),
        clips=None if v.clips is None else {n: r for n, r in v.clips.items()
                                            if not r.mirror_of})
        for k, v in SETTINGS.playback.items()}
    plain = PoseLayer(dataclasses.replace(SETTINGS, playback=pb))
    look = QUAT[0]
    plain.add_look(look, look_path(look))
    a = _drive(layer, look, False, 0.4, 0.5)
    b = _drive(plain, look, False, 0.4, 0.5)
    lh = _drive(layer, look, True, 0.4, 0.5)
    got = layer.compose(look, [a, lh]).global_trs[0]
    assert np.array_equal(got, plain.compose(look, [b]).global_trs[0])
    assert plain.rig(look).mirror is None
    assert not plain.new_state(look, "idle", "player", True).mirrored


# ---------------------------------------------------------------------------
# the deal
# ---------------------------------------------------------------------------

def _u(uid, zombie=False):
    return SimpleNamespace(id=uid, is_zombie=zombie)


def test_share_zero_deals_no_left_hander_and_one_deals_all():
    """PROPERTY: a share of 0 deals every unit right-handed (today's
    picture), 1 deals every unit left-handed.

    BREAKS IF: the rounding deals a left-hander at share 0."""
    units = [_u(i, i % 3 == 0) for i in range(40)]
    assert not any(HandAssigner({"player": 0.0, "zombie": 0.0}).left_handed(units))
    assert all(HandAssigner({"player": 1.0, "zombie": 1.0}).left_handed(units))


def test_the_deal_is_stable_per_unit_and_holds_the_share():
    """PROPERTY: a unit's handedness is dealt once and kept (later units,
    another draw order, turning zombie change nothing), and among the
    first n units of a role ``round(n x share)`` are left-handed.

    BREAKS IF: the deal is redrawn per frame or per call order, or counts
    across roles."""
    share = handedness_table(CFG)
    h = HandAssigner(share)
    first = h.left_handed([_u(i) for i in range(1, 11)])
    again = h.left_handed([_u(i, zombie=True) for i in range(30, 0, -1)])[::-1]
    assert again[:10] == first
    for n in range(1, 30):
        g = HandAssigner({"player": 0.25, "zombie": 0.0})
        got = g.left_handed([_u(i) for i in range(n)] + [_u(100 + i, True) for i in range(n)])
        assert sum(got[:n]) == math.floor(n * 0.25 + 0.5) and not any(got[n:])


def test_the_handedness_door_refuses_a_bad_share():
    """PROPERTY: ``[render.unit_looks] left_handed`` refuses a share
    outside [0, 1], a missing role and an unknown role, by name.

    BREAKS IF: a typo silently deals nobody left-handed."""
    good = vars(CFG.render.unit_looks)
    for bad, needle in (({"player": 1.5, "zombie": 0.1}, "player"),
                        ({"player": 0.1}, "zombie"),
                        ({"player": 0.1, "zombie": 0.1, "dog": 0.1}, "dog")):
        cfg = SimpleNamespace(render=SimpleNamespace(
            unit_looks=SimpleNamespace(**{**good, "left_handed": bad})))
        with pytest.raises(ValueError, match=needle):
            handedness_table(cfg)
