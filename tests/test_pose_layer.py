"""#162 -- the pose layer (renderer/pose_clips.py, pose_layer.py, the
UnitModelRenderer wiring), as properties.

Spec: docs/animation_milestones_design_2026-11-02.md §4.5 ("Tests, as
properties" 1-11; 12 is the GL check, tests/test_pose_inject_gl.py). Every
test names the property it protects and the change that must break it.

Headless: imports pyray (no window) only to read raylib's own baked clips
(``load_model_animations`` needs no GL context) and to import the renderer.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_pose_layer.py -q
"""
from __future__ import annotations

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

import pyray as rl  # noqa: E402

from config import CFG  # noqa: E402
from renderer import pose_layer as pl_mod  # noqa: E402
from renderer.pose_clips import (ClipRow, PoseDataError, build_clip_bank,  # noqa: E402
                                 quat_mul, quat_conj, read_glb)
from renderer.pose_layer import (NO_CLIP, MaskRule, MaskTable, PoseConfigError,  # noqa: E402
                                 PoseLayer, pose_settings, sample_bank)
from renderer.unit_looks import all_look_names, look_path, looks_table  # noqa: E402

from _pose_glb import (CRITTER_BONES, axis_quat, critter_clips,  # noqa: E402
                       critter_settings, write_glb)

SHIPPED = all_look_names(looks_table(CFG))
SETTINGS = pose_settings(CFG)


# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def layer() -> PoseLayer:
    """One pose layer over every shipped look, with the shipped settings."""
    pl = PoseLayer(SETTINGS)
    for name in SHIPPED:
        pl.add_look(name, look_path(name))
    return pl


@pytest.fixture(scope="module")
def critter(tmp_path_factory):
    path = write_glb(tmp_path_factory.mktemp("critter") / "critter.glb",
                     CRITTER_BONES, critter_clips())
    settings = pose_settings(critter_settings())
    pl = PoseLayer(settings)
    pl.add_look("critter", path)
    return pl, path


def _state(pl: PoseLayer, look: str, clips, time=0.0):
    """A PoseState playing ``clips`` (one name per layer, or one for all)."""
    rig = pl.rig(look)
    if isinstance(clips, str):
        clips = [clips] * len(pl.settings.layers)
    ps = pl.new_state(look, pl.settings.fallback_state)
    ps.clip[:] = [rig.bank.index(c) for c in clips]
    ps.time[:] = time
    return ps


def _angle(q1, q2):
    d = np.abs((q1 * q2).sum(-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(d)


def _state_clip(layer_name: str, state: str) -> str:
    return SETTINGS.states[state][layer_name]


WALK = _state_clip(SETTINGS.layers[0], "walk")
IDLE = _state_clip(SETTINGS.layers[0], SETTINGS.fallback_state)


# ---------------------------------------------------------------------------
# (1) sampler fidelity against raylib's own baked frames
# ---------------------------------------------------------------------------

# raylib's glTF loader samples a clip at frame f's time f * 17 ms
# (rmodels.c, GLTF_ANIMDELAY). raylib 5.5's loader keeps ONE interpolation
# per bone -- the LAST channel's that targets it (``boneChannels[k]
# .interpolationType`` in LoadModelAnimationsGLTF) -- and applies it to all
# three of the bone's channels, so a bone whose last channel is STEP holds
# its floor key in every channel. That is a quirk of that loader, NOT what the
# pose layer reproduces (it samples each channel by its own interpolant).
_RAYLIB_DT = 0.017


def _raylib_step_bones(path: Path, clip: str, parents: np.ndarray) -> np.ndarray:
    """Bones raylib 5.5 holds on their floor key in ``clip``: the last channel
    of the bone AND of every ancestor is STEP (or the bone has no channel)."""
    doc, _ = read_glb(path)
    joints = doc["skins"][0]["joints"]
    jpos = {n: i for i, n in enumerate(joints)}
    anim = next(a for a in doc["animations"] if a["name"] == clip)
    last = {}
    for ch in anim["channels"]:
        node = ch["target"]["node"]
        if node in jpos:
            last[jpos[node]] = anim["samplers"][ch["sampler"]].get("interpolation", "LINEAR")
    own = np.array([last.get(j, "STEP") == "STEP" for j in range(len(joints))])
    chain = own.copy()
    for j in range(len(joints)):           # parents precede children in these files
        p = parents[j]
        while p >= 0:
            chain[j] &= own[p]
            p = parents[p]
    return chain


@pytest.mark.parametrize("look", SHIPPED)
def test_sampler_reproduces_raylibs_frames_at_key_times(layer, look):
    """PROPERTY (1, §4.5): our composed GLOBAL pose of every clip of every
    shipped look equals raylib's own ``framePoses`` (model-space TRS) where
    the two sample the same instant: frame 0 of every clip, and -- on the
    raylib 5.5 binding, whose loader holds a bone's floor key when its last
    channel is STEP -- every frame for every bone that holds its floor key,
    compared at that key. Root-motion-stripped clips differ by design and
    are skipped.

    BREAKS IF: the bone order, the parent chain, the quaternion convention
    (xyzw / multiplication order), the inverse-bind / rest handling, the
    global-TRS layout or the key time base (key step, floor key) drifts.
    """
    tol = 5e-5
    rig = layer.rig(look)
    bank = rig.bank
    path = look_path(look)
    count_field = ("frameCount" if "frameCount" in
                   {f[0] for f in rl.ffi.typeof("ModelAnimation").fields} else "keyframeCount")
    poses_field = "framePoses" if count_field == "frameCount" else "keyframePoses"
    quirk_known = poses_field == "framePoses"         # the 5.5 loader above
    n = rl.ffi.new("int *", 0)
    anims = rl.load_model_animations(str(path), n)
    compared_frames = 0
    try:
        for i in range(n[0]):
            a = anims[i]
            name = rl.ffi.string(a.name).decode()
            c = bank.index(name)
            if bank.root_motion[c] != "keep":
                continue
            J = int(a.boneCount)
            frames = int(getattr(a, count_field))
            fposes = getattr(a, poses_field)
            step_bones = _raylib_step_bones(path, name, bank.skeleton.parents)
            check = [0] + (list(range(1, frames)) if quirk_known and step_bones.any() else [])
            times = []
            for f in check:
                k = min(int(np.floor(f * _RAYLIB_DT / bank.key_dt[c] + 1e-4)), bank.n_keys[c] - 1)
                times.append(k * bank.key_dt[c])
            out = layer.compose(look, [_state(layer, look, name, t) for t in times]).global_trs
            for row, f in zip(out, check):
                ref = np.frombuffer(rl.ffi.buffer(fposes[f], J * 40),
                                    dtype=np.float32).reshape(J, 10)
                bones = slice(None) if f == 0 else step_bones
                dt = np.abs(row[bones, :3] - ref[bones, :3]).max()
                dq = np.minimum(np.abs(row[bones, 3:7] - ref[bones, 3:7]),
                                np.abs(row[bones, 3:7] + ref[bones, 3:7])).max()
                ds = np.abs(row[bones, 7:] - ref[bones, 7:]).max()
                assert max(dt, dq, ds) <= tol, (look, name, f, dt, dq, ds)
                compared_frames += 1
    finally:
        rl.unload_model_animations(anims, n[0])
    assert compared_frames > bank.n_clips, "fidelity check compared almost nothing"


# ---------------------------------------------------------------------------
# (2) continuity
# ---------------------------------------------------------------------------

def test_sampling_is_sub_frame_and_continuous_across_keys(layer):
    """PROPERTY (2): the walk clip's sampled local pose moves BETWEEN keys
    (sub-frame, not integer frames) and has no jump at any key boundary:
    the change over +-0.1 ms around each key is tiny, while the pose a
    quarter and three quarters into a key interval differs.

    BREAKS IF: sampling snaps to integer frames (the old player), the key
    index is off by one at a boundary, or the key step is wrong.
    """
    look = SHIPPED[0]
    bank = layer.rig(look).bank
    c = bank.index(WALK)
    dt = bank.key_dt[c]
    ks = np.arange(1, bank.n_keys[c] - 1)
    eps = 1e-4
    t_a, r_a, _ = sample_bank(bank, np.full(len(ks), c), ks * dt - eps)
    t_b, r_b, _ = sample_bank(bank, np.full(len(ks), c), ks * dt + eps)
    assert np.abs(t_a - t_b).max() < 1e-3
    assert _angle(r_a, r_b).max() < 1e-2
    _, r_q1, _ = sample_bank(bank, np.full(len(ks), c), ks * dt + 0.25 * dt)
    _, r_q3, _ = sample_bank(bank, np.full(len(ks), c), ks * dt + 0.75 * dt)
    assert _angle(r_q1, r_q3).max() > 1e-3, "pose does not move inside a key interval"


def test_loop_seam_wraps_and_is_continuous(layer):
    """PROPERTY (2): a looping clip's time wraps -- clip time ``dur + x``
    samples exactly ``x`` -- and, the shipped loop clips being authored
    closed (last key == first key), the pose just before the seam equals the
    pose just after it.

    BREAKS IF: looping clamps or resets instead of wrapping, the wrap uses
    the wrong duration, or the last interval is dropped (off-by-one).
    """
    look = SHIPPED[0]
    bank = layer.rig(look).bank
    for clip in {WALK, IDLE}:
        c = bank.index(clip)
        assert bank.loop[c]
        dur = bank.duration[c]
        xs = np.array([0.01, 0.3 * dur, 0.77 * dur])
        t1, r1, _ = sample_bank(bank, np.full(3, c), xs)
        t2, r2, _ = sample_bank(bank, np.full(3, c), xs + dur)
        assert np.abs(t1 - t2).max() < 1e-5 and _angle(r1, r2).max() < 1e-3
        first, last = bank.offset[c], bank.offset[c] + bank.n_keys[c] - 1
        if _angle(bank.r[first], bank.r[last]).max() < 1e-4:     # closed loop data
            ta, ra, _ = sample_bank(bank, [c], [dur - 1e-4])
            tb, rb, _ = sample_bank(bank, [c], [1e-4])
            assert np.abs(ta - tb).max() < 1e-3 and _angle(ra, rb).max() < 1e-2


def test_interpolation_takes_the_short_arc(tmp_path):
    """PROPERTY (2): two neighbouring keys stored with opposite quaternion
    signs (the same small turn written as q and -q') interpolate along the
    short arc: the mid pose lies between them, not swung through ~360 deg.

    BREAKS IF: the bake's hemisphere-continuity fix or nlerp's sign check is
    removed.
    """
    q0 = np.array(axis_quat([0, 1, 0], 0.1))
    q1 = -np.array(axis_quat([0, 1, 0], 0.3))
    path = write_glb(tmp_path / "flip.glb", CRITTER_BONES[:1] + [dict(CRITTER_BONES[1])],
                     [{"name": "flip", "channels": [(1, "rotation", "LINEAR",
                                                     [0.0, 1.0], [q0, q1])]}])
    bank = build_clip_bank(path)
    _, r, _ = sample_bank(bank, [0], [0.5])
    mid = r[0, 1]
    assert _angle(mid, q0) < 0.2 + 1e-3 and _angle(mid, -q1) < 0.2 + 1e-3


# ---------------------------------------------------------------------------
# (3) masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("look", SHIPPED)
def test_shipped_masks_partition_every_skeleton(layer, look):
    """PROPERTY (3): on every shipped look the configured masks resolve and
    partition the skeleton -- every bone in exactly one mask, every layer
    has a non-empty mask.

    BREAKS IF: a mask rule is mistyped or a bone is left out / doubled.
    """
    masks = layer.rig(look).masks
    assert (masks.weights.sum(0) == 1).all()
    for layer_name in SETTINGS.layers:
        assert masks.weights[masks.index(layer_name)].sum() > 0


def _rules(**over):
    base = {"hind": MaskRule(("pelvis_x",), ("back_x",)),
            "front": MaskRule(("back_x",), ("skull_x",)),
            "top": MaskRule(("skull_x",))}
    base.update(over)
    return base


@pytest.mark.parametrize("rules, needle", [
    (_rules(top=MaskRule(("skul_x",))), "skul_x"),                       # typo
    (_rules(front=MaskRule(("back_x",))), "more than one mask"),          # overlap
    (_rules(top=MaskRule(("skull_x",), ("skull_x",))), "no mask"),        # gap
])
def test_masks_refuse_unknown_bones_overlap_and_gaps(critter, rules, needle):
    """PROPERTY (3): a mask rule naming a bone the skeleton lacks, two masks
    sharing a bone, or a bone in no mask is refused with the reason named.

    BREAKS IF: unknown names are skipped silently or the partition check is
    dropped (a typo'd list would animate some bones from the wrong layer).
    """
    pl, _ = critter
    with pytest.raises(PoseConfigError, match=re.escape(needle)):
        MaskTable.from_rules(pl.rig("critter").bank.skeleton, rules)


def test_a_mask_that_drives_no_layer_is_refused_at_load(critter):
    """PROPERTY (3): when the masks partition the skeleton but one of them
    is not a configured layer, its bones would have no clip -- the look is
    refused at load, naming the mask.

    BREAKS IF: such bones silently take some other layer's (or no) pose.
    """
    _, path = critter
    d = critter_settings()
    masks = d["render"]["pose"]["skeletons"]["critter"]["masks"]
    masks["hind"] = {"subtree": ["pelvis_x"], "minus": ["back_x", "hindleg_l"]}
    masks["kicker"] = {"subtree": ["hindleg_l"]}
    with pytest.raises(PoseConfigError, match="kicker"):
        PoseLayer(pose_settings(d)).add_look("critter", path)


# ---------------------------------------------------------------------------
# (4) layer independence, (5) inheritance
# ---------------------------------------------------------------------------

def test_layers_are_independent(layer):
    """PROPERTY (4): changing ONLY the upper layer's clip leaves every bone
    owned by the other layers unchanged in local space, and the bones of the
    first (root-holding) layer unchanged in model space too.

    BREAKS IF: layers blend across their masks, the mask selection picks the
    wrong layer, or a layer's clip leaks into another's bones.
    """
    look = SHIPPED[0]
    rig = layer.rig(look)
    L = SETTINGS.layers
    other = next(n for n in rig.bank.names if n not in (WALK, IDLE))
    a = _state(layer, look, [WALK if l != L[1] else IDLE for l in L], 0.4)
    b = _state(layer, look, [WALK if l != L[1] else other for l in L], 0.4)
    out = layer.compose(look, [a, b])
    upper = rig.masks.weights[rig.masks.index(L[1])] > 0
    legs = rig.masks.weights[rig.masks.index(L[0])] > 0
    assert np.abs(out.local_t[0, ~upper] - out.local_t[1, ~upper]).max() == 0.0
    assert np.abs(out.local_r[0, ~upper] - out.local_r[1, ~upper]).max() == 0.0
    assert np.abs(out.global_trs[0, legs] - out.global_trs[1, legs]).max() < 1e-6
    assert _angle(out.local_r[0, upper], out.local_r[1, upper]).max() > 1e-3


def test_upper_body_inherits_the_legs_motion(layer):
    """PROPERTY (5): with the upper layer's clip and time held, changing the
    legs layer's clip moves upper bones in MODEL space (they ride the hips),
    while their local pose is unchanged.

    BREAKS IF: layers are composed in global space (each bone's model-space
    pose copied from its own layer), cutting the parent chain.
    """
    look = SHIPPED[0]
    rig = layer.rig(look)
    L = SETTINGS.layers
    a = _state(layer, look, [IDLE if l == L[0] else WALK for l in L], 0.3)
    b = _state(layer, look, [WALK] * len(L), 0.3)
    b.time[0] = 0.9
    out = layer.compose(look, [a, b])
    upper = rig.masks.weights[rig.masks.index(L[1])] > 0
    assert np.abs(out.local_r[0, upper] - out.local_r[1, upper]).max() == 0.0
    assert np.abs(out.global_trs[0, upper, :3] - out.global_trs[1, upper, :3]).max() > 1e-3


# ---------------------------------------------------------------------------
# (6) crossfade
# ---------------------------------------------------------------------------

def test_crossfade_runs_from_source_to_target_monotonically(layer):
    """PROPERTY (6): a crossfade at weight 0 is the source pose, at weight 1
    the target pose, and the angle to the target never grows as the weight
    rises.

    BREAKS IF: the ramp's ends are swapped, the blend overshoots, or a
    quaternion sign flip sends it the long way round.
    """
    look = SHIPPED[0]
    source = _state(layer, look, IDLE, 0.7)
    target = _state(layer, look, WALK, 0.2)
    ref = layer.compose(look, [source, target])
    rig = layer.rig(look)
    weights = np.linspace(0.0, 1.0, 21)
    states = []
    for w in weights:
        ps = _state(layer, look, WALK, 0.2)
        ps.prev_clip[:] = rig.bank.index(IDLE)
        ps.prev_time[:] = 0.7
        ps.fade[:] = w
        states.append(ps)
    out = layer.compose(look, states)
    # (q and -q are one rotation: compare by angle)
    assert _angle(out.local_r[0], ref.local_r[0]).max() < 1e-3
    assert np.abs(out.local_t[0] - ref.local_t[0]).max() < 1e-6
    assert _angle(out.local_r[-1], ref.local_r[1]).max() < 1e-3
    assert np.abs(out.local_t[-1] - ref.local_t[1]).max() < 1e-6
    assert np.abs(out.global_trs[-1, :, :3] - ref.global_trs[1, :, :3]).max() < 1e-5
    ang = _angle(out.local_r.astype(np.float64),
                 ref.local_r[1][None].astype(np.float64))      # (W, J)
    # 1e-3 rad: arccos of a float32 dot near 1 resolves ~5e-4 rad, so a bone
    # that does not move between the clips reads as jitter around 0
    assert (np.diff(ang, axis=0) <= 1e-3).all()
    moving = ang[0] > 0.05
    assert (np.diff(ang[:, moving], axis=0) <= 1e-4).all()
    assert ang[0].max() > 0.05, "source and target too alike to test"


def test_a_state_change_crossfades_over_crossfade_s(layer):
    """PROPERTY (6): driving a unit from one state to another starts a ramp
    that is under way after half of ``crossfade_s`` and finished (source
    dropped) after it -- and driving the same state again starts nothing.

    BREAKS IF: a change snaps, the ramp ignores ``dt`` or never settles, or
    re-asserting a state restarts its clips.
    """
    look = SHIPPED[0]
    cf = SETTINGS.crossfade_s
    assert cf > 0
    ps = layer.new_state(look, "idle")
    layer.drive(ps, "idle", 0.1)
    layer.drive(ps, "walk", cf / 2)
    assert ((ps.fade > 0.3) & (ps.fade < 0.7)).all() and (ps.prev_clip >= 0).all()
    t_before = ps.time.copy()
    layer.drive(ps, "walk", cf)
    assert (ps.fade == 1.0).all() and (ps.prev_clip == NO_CLIP).all()
    assert (ps.time > t_before).all()


# ---------------------------------------------------------------------------
# (7) additive identity
# ---------------------------------------------------------------------------

def test_additive_at_its_reference_or_weight_zero_changes_nothing(layer):
    """PROPERTY (7, §4.5 correction 3): an additive clip played at its own
    reference frame, or at weight 0, leaves the pose exactly as without it;
    played elsewhere at full weight it changes the pose.

    BREAKS IF: the additive delta is taken against the bind rest (it would
    add the clip's whole stance), the weight is ignored, or the delta is
    applied in the wrong order.
    """
    look = SHIPPED[0]
    bank = layer.rig(look).bank
    # any other clip that moves: the additive's identity is about its own
    # reference frame, whichever clip it is
    add_clip = next(n for n in bank.names if n not in (WALK, IDLE)
                    and bank.duration[bank.index(n)] > 0.2)
    plain = _state(layer, look, WALK, 0.5)
    at_ref = _state(layer, look, WALK, 0.5)
    layer.set_additive(at_ref, add_clip, 1.0, time=0.0)
    zero_w = _state(layer, look, WALK, 0.5)
    layer.set_additive(zero_w, add_clip, 0.0, time=0.2)
    moved = _state(layer, look, WALK, 0.5)
    c = bank.index(add_clip)
    layer.set_additive(moved, add_clip, 1.0, time=0.5 * bank.duration[c])
    out = layer.compose(look, [plain, at_ref, zero_w, moved])
    for k in (1, 2):
        assert np.abs(out.global_trs[k] - out.global_trs[0]).max() < 1e-5
    assert np.abs(out.global_trs[3] - out.global_trs[0]).max() > 1e-3
    # the clip's reference frame is not the bind rest (else this is vacuous)
    ref = bank.offset[c] + bank.additive_ref[c]
    assert _angle(bank.r[ref], bank.skeleton.rest_r).max() > 1e-2


# ---------------------------------------------------------------------------
# (8) root motion
# ---------------------------------------------------------------------------

def test_root_motion_strip_keeps_the_root_in_place(critter):
    """PROPERTY (8, §4.5 correction 6): a clip baked with
    ``root_motion = "strip"`` keeps every root bone's horizontal (x, z)
    model-space translation constant over the whole clip, while its vertical
    bob and the body's motion remain; the same clip kept travels.

    BREAKS IF: the strip zeroes the wrong axes (the vertical), forgets a
    root, or is not applied at bake.
    """
    _, path = critter
    keep = build_clip_bank(path, {"trot": ClipRow(root_motion="keep")})
    strip = build_clip_bank(path, {"trot": ClipRow(root_motion="strip")})
    ts = np.linspace(0.0, 1.0, 41)
    for bank, travels in ((keep, True), (strip, False)):
        c = bank.index("trot")
        pl = PoseLayer(pose_settings(critter_settings()))
        pl.add_look("c", bank=bank)
        states = [_state(pl, "c", "trot", t) for t in ts]
        g = pl.compose("c", states).global_trs
        roots = bank.skeleton.roots
        span = g[:, roots][..., [0, 2]].max(0) - g[:, roots][..., [0, 2]].min(0)
        assert (span.max() > 1.0) == travels
        assert g[:, roots, 1].max() - g[:, roots, 1].min() > 0.05     # the bob stays
        del c


def test_shipped_root_motion_rows_are_stripped(layer):
    """PROPERTY (8): every clip the config marks ``root_motion = "strip"``
    is stripped in every shipped look that has it (its root bones' x, z stay
    put over the clip).

    BREAKS IF: the config rows stop reaching the bake.
    """
    stripped = [n for n, row in SETTINGS.clips.items() if row.root_motion == "strip"]
    seen = 0
    for look in SHIPPED:
        bank = layer.rig(look).bank
        for name in stripped:
            if not bank.has(name):
                continue
            c = bank.index(name)
            rows = slice(bank.offset[c], bank.offset[c] + bank.n_keys[c])
            roots = bank.skeleton.roots
            xz = bank.t[rows][:, roots][..., [0, 2]]
            assert np.abs(xz - xz[0]).max() < 1e-6, (look, name)
            seen += 1
    assert seen > 0


# ---------------------------------------------------------------------------
# (9) skeleton-agnostic
# ---------------------------------------------------------------------------

def test_a_foreign_skeleton_composes_with_its_own_masks(critter):
    """PROPERTY (9): a five-bone skeleton with its own bone names, masks and
    state table composes through the same code: finite poses, each layer
    driving its own bones, the skull riding the back's sway.

    BREAKS IF: anything in the pose layer assumes the shipped rig (its bone
    count, names, depth or a hard-coded mask).
    """
    pl, _ = critter
    a = pl.new_state("critter", "walk")
    pl.drive(a, "walk", 0.3)
    b = pl.new_state("critter", "idle")
    out = pl.compose("critter", [a, b])
    assert np.isfinite(out.global_trs).all() and np.isfinite(out.skin).all()
    assert out.global_trs.shape[1] == len(CRITTER_BONES)
    skull = [b_["name"] for b_ in CRITTER_BONES].index("skull_x")
    assert np.abs(out.global_trs[0, skull, :3] - out.global_trs[1, skull, :3]).max() > 1e-2


def test_the_pure_pose_modules_name_no_bone():
    """PROPERTY (9, §2 rule 1): ``pose_clips.py`` and ``pose_layer.py`` hold
    no bone name of the shipped skeleton as a string literal -- regions and
    states are data (``[render.pose]``), so a different skeleton is a data
    change.

    BREAKS IF: a bone name is hard-coded in either module.
    """
    names = set()
    for look in SHIPPED:
        doc, _ = read_glb(look_path(look))
        names |= {doc["nodes"][j]["name"] for j in doc["skins"][0]["joints"]}
    for mod in ("pose_clips.py", "pose_layer.py", "pose_modifiers.py", "pose_posture.py"):
        src = (ROOT / "renderer" / mod).read_text(encoding="utf-8")
        literals = set(re.findall(r"""["']([^"'\n]+)["']""", src))
        assert not (literals & names), (mod, sorted(literals & names))


# ---------------------------------------------------------------------------
# glb refusals (correction 4)
# ---------------------------------------------------------------------------

def _bones_with(**edit):
    bones = [dict(b) for b in CRITTER_BONES]
    bones[1].update(edit)
    return bones


@pytest.mark.parametrize("kind", ["cubic", "matrix", "two_skins", "armature_moves"])
def test_the_reader_refuses_what_it_does_not_implement(tmp_path, kind):
    """PROPERTY (§4.5 correction 4): a CUBICSPLINE sampler, a ``matrix``
    joint node, more than one skin, or a transform above the skin's roots is
    refused by name (a foreign skeleton fails loudly instead of animating
    wrongly).

    BREAKS IF: any of those is silently read (e.g. a cubic sampler treated
    as linear, its tangents read as keys).
    """
    bones, clips, kw = CRITTER_BONES, critter_clips(), {}
    if kind == "cubic":
        q = axis_quat([0, 0, 1], 0.2)
        clips = [{"name": "c", "channels": [(1, "rotation", "CUBICSPLINE", [0.0, 1.0],
                                             [q] * 6)]}]
        needle = "CUBICSPLINE"
    elif kind == "matrix":
        bones = _bones_with(matrix=list(np.eye(4).ravel()))
        needle = "matrix"
    elif kind == "two_skins":
        kw["extra_skin"] = True
        needle = "2 skins"
    else:
        kw["armature"] = {"scale": [0.01, 0.01, 0.01]}
        needle = "transform"
    path = write_glb(tmp_path / f"{kind}.glb", bones, clips, **kw)
    with pytest.raises(PoseDataError, match=needle):
        build_clip_bank(path)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def _broken(edit):
    d = critter_settings()
    edit(d["render"]["pose"])
    return d


@pytest.mark.parametrize("edit, needle", [
    (lambda p: p.pop("rate"), "rate"),
    (lambda p: p.update(rate="fast"), "rate"),
    (lambda p: p.update(enabled=1), "enabled"),
    (lambda p: p.update(surprise=3), "surprise"),
    (lambda p: p["states"]["walk"].pop("top"), "walk"),
    (lambda p: p.update(fallback_state="run"), "fallback_state"),
    (lambda p: p["clips"].update(trot={"root_motion": "drop"}), "root_motion"),
    (lambda p: p.update(layers=["hind", "front", "tail"]), "tail"),
])
def test_settings_refuse_missing_unknown_and_mistyped_keys(edit, needle):
    """PROPERTY: ``[render.pose]`` is refused, naming the key, when a key is
    missing, unknown or of the wrong type, a state lacks a layer's clip, the
    fallback state does not exist, a clip row's mode is unknown or a layer
    has no mask.

    BREAKS IF: a setting falls back to a silent default (a typo would pick
    the wrong animation path without a word).
    """
    with pytest.raises(PoseConfigError, match=needle):
        pose_settings(_broken(edit))


def test_shipped_config_is_valid_and_leaves_the_layer_off():
    """PROPERTY (constraint 2 of #162): the shipped ``[render.pose]`` table
    parses, and the layer is OFF -- turning it on is Erik's HUMAN-TEST call.

    BREAKS IF: the default is flipped on, or the shipped table stops parsing.
    """
    assert SETTINGS.enabled is False
    assert set(SETTINGS.states) >= {"idle", "walk"}


def test_the_default_rate_keeps_todays_clip_speed(layer):
    """PROPERTY (§4.5 correction 1): at the configured ``rate`` a looping
    clip's period in wall seconds equals the old player's to within one of
    its frames. The old player stepped ``_ANIM_FPS`` raylib frames per wall
    second over frames baked every 17 ms, i.e. it ran every clip at
    ``_ANIM_FPS x 0.017`` of its authored speed.

    BREAKS IF: the rate default changes (a deliberate feel change: Erik's
    HUMAN-TEST ruling), or the time base doubles every speed (rate 1).
    """
    from renderer import unit_model_renderer as umr
    look = SHIPPED[0]
    bank = layer.rig(look).bank
    for clip in (WALK, IDLE):
        c = bank.index(clip)
        old_frames = int(bank.duration[c] * 1000.0 / 17.0) + 1
        old_period = old_frames / umr._ANIM_FPS
        new_period = bank.duration[c] / (SETTINGS.rate * bank.rate[c])
        assert abs(new_period - old_period) <= 1.0 / umr._ANIM_FPS + 1e-6, (clip, old_period, new_period)


# ---------------------------------------------------------------------------
# the renderer wiring: (10) render-only, (11) off path, events
# ---------------------------------------------------------------------------

class _FrozenUnit:
    """A unit stub that raises on ANY attribute write."""

    def __init__(self, uid, x, y, facing=0.0, path=None, zombie=False):
        object.__setattr__(self, "_d", dict(id=uid, x=x, y=y, facing=facing,
                                            move_path=path or [], alive=True,
                                            footprint=3, is_zombie=zombie))

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AssertionError(f"the renderer wrote {name!r} on a unit")

    def __delattr__(self, name):
        raise AssertionError(f"the renderer deleted {name!r} on a unit")


class _FakeSkinner:
    def __init__(self):
        self.prepared, self.applied = [], []

    def prepare(self, look, model, template_anim, bone_names):
        self.prepared.append((look, tuple(bone_names)))

    def apply(self, look, model, pose_row):
        self.applied.append((look, model, np.array(pose_row)))

    def release(self):
        pass


class _FakeRl:
    """The bits of pyray the draw path touches, recording skinning calls."""

    def __init__(self):
        self.updates = []

    def update_model_animation(self, model, anim, frame):
        self.updates.append((model, anim, frame))

    def __getattr__(self, name):          # begin/end_mode_3d, draw_*, Vector3, Color
        return lambda *a, **k: None


def _renderer(monkeypatch, enabled: bool, skinner=None):
    from renderer import unit_model_renderer as umr
    from renderer.unit_looks import LookAssigner
    monkeypatch.setattr(CFG.render.pose, "enabled", enabled)
    r = umr.UnitModelRenderer()
    table = looks_table(CFG)
    for name in SHIPPED:
        model = SimpleNamespace(boneCount=53, name=name)
        anims = [SimpleNamespace(tag=(name, k), frameCount=40) for k in range(4)]
        r._looks[name] = umr._Look(name=name, path=look_path(name), model=model,
                                   anims=anims, n_anims=4,
                                   clip_index={IDLE: 0, WALK: 1}, native_height=1.8)
        r._resolve[name] = r._looks[name]
    r._assigner = LookAssigner(table)
    r._loaded = True
    r._setup_pose(skinner=skinner)
    return r


def _draw(r, units, clock):
    r.draw_units(units, 8.0, clock, camera3d=None, light_view=None)


def test_off_path_builds_nothing_and_plays_the_looks_own_clip(monkeypatch):
    """PROPERTY (11, §4.5 correction 9): with ``[render.pose] enabled =
    false`` the pose layer is never constructed and every unit is skinned
    by ``update_model_animation(look.model, <one of the look's OWN clips>,
    <int frame>)`` -- the path the game drew before the pose layer; event
    forwarding is a no-op.

    BREAKS IF: the off path builds or consults the pose layer, or hands
    raylib anything but the look's own clip at an integer frame.
    """
    from renderer import unit_model_renderer as umr

    def _never(*a, **k):
        raise AssertionError("pose layer constructed with the setting off")
    monkeypatch.setattr(pl_mod.PoseLayer, "__init__", _never)
    fake = _FakeRl()
    monkeypatch.setattr(umr, "rl", fake)
    r = _renderer(monkeypatch, enabled=False, skinner=_FakeSkinner())
    assert r._pose is None and r._skinner is None
    units = [_FrozenUnit(1, 4, 4), _FrozenUnit(2, 9, 4, path=[(1, 1)], zombie=True)]
    for k in range(3):
        _draw(r, units, 1.0 + k / 30)
    r.queue_events([SimpleNamespace(unit_id=1)])
    assert len(fake.updates) == 3 * len(units)
    own = {id(a) for lk in r._looks.values() for a in lk.anims}
    for model, anim, frame in fake.updates:
        assert id(anim) in own and type(frame) is int


def test_on_path_is_render_only_and_batched_per_look(monkeypatch):
    """PROPERTY (10): with the pose layer on, drawing reads units and never
    writes them (a frozen unit raises on any write), the pose reaches each
    unit through the skinner as a ``(J, 10)`` row of finite numbers, raylib's
    own clip player is not called, and the layer's state lives in the
    renderer, keyed by unit id.

    BREAKS IF: any animation state is written to a unit, the draw bypasses
    the skinner, or the skinner gets a malformed row.
    """
    from renderer import unit_model_renderer as umr
    fake = _FakeRl()
    monkeypatch.setattr(umr, "rl", fake)
    sk = _FakeSkinner()
    r = _renderer(monkeypatch, enabled=True, skinner=sk)
    assert r._pose is not None and {p[0] for p in sk.prepared} == set(SHIPPED)
    units = [_FrozenUnit(i, 3 * i, 5, path=[(0, 0)] if i % 2 else None,
                         zombie=i > 3) for i in range(1, 8)]
    for k in range(4):
        _draw(r, units, 2.0 + k / 30)
    assert fake.updates == []
    assert len(sk.applied) == 4 * len(units)
    for look, _, row in sk.applied:
        assert row.shape == (r._pose.rig(look).bank.skeleton.n_bones, 10)
        assert np.isfinite(row).all()
    assert set(r._anim) == {u.id for u in units}
    assert all(st.pose is not None for st in r._anim.values())


def test_events_queue_per_unit_bounded_and_pruned(monkeypatch):
    """PROPERTY (#162 events): with the pose layer on, each event carrying a
    ``unit_id`` lands in that unit's queue, in order, never more than
    ``event_queue`` long; a unit the renderer has dropped loses its queue
    once stale.

    BREAKS IF: events are routed to the wrong unit, the queue grows without
    bound, or a dead unit's queue lives forever.
    """
    from renderer import unit_model_renderer as umr
    monkeypatch.setattr(umr, "rl", _FakeRl())
    r = _renderer(monkeypatch, enabled=True, skinner=_FakeSkinner())
    cap = r._pose.settings.event_queue
    u = _FrozenUnit(7, 2, 2)
    _draw(r, [u], 1.0)
    evs = [SimpleNamespace(unit_id=7, n=k) for k in range(cap + 3)]
    r.queue_events(evs + [SimpleNamespace(unit_id=8, n=-1), SimpleNamespace(kind="x")])
    got = r._pose.events(7)
    assert len(got) == cap and [e.n for e in got] == [e.n for e in evs[-cap:]]
    assert [e.n for e in r._pose.events(8)] == [-1]
    _draw(r, [], 1.0 + 5 * umr._STALE_SECONDS)
    assert r._pose.events(7) == () and r._pose.events(8) == ()
