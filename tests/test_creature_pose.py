"""#106 P7 -- the octopus model's pose FROM the simulated chain
(``renderer/creature_pose.py``) and its raylib half's math
(``renderer/creature_model.py``). Pure: no window, no engine.

Every test names the property it protects and the change that must break it.
Random chains are smooth curves built in the WORLD frame (grid axes, the
facing convention) around a crown at a random heading, exactly as
``read_chain`` hands them over.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_creature_pose.py -q
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from renderer import creature_model as CM  # noqa: E402
from renderer import creature_pose as CP  # noqa: E402
from simulation.creatures import get_creature  # noqa: E402

OM = CP.octopus_math()
ARM = sys.modules["arm_math"]
CDEF = get_creature("octopus")
RIG = CP.build_rig(CDEF)
J, NJ = RIG.junction, RIG.joints


def _world_from_model(m, crown, heading):
    """The inverse of ``creature_pose.to_model`` (test-side, for building
    chains in the model frame)."""
    c, s = math.cos(heading), math.sin(heading)
    X, Y, Z = m[..., 0], m[..., 1], m[..., 2]
    out = np.empty(m.shape)
    out[..., 0] = crown[0] + Z * c - X * s
    out[..., 1] = crown[1] - Z * s - X * c
    out[..., 2] = crown[2] + Y
    return out


def _chain(rng, *, bend=0.6, lift=0.0, stretch=(1.0, 1.25), heading=None, crown=None):
    """A smooth random chain of the row's shape: each arm starts on the
    crown rim at its row angle and turns by a smooth random curvature (and a
    small random pitch), segment lengths the row's times a random stretch.
    Returns a ChainView-like namespace (what ``pose_from_chain`` reads)."""
    b = CDEF.body
    A, n = int(b.arms), int(b.nodes_per_arm)
    heading = rng.uniform(-math.pi, math.pi) if heading is None else heading
    crown = np.array([rng.uniform(5, 15), rng.uniform(5, 15), 0.0]) if crown is None else crown
    seg = b.arm_length_m / (n - 1)
    nodes = np.zeros((A, n, 3))
    for a in range(A):
        phi = heading + math.radians(b.arm_angles_deg[a])
        k = np.linspace(0, 1, n - 1)
        turn = bend * (rng.uniform(-1, 1) * np.sin(math.pi * k * rng.uniform(0.5, 2.0))
                       + rng.uniform(-1, 1) * k)
        ang = phi + np.cumsum(turn / (n - 1) * 3.0)
        pitch = lift * np.sin(math.pi * k) * rng.uniform(-1, 1)
        ls = seg * rng.uniform(*stretch)
        p = np.array([crown[0] + b.body_radius_m * math.cos(phi),
                      crown[1] - b.body_radius_m * math.sin(phi), 0.0])
        nodes[a, 0] = p
        for i in range(n - 1):
            d = np.array([math.cos(ang[i]) * math.cos(pitch[i]),
                          -math.sin(ang[i]) * math.cos(pitch[i]),
                          math.sin(pitch[i])])
            p = p + ls * d
            nodes[a, i + 1] = p
    return SimpleNamespace(nodes=nodes.astype(np.float32).astype(float), crown=crown,
                           heading=float(heading))


def _dist_to_polyline(p, poly):
    a, b = poly[:-1], poly[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * d - p, axis=-1).min())


def _arm(pose, k):
    o = RIG.offsets[k]
    return pose.points[o:o + NJ], pose.rotations[o:o + NJ], pose.scales[o:o + NJ]


WIGGLE = CP.CreatureRenderSettings(0.0, 1.0, 0.3, 2.0, 0.5, 1.0, 0)


# ---------------------------------------------------------------------------
# The rig is octopus_math's, read once
# ---------------------------------------------------------------------------
def test_rig_tables_are_octopus_maths_at_the_matched_size():
    """PROPERTY: the rig's segment lengths and joint scales are exactly
    ``octopus_math``'s (summed to game joints) at the size the door picked,
    for any stretch -- the pose module holds no second copy of the rig.

    BREAKS IF: the linear tables drift from ``arm_segment_lengths`` /
    ``arm_joint_scales`` (a re-derived ramp), or the rig is read at another
    size than the door's."""
    prev = OM.SIZE
    try:
        OM.set_size(RIG.size)
        for s in (0.0, 0.25, 0.7, 1.0):
            seg = (OM.arm_segment_lengths(s) * OM.OCTOPUS_SCALE).reshape(-1, OM.GAME_EVERY).sum(1)
            assert np.allclose(RIG.seg_a + s * RIG.seg_b, seg, rtol=1e-12, atol=0)
            sc = OM.arm_joint_scales(s)[::OM.GAME_EVERY]
            assert np.allclose(1.0 / np.sqrt(1.0 + s * RIG.scale_c), sc, rtol=1e-12, atol=0)
        assert RIG.bone_names == tuple(OM.bones("game"))
    finally:
        OM.set_size(prev)


# ---------------------------------------------------------------------------
# The size door
# ---------------------------------------------------------------------------
def test_the_size_door_matches_the_row_and_refuses_a_mismatch():
    """PROPERTY: the shipped octopus row (2.52 m arm) is the art's mid size;
    an arm 2 % off any art size is REFUSED, by the door and by build_rig.

    BREAKS IF: the tolerance is loosened past 1 %, the door picks a size
    without checking it, or the creature row's arm length moves away from
    every art size (then the drawn animal would not be the simulated one)."""
    assert CP.match_size(CDEF.body.arm_length_m) == "mid"
    assert CP.match_size(1.80) == "small_preview"
    for bad in (2.52 * 1.02, 2.0, 5.0):
        with pytest.raises(ValueError, match="size door"):
            CP.match_size(bad)
    wrong = replace(CDEF, body=replace(CDEF.body, arm_length_m=2.52 * 1.02))
    with pytest.raises(ValueError, match="size door"):
        CP.build_rig(wrong)


# ---------------------------------------------------------------------------
# Lengths and stations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_every_rig_segment_keeps_its_length(seed):
    """PROPERTY: for random chains, every arm segment of the pose (joint g
    -> g+1) is exactly the rig's length at the arm's solved stretch (1e-6
    relative) -- with and without the render wiggle, which only rotates
    segments.

    BREAKS IF: a station is placed by arc length instead of a chord, the
    wiggle translates instead of rotating, or the stretch is applied to the
    crown joints."""
    rng = np.random.default_rng(seed)
    for dials in (CP.NO_WIGGLE, WIGGLE):
        v = _chain(rng, lift=0.3)
        pose = CP.pose_from_chain(v, RIG, dials, t_s=rng.uniform(0, 10))
        for k in range(8):
            pts, _, _ = _arm(pose, k)
            got = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            want = RIG.seg_a + pose.stretch[k] * RIG.seg_b
            assert np.allclose(got, want, rtol=1e-6, atol=0), (k, got - want)


@pytest.mark.parametrize("seed", range(6))
def test_every_free_station_lies_on_the_chain(seed):
    """PROPERTY: with the lift and the wiggle off, every free-arm station
    (joint J+1 .. tip) lies on the arm's track -- the junction, then the
    chain's nodes from ``track_k0`` -- within 1 mm, and every station past
    the track's first leg lies on the CHAIN's own polyline within 1 mm; the
    tip lands within 5 mm of the chain's tip when the stretch is in range.

    BREAKS IF: the walk skips the corner a chord cuts (a station off the
    curve), the secant correction is dropped (the tip overshoots by the
    chord deficit, ~1-5 cm), or the arm order between the chain and the rig
    is wrong (stations on another arm's chain)."""
    rng = np.random.default_rng(100 + seed)
    v = _chain(rng, lift=0.3)
    pose = CP.pose_from_chain(v, RIG, CP.NO_WIGGLE)
    nodes_m = CP.to_model(v.nodes, v.crown, v.heading)
    for a in range(8):
        k = RIG.arm_map[a]
        pts, _, _ = _arm(pose, k)
        track = pose.track[k]
        chain = nodes_m[a]
        assert np.allclose(track[1:], chain[RIG.track_k0:])
        first_leg = np.linalg.norm(track[1] - track[0])
        for g in range(J + 1, NJ):
            if not pose.on_track[k, g]:
                continue
            assert _dist_to_polyline(pts[g], track) < 1e-3, (a, g)
            if np.linalg.norm(pts[g] - track[0]) > first_leg + 1e-9:
                assert _dist_to_polyline(pts[g], chain) < 1e-3, (a, g)
        if 0.0 < pose.stretch[k] < 1.0:
            assert pose.on_track[k].all(), (a, pose.on_track[k])
            assert np.linalg.norm(pts[-1] - chain[-1]) < 5e-3, a


def test_a_chain_on_the_bind_arm_gives_the_bind_pose():
    """PROPERTY: a chain lying straight along each art arm's bind direction
    from the junction, as long as the art's free arm at rest, poses EXACTLY
    the bind pose: bind heads, identity rotations, unit scales.

    BREAKS IF: the model frame's axes or handedness are wrong (the crown
    pose -> model transform), the junction is not the track's start, the
    minimal rotation is not the identity on aligned vectors, or the stretch
    solve is off at rest."""
    rng = np.random.default_rng(7)
    heading = rng.uniform(-math.pi, math.pi)
    crown = np.array([9.0, 6.0, 0.0])
    n = int(CDEF.body.nodes_per_arm)
    m = n - RIG.track_k0                       # chain nodes on the track
    nodes_m = np.zeros((8, n, 3))
    for a in range(8):
        k = RIG.arm_map[a]
        junc = RIG.crown_pts[k, J]
        s = np.linspace(0.0, RIG.free_rest, m + 1)[1:]
        nodes_m[a, RIG.track_k0:] = junc + s[:, None] * RIG.bind_dir[k]
        nodes_m[a, :RIG.track_k0] = junc
    v = SimpleNamespace(nodes=_world_from_model(nodes_m, crown, heading), crown=crown,
                        heading=heading)
    pose = CP.pose_from_chain(v, RIG, CP.NO_WIGGLE)
    o = RIG.offsets[0]
    assert np.abs(pose.stretch).max() < 1e-9
    assert np.allclose(pose.points[o:], RIG.bind_heads[o:], atol=1e-9)
    assert np.allclose(pose.rotations[o:], np.eye(3), atol=1e-9)
    assert np.allclose(pose.scales, 1.0, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_a_mirrored_chain_gives_the_exactly_mirrored_pose(seed):
    """PROPERTY: mirroring the chain across the animal's long axis (each arm
    swapped with its row mirror, the body-frame y negated) mirrors the pose
    of every arm bone: L_n's heads, rotations and scales become R_n's under
    X -> -X (S R S for the rotations) -- wiggle on included.

    BREAKS IF: the arm map pairs the wrong arms, a sign keys on the arm
    index instead of its side (twist, tip curl), or the tip curl's phase
    differs within a mirror pair."""
    rng = np.random.default_rng(200 + seed)
    v = _chain(rng, lift=0.3, heading=0.0, crown=np.zeros(3))
    ang = np.asarray(CDEF.body.arm_angles_deg, float)
    mirror_of = [int(np.argmin(np.abs(ang + a))) for a in ang]
    mn = v.nodes[mirror_of].copy()
    mn[..., 1] = -mn[..., 1]
    vm = SimpleNamespace(nodes=mn, crown=np.zeros(3), heading=0.0)
    S = np.diag([-1.0, 1.0, 1.0])
    for dials in (CP.NO_WIGGLE, WIGGLE):
        t = 1.7
        p1 = CP.pose_from_chain(v, RIG, dials, t)
        p2 = CP.pose_from_chain(vm, RIG, dials, t)
        for k in range(8):
            km = (k + 4) % 8
            a1, r1, s1 = _arm(p1, k)
            a2, r2, s2 = _arm(p2, km)
            assert np.allclose(a2, a1 @ S, atol=1e-9), k
            assert np.allclose(r2, S @ r1 @ S, atol=1e-9), k
            assert np.allclose(s2, s1, atol=1e-12), k


# ---------------------------------------------------------------------------
# Stretch and thinning
# ---------------------------------------------------------------------------
def test_stretch_factor_one_is_scale_one_and_stretch_max_thins_by_the_rule():
    """PROPERTY: an arm at its rest length draws at scale 1 everywhere; at the
    row's ``stretch_max`` the fully ramped joints thin to 1/sqrt(stretch_max)
    (the muscular hydrostat), the crown joints stay 1; and the solved stretch
    is clamped to [0, 1].

    BREAKS IF: the thinning law changes (e.g. 1/stretch), the crown joints
    stretch, or a chain longer than stretch_max lets the art arm grow past
    its rig."""
    s0 = 1.0 / np.sqrt(1.0 + 0.0 * RIG.scale_c)
    assert (s0 == 1.0).all()
    s1 = 1.0 / np.sqrt(1.0 + 1.0 * RIG.scale_c)
    assert (s1[:J + 1] == 1.0).all()
    assert np.isclose(s1[-1], 1.0 / math.sqrt(RIG.stretch_max), rtol=1e-12)
    assert np.all(np.diff(s1) <= 1e-15)
    assert CP.solve_stretch(RIG, RIG.free_rest) == 0.0
    assert CP.solve_stretch(RIG, RIG.free_rest + RIG.free_span) == 1.0
    assert CP.solve_stretch(RIG, 10.0) == 1.0 and CP.solve_stretch(RIG, 0.1) == 0.0
    # through the pose: a very long chain is clamped, the tip runs off the
    # track's end no further than the clamp allows
    rng = np.random.default_rng(3)
    v = _chain(rng, stretch=(1.6, 1.6))
    pose = CP.pose_from_chain(v, RIG, CP.NO_WIGGLE)
    assert (pose.stretch == 1.0).all()
    for k in range(8):
        assert np.allclose(_arm(pose, k)[2], s1)


# ---------------------------------------------------------------------------
# Roll
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_roll_is_carried_by_parallel_transport(seed):
    """PROPERTY: on smooth chains, the rotation between neighbouring arm
    bones is the MINIMAL one between their segment directions -- its angle
    equals the turn between the segments (no roll added, no 180-degree flip
    between neighbours) -- and the first free bone turns the bind direction
    onto its segment.

    BREAKS IF: bone frames are rebuilt per bone from a fixed up vector (a
    Frenet / look-at frame: roll jumps where the arm passes vertical), or
    the transport skips a bone."""
    rng = np.random.default_rng(300 + seed)
    v = _chain(rng, lift=0.4)
    pose = CP.pose_from_chain(v, RIG, CP.NO_WIGGLE)
    for k in range(8):
        pts, rots, _ = _arm(pose, k)
        t = np.diff(pts[J:], axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        assert np.allclose(rots[J] @ RIG.bind_dir[k], t[0], atol=1e-9)
        for g in range(J, NJ - 2):
            rel = rots[g + 1] @ rots[g].T
            ang_rel = math.acos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
            ang_seg = math.acos(np.clip(t[g - J] @ t[g - J + 1], -1.0, 1.0))
            assert abs(ang_rel - ang_seg) < 1e-7, (k, g, ang_rel, ang_seg)
            assert ang_rel < math.pi / 2
            assert np.allclose(rots[g + 1] @ RIG.bind_dir[k], t[g - J + 1], atol=1e-9)


# ---------------------------------------------------------------------------
# The dials
# ---------------------------------------------------------------------------
def test_shipped_creature_dials_are_complete_and_zero_means_off():
    """PROPERTY: ``[render.creature]`` has every dial; with every amplitude,
    the twist and the lift at 0 the pose is exactly the chain's alone
    (``NO_WIGGLE``) whatever the render clock.

    BREAKS IF: a wiggle leaks in at amplitude 0, or the shipped section
    loses a key (the game would raise on the first creature frame)."""
    d = CP.creature_render_settings(CFG)
    off = replace(d, breathing_amplitude=0.0, tip_curl_amplitude=0.0, twist=0.0,
                  arm_lift=0.0)
    v = _chain(np.random.default_rng(11))
    a = CP.pose_from_chain(v, RIG, off, t_s=3.3)
    b = CP.pose_from_chain(v, RIG, CP.NO_WIGGLE, t_s=0.0)
    for x, y in ((a.points, b.points), (a.rotations, b.rotations), (a.scales, b.scales)):
        assert np.array_equal(x, y)


def test_a_missing_or_mistyped_creature_dial_fails_loudly():
    """PROPERTY: the dial reader refuses a missing section, a missing key, a
    mistyped value and an out-of-range one, naming it.

    BREAKS IF: a default is substituted silently (a typo in config.toml
    would then switch an effect off unnoticed)."""
    good = dict(vars(CFG.render.creature))

    def cfg(**over):
        sec = dict(good, **over)
        sec = {k: v for k, v in sec.items() if v is not None}
        return SimpleNamespace(render=SimpleNamespace(creature=SimpleNamespace(**sec)))

    CP.creature_render_settings(cfg())
    with pytest.raises(ValueError, match="no \\[render.creature\\]"):
        CP.creature_render_settings(SimpleNamespace(render=SimpleNamespace()))
    with pytest.raises(ValueError, match="missing 'twist'"):
        CP.creature_render_settings(cfg(twist=None))
    with pytest.raises(ValueError, match="hit_flash_frames"):
        CP.creature_render_settings(cfg(hit_flash_frames=2.5))
    with pytest.raises(ValueError, match="arm_lift"):
        CP.creature_render_settings(cfg(arm_lift="1"))
    with pytest.raises(ValueError, match="breathing_amplitude"):
        CP.creature_render_settings(cfg(breathing_amplitude=1.5))
    with pytest.raises(ValueError, match="tip_curl_period_s"):
        CP.creature_render_settings(cfg(tip_curl_period_s=0.0))


def test_the_breathing_moves_only_the_mantle_walls():
    """PROPERTY: the breathing dial squeezes the mantle walls along
    ``octopus_math``'s own squeeze (its exact pose at that squeeze) and
    moves no arm bone.

    BREAKS IF: the breath is a scale of the whole body, or reaches the arms."""
    v = _chain(np.random.default_rng(12))
    d = replace(CP.NO_WIGGLE, breathing_amplitude=1.0, breathing_period_s=4.0)
    a = CP.pose_from_chain(v, RIG, d, t_s=2.0)        # the breath's peak: squeeze 1
    b = CP.pose_from_chain(v, RIG, CP.NO_WIGGLE)
    assert np.allclose(a.points[:6], RIG.body_pts[1], atol=1e-12)
    assert np.array_equal(a.points[6:], b.points[6:])
    prev = OM.SIZE
    try:
        OM.set_size(RIG.size)
        ref, _ = OM.pose_octopus(OM.BodyParams(mantle_squeeze=1.0), None, "game")
    finally:
        OM.set_size(prev)
    assert np.allclose(RIG.body_pts[1], ref[:6], atol=1e-12)


def test_the_pose_never_writes_into_the_view():
    """PROPERTY: the pose is a pure read of its view (render never writes
    back). BREAKS IF: an in-place op lands on ``view.nodes``."""
    v = _chain(np.random.default_rng(13))
    before = v.nodes.copy(), v.crown.copy()
    CP.pose_from_chain(v, RIG, WIGGLE, 1.0)
    assert np.array_equal(v.nodes, before[0]) and np.array_equal(v.crown, before[1])


# ---------------------------------------------------------------------------
# The raylib half's math (creature_model)
# ---------------------------------------------------------------------------
def _random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    -1).reshape(n, 3, 3)


def test_the_vectorised_frame_write_is_the_prototypes_live_octopus():
    """PROPERTY: ``frame_transforms`` writes, bone for bone, what the art
    prototype's ``LiveOctopus.write_pose`` writes (the formula its 11.4 um
    raylib parity was measured with): rotation = arm_math.quaternion(delta)
    x bind (up to the quaternion's sign), translation pre-compensated
    ``p / s + (s - 1) R b`` where s != 1, scale s.

    BREAKS IF: the Hamilton product's order flips, the compensation is
    dropped (a stretched arm thrown ~0.2 m), or the quaternion branches
    disagree with arm_math's (a > 180 degree rotation flipped)."""
    rng = np.random.default_rng(5)
    n = 60
    R = _random_rotations(rng, n)
    R[:4] = np.diag([1.0, -1.0, -1.0])            # the trace < 0 branches
    R[4] = np.diag([-1.0, 1.0, -1.0])
    R[5] = np.diag([-1.0, -1.0, 1.0])
    p = rng.normal(size=(n, 3))
    s = np.where(rng.random(n) < 0.5, 1.0, rng.uniform(0.8, 1.0, n))
    b = rng.normal(size=(n, 3))
    bq = rng.normal(size=(n, 4))
    bq /= np.linalg.norm(bq, axis=1, keepdims=True)
    rows = CM.frame_transforms(p, R, s, b, bq)
    for j in range(n):
        x, y, z, w = ARM.quaternion(R[j])
        bx, by, bz, bw = bq[j]
        q = np.array([w * bx + x * bw + y * bz - z * by, w * by - x * bz + y * bw + z * bx,
                      w * bz + x * by - y * bx + z * bw, w * bw - x * bx - y * by - z * bz])
        got = rows[j, 3:7].astype(float)
        assert min(np.abs(got - q).max(), np.abs(got + q).max()) < 1e-6, j
        t = p[j] / s[j] + (s[j] - 1.0) * (R[j] @ b[j]) if s[j] != 1.0 else p[j]
        assert np.allclose(rows[j, 0:3], t, atol=1e-5)
        assert np.allclose(rows[j, 7:10], s[j])


def test_the_bind_door_refuses_a_model_of_another_size():
    """PROPERTY: the bind door passes the rig's own bind pose and refuses one
    built at another size (here the art's 'big' row: every head ~35 % out).

    BREAKS IF: the tolerance is dropped or the comparison ignores the arms."""
    assert CM.check_bind_pose(RIG.bind_heads, RIG.bind_heads, "same") == 0.0
    with pytest.raises(ValueError, match="another size"):
        CM.check_bind_pose(RIG.bind_heads * (1.885 / 1.40), RIG.bind_heads, "big")


def test_the_pose_is_cheap():
    """PROPERTY (reported, loose ceiling): one octopus's pose costs well under
    the 3.2 ms the prototype's pure-Python ``pose_octopus`` measured -- the
    brief's target is 1 ms; the gate is 3 ms so a loaded CI box never
    flakes. BREAKS IF: the walk or the transport goes back to per-bone
    Python matrices."""
    import time
    v = _chain(np.random.default_rng(21))
    d = CP.creature_render_settings(CFG)
    for _ in range(5):
        CP.pose_from_chain(v, RIG, d, 0.5)
    t0 = time.perf_counter()
    for i in range(50):
        CP.pose_from_chain(v, RIG, d, 0.5 + i * 0.03)
    ms = (time.perf_counter() - t0) / 50 * 1e3
    assert ms < 3.0, ms


def test_each_chain_arm_is_drawn_by_the_art_arm_beside_it():
    """PROPERTY: on the sim's own rest pose (``creature_system.rest_pose``:
    every arm straight and radial at its row angle), the art arm that draws
    chain arm ``a`` is the one whose junction lies nearest that arm's first
    track node -- and no two chain arms share an art arm. Checked from the
    geometry, not from the map itself.

    BREAKS IF: the arm map is built from indices instead of azimuths (the
    row lists L1..L4 then R4..R1, the rig L1..L4 then R1..R4), or the model
    frame mirrors left and right."""
    from simulation import creature_fixed as CF
    from simulation.creature_system import rest_pose
    heading = 0.7
    centre = (CF.chain_from_m(6.0), CF.chain_from_m(7.0))
    p, _, _ = rest_pose(CDEF, centre, CF.chain_from_m(heading))
    nodes = CF.chain_to_m(p)
    crown = np.array([6.0, 7.0, 0.0])
    m = CP.to_model(nodes, crown, heading)
    junc = RIG.crown_pts[:, J]
    nearest = [int(np.argmin(np.linalg.norm(junc[:, [0, 2]] - m[a, RIG.track_k0, [0, 2]],
                                            axis=1))) for a in range(8)]
    assert nearest == RIG.arm_map.tolist()
    assert sorted(nearest) == list(range(8))
