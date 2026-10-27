"""Whole-octopus prototype: body, crown/web junction, eight arms, per-limb controls.

Pure Python + NumPy (no Blender, no game). Builds on `arm_math` (the one-arm proof):
the hero arm IS `arm_math.geometry()` cut at the junction ring, the taper is
`arm_math.radius`, the skin blend is `arm_math.weights_at`'s two-joint rule and
the runtime quaternion is `arm_math.quaternion`. `arm_math.pose` is untouched,
so the one-arm proof still reproduces bit for bit.

Model frame = glTF / raylib: Y up, +Z FORWARD (the game's model forward at yaw
0), so the animal's LEFT is +X. Ground is y = 0. Everything is authored in
"authored units" (the proof's metres: a 0.71 m body, a 1.2 m arm) and
multiplied by OCTOPUS_SCALE on output. The size and the proportions are DIALS,
one row of the SIZES table (issue #106): `set_size(name)` or the
OCTOPUS_SIZE environment variable picks the row.

Controls (see CONTROLS.md): per arm seven floats, per body six floats, all
bounded. `pose_octopus(body, arms, lod)` returns one absolute head position and
one deformation rotation (relative to the bind pose) per bone, in `bones(lod)`
order; `with_scale=True` adds the per-bone uniform scale the arm `stretch`
control needs (its thinning). Arm k's bones depend ONLY on arm k's params and
the shared body frame (`body_pitch`, `crown_spread`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields

import numpy as np

import arm_math as arm

# Arm instancing and the junction ring -------------------------------------
HERO_RINGS = 129                      # arm_math.geometry(): linspace(0, 1, 129)
JUNCTION_RING = 20                    # hero tube ring where the instanced arm starts
S_J = JUNCTION_RING / (HERO_RINGS - 1)  # 0.15625
X_J = S_J * arm.LENGTH                # 0.1875 m from the mount point
JUNCTION_JOINT = 5                    # X_J == 5 * arm.STEP: the junction ring sits ON hero joint 5
GAME_EVERY = 2                        # game LOD keeps every 2nd hero joint
GAME_SEGMENTS = arm.SEGMENTS // GAME_EVERY

# --------------------------------------------------------------------------
# SIZE AND PROPORTION (#106). One row per size; baby and boss are later rows.
#   body_length_m   world length of the body, front of the head to the back of
#                   the mantle (the authored body is BODY_LENGTH_AUTHORED long)
#   arm_body_ratio  rest arm length (mount point in the crown -> tip, straight)
#                   divided by the body length; the arm GIRTH stays the body's
#                   (the junction ring is part of the crown), so a higher ratio
#                   is a longer, more slender arm, a lower one a shorter, stouter
#   stretch_max     arm length factor at `stretch` = 1 (the free arm beyond the
#                   crown elongates by this factor and thins as 1/sqrt of it)
# Erik's ruling of 2026-10-28 (#106): option A's PROPORTION on bigger bodies --
# every row of the family has arms ARM_BODY_RATIO (1.8) x its body length and
# the same stretch fraction; the fixed 2.0 / 2.5-marine span of the first three
# options is dropped (the ratio rules, the span follows). "mid" (option B's
# 1.4 m body) is the default; "big" (option C's 1.885 m body, "the big baddie")
# previews a later larger size; "small_preview" is option A's 1.0 m body. The
# #106 option rows (mid_bulk / mid / mid_heavy) are in git at 65df56a.
MARINE_HEIGHT_M = 1.885               # the space marine (assets/models/space_marine), every look's scale
BODY_LENGTH_AUTHORED = (0.07 + 0.15) - (-0.215 - 0.275)   # 0.71: head front (HEAD_C+HEAD_R) .. mantle back
ARM_BODY_RATIO = 1.8                  # Erik: "arms about 1.8 x the body" (option A had 1.777)
FAMILY_STRETCH_MAX = 1.40             # free arm x1.40 -> whole arm x1.31: mid's 2.52 m arm reaches 3.31 m


@dataclass(frozen=True)
class SizeRow:
    body_length_m: float
    arm_body_ratio: float
    stretch_max: float
    note: str = ""


SIZES = {
    "mid": SizeRow(1.40, ARM_BODY_RATIO, FAMILY_STRETCH_MAX,
                   "the default, medium size (option B's body, option A's proportion): body 1.4 m long"),
    "small_preview": SizeRow(1.00, ARM_BODY_RATIO, FAMILY_STRETCH_MAX,
                             "option A's body (mantle 0.66 x 0.78 m) at the family proportion"),
    "big": SizeRow(1.885, ARM_BODY_RATIO, FAMILY_STRETCH_MAX,
                   "preview of a later larger size (option C's body, as long as a marine is tall)"),
    "proof": SizeRow(BODY_LENGTH_AUTHORED, arm.LENGTH / BODY_LENGTH_AUTHORED, 1.4,
                     "the pre-#106 animal: authored units at scale 1, the 1.2 m proof arm"),
}
SIZE = os.environ.get("OCTOPUS_SIZE", "mid")

# Arm-length law: per hero segment a length factor, 1 inside the crown and
# ramped in over RAMP_SEGMENTS segments past the junction ring (no kink in the
# taper, no motion at the junction). The SAME ramp carries the rest length
# (ARM_LAMBDA, from the size row: the bind mesh) and the stretch control (pose).
RAMP_SEGMENTS = 3.0
STRETCH_RANGE = (0.0, 1.0)


def _ramp():
    i = np.arange(arm.SEGMENTS, dtype=float)
    x = np.clip((i - (JUNCTION_JOINT + 0.5)) / RAMP_SEGMENTS, 0, 1)
    return x * x * (3 - 2 * x)


RAMP = _ramp()                        # segments 0..5: 0; 6: .074, 7: .5, 8: .926, 9..31: 1


def set_size(name):
    """Make SIZES[name] the active row: rebinds OCTOPUS_SCALE, ARM_LAMBDA, STRETCH_MAX.

    Everything that builds or poses reads these module values at call time, so
    one process can build several sizes in turn (the body-plan renders do).
    """
    global SIZE, OCTOPUS_SCALE, ARM_LAMBDA, STRETCH_MAX, BIND_SEGMENTS, BIND_JOINT_X
    row = SIZES[name]
    SIZE = name
    OCTOPUS_SCALE = row.body_length_m / BODY_LENGTH_AUTHORED
    target = row.arm_body_ratio * BODY_LENGTH_AUTHORED          # rest arm length, authored units
    ARM_LAMBDA = 1 + (target - arm.LENGTH) / (arm.STEP * RAMP.sum())
    if abs(ARM_LAMBDA - 1) < 1e-12:
        ARM_LAMBDA = 1.0                                         # the proof's arm, bit for bit
    if not ARM_LAMBDA > 0.2:
        raise ValueError(f"size {name}: arm_body_ratio {row.arm_body_ratio} too short for the crown")
    STRETCH_MAX = row.stretch_max
    BIND_SEGMENTS = arm.STEP * (1 + (ARM_LAMBDA - 1) * RAMP)     # (32,) rest segment lengths
    BIND_JOINT_X = np.concatenate([[0.0], np.cumsum(BIND_SEGMENTS)])
    return row


set_size(SIZE)


def bind_x(x_mat):
    """Material arm coordinate (the proof's x, 0..1.2) -> rest distance along the arm.

    Piecewise linear with a knot at every hero joint; identity inside the crown
    (x <= 6 * STEP) and for x < 0, so the junction ring and the crown never move.
    """
    x_mat = np.asarray(x_mat, float)
    knots = np.arange(arm.SEGMENTS + 1) * arm.STEP
    return np.where(x_mat <= knots[JUNCTION_JOINT + 1], x_mat, np.interp(x_mat, knots, BIND_JOINT_X))


def material_x(x_bind):
    """Inverse of `bind_x`."""
    x_bind = np.asarray(x_bind, float)
    knots = np.arange(arm.SEGMENTS + 1) * arm.STEP
    return np.where(x_bind <= knots[JUNCTION_JOINT + 1], x_bind, np.interp(x_bind, BIND_JOINT_X, knots))


def arm_length(stretch=0.0):
    """Arm length, mount point to tip, straight, in authored units."""
    return float(arm_segment_lengths(stretch).sum())


def stretch_factor(stretch):
    """Free-arm length factor (relative to the rest arm) at a `stretch` control value."""
    return 1 + stretch * (STRETCH_MAX - 1)


def arm_segment_lengths(stretch=0.0):
    """The 32 hero segment lengths (authored units) of an arm at `stretch`: rest x elongation."""
    return BIND_SEGMENTS * (1 + (stretch_factor(stretch) - 1) * RAMP)


def arm_joint_scales(stretch=0.0):
    """Per hero joint uniform scale: 1/sqrt(elongation of the INCOMING segment).

    Volume-conserving thinning (radius ~ 1/sqrt(length factor), the muscular
    hydrostat). Joint 0..6 are exactly 1 (their incoming segments are not
    stretched), so the junction ring and every crown vertex stay put.
    """
    e = 1 + (stretch_factor(stretch) - 1) * RAMP
    return np.concatenate([[1.0], 1 / np.sqrt(e)])

# Arm crown layout (mirrored). Order = octopus naming, L = animal's left (+X).
ARM_NAMES = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]
_AZ_DEG = {"1": 22.0, "2": 66.0, "3": 108.0, "4": 143.0}
ARM_AZIMUTH = np.radians([(_AZ_DEG[n[1]] if n[0] == "L" else -_AZ_DEG[n[1]]) for n in ARM_NAMES])
ARM_SIDE = np.array([1.0 if n[0] == "L" else -1.0 for n in ARM_NAMES])
MOUNT_RHO = 0.08                      # mount point distance from the crown axis
MOUNT_Y = 0.11                        # arm-axis height at the mount
DROOP = math.radians(3.0)             # bind arms slope gently to the ground

# Control ranges / gains ----------------------------------------------------
A_CURL = 5.35                         # total bend at curl = 1 (the proof's curl)
A_TIP = 3.2                           # extra distal bend at |tip_curl| = 1
TIP_FROM = 0.55
TWIST_MAX = math.radians(100.0)       # total axial twist at |twist| = 1
LIFT_MAX = math.radians(40.0)
ROOT_JOINTS = 6                       # lift/swing/spread bend in over joints 1..6 (s <= 0.19)
SWING_MAX = math.radians(18.0)
SPREAD_UP = math.radians(25.0)        # crown_spread = +1
SPREAD_DOWN = math.radians(80.0)      # crown_spread = -1 (arms close into a bundle)
BODY_PITCH_MAX = math.radians(90.0)
MANTLE_PITCH_MAX = math.radians(20.0)
SQUEEZE_DEPTH = 0.065                 # mantle wall travel at squeeze = 1 (m)
SIPHON_YAW_MAX = math.radians(70.0)
SIPHON_PITCH_MAX = math.radians(45.0)


@dataclass
class ArmParams:
    """One arm's controller output. All defaults = straight bind arm."""
    curl: float = 0.0        # [0, 1]   total bend along the arm, growing toward the tip
    bend_dir: float = 0.0    # [-1, 1]  bend azimuth around the arm axis, in turns of pi:
                             #          0 oral (suckers inward), +0.5 toward the animal's
                             #          front, -0.5 toward its back, +-1 dorsal (periodic)
    twist: float = 0.0       # [-1, 1]  axial twist, total +-100 deg base->tip
    tip_curl: float = 0.0    # [-1, 1]  extra bend on the distal 45 %, same plane, signed
    lift: float = 0.0        # [-1, 1]  base pitch, +-40 deg (+ raises the arm)
    swing: float = 0.0       # [-1, 1]  base yaw in the crown, +-18 deg (+ toward the front)
    stretch: float = 0.0     # [0, 1]   elongation of the free arm: 0 rest length, 1 = the size
                             #          row's stretch_max; thins as 1/sqrt(length factor)

    RANGES = {"curl": (0.0, 1.0), "bend_dir": (-1.0, 1.0), "twist": (-1.0, 1.0),
              "tip_curl": (-1.0, 1.0), "lift": (-1.0, 1.0), "swing": (-1.0, 1.0),
              "stretch": STRETCH_RANGE}


@dataclass
class BodyParams:
    mantle_squeeze: float = 0.0  # [0, 1]   jet contraction of the mantle walls
    mantle_pitch: float = 0.0    # [-1, 1]  mantle tilt about the neck, +-20 deg (+ raises)
    siphon_yaw: float = 0.0      # [-1, 1]  +-70 deg (+ toward the animal's left)
    siphon_pitch: float = 0.0    # [-1, 1]  +-45 deg (+ up)
    crown_spread: float = 0.0    # [-1, 1]  shared arm-base pitch: -1 closes the crown
                                 #          80 deg down into a bundle, +1 opens 25 deg up
    body_pitch: float = 0.0      # [-1, 1]  whole-body pitch about the crown, +-90 deg
                                 #          (-1: oral face forward, mantle leading: jet pose)

    RANGES = {"mantle_squeeze": (0.0, 1.0), "mantle_pitch": (-1.0, 1.0),
              "siphon_yaw": (-1.0, 1.0), "siphon_pitch": (-1.0, 1.0),
              "crown_spread": (-1.0, 1.0), "body_pitch": (-1.0, 1.0)}


def _check(params):
    for f in fields(params):
        lo, hi = params.RANGES[f.name]
        v = getattr(params, f.name)
        if not (math.isfinite(v) and lo - 1e-9 <= v <= hi + 1e-9):
            raise ValueError(f"{type(params).__name__}.{f.name}={v} outside [{lo}, {hi}]")


# Body anatomy (scale 1, metres) -------------------------------------------
HEAD_C, HEAD_R = np.array([0., .20, .07]), np.array([.165, .13, .15])
MANTLE_C, MANTLE_R = np.array([0., .37, -.215]), np.array([.235, .200, .275])
EYE_C = np.array([.115, .268, .088])      # left eye bump centre (mirror for right)
EYE_BUMP_R, EYE_BALL_R = .056, .042
EYE_BALL_OUT = .030                     # eyeball centre, outward from the bump centre
NECK = np.array([0., .30, -.06])          # mantle pivot
CROWN_PIVOT = np.array([0., .15, 0.])     # body_pitch pivot
SIPHON_BASE = np.array([.118, .135, -.045])
SIPHON_DIR = np.array([.45, -.18, .87])   # rest aim: forward, out, slightly down
SIPHON_LEN, SIPHON_R0, SIPHON_R1 = .085, .030, .022
WALL_OFFSETS = {"mantle_squeeze_L": np.array([MANTLE_R[0], 0., 0.]),
                "mantle_squeeze_R": np.array([-MANTLE_R[0], 0., 0.]),
                "mantle_squeeze_D": np.array([0., MANTLE_R[1], 0.])}
WALL_DIR = {"mantle_squeeze_L": np.array([-1., 0., 0.]),
            "mantle_squeeze_R": np.array([1., 0., 0.]),
            "mantle_squeeze_D": np.array([0., -1., 0.])}
BODY_BONES = ["body", "mantle", "mantle_squeeze_L", "mantle_squeeze_R", "mantle_squeeze_D", "siphon"]
BODY_PARENT = {"body": None, "mantle": "body", "mantle_squeeze_L": "mantle",
               "mantle_squeeze_R": "mantle", "mantle_squeeze_D": "mantle", "siphon": "body"}


def arm_joint_count(lod):
    return arm.SEGMENTS + 1 if lod == "hero" else GAME_SEGMENTS + 1


def bones(lod="hero"):
    names = list(BODY_BONES)
    for n in ARM_NAMES:
        names += [f"{n}_{i:02d}" for i in range(arm_joint_count(lod))]
    return names


def bone_parent(name):
    if name in BODY_PARENT:
        return BODY_PARENT[name]
    arm_name, i = name.split("_")
    return "body" if int(i) == 0 else f"{arm_name}_{int(i) - 1:02d}"


def arm_bone_offset(k, lod="hero"):
    return len(BODY_BONES) + k * arm_joint_count(lod)


# Rotation helpers ------------------------------------------------------------
def rot_axis(axis, angle):
    """Rodrigues; axis (..., 3) unit, angle (...) -> (..., 3, 3)."""
    axis = np.asarray(axis, float)
    angle = np.asarray(angle, float)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    m = np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s,
                  y * x * C + z * s, c + y * y * C, y * z * C - x * s,
                  z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)
    return m.reshape(*angle.shape, 3, 3)


def _rx(a):
    return rot_axis(np.array([1., 0., 0.]), a)


def mount_frame(k):
    """Bind root frame of arm k: columns = (along, oral, across) in the model frame.

    along = radial at the arm's azimuth with a small droop, oral = down-ish (the
    suckers face the ground), across = along x oral = the increasing-azimuth
    direction. A proper rotation (det +1) for every arm, left and right.
    """
    phi = ARM_AZIMUTH[k]
    radial = np.array([math.sin(phi), 0., math.cos(phi)])
    along = math.cos(DROOP) * radial + math.sin(DROOP) * np.array([0., -1., 0.])
    across = np.array([math.cos(phi), 0., -math.sin(phi)])
    oral = np.cross(across, along)
    return np.column_stack([along, oral, across])


MOUNT_FRAMES = np.array([mount_frame(k) for k in range(8)])
MOUNT_POINTS = np.array([[MOUNT_RHO * math.sin(p), MOUNT_Y, MOUNT_RHO * math.cos(p)] for p in ARM_AZIMUTH])


def _cumulative_bend(s, p: ArmParams):
    u = np.clip((s - TIP_FROM) / (1 - TIP_FROM), 0, 1)
    return A_CURL * p.curl * s ** 1.45 + A_TIP * p.tip_curl * u * u


def arm_local_curve(p: ArmParams, k: int, root_extra_lift=0.0):
    """Arm k's 33 hero joint frames relative to its own bind root frame.

    Returns (points (33,3) in arm-local metres, frames (33,3,3) local->local).
    Discrete Cosserat-style rod: segment i has length `arm_segment_lengths(
    p.stretch)[i]` exactly (the rest length from the size row, times the
    stretch elongation past the junction); joint i
    turns the material frame by the bend increment about a fixed material axis
    (from bend_dir) after the twist increment about the tangent. Root pitch
    (lift + crown spread) and yaw (swing) act at the mount point. No Frenet
    frame, no normalisation step: frames stay rotations by construction.
    """
    side = ARM_SIDE[k]
    theta = math.pi * p.bend_dir
    # Bend toward d = cos(theta) oral - side*sin(theta) across (front for both sides).
    axis = np.array([0., side * math.sin(theta), math.cos(theta)])
    # axis = along x d, normalised (along = +X local)
    s = np.arange(arm.SEGMENTS + 1) / arm.SEGMENTS
    g = _cumulative_bend(s, p)
    delta = np.diff(g)                                  # joint i = 1..32
    twist_step = -side * p.twist * TWIST_MAX / arm.SEGMENTS
    steps = _rx(np.full(arm.SEGMENTS, twist_step)) @ rot_axis(np.broadcast_to(axis, (arm.SEGMENTS, 3)), delta)
    # Root pitch (lift + crown spread) and yaw (swing) are BENT IN over the root
    # joints 1..ROOT_JOINTS (smooth weights), not pivoted at the mount: a rigid
    # pivot inside the crown folds the crown skin (property check, 2026-10-27).
    lift = p.lift * LIFT_MAX + root_extra_lift
    yaw = side * p.swing * SWING_MAX          # about material -oral (~ model up)
    wr = np.zeros(arm.SEGMENTS)
    u = (np.arange(ROOT_JOINTS) + .5) / ROOT_JOINTS
    wr[:ROOT_JOINTS] = np.sin(np.pi * u) / np.sin(np.pi * u).sum()
    root_steps = rot_axis(np.broadcast_to([0., 0., 1.], (arm.SEGMENTS, 3)), -lift * wr) @         rot_axis(np.broadcast_to([0., -1., 0.], (arm.SEGMENTS, 3)), yaw * wr)
    steps = root_steps @ steps
    frames = np.empty((arm.SEGMENTS + 1, 3, 3))
    frames[0] = np.eye(3)
    for i in range(arm.SEGMENTS):
        frames[i + 1] = frames[i] @ steps[i]
    # The terminal joint repeats the last segment frame (as in the proof).
    frames[-1] = frames[-2]
    seg = arm_segment_lengths(p.stretch)
    points = np.vstack([np.zeros(3), np.cumsum(frames[:-1, :, 0] * seg[:, None], axis=0)])
    return points, frames


def body_root(body: BodyParams):
    a = -body.body_pitch * BODY_PITCH_MAX       # -1 => oral axis points forward
    R = _rx(a)
    t = CROWN_PIVOT - R @ CROWN_PIVOT
    return R, t


def pose_octopus(body: BodyParams = None, arms=None, lod="hero", with_scale=False):
    """All bone head positions and deformation rotations, in `bones(lod)` order.

    Returns (points (n,3), rotations (n,3,3)), scaled by OCTOPUS_SCALE; with
    `with_scale` also scales (n,): each bone's UNIFORM scale about its head
    (1 everywhere except a stretched arm past the crown: `arm_joint_scales`).
    A skin that ignores the scales shows a stretched arm at its rest girth.
    Deterministic pure function of its arguments.
    """
    body = body or BodyParams()
    arms = arms if arms is not None else [ArmParams() for _ in range(8)]
    if len(arms) != 8:
        raise ValueError("exactly eight ArmParams")
    _check(body)
    for a in arms:
        _check(a)
    Rr, tr = body_root(body)
    names = bones(lod)
    pts = np.empty((len(names), 3))
    rots = np.empty((len(names), 3, 3))
    # body
    pts[0], rots[0] = tr, Rr
    # mantle about the neck
    Rm = Rr @ _rx(body.mantle_pitch * MANTLE_PITCH_MAX)
    pts[1], rots[1] = Rr @ NECK + tr, Rm
    for i, w in enumerate(BODY_BONES[2:5], start=2):
        head = MANTLE_C + WALL_OFFSETS[w]
        moved = head + body.mantle_squeeze * SQUEEZE_DEPTH * WALL_DIR[w]
        pts[i] = Rm @ (moved - NECK) + pts[1]
        rots[i] = Rm
    # siphon
    Rs = rot_axis(np.array([0., 1., 0.]), body.siphon_yaw * SIPHON_YAW_MAX) @ \
        rot_axis(_siphon_pitch_axis(), body.siphon_pitch * SIPHON_PITCH_MAX)
    pts[5], rots[5] = Rr @ SIPHON_BASE + tr, Rr @ Rs
    extra = _spread(body)
    every = 1 if lod == "hero" else GAME_EVERY
    scales = np.ones(len(names)) if with_scale else None
    for k, p in enumerate(arms):
        lp, lf = arm_local_curve(p, k, extra)
        M = MOUNT_FRAMES[k]
        W = Rr @ M                         # local -> model, at this pose
        o = arm_bone_offset(k, lod)
        idx = np.arange(0, arm.SEGMENTS + 1, every)
        pts[o:o + len(idx)] = (W @ lp[idx].T).T + Rr @ MOUNT_POINTS[k] + tr
        rots[o:o + len(idx)] = W @ lf[idx] @ M.T     # deformation relative to bind
        if with_scale and p.stretch != 0.0:
            scales[o:o + len(idx)] = arm_joint_scales(p.stretch)[idx]
    if with_scale:
        return pts * OCTOPUS_SCALE, rots, scales
    return pts * OCTOPUS_SCALE, rots


def _spread(body: BodyParams):
    """Shared crown pitch: C1 through 0 (gain blends from down to up)."""
    c = body.crown_spread
    # f(c) = a c + b c|c| ... choose f(1) = up, f(-1) = -down, f'(0) = (up+down)/2.
    a = 0.5 * (SPREAD_UP + SPREAD_DOWN)
    b = 0.5 * (SPREAD_UP - SPREAD_DOWN)
    return a * c + b * c * abs(c)


def _siphon_pitch_axis():
    d = SIPHON_DIR / np.linalg.norm(SIPHON_DIR)
    ax = np.cross(d, [0., 1., 0.])
    return ax / np.linalg.norm(ax)


def bind_pose(lod="hero"):
    """Rest joint positions (n,3) at scale 1 (bind rotations are identity)."""
    pts, _ = pose_octopus(BodyParams(), [ArmParams() for _ in range(8)], lod)
    return pts / OCTOPUS_SCALE


# --------------------------------------------------------------------------
# Linear blend skinning (general, 4 influences), the one reference both the
# Blender parity check and the raylib parity check compare against.
def skin(vertices, joints, weights, bind_points, points, rotations, scales=None):
    """v' = sum_j w_j (s_j R_j (v - b_j) + p_j); `scales` None means all 1."""
    local = vertices[:, None, :] * OCTOPUS_SCALE - bind_points[joints] * OCTOPUS_SCALE
    moved = np.einsum("nkij,nkj->nki", rotations[joints], local)
    if scales is not None:
        moved = moved * scales[joints][..., None]
    moved = moved + points[joints]
    return (moved * weights[..., None]).sum(axis=1)


def span_m(stretch=0.0):
    """Largest tip-to-tip distance (world metres) with all eight arms straight at `stretch`."""
    pts, _ = pose_octopus(BodyParams(), [ArmParams(stretch=stretch) for _ in range(8)], "hero")
    tips = np.array([pts[arm_bone_offset(k) + arm.SEGMENTS] for k in range(8)])
    return float(np.linalg.norm(tips[:, None] - tips[None], axis=-1).max())


def size_report(name=None):
    """The active (or named) size row in metres, marines and game tiles (0.333 m)."""
    keep = SIZE
    if name is not None:
        set_size(name)
    try:
        row = SIZES[SIZE]
        S = OCTOPUS_SCALE
        rest, full = span_m(0.0), span_m(1.0)
        marine_tiles = 6.0                    # the renderer draws the marine 6 tiles tall
        tiles = lambda m: m / MARINE_HEIGHT_M * marine_tiles
        out = dict(size=SIZE, note=row.note, scale=S, body_length_m=row.body_length_m,
                   mantle_width_m=2 * MANTLE_R[0] * S, mantle_length_m=2 * MANTLE_R[2] * S,
                   arm_length_rest_m=arm_length(0) * S, arm_length_stretched_m=arm_length(1) * S,
                   arm_body_ratio=row.arm_body_ratio, arm_lambda=ARM_LAMBDA, stretch_max=STRETCH_MAX,
                   arm_base_diameter_m=2 * float(arm.radius(S_J)) * S,
                   span_rest_m=rest, span_stretched_m=full,
                   span_rest_marines=rest / MARINE_HEIGHT_M, span_stretched_marines=full / MARINE_HEIGHT_M,
                   span_rest_game_tiles=tiles(rest), span_stretched_game_tiles=tiles(full),
                   span_rest_tiles_at_0333m=rest / 0.333, span_stretched_tiles_at_0333m=full / 0.333)
    finally:
        set_size(keep)
    return out


# --------------------------------------------------------------------------
# Implicit body used ONLY to place the crown/body grid vertices.
def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _ellipsoid(p, c, r):
    q = (p - c) / r
    return (np.linalg.norm(q, axis=-1) - 1) * r.min()


WEB_RHO_MIN = 0.175            # web rim between arms (shallow webbing)
WEB_T0, WEB_T1 = 0.052, 0.010  # web half thickness at the axis / at the rim


def ring_rho():
    ring = MOUNT_RHO + X_J * math.cos(DROOP)
    r = float(arm.radius(S_J))
    return ring, r, math.hypot(ring, r)


def _web_rim(psi):
    """Scalloped rim radius of the web as a function of crown azimuth."""
    ring, r, rho_lat = ring_rho()
    half = np.arcsin(r / ring)
    az = np.sort(ARM_AZIMUTH)
    out = np.full(np.shape(psi), 0.97 * rho_lat)
    psi = np.asarray(psi)
    for i in range(8):
        a0, a1 = az[i], az[(i + 1) % 8] + (2 * math.pi if i == 7 else 0)
        lo, hi = a0 + half, a1 - half
        d = np.mod(psi - lo, 2 * math.pi)
        inside = d < (hi - lo)
        t = np.clip(d / (hi - lo), 0, 1)
        dip = np.sin(math.pi * t) ** 0.6
        out = np.where(inside, 0.97 * rho_lat - (0.97 * rho_lat - WEB_RHO_MIN) * dip, out)
    return out


def body_sdf(p):
    p = np.asarray(p, float)
    head = _ellipsoid(p, HEAD_C, HEAD_R)
    mantle = _ellipsoid(p, MANTLE_C, MANTLE_R)
    d = _smin(head, mantle, 0.11)
    for sx in (1, -1):
        e = EYE_C * [sx, 1, 1]
        d = _smin(d, np.linalg.norm(p - e, axis=-1) - EYE_BUMP_R, 0.03)
    crown = None
    for k in range(8):
        local = (p - MOUNT_POINTS[k]) @ MOUNT_FRAMES[k]
        x, y, z = local[..., 0], local[..., 1], local[..., 2]
        s = np.clip(x / arm.LENGTH, 0, 1)
        q = np.sqrt((y / .8) ** 2 + z ** 2)
        r = arm.radius(s)
        tube = 0.85 * (q - r)
        tube = np.maximum(tube, np.maximum(x - X_J, -x - 0.05))
        crown = tube if crown is None else _smin(crown, tube, 0.05)
    rho = np.hypot(p[..., 0], p[..., 2])
    psi = np.arctan2(p[..., 0], p[..., 2])
    ymid = MOUNT_Y - math.tan(DROOP) * (rho - MOUNT_RHO)
    rim = _web_rim(psi)
    t = WEB_T1 + (WEB_T0 - WEB_T1) * np.clip(1 - rho / rim, 0, 1)
    rr = WEB_T1 * 0.9
    qx = np.abs(p[..., 1] - ymid) - (t - rr)
    qy = rho - (rim - rr)
    web = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rr
    crown = _smin(crown, web, 0.03)
    oral = _ellipsoid(p, np.array([0., .07, 0.]), np.array([.125, .045, .125]))
    crown = _smin(crown, oral, 0.04)
    return _smin(d, crown, 0.06)


def cast(centre, dirs, t_max=0.9, step=0.004, iters=22):
    """Outermost surface hit along rays centre + t dir (marched from outside in)."""
    dirs = np.asarray(dirs, float)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    n = len(dirs)
    ts = np.arange(t_max, 0, -step)
    hit = np.full(n, np.nan)
    for chunk in range(0, n, 2048):
        d = dirs[chunk:chunk + 2048]
        pts = centre + ts[None, :, None] * d[:, None, :]
        v = body_sdf(pts.reshape(-1, 3)).reshape(len(d), len(ts))
        inside = v < 0
        first = np.argmax(inside, axis=1)
        ok = inside[np.arange(len(d)), first] & ~inside[:, 0]
        lo = ts[first]              # inside
        hi = np.minimum(lo + step, t_max)
        for _ in range(iters):
            mid = (lo + hi) / 2
            m = body_sdf(centre + mid[:, None] * d) < 0
            lo, hi = np.where(m, mid, lo), np.where(m, hi, mid)
        hit[chunk:chunk + 2048] = np.where(ok, (lo + hi) / 2, np.nan)
    if np.isnan(hit).any():
        raise ValueError("ray missed the body surface")
    return centre + hit[:, None] * dirs


def _slerp(a, b, u):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.clip((a * b).sum(-1, keepdims=True), -1, 1)
    om = np.arccos(dot)
    so = np.sin(om)
    so = np.where(so < 1e-9, 1, so)
    return (np.sin((1 - u) * om) * a + np.sin(u * om) * b) / so


TOP_CENTRE = np.array([0., .22, -.06])
TOP_POLE_DIR = np.array([0., 1., -.55])
BOT_CENTRE = np.array([0., .10, 0.])
BOT_POLE_DIR = np.array([0., -1., 0.])


def _sheet(boundary, centre, pole_dir, rows, samples=180, grade=1.0):
    """Grid of rows x len(boundary) on the implicit body, row 0 = boundary, last = pole."""
    nb = len(boundary)
    u = np.linspace(0, 1, samples)
    dB = boundary - centre
    dirs = _slerp(dB[:, None, :], np.broadcast_to(pole_dir, dB.shape)[:, None, :], u[None, :, None])
    pts = cast(centre, dirs[:, 1:].reshape(-1, 3)).reshape(nb, samples - 1, 3)
    pts = np.concatenate([boundary[:, None, :], pts], axis=1)
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    cum = np.concatenate([np.zeros((nb, 1)), np.cumsum(seg, axis=1)], axis=1)
    frac = (np.arange(rows + 1) / rows) ** grade
    grid = np.empty((rows + 1, nb, 3))
    for c in range(nb):
        target = frac * cum[c, -1]
        for k in range(3):
            grid[:, c, k] = np.interp(target, cum[c], pts[c, :, k])
    grid[-1] = grid[-1].mean(axis=0)            # one pole point
    # Light tangential relaxation with re-projection (interior rows only).
    for _ in range(6):
        inner = grid[1:-1]
        avg = 0.25 * (np.roll(inner, 1, axis=1) + np.roll(inner, -1, axis=1) + grid[:-2] + grid[2:])
        moved = inner + 0.5 * (avg - inner)
        grid[1:-1] = cast(centre, (moved - centre).reshape(-1, 3)).reshape(moved.shape)
    return grid


# --------------------------------------------------------------------------
def _ring_local(s, sectors):
    r = float(arm.radius(s))
    a = 2 * np.pi * np.arange(sectors) / sectors
    return np.column_stack([np.full(sectors, s * arm.LENGTH), .8 * r * np.cos(a), r * np.sin(a)])


def to_model(k, local):
    return np.asarray(local) @ MOUNT_FRAMES[k].T + MOUNT_POINTS[k]


def _half_orders(sectors):
    hi, lo = sectors // 4, 3 * sectors // 4
    oral = [lo + i for i in range(sectors - lo)] + list(range(0, hi + 1))
    dorsal = list(range(lo, hi - 1, -1))
    return oral, dorsal


def _rim_points(count):
    """Web rim vertices between consecutive (by azimuth) arms, excluding laterals."""
    ring, r, rho_lat = ring_rho()
    half = math.asin(r / ring)
    order = np.argsort(ARM_AZIMUTH)
    out = []
    for i in range(8):
        a0 = ARM_AZIMUTH[order[i]] + half
        a1 = ARM_AZIMUTH[order[(i + 1) % 8]] - half + (2 * math.pi if i == 7 else 0)
        psis = a0 + (a1 - a0) * (np.arange(1, count + 1) / (count + 1))
        dirs = []
        for psi in psis:
            radial = np.array([math.sin(psi), 0., math.cos(psi)])
            d = math.cos(DROOP) * radial + math.sin(DROOP) * np.array([0., -1., 0.])
            dirs.append(d)
        dirs = np.array(dirs)
        start = np.array([0., MOUNT_Y + math.tan(DROOP) * MOUNT_RHO, 0.])
        out.append(cast(start, dirs, t_max=0.6, step=0.002))
    return order, out


def _seam_snap(grid):
    """Near each junction ring, pull crown vertices that lie ON the arm root onto the
    exact instanced-tube surface, so the seam has no step from the implicit blend
    (the smooth union inflates the root by up to k/4). Row 0 is untouched."""
    g = grid.copy()
    pts = g[1:].reshape(-1, 3)
    for k in range(8):
        local = (pts - MOUNT_POINTS[k]) @ MOUNT_FRAMES[k]
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        r = arm.radius(np.clip(x / arm.LENGTH, 0, 1))
        q = np.sqrt((y / .8) ** 2 + z ** 2)
        w = _smoothstep((x - 0.55 * X_J) / (0.45 * X_J)) * _smoothstep((1.18 - q / r) / 0.12) * (x <= X_J + 1e-6)
        scale = np.where(q > 1e-9, r / np.maximum(q, 1e-9), 1)
        target = np.column_stack([x, y * scale, z * scale]) @ MOUNT_FRAMES[k].T + MOUNT_POINTS[k]
        pts = pts + w[:, None] * (target - pts)
    g[1:] = pts.reshape(g[1:].shape)
    return g


COLLAR_LEN = 0.075


def _collar(grid, cols, sectors, sign):
    """G1 seam: the first rows inward of each junction ring follow the arm tube's
    own surface lines (same ring angle, decreasing s), blending into the sheet
    by COLLAR_LEN of arc. Strongest on the dorsal (sign -1) / oral (+1) crest,
    zero at the laterals where the web takes over."""
    g = grid.copy()
    for c, col in enumerate(cols):
        if col[0] != "ring":
            continue
        k, j = col[1], col[2]
        a = 2 * math.pi * j / sectors
        strength = float(_smoothstep((sign * math.cos(a) - 0.05) / 0.5))
        if strength <= 0:
            continue
        dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(grid[:, c], axis=0), axis=1))])
        for r in range(1, len(g) - 1):
            if dist[r] >= COLLAR_LEN:
                break
            sv = S_J - dist[r] / arm.LENGTH
            P = to_model(k, _ring_local(sv, sectors)[j])
            b = float(_smoothstep(dist[r] / COLLAR_LEN))
            g[r, c] = grid[r, c] + strength * (1 - b) * (P - grid[r, c])
    # Relax the seam band (rows 1..m, row 0 and row m+1 held): removes the shear
    # folds where collar columns meet the web-bound lateral columns.
    m = max(2, int(np.searchsorted(np.linalg.norm(np.diff(g[:, 0], axis=0), axis=1).cumsum(), COLLAR_LEN * 1.3)) + 1)
    m = min(m, len(g) - 3)
    for _ in range(12):
        band = g[1:m + 1]
        avg = 0.25 * (np.roll(band, 1, axis=1) + np.roll(band, -1, axis=1) + g[0:m] + g[2:m + 2])
        g[1:m + 1] = band + 0.45 * (avg - band)
    return g


def crown_mesh(sectors, web_cols, top_rows, bot_rows, base_index_of_ring):
    """Body + crown + web surface whose boundary is the 8 junction rings.

    `base_index_of_ring(k, j)` -> global vertex index of arm k's junction-ring
    vertex j (the arm tube owns those vertices; this mesh only references
    them). Returns (new_vertices, faces, info) where new vertex indices start
    at the caller's `start` offset (supplied through info by `assemble`).
    """
    order, rims = _rim_points(web_cols)
    oral, dorsal = _half_orders(sectors)
    ring_pos = {k: to_model(k, _ring_local(S_J, sectors)) for k in range(8)}
    top_cols, bot_cols = [], []     # list of ("ring", k, j) or ("rim", i, c)
    for i, k in enumerate(order):
        top_cols += [("ring", k, j) for j in dorsal[:-1]] + [("ring", k, dorsal[-1])]
        bot_cols += [("ring", k, j) for j in oral[:-1]] + [("ring", k, oral[-1])]
        top_cols += [("rim", i, c) for c in range(web_cols)]
        bot_cols += [("rim", i, c) for c in range(web_cols)]

    def pos(col):
        return ring_pos[col[1]][col[2]] if col[0] == "ring" else rims[col[1]][col[2]]

    top_b = np.array([pos(c) for c in top_cols])
    bot_b = np.array([pos(c) for c in bot_cols])
    top = _collar(_seam_snap(_sheet(top_b, TOP_CENTRE, TOP_POLE_DIR, top_rows, grade=1.15)), top_cols, sectors, -1)
    bot = _collar(_seam_snap(_sheet(bot_b, BOT_CENTRE, BOT_POLE_DIR, bot_rows, grade=1.0)), bot_cols, sectors, 1)
    return dict(top_cols=top_cols, bot_cols=bot_cols, top=top, bot=bot, rims=rims, order=order)


# --------------------------------------------------------------------------
# Skin weights for crown/body vertices.
def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def body_weights(v, lod):
    """Weights over bone names for body/crown vertices (<= 4 influences kept).

    Arm influence = angular partition between the two nearest arms (by crown
    azimuth) x a radial ramp along each arm's own axis that reaches 1 at the
    junction ring, where the weights equal the tube's two-joint weights exactly.
    The rest goes to body / mantle (+ the translation-only squeeze walls).
    """
    names = bones(lod)
    index = {n: i for i, n in enumerate(names)}
    step = arm.STEP * (1 if lod == "hero" else GAME_EVERY)
    nseg = arm_joint_count(lod) - 1
    n = len(v)
    W = np.zeros((n, len(names)))
    psi = np.arctan2(v[:, 0], v[:, 2])
    order = np.argsort(ARM_AZIMUTH)
    az = ARM_AZIMUTH[order]
    ring, r, _ = ring_rho()
    half = math.asin(r / ring)
    arm_w = np.zeros((n, 8))
    for i in range(8):
        k0, k1 = order[i], order[(i + 1) % 8]
        a0 = az[i]
        a1 = az[(i + 1) % 8] + (2 * math.pi if i == 7 else 0)
        d = np.mod(psi - a0, 2 * math.pi)
        inside = d <= (a1 - a0)
        # plateau over each arm's own angular half-width, then a smooth handover
        t = _smoothstep((d - half) / max(1e-9, (a1 - a0) - 2 * half))
        arm_w[:, k0] += np.where(inside, 1 - t, 0)
        arm_w[:, k1] += np.where(inside, t, 0)
    for k in range(8):
        local = (v - MOUNT_POINTS[k]) @ MOUNT_FRAMES[k]
        x = local[:, 0]
        q = np.sqrt((local[:, 1] / .8) ** 2 + local[:, 2] ** 2) / arm.radius(np.clip(x / arm.LENGTH, 0, 1))
        ramp = _smoothstep((x - 0.0) / (0.75 * X_J)) * _smoothstep((2.6 - q) / 1.2)
        a = arm_w[:, k] * ramp
        f = np.clip(x, 0, X_J) / step
        j0 = np.minimum(f.astype(int), nseg - 1)
        t = f - j0
        o = arm_bone_offset(k, lod)
        np.add.at(W, (np.arange(n), o + j0), a * (1 - t))
        np.add.at(W, (np.arange(n), o + j0 + 1), a * t)
    rest = 1 - W.sum(axis=1)
    # mantle share of the rest
    mu = _smoothstep((-(v[:, 2] - NECK[2]) + 0.04) / 0.24) * _smoothstep((v[:, 1] - 0.15) / 0.14)
    nrm = (v - MANTLE_C) / MANTLE_R
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    wl = 0.7 * np.maximum(nrm[:, 0], 0) ** 2
    wr = 0.7 * np.maximum(-nrm[:, 0], 0) ** 2
    wd = 0.7 * np.maximum(nrm[:, 1], 0) ** 2
    m = rest * mu
    W[:, index["mantle_squeeze_L"]] += m * wl
    W[:, index["mantle_squeeze_R"]] += m * wr
    W[:, index["mantle_squeeze_D"]] += m * wd
    W[:, index["mantle"]] += m * (1 - wl - wr - wd)
    W[:, index["body"]] += rest * (1 - mu)
    return _top4(W)


def _top4(W):
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, axis=1)
    w = np.where(w < 1e-6, 0, w)
    w /= w.sum(axis=1, keepdims=True)
    return idx, w


def _single(n, bone_index):
    j = np.zeros((n, 4), int)
    j[:, 0] = bone_index
    w = np.zeros((n, 4))
    w[:, 0] = 1
    return j, w


# --------------------------------------------------------------------------
# Eyes, siphon, game tube.
def eye_mesh(sx, seg_u, seg_v):
    """UV sphere; UVs: u around the eye's outward axis, v from the cornea pole."""
    out = np.array([sx * .80, .45, .30])
    out /= np.linalg.norm(out)
    c = EYE_C * [sx, 1, 1] + EYE_BALL_OUT * out
    up = np.cross(out, [0., 0., 1.]) if sx > 0 else np.cross([0., 0., 1.], out)
    up = np.array([0., 1., 0.]) - np.dot([0., 1., 0.], out) * out
    up /= np.linalg.norm(up)
    side = np.cross(out, up)
    verts, uvs = [c + out * EYE_BALL_R], [(0.5, 0.5)]
    for i in range(1, seg_v):
        th = math.pi * i / seg_v
        for j in range(seg_u):
            ph = 2 * math.pi * j / seg_u
            d = math.cos(th) * out + math.sin(th) * (math.cos(ph) * up + math.sin(ph) * side)
            verts.append(c + EYE_BALL_R * d)
            rr = 0.5 * th / math.pi
            uvs.append((0.5 + rr * math.sin(ph), 0.5 + rr * math.cos(ph)))
    verts.append(c - out * EYE_BALL_R)
    uvs.append((0.5, 0.0))
    faces = []
    for j in range(seg_u):
        faces.append([0, 1 + j, 1 + (j + 1) % seg_u])
    for i in range(seg_v - 2):
        for j in range(seg_u):
            a = 1 + i * seg_u + j
            b = 1 + i * seg_u + (j + 1) % seg_u
            faces.append([a, a + seg_u, b + seg_u, b])
    last = len(verts) - 1
    base = 1 + (seg_v - 2) * seg_u
    for j in range(seg_u):
        faces.append([base + j, last, base + (j + 1) % seg_u])
    if sx < 0:
        faces = [f[::-1] for f in faces]
    return np.array(verts), faces, uvs


def siphon_mesh(seg, rings):
    """Short open funnel with a lip and an inner bore (closed manifold)."""
    d = SIPHON_DIR / np.linalg.norm(SIPHON_DIR)
    a = np.cross(d, [0., 1., 0.])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    profile = []
    for i in range(rings):
        t = i / (rings - 1)
        profile.append((-0.04 + t * (SIPHON_LEN + 0.04), SIPHON_R0 + (SIPHON_R1 - SIPHON_R0) * t))
    profile.append((SIPHON_LEN + .006, SIPHON_R1 * .92))
    profile.append((SIPHON_LEN + .004, SIPHON_R1 * .55))
    profile.append((SIPHON_LEN - .03, SIPHON_R1 * .45))
    verts = []
    for h, r in profile:
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            verts.append(SIPHON_BASE + h * d + r * (math.cos(ph) * a + math.sin(ph) * b))
    faces = []
    for i in range(len(profile) - 1):
        for j in range(seg):
            p, q = i * seg + j, i * seg + (j + 1) % seg
            faces.append([p, q, q + seg, p + seg])
    # caps (base buried inside the body; bore bottom)
    for ring_i, flip in ((0, True), (len(profile) - 1, False)):
        c = len(verts)
        verts.append(np.mean(verts[ring_i * seg:(ring_i + 1) * seg], axis=0))
        for j in range(seg):
            f = [ring_i * seg + j, ring_i * seg + (j + 1) % seg, c]
            faces.append(f[::-1] if flip else f)
    return np.array(verts), faces


def game_tube(sectors, n_rings):
    """Arm tube for the game LOD in arm-local metres: rings from S_J to s=127/128 + tip."""
    s_last = (HERO_RINGS - 2) / (HERO_RINGS - 1)
    u = np.linspace(0, 1, n_rings)
    svals = S_J + (s_last - S_J) * (1 - (1 - u) ** 1.25)
    rings = [_ring_local(s, sectors) for s in svals]
    verts = np.vstack(rings + [np.array([[arm.LENGTH, 0., 0.]])])
    faces = []
    for i in range(n_rings - 1):
        for j in range(sectors):
            a, b = i * sectors + j, i * sectors + (j + 1) % sectors
            faces.append([a, a + sectors, b + sectors, b])
    tip = len(verts) - 1
    for j in range(sectors):
        faces.append([(n_rings - 1) * sectors + j, tip, (n_rings - 1) * sectors + (j + 1) % sectors])
    return verts, faces, svals


def arm_joint_coord(x_mat, lod="hero"):
    """Fractional joint index (in `lod`'s arm chain) of material arm coordinate x_mat.

    Interpolated in REST space between the lod's joints, so a vertex keeps its
    fraction between its two joints whatever the rest segment lengths (hero:
    exactly x_mat / STEP, the knots being the joints).
    """
    every = 1 if lod == "hero" else GAME_EVERY
    jx = BIND_JOINT_X[::every]
    return np.interp(bind_x(np.clip(x_mat, 0, arm.LENGTH)), jx, np.arange(len(jx), dtype=float))


def hero_arm():
    """The proof's arm geometry (arm_math.geometry) cut at the junction ring.

    Returns arm-local vertices (BASE removed), faces, the junction-ring index
    map, cups (start/stop ranges in the new numbering, s) and the 'kind' array
    (0 tube, 1 cup). Cups inward of the junction ride on the crown underside
    (moved onto it later); cups within 0.06 of the mouth are dropped.
    Arm x is in MATERIAL coordinates here (the proof's); `assemble` maps it to
    the rest length. A shorter-than-proof arm (ARM_LAMBDA < 1) keeps only the
    cup stations whose rest spacing is at least the proof's own spacing rule,
    so suckers never overlap.
    """
    d = arm.geometry()
    v = d["vertices"] - arm.BASE
    sectors = d["sectors"]
    keep_tube = list(range(JUNCTION_RING * sectors, (HERO_RINGS - 1) * sectors)) + [d["tube_count"] - 2]
    keep = list(keep_tube)
    cups = [c for c in d["cups"] if c["s"] >= 0.06]
    stations, last = set(), None
    for s in sorted({c["s"] for c in cups}):
        r = [c["radius"] for c in cups if c["s"] == s][0]
        if last is None or s * arm.LENGTH < X_J + 0.01 or \
                float(bind_x(s * arm.LENGTH) - bind_x(last[0] * arm.LENGTH)) >= 0.97 * max(.013 * arm.LENGTH, 2.60 * last[1]):
            stations.add(s)
            last = (s, r)
    cups = [c for c in cups if c["s"] in stations]
    for c in cups:
        keep += list(range(c["start"], c["stop"]))
    remap = -np.ones(len(v), int)
    remap[keep] = np.arange(len(keep))
    faces = []
    for f in d["faces"]:
        m = remap[f]
        if (m >= 0).all():
            faces.append(m.tolist())
    kind = np.zeros(len(keep), int)
    kind[len(keep_tube):] = 1
    new_cups = []
    for c in cups:
        new_cups.append(dict(start=int(remap[c["start"]]), stop=int(remap[c["stop"] - 1]) + 1, s=c["s"],
                             centre=np.array(c["centre"]) - arm.BASE, radius=c["radius"]))
    attach = np.concatenate([v[keep_tube, 0], np.concatenate(
        [np.full(c["stop"] - c["start"], c["s"] * arm.LENGTH) for c in cups])])
    # sucker "bowl" factor for the hero shader: inner profile rings (cup lip interior)
    bowl = np.zeros(len(keep))
    for c in cups:
        n = c["stop"] - c["start"]   # foot centre, 5 rings x 12, bowl centre
        prof = [0] + np.repeat(np.arange(1, 6), 12).tolist() + [6]
        b = np.array([0, 0, 0, .2, .6, 1, 1.0])[prof]
        bowl[remap[c["start"]]:remap[c["start"]] + n] = b
    ring_index = {j: j for j in range(sectors)}  # junction ring = first `sectors` new vertices
    return dict(vertices=v[keep], faces=faces, kind=kind, cups=new_cups, attach=attach,
                bowl=bowl, ring_index=ring_index, sectors=sectors,
                tube_rings=HERO_RINGS - 1 - JUNCTION_RING)


def assemble(lod="hero"):
    """The whole animal as one mesh at scale 1: vertices, faces, skin, attributes.

    Arm k's tube vertices [0, sectors) are its junction ring and are shared by
    the crown mesh. Attributes: kind (0 skin, 1 cup, 2 eye, 3 siphon),
    pale (0 dorsal .. 1 oral/underside), bowl (cup interior), mouth.
    """
    names = bones(lod)
    bindp = bind_pose(lod)
    V, F, J, Wt, kind, pale, bowl, uv_eye = [], [], [], [], [], [], [], {}
    nv = 0
    arm_ring = {}
    arm_ranges = []
    if lod == "hero":
        ha = hero_arm()
        sectors = ha["sectors"]
        web_cols, top_rows, bot_rows = 5, 30, 14
    else:
        sectors = 12
        web_cols, top_rows, bot_rows = 3, 15, 7
        gt, gf, gs = game_tube(sectors, 44)
    step = arm.STEP * (1 if lod == "hero" else GAME_EVERY)
    for k in range(8):
        if lod == "hero":
            lv, lf = ha["vertices"], ha["faces"]
            attach = ha["attach"]
            kd, bw = ha["kind"], ha["bowl"]
        else:
            lv, lf = gt, gf
            attach = lv[:, 0]
            kd, bw = np.zeros(len(lv), int), np.zeros(len(lv))
        # material -> rest arm length: the tube point by point, each cup rigidly by its station
        lvb = np.array(lv, float)
        lvb[:, 0] += bind_x(attach) - attach
        mv = to_model(k, lvb)
        f = arm_joint_coord(attach, lod)
        a = np.minimum(f.astype(int), arm_joint_count(lod) - 2)
        t = f - a
        o = arm_bone_offset(k, lod)
        jj = np.zeros((len(lv), 4), int)
        ww = np.zeros((len(lv), 4))
        jj[:, 0], jj[:, 1] = o + a, o + a + 1
        ww[:, 0], ww[:, 1] = 1 - t, t
        # Pale = oral side of the tube (local +Y), like the proof's atlas.
        ang = np.arctan2(lv[:, 2], lv[:, 1] / .8)
        pl = np.clip((np.cos(ang) - .05) / .45, 0, 1)
        pl = np.where(kd == 1, 1.0, pl)
        V.append(mv); F += [[i + nv for i in face] for face in lf]
        J.append(jj); Wt.append(ww); kind.append(kd); pale.append(pl); bowl.append(bw)
        arm_ring[k] = nv
        arm_ranges.append((nv, nv + len(lv)))
        nv += len(lv)
    crown = crown_mesh(sectors, web_cols, top_rows, bot_rows, None)
    # Crown vertices: row 0 references arm rings (ring cols) or new rim vertices.
    rim_index = {}
    newv = []

    def rim_vid(i, c):
        nonlocal nv
        key = (i, c)
        if key not in rim_index:
            rim_index[key] = nv + len(newv)
            newv.append(crown["rims"][i][c])
        return rim_index[key]

    def col_vid(col):
        if col[0] == "ring":
            return arm_ring[col[1]] + col[2]
        return rim_vid(col[1], col[2])

    sheets = []
    for name, grid, cols in (("top", crown["top"], crown["top_cols"]), ("bot", crown["bot"], crown["bot_cols"])):
        rows, ncol = grid.shape[0] - 1, grid.shape[1]
        ids = np.empty((rows + 1, ncol), int)
        ids[0] = [col_vid(c) for c in cols]
        for r in range(1, rows):
            for c in range(ncol):
                ids[r, c] = nv + len(newv)
                newv.append(grid[r, c])
        pole = nv + len(newv)
        newv.append(grid[-1, 0])
        ids[-1] = pole
        sheets.append((name, ids))
        for r in range(rows):
            for c in range(ncol):
                a, b = ids[r, c], ids[r, (c + 1) % ncol]
                cc, d = ids[r + 1, (c + 1) % ncol], ids[r + 1, c]
                face = [a, b, cc, d] if r < rows - 1 else [a, b, d]
                if name == "bot":
                    face = face[::-1]
                if len(set(face)) == len(face):
                    F.append(face)
    newv = np.array(newv)
    cj, cw = body_weights(newv, lod)
    V.append(newv); J.append(cj); Wt.append(cw)
    kind.append(np.zeros(len(newv), int))
    # pale: underside sheet (and web underside) ~ 1, top 0, blend by height
    bot_ids = set(sheets[1][1][1:].ravel().tolist())
    pl = np.array([1.0 if (nv + i) in bot_ids else 0.0 for i in range(len(newv))])
    pale.append(pl)
    bowl.append(np.zeros(len(newv)))
    crown_range = (nv, nv + len(newv))
    nv += len(newv)
    # Mouth factor (bottom pole region).
    # Cups that ride on the crown: move each cup onto the implicit underside.
    V = np.vstack(V)
    if lod == "hero":
        for k in range(8):
            base = arm_ranges[k][0]
            for c in ha["cups"]:
                if c["s"] * arm.LENGTH >= X_J + 0.01:
                    continue
                centre = to_model(k, c["centre"])
                nrm = MOUNT_FRAMES[k] @ np.array([.035, 1., (1 if c["centre"][2] > 0 else -1) * .38])
                nrm /= np.linalg.norm(nrm)
                hit = cast(centre - nrm * 0.03, nrm[None], t_max=0.12, step=0.001)[0]
                V[base + c["start"]:base + c["stop"]] += hit - centre
    J = np.vstack(J)
    Wt = np.vstack(Wt)
    kind = np.concatenate(kind)
    pale = np.concatenate(pale)
    bowl = np.concatenate(bowl)
    # Eyes and siphon (rigid islands).
    eye_uv = []
    eye_ranges = []
    for sx in (1, -1):
        ev, ef, euv = eye_mesh(sx, 40 if lod == "hero" else 14, 24 if lod == "hero" else 8)
        F += [[i + nv for i in f] for f in ef]
        V = np.vstack([V, ev])
        j, w = _single(len(ev), names.index("body"))
        J = np.vstack([J, j]); Wt = np.vstack([Wt, w])
        kind = np.concatenate([kind, np.full(len(ev), 2)])
        pale = np.concatenate([pale, np.zeros(len(ev))]); bowl = np.concatenate([bowl, np.zeros(len(ev))])
        eye_ranges.append((nv, nv + len(ev)))
        eye_uv.append(euv)
        nv += len(ev)
    sv, sf = siphon_mesh(16 if lod == "hero" else 8, 6 if lod == "hero" else 3)
    F += [[i + nv for i in f] for f in sf]
    V = np.vstack([V, sv])
    j, w = _single(len(sv), names.index("siphon"))
    J = np.vstack([J, j]); Wt = np.vstack([Wt, w])
    kind = np.concatenate([kind, np.full(len(sv), 3)])
    pale = np.concatenate([pale, np.full(len(sv), .35)]); bowl = np.concatenate([bowl, np.zeros(len(sv))])
    siphon_range = (nv, nv + len(sv))
    nv += len(sv)
    mouth = np.zeros(nv)
    bp = sheets[1][1][-1, 0]
    dist = np.linalg.norm(V - V[bp], axis=1)
    mouth[crown_range[0]:crown_range[1]] = np.clip(1.6 - dist[crown_range[0]:crown_range[1]] / .028, 0, 1)
    return dict(vertices=V, faces=F, joints=J, weights=Wt, kind=kind, pale=pale, bowl=bowl, mouth=mouth,
                bones=names, bind=bindp, arm_ranges=arm_ranges, crown_range=crown_range,
                eye_ranges=eye_ranges, eye_uv=eye_uv, siphon_range=siphon_range,
                sheets={n: ids for n, ids in sheets}, sectors=sectors, lod=lod)


# --------------------------------------------------------------------------
# Named poses and the swim cycle (demonstration data, not behaviour).
def rest_arms():
    """The sheet's relaxed, mirrored modelling pose: gentle curve, hooked tips."""
    arms = []
    for n in ARM_NAMES:
        i = int(n[1])
        # front pair curves outward-back, the rest toward the front; tips hook
        bend = {1: -0.5, 2: 0.5, 3: 0.5, 4: 0.5}[i]
        tip = {1: 0.55, 2: 0.45, 3: -0.40, 4: 0.50}[i]
        arms.append(ArmParams(curl={1: .09, 2: .10, 3: .10, 4: .08}[i], bend_dir=bend, twist=0.0,
                              tip_curl=tip, lift=0.04, swing=0.0))
    return arms


def named_poses():
    rest = rest_arms()
    curled = [ArmParams(curl=0.85, bend_dir=0.0, twist=0.25, tip_curl=0.6, lift=0.35) for _ in ARM_NAMES]
    reach = [ArmParams(**vars(a)) for a in rest]
    reach[0] = ArmParams(curl=0.05, bend_dir=0.5, twist=-0.3, tip_curl=-0.25, lift=0.55, swing=0.6, stretch=0.8)
    swim_body = BodyParams(mantle_squeeze=0.9, mantle_pitch=-0.2, siphon_yaw=0.0, siphon_pitch=-0.6,
                           crown_spread=-0.85, body_pitch=0.0)
    swim = [ArmParams(curl=0.08, bend_dir=1.0, twist=0.0, tip_curl=0.25, lift=0.0) for _ in ARM_NAMES]
    walk = []
    for k, n in enumerate(ARM_NAMES):
        phase = (k % 4) / 4 + (0.5 if n[0] == "R" else 0)
        walk.append(ArmParams(curl=0.25 + 0.2 * math.sin(2 * math.pi * phase), bend_dir=0.5 * math.cos(2 * math.pi * phase),
                              twist=0.3 * math.sin(2 * math.pi * phase), tip_curl=0.5,
                              lift=0.35 * max(0, math.sin(2 * math.pi * phase)), swing=0.8 * math.cos(2 * math.pi * phase)))
    return {
        "rest": (BodyParams(), rest),
        "curled": (BodyParams(crown_spread=0.15), curled),
        "reach": (BodyParams(), reach),
        "swim": (swim_body, swim),
        "walk": (BodyParams(mantle_pitch=0.15, siphon_yaw=-0.4), walk),
    }


SWIM_PERIOD = 2.4
WALK_PERIOD = 3.0


def walk_cycle(seconds):
    """Asymmetric crawl: arm pairs swing and lift in a travelling wave (demo data)."""
    ph0 = (seconds % WALK_PERIOD) / WALK_PERIOD
    arms = []
    for n in ARM_NAMES:
        ph = 2 * math.pi * (ph0 + (int(n[1]) - 1) / 4 + (0.5 if n[0] == "R" else 0))
        arms.append(ArmParams(curl=0.22 + 0.12 * math.sin(ph), bend_dir=0.5 * math.cos(ph),
                              twist=0.25 * math.sin(ph), tip_curl=0.45,
                              lift=0.3 * (0.5 + 0.5 * math.sin(ph)), swing=0.7 * math.cos(ph)))
    return BodyParams(mantle_pitch=0.1 + 0.08 * math.sin(2 * math.pi * ph0), siphon_yaw=-0.3), arms


def demo_cycle(seconds):
    """Game preview loop: one crawl period, then two swim strokes (switches are not blended)."""
    t = seconds % (WALK_PERIOD + 2 * SWIM_PERIOD)
    if t < WALK_PERIOD:
        return walk_cycle(t)
    return swim_cycle(t - WALK_PERIOD)


def swim_cycle(seconds):
    """Jet swimming: mantle contracts (power) then refills; arms trail and flare."""
    ph = (seconds % SWIM_PERIOD) / SWIM_PERIOD
    power = 0.5 - 0.5 * math.cos(2 * math.pi * ph)          # smooth 0..1..0
    body = BodyParams(mantle_squeeze=0.85 * power, mantle_pitch=-0.15 + 0.1 * power,
                      siphon_pitch=-0.6, siphon_yaw=0.0,
                      crown_spread=-0.55 - 0.35 * power, body_pitch=0.0)
    arms = []
    for k, n in enumerate(ARM_NAMES):
        lag = 2 * math.pi * (ph - 0.08 * (int(n[1]) - 1))
        flare = 0.5 - 0.5 * math.cos(lag)
        arms.append(ArmParams(curl=0.08 + 0.14 * (1 - flare), bend_dir=1.0, twist=0.12 * math.sin(lag),
                              tip_curl=0.15 + 0.35 * (1 - flare), lift=0.25 * (1 - flare) - 0.1, swing=0.0))
    return body, arms
