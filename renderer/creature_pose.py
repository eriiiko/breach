"""creature_pose -- the octopus model's bone pose FROM the simulated chain
(#106 P7). Pure numpy, raylib-free, render-only (float; never written back).

The sim owns the animal: eight arm chains of ``n`` nodes (``creature_system.
read_chain`` -> ``ChainView``, world metres, the facing convention). The game
model (``assets/models/octopus/octopus.glb``) is skinned to ONE skeleton,
``octopus_math.bones("game")``: six body bones and eight arms of 17 joints
(every 2nd joint of the 33-joint hero curve). This module turns the first
into the second, once per drawn frame:

* the RIG CONVENTIONS -- bone order, the bind pose, the mount frames, the
  segment lengths and the stretch's thinning -- are ``octopus_math``'s, the
  art scripts' ONE copy (``art/characters/octopus/scripts``), imported here
  and read ONCE into an :class:`OctopusRig` at the size the creature row
  names. Nothing below re-derives them.
* THE SIZE DOOR (:func:`match_size`): the art row whose rest arm length
  (``arm_body_ratio x body_length_m``) is the creature row's ``arm_length_m``
  within 1 %, else refused -- the art size and the sim size must agree.
* the model frame: the model sits at the crown with yaw = the heading (the
  game's ``Unit.facing`` -> raylib yaw, ``unit_model_renderer``), so a node is
  taken to model axes by the crown pose alone (X = the animal's left, Y up,
  Z forward; the deck is y = 0 under a crown on the deck).
* per arm, the joints INSIDE the crown (game joints 0 .. J, J the junction
  ring, whose incoming segments never stretch) ride the crown rigidly at
  their bind places. The FREE arm's stations J+1 .. 16 are walked along the
  arm's TRACK -- the junction J, then the chain's nodes from the first one
  outside the junction's radius -- each the first point of the track at the
  rig's segment length from the previous station (a sphere-polyline
  intersection, so every station lies ON the track and every segment keeps
  its length exactly). The segment lengths are the rig's at the arm's
  stretch control, which is SOLVED so that the free arm ends at the chain's
  tip (clamped to [0, stretch_max]): the art arm starts at a mount inside the
  crown, the sim arm at the crown's rim, so the chain from the junction out
  is longer than the art's free arm at rest (about 1.17x at the mid row) --
  see the P7 report.
* bone rotations (deformation relative to the bind pose, whose arm bones all
  share the mount frame): the junction bone turns the bind direction onto its
  first segment by the MINIMAL rotation; each next bone adds the minimal
  rotation between consecutive segment directions -- parallel transport, the
  chain's Bishop idea, so roll is continuous along the arm. The tip joint
  repeats the last segment's frame (``pose_octopus``'s rule).
* per-bone scale: ``octopus_math.arm_joint_scales`` at the arm's stretch (the
  muscular hydrostat's 1/sqrt thinning; 1 inside the crown).
* render-only wiggle on top, ``[render.creature]`` (read every frame,
  refused missing or mistyped; each 0 switches its effect off): breathing
  (the mantle squeeze on the render clock), a tip-curl wiggle (the distal
  45 % swings in the deck plane, ``octopus_math``'s tip-curl profile), a
  twist (roll about each segment, ``TWIST_MAX`` at the tip), and the arm
  lift (the drawn arm's centre line one drawn radius above its node, so an
  arm on the deck lies ON it).

v1 leaves the body pitch and the crown spread at 0 (the body bones are the
bind pose plus the breathing).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
#: Where the octopus rig's ONE copy lives (pure numpy; the Blender builds,
#: the verify script and this module all import it).
ART_SCRIPTS = _REPO_ROOT / "art" / "characters" / "octopus" / "scripts"
#: The size door's tolerance on the rest arm length (relative).
SIZE_TOLERANCE = 0.01


def _load(name: str):
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, ART_SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:      # pragma: no cover - missing art
        raise ImportError(f"no {name}.py in {ART_SCRIPTS}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def octopus_math():
    """The art scripts' ``octopus_math`` (and its ``arm_math``), imported by
    file -- the ONE copy of the rig conventions."""
    _load("arm_math")
    return _load("octopus_math")


# ---------------------------------------------------------------------------
# The dials: [render.creature]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CreatureRenderSettings:
    """``[render.creature]`` (read every frame). 0 switches an effect off."""
    breathing_amplitude: float   # mantle squeeze at the breath's peak, 0 .. 1
    breathing_period_s: float    # one breath, seconds (> 0)
    tip_curl_amplitude: float    # the tip wiggle, octopus_math tip_curl units 0 .. 1
    tip_curl_period_s: float     # one wiggle, seconds (> 0)
    twist: float                 # roll along the arm, -1 .. 1 (TWIST_MAX at the tip)
    arm_lift: float              # the drawn centre line above its node, in drawn radii (>= 0)
    hit_flash_frames: int        # frames a struck node is marked (0 = off)


_DIAL_KEYS = {"breathing_amplitude": float, "breathing_period_s": float,
              "tip_curl_amplitude": float, "tip_curl_period_s": float,
              "twist": float, "arm_lift": float, "hit_flash_frames": int}


def creature_render_settings(cfg) -> CreatureRenderSettings:
    """``[render.creature]`` from a loaded config (``CFG``). Raises ValueError
    naming the key when the section or a key is missing, mistyped or out of
    range -- a wiggle that silently fell back to a default would hide a
    typo."""
    section = getattr(getattr(cfg, "render", None), "creature", None)
    if section is None:
        raise ValueError("config.toml has no [render.creature] section")
    vals = {}
    for key, kind in _DIAL_KEYS.items():
        if not hasattr(section, key):
            raise ValueError(f"[render.creature] is missing {key!r}")
        v = getattr(section, key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (
                kind is int and not isinstance(v, int)):
            raise ValueError(f"[render.creature] {key} must be a {kind.__name__}, "
                             f"got {v!r}")
        vals[key] = kind(v)
    for key in ("breathing_amplitude", "tip_curl_amplitude"):
        if not 0.0 <= vals[key] <= 1.0:
            raise ValueError(f"[render.creature] {key} must be in [0, 1], got {vals[key]!r}")
    for key in ("breathing_period_s", "tip_curl_period_s"):
        if not vals[key] > 0.0:
            raise ValueError(f"[render.creature] {key} must be > 0, got {vals[key]!r}")
    if not -1.0 <= vals["twist"] <= 1.0:
        raise ValueError(f"[render.creature] twist must be in [-1, 1], got {vals['twist']!r}")
    if vals["arm_lift"] < 0.0 or vals["hit_flash_frames"] < 0:
        raise ValueError("[render.creature] arm_lift and hit_flash_frames must be >= 0")
    return CreatureRenderSettings(**vals)


#: Every wiggle off, the lift off: the pose is the chain's alone (the gates').
NO_WIGGLE = CreatureRenderSettings(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0)


# ---------------------------------------------------------------------------
# The size door
# ---------------------------------------------------------------------------
def match_size(arm_length_m: float, om=None) -> str:
    """The ``octopus_math.SIZES`` row whose rest arm length (mount -> tip,
    straight: ``arm_length() x OCTOPUS_SCALE`` at that row) is the creature
    row's ``arm_length_m`` within :data:`SIZE_TOLERANCE`; the closest such row.
    Refused (ValueError naming every row's length) when none is: the art and
    the sim must be the same animal."""
    om = octopus_math() if om is None else om
    L = float(arm_length_m)
    if not L > 0.0:
        raise ValueError(f"size door: arm length {arm_length_m!r} m")
    prev = om.SIZE
    lengths = {}
    try:
        for name in om.SIZES:
            om.set_size(name)
            lengths[name] = om.arm_length(0.0) * om.OCTOPUS_SCALE
    finally:
        om.set_size(prev)
    best = min(lengths, key=lambda n: abs(lengths[n] - L))
    if abs(lengths[best] - L) > SIZE_TOLERANCE * L:
        rows = ", ".join(f"{n} {v:.3f} m" for n, v in lengths.items())
        raise ValueError(
            f"size door: the creature's arm is {L:.3f} m and no octopus art size "
            f"matches it within {SIZE_TOLERANCE:.0%} ({rows}) -- the drawn animal "
            f"would not be the simulated one")
    return best


# ---------------------------------------------------------------------------
# The rig, read once
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OctopusRig:
    """Everything the pose reads from ``octopus_math``, at one size, in the
    model frame's metres (OCTOPUS_SCALE applied). Arm-indexed arrays are by
    ART arm (``ARM_NAMES``: L1..L4, R1..R4); ``arm_map[a]`` is the art arm of
    the creature row's arm ``a``."""
    size: str
    scale: float                 # OCTOPUS_SCALE: authored -> metres
    stretch_max: float
    bone_names: tuple            # bones("game")
    joints: int                  # per arm (17)
    junction: int                # the last crown joint (3): joints 0..J never stretch
    offsets: np.ndarray          # (8,) arm_bone_offset(k, "game")
    side: np.ndarray             # (8,) +1 left, -1 right
    body_pts: np.ndarray         # (2, 6, 3) body bone heads at squeeze 0 and 1
    crown_pts: np.ndarray        # (8, J+1, 3) the crown joints' bind heads
    bind_dir: np.ndarray         # (8, 3) the bind arm direction (mount frame 'along')
    bind_heads: np.ndarray       # (bones, 3) the bind pose, metres
    seg_a: np.ndarray            # (joints-1,) segment lengths at stretch 0
    seg_b: np.ndarray            # (joints-1,) d(length)/d(stretch) (exactly linear)
    scale_c: np.ndarray          # (joints,) scale = 1/sqrt(1 + stretch * c)
    free_rest: float             # the free arm's length (J -> tip) at stretch 0
    free_span: float             # its growth from stretch 0 to 1
    arm_map: np.ndarray          # (A,) creature arm -> art arm
    track_k0: int                # the first chain node on the track
    node_radius: np.ndarray      # (n,) the drawn arm radius at each chain node
    tip_from: float              # octopus_math.TIP_FROM
    a_tip: float                 # octopus_math.A_TIP
    twist_max: float             # octopus_math.TWIST_MAX


def build_rig(cdef, om=None) -> OctopusRig:
    """The rig for creature row ``cdef`` (``creatures.CreatureDef``): the size
    door, then every table read once from ``octopus_math`` at that size (the
    module's active size is restored after)."""
    om = octopus_math() if om is None else om
    am = sys.modules["arm_math"]
    b = cdef.body
    size = match_size(b.arm_length_m, om)
    if int(b.arms) != len(om.ARM_NAMES):
        raise ValueError(f"the octopus rig has {len(om.ARM_NAMES)} arms, the creature "
                         f"row {cdef.name!r} {b.arms}")
    prev = om.SIZE
    try:
        om.set_size(size)
        S = float(om.OCTOPUS_SCALE)
        names = tuple(om.bones("game"))
        every = int(om.GAME_EVERY)
        nj = om.arm_joint_count("game")
        # the crown joints: game joints whose hero joint's incoming segments
        # all carry a zero stretch ramp (the junction ring and inside)
        fixed_hero = int(np.flatnonzero(np.cumsum(om.RAMP) > 0)[0])   # hero joints 0..this
        J = fixed_hero // every
        body0, _ = om.pose_octopus(om.BodyParams(), None, "game")
        body1, _ = om.pose_octopus(om.BodyParams(mantle_squeeze=1.0), None, "game")
        offsets = np.array([om.arm_bone_offset(k, "game") for k in range(8)])
        crown = np.stack([body0[o:o + J + 1] for o in offsets])
        bind_dir = np.array([om.MOUNT_FRAMES[k][:, 0] for k in range(8)])
        h0 = om.arm_segment_lengths(0.0) * S
        h1 = om.arm_segment_lengths(1.0) * S
        g0 = h0.reshape(-1, every).sum(1)
        g1 = h1.reshape(-1, every).sum(1)
        e1 = 1.0 / om.arm_joint_scales(1.0)[::every] ** 2
        rig_side = np.asarray(om.ARM_SIDE, float)
        az = np.degrees(np.asarray(om.ARM_AZIMUTH, float))
        stretch_max = float(om.STRETCH_MAX)
        # the drawn radius at each chain node (rest arc fraction -> the art's
        # material coordinate -> arm_math.radius)
        n = int(b.nodes_per_arm)
        x_bind = np.linspace(0.0, om.arm_length(0.0), n)
        s_mat = np.clip(om.material_x(x_bind) / am.LENGTH, 0.0, 1.0)
        node_r = np.asarray(am.radius(s_mat), float) * S
        tip_from, a_tip, twist_max = float(om.TIP_FROM), float(om.A_TIP), float(om.TWIST_MAX)
    finally:
        om.set_size(prev)
    # creature arm -> art arm by azimuth (both CCW from the heading seen from
    # above); one-to-one or refused
    sim_az = np.asarray(b.arm_angles_deg, float)
    diff = np.abs((sim_az[:, None] - az[None, :] + 180.0) % 360.0 - 180.0)
    arm_map = diff.argmin(1)
    if sorted(arm_map.tolist()) != list(range(8)):
        raise ValueError(f"creature arms {list(sim_az)} deg do not map one-to-one "
                         f"onto the rig's {list(np.round(az, 1))} deg")
    # the track's first chain node: the first whose rest radius from the crown
    # centre (rim + k x segment) is beyond the junction's by half a chain
    # segment -- a node level with the junction would make the track's first
    # leg a steep step down from the crown
    j_rad = float(np.hypot(crown[0, J, 0], crown[0, J, 2]))
    seg_sim = b.arm_length_m / (n - 1)
    k0 = int(np.argmax(b.body_radius_m + np.arange(n) * seg_sim > j_rad + 0.5 * seg_sim))
    seg_a, seg_b = g0, g1 - g0
    return OctopusRig(
        size=size, scale=S, stretch_max=stretch_max, bone_names=names, joints=nj,
        junction=J, offsets=offsets, side=rig_side,
        body_pts=np.stack([body0[:6], body1[:6]]), crown_pts=crown, bind_dir=bind_dir,
        bind_heads=body0 * 1.0,
        seg_a=seg_a, seg_b=seg_b, scale_c=e1 - 1.0,
        free_rest=float(seg_a[J:].sum()), free_span=float(seg_b[J:].sum()),
        arm_map=arm_map, track_k0=k0, node_radius=node_r,
        tip_from=tip_from, a_tip=a_tip, twist_max=twist_max)


# ---------------------------------------------------------------------------
# Small vectorised kits
# ---------------------------------------------------------------------------
def to_model(points, crown, heading: float) -> np.ndarray:
    """World points (..., 3) (grid axes: x East, y South, z up) -> the model
    frame of a model at ``crown`` with yaw = ``heading`` (the facing
    convention): X = the animal's left, Y = up (z - crown z), Z = forward."""
    d = np.asarray(points, float) - np.asarray(crown, float)
    c, s = math.cos(heading), math.sin(heading)
    out = np.empty(d.shape, float)
    out[..., 0] = -d[..., 0] * s - d[..., 1] * c
    out[..., 1] = d[..., 2]
    out[..., 2] = d[..., 0] * c - d[..., 1] * s
    return out


def _mm(a, b) -> np.ndarray:
    """Batched 3x3 products as an elementwise einsum -- never ``@``: the
    frame path hands no contraction to BLAS (tests/test_no_blas_in_frame_path)."""
    return np.einsum("...ij,...jk->...ik", a, b)


def min_rotation(a, b) -> np.ndarray:
    """The minimal rotations (..., 3, 3) taking unit vectors ``a`` onto ``b``
    (Rodrigues about a x b); an antiparallel pair turns pi about an axis
    normal to ``a``."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    v = np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                  a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                  a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)
    c = (a * b).sum(-1)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -v[..., 2], v[..., 1]
    K[..., 1, 0], K[..., 1, 2] = v[..., 2], -v[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -v[..., 1], v[..., 0]
    den = 1.0 + c
    flip = den < 1e-9
    f = np.where(flip, 0.0, 1.0 / np.where(flip, 1.0, den))
    R = np.eye(3) + K + _mm(K, K) * f[..., None, None]
    if flip.any():
        ax = np.cross(a[flip], [0.0, 1.0, 0.0])
        bad = np.linalg.norm(ax, axis=-1) < 1e-6
        ax[bad] = np.cross(a[flip][bad], [1.0, 0.0, 0.0])
        ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
        R[flip] = 2.0 * ax[:, :, None] * ax[:, None, :] - np.eye(3)
    return R


def axis_rotation(axis, angle) -> np.ndarray:
    """Rodrigues: rotations (..., 3, 3) by ``angle`` about unit ``axis``."""
    axis = np.asarray(axis, float)
    angle = np.asarray(angle, float)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    m = np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s,
                  y * x * C + z * s, c + y * y * C, y * z * C - x * s,
                  z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)
    return m.reshape(angle.shape + (3, 3))


def walk_stations(track: np.ndarray, start: np.ndarray, lengths: np.ndarray):
    """Stations along each arm's polyline ``track`` (arms, m, 3), from
    ``start`` (arms, 3) -- the track's first point -- each the FIRST point of
    the track (beyond the previous station) at Euclidean distance
    ``lengths[:, i]`` from the previous one: where the track leaves the
    sphere of that radius about the previous station (a sphere-segment
    intersection, the larger root). Where the track runs out the station
    continues straight along its last segment. Returns (stations (arms, k+1,
    3) including the start, ``on_track`` (arms, k+1) bool).

    Scalar Python per arm: a station's exit is nearly always on the current
    segment or the next, so a short forward scan beats a vectorised test of
    every segment (measured: ~4x)."""
    A = track.shape[0]
    k = lengths.shape[1]
    D = np.diff(track, axis=1)
    segs = np.concatenate([track[:, :-1], D, (D * D).sum(-1)[..., None]], axis=-1).tolist()
    ln = lengths.tolist()
    st = np.asarray(start, float).tolist()
    sqrt = math.sqrt
    pts, flags = [], []
    for a in range(A):
        G = segs[a]
        ns = len(G)
        cx, cy, cz = st[a]
        row = [(cx, cy, cz)]
        fl = [True]
        j, t = 0, 0.0
        ok = True
        ax, ay, az, dx, dy, dz, aa = G[-1]
        ll = sqrt(aa) or 1.0
        lx, ly, lz = dx / ll, dy / ll, dz / ll
        for L in ln[a]:
            LL = L * L
            while ok and j < ns:
                ax, ay, az, dx, dy, dz, aa = G[j]
                if aa > 1e-18:
                    wx, wy, wz = ax - cx, ay - cy, az - cz
                    bb = wx * dx + wy * dy + wz * dz
                    disc = bb * bb - aa * (wx * wx + wy * wy + wz * wz - LL)
                    if disc >= 0.0:
                        r = (-bb + sqrt(disc)) / aa
                        if t <= r <= 1.0 + 1e-9:
                            if r > 1.0:
                                r = 1.0
                            cx, cy, cz = ax + r * dx, ay + r * dy, az + r * dz
                            t = r
                            break
                j += 1
                t = 0.0
            else:
                ok = False
                cx, cy, cz = cx + L * lx, cy + L * ly, cz + L * lz
            row.append((cx, cy, cz))
            fl.append(ok)
        pts.append(row)
        flags.append(fl)
    out, on = np.array(pts), np.array(flags)
    return out, on


# ---------------------------------------------------------------------------
# The pose
# ---------------------------------------------------------------------------
@dataclass
class CreaturePose:
    """One frame's bone pose in ``rig.bone_names`` order, model-frame metres:
    ``points`` (bones, 3) heads, ``rotations`` (bones, 3, 3) deformation
    relative to the bind pose, ``scales`` (bones,) uniform -- what
    ``octopus_math.pose_octopus(..., with_scale=True)`` returns -- plus, per
    ART arm, the stretch control used and the stations' track flags."""
    points: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    stretch: np.ndarray          # (8,) the solved stretch control, 0 .. 1
    on_track: np.ndarray         # (8, joints) True where a station lies on its track
    track: np.ndarray            # (8, m, 3) the polylines walked (model frame)


def solve_stretch(rig: OctopusRig, track_length) -> np.ndarray:
    """The stretch control (0 .. 1) whose free arm is ``track_length`` long
    (exact: the rig's segment lengths are linear in it), clamped."""
    s = (np.asarray(track_length, float) - rig.free_rest) / rig.free_span
    return np.clip(s, 0.0, 1.0)


def pose_from_chain(view, rig: OctopusRig, dials: CreatureRenderSettings = NO_WIGGLE,
                    t_s: float = 0.0) -> CreaturePose:
    """The model's bone pose from one ``ChainView`` (see the module doc)."""
    J, nj = rig.junction, rig.joints
    nodes = to_model(view.nodes, view.crown, float(view.heading))       # (A, n, 3)
    Q = np.empty_like(nodes)
    Q[rig.arm_map] = nodes                                               # art order
    if dials.arm_lift:
        Q[..., 1] += dials.arm_lift * rig.node_radius
    junc = rig.crown_pts[:, J]                                           # (8, 3)
    track = np.concatenate([junc[:, None], Q[:, rig.track_k0:]], axis=1)
    tl = np.linalg.norm(np.diff(track, axis=1), axis=-1).sum(1)
    st = solve_stretch(rig, tl)
    seg = rig.seg_a[None, :] + st[:, None] * rig.seg_b[None, :]           # (8, nj-1)
    free, on = walk_stations(track, junc, seg[:, J:])
    # ONE secant correction: chords cut the track's corners, so lengths summing
    # to the track's arc length overshoot its tip by the chord deficit; shorten
    # the arm by that overshoot and walk it again (arms below stretch_max only)
    over = ~on[:, -1] & (st < 1.0)
    if over.any():
        d = np.linalg.norm(free[over, -1] - track[over, -1], axis=-1)
        st[over] = np.clip(st[over] - d / rig.free_span, 0.0, 1.0)
        seg[over] = rig.seg_a[None, :] + st[over, None] * rig.seg_b[None, :]
        free[over], on[over] = walk_stations(track[over], junc[over], seg[over, J:])
    S = np.concatenate([rig.crown_pts[:, :J], free], axis=1)             # (8, nj, 3)
    on_track = np.concatenate([np.ones((8, J), bool), on], axis=1)

    if dials.tip_curl_amplitude:
        # the distal stations swing in the deck plane on the tip-curl profile
        g0 = J + int(math.ceil(rig.tip_from * (nj - 1 - J)))
        u = np.clip((np.arange(nj - 1) - g0 + 1) / max(1, nj - 1 - g0), 0.0, 1.0)
        # each mirror pair (L_n, R_n) shares a phase, so a mirrored chain
        # stays a mirrored pose
        ph = 2.0 * math.pi * t_s / dials.tip_curl_period_s + 0.9 * (np.arange(8) % 4)
        amp = rig.side * dials.tip_curl_amplitude * rig.a_tip * np.sin(ph)  # (8,)
        ang = amp[:, None] * u[None, :] ** 2                             # (8, nj-1)
        e = np.diff(S, axis=1)
        R = axis_rotation(np.broadcast_to([0.0, 1.0, 0.0], ang.shape + (3,)), ang)
        e = np.einsum("asij,asj->asi", R, e)
        S[:, g0:] = S[:, g0 - 1:g0] + np.cumsum(e[:, g0 - 1:], axis=1)

    e = np.diff(S[:, J:], axis=1)                                        # (8, nj-1-J, 3)
    tdir = e / np.linalg.norm(e, axis=-1, keepdims=True)
    rots = np.empty((8, nj, 3, 3))
    rots[:, :J] = np.eye(3)
    steps = min_rotation(tdir[:, :-1], tdir[:, 1:])                      # (8, nj-2-J)
    R = min_rotation(rig.bind_dir, tdir[:, 0])
    rots[:, J] = R
    for g in range(steps.shape[1]):
        R = _mm(steps[:, g], R)
        rots[:, J + 1 + g] = R
    rots[:, nj - 1] = rots[:, nj - 2]                                    # the tip repeats
    if dials.twist:
        frac = np.clip((np.arange(nj) - J) / (nj - 1 - J), 0.0, 1.0)
        tang = np.concatenate([np.zeros((8, J, 3)), tdir, tdir[:, -1:]], axis=1)
        tang[:, :J] = rig.bind_dir[:, None]
        ang = (-rig.side[:, None]) * dials.twist * rig.twist_max * frac[None, :]
        rots = _mm(axis_rotation(tang, ang), rots)
    sc = 1.0 / np.sqrt(1.0 + st[:, None] * rig.scale_c[None, :])           # (8, nj)

    nb = len(rig.bone_names)
    pts = np.empty((nb, 3))
    rot = np.empty((nb, 3, 3))
    scl = np.ones(nb)
    sq = 0.0
    if dials.breathing_amplitude:
        sq = dials.breathing_amplitude * 0.5 * (
            1.0 - math.cos(2.0 * math.pi * t_s / dials.breathing_period_s))
    pts[:6] = rig.body_pts[0] + sq * (rig.body_pts[1] - rig.body_pts[0])
    rot[:6] = np.eye(3)
    o = rig.offsets[0]
    pts[o:o + 8 * nj] = S.reshape(-1, 3)
    rot[o:o + 8 * nj] = rots.reshape(-1, 3, 3)
    scl[o:o + 8 * nj] = sc.reshape(-1)
    return CreaturePose(points=pts, rotations=rot, scales=scl, stretch=st,
                        on_track=on_track, track=track)


__all__ = ["ART_SCRIPTS", "SIZE_TOLERANCE", "octopus_math", "CreatureRenderSettings",
           "creature_render_settings", "NO_WIGGLE", "match_size", "OctopusRig",
           "build_rig", "to_model", "min_rotation", "axis_rotation", "walk_stations",
           "CreaturePose", "solve_stretch", "pose_from_chain"]
