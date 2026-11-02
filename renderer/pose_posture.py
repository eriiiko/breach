"""Posture -- the pose layer's continuous body carriage (M14 experiment E4, #164).

Pure numpy, render-only, pyray-free. A POSTURE PROFILE is one row of the
zombie motion brief's §3 parameters (``docs/zombie_motion_brief_2026-11-02.md``):

    spine_pitch (deg)  knee_bend (hip drop, fraction of the leg)
    shoulder_protraction  arm_forward (deg)  arm_open  head_drop
    tempo_jitter  twitch

applied with an AMOUNT ``a`` in [0, 1] per unit (the human is ``a = 0``; until
the sim says otherwise ``a`` is 1 for a zombie). The profile x ``a`` becomes
per-bone LOCAL offsets added after the clip (after the overrides and the
additive, before FK; design §4.5 correction 3), so it rides on whatever clip
plays -- the walk, the idle, a melee one-shot -- and the post-FK twist still
runs after it.

Skeleton-agnostic: which bone moves, about which axis and by how many degrees
per unit of a parameter is data, ``[render.pose.skeletons.*.posture]``; no bone
is named here. An axis is given either in MODEL space at rest (``space =
"model"``: e.g. the lateral axis a spine bone pitches about; converted once at
load into the bone's parent frame at rest, so the offset turns WITH the torso
the clip moves) or in the bone's own frame (``"local"``: a finger's curl).

``knee_bend`` is geometry, not a degrees row: the hips drop by ``knee_bend``
x the leg length and each leg (thigh, shin, foot) is folded so the
hip-to-ankle line keeps its direction and shortens by the drop -- the feet
stay near the floor without IK. The line is assumed vertical; a leg mid-stride
(the line tilted by ``theta``) sinks its ankle by ``drop (1 - cos theta)``
(about a centimetre at the stalk values), documented, not corrected.

``tempo_jitter`` is a per-unit irregular playback rate ``1 + a J n(tau)``,
``twitch`` a small per-bone rotation ``a T deg n_c(tau)``; ``n`` is a mean of
sines on the unit's own wall clock ``tau`` with per-unit phases and
frequencies drawn from a LOCAL generator seeded by ``(seed, unit id)`` --
deterministic, zero-mean, bounded by 1, never the sim's RNG. The same
generator gives each unit small offsets in lean and arm carry (brief line 19).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .pose_clips import Skeleton, quat_mul, quat_to_mat3

# The brief's §3 parameters, in this order everywhere (profile vectors).
PARAMS = ("spine_pitch", "knee_bend", "shoulder_protraction", "arm_forward",
          "arm_open", "head_drop", "tempo_jitter", "twitch")
# The parameters a skeleton maps to bone rotations (degrees per unit).
ROTATION_PARAMS = ("spine_pitch", "shoulder_protraction", "arm_forward",
                   "arm_open", "head_drop")
SPACES = ("model", "local")
_P = {name: i for i, name in enumerate(PARAMS)}


class PostureConfigError(ValueError):
    """Raised by the posture door; re-raised by ``pose_layer`` as its own."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostureEntry:
    """Bones that turn by ``deg`` degrees per unit of their parameter about
    ``axis`` (model space at rest, or the bone's own frame)."""
    bones: Tuple[str, ...]
    axis: Tuple[float, float, float]
    deg: float
    space: str


@dataclass(frozen=True)
class KneeRule:
    hips: str                                   # the bone that drops
    legs: Tuple[Tuple[str, str, str], ...]      # (thigh, shin, foot) chains


@dataclass(frozen=True)
class PostureRule:
    """One skeleton's mapping ``[render.pose.skeletons.<s>.posture]``."""
    up: Tuple[float, float, float]
    forward: Tuple[float, float, float]
    rotations: Dict[str, Tuple[PostureEntry, ...]]   # ROTATION_PARAMS -> entries
    twitch: Tuple[PostureEntry, ...]                 # deg per unit of `twitch`
    knee: KneeRule


@dataclass(frozen=True)
class PostureSettings:
    """``[render.pose.posture]``: the profile a unit's amount scales, every
    profile row, the per-unit variation and the noise clocks."""
    profile: str
    profiles: Dict[str, np.ndarray]           # name -> (P,) in PARAMS order
    variation: Dict[str, float]               # param -> +- fraction per unit
    seed: int
    tempo_hz: Tuple[float, ...]
    twitch_hz: Tuple[float, ...]
    hz_spread: float                          # +- fraction per unit on every frequency
    desync_start: bool                        # a posed unit's loops start at a random time

    @property
    def vector(self) -> np.ndarray:
        return self.profiles[self.profile]


_POSTURE_KEYS = ("profile", "profiles", "variation", "seed", "tempo_hz",
                 "twitch_hz", "hz_spread", "desync_start")
_RULE_KEYS = ("up", "forward", "twitch", "knee_bend") + ROTATION_PARAMS
_ENTRY_KEYS = ("bones", "axis", "deg", "space")
_KNEE_KEYS = ("hips", "legs")
# per-parameter range in a profile row (spine / arm in degrees)
_RANGE = {"spine_pitch": (0.0, 90.0), "knee_bend": (0.0, 0.5),
          "shoulder_protraction": (0.0, 1.0), "arm_forward": (0.0, 120.0),
          "arm_open": (0.0, 1.0), "head_drop": (0.0, 1.0),
          "tempo_jitter": (0.0, 1.0), "twitch": (0.0, 1.0)}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _exact(d, keys: Sequence[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise PostureConfigError(f"{where} must be a table")
    for k in keys:
        if k not in d:
            raise PostureConfigError(f"{where} is missing {k!r}")
    bad = sorted(set(d) - set(keys))
    if bad:
        raise PostureConfigError(f"{where} has unknown keys {bad}")
    return d


def _vec3(v, where: str) -> Tuple[float, float, float]:
    if not isinstance(v, list) or len(v) != 3 or not all(_num(x) for x in v) \
            or float(np.linalg.norm(np.asarray(v, float))) < 1e-9:
        raise PostureConfigError(f"{where} must be a non-zero [x, y, z], got {v!r}")
    a = np.asarray(v, float)
    a = a / np.linalg.norm(a)
    return (float(a[0]), float(a[1]), float(a[2]))


def _hz(v, where: str) -> Tuple[float, ...]:
    if not isinstance(v, list) or not v or not all(_num(x) and x > 0 for x in v):
        raise PostureConfigError(f"{where} must be a non-empty list of frequencies > 0, got {v!r}")
    return tuple(float(x) for x in v)


def _entries(v, where: str) -> Tuple[PostureEntry, ...]:
    if not isinstance(v, list):
        raise PostureConfigError(f"{where} must be a list of {{bones, axis, deg, space}} rows")
    out = []
    for k, row in enumerate(v):
        w = f"{where}[{k}]"
        row = _exact(row, _ENTRY_KEYS, w)
        bones = row["bones"]
        if not isinstance(bones, list) or not bones or \
                not all(isinstance(b, str) and b for b in bones):
            raise PostureConfigError(f"{w}.bones must be a list of bone names, got {bones!r}")
        if not _num(row["deg"]) or abs(row["deg"]) > 360:
            raise PostureConfigError(f"{w}.deg must be a number in [-360, 360], got {row['deg']!r}")
        if row["space"] not in SPACES:
            raise PostureConfigError(f"{w}.space must be one of {SPACES}, got {row['space']!r}")
        out.append(PostureEntry(tuple(bones), _vec3(row["axis"], f"{w}.axis"),
                                float(row["deg"]), row["space"]))
    return tuple(out)


def parse_rule(raw, where: str) -> PostureRule:
    """A skeleton's ``posture`` table (bone names checked later, per look)."""
    d = _exact(raw, _RULE_KEYS, where)
    rotations = {p: _entries(d[p], f"{where}.{p}") for p in ROTATION_PARAMS}
    kn = _exact(d["knee_bend"], _KNEE_KEYS, f"{where}.knee_bend")
    if not isinstance(kn["hips"], str) or not kn["hips"]:
        raise PostureConfigError(f"{where}.knee_bend.hips must be a bone name, got {kn['hips']!r}")
    legs = kn["legs"]
    if not isinstance(legs, list) or not all(
            isinstance(leg, list) and len(leg) == 3 and all(isinstance(b, str) and b for b in leg)
            for leg in legs):
        raise PostureConfigError(f"{where}.knee_bend.legs must be a list of "
                                 f"[thigh, shin, foot] bone names, got {legs!r}")
    up = _vec3(d["up"], f"{where}.up")
    fwd = _vec3(d["forward"], f"{where}.forward")
    if abs(float(np.dot(up, fwd))) > 1e-3:
        raise PostureConfigError(f"{where}: up and forward must be perpendicular")
    return PostureRule(up=up, forward=fwd, rotations=rotations,
                       twitch=_entries(d["twitch"], f"{where}.twitch"),
                       knee=KneeRule(kn["hips"], tuple(tuple(leg) for leg in legs)))


def parse_settings(raw, where: str = "[render.pose.posture]") -> PostureSettings:
    d = _exact(raw, _POSTURE_KEYS, where)
    profiles_raw = d["profiles"]
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise PostureConfigError(f"{where}.profiles must hold at least one profile row")
    profiles: Dict[str, np.ndarray] = {}
    for name, row in profiles_raw.items():
        w = f"{where}.profiles.{name}"
        row = _exact(row, PARAMS, w)
        vec = np.zeros(len(PARAMS))
        for p in PARAMS:
            lo, hi = _RANGE[p]
            v = row[p]
            if not _num(v) or not lo <= v <= hi:
                raise PostureConfigError(f"{w} {p} must be a number in [{lo}, {hi}], got {v!r}")
            vec[_P[p]] = float(v)
        profiles[name] = vec
    if d["profile"] not in profiles:
        raise PostureConfigError(f"{where} profile {d['profile']!r} is not a profile row")
    var = d["variation"]
    if not isinstance(var, dict):
        raise PostureConfigError(f"{where}.variation must be a table of parameter = fraction")
    for p, v in var.items():
        if p not in PARAMS:
            raise PostureConfigError(f"{where}.variation names {p!r}, not a posture parameter")
        if not _num(v) or not 0.0 <= v < 1.0:
            raise PostureConfigError(f"{where}.variation {p} must be a number in [0, 1), got {v!r}")
    seed = d["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PostureConfigError(f"{where} seed must be an int >= 0, got {seed!r}")
    spread = d["hz_spread"]
    if not _num(spread) or not 0.0 <= spread < 1.0:
        raise PostureConfigError(f"{where} hz_spread must be a number in [0, 1), got {spread!r}")
    if not isinstance(d["desync_start"], bool):
        raise PostureConfigError(f"{where} desync_start must be a bool, got {d['desync_start']!r}")
    return PostureSettings(profile=d["profile"], profiles=profiles,
                           variation={p: float(v) for p, v in var.items()}, seed=seed,
                           tempo_hz=_hz(d["tempo_hz"], f"{where} tempo_hz"),
                           twitch_hz=_hz(d["twitch_hz"], f"{where} twitch_hz"),
                           hz_spread=float(spread), desync_start=d["desync_start"])


# ---------------------------------------------------------------------------
# Per-unit seed: phases, frequencies, variation
# ---------------------------------------------------------------------------

@dataclass
class UnitNoise:
    """One unit's draws from its own generator (``(seed, unit id)``)."""
    tempo_phase: np.ndarray        # (K_t,)
    tempo_hz: np.ndarray           # (K_t,)
    twitch_phase: np.ndarray       # (C, K_w) per twitch channel
    twitch_hz: np.ndarray          # (C, K_w)
    scale: np.ndarray              # (P,) the unit's variation factors
    start: float                   # in [0, 1): where its loops start


def unit_noise(settings: PostureSettings, uid: int, n_channels: int) -> UnitNoise:
    """The unit's draws -- a pure function of ``(settings.seed, uid)`` (and
    the twitch channel count), from a LOCAL generator: never numpy's global
    state, never the sim's."""
    rng = np.random.default_rng([settings.seed, int(uid) & 0xFFFFFFFF])
    kt, kw = len(settings.tempo_hz), len(settings.twitch_hz)
    sp = settings.hz_spread
    tempo_phase = rng.uniform(0.0, 2 * np.pi, kt)
    tempo_hz = np.asarray(settings.tempo_hz) * rng.uniform(1 - sp, 1 + sp, kt)
    twitch_phase = rng.uniform(0.0, 2 * np.pi, (n_channels, kw))
    twitch_hz = np.asarray(settings.twitch_hz)[None, :] * rng.uniform(1 - sp, 1 + sp, (n_channels, kw))
    scale = np.ones(len(PARAMS))
    u = rng.uniform(-1.0, 1.0, len(PARAMS))
    for p, frac in settings.variation.items():
        scale[_P[p]] = 1.0 + frac * u[_P[p]]
    return UnitNoise(tempo_phase, tempo_hz, twitch_phase, twitch_hz, scale,
                     float(rng.uniform(0.0, 1.0)))


def smooth_noise(phase: np.ndarray, hz: np.ndarray, tau) -> np.ndarray:
    """The mean of sines ``sin(2 pi f tau + phase)`` over the last axis:
    smooth, zero-mean over a long window, bounded by 1 in magnitude.
    ``tau`` broadcasts against ``phase[..., 0]``."""
    tau = np.asarray(tau, dtype=np.float64)[..., None]
    return np.sin(2.0 * np.pi * hz * tau + phase).mean(-1)


# ---------------------------------------------------------------------------
# Resolved on a look's skeleton
# ---------------------------------------------------------------------------

def _rest_globals(sk: Skeleton) -> np.ndarray:
    """Rest-pose model-space rotations ``(J, 3, 3)`` and positions ``(J, 3)``."""
    rot = quat_to_mat3(sk.rest_r) * sk.rest_s[:, None, :]
    g_r = np.empty_like(rot)
    g_t = np.empty((sk.n_bones, 3))
    for lvl in sk.levels:                       # load time, per depth level
        par = sk.parents[lvl]
        if par[0] < 0:
            g_r[lvl] = rot[lvl]
            g_t[lvl] = sk.rest_t[lvl]
        else:
            g_r[lvl] = np.einsum("nij,njk->nik", g_r[par], rot[lvl])
            g_t[lvl] = np.einsum("nij,nj->ni", g_r[par], sk.rest_t[lvl]) + g_t[par]
    return g_r, g_t


@dataclass(frozen=True)
class ResolvedPosture:
    """A posture rule on one look's skeleton, as dense arrays: rotation
    vectors (radians) per unit of each parameter, in the frame they are
    applied in (``pre`` = the parent frame, left-multiplied; ``post`` = the
    bone's own, right-multiplied)."""
    pre: np.ndarray            # (P, J, 3)
    post: np.ndarray           # (P, J, 3)
    twitch_pre: np.ndarray     # (C, J, 3) one channel per (entry, bone), at 1 degree
    twitch_post: np.ndarray    # (C, J, 3)
    twitch_deg: np.ndarray     # (C,)
    hips: int
    drop_dir: np.ndarray       # (3,) "down" in the hips' parent frame
    leg_len: float             # the mean thigh + shin length (model units)
    legs: np.ndarray           # (Lg, 3) bone indices (thigh, shin, foot)
    leg_a: np.ndarray          # (Lg,) thigh length
    leg_b: np.ndarray          # (Lg,) shin length
    leg_axis: np.ndarray       # (Lg, 3, 3) the flex axis in each bone's parent frame, signed

    @property
    def n_channels(self) -> int:
        return len(self.twitch_deg)


def resolve(rule: PostureRule, sk: Skeleton, where: str) -> ResolvedPosture:
    """Resolve ``rule`` on ``sk``; refuses (naming it) a bone the skeleton
    lacks and a leg that is not a parent -> child -> grandchild chain."""
    J = sk.n_bones

    def bone(name: str) -> int:
        if name not in sk.names:
            raise PostureConfigError(f"{where}: unknown bone {name!r} (not in {sk.source})")
        return sk.index(name)

    g_r, g_t = _rest_globals(sk)
    par_r = np.where(sk.parents[:, None, None] >= 0,
                     g_r[np.maximum(sk.parents, 0)], np.eye(3)[None])
    par_r = par_r / np.linalg.norm(par_r, axis=-2, keepdims=True)

    def axes(entry: PostureEntry, j: int) -> Tuple[np.ndarray, bool]:
        a = np.asarray(entry.axis, float)
        if entry.space == "model":
            return np.einsum("ji,j->i", par_r[j], a), True          # R_parent^T a
        return a, False

    P = len(PARAMS)
    pre = np.zeros((P, J, 3))
    post = np.zeros((P, J, 3))
    for p in ROTATION_PARAMS:
        for entry in rule.rotations[p]:
            for name in entry.bones:
                j = bone(name)
                ax, is_pre = axes(entry, j)
                (pre if is_pre else post)[_P[p], j] += np.radians(entry.deg) * ax
    chans = [(entry, bone(name)) for entry in rule.twitch for name in entry.bones]
    C = len(chans)
    tw_pre = np.zeros((C, J, 3))
    tw_post = np.zeros((C, J, 3))
    for c, (entry, j) in enumerate(chans):
        ax, is_pre = axes(entry, j)
        (tw_pre if is_pre else tw_post)[c, j] = np.radians(1.0) * ax

    hips = bone(rule.knee.hips)
    up = np.asarray(rule.up, float)
    fwd = np.asarray(rule.forward, float)
    lateral = np.cross(up, fwd)                  # the flex axis (model space)
    drop_dir = np.einsum("ji,j->i", par_r[hips], -up)
    legs, la, lb, laxis = [], [], [], []
    for th_n, sh_n, ft_n in rule.knee.legs:
        th, sh, ft = bone(th_n), bone(sh_n), bone(ft_n)
        if sk.parents[sh] != th or sk.parents[ft] != sh:
            raise PostureConfigError(f"{where}.knee_bend: leg [{th_n!r}, {sh_n!r}, {ft_n!r}] "
                                     "is not a thigh -> shin -> foot chain")
        # the sign that swings the knee FORWARD when the thigh turns about
        # the lateral axis at the hip (measured on the rest pose)
        knee_rel = g_t[sh] - g_t[th]
        swing = np.cross(lateral, knee_rel)
        sign = 1.0 if float(np.dot(swing, fwd)) >= 0.0 else -1.0
        ax = [np.einsum("ji,j->i", par_r[j], lateral) for j in (th, sh, ft)]
        laxis.append(np.stack([sign * ax[0], -sign * ax[1], sign * ax[2]]))
        legs.append((th, sh, ft))
        la.append(float(np.linalg.norm(g_t[sh] - g_t[th])))
        lb.append(float(np.linalg.norm(g_t[ft] - g_t[sh])))
    # no leg chains: the hips' rest height stands in for the leg
    leg_len = float(np.mean(np.add(la, lb))) if legs else abs(float(g_t[hips] @ up))
    return ResolvedPosture(
        pre=pre.astype(np.float32), post=post.astype(np.float32),
        twitch_pre=tw_pre.astype(np.float32), twitch_post=tw_post.astype(np.float32),
        twitch_deg=np.array([e.deg for e, _ in chans]), hips=hips,
        drop_dir=drop_dir.astype(np.float32), leg_len=leg_len,
        legs=np.array(legs, dtype=np.int64).reshape(-1, 3),
        leg_a=np.array(la), leg_b=np.array(lb),
        leg_axis=np.array(laxis, dtype=np.float32).reshape(-1, 3, 3))


# ---------------------------------------------------------------------------
# Apply (batched over the posed units of one look)
# ---------------------------------------------------------------------------

def rotvec_quat(v: np.ndarray) -> np.ndarray:
    """Rotation vectors ``(..., 3)`` -> unit quaternions ``(..., 4)`` xyzw."""
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * ang
    k = np.where(ang > 1e-12, np.sin(half) / np.where(ang > 1e-12, ang, 1.0), 0.5)
    return np.concatenate([v * k, np.cos(half)], axis=-1)


def _knee_angle(q: np.ndarray) -> np.ndarray:
    """The angle between a bone's +Y (its length axis, glTF / Blender
    convention) and its parent's +Y, from the local rotation."""
    x, z = q[..., 0], q[..., 2]
    return np.arccos(np.clip(1.0 - 2.0 * (x * x + z * z), -1.0, 1.0))


def _hip_angle(a, b, k):
    """Angle at the hip between the thigh and the hip -> ankle line."""
    return np.arctan2(b * np.sin(k), a + b * np.cos(k))


def apply(res: ResolvedPosture, lt: np.ndarray, lr: np.ndarray,
          params: np.ndarray, twitch_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Posture on local poses ``lt (n, J, 3)``, ``lr (n, J, 4)`` of n posed
    units. ``params (n, P)`` = profile x amount x variation (spine and arm
    in degrees); ``twitch_vals (n, C)`` = degrees per twitch channel. Returns
    new arrays."""
    pre = np.einsum("np,pjk->njk", params.astype(np.float32), res.pre)
    post = np.einsum("np,pjk->njk", params.astype(np.float32), res.post)
    if res.n_channels:
        tv = twitch_vals.astype(np.float32)
        pre = pre + np.einsum("nc,cjk->njk", tv, res.twitch_pre)
        post = post + np.einsum("nc,cjk->njk", tv, res.twitch_post)
    lt = lt.copy()
    drop = params[:, _P["knee_bend"]] * res.leg_len                     # (n,)
    lt[:, res.hips] += (drop[:, None] * res.drop_dir[None, :]).astype(lt.dtype)
    if len(res.legs):
        a, b = res.leg_a[None, :], res.leg_b[None, :]                   # (1, Lg)
        k0 = _knee_angle(lr[:, res.legs[:, 1]].astype(np.float64))      # (n, Lg)
        d0 = np.sqrt(a * a + b * b + 2 * a * b * np.cos(k0))
        d1 = np.clip(d0 - drop[:, None], abs(a - b) + 1e-6, None)
        k1 = np.arccos(np.clip((d1 * d1 - a * a - b * b) / (2 * a * b), -1.0, 1.0))
        dk = k1 - k0
        da = _hip_angle(a, b, k1) - _hip_angle(a, b, k0)
        ang = np.stack([da, dk, dk - da], axis=-1)                       # (n, Lg, 3)
        rv = ang[..., None] * res.leg_axis[None]                         # (n, Lg, 3, 3)
        pre[:, res.legs] += rv.astype(np.float32)
    q_pre = rotvec_quat(pre)
    q_post = rotvec_quat(post)
    lr = quat_mul(quat_mul(q_pre, lr), q_post).astype(lr.dtype)
    return lt, lr


__all__ = ["PARAMS", "ROTATION_PARAMS", "SPACES", "PostureConfigError",
           "PostureEntry", "KneeRule", "PostureRule", "PostureSettings",
           "ResolvedPosture", "UnitNoise", "parse_rule", "parse_settings",
           "unit_noise", "smooth_noise", "resolve", "apply", "rotvec_quat"]
