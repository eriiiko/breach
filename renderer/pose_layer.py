"""Pose layer -- THE place a 3D unit's bone pose is composed (M13, #162).

Pure numpy, render-only, pyray-free. Sources in, bone matrices out:

    per-unit PoseState --(table: state -> clip per layer)--> batched compose
      sample every layer's clip (sub-frame, on the clip's own key grid)
      -> crossfade ramp per layer (nlerp source -> target)
      -> override layers selected per bone by the MASK table (a partition)
      -> additive layer (local delta against the clip's own reference frame)
      -> posture (#164: a profile x the unit's amount as local offsets,
         renderer/pose_posture.py; units at amount 0 untouched)
      -> FK as 4x4 products, one per depth level (never a per-bone loop)
      -> procedural modifiers, post-FK (#163: the aim-vs-travel twist,
         renderer/pose_modifiers.py)
      -> global TRS (N, J, 10) and skin matrices (N, J, 4, 4)

A skinner (``renderer/pose_skinner.py``) takes a unit's global-TRS row to the
mesh; which skinner (CPU inject now, GPU later, W13.2) changes nothing here.

Design: ``docs/animation_milestones_design_2026-11-02.md`` §4.1 and §4.5 (the
spec). Rules this module keeps:

* Skeleton-agnostic: no bone is named in code. Regions are masks built from
  ``[render.pose.skeletons.*.masks]`` subtree / minus rules over bone names,
  validated against each look's skeleton at load.
* Playback ``rate`` is data; the default 0.51 reproduces the speed raylib's
  integer-frame player gave every clip before (§4.5 correction 1).
* Additive delta = ``inv(ref) * clip`` in local space, ``ref`` = the clip's
  own reference frame (frame 0 by default), never the bind rest
  (correction 3); translation ``T + w (T_clip - T_ref)``.
* Animation never writes sim state: nothing here sees a ``Unit``; per-unit
  state is keyed by the renderer, which reads units read-only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import (Any, Deque, Dict, List, Mapping, Optional, Protocol,
                    Sequence, Tuple)

import numpy as np

from .pose_clips import (ROOT_MOTION_MODES, ClipBank, ClipRow, Skeleton,
                         build_clip_bank, mat3_to_quat, quat_conj, quat_mul,
                         quat_nlerp, read_glb, read_skeleton, trs_matrices)

GLOBAL_TRS_WIDTH = 10          # translation 3, rotation 4 (xyzw), scale 3
NO_CLIP = -1


class PoseConfigError(ValueError):
    """A ``[render.pose]`` setting the pose layer refuses (names the key)."""


# ---------------------------------------------------------------------------
# Settings ([render.pose])
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskRule:
    """A body region: every bone under ``subtree`` roots, minus every bone
    under ``minus`` roots."""
    subtree: Tuple[str, ...]
    minus: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TwistRule:
    """The aim-vs-travel twist's bones on one skeleton (#163): ``lower``'s
    subtree yaws toward travel; ``counter`` (a chain inside it, root-most
    first) turns the torso back, each bone by its ``share`` of the twist
    (the shares sum to 1, so the last bone's subtree keeps the facing)."""
    lower: str
    counter: Tuple[str, ...]
    shares: Tuple[float, ...]
    # #184: the vertical line the lower body yaws about: "bone" = through the
    # lower bone's position this frame (#163); "origin" = through the model's
    # origin, the unit's own position -- a FIXED line, so a clip whose hips
    # sway does not slide its planted feet when the legs are twisted
    pivot: str = "bone"


@dataclass(frozen=True)
class FireSettings:
    """``[render.pose.fire]`` (#163): a launch event plays ``shot_clip`` from
    its start on ``shot_layer`` (retriggered by every launch) and holds the
    ``overlay`` row on its layers until ``aim_hold_s`` after the last one."""
    overlay: str
    aim_hold_s: float
    shot_clip: str
    shot_layer: str


@dataclass(frozen=True)
class HitSettings:
    """``[render.pose.hit]`` (#163): a hit of one of ``causes`` plays ``clip``
    as an ADDITIVE over ``masks``, ramped in / out, its weight
    ``weight_min + (weight_max - weight_min) * clamp(damage / damage_full)``."""
    clip: str
    masks: Tuple[str, ...]
    causes: Tuple[str, ...]
    ramp_in_s: float
    ramp_out_s: float
    weight_min: float
    weight_max: float
    damage_full: float


@dataclass(frozen=True)
class StaggerSettings:
    """``[render.pose.stagger]`` (#185): a ``StaggerEvent`` (the sim's sway
    overflowed: the unit is pushed) plays a flinch-style ADDITIVE over
    ``masks``, ramped in / out, at ``weight``, from the side the push came
    from -- the skeleton's ``hit_directions`` clips where it has them, else
    ``clip``. It shares the flinch's one additive slot: the latest of a hit
    and a stagger plays."""
    clip: str
    masks: Tuple[str, ...]
    ramp_in_s: float
    ramp_out_s: float
    weight: float


@dataclass(frozen=True)
class AimTravelSettings:
    """``[render.pose.aim_travel]`` (#163, design §4.5 correction 7), angles
    in radians: the legs yaw toward travel within ``+-theta``; past
    ``theta + hysteresis`` they backpedal (clip rate negative, yaw
    reflected) until back under ``theta - hysteresis``. The travel angle is
    low-passed with time constant ``smooth_s``; the yaw moves at most
    ``twist_rate`` per second."""
    theta: float
    hysteresis: float
    smooth_s: float
    twist_rate: float


@dataclass(frozen=True)
class MeleeSettings:
    """``[render.pose.melee]`` (#164): a hit of one of ``causes`` credited to
    a unit whose role has the melee reaction makes THAT unit (the attacker)
    play one of ``clips`` (in turn) as a one-shot on ``layers``; while it
    plays, the posture parameters in ``suppress`` fade out by its weight
    (an arm carried forward would lift a punching arm over the head)."""
    clips: Tuple[str, ...]
    layers: Tuple[str, ...]
    causes: Tuple[str, ...]
    suppress: Tuple[str, ...]


@dataclass(frozen=True)
class RoleSettings:
    """``[render.pose.roles.<role>]`` (#164): the reactions a unit of the
    role plays (of ``REACTIONS``) and its posture amount ``a`` in [0, 1]."""
    reactions: Tuple[str, ...]
    posture: float


@dataclass(frozen=True)
class TimedShotSettings:
    """``[render.pose.reload]`` / ``[render.pose.throw]`` (#184): a one-shot
    ``clip`` on ``layers``, time-scaled to the event's length in sim ticks
    -- the reload's whole clip over its ticks; the throw so that its
    ``release_s`` (clip seconds) lands at the end of the wind-up.
    ``clip`` :data:`ABSENT_CLIP` = the reaction does nothing (no clip yet)."""
    clip: str
    layers: Tuple[str, ...]
    release_s: float = 0.0


REACTIONS = ("fire", "hit", "melee", "reload", "throw", "stagger")

# A ``clip_names`` value meaning "this skeleton has no such clip" (#179): a
# state falls back as for any missing clip; an overlay leaves that layer
# alone; the shot / flinch / strike that needs it does nothing on the look.
ABSENT_CLIP = ""


@dataclass(frozen=True)
class SkeletonRole:
    """``[render.pose.skeletons.<s>.roles.<role>]`` (#179): a role's carriage
    on looks of ONE skeleton -- its posture amount there (replacing the
    role's global ``posture``) and state rows that replace the global ones
    for units of the role (e.g. a zombie walking the skeleton's own hunch
    clip instead of the human walk + posture)."""
    posture: float
    states: Dict[str, Dict[str, str]]


@dataclass(frozen=True)
class SkeletonPlayback:
    """How ONE skeleton's looks play the shared tables (#179):

    * ``clip_names``: the canonical clip names the global tables use (the
      Quaternius pack's) -> this skeleton's own clip names; a name that is
      no key is looked up as it is. :data:`ABSENT_CLIP` = no such clip.
    * ``rate``: clip seconds per wall second for this skeleton, replacing
      the global ``rate`` (None = the global one).
    * ``clips``: per-clip rows keyed by THIS skeleton's clip names,
      replacing the global ``[render.pose.clips]`` (None = the global rows).
    * ``roles``: per-role overrides on this skeleton (:class:`SkeletonRole`).

    #184 adds:

    * ``states``: state rows keyed by THIS skeleton's clip names, for every
      role on its looks -- replacing a global row of the same name, or a
      state of the skeleton's own (a strafe) the global table lacks. A
      role's own row (``roles``) still wins.
    * ``locomotion``: the gait / direction / derived-rate table
      (``pose_locomotion.LocomotionRule``); None = the renderer's idle /
      walk and the two-sector twist, as before.
    * ``hit_directions``: per-direction flinch clips, ``(clip, from)`` with
      ``from`` the direction the hit comes from relative to the facing
      (radians, + = the unit's left); empty = the single ``[render.pose.hit]``
      clip.
    * ``sockets``: socket name -> bone name (a held prop's hand).
    """
    clip_names: Dict[str, str] = field(default_factory=dict)
    rate: Optional[float] = None
    clips: Optional[Dict[str, ClipRow]] = None
    roles: Dict[str, SkeletonRole] = field(default_factory=dict)
    states: Dict[str, Dict[str, str]] = field(default_factory=dict)
    locomotion: Optional[Any] = None
    hit_directions: Tuple[Tuple[str, float], ...] = ()
    sockets: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseSettings:
    enabled: bool
    rate: float                                   # global playback rate
    crossfade_s: float                            # clip-change ramp, seconds
    layers: Tuple[str, ...]                       # override layers, compose order
    fallback_state: str                           # state a missing clip falls back to
    event_queue: int                              # per-unit event queue length
    skeletons: Tuple[Tuple[str, Dict[str, MaskRule]], ...]
    states: Dict[str, Dict[str, str]]             # state -> layer -> clip name
    clips: Dict[str, ClipRow]                     # clip name -> playback row
    overlays: Dict[str, Dict[str, str]] = field(default_factory=dict)  # name -> layer -> clip
    fire: Optional[FireSettings] = None
    hit: Optional[HitSettings] = None
    aim_travel: Optional[AimTravelSettings] = None
    twists: Dict[str, TwistRule] = field(default_factory=dict)  # skeleton table -> twist
    # #164: posture (pose_posture.PostureSettings / PostureRule per skeleton
    # table), the melee one-shot and the per-role reactions + posture amount
    posture: Optional[Any] = None
    postures: Dict[str, Any] = field(default_factory=dict)
    melee: Optional[MeleeSettings] = None
    roles: Dict[str, RoleSettings] = field(default_factory=dict)
    # #179: per-skeleton clip names, rate, clip rows and role overrides
    # (a skeleton table without them plays the global ones)
    playback: Dict[str, SkeletonPlayback] = field(default_factory=dict)
    # #184: the reload and throw one-shots (clip "" = off)
    reload: Optional[TimedShotSettings] = None
    throw: Optional[TimedShotSettings] = None
    # #185: the push reaction (a StaggerEvent)
    stagger: Optional[StaggerSettings] = None


def _as_dict(obj) -> Dict[str, Any]:
    """A config table as a plain dict (``CFG`` Namespaces or dicts)."""
    if isinstance(obj, Mapping):
        return {k: _as_dict(v) if _is_table(v) else v for k, v in obj.items()}
    return {k: _as_dict(v) if _is_table(v) else v for k, v in vars(obj).items()}


def _is_table(v) -> bool:
    return isinstance(v, Mapping) or (hasattr(v, "__dict__")
                                      and not isinstance(v, (str, bytes, int, float, list)))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _names(value, where: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise PoseConfigError(f"{where} must be a list of bone names, got {value!r}")
    return tuple(value)


_TOP_KEYS = ("enabled", "rate", "crossfade_s", "layers", "fallback_state",
             "event_queue", "skeletons", "states", "clips", "overlays", "fire",
             "hit", "aim_travel", "posture", "melee", "roles", "reload", "throw",
             "stagger")
_MELEE_KEYS = ("clips", "layers", "causes", "suppress")
_RELOAD_KEYS = ("clip", "layers")
_THROW_KEYS = ("clip", "layers", "release_s")
_HIT_DIR_KEYS = ("clip", "from_deg")
_ROLE_KEYS = ("reactions", "posture")
_CLIP_KEYS = ("loop", "rate", "root_motion", "additive_ref")
_FIRE_KEYS = ("overlay", "aim_hold_s", "shot_clip", "shot_layer")
_HIT_KEYS = ("clip", "masks", "causes", "ramp_in_s", "ramp_out_s", "weight_min",
             "weight_max", "damage_full")
_STAGGER_KEYS = ("clip", "masks", "ramp_in_s", "ramp_out_s", "weight")
_AIM_TRAVEL_KEYS = ("theta_deg", "hysteresis_deg", "smooth_s", "twist_rate_deg_s")
_TWIST_KEYS = ("lower", "counter", "shares")
TWIST_PIVOTS = ("bone", "origin")                  # #184: [..twist] pivot (optional)
_SKELETON_KEYS = ("masks", "twist", "posture")
# #179: optional per-skeleton playback keys; absent = the global table
# (#184: states, locomotion, hit_directions, sockets)
_SKELETON_PLAYBACK_KEYS = ("clip_names", "rate", "clips", "roles", "states",
                           "locomotion", "hit_directions", "sockets")
_SKELETON_ROLE_KEYS = ("posture", "states")


def _parse_clip_rows(raw, where: str) -> Dict[str, ClipRow]:
    """A table of per-clip playback rows (``[render.pose.clips]``, or a
    skeleton's own ``clips``). A row's keys are optional (``ClipRow``'s
    defaults); an unknown or mistyped one is refused, naming it."""
    if not isinstance(raw, dict):
        raise PoseConfigError(f"{where} must be a table of clip rows")
    clips: Dict[str, ClipRow] = {}
    for c_name, row in raw.items():
        w = f"{where} {c_name}"
        if not isinstance(row, dict):
            raise PoseConfigError(f"{w} must be a table")
        bad = sorted(set(row) - set(_CLIP_KEYS))
        if bad:
            raise PoseConfigError(f"{w} has unknown keys {bad}")
        loop = row.get("loop", True)
        rate = row.get("rate", 1.0)
        rm = row.get("root_motion", ClipRow().root_motion)
        ref = row.get("additive_ref", 0)
        if not isinstance(loop, bool):
            raise PoseConfigError(f"{w} loop must be a bool, got {loop!r}")
        if not _num(rate) or not rate > 0:
            raise PoseConfigError(f"{w} rate must be a number > 0, got {rate!r}")
        if rm not in ROOT_MOTION_MODES:
            raise PoseConfigError(f"{w} root_motion must be one of "
                                  f"{ROOT_MOTION_MODES}, got {rm!r}")
        if not isinstance(ref, int) or isinstance(ref, bool) or ref < 0:
            raise PoseConfigError(f"{w} additive_ref must be an int >= 0, got {ref!r}")
        clips[c_name] = ClipRow(loop=loop, rate=float(rate), root_motion=rm,
                                additive_ref=ref)
    return clips


def _state_row(row, layers: Sequence[str], where: str) -> Dict[str, str]:
    if not isinstance(row, dict) or set(row) != set(layers) or \
            not all(isinstance(v, str) and v for v in row.values()):
        raise PoseConfigError(f"{where} must name one clip for each layer "
                              f"{list(layers)}, got {row!r}")
    return dict(row)


def _parse_playback(sk_name: str, sk: Dict[str, Any], layers: Sequence[str],
                    canonical: Sequence[str], roles: Mapping[str, Any],
                    states: Mapping[str, Any]) -> SkeletonPlayback:
    """A skeleton table's optional playback keys (#179). ``canonical`` =
    every clip name the global tables use: a ``clip_names`` key outside it
    is refused (a typo would map nothing)."""
    where = f"[render.pose.skeletons.{sk_name}]"
    names_raw = sk.get("clip_names", {})
    if not isinstance(names_raw, dict) or not all(
            isinstance(v, str) for v in names_raw.values()):
        raise PoseConfigError(f"{where} clip_names must be a table of canonical "
                              f"name = this skeleton's clip name, got {names_raw!r}")
    bad = sorted(set(names_raw) - set(canonical))
    if bad:
        raise PoseConfigError(f"{where} clip_names {bad} are not clip names any "
                              f"[render.pose] table uses ({sorted(set(canonical))})")
    rate = sk.get("rate")
    if rate is not None:
        rate = _number(rate, f"{where} rate", 0.0, lo_open=True)
    clips = (_parse_clip_rows(sk["clips"], f"{where} clips")
             if "clips" in sk else None)
    # #184: the skeleton's own state rows (its own clip names)
    own_raw = sk.get("states", {})
    if not isinstance(own_raw, dict):
        raise PoseConfigError(f"{where} states must be a table of state rows")
    own_states = {n: _state_row(v, layers, f"{where} states.{n}")
                  for n, v in own_raw.items()}
    all_states = set(states) | set(own_states)
    roles_raw = sk.get("roles", {})
    if not isinstance(roles_raw, dict):
        raise PoseConfigError(f"{where} roles must be a table of role rows")
    sroles: Dict[str, SkeletonRole] = {}
    for role, row in roles_raw.items():
        rw = f"{where} roles.{role}"
        if role not in roles:
            raise PoseConfigError(f"{rw}: {role!r} is not a [render.pose.roles] role "
                                  f"{sorted(roles)}")
        r = _exact_table(row, _SKELETON_ROLE_KEYS, rw)
        st = r["states"]
        if not isinstance(st, dict):
            raise PoseConfigError(f"{rw} states must be a table of state rows")
        for st_name in st:
            if st_name not in all_states:
                raise PoseConfigError(f"{rw} states.{st_name}: not a [render.pose.states] "
                                      f"state nor one of the skeleton's own "
                                      f"{sorted(all_states)}")
        sroles[role] = SkeletonRole(
            posture=_number(r["posture"], f"{rw} posture", 0.0, hi=1.0),
            states={n: _state_row(v, layers, f"{rw} states.{n}") for n, v in st.items()})
    loco = None
    if "locomotion" in sk:
        from .pose_locomotion import LocomotionConfigError, parse_locomotion
        try:
            loco = parse_locomotion(sk["locomotion"], f"{where} locomotion", list(roles),
                                    sorted(all_states))
        except LocomotionConfigError as exc:
            raise PoseConfigError(str(exc)) from None
    hd_raw = sk.get("hit_directions", [])
    if not isinstance(hd_raw, list):
        raise PoseConfigError(f"{where} hit_directions must be a list of rows")
    hit_dirs = []
    for i, row in enumerate(hd_raw):
        hw = f"{where} hit_directions[{i}]"
        r = _exact_table(row, _HIT_DIR_KEYS, hw)
        f_deg = _number(r["from_deg"], f"{hw} from_deg", -180.0, hi=180.0)
        hit_dirs.append((_name(r["clip"], f"{hw} clip"), float(np.radians(f_deg))))
    sock_raw = sk.get("sockets", {})
    if not isinstance(sock_raw, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) and v for k, v in sock_raw.items()):
        raise PoseConfigError(f"{where} sockets must be a table of socket = bone name, "
                              f"got {sock_raw!r}")
    return SkeletonPlayback(clip_names=dict(names_raw), rate=rate, clips=clips,
                            roles=sroles, states=own_states, locomotion=loco,
                            hit_directions=tuple(hit_dirs), sockets=dict(sock_raw))


def _exact_table(d, keys: Sequence[str], where: str) -> Dict[str, Any]:
    """``d`` as a table holding exactly ``keys`` (missing / unknown named)."""
    if not isinstance(d, dict):
        raise PoseConfigError(f"{where} must be a table")
    for key in keys:
        if key not in d:
            raise PoseConfigError(f"{where} is missing {key!r}")
    unknown = sorted(set(d) - set(keys))
    if unknown:
        raise PoseConfigError(f"{where} has unknown keys {unknown}")
    return d


def _number(v, where: str, lo: float, lo_open: bool = False,
            hi: Optional[float] = None) -> float:
    ok = _num(v) and (v > lo if lo_open else v >= lo) and (hi is None or v <= hi)
    if not ok:
        rng = f"{'>' if lo_open else '>='} {lo}" + (f" and <= {hi}" if hi is not None else "")
        raise PoseConfigError(f"{where} must be a number {rng}, got {v!r}")
    return float(v)


def _name(v, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise PoseConfigError(f"{where} must be a non-empty string, got {v!r}")
    return v


def _parse_twist(raw, where: str) -> TwistRule:
    raw = dict(raw) if isinstance(raw, dict) else raw
    pivot = raw.pop("pivot", "bone") if isinstance(raw, dict) else "bone"
    if pivot not in TWIST_PIVOTS:
        raise PoseConfigError(f"{where}.pivot must be one of {TWIST_PIVOTS}, got {pivot!r}")
    t = _exact_table(raw, _TWIST_KEYS, where)
    lower = _name(t["lower"], f"{where}.lower")
    counter = _names(t["counter"], f"{where}.counter")
    shares = t["shares"]
    if not isinstance(shares, list) or len(shares) != len(counter) or \
            not all(_num(s) and s > 0 for s in shares):
        raise PoseConfigError(f"{where}.shares must be one number > 0 per counter "
                              f"bone, got {shares!r}")
    if abs(sum(shares) - 1.0) > 1e-6:
        raise PoseConfigError(f"{where}.shares must sum to 1 (the torso ends on the "
                              f"facing), got {sum(shares)!r}")
    return TwistRule(lower=lower, counter=counter, shares=tuple(float(s) for s in shares),
                     pivot=pivot)


def pose_settings(cfg) -> PoseSettings:
    """``[render.pose]`` from a loaded config (``CFG``) or a plain dict holding
    a ``render.pose`` table. Every key is required; a missing, unknown or
    mistyped one raises :class:`PoseConfigError` naming it. Bone names are
    checked against a real skeleton later (:meth:`MaskTable.from_rules`)."""
    if isinstance(cfg, Mapping):
        section = cfg.get("render", {}).get("pose")
    else:
        section = getattr(getattr(cfg, "render", None), "pose", None)
    if section is None:
        raise PoseConfigError("config.toml has no [render.pose] section")
    d = _as_dict(section)
    for key in _TOP_KEYS:
        if key not in d:
            raise PoseConfigError(f"[render.pose] is missing {key!r}")
    unknown = sorted(set(d) - set(_TOP_KEYS))
    if unknown:
        raise PoseConfigError(f"[render.pose] has unknown keys {unknown}")
    if not isinstance(d["enabled"], bool):
        raise PoseConfigError(f"[render.pose] enabled must be a bool, got {d['enabled']!r}")
    if not _num(d["rate"]) or not d["rate"] > 0:
        raise PoseConfigError(f"[render.pose] rate must be a number > 0, got {d['rate']!r}")
    if not _num(d["crossfade_s"]) or d["crossfade_s"] < 0:
        raise PoseConfigError("[render.pose] crossfade_s must be a number >= 0, "
                              f"got {d['crossfade_s']!r}")
    if not isinstance(d["event_queue"], int) or isinstance(d["event_queue"], bool) \
            or d["event_queue"] < 1:
        raise PoseConfigError("[render.pose] event_queue must be an int >= 1, "
                              f"got {d['event_queue']!r}")
    layers = d["layers"]
    if not isinstance(layers, list) or not layers or \
            not all(isinstance(x, str) and x for x in layers) or len(set(layers)) != len(layers):
        raise PoseConfigError("[render.pose] layers must be a non-empty list of "
                              f"distinct mask names, got {layers!r}")
    layers = tuple(layers)

    skel_tables = d["skeletons"]
    if not isinstance(skel_tables, dict) or not skel_tables:
        raise PoseConfigError("[render.pose.skeletons] must hold at least one skeleton table")
    from .pose_posture import PostureConfigError, parse_rule, parse_settings
    skeletons = []
    twists: Dict[str, TwistRule] = {}
    postures: Dict[str, Any] = {}
    for sk_name, sk in skel_tables.items():
        where = f"[render.pose.skeletons.{sk_name}]"
        if not isinstance(sk, dict) or not set(_SKELETON_KEYS) <= set(sk) \
                or set(sk) - set(_SKELETON_KEYS) - set(_SKELETON_PLAYBACK_KEYS):
            raise PoseConfigError(f"{where} must hold a 'masks', a 'twist' and a "
                                  "'posture' table, and optionally "
                                  f"{list(_SKELETON_PLAYBACK_KEYS)}; got "
                                  f"{sorted(sk) if isinstance(sk, dict) else sk!r}")
        twists[sk_name] = _parse_twist(sk["twist"], f"{where} twist")
        try:
            postures[sk_name] = parse_rule(sk["posture"], f"{where} posture")
        except PostureConfigError as exc:
            raise PoseConfigError(str(exc)) from None
        masks = {}
        for m_name, rule in sk["masks"].items():
            mw = f"{where} masks.{m_name}"
            if not isinstance(rule, dict) or "subtree" not in rule \
                    or set(rule) - {"subtree", "minus"}:
                raise PoseConfigError(f"{mw} must be {{subtree = [...], minus = [...]}}")
            masks[m_name] = MaskRule(_names(rule["subtree"], f"{mw}.subtree"),
                                     _names(rule.get("minus", []), f"{mw}.minus")
                                     if rule.get("minus") else ())
        for layer in layers:
            if layer not in masks:
                raise PoseConfigError(f"{where} has no mask for layer {layer!r}")
        skeletons.append((sk_name, masks))

    states_raw = d["states"]
    if not isinstance(states_raw, dict) or not states_raw:
        raise PoseConfigError("[render.pose.states] must hold at least one state")
    states: Dict[str, Dict[str, str]] = {}
    for st_name, row in states_raw.items():
        states[st_name] = _state_row(row, layers, f"[render.pose.states] {st_name}")
    fallback = d["fallback_state"]
    if fallback not in states:
        raise PoseConfigError(f"[render.pose] fallback_state {fallback!r} is not a state")

    clips = _parse_clip_rows(d["clips"], "[render.pose.clips]")

    # overlays: partial state rows (a subset of the layers), #163
    ov_raw = d["overlays"]
    if not isinstance(ov_raw, dict):
        raise PoseConfigError("[render.pose.overlays] must be a table of rows")
    overlays: Dict[str, Dict[str, str]] = {}
    for ov_name, row in ov_raw.items():
        where = f"[render.pose.overlays] {ov_name}"
        if not isinstance(row, dict) or not row or set(row) - set(layers) or \
                not all(isinstance(v, str) and v for v in row.values()):
            raise PoseConfigError(f"{where} must name a clip for one or more of the "
                                  f"layers {list(layers)}, got {row!r}")
        overlays[ov_name] = dict(row)

    f = _exact_table(d["fire"], _FIRE_KEYS, "[render.pose.fire]")
    fire = FireSettings(
        overlay=_name(f["overlay"], "[render.pose.fire] overlay"),
        aim_hold_s=_number(f["aim_hold_s"], "[render.pose.fire] aim_hold_s", 0.0,
                           lo_open=True),
        shot_clip=_name(f["shot_clip"], "[render.pose.fire] shot_clip"),
        shot_layer=_name(f["shot_layer"], "[render.pose.fire] shot_layer"))
    if fire.overlay not in overlays:
        raise PoseConfigError(f"[render.pose.fire] overlay {fire.overlay!r} is not "
                              "a [render.pose.overlays] row")
    if fire.shot_layer not in layers:
        raise PoseConfigError(f"[render.pose.fire] shot_layer {fire.shot_layer!r} "
                              f"is not one of the layers {list(layers)}")

    h = _exact_table(d["hit"], _HIT_KEYS, "[render.pose.hit]")
    hw = "[render.pose.hit]"
    hit_masks = _names(h["masks"], f"{hw} masks")
    for sk_name, masks in skeletons:
        missing = [m for m in hit_masks if m not in masks]
        if missing:
            raise PoseConfigError(f"{hw} masks {missing} are not masks of "
                                  f"[render.pose.skeletons.{sk_name}]")
    causes = h["causes"]
    if not isinstance(causes, list) or not all(isinstance(c, str) and c for c in causes):
        raise PoseConfigError(f"{hw} causes must be a list of cause names, got {causes!r}")
    w_min = _number(h["weight_min"], f"{hw} weight_min", 0.0, hi=1.0)
    w_max = _number(h["weight_max"], f"{hw} weight_max", 0.0, hi=1.0)
    if w_max < w_min:
        raise PoseConfigError(f"{hw} weight_max ({w_max}) is below weight_min ({w_min})")
    hit = HitSettings(
        clip=_name(h["clip"], f"{hw} clip"), masks=hit_masks, causes=tuple(causes),
        ramp_in_s=_number(h["ramp_in_s"], f"{hw} ramp_in_s", 0.0),
        ramp_out_s=_number(h["ramp_out_s"], f"{hw} ramp_out_s", 0.0),
        weight_min=w_min, weight_max=w_max,
        damage_full=_number(h["damage_full"], f"{hw} damage_full", 0.0, lo_open=True))

    g = _exact_table(d["stagger"], _STAGGER_KEYS, "[render.pose.stagger]")
    gw = "[render.pose.stagger]"
    st_masks = _names(g["masks"], f"{gw} masks")
    for sk_name, masks in skeletons:
        missing = [m for m in st_masks if m not in masks]
        if missing:
            raise PoseConfigError(f"{gw} masks {missing} are not masks of "
                                  f"[render.pose.skeletons.{sk_name}]")
    stagger = StaggerSettings(
        clip=_name(g["clip"], f"{gw} clip"), masks=st_masks,
        ramp_in_s=_number(g["ramp_in_s"], f"{gw} ramp_in_s", 0.0),
        ramp_out_s=_number(g["ramp_out_s"], f"{gw} ramp_out_s", 0.0),
        weight=_number(g["weight"], f"{gw} weight", 0.0, hi=1.0))

    a = _exact_table(d["aim_travel"], _AIM_TRAVEL_KEYS, "[render.pose.aim_travel]")
    aw = "[render.pose.aim_travel]"
    theta = _number(a["theta_deg"], f"{aw} theta_deg", 0.0, lo_open=True, hi=180.0)
    hyst = _number(a["hysteresis_deg"], f"{aw} hysteresis_deg", 0.0)
    if hyst >= theta or theta + hyst > 180.0:
        raise PoseConfigError(f"{aw} hysteresis_deg must be below theta_deg and "
                              f"theta_deg + hysteresis_deg <= 180, got {hyst!r}")
    aim_travel = AimTravelSettings(
        theta=np.radians(theta), hysteresis=np.radians(hyst),
        smooth_s=_number(a["smooth_s"], f"{aw} smooth_s", 0.0),
        twist_rate=np.radians(_number(a["twist_rate_deg_s"], f"{aw} twist_rate_deg_s",
                                      0.0, lo_open=True)))

    try:
        posture = parse_settings(d["posture"])
    except PostureConfigError as exc:
        raise PoseConfigError(str(exc)) from None

    mw = "[render.pose.melee]"
    m = _exact_table(d["melee"], _MELEE_KEYS, mw)
    m_clips = _names(m["clips"], f"{mw} clips")
    m_layers = _names(m["layers"], f"{mw} layers")
    bad = [x for x in m_layers if x not in layers]
    if bad:
        raise PoseConfigError(f"{mw} layers {bad} are not layers {list(layers)}")
    m_causes = m["causes"]
    if not isinstance(m_causes, list) or not m_causes or \
            not all(isinstance(c, str) and c for c in m_causes):
        raise PoseConfigError(f"{mw} causes must be a non-empty list of cause names, "
                              f"got {m_causes!r}")
    from .pose_posture import PARAMS
    sup = m["suppress"]
    if not isinstance(sup, list) or not all(isinstance(p, str) for p in sup):
        raise PoseConfigError(f"{mw} suppress must be a list of posture parameters, got {sup!r}")
    bad = [p for p in sup if p not in PARAMS]
    if bad:
        raise PoseConfigError(f"{mw} suppress {bad} are not posture parameters {list(PARAMS)}")
    melee = MeleeSettings(clips=m_clips, layers=m_layers, causes=tuple(m_causes),
                          suppress=tuple(sup))

    rw = "[render.pose.roles]"
    roles_raw = d["roles"]
    if not isinstance(roles_raw, dict) or not roles_raw:
        raise PoseConfigError(f"{rw} must hold at least one role row")
    roles: Dict[str, RoleSettings] = {}
    for role, row in roles_raw.items():
        r = _exact_table(row, _ROLE_KEYS, f"{rw} {role}")
        rx = r["reactions"]
        if not isinstance(rx, list) or not all(isinstance(x, str) for x in rx):
            raise PoseConfigError(f"{rw} {role} reactions must be a list of {list(REACTIONS)}, "
                                  f"got {rx!r}")
        bad = [x for x in rx if x not in REACTIONS]
        if bad:
            raise PoseConfigError(f"{rw} {role} reactions {bad} are not reactions "
                                  f"{list(REACTIONS)}")
        roles[role] = RoleSettings(
            reactions=tuple(rx),
            posture=_number(r["posture"], f"{rw} {role} posture", 0.0, hi=1.0))

    # #184: reload / throw one-shots (clip "" = off until the clip exists)
    def timed(key: str, keys: Sequence[str]) -> TimedShotSettings:
        tw = f"[render.pose.{key}]"
        t = _exact_table(d[key], keys, tw)
        clip = t["clip"]
        if not isinstance(clip, str):
            raise PoseConfigError(f"{tw} clip must be a clip name or \"\" (none yet), "
                                  f"got {clip!r}")
        t_layers = _names(t["layers"], f"{tw} layers")
        bad = [x for x in t_layers if x not in layers]
        if bad:
            raise PoseConfigError(f"{tw} layers {bad} are not layers {list(layers)}")
        rel = _number(t["release_s"], f"{tw} release_s", 0.0) if "release_s" in keys else 0.0
        return TimedShotSettings(clip=clip, layers=t_layers, release_s=rel)

    reload = timed("reload", _RELOAD_KEYS)
    throw = timed("throw", _THROW_KEYS)

    # #179: per-skeleton playback (clip names, rate, rows, role overrides)
    canonical = sorted(({c for row in states.values() for c in row.values()}
                        | {c for row in overlays.values() for c in row.values()}
                        | {fire.shot_clip, hit.clip, stagger.clip} | set(melee.clips)
                        | {reload.clip, throw.clip}) - {ABSENT_CLIP})
    playback = {sk_name: _parse_playback(sk_name, sk, layers, canonical, roles, states)
                for sk_name, sk in skel_tables.items()}

    return PoseSettings(enabled=d["enabled"], rate=float(d["rate"]),
                        crossfade_s=float(d["crossfade_s"]), layers=layers,
                        fallback_state=fallback, event_queue=d["event_queue"],
                        skeletons=tuple(skeletons), states=states, clips=clips,
                        overlays=overlays, fire=fire, hit=hit, aim_travel=aim_travel,
                        twists=twists, posture=posture, postures=postures,
                        melee=melee, roles=roles, playback=playback,
                        reload=reload, throw=throw, stagger=stagger)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaskTable:
    """Named body regions over one skeleton. ``weights[m]`` is mask ``m``'s
    per-bone weight (0/1); ``owner[j]`` the mask that owns bone ``j`` (the
    masks PARTITION the skeleton)."""
    names: Tuple[str, ...]
    weights: np.ndarray            # (M, J) float32
    owner: np.ndarray              # (J,) int

    def index(self, name: str) -> int:
        return self.names.index(name)

    @classmethod
    def from_rules(cls, skeleton: Skeleton, rules: Mapping[str, MaskRule],
                   where: str = "masks") -> "MaskTable":
        """Resolve subtree/minus rules on ``skeleton``. Refuses (naming it) a
        bone the skeleton lacks, a bone in two masks and a bone in none."""
        J = skeleton.n_bones
        names = tuple(rules)
        weights = np.zeros((len(names), J), dtype=np.float32)
        for m, name in enumerate(names):
            rule = rules[name]
            inside = np.zeros(J, dtype=bool)
            for bone in rule.subtree:
                if bone not in skeleton.names:
                    raise PoseConfigError(f"{where}.{name}: unknown bone {bone!r} "
                                          f"(not in {skeleton.source})")
                inside |= skeleton.subtree(skeleton.index(bone))
            for bone in rule.minus:
                if bone not in skeleton.names:
                    raise PoseConfigError(f"{where}.{name}: unknown bone {bone!r} "
                                          f"(not in {skeleton.source})")
                inside &= ~skeleton.subtree(skeleton.index(bone))
            weights[m] = inside
        count = weights.sum(0)
        if (count > 1).any():
            twice = [skeleton.names[j] for j in np.flatnonzero(count > 1)]
            raise PoseConfigError(f"{where}: bones in more than one mask: {twice}")
        if (count < 1).any():
            none = [skeleton.names[j] for j in np.flatnonzero(count < 1)]
            raise PoseConfigError(f"{where}: bones in no mask: {none}")
        return cls(names=names, weights=weights,
                   owner=np.argmax(weights, axis=0).astype(np.int64))


def _glb_skeleton(path) -> Skeleton:
    """A look file's skeleton alone (load time: picks its skeleton table,
    whose clip rows the bake then takes, #179)."""
    from pathlib import Path
    doc, binary = read_glb(path)
    return read_skeleton(doc, binary, Path(path).name)


def masks_for(skeleton: Skeleton, settings: PoseSettings) -> Tuple[str, MaskTable]:
    """The first ``[render.pose.skeletons.*]`` table whose rules resolve on
    ``skeleton`` (all its bones exist, its masks partition the skeleton).
    Raises naming every table's reason when none does."""
    reasons = []
    for sk_name, rules in settings.skeletons:
        try:
            return sk_name, MaskTable.from_rules(
                skeleton, rules, f"[render.pose.skeletons.{sk_name}] masks")
        except PoseConfigError as exc:
            reasons.append(str(exc))
    raise PoseConfigError(f"{skeleton.source}: no [render.pose.skeletons] table "
                          "fits its skeleton -- " + "; ".join(reasons))


# ---------------------------------------------------------------------------
# Per-unit state
# ---------------------------------------------------------------------------

class PoseState:
    """One unit's pose-layer state (renderer-side, keyed by unit id, NEVER on
    ``Unit``). Per override layer: current clip + time, the clip it is
    fading from + its time, and the fade weight (1 = settled). Times are clip
    seconds. One optional additive (clip, time, weight, per-bone weights).

    #163 adds the reactions and the aim-vs-travel twist: ``layer_rate``
    (per layer, -1 = the clip plays backwards: a backpedal), ``twist`` (the
    lower body's yaw about the model's vertical, radians, read by the twist
    modifier), the low-passed travel direction relative to the facing
    (``travel``, a unit vector) and the backpedal flag; the aim hold left
    (``aim_left``, wall seconds), a pending shot retrigger, and the flinch
    envelope (``flinch_t`` wall seconds since the hit, < 0 = none)."""

    __slots__ = ("look", "clip", "time", "prev_clip", "prev_time", "fade",
                 "add_clip", "add_time", "add_weight", "add_bones", "state",
                 "layer_rate", "twist", "travel", "back", "aim_left",
                 "shot_pending", "flinch_t", "flinch_peak",
                 "posture", "noise", "tau", "tempo", "melee_pending",
                 "melee_next", "melee_w",
                 # #184: locomotion (drawn speed, gait, sector, gait phase),
                 # the per-hit flinch clip, the timed one-shot (reload /
                 # throw), and per-prop socket overrides (none yet)
                 "speed", "motion_acc", "motion_d", "motion_t", "gait", "sector", "gait_phase",
                 "loco_state", "flinch_clip", "timed", "prop_sockets",
                 # #185: the running additive's envelope (a hit's or a
                 # stagger's ramps) and its bone weights
                 "flinch_ramps", "flinch_bones")

    def __init__(self, look: str, clips: np.ndarray) -> None:
        n = len(clips)
        self.look = look
        self.clip = np.array(clips, dtype=np.int64)
        self.time = np.zeros(n)
        self.prev_clip = np.full(n, NO_CLIP, dtype=np.int64)
        self.prev_time = np.zeros(n)
        self.fade = np.ones(n)
        self.add_clip = NO_CLIP
        self.add_time = 0.0
        self.add_weight = 0.0
        self.add_bones: Optional[np.ndarray] = None    # (J,) weights; None = every bone
        self.state: Optional[str] = None
        self.layer_rate = np.ones(n)
        self.twist = 0.0
        self.travel = np.array([1.0, 0.0])
        self.back = False
        self.aim_left = 0.0
        self.shot_pending = False
        self.flinch_t = -1.0
        self.flinch_peak = 0.0
        # #164: posture amount a (0 = none: the pose is exactly as without
        # posture), the unit's noise draws (pose_posture.UnitNoise), its
        # wall clock for the noise, the tempo multiplier on looping clips,
        # the melee one-shot (pending, which clip next, current weight)
        self.posture = 0.0
        self.noise = None
        self.tau = 0.0
        self.tempo = 1.0
        self.melee_pending = False
        self.melee_next = 0
        self.melee_w = 0.0
        # #184
        self.speed = 0.0               # drawn speed, model m per wall s
        self.motion_acc = 0.0          # wall s since the last position change
        self.motion_d = 0.0            # low-passed distance per sample (model m)
        self.motion_t = 0.0            # low-passed wall s per sample
        self.gait = -1                 # gait row index, -1 = idle
        self.sector = -1               # sector index within the gait
        self.gait_phase = 0.0          # [0, 1): the shared phase of the gait clips
        self.loco_state: Optional[str] = None
        self.flinch_clip = NO_CLIP     # the clip of the current flinch
        self.flinch_ramps = (0.0, 0.0)  # (ramp_in_s, ramp_out_s) of the current one
        self.flinch_bones: Optional[np.ndarray] = None
        self.timed: Optional["TimedShot"] = None
        self.prop_sockets: Dict[str, str] = {}   # prop -> socket (a future reaction's row)


@dataclass
class TimedShot:
    """A running time-scaled one-shot (#184, reload / throw): ``clip`` on
    ``layers`` at clip time ``t``, advancing ``speed`` clip s per wall s
    until ``t`` reaches ``change_at``, then ``speed_after``."""
    clip: int
    layers: Tuple[int, ...]
    t: float
    speed: float
    change_at: float
    speed_after: float

    def step(self, dt: float) -> None:
        if self.t < self.change_at:
            nxt = self.t + dt * self.speed
            if nxt > self.change_at and self.speed > 0.0:
                rest = dt - (self.change_at - self.t) / self.speed
                nxt = self.change_at + rest * self.speed_after
            self.t = nxt
        else:
            self.t += dt * self.speed_after


@dataclass
class PoseBatch:
    """The output of one :meth:`PoseLayer.compose` call, rows in input order."""
    global_trs: np.ndarray       # (N, J, 10) float32: T xyz, R xyzw, S xyz (model space)
    skin: np.ndarray             # (N, J, 4, 4) float32: global @ inverse bind
    local_t: np.ndarray          # (N, J, 3) the composed local pose (pre-FK)
    local_r: np.ndarray          # (N, J, 4)


class PoseModifier(Protocol):
    """A procedural post-FK stage (look-at, aim-vs-travel twist, ...; §4.5
    corrections 5 and 7). ``prepare`` resolves its bones on a look once, at
    load; ``apply`` edits the batch's global matrices IN PLACE and is
    responsible for re-propagating any subtree it moves. Built: the
    aim-vs-travel twist (``pose_modifiers.AimTravelTwist``, #163)."""

    def prepare(self, rig: "LookRig") -> None: ...

    def apply(self, rig: "LookRig", states: Sequence[PoseState],
              globals_: np.ndarray) -> None: ...


@dataclass
class LookRig:
    """One look's pose data: its clip bank, masks and the state table
    resolved to its own clip ids (``(L,)`` per state; an overlay row holds
    ``NO_CLIP`` on the layers it leaves alone), and the #163 reactions'
    clip ids and bone weights."""
    name: str
    bank: ClipBank
    masks: MaskTable
    skeleton_table: str
    layer_masks: np.ndarray                       # (L,) mask index per layer
    owner_layer: np.ndarray                       # (J,) the layer driving each bone
    state_clips: Dict[str, np.ndarray] = field(default_factory=dict)
    overlay_clips: Dict[str, np.ndarray] = field(default_factory=dict)
    shot_clip: int = NO_CLIP
    hit_clip: int = NO_CLIP
    hit_bones: Optional[np.ndarray] = None        # (J,) the flinch's bone weights
    stagger_clip: int = NO_CLIP                   # (#185) the stagger's single clip
    stagger_bones: Optional[np.ndarray] = None    # (#185) its bone weights
    twist_layer: int = 0                          # the layer the twist's lower bone is on
    posture: Optional[Any] = None                 # pose_posture.ResolvedPosture (#164)
    melee_clips: Optional[np.ndarray] = None      # (#164) clip ids, played in turn
    melee_layers: Optional[np.ndarray] = None     # layer indices
    # #179: the skeleton's playback -- its rate (clip s per wall s), its
    # canonical -> own clip names, and per-role state rows / posture amounts
    rate: float = 1.0
    clip_names: Dict[str, str] = field(default_factory=dict)
    role_state_clips: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    role_posture: Dict[str, float] = field(default_factory=dict)
    # #184: locomotion (pose_locomotion.ResolvedLocomotion; None = none),
    # the per-direction flinch clips ((clip, from), empty = hit_clip), the
    # reload / throw one-shot clips, and the sockets (name -> bone index)
    locomotion: Optional[Any] = None
    hit_dirs: Tuple[Tuple[int, float], ...] = ()
    reload_clip: int = NO_CLIP
    throw_clip: int = NO_CLIP
    sockets: Dict[str, int] = field(default_factory=dict)

    def clip_name(self, canonical: str) -> str:
        """This look's own name for a clip the tables name (#179); a name
        that is no ``clip_names`` key is its own."""
        return self.clip_names.get(canonical, canonical)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------

class PoseLayer:
    """Owns every look's :class:`LookRig`, drives :class:`PoseState`s from the
    state table, composes them in batches, and keeps each unit's event queue
    (#163: drained once per frame into the reactions -- :meth:`fire`,
    :meth:`hit` -- beside the travel steer, :meth:`steer`)."""

    def __init__(self, settings: PoseSettings) -> None:
        from .pose_modifiers import AimTravelTwist
        self.settings = settings
        self._rigs: Dict[str, LookRig] = {}
        self._events: Dict[int, Deque] = {}
        self._event_time: Dict[int, float] = {}
        self.warnings: List[str] = []
        # the post-FK modifiers, in compose order (#163: the twist)
        self.modifiers: Tuple[PoseModifier, ...] = (
            (AimTravelTwist(settings.twists),) if settings.twists else ())

    # --- looks ---------------------------------------------------------
    def add_look(self, name: str, glb_path=None, bank: Optional[ClipBank] = None) -> LookRig:
        """Bake one look's clips (from its ``.glb``, or a ready bank) and
        resolve the masks and the state table on its skeleton. A state clip
        the look lacks falls back to the fallback state's clip for that
        layer (recorded in ``warnings``); a fallback clip it lacks raises."""
        s = self.settings
        if bank is None:
            # #179: the skeleton table first (its own clip rows shape the bake)
            sk_name, masks = masks_for(_glb_skeleton(glb_path), s)
            pb = s.playback.get(sk_name, SkeletonPlayback())
            bank = build_clip_bank(glb_path, s.clips if pb.clips is None else pb.clips)
        else:
            sk_name, masks = masks_for(bank.skeleton, s)
            pb = s.playback.get(sk_name, SkeletonPlayback())
        names = pb.clip_names
        for canonical, own in names.items():
            if own != ABSENT_CLIP and not bank.has(own):
                raise PoseConfigError(
                    f"{name}: [render.pose.skeletons.{sk_name}] clip_names maps "
                    f"{canonical!r} to {own!r}, which is not a clip of the look")

        def own(clip: str) -> str:
            return names.get(clip, clip)

        layer_masks = np.array([masks.index(l) for l in s.layers], dtype=np.int64)
        layer_of_mask = np.full(len(masks.names), -1, dtype=np.int64)
        layer_of_mask[layer_masks] = np.arange(len(s.layers))
        owner_layer = layer_of_mask[masks.owner]
        if (owner_layer < 0).any():
            orphans = sorted({masks.names[m] for m in masks.owner[owner_layer < 0]})
            raise PoseConfigError(f"{name}: masks {orphans} own bones but are no "
                                  f"layer of [render.pose] layers {list(s.layers)}")
        rig = LookRig(name=name, bank=bank, masks=masks, skeleton_table=sk_name,
                      layer_masks=layer_masks, owner_layer=owner_layer,
                      rate=s.rate if pb.rate is None else pb.rate,
                      clip_names=dict(names))
        fb = {layer: own(clip) for layer, clip in s.states[s.fallback_state].items()}
        for layer in s.layers:
            if not bank.has(fb[layer]):
                raise PoseConfigError(f"{name}: fallback state {s.fallback_state!r} "
                                      f"clip {fb[layer]!r} is not in the look")
        for st_name, row in s.states.items():
            ids = []
            for layer in s.layers:
                clip = own(row[layer])
                if not bank.has(clip):
                    self.warnings.append(f"{name}: state {st_name!r} layer {layer!r} "
                                         f"clip {clip or row[layer]!r} missing; "
                                         f"plays {fb[layer]!r}")
                    clip = fb[layer]
                ids.append(bank.index(clip))
            rig.state_clips[st_name] = np.array(ids, dtype=np.int64)
        # #184: the skeleton's own state rows (its own clip names); a clip
        # the look lacks falls back as a global row's does (a skeleton's
        # looks need not all hold every clip: the SOMA worker has no strafes)
        for st_name, row in pb.states.items():
            ids = []
            for layer in s.layers:
                clip = row[layer]
                if not bank.has(clip):
                    self.warnings.append(f"{name}: [render.pose.skeletons.{sk_name}] "
                                         f"states.{st_name} layer {layer!r} clip {clip!r} "
                                         f"missing; plays {fb[layer]!r}")
                    clip = fb[layer]
                ids.append(bank.index(clip))
            rig.state_clips[st_name] = np.array(ids, dtype=np.int64)
        # #179: the skeleton's own rows for a role (clips required: the
        # skeleton table names them for its own looks)
        for role, sr in pb.roles.items():
            rig.role_posture[role] = sr.posture
            rows = {}
            for st_name, row in sr.states.items():
                what = f"[render.pose.skeletons.{sk_name}] roles.{role} states.{st_name}"
                for layer in s.layers:
                    if own(row[layer]) == ABSENT_CLIP or not bank.has(own(row[layer])):
                        raise PoseConfigError(f"{name}: {what} clip {row[layer]!r} "
                                              "is not in the look")
                rows[st_name] = np.array([bank.index(own(row[layer])) for layer in s.layers],
                                         dtype=np.int64)
            rig.role_state_clips[role] = rows
        # #163 reactions: their clips are not optional (a look that lacks
        # one cannot play what the config asks of it), so a missing one
        # raises -- unless the skeleton's clip_names says it has none
        # (ABSENT_CLIP, #179): then that reaction does nothing on the look
        for ov_name, row in s.overlays.items():
            ids = np.full(len(s.layers), NO_CLIP, dtype=np.int64)
            for layer, clip in row.items():
                ids[s.layers.index(layer)] = self._clip_or_raise(
                    bank, own(clip), name, f"overlay {ov_name!r}")
            rig.overlay_clips[ov_name] = ids
        if s.fire is not None:
            rig.shot_clip = self._clip_or_raise(bank, own(s.fire.shot_clip), name,
                                                "[render.pose.fire] shot_clip")
        if s.hit is not None:
            rig.hit_clip = self._clip_or_raise(bank, own(s.hit.clip), name,
                                               "[render.pose.hit] clip")
            rig.hit_bones = masks.weights[[masks.index(m) for m in s.hit.masks]].sum(0)
            # #184: per-direction flinch clips (the skeleton's own names)
            rig.hit_dirs = tuple(
                (self._clip_or_raise(bank, clip, name,
                                     f"[render.pose.skeletons.{sk_name}] hit_directions"),
                 frm) for clip, frm in pb.hit_directions)
        if s.stagger is not None:                 # #185: the push reaction
            rig.stagger_clip = self._clip_or_raise(bank, own(s.stagger.clip), name,
                                                   "[render.pose.stagger] clip")
            rig.stagger_bones = masks.weights[
                [masks.index(m) for m in s.stagger.masks]].sum(0)
        # #184: reload / throw -- a look without the clip does nothing (only
        # the looks that reload or throw need it); the renderer refuses a
        # named clip that NO look holds
        for key, attr in (("reload", "reload_clip"), ("throw", "throw_clip")):
            ts = getattr(s, key)
            if ts is not None and ts.clip != ABSENT_CLIP and own(ts.clip) != ABSENT_CLIP:
                if bank.has(own(ts.clip)):
                    setattr(rig, attr, bank.index(own(ts.clip)))
                else:
                    self.warnings.append(f"{name}: [render.pose.{key}] clip "
                                         f"{own(ts.clip)!r} missing; the {key} reaction "
                                         "does nothing on this look")
        # #184: sockets (a look whose skeleton lacks the bone holds no prop there)
        for sock, bone in pb.sockets.items():
            if bone not in bank.skeleton.names:
                raise PoseConfigError(f"{name}: [render.pose.skeletons.{sk_name}] sockets."
                                      f"{sock} bone {bone!r} is not in {bank.skeleton.source}")
            rig.sockets[sock] = bank.skeleton.index(bone)
        if s.melee is not None:
            ids = [self._clip_or_raise(bank, own(c), name, "[render.pose.melee] clips")
                   for c in s.melee.clips]
            ids = [c for c in ids if c != NO_CLIP]
            if ids:
                rig.melee_clips = np.array(ids, dtype=np.int64)
                rig.melee_layers = np.array([s.layers.index(l) for l in s.melee.layers],
                                            dtype=np.int64)
        rule = s.postures.get(sk_name)
        if rule is not None:
            from .pose_posture import PostureConfigError, resolve
            try:
                rig.posture = resolve(rule, bank.skeleton,
                                      f"[render.pose.skeletons.{sk_name}] posture")
            except PostureConfigError as exc:
                raise PoseConfigError(str(exc)) from None
        for mod in self.modifiers:
            mod.prepare(rig)
        if pb.locomotion is not None:                    # #184
            from .pose_locomotion import LocomotionConfigError, resolve as loco_resolve
            if s.aim_travel is None:
                raise PoseConfigError(f"{name}: [render.pose.skeletons.{sk_name}] "
                                      "locomotion needs [render.pose.aim_travel]")
            try:
                rig.locomotion = loco_resolve(
                    pb.locomotion, rig, lambda st, role: self._state_ids(rig, st, role),
                    f"{name}: [render.pose.skeletons.{sk_name}] locomotion")
            except LocomotionConfigError as exc:
                raise PoseConfigError(str(exc)) from None
            self.warnings.extend(rig.locomotion.warnings)
        self._rigs[name] = rig
        return rig

    @staticmethod
    def _clip_or_raise(bank: ClipBank, clip: str, look: str, what: str) -> int:
        if clip == ABSENT_CLIP:                  # #179: the skeleton has none
            return NO_CLIP
        if not bank.has(clip):
            raise PoseConfigError(f"{look}: {what} clip {clip!r} is not in the look")
        return bank.index(clip)

    def rig(self, name: str) -> LookRig:
        return self._rigs[name]

    @property
    def looks(self) -> Tuple[str, ...]:
        return tuple(self._rigs)

    # --- state driving ------------------------------------------------
    def _state_ids(self, rig: LookRig, state: str,
                   role: Optional[str] = None) -> np.ndarray:
        if role is not None:                      # #179: the skeleton's role rows
            ids = rig.role_state_clips.get(role, {}).get(state)
            if ids is not None:
                return ids
        ids = rig.state_clips.get(state)
        if ids is None:
            ids = rig.state_clips[self.settings.fallback_state]
        return ids

    def new_state(self, look: str, state: str, role: Optional[str] = None) -> PoseState:
        """A unit's first state: its clips start at time 0, no crossfade."""
        rig = self._rigs[look]
        ps = PoseState(look, self._state_ids(rig, state, role))
        ps.state = state
        ps.flinch_clip = rig.hit_clip
        return ps

    def role_posture(self, look: str, role: str) -> float:
        """The posture amount of a unit of ``role`` drawn with ``look``: the
        look's skeleton's own row when it has one (#179), else the role's
        ``[render.pose.roles]`` posture."""
        rig = self._rigs[look]
        if role in rig.role_posture:
            return rig.role_posture[role]
        return self.settings.roles[role].posture

    def set_layer_clip(self, ps: PoseState, layer: int, clip: int) -> None:
        """Play ``clip`` on override layer ``layer``: a change starts a
        crossfade from what played (its time keeps running) to the new clip
        at time 0. The same clip again changes nothing."""
        if ps.clip[layer] == clip:
            return
        if self.settings.crossfade_s > 0.0:
            ps.prev_clip[layer] = ps.clip[layer]
            ps.prev_time[layer] = ps.time[layer]
            ps.fade[layer] = 0.0
        else:
            ps.prev_clip[layer] = NO_CLIP
            ps.fade[layer] = 1.0
        ps.clip[layer] = clip
        ps.time[layer] = 0.0

    def restart_layer_clip(self, ps: PoseState, layer: int, clip: int) -> None:
        """Play ``clip`` on ``layer`` FROM ITS START even when it is already
        playing (a retriggered one-shot, #163): crossfades from the pose the
        layer shows now to the clip's first key."""
        if self.settings.crossfade_s > 0.0:
            ps.prev_clip[layer] = ps.clip[layer]
            ps.prev_time[layer] = ps.time[layer]
            ps.fade[layer] = 0.0
        ps.clip[layer] = clip
        ps.time[layer] = 0.0

    def set_additive(self, ps: PoseState, clip: Optional[str], weight: float = 1.0,
                     mask=None, time: float = 0.0) -> None:
        """Set (or, with ``clip=None``, clear) the unit's additive layer.
        ``mask``: a mask name, a list of them, or None for every bone."""
        if clip is None:
            ps.add_clip = NO_CLIP
            return
        rig = self._rigs[ps.look]
        ps.add_clip = rig.bank.index(rig.clip_name(clip))
        ps.add_time = float(time)
        ps.add_weight = float(weight)
        if mask is None:
            ps.add_bones = None
        else:
            names = [mask] if isinstance(mask, str) else list(mask)
            ps.add_bones = rig.masks.weights[[rig.masks.index(m) for m in names]].sum(0)

    # --- reactions (#163) ----------------------------------------------
    def fire(self, ps: PoseState) -> None:
        """A round left this unit's barrel: retrigger the shot one-shot and
        (re)start the aim hold. Takes effect in the next :meth:`drive`."""
        f = self.settings.fire
        if f is None:
            return
        ps.aim_left = f.aim_hold_s
        ps.shot_pending = True

    def hit(self, ps: PoseState, damage: float, cause: Optional[str] = None,
            source: Optional[float] = None) -> None:
        """This unit was hit: (re)start the flinch additive at a weight that
        rises with ``damage``, clamped to ``[weight_min, weight_max]``. A
        cause outside ``[render.pose.hit] causes`` does nothing (a burn's
        per-tick damage must not make the unit twitch). A retrigger restarts
        the clip: a second jolt (the clip's first key is its own additive
        reference, so the restart lands on the un-flinched pose).

        #184: ``source`` is the direction the hit came from, relative to the
        unit's facing (radians, + = its left; None = unknown, taken as the
        front). On a look whose skeleton has ``hit_directions`` the flinch
        clip is the row nearest it; otherwise the single clip."""
        h = self.settings.hit
        if h is None or (cause is not None and cause not in h.causes):
            return
        x = min(1.0, max(0.0, float(damage) / h.damage_full))
        rig = self._rigs[ps.look]
        ps.flinch_peak = h.weight_min + (h.weight_max - h.weight_min) * x
        ps.flinch_t = 0.0
        ps.flinch_clip = self.flinch_clip_for(rig, source)
        ps.flinch_ramps = (h.ramp_in_s, h.ramp_out_s)
        ps.flinch_bones = rig.hit_bones

    def stagger(self, ps: PoseState, source: Optional[float] = None) -> None:
        """This unit was pushed past its sway (#185, a ``StaggerEvent``):
        (re)start the stagger additive at ``[render.pose.stagger] weight``
        from ``source``, the side the push came from relative to the facing
        (radians, + = its left; None = the front) -- the skeleton's
        ``hit_directions`` row nearest it, else the row's single clip. It
        takes over the flinch's additive slot (envelope, masks) until it
        ends or the next hit restarts it."""
        g = self.settings.stagger
        if g is None:
            return
        rig = self._rigs[ps.look]
        ps.flinch_peak = g.weight
        ps.flinch_t = 0.0
        ps.flinch_clip = self.flinch_clip_for(rig, source, rig.stagger_clip)
        ps.flinch_ramps = (g.ramp_in_s, g.ramp_out_s)
        ps.flinch_bones = rig.stagger_bones

    @staticmethod
    def flinch_clip_for(rig: LookRig, source: Optional[float],
                        single: Optional[int] = None) -> int:
        """The flinch clip of a hit (or, #185, a push) from ``source`` (see
        :meth:`hit`); ``single`` is the clip of a skeleton without
        ``hit_directions`` (default: the hit's)."""
        if not rig.hit_dirs:
            return rig.hit_clip if single is None else single
        a = 0.0 if source is None else float(source)

        def gap(row) -> float:
            return abs((a - row[1] + np.pi) % (2.0 * np.pi) - np.pi)
        return int(min(rig.hit_dirs, key=gap)[0])

    def reload(self, ps: PoseState, seconds: float) -> None:
        """A reload started (#184): the reload clip plays over exactly
        ``seconds`` (the event's ticks / ticks per second) on the row's
        layers. Nothing on a look without the clip, or for ``seconds <= 0``."""
        rig = self._rigs[ps.look]
        ts = self.settings.reload
        if ts is None or rig.reload_clip == NO_CLIP or not seconds > 0.0:
            return
        dur = float(rig.bank.duration[rig.reload_clip])
        speed = dur / float(seconds)
        ps.timed = TimedShot(clip=rig.reload_clip,
                             layers=tuple(self.settings.layers.index(l) for l in ts.layers),
                             t=0.0, speed=speed, change_at=dur, speed_after=speed)

    def throw(self, ps: PoseState, windup_s: float) -> None:
        """A throw's wind-up started (#184): the throw clip plays so that its
        ``release_s`` lands exactly ``windup_s`` from now (the grenade leaves
        at the end of the wind-up) and at its authored speed after. A wind-up
        at least as long as the clip's own (``release_s`` at authored speed)
        plays the clip from its start, slowed; a shorter one starts the clip
        part-way, at authored speed; ``windup_s <= 0`` (an instant throw)
        plays the follow-through from the release on."""
        rig = self._rigs[ps.look]
        ts = self.settings.throw
        if ts is None or rig.throw_clip == NO_CLIP:
            return
        c = rig.throw_clip
        dur = float(rig.bank.duration[c])
        rel = min(ts.release_s, dur)
        authored = rig.rate * float(rig.bank.rate[c])
        w = max(0.0, float(windup_s))
        if w * authored >= rel and w > 0.0:
            t0, speed = 0.0, rel / w
        else:
            t0, speed = rel - w * authored, authored
        ps.timed = TimedShot(clip=c,
                             layers=tuple(self.settings.layers.index(l) for l in ts.layers),
                             t=t0, speed=speed, change_at=rel, speed_after=authored)

    def socket_bone(self, ps: PoseState, prop: str, default_socket: str) -> int:
        """The bone a held prop hangs on for this unit (#184): the unit's
        socket for ``prop`` (``ps.prop_sockets`` -- where a future reaction
        row moves a prop, e.g. the carbine to the left hand for a throw;
        nothing writes it yet) or ``default_socket``, resolved on the unit's
        look; -1 when its skeleton has no such socket."""
        socket = ps.prop_sockets.get(prop, default_socket)
        return self._rigs[ps.look].sockets.get(socket, -1)

    def locomotes(self, look: str, role: Optional[str]) -> bool:
        """Whether units of ``role`` drawn with ``look`` are driven by the
        locomotion table (#184)."""
        loco = self._rigs[look].locomotion
        return loco is not None and role in loco.roles

    def locomote(self, ps: PoseState, state: str, moved_m: float,
                 travel: Optional[float], moving: bool, dt: float,
                 role: Optional[str] = None, start_span_s: float = 1.0 / 24.0) -> str:
        """The unit's motion this frame (#163, #184); returns the state to
        :meth:`drive`. On a look whose skeleton carries a ``locomotion``
        table for ``role``: the gait, the direction and the gait phase from
        the drawn speed (``moved_m``: model metres moved since the last
        frame) and the relative ``travel`` (``pose_locomotion.locomote``),
        replacing the renderer's ``state`` when it is one the table drives
        (its idle or a gait / sector state). Otherwise exactly
        :meth:`steer` (``state`` returned unchanged). ``start_span_s``: one
        sim tick in wall seconds (the span of a first delta after a stop)."""
        rig = self._rigs[ps.look]
        loco = rig.locomotion
        if loco is None or role not in loco.roles:
            self.steer(ps, travel, moving, dt)
            return state
        if state not in loco.rule.states():
            return state
        from .pose_locomotion import locomote
        return locomote(ps, loco, loco.roles[role], self.settings.aim_travel,
                        moved_m, travel, dt, start_span_s)

    def melee(self, ps: PoseState) -> None:
        """This unit's melee strike landed (#164): the next :meth:`drive`
        plays the next melee clip from its start on the melee layers."""
        if self.settings.melee is not None:
            ps.melee_pending = True

    def set_posture(self, ps: PoseState, uid: int, amount: float) -> None:
        """The unit's posture amount ``a`` in [0, 1] this frame (#164; a
        float so a gradual infection can feed it later). The first time a
        unit's amount is above 0 it draws its noise from its own generator
        (``(seed, uid)``) and, with ``desync_start``, its looping clips jump
        to its own start time -- so a horde does not walk in step. A unit at
        ``a = 0`` composes exactly as without posture."""
        a = min(1.0, max(0.0, float(amount)))
        p = self.settings.posture
        rig = self._rigs[ps.look]
        if p is None or rig.posture is None:
            ps.posture = 0.0
            return
        if a > 0.0 and ps.noise is None:
            from .pose_posture import unit_noise
            ps.noise = unit_noise(p, uid, rig.posture.n_channels)
            if p.desync_start:
                bank = rig.bank
                ps.time = np.where(bank.loop[ps.clip],
                                   ps.noise.start * bank.duration[ps.clip], ps.time)
        ps.posture = a

    def _posture_params(self, ps: PoseState) -> np.ndarray:
        """``(P,)``: the profile x amount x the unit's variation, the melee's
        suppressed parameters faded by its weight."""
        p = self.settings.posture
        v = p.vector * ps.posture * ps.noise.scale
        m = self.settings.melee
        if m is not None and ps.melee_w > 0.0 and m.suppress:
            from .pose_posture import PARAMS
            idx = [PARAMS.index(x) for x in m.suppress]
            v = v.copy()
            v[idx] *= 1.0 - ps.melee_w
        return v

    def _tempo(self, ps: PoseState) -> float:
        """The unit's looping-clip rate multiplier ``1 + a J n(tau)`` (1 at
        ``a = 0``); ``J <= 1`` and ``|n| <= 1`` keep it >= 0."""
        if ps.posture <= 0.0 or ps.noise is None:
            return 1.0
        from .pose_posture import PARAMS, smooth_noise
        j = float(self._posture_params(ps)[PARAMS.index("tempo_jitter")])
        if j <= 0.0:
            return 1.0
        n = float(smooth_noise(ps.noise.tempo_phase, ps.noise.tempo_hz, ps.tau))
        return max(0.0, 1.0 + min(1.0, j) * n)

    def steer(self, ps: PoseState, travel: Optional[float], moving: bool,
              dt: float) -> None:
        """The aim-vs-travel input (#163): ``travel`` is the direction the
        unit moved this frame RELATIVE to its facing, as a yaw about the
        model's vertical (radians), or None when it did not move measurably.
        Low-pass, backpedal switch and the twist's slew: see
        :func:`pose_modifiers.steer`."""
        if self.settings.aim_travel is None or not self.modifiers:
            return
        from .pose_modifiers import steer
        steer(ps, self.settings.aim_travel, self._rigs[ps.look].twist_layer,
              travel, moving, dt)

    def drive(self, ps: PoseState, state: str, dt: float,
              role: Optional[str] = None) -> None:
        """Per frame: pick each layer's clip -- ``state``'s row of the table
        (the look's skeleton's own row for ``role`` when it has one, #179),
        under the fire overlay while the aim hold lasts, under the shot
        one-shot while it plays -- (crossfading every change), advance the
        clocks by ``dt`` wall seconds, then the aim hold and the flinch."""
        rig = self._rigs[ps.look]
        s = self.settings
        want = self._state_ids(rig, state, role).copy()
        f = s.fire
        if f is not None:
            if ps.aim_left > 0.0:
                ov = rig.overlay_clips[f.overlay]
                want = np.where(ov >= 0, ov, want)
            sl = s.layers.index(f.shot_layer)
            if ps.shot_pending:
                if rig.shot_clip != NO_CLIP:
                    self.restart_layer_clip(ps, sl, rig.shot_clip)
                ps.shot_pending = False
            if ps.clip[sl] == rig.shot_clip and ps.aim_left > 0.0 and \
                    ps.time[sl] < rig.bank.duration[rig.shot_clip]:
                want[sl] = rig.shot_clip                 # the shot plays out
        mc = rig.melee_clips
        if mc is not None:                                # #164: the melee one-shot
            if ps.melee_pending:
                clip = int(mc[ps.melee_next % len(mc)])
                for ml in rig.melee_layers:
                    self.restart_layer_clip(ps, int(ml), clip)
                ps.melee_next += 1
                ps.melee_pending = False
            for ml in rig.melee_layers:
                c = ps.clip[ml]
                if np.isin(c, mc) and ps.time[ml] < rig.bank.duration[c]:
                    want[ml] = c                          # the strike plays out
        tm = ps.timed                                     # #184: reload / throw
        if tm is not None and tm.t >= rig.bank.duration[tm.clip]:
            ps.timed = tm = None                          # played out: back to the state
        if tm is not None:
            for tl in tm.layers:
                want[tl] = tm.clip
        for layer in range(len(want)):
            self.set_layer_clip(ps, layer, int(want[layer]))
        ps.state = state
        ps.tempo = self._tempo(ps)
        self.advance(ps, dt)
        if rig.locomotion is not None and role in rig.locomotion.roles:
            from .pose_locomotion import lock_phase
            lock_phase(ps, rig.locomotion, rig.bank)      # #184: one gait phase
        if tm is not None:
            for tl in tm.layers:
                ps.time[tl] = min(tm.t, float(rig.bank.duration[tm.clip]))
            tm.step(max(dt, 0.0))
        ps.tau += max(dt, 0.0)
        ps.aim_left = max(0.0, ps.aim_left - max(dt, 0.0))
        self._flinch(ps, rig, dt)
        if mc is not None:
            ml = int(rig.melee_layers[0])
            if np.isin(ps.clip[ml], mc):
                ps.melee_w = float(ps.fade[ml])
            elif ps.prev_clip[ml] >= 0 and np.isin(ps.prev_clip[ml], mc):
                ps.melee_w = float(1.0 - ps.fade[ml])
            else:
                ps.melee_w = 0.0

    def _flinch(self, ps: PoseState, rig: LookRig, dt: float) -> None:
        """The flinch envelope, ``flinch_t`` wall seconds after the hit: a
        ramp from 0 up to ``flinch_peak`` over ``ramp_in_s``,
        times a ramp down to 0 over the clip's last ``ramp_out_s`` -- 0 at
        its end, where the additive is dropped (the pose is exactly the
        un-flinched one again)."""
        if ps.flinch_t < 0.0:
            return
        c = ps.flinch_clip if ps.flinch_clip != NO_CLIP else rig.hit_clip   # #184
        if c == NO_CLIP:                       # #179: the look has no flinch clip
            ps.flinch_t = -1.0
            return
        ps.flinch_t += max(dt, 0.0)
        speed = rig.rate * rig.bank.rate[c]
        wall_dur = rig.bank.duration[c] / speed
        tau = ps.flinch_t
        if tau >= wall_dur:
            ps.flinch_t = -1.0
            ps.add_clip = NO_CLIP
            ps.add_weight = 0.0
            return
        ramp_in, ramp_out = ps.flinch_ramps          # #185: the hit's or the stagger's
        up = 1.0 if ramp_in <= 0.0 else min(1.0, tau / ramp_in)
        down = 1.0 if ramp_out <= 0.0 else min(1.0, (wall_dur - tau) / ramp_out)
        ps.add_clip = c
        ps.add_bones = ps.flinch_bones if ps.flinch_bones is not None else rig.hit_bones
        ps.add_time = tau * speed
        ps.add_weight = ps.flinch_peak * up * down

    def advance(self, ps: PoseState, dt: float) -> None:
        """Advance every clock by ``dt`` x the global rate x the clip's rate
        x the layer's rate (negative = backwards, #163); loops wrap,
        one-shots hold their end key; ramps move by ``dt / crossfade_s`` and
        a settled fade drops its source."""
        if dt <= 0.0:
            return
        rig = self._rigs[ps.look]
        bank = rig.bank
        step = dt * rig.rate * ps.layer_rate
        if ps.tempo != 1.0:                       # #164: tempo jitter, looping clips only
            ps.time = _wrap(bank, ps.clip, ps.time + step * bank.rate[ps.clip]
                            * np.where(bank.loop[ps.clip], ps.tempo, 1.0))
        else:
            ps.time = _wrap(bank, ps.clip, ps.time + step * bank.rate[ps.clip])
        fading = ps.prev_clip >= 0
        if fading.any():
            pc = np.where(fading, ps.prev_clip, ps.clip)
            pstep = step * (np.where(bank.loop[pc], ps.tempo, 1.0) if ps.tempo != 1.0 else 1.0)
            ps.prev_time = np.where(
                fading, _wrap(bank, pc, ps.prev_time + pstep * bank.rate[pc]), 0.0)
            cf = self.settings.crossfade_s
            ps.fade = np.where(fading, np.minimum(1.0, ps.fade + (dt / cf if cf > 0 else 1.0)), 1.0)
            done = ps.fade >= 1.0
            ps.prev_clip = np.where(done, NO_CLIP, ps.prev_clip)
        if ps.add_clip >= 0:
            ac = np.array([ps.add_clip])
            a_step = dt * rig.rate * bank.rate[ac[0]]
            ps.add_time = float(_wrap(bank, ac, np.array([ps.add_time + a_step]))[0])

    # --- compose ---------------------------------------------------------
    def compose(self, look: str, states: Sequence[PoseState],
                modifiers: Optional[Sequence[PoseModifier]] = None) -> PoseBatch:
        """Compose every state of ONE look in one batch. Compose order is
        fixed: override layers (with their crossfades), selected per bone by
        the mask partition -> additive -> FK -> modifiers -> skin.
        ``modifiers`` None = the layer's own (:attr:`modifiers`)."""
        if modifiers is None:
            modifiers = self.modifiers
        rig = self._rigs[look]
        bank = rig.bank
        sk = bank.skeleton
        N = len(states)
        L = len(self.settings.layers)
        J = sk.n_bones
        if N == 0:
            z = np.zeros
            return PoseBatch(z((0, J, GLOBAL_TRS_WIDTH), np.float32),
                             z((0, J, 4, 4), np.float32), z((0, J, 3), np.float32),
                             z((0, J, 4), np.float32))
        clip = np.stack([ps.clip for ps in states], axis=1)        # (L, N)
        time = np.stack([ps.time for ps in states], axis=1)
        t, r, s = sample_bank(bank, clip.ravel(), time.ravel())
        t = t.reshape(L, N, J, 3)
        r = r.reshape(L, N, J, 4)
        if s is not None:
            s = s.reshape(L, N, J, 3)
        prev = np.stack([ps.prev_clip for ps in states], axis=1)
        fl, fn = np.nonzero(prev >= 0)
        if len(fl):
            ptime = np.stack([ps.prev_time for ps in states], axis=1)
            fade = np.stack([ps.fade for ps in states], axis=1)
            pt, pr, pscale = sample_bank(bank, prev[fl, fn], ptime[fl, fn])
            w = fade[fl, fn].astype(np.float32)[:, None, None]
            t[fl, fn] = pt + (t[fl, fn] - pt) * w
            r[fl, fn] = quat_nlerp(pr, r[fl, fn], w)
            if s is not None:
                s[fl, fn] = pscale + (s[fl, fn] - pscale) * w
        # each bone from the layer whose mask owns it
        owner_layer = rig.owner_layer
        jj = np.arange(J)
        lt = t[owner_layer, :, jj].transpose(1, 0, 2)               # (N, J, 3)
        lr = r[owner_layer, :, jj].transpose(1, 0, 2)
        ls = s[owner_layer, :, jj].transpose(1, 0, 2) if s is not None else None
        # additive
        add = np.array([ps.add_clip for ps in states])
        an = np.flatnonzero(add >= 0)
        if len(an):
            ac = add[an]
            at, ar, _ = sample_bank(bank, ac, np.array([states[i].add_time for i in an]))
            ref = bank.offset[ac] + bank.additive_ref[ac]
            rt, rr = bank.t[ref], bank.r[ref]
            ones = np.ones(J, dtype=np.float32)
            mw = np.stack([ones if states[i].add_bones is None else states[i].add_bones
                           for i in an])
            w = (np.array([states[i].add_weight for i in an])[:, None] * mw).astype(np.float32)
            delta = quat_mul(quat_conj(rr), ar)
            ident = np.zeros_like(delta)
            ident[..., 3] = 1.0
            lr = lr.copy()
            lt = lt.copy()
            lr[an] = quat_mul(lr[an], quat_nlerp(ident, delta, w[..., None]))
            lt[an] = lt[an] + (at - rt) * w[..., None]
        # posture (#164): profile x amount, post-clip, pre-FK; a = 0 rows untouched
        if rig.posture is not None:
            pn = np.flatnonzero(np.array([ps.posture for ps in states]) > 0.0)
            if len(pn):
                from .pose_posture import PARAMS, apply, smooth_noise
                res = rig.posture
                params = np.stack([self._posture_params(states[i]) for i in pn])
                if res.n_channels:
                    tw = params[:, PARAMS.index("twitch")]
                    noise = smooth_noise(
                        np.stack([states[i].noise.twitch_phase for i in pn]),
                        np.stack([states[i].noise.twitch_hz for i in pn]),
                        np.array([states[i].tau for i in pn])[:, None])
                    tv = tw[:, None] * res.twitch_deg[None, :] * noise
                else:
                    tv = np.zeros((len(pn), 0))
                lt = lt.copy()
                lr = lr.copy()
                lt[pn], lr[pn] = apply(res, lt[pn], lr[pn], params, tv)
        g = forward_kinematics(sk, lt, lr, ls)
        for mod in modifiers:
            mod.apply(rig, states, g)
        return PoseBatch(global_trs=global_trs(g), skin=_mat4_mul(g, sk.bind_inverse.astype(np.float32)),
                         local_t=lt, local_r=lr)

    # --- events (#162 queue; #163 drains it) ------------------------------
    def queue_event(self, uid: int, event, now: float) -> None:
        q = self._events.get(uid)
        if q is None:
            q = deque(maxlen=self.settings.event_queue)
            self._events[uid] = q
        q.append(event)
        self._event_time[uid] = now

    def events(self, uid: int) -> Tuple:
        return tuple(self._events.get(uid, ()))

    def take_events(self, uid: int) -> Tuple:
        """The unit's queued events, oldest first, and empty its queue (each
        event is acted on once, #163)."""
        q = self._events.get(uid)
        if not q:
            return ()
        out = tuple(q)
        q.clear()
        return out

    def prune_events(self, live: Mapping, now: float, stale_s: float) -> None:
        """Drop the queue of a unit the renderer no longer keeps state for,
        once nothing has been queued for it in ``stale_s``."""
        for uid in [u for u in self._events
                    if u not in live and now - self._event_time.get(u, now) > stale_s]:
            del self._events[uid]
            self._event_time.pop(uid, None)


# ---------------------------------------------------------------------------
# Batched kernels
# ---------------------------------------------------------------------------

def _wrap(bank: ClipBank, clips: np.ndarray, times: np.ndarray) -> np.ndarray:
    dur = bank.duration[clips]
    looped = np.where(dur > 0, np.mod(times, np.where(dur > 0, dur, 1.0)), 0.0)
    return np.where(bank.loop[clips], looped, np.clip(times, 0.0, dur))


def sample_bank(bank: ClipBank, clips: np.ndarray, times: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Local TRS of ``clips[i]`` at clip time ``times[i]`` (sub-frame: lerp /
    nlerp between the two grid keys around it; a STEP channel holds its key).
    Times outside the clip wrap (loop) or clamp (one-shot). Shapes
    ``(M, J, 3)``, ``(M, J, 4)``, ``(M, J, 3)`` or None."""
    clips = np.asarray(clips, dtype=np.int64)
    t = _wrap(bank, clips, np.asarray(times, dtype=np.float64))
    n = bank.n_keys[clips]
    f = t / bank.key_dt[clips]
    # +1e-6 key: a time ON a key (k * dt, which may divide to k - 1e-15)
    # selects that key, so a STEP channel shows it rather than the previous
    k = np.clip(np.floor(f + 1e-6).astype(np.int64), 0, n - 1)
    k1 = np.minimum(k + 1, n - 1)
    u = np.clip(f - k, 0.0, 1.0).astype(np.float32)
    g0 = bank.offset[clips] + k
    g1 = bank.offset[clips] + k1
    ut = (u[:, None] * ~bank.step_t[clips])[..., None]
    ur = (u[:, None] * ~bank.step_r[clips])[..., None]
    t0 = bank.t[g0]
    out_t = t0 + (bank.t[g1] - t0) * ut
    out_r = quat_nlerp(bank.r[g0], bank.r[g1], ur).astype(np.float32)
    out_s = None
    if bank.s is not None:
        us = (u[:, None] * ~bank.step_s[clips])[..., None]
        s0 = bank.s[g0]
        out_s = s0 + (bank.s[g1] - s0) * us
    return out_t.astype(np.float32), out_r, out_s


def _mat4_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched 4x4 products WITHOUT BLAS (the frame-path rule: MKL's thread
    pool stalls ``@`` under CPU contention; plain einsum, no optimizer)."""
    return np.einsum("...ij,...jk->...ik", a, b)


def forward_kinematics(sk: Skeleton, t: np.ndarray, r: np.ndarray,
                       s: Optional[np.ndarray]) -> np.ndarray:
    """Global (model-space) bone matrices ``(N, J, 4, 4)`` from local TRS,
    one batched product per depth level."""
    local = trs_matrices(t, r, s).astype(np.float32)
    g = np.empty_like(local)
    for lvl in sk.levels:
        par = sk.parents[lvl]
        if par[0] < 0:                    # depth 0: the roots
            g[:, lvl] = local[:, lvl]
        else:
            g[:, lvl] = _mat4_mul(g[:, par], local[:, lvl])
    return g


def global_trs(g: np.ndarray) -> np.ndarray:
    """Global matrices -> ``(N, J, 10)`` float32 rows ``T, R(xyzw), S`` (the
    layout of raylib's ``Transform``)."""
    rot = g[..., :3, :3]
    scale = np.linalg.norm(rot, axis=-2)
    q = mat3_to_quat(rot / scale[..., None, :])
    out = np.empty(g.shape[:-2] + (GLOBAL_TRS_WIDTH,), dtype=np.float32)
    out[..., 0:3] = g[..., :3, 3]
    out[..., 3:7] = q
    out[..., 7:10] = scale
    return out


__all__ = [
    "PoseConfigError", "PoseSettings", "MaskRule", "MaskTable", "PoseState",
    "TwistRule", "FireSettings", "HitSettings", "StaggerSettings", "AimTravelSettings",
    "MeleeSettings", "RoleSettings", "REACTIONS", "SkeletonPlayback",
    "SkeletonRole", "ABSENT_CLIP", "TimedShotSettings", "TimedShot",
    "PoseBatch", "PoseModifier", "LookRig", "PoseLayer", "pose_settings",
    "masks_for", "sample_bank", "forward_kinematics", "global_trs",
    "GLOBAL_TRS_WIDTH", "NO_CLIP",
]
