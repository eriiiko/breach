"""Pose layer -- THE place a 3D unit's bone pose is composed (M13, #162).

Pure numpy, render-only, pyray-free. Sources in, bone matrices out:

    per-unit PoseState --(table: state -> clip per layer)--> batched compose
      sample every layer's clip (sub-frame, on the clip's own key grid)
      -> crossfade ramp per layer (nlerp source -> target)
      -> override layers selected per bone by the MASK table (a partition)
      -> additive layer (local delta against the clip's own reference frame)
      -> FK as 4x4 products, one per depth level (never a per-bone loop)
      -> procedural modifiers, post-FK (interface only; none built yet)
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
                         quat_nlerp, trs_matrices)

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
             "event_queue", "skeletons", "states", "clips")
_CLIP_KEYS = ("loop", "rate", "root_motion", "additive_ref")


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
    skeletons = []
    for sk_name, sk in skel_tables.items():
        where = f"[render.pose.skeletons.{sk_name}]"
        if not isinstance(sk, dict) or set(sk) != {"masks"}:
            raise PoseConfigError(f"{where} must hold exactly a 'masks' table")
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
        where = f"[render.pose.states] {st_name}"
        if not isinstance(row, dict) or set(row) != set(layers) or \
                not all(isinstance(v, str) and v for v in row.values()):
            raise PoseConfigError(f"{where} must name one clip for each layer "
                                  f"{list(layers)}, got {row!r}")
        states[st_name] = dict(row)
    fallback = d["fallback_state"]
    if fallback not in states:
        raise PoseConfigError(f"[render.pose] fallback_state {fallback!r} is not a state")

    clips_raw = d["clips"]
    if not isinstance(clips_raw, dict):
        raise PoseConfigError("[render.pose.clips] must be a table of clip rows")
    clips: Dict[str, ClipRow] = {}
    for c_name, row in clips_raw.items():
        where = f"[render.pose.clips] {c_name}"
        if not isinstance(row, dict):
            raise PoseConfigError(f"{where} must be a table")
        bad = sorted(set(row) - set(_CLIP_KEYS))
        if bad:
            raise PoseConfigError(f"{where} has unknown keys {bad}")
        loop = row.get("loop", True)
        rate = row.get("rate", 1.0)
        rm = row.get("root_motion", ClipRow().root_motion)
        ref = row.get("additive_ref", 0)
        if not isinstance(loop, bool):
            raise PoseConfigError(f"{where} loop must be a bool, got {loop!r}")
        if not _num(rate) or not rate > 0:
            raise PoseConfigError(f"{where} rate must be a number > 0, got {rate!r}")
        if rm not in ROOT_MOTION_MODES:
            raise PoseConfigError(f"{where} root_motion must be one of "
                                  f"{ROOT_MOTION_MODES}, got {rm!r}")
        if not isinstance(ref, int) or isinstance(ref, bool) or ref < 0:
            raise PoseConfigError(f"{where} additive_ref must be an int >= 0, got {ref!r}")
        clips[c_name] = ClipRow(loop=loop, rate=float(rate), root_motion=rm,
                                additive_ref=ref)
    return PoseSettings(enabled=d["enabled"], rate=float(d["rate"]),
                        crossfade_s=float(d["crossfade_s"]), layers=layers,
                        fallback_state=fallback, event_queue=d["event_queue"],
                        skeletons=tuple(skeletons), states=states, clips=clips)


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
    seconds. One optional additive (clip, time, weight, mask)."""

    __slots__ = ("look", "clip", "time", "prev_clip", "prev_time", "fade",
                 "add_clip", "add_time", "add_weight", "add_mask", "state")

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
        self.add_mask = NO_CLIP          # NO_CLIP = every bone
        self.state: Optional[str] = None


@dataclass
class PoseBatch:
    """The output of one :meth:`PoseLayer.compose` call, rows in input order."""
    global_trs: np.ndarray       # (N, J, 10) float32: T xyz, R xyzw, S xyz (model space)
    skin: np.ndarray             # (N, J, 4, 4) float32: global @ inverse bind
    local_t: np.ndarray          # (N, J, 3) the composed local pose (pre-FK)
    local_r: np.ndarray          # (N, J, 4)


class PoseModifier(Protocol):
    """A procedural post-FK stage (look-at, aim-vs-travel twist, ...; §4.5
    corrections 5 and 7). Edits the batch's global matrices IN PLACE and is
    responsible for re-propagating any subtree it moves. None is built yet."""

    def apply(self, rig: "LookRig", states: Sequence[PoseState],
              globals_: np.ndarray) -> None: ...


@dataclass
class LookRig:
    """One look's pose data: its clip bank, masks and the state table
    resolved to its own clip ids (``(L,)`` per state)."""
    name: str
    bank: ClipBank
    masks: MaskTable
    skeleton_table: str
    layer_masks: np.ndarray                       # (L,) mask index per layer
    state_clips: Dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------

class PoseLayer:
    """Owns every look's :class:`LookRig`, drives :class:`PoseState`s from the
    state table, composes them in batches, and keeps each unit's event queue
    (stored only in #162; read by later slices)."""

    def __init__(self, settings: PoseSettings) -> None:
        self.settings = settings
        self._rigs: Dict[str, LookRig] = {}
        self._events: Dict[int, Deque] = {}
        self._event_time: Dict[int, float] = {}
        self.warnings: List[str] = []

    # --- looks ---------------------------------------------------------
    def add_look(self, name: str, glb_path=None, bank: Optional[ClipBank] = None) -> LookRig:
        """Bake one look's clips (from its ``.glb``, or a ready bank) and
        resolve the masks and the state table on its skeleton. A state clip
        the look lacks falls back to the fallback state's clip for that
        layer (recorded in ``warnings``); a fallback clip it lacks raises."""
        s = self.settings
        if bank is None:
            bank = build_clip_bank(glb_path, s.clips)
        sk_name, masks = masks_for(bank.skeleton, s)
        layer_masks = np.array([masks.index(l) for l in s.layers], dtype=np.int64)
        rig = LookRig(name=name, bank=bank, masks=masks, skeleton_table=sk_name,
                      layer_masks=layer_masks)
        fb = s.states[s.fallback_state]
        for layer in s.layers:
            if not bank.has(fb[layer]):
                raise PoseConfigError(f"{name}: fallback state {s.fallback_state!r} "
                                      f"clip {fb[layer]!r} is not in the look")
        for st_name, row in s.states.items():
            ids = []
            for layer in s.layers:
                clip = row[layer]
                if not bank.has(clip):
                    self.warnings.append(f"{name}: state {st_name!r} layer {layer!r} "
                                         f"clip {clip!r} missing; plays {fb[layer]!r}")
                    clip = fb[layer]
                ids.append(bank.index(clip))
            rig.state_clips[st_name] = np.array(ids, dtype=np.int64)
        self._rigs[name] = rig
        return rig

    def rig(self, name: str) -> LookRig:
        return self._rigs[name]

    @property
    def looks(self) -> Tuple[str, ...]:
        return tuple(self._rigs)

    # --- state driving ------------------------------------------------
    def _state_ids(self, rig: LookRig, state: str) -> np.ndarray:
        ids = rig.state_clips.get(state)
        if ids is None:
            ids = rig.state_clips[self.settings.fallback_state]
        return ids

    def new_state(self, look: str, state: str) -> PoseState:
        """A unit's first state: its clips start at time 0, no crossfade."""
        rig = self._rigs[look]
        ps = PoseState(look, self._state_ids(rig, state))
        ps.state = state
        return ps

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

    def set_additive(self, ps: PoseState, clip: Optional[str], weight: float = 1.0,
                     mask: Optional[str] = None, time: float = 0.0) -> None:
        """Set (or, with ``clip=None``, clear) the unit's additive layer."""
        if clip is None:
            ps.add_clip = NO_CLIP
            return
        rig = self._rigs[ps.look]
        ps.add_clip = rig.bank.index(clip)
        ps.add_time = float(time)
        ps.add_weight = float(weight)
        ps.add_mask = NO_CLIP if mask is None else rig.masks.index(mask)

    def drive(self, ps: PoseState, state: str, dt: float) -> None:
        """Per frame: play ``state``'s clips (from the table) on every layer,
        then advance the clocks by ``dt`` wall seconds."""
        if state != ps.state:
            ids = self._state_ids(self._rigs[ps.look], state)
            for layer in range(len(ids)):
                self.set_layer_clip(ps, layer, int(ids[layer]))
            ps.state = state
        self.advance(ps, dt)

    def advance(self, ps: PoseState, dt: float) -> None:
        """Advance every clock by ``dt`` x the global rate x the clip's rate;
        loops wrap, one-shots hold their last key; ramps move by
        ``dt / crossfade_s`` and a settled fade drops its source."""
        if dt <= 0.0:
            return
        bank = self._rigs[ps.look].bank
        step = dt * self.settings.rate
        ps.time = _wrap(bank, ps.clip, ps.time + step * bank.rate[ps.clip])
        fading = ps.prev_clip >= 0
        if fading.any():
            pc = np.where(fading, ps.prev_clip, ps.clip)
            ps.prev_time = np.where(
                fading, _wrap(bank, pc, ps.prev_time + step * bank.rate[pc]), 0.0)
            cf = self.settings.crossfade_s
            ps.fade = np.where(fading, np.minimum(1.0, ps.fade + (dt / cf if cf > 0 else 1.0)), 1.0)
            done = ps.fade >= 1.0
            ps.prev_clip = np.where(done, NO_CLIP, ps.prev_clip)
        if ps.add_clip >= 0:
            ac = np.array([ps.add_clip])
            ps.add_time = float(_wrap(bank, ac, np.array([ps.add_time + step * bank.rate[ac[0]]]))[0])

    # --- compose ---------------------------------------------------------
    def compose(self, look: str, states: Sequence[PoseState],
                modifiers: Sequence[PoseModifier] = ()) -> PoseBatch:
        """Compose every state of ONE look in one batch. Compose order is
        fixed: override layers (with their crossfades), selected per bone by
        the mask partition -> additive -> FK -> modifiers -> skin."""
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
        layer_of_mask = np.full(len(rig.masks.names), -1, dtype=np.int64)
        layer_of_mask[rig.layer_masks] = np.arange(L)
        owner_layer = layer_of_mask[rig.masks.owner]
        if (owner_layer < 0).any():
            raise PoseConfigError(f"{look}: bones owned by a mask that is no layer")
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
            amask = np.array([states[i].add_mask for i in an])
            mw = np.where(amask[:, None] >= 0, rig.masks.weights[np.maximum(amask, 0)], 1.0)
            w = (np.array([states[i].add_weight for i in an])[:, None] * mw).astype(np.float32)
            delta = quat_mul(quat_conj(rr), ar)
            ident = np.zeros_like(delta)
            ident[..., 3] = 1.0
            lr = lr.copy()
            lt = lt.copy()
            lr[an] = quat_mul(lr[an], quat_nlerp(ident, delta, w[..., None]))
            lt[an] = lt[an] + (at - rt) * w[..., None]
        g = forward_kinematics(sk, lt, lr, ls)
        for mod in modifiers:
            mod.apply(rig, states, g)
        return PoseBatch(global_trs=global_trs(g), skin=g @ sk.bind_inverse.astype(np.float32),
                         local_t=lt, local_r=lr)

    # --- events (stored only in #162) ------------------------------------
    def queue_event(self, uid: int, event, now: float) -> None:
        q = self._events.get(uid)
        if q is None:
            q = deque(maxlen=self.settings.event_queue)
            self._events[uid] = q
        q.append(event)
        self._event_time[uid] = now

    def events(self, uid: int) -> Tuple:
        return tuple(self._events.get(uid, ()))

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
    k = np.clip(np.floor(f).astype(np.int64), 0, n - 1)
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
            g[:, lvl] = g[:, par] @ local[:, lvl]
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
    "PoseBatch", "PoseModifier", "LookRig", "PoseLayer", "pose_settings",
    "masks_for", "sample_bank", "forward_kinematics", "global_trs",
    "GLOBAL_TRS_WIDTH", "NO_CLIP",
]
