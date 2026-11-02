"""Pose modifiers -- the pose layer's procedural post-FK stages (M13, #163).

Pure numpy, render-only, pyray-free. A modifier edits a composed batch's
GLOBAL (model-space) bone matrices in place, after FK and before skinning
(design ``docs/animation_milestones_design_2026-11-02.md`` §4.5 correction 3:
base -> overrides -> additives -> procedural modifiers -> skin), and moves
whole subtrees, so nothing below a turned bone is left behind.

:class:`AimTravelTwist` -- aim vs travel (§4.5 correction 7). The body is
drawn yawed by ``unit.facing``; a unit that walks one way and aims another
would skate sideways (the clip pack has no strafe or backpedal clips). So
the LOWER body (the twist's ``lower`` bone and its subtree: hips, legs) is
yawed toward the travel direction, within ``+-theta`` of the facing, and a
chain of ``counter`` bones (the spine) turns the torso back by the same
angle, in shares -- so the torso, arms and head keep the facing exactly.
Past ``theta`` the legs BACKPEDAL: the leg layer's clip runs backwards
(``PoseState.layer_rate`` -1) and the yaw is reflected (the legs face away
from the travel), so the feet keep moving the way the body goes.

:func:`steer` turns the renderer's per-frame travel reading into the twist:
the travel direction (relative to the facing) is low-passed (it is a noisy
position delta, the renderer having no velocity field), the forward /
backpedal switch has hysteresis, and the twist moves at a bounded angular
speed, so a switch swings the legs round instead of popping them.

Skeleton-agnostic: the bones are data (``[render.pose.skeletons.*.twist]``),
resolved and checked per look in :meth:`AimTravelTwist.prepare`; no bone is
named here. The vertical is the model's +Y (glTF is Y-up, spec §3.4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .pose_layer import (AimTravelSettings, LookRig, PoseConfigError, PoseState,
                         TwistRule)

# glTF is Y-up: the model's vertical (the axis the renderer yaws a unit
# about) is +Y; a yaw by ``a`` takes forward +Z to (sin a, 0, cos a).
_UP_AXIS = 1


@dataclass(frozen=True)
class _ResolvedTwist:
    lower: int                                   # the lower bone's index
    lower_sub: np.ndarray                        # bone indices it carries
    counter: Tuple[Tuple[int, np.ndarray, float], ...]   # (bone, subtree, share)


def _yaw_about(angle: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """``(n, 4, 4)`` matrices rotating by ``angle[k]`` about the vertical
    line through ``pivot[k]`` (model space)."""
    n = len(angle)
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros((n, 4, 4), dtype=np.float32)
    m[:, 0, 0] = c
    m[:, 0, 2] = s
    m[:, 2, 0] = -s
    m[:, 2, 2] = c
    m[:, _UP_AXIS, _UP_AXIS] = 1.0
    m[:, 3, 3] = 1.0
    # translation = p - R p (the pivot stays put)
    px, pz = pivot[:, 0], pivot[:, 2]
    m[:, 0, 3] = px - (c * px + s * pz)
    m[:, 2, 3] = pz - (-s * px + c * pz)
    return m


def _move_subtree(g: np.ndarray, m: np.ndarray, bones: np.ndarray) -> None:
    """``g[:, bones] = m @ g[:, bones]`` per unit, without BLAS (the frame-
    path rule: plain einsum, no optimizer)."""
    g[:, bones] = np.einsum("nij,nbjk->nbik", m, g[:, bones])


class AimTravelTwist:
    """The aim-vs-travel post-FK modifier (see the module docstring). Reads
    ``PoseState.twist`` (radians about the model's vertical; 0 = no change,
    and such units are skipped, so their pose is bit-identical)."""

    def __init__(self, twists: Mapping[str, TwistRule]) -> None:
        self._twists = dict(twists)
        self._rigs: Dict[str, _ResolvedTwist] = {}

    def prepare(self, rig: LookRig) -> None:
        """Resolve the twist's bones on ``rig``'s skeleton (its mask table's
        twist rule). Refuses, naming it: a bone the skeleton lacks, a counter
        bone outside the lower bone's subtree, or counter bones that are not
        one chain (each inside the previous one's subtree) -- else the torso
        would not end on the facing."""
        where = f"[render.pose.skeletons.{rig.skeleton_table}] twist"
        rule = self._twists.get(rig.skeleton_table)
        if rule is None:
            raise PoseConfigError(f"{where} is missing")
        sk = rig.bank.skeleton

        def bone(name: str) -> int:
            if name not in sk.names:
                raise PoseConfigError(f"{where}: unknown bone {name!r} (not in {sk.source})")
            return sk.index(name)

        lower = bone(rule.lower)
        lower_sub = sk.subtree(lower)
        counter: List[Tuple[int, np.ndarray, float]] = []
        inside = lower_sub
        for name, share in zip(rule.counter, rule.shares):
            c = bone(name)
            if c == lower or not inside[c]:
                raise PoseConfigError(
                    f"{where}: counter bone {name!r} is not below "
                    f"{rule.lower if not counter else rule.counter[len(counter) - 1]!r} "
                    "(the counter bones must be one chain inside the lower bone's subtree)")
            sub = sk.subtree(c)
            counter.append((c, np.flatnonzero(sub), float(share)))
            inside = sub
        self._rigs[rig.name] = _ResolvedTwist(lower=lower,
                                              lower_sub=np.flatnonzero(lower_sub),
                                              counter=tuple(counter))
        rig.twist_layer = int(rig.owner_layer[lower])

    def apply(self, rig: LookRig, states: Sequence[PoseState], g: np.ndarray) -> None:
        twist = np.array([ps.twist for ps in states], dtype=np.float64)
        rows = np.flatnonzero(twist != 0.0)
        if rows.size == 0:
            return
        r = self._rigs[rig.name]
        d = twist[rows]
        gs = g[rows]
        _move_subtree(gs, _yaw_about(d, gs[:, r.lower, :3, 3]), r.lower_sub)
        for c, sub, share in r.counter:          # the chain, root-most first
            _move_subtree(gs, _yaw_about(-share * d, gs[:, c, :3, 3]), sub)
        g[rows] = gs


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def steer(ps: PoseState, a: AimTravelSettings, legs_layer: int,
          travel: Optional[float], moving: bool, dt: float) -> None:
    """One frame of the aim-vs-travel input.

    ``travel``: the unit's travel direction this frame RELATIVE to its
    facing, as a yaw about the model's vertical (radians), or None when it
    did not move measurably this frame. A unit that is not moving at all
    steers its legs back under the body (relative travel 0); one that is
    moving but showed no delta this frame (frames between sim ticks) holds.

    The relative travel is low-passed as a vector (time constant
    ``smooth_s``; a vector, so it never wraps the long way and a reversal
    passes through, never stalls); its angle ``alpha`` picks forward
    (``|alpha| <= theta``, the legs yaw to ``alpha``) or backpedal (past
    ``theta + hysteresis``; the leg clip plays backwards and the legs yaw to
    ``alpha - pi``, i.e. they face away from the travel), each clamped to
    ``+-theta``; the twist moves toward that goal at most ``twist_rate * dt``.
    """
    if dt <= 0.0:
        return
    if travel is not None:
        target = (math.cos(travel), math.sin(travel))
    elif not moving:
        target = (1.0, 0.0)
    else:
        target = None
    if target is not None:
        k = 1.0 if a.smooth_s <= 0.0 else 1.0 - math.exp(-dt / a.smooth_s)
        ps.travel = ps.travel + (np.array(target) - ps.travel) * k
    alpha = math.atan2(float(ps.travel[1]), float(ps.travel[0]))
    if ps.back:
        if abs(alpha) < a.theta - a.hysteresis:
            ps.back = False
    elif abs(alpha) > a.theta + a.hysteresis:
        ps.back = True
    goal = _wrap_pi(alpha - math.pi) if ps.back else alpha
    goal = min(a.theta, max(-a.theta, goal))
    step = a.twist_rate * dt
    ps.twist += min(step, max(-step, goal - ps.twist))
    ps.layer_rate[legs_layer] = -1.0 if ps.back else 1.0


__all__ = ["AimTravelTwist", "steer"]
