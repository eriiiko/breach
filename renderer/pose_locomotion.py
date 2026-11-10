"""Locomotion -- gait, direction and playback rate from the unit's DRAWN
speed (#184; the pose layer, M13). Pure numpy, render-only, pyray-free.

A skeleton table may carry a ``locomotion`` table
(``[render.pose.skeletons.<s>.locomotion]``); a unit of a role it names is
then driven by what it is seen to do on screen instead of the renderer's
idle / walk guess:

* **Drawn speed.** The renderer hands over how far the unit moved since the
  last frame (model metres: tiles x the drawn tile's model metres). Units
  move once per SIM TICK, so most frames between two ticks show no delta:
  a sample is the distance since the last change over the wall time since
  it, low-passed (``speed_smooth_s``); a unit is stopped only after
  ``stop_after_s`` with no change (so the frames between ticks never read
  as standing). A jump past ``teleport_m`` is not walking.
* **Stride speed.** Every looping clip's in-place stride speed is MEASURED
  at load from the clip itself (:func:`measure_gait`: the planted feet's
  median speed against the ground, along the clip's own travel), with the
  clip's travel direction and its gait phase (when the left foot is
  highest). A clip's playback rate is then ``drawn speed / stride speed`` --
  never a literal, so the feet do not skate at any speed the sim chooses.
* **Gait.** The role's ``gaits`` rows, slowest first (walk, run, sprint).
  The choice is RELATIVE TO THE CLIPS, never a speed in m/s: a gait steps
  up when its own clip would have to play faster than its ``max_rate``
  (times ``1 + hysteresis``) and back down when the slower gait's clip
  would play under its ``max_rate`` (times ``1 - hysteresis``); the first
  gait starts above ``min_rate`` and stops below it, likewise.
* **Direction.** A gait with ``sectors`` picks one state per sector of the
  angle between travel and facing (``center_deg``, + = the unit's left;
  each sector reaches half-way to its neighbours, sticky by
  ``sector_hysteresis_deg``). The aim-vs-travel twist then turns the legs
  by the RESIDUAL only -- the travel against the playing clip's own
  measured travel -- so a strafe clip needs no 90 deg twist. A gait without
  sectors keeps the two-sector rule (forward within ``theta``, backpedal
  past it) of ``[render.pose.aim_travel]``.
* **Phase.** All looping gait clips of a unit share ONE gait phase, advanced
  by the distance walked over the playing legs clip's cycle length; each
  clip's time is that phase at its own offset. So a direction or gait
  switch carries the step over (the feet do not pop; the crossfade still
  ramps), and an upper-body clip of another gait (the rifle carry over the
  run) swings in step with the legs.

No bone is named here: the feet are the table's ``feet``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# a looping clip whose planted feet move slower than this (model m/s) does
# not stride (an idle, a held aim): it is no gait clip and is never phase-locked
STRIDE_MIN = 0.2
# the fraction of a clip's keys, lowest first, a foot counts as planted in
STANCE_SHARE = 0.30
# a delta below this (model metres) is no movement (float noise)
_MOVE_EPS_M = 1e-6

_LOCO_KEYS = ("feet", "roles")
_ROLE_KEYS = ("idle", "hysteresis", "sector_hysteresis_deg", "speed_smooth_s",
              "stop_after_s", "teleport_m", "gaits")
_GAIT_KEYS = ("state", "min_rate", "max_rate", "sectors")
_SECTOR_KEYS = ("state", "center_deg")


class LocomotionConfigError(ValueError):
    """A ``locomotion`` setting refused (names the key)."""


# ---------------------------------------------------------------------------
# Measured from the clip
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaitMeasure:
    """One looping clip's gait, measured from its keys (:func:`measure_gait`).

    ``stride`` model m/s of clip time (the planted feet's speed against the
    ground); ``travel`` the direction the clip walks, a yaw about the
    model's vertical (radians; 0 = forward +z, + = toward +x, the body's
    left); ``offset`` the clip time, as a fraction of its cycle, at which
    the first foot is highest (the gait phase's zero); ``cycle_m`` the
    distance one cycle covers (``stride x duration``)."""
    stride: float
    travel: float
    offset: float
    cycle_m: float


def measure_gait(bank, clip: int, feet: Sequence[int]) -> Optional[GaitMeasure]:
    """THE stride measurement (#184; was ``tests/test_pose_soma.py``'s
    ``_stance_speed``): FK over every key of ``clip``; per foot, its keys in
    the lowest ``STANCE_SHARE`` of its height are planted; the mean planted
    velocity (horizontal) gives the direction the ground moves under the
    body (the clip travels the other way) and the median planted speed
    along it the stride speed. The phase is the fundamental of the feet's
    height difference (first foot minus second) over one loop. None for a
    clip that does not stride (``< STRIDE_MIN``) or does not loop."""
    from .pose_layer import forward_kinematics
    if not bank.loop[clip]:
        return None
    sk = bank.skeleton
    o, k = int(bank.offset[clip]), int(bank.n_keys[clip])
    if k < 4:
        return None
    pos = forward_kinematics(sk, bank.t[o:o + k], bank.r[o:o + k], None)[..., :3, 3]
    pos = pos.astype(np.float64)
    dt = float(bank.key_dt[clip])
    planted = []
    for j in feet:
        y = pos[:, j, 1]
        v = np.gradient(pos[:, j][:, [0, 2]], dt, axis=0)
        planted.append(v[y <= np.percentile(y, STANCE_SHARE * 100.0)])
    v = np.concatenate(planted)
    mean = v.mean(0)
    n = float(np.linalg.norm(mean))
    if n < 1e-9:
        return None
    ground = mean / n                               # (x, z) the ground moves to
    stride = float(np.median(v @ ground))
    if not stride >= STRIDE_MIN:
        return None
    travel = math.atan2(-ground[0], -ground[1])
    # the loop's last key repeats its first: one cycle is keys 0 .. k-2
    d = pos[:-1, feet[0], 1] - pos[:-1, feet[-1], 1]
    m = len(d)
    c = np.sum((d - d.mean()) * np.exp(-2j * math.pi * np.arange(m) / m))
    offset = float((-np.angle(c) / (2.0 * math.pi)) % 1.0)
    return GaitMeasure(stride=stride, travel=travel, offset=offset,
                       cycle_m=stride * float(bank.duration[clip]))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectorRow:
    state: str
    center: float                 # radians, + = the unit's left


@dataclass(frozen=True)
class GaitRow:
    state: str
    min_rate: float               # first gait only (0 = none on the others)
    max_rate: Optional[float]     # None on the last gait
    sectors: Tuple[SectorRow, ...] = ()


@dataclass(frozen=True)
class RoleLocomotion:
    idle: str
    hysteresis: float             # fraction of a rate threshold
    sector_hysteresis: float      # radians
    speed_smooth_s: float         # the drawn speed's low-pass
    stop_after_s: float           # no delta this long = stopped (> one move's gap)
    teleport_m: float             # a delta this long is no walking
    gaits: Tuple[GaitRow, ...]


@dataclass(frozen=True)
class LocomotionRule:
    """One skeleton table's ``locomotion`` (bone names, unresolved)."""
    feet: Tuple[str, ...]
    roles: Dict[str, RoleLocomotion]

    def states(self) -> Tuple[str, ...]:
        out = []
        for r in self.roles.values():
            out.append(r.idle)
            for g in r.gaits:
                out.append(g.state)
                out.extend(s.state for s in g.sectors)
        return tuple(dict.fromkeys(out))


def _exact(d, keys: Sequence[str], where: str, optional: Sequence[str] = ()) -> dict:
    if not isinstance(d, dict):
        raise LocomotionConfigError(f"{where} must be a table")
    for key in keys:
        if key not in d and key not in optional:
            raise LocomotionConfigError(f"{where} is missing {key!r}")
    bad = sorted(set(d) - set(keys))
    if bad:
        raise LocomotionConfigError(f"{where} has unknown keys {bad}")
    return d


def _num(v, where: str, lo: float, lo_open: bool = False) -> float:
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and \
        (v > lo if lo_open else v >= lo)
    if not ok:
        raise LocomotionConfigError(
            f"{where} must be a number {'>' if lo_open else '>='} {lo}, got {v!r}")
    return float(v)


def _str(v, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise LocomotionConfigError(f"{where} must be a non-empty string, got {v!r}")
    return v


def parse_locomotion(raw, where: str, roles: Sequence[str],
                     states: Sequence[str]) -> LocomotionRule:
    """A ``locomotion`` table. ``roles`` / ``states``: the names it may use
    (``[render.pose.roles]``; the global states and the skeleton's own).
    Every key is required but ``min_rate`` (first gait only, required
    there), ``max_rate`` (every gait but the last, required there) and
    ``sectors``; a missing, unknown, mistyped or misplaced one is refused,
    naming it."""
    d = _exact(raw, _LOCO_KEYS, where)
    feet = d["feet"]
    if not isinstance(feet, list) or len(feet) != 2 or \
            not all(isinstance(f, str) and f for f in feet):
        raise LocomotionConfigError(f"{where} feet must be two bone names (the first "
                                    f"foot's swing is the gait phase's zero), got {feet!r}")
    rr = d["roles"]
    if not isinstance(rr, dict) or not rr:
        raise LocomotionConfigError(f"{where} roles must hold at least one role table")
    out: Dict[str, RoleLocomotion] = {}
    for role, row in rr.items():
        rw = f"{where} roles.{role}"
        if role not in roles:
            raise LocomotionConfigError(f"{rw}: {role!r} is not a [render.pose.roles] "
                                        f"role {sorted(roles)}")
        r = _exact(row, _ROLE_KEYS, rw)
        idle = _str(r["idle"], f"{rw} idle")
        hyst = _num(r["hysteresis"], f"{rw} hysteresis", 0.0)
        if hyst >= 1.0:
            raise LocomotionConfigError(f"{rw} hysteresis must be below 1, got {hyst!r}")
        shyst = math.radians(_num(r["sector_hysteresis_deg"],
                                  f"{rw} sector_hysteresis_deg", 0.0))
        graw = r["gaits"]
        if not isinstance(graw, list) or not graw:
            raise LocomotionConfigError(f"{rw} gaits must be a non-empty list of rows")
        gaits = []
        for i, g in enumerate(graw):
            gw = f"{rw} gaits[{i}]"
            g = _exact(g, _GAIT_KEYS, gw, optional=("min_rate", "max_rate", "sectors"))
            last = i == len(graw) - 1
            if (i == 0) != ("min_rate" in g):
                raise LocomotionConfigError(f"{gw}: min_rate belongs on the first gait "
                                            "only, and is required there")
            if last == ("max_rate" in g):
                raise LocomotionConfigError(f"{gw}: max_rate belongs on every gait but the "
                                            "last, and is required there")
            min_rate = _num(g["min_rate"], f"{gw} min_rate", 0.0, lo_open=True) \
                if "min_rate" in g else 0.0
            max_rate = _num(g["max_rate"], f"{gw} max_rate", 0.0, lo_open=True) \
                if "max_rate" in g else None
            secs = []
            sraw = g.get("sectors", [])
            if not isinstance(sraw, list):
                raise LocomotionConfigError(f"{gw} sectors must be a list of rows")
            for k, s in enumerate(sraw):
                sw = f"{gw} sectors[{k}]"
                s = _exact(s, _SECTOR_KEYS, sw)
                c = s["center_deg"]
                if not isinstance(c, (int, float)) or isinstance(c, bool) or \
                        not -180.0 <= c <= 180.0:
                    raise LocomotionConfigError(f"{sw} center_deg must be a number in "
                                                f"[-180, 180], got {c!r}")
                secs.append(SectorRow(state=_str(s["state"], f"{sw} state"),
                                      center=math.radians(float(c))))
            centres = [round(math.degrees(s.center)) % 360 for s in secs]
            if len(set(centres)) != len(centres):
                raise LocomotionConfigError(f"{gw} sectors have two rows at one centre")
            gaits.append(GaitRow(state=_str(g["state"], f"{gw} state"), min_rate=min_rate,
                                 max_rate=max_rate,
                                 sectors=tuple(sorted(secs, key=lambda s: s.center))))
        out[role] = RoleLocomotion(
            idle=idle, hysteresis=hyst, sector_hysteresis=shyst,
            speed_smooth_s=_num(r["speed_smooth_s"], f"{rw} speed_smooth_s", 0.0),
            stop_after_s=_num(r["stop_after_s"], f"{rw} stop_after_s", 0.0, lo_open=True),
            teleport_m=_num(r["teleport_m"], f"{rw} teleport_m", 0.0, lo_open=True),
            gaits=tuple(gaits))
    rule = LocomotionRule(feet=tuple(feet), roles=out)
    unknown = [s for s in rule.states() if s not in states]
    if unknown:
        raise LocomotionConfigError(f"{where} names states {unknown} that are neither "
                                    f"[render.pose.states] rows nor the skeleton's own")
    return rule


# ---------------------------------------------------------------------------
# Resolved on one look
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedSector:
    state: str
    center: float
    lo: float                     # half-way to the previous sector (radians, > 0)
    hi: float                     # half-way to the next
    legs_clip: int


@dataclass(frozen=True)
class ResolvedGait:
    state: str
    min_rate: float
    max_rate: Optional[float]
    legs_clip: int                # the state's legs clip (no sectors)
    sectors: Tuple[ResolvedSector, ...] = ()


@dataclass(frozen=True)
class ResolvedRole:
    idle: str
    hysteresis: float
    sector_hysteresis: float
    speed: RoleLocomotion         # its speed estimator's dials
    gaits: Tuple[ResolvedGait, ...]


@dataclass
class ResolvedLocomotion:
    """A look's locomotion: the measured gait clips and the role rows with
    their legs clips."""
    rule: LocomotionRule
    legs_layer: int
    measures: Dict[int, GaitMeasure] = field(default_factory=dict)
    roles: Dict[str, ResolvedRole] = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _sector_bounds(centres: Sequence[float]) -> Tuple[Tuple[float, float], ...]:
    n = len(centres)
    if n == 1:
        return ((math.pi, math.pi),)
    out = []
    for i, c in enumerate(centres):
        prev = centres[i - 1] - (2.0 * math.pi if i == 0 else 0.0)
        nxt = centres[(i + 1) % n] + (2.0 * math.pi if i == n - 1 else 0.0)
        out.append(((c - prev) / 2.0, (nxt - c) / 2.0))
    return tuple(out)


def resolve(rule: LocomotionRule, rig, state_ids: Callable[[str, str], np.ndarray],
            where: str) -> ResolvedLocomotion:
    """Resolve ``rule`` on ``rig`` (a ``LookRig`` without its locomotion):
    the feet, every looping clip's gait (:func:`measure_gait`), and each
    role's states to their legs clip. ``state_ids(state, role)`` is the
    layer's state lookup. Refuses, naming it: a foot the skeleton lacks, the
    two feet on different layers, and a gait state whose legs clip does not
    stride. A SECTOR whose legs clip does not stride on this look (the look
    lacks the clip: its state fell back) is dropped, with a warning; a gait
    left with fewer than two sectors keeps the two-sector rule."""
    sk = rig.bank.skeleton
    feet = []
    for name in rule.feet:
        if name not in sk.names:
            raise LocomotionConfigError(f"{where}: unknown foot bone {name!r} "
                                        f"(not in {sk.source})")
        feet.append(sk.index(name))
    layers = {int(rig.owner_layer[j]) for j in feet}
    if len(layers) != 1:
        raise LocomotionConfigError(f"{where}: the feet {list(rule.feet)} are on "
                                    "different layers")
    legs = layers.pop()
    res = ResolvedLocomotion(rule=rule, legs_layer=legs)
    bank = rig.bank
    for c in range(bank.n_clips):
        m = measure_gait(bank, c, feet)
        if m is not None:
            res.measures[c] = m

    def legs_clip(state: str, role: str, what: str) -> int:
        c = int(state_ids(state, role)[legs])
        if c not in res.measures:
            raise LocomotionConfigError(
                f"{where} roles.{role} {what} {state!r}: its legs clip "
                f"{bank.names[c]!r} does not stride (a gait state's legs must walk)")
        return c

    for role, rl in rule.roles.items():
        gaits = []
        for g in rl.gaits:
            kept = []
            for s in g.sectors:
                c = int(state_ids(s.state, role)[legs])
                if c in res.measures:
                    kept.append((s, c))
                else:
                    res.warnings.append(
                        f"{where} roles.{role} sector {s.state!r}: its legs clip "
                        f"{bank.names[c]!r} does not stride on this look; sector dropped")
            if len(kept) < 2:
                kept = []
            secs = []
            bounds = _sector_bounds([s.center for s, _ in kept])
            for (s, c), (lo, hi) in zip(kept, bounds):
                secs.append(ResolvedSector(state=s.state, center=s.center, lo=lo, hi=hi,
                                           legs_clip=c))
            gaits.append(ResolvedGait(state=g.state, min_rate=g.min_rate,
                                      max_rate=g.max_rate,
                                      legs_clip=legs_clip(g.state, role, "gait"),
                                      sectors=tuple(secs)))
        res.roles[role] = ResolvedRole(idle=rl.idle, hysteresis=rl.hysteresis,
                                       sector_hysteresis=rl.sector_hysteresis,
                                       speed=rl, gaits=tuple(gaits))
    return res


# ---------------------------------------------------------------------------
# Per frame
# ---------------------------------------------------------------------------

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def observe_speed(ps, rule: RoleLocomotion, moved_m: float, dt: float,
                  start_span_s: float) -> None:
    """Update ``ps.speed`` (model m per wall s) from this frame's position
    delta ``moved_m``. A frame without a delta holds the estimate (between
    two sim ticks) until ``stop_after_s`` has passed without one: then the
    unit is stopped (0). Each delta is a sample of (distance, wall time
    since the last delta -- or ``start_span_s``, one sim tick, for the first
    delta after a stop); distance and time are low-passed APART (time
    constant ``speed_smooth_s``, one weight per sample) and the speed is
    their ratio -- a mean of per-sample ratios would read fast, the frames
    between ticks alternating 2 and 3 at 60 fps and 24 tps."""
    ps.motion_acc += max(dt, 0.0)
    if moved_m > _MOVE_EPS_M:
        if moved_m > rule.teleport_m:
            ps.speed = ps.motion_d = ps.motion_t = 0.0
            ps.motion_acc = 0.0
            return
        stopped = ps.speed <= 0.0 or ps.motion_acc > rule.stop_after_s
        span = max(start_span_s if stopped else ps.motion_acc, 1e-6)
        if stopped or rule.speed_smooth_s <= 0.0:
            ps.motion_d, ps.motion_t = moved_m, span
        else:
            # one weight per sample (by the SMOOTHED span: a weight that
            # followed each sample's own span would favour the long ones)
            k = 1.0 - math.exp(-ps.motion_t / rule.speed_smooth_s)
            ps.motion_d += (moved_m - ps.motion_d) * k
            ps.motion_t += (span - ps.motion_t) * k
        ps.speed = ps.motion_d / ps.motion_t
        ps.motion_acc = 0.0
    elif ps.motion_acc > rule.stop_after_s:
        ps.speed = 0.0


def _rate(res: ResolvedLocomotion, clip: int, speed: float) -> float:
    return speed / res.measures[clip].stride


def _sector_of(g: ResolvedGait, alpha: float, current: int, hyst: float) -> int:
    """The sector of ``alpha``: the current one while ``alpha`` stays within
    its reach plus ``hyst``, else the one whose reach holds it."""
    if 0 <= current < len(g.sectors):
        s = g.sectors[current]
        r = _wrap(alpha - s.center)
        if -s.lo - hyst <= r <= s.hi + hyst:
            return current
    for i, s in enumerate(g.sectors):
        r = _wrap(alpha - s.center)
        if -s.lo <= r <= s.hi:
            return i
    return 0


def _gait_legs(g: ResolvedGait, sector: int) -> int:
    return g.sectors[sector].legs_clip if g.sectors else g.legs_clip


def locomote(ps, res: ResolvedLocomotion, role: ResolvedRole, aim_travel,
             moved_m: float, travel: Optional[float], dt: float,
             start_span_s: float) -> str:
    """One frame of a locomoting unit: its drawn speed, its gait (by the
    clips' rates, with hysteresis), its sector, the legs' twist (the
    residual against the playing clip's own travel; the two-sector rule on
    a gait without sectors) and its gait phase. Returns the state to drive.
    ``travel``: the relative travel this frame (as :func:`pose_modifiers.steer`
    takes it), or None without a delta."""
    from .pose_modifiers import backpedal_goal, lowpass_travel, slew_twist
    if dt <= 0.0:
        return ps.loco_state or role.idle
    observe_speed(ps, role.speed, moved_m, dt, start_span_s)
    moving = ps.speed > 0.0
    if ps.gait < 0 and moving and travel is not None:
        ps.travel = np.array([math.cos(travel), math.sin(travel)])   # a start: at once
    alpha = lowpass_travel(ps, aim_travel, travel, moving, dt)
    h = role.hysteresis
    gaits = role.gaits

    def sector_for(i: int) -> int:
        g = gaits[i]
        if not g.sectors:
            return 0
        cur = ps.sector if ps.gait == i else -1
        return _sector_of(g, alpha, cur, role.sector_hysteresis)

    def rate(i: int) -> float:
        return _rate(res, _gait_legs(gaits[i], sector_for(i)), ps.speed)

    g = ps.gait
    if g < 0:
        if moving and rate(0) > gaits[0].min_rate * (1.0 + h):
            g = 0
    elif g == 0 and rate(0) < gaits[0].min_rate * (1.0 - h):
        g = -1
    if g >= 0:
        while g + 1 < len(gaits) and rate(g) > gaits[g].max_rate * (1.0 + h):
            g += 1
        while g > 0 and rate(g - 1) < gaits[g - 1].max_rate * (1.0 - h):
            g -= 1
    if g != ps.gait and g >= 0 and gaits[g].sectors:
        ps.sector = -1
    sector = sector_for(g) if g >= 0 else 0
    ps.gait, ps.sector = g, sector
    if g < 0:
        ps.back = False
        slew_twist(ps, aim_travel, 0.0, dt)
        state, legs, direction = role.idle, None, 1.0
    else:
        gr = gaits[g]
        legs = _gait_legs(gr, sector)
        if gr.sectors:
            s = gr.sectors[sector]
            hy = role.sector_hysteresis
            r = min(s.hi + hy, max(-s.lo - hy, _wrap(alpha - s.center)))
            ps.back = False
            slew_twist(ps, aim_travel, _wrap(r + s.center - res.measures[legs].travel), dt)
            state, direction = s.state, 1.0
        else:
            slew_twist(ps, aim_travel, backpedal_goal(ps, aim_travel, alpha), dt)
            state, direction = gr.state, (-1.0 if ps.back else 1.0)
    ps.layer_rate[res.legs_layer] = direction
    if legs is not None:
        ps.gait_phase = (ps.gait_phase + direction * ps.speed * dt
                         / res.measures[legs].cycle_m) % 1.0
    ps.loco_state = state
    return state


def lock_phase(ps, res: ResolvedLocomotion, bank) -> None:
    """Every looping gait clip on every layer (and every crossfade source)
    shows the unit's gait phase at its own offset -- the lock that carries
    the step over a switch. Called after the layer's own clock advance."""
    for arr_clip, arr_time in ((ps.clip, ps.time), (ps.prev_clip, ps.prev_time)):
        for layer in range(len(arr_clip)):
            m = res.measures.get(int(arr_clip[layer]))
            if m is not None:
                arr_time[layer] = ((ps.gait_phase + m.offset) % 1.0) \
                    * float(bank.duration[int(arr_clip[layer])])


__all__ = ["GaitMeasure", "measure_gait", "LocomotionRule", "RoleLocomotion",
           "GaitRow", "SectorRow", "LocomotionConfigError", "parse_locomotion",
           "ResolvedLocomotion", "ResolvedRole", "ResolvedGait", "ResolvedSector",
           "resolve", "observe_speed", "locomote", "lock_phase", "STRIDE_MIN",
           "STANCE_SHARE"]
