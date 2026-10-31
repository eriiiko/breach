"""The follow camera (#128): a camera that goes where the action is in a
replay (``main.py --replay``) or a watched match (``main.py --control match``).

PURE and raylib-free (the UI split: decisions here, as data; main.py only
polls the cycle key and hands the result to the one ``Camera2D``). It reads
the sim's units and the tick's events, never writes them -- render-only state.

Three parts:

* **Where the action is** -- :class:`ActionTracker` keeps a short, decaying
  history of who was in a fight (every ``UnitHitEvent`` / ``UnitKilledEvent``
  heats its victim and its attacker; the heat e-folds over
  ``action_memory_s`` of SIM time), and :func:`action_target` turns it into
  one point: a weighted mean of the LIVING units' centres, each unit's weight
  its local density (how many living units stand within
  ``action_radius_tiles``) plus ``action_event_weight`` times its heat. So the
  camera goes where units are fighting and, when nothing fights, to the
  biggest crowd. A weighted mean of living units always lies inside their
  convex hull; a dead unit carries no weight at all.
* **What is followed** -- :class:`FollowCamera` cycles ``action`` -> every
  living unit (team, then slot) -> ``free`` -> ``action``; a followed unit that
  dies falls back to ``action``; any manual pan drops to ``free`` (the player
  always wins).
* **How it moves** -- :func:`smooth_step`, an exponential approach with a time
  constant in seconds and a dead zone in tiles, on the VIEW CENTRE; the
  camera's own ``Camera2D.pan`` clamps it to the world. Zoom is never touched.

Settings: ``[replay]`` ``camera`` (``"action"`` / ``"free"``, the mode a
window starts in), ``time_constant_s``, ``dead_zone_tiles``,
``action_memory_s``, ``action_event_weight``, ``action_radius_tiles``, read
by :func:`follow_settings` through ``match_replay.replay_key`` -- the same
refusing door as ``checkpoint_every_ticks``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

#: The modes ``[replay] camera`` may name: where a window's camera starts.
FOLLOW_MODES = ("action", "free")
ACTION = "action"
FREE = "free"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FollowSettings:
    camera: str                  # "action" | "free": the starting target
    time_constant_s: float       # e-fold of the approach, seconds (> 0)
    dead_zone_tiles: float       # no motion while the target is this close
    action_memory_s: float       # e-fold of a fight's heat, sim seconds
    action_event_weight: float   # one hit's weight against one unit's density
    action_radius_tiles: float   # neighbourhood of the density weight


def _number(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float))


def follow_settings(cfg) -> FollowSettings:
    """The follow camera's ``[replay]`` keys of a loaded config (``CFG``),
    each refused when missing or mistyped (``match_replay.replay_key``)."""
    from match_replay import replay_key
    pos = (lambda v: _number(v) and v > 0, "a number > 0")
    nonneg = (lambda v: _number(v) and v >= 0, "a number >= 0")
    camera = replay_key(cfg, "camera", lambda v: v in FOLLOW_MODES,
                        f"one of {', '.join(map(repr, FOLLOW_MODES))}")
    return FollowSettings(
        camera=camera,
        time_constant_s=float(replay_key(cfg, "time_constant_s", *pos)),
        dead_zone_tiles=float(replay_key(cfg, "dead_zone_tiles", *nonneg)),
        action_memory_s=float(replay_key(cfg, "action_memory_s", *pos)),
        action_event_weight=float(replay_key(cfg, "action_event_weight",
                                             *nonneg)),
        action_radius_tiles=float(replay_key(cfg, "action_radius_tiles",
                                             *pos)),
    )


# ---------------------------------------------------------------------------
# Where the action is
# ---------------------------------------------------------------------------
def unit_centre(u) -> Tuple[float, float]:
    """A unit's centre in world tiles (its footprint anchors at ``x, y``)."""
    half = 0.5 * float(getattr(u, "footprint", 1))
    return float(u.x) + half, float(u.y) + half


def living(units: Iterable) -> list:
    return [u for u in units if u.alive]


def action_target(units: Iterable, recent_events: Dict[int, float], *,
                  event_weight: float, radius_tiles: float
                  ) -> Optional[Tuple[float, float]]:
    """The centre of the action: the mean of the living units' centres, each
    weighted by its local density (living units within ``radius_tiles``,
    itself included) plus ``event_weight`` times its fight heat
    ``recent_events[unit.id]`` (an :class:`ActionTracker`'s). None when no
    unit lives. A convex combination of living units: never outside their
    hull, never on a dead unit."""
    alive = living(units)
    if not alive:
        return None
    pts = [unit_centre(u) for u in alive]
    r2 = radius_tiles * radius_tiles
    sx = sy = sw = 0.0
    for u, (x, y) in zip(alive, pts):
        density = sum(1 for (px, py) in pts
                      if (px - x) ** 2 + (py - y) ** 2 <= r2)
        w = density + event_weight * float(recent_events.get(u.id, 0.0))
        sx += w * x
        sy += w * y
        sw += w
    return sx / sw, sy / sw


class ActionTracker:
    """A short, decaying memory of who fought. :meth:`observe` once per sim
    tick with that tick's events; every hit or kill adds 1 to its victim's
    heat and to its attacker's (``attacker_id`` >= 0), and all heat decays by
    ``exp(-dt_s / memory_s)`` per tick. Sim time, so a paused replay keeps its
    picture of the fight."""

    def __init__(self, memory_s: float):
        if not memory_s > 0:
            raise ValueError(f"memory_s must be > 0, got {memory_s!r}")
        self.memory_s = float(memory_s)
        self.heat: Dict[int, float] = {}

    def reset(self) -> None:
        self.heat.clear()

    def observe(self, tick_events: Iterable, dt_s: float) -> None:
        from simulation.events import UnitHitEvent, UnitKilledEvent
        keep = math.exp(-float(dt_s) / self.memory_s)
        heat = {k: v * keep for k, v in self.heat.items() if v * keep > 1e-3}
        for e in tick_events:
            if isinstance(e, (UnitHitEvent, UnitKilledEvent)):
                for uid in (e.unit_id, e.attacker_id):
                    if uid is not None and uid >= 0:
                        heat[uid] = heat.get(uid, 0.0) + 1.0
        self.heat = heat


# ---------------------------------------------------------------------------
# How it moves
# ---------------------------------------------------------------------------
def smooth_step(cam_pos: Tuple[float, float], target_pos: Tuple[float, float],
                dt: float, settings: FollowSettings) -> Tuple[float, float]:
    """One frame of the approach of the view centre ``cam_pos`` to
    ``target_pos`` (tiles): still while the target is within
    ``dead_zone_tiles``; otherwise the distance BEYOND the dead zone shrinks
    by ``1 - exp(-dt / time_constant_s)`` along the line to the target --
    frame-rate independent, monotone, never past the target."""
    cx, cy = cam_pos
    dx, dy = target_pos[0] - cx, target_pos[1] - cy
    d = math.hypot(dx, dy)
    excess = d - settings.dead_zone_tiles
    if excess <= 0.0 or dt <= 0.0:
        return cx, cy
    move = excess * (1.0 - math.exp(-dt / settings.time_constant_s))
    return cx + dx / d * move, cy + dy / d * move


# ---------------------------------------------------------------------------
# What is followed
# ---------------------------------------------------------------------------
def cycle_order(units: Iterable) -> list:
    """The living units in cycle order: team, then slot (then id)."""
    return sorted(living(units), key=lambda u: (int(u.team), int(u.slot),
                                                int(u.id)))


class FollowCamera:
    """What the camera follows and where it moves: ``ACTION``, a unit id, or
    ``FREE``. One per window; main.py calls :meth:`observe` per sim tick,
    :meth:`cycle` on the cycle key, :meth:`manual_pan` when the player pans,
    and :meth:`apply` once per frame."""

    def __init__(self, settings: FollowSettings):
        self.settings = settings
        self.tracker = ActionTracker(settings.action_memory_s)
        self.target = ACTION if settings.camera == ACTION else FREE
        self._snap = True

    # -- state changes ----------------------------------------------------
    def reset(self) -> None:
        """A new sim (a replay's rebuild / seek): forget the fight, and snap
        to the target on the next frame instead of flying there."""
        self.tracker.reset()
        self._snap = True

    def observe(self, tick_events, dt_s: float) -> None:
        self.tracker.observe(tick_events, dt_s)

    def manual_pan(self) -> None:
        self.target = FREE

    def cycle(self, units) -> None:
        """``action`` -> each living unit (team, slot) -> ``free`` ->
        ``action``. A dead current unit counts as ``action`` (it already fell
        back there)."""
        order = cycle_order(units)
        cur = self._resolved(units)
        if cur == FREE:
            self.target = ACTION
        elif cur == ACTION:
            self.target = order[0].id if order else FREE
        else:
            ids = [u.id for u in order]
            i = ids.index(cur)
            self.target = ids[i + 1] if i + 1 < len(ids) else FREE

    # -- reads ------------------------------------------------------------
    def _resolved(self, units):
        """The target as it stands: a followed unit that died is ``action``
        (and stays so)."""
        if self.target in (ACTION, FREE):
            return self.target
        u = next((u for u in units if u.id == self.target), None)
        if u is None or not u.alive:
            self.target = ACTION
        return self.target

    def label(self, units) -> str:
        """The HUD's ``cam: ...`` value: ``action``, the unit's name, ``free``."""
        t = self._resolved(units)
        if t in (ACTION, FREE):
            return t
        u = next(u for u in units if u.id == t)
        return str(getattr(u, "name", f"unit {t}"))

    def hud_line(self, units) -> str:
        return f"cam: {self.label(units)}"

    def target_pos(self, units) -> Optional[Tuple[float, float]]:
        t = self._resolved(units)
        if t == FREE:
            return None
        if t == ACTION:
            return action_target(
                units, self.tracker.heat,
                event_weight=self.settings.action_event_weight,
                radius_tiles=self.settings.action_radius_tiles)
        return unit_centre(next(u for u in units if u.id == t))

    # -- the frame --------------------------------------------------------
    def apply(self, camera, units, dt: float) -> None:
        """Move ``camera`` (a ``renderer.camera.Camera2D``) one frame toward
        the target -- its centre, through ``Camera2D.pan``, which clamps to
        the world. ``FREE`` (or nothing alive) leaves it alone."""
        target = self.target_pos(units)
        if target is None:
            return
        w, h = camera.visible_tiles()
        centre = (camera.pos_tile_x + 0.5 * w, camera.pos_tile_y + 0.5 * h)
        if self._snap:
            new = target
            self._snap = False
        else:
            new = smooth_step(centre, target, dt, self.settings)
        camera.pan(new[0] - centre[0], new[1] - centre[1])


__all__ = ["FOLLOW_MODES", "ACTION", "FREE", "FollowSettings",
           "follow_settings", "unit_centre", "action_target",
           "ActionTracker", "smooth_step", "cycle_order", "FollowCamera"]
