"""``BotTeam`` — the rule-based bot team, the match controller ``"bots"`` (#121).

rl-ready scope §5 item 3: the first opponent and the baseline for every later
agent. One :class:`~control_source.ControlSource` per team, built by the match
controller registry (``control_source.create_controller("bots", team)``),
called once per tick with ``renderer=None`` by ``tools/run_match.py`` (and by
``main.py --control match`` through :class:`control_match.MatchObserver`).

It drives EVERY living unit of its team through the SAME per-tick intent
facade the gamepad uses (``Simulation.set_move_dir`` / ``clear_move_dir`` /
``set_aim`` / ``set_trigger`` / ``throw_grenade_intent``) — never a parallel
movement or firing path. A new behaviour is a rule (a method) here, never a
second controller path.

The rules, one method each, evaluated per unit per tick:

1. **Shoot on sight** (:meth:`_pick_target`) — the nearest enemy the unit
   itself sees through the sight oracle (``vision.state_for``), within its
   weapon's range, whose line of fire has no teammate in its first
   ``friendly_line_tiles``; the unit stops, aims at it and holds the trigger.
2. **Grenade a group** (:meth:`_grenade_rule`) — two or more seen enemies
   within ``group_radius`` of each other, the group centre farther than
   ``danger_radius`` from the thrower and from every teammate, within the
   hand grenade's throw range, a grenade in hand and the per-bot cooldown run
   out: lob at the centre.
3. **Fall back** (:meth:`_falling_back`) — below ``fallback_hp_frac`` of
   vitality with no living teammate within ``fallback_r``: walk home (own
   spawn room), still shooting what it sees.
4. **Advance** (:meth:`_advance`) — nothing in sight: walk to the team's
   last known enemy position (any teammate's sighting), else to the enemy
   spawn room's centre; A* (``pathfinding.astar``, the only pathfinder)
   replanned every ``replan_ticks``, on arrival or on a new goal, followed
   with MOVE_DIR at the walking speed mode.
5. **Zombies** are enemies like any other unit of another team.

Determinism. Every decision reads synced sim state only — integer centre
tiles (squared-distance compares in int), Q16.16-quantized positions, the
per-tick vision cache — and iterates ``sim.units`` in id order; ties go to
the lower unit id, so the bots draw nothing from ``sim.rng`` at all (the
bullets they fire do). Directions enter the sim through the one float ->
Q16.16 seam, :func:`control_source.quantize_stick_direction`. Calling
``handle_frame`` twice in one tick is a no-op (the guard on ``sim.tick``), so
a per-frame caller cannot double-decide.

Settings: ``config.toml`` ``[bots]``, read through :func:`bot_settings`, which
refuses a missing or mistyped key. Pyray-free (headless matches).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG
from control_source import ControlSource, quantize_stick_direction
from pathfinding import astar
from simulation.factions import ZOMBIE_TEAM
from simulation.intents import FP_ONE
from simulation.orders import ORDER_MOVE_ATTACK
from simulation.stats import effective_vitality
from simulation.unit_fixed import quantize_scalar
from simulation.weapons import get_tables as weapon_tables
from simulation import vision

#: The speed mode the bots walk at (the gamepad's walk).
WALK_MODE = ORDER_MOVE_ATTACK
#: A waypoint is reached within a quarter tile on both axes (Q16.16).
ARRIVE_TOL_Q = FP_ONE // 4
#: Bots steer for the MIDDLE of a waypoint tile (``tile + 1/2``): the sim
#: anchors a footprint at ``int(u.x), int(u.y)``, so a unit parked on a tile
#: edge (``y = 24.99``) sits on the wrong row and, steering for the edge,
#: creeps toward it forever (seen on ship_demo's doors). From the middle
#: every move has a whole component and the sim's wall slide carries it.
MID_Q = FP_ONE // 2

# key -> type of the [bots] table; every key is required.
_BOT_KEYS = {
    "replan_ticks": int,
    "group_radius": int,
    "danger_radius": int,
    "grenade_cooldown_s": float,
    "fallback_hp_frac": float,
    "fallback_r": int,
    "friendly_line_tiles": int,
}


@dataclass(frozen=True)
class BotSettings:
    replan_ticks: int
    group_radius: int
    danger_radius: int
    grenade_cooldown_s: float
    fallback_hp_frac: float
    fallback_r: int
    friendly_line_tiles: int


def bot_settings(cfg) -> BotSettings:
    """The ``[bots]`` table of a loaded config (``CFG``). Raises ValueError
    naming the key when the section or a key is missing, mistyped or out of
    range — a bot threshold that silently fell back to a default would hide
    a typo in config.toml."""
    section = getattr(cfg, "bots", None)
    if section is None:
        raise ValueError("config.toml has no [bots] section")
    values = {}
    for key, kind in _BOT_KEYS.items():
        if not hasattr(section, key):
            raise ValueError(f"[bots] is missing {key!r}")
        v = getattr(section, key)
        if isinstance(v, bool) or not isinstance(
                v, (int,) if kind is int else (int, float)):
            raise ValueError(f"[bots] {key} must be a {kind.__name__}, "
                             f"got {v!r}")
        values[key] = kind(v)
    for key in ("replan_ticks", "group_radius", "danger_radius", "fallback_r",
                "friendly_line_tiles"):
        if values[key] < 1:
            raise ValueError(f"[bots] {key} must be >= 1, got {values[key]}")
    if not values["grenade_cooldown_s"] > 0.0:
        raise ValueError("[bots] grenade_cooldown_s must be > 0, got "
                         f"{values['grenade_cooldown_s']!r}")
    if not 0.0 < values["fallback_hp_frac"] <= 1.0:
        raise ValueError("[bots] fallback_hp_frac must be in (0, 1], got "
                         f"{values['fallback_hp_frac']!r}")
    return BotSettings(**values)


def _centre(u) -> tuple:
    return (u.center_tile_x(), u.center_tile_y())


def _d2(a, b) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _direction_q(dx, dy):
    """A Q16.16 unit vector along ``(dx, dy)`` (any scale), through the one
    control-seam quantizer; ``None`` for a zero vector."""
    qx, qy, mag = quantize_stick_direction(float(dx), float(dy), 0.0)
    return None if mag == 0 else (qx, qy)


class _BotState:
    """One unit's controller-side memory (never on the Unit: not synced)."""
    __slots__ = ("path", "idx", "goal", "replan_at", "last_throw_tick")

    def __init__(self):
        self.path = None          # list of top-left tiles, or None
        self.idx = 0
        self.goal = None          # the top-left tile the path leads to
        self.replan_at = 0
        self.last_throw_tick = None


class BotTeam(ControlSource):
    """Drive every living unit of ``team`` by the five rules above."""

    def __init__(self, team: int, settings: BotSettings | None = None):
        self.team = int(team)
        self.settings = bot_settings(CFG) if settings is None else settings
        self.cooldown_ticks = max(1, int(round(
            self.settings.grenade_cooldown_s * CFG.clock.ticks_per_second)))
        grenade = weapon_tables().weapons.by_name["hand_grenade"]
        self.fuse_seconds = float(grenade.fuse_default_seconds)
        self.throw_range = int(grenade.max_throw_range)
        #: The team's shared last known enemy position (a top-left tile, so
        #: it is a footprint a unit stood on), or None.
        self.last_known = None
        self._state: dict = {}
        self._last_tick = None
        self._passable = None     # (tick, footprint, bool grid)

    # ------------------------------------------------------------------
    def handle_frame(self, sim, renderer=None) -> None:
        tick = int(sim.tick)
        if tick == self._last_tick:
            return
        self._last_tick = tick
        mine = [u for u in sim.units
                if u.team == self.team and u.alive and not u.is_zombie]
        if not mine:
            return
        vis = vision.state_for(sim)
        enemies = [u for u in sim.units if u.alive and u.team != self.team]
        self._update_last_known(mine, enemies, vis)
        for u in mine:
            self._drive(sim, u, mine, enemies, vis)

    def _drive(self, sim, u, mine, enemies, vis) -> None:
        st = self._state.setdefault(u.id, _BotState())
        seen = [e for e in enemies if u.id in vis.observers_of(e.id)]
        target = self._pick_target(u, mine, seen)
        self._grenade_rule(sim, u, mine, seen, st)
        move_dir = None
        if self._falling_back(u, mine):
            move_dir = self._walk_to(sim, u, st, self._home_goal(sim, u))
        elif target is not None:
            sim.clear_move_dir(u.id)          # stand and shoot
            st.path = None
        else:
            move_dir = self._advance(sim, u, st)
        if target is not None:
            (tx, ty), (ux, uy) = _centre(target), _centre(u)
            aim = _direction_q(tx - ux, ty - uy)
            if aim is not None:
                sim.set_aim(u.id, *aim)
            sim.set_trigger(u.id, True)
        else:
            if move_dir is not None:
                sim.set_aim(u.id, *move_dir)  # look where it walks
            sim.set_trigger(u.id, False)

    # ------------------------------------------------------------------
    # Rule 1 — shoot on sight
    # ------------------------------------------------------------------
    def _pick_target(self, u, mine, seen):
        """The nearest seen enemy in weapon range with no teammate on the
        first ``friendly_line_tiles`` of the line of fire (ties: lower id)."""
        weapon_id = getattr(u, "weapon_id", "")
        if not weapon_id or not seen:
            return None
        rng_tiles = int(weapon_tables().weapons.by_name[weapon_id].range_tiles)
        uc = _centre(u)
        best = None
        for e in seen:                      # id order
            d2 = _d2(uc, _centre(e))
            if d2 > rng_tiles * rng_tiles:
                continue
            if best is not None and d2 >= best[0]:
                continue
            if self._teammate_on_line(u, e, mine):
                continue
            best = (d2, e)
        return None if best is None else best[1]

    def _teammate_on_line(self, u, target, mine) -> bool:
        """Is a teammate's footprint on the segment shooter -> target within
        its first ``friendly_line_tiles``? Integer geometry: the mate's
        centre projects onto the line ahead of the shooter (0 < along <= L)
        and lies within ``footprint // 2 + 1`` tiles of it."""
        sx, sy = _centre(u)
        tx, ty = _centre(target)
        dx, dy = tx - sx, ty - sy
        dd = dx * dx + dy * dy
        if dd == 0:
            return False
        L = self.settings.friendly_line_tiles
        for m in mine:
            if m is u:
                continue
            mx, my = _centre(m)
            vx, vy = mx - sx, my - sy
            along = vx * dx + vy * dy               # = |v||d| cos
            if along <= 0:
                continue
            # along/|d| <= L  and  along/|d| <= |d| (not past the target)
            if along * along > L * L * dd or along > dd:
                continue
            cross = vx * dy - vy * dx               # = |v||d| sin
            c = m.footprint // 2 + 1
            if cross * cross <= c * c * dd:
                return True
        return False

    # ------------------------------------------------------------------
    # Rule 2 — grenade a group
    # ------------------------------------------------------------------
    def _grenade_rule(self, sim, u, mine, seen, st) -> bool:
        if getattr(u, "has_grenade", 0) <= 0 or len(seen) < 2:
            return False
        tick = int(sim.tick)
        if (st.last_throw_tick is not None
                and tick - st.last_throw_tick < self.cooldown_ticks):
            return False
        s = self.settings
        uc = _centre(u)
        gr2 = s.group_radius * s.group_radius
        dr2 = s.danger_radius * s.danger_radius
        best = None
        for e in seen:                              # id order
            ec = _centre(e)
            group = [f for f in seen if _d2(ec, _centre(f)) <= gr2]
            if len(group) < 2:
                continue
            cx = sum(f.center_tile_x() for f in group) // len(group)
            cy = sum(f.center_tile_y() for f in group) // len(group)
            c = (cx, cy)
            d2u = _d2(uc, c)
            if d2u <= dr2 or d2u > self.throw_range * self.throw_range:
                continue
            if any(_d2(_centre(m), c) <= dr2 for m in mine):
                continue
            if best is None or len(group) > best[0]:
                best = (len(group), c)
        if best is None:
            return False
        c = best[1]
        d = _direction_q(c[0] - uc[0], c[1] - uc[1])
        if d is None:
            return False
        reach_q = quantize_scalar(float(_d2(uc, c)) ** 0.5)
        sim.throw_grenade_intent(u.id, d[0], d[1], self.fuse_seconds,
                                 reach_q=reach_q)
        st.last_throw_tick = tick
        return True

    # ------------------------------------------------------------------
    # Rule 3 — fall back
    # ------------------------------------------------------------------
    def _falling_back(self, u, mine) -> bool:
        s = self.settings
        if not u.current_hp < s.fallback_hp_frac * effective_vitality(u):
            return False
        r2 = s.fallback_r * s.fallback_r
        uc = _centre(u)
        return not any(m is not u and _d2(uc, _centre(m)) <= r2
                       for m in mine)

    def _home_goal(self, sim, u):
        return self._room_goal(sim, self.team, u.footprint)

    # ------------------------------------------------------------------
    # Rule 4 — advance
    # ------------------------------------------------------------------
    def _update_last_known(self, mine, enemies, vis) -> None:
        """Any teammate's sighting updates the team's last known enemy
        position: the seen enemy nearest to any of the team's units."""
        seen_ids = set(vis.visible_by_team.get(self.team, ()))
        best = None
        for e in enemies:                           # id order
            if e.id not in seen_ids:
                continue
            ec = _centre(e)
            d2 = min(_d2(ec, _centre(m)) for m in mine)
            if best is None or d2 < best[0]:
                best = (d2, e)
        if best is not None:
            e = best[1]
            self.last_known = (e.tile_x, e.tile_y)

    def _advance(self, sim, u, st):
        fp = u.footprint
        goal = self.last_known
        if goal is not None:
            moving = self._walk_to(sim, u, st, goal)
            if moving is not None:
                return moving
            if self._at(u, goal) or st.path == []:
                # Arrived (nobody here any more) or no route there: forget it
                # and head for the enemy spawn room.
                self.last_known = None
            else:
                return None
        return self._walk_to(sim, u, st, self._enemy_room_goal(sim, fp))

    # ------------------------------------------------------------------
    # Movement — A* + MOVE_DIR
    # ------------------------------------------------------------------
    @staticmethod
    def _pos_q(u):
        return quantize_scalar(u.x), quantize_scalar(u.y)

    def _at(self, u, tile) -> bool:
        px, py = self._pos_q(u)
        return (abs(px - ((tile[0] << 16) + MID_Q)) <= ARRIVE_TOL_Q
                and abs(py - ((tile[1] << 16) + MID_Q)) <= ARRIVE_TOL_Q)

    def _walk_to(self, sim, u, st, goal):
        """Walk ``u`` toward the top-left tile ``goal``; returns the Q16.16
        direction it was given, or None when it stands (arrived, no goal,
        no route)."""
        if goal is None:
            sim.clear_move_dir(u.id)
            return None
        tick = int(sim.tick)
        # Replan when there is no path, it is used up (arrival), or the
        # cadence comes round -- a changed goal waits for the cadence (never
        # an A* every tick while a teammate's sighting moves the goal).
        if (st.path is None or tick >= st.replan_at
                or st.idx >= len(st.path)):
            if self._at(u, goal):
                sim.clear_move_dir(u.id)
                st.path, st.goal = None, goal
                return None
            st.path = self._plan(sim, u, goal)
            st.idx = 1
            st.goal = goal
            st.replan_at = tick + self.settings.replan_ticks
        path = st.path
        while st.idx < len(path) and self._at(u, path[st.idx]):
            st.idx += 1
        if st.idx >= len(path):
            # Arrived at the last waypoint (or no route): stand; the next
            # tick replans (on arrival) or finds itself at the goal.
            sim.clear_move_dir(u.id)
            return None
        wx, wy = path[st.idx]
        px, py = self._pos_q(u)
        d = _direction_q((wx << 16) + MID_Q - px, (wy << 16) + MID_Q - py)
        if d is None:
            sim.clear_move_dir(u.id)
            return None
        sim.set_move_dir(u.id, d[0], d[1], WALK_MODE)
        return d

    def _plan(self, sim, u, goal) -> list:
        grid = self._passable_grid(sim, u.footprint)
        h, w = sim.gmap.material.shape
        start = (u.tile_x, u.tile_y)

        def blocked(x, y, _g=grid):
            return not _g[y, x]
        return astar(start[0], start[1], goal[0], goal[1], blocked, w, h)

    def _passable_grid(self, sim, fp):
        """``grid[y, x]`` == ``gmap.is_passable_block(y, x, fp)`` for every
        in-bounds top-left, built once per tick (vectorised, same law: every
        tile of the block has mobility > 0)."""
        c = self._passable
        if c is not None and c[0] == sim.tick and c[1] == fp:
            return c[2]
        gmap = sim.gmap
        ok = gmap.materials.mobility[gmap.material] > 0
        win = np.lib.stride_tricks.sliding_window_view(ok, (fp, fp))
        grid = win.all(axis=(2, 3))
        self._passable = (sim.tick, fp, grid)
        return grid

    # ------------------------------------------------------------------
    # Goals — spawn rooms
    # ------------------------------------------------------------------
    def _rooms(self, sim) -> dict:
        return {r.team: r for r in getattr(sim.level, "spawn_rooms", ())}

    def _enemy_room_goal(self, sim, fp):
        rooms = self._rooms(sim)
        others = sorted(t for t in rooms
                        if t != self.team and t != ZOMBIE_TEAM)
        if not others:
            return None
        return self._room_goal(sim, others[0], fp)

    def _room_goal(self, sim, team, fp):
        """The top-left tile of a footprint centred in ``team``'s spawn room,
        or the room's slot-0 formation cell if the centre is blocked."""
        room = self._rooms(sim).get(team)
        if room is None:
            return None
        gx = room.x + room.w // 2 - fp // 2
        gy = room.y + room.h // 2 - fp // 2
        if sim.gmap.is_passable_block(gy, gx, fp):
            return (gx, gy)
        from simulation.match import placement_cells
        return tuple(placement_cells(room, 1)[0])


__all__ = ["BotTeam", "BotSettings", "bot_settings", "WALK_MODE"]
