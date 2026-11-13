"""simulation/team_picture.py -- what a team knows of the enemy (#195 1d).

THE one picture a marine team holds of the units it has seen
(``docs/rl_agent_design_2026-11-11.md`` 3.6, 4.5, 9; the patch note
``docs/rl_agent_rung1_note_1d_2026-11-12.md``): per team, one
:class:`Record` per enemy -- the place, class, side, motion and tick of
the team's last sighting of it, never a live unit. The layered bots read it
(``control_bots.TeamBrain``); the agents' observation (1g) and the aim mask
(1h) will read the same records -- never a second picture.

Switched on only for a match: ``Simulation.team_picture`` is ``None`` until
``Simulation.start_team_picture(teams)`` (``simulation.match.spawn_match``
calls it for every marine team of the spec; a hand-built sim calls it
itself), which takes the first update at once. The conductor then updates
it ONCE per tick, at the very end of ``_step_post_physics`` (slot 11, after
the tick advance and ``on_tick_end``, so a conversion has happened and the
records carry the tick the next controllers run at). Never on demand:
:meth:`TeamPicture.update` raises on a skipped tick and
:meth:`TeamPicture.records` raises when the picture is not at ``sim.tick``.
Host-side memory, not synced, not hashed, in no digest or recording: it is
rebuilt by re-simulation only.

A record is never refreshed from the live unit (step (ii) of 1d): its
class, side and place stay as the team saw them. It is dropped when the
team SAW the death -- the victim of one of the step's ``UnitKilledEvent`` events
whose record was written at the observation just before that step (a kill
of a unit the team did not see at that moment, a frag's included, is not a
seen death) -- and when it is older than ``[team_picture] memory_s``; so a
record outlives an unseen death (or an unseen conversion) for its whole
memory time, and a converted unit seen again gets a fresh zombie record.

The fire lanes are read from STATE (design 4.5): :func:`fire_lanes` (every
living marine of the team with its trigger held, from its centre along its
facing to its weapon's range, cut at the first solid tile) through the one
drawing routine :func:`draw_lanes`.

THE TRACKERS (#195 1g, ``docs/rl_agent_obs_spec_v1_2026-11-12.md`` 1.6;
design 9: memory updated every tick, never on demand, not unit state, not
synced, rebuilt by re-simulation). Members of the picture on the SAME
slot-11 driver, so no reader ever computes memory on demand:

- ``motion[uid]``: ``(pos_q_prev, pos_q_now)`` of every living unit, Q16
  tiles; the velocity is their difference, ``(0, 0)`` on the first update;
- ``last_hit[uid]``: ``(tick, dir_x_q, dir_y_q, cause)`` of the last
  ``UnitHitEvent`` with ``damage > 0`` on the unit, events in emission order;
- the damage window, per team: the hits on a unit the team DEALT (the
  attacker a living non-zombie unit of the team) or SAW (the victim's record
  stood at ``tick - 1`` BEFORE this update touched the records -- the seen
  death's own moment), kept ``window_ticks`` (the design's "last second",
  ``ticks_per_second`` quantized once), read through :meth:`damage_last_s`;
- ``bars[uid]``: the unit's ``current_hp`` at the first update at which it
  EXISTS (its spawn, so an enemy first seen wounded reads its true ratio) and
  again at the update that first sees ``is_zombie`` (the conversion ran in
  ``on_tick_end`` before slot 11);
- ``death_tick[uid]``: the first update that saw the unit dead or turned
  (the full view's dead row, 5.6);
- the object tracker, per team: the live grenades (``sim.projectiles``) and
  charges (``sim.planted_charges``) the team has SEEN -- a tile some living
  non-zombie unit of the team has a line of SIGHT to (``vision.rays_clear``,
  ``physical=False``, one call per team per tick over every (mate centre,
  object tile) pair) -- each given the team's OWN object id,
  ``OBJECT_UID_BASE + k`` for the team's k-th first-seen object of the
  episode, kept while the object lives (re-seeing it after a gap gives the
  same id: the team's memory, as a record); ``seen_now[team]`` is the list the
  gate passes this tick, ``sim.projectiles`` then ``sim.planted_charges``
  order. An object never seen has no id.

The observation reads them ONLY through ``obs_spec_v1.TeamKnowledge`` (the
0.5 view), in scope: a mate's motion, a mate's or a SEEN enemy's bar and
damage, the observing soldier's own last hit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from simulation import vision
from simulation.events import UnitHitEvent, UnitKilledEvent
from simulation.unit_fixed import cos_q16, shr_trunc, sin_q16

#: A team's object ids start here (``obs spec`` 2.1): ``OBJECT_UID_BASE + k``
#: for the team's k-th first-seen grenade or charge, never a unit id.
OBJECT_UID_BASE = 1 << 20


# ---------------------------------------------------------------------------
# Settings -- the [team_picture] table (the refusing door)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PictureSettings:
    """``[team_picture]`` as the picture reads it."""
    memory_s: float


def picture_settings(cfg) -> PictureSettings:
    """The ``[team_picture]`` table of a loaded config. Raises ValueError
    naming the key when the section or ``memory_s`` is missing, mistyped or
    negative."""
    section = getattr(cfg, "team_picture", None)
    if section is None:
        raise ValueError("config.toml has no [team_picture] section")
    known = {"memory_s"}
    for key in vars(section):
        if key not in known:
            raise ValueError(f"[team_picture] has unknown key {key!r} "
                             f"(known: {sorted(known)})")
    if not hasattr(section, "memory_s"):
        raise ValueError("[team_picture] is missing 'memory_s'")
    v = section.memory_s
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"[team_picture] memory_s must be a number, got {v!r}")
    if v < 0:
        raise ValueError(f"[team_picture] memory_s must be >= 0, got {v!r}")
    return PictureSettings(memory_s=float(v))


def memory_ticks(settings: PictureSettings, tps: float) -> int:
    """``memory_s`` in whole ticks, at least one (quantized once)."""
    return max(1, int(round(settings.memory_s * tps)))


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Record:
    """A team's last sighting of one unit: plain values, no live object.

    ``tile`` is the top-left ``(x, y)``, ``centre`` the centre tile, ``pos_q``
    the position in Q16.16 tiles, ``vel_q`` the displacement in Q16 tiles per
    tick between this sighting and the one the tick before (``(0, 0)`` on a
    first sighting or after a gap), ``tick`` the sighting's tick,
    ``facing_q`` the unit's facing at the sighting (Q16.16 radians; #195 1g:
    the team saw which way the enemy faced)."""
    uid: int
    team: int
    zombie: bool
    loadout: str
    tile: tuple
    centre: tuple
    pos_q: tuple
    vel_q: tuple
    tick: int
    facing_q: int = 0


def _sighting(u, tick: int, prev) -> Record:
    pos_q = (int(u.x_q), int(u.y_q))
    if prev is not None and prev.tick == tick - 1:
        vel_q = (pos_q[0] - prev.pos_q[0], pos_q[1] - prev.pos_q[1])
    else:
        vel_q = (0, 0)
    return Record(uid=int(u.id), team=int(u.team), zombie=bool(u.is_zombie),
                  loadout=str(u.loadout_name or ""),
                  tile=(int(u.tile_x), int(u.tile_y)),
                  centre=(int(u.center_tile_x()), int(u.center_tile_y())),
                  pos_q=pos_q, vel_q=vel_q, tick=int(tick),
                  facing_q=int(u.facing_q))


def object_tile(obj) -> tuple:
    """The ``(y, x)`` tile of a live grenade or charge, as the entry shows
    it: a projectile at ``(int(fy), int(fx))``, a charge at ``(y, x)``."""
    if hasattr(obj, "det_tick"):            # a PlantedCharge
        return (int(obj.y), int(obj.x))
    return (int(obj.fy), int(obj.fx))


def live_objects(sim) -> list:
    """The grenades and charges that exist now, in the one order the trackers
    and the list use: every ``Projectile`` of ``sim.projectiles`` not
    ``detonated``, then every ``PlantedCharge`` of ``sim.planted_charges``
    that is ``live``."""
    out = [p for p in sim.projectiles if not p.detonated]
    out.extend(c for c in sim.planted_charges if c.live)
    return out


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------
class TeamPicture:
    """Every marine team's records, updated once per tick by the conductor."""

    def __init__(self, sim, teams, memory_ticks: int,
                 window_ticks: int | None = None):
        self._sim = sim             # the sim it belongs to (the tick check)
        self.teams = tuple(sorted(int(t) for t in teams))
        self.memory_ticks = int(memory_ticks)
        #: the damage window's length (the design's "last second")
        self.window_ticks = (max(1, int(round(float(sim._tps))))
                             if window_ticks is None else int(window_ticks))
        self._records = {t: {} for t in self.teams}
        self._views = {t: MappingProxyType(self._records[t])
                       for t in self.teams}
        self.tick = None            # the tick the picture is at
        # -- the trackers (#195 1g; the module doc) --
        self.motion: dict = {}      # uid -> ((x_q, y_q) prev, (x_q, y_q) now)
        self.last_hit: dict = {}    # uid -> (tick, dir_x_q, dir_y_q, cause)
        self._damage = {t: {} for t in self.teams}   # team -> uid -> deque
        self.bars: dict = {}        # uid -> hp at spawn / at turning
        self._turned: set = set()   # uids whose zombie bar was taken
        self.death_tick: dict = {}  # uid -> the first update seeing it gone
        self._objects = {t: {} for t in self.teams}  # team -> obj -> team uid
        self._obj_next = {t: 0 for t in self.teams}
        self.seen_now = {t: [] for t in self.teams}  # team -> [(obj, uid)]

    # -- the driver -----------------------------------------------------
    def update(self, sim) -> None:
        """Bring the picture to ``sim.tick``: a no-op when already there,
        and an error unless ``sim.tick`` is the next tick (a missed or a
        doubled update is the bug, never patched over)."""
        if sim is not self._sim:
            raise RuntimeError("a team picture is updated by its own sim")
        tick = int(sim.tick)
        if self.tick == tick:
            return
        if self.tick is not None and tick != self.tick + 1:
            raise RuntimeError(
                f"team picture at tick {self.tick} asked to update at tick "
                f"{tick}: the conductor must update it once every tick")
        vis = vision.state_for(sim)
        first = self.tick is None
        # this step's deaths (the first update, at the start, has no step)
        events = () if first else tuple(sim.tick_events)
        killed = sorted({int(ev.unit_id) for ev in events
                         if isinstance(ev, UnitKilledEvent)})
        hits = [ev for ev in events
                if isinstance(ev, UnitHitEvent) and ev.damage > 0]
        by_id = {u.id: u for u in sim.units}
        self._track_units(sim, by_id, tick, hits)
        for team in self.teams:
            recs = self._records[team]
            # the damage window (1.6): against the records as they stand
            # BEFORE this update touches them -- the seen death's moment
            self._track_damage(team, recs, by_id, tick, hits)
            # (1) a SEEN death: the victim was in the team's sight at the
            # observation just before the step it died in
            for uid in killed:
                rec = recs.get(uid)
                if rec is not None and rec.tick == tick - 1:
                    del recs[uid]
            # (2) the memory: kept while tick - rec.tick <= memory_ticks
            for uid in sorted(recs):
                if tick - recs[uid].tick > self.memory_ticks:
                    del recs[uid]
            # (3) this tick's sightings
            for uid in vis.visible_by_team.get(team, ()):
                recs[uid] = _sighting(by_id[uid], tick, recs.get(uid))
        self._track_objects(sim, tick)
        self.tick = tick

    def _track_units(self, sim, by_id, tick, hits) -> None:
        """The per-unit trackers: motion, last hit, bars, death tick."""
        motion = self.motion
        for u in sim.units:
            uid = int(u.id)
            if uid not in self.bars:            # exists: its spawn
                self.bars[uid] = float(u.current_hp)
            if u.is_zombie and uid not in self._turned:
                self._turned.add(uid)
                self.bars[uid] = float(u.current_hp)  # at turning
                if uid not in self.death_tick and uid in motion:
                    self.death_tick[uid] = tick       # turned (a marine)
            if not u.alive:
                if uid not in self.death_tick:
                    self.death_tick[uid] = tick
                continue
            now = (int(u.x_q), int(u.y_q))
            prev = motion.get(uid)
            motion[uid] = (now if prev is None else prev[1], now)
        for ev in hits:                        # emission order: the last wins
            self.last_hit[int(ev.unit_id)] = (tick, int(ev.dir_x_q),
                                              int(ev.dir_y_q), str(ev.source))

    def _track_damage(self, team, recs, by_id, tick, hits) -> None:
        """``team``'s damage window: the hits it dealt or saw this step,
        then the entries older than ``window_ticks`` dropped."""
        win = self._damage[team]
        seen_prev = {uid for uid, rec in recs.items() if rec.tick == tick - 1}
        for ev in hits:
            v = int(ev.unit_id)
            a = by_id.get(int(ev.attacker_id))
            dealt = (a is not None and int(a.team) == team
                     and not a.is_zombie)
            if dealt or v in seen_prev:
                win.setdefault(v, deque()).append((tick, float(ev.damage)))
        oldest = tick - self.window_ticks
        for uid in list(win):
            q = win[uid]
            while q and q[0][0] <= oldest:
                q.popleft()
            if not q:
                del win[uid]

    def _track_objects(self, sim, tick) -> None:
        """The per-team object tracker: the sight gate over every (mate
        centre, object tile) pair in ONE ``rays_clear`` per team; a first
        sighting draws the team's next id; a dead object is forgotten."""
        objs = live_objects(sim)
        alive = set(map(id, objs))
        for team in self.teams:
            ids = self._objects[team]
            for obj in [o for o in ids if id(o) not in alive]:
                del ids[obj]
            seen = []
            mates = [u for u in sim.units
                     if u.alive and not u.is_zombie and int(u.team) == team]
            if objs and mates:
                tiles = [object_tile(o) for o in objs]
                rows = [(m.center_tile_y(), m.center_tile_x(), ty, tx)
                        for m in mates for (ty, tx) in tiles]
                clear = vision.rays_clear(sim, rows, physical=False)
                clear = clear.reshape(len(mates), len(objs)).any(axis=0)
                for obj, ok in zip(objs, clear.tolist()):
                    if not ok:
                        continue
                    uid = ids.get(obj)
                    if uid is None:
                        uid = OBJECT_UID_BASE + self._obj_next[team]
                        self._obj_next[team] += 1
                        ids[obj] = uid
                    seen.append((obj, uid))
            self.seen_now[team] = seen

    # -- the readers ----------------------------------------------------
    def records(self, team: int):
        """``team``'s records, a read-only mapping uid -> :class:`Record`;
        raises unless the picture is at its sim's tick (it is never brought
        up to date on demand)."""
        now = int(self._sim.tick)
        if self.tick != now:
            raise RuntimeError(
                f"team picture is at tick {self.tick}, the sim at {now}: it "
                f"is updated by the conductor only, once per tick")
        return self._views[int(team)]

    def known(self, team: int, zombies=None) -> list:
        """``team``'s records in uid order (``zombies``: only / none)."""
        recs = self.records(team)
        return [recs[uid] for uid in sorted(recs)
                if zombies is None or recs[uid].zombie == zombies]

    def velocity_q(self, uid: int) -> tuple:
        """The motion tracker's one-tick displacement of ``uid`` in Q16
        tiles, ``(0, 0)`` before its second update or for an unknown uid."""
        m = self.motion.get(int(uid))
        if m is None:
            return (0, 0)
        return (m[1][0] - m[0][0], m[1][1] - m[0][1])

    def damage_last_s(self, team: int, uid: int) -> float:
        """The damage window's sum for ``uid`` as ``team`` dealt or saw it
        over the last ``window_ticks`` ticks (0.0 when none)."""
        q = self._damage[int(team)].get(int(uid))
        return float(sum(d for _, d in q)) if q else 0.0

    def object_ids(self, team: int):
        """``team``'s object tracker as a read-only mapping obj -> the team's
        object id, for every live object the team has seen."""
        return MappingProxyType(self._objects[int(team)])

    # -- tests and tools: put a sighting in by hand ---------------------
    def _write(self, sim, team: int, u) -> Record:
        """Write ``team``'s sighting of ``u`` at ``sim.tick`` (the update's
        own writer; for tests that place a unit in the picture)."""
        recs = self._records[int(team)]
        rec = _sighting(u, int(sim.tick), recs.get(u.id))
        recs[u.id] = rec
        return rec


# ---------------------------------------------------------------------------
# The fire lanes' drawing routine
# ---------------------------------------------------------------------------
def draw_lanes(h: int, w: int, segments, cell: int, halfwidth: int):
    """THE fire-lane rasteriser (design 4.5): a coarse grid ``(h // cell + 1,
    w // cell + 1)`` of who fires along each cell -- -1 nobody, an owner id
    its lane alone, -2 several owners. ``segments`` are ``(owner_id, (sx,
    sy), (ex, ey))`` in tiles; each is rasterised in integers from start to
    end and widened by ``halfwidth`` cells. The result does not depend on
    the segments' order."""
    k = int(cell)
    grid = np.full((h // k + 1, w // k + 1), -1, dtype=np.int64)
    gh, gw = grid.shape
    r = int(halfwidth)
    for uid, (sx, sy), (tx, ty) in segments:
        n = max(abs(tx - sx), abs(ty - sy)) // max(1, k // 2) + 1
        for i in range(1, n + 1):
            gx = (sx + (tx - sx) * i // n) // k
            gy = (sy + (ty - sy) * i // n) // k
            ys = slice(max(0, gy - r), min(gh, gy + r + 1))
            xs = slice(max(0, gx - r), min(gw, gx + r + 1))
            c = grid[ys, xs]
            c[(c != -1) & (c != uid)] = -2
            c[c == -1] = uid
    return grid


def fire_lanes(sim, team: int, cell: int, halfwidth: int):
    """``team``'s fire lanes from STATE (design 4.5; read-only, no memory):
    for each living non-zombie unit of ``team`` (id order) whose
    ``live_trigger`` is held, whose hands are free to fire at the coming
    tick (``verb_gates.can_fire`` at the tick's start: a unit winding up a
    throw draws no lane, 183-no-fire-windup) and whose weapon row has
    ``range_tiles > 0``,
    a segment from its centre tile S along its facing (``facing_q``, the
    grid y-down) to E = S + ``range_tiles`` -- cut at the last tile before
    the first solid (or off-map) tile of a per-tile walk S -> E -- drawn
    through :func:`draw_lanes`. The clip is a drawing step on the coarse
    grid, not a line-of-fire test. The body is :func:`fire_lanes_of` over
    the sim's units (the observation hands it the team's mates, #195 1g)."""
    return fire_lanes_of(sim.gmap, sim.weapons_tables, sim.units, team,
                         cell, halfwidth, tick=sim.tick)


def fire_lanes_of(gmap, weapons_tables, members, team: int, cell: int,
                  halfwidth: int, *, tick: int):
    """:func:`fire_lanes` over an explicit unit list ``members`` (every
    unit of the team that may own a lane must be in it; the observation
    hands it the team's mates, or one mate for its own lane). ``tick`` is
    the coming tick (the sim's ``tick`` at the read), for the trigger's
    gate."""
    from simulation import verb_gates
    solid = gmap.solid
    h, w = solid.shape
    rows = weapons_tables.weapons.by_name
    segments = []
    for u in sorted(members, key=lambda u: u.id):
        if (u.team != team or not u.alive or u.is_zombie
                or not getattr(u, "live_trigger", False)
                or not verb_gates.can_fire(u, tick, at_tick_start=True)):
            continue
        row = rows.get(getattr(u, "weapon_id", "") or "")
        rng = int(getattr(row, "range_tiles", 0) or 0) if row else 0
        if rng <= 0:
            continue
        sx, sy = int(u.center_tile_x()), int(u.center_tile_y())
        f = int(u.facing_q)
        dx = shr_trunc(rng * cos_q16(f), 16)
        dy = -shr_trunc(rng * sin_q16(f), 16)
        n = max(abs(dx), abs(dy))
        ex, ey = sx, sy
        for i in range(1, n + 1):
            px, py = sx + dx * i // n, sy + dy * i // n
            if not (0 <= px < w and 0 <= py < h) or solid[py, px]:
                break
            ex, ey = px, py
        segments.append((int(u.id), (sx, sy), (ex, ey)))
    return draw_lanes(h, w, segments, cell, halfwidth)
