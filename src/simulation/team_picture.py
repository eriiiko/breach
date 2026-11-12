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
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from simulation import vision
from simulation.events import UnitKilledEvent
from simulation.unit_fixed import cos_q16, quantize_scalar, shr_trunc, sin_q16


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
    first sighting or after a gap), ``tick`` the sighting's tick."""
    uid: int
    team: int
    zombie: bool
    loadout: str
    tile: tuple
    centre: tuple
    pos_q: tuple
    vel_q: tuple
    tick: int


def _sighting(u, tick: int, prev) -> Record:
    pos_q = (quantize_scalar(u.x), quantize_scalar(u.y))
    if prev is not None and prev.tick == tick - 1:
        vel_q = (pos_q[0] - prev.pos_q[0], pos_q[1] - prev.pos_q[1])
    else:
        vel_q = (0, 0)
    return Record(uid=int(u.id), team=int(u.team), zombie=bool(u.is_zombie),
                  loadout=str(u.loadout_name or ""),
                  tile=(int(u.tile_x), int(u.tile_y)),
                  centre=(int(u.center_tile_x()), int(u.center_tile_y())),
                  pos_q=pos_q, vel_q=vel_q, tick=int(tick))


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------
class TeamPicture:
    """Every marine team's records, updated once per tick by the conductor."""

    def __init__(self, sim, teams, memory_ticks: int):
        self._sim = sim             # the sim it belongs to (the tick check)
        self.teams = tuple(sorted(int(t) for t in teams))
        self.memory_ticks = int(memory_ticks)
        self._records = {t: {} for t in self.teams}
        self._views = {t: MappingProxyType(self._records[t])
                       for t in self.teams}
        self.tick = None            # the tick the picture is at

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
        # this step's deaths (the first update, at the start, has no step)
        killed = () if self.tick is None else sorted(
            {int(ev.unit_id) for ev in sim.tick_events
             if isinstance(ev, UnitKilledEvent)})
        by_id = {u.id: u for u in sim.units}
        for team in self.teams:
            recs = self._records[team]
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
        self.tick = tick

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
    ``live_trigger`` is held and whose weapon row has ``range_tiles > 0``,
    a segment from its centre tile S along its facing (``facing_q``, the
    grid y-down) to E = S + ``range_tiles`` -- cut at the last tile before
    the first solid (or off-map) tile of a per-tile walk S -> E -- drawn
    through :func:`draw_lanes`. The clip is a drawing step on the coarse
    grid, not a line-of-fire test."""
    gmap = sim.gmap
    solid = gmap.solid
    h, w = solid.shape
    rows = sim.weapons_tables.weapons.by_name
    segments = []
    for u in sorted(sim.units, key=lambda u: u.id):
        if (u.team != team or not u.alive or u.is_zombie
                or not getattr(u, "live_trigger", False)):
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
