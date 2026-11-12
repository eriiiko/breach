"""simulation/obs_spec_v1.py -- observation spec ``v1`` as code (#195 1g).

THE contract is ``docs/rl_agent_obs_spec_v1_2026-11-12.md`` (document
version 2): every feature a trained network reads is a row of the tables
below -- its position, its name, the seam it is read from, the table value
that divides it, whether the one squash is applied -- and is a RATIO of a
table value, an angle's cosine and sine, a flag, or a kind (a small integer
the network embeds). Never a game number. ``SPEC_NAME = "v1"``.

Patch (a) builds the soldier's VECTOR (section 1; its field-dependent
``air`` block, cols 58-65, and ``felt_temp``, col 96, read 0 until patch
(b)), the LIST (section 2: the five blocks, the 67-column entry; cols 28,
52-53 and 66 read 0 until 1h / (b)) and serves them through the sensor
accessor's ``observe`` door into the env's buffers (section 4). The windows
are patch (b), the full view patch (c).

THE WALL (spec 0.5, 5.7). The builders of sections 1-3 never receive the
``Simulation``: they take ``(know, u)`` -- ONE :class:`TeamKnowledge` per
team per decision, built by :meth:`TeamKnowledge.build`, the only
observation function that touches ``sim`` (with :func:`observe_rows`, the
door that hands it the sim) -- and the observing soldier. Every routine
they call receives ``know.world``, an :class:`ObsWorld` facade with an
explicit attribute list and NO fallback, as its ``sim``: a read of
``units``, ``roster``, ``get_unit``, ``projectiles``, ``planted_charges``,
``team_picture``, ``tick_events`` or a vision cache raises
``AttributeError`` at run time. The trackers answer only in scope (a mate's
motion; a mate's or a SEEN enemy's bar and damage; a mate's last hit).
Every cache is keyed by the inputs it was computed from.

Pure host Python over the sim, the team picture and the trackers; imports
no controller and nothing from ``renderer/``.
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from simulation import (combat, horde, line_of_fire, room_graph, turning,
                        unit_impulse, unit_schema, verb_gates, vision)
from simulation.damage import KINETIC
from simulation.entities.door import DOOR_OPEN
from simulation.events import CAUSE_NAMES
from simulation.movement import THROTTLE_ONE
from simulation.status import N_STATUS_KINDS, STATUS_REGISTRY, composed_flags
from simulation.team_picture import OBJECT_UID_BASE, object_tile
from simulation.turning import PI_Q16, wrap_q16
from simulation.unit_fixed import (FP_ONE, atan2_q16, cos_q16, quantize_scalar,
                                   sin_q16)
from simulation.unit_orders import (NO_LIMIT, NO_PLACE, NO_TARGET,
                                    ORDER_OPENING_BASE, ORDER_WHAT, graph_of)
from simulation.unit_schema import THROW_NO_REACH
from simulation.weapons import FRAG_ROW, PLANT_WEAPON, fuse_ticks_from_seconds

SPEC_NAME = "v1"

#: The squash's cap (spec 0.2): ``sign(x) (1 + ln|x|)`` beyond 1, at most 4.
SQUASH_CAP = 4.0

# -- the list's blocks (spec 2.1) --------------------------------------------
TYPE_MATE, TYPE_SEEN, TYPE_RECORD, TYPE_OBJECT, TYPE_DOOR = 0, 1, 2, 3, 4
#: The fixed caps after the mate block (whose cap is ``S - 1``).
CAPS_FIXED = (16, 8, 6, 4)
#: ``E = S + 33``: the entries per soldier for ``S`` roster slots.
ENTRIES_BEYOND_SLOTS = sum(CAPS_FIXED) - 1
#: A door's ``list_uid`` is its ordinal plus this (doors are public).
DOOR_UID_BASE = 1 << 22
#: The hand-grenade rows the vector's layout holds (cols 29..40, two per
#: row); the context build refuses a table of another count.
HAND_GRENADE_SLOTS = 6

# -- the denominators: the closed vocabulary a column may be divided by -----
FLAG, KIND, ANGLE, UNIT = "flag", "kind", "angle", "unit interval"
#: Every table value a column may be a ratio of (spec sections 1-2); the
#: no-raw-number gate holds every column to this set or to FLAG / KIND /
#: ANGLE / UNIT. ``(b)`` marks a column patch (b) fills (0 in (a)).
DENOMINATORS = frozenset({
    "bars", "turn_rate", "PI_Q16", "v_ref", "v_fall", "FP_ONE", "react_ticks",
    "STATUS_SCALE_TICKS", "mag_size", "magazines", "reload_ticks",
    "loadout_count", "throw_windup_ticks", "explosives", "plant_ticks",
    "c4_fuse_max_ticks", "rof_interval_ticks", "shove_cooldown_ticks",
    "L_ref", "ref_rof", "ref_spread", "air (b)", "temperature_max (b)",
    "memory_ticks", "limit_ticks", "n_t", "team_ammo_total", "enemies_n",
    "zombies", "d_c", "x_max", "HALF_TURN_TICKS", "THROTTLE_ONE",
    "fuse_max_ticks",
})


@dataclass(frozen=True)
class Feature:
    """One column: its position, name, what divides it (a
    :data:`DENOMINATORS` name, or FLAG / KIND / ANGLE / UNIT) and whether
    the squash is applied to the ratio."""
    pos: int
    name: str
    denom: str
    squash: bool = False


def _cols(start, rows):
    out = []
    for i, (name, denom, sq) in enumerate(rows):
        out.append(Feature(start + i, name, denom, sq))
    return out


# -- section 1: the soldier's vector (V = 97) --------------------------------
VECTOR_COLUMNS = tuple(
    _cols(0, [("hp", "bars", True), ("facing_cos", ANGLE, False),
              ("facing_sin", ANGLE, False), ("omega", "turn_rate", True),
              ("turn_left", "PI_Q16", False), ("vel_x", "v_ref", True),
              ("vel_y", "v_ref", True), ("pushed_x", "v_fall", True),
              ("pushed_y", "v_fall", True), ("effort", "FP_ONE", True),
              ("react", "react_ticks", False), ("prone", FLAG, False),
              ("can_move", FLAG, False), ("can_act", FLAG, False),
              ("can_aim", FLAG, False)])
    + _cols(15, [(f"status_left[{k}]", "STATUS_SCALE_TICKS", True)
                 for k in range(N_STATUS_KINDS)])
    + _cols(24, [("rounds", "mag_size", False), ("mags", "magazines", True),
                 ("ammo_untracked", FLAG, False), ("reloading", FLAG, False),
                 ("reload_left", "reload_ticks", False)])
    + _cols(29, sum(([(f"grenade[{k}]", "loadout_count", False),
                      (f"grenade_kind_ok[{k}]", FLAG, False)]
                     for k in range(HAND_GRENADE_SLOTS)), []))
    + _cols(41, [("throw_windup", FLAG, False),
                 ("throw_left", "throw_windup_ticks", False),
                 ("throw_kind", KIND, False), ("charges", "explosives", False),
                 ("has_charge_kind", FLAG, False), ("planting", FLAG, False),
                 ("plant_left", "plant_ticks", False),
                 ("plant_timer", "c4_fuse_max_ticks", False),
                 ("melee_cool", "rof_interval_ticks", False),
                 ("has_melee", FLAG, False),
                 ("shove_cool", "shove_cooldown_ticks", False),
                 ("class", KIND, False), ("weapon", KIND, False),
                 ("range", "L_ref", True), ("rof", "ref_rof", True),
                 ("spread", "ref_spread", True), ("trigger", FLAG, False)])
    + _cols(58, [(f"air[{p}]", "air (b)", True) for p in range(8)])
    + _cols(66, [("hit_recent", FLAG, False), ("hit_age", "memory_ticks", False),
                 ("hit_from_cos", ANGLE, False), ("hit_from_sin", ANGLE, False),
                 ("hit_cause", KIND, False)])
    + _cols(71, [(f"prev[{h}]", KIND, False) for h in range(6)])
    + _cols(77, [("time_left", "limit_ticks", False),
                 ("mates_alive", "n_t", False),
                 ("team_ammo", "team_ammo_total", True),
                 ("enemies_known", "enemies_n", False),
                 ("enemies_seen", "enemies_n", False),
                 ("zombies_known", "zombies", True),
                 ("zombies_near", "zombies", True), ("order_what", KIND, False),
                 ("order_has_limit", FLAG, False),
                 ("order_left", "limit_ticks", False), ("inside", FLAG, False),
                 ("place_kind", KIND, False), ("place_dx", "L_ref", True),
                 ("place_dy", "L_ref", True), ("place_dist_c", "d_c", False),
                 ("compass_dx", ANGLE, False), ("compass_dy", ANGLE, False),
                 ("route_len", "L_ref", True), ("has_target", FLAG, False),
                 ("felt_temp", "temperature_max (b)", True)])
)
V = len(VECTOR_COLUMNS)
assert V == 97 and [f.pos for f in VECTOR_COLUMNS] == list(range(V))

# -- section 2: the entry (F = 67) -------------------------------------------
ENTRY_COLUMNS = tuple(
    _cols(0, [("type", KIND, False), ("dx", "L_ref", True), ("dy", "L_ref", True),
              ("dist_c", "d_c", False), ("cos", ANGLE, False),
              ("sin", ANGLE, False), ("vel_x", "v_ref", True),
              ("vel_y", "v_ref", True), ("side", KIND, False),
              ("class", KIND, False), ("hp", "bars", True),
              ("exposure", "x_max", False), ("firing", FLAG, False),
              ("reloading", FLAG, False), ("reload_left", "reload_ticks", False),
              ("winding", FLAG, False), ("throw_left", "throw_windup_ticks", False),
              ("throw_kind", KIND, False), ("planting", FLAG, False),
              ("plant_left", "plant_ticks", False),
              ("plant_timer", "c4_fuse_max_ticks", False),
              ("stunned", FLAG, False), ("blinded", FLAG, False),
              ("down", FLAG, False), ("seen_by_me", FLAG, False),
              ("lof_solid", FLAG, False), ("lof_mate", FLAG, False),
              ("ticks_to_on", "HALF_TURN_TICKS", True),
              ("on_next_tick", FLAG, False), ("in_range", FLAG, False),
              ("in_throw_range", FLAG, False), ("facing_on_me", FLAG, False),
              ("aims_at", KIND, False), ("zombie_state", KIND, False),
              ("mates_aiming", "n_t", False), ("mates_firing", "n_t", False),
              ("dmg_last_s", "bars", True), ("priority", FLAG, False),
              ("frag_catch", "n_t", False), ("aim_cos", ANGLE, False),
              ("aim_sin", ANGLE, False), ("aim_is_unit", FLAG, False),
              ("feet_dx", ANGLE, False), ("feet_dy", ANGLE, False),
              ("throttle", "THROTTLE_ONE", False), ("land_dx", "L_ref", True),
              ("land_dy", "L_ref", True), ("order_what", KIND, False),
              ("order_place_dx", "L_ref", True), ("order_place_dy", "L_ref", True),
              ("order_same_place", FLAG, False),
              ("order_left", "limit_ticks", False),
              ("me_in_its_lane", FLAG, False), ("it_in_my_lane", FLAG, False),
              ("age", "memory_ticks", False), ("obj_kind", KIND, False),
              ("whose", KIND, False), ("time_left", "fuse_max_ticks", True),
              ("remote", FLAG, False), ("blast_r", "L_ref", True),
              ("in_blast", FLAG, False), ("on_floor", FLAG, False),
              ("open", FLAG, False), ("in_reach", FLAG, False),
              ("hull", FLAG, False), ("wired", FLAG, False),
              ("prev_aim", FLAG, False)])
)
F = len(ENTRY_COLUMNS)
assert F == 67 and [f.pos for f in ENTRY_COLUMNS] == list(range(F))

_VC = {f.name: f.pos for f in VECTOR_COLUMNS}
_EC = {f.name: f.pos for f in ENTRY_COLUMNS}


# ---------------------------------------------------------------------------
# The squash and the compressed distance (spec 0.1, 0.2)
# ---------------------------------------------------------------------------
def squash(x: float) -> float:
    """THE one curve: the identity on ``[-1, 1]``, ``sign(x) (1 + ln|x|)``
    beyond, capped at ``+-SQUASH_CAP``. Continuous and monotone."""
    x = float(x)
    ax = abs(x)
    if ax <= 1.0:
        return x
    v = min(SQUASH_CAP, 1.0 + math.log(ax))  # ingress-exempt: observation output for the network, float by design; never written to synced sim state
    return v if x > 0.0 else -v


def ratio(x, denom) -> float:
    """``x / denom``; 0 when the table value is 0 (a kind the loadout lacks,
    a match without zombies) -- the mask, never 0 / 0."""
    d = float(denom)
    return 0.0 if d == 0.0 else float(x) / d


def d_c(d: float, l_ref: int) -> float:
    """The compressed distance: ``ln(1 + d) / ln(1 + L_ref)``, squashed."""
    return squash(math.log1p(max(0.0, float(d))) / math.log1p(float(l_ref)))  # ingress-exempt: observation output for the network, float by design; never written to synced sim state


# ---------------------------------------------------------------------------
# The per-episode context: every table value read ONCE (spec 0.1, 1.7)
# ---------------------------------------------------------------------------
def _cfg_int(cfg, section: str, key: str) -> int:
    sec = getattr(cfg, section, None)
    if sec is None or not hasattr(sec, key):
        raise ValueError(f"obs spec v1: config.toml [{section}] is missing "
                         f"{key!r}")
    v = getattr(sec, key)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"obs spec v1: [{section}] {key} must be an integer "
                         f">= 0, got {v!r}")
    return int(v)


def status_scale_ticks(cfg, tables, impulse_table) -> tuple:
    """``STATUS_SCALE_TICKS[k]`` (spec 1.7): per status kind the longest
    shipped applier's duration in ticks, at least 1 -- knocked_down the
    impulse table's getup; blinded the teargas row's or the longest payload
    blinding; stunned / immobilized / paralyzed the longest weapon row's
    ``status_ticks`` of that kind; a kind with no shipped applier 1."""
    from simulation.status import BLINDED, IMMOBILIZED, KNOCKED_DOWN, \
        PARALYZED, STUNNED
    scale = [1] * N_STATUS_KINDS
    scale[KNOCKED_DOWN] = int(impulse_table.getup_ticks)
    blind = [_cfg_int(cfg, "exchange", "teargas_blind_ticks")]
    blind += [int(p.blind_ticks) for p in tables.payloads.by_name.values()]
    scale[BLINDED] = max(blind)
    for kind in (STUNNED, IMMOBILIZED, PARALYZED):
        ticks = [int(w.status_ticks) for w in tables.weapons.by_name.values()
                 if w.status_kind_id == kind]
        scale[kind] = max(ticks) if ticks else 1
    return tuple(max(1, s) for s in scale)


@dataclass(frozen=True)
class ObsContext:
    """The run's table values for one episode (``build`` reads them once:
    the reference kit, the reference length and speed, the match's rows,
    the clock, the status scales, the dials the spec names)."""
    spec: object                 # the MatchSpec (n_t per team, zombies)
    n_slots: int                 # S
    l_ref: int                   # L_ref, tiles
    v_ref_q: int                 # Q16 tiles per tick
    ref_rof_ticks: int
    ref_spread_deg: float
    ref_melee_arc_q: int
    limit_ticks: int
    memory_ticks: int
    tps: int
    hand_rows: tuple
    charge_rows: tuple
    frag_radius: int
    max_throw_range: int
    throw_windup_ticks: int
    plant_ticks: int
    c4_fuse_max_ticks: int
    hand_fuse_max_ticks: int
    status_scale: tuple
    turn_rate_q: int
    half_turn_ticks: int
    line_clear_tiles: int
    zombie_near_tiles: int
    x_max_q: int
    v_fall_q: int
    react_ticks: int
    shove_cooldown_ticks: int
    zombie_class: int            # one code past the loadout rows

    @property
    def n_entries(self) -> int:
        return self.n_slots + ENTRIES_BEYOND_SLOTS

    @property
    def caps(self) -> tuple:
        return (self.n_slots - 1,) + CAPS_FIXED

    @property
    def obj_rows(self) -> tuple:
        return self.hand_rows + self.charge_rows

    @staticmethod
    def build(sim, spec, n_slots: int) -> "ObsContext":
        from config import CFG
        from simulation.exposure import get_settings
        from simulation.loadouts import get_tables as loadout_tables
        from simulation.ruleset import MatchRules
        from simulation import team_picture as tp
        tables = sim.weapons_tables
        lt = loadout_tables()
        ref_loadout = lt.loadouts.by_name[str(CFG.marine.loadout)]
        ref = tables.weapons.by_name[ref_loadout.weapon]
        hand = tables.weapons.by_name["hand_grenade"]
        c4 = tables.weapons.by_name[PLANT_WEAPON]
        butt = (tables.weapons.by_name.get(ref_loadout.melee)
                if ref_loadout.melee else None)
        if butt is None:
            raise ValueError("obs spec v1: the [marine] loadout row has no "
                             "melee row (the reference kit's melee arc)")
        if len(tables.hand_grenade_rows) != HAND_GRENADE_SLOTS:
            raise ValueError(
                f"obs spec v1: the layout holds {HAND_GRENADE_SLOTS} "
                f"hand-grenade rows, the table has "
                f"{len(tables.hand_grenade_rows)}: a new spec version")
        tps = int(sim._tps)
        w = turning.turn_rate_q16(CFG)
        it = sim.impulse_table
        return ObsContext(
            spec=spec, n_slots=int(n_slots),
            l_ref=int(ref.range_tiles),
            v_ref_q=(FP_ONE * FP_ONE) // int(sim.speed_table.full_cost_q),
            ref_rof_ticks=int(ref.rof_interval_ticks),
            ref_spread_deg=float(ref.spread_deg),
            ref_melee_arc_q=int(butt.melee_arc_half_q),
            limit_ticks=int(MatchRules.from_spec(spec).limit_ticks(sim)),
            memory_ticks=int(tp.memory_ticks(tp.picture_settings(CFG),
                                             float(tps))),
            tps=tps,
            hand_rows=tuple(tables.hand_grenade_rows),
            charge_rows=tuple(tables.charge_rows),
            frag_radius=int(tables.payload_for_ammo(FRAG_ROW).radius),
            max_throw_range=int(hand.max_throw_range),
            throw_windup_ticks=int(hand.throw_windup_ticks),
            plant_ticks=int(c4.plant_ticks),
            c4_fuse_max_ticks=fuse_ticks_from_seconds(c4.fuse_max_seconds, tps),
            hand_fuse_max_ticks=fuse_ticks_from_seconds(hand.fuse_max_seconds,
                                                        tps),
            status_scale=status_scale_ticks(CFG, tables, it),
            turn_rate_q=int(w),
            half_turn_ticks=-(-PI_Q16 // int(w)),
            line_clear_tiles=_cfg_int(CFG, "bots", "line_clear_tiles"),
            zombie_near_tiles=_cfg_int(CFG, "bots", "zombie_near_tiles"),
            x_max_q=int(get_settings().x_max_q),
            v_fall_q=int(it.v_fall_q[0]),
            react_ticks=int(it.react_ticks),
            shove_cooldown_ticks=int(it.shove_cooldown_ticks),
            zombie_class=len(lt.loadouts.by_name),
        )


# ---------------------------------------------------------------------------
# The wall: ObsWorld and TeamKnowledge (spec 0.5)
# ---------------------------------------------------------------------------
class ObsWorld:
    """The public world a builder may read, and what the sight routines,
    the line of fire and the route take as their ``sim``. Explicit
    attributes, no fallback: a banned sim member raises ``AttributeError``."""

    __slots__ = ("gmap", "tick", "weapons_tables", "speed_table",
                 "impulse_table", "graph", "wires", "cover", "doors",
                 "_turn_rate", "_turn_accel", "_after_turn")

    def __init__(self, sim):
        self.gmap = sim.gmap
        self.tick = int(sim.tick)
        self.weapons_tables = sim.weapons_tables
        self.speed_table = sim.speed_table
        self.impulse_table = sim.impulse_table
        self.graph = graph_of(sim)
        self.wires = tuple(getattr(sim.level, "wires", ()) or ())
        self.cover = tuple(getattr(sim, "cover", ()) or ())
        self.doors = tuple(sim._doors)      # every door runtime: unfogged
        self._turn_rate = sim.unit_turn_rate_q16
        self._turn_accel = sim.unit_turn_accel_q16
        self._after_turn = sim.facing_after_turn_q16

    def unit_turn_rate_q16(self, u) -> int:
        return self._turn_rate(u)

    def unit_turn_accel_q16(self, u) -> int:
        return self._turn_accel(u)

    def facing_after_turn_q16(self, u, dx_q: int, dy_q: int) -> int:
        return self._after_turn(u, dx_q, dy_q)


class TrackerView:
    """The trackers read in scope: ``motion`` for a mate; ``bar`` and
    ``damage_last_s`` for a mate or a SEEN uid; ``last_hit`` for a mate
    (the observing soldier asks for its own). Anything else is refused."""

    __slots__ = ("_pic", "_team", "_mates", "_seen")

    def __init__(self, picture, team, mates, seen):
        self._pic, self._team = picture, int(team)
        self._mates, self._seen = frozenset(mates), frozenset(seen)

    def _mate(self, uid):
        if int(uid) not in self._mates:
            raise KeyError(f"obs spec v1: uid {uid} is not a living mate of "
                           f"team {self._team}: its memory is out of scope")

    def _known(self, uid):
        if int(uid) not in self._mates and int(uid) not in self._seen:
            raise KeyError(f"obs spec v1: uid {uid} is neither a mate nor "
                           f"seen by team {self._team} now: out of scope")

    def motion(self, uid) -> tuple:
        self._mate(uid)
        return self._pic.velocity_q(uid)

    def bar(self, uid) -> float:
        self._known(uid)
        return float(self._pic.bars.get(int(uid), 0.0))

    def damage_last_s(self, uid) -> float:
        self._known(uid)
        return self._pic.damage_last_s(self._team, uid)

    def last_hit(self, uid):
        self._mate(uid)
        return self._pic.last_hit.get(int(uid))


class TeamKnowledge:
    """What team ``team`` may know at this decision (spec 0.5), built by
    :meth:`build` -- the one observation function that reads the sim."""

    __slots__ = ("team", "tick", "mates", "roster_ids", "records", "seen",
                 "objects", "doors", "trackers", "world", "_vis", "_lanes",
                 "_aim", "_cone", "_mate_aims", "_lof")

    def __init__(self):
        self._lanes = None
        self._aim = {}
        self._cone = {}
        self._mate_aims = None
        self._lof = {}

    @classmethod
    def build(cls, sim, picture, team: int) -> "TeamKnowledge":
        if picture is None:
            raise RuntimeError("obs spec v1: the sim has no team picture "
                               "(Simulation.start_team_picture)")
        team = int(team)
        k = cls()
        k.team, k.tick = team, int(sim.tick)
        k.world = ObsWorld(sim)
        vis = vision.state_for(sim)
        k._vis = vis
        k.mates = {int(u.id): u for u in sim.units
                   if u.alive and not u.is_zombie and int(u.team) == team}
        k.roster_ids = frozenset(int(u.id) for u in sim.roster(team)
                                 if u is not None)
        k.records = picture.records(team)
        by_id = {int(u.id): u for u in sim.units}
        k.seen = {int(uid): by_id[int(uid)]
                  for uid in vis.visible_by_team.get(team, ())
                  if int(uid) in k.records}
        k.objects = tuple(picture.seen_now[team])
        k.doors = tuple(d for d in sim._doors if d.alive)
        k.trackers = TrackerView(picture, team, k.mates, k.seen)
        return k

    def seen_by(self, mate_id: int, uid: int) -> bool:
        """Whether mate ``mate_id`` itself sees ``uid`` now (its own sight);
        refused for a non-mate."""
        if int(mate_id) not in self.mates:
            raise KeyError(f"obs spec v1: {mate_id} is not a living mate of "
                           f"team {self.team}: its sight is out of scope")
        return int(mate_id) in self._vis.observers_of(int(uid))

    @property
    def lanes(self):
        """The team's fire lanes (``team_picture.fire_lanes`` over the
        mates, the bots' dials), computed once per view on first read."""
        if self._lanes is None:
            from config import CFG
            from simulation.team_picture import fire_lanes_of
            self._lanes = fire_lanes_of(
                self.world.gmap, self.world.weapons_tables,
                list(self.mates.values()), self.team,
                _cfg_int(CFG, "bots", "lane_cell"),
                _cfg_int(CFG, "bots", "lane_halfwidth"))
        return self._lanes


# ---------------------------------------------------------------------------
# aim_of (spec 1.7): the 4.1 rule judged by the team's knowledge
# ---------------------------------------------------------------------------
def cone_q(know, ctx: ObsContext, u) -> int:
    """The half-angle (Q16 radians) of ``u``'s aim cone: its ranged
    weapon's ``spread_deg``, else its melee row's arc, a zombie (or a
    weaponless body) the reference kit's melee arc."""
    key = int(u.id)
    q = know._cone.get(key)
    if q is not None:
        return q
    rows = know.world.weapons_tables.weapons.by_name
    row = rows.get(getattr(u, "weapon_id", "") or "")
    melee = rows.get(getattr(u, "melee_id", "") or "")
    if u.is_zombie:
        q = ctx.ref_melee_arc_q
    elif row is not None and int(row.range_tiles) > 0:
        q = quantize_scalar(math.radians(float(row.spread_deg)))
    elif melee is not None:
        q = int(melee.melee_arc_half_q)
    else:
        q = ctx.ref_melee_arc_q
    know._cone[key] = q
    return q


def aim_of(know, ctx: ObsContext, u, me_id: int = -1) -> tuple:
    """``(category, uid)``: among the units the team knows (the mates at
    their live centres, the records at their recorded centres), the nearest
    one other than ``u`` -- by squared centre distance, ties the lower uid --
    whose bearing from ``u`` lies within ``u``'s cone of its facing ray
    (``facing_target_q`` for a mate, ``facing_q`` otherwise) and to which
    ``vision.ray_clear`` from ``u``'s centre tile holds. Category 0 nobody
    (uid -1), 1 ``me_id`` itself, 2 a team-mate, 3 a known unit of another
    side. Cached per ``(uid, facing ray, cone, centre)`` on the view."""
    uid_u = int(u.id)
    ray = int(u.facing_target_q if uid_u in know.mates else u.facing_q)
    cone = cone_q(know, ctx, u)
    ux, uy = int(u.center_tile_x()), int(u.center_tile_y())
    key = (uid_u, ray, cone, ux, uy)
    hit = know._aim.get(key)
    if hit is None:
        cands = []
        for uid, m in know.mates.items():
            if uid != uid_u:
                cands.append((int(m.center_tile_x()), int(m.center_tile_y()),
                              uid))
        for uid, rec in know.records.items():
            if uid != uid_u:
                cands.append((rec.centre[0], rec.centre[1], uid))
        inside = []
        for cx, cy, uid in cands:
            dx, dy = cx - ux, cy - uy
            bearing = atan2_q16(-dy << 16, dx << 16)
            if abs(wrap_q16(bearing - ray)) <= cone:
                inside.append((dx * dx + dy * dy, uid, cx, cy))
        inside.sort()
        hit = -1
        for _, uid, cx, cy in inside:
            if vision.ray_clear(know.world, uy, ux, cy, cx):
                hit = uid
                break
        know._aim[key] = hit
    if hit < 0:
        return (0, -1)
    if hit == int(me_id):
        return (1, hit)
    return (2 if hit in know.mates else 3, hit)


def _mate_aims(know, ctx):
    """Every living mate's ``aim_of`` answer (uid or -1) and trigger, once
    per view: the enemy entries' ``mates_aiming`` / ``mates_firing``."""
    if know._mate_aims is None:
        know._mate_aims = [(uid, aim_of(know, ctx, m)[1],
                            bool(getattr(m, "live_trigger", False)))
                           for uid, m in know.mates.items()]
    return know._mate_aims


# ---------------------------------------------------------------------------
# route (spec 1.7): the compass's field, the horde's flood
# ---------------------------------------------------------------------------
_ROUTE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_PLACE_PLANE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def place_plane(graph) -> np.ndarray:
    """Per tile the place ``room_graph.place_at`` names (its room, else the
    lowest-id opening holding it, else ``NO_PLACE``), cached per graph."""
    plane = _PLACE_PLANE.get(graph)
    if plane is None:
        plane = np.array(graph.grid, dtype=np.int64)
        # ascending ids, filling only what is still unplaced: the lowest id
        # wins, as place_at answers it
        for o in sorted(graph.openings, key=lambda o: int(o.id)):
            sub = plane[o.y:o.y + o.h, o.x:o.x + o.w]
            sub[sub < 0] = ORDER_OPENING_BASE + int(o.id)
        _PLACE_PLANE[graph] = plane
    return plane


def route(world, graph, where: int, fp: int) -> np.ndarray:
    """The route field to place ``where`` for a body of footprint ``fp``:
    ``horde.flood`` over the walkable tiles OR the tiles of every opening a
    unit could open (``room_graph.opening_states``), seeded at every
    passable anchor whose block centre lies in the place. Cached per
    ``(where, fp)`` on ``gmap._tile_serial`` and the openings' states."""
    gmap = world.gmap
    states = room_graph.opening_states(graph, gmap, world.doors, world.wires)
    key = (int(gmap._tile_serial),
           tuple((int(s.id), bool(s.openable)) for s in states))
    slot = _ROUTE_CACHE.setdefault(gmap, {})
    hit = slot.get((graph, int(where), int(fp)))
    if hit is not None and hit[0] == key:
        return hit[1]
    tile_ok = horde.walkable_tiles(gmap).copy()
    by_id = {int(o.id): o for o in graph.openings}
    for s in states:
        if s.openable:
            o = by_id[int(s.id)]
            tile_ok[o.y:o.y + o.h, o.x:o.x + o.w] = True
    passable = horde.block_passable(tile_ok, int(fp))
    plane = place_plane(graph)
    h, w = plane.shape
    half = int(fp) // 2
    centre_place = np.full((h, w), NO_PLACE, dtype=np.int64)
    centre_place[:h - half, :w - half] = plane[half:, half:]
    seeds = np.argwhere(passable & (centre_place == int(where)))
    field = horde.flood(gmap, tile_ok, int(fp), seeds.astype(np.int32))
    slot[(graph, int(where), int(fp))] = (key, field)
    return field


# ---------------------------------------------------------------------------
# The vector (section 1)
# ---------------------------------------------------------------------------
def _loadout_of(u):
    from simulation.loadouts import get_tables
    name = getattr(u, "loadout_name", None)
    return get_tables().loadouts.by_name.get(name) if name else None


def _row_index(table: str, name) -> int:
    if not name:
        return -1
    try:
        return int(unit_schema.table_row(table, name))
    except ValueError:
        return -1


def _place_centre(graph, where: int):
    """The centre tile ``(x, y)`` of an order's place, or None."""
    if graph is None or where == NO_PLACE:
        return None
    if where < ORDER_OPENING_BASE:
        for r in graph.rooms:
            if int(r.id) == where:
                return (int(r.x) + int(r.w) // 2, int(r.y) + int(r.h) // 2)
        return None
    for o in graph.openings:
        if ORDER_OPENING_BASE + int(o.id) == where:
            return (int(o.x) + int(o.w) // 2, int(o.y) + int(o.h) // 2)
    return None


def _status_left(u, kind: int) -> int:
    best = 0
    for st in (getattr(u, "statuses", None) or ()):
        if st.kind == kind and st.remaining_ticks > best:
            best = int(st.remaining_ticks)
    return best


def _hit_from(hit):
    """``(cos, sin)`` of where the hit came from: minus its travel vector,
    normalised, y negated into the facing convention; ``(0, 0)`` without."""
    if hit is None:
        return 0.0, 0.0
    dx, dy = float(hit[1]), float(hit[2])
    n = math.hypot(dx, dy)  # ingress-exempt: observation output for the network, float by design; never written to synced sim state
    if n == 0.0:
        return 0.0, 0.0
    return -dx / n, dy / n


def build_vector(know: TeamKnowledge, u, ctx: ObsContext, out) -> None:
    """Section 1 into ``out`` (``(V,)`` float32), for the living mate ``u``
    of ``know``'s team. The ``air`` block and ``felt_temp`` read 0 in (a)."""
    out[...] = 0.0
    world = know.world
    tick = know.tick
    rows = world.weapons_tables.weapons.by_name
    weapon = rows.get(getattr(u, "weapon_id", "") or "")
    melee = rows.get(getattr(u, "melee_id", "") or "")
    loadout = _loadout_of(u)
    tr = know.trackers
    # -- 1.1 body --
    out[0] = squash(ratio(u.current_hp, tr.bar(u.id)))
    f = int(u.facing_q)
    out[1] = cos_q16(f) / FP_ONE
    out[2] = sin_q16(f) / FP_ONE
    out[3] = squash(ratio(u.omega_q, world.unit_turn_rate_q16(u)))
    out[4] = wrap_q16(int(u.facing_target_q) - f) / PI_Q16
    vx, vy = tr.motion(u.id)
    out[5] = squash(ratio(vx, ctx.v_ref_q))
    out[6] = squash(ratio(vy, ctx.v_ref_q))
    out[7] = squash(ratio(u.pvx, ctx.v_fall_q))
    out[8] = squash(ratio(u.pvy, ctx.v_fall_q))
    out[9] = squash(ratio(u.effort, FP_ONE))
    out[10] = ratio(u.react_ticks, ctx.react_ticks)
    flags = composed_flags(u)
    out[11] = float(flags.is_prone)
    out[12] = float(flags.can_move)
    out[13] = float(flags.can_act)
    out[14] = float(flags.can_aim)
    for k in range(N_STATUS_KINDS):
        out[15 + k] = squash(ratio(_status_left(u, k), ctx.status_scale[k]))
    # -- 1.2 weapon and kit --
    st = combat.mag_status(u, weapon, tick) if weapon is not None else None
    mags_total = int(loadout.magazines) if loadout is not None else 0
    if st is None:
        out[24], out[25], out[26] = 1.0, 1.0, 1.0
    else:
        rounds, spares, reloading = st
        out[24] = ratio(rounds, weapon.mag_size)
        if spares is None:
            out[25], out[26] = 1.0, 1.0
        else:
            out[25] = squash(ratio(spares, max(1, mags_total)))
        out[27] = float(reloading)
        if reloading:
            out[28] = ratio(int(u.reload_done_tick) - tick, weapon.reload_ticks)
    counts = loadout.grenades if loadout is not None else {}
    for k, row in enumerate(ctx.hand_rows):
        have = int(counts.get(row, 0))
        if have > 0:
            out[29 + 2 * k] = ratio(u.grenade_counts[k], have)
            out[30 + 2 * k] = 1.0
    winding = int(u.throw_release_tick) >= 0
    out[41] = float(winding)
    if winding:
        out[42] = ratio(int(u.throw_release_tick) - tick, ctx.throw_windup_ticks)
        out[43] = float(int(u.throw_kind))
    else:
        out[43] = -1.0
    explosives = int(loadout.explosives) if loadout is not None else 0
    out[44] = ratio(u.has_explosive, explosives)
    out[45] = float(bool(loadout is not None and loadout.charge))
    planting = int(u.plant_done_tick) >= 0
    out[46] = float(planting)
    if planting:
        out[47] = ratio(int(u.plant_done_tick) - tick, ctx.plant_ticks)
        out[48] = ratio(int(u.plant_fuse_ticks), ctx.c4_fuse_max_ticks)
    if melee is not None and int(u.last_blow_tick) >= 0:
        rof = int(melee.rof_interval_ticks)
        out[49] = ratio(max(0, int(u.last_blow_tick) + rof - tick), rof)
    out[50] = float(bool(getattr(u, "melee_id", "")))
    cd = ctx.shove_cooldown_ticks
    if int(u.last_shove_tick) >= 0:
        out[51] = ratio(max(0, int(u.last_shove_tick) + cd - tick), cd)
    out[52] = float(_row_index("loadouts", getattr(u, "loadout_name", None)))
    out[53] = float(_row_index("weapons", getattr(u, "weapon_id", "")))
    if weapon is not None:
        out[54] = squash(ratio(weapon.range_tiles, ctx.l_ref))
        out[55] = squash(ratio(weapon.rof_interval_ticks, ctx.ref_rof_ticks))
        out[56] = squash(ratio(weapon.spread_deg, ctx.ref_spread_deg))
    out[57] = float(bool(getattr(u, "live_trigger", False)))
    # -- 1.3 the air: cols 58-65 and 96 are patch (b)'s (0 here) --
    # -- 1.4 what just happened to it --
    hit = tr.last_hit(u.id)
    if hit is not None:
        age = tick - int(hit[0])
        out[66] = float(age <= ctx.memory_ticks)
        out[67] = min(1.0, ratio(age, ctx.memory_ticks))
        out[68], out[69] = _hit_from(hit)
        out[70] = float(CAUSE_NAMES.index(hit[3]))
    else:
        out[67] = 1.0
        out[70] = -1.0
    # -- 1.5 previous choices, the clock, the team, the order --
    out[71:77] = -1.0
    out[77] = ratio(ctx.limit_ticks - tick, ctx.limit_ticks)
    team = know.team
    teams = ctx.spec.teams
    n_t = int(teams[team].n) if team < len(teams) else 0
    out[78] = ratio(len(know.mates), n_t)
    team_loadout = None
    if team < len(teams):
        from simulation.loadouts import get_tables
        team_loadout = get_tables().loadouts.by_name.get(teams[team].loadout)
    total = 0
    if team_loadout is not None:
        tw = rows.get(team_loadout.weapon)
        if tw is not None:
            total = n_t * int(team_loadout.magazines) * int(tw.mag_size)
    ammo = 0
    for m in know.mates.values():
        mw = rows.get(getattr(m, "weapon_id", "") or "")
        mst = combat.mag_status(m, mw, tick) if mw is not None else None
        if mst is not None:
            r_, s_, _ = mst
            ml = _loadout_of(m)
            s_ = int(ml.magazines) if (s_ is None and ml is not None) else \
                (s_ or 0)
            ammo += int(r_) + int(s_) * int(mw.mag_size)
    out[79] = squash(ratio(ammo, total))
    others = sum(int(t.n) for i, t in enumerate(teams) if i != team)
    marines = [r for r in know.records.values() if not r.zombie]
    zombies = [r for r in know.records.values() if r.zombie]
    out[80] = ratio(len(marines), others)
    out[81] = ratio(sum(1 for r in marines if r.uid in know.seen), others)
    nz = int(ctx.spec.zombies)
    out[82] = squash(ratio(len(zombies), nz))
    near2 = ctx.zombie_near_tiles * ctx.zombie_near_tiles
    mates_c = [(int(m.center_tile_x()), int(m.center_tile_y()))
               for m in know.mates.values()]
    near = sum(1 for z in zombies
               if any((z.centre[0] - mx) ** 2 + (z.centre[1] - my) ** 2
                      <= near2 for mx, my in mates_c))
    out[83] = squash(ratio(near, nz))
    what = int(u.order_what)
    out[84] = float(what)
    cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
    if what:
        where = int(u.order_where)
        until = int(u.order_until)
        out[85] = float(until != NO_LIMIT)
        if until != NO_LIMIT:
            out[86] = ratio(until - tick, ctx.limit_ticks)
        graph = world.graph
        if graph is not None:
            out[87] = float(room_graph.place_at(graph, cy, cx) == where)
            out[88] = 1.0 if where < ORDER_OPENING_BASE else 2.0
            pc = _place_centre(graph, where)
            if pc is not None:
                pdx, pdy = pc[0] - cx, pc[1] - cy
                out[89] = squash(ratio(pdx, ctx.l_ref))
                out[90] = squash(ratio(pdy, ctx.l_ref))
                out[91] = d_c(math.hypot(pdx, pdy), ctx.l_ref)  # ingress-exempt: observation output for the network, float by design; never written to synced sim state
            field = route(world, graph, where, int(u.footprint))
            ay, ax = int(u.tile_y), int(u.tile_x)
            cands = horde.candidates(field, ay, ax)
            if cands:
                _, _, ny, nx = cands[0]
                out[92], out[93] = float(nx - ax), float(ny - ay)
            here = horde.field_at(field, ay, ax)
            out[94] = (-1.0 if here == horde.UNREACHED
                       else squash(ratio(here, ctx.l_ref)))
        else:
            out[94] = -1.0
        out[95] = float(int(u.order_target) != NO_TARGET)
    else:
        out[94] = -1.0


# ---------------------------------------------------------------------------
# The list (section 2)
# ---------------------------------------------------------------------------
def _door_centre(door) -> tuple:
    """A door's centre tile ``(x, y)``: the middle tile of its span (CHOICE:
    an even span takes the upper-middle element)."""
    t = door.span[len(door.span) // 2]
    return (int(t[1]), int(t[0]))


def _entry_common(e, etype, dx, dy, ctx):
    e[0] = float(etype)
    e[1] = squash(ratio(dx, ctx.l_ref))
    e[2] = squash(ratio(dy, ctx.l_ref))
    e[3] = d_c(math.hypot(dx, dy), ctx.l_ref)  # ingress-exempt: observation output for the network, float by design; never written to synced sim state


def _unit_columns(e, it, ctx, tick, rows):
    """Cols 11-23 of a living unit (a mate or a seen enemy); col 10, the
    hp ratio over the bar, is the caller's (the bar is read in scope)."""
    e[11] = ratio(int(it.exposure[KINETIC]), ctx.x_max_q)
    e[12] = float(bool(getattr(it, "live_trigger", False)))
    w = rows.get(getattr(it, "weapon_id", "") or "")
    st = combat.mag_status(it, w, tick) if w is not None else None
    if st is not None and st[2]:
        e[13] = 1.0
        e[14] = ratio(int(it.reload_done_tick) - tick, w.reload_ticks)
    if int(it.throw_release_tick) >= 0:
        e[15] = 1.0
        e[16] = ratio(int(it.throw_release_tick) - tick, ctx.throw_windup_ticks)
        e[17] = float(int(it.throw_kind))
    else:
        e[17] = -1.0
    if int(it.plant_done_tick) >= 0:
        e[18] = 1.0
        e[19] = ratio(int(it.plant_done_tick) - tick, ctx.plant_ticks)
        e[20] = ratio(int(it.plant_fuse_ticks), ctx.c4_fuse_max_ticks)
    fl = composed_flags(it)
    e[21] = float(not fl.can_act)
    e[22] = float(not fl.can_aim)
    e[23] = float(fl.is_prone)


def _candidates(know, u):
    """The five blocks' candidates, each ``(sort key, payload)``, nearest
    first by squared centre distance, ties by the lower id."""
    cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
    me = int(u.id)

    def d2(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2

    mates = sorted(((d2(m.center_tile_x(), m.center_tile_y()), uid, m)
                    for uid, m in know.mates.items() if uid != me),
                   key=lambda t: t[:2])
    seen, recs = [], []
    for uid, rec in know.records.items():
        key = (d2(rec.centre[0], rec.centre[1]), uid)
        if uid in know.seen:
            seen.append((*key, rec, know.seen[uid]))
        else:
            recs.append((*key, rec))
    seen.sort(key=lambda t: t[:2])
    recs.sort(key=lambda t: t[:2])
    objs = sorted(((d2(object_tile(o)[1], object_tile(o)[0]), ouid, o)
                   for o, ouid in know.objects), key=lambda t: t[:2])
    doors = sorted(((d2(*_door_centre(d)), int(d.ordinal), d)
                    for d in know.doors), key=lambda t: t[:2])
    return mates, seen, recs, objs, doors


def build_list(know: TeamKnowledge, u, ctx: ObsContext, out, mask, uid) -> None:
    """Section 2 into ``out`` (``(E, F)`` float32), ``mask`` (``(E,)``
    uint8) and ``uid`` (``(E,)`` int32) for the living mate ``u``."""
    out[...] = 0.0
    mask[...] = 0
    uid[...] = -1
    world = know.world
    tick = know.tick
    rows = world.weapons_tables.weapons.by_name
    me = int(u.id)
    cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
    my_weapon = rows.get(getattr(u, "weapon_id", "") or "")
    range2 = int(my_weapon.range_tiles) ** 2 if my_weapon is not None else -1
    spread_q = (quantize_scalar(math.radians(float(my_weapon.spread_deg)))
                if my_weapon is not None else 0)
    throw2 = ctx.max_throw_range ** 2
    frag2 = ctx.frag_radius ** 2
    a_me = world.unit_turn_accel_q16(u)
    w_me = world.unit_turn_rate_q16(u)
    dw_me = unit_impulse.pending_dw(world.impulse_table, u)
    cap = 4 * ctx.half_turn_ticks
    mates_c = [(int(m.center_tile_x()), int(m.center_tile_y()), muid)
               for muid, m in know.mates.items()]
    others_c = [(x, y) for x, y, muid in mates_c if muid != me]
    team = know.team
    teams = ctx.spec.teams
    n_t = int(teams[team].n) if team < len(teams) else 0
    my_where = int(u.order_where) if int(u.order_what) else NO_PLACE
    graph = world.graph
    mates, seen, recs, objs, doors = _candidates(know, u)
    caps = ctx.caps
    i = 0

    # -- type 0: team-mates --
    for _, muid, m in mates[:caps[0]]:
        e = out[i]
        mx, my = int(m.center_tile_x()), int(m.center_tile_y())
        _entry_common(e, TYPE_MATE, mx - cx, my - cy, ctx)
        f = int(m.facing_q)
        e[4], e[5] = cos_q16(f) / FP_ONE, sin_q16(f) / FP_ONE
        vx, vy = know.trackers.motion(muid)
        e[6], e[7] = squash(ratio(vx, ctx.v_ref_q)), squash(ratio(vy, ctx.v_ref_q))
        e[8] = float(int(m.team))
        e[9] = float(_row_index("loadouts", getattr(m, "loadout_name", None)))
        e[10] = squash(ratio(m.current_hp, know.trackers.bar(muid)))
        _unit_columns(e, m, ctx, tick, rows)
        ft = int(m.facing_target_q)
        e[39], e[40] = cos_q16(ft) / FP_ONE, sin_q16(ft) / FP_ONE
        e[41] = float(aim_of(know, ctx, m, me)[0] != 0)
        mv = getattr(m, "live_move_dir", None)
        if mv is not None:
            e[42] = int(mv.dx_q) / FP_ONE
            e[43] = int(mv.dy_q) / FP_ONE
            e[44] = ratio(int(mv.throttle_q), THROTTLE_ONE)
        if int(m.throw_release_tick) >= 0:
            rq = int(m.throw_reach_q)
            reach = 6.0 if rq == THROW_NO_REACH else rq / FP_ONE
            lx = mx + reach * (cos_q16(ft) / FP_ONE) - cx
            ly = my - reach * (sin_q16(ft) / FP_ONE) - cy
            e[45], e[46] = squash(ratio(lx, ctx.l_ref)), squash(ratio(ly, ctx.l_ref))
        what = int(m.order_what)
        e[47] = float(what)
        if what:
            pc = _place_centre(graph, int(m.order_where))
            if pc is not None:
                e[48] = squash(ratio(pc[0] - cx, ctx.l_ref))
                e[49] = squash(ratio(pc[1] - cy, ctx.l_ref))
            e[50] = float(my_where != NO_PLACE
                          and int(m.order_where) == my_where)
            until = int(m.order_until)
            if until != NO_LIMIT:
                e[51] = ratio(until - tick, ctx.limit_ticks)
        # cols 52, 53 (the lane pair) are patch (b)'s; col 66 is 1h's
        mask[i], uid[i] = 1, muid
        i += 1
    i = caps[0]

    # -- type 1: enemies seen now --
    for _, euid, rec, it in seen[:caps[1]]:
        e = out[i]
        ex, ey = rec.centre
        dx, dy = ex - cx, ey - cy
        _entry_common(e, TYPE_SEEN, dx, dy, ctx)
        f = int(it.facing_q)
        e[4], e[5] = cos_q16(f) / FP_ONE, sin_q16(f) / FP_ONE
        e[6] = squash(ratio(rec.vel_q[0], ctx.v_ref_q))
        e[7] = squash(ratio(rec.vel_q[1], ctx.v_ref_q))
        e[8] = float(int(rec.team))
        e[9] = (float(ctx.zombie_class) if rec.zombie
                else float(_row_index("loadouts", rec.loadout)))
        e[10] = squash(ratio(it.current_hp, know.trackers.bar(euid)))
        _unit_columns(e, it, ctx, tick, rows)
        e[24] = float(know.seen_by(me, euid))
        lof_key = (cx, cy, ex, ey)
        lof = know._lof.get(lof_key)
        if lof is None:
            lof = line_of_fire.solid_in_way(world, (cx, cy), (ex, ey))
            know._lof[lof_key] = lof
        e[25] = float(lof)
        e[26] = float(line_of_fire.mate_in_way((cx, cy), (ex, ey), others_c,
                                               ctx.line_clear_tiles,
                                               int(u.footprint)))
        want = atan2_q16(-dy << 16, dx << 16)
        ticks = turning.ticks_to_within(u, want, a_me, w_me, dw_me, spread_q,
                                        cap)
        e[27] = squash(ratio(ticks, ctx.half_turn_ticks))
        # col 28 (on_next_tick) is 1h's
        d2 = dx * dx + dy * dy
        e[29] = float(d2 <= range2)
        e[30] = float(d2 <= throw2)
        cat, _ = aim_of(know, ctx, it, me)
        e[31] = float(cat == 1)
        e[32] = float(cat)
        if it.is_zombie:
            e[33] = (2.0 if it.zombie_activated
                     else 3.0 if int(it.zombie_noise_ticks) > 0 else 1.0)
        aiming = [(muid, trig) for muid, aimed, trig in _mate_aims(know, ctx)
                  if muid != me and aimed == euid]
        e[34] = ratio(len(aiming), n_t)
        e[35] = ratio(sum(1 for _, trig in aiming if trig), n_t)
        e[36] = squash(ratio(know.trackers.damage_last_s(euid),
                             know.trackers.bar(euid)))
        e[37] = float(int(u.order_target) == euid)
        e[38] = ratio(sum(1 for mx, my, _ in mates_c
                          if (mx - ex) ** 2 + (my - ey) ** 2 <= frag2), n_t)
        mask[i], uid[i] = 1, euid
        i += 1
    i = caps[0] + caps[1]

    # -- type 2: last-seen records --
    for _, ruid, rec in recs[:caps[2]]:
        e = out[i]
        _entry_common(e, TYPE_RECORD, rec.centre[0] - cx, rec.centre[1] - cy,
                      ctx)
        f = int(rec.facing_q)
        e[4], e[5] = cos_q16(f) / FP_ONE, sin_q16(f) / FP_ONE
        e[6] = squash(ratio(rec.vel_q[0], ctx.v_ref_q))
        e[7] = squash(ratio(rec.vel_q[1], ctx.v_ref_q))
        e[8] = float(int(rec.team))
        e[9] = (float(ctx.zombie_class) if rec.zombie
                else float(_row_index("loadouts", rec.loadout)))
        e[54] = ratio(tick - int(rec.tick), ctx.memory_ticks)
        mask[i], uid[i] = 1, ruid
        i += 1
    i = caps[0] + caps[1] + caps[2]

    # -- type 3: grenades and charges --
    for _, ouid, o in objs[:caps[3]]:
        e = out[i]
        oy, ox = object_tile(o)
        _entry_common(e, TYPE_OBJECT, ox - cx, oy - cy, ctx)
        charge = hasattr(o, "det_tick")
        if not charge:
            ddx, ddy = o.target_fx - o.start_fx, o.target_fy - o.start_fy
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            landed = (o.fx == o.target_fx and o.fy == o.target_fy)
            if dist >= 0.1 and not landed:
                vt = o.travel_speed / ctx.tps / dist
                e[6] = squash(ratio(ddx * vt, ctx.v_ref_q / FP_ONE))
                e[7] = squash(ratio(ddy * vt, ctx.v_ref_q / FP_ONE))
        e[8], e[9] = -1.0, -1.0
        name = o.ammo_name
        e[55] = float(ctx.obj_rows.index(name)) if name in ctx.obj_rows \
            else -1.0
        owner = int(o.owner_id)
        e[56] = (0.0 if owner == me else 1.0 if owner in know.roster_ids
                 else 2.0 if owner >= 0 else 3.0)
        if charge:
            fam = ctx.c4_fuse_max_ticks
            if o.det_tick is None:
                e[58] = 1.0
            else:
                e[57] = squash(ratio(int(o.det_tick) - tick, fam))
            e[61] = 1.0
        else:
            e[57] = squash(ratio(int(o.get_detonate_tick()) - tick,
                                 ctx.hand_fuse_max_ticks))
            e[61] = float(o.fx == o.target_fx and o.fy == o.target_fy)
        try:
            r = int(world.weapons_tables.payload_for_ammo(name).radius)
        except KeyError:
            r = 0
        e[59] = squash(ratio(r, ctx.l_ref))
        e[60] = float((ox - cx) ** 2 + (oy - cy) ** 2 <= r * r)
        mask[i], uid[i] = 1, ouid
        i += 1
    i = caps[0] + caps[1] + caps[2] + caps[3]

    # -- type 4: doors --
    wired = room_graph.wired_door_ordinals(world.wires)
    for _, ordinal, d in doors[:caps[4]]:
        e = out[i]
        dx_, dy_ = _door_centre(d)
        _entry_common(e, TYPE_DOOR, dx_ - cx, dy_ - cy, ctx)
        e[8], e[9] = -1.0, -1.0
        e[62] = float(d.state == DOOR_OPEN)
        e[63] = float(verb_gates.door_in_reach(u, d))
        e[64] = float(horde.is_hull_door(world.gmap, d.span))
        e[65] = float(ordinal in wired)
        mask[i], uid[i] = 1, DOOR_UID_BASE + ordinal
        i += 1


# ---------------------------------------------------------------------------
# The door's body: every row of an env (spec 0.4, 4)
# ---------------------------------------------------------------------------
def observe_rows(sim, ctx: ObsContext, units, out: dict) -> None:
    """Write ``vector``, ``list``, ``list_mask`` and ``list_uid`` of ``out``
    (the ``(U, ...)`` views) for ``units`` (one living mate or None per
    row): one :class:`TeamKnowledge` per marine team of the rows, then the
    builders. The row flags are the env's (spec 4.2). With :meth:`TeamKnowledge.build`,
    the one place the observation reads the sim."""
    out["vector"][...] = 0.0
    out["list"][...] = 0.0
    out["list_mask"][...] = 0
    out["list_uid"][...] = -1
    knows = {}
    for i, u in enumerate(units):
        if u is None:
            continue
        team = int(u.team)
        know = knows.get(team)
        if know is None:
            know = knows[team] = TeamKnowledge.build(sim, sim.team_picture,
                                                     team)
        build_vector(know, u, ctx, out["vector"][i])
        build_list(know, u, ctx, out["list"][i], out["list_mask"][i],
                   out["list_uid"][i])


def shapes(n_envs: int, n_rows: int, n_slots: int) -> dict:
    """The spec's observation keys and shapes (section 4.1) for the env's
    layout: ``vector``, ``list``, ``list_mask``, ``list_uid``, and the two
    row flags ``valid`` / ``in_match``."""
    e = int(n_slots) + ENTRIES_BEYOND_SLOTS
    n, u = int(n_envs), int(n_rows)
    return {
        "vector": ((n, u, V), np.float32),
        "list": ((n, u, e, F), np.float32),
        "list_mask": ((n, u, e), np.uint8),
        "list_uid": ((n, u, e), np.int32),
        "valid": ((n, u), np.uint8),
        "in_match": ((n, u), np.uint8),
    }


__all__ = [
    "SPEC_NAME", "V", "F", "VECTOR_COLUMNS", "ENTRY_COLUMNS", "DENOMINATORS",
    "FLAG", "KIND", "ANGLE", "UNIT", "Feature", "squash", "ratio", "d_c",
    "ObsContext", "ObsWorld", "TeamKnowledge", "TrackerView", "aim_of",
    "cone_q", "route", "place_plane", "status_scale_ticks", "build_vector",
    "build_list", "observe_rows", "shapes", "TYPE_MATE", "TYPE_SEEN",
    "TYPE_RECORD", "TYPE_OBJECT", "TYPE_DOOR", "CAPS_FIXED",
    "ENTRIES_BEYOND_SLOTS", "OBJECT_UID_BASE", "DOOR_UID_BASE",
    "HAND_GRENADE_SLOTS",
]
