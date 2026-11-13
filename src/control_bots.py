"""``BotTeam`` -- the layered bot team (#183 G8p), match controller ``"bots"``.

``docs/bots_zombies_speeds_design_2026-11-10.md`` §2 (Erik's rulings: the
three-layer shape, bots stay Python on the host, the techniques good AND
cheap; "the best old-school AI we can give the agents as an opponent and a
baseline"). The first bots (#121) live on unchanged as ``bots_v1``
(``control_bots_v1.BotTeamV1``): a ladder of opponents.

THE THREE LAYERS
----------------
1. **Team** (:class:`TeamBrain`, one per team): the shared PICTURE (every
   enemy any teammate saw, with its last-seen tile and tick; updated every
   tick from the sight oracle, cheap), and every ``team_period_ticks``: the
   counts, the PLAY (the best-scoring row of :data:`PLAYS`), the shared
   TARGET (focus fire: most exposed by the exposure law, then weakest, then
   nearest to the team), the objective and its distance field, the Bound
   halves and the teammates' fire lanes on a small grid.
2. **Unit** (per bot, every ``unit_period_ticks`` staggered by unit id, and
   at once when something new comes into its sight): every row of
   :data:`ACTIONS` is SCORED and the best wins. An action holds a CHANNEL
   (``hands``: shoot / hold fire / frag / flashbang / shove / melee;
   ``feet``: move / retreat / stand); the resolution takes the best action,
   then the best one whose channels are still free. Between decisions the
   committed actions are EXECUTED every tick (the aim follows the target, the
   trigger obeys the turning law's predictor, the feet follow their A*
   path). A bot COMMITS to its position until its next decision (#138), and
   the next decision keeps it unless a candidate beats it by
   ``commit_margin``.
3. **Position** (:func:`choose_position`): candidate anchors near the bot,
   each judged by the named yes / no FEATURES of :data:`FEATURES` (the
   design's §2.2 list), scored by a hand-set weight per feature
   (``[bots.features]``); cheap features first over every candidate, the
   line-of-sight features on the best ``los_budget`` only. The bot paths to
   the winner with ``pathfinding.astar`` (the only pathfinder). The feature
   list is the interface a learned scorer reads later (§2.6): nothing here
   is learned.

A new behaviour is a new row of :data:`ACTIONS`, :data:`PLAYS` or
:data:`FEATURES` (with its ``[bots.*]`` weight) -- never a branch in the
executor, never a second controller.

Determinism. Bots are outside the synced digest but their intents are not:
every decision reads synced state (integer tiles, Q16 facings, the per-tick
vision cache, the field accessor), iterates units in id order, scores in
INTEGERS (every weight quantized once to Q16 at the door), and breaks ties by
row order and a hash of (match seed, unit id, tick, candidate) -- never a
drawn stream, never the wall clock, never a set's order. A bots match
replays exactly and is identical run to run (``tests/test_bots.py``).

Everything reaches the sim through the per-tick intent facade
(``set_move_dir`` / ``clear_move_dir`` / ``set_aim`` / ``set_trigger`` /
``throw_grenade_intent`` / ``shove_intent`` / ``melee_intent`` /
``use_intent``). The bots never call ``reload_intent`` (RULED: they rely on
the empty gun's automatic reload; the verb is for agents and the human).
Pyray-free; works with ``renderer=None``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import CFG
from control_bots_v1 import ARRIVE_TOL_Q, MID_Q, _direction_q
from control_source import ControlSource
from pathfinding import astar
from simulation import horde, vision
from simulation.combat import mag_status, melee_arc_target
from simulation.damage import DAMAGE_TYPE_NAMES
from simulation.factions import ZOMBIE_TEAM
from simulation.intents import FP_ONE
from simulation.movement import THROTTLE_ONE
from simulation.room_graph import place_at
from simulation.sensor_accessor import Channel, EntityFieldAccessor
from simulation.team_picture import fire_lanes
from simulation.turning import wrap_q16
from simulation.unit_fixed import atan2_q16, quantize_scalar
from simulation.unit_impulse import shove_target
from simulation import unit_orders
from simulation.weapons import FRAG_ROW, fuse_ticks_from_seconds
from simulation.weapons import get_tables as weapon_tables

#: The flashbang's hand-grenade row (G3p).
FLASH_ROW = "grenade_flash"
#: The exposure law's damage type the carbine raises (G4p): focus fire
#: reads this level.
KINETIC = next(t for t, n in DAMAGE_TYPE_NAMES.items() if n == "kinetic")
#: Unit-decision stagger: unit ``id`` decides on ticks where
#: ``(tick + id * STAGGER) % unit_period_ticks == 0``.
STAGGER = 3
_M64 = (1 << 64) - 1


# ===========================================================================
# Settings -- the [bots] table (the refusing door)
# ===========================================================================
#: Scalar keys of ``[bots]`` read by the layered bots: key -> (type, lower
#: bound). The first bots' seven keys sit beside them, read by
#: ``control_bots_v1.bot_v1_settings`` (the shared ``group_radius``,
#: ``danger_radius``, ``grenade_cooldown_s`` and ``replan_ticks`` mean the
#: same in both; ``friendly_line_tiles`` is the first bots' alone).
_SCALARS = {
    "team_period_ticks": (int, 1),
    "unit_period_ticks": (int, 1),
    "replan_ticks": (int, 1),
    "flash_memory_s": (float, 0.0),
    "candidate_radius": (int, 1),
    "candidate_stride": (int, 1),
    "reach_steps": (int, 1),
    "progress_steps": (int, 1),
    "advance_steps": (int, 1),
    "plan_max_steps": (int, 1),
    "los_budget": (int, 1),
    "commit_margin": (float, 0.0),
    "spacing_tiles": (int, 0),
    "lane_cell": (int, 1),
    "lane_halfwidth": (int, 0),
    "cover_min_tiles": (int, 1),
    "smoke_dense": (float, 0.0),
    "vacuum_atm": (float, 0.0),
    "hot_temp": (float, 0.0),
    "line_clear_tiles": (int, 0),
    "group_radius": (int, 1),
    "frag_min_group": (int, 1),
    "danger_radius": (int, 1),
    "grenade_cooldown_s": (float, 0.0),
    "zombie_engage_tiles": (int, 1),
    "zombie_near_tiles": (int, 1),
    "outnumber_margin": (int, 0),
    "bound_max_s": (float, 0.0),
    "travel_throttle": (float, 0.0),
    "door_retry_ticks": (int, 1),
}


@dataclass(frozen=True)
class BotSettings:
    """The ``[bots]`` table as the layered bots read it. ``features`` /
    ``actions`` / ``plays`` map a registry row's name to its weight in Q16
    (quantized once here: every score is an integer)."""
    scalars: dict
    features: dict
    actions: dict
    plays: dict

    def __getattr__(self, key):
        try:
            return self.__dict__["scalars"][key]
        except KeyError:
            raise AttributeError(key) from None


def _weights(section, sub: str, names) -> dict:
    table = getattr(section, sub, None)
    if table is None:
        raise ValueError(f"config.toml has no [bots.{sub}] table")
    have = vars(table)
    for name in names:
        if name not in have:
            raise ValueError(f"[bots.{sub}] is missing {name!r}")
    for name in have:
        if name not in names:
            raise ValueError(f"[bots.{sub}] has unknown row {name!r} "
                             f"(known: {list(names)})")
    out = {}
    for name in names:
        v = have[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"[bots.{sub}] {name} must be a number, "
                             f"got {v!r}")
        out[name] = int(round(float(v) * FP_ONE))
    return out


def bot_settings(cfg) -> BotSettings:
    """The ``[bots]`` table of a loaded config for the layered bots. Raises
    ValueError naming the key when the section, a scalar key, a weight table
    or one of its rows is missing, mistyped, out of range or unknown -- a
    weight that silently fell back to a default would hide a typo."""
    section = getattr(cfg, "bots", None)
    if section is None:
        raise ValueError("config.toml has no [bots] section")
    scalars = {}
    for key, (kind, lo) in _SCALARS.items():
        if not hasattr(section, key):
            raise ValueError(f"[bots] is missing {key!r}")
        v = getattr(section, key)
        if isinstance(v, bool) or not isinstance(
                v, (int,) if kind is int else (int, float)):
            raise ValueError(f"[bots] {key} must be a {kind.__name__}, "
                             f"got {v!r}")
        v = kind(v)
        if v < lo:
            raise ValueError(f"[bots] {key} must be >= {lo}, got {v!r}")
        scalars[key] = v
    if scalars["travel_throttle"] > 1.0:
        raise ValueError("[bots] travel_throttle must be in [0, 1], got "
                         f"{scalars['travel_throttle']!r}")
    return BotSettings(
        scalars=scalars,
        features=_weights(section, "features", [f.name for f in FEATURES]),
        actions=_weights(section, "actions", [a.name for a in ACTIONS]),
        plays=_weights(section, "plays", [p.name for p in PLAYS]))


# ===========================================================================
# Small integer helpers
# ===========================================================================
def _centre(u) -> tuple:
    return (u.center_tile_x(), u.center_tile_y())


def _d2(a, b) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _mix64(x):
    """splitmix64's finaliser on a numpy uint64 array (wraps mod 2^64): the
    bots' seedless tie-break hash -- a pure function, never a stream."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def tie_hash(seed: int, unit_id: int, tick: int, n: int) -> np.ndarray:
    """``n`` hashed integers in ``[0, 2^16)`` for (match seed, unit id,
    tick, index): the jitter that breaks equal scores, so two seeds play two
    matches while one seed plays one."""
    base = ((int(seed) & 0xFFFFFFFF) * 0x9E3779B97F4A7C15
            + (int(unit_id) & 0xFFFF) * 0xC2B2AE3D27D4EB4F
            + (int(tick) & 0xFFFFFFFF) * 0x165667B19E3779F9) & _M64
    with np.errstate(over="ignore"):
        x = np.uint64(base) + np.arange(n, dtype=np.uint64) * np.uint64(
            0x9E3779B97F4A7C15)
        return (_mix64(x) >> np.uint64(48)).astype(np.int64)


def los(sim, a, b) -> bool:
    """A line of sight between two centre tiles ``(x, y)``: THE sight
    oracle's occlusion (``vision.ray_clear`` = ``GameMap.has_los`` + the
    sight-blocking cover)."""
    return vision.ray_clear(sim, a[1], a[0], b[1], b[0])


# ===========================================================================
# The team layer
# ===========================================================================
@dataclass
class TeamCounts:
    """The counts the play is chosen from (§2.3: enemies seen, teammates
    alive, ammunition left, zombies near)."""
    mine: int = 0
    marines_known: int = 0
    marines_seen: int = 0
    zombies_known: int = 0
    zombies_near: int = 0
    contact: bool = False
    ammo_rounds: int = 0


@dataclass(frozen=True)
class Play:
    """A team play (§2.3): ``eligible(team, counts) -> bool``; among the
    eligible the highest ``[bots.plays]`` weight wins (ties: row order)."""
    name: str
    eligible: object


def _play_fall_back(team, c) -> bool:
    return c.zombies_near > 0 or (
        c.marines_known >= c.mine + team.s.outnumber_margin + 1)


def _play_hold(team, c) -> bool:
    return c.marines_seen > 0


def _play_bound(team, c) -> bool:
    return True


#: The plays (v1: three; room breaching is the fourth, a later round).
PLAYS = (
    Play("fall_back", _play_fall_back),
    Play("hold", _play_hold),
    Play("bound", _play_bound),
)


#: The order each play gives (#195 1f, design 5.3): the commander
#: (:class:`BotCommander`) writes ``PLAY_ORDER[brain.play]`` -- a code name
#: of ``unit_orders.ORDER_WHAT`` -- to every living marine of a door-driven
#: team. A row per :data:`PLAYS` row (a fourth play must name its order);
#: ``clear`` is not written by the bots.
PLAY_ORDER = {"fall_back": "fall_back_to", "hold": "hold", "bound": "go_to"}


def check_play_order(plays=PLAYS, table=PLAY_ORDER) -> None:
    """Refuse, by name, a play with no :data:`PLAY_ORDER` row and a row
    naming no order kind (run at import)."""
    missing = [p.name for p in plays if p.name not in table]
    if missing:
        raise ValueError(f"PLAY_ORDER: the plays {missing} name no order "
                         f"(a play must say which order the commander "
                         f"gives; #195 1f)")
    bad = {k: v for k, v in table.items()
           if v not in unit_orders.ORDER_WHAT[1:]}
    if bad:
        raise ValueError(f"PLAY_ORDER: {bad} name no order kind "
                         f"({unit_orders.ORDER_WHAT[1:]})")


check_play_order()


class TeamBrain:
    """One team's play, target and objective, read off the team picture
    (``simulation/team_picture.py``, #195 1d: the picture is the sim's, the
    bots only read it)."""

    def __init__(self, team: int, s: BotSettings, tps: float):
        self.team = int(team)
        self.s = s
        self.flash_memory_ticks = max(1, int(round(s.flash_memory_s * tps)))
        self.bound_max_ticks = max(1, int(round(s.bound_max_s * tps)))
        self.picture = {}                # enemy id -> team_picture.Record
        self.counts = TeamCounts()
        self.play = "bound"
        self.target = None               # the shared target (a unit) or None
        self.objective = None            # top-left tile (x, y) or None
        self.anchor = None               # the Hold play's anchor tile
        self.obj_field = None            # flood from the objective
        self._obj_key = None
        self.bound_moving = 0            # which Bound half moves (0 / 1)
        self.bound_since = 0
        self.halves: dict = {}           # unit id -> 0 / 1
        self.lanes = None                # coarse grid of lane owners
        self.updated_at = None
        self.centroid = (0, 0)

    # -- every tick: the picture ----------------------------------------
    def observe(self, sim) -> None:
        """Read this tick's records off the sim's team picture (the
        conductor keeps it; never built here)."""
        pic = getattr(sim, "team_picture", None)
        if pic is None:
            raise RuntimeError(
                "the layered bots read the team picture: this sim has none "
                "(a match starts it in spawn_match; a hand-built sim calls "
                "Simulation.start_team_picture)")
        self.picture = pic.records(self.team)

    def known(self, zombies=None):
        """The picture's records in id order (``zombies``: only / none)."""
        out = []
        for eid in sorted(self.picture):
            sg = self.picture[eid]
            if zombies is None or sg.zombie == zombies:
                out.append(sg)
        return out

    # -- a few times a second: counts, play, target, objective ----------
    def update(self, bots, sim, mine, vis) -> None:
        s = self.s
        tick = int(sim.tick)
        self.updated_at = tick
        c = TeamCounts(mine=len(mine))
        marines = self.known(zombies=False)
        zombies = self.known(zombies=True)
        seen_ids = set(vis.visible_by_team.get(self.team, ()))
        c.marines_known = len(marines)
        c.marines_seen = sum(1 for sg in marines if sg.uid in seen_ids)
        c.zombies_known = len(zombies)
        c.contact = bool(marines)
        near2 = s.zombie_near_tiles * s.zombie_near_tiles
        c.zombies_near = sum(
            1 for z in zombies
            if any(_d2(z.centre, _centre(m)) <= near2 for m in mine))
        for m in mine:
            st = mag_status(m, bots.weapon_of(m), tick) if bots.weapon_of(m) \
                else None
            if st is not None:
                rounds, spares, _r = st
                c.ammo_rounds += rounds + (spares or 0) * bots.weapon_of(
                    m).mag_size
        self.counts = c
        # the play: the eligible row with the highest weight
        best = None
        for p in PLAYS:
            if p.eligible(self, c):
                w = s.plays[p.name]
                if best is None or w > best[0]:
                    best = (w, p.name)
        play = best[1]
        if play != self.play:
            self.play = play
            self.bound_since = tick
            self.anchor = None
        # the shared target: focus fire
        cx = sum(m.center_tile_x() for m in mine) // len(mine)
        cy = sum(m.center_tile_y() for m in mine) // len(mine)
        self.centroid = (cx, cy)
        units = {u.id: u for u in sim.units}
        cands = [units[sg.uid] for sg in self.known() if sg.uid in seen_ids]
        self.target = min(cands, key=lambda e: focus_key(e, (cx, cy)),
                          default=None)
        # the objective
        self.objective = self._objective(bots, sim, mine)
        self._refresh_field(bots, sim, mine[0].footprint)
        # the Bound halves (roster order) and the swap
        ids = sorted(m.id for m in mine)
        self.halves = {uid: i % 2 for i, uid in enumerate(ids)}
        if self.play == "bound":
            movers = [m for m in mine
                      if self.halves[m.id] == self.bound_moving]
            arrived = all(bots.arrived(m) for m in movers)
            if (arrived and tick - self.bound_since > s.unit_period_ticks) \
                    or tick - self.bound_since >= self.bound_max_ticks:
                self.bound_moving ^= 1
                self.bound_since = tick
        # the fire lanes, from state (every teammate holding its trigger)
        self.lanes = fire_lanes(sim, self.team, s.lane_cell,
                                s.lane_halfwidth)

    def _objective(self, bots, sim, mine):
        fp = mine[0].footprint
        if self.play == "fall_back":
            return bots.room_goal(sim, self.team, fp)
        if self.play == "hold":
            if self.anchor is None:
                # hold where the team stands: the anchor is the teammate
                # nearest the centroid (a tile a body stands on)
                m = min(mine, key=lambda m: (_d2(_centre(m), self.centroid),
                                             m.id))
                self.anchor = (m.tile_x, m.tile_y)
            return self.anchor
        # bound: the shared target, the nearest known marine, else the
        # enemy spawn room
        known = self.known(zombies=False)
        if self.target is not None and not self.target.is_zombie:
            return (self.target.tile_x, self.target.tile_y)
        if known:
            sg = min(known, key=lambda sg: (_d2(sg.centre, self.centroid),
                                            sg.uid))
            return sg.tile
        return bots.enemy_room_goal(sim, fp)

    def _refresh_field(self, bots, sim, fp) -> None:
        if self.objective is None:
            self.obj_field = None
            self._obj_key = None
            return
        key = (self.objective, fp, bots.topology_key(sim))
        if key == self._obj_key:
            return
        ox, oy = self.objective
        self.obj_field = horde.flood(sim.gmap, bots.tile_ok(sim), fp,
                                     [(oy, ox)])
        self._obj_key = key


def descend(field, x: int, y: int, steps: int) -> tuple:
    """The anchor ``steps`` 4-connected steps down ``field`` from ``(x, y)``
    (each step to the lowest neighbour strictly below, ties north / west /
    east / south), or where the descent stops: the Bound mover's leap."""
    h, w = field.shape
    for _ in range(steps):
        here = int(field[y, x])
        best = None
        for dy, dx in horde.NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                v = int(field[ny, nx])
                if v < here and (best is None or v < best[0]):
                    best = (v, nx, ny)
        if best is None:
            break
        x, y = best[1], best[2]
    return (x, y)


def focus_key(e, centroid) -> tuple:
    """The focus-fire order (§2.4): most EXPOSED first (the exposure law's
    kinetic level, G4p), then the weakest (lowest hit points), then the
    nearest to ``centroid``, then the lower id."""
    return (-int(e.exposure[KINETIC]), float(e.current_hp),
            _d2(_centre(e), centroid), int(e.id))


# ===========================================================================
# The position layer
# ===========================================================================
@dataclass(frozen=True)
class Feature:
    """A position feature (§2.2): a yes / no question about each candidate
    anchor. ``fn(ctx) -> bool array`` over ``ctx``'s candidates (or the
    subset ``idx``). ``los``: it casts lines of sight, so it is evaluated
    only on the ``los_budget`` best candidates of the cheap features."""
    name: str
    fn: object
    los: bool = False


@dataclass
class PositionContext:
    """What the position layer needs about one bot's decision."""
    sim: object
    unit: object
    team: object
    bots: object
    xs: np.ndarray            # candidate anchors (top-left x)
    ys: np.ndarray
    cx: np.ndarray            # candidate centres
    cy: np.ndarray
    here: tuple               # the bot's own anchor (x, y)
    reach: np.ndarray         # flood from the bot (anchor grid)
    objective_field: object   # flood from the play's objective, or None
    target: object            # centre (x, y) the bot should see, or None
    others: list = field(default_factory=list)   # other enemies' centres
    mates: list = field(default_factory=list)    # teammates' centres + goals
    threat: object = None     # centre (x, y) the threat comes from


def _f_sees_target(ctx, idx):
    if ctx.target is None:
        return np.zeros(len(idx), dtype=bool)
    t = ctx.target
    return np.array([los(ctx.sim, (int(ctx.cx[i]), int(ctx.cy[i])), t)
                     for i in idx], dtype=bool)


def _f_outside_fire_lanes(ctx, idx):
    grid = ctx.team.lanes
    if grid is None:
        return np.ones(len(idx), dtype=bool)
    k = ctx.team.s.lane_cell
    owner = grid[ctx.cy[idx] // k, ctx.cx[idx] // k]
    return (owner == -1) | (owner == ctx.unit.id)


def _f_spaced(ctx, idx):
    sp = ctx.team.s.spacing_tiles
    ok = np.ones(len(idx), dtype=bool)
    for (mx, my) in ctx.mates:
        d2 = (ctx.cx[idx] - mx) ** 2 + (ctx.cy[idx] - my) ** 2
        ok &= d2 >= sp * sp
    return ok


def _f_cover(ctx, idx):
    """Beside a corner or behind cover relative to the threat: at least
    ``cover_min_tiles`` solid tiles in the ring around the body on the
    threat's side (a positive dot product with the direction to it)."""
    if ctx.threat is None:
        return np.zeros(len(idx), dtype=bool)
    fp = ctx.unit.footprint
    solid = ctx.sim.gmap.solid
    h, w = solid.shape
    ring = ctx.bots.ring_offsets(fp)            # (R, 2) dx, dy from anchor
    ax = ctx.xs[idx][:, None] + ring[None, :, 0]
    ay = ctx.ys[idx][:, None] + ring[None, :, 1]
    inb = (ax >= 0) & (ax < w) & (ay >= 0) & (ay < h)
    hit = np.zeros(ax.shape, dtype=bool)
    hit[inb] = solid[ay[inb], ax[inb]]
    tx, ty = ctx.threat
    vx = (ax - ctx.cx[idx][:, None])
    vy = (ay - ctx.cy[idx][:, None])
    dx = (tx - ctx.cx[idx])[:, None]
    dy = (ty - ctx.cy[idx])[:, None]
    facing = vx * dx + vy * dy > 0
    return (hit & facing).sum(axis=1) >= ctx.team.s.cover_min_tiles


def _f_hidden_from_others(ctx, idx):
    if not ctx.others:
        return np.ones(len(idx), dtype=bool)
    out = np.ones(len(idx), dtype=bool)
    for j, i in enumerate(idx):
        c = (int(ctx.cx[i]), int(ctx.cy[i]))
        out[j] = not any(los(ctx.sim, o, c) for o in ctx.others)
    return out


def _f_closer_to_objective(ctx, idx):
    f = ctx.objective_field
    if f is None:
        return np.zeros(len(idx), dtype=bool)
    hx, hy = ctx.here
    here = int(f[hy, hx])
    v = f[ctx.ys[idx], ctx.xs[idx]].astype(np.int64)
    if here == horde.UNREACHED:
        return v != horde.UNREACHED
    return (v != horde.UNREACHED) & (
        v + ctx.team.s.progress_steps <= here)


def _f_safe_ground(ctx, idx):
    """Not in fire, dense smoke or vacuum (nor on a hot tile), read at the
    body's centre tile through the field accessor (the sensors' seam)."""
    acc = ctx.bots.accessor(ctx.sim)
    ys, xs = ctx.cy[idx], ctx.cx[idx]
    s = ctx.team.s
    fire = acc.sample_many(Channel.FIRE, ys, xs)
    smoke = acc.sample_many(Channel.SMOKE, ys, xs)
    pres = acc.sample_many(Channel.PRESSURE, ys, xs)
    temp = acc.sample_many(Channel.TEMPERATURE, ys, xs)
    return ((fire <= 0) & (smoke < ctx.bots.smoke_q)
            & (pres >= ctx.bots.vacuum_q) & (temp < ctx.bots.hot_q))


def _f_reachable(ctx, idx):
    v = ctx.reach[ctx.ys[idx], ctx.xs[idx]].astype(np.int64)
    return v <= ctx.team.s.reach_steps


#: The position features (§2.2), each a yes / no question about a tile.
FEATURES = (
    Feature("sees_target", _f_sees_target, los=True),
    Feature("outside_fire_lanes", _f_outside_fire_lanes),
    Feature("spaced", _f_spaced),
    Feature("cover", _f_cover),
    Feature("hidden_from_others", _f_hidden_from_others, los=True),
    Feature("closer_to_objective", _f_closer_to_objective),
    Feature("safe_ground", _f_safe_ground),
    Feature("reachable", _f_reachable),
)


def evaluate_features(ctx, idx=None) -> dict:
    """Every feature's yes / no answer on the candidates ``idx`` (default:
    all): ``{name: bool array}`` -- the interface a learned scorer reads."""
    if idx is None:
        idx = np.arange(len(ctx.xs))
    return {f.name: f.fn(ctx, idx) for f in FEATURES}


def choose_position(ctx, current_goal=None):
    """The best candidate anchor ``(x, y)`` by the weighted features, or
    None with no candidate. The cheap features over every candidate; the
    line-of-sight features only on the ``los_budget`` best by that cheap
    score (the budget: a candidate outside it is never cast for); exact
    ties by a hash of (seed, id, tick, candidate), then candidate order.
    The bot's committed goal (always a candidate) gets ``commit_margin``
    (#138), so a decision moves it only for a clearly better tile."""
    n = len(ctx.xs)
    if n == 0:
        return None
    s = ctx.team.s
    W = s.features
    idx = np.arange(n)
    score = np.zeros(n, dtype=np.int64)
    for f in FEATURES:
        if not f.los:
            score += np.where(f.fn(ctx, idx), W[f.name], 0)
    if current_goal is not None:
        cur = (ctx.xs == current_goal[0]) & (ctx.ys == current_goal[1])
        score += np.where(cur, int(round(s.commit_margin * FP_ONE)), 0)
    score = score * (1 << 16) + tie_hash(
        ctx.bots.seed, ctx.unit.id, ctx.sim.tick, n)
    order = np.lexsort((idx, -score))
    top = order[:s.los_budget]
    final = score[top] + 0
    for f in FEATURES:
        if f.los and W[f.name] != 0:
            final += np.where(f.fn(ctx, top), W[f.name] << 16, 0)
    j = int(np.lexsort((top, -final))[0])
    i = int(top[j])
    return (int(ctx.xs[i]), int(ctx.ys[i]))


# ===========================================================================
# The unit layer: scored actions
# ===========================================================================
@dataclass(frozen=True)
class Action:
    """A bot action (§2.1). ``score(bots, uctx) -> payload or None`` (None
    = not possible now); its ``[bots.actions]`` weight is its score and the
    best wins its ``channels``. ``start(bots, uctx, st, payload)`` runs once
    when the action is chosen (a latched verb is issued there);
    ``each_tick(bots, uctx, st, payload)`` runs every tick until the next decision (a hands action aims and pulls the
    trigger there). ``look``: a feet action's facing when the hands do not
    aim -- ``"walk"`` (along the move) or ``"threat"`` (at the nearest known
    enemy: the backward walk of a retreat)."""
    name: str
    channels: tuple
    score: object
    start: object = None
    each_tick: object = None
    look: str = "walk"


@dataclass
class UnitContext:
    """What one bot's decision reads."""
    sim: object
    unit: object
    team: object
    mine: list
    seen: list                # enemies this bot sees (id order)
    tick: int


# -- the hands ---------------------------------------------------------------
def _score_shoot(bots, c):
    return bots.fire_target(c.sim, c.unit, c.mine, c.seen, c.team)


def _tick_shoot(bots, c, st, payload):
    """Aim at this tick's fire target (the focus order; it may differ from
    the decision's) and pull the trigger once on target."""
    return bots.shoot(c)


def _score_hold_fire(bots, c):
    return True


def _tick_hold_fire(bots, c, st, payload):
    c.sim.set_trigger(c.unit.id, False)
    return None


def _start_throw(kind_attr):
    def start(bots, c, st, payload):
        d, reach_q = payload
        c.sim.throw_grenade_intent(c.unit.id, d[0], d[1], bots.fuse_ticks,
                                   reach_q=reach_q,
                                   kind=getattr(bots, kind_attr))
        st.last_throw_tick = c.tick
        st.throw_aim = d
    return start


def _tick_throw(bots, c, st, payload):
    """Through the wind-up the grenade leaves along the facing the bot has
    THEN (G3p): hold the aim on the lob and the trigger released; once it
    left, shoot as usual."""
    if st.throw_aim is not None and c.unit.throw_release_tick >= 0:
        bots.aim(c.sim, c.unit, st.throw_aim)
        c.sim.set_trigger(c.unit.id, False)
        return None
    return bots.shoot(c)


def _score_frag(bots, c):
    return bots.frag_point(c)


def _score_flash(bots, c):
    return bots.flash_point(c)


def _score_adjacent(bots, c):
    return bots.adjacent_enemy(c)


def _score_melee(bots, c):
    if not getattr(c.unit, "melee_id", ""):
        return None
    return bots.adjacent_enemy(c)


def _tick_close(verb):
    def each_tick(bots, c, st, payload):
        """Face the adjacent enemy and use ``verb`` whenever it is ready and
        would reach it (a whiff costs nothing, but a shove may reach a
        teammate: it is issued only onto an enemy); shoot it point-blank
        when on target."""
        if payload is None or not payload.alive:
            return bots.shoot(c)
        bots.aim_at(c.sim, c.unit, payload)
        getattr(bots, "try_" + verb)(c.sim, c.unit, c.tick)
        return bots.shoot(c, only=payload)
    return each_tick


# -- the feet ----------------------------------------------------------------
def _score_move(bots, c):
    return c.team.play != "fall_back" and not bots.is_holder(c)


def _score_retreat(bots, c):
    return c.team.play == "fall_back" or None


def _score_stand(bots, c):
    return bots.is_holder(c) or None


def _start_position(bots, c, st, payload):
    st.goal = bots.position(c, st)


def _start_stand(bots, c, st, payload):
    st.goal = (c.unit.tile_x, c.unit.tile_y)


#: The actions (§2.1). A new behaviour is a new row (and its weight).
ACTIONS = (
    Action("shoot", ("hands",), _score_shoot, each_tick=_tick_shoot),
    Action("hold_fire", ("hands",), _score_hold_fire,
           each_tick=_tick_hold_fire),
    Action("frag", ("hands",), _score_frag, start=_start_throw("frag_kind"),
           each_tick=_tick_throw),
    Action("flash", ("hands",), _score_flash,
           start=_start_throw("flash_kind"), each_tick=_tick_throw),
    Action("shove", ("hands",), _score_adjacent,
           each_tick=_tick_close("shove")),
    Action("melee", ("hands",), _score_melee,
           each_tick=_tick_close("melee")),
    Action("move", ("feet",), _score_move, start=_start_position,
           look="walk"),
    Action("retreat", ("feet",), _score_retreat, start=_start_position,
           look="threat"),
    Action("stand", ("feet",), _score_stand, start=_start_stand,
           look="threat"),
)
ACTION_BY_NAME = {a.name: a for a in ACTIONS}


def resolve_actions(bots, c) -> dict:
    """Score every action and resolve: the best first, then the best whose
    channels are free (ties: row order). ``{channel: (name, payload)}``."""
    W = bots.s.actions
    scored = []
    for k, a in enumerate(ACTIONS):
        p = a.score(bots, c)
        if p is None or p is False:
            continue
        scored.append((-W[a.name], k, a, p))
    scored.sort(key=lambda r: (r[0], r[1]))
    taken = {}
    for _w, _k, a, p in scored:
        if any(ch in taken for ch in a.channels):
            continue
        for ch in a.channels:
            taken[ch] = (a.name, p)
    return taken


# ===========================================================================
# The controller
# ===========================================================================
class _BotState:
    """One bot's controller-side memory (never on the Unit: not synced)."""
    __slots__ = ("hands", "feet", "goal", "path", "idx", "path_goal",
                 "replan_at", "decided_at", "seen_ids", "last_throw_tick",
                 "throw_aim", "door_used", "aim")

    def __init__(self):
        self.hands = ("hold_fire", True)
        self.feet = ("stand", True)
        self.goal = None
        self.path = None
        self.idx = 0
        self.path_goal = None
        self.replan_at = 0
        self.decided_at = None
        self.seen_ids = ()
        self.last_throw_tick = None
        self.throw_aim = None
        self.door_used = {}
        self.aim = None


class BotTeam(ControlSource):
    """Drive every living marine of ``team`` through the three layers."""

    def __init__(self, team: int, settings: BotSettings | None = None):
        self.team = int(team)
        self.s = bot_settings(CFG) if settings is None else settings
        tps = float(CFG.clock.ticks_per_second)
        self.tps = tps
        self.brain = TeamBrain(team, self.s, tps)
        self.cooldown_ticks = max(1, int(round(
            self.s.grenade_cooldown_s * tps)))
        tables = weapon_tables()
        grenade = tables.weapons.by_name["hand_grenade"]
        self.fuse_ticks = fuse_ticks_from_seconds(
            grenade.fuse_default_seconds, tps)
        rows = list(tables.hand_grenade_rows)
        self.frag_kind = rows.index(FRAG_ROW)
        self.flash_kind = rows.index(FLASH_ROW) if FLASH_ROW in rows else None
        flash = None
        if self.flash_kind is not None:
            flash = tables.payloads.by_name.get(
                tables.ammo.by_name[FLASH_ROW].payload)
        self.blind_r = int(np.ceil(flash.blind_radius)) if flash else 0
        self.throw_range = int(grenade.max_throw_range)
        self.seed = 0
        self._state: dict = {}
        self._last_tick = None
        self._grid = None          # (key, tile_ok, block grid, door tiles)
        self._acc = None
        self._aimed = False
        self._planned_tick = None
        self._rings = {}
        self._spread_q = {}
        self._weapons = {}
        self.smoke_q = int(round(self.s.smoke_dense * FP_ONE))
        self.vacuum_q = int(round(self.s.vacuum_atm * FP_ONE))
        self.hot_q = int(round(self.s.hot_temp * FP_ONE))

    # ------------------------------------------------------------------
    def handle_frame(self, sim, renderer=None) -> None:
        tick = int(sim.tick)
        if tick == self._last_tick:
            return
        self._last_tick = tick
        self.seed = int(getattr(sim, "_seed", 0) or 0)
        mine = [u for u in sim.units
                if u.team == self.team and u.alive and not u.is_zombie]
        if not mine:
            return
        vis = vision.state_for(sim)
        enemies = [u for u in sim.units if u.alive and u.team != self.team]
        brain = self.brain
        brain.observe(sim)
        if (brain.updated_at is None
                or tick - brain.updated_at >= self.s.team_period_ticks):
            brain.update(self, sim, mine, vis)
        for u in mine:
            st = self._state.setdefault(u.id, _BotState())
            seen = [e for e in enemies if u.id in vis.observers_of(e.id)]
            c = UnitContext(sim, u, brain, mine, seen, tick)
            if self._due(u, st, c):
                self._decide(c, st)
            self._execute(c, st)

    # -- when a bot decides ---------------------------------------------
    def _due(self, u, st, c) -> bool:
        if st.decided_at is None:
            return True
        if u.throw_release_tick >= 0:
            return False                # a throw's wind-up runs to its end
        if (c.tick + u.id * STAGGER) % self.s.unit_period_ticks == 0:
            return True
        ids = tuple(e.id for e in c.seen)
        if any(i not in st.seen_ids for i in ids):
            return True                 # something new in sight
        if c.tick - st.decided_at >= 2 and self.adjacent_enemy(c) is not None \
                and st.hands[0] not in ("shove", "melee"):
            return True                 # contact
        return False

    def _decide(self, c, st) -> None:
        st.decided_at = c.tick
        st.seen_ids = tuple(e.id for e in c.seen)
        taken = resolve_actions(self, c)
        st.hands = taken.get("hands", ("hold_fire", True))
        st.feet = taken.get("feet", ("stand", True))
        for name, payload in (st.hands, st.feet):
            act = ACTION_BY_NAME[name]
            if act.start is not None:
                act.start(self, c, st, payload)

    # -- the position layer's call ---------------------------------------
    def position(self, c, st):
        u = c.unit
        field = self.objective_field_for(c)
        ahead = None
        if field is not None:
            ahead = descend(field, u.tile_x, u.tile_y, self.s.advance_steps)
        xs, ys = self.candidates(c.sim, u, st.goal, ahead)
        if len(xs) == 0:
            return (u.tile_x, u.tile_y)
        ctx = self.position_context(c, xs, ys)
        best = choose_position(ctx, st.goal)
        return (u.tile_x, u.tile_y) if best is None else best

    def candidates(self, sim, u, goal=None, ahead=None):
        """The position layer's candidate anchors for ``u``: the passable
        anchors on a ``candidate_stride`` lattice through its own anchor
        within ``candidate_radius`` tiles, plus its own anchor, its
        committed ``goal`` and the anchor ``ahead`` (``advance_steps`` down
        the play's objective field). Two int64 arrays ``(xs, ys)``."""
        s = self.s
        grid = self.block_grid(sim, u.footprint)
        ax, ay = u.tile_x, u.tile_y
        R, k = s.candidate_radius, s.candidate_stride
        h, w = grid.shape
        ys = np.arange(ay - (R // k) * k, ay + R + 1, k)
        xs = np.arange(ax - (R // k) * k, ax + R + 1, k)
        ys = ys[(ys >= 0) & (ys < h)]
        xs = xs[(xs >= 0) & (xs < w)]
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        gy, gx = gy.ravel(), gx.ravel()
        keep = grid[gy, gx]
        gy, gx = gy[keep], gx[keep]
        extra = [(ax, ay)]
        for p in (goal, ahead):
            if p is not None and p not in extra:
                extra.append(p)
        for (ex, ey) in extra:
            if 0 <= ey < h and 0 <= ex < w and grid[ey, ex] and not np.any(
                    (gx == ex) & (gy == ey)):
                gx = np.append(gx, ex)
                gy = np.append(gy, ey)
        return gx.astype(np.int64), gy.astype(np.int64)

    def position_context(self, c, xs, ys) -> PositionContext:
        """The :class:`PositionContext` of bot ``c.unit`` over the candidate
        anchors ``(xs, ys)``: its reach (a flood from its anchor), the play's
        objective field, the target it should see (the shared target, else
        the nearest known enemy marine), the two nearest other known enemies,
        its teammates' centres and committed goals, the threat's side."""
        sim, u, team = c.sim, c.unit, c.team
        fp = u.footprint
        ax, ay = u.tile_x, u.tile_y
        xs = np.asarray(xs, dtype=np.int64)
        ys = np.asarray(ys, dtype=np.int64)
        reach = horde.flood(sim.gmap, self.tile_ok(sim), fp, [(ay, ax)])
        uc = _centre(u)
        target = None
        tgt = team.target
        if tgt is not None and tgt.id in team.picture:
            target = team.picture[tgt.id].centre
        else:
            known = team.known(zombies=False)
            if known:
                target = min(known, key=lambda sg: (_d2(sg.centre, uc),
                                                    sg.uid)).centre
        others = sorted(
            (sg for sg in team.known()
             if target is None or sg.centre != target),
            key=lambda sg: (_d2(sg.centre, uc), sg.uid))[:2]
        mates = []
        for m in c.mine:
            if m is u:
                continue
            mates.append(_centre(m))
            mst = self._state.get(m.id)
            if mst is not None and mst.goal is not None:
                mates.append((mst.goal[0] + m.footprint // 2,
                              mst.goal[1] + m.footprint // 2))
        return PositionContext(
            sim=sim, unit=u, team=team, bots=self, xs=xs, ys=ys,
            cx=xs + fp // 2, cy=ys + fp // 2, here=(ax, ay), reach=reach,
            objective_field=self.objective_field_for(c),
            target=target, others=[o.centre for o in others], mates=mates,
            threat=self.threat_of(c))

    def objective_field_for(self, c):
        """The play's objective for this bot: the team's (a Bound holder and
        a standing bot have none)."""
        if self.is_holder(c):
            return None
        return c.team.obj_field

    def threat_of(self, c):
        """Where this bot's threat comes from: the nearest known enemy's
        centre, else the team's objective."""
        uc = _centre(c.unit)
        known = c.team.known()
        if known:
            return min(known, key=lambda sg: (_d2(sg.centre, uc),
                                              sg.uid)).centre
        o = c.team.objective
        if o is None:
            return None
        fp = c.unit.footprint
        return (o[0] + fp // 2, o[1] + fp // 2)

    # -- the executor: every tick ----------------------------------------
    def _execute(self, c, st):
        """Every tick: the feet follow their path, then the hands action
        runs its ``each_tick``; when the hands aimed nowhere this tick the
        bot looks along its walk or at the threat (its feet action's
        ``look``)."""
        sim, u = c.sim, c.unit
        move_dir = self._follow(sim, u, st, c)
        self._aimed = False
        name, payload = st.hands
        act = ACTION_BY_NAME[name]
        if act.each_tick is not None:
            act.each_tick(self, c, st, payload)
        if not self._aimed:
            look = None
            if ACTION_BY_NAME[st.feet[0]].look == "threat":
                th = self.threat_of(c)
                if th is not None:
                    uc = _centre(u)
                    look = _direction_q(th[0] - uc[0], th[1] - uc[1])
            if look is None:
                look = move_dir
            if look is not None:
                sim.set_aim(u.id, *look)

    def shoot(self, c, only=None):
        """This tick's shot: aim at the fire target (``only``: that unit or
        nothing) and pull the trigger when on target. Returns the target or
        None (the trigger released)."""
        sim, u = c.sim, c.unit
        t = self.fire_target(sim, u, c.mine, c.seen, c.team)
        if t is None or (only is not None and t is not only):
            sim.set_trigger(u.id, False)
            return None
        aim = self.aim_at(sim, u, t)
        fire = aim is None or self._on_target(sim, u, aim)
        sim.set_trigger(u.id, bool(fire))
        return t

    def _dir_to(self, u, t):
        (tx, ty), (ux, uy) = _centre(t), _centre(u)
        return _direction_q(tx - ux, ty - uy)

    def aim(self, sim, u, d) -> None:
        """Hold the aim ``d`` (a Q16.16 vector) this tick."""
        sim.set_aim(u.id, *d)
        self._aimed = True

    def aim_at(self, sim, u, t):
        d = self._dir_to(u, t)
        if d is not None:
            self.aim(sim, u, d)
        return d

    def try_shove(self, sim, u, tick) -> None:
        last = int(getattr(u, "last_shove_tick", -1))
        if last >= 0 and tick - last < sim.impulse_table.shove_cooldown_ticks:
            return
        t = shove_target(sim.impulse_table, sim.units, u)
        if t is not None and t.team != u.team:
            sim.shove_intent(u.id)

    def try_melee(self, sim, u, tick) -> None:
        mid = getattr(u, "melee_id", "")
        if not mid:
            return
        weapon = weapon_tables().weapons.by_name[mid]
        if (u.last_blow_tick >= 0
                and tick - u.last_blow_tick < weapon.rof_interval_ticks):
            return
        if melee_arc_target(sim.units, u, weapon) is not None:
            sim.melee_intent(u.id)

    # -- shooting --------------------------------------------------------
    def weapon_of(self, u):
        wid = getattr(u, "weapon_id", "")
        if not wid:
            return None
        w = self._weapons.get(wid)
        if w is None:
            w = weapon_tables().weapons.by_name[wid]
            self._weapons[wid] = w
        return w

    def fire_target(self, sim, u, mine, seen, team):
        """Whom ``u`` shoots now: the team's shared target if ``u`` sees it,
        else its own best by the focus order -- in range, a zombie only
        within ``zombie_engage_tiles`` (a far zombie is left alone: every shot
        is heard), never through a teammate on the line."""
        weapon = self.weapon_of(u)
        if weapon is None or not seen:
            return None
        rng = int(weapon.range_tiles)
        zr = self.s.zombie_engage_tiles
        uc = _centre(u)
        ok = []
        for e in seen:
            d2 = _d2(uc, _centre(e))
            if d2 > rng * rng:
                continue
            if e.is_zombie and d2 > zr * zr:
                continue
            ok.append(e)
        if not ok:
            return None
        cand = sorted(ok, key=lambda e: focus_key(e, team.centroid))
        if team.target is not None and team.target in ok:
            cand.remove(team.target)
            cand.insert(0, team.target)
        for e in cand:
            if not self._teammate_on_line(u, e, mine):
                return e
        return None

    def _on_target(self, sim, u, aim) -> bool:
        """The bots' trigger discipline (G1p): fire only when the facing
        after this tick's turn (``Simulation.facing_after_turn_q16``) is
        within the weapon's ``spread_deg`` of the aim."""
        after = sim.facing_after_turn_q16(u, aim[0], aim[1])
        want = atan2_q16(-aim[1], aim[0])
        return abs(wrap_q16(want - after)) <= self._weapon_spread_q(u)

    def _weapon_spread_q(self, u) -> int:
        import math
        weapon = self.weapon_of(u)
        key = weapon.name if weapon is not None else ""
        q = self._spread_q.get(key)
        if q is None:
            deg = float(weapon.spread_deg) if weapon is not None else 0.0
            q = quantize_scalar(math.radians(deg))
            self._spread_q[key] = q
        return q

    def _teammate_on_line(self, u, target, mine) -> bool:
        """Is a teammate's body near the segment shooter -> target, anywhere
        along it (a round flies on past the first bots' 12 tiles)? Integer
        geometry: the projection ahead of the shooter and not past the
        target, the perpendicular distance within half a body plus
        ``line_clear_tiles`` (the carbine's spread, a moving mate)."""
        sx, sy = _centre(u)
        tx, ty = _centre(target)
        dx, dy = tx - sx, ty - sy
        dd = dx * dx + dy * dy
        if dd == 0:
            return False
        for m in mine:
            if m is u:
                continue
            mx, my = _centre(m)
            vx, vy = mx - sx, my - sy
            along = vx * dx + vy * dy
            if along <= 0 or along > dd:
                continue
            cross = vx * dy - vy * dx
            cr = m.footprint // 2 + self.s.line_clear_tiles
            if cross * cross <= cr * cr * dd:
                return True
        return False

    # -- throws ----------------------------------------------------------
    def _can_throw(self, c, kind) -> bool:
        u = c.unit
        if kind is None or u.grenade_counts[kind] <= 0:
            return False
        if u.throw_release_tick >= 0:
            return False
        st = self._state.get(u.id)
        last = None if st is None else st.last_throw_tick
        return last is None or c.tick - last >= self.cooldown_ticks

    def _lob(self, uc, point):
        d = _direction_q(point[0] - uc[0], point[1] - uc[1])
        if d is None:
            return None
        reach_q = quantize_scalar(float(_d2(uc, point)) ** 0.5)
        return (d, reach_q)

    def frag_point(self, c):
        """A frag at the centre of the largest group of enemies the team
        sees (zombies count: the blast's heat is the horde's answer), at
        least ``frag_min_group`` within ``group_radius`` of each other,
        farther than ``danger_radius`` from every teammate (the thrower
        included), within the throw's range. ``(dir, reach_q)`` or None."""
        if not self._can_throw(c, self.frag_kind):
            return None
        s = self.s
        team = c.team
        u = c.unit
        uc = _centre(u)
        seen_ids = set(vision.state_for(c.sim).visible_by_team.get(
            self.team, ()))
        pts = [sg for sg in team.known() if sg.uid in seen_ids]
        if len(pts) < s.frag_min_group:
            return None
        gr2 = s.group_radius * s.group_radius
        dr2 = s.danger_radius * s.danger_radius
        tr2 = self.throw_range * self.throw_range
        best = None
        for sg in pts:
            group = [o for o in pts if _d2(sg.centre, o.centre) <= gr2]
            if len(group) < s.frag_min_group:
                continue
            gx = sum(o.centre[0] for o in group) // len(group)
            gy = sum(o.centre[1] for o in group) // len(group)
            p = (gx, gy)
            d2u = _d2(uc, p)
            if d2u <= dr2 or d2u > tr2:
                continue
            if any(_d2(_centre(m), p) <= dr2 for m in c.mine):
                continue
            if best is None or len(group) > best[0]:
                best = (len(group), p)
        if best is None:
            return None
        return self._lob(uc, best[1])

    def flash_point(self, c):
        """A flashbang onto a known enemy marine the team does NOT see now
        (seen within ``flash_memory_s``), while the team bounds onto it --
        never where a teammate (the thrower included) is within the flash's
        blinding radius with a line of sight to it: the flash blinds anyone
        looking, and the thrower faces the throw at the release."""
        if c.team.play != "bound" or not self._can_throw(c, self.flash_kind):
            return None
        u = c.unit
        uc = _centre(u)
        seen_ids = set(vision.state_for(c.sim).visible_by_team.get(
            self.team, ()))
        tr2 = self.throw_range * self.throw_range
        best = None
        for sg in c.team.known(zombies=False):
            if sg.uid in seen_ids:
                continue
            if c.tick - sg.tick > c.team.flash_memory_ticks:
                continue
            p = sg.centre
            if _d2(uc, p) > tr2:
                continue
            if not self.flash_safe(c.sim, c.mine, p):
                continue
            key = (_d2(uc, p), sg.uid)
            if best is None or key < best[0]:
                best = (key, p)
        if best is None:
            return None
        return self._lob(uc, best[1])

    def flash_safe(self, sim, mine, p) -> bool:
        """No teammate within the flash's blinding radius of ``p`` with a
        line of sight to it."""
        r2 = self.blind_r * self.blind_r
        for m in mine:
            mc = _centre(m)
            if _d2(mc, p) <= r2 and los(sim, mc, p):
                return False
        return True

    def adjacent_enemy(self, c):
        """The adjacent enemy (zombies included) nearest this bot, or None
        (centre distance within the bodies' half widths plus one tile -- the
        shove's reach)."""
        u = c.unit
        uc = _centre(u)
        best = None
        for e in c.sim.units:
            if not e.alive or e.team == u.team:
                continue
            r = (u.footprint + e.footprint) // 2 + 1
            d2 = _d2(uc, _centre(e))
            if d2 <= r * r and (best is None or (d2, e.id) < best[0]):
                best = ((d2, e.id), e)
        return None if best is None else best[1]

    def is_holder(self, c) -> bool:
        """A Bound bot of the half that holds and watches."""
        team = c.team
        return (team.play == "bound"
                and team.halves.get(c.unit.id, 0) != team.bound_moving)

    # -- movement --------------------------------------------------------
    @staticmethod
    def _pos_q(u):
        return quantize_scalar(u.x), quantize_scalar(u.y)

    def _at(self, u, tile) -> bool:
        px, py = self._pos_q(u)
        return (abs(px - ((tile[0] << 16) + MID_Q)) <= ARRIVE_TOL_Q
                and abs(py - ((tile[1] << 16) + MID_Q)) <= ARRIVE_TOL_Q)

    def arrived(self, u) -> bool:
        st = self._state.get(u.id)
        if st is None or st.goal is None:
            return True
        return self._at(u, st.goal) or st.path == []

    def _follow(self, sim, u, st, c):
        """Walk toward the committed goal along an A* path (replanned when
        the goal changes, on the cadence, or when used up); returns the
        Q16.16 move direction, or None when the bot stands."""
        goal = st.goal
        if goal is None or self._at(u, goal):
            sim.clear_move_dir(u.id)
            return None
        tick = c.tick
        if (st.path is None or st.path_goal != goal or tick >= st.replan_at
                or st.idx >= len(st.path)):
            if self._planned_tick == tick:
                # ONE path search per team per tick (the cost is staggered):
                # the others keep their path -- or stand -- for a tick
                st.replan_at = tick + 1
                if st.path is None or st.path_goal != goal:
                    sim.clear_move_dir(u.id)
                    return None
            else:
                self._planned_tick = tick
                st.path = self._plan(sim, u, c.mine, goal)
                st.idx = 1
                st.path_goal = goal
                st.replan_at = tick + self.s.replan_ticks
        path = st.path
        while st.idx < len(path) and self._at(u, path[st.idx]):
            st.idx += 1
        if st.idx >= len(path):
            sim.clear_move_dir(u.id)
            return None
        wx, wy = path[st.idx]
        self._open_door(sim, u, st, (wx, wy), tick)
        px, py = self._pos_q(u)
        d = _direction_q((wx << 16) + MID_Q - px, (wy << 16) + MID_Q - py)
        if d is None:
            sim.clear_move_dir(u.id)
            return None
        throttle = 0
        if not c.seen and not c.team.counts.contact:
            throttle = int(round(self.s.travel_throttle * THROTTLE_ONE))
        sim.set_move_dir(u.id, d[0], d[1], throttle)
        return d

    def _plan(self, sim, u, mine, goal) -> list:
        """The A* path to ``goal``: AROUND the standing teammates when that
        way exists (#142: a body never passes through a teammate if it need
        not), else through them, else none. Whether a way exists -- and is
        at most ``plan_max_steps`` long -- is asked of the flood first
        (``horde.flood``, one C++ pass), so A* never runs a search that
        cannot succeed or must wander (a failed search expands the whole
        map: the multi-ms spike); a goal farther than that is no path (the
        bot stands until its next decision chooses again)."""
        fp = u.footprint
        h, w = sim.gmap.material.shape
        tile_ok = self.tile_ok(sim)
        clear_ok = tile_ok.copy()
        for m in mine:
            if m is not u:
                clear_ok[m.tile_y:m.tile_y + m.footprint,
                         m.tile_x:m.tile_x + m.footprint] = False
        for ok in (clear_ok, tile_ok):
            dist = horde.flood(sim.gmap, ok, fp, [(goal[1], goal[0])])
            if int(dist[u.tile_y, u.tile_x]) > self.s.plan_max_steps:
                continue                       # UNREACHED is larger still
            grid = (self.block_grid(sim, fp) if ok is tile_ok
                    else horde.block_passable(ok, fp))
            path = astar(u.tile_x, u.tile_y, goal[0], goal[1], None, w, h,
                         passable=grid)
            if path:
                return path
        return []

    def _open_door(self, sim, u, st, way, tick) -> None:
        """#142: a closed (non-hull) door in the next waypoint's block is
        opened through the USE verb once the bot touches it (USE toggles,
        so only a door not already asked open, and not again within
        ``door_retry_ticks``)."""
        doors = self.door_tiles(sim)
        if doors is None:
            return
        fp = u.footprint
        wx, wy = way
        block = doors[wy:wy + fp, wx:wx + fp]
        if not block.any():
            return
        oy, ox = np.argwhere(block)[0]
        door = sim.door_at(int(wy + oy), int(wx + ox))
        if door is None or getattr(door, "want_open", False):
            return
        key = id(door)
        last = st.door_used.get(key)
        if last is not None and tick - last < self.s.door_retry_ticks:
            return
        # touching: the door tile within the footprint's one-tile ring
        x0, y0 = u.tile_x, u.tile_y
        if not (x0 - 1 <= wx + ox <= x0 + fp and y0 - 1 <= wy + oy <= y0 + fp):
            return
        sim.use_intent(u.id)
        st.door_used[key] = tick

    # -- shared map caches -----------------------------------------------
    def topology_key(self, sim):
        return int(getattr(sim.gmap, "_tile_serial", 0))

    def _grids(self, sim):
        key = self.topology_key(sim)
        if self._grid is not None and self._grid[0] == key:
            return self._grid
        gmap = sim.gmap
        walk = horde.walkable_tiles(gmap)
        drives = getattr(sim, "_door_drives", {}) or {}
        doors = [d for d in getattr(sim, "_doors", ()) or ()
                 if getattr(d, "ordinal", None) not in drives]
        door_tiles = horde.bashable_door_tiles(gmap, doors)
        tile_ok = walk | door_tiles
        self._grid = (key, tile_ok, {}, door_tiles if door_tiles.any()
                      else None)
        return self._grid

    def tile_ok(self, sim):
        """Tiles a bot's body may cover on its way: walkable, or a closed
        door it can open (#142)."""
        return self._grids(sim)[1]

    def door_tiles(self, sim):
        return self._grids(sim)[3]

    def block_grid(self, sim, fp):
        g = self._grids(sim)
        grid = g[2].get(fp)
        if grid is None:
            grid = horde.block_passable(g[1], fp)
            g[2][fp] = grid
        return grid

    def ring_offsets(self, fp) -> np.ndarray:
        r = self._rings.get(fp)
        if r is None:
            pts = [(dx, dy) for dy in range(-1, fp + 1)
                   for dx in range(-1, fp + 1)
                   if dx in (-1, fp) or dy in (-1, fp)]
            r = np.array(pts, dtype=np.int64)
            self._rings[fp] = r
        return r

    def accessor(self, sim):
        if self._acc is None or self._acc[0] is not sim.gmap:
            self._acc = (sim.gmap, EntityFieldAccessor(sim.gmap))
        return self._acc[1]

    # -- goals: spawn rooms ----------------------------------------------
    def _rooms(self, sim) -> dict:
        return {r.team: r for r in getattr(sim.level, "spawn_rooms", ())}

    def enemy_room_goal(self, sim, fp):
        rooms = self._rooms(sim)
        others = sorted(t for t in rooms
                        if t != self.team and t != ZOMBIE_TEAM)
        if not others:
            return None
        return self.room_goal(sim, others[0], fp)

    def room_goal(self, sim, team, fp):
        """The top-left tile of a body centred in ``team``'s spawn room, or
        the room's slot-0 formation cell when the centre is blocked."""
        room = self._rooms(sim).get(team)
        if room is None:
            return None
        gx = room.x + room.w // 2 - fp // 2
        gy = room.y + room.h // 2 - fp // 2
        if sim.gmap.is_passable_block(gy, gx, fp):
            return (gx, gy)
        from simulation.match import placement_cells
        return tuple(placement_cells(room, 1)[0])


class BotCommander(BotTeam):
    """The COMMANDER (#195 1f, design 5.3): the bots' TEAM layer alone,
    for a team driven through the door (a policy's). Once per tick it reads
    the team picture and, every ``team_period_ticks``, takes the team
    layer's decision (play, shared target, objective; no unit layer, no
    ``_BotState``, so ``arrived`` is True for every unit and the Bound
    halves swap on their timers alone) -- and translates it into ONE order
    for every living marine of the team, issued through
    ``Simulation.order_intent`` only when it differs from what the unit
    holds: what = ``PLAY_ORDER[play]``, where = ``room_graph.place_at`` of
    the objective body's CENTRE tile, until told otherwise, the shared
    target's id (or none). An objective on no room and no opening (or no
    objective) clears the order. Draws nothing, writes nothing else, calls
    no other verb. Registered as ``commander`` (not nameable); built by
    ``control_source.create_match_controllers``."""

    def handle_frame(self, sim, renderer=None) -> None:
        tick = int(sim.tick)
        if tick == self._last_tick:
            return
        self._last_tick = tick
        self.seed = int(getattr(sim, "_seed", 0) or 0)
        mine = [u for u in sim.units
                if u.team == self.team and u.alive and not u.is_zombie]
        if not mine:
            return
        brain = self.brain
        brain.observe(sim)
        if (brain.updated_at is None
                or tick - brain.updated_at >= self.s.team_period_ticks):
            brain.update(self, sim, mine, vision.state_for(sim))
            self._issue(sim, mine)

    def order_for(self, sim, mine) -> tuple:
        """The team's order now (:func:`team_order` of this brain)."""
        return team_order(self.brain, sim, mine, self.team)

    def _issue(self, sim, mine) -> None:
        issue_team_order(sim, mine, self.order_for(sim, mine))


def team_order(brain, sim, mine, team) -> tuple:
    """The order a team's brain gives now, as the four held fields
    ``(what, where, until, target)`` (``unit_orders.NO_ORDER`` when the
    objective names no place): what = ``PLAY_ORDER[play]``, where =
    ``room_graph.place_at`` of the objective body's centre tile, until told
    otherwise, the shared target's id. The commander's rule, lifted (#195
    1j) so the door bots' own brain writes its team's orders by the same
    function."""
    graph = unit_orders.graph_of(sim)
    if brain.objective is None or graph is None:
        return unit_orders.NO_ORDER
    fp = mine[0].footprint
    ox, oy = brain.objective
    where = place_at(graph, oy + fp // 2, ox + fp // 2)
    if where == unit_orders.NO_PLACE:
        return unit_orders.NO_ORDER
    what = unit_orders.ORDER_WHAT.index(PLAY_ORDER[brain.play])
    t = brain.target
    target = (int(t.id) if t is not None and t.alive
              and t.team != team else unit_orders.NO_TARGET)
    return (what, where, unit_orders.NO_LIMIT, target)


def issue_team_order(sim, mine, want) -> int:
    """Issue ``want`` (:func:`team_order`) through ``Simulation.
    order_intent`` to every unit of ``mine`` (id order) that does not hold
    it already -- issue on change only. Returns the calls made."""
    what, where, _until, target = want
    args = ((0, unit_orders.NO_PLACE, 0, unit_orders.NO_TARGET)
            if what == unit_orders.ORDER_NONE else (what, where, 0,
                                                    target))
    n = 0
    for u in sorted(mine, key=lambda u: u.id):
        if unit_orders.held(u) != want:
            sim.order_intent(u.id, *args)
            n += 1
    return n


__all__ = ["BotTeam", "BotCommander", "PLAY_ORDER", "check_play_order",
           "team_order", "issue_team_order",
           "BotSettings", "bot_settings", "TeamBrain", "FEATURES",
           "ACTIONS", "PLAYS", "Feature", "Action", "Play",
           "PositionContext", "evaluate_features", "choose_position",
           "resolve_actions", "focus_key", "tie_hash"]
