"""simulation/obs_spec_v1.py -- observation spec ``v1`` as code (#195 1g).

THE contract is ``docs/rl_agent_obs_spec_v1_2026-11-12.md`` (document
version 2): every feature a trained network reads is a row of the tables
below -- its position, its name, the seam it is read from, the table value
that divides it, whether the one squash is applied -- and is a RATIO of a
table value, an angle's cosine and sine, a flag, or a kind (a small integer
the network embeds). Never a game number. ``SPEC_NAME = "v1"``.

Patch (a) built the soldier's VECTOR (section 1), the LIST (section 2: the
five blocks, the 67-column entry) and served them through the sensor
accessor's ``observe`` door into the env's buffers (section 4). Patch (b)
builds THE TWO WINDOWS (section 3): the eight physics planes read ONCE per
env per decision for the whole map (:class:`FieldPlanes`) and reduced to
blocks for the far window (the MAX over a block's tiles, the coupling rows'
own reduction; the wind by the block's largest load and that tile's
direction), the structure planes, the sight planes (``watched`` from the
RECORDED centres through the many-tiles routine ``vision.window_clear``),
the lanes plane (the picture's ``fire_lanes``), the far unit planes from the
WHOLE picture, the order's ``route`` (the horde's flood) and ``place``
planes -- and with them the vector's ``air`` block (cols 58-65, the near
window's values at the soldier's centre tile) and ``felt_temp`` (col 96,
``exchange.felt_temperature``, the heat row's own law) and the mate
entries' lane pair (cols 52-53). Cols 28 and 66 read 0 until 1h.

Patch (c) builds THE FULL VIEW (section 5), ``FULL_VIEW_NAME = "full_v1"``:
what the training-only VALUE network is shown beside the observation, served
by the same accessor door into the farm's own buffers under
``info["full_view"]`` (its terminal twin ``info["final_full_view"]``), NEVER
a key of ``obs``. Per row: ``full_list`` -- up to :data:`FULL_ENTRIES`
entries of :data:`FULL_F` columns, the soldier itself first (a dead row: the
corpse of 5.6, ``type`` 0, ``hp`` 0, ``age``), then EVERY other living unit
of every side where it truly is (the marines in (team, slot) order, then the
zombies nearest first), every live grenade and charge, every living door;
the first 67 columns 2.2's entry built with the same helpers against the
observing soldier, then :data:`FULL_EXTRA_COLUMNS` (the team's ``knowledge``
of the entry, its kit, four statuses, its order) --, ``full_mask``,
``full_uid``, ``full_match`` (:func:`full_match_columns`: per team of the
run its living count and health balance, the horde awake / dormant, the
clock) and ``full_watched`` (3.5's plane from EVERY living enemy's LIVE
centre). The full-view builders take the sim's state through ONE
:class:`FullView` per env per decision (built by :meth:`FullView.build`, the
other side of the wall) and keep THEIR OWN caches on it -- never the
observation's (``TeamKnowledge``'s): nothing the full view computes reaches
``obs``, the pair test (f) of spec 7 holds it.

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

from simulation import (atmosphere_fixed, combat, exchange, fire_fixed,
                        gas_fixed, horde, line_of_fire, room_graph, turning,
                        unit_impulse, unit_schema, verb_gates, vision)
from simulation.damage import KINETIC, POISON, mitigation_for
from simulation.entities.door import DOOR_CLOSED, DOOR_OPEN
from simulation.events import CAUSE_NAMES
from simulation.gases import POISON as GAS_POISON, SMOKE as GAS_SMOKE, \
    TEARGAS as GAS_TEARGAS
from simulation.movement import THROTTLE_ONE
from simulation.status import (BURNING, KNOCKED_DOWN, N_STATUS_KINDS,
                               POISONED, STATUS_REGISTRY, STUNNED,
                               composed_flags)
from simulation.team_picture import (OBJECT_UID_BASE, fire_lanes_of,
                                     live_objects, object_tile)
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
#: A count plane (spec 0.3: a uint8 count, the far unit planes) -- a plane
#: kind only, never a vector or entry column.
COUNT = "count"
#: Every table value a column or a plane may be a ratio of (spec sections
#: 1-3); the no-raw-number gate holds every column and plane to this set or
#: to FLAG / KIND / ANGLE / UNIT (a plane also COUNT).
DENOMINATORS = frozenset({
    "bars", "turn_rate", "PI_Q16", "v_ref", "v_fall", "FP_ONE", "react_ticks",
    "STATUS_SCALE_TICKS", "mag_size", "magazines", "reload_ticks",
    "loadout_count", "throw_windup_ticks", "explosives", "plant_ticks",
    "c4_fuse_max_ticks", "rof_interval_ticks", "shove_cooldown_ticks",
    "L_ref", "ref_rof", "ref_spread", "memory_ticks", "limit_ticks", "n_t",
    "team_ammo_total", "enemies_n", "zombies", "d_c", "x_max",
    "HALF_TURN_TICKS", "THROTTLE_ONE", "fuse_max_ticks",
    # the planes' (spec 3.2, 3.4, 3.8; the vector's air block reads them)
    "temperature_max", "fire_full_scale", "poison_min_density",
    "teargas_blind_density", "smoke_dense", "pressure_band", "cap_q",
})

# -- section 3: the two windows (spec 3.1, 3.9) -----------------------------
#: Both windows are ``WIN x WIN`` cells centred on the soldier (tiles near,
#: blocks of ``BLOCK x BLOCK`` tiles far, the block grid anchored at tile
#: (0, 0)); neither rotates.
WIN, BLOCK = 25, 4
HALF = WIN // 2
#: The eight physics planes in their order (spec 3.2), in BOTH windows.
PHYSICS_PLANES = ("temperature", "fire", "poison", "teargas", "smoke",
                  "pressure", "wind_x", "wind_y")
#: ``near_f`` (8, 25, 25) float16 and ``near_u`` (5, 25, 25) uint8.
NEAR_F_PLANES = PHYSICS_PLANES
NEAR_U_PLANES = ("wall", "door", "watched", "sees_target", "lanes")
#: ``far_f`` (10, 25, 25) float16 and ``far_u`` (5, 25, 25) uint8.
FAR_F_PLANES = ("wall_share",) + PHYSICS_PLANES + ("route",)
FAR_U_PLANES = ("door", "mates", "enemies", "records", "place")
#: The door plane's codes (spec 3.4).
DOOR_NONE, DOOR_CODE_OPEN, DOOR_CODE_CLOSED = 0, 1, 2
#: Per plane what divides it (a :data:`DENOMINATORS` name, or a kind) and
#: whether the squash is applied -- the planes' row of the spec's tables,
#: scanned by the no-raw-number gate as the columns are.
PLANE_COLUMNS = MappingProxyType({
    "temperature": ("temperature_max", True), "fire": ("fire_full_scale", True),
    "poison": ("poison_min_density", True),
    "teargas": ("teargas_blind_density", True), "smoke": ("smoke_dense", True),
    "pressure": ("pressure_band", True), "wind_x": ("cap_q", True),
    "wind_y": ("cap_q", True), "wall": (FLAG, False), "door": (KIND, False),
    "watched": (FLAG, False), "sees_target": (FLAG, False),
    "lanes": (FLAG, False), "wall_share": (UNIT, False),
    "route": ("L_ref", True), "mates": (COUNT, False),
    "enemies": (COUNT, False), "records": (COUNT, False),
    "place": (FLAG, False),
})
#: Outside the map (spec 3.1, a CHOICE): ``wall`` 1, ``wall_share`` 1,
#: ``route`` -1, every other plane 0.
OUTSIDE_FILL = MappingProxyType({"wall": 1, "wall_share": 1.0, "route": -1.0})


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
    + _cols(58, [(f"air[{p}]", PLANE_COLUMNS[name][0], True)
                 for p, name in enumerate(PHYSICS_PLANES)])
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
                 ("felt_temp", "temperature_max", True)])
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

# -- section 5: the full view (spec 5.1-5.6) ---------------------------------
FULL_VIEW_NAME = "full_v1"
#: ``full_list`` holds at most this many entries (spec 5.3).
FULL_ENTRIES = 64
#: The ``knowledge`` column's codes (spec 5.3 col 67): my team sees it, has
#: a record of it, does not know of it.
KNOW_SEEN, KNOW_RECORD, KNOW_UNKNOWN = 0, 1, 2
#: The four status kinds the full view's entry carries (spec 5.3 cols
#: 80-83, a CHOICE: the kinds a shipped path applies).
FULL_STATUS_KINDS = (KNOCKED_DOWN, STUNNED, BURNING, POISONED)
#: The 18 columns beyond 2.2's entry (spec 5.3): ``F .. F + 17``.
FULL_EXTRA_COLUMNS = tuple(
    _cols(F, [("knowledge", KIND, False), ("rounds", "mag_size", False),
              ("mags", "magazines", True), ("ammo_untracked", FLAG, False)])
    + _cols(F + 4, [(f"grenade[{k}]", "loadout_count", False)
                    for k in range(HAND_GRENADE_SLOTS)])
    + _cols(F + 10, [("charges", "explosives", False),
                     ("melee_cool", "rof_interval_ticks", False),
                     ("shove_cool", "shove_cooldown_ticks", False)])
    + _cols(F + 13, [(f"status_left[{k}]", "STATUS_SCALE_TICKS", True)
                     for k in FULL_STATUS_KINDS])
    + _cols(F + 17, [("order_what", KIND, False)])
)
#: ``F + 18``: a full-list entry's width.
FULL_F = F + len(FULL_EXTRA_COLUMNS)
assert FULL_F == 85 and [f.pos for f in FULL_EXTRA_COLUMNS] == list(range(F, FULL_F))
#: The keys of ``info["full_view"]`` in the layout's order (spec 5.2).
FULL_KEYS = ("full_list", "full_mask", "full_uid", "full_match",
             "full_watched")
_XC = {f.name: f.pos for f in FULL_EXTRA_COLUMNS}


def full_match_columns(n_teams: int) -> tuple:
    """``full_match``'s ``M = 2 T + 3`` columns (spec 5.4) for a run of
    ``n_teams`` marine teams: per team its living count over ``n_t`` and
    its BALANCE (the sum of its living units' ``hp / bar`` over ``n_t``,
    squashed: a bar may be exceeded), then the horde awake and dormant
    over ``spec.zombies`` (squashed: a converted marine adds one) and
    ``time_left`` over ``limit_ticks``."""
    rows = []
    for t in range(int(n_teams)):
        rows += [(f"alive[{t}]", "n_t", False), (f"balance[{t}]", "n_t", True)]
    rows += [("zombies_awake", "zombies", True),
             ("zombies_dormant", "zombies", True),
             ("time_left", "limit_ticks", False)]
    return tuple(_cols(0, rows))


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


def squash_array(x) -> np.ndarray:
    """:func:`squash` over a float64 array, element for element (the planes;
    the same curve, the same cap)."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    big = ax > 1.0
    out = x.copy()
    if big.any():
        v = np.minimum(SQUASH_CAP, 1.0 + np.log(ax[big]))  # ingress-exempt: observation output for the network, float by design; never written to synced sim state
        out[big] = np.where(x[big] > 0.0, v, -v)
    return out


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


def _cfg_float(cfg, section: str, key: str) -> float:
    sec = getattr(cfg, section, None)
    if sec is None or not hasattr(sec, key):
        raise ValueError(f"obs spec v1: config.toml [{section}] is missing "
                         f"{key!r}")
    v = getattr(sec, key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError(f"obs spec v1: [{section}] {key} must be a number "
                         f"> 0 (a plane's denominator), got {v!r}")
    return float(v)


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
    # -- the planes' table values (spec 3.2; patch (b)) --
    poison_min_density: float    # [exchange] poison_min_density
    teargas_blind_density: float  # [exchange] teargas_blind_density
    smoke_dense: float           # [bots] smoke_dense
    cap_q: int                   # ImpulseTable.cap_q: one tick of hold
    heat_dials: tuple            # exchange.heat_dials(CFG)
    # -- the full view's (spec 5.4; patch (c)) --
    n_teams: int = 0             # T: the run's marine teams (full_match)

    @property
    def n_entries(self) -> int:
        return self.n_slots + ENTRIES_BEYOND_SLOTS

    @property
    def n_match(self) -> int:
        """``M = 2 T + 3``: ``full_match``'s width (spec 5.4)."""
        return 2 * self.n_teams + 3

    @property
    def caps(self) -> tuple:
        return (self.n_slots - 1,) + CAPS_FIXED

    @property
    def obj_rows(self) -> tuple:
        return self.hand_rows + self.charge_rows

    @staticmethod
    def build(sim, spec, n_slots: int, n_teams: int | None = None) -> "ObsContext":
        """The context for one episode: ``n_slots`` the layout's ``S``,
        ``n_teams`` its ``T`` (default: the match's own team count; the
        env passes the run's, so ``full_match`` is one width per run)."""
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
            poison_min_density=_cfg_float(CFG, "exchange", "poison_min_density"),
            teargas_blind_density=_cfg_float(CFG, "exchange",
                                             "teargas_blind_density"),
            smoke_dense=_cfg_float(CFG, "bots", "smoke_dense"),
            cap_q=int(it.cap_q),
            heat_dials=exchange.heat_dials(CFG),
            n_teams=int(len(spec.teams) if n_teams is None else n_teams),
        )


# ---------------------------------------------------------------------------
# The wall: ObsWorld and TeamKnowledge (spec 0.5)
# ---------------------------------------------------------------------------
class ObsWorld:
    """The public world a builder may read, and what the sight routines,
    the line of fire and the route take as their ``sim``. Explicit
    attributes, no fallback: a banned sim member raises ``AttributeError``."""

    __slots__ = ("gmap", "tick", "weapons_tables", "speed_table",
                 "impulse_table", "graph", "wires", "cover", "doors", "_planes",
                 "_turn_rate", "_turn_accel", "_after_turn")

    def __init__(self, sim, planes=None):
        self.gmap = sim.gmap
        self.tick = int(sim.tick)
        self.weapons_tables = sim.weapons_tables
        self.speed_table = sim.speed_table
        self.impulse_table = sim.impulse_table
        self.graph = graph_of(sim)
        self.wires = tuple(getattr(sim.level, "wires", ()) or ())
        self.cover = tuple(getattr(sim, "cover", ()) or ())
        self.doors = tuple(sim._doors)      # every door runtime: unfogged
        # the whole map's planes (spec 3.2-3.4): read once per env per
        # decision by observe_rows and handed in; a direct build reads them
        # on the first ask (the full view's facade never asks)
        self._planes = planes
        self._turn_rate = sim.unit_turn_rate_q16
        self._turn_accel = sim.unit_turn_accel_q16
        self._after_turn = sim.facing_after_turn_q16

    @property
    def planes(self) -> "FieldPlanes":
        if self._planes is None:
            self._planes = FieldPlanes(self.gmap, self.doors)
        return self._planes

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
                 "_lane_of", "_aim", "_cone", "_mate_aims", "_lof")

    def __init__(self):
        self._lanes = None
        self._lane_of = {}
        self._aim = {}
        self._cone = {}
        self._mate_aims = None
        self._lof = {}

    @classmethod
    def build(cls, sim, picture, team: int, planes=None) -> "TeamKnowledge":
        """The view for ``team`` at this decision. ``planes``: the env's
        :class:`FieldPlanes` of this decision when the caller read them
        already (``observe_rows``: once per env, not per team)."""
        if picture is None:
            raise RuntimeError("obs spec v1: the sim has no team picture "
                               "(Simulation.start_team_picture)")
        team = int(team)
        k = cls()
        k.team, k.tick = team, int(sim.tick)
        k.world = ObsWorld(sim, planes)
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
    def lane_dials(self) -> tuple:
        """``(lane_cell, lane_halfwidth)``: the bots' lane dials (spec 3.6,
        a CHOICE: one source for the lanes every reader sees)."""
        from config import CFG
        return (_cfg_int(CFG, "bots", "lane_cell"),
                _cfg_int(CFG, "bots", "lane_halfwidth"))

    @property
    def lanes(self):
        """The team's fire lanes (``team_picture.fire_lanes`` over the
        mates, the bots' dials), computed once per view on first read."""
        if self._lanes is None:
            cell, hw = self.lane_dials
            self._lanes = fire_lanes_of(
                self.world.gmap, self.world.weapons_tables,
                list(self.mates.values()), self.team, cell, hw)
        return self._lanes

    def lane_of(self, uid: int):
        """ONE mate's own lane: the rasteriser's cells for that owner alone
        (``fire_lanes_of`` over the one unit; spec 2.2 cols 52-53, a
        CHOICE), computed once per view per mate; refused for a non-mate."""
        uid = int(uid)
        grid = self._lane_of.get(uid)
        if grid is None:
            m = self.mates.get(uid)
            if m is None:
                raise KeyError(f"obs spec v1: {uid} is not a living mate of "
                               f"team {self.team}: its lane is out of scope")
            cell, hw = self.lane_dials
            grid = fire_lanes_of(self.world.gmap, self.world.weapons_tables,
                                 [m], self.team, cell, hw)
            self._lane_of[uid] = grid
        return grid


# ---------------------------------------------------------------------------
# aim_of (spec 1.7): the 4.1 rule judged by the team's knowledge
# ---------------------------------------------------------------------------
def _cone_of(rows, ctx: ObsContext, u) -> int:
    """The half-angle (Q16 radians) of ``u``'s aim cone: its ranged
    weapon's ``spread_deg``, else its melee row's arc, a zombie (or a
    weaponless body) the reference kit's melee arc. Pure; the two views
    cache it each on its own."""
    row = rows.get(getattr(u, "weapon_id", "") or "")
    melee = rows.get(getattr(u, "melee_id", "") or "")
    if u.is_zombie:
        return ctx.ref_melee_arc_q
    if row is not None and int(row.range_tiles) > 0:
        return quantize_scalar(math.radians(float(row.spread_deg)))
    if melee is not None:
        return int(melee.melee_arc_half_q)
    return ctx.ref_melee_arc_q


def cone_q(know, ctx: ObsContext, u) -> int:
    """:func:`_cone_of` cached per uid on the team's view."""
    key = int(u.id)
    q = know._cone.get(key)
    if q is None:
        q = know._cone[key] = _cone_of(know.world.weapons_tables.weapons.by_name,
                                       ctx, u)
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
# The two windows (section 3): the map's planes, the far reduce, the crops
# ---------------------------------------------------------------------------
def blocks(plane, fill) -> np.ndarray:
    """``plane`` (h, w) padded to a multiple of :data:`BLOCK` with ``fill``
    (the block grid anchored at tile (0, 0)) and laid out as ``(H, W,
    BLOCK * BLOCK)``: block ``[i, j]``'s tiles in row-major order -- the
    one layout the far reduce reads (spec 3.3: a max, a min, a count, or
    the argmax's tile)."""
    plane = np.asarray(plane)
    h, w = plane.shape
    hb, wb = -(-h // BLOCK), -(-w // BLOCK)
    pad = np.full((hb * BLOCK, wb * BLOCK), fill, dtype=plane.dtype)
    pad[:h, :w] = plane
    return (pad.reshape(hb, BLOCK, wb, BLOCK).transpose(0, 2, 1, 3)
            .reshape(hb, wb, BLOCK * BLOCK))


def crop(plane, y0: int, x0: int, fill) -> np.ndarray:
    """The ``WIN x WIN`` window of ``plane`` whose top-left cell is
    ``(y0, x0)``; cells outside the plane read ``fill`` (spec 3.1)."""
    plane = np.asarray(plane)
    h, w = plane.shape
    out = np.full((WIN, WIN), fill, dtype=plane.dtype)
    sy0, sy1 = max(y0, 0), min(y0 + WIN, h)
    sx0, sx1 = max(x0, 0), min(x0 + WIN, w)
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = plane[sy0:sy1, sx0:sx1]
    return out


def inside_mask(h: int, w: int, y0: int, x0: int) -> np.ndarray:
    """Which cells of the window at ``(y0, x0)`` lie on an ``(h, w)``
    plane."""
    yy, xx = np.mgrid[y0:y0 + WIN, x0:x0 + WIN]
    return (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)


class FieldPlanes:
    """The whole map's planes for ONE decision of one env (spec 3.2-3.4),
    read once (``observe_rows``) and cropped per soldier: ``raw[name]``
    the eight physics planes in their real units (float64; the wind pair
    as the per-tile one-tick impulse of the wind, Q16 gram-tiles per tick,
    along the tile's wind -- the unit-free half of 3.2's wind law, divided
    per soldier by its mass and the hold), ``far_raw[name]`` their block
    reductions (the MAX over the block's tiles; the wind by the block's
    largest load ``W`` and that tile's direction, as ``footprint_wind``),
    ``wall`` / ``wall_share``, ``door`` / ``far_door``. A pure read of the
    map and the living doors; nothing of units."""

    __slots__ = ("shape", "raw", "far_raw", "wall", "wall_share", "door",
                 "far_door")

    def __init__(self, gmap, doors, impulse_table=None):
        g = gmap
        solid = np.asarray(g.solid, dtype=bool)
        h, w = solid.shape
        self.shape = (h, w)
        raw = {
            "temperature": gas_fixed.dequantize_q16(g.temperature),
            "fire": fire_fixed.dequantize(g.fire),
            "poison": gas_fixed.dequantize_trace(g.gas[GAS_POISON]),
            "teargas": gas_fixed.dequantize_trace(g.gas[GAS_TEARGAS]),
            "smoke": gas_fixed.dequantize_trace(g.gas[GAS_SMOKE]),
            "pressure": atmosphere_fixed.dequantize(g.atmosphere),
        }
        # the wind (spec 3.2 rows 6, 7): W_i, then one tick of impulse on
        # the standing body -- unit_impulse.wind_impulse_q's law
        # ((W >> 16) * k) >> 16 in float64 -- along the tile's wind
        table = (unit_impulse.table_for(g) if impulse_table is None
                 else impulse_table)
        k = float(table.wind_k_q)
        load, vx, vy = g.wind_load_plane()
        j = np.floor(np.floor(load / FP_ONE) * k / FP_ONE) if k > 0.0 \
            else np.zeros_like(load)
        n = np.hypot(vx, vy)  # ingress-exempt: observation output for the network, float by design; never written to synced sim state
        with np.errstate(divide="ignore", invalid="ignore"):
            dx = np.where(n > 0.0, vx / n, 0.0)
            dy = np.where(n > 0.0, vy / n, 0.0)
        raw["wind_x"] = j * dx
        raw["wind_y"] = j * dy
        self.raw = raw
        far = {name: blocks(raw[name], 0.0).max(axis=-1)
               for name in PHYSICS_PLANES[:6]}
        at = blocks(load, 0.0).argmax(axis=-1)[..., None]
        for name in ("wind_x", "wind_y"):
            far[name] = np.take_along_axis(blocks(raw[name], 0.0), at,
                                           axis=-1)[..., 0]
        self.far_raw = far
        self.wall = solid.astype(np.uint8)
        self.wall_share = (blocks(self.wall, 1).astype(np.int64).sum(axis=-1)
                           / float(BLOCK * BLOCK))
        door = np.zeros((h, w), dtype=np.uint8)
        for d in doors:
            if not d.alive:
                continue
            code = (DOOR_CODE_OPEN if d.state == DOOR_OPEN
                    else DOOR_CODE_CLOSED if d.state == DOOR_CLOSED
                    else DOOR_NONE)
            if code:
                for t in d.span:
                    ty, tx = int(t[0]), int(t[1])
                    if 0 <= ty < h and 0 <= tx < w:
                        door[ty, tx] = max(door[ty, tx], code)
        self.door = door
        self.far_door = blocks(door, 0).max(axis=-1)


def _div(x, d):
    """``x / d`` over an array or a scalar; zeros when the table value is 0
    (the ``ratio`` rule)."""
    d = float(d)
    if d == 0.0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return np.asarray(x, dtype=np.float64) / d


def physics_value(name: str, raw, u, ctx: ObsContext):
    """One physics plane's stored ratio (before the squash) for the
    observing soldier ``u`` from its real-unit value(s) ``raw`` (spec 3.2's
    "divided by" column): temperature over the profile's ``temperature_max``;
    fire as it is; poison over ``poison_min_density`` times the unit's
    ``resist_mult[POISON]``; teargas over ``teargas_blind_density``; smoke
    over ``smoke_dense``; pressure as the profile's band (0 at the low
    limit, 1 at the high); the wind pair as the one-tick impulse over the
    body's mass times the hold (``cap_q``). The same function for the
    windows and the vector's ``air`` block."""
    env = getattr(u, "environment", None)
    if name == "temperature":
        return _div(raw, getattr(env, "temperature_max", 0.0))
    if name == "fire":
        return np.asarray(raw, dtype=np.float64)
    if name == "poison":
        return (_div(raw, ctx.poison_min_density)
                * float(mitigation_for(u).resist_mult[POISON]))
    if name == "teargas":
        return _div(raw, ctx.teargas_blind_density)
    if name == "smoke":
        return _div(raw, ctx.smoke_dense)
    if name == "pressure":
        lo = float(getattr(env, "pressure_min", 0.0))
        hi = float(getattr(env, "pressure_max", 0.0))
        return _div(np.asarray(raw, dtype=np.float64) - lo, hi - lo)
    if name in ("wind_x", "wind_y"):
        return _div(raw, float(int(u.mass_g)) * float(ctx.cap_q))
    raise KeyError(f"obs spec v1: no physics plane {name!r}")


_PLACE_BLOCKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def place_blocks(graph, where: int) -> np.ndarray:
    """The far ``place`` plane over the whole block grid (spec 3.8): 1 on a
    block holding a tile of the place -- a room's tiles (``graph.grid ==
    room``), an opening's rectangle -- cached per ``(graph, where)``."""
    slot = _PLACE_BLOCKS.setdefault(graph, {})
    plane = slot.get(int(where))
    if plane is None:
        grid = np.asarray(graph.grid)
        mask = np.zeros(grid.shape, dtype=np.uint8)
        if where < ORDER_OPENING_BASE:
            mask[grid == int(where)] = 1
        else:
            for o in graph.openings:
                if ORDER_OPENING_BASE + int(o.id) == int(where):
                    mask[o.y:o.y + o.h, o.x:o.x + o.w] = 1
        plane = blocks(mask, 0).max(axis=-1)
        slot[int(where)] = plane
    return plane


def build_windows(know: TeamKnowledge, u, ctx: ObsContext, near_f, near_u,
                  far_f, far_u) -> None:
    """Section 3 into ``near_f`` (8, 25, 25), ``near_u`` (5, 25, 25),
    ``far_f`` (10, 25, 25) and ``far_u`` (5, 25, 25) for the living mate
    ``u`` of ``know``'s team: the crops of the map's planes with the
    soldier's own denominators (3.2, 3.3), the structure planes (3.4),
    ``watched`` from the RECORDED centres through ``vision.window_clear``
    and ``sees_target`` 0 until 1h (3.5), the lanes plane (3.6), the far
    unit planes from the whole picture (3.7), the order's ``route`` and
    ``place`` planes (3.8)."""
    world = know.world
    P = world.planes
    h, w = P.shape
    me = int(u.id)
    cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
    y0, x0 = cy - HALF, cx - HALF
    by0, bx0 = cy // BLOCK - HALF, cx // BLOCK - HALF
    hb, wb = P.wall_share.shape
    # -- near (3.1, 3.2, 3.4) --
    inside = inside_mask(h, w, y0, x0)
    for k, name in enumerate(PHYSICS_PLANES):
        v = squash_array(physics_value(name, crop(P.raw[name], y0, x0, 0.0),
                                       u, ctx))
        v[~inside] = 0.0
        near_f[k] = v
    near_u[0] = crop(P.wall, y0, x0, OUTSIDE_FILL["wall"])
    near_u[1] = crop(P.door, y0, x0, 0)
    # -- the sight planes (3.5): from every KNOWN enemy's recorded centre --
    watched = np.zeros((WIN, WIN), dtype=bool)
    for rec in know.records.values():
        rx, ry = int(rec.centre[0]), int(rec.centre[1])
        watched |= vision.window_clear(world, ry, rx, y0, x0, WIN, WIN,
                                       to_window=False)
    near_u[2] = watched
    near_u[3] = 0                       # sees_target: 1h's aim fills it
    # -- the lanes plane (3.6): a team-mate's line of fire, never mine --
    grid = know.lanes
    cell = know.lane_dials[0]
    yy, xx = np.mgrid[y0:y0 + WIN, x0:x0 + WIN]
    owner = np.full((WIN, WIN), -1, dtype=np.int64)
    owner[inside] = grid[yy[inside] // cell, xx[inside] // cell]
    near_u[4] = (owner == -2) | ((owner >= 0) & (owner != me))
    # -- far (3.3, 3.4) --
    binside = inside_mask(hb, wb, by0, bx0)
    far_f[0] = crop(P.wall_share, by0, bx0, OUTSIDE_FILL["wall_share"])
    for k, name in enumerate(PHYSICS_PLANES):
        v = squash_array(physics_value(name, crop(P.far_raw[name], by0, bx0,
                                                  0.0), u, ctx))
        v[~binside] = 0.0
        far_f[1 + k] = v
    far_u[0] = crop(P.far_door, by0, bx0, 0)
    # -- the far unit planes (3.7): the whole picture, recorded centres --
    far_u[1:4] = 0

    def bump(k, x, y):
        bi, bj = y // BLOCK - by0, x // BLOCK - bx0
        if 0 <= bi < WIN and 0 <= bj < WIN:
            far_u[k, bi, bj] += 1

    for muid, m in know.mates.items():
        if muid != me:
            bump(1, int(m.center_tile_x()), int(m.center_tile_y()))
    for uid, rec in know.records.items():
        bump(2 if uid in know.seen else 3, int(rec.centre[0]),
             int(rec.centre[1]))
    # -- the order's planes (3.8) --
    far_f[9] = OUTSIDE_FILL["route"]
    far_u[4] = 0
    graph = world.graph
    if int(u.order_what) and graph is not None:
        where = int(u.order_where)
        field = route(world, graph, where, int(u.footprint))
        fb = blocks(field, horde.UNREACHED).min(axis=-1)
        reached = fb != horde.UNREACHED
        rp = np.full(fb.shape, OUTSIDE_FILL["route"], dtype=np.float64)
        rp[reached] = squash_array(_div(fb[reached], ctx.l_ref))
        far_f[9] = crop(rp, by0, bx0, OUTSIDE_FILL["route"])
        far_u[4] = crop(place_blocks(graph, where), by0, bx0, 0)


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


def _kit_values(u, weapon, melee, loadout, tick: int, ctx: ObsContext) -> tuple:
    """The kit ratios of spec 1.2 that the soldier's vector (cols 24-26,
    29.., 44, 49, 51) and the full view's extra columns (5.3 cols 68-79)
    both carry, as ``(rounds, mags, untracked, reloading, reload_left,
    grenade[0..], charges, melee_cool, shove_cool)`` -- one reader, so the
    two agree column for column."""
    st = combat.mag_status(u, weapon, tick) if weapon is not None else None
    mags_total = int(loadout.magazines) if loadout is not None else 0
    reloading = reload_left = 0.0
    if st is None:
        rounds = mags = untracked = 1.0
    else:
        r_, spares, rel = st
        rounds = ratio(r_, weapon.mag_size)
        if spares is None:
            mags = untracked = 1.0
        else:
            mags, untracked = squash(ratio(spares, max(1, mags_total))), 0.0
        reloading = float(rel)
        if rel:
            reload_left = ratio(int(u.reload_done_tick) - tick,
                                weapon.reload_ticks)
    counts = loadout.grenades if loadout is not None else {}
    grenades = []
    for k, row in enumerate(ctx.hand_rows):
        have = int(counts.get(row, 0))
        grenades.append(ratio(u.grenade_counts[k], have) if have > 0 else 0.0)
    explosives = int(loadout.explosives) if loadout is not None else 0
    charges = ratio(u.has_explosive, explosives)
    melee_cool = 0.0
    if melee is not None and int(u.last_blow_tick) >= 0:
        rof = int(melee.rof_interval_ticks)
        melee_cool = ratio(max(0, int(u.last_blow_tick) + rof - tick), rof)
    shove_cool = 0.0
    cd = ctx.shove_cooldown_ticks
    if int(u.last_shove_tick) >= 0:
        shove_cool = ratio(max(0, int(u.last_shove_tick) + cd - tick), cd)
    return (rounds, mags, untracked, reloading, reload_left, tuple(grenades),
            charges, melee_cool, shove_cool)


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
    kit = _kit_values(u, weapon, melee, loadout, tick, ctx)
    out[24], out[25], out[26], out[27], out[28] = kit[:5]
    counts = loadout.grenades if loadout is not None else {}
    for k, row in enumerate(ctx.hand_rows):
        if int(counts.get(row, 0)) > 0:
            out[29 + 2 * k] = kit[5][k]
            out[30 + 2 * k] = 1.0
    winding = int(u.throw_release_tick) >= 0
    out[41] = float(winding)
    if winding:
        out[42] = ratio(int(u.throw_release_tick) - tick, ctx.throw_windup_ticks)
        out[43] = float(int(u.throw_kind))
    else:
        out[43] = -1.0
    out[44] = kit[6]
    out[45] = float(bool(loadout is not None and loadout.charge))
    planting = int(u.plant_done_tick) >= 0
    out[46] = float(planting)
    if planting:
        out[47] = ratio(int(u.plant_done_tick) - tick, ctx.plant_ticks)
        out[48] = ratio(int(u.plant_fuse_ticks), ctx.c4_fuse_max_ticks)
    out[49] = kit[7]
    out[50] = float(bool(getattr(u, "melee_id", "")))
    out[51] = kit[8]
    out[52] = float(_row_index("loadouts", getattr(u, "loadout_name", None)))
    out[53] = float(_row_index("weapons", getattr(u, "weapon_id", "")))
    if weapon is not None:
        out[54] = squash(ratio(weapon.range_tiles, ctx.l_ref))
        out[55] = squash(ratio(weapon.rof_interval_ticks, ctx.ref_rof_ticks))
        out[56] = squash(ratio(weapon.spread_deg, ctx.ref_spread_deg))
    out[57] = float(bool(getattr(u, "live_trigger", False)))
    # -- 1.3 the air where it stands: the near window's value at the centre
    # tile (the same plane builders, 3.2) and the heat row's own felt
    # temperature over the profile's edge --
    P = world.planes
    cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
    if 0 <= cy < P.shape[0] and 0 <= cx < P.shape[1]:
        for k, name in enumerate(PHYSICS_PLANES):
            out[58 + k] = squash(physics_value(name, P.raw[name][cy, cx], u,
                                               ctx))
    t_felt = exchange.felt_temperature(world.gmap, u, ctx.heat_dials)[0]
    out[96] = squash(ratio(t_felt, getattr(getattr(u, "environment", None),
                                           "temperature_max", 0.0)))
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


@dataclass(frozen=True)
class _Aimer:
    """The observing soldier's own terms for the enemy columns 27, 29, 30
    (``ticks_to_on``, ``in_range``, ``in_throw_range``) and ``frag_catch``:
    its weapon's reach and spread, its turning dials and pending spin, the
    cap -- read once per soldier per decision, by either view."""
    range2: int
    spread_q: int
    throw2: int
    frag2: int
    a: int
    w: int
    dw: int
    cap: int

    @staticmethod
    def build(world, u, ctx: ObsContext) -> "_Aimer":
        rows = world.weapons_tables.weapons.by_name
        my_weapon = rows.get(getattr(u, "weapon_id", "") or "")
        return _Aimer(
            range2=(int(my_weapon.range_tiles) ** 2 if my_weapon is not None
                    else -1),
            spread_q=(quantize_scalar(math.radians(float(my_weapon.spread_deg)))
                      if my_weapon is not None else 0),
            throw2=ctx.max_throw_range ** 2, frag2=ctx.frag_radius ** 2,
            a=world.unit_turn_accel_q16(u), w=world.unit_turn_rate_q16(u),
            dw=unit_impulse.pending_dw(world.impulse_table, u),
            cap=4 * ctx.half_turn_ticks)


def _enemy_geometry(e, aimer: _Aimer, u, dx: int, dy: int, ctx: ObsContext):
    """Cols 27, 29, 30 of an enemy entry at offset ``(dx, dy)`` from ``u``:
    the ticks to bring the aim within the spread (``turning.ticks_to_within``,
    a call of the law) over the half turn, the reach flags."""
    want = atan2_q16(-dy << 16, dx << 16)
    ticks = turning.ticks_to_within(u, want, aimer.a, aimer.w, aimer.dw,
                                    aimer.spread_q, aimer.cap)
    e[27] = squash(ratio(ticks, ctx.half_turn_ticks))
    d2 = dx * dx + dy * dy
    e[29] = float(d2 <= aimer.range2)
    e[30] = float(d2 <= aimer.throw2)


def _mate_tail(e, m, mx: int, my: int, cx: int, cy: int, tick: int,
               ctx: ObsContext, graph, my_where: int, aim_is_unit: bool):
    """Cols 39-51 of a team-mate entry (spec 2.2, the team display on a
    team-mate): its facing TARGET, whether that aim rests on a unit (the
    caller's ``aim_of`` answer), its feet, where its wound-up throw lands
    from ME, its order's kind, place and clock. The lane pair (52, 53) is
    the caller's: each view has its own lanes."""
    ft = int(m.facing_target_q)
    e[39], e[40] = cos_q16(ft) / FP_ONE, sin_q16(ft) / FP_ONE
    e[41] = float(aim_is_unit)
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
        e[50] = float(my_where != NO_PLACE and int(m.order_where) == my_where)
        until = int(m.order_until)
        if until != NO_LIMIT:
            e[51] = ratio(until - tick, ctx.limit_ticks)


def _object_entry(e, o, cx: int, cy: int, me: int, roster_ids, tick: int,
                  ctx: ObsContext, tables) -> None:
    """A grenade or a charge's entry (spec 2.2, type 3): cols 0-9 and
    55-61 -- its tile from me, a grenade's flight as a velocity, ``side``
    -1, its ammo row, ``whose`` by the roster (never a live owner read),
    the time left over its family's fuse, the remote flag, the blast
    radius and whether I stand in it, whether it lies on the floor."""
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
    e[55] = float(ctx.obj_rows.index(name)) if name in ctx.obj_rows else -1.0
    owner = int(o.owner_id)
    e[56] = (0.0 if owner == me else 1.0 if owner in roster_ids
             else 2.0 if owner >= 0 else 3.0)
    if charge:
        if o.det_tick is None:
            e[58] = 1.0
        else:
            e[57] = squash(ratio(int(o.det_tick) - tick, ctx.c4_fuse_max_ticks))
        e[61] = 1.0
    else:
        e[57] = squash(ratio(int(o.get_detonate_tick()) - tick,
                             ctx.hand_fuse_max_ticks))
        e[61] = float(o.fx == o.target_fx and o.fy == o.target_fy)
    try:
        r = int(tables.payload_for_ammo(name).radius)
    except KeyError:
        r = 0
    e[59] = squash(ratio(r, ctx.l_ref))
    e[60] = float((ox - cx) ** 2 + (oy - cy) ** 2 <= r * r)


def _door_entry(e, d, u, cx: int, cy: int, ctx: ObsContext, gmap,
                wired) -> None:
    """A door's entry (spec 2.2, type 4): cols 0-9 and 62-65 -- its span's
    centre from me, ``side`` / ``class`` -1, open, in the use verb's reach
    of ``u``, a hull door, wired."""
    dx_, dy_ = _door_centre(d)
    _entry_common(e, TYPE_DOOR, dx_ - cx, dy_ - cy, ctx)
    e[8], e[9] = -1.0, -1.0
    e[62] = float(d.state == DOOR_OPEN)
    e[63] = float(verb_gates.door_in_reach(u, d))
    e[64] = float(horde.is_hull_door(gmap, d.span))
    e[65] = float(int(d.ordinal) in wired)


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
    aimer = _Aimer.build(world, u, ctx)
    frag2 = aimer.frag2
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
        _mate_tail(e, m, mx, my, cx, cy, tick, ctx, graph, my_where,
                   aim_of(know, ctx, m, me)[0] != 0)
        # the lane pair (3.6, a CHOICE): each owner's OWN rasterised lane
        cell = know.lane_dials[0]
        e[52] = float(know.lane_of(muid)[cy // cell, cx // cell] == muid)
        e[53] = float(know.lane_of(me)[my // cell, mx // cell] == me)
        # col 66 (prev_aim) is 1h's
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
        _enemy_geometry(e, aimer, u, dx, dy, ctx)   # col 28 is 1h's
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
        _object_entry(out[i], o, cx, cy, me, know.roster_ids, tick, ctx,
                      world.weapons_tables)
        mask[i], uid[i] = 1, ouid
        i += 1
    i = caps[0] + caps[1] + caps[2] + caps[3]

    # -- type 4: doors --
    wired = room_graph.wired_door_ordinals(world.wires)
    for _, ordinal, d in doors[:caps[4]]:
        _door_entry(out[i], d, u, cx, cy, ctx, world.gmap, wired)
        mask[i], uid[i] = 1, DOOR_UID_BASE + ordinal
        i += 1


# ---------------------------------------------------------------------------
# The full view (section 5): the other side of the wall
# ---------------------------------------------------------------------------
class FullView:
    """What the value network is shown, read from the sim ITSELF (spec 5.2:
    "the other side of the wall"), ONE per env per decision, built by
    :meth:`build` -- the one full-view function that touches ``sim``. It
    holds every living unit, every live object and door, the picture and
    the sight cache, and the full view's OWN caches: the aim cone per uid,
    the every-unit aim answer per ``(uid, ray, cone, centre)``, the line of
    fire per tile pair, the lanes per team and per mate, the mates' aims per
    team, and ``full_watched``'s window per ``(enemy centre, window)``.
    Nothing here is ``TeamKnowledge``'s and nothing of it reaches ``obs``:
    the pair test (f) of spec 7 holds it."""

    __slots__ = ("tick", "world", "picture", "vis", "living", "marines",
                 "zombies", "objects", "doors", "rows", "_roster", "_cone",
                 "_aim", "_lof", "_lanes", "_lane_of", "_mate_aims",
                 "_watched", "_lane_dials", "_wired", "_match",
                 "_seen_objects", "_known_objects", "_static")

    def __init__(self):
        self._roster = {}
        self._cone = {}
        self._aim = {}
        self._lof = {}
        self._lanes = {}
        self._lane_of = {}
        self._mate_aims = {}
        self._watched = {}
        self._lane_dials = None
        self._wired = None
        self._match = None
        self._seen_objects = {}
        self._known_objects = {}
        self._static = {}

    @classmethod
    def build(cls, sim, ctx: ObsContext) -> "FullView":
        """The env's full view at this decision: every living unit of every
        side, the marines in ``(team, slot)`` order, the objects in the
        trackers' order, the living doors, the picture, the sight cache."""
        picture = sim.team_picture
        if picture is None:
            raise RuntimeError("obs spec v1: the sim has no team picture "
                               "(Simulation.start_team_picture)")
        fv = cls()
        fv.tick = int(sim.tick)
        fv.world = ObsWorld(sim, planes=None)     # its planes never asked for
        fv.picture = picture
        fv.vis = vision.state_for(sim)
        fv.living = [u for u in sim.units if u.alive]
        fv.marines = sorted((u for u in fv.living if not u.is_zombie),
                            key=lambda u: (int(u.team), int(getattr(u, "slot", 0)),
                                           int(u.id)))
        fv.zombies = [u for u in fv.living if u.is_zombie]
        fv.objects = tuple(live_objects(sim))
        fv.doors = tuple(d for d in sim._doors if d.alive)
        fv.rows = fv.world.weapons_tables.weapons.by_name
        for t in range(len(ctx.spec.teams)):
            fv._roster[t] = frozenset(int(u.id) for u in sim.roster(t)
                                      if u is not None)
        return fv

    # -- the team's side of the full view ---------------------------------
    def roster_ids(self, team: int) -> frozenset:
        return self._roster.get(int(team), frozenset())

    def mates(self, team: int) -> dict:
        """``team``'s living non-zombie units by id."""
        return {int(u.id): u for u in self.marines if int(u.team) == int(team)}

    def seen_objects(self, team: int) -> dict:
        """``{id(obj): team uid}`` of the objects ``team`` sees now."""
        team = int(team)
        got = self._seen_objects.get(team)
        if got is None:
            got = self._seen_objects[team] = {
                id(o): ouid for o, ouid in self.picture.seen_now.get(team, ())}
        return got

    def known_objects(self, team: int) -> frozenset:
        """``id(obj)`` of every live object ``team`` has ever seen."""
        team = int(team)
        got = self._known_objects.get(team)
        if got is None:
            got = self._known_objects[team] = frozenset(
                id(o) for o in self.picture.object_ids(team))
        return got

    @property
    def lane_dials(self) -> tuple:
        if self._lane_dials is None:
            from config import CFG
            self._lane_dials = (_cfg_int(CFG, "bots", "lane_cell"),
                                _cfg_int(CFG, "bots", "lane_halfwidth"))
        return self._lane_dials

    def lane_of(self, team: int, m) -> np.ndarray:
        """ONE mate's own rasterised lane (the full view's own copy)."""
        key = (int(team), int(m.id))
        grid = self._lane_of.get(key)
        if grid is None:
            cell, hw = self.lane_dials
            grid = self._lane_of[key] = fire_lanes_of(
                self.world.gmap, self.world.weapons_tables, [m], int(team),
                cell, hw)
        return grid

    @property
    def wired(self) -> frozenset:
        if self._wired is None:
            self._wired = room_graph.wired_door_ordinals(self.world.wires)
        return self._wired

    # -- the caches ---------------------------------------------------------
    def cone(self, ctx: ObsContext, u) -> int:
        key = int(u.id)
        q = self._cone.get(key)
        if q is None:
            q = self._cone[key] = _cone_of(self.rows, ctx, u)
        return q

    def aim(self, ctx: ObsContext, u, use_target: bool) -> int:
        """The 4.1 rule with EVERY living unit as a candidate (spec 5.3):
        the nearest other living unit within ``u``'s cone of its ray (its
        facing TARGET when ``use_target`` -- a team-mate of the observing
        soldier -- else its facing) to which the ray is clear; its uid, or
        -1. Cached per ``(uid, ray, cone, centre)``."""
        uid_u = int(u.id)
        ray = int(u.facing_target_q if use_target else u.facing_q)
        cone = self.cone(ctx, u)
        ux, uy = int(u.center_tile_x()), int(u.center_tile_y())
        key = (uid_u, ray, cone, ux, uy)
        hit = self._aim.get(key)
        if hit is None:
            inside = []
            for v in self.living:
                vid = int(v.id)
                if vid == uid_u:
                    continue
                vx, vy = int(v.center_tile_x()), int(v.center_tile_y())
                dx, dy = vx - ux, vy - uy
                bearing = atan2_q16(-dy << 16, dx << 16)
                if abs(wrap_q16(bearing - ray)) <= cone:
                    inside.append((dx * dx + dy * dy, vid, vx, vy))
            inside.sort()
            hit = -1
            for _, vid, vx, vy in inside:
                if vision.ray_clear(self.world, uy, ux, vy, vx):
                    hit = vid
                    break
            self._aim[key] = hit
        return hit

    def mate_aims(self, ctx: ObsContext, team: int) -> list:
        """Every living mate of ``team``'s every-unit aim answer and
        trigger, once per decision per team."""
        team = int(team)
        got = self._mate_aims.get(team)
        if got is None:
            got = self._mate_aims[team] = [
                (muid, self.aim(ctx, m, True),
                 bool(getattr(m, "live_trigger", False)))
                for muid, m in self.mates(team).items()]
        return got

    def lof_batch(self, cx: int, cy: int, targets) -> None:
        """Fill the line-of-fire cache from ``(cx, cy)`` to every centre of
        ``targets`` not yet in it, in ONE sight call
        (``line_of_fire.solid_in_way_many``)."""
        want = [(ex, ey) for ex, ey in targets
                if (cx, cy, ex, ey) not in self._lof]
        if want:
            got = line_of_fire.solid_in_way_many(self.world, (cx, cy), want)
            for (ex, ey), v in zip(want, got):
                self._lof[(cx, cy, ex, ey)] = bool(v)

    def lof(self, cx: int, cy: int, ex: int, ey: int) -> bool:
        key = (cx, cy, ex, ey)
        got = self._lof.get(key)
        if got is None:
            got = self._lof[key] = line_of_fire.solid_in_way(
                self.world, (cx, cy), (ex, ey))
        return got

    def static(self, ctx: ObsContext, it) -> np.ndarray:
        """A living unit's observer-independent columns, once per unit per
        decision: cols 4-10 (its facing, the motion tracker's velocity --
        the full view knows every unit's truth --, side, class, hp over its
        bar), 11-23 (``_unit_columns``) and ``F + 1 .. F + 17`` (the kit,
        four statuses, the order). The rows copy them in."""
        uid = int(it.id)
        st = self._static.get(uid)
        if st is None:
            st = np.zeros(FULL_F, dtype=np.float32)
            f = int(it.facing_q)
            st[4], st[5] = cos_q16(f) / FP_ONE, sin_q16(f) / FP_ONE
            vx, vy = self.picture.velocity_q(uid)
            st[6] = squash(ratio(vx, ctx.v_ref_q))
            st[7] = squash(ratio(vy, ctx.v_ref_q))
            st[8] = float(int(it.team))
            st[9] = (float(ctx.zombie_class) if it.is_zombie
                     else float(_row_index("loadouts",
                                           getattr(it, "loadout_name", None))))
            st[10] = squash(ratio(it.current_hp,
                                  self.picture.bars.get(uid, 0.0)))
            _unit_columns(st, it, ctx, self.tick, self.rows)
            _full_extra(st, self, it, self.tick, ctx)
            self._static[uid] = st
        return st

    def watched_from(self, ey: int, ex: int, y0: int, x0: int) -> np.ndarray:
        """The window at ``(y0, x0)`` an enemy at ``(ey, ex)`` sees
        (``vision.window_clear`` FROM its tile), the full view's own copy."""
        key = (ey, ex, y0, x0)
        got = self._watched.get(key)
        if got is None:
            got = self._watched[key] = vision.window_clear(
                self.world, ey, ex, y0, x0, WIN, WIN, to_window=False)
        return got

    # -- the match (5.4): one vector per env per decision -------------------
    def match(self, ctx: ObsContext) -> np.ndarray:
        if self._match is None:
            m = np.zeros(ctx.n_match, dtype=np.float32)
            teams = ctx.spec.teams
            bars = self.picture.bars
            for t in range(ctx.n_teams):
                if t >= len(teams):
                    continue
                n_t = int(teams[t].n)
                mine = [u for u in self.marines if int(u.team) == t]
                m[2 * t] = ratio(len(mine), n_t)
                balance = sum(ratio(u.current_hp, bars.get(int(u.id), 0.0))
                              for u in mine)
                m[2 * t + 1] = squash(ratio(balance, n_t))
            nz = int(ctx.spec.zombies)
            awake = sum(1 for z in self.zombies if z.zombie_activated)
            k = 2 * ctx.n_teams
            m[k] = squash(ratio(awake, nz))
            m[k + 1] = squash(ratio(len(self.zombies) - awake, nz))
            m[k + 2] = ratio(ctx.limit_ticks - self.tick, ctx.limit_ticks)
            self._match = m
        return self._match


def _full_entries(fv: FullView, u, cx: int, cy: int) -> tuple:
    """The full list's entries after the soldier's own (spec 5.3): the
    other living marines in ``(team, slot)`` order, the zombies nearest
    first, the objects nearest first (ties in the trackers' order), the
    doors nearest first (ties by ordinal); over :data:`FULL_ENTRIES` the
    farthest zombies are dropped first, then the farthest objects (never a
    marine; a door only past that, a CHOICE for an absurd map). Returns
    ``(marines, zombies, objects, doors)`` as kept."""
    me = int(u.id)

    def d2(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2

    marines = [m for m in fv.marines if int(m.id) != me]
    # every living zombie: a TURNED soldier's own zombie is among them, shown
    # as the zombie it is behind its corpse entry (5.6, a CHOICE)
    zombies = sorted(((d2(z.center_tile_x(), z.center_tile_y()), int(z.id), z)
                      for z in fv.zombies), key=lambda t: t[:2])
    objects = sorted(((d2(object_tile(o)[1], object_tile(o)[0]), k, o)
                      for k, o in enumerate(fv.objects)), key=lambda t: t[:2])
    doors = sorted(((d2(*_door_centre(d)), int(d.ordinal), d)
                    for d in fv.doors), key=lambda t: t[:2])
    room = FULL_ENTRIES - 1 - len(marines)
    if room < len(doors):
        doors = doors[:max(0, room)]
    room -= len(doors)
    objects = objects[:max(0, min(len(objects), room))]
    zombies = zombies[:max(0, min(len(zombies), room - len(objects)))]
    return (marines, [z for _, _, z in zombies], [o for _, _, o in objects],
            [d for _, _, d in doors])


def _full_extra(e, fv: FullView, it, tick: int, ctx: ObsContext) -> None:
    """Cols ``F+1..F+17`` of a UNIT entry (spec 5.3): its kit (the vector's
    own reader), four statuses, its order; col ``F`` (the team's knowledge
    of it) is the row's."""
    weapon = fv.rows.get(getattr(it, "weapon_id", "") or "")
    melee = fv.rows.get(getattr(it, "melee_id", "") or "")
    kit = _kit_values(it, weapon, melee, _loadout_of(it), tick, ctx)
    e[F + 1], e[F + 2], e[F + 3] = kit[0], kit[1], kit[2]
    for k in range(HAND_GRENADE_SLOTS):
        e[F + 4 + k] = kit[5][k]
    e[F + 10], e[F + 11], e[F + 12] = kit[6], kit[7], kit[8]
    for j, kind in enumerate(FULL_STATUS_KINDS):
        e[F + 13 + j] = squash(ratio(_status_left(it, kind),
                                     ctx.status_scale[kind]))
    e[F + 17] = float(int(it.order_what))


def _knowledge(uid: int, recs, seen) -> int:
    """Spec 5.3 col 67: the team sees it, has a record of it, or not."""
    if uid in seen:
        return KNOW_SEEN
    if uid in recs:
        return KNOW_RECORD
    return KNOW_UNKNOWN


def build_full_list(fv: FullView, team: int, u, alive: bool, ctx: ObsContext,
                    out, mask, uid) -> None:
    """Section 5.3 into ``out`` (``(FULL_ENTRIES, FULL_F)`` float32),
    ``mask`` and ``uid`` for the roster unit ``u`` of ``team``'s row:
    entry 0 the soldier itself (``alive``: its live entry at offset 0; else
    the corpse of 5.6 -- ``type`` 0, ``hp`` 0, ``age`` = the ticks since the
    picture first saw it dead or turned over ``limit_ticks``), then every
    other living unit where it truly is (a team-mate as a type-0 entry, any
    other unit as type 1 with the LIVE unit's columns and ``aims_at`` over
    every living unit), every live object (``whose`` by the roster, the
    uid ``OBJECT_UID_BASE`` + its place in the block), every living door --
    the first 67 columns 2.2's entry by the same helpers against ``u``,
    then :data:`FULL_EXTRA_COLUMNS`."""
    out[...] = 0.0
    mask[...] = 0
    uid[...] = -1
    world = fv.world
    tick = fv.tick
    rows = fv.rows
    team = int(team)
    me = int(u.id)
    cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
    aimer = _Aimer.build(world, u, ctx)
    mates = fv.mates(team)
    mates_c = [(int(m.center_tile_x()), int(m.center_tile_y()), muid)
               for muid, m in mates.items()]
    others_c = [(x, y) for x, y, muid in mates_c if muid != me]
    teams = ctx.spec.teams
    n_t = int(teams[team].n) if team < len(teams) else 0
    my_where = int(u.order_where) if int(u.order_what) else NO_PLACE
    graph = world.graph
    recs = fv.picture.records(team)
    seen = frozenset(fv.vis.visible_by_team.get(team, ()))
    cell = fv.lane_dials[0]
    marines, zombies, objects, doors = _full_entries(fv, u, cx, cy)
    enemies = [m for m in marines if int(m.team) != team] + zombies
    fv.lof_batch(cx, cy, [(int(v.center_tile_x()), int(v.center_tile_y()))
                          for v in enemies])

    def unit_head(e, it, etype, ex, ey):
        """Cols 0-3 of a living unit's entry from me, then its
        observer-independent columns copied in (``FullView.static``)."""
        _entry_common(e, etype, ex - cx, ey - cy, ctx)
        st = fv.static(ctx, it)
        e[4:24] = st[4:24]
        e[F + 1:FULL_F] = st[F + 1:FULL_F]

    def mate_entry(e, m):
        mx, my = int(m.center_tile_x()), int(m.center_tile_y())
        unit_head(e, m, TYPE_MATE, mx, my)
        _mate_tail(e, m, mx, my, cx, cy, tick, ctx, graph, my_where,
                   fv.aim(ctx, m, True) >= 0)
        if int(m.id) != me:          # my own lane marks nothing (a CHOICE)
            e[52] = float(fv.lane_of(team, m)[cy // cell, cx // cell] == int(m.id))
            e[53] = float(fv.lane_of(team, u)[my // cell, mx // cell] == me)
        e[F] = float(KNOW_SEEN)

    def enemy_entry(e, it):
        ex, ey = int(it.center_tile_x()), int(it.center_tile_y())
        unit_head(e, it, TYPE_SEEN, ex, ey)
        euid = int(it.id)
        e[24] = float(me in fv.vis.observers_of(euid))
        e[25] = float(fv.lof(cx, cy, ex, ey))
        e[26] = float(line_of_fire.mate_in_way((cx, cy), (ex, ey), others_c,
                                               ctx.line_clear_tiles,
                                               int(u.footprint)))
        _enemy_geometry(e, aimer, u, ex - cx, ey - cy, ctx)
        hit = fv.aim(ctx, it, False)
        cat = 0 if hit < 0 else 1 if hit == me else 2 if hit in mates else 3
        e[31] = float(cat == 1)
        e[32] = float(cat)
        if it.is_zombie:
            e[33] = (2.0 if it.zombie_activated
                     else 3.0 if int(it.zombie_noise_ticks) > 0 else 1.0)
        aiming = [(muid, trig) for muid, aimed, trig in fv.mate_aims(ctx, team)
                  if muid != me and aimed == euid]
        e[34] = ratio(len(aiming), n_t)
        e[35] = ratio(sum(1 for _, trig in aiming if trig), n_t)
        e[36] = squash(ratio(fv.picture.damage_last_s(team, euid),
                             fv.picture.bars.get(euid, 0.0)))
        e[37] = float(int(u.order_target) == euid)
        e[38] = ratio(sum(1 for mx, my, _ in mates_c
                          if (mx - ex) ** 2 + (my - ey) ** 2 <= aimer.frag2),
                      n_t)
        e[F] = float(_knowledge(euid, recs, seen))

    # -- entry 0: the soldier itself (5.3; the corpse of 5.6) --
    e = out[0]
    if alive:
        mate_entry(e, u)
    else:
        _entry_common(e, TYPE_MATE, 0, 0, ctx)
        f = int(u.facing_q)
        e[4], e[5] = cos_q16(f) / FP_ONE, sin_q16(f) / FP_ONE
        e[8] = float(team)
        e[9] = float(_row_index("loadouts", getattr(u, "loadout_name", None)))
        died = fv.picture.death_tick.get(me, tick)
        e[54] = ratio(tick - int(died), ctx.limit_ticks)
    mask[0], uid[0] = 1, me
    i = 1
    # -- the other living units: the marines, then the zombies --
    for m in marines:
        e = out[i]
        if int(m.team) == team:
            mate_entry(e, m)
        else:
            enemy_entry(e, m)
        mask[i], uid[i] = 1, int(m.id)
        i += 1
    for z in zombies:
        enemy_entry(out[i], z)
        mask[i], uid[i] = 1, int(z.id)
        i += 1
    # -- the objects (type 3): whose by the roster, the block position's uid --
    roster_ids = fv.roster_ids(team)
    seen_objs = fv.seen_objects(team)
    known_objs = fv.known_objects(team)
    for k, o in enumerate(objects):
        e = out[i]
        _object_entry(e, o, cx, cy, me, roster_ids, tick, ctx,
                      world.weapons_tables)
        e[F] = float(KNOW_SEEN if id(o) in seen_objs
                     else KNOW_RECORD if id(o) in known_objs else KNOW_UNKNOWN)
        mask[i], uid[i] = 1, OBJECT_UID_BASE + k
        i += 1
    # -- the doors (type 4): public --
    wired = fv.wired
    for d in doors:
        _door_entry(out[i], d, u, cx, cy, ctx, world.gmap, wired)
        mask[i], uid[i] = 1, DOOR_UID_BASE + int(d.ordinal)
        i += 1


def build_full_watched(fv: FullView, team: int, u, out) -> None:
    """Section 5.5 into ``out`` (``(WIN, WIN)`` uint8): 3.5's ``watched``
    from EVERY living enemy of ``team`` at its LIVE centre, zombies
    included, dormant or not -- one ``window_clear`` per enemy, the full
    view's own."""
    cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
    y0, x0 = cy - HALF, cx - HALF
    watched = np.zeros((WIN, WIN), dtype=bool)
    for v in fv.living:
        if int(v.team) == int(team) and not v.is_zombie:
            continue
        watched |= fv.watched_from(int(v.center_tile_y()),
                                   int(v.center_tile_x()), y0, x0)
    out[...] = watched


def observe_full_rows(sim, ctx: ObsContext, units, out: dict) -> None:
    """Write :data:`FULL_KEYS` of ``out`` (the ``(U, ...)`` views) for
    ``units`` -- per row its ROSTER unit, dead or turned included (spec 5.6:
    a row with ``in_match`` 1 gets its view), or None for an empty roster
    place --, through ONE :class:`FullView` for the env: the full list,
    the match vector (one per env, copied per row) and the watched-by-
    everyone plane. The full view's door to the sim, beside
    :meth:`FullView.build`."""
    out["full_list"][...] = 0.0
    out["full_mask"][...] = 0
    out["full_uid"][...] = -1
    out["full_match"][...] = 0.0
    out["full_watched"][...] = 0
    if not any(u is not None for u in units):
        return
    fv = FullView.build(sim, ctx)
    S = ctx.n_slots
    match = fv.match(ctx)
    for i, u in enumerate(units):
        if u is None:
            continue
        team = i // S
        alive = bool(u.alive) and int(u.team) == team
        build_full_list(fv, team, u, alive, ctx, out["full_list"][i],
                        out["full_mask"][i], out["full_uid"][i])
        out["full_match"][i] = match
        build_full_watched(fv, team, u, out["full_watched"][i])


def full_shapes(n_envs: int, n_rows: int, n_teams: int) -> dict:
    """The full view's keys and shapes (spec 5.2) for the env's layout."""
    n, u = int(n_envs), int(n_rows)
    return {
        "full_list": ((n, u, FULL_ENTRIES, FULL_F), np.float32),
        "full_mask": ((n, u, FULL_ENTRIES), np.uint8),
        "full_uid": ((n, u, FULL_ENTRIES), np.int32),
        "full_match": ((n, u, 2 * int(n_teams) + 3), np.float32),
        "full_watched": ((n, u, WIN, WIN), np.uint8),
    }


# ---------------------------------------------------------------------------
# The door's body: every row of an env (spec 0.4, 4)
# ---------------------------------------------------------------------------
#: The keys ``observe_rows`` writes, in the layout's order (spec 4.1).
OBS_KEYS = ("vector", "list", "list_mask", "list_uid", "near_f", "near_u",
            "far_f", "far_u")


def observe_rows(sim, ctx: ObsContext, units, out: dict) -> None:
    """Write :data:`OBS_KEYS` of ``out`` (the ``(U, ...)`` views) for
    ``units`` (one living mate or None per row): the map's planes read
    ONCE for the env (:class:`FieldPlanes`), one :class:`TeamKnowledge` per
    marine team of the rows, then the builders. The row flags are the env's
    (spec 4.2). With :meth:`TeamKnowledge.build`, the one place the
    observation reads the sim."""
    out["vector"][...] = 0.0
    out["list"][...] = 0.0
    out["list_mask"][...] = 0
    out["list_uid"][...] = -1
    out["near_f"][...] = 0.0
    out["near_u"][...] = 0
    out["far_f"][...] = 0.0
    out["far_u"][...] = 0
    if not any(u is not None for u in units):
        return
    planes = FieldPlanes(sim.gmap, tuple(sim._doors), sim.impulse_table)
    knows = {}
    for i, u in enumerate(units):
        if u is None:
            continue
        team = int(u.team)
        know = knows.get(team)
        if know is None:
            know = knows[team] = TeamKnowledge.build(sim, sim.team_picture,
                                                     team, planes=planes)
        build_vector(know, u, ctx, out["vector"][i])
        build_list(know, u, ctx, out["list"][i], out["list_mask"][i],
                   out["list_uid"][i])
        build_windows(know, u, ctx, out["near_f"][i], out["near_u"][i],
                      out["far_f"][i], out["far_u"][i])


def shapes(n_envs: int, n_rows: int, n_slots: int) -> dict:
    """The spec's observation keys and shapes (section 4.1) for the env's
    layout: ``vector``, ``list``, ``list_mask``, ``list_uid``, the four
    window arrays (3.9) and the two row flags ``valid`` / ``in_match``."""
    e = int(n_slots) + ENTRIES_BEYOND_SLOTS
    n, u = int(n_envs), int(n_rows)
    return {
        "vector": ((n, u, V), np.float32),
        "list": ((n, u, e, F), np.float32),
        "list_mask": ((n, u, e), np.uint8),
        "list_uid": ((n, u, e), np.int32),
        "near_f": ((n, u, len(NEAR_F_PLANES), WIN, WIN), np.float16),
        "near_u": ((n, u, len(NEAR_U_PLANES), WIN, WIN), np.uint8),
        "far_f": ((n, u, len(FAR_F_PLANES), WIN, WIN), np.float16),
        "far_u": ((n, u, len(FAR_U_PLANES), WIN, WIN), np.uint8),
        "valid": ((n, u), np.uint8),
        "in_match": ((n, u), np.uint8),
    }


__all__ = [
    "SPEC_NAME", "V", "F", "VECTOR_COLUMNS", "ENTRY_COLUMNS", "DENOMINATORS",
    "FLAG", "KIND", "ANGLE", "UNIT", "COUNT", "Feature", "squash",
    "squash_array", "ratio", "d_c", "ObsContext", "ObsWorld", "TeamKnowledge",
    "TrackerView", "aim_of", "cone_q", "route", "place_plane",
    "status_scale_ticks", "build_vector", "build_list", "build_windows",
    "observe_rows", "shapes", "OBS_KEYS", "TYPE_MATE", "TYPE_SEEN",
    "TYPE_RECORD", "TYPE_OBJECT", "TYPE_DOOR", "CAPS_FIXED",
    "ENTRIES_BEYOND_SLOTS", "OBJECT_UID_BASE", "DOOR_UID_BASE",
    "HAND_GRENADE_SLOTS", "WIN", "HALF", "BLOCK", "PHYSICS_PLANES",
    "NEAR_F_PLANES", "NEAR_U_PLANES", "FAR_F_PLANES", "FAR_U_PLANES",
    "PLANE_COLUMNS", "OUTSIDE_FILL", "DOOR_NONE", "DOOR_CODE_OPEN",
    "DOOR_CODE_CLOSED", "FieldPlanes", "blocks", "crop", "inside_mask",
    "physics_value", "place_blocks",
    # the full view (section 5, patch (c))
    "FULL_VIEW_NAME", "FULL_ENTRIES", "FULL_F", "FULL_EXTRA_COLUMNS",
    "FULL_KEYS", "FULL_STATUS_KINDS", "KNOW_SEEN", "KNOW_RECORD",
    "KNOW_UNKNOWN", "full_match_columns", "FullView", "build_full_list",
    "build_full_watched", "observe_full_rows", "full_shapes",
]
