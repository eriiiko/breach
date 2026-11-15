"""THE device tables of the unit kernels (#29 P6.2; residency design §4.11 and
§4.4b, "Re-check at P6.2's start", item 1).

Every TABLE value a device unit kernel reads -- a dial of the unit impulse
law, the turning law's class dials, a stability, a status kind's flags, the
exposure decay, a class's full health, the zombie team, a gas plane's index --
is ONE ROW of :data:`ROWS`: its name, its width (an int, or a callable reading
the config table it spans, as the store's block widths do), the function that
reads its value(s) from the SAME validated object the reference law reads
(``sim.impulse_table``, ``Simulation._marine_turn_rate_q16``, the status
registry, ``exposure.get_settings()``, the species and env-profile tables,
``unit_traits.class_hp_q`` ...) -- never re-derived from config -- and the
range the kernels' integer arithmetic was proved for (``lo`` .. ``hi``,
refused by name at :func:`values` when a value leaves it).

The rows of one env are one int64 vector of width ``T`` (:func:`layout`); a
batch is ``(N, T)``, one row per env, because some tables are quantized PER
MAP (the impulse table: its tile size). C++ reads the offsets ONLY from the
GENERATED, checked-in ``cpp/src/unit_tables_layout.h`` (:func:`header_text`,
written by ``tools/gen_unit_layout.py --write`` beside the store's layout;
``tests/test_unit_tables.py`` refuses a stale copy, and ``DeviceStore``
refuses a compiled module whose tables fingerprint is not this schema's). The
same header carries the per-unit constants a kernel needs by name: each hook's
index in the ``traits`` block (``UNIT_HOOK_<NAME>``) and each status kind's
id (``UNIT_STATUS_<NAME>``) -- they are part of the fingerprint.

A NEW TABLE VALUE A KERNEL READS IS: one row here (from the reference's own
object, with its proved range), ``tools/gen_unit_layout.py --write``, commit
the header, rebuild ``cpp/build_cuda``. A group-D unit value (a stability, a
class hp) is reached on the device through the group-S column that selects
its row (``env_profile_name``, ``species_id``, ``is_zombie``), never stored
per unit.

NOT PART OF THE PLAYED TICK: read by the unit store's driver
(``unit_store.DeviceStore``) for the Class-T kernels only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from simulation import unit_fixed

ONE = unit_fixed.FP_ONE
I64_MIN = -(2 ** 63)
I64_MAX = 2 ** 63 - 1
#: The generated header (C++'s copy of the table layout).
HEADER_REL = "cpp/src/unit_tables_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"


# ---------------------------------------------------------------------------
# The sources (each the object a reference law reads)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableSource:
    """What one env's rows are read from: the sim's own per-map objects and
    the process-wide validated tables."""
    sim: Any

    @property
    def impulse(self):
        return self.sim.impulse_table


def _profiles():
    from simulation.loadouts import get_tables
    return get_tables().env_profiles.by_name


def _species():
    from simulation.species import SPECIES_REGISTRY
    return SPECIES_REGISTRY


def _status_registry():
    from simulation.status import STATUS_REGISTRY
    return STATUS_REGISTRY


def _q(v) -> int:
    return unit_fixed.quantize_scalar(float(v))


# The status flag bits (``status_flags``): what a kind TAKES AWAY (an AND
# composition on the device is an OR of these) and whether it lays the unit
# prone.
FLAG_NO_MOVE, FLAG_NO_ACT, FLAG_NO_AIM, FLAG_PRONE = 1, 2, 4, 8
#: ``status_stacking`` codes.
STACKING_CODES = {"refresh": 0, "stack": 1, "max": 2}


def _status_flags(src):
    out = []
    for r in _status_registry():
        f = 0
        f |= 0 if r.can_move else FLAG_NO_MOVE
        f |= 0 if r.can_act else FLAG_NO_ACT
        f |= 0 if r.can_aim else FLAG_NO_AIM
        f |= FLAG_PRONE if r.is_prone else 0
        out.append(f)
    return tuple(out)


def _status_stacking(src):
    return tuple(STACKING_CODES[r.stacking] for r in _status_registry())


def _stability_profiles(src):
    return tuple(_q(getattr(p, "stability", 1.0))
                 for p in _profiles().values())


def _stability_species(src):
    return tuple(_q(getattr(s.environment, "stability", 1.0))
                 for s in _species().values())


def _stability_zombie(src):
    from simulation.species import ZOMBIE_STABILITY
    return _q(ZOMBIE_STABILITY)


def _hp_class(zombie):
    def f(src):
        from simulation.unit_traits import class_hp_q
        return class_hp_q(zombie)
    return f


def _zombie_team(src):
    from simulation.factions import ZOMBIE_TEAM
    return int(ZOMBIE_TEAM)


def _gas_index(name):
    def f(src):
        from simulation import gases
        return int(getattr(gases, name))
    return f


def _exposure_decay(src):
    from simulation import exposure
    return int(exposure.get_settings().decay_q)


def _stack_cap(src):
    from simulation.unit_schema import STATUS_STACK_CAP
    return int(STATUS_STACK_CAP)


def _n_kinds():
    return len(_status_registry())


def _n_profiles():
    return len(_profiles())


def _n_species():
    return len(_species())


def _n_materials():
    from simulation.materials import MATERIAL_NAMES
    return len(MATERIAL_NAMES)


def _material_mobility(src):
    """The mobility column of the env's own materials table (``GameMap.
    materials``: what ``is_passable_block`` / ``footprint_mobility`` read)."""
    return tuple(int(m) for m in src.sim.gmap.materials.mobility)
# -- #29 P6.3's sources -----------------------------------------------------------
def _n_dtypes():
    from simulation.damage import N_DAMAGE_TYPES
    return int(N_DAMAGE_TYPES)


def _zombie_mitigation():
    from simulation.species import ZOMBIE_MITIGATION
    return ZOMBIE_MITIGATION


def _per_species(f):
    out = []
    for s in _species().values():
        out.extend(int(x) for x in f(s))
    return tuple(out)


def _per_profile(f):
    out = []
    for p in _profiles().values():
        out.extend(int(x) for x in f(p))
    return tuple(out)


def _expo():
    from simulation import exposure
    return exposure.get_settings()


def _resist_of(env):
    """exposure.resist_for's non-zombie read of an environment profile."""
    from simulation.exposure import NO_RESIST
    return tuple(getattr(env, "expose_resist", NO_RESIST) or NO_RESIST)


def _status_causes(src):
    from simulation.events import CAUSE_NAMES
    return tuple(-1 if r.dtype is None else CAUSE_NAMES.index(r.name)
                 for r in _status_registry())


def _cfg():
    from config import CFG
    return CFG


def _heat_dials_q():
    from simulation import exchange
    return exchange.heat_dials_q(exchange.heat_dials(_cfg()))


def _heat_fine_bits(src):
    from simulation import optics_fixed
    return int(optics_fixed.sweep_fine_bits(src.sim.gmap))


def _trace_q(v):
    from simulation import gas_fixed
    return int(gas_fixed.quantize_trace_scalar(float(v)))


def _poison_mod():
    from simulation import poison
    return poison


def _poison():
    return _poison_mod().get_settings()


def _poison_dose_den(src):
    from simulation import gas_fixed
    return int(gas_fixed.TRACE_ONE) * int(src.sim._tps)


def _footprint_rowmajor(src):
    for s in _species().values():
        want = tuple((dx, dy) for dy in range(3) for dx in range(3))
        if tuple(tuple(o) for o in s.default_offsets) != want:
            return 0          # refused by the row's range (lo = hi = 1)
    return 1


# -- #29 P6.4a's sources (slot 1b's verbs, the done kernel) -----------------------
def _weapon_rows(src):
    """The sim's weapon rows in the unit schema's ``weapons`` table order (a
    ``weapon_id`` / ``magazines_weapon`` column's row index); refused when the
    sim's tables do not hold exactly those names in that order."""
    from simulation.unit_schema import TABLES
    names = tuple(TABLES["weapons"]())
    rows = src.sim.weapons_tables.weapons.by_name
    if tuple(rows) != names:
        raise ValueError("unit tables: the sim's weapon rows are not the unit "
                         "schema's weapons table")
    return [rows[n] for n in names]


def _n_weapons():
    from simulation.unit_schema import TABLES
    return len(TABLES["weapons"]())


def _hand_row(src):
    return src.sim.weapons_tables.weapons.by_name["hand_grenade"]


def _hand_fuse(attr):
    def f(src):
        from simulation.weapons import fuse_ticks_from_seconds
        return int(fuse_ticks_from_seconds(getattr(_hand_row(src), attr),
                                           src.sim._tps))
    return f


def _plant_class_ticks(src):
    from simulation.weapons import PLANT_WEAPON
    return int(src.sim.weapons_tables.weapons.by_name[PLANT_WEAPON].plant_ticks)  # traits-class: the device tables upload the CLASS value; the kernel applies HANDLING through unit_lib::handling_ticks


def _charge_capacity(src):
    from simulation.unit_schema import CHARGE_CAPACITY
    return int(CHARGE_CAPACITY)


def _match_limit_ticks(src):
    """``MatchRules.limit_ticks`` of the sim's ruleset; 0 when the sim plays
    no match (its ruleset has no time limit)."""
    from simulation.ruleset import MatchRules
    rules = getattr(src.sim, "ruleset", None)
    if not isinstance(rules, MatchRules):
        return 0
    return int(rules.limit_ticks(src.sim))


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableRow:
    """One table value (or a fixed run of them) a kernel reads: ``value``
    reads it from the reference's own object (``src``: a
    :class:`TableSource`); ``lo`` .. ``hi`` is the range the kernels'
    arithmetic was proved for (refused by name outside it)."""
    name: str
    width: int | Callable[[], int]
    value: Callable[[TableSource], Any]
    lo: int
    hi: int
    note: str = ""

    def w(self) -> int:
        return self.width() if callable(self.width) else int(self.width)


_V = 1 << 22          # a velocity bound: every square of a sum of two < 2^47
_IMP = lambda attr: (lambda src: getattr(src.impulse, attr))  # noqa: E731

ROWS: tuple[TableRow, ...] = (
    # -- the unit impulse law (unit_impulse.ImpulseTable, per map) ----------
    TableRow("impulse_cap_q", 1, _IMP("cap_q"), 1, _V,
             "one tick of standing hold, Q16 tiles/tick"),
    TableRow("impulse_cap_prone_q", 1, _IMP("cap_prone_q"), 1, _V,
             "one tick of a prone body's friction"),
    TableRow("impulse_zombie_hold_q16", 1, _IMP("zombie_hold_q16"), 0,
             1 << 20, "the zombie state's hold multiplier, Q16"),
    TableRow("impulse_v_inplace_q", 3, _IMP("v_inplace_q"), 1, _V,
             "(fwd, side, back) in-place capacity"),
    TableRow("impulse_v_fall_q", 3, _IMP("v_fall_q"), 1, _V,
             "(fwd, side, back) fall limit at stability 1"),
    TableRow("impulse_react_ticks", 1, _IMP("react_ticks"), 0, 1 << 16,
             "the reaction delay after an overflow"),
    TableRow("impulse_n_rec", 1, _IMP("n_rec"), 1, 256,
             "the recovery horizon (n_rec * margin squared fits int64)"),
    TableRow("impulse_inertia_g", 1, _IMP("inertia_g"), 1, 1 << 40,
             "yaw inertia, gram-tile^2"),
    TableRow("impulse_getup_ticks", 1, _IMP("getup_ticks"), 1, 1 << 20,
             "the knockdown's duration"),
    TableRow("impulse_wind_k_q", 1, _IMP("wind_k_q"), 0, 1 << 40,
             "the wind's impulse per unit of load, standing (Q16)"),
    TableRow("impulse_wind_k_prone_q", 1, _IMP("wind_k_prone_q"), 0,
             1 << 40, "... prone"),
    # -- the turning law's class dials (the TURN hook's class values) -------
    TableRow("turn_rate_class_q16", 1,
             lambda src: src.sim._marine_turn_rate_q16, 1, 1 << 18,  # traits-class: the device tables upload the CLASS value; the kernel applies TURN through unit_lib::scaled
             "[marine] turn_rate_deg_per_s per tick, Q16 radians"),
    TableRow("turn_accel_class_q16", 1,
             lambda src: src.sim._marine_turn_accel_q16, 1, 1 << 18,  # traits-class: as above
             "[marine] turn_accel_rad_per_s2 per tick^2, Q16 radians"),
    # -- stability (the HOLD hook's class value: exchange._stability_for) ---
    TableRow("stability_zombie_q16", 1, _stability_zombie, 0, 1 << 20,
             "species.ZOMBIE_STABILITY, quantized"),
    TableRow("stability_profile_q16", _n_profiles, _stability_profiles, 0,
             1 << 20, "per [env_profiles.*] row (env_profile_name's index)"),
    TableRow("stability_species_q16", _n_species, _stability_species, 0,
             1 << 20, "per species (species_id's index): a unit with no "
             "env profile reads its species' environment"),
    # -- the status registry ---------------------------------------------------
    TableRow("status_flags", _n_kinds, _status_flags, 0, 15,
             "per kind: FLAG_NO_MOVE | FLAG_NO_ACT | FLAG_NO_AIM | FLAG_PRONE"),
    TableRow("status_stacking", _n_kinds, _status_stacking, 0, 2,
             "per kind: 0 refresh, 1 stack, 2 max"),
    TableRow("status_stack_cap", 1, _stack_cap, 1, 1 << 10,
             "unit_schema.STATUS_STACK_CAP"),
    # -- exposure ---------------------------------------------------------------
    TableRow("exposure_decay_q", 1, _exposure_decay, 0, ONE,
             "[damage.exposure]'s per-tick decay, Q16"),
    # -- the class's full health (the HP hook's class values) ----------------
    TableRow("hp_class_marine_q", 1, _hp_class(False), 1, 1 << 40,
             "unit_traits.class_hp_q(False)"),
    TableRow("hp_class_zombie_q", 1, _hp_class(True), 1, 1 << 40,
             "unit_traits.class_hp_q(True)"),
    # -- identities -------------------------------------------------------------
    TableRow("zombie_team", 1, _zombie_team, 0, I64_MAX,
             "factions.ZOMBIE_TEAM"),
    TableRow("gas_o2", 1, _gas_index("O2"), 0, 255,
             "the O2 bulk plane's index in GameMap.gas"),
    TableRow("gas_n2", 1, _gas_index("INERT_N2"), 0, 255,
             "the inert N2 bulk plane's index in GameMap.gas"),
    # -- #29 P6.2m: slot 3, movement (the walk) -------------------------------
    # The [speeds] table quantized for the map (sim.speed_table, the object
    # Simulation._step_move_dir reads): the marine's walk / full speed as Q16
    # ticks per tile (the WALK / SPRINT hooks' class values; the kernel applies
    # the hooks inside default_speed's rounding) and the direction factors.
    TableRow("speed_walk_cost_q", 1,
             lambda src: src.sim.speed_table.walk_cost_q, ONE + 1, 1 << 26,  # traits-class: the device tables upload the CLASS cost; the kernel applies WALK inside default_speed
             "[speeds] marine_walk_mps as Q16 ticks per tile"),
    TableRow("speed_full_cost_q", 1,
             lambda src: src.sim.speed_table.full_cost_q, ONE + 1, 1 << 26,  # traits-class: as above, SPRINT
             "[speeds] marine_full_mps as Q16 ticks per tile"),
    TableRow("speed_side_q", 1, lambda src: src.sim.speed_table.side_q, 1, ONE,
             "[speeds] sideways_frac, Q16"),
    TableRow("speed_back_q", 1, lambda src: src.sim.speed_table.back_q, 1, ONE,
             "[speeds] backward_frac, Q16"),
    TableRow("impulse_effort_edge_q", 1, _IMP("effort_edge_q"), 0, ONE,
             "the effort's slow edge k_slow / k_hold (effort_scale_q16)"),
    TableRow("material_mobility", _n_materials, _material_mobility,
             -(1 << 20), 1 << 20,
             "per material id: the materials table's mobility column (milli-"
             "units; > 0 enterable), the one GameMap.is_passable_block and "
             "footprint_mobility project GameMap.material through"),
    # == #29 P6.3: the damage pipeline (damage.apply_packet) and the slots 2b,
    # 9c, 9c3 on it. Every value from the object the reference law reads. ====
    # -- mitigation (damage.mitigation_for: the zombie state's, else the
    #    species'), Q16 per damage type --------------------------------------
    TableRow("mitig_armor_zombie_q", lambda: _n_dtypes(),
             lambda src: _zombie_mitigation().armor_q, 0, 1 << 40,
             "species.ZOMBIE_MITIGATION.armor_q"),
    TableRow("mitig_resist_zombie_q", lambda: _n_dtypes(),
             lambda src: _zombie_mitigation().resist_q, 0, 1 << 20,
             "species.ZOMBIE_MITIGATION.resist_q (<= 16: the product fits)"),
    TableRow("mitig_armor_species_q", lambda: _n_species() * _n_dtypes(),
             lambda src: _per_species(lambda s: s.mitigation.armor_q), 0,
             1 << 40, "per species (species_id's index) x damage type"),
    TableRow("mitig_resist_species_q", lambda: _n_species() * _n_dtypes(),
             lambda src: _per_species(lambda s: s.mitigation.resist_q), 0,
             1 << 20, "per species x damage type"),
    # -- exposure (simulation.exposure: scale_q, raise_amount, the cap) -----
    TableRow("expose_k_q", 1, lambda src: _expo().k_q, 0, 1 << 22,
             "[damage.exposure] k_expose, Q16"),
    TableRow("expose_x_max_q", 1, lambda src: _expo().x_max_q, 0, 1 << 40,
             "[damage.exposure] x_max, Q16"),
    TableRow("expose_resist_zombie_q", lambda: _n_dtypes(),
             lambda src: tuple(_expo().zombie_resist), 0, ONE,
             "[zombie] expose_resist (exposure.resist_for on is_zombie)"),
    TableRow("expose_resist_profile_q", lambda: _n_profiles() * _n_dtypes(),
             lambda src: _per_profile(lambda p: _resist_of(p)), 0, ONE,
             "per [env_profiles.*] row x damage type"),
    TableRow("expose_resist_species_q", lambda: _n_species() * _n_dtypes(),
             lambda src: _per_species(lambda s: _resist_of(s.environment)), 0,
             ONE, "per species x damage type (a unit with no env profile)"),
    # -- the status registry's damage over time (status.tick_statuses) -------
    TableRow("status_dtype", _n_kinds,
             lambda src: tuple(-1 if r.dtype is None else int(r.dtype)
                               for r in _status_registry()), -1, 63,
             "per kind: the DoT / HoT damage type, -1 = none"),
    TableRow("status_cause", _n_kinds, _status_causes, -1, 63,
             "per kind: its cause's index in events.CAUSE_NAMES (an emitting "
             "row's name), -1 = no emission"),
    # -- the heat row (exchange.apply_environmental_damage / felt_temperature_q)
    TableRow("heat_gain_q", 1, lambda src: _heat_dials_q()[0], 0, 1 << 36,
             "flux_to_temp * absorption * (1 - reflectivity), Q16"),
    TableRow("heat_ambient_q", 1, lambda src: _heat_dials_q()[1],
             -(1 << 40), 1 << 40, "[combat] heat_ambient_ref, Q16"),
    TableRow("heat_overtemp_q", 1,
             lambda src: _q(_cfg().combat.heat_overtemp_scale), 0, 1 << 31,
             "[combat] heat_overtemp_scale, Q16"),
    TableRow("heat_temp_max_profile_q", _n_profiles,
             lambda src: _per_profile(lambda p: (_q(p.temperature_max),)),
             -(1 << 40), 1 << 40, "per [env_profiles.*] row: temperature_max"),
    TableRow("heat_temp_max_species_q", _n_species,
             lambda src: _per_species(
                 lambda s: (_q(s.environment.temperature_max),)),
             -(1 << 40), 1 << 40, "per species: its environment's"),
    TableRow("heat_rate_profile_q", _n_profiles,
             lambda src: _per_profile(
                 lambda p: (_q(p.environmental_damage_rate),)), 0, 1 << 31,
             "per [env_profiles.*] row: environmental_damage_rate"),
    TableRow("heat_rate_species_q", _n_species,
             lambda src: _per_species(
                 lambda s: (_q(s.environment.environmental_damage_rate),)),
             0, 1 << 31, "per species: its environment's"),
    TableRow("heat_tps", 1, lambda src: int(src.sim._tps), 1, 1 << 16,
             "the sim's ticks per second (the row's dt)"),
    TableRow("heat_fine_bits", 1, _heat_fine_bits, 0, 40,
             "optics_fixed.sweep_fine_bits(gmap): rad_flux_sweep's currency "
             "(per map)"),
    # -- the gas rows (exchange.apply_teargas_blind / apply_poison_dose) ----
    TableRow("gas_teargas", 1, _gas_index("TEARGAS"), 0, 255,
             "the teargas trace plane's index in GameMap.gas"),
    TableRow("gas_poison", 1, _gas_index("POISON"), 0, 255,
             "the poison trace plane's index (also its credit channel)"),
    TableRow("teargas_thresh_q", 1,
             lambda src: _trace_q(_cfg().exchange.teargas_blind_density), 1,
             (1 << 31) - 1, "[exchange] teargas_blind_density, TRACE currency"),
    TableRow("teargas_blind_ticks", 1,
             lambda src: int(_cfg().exchange.teargas_blind_ticks), 1, 1 << 20,
             "[exchange] teargas_blind_seconds in ticks"),
    TableRow("poison_thresh_q", 1,
             lambda src: _trace_q(_cfg().exchange.poison_min_density), 1,
             (1 << 31) - 1, "[exchange] poison_min_density, TRACE currency"),
    TableRow("poison_gas_q", 1, lambda src: _poison().gas_q, 0,
             (1 << 31) - 1, "[damage.poison] gas_units_per_second, Q16"),
    TableRow("poison_dose_den", 1, _poison_dose_den, 1, 1 << 40,
             "TRACE_ONE x ticks per second (the dose's one rational)"),
    # -- the poison law (simulation.poison) ----------------------------------
    TableRow("poison_onset_class_q", 1,
             lambda src: _poison().class_onset_q, 1, ONE,  # traits-class: the device tables upload the CLASS value; the kernel applies TOX_ONSET through unit_lib::scaled
             "a_onset, Q16 per tick"),
    TableRow("poison_decay_class_q", 1,
             lambda src: _poison().class_decay_q, 1, ONE,  # traits-class: as above (TOX_DECAY)
             "a_decay, Q16 per tick"),
    TableRow("poison_harm_class_q", 1,
             lambda src: _poison().class_harm_q, 1, 1 << 28,  # traits-class: as above (TOX_HARM)
             "hp_per_unit, Q16 (the door: < 4096 points)"),
    TableRow("poison_x_max", 1, lambda src: _poison_mod().X_MAX, 1,
             (1 << 31) - 1, "poison.X_MAX: the body's store"),
    TableRow("poison_flush_q", 1, lambda src: _poison_mod().FLUSH_Q, 0, ONE,
             "poison.FLUSH_Q"),
    # -- the footprint the device walks ---------------------------------------
    TableRow("footprint_rowmajor", 1, _footprint_rowmajor, 1, 1,
             "1 = every species' default offsets are the footprint square in "
             "row-major order (unit_schema._d_offsets for every footprint), "
             "the tile order a kernel's reduction walks; refused otherwise"),
    # -- #29 P6.4a: slot 1b's verbs and the done kernel -----------------------
    TableRow("weapon_mag_size", _n_weapons,
             lambda src: tuple(int(w.mag_size) for w in _weapon_rows(src)),
             0, 1 << 20, "per weapons-table row: mag_size (0 = untracked)"),
    TableRow("weapon_reload_ticks", _n_weapons,
             lambda src: tuple(int(w.reload_ticks) for w in _weapon_rows(src)),  # traits-class: the device tables upload the CLASS value; the kernel applies HANDLING through unit_lib::handling_ticks
             0, 1 << 16, "per weapons-table row: the class reload stall, ticks "
             "(the HANDLING hook's operand: x 2^19 < 2^35)"),
    TableRow("hand_windup_class_ticks", 1,
             lambda src: int(_hand_row(src).throw_windup_ticks), 0, 1 << 16,  # traits-class: the device tables upload the CLASS value; the kernel applies HANDLING through unit_lib::handling_ticks
             "[weapons.hand_grenade] throw_windup_ticks (HANDLING's operand)"),
    TableRow("hand_max_reach_q", 1,
             lambda src: int(_hand_row(src).max_throw_range) << 16, 0, 1 << 26,
             "int(max_throw_range) << 16: the throw's reach clamp, Q16 tiles"),
    TableRow("hand_fuse_lo_ticks", 1, _hand_fuse("fuse_min_seconds"), 0,
             1 << 24, "the hand grenade's fuse range, ticks (the sim's tps)"),
    TableRow("hand_fuse_hi_ticks", 1, _hand_fuse("fuse_max_seconds"), 0,
             1 << 24, "the hand grenade's fuse range, ticks"),
    TableRow("plant_class_ticks", 1, _plant_class_ticks, 0, 1 << 16,
             "the plant row's (weapons.PLANT_WEAPON) plant_ticks"),
    TableRow("charge_capacity", 1, _charge_capacity, 1, 1 << 10,
             "unit_schema.CHARGE_CAPACITY: live charges a sim holds"),
    TableRow("shove_q", 1, _IMP("shove_q"), 0, 1 << 32,
             "the shove's impulse, Q16 gram-tiles/tick (x the PUSH hook <= "
             "8 ONE < 2^35: its axis split below 2^51, its squared magnitude "
             "taken in 128 bits)"),
    TableRow("shove_cone_q", 1, _IMP("shove_cone_q"), 0, 205887,
             "the shove's half-angle, Q16 radians (<= PI)"),
    TableRow("shove_cooldown_ticks", 1, _IMP("shove_cooldown_ticks"), 1,
             1 << 20, "the shove's cooldown, ticks"),
    TableRow("impulse_r_body_q", 1, _IMP("r_body_q"), 0, 1 << 20,
             "the body radius every lever is clamped to, Q16 tiles"),
    TableRow("match_limit_ticks", 1, _match_limit_ticks, 0, 1 << 40,
             "MatchRules.limit_ticks (0 = the sim plays no match)"),
)  # (#29 P6.4b's block is appended to ROWS below, after its sources)


# == #29 P6.4b: the topology seam (cpp/src/cuda_topology.cuh: on_tile_changed,
# destroy_wall) -- every value from the env's own GameMap (its materials table,
# its ambient seed, its boundary), per map. A float-valued plane's value is the
# float32's exact bit pattern (the one numpy stores), never re-derived. ======
_I32_MIN, _I32_MAX = -(2 ** 31), 2 ** 31 - 1


def _mats(src):
    return src.sim.gmap.materials


def _per_mat(f):
    """Per material id: ``f(table, i)``, an int (or a tuple, flattened)."""
    def g(src):
        tbl = _mats(src)
        out = []
        for i in range(_n_materials()):
            v = f(tbl, i)
            out.extend(v if isinstance(v, tuple) else (v,))
        return tuple(out)
    return g


def _f32_bits(x) -> int:
    """The int32 bit pattern of ``x`` stored into a float32 plane."""
    return int(np.array([x], dtype=np.float32).view(np.int32)[0])


def _wall_hp_q(tbl, i):
    from simulation import wall_fixed
    return int(wall_fixed.quantize_scalar(float(tbl.hp[i])))


def _mat_solid(tbl, i):
    # on_tile_changed: solid = (the float32 permeability plane <= 0.0)
    return int(bool(np.float32(float(tbl.permeability[i])) <= 0.0))


def _mat_id(name):
    def f(src):
        from simulation import materials
        return int(getattr(materials, name))
    return f


def _ambient_seed(k):
    def f(src):
        return int(src.sim.gmap.ambient_seed()[k])
    return f


_TOPO_ROWS: tuple = (
    TableRow("topo_mat_air", 1, _mat_id("MAT_AIR"), 0, 127,
             "materials.MAT_AIR: destroy_wall's gate and its new material"),
    TableRow("topo_mat_hull", 1, _mat_id("MAT_HULL"), 0, 127,
             "materials.MAT_HULL: the edge-hull breach"),
    TableRow("topo_boundary_ambient", 1,
             lambda src: int(src.sim.gmap._boundary == "ambient"), 0, 1,
             "1 = the map's breach mask is is_ambient (planetside), 0 = "
             "is_vacuum (space)"),
    TableRow("topo_amb_n_total_q", 1, _ambient_seed(0), 1, _I32_MAX,
             "GameMap.ambient_seed()'s n_total_q: the destruction seed's "
             "constant total (an int32 gas store)"),
    TableRow("topo_amb_o2_q", 1, _ambient_seed(1), 0, _I32_MAX,
             "ambient_seed()'s o2_q: the composition fallback"),
    TableRow("topo_amb_pin_q", 1, _ambient_seed(3), 0, _I32_MAX,
             "ambient_seed()'s pin_q: a destroyed tile's atmosphere"),
    TableRow("topo_t_amb_q", 1, lambda src: int(src.sim.gmap.gas_t_amb_raw()),
             1, _I32_MAX, "GameMap.gas_t_amb_raw(): T_AMB_K raw (the mint)"),
    TableRow("mat_light_atten_bits", lambda: 3 * _n_materials(),
             _per_mat(lambda t, i: tuple(_f32_bits(t.light_atten[i][c])
                                         for c in range(3))),
             _I32_MIN, _I32_MAX, "per material x channel: light_atten's "
             "float32 bits"),
    TableRow("mat_heat_atten_bits", _n_materials,
             _per_mat(lambda t, i: _f32_bits(float(t.heat_atten[i]))),
             _I32_MIN, _I32_MAX, "per material: heat_atten's float32 bits"),
    TableRow("mat_heat_atten_q", _n_materials,
             _per_mat(lambda t, i: int(t.heat_atten_q16[i])), 0, ONE,
             "per material: heat_atten_q16"),
    TableRow("mat_light_atten_q", lambda: 3 * _n_materials(),
             _per_mat(lambda t, i: tuple(int(t.light_atten_q16[i][c])
                                         for c in range(3))), 0, ONE,
             "per material x channel: light_atten_q16"),
    TableRow("mat_flammable", _n_materials,
             _per_mat(lambda t, i: int(bool(t.flammable[i]))), 0, 1,
             "per material: flammable"),
    TableRow("mat_wall_hp_q", _n_materials, _per_mat(_wall_hp_q), _I32_MIN,
             _I32_MAX, "per material: wall_fixed.quantize_scalar(hp)"),
    TableRow("mat_conductivity_bits", _n_materials,
             _per_mat(lambda t, i: _f32_bits(float(t.conductivity[i]))),
             _I32_MIN, _I32_MAX, "per material: conductivity's float32 bits"),
    TableRow("mat_heat_inv_shift", _n_materials,
             _per_mat(lambda t, i: int(t.heat_inv_shift[i])), -64, 64,
             "per material: heat_inv_shift"),
    TableRow("mat_thermal_solid", _n_materials,
             _per_mat(lambda t, i: int(bool(t.thermal_solid[i]))), 0, 1,
             "per material: thermal_solid"),
    TableRow("mat_fuel_recip", _n_materials,
             _per_mat(lambda t, i: int(t.fuel_recip[i])), 0, I64_MAX,
             "per material: fuel_recip (an int64 plane)"),
    TableRow("mat_fire_t_ext_q", _n_materials,
             _per_mat(lambda t, i: int(t.fire_T_ext_q16[i])), _I32_MIN,
             _I32_MAX, "per material: fire_T_ext_q16"),
    TableRow("mat_fuel_per_o2_q", _n_materials,
             _per_mat(lambda t, i: int(t.fuel_per_o2_q16[i])), _I32_MIN,
             _I32_MAX, "per material: fuel_per_o2_q16"),
    TableRow("mat_face_shift", lambda: _n_materials() ** 2,
             _per_mat(lambda t, i: tuple(int(t.face_shift_table[i][j])
                                         for j in range(_n_materials()))),
             _I32_MIN, _I32_MAX, "face_shift_table[i][j], row-major"),
    TableRow("mat_no_face", 1, lambda src: int(_mats(src).no_face), _I32_MIN,
             _I32_MAX, "materials.no_face (the grid edge's face)"),
    TableRow("mat_permeability_bits", _n_materials,
             _per_mat(lambda t, i: _f32_bits(float(t.permeability[i]))),
             _I32_MIN, _I32_MAX, "per material: permeability's float32 bits"),
    TableRow("mat_wave_absorb_bits", _n_materials,
             _per_mat(lambda t, i: _f32_bits(float(t.wave_absorb[i]))),
             _I32_MIN, _I32_MAX, "per material: wave_absorb's float32 bits"),
    TableRow("mat_solid", _n_materials, _per_mat(_mat_solid), 0, 1,
             "per material: the solid predicate on its float32 permeability"),
    TableRow("mat_blocks_sight", _n_materials,
             _per_mat(lambda t, i: int(bool(t.blocks_sight[i]))), 0, 1,
             "per material: blocks_sight (the sight screen, when not solid)"),
)
ROWS = ROWS + _TOPO_ROWS


# == #29 P7a.1: slot 4a, the melee verb (combat.process_melee_intents /
# melee_arc_target / melee_blow; verb_gates.can_melee) -- per weapons-table
# row (the schema's order, the melee_id column's index), from the sim's own
# WeaponDef rows (WeaponTable derived every one of these once, at load) and
# the impulse table (per map). ==============================================
def _melee_col(f):
    return lambda src: tuple(int(f(w)) for w in _weapon_rows(src))


def _melee_push(src):
    tab = src.impulse.melee_push_q  # traits-class: the device tables upload the CLASS value; the kernel applies PUSH through unit_lib::scaled
    return tuple(int(tab.get(w.name, 0)) for w in _weapon_rows(src))


_MELEE_ROWS: tuple = (
    TableRow("weapon_melee_damage_q", _n_weapons,
             _melee_col(lambda w: int(w.melee_damage) << 16), 0, 1 << 24,  # traits-class: the device tables upload the CLASS value; the kernel applies STRIKE through unit_lib::scaled
             "per weapons-table row: unit_traits.strike_damage_q's base, "
             "melee_damage << 16 (no crit: the loadouts door refuses one on a "
             "melee row; x the STRIKE hook <= 2^19 < 2^43)"),
    TableRow("weapon_melee_dtype", _n_weapons,
             _melee_col(lambda w: -1 if w.melee_dtype_id is None
                        else w.melee_dtype_id), -1, 63,
             "per weapons-table row: melee_dtype_id (-1 none)"),
    TableRow("weapon_melee_status_kind", _n_weapons,
             _melee_col(lambda w: -1 if w.status_kind_id is None
                        else w.status_kind_id), -1, 63,
             "per weapons-table row: the delivery-site status kind (-1 none)"),
    TableRow("weapon_melee_status_ticks", _n_weapons,
             _melee_col(lambda w: w.status_ticks), 0, 1 << 24,
             "per weapons-table row: the status's derived duration, ticks"),
    TableRow("weapon_melee_arc_half_q", _n_weapons,
             _melee_col(lambda w: w.melee_arc_half_q), 0, 205887,
             "per weapons-table row: melee_arc_half_q, Q16 radians (<= PI)"),
    TableRow("weapon_rof_ticks", _n_weapons,
             _melee_col(lambda w: w.rof_interval_ticks), 0, 1 << 30,
             "per weapons-table row: rof_interval_ticks (can_melee's cadence)"),
    TableRow("weapon_melee_crit", _n_weapons,
             _melee_col(lambda w: float(w.crit_chance) > 0.0), 0, 1,  # p3e-door: a flag, the sign of the row's crit_chance
             "per weapons-table row: 1 when crit_chance > 0 (melee_blow would "
             "draw: the verb's device form refuses such a striker by name)"),
    TableRow("weapon_melee_push_q", _n_weapons, _melee_push, 0, 1 << 40,
             "per weapons-table row: the impulse table's melee_push_q (0 none), "
             "Q16 gram-tiles/tick (x the PUSH hook <= 2^19; the push is refused "
             "past 2^46 so its axis split stays inside int64)"),
)
ROWS = ROWS + _MELEE_ROWS


# == #29 P7.0a: slot 9d, ignition from temperature (combat.
# apply_temperature_ignition) -- the per-material threshold from the env's own
# materials table, the gas indices from its own gas table, the two dials
# through combat.ignition_params (the conductor's one read) and the fraction's
# floor from combat.IGNITION_X_N_FLOOR, each quantized as the law does per
# call (fire_fixed.quantize_scalar). =========================================
_I62 = 1 << 62


def _ign_q(f):
    def g(src):
        from simulation import fire_fixed
        return int(fire_fixed.quantize_scalar(float(f())))
    return g


def _ign_params():
    from simulation import combat
    return combat.ignition_params(_cfg())


def _ign_floor():
    from simulation import combat
    return combat.IGNITION_X_N_FLOOR


def _ign_gas(name):
    return lambda src: int(src.sim.gmap.gases.name_to_id[name])


_IGNITION_ROWS: tuple = (
    TableRow("ign_temp_q", _n_materials,
             _per_mat(lambda t, i: int(t.ignition_temp_q16[i])), -_I62, _I62,
             "per material: ignition_temp_q16 (int64; compared with the int32 "
             "temperature widened)"),
    TableRow("ign_x_ext_q", 1, _ign_q(lambda: _ign_params()[0]), -_I62, _I62,
             "quantize(o2_frac_ext): the O2 mole fraction the tile must exceed"),
    TableRow("ign_seed_q", 1, _ign_q(lambda: _ign_params()[1]), _I32_MIN, _I32_MAX,
             "quantize(ignition_seed): the fire a seeded tile gets (int32 plane)"),
    TableRow("ign_x_n_floor_q", 1, _ign_q(_ign_floor), 1, _I32_MAX,
             "quantize(IGNITION_X_N_FLOOR): the fraction's denominator floor "
             "(> 0: the divide is always defined)"),
    TableRow("ign_gas_o2", 1, _ign_gas("o2"), 0, 255,
             "the map's gases.name_to_id['o2'] (the plane the law reads)"),
    TableRow("ign_gas_n2", 1, _ign_gas("inert_n2"), 0, 255,
             "the map's gases.name_to_id['inert_n2']"),
)
ROWS = ROWS + _IGNITION_ROWS


# == #29 P7a.2: slot 4's shooting and slot 2''s rounds (combat._directional_fire,
# fire_burst, BulletInFlight.advance, start_spray_burst_directional,
# attack_resolver's exposure / crit rolls) -- per weapons-table row and per
# ammo-table row (the unit schema's orders) from the sim's OWN WeaponDef /
# AmmoDef / PayloadDef rows (sim.weapons_tables: the level's tile size made
# the ranges and the hearing radii), the impulse table (per map), the map's
# materials (cover), the env / species profiles (the crit arcs) and the
# config doors the law quantizes per call (the zombie multiplier, the crit
# multipliers), each quantized ONCE here exactly as the law does. ==========
#: The archetype codes a weapon row carries (``weapon_archetype``); 0 = none.
ARCHETYPE_CODES = {"projectile": 1, "hitscan": 2, "spray": 3, "melee": 4,
                   "lobbed": 5, "placed": 6}


def _ammo_rows(src):
    """The sim's ammo rows in the unit schema's ``ammo`` table order;
    refused when the sim's tables do not hold exactly those names."""
    from simulation.unit_schema import TABLES
    names = tuple(TABLES["ammo"]())
    rows = src.sim.weapons_tables.ammo.by_name
    if tuple(rows) != names:
        raise ValueError("unit tables: the sim's ammo rows are not the unit "
                         "schema's ammo table")
    return [rows[n] for n in names]


def _n_ammo():
    from simulation.unit_schema import TABLES
    return len(TABLES["ammo"]())


def _weapon_ammo(src):
    """Per weapon row: the ammo row ``WeaponsTables.ammo_for_weapon`` picks
    (its index in the ammo table), -1 where it picks none."""
    from simulation.unit_schema import TABLES
    names = tuple(TABLES["ammo"]())
    out = []
    for w in _weapon_rows(src):
        try:
            a = src.sim.weapons_tables.ammo_for_weapon(w)
        except (KeyError, ValueError):
            a = None
        out.append(-1 if a is None else names.index(a.name))
    return tuple(out)


def _ammo_payload(src):
    """Per ammo row: its ``[payloads.*]`` row's index in the sim's payload
    table (``applier.payload_index``'s), -1 for a plain round."""
    names = tuple(src.sim.weapons_tables.payloads.by_name)
    return tuple(names.index(a.payload) if getattr(a, "payload", "") else -1
                 for a in _ammo_rows(src))


def _ammo_col(f):
    return lambda src: tuple(int(f(a)) for a in _ammo_rows(src))


def _ammo_impulse(attr):
    def f(src):
        tab = getattr(src.impulse, attr)
        return tuple(int(tab.get(a.name, 0)) for a in _ammo_rows(src))
    return f


def _cone_q(attr):
    def f(src):
        from simulation.combat import cone_q16_of_deg
        return tuple(int(cone_q16_of_deg(getattr(w, attr)))  # traits-class: the CLASS cone the table uploads; SPREAD is the kernel's
                     for w in _weapon_rows(src))
    return f


def _bullet_mult(src):
    from simulation.combat import _bullet_mult_q
    return int(_bullet_mult_q())


def _crit_mult(attr, default):
    def f(src):
        from simulation import attack_resolver
        return int(attack_resolver._mult_q(getattr(_cfg().combat, attr, default)))
    return f


def _arc_half(per, attr, default):
    def f(src):
        from simulation import attack_resolver
        rows = _profiles().values() if per == "profile" else \
            (s.environment for s in _species().values())
        return tuple(int(attack_resolver._half_arc_q(getattr(p, attr, default)))
                     for p in rows)
    return f


_SHOOT_ROWS: tuple = (
    TableRow("weapon_archetype", _n_weapons,
             _melee_col(lambda w: ARCHETYPE_CODES[w.archetype]), 1, 6,
             "per weapons-table row: its archetype (ARCHETYPE_CODES)"),
    TableRow("weapon_shots_per_trigger", _n_weapons,
             _melee_col(lambda w: w.shots_per_trigger), 0, 64,
             "per weapons-table row: shots_per_trigger (the burst's rounds)"),
    TableRow("weapon_range_tiles", _n_weapons,
             _melee_col(lambda w: w.range_tiles), 0, 1 << 20,
             "per weapons-table row: range_tiles at the level's tile size (a "
             "round's remaining_steps; a spray's nominal target)"),
    TableRow("weapon_cone_aim_q", _n_weapons, _cone_q("spread_deg"), 0, 205887,  # traits-class: the device tables upload the CLASS value; the kernel applies SPREAD through unit_lib::scaled
             "per weapons-table row: the combat cone door of spread_deg, the "
             "class cone when the shooter can aim (Q16 radians, <= PI)"),
    TableRow("weapon_cone_snap_q", _n_weapons, _cone_q("spread_snap_deg"), 0,  # traits-class: as above
             205887, "per weapons-table row: the cone door of spread_snap_deg, "
             "the class cone when its statuses forbid aiming"),
    TableRow("weapon_ammo", _n_weapons, _weapon_ammo, -1, 1 << 16,
             "per weapons-table row: ammo_for_weapon's row (-1 none)"),
    TableRow("weapon_crit_on", _n_weapons,
             _melee_col(lambda w: float(w.crit_chance) > 0.0), 0, 1,  # p3e-door: the march's lazy gate, the sign of crit_chance
             "per weapons-table row: 1 when crit_chance > 0 (the march draws "
             "SHOT_CRIT only then)"),
    TableRow("weapon_crit_chance_q", _n_weapons,
             _melee_col(lambda w: unit_fixed.quantize_scalar(float(w.crit_chance))),  # p3e-door: roll_crit's
             0, ONE, "per weapons-table row: quantize(crit_chance), roll_crit's "
             "per-call door"),
    TableRow("weapon_crit_mult_q", _n_weapons,
             _melee_col(lambda w: unit_fixed.quantize_scalar(float(w.crit_mult))),  # p3e-door: scale_half_away's
             0, 1 << 24, "per weapons-table row: quantize(crit_mult), "
             "scale_half_away's per-call door"),
    TableRow("weapon_loudness_tiles", _n_weapons,
             _melee_col(lambda w: getattr(w, "loudness_tiles", 0)), 0, 1 << 20,
             "per weapons-table row: loudness_tiles (the noise radius; 0 silent)"),
    TableRow("weapon_burst_ticks", _n_weapons,
             _melee_col(lambda w: getattr(w, "burst_ticks", 0)), 0, 1 << 24,
             "per weapons-table row: burst_ticks (a spray burst's length)"),
    TableRow("ammo_speed_q16", _n_ammo, _ammo_col(lambda a: a.speed_q16), 0,
             1 << 30, "per ammo row: speed_q16 (the march's budget per tick)"),
    TableRow("ammo_damage", _n_ammo, _ammo_col(lambda a: int(a.damage)), 0,
             1 << 20, "per ammo row: damage (whole points; 0 = no direct hit)"),
    TableRow("ammo_dtype", _n_ammo, _ammo_col(lambda a: a.dtype_id), 0, 63,
             "per ammo row: dtype_id"),
    TableRow("ammo_ap_q", _n_ammo,
             _ammo_col(lambda a: unit_fixed.quantize_scalar(float(a.ap))),  # p3e-door: mitigate_q's
             0, 1 << 30, "per ammo row: the AP quantized as mitigate_q does"),
    TableRow("ammo_expose_q", _n_ammo,
             _ammo_col(lambda a: getattr(a, "expose_q", 0)), 0, 1 << 30,
             "per ammo row: expose_q (the focus-fire raise a hit carries)"),
    TableRow("ammo_wall_live", _n_ammo,
             _ammo_col(lambda a: a.wall_damage > 0), 0, 1,
             "per ammo row: 1 when wall_damage > 0 (chew_wall acts)"),
    TableRow("ammo_wall_damage_q", _n_ammo,
             _ammo_col(lambda a: (wall_q(a.wall_damage) if a.wall_damage > 0
                                  else 0)), 0, 1 << 30,
             "per ammo row: wall_fixed.quantize_scalar(float(wall_damage)), "
             "chew_wall's per-call door (0 when not live)"),
    TableRow("ammo_payload", _n_ammo, _ammo_payload, -1, 1 << 16,
             "per ammo row: its payload's row in the sim's payload table (-1 "
             "a plain round)"),
    TableRow("ammo_push_q", _n_ammo, _ammo_impulse("ammo_push_q"), 0, 1 << 40,
             "per ammo row: the impulse table's ammo_push_q (0 none; per map)"),
    TableRow("ammo_recoil_q", _n_ammo, _ammo_impulse("ammo_recoil_q"), 0,
             1 << 40, "per ammo row: the impulse table's ammo_recoil_q (0 none)"),
    TableRow("impulse_twist_q16", 1, _IMP("twist_q16"), 0, 1 << 20,
             "the impulse table's twist_q16 (a round's push spin)"),
    TableRow("zombie_bullet_mult_q", 1, _bullet_mult, 0, 1 << 24,
             "combat._bullet_mult_q(): [zombie] bullet_damage_multiplier, the "
             "law's per-call door"),
    TableRow("crit_front_mult_q", 1, _crit_mult("crit_front_mult", 1.0), 0,
             1 << 24, "attack_resolver.arc_multiplier's front multiplier, Q16"),
    TableRow("crit_flank_mult_q", 1, _crit_mult("crit_flank_mult", 2.0), 0,
             1 << 24, "the flank multiplier, Q16"),
    TableRow("crit_behind_mult_q", 1, _crit_mult("crit_behind_mult", 4.0), 0,
             1 << 24, "the behind multiplier, Q16"),
    TableRow("arc_front_half_profile_q", _n_profiles,
             _arc_half("profile", "front_arc_deg", 120.0), 0, 1 << 20,
             "per env profile: half its front arc (Q16 radians)"),
    TableRow("arc_behind_half_profile_q", _n_profiles,
             _arc_half("profile", "behind_arc_deg", 90.0), 0, 1 << 20,
             "per env profile: half its behind arc"),
    TableRow("arc_front_half_species_q", _n_species,
             _arc_half("species", "front_arc_deg", 120.0), 0, 1 << 20,
             "per species (its environment): half the front arc"),
    TableRow("arc_behind_half_species_q", _n_species,
             _arc_half("species", "behind_arc_deg", 90.0), 0, 1 << 20,
             "per species: half the behind arc"),
    TableRow("mat_cover_exposure_q", _n_materials,
             _per_mat(lambda t, i: int(t.cover_exposure_q16[i])), 0, ONE,
             "per material: cover_exposure_q16 (attack_resolver."
             "cover_exposure_at; ONE = no concealment, no draw)"),
)


def wall_q(v) -> int:
    from simulation import wall_fixed
    return int(wall_fixed.quantize_scalar(float(v)))   # p3e-door: chew_wall's


ROWS = ROWS + _SHOOT_ROWS

ROW_BY_NAME = {r.name: r for r in ROWS}
assert len(ROW_BY_NAME) == len(ROWS), "a table row is declared twice"


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableColumn:
    name: str
    offset: int
    width: int


@dataclass(frozen=True)
class Layout:
    columns: tuple
    width: int
    hooks: tuple              # hook names, traits-block order
    status_kinds: tuple       # status kind names, registry order
    fingerprint: str

    def column(self, name: str) -> TableColumn:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"unit tables: no row {name!r}")


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    """THE table layout from :data:`ROWS` (cached on the widths, which read
    config tables)."""
    from simulation.unit_traits import HOOKS
    widths = tuple(r.w() for r in ROWS)
    hooks = tuple(h.name for h in HOOKS)
    kinds = tuple(r.name for r in _status_registry())
    key = (widths, hooks, kinds)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    cols, off = [], 0
    for r, w in zip(ROWS, widths):
        if w < 1:
            raise ValueError(f"unit tables: row {r.name!r} has width {w}")
        cols.append(TableColumn(r.name, off, w))
        off += w
    text = "".join(f"{c.name}:{c.offset}:{c.width}\n" for c in cols)
    text += f"T:{off}\n" + "".join(f"hook:{i}:{n}\n"
                                   for i, n in enumerate(hooks))
    text += "".join(f"status:{i}:{n}\n" for i, n in enumerate(kinds))
    text += "".join(f"arch:{v}:{k}\n" for k, v in ARCHETYPE_CODES.items())
    lay = Layout(columns=tuple(cols), width=off, hooks=hooks,
                 status_kinds=kinds,
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())
    _LAYOUT_CACHE[key] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/unit_tables_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/unit_tables.py",
        "// (#29 P6.2, residency design 4.11 / 4.4b) -- NEVER EDIT BY HAND.",
        "// The device tables of the unit kernels: one int64 vector of",
        "// UNIT_TABLES_W per env, each row read from the reference's own",
        "// validated object (the impulse table, the turning dials, the status",
        "// registry, ...). A new table value is a row of unit_tables.ROWS, then",
        f"// `{GENERATOR}` and commit; tests/test_unit_tables.py refuses",
        "// a stale copy. Also: each hook's index in the `traits` block and each",
        "// status kind's id (part of the fingerprint).",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace unit_tables {",
        "",
        f'constexpr const char* UNIT_TABLES_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int UNIT_TABLES_W = {lay.width};",
        "",
        "// <row>: its first int64 in an env's vector; <row>_W: its width.",
    ]
    for c in lay.columns:
        n = c.name.upper()
        lines.append(f"constexpr int UNIT_TAB_{n} = {c.offset};")
        lines.append(f"constexpr int UNIT_TAB_{n}_W = {c.width};")
    lines += ["", "// The hooks' indices in the unit's `traits` block "
              "(unit_traits.HOOKS order)."]
    for i, n in enumerate(lay.hooks):
        lines.append(f"constexpr int UNIT_HOOK_{n.upper()} = {i};")
    lines.append(f"constexpr int UNIT_N_HOOKS = {len(lay.hooks)};")
    lines += ["", "// The status kinds' ids (status.STATUS_REGISTRY order)."]
    for i, n in enumerate(lay.status_kinds):
        lines.append(f"constexpr int UNIT_STATUS_{n.upper()} = {i};")
    lines.append(f"constexpr int UNIT_N_STATUS_KINDS = {len(lay.status_kinds)};")
    lines += ["", "// status_flags bits; status_stacking codes.",
              f"constexpr int64_t UNIT_FLAG_NO_MOVE = {FLAG_NO_MOVE};",
              f"constexpr int64_t UNIT_FLAG_NO_ACT = {FLAG_NO_ACT};",
              f"constexpr int64_t UNIT_FLAG_NO_AIM = {FLAG_NO_AIM};",
              f"constexpr int64_t UNIT_FLAG_PRONE = {FLAG_PRONE};"]
    for k, v in STACKING_CODES.items():
        lines.append(f"constexpr int64_t UNIT_STACK_{k.upper()} = {v};")
    lines += ["", "// weapon_archetype codes (#29 P7a.2: unit_tables.ARCHETYPE_CODES)."]
    for k, v in ARCHETYPE_CODES.items():
        lines.append(f"constexpr int64_t UNIT_ARCH_{k.upper()} = {v};")
    lines += ["", "}  // namespace unit_tables", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The values
# ---------------------------------------------------------------------------
def values(sim) -> np.ndarray:
    """One env's table vector (``(T,)`` int64) read from ``sim``'s objects.
    Refused by name: a value that is not an int, a row whose value count is
    not its width, a value outside its row's proved range."""
    lay = layout()
    src = TableSource(sim)
    out = np.zeros(lay.width, dtype=np.int64)
    for r, c in zip(ROWS, lay.columns):
        v = r.value(src)
        vals = tuple(v) if isinstance(v, (tuple, list)) else (v,)
        if len(vals) != c.width:
            raise ValueError(f"unit tables: row {r.name!r} has {len(vals)} "
                             f"values, its width is {c.width}")
        for i, x in enumerate(vals):
            if isinstance(x, bool) or int(x) != x:
                raise ValueError(f"unit tables: row {r.name!r}[{i}] = {x!r} "
                                 f"is not an integer")
            x = int(x)
            if not r.lo <= x <= r.hi:
                raise ValueError(
                    f"unit tables: row {r.name!r}[{i}] = {x} is outside the "
                    f"range the unit kernels were proved for [{r.lo}, "
                    f"{r.hi}]")
            out[c.offset + i] = x
    return out


def values_batch(sims) -> np.ndarray:
    """``(N, T)``: env e's row is :func:`values` of ``sims[e]``."""
    sims = list(sims)
    out = np.zeros((len(sims), layout().width), dtype=np.int64)
    for e, s in enumerate(sims):
        out[e] = values(s)
    return out


def read(vec, name: str):
    """A row's value(s) from one env's vector (an int, or a tuple for a row
    wider than one)."""
    c = layout().column(name)
    v = [int(x) for x in np.asarray(vec)[c.offset:c.offset + c.width]]
    return v[0] if c.width == 1 else tuple(v)


__all__ = ["HEADER_REL", "GENERATOR", "TableSource", "TableRow", "ROWS",
           "ROW_BY_NAME", "FLAG_NO_MOVE", "FLAG_NO_ACT", "FLAG_NO_AIM",
           "FLAG_PRONE", "STACKING_CODES", "Layout", "layout", "header_text",
           "values", "values_batch", "read"]
