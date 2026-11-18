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


def _body_mod():
    from simulation import body_temp
    return body_temp


def _body_rates(src, env):
    """``body_temp.rates_q`` of ``env``'s capacity on this env's map."""
    b = _body_mod()
    return b.rates_q(src.sim.gmap, b.capacity_of(env))


def _stress_mod():
    from simulation import stress
    return stress


def _stress():
    return _stress_mod().get_settings()


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
    # -- #215: the body temperature (simulation.body_temp, the heat row's
    #    law) and the stress level (simulation.stress) -----------------------
    TableRow("body_a_profile_q", _n_profiles,
             lambda src: _per_profile(lambda p: (_body_rates(src, p)[0],)), 1,
             ONE, "per [env_profiles.*] row: body_temp.rates_q's a (per map)"),
    TableRow("body_a_rad_profile_q", _n_profiles,
             lambda src: _per_profile(lambda p: (_body_rates(src, p)[1],)), 0,
             ONE, "per [env_profiles.*] row: its radiated share a_rad"),
    TableRow("body_a_species_q", _n_species,
             lambda src: _per_species(
                 lambda s: (_body_rates(src, s.environment)[0],)), 1, ONE,
             "per species: its environment's a (per map)"),
    TableRow("body_a_rad_species_q", _n_species,
             lambda src: _per_species(
                 lambda s: (_body_rates(src, s.environment)[1],)), 0, ONE,
             "per species: its environment's a_rad"),
    TableRow("body_k_temp_q", 1, lambda src: _body_mod().k_temp_q(), 0,
             1 << 20, "temperature_scale k_temp_to_kelvin, Q16 (game degrees "
             "to C)"),
    TableRow("stress_per_hp_q", 1,
             lambda src: _stress().weights_q["damage"], 0, 1 << 28,
             "[damage.stress] stress_per_hp, level counts (2^-24 points) per "
             "hit point (the damage producer row)"),
    TableRow("stress_hit_fills_q", 1, lambda src: _stress_mod().HIT_FILLS_Q,
             1, 1 << 44, "stress.HIT_FILLS_Q: a hit that fills the level"),
    TableRow("stress_rate_q", 1, lambda src: _stress().rate_q, 1,
             1 << 24, "the half-life's per-tick fraction, 2^-24 "
             "(clearing.RATE_SHIFT)"),
    TableRow("stress_flush_q", 1, lambda src: _stress_mod().FLUSH_Q, 0, 1 << 24,
             "stress.FLUSH_Q"),
    TableRow("stress_max", 1, lambda src: _stress_mod().STRESS_MAX, 1,
             1 << 38, "stress.STRESS_MAX: the level's int64 bound"),
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
    TableRow("hand_release_forward_q", 1,
             lambda src: int(_hand_row(src).release_forward_q), -(1 << 24), 1 << 24,
             "#215: [weapons.hand_grenade] release_forward_m in Q16 tiles of "
             "the sim's map (combat.release_point's forward offset)"),
    TableRow("hand_release_side_q", 1,
             lambda src: int(_hand_row(src).release_side_q), -(1 << 24), 1 << 24,
             "#215: release_side_m in Q16 tiles (to the thrower's right)"),
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


# == #29 P7b: the grenades in flight and the planted charges (slot 1b's
# appends, slot 2's flight and fuse-out, charges.tick): the flight law's
# constants as combat.Projectile reads them, the throw's kind -> round, and
# every round's payload row (a PAYLOAD record's ``payload``). ==============
def _n_hand_kinds():
    from simulation.weapons import get_tables
    return len(get_tables().hand_grenade_rows)


def _ammo_rows(src):
    """The sim's ammo rows in the unit schema's ``ammo`` table order (an
    ``ammo_name`` column's row index); refused when they differ."""
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


def _hand_ammo_rows(src):
    from simulation.unit_schema import table_row
    return tuple(table_row("ammo", n) for n in src.sim.weapons_tables.hand_grenade_rows)


def _hand_speed_q(src):
    # combat.Projectile's door: weapon_tables() (the module's own table
    # object), the round's travel speed quantized once
    from simulation import combat, unit_fixed
    by = combat.weapon_tables().ammo.by_name
    return tuple(unit_fixed.quantize_scalar(  # p3e-door: Projectile's travel_speed_q
        float(by[n].travel_speed_tiles_per_second))
        for n in src.sim.weapons_tables.hand_grenade_rows)


def _ammo_payload_rows(src):
    names = list(src.sim.weapons_tables.payloads.by_name)
    return tuple(names.index(a.payload) if a.payload else -1
                 for a in _ammo_rows(src))


def _flight_eps(src):
    from simulation import combat
    return int(combat._FLIGHT_EPS_Q)


def _order_grenade(src):
    from simulation.orders import ORDER_GRENADE
    return int(ORDER_GRENADE)


def _bounce_restitution_q(src):
    return int(_hand_row(src).bounce_restitution_q)


def _bounce_max(src):
    return int(_hand_row(src).bounce_max)


def _bounce_min_q(src):
    from simulation import combat
    return int(combat.BOUNCE_MIN_Q)


def _clock_tps(src):
    from config import CFG
    return int(CFG.clock.ticks_per_second)


_OBJECT_ROWS: tuple = (
    TableRow("clock_tps", 1, _clock_tps, 1, 256,
             "[clock] ticks_per_second as Projectile.update_position reads "
             "it (CFG, per call); <= 256 keeps tps x dist < 2^35"),
    TableRow("grenade_flight_eps_q", 1, _flight_eps, 0, 1 << 20,
             "combat._FLIGHT_EPS_Q: a flight shorter than this lands at once"),
    TableRow("order_grenade", 1, _order_grenade, 0, 1 << 20,
             "orders.ORDER_GRENADE: the proj_type whose fuse-out detonates"),
    TableRow("grenade_bounce_restitution_q", 1, _bounce_restitution_q, 0, 1 << 16,
             "#215: [weapons.hand_grenade] bounce_restitution_q (Q16, 0..1) as "
             "Simulation._update_grenades reads it off the sim's own row"),
    TableRow("grenade_bounce_max", 1, _bounce_max, 0, 64,
             "#215: [weapons.hand_grenade] bounce_max; <= 64 bounds the "
             "kernel's settle loop"),
    TableRow("grenade_bounce_min_q", 1, _bounce_min_q, 0, 1 << 20,
             "#215: combat.BOUNCE_MIN_Q: a bounced leg shorter than this is "
             "not flown (the grenade rests)"),
    TableRow("hand_grenade_ammo_row", _n_hand_kinds, _hand_ammo_rows, 0, 1 << 20,
             "per throw kind (hand_grenade_rows order): its [ammo.*] row's "
             "index in the unit schema's ammo table (a grenade's ammo_name)"),
    TableRow("hand_grenade_speed_q", _n_hand_kinds, _hand_speed_q, 0, 1 << 24,
             "per throw kind: Projectile.travel_speed_q of its round, Q16 "
             "tiles per second"),
    TableRow("ammo_payload_row", _n_ammo, _ammo_payload_rows, -1, 1 << 20,
             "per [ammo.*] row (the schema's order): its payload's index in "
             "sim.weapons_tables.payloads (payload_for_ammo; -1 none: a "
             "fuse-out of it is refused as payload_for_ammo raises)"),
)
ROWS = ROWS + _OBJECT_ROWS

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


# == #29 P7a.3: slot 4b, the sprays (combat.process_sprays / deposit_spray_cone
# / spray_cone_tiles), per weapons-table row from the sim's own WeaponDef rows
# and the round each fires (WeaponsTables.ammo_for_weapon), each quantized by
# the function the law quantizes it with per call (unit_fixed.quantize_scalar,
# gas_fixed.quantize_trace_scalar, combat.cone_q16_of_deg); the spray's two
# edit source ids (combat._SRC_SPRAY_*); and THE DEVICE EDIT DOOR's policy
# clamps per field ordinal (field_edit._clamp_q of FIELD_POLICY's pair, in the
# field's currency, as field_edit._edit_door forms them). ===================
def _spray_ammo(src):
    """Per weapons-table row: the round ``ammo_for_weapon`` resolves (None
    when it raises: a melee row's ``ammo_family = "none"``)."""
    tabs = src.sim.weapons_tables
    out = []
    for w in _weapon_rows(src):
        try:
            out.append(tabs.ammo_for_weapon(w))
        except KeyError:
            out.append(None)
    return out


def _spray_gas_ids(src):
    """Per weapons-table row: the gas slice of its round's ``gas_species`` in
    the map's own gas table (-1 none). A CONSERVATIVE slice is refused by
    name here: the "gas" policy refuses it at the flush (bulk N is what
    gas_energy is denominated against), so no device row carries one."""
    gases = src.sim.gmap.gases
    out = []
    for a in _spray_ammo(src):
        if a is None or not a.gas_species:
            out.append(-1)
            continue
        gid = int(gases.name_to_id[a.gas_species])
        if bool(gases.conservative[gid]):
            raise ValueError(f"unit tables: the round {a.name!r} sprays the "
                             f"CONSERVATIVE gas {a.gas_species!r} (the gas policy "
                             f"refuses it)")
        out.append(gid)
    return tuple(out)


def _spray_cone_half(src):
    from simulation import combat
    return tuple(combat.cone_q16_of_deg(w.cone_half_angle_degrees)  # traits-class: a jet's cone, no SPREAD hook (as the law reads it)
                 for w in _weapon_rows(src))


def _spray_src(name):
    def g(src):
        from simulation import combat
        return int(getattr(combat, name))
    return g


def _edit_clamps(src):
    """Per field ordinal (field_edit.DEVICE_FIELD_ORDINALS): (flags, lo, hi)
    of the policy's default clamp in the field's currency -- flags bit 0 a
    lower bound, bit 1 an upper (pack_for_device's first pair); 0 for an
    unbounded side."""
    from simulation import field_edit as fe
    out = []
    for name in sorted(fe.FIELD_POLICY):
        pol = fe.FIELD_POLICY[name]
        pair = fe._clamp_q(pol.clamp, fe._CURRENCY[pol.dtype])
        flags = lo = hi = 0
        if pair is not None:
            if pair[0] is not None:
                flags |= 1
                lo = int(pair[0])
            if pair[1] is not None:
                flags |= 2
                hi = int(pair[1])
        out.append((flags, lo, hi))
    return out


def _n_fields():
    from simulation import field_edit as fe
    return len(fe.FIELD_POLICY)


_SPRAY_ROWS: tuple = (
    TableRow("spray_range_tiles", _n_weapons,
             _melee_col(lambda w: w.range_tiles), 0, 1 << 14,
             "per weapons-table row: range_tiles (the cone's integer range; <= "
             "2^14 keeps dot_q^2 and dist_sq * c_q^2 inside int64)"),
    TableRow("spray_cone_half_q", _n_weapons, _spray_cone_half, 0, 102944,
             "per weapons-table row: cone_q16_of_deg(cone_half_angle_degrees), "  # traits-class: a jet's cone, no SPREAD hook (the law reads the class value)
             "Q16 radians (< 90 degrees: the dot_q >= 0 gate loses nothing)"),
    TableRow("spray_has_ammo", _n_weapons,
             lambda src: tuple(int(a is not None) for a in _spray_ammo(src)), 0, 1,
             "per weapons-table row: 1 when ammo_for_weapon resolves a round "
             "(a burst on a row without one faults: the law would raise)"),
    TableRow("spray_heat_q", _n_weapons,
             lambda src: tuple(0 if a is None else unit_fixed.quantize_scalar(
                 float(a.heat_deposit)) for a in _spray_ammo(src)),  # p3e-door: deposit_spray_cone's
             0, 1 << 48,
             "per weapons-table row: quantize_scalar(ammo.heat_deposit), Q16 heat "
             "per tick at d = 0"),
    TableRow("spray_gas_q", _n_weapons,
             lambda src: tuple(0 if a is None else _trace_q(a.gas_amount)
                               for a in _spray_ammo(src)),
             0, 1 << 48,
             "per weapons-table row: quantize_trace_scalar(ammo.gas_amount), the "
             "trace currency per tick at d = 0"),
    TableRow("spray_gas_id", _n_weapons, _spray_gas_ids, -1, 255,
             "per weapons-table row: the map's gas slice of the round's "
             "gas_species (-1 none; a conservative slice refused at the read)"),
    TableRow("spray_src_heat", 1, _spray_src("_SRC_SPRAY_HEAT"), 0, 1 << 20,
             "combat._SRC_SPRAY_HEAT: a spray heat edit's source_id"),
    TableRow("spray_src_gas", 1, _spray_src("_SRC_SPRAY_GAS"), 0, 1 << 20,
             "combat._SRC_SPRAY_GAS: a spray gas edit's source_id"),
    TableRow("edit_clamp_flags", _n_fields,
             lambda src: tuple(c[0] for c in _edit_clamps(src)), 0, 3,
             "per field ordinal: the policy clamp's bounds present (bit 0 lo, "
             "bit 1 hi), the device edit door's EDIT row column"),
    TableRow("edit_clamp_lo", _n_fields,
             lambda src: tuple(c[1] for c in _edit_clamps(src)), I64_MIN, I64_MAX,
             "per field ordinal: the policy clamp's lower bound in counts (0 none)"),
    TableRow("edit_clamp_hi", _n_fields,
             lambda src: tuple(c[2] for c in _edit_clamps(src)), I64_MIN, I64_MAX,
             "per field ordinal: the policy clamp's upper bound in counts (0 none)"),
)


def _trace_q(v) -> int:
    from simulation import gas_fixed
    return int(gas_fixed.quantize_trace_scalar(float(v)))  # p3e-door: deposit_spray_cone's


ROWS = ROWS + _SPRAY_ROWS
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


# == #29 P7c.1: THE PAYLOAD FUNCTION, the disc path (payloads.execute_payload
# for a row with strength = 0, no bulk gas, no explosive: physics.
# apply_explosion, exchange.apply_blast_damage, flash_blind.apply_flash_blind,
# emit_gas, ignite_ring, deposit_heat, noise.emit, the blast log). Per
# [payloads.*] row of the sim's OWN payload table (sim.weapons_tables.
# payloads, the object execute_payload is handed), each value quantized by
# the function the law quantizes it with per call (unit_fixed /
# wall_fixed.quantize_scalar, field_edit._quantize_in into the edit's field
# currency); the edit source ids; the blast-wall materials per material id;
# the blast damage threshold; the flash's front angle; the smoke clear's
# amount; the ignition ring's heat and clamp. ================================
def _n_payloads():
    from simulation.weapons import get_tables
    return len(get_tables().payloads.by_name)


def _payload_rows(src):
    """The sim's payload rows in the process table's order (a PAYLOAD
    record's ``payload`` index is ``applier.payload_index``: the sim's own
    table); refused when the names differ."""
    from simulation.weapons import get_tables
    names = tuple(get_tables().payloads.by_name)
    rows = src.sim.weapons_tables.payloads.by_name
    if tuple(rows) != names:
        raise ValueError(f"unit tables: the sim's payload rows {tuple(rows)} are "
                         f"not the payload table's {names}")
    return tuple(rows[n] for n in names)


def _pl(f):
    return lambda src: tuple(int(f(p)) for p in _payload_rows(src))


def _pl_radius(p):
    """``int(radius)`` -- apply_explosion's ``range(-radius, radius + 1)`` and
    the blast log's ``int(payload.radius)``; refused unless the radius is an
    integer (``range`` raises on any other)."""
    r = p.radius
    if isinstance(r, bool) or int(r) != r:
        raise ValueError(f"unit tables: payloads.{p.name}.radius {r!r} is not an "
                         f"integer (apply_explosion's range() refuses it)")
    return int(r)


def _pl_q(attr):
    def f(p):
        return unit_fixed.quantize_scalar(float(getattr(p, attr)))  # p3e-door: the law's per-call float
    return f


def _pl_cur(attr, field):
    """``field_edit._quantize_in(float(attr), currency)``: the edit door's
    quantization of a DISC edit's float amount into its field's currency."""
    def f(p):
        from simulation import field_edit as fe
        return fe._quantize_in(float(getattr(p, attr)),          # p3e-door: _edit_door's
                               fe._CURRENCY[fe.FIELD_POLICY[field].dtype])
    return f


def _pl_inner(p):
    """apply_explosion's smoke-clear radius: 2/5 of ``int(radius)`` on the Q16
    grid, handed to the DISC as its exact float view, re-quantized by the
    region walk (``quantize_scalar(dequantize_scalar(q)) == q``)."""
    q = unit_fixed.div_round(int(p.radius) * 2 * unit_fixed.FP_ONE, 5)
    return unit_fixed.quantize_scalar(unit_fixed.dequantize_scalar(q))  # p3e-door: the DISC's radius


def _pl_wall(p):
    from simulation import wall_fixed
    return int(wall_fixed.quantize_scalar(float(p.wall_damage)))  # p3e-door: apply_explosion's / the blast log's


def _pl_gas_id(src):
    """Per payload row: the map's gas slice of ``gas_species`` (-1 none). A
    CONSERVATIVE slice is refused by name (the gas policy refuses it at the
    flush: no row the function applies carries one)."""
    gases = src.sim.gmap.gases
    out = []
    for p in _payload_rows(src):
        if not p.gas_species:
            out.append(-1)
            continue
        gid = int(gases.name_to_id[p.gas_species])
        if bool(gases.conservative[gid]):
            raise ValueError(f"unit tables: payloads.{p.name} emits the CONSERVATIVE "
                             f"gas {p.gas_species!r} through emit_gas (the gas "
                             f"policy refuses it)")
        out.append(gid)
    return tuple(out)


def _blast_walls(src):
    from simulation import physics
    walls = {int(m) for m in physics._BLAST_WALL_MATERIALS}
    return tuple(int(i in walls) for i in range(_n_materials()))


def _blast_threshold(src):
    """``CFG.combat.blast_damage_threshold`` as apply_blast_damage compares an
    integer damage with it (``damage >= threshold``): its ceiling (exact for
    an integer threshold); refused when not finite."""
    import math
    t = float(_cfg().combat.blast_damage_threshold)
    if not math.isfinite(t):
        raise ValueError(f"unit tables: blast_damage_threshold {t!r} is not finite")
    return int(math.ceil(t))


def _src_id(mod, name):
    def g(src):
        import importlib
        return int(getattr(importlib.import_module(mod), name))
    return g


def _ignite_heat_q(src):
    from simulation import field_edit as fe, payloads
    if not payloads.IGNITE_HEAT > 0.0:
        raise ValueError("unit tables: payloads.IGNITE_HEAT <= 0 (the ring would "
                         "enqueue no heat: the kernel always enqueues it)")
    return fe._quantize_in(float(payloads.IGNITE_HEAT), fe._CURRENCY["heat"])  # p3e-door: _edit_door's


def _ignite_clamp(src):
    """(flags, lo, hi) of ignite_ring's explicit clamp in the fire currency
    (field_edit._clamp_q, as _edit_door forms it)."""
    from simulation import field_edit as fe, payloads
    pair = fe._clamp_q(payloads.IGNITE_CLAMP, fe._CURRENCY["fire"])
    flags = lo = hi = 0
    if pair is not None:
        if pair[0] is not None:
            flags |= 1
            lo = int(pair[0])
        if pair[1] is not None:
            flags |= 2
            hi = int(pair[1])
    return (flags, lo, hi)


def _smoke_clear_q(src):
    from simulation import field_edit as fe, physics
    return fe._quantize_in(float(physics._SMOKE_CLEAR_AMOUNT),     # p3e-door: _edit_door's
                           fe._CURRENCY["gas"])


def _payload_refuse(src):
    from simulation import applier
    return tuple(applier.payload_refusal(p) for p in _payload_rows(src))


def _gas_ids_smoke(src):
    from simulation import gases
    return int(gases.SMOKE)


def _flash_front(src):
    from simulation import flash_blind
    return int(flash_blind.FRONT_Q16)


_R_MAX = 1 << 10          # a disc radius: d2 << 32 and r_q^2 stay inside int64
_PAYLOAD_ROWS: tuple = (
    TableRow("payload_refuse", _n_payloads, _payload_refuse, 0, 4,
             "per payload row: applier.payload_refusal (0 = the disc path the "
             "PAYLOAD function applies; else refused by name, nothing written)"),
    TableRow("payload_radius", _n_payloads, _pl(_pl_radius), 0, _R_MAX,
             "per payload row: int(radius), the blast disc (apply_explosion's "
             "loop, the blast log); <= 2^10"),
    TableRow("payload_radius_q", _n_payloads, _pl(_pl_q("radius")), 0, _R_MAX << 16,
             "per payload row: quantize_scalar(float(radius)): the atmosphere "
             "DISC's radius and apply_blast_damage's r_q"),
    TableRow("payload_atm_q", _n_payloads, _pl(_pl_cur("pressure", "atmosphere")),
             -(1 << 46), 1 << 46,
             "per payload row: the atmosphere DISC's amount, float(pressure) in "
             "the bulk currency (_edit_door)"),
    TableRow("payload_press_q", _n_payloads, _pl(_pl_q("pressure")), -(1 << 28), 1 << 28,
             "per payload row: quantize_scalar(float(pressure)), the wave "
             "stencil's (x 4 below 2^31: falloff_round's proved v)"),
    TableRow("payload_wall_q", _n_payloads, _pl(_pl_wall), -(1 << 30), 1 << 30,
             "per payload row: wall_fixed.quantize_scalar(float(wall_damage)): "
             "the disc's wall damage and the blast log's"),
    TableRow("payload_inner_q", _n_payloads, _pl(_pl_inner), 0, _R_MAX << 16,
             "per payload row: the smoke clear's Q16 radius (2/5 of int(radius))"),
    TableRow("payload_blast_live", _n_payloads,
             _pl(lambda p: p.unit_damage > 0), 0, 1,
             "per payload row: unit_damage > 0 (execute_payload's test on the "
             "raw value)"),
    TableRow("payload_unit_damage", _n_payloads, _pl(lambda p: int(p.unit_damage)),
             -(1 << 20), 1 << 20,
             "per payload row: int(unit_damage), apply_blast_damage's max_d"),
    TableRow("payload_loudness_tiles", _n_payloads,
             _pl(lambda p: getattr(p, "loudness_tiles", 0)), 0, 1 << 20,
             "per payload row: loudness_tiles (noise.emit's radius; 0 = silent)"),
    TableRow("payload_blind_radius_q", _n_payloads,
             _pl(lambda p: getattr(p, "blind_radius_q", 0)), -(1 << 26), _R_MAX << 16,
             "per payload row: blind_radius_q (the door's; > 0 = the flash "
             "blinds)"),
    TableRow("payload_blind_ticks", _n_payloads, _pl(lambda p: p.blind_ticks),
             0, 1 << 16, "per payload row: blind_ticks (the door's)"),
    TableRow("payload_blind_away_q", _n_payloads, _pl(lambda p: p.blind_away_q),
             0, ONE, "per payload row: blind_away_q (the door's Q16 fraction)"),
    TableRow("payload_gas_id", _n_payloads, _pl_gas_id, -1, 255,
             "per payload row: emit_gas's slice (the map's gases.name_to_id of "
             "gas_species; -1 none)"),
    TableRow("payload_gas_q", _n_payloads, _pl(_pl_cur("gas_amount", "gas")),
             -(1 << 46), 1 << 46,
             "per payload row: emit_gas's DISC amount in the trace currency"),
    TableRow("payload_gas_radius_q", _n_payloads, _pl(_pl_q("gas_radius")),
             -(1 << 26), _R_MAX << 16,
             "per payload row: quantize_scalar(float(gas_radius)), emit_gas's "
             "DISC radius and its claims' disc"),
    TableRow("payload_ignite_live", _n_payloads, _pl(lambda p: p.ignite_radius > 0),
             0, 1, "per payload row: ignite_radius > 0 (the raw test)"),
    TableRow("payload_ignite_radius_q", _n_payloads, _pl(_pl_q("ignite_radius")),
             -(1 << 26), _R_MAX << 16,
             "per payload row: quantize_scalar(float(ignite_radius)), the ring's "
             "DISC radius (flame, heat and claims)"),
    TableRow("payload_ignite_q", _n_payloads, _pl(_pl_cur("ignite_intensity", "fire")),
             -(1 << 46), 1 << 46,
             "per payload row: the ring's flame amount in the fire currency"),
    TableRow("payload_heat_live", _n_payloads,
             _pl(lambda p: getattr(p, "heat_amount", 0.0) > 0), 0, 1,
             "per payload row: heat_amount > 0 (the raw test)"),
    TableRow("payload_heat_q", _n_payloads, _pl(_pl_q("heat_amount")),
             -(1 << 46), 1 << 46,
             "per payload row: quantize_scalar(float(heat_amount)), deposit_heat's "
             "amount_q (no air read on the disc path)"),
    TableRow("payload_heat_radius_q", _n_payloads, _pl(_pl_q("heat_radius")),
             -(1 << 26), _R_MAX << 16,
             "per payload row: quantize_scalar(float(heat_radius)), the splash's "
             "DISC radius and its claims' disc"),
    TableRow("payload_src_explosion", 1, _src_id("simulation.physics", "_SRC_EXPLOSION"),
             0, 1 << 20, "physics._SRC_EXPLOSION: apply_explosion's edits' source_id"),
    TableRow("payload_src_gas", 1, _src_id("simulation.payloads", "_SRC_PAYLOAD_GAS"),
             0, 1 << 20, "payloads._SRC_PAYLOAD_GAS: emit_gas's source_id"),
    TableRow("payload_src_ignite", 1,
             _src_id("simulation.payloads", "_SRC_PAYLOAD_IGNITE"), 0, 1 << 20,
             "payloads._SRC_PAYLOAD_IGNITE: ignite_ring's source_id"),
    TableRow("payload_src_heat", 1, _src_id("simulation.payloads", "_SRC_PAYLOAD_HEAT"),
             0, 1 << 20, "payloads._SRC_PAYLOAD_HEAT: deposit_heat's source_id"),
    TableRow("payload_smoke_slice", 1, _gas_ids_smoke, 0, 255,
             "gases.SMOKE: the slice a 'smoke' edit names (pack_for_device's)"),
    TableRow("payload_smoke_clear_q", 1, _smoke_clear_q, 0, 1 << 55,
             "physics._SMOKE_CLEAR_AMOUNT in the trace currency (the clear's "
             "REMOVE amount)"),
    TableRow("payload_ignite_heat_q", 1, _ignite_heat_q, 1, 1 << 46,
             "payloads.IGNITE_HEAT in the heat currency (the ring's FLAT heat)"),
    TableRow("payload_ignite_clamp", 3, _ignite_clamp, -(1 << 46), 1 << 46,
             "payloads.IGNITE_CLAMP as (flags, lo, hi) in the fire currency "
             "(_clamp_q: the ring's own clamp)"),
    TableRow("blast_wall_mat", _n_materials, _blast_walls, 0, 1,
             "per material id: 1 when physics._BLAST_WALL_MATERIALS holds it "
             "(only those take the disc's wall damage)"),
    TableRow("blast_damage_threshold", 1, _blast_threshold, -(1 << 40), 1 << 40,
             "ceil(CFG.combat.blast_damage_threshold): apply_blast_damage's "
             "chip-damage floor (an integer damage at or above it lands)"),
    TableRow("flash_front_q16", 1, _flash_front, 1, 205887,
             "flash_blind.FRONT_Q16: the facing cone of the whole blinding"),
)
ROWS = ROWS + _PAYLOAD_ROWS


# == #29 P7c.2: THE PAYLOAD FUNCTION, the strength path, the air read, bulk gas
# and the multi-tick source (payloads.execute_payload's generic explosion:
# blast_expansion.reach_tiles / start_source / start_expansion, blast_air.
# read_blast_air, deposit_over_tiles with linear_weights, add_explosion_smoke
# with blast_smoke_peak_q, inject_bulk_gas). Every float the law converts per
# call is a row read through the law's own door function on the sim's own row
# and map (its tile size), never re-derived. ================================
def _pl_generic(p):
    return int(getattr(p, "strength", 0.0) > 0.0)


def _pl_air_mode(p):
    """execute_payload's air read: 2 over the walked heat tiles (a strength
    row), 1 over the disc (``blast_air.disc_tiles``: a row with an explosive),
    0 none (``air is None``)."""
    if _pl_generic(p):
        return 2
    return 1 if getattr(p, "explosive", "") else 0


def _pl_air_disc_r(p):
    from simulation import blast_air
    if _pl_air_mode(p) != 1:
        return 0
    return blast_air.disc_reach(max(p.radius, getattr(p, "heat_radius", 0.0)))


def _pl_expansion_live(p):
    return int(not _pl_generic(p) and getattr(p, "expansion_m3", 0.0) > 0.0)


def _pl_ticks(p):
    return max(int(p.expansion_ticks), 1) if _pl_expansion_live(p) else 1


def _pl_source_r2(p):
    from simulation import blast_expansion as bx
    if _pl_generic(p):
        return int(p.source_r2)
    return bx.expansion_r2(p) if _pl_expansion_live(p) else 0


def _pl_source_total(src):
    from simulation import blast_expansion as bx
    out = []
    for p in _payload_rows(src):
        if _pl_generic(p):
            out.append(bx.source_total_q(src.sim.gmap, p.expansion_m3, 1))
        elif _pl_expansion_live(p):
            out.append(bx.source_total_q(src.sim.gmap, p.expansion_m3, _pl_ticks(p)))
        else:
            out.append(0)
    return tuple(int(v) for v in out)


def _pl_heat_base(src):
    from simulation import payloads
    return tuple(int(payloads.heat_base_q(src.sim.gmap, p)) if _pl_generic(p) else 0
                 for p in _payload_rows(src))


def _pl_soot_units(p):
    from simulation import payloads
    return payloads.soot_units_q(p)


def _pl_bulk_slice(src):
    """inject_bulk_gas' slice (the map's ``name_to_id`` of ``bulk_gas``; -1
    none); a non-conservative one is refused by name (the reference raises)."""
    gases = src.sim.gmap.gases
    out = []
    for p in _payload_rows(src):
        if not getattr(p, "bulk_gas", ""):
            out.append(-1)
            continue
        g = int(gases.name_to_id[p.bulk_gas])
        if not bool(gases.conservative[g]):
            raise ValueError(f"unit tables: payloads.{p.name}.bulk_gas {p.bulk_gas!r} "
                             f"is not a bulk gas (inject_bulk_gas refuses it)")
        out.append(g)
    return tuple(out)


def _src_smoke(src):
    from simulation import physics
    return int(physics._SRC_EXPLOSION_SMOKE)


def _blast_smoke_noise_q(src):
    """The smoke's per-tile noise as every edit door quantizes it
    (``payloads._blast_noise``: [physics] explosion_smoke_noise clamped to
    [0, 1]; ``min(quantize_scalar(noise), ONE)``, 0 for no noise)."""
    from simulation import payloads
    noise = payloads._blast_noise()
    if not noise > 0.0:
        return 0
    return min(unit_fixed.quantize_scalar(float(noise)), ONE)   # p3e-door: _edit_door's


def _blast_air_q(src):
    from simulation import blast_air
    s = blast_air.blast_air_settings()
    if not (0 <= s.o2_low_q < s.o2_full_q and 0 <= s.p_ext_q < s.p_full_q):
        raise ValueError("unit tables: the blast air read's ramps need lo < hi "
                         f"(o2 {s.o2_low_q} / {s.o2_full_q}, p {s.p_ext_q} / {s.p_full_q}): "
                         "_ramp_q divides by hi - lo")
    return (s.o2_low_q, s.o2_full_q, s.starved_smoke_q, s.detonation_share_q,
            s.p_ext_q, s.p_full_q)


def _s_max_q(src):
    from simulation import blast_expansion
    return int(blast_expansion.S_MAX_Q)


def _bulk_rails(src):
    from simulation import vent_system
    lo, hi = vent_system._t_rails_q()
    return (int(lo), int(hi))


_STRENGTH_ROWS: tuple = (
    TableRow("payload_generic", _n_payloads, _pl(_pl_generic), 0, 1,
             "per payload row: strength > 0 (THE generic explosion: the walks, the "
             "fixed-total deposits, start_source)"),
    TableRow("payload_air_mode", _n_payloads, _pl(_pl_air_mode), 0, 2,
             "per payload row: the air read -- 2 over the walked heat tiles, 1 over "
             "blast_air.disc_tiles (an explosive), 0 none"),
    TableRow("payload_air_disc_r", _n_payloads, _pl(_pl_air_disc_r), 0, _R_MAX,
             "per payload row: blast_air.disc_reach(max(radius, heat_radius)) of a "
             "mode-1 read (the disc d2 <= r^2 of its open tiles)"),
    TableRow("payload_heat_r2", _n_payloads,
             _pl(lambda p: int(getattr(p, "heat_r2", 0)) if _pl_generic(p) else 0),
             0, 1 << 20, "per payload row: heat_r2, the heat / smoke walk's r2"),
    TableRow("payload_source_r2", _n_payloads, _pl(_pl_source_r2), 0, 1 << 20,
             "per payload row: the source walk's r2 (source_r2 of a strength row; "
             "blast_expansion.expansion_r2 of a comparison row; 0 none)"),
    TableRow("payload_source_total_q", _n_payloads, _pl_source_total, 0, 1 << 50,
             "per payload row: blast_expansion.source_total_q(gmap, expansion_m3, "
             "ticks) (1 tick for a strength row), the total the walk divides"),
    TableRow("payload_expansion_live", _n_payloads, _pl(_pl_expansion_live), 0, 1,
             "per payload row: start_expansion runs (no strength, expansion_m3 > 0)"),
    TableRow("payload_expansion_ticks", _n_payloads, _pl(_pl_ticks), 1, 1 << 20,
             "per payload row: max(int(expansion_ticks), 1) of a comparison row"),
    TableRow("payload_smoke_live", _n_payloads, _pl(lambda p: bool(p.emit_blast_smoke)),
             0, 1, "per payload row: emit_blast_smoke (a non-strength row's "
                   "add_explosion_smoke)"),
    TableRow("payload_soot_units_q", _n_payloads, _pl(_pl_soot_units), 0, 1 << 29,
             "per payload row: payloads.soot_units_q, the charge's soot in smoke units "
             "in the trace currency (< 2^29: units * 2^33 inside int64)"),
    TableRow("payload_heat_base_q", _n_payloads, _pl_heat_base, 0, 1 << 46,
             "per payload row: payloads.heat_base_q (a strength row's heat in the "
             "map's heat units, Q16; x the Q16 air read inside int64)"),
    TableRow("payload_bulk_slice", _n_payloads, _pl_bulk_slice, -1, 255,
             "per payload row: inject_bulk_gas' conservative slice (-1 none)"),
    TableRow("payload_bulk_q", _n_payloads, _pl(lambda p: int(getattr(p, "bulk_amount_q", 0))),
             -(1 << 53), 1 << 53, "per payload row: bulk_amount_q (the door's; <= 0 "
                                  "deposits nothing)"),
    TableRow("payload_bulk_radius_q", _n_payloads, _pl(_pl_q("bulk_radius")),
             -(1 << 26), _R_MAX << 16,
             "per payload row: quantize_scalar(float(bulk_radius)), bulk_gas_tiles' "
             "DISC radius"),
    TableRow("payload_bulk_temp_q", _n_payloads, _pl(lambda p: int(getattr(p, "bulk_temp_q", 0))),
             -(1 << 30) + 1, (1 << 30) - 1,
             "per payload row: bulk_temp_q (relative Q16, the deposit's t_dep)"),
    TableRow("payload_src_explosion_smoke", 1, _src_smoke, 0, 1 << 20,
             "physics._SRC_EXPLOSION_SMOKE: the blast smoke's source_id"),
    TableRow("blast_smoke_noise_q", 1, _blast_smoke_noise_q, 0, ONE,
             "the blast smoke's per-tile noise at the edit door (payloads._blast_noise "
             "quantized as _edit_door / deposit_over_tiles / blast_smoke_peak_q do)"),
    TableRow("blast_air_q", 6, _blast_air_q, 0, 1 << 30,
             "blast_air.blast_air_settings(): o2_low_q, o2_full_q, starved_smoke_q, "
             "detonation_share_q, p_ext_q, p_full_q (read once per table build, not "
             "per call)"),
    TableRow("blast_s_max_q", 1, _s_max_q, 1, (1 << 31) - 1,
             "blast_expansion.S_MAX_Q: add_source's saturation"),
    TableRow("bulk_t_rails", 2, _bulk_rails, -(1 << 30) + 1, (1 << 30) - 1,
             "vent_system._t_rails_q(): inject_bulk_gas' (T_MIN, T_MAX_PHYS) rails"),
)
ROWS = ROWS + _STRENGTH_ROWS


# == #29 P6b.1: the doors (door_system.sweep_doors), seal / unseal /
# can_seal_tiles and the gas-N primitives in the Device topology seam --
# each read from the sim's own map (its materials and gas tables) or the
# constant the reference reads ==================================================
def _n_gases():
    from simulation.gases import N_GASES
    return int(N_GASES)


def _seal_mat_ok(tbl, i):
    # _seal_blockers' check 3: refused when float(permeability[mid]) > 0.0
    # (on the TABLE's own value, not the float32 plane's: the solid predicate
    # `mat_solid` is the plane's); 1 = a material seal_tiles accepts
    return int(not float(tbl.permeability[i]) > 0.0)


def _gas_trace(src):
    # GameMap._trace_slice: a slice that is not conservative (trace currency,
    # the counted saturation)
    cons = src.sim.gmap.gases.conservative
    return tuple(int(not bool(cons[g])) for g in range(_n_gases()))


def _inject_o2_frac_q(src):
    from simulation import gas_fixed
    from simulation.gamemap import GameMap
    return int(gas_fixed.quantize_q16_scalar(GameMap.INJECT_O2_FRACTION))


def _door_states(src):
    from simulation.entities import door as door_schema
    return (int(door_schema.DOOR_CLOSED), int(door_schema.DOOR_OPEN),
            int(door_schema.DOOR_DESTROYED))


_DOOR_ROWS: tuple = (
    TableRow("door_mat_closed", 1, _mat_id("MAT_DOOR_CLOSED"), 0, 127,
             "materials.MAT_DOOR_CLOSED: a closed door's material (the sweep's "
             "external-destruction test and the seal's material)"),
    TableRow("seal_mat_ok", _n_materials, _per_mat(_seal_mat_ok), 0, 1,
             "per material: 1 = not (float(permeability) > 0.0), the material "
             "seal_tiles accepts (its float compare made once, here)"),
    TableRow("gas_trace", _n_gases, _gas_trace, 0, 1,
             "per gas slice: 1 = a trace slice (not conservative: the counted "
             "saturation; no energy)"),
    TableRow("inject_o2_frac_q", 1, _inject_o2_frac_q, 0, ONE,
             "quantize_q16_scalar(GameMap.INJECT_O2_FRACTION): inject_gas_n's "
             "fixed O2 share"),
    TableRow("door_states", 3, _door_states, 0, 2,
             "entities.door's DOOR_CLOSED, DOOR_OPEN, DOOR_DESTROYED (the door "
             "table's state codes)"),
)
ROWS = ROWS + _DOOR_ROWS


# == #29 P6b.2: the vents (vent_system.sweep_vents) -- the filter table, the
# trace slots' gas ids, the T-mix rails and the near-empty floor, each read
# from the reference's own validated object =================================
def _n_filter_cells():
    from simulation import unit_schema
    return max(1, len(unit_schema.filter_names())) * unit_schema.VENT_N_TRACE


def _vent_filter_eff_q(src):
    # vent_system._duct_filter_efficiency_q of every [filters.*] row, in
    # unit_schema.filter_names() order (the duct row's `filter` index): the
    # reference's own quantization and [0, 1] validation
    from config import CFG
    from simulation import unit_schema, vent_system
    out = []
    for name in unit_schema.filter_names():
        out.extend(vent_system._duct_filter_efficiency_q(
            {"filter": name}, getattr(CFG, "filters", None), f"<table row {name}>"))
    return tuple(out) if out else (0,) * unit_schema.VENT_N_TRACE


def _vent_trace_ids(src):
    from simulation import vent_system
    return tuple(int(g) for g in vent_system._TRACE_IDS)


def _vent_rails(src):
    # the sim's cached rails (build_vents reads them once, on a level with
    # ducts); a duct-free sim reads the same function
    from simulation import vent_system
    lo, hi = src.sim._vent_t_min_q, src.sim._vent_t_max_phys_q
    if lo is None or hi is None:
        lo, hi = vent_system._t_rails_q()
    return int(lo), int(hi)


def _vent_n_eps(src):
    from simulation import vent_system
    return int(vent_system.N_EPS)


_VENT_ROWS: tuple = (
    TableRow("vent_filter_eff_q", _n_filter_cells, _vent_filter_eff_q, 0, ONE,
             "per [filters.*] row (unit_schema.filter_names order) x trace slot: "
             "vent_system._duct_filter_efficiency_q (a Q16 fraction of removed mass)"),
    TableRow("vent_trace_ids", 5, _vent_trace_ids, 0, 63,
             "vent_system._TRACE_IDS: the gas slice of each plenum trace slot"),
    TableRow("vent_t_min_q", 1, lambda src: _vent_rails(src)[0], -(1 << 30), 1 << 30,
             "vent_system._t_rails_q()[0]: quantize_q16_scalar([physics.eos].T_MIN), "
             "the deposit's low rail"),
    TableRow("vent_t_max_q", 1, lambda src: _vent_rails(src)[1], -(1 << 30), 1 << 30,
             "vent_system._t_rails_q()[1]: quantize_q16_scalar([physics.thermal]."
             "T_MAX_PHYS), the deposit's high rail"),
    TableRow("vent_n_eps", 1, _vent_n_eps, 1, 1 << 20,
             "vent_system.N_EPS: the near-empty plenum's bulk floor (the E/N "
             "divide below it wipes the energy)"),
)
ROWS = ROWS + _VENT_ROWS


# == #29 P7d: the horde (slot 5, ai_zombie.update_zombies_tick) -- the sim's own
# HordeSettings (horde.horde_settings: [zombie] through the refusing door,
# quantized ONCE for the map), the bite's [zombie] values and the zombie
# cadences of the sim's own [speeds] table ===================================
def _hs(attr, cast=int):
    return lambda src: cast(getattr(src.sim.horde_settings, attr))


def _bash_damage_q(src):
    # combat.chew_wall's per-call door: wall_fixed.quantize_scalar(float(d)),
    # 0 for d <= 0 (the no-op; the settings door refuses d <= 0 anyway)
    from simulation import wall_fixed
    d = float(src.sim.horde_settings.door_bash_damage)
    return wall_fixed.quantize_scalar(d) if d > 0 else 0       # p3e-door: chew_wall's


def _bite_class_q(src):
    # unit_traits.strike_damage_q(z)'s class value (weapon None): the bite's
    # [zombie] melee_damage quantized once; the kernel applies the STRIKE hook
    from config import CFG
    from simulation import unit_fixed
    return unit_fixed.quantize_scalar(float(CFG.zombie.melee_damage))  # traits-class: the device tables upload the CLASS value; the kernel applies STRIKE through unit_lib::scaled  # p3e-door: strike_damage_q's


def _attack_cooldown(src):
    from config import CFG
    return int(CFG.zombie.attack_cooldown_ticks)


_R2 = 1 << 40           # a squared radius: every d2 compare stays in int64
_HORDE_ROWS: tuple = (
    TableRow("horde_trigger_r2", 1, _hs("trigger_r2"), 0, _R2,
             "HordeSettings.trigger_r2: pass 1's sight radius squared (strictly inside)"),
    TableRow("horde_propagation_r2", 1, _hs("propagation_r2"), 0, _R2,
             "HordeSettings.propagation_r2: pass 2's chain radius squared"),
    TableRow("horde_refresh_ticks", 1, _hs("field_refresh_ticks"), 1, 1 << 30,
             "HordeSettings.field_refresh_ticks: the cache drops at tick % this == 0"),
    TableRow("horde_wander", 1, _hs("wander"), 0, 1, "[zombie] wander (a switch)"),
    TableRow("horde_wander_period_ticks", 1, _hs("wander_period_ticks"), 1, 1 << 30,
             "HordeSettings.wander_period_ticks"),
    TableRow("horde_wander_leash_r2", 1, _hs("wander_leash_r2"), 0, _R2,
             "HordeSettings.wander_leash_r2"),
    TableRow("horde_hearing", 1, _hs("hearing"), 0, 1, "[zombie] hearing (a switch)"),
    TableRow("horde_noise_timeout_ticks", 1, _hs("noise_timeout_ticks"), 0, 1 << 30,
             "HordeSettings.noise_timeout_ticks"),
    TableRow("horde_noise_fields_max", 1, _hs("noise_fields_max"), 1, 1 << 20,
             "HordeSettings.noise_fields_max: the live noise targets kept"),
    TableRow("horde_noise_arrive_tiles", 1, _hs("noise_arrive_tiles"), 0, 1 << 30,
             "HordeSettings.noise_arrive_tiles"),
    TableRow("horde_fear_fire", 1, _hs("fear_fire"), 0, 1, "[zombie] fear_fire (a switch)"),
    TableRow("horde_fear_temp_q16", 1, _hs("fear_temp_q16"), 1, 1 << 31,
             "HordeSettings.fear_temp_q16: the feared temperature mirror (>=)"),
    TableRow("horde_bash_doors", 1, _hs("bash_doors"), 0, 1, "[zombie] bash_doors (a switch)"),
    TableRow("horde_bash_damage_q", 1, _bash_damage_q, 0, 1 << 31,
             "chew_wall's quantize_scalar(float(door_bash_damage)) (0: the no-op)"),
    TableRow("horde_bash_loudness_tiles", 1, _hs("bash_loudness_tiles"), 0, 1 << 20,
             "HordeSettings.bash_loudness_tiles: a strike's noise radius (0 silent)"),
    TableRow("horde_sprint", 1, _hs("sprint"), 0, 1, "[zombie] sprint (a switch)"),
    TableRow("horde_sprint_r2", 1, _hs("sprint_r2"), 0, _R2, "HordeSettings.sprint_r2"),
    TableRow("zombie_attack_cooldown_ticks", 1, _attack_cooldown, 0, 1 << 30,
             "[zombie] attack_cooldown_ticks: the bite's and the bash's cadence"),
    TableRow("zombie_bite_class_q", 1, _bite_class_q, 0, 1 << 24,
             "unit_traits.strike_damage_q's bite base: quantize_scalar([zombie] "
             "melee_damage) (x the STRIKE hook <= 2^19 < 2^43)"),
    TableRow("zombie_shamble_ticks", 1, lambda src: int(src.sim.speed_table.zombie_shamble_ticks),  # traits-class: the class cadence; the kernel applies WALK through default_speed_q
             1, 1 << 29, "SpeedTable.zombie_shamble_ticks (ticks per tile)"),  # traits-class: the class cadence (WALK applied in default_speed_q)
    TableRow("zombie_sprint_ticks", 1, lambda src: int(src.sim.speed_table.zombie_sprint_ticks),  # traits-class: the class cadence; the kernel applies SPRINT through default_speed_q
             1, 1 << 29, "SpeedTable.zombie_sprint_ticks (ticks per tile)"),  # traits-class: the class cadence (SPRINT applied in default_speed_q)
)
ROWS = ROWS + _HORDE_ROWS

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
