"""Weapon / ammo / payload data tables (mechanics/03 §4 — the three tables, W1).

A weapon in Breach is **a row of data, not a system** (mechanics/03). This
module is the data layer: it loads the three ``config.toml`` tables —

    [weapons.<name>]   the delivery instrument (archetype, cones, cadence, AP)
    [ammo.<name>]      the round (family, damage, dtype, speed, payload ref)
    [payloads.<name>]  what happens at the destination (blast / gas / ignition)

— into name-keyed tables mirroring :class:`simulation.gases.GasTable` /
:class:`simulation.materials.MaterialTable`: built from the ``CFG`` namespaces,
plain dict-of-dicts accepted for tests, loud ``KeyError``/``ValueError`` on
structural problems.

W1 scope (mechanics/03 §8): **data only, zero behavior change.** The three
shipped weapons are re-homed onto rows (rifle → ``k5_carbine`` +
``rifle_556_standard``, grenade → ``hand_grenade`` + ``grenade_frag`` +
``frag_standard``, door charge → ``breach_charge`` + ``demo_breach`` +
``breach_focus``) with the SAME numbers; consumers (combat.py / simulation.py /
input_handler.py) look the rows up by literal name exactly where they used to
read ``CFG.weapons.rifle.*`` etc. The unified march, spread split, crits,
payload executor, and ammo economy activate in W2/W3.

Determinism (engine/14): numeric columns are stored **as loaded** from TOML
(plain ints / floats — ingress door 2 happens at the consumption sites, which
in W1 keep their existing exact arithmetic). No RNG, no transcendentals, no
per-tick float rescale lives here. ``rof_interval_ticks`` is the one derived
column (``config.ticks_from_seconds``, the same derivation the old
``burst_interval_ticks`` config key used).

Hot-reload (engine/12 §5): construction-bound, matching the material/gas-table
precedent — the tables are (re)built at :class:`Simulation` construction/reset
via :func:`rebuild_tables`; the Ctrl+R path calls only ``CFG.reload()`` and
does NOT rebuild them (exactly as ``GameMap.reload_material_table`` exists but
is not wired to Ctrl+R). Editing a weapon row therefore needs a sim reset /
restart to take effect.
"""
from __future__ import annotations

from config import ticks_from_seconds
from simulation.damage import DAMAGE_TYPE_NAMES


# ---------------------------------------------------------------------------
# The six delivery archetypes (mechanics/03 §1) — a CLOSED set: code knows
# these; every concrete weapon is config bound to one of them. Membership
# tests only — never iterate this for anything order-dependent.
# ---------------------------------------------------------------------------
WEAPON_ARCHETYPES = frozenset(
    {"hitscan", "projectile", "lobbed", "placed", "spray", "melee"}
)

# Ammo dtype names -> the mechanics/06 damage-type ids, derived from the
# damage module's registry (single source of truth, includes the reserved
# types). Validated at load so a typo'd dtype is loud at startup, not silent
# until W2 consumes it.
DTYPE_BY_NAME = {name: dtype_id for dtype_id, name in DAMAGE_TYPE_NAMES.items()}


# ---------------------------------------------------------------------------
# Row classes — one instance per config row, readable domain names per the
# mechanics/03 §4 column tables. Plain attribute bags: values arrive as
# floats/ints from TOML and are stored as-is.
# ---------------------------------------------------------------------------
class WeaponDef:
    """One ``[weapons.<name>]`` row — the delivery instrument (mechanics/03 §4).

    Columns absent from the TOML row take the documented defaults (the tables
    are wide and heterogeneous — fuse columns only mean something on LOBBED,
    spread only on ranged), so rows stay lean. ``rof_interval_ticks`` is
    derived by :class:`WeaponTable` (0 seconds -> 0 ticks == no cadence gate;
    otherwise ``ticks_from_seconds``, identical to the old
    ``burst_interval_ticks`` derivation).
    """

    def __init__(self, name, archetype, ammo_family="none",
                 spread_deg=0.0, spread_snap_deg=0.0, range_tiles=0,
                 shots_per_trigger=1, rof_interval_seconds=0.0,
                 mag_size=0, reload_seconds=0.0, ap_cost=1,
                 crit_chance=0.0, crit_mult=2.0,
                 mass_kg=0.0, loudness=0.0,
                 max_throw_range=0, fuse_min_seconds=0.0,
                 fuse_max_seconds=0.0, fuse_default_seconds=0.0):
        self.name = name
        self.archetype = archetype              # one of WEAPON_ARCHETYPES
        self.ammo_family = ammo_family          # "none" = feeds on nothing (melee)
        self.spread_deg = spread_deg            # aimed-fire cone half-angle
        self.spread_snap_deg = spread_snap_deg  # snap/auto-fire cone (W2 split)
        self.range_tiles = range_tiles          # hard march-length cap
        self.shots_per_trigger = shots_per_trigger   # burst / pellet count
        self.rof_interval_seconds = rof_interval_seconds  # cadence gate
        self.rof_interval_ticks = 0             # derived by WeaponTable
        self.mag_size = mag_size                # 0 = ammo untracked (exactly pre-W3)
        self.reload_seconds = reload_seconds
        self.reload_ticks = 0                   # derived by WeaponTable (W3 economy)
        self.ap_cost = ap_cost                  # order cost (turn system)
        self.crit_chance = crit_chance          # §3 crit base (activates W2)
        self.crit_mult = crit_mult
        self.mass_kg = mass_kg                  # handling / melee impulse (future)
        self.loudness = loudness                # reserved — stealth layer, no consumer
        # LOBBED extras (the shipped grenade UI knobs):
        self.max_throw_range = max_throw_range
        self.fuse_min_seconds = fuse_min_seconds
        self.fuse_max_seconds = fuse_max_seconds
        self.fuse_default_seconds = fuse_default_seconds

    def __repr__(self):
        return (f"WeaponDef({self.name!r}, archetype={self.archetype!r}, "
                f"ammo_family={self.ammo_family!r})")


class AmmoDef:
    """One ``[ammo.<name>]`` row — the round (mechanics/03 §4).

    ``payload`` is an optional ``[payloads.*]`` row name ("" = the round
    carries none — plain kinetic). ``speed_tiles_per_tick`` is the W2 unified
    march's data-of-record; ``travel_speed_tiles_per_second`` is the shipped
    ``Projectile`` consumer's unit (kept so ``update_position()`` arithmetic
    stays bit-identical in W1 — see the ``grenade_frag`` config row).

    W2 columns: ``wall_damage`` — bullet chew (mechanics/03 §3): wall HP
    deposited where the round stops on a solid tile / where a cover tile
    absorbs it (and the beam's bite at its stopping solid). ``speed_q16`` is
    DERIVED — ``speed_tiles_per_tick`` quantized ONCE onto the Q16.16 grid at
    table build (ingress door 2); the march's per-tick step budget is pure
    integer arithmetic on it (``combat.BulletInFlight``).
    """

    def __init__(self, name, family, dtype, damage=0, ap=0,
                 speed_tiles_per_tick=0.0, travel_speed_tiles_per_second=0.0,
                 payload="", wall_damage=0):
        self.name = name
        self.family = family                    # must match a weapon's ammo_family
        self.dtype = dtype                      # mechanics/06 type name string
        self.dtype_id = DTYPE_BY_NAME[dtype]    # the damage-module constant
        self.damage = damage
        self.ap = ap                            # armor pierce
        self.speed_tiles_per_tick = speed_tiles_per_tick
        self.travel_speed_tiles_per_second = travel_speed_tiles_per_second
        self.payload = payload                  # "" = none
        self.wall_damage = wall_damage          # bullet chew (W2, mechanics/03 §3)
        self.speed_q16 = 0                      # derived by AmmoTable (door 2)

    def __repr__(self):
        return (f"AmmoDef({self.name!r}, family={self.family!r}, "
                f"dtype={self.dtype!r}, damage={self.damage!r})")


class PayloadDef:
    """One ``[payloads.<name>]`` row — what happens at the destination
    (mechanics/03 §4). Executed by the payload EXECUTOR
    (:func:`simulation.payloads.execute_payload`, W3) via FieldEdit / the
    physics entry points.

    The W1-finding smoke boolean SPLIT (mechanics/03 §8): ``clear_smoke``
    documents ``apply_explosion``'s built-in inner-radius smoke clearing —
    v1 keeps that clear INSIDE ``apply_explosion``, so the column is
    data-of-record, wired live when FieldEdit takes over the explosion
    internals; ``emit_blast_smoke`` is LIVE and gates the
    ``add_explosion_smoke`` textured cloud. Both must be true on
    ``frag_standard`` AND ``breach_focus``."""

    def __init__(self, name, radius=0, pressure=0.0, wall_damage=0,
                 unit_damage=0, gas_species="", gas_amount=0.0, gas_radius=0,
                 ignite_radius=0.0, ignite_intensity=0.0, clear_smoke=False,
                 emit_blast_smoke=False):
        self.name = name
        self.radius = radius                    # blast radius (tiles)
        self.pressure = pressure                # wave source magnitude
        self.wall_damage = wall_damage
        self.unit_damage = unit_damage          # BLAST packets w/ falloff
        self.gas_species = gas_species          # "" = no gas emission
        self.gas_amount = gas_amount
        self.gas_radius = gas_radius
        self.ignite_radius = ignite_radius
        self.ignite_intensity = ignite_intensity
        self.clear_smoke = clear_smoke          # data-of-record (v1: inside apply_explosion)
        self.emit_blast_smoke = emit_blast_smoke  # LIVE: gates add_explosion_smoke (W3)

    def __repr__(self):
        return (f"PayloadDef({self.name!r}, radius={self.radius!r}, "
                f"pressure={self.pressure!r})")


# ---------------------------------------------------------------------------
# Row/field accessors — mirror GasTable._get_row/_get_field: a config section
# is either a config.Namespace (attributes) or a plain dict (tests).
# ---------------------------------------------------------------------------
_REQUIRED = object()   # sentinel: no default -> missing column is a KeyError


def _iter_rows(section_cfg):
    """Yield ``(row_name, row)`` from a section Namespace or dict, in config
    order (both preserve insertion order)."""
    if isinstance(section_cfg, dict):
        return section_cfg.items()
    return vars(section_cfg).items()


def _get_field(row, section, row_name, col, default=_REQUIRED):
    """Fetch ``col`` from a row (Namespace or dict). Missing + no default is
    a loud KeyError naming the config row, mirroring GasTable._get_field."""
    if isinstance(row, dict):
        present = col in row
        value = row.get(col)
    else:
        present = hasattr(row, col)
        value = getattr(row, col, None)
    if not present:
        if default is _REQUIRED:
            raise KeyError(f"{section}.{row_name} missing column '{col}'")
        return default
    return value


# ---------------------------------------------------------------------------
# The three tables, keyed by name
# ---------------------------------------------------------------------------
class WeaponTable:
    """``[weapons.*]`` rows keyed by name. Validates every archetype against
    the closed :data:`WEAPON_ARCHETYPES` set and derives
    ``rof_interval_ticks`` from ``rof_interval_seconds`` at the given tick
    rate (the old ``burst_interval_ticks`` derivation, moved onto the row)."""

    def __init__(self, weapons_cfg, ticks_per_second):
        self.by_name: dict[str, WeaponDef] = {}
        for name, row in _iter_rows(weapons_cfg):
            archetype = str(_get_field(row, "weapons", name, "archetype"))
            if archetype not in WEAPON_ARCHETYPES:
                raise ValueError(
                    f"weapons.{name}.archetype {archetype!r} is not one of "
                    f"the six delivery archetypes {sorted(WEAPON_ARCHETYPES)} "
                    f"(mechanics/03 §1 — the set is closed)")

            def col(c, default=_REQUIRED, _row=row, _name=name):
                return _get_field(_row, "weapons", _name, c, default)

            w = WeaponDef(
                name=name,
                archetype=archetype,
                ammo_family=str(col("ammo_family", "none")),
                spread_deg=col("spread_deg", 0.0),
                spread_snap_deg=col("spread_snap_deg", 0.0),
                range_tiles=col("range_tiles", 0),
                shots_per_trigger=col("shots_per_trigger", 1),
                rof_interval_seconds=col("rof_interval_seconds", 0.0),
                mag_size=col("mag_size", 0),
                reload_seconds=col("reload_seconds", 0.0),
                ap_cost=col("ap_cost", 1),
                crit_chance=col("crit_chance", 0.0),
                crit_mult=col("crit_mult", 2.0),
                mass_kg=col("mass_kg", 0.0),
                loudness=col("loudness", 0.0),
                max_throw_range=col("max_throw_range", 0),
                fuse_min_seconds=col("fuse_min_seconds", 0.0),
                fuse_max_seconds=col("fuse_max_seconds", 0.0),
                fuse_default_seconds=col("fuse_default_seconds", 0.0),
            )
            # Cadence: 0 s = no gate = 0 ticks. Non-zero goes through the same
            # max(1, round(s * tps)) the old burst_interval_ticks key used, so
            # the k5's 0.16666667 s derives to the identical integer (4 @ 24).
            if w.rof_interval_seconds > 0:
                w.rof_interval_ticks = ticks_from_seconds(
                    w.rof_interval_seconds, ticks_per_second)
            else:
                w.rof_interval_ticks = 0
            # Reload stall (W3 ammo economy): same seconds -> integer-ticks
            # derivation (door 1). Only consulted when mag_size > 0.
            if w.reload_seconds > 0:
                w.reload_ticks = ticks_from_seconds(
                    w.reload_seconds, ticks_per_second)
            else:
                w.reload_ticks = 0
            self.by_name[name] = w
        self.names = list(self.by_name)


class AmmoTable:
    """``[ammo.*]`` rows keyed by name. ``family`` and ``dtype`` are required
    (dtype validated against the mechanics/06 names); the rest default."""

    def __init__(self, ammo_cfg):
        # Lazy import (pure-Python quantize twin, no compiled module) — keeps
        # weapons.py import-light for asset tools.
        from simulation import unit_fixed
        self.by_name: dict[str, AmmoDef] = {}
        for name, row in _iter_rows(ammo_cfg):
            dtype = str(_get_field(row, "ammo", name, "dtype"))
            if dtype not in DTYPE_BY_NAME:
                raise ValueError(
                    f"ammo.{name}.dtype {dtype!r} is not a mechanics/06 "
                    f"damage type {sorted(DTYPE_BY_NAME)}")
            a = AmmoDef(
                name=name,
                family=str(_get_field(row, "ammo", name, "family")),
                dtype=dtype,
                damage=_get_field(row, "ammo", name, "damage", 0),
                ap=_get_field(row, "ammo", name, "ap", 0),
                speed_tiles_per_tick=_get_field(
                    row, "ammo", name, "speed_tiles_per_tick", 0.0),
                travel_speed_tiles_per_second=_get_field(
                    row, "ammo", name, "travel_speed_tiles_per_second", 0.0),
                payload=str(_get_field(row, "ammo", name, "payload", "")),
                wall_damage=_get_field(row, "ammo", name, "wall_damage", 0),
            )
            # speed_q16: the authored tiles-per-tick quantized ONCE onto the
            # Q16.16 grid (ingress door 2) — the unified march's integer
            # step-budget source (W2). 96.0 -> 6291456 exactly.
            a.speed_q16 = unit_fixed.quantize_scalar(
                float(a.speed_tiles_per_tick))
            self.by_name[name] = a
        self.names = list(self.by_name)


class PayloadTable:
    """``[payloads.*]`` rows keyed by name. Every column defaults (a payload
    row lists only what it does — 'rest zero/empty'). A nonempty
    ``gas_species`` is validated against the canonical gas-name set
    (simulation.gases.GAS_NAMES — the ``gmap.gas`` slice vocabulary) so a
    typo'd species is loud at startup, not at the first detonation."""

    def __init__(self, payloads_cfg):
        # Lazy import (module constants only, mirrors AmmoTable's unit_fixed).
        from simulation.gases import GAS_NAMES
        valid_gases = set(GAS_NAMES.values())
        self.by_name: dict[str, PayloadDef] = {}
        for name, row in _iter_rows(payloads_cfg):
            def col(c, default, _row=row, _name=name):
                return _get_field(_row, "payloads", _name, c, default)

            gas_species = str(col("gas_species", ""))
            if gas_species and gas_species not in valid_gases:
                raise ValueError(
                    f"payloads.{name}.gas_species {gas_species!r} is not a "
                    f"known gas (engine/05 §6.2): {sorted(valid_gases)}")
            self.by_name[name] = PayloadDef(
                name=name,
                radius=col("radius", 0),
                pressure=col("pressure", 0.0),
                wall_damage=col("wall_damage", 0),
                unit_damage=col("unit_damage", 0),
                gas_species=gas_species,
                gas_amount=col("gas_amount", 0.0),
                gas_radius=col("gas_radius", 0),
                ignite_radius=col("ignite_radius", 0.0),
                ignite_intensity=col("ignite_intensity", 0.0),
                clear_smoke=bool(col("clear_smoke", False)),
                emit_blast_smoke=bool(col("emit_blast_smoke", False)),
            )
        self.names = list(self.by_name)


# ---------------------------------------------------------------------------
# The convenience bundle: all three tables + cross-table validation
# ---------------------------------------------------------------------------
class WeaponsTables:
    """The three tables built together, with the cross-refs validated:

    - every ammo row's ``family`` is accepted by at least one weapon;
    - every weapon with ``ammo_family != "none"`` has at least one ammo row
      to feed on (``"none"`` — melee — requires none);
    - every ammo ``payload`` ref (when present) resolves in the payload table.

    Rebuild after a config hot-reload the same way as ``GasTable`` — i.e. at
    :class:`Simulation` construction/reset (:func:`rebuild_tables`); Ctrl+R
    alone does not rebuild (engine/12 §5, the material-table precedent).
    """

    def __init__(self, weapons_cfg, ammo_cfg, payloads_cfg, ticks_per_second):
        self.weapons = WeaponTable(weapons_cfg, ticks_per_second)
        self.ammo = AmmoTable(ammo_cfg)
        self.payloads = PayloadTable(payloads_cfg)
        self._validate_cross_refs()

    def _validate_cross_refs(self):
        accepted_families = {w.ammo_family
                             for w in self.weapons.by_name.values()
                             if w.ammo_family != "none"}
        ammo_families = {a.family for a in self.ammo.by_name.values()}

        for a in self.ammo.by_name.values():
            if a.family not in accepted_families:
                raise ValueError(
                    f"ammo.{a.name}.family {a.family!r} is not accepted by "
                    f"any weapon (no [weapons.*] row has ammo_family="
                    f"{a.family!r})")
            if a.payload and a.payload not in self.payloads.by_name:
                raise ValueError(
                    f"ammo.{a.name}.payload {a.payload!r} does not resolve: "
                    f"no [payloads.{a.payload}] row")

        for w in self.weapons.by_name.values():
            if w.ammo_family != "none" and w.ammo_family not in ammo_families:
                raise ValueError(
                    f"weapons.{w.name}.ammo_family {w.ammo_family!r} has no "
                    f"ammo rows (no [ammo.*] row with family="
                    f"{w.ammo_family!r}); use ammo_family=\"none\" for "
                    f"weapons that feed on nothing")

    def payload_for_ammo(self, ammo_name):
        """Resolve an ammo row's payload ref to its :class:`PayloadDef`.
        Loud KeyError if the round carries none."""
        a = self.ammo.by_name[ammo_name]
        if not a.payload:
            raise KeyError(f"ammo.{ammo_name} carries no payload ref")
        return self.payloads.by_name[a.payload]

    def ammo_for_weapon(self, weapon):
        """Resolve the round a weapon fires (W2 dispatch): the FIRST ammo row
        in table (config) order whose ``family`` matches the weapon's
        ``ammo_family``. Deterministic — dicts preserve insertion order, and
        the cross-ref validation guarantees at least one row exists. Real
        per-unit ammo SELECTION (AP rounds, incendiary shells) is the W3
        economy; until then every family has one standard round and this is
        it. Accepts a :class:`WeaponDef` or a weapon name. Loud KeyError for
        ``ammo_family == "none"`` (melee feeds on nothing)."""
        if isinstance(weapon, str):
            weapon = self.weapons.by_name[weapon]
        if weapon.ammo_family == "none":
            raise KeyError(
                f"weapons.{weapon.name} has ammo_family='none' — no round to "
                f"resolve (melee)")
        for a in self.ammo.by_name.values():
            if a.family == weapon.ammo_family:
                return a
        raise KeyError(   # unreachable after _validate_cross_refs; stay loud
            f"no [ammo.*] row with family={weapon.ammo_family!r}")

    @classmethod
    def from_config(cls, cfg=None):
        """Build all three from the global :data:`config.CFG` (or a provided
        config object with ``.weapons`` / ``.ammo`` / ``.payloads`` /
        ``.clock.ticks_per_second``)."""
        if cfg is None:
            from config import CFG
            cfg = CFG
        return cls(cfg.weapons, cfg.ammo, cfg.payloads,
                   cfg.clock.ticks_per_second)


# ---------------------------------------------------------------------------
# Module-level access — the W1 consumer shape. CFG itself is a module-global
# singleton read at call time by all the shipped consumers; the tables are a
# pure, config-static projection of it, so a module-level cache is exactly as
# deterministic. Simulation._reset_internal calls rebuild_tables() so a fresh
# facade always reflects the current CFG (mirroring GameMap rebuilding the
# material/gas tables at construction).
# ---------------------------------------------------------------------------
_TABLES: WeaponsTables | None = None


def get_tables() -> WeaponsTables:
    """The shared tables, built lazily from CFG on first use."""
    global _TABLES
    if _TABLES is None:
        _TABLES = WeaponsTables.from_config()
    return _TABLES


def rebuild_tables() -> WeaponsTables:
    """Rebuild the shared tables from the live CFG (call after CFG.reload()
    / at Simulation construction). Returns the fresh bundle."""
    global _TABLES
    _TABLES = WeaponsTables.from_config()
    return _TABLES


__all__ = [
    "WEAPON_ARCHETYPES", "DTYPE_BY_NAME",
    "WeaponDef", "AmmoDef", "PayloadDef",
    "WeaponTable", "AmmoTable", "PayloadTable", "WeaponsTables",
    "get_tables", "rebuild_tables",
]
