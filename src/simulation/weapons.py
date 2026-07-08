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

# The archetypes a FIRE order can trigger (process_shooting's dispatch).
# LOBBED / PLACED ride their own order flows (grenade / explosive modes) and
# never take a trigger pull — the playground weapon-cycle debug key cycles
# through THIS set only, so a debug-armed unit can always actually fire.
FIRE_ORDER_ARCHETYPES = frozenset({"hitscan", "projectile", "spray", "melee"})

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
                 fuse_max_seconds=0.0, fuse_default_seconds=0.0,
                 cone_half_angle_degrees=0.0, burst_seconds=0.0,
                 melee_damage=0, melee_dtype="",
                 status_kind="", status_seconds=0.0,
                 range_m=0.0, default_ammo=""):
        self.name = name
        self.archetype = archetype              # one of WEAPON_ARCHETYPES
        self.ammo_family = ammo_family          # "none" = feeds on nothing (melee)
        self.spread_deg = spread_deg            # aimed-fire cone half-angle
        self.spread_snap_deg = spread_snap_deg  # snap/auto-fire cone (W2 split)
        # RANGE (W6 — meter-based ranges, Erik's design decision 2026-07-07).
        # The AUTHORED column is range_m — a physical reach in METERS, so a
        # weapon's range no longer depends on the level's grid resolution.
        # range_tiles is DERIVED at table build from the level's tile size
        # (WeaponTable, tile_size_m):
        #
        #     range_tiles = max(1, int(range_m / tile_size_m + 0.5))
        #
        # — one correctly-rounded IEEE divide (door 3) + round-half-up to an
        # int, computed ONCE at load (door 2, the quantize-once rule); every
        # consumer (the march length, the fire-order range gate, the spray
        # cone) keeps reading the integer range_tiles exactly as before.
        # CONVENTION OF RECORD: the pinned test worlds are 1.0 m/tile, so
        # range_m there IS the old tile count (k5: 90 tiles -> 90.0 m); the
        # playground (0.333 m/tile) now derives 3x the tiles for the same
        # physical reach. Direct construction with range_tiles (the dict-table
        # test path) stays valid: range_m = 0 leaves range_tiles as passed.
        self.range_m = range_m                  # authored physical reach (m)
        self.range_tiles = range_tiles          # hard march-length cap (derived)
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
        # default_ammo (W6): the STATIC round-selection seam — "" = the
        # shipped first-family-match (ammo_for_weapon, the W2 rule); a row
        # name = THIS weapon's standard round. Lets two weapons share an
        # ammo family honestly (the P12 and MP-11 both eat 9mm but load
        # different rounds; the Lance-5 draws the heavy cell). Validated at
        # cross-ref time (row exists + family matches). Per-UNIT ammo
        # SELECTION (the loadout UI) stays future work (mechanics/03 §7).
        self.default_ammo = default_ammo        # "" = first-family-match
        # LOBBED extras (the shipped grenade UI knobs):
        self.max_throw_range = max_throw_range
        self.fuse_min_seconds = fuse_min_seconds
        self.fuse_max_seconds = fuse_max_seconds
        self.fuse_default_seconds = fuse_default_seconds
        # SPRAY extras (mechanics/03 §5, W4). CONVENTION OF RECORD: the
        # armory table (§6) quotes the FULL cone angle ("30° cone"); config
        # authors the HALF-angle (the membership test's natural quantity —
        # bearing-off-axis <= half angle), so Dragon-7's 30° cone is
        # cone_half_angle_degrees = 15.0. burst_ticks is derived by
        # WeaponTable (seconds -> integer ticks, door 1 — the reload_ticks
        # twin): one trigger = one burst = burst_ticks consecutive ticks of
        # cone deposits.
        self.cone_half_angle_degrees = cone_half_angle_degrees
        self.burst_seconds = burst_seconds
        self.burst_ticks = 0                    # derived by WeaponTable (W4)
        # MELEE extras (mechanics/03 §5, W5). Melee feeds on nothing
        # (ammo_family "none"), so the strike's packet numbers live ON THE
        # WEAPON ROW: melee_damage + melee_dtype are the DamagePacket
        # amount/type (the ammo damage/dtype twins). status_kind /
        # status_seconds are the DELIVERY-SITE status application (the §1
        # two-terminals wording: "a baton applies STUNNED where it connects;
        # packets themselves stay damage-only") — "" = the weapon applies no
        # status (the knife). status_ticks / the id twins are derived by
        # WeaponTable (seconds -> integer ticks, door 1; names -> registry
        # ids, loud on typos). Non-melee rows leave all four at their
        # 0/"" defaults — dead data.
        self.melee_damage = melee_damage        # packet amount (int, door 2)
        self.melee_dtype = melee_dtype          # mechanics/06 type name string
        self.melee_dtype_id = None              # derived by WeaponTable (W5)
        self.status_kind = status_kind          # "" = no status applied
        self.status_kind_id = None              # derived by WeaponTable (W5)
        self.status_seconds = status_seconds
        self.status_ticks = 0                   # derived by WeaponTable (W5)

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

    W4 columns (SPRAY rounds, mechanics/03 §5): ``heat_deposit`` — flame
    heat energy per tick at zero distance (the cone scales it by the linear
    falloff and it quantizes ONCE per tile at the FieldEdit heat combine);
    ``gas_species`` + ``gas_amount`` — the per-tick gas emission into the
    engine/05 §6.2 slice (falloff-scaled the same way; resolved BY NAME at
    deposit time via ``gmap.gases.name_to_id``, the emit_gas rule).
    Non-spray rounds leave all three at their 0/"" defaults — dead data.
    """

    def __init__(self, name, family, dtype, damage=0, ap=0,
                 speed_tiles_per_tick=0.0, travel_speed_tiles_per_second=0.0,
                 payload="", wall_damage=0,
                 heat_deposit=0.0, gas_species="", gas_amount=0.0,
                 glow=""):
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
        # SPRAY deposit columns (W4, mechanics/03 §5):
        self.heat_deposit = heat_deposit        # flame heat/tick at d=0 (0 = none)
        self.gas_species = gas_species          # "" = no gas emission
        self.gas_amount = gas_amount            # gas density/tick at d=0
        # glow (W6, RENDER-ONLY): a nonempty profile name makes the round's
        # in-flight march emit one ProjectileGlowEvent per tick (the
        # LaserFiredEvent precedent — the renderer draws a glowing bolt +
        # a transient light; the sim never reads this column back). The
        # plasma casters author "plasma". Pure data — event emission is a
        # pure function of already-synced state, and the determinism digest
        # hashes only UnitHit/UnitKilled events (field_ab_harness
        # _SYNCED_EVENT_TYPES), so a glowing round moves no digest.
        self.glow = glow

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
                 emit_blast_smoke=False, heat_amount=0.0, heat_radius=0.0):
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
        # Heat splash (W6 — the plasma payload): a one-shot DISC ADD of
        # ``heat_amount`` heat units (linear falloff to ``heat_radius``)
        # into the engine/06 ``heat`` ingress buffer at the detonation tile
        # (payloads.deposit_heat). The C++ TemperatureSolver converts it to
        # temperature the same tick, so the splash IGNITES through physics
        # (and cooks units via the existing heat|max row) — the SPRAY
        # two-terminals discipline applied to a detonation. 0 = none.
        self.heat_amount = heat_amount
        self.heat_radius = heat_radius

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
    rate (the old ``burst_interval_ticks`` derivation, moved onto the row).

    W6 (meter-based ranges): ``tile_size_m`` is the LEVEL's physical tile
    size (``gmap.tile_size_m`` — the Simulation passes it at construction;
    the bare default 1.0 is the pinned-test-world convention, where meters
    and tiles coincide). A row authoring ``range_m`` derives its integer
    ``range_tiles`` HERE, once, at table build — the quantize-once rule
    (engine/14 door 2): ``max(1, int(range_m / tile_size_m + 0.5))``, one
    correctly-rounded IEEE divide (door 3) + round-half-up. Authoring BOTH
    ``range_m`` and ``range_tiles`` on one row is ambiguous and loud."""

    def __init__(self, weapons_cfg, ticks_per_second, tile_size_m=1.0):
        self.tile_size_m = float(tile_size_m)
        if not self.tile_size_m > 0:
            raise ValueError(
                f"WeaponTable: tile_size_m must be > 0 (got {tile_size_m!r})")
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
                cone_half_angle_degrees=col("cone_half_angle_degrees", 0.0),
                burst_seconds=col("burst_seconds", 0.0),
                melee_damage=col("melee_damage", 0),
                melee_dtype=str(col("melee_dtype", "")),
                status_kind=str(col("status_kind", "")),
                status_seconds=col("status_seconds", 0.0),
                range_m=col("range_m", 0.0),
                default_ammo=str(col("default_ammo", "")),
            )
            # METER-BASED RANGE (W6, Erik's 2026-07-07 decision): range_m is
            # the authored physical reach; range_tiles derives ONCE here at
            # the level's tile size (quantize-once, door 2; the divide is
            # door 3; round-half-up). Rows may still author range_tiles
            # directly (the dict-table test path) — never both.
            if w.range_m and w.range_m > 0:
                if w.range_tiles:
                    raise ValueError(
                        f"weapons.{name}: authors BOTH range_m "
                        f"({w.range_m!r}) and range_tiles "
                        f"({w.range_tiles!r}) — ambiguous; author range_m "
                        f"(the W6 meter convention) and let the table "
                        f"derive the tiles")
                derived = int(float(w.range_m) / self.tile_size_m + 0.5)
                w.range_tiles = derived if derived > 1 else 1
            # SPRAY rows (W4) must author a real cone: a spray with no
            # half-angle / burst / range would deposit nothing (or forever)
            # — a config bug, loud at load rather than silent at the trigger.
            if archetype == "spray":
                if not (w.cone_half_angle_degrees > 0
                        and w.burst_seconds > 0 and w.range_tiles > 0):
                    raise ValueError(
                        f"weapons.{name}: archetype 'spray' requires "
                        f"cone_half_angle_degrees > 0, burst_seconds > 0 and "
                        f"a range (range_m or range_tiles) > 0 (got "
                        f"{w.cone_half_angle_degrees!r} / "
                        f"{w.burst_seconds!r} / {w.range_tiles!r})")
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
            # SPRAY burst length (W4): same derivation. 1.5 s -> 36 @ 24 tps.
            if w.burst_seconds > 0:
                w.burst_ticks = ticks_from_seconds(
                    w.burst_seconds, ticks_per_second)
            else:
                w.burst_ticks = 0
            # MELEE rows (W5): the strike packet lives on the weapon row
            # (melee feeds on nothing), so a melee row without damage/dtype
            # is a config bug — loud at load, not silent at the strike.
            if w.melee_dtype and w.melee_dtype not in DTYPE_BY_NAME:
                raise ValueError(
                    f"weapons.{name}.melee_dtype {w.melee_dtype!r} is not a "
                    f"mechanics/06 damage type {sorted(DTYPE_BY_NAME)}")
            if w.melee_dtype:
                w.melee_dtype_id = DTYPE_BY_NAME[w.melee_dtype]
            if archetype == "melee":
                if not (w.melee_damage > 0 and w.melee_dtype):
                    raise ValueError(
                        f"weapons.{name}: archetype 'melee' requires "
                        f"melee_damage > 0 and a melee_dtype (got "
                        f"{w.melee_damage!r} / {w.melee_dtype!r})")
            # Delivery-site status columns (W5): the name resolves against
            # the mechanics/06 §4 registry (lazy import, the AmmoTable
            # unit_fixed precedent) and the duration derives to integer
            # ticks (door 1 — the reload_ticks twin). A status with no
            # positive duration is a config bug, loud at load.
            if w.status_kind:
                from simulation.status import STATUS_REGISTRY
                kinds_by_name = {row.name: row.kind for row in STATUS_REGISTRY}
                if w.status_kind not in kinds_by_name:
                    raise ValueError(
                        f"weapons.{name}.status_kind {w.status_kind!r} is "
                        f"not a mechanics/06 §4 status kind "
                        f"{sorted(kinds_by_name)}")
                if not w.status_seconds > 0:
                    raise ValueError(
                        f"weapons.{name}: status_kind {w.status_kind!r} "
                        f"requires status_seconds > 0 (got "
                        f"{w.status_seconds!r})")
                w.status_kind_id = kinds_by_name[w.status_kind]
                w.status_ticks = ticks_from_seconds(
                    w.status_seconds, ticks_per_second)
            self.by_name[name] = w
        self.names = list(self.by_name)


class AmmoTable:
    """``[ammo.*]`` rows keyed by name. ``family`` and ``dtype`` are required
    (dtype validated against the mechanics/06 names); the rest default."""

    def __init__(self, ammo_cfg):
        # Lazy imports (pure-Python quantize twin + gas name vocabulary, no
        # compiled module) — keeps weapons.py import-light for asset tools.
        from simulation import unit_fixed
        from simulation.gases import GAS_NAMES
        valid_gases = set(GAS_NAMES.values())
        self.by_name: dict[str, AmmoDef] = {}
        for name, row in _iter_rows(ammo_cfg):
            dtype = str(_get_field(row, "ammo", name, "dtype"))
            if dtype not in DTYPE_BY_NAME:
                raise ValueError(
                    f"ammo.{name}.dtype {dtype!r} is not a mechanics/06 "
                    f"damage type {sorted(DTYPE_BY_NAME)}")
            gas_species = str(_get_field(row, "ammo", name, "gas_species", ""))
            if gas_species and gas_species not in valid_gases:
                # The PayloadTable rule (W3), applied to the W4 spray column:
                # a typo'd species is loud at startup, not at the first burst.
                raise ValueError(
                    f"ammo.{name}.gas_species {gas_species!r} is not a "
                    f"known gas (engine/05 §6.2): {sorted(valid_gases)}")
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
                heat_deposit=_get_field(row, "ammo", name, "heat_deposit", 0.0),
                gas_species=gas_species,
                gas_amount=_get_field(row, "ammo", name, "gas_amount", 0.0),
                glow=str(_get_field(row, "ammo", name, "glow", "")),
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
                heat_amount=col("heat_amount", 0.0),
                heat_radius=col("heat_radius", 0.0),
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

    def __init__(self, weapons_cfg, ammo_cfg, payloads_cfg, ticks_per_second,
                 tile_size_m=1.0):
        self.tile_size_m = float(tile_size_m)   # the W6 meter-range binding
        self.weapons = WeaponTable(weapons_cfg, ticks_per_second,
                                   tile_size_m=tile_size_m)
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
            # default_ammo (W6): must resolve, and must feed the weapon.
            if w.default_ammo:
                a = self.ammo.by_name.get(w.default_ammo)
                if a is None:
                    raise ValueError(
                        f"weapons.{w.name}.default_ammo {w.default_ammo!r} "
                        f"does not resolve: no [ammo.{w.default_ammo}] row")
                if a.family != w.ammo_family:
                    raise ValueError(
                        f"weapons.{w.name}.default_ammo {w.default_ammo!r} "
                        f"is family {a.family!r} but the weapon eats "
                        f"{w.ammo_family!r}")

    def payload_for_ammo(self, ammo_name):
        """Resolve an ammo row's payload ref to its :class:`PayloadDef`.
        Loud KeyError if the round carries none."""
        a = self.ammo.by_name[ammo_name]
        if not a.payload:
            raise KeyError(f"ammo.{ammo_name} carries no payload ref")
        return self.payloads.by_name[a.payload]

    def ammo_for_weapon(self, weapon):
        """Resolve the round a weapon fires (W2 dispatch): the weapon's
        ``default_ammo`` row when authored (W6 — the static round-selection
        seam, validated at cross-ref time), else the FIRST ammo row in table
        (config) order whose ``family`` matches the weapon's ``ammo_family``
        (the shipped W2 rule — deterministic: dicts preserve insertion
        order, and the cross-ref validation guarantees at least one row
        exists). Per-UNIT ammo SELECTION (AP rounds mid-mission) stays
        future work. Accepts a :class:`WeaponDef` or a weapon name. Loud
        KeyError for ``ammo_family == "none"`` (melee feeds on nothing)."""
        if isinstance(weapon, str):
            weapon = self.weapons.by_name[weapon]
        if weapon.ammo_family == "none":
            raise KeyError(
                f"weapons.{weapon.name} has ammo_family='none' — no round to "
                f"resolve (melee)")
        if weapon.default_ammo:
            return self.ammo.by_name[weapon.default_ammo]
        for a in self.ammo.by_name.values():
            if a.family == weapon.ammo_family:
                return a
        raise KeyError(   # unreachable after _validate_cross_refs; stay loud
            f"no [ammo.*] row with family={weapon.ammo_family!r}")

    @classmethod
    def from_config(cls, cfg=None, tile_size_m=1.0):
        """Build all three from the global :data:`config.CFG` (or a provided
        config object with ``.weapons`` / ``.ammo`` / ``.payloads`` /
        ``.clock.ticks_per_second``). ``tile_size_m`` is the level's tile
        size for the W6 meter->tile range derivation; the 1.0 default is the
        pinned-test-world convention (meters == tiles)."""
        if cfg is None:
            from config import CFG
            cfg = CFG
        return cls(cfg.weapons, cfg.ammo, cfg.payloads,
                   cfg.clock.ticks_per_second, tile_size_m=tile_size_m)


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
    """The shared tables, built lazily from CFG on first use (at the 1.0
    m/tile default binding — the pinned-test-world convention). A live
    Simulation rebuilds them at construction with ITS level's tile size
    (:func:`rebuild_tables`), so in-game consumers always see the meter
    ranges derived for the loaded level."""
    global _TABLES
    if _TABLES is None:
        _TABLES = WeaponsTables.from_config()
    return _TABLES


def rebuild_tables(tile_size_m=1.0) -> WeaponsTables:
    """Rebuild the shared tables from the live CFG (call after CFG.reload()
    / at Simulation construction). ``tile_size_m`` = the loaded level's tile
    size (W6 meter ranges — the Simulation passes ``gmap.tile_size_m``).
    Returns the fresh bundle."""
    global _TABLES
    _TABLES = WeaponsTables.from_config(tile_size_m=tile_size_m)
    return _TABLES


__all__ = [
    "WEAPON_ARCHETYPES", "FIRE_ORDER_ARCHETYPES", "DTYPE_BY_NAME",
    "WeaponDef", "AmmoDef", "PayloadDef",
    "WeaponTable", "AmmoTable", "PayloadTable", "WeaponsTables",
    "get_tables", "rebuild_tables",
]
