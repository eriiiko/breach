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

from numbers import Integral

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


#: The ammo family of a thrown grenade, and the row a bare-int loadout
#: ``grenades`` means (G3p).
HAND_GRENADE_FAMILY = "hand_grenade"
FRAG_ROW = "grenade_frag"
#: The ammo family of a planted charge, the rows a loadout's ``charge`` may
#: name (#193), and the ``[weapons.*]`` row whose ``plant_seconds`` and fuse
#: range are the plant verb's (``Simulation.plant_charge_intent``) -- as the
#: throw verb reads ``hand_grenade``'s.
CHARGE_FAMILY = "demo_charge"
PLANT_WEAPON = "c4_satchel"


def fuse_ticks_from_seconds(seconds, ticks_per_second) -> int:
    """A fuse in seconds as the integer ticks the throw verb takes: the
    truncation the projectile always used (``int(seconds * tps)``), so a
    caller converting at its own door detonates on the tick it always did."""
    return int(float(seconds) * ticks_per_second)


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
                 range_m=0.0, default_ammo="", throw_windup_seconds=0.0,
                 melee_push_ns=0.0, melee_arc_half_deg=60.0,
                 plant_seconds=0.0):
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
        # G9p: the hearing radius of a shot, METRES (simulation.noise);
        # REQUIRED on a row that fires (projectile / hitscan / spray),
        # quantized once to ``loudness_tiles`` at the table build.
        self.loudness = loudness
        self.loudness_tiles = 0
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
        # The throw's WIND-UP (gameplay pack G3p): seconds between the throw
        # verb being consumed and the grenade leaving the hand; derived to
        # integer ticks by WeaponTable (0 = the grenade leaves the same tick,
        # exactly the pre-G3p throw).
        self.throw_windup_seconds = throw_windup_seconds
        self.throw_windup_ticks = 0             # derived by WeaponTable (G3p)
        self.plant_seconds = plant_seconds
        self.plant_ticks = 0                    # derived by WeaponTable (#193)
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
        # The melee VERB's two columns (gameplay pack G6p): the blow's push on
        # the unit it hits in N s (through the unit impulse law, along the
        # striker -> target line; 0 = no push) and the half-angle in degrees
        # of the facing arc ``Simulation.melee_intent`` strikes within;
        # ``melee_arc_half_q`` (Q16 radians) is derived by WeaponTable (once).
        self.melee_push_ns = melee_push_ns
        self.melee_arc_half_deg = melee_arc_half_deg
        self.melee_arc_half_q = 0               # derived by WeaponTable (G6p)

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
                 glow="", expose=0.0):
        self.name = name
        # EXPOSURE (gameplay pack G4p, simulation.exposure): how far one hit
        # of this round raises the target's exposure of the round's dtype
        # (before the target's resistance). 0 = the round exposes nothing.
        # ``expose_q`` is the door's Q16 quantization; the law reads only it.
        self.expose = float(expose)
        self.expose_q = 0                       # derived by AmmoTable
        # MOMENTUM (gameplay pack G5p-a2, simulation.unit_impulse): the
        # round's real mass and muzzle speed (``bullet_g``, ``muzzle_mps``;
        # absent = 0, a round that pushes nothing). ``momentum_ns`` (N s =
        # bullet_g / 1000 * muzzle_mps) is what a hit pushes its target with
        # and its recoil pushes the shooter with, through the impulse
        # table's multipliers, quantized once per map there.
        self.bullet_g = 0.0
        self.muzzle_mps = 0.0
        self.momentum_ns = 0.0
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
    blast's textured cloud, and a ``strength`` row must carry it."""

    def __init__(self, name, radius=0, pressure=0.0, wall_damage=0,
                 unit_damage=0, gas_species="", gas_amount=0.0, gas_radius=0,
                 ignite_radius=0.0, ignite_intensity=0.0, clear_smoke=False,
                 emit_blast_smoke=False, heat_amount=0.0, heat_radius=0.0,
                 explosive="", explosive_kg=0.0, blast_soot_g=0.0,
                 bulk_gas="", bulk_amount=0.0, bulk_radius=0.0, bulk_temp=0.0,
                 bulk_amount_q=0, bulk_temp_q=0,
                 expansion_m3=0.0, expansion_radius=0.0, expansion_ticks=1,
                 strength=0.0, heat_mj=0.0, source_r2=0, heat_r2=0,
                 flash=0.0, flash_decay_s=0.3, flash_color=(1.0, 1.0, 1.0),
                 blind_radius=0.0, blind_seconds=0.0, blind_away_frac=1.0,
                 blind_radius_q=0, blind_ticks=0, blind_away_q=1 << 16):
        self.name = name
        # G9p: the detonation's hearing radius (metres; tiles once built).
        self.loudness = 0.0
        self.loudness_tiles = 0
        # G3p: the detonation BLINDS units (docs/gameplay_pack_v1_design
        # §3 G3p; simulation/flash_blind.py): every living unit within
        # ``blind_radius`` tiles with a line of sight to the detonation is
        # BLINDED for up to ``blind_seconds`` (quantized once to ``blind_ticks``),
        # less with distance and with its back to the flash, never below
        # ``blind_away_frac`` of it by facing alone. ``blind_radius`` 0 = this
        # row blinds nobody. The ``*_q`` / ``blind_ticks`` are the door's
        # quantizations (Q16 tiles, integer ticks, Q16 fraction): the law
        # reads only those.
        self.blind_radius = float(blind_radius)
        self.blind_seconds = float(blind_seconds)
        self.blind_away_frac = float(blind_away_frac)
        self.blind_radius_q = int(blind_radius_q)
        self.blind_ticks = int(blind_ticks)
        self.blind_away_q = int(blind_away_q)
        # G4 P5 (#75): the detonation's FLASH, a render-only row property
        # (docs/explosion_look_design_2026-11-07.md §3.5): ``flash`` light units
        # at peak (0 = none), fading as exp(-age / flash_decay_s) on the sim
        # clock, ``flash_color`` RGB in [0, 1]. Carried on the ExplosionEvent;
        # renderer/frame_lights.blast_flash_lights lights it. Never a sim field.
        self.flash = float(flash)
        self.flash_decay_s = float(flash_decay_s)
        self.flash_color = tuple(float(c) for c in flash_color)
        # #31: a row with ``strength > 0`` IS a generic explosion. The door
        # derives what the strength means from [blast]: ``expansion_m3``, the
        # air it pushes out in one tick; ``heat_mj``, its heat; ``explosive_kg``,
        # its charge. The ``*_r2`` are the squared radii its walks stop at
        # (blast_expansion.reach_tiles), integers, so walk and credit agree.
        self.strength = strength
        self.heat_mj = heat_mj
        self.source_r2 = source_r2
        self.heat_r2 = heat_r2
        # A comparison row's source (levels/blast_quad, blast_div): the air it
        # pushes out, along open floor within expansion_radius, over the ticks.
        self.expansion_m3 = expansion_m3
        self.expansion_radius = expansion_radius
        self.expansion_ticks = expansion_ticks
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
        # #12 handle 3 (2026-10-04): the blast's smoke is PHYSICAL -- the
        # charge's explosive and mass give its soot, explosive_kg x 1000 x
        # [explosives.<explosive>] soot_yield grams (resolved at table load);
        # the executor deposits exactly that mass (gases.smoke_units_of_soot).
        self.explosive = explosive
        self.explosive_kg = explosive_kg
        self.blast_soot_g = blast_soot_g
        # Heat splash (W6 — the plasma payload): a one-shot DISC ADD of
        # ``heat_amount`` heat units (linear falloff to ``heat_radius``)
        # into the engine/06 ``heat`` ingress buffer at the detonation tile
        # (payloads.deposit_heat). The C++ TemperatureSolver converts it to
        # temperature the same tick, so the splash IGNITES through physics
        # (and cooks units via the existing heat|max row) — the SPRAY
        # two-terminals discipline applied to a detonation. 0 = none.
        self.heat_amount = heat_amount
        self.heat_radius = heat_radius
        # Bulk-gas effect (#133 E.2, levelgen vocabulary Part E): a one-shot
        # injection of ONE conservative bulk gas (``"o2"`` / ``"inert_n2"``)
        # through the gas pump primitive ``GameMap.inject_gas_n_vec``
        # (payloads.inject_bulk_gas) -- the only honest path to O2/N2, since
        # the FieldEdit ``gas`` policy refuses a conservative slice.
        # ``bulk_amount`` is bulk N per tile in the pump's real unit (1.0 =
        # one tile's whole bulk N at 1 atm ambient, so ambient air holds 0.21
        # O2 + 0.79 N2), FLAT over every open tile of the ``bulk_radius``
        # disc; ``bulk_temp`` is the deposit's temperature in game degrees
        # above ambient (0 = born at ambient). The ``*_q`` twins are those two
        # quantized ONCE at the table door (gas_fixed); the executor reads
        # only them. "" = no bulk gas.
        self.bulk_gas = bulk_gas
        self.bulk_amount = bulk_amount
        self.bulk_radius = bulk_radius
        self.bulk_temp = bulk_temp
        self.bulk_amount_q = int(bulk_amount_q)
        self.bulk_temp_q = int(bulk_temp_q)

    def __repr__(self):
        return (f"PayloadDef({self.name!r}, radius={self.radius!r}, "
                f"pressure={self.pressure!r})")


# ---------------------------------------------------------------------------
# Row/field accessors — mirror GasTable._get_row/_get_field: a config section
# is either a config.Namespace (attributes) or a plain dict (tests).
# ---------------------------------------------------------------------------
_REQUIRED = object()   # sentinel: no default -> missing column is a KeyError
#: The archetypes whose trigger pull is a SHOT -- a noise (G9p): their rows
#: must author ``loudness`` (metres).
_FIRING_ARCHETYPES = ("projectile", "hitscan", "spray")


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
                loudness=col("loudness", _REQUIRED
                             if archetype in _FIRING_ARCHETYPES else 0.0),
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
                throw_windup_seconds=col("throw_windup_seconds", 0.0),
                melee_push_ns=col("melee_push_ns", 0.0),
                melee_arc_half_deg=col("melee_arc_half_deg", 60.0),
                plant_seconds=col("plant_seconds", 0.0),
            )
            # The melee verb's columns (G6p): the push a finite number >= 0,
            # the arc in (0, 180] degrees; the arc quantized ONCE to Q16
            # radians (the turning law's currency, wrap period 2*PI_Q16).
            _p = w.melee_push_ns
            if (isinstance(_p, bool) or not isinstance(_p, (int, float))
                    or not _p >= 0.0 or _p == float("inf")):
                raise ValueError(
                    f"weapons.{name}.melee_push_ns {_p!r} must be a finite "
                    f"number >= 0 (N s; 0 = no push)")
            _a = w.melee_arc_half_deg
            if (isinstance(_a, bool) or not isinstance(_a, (int, float))
                    or not 0.0 < _a <= 180.0):
                raise ValueError(
                    f"weapons.{name}.melee_arc_half_deg {_a!r} must be a "
                    f"number in (0, 180] (degrees)")
            _ld = w.loudness
            if (isinstance(_ld, bool) or not isinstance(_ld, (int, float))
                    or not _ld >= 0.0 or _ld == float("inf")):
                raise ValueError(
                    f"weapons.{name}.loudness {_ld!r} must be a finite "
                    f"number >= 0 (metres of hearing radius)")
            from simulation.noise import radius_tiles as _radius_tiles
            w.loudness_tiles = _radius_tiles(_ld, self.tile_size_m)
            from simulation import unit_fixed as _uf
            import math as _math
            w.melee_arc_half_q = _uf.quantize_scalar(_math.radians(float(_a)))
            # The throw wind-up (G3p): seconds -> integer ticks, the
            # reload_ticks derivation (door 1); 0 stays 0 (no wind-up).
            _wu = w.throw_windup_seconds
            if (isinstance(_wu, bool) or not isinstance(_wu, (int, float))
                    or not _wu >= 0.0 or _wu != _wu or _wu == float("inf")):
                raise ValueError(
                    f"weapons.{name}.throw_windup_seconds {_wu!r} must be a "
                    f"finite number >= 0 (seconds; 0 = no wind-up)")
            w.throw_windup_ticks = (ticks_from_seconds(_wu, ticks_per_second)
                                    if _wu > 0 else 0)
            # The plant (#193): seconds -> integer ticks, the wind-up's
            # derivation; 0 stays 0 (the charge lies down on the verb's tick).
            _pl = w.plant_seconds
            if (isinstance(_pl, bool) or not isinstance(_pl, (int, float))
                    or not _pl >= 0.0 or _pl != _pl or _pl == float("inf")):
                raise ValueError(
                    f"weapons.{name}.plant_seconds {_pl!r} must be a "
                    f"finite number >= 0 (seconds; 0 = an instant plant)")
            w.plant_ticks = (ticks_from_seconds(_pl, ticks_per_second)
                             if _pl > 0 else 0)
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


def _momentum_columns(row, name):
    """The ``[ammo.*] bullet_g`` / ``muzzle_mps`` pair (G5p-a2, the unit
    impulse law): (0.0, 0.0) when both are absent; refused by name when only
    one is given or either is not a finite number >= 0."""
    import math
    has = [(k in row) if isinstance(row, dict) else hasattr(row, k)
           for k in ("bullet_g", "muzzle_mps")]
    if not any(has):
        return 0.0, 0.0
    if not all(has):
        raise ValueError(f"ammo.{name}: bullet_g and muzzle_mps come as a "
                         f"pair (the round's momentum), one is missing")
    out = []
    for k in ("bullet_g", "muzzle_mps"):
        v = row[k] if isinstance(row, dict) else getattr(row, k)
        if isinstance(v, bool) or not isinstance(v, (int, float))                 or not math.isfinite(float(v)) or v < 0:
            raise ValueError(f"ammo.{name}.{k} must be a finite number >= 0, "
                             f"got {v!r}")
        out.append(float(v))
    return out[0], out[1]


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
            # expose (G4p): refused by name when negative or not a number,
            # quantized once to the Q16 raise per hit.
            from simulation.exposure import expose_q as _expose_q
            raw_expose = _get_field(row, "ammo", name, "expose", 0.0)
            a.expose_q = _expose_q(raw_expose, f"ammo.{name}.expose")
            a.expose = float(raw_expose)
            # bullet_g / muzzle_mps (G5p-a2): both or neither, each a finite
            # number >= 0 -- refused by name otherwise.
            a.bullet_g, a.muzzle_mps = _momentum_columns(row, name)
            a.momentum_ns = a.bullet_g / 1000.0 * a.muzzle_mps
            # speed_q16: the authored tiles-per-tick quantized ONCE onto the
            # Q16.16 grid (ingress door 2) — the unified march's integer
            # step-budget source (W2). 96.0 -> 6291456 exactly.
            a.speed_q16 = unit_fixed.quantize_scalar(
                float(a.speed_tiles_per_tick))
            self.by_name[name] = a
        self.names = list(self.by_name)


# What a strength row may not author: the door derives or forbids each of these,
# so a row cannot say two things about one blast (#31, design section 2).
_STRENGTH_REFUSES = ("pressure", "heat_amount", "heat_radius", "expansion_m3",
                     "expansion_radius", "expansion_ticks", "explosive_kg",
                     "ignite_radius", "ignite_intensity")


def _strength_columns(name, row, col, emit_blast_smoke, explosive, blast_cfg):
    """The ``[payloads.*] strength`` door (#31): refuse what a generic explosion
    may not carry, by the KEY's presence (``col`` cannot tell an authored 0.0
    from an absent column), and derive everything else from ``[blast]``. Returns
    the PayloadDef keyword arguments, or {} for a row with no strength."""
    import math
    from simulation.blast_expansion import blast_settings, source_steps
    strength = float(col("strength", 0.0))
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError(f"payloads.{name}.strength {strength!r} must be >= 0")
    if strength == 0.0:
        return {}
    for key in _STRENGTH_REFUSES:
        if (key in row) if isinstance(row, dict) else hasattr(row, key):
            raise ValueError(
                f"payloads.{name} has strength and {key}: a strength row's "
                f"blast is derived from [blast], never authored")
    if not (explosive and emit_blast_smoke):
        raise ValueError(
            f"payloads.{name} has strength but needs explosive and "
            f"emit_blast_smoke = true: a blast leaves its charge's soot")
    if not col("radius", 0) > 0:
        raise ValueError(
            f"payloads.{name} has strength but radius <= 0: its direct damage "
            f"and the blast log both need the disc")
    b = blast_settings(blast_cfg)
    r_s = source_steps(strength)
    return dict(
        strength=strength,
        expansion_m3=b.expansion_m3_per_strength * strength,
        heat_mj=b.heat_mj_per_strength * strength,
        explosive_kg=b.grenade_kg * strength,
        source_r2=r_s * r_s,
        heat_r2=4 * r_s * r_s)


# The flash columns (G4 P5, #75): any other ``flash_*`` key is a typo, refused.
_FLASH_COLUMNS = ("flash", "flash_decay_s", "flash_color")


def _flash_columns(name, row, col):
    """The ``[payloads.*]`` flash door (G4 P5, #75; design §3.5 + §6 S9): read
    and validate ``flash`` (light units at peak, >= 0, default 0 = none),
    ``flash_decay_s`` (the e-fold, seconds, > 0 when there is a flash; default
    0.3) and ``flash_color`` (three numbers in [0, 1], default white). Refuses
    any key starting ``flash`` that is not one of the three, so a misspelt
    column is loud at load instead of a silent default. Returns the PayloadDef
    keyword arguments. Render-only: nothing in the sim reads them."""
    import math
    keys = row.keys() if isinstance(row, dict) else vars(row).keys()
    bad = sorted(k for k in keys
                 if str(k).startswith("flash") and k not in _FLASH_COLUMNS)
    if bad:
        raise ValueError(f"payloads.{name} has unknown flash column(s) {bad}: "
                         f"the flash columns are {list(_FLASH_COLUMNS)}")
    flash = float(col("flash", 0.0))
    decay = float(col("flash_decay_s", 0.3))
    color = col("flash_color", (1.0, 1.0, 1.0))
    if not (math.isfinite(flash) and flash >= 0.0):
        raise ValueError(f"payloads.{name}.flash {flash!r} must be >= 0 "
                         f"(light units at peak; 0 = no flash)")
    if flash > 0.0 and not (math.isfinite(decay) and decay > 0.0):
        raise ValueError(f"payloads.{name}.flash_decay_s {decay!r} must be > 0 "
                         f"for a row with a flash (the e-fold, seconds)")
    try:
        color = tuple(float(c) for c in color)
    except TypeError:
        raise ValueError(f"payloads.{name}.flash_color {color!r} must be three "
                         f"numbers in [0, 1]") from None
    if len(color) != 3 or not all(math.isfinite(c) and 0.0 <= c <= 1.0
                                  for c in color):
        raise ValueError(f"payloads.{name}.flash_color {color!r} must be three "
                         f"numbers in [0, 1]")
    return dict(flash=flash, flash_decay_s=decay, flash_color=color)


# The blind columns (G3p): any other ``blind_*`` key is a typo, refused.
_BLIND_COLUMNS = ("blind_radius", "blind_seconds", "blind_away_frac")


def _blind_columns(name, row, col, ticks_per_second):
    """The ``[payloads.*]`` blind door (G3p, design §3 G3p): read and validate
    ``blind_radius`` (tiles, >= 0; absent or 0 = this row blinds nobody),
    ``blind_seconds`` (> 0, REQUIRED when the radius is) and
    ``blind_away_frac`` (in [0, 1], REQUIRED when the radius is: the share of
    the blinding that is left when the unit faces straight away from the
    flash). Refuses any key starting ``blind`` that is not one of the three,
    and a ``blind_seconds`` / ``blind_away_frac`` on a row with no radius
    (dead data is a typo). Quantizes ONCE: the radius to Q16 tiles, the
    duration to integer ticks (``ticks_from_seconds``), the fraction to Q16.
    Returns the PayloadDef keyword arguments."""
    import math
    from simulation import unit_fixed
    keys = row.keys() if isinstance(row, dict) else vars(row).keys()
    bad = sorted(k for k in keys
                 if str(k).startswith("blind") and k not in _BLIND_COLUMNS)
    if bad:
        raise ValueError(f"payloads.{name} has unknown blind column(s) {bad}: "
                         f"the blind columns are {list(_BLIND_COLUMNS)}")
    radius = col("blind_radius", 0.0)
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not (math.isfinite(radius) and radius >= 0.0)):
        raise ValueError(f"payloads.{name}.blind_radius {radius!r} must be a "
                         f"finite number >= 0 (tiles; 0 = blinds nobody)")
    radius = float(radius)
    if radius == 0.0:
        stray = [c for c in ("blind_seconds", "blind_away_frac")
                 if c in (row if isinstance(row, dict) else vars(row))]
        if stray:
            raise ValueError(f"payloads.{name} sets {stray} without a "
                             f"blind_radius > 0 (the row would blind nobody)")
        return {}
    secs = col("blind_seconds", None)
    away = col("blind_away_frac", None)
    if (isinstance(secs, bool) or not isinstance(secs, (int, float))
            or not (math.isfinite(secs) and secs > 0.0)):
        raise ValueError(f"payloads.{name}.blind_seconds {secs!r} must be a "
                         f"finite number > 0 (required with blind_radius)")
    if (isinstance(away, bool) or not isinstance(away, (int, float))
            or not (math.isfinite(away) and 0.0 <= away <= 1.0)):
        raise ValueError(f"payloads.{name}.blind_away_frac {away!r} must be a "
                         f"number in [0, 1] (required with blind_radius)")
    radius_q = unit_fixed.quantize_scalar(radius)
    if radius_q <= 0:
        raise ValueError(f"payloads.{name}.blind_radius {radius!r} quantizes "
                         f"to 0 Q16 tiles")
    if ticks_per_second is None:
        from config import CFG
        ticks_per_second = CFG.clock.ticks_per_second
    return dict(blind_radius=radius, blind_seconds=float(secs),
                blind_away_frac=float(away), blind_radius_q=radius_q,
                blind_ticks=ticks_from_seconds(float(secs), ticks_per_second),
                blind_away_q=unit_fixed.quantize_scalar(float(away)))


def _bulk_gas_columns(name, col, gas_species, bulk_gases, trace_gases):
    """The ``[payloads.*]`` bulk-gas door (#133 E.2): read and validate the
    ``bulk_gas`` / ``bulk_amount`` / ``bulk_radius`` / ``bulk_temp`` columns
    and quantize the amount and temperature ONCE. Returns the PayloadDef
    keyword arguments. One path per gas: a bulk gas is never a
    ``gas_species`` (the FieldEdit ``gas`` policy refuses a conservative
    slice -- it would fail at the flush, mid-match), and a trace gas is never
    a ``bulk_gas`` (that is ``emit_gas``'s business)."""
    import math
    from simulation import gas_fixed as _gas_fx
    if gas_species in bulk_gases:
        raise ValueError(
            f"payloads.{name}.gas_species {gas_species!r} is a conservative "
            f"BULK gas: FieldEdit refuses it -- use bulk_gas = "
            f"{gas_species!r} with bulk_amount / bulk_radius instead")
    bulk_gas = str(col("bulk_gas", ""))
    bulk_amount = float(col("bulk_amount", 0.0))
    bulk_radius = float(col("bulk_radius", 0.0))
    bulk_temp = float(col("bulk_temp", 0.0))
    if not bulk_gas:
        if bulk_amount or bulk_radius or bulk_temp:
            raise ValueError(
                f"payloads.{name} sets bulk_amount / bulk_radius / bulk_temp "
                f"without bulk_gas -- name the gas ({sorted(bulk_gases)})")
        return {}
    if bulk_gas in trace_gases:
        raise ValueError(
            f"payloads.{name}.bulk_gas {bulk_gas!r} is a TRACE gas -- trace "
            f"gas is emitted with gas_species (emit_gas); bulk_gas takes "
            f"only {sorted(bulk_gases)}")
    if bulk_gas not in bulk_gases:
        raise ValueError(
            f"payloads.{name}.bulk_gas {bulk_gas!r} is not a bulk gas "
            f"({sorted(bulk_gases)})")
    if not (math.isfinite(bulk_amount) and bulk_amount > 0.0):
        raise ValueError(
            f"payloads.{name}.bulk_amount {bulk_amount!r} must be > 0 (bulk N "
            f"per tile; 1.0 = a whole tile of ambient air)")
    if not (math.isfinite(bulk_radius) and bulk_radius > 0.0):
        raise ValueError(
            f"payloads.{name}.bulk_radius {bulk_radius!r} must be > 0 (tiles)")
    if not math.isfinite(bulk_temp):
        raise ValueError(
            f"payloads.{name}.bulk_temp {bulk_temp!r} is not finite")
    amount_q = _gas_fx.quantize_bulk_scalar(bulk_amount)
    if amount_q <= 0:
        raise ValueError(
            f"payloads.{name}.bulk_amount {bulk_amount!r} quantizes to 0 bulk "
            f"counts -- below one LSB of the bulk currency")
    return dict(bulk_gas=bulk_gas, bulk_amount=bulk_amount,
                bulk_radius=bulk_radius, bulk_temp=bulk_temp,
                bulk_amount_q=amount_q,
                bulk_temp_q=_gas_fx.quantize_q16_scalar(bulk_temp))


class PayloadTable:
    """``[payloads.*]`` rows keyed by name. Every column defaults (a payload
    row lists only what it does — 'rest zero/empty'). A nonempty
    ``gas_species`` is validated against the canonical gas-name set
    (simulation.gases.GAS_NAMES — the ``gmap.gas`` slice vocabulary) so a
    typo'd species is loud at startup, not at the first detonation."""

    def __init__(self, payloads_cfg, explosives_cfg=None, blast_cfg=None,
                 ticks_per_second=None, tile_size_m=1.0):
        # Lazy import (module constants only, mirrors AmmoTable's unit_fixed).
        from simulation.gases import GAS_NAMES, N_GASES, N_TRACE_GASES
        valid_gases = set(GAS_NAMES.values())
        # The conservative bulk pair are exactly the ids >= N_TRACE_GASES
        # (GasTable's own layout check) -- the bulk-gas effect's vocabulary.
        bulk_gases = {GAS_NAMES[g] for g in range(N_TRACE_GASES, N_GASES)}
        trace_gases = valid_gases - bulk_gases
        # #12 handle 3: [explosives.<name>] soot_yield (g soot / g explosive).
        # None reads CFG.explosives (the GasTable / CFG.smoke idiom).
        if explosives_cfg is None:
            from config import CFG as _CFG
            explosives_cfg = getattr(_CFG, "explosives", None)
        soot_yield = {}
        if explosives_cfg is not None:
            for ename, erow in _iter_rows(explosives_cfg):
                y = float(_get_field(erow, "explosives", ename, "soot_yield"))
                if not (0.0 <= y <= 1.0):
                    raise ValueError(f"explosives.{ename}.soot_yield {y!r} outside "
                                     f"[0, 1] (grams of soot per gram of explosive)")
                soot_yield[ename] = y
        self.by_name: dict[str, PayloadDef] = {}
        for name, row in _iter_rows(payloads_cfg):
            def col(c, default, _row=row, _name=name):
                return _get_field(_row, "payloads", _name, c, default)

            gas_species = str(col("gas_species", ""))
            if gas_species and gas_species not in valid_gases:
                raise ValueError(
                    f"payloads.{name}.gas_species {gas_species!r} is not a "
                    f"known gas (engine/05 §6.2): {sorted(valid_gases)}")
            bulk = _bulk_gas_columns(name, col, gas_species, bulk_gases,
                                     trace_gases)
            flash = _flash_columns(name, row, col)
            blind = _blind_columns(name, row, col, ticks_per_second)
            emit_blast_smoke = bool(col("emit_blast_smoke", False))
            explosive = str(col("explosive", ""))
            # A strength row's charge is derived, so it is known before the
            # blast-smoke check below reads it.
            blast = _strength_columns(name, row, col, emit_blast_smoke,
                                      explosive, blast_cfg)
            explosive_kg = blast.pop("explosive_kg",
                                     float(col("explosive_kg", 0.0)))
            if emit_blast_smoke:
                # The blast's smoke is the charge's soot: a row that makes
                # blast smoke must say what it is made of and how much.
                if explosive not in soot_yield:
                    raise ValueError(
                        f"payloads.{name} emits blast smoke but its explosive "
                        f"{explosive!r} is not an [explosives.*] row "
                        f"({sorted(soot_yield)}) -- #12 handle 3")
                if not explosive_kg > 0.0:
                    raise ValueError(
                        f"payloads.{name} emits blast smoke but explosive_kg = "
                        f"{explosive_kg!r} (the charge mass, kg, must be > 0)")
            blast_soot_g = (explosive_kg * 1000.0 * soot_yield[explosive]
                            if emit_blast_smoke else 0.0)
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
                emit_blast_smoke=emit_blast_smoke,
                heat_amount=col("heat_amount", 0.0),
                heat_radius=col("heat_radius", 0.0),
                explosive=explosive,
                explosive_kg=explosive_kg,
                blast_soot_g=blast_soot_g,
                expansion_m3=blast.pop("expansion_m3",
                                       col("expansion_m3", 0.0)),
                expansion_radius=col("expansion_radius", 0.0),
                expansion_ticks=col("expansion_ticks", 1),
                **blast,
                **bulk,
                **flash,
                **blind,
            )
            # G9p: the detonation's hearing radius, METRES (optional; 0 =
            # silent), quantized once to tiles at this map's tile size.
            _ld = col("loudness", 0.0)
            if (isinstance(_ld, bool) or not isinstance(_ld, (int, float))
                    or not _ld >= 0.0 or _ld == float("inf")):
                raise ValueError(
                    f"payloads.{name}.loudness {_ld!r} must be a finite "
                    f"number >= 0 (metres of hearing radius; 0 = silent)")
            from simulation.noise import radius_tiles as _radius_tiles
            pdef = self.by_name[name]
            pdef.loudness = float(_ld)
            pdef.loudness_tiles = _radius_tiles(_ld, float(tile_size_m))
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
        self.payloads = PayloadTable(payloads_cfg,
                                     ticks_per_second=ticks_per_second,
                                     tile_size_m=tile_size_m)
        self._validate_cross_refs()
        # The hand-grenade KINDS (G3p): the ``[ammo.*]`` rows of the
        # hand-grenade family, in config (table) order -- the index of a row
        # in this tuple IS the throw verb's ``kind`` and the slot of its
        # count on a unit. Append a new row at the END of the family's rows
        # so an existing index keeps its meaning (``grenade_frag`` is first:
        # kind 0, the default).
        self.hand_grenade_rows = tuple(
            n for n, a in self.ammo.by_name.items()
            if a.family == HAND_GRENADE_FAMILY)
        #: The planted-charge rows (#193): the ``[ammo.*]`` rows of the
        #: demo-charge family, the rows a loadout's ``charge`` may name.
        self.charge_rows = tuple(
            n for n, a in self.ammo.by_name.items()
            if a.family == CHARGE_FAMILY)

    def hand_grenade_kind(self, kind) -> str:
        """The ``[ammo.*]`` row name of throw kind ``kind``. Refuses a
        non-int or an index outside the family, naming the kinds."""
        if (isinstance(kind, bool) or not isinstance(kind, Integral)
                or not 0 <= kind < len(self.hand_grenade_rows)):
            raise ValueError(
                f"unknown grenade kind {kind!r}: the hand-grenade kinds are "
                f"{dict(enumerate(self.hand_grenade_rows))}")
        return self.hand_grenade_rows[int(kind)]

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
