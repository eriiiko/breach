"""Material ids + the material-property table (ch.02 — Material System).

Single source of truth for the ``MAT_*`` ids (previously duplicated in
``gamemap.py`` and ``level_loader.py``) and for the per-material property
table that derives every per-tile constant from the material id.

The table is the **data-driven** foundation described in
``docs/architecture/02_material_system.md``: adding a material is one
``config.toml`` row + one CSV mapping. Properties are stored as per-id numpy
arrays so the derived caches (see :mod:`simulation.gamemap`) can be built with
a single fancy-index lookup ``column[material]``.

CPU-only foundation: the optics (``light_atten``/``heat_atten``) and acoustics
(``wave_*``) columns are *stored* here but consumed by nobody yet — they wire
into the ray/wave passes in later chapters.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Material IDs — the single source of truth.
# Renderer keeps its own color table; these are the gameplay/physics ids.
# Order must match the CSV mapping in :func:`level_loader.materials_from_tilemap`.
# ---------------------------------------------------------------------------
MAT_AIR = 0
MAT_HULL = 1
MAT_WOOD = 2
MAT_DOOR = 3
MAT_STEEL = 4
MAT_GLASS = 5
MAT_FURNITURE = 6
# A6 (docs/a6_doors_v0_impl_2026-07-19.md §1): the ENTITY door's CLOSED
# stamp — the [materials.door] column values with mobility = 0, so a closed
# entity door blocks movement/LOS/flow/burst/burn through the exact seams
# walls already use. MAT_DOOR itself is untouched: it stays the LEGACY
# painted door (walkable-but-flow-solid) until the A7 migration.
MAT_DOOR_CLOSED = 7
# P-F4a (docs/fire_realism_design_2026-08-01.md v5.2 execution order): the
# CAMPFIRE REFERENCE OBJECT — a 1-3 kg effective-class fuel row, the tuning
# reference the campfire arc's dials (growth tempo, size, knee position,
# part-burn fraction) are calibrated against (v4 ruling 2; plain edition
# §10). Bench-registered here (a real material id, so MaterialTable's
# contiguity check + every load-time check stay honest) but not placed in
# any shipped level — see [materials.kindling] in config.toml for the row
# and its per-column deviations from furniture.
MAT_KINDLING = 8
# Props & vegetation arc #60 P3 (docs/architecture/graphics/props_and_
# vegetation.md §4.1, Erik's ruling 2026-09-07 §6.1.2): the prop entity's
# ONE material row — "fully walkable, no wind/vision/movement interaction,
# flammable, fuel ~= 2x furniture". APPENDED (ids are positional and
# contiguous — this must always be the LAST id, or every level
# re-materializes).
MAT_FOLIAGE = 9

# Config-key <-> id mapping. The key is the ``[materials.<name>]`` table name.
# Listed in id order; ``MaterialTable`` validates contiguity.
MATERIAL_NAMES = {
    MAT_AIR: "air",
    MAT_HULL: "hull",
    MAT_WOOD: "wood",
    MAT_DOOR: "door",
    MAT_STEEL: "steel",
    MAT_GLASS: "glass",
    MAT_FURNITURE: "furniture",
    MAT_DOOR_CLOSED: "door_closed",
    MAT_KINDLING: "kindling",
    MAT_FOLIAGE: "foliage",
}

# Scalar columns: name -> numpy dtype. ``light_atten`` is handled separately
# because it is a per-channel RGB triple, not a scalar.
_SCALAR_COLUMNS = {
    "hp": np.float32,
    "flammable": bool,
    # mobility: fixed-point integer milli-units — the ease-of-movement
    # coefficient that replaces the old ``passable`` boolean (mobility design
    # §2/§6). 1000 = normal walking speed (air, open door), 400 = furniture
    # (40% speed, 2.5x step time), 0 = impassable (a wall). The walkability
    # predicate is the derived view ``mobility > 0`` (gamemap.is_passable);
    # the cadence speed_fn area-averages it over the footprint. Stored int so
    # the runtime cost expression is pure integer arithmetic (§3).
    "mobility": np.int64,
    "conductivity": np.float32,
    # R14 (thermal model v2 design 2026-09-19): a row states its REAL physical
    # capacity as two authored numbers -- `density` kg/m3 and `specific_heat`
    # J/(kg.K) -- and `thermal_mass` is DERIVED from their product (see
    # `derive_thermal_mass` below). Authored SEPARATELY rather than as one
    # `rho_c` column because they are two independent measured facts with
    # separate literature sources, because the `c_p` drift is its own ACCEPTED
    # GAP in the design, and above all because T5's fuel-mass derivation needs
    # the MASS (`density * V_tile`) on its own -- a lumped `rho_c` could not be
    # re-split into one.
    # float64, not float32 like the legacy columns: these are AUTHORING inputs
    # to a load-time derivation, never projected per tile and never in a
    # digest, so there is no reason to round them at the door -- and the R14
    # property gate then compares the derivation against the stored columns
    # exactly.
    "density": np.float64,
    "specific_heat": np.float64,
    "ignition_temp": np.float32,
    "heat_atten": np.float32,
    "wave_absorb": np.float32,
    "blast_resist": np.float32,
}


# Conduction log-bucket constants (engine/06 §2.4–§2.5, proposal §7.2
# [physics.thermal]). Defaults mirror config.toml; the live values are read from
# CFG and threaded in via :meth:`from_config` so the table tracks config edits.
# Kept here so a dict-built table (tests) and any config-less build still produce
# a valid face table.
class _ThermalOverride:
    """A read-only view of a ``[physics.thermal]`` namespace (or dict) with a
    few keys replaced. Used by :meth:`MaterialTable.from_config` to inject the
    tick length from ``[clock]`` without giving the thermal block a second,
    drift-prone copy of it. Accepts and presents the same duck type
    :meth:`MaterialTable._thermal_get` reads."""

    __slots__ = ("_base", "_over")

    def __init__(self, base, over):
        self._base, self._over = base, dict(over)

    def __getattr__(self, name):
        if name in self._over:
            return self._over[name]
        base = self._base
        if base is None:
            raise AttributeError(name)
        if isinstance(base, dict):
            if name not in base:
                raise AttributeError(name)
            return base[name]
        return getattr(base, name)


_THERMAL_DEFAULTS = {
    "TEMP_SCALE": 65536,  # Q16.16, == HEAT_SCALE (shared temperature/heat domain)
    "SHIFT_MIN": 2,       # rate floor / stability bound (4 * 1/4 <= 1)
    "NO_FACE": 63,        # sentinel: kappa==0 face / grid edge -> zero conduction
    # T5b / R10: the CFL stability anchor (SHIFT_AT_REF / KAPPA_REF) is RETIRED.
    # The three constants below are what sets the absolute rate now, and they
    # are physics, not dials. See _build_conduction_tables.
    "h_conv": 6.0,             # W/(m2.K) -- natural-convection coefficient at a
                               # solid|gas boundary (Churchill & Chu 1975)
    "tile_size_ref_m": 0.333,  # the reference dx the table is built at
    "TICK_DT_S": 1.0 / 24.0,   # fallback dt; from_config overrides from [clock]
    "c_v": 0.0076849,          # air's rho*c_v in thermal_mass column units --
                               # the gas side's capacity in the face derivation
    "ceiling_h": 2.5,          # m of deck height; from_config injects the real
                               # one from [physics.water], the ONE source
    # COOL-SHIFT AXIS (2026-07-30): the global that seeds the per-material
    # T5b step 7 / R1: "COOL_SHIFT" (the per-row default) stood here.
}

# T5b step 7 / R1: `_COOL_SHIFT_MAX` and the column's validation bounds
# stood here, with their rationale. Deleted with Pass 3.

# FUEL-FRACTION AXIS (2026-07-30) — the reciprocal shift `fixedpoint::make_recip`
# uses (``fixed_point.h``: ``constexpr int RECIP_SHIFT = 32``). The fire logistic
# divides by a per-tile constant with a load-time reciprocal + a 128-bit multiply
# (``recip_mul``), never a runtime divide, because the sim path is Q16.16 integer
# and determinism is a hard requirement. This mirror exists so the per-material
# reciprocal can be baked HERE, where the material table lives, instead of
# shipping an `hp` plane to C++ and dividing per cell.
_FUEL_RECIP_SHIFT = 32

# --- PER-MATERIAL EXTINCTION TEMPERATURE (P-R3, 2026-07-31 — docs/radiation_
# raycaster_extinction_ruling_2026-07-31.md A3 ride-along) ------------------
#
# `[physics.fire]` defaults consumed by the `fire_T_ext` derivation. Mirrors
# config.toml; the live values are threaded in via :meth:`from_config` so the
# table tracks config edits, the same contract `_THERMAL_DEFAULTS` has. Kept
# here so a dict-built table (tests) and any config-less build still produce a
# valid `fire_T_ext_q16` column.
#
# Until R11 (thermal model v1, 2026-09-19) this dict also carried the seven
# dials the load-time ignition-seed sustain check read (`fire_T_span`,
# `k_grow`, `k_die`, the three O2 fractions, `ignition_seed`). That check is
# deleted with `cool_shift` — its gain was `bed_per_I * 2^(cool_shift -
# heat_inv_shift)`, and R1 removes the loss channel it was denominated in —
# so those rows are gone. The dials themselves are untouched in config.toml;
# combustion still reads them.
_FIRE_DEFAULTS = {
    # THE Δ: fire_T_ext[mat] = ignition_temp[mat] - ignition_to_ext_delta.
    "ignition_to_ext_delta": 100.0,
}


def quantize_q16(v) -> int:
    """Real value -> Q16.16 int, ROUND-HALF-AWAY-FROM-ZERO.

    THE CONTRACT: bit-identical to C++ ``fixedpoint::quantize``, which is

        double scaled = v * 65536.0;
        return (int32)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);

    i.e. one IEEE-754 binary64 multiply, ±0.5, then truncation TOWARD ZERO.
    Python's ``*`` is that same binary64 multiply and ``int()`` is that same
    truncation, so the two agree on every input on every machine — the same
    "IEEE double is bit-identical cross-machine for load-time scalar constants"
    rule ``fuel_recip_from_hp`` above rests on. This is what makes a UNIFORM
    ``fire_T_ext_q16`` plane byte-identical to the C++ scalar fallback, which is
    the axis's back-compat gate.
    """
    scaled = float(v) * 65536.0
    return int(scaled + 0.5) if scaled >= 0.0 else int(scaled - 0.5)


def fire_T_ext_from_ignition(ignition_temp, delta) -> float:
    """``fire_T_ext[mat] = ignition_temp[mat] - ignition_to_ext_delta``.

    DERIVED, NOT A DIAL — there is no per-material ``fire_T_ext`` config column
    and there must never be one. `fire_T_ext` sits on the same physical axis as
    `ignition_temp` (both are "the temperature at which this material's
    pyrolysis does/doesn't carry itself"), so the invariant that matters —
    ``fire_T_ext < ignition_temp``, i.e. a tile cannot ignite below its own
    sustain floor and snap straight back out — becomes STRUCTURAL rather than
    something a config author must remember. The shipped global 350 violated it
    for BOTH flammable materials (wood 300, furniture 280).

    ONE new global (`ignition_to_ext_delta`), zero new per-material columns —
    the same cool-shift-vacuum-offset precedent. `fire_T_span` deliberately
    stays global: it is the WIDTH of the `hot` ramp, not its foot.

    Non-flammable materials get the same arithmetic rather than a special case.
    Their value is never read (the fire logistic runs under
    ``if (!flammable[i]) continue``), so the choice is free; deriving it anyway
    means a future flammable material needs no code edit, and it keeps the
    column a pure function of one input column. Materials with
    ``ignition_temp == 0`` therefore carry a negative, unread, -Δ.
    """
    return float(ignition_temp) - float(delta)


def fuel_recip_from_hp(hp) -> int:
    """Bake ``round(2**32 / hp)`` exactly as ``fixedpoint::make_recip`` does.

    THE CONTRACT: this must be bit-identical to the C++ ``make_recip``, which is

        double r = (double)((int64_t)1 << 32) / divisor_real;
        return (int64_t)(r + 0.5);

    i.e. one IEEE-754 binary64 divide, ``+ 0.5``, then truncation toward zero.
    Python's ``/`` on ints/floats IS that same binary64 divide and ``int()`` IS
    that same truncation, so the two agree on every input, on every machine —
    the same "IEEE double is bit-identical cross-machine for load-time scalar
    constants" rule the whole fixed-point migration rests on (S1 locked
    decision). ``tests/test_fuel_fraction_axis.py`` pins the agreement against
    the real C++ ``make_recip`` (exposed as ``breach_physics.fp_make_recip``)
    for every shipped material and a wide sweep, so a divergence cannot pass CI.

    ``hp <= 0`` (air, and any future massless material) returns **0**, the
    deliberate safe value: it is never a divide, and ``recip_mul(x, 0) == 0``
    makes the fuel fraction read F = 0, "no fuel here" — the honest answer for a
    tile with no substance. Those tiles are unreachable in practice (the
    logistic runs under ``if (!flammable[i]) continue`` and nothing flammable
    has hp 0), but a sentinel that quietly means "infinite fuel" would be a trap
    waiting for the first flammable-gas material, so it means the opposite.
    """
    hp_f = float(hp)
    if not (hp_f > 0.0):
        return 0
    return int((float(1 << _FUEL_RECIP_SHIFT) / hp_f) + 0.5)


# ---------------------------------------------------------------------------
# R14 — MATERIALS ARE AUTHORED BY DENSITY (thermal model v2 design 2026-09-19,
# ruling R14; row-by-row derivation in report_t3.md D1 and report_t3b.md).
#
# A `[materials.*]` row states its real `density` and `specific_heat`;
# `thermal_mass` is DERIVED from their product, snapped to a power of two (R5,
# because the column sits on a bit-shift). No row authors `thermal_mass` any
# more -- with ONE exception, the literal 0 that DECLARES the gas thermal
# regime, which was never a capacity value in the first place (see the column
# docs on `[materials.air]`).
#
# THE UNIT IS A RATIO OF TWO CAPACITIES, NOT A JOULE COUNT. This is what makes
# the derivation resolution-independent, so state it exactly. R13 pins the heat
# currency by naming a MATERIAL and a COLUMN VALUE -- "wood at ~12 % moisture
# content, rho*c = 0.9 MJ/(m3.K), is thermal_mass 8" -- so for any row
#
#     thermal_mass = C_row / C_pin  * THERMAL_MASS_PIN
#                  = (rho_c_row * V_tile) / (RHO_C_PIN * V_tile) * 8
#                  = rho_c_row / (RHO_C_PIN / 8)
#
# and V_tile CANCELS IDENTICALLY -- both capacities are measured on the SAME
# tile, whatever size that tile is. So the derivation needs no tile geometry
# and the table stays global, at every spatial resolution, by construction
# rather than by luck. (report_t3b.md section 1 carries the full argument and
# the per-level table.)
#
# What does NOT cancel, and is therefore still resolution-dependent, is the
# ABSOLUTE worth of one heat count, `J_per_count = RHO_C_PIN * V_tile /
# (8 * 65536)`: it scales with the tile volume. That number lives on the SOURCE
# side (`rad_scale`, the combustion deposit), it is a single scalar rather than
# ten table rows, and making it per-level is report_p2b.md section 13 item 7 --
# T5's business, deliberately not T3b's.
# ---------------------------------------------------------------------------
# R14, THE FUEL HALF (thermal model v2, Erik 2026-09-19; report_t3.md D5 §6.3).
#
# `hp` was doing two unrelated jobs -- STRUCTURAL INTEGRITY (what combat
# damages, what the health bar shows) and FUEL STORE (how much O2 a tile's
# combustion can consume before it is spent) -- and they want different
# numbers. A massive wood tile is 154 kg; `hp = 60` at the shipped global
# `fuel_per_o2 = 0.7` implies 31.9 kg of fuel, 4.8x short. Furniture is 9.7x
# short. Erik: *"i think u may change the hp to make it consistent."*
#
# THE SEPARATION. `hp` keeps its meaning untouched. What derives from mass is
# the EXCHANGE RATE `fuel_per_o2[mat]` -- the hp a tile pays per unit of O2 its
# fire consumes -- chosen so that spending the whole bar consumes exactly the
# tile's real combustible mass:
#
#     fuel_per_o2[mat] = hp[mat] / O2_UNITS_PER_TILE[mat]
#     O2_UNITS_PER_TILE[mat] = density[mat] * V_tile / KG_FUEL_PER_N_O2
#
# so the fuel STORE is physics and the hp BAR stays a gameplay quantity. It was
# a single global 0.7 before, which made the store proportional to `hp` and
# therefore structural rather than physical.
#
# KG_FUEL_PER_N_O2 -- the fuel mass a unit of N_O2 burns, from two cited
# constants and nothing fitted:
#   * one unit of N_O2 is one atmosphere of O2 in one tile = 11.525 mol =
#     0.36878 kg (p*V/(R*T) at the engine's own ambient);
#   * Huggett's constant, 13.1 +/- 0.7 MJ per kg of O2 consumed, near-universal
#     across organic fuels (Huggett 1980, archived under docs/papers/), so one
#     unit of N_O2 releases 4.831 MJ;
#   * wood's EFFECTIVE (cone-calorimeter) heat of combustion is 13 MJ/kg -- the
#     gross 18-20 MJ/kg less the char that never flames (Drysdale ch. 1
#     Table 1.13; Babrauskas, *Heat Release in Fires*).
#   => 4.831 / 13 = 0.3716 kg of wood per unit of N_O2.
#
# TILE GEOMETRY. `V_tile = tile_size_ref_m^2 * ceiling_h`. Unlike
# `thermal_mass` (where the tile volume cancels -- report_t3b.md §1.1), the fuel
# MASS is an absolute quantity and does not cancel, so this column is
# tile-size dependent for the same reason the conduction table and
# `rad_scale_derived` are. Same reference, same open question (T3 §8 q9).
KG_FUEL_PER_N_O2 = 0.3716181984470593   # kg of cellulosic fuel per N_O2 unit

# ---------------------------------------------------------------------------
# THE REFERENCE SUBSTANCE — the metre-stick of the whole thermal model
# (M2, docs/thin_material_rows_design_2026-09-20.md §7).
#
# `THERMAL_MASS_UNIT` is a DEFINITION, not a measurement. Every physical
# quantity in the thermal model is denominated against it: `J_per_count`, the
# arc #54 energy ledger, `c_v`, the marine burn band in kW/m2. Move it and
# EVERY material in the game silently re-scales, with nothing failing, because
# everything is *relative* to it.
#
# R13 chose the value by READING IT OFF A ROW -- `wood` was a solid tile, its
# rho*c was 0.899 MJ/(m3.K), and it carried `thermal_mass = 8`. After M2 NO ROW
# IS AUTHORED AT FULL FILL, so that visible anchor disappears while the value
# stays exactly correct (a definition does not expire because nothing is
# currently one metre long).
#
# THE HAZARD this block exists to close: a future reader opens `wood`, sees
# `thickness_m = 0.005` / `fill_fraction = 0.015`, concludes the pin is stale,
# and "fixes" it -- re-scaling every material and every derived physical number
# at once. That is the same shape as the 2^16 transcription error this arc
# spent a whole patch finding (report_t5b.md §6.3).
#
# THE RULE: *the thermal_mass unit scale is a definition anchored on a named
# REFERENCE SUBSTANCE, and is never read off, nor re-derived from, a material
# row.* The constants below name that substance, so "no row is currently this
# substance at full fill" reads as obviously fine rather than obviously broken.
# `tests/test_m2_reference_substance_pin.py` gates it.
REFERENCE_SUBSTANCE = "solid softwood construction lumber at 12 % moisture content"
REFERENCE_DENSITY = 555.0        # kg/m3   -- Wood Handbook FPL-GTR-190 Table 5-3
REFERENCE_SPECIFIC_HEAT = 1620.0  # J/(kg.K) -- FPL-GTR-190 ch.4 at 12 % MC, 293 K
# The reference substance's volumetric heat capacity, 899 100 J/(m3.K). R13's
# pin is this number ROUNDED to two significant figures, which is the ruled
# value and the one the whole engine is denominated against -- so the pin is
# stated as the literal it is, and the anchor is asserted, never recomputed.
REFERENCE_RHO_C = REFERENCE_DENSITY * REFERENCE_SPECIFIC_HEAT
RHO_C_PIN = 0.9e6          # J/(m3.K) -- R13's pin: the reference substance, rounded
THERMAL_MASS_PIN = 8       # the column value that pin carries
# How far the ruled pin may sit from the reference substance it names. 0.9e6 vs
# 899 100 is 0.10 %; anything beyond this band means the pin and its stated
# anchor have come apart and one of them is wrong.
PIN_ANCHOR_TOLERANCE = 0.005
# One `thermal_mass` unit, in volumetric heat capacity: 112 500 J/(m3.K).
THERMAL_MASS_UNIT = RHO_C_PIN / THERMAL_MASS_PIN


# ---------------------------------------------------------------------------
# M2 — AUTHOR DIMENSIONS, DERIVE FILL (design §4, Erik's ruling 2026-09-20)
#
# A row keeps its REAL, CITED `density` and `specific_heat` and states the
# object's physical GEOMETRY as `thickness_m`. `fill_fraction`, `mass` and
# `thermal_mass` are all DERIVED from those plus the tile geometry; none of
# the three may be authored.
#
# WHY A THICKNESS AND NOT A FILL FRACTION. The material table is GLOBAL while
# `tile_size_m` is PER LEVEL (config.toml says so in as many words), and the
# table's geometry is built at `tile_size_ref_m`. An authored fill fraction
# would therefore bake the REFERENCE tile size into the row's physical meaning:
# on a 1.0 m level the "same" row would silently become a 1.5 cm slab instead
# of a 0.5 cm one. An authored THICKNESS states a fact about the object that no
# tile size can falsify -- the same lesson R14 already learned one level down
# (author by `density` so `V_tile` cancels), applied to geometry.
#
# THE STRUCTURAL PAYOFF: ignition time becomes tile-size invariant. For a panel
# the capacity `C = rho*c*thickness*tile_w*ceiling_h` scales as `tile_w`, and
# the incident power arrives on an exposed face of area `tile_w*ceiling_h` so
# it scales as `tile_w` too -- `dT/dt = P/C` is therefore INDEPENDENT of tile
# size. With an authored fill fraction it is not. Gated by
# `tests/test_m2_dimensions_and_thin_rows.py`.
#
# WHY NOT FOLDED INTO `density` OR `heat_atten`: `density = 8.33` would be a
# lie about wood, where `density = 555, thickness_m = 0.005` is two true
# statements each independently citable; and 5 mm of wood is still optically
# OPAQUE, so `heat_atten` stays 0.90 (emissivity, only -- forever) while the
# mass drops 67x. A single smeared density cannot express that.
#
# ACCEPTED GAP (design §4, Erik's ruling 2026-09-21): rows that STAND ON THE
# FLOOR rather than span the tile (`furniture`, `kindling`, `foliage`) are
# authored as an EQUIVALENT SLAB THICKNESS, which is the right single parameter
# for a lump but scales as `tile_w` where a floor-standing object's mass really
# scales as `tile_w^2`. A `form` discriminator is added when a level at a
# different tile size actually ships. Recorded, not fixed.
# ---------------------------------------------------------------------------

# THE LUMPED-VALIDITY CRITERION (design §3). Heat entering a surface penetrates
# roughly `d = sqrt(alpha*t)`; with `alpha ~ 1.5e-7 m2/s` for wood a 60 s
# ignition exposure reaches 3.0 mm. A lumped SINGLE-TEMPERATURE node is a
# faithful model of an object thin compared to that depth -- the textbook
# small-Biot regime -- and 6 mm is the ruled edge of it.
#
# The failure mode beyond the limit is GRACEFUL AND ONE-DIRECTIONAL: a too-thick
# lump spreads incident heat through more mass than really participates, so it
# ignites SLOWER than reality, never faster. There is no blow-up, only a growing
# conservative error. That is why an exemption can be carried at all -- and why
# nothing near a full 154 kg tile can be.
THIN_LIMIT_M = 0.006

# The ONE recorded exemption from the lumped criterion, carried explicitly with
# its reason in the style of the `ingress-exempt:` convention (design §6/§11.2).
# An exemption is a DECISION with a stated price, never a silent pass: a row not
# in this dict that breaks the criterion is REFUSED BY NAME at the door.
LUMPED_CRITERION_EXEMPT = {
    # lumped-exempt: 5 kg of crate stock is ~10.8 mm equivalent slab, outside
    # the 6 mm criterion. Accepted as a STATED APPROXIMATION (design §6): the
    # error is conservative (it under-predicts ignition speed), it is the
    # heaviest row still defensible as one node, and cover the player hides
    # behind should not catch as eagerly as a thin panel. Erik's ruling
    # 2026-09-21; do NOT "fix" this by thinning the row.
    "furniture": "ACCEPTED GAP (design §6): ~10.8 mm equivalent slab, outside "
                 "the 6 mm lumped criterion. The error is conservative (slower "
                 "ignition than reality) and the row is deliberately the "
                 "heaviest one-node fuel in the game.",
}

# Columns that are DERIVED at this door and must never appear in a row. Each
# maps to the authored inputs it comes from, so the refusal says what to author
# instead. (`thermal_mass` has its own, older refusal with the gas-regime
# exception, so it is not listed here.)
_DERIVED_NEVER_AUTHORED = {
    "fill_fraction": "`density`, `specific_heat` and `thickness_m`",
    "mass": "`density`, `specific_heat` and `thickness_m`",
    "mass_kg": "`density`, `specific_heat` and `thickness_m`",
}

# The geometric midpoint factor for the power-of-two snap. `math.sqrt` is the
# ONE transcendental-looking call the ingress rule allows (door 3: IEEE-754
# requires it correctly rounded, so it is bit-identical cross-machine).
_SQRT2 = math.sqrt(2.0)


# The SIGNED exponent floor of `thermal_mass` (M1,
# docs/thin_material_rows_design_2026-09-20.md section 5). `thermal_mass ==
# 2**s` and the engine stores `s` as `heat_inv_shift` (int32 — always signed);
# `conduction::cell_capacity_q` builds a Q16.16 capacity `1 << (s + 16)` from
# it, so `s = -16` is ONE raw count of capacity and anything below it would be
# zero. THE REPRESENTATION FLOOR, not a policy: the old floor of 1 was a guard
# against rows that could not yet be authored, and M1 lifts it.
THERMAL_MASS_EXP_MIN = -16


def pow2_snap(x, _what="value") -> int:
    """The SIGNED exponent of the nearest power of two to ``x`` **in log space**.

    Returns ``s`` such that the snapped value is ``2**s``, with
    ``s >= THERMAL_MASS_EXP_MIN``. It returned ``1 << s`` and REFUSED ``s < 0``
    before M1; the refusal was a guard against an inexpressible row, and the
    design's thin flammable rows (0.10-0.26 units) are exactly the rows it was
    waiting for. Nothing about the arithmetic changed -- `heat_inv_shift` has
    always been signed and the capacity has always been Q16.16.

    R5 keeps `thermal_mass` a power of two because it rides a bit-shift; the
    snap is geometric (a capacity is a multiplicative quantity, and report_t3.md
    section 2.1 costs the snap in log space: at worst +41.4 % / -29.3 %).

    Computed WITHOUT a logarithm. The geometric midpoint between ``2**k`` and
    ``2**(k+1)`` is ``2**k * sqrt(2)``, so the snap is a bracket-and-compare:
    exact binary scaling by 2 (never inexact) plus one correctly-rounded
    ``sqrt``. That keeps it inside the number-ingress doors with no exemption --
    unlike ``_build_conduction_tables``, whose ``math.log2`` carries one.

    Ties (an ``x`` landing exactly on ``2**k * sqrt(2)``, which no real rho*c
    does) round UP, deterministically.

    BELOW ``2**THERMAL_MASS_EXP_MIN`` it still REFUSES, by name -- a row the
    representation cannot hold is reported, never silently clamped (design v2
    section 11 property 3: a load-time error, not a clamp).
    """
    if not (x > 0.0):
        raise ValueError(f"{_what}: cannot snap a non-positive value {x!r} "
                         f"to a power of two")
    k = 0
    lo = 1.0
    while lo * 2.0 <= x:        # exact: multiplying a binary float by 2
        lo *= 2.0
        k += 1
    while lo > x:               # exact: halving is exact too
        lo *= 0.5
        k -= 1
    exp = k + 1 if x >= lo * _SQRT2 else k
    if exp < THERMAL_MASS_EXP_MIN:
        raise ValueError(
            f"{_what}: rho*c / {THERMAL_MASS_UNIT:.0f} = {x!r} snaps below "
            f"2**{THERMAL_MASS_EXP_MIN}, the thermal_mass REPRESENTATION floor "
            f"(`cell_capacity_q` builds `1 << (s + 16)`, so a smaller exponent "
            f"is zero capacity). A material this light thermally is not "
            f"expressible; report it rather than inflating the row "
            f"(design v2 R14, report_t3.md D1, thin-rows design section 5)")
    return exp


def derive_fill_fraction(thickness_m, tile_size_m, _what="material") -> float:
    """M2: the share of its tile a row's matter occupies, from its GEOMETRY.

    THE ONE PLACE this is computed. ``thickness_m is None`` means the row is
    **SOLID** — it fills its tile — and returns ``1.0``. That is every row's
    pre-M2 behaviour, and it is the one fill value that is genuinely scale-free
    (a bulkhead is solid steel however big the tile is), so stating it by
    omission bakes in no reference tile size.

    Otherwise the row is a **PANEL**: its matter spans the tile width and the
    full deck height but is only ``thickness_m`` deep, so

        fill = (thickness_m * tile_w * ceiling_h) / (tile_w^2 * ceiling_h)
             = thickness_m / tile_w

    — the ceiling height and one factor of the tile width cancel, which is why
    this is a pure LENGTH RATIO and needs no ``ceiling_h`` at all. It is also
    what makes ``dT/dt`` tile-size invariant: ``fill ∝ 1/tile_w`` exactly
    cancels the ``tile_w^2`` of the tile volume, leaving a capacity linear in
    ``tile_w`` — the same power the incident face area carries.

    REFUSES, never clamps (design §11.3): a non-positive thickness, or one that
    derives a fill outside ``(0, 1]``, is a LOAD-TIME ERROR. A row that
    overflows its tile at some tile size is an authoring bug that must be
    reported, not silently squeezed in.
    """
    if thickness_m is None:
        return 1.0
    t = float(thickness_m)
    if not (t > 0.0):
        raise ValueError(
            f"{_what}: thickness_m must be > 0 — it is the object's PHYSICAL "
            f"depth, the number the lumped-validity criterion is tested on "
            f"(design §11.3). Omit the key entirely for a SOLID row that fills "
            f"its tile; got {thickness_m!r}")
    tw = float(tile_size_m)
    if not (tw > 0.0):
        raise ValueError(
            f"{_what}: tile_size_m must be > 0 to derive a fill fraction, got "
            f"{tile_size_m!r}")
    f = t / tw
    if not (0.0 < f <= 1.0):
        raise ValueError(
            f"{_what}: thickness_m = {t!r} at tile width {tw!r} derives "
            f"fill_fraction = {f!r}, outside (0, 1] — the object does not fit "
            f"in its own tile. This is a LOAD-TIME ERROR, not a clamp (design "
            f"§11.3): either the thickness or the tile size is wrong")
    return f


def derive_thermal_mass_exp(density, specific_heat, fill_fraction=1.0,
                            _what="material") -> int:
    """R14: the ``thermal_mass`` EXPONENT from a row's real ``rho`` and ``c``.

    ``pow2_snap(rho * c * fill / THERMAL_MASS_UNIT)``, i.e. the signed ``s`` with
    ``thermal_mass == 2**s``. THE ONE PLACE this is computed -- `MaterialTable`
    calls it, the property gate calls it, and nothing else may re-derive it (the
    failure mode R14's implementation had to avoid is two sites computing
    `heat_inv_shift`).

    The EXPONENT is the primitive since M1, not the value: it IS
    `heat_inv_shift`, so the table reads one number instead of deriving a value
    and then recovering its log -- and a fractional value (2**-3 == 0.125) has
    no `bit_length` to recover it from. :func:`derive_thermal_mass` is the thin
    wrapper that states the same answer as a capacity.

    M2: ``fill_fraction`` is the THIRD physical input — the share of the tile
    the object's matter occupies, DERIVED from its authored geometry by
    :func:`derive_fill_fraction`. It is DIMENSIONLESS, so R14's cancellation
    argument survives intact: `thermal_mass` is still a ratio of two capacities
    measured on the same tile, and no absolute tile volume enters here. A
    `fill_fraction` of 1.0 (a SOLID row, the default) is exactly the pre-M2
    expression, value for value.
    """
    rho = float(density)
    c = float(specific_heat)
    if not (rho > 0.0) or not (c > 0.0):
        raise ValueError(
            f"{_what}: density and specific_heat must both be > 0 -- "
            f"`thermal_mass` is DERIVED from their product (design v2 R14); "
            f"got density={density!r}, specific_heat={specific_heat!r}")
    fill = float(fill_fraction)
    if not (0.0 < fill <= 1.0):
        raise ValueError(
            f"{_what}: fill_fraction must lie in (0, 1] -- it is the share of "
            f"the tile this row's matter occupies, DERIVED from its authored "
            f"`thickness_m` (M2, design §4); got {fill_fraction!r}")
    return pow2_snap(rho * c * fill / THERMAL_MASS_UNIT, _what)


def derive_thermal_mass(density, specific_heat, fill_fraction=1.0,
                        _what="material") -> float:
    """R14's ``thermal_mass`` VALUE: ``2.0 ** derive_thermal_mass_exp(...)``.

    A float since M1, because the column can now be below 1 (a 5 mm wood panel
    is 0.125 units). `math.ldexp` rather than `2.0 ** s`: an exact binary
    scaling with no pow, so it stays inside the number-ingress doors.
    """
    return math.ldexp(1.0, derive_thermal_mass_exp(
        density, specific_heat, fill_fraction, _what))


class MaterialTable:
    """Per-material property table, indexed by material id.

    Built from the ``[materials]`` section of ``config.toml`` (the named-key
    dict format from ch.02). Each scalar column is exposed as a 1-D numpy array
    indexed by material id (``table.hp[material_id]``); ``light_atten`` is an
    ``(N, 3)`` RGB array. Per-tile derived caches index these columns directly
    with the ``material`` grid: e.g. ``table.hp[gmap.material]``.

    Rebuild via :meth:`from_config` after a config hot-reload.
    """

    def __init__(self, materials_cfg, thermal_cfg=None, fire_cfg=None,
                 comb_cfg=None, res_factor=1):
        """Build from the ``CFG.materials`` namespace (or any equivalent).

        ``materials_cfg`` is the :class:`config.Namespace` for ``[materials]``;
        each attribute (``air``, ``hull``, ...) is itself a namespace of the
        named columns. A plain dict-of-dicts is also accepted (for tests).

        ``thermal_cfg`` is the optional ``[physics.thermal]`` namespace (or dict)
        carrying the conduction constants (``SHIFT_MIN``, ``NO_FACE``,
        ``h_conv``, ``tile_size_ref_m``, ``TICK_DT_S``, ``c_v``). When omitted
        the :data:`_THERMAL_DEFAULTS` are used so a dict-built table (tests)
        still produces a valid face-shift table. ``SHIFT_AT_REF`` /
        ``KAPPA_REF`` are GONE -- T5b / R10 retired the CFL stability anchor.

        ``fire_cfg`` is the optional ``[physics.fire]`` namespace (or dict). It
        supplies ``ignition_to_ext_delta`` for the per-material ``fire_T_ext``
        derivation (P-R3, ruling A3), plus the dials the ``ignition_seed``
        load-time check reads. When omitted the :data:`_FIRE_DEFAULTS` are used,
        for the same reason ``thermal_cfg`` has defaults.

        ``comb_cfg`` is the optional ``[physics.combustion]`` namespace (or
        dict). It is ACCEPTED AND UNUSED since R11 (thermal model v1,
        2026-09-19) deleted the load-time ignition-seed sustain check, its only
        reader. The parameter stays because it is part of this constructor's
        published signature and callers pass it positionally
        (:meth:`from_config`, ``tests/test_optics_ingress.py``), and because a
        combustion-derived material column is a live prospect on this arc.

        ``res_factor`` is the level's ``--res`` replication factor (M2, design
        §4). The row's geometry is derived at the BASE tile size and its MASS is
        then divided across the ``res_factor**2`` runtime tiles each base tile
        became, so the total combustible mass in a wall is invariant under
        ``--res`` — following the existing ``tile_size_m_base``/``res_factor``
        doctrine (``door_system.py``, ``cover_system.py``: *quantize at BASE
        resolution, replicate by res_factor*) rather than inventing a second
        convention. Deriving at the LIVE tile size instead would put ``N`` times
        the wood in a wall at ``--res N``, i.e. a dev tool that distorts the very
        thing under development. Defaults to 1, so every caller that builds a
        table without a level gets the unscaled base numbers.
        """
        ids = sorted(MATERIAL_NAMES)
        # Contiguity: ids must be 0..N-1 so an array indexed by id has no gaps.
        if ids != list(range(len(ids))):
            raise ValueError(f"MATERIAL_NAMES ids must be contiguous 0..N-1, got {ids}")
        self.n = len(ids)
        self.names = [MATERIAL_NAMES[i] for i in ids]

        rows = [self._get_row(materials_cfg, MATERIAL_NAMES[i]) for i in ids]

        for col, dtype in _SCALAR_COLUMNS.items():
            values = [self._get_field(row, name, col)
                      for row, name in zip(rows, self.names)]
            setattr(self, col, np.array(values, dtype=dtype))

        # ---- M2: GEOMETRY IN, FILL/MASS OUT (design §4, §11) ---------------
        # The tile geometry the table's material rows are derived at. This is
        # `tile_size_ref_m` — THE reference the whole thermal denomination
        # already shares (the conduction table, `rad_scale`, `J_per_count`) —
        # NOT the live level's tile size. Making only this column per-level
        # while the SOURCE side stays pinned at the reference would put the two
        # halves of the same physics on different rulers; per-level thermal
        # geometry is report_p2b.md §13 item 7 / q3, deliberately unstarted
        # (design §4's recorded narrowing).
        rf = int(res_factor)
        if rf < 1:
            raise ValueError(
                f"MaterialTable: res_factor must be >= 1 (it is the --res "
                f"replication factor a base tile was split by), got "
                f"{res_factor!r}")
        self.res_factor = rf
        self.tile_size_base_m = float(
            self._thermal_get(thermal_cfg, "tile_size_ref_m"))
        self.ceiling_h = float(self._thermal_get(thermal_cfg, "ceiling_h"))
        # V_tile at the BASE resolution; the runtime tile is this over rf**2.
        v_tile_base = self.tile_size_base_m ** 2 * self.ceiling_h

        # thickness_m: the ONE AUTHORED geometry column. Absent == SOLID (the
        # row fills its tile) — see `derive_fill_fraction`. Stored as 0.0 for
        # a solid row so the column is a plain float array; `fill_fraction`,
        # not this, is what every derivation reads.
        thickness = []
        fill = []
        for row, name in zip(rows, self.names):
            what = f"materials.{name}"
            # The derived columns may never be authored (design §11.1). Same
            # shape as R14's `thermal_mass` refusal: a second source of truth
            # for a derived number is exactly what this design exists to remove.
            for col, inputs in _DERIVED_NEVER_AUTHORED.items():
                if self._get_field_opt(row, col) is not None:
                    raise ValueError(
                        f"{what}.{col} is DERIVED at load from {inputs} plus "
                        f"the tile geometry (M2, design §4/§11.1) and must not "
                        f"be authored. Author the row's real dimensions instead "
                        f"and let the door derive this")
            t = self._get_field_opt(row, "thickness_m")
            f = derive_fill_fraction(t, self.tile_size_base_m, what)
            thickness.append(0.0 if t is None else float(t))
            fill.append(f)
        self.thickness_m = np.array(thickness, dtype=np.float64)
        self.fill_fraction = np.array(fill, dtype=np.float64)

        # THE LUMPED-VALIDITY DOOR (design §3/§11.2). A `flammable` row is a
        # row the fire model will carry as a SINGLE TEMPERATURE NODE, and that
        # is only honest for an object thin compared to the ~3 mm a 60 s
        # exposure penetrates. So a flammable row must STATE its thickness and
        # that thickness must be inside the criterion — the model REFUSES BY
        # NAME a row it cannot model, rather than modelling it badly.
        #
        # This is the door that makes Erik's ruling structural: NO MASSIVE
        # OBJECT EVER BURNS. A solid timber wall, a heavy door and a beam are
        # permanently fire-resistant, which is physically true and is accepted
        # as a game-design constraint. The old table said the opposite in four
        # places and could not deliver it: `wood` has declared itself
        # `flammable = true, ignition_temp = 300` while being a 154 kg block
        # that a 17.7 kW fire warms 0.07 K/s.
        for row, name, flam, t, f in zip(rows, self.names,
                                         self.flammable.tolist(),
                                         thickness, fill):
            if not bool(flam):
                continue
            if self._get_field_opt(row, "thickness_m") is None:
                raise ValueError(
                    f"materials.{name}: a FLAMMABLE row must author "
                    f"`thickness_m` (M2, design §11.2). The fire model carries "
                    f"a burning tile as ONE lumped temperature node, which is "
                    f"only faithful for an object thinner than the ~3 mm heat "
                    f"penetrates in a 60 s exposure — so the row must state the "
                    f"thickness the criterion is tested on. A row with no "
                    f"thickness FILLS ITS TILE ({f * 100:.0f} % fill, "
                    f"{self.density[self.names.index(name)] * v_tile_base:.0f} "
                    f"kg), and a block that heavy genuinely does not ignite: "
                    f"author it non-flammable, or give it its real dimensions")
            if t > THIN_LIMIT_M and name not in LUMPED_CRITERION_EXEMPT:
                raise ValueError(
                    f"materials.{name}: thickness_m = {t!r} exceeds "
                    f"THIN_LIMIT_M = {THIN_LIMIT_M} (design §3, the small-Biot "
                    f"lumped-validity criterion), so a single-temperature node "
                    f"is not a faithful model of it. REFUSED rather than "
                    f"modelled badly. Either thin the row, make it "
                    f"non-flammable, or — if the approximation is deliberate — "
                    f"record it in `LUMPED_CRITERION_EXEMPT` with its reason, "
                    f"the way `furniture` is")

        # mass_kg: the DERIVED combustible/thermal mass of ONE RUNTIME TILE of
        # this material. `rho * fill * V_tile_base` is the mass of a BASE tile;
        # `--res N` split that base tile into N**2 runtime tiles, so each holds
        # 1/N**2 of it and the wall's TOTAL mass is invariant (design §11.3c).
        self.mass_kg = np.array(
            [float(rho) * float(f) * v_tile_base / float(rf * rf)
             for rho, f in zip(self.density.tolist(), fill)],
            dtype=np.float64)

        # thermal_mass: DERIVED, never authored (R14 — thermal model v2 design
        # 2026-09-19). Each row states its real `density` and `specific_heat`;
        # the column is `pow2_snap(rho * c / THERMAL_MASS_UNIT)`, computed in
        # the ONE place that computes it (`derive_thermal_mass`).
        #
        # THE ONE LEGAL AUTHORED VALUE IS 0 — the GAS-THERMAL-REGIME
        # DECLARATION (air). It is not a capacity and never was: air's real
        # rho*c_v is 864.5 J/(m3.K), which is 0.0077 column units — SEVEN
        # doublings below the column's floor of 1. A gas tile's capacity is not
        # a smaller number in this column, it is a DIFFERENT REPRESENTATION
        # (`N * c_v` on the gas field), so the row declares the regime instead
        # of pretending to a capacity. Any other authored `thermal_mass` is
        # rejected by name: it would be a second source of truth for
        # `heat_inv_shift`, which is exactly what R14 exists to remove.
        #
        # M1: THE SNAP RETURNS THE EXPONENT, and `heat_inv_shift` is that
        # exponent verbatim. ONE derivation per row feeds BOTH columns, so the
        # divisor and the capacity cannot disagree; the old shape (derive a
        # value, then recover its log with `bit_length`) could not express a
        # capacity below 1 at all, and would have rounded 0.125 to zero -- i.e.
        # silently into the GAS regime.
        #
        # M2: the row's capacity is `rho * c * fill_fraction`. `fill_fraction`
        # is DIMENSIONLESS, so R14's cancellation argument is untouched — the
        # column is still a ratio of two capacities on the SAME tile — but a
        # 5 mm wood panel now carries 0.015 of a solid tile's capacity instead
        # of all of it, which is the whole of this patch.
        thermal_mass = []
        exps = []                  # None == the gas-regime declaration
        for row, name, f in zip(rows, self.names, fill):
            declared = self._get_field_opt(row, "thermal_mass")
            if declared is not None:
                if float(declared) != 0.0:
                    raise ValueError(
                        f"materials.{name}.thermal_mass is DERIVED from "
                        f"`density * specific_heat` (design v2 R14) and must "
                        f"not be authored; the only legal authored value is 0, "
                        f"which DECLARES the gas thermal regime. Got "
                        f"{declared!r} — author `density` / `specific_heat` "
                        f"instead and let the snap land the column"
                    )
                thermal_mass.append(0.0)
                exps.append(None)
                continue
            e = derive_thermal_mass_exp(
                self._get_field(row, name, "density"),
                self._get_field(row, name, "specific_heat"),
                f,
                f"materials.{name}")
            exps.append(e)
            thermal_mass.append(math.ldexp(1.0, e))
        self.thermal_mass = np.array(thermal_mass, dtype=np.float32)

        # heat_inv_shift: per-id log2(thermal_mass) (engine/06 §1.2). The
        # heat -> temperature conversion is `temperature += heat >> shift`, a
        # pure arithmetic right shift (no divide, bit-identical cross-machine),
        # so `thermal_mass` MUST be a power of two. Validate here and freeze the
        # integer shift; the per-tile cache (GameMap.heat_inv_shift) is this
        # column indexed by the material grid.
        #
        # THERMAL-MASS AXIS (docs/thermal_mass_axis_design_2026-07-25.md §2.1;
        # build addendum 2026-07-30 D2): `thermal_mass == 0` is LEGAL and means
        # "this material lives in the GAS thermal regime" — air, and any future
        # gas-like row. It is the ONLY value here that is not a power of two,
        # because it is not a divisor at all: the derived ``thermal_solid`` mask
        # (below) routes those tiles away from the shift path entirely, so the
        # stored shift is a never-read placeholder.
        #
        # M1: the shift is the snap's OWN answer (`exps`), not a log recovered
        # from the value — so a NEGATIVE exponent survives the trip. The
        # self-check below is on that round trip instead: `2**s` must be exactly
        # what the column carries. It is what the whole heat->T convert rests
        # on and it is free to assert.
        shifts = []
        for tm, e, name in zip(self.thermal_mass.tolist(), exps, self.names):
            if e is None:
                shifts.append(0)             # placeholder: never read (gas regime)
                continue
            if math.ldexp(1.0, e) != float(tm):
                raise ValueError(
                    f"materials.{name}.thermal_mass {tm!r} is not 2**{e} — the "
                    f"column and `heat_inv_shift` must be the same number in "
                    f"two forms (it sits on the heat->temperature divide); "
                    f"since R14 the column is derived, so this is a "
                    f"`pow2_snap` bug, not a config error"
                )
            shifts.append(e)
        self.heat_inv_shift = np.array(shifts, dtype=np.int32)

        # thermal_solid: the per-id THERMAL-MEDIUM axis (thermal-mass design
        # §2.1/§2.2). `thermal_mass > 0` -> this material takes the SOLID
        # thermal regime (bit-shift heat->T convert, conduction
        # ambient decay); `== 0` -> the GAS regime (advection + the N-divided
        # radiative deposit, no ambient decay). M1: derived from the FLOAT
        # column, which IS the definition (`thermal_mass > 0`) rather than a
        # proxy for it — the old `int(round(tm)) > 0` would have called a
        # 0.125-unit panel a GAS, silently, the day M2 authors one. The gas
        # sentinel does not collide: 0.125 > 0 holds and 0.0 does not. This is deliberately NOT `permeability <= 0`: flow
        # (`solid`) and thermal identity are separate axes — furniture is
        # permeable (gas seeps past a crate) AND a thermal solid (a crate has
        # an object temperature). The per-tile projection is
        # ``GameMap.thermal_solid``.
        self.thermal_solid = np.array(
            [float(t) > 0.0 for t in self.thermal_mass.tolist()], dtype=bool)
        _is_ts = self.thermal_solid.tolist()

        # ---- ray-engine-v2 P1 (design v3 §2.3): the HEAT-EXTINCTION INGRESS
        # INVARIANTS, checked at the material door, each a named rejection:
        #   * 0 <= heat_atten <= 1 — the sweep's positivity rests on
        #     0 <= a <= d <= ONE (truncation only decreases, and a + b <= ONE
        #     bounds what a cell can absorb by what it received). The filter
        #     table's [0, 1] rule is the precedent.
        #   * heat_atten > 0 => thermal_mass > 0 — an absorbing cell the fold
        #     ignores (a gas-regime cell that took radiation) would be an
        #     UNCOUNTED sink: today every absorbing material is a thermal solid
        #     (temperature_solver.cpp says so, belt and braces); this makes it
        #     loud. `heat_atten == 0` on a NON-FLAMMABLE thermal solid is fine.
        #   * THE LOSS-CHANNEL INVARIANT (thermal model v2 §3.1 / §6 item 1,
        #     R12) — `flammable && thermal_solid` REQUIRES `heat_atten > 0`.
        #     With `cool_shift` deleted (R1) and conduction at real physical
        #     rates (R10), IN-PLANE RADIATION IS THE ONLY MEANINGFUL LOSS
        #     CHANNEL a solid has. A flammable thermal solid at heat_atten = 0
        #     is therefore an ENERGY RATCHET: combustion writes heat into it and
        #     nothing can take the energy out, so it climbs to T_MAX_PHYS and
        #     re-ignites its neighbours forever — silently, because every
        #     channel involved is correctly booked. A small `conductivity` would
        #     NOT save it (under R10 a cellulosic solid-solid face is exactly
        #     zero below a 256 K gap), which is why the fix has to be radiative
        #     and why this is a DOOR rather than a warning. This is design v3
        #     row 6's "foliage stays heat_atten = 0.0" being superseded by R12.
        # The per-id Q16 column below is quantized ONCE here (door 2), through
        # the optics boundary module, and projected per tile by GameMap
        # (`heat_atten_q`) exactly as `fire_T_ext_q16` is — never an inline
        # `* 65536`.
        from simulation import optics_fixed as _optics_fx
        for name, atten, is_ts, flam in zip(self.names, self.heat_atten.tolist(),
                                            _is_ts, self.flammable.tolist()):
            atten_f = float(atten)
            if not (0.0 <= atten_f <= 1.0):
                raise ValueError(
                    f"materials.{name}.heat_atten must lie in [0, 1] (it is a "
                    f"Q16 extinction coefficient on the radiation sweep's planes; "
                    f"design v3 section 2.3: 0 <= a <= d <= ONE); got {atten!r}"
                )
            if atten_f > 0.0 and not is_ts:
                raise ValueError(
                    f"materials.{name}: heat_atten = {atten!r} > 0 requires "
                    f"thermal_mass > 0 — an absorbing material in the gas thermal "
                    f"regime would take radiation the Pass-1 fold never converts "
                    f"(an uncounted sink; design v3 section 2.3)"
                )
            if bool(flam) and is_ts and atten_f <= 0.0:
                raise ValueError(
                    f"materials.{name}: a FLAMMABLE THERMAL SOLID must have "
                    f"heat_atten > 0 (got {atten!r}) — radiation is its only "
                    f"meaningful loss channel once cool_shift is gone (thermal "
                    f"model v2 R1) and conduction runs at real rates (R10), so "
                    f"at 0 it is an energy ratchet: combustion heats it and "
                    f"nothing can cool it, and it climbs to T_MAX_PHYS and "
                    f"re-ignites its neighbours forever. Give the row its real "
                    f"emissivity (R12 did this for foliage: 0.90)"
                )
        self.heat_atten_q16 = _optics_fx.quantize(self.heat_atten)

        # ---- T5b step 7 / thermal model v2 R1: `cool_shift` IS DELETED ----
        # The per-material AMBIENT-DECAY column built here (`T -= T >>
        # cool_shift`, the LOSS-side twin of `thermal_mass`) is gone, with its
        # validation, its `COOL_SHIFT` default and its per-tile projection.
        # Pass 3 is deleted on both backends: the radiation sweep computes the
        # real loss now, and a hand-rolled Newtonian relaxation beside it
        # counted the same physics twice.
        #
        # The DOOR that replaces it is the loss-channel invariant above: a
        # flammable thermal solid must have `heat_atten > 0`. That is the same
        # guarantee this column used to provide (furniture's conductivity is 0,
        # so the decay was its ONLY loss channel) -- stated as a physical
        # requirement instead of a dial.
        #
        # A row may still CARRY a `cool_shift` key; it is ignored rather than
        # rejected, so a level or config that has not been swept still loads.
        # config.toml's own rows are struck in the same commit.

        # fuel_recip: the per-id FUEL-FRACTION NORMALISER — the reciprocal of
        # this material's OWN full-health `hp`, baked once at LOAD in the exact
        # form `fixedpoint::make_recip` bakes it (fuel-fraction axis,
        # 2026-07-30). The fire logistic's fuel term is
        #     F = clamp01(wall_hp[i] / <this tile's full hp>)
        # "the fraction of THIS tile's fuel still left", so the divisor is a
        # per-material quantity by nature. It was one global
        # ([physics.fire] fuel_ref = 60.0) — which is WOOD's hp — so every
        # material whose hp differs from wood's read a permanently wrong fuel
        # fraction: a brand-new furniture crate (hp 30) reported F = 0.5, i.e.
        # half burnt out the instant it was placed. Since sustain needs
        # k_die/k_grow < a/(1-a) with a = F*o2f*hot, that halving alone put a
        # crate fire below the sustain ceiling at ambient O2 at ANY intensity
        # or temperature. Lowering the global instead is not a fix: at
        # fuel_ref = 30 wood (hp 60) would clamp at F = 1 until it had already
        # lost half its mass, destroying its burn-down curve. One number cannot
        # serve two materials — the same argument `thermal_mass`, `cool_shift`
        # and `fire_T_ext` each won before it. The per-tile projection is
        # ``GameMap.fuel_recip``.
        #
        # DERIVED, NOT A DIAL: there is no `fuel_recip` config column and there
        # must never be one — it is a pure function of the row's existing `hp`,
        # so the fuel fraction and the health bar can never disagree.
        self.fuel_recip = np.array(
            [fuel_recip_from_hp(v) for v in self.hp.tolist()], dtype=np.int64)

        # ---- R14's FUEL HALF: `fuel_per_o2`, DERIVED per material (T5b) ----
        # The hp a tile pays per unit of O2 its own fire consumes, set so that
        # spending the whole bar consumes exactly the tile's real combustible
        # mass (the module-level KG_FUEL_PER_N_O2 block carries the derivation
        # and its three citations):
        #
        #     fuel_per_o2[mat] = hp[mat] * KG_FUEL_PER_N_O2 / (density * V_tile)
        #
        # DERIVED, NOT A DIAL, and for the same reason `fuel_recip` is: a second
        # authored number here could disagree with the row's own mass, which is
        # precisely the inconsistency R14 exists to remove. The global
        # `[physics.combustion] fuel_per_o2` survives ONLY as the solver's
        # fallback when a caller supplies no per-tile plane (the `fuel_ref` /
        # `o2_frac_amb` tombstone precedent); the live engine always supplies
        # one.
        #
        # THE CONSEQUENCE R14 FLAGGED IS CLOSED BY M2 (report_t5b.md §12 q2,
        # ruled by Erik 2026-09-20/21). Under R14 every flammable row shared one
        # density (555) and therefore one 154 kg block and one 414-unit O2
        # store: the fuel distinction between `wood`, `furniture`, `kindling`
        # and `foliage` had COLLAPSED. The fix is NOT fake bulk densities -- it
        # is GEOMETRY. Each row keeps the real cited density of the matter it is
        # made of and states how much of that matter is actually there, so the
        # four rows now hold 2.3 / 5.0 / 2.0 / 2.0 kg and differ honestly.
        #
        # M2 (design §8): the mass is now the row's DERIVED `mass_kg` — its
        # real combustible mass after geometry, per RUNTIME tile — instead of
        # `density * V_tile`, which assumed every row filled its tile. That
        # single substitution is what makes "the fuel store binds, not the
        # timer" fall out for free: `wood` holds 6.2 units of O2 instead of
        # 414, so the same 60 hp bar is spent over a 67x stronger physical
        # channel. No new mechanism, and the whole bar is still consumed
        # exactly when all the fuel is burned.
        fpo = []
        for name, hp_v, mass_kg in zip(self.names, self.hp.tolist(),
                                       self.mass_kg.tolist()):
            o2_units = float(mass_kg) / KG_FUEL_PER_N_O2
            fpo.append(float(hp_v) / o2_units if o2_units > 0.0 else 0.0)
        self.fuel_per_o2 = np.array(fpo, dtype=np.float64)
        self.fuel_per_o2_q16 = np.array([quantize_q16(v) for v in fpo],
                                        dtype=np.int32)
        for name, hp_v, q in zip(self.names, self.hp.tolist(),
                                 self.fuel_per_o2_q16.tolist()):
            if bool(self.flammable[list(self.names).index(name)]) and q <= 0:
                raise ValueError(
                    f"materials.{name}: a FLAMMABLE row derived "
                    f"fuel_per_o2 = 0 (hp={hp_v}) -- it would burn for ever, "
                    f"since its hp bar never empties. Give it a positive `hp` "
                    f"or make it non-flammable (design v2 R14)")

        # fire_T_ext / fire_T_ext_q16: the per-id EXTINCTION TEMPERATURE — the
        # foot of the fire logistic's `hot` ramp, `hot = clamp01((T -
        # fire_T_ext[mat]) / fire_T_span)` (P-R3, ruling A3 ride-along
        # 2026-07-31). DERIVED from this row's own `ignition_temp` minus the ONE
        # new global `[physics.fire] ignition_to_ext_delta`; see
        # `fire_T_ext_from_ignition` above for why, and why non-flammables are
        # derived rather than special-cased.
        #
        # It was one global ([physics.fire] fire_T_ext = 350) standing in for a
        # per-material quantity, and at 350 it exceeded BOTH shipped ignition
        # temps (wood 300, furniture 280) — so a tile could ignite at 300 and
        # sit permanently below its own extinction floor. Same defect shape as
        # `fuel_recip` (wood's hp for every material), `cool_shift` (one e-fold
        # for every material) and `o2_frac_amb` (ambient as the full-response
        # reference) before it.
        #
        # The _q16 column is the one the sim reads: QUANTIZED ONCE AT LOAD into
        # the Q16.16 domain shared by `heat`/`temperature`, in exactly the form
        # C++ `fixedpoint::quantize` bakes it, so the per-tile projection
        # (`GameMap.fire_T_ext_plane`) is a direct integer subtrahend for
        # `T - fire_T_ext[i]` — no per-tick rescale, no float on the gate path.
        # int32 because that IS the plane's dtype at the C++ boundary.
        delta = float(self._fire_get(fire_cfg, "ignition_to_ext_delta"))
        self.ignition_to_ext_delta = delta
        self.fire_T_ext = np.array(
            [fire_T_ext_from_ignition(it, delta)
             for it in self.ignition_temp.tolist()], dtype=np.float32)
        self.fire_T_ext_q16 = np.array(
            [quantize_q16(v) for v in self.fire_T_ext.tolist()], dtype=np.int32)

        # --- Conduction face-shift tables (engine/06 §2.4–§2.5) ---------------
        # All log2 / harmonic-mean / division happens HERE, at LOAD, in float;
        # the runtime conduction pass is a pure signed-add + arithmetic shift.
        self._build_conduction_tables(thermal_cfg)

        # ignition_temp_q16: the per-material ignition threshold QUANTIZED ONCE
        # AT LOAD into the Q16.16 fixed-point domain shared by `heat` /
        # `temperature` (engine/06 §3, proposal §1.2/§7.1). Stored as
        # round(ignition_temp * TEMP_SCALE) with a pinned rounding mode, so the
        # ignition consumer's runtime test is a direct integer compare
        # `temperature[tile] >= ignition_temp_q16[material]` against the Q16.16
        # `temperature` field — no per-tick rescale, no float on the threshold
        # path. This is the single most determinism-critical conversion in the
        # system; it is fixed here, never recomputed per tick. int64 so the
        # multiply can't overflow before it lands (the field itself is int32, but
        # a threshold beyond INT32_MAX simply never fires, which is correct).
        temp_scale = self._thermal_get(thermal_cfg, "TEMP_SCALE")
        self.ignition_temp_q16 = np.array(
            [int(round(float(it) * temp_scale))
             for it in self.ignition_temp.tolist()],
            dtype=np.int64,
        )

        # light_atten: per-channel RGB, (N, 3) float32.
        atten = np.zeros((self.n, 3), dtype=np.float32)
        for idx, (row, name) in enumerate(zip(rows, self.names)):
            triple = self._get_field(row, name, "light_atten")
            arr = np.asarray(triple, dtype=np.float32)
            if arr.shape != (3,):
                raise ValueError(
                    f"materials.{name}.light_atten must be a [R,G,B] triple, "
                    f"got {triple!r}"
                )
            atten[idx] = arr
        self.light_atten = atten
        # ---- ray-engine-v2 P6a (design v3 §2.3 / §4.1 / §5): the LIGHT
        # extinction's ingress and its Q16 column. The sweep's light channels
        # read `a_c` per channel exactly as the heat channel reads `a` -- the
        # material share absorbs AND emits (Kirchhoff per channel) -- so the
        # invariant is the heat door's, per channel: 0 <= light_atten <= 1,
        # rejected by name, never clamped. No `thermal_mass` rule: absorbed
        # light is not booked into any temperature (light has no material
        # feedback, §4.1), so a light-absorbing gas-regime row is not an
        # uncounted sink. Quantized ONCE here through the optics boundary module
        # and projected per tile by GameMap (`light_atten_q`), the heat plane's
        # seam.
        for name, trip in zip(self.names, self.light_atten.tolist()):
            if not all(0.0 <= float(v) <= 1.0 for v in trip):
                raise ValueError(
                    f"materials.{name}.light_atten must lie in [0, 1] on every "
                    f"channel (a Q16 extinction on the radiation sweep's light "
                    f"planes; design v3 section 2.3: 0 <= a <= d <= ONE); got "
                    f"{trip!r}")
        from simulation import optics_fixed as _optics_fx_l
        self.light_atten_q16 = np.ascontiguousarray(
            _optics_fx_l.quantize(self.light_atten), dtype=np.int32)   # (N, 3)

        # permeability: gas + smoke flow coefficient (0 = sealed wall, 1 = open
        # air; partial = a leaky/porous material). OPTIONAL column — if a
        # material omits it, derive a behaviour-preserving default: sealed where
        # the material occludes light, open otherwise (== the legacy is_wall
        # set). Config may set it explicitly to decouple flow from optics — e.g.
        # a grill is opaque-ish to nothing but highly permeable, glass is opaque
        # to flow but clear to light. Consumed by the gas/smoke boundary
        # (ch.02/03/04); see docs/architecture/engine/03_material_system.md.
        occludes_per_id = self.light_atten.max(axis=1) > 0.0
        perm = []
        for idx, (row, name) in enumerate(zip(rows, self.names)):
            val = self._get_field_opt(row, "permeability")
            perm.append(float(val) if val is not None
                        else (0.0 if occludes_per_id[idx] else 1.0))
        self.permeability = np.array(perm, dtype=np.float32)

        # burst_threshold: max pressure DIFFERENTIAL a wall tile can hold across
        # its opposing sides before it fails (over-pressure relief valve, ch.04
        # §5). OPTIONAL column — a material that omits it defaults to 0.0, which
        # the burst scan treats as "n/a" (a 0-threshold material never bursts;
        # see GameMap.find_burst_walls). Interior pressure is ~1.0, so real
        # thresholds sit comfortably above 1 to never pop a normal ship.
        burst = []
        for row, name in zip(rows, self.names):
            val = self._get_field_opt(row, "burst_threshold")
            burst.append(float(val) if val is not None else 0.0)
        self.burst_threshold = np.array(burst, dtype=np.float32)

        # cover_exposure: the exposure-vs-cover probability (mechanics/03 §3,
        # mechanics/06 §5 — the W2 attack resolver). 1.0 = no concealment (the
        # lazy-roll rule: a shot approaching through this tile draws NOTHING);
        # < 1.0 = soft cover — a marching shot entering a unit footprint from
        # this tile connects with probability cover_exposure, else it is
        # absorbed by the tile (wall-damage chew). OPTIONAL column defaulting
        # to 1.0 so dict-built test tables stay valid; config.toml authors it
        # EXPLICITLY on every row. Consumed as a load-time float32 constant
        # compared against a door-4 uniform draw (an exact compare — the cast
        # here is the once-at-load ingress step; see attack_resolver).
        cover = []
        for row, name in zip(rows, self.names):
            val = self._get_field_opt(row, "cover_exposure")
            cover.append(float(val) if val is not None else 1.0)
        self.cover_exposure = np.array(cover, dtype=np.float32)

    # -- conduction face-shift tables (engine/06 §2.4–§2.5) --------------
    def _build_conduction_tables(self, thermal_cfg):
        """Build ``face_shift_table[N][N]`` (and ``self_shift[N]``, its
        diagonal) at REAL PHYSICAL RATES -- thermal model v2 **R10**, derived in
        ``report_t3.md`` D2/D3.

        **WHAT R10 CHANGED.** The old table was a log bucket anchored on
        ``SHIFT_AT_REF = 2`` at ``KAPPA_REF = 50``: "hull conducts a quarter of
        the gap per tick", which is the fastest rate an explicit 4-neighbour
        stencil is stable at. That is a CFL bound wearing a physics hat -- it
        has no capacity, no tile size and no timestep in it, so it could not
        have been right by construction, and it ran solid-solid conduction
        65 000-130 000x too fast (T3 D2 section 3.2). Both constants are
        retired. The rate is now

            2^-s  =  the real per-tick fraction of the gap this face moves

        and ``s`` is derived per PAIR from matter and geometry alone.

        **TWO LAWS, because a solid|gas interface is not conduction.**

        *Solid|solid and gas|gas* -- Fourier conduction through two half-cells
        in series, which is exactly what the harmonic mean of the two
        conductivities already expressed::

            hm = 2*ka*kb / (ka + kb)                      [W/(m.K)]
            s  = round(log2( rho_c_min * dx^2 / (hm * dt) ))

        *Solid|gas* -- CONVECTION through a sub-tile boundary layer, not a
        half-tile of still air. ``h`` is a measured quantity with standard
        correlations (Churchill & Chu 1975; Incropera eq. 9.26), so this sits
        inside R7: pure conduction is simply the WRONG LAW at a wall, and using
        it would have made this face 64x too weak. The conductance is the
        boundary layer ALONE -- a convecting gas cell is well-mixed, which is
        what ``h`` already describes, and the solid's own half-cell resistance
        is not added (T3 section 8 q5's second option, "use h alone")::

            s  = round(log2( rho_c_min * dx / (h_conv * dt) ))

        Note ``dx``, not ``dx^2``: ``h`` is a conductance per unit area,
        ``kappa`` is not. At the shipped numbers this lands on **shift 10 for
        every solid|gas pair, which is exactly what the retired anchor happened
        to ship** -- T3 measured the live face and found it already implements
        ``h = 6.75 W/(m2.K)`` against the derived 6.0, inside the shift
        quantisation. So nothing moves at the wall; what changes is that the
        rate is derived instead of accidental.

        ``rho_c_min`` is the SMALLER-capacity side of the pair --
        ``thermal_mass * THERMAL_MASS_UNIT`` for a thermal solid, ``c_v *
        THERMAL_MASS_UNIT`` (air's real rho*c_v, 864.5 J/(m3.K)) for a gas cell
        at N = 1. It is the side that responds fastest, and it is the capacity
        the solver's own ``min(cap_i, cap_j)`` prices the face quantum at.

        **``kappa == 0`` still means NO FACE, on both laws.** It is the
        structural no-conduction declaration (``furniture`` / ``kindling`` /
        ``foliage``: a crate burns, it does not conduct), and T3 section 8 q3
        leaves opening those rows' convection face to Erik. A row that wants a
        face states a conductivity.

        **THE TABLE IS NOW TILE-SIZE DEPENDENT** (T3 section 3.5) -- the solid
        face goes as ``1/dx^2``, the gas face as ``1/dx`` -- and this table is
        GLOBAL while ``tile_size_m`` is per level. It is built at
        ``tile_size_ref_m`` (0.333, the shipped-level value), exactly as
        ``rad_scale_derived`` already is. On a 1.0 m level every solid face is
        then 3 shifts too fast and every gas face 2. Deriving per level is
        T3 section 8 q9, Erik's open question; this is the same position
        ``report_p2b.md`` section 13 item 7 records for the emission scale,
        deliberately not decided here.

        Everything below is a LOAD-TIME float computation; the runtime pass only
        indexes ``face_shift_table[mat_a][mat_b]`` and shifts -- no float, no
        division, bit-identical cross-machine (proposal section 2.7).
        """
        shift_min = int(self._thermal_get(thermal_cfg, "SHIFT_MIN"))
        no_face = int(self._thermal_get(thermal_cfg, "NO_FACE"))
        h_conv = float(self._thermal_get(thermal_cfg, "h_conv"))
        dx = float(self._thermal_get(thermal_cfg, "tile_size_ref_m"))
        dt = float(self._thermal_get(thermal_cfg, "TICK_DT_S"))
        c_v = float(self._thermal_get(thermal_cfg, "c_v"))
        self.no_face = no_face
        if not (h_conv > 0.0 and dx > 0.0 and dt > 0.0 and c_v > 0.0):
            raise ValueError(
                f"[physics.thermal]: the conduction table needs h_conv > 0, "
                f"tile_size_ref_m > 0, c_v > 0 and a positive tick dt; got "
                f"h_conv={h_conv!r}, tile_size_ref_m={dx!r}, c_v={c_v!r}, "
                f"dt={dt!r}")

        # BOTH SIDES OF THE RATE ARE TILE-AVERAGED, OR NEITHER IS (M2).
        #
        # `rho_c` below is the TILE-AVERAGED capacity: since M2 it carries the
        # row's `fill_fraction`, because `thermal_mass` does. The conductance
        # must carry the SAME factor or the pair is inconsistent -- and the
        # inconsistency is not small. A tile holding a 5 mm wood panel has 67x
        # less matter in it, so it has 67x less capacity AND 67x less
        # cross-section for heat to travel along:
        #
        #     conductance per tile pair = kappa * (thickness * ceiling_h) / dx
        #     tile capacity             = rho*c * (thickness * ceiling_h * dx)
        #     => dT/dt = kappa * dT / (rho*c * dx^2)     -- the fill CANCELS
        #
        # which is just the statement that in-plane thermal DIFFUSIVITY
        # `alpha = kappa/(rho*c)` is a property of the MATTER: making a wall
        # thinner does not make heat travel along it faster. Scaling only the
        # capacity would have made `wood`'s solid|solid face shift 24 -> 18,
        # i.e. heat spreading along a wooden wall 64x faster than wood does,
        # with nothing in the design asking for it.
        #
        # FOUND BY LOOKING, not by a gate: the design's own consumer list does
        # not name the conduction table (M1's finding, made standing guidance).
        # `tests/test_m2_dimensions_and_thin_rows.py` now pins the property.
        fill = self.fill_fraction.astype(np.float64)
        kappa = self.conductivity.astype(np.float64) * fill
        n = self.n

        # rho*c per material in SI, from the SAME columns `thermal_mass` itself
        # derives from -- never a second source of truth (R14). A gas row
        # (`thermal_mass == 0`) is priced at `c_v` column units, which IS air's
        # real rho*c_v: 0.0076849 * 112500 = 864.5 J/(m3.K). (A gas row's fill
        # is 1.0 -- a tile of air is full of air -- so the gas branch is
        # untouched by the tile-averaging above.)
        rho_c = np.where(self.thermal_solid,
                         self.thermal_mass.astype(np.float64) * THERMAL_MASS_UNIT,
                         c_v * THERMAL_MASS_UNIT)

        def _clamp_shift(s):
            s = int(round(s))
            if s < shift_min:
                s = shift_min
            if s > no_face:
                s = no_face
            return s

        # face_shift_table[a][b] -- symmetric NxN, one of the two laws per pair.
        face = np.full((n, n), no_face, dtype=np.int32)
        for a in range(n):
            ka = kappa[a]
            for b in range(n):
                kb = kappa[b]
                if ka <= 0.0 or kb <= 0.0:
                    face[a, b] = no_face        # kappa==0 either side -> no face
                    continue
                rc_min = min(float(rho_c[a]), float(rho_c[b]))
                if bool(self.thermal_solid[a]) != bool(self.thermal_solid[b]):
                    # solid|gas -- convection through the boundary layer alone.
                    gap_frac = (h_conv * dt) / (rc_min * dx)
                else:
                    # solid|solid or gas|gas -- Fourier, two half-cells in series.
                    hm = 2.0 * ka * kb / (ka + kb)
                    gap_frac = (hm * dt) / (rc_min * dx * dx)
                # ingress-exempt: config-time table build. log2 of a positive
                # double, rounded to an INTEGER shift -- the rounded integer is
                # empirically cross-machine stable (Ada 2026-07 per-field run),
                # and the derived shifts sit far from a .5 boundary (the closest
                # is air|air at 16.45). TODO(stats-redesign): replace with an
                # integer log2 (bit_length) to close the door.
                face[a, b] = _clamp_shift(-math.log2(gap_frac))
        self.face_shift_table = face

        # self_shift[a] -- a material's face with ITSELF. It used to be an
        # independently-computed log bucket (`SHIFT_AT_REF - log2(kappa/
        # KAPPA_REF)`), i.e. a second source of truth for the same physical
        # rate; it is now simply the table's diagonal, so the two cannot
        # disagree. Nothing in the engine reads it -- it is a reporting and
        # test-facing column.
        self.self_shift = np.array([face[a, a] for a in range(n)], dtype=np.int32)

    # -- accessors -------------------------------------------------------
    @staticmethod
    def _fire_get(fire_cfg, name):
        """Read one ``[physics.fire]`` constant, falling back to
        :data:`_FIRE_DEFAULTS` so a dict-built / config-less table still bakes a
        valid ``fire_T_ext`` column. Accepts a namespace or a plain dict."""
        if fire_cfg is None:
            return _FIRE_DEFAULTS[name]
        if isinstance(fire_cfg, dict):
            return fire_cfg.get(name, _FIRE_DEFAULTS[name])
        return getattr(fire_cfg, name, _FIRE_DEFAULTS[name])

    @staticmethod
    def _thermal_get(thermal_cfg, name):
        """Read one ``[physics.thermal]`` constant, falling back to
        :data:`_THERMAL_DEFAULTS` so a dict-built / config-less table still
        produces valid load-time tables. Accepts a namespace or a plain dict."""
        if thermal_cfg is None:
            return _THERMAL_DEFAULTS[name]
        if isinstance(thermal_cfg, dict):
            return thermal_cfg.get(name, _THERMAL_DEFAULTS[name])
        return getattr(thermal_cfg, name, _THERMAL_DEFAULTS[name])

    @staticmethod
    def _get_row(cfg, name):
        if isinstance(cfg, dict):
            if name not in cfg:
                raise KeyError(f"config [materials] missing required material '{name}'")
            return cfg[name]
        if not hasattr(cfg, name):
            raise KeyError(f"config [materials] missing required material '{name}'")
        return getattr(cfg, name)

    @staticmethod
    def _get_field(row, mat_name, col):
        if isinstance(row, dict):
            if col not in row:
                raise KeyError(f"materials.{mat_name} missing column '{col}'")
            return row[col]
        if not hasattr(row, col):
            raise KeyError(f"materials.{mat_name} missing column '{col}'")
        return getattr(row, col)

    @staticmethod
    def _get_field_opt(row, col):
        """Like ``_get_field`` but returns ``None`` for an absent column.

        Used for *optional* columns (e.g. ``permeability``) that carry a
        derived default when omitted, so existing configs need no edit.
        """
        if isinstance(row, dict):
            return row.get(col)
        return getattr(row, col, None)

    @classmethod
    def from_config(cls, cfg=None, level_data=None):
        """Build from the global :data:`config.CFG` (or a provided config).

        Threads the ``[physics.thermal]`` namespace (conduction log-bucket
        constants) and the ``[physics.fire]`` namespace (the
        ``ignition_to_ext_delta`` the per-material ``fire_T_ext`` derives from,
        plus the dials the ignition-seed check reads) into the table so both
        track config edits. Tolerates a config without either block (falls back
        to defaults).

        ``level_data`` is optional and supplies ONE thing (M2, design §4): the
        level's ``res_factor``, so a ``--res N`` run divides each row's derived
        mass across the ``N**2`` runtime tiles its base tile became. Omitted (a
        tool, a bench, a unit test) the table is built at ``res_factor = 1``,
        i.e. the unscaled base numbers. NOTE this is deliberately NOT the
        level's tile SIZE: the material table's geometry stays pinned to
        ``tile_size_ref_m``, the same reference the conduction table and
        ``rad_scale`` are built at, because making one half of the thermal model
        per-level while the other half stays pinned would put them on different
        rulers (q3; design §4's recorded narrowing).
        """
        if cfg is None:
            from config import CFG
            cfg = CFG
        thermal_cfg = getattr(getattr(cfg, "physics", None), "thermal", None)
        # T5b / R10: the conduction table is `2^-s = rate * dt`, so it needs the
        # TICK, and the tick has exactly one source of truth -- [clock]
        # ticks_per_second. It is injected here rather than duplicated as a
        # [physics.thermal] key, so the table can never disagree with the clock
        # the engine actually runs at.
        # ...and `ceiling_h`, R14's fuel-mass derivation's other geometric
        # input, whose one source of truth is [physics.water] (the water
        # solver's own air column). Same reason: no second copy to drift.
        over = {}
        tps = float(getattr(getattr(cfg, "clock", None), "ticks_per_second", 0.0))
        if tps > 0.0:
            over["TICK_DT_S"] = 1.0 / tps
        ch = getattr(getattr(getattr(cfg, "physics", None), "water", None),
                     "ceiling_h", None)
        if ch is not None and float(ch) > 0.0:
            over["ceiling_h"] = float(ch)
        if over:
            thermal_cfg = _ThermalOverride(thermal_cfg, over)
        fire_cfg = getattr(getattr(cfg, "physics", None), "fire", None)
        # P-R4: the seed check's gain chain now runs through the combustion
        # fuel-bed deposit (k_fire_heat is retired), so [physics.combustion]
        # rides along too.
        comb_cfg = getattr(getattr(cfg, "physics", None), "combustion", None)
        # The --res replication factor, read the way every other consumer of it
        # reads it (gamemap.py, pump_system.py, vent_system.py): absent or 0
        # means 1.
        rf = int(getattr(level_data, "res_factor", 1) or 1) if level_data is not None else 1
        return cls(cfg.materials, thermal_cfg, fire_cfg, comb_cfg, res_factor=rf)

    def occludes(self, material_grid):
        """Static occlusion mask: a tile occludes if it attenuates any channel.

        Replaces the hardcoded ``np.isin(m, [HULL, WOOD, DOOR])`` — derived
        purely from the table so adding an opaque material needs no code edit.
        A door has ``light_atten = [1,1,1]`` so it occludes; air has
        ``[0,0,0]`` so it does not. (Glass attenuates but does not fully
        occlude — see note in ch.02; for the current behaviour-preserving
        set, only air is fully transparent.)
        """
        per_id = self.light_atten.max(axis=1) > 0.0
        return per_id[material_grid]
