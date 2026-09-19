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
_THERMAL_DEFAULTS = {
    "TEMP_SCALE": 65536,  # Q16.16, == HEAT_SCALE (shared temperature/heat domain)
    "SHIFT_AT_REF": 2,    # metal self-rate = 1/4 (fastest stable on 4-nbr)
    "SHIFT_MIN": 2,       # rate floor / stability bound (4 * 1/4 <= 1)
    "KAPPA_REF": 50.0,    # reference conductivity (hull) for the log bucket
    "NO_FACE": 63,        # sentinel: kappa==0 face / grid edge -> zero conduction
    # COOL-SHIFT AXIS (2026-07-30): the global that seeds the per-material
    # `cool_shift` column when a row omits it. Kept a live job so the axis is
    # additive — see the `cool_shift` block in __init__.
    "COOL_SHIFT": 5,
}

# COOL-SHIFT AXIS — validation bounds for the per-material `cool_shift` column
# (the per-tick ambient decay `T -= T >> cool_shift`, engine/06 §3).
#
# FLOOR: ``SHIFT_MIN`` (2), the table's existing "rate floor / stability bound"
# convention, reused here for the same reason it exists on the conduction side —
# it caps the per-tick fraction a single cell may shed at 1/4. The floor is
# LOAD-BEARING at the bottom end: shift 0 means ``T -= T``, an instant total
# wipe of the field every tick (no thermal state can exist at all), and shift 1
# halves every solid's temperature 24x a second. Neither is a dial, they are
# bugs; the loader rejects them by name.
#
# CEILING: at Q16.16 the whole physical temperature range tops out near
# ``T_MAX_PHYS * 65536 ~ 2^30``, so a shift past ~30 sheds literally 0 counts
# per tick, and 20 is already an e-fold of 2^20/24 == 12 hours of game time —
# indistinguishable from "never cools" and far likelier to be a typo (a decimal
# slip, a Kelvin value pasted into the wrong column) than an intent.
_COOL_SHIFT_MAX = 20

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
RHO_C_PIN = 0.9e6          # J/(m3.K) -- R13's pin: wood at ~12 % MC
THERMAL_MASS_PIN = 8       # the column value that pin carries
# One `thermal_mass` unit, in volumetric heat capacity: 112 500 J/(m3.K).
THERMAL_MASS_UNIT = RHO_C_PIN / THERMAL_MASS_PIN

# The geometric midpoint factor for the power-of-two snap. `math.sqrt` is the
# ONE transcendental-looking call the ingress rule allows (door 3: IEEE-754
# requires it correctly rounded, so it is bit-identical cross-machine).
_SQRT2 = math.sqrt(2.0)


def pow2_snap(x, _what="value") -> int:
    """Nearest power of two to ``x`` **in log space**, as an integer >= 1.

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
    if exp < 0:
        raise ValueError(
            f"{_what}: rho*c / {THERMAL_MASS_UNIT:.0f} = {x!r} snaps BELOW 1, "
            f"and the thermal_mass column's floor is 1 -- it must be a power of "
            f"two >= 1 (0 is taken: it declares the GAS thermal regime). A "
            f"material this light thermally is not expressible; report it "
            f"rather than inflating the row (design v2 R14, report_t3.md D1)")
    return 1 << exp


def derive_thermal_mass(density, specific_heat, _what="material") -> int:
    """R14: ``thermal_mass`` from a row's real ``rho`` and ``c``.

    ``pow2_snap(rho * c / THERMAL_MASS_UNIT)``. THE ONE PLACE this is computed
    -- `MaterialTable` calls it, the property gate calls it, and nothing else
    may re-derive it (the failure mode R14's implementation had to avoid is two
    sites computing `heat_inv_shift`).
    """
    rho = float(density)
    c = float(specific_heat)
    if not (rho > 0.0) or not (c > 0.0):
        raise ValueError(
            f"{_what}: density and specific_heat must both be > 0 -- "
            f"`thermal_mass` is DERIVED from their product (design v2 R14); "
            f"got density={density!r}, specific_heat={specific_heat!r}")
    return pow2_snap(rho * c / THERMAL_MASS_UNIT, _what)


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
                 comb_cfg=None):
        """Build from the ``CFG.materials`` namespace (or any equivalent).

        ``materials_cfg`` is the :class:`config.Namespace` for ``[materials]``;
        each attribute (``air``, ``hull``, ...) is itself a namespace of the
        named columns. A plain dict-of-dicts is also accepted (for tests).

        ``thermal_cfg`` is the optional ``[physics.thermal]`` namespace (or dict)
        carrying the conduction log-bucket constants (``SHIFT_AT_REF``,
        ``SHIFT_MIN``, ``KAPPA_REF``, ``NO_FACE``). When omitted the
        :data:`_THERMAL_DEFAULTS` are used so a dict-built table (tests) still
        produces a valid face-shift table.

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
        thermal_mass = []
        for row, name in zip(rows, self.names):
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
                continue
            thermal_mass.append(float(derive_thermal_mass(
                self._get_field(row, name, "density"),
                self._get_field(row, name, "specific_heat"),
                f"materials.{name}")))
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
        # gas-like row. It is the ONLY non-power-of-two value accepted, because
        # it is not a divisor at all: the derived ``thermal_solid`` mask
        # (below) routes those tiles away from the shift path entirely, so the
        # stored shift is a never-read placeholder. Everything >= 1 keeps
        # today's power-of-two contract exactly.
        #
        # Since R14 the column is DERIVED (above), so the power-of-two rejection
        # below is a SELF-CHECK on `pow2_snap` rather than an author-facing
        # door — kept because it is free and because it is the invariant the
        # whole heat->T convert rests on.
        shifts = []
        tm_ints = []
        for tm, name in zip(self.thermal_mass.tolist(), self.names):
            tm_int = int(round(tm))
            if tm_int == 0:
                tm_ints.append(0)
                shifts.append(0)             # placeholder: never read (gas regime)
                continue
            if tm_int < 0 or (tm_int & (tm_int - 1)) != 0:
                raise ValueError(
                    f"materials.{name}.thermal_mass must be 0 (the gas thermal "
                    f"regime) or a power of two >= 1 (it sits on the "
                    f"heat->temperature divide); got {tm!r} — since R14 the "
                    f"column is derived, so this is a `pow2_snap` bug, not a "
                    f"config error"
                )
            tm_ints.append(tm_int)
            shifts.append(tm_int.bit_length() - 1)   # log2 of a power of two
        self.heat_inv_shift = np.array(shifts, dtype=np.int32)

        # thermal_solid: the per-id THERMAL-MEDIUM axis (thermal-mass design
        # §2.1/§2.2). `thermal_mass > 0` -> this material takes the SOLID
        # thermal regime (bit-shift heat->T convert, conduction, COOL_SHIFT
        # ambient decay); `== 0` -> the GAS regime (advection + the N-divided
        # radiative deposit, no ambient decay). It is derived from the SAME
        # rounded integers the shifts are, so the mask and the divisor can
        # never disagree. This is deliberately NOT `permeability <= 0`: flow
        # (`solid`) and thermal identity are separate axes — furniture is
        # permeable (gas seeps past a crate) AND a thermal solid (a crate has
        # an object temperature). The per-tile projection is
        # ``GameMap.thermal_solid``.
        self.thermal_solid = np.array([t > 0 for t in tm_ints], dtype=bool)

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
        for name, atten, tm_int, flam in zip(self.names, self.heat_atten.tolist(),
                                             tm_ints, self.flammable.tolist()):
            atten_f = float(atten)
            if not (0.0 <= atten_f <= 1.0):
                raise ValueError(
                    f"materials.{name}.heat_atten must lie in [0, 1] (it is a "
                    f"Q16 extinction coefficient on the radiation sweep's planes; "
                    f"design v3 section 2.3: 0 <= a <= d <= ONE); got {atten!r}"
                )
            if atten_f > 0.0 and tm_int <= 0:
                raise ValueError(
                    f"materials.{name}: heat_atten = {atten!r} > 0 requires "
                    f"thermal_mass > 0 — an absorbing material in the gas thermal "
                    f"regime would take radiation the Pass-1 fold never converts "
                    f"(an uncounted sink; design v3 section 2.3)"
                )
            if bool(flam) and tm_int > 0 and atten_f <= 0.0:
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

        # cool_shift: the per-id AMBIENT-DECAY shift — the LOSS-side twin of
        # `thermal_mass` (engine/06 §3; cool-shift axis 2026-07-30). The
        # cooling pass on a THERMAL-SOLID tile is
        #     T -= T >> cool_shift          (T is ΔT above ambient)
        # so at the 24 Hz tick the e-fold time is 2^cool_shift / 24 s.
        #
        # WHY IT IS PER-MATERIAL: it was one global ([physics.thermal]
        # COOL_SHIFT) until the thermal-mass arc routed furniture into the
        # solid thermal regime. furniture carries conductivity = 0 (NO_FACE
        # both ways), so this decay is a crate's ONE loss channel — and a shift
        # fast enough for thin hull plate (5 == 1.3 s) is absurd for a wooden
        # crate, while a shift slow enough for wood (12 == 171 s) is absurd for
        # plate. One number cannot serve both; the gain side already won this
        # argument with `thermal_mass`.
        #
        # OPTIONAL COLUMN: a row that omits it inherits the global COOL_SHIFT,
        # which is exactly the pre-axis behaviour — so every dict-built table
        # (tests) and any config predating the column stays valid, and the
        # global keeps a live job instead of becoming dead weight.
        #
        # INTEGER ONLY: it is a shift count consumed by a C++ arithmetic right
        # shift, never a float — a fractional value here would be a silent
        # truncation, so it is rejected. Bounds + rationale: `_COOL_SHIFT_MAX`
        # and SHIFT_MIN above.
        #
        # The VACUUM-exposed rate is NOT a second column (that would put two
        # dials on one material and let them drift apart). It is the same
        # per-material shift with the GLOBAL OFFSET applied at the cooling site:
        #     exposed -> max(SHIFT_MIN, cool_shift - (COOL_SHIFT - COOL_SHIFT_VACUUM))
        # i.e. "vacuum sheds two shifts (4x) faster" is one rule for every
        # material. With every row seeded at COOL_SHIFT == 5 this reproduces the
        # old 5/3 pair exactly. The per-tile projection is `GameMap.cool_shift`.
        shift_min = int(self._thermal_get(thermal_cfg, "SHIFT_MIN"))
        cool_default = int(self._thermal_get(thermal_cfg, "COOL_SHIFT"))
        cool_shifts = []
        for row, name in zip(rows, self.names):
            raw = self._get_field_opt(row, "cool_shift")
            if raw is None:
                cool_shifts.append(cool_default)
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(
                    f"materials.{name}.cool_shift must be an INTEGER shift "
                    f"count (it drives the arithmetic right shift "
                    f"`T -= T >> cool_shift`); got {raw!r}"
                )
            cs = int(raw)
            if cs != raw:
                raise ValueError(
                    f"materials.{name}.cool_shift must be an INTEGER shift "
                    f"count (it drives the arithmetic right shift "
                    f"`T -= T >> cool_shift`); got {raw!r}"
                )
            if cs < shift_min:
                raise ValueError(
                    f"materials.{name}.cool_shift must be >= SHIFT_MIN "
                    f"({shift_min}) — the per-tick decay fraction is 1/2^shift, "
                    f"so 0 means `T -= T` (an instant total wipe of the "
                    f"temperature field) and 1 halves it every tick; got {cs}"
                )
            if cs > _COOL_SHIFT_MAX:
                raise ValueError(
                    f"materials.{name}.cool_shift must be <= "
                    f"{_COOL_SHIFT_MAX} — beyond that the e-fold time "
                    f"(2^shift / 24 s) exceeds 12 hours of game time, which is "
                    f"indistinguishable from 'never cools' at Q16.16 and is "
                    f"almost certainly a typo; got {cs}"
                )
            cool_shifts.append(cs)
        self.cool_shift = np.array(cool_shifts, dtype=np.int32)

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
        """Build ``self_shift[N]`` and ``face_shift_table[N][N]`` from the
        per-material ``conductivity`` column (engine/06 §2.4–§2.5, proposal §2).

        These are the LOAD-TIME float computations (base-2 log buckets + the
        harmonic-mean face resolve). The runtime conduction pass only ever
        indexes ``face_shift_table[mat_a][mat_b]`` and shifts — no float, no
        division — so the whole spread is division-free and bit-identical
        cross-machine (proposal §2.7).

        ``self_shift[a]`` — the material's own log-bucket shift (§2.4):

            shift = clamp(SHIFT_MIN,
                          round(SHIFT_AT_REF - log2(kappa / KAPPA_REF)),
                          NO_FACE)            # NO_FACE if kappa == 0

        ``face_shift_table[a][b]`` — the shift for a face BETWEEN materials a, b,
        from the HARMONIC MEAN of their conductivities (§2.5), so two resistances
        in series add (a wood/metal face conducts at ~the wood, slow, rate; an
        arithmetic mean would leak heat into insulators too fast):

            hm = 2*ka*kb / (ka + kb)
            face = clamp(SHIFT_MIN, round(-log2(hm / KAPPA_REF)), NO_FACE)
                   NO_FACE if either kappa == 0

        Symmetric N×N. NO_FACE on every face a kappa==0 material touches makes
        the air no-op STRUCTURAL (not a runtime value-branch) — see §2.6.
        """
        shift_at_ref = float(self._thermal_get(thermal_cfg, "SHIFT_AT_REF"))
        shift_min = int(self._thermal_get(thermal_cfg, "SHIFT_MIN"))
        kappa_ref = float(self._thermal_get(thermal_cfg, "KAPPA_REF"))
        no_face = int(self._thermal_get(thermal_cfg, "NO_FACE"))
        self.no_face = no_face

        kappa = self.conductivity.astype(np.float64)
        n = self.n

        def _clamp_shift(s):
            s = int(round(s))
            if s < shift_min:
                s = shift_min
            if s > no_face:
                s = no_face
            return s

        # self_shift[a] — per-material log-bucket self-rate (§2.4).
        self_shift = np.empty(n, dtype=np.int32)
        for a in range(n):
            ka = kappa[a]
            if ka <= 0.0:
                self_shift[a] = no_face
            else:
                self_shift[a] = _clamp_shift(
                    # ingress-exempt: config-time table build; log2 is exact on
                    # the power-of-two kappa ratios in config, and the rounded
                    # INTEGER shift is empirically cross-machine stable (Ada
                    # 2026-07 per-field run). TODO(stats-redesign): replace
                    # with an integer log2 (bit_length) to close the door.
                    shift_at_ref - math.log2(ka / kappa_ref)
                )
        self.self_shift = self_shift

        # face_shift_table[a][b] — harmonic-mean face resolve (§2.5), symmetric.
        face = np.full((n, n), no_face, dtype=np.int32)
        for a in range(n):
            ka = kappa[a]
            for b in range(n):
                kb = kappa[b]
                if ka <= 0.0 or kb <= 0.0:
                    face[a, b] = no_face        # kappa==0 either side -> no face
                    continue
                hm = 2.0 * ka * kb / (ka + kb)  # harmonic mean (one float div)
                # ingress-exempt: same config-time integer-shift build as
                # self_shift above (see TODO there).
                face[a, b] = _clamp_shift(-math.log2(hm / kappa_ref))
        self.face_shift_table = face

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
    def from_config(cls, cfg=None):
        """Build from the global :data:`config.CFG` (or a provided config).

        Threads the ``[physics.thermal]`` namespace (conduction log-bucket
        constants) and the ``[physics.fire]`` namespace (the
        ``ignition_to_ext_delta`` the per-material ``fire_T_ext`` derives from,
        plus the dials the ignition-seed check reads) into the table so both
        track config edits. Tolerates a config without either block (falls back
        to defaults).
        """
        if cfg is None:
            from config import CFG
            cfg = CFG
        thermal_cfg = getattr(getattr(cfg, "physics", None), "thermal", None)
        fire_cfg = getattr(getattr(cfg, "physics", None), "fire", None)
        # P-R4: the seed check's gain chain now runs through the combustion
        # fuel-bed deposit (k_fire_heat is retired), so [physics.combustion]
        # rides along too.
        comb_cfg = getattr(getattr(cfg, "physics", None), "combustion", None)
        return cls(cfg.materials, thermal_cfg, fire_cfg, comb_cfg)

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
