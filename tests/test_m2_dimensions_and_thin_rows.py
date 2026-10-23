"""M2 — author DIMENSIONS, derive fill; the four thin rows; the validators.

Design of record: ``docs/thin_material_rows_design_2026-09-20.md`` §4, §6, §7,
§11. Patch M2 of the thermal arc, on top of M1's signed ``thermal_mass``
exponents.

WHAT THIS FILE IS FOR. M2 moves authored values, so the goldens move with it and
a re-baselined golden records whatever the code does — it is not evidence the new
numbers are right. Every number below is therefore checked against an INDEPENDENT
ORACLE: the row's own cited ``density`` and ``specific_heat``, its authored
``thickness_m``, and the tile geometry, recomputed HERE from first principles and
never read off the derived columns the engine built.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG                                        # noqa: E402
from simulation.gamemap import GameMap                        # noqa: E402
from simulation.materials import (                            # noqa: E402
    LUMPED_CRITERION_EXEMPT, MAT_AIR, MAT_FURNITURE, MAT_HULL, MAT_KINDLING,
    MAT_WOOD, MATERIAL_NAMES, PIN_ANCHOR_TOLERANCE, REFERENCE_DENSITY,
    REFERENCE_RHO_C, REFERENCE_SPECIFIC_HEAT, REFERENCE_SUBSTANCE, RHO_C_PIN,
    THERMAL_MASS_PIN, THERMAL_MASS_UNIT, THIN_LIMIT_M, MaterialTable,
    derive_fill_fraction, derive_thermal_mass_exp, pow2_snap,
)


TILE_REF = float(getattr(CFG.physics.thermal, "tile_size_ref_m"))
CEILING_H = float(CFG.physics.water.ceiling_h)


# ---------------------------------------------------------------------------
# Fixtures: a table built from a MUTABLE copy of the shipped rows, so a
# validator can be shown REFUSING something as well as accepting it.
# ---------------------------------------------------------------------------
def _rows(**edits):
    """The shipped `[materials]` block as a plain dict-of-dicts, optionally
    edited. A dict-built table is the loader's own documented test path."""
    out = {}
    for mid in sorted(MATERIAL_NAMES):
        name = MATERIAL_NAMES[mid]
        src = getattr(CFG.materials, name)
        row = {k: getattr(src, k) for k in dir(src) if not k.startswith("_")}
        out[name] = row
    for name, over in edits.items():
        for k, v in over.items():
            if v is None:
                out[name].pop(k, None)
            else:
                out[name][k] = v
    return out


def _table(**edits):
    return MaterialTable(_rows(**edits))


# ===========================================================================
# 1. The reference-substance pin (design §7)
# ===========================================================================
def test_the_unit_scale_is_a_definition_anchored_on_a_named_substance():
    """PROPERTY: **the thermal_mass unit scale is a DEFINITION; moving it
    silently re-scales every material in the game.**

    `THERMAL_MASS_UNIT` is the metre-stick every physical quantity in the
    thermal model is denominated against — `J_per_count`, the arc #54 energy
    ledger, `c_v`, the marine burn band in kW/m2. Nothing fails if it moves;
    everything just means something else, because everything is RELATIVE to it.
    That is the same shape as the 2^16 transcription error this arc spent a
    whole patch finding.

    R13 chose the value by READING IT OFF A ROW: `wood` was a solid tile, its
    rho*c was 0.899 MJ/(m3.K), and it carried `thermal_mass = 8`. After M2 no
    row is authored at full fill, so that visible anchor is gone while the value
    stays exactly correct — and the hazard is a future reader opening `wood`,
    seeing `fill_fraction = 0.015`, concluding the pin is stale and "fixing" it.

    So the pin cites a SUBSTANCE, not a row. This test holds it to that
    substance and nothing else.

    BREAKS IF: the pin is edited, the reference substance's cited numbers are
    edited without the pin being re-ruled, or the two come apart by more than
    the stated tolerance.
    """
    # The definition itself, stated as a literal so an edit is visible in a diff.
    assert RHO_C_PIN == 0.9e6
    assert THERMAL_MASS_PIN == 8
    assert THERMAL_MASS_UNIT == 112_500.0
    assert THERMAL_MASS_UNIT == RHO_C_PIN / THERMAL_MASS_PIN

    # ...and it is ANCHORED on a named substance, whose two numbers are cited.
    assert "softwood" in REFERENCE_SUBSTANCE
    assert REFERENCE_DENSITY == 555.0 and REFERENCE_SPECIFIC_HEAT == 1620.0
    assert REFERENCE_RHO_C == REFERENCE_DENSITY * REFERENCE_SPECIFIC_HEAT
    rel = abs(REFERENCE_RHO_C - RHO_C_PIN) / RHO_C_PIN
    assert rel < PIN_ANCHOR_TOLERANCE, (
        f"the ruled pin {RHO_C_PIN} and the reference substance it names "
        f"({REFERENCE_SUBSTANCE}, rho*c = {REFERENCE_RHO_C}) are {rel:.3%} "
        f"apart — beyond the {PIN_ANCHOR_TOLERANCE:.1%} band. One of them is "
        f"wrong, and the pin is a RULING: do not move it to match the row")
    # A solid tile of the reference substance IS THERMAL_MASS_PIN units, which
    # is the sentence the pin means. No shipped row need be that substance.
    assert pow2_snap(REFERENCE_RHO_C * 1.0 / THERMAL_MASS_UNIT) == 3   # 2**3 == 8


def test_the_pin_is_not_read_off_a_material_row():
    """PROPERTY: the unit scale is INDEPENDENT of the material table — doubling
    every row's density does not move it by one part.

    The structural half of design §7: a pin derived from a row lookup would
    track the rows, and then re-authoring a row would silently re-scale the
    currency every other number in the model is quoted in. This asserts the
    independence directly rather than trusting that nobody wrote the lookup.

    BREAKS IF: `THERMAL_MASS_UNIT` (or `RHO_C_PIN`) is ever computed from
    `MaterialTable`, from `CFG.materials`, or from any particular row.
    """
    before = (RHO_C_PIN, THERMAL_MASS_PIN, THERMAL_MASS_UNIT)
    rows = _rows()
    for row in rows.values():
        row["density"] = float(row["density"]) * 2.0
    tbl = MaterialTable(rows)
    import simulation.materials as M
    after = (M.RHO_C_PIN, M.THERMAL_MASS_PIN, M.THERMAL_MASS_UNIT)
    assert after == before, (
        "building a material table moved the unit scale — the pin is being "
        "read off the rows, which is exactly what design §7 forbids")
    # non-vacuity: the doubled table really is a different table.
    assert int(tbl.heat_inv_shift[MAT_HULL]) != int(
        MaterialTable.from_config(CFG).heat_inv_shift[MAT_HULL])


# ===========================================================================
# 2. The four rows, against an INDEPENDENT first-principles derivation
# ===========================================================================
def _oracle(rho, c, thickness_m, tile_w=TILE_REF, ceiling_h=CEILING_H,
            res_factor=1):
    """The whole M2 derivation, recomputed from the AUTHORED numbers alone.

    Deliberately written in the long form (object volume / tile volume) rather
    than the engine's collapsed `thickness / tile_w`, so the two are not the
    same expression twice: if the engine's shortcut is ever wrong, this
    disagrees with it.
    """
    v_tile = tile_w * tile_w * ceiling_h
    v_object = thickness_m * tile_w * ceiling_h      # a panel spanning the tile
    fill = v_object / v_tile
    mass_base = rho * v_object
    return dict(
        fill=fill,
        mass_kg=mass_base / (res_factor * res_factor),
        raw_units=rho * c * fill / THERMAL_MASS_UNIT,
        capacity_J_per_K=None,
    )


# (row, authored thickness_m, expected kg at the reference tile, expected 2**exp)
DESIGN_ROWS = (
    ("wood",      0.0050, 2.31, -3),
    ("kindling",  0.0043, 1.99, -3),
    ("foliage",   0.0043, 1.99, -3),
    ("furniture", 0.0108, 4.99, -2),
)


@pytest.mark.parametrize("name,thickness,kg,exp", DESIGN_ROWS)
def test_each_thin_row_derives_its_design_mass_and_exponent(name, thickness,
                                                            kg, exp):
    """PROPERTY: each of the four flammable rows derives the mass and capacity
    its AUTHORED geometry implies — recomputed here from `density`,
    `specific_heat`, `thickness_m` and the tile geometry, never read off the
    engine's own derived columns.

    This is the oracle for M2's numbers. The goldens are re-baselined by this
    patch, so they record whatever the code does and cannot serve; the row's
    own cited physical constants can, and do.

    The design's §6 table is reproduced to the digit by this derivation
    (wood 1.50 % / 2.31 kg / 0.1200 units, kindling and foliage 1.29 % /
    1.99 kg / 0.1032, furniture 3.24 % / 4.99 kg / 0.2596), which is why the
    thicknesses above are AUTHORED facts about objects and not fitted numbers.

    BREAKS IF: a row's thickness, density or specific_heat is edited; the fill
    derivation stops being the object-volume/tile-volume ratio; or the snap
    moves a row onto a different capacity.
    """
    tbl = MaterialTable.from_config(CFG)
    mid = tbl.names.index(name)
    assert float(tbl.thickness_m[mid]) == thickness, (
        f"{name} no longer authors thickness_m = {thickness}")

    o = _oracle(float(tbl.density[mid]), float(tbl.specific_heat[mid]), thickness)
    assert float(tbl.fill_fraction[mid]) == pytest.approx(o["fill"], rel=1e-12)
    assert float(tbl.mass_kg[mid]) == pytest.approx(o["mass_kg"], rel=1e-12)
    assert float(tbl.mass_kg[mid]) == pytest.approx(kg, abs=0.01)

    # the capacity, snapped, and the signed exponent it lands on
    assert pow2_snap(o["raw_units"]) == exp
    assert int(tbl.heat_inv_shift[mid]) == exp
    assert float(tbl.thermal_mass[mid]) == math.ldexp(1.0, exp)

    # the snap's cost, so a row that drifts to the edge of a bucket is visible
    snap_err = abs(math.ldexp(1.0, exp) - o["raw_units"]) / o["raw_units"]
    assert snap_err < 0.30, (
        f"{name}: the power-of-two snap costs {snap_err:.1%} "
        f"({o['raw_units']:.4f} -> {math.ldexp(1.0, exp)}) — R5's accepted band "
        f"is +41.4 % / -29.3 %, so this is inside it, but check the row")


def test_the_thin_rows_are_a_sixty_fold_drop_from_the_block_they_replace():
    """PROPERTY: the design's actual claim — these rows are objects a fire can
    heat, where R14's were blocks it cannot.

    A solid tile of the reference substance is 153.9 kg with a 249.5 kJ/K
    capacity; a 17.7 kW fire raises it 0.07 K/s and it never reaches its own
    `hot` gate (report_t5b §6.3, the measurement that forced this design). The
    four rows are 1.99-4.99 kg with 3.9-7.8 kJ/K, and the SAME fire takes them
    to a 300 K rise in 66-132 s — inside the 30-70 s literature band for
    piloted ignition of wood, with nothing tuned.

    Asserted as an ORDER-OF-MAGNITUDE property, not as the seconds: the exact
    time depends on M3's `H_bed`, which is not settled here.

    BREAKS IF: a flammable row goes back to full fill, or the capacity drops so
    far that ignition becomes instantaneous (which would be its own problem).
    """
    tbl = MaterialTable.from_config(CFG)
    v_tile = TILE_REF * TILE_REF * CEILING_H
    solid_block_kg = REFERENCE_DENSITY * v_tile
    assert solid_block_kg == pytest.approx(153.9, abs=0.1)
    for name, _t, _kg, _e in DESIGN_ROWS:
        mid = tbl.names.index(name)
        ratio = solid_block_kg / float(tbl.mass_kg[mid])
        assert 20.0 < ratio < 120.0, (
            f"{name} is {float(tbl.mass_kg[mid]):.2f} kg, {ratio:.0f}x lighter "
            f"than the solid block it replaces — outside the 20-120x band the "
            f"design's four rows occupy")


# ===========================================================================
# 3. The validators (design §11.1-§11.3)
# ===========================================================================
@pytest.mark.parametrize("col", ("fill_fraction", "mass", "mass_kg"))
def test_a_derived_column_may_not_be_authored(col):
    """PROPERTY (§11.1): `fill_fraction`, `mass` and `thermal_mass` are DERIVED
    and the door REFUSES a row that authors one, BY NAME.

    Same shape as R14's `thermal_mass` refusal and for the same reason: a second
    source of truth for a derived number can disagree with the row's own
    geometry, which is the inconsistency this design exists to remove.

    BREAKS IF: the refusal is dropped, or a derived column quietly becomes an
    override.
    """
    with pytest.raises(ValueError, match=col):
        _table(wood={col: 0.5})


def test_thermal_mass_may_still_not_be_authored_except_as_the_gas_sentinel():
    """PROPERTY: R14's older refusal survives M2 unchanged — `thermal_mass` may
    be authored only as the literal 0 that DECLARES the gas thermal regime.

    BREAKS IF: M2's new derived-column door replaces R14's instead of joining
    it, and the gas-regime declaration stops being legal (air would refuse to
    load) or an arbitrary capacity starts being accepted.
    """
    assert float(_table().thermal_mass[MAT_AIR]) == 0.0      # the sentinel loads
    with pytest.raises(ValueError, match="thermal_mass"):
        _table(wood={"thermal_mass": 8.0})


def test_a_flammable_row_must_author_its_thickness():
    """PROPERTY (§11.2): a `flammable` row must STATE its geometry. A row with
    no `thickness_m` fills its tile, and Erik's ruling is that no massive object
    ever burns — so the door refuses it BY NAME rather than shipping a 154 kg
    block that declares itself fuel and cannot deliver.

    That is not hypothetical: `wood` carried `flammable = true,
    ignition_temp = 300` for the whole of R14 while being exactly that block.

    BREAKS IF: the door stops requiring geometry from a flammable row, which
    would let the 154 kg fuel tile back in silently.
    """
    with pytest.raises(ValueError, match="thickness_m"):
        _table(wood={"thickness_m": None})


def test_a_flammable_row_thicker_than_the_lumped_limit_is_refused_by_name():
    """PROPERTY (§11.2/§3): the small-Biot LUMPED-VALIDITY CRITERION is a DOOR.
    A flammable row thicker than `THIN_LIMIT_M` is outside the domain where a
    single-temperature node is a faithful model of it, and the model REFUSES it
    rather than modelling it badly.

    The number is physics: heat penetrates `sqrt(alpha*t)`, ~3.0 mm in a 60 s
    ignition exposure for wood, so ~6 mm is the edge of the lumped regime.

    BREAKS IF: the criterion becomes a warning, a clamp, or is dropped — any of
    which lets a row the model cannot represent ship silently.
    """
    assert THIN_LIMIT_M == 0.006
    with pytest.raises(ValueError, match="THIN_LIMIT_M"):
        _table(wood={"thickness_m": THIN_LIMIT_M * 2.0})
    # at the limit exactly it is ACCEPTED: the criterion is inclusive, and an
    # off-by-one in the refusal alone would otherwise go unseen.
    _table(wood={"thickness_m": THIN_LIMIT_M})


def test_the_one_lumped_exemption_is_named_and_carries_its_reason():
    """PROPERTY: `furniture` at 10.8 mm is outside the criterion and is carried
    as ONE explicitly named exemption with its reason, in the style of the
    `ingress-exempt:` convention — never as a silent pass.

    ACCEPTED GAP (design §6, Erik's ruling 2026-09-21): the error is
    CONSERVATIVE — a too-thick lump spreads incident heat through more mass than
    really participates, so it ignites SLOWER than reality, never faster. Do not
    close this gap; do not thin the row.

    BREAKS IF: the exemption list grows without a reason, or a row is exempted
    that is nowhere near the criterion (the list is a door, not a bypass).
    """
    tbl = MaterialTable.from_config(CFG)
    assert set(LUMPED_CRITERION_EXEMPT) == {"furniture"}
    for name, reason in LUMPED_CRITERION_EXEMPT.items():
        mid = tbl.names.index(name)
        assert float(tbl.thickness_m[mid]) > THIN_LIMIT_M, (
            f"{name} is exempted from a criterion it already satisfies — "
            f"delete the exemption rather than carrying a dead one")
        assert float(tbl.thickness_m[mid]) < 4.0 * THIN_LIMIT_M, (
            f"{name} is exempted at {tbl.thickness_m[mid]} m, far outside the "
            f"criterion — an exemption is for a STATED APPROXIMATION, not for "
            f"a row the model cannot represent at all")
        assert len(reason) > 40 and "ACCEPTED GAP" in reason
    # every OTHER flammable row is inside the criterion on its own merits
    for mid, flam in enumerate(tbl.flammable.tolist()):
        if flam and tbl.names[mid] not in LUMPED_CRITERION_EXEMPT:
            assert float(tbl.thickness_m[mid]) <= THIN_LIMIT_M, tbl.names[mid]


def test_a_row_that_overflows_its_tile_is_a_load_time_error_not_a_clamp():
    """PROPERTY (§11.3): `thickness_m > 0`, and the derived `fill_fraction`
    lands in (0, 1]. A row that does not fit in its own tile is REPORTED, never
    silently squeezed in.

    BREAKS IF: the fill is clamped to 1.0, or a non-positive thickness is
    accepted (which would make the lumped criterion vacuous — the authored
    number IS the tested number).
    """
    with pytest.raises(ValueError, match="thickness_m"):
        derive_fill_fraction(0.0, TILE_REF)
    with pytest.raises(ValueError, match="thickness_m"):
        derive_fill_fraction(-0.005, TILE_REF)
    with pytest.raises(ValueError, match=r"fill_fraction"):
        derive_fill_fraction(TILE_REF * 1.5, TILE_REF)
    # exactly full is legal — that is a solid tile, and it is the default.
    assert derive_fill_fraction(TILE_REF, TILE_REF) == 1.0
    assert derive_fill_fraction(None, TILE_REF) == 1.0
    # and the whole shipped table lands inside the band on every shipped level
    for tile_w in (0.333, 0.5, 1.0):
        tbl = _table()
        for name, t in zip(tbl.names, tbl.thickness_m.tolist()):
            f = derive_fill_fraction(t or None, tile_w, name)
            assert 0.0 < f <= 1.0, (name, tile_w, f)


def test_an_absent_thickness_means_solid_and_that_is_every_structural_row():
    """PROPERTY: omitting `thickness_m` declares a SOLID row (fill 1.0) — the
    one fill value that is scale-free, so stating it by omission bakes in no
    reference tile size. Every structural row does exactly that, and M2 leaves
    all of them bit-identical.

    BREAKS IF: the default stops being 1.0, or a structural row acquires a
    thickness and quietly stops being solid matter.
    """
    tbl = MaterialTable.from_config(CFG)
    for mid in (MAT_AIR, MAT_HULL):
        assert float(tbl.thickness_m[mid]) == 0.0
        assert float(tbl.fill_fraction[mid]) == 1.0
    # hull is unmoved by the whole patch: a solid row's capacity is its rho*c.
    assert int(tbl.heat_inv_shift[MAT_HULL]) == derive_thermal_mass_exp(
        float(tbl.density[MAT_HULL]), float(tbl.specific_heat[MAT_HULL]), 1.0)


# ===========================================================================
# 4. §11.3b — TILE-SIZE INVARIANCE: the property authoring dimensions BUYS
# ===========================================================================
def _dT_dt(tbl, mid, tile_w, flux_w_per_m2=12_000.0 / (0.333 * 2.5)):
    """A panel row's temperature rise per second under a fixed incident FLUX.

    Incident power arrives on the exposed face, area `tile_w * ceiling_h`; the
    capacity is `rho * c * fill * V_tile`. Uses the UNSNAPPED capacity: the
    power-of-two snap is a separate quantization with its own gate, and mixing
    it in here would test the snap rather than the geometry.
    """
    v_tile = tile_w * tile_w * CEILING_H
    area = tile_w * CEILING_H
    C = (float(tbl.density[mid]) * float(tbl.specific_heat[mid])
         * float(tbl.fill_fraction[mid]) * v_tile)
    return (flux_w_per_m2 * area) / C


def test_a_panel_rows_heating_rate_is_identical_at_every_tile_size():
    """PROPERTY (§11.3b): a panel row's derived `dT/dt` under a fixed incident
    flux is IDENTICAL at `tile_size_m` 0.333 and 1.0. **This is the property
    authoring by dimensions buys, so it is the property that gates it.**

    The mechanism: capacity `C = rho*c*thickness*tile_w*ceiling_h` scales as
    `tile_w`, and the incident power arrives on a face of area
    `tile_w*ceiling_h` so it scales as `tile_w` too — `dT/dt = P/C` is therefore
    independent of tile size. A 0.5 cm panel ignites in the same number of
    seconds on every level, which is what a physical model should say.

    It is the PRIMARY justification for the whole column (design §4, Erik's
    2026-09-21 ruling): a future fork at a different tile size must get correct
    physics without re-authoring every material row.

    BREAKS IF: `fill_fraction` stops being `thickness / tile_w` — an authored
    fill, a fill built from an area or a volume ratio that does not cancel, or a
    thickness reinterpreted per level.
    """
    rates = {}
    for tile_w in (0.333, 0.5, 1.0, 2.0):
        tbl = MaterialTable(_rows(), {"tile_size_ref_m": tile_w,
                                      "ceiling_h": CEILING_H})
        rates[tile_w] = {n: _dT_dt(tbl, tbl.names.index(n), tile_w)
                         for n, _t, _k, _e in DESIGN_ROWS}
    base = rates[0.333]
    for tile_w, got in rates.items():
        for name, r in got.items():
            assert r == pytest.approx(base[name], rel=1e-12), (
                f"{name}: dT/dt = {r:.4f} K/s at tile {tile_w} m vs "
                f"{base[name]:.4f} at 0.333 — authoring by DIMENSIONS is "
                f"supposed to make this invariant (design §4/§11.3b)")
    # the rate is also just `flux / (rho*c*thickness)`, with no geometry left
    tbl = MaterialTable.from_config(CFG)
    mid = MAT_WOOD
    flux = 12_000.0 / (0.333 * 2.5)
    assert base["wood"] == pytest.approx(
        flux / (float(tbl.density[mid]) * float(tbl.specific_heat[mid])
                * float(tbl.thickness_m[mid])), rel=1e-12)


def test_an_authored_fill_fraction_would_NOT_be_tile_size_invariant():
    """NON-VACUITY for the test above: the invariance is a property of THIS
    AUTHORING, not of the arithmetic.

    Held at a FIXED fill fraction — the alternative design §4 rejects — the same
    `dT/dt` varies by exactly the tile-size ratio: the row's physical meaning
    would silently become a 1.5 cm slab on a 1.0 m level instead of a 0.5 cm
    one. Measured here so "identical" above cannot be true by construction.

    BREAKS IF: this stops discriminating, i.e. the probe has gone blind.
    """
    rho, c = REFERENCE_DENSITY, REFERENCE_SPECIFIC_HEAT
    fill = 0.015015015015015015           # wood's fill AT 0.333, authored
    flux = 12_000.0 / (0.333 * 2.5)

    def rate(tile_w):
        v_tile = tile_w * tile_w * CEILING_H
        return (flux * tile_w * CEILING_H) / (rho * c * fill * v_tile)

    assert rate(1.0) / rate(0.333) == pytest.approx(0.333, rel=1e-12), (
        "an authored fill fraction is supposed to be 3x wrong at 1.0 m — if it "
        "is not, this non-vacuity probe no longer measures anything")


# ===========================================================================
# 5. §11.3c — --res MASS INVARIANCE (a convenience property, cheap to pin)
# ===========================================================================
def test_total_combustible_mass_in_a_wall_is_invariant_under_res():
    """PROPERTY (§11.3c): the total combustible mass in a wall is IDENTICAL at
    `--res 1` and `--res 2`.

    `--res N` replicates each BASE tile into an N x N block, so the row's mass
    is derived at the base tile size and then divided across the N**2 runtime
    tiles. Deriving at the LIVE tile size instead puts N times the wood in a
    wall — a dev tool that distorts the very thing under development.

    Per Erik's 2026-09-21 ruling this is a CONVENIENCE property, not a
    foundational invariant (`--res` is a dev tool, and nothing in the design is
    contorted to preserve it). It is pinned because it is nearly free.

    BREAKS IF: someone derives the mass at the live tile size, or `res_factor`
    stops reaching the material table.
    """
    per_base_tile = {}
    for rf in (1, 2, 3):
        tbl = MaterialTable(_rows(), res_factor=rf)
        for name, _t, _k, _e in DESIGN_ROWS:
            mid = tbl.names.index(name)
            # one BASE tile became rf**2 runtime tiles
            per_base_tile.setdefault(name, []).append(
                float(tbl.mass_kg[mid]) * rf * rf)
    for name, totals in per_base_tile.items():
        assert totals[0] == pytest.approx(totals[1], rel=1e-12), name
        assert totals[0] == pytest.approx(totals[2], rel=1e-12), name

    # ...and the THERMAL capacity per tile is res-invariant on its own, because
    # `fill_fraction` is a ratio: each sub-tile is still the same panel slice.
    t1, t2 = MaterialTable(_rows(), res_factor=1), MaterialTable(_rows(), res_factor=2)
    assert np.array_equal(t1.thermal_mass, t2.thermal_mass)
    assert np.array_equal(t1.fill_fraction, t2.fill_fraction)


def test_the_engine_actually_threads_res_factor_into_the_material_table():
    """PROPERTY: the invariance above is not just a property of the table's
    constructor — it survives the REAL path, `main.py --res N` -> `_upscale_level`
    -> `GameMap` -> `MaterialTable.from_config(CFG, level)`.

    A seam that is correct in isolation and unwired is the failure mode this
    arc keeps finding (M1: three live consumers invisible to the whole suite),
    so the wiring is gated end to end and by summing the ACTUAL grid.

    BREAKS IF: `GameMap` stops passing the level, `_upscale_level` stops
    recording `res_factor`, or the division is applied somewhere that the
    per-tile projection does not see.
    """
    import copy
    from level_loader import LevelData
    from main import _upscale_level

    def wall_mass(rf):
        tm = np.zeros((8, 8), dtype=np.int32)
        tm[0, :] = 1
        tm[-1, :] = 1
        tm[:, 0] = 1
        tm[:, -1] = 1
        tm[3, 2:6] = 2                      # a four-tile WOOD wall
        lvl = LevelData(name="m2_res", version="2", path=Path("."),
                        tilemap=tm, tile_size_m=0.333, diffuse_path=Path("."))
        if rf > 1:
            _upscale_level(lvl, rf)
        g = GameMap(lvl)
        n_wood = int((g.material == MAT_WOOD).sum())
        assert n_wood == 4 * rf * rf, (rf, n_wood)
        return n_wood * float(g.materials.mass_kg[MAT_WOOD]), g

    m1, g1 = wall_mass(1)
    m2, g2 = wall_mass(2)
    assert int(g2.materials.res_factor) == 2, "res_factor never reached the table"
    assert m1 == pytest.approx(m2, rel=1e-12), (
        f"the wall holds {m1:.3f} kg of wood at --res 1 and {m2:.3f} kg at "
        f"--res 2 — the mass is being derived at the LIVE tile size")
    # non-vacuity: at --res 2 there really are 4x the tiles, each 1/4 the mass.
    assert float(g2.materials.mass_kg[MAT_WOOD]) == pytest.approx(
        float(g1.materials.mass_kg[MAT_WOOD]) / 4.0, rel=1e-12)
    # the CAPACITY per tile is unchanged, so the fire model is not re-scaled
    assert int(g1.heat_inv_shift[3, 3]) == int(g2.heat_inv_shift[6, 6])


# ===========================================================================
# 6. The consumer the design's list does not name (M1's standing finding)
# ===========================================================================
def test_in_plane_conduction_is_a_property_of_the_matter_not_the_thickness():
    """PROPERTY: a row's solid|solid conduction face shift is set by its
    MATTER's diffusivity `alpha = kappa/(rho*c)`, not by how thick the object
    is. Making a wooden wall thinner does not make heat travel ALONG it faster.

    THE MECHANISM, and why this test exists. `_build_conduction_tables` prices a
    face with `rho_c_min` taken from `thermal_mass` — which M2 made carry the
    row's `fill_fraction`. The conductance must carry the SAME factor or the
    pair is inconsistent:

        conductance per tile pair = kappa * (thickness * ceiling_h) / dx
        tile capacity             = rho*c * (thickness * ceiling_h * dx)
        => dT/dt = kappa * dT / (rho*c * dx^2)        -- the fill CANCELS

    Scaling only the capacity moved `wood|wood` from shift 24 to 18, i.e. heat
    spreading along a wooden wall 64x faster than wood does, with nothing in the
    design asking for it.

    FOUND BY LOOKING, not by a gate: the design's consumer list does not name
    the conduction table at all (M1's finding, now standing guidance — a design
    that enumerates sites may not be exhaustive).

    BREAKS IF: the tile-averaging is applied to the capacity WITHOUT the
    conductance (or the other way round).
    """
    solid = MaterialTable.from_config(CFG)
    # A homogeneous face of a row is its own diffusivity, whatever its geometry:
    # re-derive wood's self-face at its REAL matter and at full fill, and get
    # the same shift.
    thin = int(solid.face_shift_table[MAT_WOOD, MAT_WOOD])
    fat = int(MaterialTable(_rows(wood={"thickness_m": None,
                                        "flammable": False}))
              .face_shift_table[MAT_WOOD, MAT_WOOD])
    assert thin == fat, (
        f"wood|wood conducts at shift {thin} as a 5 mm panel and {fat} as a "
        f"solid block — in-plane diffusivity is a property of the MATTER")

    # non-vacuity: the probe DOES move when the matter moves.
    slower = int(MaterialTable(_rows(wood={"conductivity": 0.012}))
                 .face_shift_table[MAT_WOOD, MAT_WOOD])
    assert slower > thin, (
        "a 10x less conductive wood must conduct more slowly — this probe is "
        "blind if it does not move")


def test_the_fuel_store_is_the_rows_derived_mass_not_a_full_tile():
    """PROPERTY (§8): the O2 store a tile's fire can consume is the row's REAL
    combustible mass. That is `hp / fuel_per_o2`, and M2 makes it fall out of
    the geometry with NO NEW MECHANISM — the whole `hp` bar is still spent
    exactly when all the fuel is burned, but over 6.2 units of O2 for `wood`
    instead of 414.

    Erik's ruling: **the fuel store should bind, not the timer.** report_t5b
    §5.2 measured the `wall_damage` burn-down timer dominating the physical
    channel ~10:1; a 67x stronger physical channel is what flips that, and it
    costs zero synced state. M3 deletes the timer; M2 is what makes that cheap.

    BREAKS IF: the fuel mass goes back to `density * V_tile`, or `fuel_per_o2`
    stops being derived per row.
    """
    from simulation.materials import KG_FUEL_PER_N_O2
    tbl = MaterialTable.from_config(CFG)
    for name, _t, kg, _e in DESIGN_ROWS:
        mid = tbl.names.index(name)
        store = float(tbl.hp[mid]) / float(tbl.fuel_per_o2[mid])
        assert store == pytest.approx(
            float(tbl.mass_kg[mid]) / KG_FUEL_PER_N_O2, rel=1e-9), name
        assert store < 20.0, (
            f"{name} still holds {store:.0f} units of O2 — a full tile of wood "
            f"is 414, and the whole point of M2 is that it no longer is one")
    # and it really did move by ~two orders of magnitude for `wood`
    solid_store = (REFERENCE_DENSITY * TILE_REF * TILE_REF * CEILING_H
                   / KG_FUEL_PER_N_O2)
    mid = tbl.names.index("wood")
    assert solid_store / (float(tbl.hp[mid]) / float(tbl.fuel_per_o2[mid])) > 50.0


def test_the_kindling_row_is_the_campfire_reference_object_again():
    """PROPERTY: `kindling` is back inside the 1-3 kg band its own header has
    declared since P-F4a ("the CAMPFIRE REFERENCE OBJECT -- a 1-3 kg
    effective-class fuel row").

    R14's uniform `density = 555` at full fill flattened it to 153.9 kg and left
    the intent alive only in `hp = 8`. It is the row the fire bench is
    re-anchored on (design §8, Erik's ruling), so its mass is load-bearing for
    every number that bench reports.

    BREAKS IF: kindling is re-authored outside the band the bench's reference
    numbers are quoted against.
    """
    tbl = MaterialTable.from_config(CFG)
    kg = float(tbl.mass_kg[MAT_KINDLING])
    assert 1.0 <= kg <= 3.0, (
        f"kindling is {kg:.2f} kg, outside the 1-3 kg effective-fuel band its "
        f"own row header declares and the re-anchored fire bench is quoted at")
    # ...and it is lighter than the crate, which is the ordering the two rows
    # exist to express.
    assert kg < float(tbl.mass_kg[MAT_FURNITURE])
