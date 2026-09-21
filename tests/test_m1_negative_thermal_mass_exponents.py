"""M1 — NEGATIVE `thermal_mass` EXPONENTS (docs/thin_material_rows_design_2026-09-20.md §5).

`thermal_mass` could not go below 1 before this patch. That was a GUARD, not an
arithmetic limit: `heat_inv_shift` has always been `int32_t` — signed — and
`conduction::cell_capacity_q` has always built a Q16.16 capacity `1 << (s + 16)`
from it, where `s = -2` is `1 << 14 == 16384 ==` exactly 0.25. M1 lifts the
guard and makes every consumer of a negative exponent correct, so that M2 can
author the thin flammable rows the design needs (0.10–0.26 units).

**M1 CHANGES NO MATERIAL ROW.** Every shipped row still carries a non-negative
exponent, so the shipped engine must behave EXACTLY as before — which is why
these tests come in two halves: the NEUTRALITY half (the new signed forms are
the old forms, value for value, on every non-negative exponent) and the NEW
BEHAVIOUR half (what a negative exponent does, pinned against exact integer
arithmetic computed here rather than against anything the engine produced).

THE ORACLE PROBLEM, stated. This patch touches the digest-bound temperature
path, so a re-baselined golden would record whatever the code does and prove
nothing. It is not used here. The oracles are:
  * for neutrality — the goldens that were NOT re-baselined (M1 moved none);
  * for the new behaviour — exact Python integer arithmetic in these tests,
    and the committed integer reference
    (docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py), which was
    changed FIRST and whose gates were re-run before the C++ was written.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp                                   # noqa: E402
from simulation.materials import (                            # noqa: E402
    THERMAL_MASS_EXP_MIN, THERMAL_MASS_UNIT, derive_thermal_mass,
    derive_thermal_mass_exp, pow2_snap,
)

FP_ONE = 65536
INT32_MAX = 2 ** 31 - 1
INT64_MAX = 2 ** 63 - 1


def _q(x: float) -> int:
    """fixedpoint::quantize — round half away from zero."""
    return int(math.floor(x * FP_ONE + 0.5) if x >= 0 else math.ceil(x * FP_ONE - 0.5))


# ===========================================================================
# 1. The snap
# ===========================================================================
def test_pow2_snap_returns_the_signed_exponent_not_the_value():
    """PROPERTY: `pow2_snap(x)` returns the EXPONENT `s`, and `2**s` is the
    log-space-nearest power of two to `x` — for either sign of `s`.

    The exponent is the primitive because it IS `heat_inv_shift`. Returning the
    value instead (the pre-M1 shape) cannot express a capacity below 1 at all,
    and there is no `bit_length` to recover the log of 0.125 from.

    BREAKS IF: the snap goes back to returning `1 << exp`; or the snap stops
    being log-space nearest (the geometric midpoint 2**k * sqrt(2)).
    """
    for x, want in [(8.0, 3), (1.0, 0), (0.5, -1), (0.25, -2), (0.125, -3)]:
        assert pow2_snap(x) == want, x
    # log-space nearest, on BOTH sides of zero: 0.75 is above 0.5*sqrt(2)
    # (0.7071) so it snaps UP to 2**0; 0.70 is below it and snaps DOWN.
    assert pow2_snap(0.75) == 0
    assert pow2_snap(0.70) == -1


def test_the_design_s_thin_rows_are_expressible():
    """PROPERTY: the raw `thermal_mass` values the design's §6 table derives —
    0.120 (wood/kindling/foliage) and 0.260 (furniture) — snap to the exponents
    §6 names, WITHOUT raising.

    This is the whole purpose of M1 stated as a test: the guard used to refuse
    exactly these numbers. It pins the raw→exponent step only; M2 derives the
    raw numbers themselves from authored dimensions and owns that half.

    BREAKS IF: the `exp < 0` refusal returns, or the snap's rounding moves so a
    design row lands on a different capacity than the table was costed at.
    """
    assert pow2_snap(0.120) == -3        # -> 0.125, the +4 % snap cost §6 names
    assert pow2_snap(0.104) == -3
    assert pow2_snap(0.260) == -2        # -> 0.250
    assert derive_thermal_mass_exp(555.0, 1620.0 * 0.0150) == -3


def test_derive_thermal_mass_states_the_same_answer_as_a_capacity():
    """PROPERTY: `derive_thermal_mass(...) == 2.0 ** derive_thermal_mass_exp(...)`,
    exactly, and the value is float32-exact for every legal exponent.

    The two functions must never be able to disagree — that is the one failure
    mode R14's implementation was written to avoid (two sites computing
    `heat_inv_shift`), and M1 splits the function in two for the first time.

    BREAKS IF: one of the two grows its own arithmetic, or the value form starts
    rounding (a `float32` column that cannot hold `2**-16` exactly would make the
    round-trip self-check in `MaterialTable` fire).
    """
    for s in range(THERMAL_MASS_EXP_MIN, 13):
        rho_c = math.ldexp(1.0, s) * THERMAL_MASS_UNIT
        assert derive_thermal_mass_exp(rho_c, 1.0) == s
        v = derive_thermal_mass(rho_c, 1.0)
        assert v == math.ldexp(1.0, s)
        assert float(np.float32(v)) == v, f"2**{s} is not float32-exact"


# ===========================================================================
# 2. The capacity — property 5 of the design's §11
# ===========================================================================
def test_cap_used_round_trips_every_exponent_from_the_floor_to_the_ceiling():
    """PROPERTY (design §11.5): for a thermal solid,
    ``cell_capacity_q(s).cap_used == 2**(s + 16)`` for every
    ``s in [CAP_SHIFT_MIN, CAP_SHIFT_MAX]`` — the Q16.16 capacity IS the
    exponent, for either sign, with no drift and no clamp inside the band.

    This is the identity the whole change rests on: `s = -2` must be 16384,
    which is 0.25 in Q16.16 exactly. It is asserted against arithmetic computed
    HERE, not against anything the engine emitted.

    BREAKS IF: `if (s < 0) s = 0` comes back (every negative `s` would collapse
    to 65536); or `CAP_SHIFT_MIN` moves off `-FP_SHIFT`, which would either
    forbid representable capacities or admit a zero one.
    """
    assert bp.CAP_SHIFT_MIN == -16 == THERMAL_MASS_EXP_MIN
    for s in range(bp.CAP_SHIFT_MIN, bp.CAP_SHIFT_MAX + 1):
        cap_used, cap_real = bp.conduction_cell_capacity_q(
            True, s, 0, _q(0.01), _q(1.0))
        assert cap_used == 2 ** (s + 16), s
        assert cap_real == 2 ** (s + 16), s
    # The floor's REASON: one raw count, never zero — the conduction endpoint
    # divide must not fault.
    assert bp.conduction_cell_capacity_q(True, bp.CAP_SHIFT_MIN, 0,
                                         _q(0.01), _q(1.0))[0] == 1
    # Below the floor it CLAMPS to the floor rather than shifting by a negative
    # amount (the materials door is what refuses such a row; this is the C++
    # side's hygiene, and it must never produce 0).
    assert bp.conduction_cell_capacity_q(True, -40, 0, _q(0.01), _q(1.0))[0] == 1


def test_the_gas_sentinel_does_not_collide_with_a_sub_unit_capacity():
    """PROPERTY: `thermal_mass == 0` (the GAS-regime declaration) and
    `thermal_mass == 0.125` (a thin panel) are different things, and the C++
    tells them apart by the `is_ts` FLAG it is handed, never by the value.

    The hazard M1 creates is exactly this: `int(round(0.125)) == 0`, so any
    consumer that rounds the column to an integer would silently reclassify a
    thin panel as air. `MaterialTable.thermal_solid` is `thermal_mass > 0` on
    the float column for this reason.

    BREAKS IF: `thermal_solid` goes back to being derived from a rounded
    integer, or the C++ starts testing the capacity value instead of `is_ts`.
    """
    solid_used, _ = bp.conduction_cell_capacity_q(True, -3, 0, _q(0.01), _q(1.0))
    gas_used, _ = bp.conduction_cell_capacity_q(False, -3, 0, _q(0.01), _q(1.0))
    assert solid_used == 2 ** 13                     # 0.125 in Q16.16
    assert gas_used != solid_used                    # the gas branch, untouched

    from config import CFG
    from simulation.materials import MaterialTable
    tbl = MaterialTable.from_config(CFG)
    assert tbl.thermal_mass.dtype == np.float32
    assert tbl.heat_inv_shift.dtype == np.int32      # SIGNED, and always was


# ===========================================================================
# 3. The signed shift in the kit
# ===========================================================================
_SHIFT_OPERANDS = [0, 1, 7, 65535, 65536, 123456789, 2 ** 31 - 1, 2 ** 40 + 12345,
                   -1, -7, -65535, -123456789, -(2 ** 31), -(2 ** 40 + 12345)]


@pytest.mark.parametrize("x", _SHIFT_OPERANDS)
@pytest.mark.parametrize("s", list(range(0, 17)))
def test_the_signed_shift_is_the_old_shift_on_every_non_negative_exponent(x, s):
    """PROPERTY (NEUTRALITY): `shr_round0_signed_i64(x, s) ==
    shr_round0_i64(x, s)` for every `s >= 0`, on both signs of `x`.

    This identity is why M1 moves no golden: every shipped material row has a
    non-negative exponent, so every site that swapped to the signed twin
    computes the same integer it always did.

    BREAKS IF: the signed twin changes the rounding on the non-negative branch
    (e.g. becomes a plain `>>`, which rounds toward -inf for negative x), which
    would move the radiation fold's DC balance over a long burn.
    """
    assert bp.fp_shr_round0_signed_i64(x, s) == bp.fp_shr_round0_i64(x, s)


@pytest.mark.parametrize("x", _SHIFT_OPERANDS)
@pytest.mark.parametrize("s", list(range(-16, 0)))
def test_the_left_shift_branch_is_EXACT(x, s):
    """PROPERTY (design §11.5, second half): for `s < 0` the signed twin
    MULTIPLIES by `2**-s` and is EXACT — a left shift loses nothing, where the
    right shift truncates.

    Asserted against Python's arbitrary-precision integers, which is an oracle
    independent of the engine entirely.

    BREAKS IF: the negative branch is implemented as a divide, a truncating
    round-trip, or a `<<` on a value already narrowed to int32.
    """
    assert bp.fp_shr_round0_signed_i64(x, s) == x * 2 ** (-s)


def test_the_negative_branch_is_lossless_where_the_positive_branch_is_not():
    """PROPERTY: the two branches are NOT mirror images — dividing then
    multiplying back loses counts, multiplying then dividing back does not.

    This is the sentence in design §5 row 3 ("the negative branch is exact")
    turned into a check, so that a future reader cannot conclude the sign is
    cosmetic and "simplify" the twin into one truncating expression.

    BREAKS IF: someone implements `s < 0` as `shr_round0_i64(x, 0) << -s` via a
    narrowed intermediate, or routes both branches through one rounding step.
    """
    x = 12345678901          # not a multiple of 2**4
    up = bp.fp_shr_round0_signed_i64(x, -4)
    assert bp.fp_shr_round0_signed_i64(up, 4) == x        # lossless round trip
    down = bp.fp_shr_round0_signed_i64(x, 4)
    assert bp.fp_shr_round0_signed_i64(down, -4) != x     # the truncating one


def test_the_signed_shift_saturates_rather_than_wrapping():
    """PROPERTY: an operand large enough that `x * 2**-s` leaves int64 returns
    the int64 rail, not a wrapped value.

    Signed overflow is UB and this is a determinism TU; the rail is a
    diagnostic that no caller depends on, but it must exist.

    BREAKS IF: the twin becomes a bare `x << (-s)`.
    """
    assert bp.fp_shr_round0_signed_i64(2 ** 62, -4) == INT64_MAX
    assert bp.fp_shr_round0_signed_i64(-(2 ** 62), -4) == -INT64_MAX - 1


# ===========================================================================
# 4. THE OVERFLOW SITE — design §5 row 4, "the one real risk"
# ===========================================================================
_DIALS = dict(no_face=63, o2_vacuum_thresh=0.3, c_v=0.0076849,
              n_floor_heat=0.01, gas_advection_rate=900.0, t_max_phys=16000.0)


def _solver():
    s = bp.TemperatureSolver()
    for k, v in _DIALS.items():
        setattr(s, {"no_face": "no_face", "o2_vacuum_thresh": "o2_vacuum_thresh",
                    "c_v": "c_v", "n_floor_heat": "n_floor_heat",
                    "gas_advection_rate": "gas_advection_rate",
                    "t_max_phys": "T_MAX_PHYS"}[k], v)
    return s


def _deposit_scene(deposit: int, his: int, h=3, w=3):
    """One thermal solid at the centre of an otherwise inert grid, holding
    `deposit` raw Q16.16 heat counts and a `heat_inv_shift` of `his`. Every face
    is NO_FACE, so Pass 2 moves nothing and Pass 1's deposit is the only channel
    that can change `temperature`."""
    temp = np.zeros((h, w), dtype=np.int32)
    heat = np.zeros((h, w), dtype=np.int32)
    shift = np.zeros((h, w), dtype=np.int32)
    solid = np.zeros((h, w), dtype=bool)
    vac = np.zeros((h, w), dtype=bool)
    atm = np.full((h, w), FP_ONE, dtype=np.int32)
    fs = np.full((h, w, 4), _DIALS["no_face"], dtype=np.int32)
    ts = np.zeros((h, w), dtype=bool)
    c = (h // 2, w // 2)
    heat[c] = deposit
    shift[c] = his
    solid[c] = True
    ts[c] = True
    return (np.ascontiguousarray(temp), np.ascontiguousarray(heat),
            np.ascontiguousarray(shift), np.ascontiguousarray(fs),
            np.ascontiguousarray(solid), np.ascontiguousarray(vac),
            np.ascontiguousarray(atm), np.ascontiguousarray(ts), c)


def _run_deposit(deposit: int, his: int):
    s = _solver()
    temp, heat, shift, fs, solid, vac, atm, ts, c = _deposit_scene(deposit, his)
    hits0 = int(s.t_max_phys_hits)
    s.step(temp, heat, shift, fs, solid, vac, atm, thermal_solid=ts)
    return int(temp[c]), int(s.t_max_phys_hits) - hits0


# `deposit * 16` = 2**32 + 500_000_000: an int32 product WRAPS to a small,
# perfectly plausible positive temperature instead of saturating, so the wrong
# answer is not obviously wrong. This is the case that makes the test sharp.
_WRAP_DEPOSIT = (2 ** 32 + 500_000_000) // 16
assert _WRAP_DEPOSIT * 16 == 2 ** 32 + 500_000_000


def test_the_solid_heat_deposit_survives_a_negative_exponent_that_overflows_int32():
    """PROPERTY (design §5 row 4): at a NEGATIVE `heat_inv_shift` the Pass-1
    solid heat deposit is a MULTIPLY, and it is carried in int64 so the only
    narrowing is the saturating add — the `T_MAX_PHYS` rail is REACHED, never
    wrapped past.

    THIS TEST IS THE PATCH'S MAIN GATE AND IT MUST NOT PASS BEFORE THE FIX.
    The shipped line was `int32_t gain = deposit >> shift`. With `shift = -4`
    that is undefined behaviour; with the obvious sign-only repair
    (`(int32_t)(deposit << -shift)`) the product `deposit * 16` leaves int32 and
    WRAPS to 500 000 000 raw — about 7 629 game — which is a believable
    temperature well under the 16 000-game rail. The engine would then report a
    warm tile where the physics says a railed one, silently.

    Both numbers are computed here in Python integers; nothing in this test
    consults a golden.

    BREAKS IF: site 4's int64 routing is reverted (verified by doing exactly
    that: the assertion below fails with 500 000 000). It does NOT break if
    only the sign handling is added, which is the point.
    """
    rail = _q(_DIALS["t_max_phys"])
    got, hits = _run_deposit(_WRAP_DEPOSIT, -4)

    wrapped = (_WRAP_DEPOSIT * 16) - 2 ** 32          # what int32 would produce
    assert 0 < wrapped < rail, "the scene must make the WRONG answer plausible"

    assert got == rail, (
        f"deposit {_WRAP_DEPOSIT} at heat_inv_shift -4 landed {got}; the exact "
        f"int64 gain is {_WRAP_DEPOSIT * 16} which rails at {rail}. "
        f"{wrapped} would mean the int32 product wrapped.")
    assert got != wrapped
    assert hits == 1, "the rail must actually be the thing that bound"


def test_the_same_deposit_at_a_non_negative_exponent_is_unchanged():
    """CONTROL for the test above: the identical scene at `heat_inv_shift = +3`
    lands the exact `deposit >> 3`, below the rail and untouched by it.

    Without this control the overflow test could be passing because the deposit
    path rails on everything, which would make it vacuous.

    BREAKS IF: the widening changes the non-negative branch — i.e. if M1 stopped
    being behaviour-neutral on every shipped row.
    """
    got, hits = _run_deposit(_WRAP_DEPOSIT, 3)
    assert got == _WRAP_DEPOSIT >> 3
    assert got < _q(_DIALS["t_max_phys"])
    assert hits == 0


@pytest.mark.parametrize("his", [-1, -2, -3, -4, -8])
def test_a_sub_rail_negative_exponent_deposit_is_the_exact_product(his):
    """PROPERTY: below the rail, a negative-exponent deposit lands EXACTLY
    `deposit * 2**-his` — no rounding, no clamp, no saturation.

    The rail case above proves the overflow is handled; this proves the ordinary
    case is exact, which is the behaviour every M2 row will actually run in.

    BREAKS IF: the negative branch acquires a rounding step, or the gain is
    narrowed to int32 anywhere before the saturating add.
    """
    rail = _q(_DIALS["t_max_phys"])
    deposit = rail // (2 ** (-his)) // 4          # comfortably under the rail
    got, hits = _run_deposit(deposit, his)
    assert got == deposit * 2 ** (-his)
    assert hits == 0


def test_the_radiation_fold_takes_a_negative_exponent_too():
    """PROPERTY: Pass 1's SIGNED radiation fold (`rad_net`) converts through the
    same signed exponent, so a thin panel's radiative gain AND loss scale up
    with its small capacity.

    The fold was already int64 (`shr_round0_i64` + `sat_add_q16_i64`), so this is
    not an overflow site — it is a UB site: `>>` by a negative amount. The test
    exists because "already wide" is not "already signed".

    BREAKS IF: the fold goes back to `shr_round0_i64` (undefined behaviour at a
    negative exponent — in practice on x86, a shift by `s & 63`, i.e. a shift by
    60 for s = -4, which silently yields 0 and a fire that cannot cool).
    """
    s = _solver()
    temp, heat, shift, fs, solid, vac, atm, ts, c = _deposit_scene(0, -4)
    rad = np.zeros(temp.shape, dtype=np.int64)
    rad[c] = -1_000_000                      # a radiative LOSS
    temp[c] = 40_000_000
    s.step(temp, heat, shift, fs, solid, vac, atm, thermal_solid=ts,
           rad_net=np.ascontiguousarray(rad), clamp_enabled=False)
    assert int(temp[c]) == 40_000_000 + (-1_000_000 * 16)


# ===========================================================================
# 5. Neutrality of the whole change, stated where a reader will look for it
# ===========================================================================
# DELETED BY M2, DELIBERATELY AND ON SCHEDULE:
# `test_the_shipped_material_table_is_unmoved_by_M1`.
#
# It asserted that every shipped row still carried a NON-NEGATIVE exponent,
# which was M1's neutrality claim -- the sentence its "no golden moved" evidence
# rested on. M1's own report names this test as the one M2 must delete "in the
# same commit that lands the rows", because the rows are the whole point of M2
# and the claim is therefore SPENT, not broken. Recorded here rather than simply
# removed, so a reader who goes looking for M1's neutrality evidence finds out
# what happened to it instead of finding nothing.
#
# The property that REPLACES it, and which grows correctly, is
# `tests/test_thermal_mass_axis.py::
#  test_a_flammable_row_is_thin_and_a_structural_row_is_solid`.
#
# Everything else in this file still gates M1 and is untouched: the signed
# shift's two branches, the capacity round trip, the int32 overflow site, the
# gas sentinel, and the four extra consumer sites in the sibling file. Those are
# now exercised by SHIPPED rows for the first time, which is a strengthening.
