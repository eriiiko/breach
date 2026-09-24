"""The ray-engine-v2 INTEGER REFERENCE's gates, as pytest (P0, arc #12).

The reference and its gate runner live where the rest of the arc's instruments
live, `docs/ray_engine_v2_scheme_study_2026-09-13/`, because they are a study, not
engine code: nothing in `src/` or `cpp/` imports them, and P1's C++ is written
against them. This file is the thin wrapper that keeps them running in CI, so the
reference cannot rot while the design it measures is still being built. The
`sys.path` insertion below is deliberate and is the only way to import from a docs
folder that is not a package; it is confined to this file.

Every test runs its gate in FAST mode (smaller grids and tick counts, the SAME
coverage: both transports, S16 and S12, leak on and off, bodies present, the whole
4000-bucket table). The whole file is a few seconds.

Each gate's own docstring in `sweep_ref_q_gates.py` names the property it protects
and the change that must break it; the one-line summary is repeated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STUDY = Path(__file__).resolve().parents[1] / "docs" / "ray_engine_v2_scheme_study_2026-09-13"
if str(_STUDY) not in sys.path:
    sys.path.insert(0, str(_STUDY))

import sweep_ref_q as R            # noqa: E402
import sweep_ref_q_gates as G      # noqa: E402


def _run(gate):
    ok, lines = gate(fast=True)
    report = "\n".join(lines)
    assert ok, f"\n{gate.__doc__}\n{report}"
    return report


def test_config_dials_have_not_drifted():
    """The reference's six dials still equal config.toml's.

    Breaks if: rad_scale, kelvin_ambient, k_temp_to_kelvin or T_MAX_PHYS is
    retuned without re-running P0 -- which silently invalidates every number in
    report_p0.md and every gate below -- or c_v / n_floor_heat move under the gas
    capacity chain (P5a), which invalidates the gas-stiffness numbers
    (p5a_gas_stiffness_study.py).
    """
    ok, detail = R.config_dials_match()
    assert ok, f"config.toml has moved under the reference: {detail}"


def test_conservation_identity_is_exact():
    """G1: sum(rad_net) + sum(rad_flux) + sum(rad_amb) == 0 exactly, in int64 --
    also with absorbing SMOKE cells (P5a), where a gas cell must book a non-zero
    rad_net somewhere.

    Breaks if: any writer books one side of a transfer and not the other -- the
    remainder split, the body's ambient re-emission, or either of the virtual
    ring's two books -- or the smoke term stops reaching the sweep's arithmetic.
    """
    print(_run(G.gate1_conservation))


def test_uniform_ambient_is_a_per_cell_fixed_point():
    """G2a: a uniform ambient field leaves all three planes exactly zero, per cell,
    with bodies present -- and with absorbing SMOKE at ambient, bodies standing in
    it (P5a).

    Breaks if: the body stops re-emitting at ambient (Erik's ruling, design row 25)
    or the Fleck factor goes back to multiplying the whole emission (row 22) --
    both alternatives are measured in the gate and both are non-zero -- or a gas
    cell's emission stops being the arithmetic its absorption is.
    """
    print(_run(G.gate2a_uniform_ambient))


def test_enclosed_isothermal_box_inner_layer_is_zero():
    """G2b: the inner layer of a sealed isothermal box is an exact per-cell zero
    with f < 1 forced -- and the same box FILLED WITH SMOKE at the walls'
    temperature is an exact zero at every inner cell, smoke included (P5a).

    Breaks if: emission stops being the same arithmetic as absorption (on a solid
    OR a gas cell), or the downwind split stops carrying its remainder.
    """
    print(_run(G.gate2b_isothermal_box))


def test_stream_is_positive_and_illegal_extinction_is_rejected():
    """G3: no negative stream anywhere, smoke included; a > d, d > ONE and k > ONE
    raise, and so do the gas extinction's own illegal inputs (a heat_absorb
    outside [0, 4096] in Q16, a table/plane mismatch, more planes than the
    headroom covers, gas without the thermal-solid mask).

    Breaks if: absorption stops being bounded by a + b <= ONE, or an illegal
    extinction plane becomes measurable instead of rejected at the door.
    """
    print(_run(G.gate3_positivity))


def test_counters_are_silent_where_the_design_says_they_are():
    """G4: the low rail and the clamp are silent on a marine beside an ambient
    wall, and the clamp fires on an over-driven scene.

    Breaks if: the body becomes a pure radiative sink again -- the gate measures
    that variant and it trips the low rail on every wall cell, every tick.
    """
    print(_run(G.gate4_counters))


def test_maximum_principle_holds_on_the_radiative_substep():
    """G5: T_new <= max(T_before, E_inv(Phi)) for every cell, evaluated on the
    radiative sub-step alone.

    Breaks if: the clamp is removed (the gate measures the un-clamped climb), or
    it is written in the bare v2 form, which would cool a burning crate toward
    ambient every tick (also measured).
    """
    print(_run(G.gate5_maximum_principle))


def test_shear_is_rounder_than_step_and_matches_the_float_footprint():
    """G6: the 64-direction ignition footprint -- shear beats step for a 1-tile and
    a 5x5 fire, and the integer gather reproduces the float push footprint.

    Breaks if: the transport constants s_m change, or the half-offset ordinate set
    is replaced by one with an ordinate on an axis.
    """
    print(_run(G.gate6_isotropy))


def test_integer_gather_agrees_with_the_float_push_reference():
    """G7: max|int - float| / max|rad_net| stays below 1e-4 (it is ~1e-6, the shift
    truncation), against `sweep_ref.py` itself where the two are identical.

    Breaks if: the gather form stops reproducing the push form's faces, or the
    virtual ring stops reproducing sweep_ref.py's boundary seeding.
    """
    print(_run(G.gate7_float_agreement))


def test_fleck_integer_form_matches_the_float_within_one_count():
    """G8: f_q24 = floordiv((T_abs<<24), max(T_abs, 4L)) is within ONE Q24 COUNT of
    1/(1 + alpha*g), alpha = max(0, 1 - 1/g), over the whole table, and
    f_q24 == 2^24 EXACTLY whenever 4L <= T_abs (alpha floor 0, design row 39).

    Breaks if: alpha's max() is re-expanded wrongly -- notably a return to the
    superseded floor of one half, which damps at EVERY L > 0 and is measured on
    the same probe; the denominator moves off the ABSOLUTE temperature; the
    division stops being the kit's exact floordiv; or f goes back to Q16 -- one
    Q16 count is 256x this bound, measured beside it.
    """
    print(_run(G.gate8_fleck_form))


def test_e_inv_is_idempotent_and_saturates_below_the_rail():
    """G9: E°[E_inv(Phi)] <= Phi, E_inv below E°[0] is 0, and saturation is 15996
    game -- below T_MAX_PHYS, so the clamp binds first on the radiative sub-step.

    Breaks if: the search returns a bucket's high edge, the fixed 12 trips stop
    covering 4000 buckets, or the sub-E°[0] case is left undefined again.
    """
    print(_run(G.gate9_e_inv))


def test_stability_and_equilibrium_on_the_new_forms():
    """G10: at alpha floor 0 (design row 39) the damping is OFF below g = 1 -- the
    1263-game cooling march equals the explicit one count for count, which costs
    forward Euler's -5.7 % and is measured against the superseded floor's +0.23 %
    on the same march -- while above g = 1 the damping still stops the explicit
    rail-to-zero, the march stays monotone and positive from every start up to the
    table top, and the clamp reproduces the exact equilibrium where Fleck alone
    runs to millions.

    Breaks if: the excess form is replaced; alpha's floor moves back off 0 (the
    fire-range march stops matching explicit and the un-clamped 1263-game
    equilibrium jumps from +0.16 % to +4.6 %); or the clamp is dropped from the
    fold.
    """
    print(_run(G.gate10_stability))


def test_headroom_stays_inside_int64():
    """G11: per-cell sums below 2^46 and no product above 2^63 on a scene seeded at
    the table top -- including the Q24 Fleck product `ex_m * f_q24`, the widest one,
    at S16 AND S12.

    Breaks if: the ordinate count, the ordinate weight, the table top or the Fleck
    factor's fixed point grows enough to need more than int64 -- the one arithmetic
    assumption the whole scheme rests on. Since P5a also: the gas density sum at
    the door's maximum heat_absorb over 16 planes of INT32_MAX density fits int64
    and twice that bound would not (so the door's 4096 is the arithmetic's own
    limit), and a room of opaque smoke at the table top keeps the per-cell bounds.
    Since P5b: the gas arm's L at its widest (a_gas = ONE, the table top, a 0-K
    sky, N at the floor) keeps 4L inside int64 on both tables in the fold's
    currency, and on the live table for ANY positive currency.
    """
    print(_run(G.gate11_headroom))


def test_damped_source_is_monotone_in_temperature():
    """G12: what a cell EMITS -- E°[0] + f*(E°[T] - E°[0]) -- never falls as T
    rises, over the whole 4000-bucket table, for every absorbing material row
    config.toml ships. A hotter body must not radiate less.

    Re-checked at alpha floor 0 (P2a, design row 39): unchanged, 0 backward steps
    on every shipped row.

    Breaks if: f returns to Q16 (the gate measures that form on the same probe and
    it is NOT monotone -- 2033 backward steps, worst 2.47 %); the E° bake or
    rad_scale is retuned until a bucket's rise no longer clears one count of f; or
    a NEW material row ships `heat_atten > 0` with a thermal mass small enough to
    quantize its own emission -- which is P0 section 0.4's proposed ingress rule
    and is a failure this gate should raise, not hide.
    """
    print(_run(G.gate12_damped_source_is_monotone))


def test_gas_extinction_rides_the_density_law():
    """G13 (P5a, design 6.3): smoke's heat extinction. a_gas counts absorbers
    (k x density = k x a_gas to truncation, up to the ONE cap; a negative density
    absorbs nothing); what thin smoke does not absorb continues down the stream
    (an absorber behind a smoke column receives strictly less as it thickens, and
    cold opaque smoke is a perfect shield); a cell below N_EPS_RAW bulk is
    invisible even when hot; a thermal solid ignores the gas in its pores; the
    stamped total on a gas cell is a MAX; and the gas L_q chain (the gas arm's,
    wired at P5b: G14) is the solid chain at unit capacity and follows
    L = a*ex/(max(N, n_floor)*c_v).

    Breaks if: a second density factor is applied on top of the extinction (the
    v2.4 min(N, N_AMB)/N_AMB -- design 6.3's double debit), the floor moves off
    N_EPS_RAW or off the BULK count, thermal solids take the smoke term, the body
    share is summed instead of MAXed, or the staged chain drops or reorders a
    reciprocal. Each of these was injected once and turned this gate red (P5b
    added one: the gas arm reading the raw bulk count without its n_floor floor).
    """
    print(_run(G.gate13_gas_extinction))


def test_gas_fleck_arm_keeps_radiative_cooling_monotone():
    """G14 (P5b, design 2.8 / 6.3): THE GAS ARM OF THE FLECK PRE-PASS. On the LIVE
    table in the fold's gas currency, a hot absorbing gas cell radiating into an
    ambient room cools MONOTONICALLY and NEVER BELOW AMBIENT from every start up
    to the table top -- pure soot and a typical mix, at ambient density,
    isobaric, and at the n_floor_heat density (the stiffest any gas cell can
    be) -- while the L = 0 arm P5a shipped overshoots below ambient in ONE step
    at EXACTLY the starts where g > 4T/T_abs. The pre-pass calls the arm on every
    absorbing gas cell and nowhere else, refuses a non-positive c_v or n_floor,
    and what the cell emits steps backward in T by less than gate 12's
    pathological bound.

    Breaks if: the arm is disconnected (L = 0: every case undershoots), priced in
    a currency other than the fold's (c_v = 1, the pre-P5a premise: 130x
    under-damped, every case undershoots), or reads the material `a` instead of
    a_gas in the pre-pass (the pre-pass check goes red). Each was injected once
    into the reference and turned this gate red.
    """
    print(_run(G.gate14_gas_fleck_arm))


def test_gas_fold_clamps_in_energy_form_and_closes_the_boundary():
    """G15 (P5c, design 2.8 / 6.3 / 8.4): THE TEMPERATURE FOLD'S GAS BRANCH, the
    arithmetic temperature_solver.cpp's gas branch and its CUDA twin transcribe.
    The clamp in its energy form lands every clamped gas cell's mirror EXACTLY on
    max(T_before, E_inv(Phi)) and keeps the cell's sub-LSB residual E mod N (the
    design's letter N*(T_target + t_amb) - E hits the same mirror but drains it
    -- measured beside it); sum(Eg) moves by e_gas_deposit_sum + e_gas_rail_sum
    to the count (group 1, no new group); e_rad_clamp_drop_sum is the withheld
    step priced at cap_real on gas AND on thermal solids; the maximum principle
    holds on gas (and fails with the clamp off); a sealed room of hot smoke cools
    through the whole tick, never rising, never below ambient, walls warming; a
    smoke layer shields a target on every tick against clear air, also once it
    has heated and re-radiates; and design 8.4's sweep->fold boundary stays
    inside the conversions' own truncation plus the counted drop.

    Breaks if: the gas branch uses the design's letter (the residual drains:
    (a) red), a scaled dE (the target is missed), a second currency (c_v = 1:
    the landing is off), or is disconnected (the smoke never cools and the
    boundary leaks 276x its bound). Each was injected once into the reference and
    turned this gate red.
    """
    print(_run(G.gate15_gas_fold))


if __name__ == "__main__":       # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
