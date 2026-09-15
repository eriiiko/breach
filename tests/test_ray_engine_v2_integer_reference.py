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
4000-bucket table). The whole file is about two seconds.

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
    """The reference's four dials still equal config.toml's.

    Breaks if: rad_scale, kelvin_ambient, k_temp_to_kelvin or T_MAX_PHYS is
    retuned without re-running P0 -- which silently invalidates every number in
    report_p0.md and every gate below.
    """
    ok, detail = R.config_dials_match()
    assert ok, f"config.toml has moved under the reference: {detail}"


def test_conservation_identity_is_exact():
    """G1: sum(rad_net) + sum(rad_flux) + sum(rad_amb) == 0 exactly, in int64.

    Breaks if: any writer books one side of a transfer and not the other -- the
    remainder split, the body's ambient re-emission, or either of the virtual
    ring's two books.
    """
    print(_run(G.gate1_conservation))


def test_uniform_ambient_is_a_per_cell_fixed_point():
    """G2a: a uniform ambient field leaves all three planes exactly zero, per cell,
    with bodies present.

    Breaks if: the body stops re-emitting at ambient (Erik's ruling, design row 25)
    or the Fleck factor goes back to multiplying the whole emission (row 22) --
    both alternatives are measured in the gate and both are non-zero.
    """
    print(_run(G.gate2a_uniform_ambient))


def test_enclosed_isothermal_box_inner_layer_is_zero():
    """G2b: the inner layer of a sealed isothermal box is an exact per-cell zero
    with f < 1 forced.

    Breaks if: emission stops being the same arithmetic as absorption, or the
    downwind split stops carrying its remainder.
    """
    print(_run(G.gate2b_isothermal_box))


def test_stream_is_positive_and_illegal_extinction_is_rejected():
    """G3: no negative stream anywhere; a > d, d > ONE and k > ONE raise.

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
    """G8: f_q = floordiv((T_abs<<16), max(T_abs+2L, 4L)) is within one count of
    1/(1 + alpha*g) over the whole table, and f_q == ONE iff L_q == 0.

    Breaks if: alpha's max() is re-expanded wrongly, the denominator moves off the
    ABSOLUTE temperature, or the division stops being the kit's exact floordiv.
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
    """G10: the Fleck factor beats explicit against the analytic cooling curve, is
    monotone and positive from every start up to the table top, and the clamp
    reproduces the exact equilibrium where Fleck alone runs to millions.

    Breaks if: the excess form is replaced, alpha's branch changes, or the clamp
    is dropped from the fold.
    """
    print(_run(G.gate10_stability))


def test_headroom_stays_inside_int64():
    """G11: per-cell sums below 2^46 and no product above 2^62 on a scene seeded at
    the table top.

    Breaks if: the ordinate count, the ordinate weight or the table top grows
    enough to need more than int64 -- the one arithmetic assumption the whole
    scheme rests on.
    """
    print(_run(G.gate11_headroom))


if __name__ == "__main__":       # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
