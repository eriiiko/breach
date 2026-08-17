"""P-E4 -- SS2.4 trust gate + SS2.7 reversible compression work
(energy-books arc). Design: ``docs/energy_transport_design_2026-08-16.md``
v2.2 SS2.4/SS2.7. As-built: ``docs/e1_p_e4_asbuilt_2026-08-17.md``.

This module drives ``eos_kick_compression_ref`` (the CPU P6.4 reference --
the SAME step-4c loop the live ``EOSSolver::step`` runs) with a single probe
cell and an engineered divergence, rather than a re-derived Python mirror --
the design's explicit instruction ("a unit test driving one cell with
alternating +/-div at fixed |k|"). SS2.7 is invisible to every other battery
in the suite (no test pins the expansion RATE numerically), so this file is
the only oracle that can catch the highest-risk line in the patch: a plain
`/` in the expansion branch would truncate toward zero and MINT on a
sub-ambient cell, and CPU<->CUDA parity cannot see it (both backends agree on
the same wrong answer).

THE PROBE: a 5x5 grid, one cell at the centre, p_new uniform (kick's
pressure gradient is exactly zero everywhere -> step 4/drag is a no-op on
wind, so step 4c reads back EXACTLY the wind field this file sets), ambient
gas density (n_bulk ~1.0, far above n_work_ref's default trust band 0.25 so
the SS2.4 fade is ~1.0 and does not confound the SS2.7 measurement -- the
fade itself is exercised separately by test_p_e4_trust_gate.py / the
hot-rail scenario), absorb=0, k_drag=0 (dormant), is_vacuum/solid clear.
Divergence is engineered via the probe cell's east/west wind neighbours
only (duy is left exactly 0).

RIGOROUS BOUND (derived, not merely quoted from the design): AT the clamp
(design SS2.7's "at the clamp" case) the pre-clamp |k| is driven far past
+/-T_WORK_CLAMP in BOTH directions, so w == work_clamp_q EXACTLY in both
legs of a cycle regardless of any upstream rounding noise in the divergence
calculation -- this is what makes the tight algebraic identity checkable
bit-for-bit rather than merely "measured": writing T1_int = ceil(real T1)
(compression rounds its increment UP) and T2_int = floor(real T2) (the
floor-toward-inf expansion), compress-then-expand recovers T0 EXACTLY and
expand-then-compress loses AT MOST 1 raw count, one-way (never creates).
Below the clamp the two legs' |k| are only equal up to the divergence
pipeline's own truncation noise (a few raw counts, unrelated to SS2.7), so
that config is reported as a measurement with a loose bound rather than an
exact pin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

FP_ONE = 65536
GRID = 5
CY = CX = GRID // 2

CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    n_floor_solver=1e-3, t_min=-289.0, t_work_clamp=0.5,
    t_max_phys=16000.0, u_max=1000.0,
    k_drag=0.0, k_drag_heat_frac=1.0, c_v=1.0,
    n_work_ref=0.25,   # the shipped default -- ambient N below keeps fade ~1
)


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _base_state(T0_real):
    """A 5x5 ambient-air grid, uniform pressure, probe cell temperature T0."""
    h = w = GRID
    wind_x = np.zeros((h, w), dtype=np.int32)
    wind_y = np.zeros((h, w), dtype=np.int32)
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[CY, CX] = int(_q(T0_real))
    p_new = np.full((h, w), int(_q(1.0)), dtype=np.int32)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0, :, :] = int(_q(0.21))
    gas[1, :, :] = int(_q(0.79))
    gas_conservative = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    return dict(wind_x=wind_x, wind_y=wind_y, temperature=temperature,
                p_new=p_new, gas=gas, gas_conservative=gas_conservative,
                solid=solid, is_vacuum=is_vacuum, absorb=absorb)


def _apply(T0_real, mode, v_mps, dt=1.0 / 24.0, c_local=300.0):
    """Run ONE step-4/4c cycle on the probe cell; mode is 'compress' (div<0,
    k<0) or 'expand' (div>0, k>=0). Returns (T_after_real, work_clamp_hits,
    energy_floor_hits, t_max_phys_hits)."""
    st = _base_state(T0_real)
    vv = int(_q(v_mps))
    if mode == "compress":
        # converging flow at the probe cell: u(west) > 0 (into the cell from
        # the left), u(east) < 0 (into the cell from the right) -> div < 0.
        st["wind_x"][CY, CX - 1] = vv
        st["wind_x"][CY, CX + 1] = -vv
    elif mode == "expand":
        st["wind_x"][CY, CX - 1] = -vv
        st["wind_x"][CY, CX + 1] = vv
    else:
        raise ValueError(mode)
    res = bp.eos_kick_compression_ref(
        st["wind_x"], st["wind_y"], st["temperature"], st["p_new"],
        st["gas"], st["gas_conservative"], st["solid"], st["is_vacuum"],
        st["absorb"], dt, int(_q(c_local)), thermal_solid=None, **CONSTS)
    names = ("dig_vel", "dig_comp", "u_clamp", "u_max", "work_clamp",
             "energy_floor", "t_max_phys", "ke_drag_removed", "e_drag_deposit",
             "e_drag_drop_sum", "e_drag_rail_clipped")
    r = dict(zip(names, res))
    T_after_raw = int(st["temperature"][CY, CX])
    return T_after_raw, r["work_clamp"], r["energy_floor"], r["t_max_phys"]


# ---------------------------------------------------------------------------
# 1. AT THE CLAMP -- the rigorous, bit-exact bound (derived in the module
#    docstring): w is IDENTICAL in both legs of the cycle because both
#    directions saturate the SAME work_clamp_q, independent of any upstream
#    divergence-pipeline rounding noise.
# ---------------------------------------------------------------------------
def _cycle_at_clamp(T0_raw, order):
    """order: 'compress_then_expand' or 'expand_then_compress'. V is huge
    (300 m/s -> pre-clamp |k| ~15, far past T_WORK_CLAMP=0.5) so both legs
    saturate to EXACTLY work_clamp_q in both directions."""
    T0_real = T0_raw / FP_ONE
    modes = (("compress", "expand") if order == "compress_then_expand"
             else ("expand", "compress"))
    T1_raw, wc1, ef1, tm1 = _apply(T0_real, modes[0], 300.0)
    T2_raw, wc2, ef2, tm2 = _apply(T1_raw / FP_ONE, modes[1], 300.0)
    assert wc1 >= 1 and wc2 >= 1, (
        f"{order}: the clamp did not engage on both legs (wc1={wc1} wc2={wc2}) "
        "-- the config is not actually AT the clamp")
    assert ef1 == 0 and ef2 == 0 and tm1 == 0 and tm2 == 0, (
        f"{order}: an unrelated rail fired (T_MIN/T_MAX_PHYS) -- the probe "
        "temperature is not isolated from the other guards")
    return T2_raw - T0_raw


# NOTE: the sub-ambient probe temperature is -100, not the more dramatic
# -200/-289 used elsewhere in the arc's tests -- AT the clamp, compression
# multiplies by (1+work_clamp_q) = 1.5, and -200*1.5 = -300 would cross the
# T_MIN=-289 floor and engage an UNRELATED rail (the design's own guard,
# correctly firing) rather than exercising the SS2.7 identity cleanly. -100
# stays comfortably inside the floor (-100*1.5=-150) while still being
# genuinely sub-ambient.
def test_at_clamp_compress_then_expand_is_exact_both_signs_of_T():
    for tag, T0_real in (("positive", 300.0), ("sub_ambient", -100.0)):
        T0_raw = int(_q(T0_real))
        residual = _cycle_at_clamp(T0_raw, "compress_then_expand")
        assert residual == 0, (
            f"AT-CLAMP compress-then-expand ({tag} T): residual {residual} "
            "raw counts -- design SS2.7 predicts an EXACT identity in this "
            "order (compression rounds its increment UP, expansion floors, "
            "and the two cancel exactly when w matches bit-for-bit)")


def test_at_clamp_expand_then_compress_is_one_way_within_1_lsb_both_signs_of_T():
    for tag, T0_real in (("positive", 300.0), ("sub_ambient", -100.0)):
        T0_raw = int(_q(T0_real))
        residual = _cycle_at_clamp(T0_raw, "expand_then_compress")
        assert residual in (0, -1), (
            f"AT-CLAMP expand-then-compress ({tag} T): residual {residual} "
            "raw counts -- design SS2.7 predicts a ONE-WAY loss of AT MOST "
            "1 raw count (never a gain -- a positive residual here would be "
            "a mint)")


# ---------------------------------------------------------------------------
# 2. BELOW THE CLAMP -- measured (not a tight bit-exact pin). Unlike the
#    at-clamp case, the two legs' |k| are only equal up to the DIVERGENCE
#    PIPELINE's own truncation noise (mul_q16 rounds a negative product
#    toward -inf, not symmetrically with its negation) -- a mismatch of a
#    few raw q16 COUNTS IN w, unrelated to SS2.7's floor/mul_q16 story, that
#    gets amplified by T0 itself (residual ~ T0 * delta_w). Measured: at
#    T0=300 and v=2 m/s (k~0.1), a ~1-ULP w mismatch shows up as ~270 raw
#    counts of T (~0.14% of T0) -- reported per the design's own
#    "measurement, not an assertion" framing for the general case, with a
#    RELATIVE regression bound loose enough to absorb that pipeline noise
#    but tight enough to catch a genuinely broken mechanism (which would
#    show an O(w) ~ 10%+ error, not a sub-1% one).
# ---------------------------------------------------------------------------
def _cycle_below_clamp(T0_raw, order, v_mps=2.0):
    T0_real = T0_raw / FP_ONE
    modes = (("compress", "expand") if order == "compress_then_expand"
             else ("expand", "compress"))
    T1_raw, wc1, ef1, tm1 = _apply(T0_real, modes[0], v_mps)
    T2_raw, wc2, ef2, tm2 = _apply(T1_raw / FP_ONE, modes[1], v_mps)
    assert wc1 == 0 and wc2 == 0, (
        f"{order}: the clamp engaged (wc1={wc1} wc2={wc2}) -- v_mps={v_mps} "
        "is not actually below the clamp")
    return T2_raw - T0_raw


def test_below_clamp_residual_is_small_both_orders_both_signs_of_T(capsys):
    rows = []
    for order in ("compress_then_expand", "expand_then_compress"):
        for tag, T0_real in (("positive", 300.0), ("sub_ambient", -200.0)):
            T0_raw = int(_q(T0_real))
            residual = _cycle_below_clamp(T0_raw, order)
            rows.append((order, tag, residual))
            bound = max(8, abs(T0_raw) // 200)   # ~0.5% of |T0_raw|, floor 8
            assert abs(residual) <= bound, (
                f"BELOW-CLAMP {order} ({tag} T): residual {residual} raw "
                f"counts exceeds the {bound}-count noise-floor bound "
                "(~0.5% of T0) -- unexpectedly large for a sub-clamp cycle")
    with capsys.disabled():
        print("\n  P-E4 SS2.7 unit oracle -- BELOW-CLAMP measured residuals "
              "(raw counts, v=2.0 m/s):")
        for order, tag, residual in rows:
            print(f"    {order:22s} T={tag:11s} residual={residual:+d}")


# ---------------------------------------------------------------------------
# 3. The asymmetric-cycle figure (design SS2.7's worked example): one
#    compression at w~0.4 against TWO expansions at w~0.2 each. Measures the
#    NEW law's per-cycle proportional loss against the doc's analytic
#    reference figures (~2.8%/cycle new vs ~10.4%/cycle under the retired
#    T*(1-k) law -- computed here in real (non-quantized) arithmetic, since
#    the old law no longer exists as compiled code to drive directly).
# ---------------------------------------------------------------------------
def test_asymmetric_cycle_matches_the_designs_worked_example(capsys):
    T0_real = 300.0
    T0_raw = int(_q(T0_real))
    # v_mps=8.0 -> k_real = 0.05*v = 0.4 (compression); v_mps=4.0 -> k=0.2
    # (expansion), from k_real = (gamma-1)*div*dt = 0.4*(3v)*(1/24) = 0.05v
    # at this probe's dx=1/3, dt=1/24, gamma=1.4.
    T1_raw, wc1, ef1, tm1 = _apply(T0_real, "compress", 8.0)
    assert wc1 == 0 and ef1 == 0 and tm1 == 0
    T2_raw, wc2, ef2, tm2 = _apply(T1_raw / FP_ONE, "expand", 4.0)
    assert wc2 == 0 and ef2 == 0 and tm2 == 0
    T3_raw, wc3, ef3, tm3 = _apply(T2_raw / FP_ONE, "expand", 4.0)
    assert wc3 == 0 and ef3 == 0 and tm3 == 0

    measured_frac_loss = 1.0 - (T3_raw / FP_ONE) / T0_real

    analytic_new = 1.0 - (1.0 + 0.4) / ((1.0 + 0.2) * (1.0 + 0.2))
    analytic_old = 1.0 - (1.0 + 0.4) * (1.0 - 0.2) * (1.0 - 0.2)

    with capsys.disabled():
        print(f"\n  P-E4 SS2.7 asymmetric-cycle figure (1 compression w~0.4, "
              f"2 expansions w~0.2 each):")
        print(f"    measured NEW-law loss:   {measured_frac_loss * 100:.3f}%")
        print(f"    analytic NEW-law loss:   {analytic_new * 100:.3f}% "
              f"(design's worked example: ~2.8%)")
        print(f"    analytic OLD-law loss:   {analytic_old * 100:.3f}% "
              f"(design's worked example: ~10.4%)")

    # The measured figure must land close to the analytic NEW-law
    # prediction (small gap = Q16.16 quantization of the achieved w's away
    # from the nominal 0.4/0.2) and nowhere near the OLD law's figure.
    assert abs(measured_frac_loss - analytic_new) < 0.01, (
        f"measured NEW-law loss {measured_frac_loss:.4f} strayed from the "
        f"analytic prediction {analytic_new:.4f} by more than 1pp")
    assert measured_frac_loss < analytic_old - 0.03, (
        "the measured NEW-law loss is not clearly better than the retired "
        "OLD law's figure")
