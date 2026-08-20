"""P-E4 -- SS2.4 trust gate + SS2.7 reversible compression work, REWRITTEN for
the T_abs compression-work law (P-W1b). Design:
``docs/tabs_compression_work_design_2026-08-20.md`` SS2 (the pinned
arithmetic) + critique ``docs/tabs_compression_work_critiques_2026-08-20.md``
A6 (the reversibility proof) / A7 (the below-clamp bound) / A8+B-F10 (the
dial-derived clamp oracle). Energy-books lineage:
``docs/energy_transport_design_2026-08-16.md`` v2.2 SS2.4/SS2.7,
``docs/e1_p_e4_asbuilt_2026-08-17.md``.

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
only (duy is left exactly 0). T_AMB_K=290 (CONSTS's ``t_amb_k``, threaded
explicitly per design SS5's Python-caller list rather than left on the
binding's default -- dial-derived, not a hardcoded literal) makes the
stored ``temperature[]`` plane's convention (ambient-relative, 0 = ambient)
explicit at every call site below.

THE LAW (design SS2): the 4c arithmetic's interior now runs on ABSOLUTE
temperature, then shifts back to the stored ambient-relative convention.
Per cell: ``t_abs = T + t_amb_q`` (int64). Compression (k < 0, magnitude w
after the fade+clamp): ``t_new = T + floor(w * t_abs)`` via the existing
mul_q16/SAR convention -- heating rounds UP because a negative product
floors toward -inf and the code then negates it. Expansion (k >= 0,
including the pinned k==0 identity, D-4): ``t_new = floor(t_abs / (1+w)) -
t_amb_q``, i.e. the reversible inverse computed on t_abs THEN shifted back.

A6's REVERSIBILITY PROOF (the algebraic backbone this file's at-clamp tests
assert): with ``a = t_abs > 0``, compression is ``C(a) = a + ceil(w*a) =
ceil(a(1+w))`` (mul_q16 floors the negative dT => the increment itself
rounds UP) and expansion is ``E(a) = floor(a/(1+w))``. Then:

  E(C(a)) = a                              EXACTLY  (compress-then-expand)
  C(E(a)) in {a, a-1}                      one-way, at most 1 raw count
                                            (expand-then-compress)

The +/-t_amb_q shift cancels across any closed cycle (it is added going in
and subtracted coming out at every leg), so this identity holds on the
STORED ambient-relative T exactly as it holds on the internal t_abs -- this
is why the two AT-CLAMP tests below assert the SAME residual bounds
(0 / {0,-1}) the old ambient-relative law's tests asserted: the identity
was never about which quantity C/E operate on, only about ceil/floor being
exact inverses at a shared w. **A red on either AT-CLAMP test is therefore a
real transcription bug in the C++ arithmetic, not a test that needs
updating -- STOP, per critique A6/C15.**

RIGOROUS BOUND (derived, not merely quoted from the design): AT the clamp
(design SS2's "at the clamp" case) the pre-clamp |k| is driven far past
+/-T_WORK_CLAMP in BOTH directions, so w == work_clamp_q EXACTLY in both
legs of a cycle regardless of any upstream rounding noise in the divergence
calculation -- this is what makes the tight algebraic identity checkable
bit-for-bit rather than merely "measured". Below the clamp the two legs'
|k| are only equal up to the divergence pipeline's own truncation noise (a
few raw counts, unrelated to the reversibility identity), so that config is
reported as a measurement with a loose bound rather than an exact pin.
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

T_AMB_K = 290.0   # the shipped default; threaded explicitly below (not the
                   # binding's implicit default) so the oracle in section 4
                   # is genuinely dial-derived rather than a coincidence.


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


# The SAME A7-floored fold the C++ side performs (eos_solver.cpp:372 /
# kick_scalar_folds()): max(1, quantize(T_AMB_K)). At the shipped default the
# floor never binds -- present anyway so this constant is traceably the same
# expression as the arithmetic under test, not a re-derived literal.
T_AMB_Q_RAW = max(1, int(_q(T_AMB_K)))

CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    n_floor_solver=1e-3, t_min=-289.0, t_work_clamp=0.5,
    t_max_phys=16000.0, u_max=1000.0,
    k_drag=0.0, k_drag_heat_frac=1.0, c_v=1.0,
    n_work_ref=0.25,   # the shipped default -- ambient N below keeps fade ~1
    t_amb_k=T_AMB_K,   # T_ABS COMPRESSION WORK (P-W1b, design SS5): threaded
                       # explicitly -- this file is one of design SS5's listed
                       # Python callers.
)


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
    # VELOCITY-CLAMP (P-V1, D2v2): signature only — a uniform (h,w) cap²
    # plane at c_local² (D5: trusted verbatim, matches the old scalar's
    # effective cap exactly since c_local=300.0 < u_max=1000.0 here). Its
    # neighbours sit at T=0 -> cap = c_amb = 300 = |u| exactly, and
    # rad > cap2 is strict -> no clamp, same as today. Do NOT "fix" the
    # 300.0 default.
    cap2 = np.full((GRID, GRID), int(_q(c_local)) ** 2, dtype=np.int64)
    res = bp.eos_kick_compression_ref(
        st["wind_x"], st["wind_y"], st["temperature"], st["p_new"],
        st["gas"], st["gas_conservative"], st["solid"], st["is_vacuum"],
        st["absorb"], dt, cap2, thermal_solid=None, **CONSTS)
    names = ("dig_vel", "dig_comp", "u_clamp", "u_max", "work_clamp",
             "energy_floor", "t_max_phys", "ke_drag_removed", "e_drag_deposit",
             "e_drag_drop_sum", "e_drag_rail_clipped")
    r = dict(zip(names, res))
    T_after_raw = int(st["temperature"][CY, CX])
    return T_after_raw, r["work_clamp"], r["energy_floor"], r["t_max_phys"]


# ---------------------------------------------------------------------------
# 1. AT THE CLAMP -- the rigorous, bit-exact bound (derived in the module
#    docstring, A6): w is IDENTICAL in both legs of the cycle because both
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


# PROBE TEMPERATURES, RE-JUSTIFIED FOR THE NEW LAW (critique A7/C15): the old
# file picked -100 (not the more dramatic -200/-289 used elsewhere) because
# under the RETIRED ambient-relative law compression multiplied T directly,
# and -200*1.5 = -300 would cross T_MIN=-289 and engage an unrelated rail.
# That reasoning is VOID under the new law: compression now runs on t_abs =
# T + 290, and at T=-200 that is t_abs=90 -- compression WARMS toward
# ambient, it can no longer drive T further negative at all. The new
# floor-crossing risk (if any) comes from EXPANSION shrinking t_abs toward
# its own floor (t_abs=1 <=> T=T_MIN=-289), not from compression. Using
# -200 real: at the clamp (w=0.5, factor 1.5), compression gives
# (-200+290)*1.5-290 = -155 (critique A7's own worked number) -- nowhere
# near -289 -- so -200 is not merely "still safe", it is MORE representative
# of a genuinely cold probe than the old file's -100, and is used here
# (matching the below-clamp probe at section 2, which already used -200).
def test_at_clamp_compress_then_expand_is_exact_both_signs_of_T():
    for tag, T0_real in (("positive", 300.0), ("sub_ambient", -200.0)):
        T0_raw = int(_q(T0_real))
        residual = _cycle_at_clamp(T0_raw, "compress_then_expand")
        assert residual == 0, (
            f"AT-CLAMP compress-then-expand ({tag} T): residual {residual} "
            "raw counts -- design SS2's A6 proof predicts an EXACT identity "
            "in this order (compression rounds its increment UP, expansion "
            "floors, and the two cancel exactly when w matches bit-for-bit) "
            "-- a red here is a real transcription bug in the C++, not a "
            "test to adjust (STOP, per critique A6/C15)")


def test_at_clamp_expand_then_compress_is_one_way_within_1_lsb_both_signs_of_T():
    for tag, T0_real in (("positive", 300.0), ("sub_ambient", -200.0)):
        T0_raw = int(_q(T0_real))
        residual = _cycle_at_clamp(T0_raw, "expand_then_compress")
        assert residual in (0, -1), (
            f"AT-CLAMP expand-then-compress ({tag} T): residual {residual} "
            "raw counts -- design SS2's A6 proof predicts a ONE-WAY loss of "
            "AT MOST 1 raw count (never a gain -- a positive residual here "
            "would be a mint) -- a red here is a real transcription bug in "
            "the C++, not a test to adjust (STOP, per critique A6/C15)")


# ---------------------------------------------------------------------------
# 2. BELOW THE CLAMP -- measured (not a tight bit-exact pin). Unlike the
#    at-clamp case, the two legs' |k| are only equal up to the DIVERGENCE
#    PIPELINE's own truncation noise (mul_q16 rounds a negative product
#    toward -inf, not symmetrically with its negation) -- a mismatch of a
#    few raw q16 COUNTS IN w, unrelated to the reversibility identity, that
#    gets amplified by the quantity the compression/expansion arithmetic
#    actually operates on. Critique A7: under the NEW law that quantity is
#    t_abs = T + t_amb_q, not T -- so the noise-floor bound must re-key on
#    |T0_raw + t_amb_q|, not |T0_raw| (the old file's key, silently correct
#    only because the old law's compression term was T*(1+w), i.e. T itself
#    was the operand). A RELATIVE regression bound loose enough to absorb
#    that pipeline noise but tight enough to catch a genuinely broken
#    mechanism (which would show an O(w) ~ 10%+ error, not a sub-1% one).
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
            # re-keyed on |T0_raw + t_amb_q| (critique A7) -- the quantity
            # the arithmetic actually operates on under the new law.
            t_abs_raw = abs(T0_raw + T_AMB_Q_RAW)
            bound = max(8, t_abs_raw // 200)   # ~0.5% of |t_abs|, floor 8
            assert abs(residual) <= bound, (
                f"BELOW-CLAMP {order} ({tag} T): residual {residual} raw "
                f"counts exceeds the {bound}-count noise-floor bound "
                "(~0.5% of |T0_raw + t_amb_q|) -- unexpectedly large for a "
                "sub-clamp cycle")
    with capsys.disabled():
        print("\n  P-E4 T_abs-law unit oracle -- BELOW-CLAMP measured "
              "residuals (raw counts, v=2.0 m/s):")
        for order, tag, residual in rows:
            print(f"    {order:22s} T={tag:11s} residual={residual:+d}")


# ---------------------------------------------------------------------------
# 3. The asymmetric-cycle figure (design SS2's worked example, re-derived on
#    the (T+t_amb) base per P-W1b's mandate): one compression at w~0.4
#    against TWO expansions at w~0.2 each. Under the OLD ambient-relative
#    law the per-cycle proportional loss of T itself was the clean
#    1-(1+wc)(1-we)^2 form (~10.4%) -- that arithmetic is UNCHANGED as a
#    fact about the retired law and is kept below purely as the "nowhere
#    near this" comparison point. Under the NEW law the multiplicative
#    factors apply to t_abs, not T, so T's own fractional loss picks up an
#    extra additive term from the +/-t_amb shift:
#
#      t_abs0 = T0 + t_amb;  t_abs3 = t_abs0*(1+wc) / (1+we)**2
#      T3 = t_abs3 - t_amb
#      loss = 1 - T3/T0
#           = 1 - [ (T0+t_amb)*(1+wc)/(1+we)**2 - t_amb ] / T0
#
#    At T0=300, t_amb=290, wc=0.4, we=0.2 this evaluates to ~5.463% (roughly
#    DOUBLE the old law's ~2.8%-if-computed-on-T-alone figure, and just over
#    HALF the retired law's own ~10.4% T-loss figure) -- computed here in
#    real (non-quantized) arithmetic since T_AMB_K/T_WORK_CLAMP etc. are the
#    dials under test, not hardcoded.
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

    wc, we = 0.4, 0.2
    t_amb = T_AMB_K
    analytic_new = 1.0 - (((T0_real + t_amb) * (1.0 + wc) / (1.0 + we) ** 2)
                          - t_amb) / T0_real
    analytic_old_on_T_alone = 1.0 - (1.0 + wc) * (1.0 - we) ** 2

    with capsys.disabled():
        print(f"\n  P-E4 T_abs-law asymmetric-cycle figure (1 compression "
              f"w~0.4, 2 expansions w~0.2 each):")
        print(f"    measured NEW-law loss:        {measured_frac_loss * 100:.4f}%")
        print(f"    analytic NEW-law loss:        {analytic_new * 100:.4f}% "
              f"(re-derived on the (T+t_amb) base, design SS2/P-W1b)")
        print(f"    RETIRED law's T-loss figure:  {analytic_old_on_T_alone * 100:.4f}% "
              f"(design's original worked example, ~10.4% -- unaffected as a "
              f"fact about the dead law, kept for comparison only)")

    # The measured figure must land close to the re-derived analytic
    # NEW-law prediction (small gap = Q16.16 quantization of the achieved
    # w's away from the nominal 0.4/0.2 -- measured empirically at ~0.004pp).
    assert abs(measured_frac_loss - analytic_new) < 0.005, (
        f"measured NEW-law loss {measured_frac_loss:.4f} strayed from the "
        f"re-derived analytic prediction {analytic_new:.4f} by more than "
        "0.5pp")
    # And it should be nowhere near the OLD law's T-loss figure (this is not
    # a coincidence-of-magnitude check -- the two laws differ structurally).
    assert abs(measured_frac_loss - analytic_old_on_T_alone) > 0.03, (
        "the measured NEW-law loss is suspiciously close to the retired "
        "law's figure -- check the arithmetic actually changed")


# ---------------------------------------------------------------------------
# 4. THE DIAL-DERIVED CLAMP ORACLE (critique A8/B-F10). Unlike sections 1-3,
#    this is not a measurement against an analytic approximation -- it is
#    the EXACT closed-form value of the expansion branch when w is pinned at
#    T_WORK_CLAMP by the rail, computed IN THIS TEST from the same dials
#    CONSTS passes to the C++ (T_AMB_K, T_WORK_CLAMP), using the SAME
#    quantize() convention (_q, round-half-away-from-zero) and the SAME
#    floor-toward-inf division (Python's `//` on two positive operands is
#    floordiv_q's convention exactly). A probe starting at ambient
#    (T_rel=0, t_abs=t_amb_q) driven with a huge expansion divergence (so
#    w saturates to work_clamp_q exactly, the same technique section 1
#    uses) must land EXACTLY on this value -- tuning T_WORK_CLAMP later
#    changes `expected` right along with it, so this oracle cannot go stale.
# ---------------------------------------------------------------------------
def test_dial_derived_clamp_oracle_at_ambient_expansion(capsys):
    work_clamp_q = int(_q(CONSTS["t_work_clamp"]))
    # expected = floordiv(t_amb_q << 16, FP_ONE + quantize(T_WORK_CLAMP)) - t_amb_q
    expected = (T_AMB_Q_RAW << 16) // (FP_ONE + work_clamp_q) - T_AMB_Q_RAW

    T_after_raw, wc, ef, tm = _apply(0.0, "expand", 300.0)
    assert wc >= 1, (
        "expansion at v=300 must saturate the clamp -- the oracle assumes "
        "w == work_clamp_q exactly")
    assert ef == 0 and tm == 0, "an unrelated rail fired at an ambient probe"
    assert T_after_raw == expected, (
        f"dial-derived clamp oracle mismatch: T_after={T_after_raw} raw, "
        f"expected={expected} raw -- the expansion branch's t_abs/shift-back "
        "arithmetic does not match design SS2's pinned form")

    # Shipped-default sanity comment (NOT re-derived from a literal --
    # cross-check only, per critique A8): at T_AMB_K=290, T_WORK_CLAMP=0.5,
    # `expected` evaluates to -6,335,147 raw == -96.6666 game-deg -- the
    # design's headline "290 K -> 193.3 K, i.e. -96.67 game-deg" figure for
    # honest expansion at the work clamp (design SS1).
    if T_AMB_K == 290.0 and CONSTS["t_work_clamp"] == 0.5:
        assert expected == -6_335_147, (
            "shipped-default sanity check failed -- either the dials moved "
            "without updating this comment, or the oracle formula itself "
            "regressed")

    with capsys.disabled():
        print(f"\n  P-E4 B-F10 dial-derived clamp oracle: expected={expected} "
              f"raw ({expected / FP_ONE:.4f} game-deg) at T_AMB_K={T_AMB_K}, "
              f"T_WORK_CLAMP={CONSTS['t_work_clamp']}")
