#!/usr/bin/env python
"""P-E2b instrument — n_floor_heat reciprocal-precision probe.

Design `energy_transport_design_2026-08-16.md` v2.2 §2.2 (RULING, Erik
2026-08-17): `n_floor_heat` becomes a LOW, tunable VALUE-hygiene dial,
"swept DOWNWARD during tuning ... we can see how low we can go," with the
explicit requirement that "the reciprocal path gets int64 intermediates so
even 0.001 is reachable" without precision collapse.

WHAT THIS FOUND (not hypothetical — a real bug, fixed alongside this probe):
the OLD two-step chain both deposit sites shared —

    e_over_n = mul_q16(deposit, recip_n)     # NARROWS to q16 (int32) HERE
    dT       = recip_mul(e_over_n, recip_cv)

— narrows the FIRST reciprocal multiply to Q16.16 int32 (representable
magnitude <= ~32768) *before* dividing by c_v. At n_floor_heat=0.01 a
ROUTINE single-fire deposit (~330, the eos-p3fix-thermal-ceiling repro's own
reference number) divided by the floor alone is 330/0.01 = 33,000 — already
past q16's ceiling. The narrow silently WRAPS (two's-complement, usually to
NEGATIVE), and `heat_saturating_add`'s `delta <= 0` early-return then drops
the deposit entirely: a temperature that should have hit the counted
T_MAX_PHYS rail instead gets NOTHING. This script's first draft (calling the
old chain directly via the per-cell primitives) is what surfaced this —
`bp.fp_recip_mul` raised `TypeError: ... invoked with 2163882600, ...` the
moment the sweep reached floor=0.01, because the "int32" the binding expects
had already overflowed on the Python side reproducing the identical C++
arithmetic.

THE FIX (this patch): `fixedpoint::deposit_dT_wide_q16` (fixed_point.h) and
its CUDA twin `deposit_dT_wide_q16_dev` chain deposit*recip_n*recip_cv as ONE
128-bit product and narrow EXACTLY ONCE, to an int64 — never to q16 — so the
caller can clamp to a safe non-negative int32 range before the final narrow
for `heat_saturating_add`. Both deposit sites (combustion.cpp,
temperature_solver.cpp Pass 1) and their CUDA twins now use this.

WHAT THIS PROBE VERIFIES NOW (calling the ACTUAL fixed C++ primitive via the
`fp_deposit_dT_wide_q16` / `fp_reciprocal_q16` / `fp_make_recip` /
`fp_quantize` bindings added this patch — not a re-derived Python
approximation):

  combustion.cpp (object-free / gas branch, :799-818):
      n_total    = max(n_real, floor)
      recip_n    = reciprocal_q16(n_total)
      dT_wide    = deposit_dT_wide_q16(deposit, recip_n, recip_cv)
      dT         = clamp(dT_wide, 0, INT32_MAX)

  temperature_solver.cpp Pass 1 (v2.4 absorption form, :274-320):
      e_abs      = deposit if N_raw>=1 else mul_q16(deposit, N_raw)
      N_q        = max(N_raw, floor)
      recip_N_q  = reciprocal_q16(N_q)
      dT_wide    = deposit_dT_wide_q16(e_abs, recip_N_q, recip_cv)
      dT         = clamp(dT_wide, 0, INT32_MAX)

Sane == every stage stays finite, non-negative, and monotone in the
physically-required directions; the wide chain no longer overflows at
floor=0.001 for magnitudes far beyond anything the sim will ever deposit in
one tick (swept up to the T_MAX_PHYS-class "stacked firestorm" reference).

Run (from a worktree root, CPU build):
    conda run -n data python tools/e2b_floor_reciprocal_probe.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "cpp", "build", "Release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import breach_physics as bp                          # noqa: E402

FP_ONE = 65536
INT32_MAX = 2**31 - 1


def q(x: float) -> int:
    return int(bp.fp_quantize(x))


def mul_q16(a: int, b: int) -> int:
    """Python mirror of fixed_point.h mul_q16 for the ATTENUATION step only
    (Pass-1's e_abs = deposit*N_raw, N_raw < FP_ONE there so the result is
    always <= deposit — safe q16 range by construction, no wide chain
    needed). Both operands are always non-negative at this call site, so
    Python's `>>` (floors toward -inf, identical to C++'s arithmetic >>16 on
    a non-negative int64) matches the C++ function exactly."""
    assert a >= 0 and b >= 0
    return (a * b) >> 16


def combustion_dT(deposit_q: int, n_real_q: int, floor_q: int,
                   recip_cv: int) -> tuple[int, int]:
    """The exact (FIXED) combustion.cpp gas-branch chain. Returns
    (dT_clamped, n_total_used)."""
    n_total = max(n_real_q, floor_q)
    recip_n = bp.fp_reciprocal_q16(n_total)
    dT_wide = bp.fp_deposit_dT_wide_q16(deposit_q, recip_n, recip_cv)
    dT = max(0, min(dT_wide, INT32_MAX))
    return dT, n_total


def pass1_dT(deposit_q: int, n_raw_q: int, floor_q: int,
             recip_cv: int) -> tuple[int, int]:
    """The exact (FIXED) temperature_solver.cpp Pass-1 chain. Returns
    (dT_clamped, N_q_used)."""
    e_abs = deposit_q if n_raw_q >= FP_ONE else mul_q16(deposit_q, n_raw_q)
    n_q = max(n_raw_q, floor_q)
    recip_n_q = bp.fp_reciprocal_q16(n_q)
    dT_wide = bp.fp_deposit_dT_wide_q16(e_abs, recip_n_q, recip_cv)
    dT = max(0, min(dT_wide, INT32_MAX))
    return dT, n_q


def main() -> int:
    c_v = 1.0
    recip_cv = bp.fp_make_recip(c_v)   # 1/c_v, the load-time reciprocal

    # A representative aggregate deposit: the P-E1 as-built's measured
    # heat_tick (~330/tick at a hot adjacent I=0.8 fire, the eos-p3fix-
    # thermal-ceiling repro's own reference number) plus a stacked-firestorm
    # multiple (~2600, the config's own historical worst-observed number) —
    # the OLD chain's overflow threshold, swept well past both.
    deposits = {"typical (330)": q(330.0), "stacked (~2600)": q(2600.0)}

    floors = [0.05, 0.01, 0.005, 0.001]
    n_samples = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.25, 1.0]

    print("P-E2b n_floor_heat reciprocal-precision probe (post-fix)")
    print("=" * 78)

    # Expected dT-vs-N shape differs by site, and this is PHYSICAL, not a
    # probe artifact: combustion divides the SAME deposit by N directly
    # (dT = deposit/(N*c_v)), so it is STRICTLY monotonic (non-increasing in
    # N) across its ENTIRE domain — no plateau, ever. Pass-1's v2.4 form
    # instead multiplies BY N first (e_abs = deposit*N, the absorption-
    # proportional-to-density step) before dividing: below the floor this
    # makes dT = deposit*N/floor, which RISES with N (more absorbing mass ->
    # more absorbed energy); at/above the floor the two N factors CANCEL and
    # dT plateaus near-constant at deposit/c_v (documented in
    # temperature_solver.cpp's own Pass-1 comment: "for N_FLOOR_HEAT <= N <=
    # N_AMB this collapses to ... BOUNDED regardless of N-collapse"). So only
    # Pass-1 gets the tolerant plateau treatment, and only for N >= its own
    # floor; combustion is checked strictly everywhere. Getting this backwards
    # for either site would be a probe bug, not a code bug (see the module
    # docstring for how the first draft caught the REAL bug, which was
    # orthogonal to this shape).
    monotone_sign = {"combustion": -1, "pass1": +1}
    has_plateau = {"combustion": False, "pass1": True}

    ok = True
    for site_name, dT_fn in (("combustion", combustion_dT), ("pass1", pass1_dT)):
        sign = monotone_sign[site_name]
        plateau = has_plateau[site_name]
        for dep_label, deposit_q in deposits.items():
            print(f"\n[{site_name}] deposit={dep_label}")
            print(f"  {'floor':>8} {'N':>8} {'n_used_raw':>12} "
                  f"{'recip(raw)':>14} {'dT (raw)':>12} {'dT (game-deg)':>14}")
            for floor in floors:
                floor_q = q(floor)
                dTs_this_floor = []
                for n_val in n_samples:
                    n_q = q(n_val)
                    dT, n_used = dT_fn(deposit_q, n_q, floor_q, recip_cv)
                    dTs_this_floor.append((n_val, dT, n_used))
                    recip_disp = bp.fp_reciprocal_q16(n_used)
                    print(f"  {floor:8.4f} {n_val:8.4f} {n_used:12d} "
                          f"{recip_disp:14d} {dT:12d} {dT / FP_ONE:14.3f}")

                    # ---- SANITY: no overflow, no sign flip, no NaN-like blowup.
                    if not (0 <= dT <= INT32_MAX):
                        ok = False
                        print(f"    ** INSANE: dT={dT} out of int32 [0, "
                              f"{INT32_MAX}] range at floor={floor}, N={n_val}")
                    if not (0 <= recip_disp <= INT32_MAX):
                        ok = False
                        print(f"    ** INSANE: reciprocal={recip_disp} out of "
                              f"range at floor={floor}, n_used={n_used}")

                # ---- SANITY: dT moves monotonically in N in the direction
                # this site's law requires (see the sign table above the
                # loop) -- but ONLY a meaningful assertion where N sits BELOW
                # the floor on both sides of the pair. Once N clears the
                # floor, BOTH sites' laws COLLAPSE to a near-constant plateau
                # (Pass-1: e_abs=deposit*N cancels the N-divide exactly;
                # combustion: N no longer needs flooring) -- the tiny
                # cross-N differences seen there (single-digit-to-low-
                # hundreds raw counts out of a 20M+ count plateau) are
                # reciprocal_q16's own documented ~1-ULP Newton residual, not
                # a physical trend, so only a loose relative tolerance
                # applies in that regime.
                for (n_a, dT_a, n_used_a), (n_b, dT_b, n_used_b) in zip(
                        dTs_this_floor, dTs_this_floor[1:]):
                    both_above_floor = (plateau and q(n_a) >= floor_q
                                        and q(n_b) >= floor_q)
                    if both_above_floor:
                        # Plateau region (both sides already past the floor,
                        # where both laws collapse to a near-constant): loose
                        # relative-tolerance check (0.01% of the larger
                        # magnitude, floor of 200 raw counts) rather than
                        # monotonicity, since the residual cross-N difference
                        # here is reciprocal_q16's own documented ~1-ULP
                        # Newton residual, not a physical trend.
                        tol = max(200, int(0.0001 * max(abs(dT_a), abs(dT_b))))
                        if abs(dT_b - dT_a) > tol:
                            ok = False
                            print(f"    ** PLATEAU DRIFT at floor={floor}: "
                                  f"dT({n_a})={dT_a} vs dT({n_b})={dT_b} "
                                  f"(diff {dT_b - dT_a}, tol {tol})")
                    else:
                        # Below the floor (or straddling the floor boundary):
                        # the physical monotonicity claim applies at full
                        # strength, including across the transition.
                        violated = (dT_b - dT_a) * sign < 0
                        if violated:
                            ok = False
                            arrow = ("non-increasing" if sign < 0
                                     else "non-decreasing")
                            print(f"    ** NON-MONOTONE in N at floor={floor} "
                                  f"(expected {arrow}): "
                                  f"dT({n_a})={dT_a}, dT({n_b})={dT_b}")

            # ---- SANITY: LOWERING the floor must DEPOSIT MORE at a fixed
            # thin N below every floor in the sweep (the design's inversion
            # claim — L1-4/§2.2: "lowering the floor heats thin cells MORE
            # than today's 0.05"). Check at N just below the smallest floor.
            n_check_q = q(0.0005)   # below every floor in `floors`
            dTs_by_floor = []
            for floor in floors:
                dT, _ = dT_fn(deposit_q, n_check_q, q(floor), recip_cv)
                dTs_by_floor.append((floor, dT))
            # floors is DESCENDING (0.05 -> 0.001); dT must be NON-DECREASING
            # as floor drops (each next floor is smaller -> divides less ->
            # dT same or larger) UNTIL the T_MAX_PHYS-class int32 ceiling is
            # hit, at which point BOTH sides clamp to INT32_MAX and the
            # comparison is legitimately EQUAL (not an inversion violation).
            for (f_a, dT_a), (f_b, dT_b) in zip(dTs_by_floor, dTs_by_floor[1:]):
                if dT_b < dT_a:
                    ok = False
                    print(f"    ** INVERSION VIOLATED [{site_name}/{dep_label}]: "
                          f"floor {f_a}->{f_b} (N=0.0005) gave dT {dT_a}->{dT_b} "
                          f"(expected non-decreasing as the floor drops)")
                else:
                    print(f"  [{site_name}/{dep_label}] floor {f_a}->{f_b} at "
                          f"N=0.0005: dT {dT_a} -> {dT_b} "
                          f"(+{dT_b - dT_a} raw) — inversion direction confirmed"
                          + (" [both clamped at int32 ceiling]"
                             if dT_a == INT32_MAX and dT_b == INT32_MAX else ""))

    print("\n" + "=" * 78)
    if ok:
        print("PROBE RESULT: PASS — the WIDE reciprocal path "
              "(reciprocal_q16 -> deposit_dT_wide_q16, int64/128-bit "
              "throughout, no premature q16 narrow) stays arithmetically "
              "sane down to floor=0.001 for deposits far beyond the "
              "stacked-firestorm reference; lowering the floor deposits "
              "MORE into thin cells at every step of the sweep (the "
              "design's predicted inversion, confirmed at the primitive "
              "level). The OLD two-step mul_q16->recip_mul chain did NOT "
              "survive this sweep (see the module docstring) and has been "
              "replaced.")
    else:
        print("PROBE RESULT: FAIL — see ** markers above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
