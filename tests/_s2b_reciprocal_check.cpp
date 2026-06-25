// S2b unit proof — the per-cell integer reciprocal (fixed_point.h
// reciprocal_q16), standalone (no pybind).
//
// reciprocal_q16(denom_q) returns the Q16.16 reciprocal 2^16 / denom_real via a
// pure-integer 3-step Newton-Raphson (r <- r*(2 - d*r)) seeded from a power-of-2.
// It is the S2 per-cell divide primitive: S2b uses it for the integer-SL
// bilinear renorm 1/wsum (wsum in (0,1]); S2c reuses it for the GS Dinv
// 1/(1+mu*wsum) (denom > 1). So it must be accurate AND deterministic across the
// WHOLE positive range, not just wsum<=1.
//
// Proves three properties:
//
//   (1) ACCURACY vs 1/d — over a wide sweep of positive Q16.16 denominators
//       (small fractions through large values), the integer reciprocal matches
//       the true 2^16/denom_real to a tight tolerance. The hard cases are SMALL
//       denominators (1/d is large, sensitive) and the two call-site regimes:
//       wsum in (0,1] (recip >= 1) and Dinv-style denom in [1, ~40] (recip < 1).
//
//   (2) DETERMINISM — pure integer ops, so reciprocal_q16(d) called twice is
//       bit-identical (trivially true; asserted to pin the "no float leaked in"
//       contract the way the prototype's checksum assert does).
//
//   (3) MONOTONICITY-ish SANITY — recip(d) is non-increasing as d grows (a
//       larger denominator gives a smaller-or-equal reciprocal), catching a
//       broken seed/convergence that would non-monotonically jump.
//
// Build (MSVC):
//   cl /std:c++20 /O2 /EHsc /I cpp/src tests/_s2b_reciprocal_check.cpp \
//      /Fe:tests/_s2b_reciprocal_check.exe
// Build (gcc/clang):
//   g++ -std=c++20 -O2 -I cpp/src tests/_s2b_reciprocal_check.cpp \
//      -o tests/_s2b_reciprocal_check
//
// Exit 0 == all proofs hold.

#include "fixed_point.h"
#include <cstdint>
#include <cstdio>
#include <random>
#include <cmath>

using namespace fixedpoint;

// True Q16.16 reciprocal in double: 2^16 / denom_real, where denom_real =
// denom_q / 2^16, i.e. 2^32 / denom_q. (double has 52 mantissa bits >> the 31
// we need, so this reference is exact to far better than 1 count.)
static double ref_recip(q16 denom_q) {
    return (double)((int64_t)1 << 32) / (double)denom_q;
}

int main() {
    std::mt19937_64 rng(0xC0FFEE5EEDD00DULL);
    int fails = 0;

    // (1) ACCURACY over a wide sweep. The tolerance is on the RECIPROCAL VALUE.
    // Newton with a power-of-2 seed + 3 iterations + >>16 truncation each step
    // lands within a few counts of the true reciprocal across the range; we use
    // a relative tolerance (the absolute error grows with the magnitude of 1/d)
    // plus a small absolute floor for large denominators (recip near 0).
    {
        int acc_fail = 0;
        int worst_print = 0;
        double worst_rel = 0.0;
        // Two regimes matter most (the two call sites):
        //   A) wsum in (0,1]   -> denom_q in (0, FP_ONE], recip >= 1
        //   B) Dinv 1/(1+mu*w) -> denom_q in [FP_ONE, ~64*FP_ONE], recip in (0,1]
        // plus a broad random sweep across both.
        for (int trial = 0; trial < 400000; ++trial) {
            q16 denom_q;
            if (trial < 100000) {
                // Regime A: wsum in [1/256, 1]  (the SL bilinear partial-weight
                // range; the caller floors wsum at FP_ONE>>8 = 256 counts).
                std::uniform_int_distribution<int32_t> d(FP_ONE >> 8, FP_ONE);
                denom_q = d(rng);
            } else if (trial < 200000) {
                // Regime B: Dinv denom in [1, 64]  (mu~8.3, wsum<=4 -> 1+mu*wsum
                // up to ~34; pad to 64 for headroom).
                std::uniform_int_distribution<int32_t> d(FP_ONE, 64 * FP_ONE);
                denom_q = d(rng);
            } else {
                // Broad sweep: any positive denom from ~1/4096 up to ~1024.
                std::uniform_int_distribution<int32_t> d(FP_ONE >> 12, 1024 * FP_ONE);
                denom_q = d(rng);
            }
            const q16 got = reciprocal_q16(denom_q);
            const double want = ref_recip(denom_q);
            const double err = std::fabs((double)got - want);
            const double rel = err / (want > 1.0 ? want : 1.0);
            // Accept if within 4 counts absolute OR 1e-3 relative — Newton with
            // 3 iters + truncation is ~1 ULP in the well-seeded regime, looser
            // for the smallest denominators where 1/d is huge.
            if (err > 4.0 && rel > 1e-3) {
                if (worst_print < 8) {
                    printf("  acc miss denom_q=%d got=%d want=%.2f err=%.2f rel=%.2e\n",
                           denom_q, got, want, err, rel);
                    worst_print++;
                }
                acc_fail++;
            }
            if (rel > worst_rel) worst_rel = rel;
        }
        if (acc_fail) { printf("FAIL accuracy: %d/400000 out of tolerance\n", acc_fail); fails++; }
        else printf("PASS accuracy: 400000/400000 within tol (worst rel=%.2e)\n", worst_rel);
    }

    // (2) DETERMINISM — same input -> same output, bit-identical.
    {
        int det_fail = 0;
        for (int trial = 0; trial < 100000; ++trial) {
            std::uniform_int_distribution<int32_t> d(1, 1024 * FP_ONE);
            const q16 denom_q = d(rng);
            if (reciprocal_q16(denom_q) != reciprocal_q16(denom_q)) det_fail++;
        }
        if (det_fail) { printf("FAIL determinism: %d non-reproducible\n", det_fail); fails++; }
        else printf("PASS determinism: 100000/100000 bit-identical run-to-run\n");
    }

    // (3) MONOTONICITY-ish — recip is non-increasing as denom grows, EXCEPT for a
    // <=1-ULP up-tick at a round-to-nearest boundary (the reciprocal is rounded,
    // not truncated, so it can wobble by 1 count at the half-way crossings; that
    // is correct + deterministic). We allow cur <= prev + 1; a jump of >1 would
    // signal a broken seed/convergence.
    {
        int mono_fail = 0;
        q16 prev = reciprocal_q16(FP_ONE >> 8);
        for (q16 d = (FP_ONE >> 8) + 1; d <= 256 * FP_ONE; d += 37) {
            const q16 cur = reciprocal_q16(d);
            if (cur > prev + 1) {       // a larger denom jumped the reciprocal up >1 ULP
                if (mono_fail < 8)
                    printf("  mono break d=%d cur=%d > prev=%d (+%d)\n",
                           d, cur, prev, cur - prev);
                mono_fail++;
            }
            prev = cur;
        }
        if (mono_fail) { printf("FAIL monotonicity: %d non-monotone steps\n", mono_fail); fails++; }
        else printf("PASS monotonicity: reciprocal non-increasing (<=1 ULP round wobble)\n");
    }

    // denom <= 0 self-guard
    if (reciprocal_q16(0) != 0 || reciprocal_q16(-5) != 0) {
        printf("FAIL denom<=0 guard\n"); fails++;
    } else printf("PASS denom<=0 guard -> 0\n");

    if (fails == 0) { printf("ALL PASS\n"); return 0; }
    printf("FAILS=%d\n", fails);
    return 1;
}
