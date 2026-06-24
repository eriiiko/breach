// S1 adversarial-review proof harness (standalone, no pybind needed).
//
// Proves the two water_solver.cpp fixes at the ARITHMETIC level, over a wide
// random sweep including negatives:
//
//   Fix 1 (outflow limiter, MAGNITUDE scale): scale_mag(x, scale) for any
//          x and any scale in [0, FP_ONE] never GROWS |x| (|scaled| <= |x|),
//          whereas the old mul_q16(x, scale) grows |x| by up to 1 count for
//          negative x. We measure the worst-case over-grow of both and assert
//          scale_mag's is 0. We also reproduce the donor-cell invariant: the
//          SUM of magnitudes of a cell's scaled outgoing faces is <= out_sum
//          scaled (<= depth), i.e. no over-drain.
//
//   Fix 2 (flux_to_dq one-truncation): the MSVC two-truncation form
//          (flux>>16 then mul_q16) vs the __int128 / _mul128 one-truncation
//          form. We compute BOTH for every sample and report (a) how often they
//          differ (the ~5% desync) and (b) that the new one-truncation MSVC
//          form (_mul128 combine) is BIT-IDENTICAL to the __int128 reference on
//          this machine (it has __int128 under clang/gcc; under MSVC the
//          _mul128 path is the reference and we cross-check the algebraic
//          128-bit form).
//
// Build (MSVC):
//   cl /std:c++20 /O2 /EHsc /I cpp/src tests/_s1_flux_truncation_check.cpp \
//      /Fe:tests/_s1_flux_truncation_check.exe
// Build (gcc/clang, has __int128):
//   g++ -std=c++20 -O2 -I cpp/src tests/_s1_flux_truncation_check.cpp \
//      -o tests/_s1_flux_truncation_check
//
// Exit 0 == all proofs hold.

#include "fixed_point.h"
#include <cstdint>
#include <cstdio>
#include <random>
#include <algorithm>

#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
#include <intrin.h>
#endif

using namespace fixedpoint;

// ---- the OLD limiter scale (mul_q16, toward -inf) — for the before/after -----
static q16 scale_old(q16 x, q16 scale) {
    return mul_q16(x, scale);   // the buggy form: >>16 grows negative magnitude
}

// ---- reference one-truncation flux->dq (the __int128 path semantics) ---------
// Computed in a portable 128-bit-accurate way for the REFERENCE (we use long
// double-free exact bigint via two 64-bit halves only where __int128 is absent).
#if defined(__SIZEOF_INT128__)
static q16 flux_to_dq_int128(int64_t flux_wide, q16 dt_over_dx_q) {
    __int128 p = (__int128)flux_wide * (__int128)dt_over_dx_q;
    return (q16)(p >> 32);
}
#endif

// ---- the NEW MSVC one-truncation form (the _mul128 + recip_mul combine) ------
static q16 flux_to_dq_mul128(int64_t flux_wide, q16 dt_over_dx_q) {
#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
    long long hi;
    long long lo = _mul128((long long)flux_wide, (long long)dt_over_dx_q, &hi);
    unsigned long long ulo = (unsigned long long)lo;
    long long res = (long long)((ulo >> 32) | ((unsigned long long)hi << (64 - 32)));
    return (q16)res;
#else
    // On a __int128 toolchain, reproduce the SAME hi:lo combine by hand from the
    // 128-bit product so we are testing the identical bit recipe MSVC runs.
    __int128 p = (__int128)flux_wide * (__int128)dt_over_dx_q;
    unsigned __int128 up = (unsigned __int128)p;
    long long hi = (long long)(p >> 64);
    unsigned long long lo = (unsigned long long)up;     // low 64 bits
    long long res = (long long)((lo >> 32) | ((unsigned long long)hi << (64 - 32)));
    return (q16)res;
#endif
}

// ---- the OLD MSVC two-truncation form (flux>>16 then mul_q16) -----------------
static q16 flux_to_dq_two_trunc(int64_t flux_wide, q16 dt_over_dx_q) {
    const q16 flux_q16 = (q16)(flux_wide >> 16);
    return mul_q16(flux_q16, dt_over_dx_q);
}

// ---- an INDEPENDENT exact reference: (flux_wide * dt_over_dx_q) >> 32 computed
//      via an exact signed 128-bit product from 64-bit halves (NO __int128, NO
//      _mul128) and a true arithmetic >>32. This is the ground truth that BOTH
//      the __int128 path and the _mul128 combine must equal. Runs on every
//      toolchain, so it proves Fix 2's one-truncation form is correct on MSVC
//      too (where the __int128 reference is unavailable).
static q16 flux_to_dq_exact_ref(int64_t flux_wide, q16 dt_over_dx_q) {
    // signed*signed exact 128-bit via sign-extension of |a|*|b|.
    const bool neg = (flux_wide < 0) ^ (dt_over_dx_q < 0);
    unsigned long long a = (unsigned long long)(flux_wide < 0 ? -flux_wide : flux_wide);
    unsigned long long b = (unsigned long long)(dt_over_dx_q < 0 ? -(long long)dt_over_dx_q
                                                                 : (long long)dt_over_dx_q);
    // 64x64 -> 128 unsigned via 32-bit halves.
    const unsigned long long aL = a & 0xFFFFFFFFull, aH = a >> 32;
    const unsigned long long bL = b & 0xFFFFFFFFull, bH = b >> 32;
    const unsigned long long ll = aL * bL;
    const unsigned long long lh = aL * bH;
    const unsigned long long hl = aH * bL;
    const unsigned long long hh = aH * bH;
    const unsigned long long mid = (ll >> 32) + (lh & 0xFFFFFFFFull) + (hl & 0xFFFFFFFFull);
    unsigned long long lo = (ll & 0xFFFFFFFFull) | (mid << 32);
    unsigned long long hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    // two's-complement negate the 128-bit magnitude if the product is negative.
    if (neg) { lo = ~lo + 1ull; hi = ~hi + (lo == 0ull ? 1ull : 0ull); }
    // arithmetic >>32 of the signed 128-bit value {hi:lo}: take low 32 of hi into
    // top of result, combine with hi 32 of lo. The result lands in 64 bits well
    // within range for our inputs; cast low 64 to signed then to q16.
    long long res64 = (long long)((lo >> 32) | (hi << 32));
    return (q16)res64;
}

int main() {
    std::mt19937_64 rng(1234567ull);

    // ===================== Fix 1: scale_mag never grows |x| ===================
    // x is a per-face dq in Q16.16; scale is a limiter factor in [0, FP_ONE].
    // Sweep large magnitudes incl. v_max*deep-depth-scale fluxes.
    int64_t worst_overgrow_new = 0;   // max( |scaled| - |x| ) over the sweep, scale_mag
    int64_t worst_overgrow_old = 0;   // same for the old mul_q16 form
    int neg_grow_old = 0;             // count of negative-x cases the OLD form grew
    long long old_more_neg = 0;       // cases where OLD |result| > NEW |result| (the
                                      // per-face 1-count difference that sums to drain)
    int64_t worst_old_minus_new = 0;  // worst (|old| - |new|) per face
    const long long N = 20'000'000LL;
    std::uniform_int_distribution<int64_t> dx_dist(-(1LL << 30), (1LL << 30));
    std::uniform_int_distribution<int32_t> sc_dist(0, FP_ONE);  // [0, 1.0]
    for (long long k = 0; k < N; ++k) {
        const q16 x     = (q16)dx_dist(rng);
        const q16 scale = (q16)sc_dist(rng);
        const int64_t ax = x < 0 ? -(int64_t)x : (int64_t)x;

        const q16 sn = scale_mag(x, scale);
        const q16 so = scale_old(x, scale);
        const int64_t asn = sn < 0 ? -(int64_t)sn : (int64_t)sn;
        const int64_t aso = so < 0 ? -(int64_t)so : (int64_t)so;

        worst_overgrow_new = std::max(worst_overgrow_new, asn - ax);
        worst_overgrow_old = std::max(worst_overgrow_old, aso - ax);
        if (x < 0 && aso > ax) ++neg_grow_old;
        if (aso > asn) { ++old_more_neg; worst_old_minus_new =
                         std::max(worst_old_minus_new, aso - asn); }
    }

    // ===================== Fix 1: donor over-drain proof ======================
    // A donor cell has up to 4 outgoing faces. out_sum (>depth) triggers the
    // limiter: scale = (depth<<16)/out_sum. After scaling EACH outgoing face by
    // scale_mag, the SUM of scaled magnitudes must be <= depth (no over-drain;
    // max(depth,0) can never inject). We sweep random (depth, 4 face magnitudes).
    int64_t worst_overdrain_new = 0;  // max( sum_scaled - depth ) when limiter fires
    int64_t worst_overdrain_old = 0;
    std::uniform_int_distribution<int64_t> depth_dist(0, 8LL * FP_ONE);    // [0,8m]
    std::uniform_int_distribution<int64_t> face_dist(-3LL * FP_ONE, 3LL * FP_ONE);
    const long long M = 8'000'000LL;
    for (long long k = 0; k < M; ++k) {
        const int64_t depth = depth_dist(rng);
        q16 f[4];
        int64_t out_sum = 0;
        for (int j = 0; j < 4; ++j) {
            f[j] = (q16)face_dist(rng);
            out_sum += (f[j] < 0 ? -(int64_t)f[j] : (int64_t)f[j]);  // all treated as outgoing
        }
        if (out_sum <= depth) continue;               // limiter would not fire
        const q16 scale = (q16)((depth << FP_SHIFT) / out_sum);
        int64_t sum_new = 0, sum_old = 0;
        for (int j = 0; j < 4; ++j) {
            const q16 a = scale_mag(f[j], scale);
            const q16 b = scale_old(f[j], scale);
            sum_new += (a < 0 ? -(int64_t)a : (int64_t)a);
            sum_old += (b < 0 ? -(int64_t)b : (int64_t)b);
        }
        worst_overdrain_new = std::max(worst_overdrain_new, sum_new - depth);
        worst_overdrain_old = std::max(worst_overdrain_old, sum_old - depth);
    }

    // ===================== Fix 2: two-vs-one truncation =======================
    // Sweep random wide fluxes (Q32.32) incl. negatives and random dt_over_dx_q.
    long long differ = 0, total = 0;
    int64_t worst_two_vs_one = 0;        // |two_trunc - one_trunc| worst case
    int64_t mismatch_new_vs_ref = 0;     // must be 0: new MSVC form == int128 ref
    int64_t mismatch_new_vs_exact = 0;   // must be 0: new form == exact 128-bit ref
    std::uniform_int_distribution<int64_t> flux_dist(-(1LL << 50), (1LL << 50));
    std::uniform_int_distribution<int32_t> dtdx_dist(1, FP_ONE);  // (0,1] typical
    const long long F = 20'000'000LL;
    for (long long k = 0; k < F; ++k) {
        const int64_t flux = flux_dist(rng);
        const q16 dtdx = (q16)dtdx_dist(rng);
        const q16 one_t = flux_to_dq_mul128(flux, dtdx);
        const q16 two_t = flux_to_dq_two_trunc(flux, dtdx);
        const q16 exact = flux_to_dq_exact_ref(flux, dtdx);
        if (one_t != exact) ++mismatch_new_vs_exact;
#if defined(__SIZEOF_INT128__)
        const q16 ref = flux_to_dq_int128(flux, dtdx);
        if (one_t != ref) ++mismatch_new_vs_ref;
#endif
        ++total;
        if (one_t != two_t) {
            ++differ;
            int64_t d = (int64_t)two_t - (int64_t)one_t;
            if (d < 0) d = -d;
            worst_two_vs_one = std::max(worst_two_vs_one, d);
        }
    }

    std::printf("=== Fix 1: limiter MAGNITUDE scale (scale_mag vs mul_q16) ===\n");
    std::printf("  sweep N=%lld  random x in +/-2^30, scale in [0,1.0]\n", N);
    std::printf("  worst |scaled|-|x|  NEW scale_mag = %lld  (must be <= 0)\n",
                (long long)worst_overgrow_new);
    std::printf("  worst |scaled|-|x|  OLD mul_q16   = %lld  (the +1 over-grow bug)\n",
                (long long)worst_overgrow_old);
    std::printf("  negative-x cases the OLD form grew = %d\n", neg_grow_old);
    std::printf("  faces where OLD |scaled| > NEW |scaled| = %lld  (worst diff = %lld count)\n",
                old_more_neg, (long long)worst_old_minus_new);
    std::printf("=== Fix 1: donor over-drain (sum scaled outflow vs depth) ===\n");
    std::printf("  sweep M=%lld limiter-firing 4-face donors\n", M);
    std::printf("  worst over-drain NEW = %lld counts  (must be <= 0)\n",
                (long long)worst_overdrain_new);
    std::printf("  worst over-drain OLD = %lld counts  (the leak)\n",
                (long long)worst_overdrain_old);
    std::printf("=== Fix 2: flux_to_dq one- vs two-truncation ===\n");
    std::printf("  sweep F=%lld random wide fluxes incl. negatives\n", F);
    std::printf("  two-trunc vs one-trunc differ on %lld / %lld faces = %.2f%%\n",
                differ, total, 100.0 * (double)differ / (double)total);
    std::printf("  worst |two - one| = %lld count(s)\n", (long long)worst_two_vs_one);
    std::printf("  one-trunc form vs EXACT 128-bit ref mismatches = %lld (must be 0)\n",
                (long long)mismatch_new_vs_exact);
#if defined(__SIZEOF_INT128__)
    std::printf("  one-trunc form vs __int128 path mismatches     = %lld (must be 0)\n",
                (long long)mismatch_new_vs_ref);
#else
    std::printf("  (MSVC build: __int128 ref unavailable; the exact 128-bit ref above stands in)\n");
#endif

    bool ok = (worst_overgrow_new <= 0)
           && (worst_overdrain_new <= 0)
           && (mismatch_new_vs_ref == 0)
           && (mismatch_new_vs_exact == 0);
    std::printf("\nRESULT: %s\n", ok ? "PASS (all proofs hold)" : "FAIL");
    return ok ? 0 : 1;
}
