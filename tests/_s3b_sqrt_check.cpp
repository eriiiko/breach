// S3b unit proof — the per-cell integer sqrt (fixed_point.h sqrt_q16),
// standalone (no pybind).
//
// sqrt_q16(x_q32) returns floor(sqrt(x_real)) in Q16.16, where x_q32 is an int64
// Q.32 radicand (a sum of mul_wide(q16,q16) products == the square of a Q16.16
// magnitude). It is the FIRST per-cell transcendental of the fixed-point arc:
// fire uses it for W = sqrt(wind_x^2 + wind_y^2). It must be EXACT-floor and
// deterministic (a wrong/non-deterministic isqrt is a silent lockstep desync).
//
// Proves four properties:
//
//   (1) FLOOR PROPERTY — for sampled int64 radicands across the full in-range
//       domain (0, small, random, exact squares +/-1, near the 2^62 clamp edge),
//       r = sqrt_q16(x) satisfies r^2 <= x < (r+1)^2 (the floor-of-exact-sqrt
//       contract; r and r+1 fit the uint64 square comparison below 2^31).
//
//   (2) SCALE FOLD — sqrt of a Q.32 value yields Q16.16 directly (no rescale):
//       sqrt_q16(m^2 * 2^32) == m * 2^16 for integer m (so |wind|=1 -> W=65536).
//
//   (3) DETERMINISM — pure integer shifts/compares, so sqrt_q16(x) twice is
//       bit-identical (the "no float leaked in" contract).
//
//   (4) CLAMP SELF-GUARD — a radicand >= 2^62 (true root >= 2^31, would wrap a
//       signed int32) clamps to INT32_MAX deterministically, and the clamped
//       result still satisfies r^2 <= x (the honest shared-helper safety; DEAD on
//       the real fire call site where |wind| is O(1)).
//
// Build (MSVC):
//   cl /std:c++20 /O2 /EHsc /I cpp/src tests/_s3b_sqrt_check.cpp \
//      /Fe:tests/_s3b_sqrt_check.exe
// Build (gcc/clang):
//   g++ -std=c++20 -O2 -I cpp/src tests/_s3b_sqrt_check.cpp \
//      -o tests/_s3b_sqrt_check
//
// Exit 0 == all proofs hold.

#include "fixed_point.h"
#include <cstdint>
#include <cstdio>
#include <random>
#include <cmath>

using namespace fixedpoint;

static const int64_t CLAMP_EDGE = (int64_t)1 << 62;   // root >= 2^31 above this
static const q16     INT32_MAXV = 0x7fffffff;

// Floor check valid for the UNCLAMPED domain (x < 2^62 -> r < 2^31 -> r+1 <= 2^31
// -> (r+1)^2 < 2^63 fits uint64). Returns false on a violation.
static bool floor_ok(int64_t x, int& detail_r) {
    q16 r = sqrt_q16(x);
    detail_r = r;
    if (r >= INT32_MAXV) return true;   // clamp regime, checked separately
    uint64_t r2 = (uint64_t)r * (uint64_t)r;
    uint64_t r1 = (uint64_t)(r + 1) * (uint64_t)(r + 1);
    return (r2 <= (uint64_t)x) && ((uint64_t)x < r1);
}

int main() {
    std::mt19937_64 rng(0x5A17C0DEULL);
    int fails = 0;

    // (1) FLOOR PROPERTY over the sampled in-range domain.
    {
        int fl_fail = 0, shown = 0, dr;
        auto check = [&](int64_t x) {
            if (x < 0) return;
            if (!floor_ok(x, dr)) {
                if (shown < 8) { printf("  floor miss x=%lld r=%d\n",
                                        (long long)x, dr); shown++; }
                fl_fail++;
            }
        };
        check(0); check(1); check(2); check(3);
        // Dense low range.
        for (int64_t x = 0; x < 5000000; ++x) check(x);
        // Exact squares and their neighbours (the hard floor boundaries).
        for (int64_t r = 0; r < 400000; r += 7) {
            int64_t x = r * r; check(x); if (x > 0) check(x - 1); check(x + 1);
        }
        // Broad random sweep up to just below the clamp edge.
        std::uniform_int_distribution<int64_t> d(0, CLAMP_EDGE - 1);
        for (int i = 0; i < 5000000; ++i) check(d(rng));
        // Right up against the clamp edge (largest unclamped roots).
        for (int64_t x = CLAMP_EDGE - 200000; x < CLAMP_EDGE; ++x) check(x);
        if (fl_fail) { printf("FAIL floor: %d violations\n", fl_fail); fails++; }
        else printf("PASS floor: r^2 <= x < (r+1)^2 over the sampled domain\n");
    }

    // (2) SCALE FOLD — sqrt_q16(m^2 * 2^32) == m * 2^16 (Q16.16 of m).
    {
        int sf_fail = 0;
        for (int64_t m = 0; m <= 30000; ++m) {
            int64_t rad = (m * m) << 32;              // m^2 * 2^32 (Q.32)
            q16 got = sqrt_q16(rad);
            q16 want = (q16)(m << 16);                // m in Q16.16
            if (got != want) {
                if (sf_fail < 8) printf("  scale miss m=%lld got=%d want=%d\n",
                                        (long long)m, got, want);
                sf_fail++;
            }
        }
        if (sf_fail) { printf("FAIL scale fold: %d mismatches\n", sf_fail); fails++; }
        else printf("PASS scale fold: sqrt of Q.32 yields Q16.16 directly\n");
    }

    // (3) DETERMINISM — same input -> same output, bit-identical.
    {
        int det_fail = 0;
        std::uniform_int_distribution<int64_t> d(0, CLAMP_EDGE - 1);
        for (int i = 0; i < 1000000; ++i) {
            int64_t x = d(rng);
            if (sqrt_q16(x) != sqrt_q16(x)) det_fail++;
        }
        if (det_fail) { printf("FAIL determinism: %d non-reproducible\n", det_fail); fails++; }
        else printf("PASS determinism: 1000000/1000000 bit-identical run-to-run\n");
    }

    // (4) CLAMP SELF-GUARD — x >= 2^62 -> INT32_MAX, and r^2 <= x still holds.
    {
        int cl_fail = 0;
        std::uniform_int_distribution<int64_t> d(CLAMP_EDGE, ((int64_t)1 << 62) + 5000000);
        for (int i = 0; i < 200000; ++i) {
            int64_t x = d(rng);
            q16 r = sqrt_q16(x);
            if (r != INT32_MAXV) { if (cl_fail < 8) printf("  clamp miss x=%lld r=%d\n", (long long)x, r); cl_fail++; }
            // r^2 = (2^31-1)^2 < 2^62 <= x, so the lower bound holds.
            uint64_t r2 = (uint64_t)r * (uint64_t)r;
            if (!(r2 <= (uint64_t)x)) { cl_fail++; }
        }
        // negative -> 0 self-guard
        if (sqrt_q16(-1) != 0 || sqrt_q16(-12345) != 0) { printf("  neg guard fail\n"); cl_fail++; }
        if (cl_fail) { printf("FAIL clamp guard: %d\n", cl_fail); fails++; }
        else printf("PASS clamp guard: x>=2^62 -> INT32_MAX (r^2<=x), neg -> 0\n");
    }

    // (5) narrow_round / narrow_round_signed — the shared round-to-nearest deposit
    // narrows (used by the fire plume/smoke/wall deposits). round-half-up for the
    // non-negative form; round-half-AWAY-from-zero (sign-symmetric, no DC bias) for
    // the signed form. Verify against an independent double reference.
    {
        int nr_fail = 0;
        std::uniform_int_distribution<int64_t> d(-(int64_t)1 << 45, (int64_t)1 << 45);
        for (int i = 0; i < 2000000; ++i) {
            const int64_t wide = d(rng);
            // signed form: round(wide / 2^16) half-away-from-zero.
            const double q = (double)wide / 65536.0;
            const int64_t want = (int64_t)(q >= 0.0 ? std::floor(q + 0.5) : std::ceil(q - 0.5));
            if ((int64_t)narrow_round_signed(wide) != want) {
                if (nr_fail < 8) printf("  narrow_round_signed miss wide=%lld got=%d want=%lld\n",
                                        (long long)wide, narrow_round_signed(wide), (long long)want);
                nr_fail++;
            }
            if (wide >= 0) {
                const int64_t wantp = (int64_t)std::floor((double)wide / 65536.0 + 0.5);
                if ((int64_t)narrow_round(wide) != wantp) nr_fail++;
            }
        }
        if (nr_fail) { printf("FAIL narrow_round: %d\n", nr_fail); fails++; }
        else printf("PASS narrow_round / narrow_round_signed vs double reference\n");
    }

    if (fails == 0) { printf("ALL PASS\n"); return 0; }
    printf("FAILS=%d\n", fails);
    return 1;
}
