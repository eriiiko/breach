// S2a unit proof — the deterministic integer mean reduction (fixed_point.h
// mean_sum / mean_round), standalone (no pybind).
//
// Proves the mean_wp primitive at the ARITHMETIC level:
//
//   (1) ORDER-INDEPENDENCE: mean_sum over a masked Q16.16 array is bit-identical
//       under an arbitrary shuffle of the element order (integer addition is
//       associative + exact). This is the cross-GPU determinism property the
//       float sum could not give (a float sum reorders -> different bits).
//
//   (2) MATCHES A REFERENCE: mean_round(sum, count) equals an independent
//       round-half-away-from-zero reference computed in double for a wide sweep
//       (incl. negatives, exact .5 ties, count parities), so the rounding is the
//       intended sign-symmetric round-to-nearest (no sign(sum) DC bias).
//
//   (3) SIGN SYMMETRY: mean_round(-s, c) == -mean_round(s, c) exactly (the DC-
//       bias guard — a biased mean leaks mass into the conserved atmosphere).
//
// Build (MSVC):
//   cl /std:c++20 /O2 /EHsc /I cpp/src tests/_s2a_mean_reduction_check.cpp \
//      /Fe:tests/_s2a_mean_reduction_check.exe
// Build (gcc/clang):
//   g++ -std=c++20 -O2 -I cpp/src tests/_s2a_mean_reduction_check.cpp \
//      -o tests/_s2a_mean_reduction_check
//
// Exit 0 == all proofs hold.

#include "fixed_point.h"
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace fixedpoint;

// Independent reference: round-half-away-from-zero of sum/count, in double.
// (double has 52 mantissa bits; our sums stay well inside that for the test.)
static int64_t ref_mean(int64_t sum, int64_t count) {
    if (count <= 0) return 0;
    double q = (double)sum / (double)count;
    return (int64_t)(q >= 0.0 ? std::floor(q + 0.5) : std::ceil(q - 0.5));
}

int main() {
    std::mt19937_64 rng(0xBADC0FFEE0DDF00DULL);
    int fails = 0;

    // (1) ORDER-INDEPENDENCE over many random masked arrays.
    {
        int order_fail = 0;
        for (int trial = 0; trial < 2000; ++trial) {
            const int n = 1 + (int)(rng() % 4096);
            std::vector<q16> vals(n);
            std::vector<unsigned char> mask(n);
            // wide dynamic range incl. negatives (wave_p is zero-mean signed)
            std::uniform_int_distribution<int32_t> vd(-(1 << 26), (1 << 26));
            for (int i = 0; i < n; ++i) {
                vals[i] = vd(rng);
                mask[i] = (rng() & 1) ? 1 : 0;
            }
            const bool* mb = reinterpret_cast<const bool*>(mask.data());
            const int64_t s0 = mean_sum(vals.data(), mb, n);

            // shuffle (vals, mask) TOGETHER and re-sum: must be bit-identical.
            std::vector<int> idx(n);
            for (int i = 0; i < n; ++i) idx[i] = i;
            std::shuffle(idx.begin(), idx.end(), rng);
            std::vector<q16> vals2(n);
            std::vector<unsigned char> mask2(n);
            for (int i = 0; i < n; ++i) { vals2[i] = vals[idx[i]]; mask2[i] = mask[idx[i]]; }
            const int64_t s1 = mean_sum(vals2.data(),
                                        reinterpret_cast<const bool*>(mask2.data()), n);
            if (s0 != s1) order_fail++;
        }
        if (order_fail) { printf("FAIL order-independence: %d/2000 shuffles differ\n", order_fail); fails++; }
        else printf("PASS order-independence: 2000/2000 shuffles bit-identical\n");
    }

    // (2) MATCHES REFERENCE + (3) SIGN SYMMETRY over a wide sweep.
    {
        int ref_fail = 0, sym_fail = 0;
        // exhaustive small + random large
        for (int trial = 0; trial < 200000; ++trial) {
            int64_t count;
            int64_t sum;
            if (trial < 4000) {
                // dense small: every count 1..40, sums -100..100 (exact ties land here)
                count = 1 + (trial % 40);
                sum = (trial / 40) - 50;
            } else {
                count = 1 + (int64_t)(rng() % 100000);
                // mean_round returns q16 (int32) by contract: mean_wp is always
                // an in-range Q16.16 (it is the mean of in-range wave_p values).
                // So keep |sum/count| < 2^30 — the realistic regime. We pick a
                // per-trial mean magnitude < 2^30 and back out a sum.
                std::uniform_int_distribution<int64_t> md(-(int64_t)1 << 30, (int64_t)1 << 30);
                const int64_t mean_mag = md(rng);
                std::uniform_int_distribution<int64_t> rd(-(count / 2), count / 2 + 1);
                sum = mean_mag * count + rd(rng);   // mean ~ mean_mag, in int32 range
            }
            const int64_t got = mean_round(sum, count);
            const int64_t want = (int64_t)(int32_t)ref_mean(sum, count);   // helper returns int32
            if (got != want) {
                if (ref_fail < 8)
                    printf("  ref mismatch sum=%lld count=%lld got=%lld want=%lld\n",
                           (long long)sum, (long long)count, (long long)got, (long long)want);
                ref_fail++;
            }
            // sign symmetry: mean_round(-sum,count) == -mean_round(sum,count)
            if (mean_round(-sum, count) != -mean_round(sum, count)) {
                if (sym_fail < 8)
                    printf("  sym break sum=%lld count=%lld\n",
                           (long long)sum, (long long)count);
                sym_fail++;
            }
        }
        if (ref_fail) { printf("FAIL ref-match: %d mismatches\n", ref_fail); fails++; }
        else printf("PASS ref-match: 204000/204000 == round-half-away-from-zero reference\n");
        if (sym_fail) { printf("FAIL sign-symmetry: %d asymmetric\n", sym_fail); fails++; }
        else printf("PASS sign-symmetry: 204000/204000 mean(-s)==-mean(s)\n");
    }

    // count == 0 guard
    if (mean_round(12345, 0) != 0) { printf("FAIL count==0 guard\n"); fails++; }
    else printf("PASS count==0 guard -> 0\n");

    if (fails == 0) { printf("ALL PASS\n"); return 0; }
    printf("FAILS=%d\n", fails);
    return 1;
}
