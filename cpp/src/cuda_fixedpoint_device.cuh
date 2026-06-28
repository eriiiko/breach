#pragma once
// ============================================================================
// CUDA-S4 — shared __device__ fixed-point helpers (the 128-bit + reciprocal kit)
// ============================================================================
//
// S3 (cuda_water.cu) introduced two file-local device helpers — a signed
// 128-bit product-shift (`mul128_shr_signed`) and the central-difference
// reciprocal multiply (`recip_mul_dev`). S4 (smoke) and S7 (atmosphere GS) need
// the SAME 128-bit primitive plus the per-cell Newton reciprocal
// `reciprocal_q16`, so this header EXTRACTS the shared device math into one place
// instead of copy-pasting it into every .cu.
//
// WHY a device header and not `fixedpoint::recip_mul` / `fixedpoint::reciprocal_q16`
// directly: under the MSVC-host nvcc on this box, __int128 / __SIZEOF_INT128__ are
// NOT defined for the DEVICE pass (a gcc/clang extension absent from the
// MSVC-targeting device compiler). So a device instantiation of the header's
// FP_HD `recip_mul`/`reciprocal_q16` would resolve to the host-only `_mul128`
// branch and fail to compile if ODR-used on the device. The helpers here use
// CUDA's __mul64hi intrinsic (for the 128-bit case) or pure int64 arithmetic (for
// the Newton reciprocal, which never needs more than 64 bits over the q16 range)
// — and are bit-identical to the host integer math by construction.
//
// Include this from a .cu only (it pulls in <cuda_runtime.h> for __mul64hi and
// uses __device__). The plain-C++ .cpp TUs never see it.

#include "fixed_point.h"   // q16, FP_ONE, FP_SHIFT, RECIP_SHIFT
#include <cuda_runtime.h>
#include <cstdint>

namespace breach_cuda {

// ---- device 128-bit signed product, arithmetic-shifted right ---------------
// mul128_shr_signed(a, b, S) == (int64_t)( (a*b) >> S ), 0 < S < 64, where a*b is
// the full SIGNED 128-bit product. lo = low 64 bits (mod 2^64), hi = signed high
// 64 bits (__mul64hi). The arithmetic 128-bit >>S re-combines them EXACTLY the way
// the MSVC host code (fixed_point.h's _mul128 paths in recip_mul / flux_to_dq)
// does: ((uint64_t)lo >> S) | ((uint64_t)hi << (64 - S)). Bit-identical to the CPU
// by construction (the same single arithmetic >>S of the full 128-bit product).
// See [[fixed_point_migration_lessons]] #10 + cuda_water.cu's original comment.
__device__ __forceinline__ int64_t mul128_shr_signed(int64_t a, int64_t b, int S) {
    const long long hi = __mul64hi((long long)a, (long long)b);   // signed hi 64
    const unsigned long long lo = (unsigned long long)((long long)a * (long long)b);
    const long long res = (long long)((lo >> S) |
                                      ((unsigned long long)hi << (64 - S)));
    return (int64_t)res;
}

// ---- recip_mul (the make_recip reciprocal multiply) ------------------------
// x_q16 (Q16.16) divided by the real divisor whose Q.RECIP_SHIFT reciprocal is
// `recip`. Result Q16.16. The header fixedpoint::recip_mul is FP_HD, but its
// device instantiation would resolve to the host-only _mul128 branch under
// MSVC-host nvcc — so we call this 128-bit-via-__mul64hi version on the device
// instead (RECIP_SHIFT == 32). Bit-identical to the host _mul128 path.
__device__ __forceinline__ q16 recip_mul_dev(q16 x_q16, int64_t recip) {
    return (q16)mul128_shr_signed((int64_t)x_q16, recip, fixedpoint::RECIP_SHIFT);
}

// ---- reciprocal_q16_dev — the per-cell Newton reciprocal -------------------
// A VERBATIM device port of fixedpoint::reciprocal_q16 (fixed_point.h:261), the
// integer Newton-Raphson reciprocal 1/denom in Q16.16 the smoke bilinear renorm
// (and the S7 GS Dinv) use. EVERY arithmetic step is identical to the host:
//   * the bit_length seed (power-of-2 + top-2-bit refinement), and
//   * the 4 Newton iterations  r <- round_to_nearest( r*(2 - round_to_nearest(d*r)) ).
//
// CRITICAL: the host code does NOT use any 128-bit intermediate here — both
// products `denom_q * r` and `r * (two_q - dr)` fit in a signed int64 over the
// ENTIRE positive q16 range (verified: |denom_q| < 2^31, r < 2^32, so
// denom_q*r < 2^63; and 2-d*r stays ~O(1) so r*(2-d*r) ~ r < 2^32). So this is a
// pure int64/int32 port — the device int64 ops are bit-identical to the host's by
// the integer-determinism contract (no float, no toolchain-dependent rounding).
// (Hence NO mul128_shr_signed swap is needed in the Newton body — the host has no
// _mul128 there to swap; the §0 spec's "swap any host _mul128/recip_mul 128-bit
// step" applies vacuously to reciprocal_q16, which has none.)
//
// Self-guards (denom<=0 -> 0; denom<3 -> floored to 3) are replicated verbatim;
// both are DEAD on the real smoke call site (wsum >= WSUM_FLOOR_Q = 256) but kept
// so the device helper matches the host helper byte-for-byte for ALL inputs.
__device__ __forceinline__ q16 reciprocal_q16_dev(q16 denom_q) {
    if (denom_q <= 0) return 0;          // caller clamps to a floor; self-guard
    if (denom_q < 3) denom_q = 3;        // {1,2}: 2^32/denom_q overflows int32 (dead)
    // Seed r0 ~= 2^(32 - bitlen(denom_q)), the power-of-2 reciprocal.
    int bitlen = 0;
    for (uint32_t v = (uint32_t)denom_q; v; v >>= 1) ++bitlen;   // bit_length
    const int shift = 32 - bitlen;
    int64_t r;
    if (shift >= 1) {
        const int64_t base = (int64_t)1 << shift;            // 2^(32-bitlen)
        const bool upper_half = (bitlen >= 2) &&
            ((denom_q >> (bitlen - 2)) & 1);
        r = upper_half ? base : (base + (base >> 1));
    } else {
        r = (shift >= 0) ? ((int64_t)1 << shift) : 1;
        if (r < 1) r = 1;
    }
    const int64_t two_q = (int64_t)fixedpoint::FP_ONE << 1;            // 2.0 Q16.16
    const int64_t HALF = (int64_t)1 << (fixedpoint::FP_SHIFT - 1);     // 0.5 ULP
    for (int it = 0; it < 4; ++it) {     // 4 Newton iterations: r <- r*(2 - d*r)
        const int64_t dr  = ((int64_t)denom_q * r + HALF) >> fixedpoint::FP_SHIFT;
        const int64_t cor = (r * (two_q - dr) + HALF) >> fixedpoint::FP_SHIFT;
        r = cor;
    }
    return (q16)r;
}

}  // namespace breach_cuda
