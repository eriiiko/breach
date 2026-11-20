#pragma once
// ==========================================================================
// #29 P8.1: THE DEVICE OBSERVATION'S FLOAT KIT (Class T). The observation's
// own header (never the unit library, cuda_unit_lib.cuh, which stays
// integer): the float64 arithmetic the host builders (obs_spec_v1) do, in
// THEIR operation order, and the ONE cast of each value into its served
// dtype.
//
// NO CONTRACTION. Every float64 +, *, / here is an explicit round-to-nearest
// intrinsic (__dadd_rn, __dmul_rn, __ddiv_rn): nvcc never fuses an intrinsic
// into an FMA, whatever the flags. The breach_physics target also compiles
// every .cu with --fmad=false (cpp/CMakeLists.txt), so a plain expression
// would not contract either; the intrinsics make the kit independent of that
// flag. Held by tests/cuda_obs_check.py part FLOATKIT (a*b + c on inputs where
// the fused form differs, against numpy's two roundings).
//
// THE SQUASH (obs_spec_v1.squash): the identity on [-1, 1], else
// sign(x) * min(SQUASH_CAP, 1 + log|x|) with CUDA's own double log (design
// "P8 RE-CUT, VERSION 2": no shared log module; the corpus gate compares the
// SERVED values as raw bits and fails on one flip).
//
// THE CASTS. float64 -> float32: CUDA's (float) is __double2float_rn, one
// correctly rounded step (nearest even, subnormals kept: -ftz=false), as
// numpy's astype(float32). float64 -> float16: a HAND conversion on the
// integer bits (critique finding 5: __double2half is a software path on the
// architectures below sm_90 and its rounding is not documented as one step),
// rounding ONCE to nearest even, with subnormals, overflow to inf, signed
// zero, and numpy's NaN rule (npy_doublebits_to_halfbits: the top mantissa
// bits kept, forced non-zero). Held EXHAUSTIVELY against numpy (part
// FLOATKIT: every float16 value, every midpoint and its float64 neighbours).
// ==========================================================================
#include <cstdint>

#include <cuda_runtime.h>

namespace breach_cuda {
namespace obs_f64 {

// -- the basic operations, never fused ---------------------------------------
__device__ __forceinline__ double add(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double sub(double a, double b) { return __dadd_rn(a, -b); }
__device__ __forceinline__ double mul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double div(double a, double b) { return __ddiv_rn(a, b); }

// A float64 carried as its own bit pattern (an obs-table row, a header bits
// constant): the value itself, never a re-quantization.
__device__ __forceinline__ double of_bits(int64_t bits) { return __longlong_as_double(bits); }

// obs_spec_v1.ratio: float(x) / float(d), 0 when the table value is 0. An
// int64 below 2^53 converts exactly (every observation operand is).
__device__ __forceinline__ double ratio(int64_t x, int64_t d) {
    return d == 0 ? 0.0 : div((double)x, (double)d);
}

// obs_spec_v1.squash, with the cap as its bit pattern.
__device__ __forceinline__ double squash(double x, double cap) {
    const double ax = fabs(x);
    if (ax <= 1.0) return x;
    double v = add(1.0, log(ax));
    v = v < cap ? v : cap;            // Python's min(cap, v): equal values are one value
    return x > 0.0 ? v : -v;
}

// -- the casts ---------------------------------------------------------------
// A NaN keeps numpy's (the C cast's, x86 cvtsd2ss) pattern: quieted, the top
// 22 payload bits kept, the sign kept; CUDA's cvt would give its canonical
// NaN. (No served value is a NaN; the rule keeps "raw bits" total.)
__device__ __forceinline__ float to_f32(double x) {
    if (isnan(x)) {
        const uint64_t b = (uint64_t)__double_as_longlong(x);
        const uint32_t r = (uint32_t)((b >> 32) & 0x80000000u) | 0x7fc00000u |
                           (uint32_t)((b & 0x000fffffffffffffULL) >> 29);
        return __uint_as_float(r);
    }
    return __double2float_rn(x);
}

// float64 -> float16 bits, rounded ONCE to nearest even (numpy's direct
// conversion, npy_doublebits_to_halfbits).
__device__ __forceinline__ uint16_t to_half_bits(double x) {
    const uint64_t b = (uint64_t)__double_as_longlong(x);
    const uint16_t sign = (uint16_t)((b >> 48) & 0x8000u);
    const uint64_t exp = (b >> 52) & 0x7ffu;
    const uint64_t man = b & 0x000fffffffffffffULL;
    if (exp == 0x7ffu) {                                   // inf / NaN
        if (man == 0) return (uint16_t)(sign | 0x7c00u);
        uint16_t r = (uint16_t)(0x7c00u + (man >> 42));
        if (r == 0x7c00u) ++r;                             // stays a NaN
        return (uint16_t)(sign | r);
    }
    const int e = (int)exp - 1023;                         // unbiased
    if (e >= 16) return (uint16_t)(sign | 0x7c00u);        // |x| >= 2^16: overflow
    if (e >= -14) {                                        // a normal half (or its carry)
        uint64_t m = man >> 42;
        const uint64_t rem = man & ((1ULL << 42) - 1);
        const uint64_t halfway = 1ULL << 41;
        if (rem > halfway || (rem == halfway && (m & 1u))) ++m;
        // the carry out of the mantissa moves the exponent (31 << 10 is inf)
        return (uint16_t)(sign | ((uint16_t)((e + 15) << 10) + (uint16_t)m));
    }
    if (exp == 0) return sign;                             // a float64 subnormal / zero
    // a half subnormal: m = sig * 2^(e - 52 + 24), sig the 53-bit significand
    const uint64_t sig = man | (1ULL << 52);
    const int shift = 28 - e;                              // >= 43
    if (shift > 63) return sign;                           // below half the smallest step
    uint64_t m = sig >> shift;
    const uint64_t rem = sig & ((1ULL << shift) - 1);
    const uint64_t halfway = 1ULL << (shift - 1);
    if (rem > halfway || (rem == halfway && (m & 1u))) ++m;
    return (uint16_t)(sign | (uint16_t)m);                 // m == 1024 is the smallest normal
}

}  // namespace obs_f64
}  // namespace breach_cuda
