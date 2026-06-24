#pragma once
// ============================================================================
// Q16.16 fixed-point toolkit — the reusable integer-determinism kit.
// ============================================================================
//
// The fixed-point determinism arc (docs/fixed_point_migration_plan.md) replaces
// the synced float sim fields with int32 Q16.16, so the lockstep state is
// BIT-IDENTICAL across machines/GPUs (integer +/-/*/>> are exact and
// associative; IEEE float is not, cross-vendor). This header is the shared
// kit — S1 (water) is the first consumer; S2 (atmosphere/smoke) and S3 reuse it.
//
// FORMAT: Q16.16 == int32 with an implicit scale of 2^16 == 65536. The integer
// part is the top 16 bits, the fraction the bottom 16. One real unit == 65536
// raw counts. Range: [-32768, +32768) with a resolution of 1/65536 ~= 1.526e-5.
// (This matches raycaster.h's HEAT_SCALE/temperature domain exactly, so the
// water + heat fixed-point worlds share one scale.)
//
// ROUNDING CONVENTION (documented, load-bearing):
//   * mul_q16 TRUNCATES toward -inf via an arithmetic right shift `>> 16`, the
//     SAME idiom temperature_solver.cpp uses for its conduction-flux shift
//     (`acc += (int64)(tn - ti) >> s`). We do NOT round-to-nearest in mul_q16:
//     truncation is cheaper, branch-free, and — crucially for conservation —
//     when the SAME product is gathered once and applied as +flux to one cell
//     and -flux to its neighbour, the identical (truncated) int64 value cancels
//     exactly, so no mass is created or destroyed at the >>16 narrow (S1 P2).
//   * quantize() (float/double -> Q16.16) rounds to NEAREST (like
//     raycaster.h::heat_quantize), because that is a one-time load/boundary cast
//     where round-to-nearest minimises the quantization bias of a constant.
//   * The arithmetic right shift on a NEGATIVE int rounds toward -inf in C++20
//     (>> on signed is implementation-defined pre-C++20 but arithmetic on every
//     compiler we target: MSVC/clang/gcc all emit SAR). This is deterministic
//     and identical cross-machine. Where a SYMMETRIC round-toward-0 is wanted
//     (so +x and -x behave the same), use shr_round0().
//
// All helpers are header-only inline / constexpr so they fold into the /fp:strict
// water TU with zero call overhead and no float (these are pure integer ops).

#include <cstdint>

// Q16.16 type alias. A plain int32, scaled by 2^16. (Alias, not a wrapper class:
// the solver loops want raw int arithmetic and the pybind boundary hands us
// int32 buffers directly — a wrapper would fight both.)
using q16 = int32_t;

namespace fixedpoint {

// The scale: one real unit == FP_ONE raw counts. Shares raycaster.h's HEAT_SCALE.
constexpr int32_t FP_SHIFT = 16;
constexpr int32_t FP_ONE   = 1 << FP_SHIFT;   // 65536
constexpr int64_t FP_ONE64 = (int64_t)1 << FP_SHIFT;

// ---- quantize / dequantize (boundary casts) -------------------------------
//
// quantize: a real value -> Q16.16, ROUND TO NEAREST, computed in double so the
// product is exact for any in-range input (double has 52 mantissa bits >> the
// 31 we need). Round-half-away-from-zero via +/-0.5 (matches heat_quantize for
// non-negative inputs; symmetric for negatives). NO clamping here — water depths
// and velocities are far inside the +/-32768 range; the caller owns range safety.
inline q16 quantize(double v) {
    double scaled = v * (double)FP_ONE;
    return (q16)(scaled >= 0.0 ? (scaled + 0.5) : (scaled - 0.5));
}

// dequantize: Q16.16 -> double real value (exact; the renderer + float bridges
// read through this).
inline double dequantize(q16 v) {
    return (double)v / (double)FP_ONE;
}

// dequantize to float (the render/bridge boundary often wants float32 to match
// the still-float atmosphere/smoke fields exactly).
inline float dequantize_f(q16 v) {
    return (float)((double)v / (double)FP_ONE);
}

// ---- the core multiply ----------------------------------------------------
//
// mul_q16(a, b) == round_toward_neg_inf(a*b / 2^16), the Q16.16 product. The
// int64 intermediate cannot overflow for any int32 inputs (|a*b| < 2^62). The
// arithmetic >> 16 narrows back to Q16.16, TRUNCATING (see the rounding note).
inline q16 mul_q16(q16 a, q16 b) {
    return (q16)(((int64_t)a * (int64_t)b) >> FP_SHIFT);
}

// The full int64 product BEFORE the narrow — used by the conservative flux
// gather (S1 §2 / P2): the face flux is gathered ONCE as this int64, applied
// +/- to the two cells, and only narrowed at the divergence apply so the round
// is shared. Keeping the wide value lets the divergence sum carry full precision.
inline int64_t mul_wide(q16 a, q16 b) {
    return (int64_t)a * (int64_t)b;
}

// Narrow a wide int64 Q(32).(32)-ish product (an accumulated sum of mul_wide
// terms, still carrying the 2^16 scale-squared) back to Q16.16 by >> 16. One
// shared truncation point.
inline q16 narrow(int64_t wide) {
    return (q16)(wide >> FP_SHIFT);
}

// ---- precomputed reciprocal (divide -> multiply) --------------------------
//
// A divide by a runtime-known-but-loop-invariant value becomes a reciprocal
// multiply. We store the reciprocal as a Q.K wide fixed-point (K = RECIP_SHIFT
// fractional bits, wider than 16 so small divisors keep precision), computed
// ONCE in double at load (double divide is correctly-rounded -> deterministic
// for a given divisor). recip_mul then does x * recip >> K.
//
//   recip = round( 2^K / divisor_real )          (a wide int64)
//   x_q16 / divisor_real  ==  (x_q16 * recip) >> K   (Q16.16 result)
//
// K = 32 gives ~9 decimal digits of reciprocal precision — ample for the water
// constants (two_dx, dt_over_dx). The product x_q16 * recip is int64*int64 ->
// needs care: x_q16 < 2^31, recip < 2^(32 - log2(divisor)). For our divisors
// (>~ 1e-3) recip < 2^42, so the product < 2^73 — that OVERFLOWS int64. So
// recip_mul uses __int128 when available, else a 64-bit-safe path. For the
// water core the divisors are all >= ~0.03 (two_dx, dt_over_dx at sane dt/dx),
// keeping recip < 2^37 and the product < 2^68 — still > 64 bits. We therefore
// use a 128-bit intermediate (MSVC: _mul128 / __int128 on clang-cl & gcc).
constexpr int RECIP_SHIFT = 32;

inline int64_t make_recip(double divisor_real) {
    // 2^32 / divisor, rounded to nearest. divisor_real must be > 0.
    double r = (double)((int64_t)1 << RECIP_SHIFT) / divisor_real;
    return (int64_t)(r + 0.5);
}

// x_q16 (a Q16.16 value) divided by the real divisor whose reciprocal is recip.
// Result is Q16.16. Uses a 128-bit intermediate so x*recip never overflows.
#if defined(__SIZEOF_INT128__)
inline q16 recip_mul(q16 x_q16, int64_t recip) {
    __int128 prod = (__int128)x_q16 * (__int128)recip;
    return (q16)(prod >> RECIP_SHIFT);
}
#elif defined(_MSC_VER)
} // namespace fixedpoint
#include <intrin.h>
namespace fixedpoint {
inline q16 recip_mul(q16 x_q16, int64_t recip) {
    // 128-bit signed product via _mul128, then shift right by RECIP_SHIFT.
    long long hi;
    long long lo = _mul128((long long)x_q16, (long long)recip, &hi);
    // Arithmetic 128-bit >> RECIP_SHIFT (RECIP_SHIFT < 64): combine hi:lo.
    unsigned long long ulo = (unsigned long long)lo;
    long long res = (long long)((ulo >> RECIP_SHIFT) |
                                ((unsigned long long)hi << (64 - RECIP_SHIFT)));
    return (q16)res;
}
#else
inline q16 recip_mul(q16 x_q16, int64_t recip) {
    // Portable fallback: double-precision multiply (deterministic for the
    // single-divisor constants we use; only hit on exotic toolchains).
    return (q16)(((__int128_t)x_q16 * recip) >> RECIP_SHIFT);
}
#endif

// ---- symmetric round-toward-0 shift ---------------------------------------
// `x >> s` rounds toward -inf for negative x. For a symmetric divide-by-2^s
// (so +x and -x lose magnitude equally) use this (temperature_solver's cooling
// idiom). Deterministic, identical cross-machine.
inline q16 shr_round0(q16 x, int s) {
    return (x < 0) ? -((-x) >> s) : (x >> s);
}

// ---- magnitude-first Q16.16 scale (sign-preserving, shrink-only) -----------
// Multiply a Q16.16 value `x` by a Q16.16 factor `scale` in [0, FP_ONE] while
// truncating on the MAGNITUDE (toward 0), not toward -inf. mul_q16's `>>16`
// rounds a NEGATIVE product toward -inf, which GROWS a negative value's
// magnitude by up to 1 count. For an outflow limiter (scale <= 1.0) that would
// let a scaled-down outgoing face delta get *larger* in magnitude — the exact
// over-drain that breaks the "limiter bounds outflow to <= depth" guarantee.
// Mirroring shr_round0's idiom, we scale |x| then re-apply the sign, so the
// scaled magnitude can only SHRINK (== |x|*scale truncated toward 0). For
// scale == FP_ONE (1.0) this is the identity. The same scaled value is then
// applied +/- symmetrically to a face's two cells -> conservation is preserved.
inline q16 scale_mag(q16 x, q16 scale) {
    const int64_t mag = (int64_t)(x < 0 ? -x : x);
    const q16 scaled  = (q16)((mag * (int64_t)scale) >> FP_SHIFT);  // toward 0
    return (x < 0) ? -scaled : scaled;
}

// ---- fixed-point ceil-divide (the substep-count cliff) --------------------
//
// n = ceil(a_real / b_real), computed entirely in integer from the Q16.16
// operands a_q, b_q (same scale cancels): ceil(a_q / b_q). A 1-ULP float slip
// here flips n and desyncs the whole tick (physics_engine.cpp:286-288), so the
// integer ceil is the cross-GPU fix. b_q must be > 0 (a CFL bound is positive).
//   ceil(a/b) for positive a,b == (a + b - 1) / b   (integer divide).
// We guard a < 0 (shouldn't happen — sim_time >= 0) by flooring to 0.
inline int32_t ceil_div(q16 a_q, q16 b_q) {
    if (a_q <= 0) return 0;
    return (int32_t)(((int64_t)a_q + b_q - 1) / b_q);
}

// ---- tan(theta) via a low-degree odd polynomial ---------------------------
//
// The ship-tilt term needs tan(tilt_x), tan(tilt_y) ONCE per tick (scalar, not
// per-cell). A short odd Taylor series is bit-exact-reproducible (pure integer
// multiplies) and accurate enough over the clamped tilt range:
//   tan(t) ~= t + t^3/3 + 2 t^5/15        (degree 5, 3 terms)
// MEASURED relative error (vs std::tan): ~1e-4 up to 20deg, 4.2e-4 at 25deg,
// 1.2e-3 at 30deg, 2.9e-3 at the 35deg clamp edge. So it is < 0.1% over the
// |tilt| <= 25deg core range and ~0.1-0.3% out at the clamp boundary — well
// within "perceptually irrelevant" for a tilt that only NUDGES the surface
// potential (it is NOT a conserved flux; the worst-case 0.3% is 0.002 of tan).
// If a tighter bound is ever wanted, add the degree-7 term (17 t^7 / 315). The
// caller MUST clamp t to a sane max (+/-35deg) first so the series stays in its
// accurate regime — past that, tan blows up and a low-degree poly diverges.
//
//   t3 = t^3 = mul(mul(t,t), t);  term3 = t3 / 3
//   t5 = t3 * t^2;                term5 = (2 * t5) / 15
//   tan ~= t + term3 + term5
// Divides by the small odd constants 3, 15 are exact integer divides on the
// Q16.16 value (deterministic). Done with mul_q16 throughout (truncating; the
// scalar magnitude << conservation concerns — this term only tilts the surface
// potential, it is not a conserved flux).
inline q16 tan_poly(q16 t) {
    const q16 t2 = mul_q16(t, t);
    const q16 t3 = mul_q16(t2, t);
    const q16 t5 = mul_q16(t3, t2);
    const q16 term3 = t3 / 3;        // exact integer divide on Q16.16
    const q16 term5 = (q16)((2 * (int64_t)t5) / 15);
    return t + term3 + term5;
}

} // namespace fixedpoint
