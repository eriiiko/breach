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

// ---- host/device portability (CUDA-S0) -------------------------------------
// FP_HD marks a helper callable from BOTH a CPU .cpp TU and a CUDA __device__
// kernel. Under nvcc (__CUDACC__ is defined when compiling a .cu) it expands to
// `__host__ __device__`; under a plain C++ host compiler (cl.exe on the .cpp
// TUs, even in the CUDA build) it expands to NOTHING — so the CPU build is
// byte-for-byte unchanged. Only the PURE-INTEGER helpers (plus the double
// boundary casts) are FP_HD. recip_mul is now FP_HD too (CUDA-S3 water promoted
// it — the water velocity kick's central-difference gradient calls it on the
// device): under nvcc the __SIZEOF_INT128__ branch compiles for BOTH host and
// device, and the MSVC _mul128 branch is host-only and never reached on device
// (the sanctioned G3/G4 promotion). smoke_cliff_count stays HOST-ONLY for now:
// its 128-bit path uses MSVC _umul128 host intrinsics and only the substep-count
// cliff (a host-side derivation) needs it. Nothing FP_HD calls smoke_cliff_count.
#if defined(__CUDACC__)
  #define FP_HD __host__ __device__
#else
  #define FP_HD
#endif

// Q16.16 type alias. A plain int32, scaled by 2^16. (Alias, not a wrapper class:
// the solver loops want raw int arithmetic and the pybind boundary hands us
// int32 buffers directly — a wrapper would fight both.)
using q16 = int32_t;

namespace fixedpoint {

// The scale: one real unit == FP_ONE raw counts. Shares raycaster.h's HEAT_SCALE.
constexpr int32_t FP_SHIFT = 16;
constexpr int32_t FP_ONE   = 1 << FP_SHIFT;   // 65536
constexpr int64_t FP_ONE64 = (int64_t)1 << FP_SHIFT;

// Format invariants, compile-checked on BOTH host and device (the CUDA-S0 gate
// "toolkit static_asserts pass on device"). These are not tunables — they ARE
// the Q16.16 format — so baking them in is correct. The SAR check pins the
// load-bearing assumption that `>>` on a negative signed int is ARITHMETIC
// (round-toward -inf), which the whole truncation contract relies on and C++20
// guarantees; if a toolchain ever broke it, this fails to compile rather than
// silently desyncing.
static_assert(FP_SHIFT == 16, "Q16.16 format invariant: shift is 16");
static_assert(FP_ONE == 65536, "Q16.16 scale invariant: one unit == 65536 counts");
static_assert(FP_ONE64 == 65536LL, "Q16.16 64-bit scale invariant");
static_assert(sizeof(q16) == 4, "q16 must be a 32-bit integer");
static_assert(((q16)-4 >> 1) == (q16)-2, "signed >> must be arithmetic (SAR)");

// ---- quantize / dequantize (boundary casts) -------------------------------
//
// quantize: a real value -> Q16.16, ROUND TO NEAREST, computed in double so the
// product is exact for any in-range input (double has 52 mantissa bits >> the
// 31 we need). Round-half-away-from-zero via +/-0.5 (matches heat_quantize for
// non-negative inputs; symmetric for negatives). NO clamping here — water depths
// and velocities are far inside the +/-32768 range; the caller owns range safety.
FP_HD inline q16 quantize(double v) {
    double scaled = v * (double)FP_ONE;
    return (q16)(scaled >= 0.0 ? (scaled + 0.5) : (scaled - 0.5));
}

// dequantize: Q16.16 -> double real value (exact; the renderer + float bridges
// read through this).
FP_HD inline double dequantize(q16 v) {
    return (double)v / (double)FP_ONE;
}

// dequantize to float (the render/bridge boundary often wants float32 to match
// the still-float atmosphere/smoke fields exactly).
FP_HD inline float dequantize_f(q16 v) {
    return (float)((double)v / (double)FP_ONE);
}

// ---- the core multiply ----------------------------------------------------
//
// mul_q16(a, b) == round_toward_neg_inf(a*b / 2^16), the Q16.16 product. The
// int64 intermediate cannot overflow for any int32 inputs (|a*b| < 2^62). The
// arithmetic >> 16 narrows back to Q16.16, TRUNCATING (see the rounding note).
FP_HD inline q16 mul_q16(q16 a, q16 b) {
    return (q16)(((int64_t)a * (int64_t)b) >> FP_SHIFT);
}

// The full int64 product BEFORE the narrow — used by the conservative flux
// gather (S1 §2 / P2): the face flux is gathered ONCE as this int64, applied
// +/- to the two cells, and only narrowed at the divergence apply so the round
// is shared. Keeping the wide value lets the divergence sum carry full precision.
FP_HD inline int64_t mul_wide(q16 a, q16 b) {
    return (int64_t)a * (int64_t)b;
}

// Narrow a wide int64 Q(32).(32)-ish product (an accumulated sum of mul_wide
// terms, still carrying the 2^16 scale-squared) back to Q16.16 by >> 16. One
// shared truncation point.
FP_HD inline q16 narrow(int64_t wide) {
    return (q16)(wide >> FP_SHIFT);
}

// ROUND-TO-NEAREST narrow (the UNBIASED-DEPOSIT sibling of narrow()): for a
// non-negative wide product, +0.5 ULP then >>16 (round-half-up, symmetric for the
// non-negative case). Use for SOURCE deposits (S2a/S2c lesson: a non-conserved
// deposit wants round-to-nearest so a long run accumulates no DC truncation bias —
// the cancelling-flux PAIRS keep the plain truncating narrow()). The caller asserts
// `wide >= 0`; for a possibly-signed wide use narrow_round_signed.
FP_HD inline q16 narrow_round(int64_t wide) {
    const int64_t HALF = (int64_t)1 << (FP_SHIFT - 1);   // 0.5 ULP
    return (q16)((wide + HALF) >> FP_SHIFT);
}

// Sign-symmetric round-to-nearest narrow (round-half-away-from-zero) for a wide
// product of EITHER sign — rounds |wide| then re-applies the sign, so + and - are
// treated identically (no sign(wide) DC bias). The deposit analogue of shr_round0.
FP_HD inline q16 narrow_round_signed(int64_t wide) {
    return (wide >= 0) ? narrow_round(wide) : (q16)(-narrow_round(-wide));
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
FP_HD inline q16 recip_mul(q16 x_q16, int64_t recip) {
    __int128 prod = (__int128)x_q16 * (__int128)recip;
    return (q16)(prod >> RECIP_SHIFT);
}
#elif defined(_MSC_VER)
} // namespace fixedpoint
#include <intrin.h>
namespace fixedpoint {
FP_HD inline q16 recip_mul(q16 x_q16, int64_t recip) {
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
FP_HD inline q16 recip_mul(q16 x_q16, int64_t recip) {
    // Portable fallback: double-precision multiply (deterministic for the
    // single-divisor constants we use; only hit on exotic toolchains).
    return (q16)(((__int128_t)x_q16 * recip) >> RECIP_SHIFT);
}
#endif

// ---- symmetric round-toward-0 shift ---------------------------------------
// `x >> s` rounds toward -inf for negative x. For a symmetric divide-by-2^s
// (so +x and -x lose magnitude equally) use this (temperature_solver's cooling
// idiom). Deterministic, identical cross-machine.
FP_HD inline q16 shr_round0(q16 x, int s) {
    return (x < 0) ? -((-x) >> s) : (x >> s);
}

// ---- per-cell integer reciprocal (Newton-Raphson, GPU-clean) --------------
//
// reciprocal_q16(denom_q) == round_toward_neg_inf(2^16 / denom_real) in Q16.16,
// computed with PURE INTEGER arithmetic (no float, no divide on the hot path —
// only the seed's power-of-2 shift and three Newton refinements). This is the
// S2 per-cell divide primitive: a divisor that is BOTH runtime-derived AND
// different per cell, so make_recip (double-at-load for one fixed divisor) does
// not apply. Two call sites share it:
//   * S2b — the integer-SL bilinear renorm 1/wsum (wsum in (0,1], a sealed-
//     corner partial weight sum). This is the prototype's `_q_recip`
//     (tools/s2_advection_demo/advection_demo.py) ported verbatim, then
//     GENERALISED below to any positive Q16.16 denominator.
//   * S2c — the Gauss-Seidel per-cell divisor Dinv = 1/(1 + mu*wsum) (denom > 1).
//     S2c drops in here unchanged; this is why the seed handles denom > 1 too.
//
// THE METHOD — Newton-Raphson for 1/d:  r <- r*(2 - d*r), which doubles the
// number of correct bits each step and converges quadratically for any seed in
// (0, 2/d). We seed from a power-of-2 reciprocal keyed on the bit length of
// denom_q (a rough 2^-floor(log2 d)), which is always within the convergence
// basin, then iterate 3 times.
//
// SEED (the generalisation past the prototype). denom_q is Q16.16, so
//   denom_real ~= 2^(bitlen - 16),  and  1/denom_real ~= 2^(16 - bitlen).
// The Q16.16 reciprocal r = (1/denom_real) * 2^16 ~= 2^(32 - bitlen). So
// shift = 32 - bitlen and r0 = 1 << shift when shift >= 0. For denom_real > 1
// (bitlen > 32, e.g. the GS Dinv at denom ~= 9) shift goes NEGATIVE -> r0 is a
// right shift, r0 = 1 >> (-shift), which floors to 0 for a small reciprocal; we
// clamp the seed to >= 1 (one count) so the first Newton step still climbs. The
// prototype only ever saw wsum <= 1 (shift >= 0, r0 >= 1); the >> branch + the
// 1-count floor is the only addition, and it leaves the wsum<=1 path identical.
//
// ROUNDING (documented, load-bearing): every multiply is mul_q16 (the SAME >>16
// truncation toward -inf as the rest of the kit), so the result is a *truncated*
// Q16.16 reciprocal — reproducible bit-for-bit on every peer. It is NOT
// correctly-rounded (a Newton fixed-iteration cannot be), but it is DETERMINISTIC
// and converges to within ~1 ULP of 2^16/denom over the tested range. The
// caller must clamp denom_q to a positive FLOOR before calling (a zero or
// negative denom is undefined); reciprocal_q16 self-guards denom_q <= 0 by
// returning 0, and floors denom_q to 3 for the {1,2} case (where the true
// reciprocal 2^32/denom_q is >= 2^31 and would wrap to negative int32 garbage).
// Both guards are DEAD on the real call sites (SL renorm: wsum >= WSUM_FLOOR_Q
// = 256; GS Dinv: denom_q = 1 + mu*wsum in Q16.16 >= FP_ONE = 65536) — they
// exist so the SHARED helper is honestly safe for any future caller, not relying
// on every caller's floor being right.
FP_HD inline q16 reciprocal_q16(q16 denom_q) {
    if (denom_q <= 0) return 0;          // caller clamps to a floor; self-guard
    if (denom_q < 3) denom_q = 3;        // {1,2}: 2^32/denom_q overflows int32 -> floor (dead path)
    // Seed r0 ~= 2^(32 - bitlen(denom_q)), the power-of-2 reciprocal.
    int bitlen = 0;
    for (uint32_t v = (uint32_t)denom_q; v; v >>= 1) ++bitlen;   // bit_length
    // A power-of-2 seed alone is up to 2x too small (worst case: denom_q just
    // above a power of two), giving only ~1 correct bit -> 3 Newton iters land
    // at ~8 bits (rel error ~4e-3, demonstrably too loose for a reusable helper
    // whose accuracy feeds the GS Dinv's convergence). Refine the seed to the
    // TOP TWO bits: within the binade denom_q = 2^(bitlen-1)*(1+f), f in [0,1);
    // the 2nd bit (bit bitlen-2) splits the binade in half. A seed of
    // 1.5 * 2^(32-bitlen) for the LOWER half (f<0.5) and 1.0 * 2^(32-bitlen) for
    // the UPPER half (f>=0.5) halves the worst-case seed error to ~1.4x (~2 bits),
    // and 4 Newton iterations (~2 bits doubling to 4,8,16,32) then clear the
    // whole positive range to << 1 ULP. Pure integer + a single conditional —
    // deterministic, GPU-clean.
    const int shift = 32 - bitlen;
    int64_t r;
    if (shift >= 1) {
        const int64_t base = (int64_t)1 << shift;            // 2^(32-bitlen)
        // 2nd-highest bit set -> upper half of the binade (f>=0.5) -> seed ~1.0x;
        // clear -> lower half -> seed ~1.5x (base + base/2).
        const bool upper_half = (bitlen >= 2) &&
            ((denom_q >> (bitlen - 2)) & 1);
        r = upper_half ? base : (base + (base >> 1));
    } else {
        // denom_real > 1 (bitlen >= 32): the reciprocal is < 1 count at the
        // power-of-2 seed; floor to 1 count and let Newton climb down/refine.
        r = (shift >= 0) ? ((int64_t)1 << shift) : 1;
        if (r < 1) r = 1;
    }
    const int64_t two_q = (int64_t)FP_ONE << 1;            // 2.0 in Q16.16
    const int64_t HALF = (int64_t)1 << (FP_SHIFT - 1);     // 0.5 ULP rounding bias
    for (int it = 0; it < 4; ++it) {     // 4 Newton iterations: r <- r*(2 - d*r)
        // ROUND-TO-NEAREST narrow (+0.5 ULP before >>16) instead of truncate.
        // Newton's `r*(2 - d*r)` converges to 1/d FROM BELOW under a truncating
        // shift, so a plain >>16 leaves the result 1 ULP low at the exact points
        // (e.g. reciprocal_q16(FP_ONE) = 65535, not 65536) — which would make the
        // SL renorm at wsum==1.0 shave 1 ULP off EVERY cell every step (a uniform
        // decay the float SL does not have, on top of the intended truncation
        // decay). Round-to-nearest here lands the reciprocal on the true value at
        // the exact points, so wsum==1.0 is a clean identity; the only remaining
        // non-conservation is the deliberate sample-truncation decay. Both
        // operands are positive, so +HALF then >>16 is symmetric + deterministic.
        const int64_t dr  = ((int64_t)denom_q * r + HALF) >> FP_SHIFT;   // d*r (-> ~1.0)
        const int64_t cor = (r * (two_q - dr) + HALF) >> FP_SHIFT;       // r*(2 - d*r)
        r = cor;
    }
    return (q16)r;
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
FP_HD inline q16 scale_mag(q16 x, q16 scale) {
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
FP_HD inline int32_t ceil_div(q16 a_q, q16 b_q) {
    if (a_q <= 0) return 0;
    return (int32_t)(((int64_t)a_q + b_q - 1) / b_q);
}

// ---- the SMOKE CFL substep-count cliff (Bedrock cliff-patch) ---------------
//
// n_smoke = ceil( 4 * sim_time * d_smoke_max * (1 + wind_diffusion_scale * |wind|^2_max) ),
// the forward-Euler smoke-diffusion stability floor (physics_engine.cpp), made
// INTEGER so the substep COUNT is bit-identical across machines/GPUs (the last
// double on the determinism-critical substep-count path — completes Bedrock).
//
// Unlike ceil_div, the RESULT here is NOT a Q16.16 value — it is a plain COUNT
// that can reach the thousands under a shockwave (the run measured ~3600 at a
// strong blast). So we cannot stage it through a Q16.16 intermediate (which would
// overflow the +/-32768 range). Instead we build one exact rational and take an
// integer ceil:
//
//   R = 4*sim_time*d_smoke_max * (1 + wds*mws)
//     = (c4st_q/2^16) * (dsmoke_q/2^16) * ( (2^48 + wds_q*mws_q32) / 2^48 )
//     = base_q32 * wmult_q48 / 2^80,      base_q32 = c4st_q*dsmoke_q  (scale 2^32)
//                                         wmult_q48 = 2^48 + wds_q*mws_q32 (scale 2^48)
//   n_smoke = ceil(R) = (base_q32*wmult_q48 + 2^80 - 1) >> 80.
//
// INPUTS (all quantized at LOAD / once-per-tick boundaries — the LOCKED S1 idiom,
// "a load-time double->quantize is free / correctly-rounded"; no per-cell float):
//   c4st_q   = quantize(4*sim_time)          Q16.16   (sim_time is the tick length)
//   dsmoke_q = quantize(d_smoke_max)         Q16.16   (per-tick MAX over the gas table)
//   wds_q    = quantize(wind_diffusion_scale) Q16.16
//   mws_q32  = the INTEGER Q.32 spatial-max of |wind|^2 over the grid (an int64
//              order-free max reduction the engine already computes from the
//              Q16.16 wind components via mul_wide — kept verbatim).
//
// OVERFLOW (why 128-bit). wds_q*mws_q32 overflows int64 even at modest wind
// (peak |wind|~44 -> ~5.4e19 > 9.2e18 = int64 max), so wmult_q48 and the product
// genuinely need 128 bits. At the format-max wind component (32767) the product
// reaches ~112 bits — still inside UNSIGNED 128-bit (127 bits). Every quantity is
// non-negative, so we use UNSIGNED 128-bit throughout (mirrors recip_mul's MSVC
// _mul128 dual-path, but unsigned). The SAME two truncations (the >>80) on every
// toolchain -> cross-toolchain bit-identity (S1 Lesson #4).
//
// SATURATION. A defensive cap (SMOKE_N_CAP) bounds an absurd-input n_smoke so the
// substep loop can never be handed a runaway count (e.g. a corrupt/extreme wind):
// it is DEAD in real play (the measured peak ~3600 << the cap) and matches the
// honest shared-helper self-guard idiom (sqrt_q16/reciprocal_q16). The double
// version had no cap, but the cap only changes behaviour at counts so large the
// physics is already sub-stepped to a standstill either way.
constexpr int32_t SMOKE_N_CAP = 1 << 20;   // 1,048,576: absurd-wind floor guard

#if defined(__SIZEOF_INT128__)
inline int32_t smoke_cliff_count(q16 c4st_q, q16 dsmoke_q, q16 wds_q,
                                 int64_t mws_q32) {
    if (c4st_q <= 0 || dsmoke_q <= 0) return 1;   // no diffusion / no tick -> 1 step
    if (mws_q32 < 0) mws_q32 = 0;                  // |wind|^2 is non-negative
    const unsigned __int128 base_q32 =
        (unsigned __int128)(uint64_t)((int64_t)c4st_q * (int64_t)dsmoke_q);   // 2^32
    const unsigned __int128 wmult_q48 =
        ((unsigned __int128)1 << 48)
        + (unsigned __int128)(uint64_t)((int64_t)wds_q * mws_q32);            // 2^48
    const unsigned __int128 prod = base_q32 * wmult_q48;                       // 2^80
    const unsigned __int128 one80 = (unsigned __int128)1 << 80;
    unsigned __int128 n = (prod + (one80 - 1)) >> 80;                          // ceil
    if (n < 1) n = 1;
    if (n > (unsigned __int128)SMOKE_N_CAP) return SMOKE_N_CAP;
    return (int32_t)n;
}
#elif defined(_MSC_VER)
// (<intrin.h> is already included above for recip_mul's _mul128 path.)
inline int32_t smoke_cliff_count(q16 c4st_q, q16 dsmoke_q, q16 wds_q,
                                 int64_t mws_q32) {
    if (c4st_q <= 0 || dsmoke_q <= 0) return 1;
    if (mws_q32 < 0) mws_q32 = 0;
    // base_q32 = c4st_q*dsmoke_q (< 2^36) and the wind term wds_q*mws_q32 each fit
    // a 64-bit value? NO — wds_q*mws_q32 overflows 64 bits. Build wmult_q48 as a
    // 128-bit {hi,lo} via _umul128, add 2^48 into the low word (carry into hi),
    // then multiply by the 64-bit base_q32 (a 64x128 product) and >>80 the result.
    const uint64_t base_q32 = (uint64_t)((int64_t)c4st_q * (int64_t)dsmoke_q); // < 2^36
    // wind term = wds_q * mws_q32 as a 128-bit unsigned {wm_hi:wm_lo}.
    uint64_t wm_hi;
    uint64_t wm_lo = _umul128((uint64_t)(int64_t)wds_q,
                              (uint64_t)mws_q32, &wm_hi);
    // wmult_q48 = (wm_hi:wm_lo) + 2^48  (2^48 fits the low word; add with carry).
    uint64_t wmult_lo = wm_lo + ((uint64_t)1 << 48);
    uint64_t wmult_hi = wm_hi + (wmult_lo < wm_lo ? 1u : 0u);   // propagate carry
    // prod = base_q32 * (wmult_hi:wmult_lo). base_q32 is < 2^36, so:
    //   prod = base_q32*wmult_lo (128-bit) + (base_q32*wmult_hi << 64).
    uint64_t p_hi;
    uint64_t p_lo = _umul128(base_q32, wmult_lo, &p_hi);        // low 128 bits
    p_hi += base_q32 * wmult_hi;                                // add the hi*2^64 part
    // ceil(prod / 2^80): add (2^80 - 1) then >>80. 2^80 spans into the high word
    // (bit 80 == bit 16 of p_hi). Add the rounding bias 2^80-1 = (low 80 bits all
    // set): low 64 bits all set + bits 64..79 of the high word set.
    //   bias_lo = 0xFFFFFFFFFFFFFFFF (2^64 - 1); bias_hi = 2^16 - 1 (bits 64..79).
    const uint64_t add_lo = 0xFFFFFFFFFFFFFFFFULL;
    const uint64_t add_hi = ((uint64_t)1 << 16) - 1;
    uint64_t r_lo = p_lo + add_lo;
    uint64_t carry = (r_lo < p_lo) ? 1u : 0u;
    uint64_t r_hi = p_hi + add_hi + carry;
    // >>80: drop the low 64 bits, then >>16 the high word.
    uint64_t n = r_hi >> 16;
    if (n < 1) n = 1;
    if (n > (uint64_t)SMOKE_N_CAP) return SMOKE_N_CAP;
    return (int32_t)n;
}
#else
inline int32_t smoke_cliff_count(q16 c4st_q, q16 dsmoke_q, q16 wds_q,
                                 int64_t mws_q32) {
    // Portable INTEGER fallback (exotic toolchains with neither __int128 nor
    // _umul128). 64-bit-only: split wmult_q48 = 2^48 + wds_q*mws_q32 into a
    // staged product so no single multiply exceeds 64 bits. This loses the low
    // bits of the wind term (a deterministic, monotone-up rounding) but stays
    // integer + machine-reproducible; the __int128 / _umul128 paths above are the
    // exact cross-machine contract on every real target.
    if (c4st_q <= 0 || dsmoke_q <= 0) return 1;
    if (mws_q32 < 0) mws_q32 = 0;
    const uint64_t base_q32 = (uint64_t)((int64_t)c4st_q * (int64_t)dsmoke_q);
    // n = ceil(base_q32 * wmult_q48 / 2^80). Compute base*one (the +1 part) and
    // base*wind separately, each ceil'd up, then sum (an over-estimate by <=1,
    // safe — more substeps is always CFL-stable):
    //   part_const = ceil(base_q32 / 2^32)                       (the "1")
    //   part_wind  = ceil(base_q32 * (wds_q>>shrink) * (mws_q32>>shrink2) / 2^80)
    const uint64_t TWO32 = (uint64_t)1 << 32;
    uint64_t part_const = (base_q32 + (TWO32 - 1)) >> 32;
    // wind term scale 2^80: base_q32(2^32) * wds_q(2^16) * mws_q32(2^32) / 2^80.
    // Reduce mws_q32 to Q.16 first (>>16, monotone-down) to keep products in 64b.
    const uint64_t mws_q16 = (uint64_t)mws_q32 >> 16;            // scale 2^16
    // base_q32 * wds_q  -> scale 2^48, may exceed 64b for huge base; base<2^36,
    // wds_q<2^31 -> <2^67. Narrow base to 2^16 first (>>16): bd16 = base_q32>>16.
    const uint64_t bd16 = base_q32 >> 16;                        // scale 2^16
    // now bd16(2^16)*wds_q(2^16)*mws_q16(2^16) / 2^48 == wind term (scale 2^0).
    // bd16<2^20, wds_q<2^31 -> <2^51; *mws_q16(<2^31) -> <2^82 — still overflow.
    // Two-stage: m1 = ceil(bd16*wds_q / 2^16); then ceil(m1*mws_q16 / 2^32).
    uint64_t m1 = (bd16 * (uint64_t)(int64_t)wds_q + ((uint64_t)1<<16) - 1) >> 16;
    uint64_t part_wind = (m1 * mws_q16 + ((uint64_t)1<<32) - 1) >> 32;
    uint64_t n = part_const + part_wind;
    if (n < 1) n = 1;
    if (n > (uint64_t)SMOKE_N_CAP) return SMOKE_N_CAP;
    return (int32_t)n;
}
#endif

// ---- deterministic integer mean reduction (mean_wp, S2a) ------------------
//
// The wave->atmosphere transfer subtracts mean(wave_p) so the deposit is
// DC-free (mass-neutral) — the atmosphere is the conserved field, and a biased
// mean leaks a DC drift into EVERY cell (S2a P2 hazard, map §7.1 / plan §6.6).
//
// The reduction is split so the two determinism properties are each visible:
//   * mean_sum(values, mask, n)  — an int64 sum over a boolean mask. INTEGER
//     addition is associative + exact, so the sum is ORDER-FREE: identical on a
//     CPU scalar loop, a SIMD lane-tree, or a CUDA warp shuffle (spike-0a proved
//     this empirically vs the float atomicAdd that jittered). No `<<16`: the
//     summands are ALREADY Q16.16, so their int64 sum is a Q16.16 quantity too.
//   * mean_round(sum, count)     — sum / count, ROUND-TO-NEAREST, sign-SYMMETRIC
//     (round-half-away-from-zero, the same convention as quantize()). The naive
//     `sum/count` truncates toward 0 and biases the mean by -sign(sum) (a DC
//     drift); we bias the dividend by +/- count/2 so + and - round identically:
//       mean = (sum >= 0) ? (sum + count/2) / count : (sum - count/2) / count
//     There is NO pre-shift — `sum` is Q16.16, `count` is a plain integer, so
//     `sum / count` is the Q16.16 mean directly (this is the M3 sharp edge: it
//     differs from the GS divide, which DOES pre-shift a Q16.16 by <<16 before
//     dividing by a Q16.16 divisor). count must be > 0 (the caller guards
//     count == 0 -> mean 0, no division).
FP_HD inline int64_t mean_sum(const q16* values, const bool* mask, int n) {
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
        if (mask[i]) sum += (int64_t)values[i];   // exact, order-free
    }
    return sum;
}

FP_HD inline q16 mean_round(int64_t sum, int64_t count) {
    if (count <= 0) return 0;
    // round-half-away-from-zero, sign-symmetric (no sign(sum) DC bias).
    const int64_t half = count / 2;
    const int64_t m = (sum >= 0) ? (sum + half) / count
                                 : (sum - half) / count;
    return (q16)m;
}

// ---- deterministic integer sqrt (the first per-cell transcendental, S3b) ---
//
// sqrt_q16(x_q32) == floor( sqrt(x_real) ) in Q16.16, where x_q32 is an int64
// carrying a Q.32 value (a sum of mul_wide(q16,q16) products — i.e. the SQUARE
// of a Q16.16 magnitude, scale 2^32). Used by fire for W = sqrt(wind_x²+wind_y²):
//
//     int64_t rad = mul_wide(wx, wx) + mul_wide(wy, wy);   // Q.32
//     q16     W   = sqrt_q16(rad);                          // Q16.16
//
// WHY a plain isqrt of the Q.32 value yields the Q16.16 result directly: if the
// real magnitude is m and wx,wy are Q16.16 (wx = m_x·2^16), then
//   rad = (m_x·2^16)² + (m_y·2^16)² = (m_x²+m_y²)·2^32 = m²·2^32.
//   isqrt(rad) = floor( sqrt(m²·2^32) ) = floor( m·2^16 )  == the Q16.16 of m.
// So sqrt(2^32) = 2^16 folds the scale exactly — NO rescale needed.
//
// THE METHOD — a FIXED-ITERATION binary digit-recurrence isqrt of a 64-bit
// unsigned radicand. Pure integer shifts/compares, BRANCH-IDENTICAL across all
// lanes/architectures (the count is fixed at 32 iterations — one per result bit,
// since isqrt of a 64-bit value has <= 32 bits). It returns floor(√), EXACT, with
// NO rounding-mode ambiguity, NO LUT, NO polynomial, NO libm — the cleanest
// possible transcendental, identical on CPU and any future __device__ port (the
// master plan §5.3 same-integer-algorithm contract). Floor truncates W toward 0
// by < 1 LSB (~1.5e-5) — a deterministic, perceptually-invisible bias on the
// gentle, non-conserved wind-fan/strip tuning terms (S3b Open Q1: floor ratified).
//
// The radicand is non-negative by construction (a sum of squares). A negative
// input (shouldn't happen) floors to 0 via the unsigned cast guard.
FP_HD inline q16 sqrt_q16(int64_t x_q32) {
    if (x_q32 <= 0) return 0;
    uint64_t x = (uint64_t)x_q32;
    // Binary digit-by-digit isqrt: `bit` walks the highest power-of-4 <= x down
    // to 1; `res` accumulates the result. 32 fixed iterations (bit from 1<<62 to
    // 1<<0 in steps of 4 == 31 squared-digit positions; start at the top even bit
    // so the loop is a constant 32 trips regardless of x — branch-identical).
    uint64_t res = 0;
    // Start `bit` at the highest even bit position (1 << 62). A FIXED 32-trip
    // loop (the 32 even bit positions 62,60,...,0) — no data-dependent trip count.
    uint64_t bit = (uint64_t)1 << 62;
    for (int k = 0; k < 32; ++k) {
        const uint64_t t = res + bit;
        res >>= 1;
        if (x >= t) {
            x -= t;
            res += bit;
        }
        bit >>= 2;
    }
    // SELF-GUARD (honest shared-helper safety, like reciprocal_q16's floor): the
    // floor(√) of a Q.32 radicand >= 2^62 is >= 2^31, which would WRAP a signed
    // int32 to negative garbage. Clamp to INT32_MAX. This is DEAD on the real fire
    // call site (|wind| is gradient-scale O(1), spiking to maybe ~100 under a
    // shockwave -> rad ~= 8.6e13 << 2^62, res ~= 6.5e6 << 2^31) — the master plan
    // §8.1 overflow bound — but it keeps sqrt_q16 safe for any future caller and
    // makes the value deterministic rather than UB at the extreme.
    if (res > (uint64_t)0x7fffffff) return (q16)0x7fffffff;
    return (q16)res;
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
FP_HD inline q16 tan_poly(q16 t) {
    const q16 t2 = mul_q16(t, t);
    const q16 t3 = mul_q16(t2, t);
    const q16 t5 = mul_q16(t3, t2);
    const q16 term3 = t3 / 3;        // exact integer divide on Q16.16
    const q16 term5 = (q16)((2 * (int64_t)t5) / 15);
    return t + term3 + term5;
}

} // namespace fixedpoint
