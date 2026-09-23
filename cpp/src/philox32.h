// philox32.h -- VENDORED verbatim, do not hand-edit.
//
// Source: github.com/eriiiko/philox32 (MIT, Copyright (c) 2026 Erik Steen),
// file philox32/philox32.h, commit 6c33a3d0737e1f9f64b7dd84bdbda02fbbc863d2.
// SHA-256 of the vendored file at that commit: 46e9beadfb61c25142dda50722e1f1c2eef1ac7c4c67d59161bec5e0865d93b3
// To update: re-vendor from that commit (copy the file verbatim, refresh this
// header and its recorded hash) -- never hand-edit the body below the marker.
//
// Algorithm citation:
// John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw, "Parallel
// Random Numbers: As Easy as 1, 2, 3", Proceedings of the 2011 International
// Conference for High Performance Computing, Networking, Storage and
// Analysis (SC'11), DOI 10.1145/2063384.2063405.
//
// ---- vendored body (byte-for-byte from upstream) begins below ----
// philox32.h -- Philox-4x32-10, header-only, integer-only, host + CUDA/HIP device.
//
// Algorithm: John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw,
//   "Parallel random numbers: as easy as 1, 2, 3", Proceedings of SC'11,
//   DOI 10.1145/2063384.2063405.  Reference implementation: Random123
//   (https://github.com/DEShawResearch/random123, BSD-3-Clause).  This file is
//   an independent re-implementation gated against Random123's published
//   known-answer vectors (tests/kat_philox4x32_10.txt) and, where a GPU is
//   present, against NVIDIA cuRAND's curand_Philox4x32_10 (tests/curand_oracle.cu).
//
// License: MIT, Copyright (c) 2026 Erik Steen.  See LICENSE.
//
// WHAT
//   Philox is a counter-based, stateless, keyed RNG: out = F(ctr, key).  There
//   is no internal state to seed, advance or share -- the caller names the draw
//   it wants (by counter + key) and gets the same 128 bits on every platform.
//   Only uint32_t / uint64_t arithmetic is used (32x32->64 products via a
//   uint64_t multiply), so host and device agree bit for bit, and so does the
//   pure-Python twin in philox32.py.
//
// API (every function is PHILOX32_HD: callable from host and device code)
//   philox32_4x32_10(ctr[4], key[2], out[4])   the block function, 10 rounds
//   philox32_draw(seed_lo, seed_hi, agent_id, tick, draw_index, stream_salt, out[4])
//                                              the block function on the layout below
//   philox32_uniform_q16(w)  -> Q16.16 in [0, 1)        == w >> 16
//   philox32_angle_q16(w)    -> Q16.16 in [0, 2*pi)     == (w * TWO_PI_Q16) >> 32
//   philox32_below(w, n)     -> integer in [0, n)       == (w * n) >> 32
//   philox32_make_key / philox32_make_counter   the layout below, made executable
//
// RECOMMENDED KEY / COUNTER LAYOUT FOR AGENT SIMULATIONS
//   key     = (seed_lo, seed_hi)                        -- the seed and nothing else
//   counter = (tick, draw_index, stream_salt, agent_id)
//   - seed is the simulation's 64-bit seed, split into two words; it is the
//     whole key, so two different seeds never share a stream with any
//     (agent, tick, salt, index) combination -- a seed x agent sweep yields
//     distinct streams for every cell (gated by tests/test_layout.py).
//   - agent_id makes every agent an independent stream within a seed.
//   - stream_salt separates purposes within one agent and tick (movement vs
//     combat vs spawn...) so adding a new consumer never shifts the draws of
//     an existing one.  Keep the salts in ONE enum per consuming project.
//   - draw_index counts the blocks a consumer takes within one tick.  Each
//     block yields four 32-bit words -- use out[0..3] before bumping it.
//   The C++ helpers take uint32_t and therefore truncate silently; the Python
//   twin is the strict side and rejects anything outside [0, 2^32).
//
// DETERMINISM NOTES
//   - No floating point anywhere: the output mappings are integer shifts and
//     one 64-bit product each.  TWO_PI_Q16 = 411775 = round(2*pi * 65536) is a
//     checked-in constant, never computed at run time.
//   - Outputs are pure functions of (ctr, key).  Verified bit-identical, on the
//     machine named in VERIFIED.md, between MSVC 19.4x x64 host code, nvcc 12.4
//     device code (sm_86), CPython 3.11 and numpy 2.x, and against cuRAND's
//     independent implementation.  GCC and Clang are expected to agree -- the
//     header contains no implementation-defined construct (unsigned arithmetic,
//     explicit uint64_t widening, no signed shifts) -- and are gated by the CI
//     matrix in .github/workflows/ci.yml.
#ifndef PHILOX32_H
#define PHILOX32_H

#include <stdint.h>

// Host+device annotation.  Predefine PHILOX32_HD to use your own.
#ifndef PHILOX32_HD
#if defined(__CUDACC__) || defined(__HIPCC__)
#define PHILOX32_HD __host__ __device__
#else
#define PHILOX32_HD
#endif
#endif

// Round multipliers and Weyl key increments (Random123 PHILOX_M4x32_* / PHILOX_W32_*).
#define PHILOX32_M0 0xD2511F53u
#define PHILOX32_M1 0xCD9E8D57u
#define PHILOX32_W0 0x9E3779B9u
#define PHILOX32_W1 0xBB67AE85u
#define PHILOX32_ROUNDS 10

// round(2*pi * 65536): 2*pi in Q16.16.  Macro for preprocessor use, typed
// constant for arithmetic.
#define PHILOX32_TWO_PI_Q16 411775u
static const uint32_t philox32_two_pi_q16 = PHILOX32_TWO_PI_Q16;

// One Philox round (internal).  hi/lo of 32x32->64 products, then the permute + key xor.
PHILOX32_HD inline void philox32_detail_round(uint32_t ctr[4], const uint32_t key[2])
{
    const uint64_t p0 = (uint64_t)PHILOX32_M0 * (uint64_t)ctr[0];
    const uint64_t p1 = (uint64_t)PHILOX32_M1 * (uint64_t)ctr[2];
    const uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    const uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    const uint32_t c1 = ctr[1], c3 = ctr[3];
    ctr[0] = hi1 ^ c1 ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ c3 ^ key[1];
    ctr[3] = lo0;
}

// Philox-4x32-10: round 1 with the raw key, then (bump key, round) x 9.
PHILOX32_HD inline void philox32_4x32_10(const uint32_t ctr[4], const uint32_t key[2], uint32_t out[4])
{
    uint32_t c[4] = { ctr[0], ctr[1], ctr[2], ctr[3] };
    uint32_t k[2] = { key[0], key[1] };
    philox32_detail_round(c, k);
    for (int r = 1; r < PHILOX32_ROUNDS; ++r) {
        k[0] += PHILOX32_W0;   // uint32_t wrap-around is well defined
        k[1] += PHILOX32_W1;
        philox32_detail_round(c, k);
    }
    out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; out[3] = c[3];
}

// --------------------------------------------------------------- output mappings
// Uniform in [0, 1) as Q16.16 (65536 == 1.0): the top 16 bits of w.
// Range 0..65535.  Exactly uniform (each value owns 2^16 words), but that is
// also the resolution: 65536 attainable values.  A consumer needing finer
// resolution takes the raw 32-bit word.
PHILOX32_HD constexpr uint32_t philox32_uniform_q16(uint32_t w)
{
    return w >> 16;
}

// Uniform angle in [0, 2*pi) as Q16.16: floor(w * TWO_PI_Q16 / 2^32).
// Range 0..PHILOX32_TWO_PI_Q16-1.  Bias: 2^32 / 411775 = 10430.374, so each
// output value owns 10430 or 10431 words -- a maximum relative excess of 9.6e-5.
PHILOX32_HD constexpr uint32_t philox32_angle_q16(uint32_t w)
{
    return (uint32_t)(((uint64_t)w * (uint64_t)PHILOX32_TWO_PI_Q16) >> 32);
}

// Integer in [0, n): floor(w * n / 2^32) -- Lemire's multiply-shift, no
// division.  Range 0..n-1 (n == 0 gives 0).  Bias: values own floor or ceil of
// 2^32/n words, so the relative excess of the fattest bin is < n / 2^32
// (2.3e-9 for n = 10, 2.3e-4 for n = 1e6).  There is deliberately no
// rejection-sampling "exact" variant: rejection makes the NUMBER OF WORDS
// CONSUMED depend on the draw, so two consumers' draw_index bookkeeping would
// diverge on data -- a determinism hazard in a lockstep simulation.  If the
// bias matters, draw two words and reduce a 64-bit product, or keep n small.
PHILOX32_HD constexpr uint32_t philox32_below(uint32_t w, uint32_t n)
{
    return (uint32_t)(((uint64_t)w * (uint64_t)n) >> 32);
}

// ------------------------------------------------------------------ the layout
PHILOX32_HD inline void philox32_make_key(uint32_t seed_lo, uint32_t seed_hi, uint32_t key[2])
{
    key[0] = seed_lo;
    key[1] = seed_hi;
}

PHILOX32_HD inline void philox32_make_counter(uint32_t tick, uint32_t draw_index, uint32_t stream_salt,
                                              uint32_t agent_id, uint32_t ctr[4])
{
    ctr[0] = tick;
    ctr[1] = draw_index;
    ctr[2] = stream_salt;
    ctr[3] = agent_id;
}

// The block function on the recommended layout: what every call site would
// otherwise open-code.
PHILOX32_HD inline void philox32_draw(uint32_t seed_lo, uint32_t seed_hi, uint32_t agent_id,
                                      uint32_t tick, uint32_t draw_index, uint32_t stream_salt,
                                      uint32_t out[4])
{
    uint32_t key[2], ctr[4];
    philox32_make_key(seed_lo, seed_hi, key);
    philox32_make_counter(tick, draw_index, stream_salt, agent_id, ctr);
    philox32_4x32_10(ctr, key, out);
}

#endif // PHILOX32_H
