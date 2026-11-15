#pragma once
// ============================================================================
// #29 P7.0a -- THE DEVICE DRAWS (residency design 4.4c decision 5): the device
// forms of simulation.draws -- MatchDraws.at / .next, cone_offset_q16 and
// roll_q16 -- over the vendored Philox-4x32-10 (philox32.h; Salmon, Moraes,
// Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11, DOI
// 10.1145/2063384.2063405). Integer for integer the Python; held equal on
// synthetic draws over the whole range by tests/cuda_unit_kit_check.py part
// DRAWS through the TEST-ONLY probe cuda_unit_kit_probe.cu.
//
//   the key      DrawKey: (seed_lo, seed_hi) = the key, tick = the counter's
//                first word, read per env from a (N, 3) int64 row
//                (simulation.unit_draws.draw_key: sim.draws.key and
//                sim.draws.tick & 0xFFFFFFFF, at the slot's entry)
//   .at          draw_at(key, salt, ident, index)
//   .next        DrawSite {salt, ident, k} + draw_next(key, site): k is a
//                counter LOCAL TO THE ROW'S THREAD, exact because every .next
//                user of one salt sits in ONE kernel per tick (the generated
//                header records the ONE src/ site of each such salt;
//                tests/test_unit_draws.py fails on a second)
//   the maps     cone_offset_q16(word, cone_q), roll_q16(word, p_q)
//
// The salts are the GENERATED unit_draws::SALT_* (unit_draws_layout.h), never
// typed by hand. An operand outside the Python's domain sets *bad (the caller
// FAULTS): an id outside [-1, 2^32 - 1), an index outside [0, 2^32), a key
// word outside [0, 2^32), a cone outside [0, CONE_MAX].
//
// Device-only header; CLASS T: nothing in the played tick includes it.
#include <cstdint>

#include "philox32.h"              // the vendored block function (PHILOX32_HD)
#include "unit_draws_layout.h"     // GENERATED: the salts, the key row, the .next sites

namespace breach_cuda {
namespace unit_draws_dev {

using namespace unit_draws;

constexpr int64_t WORD_MAX = 0xFFFFFFFFLL;
//: The widest cone the map's 64-bit product holds: (2c + 1) < 2^32.
constexpr int64_t CONE_MAX = 0x7FFFFFFFLL;

struct DrawKey {
    uint32_t key[2];
    uint32_t tick;
};

__device__ __forceinline__ bool word_ok(int64_t v) { return v >= 0 && v <= WORD_MAX; }

// Env e's key row of a (N, DRAW_KEY_W) int64 array.
__device__ inline DrawKey draw_key(const int64_t* keys, int e, bool* bad) {
    const int64_t* k = keys + (size_t)e * DRAW_KEY_W;
    if (!(word_ok(k[DRAW_KEY_SEED_LO]) && word_ok(k[DRAW_KEY_SEED_HI]) &&
          word_ok(k[DRAW_KEY_TICK])))
        *bad = true;
    DrawKey d;
    d.key[0] = (uint32_t)k[DRAW_KEY_SEED_LO];
    d.key[1] = (uint32_t)k[DRAW_KEY_SEED_HI];
    d.tick = (uint32_t)k[DRAW_KEY_TICK];
    return d;
}

// draws._id_word: -1 ("no unit") is 2^32 - 1; anything else in [0, 2^32 - 1).
__device__ __forceinline__ uint32_t id_word(int64_t ident, bool* bad) {
    if (ident == -1) return 0xFFFFFFFFu;
    if (ident < 0 || ident >= WORD_MAX) {
        *bad = true;
        return 0u;
    }
    return (uint32_t)ident;
}

// Draw.word: the first word of the block (tick, index, salt, id) under the key.
__device__ __forceinline__ uint32_t block_word(const DrawKey& k, uint32_t index, uint32_t salt,
                                               uint32_t idw) {
    const uint32_t ctr[4] = {k.tick, index, salt, idw};
    uint32_t out[4];
    philox32_4x32_10(ctr, k.key, out);
    return out[0];
}

// MatchDraws.at(salt, ident, index).word()
__device__ inline uint32_t draw_at(const DrawKey& k, uint32_t salt, int64_t ident, int64_t index,
                                   bool* bad) {
    const uint32_t idw = id_word(ident, bad);
    if (!word_ok(index)) {
        *bad = true;
        return 0u;
    }
    return block_word(k, (uint32_t)index, salt, idw);
}

// One .next site: (salt, ident) and its count this tick, held by the thread.
struct DrawSite {
    uint32_t salt;
    uint32_t idw;
    uint32_t k;
};

__device__ inline DrawSite draw_site(uint32_t salt, int64_t ident, bool* bad) {
    DrawSite s;
    s.salt = salt;
    s.idw = id_word(ident, bad);
    s.k = 0u;
    return s;
}

// MatchDraws.next(salt, ident).word(): index 0, 1, ... in the order the site asks.
__device__ inline uint32_t draw_next(const DrawKey& k, DrawSite& s) {
    return block_word(k, s.k++, s.salt, s.idw);
}

// draws.cone_offset_q16: ((u (2c + 1)) >> 32) - c, in [-c, c].
__device__ inline int64_t cone_offset_q16(uint32_t u, int64_t cone_q, bool* bad) {
    if (cone_q < 0 || cone_q > CONE_MAX) {
        *bad = true;
        return 0;
    }
    return (int64_t)(((uint64_t)u * (uint64_t)(2 * cone_q + 1)) >> 32) - cone_q;
}

// draws.roll_q16: True with probability p_q / 65536.
__device__ __forceinline__ bool roll_q16(uint32_t u, int64_t p_q) {
    return (int64_t)(u >> 16) < p_q;
}

}  // namespace unit_draws_dev
}  // namespace breach_cuda
