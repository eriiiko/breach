#pragma once
// ============================================================================
// #29 P7a.3 -- THE DEVICE EDIT DOOR's device functions, step (a) (residency
// design 4.4c, decision 7), and THE CLAIMS LIST's writer (decision 4). See
// cuda_edit_door.h (the buffers) and src/simulation/edit_door.py.
//
// door_tile_edit: ONE FieldEdit of region TILE, falloff FLAT, no noise, no
// scale and no clamp of its own (the field policy's), with its amount already
// in the field's currency (FieldEdit.amount_q), handed to the door as the EDIT
// row field_edit.pack_for_device writes for it (field_edit._edit_door: the
// amount as given, den ONE, the noise's low bound ONE, the policy clamp in
// counts -- the tables' edit_clamp_* rows, field_edit._clamp_q of the policy's
// pair) and the TILE rows of its region (_iter_region's TILE: the one tile,
// weight ONE, when it lies in the grid; nothing otherwise), keyed by the
// env's seq, which then counts on. E_ENV is 0 as in the pack (the batch's
// consumer sets it). The EDIT row carries the sort key's source_id and
// region after the 16 packed columns (step (c)'s sort).
// A producer that needs another region (TILES, the falloff weights' TILES
// form: P7c.2) or noise adds its device form HERE, never in its own TU.
//
// #29 P7c.1 -- door_disc_edit: ONE FieldEdit of region DISC, FLAT or LINEAR,
// no noise, no scale, its amount already in the field's currency and its
// clamp given (the policy's -- the edit_clamp_* rows -- or the edit's own,
// e.g. ignite_ring's IGNITE_CLAMP row), as pack_for_device writes it: the
// EDIT row (den ONE, low ONE, key = the env's seq, the region's arguments
// (cy, cx, r_q, falloff)) and the TILE rows of _iter_region's DISC -- every
// in-grid tile of the box ceil(r) around the centre whose d2 * 2^32 < r_q^2,
// row-major, the weight ONE (FLAT) or falloff_round(ONE, d2 << 32, r_q^2)
// (LINEAR); none when r_q <= 0. The skip masks are the flush's (never
// here). disc_claims: damage.TickCredit.claim of the same disc's tiles
// (payloads._claim_disc: FLAT membership, row-major), each through claim().
// Both are proved for r_q <= DISC_R_MAX (the box's d2 << 32 and r_q^2 inside
// int64) and a centre within +-DISC_CENTRE_MAX; the caller checks first.
//
// claim: damage.TickCredit.claim of one tile -- a no-op for the environment
// (attacker -1); otherwise one (channel, y, x, attacker) record on the env's
// claims list, EVERY claim in the reference's order (duplicates included):
// "first claim wins" is the reader's (credit_at scans first match first).
//
// Device-only header; CLASS T: nothing in the played tick includes it.
#include <cstdint>

#include "cuda_edit_door.h"
#include "cuda_unit_lib.cuh"       // RecordListView, RecordSink, emit_record, list_env_sink

namespace breach_cuda {
namespace edit_door {

using unit_lib::RecordSink;

struct DoorView {
    RecordListView edits;      // (N, cap, EDIT_DOOR_COLS)
    RecordListView tiles;      // (N, cap, EDQ_TILE_COLS)
    int64_t* seq = nullptr;    // (N,)
};

constexpr int64_t ONE_Q16 = 65536;
constexpr int64_t REGION_TILE = 0;      // field_edit.Region.TILE
constexpr int64_t REGION_DISC = 1;      // field_edit.Region.DISC
constexpr int64_t FALLOFF_FLAT = 0;     // field_edit.Falloff.FLAT
constexpr int64_t FALLOFF_LINEAR = 1;   // field_edit.Falloff.LINEAR
//: The DISC form's proved range: a Q16 radius of at most 2^26 (2^10 tiles:
//: r_q^2 < 2^52, a box offset's d2 << 32 < 2^53) and a centre within +-2^20.
constexpr int64_t DISC_R_MAX = (int64_t)1 << 26;
constexpr int64_t DISC_CENTRE_MAX = (int64_t)1 << 20;

// One TILE edit at the door (see the header). `field` an EDQ_F_* ordinal,
// `slice` the gas slice (-1 for a scalar field), `mode` field_edit.EditMode.
__device__ inline void door_tile_edit(const DoorView& D, int e, const int64_t* tab, int field,
                                      int64_t slice, int64_t mode, int64_t amount_q,
                                      int64_t source, int64_t r, int64_t c, int h, int w) {
    RecordSink es = unit_lib::list_env_sink(D.edits, e);
    RecordSink ts = unit_lib::list_env_sink(D.tiles, e);
    const int64_t idx = *es.count;              // the edit's index in the env's own list
    int64_t row[EDIT_DOOR_COLS];
    for (int k = 0; k < EDIT_DOOR_COLS; ++k) row[k] = 0;
    row[EDQ_E_ENV] = 0;
    row[EDQ_E_FIELD] = field;
    row[EDQ_E_SLICE] = slice;
    row[EDQ_E_MODE] = mode;
    row[EDQ_E_AMOUNT] = amount_q;
    row[EDQ_E_DEN] = ONE_Q16;                   // TILE, no noise, no scale
    row[EDQ_E_NOISY] = 0;
    row[EDQ_E_LOW] = ONE_Q16;
    row[EDQ_E_KEY] = D.seq[e];
    row[EDQ_E_CLAMP_FLAGS] = tab[unit_tables::UNIT_TAB_EDIT_CLAMP_FLAGS + field];
    row[EDQ_E_LO] = tab[unit_tables::UNIT_TAB_EDIT_CLAMP_LO + field];
    row[EDQ_E_HI] = tab[unit_tables::UNIT_TAB_EDIT_CLAMP_HI + field];
    row[EDIT_DOOR_SRC] = source;
    row[EDIT_DOOR_REGION] = REGION_TILE;
    row[EDIT_DOOR_ARG + 0] = r;                 // #29 P7c.1: the region's arguments
    row[EDIT_DOOR_ARG + 1] = c;
    unit_lib::emit_record(es, row);
    D.seq[e] += 1;
    if (r < 0 || r >= h || c < 0 || c >= w) return;   // _iter_region's TILE: nothing
    const int64_t t[EDQ_TILE_COLS] = {idx, r * (int64_t)w + c, ONE_Q16, 1};
    unit_lib::emit_record(ts, t);
}

// _iter_region's DISC box: the clipped rows / cols of ceil(r_q) around the
// centre (empty when r_q <= 0: r0 > r1).
struct DiscBox {
    int64_t r0, r1, c0, c1, r2;
};
__device__ __forceinline__ DiscBox disc_box(int64_t cy, int64_t cx, int64_t r_q, int h, int w) {
    DiscBox b{1, 0, 1, 0, 0};
    if (r_q <= 0) return b;
    const int64_t ext = (r_q + ONE_Q16 - 1) >> 16;           // ceil(radius)
    b.r2 = r_q * r_q;
    b.r0 = cy - ext > 0 ? cy - ext : 0;
    b.r1 = cy + ext < h - 1 ? cy + ext : h - 1;
    b.c0 = cx - ext > 0 ? cx - ext : 0;
    b.c1 = cx + ext < w - 1 ? cx + ext : w - 1;
    return b;
}

// One DISC edit at the door (see the header). `clamp` = (flags, lo, hi).
// Returns false (nothing written) when a weight leaves the kit's range.
__device__ inline bool door_disc_edit(const DoorView& D, int e, int field, int64_t slice,
                                      int64_t mode, int64_t amount_q, int64_t source,
                                      int64_t cy, int64_t cx, int64_t r_q, int64_t falloff,
                                      int64_t clamp_flags, int64_t clamp_lo, int64_t clamp_hi,
                                      int h, int w) {
    RecordSink es = unit_lib::list_env_sink(D.edits, e);
    RecordSink ts = unit_lib::list_env_sink(D.tiles, e);
    const int64_t idx = *es.count;              // the edit's index in the env's own list
    int64_t row[EDIT_DOOR_COLS];
    for (int k = 0; k < EDIT_DOOR_COLS; ++k) row[k] = 0;
    row[EDQ_E_ENV] = 0;
    row[EDQ_E_FIELD] = field;
    row[EDQ_E_SLICE] = slice;
    row[EDQ_E_MODE] = mode;
    row[EDQ_E_AMOUNT] = amount_q;
    row[EDQ_E_DEN] = ONE_Q16;                   // a DISC, no noise, no scale
    row[EDQ_E_NOISY] = 0;
    row[EDQ_E_LOW] = ONE_Q16;
    row[EDQ_E_KEY] = D.seq[e];
    row[EDQ_E_CLAMP_FLAGS] = clamp_flags;
    row[EDQ_E_LO] = clamp_lo;
    row[EDQ_E_HI] = clamp_hi;
    row[EDIT_DOOR_SRC] = source;
    row[EDIT_DOOR_REGION] = REGION_DISC;
    row[EDIT_DOOR_ARG + 0] = cy;
    row[EDIT_DOOR_ARG + 1] = cx;
    row[EDIT_DOOR_ARG + 2] = r_q;
    row[EDIT_DOOR_ARG + 3] = falloff;
    unit_lib::emit_record(es, row);
    D.seq[e] += 1;
    const DiscBox b = disc_box(cy, cx, r_q, h, w);
    for (int64_t r = b.r0; r <= b.r1; ++r) {
        const int64_t dr = r - cy;
        for (int64_t c = b.c0; c <= b.c1; ++c) {
            const int64_t dc = c - cx;
            const int64_t d2 = (dr * dr + dc * dc) << 32;
            if (!(d2 < b.r2)) continue;
            int64_t wq = ONE_Q16;
            if (falloff == FALLOFF_LINEAR) {
                wq = fixedpoint::falloff_round_q(ONE_Q16, d2, b.r2);
                if (wq < 0) return false;       // FALLOFF_REFUSED: never in the range
            }
            const int64_t t[EDQ_TILE_COLS] = {idx, r * (int64_t)w + c, wq, 1};
            unit_lib::emit_record(ts, t);
        }
    }
    return true;
}

// payloads._claim_disc: TickCredit.claim of the DISC's tiles (FLAT
// membership, row-major) for `attacker` (the environment's: nothing).
__device__ inline void disc_claims(RecordSink& cs, int64_t channel, int64_t cy, int64_t cx,
                                   int64_t r_q, int64_t attacker, int h, int w) {
    if (attacker == -1) return;
    const DiscBox b = disc_box(cy, cx, r_q, h, w);
    for (int64_t r = b.r0; r <= b.r1; ++r) {
        const int64_t dr = r - cy;
        for (int64_t c = b.c0; c <= b.c1; ++c) {
            const int64_t dc = c - cx;
            if (((dr * dr + dc * dc) << 32) < b.r2) {
                const int64_t rec[4] = {channel, r, c, attacker};
                unit_lib::emit_record(cs, rec);
            }
        }
    }
}

// damage.TickCredit.claim of one tile on the env's claims list.
__device__ inline void claim(RecordSink& cs, int64_t channel, int64_t y, int64_t x,
                             int64_t attacker) {
    if (attacker == -1) return;                 // events.NO_ATTACKER: the environment's
    const int64_t rec[4] = {channel, y, x, attacker};
    unit_lib::emit_record(cs, rec);
}

static_assert(unit_tables::UNIT_TAB_EDIT_CLAMP_FLAGS_W == EDQ_N_FIELDS &&
                  unit_tables::UNIT_TAB_EDIT_CLAMP_LO_W == EDQ_N_FIELDS &&
                  unit_tables::UNIT_TAB_EDIT_CLAMP_HI_W == EDQ_N_FIELDS,
              "the edit door's clamp rows are one per field ordinal");

}  // namespace edit_door
}  // namespace breach_cuda
