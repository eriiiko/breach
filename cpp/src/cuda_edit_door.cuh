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
// A producer that needs another region (DISC, TILES, the falloff weights:
// P7c.1 / P7c.2) or noise adds its device form HERE, never in its own TU.
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
constexpr int64_t FALLOFF_FLAT = 0;     // field_edit.Falloff.FLAT (implied)

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
    unit_lib::emit_record(es, row);
    D.seq[e] += 1;
    if (r < 0 || r >= h || c < 0 || c >= w) return;   // _iter_region's TILE: nothing
    const int64_t t[EDQ_TILE_COLS] = {idx, r * (int64_t)w + c, ONE_Q16, 1};
    unit_lib::emit_record(ts, t);
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
