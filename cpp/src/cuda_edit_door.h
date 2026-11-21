#pragma once
// ============================================================================
// #29 P7a.3 -- THE DEVICE EDIT DOOR, step (a) (residency design 4.4c,
// decision 7): where a DEVICE producer -- an effect function of the per-env
// serial applier -- hands its FieldEdits, as the rows the device edit queue
// reads (cuda_resident.h's EDQ_* columns; field_edit.pack_for_device's EDIT
// and TILE rows), through device forms of the door's quantization and of the
// region walk for the regions it uses (cuda_edit_door.cuh). See
// src/simulation/edit_door.py (the Python face and the host reference).
//
// EditDoor holds, per env:
//   * the EDIT rows: a record list of EDIT_DOOR_COLS int64 -- the 16 columns
//     of pack_for_device's EDIT row, then the edit's SORT KEY's two columns
//     the packed row does not carry (its source_id and its region), which
//     step (c)'s per-env sort (field, source, region, seq) needs;
//   * the TILE rows: a record list of EDQ_TILE_COLS int64, the edit-row index
//     LOCAL to the env's own list (the producer's first edit is 0);
//   * `seq`: the per-env edit counter (the queue's `seq`, the noise key). It
//     RESETS AT THE 6b FLUSH (begin()), never at the tick end, and is set by
//     set_seq() when its base is a harness input (the host window's seq).
// The rows are in ENQUEUE order (step (a)); the sort and K1 / K2 fed from them
// are step (c) (P7e). The played edit queue (Class P) is untouched.
//
// CAPACITY: past a list's cap nothing is written, its dropped count rises and
// the Python face refuses the run naming the list and the env.
//
// Plain C++ declarations (no CUDA types) so bindings.cpp (cl.exe) can include
// it. Compiled only when BREACH_CUDA. CLASS T: nothing in the played tick
// uses it.
#include <cstdint>

#include "cuda_resident.h"      // EDQ_EDIT_COLS, EDQ_TILE_COLS, EDQ_E_*
#include "cuda_unit_events.h"   // UnitRecordList

namespace breach_cuda {

//: The device door's EDIT row: pack_for_device's 16 columns, then the sort
//: key's source_id and region, then (#29 P7c.1) the region's four arguments
//: (edit_door.ARG_COL): a TILE (r, c, 0, 0), a DISC (cy, cx, r_q, falloff), so
//: the chained landing rebuilds the very edit.
constexpr int EDIT_DOOR_SRC = EDQ_EDIT_COLS;
constexpr int EDIT_DOOR_REGION = EDQ_EDIT_COLS + 1;
constexpr int EDIT_DOOR_ARG = EDQ_EDIT_COLS + 2;
constexpr int EDIT_DOOR_ARGS = 4;
constexpr int EDIT_DOOR_COLS = EDQ_EDIT_COLS + 2 + EDIT_DOOR_ARGS;

class EditDoor {
public:
    // Refuses n_env < 1, edit_cap < 1, tile_cap < 1.
    EditDoor(int n_env, int edit_cap, int tile_cap);
    ~EditDoor();
    EditDoor(const EditDoor&) = delete;
    EditDoor& operator=(const EditDoor&) = delete;

    int n_env() const { return edits_.n_env(); }
    UnitRecordList& edits() { return edits_; }
    UnitRecordList& tiles() { return tiles_; }
    int64_t* seq_dev() { return d_seq_; }          // (N,)

    // The 6b flush's reset: every env's lists empty and its seq 0.
    void begin();
    // #29 P7e.3: env e's reset alone (an env reset): its lists empty, its seq
    // 0 -- #29 P8.9a: or `seq` (the new episode's host queue seq, the noise
    // key's absolute base), set on the tick stream with no host sync.
    void reset_env(int e, int64_t seq = 0);
    // Every env's seq from host memory (N int64): the window's base.
    void set_seq(const int64_t* seq);
    // The seq counters (N int64) to host memory; returns once landed.
    void download_seq(int64_t* seq);

private:
    UnitRecordList edits_;
    UnitRecordList tiles_;
    int64_t* d_seq_ = nullptr;
};

}  // namespace breach_cuda
