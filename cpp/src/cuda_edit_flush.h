#pragma once
// ============================================================================
// #29 P7e.3 -- THE EDIT DOOR, STEP (c): slot 6b's flush FROM THE DEVICE DOOR
// (residency design, "P7e RE-CUT (2026-11-19)", the P7e.3 row; inventory
// docs/residency_p7e_inventory_2026-11-19.md part D). See cuda_edit_flush.cu
// and src/simulation/edit_door.py (DeviceEditFlush, the Python face).
//
// EditFlush holds the buffers the device edit queue's K1 / K2
// (cuda_edit_queue.cu, unchanged) read, BUILT ON THE DEVICE from the env's
// EditDoor rows (the window: every row a device producer wrote since the last
// flush, across the tick boundary), plus the books and the per-env fault word:
//   edits  (N * edit_cap + 1, EDQ_EDIT_COLS): env e's rows in the flush's
//          applied order (the host's stable sort (field, source, region, seq))
//          at e * edit_cap + rank, E_ENV = e; the last row a zero-amount DUMMY
//          the padding tiles name;
//   tiles  (N * tile_cap, EDQ_TILE_COLS): every env's tiles carried with their
//          edit, compacted per env in applied order (env e's at [off[e],
//          off[e+1])), the edit index batch-global; the rest padding (the
//          dummy, cell 0) so K1 takes the capacity and the host reads no count;
//   off    (N + 1) the env tile offsets; env (N, EDQ_ENV_COLS) the per-env
//          scalars (the tick word and the draw key from THE TICK ROW, the
//          seam's ambient from the device tables' topo_t_amb_q);
//   books  (N, EDQ_SLOTS) K2's books, then (N) the fault words: THE ONE
//          download after the flush (cuda_edit_queue.cu's failure slot and
//          the build's refusals read together, one sync).
//
// Plain C++ declarations (no CUDA types) so bindings.cpp (cl.exe) can include
// it. Compiled only when BREACH_CUDA. CLASS T: nothing in the played tick
// uses it (the played tick keeps the host pack until P7e.4).
#include <cstdint>

#include "cuda_edit_door.h"

namespace breach_cuda {

class UnitTablesDevice;

//: The build's refusals (edit_door.FLUSH_FAULTS names them): the lowest code
//: of an env wins (then its lowest row), so the named refusal is the same on
//: every run. Each is a row pack_for_device refuses, raises on or falls back
//: on (no host fallback exists inside a device tick).
enum EditFlushFault : int {
    EF_NONE = 0,
    EF_DOOR_OVERFLOW = 1,   // a door list dropped rows / a count outside its cap
    EF_FIELD = 2,           // a field ordinal the device has no policy for
    EF_WATER_DEPTH = 3,     // DEVICE_REFUSED_FIELDS (the host flushed it)
    EF_GAS_SLICE = 4,       // a gas slice outside [0, G) (numpy's index wrap)
    EF_CONSERVATIVE = 5,    // a CONSERVATIVE slice through the gas policy (raises)
    EF_MODE = 6,            // an EditMode outside ADD / REMOVE / MAX
    EF_DEN = 7,             // den outside (0, 2^62)
    EF_NOISE_LOW = 8,       // a noisy row's low bound outside [0, ONE]
    EF_KEY = 9,             // keys not the window's consecutive seqs
    EF_TILE = 10,           // a tile outside its env's edits or the grid
    EF_PRODUCT = 11,        // |amount x weight x mult x scale| >= 2^125
    EF_CONTRIB = 12,        // the contribution >= 2^55
    EF_N_CODES = 13,
};

class EditFlush {
public:
    // Refuses n_env < 1, edit_cap < 1, tile_cap < 1.
    EditFlush(int n_env, int edit_cap, int tile_cap);
    ~EditFlush();
    EditFlush(const EditFlush&) = delete;
    EditFlush& operator=(const EditFlush&) = delete;

    int n_env() const { return n_env_; }
    int edit_cap() const { return edit_cap_; }
    int tile_cap() const { return tile_cap_; }

    // THE FLUSH (slot 6b): build the buffers from `door`'s rows (refusing an
    // env by its fault word), K1 / K2 on the planes, then door.begin() (the
    // rows empty, every seq 0). Nothing is downloaded. `planes`: the device
    // addresses of gas, temperature, gas_energy, fire, heat, solid, is_vacuum,
    // is_ambient (0 = none), thermal_solid, flammable, (N, ...) each.
    void run(EditDoor& door, const UnitTablesDevice& tables, std::uintptr_t tick_row,
             const std::uintptr_t* planes, int h, int w, int n_gases, int o2_idx,
             int n2_idx, uint32_t salt);
    // THE GATE'S REFERENCE: K1 / K2 fed from a host pack (physics_runner.
    // _edit_buffer's layout) on the same planes, into the same books.
    void run_host_pack(const int64_t* buf, int n_edits, int n_tiles,
                       const std::uintptr_t* planes, int h, int w, int n_gases,
                       int o2_idx, int n2_idx, uint32_t salt);
    // THE ONE READ: books (N * EDQ_SLOTS) then the fault words (N; ~0 = none).
    void download_books(int64_t* books, uint64_t* faults);
    // The gate's download of the built buffers.
    void download_rows(int64_t* edits, int64_t* tiles, int64_t* off, int64_t* env);

private:
    int n_env_ = 0, edit_cap_ = 0, tile_cap_ = 0;
    int64_t* d_edits_ = nullptr;
    int64_t* d_tiles_ = nullptr;
    int64_t* d_off_ = nullptr;
    int64_t* d_env_ = nullptr;
    int64_t* d_rank_ = nullptr;     // (N, edit_cap): source row -> applied position
    int64_t* d_tsrc_ = nullptr;     // (N, edit_cap): source row's first tile
    int64_t* d_tcnt_ = nullptr;     // (N, edit_cap): applied position's tile count
    int64_t* d_tpre_ = nullptr;     // (N, edit_cap): applied position's tile prefix
    int64_t* d_total_ = nullptr;    // (N)
    int64_t* d_contrib_ = nullptr;  // (N * tile_cap)
    unsigned long long* d_books_ = nullptr;   // (N, EDQ_SLOTS) + (N) faults
    int64_t* d_host_ = nullptr;     // the host pack's staging (run_host_pack)
    size_t host_cap_ = 0;
};

// TEST-ONLY: append host rows to `door`'s env e as a device producer would
// (at the env's count, keyed by its seq: each row's key must equal the seq it
// lands at, else nothing is written and false is returned; the seq counts on).
// The gate's way to fill a window across the tick boundary from the host
// producers' door_rows.
bool edit_door_append_probe(EditDoor& door, int env, const int64_t* edits, int n_edits,
                            const int64_t* tiles, int n_tiles);

}  // namespace breach_cuda
