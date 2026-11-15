#pragma once
// ============================================================================
// #29 P6.4b -- THE PER-ENV SERIAL APPLIER and THE TICK-END CLEARS on the
// device (residency design §4.3, "The per-env serial applier"; §4.4b's P6.4
// row). See src/simulation/applier.py (the ONE table; GENERATED
// cpp/src/applier_layout.h) and cpp/src/cuda_topology.cuh (the topology seam
// as device functions).
// ============================================================================
//
// FieldApplier: the per-env EFFECT LIST (N, cap, APPLIER_F) int64 with a
// REQUESTED count per env (which may exceed cap: the first cap are applied,
// the rest counted in the env's `dropped` book) and the per-env BOOK channels
// (N, APPLIER_N_BOOKS) int64. stage() copies a host list through pinned memory
// on the tick stream.
//
// applier_run: ONE BLOCK per env (grid (1, N), blockIdx.y the env -- the bgrid
// convention), thread 0 walking the env's effects in list order (the
// reference's order), each effect a device function of the topology seam
// writing the env's planes in place and its books. The other threads of the
// block are idle today: the fan-out of an effect's tile set over them, with a
// barrier between effects, arrives with its first user (P7c's deposits).
// The books are zeroed before and downloaded after (a stream sync: allowed,
// nothing in the played tick calls this).
//
// tick_end_clears: Simulation._clear_tick_buffers' planes (heat, rad_net,
// rad_amb, rad_flux) zeroed on every env whose `active` byte is set (the
// reference clears them only with a physics runner), an (N, h, w) launch on
// the tick stream.
//
// Plain C++ declarations (no CUDA types) so bindings.cpp (cl.exe) can include
// it. Compiled only when BREACH_CUDA. CLASS T: nothing in the played tick
// uses it.
#include <cstddef>
#include <cstdint>

#include "applier_layout.h"     // GENERATED: the record, books, planes
#include "cuda_unit_store.h"    // UnitTablesDevice

namespace breach_cuda {

class FieldApplier {
public:
    // Refuses n_env < 1 or cap < 1.
    FieldApplier(int n_env, int cap);
    ~FieldApplier();
    FieldApplier(const FieldApplier&) = delete;
    FieldApplier& operator=(const FieldApplier&) = delete;

    int n_env() const { return n_env_; }
    int cap() const { return cap_; }
    int64_t* recs_dev() { return d_recs_; }       // (N, cap, F)
    int64_t* counts_dev() { return d_cnt_; }      // (N,)
    int64_t* books_dev() { return d_books_; }     // (N, N_BOOKS)

    // A packed list (N * cap * F int64) and the requested counts (N int64)
    // to the device (pinned staging, async on the tick stream).
    void stage(const int64_t* recs, const int64_t* counts);
    // The books (N * N_BOOKS int64) to host memory; returns once landed.
    void download_books(int64_t* host);

private:
    int n_env_ = 0, cap_ = 0;
    int64_t* d_recs_ = nullptr;
    int64_t* d_cnt_ = nullptr;
    int64_t* d_books_ = nullptr;
    int64_t* h_stage_ = nullptr;       // pinned: recs then counts
    int64_t* h_books_ = nullptr;       // pinned
    void* staged_ = nullptr;           // cudaEvent_t: the last stage copy
};

// The topology seam's planes: device addresses of the (N, ...) arrays in
// applier_layout.h's APPLIER_PLANE_* order, each in the GameMap field's own
// dtype and shape (ResidentBatch.buffers as they are); gas is (N, n_gases, h, w).
struct ApplierPlanes {
    std::uintptr_t ptr[applier::APPLIER_N_PLANES] = {};
    int n_gases = 0;
    int h = 0;
    int w = 0;
};

// The applier over every env's staged list; `books_out` (N * N_BOOKS int64
// host) receives the books.
void applier_run(FieldApplier& a, const UnitTablesDevice& tables, const ApplierPlanes& planes,
                 int64_t* books_out);

// The tick-end clears: `ptr` the device addresses of the (N, h, w) planes in
// APPLIER_CLEAR_* order; `active` a device pointer to (N,) uint8 (nullptr: all).
void tick_end_clears(int n_env, int h, int w,
                     const std::uintptr_t (&ptr)[applier::APPLIER_N_CLEARS],
                     std::uintptr_t active);

// The compiled layout (what simulation.applier.DeviceApplier checks).
const char* applier_fingerprint();

// TEST-ONLY (cuda_topology_probe.cu): the staged list's records read as
// (y, x, new material in `arg`): `material = m; on_tile_changed` per record,
// serially per env; the books into `books_out` (applied = records done,
// faults = a record out of the grid or the table).
void topology_probe_tile_changed(FieldApplier& a, const UnitTablesDevice& tables,
                                 const ApplierPlanes& planes, int64_t* books_out);

}  // namespace breach_cuda
