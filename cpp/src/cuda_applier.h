#pragma once
// ============================================================================
// #29 P6.4b -- THE PER-ENV SERIAL APPLIER and THE TICK-END CLEARS on the
// device (residency design §4.3, "The per-env serial applier"; §4.4b's P6.4
// row). See src/simulation/applier.py (the ONE table; GENERATED
// cpp/src/applier_layout.h) and cpp/src/cuda_topology.cuh (the topology seam
// as device functions).
// #29 P7a.1 -- THE EFFECT LIST (design 4.4c, decisions 1, 2): the list is an
// instance of the record list (UnitRecordList: per-row staging and ONE gather
// per env in row order, so a per-row producer's records land in the
// reference's order; a serial producer appends directly); the applier's reach
// grows from the planes to the unit store, the device tables and the event
// ring (env_sink); one kernel walks it in a STAGE (APPLIER_STAGE_*), and a HIT
// in a stage that LANDS uses the exposure landing (cuda_unit_damage.cuh).
// ============================================================================
//
// FieldApplier: the per-env EFFECT LIST (a UnitRecordList of APPLIER_F-wide
// records, `cap` per env, `row_cap` staged per row of an `n_units`-row store)
// and the per-env BOOK channels (N, APPLIER_N_BOOKS) int64, plus the walk's
// scratch (the exposure landing, (N, cap, LANDING_W); the rows a walk
// changed). stage() SETS every env's list from host records (the first cap
// kept, the rest counted as dropped); a device producer appends after
// whatever the list holds (its gather continues the list); begin() empties it.
//
// applier_run: ONE BLOCK per env (grid (1, N), blockIdx.y the env -- the bgrid
// convention), thread 0 walking the env's effects in list order (the
// reference's order), each effect a device function: DESTROY_WALL and CHEW
// over the topology seam (the env's planes, in place, and its books); HIT
// (apply_packet on the target's row, its events through the env's ring --
// the landing in a stage that lands), STATUS (apply_status if alive), PUSH
// (impulse_add if alive) on the store; PAYLOAD refused (counted in the
// `refused` book: no device function until P7c.1). The other threads of the
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

#include "applier_layout.h"     // GENERATED: the record, kinds, stages, books, planes
#include "cuda_unit_events.h"   // UnitRecordList, UnitEventRing
#include "cuda_unit_store.h"    // UnitStoreDevice, UnitTablesDevice

namespace breach_cuda {

class ObjectTable;   // cuda_object_table.h (#29 P6b.1: the DOORS stage)

class FieldApplier {
public:
    // Refuses n_env < 1, cap < 1, n_units < 1 or row_cap < 1.
    FieldApplier(int n_env, int cap, int n_units, int row_cap);
    ~FieldApplier();
    FieldApplier(const FieldApplier&) = delete;
    FieldApplier& operator=(const FieldApplier&) = delete;

    int n_env() const { return list_.n_env(); }
    int cap() const { return list_.cap(); }
    int n_units() const { return list_.n_units(); }
    int row_cap() const { return list_.row_cap(); }
    UnitRecordList& list() { return list_; }       // THE effect list
    int64_t* books_dev() { return d_books_; }      // (N, N_BOOKS)
    int64_t* landing_dev() { return d_land_; }     // (N, cap, LANDING_W)

    // Empty every env's list (counts, dropped, the staging).
    void begin() { list_.begin(); }
    // SET every env's list: `recs` (N * cap * F int64, each env's first cap
    // records) and the REQUESTED counts (N int64; past cap counted in the
    // list's dropped). Pinned staging, async on the tick stream.
    void stage(const int64_t* recs, const int64_t* counts);
    // The books (N * N_BOOKS int64) to host memory; returns once landed.
    void download_books(int64_t* host);
    // The rows a walk changed: a (N, U) byte scratch grown to `n` rows.
    uint8_t* touched(int n_units);

private:
    UnitRecordList list_;
    int64_t* d_books_ = nullptr;
    int64_t* d_land_ = nullptr;
    int64_t* h_stage_ = nullptr;       // pinned: recs, then counts, then dropped
    int64_t* h_books_ = nullptr;       // pinned
    uint8_t* d_touched_ = nullptr;
    size_t touched_cap_ = 0;
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

// What applier_run returns besides the books: per env the rows the walk
// changed and the ring's dropped events (N int64 each, host).
struct ApplierOut {
    int64_t* books = nullptr;       // N * N_BOOKS
    int64_t* coverage = nullptr;    // N
    int64_t* ring_dropped = nullptr;   // N (may be null)
};

// #29 P7a.3: the walk's further reach -- the per-env lists an effect reads
// or writes besides the planes, the rows and the ring: THE CELL BUFFER a
// SPRAY_CELLS record's span points into (cuda_unit_spray.cuh, (y, x, div)
// rows), THE DEVICE EDIT DOOR its edits go through (cuda_edit_door.h) and
// THE CLAIMS LIST its credit claims are appended to (damage.TickCredit's
// rows (channel, y, x, attacker): credit_at's table). Any may be null: an
// effect that needs a missing one FAULTS.
class EditDoor;
struct ApplierLists {
    UnitRecordList* cells = nullptr;
    EditDoor* door = nullptr;
    UnitRecordList* claims = nullptr;
    // #29 P6b.1: the DOORS stage's door table (unit_schema.DOOR_TABLE) and
    // its tile side table (DOOR_TILE_TABLE); null for the other stages.
    ObjectTable* door_table = nullptr;
    ObjectTable* door_tiles = nullptr;
    // #29 P6b.2: the VENTS stage's duct table (unit_schema.DUCT_TABLE) and
    // vent table (VENT_TABLE); null for the other stages.
    ObjectTable* duct_table = nullptr;
    ObjectTable* vent_table = nullptr;
};

// The applier over every env's list in stage `stage` (APPLIER_STAGE_*), on
// the store's rows and the env's planes; `ring` (may be null: a HIT then
// faults) receives the events; `lists` (may be null) the walk's lists.
void applier_run(FieldApplier& a, UnitStoreDevice& store, const UnitTablesDevice& tables,
                 const ApplierPlanes& planes, int stage, UnitEventRing* ring,
                 const ApplierOut& out, const ApplierLists* lists = nullptr);

// The tick-end clears: `ptr` the device addresses of the (N, h, w) planes in
// APPLIER_CLEAR_* order; `active` a device pointer to (N,) uint8 (nullptr: all).
void tick_end_clears(int n_env, int h, int w,
                     const std::uintptr_t (&ptr)[applier::APPLIER_N_CLEARS],
                     std::uintptr_t active);

// The compiled layout (what simulation.applier.DeviceApplier checks).
const char* applier_fingerprint();

// TEST-ONLY (cuda_topology_probe.cu): the staged list's records read as
// (y, x, new material) in their first three arg slots: `material = m;
// on_tile_changed` per record, serially per env; the books into `books_out`
// (applied = records done, faults = a record out of the grid or the table).
void topology_probe_tile_changed(FieldApplier& a, const UnitTablesDevice& tables,
                                 const ApplierPlanes& planes, int64_t* books_out);

// #29 P6b.1, TEST-ONLY (cuda_topology_probe.cu): per env its first n_ops[e]
// of `ops` (N, k_cap, 8) -- seal / unseal / can_seal over spans of `tiles`
// (N, t_cap, 2), inject / extract / transfer, _aperture_blocked -- serially
// on the planes in place; results (N, k_cap, 9): code, value, the per-slice
// withdrawal; books (N, N_BOOKS).
void topology_probe_seam(const UnitTablesDevice& tables, const ApplierPlanes& planes,
                         const int64_t* ops, const int64_t* n_ops, int k_cap,
                         const int64_t* tiles, int t_cap, int64_t* results_out,
                         int64_t* books_out);

}  // namespace breach_cuda
