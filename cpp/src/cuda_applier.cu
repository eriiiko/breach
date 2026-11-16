// ============================================================================
// #29 P6.4b -- THE PER-ENV SERIAL APPLIER (slot 9's burn-through) and THE
// TICK-END CLEARS on the device. See cuda_applier.h; residency design §4.3.
// #29 P7a.1 -- THE EFFECT LIST and its STAGES (design 4.4c, decisions 1, 2):
// the list is a UnitRecordList; the walk reaches the unit store, the device
// tables and the event ring; slot 4c (RESOLVE) is its first stage over unit
// effects, with the exposure landing.
//
// k_applier: one BLOCK per env (blockIdx.y), thread 0 walking the env's
// effect list in order -- serial, because the reference is: destroy_wall's
// breach rule reads the tiles an earlier destroy of the same walk changed,
// its membership diff and seed read the planes the previous effect wrote; a
// HIT reads the hp an earlier HIT left (the first packet to zero is the
// kill); a STATUS or PUSH acts only on a unit still alive at its turn; a
// CHEW reads the wall an earlier CHEW left. Each effect is a device function
// writing the env's planes / rows in place and its book channels.
//
// k_tick_end_clears: one thread per cell, (N, h, w) (bgrid), a masked env
// (`active` 0: the reference clears only with a physics runner) untouched.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_batch.cuh"          // tstream, benv, bptr, bgrid, env_off
#include "cuda_topology.cuh"       // the topology seam
#include "cuda_object_table.h"     // #29 P6b.1: the door tables (the DOORS stage)
#include "cuda_unit_doors.cuh"     // #29 P6b.1: the door sweep
#include "cuda_unit_damage.cuh"    // apply_packet, the landing, the sinks, the library
#include "cuda_unit_spray.cuh"     // #29 P7a.3: SPRAY_CELLS (the edit door, the claims)
#include "unit_tables_layout.h"    // GENERATED: the device tables

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

using namespace applier;

inline void ap_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in applier/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// What the walk reaches besides the planes.
struct WalkUnits {
    int64_t* store;          // (N, U, W)
    int n_units;
    UnitRingView ring;       // ring.ring == nullptr: no ring (a HIT faults)
    int64_t* landing;        // (N, cap, LANDING_W)
    uint8_t* touched;        // (N, U)
    unsigned long long* cnt; // the store's counts (coverage)
};

// #29 P6b.1: the entity tables a stage walks (the DOORS stage: the door
// table and its tile side table; null otherwise).
struct WalkTables {
    int64_t* doors = nullptr;          // (N, door cap, OBJ_DOOR_W)
    const int64_t* n_doors = nullptr;  // (N,)
    int door_cap = 0;
    int64_t* tiles = nullptr;          // (N, tile cap, OBJ_DOOR_TILE_W)
    const int64_t* n_tiles = nullptr;  // (N,)
    int tile_cap = 0;
};

// #29 P7a.3: the walk's lists (ApplierLists' device views; has_* false when
// the caller handed none: an effect needing it FAULTS).
struct WalkLists {
    bool has_cells = false, has_door = false, has_claims = false;
    RecordListView cells;
    edit_door::DoorView door;
    RecordListView claims;
};

// The row of an effect's `row` field, or nullptr (out of the env's rows or a
// padding row: never a target).
__device__ __forceinline__ int64_t* effect_row(const WalkUnits& U, int64_t r) {
    if (r < 0 || r >= U.n_units) return nullptr;
    int64_t* row = unit_lib::unit_row(U.store, U.n_units, (int)r);
    return unit_lib::row_valid(row) ? row : nullptr;
}

// The applier: one block per env, thread 0 the walk (the block's other
// threads are the fan-out's, idle until P7c gives an effect a tile set).
__global__ void k_applier(int stage, const int64_t* recs, const int64_t* counts,
                          const unsigned long long* dropped, int cap, topo::PlanePtrs P, int h,
                          int w, int n_gases, const int64_t* tables, int64_t* books,
                          WalkUnits U, WalkLists WL, WalkTables X) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = benv();
    const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
    int64_t* B = books + (size_t)e * APPLIER_N_BOOKS;
    const topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
    if (stage == APPLIER_STAGE_DOORS) {
        // #29 P6b.1: slot 9e(d)'s door sweep walks the door TABLE, not the
        // effect list (decision 1: a stage of the one walker over an entity
        // table, each door applied at once).
        if (n_gases != unit_tables::UNIT_TAB_GAS_TRACE_W) {
            B[APPLIER_BOOK_FAULTS] += 1;
            return;
        }
        doors::sweep(E, tab, X.doors + (size_t)e * X.door_cap * object_tables::OBJ_DOOR_W,
                     X.n_doors[e], X.tiles + (size_t)e * X.tile_cap * object_tables::OBJ_DOOR_TILE_W,
                     X.n_tiles[e], U.store, U.n_units, B);
        return;
    }
    const int64_t n = counts[e];
    const int64_t k = n < (int64_t)cap ? n : (int64_t)cap;
    B[APPLIER_BOOK_DROPPED] += (int64_t)dropped[e];
    const int64_t* L = recs + (size_t)e * cap * APPLIER_F;
    // the stage's mode for a HIT (decision 2: per (stage, kind))
    const bool lands = (stage == APPLIER_STAGE_RESOLVE) ? APPLIER_STAGE_RESOLVE_LANDS
                                                        : APPLIER_STAGE_TOPOLOGY_LANDS;
    unit_lib::ExposureLanding land;
    land.buf = U.landing + (size_t)e * cap * unit_lib::LANDING_W;
    land.cap = cap;
    uint8_t* touched = U.touched + (size_t)e * U.n_units;
    for (int r = 0; r < U.n_units; ++r) touched[r] = 0;
    unit_lib::EventSink sink{nullptr, 0, nullptr, nullptr};
    if (U.ring.ring != nullptr) sink = unit_lib::env_sink(U.ring);
    for (int64_t j = 0; j < k; ++j) {
        const int64_t* r = L + (size_t)j * APPLIER_F;
        const int64_t kind = r[APPLIER_FX_KIND];
        if (kind == APPLIER_KIND_DESTROY_WALL) {
            const int64_t y = r[APPLIER_DESTROY_WALL_Y], x = r[APPLIER_DESTROY_WALL_X];
            // out of the grid: destroy_wall's silent return (never an int cast
            // of a coordinate that does not fit)
            if (y < 0 || y >= h || x < 0 || x >= w) continue;
            if (topo::destroy_wall(E, tab, (int)y, (int)x, B))
                B[APPLIER_BOOK_APPLIED] += 1;
        } else if (kind == APPLIER_KIND_HIT) {
            const int64_t ri = r[APPLIER_HIT_ROW];
            int64_t* row = effect_row(U, ri);
            if (row == nullptr || U.ring.ring == nullptr) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            unit_lib::Packet p;
            p.amount_q = r[APPLIER_HIT_AMOUNT_Q];
            p.dtype = r[APPLIER_HIT_DTYPE];
            p.attacker = r[APPLIER_HIT_ATTACKER];
            p.indirect = r[APPLIER_HIT_INDIRECT] != 0;
            p.expose = r[APPLIER_HIT_EXPOSE];
            p.dir_x_q = r[APPLIER_HIT_DIR_X_Q];
            p.dir_y_q = r[APPLIER_HIT_DIR_Y_Q];
            p.ap_q = r[APPLIER_HIT_AP_Q];
            p.cause = r[APPLIER_HIT_CAUSE];
            p.mark_killed_by_zombie = false;
            bool changed = false, killed = false, bad = false;
            unit_lib::apply_packet(row, tab, p, sink, changed, &killed, &bad,
                                   lands ? &land : nullptr, ri);
            if (bad) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            B[APPLIER_BOOK_APPLIED] += 1;
            if (changed) touched[ri] = 1;
        } else if (kind == APPLIER_KIND_STATUS) {
            const int64_t ri = r[APPLIER_STATUS_ROW];
            int64_t* row = effect_row(U, ri);
            if (row == nullptr) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            if (!unit_lib::row_alive(row)) continue;      // statuses freeze on corpses
            bool changed = false;
            const int res = unit_lib::apply_status(
                row, tab, r[APPLIER_STATUS_STATUS], r[APPLIER_STATUS_MAGNITUDE_Q],
                r[APPLIER_STATUS_DURATION], r[APPLIER_STATUS_SOURCE], changed);
            if (res == unit_lib::APPLY_BAD) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            B[APPLIER_BOOK_APPLIED] += 1;                  // a stack-cap refusal changes nothing
            if (changed) touched[ri] = 1;
        } else if (kind == APPLIER_KIND_PUSH) {
            const int64_t ri = r[APPLIER_PUSH_ROW];
            int64_t* row = effect_row(U, ri);
            if (row == nullptr) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            if (!unit_lib::row_alive(row)) continue;      // a corpse takes no push
            bool changed = false, bad = false;
            unit_lib::impulse_add(row, tab, r[APPLIER_PUSH_JX_Q], r[APPLIER_PUSH_JY_Q],
                                  r[APPLIER_PUSH_LEVER_Q], r[APPLIER_PUSH_TWIST_Q16], changed,
                                  &bad);
            if (bad) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            B[APPLIER_BOOK_APPLIED] += 1;
            if (changed) touched[ri] = 1;
        } else if (kind == APPLIER_KIND_CHEW) {
            // combat.chew_wall: a no-op for wall_damage <= 0 and on air; the
            // wall's hp less the per-call quantized damage, destroy_wall at <= 0
            if (r[APPLIER_CHEW_LIVE] == 0) continue;
            const int64_t y = r[APPLIER_CHEW_Y], x = r[APPLIER_CHEW_X];
            if (y < 0 || y >= h || x < 0 || x >= w) {     // numpy would wrap or raise
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            const int i = (int)y * w + (int)x;
            if ((int64_t)E.material[i] == tab[unit_tables::UNIT_TAB_TOPO_MAT_AIR]) continue;
            const int64_t hp = (int64_t)E.wall_hp[i] - r[APPLIER_CHEW_DAMAGE_Q];
            if (hp < -2147483648LL || hp > 2147483647LL) {   // int32's range
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            E.wall_hp[i] = (int32_t)hp;
            B[APPLIER_BOOK_APPLIED] += 1;
            if (hp <= 0) topo::destroy_wall(E, tab, (int)y, (int)x, B);
        } else if (kind == APPLIER_KIND_SPRAY_CELLS) {
            // #29 P7a.3: combat._enqueue_spray_cells -- the edits through the
            // door, the claims onto the claims list (cuda_unit_spray.cuh)
            if (!WL.has_cells || !WL.has_door || !WL.has_claims) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            unit_lib::RecordSink cs = unit_lib::list_env_sink(WL.claims, e);
            if (!unit_spray::apply_spray_cells(r, WL.cells, e, WL.door, cs, tab, h, w, n_gases)) {
                B[APPLIER_BOOK_FAULTS] += 1;
                continue;
            }
            B[APPLIER_BOOK_APPLIED] += 1;
        } else if (kind == APPLIER_KIND_PAYLOAD) {
            B[APPLIER_BOOK_REFUSED] += 1;                  // no device function until P7c.1
        } else {
            B[APPLIER_BOOK_FAULTS] += 1;                   // an unknown kind: never packed
        }
    }
    if (lands) unit_lib::land_exposure(land, U.store, U.n_units, tab, touched);
    for (int r = 0; r < U.n_units; ++r)
        if (touched[r]) unit_lib::count_changed(U.cnt);
}

__global__ void k_tick_end_clears(int n, int32_t* heat, int64_t* rad_net, int64_t* rad_amb,
                                  int64_t* rad_flux, const uint8_t* active) {
    if (env_off(active)) return;
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    bptr(heat, (size_t)n)[i] = 0;
    bptr(rad_net, (size_t)n)[i] = 0;
    bptr(rad_amb, (size_t)n)[i] = 0;
    bptr(rad_flux, (size_t)n)[i] = 0;
}

static_assert(APPLIER_CLEAR_HEAT_BYTES == 4 && APPLIER_CLEAR_RAD_NET_BYTES == 8 &&
              APPLIER_CLEAR_RAD_AMB_BYTES == 8 && APPLIER_CLEAR_RAD_FLUX_BYTES == 8,
              "the clears' dtypes moved: update k_tick_end_clears");
static_assert(APPLIER_N_STAGES == 3, "a new stage: give it its mode in k_applier");

}  // namespace

FieldApplier::FieldApplier(int n_env, int cap, int n_units, int row_cap)
    : list_(n_env, n_units, cap, APPLIER_F, row_cap) {
    const size_t nr = (size_t)n_env * cap * APPLIER_F;
    const size_t nb = (size_t)n_env * APPLIER_N_BOOKS;
    const size_t nl = (size_t)n_env * cap * unit_lib::LANDING_W;
    ap_check(cudaMalloc(reinterpret_cast<void**>(&d_books_), nb * sizeof(int64_t)), "malloc books");
    ap_check(cudaMalloc(reinterpret_cast<void**>(&d_land_), nl * sizeof(int64_t)), "malloc landing");
    ap_check(cudaMallocHost(reinterpret_cast<void**>(&h_stage_), (nr + 2 * (size_t)n_env) *
                                                                     sizeof(int64_t)),
             "malloc pinned stage");
    ap_check(cudaMallocHost(reinterpret_cast<void**>(&h_books_), nb * sizeof(int64_t)),
             "malloc pinned books");
    cudaEvent_t ev = nullptr;
    ap_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
    staged_ = ev;
    ap_check(cudaMemsetAsync(d_books_, 0, nb * sizeof(int64_t), tstream()), "zero books");
    ap_check(cudaMemsetAsync(d_land_, 0, nl * sizeof(int64_t), tstream()), "zero landing");
    touched(n_units);
    ap_check(cudaEventRecord(ev, tstream()), "record zero");
}

FieldApplier::~FieldApplier() {
    if (staged_) cudaEventSynchronize(static_cast<cudaEvent_t>(staged_));
    cudaStreamSynchronize(tstream());
    if (staged_) cudaEventDestroy(static_cast<cudaEvent_t>(staged_));
    if (d_books_) cudaFree(d_books_);
    if (d_land_) cudaFree(d_land_);
    if (d_touched_) cudaFree(d_touched_);
    if (h_stage_) cudaFreeHost(h_stage_);
    if (h_books_) cudaFreeHost(h_books_);
}

uint8_t* FieldApplier::touched(int n_units) {
    const size_t want = (size_t)n_env() * (size_t)(n_units > 0 ? n_units : 1);
    if (want > touched_cap_) {
        cudaStreamSynchronize(tstream());
        if (d_touched_) cudaFree(d_touched_);
        d_touched_ = nullptr;
        ap_check(cudaMalloc(reinterpret_cast<void**>(&d_touched_), want), "malloc touched");
        touched_cap_ = want;
    }
    return d_touched_;
}

void FieldApplier::stage(const int64_t* recs, const int64_t* counts) {
    const int n = n_env(), c = cap();
    const size_t nr = (size_t)n * c * APPLIER_F;
    // the pinned buffer is free once the previous stage copy read it
    ap_check(cudaEventSynchronize(static_cast<cudaEvent_t>(staged_)), "stage wait");
    std::memcpy(h_stage_, recs, nr * sizeof(int64_t));
    int64_t* cnt = h_stage_ + nr;
    unsigned long long* drp = reinterpret_cast<unsigned long long*>(h_stage_ + nr + n);
    for (int e = 0; e < n; ++e) {
        const int64_t req = counts[e] < 0 ? 0 : counts[e];
        cnt[e] = req < (int64_t)c ? req : (int64_t)c;
        drp[e] = (unsigned long long)(req - cnt[e]);
    }
    list_.begin();                            // the staging counts to 0
    ap_check(cudaMemcpyAsync(list_.records_dev(), h_stage_, nr * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D recs");
    ap_check(cudaMemcpyAsync(list_.count_dev(), cnt, (size_t)n * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D counts");
    ap_check(cudaMemcpyAsync(list_.dropped_dev(), drp, (size_t)n * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice, tstream()), "H2D dropped");
    ap_check(cudaEventRecord(static_cast<cudaEvent_t>(staged_), tstream()), "record stage");
}

void FieldApplier::download_books(int64_t* host) {
    const size_t nb = (size_t)n_env() * APPLIER_N_BOOKS;
    ap_check(cudaMemcpyAsync(h_books_, d_books_, nb * sizeof(int64_t), cudaMemcpyDeviceToHost,
                             tstream()), "D2H books");
    ap_check(cudaStreamSynchronize(tstream()), "books sync");
    std::memcpy(host, h_books_, nb * sizeof(int64_t));
}

void applier_run(FieldApplier& a, UnitStoreDevice& store, const UnitTablesDevice& tables,
                 const ApplierPlanes& planes, int stage, UnitEventRing* ring,
                 const ApplierOut& out, const ApplierLists* lists) {
    if (tables.n_env() != a.n_env() || store.n_env() != a.n_env())
        throw std::invalid_argument("applier_run: the tables / store are for another env count");
    if (stage < 0 || stage >= APPLIER_N_STAGES)
        throw std::invalid_argument("applier_run: no such stage");
    if (planes.h < 1 || planes.w < 1 || planes.n_gases < 1)
        throw std::invalid_argument("applier_run: the planes' shape is empty");
    if (ring != nullptr && (ring->n_env() != store.n_env() || ring->n_units() != store.n_units()))
        throw std::invalid_argument("applier_run: the event ring is for another store");
    topo::PlanePtrs P;
    for (int k = 0; k < APPLIER_N_PLANES; ++k) {
        if (planes.ptr[k] == 0)
            throw std::invalid_argument("applier_run: a plane pointer is null");
        P.p[k] = (unsigned long long)planes.ptr[k];
    }
    WalkUnits U;
    U.store = store.rows();
    U.n_units = store.n_units();
    U.ring = ring != nullptr ? ring_view(*ring) : UnitRingView{};
    U.landing = a.landing_dev();
    U.touched = a.touched(store.n_units());
    U.cnt = store.counts_dev();
    ap_check(cudaMemsetAsync(a.books_dev(), 0,
                             (size_t)a.n_env() * APPLIER_N_BOOKS * sizeof(int64_t), tstream()),
             "zero books");
    store.begin_counts();
    WalkLists WL;
    if (lists != nullptr) {
        const int n = a.n_env();
        if (lists->cells != nullptr) {
            if (lists->cells->n_env() != n || lists->cells->width() != unit_spray::CELL_W)
                throw std::invalid_argument("applier_run: the cell buffer is for another env "
                                            "count or not (y, x, div) wide");
            WL.has_cells = true;
            WL.cells = list_view(*lists->cells);
        }
        if (lists->door != nullptr) {
            if (lists->door->n_env() != n)
                throw std::invalid_argument("applier_run: the edit door is for another env count");
            WL.has_door = true;
            WL.door.edits = list_view(lists->door->edits());
            WL.door.tiles = list_view(lists->door->tiles());
            WL.door.seq = lists->door->seq_dev();
        }
        if (lists->claims != nullptr) {
            if (lists->claims->n_env() != n || lists->claims->width() != 4)
                throw std::invalid_argument("applier_run: the claims list is for another env "
                                            "count or not (channel, y, x, attacker) wide");
            WL.has_claims = true;
            WL.claims = list_view(*lists->claims);
        }
    }
    WalkTables X;
    if (stage == APPLIER_STAGE_DOORS) {
        ObjectTable* doors = lists != nullptr ? lists->door_table : nullptr;
        ObjectTable* tiles = lists != nullptr ? lists->door_tiles : nullptr;
        if (lists == nullptr || lists->door_table == nullptr || lists->door_tiles == nullptr)
            throw std::invalid_argument("applier_run: the DOORS stage needs the door table and "
                                        "its tile table");
        if (doors->n_env() != a.n_env() || tiles->n_env() != a.n_env() ||
            doors->width() != object_tables::OBJ_DOOR_W ||
            tiles->width() != object_tables::OBJ_DOOR_TILE_W)
            throw std::invalid_argument("applier_run: the door tables are not the door / "
                                        "door tile layouts for these envs");
        X.doors = doors->records_dev();
        X.n_doors = doors->count_dev();
        X.door_cap = doors->cap();
        X.tiles = tiles->records_dev();
        X.n_tiles = tiles->count_dev();
        X.tile_cap = tiles->cap();
    }
    UnitRecordList& L = a.list();
    k_applier<<<dim3(1, (unsigned)a.n_env()), 32, 0, tstream()>>>(
        stage, L.records_dev(), L.count_dev(), L.dropped_dev(), L.cap(), P, planes.h, planes.w,
        planes.n_gases, tables.values(), a.books_dev(), U, WL, X);
    ap_check(cudaGetLastError(), "applier launch");
    a.download_books(out.books);
    std::unique_ptr<int64_t[]> f(new int64_t[(size_t)a.n_env()]);
    std::unique_ptr<int64_t[]> rf(new int64_t[(size_t)a.n_env()]);
    std::unique_ptr<int64_t[]> cv(new int64_t[(size_t)a.n_env()]);
    store.end_counts(out.coverage != nullptr ? out.coverage : cv.get(), f.get(), rf.get());
    if (out.ring_dropped != nullptr) {
        if (ring != nullptr) {
            ring->download(nullptr, nullptr, out.ring_dropped);
        } else {
            for (int e = 0; e < a.n_env(); ++e) out.ring_dropped[e] = 0;
        }
    }
}

void tick_end_clears(int n_env, int h, int w,
                     const std::uintptr_t (&ptr)[APPLIER_N_CLEARS], std::uintptr_t active) {
    if (n_env < 1 || h < 1 || w < 1)
        throw std::invalid_argument("tick_end_clears: empty shape");
    for (int k = 0; k < APPLIER_N_CLEARS; ++k)
        if (ptr[k] == 0) throw std::invalid_argument("tick_end_clears: a plane pointer is null");
    const int n = h * w;
    k_tick_end_clears<<<bgrid(n, n_env), 256, 0, tstream()>>>(
        n, reinterpret_cast<int32_t*>(ptr[APPLIER_CLEAR_HEAT]),
        reinterpret_cast<int64_t*>(ptr[APPLIER_CLEAR_RAD_NET]),
        reinterpret_cast<int64_t*>(ptr[APPLIER_CLEAR_RAD_AMB]),
        reinterpret_cast<int64_t*>(ptr[APPLIER_CLEAR_RAD_FLUX]),
        reinterpret_cast<const uint8_t*>(active));
    ap_check(cudaGetLastError(), "clears launch");
}

const char* applier_fingerprint() { return APPLIER_FINGERPRINT; }

}  // namespace breach_cuda
