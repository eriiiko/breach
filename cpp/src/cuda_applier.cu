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
#include "cuda_unit_vents.cuh"     // #29 P6b.2: the vent circulation sweep
#include "cuda_unit_damage.cuh"    // apply_packet, the landing, the sinks, the library
#include "cuda_unit_spray.cuh"     // #29 P7a.3: SPRAY_CELLS (the edit door, the claims)
#include "cuda_unit_payload.cuh"   // #29 P7c.1: PAYLOAD (the disc path); P7c.2: every row
#include "cuda_unit_blast.cuh"     // #29 P7c.2: the walks' scratch, the SOURCES stage
#include "cuda_unit_signals.cuh"   // #29 P6b.3: the signal stages (9e a, b, c, d pumps, e)
#include "cuda_unit_horde.cuh"     // #29 P7d: the HORDE stage (slot 5) and the flood
#include "cuda_unit_hazards.cuh"   // #29 P6b.4: the timed charges and heat sources (9e d', d'')
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
    // #29 P6b.2: the VENTS stage's duct and vent tables
    int64_t* ducts = nullptr;          // (N, duct cap, OBJ_DUCT_W)
    const int64_t* n_ducts = nullptr;  // (N,)
    int duct_cap = 0;
    int64_t* vents = nullptr;          // (N, vent cap, OBJ_VENT_W)
    const int64_t* n_vents = nullptr;  // (N,)
    int vent_cap = 0;
    // #29 P7c.2: the SOURCES stage's blast-source table and tile table (and
    // its hand-out plane, (N, h, w); null: none)
    int64_t* sources = nullptr;
    int64_t* n_sources = nullptr;
    int source_cap = 0;
    int64_t* stiles = nullptr;
    int64_t* n_stiles = nullptr;
    int stile_cap = 0;
    int32_t* source_out = nullptr;
    // #29 P6b.3: the signal stages' tables (rows and counts each; the door
    // table above for EMIT / DOOR_INPUTS) and the EMIT stage's ticks
    int64_t* sig[8] = {};
    const int64_t* n_sig[8] = {};
    int sig_cap[8] = {};
    const int64_t* trow = nullptr;     // #29 P7e.1: the tick row (N, TICK_ROW_W)
    // #29 P6b.4: the TIMED_CHARGES / HEAT_SOURCES stages' entity tables
    int64_t* timed = nullptr;
    const int64_t* n_timed = nullptr;
    int timed_cap = 0;
    int64_t* heat = nullptr;
    const int64_t* n_heat = nullptr;
    int heat_cap = 0;
};

// #29 P6b.3: the signal tables' order in WalkTables::sig.
enum SigTab { ST_BUS = 0, ST_SENSOR, ST_STILE, ST_NODE, ST_NIN, ST_PUMP, ST_PIN, ST_DRIVE };
constexpr int SIG_W[8] = {object_tables::OBJ_SIGNAL_W, object_tables::OBJ_SENSOR_W,
                          object_tables::OBJ_SENSOR_TILE_W, object_tables::OBJ_NODE_W,
                          object_tables::OBJ_NODE_IN_W, object_tables::OBJ_PUMP_W,
                          object_tables::OBJ_PUMP_IN_W, object_tables::OBJ_DOOR_DRIVE_W};

__device__ inline signals::Tables signal_view(const WalkTables& X, int e) {
    signals::Tables T;
    int64_t** rows[8] = {&T.bus, &T.sensors, &T.stiles, &T.nodes, &T.nins, &T.pumps,
                         &T.pins, &T.drives};
    int64_t* cnts[8] = {&T.n_bus, &T.n_sensors, &T.n_stiles, &T.n_nodes, &T.n_nins,
                        &T.n_pumps, &T.n_pins, &T.n_drives};
    const int w[8] = {object_tables::OBJ_SIGNAL_W, object_tables::OBJ_SENSOR_W,
                      object_tables::OBJ_SENSOR_TILE_W, object_tables::OBJ_NODE_W,
                      object_tables::OBJ_NODE_IN_W, object_tables::OBJ_PUMP_W,
                      object_tables::OBJ_PUMP_IN_W, object_tables::OBJ_DOOR_DRIVE_W};
    for (int t = 0; t < 8; ++t) {
        if (X.sig[t] == nullptr) continue;
        *rows[t] = X.sig[t] + (size_t)e * X.sig_cap[t] * w[t];
        *cnts[t] = X.n_sig[t][e];
    }
    if (X.doors != nullptr) {
        T.doors = X.doors + (size_t)e * X.door_cap * object_tables::OBJ_DOOR_W;
        T.n_doors = X.n_doors[e];
    }
    return T;
}

// #29 P7a.3: the walk's lists (ApplierLists' device views; has_* false when
// the caller handed none: an effect needing it FAULTS).
struct WalkLists {
    bool has_cells = false, has_door = false, has_claims = false;
    bool has_noise = false, has_blast = false;      // #29 P7c.1
    bool has_sources = false;                       // #29 P7c.2
    RecordListView cells;
    edit_door::DoorView door;
    RecordListView claims;
    RecordListView noise;                           // #29 P7c.1: the noise list
    RecordListView blast;                           // #29 P7c.1: the blast log
    RecordListView sources;                         // #29 P7c.2: the blast sources
    RecordListView source_tiles;                    //   and their tile side table
    // #29 P6b.4: the stage's OWN effect list (the TIMED_CHARGES / HEAT_SOURCES
    // stages append each PAYLOAD they applied to it; never walked again)
    bool has_own = false;
    RecordListView own;
    // #29 P7c.2: the walks' scratch base ((h * w) cells per env, see
    // FieldApplier::walk_scratch)
    uint8_t* walk = nullptr;
    int walk_cells = 0;
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
                          WalkUnits U, WalkLists WL, WalkTables X,
                          horde_dev::HordeBuffers HB) {
    if (stage == APPLIER_STAGE_HORDE) {
        // #29 P7d: slot 5 -- the WHOLE block (passes 1 and 2 wide, pass 3 on
        // thread 0, the flood fanned out): cuda_unit_horde.cuh
        __shared__ horde_dev::Shared S;
        if (blockIdx.x != 0) return;
        const int e = benv();
        const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
        int64_t* B = books + (size_t)e * APPLIER_N_BOOKS;
        topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
        if (X.trow != nullptr)
            E.tiles = const_cast<int64_t*>(unit_draws_dev::trow_env(X.trow, e)) +
                      unit_draws::TICK_ROW_TILES_CHANGED;
        if (U.ring.ring == nullptr || !WL.has_noise || X.doors == nullptr || X.tiles == nullptr ||
            HB.hdr == nullptr || HB.latch == nullptr || X.trow == nullptr) {
            if (threadIdx.x == 0) B[APPLIER_BOOK_FAULTS] += 1;   // the stage needs them all
            return;
        }
        unit_lib::EventSink sink = unit_lib::env_sink(U.ring);
        unit_lib::RecordSink ns = unit_lib::list_env_sink(WL.noise, e);
        const size_t n = (size_t)h * w, Un = (size_t)U.n_units;
        horde_dev::Env V;
        V.E = &E;
        V.tab = tab;
        V.store = U.store;
        V.n_units = U.n_units;
        V.touched = U.touched + (size_t)e * Un;
        V.sink = &sink;
        V.noise_out = &ns;
        V.doors = X.doors + (size_t)e * X.door_cap * object_tables::OBJ_DOOR_W;
        V.n_doors = X.n_doors[e];
        V.dtiles = X.tiles + (size_t)e * X.tile_cap * object_tables::OBJ_DOOR_TILE_W;
        V.n_dtiles = X.n_tiles[e];
        V.hdr = HB.hdr + (size_t)e * APPLIER_HORDE_HDR_W;
        V.keys = HB.keys + (size_t)e * HB.fcap * APPLIER_HORDE_KEY_W;
        V.dist = HB.dist + (size_t)e * HB.fcap * n;
        V.masks = HB.masks + (size_t)e * APPLIER_HORDE_N_MASKS * n;
        V.pass = HB.pass + (size_t)e * n;
        V.players = HB.flags + (size_t)e * 2 * Un;
        V.zsel = V.players + Un;
        V.q[0] = HB.queues + (size_t)e * 2 * n;
        V.q[1] = V.q[0] + n;
        V.live = HB.live + (size_t)e * Un * 3;
        V.latch = HB.latch + (size_t)e * (APPLIER_HORDE_LATCH_W + Un);
        V.fcap = HB.fcap;
        V.tick = unit_draws_dev::trow_tick(X.trow, e);        // #29 P7e.1: the tick row
        V.serial = E.tiles;
        V.B = B;
        horde_dev::stage(V, S, WL.noise, e);
        __syncthreads();
        if (threadIdx.x == 0)
            for (int r = 0; r < U.n_units; ++r)
                if (V.touched[r]) unit_lib::count_changed(U.cnt);
        return;
    }
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = benv();
    const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
    int64_t* B = books + (size_t)e * APPLIER_N_BOOKS;
    topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
    if (X.trow != nullptr)                  // #29 P7e.1: the seam bumps the tick row
        E.tiles = const_cast<int64_t*>(unit_draws_dev::trow_env(X.trow, e)) +
                  unit_draws::TICK_ROW_TILES_CHANGED;
    // #29 P6b.3: slot 9e's signal sub-steps walk the signal TABLES (decision
    // 1: stages of the one walker, each entity applied at once)
    if (stage == APPLIER_STAGE_EMIT || stage == APPLIER_STAGE_LOGIC ||
        stage == APPLIER_STAGE_DOOR_INPUTS || stage == APPLIER_STAGE_PUMPS ||
        stage == APPLIER_STAGE_SWAP) {
        const signals::Tables T = signal_view(X, e);
        if (stage == APPLIER_STAGE_EMIT) {
            if (n_gases != unit_tables::UNIT_TAB_GAS_TRACE_W) {
                B[APPLIER_BOOK_FAULTS] += 1;
                return;
            }
            signals::emit(E, tab, T, X.trow != nullptr ? unit_draws_dev::trow_tick(X.trow, e) : 0,
                          U.store, U.n_units, B);
        } else if (stage == APPLIER_STAGE_LOGIC) {
            signals::logic(T, B);
        } else if (stage == APPLIER_STAGE_DOOR_INPUTS) {
            signals::door_inputs(T, B);
        } else if (stage == APPLIER_STAGE_PUMPS) {
            if (n_gases != unit_tables::UNIT_TAB_GAS_TRACE_W) {
                B[APPLIER_BOOK_FAULTS] += 1;
                return;
            }
            signals::pumps(E, tab, T, B);
        } else {
            signals::swap(T, B);
        }
        return;
    }
    if (stage == APPLIER_STAGE_VENTS) {
        // #29 P6b.2: slot 9e(d)'s vent circulation walks the duct and vent
        // TABLES (decision 1: a stage of the one walker, each duct at once)
        if (n_gases != unit_tables::UNIT_TAB_GAS_TRACE_W) {
            B[APPLIER_BOOK_FAULTS] += 1;
            return;
        }
        vents::sweep(E, tab, X.ducts + (size_t)e * X.duct_cap * object_tables::OBJ_DUCT_W,
                     X.n_ducts[e], X.vents + (size_t)e * X.vent_cap * object_tables::OBJ_VENT_W,
                     X.n_vents[e], B);
        return;
    }
    if (stage == APPLIER_STAGE_SOURCES) {
        // #29 P7c.2: slot 7's head, take_source_plane over the blast-source
        // TABLE (decision 1: a stage of the one walker, each source at once)
        if (!unit_blast::take_sources(
                E, tab, X.source_out != nullptr ? X.source_out + (size_t)e * E.n : nullptr,
                X.sources + (size_t)e * X.source_cap * object_tables::OBJ_SOURCE_W,
                X.n_sources + e,
                X.stiles + (size_t)e * X.stile_cap * object_tables::OBJ_SOURCE_TILE_W,
                X.n_stiles + e, B))
            B[APPLIER_BOOK_FAULTS] += 1;
        return;
    }
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
    const bool lands = (stage == APPLIER_STAGE_RESOLVE)  ? APPLIER_STAGE_RESOLVE_LANDS
                       : (stage == APPLIER_STAGE_DETONATE) ? APPLIER_STAGE_DETONATE_LANDS
                                                           : APPLIER_STAGE_TOPOLOGY_LANDS;
    unit_lib::ExposureLanding land;
    land.buf = U.landing + (size_t)e * cap * unit_lib::LANDING_W;
    land.cap = cap;
    uint8_t* touched = U.touched + (size_t)e * U.n_units;
    for (int r = 0; r < U.n_units; ++r) touched[r] = 0;
    unit_lib::EventSink sink{nullptr, 0, nullptr, nullptr};
    if (U.ring.ring != nullptr) sink = unit_lib::env_sink(U.ring);
    // #29 P7c.2: the env's walk scratch (the mark bytes, then the int32 lists)
    unit_blast::Scratch scratch{};
    const bool has_scratch = WL.walk != nullptr && WL.walk_cells >= h * w;
    if (has_scratch) {
        const size_t nc = (size_t)WL.walk_cells, N = gridDim.y;
        uint8_t* base = WL.walk;
        scratch.mark = base + (size_t)e * nc;
        int32_t* lists = reinterpret_cast<int32_t*>(base + ((N * nc + 7) & ~(size_t)7));
        scratch.queue = lists + ((size_t)0 * N + e) * nc;
        scratch.heat = lists + ((size_t)1 * N + e) * nc;
        scratch.src = lists + ((size_t)2 * N + e) * nc;
    }
    // #29 P7c.1 / P6b.4: ONE PAYLOAD application (payloads.execute_payload,
    // cuda_unit_payload.cuh) on this env's reach, booked; the effect list's
    // PAYLOAD records and the two 9e stages' detonations all go through it
    auto apply_payload_rec = [&](const int64_t* r) -> int {
        unit_lib::RecordSink cs{}, ns{}, bs{}, ss{}, ts{};
        unit_payload::PayloadReach R;
        R.store = U.store;
        R.n_units = U.n_units;
        R.touched = touched;
        R.sink = U.ring.ring != nullptr ? &sink : nullptr;
        R.door = WL.has_door ? &WL.door : nullptr;
        if (WL.has_claims) cs = unit_lib::list_env_sink(WL.claims, e);
        if (WL.has_noise) ns = unit_lib::list_env_sink(WL.noise, e);
        if (WL.has_blast) bs = unit_lib::list_env_sink(WL.blast, e);
        R.claims = WL.has_claims ? &cs : nullptr;
        R.noise = WL.has_noise ? &ns : nullptr;
        R.blast = WL.has_blast ? &bs : nullptr;
        R.scratch = has_scratch ? &scratch : nullptr;      // #29 P7c.2
        if (WL.has_sources) {
            ss = unit_lib::list_env_sink(WL.sources, e);
            ts = unit_lib::list_env_sink(WL.source_tiles, e);
            R.sources = &ss;
            R.source_tiles = &ts;
        }
        const int res = unit_payload::apply_payload(r, E, tab, e, R, B);
        if (res == unit_payload::PAYLOAD_REFUSED) {
            B[APPLIER_BOOK_REFUSED] += 1;
        } else if (res == unit_payload::PAYLOAD_BULK_REFUSED) {
            B[APPLIER_BOOK_BULK_REFUSED] += 1;
        } else if (res == unit_payload::PAYLOAD_FAULT) {
            B[APPLIER_BOOK_FAULTS] += 1;
        } else {
            B[APPLIER_BOOK_APPLIED] += 1;
        }
        return res;
    };
    if (stage == APPLIER_STAGE_TIMED_CHARGES || stage == APPLIER_STAGE_HEAT_SOURCES) {
        // #29 P6b.4: slot 9e(d') / 9e(d'') walk their entity TABLES (decision
        // 1), each detonation applied at once through apply_payload_rec
        unit_lib::RecordSink own{};
        if (WL.has_own) own = unit_lib::list_env_sink(WL.own, e);
        unit_lib::RecordSink* op = WL.has_own ? &own : nullptr;
        if (n_gases != unit_tables::UNIT_TAB_GAS_TRACE_W) {
            B[APPLIER_BOOK_FAULTS] += 1;
        } else if (stage == APPLIER_STAGE_TIMED_CHARGES) {
            hazards::timed_sweep(E, tab, X.timed + (size_t)e * X.timed_cap * object_tables::OBJ_TIMED_CHARGE_W,
                                 X.n_timed[e], apply_payload_rec, op, B);
        } else if (!WL.has_blast) {
            B[APPLIER_BOOK_FAULTS] += 1;                   // the sweep reads the blast log
        } else {
            unit_lib::RecordSink bl = unit_lib::list_env_sink(WL.blast, e);
            unit_lib::RecordSink cl{};
            if (WL.has_claims) cl = unit_lib::list_env_sink(WL.claims, e);
            hazards::heat_sweep(E, tab, e, X.heat + (size_t)e * X.heat_cap * object_tables::OBJ_HEAT_SOURCE_W,
                                X.n_heat[e], bl, WL.has_door ? &WL.door : nullptr,
                                WL.has_claims ? &cl : nullptr, apply_payload_rec, op, B);
        }
        // the own list's overflow (the count at the head was this list's: 0)
        if (WL.has_own) B[APPLIER_BOOK_DROPPED] = (int64_t)WL.own.dropped[e];
        for (int r = 0; r < U.n_units; ++r)
            if (touched[r]) unit_lib::count_changed(U.cnt);
        return;
    }
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
            // (#29 P7d: topo::chew_wall, the same law, shared with slot 5's bash)
            const int cr = topo::chew_wall(E, tab, r[APPLIER_CHEW_Y], r[APPLIER_CHEW_X],
                                           r[APPLIER_CHEW_DAMAGE_Q], r[APPLIER_CHEW_LIVE] != 0, B);
            if (cr == topo::CHEW_FAULT) B[APPLIER_BOOK_FAULTS] += 1;
            else if (cr == topo::CHEW_APPLIED) B[APPLIER_BOOK_APPLIED] += 1;
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
            // #29 P7c.1: payloads.execute_payload (cuda_unit_payload.cuh); a
            // refused row writes nothing
            apply_payload_rec(r);
        } else {
            B[APPLIER_BOOK_FAULTS] += 1;                   // an unknown kind: never packed
        }
    }
    if (lands) unit_lib::land_exposure(land, U.store, U.n_units, tab, touched);
    for (int r = 0; r < U.n_units; ++r)
        if (touched[r]) unit_lib::count_changed(U.cnt);
}

__global__ void k_tick_end_clears(int n, int32_t* heat, const uint8_t* active) {
    if (env_off(active)) return;
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    bptr(heat, (size_t)n)[i] = 0;
}

// #29 P7e.1: `heat` is the one clear plane (the legacy rad_* planes left).
static_assert(APPLIER_N_CLEARS == 1 && APPLIER_CLEAR_HEAT_BYTES == 4,
              "the clears' dtypes moved: update k_tick_end_clears");
static_assert(APPLIER_N_STAGES == 14, "a new stage: give it its mode in k_applier");

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
    if (staged_) counted_event_sync(static_cast<cudaEvent_t>(staged_), "cuda_applier.cu:536");
    counted_sync(tstream(), "cuda_applier.cu:537");
    if (staged_) cudaEventDestroy(static_cast<cudaEvent_t>(staged_));
    if (d_books_) cudaFree(d_books_);
    if (d_land_) cudaFree(d_land_);
    if (d_touched_) cudaFree(d_touched_);
    if (d_walk_) cudaFree(d_walk_);
    if (h_stage_) cudaFreeHost(h_stage_);
    if (h_books_) cudaFreeHost(h_books_);
}

uint8_t* FieldApplier::touched(int n_units) {
    const size_t want = (size_t)n_env() * (size_t)(n_units > 0 ? n_units : 1);
    if (want > touched_cap_) {
        counted_sync(tstream(), "cuda_applier.cu:550");
        if (d_touched_) cudaFree(d_touched_);
        d_touched_ = nullptr;
        ap_check(cudaMalloc(reinterpret_cast<void**>(&d_touched_), want), "malloc touched");
        touched_cap_ = want;
    }
    return d_touched_;
}

void* FieldApplier::walk_scratch(int n_cells) {
    if (n_cells > walk_cells_) {
        counted_sync(tstream(), "cuda_applier.cu:561");
        if (d_walk_) cudaFree(d_walk_);
        d_walk_ = nullptr;
        const size_t N = (size_t)n_env(), nc = (size_t)n_cells;
        const size_t marks = (N * nc + 7) & ~(size_t)7;
        const size_t bytes = marks + 3 * N * nc * sizeof(int32_t);
        ap_check(cudaMalloc(&d_walk_, bytes), "malloc walk scratch");
        ap_check(cudaMemsetAsync(d_walk_, 0, bytes, tstream()), "zero walk scratch");
        walk_cells_ = n_cells;
    }
    return d_walk_;
}

void FieldApplier::stage(const int64_t* recs, const int64_t* counts) {
    const int n = n_env(), c = cap();
    const size_t nr = (size_t)n * c * APPLIER_F;
    // the pinned buffer is free once the previous stage copy read it
    ap_check(counted_event_sync(static_cast<cudaEvent_t>(staged_), "cuda_applier.cu:578"), "stage wait");
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
    ap_check(counted_sync(tstream(), "cuda_applier.cu:601"), "books sync");
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
    // #29 P6b.3: the LOGIC, DOOR_INPUTS and SWAP stages read no plane
    const bool planeless = stage == APPLIER_STAGE_LOGIC || stage == APPLIER_STAGE_DOOR_INPUTS ||
                           stage == APPLIER_STAGE_SWAP;
    for (int k = 0; k < APPLIER_N_PLANES; ++k) {
        if (planes.ptr[k] == 0 && !planeless)
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
        // #29 P7c.1: the noise list and the blast log (object tables)
        if (lists->noise != nullptr) {
            if (lists->noise->n_env() != n || lists->noise->width() != object_tables::OBJ_NOISE_W)
                throw std::invalid_argument("applier_run: the noise list is for another env "
                                            "count or not the noise table's width");
            WL.has_noise = true;
            WL.noise = list_view(*lists->noise);
        }
        if (lists->blast_log != nullptr) {
            if (lists->blast_log->n_env() != n || lists->blast_log->width() != object_tables::OBJ_BLAST_W)
                throw std::invalid_argument("applier_run: the blast log is for another env "
                                            "count or not the blast table's width");
            WL.has_blast = true;
            WL.blast = list_view(*lists->blast_log);
        }
        // #29 P7c.2: the blast-source table and its tile table (both or none)
        if ((lists->source_table == nullptr) != (lists->source_tiles == nullptr))
            throw std::invalid_argument("applier_run: the blast-source table and its tile "
                                        "table come together");
        if (lists->source_table != nullptr) {
            if (lists->source_table->n_env() != n || lists->source_tiles->n_env() != n ||
                lists->source_table->width() != object_tables::OBJ_SOURCE_W ||
                lists->source_tiles->width() != object_tables::OBJ_SOURCE_TILE_W)
                throw std::invalid_argument("applier_run: the blast-source tables are not the "
                                            "source / source tile layouts for these envs");
            WL.has_sources = true;
            WL.sources = list_view(*lists->source_table);
            WL.source_tiles = list_view(*lists->source_tiles);
        }
    }
    // #29 P7c.2: every walk has its scratch (h * w cells per env)
    WL.walk = static_cast<uint8_t*>(a.walk_scratch(planes.h * planes.w));
    WL.walk_cells = a.walk_cells();
    WalkTables X;
    if (lists != nullptr && lists->tick_row != 0)    // #29 P7e.1: the tick row
        X.trow = reinterpret_cast<const int64_t*>(lists->tick_row);
    // #29 P6b.3: the signal stages' tables (each stage checks the ones it reads)
    const bool sig_stage = stage == APPLIER_STAGE_EMIT || stage == APPLIER_STAGE_LOGIC ||
                           stage == APPLIER_STAGE_DOOR_INPUTS || stage == APPLIER_STAGE_PUMPS ||
                           stage == APPLIER_STAGE_SWAP;
    if (sig_stage) {
        if (lists == nullptr)
            throw std::invalid_argument("applier_run: a signal stage needs its tables");
        ObjectTable* tabs[8] = {lists->signal_table, lists->sensor_table, lists->sensor_tiles,
                                lists->node_table, lists->node_ins, lists->pump_table,
                                lists->pump_ins, lists->door_drives};
        // which tables (and whether the door table) each stage reads
        bool need[8] = {true, false, false, false, false, false, false, false};
        bool need_doors = false;
        if (stage == APPLIER_STAGE_EMIT) {
            need[ST_SENSOR] = need[ST_STILE] = true;
            need_doors = true;
        }
        if (stage == APPLIER_STAGE_LOGIC) need[ST_NODE] = need[ST_NIN] = true;
        if (stage == APPLIER_STAGE_DOOR_INPUTS) {
            need[ST_DRIVE] = true;
            need_doors = true;
        }
        if (stage == APPLIER_STAGE_PUMPS) need[ST_PUMP] = need[ST_PIN] = true;
        for (int t = 0; t < 8; ++t) {
            if (!need[t]) continue;
            if (tabs[t] == nullptr)
                throw std::invalid_argument("applier_run: a signal stage's table is missing");
            if (tabs[t]->n_env() != a.n_env() || tabs[t]->width() != SIG_W[t])
                throw std::invalid_argument("applier_run: a signal table is not its layout for "
                                            "these envs");
            X.sig[t] = tabs[t]->records_dev();
            X.n_sig[t] = tabs[t]->count_dev();
            X.sig_cap[t] = tabs[t]->cap();
        }
        if (need_doors) {
            ObjectTable* doors = lists->door_table;
            if (doors == nullptr || doors->n_env() != a.n_env() ||
                doors->width() != object_tables::OBJ_DOOR_W)
                throw std::invalid_argument("applier_run: the EMIT / DOOR_INPUTS stage needs the "
                                            "door table");
            X.doors = doors->records_dev();
            X.n_doors = doors->count_dev();
            X.door_cap = doors->cap();
        }
    }
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
    if (stage == APPLIER_STAGE_VENTS) {
        ObjectTable* ducts = lists != nullptr ? lists->duct_table : nullptr;
        ObjectTable* vts = lists != nullptr ? lists->vent_table : nullptr;
        if (ducts == nullptr || vts == nullptr)
            throw std::invalid_argument("applier_run: the VENTS stage needs the duct table and "
                                        "the vent table");
        if (ducts->n_env() != a.n_env() || vts->n_env() != a.n_env() ||
            ducts->width() != object_tables::OBJ_DUCT_W ||
            vts->width() != object_tables::OBJ_VENT_W || vts->cap() > object_tables::OBJ_VENT_CAP)
            throw std::invalid_argument("applier_run: the duct / vent tables are not the duct / "
                                        "vent layouts for these envs");
        X.ducts = ducts->records_dev();
        X.n_ducts = ducts->count_dev();
        X.duct_cap = ducts->cap();
        X.vents = vts->records_dev();
        X.n_vents = vts->count_dev();
        X.vent_cap = vts->cap();
    }
    if (stage == APPLIER_STAGE_SOURCES) {
        if (lists == nullptr || lists->source_table == nullptr)
            throw std::invalid_argument("applier_run: the SOURCES stage needs the blast-source "
                                        "table and its tile table");
        X.sources = lists->source_table->records_dev();
        X.n_sources = lists->source_table->count_dev();
        X.source_cap = lists->source_table->cap();
        X.stiles = lists->source_tiles->records_dev();
        X.n_stiles = lists->source_tiles->count_dev();
        X.stile_cap = lists->source_tiles->cap();
        X.source_out = reinterpret_cast<int32_t*>(lists->source_out);
    }
    // #29 P7d: the HORDE stage's door tables (the bashable mask), its cache,
    // scratch and latch (device addresses; HordeReach)
    horde_dev::HordeBuffers HB;
    if (stage == APPLIER_STAGE_HORDE) {
        if (lists == nullptr || lists->door_table == nullptr || lists->door_tiles == nullptr ||
            lists->noise == nullptr || ring == nullptr)
            throw std::invalid_argument("applier_run: the HORDE stage needs the door tables, the "
                                        "noise list and the event ring");
        const HordeReach& R = lists->horde;
        if (R.hdr == 0 || R.keys == 0 || R.dist == 0 || R.masks == 0 || R.pass == 0 ||
            R.flags == 0 || R.queues == 0 || R.live == 0 || R.latch == 0 || R.fcap < 1)
            throw std::invalid_argument("applier_run: the HORDE stage needs the horde cache, its "
                                        "scratch and the latch");
        if (lists->door_table->width() != object_tables::OBJ_DOOR_W ||
            lists->door_tiles->width() != object_tables::OBJ_DOOR_TILE_W ||
            lists->door_table->n_env() != a.n_env() || lists->door_tiles->n_env() != a.n_env())
            throw std::invalid_argument("applier_run: the door tables are not the door / door "
                                        "tile layouts for these envs");
        X.doors = lists->door_table->records_dev();
        X.n_doors = lists->door_table->count_dev();
        X.door_cap = lists->door_table->cap();
        X.tiles = lists->door_tiles->records_dev();
        X.n_tiles = lists->door_tiles->count_dev();
        X.tile_cap = lists->door_tiles->cap();
        HB.hdr = reinterpret_cast<int64_t*>(R.hdr);
        HB.keys = reinterpret_cast<int64_t*>(R.keys);
        HB.dist = reinterpret_cast<int32_t*>(R.dist);
        HB.masks = reinterpret_cast<uint8_t*>(R.masks);
        HB.pass = reinterpret_cast<uint8_t*>(R.pass);
        HB.flags = reinterpret_cast<uint8_t*>(R.flags);
        HB.queues = reinterpret_cast<int32_t*>(R.queues);
        HB.live = reinterpret_cast<int64_t*>(R.live);
        HB.latch = reinterpret_cast<const int64_t*>(R.latch);
        HB.fcap = R.fcap;
    }
    // #29 P6b.4: the TIMED_CHARGES / HEAT_SOURCES stages' tables and own list
    if (stage == APPLIER_STAGE_TIMED_CHARGES || stage == APPLIER_STAGE_HEAT_SOURCES) {
        const bool timed = stage == APPLIER_STAGE_TIMED_CHARGES;
        ObjectTable* t = lists == nullptr ? nullptr
                                          : (timed ? lists->timed_table : lists->heat_table);
        if (t == nullptr)
            throw std::invalid_argument(timed ? "applier_run: the TIMED_CHARGES stage needs the "
                                                "timed-charge table"
                                              : "applier_run: the HEAT_SOURCES stage needs the "
                                                "heat-source table");
        if (t->n_env() != a.n_env() ||
            t->width() != (timed ? object_tables::OBJ_TIMED_CHARGE_W
                                 : object_tables::OBJ_HEAT_SOURCE_W))
            throw std::invalid_argument("applier_run: the timed-charge / heat-source table is "
                                        "not its layout for these envs");
        if (timed) {
            X.timed = t->records_dev();
            X.n_timed = t->count_dev();
            X.timed_cap = t->cap();
        } else {
            X.heat = t->records_dev();
            X.n_heat = t->count_dev();
            X.heat_cap = t->cap();
        }
        WL.has_own = true;
        WL.own = list_view(a.list());
    }
    UnitRecordList& L = a.list();
    const int threads = stage == APPLIER_STAGE_HORDE ? horde_dev::HORDE_BT : 32;
    k_applier<<<dim3(1, (unsigned)a.n_env()), threads, 0, tstream()>>>(
        stage, L.records_dev(), L.count_dev(), L.dropped_dev(), L.cap(), P, planes.h, planes.w,
        planes.n_gases, tables.values(), a.books_dev(), U, WL, X, HB);
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
        reinterpret_cast<const uint8_t*>(active));
    ap_check(cudaGetLastError(), "clears launch");
}

const char* applier_fingerprint() { return APPLIER_FINGERPRINT; }

namespace {
__global__ void k_blast_probe(topo::PlanePtrs P, int h, int w, int n_gases,
                              const int64_t* tables, const int64_t* q, int K, int M,
                              int64_t* out, uint8_t* walk, int walk_cells) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = benv();
    const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
    const topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
    const size_t nc = (size_t)walk_cells, N = gridDim.y;
    unit_blast::Scratch S;
    S.mark = walk + (size_t)e * nc;
    int32_t* lists = reinterpret_cast<int32_t*>(walk + ((N * nc + 7) & ~(size_t)7));
    S.queue = lists + ((size_t)0 * N + e) * nc;
    S.heat = lists + ((size_t)1 * N + e) * nc;
    S.src = lists + ((size_t)2 * N + e) * nc;
    const int W = 3 + 2 * M;
    for (int k = 0; k < K; ++k) {
        const int64_t* a = q + ((size_t)e * K + k) * 7;
        int64_t* o = out + ((size_t)e * K + k) * W;
        int n = 0;
        if (a[6] == 0)
            n = unit_blast::reach(E, a[0], a[1], a[2], a[3] != 0, a[4], a[5], S, S.heat);
        else
            n = unit_blast::disc_tiles(E, a[0], a[1], a[2], S.heat);
        bool one = true;
        o[1] = unit_blast::air_read(E, tab, S.heat, n, &one);
        o[0] = n;
        o[2] = one ? 1 : 0;
        for (int j = 0; j < n && j < M; ++j) {
            o[3 + j] = S.heat[j];
            o[3 + M + j] = unit_blast::air_smoke_q(E, tab, S.heat[j]);
        }
    }
}
}  // namespace

void blast_probe(FieldApplier& a, const UnitTablesDevice& tables, const ApplierPlanes& planes,
                 const int64_t* q, int K, int M, int64_t* out) {
    const int N = a.n_env();
    if (tables.n_env() != N || K < 1 || M < 1)
        throw std::invalid_argument("blast_probe: the tables are for another env count, or K / M < 1");
    topo::PlanePtrs P;
    for (int k = 0; k < APPLIER_N_PLANES; ++k) {
        if (planes.ptr[k] == 0) throw std::invalid_argument("blast_probe: a plane pointer is null");
        P.p[k] = (unsigned long long)planes.ptr[k];
    }
    uint8_t* walk = static_cast<uint8_t*>(a.walk_scratch(planes.h * planes.w));
    const size_t nq = (size_t)N * K * 7, no = (size_t)N * K * (3 + 2 * M);
    int64_t *dq = nullptr, *dout = nullptr;
    ap_check(cudaMalloc(reinterpret_cast<void**>(&dq), nq * 8), "probe malloc q");
    ap_check(cudaMalloc(reinterpret_cast<void**>(&dout), no * 8), "probe malloc out");
    ap_check(cudaMemcpyAsync(dq, q, nq * 8, cudaMemcpyHostToDevice, tstream()), "probe H2D");
    ap_check(cudaMemsetAsync(dout, 0, no * 8, tstream()), "probe zero");
    k_blast_probe<<<dim3(1, (unsigned)N), 32, 0, tstream()>>>(
        P, planes.h, planes.w, planes.n_gases, tables.values(), dq, K, M, dout, walk,
        a.walk_cells());
    ap_check(cudaGetLastError(), "blast probe launch");
    ap_check(cudaMemcpyAsync(out, dout, no * 8, cudaMemcpyDeviceToHost, tstream()), "probe D2H");
    ap_check(counted_sync(tstream(), "cuda_applier.cu:945"), "probe sync");
    cudaFree(dq);
    cudaFree(dout);
}

}  // namespace breach_cuda
