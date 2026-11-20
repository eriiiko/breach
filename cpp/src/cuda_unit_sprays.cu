// ============================================================================
// #29 P7a.3 -- SLOT 4b, THE SPRAYS on the device unit store (combat.
// process_sprays: the second half of the attack slot's phase A), the effect
// list's third producer (after slot 4's shots and 4a's blows). See
// cuda_unit_kernels.h; residency design 4.4c (the cut table's P7a.3 row,
// decision 1's SPRAY_CELLS).
//
// Integer for integer the reference, per valid row with spray_ticks_left > 0,
// in ROW (= id) order (ONE block per env: thread 0 walks the rows, the block's
// threads split each cone's box):
//   * a dead row: the burst dies with it (ticks_left 0, the target and the
//     aim cleared), nothing recorded;
//   * composed can_act False: the interruption, the same clear (the WEGO
//     fire order's consumption has no device form: the continuous burst rides
//     none -- the door refuses WEGO orders);
//   * otherwise the cone (unit_spray::cone_params over the PHYSICAL sight,
//     against the walls standing at the slot's start) decided over its box,
//     its cells appended IN ROW-MAJOR ORDER to THE CELL BUFFER and ONE
//     SPRAY_CELLS record (owner, the weapon's heat_q / gas slice / gas_q
//     rows, the cells' span) appended to THE EFFECT LIST after what it holds;
//     then the countdown at once (and the clear when it reaches 0).
// The render-only SprayJetEvent has no device form. No draw.
// REFUSED (counted apart, named by the Python face): a burst with no aim
// angle -- the WEGO start_spray_burst's tile-derived bearing, out of scope.
// FAULTS (the row left as it was, nothing appended): a weapon row outside the
// tables or with no round (ammo_for_weapon would raise), a cone outside the
// proved range (cuda_unit_spray.cuh), a box larger than the launch's scratch.
//
// unit_spray_cone_probe (TEST-ONLY): the same decide / emit pair on synthetic
// cones, one block per query.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_unit_kernels.h"
#include "cuda_unit_spray.cuh"

#include <memory>
#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

using namespace unit_lib;
using namespace applier;
using unit_spray::ConeParams;

constexpr int SPRAY_THREADS = 128;
//: The decisions of decide_row.
constexpr int D_NONE = 0, D_CONE = 1, D_FAULT = 2, D_REFUSE = 3, D_WRITE = 4;

// A grown, never shrunk device scratch (no allocation in a steady state).
struct Scratch {
    int64_t* d = nullptr;
    size_t cap = 0;
    int64_t* get(size_t n) {
        if (n < 1) n = 1;
        if (n > cap) {
            counted_sync(tstream(), "cuda_unit_sprays.cu:61");
            if (d) cudaFree(d);
            d = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&d), n * sizeof(int64_t)) != cudaSuccess)
                throw std::runtime_error("unit_sprays: scratch allocation failed");
            cap = n;
        }
        return d;
    }
};
Scratch g_box;      // (N, box_max): the decided box of the cone in hand
Scratch g_pbox;     // the probe's (N * R, box_max)
Scratch g_pout;     // the probe's outputs

__device__ __forceinline__ void clear_burst(int64_t* loc) {
    loc[UNIT_COL_SPRAY_TICKS_LEFT] = 0;
    for (int k = 0; k < UNIT_COL_SPRAY_TARGET_W; ++k) loc[UNIT_COL_SPRAY_TARGET + k] = 0;
    for (int k = 0; k < UNIT_COL_SPRAY_AIM_ANGLE_W; ++k) loc[UNIT_COL_SPRAY_AIM_ANGLE + k] = 0;
}

// Thread 0's decision for one row (process_sprays' loop body before the
// deposit). `loc` holds the row's new state for D_WRITE; for D_CONE the cone
// and the weapon row `m`.
__device__ inline int decide_row(const int64_t* row, const int64_t* tab, int h, int w,
                                 int64_t* loc, ConeParams& P, int64_t& m) {
    if (!row_valid(row) || row[UNIT_COL_SPRAY_TICKS_LEFT] <= 0) return D_NONE;
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    if (!row_alive(row)) {                       // the burst dies with the unit
        clear_burst(loc);
        return D_WRITE;
    }
    bool bad = false;
    const bool act = can_act(row, tab, &bad);
    if (bad) return D_FAULT;
    if (!act) {                                  // the interruption
        clear_burst(loc);
        return D_WRITE;
    }
    if (row[UNIT_COL_SPRAY_AIM_ANGLE] == 0) return D_REFUSE;   // WEGO's tile bearing
    m = row[UNIT_COL_WEAPON_ID];
    if (m < 0 || m >= unit_tables::UNIT_TAB_SPRAY_HAS_AMMO_W ||
        tab[unit_tables::UNIT_TAB_SPRAY_HAS_AMMO + m] == 0)
        return D_FAULT;                          // weapons.by_name / ammo_for_weapon raise
    if (!unit_spray::cone_params(row, tab, m, row[UNIT_COL_SPRAY_AIM_ANGLE + 1], h, w, P, &bad) ||
        bad)
        return D_FAULT;
    return D_CONE;
}

__global__ void k_sprays(int64_t* store, int n_units, const int64_t* tables, const uint8_t* solid,
                         int h, int w, RecordListView fx, RecordListView cells, int64_t* box,
                         int64_t box_max, unsigned long long* cnt) {
    const int e = benv();
    const int64_t* tab = env_tables(tables);
    const uint8_t* S = bptr(solid, (size_t)h * w);
    int64_t* dec = box + (size_t)e * box_max;
    __shared__ int s_what;
    __shared__ ConeParams s_P;
    __shared__ int64_t s_m;
    __shared__ int64_t s_loc[UNIT_STORE_W];
    for (int r = 0; r < n_units; ++r) {
        int64_t* row = unit_row(store, n_units, r);
        if (threadIdx.x == 0) {
            ConeParams P;
            int64_t m = 0;
            int what = decide_row(row, tab, h, w, s_loc, P, m);
            if (what == D_CONE && unit_spray::box_size(P) > box_max) what = D_FAULT;
            s_what = what;
            s_P = P;
            s_m = m;
        }
        __syncthreads();
        const int what = s_what;
        if (what == D_CONE) unit_spray::decide_box(s_P, S, h, w, dec);
        __syncthreads();
        if (threadIdx.x == 0) {
            int final = what;
            if (what == D_CONE) {
                RecordSink cs = list_env_sink(cells, e);
                const int64_t start = *cs.count;
                if (!unit_spray::emit_box(s_P, dec, cs)) {
                    final = D_FAULT;
                } else {
                    const int64_t m = s_m;
                    int64_t rec[APPLIER_F];
                    for (int f = 0; f < APPLIER_F; ++f) rec[f] = 0;
                    rec[APPLIER_FX_KIND] = APPLIER_KIND_SPRAY_CELLS;
                    rec[APPLIER_SPRAY_CELLS_OWNER] = row[UNIT_COL_ID];
                    rec[APPLIER_SPRAY_CELLS_HEAT_Q] = tab[unit_tables::UNIT_TAB_SPRAY_HEAT_Q + m];
                    rec[APPLIER_SPRAY_CELLS_GAS_ID] = tab[unit_tables::UNIT_TAB_SPRAY_GAS_ID + m];
                    rec[APPLIER_SPRAY_CELLS_GAS_Q] = tab[unit_tables::UNIT_TAB_SPRAY_GAS_Q + m];
                    rec[APPLIER_SPRAY_CELLS_START] = start;
                    rec[APPLIER_SPRAY_CELLS_COUNT] = *cs.count - start;
                    RecordSink es = list_env_sink(fx, e);
                    emit_record(es, rec);
                    // the countdown, at once
                    for (int c = 0; c < UNIT_STORE_W; ++c) s_loc[c] = row[c];
                    s_loc[UNIT_COL_SPRAY_TICKS_LEFT] = row[UNIT_COL_SPRAY_TICKS_LEFT] - 1;
                    if (s_loc[UNIT_COL_SPRAY_TICKS_LEFT] <= 0) clear_burst(s_loc);
                    final = D_WRITE;
                }
            }
            if (final == D_FAULT) {
                count_fault(cnt);
            } else if (final == D_REFUSE) {
                count_refused(cnt);
            } else if (final == D_WRITE) {
                bool changed = false;
                for (int c = 0; c < UNIT_STORE_W; ++c) put(row, c, s_loc[c], changed);
                if (changed) count_changed(cnt);
            }
        }
        __syncthreads();
    }
}

// TEST-ONLY: one block per (query, env). args (N, R, 8): ay, ax, angle_q,
// range, half_q, ex_x, ex_y, fp. out (N, R, max_out, 3) the cells in order;
// meta (N, R, 2): the cell count (all of them, even past max_out), bad.
__global__ void k_cone_probe(const uint8_t* solid, int h, int w, const int64_t* args, int n_rows,
                             int64_t* box, int64_t box_max, int64_t* out, int max_out,
                             int64_t* meta) {
    const int e = benv();
    const int q = (int)blockIdx.x;
    const size_t k = (size_t)e * n_rows + q;
    const int64_t* a = args + k * 8;
    const uint8_t* S = bptr(solid, (size_t)h * w);
    int64_t* dec = box + k * box_max;
    __shared__ ConeParams s_P;
    __shared__ int s_ok;
    if (threadIdx.x == 0) {
        ConeParams P;
        bool bad = false;
        const bool ok = unit_spray::cone_from(a[0], a[1], a[5], a[6], a[7], a[2], a[3], a[4], h,
                                              w, P, &bad);
        s_ok = ok && !bad && unit_spray::box_size(P) <= box_max;
        s_P = P;
    }
    __syncthreads();
    if (s_ok) unit_spray::decide_box(s_P, S, h, w, dec);
    __syncthreads();
    if (threadIdx.x != 0) return;
    int64_t* M = meta + k * 2;
    M[0] = 0;
    M[1] = s_ok ? 0 : 1;
    if (!s_ok) return;
    __shared__ int64_t n;
    __shared__ unsigned long long past;   // cells beyond max_out (counted, not written;
    n = 0;                                // shared: emit_record's atomicAdd needs it)
    past = 0;
    RecordSink sink{out + k * (size_t)max_out * 3, max_out, &n, &past, 3};
    if (!unit_spray::emit_box(s_P, dec, sink)) {
        M[1] = 1;
        return;
    }
    M[0] = n + (int64_t)past;
}

inline void sp_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in " << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

}  // namespace

void unit_sprays(UnitStoreDevice& store, const UnitTablesDevice& tables, FieldApplier& effects,
                 UnitRecordList& cells, std::uintptr_t solid, int h, int w, int64_t box_max,
                 const UnitDamageCounts& out) {
    unit_check_pair(store, tables, "unit_sprays");
    UnitRecordList& L = effects.list();
    if (L.n_env() != store.n_env() || L.width() != APPLIER_F)
        throw std::invalid_argument("unit_sprays: the effect list is for another store / layout");
    if (cells.n_env() != store.n_env() || cells.width() != unit_spray::CELL_W)
        throw std::invalid_argument("unit_sprays: the cell buffer is for another env count or "
                                    "is not (y, x, div) wide");
    if (solid == 0 || h < 1 || w < 1 || box_max < 1)
        throw std::invalid_argument("unit_sprays: the solid plane, its shape and box_max");
    const int n = store.n_env(), U = store.n_units();
    int64_t* box = g_box.get((size_t)n * box_max);
    const RecordListView fx = list_view(L), cv = list_view(cells);
    store.begin_counts();
    k_sprays<<<dim3(1, n), SPRAY_THREADS, 0, tstream()>>>(
        store.rows(), U, tables.values(), reinterpret_cast<const uint8_t*>(solid), h, w, fx, cv,
        box, box_max, store.counts_dev());
    sp_check(cudaGetLastError(), "unit_sprays");
    store.end_counts(out.coverage, out.faults, out.refused);
    if (out.dropped) {
        // the effect list's and the cell buffer's dropped records, summed per env
        std::unique_ptr<int64_t[]> a(new int64_t[(size_t)n]), b(new int64_t[(size_t)n]);
        L.download(nullptr, nullptr, a.get());
        cells.download(nullptr, nullptr, b.get());
        for (int e = 0; e < n; ++e) out.dropped[e] = a[e] + b[e];
    }
}

void unit_spray_cone_probe(std::uintptr_t solid, int n_env, int h, int w, const int64_t* args,
                           int n_rows, int64_t box_max, int max_out, int64_t* out,
                           int64_t* meta) {
    if (solid == 0 || n_env < 1 || h < 1 || w < 1 || n_rows < 1 || box_max < 1 || max_out < 1)
        throw std::invalid_argument("unit_spray_cone_probe: empty input");
    const size_t nq = (size_t)n_env * n_rows;
    int64_t* box = g_pbox.get(nq * box_max);
    int64_t* d = g_pout.get(nq * 8 + nq * (size_t)max_out * 3 + nq * 2);
    int64_t* d_args = d;
    int64_t* d_out = d + nq * 8;
    int64_t* d_meta = d_out + nq * (size_t)max_out * 3;
    sp_check(cudaMemcpyAsync(d_args, args, nq * 8 * sizeof(int64_t), cudaMemcpyHostToDevice,
                             tstream()), "probe H2D");
    k_cone_probe<<<dim3((unsigned)n_rows, (unsigned)n_env), SPRAY_THREADS, 0, tstream()>>>(
        reinterpret_cast<const uint8_t*>(solid), h, w, d_args, n_rows, box, box_max, d_out,
        max_out, d_meta);
    sp_check(cudaGetLastError(), "unit_spray_cone_probe");
    sp_check(cudaMemcpyAsync(out, d_out, nq * (size_t)max_out * 3 * sizeof(int64_t),
                             cudaMemcpyDeviceToHost, tstream()), "probe D2H");
    sp_check(cudaMemcpyAsync(meta, d_meta, nq * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost,
                             tstream()), "probe D2H meta");
    sp_check(counted_sync(tstream(), "cuda_unit_sprays.cu:280"), "probe sync");
}

}  // namespace breach_cuda
