// ============================================================================
// #29 P7e.2 -- THE DEVICE SORTS AND 9b (residency design, "P7e RE-CUT
// (2026-11-19)", the P7e.2 row; inventory docs/residency_p7e_inventory_
// 2026-11-19.md parts A, E, H.2): three producers that leave the host out of
// slots 6, 9 and 9b. Each is integer for integer its reference; CLASS T
// (training only): nothing in the played tick calls this TU.
//
// 1. THE BURN-THROUGH SORT (slot 9's input). Slot 7's fire writes the walls
//    that burned through as packed in-env linear indices in atomic ARRIVAL
//    order (cuda_fire.cu); the host sorted them ascending
//    (fire_destroyed_row_major: ascending linear index == the CPU fire's
//    row-major walk), and GameMap.destroy_wall is order-dependent. Here one
//    block per env RANKS every index (rank = the indices below it + the equal
//    ones before it: a stable rank, ties included) and writes its DESTROY_WALL
//    record straight to that rank's slot of the applier's effect list, AFTER
//    what the list holds; past the list's cap the rest are counted dropped (the
//    walk refuses the run by name: EffectOverflow). An index outside the grid,
//    or a count outside [0, stride], FAULTS the env and writes nothing.
//
// 2. SLOT 9b (GameMap.find_burst_walls + Simulation._burst_walls' destroy loop).
//    For each solid tile whose material's burst row is live: the spread
//    max - min of its open 4-neighbours' pressures (a solid neighbour is no
//    side; an is_vacuum side holds 0, an is_ambient side the pin, else the
//    atmosphere), in raw Q16 int64 -- exactly the float64 spread times 2^16
//    (unit_tables' burst_thresh_q has the proof) --; a tile bursts when it has
//    a side and spread > burst_thresh_q[material]. The reference sorts the
//    bursting tiles by spread DESCENDING, ties in the row-major scan order
//    (a stable argsort), and keeps the first burst_cap: here one block per
//    env takes them one at a time, each round the maximum under that total
//    order among the tiles strictly after the previous pick (a block
//    reduction), and appends its DESTROY_WALL record. The records are walked
//    by the TOPOLOGY stage (the one applier, the topology seam, the tick
//    row's tile count). A material id outside the table FAULTS the env and
//    nothing is appended.
//
// 3. THE STAMP FROM THE UNIT STORE (slot 6, GameMap._stamp_units_cpp). The
//    rows the stamp kernel (cuda_stamp.cu) consumes, built from the store in
//    the host stamp's row order -- per env, unit row order, then the
//    footprint's row-major tiles (the tables' footprint_rowmajor row), the
//    living, valid units only, the in-bounds tiles only -- compacted through
//    a per-(env, row) tile count and a prefix over (env, row); the slots past
//    the total are written env = -1 (the stamp kernel skips them), so the
//    launch takes the capacity and the host never reads the count. The
//    per-unit values are the map defaults (stamp_* rows; a sim with an optics
//    override is refused at the table build). Then THE REFILL
//    (_stamp_units_cpp's atmosphere refill): `freed` = the walls of the last
//    stamp that are no longer walls (prev obstacles and not permeability <= 0)
//    and not vacuum, captured BEFORE the stamp's reset; after the stamp, each
//    freed tile in row-major order takes _neighbor_mean of the atmosphere
//    (the open, non-vacuum 4-neighbours, an integer mean truncated toward
//    zero), serially -- a later tile reads an earlier one's refilled value, as
//    the host loop does: one block per env walks the grid in chunks, thread 0
//    walking a chunk only when some tile of it is freed.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"
#include "cuda_applier.h"
#include "cuda_resident.h"     // stamp_launch_resident (cuda_stamp.cu)

#include <stdexcept>

namespace breach_cuda {

namespace {

using namespace unit_lib;
using namespace applier;

constexpr int SORT_BT = 256;

inline void so_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("unit sorts: ") + what + ": " +
                                 cudaGetErrorString(e));
}

__device__ inline void write_destroy(int64_t* rec, int64_t y, int64_t x) {
    for (int f = 0; f < APPLIER_F; ++f) rec[f] = 0;
    rec[0] = APPLIER_KIND_DESTROY_WALL;
    rec[APPLIER_DESTROY_WALL_Y] = y;
    rec[APPLIER_DESTROY_WALL_X] = x;
}

// ---- 1. the burn-through sort ------------------------------------------------------
__global__ void k_burn_sort(const int* counts, const int* lists, int stride, int h, int w,
                            RecordListView L, unsigned long long* cnt) {
    const int e = benv();
    const int n_cells = h * w;
    __shared__ int64_t base;
    __shared__ int n;
    if (threadIdx.x == 0) {
        base = L.count[e];
        n = counts[e];
    }
    __syncthreads();
    const int* l = lists + (size_t)e * stride;
    int bad = (n < 0 || n > stride) ? 1 : 0;
    if (!bad)
        for (int k = threadIdx.x; k < n; k += blockDim.x)
            if (l[k] < 0 || l[k] >= n_cells) bad = 1;
    if (__syncthreads_or(bad)) {
        if (threadIdx.x == 0) count_fault(cnt);
        return;
    }
    int64_t* recs = L.recs + (size_t)e * L.cap * L.width;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        const int v = l[k];
        int64_t rank = 0;
        for (int j = 0; j < n; ++j) {
            const int u = l[j];
            rank += (u < v || (u == v && j < k)) ? 1 : 0;
        }
        const int64_t slot = base + rank;
        if (slot < L.cap) write_destroy(recs + slot * L.width, v / w, v % w);
        count_changed(cnt);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int64_t total = base + n;
        if (total > L.cap) {
            atomicAdd(L.dropped + e, (unsigned long long)(total - L.cap));
            L.count[e] = L.cap;
        } else {
            L.count[e] = total;
        }
    }
}

// ---- 2. slot 9b ------------------------------------------------------------------------
struct BurstPlanes {
    const int8_t* material;
    const uint8_t* solid;
    const uint8_t* vacuum;
    const uint8_t* ambient;
    const int32_t* atm;
};

// The tile's spread when it bursts (the reference's has_side & spread > t).
__device__ inline bool burst_spread(const BurstPlanes& P, const int64_t* tab, int h, int w, int i,
                                    int64_t* spread) {
    if (!P.solid[i]) return false;
    const int64_t t = tab[UNIT_TAB_BURST_THRESH_Q + (int)P.material[i]];
    if (t < 0) return false;
    const int y = i / w, x = i % w;
    const int dys[4] = {-1, 1, 0, 0}, dxs[4] = {0, 0, -1, 1};
    int64_t lo = 0, hi = 0;
    bool side = false;
    for (int k = 0; k < 4; ++k) {
        const int ny = y + dys[k], nx = x + dxs[k];
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
        const int j = ny * w + nx;
        if (P.solid[j]) continue;
        const int64_t p = P.vacuum[j] ? 0
                          : P.ambient[j] ? tab[UNIT_TAB_BURST_P_AMB_Q] : (int64_t)P.atm[j];
        if (!side) {
            lo = hi = p;
            side = true;
        } else {
            lo = p < lo ? p : lo;
            hi = p > hi ? p : hi;
        }
    }
    if (!side) return false;
    *spread = hi - lo;
    return *spread > t;
}

// (s, i) before (bs, bi) in the reference's order: a larger spread, or an
// equal one earlier in the row-major scan.
__device__ __forceinline__ bool before(int64_t s, int i, int64_t bs, int bi) {
    return bi < 0 || s > bs || (s == bs && i < bi);
}

__global__ void k_burst(const int64_t* tables, BurstPlanes P0, int h, int w, RecordListView L,
                        unsigned long long* cnt) {
    const int e = benv();
    const int n = h * w;
    const int64_t* tab = env_tables(tables);
    BurstPlanes P;
    P.material = P0.material + (size_t)e * n;
    P.solid = P0.solid + (size_t)e * n;
    P.vacuum = P0.vacuum + (size_t)e * n;
    P.ambient = P0.ambient + (size_t)e * n;
    P.atm = P0.atm + (size_t)e * n;
    int bad = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (!P.solid[i]) continue;
        const int m = (int)P.material[i];
        if (m < 0 || m >= UNIT_TAB_BURST_THRESH_Q_W) bad = 1;
    }
    if (__syncthreads_or(bad)) {
        if (threadIdx.x == 0) count_fault(cnt);
        return;
    }
    __shared__ int64_t bs[SORT_BT];
    __shared__ int bi[SORT_BT];
    __shared__ int64_t prev_s;
    __shared__ int prev_i;
    if (threadIdx.x == 0) {
        prev_s = 0;
        prev_i = -1;
    }
    __syncthreads();
    RecordSink sink = list_env_sink(L, e);
    const int64_t cap = tab[UNIT_TAB_BURST_CAP];
    for (int64_t r = 0; r < cap; ++r) {
        int64_t my_s = 0;
        int my_i = -1;
        const int64_t ps = prev_s;
        const int pi = prev_i;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            int64_t s;
            if (!burst_spread(P, tab, h, w, i, &s)) continue;
            // strictly after the previous pick in the order
            if (pi >= 0 && !(s < ps || (s == ps && i > pi))) continue;
            if (before(s, i, my_s, my_i)) {
                my_s = s;
                my_i = i;
            }
        }
        bs[threadIdx.x] = my_s;
        bi[threadIdx.x] = my_i;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if ((int)threadIdx.x < off) {
                const int o = threadIdx.x + off;
                if (bi[o] >= 0 && before(bs[o], bi[o], bs[threadIdx.x], bi[threadIdx.x])) {
                    bs[threadIdx.x] = bs[o];
                    bi[threadIdx.x] = bi[o];
                }
            }
            __syncthreads();
        }
        const int pick = bi[0];
        if (pick < 0) break;                    // every bursting tile taken
        if (threadIdx.x == 0) {
            int64_t rec[APPLIER_F];
            write_destroy(rec, pick / w, pick % w);
            emit_record(sink, rec);
            count_changed(cnt);
            prev_s = bs[0];
            prev_i = pick;
        }
        __syncthreads();
    }
}

// ---- 3. the stamp rows and the refill --------------------------------------------------
// The tiles row r stamps: its fp x fp footprint at (tile_of(x), tile_of(y)),
// in bounds. A footprint outside [1, FOOTPRINT_MAX] faults (no tile).
__device__ inline int stamp_tiles(const int64_t* row, int h, int w, int64_t* ax, int64_t* ay,
                                  int64_t* fp, bool* bad) {
    if (!row_valid(row) || !row_alive(row)) return 0;
    *fp = row[UNIT_COL_FOOTPRINT];
    if (*fp < 1 || *fp > FOOTPRINT_MAX) {
        *bad = true;
        return 0;
    }
    *ax = tile_of(row[UNIT_COL_X_Q]);
    *ay = tile_of(row[UNIT_COL_Y_Q]);
    int k = 0;
    for (int64_t dy = 0; dy < *fp; ++dy)
        for (int64_t dx = 0; dx < *fp; ++dx) {
            const int64_t ty = *ay + dy, tx = *ax + dx;
            if (ty >= 0 && ty < h && tx >= 0 && tx < w) ++k;
        }
    return k;
}

__global__ void k_stamp_count(const int64_t* store, int U, int h, int w, int64_t* ntiles,
                              unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= U) return;
    const int64_t* row = unit_row(const_cast<int64_t*>(store), U, r);
    int64_t ax = 0, ay = 0, fp = 0;
    bool bad = false;
    ntiles[(size_t)benv() * U + r] = stamp_tiles(row, h, w, &ax, &ay, &fp, &bad);
    if (bad) count_fault(cnt);
}

// The exclusive prefix over (env, row) -- N * U entries, one thread: the
// stamp's row order is env-major, then unit order -- and the total.
__global__ void k_stamp_prefix(int64_t* ntiles, int total_rows, int64_t* total) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int64_t run = 0;
    for (int k = 0; k < total_rows; ++k) {
        const int64_t c = ntiles[k];
        ntiles[k] = run;
        run += c;
    }
    *total = run;
}

__global__ void k_stamp_pad(int32_t* rows, int cap, const int64_t* total) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= cap || (int64_t)r < *total) return;
    int32_t* o = rows + (size_t)r * STAMP_ROW_COLS;
    o[0] = -1;
    for (int c = 1; c < STAMP_ROW_COLS; ++c) o[c] = 0;
}

__global__ void k_stamp_fill(const int64_t* store, const int64_t* tables, int U, int h, int w,
                             const int64_t* offsets, int32_t* rows, int cap) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= U) return;
    const int e = benv();
    const int64_t* row = unit_row(const_cast<int64_t*>(store), U, r);
    const int64_t* tab = env_tables(tables);
    int64_t ax = 0, ay = 0, fp = 0;
    bool bad = false;
    if (stamp_tiles(row, h, w, &ax, &ay, &fp, &bad) == 0) return;
    int64_t at = offsets[(size_t)e * U + r];
    for (int64_t dy = 0; dy < fp; ++dy)
        for (int64_t dx = 0; dx < fp; ++dx) {
            const int64_t ty = ay + dy, tx = ax + dx;
            if (!(ty >= 0 && ty < h && tx >= 0 && tx < w)) continue;
            if (at < cap) {
                int32_t* o = rows + (size_t)at * STAMP_ROW_COLS;
                o[0] = e;
                o[1] = (int32_t)ty;
                o[2] = (int32_t)tx;
                o[3] = (int32_t)tab[UNIT_TAB_STAMP_PERM_BITS];
                o[4] = (int32_t)tab[UNIT_TAB_STAMP_WABSORB_BITS];
                o[5] = (int32_t)tab[UNIT_TAB_STAMP_HEAT_Q];
                o[6] = (int32_t)tab[UNIT_TAB_STAMP_LIGHT_Q + 0];
                o[7] = (int32_t)tab[UNIT_TAB_STAMP_LIGHT_Q + 1];
                o[8] = (int32_t)tab[UNIT_TAB_STAMP_LIGHT_Q + 2];
            }
            ++at;
        }
}

// freed = prev obstacles & ~(permeability <= 0) & ~is_vacuum, BEFORE the reset.
__global__ void k_stamp_freed(const uint8_t* obstacles, const float* perm, const uint8_t* vacuum,
                              uint8_t* freed, int total) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    freed[i] = (obstacles[i] && !(perm[i] <= 0.0f) && !vacuum[i]) ? 1 : 0;
}

__global__ void k_stamp_refill(const uint8_t* freed, const uint8_t* solid, const uint8_t* vacuum,
                               int32_t* atm, int h, int w, unsigned long long* cnt) {
    const int e = benv();
    const int n = h * w;
    const uint8_t* fr = freed + (size_t)e * n;
    const uint8_t* so = solid + (size_t)e * n;
    const uint8_t* va = vacuum + (size_t)e * n;
    int32_t* a = atm + (size_t)e * n;
    for (int c0 = 0; c0 < n; c0 += blockDim.x) {
        const int i = c0 + (int)threadIdx.x;
        const int f = (i < n && fr[i]) ? 1 : 0;
        if (__syncthreads_or(f) && threadIdx.x == 0) {
            const int c1 = c0 + (int)blockDim.x < n ? c0 + (int)blockDim.x : n;
            for (int j = c0; j < c1; ++j) {
                if (!fr[j]) continue;
                const int y = j / w, x = j % w;
                const int dys[4] = {-1, 1, 0, 0}, dxs[4] = {0, 0, -1, 1};
                int64_t total = 0, count = 0;
                for (int k = 0; k < 4; ++k) {
                    const int ny = y + dys[k], nx = x + dxs[k];
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                    const int q = ny * w + nx;
                    if (so[q] || va[q]) continue;
                    total += (int64_t)a[q];
                    ++count;
                }
                a[j] = count > 0 ? (int32_t)(total / count) : 0;   // truncation toward zero
                count_changed(cnt);
            }
        }
        __syncthreads();
    }
}

}  // namespace

void unit_burn_sort(UnitStoreDevice& store, FieldApplier& a, std::uintptr_t counts,
                    std::uintptr_t lists, int stride, int h, int w, const UnitKernelCounts& out) {
    const int n = store.n_env();
    if (a.n_env() != n || a.list().width() != APPLIER_F)
        throw std::invalid_argument("unit_burn_sort: the applier is not the store's envs");
    if (counts == 0 || lists == 0 || stride < 1 || h < 1 || w < 1)
        throw std::invalid_argument("unit_burn_sort: the counts (N,) int and lists (N, stride) "
                                    "int device arrays");
    store.begin_counts();
    k_burn_sort<<<dim3(1, (unsigned)n), SORT_BT, 0, tstream()>>>(
        reinterpret_cast<const int*>(counts), reinterpret_cast<const int*>(lists), stride, h, w,
        list_view(a.list()), store.counts_dev());
    so_check(cudaGetLastError(), "burn sort launch");
    store.end_counts(out.coverage, out.faults, out.refused);
}

void unit_burst_walls(UnitStoreDevice& store, const UnitTablesDevice& tables, FieldApplier& a,
                      const BurstScanPlanes& planes, const UnitKernelCounts& out) {
    const int n = store.n_env();
    unit_check_pair(store, tables, "unit_burst_walls");
    if (a.n_env() != n || a.list().width() != APPLIER_F)
        throw std::invalid_argument("unit_burst_walls: the applier is not the store's envs");
    if (!planes.material || !planes.solid || !planes.is_vacuum || !planes.is_ambient ||
        !planes.atmosphere || planes.h < 1 || planes.w < 1)
        throw std::invalid_argument("unit_burst_walls: a null plane or an empty grid");
    BurstPlanes P;
    P.material = reinterpret_cast<const int8_t*>(planes.material);
    P.solid = reinterpret_cast<const uint8_t*>(planes.solid);
    P.vacuum = reinterpret_cast<const uint8_t*>(planes.is_vacuum);
    P.ambient = reinterpret_cast<const uint8_t*>(planes.is_ambient);
    P.atm = reinterpret_cast<const int32_t*>(planes.atmosphere);
    store.begin_counts();
    k_burst<<<dim3(1, (unsigned)n), SORT_BT, 0, tstream()>>>(
        tables.values(), P, planes.h, planes.w, list_view(a.list()), store.counts_dev());
    so_check(cudaGetLastError(), "burst launch");
    store.end_counts(out.coverage, out.faults, out.refused);
}

void unit_stamp_rows(UnitStoreDevice& store, const UnitTablesDevice& tables,
                     const StampScratch& s, int h, int w, const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_stamp_rows");
    const int n = store.n_env(), U = store.n_units();
    if (!s.rows || !s.total || !s.ntiles || s.cap < 1 || h < 1 || w < 1)
        throw std::invalid_argument("unit_stamp_rows: the rows (cap, 9) int32, the total and "
                                    "the (N, U) int64 scratch");
    int64_t* ntiles = reinterpret_cast<int64_t*>(s.ntiles);
    int64_t* total = reinterpret_cast<int64_t*>(s.total);
    int32_t* rows = reinterpret_cast<int32_t*>(s.rows);
    store.begin_counts();
    k_stamp_count<<<bgrid(U, n), 64, 0, tstream()>>>(store.rows(), U, h, w, ntiles,
                                                      store.counts_dev());
    k_stamp_prefix<<<1, 1, 0, tstream()>>>(ntiles, n * U, total);
    k_stamp_pad<<<(s.cap + 255) / 256, 256, 0, tstream()>>>(rows, s.cap, total);
    k_stamp_fill<<<bgrid(U, n), 64, 0, tstream()>>>(store.rows(), tables.values(), U, h, w,
                                                     ntiles, rows, s.cap);
    so_check(cudaGetLastError(), "stamp rows launch");
    store.end_counts(out.coverage, out.faults, out.refused);
}

void unit_stamp(UnitStoreDevice& store, const UnitTablesDevice& tables, const StampScratch& s,
                const StampPlanes& p, const UnitKernelCounts& rows_out,
                const UnitKernelCounts& out) {
    const int n = store.n_env();
    if (!p.permeability || !p.wave_absorb || !p.heat_atten_q || !p.light_atten_q ||
        !p.obstacles || !p.dyn_permeability || !p.dyn_wave_absorb || !p.dyn_heat_atten_q ||
        !p.dyn_light_atten_q || !p.solid || !p.is_vacuum || !p.atmosphere || !s.winner ||
        !s.freed || p.h < 1 || p.w < 1)
        throw std::invalid_argument("unit_stamp: a null plane or scratch, or an empty grid");
    unit_stamp_rows(store, tables, s, p.h, p.w, rows_out);
    const int total = n * p.h * p.w;
    k_stamp_freed<<<(total + 255) / 256, 256, 0, tstream()>>>(
        reinterpret_cast<const uint8_t*>(p.obstacles), reinterpret_cast<const float*>(p.permeability),
        reinterpret_cast<const uint8_t*>(p.is_vacuum), reinterpret_cast<uint8_t*>(s.freed), total);
    so_check(cudaGetLastError(), "stamp freed launch");
    stamp_launch_resident(n, p.h, p.w, s.cap, reinterpret_cast<const float*>(p.permeability),
                          reinterpret_cast<const float*>(p.wave_absorb),
                          reinterpret_cast<const int32_t*>(p.heat_atten_q),
                          reinterpret_cast<const int32_t*>(p.light_atten_q),
                          reinterpret_cast<bool*>(p.obstacles),
                          reinterpret_cast<float*>(p.dyn_permeability),
                          reinterpret_cast<float*>(p.dyn_wave_absorb),
                          reinterpret_cast<int32_t*>(p.dyn_heat_atten_q),
                          reinterpret_cast<int32_t*>(p.dyn_light_atten_q),
                          reinterpret_cast<const int32_t*>(s.rows),
                          reinterpret_cast<int32_t*>(s.winner));
    store.begin_counts();
    k_stamp_refill<<<dim3(1, (unsigned)n), SORT_BT, 0, tstream()>>>(
        reinterpret_cast<const uint8_t*>(s.freed), reinterpret_cast<const uint8_t*>(p.solid),
        reinterpret_cast<const uint8_t*>(p.is_vacuum), reinterpret_cast<int32_t*>(p.atmosphere),
        p.h, p.w, store.counts_dev());
    so_check(cudaGetLastError(), "stamp refill launch");
    store.end_counts(out.coverage, out.faults, out.refused);
}

}  // namespace breach_cuda
