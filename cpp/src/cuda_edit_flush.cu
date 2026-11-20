// ============================================================================
// #29 P7e.3 -- THE EDIT DOOR, STEP (c) (see cuda_edit_flush.h). Slot 6b's
// FieldEdit flush fed from the DEVICE door's rows: the host's sort, pack and
// buffer (EditQueue.take_ordered, field_edit.pack_for_device,
// physics_runner._edit_buffer) as device kernels, then the device edit
// queue's K1 / K2 (cuda_edit_queue.cu) UNCHANGED, then the door's reset.
// CLASS T (training only): nothing in the played tick calls this TU.
//
// THE ORACLE is the host pack of the same edits: pack_for_device's EDIT and
// TILE rows of take_ordered's list, laid out by _edit_buffer. Three kernels:
//
//   k_flush_edits (one block per env): REFUSE an env whose rows the host
//     pack would refuse, raise on or fall back on (the codes of
//     cuda_edit_flush.h; the lowest code, then the lowest row, wins), then
//     RANK every EDIT row by (field ordinal, source, region) with the row's
//     index as the tie: the door writes the rows at consecutive seqs (checked:
//     EF_KEY), so the index order IS the seq order and the rank is the host's
//     stable sort (field, source_id, region, seq) -- the ordinal is the
//     field name's lexicographic rank (field_edit.DEVICE_FIELD_ORDINALS), so
//     the two orders agree. A rank, not a sort network: exact with ties, no
//     scratch beyond one int64 per row, O(E^2 / 256) compares per env (E the
//     window's rows, at most the door's 8192; tens in a busy tick). Each row
//     lands at e * edit_cap + rank (E_ENV = e: the noise reads it), its tile
//     count beside it; thread 0 then takes the prefix of the counts in the
//     applied order (serial, E steps).
//   k_flush_tiles (one block per env): the env's tile offset (the totals of
//     the envs before it), its scalars from THE TICK ROW (tick word, seed
//     lo, seed hi: draws.MatchDraws' key and tick, as pack_for_device reads
//     them) and the tables (topo_t_amb_q: gas_t_amb_raw()), and every TILE row
//     moved to its edit's applied place (the edit index batch-global). The
//     slots past the batch's total are padding (the dummy edit, cell 0, weight
//     0): K1 takes the capacity, K2 walks only [off[e], off[e+1]).
//
// The range checks are pack_for_device's (field_edit.py): den < 2^62, the
// product |amount x weight x mult_max x scale| < 2^125 (mult_max ONE for a
// noisy row), the contribution (2 bound + den) // (2 den) < 2^55; the row's
// "fits int64" is the device's int64 itself. An env that faults keeps its
// planes untouched (its tile range is empty) and the Python face refuses the
// flush by name at THE ONE READ (books + faults, one copy).
// ============================================================================
#include "cuda_edit_flush.h"

#include <cuda_runtime.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cuda_batch.cuh"          // tstream
#include "cuda_resident.h"         // EDQ_*, edit_queue_launch_resident
#include "cuda_unit_store.h"       // UnitTablesDevice
#include "unit_draws_layout.h"     // TICK_ROW_*
#include "unit_tables_layout.h"    // UNIT_TAB_*

namespace breach_cuda {

namespace {

constexpr int FLUSH_BT = 256;
constexpr int64_t ONE_Q16 = 65536;
constexpr unsigned long long NO_FAULT = ~0ULL;

inline void ef_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in edit_flush/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

struct DoorRows {
    const int64_t* edits;              // (N, ecap, EDIT_DOOR_COLS)
    const int64_t* ecount;             // (N)
    const unsigned long long* edrop;   // (N)
    int ecap;
    const int64_t* tiles;              // (N, tcap, EDQ_TILE_COLS)
    const int64_t* tcount;
    const unsigned long long* tdrop;
    int tcap;
    const int64_t* seq;                // (N)
};

struct FlushBufs {
    int64_t* edits;      // (N * ecap + 1, EDQ_EDIT_COLS)
    int64_t* tiles;      // (N * tcap, EDQ_TILE_COLS)
    int64_t* off;        // (N + 1)
    int64_t* env;        // (N, EDQ_ENV_COLS)
    int64_t* rank;       // (N, ecap)
    int64_t* tsrc;       // (N, ecap)
    int64_t* tcnt;       // (N, ecap)
    int64_t* tpre;       // (N, ecap)
    int64_t* total;      // (N)
    unsigned long long* fault;   // (N)
};

__device__ inline void fault(unsigned long long* f, int code, int64_t row) {
    const unsigned long long r = row < 0 ? 0ULL : (unsigned long long)row;
    atomicMin(f, ((unsigned long long)code << 32) | (r & 0xffffffffULL));
}

__device__ inline unsigned long long uabs(int64_t v) {
    return v < 0 ? 0ULL - (unsigned long long)v : (unsigned long long)v;
}

// x *= f for x < 2^125: false when the product reaches 2^125.
struct U128 { unsigned long long hi, lo; };
__device__ inline bool mul_below_2_125(U128& x, unsigned long long f) {
    const unsigned long long lo = x.lo * f;
    const unsigned long long lo_hi = __umul64hi(x.lo, f);
    if (__umul64hi(x.hi, f) != 0ULL) return false;
    const unsigned long long hl = x.hi * f;
    const unsigned long long hi = hl + lo_hi;
    if (hi < hl) return false;
    x.hi = hi;
    x.lo = lo;
    return (hi >> 61) == 0ULL;
}

// pack_for_device's per-tile bound: EF_PRODUCT / EF_CONTRIB or 0.
__device__ inline int tile_bound(int64_t amount, int64_t weight, bool noisy, int64_t scale,
                                 int64_t den) {
    if (amount == 0 || weight == 0 || scale == 0) return 0;   // bound 0 (den > 0)
    U128 b{0ULL, uabs(amount)};
    if (!mul_below_2_125(b, uabs(weight))) return EF_PRODUCT;
    if (noisy && !mul_below_2_125(b, (unsigned long long)ONE_Q16)) return EF_PRODUCT;
    if (!mul_below_2_125(b, uabs(scale))) return EF_PRODUCT;
    // (2 b + den) // (2 den) >= 2^55  <=>  2 b + den >= den << 56 (den < 2^62)
    U128 l{(b.hi << 1) | (b.lo >> 63), b.lo << 1};
    const unsigned long long s = l.lo + (unsigned long long)den;
    l.hi += s < l.lo ? 1ULL : 0ULL;
    l.lo = s;
    const U128 r{(unsigned long long)den >> 8, (unsigned long long)den << 56};
    if (l.hi > r.hi || (l.hi == r.hi && l.lo >= r.lo)) return EF_CONTRIB;
    return 0;
}

__device__ inline bool key_less(const int64_t* a, int ia, const int64_t* b, int ib) {
    if (a[EDQ_E_FIELD] != b[EDQ_E_FIELD]) return a[EDQ_E_FIELD] < b[EDQ_E_FIELD];
    if (a[EDIT_DOOR_SRC] != b[EDIT_DOOR_SRC]) return a[EDIT_DOOR_SRC] < b[EDIT_DOOR_SRC];
    if (a[EDIT_DOOR_REGION] != b[EDIT_DOOR_REGION])
        return a[EDIT_DOOR_REGION] < b[EDIT_DOOR_REGION];
    return ia < ib;
}

// The first tile of edit k in an env's tiles (their edit indices ascending).
__device__ inline int64_t lower_tile(const int64_t* t, int64_t nt, int64_t k) {
    int64_t lo = 0, hi = nt;
    while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (t[mid * EDQ_TILE_COLS] < k) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__global__ void k_flush_edits(DoorRows D, FlushBufs B, const int64_t* tables, int tab_w,
                              int n_gases, int n_cells) {
    const int e = blockIdx.x;
    unsigned long long* F = B.fault + e;
    __shared__ int64_t n, nt;
    if (threadIdx.x == 0) {
        n = D.ecount[e];
        nt = D.tcount[e];
        if (D.edrop[e] || D.tdrop[e] || n < 0 || n > D.ecap || nt < 0 || nt > D.tcap)
            fault(F, EF_DOOR_OVERFLOW, 0);
        else if (n > 0 && D.seq[e] != D.edits[(size_t)e * D.ecap * EDIT_DOOR_COLS +
                                              EDQ_E_KEY] + n)
            fault(F, EF_KEY, n - 1);
    }
    __syncthreads();
    if (*(volatile unsigned long long*)F != NO_FAULT) {
        if (threadIdx.x == 0) B.total[e] = 0;
        return;
    }
    const int64_t* er0 = D.edits + (size_t)e * D.ecap * EDIT_DOOR_COLS;
    const int64_t* tr0 = D.tiles + (size_t)e * D.tcap * EDQ_TILE_COLS;
    const int64_t* tab = tables + (size_t)e * tab_w;
    // ---- the refusals: the EDIT rows ----
    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
        const int64_t* r = er0 + k * EDIT_DOOR_COLS;
        const int64_t fld = r[EDQ_E_FIELD];
        if (fld < 0 || fld >= EDQ_N_FIELDS) { fault(F, EF_FIELD, k); continue; }
        if (fld == EDQ_F_WATER_DEPTH) fault(F, EF_WATER_DEPTH, k);
        if (fld == EDQ_F_GAS || fld == EDQ_F_SMOKE) {
            const int64_t g = r[EDQ_E_SLICE];
            if (g < 0 || g >= n_gases) fault(F, EF_GAS_SLICE, k);
            else if (tab[unit_tables::UNIT_TAB_GAS_TRACE + g] == 0)
                fault(F, EF_CONSERVATIVE, k);
        }
        if (r[EDQ_E_MODE] < 0 || r[EDQ_E_MODE] > 2) fault(F, EF_MODE, k);
        if (r[EDQ_E_DEN] <= 0 || r[EDQ_E_DEN] >= ((int64_t)1 << 62)) fault(F, EF_DEN, k);
        if (r[EDQ_E_NOISY] && (r[EDQ_E_LOW] < 0 || r[EDQ_E_LOW] > ONE_Q16))
            fault(F, EF_NOISE_LOW, k);
        if (r[EDQ_E_KEY] != er0[EDQ_E_KEY] + k) fault(F, EF_KEY, k);
    }
    __syncthreads();
    // ---- the refusals: the TILE rows (the edit rows' den is checked) ----
    if (*(volatile unsigned long long*)F == NO_FAULT) {
        for (int64_t j = threadIdx.x; j < nt; j += blockDim.x) {
            const int64_t* t = tr0 + j * EDQ_TILE_COLS;
            const int64_t k = t[0];
            if (k < 0 || k >= n || t[1] < 0 || t[1] >= n_cells ||
                (j > 0 && t[-EDQ_TILE_COLS] > k)) {
                fault(F, EF_TILE, j);
                continue;
            }
            const int64_t* r = er0 + k * EDIT_DOOR_COLS;
            const int c = tile_bound(r[EDQ_E_AMOUNT], t[2], r[EDQ_E_NOISY] != 0, t[3],
                                     r[EDQ_E_DEN]);
            if (c) fault(F, c, k);
        }
    }
    __syncthreads();
    if (*(volatile unsigned long long*)F != NO_FAULT) {
        if (threadIdx.x == 0) B.total[e] = 0;
        return;
    }
    // ---- the rank, the rows in applied order, the tile counts ----
    const size_t base = (size_t)e * D.ecap;
    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
        const int64_t* r = er0 + k * EDIT_DOOR_COLS;
        int64_t p = 0;
        for (int64_t j = 0; j < n; ++j)
            p += key_less(er0 + j * EDIT_DOOR_COLS, (int)j, r, (int)k) ? 1 : 0;
        int64_t* dst = B.edits + (base + p) * EDQ_EDIT_COLS;
        for (int c = 0; c < EDQ_EDIT_COLS; ++c) dst[c] = r[c];
        dst[EDQ_E_ENV] = e;
        const int64_t t0 = lower_tile(tr0, nt, k);
        B.rank[base + k] = p;
        B.tsrc[base + k] = t0;
        B.tcnt[base + p] = lower_tile(tr0, nt, k + 1) - t0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int64_t run = 0;
        for (int64_t p = 0; p < n; ++p) {
            B.tpre[base + p] = run;
            run += B.tcnt[base + p];
        }
        B.total[e] = run;
    }
}

__global__ void k_flush_tiles(DoorRows D, FlushBufs B, const int64_t* tables, int tab_w,
                              const int64_t* tick_row, int n_env) {
    const int e = blockIdx.x;
    __shared__ int64_t off_e, grand;
    if (threadIdx.x == 0) {
        int64_t s = 0, g = 0;
        for (int j = 0; j < n_env; ++j) {
            if (j < e) s += B.total[j];
            g += B.total[j];
        }
        off_e = s;
        grand = g;
        B.off[e] = s;
        if (e == n_env - 1) B.off[n_env] = g;
        const int64_t* tr = tick_row + (size_t)e * unit_draws::TICK_ROW_W;
        int64_t* es = B.env + (size_t)e * EDQ_ENV_COLS;
        // pack_for_device: (rng.tick & M32, rng.key lo, rng.key hi, t_amb_raw)
        es[0] = tr[unit_draws::TICK_ROW_DRAW_TICK] & 0xffffffffLL;
        es[1] = tr[unit_draws::TICK_ROW_SEED_LO];
        es[2] = tr[unit_draws::TICK_ROW_SEED_HI];
        es[3] = tables[(size_t)e * tab_w + unit_tables::UNIT_TAB_TOPO_T_AMB_Q];
        if (e == 0) {   // the dummy edit the padding tiles name: amount 0, den 1
            int64_t* d = B.edits + (size_t)n_env * D.ecap * EDQ_EDIT_COLS;
            for (int c = 0; c < EDQ_EDIT_COLS; ++c) d[c] = 0;
            d[EDQ_E_FIELD] = EDQ_F_HEAT;
            d[EDQ_E_DEN] = 1;
        }
    }
    __syncthreads();
    const size_t base = (size_t)e * D.ecap;
    if (B.fault[e] == NO_FAULT) {
        const int64_t nt = D.tcount[e];
        const int64_t* tr0 = D.tiles + (size_t)e * D.tcap * EDQ_TILE_COLS;
        for (int64_t j = threadIdx.x; j < nt; j += blockDim.x) {
            const int64_t* t = tr0 + j * EDQ_TILE_COLS;
            const int64_t k = t[0];
            const int64_t p = B.rank[base + k];
            int64_t* dst = B.tiles +
                (size_t)(off_e + B.tpre[base + p] + (j - B.tsrc[base + k])) * EDQ_TILE_COLS;
            dst[0] = (int64_t)base + p;
            dst[1] = t[1];
            dst[2] = t[2];
            dst[3] = t[3];
        }
    }
    const int64_t cap = (int64_t)n_env * D.tcap;
    const int64_t dummy = (int64_t)n_env * D.ecap;
    for (int64_t i = grand + (int64_t)e * blockDim.x + threadIdx.x; i < cap;
         i += (int64_t)n_env * blockDim.x) {
        int64_t* dst = B.tiles + (size_t)i * EDQ_TILE_COLS;
        dst[0] = dummy;
        dst[1] = 0;
        dst[2] = 0;
        dst[3] = 1;
    }
}

template <class T>
T* dev_alloc(size_t n, const char* what) {
    T* p = nullptr;
    ef_check(cudaMalloc(reinterpret_cast<void**>(&p), (n ? n : 1) * sizeof(T)), what);
    return p;
}

}  // namespace

EditFlush::EditFlush(int n_env, int edit_cap, int tile_cap)
    : n_env_(n_env), edit_cap_(edit_cap), tile_cap_(tile_cap) {
    if (n_env < 1 || edit_cap < 1 || tile_cap < 1)
        throw std::invalid_argument("EditFlush: n_env, edit_cap and tile_cap must be >= 1");
    const size_t N = (size_t)n_env, E = N * edit_cap, T = N * tile_cap;
    d_edits_ = dev_alloc<int64_t>((E + 1) * EDQ_EDIT_COLS, "malloc edits");
    d_tiles_ = dev_alloc<int64_t>(T * EDQ_TILE_COLS, "malloc tiles");
    d_off_ = dev_alloc<int64_t>(N + 1, "malloc off");
    d_env_ = dev_alloc<int64_t>(N * EDQ_ENV_COLS, "malloc env");
    d_rank_ = dev_alloc<int64_t>(E, "malloc rank");
    d_tsrc_ = dev_alloc<int64_t>(E, "malloc tsrc");
    d_tcnt_ = dev_alloc<int64_t>(E, "malloc tcnt");
    d_tpre_ = dev_alloc<int64_t>(E, "malloc tpre");
    d_total_ = dev_alloc<int64_t>(N, "malloc total");
    d_contrib_ = dev_alloc<int64_t>(T, "malloc contrib");
    d_books_ = dev_alloc<unsigned long long>(N * EDQ_SLOTS + N, "malloc books");
    ef_check(cudaMemset(d_books_, 0, (N * EDQ_SLOTS) * sizeof(unsigned long long)),
             "zero books");
    ef_check(cudaMemset(d_books_ + N * EDQ_SLOTS, 0xff, N * sizeof(unsigned long long)),
             "clear faults");
}

EditFlush::~EditFlush() {
    counted_sync(tstream(), "cuda_edit_flush.cu:333");
    for (int64_t* p : {d_edits_, d_tiles_, d_off_, d_env_, d_rank_, d_tsrc_, d_tcnt_, d_tpre_,
                       d_total_, d_contrib_, d_host_})
        if (p) cudaFree(p);
    if (d_books_) cudaFree(d_books_);
}

namespace {
void check_planes(const std::uintptr_t* planes, const char* who) {
    // gas, temperature, gas_energy, fire, heat, solid, is_vacuum, [is_ambient],
    // thermal_solid, flammable
    for (int k = 0; k < 10; ++k)
        if (k != 7 && planes[k] == 0)
            throw std::invalid_argument(std::string(who) + ": a plane the queue writes or "
                                        "reads is missing");
}

void launch_queue(int N, int h, int w, int n_gases, int o2, int n2, int n_tiles, uint32_t salt,
                  const int64_t* edits, const int64_t* tiles, const int64_t* off,
                  const int64_t* env, const std::uintptr_t* p, int64_t* contrib,
                  unsigned long long* books) {
    edit_queue_launch_resident(
        N, h, w, n_gases, o2, n2, n_tiles, salt, edits, tiles, off, env,
        reinterpret_cast<int32_t*>(p[0]), reinterpret_cast<int32_t*>(p[1]),
        reinterpret_cast<int64_t*>(p[2]), reinterpret_cast<int32_t*>(p[3]),
        reinterpret_cast<int32_t*>(p[4]), reinterpret_cast<const bool*>(p[5]),
        reinterpret_cast<const bool*>(p[6]),
        p[7] ? reinterpret_cast<const bool*>(p[7]) : nullptr,
        reinterpret_cast<const bool*>(p[8]), reinterpret_cast<const bool*>(p[9]), contrib,
        books);
}
}  // namespace

void EditFlush::run(EditDoor& door, const UnitTablesDevice& tables, std::uintptr_t tick_row,
                    const std::uintptr_t* planes, int h, int w, int n_gases, int o2_idx,
                    int n2_idx, uint32_t salt) {
    if (door.n_env() != n_env_ || tables.n_env() != n_env_)
        throw std::invalid_argument("edit flush: the door, the tables and the flush must "
                                    "hold one batch");
    if (door.edits().cap() != edit_cap_ || door.tiles().cap() != tile_cap_)
        throw std::invalid_argument("edit flush: the flush's capacities are not the door's");
    if (h <= 0 || w <= 0 || tick_row == 0)
        throw std::invalid_argument("edit flush: a grid and the tick row are required");
    if (n_gases <= 0 || n_gases > EDQ_GAS_MAX || n_gases > unit_tables::UNIT_TAB_GAS_TRACE_W)
        throw std::invalid_argument("edit flush: n_gases outside the tables' gas rows");
    if (o2_idx < 0 || o2_idx >= n_gases || n2_idx < 0 || n2_idx >= n_gases)
        throw std::invalid_argument("edit flush: a bulk gas index outside the block");
    check_planes(planes, "edit flush");
    cudaStream_t st = tstream();
    const size_t N = (size_t)n_env_;
    ef_check(cudaMemsetAsync(d_books_, 0, N * EDQ_SLOTS * sizeof(unsigned long long), st),
             "zero books");
    ef_check(cudaMemsetAsync(d_books_ + N * EDQ_SLOTS, 0xff, N * sizeof(unsigned long long),
                             st), "clear faults");
    DoorRows D;
    D.edits = door.edits().records_dev();
    D.ecount = door.edits().count_dev();
    D.edrop = door.edits().dropped_dev();
    D.ecap = door.edits().cap();
    D.tiles = door.tiles().records_dev();
    D.tcount = door.tiles().count_dev();
    D.tdrop = door.tiles().dropped_dev();
    D.tcap = door.tiles().cap();
    D.seq = door.seq_dev();
    FlushBufs B{d_edits_, d_tiles_, d_off_, d_env_, d_rank_, d_tsrc_, d_tcnt_, d_tpre_,
                d_total_, d_books_ + N * EDQ_SLOTS};
    k_flush_edits<<<n_env_, FLUSH_BT, 0, st>>>(D, B, tables.values(), tables.width(), n_gases,
                                               h * w);
    ef_check(cudaGetLastError(), "build edits");
    k_flush_tiles<<<n_env_, FLUSH_BT, 0, st>>>(D, B, tables.values(), tables.width(),
                                               reinterpret_cast<const int64_t*>(tick_row),
                                               n_env_);
    ef_check(cudaGetLastError(), "build tiles");
    launch_queue(n_env_, h, w, n_gases, o2_idx, n2_idx, n_env_ * tile_cap_, salt, d_edits_,
                 d_tiles_, d_off_, d_env_, planes, d_contrib_, d_books_);
    door.begin();   // THE 6b RESET: the window's rows empty, every seq 0
}

void EditFlush::run_host_pack(const int64_t* buf, int n_edits, int n_tiles,
                              const std::uintptr_t* planes, int h, int w, int n_gases,
                              int o2_idx, int n2_idx, uint32_t salt) {
    if (buf == nullptr || n_edits < 0 || n_tiles < 0 || h <= 0 || w <= 0)
        throw std::invalid_argument("edit flush (host pack): bad records");
    if (n_gases <= 0 || n_gases > EDQ_GAS_MAX || o2_idx < 0 || o2_idx >= n_gases ||
        n2_idx < 0 || n2_idx >= n_gases)
        throw std::invalid_argument("edit flush (host pack): bad gas block");
    check_planes(planes, "edit flush (host pack)");
    cudaStream_t st = tstream();
    const size_t N = (size_t)n_env_;
    const size_t count = (N + 1) + N * EDQ_ENV_COLS + (size_t)n_edits * EDQ_EDIT_COLS +
                         (size_t)n_tiles * EDQ_TILE_COLS;
    if (count > host_cap_) {
        if (d_host_) {
            counted_sync(st, "cuda_edit_flush.cu:426");
            cudaFree(d_host_);
        }
        d_host_ = dev_alloc<int64_t>(count, "malloc host pack");
        host_cap_ = count;
    }
    ef_check(cudaMemcpyAsync(d_host_, buf, count * sizeof(int64_t), cudaMemcpyHostToDevice, st),
             "H2D host pack");
    ef_check(counted_sync(st, "cuda_edit_flush.cu:434"), "host pack sync");   // buf is pageable host memory
    ef_check(cudaMemsetAsync(d_books_, 0, N * EDQ_SLOTS * sizeof(unsigned long long), st),
             "zero books");
    ef_check(cudaMemsetAsync(d_books_ + N * EDQ_SLOTS, 0xff, N * sizeof(unsigned long long),
                             st), "clear faults");
    const int64_t* off = d_host_;
    const int64_t* env = off + (N + 1);
    const int64_t* edits = env + N * EDQ_ENV_COLS;
    const int64_t* tiles = edits + (size_t)n_edits * EDQ_EDIT_COLS;
    if ((size_t)n_tiles > N * tile_cap_)
        throw std::invalid_argument("edit flush (host pack): more tiles than the scratch");
    launch_queue(n_env_, h, w, n_gases, o2_idx, n2_idx, n_tiles, salt, edits, tiles, off, env,
                 planes, d_contrib_, d_books_);
}

void EditFlush::download_books(int64_t* books, uint64_t* faults) {
    const size_t N = (size_t)n_env_;
    std::vector<unsigned long long> h(N * EDQ_SLOTS + N);
    ef_check(cudaMemcpyAsync(h.data(), d_books_, h.size() * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, tstream()), "D2H books");
    ef_check(counted_sync(tstream(), "cuda_edit_flush.cu:454"), "books sync");
    std::memcpy(books, h.data(), N * EDQ_SLOTS * sizeof(int64_t));
    std::memcpy(faults, h.data() + N * EDQ_SLOTS, N * sizeof(uint64_t));
}

void EditFlush::download_rows(int64_t* edits, int64_t* tiles, int64_t* off, int64_t* env) {
    const size_t N = (size_t)n_env_;
    cudaStream_t st = tstream();
    ef_check(cudaMemcpyAsync(edits, d_edits_, (N * edit_cap_ + 1) * EDQ_EDIT_COLS * 8,
                             cudaMemcpyDeviceToHost, st), "D2H edits");
    ef_check(cudaMemcpyAsync(tiles, d_tiles_, N * tile_cap_ * EDQ_TILE_COLS * 8,
                             cudaMemcpyDeviceToHost, st), "D2H tiles");
    ef_check(cudaMemcpyAsync(off, d_off_, (N + 1) * 8, cudaMemcpyDeviceToHost, st), "D2H off");
    ef_check(cudaMemcpyAsync(env, d_env_, N * EDQ_ENV_COLS * 8, cudaMemcpyDeviceToHost, st),
             "D2H env");
    ef_check(counted_sync(st, "cuda_edit_flush.cu:469"), "rows sync");
}

bool edit_door_append_probe(EditDoor& door, int env, const int64_t* edits, int n_edits,
                            const int64_t* tiles, int n_tiles) {
    const int N = door.n_env();
    if (env < 0 || env >= N || n_edits < 0 || n_tiles < 0)
        throw std::invalid_argument("edit_door_append_probe: bad env or counts");
    UnitRecordList& EL = door.edits();
    UnitRecordList& TL = door.tiles();
    cudaStream_t st = tstream();
    int64_t ec = 0, tc = 0, seq = 0;
    ef_check(cudaMemcpyAsync(&ec, EL.count_dev() + env, 8, cudaMemcpyDeviceToHost, st), "D2H");
    ef_check(cudaMemcpyAsync(&tc, TL.count_dev() + env, 8, cudaMemcpyDeviceToHost, st), "D2H");
    ef_check(cudaMemcpyAsync(&seq, door.seq_dev() + env, 8, cudaMemcpyDeviceToHost, st), "D2H");
    ef_check(counted_sync(st, "cuda_edit_flush.cu:484"), "probe sync");
    if (ec + n_edits > EL.cap() || tc + n_tiles > TL.cap()) return false;
    for (int k = 0; k < n_edits; ++k)
        if (edits[(size_t)k * EDIT_DOOR_COLS + EDQ_E_KEY] != seq + k) return false;
    for (int j = 0; j < n_tiles; ++j)
        if (tiles[(size_t)j * EDQ_TILE_COLS] < 0 || tiles[(size_t)j * EDQ_TILE_COLS] >= n_edits)
            return false;
    std::vector<int64_t> t(tiles, tiles + (size_t)n_tiles * EDQ_TILE_COLS);
    for (int j = 0; j < n_tiles; ++j) t[(size_t)j * EDQ_TILE_COLS] += ec;   // the env's index
    if (n_edits)
        ef_check(cudaMemcpyAsync(EL.records_dev() + ((size_t)env * EL.cap() + ec) * EL.width(),
                                 edits, (size_t)n_edits * EL.width() * 8,
                                 cudaMemcpyHostToDevice, st), "H2D edits");
    if (n_tiles)
        ef_check(cudaMemcpyAsync(TL.records_dev() + ((size_t)env * TL.cap() + tc) * TL.width(),
                                 t.data(), t.size() * 8, cudaMemcpyHostToDevice, st),
                 "H2D tiles");
    const int64_t ec2 = ec + n_edits, tc2 = tc + n_tiles, seq2 = seq + n_edits;
    ef_check(cudaMemcpyAsync(EL.count_dev() + env, &ec2, 8, cudaMemcpyHostToDevice, st), "H2D");
    ef_check(cudaMemcpyAsync(TL.count_dev() + env, &tc2, 8, cudaMemcpyHostToDevice, st), "H2D");
    ef_check(cudaMemcpyAsync(door.seq_dev() + env, &seq2, 8, cudaMemcpyHostToDevice, st),
             "H2D");
    ef_check(counted_sync(st, "cuda_edit_flush.cu:506"), "probe sync");
    return true;
}

}  // namespace breach_cuda
