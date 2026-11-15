// ============================================================================
// #29 P6.4b -- THE PER-ENV SERIAL APPLIER (slot 9's burn-through) and THE
// TICK-END CLEARS on the device. See cuda_applier.h; residency design §4.3.
//
// k_applier: one BLOCK per env (blockIdx.y), thread 0 walking the env's
// effect list in order -- serial, because the reference is: destroy_wall's
// breach rule reads the tiles an earlier destroy of the same walk changed,
// and its membership diff and seed read the planes the previous effect wrote.
// Each effect is a device function of the topology seam (cuda_topology.cuh)
// writing the env's planes in place and its book channels.
//
// k_tick_end_clears: one thread per cell, (N, h, w) (bgrid), a masked env
// (`active` 0: the reference clears only with a physics runner) untouched.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_batch.cuh"          // tstream, benv, bptr, bgrid, env_off
#include "cuda_topology.cuh"       // the topology seam
#include "unit_tables_layout.h"    // GENERATED: the device tables

#include <cuda_runtime.h>

#include <cstring>
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

// The applier: one block per env, thread 0 the walk (the block's other
// threads are the fan-out's, idle until P7c gives an effect a tile set).
__global__ void k_applier(const int64_t* recs, const int64_t* counts, int cap,
                          topo::PlanePtrs P, int h, int w, int n_gases,
                          const int64_t* tables, int64_t* books) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = benv();
    const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
    int64_t* B = books + (size_t)e * APPLIER_N_BOOKS;
    const topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
    const int64_t requested = counts[e];
    const int64_t k = requested < (int64_t)cap ? requested : (int64_t)cap;
    if (requested > (int64_t)cap) B[APPLIER_BOOK_DROPPED] += requested - (int64_t)cap;
    const int64_t* L = recs + (size_t)e * cap * APPLIER_F;
    for (int64_t j = 0; j < k; ++j) {
        const int64_t* r = L + (size_t)j * APPLIER_F;
        const int64_t kind = r[APPLIER_FX_KIND];
        if (kind == APPLIER_KIND_DESTROY_WALL) {
            const int64_t y = r[APPLIER_FX_Y], x = r[APPLIER_FX_X];
            // out of the grid: destroy_wall's silent return (never an int cast
            // of a coordinate that does not fit)
            if (y < 0 || y >= h || x < 0 || x >= w) continue;
            if (topo::destroy_wall(E, tab, (int)y, (int)x, B))
                B[APPLIER_BOOK_APPLIED] += 1;
        } else {
            B[APPLIER_BOOK_FAULTS] += 1;      // an unknown kind: never packed
        }
    }
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

}  // namespace

FieldApplier::FieldApplier(int n_env, int cap) : n_env_(n_env), cap_(cap) {
    if (n_env < 1 || cap < 1)
        throw std::invalid_argument("FieldApplierBuffer: n_env and cap must be >= 1");
    const size_t nr = (size_t)n_env * cap * APPLIER_F;
    const size_t nb = (size_t)n_env * APPLIER_N_BOOKS;
    ap_check(cudaMalloc(reinterpret_cast<void**>(&d_recs_), nr * sizeof(int64_t)), "malloc recs");
    ap_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_), n_env * sizeof(int64_t)), "malloc counts");
    ap_check(cudaMalloc(reinterpret_cast<void**>(&d_books_), nb * sizeof(int64_t)), "malloc books");
    ap_check(cudaMallocHost(reinterpret_cast<void**>(&h_stage_), (nr + n_env) * sizeof(int64_t)),
             "malloc pinned stage");
    ap_check(cudaMallocHost(reinterpret_cast<void**>(&h_books_), nb * sizeof(int64_t)),
             "malloc pinned books");
    cudaEvent_t ev = nullptr;
    ap_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
    staged_ = ev;
    ap_check(cudaMemsetAsync(d_recs_, 0, nr * sizeof(int64_t), tstream()), "zero recs");
    ap_check(cudaMemsetAsync(d_cnt_, 0, n_env * sizeof(int64_t), tstream()), "zero counts");
    ap_check(cudaMemsetAsync(d_books_, 0, nb * sizeof(int64_t), tstream()), "zero books");
    ap_check(cudaEventRecord(ev, tstream()), "record zero");
}

FieldApplier::~FieldApplier() {
    if (staged_) cudaEventSynchronize(static_cast<cudaEvent_t>(staged_));
    cudaStreamSynchronize(tstream());
    if (staged_) cudaEventDestroy(static_cast<cudaEvent_t>(staged_));
    if (d_recs_) cudaFree(d_recs_);
    if (d_cnt_) cudaFree(d_cnt_);
    if (d_books_) cudaFree(d_books_);
    if (h_stage_) cudaFreeHost(h_stage_);
    if (h_books_) cudaFreeHost(h_books_);
}

void FieldApplier::stage(const int64_t* recs, const int64_t* counts) {
    const size_t nr = (size_t)n_env_ * cap_ * APPLIER_F;
    // the pinned buffer is free once the previous stage copy read it
    ap_check(cudaEventSynchronize(static_cast<cudaEvent_t>(staged_)), "stage wait");
    std::memcpy(h_stage_, recs, nr * sizeof(int64_t));
    std::memcpy(h_stage_ + nr, counts, (size_t)n_env_ * sizeof(int64_t));
    ap_check(cudaMemcpyAsync(d_recs_, h_stage_, nr * sizeof(int64_t), cudaMemcpyHostToDevice,
                             tstream()), "H2D recs");
    ap_check(cudaMemcpyAsync(d_cnt_, h_stage_ + nr, (size_t)n_env_ * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D counts");
    ap_check(cudaEventRecord(static_cast<cudaEvent_t>(staged_), tstream()), "record stage");
}

void FieldApplier::download_books(int64_t* host) {
    const size_t nb = (size_t)n_env_ * APPLIER_N_BOOKS;
    ap_check(cudaMemcpyAsync(h_books_, d_books_, nb * sizeof(int64_t), cudaMemcpyDeviceToHost,
                             tstream()), "D2H books");
    ap_check(cudaStreamSynchronize(tstream()), "books sync");
    std::memcpy(host, h_books_, nb * sizeof(int64_t));
}

void applier_run(FieldApplier& a, const UnitTablesDevice& tables, const ApplierPlanes& planes,
                 int64_t* books_out) {
    if (tables.n_env() != a.n_env())
        throw std::invalid_argument("applier_run: the tables are for another env count");
    if (planes.h < 1 || planes.w < 1 || planes.n_gases < 1)
        throw std::invalid_argument("applier_run: the planes' shape is empty");
    topo::PlanePtrs P;
    for (int k = 0; k < APPLIER_N_PLANES; ++k) {
        if (planes.ptr[k] == 0)
            throw std::invalid_argument("applier_run: a plane pointer is null");
        P.p[k] = (unsigned long long)planes.ptr[k];
    }
    ap_check(cudaMemsetAsync(a.books_dev(), 0,
                             (size_t)a.n_env() * APPLIER_N_BOOKS * sizeof(int64_t), tstream()),
             "zero books");
    k_applier<<<dim3(1, (unsigned)a.n_env()), 32, 0, tstream()>>>(
        a.recs_dev(), a.counts_dev(), a.cap(), P, planes.h, planes.w, planes.n_gases,
        tables.values(), a.books_dev());
    ap_check(cudaGetLastError(), "applier launch");
    a.download_books(books_out);
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
