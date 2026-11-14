// ============================================================================
// #29 P6.3 -- THE EVENT RING: its buffers, begin / gather / download. See
// cuda_unit_events.h; residency design §4.5.
//
// The gather is ONE thread per env (blockIdx.y, the bgrid convention): it
// walks the env's rows in order and appends each row's staged records, so the
// ring holds the reference's emission order (per slot: id order; within a
// unit, its law's order) whatever order the slot kernel's threads ran in.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_events.h"
#include "cuda_batch.cuh"          // tstream, benv
#include "unit_events_layout.h"    // GENERATED: the record layout

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

using namespace unit_events;

inline void ev_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_events/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

__global__ void k_ring_gather(int64_t* ring, int64_t* count, unsigned long long* dropped,
                              int64_t* stage, int64_t* stage_n, int n_units, int e_max) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    int64_t* rg = ring + (size_t)e * e_max * UNIT_EVENT_F;
    int64_t n = count[e];
    unsigned long long lost = 0;
    for (int r = 0; r < n_units; ++r) {
        const size_t rr = (size_t)e * n_units + r;
        const int64_t c = stage_n[rr];
        const int64_t* src = stage + rr * UNIT_EVENT_ROW_CAP * UNIT_EVENT_F;
        for (int64_t j = 0; j < c; ++j) {
            if (n >= e_max) {
                ++lost;
                continue;
            }
            for (int f = 0; f < UNIT_EVENT_F; ++f)
                rg[n * UNIT_EVENT_F + f] = src[j * UNIT_EVENT_F + f];
            ++n;
        }
        stage_n[rr] = 0;
    }
    count[e] = n;
    dropped[e] += lost;
}

}  // namespace

UnitEventRing::UnitEventRing(int n_env, int n_units, int e_max)
    : n_env_(n_env), n_units_(n_units), e_max_(e_max) {
    if (n_env < 1 || n_units < 1 || e_max < 1)
        throw std::invalid_argument("UnitEventRingBuffer: n_env, n_units and e_max must be >= 1");
    const size_t ring = (size_t)n_env * e_max * UNIT_EVENT_F;
    const size_t stage = (size_t)n_env * n_units * UNIT_EVENT_ROW_CAP * UNIT_EVENT_F;
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_ring_), ring * sizeof(int64_t)), "malloc ring");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_count_), n_env * sizeof(int64_t)), "malloc count");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_drop_), n_env * sizeof(unsigned long long)),
             "malloc dropped");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_stage_), stage * sizeof(int64_t)), "malloc stage");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_stage_n_),
                        (size_t)n_env * n_units * sizeof(int64_t)), "malloc stage counts");
    ev_check(cudaMemsetAsync(d_ring_, 0, ring * sizeof(int64_t), tstream()), "zero ring");
    begin();
}

UnitEventRing::~UnitEventRing() {
    cudaStreamSynchronize(tstream());
    if (d_ring_) cudaFree(d_ring_);
    if (d_count_) cudaFree(d_count_);
    if (d_drop_) cudaFree(d_drop_);
    if (d_stage_) cudaFree(d_stage_);
    if (d_stage_n_) cudaFree(d_stage_n_);
}

void UnitEventRing::begin() {
    ev_check(cudaMemsetAsync(d_count_, 0, n_env_ * sizeof(int64_t), tstream()), "zero count");
    ev_check(cudaMemsetAsync(d_drop_, 0, n_env_ * sizeof(unsigned long long), tstream()),
             "zero dropped");
    ev_check(cudaMemsetAsync(d_stage_n_, 0, (size_t)n_env_ * n_units_ * sizeof(int64_t),
                             tstream()), "zero stage counts");
}

void UnitEventRing::gather() {
    k_ring_gather<<<dim3(1, n_env_), 1, 0, tstream()>>>(d_ring_, d_count_, d_drop_, d_stage_,
                                                        d_stage_n_, n_units_, e_max_);
    ev_check(cudaGetLastError(), "gather launch");
}

void UnitEventRing::download(int64_t* recs, int64_t* counts, int64_t* dropped) {
    if (recs)
        ev_check(cudaMemcpyAsync(recs, d_ring_, (size_t)n_env_ * e_max_ * UNIT_EVENT_F *
                                 sizeof(int64_t), cudaMemcpyDeviceToHost, tstream()),
                 "D2H ring");
    if (counts)
        ev_check(cudaMemcpyAsync(counts, d_count_, n_env_ * sizeof(int64_t),
                                 cudaMemcpyDeviceToHost, tstream()), "D2H count");
    if (dropped)
        ev_check(cudaMemcpyAsync(dropped, d_drop_, n_env_ * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost, tstream()), "D2H dropped");
    ev_check(cudaStreamSynchronize(tstream()), "ring sync");
}

int unit_events_width() { return UNIT_EVENT_F; }
int unit_events_row_cap() { return UNIT_EVENT_ROW_CAP; }
const char* unit_events_fingerprint() { return UNIT_EVENTS_FINGERPRINT; }

}  // namespace breach_cuda
