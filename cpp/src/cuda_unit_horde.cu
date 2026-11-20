// ============================================================================
// #29 P7d -- TEST-ONLY: THE FLOOD of the HORDE stage on given masks and seeds
// (cuda_unit_horde.cuh's flood_block, the same function the stage runs), one
// block per env, so tests/cuda_unit_horde_check.py part FLOOD holds it equal
// to the C++ reference (horde_field.cpp) cell for cell on real and random
// maps. The stage itself is the applier's (cuda_applier.cu).
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_batch.cuh"          // tstream, benv
#include "cuda_unit_horde.cuh"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

inline void hp_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in horde_flood_probe/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

struct ProbeSeeds {
    const int64_t* s;     // (k, 2) of this env
    int n;
    __device__ int count() const { return n; }
    __device__ int64_t y(int i) const { return s[2 * i]; }
    __device__ int64_t x(int i) const { return s[2 * i + 1]; }
};

__global__ void k_flood_probe(int h, int w, const uint8_t* tile_ok, const int64_t* fp,
                              const int64_t* seeds, const int64_t* n_seeds, int k_cap,
                              int32_t* dist, uint8_t* pass, int32_t* queues) {
    __shared__ horde_dev::Shared S;
    const int e = benv();
    const size_t n = (size_t)h * w;
    ProbeSeeds sd{seeds + (size_t)e * k_cap * 2, (int)n_seeds[e]};
    horde_dev::flood_block(tile_ok + e * n, h, w, (int)fp[e], dist + e * n, pass + e * n,
                           queues + 2 * e * n, queues + (2 * e + 1) * n, sd, S);
}

}  // namespace

void horde_flood_probe(int n_env, int h, int w, std::uintptr_t tile_ok, const int64_t* fp,
                       const int64_t* seeds, const int64_t* n_seeds, int k_cap,
                       std::uintptr_t dist) {
    if (n_env < 1 || h < 1 || w < 1 || k_cap < 1 || !tile_ok || !dist)
        throw std::invalid_argument("horde_flood_probe: empty shape or a null plane");
    for (int e = 0; e < n_env; ++e)
        if (n_seeds[e] < 0 || n_seeds[e] > k_cap || fp[e] < 0 ||
            fp[e] > unit_lib::FOOTPRINT_MAX)
            throw std::invalid_argument("horde_flood_probe: a seed count or footprint out of range");
    const size_t n = (size_t)h * w;
    int64_t *d_fp = nullptr, *d_seeds = nullptr, *d_n = nullptr;
    uint8_t* d_pass = nullptr;
    int32_t* d_q = nullptr;
    hp_check(cudaMalloc(reinterpret_cast<void**>(&d_fp), n_env * sizeof(int64_t)), "malloc");
    hp_check(cudaMalloc(reinterpret_cast<void**>(&d_n), n_env * sizeof(int64_t)), "malloc");
    hp_check(cudaMalloc(reinterpret_cast<void**>(&d_seeds),
                        (size_t)n_env * k_cap * 2 * sizeof(int64_t)), "malloc");
    hp_check(cudaMalloc(reinterpret_cast<void**>(&d_pass), n_env * n), "malloc");
    hp_check(cudaMalloc(reinterpret_cast<void**>(&d_q), 2 * n_env * n * sizeof(int32_t)),
             "malloc");
    hp_check(cudaMemcpyAsync(d_fp, fp, n_env * sizeof(int64_t), cudaMemcpyHostToDevice,
                             tstream()), "H2D");
    hp_check(cudaMemcpyAsync(d_n, n_seeds, n_env * sizeof(int64_t), cudaMemcpyHostToDevice,
                             tstream()), "H2D");
    hp_check(cudaMemcpyAsync(d_seeds, seeds, (size_t)n_env * k_cap * 2 * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D");
    k_flood_probe<<<dim3(1, (unsigned)n_env), horde_dev::HORDE_BT, 0, tstream()>>>(
        h, w, reinterpret_cast<const uint8_t*>(tile_ok), d_fp, d_seeds, d_n, k_cap,
        reinterpret_cast<int32_t*>(dist), d_pass, d_q);
    hp_check(cudaGetLastError(), "launch");
    hp_check(counted_sync(tstream(), "cuda_unit_horde.cu:81"), "sync");
    cudaFree(d_fp);
    cudaFree(d_n);
    cudaFree(d_seeds);
    cudaFree(d_pass);
    cudaFree(d_q);
}

}  // namespace breach_cuda
