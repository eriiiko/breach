// ============================================================================
// #29 P6.1 -- THE DEVICE UNIT STORE and its pilot kernel (slot 1c, exposure
// decay). See cuda_unit_store.h; residency design §4.4b.
//
// The store is one (N, U, W) int64 buffer in the layout of the GENERATED
// unit_store_layout.h. Every copy runs on THE tick stream through pinned
// staging (the stamp rows' and the edit pack's idiom); an event guards the
// staging buffer, so a second upload never overwrites bytes the first copy
// has not read yet. Every unit kernel is born (N, U): bgrid(U, N), one thread
// per unit row, blockIdx.y the env, the row pointer rebased by the env's
// stride (bptr), a padding row (valid 0) skipped.
//
// The exposure decay is src/simulation/exposure.py's law integer for integer
// (decay_units / decay_step): an arithmetic change lands there first, then
// here; tests/cuda_unit_store_check.py holds the two equal at tol 0 on live
// matches (the shadow lockstep). Like decay_units it reads no life state: a
// dead unit's block is zeroed at its death, so it has nothing to decay.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_store.h"
#include "cuda_batch.cuh"          // tstream, bgrid, bptr, benv
#include "fixed_point.h"           // mul128_shr (FP_HD)
#include "unit_store_layout.h"     // GENERATED: the row layout

#include <cuda_runtime.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace breach_cuda {

namespace {

inline void us_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_store/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// exposure.decay_step: max(0, x - max(1, (x * d) >> 16)), 0 for x <= 0. The
// product is formed in 128 bits (the kit's mul128_shr), so for every int64
// x > 0 and d in [0, 2^16] the shifted value is exact and <= x: no wrap, the
// result Python's unbounded ints give.
__device__ __forceinline__ int64_t exposure_decay_step(int64_t x, int64_t d) {
    if (x <= 0) return 0;
    int64_t p = fixedpoint::mul128_shr(x, d, fixedpoint::FP_SHIFT);
    if (p < 1) p = 1;
    const int64_t r = x - p;
    return r > 0 ? r : 0;
}

// Slot 1c: one thread per unit row of one env.
__global__ void k_exposure_decay(int64_t* store, int n_units, int64_t decay_q,
                                 unsigned long long* coverage) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* env = bptr(store, (size_t)n_units * unit_store::UNIT_STORE_W);
    int64_t* row = env + (size_t)r * unit_store::UNIT_STORE_W;
    if (row[unit_store::UNIT_COL_VALID] == 0) return;
    int64_t* x = row + unit_store::UNIT_COL_EXPOSURE;
    bool changed = false;
    for (int t = 0; t < unit_store::UNIT_COL_EXPOSURE_W; ++t) {
        const int64_t v = x[t];
        if (v != 0) {
            const int64_t nv = exposure_decay_step(v, decay_q);
            if (nv != v) {
                x[t] = nv;
                changed = true;
            }
        }
    }
    if (changed) atomicAdd(&coverage[benv()], 1ULL);
}

}  // namespace

UnitStoreDevice::UnitStoreDevice(int n_env, int n_units, int width)
    : n_env_(n_env), n_units_(n_units), width_(width) {
    if (n_env < 1 || n_units < 1)
        throw std::invalid_argument("UnitStoreBuffer: n_env and n_units must be >= 1");
    if (width != unit_store::UNIT_STORE_W) {
        std::ostringstream os;
        os << "UnitStoreBuffer: width " << width << " is not the compiled layout's "
           << unit_store::UNIT_STORE_W
           << " (run tools/gen_unit_layout.py --write and rebuild cpp/build_cuda)";
        throw std::invalid_argument(os.str());
    }
    const size_t bytes = count() * sizeof(int64_t);
    us_check(cudaMalloc(reinterpret_cast<void**>(&d_), bytes), "malloc store");
    us_check(cudaMallocHost(reinterpret_cast<void**>(&h_), bytes), "malloc pinned store");
    us_check(cudaMalloc(reinterpret_cast<void**>(&d_cov_),
                        (size_t)n_env_ * sizeof(unsigned long long)), "malloc coverage");
    us_check(cudaMallocHost(reinterpret_cast<void**>(&h_cov_),
                            (size_t)n_env_ * sizeof(unsigned long long)), "malloc pinned coverage");
    cudaEvent_t ev = nullptr;
    us_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
    staged_ = ev;
    // A fresh store is all zero: every row padding until the first upload.
    us_check(cudaMemsetAsync(d_, 0, bytes, tstream()), "zero store");
    us_check(cudaEventRecord(ev, tstream()), "record zero");
}

UnitStoreDevice::~UnitStoreDevice() {
    // Never throw from a destructor: finish the stream's work on the buffers,
    // then free them (errors ignored).
    if (staged_) cudaEventSynchronize(static_cast<cudaEvent_t>(staged_));
    cudaStreamSynchronize(tstream());
    if (staged_) cudaEventDestroy(static_cast<cudaEvent_t>(staged_));
    if (d_) cudaFree(d_);
    if (h_) cudaFreeHost(h_);
    if (d_cov_) cudaFree(d_cov_);
    if (h_cov_) cudaFreeHost(h_cov_);
}

void UnitStoreDevice::stage_wait() {
    us_check(cudaEventSynchronize(static_cast<cudaEvent_t>(staged_)), "staging wait");
}

void UnitStoreDevice::upload(const int64_t* host) {
    if (host == nullptr) throw std::invalid_argument("UnitStoreBuffer.upload: null rows");
    stage_wait();
    const size_t bytes = count() * sizeof(int64_t);
    std::memcpy(h_, host, bytes);
    us_check(cudaMemcpyAsync(d_, h_, bytes, cudaMemcpyHostToDevice, tstream()), "H2D store");
    us_check(cudaEventRecord(static_cast<cudaEvent_t>(staged_), tstream()), "record H2D");
}

void UnitStoreDevice::download(int64_t* host) {
    if (host == nullptr) throw std::invalid_argument("UnitStoreBuffer.download: null out");
    stage_wait();
    const size_t bytes = count() * sizeof(int64_t);
    us_check(cudaMemcpyAsync(h_, d_, bytes, cudaMemcpyDeviceToHost, tstream()), "D2H store");
    us_check(cudaEventRecord(static_cast<cudaEvent_t>(staged_), tstream()), "record D2H");
    stage_wait();
    std::memcpy(host, h_, bytes);
}

void UnitStoreDevice::exposure_decay(int64_t decay_q, int64_t* coverage_out) {
    if (decay_q < 0 || decay_q > (int64_t)fixedpoint::FP_ONE) {
        std::ostringstream os;
        os << "UnitStoreBuffer.exposure_decay: decay_q " << decay_q
           << " outside [0, " << fixedpoint::FP_ONE << "]";
        throw std::invalid_argument(os.str());
    }
    if (coverage_out == nullptr)
        throw std::invalid_argument("UnitStoreBuffer.exposure_decay: null coverage");
    const size_t cov_bytes = (size_t)n_env_ * sizeof(unsigned long long);
    us_check(cudaMemsetAsync(d_cov_, 0, cov_bytes, tstream()), "zero coverage");
    k_exposure_decay<<<bgrid(n_units_, n_env_), 256, 0, tstream()>>>(
        d_, n_units_, decay_q, d_cov_);
    us_check(cudaGetLastError(), "exposure decay launch");
    us_check(cudaMemcpyAsync(h_cov_, d_cov_, cov_bytes, cudaMemcpyDeviceToHost, tstream()),
             "D2H coverage");
    us_check(cudaStreamSynchronize(tstream()), "exposure decay sync");
    for (int e = 0; e < n_env_; ++e) coverage_out[e] = (int64_t)h_cov_[e];
}

int unit_store_width() { return unit_store::UNIT_STORE_W; }
const char* unit_store_fingerprint() { return unit_store::UNIT_STORE_FINGERPRINT; }
int unit_store_valid_col() { return unit_store::UNIT_COL_VALID; }
int unit_store_exposure_col() { return unit_store::UNIT_COL_EXPOSURE; }
int unit_store_exposure_width() { return unit_store::UNIT_COL_EXPOSURE_W; }

}  // namespace breach_cuda
