// ============================================================================
// #29 P6.1 / P6.2 -- THE DEVICE UNIT STORE and THE DEVICE TABLES: the buffers
// and their copies. See cuda_unit_store.h; residency design §4.4b, §4.11.
//
// The store is one (N, U, W) int64 buffer in the layout of the GENERATED
// unit_store_layout.h; the tables one (N, T) int64 buffer in the layout of the
// GENERATED unit_tables_layout.h. Every copy runs on THE tick stream through
// pinned staging (the stamp rows' and the edit pack's idiom); an event guards
// the store's staging buffer, so a second upload never overwrites bytes the
// first copy has not read yet.
//
// No slot kernel lives here (P6.2: a slot's kernel is a free function in its
// slot family's TU -- cuda_unit_exposure.cu, cuda_unit_impulse.cu,
// cuda_unit_orders.cu, cuda_unit_conversion.cu -- over the device unit
// library cuda_unit_lib.cuh); this TU owns the buffers, the counts and the
// input staging they share.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_store.h"
#include "cuda_batch.cuh"          // tstream
#include "unit_store_layout.h"     // GENERATED: the row layout
#include "unit_tables_layout.h"    // GENERATED: the table layout

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
    const size_t cnt_bytes = (size_t)3 * n_env_ * sizeof(unsigned long long);
    us_check(cudaMalloc(reinterpret_cast<void**>(&d_), bytes), "malloc store");
    us_check(cudaMallocHost(reinterpret_cast<void**>(&h_), bytes), "malloc pinned store");
    us_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_), cnt_bytes), "malloc counts");
    us_check(cudaMallocHost(reinterpret_cast<void**>(&h_cnt_), cnt_bytes), "malloc pinned counts");
    cudaEvent_t ev = nullptr;
    us_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
    staged_ = ev;
    // A fresh store is all zero: every row padding until the first upload.
    us_check(cudaMemsetAsync(d_, 0, bytes, tstream()), "zero store");
    us_check(cudaMemsetAsync(d_cnt_, 0, cnt_bytes, tstream()), "zero counts");
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
    if (d_cnt_) cudaFree(d_cnt_);
    if (h_cnt_) cudaFreeHost(h_cnt_);
    for (int s = 0; s < INPUT_SLOTS; ++s) {
        if (d_in_[s]) cudaFree(d_in_[s]);
        if (h_in_[s]) cudaFreeHost(h_in_[s]);
    }
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

void UnitStoreDevice::begin_counts() {
    us_check(cudaMemsetAsync(d_cnt_, 0, (size_t)3 * n_env_ * sizeof(unsigned long long),
                             tstream()), "zero counts");
}

void UnitStoreDevice::end_counts(int64_t* coverage, int64_t* faults, int64_t* refused) {
    const size_t cnt_bytes = (size_t)3 * n_env_ * sizeof(unsigned long long);
    us_check(cudaMemcpyAsync(h_cnt_, d_cnt_, cnt_bytes, cudaMemcpyDeviceToHost, tstream()),
             "D2H counts");
    us_check(cudaStreamSynchronize(tstream()), "counts sync");
    for (int e = 0; e < n_env_; ++e) {
        if (coverage) coverage[e] = (int64_t)h_cnt_[e];
        if (faults) faults[e] = (int64_t)h_cnt_[n_env_ + e];
        if (refused) refused[e] = (int64_t)h_cnt_[2 * n_env_ + e];
    }
}

const int64_t* UnitStoreDevice::stage_input(int slot, const int64_t* host, size_t n) {
    if (slot < 0 || slot >= INPUT_SLOTS)
        throw std::invalid_argument("UnitStoreBuffer.stage_input: slot out of range");
    if (n == 0) return nullptr;
    if (host == nullptr) throw std::invalid_argument("UnitStoreBuffer.stage_input: null input");
    // The previous use of this slot's buffers must be done (a kernel driver
    // ends with end_counts' stream sync; this keeps the rule local).
    us_check(cudaStreamSynchronize(tstream()), "input staging wait");
    if (n > in_cap_[slot]) {
        if (d_in_[slot]) cudaFree(d_in_[slot]);
        if (h_in_[slot]) cudaFreeHost(h_in_[slot]);
        d_in_[slot] = nullptr;
        h_in_[slot] = nullptr;
        in_cap_[slot] = 0;
        us_check(cudaMalloc(reinterpret_cast<void**>(&d_in_[slot]), n * sizeof(int64_t)),
                 "malloc input");
        us_check(cudaMallocHost(reinterpret_cast<void**>(&h_in_[slot]), n * sizeof(int64_t)),
                 "malloc pinned input");
        in_cap_[slot] = n;
    }
    std::memcpy(h_in_[slot], host, n * sizeof(int64_t));
    us_check(cudaMemcpyAsync(d_in_[slot], h_in_[slot], n * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D input");
    return d_in_[slot];
}

UnitTablesDevice::UnitTablesDevice(int n_env, int width) : n_env_(n_env), width_(width) {
    if (n_env < 1) throw std::invalid_argument("UnitTablesBuffer: n_env must be >= 1");
    if (width != unit_tables::UNIT_TABLES_W) {
        std::ostringstream os;
        os << "UnitTablesBuffer: width " << width << " is not the compiled tables' "
           << unit_tables::UNIT_TABLES_W
           << " (run tools/gen_unit_layout.py --write and rebuild cpp/build_cuda)";
        throw std::invalid_argument(os.str());
    }
    us_check(cudaMalloc(reinterpret_cast<void**>(&d_),
                        (size_t)n_env_ * width_ * sizeof(int64_t)), "malloc tables");
    us_check(cudaMemsetAsync(d_, 0, (size_t)n_env_ * width_ * sizeof(int64_t), tstream()),
             "zero tables");
}

UnitTablesDevice::~UnitTablesDevice() {
    cudaStreamSynchronize(tstream());
    if (d_) cudaFree(d_);
}

void UnitTablesDevice::upload(const int64_t* host) {
    if (host == nullptr) throw std::invalid_argument("UnitTablesBuffer.upload: null rows");
    us_check(cudaMemcpyAsync(d_, host, (size_t)n_env_ * width_ * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D tables");
    us_check(cudaStreamSynchronize(tstream()), "tables sync");
}

void UnitTablesDevice::download(int64_t* host) {
    if (host == nullptr) throw std::invalid_argument("UnitTablesBuffer.download: null out");
    us_check(cudaMemcpyAsync(host, d_, (size_t)n_env_ * width_ * sizeof(int64_t),
                             cudaMemcpyDeviceToHost, tstream()), "D2H tables");
    us_check(cudaStreamSynchronize(tstream()), "tables sync");
}

int unit_store_width() { return unit_store::UNIT_STORE_W; }
const char* unit_store_fingerprint() { return unit_store::UNIT_STORE_FINGERPRINT; }
int unit_store_valid_col() { return unit_store::UNIT_COL_VALID; }
int unit_store_exposure_col() { return unit_store::UNIT_COL_EXPOSURE; }
int unit_store_exposure_width() { return unit_store::UNIT_COL_EXPOSURE_W; }
int unit_tables_width() { return unit_tables::UNIT_TABLES_W; }
const char* unit_tables_fingerprint() { return unit_tables::UNIT_TABLES_FINGERPRINT; }

}  // namespace breach_cuda
