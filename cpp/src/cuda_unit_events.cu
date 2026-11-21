// ============================================================================
// #29 P6.3 -- THE EVENT RING: its buffers, begin / gather / download. See
// cuda_unit_events.h; residency design §4.5.
// #29 P6.4a -- the buffers and the gather are the RECORD LIST's
// (UnitRecordList), any record width and row cap; the event ring and slot
// 1b's request list are two instances.
//
// The gather is ONE thread per env (blockIdx.y, the bgrid convention): it
// walks the env's rows in order and appends each row's staged records, so the
// list holds the reference's emission order (per slot: id order; within a
// unit, its law's order) whatever order the slot kernel's threads ran in.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_books.h"   // #29 P7e.4: the deferred books
#include "cuda_unit_events.h"
#include "cuda_batch.cuh"          // tstream, benv
#include "unit_events_layout.h"    // GENERATED: the event record layout
#include "unit_requests_layout.h"  // GENERATED: the request record layout

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

inline void ev_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_records/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

__global__ void k_records_gather(int64_t* ring, int64_t* count, unsigned long long* dropped,
                                 int64_t* stage, int64_t* stage_n, int n_units, int cap,
                                 int width, int row_cap) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    int64_t* rg = ring + (size_t)e * cap * width;
    int64_t n = count[e];
    unsigned long long lost = 0;
    for (int r = 0; r < n_units; ++r) {
        const size_t rr = (size_t)e * n_units + r;
        const int64_t c = stage_n[rr];
        const int64_t* src = stage + rr * (size_t)row_cap * width;
        for (int64_t j = 0; j < c; ++j) {
            if (n >= cap) {
                ++lost;
                continue;
            }
            for (int f = 0; f < width; ++f)
                rg[n * width + f] = src[j * width + f];
            ++n;
        }
        stage_n[rr] = 0;
    }
    count[e] = n;
    dropped[e] += lost;
}

}  // namespace

UnitRecordList::UnitRecordList(int n_env, int n_units, int cap, int width, int row_cap)
    : n_env_(n_env), n_units_(n_units), cap_(cap), width_(width), row_cap_(row_cap) {
    if (n_env < 1 || n_units < 1 || cap < 1 || width < 1 || row_cap < 1)
        throw std::invalid_argument(
            "UnitRecordList: n_env, n_units, cap, width and row_cap must be >= 1");
    const size_t ring = (size_t)n_env * cap * width;
    const size_t stage = (size_t)n_env * n_units * row_cap * width;
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_ring_), ring * sizeof(int64_t)), "malloc list");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_count_), n_env * sizeof(int64_t)), "malloc count");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_drop_), n_env * sizeof(unsigned long long)),
             "malloc dropped");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_stage_), stage * sizeof(int64_t)), "malloc stage");
    ev_check(cudaMalloc(reinterpret_cast<void**>(&d_stage_n_),
                        (size_t)n_env * n_units * sizeof(int64_t)), "malloc stage counts");
    ev_check(cudaMemsetAsync(d_ring_, 0, ring * sizeof(int64_t), tstream()), "zero list");
    begin();
}

UnitRecordList::~UnitRecordList() {
    counted_sync(tstream(), "cuda_unit_events.cu:85");
    if (d_ring_) cudaFree(d_ring_);
    if (d_count_) cudaFree(d_count_);
    if (d_drop_) cudaFree(d_drop_);
    if (d_stage_) cudaFree(d_stage_);
    if (d_stage_n_) cudaFree(d_stage_n_);
}

void UnitRecordList::begin() {
    ev_check(cudaMemsetAsync(d_count_, 0, n_env_ * sizeof(int64_t), tstream()), "zero count");
    ev_check(cudaMemsetAsync(d_drop_, 0, n_env_ * sizeof(unsigned long long), tstream()),
             "zero dropped");
    ev_check(cudaMemsetAsync(d_stage_n_, 0, (size_t)n_env_ * n_units_ * sizeof(int64_t),
                             tstream()), "zero stage counts");
}

void UnitRecordList::gather() {
    k_records_gather<<<dim3(1, n_env_), 1, 0, tstream()>>>(
        d_ring_, d_count_, d_drop_, d_stage_, d_stage_n_, n_units_, cap_, width_, row_cap_);
    ev_check(cudaGetLastError(), "gather launch");
}

void UnitRecordList::download(int64_t* recs, int64_t* counts, int64_t* dropped) {
    if (recs == nullptr && books_deferring()) {      // #29 P7e.4: the counts deferred
        if (counts) books_defer("list_counts", d_count_, (size_t)n_env_);
        if (dropped) books_defer("list_dropped", d_drop_, (size_t)n_env_);
        for (int e = 0; e < n_env_; ++e) {
            if (counts) counts[e] = 0;
            if (dropped) dropped[e] = 0;
        }
        return;
    }
    if (recs)
        ev_check(cudaMemcpyAsync(recs, d_ring_, (size_t)n_env_ * cap_ * width_ * sizeof(int64_t),
                                 cudaMemcpyDeviceToHost, tstream()),
                 "D2H list");
    if (counts)
        ev_check(cudaMemcpyAsync(counts, d_count_, n_env_ * sizeof(int64_t),
                                 cudaMemcpyDeviceToHost, tstream()), "D2H count");
    if (dropped)
        ev_check(cudaMemcpyAsync(dropped, d_drop_, n_env_ * sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost, tstream()), "D2H dropped");
    ev_check(counted_sync(tstream(), "cuda_unit_events.cu:118"), "list sync");
}

void UnitRecordList::upload(const int64_t* recs, const int64_t* counts) {
    begin();    // dropped and staging counts to 0
    ev_check(cudaMemcpyAsync(d_ring_, recs, (size_t)n_env_ * cap_ * width_ * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D list");
    ev_check(cudaMemcpyAsync(d_count_, counts, (size_t)n_env_ * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D count");
    ev_check(counted_sync(tstream(), "cuda_unit_events.cu:127"), "upload sync");
}

void UnitRecordList::upload_env(int e, const int64_t* recs, int64_t count) {
    if (e < 0 || e >= n_env_)
        throw std::invalid_argument("UnitRecordList.upload_env: env outside 0 .. n_env");
    if (count < 0 || count > cap_)
        throw std::invalid_argument("UnitRecordList.upload_env: count outside 0 .. cap");
    cudaStream_t st = tstream();
    const size_t row = (size_t)cap_ * width_;
    ev_check(cudaMemcpyAsync(d_ring_ + (size_t)e * row, recs, row * sizeof(int64_t),
                             cudaMemcpyHostToDevice, st), "H2D env list");
    ev_check(cudaMemcpyAsync(d_count_ + e, &count, sizeof(int64_t), cudaMemcpyHostToDevice, st),
             "H2D env count");
    ev_check(cudaMemsetAsync(d_drop_ + e, 0, sizeof(unsigned long long), st), "zero env dropped");
    ev_check(cudaMemsetAsync(d_stage_n_ + (size_t)e * n_units_, 0,
                             (size_t)n_units_ * sizeof(int64_t), st), "zero env staging");
}

UnitEventRing::UnitEventRing(int n_env, int n_units, int e_max)
    : UnitRecordList(n_env, n_units, e_max, unit_events::UNIT_EVENT_F,
                     unit_events::UNIT_EVENT_ROW_CAP) {}

int unit_events_width() { return unit_events::UNIT_EVENT_F; }
int unit_events_row_cap() { return unit_events::UNIT_EVENT_ROW_CAP; }
const char* unit_events_fingerprint() { return unit_events::UNIT_EVENTS_FINGERPRINT; }
int unit_requests_width() { return unit_requests::UNIT_REQ_F; }
int unit_requests_row_cap() { return unit_requests::UNIT_REQ_ROW_CAP; }
const char* unit_requests_fingerprint() { return unit_requests::UNIT_REQUESTS_FINGERPRINT; }

}  // namespace breach_cuda
