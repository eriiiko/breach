// ============================================================================
// #29 P7.0b -- THE OBJECT TABLES: upload, the ordered append, the stable
// compaction. See cuda_object_table.h; residency design 4.4c decision 3.
//
// The table is the record list's buffers (UnitRecordList); the append is the
// list's own sink (cuda_unit_lib.cuh's emit_record), ONE thread per env
// (blockIdx.y, the bgrid convention) walking its records in order; the
// compaction is ONE thread per env walking its rows in order. Pure integer.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_object_table.h"
#include "cuda_batch.cuh"             // tstream, benv
#include "cuda_unit_lib.cuh"          // RecordListView, RecordSink, emit_record
#include "object_tables_layout.h"     // GENERATED: every object table's layout

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

inline void ot_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in object_table/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// The env's table as a record sink: its rows, its count, its overflow.
__device__ __forceinline__ unit_lib::RecordSink table_sink(const RecordListView& v, int e) {
    return unit_lib::RecordSink{v.recs + (size_t)e * v.cap * v.width, v.cap, v.count + e,
                                v.dropped + e, v.width};
}

__global__ void k_table_append(RecordListView v, const int64_t* src, const int64_t* src_n,
                               int k) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    unit_lib::RecordSink s = table_sink(v, e);
    const int64_t n = src_n[e];
    for (int64_t j = 0; j < n; ++j)
        unit_lib::emit_record(s, src + ((size_t)e * k + j) * v.width);
}

__global__ void k_table_compact(RecordListView v, int live) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    int64_t* rows = v.recs + (size_t)e * v.cap * v.width;
    const int64_t n = v.count[e];
    int64_t m = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t* r = rows + i * v.width;
        if (r[live] == 0) continue;
        if (m != i) {
            int64_t* d = rows + m * v.width;
            for (int f = 0; f < v.width; ++f) d[f] = r[f];
        }
        ++m;
    }
    for (int64_t i = m; i < n; ++i)
        for (int f = 0; f < v.width; ++f) rows[i * v.width + f] = 0;
    v.count[e] = m;
}

}  // namespace

ObjectTable::ObjectTable(int n_env, int cap, int width, int live, int n_units, int row_cap)
    : UnitRecordList(n_env, n_units, cap, width, row_cap), live_(live) {
    if (live < 0 || live >= width)
        throw std::invalid_argument("ObjectTable: the live column is outside the row");
}

void ObjectTable::upload(const int64_t* rows, const int64_t* counts) {
    begin();    // overflow and staging counts to 0
    ot_check(cudaMemcpyAsync(records_dev(), rows, (size_t)n_env() * cap() * width() * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()),
             "H2D rows");
    ot_check(cudaMemcpyAsync(count_dev(), counts, (size_t)n_env() * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()),
             "H2D counts");
    ot_check(cudaStreamSynchronize(tstream()), "upload sync");
}

void ObjectTable::append_host(const int64_t* recs, const int64_t* n, int k) {
    if (k < 1) return;
    if (k > src_k_) {    // the host staging grows, never per call
        cudaStreamSynchronize(tstream());
        if (d_src_) cudaFree(d_src_);
        d_src_ = nullptr;
        ot_check(cudaMalloc(reinterpret_cast<void**>(&d_src_),
                            (size_t)n_env() * k * width() * sizeof(int64_t)),
                 "malloc append staging");
        src_k_ = k;
    }
    if (!d_n_)
        ot_check(cudaMalloc(reinterpret_cast<void**>(&d_n_), (size_t)n_env() * sizeof(int64_t)),
                 "malloc append counts");
    ot_check(cudaMemcpyAsync(d_src_, recs, (size_t)n_env() * k * width() * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()),
             "H2D append");
    ot_check(cudaMemcpyAsync(d_n_, n, (size_t)n_env() * sizeof(int64_t), cudaMemcpyHostToDevice,
                             tstream()),
             "H2D append counts");
    k_table_append<<<dim3(1, n_env()), 1, 0, tstream()>>>(list_view(*this), d_src_, d_n_, k);
    ot_check(cudaGetLastError(), "append launch");
    ot_check(cudaStreamSynchronize(tstream()), "append sync");
}

ObjectTable::~ObjectTable() {
    cudaStreamSynchronize(tstream());
    if (d_src_) cudaFree(d_src_);
    if (d_n_) cudaFree(d_n_);
}

void ObjectTable::compact() {
    k_table_compact<<<dim3(1, n_env()), 1, 0, tstream()>>>(list_view(*this), live_);
    ot_check(cudaGetLastError(), "compact launch");
}

const char* object_tables_fingerprint() { return object_tables::OBJECT_TABLES_FINGERPRINT; }

}  // namespace breach_cuda
