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

__global__ void k_table_compact(RecordListView v, int live, int64_t* side, int64_t side_stride,
                                int64_t* known, int64_t known_stride) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    int64_t* rows = v.recs + (size_t)e * v.cap * v.width;
    const int64_t n = v.count[e];
    // #29 P8.3: the side array moves with its rows; `known` is recounted
    int64_t* sd = side ? side + (size_t)e * side_stride : nullptr;
    const int64_t kn = known ? known[(size_t)e * known_stride] : 0;
    int64_t kept_known = 0;
    int64_t m = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t* r = rows + i * v.width;
        if (r[live] == 0) continue;
        if (m != i) {
            int64_t* d = rows + m * v.width;
            for (int f = 0; f < v.width; ++f) d[f] = r[f];
            if (sd) sd[m] = sd[i];
        }
        if (i < kn) ++kept_known;
        ++m;
    }
    if (known) known[(size_t)e * known_stride] = kept_known;
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

void ObjectTable::append_host(const int64_t* recs, const int64_t* n, int k) {
    if (k < 1) return;
    if (k > src_k_) {    // the host staging grows, never per call
        counted_sync(tstream(), "cuda_object_table.cu:81");
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
    ot_check(counted_sync(tstream(), "cuda_object_table.cu:100"), "append sync");
}

ObjectTable::~ObjectTable() {
    counted_sync(tstream(), "cuda_object_table.cu:104");
    if (d_src_) cudaFree(d_src_);
    if (d_n_) cudaFree(d_n_);
}

void ObjectTable::compact() {
    k_table_compact<<<dim3(1, n_env()), 1, 0, tstream()>>>(list_view(*this), live_, side_,
                                                            side_stride_, known_, known_stride_);
    ot_check(cudaGetLastError(), "compact launch");
}

void ObjectTable::set_side(int64_t* side, int64_t side_stride, int64_t* known,
                           int64_t known_stride) {
    if ((side == nullptr) != (known == nullptr))
        throw std::invalid_argument("ObjectTable::set_side: a side array needs its known count");
    if (side && (side_stride < cap() || known_stride < 1))
        throw std::invalid_argument("ObjectTable::set_side: a stride shorter than the table");
    side_ = side;
    side_stride_ = side_stride;
    known_ = known;
    known_stride_ = known_stride;
}

const char* object_tables_fingerprint() { return object_tables::OBJECT_TABLES_FINGERPRINT; }

}  // namespace breach_cuda
