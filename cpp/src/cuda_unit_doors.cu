// ============================================================================
// #29 P6b.1 -- THE USE REQUEST BECOMES A TABLE WRITE (residency design 4.4c,
// the cut table's P6b.1 row; the Request list's rule: "a door's want_open"
// waited for its table). See cuda_unit_kernels.h.
//
//   unit_door_uses  slot 1b's USE records (gathered in id order by the verbs
//       kernel, cuda_unit_verbs.cu) flip the door table's `want_open`: ONE
//       thread per env walks the request list in LIST order and, for each
//       USE, sets want_open = !want_open on the door table's row `door` (the
//       door's index in Simulation._doors, the order the table holds) --
//       Simulation._direct_use / unit_requests.apply_to_host. Two uses of one
//       door in one tick flip it twice. A sibling of P7b's unit_table_appends
//       in the same form (its tables are not this one's); the request
//       records stay the transport (the shadow compares them).
//
// A record whose door row is outside the env's door table FAULTS (the host
// would raise an IndexError) and writes nothing for it.
//
// The door SWEEP (slot 9e(d)) is the applier's DOORS stage (cuda_applier.cu
// over cuda_unit_doors.cuh), not this TU.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_object_table.h"
#include "cuda_unit_lib.cuh"
#include "object_tables_layout.h"   // GENERATED: the door rows
#include "unit_requests_layout.h"   // GENERATED: the request records

#include <sstream>
#include <stdexcept>
#include <vector>

namespace breach_cuda {

namespace {

using namespace object_tables;
using namespace unit_requests;

//: The per-env counts: doors flipped, faults.
constexpr int C_CHANGED = 0, C_FAULTS = 1, C_W = 2;

int64_t* g_counts = nullptr;
size_t g_counts_cap = 0;

inline void du_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_door_uses/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

__global__ void k_door_uses(RecordListView rq, RecordListView dr, int64_t* cnt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    const int64_t* recs = rq.recs + (size_t)e * rq.cap * rq.width;
    const int64_t n = rq.count[e];
    const int64_t nd = dr.count[e];
    int64_t* doors = dr.recs + (size_t)e * dr.cap * dr.width;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t* r = recs + i * rq.width;
        if (r[UNIT_RQ_KIND] != UNIT_RQKIND_USE) continue;
        const int64_t k = r[UNIT_RQ_DOOR];
        if (k < 0 || k >= nd) {
            cnt[(size_t)e * C_W + C_FAULTS] += 1;
            continue;
        }
        int64_t* d = doors + (size_t)k * dr.width;
        d[OBJ_DOOR_COL_WANT_OPEN] = d[OBJ_DOOR_COL_WANT_OPEN] != 0 ? 0 : 1;
        cnt[(size_t)e * C_W + C_CHANGED] += 1;
    }
}

}  // namespace

void unit_door_uses(const UnitTablesDevice& tables, UnitRecordList& requests, ObjectTable& doors,
                    const ObjectSlotCounts& out) {
    const int n = tables.n_env();
    if (requests.n_env() != n || requests.width() != UNIT_REQ_F)
        throw std::invalid_argument("unit_door_uses: the request list is not the tables' envs "
                                    "at the request width");
    if (doors.n_env() != n || doors.width() != OBJ_DOOR_W || doors.live_col() != OBJ_DOOR_LIVE)
        throw std::invalid_argument("unit_door_uses: the door table is not the door layout "
                                    "for these envs");
    const size_t want = (size_t)n * C_W;
    if (want > g_counts_cap) {
        cudaStreamSynchronize(tstream());
        if (g_counts) cudaFree(g_counts);
        g_counts = nullptr;
        du_check(cudaMalloc(reinterpret_cast<void**>(&g_counts), want * sizeof(int64_t)),
                 "malloc counts");
        g_counts_cap = want;
    }
    du_check(cudaMemsetAsync(g_counts, 0, want * sizeof(int64_t), tstream()), "zero counts");
    k_door_uses<<<dim3(1, n), 1, 0, tstream()>>>(list_view(requests), list_view(doors), g_counts);
    du_check(cudaGetLastError(), "launch");
    std::vector<int64_t> h(want);
    du_check(cudaMemcpyAsync(h.data(), g_counts, want * sizeof(int64_t), cudaMemcpyDeviceToHost,
                             tstream()),
             "D2H counts");
    du_check(cudaStreamSynchronize(tstream()), "sync");
    for (int e = 0; e < n; ++e) {
        if (out.changed) out.changed[e] = h[(size_t)e * C_W + C_CHANGED];
        if (out.faults) out.faults[e] = h[(size_t)e * C_W + C_FAULTS];
        if (out.refused) out.refused[e] = 0;
        if (out.dropped) out.dropped[e] = 0;
    }
}

}  // namespace breach_cuda
