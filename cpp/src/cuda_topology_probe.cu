// ============================================================================
// #29 P6.4b -- TEST-ONLY: the topology seam's on_tile_changed on its own, on
// fixture maps (tests/cuda_unit_topology_check.py, part SEAM). destroy_wall
// only ever turns a tile into AIR, so its membership diff can never RETIRE
// (air is never less accountable than what it replaced); a seal can. This
// probe drives the device function the way every structural writer drives
// the Python one -- `material[y, x] = m; on_tile_changed(y, x)` -- for a list
// of (y, x, m), so the gate reaches both directions of the membership diff
// and every material row as a target. The staged FieldApplier list carries
// the records (kind ignored, `arg` the new material id); the books are the
// applier's. Never called by a slot.
//
// CLASS T, TEST-ONLY: nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_batch.cuh"
#include "cuda_topology.cuh"
#include "unit_tables_layout.h"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

using namespace applier;

__global__ void k_probe_tile_changed(const int64_t* recs, const int64_t* counts, int cap,
                                     topo::PlanePtrs P, int h, int w, int n_gases,
                                     const int64_t* tables, int64_t* books) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = benv();
    const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
    int64_t* B = books + (size_t)e * APPLIER_N_BOOKS;
    const topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
    const int M = unit_tables::UNIT_TAB_MAT_FLAMMABLE_W;
    const int64_t k = counts[e] < (int64_t)cap ? counts[e] : (int64_t)cap;
    for (int64_t j = 0; j < k; ++j) {
        const int64_t* r = recs + ((size_t)e * cap + (size_t)j) * APPLIER_F;
        const int64_t y = r[APPLIER_FX_A0], x = r[APPLIER_FX_A1], m = r[APPLIER_FX_A2];
        if (y < 0 || y >= h || x < 0 || x >= w || m < 0 || m >= M) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        E.material[(int)y * w + (int)x] = (int8_t)m;
        topo::on_tile_changed(E, tab, (int)y, (int)x, B);
        B[APPLIER_BOOK_APPLIED] += 1;
    }
}

}  // namespace

void topology_probe_tile_changed(FieldApplier& a, const UnitTablesDevice& tables,
                                 const ApplierPlanes& planes, int64_t* books_out) {
    topo::PlanePtrs P;
    for (int k = 0; k < APPLIER_N_PLANES; ++k) P.p[k] = (unsigned long long)planes.ptr[k];
    cudaMemsetAsync(a.books_dev(), 0, (size_t)a.n_env() * APPLIER_N_BOOKS * sizeof(int64_t),
                    tstream());
    k_probe_tile_changed<<<dim3(1, (unsigned)a.n_env()), 32, 0, tstream()>>>(
        a.list().records_dev(), a.list().count_dev(), a.cap(), P, planes.h, planes.w, planes.n_gases,
        tables.values(), a.books_dev());
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in topology_probe: " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
    a.download_books(books_out);
}

}  // namespace breach_cuda
