// ============================================================================
// #29 P6.4a -- TEST-ONLY: the hand verbs' gates of the device unit library
// (cuda_unit_lib.cuh's P6.4a block) on the store's rows, so the gate
// (tests/cuda_unit_verbs_check.py, part LIB) holds each equal to
// verb_gates.py / combat.mag_status / unit_traits._handling on synthetic rows
// over their range. See cuda_unit_kernels.h (unit_verbs_probe).
//
// CLASS T, TEST-ONLY: no slot calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

__global__ void k_verbs_probe(int64_t* store, int n_units, const int64_t* tables, int64_t kind,
                              int64_t tick, int64_t base, int64_t* out) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    int64_t* o = out + ((size_t)benv() * n_units + r) * UNIT_VERBS_PROBE_K;
    for (int k = 0; k < UNIT_VERBS_PROBE_K; ++k) o[k] = 0;
    if (!row_valid(row)) return;
    const int64_t* tab = env_tables(tables);
    bool bad = false;
    o[0] = acts(row) ? 1 : 0;
    o[1] = can_throw(row, tab, kind, &bad) ? 1 : 0;
    o[2] = can_plant(row, tab, tick, &bad) ? 1 : 0;
    o[3] = can_reload(row, tab, tick, &bad) ? 1 : 0;
    o[4] = can_shove(row, tab, tick, &bad) ? 1 : 0;
    const MagStatus m = mag_status(row, tab, tick, &bad);
    o[5] = m.tracked ? 1 : 0;
    o[6] = m.rounds;
    o[7] = m.spares;
    o[8] = m.reloading ? 1 : 0;
    o[9] = magazines_carried(row);
    o[10] = handling_ticks(row, base);
    o[11] = shove_cooled(row, tab, tick) ? 1 : 0;
    o[12] = can_act(row, tab, &bad) ? 1 : 0;
    // #29 P7a.1: the melee verb's gate (both moments), target, adjacency
    o[13] = can_melee(row, tab, tick, false, &bad) ? 1 : 0;
    o[14] = can_melee(row, tab, tick, true, &bad) ? 1 : 0;
    const int64_t mr = melee_row(row, &bad);
    o[16] = mr >= 0 ? melee_arc_target(store, n_units, r, row, tab, mr, &bad) : -2;
    o[17] = -1;
    if (base >= 0 && base < n_units) {
        const int64_t* other = unit_row(store, n_units, (int)base);
        if (row_valid(other)) o[17] = melee_adjacent(row, other, &bad) ? 1 : 0;
    }
    o[15] = bad ? 1 : 0;
}

}  // namespace

void unit_verbs_probe(UnitStoreDevice& store, const UnitTablesDevice& tables, int64_t kind,
                      int64_t tick, int64_t base, int64_t* out) {
    unit_check_pair(store, tables, "unit_verbs_probe");
    const int n = store.n_env(), U = store.n_units();
    int64_t* d_out = nullptr;
    const size_t bytes = (size_t)n * U * UNIT_VERBS_PROBE_K * sizeof(int64_t);
    if (cudaMalloc(reinterpret_cast<void**>(&d_out), bytes) != cudaSuccess)
        throw std::runtime_error("unit_verbs_probe: allocation failed");
    k_verbs_probe<<<bgrid(U, n), 128, 0, tstream()>>>(store.rows(), U, tables.values(), kind,
                                                      tick, base, d_out);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(out, d_out, bytes, cudaMemcpyDeviceToHost, tstream());
    if (e == cudaSuccess) e = cudaStreamSynchronize(tstream());
    cudaFree(d_out);
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_verbs_probe: " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

}  // namespace breach_cuda
