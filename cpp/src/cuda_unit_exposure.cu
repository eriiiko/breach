// ============================================================================
// #29 P6.1 (moved at P6.2) -- SLOT 1c, EXPOSURE DECAY, on the device unit store.
// See cuda_unit_kernels.h; residency design §4.4b.
//
// src/simulation/exposure.py's law integer for integer (decay_units /
// decay_step): an arithmetic change lands there first, then here;
// tests/cuda_unit_store_check.py and tests/cuda_unit_kernels_check.py hold the
// two equal at tol 0 on live matches (the shadow lockstep) and chained after
// slot 1b'. Like decay_units it reads no life state: a dead unit's block is
// zeroed at its death, so it has nothing to decay. The decay is the device
// tables' exposure_decay_q (exposure.get_settings().decay_q), never an
// argument.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

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
__global__ void k_exposure_decay(int64_t* store, int n_units, const int64_t* tables,
                                 unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    const int64_t decay_q = env_tables(tables)[UNIT_TAB_EXPOSURE_DECAY_Q];
    int64_t* x = row + UNIT_COL_EXPOSURE;
    bool changed = false;
    for (int t = 0; t < UNIT_COL_EXPOSURE_W; ++t) {
        const int64_t v = x[t];
        if (v != 0) put(x, t, exposure_decay_step(v, decay_q), changed);
    }
    if (changed) count_changed(cnt);
}

}  // namespace

void unit_exposure_decay(UnitStoreDevice& store, const UnitTablesDevice& tables,
                         const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_exposure_decay");
    store.begin_counts();
    k_exposure_decay<<<bgrid(store.n_units(), store.n_env()), 256, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(), store.counts_dev());
    unit_finish(store, out, "unit_exposure_decay");
}

}  // namespace breach_cuda
