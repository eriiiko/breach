// ============================================================================
// #29 P6.2 -- TEST-ONLY: THE DEVICE UNIT LIBRARY ON SYNTHETIC ROWS. See
// cuda_unit_kernels.h (unit_lib_probe); the gate is
// tests/cuda_unit_kernels_check.py part LIB, which holds composed_flags,
// apply_status (with the stack cap's refusal), the sweep and the hook
// accessor equal to status.py / unit_traits.py over their whole range,
// padding rows untouched. No slot calls it.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

#include <vector>

namespace breach_cuda {

namespace {

using namespace unit_lib;

__global__ void k_lib_probe(int64_t* store, int n_units, const int64_t* tables, int op,
                            int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t* out,
                            unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    int64_t* o = bptr(out, (size_t)n_units * UNIT_LIB_PROBE_K) + (size_t)r * UNIT_LIB_PROBE_K;
    if (!row_valid(row)) return;
    const int64_t* tab = env_tables(tables);
    bool changed = false;
    if (op == 0) {
        bool bad = false;
        const Flags f = composed_flags(row, tab, a0 != 0, &bad);
        o[0] = f.can_move;
        o[1] = f.can_act;
        o[2] = f.can_aim;
        o[3] = f.is_prone;
        o[4] = bad;
    } else if (op == 1) {
        const int res = apply_status(row, tab, a0, a1, a2, a3, changed);
        o[0] = res;
        if (res == APPLY_REFUSED) count_refused(cnt);
    } else if (op == 2) {
        sweep_statuses(row, changed);
    } else if (op == 3) {
        for (int h = 0; h < UNIT_N_HOOKS && h < UNIT_LIB_PROBE_K; ++h) o[h] = scaled(row, h, a0);
    }
    if (changed) count_changed(cnt);
}

}  // namespace

void unit_lib_probe(UnitStoreDevice& store, const UnitTablesDevice& tables, int op,
                    int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t* out,
                    const UnitKernelCounts& counts) {
    unit_check_pair(store, tables, "unit_lib_probe");
    if (out == nullptr) throw std::invalid_argument("unit_lib_probe: null out");
    const size_t n = (size_t)store.n_env() * store.n_units() * UNIT_LIB_PROBE_K;
    std::vector<int64_t> zero(n, 0);
    const int64_t* d_out_c = store.stage_input(1, zero.data(), n);
    int64_t* d_out = const_cast<int64_t*>(d_out_c);
    store.begin_counts();
    k_lib_probe<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(), op, a0, a1, a2, a3, d_out,
        store.counts_dev());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    if (cudaMemcpyAsync(out, d_out, n * sizeof(int64_t), cudaMemcpyDeviceToHost, tstream()) !=
        cudaSuccess)
        throw std::runtime_error("unit_lib_probe: D2H out");
    store.end_counts(counts.coverage, counts.faults, counts.refused);
}

}  // namespace breach_cuda
