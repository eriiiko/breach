// ============================================================================
// #29 P6.4a -- THE DONE KERNEL: the match's episode boundary on the device
// unit store (ruleset.MatchRules.is_terminal / living_teams / result). See
// cuda_unit_kernels.h; residency design §4.3 row "10 tick end" ("the done
// kernel"), §4.5 ("terminated from the MatchRules law").
//
// ONE thread per env (blockIdx.y): the marine teams with a living unit
// (valid, alive, not a zombie: a converted marine has left its team), counted
// once each; terminal when fewer than two remain or the env's tick (read after
// the step, as the caller of MatchRules.is_terminal reads it) has reached the
// match's limit (the tables' match_limit_ticks, MatchRules.limit_ticks; 0 = no
// match: no limit). The result as MatchRules.result: a win for the last team
// standing (on the limit tick too), else a draw; none while not terminal.
//
// NOT a slot of Simulation.step: its caller reads it after the tick (the
// session's is_terminal / result). CLASS T: nothing in the played tick calls
// this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

int64_t* g_out = nullptr;
size_t g_out_cap = 0;

__global__ void k_match_done(const int64_t* store, int n_units, const int64_t* tables,
                             const int64_t* trow, int64_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = (int)benv();
    const int64_t* rows = store + (size_t)e * n_units * UNIT_STORE_W;
    const int64_t* tab = tables + (size_t)e * UNIT_TABLES_W;
    int64_t n_teams = 0, team = -1;
    for (int r = 0; r < n_units; ++r) {
        const int64_t* row = rows + (size_t)r * UNIT_STORE_W;
        if (!row_valid(row) || !acts(row)) continue;
        bool seen = false;
        for (int q = 0; q < r && !seen; ++q) {
            const int64_t* o = rows + (size_t)q * UNIT_STORE_W;
            seen = row_valid(o) && acts(o) && o[UNIT_COL_TEAM] == row[UNIT_COL_TEAM];
        }
        if (seen) continue;
        if (n_teams == 0) team = row[UNIT_COL_TEAM];
        ++n_teams;
    }
    const int64_t limit = tab[UNIT_TAB_MATCH_LIMIT_TICKS];
    const bool terminal = n_teams < 2 || (limit > 0 && unit_draws_dev::trow_tick(trow, e) >= limit);
    int64_t* o = out + (size_t)e * UNIT_DONE_W;
    o[0] = terminal ? 1 : 0;
    o[1] = !terminal ? UNIT_DONE_NONE : (n_teams == 1 ? UNIT_DONE_WIN : UNIT_DONE_DRAW);
    o[2] = (terminal && n_teams == 1) ? team : -1;
    o[3] = n_teams;
}

}  // namespace

void unit_match_done(UnitStoreDevice& store, const UnitTablesDevice& tables,
                     std::uintptr_t tick_row, int64_t* out) {
    unit_check_pair(store, tables, "unit_match_done");
    if (tick_row == 0 || out == nullptr)
        throw std::invalid_argument("unit_match_done: the tick row and out (N, 4) are required");
    const int n = store.n_env();
    const int64_t* d_trow = reinterpret_cast<const int64_t*>(tick_row);
    const size_t want = (size_t)n * UNIT_DONE_W;
    if (want > g_out_cap) {
        counted_sync(tstream(), "cuda_unit_done.cu:70");
        if (g_out) cudaFree(g_out);
        g_out = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&g_out), want * sizeof(int64_t)) != cudaSuccess)
            throw std::runtime_error("unit_match_done: output allocation failed");
        g_out_cap = want;
    }
    k_match_done<<<dim3(1, n), 1, 0, tstream()>>>(store.rows(), store.n_units(), tables.values(),
                                                  d_trow, g_out);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(out, g_out, want * sizeof(int64_t), cudaMemcpyDeviceToHost, tstream());
    if (e == cudaSuccess) e = counted_sync(tstream(), "cuda_unit_done.cu:82");
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_match_done: " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

}  // namespace breach_cuda
