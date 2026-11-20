// ============================================================================
// #29 P6.3 TEST-ONLY -- the device damage pipeline (cuda_unit_damage.cuh's
// apply_packet) on SYNTHETIC rows and packets: one packet per valid row,
// applied to that row, its events through the row's staging into the ring
// (gathered in row order). The gate (tests/cuda_unit_damage_check.py, part
// APPLY) holds it to damage.apply_packet over random tables, rows and
// packets -- the branches no shipped scene reaches (a non-dyadic resist, an
// armor and its penetration, a corpse hit, the conversion flag, a heal).
//
// CLASS T, test-only: nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_damage.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

__global__ void k_apply_probe(int64_t* store, int n_units, const int64_t* tables,
                              const int64_t* packets, UnitRingView ring, int64_t* out,
                              unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    const int64_t* tab = env_tables(tables);
    const int64_t* q = packets + ((size_t)benv() * n_units + r) * UNIT_DAMAGE_PROBE_W;
    Packet p;
    p.amount_q = q[0];
    p.dtype = q[1];
    p.attacker = q[2];
    p.indirect = q[3] != 0;
    p.expose = q[4];
    p.dir_x_q = q[5];
    p.dir_y_q = q[6];
    p.ap_q = q[7];
    p.cause = q[8];
    p.mark_killed_by_zombie = q[9] != 0;
    int64_t loc[UNIT_STORE_W];
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    EventSink sink = row_sink(ring, n_units, r);
    bool bad = false, changed = false, killed = false;
    const int64_t dmg = apply_packet(loc, tab, p, sink, changed, &killed, &bad);
    int64_t* o = out + ((size_t)benv() * n_units + r) * 2;
    o[0] = bad ? 0 : dmg;
    o[1] = bad ? 0 : (killed ? 1 : 0);
    finish_row(row, loc, sink, bad, cnt);
}

}  // namespace

void unit_damage_probe(UnitStoreDevice& store, const UnitTablesDevice& tables,
                       UnitEventRing& ring, const int64_t* packets, int64_t* out,
                       const UnitDamageCounts& counts) {
    unit_check_pair(store, tables, "unit_damage_probe");
    unit_check_ring(store, ring, "unit_damage_probe");
    const size_t n = (size_t)store.n_env() * store.n_units();
    const int64_t* d_pk = store.stage_input(2, packets, n * UNIT_DAMAGE_PROBE_W);
    int64_t* d_out = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_out), n * 2 * sizeof(int64_t)) != cudaSuccess)
        throw std::runtime_error("unit_damage_probe: malloc");
    cudaMemsetAsync(d_out, 0, n * 2 * sizeof(int64_t), tstream());
    store.begin_counts();
    k_apply_probe<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(), d_pk, ring_view(ring), d_out,
        store.counts_dev());
    unit_finish_events(store, ring, counts, "unit_damage_probe");
    cudaMemcpyAsync(out, d_out, n * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost, tstream());
    counted_sync(tstream(), "cuda_unit_damage_probe.cu:71");
    cudaFree(d_out);
}

}  // namespace breach_cuda
