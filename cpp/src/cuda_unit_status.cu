// ============================================================================
// #29 P6.3 -- SLOT 2b, THE STATUSES' TICK on the device unit store. See
// cuda_unit_kernels.h; residency design §4.4b.
//
// status.tick_statuses integer for integer, per valid LIVING row (statuses
// freeze on a corpse): a non-empty list is swept of what the previous pass
// zeroed (the library's sweep_statuses), then in list order each status of a
// damage-over-time kind (the tables' status_dtype) with a non-zero magnitude
// emits ONE packet through the device apply_packet while the unit lives --
// HEAL-typed kinds the negative amount, credited to the status's source
// (direct, not indirect), its cause the row's name (status_cause) -- and every
// status counts down. A DoT kill never converts (the packet does not mark
// killed_by_zombie). Events go to the row's staging, gathered in row order
// (= id order: the reference sorts by id).
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_damage.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

__global__ void k_status_tick(int64_t* store, int n_units, const int64_t* tables,
                              UnitRingView ring, unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row) || !row_alive(row)) return;
    int64_t* s0 = row + UNIT_COL_STATUSES;
    if (s0[0] < 0) return;                       // no statuses: nothing to do
    const int64_t* tab = env_tables(tables);
    int64_t loc[UNIT_STORE_W];
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    EventSink sink = row_sink(ring, n_units, r);
    bool bad = false, changed = false;

    sweep_statuses(loc, changed);                // (a) what the previous pass zeroed
    int64_t* s = loc + UNIT_COL_STATUSES;
    for (int i = 0; i < STATUS_ROWS && !bad; ++i) {   // (b) + (c), list order
        const int64_t kind = s[4 * i];
        if (kind < 0) break;
        if (kind >= UNIT_N_STATUS_KINDS) {
            bad = true;
            break;
        }
        const int64_t dt = tab[UNIT_TAB_STATUS_DTYPE + kind];
        const int64_t mag = s[4 * i + 1];
        if (dt >= 0 && mag != 0 && row_alive(loc)) {
            if (mag > AMOUNT_MAX || mag < -AMOUNT_MAX) {
                bad = true;
                break;
            }
            Packet p;
            p.amount_q = dt == UNIT_DTYPE_HEAL ? -mag : mag;
            p.dtype = dt;
            p.attacker = s[4 * i + 3];
            p.cause = tab[UNIT_TAB_STATUS_CAUSE + kind];
            bool killed = false;
            apply_packet(loc, tab, p, sink, changed, &killed, &bad);
        }
        if (s[4 * i + 2] == INT64_MIN) {
            bad = true;
            break;
        }
        s[4 * i + 2] -= 1;
    }
    finish_row(row, loc, sink, bad, cnt);
}

}  // namespace

void unit_status_tick(UnitStoreDevice& store, const UnitTablesDevice& tables,
                      UnitEventRing& ring, const UnitDamageCounts& out) {
    unit_check_pair(store, tables, "unit_status_tick");
    unit_check_ring(store, ring, "unit_status_tick");
    store.begin_counts();
    k_status_tick<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(), ring_view(ring), store.counts_dev());
    unit_finish_events(store, ring, out, "unit_status_tick");
}

}  // namespace breach_cuda
