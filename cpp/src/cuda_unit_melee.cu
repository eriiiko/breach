// ============================================================================
// #29 P7a.1 -- SLOT 4a, THE MELEE VERB on the device unit store
// (combat.process_melee_intents: slot 4's phase A walk over the latched
// pending_melee), the EFFECT LIST's first device producer. See
// cuda_unit_kernels.h; residency design §4.3 rows "4 / 4b phase A" and "4c
// resolve", §4.4c.
//
// Integer for integer the reference, per valid row (one thread each; the
// decisions read only the slot-start rows):
//   * a row with no latch is left as it is;
//   * the latch is CONSUMED (pending_melee 0), whatever follows;
//   * verb_gates.can_melee (at_tick_start false: the verb's own read) -- else
//     nothing more;
//   * melee_arc_target over the env's other rows -- none: a whiff, no cooldown;
//   * the own-row write at once: last_blow_tick = tick;
//   * the blow RECORDED, in melee_blow's then process_melee_intents' order: a
//     HIT (the row's damage times the STRIKE hook, Q16; the row's dtype; the
//     striker credited; the melee cause; the direction striker centre ->
//     victim centre), the delivery-site STATUS when the row has one (magnitude
//     0, the row's ticks, the striker as source), a PUSH of the row's impulse
//     times the PUSH hook along striker -> victim (the facing when the centres
//     coincide) when it is not zero on both axes.
// The records are staged per row and gathered per env in ROW (= id) order onto
// whatever the effect list holds (slot 4's records in the reference's order),
// so the list holds "shooters by id, then melee by id" whatever the schedule.
// The render-only MeleeEvent has no device form. No draw: a striker whose
// melee row has a crit (melee_blow would draw; the loadouts door refuses one)
// FAULTS by name.
//
// Proved ranges (past them the row FAULTS, left as it was, its records
// dropped): a tick below TICK_MAX; a footprint in [1, FOOTPRINT_MAX]; centre
// offsets inside the kit's int32; the push below PUSH_MAX (its axis split
// inside int64).
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"
#include "unit_events_layout.h"    // GENERATED: the causes

namespace breach_cuda {

namespace {

using namespace unit_lib;
using namespace applier;

constexpr int64_t PUSH_MAX = (int64_t)1 << 46;

__device__ __forceinline__ void blank(int64_t* rec, int64_t kind) {
    for (int f = 0; f < APPLIER_F; ++f) rec[f] = 0;
    rec[APPLIER_FX_KIND] = kind;
}

// combat.melee_blow (no facing snap, no crit) + the verb's push: the records.
__device__ inline void record_blow(const int64_t* loc, const int64_t* tr, int t,
                                   const int64_t* tab, int64_t m, RecordSink& sink, bool* bad) {
    const int64_t dtype = tab[UNIT_TAB_WEAPON_MELEE_DTYPE + m];
    if (dtype < 0) {                                  // a melee row without a dtype
        *bad = true;
        return;
    }
    const int64_t fa = loc[UNIT_COL_FOOTPRINT], ft = tr[UNIT_COL_FOOTPRINT];
    const int64_t ax = centre_q(loc[UNIT_COL_X_Q], fa), ay = centre_q(loc[UNIT_COL_Y_Q], fa);
    const int64_t tx = centre_q(tr[UNIT_COL_X_Q], ft), ty = centre_q(tr[UNIT_COL_Y_Q], ft);
    const int64_t uid = loc[UNIT_COL_ID];
    int64_t rec[APPLIER_F];
    // THE STRIKE accessor: the row's damage (whole points << 16) x STRIKE
    blank(rec, APPLIER_KIND_HIT);
    rec[APPLIER_HIT_ROW] = t;
    rec[APPLIER_HIT_AMOUNT_Q] = scaled(loc, UNIT_HOOK_STRIKE, tab[UNIT_TAB_WEAPON_MELEE_DAMAGE_Q + m]);
    rec[APPLIER_HIT_DTYPE] = dtype;
    rec[APPLIER_HIT_ATTACKER] = uid >= 0 ? uid : -1;
    rec[APPLIER_HIT_INDIRECT] = 0;
    rec[APPLIER_HIT_EXPOSE] = 0;
    rec[APPLIER_HIT_DIR_X_Q] = tx - ax;
    rec[APPLIER_HIT_DIR_Y_Q] = ty - ay;
    rec[APPLIER_HIT_AP_Q] = 0;
    rec[APPLIER_HIT_CAUSE] = unit_events::UNIT_CAUSE_MELEE;
    emit_record(sink, rec);
    const int64_t sk = tab[UNIT_TAB_WEAPON_MELEE_STATUS_KIND + m];
    if (sk >= 0) {
        blank(rec, APPLIER_KIND_STATUS);
        rec[APPLIER_STATUS_ROW] = t;
        rec[APPLIER_STATUS_STATUS] = sk;
        rec[APPLIER_STATUS_MAGNITUDE_Q] = 0;
        rec[APPLIER_STATUS_DURATION] = tab[UNIT_TAB_WEAPON_MELEE_STATUS_TICKS + m];
        rec[APPLIER_STATUS_SOURCE] = uid;
        emit_record(sink, rec);
    }
    // THE PUSH accessor: the row's impulse x PUSH, along striker -> victim
    const int64_t j = scaled(loc, UNIT_HOOK_PUSH, tab[UNIT_TAB_WEAPON_MELEE_PUSH_Q + m]);
    if (j == 0) return;
    if (j > PUSH_MAX || j < -PUSH_MAX) {
        *bad = true;
        return;
    }
    int64_t bearing = loc[UNIT_COL_FACING_Q];
    if (tx != ax || ty != ay) {
        if (!fits_i32(-(ty - ay)) || !fits_i32(tx - ax)) {
            *bad = true;
            return;
        }
        bearing = fixedpoint::atan2_q16((q16)(-(ty - ay)), (q16)(tx - ax));
    }
    if (!fits_i32(bearing)) {
        *bad = true;
        return;
    }
    const int64_t sx = fixedpoint::cos_q16((q16)bearing);
    const int64_t sy = -(int64_t)fixedpoint::sin_q16((q16)bearing);
    const int64_t jx = fixedpoint::shr_trunc_q(j * sx, 16);
    const int64_t jy = fixedpoint::shr_trunc_q(j * sy, 16);
    if (jx == 0 && jy == 0) return;                  // SimultaneousEffects.push's skip
    blank(rec, APPLIER_KIND_PUSH);
    rec[APPLIER_PUSH_ROW] = t;
    rec[APPLIER_PUSH_JX_Q] = jx;
    rec[APPLIER_PUSH_JY_Q] = jy;
    rec[APPLIER_PUSH_LEVER_Q] = 0;
    rec[APPLIER_PUSH_TWIST_Q16] = ONE;
    emit_record(sink, rec);
}

__global__ void k_melee(int64_t* store, int n_units, const int64_t* tables, const int64_t* trow,
                        RecordListView fx, unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row) || row[UNIT_COL_PENDING_MELEE] == 0) return;
    const int64_t* tab = env_tables(tables);
    const int64_t tick = unit_draws_dev::trow_tick(trow, benv());
    int64_t loc[UNIT_STORE_W];
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    RecordSink sink = list_row_sink(fx, n_units, r);
    bool bad = tick < -TICK_MAX || tick > TICK_MAX;
    loc[UNIT_COL_PENDING_MELEE] = 0;                 // every latch is consumed
    if (!bad && can_melee(loc, tab, tick, false, &bad) && !bad) {
        const int64_t m = loc[UNIT_COL_MELEE_ID];    // can_melee proved it a row
        if (tab[UNIT_TAB_WEAPON_MELEE_CRIT + m] != 0) {
            bad = true;                              // melee_blow would draw
        } else {
            const int t = melee_arc_target(store, n_units, r, loc, tab, m, &bad);
            if (!bad && t >= 0) {
                loc[UNIT_COL_LAST_BLOW_TICK] = tick;
                record_blow(loc, unit_row(store, n_units, t), t, tab, m, sink, &bad);
            }
        }
    }
    if (bad) {                       // a fault: the row and its records dropped
        *sink.count = 0;
        count_fault(cnt);
        return;
    }
    bool changed = false;
    for (int c = 0; c < UNIT_STORE_W; ++c) {
        if (row[c] != loc[c]) {
            row[c] = loc[c];
            changed = true;
        }
    }
    if (changed) count_changed(cnt);
}

}  // namespace

void unit_melee(UnitStoreDevice& store, const UnitTablesDevice& tables, FieldApplier& effects,
                std::uintptr_t tick_row, const UnitDamageCounts& out) {
    unit_check_pair(store, tables, "unit_melee");
    UnitRecordList& L = effects.list();
    if (L.n_env() != store.n_env() || L.n_units() != store.n_units() ||
        L.width() != APPLIER_F) {
        std::ostringstream os;
        os << "unit_melee: the effect list holds (" << L.n_env() << " envs, " << L.n_units()
           << " rows, width " << L.width() << "), the store (" << store.n_env() << ", "
           << store.n_units() << ") and the layout (width " << APPLIER_F << ")";
        throw std::invalid_argument(os.str());
    }
    if (tick_row == 0) throw std::invalid_argument("unit_melee: the tick row (N, TICK_ROW_W)");
    const int64_t* d_trow = reinterpret_cast<const int64_t*>(tick_row);
    const int n = store.n_env(), U = store.n_units();
    const RecordListView v = list_view(L);
    store.begin_counts();
    k_melee<<<bgrid(U, n), 128, 0, tstream()>>>(store.rows(), U, tables.values(), d_trow, v,
                                                 store.counts_dev());
    L.gather();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in unit_melee: " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
    store.end_counts(out.coverage, out.faults, out.refused);
    if (out.dropped) L.download(nullptr, nullptr, out.dropped);
}

}  // namespace breach_cuda
