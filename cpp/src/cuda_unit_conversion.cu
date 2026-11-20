// ============================================================================
// #29 P6.2 -- THE TICK-END CONVERSION on the device unit store (slot 10's
// ruleset tail: ContinuousRealtime.on_tick_end ->
// ai_zombie.convert_marines_to_zombies). See cuda_unit_kernels.h; residency
// design §4.4b.
//
// Integer for integer convert_marines_to_zombies: every valid dead row that
// rises (ai_zombie.rises_as_zombie: a non-zombie a zombie killed, OR -- arc #63
// P5b -- any corpse whose infestation reached the threshold; never a destroyed
// corpse, P5c: hp <= -body_factor x max_hp) joins the horde's team
// (factions.ZOMBIE_TEAM, a table value), becomes a zombie and alive, takes the
// zombie class's full health times its HP hook (unit_traits.max_hp_q, the
// library's scaled()), and is cleared of exposure (exposure.clear), of its
// turn (turning.settle), of its push (unit_impulse.clear), of its noise and
// anchor (horde.clear), of its order (unit_orders.clear) and of poison
// (poison.clear), of stress and back to an ambient body (#215); zombie_activated
// is set and killed_by_zombie cleared. The
// `traits` block is KEPT. What has no device form: the name's "Z-" prefix
// and the momentum books (reference-only), and on_tick_end's prune of the
// detonated grenades from sim.projectiles (a host table until P7b) -- the
// slot's host side, done by its owner after the download.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"
#include "unit_events_layout.h"    // arc #63 P5a: UNIT_CAUSE_POISON_GAS (poison.clear)

namespace breach_cuda {

namespace {

using namespace unit_lib;

__global__ void k_convert_zombies(int64_t* store, int n_units, const int64_t* tables,
                                  unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    if (row_alive(row)) return;
    const int64_t* tab = env_tables(tables);
    // ai_zombie.rises_as_zombie (arc #63 P5b: + corpse.rises; a destroyed
    // corpse rises by neither trigger)
    if (corpse_destroyed(row, tab)) return;        // P5c: hp <= -body_factor x max_hp
    const bool by_kill = !row[UNIT_COL_IS_ZOMBIE] && row[UNIT_COL_KILLED_BY_ZOMBIE];
    const bool by_infest = row[UNIT_COL_INFEST_Q] >= tab[UNIT_TAB_CORPSE_THRESHOLD_Q];
    if (!by_kill && !by_infest) return;
    bool changed = false;
    put(row, UNIT_COL_TEAM, tab[UNIT_TAB_ZOMBIE_TEAM], changed);
    put(row, UNIT_COL_IS_ZOMBIE, 1, changed);
    put(row, UNIT_COL_LIFE_STATE, LIFE_ALIVE, changed);
    // unit_traits.max_hp_q on is_zombie: the zombie class's hp x the HP hook.
    put(row, UNIT_COL_CURRENT_HP,
        scaled(row, UNIT_HOOK_HP, tab[UNIT_TAB_HP_CLASS_ZOMBIE_Q]), changed);
    // exposure.clear
    for (int t = 0; t < UNIT_COL_EXPOSURE_W; ++t) put(row, UNIT_COL_EXPOSURE + t, 0, changed);
    // turning.settle
    put(row, UNIT_COL_FACING_TARGET_Q, row[UNIT_COL_FACING_Q], changed);
    put(row, UNIT_COL_OMEGA_Q, 0, changed);
    // unit_impulse.clear (the books are reference-only)
    put(row, UNIT_COL_IMP_X, 0, changed);
    put(row, UNIT_COL_IMP_Y, 0, changed);
    put(row, UNIT_COL_IMP_L, 0, changed);
    put(row, UNIT_COL_STAND_X, 0, changed);
    put(row, UNIT_COL_STAND_Y, 0, changed);
    put(row, UNIT_COL_SWX, 0, changed);
    put(row, UNIT_COL_SWY, 0, changed);
    put(row, UNIT_COL_PVX, 0, changed);
    put(row, UNIT_COL_PVY, 0, changed);
    put(row, UNIT_COL_EFFORT, 0, changed);
    put(row, UNIT_COL_REACT_TICKS, 0, changed);
    // horde.clear
    put(row, UNIT_COL_ZOMBIE_NOISE_Y, -1, changed);
    put(row, UNIT_COL_ZOMBIE_NOISE_X, -1, changed);
    put(row, UNIT_COL_ZOMBIE_NOISE_TICKS, 0, changed);
    put(row, UNIT_COL_ZOMBIE_ANCHOR_Y, -1, changed);
    put(row, UNIT_COL_ZOMBIE_ANCHOR_X, -1, changed);
    // unit_orders.clear
    put(row, UNIT_COL_ORDER_WHAT, 0, changed);
    put(row, UNIT_COL_ORDER_WHERE, -1, changed);
    put(row, UNIT_COL_ORDER_UNTIL, -1, changed);
    put(row, UNIT_COL_ORDER_TARGET, -1, changed);
    // poison.clear (NO_ATTACKER = -1)
    put(row, UNIT_COL_POISON_X0, 0, changed);
    put(row, UNIT_COL_POISON_X1, 0, changed);
    put(row, UNIT_COL_POISON_SRC, -1, changed);
    // #215: stress.clear and body_temp.clear (back to ambient)
    put(row, UNIT_COL_STRESS_Q, 0, changed);
    put(row, UNIT_COL_BODY_TEMP_Q, tab[UNIT_TAB_HEAT_AMBIENT_Q], changed);
    put(row, UNIT_COL_POISON_CAUSE, unit_events::UNIT_CAUSE_POISON_GAS, changed);   // arc #63 P5a
    // corpse.clear (arc #63 P5b): the rise spent (the hp above made the body whole)
    put(row, UNIT_COL_INFEST_Q, 0, changed);
    put(row, UNIT_COL_ZOMBIE_ACTIVATED, 1, changed);
    put(row, UNIT_COL_KILLED_BY_ZOMBIE, 0, changed);
    if (changed) count_changed(cnt);
}

}  // namespace

void unit_convert_zombies(UnitStoreDevice& store, const UnitTablesDevice& tables,
                          const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_convert_zombies");
    store.begin_counts();
    k_convert_zombies<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(), store.counts_dev());
    unit_finish(store, out, "unit_convert_zombies");
}

}  // namespace breach_cuda
