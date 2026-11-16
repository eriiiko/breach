// ============================================================================
// #29 P7a.2 -- SLOT 4's SHOOTING and SLOT 2''s ROUNDS on the device unit store
// (combat.process_shooting's continuous path, combat._directional_fire /
// fire_burst / start_spray_burst_directional; Simulation._march_rounds), the
// effect list's second producer, the round table's and the noise list's first.
// See cuda_unit_kernels.h; residency design §4.3 rows "2" / "4", §4.4c.
//
// SLOT 4 (unit_shooting), per valid row (one thread each; every decision reads
// the slot-start rows and planes, nothing another row's thread writes):
//   * skipped: a zombie or a corpse, a unit that cannot act (its composed
//     statuses), one with no weapon row, one not holding the trigger
//     (live_trigger: the continuous path; the WEGO orders are refused at the
//     door, unit_kernels.shooting_door);
//   * _directional_fire's gates: the cadence (rof), the magazine (mag_settle:
//     the lazy bind or a finished reload lands AT ONCE; a stall or a dry gun
//     stops the trigger);
//   * the archetype: melee, lobbed, placed -> nothing more; a SPRAY whose burst
//     still runs -> nothing more, else the burst armed on the row (ticks, the
//     bearing, the nominal target) and its NOISE; a PROJECTILE -> its NOISE,
//     then the burst: per round the shooter's next SHOT_CONE draw (the count
//     local to this thread, the one recorded .next site), the kit step, the
//     recoil PUSH on the shooter, THE MARCH (cuda_unit_march.cuh) and, when
//     the round still flies, the round APPENDED to the round table; a HITSCAN
//     (the beam) is REFUSED by name (P7a.2 did not build it);
//   * mag_spend (the automatic reload when the magazine empties) and
//     last_fire_tick, at once.
// The records are staged per row and gathered per env in ROW (= id) order onto
// what each list holds: the effect list (recoil, then the march's records,
// per round), the round table (the survivors of slot 2' first), the noise
// list (what slot 2 and the last ticks left there first).
//
// SLOT 2' (unit_rounds), ONE thread per env: every round of the env's table
// marched in table order (the order the reference loop walks) against the
// rows and planes as the grenades left them, its records appended to the
// env's effect list directly (a serial producer); a finished round's live
// column 0, then the stable compaction (survivors in order). The landing is
// the applier's RESOLVE stage (unit_kernels.resolve), the same law as 4c's.
//
// Proved ranges (past them the row / env FAULTS, its records dropped): a tick
// below TICK_MAX; a cone up to the draws' CONE_MAX; kit angles inside int32;
// a centre inside the kit's int32 after the shift; the march's own.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_object_table.h"
#include "cuda_unit_kernels.h"
#include "cuda_unit_march.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;
using namespace unit_march;
using namespace applier;
using namespace object_tables;

constexpr int64_t CENTRE_MAX = (int64_t)1 << 14;   // a centre tile (<< 16 fits int32)
constexpr int64_t RECOIL_MAX = (int64_t)1 << 40;

__device__ inline void emit_noise(RecordSink& s, int64_t fy, int64_t fx, int64_t r) {
    if (r <= 0) return;                          // noise.emit: radius 0 is silence
    int64_t rec[OBJ_NOISE_W];
    for (int c = 0; c < OBJ_NOISE_W; ++c) rec[c] = 0;
    rec[OBJ_NOISE_COL_FY] = fy;
    rec[OBJ_NOISE_COL_FX] = fx;
    rec[OBJ_NOISE_COL_RADIUS] = r;
    rec[OBJ_NOISE_COL_LIVE] = 1;
    emit_record(s, rec);
}

// fire_burst with a free-aim bearing (no FACE: the facing is the turning law's)
__device__ inline void fire_burst(const MarchMap& m, int r, const int64_t* loc, int64_t w,
                                  int64_t cone_class, int64_t aim, int64_t cx, int64_t cy,
                                  RecordSink& fx, RecordSink& rounds, bool* bad) {
    const int64_t* tab = m.tab;
    const int64_t a = tab[UNIT_TAB_WEAPON_AMMO + w];
    if (a < 0 || a >= UNIT_TAB_AMMO_SPEED_Q16_W) {     // ammo_for_weapon finds none
        *bad = true;
        return;
    }
    const int64_t cone = scaled(loc, UNIT_HOOK_SPREAD, cone_class);
    const int64_t n = tab[UNIT_TAB_WEAPON_SHOTS_PER_TRIGGER + w];
    const int64_t uid = loc[UNIT_COL_ID];
    const int64_t ox = cx << 16, oy = cy << 16;
    unit_draws_dev::DrawSite site = unit_draws_dev::draw_site(unit_draws::SALT_SHOT_CONE, uid, bad);
    for (int64_t k = 0; k < n && !*bad; ++k) {
        const uint32_t word = unit_draws_dev::draw_next(m.key, site);
        const int64_t angle = aim + unit_draws_dev::cone_offset_q16(word, cone, bad);
        if (!fits_i32(angle)) {
            *bad = true;
            return;
        }
        const int64_t sx = fixedpoint::cos_q16((q16)angle);
        const int64_t sy = fixedpoint::sin_q16((q16)angle);
        if (tab[UNIT_TAB_AMMO_SPEED_Q16 + a] <= 0) {    // BulletInFlight refuses
            *bad = true;
            return;
        }
        int64_t rd[OBJ_ROUND_W];
        for (int c = 0; c < OBJ_ROUND_W; ++c) rd[c] = 0;
        rd[OBJ_ROUND_COL_SHOOTER_ID] = uid;
        rd[OBJ_ROUND_COL_WEAPON] = w;
        rd[OBJ_ROUND_COL_AMMO] = a;
        rd[OBJ_ROUND_COL_PAYLOAD] = tab[UNIT_TAB_AMMO_PAYLOAD + a];
        rd[OBJ_ROUND_COL_ANGLE_Q] = angle;
        rd[OBJ_ROUND_COL_RX_Q] = ox;
        rd[OBJ_ROUND_COL_RY_Q] = oy;
        rd[OBJ_ROUND_COL_PREV_IX] = cx;
        rd[OBJ_ROUND_COL_PREV_IY] = cy;
        rd[OBJ_ROUND_COL_REMAINING_STEPS] = tab[UNIT_TAB_WEAPON_RANGE_TILES + w];
        rd[OBJ_ROUND_COL_SPEED_Q16] = tab[UNIT_TAB_AMMO_SPEED_Q16 + a];
        rd[OBJ_ROUND_COL_BUDGET_Q16] = 0;
        rd[OBJ_ROUND_COL_OX_Q] = ox;
        rd[OBJ_ROUND_COL_OY_Q] = oy;
        rd[OBJ_ROUND_COL_SX_Q] = sx;
        rd[OBJ_ROUND_COL_SY_Q] = sy;
        rd[OBJ_ROUND_COL_SPAWN_TICK] = m.tick;
        rd[OBJ_ROUND_COL_ORDINAL] = k;
        rd[OBJ_ROUND_COL_LIVE] = 1;
        // the recoil: minus the round's row momentum along its flight, no lever
        const int64_t j = tab[UNIT_TAB_AMMO_RECOIL_Q + a];
        if (j != 0) {
            if (j > RECOIL_MAX) {
                *bad = true;
                return;
            }
            const int64_t jx = fixedpoint::shr_trunc_q(j * sx, 16);
            const int64_t jy = fixedpoint::shr_trunc_q(j * sy, 16);
            if (jx != 0 || jy != 0) {
                int64_t rec[APPLIER_F];
                blank(rec, APPLIER_KIND_PUSH);
                rec[APPLIER_PUSH_ROW] = r;
                rec[APPLIER_PUSH_JX_Q] = -jx;
                rec[APPLIER_PUSH_JY_Q] = -jy;
                rec[APPLIER_PUSH_LEVER_Q] = 0;
                rec[APPLIER_PUSH_TWIST_Q16] = ONE;
                emit_record(fx, rec);
            }
        }
        if (march(m, rd, fx, bad) && !*bad) emit_record(rounds, rd);
    }
}

__global__ void k_shooting(int64_t* store, int n_units, const int64_t* tables,
                           const int64_t* ticks, const int64_t* keys, const bool* solid,
                           const int8_t* material, int h, int w, RecordListView fxv,
                           RecordListView rv, RecordListView nv, unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row) || !row_alive(row) || row[UNIT_COL_IS_ZOMBIE] != 0) return;
    const int64_t* tab = env_tables(tables);
    const int e = benv();
    const int64_t tick = ticks[e];
    bool bad = tick < -TICK_MAX || tick > TICK_MAX;
    const Flags fl = composed_flags(row, tab, false, &bad);
    if (bad) {
        count_fault(cnt);
        return;
    }
    if (!fl.can_act) return;
    const int64_t wr = weapon_row(row, &bad);
    if (bad) {
        count_fault(cnt);
        return;
    }
    if (wr < 0 || row[UNIT_COL_LIVE_TRIGGER] == 0) return;
    if (wr >= UNIT_TAB_WEAPON_ARCHETYPE_W) {
        count_fault(cnt);
        return;
    }
    // _directional_fire: the cadence gate
    if (tick - row[UNIT_COL_LAST_FIRE_TICK] < tab[UNIT_TAB_WEAPON_ROF_TICKS + wr]) return;
    int64_t loc[UNIT_STORE_W];
    for (int c = 0; c < UNIT_STORE_W; ++c) loc[c] = row[c];
    RecordSink fx = list_row_sink(fxv, n_units, r);
    RecordSink rounds = list_row_sink(rv, n_units, r);
    RecordSink noise = list_row_sink(nv, n_units, r);
    const int64_t full = tab[UNIT_TAB_WEAPON_MAG_SIZE + wr];
    bool fire = true;
    if (full > 0) {                              // mag_gate (untracked: no state)
        if (!mag_settle(loc, tab, wr, tick)) fire = false;   // the stall
        else if (loc[UNIT_COL_CURRENT_MAG] <= 0) fire = false;  // dry
    }
    bool refused = false;
    if (fire) {
        const int64_t aim = -loc[UNIT_COL_FACING_Q];         // y-up -> screen
        const int64_t cone_class = fl.can_aim ? tab[UNIT_TAB_WEAPON_CONE_AIM_Q + wr]
                                              : tab[UNIT_TAB_WEAPON_CONE_SNAP_Q + wr];
        const int64_t arch = tab[UNIT_TAB_WEAPON_ARCHETYPE + wr];
        const int64_t fp = loc[UNIT_COL_FOOTPRINT];
        const int64_t cx = tile_of(loc[UNIT_COL_X_Q]) + fp / 2;
        const int64_t cy = tile_of(loc[UNIT_COL_Y_Q]) + fp / 2;
        if (fp < 1 || cx > CENTRE_MAX || cx < -CENTRE_MAX || cy > CENTRE_MAX ||
            cy < -CENTRE_MAX || !fits_i32(aim))
            bad = true;
        MarchMap m;
        m.store = store;
        m.n_units = n_units;
        m.tab = tab;
        m.solid = solid + (size_t)e * h * w;
        m.material = material + (size_t)e * h * w;
        m.h = h;
        m.w = w;
        m.key = unit_draws_dev::draw_key(keys, e, &bad);
        m.tick = tick;
        bool shot = false;
        if (bad) {
        } else if (arch == UNIT_ARCH_SPRAY) {
            if (loc[UNIT_COL_SPRAY_TICKS_LEFT] <= 0) {   // a running burst owns the trigger
                // start_spray_burst_directional
                const int64_t rq = tab[UNIT_TAB_WEAPON_RANGE_TILES + wr] << 16;
                loc[UNIT_COL_SPRAY_TICKS_LEFT] = tab[UNIT_TAB_WEAPON_BURST_TICKS + wr];
                loc[UNIT_COL_SPRAY_AIM_ANGLE] = 1;
                loc[UNIT_COL_SPRAY_AIM_ANGLE + 1] = aim;
                loc[UNIT_COL_SPRAY_TARGET] = 1;
                loc[UNIT_COL_SPRAY_TARGET + 1] =
                    (cx << 16) + fixedpoint::mul_q16_round_q(fixedpoint::cos_q16((q16)aim), rq);
                loc[UNIT_COL_SPRAY_TARGET + 2] =
                    (cy << 16) + fixedpoint::mul_q16_round_q(fixedpoint::sin_q16((q16)aim), rq);
                emit_noise(noise, cy, cx, tab[UNIT_TAB_WEAPON_LOUDNESS_TILES + wr]);
                shot = true;
            }
        } else if (arch == UNIT_ARCH_PROJECTILE) {
            emit_noise(noise, cy, cx, tab[UNIT_TAB_WEAPON_LOUDNESS_TILES + wr]);
            fire_burst(m, r, loc, wr, cone_class, aim, cx, cy, fx, rounds, &bad);
            shot = true;
        } else if (arch == UNIT_ARCH_HITSCAN) {
            refused = true;                      // the beam: not built (P7a.2)
        }
        if (shot && !bad) {
            if (full > 0) {                      // mag_spend
                loc[UNIT_COL_CURRENT_MAG] -= 1;
                if (loc[UNIT_COL_CURRENT_MAG] <= 0) reload_begin(loc, tab, wr, tick, &bad);
            }
            loc[UNIT_COL_LAST_FIRE_TICK] = tick;
        }
    }
    if (bad || refused) {            // the row and its records dropped
        *fx.count = 0;
        *rounds.count = 0;
        *noise.count = 0;
        if (bad) count_fault(cnt);
        else count_refused(cnt);
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

// Slot 2': one thread per env walks its table of rounds in order.
__global__ void k_rounds(int64_t* store, int n_units, const int64_t* tables,
                         const int64_t* ticks, const int64_t* keys, const bool* solid,
                         const int8_t* material, int h, int w, RecordListView fxv,
                         RecordListView rv, unsigned long long* cnt) {
    if (threadIdx.x != 0) return;
    const int e = benv();
    const int64_t tick = ticks[e];
    bool bad = tick < -TICK_MAX || tick > TICK_MAX;
    MarchMap m;
    m.store = store;
    m.n_units = n_units;
    m.tab = env_tables(tables);
    m.solid = solid + (size_t)e * h * w;
    m.material = material + (size_t)e * h * w;
    m.h = h;
    m.w = w;
    m.key = unit_draws_dev::draw_key(keys, e, &bad);
    m.tick = tick;
    RecordSink fx = list_env_sink(fxv);
    int64_t* rows = rv.recs + (size_t)e * rv.cap * rv.width;
    const int64_t n = rv.count[e];
    for (int64_t i = 0; i < n && !bad; ++i) {
        int64_t* rd = rows + i * rv.width;
        march(m, rd, fx, &bad);
    }
    if (bad) count_fault(cnt);
}

void check_lists(const UnitStoreDevice& store, FieldApplier& effects, ObjectTable* rounds,
                 ObjectTable* noise, const char* who) {
    UnitRecordList& L = effects.list();
    std::ostringstream os;
    if (L.n_env() != store.n_env() || L.width() != APPLIER_F)
        os << who << ": the effect list holds " << L.n_env() << " envs, width " << L.width()
           << "; the store " << store.n_env() << " envs, the layout width " << APPLIER_F;
    else if (rounds && (rounds->n_env() != store.n_env() || rounds->width() != OBJ_ROUND_W))
        os << who << ": the round table holds " << rounds->n_env() << " envs, width "
           << rounds->width() << "; the layout width " << OBJ_ROUND_W;
    else if (noise && (noise->n_env() != store.n_env() || noise->width() != OBJ_NOISE_W))
        os << who << ": the noise list holds " << noise->n_env() << " envs, width "
           << noise->width() << "; the layout width " << OBJ_NOISE_W;
    const std::string msg = os.str();
    if (!msg.empty()) throw std::invalid_argument(msg);
}

void launch_check(const char* who) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in " << who << ": " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
}

}  // namespace

void unit_shooting(UnitStoreDevice& store, const UnitTablesDevice& tables, FieldApplier& effects,
                   ObjectTable& rounds, ObjectTable& noise, const int64_t* ticks,
                   const int64_t* keys, const ShootPlanes& p, const UnitShootCounts& out) {
    unit_check_pair(store, tables, "unit_shooting");
    check_lists(store, effects, &rounds, &noise, "unit_shooting");
    UnitRecordList& L = effects.list();
    if (L.n_units() != store.n_units() || rounds.n_units() != store.n_units() ||
        noise.n_units() != store.n_units())
        throw std::invalid_argument("unit_shooting: a list's staging rows are not the store's");
    if (ticks == nullptr || keys == nullptr || p.solid == 0 || p.material == 0)
        throw std::invalid_argument("unit_shooting: ticks (N,), keys (N, 3), solid, material");
    const int n = store.n_env(), U = store.n_units();
    const int64_t* d_ticks = store.stage_input(0, ticks, (size_t)n);
    const int64_t* d_keys = store.stage_input(1, keys, (size_t)n * unit_draws::DRAW_KEY_W);
    store.begin_counts();
    k_shooting<<<bgrid(U, n), 128, 0, tstream()>>>(
        store.rows(), U, tables.values(), d_ticks, d_keys, (const bool*)p.solid,
        (const int8_t*)p.material, p.h, p.w, list_view(L), list_view(rounds), list_view(noise),
        store.counts_dev());
    L.gather();
    rounds.gather();
    noise.gather();
    launch_check("unit_shooting");
    store.end_counts(out.coverage, out.faults, out.refused);
    if (out.dropped) L.download(nullptr, nullptr, out.dropped);
    if (out.rounds_lost) rounds.download(nullptr, nullptr, out.rounds_lost);
    if (out.noise_lost) noise.download(nullptr, nullptr, out.noise_lost);
}

void unit_rounds(UnitStoreDevice& store, const UnitTablesDevice& tables, FieldApplier& effects,
                 ObjectTable& rounds, const int64_t* ticks, const int64_t* keys,
                 const ShootPlanes& p, bool compact, const UnitShootCounts& out) {
    unit_check_pair(store, tables, "unit_rounds");
    check_lists(store, effects, &rounds, nullptr, "unit_rounds");
    if (ticks == nullptr || keys == nullptr || p.solid == 0 || p.material == 0)
        throw std::invalid_argument("unit_rounds: ticks (N,), keys (N, 3), solid, material");
    UnitRecordList& L = effects.list();
    const int n = store.n_env(), U = store.n_units();
    const int64_t* d_ticks = store.stage_input(0, ticks, (size_t)n);
    const int64_t* d_keys = store.stage_input(1, keys, (size_t)n * unit_draws::DRAW_KEY_W);
    store.begin_counts();
    k_rounds<<<dim3(1, n), 32, 0, tstream()>>>(
        store.rows(), U, tables.values(), d_ticks, d_keys, (const bool*)p.solid,
        (const int8_t*)p.material, p.h, p.w, list_view(L), list_view(rounds),
        store.counts_dev());
    launch_check("unit_rounds");
    if (compact) rounds.compact();
    store.end_counts(out.coverage, out.faults, out.refused);
    if (out.dropped) L.download(nullptr, nullptr, out.dropped);
    if (out.rounds_lost) rounds.download(nullptr, nullptr, out.rounds_lost);
}

}  // namespace breach_cuda
