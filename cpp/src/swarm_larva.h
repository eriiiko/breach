#pragma once
// ============================================================================
// swarm_larva.h -- THE LARVA LAW (arc #63 P3, docs/swarm_P3_impl.md §3.5,
// §3.6; engine/17 §4).
// ============================================================================
//
// ONE FP_HD per-slot step, larva_step_slot, run by three loops: the CPU twin
// (swarm_larva.cpp, host_loop), the per-call CUDA kernel and the resident
// CUDA stage (swarm_larva.cu over swarm_launch.cuh). Bit identity is by
// construction -- all integer, own-slot writes, read-only fields, no atomics
// in the law -- and the 3-part gate (tests/cuda_swarm_check.py) proves the
// loops, the launch geometry and the transfers.
//
// What the law does per tick (temperature only, Erik's rulings 1-11):
// head-sweep thermotaxis on the RAW own-tile temperature (here + two points
// one tile ahead at +-sweep_angle; comfort |T - T_prefer|; the tumble rate
// raised to the alarm rate when BOTH open sweep points are worse; a tumble
// turns toward the better side with probability turn_bias), straight runs,
// the wall probe and its turn, chill coma below T_ctmin with waking above
// T_coma_exit (coma: no sensing, no draws, no move), and an instant heat
// death above T_lethal_hot on the POST-move tile in every state. Every value
// it reads is a table row (cpp/src/swarm_tables_layout.h, GENERATED); no law
// constant is a kernel argument, and SWEEP_DIST_Q16 ("one tile ahead",
// ruling 2) is the law's own.
//
// THE DRAW TABLE (counter = (tick, draw_index, salt, unit_id), key = (seed
// lo, seed hi); every block below is draw_index 0 of its salt; a later need
// for more words of the same purpose takes draw_index 1, never a reorder):
//
//   salt                         when                          word  use
//   Salt.SWARM_TUMBLE ("TUMB")   every RUN larva reaching L5   W[0]  uniform_q16 < p decides the tumble
//                                                              W[1]  turn side (turn_heading)
//                                                              W[2]  turn magnitude (philox32_below)
//                                                              W[3]  reserved, unused
//   Salt.SWARM_WALL_TURN ("WALT") only when the probe is blocked W[0] turn side
//                                                              W[1]  turn magnitude
//                                                              W[2..3] reserved, unused
//
// Integers come only through philox32_uniform_q16 and philox32_below --
// never `%`, never a non-integer mapping (chapter 14, door 4).
//
// Credit: Luo L, Gershow M, Rosenzweig M, Kang K, Fang-Yen C, Garrity PA,
// Samuel ADT, "Navigational decision making in Drosophila thermotaxis",
// J Neurosci 2010;30(12):4261-72, doi:10.1523/JNEUROSCI.4090-09.2010 (run
// length and turn direction regulated; the two-point comparison is a
// discrete stand-in for one head sweep). Klein M et al., "Sensory
// determinants of behavioral dynamics in Drosophila thermotaxis", PNAS
// 2015;112(2):E220-9, doi:10.1073/pnas.1416212112 (the graded response this
// law omits: ACCEPTED GAP, sign-only). MacMillan HA, Sinclair BJ,
// "Mechanisms underlying insect chill-coma", J Insect Physiol
// 2011;57(1):12-20, doi:10.1016/j.jinsphys.2010.10.004 (CTmin, reversible
// chill coma). Monzon MA, Weidner LM, Rusch TW, Nehrozoglu S, Hamilton G,
// "High Temperature Limits of Survival and Oviposition of Phormia regina and
// Lucilia sericata", Insects 2022;13(11):991, doi:10.3390/insects13110991
// (24 h exposures; the instant 50 C kill is Erik's game compression of an
// exposure-time law, above the 24 h total-mortality point).
#include <cstdint>

#include "swarm.h"
#include "swarm_tables_layout.h"

namespace swarm {

// "One tile ahead" (ruling 2): a sampling distance on the field grid, the
// nearest distinct cell ahead -- not a body length (ruling 12).
constexpr int32_t SWEEP_DIST_Q16 = fixedpoint::FP_ONE;

// THE BRANCH TALLY (v4.7 G-d): every branch of the law, counted by the CPU
// loop only (the device passes no tally). Names: SWARM_LARVA_BRANCH_NAMES
// (swarm_larva.cpp), exported as bp.SWARM_LARVA_BRANCHES.
enum : int {
    LARVA_BR_SLOT_INVALID = 0,
    LARVA_BR_COMA_HELD,
    LARVA_BR_COMA_TO_RUN,
    LARVA_BR_RUN_TO_COMA,
    LARVA_BR_ALARM_ON,
    LARVA_BR_ALARM_OFF,
    LARVA_BR_PREF_PLUS,
    LARVA_BR_PREF_MINUS,
    LARVA_BR_PREF_NONE_EQUAL,
    LARVA_BR_PREF_NONE_BLOCKED,
    LARVA_BR_SWEEP_SOLID,
    LARVA_BR_SWEEP_OFF_GRID,
    LARVA_BR_TUMBLE_TAKEN,
    LARVA_BR_TUMBLE_NOT_TAKEN,
    LARVA_BR_TURN_BY_COIN,
    LARVA_BR_TURN_TOWARD,
    LARVA_BR_TURN_AWAY,
    LARVA_BR_PROBE_SOLID,
    LARVA_BR_PROBE_EDGE,
    LARVA_BR_MOVED,
    LARVA_BR_DIED,
    LARVA_BR_SURVIVED,
    LARVA_BR_WRAP_NEG_REMAINDER,
    LARVA_BR_COUNT
};
extern const char* const SWARM_LARVA_BRANCH_NAMES[LARVA_BR_COUNT];
extern uint64_t g_larva_branch_tally[LARVA_BR_COUNT];

// The law's values for one env, narrowed from that env's table rows through
// the GENERATED offsets (never a reinterpret_cast, never a hand offset).
struct SwarmLarvaParams {
    int32_t T_prefer_q, T_lethal_hot_q, T_ctmin_q, T_coma_exit_q;
    int32_t turn_bias_q, turn_min_q, turn_max_q, sweep_angle_q;
    int32_t step_q, p_base_q, p_alarm_q;
    uint32_t salt_tumble, salt_wall_turn;
};

FP_HD inline SwarmLarvaParams larva_params_from_table(const int64_t* tab) {
    using namespace swarm_tables;
    SwarmLarvaParams P;
    P.T_prefer_q     = (int32_t)tab[SWARM_LARVA_T_PREFER_Q];
    P.T_lethal_hot_q = (int32_t)tab[SWARM_LARVA_T_LETHAL_HOT_Q];
    P.T_ctmin_q      = (int32_t)tab[SWARM_LARVA_T_CTMIN_Q];
    P.T_coma_exit_q  = (int32_t)tab[SWARM_LARVA_T_COMA_EXIT_Q];
    P.turn_bias_q    = (int32_t)tab[SWARM_LARVA_TURN_BIAS_Q];
    P.turn_min_q     = (int32_t)tab[SWARM_LARVA_TURN_MIN_Q];
    P.turn_max_q     = (int32_t)tab[SWARM_LARVA_TURN_MAX_Q];
    P.sweep_angle_q  = (int32_t)tab[SWARM_LARVA_SWEEP_ANGLE_Q];
    P.step_q         = (int32_t)tab[SWARM_LARVA_STEP_Q];
    P.p_base_q       = (int32_t)tab[SWARM_LARVA_P_BASE_Q];
    P.p_alarm_q      = (int32_t)tab[SWARM_LARVA_P_ALARM_Q];
    P.salt_tumble    = (uint32_t)tab[SWARM_LARVA_SALT_TUMBLE];
    P.salt_wall_turn = (uint32_t)tab[SWARM_LARVA_SALT_WALL_TURN];
    return P;
}

#define LARVA_BR(k) do { if (tally) ++tally[(k)]; } while (0)

// THE PER-SLOT LAW (env e, slot s < hw[e]); returns 1 iff the slot died.
// Kernel contract (the gate's inputs lie inside it): a valid slot stands in
// the grid with a canonical heading and ai_state in {RUN, EAT, COMA}; an
// invalid slot is the all-zero form; the table rows lie in their proved
// ranges (swarm_tables.ROWS, the species rules R1-R4). Labels L0-L8 are
// docs/swarm_P3_impl.md §3.5's.
FP_HD inline int larva_step_slot(const SlotIO& io, int e, int s, uint64_t* tally) {
    const int k = e * io.max_units + s;
    int32_t* death = io.death + (int64_t)k * DEATH_WORDS;
    // L0: an empty slot writes its own (zero) record every launch.
    if (io.st.valid[k] == 0) {
        LARVA_BR(LARVA_BR_SLOT_INVALID);
        death[0] = death[1] = death[2] = death[3] = 0;
        return 0;
    }
    const int h_ = io.h, w_ = io.w;
    const int64_t plane = (int64_t)e * h_ * w_;
    const TileReader R{io.T + plane, w_};
    const bool* solid = io.solid + plane;
    const uint32_t* L = io.launch + (int64_t)e * LAUNCH_FIXED_WORDS;
    const uint32_t tick = L[LAUNCH_TICK], lo = L[LAUNCH_SEED_LO], hi = L[LAUNCH_SEED_HI];
    const SwarmLarvaParams P = larva_params_from_table(io.tables + (int64_t)e * io.table_width);

    // L1
    int32_t px = io.st.pos_x[k], py = io.st.pos_y[k];
    const int32_t h = io.st.heading[k];
    const int32_t st = io.st.ai_state[k];
    const int32_t uid = io.st.unit_id[k];
    int32_t h_new = h, st_new = st;
    bool moved = false;
    // L2
    const int32_t T_here = R(px, py);
    // L3
    if (st == AI_COMA) {
        if (T_here > P.T_coma_exit_q) {            // D8: wake, move next tick
            st_new = AI_RUN;
            LARVA_BR(LARVA_BR_COMA_TO_RUN);
        } else {
            st_new = AI_COMA;
            LARVA_BR(LARVA_BR_COMA_HELD);
        }
    } else if (T_here < P.T_ctmin_q) {             // RUN law (also AI_EAT, D9)
        st_new = AI_COMA;                          // no sensing, no draws, no move
        LARVA_BR(LARVA_BR_RUN_TO_COMA);
    } else {
        st_new = AI_RUN;
        // L4: the head sweep
        const SweepSample sw = head_sweep(R, solid, h_, w_, px, py, h, P.sweep_angle_q,
                                          SWEEP_DIST_Q16);
        if (!sw.open_plus) LARVA_BR(sw.in_plus ? LARVA_BR_SWEEP_SOLID : LARVA_BR_SWEEP_OFF_GRID);
        if (!sw.open_minus) LARVA_BR(sw.in_minus ? LARVA_BR_SWEEP_SOLID : LARVA_BR_SWEEP_OFF_GRID);
        const int64_t d_here = abs64((int64_t)sw.here - P.T_prefer_q);
        const int64_t d_plus = abs64((int64_t)sw.val_plus - P.T_prefer_q);
        const int64_t d_minus = abs64((int64_t)sw.val_minus - P.T_prefer_q);
        const bool alarm = sw.open_plus && sw.open_minus && d_plus > d_here && d_minus > d_here;
        LARVA_BR(alarm ? LARVA_BR_ALARM_ON : LARVA_BR_ALARM_OFF);
        const int32_t pref = side_preference(sw.open_plus, d_plus, sw.open_minus, d_minus);
        if (pref == SIDE_PLUS) LARVA_BR(LARVA_BR_PREF_PLUS);
        else if (pref == SIDE_MINUS) LARVA_BR(LARVA_BR_PREF_MINUS);
        else LARVA_BR((sw.open_plus && sw.open_minus) ? LARVA_BR_PREF_NONE_EQUAL
                                                      : LARVA_BR_PREF_NONE_BLOCKED);
        // L5: the tumble block (Salt.SWARM_TUMBLE, draw_index 0)
        uint32_t Wt[4];
        philox32_draw(lo, hi, (uint32_t)uid, tick, 0u, P.salt_tumble, Wt);
        const int32_t p = alarm ? P.p_alarm_q : P.p_base_q;     // in [0, 65536] by R2
        int how = TURN_BY_COIN;
        bool wneg = false;
        if (philox32_uniform_q16(Wt[0]) < (uint32_t)p) {
            LARVA_BR(LARVA_BR_TUMBLE_TAKEN);
            h_new = turn_heading(h, pref, Wt[1], Wt[2], P.turn_bias_q, P.turn_min_q,
                                 P.turn_max_q, &how, &wneg);
            LARVA_BR(how == TURN_BY_COIN ? LARVA_BR_TURN_BY_COIN
                     : how == TURN_TOWARD ? LARVA_BR_TURN_TOWARD : LARVA_BR_TURN_AWAY);
            if (wneg) LARVA_BR(LARVA_BR_WRAP_NEG_REMAINDER);
        } else {
            LARVA_BR(LARVA_BR_TUMBLE_NOT_TAKEN);
        }
        // L6: the probe along the OLD heading (D7)
        const Probe pr = probe_move(solid, h_, w_, px, py, h, P.step_q);
        if (pr.blocked) {
            LARVA_BR(pr.in ? LARVA_BR_PROBE_SOLID : LARVA_BR_PROBE_EDGE);
            uint32_t Ww[4];
            philox32_draw(lo, hi, (uint32_t)uid, tick, 0u, P.salt_wall_turn, Ww);
            h_new = turn_heading(h, pref, Ww[0], Ww[1], P.turn_bias_q, P.turn_min_q,
                                 P.turn_max_q, &how, &wneg);   // D6: overrides the tumble
            LARVA_BR(how == TURN_BY_COIN ? LARVA_BR_TURN_BY_COIN
                     : how == TURN_TOWARD ? LARVA_BR_TURN_TOWARD : LARVA_BR_TURN_AWAY);
            if (wneg) LARVA_BR(LARVA_BR_WRAP_NEG_REMAINDER);
        } else {
            px = pr.nx;
            py = pr.ny;
            moved = true;
            LARVA_BR(LARVA_BR_MOVED);
        }
    }
    // L7: the kill reads the POST-move tile, in every state (D5).
    if (R(px, py) > P.T_lethal_hot_q) {
        LARVA_BR(LARVA_BR_DIED);
        death[DEATH_UNIT_ID] = uid;
        death[DEATH_POS_X] = px;
        death[DEATH_POS_Y] = py;
        death[DEATH_HEADING] = h;                  // the heading it moved along
        io.st.pos_x[k] = 0;                        // the P2 empty-slot form
        io.st.pos_y[k] = 0;
        io.st.heading[k] = 0;
        io.st.hp[k] = 0;
        io.st.ai_state[k] = AI_DEAD;
        io.st.dist_walked[k] = 0u;
        io.st.unit_id[k] = 0;
        io.st.faction[k] = 0;
        io.st.valid[k] = 0;
        return 1;
    }
    LARVA_BR(LARVA_BR_SURVIVED);
    death[0] = death[1] = death[2] = death[3] = 0;
    // L8
    io.st.pos_x[k] = px;
    io.st.pos_y[k] = py;
    if (moved) io.st.dist_walked[k] = (uint32_t)(io.st.dist_walked[k] + (uint32_t)P.step_q);
    io.st.heading[k] = h_new;                      // a stored value or a wrap() output
    io.st.ai_state[k] = st_new;
    return 0;
}

#undef LARVA_BR

struct LarvaLaw {
    FP_HD int step_slot(const SlotIO& io, int e, int s) const {
        return larva_step_slot(io, e, s, io.tally);
    }
};

// The entries (out of line).
int swarm_larva_cpu(const LawIO& L);                    // swarm_larva.cpp
#ifdef BREACH_HAS_CUDA
int swarm_larva_cuda_percall(const LawIO& L);           // swarm_larva.cu
void swarm_larva_cuda_resident(const ResidentLaunch& R);  // swarm_larva.cu
#endif

}  // namespace swarm
