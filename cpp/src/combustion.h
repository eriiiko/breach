#pragma once
// CombustionSolver — burns fuel against REAL local O2 (EOS refactor P4,
// docs/eos_refactor_design.md §5, decisions log #12; v2.5 P5.1 stoichiometric
// fuel consumption, decisions log #17).
//
// S. Feldman, J.F. O'Brien, O. Arikan, "Animating Suspended Particle
// Explosions", SIGGRAPH 2003 — the heat + product-yield + ignition-threshold
// SOURCE-TERM STRUCTURE this pass follows (constants below are game-tuned,
// not lit-derived: no realistic combustion kinetics, per design §1's
// explicit non-goal).
//
// Runs ONCE per tick, AFTER the EOS solver materializes P (design §3.2 "step
// 6: combustion pass ... reads settled P/N/T, feeds next tick"): its N/T
// mutations are read by NEXT tick's p* = C*N_total*T, never re-entering this
// tick's Helmholtz solve (the solve already ran in step 3, on the
// PRE-combustion state).
//
// A flammable tile is itself SOLID (wood/door — FireSimulation's own
// convention: fire only ever lives on flammable WALLS) and therefore holds
// no gas of its own (bulk_transport.cpp: a solid cell always holds N == 0).
// So, exactly like FireSimulation's own O2 gate, combustion burns in the
// tile's OPEN 4-neighbours' O2/N2/black_smoke — each open neighbour is an
// independent burn site (the flame front sitting in the air pocket next to
// the fuel).
//
// v2.6 (EOS P6.9 — docs/eos_p6_9_combustion_design.md, blessed by Erik
// 2026-07-11): the pass is REFORMULATED from the old row-major SCATTER into
// TWO order-free GATHER passes so it is DIRECTION-FREE and bit-identical
// CPU<->GPU (P6.9b ports this exact algorithm to CUDA, closing P6). The
// reformulation carries FOUR blessed behavioral deltas (design §5) — see
// combustion.cpp's header for alpha/beta/gamma/delta and the golden-rebaseline
// rationale. Structure:
//
//   snapshot Tsnap = copy(temperature)   (freezes the ignition gate: a source
//                                          cannot heat AND ignite a neighbour
//                                          in the same tick — delta alpha)
//
//   Pass A — for each OPEN-air cell j (single writer of O2/SOOT/N2/T[j]):
//     gather its <=4 flammable claimant sources i (claim iff flammable[i],
//       wall_hp[i] > FUEL_FLOOR, ign[i] > 0, Tsnap[i] >= ign[i], and pass-entry
//       O2[j] > o2_thresh_burn); demand_i = burn_rate*dt (uniform).
//     D = sum(demand_i). If D <= O2[j]: alloc_i = demand_i (no contention).
//       Else EXACT-INTEGER proportional split (plain int64 /,% — NOT float, NOT
//       reciprocal_q16; conservation-exact), leftover LSBs to largest-key
//       claimants, ties -> lowest source index, and O2[j] fully DRAINS
//       (delta gamma).
//     burn_j = sum(alloc_i);  O2[j] -= burn_j;  SOOT[j] += round(burn_j*soot_
//       yield);  N2[j] += burn_j - soot  (N_total EXACTLY conserved — #12);
//       ONE aggregate deposit T[j] += burn_j*H_fuel/(c_v*max(N_total[j],
//       n_floor_heat)) against the POST-burn N_total (delta delta), T_MAX_PHYS
//       clamp + PER-CELL counter. Each alloc_i is filed on a per-face buffer.
//
//   Pass B — for each flammable source i (single writer of wall_hp[i]): sum the
//     <=4 incoming face allocations burn_i, pay wall_hp[i] -= round(fuel_per_o2
//     * burn_i), floored ONCE at FUEL_FLOOR (total-then-floor-once).
//
// The heat-deposit reciprocal uses the SAME c_v / n_floor_heat dial as
// TemperatureSolver's Pass-1 radiative deposit (design §4.3), so there is
// exactly ONE "combustion/deposit floor" in the system.
//
// v2.5 (P5.1 stoichiometric fuel consumption — docs/eos_refactor_design.md
// §5 v2.5 amendment, decisions log #17): wall_hp is MUTABLE — the SOURCE tile
// pays fuel_cost = narrow_round(mul_wide(fuel_per_o2_q, burn_i)) (round-to-
// nearest, the same unbiased-sink idiom fire_simulation.cpp's wall-damage
// depletion uses), floored at 1 Q16.16 LSB. P6.9 moves this from a per-
// neighbour floor to a total-then-floor-once in Pass B (design §3, critique
// B): both engage the floor iff the total does, so the "smolder never
// destroys" 1-LSB invariant is preserved, and they differ only by <=3 LSB
// away from the floor (inside the golden re-baseline). This is the EMBER-scale
// consumption that closes v2.4's fuel-free-smolder flag; FireSimulation's
// wall_damage pass remains the FLAME-scale (I>0) consumption. THE 1-LSB RULE
// (Erik, 2026-07-11): this pass NEVER destroys a tile and NEVER emits
// destroyed-tile events — structural destruction stays exclusively
// FireSimulation's I>0 path. A long-smoldered wall survives as charred tissue
// paper at exactly 1 LSB: easy prey for almost any other damage source (and
// for a real flame, whose damage pass CAN take it to 0). The ember state
// itself is EMERGENT (fire I == 0, T >= ignition_temp, wall_hp > FUEL_FLOOR)
// — no new state.
//
// `fire` is now UNUSED (kept in the signature for ABI stability): the old
// scatter read it only as an outcome-neutral row-major prefilter (the real
// gate is ign > 0 && Tsnap >= ign), so the gather drops it with no behavioral
// effect.
//
// o2_thresh_breathe is a SEPARATE constant, defined but NOT consumed here —
// unit suffocation is a LATER mechanics arc (design §5: "enabled here,
// wired later" — a deliberate non-goal boundary, not an oversight).
//
// GPU: still CPU-only after P6.9a (this patch). P6.9b adds cuda_combustion.cu
// mirroring the two gathers + face buffers + barrier chain, proves bit-
// identity vs this CPU reference, and unpins "combustion" from
// EOS_P6_PENDING_KERNELS — closing the P6 arc.

#include <cstdint>

class CombustionSolver {
public:
    // --- config dials (design §9; sane defaults, feel-tuned at P5) --------
    // (Empirically checked at patch time — docs/eos_refactor_design.md §5's
    // gate scenarios — against a small sealed room: these values self-starve
    // a fire within ~1-2 game-seconds and keep the transient temperature/
    // pressure spike bounded and Q16.16-safe; the room-scale "how hot does a
    // shoebox flashover get" question is explicitly a P5 feel call, not a
    // correctness one.)
    float burn_rate       = 1.0f;    // N_O2 consumed per second, per burn site
    float o2_thresh_burn  = 0.03f;   // min local N_O2 to sustain combustion
                                      // (below the fire logistic's P_min so
                                      // the VISIBLE flame dies first)
    float H_fuel           = 4.0f;   // heat yield (T-scale) per unit N_O2 burned
    float soot_yield       = 0.3f;   // fraction of consumed O2 -> black_smoke
                                      // (remainder -> inert_N2, decisions #12)
    float fuel_per_o2      = 0.7f;   // v2.5 (P5.1): wall_hp consumed per unit
                                      // N_O2 burned (wood stoichiometry burns
                                      // ~0.7 mass-units of fuel per unit O2).
                                      // THE ember-lifetime dial (design §9):
                                      // smaller -> embers glow for minutes
                                      // awaiting oxygen; larger -> they char
                                      // out fast. Quantized once per step like
                                      // every other per-step scalar.

    // v2.5 (P5.1): the fuel floor, in RAW Q16.16 counts (1 == one LSB).
    // Doubles as (a) the no-fuel gate threshold (wall_hp <= FUEL_FLOOR ->
    // the ember is out) and (b) the clamp this pass's depletion can never
    // cross. Compile-time constant — Erik's 1-LSB rule, not a dial.
    static constexpr int32_t FUEL_FLOOR = 1;

    // Unit-side suffocation mechanics (LATER arc, design §5): the minimum
    // local N_O2 a unit needs to breathe. Defined at the right layer;
    // nothing reads it yet.
    float o2_thresh_breathe = 0.08f;

    // T_MAX_PHYS (v2.4 as-built amendment, PROVISIONAL pending Erik's P5
    // review): the counted physical-maximum T rail — this pass's heat
    // deposit clamps at the ceiling (counter below). One constant shared
    // across EOSSolver/TemperatureSolver/CombustionSolver, wired from
    // [physics.thermal] by physics_runner. Full rationale: eos_solver.h.
    float T_MAX_PHYS = 16000.0f;

    // --- debug telemetry (mirrors eos_solver.h's counter idiom) -----------
    // P6.9: these now count PER-CELL (one aggregate deposit per air cell), not
    // per-source-per-neighbour as the old scatter did — their ABSOLUTE value
    // moved and no test may assert it (design §3).
    mutable int64_t heat_floor_hits = 0;   // n_floor_heat engagements
    mutable int64_t t_max_phys_hits = 0;   // T_MAX_PHYS rail engagements (v2.4)

    // gas                : (n_gases, h, w) Q16.16 density planes, mutated
    // o2_idx/inert_n2_idx/black_smoke_idx : gas ids (simulation/gases.py)
    // temperature        : (h, w) Q16.16, mutated (the heat deposit)
    // wall_hp            : (h, w) Q16.16, MUTATED (v2.5 P5.1: the fuel gate
    //                      AND the ember-scale fuel store — depleted
    //                      fuel_per_o2-proportionally, floored at FUEL_FLOOR,
    //                      never destroyed by this pass)
    // fire               : (h, w) Q16.16, UNUSED since P6.9 (was a candidate
    //                      prefilter; kept for ABI stability — see header note)
    // flammable/solid/is_vacuum : (h, w) bool masks
    // ignition_temp_q16  : (h, w) Q16.16, per-tile material threshold — the
    //                      SAME table apply_temperature_ignition uses
    //                      (simulation/materials.py; 0 == never ignites)
    // c_v, n_floor_heat  : the SAME dials as TemperatureSolver's Pass-1
    //                      radiative deposit (design §4.3)
    void step(
        int32_t* gas, int n_gases,
        int o2_idx, int inert_n2_idx, int black_smoke_idx,
        int32_t* temperature,
        int32_t* wall_hp,
        const int32_t* fire,
        const bool* flammable,
        const bool* solid,
        const bool* is_vacuum,
        const int32_t* ignition_temp_q16,
        int h, int w, float dt,
        float c_v, float n_floor_heat
    ) const;
};
