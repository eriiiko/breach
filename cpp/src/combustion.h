#pragma once
// CombustionSolver — burns fuel against REAL local O2 (EOS refactor P4,
// docs/eos_refactor_design.md §5, decisions log #12).
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
// So, exactly like FireSimulation's own O2 gate, combustion reads/writes the
// tile's OPEN 4-neighbours' O2/N2/black_smoke — each open neighbour is an
// independent burn site (the flame front sitting in the air pocket next to
// the fuel), processed in fixed row-major order (deterministic; NOT
// associative-symmetric across neighbours sharing a burning wall — same
// idiom as FireSimulation's own per-neighbour smoke-emission deposit).
//
// Per candidate tile i — flammable AND fuelled (wall_hp[i] > 0) AND
// (fire[i] > 0 OR temperature[i] >= ignition_temp_q16[i]) [a cheap row-major
// PREFILTER only; the real gate below is checked unconditionally, so this
// widening never changes the result] AND ignition_temp_q16[i] > 0 (the
// material can ignite at all) AND temperature[i] >= ignition_temp_q16[i]:
//
//   for each OPEN (non-solid, non-vacuum) 4-neighbour j:
//     if N_O2[j] > o2_thresh_burn:
//       burn = min(burn_rate*dt, N_O2[j])              (Q16.16, saturating)
//       N_O2[j]          -= burn
//       soot              = round(burn * soot_yield)
//       N_black_smoke[j] += soot
//       N_inert_N2[j]    += burn - soot                 (N_total EXACTLY
//                                                        conserved — #12)
//       T[j] += burn * H_fuel / (c_v * max(N_total[j], n_floor_heat))
//              (the design §4.3 heat-deposit reciprocal — the SAME c_v /
//               n_floor_heat dial as TemperatureSolver's Pass-1 radiative
//               deposit, so there is exactly ONE "combustion/deposit floor"
//               in the system, per design §4. N_total[j] here is the
//               POST-burn O2[j]+N2[j] sum at the SAME neighbour cell — the
//               same bulk-pair proxy the engine already uses for that
//               deposit, floored INDEPENDENTLY of every other floor.)
//
// wall_hp is READ-ONLY here (the fuel gate) — combustion does NOT deplete
// it; FireSimulation's own wall_damage stays the sole fuel-consumption
// brake, so the two passes never double-spend the same fuel store.
//
// o2_thresh_breathe is a SEPARATE constant, defined but NOT consumed here —
// unit suffocation is a LATER mechanics arc (design §5: "enabled here,
// wired later" — a deliberate non-goal boundary, not an oversight).
//
// GPU: pinned to CPU for this migration window (D7) — no CUDA kernel exists
// yet; the P6 patch enumerates "combustion pass" among its to-port kernels.

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
    mutable int64_t heat_floor_hits = 0;   // n_floor_heat engagements
    mutable int64_t t_max_phys_hits = 0;   // T_MAX_PHYS rail engagements (v2.4)

    // gas                : (n_gases, h, w) Q16.16 density planes, mutated
    // o2_idx/inert_n2_idx/black_smoke_idx : gas ids (simulation/gases.py)
    // temperature        : (h, w) Q16.16, mutated (the heat deposit)
    // wall_hp            : (h, w) Q16.16, READ-ONLY (the fuel gate)
    // fire               : (h, w) Q16.16, READ-ONLY (candidate prefilter)
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
        const int32_t* wall_hp,
        const int32_t* fire,
        const bool* flammable,
        const bool* solid,
        const bool* is_vacuum,
        const int32_t* ignition_temp_q16,
        int h, int w, float dt,
        float c_v, float n_floor_heat
    ) const;
};
