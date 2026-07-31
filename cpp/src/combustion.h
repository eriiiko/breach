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
// `fire` is READ again since the continuous-O2 law (design §2.3): it is the
// per-claimant intensity factor I_k in demand_k = burn_rate*I_k*o2f_j*dt. (P6.9
// had dropped it as an outcome-neutral prefilter; the proportional-draw law
// reinstates it as the "how hard does this source burn" term.)
//
// o2_thresh_breathe is a SEPARATE constant, defined but NOT consumed here —
// unit suffocation is a LATER mechanics arc (design §5: "enabled here,
// wired later" — a deliberate non-goal boundary, not an oversight).
//
// THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md §2 site
// 3): the header text above says "a flammable tile is itself SOLID ... and
// therefore holds no gas" — that is TRUE of wood/doors and FALSE of FURNITURE,
// which is permeable (0.5, the deliberate "shield but not seal" soft body) and
// therefore an open, gas-holding cell that CAN be a Pass-A burn site for an
// adjacent burning tile. Under the ruling's A3 its pore gas is THIN (N ~ 0.3-0.4
// of ambient), so the gas-divisor deposit dT = burn*H_fuel/(c_v*max(N,n_floor))
// would spike the OBJECT's temperature by ~2.5-3x per unit burn — the wrong
// conversion for an object, and rail-hunting.
//   RULE: on a `thermal_solid` burn site the aggregate deposit converts via the
//   tile's own `heat_inv_shift` (dT = deposit >> log2(thermal_mass)) — the
//   OBJECT path, exactly as TemperatureSolver's MEDIUM-TEST SITE 5/6 converts a
//   ray deposit. SAME energy in, object-appropriate scale; adjacent-crate fire
//   spread keeps working, now honestly.
// `thermal_solid`/`heat_inv_shift` are NULLABLE: either one null means "the
// caller has no thermal mask" and every site takes the gas path — today's
// behaviour byte-for-byte, and identical anyway on any furniture-free map,
// where every open cell has thermal_mass 0 (build addendum D4).
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
    // Continuous O2->combustion law (docs/continuous_o2_law_design_2026-07-24.md):
    // demand is now PROPORTIONAL in both fire intensity I and the O2 factor o2f
    //   demand_i = burn_rate * I_i * o2f_j * dt      (was: burn_rate * dt, gated)
    // where o2f_j is LINEAR in the air cell's O2 MOLE FRACTION X = O2/(O2+N2):
    //   o2f = clamp01((X - o2_frac_ext) / (o2_frac_full - o2_frac_ext))
    // burn_rate drops to the ceiling_h-anchored physical value (~1/50). A choked
    // (low-o2f) or low-intensity fire draws less O2 -> less heat -> "a choked
    // fire is a cool fire". o2_thresh_burn is RETIRED as a gate (below).
    // Huggett, R.C., "Estimation of rate of heat release by means of oxygen
    // consumption measurements", Fire and Materials 4(2):61-65, 1980 (~13.1 MJ/kg
    // O2) anchors the burn_rate/H_fuel scale; Peatross & Beyler 1997 the linear
    // law. Both archived under docs/papers/ (see fire_simulation.cpp header).
    float burn_rate       = 0.02f;   // N_O2 consumed per second per burn site at
                                      //  I=1, o2f=1 (ceiling_h-anchored ~1/50; was
                                      //  1.0 under the retired uniform-gated draw)
    float o2_frac_ext     = 0.13f;   // X_ext: flame-extinction O2 mole fraction
                                      //  (shared law with FireParams; 0 = pure
                                      //  proportional)
    // FULL-RESPONSE REFERENCE SPLIT (2026-07-30) — the exact twin of
    // FireParams::o2_frac_full (the two O2 laws stay bit-identical). The span's
    // upper end used to be o2_frac_amb, which made AMBIENT the ceiling (clamp01)
    // and hid every O2-enrichment route by construction. Normalizing by PURE O2
    // makes o2f a true physical fraction; ambient air lands at 0.092.
    float o2_frac_full    = 1.00f;   // X_full: the O2 mole fraction at which o2f
                                      //  reaches 1 (pure O2). NOT the ambient
                                      //  atmosphere, NOT map-overridden.
    float o2_frac_amb     = 0.21f;   // X_amb: what the ambient atmosphere IS (reads
                                      //  the level's authored [ambient] o2_frac;
                                      //  0.21 fallback — one source of truth with
                                      //  BC). NO LONGER read by step(): the law
                                      //  normalizes by o2_frac_full above.
    float o2_thresh_burn  = 0.03f;   // RETIRED as the burn gate (the o2f law is the
                                      //  throttle now); kept ONLY as an epsilon
                                      //  skip-floor — an air cell with O2 <= this
                                      //  is treated as fully starved and skipped
                                      //  (a cheap early-out, no behavioral gate)
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

    // ---- P-R4: H_bed — the FUEL-BED deposit ------------------------------
    // (docs/radiation_raycaster_extinction_ruling_2026-07-31.md A1, "Where the
    // burning tile's own temperature now comes from".)
    //
    // With the painter retired, a lone crate's radiation nets to ZERO at the
    // source and only LOSES to cooler surroundings — so combustion must own the
    // flame plateau. `H_fuel` above cannot: it is the GAS-side yield, deposited
    // into the air cell where the flame front sits, and at the blessed
    // operating point it is ~4 heat-counts/tick against the painter's ~19,000.
    // The missing term is real physics and is NOT self-radiation: a flame heats
    // its own FUEL BED — that is how fires sustain. Pass A already computes
    // every claimant's demand share, so each claimant k gets
    //
    //     heat[src_k] += (mul_q16(burn_k, H_BED_M) << H_BED_SHIFT)
    //
    // a POSITIVE, order-free add into the EXISTING `heat[]` plane (which keeps
    // its positive-saturating contract — with the painter gone, combustion and
    // weapons/payloads are its only writers; see A5's census).
    //
    // ONE LOGICAL CONSTANT, SPLIT: H_bed = H_BED_M · 2^H_BED_SHIFT. The needed
    // magnitude (order 10^5 T-counts per unit N_O2) does not fit a Q16.16
    // mantissa, and the split also protects PRECISION at the other end: a
    // claimant's per-tick burn is only ~1-4 raw Q16.16 counts, so mul_q16's
    // truncation is relatively coarse unless the mantissa carries most of the
    // magnitude. Keep H_BED_M as large as the format allows (|H_BED_M| < 32768)
    // and take the rest in the shift.
    //
    // WHAT IT IS, HONESTLY: a CALIBRATED LUMPED CONSTANT, exactly like
    // `thermal_mass`. It is Huggett-SHAPED (strictly proportional to the O2
    // actually consumed, so a choked fire deposits nothing and the plateau sags
    // with local O2 — backdraft-adjacent feel, by design) but it is NOT
    // Huggett-VALUED: `thermal_mass = 8` already lumps the ~130x surface-layer
    // factor (seed §1.4), so no J/mol anchor survives the conversion. Do not
    // read it as an enthalpy.
    float H_BED_M     = 25290.0f;   // mantissa (real units), quantized per step
    int   H_BED_SHIFT = 3;          // H_bed = H_BED_M * 2^H_BED_SHIFT = 2.023e5

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
    // fire               : (h, w) Q16.16, READ (continuous-O2 law §2.3): the
    //                      per-claimant intensity factor I_k in the O2 demand
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
        float c_v, float n_floor_heat,
        // THERMAL-MASS AXIS, P-EOS (see the header block): the per-medium
        // THERMAL mask (`thermal_mass > 0`, GameMap.thermal_solid) + the per-tile
        // convert shift (log2(thermal_mass), GameMap.heat_inv_shift). Both
        // nullable; either null -> every burn site takes the GAS deposit path,
        // i.e. the pre-patch behaviour.
        const bool* thermal_solid = nullptr,
        const int32_t* heat_inv_shift = nullptr,
        // P-R4: the `heat[]` plane (Q16.16, h*w), MUTATED — the H_bed fuel-bed
        // deposit's target. Positive-saturating adds only, so it is order-free
        // exactly as the retired ray deposit was. nullptr -> no H_bed (every
        // legacy/direct-binding caller stays byte-identical).
        int32_t* heat = nullptr,
        // ---- D1: THE DEMAND ACCUMULATOR (amendment 5, Erik's ruling) ------
        // (4, h, w) int32, SYNCED sim state (GameMap.dem_acc), MUTATED here.
        //
        // THE PROBLEM IT SOLVES. The demand was
        //     mul_q16(mul_q16(burn_cap_q, I), o2f)
        // — two CHAINED Q16.16 truncations on a quantity whose true value at
        // the blessed operating point is ~1.06 counts. Measured: 0 counts for
        // every I below 0.200, exactly 1 from 0.200 to ~0.40. A STAIRCASE with
        // a DEAD ZONE: a fire born at ignition_seed 0.12 drew no oxygen, so it
        // released no fuel-bed heat, so it cooled below its own `hot` floor and
        // died at 21 s — and even a fire seeded above the knee died as soon as
        // the normal ring-O2 dip dragged I_eq (0.2098, a 4.9% margin) back
        // through it. `H_bed` could not fix this: it multiplies a zero.
        //
        // THE FIX — ERROR FEEDBACK (the classic dithered-accumulator idiom).
        // Keep the WIDE product un-truncated, carry the sub-count remainder in
        // a per-(air-cell, face) plane, and draw whole counts as the debt
        // accrues:
        //     P    = burn_cap_q * I_q * o2f_q            (int64, scale 2^32/count)
        //     wide = acc + (P >> 1)                      (int64, scale 2^31/count)
        //     draw = wide >> 31                          (whole Q16.16 counts)
        //     acc  = wide - (draw << 31)                 ([0, 2^31) -> int32)
        // EXACT IN EXPECTATION and UNBIASED: over any window the counts drawn
        // equal the true demand to within one count, so at the operating point
        // 1 count arrives every ~1.65 ticks instead of never-then-always. The
        // Huggett `burn_rate` anchor is untouched — this changes only HOW the
        // exact product is rendered into integers, not what it is.
        //
        // WHY (4, h, w) AND NOT PER SOURCE TILE: Pass A's thread for air cell j
        // is the SINGLE WRITER of everything at index j, including its four
        // face slots (the existing `alloc_face` idiom, itself the cuda_water
        // dq_e/dq_s precedent). A per-source-tile accumulator would be written
        // by up to four air cells in one pass — atomics, and order-dependent.
        // Keyed identically to `alloc_face`: slot [d*n + j] is the debt air
        // cell j owes toward the claimant in direction D4[d].
        //
        // RESET RULE (documented because a stale debt is a real bug): a slot is
        // ZEROED the moment its neighbour stops being a burning claimant —
        // i.e. the claim gate fails (not flammable / fuel exhausted / material
        // cannot ignite / below its ignition temperature) or `fire[i] <= 0`
        // (flameless). It persists ONLY while that neighbour is actively
        // burning, so a re-ignition never inherits an old fraction. Bounded
        // exception, deliberately not chased: an air cell that early-outs
        // before the claim loop (fully O2-starved, `O2 <= o2_thresh_burn`)
        // keeps its sub-count debt until it has oxygen again — under one count.
        //
        // nullptr -> the pre-D1 chained-truncation demand, so every legacy /
        // direct-binding caller stays byte-identical.
        int32_t* dem_acc = nullptr
    ) const;
};
