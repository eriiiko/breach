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

// P-O2b — THE EXTENDED OXYGEN DRAW (Erik's Option 2b; design
// docs/fire_realism_design_2026-08-01.md v5.2 "F-O2b", v4 ruling batch item 1).
//
// A burning tile no longer draws oxygen only from its four open faces: it draws
// from every OPEN cell within BFS hop-distance <= DRAW_R, expanded THROUGH OPEN
// CELLS ONLY starting at its own open faces (never through a solid — "a wall
// breathes only via its open faces, extended outward"). This is the
// entrainment stand-in: it raises DELIVERY without inflating room O2
// inventories, so sealed-room smothering stays exactly real.
//
// It is a DETERMINISTIC GENERALIZATION of the shipped per-air-cell demand
// share, and it reduces to that law EXACTLY at DRAW_R == 1 (see W_hop below) —
// the patch's own regression oracle.
//
// -------------------------------------------------------------------------
// THE ENUMERATION (canonical fixed-offset unrolled relaxation; NO queue, NO
// truncation). The pass is a GATHER keyed on the AIR CELL (the single-writer
// discipline that makes it order-free), so the neighbourhood is enumerated in
// REVERSE: air cell j asks "which burning tiles can reach me in <= R hops?".
// The relation is symmetric — every intermediate cell on such a path is open,
// and adjacency is symmetric — so the reverse walk is the same walk.
//
// Two canonical BAKED offset tables do all the work:
//
//   OFF[]  — the SOURCE-slot table: every offset with 1 <= |dy|+|dx| <= R.
//            |OFF| = 2R(R+1) = 4 (R=1), 12 (R=2), 24 (R=3). RING-ORDERED, and
//            **RING 1 IS EXACTLY D4's ORDER (N, S, W, E)** — which is what
//            makes the widened `dem_acc` plane collapse to the shipped
//            (4, h, w) plane, slot for slot, at R = 1.
//   BALL[] — the TRAVERSAL offsets: every offset with |dy|+|dx| <= R-1, the
//            cells a min-hop path of length <= R-1 can pass through (a path of
//            length L only ever visits cells at Manhattan distance <= L, so the
//            ball provably CONTAINS every such path — no truncation).
//            |BALL| = 1 (R=1), 5 (R=2), 13 (R=3). CENTRE FIRST.
//
// The relaxation runs R-1 LEVELLED rounds over BALL: round r assigns the cells
// at BFS distance exactly r, reading only cells assigned at r-1. Within a round
// the combine is a MAX over incoming candidates, so it is order-free; across
// rounds the levelling makes it plain BFS. Each claimant slot then reduces
// lexicographically over (minimum hop distance d, then MAXIMUM path weight) —
// also an order-free reduction.
//
// PATH WEIGHT is PERMEABILITY-MULTIPLICATIVE (the physics_engine/eos_solver
// face idiom): the product of min(perm) over each traversed FACE. A crate
// (perm 0.5) attenuates the draw THROUGH itself (0.5 in, 0.5 out = 0.25), a
// wall (perm 0) blocks, a vacuum cell terminates expansion. The first step —
// burning tile -> its own open face — is NOT attenuated: the expansion starts
// AT the open face and the solid fuel tile itself is never traversed. (This is
// also what makes W_hop[1] * w_path == 1 exactly at R = 1.)
//
// THE DEMAND, per (burning tile i, reachable cell j):
//     dem = burn_cap_q * I_i * o2f_j * hotf_i * W_hop[d(j)] * w_path
// (hotf_i is R3's hot-burns-faster factor, added after this design landed —
// see the R3 block further below for its law) with W_hop a BAKED integer
// table, shipped as quantize(1/(1+d)) NORMALIZED so
// that W_hop[1] == FP_ONE:  W_hop[d] = quantize(2/(1+d)) = 1, 2/3, 1/2. At
// R = 1 every reachable cell has d == 1 and w_path == 1, so the weight is
// exactly FP_ONE and the demand is BIT-IDENTICAL to the shipped law.
//
// The per-air-cell demand-share allocation, the delta-gamma full drain, the
// per-cell O2 floor and the lowest-source-index tiebreak all generalize
// UNCHANGED (they never referred to the number of claimants, only to the set).
//
// DEPOSITS ARE RE-SITED (v5.2's rule, honouring Erik's ruling 4 "air is heated
// at the fire ONLY"): the drawn O2 is DEBITED at the donor cells, but the
// combustion HEAT and the SOOT/N2 products land at the FIRE'S OWN TILE + ITS
// OPEN FACES, never at a distant donor. See `dep_site` in combustion.cpp for
// the exact split (direct-then-remainder) and why it is identity at R = 1.
//
// THE o2f SENSOR STAYS RADIUS-1 — DELIBERATE AND STATED. `o2f_j` is still the
// DONOR cell's own O2 factor, and the fire logistic's separate O2 sensor
// (fire_simulation.cpp) still reads the burning tile's own 4-neighbour ring.
// 2b raises DELIVERY (fuel throughput, HRR); it does NOT widen the local
// sensor, so knee / extinction / smother semantics keep their meaning.
//
// R3 hot-burns-faster (fire session #12, docs/fire_3c_design_2026-09-01.md
// "Ruling R3"): the DEMAND becomes
//     demand_k = burn_cap * I_k * o2f_j * hotf_k * W_hop[d_k] * w_path_k
// hotf_k = clamp((Tsnap[k] - T_ext_k)/fire_T_span, 0, hotf_cap) — the SAME
// ramp fire_simulation.cpp's `hot` uses, read from the SAME per-material
// `fire_T_ext_plane` (nullable, falls back to the scalar `fire_T_ext` below),
// but UNCAPPED AT 1 (ceiling `hotf_cap` instead) since it is a RATE factor
// here, not the sustain gate. T source is Tsnap — the SAME pass-entry
// snapshot the claim gate above reads, preserving "a source can't heat AND
// ignite/accelerate a neighbour's demand the same tick". hotf_k joins the
// PINNED left-fold immediately after o2f_j (folded together via one mul_q16
// into `o2f_hotf` before the wide product forms — see combustion.cpp for the
// overflow bound this keeps). Fuel DRAIN needs no separate term:
// `fuel_cost ∝ O2 drawn` already, so a hotter fire that draws more O2 already
// pays proportionally more fuel — verified, not re-derived, in combustion.cpp.

#include <cstdint>

// ---------------------------------------------------------------------------
// The canonical P-O2b draw tables. ONE definition, shared by combustion.cpp and
// cuda_combustion.cu (which uploads these very arrays to __constant__ memory,
// so the two backends cannot drift).
// ---------------------------------------------------------------------------
namespace combustion_draw {

inline constexpr int R_MAX     = 3;    // largest supported DRAW_R (v5.2's sweep top)
inline constexpr int SLOTS_MAX = 24;   // 2R(R+1) at R = 3
inline constexpr int BALL_MAX  = 13;   // 2(R-1)R + 1 at R = 3

// |{(dy,dx) : 1 <= |dy|+|dx| <= R}| — the claimant-slot count at radius R.
constexpr int slot_count(int R) { return 2 * R * (R + 1); }        // 4, 12, 24
// |{(dy,dx) : |dy|+|dx| <= R-1}| — the traversal-ball size at radius R.
constexpr int ball_count(int R) { return 2 * (R - 1) * R + 1; }    // 1, 5, 13

// D4 — VERBATIM of combustion.cpp's D4 (N, S, W, E) and its opposite-face map.
inline constexpr int8_t D4_DY[4]  = {-1,  1,  0,  0};
inline constexpr int8_t D4_DX[4]  = { 0,  0, -1,  1};
inline constexpr int8_t D4_OPP[4] = { 1,  0,  3,  2};

// SOURCE-SLOT offsets (from the donor air cell TO the burning tile), ring by
// ring. RING 1 (slots 0-3) IS EXACTLY D4's ORDER — the R = 1 identity anchor:
// at R = 1 the widened dem_acc plane IS the shipped (4, h, w) plane, slot for
// slot, so its digest bytes are unchanged. Rings are a PREFIX of one another,
// so slot indices are stable as R grows.
inline constexpr int8_t OFF_DY[SLOTS_MAX] = {
    -1,  1,  0,  0,                                  // ring 1 (== D4)
    -2, -1, -1,  0,  0,  1,  1,  2,                  // ring 2
    -3, -2, -2, -1, -1,  0,  0,  1,  1,  2,  2,  3,  // ring 3
};
inline constexpr int8_t OFF_DX[SLOTS_MAX] = {
     0,  0, -1,  1,                                  // ring 1 (== D4)
     0, -1,  1, -2,  2, -1,  1,  0,                  // ring 2
     0, -1,  1, -2,  2, -3,  3, -2,  2, -1,  1,  0,  // ring 3
};

// TRAVERSAL-BALL offsets (from the donor air cell), CENTRE FIRST then ring by
// ring — again a prefix family, so ball_count(R) entries is the ball for R.
inline constexpr int8_t BALL_DY[BALL_MAX] = {
     0,                                   // ring 0 — the donor cell itself
    -1,  0,  0,  1,                       // ring 1
    -2, -1, -1,  0,  0,  1,  1,  2,       // ring 2
};
inline constexpr int8_t BALL_DX[BALL_MAX] = {
     0,                                   // ring 0
     0, -1,  1,  0,                       // ring 1
     0, -1,  1, -2,  2, -1,  1,  0,       // ring 2
};

// ---- Derived adjacency tables (constexpr-GENERATED from the two tables above,
// never hand-transcribed — a typo in OFF/BALL is the only possible error, and
// the static_asserts below pin the properties the law depends on). ------------
struct Adj { int8_t v[BALL_MAX][4]; };

// BALL_NBR[b][d] = ball index of BALL[b] + D4[d], or -1 if it leaves the ball.
constexpr Adj make_ball_nbr() {
    Adj t{};
    for (int b = 0; b < BALL_MAX; ++b) {
        for (int d = 0; d < 4; ++d) {
            t.v[b][d] = -1;
            const int dy = BALL_DY[b] + D4_DY[d];
            const int dx = BALL_DX[b] + D4_DX[d];
            for (int k = 0; k < BALL_MAX; ++k) {
                if (BALL_DY[k] == dy && BALL_DX[k] == dx) { t.v[b][d] = (int8_t)k; break; }
            }
        }
    }
    return t;
}

// BALL_SLOT[b][d] = source-slot index of BALL[b] + D4[d], or -1 when that
// offset is (0,0) — the donor cell can never be its own claimant (hop 0 is
// excluded from the draw, exactly as the shipped law never let cell j claim
// from itself).
constexpr Adj make_ball_slot() {
    Adj t{};
    for (int b = 0; b < BALL_MAX; ++b) {
        for (int d = 0; d < 4; ++d) {
            t.v[b][d] = -1;
            const int dy = BALL_DY[b] + D4_DY[d];
            const int dx = BALL_DX[b] + D4_DX[d];
            if (dy == 0 && dx == 0) continue;
            for (int k = 0; k < SLOTS_MAX; ++k) {
                if (OFF_DY[k] == dy && OFF_DX[k] == dx) { t.v[b][d] = (int8_t)k; break; }
            }
        }
    }
    return t;
}

inline constexpr Adj BALL_NBR  = make_ball_nbr();
inline constexpr Adj BALL_SLOT = make_ball_slot();

// The properties the R = 1 identity and the enumeration's exactness rest on.
static_assert(slot_count(1) == 4 && slot_count(2) == 12 && slot_count(3) == 24, "slot counts");
static_assert(ball_count(1) == 1 && ball_count(2) == 5 && ball_count(3) == 13, "ball counts");
static_assert(OFF_DY[0] == D4_DY[0] && OFF_DX[0] == D4_DX[0], "slot 0 == D4[0]");
static_assert(OFF_DY[1] == D4_DY[1] && OFF_DX[1] == D4_DX[1], "slot 1 == D4[1]");
static_assert(OFF_DY[2] == D4_DY[2] && OFF_DX[2] == D4_DX[2], "slot 2 == D4[2]");
static_assert(OFF_DY[3] == D4_DY[3] && OFF_DX[3] == D4_DX[3], "slot 3 == D4[3]");
static_assert(BALL_DY[0] == 0 && BALL_DX[0] == 0, "ball centre is index 0");
// At R = 1 the ball is the centre alone and its four steps ARE slots 0..3 in
// D4 order — this static_assert IS the R = 1 byte-identity, proved at compile
// time rather than only measured on the bench.
static_assert(BALL_SLOT.v[0][0] == 0 && BALL_SLOT.v[0][1] == 1 &&
              BALL_SLOT.v[0][2] == 2 && BALL_SLOT.v[0][3] == 3,
              "R=1: the centre's four steps are exactly D4 slots 0..3");

}  // namespace combustion_draw

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

    // --- R3 hot-burns-faster (fire session #12, docs/fire_3c_design_2026-09-
    // 01.md "Ruling R3") — the DEMAND-side twin of FireParams::hotf_cap/
    // fire_T_ext/fire_T_span. See the header block above for the law; see
    // combustion.cpp for the per-claimant computation. fire_T_ext/fire_T_span
    // MIRROR FireParams' own fields of the same name (physics_runner.py binds
    // both solvers from the SAME [physics.fire] keys, one source of truth) —
    // fire_T_ext is the FALLBACK only (the live per-material read is the
    // `fire_T_ext_plane` step() parameter below; a uniform plane holding
    // quantize(fire_T_ext) is byte-identical to passing no plane at all,
    // same nullable-plane idiom as fuel_recip/fire_T_ext_plane elsewhere).
    float fire_T_ext       = 350.0f; // FALLBACK extinction temperature (no plane)
    float fire_T_span      = 180.0f; // width of the hotf ramp above T_ext (GLOBAL)
    // hotf_cap: the SAME dial value as FireParams::hotf_cap (one enrichment/
    // overheat ceiling, bound to both solvers from ONE config key) — the
    // ceiling on hotf, which is NOT capped at 1 the way `hot` is.
    float hotf_cap         = 10.0f;  // NEW (R3): ceiling on the demand-side hotf ramp

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
    // P-E2b (energy-books arc, design §2.2/§2.5): the energy-sum TWIN of
    // heat_floor_hits — `heat_floor_hits` only ever counted the ENGAGEMENT;
    // this sums what the floor engagement DESTROYS. When N_total < floor the
    // divide uses `floor` instead of the real N, delivering only a N/floor
    // fraction of the aggregate deposit; the dropped fraction is
    // deposit*(1 - N/floor), same currency as `deposit`/`heat` (Q16.16,
    // single power). One-way DESTRUCTION (N < floor here, so >= 0 by
    // construction) — accumulates across step() calls, never reset (the
    // heat_floor_hits/t_max_phys_hits idiom of this class).
    mutable int64_t e_deposit_drop_sum = 0;

    // --- arc #54 P-G1b: THE COMBUSTION ENERGY LEDGER (design §2.7 row 3) ---
    //
    // Combustion is the arc's hardest writer seam, because its mass moves TWO
    // HOPS: O2 is debited at DONOR cells (which under an extended draw sit up
    // to `draw_r` tiles away) while the products appear at the FLAME. Under a
    // stored energy field every writer of N is a writer of T, so the energy
    // has to make the same two hops — hence the parallel `e_slot` /
    // `e_dep_site` ledger, keyed exactly like the mass buffers.
    //
    // THE TWO IDENTITIES, exact in int64, gated by tests/_fire_bench.py:
    //
    //   (A) the BOOKS identity — what this pass did to the conserved field:
    //         Δ Σ_accountable gas_energy ==
    //             − e_comb_draw_sum      (parcels leaving accountable donors)
    //             + e_comb_deliver_sum   (products landing on accountable gas)
    //             + e_comb_heat_sum      (the fire heat deposit)
    //             + e_comb_rail_sum      (the deposit-site T_MAX_PHYS rail)
    //
    //   (B) the PARCEL identity — nothing drawn is lost on the way:
    //         e_comb_draw_sum + e_comb_mint_sum ==
    //             e_comb_deliver_sum + e_soot_shed_sum
    //           + e_ts_products_sum  + e_comb_export_sum
    //
    // WHY `e_soot_shed_sum` EXISTS AT ALL (R3-#9). `soot_yield` of the burnt
    // O2 leaves the BULK books as black smoke (a trace plane), so only
    // `burn − soot` of bulk mass arrives at the flame. Delivering the WHOLE
    // parcel's energy anyway would raise the arriving mass's E/N by
    // 1/(1−soot_yield) — and in the R = 1 donor==deposit case that compounds
    // the same cell's temperature every tick with no counterparty, i.e. #54
    // again through a new door. So the parcel is split in the SAME proportion
    // as its mass, exactly, and the soot's share is shed to this counter.
    mutable int64_t e_comb_draw_sum    = 0;  // withdrawn from accountable donors
    mutable int64_t e_comb_mint_sum    = 0;  // parcels born at ambient (ring/ts donor)
    mutable int64_t e_comb_deliver_sum = 0;  // products into accountable gas cells
    mutable int64_t e_soot_shed_sum    = 0;  // the soot's share, out of the bulk books
    mutable int64_t e_ts_products_sum  = 0;  // products landing on a thermal solid
    mutable int64_t e_comb_export_sum  = 0;  // products landing on a ring cell
    mutable int64_t e_comb_heat_sum    = 0;  // the fire heat deposit (into gas E)
    mutable int64_t e_comb_rail_sum    = 0;  // deposit-site T_MAX_PHYS rail (signed)

    // --- arc #54 P-G5 (design gas_energy_thermostat_ledger_2026-08-30.md) --
    // `e_comb_solid_heat_sum` — the OBJECT-SITE heat deposit's landing on a
    // THERMAL SOLID burn cell (furniture/fuel: `object_site` above), priced
    // as the cell's ACTUAL applied ΔT (post its own T_MAX_PHYS rail) × its
    // real capacity. This write bypasses `heat[]`/TemperatureSolver::step
    // entirely (`combustion.cpp` writes `temperature[s]` directly — see the
    // `object_site` branch), so it was invisible to BOTH the gas-side
    // identity (A) above (object sites carry no `gas_energy`) and
    // TemperatureSolver's own `e_solid_deposit_sum` (which only sees its own
    // Pass 1). It is a THIRD, independent solid-heat channel discovered
    // closing the P-G5 total ledger (Erik's ruling's "book every channel"):
    // without it, `tests/_sealedbox_bisect_bench.py`'s TOTAL ledger identity
    // is BROKEN on every tick with live fire. Folds into the SOLID side's
    // closure identity (temperature_solver.h) alongside `e_solid_deposit_sum`
    // / `e_solid_cond_sum` / `e_thermostat_sum`.
    mutable int64_t e_comb_solid_heat_sum = 0;

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
        // WHY (S, h, w) AND NOT PER SOURCE TILE: Pass A's thread for air cell j
        // is the SINGLE WRITER of everything at index j, including all of its
        // claimant slots (the existing `alloc_face` idiom, itself the cuda_water
        // dq_e/dq_s precedent). A per-source-tile accumulator would be written
        // by several air cells in one pass — atomics, and order-dependent.
        // Keyed identically to `alloc_slot`: slot [s*n + j] is the debt air
        // cell j owes toward the claimant at offset OFF[s] from j.
        //
        // P-O2b: the plane WIDENS from (4, h, w) to (max_claimants, h, w), with
        // S = combustion_draw::slot_count(draw_r) live slots. The slot key is
        // the SOURCE OFFSET, not an enumeration ordinal — that is what makes a
        // carried sub-count debt still mean the same thing next tick even as
        // fires appear and die. Because OFF's ring 1 is exactly D4's order, at
        // draw_r == 1 this IS the shipped (4, h, w) plane, slot for slot, so
        // its digest bytes do not move. At draw_r > 1 the widened layout is a
        // digest-spec VERSION BUMP (tests/field_digest_spec.toml's own change
        // procedure), taken with this patch.
        //
        // RESET RULE (documented because a stale debt is a real bug): a slot is
        // ZEROED the moment its source stops being a burning claimant — i.e.
        // the claim gate fails (not flammable / fuel exhausted / material
        // cannot ignite / below its ignition temperature) or `fire[i] <= 0`
        // (flameless). P-O2b CARRIES THE RULE OVER and adds the one new way a
        // source can stop being a claimant: it is NO LONGER REACHABLE (a door
        // closed, a wall was built, the path flooded) — that slot is zeroed on
        // exactly the same footing. It persists ONLY while that source is an
        // actively burning, currently reachable claimant, so a re-ignition
        // never inherits an old fraction. An out-of-BOUNDS slot is never
        // written at all (it can only ever hold the zero it was born with) —
        // the shipped idiom, preserved verbatim so R = 1 stays byte-identical.
        // Bounded exception, deliberately not chased: an air cell that
        // early-outs before the claim loop (fully O2-starved, `O2 <=
        // o2_thresh_burn`) keeps its sub-count debt until it has oxygen again
        // — under one count.
        //
        // nullptr -> the pre-D1 chained-truncation demand, so every legacy /
        // direct-binding caller stays byte-identical.
        int32_t* dem_acc = nullptr,
        // ---- P-O2b: THE EXTENDED OXYGEN DRAW (see the header block) --------
        // draw_r         : DRAW_R, the BFS hop radius of the draw. 1 == the
        //                  shipped 4-face law, BIT FOR BIT (the regression
        //                  oracle). Ship value 2; 3 is the sweep's upper point.
        //                  HARD-CHECKED against combustion_draw::R_MAX.
        // dyn_permeability : (h, w) float, READ — the per-tile permeability the
        //                  path weight multiplies through. Quantized ONCE per
        //                  step into an integer plane at pass entry (the
        //                  load-time boundary idiom eos_solver.cpp uses for its
        //                  own min-perm face coefficients), so the draw itself
        //                  is pure integer. nullptr -> every open cell reads
        //                  permeability 1.0.
        // max_claimants  : MAX_CLAIMANTS — the DECLARED slot depth of the
        //                  dem_acc plane. HARD-CHECKED at pass entry against
        //                  slot_count(draw_r): a plane too shallow for the
        //                  radius is a VIOLATION (it would alias two sources'
        //                  debts), not a note, and throws. The per-cell
        //                  claimant count can never exceed slot_count(draw_r)
        //                  by construction — the gather loops over the slot
        //                  table itself, one slot per source offset — and that
        //                  is re-checked anyway.
        int draw_r = 1,
        const float* dyn_permeability = nullptr,
        int max_claimants = 4,
        // ---- arc #54 P-G1b: THE PARALLEL TWO-HOP ENERGY LEDGER ------------
        // (docs/gas_energy_conservation_design_2026-08-29.md §2.7, R3-#8/#9.)
        //
        // gas_energy  : (h, w) int64, MUTATED — the conserved gas thermal
        //               energy. nullptr keeps this whole pass byte-identical
        //               to pre-#54 (the direct-binding / unit-test path).
        // is_ambient  : (h, w) bool, READ — the ring mask, needed for the
        //               canonical accountable set (a ring cell holds N and can
        //               donate O2, but carries no `gas_energy`). nullptr ==
        //               space map, the usual dormancy-by-branch.
        // t_amb_q     : T_AMB_K in raw Q16.16 counts. A parcel drawn from a
        //               NON-accountable donor is MINTED at ambient (§2.7's
        //               born-at-ambient rule, and the same convention bulk
        //               transport uses for a non-participating donor).
        int64_t* gas_energy = nullptr,
        const bool* is_ambient = nullptr,
        int32_t t_amb_q = 0,
        // R3 hot-burns-faster (see the header block + fire_T_ext/fire_T_span/
        // hotf_cap above): PER-MATERIAL EXTINCTION TEMPERATURE, int32 (h,w)
        // Q16.16, OPTIONAL — the SAME plane FireSimulation::step reads
        // (GameMap.fire_T_ext_plane: ignition_temp[mat] - ignition_to_ext_
        // delta, baked once at load). nullptr -> the scalar `fire_T_ext`
        // fallback above, i.e. the pre-R3-plane law bit-for-bit (every
        // legacy/direct-binding caller that does not pass one is unmoved).
        const int32_t* fire_T_ext_plane = nullptr
    ) const;
};
