#include "combustion.h"
#include "fixed_point.h"
#include "raycaster.h"   // heat_saturating_add (shared Q16.16 domain)
#include <algorithm>
#include <cstdint>
#include <vector>

using namespace fixedpoint;

// ============================================================================
// EOS P6.9 (docs/eos_p6_9_combustion_design.md) — the row-major SCATTER pass
// is REFORMULATED into two order-free GATHER passes so combustion is direction-
// free and bit-identical CPU<->GPU (P6.9b ports this exact algorithm to CUDA).
//
// This is a BEHAVIORAL change (design §5, blessed by Erik 2026-07-11): the four
// deltas alpha-delta. alpha: ignition reads a pass-entry temperature SNAPSHOT,
// so a source can no longer heat AND ignite a furniture neighbour in the same
// tick (removes the down-right ignition cascade). beta: contested O2 splits
// proportionally (here uniformly — demand is burn_cap for every claimant)
// instead of first-come-first-served, with the fuel payment redistributed to
// match (removes the up-left O2-competition bias). gamma: a contested air cell
// now fully DRAINS its O2 (the old scatter left a sub-threshold sliver per
// source). delta: a multi-source air cell deposits ONE aggregate heat term
// against the post-total-burn N_total (the old scatter deposited per source
// against a running N that fell with each sub-burn). All four are systematic
// (new >= old), bounded, and consequences of making the pass order-free — see
// design §5 for the golden-rebaseline rationale.
//
// S. Feldman, J.F. O'Brien, O. Arikan, "Animating Suspended Particle
// Explosions", SIGGRAPH 2003 — the heat + product-yield + ignition-threshold
// source-term structure this pass follows (constants game-tuned, not lit-derived).
//
// Continuous O2->combustion law (docs/continuous_o2_law_design_2026-07-24.md):
// the per-claimant O2 DEMAND is now PROPORTIONAL in fire intensity I and the
// O2 factor o2f (linear in the air cell's O2 mole fraction) instead of a flat
// gated draw. Credit: Peatross & Beyler 1997 (linear burning-rate vs O2 volume
// fraction) + Huggett 1980 (oxygen-consumption calorimetry, the burn_rate/H_fuel
// anchor). Both archived under docs/papers/ (headers: fire_simulation.cpp).
//
// THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30.md §2 site
// 3): the ONE change here is the aggregate deposit's CONVERSION on a
// thermal_solid burn site — see combustion.h's header block and the branch at
// the deposit itself. Everything else (the claim gate, the demand law, the
// proportional split, the Dalton split, Pass B's fuel payment) is untouched.
// ============================================================================

namespace {

// 4-connected open-neighbour faces (N, S, W, E) — the SAME idiom
// FireSimulation's own O2/smoke passes use.
static constexpr int D4[][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

// Opposite-face index within D4 (N<->S, W<->E). Pass B, walking OUT from a
// source cell i in direction d to an air neighbour j, reads the allocation air
// cell j made toward i — which j filed under j's OWN outbound direction toward
// i, namely D4_OPP[d].
static constexpr int D4_OPP[4] = {1, 0, 3, 2};

static inline bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

// Integer clamp to [0, FP_ONE] (the [0,1] o2f saturation) — mirrors
// fire_simulation.cpp's clamp01_q so the two O2 laws are bit-identical.
static inline q16 clamp01_q(q16 v) {
    if (v < 0) return 0;
    if (v > FP_ONE) return FP_ONE;
    return v;
}

}  // namespace

void CombustionSolver::step(
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
        const bool* thermal_solid,
        const int32_t* heat_inv_shift,
        int32_t* heat,
        int32_t* dem_acc) const {

    if (h <= 0 || w <= 0 || dt <= 0.0f) return;
    if (o2_idx < 0 || o2_idx >= n_gases) return;
    if (inert_n2_idx < 0 || inert_n2_idx >= n_gases) return;
    if (black_smoke_idx < 0 || black_smoke_idx >= n_gases) return;
    // fire[] is READ again (continuous-O2 law §2.3): it is the per-claimant
    // intensity factor I_k in demand_k = burn_cap*I_k*o2f_j. (P6.9 had dropped
    // it as an outcome-neutral prefilter; the demand law reinstates it.)

    const int n = h * w;
    int32_t* O2   = gas + (size_t)o2_idx * n;
    int32_t* N2   = gas + (size_t)inert_n2_idx * n;
    int32_t* SOOT = gas + (size_t)black_smoke_idx * n;

    // Load-time constants (double-fold once, then quantize — the LOCKED
    // per-step-scalar idiom shared by eos_solver.cpp / fire_simulation.cpp).
    const q16 burn_cap_q   = quantize((double)burn_rate * (double)dt);
    const q16 o2_thresh_q  = quantize((double)o2_thresh_burn);
    const q16 soot_yield_q = quantize((double)soot_yield);
    const q16 H_fuel_q     = quantize((double)H_fuel);
    // v2.5 (P5.1): wall_hp consumed per unit N_O2 burned — the ember-scale
    // stoichiometric fuel cost (design §5 v2.5 amendment, decisions #17).
    const q16 fuel_per_o2_q = quantize((double)fuel_per_o2);
    // P-R4: the fuel-bed deposit's mantissa, quantized ONCE per step like every
    // other per-step scalar (combustion.h documents the split and why the
    // mantissa carries most of the magnitude). The shift is applied per-deposit
    // in int64 (see the H_bed site in Pass A) so a large burn cannot overflow.
    const q16 H_bed_m_q     = quantize((double)H_BED_M);
    const int H_bed_shift   = (H_BED_SHIFT > 0) ? H_BED_SHIFT : 0;
    const double c_v_safe  = (c_v > 0.0f) ? (double)c_v : 1.0;
    const int64_t recip_cv = make_recip(c_v_safe);              // 1/c_v, once per step
    const q16 n_floor_q    = quantize((double)n_floor_heat);
    // v2.4 T_MAX_PHYS rail (combustion.h; full rationale in eos_solver.h).
    const q16 t_max_phys_q = quantize((double)T_MAX_PHYS);

    // Continuous-O2 law (design §2.3): o2f_j = clamp01((X_j - X_ext)/(X_full -
    // X_ext)), X_j = O2[j]/max(O2[j]+N2[j], floor). SAME hoisted constants +
    // floor as fire_simulation.cpp (the two laws must read identically). X_ext
    // = 0 -> span == X_full (pure proportional). X_full <= X_ext -> step at
    // X_ext. FULL-RESPONSE REFERENCE SPLIT (2026-07-30): the upper end is the
    // PURE-O2 reference o2_frac_full, NOT o2_frac_amb (which made ambient the
    // ceiling); o2_frac_amb is no longer read here.
    const q16 x_ext_q          = quantize((double)o2_frac_ext);
    const double x_span        = (double)o2_frac_full - (double)o2_frac_ext;
    const bool   x_degenerate  = (x_span <= 0.0);
    const int64_t recip_x_span = x_degenerate ? 0 : make_recip(x_span);
    const q16 X_N_FLOOR        = quantize(0.01);   // 655 counts (see fire_simulation.cpp)

    if (burn_cap_q <= 0) return;   // nothing burns this tick (dt~0 or burn_rate 0)

    // --- Snapshot (design §3, the explicit freeze) --------------------------
    // temperature is read by the ignition GATE (Tsnap[i] >= ign[i]) AND written
    // by the heat DEPOSIT (temperature[j] += dT) — a genuine cross-cell read-
    // after-write that only an explicit pass-entry copy breaks. Reading Tsnap in
    // the gate is exactly delta alpha: a source cannot heat a neighbour and
    // ignite it the SAME tick. (O2[j] and wall_hp[i] are frozen IMPLICITLY by
    // the gather structure — design §3 — so they need no copy.)
    std::vector<q16> Tsnap(temperature, temperature + n);

    // --- Per-face allocation buffers (design §3 plumbing (a)) ---------------
    // alloc_face[d*n + j] = the O2 that air cell j allocates to the flammable
    // source in direction D4[d] of j. Pass A (single writer per air cell) fills
    // it; Pass B gathers each source's <=4 incoming faces. The cuda_water
    // dq_e/dq_s precedent — direction-keyed, single-writer, no recompute in B
    // (design §3 rejects recompute-in-B: it would fork the split logic and risk
    // O2-drained-at-j vs fuel-paid-by-i silently desynchronizing).
    std::vector<q16> alloc_face((size_t)4 * n, 0);

    // ======================================================================
    // Pass A — air cells. Single writer of O2[j], SOOT[j], N2[j],
    // temperature[j], and the four face buffers at index j.
    // ======================================================================
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int j = row + x;
            // Burn happens only in an OPEN-air cell (the flame front in the air
            // pocket next to the fuel). A flammable tile is itself solid and
            // holds no gas (bulk_transport: a solid cell has N == 0).
            if (solid[j] || is_vacuum[j]) continue;

            // Pass-entry O2 at THIS cell (Pass A is its sole writer; read-before-
            // write => every claimant sees pass-entry O2 — deltas beta/gamma).
            const q16 o2j = O2[j];
            if (o2j <= o2_thresh_q) continue;   // epsilon skip-floor (RETIRED gate)

            // o2f_j — the continuous-O2 factor at THIS air cell, LINEAR in its
            // O2 MOLE FRACTION X_j = O2[j]/(O2[j]+N2[j]) (pass-entry; N2[j] is
            // also single-written by this cell, read-before-write). N_total is
            // the conservative bulk (O2+N2), soot EXCLUDED — one source of truth
            // with the EOS/temperature N_total and the fire logistic's law.
            const int64_t n_tot_j = (int64_t)o2j + (int64_t)N2[j];
            const q16 den_j = (n_tot_j < (int64_t)X_N_FLOOR) ? X_N_FLOOR : (q16)n_tot_j;
            const q16 Xj = mul_q16(o2j, reciprocal_q16(den_j));
            const q16 o2f_j = x_degenerate
                ? ((Xj < x_ext_q) ? (q16)0 : (q16)FP_ONE)
                : clamp01_q(recip_mul(Xj - x_ext_q, recip_x_span));

            // Gather the <=4 flammable claimant sources + each one's per-claimant
            // DEMAND (design §2.3): demand_k = burn_cap * I_k * o2f_j (PINNED
            // left-fold mul_q16, truncating — a conservative request that never
            // over-draws). I_k = fire[i] is now READ (the old pass dropped it as
            // an outcome-neutral prefilter; the continuous law makes it the
            // intensity factor). A flameless claimant (I_k == 0, e.g. a hot
            // ember) demands 0 -> draws no O2, deposits no heat: "a choked/cool
            // fire consumes nothing" (design §2.3).
            int cl_dir[4];   // D4 index of the claimant (face key for alloc_face)
            int cl_src[4];   // global cell index of the claimant source
            int64_t dem[4];  // per-claimant O2 demand this tick (Q16.16 counts)
            int n_cl = 0;
            for (int d = 0; d < 4; ++d) {
                const int iy = y + D4[d][0], ix = x + D4[d][1];
                if (!in_bounds(iy, ix, h, w)) continue;
                const int i = iy * w + ix;
                // Claim gate (design §3 step 1) — the ign/T/fuel gate is the real
                // one (unchanged). fire[i] is now read for the DEMAND magnitude,
                // not as a claim prefilter.
                // D1 RESET RULE: this slot's sub-count debt survives only while
                // the neighbour is an ACTIVELY BURNING claimant. Every gate
                // below that rejects it zeroes the debt first, so a tile that
                // stops burning and later re-ignites starts from clean books.
                const size_t slot = (size_t)d * n + j;
                if (!flammable[i]) { if (dem_acc) dem_acc[slot] = 0; continue; }
                if (wall_hp[i] <= FUEL_FLOOR) {           // no fuel (P5.1 ember out)
                    if (dem_acc) dem_acc[slot] = 0; continue; }
                const q16 ign_i = ignition_temp_q16[i];
                if (ign_i <= 0) { if (dem_acc) dem_acc[slot] = 0; continue; }
                // *** IGNITION vs SUSTAIN (P-R4 finding, measured) ***
                // This gate used to be a bare `Tsnap[i] < ign_i -> skip`, i.e.
                // it demanded a tile stay above its own IGNITION temperature to
                // keep consuming oxygen. That is an ignition threshold doing a
                // sustain threshold's job — the mirror image of the defect
                // P-R3's ride-along fixed on the fire logistic ("a tile could
                // ignite below its own sustain floor"). It was INVISIBLE while
                // the painter existed, because the painter's own deposit held a
                // burning tile above `ignition_temp` from tick one. With the
                // painter retired it deadlocks the whole chain: a tile ignites
                // at exactly `ignition_temp`, cools by one `cool_shift` step
                // within a single tick, and from tick 2 is no longer allowed to
                // draw the oxygen whose heat is the only thing that could have
                // kept it there. MEASURED on the bench: T 280.0 -> 279.45 after
                // one tick, claim gate false for every tick after, zero oxygen
                // drawn for the rest of the run, fire dead at 21 s.
                // The fix is the standard hysteresis pair: IGNITION temperature
                // gates a tile that is not yet alight; a tile that IS alight
                // (fire[i] > 0) burns on, and its death is the fire logistic's
                // job through `fire_T_ext` (180 for furniture — 100 game BELOW
                // ignition_temp, exactly the band this gate was excluding) plus
                // the `I_min` snap-out. Outcome-neutral in the other direction:
                // a NON-burning tile has I == 0, so its demand was already 0 and
                // it drew nothing.
                const bool alight = (fire[i] > 0);
                // Not alight -> must clear its own IGNITION temperature, read
                // from the pass-entry SNAPSHOT so a source can never heat AND
                // ignite a neighbour in the same tick (design delta alpha).
                if (!alight && Tsnap[i] < ign_i) {
                    if (dem_acc) dem_acc[slot] = 0;
                    continue;
                }
                // Flameless claimant: demand is proportional to I, so it is
                // exactly 0. Skip rather than thread a zero through the split —
                // outcome-identical, and it keeps the debt books clean.
                if (!alight) {
                    if (dem_acc) dem_acc[slot] = 0;
                    continue;
                }
                q16 di;
                if (dem_acc != nullptr) {
                    // ---- D1: error-feedback demand (combustion.h documents
                    // the scale algebra and why the plane is face-keyed). The
                    // WIDE product is never truncated; the sub-count remainder
                    // is carried in this slot and whole counts fall out as the
                    // debt accrues. Exact in expectation, unbiased, order-free
                    // (single writer per air cell), Huggett anchor untouched.
                    //   P    = burn_cap_q * I_q * o2f_q   (<= 2^48, scale 2^32)
                    //   wide = acc + (P >> 1)             (scale 2^31 per count)
                    //   draw = wide >> 31                 (whole counts)
                    //   acc  = wide - (draw << 31)        ([0, 2^31) -> int32)
                    // The single `>> 1` costs 2^-32 of a count per tick — six
                    // orders below the ~1 count/tick draw, and it is what lets
                    // the remainder live in a plain non-negative int32 (a
                    // scale-2^32 remainder would need 33 bits).
                    const int64_t P = (int64_t)burn_cap_q
                                    * (int64_t)fire[i] * (int64_t)o2f_j;
                    const int64_t wide = (int64_t)dem_acc[slot] + (P >> 1);
                    const int64_t draw = wide >> 31;
                    dem_acc[slot] = (int32_t)(wide - (draw << 31));
                    di = (q16)draw;
                } else {
                    // Pre-D1 path (nullable back-compat): the chained
                    // truncation, kept so direct-binding callers are unmoved.
                    di = mul_q16(mul_q16(burn_cap_q, fire[i]), o2f_j);
                }
                cl_dir[n_cl] = d;
                cl_src[n_cl] = i;
                dem[n_cl] = (int64_t)di;
                ++n_cl;
            }
            if (n_cl == 0) continue;

            // --- Allocate O2[j] across the claimants (design §3 step 2) ------
            // D = Σ demand_k (now NON-uniform — demand varies with each source's
            // intensity). A whole cell whose claimants all demand 0 (all flameless
            // / choked) draws nothing — skip it (no writes, no telemetry).
            int64_t alloc[4];
            int64_t D = 0;
            for (int k = 0; k < n_cl; ++k) D += dem[k];
            if (D == 0) continue;
            int64_t burn_j;
            if (D <= (int64_t)o2j) {
                // No contention: every claimant gets its full demand; a sub-
                // threshold O2 sliver may remain (as the old uncontended burn).
                for (int k = 0; k < n_cl; ++k) alloc[k] = dem[k];
                burn_j = D;
            } else {
                // Contention: EXACT INTEGER proportional split (plain int64 `/`
                // and `%` — NOT float, NOT reciprocal_q16: integer divide has a
                // single portable answer and keeps sum(alloc) == O2[j] exactly;
                // reciprocal_q16 is ~1 ULP inexact and would break conservation).
                // num_k = O2[j] * demand_k (the demand-weighted share). Q16.16
                // scale cancels (the split is a dimensionless ratio).
                int64_t keys[4];
                int64_t sum_alloc = 0;
                for (int k = 0; k < n_cl; ++k) {
                    const int64_t num = (int64_t)o2j * dem[k];  // < 2^43
                    alloc[k] = num / D;      // floor, exact integer divide
                    keys[k]  = num % D;      // integer remainder = tiebreak key
                    sum_alloc += alloc[k];
                }
                // R leftover LSBs (provably in [0, n_cl) subset of [0,4)) go to
                // the R claimants with the largest key; ties -> lowest source
                // index (a fixed, bounded sub-LSB bias the isotropy test §6
                // tolerates at <=3 LSB).
                int64_t R = (int64_t)o2j - sum_alloc;
                bool chosen[4] = {false, false, false, false};
                for (int r = 0; r < (int)R; ++r) {
                    int best = -1;
                    for (int k = 0; k < n_cl; ++k) {
                        if (chosen[k]) continue;
                        if (best < 0 ||
                            keys[k] > keys[best] ||
                            (keys[k] == keys[best] && cl_src[k] < cl_src[best])) {
                            best = k;
                        }
                    }
                    chosen[best] = true;
                    alloc[best] += 1;
                }
                burn_j = (int64_t)o2j;   // contested cells fully drain (delta gamma)
            }

            // --- Single-writer gas + heat writes at cell j (design §3 step 3) -
            O2[j] = (q16)((int64_t)o2j - burn_j);
            // Exact Dalton split (unchanged): soot + (burn-soot) == burn, so
            // N_total is conserved to the LSB regardless of soot rounding (#12).
            const q16 soot = narrow_round(mul_wide((q16)burn_j, soot_yield_q));
            SOOT[j] += soot;
            N2[j]   += (q16)(burn_j - (int64_t)soot);

            // ONE aggregate heat deposit against the POST-burn N_total (delta
            // delta) — same idiom/dials as TemperatureSolver's Pass-1 radiative
            // deposit; a per-source replay would reintroduce an order-dependent
            // denominator and defeat isotropy. Rail counters are now PER-CELL
            // (design §3): no test may assert their absolute value.
            const q16 deposit = mul_q16((q16)burn_j, H_fuel_q);   // burn*H_fuel
            q16 dT;
            // THERMAL-MASS AXIS, P-EOS (ruling §2 site 3): the MEDIUM branch on
            // the deposit's CONVERSION. A furniture tile is an open, gas-holding
            // burn site (permeability 0.5) but it is thermally an OBJECT, and
            // under ruling A3 its pore gas is thin — so dividing by that thin N
            // would inflate the object's T by ~2.5-3x per unit burn. Convert via
            // the tile's own heat_inv_shift instead: the SAME free bit-shift
            // TemperatureSolver's MEDIUM-TEST SITE 5/6 applies to a ray deposit
            // (`deposit >> log2(thermal_mass)`), so ray heat and combustion heat
            // reach an object on ONE scale. Same energy in, object-appropriate
            // conversion; the n_floor_heat counter is deliberately NOT touched on
            // this path — there is no gas divisor here to floor.
            const bool object_site = (thermal_solid != nullptr)
                                  && (heat_inv_shift != nullptr)
                                  && thermal_solid[j];
            if (object_site) {
                const int shift = heat_inv_shift[j];   // log2(thermal_mass), >= 0
                dT = deposit >> shift;
            } else {
                q16 n_total_j = (q16)((int64_t)O2[j] + (int64_t)N2[j]);
                if (n_total_j < n_floor_q) { n_total_j = n_floor_q; ++heat_floor_hits; }
                const q16 recip_n  = reciprocal_q16(n_total_j);
                const q16 e_over_n = mul_q16(deposit, recip_n);        // .../N
                dT                 = recip_mul(e_over_n, recip_cv);    // .../c_v
            }
            heat_saturating_add(&temperature[j], dT);
            if (temperature[j] > t_max_phys_q) {                   // v2.4 rail
                temperature[j] = t_max_phys_q; ++t_max_phys_hits;
            }

            // Record each claimant's allocation on the face buffer so Pass B can
            // charge the SOURCE for the O2 it drew from this air cell.
            //
            // P-R4 (ruling A1): the same loop now also pays each claimant its
            // FUEL-BED deposit — the flame heating the surface it is burning
            // off, which is what owns the plateau now that the painter is gone.
            //   H_bed_k = mul_q16(burn_k, H_BED_M) << H_BED_SHIFT
            // strictly proportional to the O2 that claimant ACTUALLY got
            // (`alloc[k]`, not its demand) — so a choked claimant deposits
            // nothing and the plateau sags with local O2, by design.
            //
            // ORDER-FREE: a positive saturating add into `heat[]`, exactly the
            // contract the retired ray deposit used, so several air cells
            // feeding one source in any order give the same total (and the CUDA
            // twin's atomic is the same add). The shift is taken in int64 and
            // clamped before the add: `alloc[k]` can reach a full ambient O2
            // cell (~1.4e4 counts) and mul_q16 by a near-format-max mantissa
            // then << shift would otherwise leave int32.
            for (int k = 0; k < n_cl; ++k) {
                alloc_face[(size_t)cl_dir[k] * n + j] = (q16)alloc[k];
                if (heat != nullptr && alloc[k] > 0 && H_bed_m_q > 0) {
                    int64_t bed = (int64_t)mul_q16((q16)alloc[k], H_bed_m_q);
                    bed <<= H_bed_shift;
                    if (bed > (int64_t)INT32_MAX) bed = (int64_t)INT32_MAX;
                    heat_saturating_add(&heat[cl_src[k]], (int32_t)bed);
                }
            }
        }
    }

    // ======================================================================
    // Pass B — source cells. Single writer of wall_hp[i]. Each flammable source
    // sums its <=4 incoming face allocations and pays the stoichiometric fuel
    // cost ONCE for the total, floored ONCE at FUEL_FLOOR (design §3, "total-
    // then-floor-once", critique B). This never takes wall_hp below 1 LSB
    // (the "smolder never destroys" invariant — decisions #17): the floor is
    // re-applied after the single subtraction, so wall_hp[i] >= FUEL_FLOOR
    // always. Structural destruction stays exclusively FireSimulation's I>0 path.
    // ======================================================================
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (!flammable[i]) continue;
            int64_t burn_i = 0;
            for (int d = 0; d < 4; ++d) {
                const int jy = y + D4[d][0], jx = x + D4[d][1];
                if (!in_bounds(jy, jx, h, w)) continue;
                const int j = jy * w + jx;
                // The air neighbour j filed its allocation toward THIS source
                // under j's outbound direction to i, which is D4_OPP[d].
                burn_i += (int64_t)alloc_face[(size_t)D4_OPP[d] * n + j];
            }
            if (burn_i == 0) continue;   // this source drew no O2 this tick
            // round-to-nearest — the same unbiased-sink idiom fire_simulation's
            // wall_damage depletion uses.
            const q16 fuel_cost = narrow_round(mul_wide(fuel_per_o2_q, (q16)burn_i));
            wall_hp[i] -= fuel_cost;
            if (wall_hp[i] < FUEL_FLOOR) wall_hp[i] = FUEL_FLOOR;
        }
    }
}
