#include "combustion.h"
#include "fixed_point.h"
#include "raycaster.h"   // heat_saturating_add (shared Q16.16 domain)
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
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
//
// P-O2b — THE EXTENDED OXYGEN DRAW (docs/fire_realism_design_2026-08-01.md
// v5.2 "F-O2b", Erik's Option 2b). The LAW lives in combustion.h's header
// block; what changed STRUCTURALLY here is that the two gathers became THREE,
// because the deposits were re-sited away from the donor cells:
//
//   Pass A — air cells. Enumerates the burning tiles that can reach this cell
//     within DRAW_R hops (the reverse relaxation), allocates this cell's O2
//     across them, DEBITS O2 here, files each allocation on the offset-keyed
//     `alloc_slot` buffer, and pays each claimant its H_bed fuel-bed deposit.
//     It no longer deposits soot / N2 / heat: those are not this cell's.
//   Pass B — source cells. Sums its incoming allocations, pays the
//     stoichiometric fuel cost once, and decides WHERE its combustion products
//     land — its own tile plus its open faces — filing that on `dep_site`.
//   Pass C — air cells. Gathers the deposit intents aimed at this cell and
//     applies the shipped soot / N2 / heat deposit, verbatim.
//
// Single-writer discipline (and with it order freedom and CPU/GPU bit
// identity) survives intact: every plane still has exactly one writing cell.
// At DRAW_R == 1 the three passes reduce EXACTLY to the shipped two — Pass B's
// hop-2 remainder is identically zero, so each face receives back precisely
// the allocation it made, which is the old per-air-cell aggregate. That
// identity is gated byte-for-byte over the full engine.
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

// P-O2b — apply the Q16.16 DRAW WEIGHT to the wide (scale-2^32) demand product.
//
// Returns EXACTLY floor((P * wq) / 2^16) for P >= 0, computed without 128-bit
// arithmetic by splitting P at 16 bits:
//     P*wq/2^16 = hi*wq + (lo*wq)/2^16      (P = hi*2^16 + lo)
// and hi*wq is already an integer, so the floor distributes. Bounds: P <= 2^48
// and wq <= FP_ONE give hi*wq <= 2^48 and lo*wq <= 2^32 — no int64 overflow,
// and no precision is thrown away at all.
//
// THE POINT: at wq == FP_ONE this is the EXACT IDENTITY (hi<<16 | lo == P), bit
// for bit — which is why DRAW_R == 1, where every reachable cell has d == 1 and
// w_path == 1, reproduces the shipped demand byte-identically. A cheaper
// `(P >> 16) * wq` would NOT (it would drop P's low 16 bits) and a plain
// `P * wq >> 16` could overflow; this form is both exact and safe.
static inline int64_t apply_draw_weight(int64_t P, q16 wq) {
    const int64_t hi = P >> 16;
    const int64_t lo = P & 0xFFFF;
    return hi * (int64_t)wq + ((lo * (int64_t)wq) >> 16);
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
        int32_t* dem_acc,
        int draw_r,
        const float* dyn_permeability,
        int max_claimants) const {

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

    // ======================================================================
    // P-O2b — THE EXTENDED DRAW's load-time bake (combustion.h's header block
    // carries the law; this is only its arithmetic set-up).
    // ======================================================================
    namespace cd = combustion_draw;

    // HARD CHECKS. v5.2: "MAX_CLAIMANTS per air cell is a config constant with
    // a HARD assert (a hit cap is a violation, not a note)." A radius past the
    // baked tables, or a dem_acc plane too shallow for the radius, would ALIAS
    // two sources' sub-count debts onto one slot — a silent, synced-state
    // corruption. It throws instead.
    if (draw_r < 1 || draw_r > cd::R_MAX) {
        throw std::runtime_error(
            "CombustionSolver: draw_r out of range (1.." + std::to_string(cd::R_MAX) + ")");
    }
    const int n_slots = cd::slot_count(draw_r);   // 4, 12, 24
    const int n_ball  = cd::ball_count(draw_r);   // 1, 5, 13
    if (dem_acc != nullptr && max_claimants < n_slots) {
        throw std::runtime_error(
            "CombustionSolver: MAX_CLAIMANTS cap hit — dem_acc depth "
            + std::to_string(max_claimants) + " < slot_count(draw_r) "
            + std::to_string(n_slots));
    }
    // The dem_acc plane's DECLARED depth: slot s lives at [s*n + j]. Only the
    // first n_slots slots are live; a deeper plane simply carries unused rows.
    const int acc_depth = (dem_acc != nullptr) ? max_claimants : n_slots;
    (void)acc_depth;   // slots are indexed by s < n_slots <= acc_depth

    // W_hop — the BAKED hop-weight table, quantize(1/(1+d)) NORMALIZED so that
    // W_hop[1] == FP_ONE, i.e. W_hop[d] = quantize(2/(1+d)) = {1, 2/3, 1/2}.
    // The normalization is the whole R = 1 identity: at radius 1 every claimant
    // sits at d == 1 with an empty (weight-1) path, so the draw weight is
    // exactly FP_ONE and apply_draw_weight is the identity.
    q16 w_hop_q[cd::R_MAX + 1];
    w_hop_q[0] = 0;                                    // d == 0 is not a draw site
    for (int d = 1; d <= cd::R_MAX; ++d) {
        w_hop_q[d] = quantize(2.0 / (1.0 + (double)d));
    }

    // The permeability plane, QUANTIZED ONCE per step (the load-time boundary
    // idiom eos_solver.cpp uses for its own min-perm face coefficients) so the
    // path walk below is pure integer — no float in the per-cell draw. quantize
    // is monotone, so min-then-quantize and quantize-then-min agree exactly.
    // No plane -> permeability 1.0 everywhere (byte-identical to no weighting).
    std::vector<q16> perm_q;
    if (draw_r > 1 && dyn_permeability != nullptr) {
        perm_q.resize(n);
        for (int i = 0; i < n; ++i) perm_q[i] = quantize((double)dyn_permeability[i]);
    }
    const bool has_perm = !perm_q.empty();

    // --- Snapshot (design §3, the explicit freeze) --------------------------
    // temperature is read by the ignition GATE (Tsnap[i] >= ign[i]) AND written
    // by the heat DEPOSIT (temperature[j] += dT) — a genuine cross-cell read-
    // after-write that only an explicit pass-entry copy breaks. Reading Tsnap in
    // the gate is exactly delta alpha: a source cannot heat a neighbour and
    // ignite it the SAME tick. (O2[j] and wall_hp[i] are frozen IMPLICITLY by
    // the gather structure — design §3 — so they need no copy.)
    std::vector<q16> Tsnap(temperature, temperature + n);

    // --- Per-SLOT allocation buffers (design §3 plumbing (a)) ---------------
    // alloc_slot[s*n + j] = the O2 that air cell j allocates to the flammable
    // source at offset OFF[s] from j. Pass A (single writer per air cell) fills
    // it; Pass B gathers each source's incoming slots. The cuda_water dq_e/dq_s
    // precedent — offset-keyed, single-writer, no recompute in B (design §3
    // rejects recompute-in-B: it would fork the split logic and risk
    // O2-drained-at-j vs fuel-paid-by-i silently desynchronizing).
    // P-O2b widens this from the 4 faces to the n_slots draw offsets; at
    // draw_r == 1, n_slots == 4 and the slot keys ARE the D4 face keys.
    std::vector<q16> alloc_slot((size_t)n_slots * n, 0);

    // --- P-O2b: the DEPOSIT-SITE buffer (v5.2's re-sited deposits) ----------
    // dep_site[t*n + i] = the burnt-O2 quantity flammable source i deposits at
    // site t, where t in [0,4) is the face direction D4[t] out of i and t == 4
    // is i's OWN tile (which is an open, gas-holding cell exactly when i is
    // furniture). Pass B (single writer per source) fills it; Pass C gathers,
    // so every gas/heat write still has ONE writer — its own air cell.
    //
    // WHY IT EXISTS (design v5.2, honouring Erik's ruling 4 "air is heated at
    // the fire ONLY"): the O2 is debited at the DONOR cells, which under an
    // extended draw may be two or three tiles away, but the flame — and so the
    // heat, the soot and the hot-air visuals that read air temperature — is at
    // the FIRE. The products must not appear in a donor cell across the room.
    //
    // It is per-tick scratch, wiped every step, never synced and never
    // digested (design §5's rule for new planes). Cost, priced on both
    // backends: one 5*n int32 allocation + memset per tick, host-side here and
    // one cudaMalloc/cudaMemset/cudaFree in the CUDA twin.
    std::vector<q16> dep_site((size_t)5 * n, 0);

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

            // ---- P-O2b STEP 1: THE REVERSE RELAXATION -----------------------
            // Which burning tiles can reach THIS cell in <= draw_r hops? The
            // walk is the forward walk read backwards (every intermediate cell
            // is open and adjacency is symmetric), so we expand outward from j
            // through OPEN CELLS ONLY over the baked BALL offsets, in draw_r-1
            // LEVELLED rounds. `bd[b]` is the BFS hop distance from j to ball
            // cell b (-1 = unreached); `bw[b]` is the MAXIMUM permeability-
            // multiplicative path weight over j's min-hop paths to b.
            //
            // ORDER-FREEDOM, per site: round r reads only cells stamped r-1 and
            // stamps only previously-unstamped cells, so no cell both feeds and
            // is fed within a round; and the combine over the (at most four)
            // incoming candidates is a MAX, which is commutative and
            // associative. The result therefore does not depend on the order of
            // the b or d loops — nor on the order in which burning tiles are
            // enumerated anywhere in the pass.
            //
            // At draw_r == 1 this whole block is: n_ball == 1, no rounds run,
            // bd[0] == 0, bw[0] == FP_ONE. Free, and exactly the shipped law.
            int   bcell[cd::BALL_MAX];   // grid index of each ball cell (-1 = OOB/closed)
            int8_t bd[cd::BALL_MAX];
            q16   bw[cd::BALL_MAX];
            for (int b = 0; b < n_ball; ++b) {
                bd[b] = -1; bw[b] = 0; bcell[b] = -1;
                const int cy = y + cd::BALL_DY[b], cx = x + cd::BALL_DX[b];
                if (!in_bounds(cy, cx, h, w)) continue;
                const int c = cy * w + cx;
                // Expansion runs THROUGH OPEN CELLS ONLY: a solid tile is never
                // traversed (a wall breathes only via its open faces), and a
                // vacuum cell TERMINATES expansion (there is no air to walk).
                if (solid[c] || is_vacuum[c]) continue;
                bcell[b] = c;
            }
            bd[0] = 0; bw[0] = FP_ONE;    // the donor cell itself: hop 0, empty path
            for (int r = 1; r < draw_r; ++r) {
                for (int b = 0; b < n_ball; ++b) {
                    if (bd[b] >= 0 || bcell[b] < 0) continue;
                    q16 best = 0;
                    for (int d = 0; d < 4; ++d) {
                        const int nb = cd::BALL_NBR.v[b][d];
                        if (nb < 0 || nb >= n_ball) continue;
                        if (bd[nb] != (int8_t)(r - 1)) continue;   // previous level only
                        // Face weight = min(perm) across the traversed face —
                        // the physics_engine/eos_solver idiom, in integers.
                        // A crate (0.5) attenuates the draw THROUGH itself
                        // (0.5 in, 0.5 out); a wall (0) blocks outright.
                        q16 pf = FP_ONE;
                        if (has_perm) {
                            const q16 pa = perm_q[bcell[nb]], pb = perm_q[bcell[b]];
                            pf = (pa < pb) ? pa : pb;
                        }
                        const q16 cand = mul_q16(bw[nb], pf);
                        if (cand > best) best = cand;
                    }
                    // A zero-weight arrival is NOT a path (perm 0 blocks).
                    if (best > 0) { bd[b] = (int8_t)r; bw[b] = best; }
                }
            }

            // ---- P-O2b STEP 2: reduce the walk onto the SOURCE SLOTS --------
            // Every open cell the walk reached, at hop e <= draw_r-1, offers its
            // four face-neighbours as claimants at hop e+1 (the burning tile's
            // own step onto its open face is NOT permeability-attenuated: the
            // solid fuel tile is never traversed). A source can be seen from
            // several ball cells, so each slot takes the canonical best:
            // MINIMUM hop distance first, then MAXIMUM path weight — a
            // lexicographic min/max, again an order-free reduction.
            int8_t sd[cd::SLOTS_MAX];
            q16    sw[cd::SLOTS_MAX];
            for (int s = 0; s < n_slots; ++s) { sd[s] = -1; sw[s] = 0; }
            for (int b = 0; b < n_ball; ++b) {
                if (bd[b] < 0) continue;
                const int hop = (int)bd[b] + 1;      // <= draw_r by construction
                const q16 wp  = bw[b];
                for (int d = 0; d < 4; ++d) {
                    const int s = cd::BALL_SLOT.v[b][d];
                    if (s < 0 || s >= n_slots) continue;   // (0,0), or past the radius
                    if (sd[s] < 0 || hop < (int)sd[s] ||
                        (hop == (int)sd[s] && wp > sw[s])) {
                        sd[s] = (int8_t)hop; sw[s] = wp;
                    }
                }
            }

            // Gather the flammable claimant sources + each one's per-claimant
            // DEMAND (design §2.3, generalized by v5.2):
            //     demand_k = burn_cap * I_k * o2f_j * W_hop[d_k] * w_path_k
            // (PINNED left-fold mul_q16, truncating — a conservative request
            // that never over-draws). I_k = fire[i] is READ (the old pass
            // dropped it as an outcome-neutral prefilter; the continuous law
            // makes it the intensity factor). A flameless claimant (I_k == 0,
            // e.g. a hot ember) demands 0 -> draws no O2, deposits no heat: "a
            // choked/cool fire consumes nothing" (design §2.3).
            //
            // The loop walks the SLOT TABLE in its canonical order, one slot per
            // source offset, so the claimant count can never exceed n_slots —
            // the MAX_CLAIMANTS cap is structural, not hoped for.
            int cl_slot[cd::SLOTS_MAX];  // slot key of the claimant (alloc_slot key)
            int cl_src[cd::SLOTS_MAX];   // global cell index of the claimant source
            int64_t dem[cd::SLOTS_MAX];  // per-claimant O2 demand this tick (Q16.16)
            int n_cl = 0;
            for (int sl = 0; sl < n_slots; ++sl) {
                const int iy = y + cd::OFF_DY[sl], ix = x + cd::OFF_DX[sl];
                // Out of bounds: never written at all, so it can only ever hold
                // the zero it was born with — the shipped idiom, preserved
                // verbatim (zeroing here would be a no-op in practice but would
                // no longer be byte-identical for a hand-seeded dem_acc).
                if (!in_bounds(iy, ix, h, w)) continue;
                const int i = iy * w + ix;
                // Claim gate (design §3 step 1) — the ign/T/fuel gate is the real
                // one (unchanged). fire[i] is now read for the DEMAND magnitude,
                // not as a claim prefilter.
                // D1 RESET RULE: this slot's sub-count debt survives only while
                // the source is an ACTIVELY BURNING, REACHABLE claimant. Every
                // gate below that rejects it zeroes the debt first, so a tile
                // that stops burning (or stops being reachable) and later
                // re-ignites starts from clean books.
                const size_t slot = (size_t)sl * n + j;
                // P-O2b: the one NEW way a source stops being a claimant — the
                // draw can no longer reach it (a door closed, a wall was built,
                // the path flooded). Same footing as the gates below. At
                // draw_r == 1 every in-bounds slot is always reachable
                // (bd[0] == 0 unconditionally), so this never fires there.
                if (sd[sl] < 0) { if (dem_acc) dem_acc[slot] = 0; continue; }
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
                // ---- P-O2b: the DRAW WEIGHT for this (source, cell) pair ----
                //     wq = W_hop[d] * w_path
                // At draw_r == 1: d == 1 so W_hop[1] == FP_ONE, and the path is
                // empty so w_path == FP_ONE; mul_q16(FP_ONE, FP_ONE) == FP_ONE
                // exactly, and every use of wq below is then the identity.
                const q16 wq = mul_q16(w_hop_q[(int)sd[sl]], sw[sl]);
                q16 di;
                if (dem_acc != nullptr) {
                    // ---- D1: error-feedback demand (combustion.h documents
                    // the scale algebra and why the plane is slot-keyed). The
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
                    const int64_t P0 = (int64_t)burn_cap_q
                                     * (int64_t)fire[i] * (int64_t)o2f_j;
                    // P-O2b: fold the draw weight into the WIDE product, before
                    // the accumulator sees it, so the sub-count error feedback
                    // is carried on the weighted demand (a hop-2 claimant with
                    // a 2/3 weight still draws its exact share in expectation,
                    // rather than being truncated away). EXACT identity at
                    // wq == FP_ONE — see apply_draw_weight.
                    const int64_t P = apply_draw_weight(P0, wq);
                    const int64_t wide = (int64_t)dem_acc[slot] + (P >> 1);
                    const int64_t draw = wide >> 31;
                    dem_acc[slot] = (int32_t)(wide - (draw << 31));
                    di = (q16)draw;
                } else {
                    // Pre-D1 path (nullable back-compat): the chained
                    // truncation, kept so direct-binding callers are unmoved.
                    // The weight joins the same PINNED left fold, and
                    // mul_q16(x, FP_ONE) == x, so draw_r == 1 is unmoved too.
                    di = mul_q16(mul_q16(mul_q16(burn_cap_q, fire[i]), o2f_j), wq);
                }
                cl_slot[n_cl] = sl;
                cl_src[n_cl] = i;
                dem[n_cl] = (int64_t)di;
                ++n_cl;
            }
            if (n_cl == 0) continue;
            // Structural, but re-checked: the gather loops the slot table, one
            // slot per source offset, so this cannot fire. v5.2 wants a hit
            // treated as a violation rather than a note, so it throws.
            if (n_cl > n_slots) {
                throw std::runtime_error("CombustionSolver: MAX_CLAIMANTS cap hit in Pass A");
            }

            // --- Allocate O2[j] across the claimants (design §3 step 2) ------
            // D = Σ demand_k (now NON-uniform — demand varies with each source's
            // intensity). A whole cell whose claimants all demand 0 (all flameless
            // / choked) draws nothing — skip it (no writes, no telemetry).
            int64_t alloc[cd::SLOTS_MAX];
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
                int64_t keys[cd::SLOTS_MAX];
                int64_t sum_alloc = 0;
                for (int k = 0; k < n_cl; ++k) {
                    const int64_t num = (int64_t)o2j * dem[k];  // < 2^43
                    alloc[k] = num / D;      // floor, exact integer divide
                    keys[k]  = num % D;      // integer remainder = tiebreak key
                    sum_alloc += alloc[k];
                }
                // R leftover LSBs (provably in [0, n_cl)) go to the R claimants
                // with the largest key; ties -> lowest source index (a fixed,
                // bounded sub-LSB bias the isotropy test §6 tolerates at <=3
                // LSB). Order-free: the selection is a max over a fixed key +
                // a total tiebreak, so the claimant ENUMERATION order cannot
                // change the outcome — it generalizes to n_slots untouched.
                int64_t R = (int64_t)o2j - sum_alloc;
                bool chosen[cd::SLOTS_MAX];
                for (int k = 0; k < n_cl; ++k) chosen[k] = false;
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

            // --- Single-writer O2 DEBIT at cell j (design §3 step 3) ---------
            // P-O2b: this is now the ONLY gas write Pass A makes. The O2 is
            // booked where it was actually taken — the donor cell — so a sealed
            // room's total O2 inventory is exactly conserved and smothering
            // stays real (Erik's ships requirement). The combustion PRODUCTS
            // (soot, N2) and the HEAT are no longer deposited here: they belong
            // at the flame, and Pass B/Pass C put them there.
            O2[j] = (q16)((int64_t)o2j - burn_j);

            // Record each claimant's allocation on the slot buffer so Pass B can
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
                alloc_slot[(size_t)cl_slot[k] * n + j] = (q16)alloc[k];
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
    // Pass B — source cells. Single writer of wall_hp[i] AND of this source's
    // five deposit-site slots. Each flammable source sums its incoming slot
    // allocations and pays the stoichiometric fuel cost ONCE for the total,
    // floored ONCE at FUEL_FLOOR (design §3, "total-then-floor-once", critique
    // B). This never takes wall_hp below 1 LSB (the "smolder never destroys"
    // invariant — decisions #17): the floor is re-applied after the single
    // subtraction, so wall_hp[i] >= FUEL_FLOOR always. Structural destruction
    // stays exclusively FireSimulation's I>0 path.
    //
    // P-O2b adds the second job: decide WHERE this source's combustion products
    // land (see `dep_site` above and the split rule at the site).
    // ======================================================================
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (!flammable[i]) continue;
            // Gather every donor's allocation to THIS source. Donor cell for
            // slot s is j = i - OFF[s] (slot s at cell j points AT j + OFF[s]).
            // `direct[]` keeps the hop-1 part separately, per face direction:
            // slot s < 4 has OFF[s] == D4[s], so the donor sits on i's face in
            // direction D4_OPP[s].
            int64_t burn_i = 0;
            int64_t direct[4] = {0, 0, 0, 0};
            for (int s = 0; s < n_slots; ++s) {
                const int jy = y - cd::OFF_DY[s], jx = x - cd::OFF_DX[s];
                if (!in_bounds(jy, jx, h, w)) continue;
                const int j = jy * w + jx;
                const int64_t a = (int64_t)alloc_slot[(size_t)s * n + j];
                burn_i += a;
                if (s < 4) direct[cd::D4_OPP[s]] = a;
            }
            if (burn_i == 0) continue;   // this source drew no O2 this tick
            // round-to-nearest — the same unbiased-sink idiom fire_simulation's
            // wall_damage depletion uses. UNCHANGED by P-O2b: the source still
            // pays for exactly the O2 it consumed, wherever that O2 came from.
            const q16 fuel_cost = narrow_round(mul_wide(fuel_per_o2_q, (q16)burn_i));
            wall_hp[i] -= fuel_cost;
            if (wall_hp[i] < FUEL_FLOOR) wall_hp[i] = FUEL_FLOOR;

            // ---- P-O2b: WHERE THE PRODUCTS LAND (v5.2, Erik's ruling 4) ----
            // Deposit sites are the OPEN cells among {this tile} u {its four
            // faces} — "the fire's own tile + its open faces". The fire's own
            // tile counts only when it is itself an open, gas-holding cell,
            // i.e. furniture; a wood/door tile is solid and holds no gas.
            //
            // THE SPLIT RULE, chosen so that R = 1 is an EXACT identity:
            //   * what a face donated DIRECTLY (hop 1) is deposited back at
            //     THAT face — products of the O2 drawn there stay there;
            //   * the REMAINDER (everything drawn at hop >= 2, from cells that
            //     are not the fire's own faces) is split EVENLY across the open
            //     sites, exact integer divide, leftover LSBs to the LOWEST site
            //     index. Deterministic, order-free, conservative to the LSB.
            // At draw_r == 1 the remainder is identically ZERO and the own-tile
            // share is therefore ZERO too, so each face receives exactly the
            // allocation it made — which is precisely the shipped law's
            // per-air-cell aggregate. That is the whole R = 1 oracle, at this
            // site.
            int  site_cell[5];
            bool site_open[5];
            int  m = 0;
            for (int d = 0; d < 4; ++d) {
                site_open[d] = false; site_cell[d] = -1;
                const int sy = y + D4[d][0], sx = x + D4[d][1];
                if (!in_bounds(sy, sx, h, w)) continue;
                const int c = sy * w + sx;
                if (solid[c] || is_vacuum[c]) continue;
                site_open[d] = true; site_cell[d] = c; ++m;
            }
            // Slot 4 — the fire's OWN tile (open exactly when it is furniture).
            site_open[4] = (!solid[i] && !is_vacuum[i]);
            site_cell[4] = i;
            if (site_open[4]) ++m;

            const int64_t hop1 = direct[0] + direct[1] + direct[2] + direct[3];
            int64_t rem = burn_i - hop1;          // the hop >= 2 draws
            // m == 0 is unreachable when burn_i > 0 (every path starts at an
            // open face of this tile, so at least one open face exists), but a
            // lost remainder would be a silent leak, so it is guarded rather
            // than assumed.
            if (m == 0) rem = 0;
            const int64_t even = (m > 0) ? rem / m : 0;
            const int64_t extra = (m > 0) ? rem - even * m : 0;   // in [0, m)
            int taken = 0;
            for (int t = 0; t < 5; ++t) {
                if (!site_open[t]) continue;
                int64_t share = even + ((taken < (int)extra) ? 1 : 0);
                ++taken;
                if (t < 4) share += direct[t];
                dep_site[(size_t)t * n + i] = (q16)share;
            }
        }
    }

    // ======================================================================
    // Pass C — air cells, THE DEPOSIT (P-O2b, design v5.2 "deposits re-sited").
    // Single writer of SOOT[s], N2[s] and temperature[s]. Every open cell
    // gathers what the fires around it decided to deposit here: its four
    // face-neighbours' outbound shares plus, if this cell is itself a burning
    // furniture tile, its own share.
    //
    // This is the SECOND GATHER v5.2 asks for, "keyed like alloc_face" (i.e.
    // like the alloc_slot buffer that idiom became): the
    // deposit intent is written once by the source (Pass B) and read once by
    // the air cell (here), so single-writer discipline — and with it order
    // freedom and CPU/CUDA bit-identity — survives the re-siting intact.
    //
    // The arithmetic is the shipped deposit, verbatim, with `burn_j` replaced
    // by the gathered `burn_dep`. At draw_r == 1 the two are equal cell for
    // cell (see Pass B's split rule), and O2[s]/N2[s] hold exactly the values
    // the shipped in-line deposit read, so this pass is byte-identical there.
    // ======================================================================
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int s = row + x;
            if (solid[s] || is_vacuum[s]) continue;
            // Own-tile share (non-zero only for a burning furniture tile) plus
            // the four faces. Source i on face d of s filed its share toward s
            // under its OWN outbound direction to s, which is D4_OPP[d].
            int64_t burn_dep = (int64_t)dep_site[(size_t)4 * n + s];
            for (int d = 0; d < 4; ++d) {
                const int iy = y + D4[d][0], ix = x + D4[d][1];
                if (!in_bounds(iy, ix, h, w)) continue;
                const int i = iy * w + ix;
                burn_dep += (int64_t)dep_site[(size_t)D4_OPP[d] * n + i];
            }
            if (burn_dep == 0) continue;   // nothing burnt for this cell

            // Exact Dalton split (unchanged): soot + (burn-soot) == burn, so
            // the products' N_total is conserved to the LSB regardless of soot
            // rounding (#12). NOTE the honest consequence of the re-siting: the
            // O2 leaves the DONOR and the products appear at the FLAME, so
            // N_total is conserved GLOBALLY rather than per cell — which is the
            // physics (combustion products rise with the flame, and the deficit
            // they leave behind is what draws fresh air in), and is exactly
            // what makes the hot-air visuals follow the fire again.
            const q16 soot = narrow_round(mul_wide((q16)burn_dep, soot_yield_q));
            SOOT[s] += soot;
            N2[s]   += (q16)(burn_dep - (int64_t)soot);

            // ONE aggregate heat deposit against the POST-burn N_total (delta
            // delta) — same idiom/dials as TemperatureSolver's Pass-1 radiative
            // deposit; a per-source replay would reintroduce an order-dependent
            // denominator and defeat isotropy. Rail counters are PER-CELL
            // (design §3): no test may assert their absolute value.
            const q16 deposit = mul_q16((q16)burn_dep, H_fuel_q);   // burn*H_fuel
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
                                  && thermal_solid[s];
            if (object_site) {
                const int shift = heat_inv_shift[s];   // log2(thermal_mass), >= 0
                dT = deposit >> shift;
            } else {
                q16 n_total_s = (q16)((int64_t)O2[s] + (int64_t)N2[s]);
                if (n_total_s < n_floor_q) { n_total_s = n_floor_q; ++heat_floor_hits; }
                const q16 recip_n  = reciprocal_q16(n_total_s);
                const q16 e_over_n = mul_q16(deposit, recip_n);        // .../N
                dT                 = recip_mul(e_over_n, recip_cv);    // .../c_v
            }
            heat_saturating_add(&temperature[s], dT);
            if (temperature[s] > t_max_phys_q) {                   // v2.4 rail
                temperature[s] = t_max_phys_q; ++t_max_phys_hits;
            }
        }
    }
}
