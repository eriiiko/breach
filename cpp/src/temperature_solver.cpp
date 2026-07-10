// Temperature solver implementation — engine/06 §1 (heat -> temperature) +
// §2 (conduction relaxation). See temperature_solver.h for the determinism
// contract.

#include "temperature_solver.h"
#include "raycaster.h"     // HEAT_SCALE, heat_saturating_add (shared Q16.16 domain)
#include "fixed_point.h"   // S3c: quantize() for the o2_vacuum_thresh integer compare

// Direction order for the per-tile face_shift cache (MUST match the Python
// bake in GameMap: index 0=N, 1=S, 2=E, 3=W).
namespace {
    constexpr int DIR_N = 0;
    constexpr int DIR_S = 1;
    constexpr int DIR_E = 2;
    constexpr int DIR_W = 3;
    // Row/col offset per direction, same order.
    constexpr int DY[4] = { -1, +1,  0,  0 };
    constexpr int DX[4] = {  0,  0, +1, -1 };

    // ------------------------------------------------------------------
    // P2 — gas-T semi-Lagrangian advection helper (docs/eos_refactor_design.md
    // §4, §8 patch P2). Same PATTERN as smoke_dynamics.cpp's
    // `backtrace_sample_q` (the S2b SLint scheme: integer DDA wall-clip march +
    // integer bilinear sample + Newton-reciprocal renorm) — deliberately
    // NOT the same function (smoke_dynamics.cpp is untouched, out of P2's
    // scope), specialized to the temperature solver's own masks: no separate
    // obstacles/is_wall/permeability arrays are needed because `solid` already
    // IS the physics obstacle set (gamemap.py: obstacles == solid ==
    // permeability<=0), and gas-T has no partial-permeability weighting (a
    // partial-permeability tile like furniture is simply non-solid open-air,
    // per §2.2 of the design). A breach (is_vacuum && !solid) is a VENT
    // target, not a wall — the march may reach it and the bilinear sample
    // reads 0 there (heat drains with the venting gas), matching smoke's
    // is_breach carve-out exactly.
    using namespace fixedpoint;

    constexpr int32_t GAS_WSUM_FLOOR_Q = FP_ONE >> 8;    // mirrors smoke's WSUM_FLOOR_Q
    constexpr int32_t GAS_WSUM_EPS_Q   = FP_ONE >> 14;   // mirrors smoke's WSUM_EPS_Q

    inline bool gas_wall_at(int y, int x, const bool* solid, int h, int w) {
        if (y < 0 || y >= h || x < 0 || x >= w) return true;   // outside == wall
        return solid[y * w + x];
    }

    // Ported (pattern, not code) from smoke_dynamics.cpp::backtrace_sample_q.
    // See that function's header comment for the derivation of each piece
    // (DDA march / tile-center test / bilinear corner weights / Newton
    // renorm); this is the SAME arithmetic, re-typed against `solid`/
    // `is_vacuum` only.
    int32_t gas_backtrace_sample_q(
            const int32_t* src, int x, int y, int32_t bx_q, int32_t by_q,
            const bool* solid, const bool* is_vacuum, int h, int w) {
        int64_t px_q = ((int64_t)x << FP_SHIFT) + bx_q;
        int64_t py_q = ((int64_t)y << FP_SHIFT) + by_q;

        // ---- Wall-clip march (DDA, no sqrt) ----
        const int32_t abx = bx_q >= 0 ? bx_q : -bx_q;
        const int32_t aby = by_q >= 0 ? by_q : -by_q;
        const int32_t amax = abx >= aby ? abx : aby;
        int n_steps = amax >> FP_SHIFT;
        if (amax & (FP_ONE - 1)) n_steps += 1;               // ceil
        if (n_steps > 0) {
            auto floordiv = [](int32_t a, int b) -> int32_t {
                return (a >= 0) ? (a / b) : -(((-(int64_t)a) + b - 1) / b);
            };
            const int32_t sx_q = floordiv(bx_q, n_steps);
            const int32_t sy_q = floordiv(by_q, n_steps);
            int64_t cx_q = (int64_t)x << FP_SHIFT;
            int64_t cy_q = (int64_t)y << FP_SHIFT;
            for (int s = 0; s < n_steps; ++s) {
                const int64_t nxp_q = cx_q + sx_q;
                const int64_t nyp_q = cy_q + sy_q;
                const int ti = (int)((nxp_q + (FP_ONE >> 1)) >> FP_SHIFT);
                const int tj = (int)((nyp_q + (FP_ONE >> 1)) >> FP_SHIFT);
                if (gas_wall_at(tj, ti, solid, h, w)) break;
                cx_q = nxp_q;
                cy_q = nyp_q;
                if (tj >= 0 && tj < h && ti >= 0 && ti < w && is_vacuum[tj * w + ti])
                    break;                                    // reached the breach
            }
            px_q = cx_q;
            py_q = cy_q;
        }

        // ---- Clamp in-bounds (Q16.16) ----
        const int64_t hi_x = (int64_t)(w - 1) << FP_SHIFT;
        const int64_t hi_y = (int64_t)(h - 1) << FP_SHIFT;
        if (px_q < 0) px_q = 0; else if (px_q > hi_x) px_q = hi_x;
        if (py_q < 0) py_q = 0; else if (py_q > hi_y) py_q = hi_y;

        // ---- Integer bilinear sample ----
        const int x0 = (int)(px_q >> FP_SHIFT);
        const int y0 = (int)(py_q >> FP_SHIFT);
        const int x1 = (x0 + 1 <= w - 1) ? x0 + 1 : w - 1;
        const int y1 = (y0 + 1 <= h - 1) ? y0 + 1 : h - 1;
        const int32_t fx_q = (int32_t)(px_q - ((int64_t)x0 << FP_SHIFT));
        const int32_t fy_q = (int32_t)(py_q - ((int64_t)y0 << FP_SHIFT));
        const int32_t ifx_q = FP_ONE - fx_q;
        const int32_t ify_q = FP_ONE - fy_q;
        const int32_t w00 = mul_q16(ifx_q, ify_q);
        const int32_t w10 = mul_q16(fx_q,  ify_q);
        const int32_t w01 = mul_q16(ifx_q, fy_q);
        const int32_t w11 = mul_q16(fx_q,  fy_q);
        const int cyx[4][2] = { {y0, x0}, {y0, x1}, {y1, x0}, {y1, x1} };
        const int32_t cw[4] = { w00, w10, w01, w11 };

        int64_t acc = 0;
        int32_t wsum_q = 0;
        for (int k = 0; k < 4; ++k) {
            const int cy_ = cyx[k][0];
            const int cx_ = cyx[k][1];
            const int j = cy_ * w + cx_;
            if (solid[j]) continue;                            // sealed corner
            const int32_t val_q = is_vacuum[j] ? 0 : src[j];    // breach corner == 0
            acc += mul_wide(cw[k], val_q);
            wsum_q += cw[k];
        }
        if (wsum_q <= GAS_WSUM_EPS_Q) return src[y * w + x];    // negligible -> keep self

        const int32_t wsum_clamped = (wsum_q < GAS_WSUM_FLOOR_Q) ? GAS_WSUM_FLOOR_Q : wsum_q;
        const int32_t recip_q = reciprocal_q16(wsum_clamped);
        const int32_t acc_q = narrow(acc);
        return mul_q16(acc_q, recip_q);
    }
}

void TemperatureSolver::step(
    int32_t* temperature,
    const int32_t* heat,
    const int32_t* heat_inv_shift,
    const int32_t* face_shift,
    const bool* solid,
    const bool* is_vacuum,
    const int32_t* atmosphere,   // S3c: Q16.16 int32 (was float — the last float input)
    const int32_t* n_bulk,       // EOS P3: real bulk N_total (O2+N2 sum, Q16.16);
                                  // nullable — falls back to the P2 atmosphere
                                  // density-proxy (the direct-binding back-compat
                                  // path; the engine always passes the real sum)
    const int32_t* wind_x,       // P2: Q16.16 int32, may be null (Pass 0 skipped)
    const int32_t* wind_y,       // P2: Q16.16 int32, may be null (Pass 0 skipped)
    int h, int w,
    float dt                     // P2: tick's elapsed seconds; <= 0 skips Pass 0
) const {
    const int n = h * w;

    // ---- Pass 0: gas-T zero-at-vacuum + semi-Lagrangian advection (P2, §4) ----
    // Structural invariant FIRST, unconditional: an OPEN (non-solid) vacuum
    // cell — a true breach — holds no gas, so it holds no gas-T either —
    // "energy leaves with the gas" (design §4). This also defends a cell that
    // just transitioned into an open breach carrying stale T from before the
    // mask flipped. GUARD: `!solid[i]` is load-bearing here — a SOLID cell
    // that is ALSO flagged vacuum (the intact hull's own space-exposure flag;
    // gamemap.py: "an intact hull is vacuum AND solid") is NOT a breach, it is
    // a wall radiating to space via cool_shift_vacuum (Pass 3) — its T is
    // real solid-thermal-mass state and must survive across ticks. Without
    // this guard every space-facing hull tile would be wiped to 0 before Pass
    // 1 could deposit onto it, which is wrong (and was caught by the sealed-
    // room energy E2E's vacuum-exposed-hull scenario).
    for (int i = 0; i < n; ++i) {
        if (is_vacuum[i] && !solid[i]) temperature[i] = 0;
    }

    // Advection: skipped as a clean no-op when dt<=0 or wind is unavailable —
    // the Python direct-binding back-compat path (bindings.cpp) takes this
    // branch, so the shipped solid-only unit tests (test_temperature_*.py)
    // exercise EXACTLY the pre-P2 passes below, unchanged.
    if (dt > 0.0f && wind_x != nullptr && wind_y != nullptr) {
        using namespace fixedpoint;
        gas_scratch_.resize(n);
        int32_t* src = gas_scratch_.data();
        for (int i = 0; i < n; ++i) src[i] = temperature[i];

        // Same dt_adv = rate * dt convention as SmokeDynamics::step, so gas-T
        // rides the wind at the same visual scale smoke does (§9 TUNING DIAL;
        // P3 replaces `wind` with a real velocity and this rate goes away).
        const double dt_adv = (double)gas_advection_rate * (double)dt;
        const int32_t dt_adv_q = quantize(dt_adv);

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                if (solid[i] || is_vacuum[i]) continue;   // open-air mask only
                const int32_t bx_q = -mul_q16(wind_x[i], dt_adv_q);
                const int32_t by_q = -mul_q16(wind_y[i], dt_adv_q);
                temperature[i] = gas_backtrace_sample_q(
                    src, x, y, bx_q, by_q, solid, is_vacuum, h, w);
            }
        }
    }

    // ---- Pass 1: heat -> temperature conversion (proposal §1.2; P2 §4.3) ----
    // Solid tiles: UNCHANGED bit-shift path (bit-identical to pre-P2 — the
    // `if (deposit <= 0) continue` now runs before the solid branch instead of
    // after, but for a solid cell the net computation is identical: the OLD
    // code's `if (!solid[i]) continue` never touched a solid cell's arithmetic
    // either way). Open-air (non-vacuum) tiles: NEW radiation deposit
    // ΔT = ΔE / (N_total · c_v) — a per-tile dynamic-N reciprocal
    // (`reciprocal_q16`, the spike0b/S2c GS-Dinv class) composed with the
    // load-time-constant c_v reciprocal (`make_recip`/`recip_mul`, the
    // water_solver.cpp idiom), both precomputed/amortized ONCE per step (the
    // c_v recip is loop-invariant; the N recip is genuinely per-tile-per-tick,
    // so it CANNOT be a make_recip constant — that is exactly why the two
    // idioms are composed rather than either alone). `heat` is non-negative
    // (saturating accumulator), `atmosphere`/N is non-negative, c_v > 0 by
    // config contract -> every intermediate here is non-negative, so the
    // truncating narrows below cost no sign asymmetry (matches the solid
    // branch's own `>> shift` convention). Vacuum tiles: skipped (heat is
    // irrelevant there — Pass 0 already zeroed their T).
    {
        using namespace fixedpoint;
        const double c_v_safe = (c_v > 0.0f) ? (double)c_v : 1.0;
        const int64_t recip_cv = make_recip(c_v_safe);            // 1/c_v, once per step
        const int32_t n_floor_q = quantize((double)n_floor_heat); // independent floor (§4.3)

        for (int i = 0; i < n; ++i) {
            int32_t deposit = heat[i];
            if (deposit <= 0) continue;       // nothing to convert this tick
            if (solid[i]) {
                int shift = heat_inv_shift[i];    // log2(thermal_mass), >= 0
                int32_t gain = deposit >> shift;  // Q16.16 / 2^shift, still Q16.16
                heat_saturating_add(&temperature[i], gain);
            } else if (!is_vacuum[i]) {
                // EOS P3 (TODO closed): the divisor is the REAL bulk-species
                // N_total (O2+N2, passed by the engine) — the P2 atmosphere
                // density-proxy remains only as the nullable back-compat
                // fallback for the direct Python binding.
                int32_t N_q = n_bulk ? n_bulk[i] : atmosphere[i];
                if (N_q < n_floor_q) N_q = n_floor_q;    // floor independent of anything else (N_FLOOR_HEAT)
                const int32_t recip_N_q = reciprocal_q16(N_q);        // 1/N, per-tile Newton recip
                const int32_t e_over_n  = mul_q16(deposit, recip_N_q); // ΔE/N, Q16.16
                const int32_t dT = recip_mul(e_over_n, recip_cv);     // (ΔE/N)/c_v, Q16.16
                heat_saturating_add(&temperature[i], dT);
            }
        }
    }

    // DEBUG probe (temporary): T after Pass 1 (heat -> temperature convert).
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_heat = temperature[dbg_probe_idx];

    // ---- Pass 2: conduction relaxation (proposal §2.2) ----
    // Gather stencil, double-buffered so the whole pass reads the FROZEN
    // pre-conduction field and writes a fresh one (order-independent, no
    // scatter, no atomics). For tile i with 4 neighbours n ∈ {N,S,E,W}:
    //
    //     acc = Σ  (temp[n] - temp[i]) >> face_shift[i][dir]
    //     temp_new[i] = temp[i] + acc
    //
    // The DIFFERENCE is shifted, not the neighbour, so equal neighbours give
    // EXACTLY 0 (no drift) and the flux is conservative-shaped. A NO_FACE face
    // (grid edge or κ==0 either side) is skipped, so air (all NO_FACE) is a
    // structural no-op: Σr == 0 -> temp_new == temp, an air tile at 0 stays
    // bit-exactly 0. 64-bit accumulator avoids any intermediate overflow; the
    // final write fits int32 because the result is a convex combination of the
    // (already-int32) field values (§2.6 discrete maximum principle).
    scratch_.resize(n);
    int32_t* temp_new = scratch_.data();
    const int NO_FACE = no_face;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            const int32_t* fs = &face_shift[i * 4];  // [N,S,E,W] for this tile
            const int32_t ti = temperature[i];
            int64_t acc = 0;
            for (int d = 0; d < 4; ++d) {
                const int s = fs[d];
                if (s == NO_FACE) continue;          // grid edge or κ==0 -> no face
                const int ny = y + DY[d];
                const int nx = x + DX[d];
                // NO_FACE already marks grid edges, so neighbours are in-bounds;
                // guard anyway for robustness against a mis-baked cache.
                if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                const int32_t tn = temperature[ny * w + nx];
                // Signed Q16.16 difference; arithmetic right shift == ÷2^s
                // (rounds toward -inf, deterministic & identical cross-machine).
                acc += (int64_t)(tn - ti) >> s;
            }
            temp_new[i] = (int32_t)((int64_t)ti + acc);
        }
    }

    // Swap temp_new -> temperature (write the new field back in place; the
    // caller's buffer is the persistent one, scratch_ is reused next tick).
    for (int i = 0; i < n; ++i) temperature[i] = temp_new[i];
    // DEBUG probe (temporary): T after Pass 2 (conduction).
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_conduction = temperature[dbg_probe_idx];

    // ---- Pass 3: ambient cooling (proposal §3) ----
    // The LAST thermal pass (§3.5): runs AFTER conduction so this tick's fresh
    // deposit is spread across the metal BEFORE any of it is shed, and BEFORE
    // consumers so thresholds test the net post-loss temperature (the burn-out
    // mechanism). Temperature stores ΔT above ambient, so T_ambient == 0 and
    // cooling relaxes toward 0 with no subtraction:  T -= T >> shift.
    //
    // Vacuum-exposure (§3.3): a solid tile sheds 4× faster if ANY in-bounds
    // 4-neighbour is space-facing — `is_vacuum[n]` OR `atmosphere[n] <
    // o2_vacuum_thresh`. S3c: atmosphere is Q16.16 int32 now, so the threshold
    // compare is a pure INTEGER compare against `quantize(o2_vacuum_thresh)` —
    // this TU's LAST float input is gone (it is fully integer). This reuses the
    // SAME geometric N,S,E,W gather the
    // conduction pass walks (the four neighbour cells are already in hand),
    // independent of the conduction face_shift (a wall facing vacuum has a
    // NO_FACE conduction face there, but is still exposed for cooling). Ties to
    // the existing is_vacuum/atmosphere fields — no new field/buffer — so a
    // freshly-breached, now-space-facing wall flips to the fast shift instantly.
    //
    // Solid tiles only (air is already 0 and skipped, staying bit-exactly 0).
    // The signed arithmetic right shift is pinned to round toward 0 symmetrically
    // (`x<0 ? -((-x)>>s) : x>>s`) so it is deterministic / identical
    // cross-machine. The residual DEAD-BAND is intentional and preserved: the
    // last (1<<shift)-1 counts above ambient shift to 0 and never decay -> an
    // exact, jitter-free resting state at ambient (NO "+1 if nonzero" nudge).
    // Since the shifted magnitude is always <= |T|, a single isolated tile
    // relaxes toward 0 and never crosses below ambient.
    // Quantize the o2_vacuum_thresh config dial ONCE per step (round-to-nearest,
    // the load/boundary cast) — the exposure test is then a Q16.16 integer compare
    // against the int32 atmosphere field. No per-cell float.
    const int32_t thresh_q = fixedpoint::quantize((double)o2_vacuum_thresh);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            if (!solid[i]) continue;          // air / non-solid: already 0
            const int32_t t = temperature[i];
            if (t == 0) continue;             // exact rest: nothing to shed

            // Vacuum-exposure: same geometric 4-neighbour gather as conduction.
            bool exposed = false;
            for (int d = 0; d < 4; ++d) {
                const int ny = y + DY[d];
                const int nx = x + DX[d];
                if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                const int ni = ny * w + nx;
                if (is_vacuum[ni] || atmosphere[ni] < thresh_q) {  // Q16.16 int compare
                    exposed = true;
                    break;
                }
            }
            const int shift = exposed ? cool_shift_vacuum : cool_shift;

            // Signed arithmetic right shift, pinned to round toward 0 (portable,
            // deterministic). The dead-band (loss == 0 for |t| < (1<<shift))
            // gives an exact resting state at ambient.
            const int32_t loss = (t < 0) ? -((-t) >> shift) : (t >> shift);
            temperature[i] = t - loss;
        }
    }

    // DEBUG probe (temporary): T after Pass 3 (ambient cooling).
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_cooling = temperature[dbg_probe_idx];

    // STEP D (unit damage, §4) will add a further pass here, reading the
    // post-cool temperature field.
}
