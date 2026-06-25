#include "atmosphere_solver.h"
#include "fixed_point.h"   // S2a: Q16.16 toolkit (mul_q16, mul_wide, narrow,
                           //      scale_mag, mean_sum/mean_round, quantize)
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cstdint>

using namespace fixedpoint;

float AtmosphereSolver::max_dt() const {
    // Only wave CFL matters — diffusion is implicit (unconditionally stable).
    return 0.5f / std::max(c, 1e-6f);
}

// Patch 2a: step() = wave_substep() then diffuse_solve(). Kept as the
// single-substep convenience entry (and what the conservation test drives).
// The engine's run_substeps splits these — the wave loops at its CFL while the
// implicit diffusion runs ONCE per tick — so this fused form is no longer on
// the hot per-tick path, but it stays the reference single-step behaviour.
void AtmosphereSolver::step(
    q16* wave_p,            // S2a: Q16.16 int32
    q16* wave_v,            // S2a: Q16.16 int32
    q16* wave_source,       // S2a: Q16.16 int32
    q16* atmosphere,        // S2c: Q16.16 int32
    q16* wind_x,            // S2c: Q16.16 int32
    q16* wind_y,            // S2c: Q16.16 int32
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    const float* wave_absorb,
    int h, int w,
    float dt
) const {
    wave_substep(wave_p, wave_v, wave_source, atmosphere,
                 obstacles, is_wall, is_vacuum, permeability, wave_absorb,
                 h, w, dt);
    diffuse_solve(atmosphere, wave_p, wave_v, wave_source, wind_x, wind_y,
                  obstacles, is_wall, is_vacuum, permeability,
                  h, w, dt);
}

// --- Patch 2a: the explicit-wave sub-steps (1-3) ---------------------------
// Sub-steps 1 (feed source), 2 (explicit kick + per-cell absorb + wave BCs)
// and 3 (transfer wave anomaly -> atmosphere). Runs `n_wave` times at the wave
// CFL dt. No diffusion / BCs / wind here — those are diffuse_solve.
void AtmosphereSolver::wave_substep(
    q16* wave_p,            // S2a: Q16.16 int32 (acoustic anomaly, signed)
    q16* wave_v,            // S2a: Q16.16 int32 (wave velocity, signed)
    q16* wave_source,       // S2a: Q16.16 int32 (injected energy, >= 0)
    q16* atmosphere,        // S2c: Q16.16 int32 (conservative ±-pair transfer)
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    const float* wave_absorb,
    int h, int w,
    float dt
) const {
    // === S2a — the explicit wave system is now Q16.16 integer ================
    // wave_p / wave_v / wave_source are int32 Q16.16. The Q-S2-2 measurement
    // (tests/_s2a_wave_v_measure.py) confirmed wave_v stays in +/-32768 even
    // under a maximal blast (peak ~2674), so wave_v keeps Q16.16 (NO Q24.8
    // exception); but c_sq*lap reaches ~1.3e5 BEFORE *dt, so the velocity kick
    // carries c_sq*lap and -damping*wave_v in INT64 and applies *dt BEFORE the
    // narrow (the only safe order — map M5 / plan §6.5). permeability /
    // wave_absorb stay FLOAT (structural caches, not yet migrated — quantized
    // per-face on read). S2c: atmosphere is now Q16.16 int32, so the wave->
    // atmosphere transfer is a CONSERVATIVE INTEGER ±-pair (see §3 below), NOT
    // the old one-sided float deposit — exactly mass-neutral to the LSB.
    const int n = h * w;

    // --- Per-substep Q16.16 scalar constants (folded ONCE in double, then
    //     quantized — the S1 idiom: runtime scalars are real, only the FIELDS
    //     are integer). dt floats per substep (run_substeps narrows dt_actual
    //     to float at the call boundary), so these are per-call. ---
    const double dt_d   = (double)dt;
    const double c_d    = (double)c;
    const double c_sq_d = c_d * c_d;                 // 4356 at wave_c=66
    // The kick coefficients applied to the int64 intermediates:
    //   c_sq_dt  = c_sq * dt   (multiplies the Q16.16 lap, narrowed after)
    //   damp_dt  = damping*dt  (multiplies the Q16.16 wave_v, narrowed after)
    const q16 c_sq_dt_q = quantize(c_sq_d * dt_d);   // (c_sq*dt) as Q16.16
    const q16 damp_dt_q = quantize((double)damping * dt_d);
    const q16 dt_q      = quantize(dt_d);            // dt as Q16.16 (pressure update)
    // S2c: the wave->atmosphere transfer is now an INTEGER ±-pair (§3). xfer_q =
    // (transfer*dt) as Q16.16 — the per-cell deposit is round-to-nearest of
    // anomaly*xfer_q, then the SAME int is subtracted from wave_p and added to
    // atmosphere (exactly conservative — no truncating one-sided DC sink).
    const q16 xfer_q = quantize((double)transfer * dt_d);
    // Source feed: feed = min(source*feed_rate*dt, source, max_source_per_step).
    const q16 feed_rate_dt_q  = quantize((double)feed_rate * dt_d);     // feed_rate*dt
    const q16 max_source_q    = quantize((double)max_source_per_step);  // per-step cap
    const q16 source_thresh_q = quantize(0.001);     // the >0.001 feed gate
    // Per-cell absorb: a = wave_absorb*absorb_strength*dt; k = (a<1)?(1-a):0.
    const q16 absorb_str_dt_q = quantize((double)absorb_strength * dt_d);

    // --- 1. Feed wave_source into wave_p (rate-limited) ---
    // feed = source * (feed_rate*dt), then clamped to <= source and <= cap.
    // All Q16.16; integer min is exact. wave_source is non-negative.
    for (int i = 0; i < n; ++i) {
        if (wave_source[i] > source_thresh_q) {
            q16 feed = mul_q16(wave_source[i], feed_rate_dt_q);
            feed = std::min(feed, wave_source[i]);
            feed = std::min(feed, max_source_q);
            wave_p[i] += feed;
            wave_source[i] -= feed;
        }
    }

    // --- 2. Explicit wave kick: Laplacian of wave_p ---
    // The face-permeability flux is the gather-once shape (S1 §2): per face,
    // w*(p_n - p) as a WIDE int64 product (mul_wide of the Q16.16 face weight
    // and the Q16.16 difference), summed over 4 faces in int64, narrowed ONCE.
    // We store the NARROWED Q16.16 lap in the scratch (the kick re-widens it by
    // *c_sq*dt below). Reused int32 scratch (GPU-prep: no per-step alloc); every
    // lap[i] is written before read. The permeability face weight is FLOAT today
    // (permeability is not yet integer) — quantized per face into Q16.16 so the
    // product is integer; a FLOAT BRIDGE on the WEIGHT only (the field is int).
    if (lap_.size() != (size_t)n) lap_.assign(n, 0);
    q16* __restrict lap = lap_.data();
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        const int row_up   = (y > 0)     ? (y - 1) * w : row;
        const int row_down = (y < h - 1) ? (y + 1) * w : row;

        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            const q16 p = wave_p[i];
            const float perm_i = permeability[i];

            // Face flux gather in int64. quantize() of the FLOAT min-permeability
            // is the FLOAT BRIDGE on the weight (permeability stays float until a
            // later migration); the field difference (p_n - p) is exact integer.
            int64_t acc = 0;
            if (y > 0)     { const int nb = row_up + x;   acc += mul_wide(quantize((double)std::min(perm_i, permeability[nb])), wave_p[nb] - p); }
            if (y < h - 1) { const int nb = row_down + x; acc += mul_wide(quantize((double)std::min(perm_i, permeability[nb])), wave_p[nb] - p); }
            if (x > 0)     { const int nb = row + x - 1;  acc += mul_wide(quantize((double)std::min(perm_i, permeability[nb])), wave_p[nb] - p); }
            if (x < w - 1) { const int nb = row + x + 1;  acc += mul_wide(quantize((double)std::min(perm_i, permeability[nb])), wave_p[nb] - p); }

            lap[i] = narrow(acc);   // one shared truncation -> Q16.16 lap
        }
    }

    // Wave velocity update — THE OVERFLOW WATCH (map M5, Q-S2-2).
    //   wave_v += (c_sq*lap - damping*wave_v) * dt
    // c_sq*lap reaches ~1.3e5 (>32768) BEFORE *dt, so carry BOTH terms in int64
    // and apply *dt BEFORE narrowing: each int64 product is a Q(32).(32)-scaled
    // value (mul_wide of two Q16.16), summed, then narrowed ONCE. The *dt is
    // already folded into c_sq_dt_q / damp_dt_q, so the int64 sum is
    //   ( c_sq_dt * lap ) - ( damp_dt * wave_v )   in the wide scale,
    // narrowed to the Q16.16 delta. dt-before-narrow == the *dt is inside the
    // wide product, never on a pre-narrowed (already-overflowed) int32.
    for (int i = 0; i < n; ++i) {
        const int64_t kick_wide = mul_wide(c_sq_dt_q, lap[i])
                                - mul_wide(damp_dt_q, wave_v[i]);
        wave_v[i] += narrow(kick_wide);
    }

    // Wave pressure update:  wave_p += wave_v * dt   (Q16.16 multiply)
    for (int i = 0; i < n; ++i) {
        wave_p[i] += mul_q16(wave_v[i], dt_q);
    }

    // --- 2b. Lossy boundary: per-cell wave-energy ABSORPTION (4a) ---
    // a = wave_absorb*absorb_strength*dt (FLOAT BRIDGE on the absorb weight);
    // k = (a<1)?(1-a):0 in [0,1]. wave_v/wave_p are SIGNED, and k is a shrink
    // (<=1), so scale by k via scale_mag (magnitude-first, sign-symmetric — a
    // signed mul_q16's toward-(-inf) round would GROW a negative magnitude;
    // scale_mag can only shrink, the S1 absorb idiom).
    for (int i = 0; i < n; ++i) {
        const q16 a = mul_q16(quantize((double)wave_absorb[i]), absorb_str_dt_q);
        if (a > 0) {
            const q16 k = (a < FP_ONE) ? (q16)(FP_ONE - a) : 0;
            wave_v[i] = scale_mag(wave_v[i], k);
            wave_p[i] = scale_mag(wave_p[i], k);
        }
    }

    // Wave BCs: zero on walls and vacuum (exact integer 0).
    for (int i = 0; i < n; ++i) {
        if (is_wall[i] || is_vacuum[i] || obstacles[i]) {
            wave_p[i] = 0;
            wave_v[i] = 0;
        }
    }

    // --- 3. Transfer wave anomaly into atmosphere ---
    // mean_wp = ROUNDED integer mean of wave_p over the non-obstacle interior
    // (the #1 determinism hazard, map §7.1): int64 sum (order-free) then
    // round-half-away-from-zero (sign-symmetric — a biased mean is a DC leak
    // into the conserved atmosphere). The mask is !obstacle && !wall && !vacuum.
    // Reused mask scratch (GPU-prep: no per-step alloc). uint8_t storage (a
    // std::vector<bool> has no .data()); reinterpret to bool* for mean_sum.
    if (interior_mask_.size() != (size_t)n) interior_mask_.assign(n, 0);
    uint8_t* __restrict mask8 = interior_mask_.data();
    int count = 0;
    for (int i = 0; i < n; ++i) {
        const bool in = (!obstacles[i] && !is_wall[i] && !is_vacuum[i]);
        mask8[i] = in ? 1 : 0;
        if (in) count++;
    }
    const bool* mask = reinterpret_cast<const bool*>(mask8);
    const int64_t sum = mean_sum(wave_p, mask, n);
    const q16 mean_wp = mean_round(sum, count);

    // === S2c — the CONSERVATIVE INTEGER ±-pair transfer ======================
    // Move energy wave_p -> atmosphere by the anomaly (wave_p - mean_wp): per
    // interior cell, the transfer amount is gathered ONCE as a round-to-nearest
    // Q16.16 product, then the SAME integer is SUBTRACTED from wave_p and ADDED
    // to atmosphere. Because it is the identical int on both sides, (atmosphere +
    // wave_p) is preserved EXACTLY to the LSB — this is the bridge collapse the
    // whole S2 group exists for (P2 saturation). It REPLACES the S2a one-sided
    // float deposit, whose mul_q16-style >>16 truncated toward -inf and leaked a
    // systematic DC sink (~0.9%/200t — proven in the S2a review).
    //
    // The mean_wp keeps the deposit DC-free across the masked interior (Sum of
    // the rounded anomaly ~ 0), so the atmosphere does not gain a uniform DC
    // bias; the ±-pair makes whatever IS moved exactly conservative. Round-to-
    // nearest (not mul_q16 truncation) on the scalar multiply anomaly*xfer_q so
    // the per-cell deposit is unbiased (a truncating deposit re-introduces the
    // -inf sink the float bridge was avoiding).
    //
    // NOTE on the wave_p -= d coupling: the float build did NOT drain wave_p (it
    // relied on the zero-mean copy). Draining wave_p by exactly the deposited int
    // is the PHYSICAL transfer (the wave loses the energy it gives the bulk) and
    // is what makes the pair conservative; with transfer*dt small (~0.02/substep)
    // it is a gentle bleed, feel-validated against the float baseline.
    const int64_t HALF_Q = (int64_t)1 << (FP_SHIFT - 1);   // 0.5 ULP (round-nearest)
    for (int i = 0; i < n; ++i) {
        if (mask[i]) {
            const q16 anom = wave_p[i] - mean_wp;          // zero-mean (exact int)
            // d = round_to_nearest(anom * xfer_q) in Q16.16. Sign-symmetric +/-
            // 0.5 ULP so + and - deposits round identically (no DC bias).
            const int64_t prod = (int64_t)anom * (int64_t)xfer_q;
            const q16 d = (q16)((prod >= 0) ? ((prod + HALF_Q) >> FP_SHIFT)
                                            : -(((-prod) + HALF_Q) >> FP_SHIFT));
            atmosphere[i] += d;          // +flux into the conserved bulk
            wave_p[i]     -= d;          // -flux out of the wave (the same int)
        }
    }
}

// --- Patch 2a: the implicit diffusion + BCs + wind (4-7) -------------------
// Sub-steps 4 (u* = atmosphere), 5 (implicit Gauss-Seidel diffusion, μ = d_atm·
// dt), 6 (vacuum/sponge BCs), 7 (wind = -grad(atmosphere + wave_p)). Runs ONCE
// per tick at the FULL sim_time dt (so μ is large) — the implicit GS is
// unconditionally stable, so this is fine. The GS-residual hook measures the
// solve quality AFTER the sweeps but BEFORE the BC pass.
void AtmosphereSolver::diffuse_solve(
    q16* atmosphere,        // S2c: Q16.16 int32 (RB-GS, residual form, Dinv)
    q16* wave_p,            // S2a: Q16.16 int32 (read for wind — now integer)
    q16* wave_v,            // S2a: Q16.16 int32 (zeroed/scaled in the sponge BC)
    q16* wave_source,       // S2a: Q16.16 int32
    q16* wind_x,            // S2c: Q16.16 int32 (= -grad(atm+wave_p))
    q16* wind_y,            // S2c: Q16.16 int32
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    int h, int w,
    float dt
) const {
    // === S2c — the atmosphere diffusion (RB-GS) + wind are now Q16.16 integer.
    // This is the CLOSER of the S2 group: with atmosphere/wind integer there are
    // NO float bridges left inside the group (only the downstream FIRE bridge in
    // step_tail). The GS divide is replaced by the cached per-cell reciprocal
    // Dinv and run in RESIDUAL/FLUX form (a quotient form has no fixed point
    // under a truncating multiply at large μ·wsum — map §4); the wind reads the
    // integer wave_p directly (no dequantize); the sponge/vac BCs scale the
    // integer atmosphere with mul_q16 (a deliberate sink, by design).
    const int n = h * w;
    // μ = d_atm·dt as a Q16.16 scalar (folded once in double, then quantized —
    // the S1 idiom). d_atm=50, dt≈1/24 -> μ≈2.08, comfortably in range.
    const double mu_d = (double)d_atm * (double)dt;
    const q16    mu_q = quantize(mu_d);                 // μ in Q16.16
    const q16 MU_EPS_Q = 655;                           // ~0.01 in Q16.16 (μ>1e-8 gate)
    const int64_t HALF_Q = (int64_t)1 << (FP_SHIFT - 1);  // 0.5 ULP (round-nearest)

    // --- 4. Implicit diffusion on atmosphere (Red-Black Gauss-Seidel, integer)
    // Solve (I - μΔ) atm_new = atm_current in RESIDUAL form. Per updated cell:
    //   r_i = Σ_face μ·w_face·(atm_n - atm_i)  −  (atm_i - rhs_i)
    //   atm_i += mul( r_i, Dinv_i )            where Dinv_i = 1/(1 + μ·wsum_i)
    // At the fixed point equal neighbours -> r_i = 0 -> the increment truncates
    // to exactly 0 -> DRIFT-FREE (the quotient form leaks at μ·wsum≈large). The
    // face flux is gathered ONCE as an int64 (mul_wide of the Q16.16 face weight
    // μ·w and the Q16.16 difference), summed, narrowed once — the conservative
    // S1 gather shape. The 2-color schedule (red reads only black) is order-
    // independent by construction -> identical on any architecture.
    if (mu_q > MU_EPS_Q) {
        // Store RHS (current atmosphere = u*). Reused scratch (Q16.16 now).
        if (rhs_.size() != (size_t)n) rhs_.assign(n, 0);
        q16* __restrict rhs = rhs_.data();
        for (int i = 0; i < n; ++i) rhs[i] = atmosphere[i];

        // --- Build/refresh the cached per-cell Dinv on CHANGED cells only -----
        // Dinv_i = reciprocal_q16(quantize(1 + μ·wsum_real)), keyed on
        // (μ_q | obstacles | is_wall | is_vacuum | the 4 face permeabilities).
        // Most ticks nothing changes -> we rebuild NOTHING (the key matches and
        // we `continue`). Permeability is the CONTINUOUS dyn field, so wsum is a
        // genuine per-cell value (not just {0,½,1} sums) — a per-cell reciprocal
        // is needed, exactly the locked Q-S2-3 design. The CPU reciprocal_q16 is
        // PURE INTEGER (Newton) — the precompute keeps NO float (the locked
        // CPU=double-reciprocal option (a) was superseded by the proven integer
        // Newton helper, which is GPU-clean and bit-identical on every peer).
        if (dinv_.size() != (size_t)n || !dinv_valid_) {
            dinv_.assign(n, 0);
            dinv_key_.assign(n, 0);
            dinv_valid_ = true;
            // Force a full rebuild: an impossible key so every cell mismatches.
            for (int i = 0; i < n; ++i) dinv_key_[i] = (uint64_t)-1;
        }
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (obstacles[i] || is_wall[i] || is_vacuum[i]) {
                    // Excluded cell: no equation. Mark Dinv 0 + a sentinel key so
                    // if it later becomes interior it rebuilds. (The GS skips it.)
                    if (dinv_key_[i] != 0) { dinv_[i] = 0; dinv_key_[i] = 0; }
                    continue;
                }
                // The 4 face permeabilities (quantized; a face into an excluded+
                // zeroed neighbour is 0 — the conservation guard, see below).
                const float perm_i = permeability[i];
                auto face_q = [&](int nb, bool inb) -> q16 {
                    if (!inb) return 0;
                    if (obstacles[nb] || is_wall[nb]) return 0;   // Neumann/guard
                    return quantize((double)std::min(perm_i, permeability[nb]));
                };
                const q16 wu = face_q((y-1)*w+x, y > 0);
                const q16 wd = face_q((y+1)*w+x, y < h-1);
                const q16 wl = face_q(row+x-1,   x > 0);
                const q16 wr = face_q(row+x+1,   x < w-1);
                // Key: a cheap 64-bit mix of μ_q, the masks (always 0 here since
                // interior), and the 4 face weights. Any input change flips it.
                uint64_t key = (uint64_t)(uint32_t)mu_q;
                key = key * 1099511628211ull + (uint64_t)(uint32_t)wu;
                key = key * 1099511628211ull + (uint64_t)(uint32_t)wd;
                key = key * 1099511628211ull + (uint64_t)(uint32_t)wl;
                key = key * 1099511628211ull + (uint64_t)(uint32_t)wr;
                key |= 1;   // never 0 (0 is the excluded-cell sentinel)
                if (key == dinv_key_[i]) continue;    // unchanged -> reuse cache
                // wsum = wu+wd+wl+wr (Q16.16); denom = 1 + μ·wsum (Q16.16).
                const q16 wsum = wu + wd + wl + wr;
                const q16 mu_wsum = mul_q16(mu_q, wsum);
                const q16 denom_q = FP_ONE + mu_wsum;          // >= FP_ONE
                dinv_[i] = reciprocal_q16(denom_q);            // 1/(1+μ·wsum), Q16.16
                dinv_key_[i] = key;
            }
        }

        for (int iter = 0; iter < gs_iters; ++iter) {
            for (int color = 0; color < 2; ++color) {
                for (int y = 0; y < h; ++y) {
                    const int row = y * w;
                    for (int x = 0; x < w; ++x) {
                        if (((x + y) & 1) != color) continue;
                        const int i = row + x;
                        if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

                        const float perm_i = permeability[i];
                        // μ·w_face as a Q16.16 weight; a face into an excluded+
                        // zeroed cell is 0 (the CONSERVATION GUARD — a one-sided
                        // flux into a p=0 Dirichlet cell would destroy mass; the
                        // S1/3a fix, here integer). Gather Σ μ·w·(atm_n − atm_i)
                        // as an int64 (mul_wide), narrow once.
                        const q16 ai = atmosphere[i];
                        auto muw = [&](int nb, bool inb) -> q16 {
                            if (!inb) return 0;
                            if (obstacles[nb] || is_wall[nb]) return 0;
                            const q16 fw = quantize((double)std::min(perm_i, permeability[nb]));
                            return mul_q16(mu_q, fw);
                        };
                        int64_t acc = 0;
                        if (y > 0)   { const int nb=(y-1)*w+x; acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                        if (y < h-1) { const int nb=(y+1)*w+x; acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                        if (x > 0)   { const int nb=row+x-1;   acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                        if (x < w-1) { const int nb=row+x+1;   acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                        // residual = Σ μ·w·(atm_n−atm_i) − (atm_i − rhs_i)
                        const q16 flux = narrow(acc);
                        const q16 resi = flux - (ai - rhs[i]);
                        // increment = resi·Dinv, ROUND-TO-NEAREST (not mul_q16's
                        // toward-(-inf) truncation). A truncating increment biases
                        // EVERY cell's relaxation by up to -1 LSB/sweep -> a
                        // systematic DC sink during the diffusion transient (a
                        // ~1.8% mass loss vs the float GS's ~0.002%). Round-to-
                        // nearest removes the directional bias so the integer GS
                        // conserves to ~float granularity, while the residual form
                        // keeps a drift-free fixed point (resi small -> rounds to 0).
                        const int64_t inc_wide = (int64_t)resi * (int64_t)dinv_[i];
                        const q16 inc = (q16)((inc_wide >= 0)
                            ? ((inc_wide + HALF_Q) >> FP_SHIFT)
                            : -(((-inc_wide) + HALF_Q) >> FP_SHIFT));
                        atmosphere[i] = ai + inc;
                    }
                }
            }
        }

        // --- GS-residual hook (read-only; AFTER the sweeps, BEFORE the BC pass).
        // Mirror the float residual in INTEGER (plan §4 — the convergence check is
        // itself deterministic): res_i = (atm_i − rhs_i) − Σ μ·w·(atm_n − atm_i),
        // the same operator the update drives to 0. Linf over the interior,
        // normalized by max|atm|, dequantized to a float ratio for the Python
        // readout. Diagnostic only — nothing in-sim reads it.
        int64_t res_max_q = 0;
        int64_t atm_absmax_q = 0;
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;
                const float perm_i = permeability[i];
                const q16 ai = atmosphere[i];
                auto muw = [&](int nb, bool inb) -> q16 {
                    if (!inb) return 0;
                    if (obstacles[nb] || is_wall[nb]) return 0;
                    const q16 fw = quantize((double)std::min(perm_i, permeability[nb]));
                    return mul_q16(mu_q, fw);
                };
                int64_t acc = 0;
                if (y > 0)   { const int nb=(y-1)*w+x; acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                if (y < h-1) { const int nb=(y+1)*w+x; acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                if (x > 0)   { const int nb=row+x-1;   acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                if (x < w-1) { const int nb=row+x+1;   acc += mul_wide(muw(nb,true), atmosphere[nb]-ai); }
                const q16 flux = narrow(acc);
                const q16 res = (ai - rhs[i]) - flux;
                const int64_t ar = res < 0 ? -res : res;
                if (ar > res_max_q) res_max_q = ar;
                const int64_t aa = ai < 0 ? -ai : ai;
                if (aa > atm_absmax_q) atm_absmax_q = aa;
            }
        }
        last_gs_residual = (atm_absmax_q > 0)
            ? (float)((double)res_max_q / (double)atm_absmax_q)
            : (float)((double)res_max_q / (double)FP_ONE);
    } else {
        // No diffusion this step (μ ~ 0) — the operator is the identity.
        last_gs_residual = 0.0f;
    }

    // --- 5. Boundary conditions ---
    // Precompute distance-to-EXPOSED-vacuum for 2-tile sponge layer.
    // Only vacuum tiles that are NOT obstacles count as seeds (breaches).
    // Border vacuum (which is also obstacle/wall) is blocked — the sponge
    // doesn't reach through hull walls to drain the sealed interior.
    // Reused scratch via the SWAP idiom: `vac_dist` stays a genuine local
    // std::vector (so the float BC pass that READS it below keeps the exact
    // fresh-local codegen /fp:fast emitted before — a member pointer/ref defeats
    // the optimizer's no-alias view of that loop and shifts the float rounding),
    // while its allocation is RETAINED across steps in vac_dist_ (no per-step
    // alloc — the GPU-prep goal). Re-fill to 255 each step: the default 255 IS
    // read (cells that never reach 0/1/2 stay 255 and fall through the final
    // pass). NB: a member __restrict pointer would ALSO miscompile here — the
    // BFS reads neighbours an earlier iteration wrote (self-aliasing).
    std::vector<uint8_t> vac_dist;
    vac_dist.swap(vac_dist_);          // steal retained storage (capacity kept)
    vac_dist.assign(n, 255);           // size to n + re-init to 255 (no realloc if cap>=n)
    for (int i = 0; i < n; ++i) {
        if (is_vacuum[i] && !obstacles[i] && !is_wall[i]) vac_dist[i] = 0;
    }
    // Pass 1: dist=1 (only propagate to non-obstacle air tiles)
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (vac_dist[i] == 0 || obstacles[i] || is_wall[i]) continue;
            bool adj = false;
            if (y > 0     && vac_dist[(y-1)*w+x] == 0) adj = true;
            if (y < h-1   && vac_dist[(y+1)*w+x] == 0) adj = true;
            if (x > 0     && vac_dist[row+x-1]   == 0) adj = true;
            if (x < w-1   && vac_dist[row+x+1]   == 0) adj = true;
            if (adj) vac_dist[i] = 1;
        }
    }
    // Pass 2: dist=2
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (vac_dist[i] <= 1 || obstacles[i] || is_wall[i]) continue;
            bool adj = false;
            if (y > 0     && vac_dist[(y-1)*w+x] == 1) adj = true;
            if (y < h-1   && vac_dist[(y+1)*w+x] == 1) adj = true;
            if (x > 0     && vac_dist[row+x-1]   == 1) adj = true;
            if (x < w-1   && vac_dist[row+x+1]   == 1) adj = true;
            if (adj) vac_dist[i] = 2;
        }
    }

    const double eta_d = (double)std::min(breach_rate * dt, 1.0f);

    // S2c: the sponge/vacuum BCs are now fully Q16.16. The atmosphere relaxation
    // factors (1−eta), (1−eta·0.5), (1−eta·0.25) are folded once in double and
    // quantized; the atmosphere *= factor becomes mul_q16 (atmosphere is non-
    // negative here, so the >>16 truncation is a magnitude shrink — this is a
    // DELIBERATE SINK, air venting to space, NOT conserved by design, like the
    // temperature cooling). wave fields: wave_v shrinks via scale_mag (signed);
    // wave_p/wave_v zeroings + wave_source clamps are exact integer ops (S2a).
    const q16 atm_vac_k_q   = quantize(1.0 - eta_d);          // vacuum: (1−eta)
    const q16 atm_inner_k_q = quantize(1.0 - eta_d * 0.5);    // inner: (1−eta·0.5)
    const q16 atm_outer_k_q = quantize(1.0 - eta_d * 0.25);   // outer: (1−eta·0.25)
    const q16 wv_inner_k_q = quantize(1.0 - (double)std::min(30.0f * dt, 1.0f));
    const q16 wv_outer_k_q = quantize(1.0 - (double)std::min(15.0f * dt, 1.0f));
    const q16 ws_half_q    = quantize(0.5);   // wave_source outer-sponge *0.5

    for (int i = 0; i < n; ++i) {
        if (vac_dist[i] == 0) {
            // Vacuum: strong relaxation (deliberate sink)
            atmosphere[i] = mul_q16(atmosphere[i], atm_vac_k_q);
            wave_p[i] = 0;
            wave_v[i] = 0;
        } else if (obstacles[i] || is_wall[i]) {
            wave_p[i] = 0;
            wave_v[i] = 0;
            atmosphere[i] = 0;
        } else if (vac_dist[i] == 1) {
            // Inner sponge
            atmosphere[i] = mul_q16(atmosphere[i], atm_inner_k_q);
            wave_v[i] = scale_mag(wave_v[i], wv_inner_k_q);
            wave_source[i] = 0;
        } else if (vac_dist[i] == 2) {
            // Outer sponge
            atmosphere[i] = mul_q16(atmosphere[i], atm_outer_k_q);
            wave_v[i] = scale_mag(wave_v[i], wv_outer_k_q);
            wave_source[i] = mul_q16(wave_source[i], ws_half_q);
        }
    }

    // --- 6. Wind = gradient of total pressure (atmosphere + wave_p) ---
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;

            if (obstacles[i] || is_wall[i] || is_vacuum[i]) {
                wind_x[i] = 0;
                wind_y[i] = 0;
                continue;
            }

            // Total pressure for the gradient = atmosphere + wave_p, both Q16.16
            // int32 now (S2c) — the read of wave_p is INTEGER (no dequantize; the
            // last wave_p FLOAT BRIDGE in the wind term is collapsed). total() is
            // an exact integer sum.
            auto total = [&](int idx) -> q16 {
                return atmosphere[idx] + wave_p[idx];
            };
            const q16 p_here = total(i);
            const float perm_i = permeability[i];

            // Face-permeability gradient: p_side = p_here + face·(total(n)−p_here).
            // face is the quantized min-permeability (a load-boundary cast of the
            // still-float perm field — a perm bridge, NOT an atmosphere bridge).
            // For perm∈{0,1} this is bit-identical to the old mirror (face=0 ->
            // p_here reflect; face=FP_ONE -> total(n)). The face·Δ is mul_q16.
            int il = (x > 0)   ? row + x - 1 : i;
            int ir = (x < w-1) ? row + x + 1 : i;
            int iu = (y > 0)   ? (y-1)*w + x : i;
            int id = (y < h-1) ? (y+1)*w + x : i;
            const q16 f_left  = (x > 0)   ? quantize((double)std::min(perm_i, permeability[il])) : 0;
            const q16 f_right = (x < w-1) ? quantize((double)std::min(perm_i, permeability[ir])) : 0;
            const q16 f_up    = (y > 0)   ? quantize((double)std::min(perm_i, permeability[iu])) : 0;
            const q16 f_down  = (y < h-1) ? quantize((double)std::min(perm_i, permeability[id])) : 0;

            const q16 p_left  = p_here + mul_q16(f_left,  total(il) - p_here);
            const q16 p_right = p_here + mul_q16(f_right, total(ir) - p_here);
            const q16 p_up    = p_here + mul_q16(f_up,    total(iu) - p_here);
            const q16 p_down  = p_here + mul_q16(f_down,  total(id) - p_here);

            // Wind = -grad(p): air flows from high to low pressure. The ·0.5 is a
            // shr_round0 (>>1, sign-symmetric — wind is signed). 2-term central
            // diff, integer throughout.
            wind_x[i] = -shr_round0(p_right - p_left, 1);
            wind_y[i] = -shr_round0(p_down  - p_up,   1);
        }
    }

    // Retain vac_dist's storage for the next step (swap idiom; no per-step alloc).
    vac_dist.swap(vac_dist_);
}
