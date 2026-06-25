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
    float* atmosphere,
    float* wind_x,
    float* wind_y,
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
    float* atmosphere,      // float — FLOAT BRIDGE until S2c
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
    // wave_absorb / atmosphere stay FLOAT (atmosphere is the float bridge to
    // S2c; permeability/wave_absorb are not yet in the integer migration).
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
    // (the transfer deposit is done in float at the bridge — see §3 below; no
    //  xfer_q here, deliberately: a truncating integer deposit would DC-leak.)
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

    // Transfer anomaly (wave_p - mean_wp) into atmosphere — FLOAT BRIDGE until
    // S2c (atmosphere is still float). The INTEGER, order-free, sign-symmetric
    // mean_wp is the determinism-critical part (it is the global reduction —
    // identical on every peer). The deposit itself crosses into the FLOAT
    // atmosphere, so we do it the SAME WAY the float build did: dequantize the
    // integer anomaly and multiply by the real (transfer*dt) in float, then add.
    //
    // SUBTLE (the DC-leak guard, map §7.1 / plan §6.6): the per-cell anomaly
    // deposit must NOT use mul_q16 here. mul_q16's `>>16` truncates toward -inf,
    // so EVERY cell's deposit loses up to 1 LSB on the floor; summed over the
    // ~hundreds of interior cells that is a SYSTEMATIC negative DC sink (~ -count
    // counts/substep) into the conserved atmosphere — a percent-scale drain over
    // a tick even when Sum(wave_p - mean_wp) == 0. The rounded MEAN is DC-free,
    // but a TRUNCATING deposit re-introduces the bias. Doing the deposit in float
    // (round-to-nearest at the FP add) keeps it unbiased AND minimises divergence
    // from the float build across this bridge. When atmosphere goes integer (S2c)
    // this deposit must use a round-to-nearest (or conservative ±-pair) integer
    // multiply, NOT mul_q16's truncation — flagged here for S2c.
    const double xfer_d = (double)transfer * (double)dt;   // (transfer*dt) real
    const float  xfer_f = (float)xfer_d;
    for (int i = 0; i < n; ++i) {
        if (mask[i]) {
            const float anom = dequantize_f(wave_p[i] - mean_wp);  // FLOAT BRIDGE
            atmosphere[i] += anom * xfer_f;                        // unbiased FP add
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
    float* atmosphere,
    q16* wave_p,            // S2a: Q16.16 int32 (read for wind via FLOAT BRIDGE)
    q16* wave_v,            // S2a: Q16.16 int32 (zeroed/scaled in the sponge BC)
    q16* wave_source,       // S2a: Q16.16 int32
    float* wind_x,
    float* wind_y,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    int h, int w,
    float dt
) const {
    // S2c is NOT migrated yet: the atmosphere diffusion (RB-GS) + wind stay
    // FLOAT. S2a only made the WAVE fields integer, so diffuse_solve touches
    // them at two seams: (1) the sponge/vacuum BC zeroes/scales wave_p/wave_v/
    // wave_source (now integer ops — exact); (2) the wind reads wave_p as part of
    // the total pressure (DEQUANTIZED — a FLOAT BRIDGE until S2c). Everything
    // else here (the GS solve, the atmosphere sponge, the residual hook) is the
    // unchanged float S2c body.
    const int n = h * w;
    const float mu = d_atm * dt;  // implicit diffusion coefficient

    // --- 4. Implicit diffusion on atmosphere (Gauss-Seidel) ---
    // Solve: (I - mu * Δ) atm_new = atm_current
    // i.e.: (1 + 4μ) atm[i,j] - μ(neighbors) = rhs[i,j]
    // where rhs = current atmosphere values (the "u*" from IMEX).
    //
    // We iterate in-place: Gauss-Seidel naturally converges.
    // Red-black ordering for better convergence.
    if (mu > 1e-8f) {
        // Store RHS (current atmosphere = u*). Reused scratch: the copy loop
        // writes every rhs[i] before any read, so no re-init needed.
        if (rhs_.size() != (size_t)n) rhs_.assign(n, 0.0f);
        float* __restrict rhs = rhs_.data();
        for (int i = 0; i < n; ++i) rhs[i] = atmosphere[i];

        for (int iter = 0; iter < gs_iters; ++iter) {
            // Red-black Gauss-Seidel: two sweeps per iteration
            for (int color = 0; color < 2; ++color) {
                for (int y = 0; y < h; ++y) {
                    const int row = y * w;
                    for (int x = 0; x < w; ++x) {
                        if (((x + y) & 1) != color) continue;
                        const int i = row + x;

                        // Skip obstacles/walls (Neumann: don't update)
                        if (obstacles[i] || is_wall[i]) continue;
                        // Skip vacuum (handled by relaxation BC below)
                        if (is_vacuum[i]) continue;

                        // Gather neighbors with face-permeability weighting.
                        // face = min(perm[self], perm[neighbor]); the implicit
                        // operator is (I - mu*Σ face*(atm_n - atm_i)). For
                        // perm∈{0,1} this is bit-identical to the old Neumann
                        // mirror: an open neighbor (face=1) contributes mu*atm_n
                        // and 1 to the diagonal weight (== the old fixed 4mu with
                        // mirrored blocked terms cancelling), a blocked neighbor
                        // (face=0) contributes nothing — exactly the old reflect.
                        // Vacuum is NOT blocked here (perm 1) — air diffuses
                        // toward exposed breach vacuum, as before. (The sealed
                        // border is vacuum+wall, perm 0, which IS blocked.)
                        //
                        // CONSERVATION GUARD: a neighbour the solve EXCLUDES
                        // and hard-zeroes (wall/obstacle — skipped above,
                        // zeroed in the BC pass below) must contribute 0 to
                        // BOTH sums. Such a cell has no equation of its own,
                        // so a nonzero face there is a one-sided flux into a
                        // cell held at p=0 — a Dirichlet sink that DESTROYS
                        // mass. (Trigger: unit footprints stamped
                        // dyn_permeability=0.5 over solid DOOR tiles — doors
                        // are passable to movement but solid to flow — and the
                        // sealed ship drained at a few %/s.) Static walls
                        // carry perm 0, so for perm ∈ {0,1} inputs min() is
                        // already 0 and this guard changes nothing — the 3a
                        // bit-identical guarantee holds. Vacuum is NOT
                        // guarded: a breach face keeps its perm weight and
                        // drains via the relaxation BC, as designed.
                        const float perm_i = permeability[i];
                        auto face = [&](int nb) {
                            return (obstacles[nb] || is_wall[nb])
                                ? 0.0f
                                : std::min(perm_i, permeability[nb]);
                        };
                        float w_up    = (y > 0)   ? face((y-1)*w+x) : 0.0f;
                        float w_down  = (y < h-1) ? face((y+1)*w+x) : 0.0f;
                        float w_left  = (x > 0)   ? face(row+x-1)   : 0.0f;
                        float w_right = (x < w-1) ? face(row+x+1)   : 0.0f;

                        float nb = w_up   * (y > 0   ? atmosphere[(y-1)*w+x] : 0.0f)
                                 + w_down * (y < h-1 ? atmosphere[(y+1)*w+x] : 0.0f)
                                 + w_left * (x > 0   ? atmosphere[row+x-1]   : 0.0f)
                                 + w_right* (x < w-1 ? atmosphere[row+x+1]   : 0.0f);
                        float wsum = w_up + w_down + w_left + w_right;

                        atmosphere[i] = (rhs[i] + mu * nb) / (1.0f + mu * wsum);
                    }
                }
            }
        }

        // --- Patch 2a: GS-residual hook (read-only; AFTER the sweeps, BEFORE
        // the BC pass so the post-solve BC mutation can't contaminate it).
        // Residual of the implicit operator (I - μΔ)atm - rhs, evaluated with
        // the SAME face/neighbour gather the GS update used:
        //   res_i = (1 + μ·wsum)·atm_i - μ·nb - rhs_i
        // over the non-obstacle interior (the cells the solve actually updates;
        // walls/obstacles/vacuum are excluded — they have no equation). Linf
        // norm, normalized by max|atm| over the same cells. Diagnostic only —
        // nothing reads it; the diffusion is unconditionally stable regardless.
        // (At the big once-per-tick dt, μ is large; this measures whether
        // gs_iters sweeps under-relax — a tuning signal, not a correctness gate.)
        float res_max = 0.0f;
        float atm_absmax = 0.0f;
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;
                const float perm_i = permeability[i];
                auto face = [&](int nb) {
                    return (obstacles[nb] || is_wall[nb])
                        ? 0.0f
                        : std::min(perm_i, permeability[nb]);
                };
                float w_up    = (y > 0)   ? face((y-1)*w+x) : 0.0f;
                float w_down  = (y < h-1) ? face((y+1)*w+x) : 0.0f;
                float w_left  = (x > 0)   ? face(row+x-1)   : 0.0f;
                float w_right = (x < w-1) ? face(row+x+1)   : 0.0f;
                float nb = w_up   * (y > 0   ? atmosphere[(y-1)*w+x] : 0.0f)
                         + w_down * (y < h-1 ? atmosphere[(y+1)*w+x] : 0.0f)
                         + w_left * (x > 0   ? atmosphere[row+x-1]   : 0.0f)
                         + w_right* (x < w-1 ? atmosphere[row+x+1]   : 0.0f);
                float wsum = w_up + w_down + w_left + w_right;
                float res = (1.0f + mu * wsum) * atmosphere[i] - mu * nb - rhs[i];
                res_max = std::max(res_max, std::fabs(res));
                atm_absmax = std::max(atm_absmax, std::fabs(atmosphere[i]));
            }
        }
        last_gs_residual = (atm_absmax > 1e-12f) ? (res_max / atm_absmax) : res_max;
    } else {
        // No diffusion this step (μ ~ 0) — the operator is the identity, so the
        // residual is exactly 0.
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

    const float eta = std::min(breach_rate * dt, 1.0f);

    // S2a: the sponge's wave-field decays are now Q16.16. wave_v is SIGNED and
    // the factor is a shrink (<=1) -> scale_mag (magnitude-symmetric). wave_p /
    // wave_v zeroings are exact integer 0; wave_source (>= 0) uses mul_q16. The
    // scalar decay factors are folded once in double, then quantized (S1 idiom).
    const q16 wv_inner_k_q = quantize(1.0 - (double)std::min(30.0f * dt, 1.0f));
    const q16 wv_outer_k_q = quantize(1.0 - (double)std::min(15.0f * dt, 1.0f));
    const q16 ws_half_q    = quantize(0.5);   // wave_source outer-sponge *0.5

    for (int i = 0; i < n; ++i) {
        if (vac_dist[i] == 0) {
            // Vacuum: strong relaxation
            atmosphere[i] *= (1.0f - eta);
            wave_p[i] = 0;
            wave_v[i] = 0;
        } else if (obstacles[i] || is_wall[i]) {
            wave_p[i] = 0;
            wave_v[i] = 0;
            atmosphere[i] = 0.0f;
        } else if (vac_dist[i] == 1) {
            // Inner sponge
            atmosphere[i] *= (1.0f - eta * 0.5f);
            wave_v[i] = scale_mag(wave_v[i], wv_inner_k_q);
            wave_source[i] = 0;
        } else if (vac_dist[i] == 2) {
            // Outer sponge
            atmosphere[i] *= (1.0f - eta * 0.25f);
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
                wind_x[i] = 0.0f;
                wind_y[i] = 0.0f;
                continue;
            }

            // Total pressure for gradient = atmosphere (float) + wave_p.
            // S2a: wave_p is Q16.16 int32 -> DEQUANTIZE it here (FLOAT BRIDGE
            // until S2c, when atmosphere goes integer and this whole term is
            // re-expressed in Q16.16). The dequantize is exact (/65536).
            auto total = [&](int idx) {
                return atmosphere[idx] + dequantize_f(wave_p[idx]);  // FLOAT BRIDGE
            };
            float p_here = total(i);
            const float perm_i = permeability[i];

            // Face-permeability gradient: p_side = p_here + face*(total(n) -
            // p_here). For perm∈{0,1} this is bit-identical to the old mirror
            // (face=0 → p_here, exactly the reflect; face=1 → total(n)).
            // Indices clamp to self when out of bounds (face is 0 there, so
            // the term vanishes — but the read must stay in bounds).
            int il = (x > 0)   ? row + x - 1 : i;
            int ir = (x < w-1) ? row + x + 1 : i;
            int iu = (y > 0)   ? (y-1)*w + x : i;
            int id = (y < h-1) ? (y+1)*w + x : i;
            float f_left  = (x > 0)   ? std::min(perm_i, permeability[il]) : 0.0f;
            float f_right = (x < w-1) ? std::min(perm_i, permeability[ir]) : 0.0f;
            float f_up    = (y > 0)   ? std::min(perm_i, permeability[iu]) : 0.0f;
            float f_down  = (y < h-1) ? std::min(perm_i, permeability[id]) : 0.0f;

            float p_left  = p_here + f_left  * (total(il) - p_here);
            float p_right = p_here + f_right * (total(ir) - p_here);
            float p_up    = p_here + f_up    * (total(iu) - p_here);
            float p_down  = p_here + f_down  * (total(id) - p_here);

            // Wind = -grad(p): air flows from high to low pressure
            wind_x[i] = -(p_right - p_left) * 0.5f;
            wind_y[i] = -(p_down  - p_up)   * 0.5f;
        }
    }

    // Retain vac_dist's storage for the next step (swap idiom; no per-step alloc).
    vac_dist.swap(vac_dist_);
}
