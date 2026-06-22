#include "atmosphere_solver.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

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
    float* wave_p,
    float* wave_v,
    float* wave_source,
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
    float* wave_p,
    float* wave_v,
    float* wave_source,
    float* atmosphere,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    const float* wave_absorb,
    int h, int w,
    float dt
) const {
    const int n = h * w;
    const float c_sq = c * c;

    // --- 1. Feed wave_source into wave_p (rate-limited) ---
    for (int i = 0; i < n; ++i) {
        if (wave_source[i] > 0.001f) {
            float feed = wave_source[i] * feed_rate * dt;
            feed = std::min(feed, wave_source[i]);
            feed = std::min(feed, max_source_per_step);
            wave_p[i] += feed;
            wave_source[i] -= feed;
        }
    }

    // --- 2. Explicit wave kick: Laplacian of wave_p ---
    // Reused scratch (GPU-prep: no per-step alloc). Every lap[i] is written
    // below before the velocity update reads it, so no re-init needed.
    // `__restrict`: lap_ is solver-private and aliases none of the field
    // pointers — restores the fresh-local no-alias property /fp:fast relies on
    // for bit-identical codegen (a member ref/data() ptr without it desyncs).
    if (lap_.size() != (size_t)n) lap_.assign(n, 0.0f);
    float* __restrict lap = lap_.data();
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        const int row_up   = (y > 0)     ? (y - 1) * w : row;
        const int row_down = (y < h - 1) ? (y + 1) * w : row;

        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            const float p = wave_p[i];
            const float perm_i = permeability[i];

            // Face-permeability flux: face = min(perm[self], perm[n]); the
            // contribution is face*(field[n] - p). For perm∈{0,1} this is
            // bit-identical to the old obstacle mirror: face=0 (a unit/wall
            // neighbor, perm 0) → no flux, exactly like the mirror's p_n=p
            // zero term; face=1 (open neighbor) → field[n]-p, exactly like a
            // non-obstacle neighbor. Border vacuum is sealed (perm 0), breach
            // vacuum is open (perm 1) — waves propagate into it, as before.
            float lap_i = 0.0f;
            if (y > 0)     { const int n = row_up + x;   lap_i += std::min(perm_i, permeability[n]) * (wave_p[n] - p); }
            if (y < h - 1) { const int n = row_down + x; lap_i += std::min(perm_i, permeability[n]) * (wave_p[n] - p); }
            if (x > 0)     { const int n = row + x - 1;  lap_i += std::min(perm_i, permeability[n]) * (wave_p[n] - p); }
            if (x < w - 1) { const int n = row + x + 1;  lap_i += std::min(perm_i, permeability[n]) * (wave_p[n] - p); }

            lap[i] = lap_i;
        }
    }

    // Wave velocity update
    for (int i = 0; i < n; ++i) {
        wave_v[i] += (c_sq * lap[i] - damping * wave_v[i]) * dt;
    }

    // Wave pressure update
    for (int i = 0; i < n; ++i) {
        wave_p[i] += wave_v[i] * dt;
    }

    // --- 2b. Lossy boundary: per-cell wave-energy ABSORPTION (4a) ---
    // Pure energy removal at cells the wave already touches — strictly
    // stabilizing (damping out), so it needs no CFL/stability re-analysis and
    // does NOT touch the Laplacian or the global `damping`. Air has
    // wave_absorb=0 → k=1 → open-air wave behaviour is bit-identical to before.
    // A body (high wave_absorb) soaks the blast; lossy materials damp while
    // hull (low absorb) stays ringy. Scale by absorb_strength and dt; clamp
    // k∈[0,1] so a single overshooting step can't flip the sign.
    for (int i = 0; i < n; ++i) {
        const float a = wave_absorb[i] * absorb_strength * dt;
        if (a > 0.0f) {
            const float k = (a < 1.0f) ? (1.0f - a) : 0.0f;
            wave_v[i] *= k;
            wave_p[i] *= k;
        }
    }

    // Wave BCs: zero on walls and vacuum
    for (int i = 0; i < n; ++i) {
        if (is_wall[i] || is_vacuum[i] || obstacles[i]) {
            wave_p[i] = 0.0f;
            wave_v[i] = 0.0f;
        }
    }

    // --- 3. Transfer wave anomaly into atmosphere ---
    // Compute mean of wave_p (over non-obstacle tiles)
    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (!obstacles[i] && !is_wall[i] && !is_vacuum[i]) {
            sum += wave_p[i];
            count++;
        }
    }
    float mean_wp = (count > 0) ? sum / count : 0.0f;

    // Transfer anomaly (wave_p - mean) into atmosphere
    const float xfer = transfer * dt;
    for (int i = 0; i < n; ++i) {
        if (!obstacles[i] && !is_wall[i] && !is_vacuum[i]) {
            atmosphere[i] += (wave_p[i] - mean_wp) * xfer;
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
    float* wave_p,
    float* wave_v,
    float* wave_source,
    float* wind_x,
    float* wind_y,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    int h, int w,
    float dt
) const {
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

    for (int i = 0; i < n; ++i) {
        if (vac_dist[i] == 0) {
            // Vacuum: strong relaxation
            atmosphere[i] *= (1.0f - eta);
            wave_p[i] = 0.0f;
            wave_v[i] = 0.0f;
        } else if (obstacles[i] || is_wall[i]) {
            wave_p[i] = 0.0f;
            wave_v[i] = 0.0f;
            atmosphere[i] = 0.0f;
        } else if (vac_dist[i] == 1) {
            // Inner sponge
            atmosphere[i] *= (1.0f - eta * 0.5f);
            wave_v[i] *= (1.0f - std::min(30.0f * dt, 1.0f));
            wave_source[i] = 0.0f;
        } else if (vac_dist[i] == 2) {
            // Outer sponge
            atmosphere[i] *= (1.0f - eta * 0.25f);
            wave_v[i] *= (1.0f - std::min(15.0f * dt, 1.0f));
            wave_source[i] *= 0.5f;
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

            // Total pressure for gradient
            auto total = [&](int idx) { return atmosphere[idx] + wave_p[idx]; };
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
