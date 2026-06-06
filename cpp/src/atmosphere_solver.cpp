#include "atmosphere_solver.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

float AtmosphereSolver::max_dt() const {
    // Only wave CFL matters — diffusion is implicit (unconditionally stable).
    return 0.5f / std::max(c, 1e-6f);
}

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
    int h, int w,
    float dt
) const {
    const int n = h * w;
    const float c_sq = c * c;
    const float mu = d_atm * dt;  // implicit diffusion coefficient

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
    std::vector<float> lap(n);
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

    // --- 4. Implicit diffusion on atmosphere (Gauss-Seidel) ---
    // Solve: (I - mu * Δ) atm_new = atm_current
    // i.e.: (1 + 4μ) atm[i,j] - μ(neighbors) = rhs[i,j]
    // where rhs = current atmosphere values (the "u*" from IMEX).
    //
    // We iterate in-place: Gauss-Seidel naturally converges.
    // Red-black ordering for better convergence.
    if (mu > 1e-8f) {
        // Store RHS (current atmosphere = u*)
        std::vector<float> rhs(n);
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
                        const float perm_i = permeability[i];
                        float w_up    = (y > 0)   ? std::min(perm_i, permeability[(y-1)*w+x]) : 0.0f;
                        float w_down  = (y < h-1) ? std::min(perm_i, permeability[(y+1)*w+x]) : 0.0f;
                        float w_left  = (x > 0)   ? std::min(perm_i, permeability[row+x-1])   : 0.0f;
                        float w_right = (x < w-1) ? std::min(perm_i, permeability[row+x+1])   : 0.0f;

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
    }

    // --- 5. Boundary conditions ---
    // Precompute distance-to-EXPOSED-vacuum for 2-tile sponge layer.
    // Only vacuum tiles that are NOT obstacles count as seeds (breaches).
    // Border vacuum (which is also obstacle/wall) is blocked — the sponge
    // doesn't reach through hull walls to drain the sealed interior.
    std::vector<uint8_t> vac_dist(n, 255);
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
}
