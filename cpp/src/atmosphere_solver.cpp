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

            // Neumann BC: obstacle neighbors reflect.
            // Border vacuum is also obstacle → reflected (sealed).
            // Breach vacuum is NOT obstacle → waves propagate into it (correct).
            float p_up    = (y > 0     && !obstacles[row_up + x])   ? wave_p[row_up + x]   : p;
            float p_down  = (y < h - 1 && !obstacles[row_down + x]) ? wave_p[row_down + x] : p;
            float p_left  = (x > 0     && !obstacles[row + x - 1])  ? wave_p[row + x - 1]  : p;
            float p_right = (x < w - 1 && !obstacles[row + x + 1])  ? wave_p[row + x + 1]  : p;

            lap[i] = p_up + p_down + p_left + p_right - 4.0f * p;
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

        const float diag = 1.0f + 4.0f * mu;
        const float inv_diag = 1.0f / diag;

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

                        // Gather neighbors with Neumann BC
                        // Obstacles and walls reflect. Vacuum is NOT reflected here —
                        // air should diffuse toward exposed breach vacuum tiles.
                        // (The sealed border is vacuum+obstacle, which IS reflected.)
                        float a_up    = (y > 0     && !obstacles[(y-1)*w+x] && !is_wall[(y-1)*w+x])
                                        ? atmosphere[(y-1)*w+x] : atmosphere[i];
                        float a_down  = (y < h-1   && !obstacles[(y+1)*w+x] && !is_wall[(y+1)*w+x])
                                        ? atmosphere[(y+1)*w+x] : atmosphere[i];
                        float a_left  = (x > 0     && !obstacles[row+x-1]   && !is_wall[row+x-1])
                                        ? atmosphere[row+x-1]   : atmosphere[i];
                        float a_right = (x < w-1   && !obstacles[row+x+1]   && !is_wall[row+x+1])
                                        ? atmosphere[row+x+1]   : atmosphere[i];

                        atmosphere[i] = (rhs[i] + mu * (a_up + a_down + a_left + a_right)) * inv_diag;
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

            float p_left  = (x > 0     && !obstacles[row+x-1])  ? total(row+x-1) : p_here;
            float p_right = (x < w-1   && !obstacles[row+x+1])  ? total(row+x+1) : p_here;
            int rup       = (y > 0)     ? (y-1)*w : row;
            int rdn       = (y < h-1)   ? (y+1)*w : row;
            float p_up    = (!obstacles[rup+x])                  ? total(rup+x)   : p_here;
            float p_down  = (!obstacles[rdn+x])                  ? total(rdn+x)   : p_here;

            wind_x[i] = (p_right - p_left) * 0.5f;
            wind_y[i] = (p_down  - p_up)   * 0.5f;
        }
    }
}
