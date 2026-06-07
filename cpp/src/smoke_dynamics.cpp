#include "smoke_dynamics.h"
#include <algorithm>
#include <cmath>
#include <vector>

// Helper: face-permeability neighbour value with Neumann fallback.
// Returns f[self] + face*(f[neighbour] - f[self]) where
// face = min(perm[self], perm[neighbour]). For perm∈{0,1} this is
// bit-identical to the old obstacle mirror: a sealed neighbour (face=0,
// perm 0 == old obstacle) returns f[self] (the reflect); an open neighbour
// (face=1) returns f[neighbour]. Out-of-bounds also reflects (returns f[self]).
static inline float neighbor(const float* f, const float* perm, int y, int x,
                              int dy, int dx, int h, int w) {
    int self_i = y * w + x;
    int ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return f[self_i];
    int ni = ny * w + nx;
    float face = std::min(perm[self_i], perm[ni]);
    return f[self_i] + face * (f[ni] - f[self_i]);
}

// Helper: is the tile that contains the (clamped) sample point sealed to gas?
// Sealed == solid/wall/vacuum or zero permeability. Used by the semi-Lagrangian
// back-trace to stop the departure ray at the first wall it would cross, so a
// long back-trace step cannot tunnel through a one-cell-thick wall.
static inline bool sealed_at(int y, int x,
                             const bool* obstacles, const bool* is_wall,
                             const bool* is_vacuum, const float* perm,
                             int h, int w) {
    if (y < 0 || y >= h || x < 0 || x >= w) return true;  // outside == wall
    int i = y * w + x;
    return obstacles[i] || is_wall[i] || is_vacuum[i] || perm[i] <= 0.0f;
}

void SmokeDynamics::step(
    float* smoke,
    const float* wind_x,
    const float* wind_y,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    int h, int w,
    float dt
) const {
    const int n = h * w;
    const float actual_dt = dt * dt_scale;

    // --- Smoke diffusion (wind-dependent) ---
    // D_effective = d_smoke * (1 + wind_diffusion_scale * |wind|)
    // Higher wind = more turbulent mixing = smoke disperses faster.
    std::vector<float> lap(n);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            float s = smoke[i];
            float s_up    = neighbor(smoke, permeability, y, x, -1,  0, h, w);
            float s_down  = neighbor(smoke, permeability, y, x,  1,  0, h, w);
            float s_left  = neighbor(smoke, permeability, y, x,  0, -1, h, w);
            float s_right = neighbor(smoke, permeability, y, x,  0,  1, h, w);
            lap[i] = s_up + s_down + s_left + s_right - 4.0f * s;
        }
    }

    for (int i = 0; i < n; ++i) {
        float wind_sq = wind_x[i] * wind_x[i] + wind_y[i] * wind_y[i];
        float d_eff = d_smoke * (1.0f + wind_diffusion_scale * wind_sq);
        smoke[i] += d_eff * actual_dt * lap[i];
    }

    // --- Advection by precomputed wind field (semi-Lagrangian back-trace) ---
    // Unconditionally stable and checkerboard-free (the central-difference
    // stencil it replaces oscillated near breaches/explosions). This is also
    // the CUDA-ready algorithm (Stable-Fluids back-trace + texture bilerp).
    //
    // Back-trace distance preserves the *effective advection strength* of the
    // old central-difference term: that term integrated d(smoke)/dtau =
    // -wind . grad(smoke) with a tau-step of (advection_rate * actual_dt),
    // i.e. it advected smoke by the wind for that pseudo-time. The equivalent
    // semi-Lagrangian displacement is therefore wind * dt_adv with
    //   dt_adv = advection_rate * actual_dt
    // (actual_dt already folds in dt_scale), so the cloud moves at the same
    // per-substep speed. We do NOT add a CFL cap: back-trace is stable for any
    // displacement.
    const float dt_adv = advection_rate * actual_dt;

    // Double buffer: read from the post-diffusion snapshot, write the advected
    // result. Never overwrite mid-pass (a cell may be sampled by its neighbours).
    std::vector<float> src(smoke, smoke + n);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            // Skip impermeable / vacuum tiles: they hold no smoke and reading a
            // wind there is meaningless (handled by the final zeroing pass).
            if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

            // Back-trace the departure point.
            float bx = -wind_x[i] * dt_adv;   // displacement from the cell
            float by = -wind_y[i] * dt_adv;
            float px = static_cast<float>(x) + bx;
            float py = static_cast<float>(y) + by;

            // Wall-clip the back-trace ray. A long step can leap *over* a
            // one-cell-thick wall; the per-corner exclusion below only sees the
            // four cells around the landing point, not the cells the ray
            // crossed. So march from the cell toward the departure point in
            // sub-cell steps and stop just before the first sealed tile — smoke
            // is then pulled from the near side of the wall, never through it.
            {
                float dist = std::sqrt(bx * bx + by * by);
                int steps = static_cast<int>(std::ceil(dist));  // ~1 sample/cell
                if (steps > 0) {
                    float inv = 1.0f / static_cast<float>(steps);
                    float sx = bx * inv, sy = by * inv;
                    float cx = static_cast<float>(x);
                    float cy = static_cast<float>(y);
                    for (int t = 0; t < steps; ++t) {
                        float nxp = cx + sx, nyp = cy + sy;
                        int ti = static_cast<int>(std::floor(nxp + 0.5f));
                        int tj = static_cast<int>(std::floor(nyp + 0.5f));
                        if (sealed_at(tj, ti, obstacles, is_wall, is_vacuum,
                                      permeability, h, w)) {
                            break;          // stop at the last open point
                        }
                        cx = nxp; cy = nyp;
                    }
                    px = cx; py = cy;
                }
            }

            // Clamp the sample position in-bounds so we never read past the grid
            // (cell-centred sampling domain [0, w-1] x [0, h-1]).
            if (px < 0.0f)              px = 0.0f;
            else if (px > w - 1.0f)     px = static_cast<float>(w - 1);
            if (py < 0.0f)              py = 0.0f;
            else if (py > h - 1.0f)     py = static_cast<float>(h - 1);

            int x0 = static_cast<int>(std::floor(px));
            int y0 = static_cast<int>(std::floor(py));
            int x1 = std::min(x0 + 1, w - 1);
            int y1 = std::min(y0 + 1, h - 1);
            float fx = px - static_cast<float>(x0);
            float fy = py - static_cast<float>(y0);

            // Permeability-aware bilinear sample. Each of the four corners is
            // weighted by its bilinear weight, BUT a corner that is sealed
            // (solid/wall/vacuum or permeability<=0) is excluded so smoke is
            // never pulled out of, or teleported through, a wall. Weights of the
            // surviving corners are renormalised; if every corner is sealed we
            // fall back to the cell's own value (no transport this step).
            const int ci[4] = { y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1 };
            const float cw[4] = {
                (1.0f - fx) * (1.0f - fy),
                fx         * (1.0f - fy),
                (1.0f - fx) * fy,
                fx         * fy,
            };

            float acc = 0.0f;
            float wsum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                int j = ci[k];
                bool sealed = obstacles[j] || is_wall[j] || is_vacuum[j] ||
                              permeability[j] <= 0.0f;
                if (sealed) continue;
                acc  += cw[k] * src[j];
                wsum += cw[k];
            }

            smoke[i] = (wsum > 1e-6f) ? (acc / wsum) : src[i];
        }
    }

    // --- Clamp and zero walls/vacuum ---
    for (int i = 0; i < n; ++i) {
        if (is_wall[i] || is_vacuum[i]) {
            smoke[i] = 0.0f;
        } else {
            smoke[i] = std::clamp(smoke[i], 0.0f, 1.0f);
        }
    }
}
