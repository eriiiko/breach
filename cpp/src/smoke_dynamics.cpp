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

    // --- Advection by precomputed wind field ---
    // grad_p . grad_smoke formulation (matches prototype wind_test.py).
    // wind = -grad(p), so: smoke -= rate * dt * (wind . grad_smoke)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

            float ds_dx = (neighbor(smoke, permeability, y, x, 0,  1, h, w) -
                           neighbor(smoke, permeability, y, x, 0, -1, h, w)) * 0.5f;
            float ds_dy = (neighbor(smoke, permeability, y, x,  1, 0, h, w) -
                           neighbor(smoke, permeability, y, x, -1, 0, h, w)) * 0.5f;

            smoke[i] -= advection_rate * actual_dt * (wind_x[i] * ds_dx + wind_y[i] * ds_dy);
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
