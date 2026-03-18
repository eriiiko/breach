#include "smoke_dynamics.h"
#include <algorithm>
#include <cmath>
#include <vector>

// Helper: safe neighbor access with Neumann BC
static inline float neighbor(const float* f, const bool* obs, int y, int x,
                              int dy, int dx, int h, int w) {
    int ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return f[y * w + x];
    int ni = ny * w + nx;
    return obs[ni] ? f[y * w + x] : f[ni];
}

void SmokeDynamics::step(
    float* smoke,
    const float* wind_x,
    const float* wind_y,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    int h, int w,
    float dt
) const {
    const int n = h * w;

    // --- Smoke diffusion (Laplacian with Neumann BCs) ---
    std::vector<float> lap(n);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            float s = smoke[i];
            float s_up    = neighbor(smoke, obstacles, y, x, -1,  0, h, w);
            float s_down  = neighbor(smoke, obstacles, y, x,  1,  0, h, w);
            float s_left  = neighbor(smoke, obstacles, y, x,  0, -1, h, w);
            float s_right = neighbor(smoke, obstacles, y, x,  0,  1, h, w);
            lap[i] = s_up + s_down + s_left + s_right - 4.0f * s;
        }
    }

    for (int i = 0; i < n; ++i) {
        smoke[i] += d_smoke * dt * lap[i];
    }

    // --- Advection by precomputed wind field ---
    // Wind field u = (wind_x, wind_y) = gradient of atmosphere.
    // Advection: smoke += rate * dt * (u . grad_smoke)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;

            // Smoke gradient (central difference)
            float ds_dx = (neighbor(smoke, obstacles, y, x, 0,  1, h, w) -
                           neighbor(smoke, obstacles, y, x, 0, -1, h, w)) * 0.5f;
            float ds_dy = (neighbor(smoke, obstacles, y, x,  1, 0, h, w) -
                           neighbor(smoke, obstacles, y, x, -1, 0, h, w)) * 0.5f;

            // u . grad(smoke)
            smoke[i] += advection_rate * dt * (wind_x[i] * ds_dx + wind_y[i] * ds_dy);
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
