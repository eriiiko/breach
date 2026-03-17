#include "fire_simulation.h"
#include <cmath>
#include <algorithm>

// Neighbor offsets: 4-connected
static constexpr int D4[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
// 12 neighbors: 4-connected + diagonals + 2-tile range
static constexpr int D12[][2] = {
    {-1,0},{1,0},{0,-1},{0,1},
    {-2,0},{2,0},{0,-2},{0,2},
    {-1,-1},{-1,1},{1,-1},{1,1}
};

static inline bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

std::vector<std::pair<int, int>> FireSimulation::step(
    float* fire,
    float* atmosphere,
    float* smoke,
    float* wall_hp,
    const bool* is_wall,
    const bool* flammable,
    int h, int w,
    float dt
) const {
    const int n = h * w;
    const auto& p = params;

    // Early exit if no fire
    float max_fire = 0.0f;
    for (int i = 0; i < n; ++i) max_fire = std::max(max_fire, fire[i]);
    if (max_fire < 0.001f) return {};

    // --- Spread: burning tiles ignite neighboring flammable tiles ---
    // Accumulate neighbor fire intensity
    std::vector<float> nfire(n, 0.0f);
    for (const auto& d : D12) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int ny = y + d[0], nx = x + d[1];
                if (in_bounds(ny, nx, h, w)) {
                    nfire[y * w + x] += fire[ny * w + nx];
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        if (flammable[i] && fire[i] < 0.01f && nfire[i] > 0.1f) {
            fire[i] += p.spread_rate * dt * nfire[i];
        }
    }

    // --- Wind-biased spreading ---
    // Compute wind from atmosphere gradient
    std::vector<float> wind_x(n), wind_y(n);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            float a_right = (x < w-1) ? atmosphere[y*w+x+1] : atmosphere[i];
            float a_left  = (x > 0)   ? atmosphere[y*w+x-1] : atmosphere[i];
            float a_down  = (y < h-1) ? atmosphere[(y+1)*w+x] : atmosphere[i];
            float a_up    = (y > 0)   ? atmosphere[(y-1)*w+x] : atmosphere[i];
            wind_x[i] = -(a_right - a_left) * 0.5f;
            wind_y[i] = -(a_down  - a_up)   * 0.5f;
        }
    }

    // Bias ignition toward downwind direction
    for (const auto& d : D4) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int i = y * w + x;
                if (!flammable[i] || fire[i] >= 0.01f) continue;
                int ny = y + d[0], nx = x + d[1];
                if (!in_bounds(ny, nx, h, w)) continue;
                float nf_val = fire[ny * w + nx];
                if (nf_val < 0.05f) continue;
                float dot = d[1] * wind_x[i] + d[0] * wind_y[i];
                float boost = std::clamp(1.0f + 2.0f * dot, 0.0f, 3.0f);
                fire[i] += p.spread_rate * dt * nf_val * boost;
            }
        }
    }

    // --- Wind modulates fire intensity ---
    for (int i = 0; i < n; ++i) {
        if (fire[i] <= 0.01f) continue;
        float ws = std::sqrt(wind_x[i]*wind_x[i] + wind_y[i]*wind_y[i]);
        float threshold = p.k_wind_thresh * ws;
        float margin = fire[i] - threshold;
        float effect = p.k_wind_net * ws * margin;
        fire[i] += dt * effect;
    }
    for (int i = 0; i < n; ++i) if (fire[i] < 0.01f) fire[i] = 0.0f;

    // --- Burning tiles grow toward full intensity ---
    for (int i = 0; i < n; ++i) {
        if (fire[i] > 0.01f) fire[i] += 0.5f * dt;
    }

    // --- Fire only on flammable tiles ---
    for (int i = 0; i < n; ++i) {
        if (!flammable[i]) fire[i] = 0.0f;
    }

    // --- O2 check: average neighboring air atmosphere ---
    for (int i = 0; i < n; ++i) {
        if (fire[i] < 0.01f) continue;
        int y = i / w, x = i % w;
        float sum_atm = 0.0f;
        float count = 0.0f;
        for (const auto& d : D4) {
            int ny = y + d[0], nx = x + d[1];
            if (in_bounds(ny, nx, h, w) && !is_wall[ny * w + nx]) {
                sum_atm += atmosphere[ny * w + nx];
                count += 1.0f;
            }
        }
        if (count < 1.0f) count = 1.0f;
        if (sum_atm / count < p.o2_threshold) {
            fire[i] = 0.0f;
        }
    }

    // --- Fire consumes O2 from neighboring air tiles ---
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (fire[y * w + x] < 0.01f) continue;
            for (const auto& d : D4) {
                int ny = y + d[0], nx = x + d[1];
                if (in_bounds(ny, nx, h, w) && !is_wall[ny * w + nx]) {
                    atmosphere[ny * w + nx] -= p.o2_consumption * dt * fire[y * w + x];
                }
            }
        }
    }

    // --- Fire produces smoke in neighboring air tiles ---
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (fire[y * w + x] < 0.01f) continue;
            for (const auto& d : D4) {
                int ny = y + d[0], nx = x + d[1];
                if (in_bounds(ny, nx, h, w) && !is_wall[ny * w + nx]) {
                    smoke[ny * w + nx] += p.smoke_emission * dt * fire[y * w + x];
                }
            }
        }
    }

    // --- Fire damages walls, collect destroyed tiles ---
    std::vector<std::pair<int, int>> destroyed;
    for (int i = 0; i < n; ++i) {
        if (fire[i] > 0.01f) {
            wall_hp[i] -= p.wall_damage * dt * fire[i];
            if (wall_hp[i] <= 0.0f && flammable[i] && is_wall[i]) {
                destroyed.push_back({i / w, i % w});
                fire[i] = 0.0f;
            }
        }
    }

    // --- Final clamp (fire and smoke only — atmosphere is unclamped) ---
    for (int i = 0; i < n; ++i) {
        fire[i] = std::clamp(fire[i], 0.0f, 1.0f);
        smoke[i] = std::clamp(smoke[i], 0.0f, 1.0f);
    }

    return destroyed;
}
