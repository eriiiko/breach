#include "physics_engine.h"
#include <cmath>
#include <algorithm>

void PhysicsEngine::tick(
    float* atmosphere,
    float* wave_v,
    float* wave_source,
    float* wind_x,
    float* wind_y,
    float* smoke_field,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    int h, int w,
    float sim_time
) const {
    const float c_sq = wave.c * wave.c;
    const float dt_wave = 0.65f / wave.c;
    const int n_wave = std::max(1, static_cast<int>(std::ceil(sim_time / dt_wave)));
    const float actual_dt = sim_time / n_wave;

    // --- Interleaved wave + smoke substeps ---
    for (int step = 0; step < n_wave; ++step) {

        // 1. Feed wave_source into atmosphere
        for (int i = 0; i < h * w; ++i) {
            if (wave_source[i] > 0.001f) {
                float feed = wave_source[i] * wave.feed_rate * actual_dt;
                feed = std::min(feed, wave_source[i]);
                atmosphere[i] += feed;
                wave_source[i] -= feed;
            }
        }

        // 2. Compute Laplacian + update wave velocity
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            const int row_up   = (y > 0)     ? (y - 1) * w : row;
            const int row_down = (y < h - 1) ? (y + 1) * w : row;

            for (int x = 0; x < w; ++x) {
                const int i = row + x;

                float p_up    = obstacles[row_up + x]                       ? atmosphere[i] : atmosphere[row_up + x];
                float p_down  = obstacles[row_down + x]                     ? atmosphere[i] : atmosphere[row_down + x];
                float p_left  = (x > 0 && !obstacles[row + x - 1])         ? atmosphere[row + x - 1] : atmosphere[i];
                float p_right = (x < w - 1 && !obstacles[row + x + 1])     ? atmosphere[row + x + 1] : atmosphere[i];

                if (y == 0)     p_up   = atmosphere[i];
                if (y == h - 1) p_down = atmosphere[i];
                if (x == 0)     p_left = atmosphere[i];
                if (x == w - 1) p_right = atmosphere[i];

                float lap = p_up + p_down + p_left + p_right - 4.0f * atmosphere[i];
                wave_v[i] += (c_sq * lap - wave.damping * wave_v[i]) * actual_dt;
            }
        }

        // 3. Update pressure from velocity
        for (int i = 0; i < h * w; ++i) {
            atmosphere[i] += wave_v[i] * actual_dt;
        }

        // 4. Boundary conditions
        for (int i = 0; i < h * w; ++i) {
            if (is_vacuum[i]) {
                atmosphere[i] = 0.0f;
                wave_v[i] = 0.0f;
            } else if (obstacles[i]) {
                wave_v[i] = 0.0f;
            }
        }

        // 5. Compute wind field (gradient of atmosphere)
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;

                if (obstacles[i] || is_wall[i] || is_vacuum[i]) {
                    wind_x[i] = 0.0f;
                    wind_y[i] = 0.0f;
                    continue;
                }

                float p_left  = (x > 0     && !obstacles[row + x - 1])             ? atmosphere[row + x - 1] : atmosphere[i];
                float p_right = (x < w - 1 && !obstacles[row + x + 1])             ? atmosphere[row + x + 1] : atmosphere[i];
                int row_up    = (y > 0)     ? (y - 1) * w : row;
                int row_down  = (y < h - 1) ? (y + 1) * w : row;
                float p_up    = (!obstacles[row_up + x])                            ? atmosphere[row_up + x]  : atmosphere[i];
                float p_down  = (!obstacles[row_down + x])                          ? atmosphere[row_down + x]: atmosphere[i];

                wind_x[i] = (p_right - p_left) * 0.5f;
                wind_y[i] = (p_down  - p_up)   * 0.5f;
            }
        }

        // 6. Smoke advection by wind (interleaved — smoke rides the wave!)
        smoke.step(smoke_field, wind_x, wind_y, obstacles, is_wall, is_vacuum, h, w, actual_dt);
    }

    // --- Diffusion (runs at its own rate, after wave+smoke) ---
    diffusion.step(atmosphere, obstacles, is_wall, is_vacuum, h, w, sim_time);
}
