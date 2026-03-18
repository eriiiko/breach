#include "atmo_diffusion.h"
#include <cmath>
#include <algorithm>
#include <vector>

void AtmoDiffusion::step(
    float* atmosphere,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    int h, int w,
    float sim_time
) const {
    const float dt = 0.24f / std::max(d_atm, 0.01f);  // CFL-stable timestep
    const int n_steps = std::max(1, static_cast<int>(std::ceil(sim_time / dt)));
    const float actual_dt = sim_time / n_steps;

    // Temp buffer for the Laplacian result (avoids read-after-write issues)
    // 9000 floats = 36 KB, fits in L1
    std::vector<float> lap(h * w);

    for (int step = 0; step < n_steps; ++step) {

        // --- Compute Laplacian with Neumann BCs ---
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            const int row_up   = (y > 0)     ? (y - 1) * w : row;
            const int row_down = (y < h - 1) ? (y + 1) * w : row;

            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                const float p = atmosphere[i];

                float p_up    = (y > 0     && !obstacles[row_up + x])   ? atmosphere[row_up + x]   : p;
                float p_down  = (y < h - 1 && !obstacles[row_down + x]) ? atmosphere[row_down + x] : p;
                float p_left  = (x > 0     && !obstacles[row + x - 1])  ? atmosphere[row + x - 1]  : p;
                float p_right = (x < w - 1 && !obstacles[row + x + 1])  ? atmosphere[row + x + 1]  : p;

                lap[i] = p_up + p_down + p_left + p_right - 4.0f * p;
            }
        }

        // --- Apply diffusion ---
        const float coeff = d_atm * actual_dt;
        for (int i = 0; i < h * w; ++i) {
            atmosphere[i] += coeff * lap[i];
        }

        // --- Boundary conditions ---
        // Vacuum: Dirichlet BC (p=0). Walls: Neumann (don't touch).
        for (int i = 0; i < h * w; ++i) {
            if (is_vacuum[i]) {
                atmosphere[i] = 0.0f;
            }
        }
    }
}
