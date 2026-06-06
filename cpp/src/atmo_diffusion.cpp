#include "atmo_diffusion.h"
#include <cmath>
#include <algorithm>
#include <vector>

void AtmoDiffusion::step(
    float* atmosphere,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
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
                const float perm_i = permeability[i];

                // Face-permeability flux: face = min(perm[self], perm[n]);
                // contribution += face*(field[n] - p). Bit-identical to the old
                // obstacle mirror for perm∈{0,1}.
                float lap_i = 0.0f;
                if (y > 0)     { const int nb = row_up + x;   lap_i += std::min(perm_i, permeability[nb]) * (atmosphere[nb] - p); }
                if (y < h - 1) { const int nb = row_down + x; lap_i += std::min(perm_i, permeability[nb]) * (atmosphere[nb] - p); }
                if (x > 0)     { const int nb = row + x - 1;  lap_i += std::min(perm_i, permeability[nb]) * (atmosphere[nb] - p); }
                if (x < w - 1) { const int nb = row + x + 1;  lap_i += std::min(perm_i, permeability[nb]) * (atmosphere[nb] - p); }

                lap[i] = lap_i;
            }
        }

        // --- Apply diffusion ---
        const float coeff = d_atm * actual_dt;
        for (int i = 0; i < h * w; ++i) {
            atmosphere[i] += coeff * lap[i];
        }

        // --- Zero walls/vacuum (no clamping — allow negative for rarefaction) ---
        for (int i = 0; i < h * w; ++i) {
            if (is_wall[i] || is_vacuum[i]) {
                atmosphere[i] = 0.0f;
            }
        }
    }
}
