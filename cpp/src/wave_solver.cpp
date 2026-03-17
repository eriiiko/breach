#include "wave_solver.h"
#include <cmath>
#include <algorithm>

void WaveSolver::step(
    float* wave_p,
    float* wave_v,
    float* wave_source,
    float* atmosphere,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    int h, int w,
    float sim_time
) const {
    const float c_sq = c * c;
    const float dt = 0.65f / c;           // CFL-stable timestep
    const int n_steps = std::max(1, static_cast<int>(std::ceil(sim_time / dt)));
    const float actual_dt = sim_time / n_steps;  // exact coverage

    for (int step = 0; step < n_steps; ++step) {

        // --- Feed wave_source into wave_p ---
        for (int i = 0; i < h * w; ++i) {
            if (wave_source[i] > 0.001f) {
                float feed = wave_source[i] * feed_rate * actual_dt;
                feed = std::min(feed, wave_source[i]);
                wave_p[i] += feed;
                wave_source[i] -= feed;
            }
        }

        // --- Laplacian with Neumann BCs at obstacles ---
        // We need the old wave_p for the stencil, so use wave_v update in-place
        // trick: compute laplacian and update velocity in one pass, then update
        // pressure in a second pass. This avoids allocating a temp array.
        //
        // Actually we DO need a snapshot of wave_p for the stencil. Use wave_v
        // as temp isn't safe. Allocate once (on stack for small grids, heap otherwise).
        // For 120x75 = 9000 floats = 36 KB — fits in L1 cache easily.

        // We'll update wave_v in-place (it doesn't appear in the stencil).
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            const int row_up   = (y > 0)     ? (y - 1) * w : row;
            const int row_down = (y < h - 1) ? (y + 1) * w : row;

            for (int x = 0; x < w; ++x) {
                const int i = row + x;

                // Neighbor values with Neumann BC: if neighbor is obstacle,
                // reflect (use this cell's value instead)
                float p_up    = obstacles[row_up + x]                       ? wave_p[i] : wave_p[row_up + x];
                float p_down  = obstacles[row_down + x]                     ? wave_p[i] : wave_p[row_down + x];
                float p_left  = (x > 0 && !obstacles[row + x - 1])         ? wave_p[row + x - 1] : wave_p[i];
                float p_right = (x < w - 1 && !obstacles[row + x + 1])     ? wave_p[row + x + 1] : wave_p[i];

                // Boundary: clamp to self (Neumann)
                if (y == 0)     p_up   = wave_p[i];
                if (y == h - 1) p_down = wave_p[i];
                if (x == 0)     p_left = wave_p[i];
                if (x == w - 1) p_right = wave_p[i];

                float lap = p_up + p_down + p_left + p_right - 4.0f * wave_p[i];

                wave_v[i] += (c_sq * lap - damping * wave_v[i]) * actual_dt;
            }
        }

        // --- Update pressure from velocity ---
        for (int i = 0; i < h * w; ++i) {
            wave_p[i] += wave_v[i] * actual_dt;
        }

        // --- Zero pressure on walls and vacuum ---
        for (int i = 0; i < h * w; ++i) {
            if (is_wall[i] || is_vacuum[i]) {
                wave_p[i] = 0.0f;
            }
        }

        // --- Transfer wave energy to atmosphere (zero-sum) ---
        // Subtract mean so the transfer pushes air around without creating it
        float wp_sum = 0.0f;
        for (int i = 0; i < h * w; ++i) wp_sum += wave_p[i];
        float wp_mean = wp_sum / (h * w);
        for (int i = 0; i < h * w; ++i) {
            atmosphere[i] += (wave_p[i] - wp_mean) * transfer * actual_dt;
        }

        // --- Zero walls/vacuum (no clamping — allow negative for rarefaction) ---
        for (int i = 0; i < h * w; ++i) {
            if (is_wall[i] || is_vacuum[i]) {
                atmosphere[i] = 0.0f;
            }
        }
    }
}
