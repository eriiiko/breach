#pragma once
// ============================================================================
// #29 P7.0a -- THE DEVICE SIGHT FUNCTION (residency design 4.4c decision 6):
// GameMap.has_los in both of its forms, step for step (the Bresenham walk of
// the Python, int64, no float):
//
//   has_los(solid, screen, ...)      the line of SIGHT: stops on a solid tile
//                                    or on the sight screen (the grille)
//   has_los(solid, nullptr, ...)     physical=True: stops on solid alone (what
//                                    travels with the air; the spray cone's)
//
// over one env's (h, w) planes: `solid` the GameMap.solid bool plane, `screen`
// the GameMap.sight_screen bool plane (1 byte each; a slot input until P7e
// makes it resident). As in Python the start tile is tested, the end tile
// never, and a tile outside the grid stops nothing. Held equal to
// GameMap.has_los on random pairs over real maps, in a batch of different
// maps, by tests/cuda_unit_kit_check.py part SIGHT (cuda_unit_kit_probe.cu).
// The bulk form (sight_lines.cpp's rays_clear, the observation's) stays P8b's.
//
// An endpoint outside +-COORD_MAX sets *bad (the caller FAULTS; the walk would
// still be exact, but no map is that wide and the bound keeps 2 * err in
// int64 with room to spare).
//
// Device-only header; CLASS T: nothing in the played tick includes it.
#include <cstdint>

namespace breach_cuda {
namespace unit_sight {

constexpr int64_t COORD_MAX = (int64_t)1 << 30;

__device__ inline bool has_los(const uint8_t* solid, const uint8_t* screen, int h, int w,
                               int64_t fy1, int64_t fx1, int64_t fy2, int64_t fx2,
                               bool* bad) {
    if (fy1 < -COORD_MAX || fy1 > COORD_MAX || fx1 < -COORD_MAX || fx1 > COORD_MAX ||
        fy2 < -COORD_MAX || fy2 > COORD_MAX || fx2 < -COORD_MAX || fx2 > COORD_MAX) {
        *bad = true;
        return false;
    }
    const int64_t dx = fx2 > fx1 ? fx2 - fx1 : fx1 - fx2;
    const int64_t dy = fy2 > fy1 ? fy2 - fy1 : fy1 - fy2;
    const int64_t sx = fx1 < fx2 ? 1 : -1;
    const int64_t sy = fy1 < fy2 ? 1 : -1;
    int64_t err = dx - dy;
    int64_t x = fx1, y = fy1;
    // The walk reaches the end in dx + dy steps at most (each step moves x, y
    // or both toward it); the bound only guards a broken walk.
    for (int64_t step = 0; step <= dx + dy + 1; ++step) {
        if (x == fx2 && y == fy2) return true;
        if (y >= 0 && y < h && x >= 0 && x < w) {
            const size_t i = (size_t)y * (size_t)w + (size_t)x;
            if (solid[i] || (screen != nullptr && screen[i])) return false;
        }
        const int64_t e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
    *bad = true;
    return false;
}

}  // namespace unit_sight
}  // namespace breach_cuda
