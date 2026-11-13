// The sight routine for many tiles (#195 1c). See sight_lines.h.
#include "sight_lines.h"

namespace sight {

bool line_clear(const uint8_t* solid, const uint8_t* screen, int h, int w,
                bool physical, int32_t y0, int32_t x0, int32_t y1, int32_t x1) {
    const uint8_t* scr = physical ? nullptr : screen;
    int64_t x = x0, y = y0;
    const int64_t dx = x1 > x0 ? int64_t(x1) - x0 : int64_t(x0) - x1;
    const int64_t dy = y1 > y0 ? int64_t(y1) - y0 : int64_t(y0) - y1;
    const int64_t sx = x0 < x1 ? 1 : -1;
    const int64_t sy = y0 < y1 ? 1 : -1;
    int64_t err = dx - dy;
    // The correct walk reaches the end tile in exactly max(dx, dy) steps, so
    // dx + dy + 1 iterations are never exhausted by it; the bound is only a
    // guard, so that a broken walk answers wrongly (and a test sees it)
    // instead of never returning.
    for (int64_t step = 0; step <= dx + dy; ++step) {
        if (x == x1 && y == y1) return true;
        if (y >= 0 && y < h && x >= 0 && x < w) {
            const int64_t i = y * w + x;
            if (solid[i] || (scr != nullptr && scr[i])) return false;
        }
        const int64_t e2 = 2 * err;
        // Both steps read the same e2 (has_los's order: x, then y).
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 < dx) { err += dx; y += sy; }
    }
    return false;   // unreachable on the correct walk
}

void rays(const uint8_t* solid, const uint8_t* screen, int h, int w,
          bool physical, const int32_t* pairs, int64_t n, uint8_t* out) {
    for (int64_t k = 0; k < n; ++k) {
        const int32_t* p = pairs + 4 * k;
        out[k] = line_clear(solid, screen, h, w, physical, p[0], p[1], p[2],
                            p[3]) ? 1 : 0;
    }
}

void window(const uint8_t* solid, const uint8_t* screen, int h, int w,
            bool physical, int32_t y, int32_t x, int32_t wy, int32_t wx,
            int32_t wh, int32_t ww, bool to_window, uint8_t* out) {
    int64_t k = 0;
    for (int32_t j = 0; j < wh; ++j) {
        for (int32_t i = 0; i < ww; ++i, ++k) {
            const int32_t ty = wy + j, tx = wx + i;
            out[k] = (to_window
                          ? line_clear(solid, screen, h, w, physical, ty, tx, y, x)
                          : line_clear(solid, screen, h, w, physical, y, x, ty, tx))
                         ? 1 : 0;
        }
    }
}

}  // namespace sight
