// The horde field's host reference (G9p / residency P3d). See horde_field.h.
#include "horde_field.h"

#include <vector>

namespace horde {

int64_t flood(const uint8_t* tile_ok, int h, int w, int fp,
              const int32_t* seeds_yx, int n_seeds, int32_t* dist) {
    const int64_t n = static_cast<int64_t>(h) * w;
    for (int64_t i = 0; i < n; ++i) dist[i] = UNREACHED;
    if (h <= 0 || w <= 0 || fp <= 0) return 0;

    // Block passability by an integral image of the NOT-ok tiles: an anchor
    // (y, x) is passable when its fp x fp block is in bounds and holds zero
    // bad tiles. int32 is enough: the count is at most h*w.
    const int iw = w + 1;
    std::vector<int32_t> bad(static_cast<size_t>(h + 1) * iw, 0);
    for (int y = 0; y < h; ++y) {
        int32_t row = 0;
        for (int x = 0; x < w; ++x) {
            row += tile_ok[static_cast<int64_t>(y) * w + x] ? 0 : 1;
            bad[static_cast<size_t>(y + 1) * iw + (x + 1)] =
                bad[static_cast<size_t>(y) * iw + (x + 1)] + row;
        }
    }
    std::vector<uint8_t> pass(static_cast<size_t>(n), 0);
    for (int y = 0; y + fp <= h; ++y) {
        for (int x = 0; x + fp <= w; ++x) {
            const int32_t s =
                bad[static_cast<size_t>(y + fp) * iw + (x + fp)]
                - bad[static_cast<size_t>(y) * iw + (x + fp)]
                - bad[static_cast<size_t>(y + fp) * iw + x]
                + bad[static_cast<size_t>(y) * iw + x];
            pass[static_cast<size_t>(y) * w + x] = (s == 0) ? 1 : 0;
        }
    }

    // Multi-source BFS: the frontier is a flat FIFO of cell indices; every
    // cell enters it once (labelled on entry), so the loop is O(h*w).
    std::vector<int32_t> q;
    q.reserve(static_cast<size_t>(n));
    for (int s = 0; s < n_seeds; ++s) {
        const int32_t y = seeds_yx[2 * s];
        const int32_t x = seeds_yx[2 * s + 1];
        if (y < 0 || y >= h || x < 0 || x >= w) continue;
        const int32_t i = y * w + x;
        if (dist[i] == 0) continue;            // a repeated seed
        dist[i] = 0;
        q.push_back(i);
    }
    size_t head = 0;
    while (head < q.size()) {
        const int32_t i = q[head++];
        const int32_t y = i / w;
        const int32_t x = i - y * w;
        const int32_t nd = dist[i] + 1;
        // The four neighbours; the order is irrelevant to the distances.
        if (y > 0) {
            const int32_t j = i - w;
            if (pass[j] && dist[j] == UNREACHED) { dist[j] = nd; q.push_back(j); }
        }
        if (x > 0) {
            const int32_t j = i - 1;
            if (pass[j] && dist[j] == UNREACHED) { dist[j] = nd; q.push_back(j); }
        }
        if (x + 1 < w) {
            const int32_t j = i + 1;
            if (pass[j] && dist[j] == UNREACHED) { dist[j] = nd; q.push_back(j); }
        }
        if (y + 1 < h) {
            const int32_t j = i + w;
            if (pass[j] && dist[j] == UNREACHED) { dist[j] = nd; q.push_back(j); }
        }
    }
    return static_cast<int64_t>(q.size());
}

}  // namespace horde
