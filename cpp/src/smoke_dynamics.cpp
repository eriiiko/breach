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

// Helper: is the tile that contains the (clamped) sample point a SOLID WALL?
// Solid wall == obstacle / wall / zero permeability (a sealed hull is solid AND
// vacuum). A BREACH (exposed vacuum that is NOT solid) is deliberately NOT a
// wall here: the back-trace is allowed to reach a breach so smoke can vent into
// it (sampled as 0 by the bilinear pass). Used by the back-trace ray to stop at
// the first solid wall it would cross, so a long step cannot tunnel through a
// one-cell-thick wall — while still letting the sink ray reach the breach.
static inline bool solid_wall_at(int y, int x,
                                 const bool* obstacles, const bool* is_wall,
                                 const bool* is_vacuum, const float* perm,
                                 int h, int w) {
    if (y < 0 || y >= h || x < 0 || x >= w) return true;  // outside == wall
    int i = y * w + x;
    bool is_breach = is_vacuum[i] && !(obstacles[i] || is_wall[i] || perm[i] <= 0.0f);
    if (is_breach) return false;                            // venting target, not a wall
    return obstacles[i] || is_wall[i] || is_vacuum[i] || perm[i] <= 0.0f;
}

// Helper: semi-Lagrangian back-trace from cell (x,y) by displacement (bx,by),
// with the wall-clip march + permeability-aware bilinear sample, writing the
// result into smoke[i]. Shared by step()'s WIND advection and sink_hop()'s
// breach pull — they differ ONLY in how (bx,by) is computed (wind*dt vs the
// 1-cell-capped sink direction). `src` is the pre-pass snapshot the bilinear
// sample reads (never the live smoke array). Returns the new value for smoke[i].
//
// The wall-clip + breach-corner logic is identical to the original fused step:
// march toward the departure point in ~1-cell substeps, stop at the first solid
// wall (so a long step cannot tunnel a one-cell wall) or ON a breach (so the
// cell vents into the vacuum's 0); then bilinear-sample with sealed corners
// excluded and a breach corner contributing 0 (the drain).
static inline float backtrace_sample(
        const float* src, int x, int y, float bx, float by,
        const bool* obstacles, const bool* is_wall, const bool* is_vacuum,
        const float* perm, int h, int w) {
    float px = static_cast<float>(x) + bx;
    float py = static_cast<float>(y) + by;

    // Wall-clip the back-trace ray (see the original step comment): march from
    // the cell toward the departure point in sub-cell steps and stop just before
    // the first sealed tile, or ON a breach (vent target).
    {
        float dist = std::sqrt(bx * bx + by * by);
        int steps = static_cast<int>(std::ceil(dist));  // ~1 sample/cell
        if (steps > 0) {
            float inv = 1.0f / static_cast<float>(steps);
            float sx = bx * inv, sy = by * inv;
            float cx = static_cast<float>(x);
            float cy = static_cast<float>(y);
            for (int t = 0; t < steps; ++t) {
                float nxp = cx + sx, nyp = cy + sy;
                int ti = static_cast<int>(std::floor(nxp + 0.5f));
                int tj = static_cast<int>(std::floor(nyp + 0.5f));
                if (solid_wall_at(tj, ti, obstacles, is_wall, is_vacuum,
                                  perm, h, w)) {
                    break;          // stop at the last open point (wall)
                }
                cx = nxp; cy = nyp;
                int bi = tj * w + ti;
                if (tj >= 0 && tj < h && ti >= 0 && ti < w &&
                    is_vacuum[bi]) {
                    break;          // reached the breach — vent here
                }
            }
            px = cx; py = cy;
        }
    }

    // Clamp the sample position in-bounds.
    if (px < 0.0f)              px = 0.0f;
    else if (px > w - 1.0f)     px = static_cast<float>(w - 1);
    if (py < 0.0f)              py = 0.0f;
    else if (py > h - 1.0f)     py = static_cast<float>(h - 1);

    int x0 = static_cast<int>(std::floor(px));
    int y0 = static_cast<int>(std::floor(py));
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);
    float fx = px - static_cast<float>(x0);
    float fy = py - static_cast<float>(y0);

    const int ci[4] = { y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1 };
    const float cw[4] = {
        (1.0f - fx) * (1.0f - fy),
        fx         * (1.0f - fy),
        (1.0f - fx) * fy,
        fx         * fy,
    };

    float acc = 0.0f;
    float wsum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        int j = ci[k];
        // Sealed corner excluded; breach corner (vacuum & !solid) included as 0.
        bool solid_corner = obstacles[j] || is_wall[j] || perm[j] <= 0.0f;
        if (solid_corner) continue;
        bool breach_corner = is_vacuum[j];
        acc  += cw[k] * (breach_corner ? 0.0f : src[j]);
        wsum += cw[k];
    }

    return (wsum > 1e-6f) ? (acc / wsum) : src[y * w + x];
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
    // Patch 2b: dt_scale is GONE — smoke advects/diffuses on the REAL dt. The
    // visible wind-ride is preserved by the ×dt_scale²-bumped advection_rate
    // default; the DIFFUSION is now dt_scale² (≈9×) weaker than the shipped
    // build — Erik re-tunes d_smoke / the per-gas [gases.*] diffusion table.
    const float actual_dt = dt;

    // --- Smoke diffusion (wind-dependent) ---
    // D_effective = d_smoke * (1 + wind_diffusion_scale * |wind|)
    // Higher wind = more turbulent mixing = smoke disperses faster.
    // Reused scratch (GPU-prep: no per-step alloc). Every lap[i] is written
    // below before the diffusion apply loop reads it, so no re-init needed.
    // `__restrict`: lap_ is solver-private and aliases no field pointer —
    // restores the fresh-local no-alias property /fp:fast needs for identical
    // codegen (read in the pure-FP apply loop). No self-aliasing (lap is never
    // read in the loop that writes it).
    if (lap_.size() != (size_t)n) lap_.assign(n, 0.0f);
    float* __restrict lap = lap_.data();

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

    // --- Advection by precomputed wind field (semi-Lagrangian back-trace) ---
    // Unconditionally stable and checkerboard-free (the central-difference
    // stencil it replaces oscillated near breaches/explosions). This is also
    // the CUDA-ready algorithm (Stable-Fluids back-trace + texture bilerp).
    //
    // Patch 2b: WIND-ONLY now. The breach sink-pull that used to be fused into
    // this back-trace velocity is gone — it is the standalone sink_hop() pass the
    // engine runs K× per tick. The displacement is purely wind * dt_adv with
    //   dt_adv = advection_rate * actual_dt        (actual_dt == the real dt)
    // No CFL cap: back-trace is stable for any displacement.
    const float dt_adv = advection_rate * actual_dt;

    // Double buffer: read from the post-diffusion snapshot, write the advected
    // result. Never overwrite mid-pass (a cell may be sampled by its neighbours).
    // Reused scratch via the SWAP idiom (storage retained in src_ across steps).
    std::vector<float> src;
    src.swap(src_);                          // steal retained storage
    src.assign(smoke, smoke + n);            // snapshot the post-diffusion field

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            // Skip impermeable / vacuum tiles: they hold no smoke and reading a
            // wind there is meaningless (handled by the final zeroing pass).
            if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

            // Wind pull-advection: sample upwind, p = cell - wind * dt_adv.
            const float bx = -wind_x[i] * dt_adv;
            const float by = -wind_y[i] * dt_adv;
            smoke[i] = backtrace_sample(src.data(), x, y, bx, by,
                                        obstacles, is_wall, is_vacuum,
                                        permeability, h, w);
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

    // Retain src's storage for the next step (swap idiom; no per-step alloc).
    src.swap(src_);
}

// Patch 2b: the decoupled breach sink-pull — ONE 1-cell BFS-gradient hop. This
// is the EXACT mechanism formerly fused into step()'s back-trace, extracted so
// the engine can run it K× per tick on its own schedule (independent of the wave
// CFL). The back-trace velocity is the sink direction ONLY, capped at one cell.
void SmokeDynamics::sink_hop(
    float* smoke,
    const float* sink_x,
    const float* sink_y,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    int h, int w
) const {
    const int n = h * w;

    // Snapshot (double-buffer) so a cell's pull samples the pre-hop field, never
    // a half-updated neighbour. Reuse src_'s retained storage (swap idiom).
    std::vector<float> src;
    src.swap(src_);
    src.assign(smoke, smoke + n);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

            // The sink is a DRAIN: the cell back-traces TOWARD the breach the
            // sink vector points at, inheriting the emptiness down-gradient (the
            // breach corner samples 0). The displacement is sink_strength cells
            // along the unit sink vector, CAPPED at one cell (LOAD-BEARING: the
            // field is a per-cell next-hop down the BFS shortest path, not a
            // straight shot — an uncapped multi-cell ray flies off the path into
            // a wall and stalls; one cell per hop walks the path). With no breach
            // the sink field is (0,0) here, so bx=by=0 -> backtrace_sample is the
            // identity (sealed rooms untouched).
            float sink_disp = sink_strength;
            if (sink_disp > 1.0f) sink_disp = 1.0f;
            const float bx = sink_disp * sink_x[i];
            const float by = sink_disp * sink_y[i];
            smoke[i] = backtrace_sample(src.data(), x, y, bx, by,
                                        obstacles, is_wall, is_vacuum,
                                        permeability, h, w);
        }
    }

    // Clamp and zero walls/vacuum (same invariant as step()).
    for (int i = 0; i < n; ++i) {
        if (is_wall[i] || is_vacuum[i]) {
            smoke[i] = 0.0f;
        } else {
            smoke[i] = std::clamp(smoke[i], 0.0f, 1.0f);
        }
    }

    src.swap(src_);
}
