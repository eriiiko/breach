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

void SmokeDynamics::step(
    float* smoke,
    const float* wind_x,
    const float* wind_y,
    const float* sink_x,
    const float* sink_y,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    int h, int w,
    float dt
) const {
    const int n = h * w;
    const float actual_dt = dt * dt_scale;

    // --- Smoke diffusion (wind-dependent) ---
    // D_effective = d_smoke * (1 + wind_diffusion_scale * |wind|)
    // Higher wind = more turbulent mixing = smoke disperses faster.
    std::vector<float> lap(n);

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
    // Back-trace distance preserves the *effective advection strength* of the
    // old central-difference term: that term integrated d(smoke)/dtau =
    // -wind . grad(smoke) with a tau-step of (advection_rate * actual_dt),
    // i.e. it advected smoke by the wind for that pseudo-time. The equivalent
    // semi-Lagrangian displacement is therefore wind * dt_adv with
    //   dt_adv = advection_rate * actual_dt
    // (actual_dt already folds in dt_scale), so the cloud moves at the same
    // per-substep speed. We do NOT add a CFL cap: back-trace is stable for any
    // displacement.
    const float dt_adv = advection_rate * actual_dt;

    // Double buffer: read from the post-diffusion snapshot, write the advected
    // result. Never overwrite mid-pass (a cell may be sampled by its neighbours).
    std::vector<float> src(smoke, smoke + n);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            // Skip impermeable / vacuum tiles: they hold no smoke and reading a
            // wind there is meaningless (handled by the final zeroing pass).
            if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

            // Smoke-side sink-pull toward the nearest breach. The sink field is
            // a per-cell unit-ish vector pointing down the BFS distance gradient
            // to the nearest exposed-vacuum tile, and is (0,0) wherever there is
            // no path to a breach (and everywhere when the map is unbreached) —
            // so with no breach the sink term vanishes and this step is bit-
            // identical to the plain semi-Lagrangian advection (smoke v2 S1).
            // The sink is a bias inside *smoke* transport only; it never touches
            // the pressure field. It exists because with aggressive atmosphere
            // diffusion the interior wind dies as pressure flattens, leaving a
            // stubborn haze a real vent would have cleared.
            //
            // Back-trace the departure point. Wind uses the standard pull form
            //   p = cell - wind * dt_adv          (sample upwind, smoke rides wind)
            // The sink, however, is a DRAIN: a cell should inherit the emptiness
            // that lies toward the breach, so the sink term back-traces *toward*
            // the breach the sink vector points at (it samples down-gradient and
            // pulls in the vacuum's 0 via the breach-corner sampling below). For
            // a uniform saturated room pure pull-advection by a converging field
            // is the identity, so the drain comes entirely from sampling the
            // breach's emptiness — hence the sink samples toward it.
            //
            // The sink displacement is taken as ``sink_strength`` cells along the
            // unit sink vector and is CAPPED at one cell per substep: the field
            // is a per-cell next-hop direction down the BFS gradient, not a
            // straight shot to a possibly-around-a-corner breach, so the drain
            // must propagate one cell at a time (the emptied down-gradient
            // neighbour is sampled this substep; its own neighbour next substep).
            // An uncapped multi-cell sink ray would fly straight off the BFS path
            // into a wall and stall. Capping keeps the drain following the
            // gradient. The wind term keeps its full (uncapped) displacement.
            float sink_disp = sink_strength;
            if (sink_disp > 1.0f) sink_disp = 1.0f;
            float bx = -wind_x[i] * dt_adv + sink_disp * sink_x[i];
            float by = -wind_y[i] * dt_adv + sink_disp * sink_y[i];
            float px = static_cast<float>(x) + bx;
            float py = static_cast<float>(y) + by;

            // Wall-clip the back-trace ray. A long step can leap *over* a
            // one-cell-thick wall; the per-corner exclusion below only sees the
            // four cells around the landing point, not the cells the ray
            // crossed. So march from the cell toward the departure point in
            // sub-cell steps and stop just before the first sealed tile — smoke
            // is then pulled from the near side of the wall, never through it.
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
                                          permeability, h, w)) {
                            break;          // stop at the last open point (wall)
                        }
                        // Advance onto this point. If it is a breach (exposed
                        // vacuum, not a wall), advance ONTO it and stop: the
                        // bilinear sample there reads the vacuum's 0, so the
                        // cell vents into the breach. Otherwise keep marching.
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

            // Clamp the sample position in-bounds so we never read past the grid
            // (cell-centred sampling domain [0, w-1] x [0, h-1]).
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

            // Permeability-aware bilinear sample. Each of the four corners is
            // weighted by its bilinear weight, BUT a corner that is sealed
            // (solid/wall/vacuum or permeability<=0) is excluded so smoke is
            // never pulled out of, or teleported through, a wall. Weights of the
            // surviving corners are renormalised; if every corner is sealed we
            // fall back to the cell's own value (no transport this step).
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
                // A SEALED corner (solid wall / hull / zero-permeability) is
                // excluded: smoke is never pulled out of, or teleported through,
                // a wall. A BREACH corner (exposed vacuum: is_vacuum but NOT
                // solid) is different — it is genuinely empty space, so it is
                // INCLUDED with value 0. Pulling that 0 into the interior is the
                // drain: a cell back-tracing toward the breach (the sink term
                // above) inherits the vacuum's emptiness and loses smoke, which
                // is exactly venting to space. Without this a uniform saturated
                // room is invariant under pull-advection and never clears.
                bool solid_corner = obstacles[j] || is_wall[j] ||
                                    permeability[j] <= 0.0f;
                if (solid_corner) continue;
                bool breach_corner = is_vacuum[j];   // vacuum & !solid (sealed hull is solid)
                acc  += cw[k] * (breach_corner ? 0.0f : src[j]);
                wsum += cw[k];
            }

            smoke[i] = (wsum > 1e-6f) ? (acc / wsum) : src[i];
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
