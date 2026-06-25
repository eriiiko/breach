#include "smoke_dynamics.h"
#include "fixed_point.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// ============================================================================
// S2b — INTEGER semi-Lagrangian smoke + gas (Q16.16), docs/s2_fixed_point_plan.md
// §S2b. Ported from the proven prototype tools/s2_advection_demo/advection_demo.py
// (the `SLint` scheme, commit ceb601b). Bit-deterministic, a visual twin of the
// float SL, gently non-conservative (the >>16 truncation == a built-in mild
// decay; accepted Q-S2-1 — deterministic, so behaviour not desync). NO flux form,
// NO limiter, NO outflow clamp. The renorm 1/wsum is the shared
// fixedpoint::reciprocal_q16 (the Newton reciprocal S2c reuses for the GS Dinv).
// ============================================================================

using namespace fixedpoint;

// Smoke is a [0,1] tracer in Q16.16: 0 == clear, FP_ONE (65536) == fully opaque.
static constexpr int32_t SMOKE_MAX_Q = FP_ONE;
// Newton-reciprocal floor for the bilinear renorm: clamp wsum to >= this so
// 1/wsum can't blow up (mirrors the prototype's WSUM_FLOOR_Q = FP_ONE>>8). This
// is ~1/256 of a unit weight — far below any real partial-corner weight.
static constexpr int32_t WSUM_FLOOR_Q = FP_ONE >> 8;       // 256
// The "wsum negligible -> keep self" guard (~the float build's 1e-6): below this
// summed corner weight the sample is the cell's own value (mirrors the prototype's
// `wsum <= FP_ONE>>14` guard and the float `wsum > 1e-6f`).
static constexpr int32_t WSUM_EPS_Q = FP_ONE >> 14;        // 4

// Helper: face-permeability neighbour value with Neumann fallback, in Q16.16.
// Returns f[self] + face*(f[neighbour] - f[self]) where
// face = min(perm[self], perm[neighbour]) (a Q16.16 permeability). For perm in
// {0,1} this is the obstacle mirror: a sealed neighbour (face=0) returns f[self]
// (reflect); an open neighbour (face=FP_ONE) returns f[neighbour]. Out-of-bounds
// also reflects. The `face*(f[n]-f[self])` is a Q16.16 product (mul_q16). FLOAT
// BRIDGE: permeability is still float (a structural cache); quantize it per face
// to Q16.16 here exactly like the wave Laplacian's per-face permeability bridge.
static inline int32_t neighbor_q(const int32_t* f, const float* perm,
                                 int y, int x, int dy, int dx, int h, int w) {
    int self_i = y * w + x;
    int ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return f[self_i];
    int ni = ny * w + nx;
    const float face_f = std::min(perm[self_i], perm[ni]);
    const int32_t face_q = quantize((double)face_f);       // perm in [0,1] -> Q16.16
    return f[self_i] + mul_q16(face_q, f[ni] - f[self_i]);
}

// Helper: is the tile that contains the (clamped) sample point a SOLID WALL?
// (Identical logic to the original float build — operates on bool masks only, so
// it is representation-agnostic.) Solid wall == obstacle / wall / zero
// permeability (a sealed hull is solid AND vacuum). A BREACH (exposed vacuum that
// is NOT solid) is deliberately NOT a wall: the back-trace may reach a breach so
// smoke vents into it (sampled as 0 by the bilinear pass).
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

// Helper: the integer semi-Lagrangian back-trace from cell (x,y) by the Q16.16
// displacement (bx_q, by_q). Ported verbatim from the prototype's
// `_q_backtrace_sample` (the SLint scheme). Shared by step()'s WIND advection and
// sink_hop()'s breach pull — they differ ONLY in how (bx_q,by_q) is computed
// (wind*dt vs the 1-cell-capped sink direction). `src` is the pre-pass int32
// snapshot the bilinear sample reads (never the live smoke array). Returns the
// new Q16.16 value for smoke[i].
//
// Three pieces, all from the prototype:
//   * The sqrt-free DDA wall-clip march — step the DOMINANT axis cell-by-cell
//     (Chebyshev march) toward the source, stopping before a sealed tile (so it
//     cannot tunnel a 1-cell wall) or ON a breach (vent target). NO sqrt (the
//     float build's back-trace-length sqrt is gone).
//   * The integer bilinear sample — 4 corner-weight Q16.16 products accumulated
//     in int64, then narrowed; sealed corners excluded, a breach corner == 0.
//   * The renorm 1/wsum via fixedpoint::reciprocal_q16 (the shared Newton
//     reciprocal), then a pinned >>16 truncation (mul_q16) — the gentle decay.
static inline int32_t backtrace_sample_q(
        const int32_t* src, int x, int y, int32_t bx_q, int32_t by_q,
        const bool* obstacles, const bool* is_wall, const bool* is_vacuum,
        const float* perm, int h, int w) {
    // Departure point in Q16.16 (cell index << 16 + displacement).
    int64_t px_q = ((int64_t)x << FP_SHIFT) + bx_q;
    int64_t py_q = ((int64_t)y << FP_SHIFT) + by_q;

    // ---- Wall-clip march (DDA, no sqrt) ----
    // Dominant axis = the larger |displacement|. n_steps = ceil(Chebyshev dist)
    // = ceil(max(|bx|,|by|)) cells, from the Q16.16 magnitude (>>16 = floor;
    // +1 if any fraction -> ceil). March one DOMINANT-axis cell per step; the
    // minor axis advances by displacement/n_steps each step.
    const int32_t abx = bx_q >= 0 ? bx_q : -bx_q;
    const int32_t aby = by_q >= 0 ? by_q : -by_q;
    const int32_t amax = abx >= aby ? abx : aby;
    int n_steps = amax >> FP_SHIFT;
    if (amax & (FP_ONE - 1)) n_steps += 1;                  // ceil
    if (n_steps > 0) {
        // Per-step increment = displacement / n_steps in Q16.16. The prototype
        // (advection_demo.py) uses Python `//` which FLOORS toward -inf; C `/`
        // truncates toward 0, so for a NEGATIVE displacement (upwind back-trace,
        // bx_q < 0) they differ by 1 count. Match the prototype exactly with a
        // floor-divide (the proven visual twin) so the march steps identically:
        //   floor(a/b) for b>0 == (a>=0) ? a/b : -((-a + b - 1)/b)
        auto floordiv = [](int32_t a, int b) -> int32_t {
            return (a >= 0) ? (a / b) : -(((-(int64_t)a) + b - 1) / b);
        };
        const int32_t sx_q = floordiv(bx_q, n_steps);
        const int32_t sy_q = floordiv(by_q, n_steps);
        int64_t cx_q = (int64_t)x << FP_SHIFT;
        int64_t cy_q = (int64_t)y << FP_SHIFT;
        for (int s = 0; s < n_steps; ++s) {
            const int64_t nxp_q = cx_q + sx_q;
            const int64_t nyp_q = cy_q + sy_q;
            // tile center test: floor(coord + 0.5). coord+0.5 in Q16.16 =
            // nxp_q + FP_HALF; the integer tile is that >> 16.
            const int ti = (int)((nxp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            const int tj = (int)((nyp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            if (solid_wall_at(tj, ti, obstacles, is_wall, is_vacuum, perm, h, w))
                break;
            cx_q = nxp_q;
            cy_q = nyp_q;
            if (tj >= 0 && tj < h && ti >= 0 && ti < w && is_vacuum[tj * w + ti])
                break;                                       // reached the breach
        }
        px_q = cx_q;
        py_q = cy_q;
    }

    // ---- Clamp in-bounds (Q16.16) ----
    const int64_t hi_x = (int64_t)(w - 1) << FP_SHIFT;
    const int64_t hi_y = (int64_t)(h - 1) << FP_SHIFT;
    if (px_q < 0) px_q = 0; else if (px_q > hi_x) px_q = hi_x;
    if (py_q < 0) py_q = 0; else if (py_q > hi_y) py_q = hi_y;

    // ---- Integer bilinear sample ----
    const int x0 = (int)(px_q >> FP_SHIFT);                 // floor (px_q >= 0)
    const int y0 = (int)(py_q >> FP_SHIFT);
    const int x1 = (x0 + 1 <= w - 1) ? x0 + 1 : w - 1;
    const int y1 = (y0 + 1 <= h - 1) ? y0 + 1 : h - 1;
    const int32_t fx_q = (int32_t)(px_q - ((int64_t)x0 << FP_SHIFT));   // frac [0,1)
    const int32_t fy_q = (int32_t)(py_q - ((int64_t)y0 << FP_SHIFT));
    const int32_t ifx_q = FP_ONE - fx_q;                   // (1 - fx)
    const int32_t ify_q = FP_ONE - fy_q;
    // Four corner weights, each a Q16.16 product of two Q16.16 fractions.
    const int32_t w00 = mul_q16(ifx_q, ify_q);             // (1-fx)(1-fy)
    const int32_t w10 = mul_q16(fx_q,  ify_q);             // fx (1-fy)
    const int32_t w01 = mul_q16(ifx_q, fy_q);              // (1-fx) fy
    const int32_t w11 = mul_q16(fx_q,  fy_q);              // fx fy
    const int cyx[4][2] = { {y0, x0}, {y0, x1}, {y1, x0}, {y1, x1} };
    const int32_t cw[4] = { w00, w10, w01, w11 };

    int64_t acc = 0;          // int64 accumulator of weight*density (Q16.16*Q16.16 -> Q.32)
    int32_t wsum_q = 0;       // Q16.16 sum of live corner weights
    for (int k = 0; k < 4; ++k) {
        const int cy_ = cyx[k][0];
        const int cx_ = cyx[k][1];
        const int j = cy_ * w + cx_;
        if (obstacles[j] || is_wall[j] || perm[j] <= 0.0f) continue;   // sealed corner
        const int32_t val_q = is_vacuum[j] ? 0 : src[j];   // breach corner == 0
        acc += mul_wide(cw[k], val_q);                     // int64; scale = 2^32
        wsum_q += cw[k];
    }
    if (wsum_q <= WSUM_EPS_Q) return src[y * w + x];        // negligible -> keep self

    // result = acc / wsum. Narrow acc (Q.32 -> Q16.16) by >>16, then multiply by
    // recip(wsum) (Q16.16) and narrow again -> divides by wsum. recip via the
    // shared Newton reciprocal (no divide). wsum clamped to a floor first.
    const int32_t wsum_clamped = (wsum_q < WSUM_FLOOR_Q) ? WSUM_FLOOR_Q : wsum_q;
    const int32_t recip_q = reciprocal_q16(wsum_clamped);  // 1/wsum, Q16.16
    const int32_t acc_q = narrow(acc);                     // Q.32 -> Q16.16
    return mul_q16(acc_q, recip_q);                        // (sum w*d)/wsum, Q16.16
}

void SmokeDynamics::step(
    int32_t* smoke,
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
    const float actual_dt = dt;

    // --- Smoke diffusion (wind-dependent), integer Q16.16 ---
    // D_effective = d_smoke * (1 + wind_diffusion_scale * |wind|²); higher wind =
    // more turbulent mixing. The Laplacian is the permeability-weighted 4-neighbour
    // gather (neighbor_q), summed, then `smoke += d_eff*dt*lap`. The scalar
    // d_eff*dt is a per-cell positive Q16.16 coefficient computed at the float
    // bridge (d_smoke / wind_diffusion_scale / dt are config/float; |wind|² is
    // the float wind FLOAT BRIDGE until S2c), then mul_q16 onto the integer lap.
    if (lap_.size() != (size_t)n) lap_.assign(n, 0);
    int32_t* lap = lap_.data();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            const int32_t s = smoke[i];
            const int32_t s_up    = neighbor_q(smoke, permeability, y, x, -1,  0, h, w);
            const int32_t s_down  = neighbor_q(smoke, permeability, y, x,  1,  0, h, w);
            const int32_t s_left  = neighbor_q(smoke, permeability, y, x,  0, -1, h, w);
            const int32_t s_right = neighbor_q(smoke, permeability, y, x,  0,  1, h, w);
            // lap = (up + down + left + right) - 4*s. Each neighbor_q already
            // returns f[self] + face*(f[n]-f[self]); the sum-minus-4s telescopes
            // to the permeability-weighted Laplacian, exactly as the float build
            // and the prototype's `lap += nbr - s`.
            lap[i] = (s_up + s_down + s_left + s_right) - 4 * s;
        }
    }

    for (int i = 0; i < n; ++i) {
        // FLOAT BRIDGE (until S2c): the wind is float; |wind|² and d_eff are
        // computed in double, then the positive scalar coefficient d_eff*dt is
        // quantized to Q16.16 and mul_q16'd onto the integer Laplacian. d_eff*dt
        // is small (d_smoke~0.1, dt~1/24) so it sits comfortably in Q16.16.
        const double wind_sq = (double)wind_x[i] * wind_x[i]
                             + (double)wind_y[i] * wind_y[i];
        const double d_eff = (double)d_smoke
                           * (1.0 + (double)wind_diffusion_scale * wind_sq);
        const int32_t coeff_q = quantize(d_eff * (double)actual_dt);
        smoke[i] += mul_q16(coeff_q, lap[i]);
    }

    // --- Advection by precomputed wind field (integer semi-Lagrangian) ---
    // Patch 2b: WIND-ONLY (the breach sink-pull is the standalone sink_hop pass).
    // The displacement is wind * dt_adv with dt_adv = advection_rate * actual_dt.
    // FLOAT BRIDGE (until S2c): the wind is float, so the per-cell displacement
    // -wind*dt_adv is computed in double and quantized to a Q16.16 displacement
    // at the boundary (mirrors the prototype's bx_q/by_q quantize). After S2c the
    // wind is integer and this becomes an integer multiply.
    const double dt_adv = (double)advection_rate * (double)actual_dt;

    // Double buffer: snapshot the post-diffusion int32 field; the back-trace reads
    // the snapshot, writes the advected result. Reused scratch (SWAP idiom).
    std::vector<int32_t> src;
    src.swap(src_);
    src.assign(smoke, smoke + n);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;
            // Wind pull-advection: sample upwind, p = cell - wind * dt_adv.
            const double bx_f = -(double)wind_x[i] * dt_adv;
            const double by_f = -(double)wind_y[i] * dt_adv;
            const int32_t bx_q = quantize(bx_f);            // round-to-nearest boundary cast
            const int32_t by_q = quantize(by_f);
            smoke[i] = backtrace_sample_q(src.data(), x, y, bx_q, by_q,
                                          obstacles, is_wall, is_vacuum,
                                          permeability, h, w);
        }
    }

    // --- Clamp and zero walls/vacuum (integer invariant) ---
    for (int i = 0; i < n; ++i) {
        if (is_wall[i] || is_vacuum[i]) {
            smoke[i] = 0;
        } else {
            if (smoke[i] < 0) smoke[i] = 0;
            else if (smoke[i] > SMOKE_MAX_Q) smoke[i] = SMOKE_MAX_Q;
        }
    }

    src.swap(src_);                                          // retain storage
}

// Patch 2b: the decoupled breach sink-pull — ONE 1-cell BFS-gradient hop, now in
// the SAME integer-SL machinery (Q-S2-5: stays SL, a port not a flux bias). The
// back-trace velocity is the sink direction ONLY, capped at one cell. The engine
// runs it K× per tick. With no breach the sink field is all-zero -> bx=by=0 ->
// backtrace_sample_q is the identity (sealed rooms untouched).
void SmokeDynamics::sink_hop(
    int32_t* smoke,
    const float* sink_x,
    const float* sink_y,
    const bool* obstacles,
    const bool* is_wall,
    const bool* is_vacuum,
    const float* permeability,
    int h, int w
) const {
    const int n = h * w;

    std::vector<int32_t> src;
    src.swap(src_);
    src.assign(smoke, smoke + n);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            if (obstacles[i] || is_wall[i] || is_vacuum[i]) continue;

            // The sink is a DRAIN: back-trace TOWARD the breach the sink vector
            // points at, inheriting the down-gradient emptiness (the breach corner
            // samples 0). Displacement = sink_strength cells along the unit sink
            // vector, CAPPED at one cell. FLOAT BRIDGE: sink_x/sink_y + sink_strength
            // are float; the displacement is computed in double and quantized.
            double sink_disp = (double)sink_strength;
            if (sink_disp > 1.0) sink_disp = 1.0;
            const int32_t bx_q = quantize(sink_disp * (double)sink_x[i]);
            const int32_t by_q = quantize(sink_disp * (double)sink_y[i]);
            smoke[i] = backtrace_sample_q(src.data(), x, y, bx_q, by_q,
                                          obstacles, is_wall, is_vacuum,
                                          permeability, h, w);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (is_wall[i] || is_vacuum[i]) {
            smoke[i] = 0;
        } else {
            if (smoke[i] < 0) smoke[i] = 0;
            else if (smoke[i] > SMOKE_MAX_Q) smoke[i] = SMOKE_MAX_Q;
        }
    }

    src.swap(src_);
}
