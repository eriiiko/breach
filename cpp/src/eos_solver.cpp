#include "eos_solver.h"
#include "fixed_point.h"
#include "bulk_transport.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if !defined(__SIZEOF_INT128__) && defined(_MSC_VER)
#include <intrin.h>
#endif

using namespace fixedpoint;

namespace {

// ---- digest: a cheap FNV-1a-style running hash over a Q16.16 buffer ------
// Sequential, order-DEPENDENT (CPU-only lockstep gate; not the order-free
// reduction a future GPU port would need — P6, out of scope). Deterministic
// bit-for-bit given the identical input sequence on any conforming compiler
// (pure integer +/xor/*).
uint64_t digest_of(const int32_t* buf, int n, uint64_t seed) {
    uint64_t h = seed ^ 1469598103934665603ULL;   // FNV offset basis, salted
    for (int i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint32_t)buf[i];
        h *= 1099511628211ULL;                    // FNV prime
    }
    return h;
}

// ---- ONE-TRUNCATION wide*narrow multiply (the §3.4 rule-1 idiom) ---------
// wide_a is a Q.32-scale int64 (e.g. a mul_wide(q16,q16) result — represents
// a "value" that may be FAR outside q16's +/-32768 representable range, e.g.
// N_cell*c^2*dt for an O2-tank spike); b_q16 is a plain Q16.16 quantity
// (e.g. div(u), typically small). Ported verbatim from bulk_transport.cpp's
// flux_to_dq (the SAME "one shared truncation of a wide*wide product" shape,
// generalized to a runtime coefficient) — narrows the Q.48-scale product by
// >>32 ONCE, at the point the combined magnitude is expected back in range
// (if the TRUE result still exceeds int32, this is exactly what the P3
// overflow-stress-sweep gate must catch — a genuine physical blow-up, not an
// arithmetic artifact of an earlier premature narrow).
inline q16 wide_mul_q16(int64_t wide_a_q32, q16 b_q16) {
#if defined(__SIZEOF_INT128__)
    __int128 p = (__int128)wide_a_q32 * (__int128)b_q16;
    return (q16)(p >> 32);
#else
    long long hi;
    long long lo = _mul128((long long)wide_a_q32, (long long)b_q16, &hi);
    unsigned long long ulo = (unsigned long long)lo;
    long long res = (long long)((ulo >> 32) | ((unsigned long long)hi << (64 - 32)));
    return (q16)res;
#endif
}

// ---- solid-mirror neighbor read (Neumann BC helper) ----------------------
// Returns the index to read for a face neighbor: mirrors to self at a grid
// edge OR a solid neighbor (Neumann: zero cross-wall flux/gradient), exactly
// the il/ir/iu/id convention AtmosphereSolver::diffuse_solve's wind term uses.
inline int mirror_idx(int self_i, int ny, int nx, int h, int w, const bool* solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return self_i;
    const int ni = ny * w + nx;
    if (solid[ni]) return self_i;
    return ni;
}

// ---- semi-Lagrangian backtrace + bilinear sample (self-contained) --------
// Same SHAPE as smoke_dynamics.cpp's backtrace_sample_q (the proven SLint
// scheme: sqrt-free DDA wall-clip march + integer bilinear + reciprocal_q16
// renorm) — duplicated here (not exported by that TU) with the eos_solver's
// simpler single `solid`/`is_vacuum` mask pair (this engine's obstacles ==
// is_wall == solid, gamemap.py). Samples ANY Q16.16 field (u components, T).
int32_t eos_backtrace_sample_q(
        const int32_t* src, int x, int y, int32_t bx_q, int32_t by_q,
        const bool* solid, const bool* is_vacuum,
        const float* perm, int h, int w) {
    int64_t px_q = ((int64_t)x << FP_SHIFT) + bx_q;
    int64_t py_q = ((int64_t)y << FP_SHIFT) + by_q;

    const int32_t abx = bx_q >= 0 ? bx_q : -bx_q;
    const int32_t aby = by_q >= 0 ? by_q : -by_q;
    const int32_t amax = abx >= aby ? abx : aby;
    int n_steps = amax >> FP_SHIFT;
    if (amax & (FP_ONE - 1)) n_steps += 1;

    auto solid_wall_at = [&](int ty, int tx) -> bool {
        if (ty < 0 || ty >= h || tx < 0 || tx >= w) return true;
        const int i = ty * w + tx;
        const bool breach = is_vacuum[i] && !solid[i];
        if (breach) return false;
        return solid[i] || is_vacuum[i] || perm[i] <= 0.0f;
    };

    if (n_steps > 0) {
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
            const int ti = (int)((nxp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            const int tj = (int)((nyp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            if (solid_wall_at(tj, ti)) break;
            cx_q = nxp_q;
            cy_q = nyp_q;
            if (tj >= 0 && tj < h && ti >= 0 && ti < w && is_vacuum[tj * w + ti]) break;
        }
        px_q = cx_q;
        py_q = cy_q;
    }

    const int64_t hi_x = (int64_t)(w - 1) << FP_SHIFT;
    const int64_t hi_y = (int64_t)(h - 1) << FP_SHIFT;
    if (px_q < 0) px_q = 0; else if (px_q > hi_x) px_q = hi_x;
    if (py_q < 0) py_q = 0; else if (py_q > hi_y) py_q = hi_y;

    const int x0 = (int)(px_q >> FP_SHIFT);
    const int y0 = (int)(py_q >> FP_SHIFT);
    const int x1 = (x0 + 1 <= w - 1) ? x0 + 1 : w - 1;
    const int y1 = (y0 + 1 <= h - 1) ? y0 + 1 : h - 1;
    const int32_t fx_q = (int32_t)(px_q - ((int64_t)x0 << FP_SHIFT));
    const int32_t fy_q = (int32_t)(py_q - ((int64_t)y0 << FP_SHIFT));
    const int32_t ifx_q = FP_ONE - fx_q;
    const int32_t ify_q = FP_ONE - fy_q;
    const int32_t w00 = mul_q16(ifx_q, ify_q);
    const int32_t w10 = mul_q16(fx_q,  ify_q);
    const int32_t w01 = mul_q16(ifx_q, fy_q);
    const int32_t w11 = mul_q16(fx_q,  fy_q);
    const int cyx[4][2] = { {y0, x0}, {y0, x1}, {y1, x0}, {y1, x1} };
    const int32_t cw[4] = { w00, w10, w01, w11 };

    int64_t acc = 0;
    int32_t wsum_q = 0;
    for (int k = 0; k < 4; ++k) {
        const int cy_ = cyx[k][0];
        const int cx_ = cyx[k][1];
        const int j = cy_ * w + cx_;
        if (solid[j] || perm[j] <= 0.0f) continue;
        const int32_t val_q = is_vacuum[j] ? 0 : src[j];
        acc += mul_wide(cw[k], val_q);
        wsum_q += cw[k];
    }
    const int32_t WSUM_EPS_Q = FP_ONE >> 14;
    if (wsum_q <= WSUM_EPS_Q) return src[y * w + x];
    const int32_t WSUM_FLOOR_Q = FP_ONE >> 8;
    const int32_t wsum_clamped = (wsum_q < WSUM_FLOOR_Q) ? WSUM_FLOOR_Q : wsum_q;
    const int32_t recip_q = reciprocal_q16(wsum_clamped);
    const int32_t acc_q = narrow(acc);
    return mul_q16(acc_q, recip_q);
}

}  // namespace

void EOSSolver::step(
        int32_t* atmosphere,
        int32_t* p_prev,
        int32_t* wind_x, int32_t* wind_y,
        int32_t* temperature,
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int h, int w, float dt) const {

    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return;

    // Resize scratch (no-op after the first call at a given grid size).
    if ((int)n_total_.size() != n) n_total_.assign(n, 0);
    if ((int)vx_src_.size()  != n) vx_src_.assign(n, 0);
    if ((int)vy_src_.size()  != n) vy_src_.assign(n, 0);
    if ((int)t_src_.size()   != n) t_src_.assign(n, 0);
    if ((int)pstar_.size()   != n) pstar_.assign(n, 0);
    if ((int)dinv_.size()    != n) dinv_.assign(n, 0);
    if ((int)p_new_.size()   != n) p_new_.assign(n, 0);
    if ((int)div_u_.size()   != n) div_u_.assign(n, 0);

    // ---- step 0: P_prev := P (kept copy) ---------------------------------
    for (int i = 0; i < n; ++i) p_prev[i] = atmosphere[i];

    // ---- load-time (per-tick) scalar constants, folded in DOUBLE THEN
    // quantized (the S1 idiom): every combined scalar below is chosen so its
    // REAL VALUE fits Q16.16's +/-32768 range even though its FACTORS (N up
    // to 200x ambient, c_max^2 = 90000) individually do NOT — c and N are
    // NEVER quantized standalone; only c-dt / c-dt^2 combinations are. ------
    const q16 n_floor_q  = quantize((double)N_FLOOR_SOLVER);
    const q16 t_amb_q    = quantize((double)T_AMB_K);
    const q16 t_min_q    = quantize((double)T_MIN);
    const q16 c_q        = quantize((double)C);
    const q16 gamma_m1_q = quantize((double)gamma - 1.0);
    const double dt_d    = (double)dt;
    const double dx_d    = std::max((double)dx, 1e-6);
    const q16 inv_2dx_q  = quantize(1.0 / (2.0 * dx_d));       // central-diff scale
    // c^2*dt  (RHS div(u*) coefficient's per-N multiplier, §3.2 RHS term) and
    // c^2*dt^2/dx^2 (the Helmholtz face-coefficient scale, §3.4's "k_f <=
    // 2*c^2*dt^2/dx^2 ~= 11,180 at c=300" budget) — BOTH combined-at-double
    // scalars, safely within Q16.16 (verified against the design's own
    // numeric budget at the bench's dt=0.083/dx=1/3).
    const double c2 = (double)c_max * (double)c_max;
    const q16 c2dt_q      = quantize(c2 * dt_d);
    const q16 c2dt2_dx2_q = quantize(c2 * dt_d * dt_d / (dx_d * dx_d));

    // ======================================================================
    // 1. ADVECTION SUBSTEPS — n = ceil(dt/dt_adv), N_SUB_MAX-capped.
    // ======================================================================
    // u_est = min(max|u| + (max|grad P_prev|/N_hat)*dt, c_max), per §3.2.
    // max|u| — order-free integer max over sqrt_q16(vx^2+vy^2).
    q16 max_u = 0;
    for (int i = 0; i < n; ++i) {
        const int64_t rad = mul_wide(wind_x[i], wind_x[i]) + mul_wide(wind_y[i], wind_y[i]);
        const q16 mag = sqrt_q16(rad);
        if (mag > max_u) max_u = mag;
    }
    // A quick N_total snapshot from the CURRENT (pre-substep) gas state, for
    // the CFL estimate only (the post-substep N_total computed later at
    // step 2 is the one that feeds p*/Helmholtz).
    for (int i = 0; i < n; ++i) {
        int64_t sum = 0;
        for (int gi = 0; gi < n_gases; ++gi) sum += (int64_t)gas[(size_t)gi * n + i];
        n_total_[i] = (int32_t)sum;
    }
    // max over the grid of |grad(P_prev)| / N_hat_i.
    q16 max_gradP_over_N = 0;
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) continue;
            const int il = mirror_idx(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx(i, y - 1, x, h, w, solid);
            const int id = mirror_idx(i, y + 1, x, h, w, solid);
            const q16 gx = mul_q16(p_prev[ir] - p_prev[il], inv_2dx_q);
            const q16 gy = mul_q16(p_prev[id] - p_prev[iu], inv_2dx_q);
            const int64_t rad = mul_wide(gx, gx) + mul_wide(gy, gy);
            const q16 gmag = sqrt_q16(rad);
            q16 nhat = n_total_[i];
            if (nhat < n_floor_q) nhat = n_floor_q;
            const q16 inv_n = reciprocal_q16(nhat);
            const q16 ratio = mul_q16(gmag, inv_n);
            if (ratio > max_gradP_over_N) max_gradP_over_N = ratio;
        }
    }
    q16 u_est = max_u + mul_q16(max_gradP_over_N, quantize(dt_d));
    const q16 c_max_q = quantize((double)c_max);
    if (u_est > c_max_q) u_est = c_max_q;
    // dt_adv = CFL_ADV*dx/(u_est+eps). n = ceil(dt/dt_adv)
    //        = ceil(dt*(u_est+eps)/(CFL_ADV*dx)).
    const q16 eps_q = 1;   // 1 count (~1.5e-5), avoids a zero divisor at rest
    const q16 u_est_eps = u_est + eps_q;
    const q16 cfl_dx_q = quantize((double)CFL_ADV * dx_d);
    const q16 dt_q = quantize(dt_d);
    const int64_t numer_wide = mul_wide(dt_q, u_est_eps);   // Q.32
    const q16 numer_q = narrow(numer_wide);                  // Q16.16
    int n_sub = std::max(1, ceil_div(numer_q, cfl_dx_q));
    if (n_sub > N_SUB_MAX) n_sub = N_SUB_MAX;
    const double dt_s_d = dt_d / (double)n_sub;

    for (int s = 0; s < n_sub; ++s) {
        const float dt_s = (float)dt_s_d;
        const q16 dt_s_q = quantize(dt_s_d);

        // -- a. self-advect u (SL, snapshot then backtrace by u*dt_s) --------
        for (int i = 0; i < n; ++i) { vx_src_[i] = wind_x[i]; vy_src_[i] = wind_y[i]; }
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                if (solid[i]) { wind_x[i] = 0; wind_y[i] = 0; continue; }
                const int32_t bx_q = -mul_q16(vx_src_[i], dt_s_q);
                const int32_t by_q = -mul_q16(vy_src_[i], dt_s_q);
                wind_x[i] = eos_backtrace_sample_q(vx_src_.data(), x, y, bx_q, by_q,
                                                   solid, is_vacuum, dyn_permeability, h, w);
                wind_y[i] = eos_backtrace_sample_q(vy_src_.data(), x, y, bx_q, by_q,
                                                   solid, is_vacuum, dyn_permeability, h, w);
            }
        }

        // -- b. T <- SL advection (open-air / gas mask only) -----------------
        for (int i = 0; i < n; ++i) t_src_[i] = temperature[i];
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                if (solid[i]) continue;               // solid untouched (conduction owns it)
                if (is_vacuum[i]) { temperature[i] = 0; continue; }   // energy leaves w/ the gas
                const int32_t bx_q = -mul_q16(wind_x[i], dt_s_q);
                const int32_t by_q = -mul_q16(wind_y[i], dt_s_q);
                temperature[i] = eos_backtrace_sample_q(t_src_.data(), x, y, bx_q, by_q,
                                                        solid, is_vacuum, dyn_permeability, h, w);
            }
        }
        if (s == n_sub - 1) digest_advect = digest_of(wind_x, n, digest_of(wind_y, n, digest_of(temperature, n, 0)));

        // -- d. bulk O2/N2 <- donor-cell conservative flux on u --------------
        bulk_flux_transport(gas, gas_conservative, n_gases,
                            wind_x, wind_y, solid, is_vacuum,
                            dyn_permeability, h, w, dt_s);
        if (s == n_sub - 1) {
            uint64_t bfd = 0;
            for (int gi = 0; gi < n_gases; ++gi)
                bfd = digest_of(gas + (size_t)gi * n, n, bfd);
            digest_bulk_flux = bfd;
        }

        // -- e. compression work: T -= (gamma-1)*T*div(u)*dt_s ---------------
        // div(u)_i = (u_x[i+1]-u_x[i-1])/(2dx) + (u_y[i+w]-u_y[i-w])/(2dx),
        // Neumann-mirrored at solid.
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (solid[i] || is_vacuum[i]) { div_u_[i] = 0; continue; }
                const int il = mirror_idx(i, y, x - 1, h, w, solid);
                const int ir = mirror_idx(i, y, x + 1, h, w, solid);
                const int iu = mirror_idx(i, y - 1, x, h, w, solid);
                const int id = mirror_idx(i, y + 1, x, h, w, solid);
                const q16 dux = mul_q16(wind_x[ir] - wind_x[il], inv_2dx_q);
                const q16 duy = mul_q16(wind_y[id] - wind_y[iu], inv_2dx_q);
                div_u_[i] = dux + duy;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (solid[i] || is_vacuum[i]) continue;
            // (gamma-1)*div(u)*dt_s, then *T, all Q16.16 (PINNED left-fold,
            // the fire_simulation.cpp idiom): CFL_ADV<=0.5 pins
            // |(gamma-1)*div(u)*dt_s| <= (gamma-1)*2*CFL_ADV <= 0.4 < 1 (D3).
            q16 k = mul_q16(gamma_m1_q, div_u_[i]);
            k = mul_q16(k, dt_s_q);
            const q16 dT = mul_q16(k, temperature[i]);
            q16 t_new = temperature[i] - dT;
            if (t_new < t_min_q) { t_new = t_min_q; ++energy_floor_hits; }
            temperature[i] = t_new;
        }
        if (s == n_sub - 1) digest_compression = digest_of(temperature, n, 0);

        // -- f. zero u on solid (N's clamp already ran inside (d)) -----------
        for (int i = 0; i < n; ++i) {
            if (solid[i]) { wind_x[i] = 0; wind_y[i] = 0; }
        }
    }

    // ======================================================================
    // 2. p* := C * N_total * (T + T_AMB_K)      (wide mul, §3.4)
    // ======================================================================
    for (int i = 0; i < n; ++i) {
        int64_t sum = 0;
        for (int gi = 0; gi < n_gases; ++gi) sum += (int64_t)gas[(size_t)gi * n + i];
        n_total_[i] = (int32_t)sum;
    }
    for (int i = 0; i < n; ++i) {
        if (solid[i] || is_vacuum[i]) { pstar_[i] = 0; continue; }
        const int64_t t_abs_wide = (int64_t)temperature[i] + (int64_t)t_amb_q;
        const q16 t_abs = (q16)t_abs_wide;   // in-range (T_rel is small; +T_AMB_K < 32768)
        // p* = C*N*T_abs: C~3.4e-3 is tiny, so mul_q16(C,N) stays small even
        // at N=200 (~0.69) — safe to narrow once here, then again by T_abs.
        const q16 cn = mul_q16(c_q, n_total_[i]);      // C*N  (wide-then-narrow, x1)
        pstar_[i] = mul_q16(cn, t_abs);                // *T_abs (wide-then-narrow, x2)
    }
    digest_pstar = digest_of(pstar_.data(), n, 0);

    // ======================================================================
    // 3. HELMHOLTZ SOLVE — fixed S sweeps, RB-GS, wide int64 (§3.4).
    //    [I - (Nc^2)_i dt^2 . div( (1/N_hat_f) . grad )] P = p* - (Nc^2)_i dt div(u*)
    // ======================================================================
    // RHS = p* - N_i*c^2*dt*div(u*)_i. N_i*c2dt_q is formed WIDE (mul_wide;
    // Q.32-scale int64, safely representing values far past q16's +-32768
    // ceiling — e.g. N=200, c2dt_q~3753 -> real ~750600) and combined with
    // div(u*)_i via ONE truncation (wide_mul_q16, the §3.4 rule-1 idiom) —
    // N and c^2*dt are NEVER individually narrowed to a standalone q16.
    for (int i = 0; i < n; ++i) {
        if (solid[i] || is_vacuum[i]) { p_new_[i] = 0; continue; }
        const int64_t n_c2dt_wide = mul_wide(n_total_[i], c2dt_q);   // Q.32
        const q16 term = wide_mul_q16(n_c2dt_wide, div_u_[i]);
        p_new_[i] = pstar_[i] - term;
    }
    std::vector<int32_t>& rhs = pstar_;   // pstar_ no longer needed raw; alias as RHS store
    for (int i = 0; i < n; ++i) rhs[i] = p_new_[i];

    // Per-cell face weights k_f = c2dt2_dx2_q * (N_cell/N_hat_face) * perm_f.
    // N_cell/N_hat_face <= 2 ALWAYS (N_hat_face is the arithmetic mean of two
    // cells, one of which is N_cell) -> the ratio is a SAFE q16 value even
    // though N_cell/c^2 individually are not; k_f itself then fits Q16.16
    // (<= 2*c2dt2_dx2_q, ~11,180 max at c=300/the bench dt/dx -- the design's
    // own closed overflow budget, §3.4). Diagonal d = 1 + sum(k_f); Dinv =
    // reciprocal_q16(d), precomputed ONCE per cell per tick (§3.4
    // amortization — constant across the S sweeps within this tick).
    auto face_k = [&](int i, int nb_i, int ny, int nx) -> q16 {
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) return 0;
        if (solid[nb_i]) return 0;
        const float perm_f = std::min(dyn_permeability[i], dyn_permeability[nb_i]);
        if (perm_f <= 0.0f) return 0;
        q16 nhat_face = (n_total_[i] + n_total_[nb_i]) >> 1;
        if (nhat_face < n_floor_q) nhat_face = n_floor_q;
        q16 n_i = n_total_[i];
        if (n_i < n_floor_q) n_i = n_floor_q;
        const q16 inv_nhat = reciprocal_q16(nhat_face);
        const q16 ratio = mul_q16(n_i, inv_nhat);          // N_cell/N_hat_face, <= ~2.0
        const q16 perm_q = quantize((double)perm_f);
        return mul_q16(mul_q16(c2dt2_dx2_q, ratio), perm_q);
    };

    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i] || is_vacuum[i]) { dinv_[i] = 0; continue; }
            const int iu = (y > 0)   ? row - w + x : -1;
            const int id = (y < h-1) ? row + w + x : -1;
            const int il = (x > 0)   ? row + x - 1 : -1;
            const int ir = (x < w-1) ? row + x + 1 : -1;
            q16 wsum = 0;
            if (y > 0)   wsum += face_k(i, iu, y - 1, x);
            if (y < h-1) wsum += face_k(i, id, y + 1, x);
            if (x > 0)   wsum += face_k(i, il, y, x - 1);
            if (x < w-1) wsum += face_k(i, ir, y, x + 1);
            const q16 denom = FP_ONE + wsum;
            dinv_[i] = reciprocal_q16(denom);
        }
    }

    // P_new starts from p* - beta*div(u*) (== rhs); GS relaxes toward the
    // implicit solution in RESIDUAL form (the drift-free fixed point, same
    // shape as AtmosphereSolver's RB-GS).
    for (int i = 0; i < n; ++i) p_new_[i] = rhs[i];
    const int64_t HALF_Q = (int64_t)1 << (FP_SHIFT - 1);
    for (int iter = 0; iter < S; ++iter) {
        for (int color = 0; color < 2; ++color) {
            for (int y = 0; y < h; ++y) {
                const int row = y * w;
                for (int x = 0; x < w; ++x) {
                    if (((x + y) & 1) != color) continue;
                    const int i = row + x;
                    if (solid[i] || is_vacuum[i]) continue;
                    const q16 pi = p_new_[i];
                    int64_t acc = 0;
                    if (y > 0)   { const int nb = row - w + x; acc += mul_wide(face_k(i, nb, y-1, x), p_new_[nb] - pi); }
                    if (y < h-1) { const int nb = row + w + x; acc += mul_wide(face_k(i, nb, y+1, x), p_new_[nb] - pi); }
                    if (x > 0)   { const int nb = row + x - 1; acc += mul_wide(face_k(i, nb, y, x-1), p_new_[nb] - pi); }
                    if (x < w-1) { const int nb = row + x + 1; acc += mul_wide(face_k(i, nb, y, x+1), p_new_[nb] - pi); }
                    const q16 flux = narrow(acc);
                    const q16 resi = flux - (pi - rhs[i]);
                    const int64_t inc_wide = (int64_t)resi * (int64_t)dinv_[i];
                    const q16 inc = (q16)((inc_wide >= 0)
                        ? ((inc_wide + HALF_Q) >> FP_SHIFT)
                        : -(((-inc_wide) + HALF_Q) >> FP_SHIFT));
                    p_new_[i] = pi + inc;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (is_vacuum[i]) p_new_[i] = 0;
        if (solid[i]) p_new_[i] = 0;
    }
    digest_helmholtz = digest_of(p_new_.data(), n, 0);

    // ======================================================================
    // 4. u -= dt*grad(P_new)/N_hat; absorption damping; zero outside open-air.
    // ======================================================================
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i] || is_vacuum[i]) { wind_x[i] = 0; wind_y[i] = 0; continue; }
            const int il = mirror_idx(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx(i, y - 1, x, h, w, solid);
            const int id = mirror_idx(i, y + 1, x, h, w, solid);
            const q16 gx = mul_q16(p_new_[ir] - p_new_[il], inv_2dx_q);
            const q16 gy = mul_q16(p_new_[id] - p_new_[iu], inv_2dx_q);
            q16 nhat = n_total_[i];
            if (nhat < n_floor_q) nhat = n_floor_q;
            const q16 inv_n = reciprocal_q16(nhat);
            // dt*grad*inv_n via the wide-then-single-narrow idiom (a near-
            // vacuum N_FLOOR_SOLVER cell can have a large 1/N_hat — the
            // venting kick is SUPPOSED to be large there).
            const int64_t dtgx_wide = mul_wide(dt_q, gx);
            const int64_t dtgy_wide = mul_wide(dt_q, gy);
            const q16 dux = wide_mul_q16(dtgx_wide, inv_n);
            const q16 duy = wide_mul_q16(dtgy_wide, inv_n);
            wind_x[i] -= dux;
            wind_y[i] -= duy;

            // absorption damping: u *= (1 - absorb*dt), D4.
            const q16 a = mul_q16(quantize((double)dyn_wave_absorb[i]),
                                  quantize((double)absorb_strength * dt_d));
            if (a > 0) {
                const q16 k = (a < FP_ONE) ? (q16)(FP_ONE - a) : 0;
                wind_x[i] = scale_mag(wind_x[i], k);
                wind_y[i] = scale_mag(wind_y[i], k);
            }
        }
    }
    digest_velocity = digest_of(wind_x, n, digest_of(wind_y, n, 0));

    // ======================================================================
    // 5. P := P_new — materialized ONCE, stored (the `atmosphere` alias).
    // ======================================================================
    for (int i = 0; i < n; ++i) atmosphere[i] = p_new_[i];
}
