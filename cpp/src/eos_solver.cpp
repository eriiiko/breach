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
// Sequential, order-DEPENDENT (CPU-only lockstep gate; a GPU port needs its
// own order-free reduction — P6). Deterministic bit-for-bit (pure integer).
uint64_t digest_of(const int32_t* buf, int n, uint64_t seed) {
    uint64_t h = seed ^ 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint32_t)buf[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ---- 128-bit staged multiply: (a·b) >> shift, both int64 ------------------
// THE v2.2 wide idiom (§3.4 rules 1 + 4b + the joint-case ordering note):
// every coefficient×field product in the solve goes through a 128-bit
// intermediate and one arithmetic shift — no int64 product is ever formed
// raw×raw. MSVC path mirrors fixed_point.h's recip_mul _mul128 idiom.
#if defined(__SIZEOF_INT128__)
inline int64_t mul128_shr(int64_t a, int64_t b, int shift) {
    return (int64_t)(((__int128)a * (__int128)b) >> shift);
}
#else
inline int64_t mul128_shr(int64_t a, int64_t b, int shift) {
    long long hi;
    long long lo = _mul128((long long)a, (long long)b, &hi);
    unsigned long long ulo = (unsigned long long)lo;
    return (int64_t)((ulo >> shift) | ((unsigned long long)hi << (64 - shift)));
}
#endif

// ---- solid-mirror neighbor read (Neumann BC helper) ----------------------
inline int mirror_idx(int self_i, int ny, int nx, int h, int w, const bool* solid) {
    if (ny < 0 || ny >= h || nx < 0 || nx >= w) return self_i;
    const int ni = ny * w + nx;
    if (solid[ni]) return self_i;
    return ni;
}

// (the single-field eos_backtrace_sample_q was deleted — the FUSED
// 3-field version below replaced its every call site.)

// ---- FUSED 3-field backtrace (perf: the substep loop is the tick's cost
// whale at 160² — one DDA march + one bilinear weight set is shared by all
// three advected fields (vx, vy, T), which by construction ride the SAME
// displacement −u·dt_s from the same cell; computing the march thrice was
// pure waste). Also two fast paths: zero displacement returns the source
// values outright, and the all-open-corner case (wsum within 4 counts of
// 1.0) skips the Newton renorm — the ≤6e-5 relative decay it accepts is far
// below the already-accepted sample-truncation decay. Both paths are
// deterministic (documented behavior, not rounding drift).
struct FusedSample { int32_t vx, vy, t; };
// cmask (built ONCE per tick from solid/is_vacuum/perm — all constant within
// a tick; a pure table-lookup re-expression of the original float/bool
// predicate chain, BIT-IDENTITY-PRESERVING):
//   0 = sealed  (solid || perm <= 0)              — wall to the march, dead corner
//   1 = breach  (vacuum, open)                    — march target, zero-valued corner
//   2 = live    (open air)                        — regular corner
FusedSample eos_backtrace_sample3_q(
        const int32_t* src_vx, const int32_t* src_vy, const int32_t* src_t,
        int x, int y, int32_t bx_q, int32_t by_q,
        const uint8_t* cmask, int h, int w) {
    const int i0 = y * w + x;
    if (bx_q == 0 && by_q == 0) {
        return { src_vx[i0], src_vy[i0], src_t[i0] };
    }
    int64_t px_q = ((int64_t)x << FP_SHIFT) + bx_q;
    int64_t py_q = ((int64_t)y << FP_SHIFT) + by_q;

    const int32_t abx = bx_q >= 0 ? bx_q : -bx_q;
    const int32_t aby = by_q >= 0 ? by_q : -by_q;
    const int32_t amax = abx >= aby ? abx : aby;
    int n_steps = amax >> FP_SHIFT;
    if (amax & (FP_ONE - 1)) n_steps += 1;

    // Original predicate: wall == OOB || solid || perm<=0 || (vacuum-sealed);
    // a vacuum && open cell is a BREACH (not a wall). Table form: wall <=>
    // OOB || cmask == 0 for the sealed set, breach <=> cmask == 1 — the
    // vacuum&&solid / vacuum&&perm<=0 combinations land in cmask 0 exactly
    // as the original chain classified them.
    auto solid_wall_at = [&](int ty, int tx) -> bool {
        if (ty < 0 || ty >= h || tx < 0 || tx >= w) return true;
        const int i = ty * w + tx;
        if (cmask[i] == 1) return false;   // breach: march may enter
        return cmask[i] == 0 || false;     // sealed: wall
    };

    if (n_steps > 0) {
        auto floordiv = [](int32_t a, int b) -> int32_t {
            return (a >= 0) ? (a / b) : -(((-(int64_t)a) + b - 1) / b);
        };
        const int32_t sx_q = floordiv(bx_q, n_steps);
        const int32_t sy_q = floordiv(by_q, n_steps);
        int64_t cx_q = (int64_t)x << FP_SHIFT;
        int64_t cy_q = (int64_t)y << FP_SHIFT;
        for (int st = 0; st < n_steps; ++st) {
            const int64_t nxp_q = cx_q + sx_q;
            const int64_t nyp_q = cy_q + sy_q;
            const int ti = (int)((nxp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            const int tj = (int)((nyp_q + (FP_ONE >> 1)) >> FP_SHIFT);
            if (solid_wall_at(tj, ti)) break;
            cx_q = nxp_q;
            cy_q = nyp_q;
            // original: stop after stepping ONTO a vacuum tile. Any vacuum
            // tile the march can be standing on here is a BREACH (sealed
            // vacuum forms are walls and broke above) == cmask 1.
            if (tj >= 0 && tj < h && ti >= 0 && ti < w && cmask[tj * w + ti] == 1) break;
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

    int64_t ax = 0, ay = 0, at = 0;
    int32_t wsum_q = 0;
    for (int k = 0; k < 4; ++k) {
        const int j = cyx[k][0] * w + cyx[k][1];
        const uint8_t m = cmask[j];
        if (m == 0) continue;                              // sealed corner
        if (m == 1) { wsum_q += cw[k]; continue; }         // breach: value 0
        ax += mul_wide(cw[k], src_vx[j]);
        ay += mul_wide(cw[k], src_vy[j]);
        at += mul_wide(cw[k], src_t[j]);
        wsum_q += cw[k];
    }
    const int32_t WSUM_EPS_Q = FP_ONE >> 14;
    if (wsum_q <= WSUM_EPS_Q) {
        return { src_vx[i0], src_vy[i0], src_t[i0] };
    }
    if (wsum_q >= FP_ONE - 4) {   // all corners live: renorm ~= identity
        return { narrow(ax), narrow(ay), narrow(at) };
    }
    const int32_t WSUM_FLOOR_Q = FP_ONE >> 8;
    const int32_t wsum_clamped = (wsum_q < WSUM_FLOOR_Q) ? WSUM_FLOOR_Q : wsum_q;
    const int32_t recip_q = reciprocal_q16(wsum_clamped);
    return { mul_q16(narrow(ax), recip_q),
             mul_q16(narrow(ay), recip_q),
             mul_q16(narrow(at), recip_q) };
}

}  // namespace

// ===========================================================================
// The v2.2 pressure solve.
//
// PER-LEVEL OVERFLOW BUDGET (v2.2 §3.4 joint-case, closed by the
// N-CANCELLATION bound — the derivation the round-3 critique demanded):
//   row coefficient  aK_i = (γ·p*_i)·K·dt²/dx_L²   [int64 Q16.16 raw]
//   face conductance g_f  = perm_f / N̂_f           [q16 raw, N̂ floored]
//   face coupling    k_f  = aK_i·g_f
// Substituting p*_i = C·N_i·T_abs and N̂_f ≥ N_i/2 (arithmetic face mean):
//   k_f ≤ 2·γ·C·T_abs·K·dt²/dx_L²·perm  —  N CANCELS.
// At the worst joint spike (T_abs = 9300 K, bench dt=1/24, dx=1/3):
//   k_f ≤ 2·1.4·0.00345·9300·1006 ≈ 9.1e4 real (raw ≤ 6.0e9, int64-only),
// so k_f·ΔP with ΔP ≤ ~6,500 atm (raw 4.3e8) peaks ≈ 2.6e18 inside the
// 128-bit intermediate, narrowing (>>16) to ≤ 4.0e13 raw per face and
// ≤ 1.6e14 for the 4-face sum — 15+ bits of int64 headroom. The N-vs-N̂
// cancellation is why the "N=200 tank against a floored face" pathological
// case cannot occur: the floor only wins where BOTH neighbors are
// near-vacuum, where p* (hence aK) is near zero too (§3.1 property 2).
// Coarse levels only SHRINK the bound (aK scales /4 per level; g averages).
// CHOSEN OPERATION ORDER (documented per the spec's joint-case note): the
// 1/N̂ divide is folded into g BEFORE any multiply by P — the product chain
// is aK (one 128-bit stage) → ×g (128-bit) → ×ΔP (128-bit).
//
// DIAGONAL RECIPROCAL (§3.4 rule 1): d_raw = 2^16 + Σk_f_raw can reach
// ~2.4e14 raw — far past reciprocal_q16's domain, and a q16 reciprocal of
// even the AMBIENT diagonal (~5,600 real) carries only ~11 counts (~3.5
// significant bits), which the P3 gate measured as a ±0.04 atm equilibrium
// error band (the S=512 "slow creep"). The v2.2 form: one wide integer
// divide per cell per tick per level, recip = 2^48/d_raw (a Q.32
// reciprocal), applied as inc = (resi·recip) >> 32 through 128 bits.
// ===========================================================================

void EOSSolver::step(
        int32_t* atmosphere,
        int32_t* p_prev,
        int32_t* wind_x, int32_t* wind_y,
        int32_t* temperature,
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int h, int w, float dt,
        const bool* is_ambient, const int32_t* n_amb, int32_t p_amb,
        const int32_t* sponge_sigma, const int32_t* sponge_udamp) const {

    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return;

    // BC (boundary_conditions_spec_2026-07-19): planetside AMBIENT ring. ONE
    // flag gates every ambient edit in this function — a space map passes
    // is_ambient == nullptr and takes the byte-identical path (dormancy BY
    // BRANCH, spec §5; NO unconditional arithmetic change on the space path).
    const bool ambient_mode = (is_ambient != nullptr);
    // The boundary_flux rail (spec §5): zero it each tick in ambient mode; the
    // per-substep bulk reset accumulates into it. Empty on space maps.
    if (ambient_mode) {
        if ((int)boundary_flux_.size() != n_gases) boundary_flux_.assign(n_gases, 0);
        else std::fill(boundary_flux_.begin(), boundary_flux_.end(), (int64_t)0);
    } else if (!boundary_flux_.empty()) {
        boundary_flux_.clear();
    }

    if ((int)n_total_.size() != n) n_total_.assign(n, 0);
    if ((int)vx_src_.size()  != n) vx_src_.assign(n, 0);
    if ((int)vy_src_.size()  != n) vy_src_.assign(n, 0);
    if ((int)t_src_.size()   != n) t_src_.assign(n, 0);
    if ((int)pstar_.size()   != n) pstar_.assign(n, 0);
    if ((int)div_u_.size()   != n) div_u_.assign(n, 0);
    if ((int)cmask_.size()   != n) cmask_.assign(n, 0);
    if ((int)coeffE_.size()  != n) coeffE_.assign(n, 0);
    if ((int)coeffS_.size()  != n) coeffS_.assign(n, 0);

    // ---- step 0: P_prev := P ---------------------------------------------
    for (int i = 0; i < n; ++i) p_prev[i] = atmosphere[i];

    // DEBUG probe (temporary): T at step-1 entry.
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_pre_advect = temperature[dbg_probe_idx];

    // ---- per-tick scalar constants (double-fold once, then quantize) ------
    const q16 n_floor_q  = quantize((double)N_FLOOR_SOLVER);
    const q16 t_amb_q    = quantize((double)T_AMB_K);
    const q16 t_min_q    = quantize((double)T_MIN);
    const q16 t_max_phys_q = quantize((double)T_MAX_PHYS);   // v2.4 rail (see eos_solver.h)
    const q16 u_max_q      = quantize((double)U_MAX);        // v2.4 rail
    const q16 c_q        = quantize((double)C);
    const double gamma_d = (double)adiabatic_index;
    const q16 gamma_m1_q = quantize(gamma_d - 1.0);
    const double dt_d    = (double)dt;
    const q16 dt_q       = quantize(dt_d);
    const double dx_d    = std::max((double)dx, 1e-6);
    const q16 inv_2dx_q  = quantize(1.0 / (2.0 * dx_d));
    // v2.2 D-A: K = c_amb²/γ — the ONE unit bridge. WIDE int64 (real ≈64,286
    // does not fit q16's ±32768 value range — design §3.2 step 4).
    const double K_d = (double)c_max * (double)c_max / gamma_d;
    const int64_t K_raw = (int64_t)(K_d * 65536.0 + 0.5);
    // K·dt (momentum-kick stage 1, real ≈2,680 at the bench dt).
    const int64_t Kdt_raw = mul128_shr(K_raw, (int64_t)dt_q, 16);

    // ---- c_LOCAL = c_amb·sqrt(T_max_abs/T_AMB) (v2.2: c is state-derived) --
    // Per-tick max of T_abs over open air; one sqrt_q16. A stale ambient cap
    // would re-create the under-substep failure the CFL ∇P term prevents
    // (a 9000 K core's sound speed is ~5.6× ambient).
    int64_t t_max_abs_raw = (int64_t)t_amb_q;
    for (int i = 0; i < n; ++i) {
        if (solid[i] || is_vacuum[i]) continue;
        const int64_t t_abs = (int64_t)temperature[i] + (int64_t)t_amb_q;
        if (t_abs > t_max_abs_raw) t_max_abs_raw = t_abs;
    }
    const q16 c_amb_q = quantize((double)c_max);
    const q16 absorb_dt_q = quantize((double)absorb_strength * dt_d);
    const int32_t ratio_q = (int32_t)((t_max_abs_raw << 16) / (int64_t)t_amb_q);
    const q16 sqrt_ratio = sqrt_q16((int64_t)ratio_q << 16);   // Q.32 radicand
    q16 c_local_q = mul_q16(c_amb_q, sqrt_ratio);
    if (c_local_q < c_amb_q) c_local_q = c_amb_q;   // never below ambient
    dbg_last_c_local_q = c_local_q;   // P6.4 telemetry (gate input reconstruction)

    // ======================================================================
    // 1. ADVECTION SUBSTEPS — n = ceil(dt/dt_adv), N_SUB_MAX-capped.
    //    u_est = max|u| + max(K·|∇P|/N̂)·dt, capped at c_LOCAL (§3.2 v2.2 —
    //    the ∇P term goes through the SAME unit bridge K as the kick).
    // ======================================================================
    // max|u| — micro-opt (BIT-IDENTITY-PRESERVING): sqrt_q16 is monotone
    // non-decreasing, so max_i sqrt(rad_i) == sqrt(max_i rad_i) — ONE
    // 32-iteration isqrt instead of 25k of them per tick.
    int64_t max_rad = 0;
    for (int i = 0; i < n; ++i) {
        const int64_t rad = mul_wide(wind_x[i], wind_x[i]) + mul_wide(wind_y[i], wind_y[i]);
        if (rad > max_rad) max_rad = rad;
    }
    const q16 max_u = sqrt_q16(max_rad);
    // Dalton sum with the trace-mass calibration (see eos_solver.h): bulk
    // planes at full weight, trace planes scaled by trace_mass_scale.
    {
        const q16 tms_q = quantize((double)trace_mass_scale);
        for (int i = 0; i < n; ++i) n_total_[i] = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            const int32_t* plane = gas + (size_t)gi * n;
            if (gas_conservative[gi]) {
                for (int i = 0; i < n; ++i) n_total_[i] += plane[i];
            } else {
                for (int i = 0; i < n; ++i) n_total_[i] += mul_q16(tms_q, plane[i]);
            }
        }
    }
    int64_t max_du_raw = 0;   // max K·|∇P|·dt/N̂ over the grid (int64 raw)
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) continue;
            const int il = mirror_idx(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx(i, y - 1, x, h, w, solid);
            const int id = mirror_idx(i, y + 1, x, h, w, solid);
            // |∇P| components kept WIDE (a q16-narrowed gradient could wrap
            // at spike-adjacent cells).
            const int64_t gx = mul128_shr((int64_t)(p_prev[ir] - p_prev[il]),
                                          (int64_t)inv_2dx_q, 16);
            const int64_t gy = mul128_shr((int64_t)(p_prev[id] - p_prev[iu]),
                                          (int64_t)inv_2dx_q, 16);
            const int64_t agx = gx < 0 ? -gx : gx;
            const int64_t agy = gy < 0 ? -gy : gy;
            const int64_t gmag = agx > agy ? agx : agy;   // Chebyshev bound (cheap)
            if (gmag == 0) continue;   // micro-opt: du would be exactly 0 —
                                       // cannot raise the max (bit-identical)
            q16 nhat = n_total_[i];
            if (nhat < n_floor_q) nhat = n_floor_q;
            const q16 inv_n = reciprocal_q16(nhat);
            // du = (K·dt)·|∇P|·(1/N̂) — staged 128-bit, the documented order.
            const int64_t t1 = mul128_shr(Kdt_raw, gmag, 16);
            const int64_t du = mul128_shr(t1, (int64_t)inv_n, 16);
            if (du > max_du_raw) max_du_raw = du;
        }
    }
    int64_t u_est_raw = (int64_t)max_u + max_du_raw + 1;   // +1 count eps
    if (u_est_raw > (int64_t)c_local_q) u_est_raw = (int64_t)c_local_q;
    const q16 cfl_dx_q = quantize((double)CFL_ADV * dx_d);
    const int64_t numer_wide = mul128_shr((int64_t)dt_q, u_est_raw, 16);
    int n_sub = std::max(1, ceil_div(
        (q16)std::min<int64_t>(numer_wide, (int64_t)INT32_MAX), cfl_dx_q));
    if (n_sub > N_SUB_MAX) n_sub = N_SUB_MAX;
    dbg_last_n_sub = n_sub;   // P6.2 telemetry (gate input reconstruction)
    const double dt_s_d = dt_d / (double)n_sub;

    // ---- per-tick caches for the substep loop (micro-opt, all
    // BIT-IDENTITY-PRESERVING: solid/is_vacuum/dyn_permeability and dt_s are
    // constant within the tick; these tables re-express the same per-use
    // computations, evaluated once) --------------------------------------
    // corner/march mask for the fused SL sample:
    for (int i = 0; i < n; ++i) {
        if (solid[i] || dyn_permeability[i] <= 0.0f) cmask_[i] = 0;
        // BC (audit (b)): the ambient ring is a still boundary — a breach corner
        // to the SL march (zero-valued, march-target), exactly like vacuum.
        else if (is_vacuum[i] || (ambient_mode && is_ambient[i])) cmask_[i] = 1;
        else cmask_[i] = 2;
    }
    // donor-cell face coefficients (min-perm quantize x dt_s — the exact
    // legacy bulk_flux_transport per-face chain, hoisted):
    {
        const q16 dts_q_c = quantize(dt_s_d);
        for (int i = 0; i < n; ++i) { coeffE_[i] = 0; coeffS_[i] = 0; }
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (solid[i]) continue;
                if (x < w - 1 && !solid[i + 1]) {
                    const float ff = std::min(dyn_permeability[i], dyn_permeability[i + 1]);
                    if (ff > 0.0f) coeffE_[i] = mul_q16(quantize((double)ff), dts_q_c);
                }
                if (y < h - 1 && !solid[i + w]) {
                    const float ff = std::min(dyn_permeability[i], dyn_permeability[i + w]);
                    if (ff > 0.0f) coeffS_[i] = mul_q16(quantize((double)ff), dts_q_c);
                }
            }
        }
    }

    for (int s = 0; s < n_sub; ++s) {
        const float dt_s = (float)dt_s_d;
        const q16 dt_s_q = quantize(dt_s_d);

        // -- a+b. FUSED SL advection of u (self) and T ---------------------
        // All three fields ride the SAME displacement −uⁿ·dt_s (uⁿ = the
        // pre-substep velocity — the consistent SL choice, and what lets one
        // march serve three samples; T purely advected — no compression work
        // here, that is step 4c, design §3.2 corrected 2026-07-10).
        for (int i = 0; i < n; ++i) {
            vx_src_[i] = wind_x[i];
            vy_src_[i] = wind_y[i];
            t_src_[i]  = temperature[i];
        }
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                if (solid[i]) { wind_x[i] = 0; wind_y[i] = 0; continue; }
                const int32_t bx_q = -mul_q16(vx_src_[i], dt_s_q);
                const int32_t by_q = -mul_q16(vy_src_[i], dt_s_q);
                const FusedSample fs = eos_backtrace_sample3_q(
                    vx_src_.data(), vy_src_.data(), t_src_.data(),
                    x, y, bx_q, by_q,
                    cmask_.data(), h, w);
                wind_x[i] = fs.vx;
                wind_y[i] = fs.vy;
                // BC (audit (b)): ΔT=0 IS ambient — the ring holds T ≡ T_AMB_K
                // (stored ΔT 0), the vacuum idiom verbatim.
                temperature[i] = (is_vacuum[i] || (ambient_mode && is_ambient[i])) ? 0 : fs.t;
            }
        }
        if (s == n_sub - 1) digest_advect = digest_of(wind_x, n, digest_of(wind_y, n, digest_of(temperature, n, 0)));

        // -- d. bulk O2/N2 <- donor-cell conservative flux on u ------------
        // (cached-coefficient entry — the per-face min/quantize/mul chain is
        // hoisted to the per-tick cache above; arithmetic identical)
        // BC (spec §1 "N — sink becomes clamp, per substep"): the ambient ring
        // is clamped to N_amb[plane] every substep, and boundary_flux_ records
        // the exchange (spec §5). nullptr args on a space map -> byte-identical.
        bulk_flux_transport_cached(gas, gas_conservative, n_gases,
                                   wind_x, wind_y, solid, is_vacuum,
                                   coeffE_.data(), coeffS_.data(), h, w,
                                   ambient_mode ? is_ambient : nullptr,
                                   ambient_mode ? n_amb : nullptr,
                                   ambient_mode ? boundary_flux_.data() : nullptr);
        (void)dt_s;
        if (s == n_sub - 1) {
            uint64_t bfd = 0;
            for (int gi = 0; gi < n_gases; ++gi)
                bfd = digest_of(gas + (size_t)gi * n, n, bfd);
            digest_bulk_flux = bfd;
        }

        // -- f. zero u on solid --------------------------------------------
        for (int i = 0; i < n; ++i) {
            if (solid[i]) { wind_x[i] = 0; wind_y[i] = 0; }
        }
    }

    // DEBUG probe (temporary): T after the step-1 SL substep loop.
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_advect = temperature[dbg_probe_idx];

    // div(u*) from the final substep's velocity — the Helmholtz RHS term.
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            // BC (audit (b)): the ring is a Dirichlet boundary owned by the pin
            // — div(u*)=0 there, the vacuum idiom (its rhs isn't consumed since
            // the ring is excl==1, but keep it clean).
            if (solid[i] || is_vacuum[i] || (ambient_mode && is_ambient[i])) { div_u_[i] = 0; continue; }
            const int il = mirror_idx(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx(i, y - 1, x, h, w, solid);
            const int id = mirror_idx(i, y + 1, x, h, w, solid);
            const q16 dux = mul_q16(wind_x[ir] - wind_x[il], inv_2dx_q);
            const q16 duy = mul_q16(wind_y[id] - wind_y[iu], inv_2dx_q);
            div_u_[i] = dux + duy;
        }
    }

    // ======================================================================
    // 2. p* := C · N_total · (T + T_AMB_K)      (post-substep N, wide mul)
    // ======================================================================
    // Dalton sum with the trace-mass calibration (see eos_solver.h): bulk
    // planes at full weight, trace planes scaled by trace_mass_scale.
    {
        const q16 tms_q = quantize((double)trace_mass_scale);
        for (int i = 0; i < n; ++i) n_total_[i] = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            const int32_t* plane = gas + (size_t)gi * n;
            if (gas_conservative[gi]) {
                for (int i = 0; i < n; ++i) n_total_[i] += plane[i];
            } else {
                for (int i = 0; i < n; ++i) n_total_[i] += mul_q16(tms_q, plane[i]);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (solid[i] || is_vacuum[i]) { pstar_[i] = 0; continue; }
        if (debug_pstar_from_prev) {
            // MEASUREMENT-ONLY (see eos_solver.h): the paper's own p_a
            // structure — pressure evolves as its own state. Diagnostic.
            pstar_[i] = p_prev[i];
        } else {
            const int64_t t_abs_wide = (int64_t)temperature[i] + (int64_t)t_amb_q;
            const q16 t_abs = (q16)t_abs_wide;
            const q16 cn = mul_q16(c_q, n_total_[i]);
            pstar_[i] = mul_q16(cn, t_abs);
        }
        if (pstar_[i] < 0) pstar_[i] = 0;   // EOS floor (T_abs ≥ ~1 K by T_MIN)
    }
    digest_pstar = digest_of(pstar_.data(), n, 0);

    // ======================================================================
    // 3. PRESSURE SOLVE — fixed-schedule multigrid V-cycles (v2.2 D-B), or
    //    flat RB-GS on level 0 (use_multigrid=false — the MG measurement
    //    gate's A/B reference path). Operator per level (regular rows i):
    //      P_i + Σ_f k_f·(P_i − P_nb) = rhs_i,  k_f = aK_i·g_f
    //    Dirichlet (vacuum) cells: P pinned 0. Solid: excluded (faces into
    //    it carry g = 0 — the Neumann mirror). Near-vacuum rows: aK → 0 ⇒
    //    diag → 1, rhs → 0 ⇒ P → 0 — the §3.1-property-2 degeneracy, free.
    // ======================================================================

    // EOS P6.3: the solve body lives in mg_build_levels + mg_run_solve_cpu
    // (PURE CODE MOTION — identical arithmetic, identical order; the header
    // has the split rationale) so the CUDA port's host-side hierarchy build
    // and the standalone eos_mg_solve_reference drive the SAME routines as
    // this live path.
    // BC: the SHIFT (P′ = P − P_amb), the ring→Dirichlet excl, and the σ-sponge
    // (B3b) all live inside mg_build_levels, gated on is_ambient != nullptr.
    const int n_levels = mg_build_levels(pstar_.data(), div_u_.data(),
                                         n_total_.data(), p_prev,
                                         solid, is_vacuum, dyn_permeability,
                                         h, w, dt,
                                         ambient_mode ? is_ambient : nullptr,
                                         p_amb,
                                         ambient_mode ? sponge_sigma : nullptr);
    mg_run_solve_cpu(n_levels);
    MGLevel& L0 = levels_[0];

    // ======================================================================
    // 4. u -= dt·K·grad(P)/N̂ (v2.2: K MANDATORY — omitting it IS the
    //    64,000× unit bug). Whole chain int64; |u| clamped to c_LOCAL
    //    (scale-to-cap, counter-tracked); narrowed to q16 ONCE at store.
    // ======================================================================
    const int32_t* Pn = L0.P.data();
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            // BC (audit (b)): ring u ≡ 0 — a still boundary, the vacuum idiom.
            if (solid[i] || is_vacuum[i] || (ambient_mode && is_ambient[i])) { wind_x[i] = 0; wind_y[i] = 0; continue; }
            const int il = mirror_idx(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx(i, y - 1, x, h, w, solid);
            const int id = mirror_idx(i, y + 1, x, h, w, solid);
            const int64_t gx = mul128_shr((int64_t)(Pn[ir] - Pn[il]), (int64_t)inv_2dx_q, 16);
            const int64_t gy = mul128_shr((int64_t)(Pn[id] - Pn[iu]), (int64_t)inv_2dx_q, 16);
            int64_t ux = (int64_t)wind_x[i];
            int64_t uy = (int64_t)wind_y[i];
            if (gx != 0 || gy != 0) {   // micro-opt: du == 0 exactly at zero
                                        // gradient — skip the reciprocal
                                        // chain (bit-identical)
                q16 nhat = n_total_[i];
                if (nhat < n_floor_q) nhat = n_floor_q;
                const q16 inv_n = reciprocal_q16(nhat);
                // du = (K·dt)·∇P·(1/N̂) — staged 128-bit, the documented order.
                ux -= mul128_shr(mul128_shr(Kdt_raw, gx, 16), (int64_t)inv_n, 16);
                uy -= mul128_shr(mul128_shr(Kdt_raw, gy, 16), (int64_t)inv_n, 16);
            }

            // absorption damping u *= (1 − absorb·dt) (D4) — on the wide
            // value, magnitude-first (sign-symmetric shrink, scale_mag's idiom).
            // (absorb_dt_q hoisted — a per-tick constant, was quantized per cell)
            const q16 a = mul_q16(quantize((double)dyn_wave_absorb[i]), absorb_dt_q);
            if (a > 0) {
                const q16 kk = (a < FP_ONE) ? (q16)(FP_ONE - a) : 0;
                const int64_t mx = mul128_shr(ux < 0 ? -ux : ux, (int64_t)kk, 16);
                const int64_t my = mul128_shr(uy < 0 ? -uy : uy, (int64_t)kk, 16);
                ux = (ux < 0) ? -mx : mx;
                uy = (uy < 0) ? -my : my;
            }

            // BC (spec §3 rung 2, B3c): the u-DAMPING BAND — the real absorber.
            // A second magnitude-first shrink u *= (1 − k(d)) using the static
            // band coefficient k(d) (Q16, tapered k_max at the ring to 0 at the
            // inner band edge), placed immediately after the absorb chain. This
            // dissipates the outgoing MOMENTUM inside the band so it does not
            // reach the hard Dirichlet ring and reflect (the σ pressure-sponge
            // could not — it never touched u). MAGNITUDE-FIRST is mandatory: a
            // naive signed truncating multiply leaves a stuck −1-count floor
            // (fixed_point.h sign convention). Gated on ambient mode -> space
            // maps byte-identical (dormancy BY BRANCH); no-op where k(d)==0.
            if (ambient_mode && sponge_udamp) {
                const int32_t kd = sponge_udamp[i];
                if (kd > 0) {
                    const q16 kk2 = (kd < FP_ONE) ? (q16)(FP_ONE - kd) : 0;
                    const int64_t mx = mul128_shr(ux < 0 ? -ux : ux, (int64_t)kk2, 16);
                    const int64_t my = mul128_shr(uy < 0 ? -uy : uy, (int64_t)kk2, 16);
                    ux = (ux < 0) ? -mx : mx;
                    uy = (uy < 0) ? -my : my;
                }
            }

            // |u| ≤ u_cap = min(c_LOCAL, U_MAX) (state-derived cap + the v2.4
            // defense-in-depth rail; scale-to-cap preserves direction;
            // counter-tracked; the cheap Chebyshev pre-test avoids a sqrt
            // per cell).
            const q16 u_cap_q = (c_local_q < u_max_q) ? c_local_q : u_max_q;
            const bool cap_is_umax = (u_max_q < c_local_q);
            // OVERFLOW GUARD (eos-p3fix-thermal-ceiling — the measured
            // runaway-wind wrap): pre-clamp each component to ±2^30 raw
            // (16384 m/s — unphysical; purely a range guard). Pre-fix, a
            // blast-scale kick against a floored N̂ could leave |ux| ~9e15
            // raw: ux*ux OVERFLOWED int64, `rad` was garbage, sqrt_q16 saw
            // garbage (its int32 result can itself wrap for rad > ~2^61.9),
            // the clamp never engaged, and the final (int32_t) narrow at
            // store WRAPPED — the chaotic ±30k winds measured in the B4/B7
            // investigation, INDEPENDENT of the temperature wrap. With the
            // guard, rad ≤ 2·2^60 < int64 max, sqrt_q16's result ≤ ~1.5e9
            // (fits int32), so the magnitude clamp below always engages on
            // anything above u_cap and the narrow is safe by construction.
            // Behavior-neutral for every legitimate velocity (all caps are
            // orders of magnitude below the guard).
            const int64_t RAD_SAFE = (int64_t)1 << 30;
            if      (ux >  RAD_SAFE) ux =  RAD_SAFE;
            else if (ux < -RAD_SAFE) ux = -RAD_SAFE;
            if      (uy >  RAD_SAFE) uy =  RAD_SAFE;
            else if (uy < -RAD_SAFE) uy = -RAD_SAFE;
            const int64_t ax = ux < 0 ? -ux : ux;
            const int64_t ay = uy < 0 ? -uy : uy;
            if ((ax > (int64_t)u_cap_q) || (ay > (int64_t)u_cap_q)) {
                const int64_t rad = ux * ux + uy * uy;   // int64-safe (guard above)
                const q16 umag = sqrt_q16(rad);
                if (umag > u_cap_q) {
                    ++u_clamp_hits;
                    if (cap_is_umax) ++u_max_hits;
                    const q16 scale = reciprocal_q16(umag);
                    ux = mul128_shr(mul128_shr(ux, (int64_t)scale, 16), (int64_t)u_cap_q, 16);
                    uy = mul128_shr(mul128_shr(uy, (int64_t)scale, 16), (int64_t)u_cap_q, 16);
                }
            }
            wind_x[i] = (int32_t)ux;   // the ONE narrow at store — safe: |u| is
            wind_y[i] = (int32_t)uy;   // ≤ max(√2·u_cap, RAD_SAFE) ≪ int32 range
        }
    }
    digest_velocity = digest_of(wind_x, n, digest_of(wind_y, n, 0));

    // ======================================================================
    // 4c. COMPRESSION WORK — once per tick, POST-correction (§3.2 step 4c,
    //     corrected 2026-07-10): T -= (γ−1)·T·div(u_new)·dt on the CORRECTED
    //     velocity; feeds NEXT tick's p*, never this tick's solve. Factor
    //     clamped to ±T_WORK_CLAMP (counter-tracked); T floored at T_MIN
    //     (the named 4th energy sink, counter-tracked).
    // ======================================================================
    {
        const q16 work_clamp_q = quantize((double)T_WORK_CLAMP);
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                // BC (audit (b)): the ring is skipped like vacuum — no
                // compression work on the still ΔT=0 boundary.
                if (solid[i] || is_vacuum[i] || (ambient_mode && is_ambient[i])) continue;
                const int il = mirror_idx(i, y, x - 1, h, w, solid);
                const int ir = mirror_idx(i, y, x + 1, h, w, solid);
                const int iu = mirror_idx(i, y - 1, x, h, w, solid);
                const int id = mirror_idx(i, y + 1, x, h, w, solid);
                const q16 dux = mul_q16(wind_x[ir] - wind_x[il], inv_2dx_q);
                const q16 duy = mul_q16(wind_y[id] - wind_y[iu], inv_2dx_q);
                const q16 div_new = dux + duy;
                q16 k = mul_q16(gamma_m1_q, div_new);
                k = mul_q16(k, dt_q);
                if (k > work_clamp_q)       { k = work_clamp_q;  ++work_clamp_hits; }
                else if (k < -work_clamp_q) { k = -work_clamp_q; ++work_clamp_hits; }
                const q16 dT = mul_q16(k, temperature[i]);
                // eos-p3fix-thermal-ceiling: this term is MULTIPLICATIVE in
                // the current T (t_new = T*(1-k)) — the ±T_WORK_CLAMP rail
                // above only bounds the per-tick RATE, not the resulting
                // VALUE, so a persistent negative-divergence driver (a real
                // local compression pocket) compounds it geometrically
                // (measured: ~1.5x/tick at the clamp rail, reaching the
                // Q16.16 ceiling in ~8-9 ticks from a modest seed). The plain
                // subtract below used to silently WRAP once that compounding
                // exceeded int32 range — a hard correctness bug independent
                // of whatever seeds the compounding. Saturating add fixes
                // the wrap unconditionally; the v2.4 T_MAX_PHYS rail (see
                // eos_solver.h) then bounds the compounding's VALUE at the
                // physical ceiling — counted, so a scenario that leans on it
                // is visible in telemetry, never silent.
                q16 t_new = sat_add_q16(temperature[i], (q16)(-(int64_t)dT));
                if (t_new < t_min_q) { t_new = t_min_q; ++energy_floor_hits; }
                else if (t_new > t_max_phys_q) { t_new = t_max_phys_q; ++t_max_phys_hits; }
                temperature[i] = t_new;
            }
        }
    }
    digest_compression = digest_of(temperature, n, 0);
    // DEBUG probe (temporary): T after step 4c (compression work).
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_compression = temperature[dbg_probe_idx];

    // ======================================================================
    // 5. P := P_new — materialized ONCE (the `atmosphere` alias).
    // ======================================================================
    // BC (spec §1 "the shift trick"): the whole solve ran in P′ = P − P_amb, so
    // add P_amb back here — MASKED to !solid (solids left the solve at P′=0 and
    // must stay 0 ABSOLUTE; ring cells (excl==1, P′=0) and every regular cell
    // WANT the add, materializing P=P_amb at the ring and real P interior). The
    // space path is the untouched byte-identical store (dormancy BY BRANCH — NO
    // unconditional +0).
    if (ambient_mode) {
        const int64_t pa = (int64_t)p_amb;
        for (int i = 0; i < n; ++i)
            atmosphere[i] = solid[i] ? L0.P[i] : (int32_t)((int64_t)L0.P[i] + pa);
    } else {
        for (int i = 0; i < n; ++i) atmosphere[i] = L0.P[i];
    }
}

// ===========================================================================
// EOS P6.3 — the pressure solve's two internal stages, moved VERBATIM out of
// step() (pure code motion). The ONLY textual deltas vs the pre-P6.3 inline
// body: (a) the solve inputs arrive as parameters instead of the member
// caches — the live call passes exactly pstar_/div_u_/n_total_.data() and the
// engine's p_prev — and (b) the per-tick scalar folds consumed here
// (n_floor_q, gamma_q, dt_q, Kdt2dx2_raw) are re-derived from the SAME double
// expressions step() folds, in this same /fp:strict TU — identical bits by
// determinism of quantize/IEEE double. Split rationale in the header: the
// CUDA binding calls mg_build_levels then hands mg_levels() to the device
// V-cycle (cuda_mg_solve.cu); eos_mg_solve_reference calls both.
// ===========================================================================

// S8a Path A: the MG-build scalar folds, moved VERBATIM out of
// mg_build_levels (pure code motion — the header documents why: the
// device-resident build must consume these identical bits, so this
// /fp:strict TU holds the ONE transcription).
EOSSolver::MGScalarFolds EOSSolver::mg_scalar_folds(float dt) const {
    MGScalarFolds f;
    // Per-tick scalar folds (identical expressions to step()'s hoists).
    f.n_floor_q  = quantize((double)N_FLOOR_SOLVER);
    const double gamma_d = (double)adiabatic_index;
    f.gamma_q    = quantize(gamma_d);
    const double dt_d    = (double)dt;
    f.dt_q       = quantize(dt_d);
    const double dx_d    = std::max((double)dx, 1e-6);
    const double K_d = (double)c_max * (double)c_max / gamma_d;
    // K·dt²/dx² — the operator's geometric factor (real ≈1,006 at bench
    // dt/dx; ×(γp*≈1.4 ambient) reproduces the pre-v2.2 coupling ≈1,409 —
    // the round-trip product D-A preserves by construction).
    f.Kdt2dx2_raw =
        (int64_t)(K_d * dt_d * dt_d / (dx_d * dx_d) * 65536.0 + 0.5);
    return f;
}

int EOSSolver::mg_build_levels(
        const int32_t* pstar, const int32_t* div_u, const int32_t* n_total,
        const int32_t* p_prev,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt,
        const bool* is_ambient, int32_t p_amb,
        const int32_t* sponge_sigma) const {
    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return 0;
    // BC: dormancy BY BRANCH — every ambient edit below is gated on this flag.
    const bool ambient_mode = (is_ambient != nullptr);

    // Per-tick scalar folds — the ONE transcription (mg_scalar_folds above).
    const MGScalarFolds fold = mg_scalar_folds(dt);
    const q16 n_floor_q      = fold.n_floor_q;
    const q16 gamma_q        = fold.gamma_q;
    const q16 dt_q           = fold.dt_q;
    const int64_t Kdt2dx2_raw = fold.Kdt2dx2_raw;

    // --- level count (fixed by grid size — deterministic) ------------------
    int n_levels = 1;
    {
        int lh = h, lw = w;
        while (std::min(lh, lw) > mg_min_dim && n_levels < 9) {
            lh = (lh + 1) >> 1;
            lw = (lw + 1) >> 1;
            ++n_levels;
        }
    }
    if (!use_multigrid) n_levels = 1;
    if ((int)levels_.size() < n_levels) levels_.resize(n_levels);

    // --- build level 0 from the fine fields (v2.2-final SYMMETRIC form) ----
    // Row i of the analytic operator, divided by aK_i = (γ·p*_i)·K·dt²/dx²:
    //   m_i·P_i + Σ_f g_f·(P_i − P_nb) = m_i·rhs_i,   m_i = 1/aK_i
    // Same solution, SPD system (mass + symmetric face Laplacian) — the form
    // the variational multigrid below cannot amplify. Near-vacuum rows have
    // aK→0 ⇒ m→huge (clamped) ⇒ P pinned to rhs (→0): §3.1 property 2 in
    // mass form. WORK SCALE: row products are held at "F8" precision —
    // (coeff_raw·field_raw)>>8, eight fractional bits finer than Q16.16 —
    // so the weak ambient mass anchor (m ≈ 1/1409) still registers sub-count
    // residuals (a plain >>16 truncated them to zero — an invisible-DC-error
    // floor of ~0.02 atm the gate measured as drift).
    {
        MGLevel& L = levels_[0];
        L.h = h; L.w = w;
        L.excl.assign(n, 0);
        L.m.assign(n, 0);
        L.gE.assign(n, 0);
        L.gS.assign(n, 0);
        L.recip.assign(n, 0);
        L.P.assign(n, 0);
        L.b.assign(n, 0);
        L.res.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            if (solid[i]) L.excl[i] = 2;
            else if (is_vacuum[i]) L.excl[i] = 1;
            // BC (audit (c), :715-718): the ambient ring is Dirichlet too —
            // excl==1, pinned to P′=0 (≡ P=P_amb after the step-5 add-back). The
            // pin VALUE is unchanged (0) under the shift, so the zero-Dirichlet
            // MG kernels stay byte-identical (spec §1). is_vacuum is all-false
            // on ambient maps, so this catches every ring tile.
            else if (ambient_mode && is_ambient[i]) L.excl[i] = 1;
        }
        // Mass clamp: [1, 2^38] raw. The cap keeps the 128-bit mass-term
        // product (m·ΔP)>>8 inside int64 (2^38·2^31/2^8 = 2^61) while still
        // dominating any conductance by >2^20 — a capped-mass row pins P to
        // rhs within ~1e-6 relative. The floor keeps a joint-spike row
        // (aK huge ⇒ m underflows) present in the system at all.
        const int64_t M_CAP = ((int64_t)1) << 38;
        for (int i = 0; i < n; ++i) {
            if (L.excl[i] != 0) continue;
            const int64_t gp_raw = mul128_shr((int64_t)gamma_q, (int64_t)pstar[i], 16);
            int64_t aK = mul128_shr(gp_raw, Kdt2dx2_raw, 16);
            if (aK < 1) aK = 1;
            int64_t m = (((int64_t)1) << 32) / aK;   // 1/aK at Q16.16 raw
            if (m < 1) m = 1;
            if (m > M_CAP) m = M_CAP;
            L.m[i] = m;
            // rhs_i = p* − (γ·p*)·dt·div(û*) (int64 raw); b = m·rhs @F8.
            const int64_t gp_dt = mul128_shr(gp_raw, (int64_t)dt_q, 16);
            int64_t rhs_raw = (int64_t)pstar[i]
                                  - mul128_shr(gp_dt, (int64_t)div_u[i], 16);
            // BC (spec §1 "the shift trick"): solve P′ = P − P_amb. Under the
            // shift the mass anchor's constant folds to the rhs: rhs′ = rhs −
            // P_amb (the Σg face difference is shift-invariant, cancels exactly
            // in integers). Subtract BEFORE the m multiply. Branch-gated -> NO
            // unconditional −0 on the space path. m stays un-σ'd here (B3b).
            if (ambient_mode) rhs_raw -= (int64_t)p_amb;
            L.b[i] = mul128_shr(m, rhs_raw, 8);
            // Warm start from the PREVIOUS tick's solved P (`p_prev` — copied
            // from `atmosphere` at step 0): it already carries the room-scale
            // acoustic structure the smoother is slowest to build, worth ~one
            // V-cycle of error (measured at the MG gate; p* was the earlier,
            // weaker choice).
            // BC (spec §1): p_prev stores the UNSHIFTED atmosphere; re-shift the
            // warm start fresh into P′-space each tick. Branch-gated (space maps
            // keep the byte-identical p_prev[i] warm start).
            L.P[i] = ambient_mode ? (int32_t)((int64_t)p_prev[i] - (int64_t)p_amb) : p_prev[i];
        }
        // Face conductances g = perm/N̂ (the 1/N̂ divide folded BEFORE any
        // multiply by P — the documented joint-case order; arithmetic-N̂ IS
        // the harmonic mean of the two cells' 1/N conductivities).
        // N_FLOOR_SOLVER applies HERE ONLY (never to m — §3.1 property 2).
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (x < w - 1) {
                    const int j = i + 1;
                    if (L.excl[i] != 2 && L.excl[j] != 2) {
                        const float pf = std::min(dyn_permeability[i], dyn_permeability[j]);
                        if (pf > 0.0f) {
                            q16 nhat = (q16)(((int64_t)n_total[i] + n_total[j]) >> 1);
                            if (nhat < n_floor_q) nhat = n_floor_q;
                            L.gE[i] = mul_q16(quantize((double)pf), reciprocal_q16(nhat));
                        }
                    }
                }
                if (y < h - 1) {
                    const int j = i + w;
                    if (L.excl[i] != 2 && L.excl[j] != 2) {
                        const float pf = std::min(dyn_permeability[i], dyn_permeability[j]);
                        if (pf > 0.0f) {
                            q16 nhat = (q16)(((int64_t)n_total[i] + n_total[j]) >> 1);
                            if (nhat < n_floor_q) nhat = n_floor_q;
                            L.gS[i] = mul_q16(quantize((double)pf), reciprocal_q16(nhat));
                        }
                    }
                }
            }
        }
        // BC (spec §3 rung 1, B3b): the σ-SPONGE. Add the static per-cell
        // sponge mass to the level-0 row diagonal at band cells — AFTER the L.b
        // build (b used the un-σ'd m, the spec equation) and BEFORE the Galerkin
        // coarse build + recip build below, so BOTH fold σ automatically. Under
        // the shift the σ·P_amb rhs term vanishes: σ pulls P′ → 0 ≡ ambient,
        // extending the Dirichlet ring inward with a taper (an unconditionally
        // stable absorber that acts within the tick the wave arrives). Ring
        // cells are excl==1 (Dirichlet), so σ is moot there anyway; the B2 grid
        // is 0 at d==0. int64 row mass (M_CAP 2^38) keeps σ ≫ FP_ONE overflow-
        // safe. All-zero grid (width 0 / strength 0) -> no-op (rung 0 dormant).
        if (ambient_mode && sponge_sigma) {
            for (int i = 0; i < n; ++i) {
                if (L.excl[i] != 0) continue;
                const int32_t s = sponge_sigma[i];
                if (s <= 0) continue;
                int64_t ms = L.m[i] + (int64_t)s;
                if (ms > M_CAP) ms = M_CAP;
                L.m[i] = ms;
            }
        }
    }

    // --- build coarse levels: exactly-variational PC Galerkin --------------
    // Masses SUM (aggregate cell mass), face conductances SUM across the
    // (≤2) fine faces crossing each coarse interface (interior fine faces
    // cancel — this IS PᵀAP for piecewise-constant transfers, so the coarse
    // correction is an energy-norm projection: structurally convergent at
    // any depth). Coarse Dirichlet rule (v2.2, explicit): vacuum iff ALL
    // children vacuum; all-non-regular -> excluded; any regular -> regular.
    for (int lv = 1; lv < n_levels; ++lv) {
        const MGLevel& F = levels_[lv - 1];
        MGLevel& Cl = levels_[lv];
        Cl.h = (F.h + 1) >> 1;
        Cl.w = (F.w + 1) >> 1;
        const int cn = Cl.h * Cl.w;
        Cl.excl.assign(cn, 0);
        Cl.m.assign(cn, 0);
        Cl.gE.assign(cn, 0);
        Cl.gS.assign(cn, 0);
        Cl.recip.assign(cn, 0);
        Cl.P.assign(cn, 0);
        Cl.b.assign(cn, 0);
        Cl.res.assign(cn, 0);
        const int64_t M_CAP_L = ((int64_t)1) << 44;   // level cap (sums grow ×4/level)
        for (int Y = 0; Y < Cl.h; ++Y) {
            for (int X = 0; X < Cl.w; ++X) {
                const int A = Y * Cl.w + X;
                int n_child = 0, n_vac = 0, n_sol = 0;
                int64_t m_sum = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dxx = 0; dxx < 2; ++dxx) {
                        const int fy = 2 * Y + dy, fx = 2 * X + dxx;
                        if (fy >= F.h || fx >= F.w) continue;
                        const int fi = fy * F.w + fx;
                        ++n_child;
                        if (F.excl[fi] == 1) ++n_vac;
                        else if (F.excl[fi] == 2) ++n_sol;
                        else m_sum += F.m[fi];
                    }
                }
                if (n_vac == n_child) Cl.excl[A] = 1;
                else if (n_vac + n_sol == n_child) Cl.excl[A] = 2;
                else {
                    Cl.excl[A] = 0;
                    // GALERKIN DIRICHLET ANCHOR (found by the gate's own
                    // breach test — the first PC-Galerkin cut dropped it and
                    // MEASURABLY amplified ×7/cycle at the breach): every
                    // fine face from a REGULAR child to a DIRICHLET fine
                    // cell couples that child to the fixed value 0 — in
                    // PᵀAP it lands on the coarse DIAGONAL. It acts exactly
                    // like extra mass with zero rhs, and on coarse levels b
                    // comes purely from restriction, so folding it into m
                    // is the exact Galerkin term (no new field needed).
                    int64_t anchor = 0;
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dxx = 0; dxx < 2; ++dxx) {
                            const int fy = 2 * Y + dy, fx = 2 * X + dxx;
                            if (fy >= F.h || fx >= F.w) continue;
                            const int fi = fy * F.w + fx;
                            if (F.excl[fi] != 0) continue;
                            if (fx + 1 < F.w && F.excl[fi + 1] == 1)   anchor += F.gE[fi];
                            if (fx > 0 && F.excl[fi - 1] == 1)         anchor += F.gE[fi - 1];
                            if (fy + 1 < F.h && F.excl[fi + F.w] == 1) anchor += F.gS[fi];
                            if (fy > 0 && F.excl[fi - F.w] == 1)       anchor += F.gS[fi - F.w];
                        }
                    }
                    Cl.m[A] = std::min(m_sum + anchor, M_CAP_L);
                }
            }
        }
        for (int Y = 0; Y < Cl.h; ++Y) {
            for (int X = 0; X < Cl.w; ++X) {
                const int A = Y * Cl.w + X;
                if (X < Cl.w - 1) {
                    // Coarse face = SUM of crossing fine faces whose BOTH
                    // endpoints are regular (a regular->Dirichlet crossing
                    // face is the anchor term above, NOT inter-cell coupling).
                    int64_t gsum = 0;
                    const int fx = 2 * X + 1;   // fine face fx -> fx+1
                    for (int dy = 0; dy < 2; ++dy) {
                        const int fy = 2 * Y + dy;
                        if (fy >= F.h || fx + 1 >= F.w) continue;
                        const int fi = fy * F.w + fx;
                        if (F.excl[fi] == 0 && F.excl[fi + 1] == 0)
                            gsum += F.gE[fi];
                    }
                    Cl.gE[A] = gsum;            // SUM (variational), not average
                }
                if (Y < Cl.h - 1) {
                    int64_t gsum = 0;
                    const int fy = 2 * Y + 1;
                    for (int dxx = 0; dxx < 2; ++dxx) {
                        const int fx = 2 * X + dxx;
                        if (fx >= F.w || fy + 1 >= F.h) continue;
                        const int fi = fy * F.w + fx;
                        if (F.excl[fi] == 0 && F.excl[fi + F.w] == 0)
                            gsum += F.gS[fi];
                    }
                    Cl.gS[A] = gsum;
                }
            }
        }
    }

    // --- per-level diagonal reciprocals (ONE wide divide/cell/tick/level) --
    for (int lv = 0; lv < n_levels; ++lv) {
        MGLevel& L = levels_[lv];
        const int lh = L.h, lw = L.w;
        for (int y = 0; y < lh; ++y) {
            const int row = y * lw;
            for (int x = 0; x < lw; ++x) {
                const int i = row + x;
                if (L.excl[i] != 0) { L.recip[i] = 0; continue; }
                int64_t d_raw = L.m[i];
                if (x < lw - 1 && L.excl[i + 1] != 2) d_raw += L.gE[i];
                if (x > 0 && L.excl[i - 1] != 2)      d_raw += L.gE[i - 1];
                if (y < lh - 1 && L.excl[i + lw] != 2) d_raw += L.gS[i];
                if (y > 0 && L.excl[i - lw] != 2)      d_raw += L.gS[i - lw];
                if (d_raw < 1) d_raw = 1;
                L.recip[i] = (((int64_t)1) << 48) / d_raw;   // Q.32 reciprocal
            }
        }
    }

    return n_levels;
}

void EOSSolver::mg_run_solve_cpu(int n_levels) const {
    if (n_levels <= 0) return;
    const int n = levels_[0].h * levels_[0].w;   // (was step()'s local n)

    // --- the RB-GS smoother (SPD form, residual form, WIDE flux — D-C) -----
    // inc(P counts) = r8·(2^48/d) >> 40  ==  (r8·2^8)/d.
    auto smooth = [&](MGLevel& L, int sweeps) {
        const int lh = L.h, lw = L.w;
        for (int it = 0; it < sweeps; ++it) {
            for (int color = 0; color < 2; ++color) {
                for (int y = 0; y < lh; ++y) {
                    const int row = y * lw;
                    for (int x = 0; x < lw; ++x) {
                        if (((x + y) & 1) != color) continue;
                        const int i = row + x;
                        if (L.excl[i] != 0) continue;
                        const int32_t pi = L.P[i];
                        // (A·P)@F8 = m·P + Σ g·(P_i − P_nb), each product >>8.
                        int64_t ap = mul128_shr(L.m[i], (int64_t)pi, 8);
                        if (x < lw - 1 && L.excl[i + 1] != 2) {
                            const int32_t pn = (L.excl[i + 1] == 1) ? 0 : L.P[i + 1];
                            ap += mul128_shr(L.gE[i], (int64_t)(pi - pn), 8);
                        }
                        if (x > 0 && L.excl[i - 1] != 2) {
                            const int32_t pn = (L.excl[i - 1] == 1) ? 0 : L.P[i - 1];
                            ap += mul128_shr(L.gE[i - 1], (int64_t)(pi - pn), 8);
                        }
                        if (y < lh - 1 && L.excl[i + lw] != 2) {
                            const int32_t pn = (L.excl[i + lw] == 1) ? 0 : L.P[i + lw];
                            ap += mul128_shr(L.gS[i], (int64_t)(pi - pn), 8);
                        }
                        if (y > 0 && L.excl[i - lw] != 2) {
                            const int32_t pn = (L.excl[i - lw] == 1) ? 0 : L.P[i - lw];
                            ap += mul128_shr(L.gS[i - lw], (int64_t)(pi - pn), 8);
                        }
                        const int64_t r8 = L.b[i] - ap;
                        const int64_t inc = mul128_shr(r8, L.recip[i], 40);
                        L.P[i] = (int32_t)((int64_t)pi + inc);
                    }
                }
            }
        }
    };

    // --- residual r@F8 = b − A·P (into L.res) -------------------------------
    auto residual = [&](MGLevel& L) {
        const int lh = L.h, lw = L.w;
        for (int y = 0; y < lh; ++y) {
            const int row = y * lw;
            for (int x = 0; x < lw; ++x) {
                const int i = row + x;
                if (L.excl[i] != 0) { L.res[i] = 0; continue; }
                const int32_t pi = L.P[i];
                int64_t ap = mul128_shr(L.m[i], (int64_t)pi, 8);
                if (x < lw - 1 && L.excl[i + 1] != 2) {
                    const int32_t pn = (L.excl[i + 1] == 1) ? 0 : L.P[i + 1];
                    ap += mul128_shr(L.gE[i], (int64_t)(pi - pn), 8);
                }
                if (x > 0 && L.excl[i - 1] != 2) {
                    const int32_t pn = (L.excl[i - 1] == 1) ? 0 : L.P[i - 1];
                    ap += mul128_shr(L.gE[i - 1], (int64_t)(pi - pn), 8);
                }
                if (y < lh - 1 && L.excl[i + lw] != 2) {
                    const int32_t pn = (L.excl[i + lw] == 1) ? 0 : L.P[i + lw];
                    ap += mul128_shr(L.gS[i], (int64_t)(pi - pn), 8);
                }
                if (y > 0 && L.excl[i - lw] != 2) {
                    const int32_t pn = (L.excl[i - lw] == 1) ? 0 : L.P[i - lw];
                    ap += mul128_shr(L.gS[i - lw], (int64_t)(pi - pn), 8);
                }
                L.res[i] = L.b[i] - ap;
            }
        }
    };

    // --- restriction: residual SUM over children (the PC transpose) --------
    auto restrict_res = [&](const MGLevel& F, MGLevel& Cl) {
        for (int Y = 0; Y < Cl.h; ++Y) {
            for (int X = 0; X < Cl.w; ++X) {
                const int A = Y * Cl.w + X;
                Cl.P[A] = 0;                     // corrections start at 0
                if (Cl.excl[A] != 0) { Cl.b[A] = 0; continue; }
                int64_t rsum = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dxx = 0; dxx < 2; ++dxx) {
                        const int fy = 2 * Y + dy, fx = 2 * X + dxx;
                        if (fy >= F.h || fx >= F.w) continue;
                        const int fi = fy * F.w + fx;
                        if (F.excl[fi] == 0) rsum += F.res[fi];
                    }
                }
                Cl.b[A] = rsum;                  // SUM — variational (PᵀAP pair)
            }
        }
    };

    // --- prolongation: piecewise-constant injection (the exact transpose) --
    // (The spec's named bilinear prolongation paired NON-variationally with
    // the re-discretized coarse operator — the deep pyramid MEASURABLY
    // amplified (more cycles = worse). PC injection + SUM restriction is the
    // exact PᵀAP pair; the schedule freeze documents this deviation.)
    auto prolong_correct = [&](MGLevel& F, const MGLevel& Cl) {
        for (int fy = 0; fy < F.h; ++fy) {
            for (int fx = 0; fx < F.w; ++fx) {
                const int fi = fy * F.w + fx;
                if (F.excl[fi] != 0) continue;
                const int A = (fy >> 1) * Cl.w + (fx >> 1);
                if (Cl.excl[A] != 0) continue;
                F.P[fi] = (int32_t)((int64_t)F.P[fi] + (int64_t)Cl.P[A]);
            }
        }
    };

    // --- the fixed-schedule V-cycle -----------------------------------------
    if (use_multigrid && n_levels > 1) {
        for (int cyc = 0; cyc < mg_cycles; ++cyc) {
            for (int lv = 0; lv < n_levels - 1; ++lv) {
                smooth(levels_[lv], mg_nu1);
                residual(levels_[lv]);
                restrict_res(levels_[lv], levels_[lv + 1]);
            }
            smooth(levels_[n_levels - 1], mg_coarsest_sweeps);
            for (int lv = n_levels - 2; lv >= 0; --lv) {
                prolong_correct(levels_[lv], levels_[lv + 1]);
                smooth(levels_[lv], mg_nu2);
            }
        }
    } else {
        smooth(levels_[0], S);   // flat A/B reference path
    }

    MGLevel& L0 = levels_[0];
    for (int i = 0; i < n; ++i) {
        if (L0.excl[i] != 0) L0.P[i] = 0;    // vacuum Dirichlet + solid zero
    }
    digest_helmholtz = digest_of(L0.P.data(), n, 0);
}

// ===========================================================================
// EOS P6.2 — standalone CPU reference for the fused SL-advection substep loop
// (declared in eos_solver.h; rationale there). A VERBATIM replay of step()'s
// step-1a/1b/1f chain for a GIVEN n_sub:
//   * cmask build     — the same solid/perm<=0 -> 0, vacuum -> 1, live -> 2
//                       table step() builds once per tick;
//   * per substep     — src snapshot of (vx, vy, T), then the per-cell fused
//                       backtrace via the SAME file-local
//                       eos_backtrace_sample3_q (one routine, zero drift),
//                       solid cells zero u / keep T, vacuum destinations
//                       force T := 0;
//   * dt_s_q          — quantize((double)dt / (double)n_sub), exactly the
//                       dt_d/dt_s_d fold step() performs.
// The step-1f "zero u on solid" pass is subsumed: the advection pass itself
// zeroes solid cells' u, and nothing between (bulk flux writes only gas
// planes) re-touches u — replicated here by construction.
// Returns the chained FNV digest over (T, then wy, then wx) — byte-for-byte
// the digest_advect expression at step()'s last substep.
// ===========================================================================
uint64_t eos_sl_advect_reference(
        int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt, int n_sub,
        const bool* is_ambient) {
    const int n = h * w;
    if (n <= 0 || dt <= 0.0f || n_sub < 1) return 0;
    const bool ambient_mode = (is_ambient != nullptr);   // BC: dormancy by branch

    // cmask (verbatim: step()'s per-tick corner/march table).
    std::vector<uint8_t> cmask(n, 0);
    for (int i = 0; i < n; ++i) {
        if (solid[i] || dyn_permeability[i] <= 0.0f) cmask[i] = 0;
        else if (is_vacuum[i] || (ambient_mode && is_ambient[i])) cmask[i] = 1;   // BC: ring is a breach corner
        else cmask[i] = 2;
    }

    std::vector<int32_t> vx_src(n), vy_src(n), t_src(n);
    const double dt_s_d = (double)dt / (double)n_sub;   // == step()'s dt_d/n_sub
    for (int s = 0; s < n_sub; ++s) {
        const q16 dt_s_q = quantize(dt_s_d);
        for (int i = 0; i < n; ++i) {
            vx_src[i] = wind_x[i];
            vy_src[i] = wind_y[i];
            t_src[i]  = temperature[i];
        }
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                if (solid[i]) { wind_x[i] = 0; wind_y[i] = 0; continue; }
                const int32_t bx_q = -mul_q16(vx_src[i], dt_s_q);
                const int32_t by_q = -mul_q16(vy_src[i], dt_s_q);
                const FusedSample fs = eos_backtrace_sample3_q(
                    vx_src.data(), vy_src.data(), t_src.data(),
                    x, y, bx_q, by_q,
                    cmask.data(), h, w);
                wind_x[i] = fs.vx;
                wind_y[i] = fs.vy;
                temperature[i] = (is_vacuum[i] || (ambient_mode && is_ambient[i])) ? 0 : fs.t;   // BC: ΔT=0 is ambient
            }
        }
    }
    return digest_of(wind_x, n, digest_of(wind_y, n, digest_of(temperature, n, 0)));
}

// ===========================================================================
// EOS P6.4 — standalone CPU reference for the step-4 kick + step-4c
// compression work (declared in eos_solver.h; rationale there). A VERBATIM
// replay of step()'s post-solve tail on a GIVEN step-4-entry state:
//   * scalar folds    — the IDENTICAL double expressions step() performs
//                       (K_raw/Kdt_raw, inv_2dx_q, absorb_dt_q, the rail
//                       quantizes), from the same config values;
//   * Dalton sum      — step 2's n_total_ loop verbatim (bulk planes at full
//                       weight, trace planes × trace_mass_scale) — the kick's
//                       1/N̂ input is REBUILT, not approximated;
//   * step 4          — gradient/kick/absorption/rail chain copied line for
//                       line from step() (same int64 staging, same clamp
//                       order, same per-CELL counter semantics);
//   * step 4c         — the compression-work loop copied line for line.
// c_local_q comes in as a parameter (dbg_last_c_local_q telemetry): it is
// derived from the PRE-advection T scan that an isolated tail replay cannot
// see. Counters are returned per-call; digests are byte-for-byte step()'s
// digest_velocity / digest_compression expressions. Test entry only — the
// live path remains EOSSolver::step.
// ===========================================================================
void eos_kick_compression_reference(
        int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
        const int32_t* p_new,
        const int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_wave_absorb,
        int h, int w, float dt, int32_t c_local_q,
        float c_max, float dx, float adiabatic_index, float absorb_strength,
        float n_floor_solver, float t_min, float t_work_clamp,
        float t_max_phys, float u_max, float trace_mass_scale,
        uint64_t* digest_velocity_out, uint64_t* digest_compression_out,
        int64_t* counters_out /* [5] */,
        const bool* is_ambient) {
    const int n = h * w;
    const bool ambient_mode = (is_ambient != nullptr);   // BC: dormancy by branch
    for (int c = 0; c < 5; ++c) counters_out[c] = 0;
    *digest_velocity_out = 0;
    *digest_compression_out = 0;
    if (n <= 0 || dt <= 0.0f) return;

    // ---- per-tick scalar constants (step()'s folds, verbatim) -------------
    const q16 n_floor_q    = quantize((double)n_floor_solver);
    const q16 t_min_q      = quantize((double)t_min);
    const q16 t_max_phys_q = quantize((double)t_max_phys);
    const q16 u_max_q      = quantize((double)u_max);
    const double gamma_d   = (double)adiabatic_index;
    const q16 gamma_m1_q   = quantize(gamma_d - 1.0);
    const double dt_d      = (double)dt;
    const q16 dt_q         = quantize(dt_d);
    const double dx_d      = std::max((double)dx, 1e-6);
    const q16 inv_2dx_q    = quantize(1.0 / (2.0 * dx_d));
    const double K_d = (double)c_max * (double)c_max / gamma_d;
    const int64_t K_raw = (int64_t)(K_d * 65536.0 + 0.5);
    const int64_t Kdt_raw = mul128_shr(K_raw, (int64_t)dt_q, 16);
    const q16 absorb_dt_q = quantize((double)absorb_strength * dt_d);

    // ---- step 2's Dalton sum (verbatim — the kick's N̂ input) --------------
    std::vector<int32_t> n_total(n, 0);
    {
        const q16 tms_q = quantize((double)trace_mass_scale);
        for (int gi = 0; gi < n_gases; ++gi) {
            const int32_t* plane = gas + (size_t)gi * n;
            if (gas_conservative[gi]) {
                for (int i = 0; i < n; ++i) n_total[i] += plane[i];
            } else {
                for (int i = 0; i < n; ++i) n_total[i] += mul_q16(tms_q, plane[i]);
            }
        }
    }

    int64_t u_clamp_hits = 0, u_max_hits = 0, work_clamp_hits = 0,
            energy_floor_hits = 0, t_max_phys_hits = 0;

    // ---- step 4: the momentum kick (step()'s loop, verbatim) --------------
    const int32_t* Pn = p_new;
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            // BC (audit (b)): ring u ≡ 0 — a still boundary, the vacuum idiom.
            if (solid[i] || is_vacuum[i] || (ambient_mode && is_ambient[i])) { wind_x[i] = 0; wind_y[i] = 0; continue; }
            const int il = mirror_idx(i, y, x - 1, h, w, solid);
            const int ir = mirror_idx(i, y, x + 1, h, w, solid);
            const int iu = mirror_idx(i, y - 1, x, h, w, solid);
            const int id = mirror_idx(i, y + 1, x, h, w, solid);
            const int64_t gx = mul128_shr((int64_t)(Pn[ir] - Pn[il]), (int64_t)inv_2dx_q, 16);
            const int64_t gy = mul128_shr((int64_t)(Pn[id] - Pn[iu]), (int64_t)inv_2dx_q, 16);
            int64_t ux = (int64_t)wind_x[i];
            int64_t uy = (int64_t)wind_y[i];
            if (gx != 0 || gy != 0) {
                q16 nhat = n_total[i];
                if (nhat < n_floor_q) nhat = n_floor_q;
                const q16 inv_n = reciprocal_q16(nhat);
                ux -= mul128_shr(mul128_shr(Kdt_raw, gx, 16), (int64_t)inv_n, 16);
                uy -= mul128_shr(mul128_shr(Kdt_raw, gy, 16), (int64_t)inv_n, 16);
            }

            const q16 a = mul_q16(quantize((double)dyn_wave_absorb[i]), absorb_dt_q);
            if (a > 0) {
                const q16 kk = (a < FP_ONE) ? (q16)(FP_ONE - a) : 0;
                const int64_t mx = mul128_shr(ux < 0 ? -ux : ux, (int64_t)kk, 16);
                const int64_t my = mul128_shr(uy < 0 ? -uy : uy, (int64_t)kk, 16);
                ux = (ux < 0) ? -mx : mx;
                uy = (uy < 0) ? -my : my;
            }

            const q16 u_cap_q = (c_local_q < u_max_q) ? c_local_q : u_max_q;
            const bool cap_is_umax = (u_max_q < c_local_q);
            const int64_t RAD_SAFE = (int64_t)1 << 30;
            if      (ux >  RAD_SAFE) ux =  RAD_SAFE;
            else if (ux < -RAD_SAFE) ux = -RAD_SAFE;
            if      (uy >  RAD_SAFE) uy =  RAD_SAFE;
            else if (uy < -RAD_SAFE) uy = -RAD_SAFE;
            const int64_t ax = ux < 0 ? -ux : ux;
            const int64_t ay = uy < 0 ? -uy : uy;
            if ((ax > (int64_t)u_cap_q) || (ay > (int64_t)u_cap_q)) {
                const int64_t rad = ux * ux + uy * uy;
                const q16 umag = sqrt_q16(rad);
                if (umag > u_cap_q) {
                    ++u_clamp_hits;
                    if (cap_is_umax) ++u_max_hits;
                    const q16 scale = reciprocal_q16(umag);
                    ux = mul128_shr(mul128_shr(ux, (int64_t)scale, 16), (int64_t)u_cap_q, 16);
                    uy = mul128_shr(mul128_shr(uy, (int64_t)scale, 16), (int64_t)u_cap_q, 16);
                }
            }
            wind_x[i] = (int32_t)ux;
            wind_y[i] = (int32_t)uy;
        }
    }
    *digest_velocity_out = digest_of(wind_x, n, digest_of(wind_y, n, 0));

    // ---- step 4c: compression work (step()'s loop, verbatim) --------------
    {
        const q16 work_clamp_q = quantize((double)t_work_clamp);
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (solid[i] || is_vacuum[i] || (ambient_mode && is_ambient[i])) continue;   // BC: ring skipped like vacuum
                const int il = mirror_idx(i, y, x - 1, h, w, solid);
                const int ir = mirror_idx(i, y, x + 1, h, w, solid);
                const int iu = mirror_idx(i, y - 1, x, h, w, solid);
                const int id = mirror_idx(i, y + 1, x, h, w, solid);
                const q16 dux = mul_q16(wind_x[ir] - wind_x[il], inv_2dx_q);
                const q16 duy = mul_q16(wind_y[id] - wind_y[iu], inv_2dx_q);
                const q16 div_new = dux + duy;
                q16 k = mul_q16(gamma_m1_q, div_new);
                k = mul_q16(k, dt_q);
                if (k > work_clamp_q)       { k = work_clamp_q;  ++work_clamp_hits; }
                else if (k < -work_clamp_q) { k = -work_clamp_q; ++work_clamp_hits; }
                const q16 dT = mul_q16(k, temperature[i]);
                q16 t_new = sat_add_q16(temperature[i], (q16)(-(int64_t)dT));
                if (t_new < t_min_q) { t_new = t_min_q; ++energy_floor_hits; }
                else if (t_new > t_max_phys_q) { t_new = t_max_phys_q; ++t_max_phys_hits; }
                temperature[i] = t_new;
            }
        }
    }
    *digest_compression_out = digest_of(temperature, n, 0);

    counters_out[0] = u_clamp_hits;
    counters_out[1] = u_max_hits;
    counters_out[2] = work_clamp_hits;
    counters_out[3] = energy_floor_hits;
    counters_out[4] = t_max_phys_hits;
}

// ===========================================================================
// EOS P6.3 — standalone CPU reference for the multigrid pressure solve
// (declared in eos_solver.h; rationale there). Drives the SAME two internal
// routines step() calls — mg_build_levels (per-tick hierarchy: level-0
// m/gE/gS/b/excl + P_prev warm start, PC-Galerkin coarse operators, Q.32
// diagonal reciprocals) and mg_run_solve_cpu (the frozen V(nu1,nu2)xC
// schedule or the flat RB-GS path, the vacuum-Dirichlet/solid zero, and the
// digest_helmholtz FNV) — on caller-supplied solve inputs. Writes the solved
// level-0 P (== step 5's atmosphere materialization == the step-4 kick's Pn
// input) into p_out and returns the digest, byte-for-byte the value step()
// stores in digest_helmholtz for identical inputs. Test entry only — the
// live path remains EOSSolver::step.
// ===========================================================================
uint64_t eos_mg_solve_reference(
        const EOSSolver& solver,
        const int32_t* pstar, const int32_t* div_u, const int32_t* n_total,
        const int32_t* p_prev,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt,
        int32_t* p_out) {
    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return 0;
    const int n_levels = solver.mg_build_levels(
        pstar, div_u, n_total, p_prev,
        solid, is_vacuum, dyn_permeability, h, w, dt);
    solver.mg_run_solve_cpu(n_levels);
    const EOSSolver::MGLevel& L0 = solver.mg_levels()[0];
    for (int i = 0; i < n; ++i) p_out[i] = L0.P[i];
    return solver.digest_helmholtz;
}
