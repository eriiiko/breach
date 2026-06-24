#include "water_solver.h"
#include "fixed_point.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

using namespace fixedpoint;

float WaterSolver::max_dt() const {
    // Plain wave CFL at the reference column, with the head-term margin (W4):
    // linearised, the pipe model is a damped wave with c = sqrt(g*depth); the
    // pressure head stiffens the restoring force by (1 + k_p*P_REF/HEAD_REF).
    return 0.5f * dx / std::sqrt(g * h_ref * (1.0f + k_p * P_REF / HEAD_REF));
}

q16 WaterSolver::max_dt_q() const {
    // S1: the CFL bound as a Q16.16 CONSTANT. Computed ONCE in DOUBLE — IEEE
    // sqrt is correctly-rounded, so this double value is bit-identical across
    // machines/compilers — then quantized round-to-nearest. NO integer sqrt
    // (this is a config-only load-time constant, per the locked decision). The
    // substep-count cliff uses THIS (via ceil_div) so n is integer-deterministic.
    const double g_d = (double)g;
    const double h_d = (double)h_ref;
    const double kp_d = (double)k_p;
    const double margin = 1.0 + kp_d * (double)P_REF / (double)HEAD_REF;
    const double mdt = 0.5 * (double)dx / std::sqrt(g_d * h_d * margin);
    return quantize(mdt);
}

float WaterSolver::ripple_max_dt() const {
    // STATIC bound at the deep-water cap (canon §6): c <= sqrt(g*h_cap)
    // everywhere by construction. RENDER-ONLY -> stays float (no substep cliff:
    // one step_ripple call per tick).
    return 0.5f * dx / std::sqrt(g * h_cap);
}

void WaterSolver::step(q16* water_depth, q16* flow_vx, q16* flow_vy,
                       const q16* floor_height,
                       const float* atmosphere,
                       const float* wave_p,
                       const bool*  solid,
                       int h, int w, float dt,
                       float tilt_x, float tilt_y) const {
    const int n = h * w;

    // ---- Per-step Q16.16 constants (combined ONCE in double, then quantized /
    //      reciprocated). The runtime scalars dt/dx/g/damping are real; only the
    //      FIELDS are integer. Folding the products here keeps the per-cell loops
    //      pure-integer and deterministic. ------------------------------------
    const double dt_d = (double)dt;
    const double dx_d = (double)dx;
    const double g_d  = (double)g;
    const double damp_d = (double)damping;

    // Velocity-kick coefficients (Q16.16):
    //   vx += dt*(-g*dsdx - damping*vx)  ==  vx - g_dt*dsdx - damp_dt*vx
    const q16 g_dt_q   = quantize(g_d * dt_d);       // g*dt   (Q16.16)
    const q16 damp_dt_q = quantize(damp_d * dt_d);   // damp*dt(Q16.16)
    const q16 v_max_q  = quantize((double)v_max);    // clamp bound

    // Reciprocal of 2*dx (the central-difference denominator): dsdx = (s_e - s_w)
    // * recip(2*dx). Precomputed in double once -> deterministic (S1 §3).
    const int64_t recip_two_dx = make_recip(2.0 * dx_d);

    // Reciprocal of dx for the flux->depth-delta: dq = (dt/dx) * flux. We fold
    // dt and 1/dx into one reciprocal-style constant applied to the WIDE flux:
    //   dq_q16 = narrow( flux_wide * (dt/dx) )  in Q16.16.
    // flux_wide is Q32.32 (mul_wide of two Q16.16). Multiplying by the real
    // (dt/dx) and narrowing to Q16.16 is one shared truncation per face (the
    // conservation point). We do it as: dq = recip_mul_wideflux(flux_wide).
    // Implementation: dt/dx as a Q16.16 constant, then narrow the triple product.
    const q16 dt_over_dx_q = quantize(dt_d / dx_d);  // (dt/dx), dimensionless (Q16.16)

    // depth_eps as Q16.16 (snap-to-zero floor).
    const q16 depth_eps_q = quantize((double)depth_eps);

    // ---- Nullable floor: a flat-zero stand-in (Q16.16 zeros == int 0). ----
    // floor_height nullptr -> flat zero. (No /fp:fast rounding concern now — the
    // core is integer; nullptr vs explicit-zeros is bit-identical trivially.)
    auto floor_at = [&](int i) -> q16 {
        return floor_height ? floor_height[i] : 0;
    };

    // ---- 1. surface potential (Q16.16 metres) -----------------------------
    // tilt(x,y) = tan(tilt_x)*(x - cx)*dx + tan(tilt_y)*(y - cy)*dx.
    // tan via the low-degree odd poly (fixed_point.h::tan_poly) on the CLAMPED
    // tilt (|tilt| <= 35deg so the degree-5 series stays < 0.1% accurate).
    const double TILT_MAX = 0.610865;  // 35 deg in radians
    double txd = std::max(-TILT_MAX, std::min(TILT_MAX, (double)tilt_x));
    double tyd = std::max(-TILT_MAX, std::min(TILT_MAX, (double)tilt_y));
    const q16 tan_tx = tan_poly(quantize(txd));
    const q16 tan_ty = tan_poly(quantize(tyd));
    // cx, cy and dx fold into a per-column / per-row Q16.16 tilt slope. We
    // precompute the per-tile tilt as tan_t * ((idx - c)*dx). The ((idx-c)*dx)
    // factor is a real-valued position in metres; quantize per index.
    const double cx = 0.5 * (double)w;
    const double cy = 0.5 * (double)h;

    std::vector<q16> surface;
    surface.swap(surface_);
    surface.resize(n);

    // Head-term gate: with k_p == 0 the pressure fields are NEVER read (exact,
    // identical to passing none). With k_p != 0 the head term is a FLOAT BRIDGE
    // (atmosphere/wave_p are still float in S1) — computed in float and quantized
    // into the Q16.16 surface. Marked below.
    const bool head_on = (k_p != 0.0f);
    const float kp_f = k_p;

    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        // per-row Y tilt component (Q16.16 metres):
        const q16 tilt_row = mul_q16(tan_ty, quantize(((double)y - cy) * dx_d));
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            const q16 tilt_col = mul_q16(tan_tx, quantize(((double)x - cx) * dx_d));
            q16 s = floor_at(i) + tilt_col + tilt_row + water_depth[i];
            if (head_on) {
                // FLOAT BRIDGE until S2: atmosphere/wave_p are still float fields.
                // Read them in float, form the head term k_p*(atm+wave_p) in
                // float, quantize to Q16.16, add into the integer surface. (When
                // S2 makes atmosphere/wave_p integer this bridge becomes a pure
                // integer add.) atm_p/wp_p substituted to 0 if null (gated).
                const float atm_v = atmosphere ? atmosphere[i] : 0.0f;
                const float wp_v  = wave_p ? wave_p[i] : 0.0f;
                const float head_f = kp_f * (atm_v + wp_v);
                s += quantize((double)head_f);
            }
            surface[i] = s;
        }
    }

    // ---- 2. damped explicit velocity kick (central difference; Neumann mirror
    //         of the centre value at solid/out-of-bounds neighbours) ----------
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) {
                flow_vx[i] = 0;
                flow_vy[i] = 0;
                continue;
            }
            const q16 s_c = surface[i];
            const q16 s_e = (x < w - 1 && !solid[i + 1]) ? surface[i + 1] : s_c;
            const q16 s_w = (x > 0     && !solid[i - 1]) ? surface[i - 1] : s_c;
            const q16 s_s = (y < h - 1 && !solid[i + w]) ? surface[i + w] : s_c;
            const q16 s_n = (y > 0     && !solid[i - w]) ? surface[i - w] : s_c;
            // dsdx = (s_e - s_w) / (2*dx) — reciprocal multiply (Q16.16).
            const q16 dsdx = recip_mul((q16)(s_e - s_w), recip_two_dx);
            const q16 dsdy = recip_mul((q16)(s_s - s_n), recip_two_dx);
            // vx += dt*(-g*dsdx - damping*vx) == vx - g_dt*dsdx - damp_dt*vx
            q16 vx = (q16)((int64_t)flow_vx[i]
                           - mul_q16(g_dt_q, dsdx)
                           - mul_q16(damp_dt_q, flow_vx[i]));
            q16 vy = (q16)((int64_t)flow_vy[i]
                           - mul_q16(g_dt_q, dsdy)
                           - mul_q16(damp_dt_q, flow_vy[i]));
            // componentwise clamp to +-v_max.
            vx = std::max(-v_max_q, std::min(v_max_q, vx));
            vy = std::max(-v_max_q, std::min(v_max_q, vy));
            flow_vx[i] = vx;
            flow_vy[i] = vy;
        }
    }

    // ---- 3. donor-cell upwind face fluxes (PRE-update depth, gather) -------
    // The face flux is gathered ONCE as a WIDE int64 (Q32.32 = mul_wide of two
    // Q16.16). Positive flux moves mass toward +x / +y. Solid faces carry no
    // flux; border faces do not exist (grid border = wall).
    std::vector<int64_t> fx, fy;
    fx.swap(fx_);  fy.swap(fy_);
    fx.assign(n, 0);  fy.assign(n, 0);
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (x < w - 1 && !solid[i] && !solid[i + 1]) {
                // v_face = 0.5*(vx[i]+vx[i+1]) -> (sum)>>1, exact in Q16.16.
                const q16 v_face = (q16)(((int64_t)flow_vx[i] + flow_vx[i + 1]) >> 1);
                const q16 donor = (v_face > 0) ? water_depth[i] : water_depth[i + 1];
                fx[i] = mul_wide(v_face, donor);   // Q32.32 wide flux
            }
            if (y < h - 1 && !solid[i] && !solid[i + w]) {
                const q16 v_face = (q16)(((int64_t)flow_vy[i] + flow_vy[i + w]) >> 1);
                const q16 donor = (v_face > 0) ? water_depth[i] : water_depth[i + w];
                fy[i] = mul_wide(v_face, donor);
            }
        }
    }

    // ---- per-face depth-delta dq (Q16.16) — the CONSERVATIVE unit ----------
    // dq = (dt/dx) * flux. flux is Q32.32 wide; multiply by the Q16.16 (dt/dx)
    // and narrow back to Q16.16 in ONE shared truncation per face. This single
    // value is applied +dq to the donor-side neighbour and -dq to the donor in
    // the divergence pass, so the >>16 narrow can NEVER create/destroy mass.
    //   dq_q16 = (flux_wide * dt_over_dx_q) >> (16 + 16)
    //          (flux_wide carries 2^32; *Q16.16 adds 2^16; >> 32 lands at 2^16)
    auto flux_to_dq = [&](int64_t flux_wide) -> q16 {
        // flux_wide (Q32.32) * dt_over_dx_q (Q16.16) = Q48.48-ish in __int128;
        // >> 32 leaves Q16.16. Use a 128-bit intermediate to avoid overflow for
        // large fluxes (v_max * deep depth). dt_over_dx_q is small (<1 typically).
#if defined(__SIZEOF_INT128__)
        __int128 p = (__int128)flux_wide * (__int128)dt_over_dx_q;
        return (q16)(p >> 32);
#else
        // MSVC path: flux_wide * dt_over_dx_q via the double-width helper. We
        // reuse recip_mul's 128-bit shift by composing: narrow flux to Q16.16
        // first (>>16, the conservative truncation), then mul_q16 by dt_over_dx_q
        // (a second >>16). Two truncations vs one, but applied to the SINGLE
        // gathered face value -> still conservative (same value both sides).
        const q16 flux_q16 = (q16)(flux_wide >> 16);
        return mul_q16(flux_q16, dt_over_dx_q);
#endif
    };

    std::vector<q16> dq_e, dq_s;
    dq_e.swap(dq_e_);  dq_s.swap(dq_s_);
    dq_e.assign(n, 0);  dq_s.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        if (fx[i] != 0) dq_e[i] = flux_to_dq(fx[i]);
        if (fy[i] != 0) dq_s[i] = flux_to_dq(fy[i]);
    }

    // ---- per-cell OUTFLOW LIMITER (mass-exactness) ------------------------
    // A cell can be donor on up to 4 faces; worst-case its total OUTGOING dq
    // exceeds its depth and the non-negative clamp below would CREATE mass. Per
    // cell: out_sum = sum of OUTGOING dq magnitudes (in Q16.16 depth units now);
    // if out_sum > depth, scale THAT CELL'S outgoing dq by depth/out_sum.
    //   scale = (depth << 16) / out_sum    (a deterministic integer divide; the
    //   constant dx/(dt) factor is already folded into dq, so only the dynamic
    //   depth/out_sum ratio remains — an exact integer divide, identical
    //   cross-machine; the plan's "reciprocal" note was the folded dt/dx part).
    std::vector<q16> scale_q;
    scale_q.swap(scale_q_);
    scale_q.assign(n, FP_ONE);   // FP_ONE == 1.0 (unlimited) default IS read below
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            int64_t out_sum = 0;
            if (x < w - 1 && dq_e[i] > 0)     out_sum += dq_e[i];       // east, leaving
            if (x > 0     && dq_e[i - 1] < 0) out_sum -= dq_e[i - 1];   // west, leaving
            if (y < h - 1 && dq_s[i] > 0)     out_sum += dq_s[i];       // south, leaving
            if (y > 0     && dq_s[i - w] < 0) out_sum -= dq_s[i - w];   // north, leaving
            if (out_sum > (int64_t)water_depth[i]) {
                // scale = depth / out_sum, in Q16.16 = (depth << 16) / out_sum.
                // depth >= 0 and out_sum > depth >= 0, so 0 <= scale < FP_ONE.
                scale_q[i] = (q16)(((int64_t)water_depth[i] << FP_SHIFT) / out_sum);
            }
        }
    }
    // Scale each face's dq by its DONOR cell's factor (FP_ONE when unlimited).
    // The SAME scaled dq is used for both cells in the apply -> conservation.
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (x < w - 1 && dq_e[i] != 0)
                dq_e[i] = mul_q16(dq_e[i], (dq_e[i] > 0) ? scale_q[i] : scale_q[i + 1]);
            if (y < h - 1 && dq_s[i] != 0)
                dq_s[i] = mul_q16(dq_s[i], (dq_s[i] > 0) ? scale_q[i] : scale_q[i + w]);
        }
    }

    // ---- apply divergence (gather-then-apply; the conservative ± form) ----
    // depth[i] -= (dq_e[i] - dq_e[i-1]) + (dq_s[i] - dq_s[i-w]).
    // dq_e[i] is the SAME value removed from i and added to i+1 (as its dq_e[i-1]
    // inflow), so total mass is conserved to the LSB regardless of rounding.
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            const q16 d_e = (x < w - 1) ? dq_e[i]     : 0;
            const q16 d_w = (x > 0)     ? dq_e[i - 1] : 0;
            const q16 d_s = (y < h - 1) ? dq_s[i]     : 0;
            const q16 d_n = (y > 0)     ? dq_s[i - w] : 0;
            water_depth[i] = (q16)((int64_t)water_depth[i]
                                   - ((int64_t)(d_e - d_w) + (int64_t)(d_s - d_n)));
        }
    }

    // ---- 4. clamps: depth >= 0; zero on solid; snap to zero below depth_eps -
    // NOTE: these clamps can BREAK strict conservation (a negative depth pinned
    // to 0 adds mass; an eps-snap removes it) — EXACTLY as the float code did.
    // For the P2 sealed-flood conservation test, depth stays positive and above
    // eps, so neither clamp fires and Σdepth is bit-conserved. (The clamps are
    // the documented wet/dry shore handling, not a conservation leak in the
    // sealed case.)
    for (int i = 0; i < n; ++i) {
        q16 d = std::max(water_depth[i], (q16)0);
        if (solid[i] || d < depth_eps_q) d = 0;
        water_depth[i] = d;
    }

    // Retain all scratch storage for the next step (swap idiom; no per-step alloc).
    surface.swap(surface_);
    fx.swap(fx_);  fy.swap(fy_);
    dq_e.swap(dq_e_);  dq_s.swap(dq_s_);
    scale_q.swap(scale_q_);
}

void WaterSolver::step_ripple(float* ripple, float* ripple_v,
                              const q16*  water_depth,
                              const float* wave_p,
                              const bool*  solid,
                              int h, int w, float dt) const {
    // RENDER-ONLY: stays FLOAT. The only contact with the integer depth is the
    // DEQUANTIZE at the c2 = g*min(depth, h_cap) read and at the wet/dry gate.
    const int n = h * w;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float Q = (float)FP_ONE;   // 65536 — dequantize divisor

    // --- 1. splash source FIRST: a wave_p blast over wet tiles kicks the
    //        surface. wave_p nullable -> no splash. Wet tiles only.
    if (wave_p) {
        for (int i = 0; i < n; ++i) {
            // DEQUANTIZE depth for the wet test (depth > 0 <=> int depth > 0).
            if (water_depth[i] > 0) ripple_v[i] += k_splash * wave_p[i];
        }
    }

    // --- 2. damped velocity kick from the PRE-update ripple (gather-then-apply).
    //        c2 = g*min(depth, h_cap) reads water_depth DEQUANTIZED (int->float).
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) continue;                 // zeroed in pass 3
            const float r_c = ripple[i];
            const float r_e = (x < w - 1 && !solid[i + 1]) ? ripple[i + 1] : r_c;
            const float r_w = (x > 0     && !solid[i - 1]) ? ripple[i - 1] : r_c;
            const float r_s = (y < h - 1 && !solid[i + w]) ? ripple[i + w] : r_c;
            const float r_n = (y > 0     && !solid[i - w]) ? ripple[i - w] : r_c;
            const float lap = (r_n + r_s + r_e + r_w - 4.0f * r_c) * inv_dx2;
            const float depth_m = (float)water_depth[i] / Q;   // DEQUANTIZE
            const float c2  = g * std::min(depth_m, h_cap);
            ripple_v[i] += dt * (c2 * lap - gamma_r * ripple_v[i]);
        }
    }

    // --- 3. drift, clamp AFTER the drift, zero on dry/solid ---
    for (int i = 0; i < n; ++i) {
        if (solid[i] || water_depth[i] <= 0) {     // DEQUANTIZE (depth<=0 gate)
            ripple[i]   = 0.0f;
            ripple_v[i] = 0.0f;
            continue;
        }
        const float depth_m = (float)water_depth[i] / Q;       // DEQUANTIZE
        const float amp = k_amp * depth_m;
        ripple[i] = std::clamp(ripple[i] + dt * ripple_v[i], -amp, amp);
    }
}
