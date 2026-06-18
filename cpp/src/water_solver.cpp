#include "water_solver.h"
#include <cmath>
#include <algorithm>
#include <vector>

float WaterSolver::max_dt() const {
    // Plain wave CFL at the reference column, with the head-term margin (W4):
    // linearised, the pipe model is a damped wave with c = sqrt(g*depth); the
    // pressure head stiffens the restoring force by (1 + k_p*P_REF/HEAD_REF).
    return 0.5f * dx / std::sqrt(g * h_ref * (1.0f + k_p * P_REF / HEAD_REF));
}

float WaterSolver::ripple_max_dt() const {
    // STATIC bound at the deep-water cap (canon §6): c <= sqrt(g*h_cap)
    // everywhere by construction, so 0.5*dx/c_cap (~106 ms at dx = 1/3) is a
    // true CFL for step_ripple — one call per tick at any tick rate we use.
    return 0.5f * dx / std::sqrt(g * h_cap);
}

void WaterSolver::step(float* water_depth, float* flow_vx, float* flow_vy,
                       const float* floor_height,
                       const float* atmosphere,
                       const float* wave_p,
                       const bool*  solid,
                       int h, int w, float dt,
                       float tilt_x, float tilt_y) const {
    const int n = h * w;
    const float two_dx     = 2.0f * dx;
    const float dt_over_dx = dt / dx;

    // Nullable fields: substitute an internal zero buffer so the nullptr run
    // executes the IDENTICAL code path (and rounding) as an explicit-zeros
    // run — under /fp:fast a "+ 0.0f constant" branch is folded differently
    // and breaks the plan's bit-identity contract (W1 test 6). The k_p gate
    // stays exact: with k_p == 0 the pressure pointers are NEVER touched.
    // Reused scratch via swap idiom: holds zeros and is read as zeros, so it
    // stays zero across steps; re-assign zeros only when it is first needed at
    // this size. (Swapped back at end of step.)
    std::vector<float> zeros_scratch;
    zeros_scratch.swap(zeros_scratch_);
    auto zeros = [&]() -> const float* {
        if (zeros_scratch.size() != (size_t)n) zeros_scratch.assign(n, 0.0f);
        return zeros_scratch.data();
    };
    const float* floor_p = floor_height ? floor_height : zeros();
    const float* atm_p = atmosphere;
    const float* wp_p  = wave_p;
    if (k_p != 0.0f) {
        if (!atm_p) atm_p = zeros();
        if (!wp_p)  wp_p  = zeros();
    }

    // --- 1. surface potential (per cell; metres throughout) ---
    // tilt(x,y) = tan(tilt_x)*(x - cx)*dx + tan(tilt_y)*(y - cy)*dx,
    // cx,cy = grid centre (W/2, H/2).
    const float tan_tx = std::tan(tilt_x);
    const float tan_ty = std::tan(tilt_y);
    const float cx = 0.5f * static_cast<float>(w);
    const float cy = 0.5f * static_cast<float>(h);

    // Reused scratch via swap idiom: every surface[i] is written below before
    // pass 2 reads it, so just size it to n (no re-init of contents needed).
    std::vector<float> surface;
    surface.swap(surface_);
    surface.resize(n);
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        const float tilt_row = tan_ty * (static_cast<float>(y) - cy) * dx;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            const float tilt_i = tan_tx * (static_cast<float>(x) - cx) * dx + tilt_row;
            float s = floor_p[i] + tilt_i + water_depth[i];
            if (k_p != 0.0f) {
                // GATED: with k_p == 0 the pressure fields are NEVER read —
                // bit-identical to passing no pressure fields at all.
                s += k_p * (atm_p[i] + wp_p[i]);
            }
            surface[i] = s;
        }
    }

    // --- 2. damped explicit velocity kick (central difference; Neumann
    //        MIRROR of the centre value at solid neighbours; out-of-bounds
    //        neighbours are treated as solid — grid border = wall) ---
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) {
                flow_vx[i] = 0.0f;
                flow_vy[i] = 0.0f;
                continue;
            }
            const float s_c = surface[i];
            const float s_e = (x < w - 1 && !solid[i + 1]) ? surface[i + 1] : s_c;
            const float s_w = (x > 0     && !solid[i - 1]) ? surface[i - 1] : s_c;
            const float s_s = (y < h - 1 && !solid[i + w]) ? surface[i + w] : s_c;
            const float s_n = (y > 0     && !solid[i - w]) ? surface[i - w] : s_c;
            const float dsdx = (s_e - s_w) / two_dx;
            const float dsdy = (s_s - s_n) / two_dx;
            const float vx = flow_vx[i] + dt * (-g * dsdx - damping * flow_vx[i]);
            const float vy = flow_vy[i] + dt * (-g * dsdy - damping * flow_vy[i]);
            flow_vx[i] = std::clamp(vx, -v_max, v_max);
            flow_vy[i] = std::clamp(vy, -v_max, v_max);
        }
    }

    // --- 3. donor-cell upwind face fluxes from PRE-update depth (gather) ---
    // fx[i] = flux across the face between (y,x) and (y,x+1)  (0 for x == w-1)
    // fy[i] = flux across the face between (y,x) and (y+1,x)  (0 for y == h-1)
    // Positive flux moves mass toward +x / +y. Solid faces carry no flux;
    // border faces do not exist (grid border = wall).
    // Reused scratch via swap idiom: only non-solid/non-border faces are
    // written; the remaining 0s ARE read by the outflow limiter and divergence
    // passes, so re-assign 0 each step.
    std::vector<float> fx, fy;
    fx.swap(fx_);  fy.swap(fy_);
    fx.assign(n, 0.0f);  fy.assign(n, 0.0f);
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (x < w - 1 && !solid[i] && !solid[i + 1]) {
                const float v_face = 0.5f * (flow_vx[i] + flow_vx[i + 1]);
                fx[i] = v_face * (v_face > 0.0f ? water_depth[i] : water_depth[i + 1]);
            }
            if (y < h - 1 && !solid[i] && !solid[i + w]) {
                const float v_face = 0.5f * (flow_vy[i] + flow_vy[i + w]);
                fy[i] = v_face * (v_face > 0.0f ? water_depth[i] : water_depth[i + w]);
            }
        }
    }

    // --- per-cell OUTFLOW LIMITER (mass-exactness) ---
    // A cell can be donor on up to 4 faces, so worst-case outflow exceeds its
    // depth and the non-negative clamp below would CREATE mass. Per cell:
    // out_sum = sum of its outgoing fluxes; if out_sum*dt/dx > depth, scale
    // THAT CELL'S outgoing fluxes by depth*dx/(dt*out_sum).
    // Reused scratch via swap idiom: the default 1.0 (unlimited cell) IS read
    // in the scaling pass below, so re-assign 1.0 each step.
    std::vector<float> scale;
    scale.swap(scale_);
    scale.assign(n, 1.0f);
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            float out_sum = 0.0f;
            if (x < w - 1 && fx[i] > 0.0f)     out_sum += fx[i];      // east face, leaving
            if (x > 0     && fx[i - 1] < 0.0f) out_sum -= fx[i - 1];  // west face, leaving
            if (y < h - 1 && fy[i] > 0.0f)     out_sum += fy[i];      // south face, leaving
            if (y > 0     && fy[i - w] < 0.0f) out_sum -= fy[i - w];  // north face, leaving
            if (out_sum * dt_over_dx > water_depth[i]) {
                scale[i] = water_depth[i] * dx / (dt * out_sum);
            }
        }
    }
    // Scale each face flux by its DONOR cell's factor (1.0 when unlimited).
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (x < w - 1) fx[i] *= (fx[i] > 0.0f) ? scale[i] : scale[i + 1];
            if (y < h - 1) fy[i] *= (fy[i] > 0.0f) ? scale[i] : scale[i + w];
        }
    }

    // --- apply divergence (one pass; gather-then-apply, deterministic) ---
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            const float f_e = (x < w - 1) ? fx[i]     : 0.0f;
            const float f_w = (x > 0)     ? fx[i - 1] : 0.0f;
            const float f_s = (y < h - 1) ? fy[i]     : 0.0f;
            const float f_n = (y > 0)     ? fy[i - w] : 0.0f;
            water_depth[i] -= dt_over_dx * ((f_e - f_w) + (f_s - f_n));
        }
    }

    // --- 4. clamps: depth >= 0; zero on solid; snap to zero below depth_eps ---
    for (int i = 0; i < n; ++i) {
        float d = std::max(water_depth[i], 0.0f);
        if (solid[i] || d < depth_eps) d = 0.0f;
        water_depth[i] = d;
    }

    // Retain all scratch storage for the next step (swap idiom; no per-step alloc).
    zeros_scratch.swap(zeros_scratch_);
    surface.swap(surface_);
    fx.swap(fx_);  fy.swap(fy_);
    scale.swap(scale_);
}

void WaterSolver::step_ripple(float* ripple, float* ripple_v,
                              const float* water_depth,
                              const float* wave_p,
                              const bool*  solid,
                              int h, int w, float dt) const {
    const int n = h * w;
    const float inv_dx2 = 1.0f / (dx * dx);

    // --- 1. splash source FIRST (plan W6a): a wave_p blast passing over wet
    //        tiles kicks the surface. wave_p nullable -> no splash and the
    //        pointer is NEVER read (W1's nullable discipline). k_splash is
    //        the pure feel dial. Wet tiles only — dry/solid get re-zeroed in
    //        pass 3 regardless.
    if (wave_p) {
        for (int i = 0; i < n; ++i) {
            if (water_depth[i] > 0.0f) ripple_v[i] += k_splash * wave_p[i];
        }
    }

    // --- 2. damped velocity kick from the PRE-update ripple (gather-then-
    //        apply: `ripple` is only written in pass 3, so every laplacian
    //        here reads the pre-update field — deterministic, no sweep-order
    //        dependence). c2 = g*min(depth, h_cap) is SI m^2/s^2 (the deep-
    //        water splice, canon §6), so the laplacian REQUIRES the 1/dx^2 —
    //        unlike the atmosphere's tile-unit wave; do not copy-paste.
    //        Neumann mirror of the centre value at solid / out-of-bounds
    //        neighbours (grid border = wall), exactly as step() above. DRY
    //        open neighbours are read as-is — their ripple is 0 (pass 3
    //        invariant), which with c2 -> 0 makes shores self-absorbing.
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
            const float c2  = g * std::min(water_depth[i], h_cap);
            ripple_v[i] += dt * (c2 * lap - gamma_r * ripple_v[i]);
        }
    }

    // --- 3. drift, clamp AFTER the drift, zero on dry/solid ---
    //        |ripple| <= k_amp*depth: waves no taller than the water — the
    //        hard amplitude guarantee; gamma_r eats clamp-injected energy.
    //        Zeroing both fields where depth == 0 or solid is the canon §6
    //        rule that ripple exists only on standing water (and keeps the
    //        dry-tile invariant pass 2 relies on).
    for (int i = 0; i < n; ++i) {
        if (solid[i] || water_depth[i] <= 0.0f) {
            ripple[i]   = 0.0f;
            ripple_v[i] = 0.0f;
            continue;
        }
        const float amp = k_amp * water_depth[i];
        ripple[i] = std::clamp(ripple[i] + dt * ripple_v[i], -amp, amp);
    }
}
