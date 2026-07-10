#include "bulk_transport.h"
#include "fixed_point.h"
#include <algorithm>
#include <vector>
#include <cstdint>
#if !defined(__SIZEOF_INT128__) && defined(_MSC_VER)
#include <intrin.h>   // _mul128 for the one-truncation MSVC flux_to_dq path
#endif

using namespace fixedpoint;

namespace {

// One shared truncation: flux_wide (Q32.32, a mul_wide(v_face, N_donor)) times
// a per-face Q16.16 coefficient (face_permeability * dt, dx == 1 tile) ->
// Q16.16, via a 128-bit intermediate. Ported verbatim from water_solver.cpp's
// flux_to_dq (S1 §2 / P2), generalized to a runtime PER-FACE coefficient (water's
// coefficient was the single constant dt/dx; here it additionally carries the
// per-face permeability gate, so every face gets its own coefficient — computed
// once per face, applied as ONE truncation, exactly like water's dt_over_dx_q).
inline q16 flux_to_dq(int64_t flux_wide, q16 coeff_q) {
#if defined(__SIZEOF_INT128__)
    __int128 p = (__int128)flux_wide * (__int128)coeff_q;
    return (q16)(p >> 32);
#else
    long long hi;
    long long lo = _mul128((long long)flux_wide, (long long)coeff_q, &hi);
    unsigned long long ulo = (unsigned long long)lo;
    long long res = (long long)((ulo >> 32) | ((unsigned long long)hi << (64 - 32)));
    return (q16)res;
#endif
}

}  // namespace

void bulk_flux_transport(
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const int32_t* wind_x, const int32_t* wind_y,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt) {
    const int n = h * w;
    // dt as Q16.16 (dx == 1 tile, the smoke/wind convention — see
    // smoke_dynamics.cpp's wind-only advection, which likewise has no
    // separate dx divide).
    const q16 dt_q = quantize((double)dt);

    // Per-face scratch, allocated once per call and reused across the
    // (typically two) conservative planes below. P1 is CPU-only and not the
    // hot path (the sub-tick substep loops it does NOT ride), so a plain
    // local std::vector — no cross-call retained scratch — keeps this
    // function simple and self-contained; the GPU-prep no-per-tick-alloc
    // discipline is a P6 concern.
    std::vector<q16> dq_e(n), dq_s(n), scale_q(n);

    for (int gi = 0; gi < n_gases; ++gi) {
        if (!gas_conservative[gi]) continue;
        int32_t* N = gas + (size_t)gi * n;

        // Skip an all-zero plane (nothing to transport, matches numpy .any()).
        bool any = false;
        for (int i = 0; i < n; ++i) { if (N[i] != 0) { any = true; break; } }
        if (!any) continue;

        std::fill(dq_e.begin(), dq_e.end(), (q16)0);
        std::fill(dq_s.begin(), dq_s.end(), (q16)0);
        std::fill(scale_q.begin(), scale_q.end(), FP_ONE);

        // ---- 1. donor-cell upwind face fluxes (pre-update N, gather-once) ----
        // Face permeability gates flux exactly like the smoke/atmosphere
        // stencils (face = min(perm_self, perm_neighbor)). ALSO explicitly
        // require `!solid[]` on both sides — the same CONSERVATION GUARD
        // atmosphere_solver.cpp's GS documents ("a face into an excluded+
        // zeroed cell is 0 ... a one-sided flux into a p=0 Dirichlet cell
        // would destroy mass"): a solid tile's STATIC permeability is 0, but
        // the LIVE `dyn_permeability` this function reads is a per-tick
        // derived field (stamp_units) — trusting it alone to always be 0 on
        // solid would silently reopen the mass-destroying leak the S1/3a fix
        // closed elsewhere if a future caller ever passes it un-stamped.
        // Belt-and-suspenders costs nothing here (solid is already loaded).
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (solid[i]) continue;
                if (x < w - 1 && !solid[i + 1]) {
                    const float face_f = std::min(dyn_permeability[i], dyn_permeability[i + 1]);
                    if (face_f > 0.0f) {
                        const q16 face_q = quantize((double)face_f);
                        const q16 v_face = (q16)(((int64_t)wind_x[i] + wind_x[i + 1]) >> 1);
                        const q16 donor = (v_face > 0) ? N[i] : N[i + 1];
                        const int64_t flux_wide = mul_wide(v_face, donor);   // Q32.32
                        const q16 coeff_q = mul_q16(face_q, dt_q);
                        dq_e[i] = flux_to_dq(flux_wide, coeff_q);
                    }
                }
                if (y < h - 1 && !solid[i + w]) {
                    const float face_f = std::min(dyn_permeability[i], dyn_permeability[i + w]);
                    if (face_f > 0.0f) {
                        const q16 face_q = quantize((double)face_f);
                        const q16 v_face = (q16)(((int64_t)wind_y[i] + wind_y[i + w]) >> 1);
                        const q16 donor = (v_face > 0) ? N[i] : N[i + w];
                        const int64_t flux_wide = mul_wide(v_face, donor);
                        const q16 coeff_q = mul_q16(face_q, dt_q);
                        dq_s[i] = flux_to_dq(flux_wide, coeff_q);
                    }
                }
            }
        }

        // ---- 2. per-cell OUTFLOW LIMITER (mass-exactness) --------------------
        // Ported verbatim from water_solver.cpp's conservation-critical block: a
        // cell can be donor on up to 4 faces; bound its total outgoing dq to
        // <= its own N so the non-negative clamp below never creates mass.
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                int64_t out_sum = 0;
                if (x < w - 1 && dq_e[i] > 0)     out_sum += dq_e[i];       // east, leaving
                if (x > 0     && dq_e[i - 1] < 0) out_sum -= dq_e[i - 1];   // west, leaving
                if (y < h - 1 && dq_s[i] > 0)     out_sum += dq_s[i];       // south, leaving
                if (y > 0     && dq_s[i - w] < 0) out_sum -= dq_s[i - w];   // north, leaving
                if (out_sum > (int64_t)N[i]) {
                    // scale = N / out_sum, in Q16.16 = (N << 16) / out_sum.
                    scale_q[i] = (q16)(((int64_t)N[i] << FP_SHIFT) / out_sum);
                }
            }
        }
        // Scale each face's dq by its DONOR cell's factor (FP_ONE == unlimited).
        // scale_mag (not mul_q16) truncates on the MAGNITUDE — the same
        // over-drain guard water_solver.cpp documents at length.
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (x < w - 1 && dq_e[i] != 0)
                    dq_e[i] = scale_mag(dq_e[i], (dq_e[i] > 0) ? scale_q[i] : scale_q[i + 1]);
                if (y < h - 1 && dq_s[i] != 0)
                    dq_s[i] = scale_mag(dq_s[i], (dq_s[i] > 0) ? scale_q[i] : scale_q[i + w]);
            }
        }

        // ---- 3. apply divergence (gather-then-apply; the conservative ± form) --
        // dq_e[i] is the SAME value removed from i and added to i+1 (as its
        // dq_e[i-1] inflow) — total mass is conserved to the LSB regardless of
        // rounding, exactly like water_solver.cpp's divergence apply.
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                const q16 d_e = (x < w - 1) ? dq_e[i]     : 0;
                const q16 d_w = (x > 0)     ? dq_e[i - 1] : 0;
                const q16 d_s = (y < h - 1) ? dq_s[i]     : 0;
                const q16 d_n = (y > 0)     ? dq_s[i - w] : 0;
                N[i] = (int32_t)((int64_t)N[i]
                                 - ((int64_t)(d_e - d_w) + (int64_t)(d_s - d_n)));
            }
        }

        // ---- 4. clamps: N >= 0; zero on solid AND vacuum ----------------------
        // Solid never holds N (defensive — a stale value from before a tile
        // became solid must not linger). Vacuum is the DELIBERATE sink: mass
        // that flowed there this tick legitimately leaves the system (breach
        // venting), matching §2.2's "N zeroed at vacuum" rule. Both zeroings
        // are NOT part of the sealed-room conservation gate's domain (a sealed
        // room has no vacuum cells), so they never fire there.
        for (int i = 0; i < n; ++i) {
            if (solid[i] || is_vacuum[i]) {
                N[i] = 0;
            } else if (N[i] < 0) {
                N[i] = 0;
            }
        }
    }
}
