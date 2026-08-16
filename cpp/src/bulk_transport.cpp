#include "bulk_transport.h"
#include "fixed_point.h"
#include <algorithm>
#include <cassert>
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
    // Legacy entry (pybind/P1-test path): hoist the per-face coefficient
    // exactly as the caller-cached fast path does, then forward. The
    // arithmetic per face is byte-identical to the original inline form
    // (same min/quantize/mul_q16 chain, evaluated once instead of per
    // plane) — see bulk_flux_transport_cached's header comment.
    const int n = h * w;
    const q16 dt_q = quantize((double)dt);
    std::vector<q16> coeffE(n, 0), coeffS(n, 0);
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (solid[i]) continue;
            if (x < w - 1 && !solid[i + 1]) {
                const float face_f = std::min(dyn_permeability[i], dyn_permeability[i + 1]);
                if (face_f > 0.0f)
                    coeffE[i] = mul_q16(quantize((double)face_f), dt_q);
            }
            if (y < h - 1 && !solid[i + w]) {
                const float face_f = std::min(dyn_permeability[i], dyn_permeability[i + w]);
                if (face_f > 0.0f)
                    coeffS[i] = mul_q16(quantize((double)face_f), dt_q);
            }
        }
    }
    bulk_flux_transport_cached(gas, gas_conservative, n_gases,
                               wind_x, wind_y, solid, is_vacuum,
                               coeffE.data(), coeffS.data(), h, w);
}

void bulk_flux_transport_cached(
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const int32_t* wind_x, const int32_t* wind_y,
        const bool* solid, const bool* is_vacuum,
        const int32_t* coeffE, const int32_t* coeffS,
        int h, int w,
        const bool* is_ambient, const int32_t* n_amb, int64_t* boundary_flux,
        int64_t* dqsum_e, int64_t* dqsum_s) {
    const int n = h * w;
    // BC: dormancy BY BRANCH — every ambient edit below is gated on this flag,
    // so a space map (is_ambient == nullptr) takes the byte-identical path.
    const bool ambient_mode = (is_ambient != nullptr);
    // P-E1 (design §2.1.3): the APPLIED per-face dq accumulator planes. Zeroed
    // here (this entry OWNS them for the substep) and summed over the
    // conservative planes below. nullptr -> dormancy by branch, the legacy
    // byte-identical path.
    const bool accum_dq = (dqsum_e != nullptr && dqsum_s != nullptr);
    if (accum_dq) {
        for (int i = 0; i < n; ++i) { dqsum_e[i] = 0; dqsum_s[i] = 0; }
    }

    // Reused scratch (EOS P3 micro-opt: this entry rides the substep loop —
    // up to N_SUB_MAX calls/tick — so the three per-call vector allocations
    // of the P1-era entry are hoisted into thread_local storage; contents
    // are fully re-initialized below, so the reuse is arithmetic-neutral).
    static thread_local std::vector<q16> dq_e, dq_s, scale_q;
    if ((int)dq_e.size() != n) { dq_e.assign(n, 0); dq_s.assign(n, 0); scale_q.assign(n, FP_ONE); }

    for (int gi = 0; gi < n_gases; ++gi) {
        if (!gas_conservative[gi]) continue;
        int32_t* N = gas + (size_t)gi * n;

        // Skip an all-zero plane (nothing to transport, matches numpy .any()).
        // BC build-rider (ii): on an AMBIENT map the per-substep ring reset must
        // still run even for an all-zero conservative plane (a degenerate
        // o2_frac in {0,1} case), so never take the skip when ambient_mode is on
        // — the transport is a no-op on the zeros and the reset below sets the
        // ring. On a space map ambient_mode is false, so this is the exact
        // legacy skip (byte-identical). (The non-degenerate ambient map seeds
        // both bulk planes to N_amb > 0, so `any` is true anyway.)
        bool any = false;
        for (int i = 0; i < n; ++i) { if (N[i] != 0) { any = true; break; } }
        if (!any && !ambient_mode) continue;

        std::fill(dq_e.begin(), dq_e.end(), (q16)0);
        std::fill(dq_s.begin(), dq_s.end(), (q16)0);
        std::fill(scale_q.begin(), scale_q.end(), FP_ONE);

        // ---- 1. donor-cell upwind face fluxes (pre-update N, gather-once) ----
        // Same conservation-guard structure as the legacy entry (solid gate +
        // permeability gate), with the face coefficient PRECOMPUTED: a sealed
        // or solid face carries coeff 0, which produces the same zero dq the
        // legacy `face_f > 0` branch-skip did (flux_to_dq(x, 0) == 0).
        for (int y = 0; y < h; ++y) {
            const int row = y * w;
            for (int x = 0; x < w; ++x) {
                const int i = row + x;
                if (solid[i]) continue;
                if (x < w - 1 && coeffE[i] != 0) {
                    const q16 v_face = (q16)(((int64_t)wind_x[i] + wind_x[i + 1]) >> 1);
                    const q16 donor = (v_face > 0) ? N[i] : N[i + 1];
                    const int64_t flux_wide = mul_wide(v_face, donor);   // Q32.32
                    dq_e[i] = flux_to_dq(flux_wide, coeffE[i]);
                }
                if (y < h - 1 && coeffS[i] != 0) {
                    const q16 v_face = (q16)(((int64_t)wind_y[i] + wind_y[i + w]) >> 1);
                    const q16 donor = (v_face > 0) ? N[i] : N[i + w];
                    const int64_t flux_wide = mul_wide(v_face, donor);
                    dq_s[i] = flux_to_dq(flux_wide, coeffS[i]);
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

        // ---- 2c. P-E1: bank the APPLIED per-face dq (design §2.1.3) ---------
        // Placed HERE — after the limiter and after scale_mag, before the
        // divergence apply — because the energy pass must ride exactly the mass
        // the books move, not the pre-limiter intent (critique L1-10/L2-2). The
        // sum runs over the conservative planes; sign(dq) == sign(v_face) on
        // every face and every plane (donor N >= 0, coeff > 0, scale_mag
        // preserves sign), so the SUM's sign still names the same donor cell
        // each plane's own flux chose. int64: a per-plane dq is a q16, so the
        // 2-plane sum cannot overflow even at the widest N.
        if (accum_dq) {
            for (int i = 0; i < n; ++i) {
                dqsum_e[i] += (int64_t)dq_e[i];
                dqsum_s[i] += (int64_t)dq_s[i];
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

        // ---- 4. clamps: N >= 0; zero on solid/vacuum; AMBIENT ring reset ------
        // Solid never holds N (defensive — a stale value from before a tile
        // became solid must not linger). Vacuum is the DELIBERATE sink: mass
        // that flowed there this tick legitimately leaves the system (breach
        // venting), matching §2.2's "N zeroed at vacuum" rule. Both zeroings
        // are NOT part of the sealed-room conservation gate's domain (a sealed
        // room has no vacuum cells), so they never fire there.
        //
        // BC (spec §1): the ambient ring is the vacuum sink's TWIN — the sink
        // becomes a CLAMP to the infinite reservoir value N_amb[plane], applied
        // PER SUBSTEP (a per-tick reset would let ≤8 substeps drain it). The
        // rail (spec §5) records the mass exchanged with the open system:
        // Σ(N_pre_reset − N_amb), int64, per conservative plane, per substep.
        // BRANCH-gated on ambient_mode -> on a space map this middle clause is
        // dead and the loop is byte-identical to the legacy form above.
        const int32_t namb = (ambient_mode && n_amb) ? n_amb[gi] : 0;
        for (int i = 0; i < n; ++i) {
            if (solid[i] || is_vacuum[i]) {
                N[i] = 0;
            } else if (ambient_mode && is_ambient[i]) {
                if (boundary_flux) boundary_flux[gi] += (int64_t)N[i] - (int64_t)namb;
                N[i] = namb;
            } else if (N[i] < 0) {
                N[i] = 0;
            }
        }
    }
}

// ===========================================================================
// P-E1 — energy-conservative thermal transport (design §2.1). The four pinned
// stages are documented in bulk_transport.h; this is their one CPU
// transcription (cuda_bulk_transport.cu holds the kernel-for-loop twin).
// ===========================================================================
namespace {

// N_EPS (design §2.1.5): the 1-raw-count mass floor below which a cell has no
// meaningful capacity to divide by — its residual relative energy is DESTROYED
// into the signed `e_wipe_sum` and T is wiped to ambient.
constexpr int64_t N_EPS_RAW = 1;

// The participation predicate (design §2.1.2), ONE transcription: only real
// gas cells hold thermal book-energy. A `thermal_solid` tile's temperature[]
// is the OBJECT's T (ruling A1) so e there would be bogus; vacuum and the
// ambient ring are boundary channels (§4/§5), not participants.
inline bool e_participates(int i, const bool* solid, const bool* ts,
                           const bool* is_vacuum, const bool* is_ambient) {
    return !solid[i] && !ts[i] && !is_vacuum[i]
           && !(is_ambient != nullptr && is_ambient[i]);
}

}  // namespace

void bulk_flux_energy_transport_cached(
        int32_t* gas, const bool* gas_conservative, int n_gases,
        int32_t* temperature,
        const int32_t* wind_x, const int32_t* wind_y,
        const bool* solid, const bool* is_vacuum, const bool* thermal_solid_ts,
        const int32_t* coeffE, const int32_t* coeffS,
        int32_t t_min_q,
        int h, int w,
        int64_t* e_scratch, int64_t* dqsum_e, int64_t* dqsum_s,
        BulkEnergyCounters& cnt,
        const bool* is_ambient, const int32_t* n_amb, int64_t* boundary_flux) {
    const int n = h * w;
    if (n <= 0) return;
    const bool* ts = thermal_solid_ts;
    int64_t* e = e_scratch;

    // ---- stage 1: e build — e[i] = n_bulk_pre[i] * T[i], exact, unshifted ---
    // Participating cells only; everyone else carries e = 0 (never read).
    // T is FROZEN from here through stage 3: the retired SL sample was the only
    // writer of `temperature` inside a substep, and the recovery (stage 4) is
    // the last thing this function does — so every donor price below is the
    // pre-transport temperature, as design §2.1.3 requires.
    for (int i = 0; i < n; ++i) {
        if (!e_participates(i, solid, ts, is_vacuum, is_ambient)) { e[i] = 0; continue; }
        int64_t nb = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!gas_conservative[gi]) continue;
            nb += (int64_t)gas[(size_t)gi * (size_t)n + (size_t)i];
        }
        // Design §2.1.2 / L2-5: the stated map invariant. |e| <= 2^30·2^31 well
        // inside int64; the divergence apply narrows to int32 unchecked anyway,
        // so this assert is the cheapest place to notice a map that broke it.
        assert(nb < ((int64_t)1 << 30));
        e[i] = nb * (int64_t)temperature[i];
    }

    // ---- stage 2: the mass flux, banking the APPLIED per-face dq ------------
    bulk_flux_transport_cached(gas, gas_conservative, n_gases,
                               wind_x, wind_y, solid, is_vacuum,
                               coeffE, coeffS, h, w,
                               is_ambient, n_amb, boundary_flux,
                               dqsum_e, dqsum_s);

    // ---- stage 3: e apply, GATHER form (design §2.1.3/§2.1.4) --------------
    // Each cell edits ONLY its own e (the CUDA no-atomics shape, L2-11). Face
    // order is PINNED E, W, S, N — int64 sums are order-immaterial but the
    // twins must transcribe the same expression order.
    //
    // Donor side is UNIFORM: a participating donor is always debited at its OWN
    // frozen T (`dq·T_rel[donor]`), which leaves its recovered T exactly
    // invariant as mass leaves — no concentration mint. Receiver side is
    // CONDITIONAL: it is credited the donor's price only if the donor
    // PARTICIPATES; mass emerging from a thermal_solid / vacuum / ring arrives
    // carrying ZERO relative energy (rule (d) ts->air, and the §5
    // born-at-ambient class rule for the boundary channels).
    // `e_ts_residual` counts the air->ts debits ONLY — the counted destruction
    // of rule (d). air->vacuum and air->ring debits are the §4/§5 boundary
    // channels and stay uncounted here by design.
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (!e_participates(i, solid, ts, is_vacuum, is_ambient)) continue;
            const int64_t t_own = (int64_t)temperature[i];
            int64_t de = 0;
            // EAST face of i: dqsum_e[i], positive = i -> i+1.
            if (x < w - 1) {
                const int64_t q = dqsum_e[i];
                if (q > 0) {                    // i donates east
                    const int64_t phi = q * t_own;
                    de -= phi;
                    if (ts[i + 1]) cnt.e_ts_residual += phi;
                } else if (q < 0) {             // i receives from the east
                    if (e_participates(i + 1, solid, ts, is_vacuum, is_ambient))
                        de += (-q) * (int64_t)temperature[i + 1];
                }
            }
            // WEST face of i: dqsum_e[i-1], positive = (i-1) -> i.
            if (x > 0) {
                const int64_t q = dqsum_e[i - 1];
                if (q > 0) {                    // i receives from the west
                    if (e_participates(i - 1, solid, ts, is_vacuum, is_ambient))
                        de += q * (int64_t)temperature[i - 1];
                } else if (q < 0) {             // i donates west
                    const int64_t phi = (-q) * t_own;
                    de -= phi;
                    if (ts[i - 1]) cnt.e_ts_residual += phi;
                }
            }
            // SOUTH face of i: dqsum_s[i], positive = i -> i+w.
            if (y < h - 1) {
                const int64_t q = dqsum_s[i];
                if (q > 0) {                    // i donates south
                    const int64_t phi = q * t_own;
                    de -= phi;
                    if (ts[i + w]) cnt.e_ts_residual += phi;
                } else if (q < 0) {             // i receives from the south
                    if (e_participates(i + w, solid, ts, is_vacuum, is_ambient))
                        de += (-q) * (int64_t)temperature[i + w];
                }
            }
            // NORTH face of i: dqsum_s[i-w], positive = (i-w) -> i.
            if (y > 0) {
                const int64_t q = dqsum_s[i - w];
                if (q > 0) {                    // i receives from the north
                    if (e_participates(i - w, solid, ts, is_vacuum, is_ambient))
                        de += q * (int64_t)temperature[i - w];
                } else if (q < 0) {             // i donates north
                    const int64_t phi = (-q) * t_own;
                    de -= phi;
                    if (ts[i - w]) cnt.e_ts_residual += phi;
                }
            }
            e[i] += de;
        }
    }

    // ---- stage 4: recovery T = floordiv(e, n_bulk_new) (design §2.1.5) ------
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            const int i = row + x;
            if (!e_participates(i, solid, ts, is_vacuum, is_ambient)) {
                // T-WRITE guard (ruling A1): solid and thermal_solid tiles are
                // NEVER written by the EOS. Vacuum / ambient-ring cells keep
                // their per-substep wipe to dT = 0 — moved here VERBATIM from
                // the retired SL write, semantics unchanged (§2.1.1).
                if (!solid[i] && !ts[i]) temperature[i] = 0;
                continue;
            }
            int64_t n_new = 0;
            for (int gi = 0; gi < n_gases; ++gi) {
                if (!gas_conservative[gi]) continue;
                n_new += (int64_t)gas[(size_t)gi * (size_t)n + (size_t)i];
            }
            // ACTIVE-FLUX telemetry (design §2.5/§7): a participating cell with
            // ANY nonzero touching face dq. Quiescent cells rebuild EXACTLY, so
            // the §7 truncation bound is scaled by THIS measured fraction
            // rather than by the cell count (L2-10).
            const bool active =
                   (x < w - 1 && dqsum_e[i]     != 0)
                || (x > 0     && dqsum_e[i - 1] != 0)
                || (y < h - 1 && dqsum_s[i]     != 0)
                || (y > 0     && dqsum_s[i - w] != 0);
            if (active) {
                cnt.n_active_flux += 1;
                cnt.n_bulk_active_sum += n_new;
            }
            if (n_new < N_EPS_RAW) {
                // No capacity to divide by: the residual is DESTROYED (signed —
                // cold gas carries negative relative energy) and T := ambient.
                cnt.e_wipe_sum += e[i];
                temperature[i] = 0;
                continue;
            }
            // FLOOR division toward -inf — plain `/` truncates toward zero and
            // would MINT on every sub-ambient cell (fixed_point.h floordiv_q).
            int64_t t_new = floordiv_q(e[i], n_new);
            if (t_new < (int64_t)t_min_q) {
                // R3: the T_MIN rail is a counted CREATOR, in ENERGY units.
                cnt.e_floor_sum += ((int64_t)t_min_q - t_new) * n_new;
                t_new = (int64_t)t_min_q;
            }
            temperature[i] = (int32_t)t_new;
        }
    }
}
