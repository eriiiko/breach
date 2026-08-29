#include "eos_solver.h"
#include "fixed_point.h"
#include "bulk_transport.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
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
// PROMOTED into fixed_point.h as `fixedpoint::mul128_shr` (gas-energy
// conservation arc #54, design §2.5, P-G0) — this file's own copy is gone;
// `using namespace fixedpoint;` (top of file) resolves every unqualified
// `mul128_shr(...)` call below to the shared primitive, bit-identically
// (same two host branches, same shift==0 special case).

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

// ---------------------------------------------------------------------------
// P-M4b (mass-books arc): THE energy-books sum. See eos_solver.h for the
// contract and for why this is a file-scope function rather than the
// step()-local lambda it used to be.
//
// The body below is the P-E0 lambda VERBATIM — same skip-set, same nested loop
// order, same int64 accumulation, same `nb * (int64_t)temperature[i]` with NO
// offset term (no C, no s_eos_q, no + t_amb_q). That last fact is load-bearing
// for the mass-books arc: a cell joining the accountable set with T == 0
// contributes exactly nb * 0 == 0, so seeding a destroyed tile at T := 0 moves
// the books by exactly nothing (P-M3 design §4, gate 6).
// ---------------------------------------------------------------------------
int64_t eos_energy_books_sum(
        const int32_t* gas, const bool* gas_conservative, int n_gases,
        const int32_t* temperature,
        const bool* solid, const bool* is_vacuum,
        int n,
        const bool* is_ambient,
        const bool* thermal_solid,
        const int64_t* gas_energy, int32_t t_amb_raw) {
    // Same back-compat idiom as step()'s: the THERMAL axis falls back to the
    // FLOW mask when the caller has no thermal_mass plane.
    const bool* ts = (thermal_solid != nullptr) ? thermal_solid : solid;
    // Dormancy BY BRANCH, exactly as step()'s `ambient_mode` does it.
    const bool ambient_mode = (is_ambient != nullptr);
    int64_t acc = 0;
    for (int i = 0; i < n; ++i) {
        if (solid[i] || ts[i] || is_vacuum[i]
                || (ambient_mode && is_ambient[i])) continue;
        int64_t nb = 0;
        for (int gi = 0; gi < n_gases; ++gi)
            if (gas_conservative[gi])
                nb += (int64_t)gas[(size_t)gi * (size_t)n + (size_t)i];
        // arc #54 P-G1a (design §2.8): read the books OFF THE FIELD when the
        // caller has it. The per-cell RELATIVE difference (E − N·T_AMB) is
        // formed FIRST and only then summed — §2.2 forbids absolute sums.
        // Dormancy by branch: nullptr -> the pre-#54 expression, byte for byte.
        acc += (gas_energy != nullptr)
            ? (gas_energy[i] - nb * (int64_t)t_amb_raw)
            : (nb * (int64_t)temperature[i]);
    }
    return acc;
}

void EOSSolver::step(
        int32_t* atmosphere,
        int32_t* p_prev,
        int32_t* wind_x, int32_t* wind_y,
        int32_t* temperature,
        int64_t* gas_energy,                                 // arc #54 §2.2
        int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability, const float* dyn_wave_absorb,
        int h, int w, float dt,
        const bool* is_ambient, const int32_t* n_amb, int32_t p_amb,
        const int32_t* sponge_sigma, const int32_t* sponge_udamp,
        const bool* thermal_solid) const {

    const int n = h * w;
    if (n <= 0 || dt <= 0.0f) return;

    // --- THERMAL-MEDIUM mask (THERMAL-MASS AXIS, P-EOS; the governing rule and
    // the per-site rationale are in eos_solver.h's header block) --------------
    // `solid` (permeability <= 0) is a FLOW property. Steps 1b and 4c are
    // GAS-MEDIUM claims about T, so they must key on the THERMAL axis. nullptr
    // -> `solid`, which is exactly today's behaviour (and elementwise equal to
    // thermal_solid on any furniture-free map — build addendum D4). ONLY the
    // two `temperature[i]` writes and the step-1b T sample read `ts`; every
    // other `solid` / `dyn_permeability` meaning in this function (cmask,
    // mirror_idx, coeffE/S, div(u), p*, the kick, mg_build_levels) is
    // UNTOUCHED, so pressure / velocity / gas flow are unchanged.
    const bool* ts = (thermal_solid != nullptr) ? thermal_solid : solid;
    // P-E1 (design §2.1.1): the A2 T-ONLY occluder mask (`tcmask_`) and its
    // `eos_thermal_occludes` gate are RETIRED HERE — they existed only to keep
    // the semi-Lagrangian T *sample* from reading through a crate, and that
    // sample is gone (T now rides the conservative energy books, step 1d).
    // `ts` itself stays: it is the participation mask of the energy build /
    // recovery and the step-4c skip, i.e. still the THERMAL medium test.

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

    // P-E0 (energy-books design §2.5): the law-independent bracket sum
    // S = Σ n_bulk·T over the step-4c skip-set complement (!solid, !ts,
    // !vacuum, !ring); n_bulk = the gas_conservative planes as int64.
    // Range: per-cell |n_bulk·T| ≤ N_cell·T_MAX_PHYS raw ≈ 2^47 at ambient
    // counts, Σ over a map ≪ 2^62 (design §2.1.2 invariant). Pure
    // instrumentation — nothing in the sim path reads it, no digest folds it.
    eth_transport_delta = 0;    // per-tick reset (boundary_flux_ idiom)
    eth_compression_delta = 0;
    // P-E1 (design §2.1.5/§2.5): the new transport law's one-way guard terms,
    // same per-tick reset idiom.
    e_ts_residual = 0;
    e_wipe_sum = 0;
    e_floor_sum = 0;
    n_active_flux = 0;
    n_bulk_active_sum = 0;
    // P-E3 (design §2.8): the interior-drag oracle, same per-tick reset
    // idiom (P-E1's, not P-E2a's accumulate — see the as-built for why).
    ke_drag_removed = 0;
    // arc #54 P-G1a: the absolute energy counters + the new rail hit counters,
    // the SAME per-tick reset idiom (design §2.8; the closure identity is
    // stated one tick at a time, like the transport gate).
    e_entry_resync_sum = 0;
    e_transport_net_sum = 0;
    e_kick_ke_sum = 0;
    e_absorb_export_sum = 0;
    e_sponge_export_sum = 0;
    e_clamp_destroyed_sum = 0;
    e_drag_heat_sum = 0;
    e_ts_ke_sum = 0;
    e_work_export_sum = 0;
    e_ts_work_sum = 0;
    e_wall_work_probe_sum = 0;
    e_energy_floor_sum = 0;
    e_rail_sum = 0;
    e_retire_sum = 0;
    rad_clip_hits = 0;
    p_face_floor_hits = 0;
    p_face_ceil_hits = 0;
    flux_sat_hits = 0;
    // D10/D11: retired-and-zero for the whole of this tick.
    work_clamp_hits = 0;
    // P-M4b (mass-books arc): the body moved OUT to the file-scope
    // eos_energy_books_sum (declared in eos_solver.h) so the Python binding
    // measures the books through the SAME skip-set and the SAME arithmetic
    // this bracket does — one implementation, no Python transcription of the
    // four flags. `ts` is already the resolved thermal mask here, so the
    // function's own nullptr->solid fallback is a no-op on this path.
    // (arc #54 P-G1a: the lambda MOVED below the per-tick scalar folds — it
    // now reads the books off `gas_energy` and needs `t_amb_q`.)

    if ((int)n_total_.size() != n) n_total_.assign(n, 0);
    if ((int)vx_src_.size()  != n) vx_src_.assign(n, 0);
    if ((int)vy_src_.size()  != n) vy_src_.assign(n, 0);
    if ((int)t_src_.size()   != n) t_src_.assign(n, 0);
    if ((int)pstar_.size()   != n) pstar_.assign(n, 0);
    if ((int)div_u_.size()   != n) div_u_.assign(n, 0);
    if ((int)cmask_.size()   != n) cmask_.assign(n, 0);
    if ((int)coeffE_.size()  != n) coeffE_.assign(n, 0);
    if ((int)coeffS_.size()  != n) coeffS_.assign(n, 0);
    // VELOCITY-CLAMP (P-V1, D2v2): per-cell cap² scratch plane.
    if ((int)cap2_plane_.size() != n) cap2_plane_.assign(n, 0);
    // P-E1: the transient energy accumulator + applied-dq face planes.
    if ((int)e_scratch_.size() != n) e_scratch_.assign(n, 0);
    if ((int)dqsum_e_.size()   != n) dqsum_e_.assign(n, 0);
    if ((int)dqsum_s_.size()   != n) dqsum_s_.assign(n, 0);
    // arc #54 P-G1a (design §2.4/§2.5): the energy-pass scratch planes.
    if ((int)n_pre_.size()   != n) n_pre_.assign(n, 0);
    if ((int)e0_.size()      != n) e0_.assign(n, 0);
    if ((int)pcur_.size()    != n) pcur_.assign(n, 0);
    if ((int)s_plane_.size() != n) s_plane_.assign(n, 0);

    // ---- step 0: P_prev := P ---------------------------------------------
    for (int i = 0; i < n; ++i) p_prev[i] = atmosphere[i];

    // DEBUG probe (temporary): T at step-1 entry.
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_pre_advect = temperature[dbg_probe_idx];

    // ---- per-tick scalar constants (double-fold once, then quantize) ------
    const q16 n_floor_q  = quantize((double)N_FLOOR_SOLVER);
    // FLOORED AT 1 COUNT (audit Patch A / A7, 2026-08-04). t_amb_q is the
    // DIVISOR of the c_LOCAL ratio below (:308, `(t_max_abs_raw << 16) /
    // t_amb_q`) and T_AMB_K is def_readwrite-exposed to Python
    // (bindings.cpp:2134), so `solver.T_AMB_K = 0` was a reachable integer
    // divide-by-zero. Every other divide in this file is already floored; this
    // one was the exception. Same std::max idiom as dx_d below. The floor never
    // binds at the shipped 290 -> behaviour-preserving. Mirrored in
    // cuda_eos_step.cu (which also feeds the resident path's pre.t_amb_q).
    const q16 t_amb_q    = std::max<q16>(1, quantize((double)T_AMB_K));
    // s_eos_q: fold of S_EOS (phi_exp*k_temp_to_kelvin, value-frozen to 1.0
    // exactly this arc — P-K3). At s_eos_q == 65536 the product below has zero
    // low bits, so the arithmetic right shift (mul_q16 convention) is exactly
    // truncation and t_abs == T + t_amb_q for every int32 T including
    // negatives (no overflow: |product| <= 2^47). Off-identity values of
    // s_eos_q floor toward -inf at T<0 (mul_q16's documented convention) —
    // deliberate, so the asymmetry is expected when the storm session retunes
    // phi_exp.
    const q16 s_eos_q    = quantize((double)S_EOS);
    const q16 t_min_q    = quantize((double)T_MIN);
    // D-3 RELEASE-LIVE GUARD (design §4, docs/tabs_compression_work_design_
    // 2026-08-20.md): step 4c's t_abs = T + t_amb_q form is honest only while
    // S_EOS == 1 (value-frozen at P-K3) AND T_MIN > -T_AMB_K (else the
    // compression branch silently re-inverts — t_abs < 0 — the exact defect
    // this arc kills). Both are Python-writable dials, and assert() is dead
    // in the Release builds every gate and every play session uses, so this
    // is a plain, always-compiled, once-per-tick check instead. The
    // eos_kick_compression_reference test-only twin is exempt by contract
    // (it replays step() under the dials it is handed).
    if (s_eos_q != FP_ONE || t_min_q <= -(int64_t)t_amb_q) {
        throw std::runtime_error(
            "T_abs compression work requires S_EOS==1 and T_MIN > -T_AMB_K; "
            "see docs/tabs_compression_work_design_2026-08-20.md D-3");
    }
    const q16 t_max_phys_q = quantize((double)T_MAX_PHYS);   // v2.4 rail (see eos_solver.h)
    const q16 u_max_q      = quantize((double)U_MAX);        // v2.4 rail
    const q16 c_q        = quantize((double)C);
    const double gamma_d = (double)adiabatic_index;
    const q16 gamma_m1_q = quantize(gamma_d - 1.0);
    // P-G0 (gas-energy conservation arc #54, design §2.1): the two derived
    // energy-books constants, folded host-side once per tick from c_max,
    // gamma, T_AMB_K (the P-W1a fold site) -- NO CONSUMER YET (P-G1a wires
    // k_ke into the kick loop's KE brackets, design §2.3; this patch only
    // folds + guards).
    //
    // `C` is already its own def_readwrite member (defined `1/T_AMB_K` at
    // construction, eos_solver.h) and can DRIFT from T_AMB_K if a caller
    // edits only one from Python. This ALWAYS-COMPILED throw (assert() is
    // dead in every Release build every gate and play session uses -- same
    // rationale as the D-3 guard above) catches the drift once per tick,
    // beside it.
    const q16 c_from_t_amb_q = quantize(1.0 / (double)T_AMB_K);
    if (c_q != c_from_t_amb_q) {
        throw std::runtime_error(
            "EOSSolver::C must equal 1/T_AMB_K (design Sec 2.1: R_books = "
            "K*C = c_max^2/(gamma*T_AMB_K)); the two def_readwrite members "
            "have drifted apart -- see "
            "docs/gas_energy_conservation_design_2026-08-29.md Sec 2.1");
    }
    // k_ke = gamma*(gamma-1)*T_AMB_K / (2*c_max^2) -- the specific-KE-to-
    // game-deg bridge (design §2.1); ~9.0e-4 game-deg per (m/s)^2 at the
    // shipped dials, so it folds as a Q.32 `make_recip` reciprocal (a direct
    // Q16.16 fold of k_ke itself would carry only ~59 counts of precision --
    // 0.22% bias, ~6 bits).
    //
    // P-G1a CORRECTION to P-G0's inert fold: `make_recip(x) == 2^32/x`
    // (fixed_point.h:235), so the Q.32 representation of k_ke ITSELF is
    // `make_recip(1/k_ke)`, not `make_recip(k_ke)`. The name stays
    // `k_ke_recip_q32` (design §2.3 pins the consuming expression by that
    // name) but the VALUE is k_ke*2^32 == 3,874,876 at the shipped dials
    // (~2^21.9 -- exactly the "<= 2^22" the §2.3 range budget asserts).
    // Dimensional check for the pinned shift 48:
    //   du2_raw (Q32) * (k_ke*2^32) >> 48 == k_ke*|u|^2 * 2^16 == a Q16 dT.
    const double k_ke_d = gamma_d * (gamma_d - 1.0) * (double)T_AMB_K
                        / (2.0 * (double)c_max * (double)c_max);
    const int64_t k_ke_recip_q32 = make_recip(1.0 / std::max(k_ke_d, 1e-12));
    // P-M4b's books bracket (MOVED here from step()'s head at P-G1a — it now
    // reads the sum OFF `gas_energy`, so it needs `t_amb_q`). One
    // implementation of the accountable set, shared with the Python binding.
    const auto eth_books_sum = [&]() -> int64_t {
        return eos_energy_books_sum(gas, gas_conservative, n_gases,
                                    temperature, solid, is_vacuum, n,
                                    ambient_mode ? is_ambient : nullptr, ts,
                                    gas_energy, t_amb_q);
    };
    // arc #54 §2.2: the ACCOUNTABLE SET — the one canonical skip-set
    // complement, the same predicate `e_participates()` (bulk_transport.cpp)
    // and `eos_energy_books_sum` above use. ONE transcription in this file.
    const auto accountable = [&](int i) -> bool {
        return !solid[i] && !ts[i] && !is_vacuum[i]
               && !(ambient_mode && is_ambient[i]);
    };
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
    const q16 c_amb_q = quantize((double)c_max);
    // VELOCITY-CLAMP (P-V1, design v3, D2v2): the per-cell cap² fold rides
    // the SAME scan as c_LOCAL above, on the SAME tick-entry T basis — one
    // scan, one basis. Squares against squares (Erik's ruling): no per-cell
    // sqrt in the fold; only a CLAMPED cell later pays a sqrt (the kick).
    const int64_t c_amb2_q32 = (int64_t)c_amb_q * (int64_t)c_amb_q;   // Q32.32
    const int64_t u_max2_q32 = (int64_t)u_max_q * (int64_t)u_max_q;   // Q32.32
    // smallest ratio that rails at U_MAX. HOST DOUBLE FOLD — the K_raw idiom
    // above (a per-tick scalar, one transcription per side, no float in the
    // per-cell path). The naive integer form ((u_max2_q32 << 16) /
    // c_amb2_q32) OVERFLOWS int64 at the shipped dials (4.3e15 << 16 = 2^68)
    // — do not use it. Value at shipped dials: 728178 (≈ 11.11 in Q16.16).
    const double ru = (double)u_max_q / (double)c_amb_q;
    const int64_t ratio_umax = (int64_t)(ru * ru * 65536.0) + 1;

    // ---- arc #54 P-G1a: the Dalton bulk sum HOISTED above the c_LOCAL /
    // cap2 scan (PURE CODE MOTION — nothing between the old and new sites
    // read n_total_, so the values are identical). Two consumers need it
    // here now: the §2.6 N_EPS guard on the max-T scan, and the TRANSITIONAL
    // entry re-sync below. The step-2 rebuild (post-substep N) stays.
    // Dalton sum, P-T0 (design §2.6, the 0% ruling): n_total ≡ n_bulk.
    {
        for (int i = 0; i < n; ++i) n_total_[i] = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!gas_conservative[gi]) continue;
            const int32_t* plane = gas + (size_t)gi * n;
            for (int i = 0; i < n; ++i) n_total_[i] += plane[i];
        }
    }

    // ======================================================================
    // arc #54 P-G1b — THE TRANSITIONAL ENTRY RE-SYNC IS GONE. D1 IS LIVE.
    //
    // P-G1a re-derived `gas_energy := N_raw · (T_raw + T_AMB_raw)` here every
    // tick, because the §2.7 writers (combustion, the thermal solver, the
    // pumps, the seal / unseal / destroy seams, FieldEdit) still wrote
    // `temperature` alone and their writes land BETWEEN two EOS steps. P-G1b
    // moves every one of them onto the gas-energy seam, so `gas_energy` is now
    // the CROSS-TICK truth and re-deriving it from the mirror here would do
    // exactly the damage §2.6 warns about — `N·floordiv(E,N) <= E` drains up
    // to N−1 raw counts per cell per tick, and any energy a seam wrote that
    // the mirror cannot represent (a sub-count deposit) would vanish unbooked.
    //
    // `e_entry_resync_sum` survives as a RETIRED, always-zero counter (the D10
    // "retired and zero" convention): it stays in the §2.8 identity as a term
    // that is structurally 0, so the identity's Python transcription — in the
    // benches, in test_e1_hot_rail and in the ledger tools — did not have to
    // be rewritten to drop a name, and a future re-introduction of an entry
    // re-sync would have a booked home rather than being invisible.
    // ======================================================================
    e_entry_resync_sum = 0;

    // N_EPS_RAW (design §2.6): the 1-raw-count bulk floor, the SAME constant
    // bulk_transport.cpp's recovery divides against — ONE value, both files.
    const int64_t N_EPS_RAW = 1;

    int64_t t_max_abs_raw = (int64_t)t_amb_q;
    for (int i = 0; i < n; ++i) {
        // The kick's skip-set (solid||is_vacuum||ambient-ring) is a strict
        // SUPERSET of this scan's (solid||is_vacuum), so no kick-processed
        // cell ever reads the filler — u_max2_q32 is a safe defined value.
        if (solid[i] || is_vacuum[i]) { cap2_plane_[i] = u_max2_q32; continue; }
        const int64_t t_abs = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16) + (int64_t)t_amb_q;
        // arc #54 §2.6: a cell with less than one raw count of bulk N has a
        // thermodynamically meaningless temperature (that is exactly why the
        // recovery WIPES it). It must not be allowed to set c_LOCAL / n_sub
        // for the whole grid. The guard covers ONLY the max reduction — the
        // per-cell cap2 fold below keeps its own floor/ts policy verbatim.
        if ((int64_t)n_total_[i] >= N_EPS_RAW && t_abs > t_max_abs_raw)
            t_max_abs_raw = t_abs;

        // D4 + D1: ts-gas cells get the AMBIENT cap (`temperature[i]` is the
        // OBJECT's T here by ruling A1, not the gas's — reading it would
        // make furniture the one structural path to a U_MAX-scale cap);
        // floor at ambient (never below ambient, per cell — mirrors c_LOCAL's
        // own floor below). A LOCAL copy: the global t_max_abs_raw reduction
        // above must stay on the UNFLOORED, UN-ts'd t_abs.
        int64_t t_abs_cap = t_abs;
        if (ts[i] || t_abs_cap < (int64_t)t_amb_q) t_abs_cap = (int64_t)t_amb_q;
        const int64_t ratio = (t_abs_cap << 16) / (int64_t)t_amb_q;   // int64, NO narrow
        cap2_plane_[i] = (ratio >= ratio_umax)
            ? u_max2_q32                                       // rail; avoids the
            : mul128_shr(c_amb2_q32, ratio, 16);               // 2^65+ overflow path
    }
    const q16 absorb_dt_q = quantize((double)absorb_strength * dt_d);
    // P-E3 (design §2.8): the drag scalar folds — kd_q = quantize(k_drag*dt),
    // the absorb precedent exactly; heat_frac_q the plain dial fraction; the
    // c_v reciprocal (Q.32, make_recip, the SAME shared idiom Pass 1's
    // deposit already uses) folded ONCE per tick rather than per cell.
    // Dormancy branches on kd_q (QUANTIZED), never on the float k_drag — a
    // tiny k_drag (e.g. 1e-6) quantizes to 0 and a float-keyed branch would
    // disagree with CUDA about which code ran (design's explicit warning).
    const q16 kd_q = quantize((double)k_drag * dt_d);
    // drag-law v2 (design §2/§7, docs/drag_law_v2_design_2026-08-23.md):
    // kd2_q beside kd_q, the SAME per-tick-not-per-cell idiom, plus the
    // MANDATORY dormant-dial-guarded rad_dead_q32 = U0^2 (U0 = ceil(2^16/
    // kd2_q)) — the calm-cell fast-path threshold stage Q skips below. An
    // unconditional ceil-divide would be a divide-by-zero at the shipped
    // config (kd2_q == 0); kd2_q >= 1 => U0 <= 2^16 => U0^2 <= 2^32,
    // comfortably int64. Dormancy branches on the QUANTIZED kd2_q, never the
    // float (the kd_q idiom).
    const q16 kd2_q = quantize((double)k_drag2 * dt_d);
    int64_t rad_dead_q32 = 0;
    if (kd2_q > 0) {
        const int64_t U0 = ((int64_t)FP_ONE + (int64_t)kd2_q - 1) / (int64_t)kd2_q;
        rad_dead_q32 = U0 * U0;
    }
    // (arc #54 P-G1a, D5/D11: `heat_frac_q`, `recip_cv` and
    // `recip_n_work_ref` are RETIRED with their dials — the drag deposit is
    // the derived k_ke bracket now, and step 4c's trust gate is gone.)
    const int32_t ratio_q = (int32_t)((t_max_abs_raw << 16) / (int64_t)t_amb_q);
    const q16 sqrt_ratio = sqrt_q16((int64_t)ratio_q << 16);   // Q.32 radicand
    q16 c_local_q = mul_q16(c_amb_q, sqrt_ratio);
    if (c_local_q < c_amb_q) c_local_q = c_amb_q;   // never below ambient
    dbg_last_c_local_q = c_local_q;   // P6.4 telemetry (gate input reconstruction)

    // ======================================================================
    // 1. ADVECTION SUBSTEPS — n = ceil(dt/dt_adv), N_SUB_MAX-capped.
    //    u_est = max|u| + max(K·|∇P|/N̂)·dt, capped at c_LOCAL (§3.2 v2.2 —
    //    the ∇P term goes through the SAME unit bridge K as the kick).
    //    VELOCITY-CLAMP (P-V1): c_LOCAL is a GLOBAL scalar and stays one on
    //    purpose (Erik's ruling — n_sub must satisfy the worst cell); this is
    //    now its ONLY consumer — the kick's velocity ceiling moved to the
    //    per-cell cap2_plane_ above (D2v2).
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
    // (arc #54 P-G1a: the Dalton sum that stood here is HOISTED above the
    // c_LOCAL / cap2 scan — same values, same loop, one site earlier.)
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
    // D7 (VELOCITY-CLAMP, P-V1): the clip widens to max(c_LOCAL, U_MAX) —
    // stored |u| may now reach U_MAX on hot cells (the per-cell cap2_plane_
    // can rail there) even though c_LOCAL is folded from entry-T; clipping at
    // c_LOCAL alone would under-derive n_sub for that cell's substep needs.
    // Still a global scalar (Erik's ruling — n_sub must satisfy the worst cell).
    const int64_t u_est_cap = ((int64_t)c_local_q > (int64_t)u_max_q)
        ? (int64_t)c_local_q : (int64_t)u_max_q;
    if (u_est_raw > u_est_cap) u_est_raw = u_est_cap;
    const q16 cfl_dx_q = quantize((double)CFL_ADV * dx_d);
    const int64_t numer_wide = mul128_shr((int64_t)dt_q, u_est_raw, 16);
    int n_sub = std::max(1, ceil_div(
        (q16)std::min<int64_t>(numer_wide, (int64_t)INT32_MAX), cfl_dx_q));
    if (n_sub > N_SUB_MAX) n_sub = N_SUB_MAX;
    dbg_last_n_sub = n_sub;   // P6.2 telemetry (gate input reconstruction)
    const double dt_s_d = dt_d / (double)n_sub;

    // ---- arc #54 §2.1/§2.4: the FLUX constant, folded once per tick -------
    // k_work = (γ−1)/C = (γ−1)·T_AMB_K — the ONE unit bridge from the
    // engine's pressure to the books' energy currency (K CANCELS: it appears
    // in p_phys = K·p_code and again in c_v_phys = K/((γ−1)T_AMB), so the
    // work term's constant is independent of it). Derivation:
    //   d(E_books)/dt = −(K/c_v_phys)·p_code·div u = −k_work·p_code·div u.
    // k_flux folds in the sub-cycle dt_s, the 1/dx of the divergence AND the
    // ½ of the arithmetic face mean (the pass uses u_f = u_i + u_j, so no
    // separate >>1 and no per-face rounding bias — design §2.4).
    const double k_work_d = (gamma_d - 1.0) * (double)T_AMB_K;
    const q16 k_flux_q = quantize(k_work_d * dt_s_d / (2.0 * dx_d));
    // ALWAYS-COMPILED RANGE GUARD (the D-3 guard's idiom — assert() is dead
    // in every Release build every gate and play session uses). The face
    // magnitude chain is int64-safe because p_f ≤ 2^31 (int32 pressure) and
    // |u_f| ≤ 2^28 (two components, each held to the ±2^27 RAD_SAFE guard by
    // the kick), giving pu = (p_f·|u_f|)>>16 ≤ 2^43; `pu_cap` below then
    // bounds pu·k_flux_q at 2^60. That whole chain assumes k_flux_q stays a
    // sane per-tick scalar — at a pathological dx (or dt) it would not, and
    // the failure mode would be a SILENT 128→64 truncation inside the flux.
    if (k_flux_q <= 0 || k_flux_q > (1 << 24)) {
        throw std::runtime_error(
            "gas-energy flux constant k_flux_q out of range — check dt / dx / "
            "adiabatic_index / T_AMB_K; see "
            "docs/gas_energy_conservation_design_2026-08-29.md Sec 2.4");
    }
    // FLUX_MAG_CAP (§2.4, the int64 corner): saturate the per-face magnitude
    // at 2^60 so the 4-face sum (2^62) still fits. Applied at the `pu` stage
    // through a per-tick-folded cap so the saturation needs NO per-face
    // divide and the product NEVER overflows before it can be clamped —
    // identical on both sides of every face, so cancellation is exact.
    const int64_t FLUX_MAG_CAP = (int64_t)1 << 60;
    const int64_t flux_pu_cap  = FLUX_MAG_CAP / (int64_t)k_flux_q;

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
    // (THERMAL-MASS AXIS ruling A2's T-only `tcmask_` build lived here and is
    // RETIRED with the T sample — P-E1, design §2.1.1. Its whole job was to
    // stop the SL T sample reading an object's temperature as a free-energy
    // source; the energy books now solve that structurally, by never letting
    // relative energy cross a ts face at all (rule (d), §2.1.4). `cmask_` is
    // untouched, so velocity / pressure / gas flow are bit-identical.)
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

        // P-E0 bracket open: substep transport-block entry (design §2.5).
        const int64_t eth_pre_transport = eth_books_sum();

        // -- a. SL advection of u (self) ------------------------------------
        // P-E1 (design §2.1.1): the `.t` slot of the fused sample is RETIRED —
        // T-WRITE SITE 1/2 is GONE. The SL copy was the measured mint (`:423`'s
        // "FREE-ENERGY channel"): it copied a collapsing-denominator T onto real
        // mass without debiting anyone. Temperature is now transported by the
        // conservative energy books in step 1d below.
        // The fused 3-slot sampler call is kept VERBATIM (t_src_ still feeds the
        // third slot and its `.t` result is discarded): `.vx`/`.vy` do not depend
        // on the third array, so u stays bit-identical to HEAD — the whole point
        // of not re-deriving a 2-field sampler here.
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
                (void)fs.t;   // P-E1: the T slot is retired (see above).
            }
        }

        // -- d. bulk O2/N2 <- donor-cell conservative flux on u, WITH the
        //       thermal energy riding it (P-E1, design §2.1) ---------------
        // (cached-coefficient entry — the per-face min/quantize/mul chain is
        // hoisted to the per-tick cache above; the MASS arithmetic inside is
        // untouched, so gas planes stay bit-identical to HEAD.)
        // BC (spec §1 "N — sink becomes clamp, per substep"): the ambient ring
        // is clamped to N_amb[plane] every substep, and boundary_flux_ records
        // the exchange (spec §5). nullptr args on a space map -> byte-identical.
        {
            BulkEnergyCounters ec;
            bulk_flux_energy_transport_cached(
                gas, gas_conservative, n_gases,
                temperature, gas_energy, t_amb_q, t_max_phys_q,
                wind_x, wind_y, solid, is_vacuum, ts,
                coeffE_.data(), coeffS_.data(), t_min_q, h, w,
                e_scratch_.data(), n_pre_.data(),
                dqsum_e_.data(), dqsum_s_.data(), ec,
                ambient_mode ? is_ambient : nullptr,
                ambient_mode ? n_amb : nullptr,
                ambient_mode ? boundary_flux_.data() : nullptr);
            e_ts_residual     += ec.e_ts_residual;
            e_wipe_sum        += ec.e_wipe_sum;
            e_floor_sum       += ec.e_floor_sum;
            n_active_flux     += ec.n_active_flux;
            n_bulk_active_sum += ec.n_bulk_active_sum;
            // arc #54 §2.8: the transport's NET contribution to Σ_accountable
            // gas_energy — the closure identity's transport term.
            e_transport_net_sum += ec.e_transport_net;
        }
        (void)dt_s;
        // P-E0 bracket close, P-E1 bracket MOVE: the transport bracket now
        // closes after the RECOVERY (which is the last thing the call above
        // does), not merely after the mass flux — that is where the pass's
        // net contribution to the books is finally observable (design §2.5).
        eth_transport_delta += eth_books_sum() - eth_pre_transport;
        // P-E1 (design §2.1.6, declared re-baseline-class): `digest_advect`
        // MOVES ACROSS THE FLUX CALL. It hashed (wx, wy, T) BEFORE step 1d;
        // post-arc the T it must hash — T after the energy recovery — only
        // exists after it. The expression itself is unchanged, and the CUDA
        // per-call twin already took it post-loop, so the two agree.
        if (s == n_sub - 1) digest_advect = digest_of(wind_x, n, digest_of(wind_y, n, digest_of(temperature, n, 0)));
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
    // Dalton sum, P-T0 (design §2.6, the 0% ruling): n_total ≡ n_bulk — only
    // the gas_conservative pair (O2/inert_N2) contributes, at full weight.
    // trace_mass_scale is RETIRED, not wired to 0.0: trace planes are
    // skipped outright, never even read here.
    {
        for (int i = 0; i < n; ++i) n_total_[i] = 0;
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!gas_conservative[gi]) continue;
            const int32_t* plane = gas + (size_t)gi * n;
            for (int i = 0; i < n; ++i) n_total_[i] += plane[i];
        }
    }
    for (int i = 0; i < n; ++i) {
        if (solid[i] || is_vacuum[i]) { pstar_[i] = 0; continue; }
        if (debug_pstar_from_prev) {
            // MEASUREMENT-ONLY (see eos_solver.h): the paper's own p_a
            // structure — pressure evolves as its own state. Diagnostic.
            pstar_[i] = p_prev[i];
        } else {
            const int64_t t_abs_wide = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16) + (int64_t)t_amb_q;
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
    //    64,000× unit bug). Whole chain int64; |u| clamped to the PER-CELL
    //    cap2_plane_ (VELOCITY-CLAMP, P-V1, D2v2 — scale-to-cap, counter-
    //    tracked, exact rad > cap² test, no diagonal leak); narrowed to
    //    q16 ONCE at store.
    // ======================================================================
    const int32_t* Pn = L0.P.data();
    // ======================================================================
    // arc #54 §2.3 — THE PER-STAGE KINETIC-ENERGY BRACKETS.
    //
    // The kick loop applies, per cell, in order: ∇p kick → dyn_wave_absorb →
    // B3c sponge_udamp → velocity cap → staged drag L/Q. Each stage changes
    // |u|; each is ruled INDIVIDUALLY (D6), inside the loop, per cell, on the
    // cell's own N_i — no u* snapshot buffer, no second pass.
    //
    //   ∇p kick            gas_energy_i −= ΔKE   (reversible exchange, eq.2/3)
    //   dyn_wave_absorb    EXPORT (a numerical damper) — gas_energy untouched
    //   B3c sponge band    EXPORT (energy leaving to infinity)
    //   velocity cap       DESTROYED, counted (a rail)
    //   drag L + Q         gas_energy_i += −ΔKE  (structural drag heat, D5)
    //
    // ΔKE = N_i·k_ke·(|u_after|² − |u_before|²), in the PINNED operation
    // order (§2.3, R3-#3 — v3's >>32 landed ΔT in Q32, a 65,536× debit):
    //     t  = mul128_shr(k_ke_recip_q32, du2_raw, 48)   // Q32·Q32>>48 = Q16 ΔT
    //     dE = mul128_shr(N_raw,          t,       0)    // Q16·Q16    = Q32 E
    // `mul128_shr` is arithmetic on both branches (floors toward −∞ for
    // either sign) and the counter books the SAME truncated dE the field
    // gets, so no asymmetry leaks.
    //
    // BOUNDS (R3-#5). The pre-guard |u| was measured at ~2^53 raw before this
    // arc (the comment at the old cap block), and squaring that is 2^106. So
    // (a) the loaded ux/uy are clamped to ±2^27 AT LOAD — structural, not
    // inductive: FieldEdit, a level load, or the first tick after a config
    // change can hand this loop an unguarded wind; and (b) the component
    // guard MOVES to immediately after the ∇p block, UNCONDITIONALLY (outside
    // the `gx != 0 || gy != 0` micro-opt) and TIGHTENS from 2^30 to 2^27
    // (≈2000 m/s, 2× U_MAX). Every stage therefore sees |u|² ≤ 2^55 and
    // du2_raw is a Q32 (the cap2_q32 / rad convention). The clipped KE is
    // folded into the KICK bracket — the gas KEEPS the energy the clip
    // removed (the no-mint direction) — with its own counter, rad_clip_hits.
    // Moving the guard changes the absorb/sponge inputs only where it binds:
    // re-baseline-class, declared.
    // ======================================================================
    const int64_t KE_SAFE = (int64_t)1 << 27;
    // ΔKE → energy, the ONE transcription (both twins call this shape).
    const auto ke_energy = [&](int64_t n_bulk, int64_t du2_raw) -> int64_t {
        const int64_t t = mul128_shr(k_ke_recip_q32, du2_raw, 48);
        return mul128_shr(n_bulk, t, 0);
    };
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
            // (a) LOAD-SIDE CLAMP (§2.3 R3-#5b) — before any bracket opens, so
            // an unguarded stored wind is bounded rather than booked.
            if      (ux >  KE_SAFE) { ux =  KE_SAFE; ++rad_clip_hits; }
            else if (ux < -KE_SAFE) { ux = -KE_SAFE; ++rad_clip_hits; }
            if      (uy >  KE_SAFE) { uy =  KE_SAFE; ++rad_clip_hits; }
            else if (uy < -KE_SAFE) { uy = -KE_SAFE; ++rad_clip_hits; }
            // arc #54: the cell's own N — the ONE weight every bracket uses.
            // `ts` cells carry N and u but no gas_energy (D2/F5): their
            // brackets are EXPORTED to e_ts_ke_sum, never stored.
            const int64_t n_bulk_ke = (int64_t)n_total_[i];
            const bool ke_stores = accountable(i);
            int64_t u2_prev = ux * ux + uy * uy;   // ≤ 2^55
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
            // (b) THE COMPONENT GUARD, moved here and tightened to 2^27 —
            // UNCONDITIONAL (outside the micro-opt branch above, R3-#5a).
            if      (ux >  KE_SAFE) { ux =  KE_SAFE; ++rad_clip_hits; }
            else if (ux < -KE_SAFE) { ux = -KE_SAFE; ++rad_clip_hits; }
            if      (uy >  KE_SAFE) { uy =  KE_SAFE; ++rad_clip_hits; }
            else if (uy < -KE_SAFE) { uy = -KE_SAFE; ++rad_clip_hits; }
            // BRACKET 1 — the ∇p kick (reversible exchange with the field).
            {
                const int64_t u2 = ux * ux + uy * uy;
                const int64_t dE = ke_energy(n_bulk_ke, u2 - u2_prev);
                if (ke_stores) { gas_energy[i] -= dE; e_kick_ke_sum += dE; }
                else           { e_ts_ke_sum -= dE; }
                u2_prev = u2;
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
            // BRACKET 2 — dyn_wave_absorb: a NUMERICAL damper, so its removed
            // KE is EXPORTED and counted, never heated (D6). Unconditional
            // (a == 0 makes du2 exactly 0 and the counter move by 0).
            {
                const int64_t u2 = ux * ux + uy * uy;
                const int64_t dE = ke_energy(n_bulk_ke, u2 - u2_prev);
                if (ke_stores) e_absorb_export_sum -= dE;   // removed => dE ≤ 0
                else           e_ts_ke_sum         -= dE;
                u2_prev = u2;
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
            // BRACKET 3 — the B3c sponge band: models energy leaving to
            // infinity, so EXPORTED and counted (D6).
            {
                const int64_t u2 = ux * ux + uy * uy;
                const int64_t dE = ke_energy(n_bulk_ke, u2 - u2_prev);
                if (ke_stores) e_sponge_export_sum -= dE;
                else           e_ts_ke_sum         -= dE;
                u2_prev = u2;
            }

            // |u| <= per-cell cap2_plane_[i] (VELOCITY-CLAMP, P-V1, D2v2:
            // tick-entry-T fold, floors at ambient, U_MAX rail, ts->ambient —
            // ALL policy lives in the scan; D5: the kick TRUSTS the plane
            // verbatim, no re-min against U_MAX here). scale-to-cap preserves
            // direction; counter-tracked. The EXACT rad = ux²+uy² > cap² test
            // (audit defect 2: the old component Chebyshev pre-test let
            // diagonal flow up to √2×cap through — no pre-test, no leak; only
            // a CLAMPED cell pays the sqrt below).
            const int64_t cap2_q32 = cap2_plane_[i];   // D5: trusted verbatim
            const bool cap_is_umax = (cap2_q32 >= u_max2_q32);
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
            // anything above the cap and the narrow is safe by construction.
            // Behavior-neutral for every legitimate velocity (all caps are
            // orders of magnitude below the guard).
            // arc #54 §2.3 (R3-#5a): the ±2^30 component pre-clamp that stood
            // HERE has MOVED UP — to immediately after the ∇p block, tightened
            // to ±2^27 (KE_SAFE) and made unconditional — so that every KE
            // bracket, not just this clamp, sees a bounded |u|². By this point
            // |u| ≤ 2^27 already (absorb and sponge only shrink through
            // mul128_shr), so `rad` is int64-safe by construction as before.
            const int64_t rad = ux * ux + uy * uy;   // ≤ 2^55 (KE_SAFE guard)
            if (rad > cap2_q32) {
                ++u_clamp_hits;
                if (cap_is_umax) ++u_max_hits;
                const q16 umag    = sqrt_q16(rad);        // Q.32 radicand -> Q16.16
                const q16 u_cap_q = sqrt_q16(cap2_q32);   // same convention
                // D6 exact rescale (replaces the reciprocal_q16 chain, which
                // landed up to ~0.8% above cap and floored negative
                // components toward -inf): trunc-toward-0 integer divide
                // shrinks magnitude on BOTH signs identically, C++/CUDA
                // bit-identical. |ux*u_cap_q| < 2^30*2^31 = 2^61, int64-safe
                // unconditionally (sqrt_q16 self-clamps at INT32_MAX).
                ux = (ux * (int64_t)u_cap_q) / (int64_t)umag;
                uy = (uy * (int64_t)u_cap_q) / (int64_t)umag;
            }
            // BRACKET 4 — the velocity cap: a numerical RAIL whose pre-clamp
            // |u| is bounded only by the component guard, so its removed KE is
            // DESTROYED and counted, never heated (D6 — heating from a rail is
            // #54 through a new door).
            {
                const int64_t u2 = ux * ux + uy * uy;
                const int64_t dE = ke_energy(n_bulk_ke, u2 - u2_prev);
                if (ke_stores) e_clamp_destroyed_sum -= dE;
                else           e_ts_ke_sum           -= dE;
                u2_prev = u2;
            }

            // ==================================================================
            // P-E3 — INTERIOR MOMENTUM DRAG + HEAT COUNTERPARTY (energy-books
            // arc, design §2.8). PER TICK, in the step-4 kick loop, AFTER the
            // |u| cap and BEFORE the store — the kick runs once per tick, so
            // there is no per-substep factor and no n_sub dependence. Dial
            // default k_drag=0.0 -> kd_q==0 -> this whole block is skipped
            // (dormancy BY BRANCH on the QUANTIZED fold, not the float).
            // Ruling A1 pinned: ts cells skip BOTH the drag and the deposit
            // (the kick's own skip-set lacks `ts`, but the drag adds it here)
            // so the oracle stays exact with no new residual term.
            // ==================================================================
            if ((kd_q > 0 || kd2_q > 0) && !ts[i]) {
                const int64_t ux_old = ux, uy_old = uy;
                // Stage L — linear: the EXISTING lines, verbatim, now
                // inner-branched (drag-law v2, design §2). Component-wise
                // magnitude-first shrink u *= (1 - kd_q) — the absorb/sponge
                // idiom immediately above, verbatim. LOAD-BEARING beyond
                // style: magnitude-first makes |u_old|^2 - |u_new|^2 >= 0
                // STRUCTURALLY (each component's magnitude can only shrink), so
                // the deposit can never go negative from rounding and needs no
                // clamp and no signed oracle term.
                if (kd_q > 0) {
                    const q16 kk_drag = (kd_q < FP_ONE) ? (q16)(FP_ONE - kd_q) : 0;
                    const int64_t dmx = mul128_shr(ux_old < 0 ? -ux_old : ux_old, (int64_t)kk_drag, 16);
                    const int64_t dmy = mul128_shr(uy_old < 0 ? -uy_old : uy_old, (int64_t)kk_drag, 16);
                    ux = (ux_old < 0) ? -dmx : dmx;
                    uy = (uy_old < 0) ? -dmy : dmy;
                }

                // Stage Q — implicit quadratic (NEW, drag-law v2 design §2),
                // dormant by branch on kd2_q. rad1 is RECOMPUTED here (the
                // clamp block's `rad` above is stale once the rescale ran) —
                // int64-safe: |ux|,|uy| <= the post-clamp bound, far under
                // RAD_SAFE. The calm-cell fast path (design §7) skips exactly
                // when the divide below would be an exact no-op (prod == 0).
                if (kd2_q > 0) {
                    const int64_t rad1 = ux * ux + uy * uy;
                    if (rad1 >= rad_dead_q32) {
                        const q16 umag = sqrt_q16(rad1);
                        // trunc(k2*dt*|u|), the SAME 128-wide mul as the rest
                        // of this file; denom >= FP_ONE+1 whenever this branch
                        // runs (rad1 >= rad_dead_q32 <=> prod >= 1 by
                        // construction — see rad_dead_q32's fold above).
                        const int64_t prod  = mul128_shr((int64_t)kd2_q, (int64_t)umag, 16);
                        const int64_t denom = (int64_t)FP_ONE + prod;
                        // trunc-toward-0 int64 divide — the clamp-rescale
                        // idiom (:867-868), exact sign symmetry, shrink-only.
                        ux = (ux * (int64_t)FP_ONE) / denom;
                        uy = (uy * (int64_t)FP_ONE) / denom;
                    }
                }

                // Energy booking — the EXISTING block, verbatim, now booking
                // the COMBINED Δ(|u|^2) across whichever of stage L / stage Q
                // actually ran (ux_old/uy_old were captured before stage L).
                // Δ(|u|^2), raw (Q32-ish, plain int64-safe post-cap per the
                // |u| <= sqrt(cap2_q32) + ~2 counts <= U_MAX bound above (D6) —
                // the SAME "int64-safe post-cap" property the design's ΔE_cell
                // derivation relies on).
                const int64_t du2_raw = (ux_old * ux_old + uy_old * uy_old)
                                       - (ux * ux + uy * uy);   // >= 0 structurally

                // n-weighted oracle, raw Q16.16^2 — KEPT (design §2.8): it is
                // the raw KE the drag removed, independent of the constant the
                // deposit prices it at, and the rewritten P-E3 gate reads it.
                const int64_t n_bulk = (int64_t)n_total_[i];
                ke_drag_removed += mul128_shr(n_bulk, du2_raw, 16);

                // BRACKET 5 — DRAG HEAT (D5). The removed KE is STRUCTURAL
                // heat: the whole of it, at the DERIVED k_ke, straight into
                // the cell's gas_energy. No heat FRACTION dial (retired), no
                // c_v divide (that convention dial belongs to the radiation
                // deposit, not here — §2.1), and NO T_MAX_PHYS rail at the
                // deposit site: the once-per-tick recovery (§2.6) owns the
                // rails now, so a deposit can never be silently dropped.
                // `ts` cells never reach here (the `!ts[i]` guard above).
                const int64_t dE_drag = ke_energy(n_bulk, du2_raw);   // ≥ 0
                gas_energy[i] += dE_drag;
                e_drag_heat_sum += dE_drag;
            }

            wind_x[i] = (int32_t)ux;   // the ONE narrow at store — safe: |u| is
            wind_y[i] = (int32_t)uy;   // ≤ max(u_cap, RAD_SAFE) ≪ int32 range
                                       // (D6/D2v2 close the √2 diagonal-leak
                                       // slack this bound used to carry)
        }
    }
    digest_velocity = digest_of(wind_x, n, digest_of(wind_y, n, 0));

    // ======================================================================
    // 4c. COMPRESSION WORK -- DELETED (gas-energy conservation arc #54,
    //     design SS1/SS2.4, P-G1a).
    //
    // What stood here: T_i <- T_i(1 + w_i) / T_i/(1 + w_i), with
    // w_i = (gamma-1)*div_i*dt, +/-T_WORK_CLAMP-railed and trust-gated.
    // Reversible per cell -- but the BOOKS quantity is Sum N_i T_i, and its
    // change, -(gamma-1)dt*Sum N_i T_abs,i div_i, does NOT telescope
    // (Sum div_i = 0 over a sealed region, Sum N_i T_i div_i != 0 the moment
    // N or T is non-uniform). That is issue #54 exactly: measured over 18 s
    // on the sealed-box bench it destroyed 160k cell*atm*deg in the hall and
    // pumped it into cavities (+121 in the box, -20 in the arena, 21 m/s).
    //
    // Its replacement is step 6 below -- Kwatra's conservative FLUX form,
    // which applies each face with opposite signs to its two cells and so
    // telescopes to 0 over any region whose faces are all interior or wall,
    // as an integer sum, to the LSB. It must run AFTER step 5 because it
    // consumes the ABSOLUTE solved pressure p^{n+1} (on an ambient map the
    // solve runs shifted, P' = P - p_amb, and step 5 is where that is
    // restored). digest_compression / dbg_T_post_compression move with it.
    // ======================================================================

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

    // ======================================================================
    // 6. THE FACE-FLUX ENERGY STEP (gas-energy conservation arc #54, design
    //    §2.4/§2.5 — Kwatra eq. 3 with the eq. 15 face pressure). REPLACES
    //    step 4c.
    //
    // OPERANDS (pinned): the ABSOLUTE solved pressure p^{n+1} — hence AFTER
    // step 5's un-shift, never the shifted P' — and `u_new`, the STORED
    // velocity after absorb / sponge / cap / drag (not the projected
    // u^{n+1}; where those bands vary spatially they do spurious but
    // CONSERVATIVE, redistributive work — §7, the same class HEAD's 4c had).
    //
    // SHAPE (§2.5, parity-safe): per sub-cycle, TWO passes, both per-cell
    // 5-point GATHER, no atomics, no face buffer. Every cell recomputes its
    // four faces from identical inputs in CANONICAL orientation (i = the
    // LOWER linear index, j = the higher; east and south faces are owned by
    // i), truncates the magnitude ONCE and applies the sign AFTER — so the
    // two sides of a face see the same int64 and cancellation is exact.
    // Pass A prices the faces and computes the donor-only positivity scale;
    // pass B applies. The CPU twin NEEDS both passes: s_i is not knowable
    // until the cell's whole face set is priced, and a fused sweep would be
    // order-dependent (AB tol 0). On device this becomes K3, two launches
    // after K_store_atm (P-G2).
    //
    // FACE CLASSES (§2.4):
    //   both accountable                  INTERIOR — eq. 15 face pressure,
    //                                     applied to BOTH cells (telescopes);
    //   one accountable, other vacuum/ring OUTFLOW — p_f = p_acc, u_f = u_acc
    //                                     substituted into the FIXED i/j
    //                                     orientation (F14: the accountable
    //                                     side may be i OR j; the sign never
    //                                     flips), applied to the accountable
    //                                     cell only, booked to
    //                                     e_work_export_sum. This is the
    //                                     breach rarefaction; treating these
    //                                     as walls (v1) would HEAT the mouth.
    //   other solid OR thermal_solid      WALL, û_f = 0 — NO FACE. Furniture
    //                                     is a wall to the energy step (F4):
    //                                     a one-sided flux from a ts cell is
    //                                     an unbounded uncounted source driven
    //                                     by the OBJECT's temperature. The
    //                                     lost work is a D4-class accepted
    //                                     gap with a probe (e_ts_work_sum).
    // ======================================================================
    {
        // The floor the positivity rail defends: T_MIN in ABSOLUTE raw. The
        // D-3 guard (T_MIN > -T_AMB_K, checked at the top of this function)
        // makes it strictly positive, which is what lets `head` below be a
        // meaningful non-negative budget.
        const int64_t t_min_abs_raw = (int64_t)t_min_q + (int64_t)t_amb_q;

        // E^{(0)} — the ONE baseline of the increment-form pressure refresh.
        for (int i = 0; i < n; ++i) e0_[i] = gas_energy[i];

        // ---- the per-face price, the ONE transcription both passes call ---
        // (lo, hi) is the canonical pair (lo < hi). `east` selects the wind
        // component. Returns the SIGNED flux: positive == energy flows
        // lo -> hi. `cls` reports the face class to the caller.
        //   0 = no face (wall / both non-accountable / N sum 0)
        //   1 = interior      2 = outflow with lo accountable
        //   3 = outflow with hi accountable
        // `book` is TRUE only on pass B: every face is priced twice per
        // sub-cycle (once per pass) and, for an interior face, once from each
        // side — so the telemetry is booked from exactly ONE of those visits.
        const auto face_flux = [&](int lo, int hi, bool east,
                                   int& cls, bool book) -> int64_t {
            cls = 0;
            const bool a_lo = accountable(lo), a_hi = accountable(hi);
            if (!a_lo && !a_hi) return 0;
            // WALL: solid or thermal_solid on either side kills the face.
            // (`ts` already implies !accountable, so only the non-accountable
            // side can be a wall here.)
            if (solid[lo] || solid[hi] || ts[lo] || ts[hi]) {
                // D4-class PROBE (§7): the pressure work the step drops at
                // this wall. Reported by the SB gate, never applied.
                if (book) {
                    const int acc_i = a_lo ? lo : hi;
                    const int64_t pa = pcur_[acc_i];
                    const int64_t ua = east ? wind_x[acc_i] : wind_y[acc_i];
                    const int64_t m = mul128_shr(
                        mul128_shr(pa, ua < 0 ? -ua : ua, 16),
                        (int64_t)k_flux_q, 0);
                    // A structural WALL (solid) is the D4 stencil-mismatch
                    // probe; a NON-solid thermal_solid (furniture the gas can
                    // seep through) is the separate ts accepted gap. `solid`
                    // is tested FIRST because a wall tile normally carries
                    // thermal mass too, so `ts` alone would swallow the whole
                    // D4 term and report it as furniture.
                    if (solid[lo] || solid[hi]) e_wall_work_probe_sum += m;
                    else                        e_ts_work_sum += m;
                }
                return 0;
            }
            int64_t p_f, u_f;
            if (a_lo && a_hi) {
                const int64_t n_lo = (int64_t)n_total_[lo];
                const int64_t n_hi = (int64_t)n_total_[hi];
                const int64_t ns = n_lo + n_hi;
                if (ns <= 0) return 0;
                // eq. 15 LITERALLY (D3): harmonic-flavoured N x arithmetic
                // T_abs. Both p operands are already floored >= 0 in pcur_,
                // so this is a weighted average of non-negatives — no
                // floordiv sign trap, p_f <= max(p_lo, p_hi) <= INT32_MAX.
                // Numerator: p(2^31) * N(2^30) x2 = 2^62, int64-safe.
                p_f = floordiv_q((int64_t)pcur_[hi] * n_lo
                                     + (int64_t)pcur_[lo] * n_hi, ns);
                // ARITHMETIC face mean (F16, a deliberate deviation from
                // Kwatra eq. 13's density-weighted û): it is exactly the face
                // value our centred div_u_ stencil implies, so Sigma_f p_f û_f
                // vanishes where the solve zeroed the divergence. The 1/2 is
                // folded into k_flux_q — no separate >>1, no rounding bias.
                u_f = (int64_t)(east ? wind_x[lo] : wind_y[lo])
                    + (int64_t)(east ? wind_x[hi] : wind_y[hi]);
                cls = 1;
            } else {
                const int acc_i = a_lo ? lo : hi;
                p_f = pcur_[acc_i];
                // ring/vacuum u == 0, so (u_acc + 0)/2 * 2 == u_acc — the
                // value the divergence stencil reads there (mirror_idx keys
                // on solid only).
                u_f = east ? wind_x[acc_i] : wind_y[acc_i];
                // The face is TWO-WAY on purpose. It was briefly made
                // outflow-only during P-G1a (the name "OUTFLOW face" and §3's
                // one-directional description invite it), and that is WRONG,
                // measured: an open boundary then becomes a refrigerator. The
                // reservoir's inward p·u work is precisely the FLOW WORK
                // (the p/rho half of enthalpy) that the arriving gas turns
                // into kinetic energy, which the kick debits and the
                // absorb/sponge bands then destroy. Cut the import and the
                // interior loses that energy with nothing replacing it: on
                // tests/test_air_boundary.py's GATE 2 rush-in the interior
                // recovered to 43% of P_amb instead of >90%, freezing onto
                // the T_MIN rail (e_rail_sum ~ +3e15 raw per tick). §2.7's
                // born-at-ambient credit supplies the INTERNAL energy of the
                // arriving mass; this face supplies its flow work. Both.
                cls = a_lo ? 2 : 3;
            }
            if (u_f == 0 || p_f <= 0) return 0;
            const int64_t uabs = u_f < 0 ? -u_f : u_f;
            // The PINNED magnitude chain (§2.4, R3-#4): Q16*Q16>>16 = Q16,
            // then *Q16>>0 = Q32 — the field's own currency. Shifts total 16.
            int64_t pu = mul128_shr(p_f, uabs, 16);
            if (pu > flux_pu_cap) { pu = flux_pu_cap; if (book) ++flux_sat_hits; }
            const int64_t mag = pu * (int64_t)k_flux_q;   // <= 2^60 by the cap
            return (u_f > 0) ? mag : -mag;   // sign AFTER truncation
        };

        for (int k = 0; k < n_sub; ++k) {
            // ---- (a) the sub-cycle pressure refresh, INCREMENT FORM -------
            // ONE definition of p across all sub-cycles (R3-#6): sub-cycle 1
            // is EXACTLY the solved p^{n+1}; later ones carry the
            // EOS-consistent correction for the energy already moved.
            // Refreshing to C*E outright would switch the operand to p* after
            // sub-cycle 1 — the very defect v2 resolved. Without any refresh,
            // sub-cycling with frozen operands is arithmetically identical to
            // ONE pass at dt (F2) and the positivity bound is not geometric.
            //
            // The per-cell floor at 0 (F15) is applied HERE, once, into the
            // plane both passes read — so every eq.-15 operand is
            // non-negative and identical on both sides of every face.
            for (int i = 0; i < n; ++i) {
                if (!accountable(i)) { pcur_[i] = 0; continue; }
                int64_t p = (int64_t)atmosphere[i]
                          + mul128_shr((int64_t)c_q, gas_energy[i] - e0_[i], 32);
                if (p < 0) { p = 0; ++p_face_floor_hits; }
                // THE PHYSICAL CEILING on the refreshed operand (arc #54; a
                // case §2.4 does not settle — recorded in the as-built).
                //
                // R3-#6's increment form is self-limiting in the OUTFLOW
                // direction — that is the whole argument for it ("the outflow
                // shrinks as E_i shrinks and the bound is geometric"). In the
                // INFLOW direction it runs the other way: E rises, so p rises,
                // so the next sub-cycle imports MORE. With n_sub up to 8 that
                // is a within-tick positive feedback, and it is exactly what
                // GATE 2's rush-in trips (a 40x40 interior slammed to 0.1 atm
                // / 0.1 N and opened to the ring: the solve lifts P to ~0.98
                // acoustically while N is still 0.1, the kick puts |u| at
                // ~200 m/s, and a single ring face delivers many times the
                // receiving cell's whole energy).
                //
                // The ceiling is NOT a new rail: it is §2.2's own stated
                // bound, `E <= N·(T_MAX_PHYS + T_AMB)`, expressed as the
                // pressure it implies through the EOS the refresh is built
                // on. Past it the cell's temperature is meaningless anyway —
                // that is what T_MAX_PHYS means (eos_solver.h) — so letting
                // the operand run past it only feeds a runaway the §2.6
                // recovery would have to clip afterwards. Below the ceiling
                // this is dormant, so ordinary play is untouched.
                const int64_t e_ceil = (int64_t)n_total_[i]
                    * ((int64_t)t_max_phys_q + (int64_t)t_amb_q);
                const int64_t p_ceil = mul128_shr((int64_t)c_q, e_ceil, 32);
                if (p > p_ceil) { p = p_ceil; ++p_face_ceil_hits; }
                if (p > (int64_t)INT32_MAX) p = INT32_MAX;
                pcur_[i] = (int32_t)p;
            }

            // ---- (b) PASS A: OUT_i and the donor-only rail scale s_i ------
            // DONOR-ONLY (F3): a rail that also scaled incoming credit could
            // not reach a fixed point in one pass, and every incoming credit
            // is >= 0, so ignoring them is safe.
            for (int y = 0; y < h; ++y) {
                const int row = y * w;
                for (int x = 0; x < w; ++x) {
                    const int i = row + x;
                    if (!accountable(i)) { s_plane_[i] = FP_ONE; continue; }
                    int64_t out = 0;
                    int cls;
                    // EAST face of i: pair (i, i+1), i is lo.
                    if (x < w - 1) {
                        const int64_t f = face_flux(i, i + 1, true, cls, false);
                        if (cls != 0 && f > 0) out += f;      // leaves i
                    }
                    // WEST face of i: pair (i-1, i), i is hi.
                    if (x > 0) {
                        const int64_t f = face_flux(i - 1, i, true, cls, false);
                        if (cls != 0 && f < 0) out += -f;     // leaves i
                    }
                    // SOUTH face of i: pair (i, i+w), i is lo.
                    if (y < h - 1) {
                        const int64_t f = face_flux(i, i + w, false, cls, false);
                        if (cls != 0 && f > 0) out += f;
                    }
                    // NORTH face of i: pair (i-w, i), i is hi.
                    if (y > 0) {
                        const int64_t f = face_flux(i - w, i, false, cls, false);
                        if (cls != 0 && f < 0) out += -f;
                    }
                    int64_t head = gas_energy[i]
                                 - (int64_t)n_total_[i] * t_min_abs_raw;
                    if (head < 0) head = 0;
                    if (head >= out) {
                        s_plane_[i] = FP_ONE;                 // the common case
                    } else {
                        // NO 128-bit divide (device portability) and no
                        // `head << 16` (R3-#7: that overflows at head ~ 2^60).
                        // The +1 keeps s_i*OUT_i/2^16 <= head_i STRICTLY.
                        s_plane_[i] = (int32_t)floordiv_q(head, (out >> 16) + 1);
                    }
                }
            }

            // ---- (c) PASS B: apply, each face scaled by its DONOR's s -----
            // Donorship follows sign(u_f), so this pass reads s at self + the
            // 4 neighbours. The MAGNITUDE is scaled and the sign re-applied
            // (the house scale_mag idiom) rather than scaling the signed
            // value: mul128_shr floors toward -inf, so scaling a NEGATIVE
            // flux directly would round its magnitude UP and could breach
            // `Sum applied <= head`. Magnitude-first is sign-symmetric and
            // identical on both sides of the face, so cancellation stays exact.
            // `self` is the cell whose gather this is; the suppressed-transfer
            // telemetry is booked only when SELF is the donor, so an interior
            // face (visited from both sides) contributes exactly once.
            const auto apply_scale = [&](int64_t f, int donor, int self) -> int64_t {
                const int64_t s = accountable(donor) ? (int64_t)s_plane_[donor]
                                                     : (int64_t)FP_ONE;
                if (s >= (int64_t)FP_ONE) return f;
                const int64_t m = mul128_shr(f < 0 ? -f : f, s, 16);
                if (donor == self) e_energy_floor_sum += (f < 0 ? -f : f) - m;
                return (f < 0) ? -m : m;
            };
            for (int y = 0; y < h; ++y) {
                const int row = y * w;
                for (int x = 0; x < w; ++x) {
                    const int i = row + x;
                    if (!accountable(i)) continue;
                    int64_t de = 0, exp_out = 0;
                    int cls;
                    if (x < w - 1) {                 // EAST: i is lo
                        const int64_t f = face_flux(i, i + 1, true, cls, true);
                        if (cls != 0) {
                            const int64_t a = apply_scale(f, f > 0 ? i : i + 1, i);
                            de -= a;                 // lo loses when f > 0
                            if (cls != 1) exp_out += a;
                        }
                    }
                    if (x > 0) {                     // WEST: i is hi
                        const int64_t f = face_flux(i - 1, i, true, cls, true);
                        if (cls != 0) {
                            const int64_t a = apply_scale(f, f > 0 ? i - 1 : i, i);
                            de += a;                 // hi gains when f > 0
                            if (cls != 1) exp_out -= a;
                        }
                    }
                    if (y < h - 1) {                 // SOUTH: i is lo
                        const int64_t f = face_flux(i, i + w, false, cls, true);
                        if (cls != 0) {
                            const int64_t a = apply_scale(f, f > 0 ? i : i + w, i);
                            de -= a;
                            if (cls != 1) exp_out += a;
                        }
                    }
                    if (y > 0) {                     // NORTH: i is hi
                        const int64_t f = face_flux(i - w, i, false, cls, true);
                        if (cls != 0) {
                            const int64_t a = apply_scale(f, f > 0 ? i - w : i, i);
                            de += a;
                            if (cls != 1) exp_out -= a;
                        }
                    }
                    gas_energy[i] += de;
                    e_work_export_sum += exp_out;
                }
            }
        }
    }

    // ======================================================================
    // 7. RECOVERY — the mirror refresh + the ONLY rails (design §2.6).
    //
    // T_rel,i = floordiv(gas_energy_i, N_i) - T_AMB_raw on the accountable
    // set, with bulk_transport.cpp's divide policy VERBATIM (N_EPS_RAW = 1:
    // below it there is no capacity to divide by, so wipe to ambient and book
    // e_wipe_sum; never divide by 0 — the same on both backends).
    //
    // ONCE PER TICK, over the WHOLE accountable set. That cadence is what
    // bounds the stored E (§2.2: N < 2^30 and T_abs <= 2^30 give E <= 2^60
    // ONLY because T_MAX_PHYS runs here every tick), what keeps p* = C*E
    // inside int32, and what keeps t_max_phys_hits meaningful
    // (tests/test_air_boundary.py:820's `== 0` STOP).
    //
    // The rails clamp the mirror AND write gas_energy back ONLY when a rail
    // BINDS, booking the delta to e_rail_sum. NEVER otherwise: N*floordiv(E,N)
    // <= E would drain up to N-1 raw per cell per tick — exactly the drip
    // class this arc exists to kill.
    // ======================================================================
    {
        const int64_t N_EPS = N_EPS_RAW;
        for (int i = 0; i < n; ++i) {
            if (!accountable(i)) continue;
            const int64_t nb = (int64_t)n_total_[i];
            if (nb < N_EPS) {
                const int64_t e_amb = nb * (int64_t)t_amb_q;
                e_wipe_sum += gas_energy[i] - e_amb;
                gas_energy[i] = e_amb;
                temperature[i] = 0;
                continue;
            }
            int64_t t_rel = floordiv_q(gas_energy[i], nb) - (int64_t)t_amb_q;
            if (t_rel < (int64_t)t_min_q) {
                t_rel = (int64_t)t_min_q;
                ++energy_floor_hits;
                const int64_t e_new = nb * (t_rel + (int64_t)t_amb_q);
                e_rail_sum += e_new - gas_energy[i];
                gas_energy[i] = e_new;
            } else if (t_rel > (int64_t)t_max_phys_q) {
                t_rel = (int64_t)t_max_phys_q;
                ++t_max_phys_hits;
                const int64_t e_new = nb * (t_rel + (int64_t)t_amb_q);
                e_rail_sum += e_new - gas_energy[i];
                gas_energy[i] = e_new;
            }
            temperature[i] = (int32_t)t_rel;
        }
    }
    // The two step-4c checkpoints MOVE here with the law they measured: the
    // T plane's post-tick digest is now taken after the recovery, which is
    // where T finally exists for this tick (the P-E1 `digest_advect` move
    // precedent, §2.1.6 — re-baseline-class, declared).
    digest_compression = digest_of(temperature, n, 0);
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_compression = temperature[dbg_probe_idx];
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
// EOS P6.2 — standalone CPU reference for the SL-advection substep loop
// (declared in eos_solver.h; rationale there). A VERBATIM replay of step()'s
// step-1a/1f chain for a GIVEN n_sub:
//   * cmask build     — the same solid/perm<=0 -> 0, vacuum -> 1, live -> 2
//                       table step() builds once per tick;
//   * per substep     — src snapshot of (vx, vy, T), then the per-cell fused
//                       backtrace via the SAME file-local
//                       eos_backtrace_sample3_q (one routine, zero drift),
//                       solid cells zero u;
//   * dt_s_q          — quantize((double)dt / (double)n_sub), exactly the
//                       dt_d/dt_s_d fold step() performs.
// The step-1f "zero u on solid" pass is subsumed: the advection pass itself
// zeroes solid cells' u, and nothing between (bulk flux writes only gas
// planes + T) re-touches u — replicated here by construction.
//
// P-E1 (design §2.1.1): U-ONLY. The `.t` slot is retired; `temperature` is a
// READ-ONLY src slot whose sampled `.t` is discarded (as in step()), and the
// return is the chained FNV over (wy, wx). No longer == digest_advect — see
// the header block for the contract change.
// ===========================================================================
uint64_t eos_sl_advect_reference(
        int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_permeability,
        int h, int w, float dt, int n_sub,
        const bool* is_ambient, const bool* thermal_solid) {
    const int n = h * w;
    if (n <= 0 || dt <= 0.0f || n_sub < 1) return 0;
    const bool ambient_mode = (is_ambient != nullptr);   // BC: dormancy by branch
    // P-E1: `thermal_solid` is RETIRED here with the T sample (design §2.1.1) —
    // the A2 T-only occluder mask had no consumer once `.t` went. Kept in the
    // signature for ABI/back-compat, exactly as P-T0 kept `inert_n2_idx`.
    (void)thermal_solid;

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
                (void)fs.t;   // P-E1: the T slot is retired (step(), verbatim).
            }
        }
    }
    return digest_of(wind_x, n, digest_of(wind_y, n, 0));
}

// ===========================================================================
// EOS P6.4 — standalone CPU reference for the step-4 kick + step-4c
// compression work (declared in eos_solver.h; rationale there). A VERBATIM
// replay of step()'s post-solve tail on a GIVEN step-4-entry state:
//   * scalar folds    — the IDENTICAL double expressions step() performs
//                       (K_raw/Kdt_raw, inv_2dx_q, absorb_dt_q, the rail
//                       quantizes), from the same config values;
//   * Dalton sum      — step 2's n_total_ loop verbatim (P-T0, design §2.6:
//                       n_total ≡ n_bulk, the gas_conservative pair at full
//                       weight) — the kick's 1/N̂ input is REBUILT, not
//                       approximated;
//   * step 4          — gradient/kick/absorption/rail chain copied line for
//                       line from step() (same int64 staging, same clamp
//                       order, same per-CELL counter semantics);
//   * step 4c         — the compression-work loop copied line for line.
// VELOCITY-CLAMP (P-V1, design v3): the contract INVERTS here — `cap2_plane`
// comes in as a parameter, folded from the SAME tick-entry T state
// (`temperature` on entry, i.e. this replay's own t0) via formula A. That is
// now TRUE, not a limitation: the isolated tail replay CAN see its own
// step-4-entry T, so the cap is fully derivable from the replay's own inputs
// (no more `dbg_last_c_local_q` telemetry dependency the way the old scalar
// cap needed — the pre-P-V1 comment here said the opposite). Counters are
// returned per-call; digests are byte-for-byte step()'s digest_velocity /
// digest_compression expressions. Test entry only — the live path remains
// EOSSolver::step.
// ===========================================================================
void eos_kick_compression_reference(
        int32_t* wind_x, int32_t* wind_y, int32_t* temperature,
        int64_t* gas_energy,                                 // arc #54 §2.3
        const int32_t* p_new,
        const int32_t* gas, const bool* gas_conservative, int n_gases,
        const bool* solid, const bool* is_vacuum,
        const float* dyn_wave_absorb,
        int h, int w, float dt, const int64_t* cap2_plane,   // D2v2 (h,w) Q32.32, >= 0
        float c_max, float dx, float adiabatic_index, float absorb_strength,
        float n_floor_solver, float t_min,
        float t_max_phys, float u_max,   // t_work_clamp param RETIRED (arc #54)
        float k_drag, float k_drag2, float c_v,
        float t_amb_k,
        uint64_t* digest_velocity_out, uint64_t* digest_compression_out,
        int64_t* counters_out /* [9] */,
        const bool* is_ambient, const bool* thermal_solid,
        const int32_t* sponge_udamp) {
    const int n = h * w;
    const bool ambient_mode = (is_ambient != nullptr);   // BC: dormancy by branch
    // THERMAL-MASS AXIS, P-EOS: step()'s `ts` fold, verbatim. arc #54 §2.3
    // gives it a SECOND job here: a ts-gas cell is kicked / absorbed / sponged
    // / capped like any other cell (the kick's skip-set is solid || vacuum ||
    // ring, not ts), but it carries NO gas_energy — every bracket it opens is
    // EXPORTED to the ts counter instead of stored (F5).
    const bool* ts = (thermal_solid != nullptr) ? thermal_solid : solid;
    for (int c = 0; c < 9; ++c) counters_out[c] = 0;
    *digest_velocity_out = 0;
    *digest_compression_out = 0;
    if (n <= 0 || dt <= 0.0f) return;
    (void)c_v;   // arc #54 D5: the drag deposit is the derived k_ke now; c_v
                 // stays in the signature as the config echo only.

    // ---- per-tick scalar constants (step()'s folds, verbatim) -------------
    const q16 n_floor_q    = quantize((double)n_floor_solver);
    const q16 t_max_phys_q = quantize((double)t_max_phys);
    const q16 u_max_q      = quantize((double)u_max);
    (void)t_max_phys_q;   // arc #54: the T rails moved to step()'s §2.6
    // recovery; this isolated tail replay no longer owns them.
    // VELOCITY-CLAMP (P-V1, D2v2): u_max2_q32 for the kick's cap_is_umax test
    // (D3) — the SAME fold every kick site derives from u_max_q.
    const int64_t u_max2_q32 = (int64_t)u_max_q * (int64_t)u_max_q;
    const double gamma_d   = (double)adiabatic_index;
    const double dt_d      = (double)dt;
    const q16 dt_q         = quantize(dt_d);
    const double dx_d      = std::max((double)dx, 1e-6);
    const q16 inv_2dx_q    = quantize(1.0 / (2.0 * dx_d));
    const double K_d = (double)c_max * (double)c_max / gamma_d;
    const int64_t K_raw = (int64_t)(K_d * 65536.0 + 0.5);
    const int64_t Kdt_raw = mul128_shr(K_raw, (int64_t)dt_q, 16);
    const q16 absorb_dt_q = quantize((double)absorb_strength * dt_d);
    // P-E3 (design §2.8): the drag scalar folds, verbatim step()'s.
    const q16 kd_q = quantize((double)k_drag * dt_d);
    // drag-law v2 (design §2/§7): kd2_q + the dormant-guarded rad_dead_q32,
    // verbatim step()'s fold.
    const q16 kd2_q = quantize((double)k_drag2 * dt_d);
    int64_t rad_dead_q32 = 0;
    if (kd2_q > 0) {
        const int64_t U0 = ((int64_t)FP_ONE + (int64_t)kd2_q - 1) / (int64_t)kd2_q;
        rad_dead_q32 = U0 * U0;
    }
    // arc #54 §2.1: the derived KE constant, folded through the IDENTICAL
    // double expression step() uses (make_recip(1/k_ke) == k_ke·2^32 — see
    // the live path's fold for the shift-48 derivation). This test-only twin
    // is exempt from the D-3 / C-vs-T_AMB_K guards by contract (it replays
    // step() under the dials it is handed), but the CONSTANT must match.
    const double k_ke_d = gamma_d * (gamma_d - 1.0) * (double)t_amb_k
                        / (2.0 * (double)c_max * (double)c_max);
    const int64_t k_ke_recip_q32 = make_recip(1.0 / std::max(k_ke_d, 1e-12));
    const int64_t KE_SAFE = (int64_t)1 << 27;
    const auto ke_energy = [&](int64_t n_bulk, int64_t du2_raw) -> int64_t {
        const int64_t t = mul128_shr(k_ke_recip_q32, du2_raw, 48);
        return mul128_shr(n_bulk, t, 0);
    };
    int64_t ke_drag_removed = 0, e_drag_heat_sum = 0;

    // ---- step 2's Dalton sum (verbatim — the kick's N̂ input) --------------
    // P-T0 (design §2.6): n_total ≡ n_bulk; trace planes skipped outright.
    std::vector<int32_t> n_total(n, 0);
    {
        for (int gi = 0; gi < n_gases; ++gi) {
            if (!gas_conservative[gi]) continue;
            const int32_t* plane = gas + (size_t)gi * n;
            for (int i = 0; i < n; ++i) n_total[i] += plane[i];
        }
    }

    int64_t u_clamp_hits = 0, u_max_hits = 0;

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
            // arc #54 §2.3 (a): the LOAD-SIDE clamp, step()'s verbatim.
            if      (ux >  KE_SAFE) ux =  KE_SAFE;
            else if (ux < -KE_SAFE) ux = -KE_SAFE;
            if      (uy >  KE_SAFE) uy =  KE_SAFE;
            else if (uy < -KE_SAFE) uy = -KE_SAFE;
            const int64_t n_bulk_ke = (int64_t)n_total[i];
            const bool ke_stores = !ts[i];
            int64_t u2_prev = ux * ux + uy * uy;
            if (gx != 0 || gy != 0) {
                q16 nhat = n_total[i];
                if (nhat < n_floor_q) nhat = n_floor_q;
                const q16 inv_n = reciprocal_q16(nhat);
                ux -= mul128_shr(mul128_shr(Kdt_raw, gx, 16), (int64_t)inv_n, 16);
                uy -= mul128_shr(mul128_shr(Kdt_raw, gy, 16), (int64_t)inv_n, 16);
            }
            // arc #54 §2.3 (b): the component guard, moved here from the cap
            // block and tightened to ±2^27, UNCONDITIONAL — step()'s verbatim.
            if      (ux >  KE_SAFE) ux =  KE_SAFE;
            else if (ux < -KE_SAFE) ux = -KE_SAFE;
            if      (uy >  KE_SAFE) uy =  KE_SAFE;
            else if (uy < -KE_SAFE) uy = -KE_SAFE;
            // BRACKET 1 — the ∇p kick (the one bracket that touches the field).
            {
                const int64_t u2 = ux * ux + uy * uy;
                const int64_t dE = ke_energy(n_bulk_ke, u2 - u2_prev);
                if (ke_stores && gas_energy) gas_energy[i] -= dE;
                u2_prev = u2;
            }

            const q16 a = mul_q16(quantize((double)dyn_wave_absorb[i]), absorb_dt_q);
            if (a > 0) {
                const q16 kk = (a < FP_ONE) ? (q16)(FP_ONE - a) : 0;
                const int64_t mx = mul128_shr(ux < 0 ? -ux : ux, (int64_t)kk, 16);
                const int64_t my = mul128_shr(uy < 0 ? -uy : uy, (int64_t)kk, 16);
                ux = (ux < 0) ? -mx : mx;
                uy = (uy < 0) ? -my : my;
            }
            // BRACKET 2 — dyn_wave_absorb: EXPORTED, gas_energy untouched (D6).
            u2_prev = ux * ux + uy * uy;

            // B3c SPONGE VELOCITY DAMPING — restored by audit Patch A / A6
            // (2026-08-04). VERBATIM from step(), placed identically:
            // immediately after the absorb chain, before the u_cap clamp.
            // MAGNITUDE-FIRST is mandatory (fixed_point.h sign convention: a
            // naive signed truncating multiply leaves a stuck -1-count floor).
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
            // BRACKET 3 — the B3c sponge band: EXPORTED (D6).
            u2_prev = ux * ux + uy * uy;

            // VELOCITY-CLAMP (P-V1, D2v2/D5/D6): per-cell plane, trusted
            // verbatim; exact rad > cap² test (no Chebyshev pre-test, no
            // diagonal leak) — step()'s block, verbatim. arc #54: the ±2^30
            // component pre-clamp that stood here MOVED UP to the ±2^27
            // KE_SAFE guard after the ∇p block, so `rad` is bounded on entry.
            const int64_t cap2_q32 = cap2_plane[i];
            const bool cap_is_umax = (cap2_q32 >= u_max2_q32);
            const int64_t rad = ux * ux + uy * uy;
            if (rad > cap2_q32) {
                ++u_clamp_hits;
                if (cap_is_umax) ++u_max_hits;
                const q16 umag    = sqrt_q16(rad);
                const q16 u_cap_q = sqrt_q16(cap2_q32);
                ux = (ux * (int64_t)u_cap_q) / (int64_t)umag;   // D6 exact rescale
                uy = (uy * (int64_t)u_cap_q) / (int64_t)umag;
            }
            // BRACKET 4 — the velocity cap: DESTROYED and counted (D6).
            u2_prev = ux * ux + uy * uy;

            // P-E3 — interior drag + heat counterparty (design §2.8), VERBATIM
            // from step()'s kick loop: PER TICK, after the |u| cap, before the
            // store; ts cells skip both the drag and the deposit (ruling A1).
            // drag-law v2 (design §2): the two-term law, stage L / stage Q.
            if ((kd_q > 0 || kd2_q > 0) && !ts[i]) {
                const int64_t ux_old = ux, uy_old = uy;
                if (kd_q > 0) {
                    const q16 kk_drag = (kd_q < FP_ONE) ? (q16)(FP_ONE - kd_q) : 0;
                    const int64_t dmx = mul128_shr(ux_old < 0 ? -ux_old : ux_old, (int64_t)kk_drag, 16);
                    const int64_t dmy = mul128_shr(uy_old < 0 ? -uy_old : uy_old, (int64_t)kk_drag, 16);
                    ux = (ux_old < 0) ? -dmx : dmx;
                    uy = (uy_old < 0) ? -dmy : dmy;
                }

                if (kd2_q > 0) {
                    const int64_t rad1 = ux * ux + uy * uy;
                    if (rad1 >= rad_dead_q32) {
                        const q16 umag = sqrt_q16(rad1);
                        const int64_t prod  = mul128_shr((int64_t)kd2_q, (int64_t)umag, 16);
                        const int64_t denom = (int64_t)FP_ONE + prod;
                        ux = (ux * (int64_t)FP_ONE) / denom;
                        uy = (uy * (int64_t)FP_ONE) / denom;
                    }
                }

                const int64_t du2_raw = (ux_old * ux_old + uy_old * uy_old)
                                       - (ux * ux + uy * uy);   // >= 0 structurally
                const int64_t n_bulk = (int64_t)n_total[i];
                ke_drag_removed += mul128_shr(n_bulk, du2_raw, 16);
                // BRACKET 5 — DRAG HEAT at the derived k_ke (D5), step()'s.
                const int64_t dE_drag = ke_energy(n_bulk, du2_raw);
                if (gas_energy) gas_energy[i] += dE_drag;
                e_drag_heat_sum += dE_drag;
            }
            (void)u2_prev;

            wind_x[i] = (int32_t)ux;
            wind_y[i] = (int32_t)uy;
        }
    }
    *digest_velocity_out = digest_of(wind_x, n, digest_of(wind_y, n, 0));

    // ---- step 4c: DELETED (arc #54 D11 — see step()'s own note). This
    // reference no longer writes `temperature` at all: the recovery that now
    // owns the T rails is a WHOLE-GRID once-per-tick pass in step(), after
    // the face-flux energy step, and is out of this isolated tail replay's
    // scope by construction. The digest is still taken (unchanged plane) so
    // the return shape and every positional unpack survive.
    *digest_compression_out = digest_of(temperature, n, 0);

    counters_out[0] = u_clamp_hits;
    counters_out[1] = u_max_hits;
    counters_out[2] = 0;                  // work_clamp_hits — retired (D10)
    counters_out[3] = 0;                  // energy_floor_hits — moved to §2.6
    counters_out[4] = 0;                  // t_max_phys_hits — moved to §2.6
    counters_out[5] = ke_drag_removed;
    counters_out[6] = e_drag_heat_sum;    // ex e_drag_deposit (D10 slot 6)
    counters_out[7] = 0;                  // e_drag_drop_sum — retired
    counters_out[8] = 0;                  // e_drag_rail_clipped — retired
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
