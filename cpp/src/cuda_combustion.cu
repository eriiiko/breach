// ============================================================================
// EOS P6.9b combustion solver implementation — see cuda_combustion.h.
// A bit-identical GPU port of CombustionSolver::step (combustion.cpp), the
// P6.9a two-gather reformulation (docs/eos_p6_9_combustion_design.md §3–§4),
// widened to three gathers by P-O2b and put back ON the arc #54 gas energy
// seam by #71 (residency arc #29 P2).
// This is a FAITHFUL bit-identity port, NOT a redesign: the CPU pass was
// reformulated into order-free gathers precisely so it runs identically on
// CPU and GPU (design §3: "Runs identically on CPU and GPU").
//
// Every per-cell op is a VERBATIM device transcription of the CPU loops — same
// integer ops, same branch structure, same fixed remainder tiebreak. The
// proportional split uses plain int64 `/` and `%` (design §3 step 2): integer
// divide has a single portable answer (floor of the quotient), bit-identical on
// CPU and CUDA — it is FLOAT division that is forbidden, never integer. The
// heat-deposit reciprocals use reciprocal_q16_dev / recip_mul_dev (the CUDA-S4
// device kit, bit-identical to fixedpoint::reciprocal_q16 / recip_mul); the
// aggregate deposit's saturating add is heat_saturating_add_dev (verbatim of
// fixed_point.h::heat_saturating_add). mul_wide / narrow_round / mul_q16 are the
// FP_HD host/device helpers used directly; so are the gas energy seam
// (gas_energy.h: accountable, mirror_q, deposit_railed) and the solid capacity
// kit (temperature_solver.h: conduction::cell_capacity_q) — ONE copy of each
// rule, called from both backends.
//
// #71 — WHAT WAS WRONG (fixed 2026-11-07): this twin predated arc #54. It took
// no `gas_energy`, so (a) Pass A never debited the donor's energy nor refreshed
// its mirror, (b) Pass B carried no energy to the deposit sites, and (c) Pass C
// added the fire's dT straight into `temperature` (the pre-#54 T-form) where
// the CPU lands N·dT in the conserved field through the railed seam and
// re-derives T from it. Nothing in the e_comb_* books moved on the GPU, and the
// runner dropped even the rail counts the twin returned. On a burning scene
// the two backends parted on tick 0 (6 cells around the fire) and everywhere
// by tick 1. The twin now carries the parallel two-hop energy ledger
// (e_slot / e_dep_site) and books every channel into a per-env block.
//
// S. Feldman, J.F. O'Brien, O. Arikan, "Animating Suspended Particle
// Explosions", SIGGRAPH 2003 — the heat + product-yield + ignition-threshold
// source-term structure this pass follows (constants game-tuned, not lit-derived).
// ============================================================================
#include "cuda_combustion.h"
#include "cuda_resident.h"   // combustion_launch_resident + CombustionScalars (#29 P2)
#include "combustion.h"    // CombustionSolver::FUEL_FLOOR + the combustion_draw tables
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow_round, make_recip
#include "cuda_fixedpoint_device.cuh"  // reciprocal_q16_dev, recip_mul_dev, heat_saturating_add_dev
#include "o2_pressure_factor.h"        // issue #7: the SAME FP_HD g(p) the CPU pass calls
#include "gas_energy.h"                // arc #54: THE gas energy seam (FP_HD)
#include "temperature_solver.h"        // arc #54 P-G5: conduction::cell_capacity_q (FP_HD)

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_batch.cuh"   // #29 P5: the tick stream

using namespace fixedpoint;

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in combustion_step/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// 4-connected open-neighbour faces (N, S, W, E) — VERBATIM of combustion.cpp's
// D4 = {{-1,0},{1,0},{0,-1},{0,1}}. D4_OPP[d] is the opposite face within D4
// (N<->S, W<->E): Pass B, walking OUT from source i in direction d to air
// neighbour j, reads the allocation j filed toward i under j's OWN outbound
// direction to i, namely D4_OPP[d].
__device__ __constant__ int D4_dy[4]  = {-1, 1, 0, 0};
__device__ __constant__ int D4_dx[4]  = { 0, 0, -1, 1};
__device__ __constant__ int D4_OPP[4] = { 1, 0, 3, 2};

// ---- P-O2b: the canonical draw tables in __constant__ memory ---------------
// These are NOT re-authored here. They are UPLOADED from combustion.h's
// `combustion_draw` namespace — the single definition the CPU solver walks —
// via cudaMemcpyToSymbol (combustion_upload_tables). A transcription drift
// between the backends is therefore structurally impossible, which matters
// more than the few hundred bytes of copy: the tables ARE the law's geometry,
// and a single swapped offset would be a silent, tol-0-invisible fork on one
// machine. (W_hop, a quantized scalar table, rides CombustionScalars instead.)
namespace cd = combustion_draw;
__device__ __constant__ int8_t c_off_dy[cd::SLOTS_MAX];
__device__ __constant__ int8_t c_off_dx[cd::SLOTS_MAX];
__device__ __constant__ int8_t c_ball_dy[cd::BALL_MAX];
__device__ __constant__ int8_t c_ball_dx[cd::BALL_MAX];
__device__ __constant__ int8_t c_ball_nbr[cd::BALL_MAX * 4];
__device__ __constant__ int8_t c_ball_slot[cd::BALL_MAX * 4];

__device__ __forceinline__ bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

// P-O2b — VERBATIM device port of combustion.cpp's apply_draw_weight: the
// EXACT floor((P*wq)/2^16) for P >= 0, split at 16 bits so no 128-bit
// arithmetic and no overflow, and the exact IDENTITY at wq == FP_ONE (which is
// what makes draw_r == 1 byte-identical on this backend too).
__device__ __forceinline__ long long apply_draw_weight_dev(long long P, q16 wq) {
    const long long hi = P >> 16;
    const long long lo = P & 0xFFFF;
    return hi * (long long)wq + ((lo * (long long)wq) >> 16);
}

// Integer clamp to [0, FP_ONE] (the [0,1] o2f saturation) — VERBATIM device
// port of combustion.cpp's file-local clamp01_q (mirrors cuda_fire.cu's
// clamp01_q_dev so the two O2 laws are bit-identical).
__device__ __forceinline__ q16 clamp01_q_dev(q16 v) {
    if (v < 0) return 0;
    if (v > FP_ONE) return FP_ONE;
    return v;
}

// Integer clamp to [0, cap] (R3 hotf) — VERBATIM device port of
// combustion.cpp's file-local clamp0cap_q (mirrors cuda_fire.cu's
// clamp0cap_q_dev).
__device__ __forceinline__ q16 clamp0cap_q_dev(q16 v, q16 cap) {
    if (v < 0) return 0;
    if (v > cap) return cap;
    return v;
}

// P-R4: SATURATING integer atomic add — the device twin of fixed_point.h's
// `heat_saturating_add` (and a verbatim copy of cuda_raycaster.cu's
// heat_atomic_sat_add — the only copy since P4 deleted that file). Needed here
// because several AIR cells can feed the same flammable claimant, so the
// H_bed deposits at one source cell race. Order-free
// for non-negative deltas: a saturating add under a monotone clamp is
// associative + commutative, so the total is bit-identical to the CPU's ordered
// sequence of heat_saturating_add calls.
__device__ __forceinline__ void heat_atomic_sat_add_dev(int32_t* addr, int32_t delta) {
    if (delta <= 0) return;
    int32_t old = *addr, assumed;
    do {
        assumed = old;
        int32_t sum = (assumed > 0x7fffffff - delta) ? 0x7fffffff : (assumed + delta);
        old = atomicCAS(addr, assumed, sum);
    } while (assumed != old);
}

// One book slot += v. Integer atomicAdd is order-free modulo 2^64, so a signed
// book rides two's complement and reads back exactly as the CPU's int64 sum
// (the cuda_temperature.cu `cadd` idiom).
__device__ __forceinline__ void cadd(unsigned long long* c, int slot, int64_t v) {
    if (v != 0) atomicAdd(&c[slot], (unsigned long long)v);
}

// ---- K0: the pass-entry snapshot + the scratch / book zeroing --------------
// Tsnap <- temperature (the explicit freeze: a source cannot heat AND ignite a
// neighbour the same tick, delta alpha), and every per-tick scratch plane and
// the book block back to zero — the CPU's fresh std::vector(…, 0)s.
__global__ void combustion_prep(
        const int32_t* __restrict__ temperature, int32_t* __restrict__ tsnap,
        int32_t* __restrict__ alloc, int32_t* __restrict__ dep,
        int64_t* __restrict__ e_slot, int64_t* __restrict__ e_dep,
        unsigned long long* __restrict__ cnt,
        int n_env, int n, int n_slots) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long long k = tid; k < (long long)n_env * COMB_CNT_SLOTS; k += stride)
        cnt[k] = 0ULL;
    const long long total = (long long)n_env * n;
    for (long long idx = tid; idx < total; idx += stride) {
        const long long e = idx / n;
        const long long j = idx - e * n;
        tsnap[idx] = temperature[idx];
        for (int s = 0; s < n_slots; ++s) alloc[(e * n_slots + s) * n + j] = 0;
        for (int t = 0; t < 5; ++t) dep[(e * 5 + t) * n + j] = 0;
        if (e_slot != nullptr)
            for (int s = 0; s < n_slots; ++s) e_slot[(e * n_slots + s) * n + j] = 0;
        if (e_dep != nullptr)
            for (int t = 0; t < 5; ++t) e_dep[(e * 5 + t) * n + j] = 0;
    }
}

// ---- K1: Pass A — air cells (combustion.cpp Pass A) ------------------------
// One thread per (env, cell j). Single writer of O2[j], of temperature[j] and
// gas_energy[j] (the donor's energy leaving with its O2), and of ALL of the
// claimant-slot buffers at index j. Reads tsnap (the snapshot gate), the
// pass-entry O2 at the OWN cell (read-before-write => every claimant sees
// pass-entry O2 — deltas beta/gamma), and pre-payment wall_hp (K2 writes it,
// barriered after this kernel — so this read is pre-payment, matching the CPU).
//
// P-O2b: TEMPLATED ON THE DRAW RADIUS. R is a compile-time constant so the
// per-thread arrays below are sized exactly (NSLOT = 4 / 12 / 24) and every
// loop bound is known to ptxas — the difference between an unrolled,
// register-resident gather and a fully indirect local-memory one. The register
// budget was round-3 F6's named concern; see the cost note at the launch site.
template <int R>
__global__ void combustion_pass_a(
        int n_env, int h, int w, int n_gases, int o2_idx, int inert_n2_idx,
        int acc_depth,
        int32_t* __restrict__ gas,
        const int32_t* __restrict__ tsnap, int32_t* __restrict__ temperature,
        const int32_t* __restrict__ wall_hp, const int32_t* __restrict__ fire,
        const bool* __restrict__ flammable, const bool* __restrict__ solid,
        const bool* __restrict__ is_vacuum,
        const bool* __restrict__ thermal_solid,   // nullable
        const bool* __restrict__ is_ambient,      // nullable
        const int32_t* __restrict__ ignition_temp_q16,
        int32_t* __restrict__ alloc_slot,
        const int32_t* __restrict__ perm_q,       // nullable
        int32_t* __restrict__ heat,               // nullable (P-R4)
        int32_t* __restrict__ dem_acc,            // nullable (D1)
        const int32_t* __restrict__ fire_T_ext_plane,   // nullable (R3)
        const int32_t* __restrict__ atmosphere,         // nullable (#7)
        int64_t* __restrict__ gas_energy,         // nullable (#54)
        int64_t* __restrict__ e_slot,             // with gas_energy
        const int32_t* __restrict__ burn_cap_env, const int32_t* __restrict__ t_amb_env,
        CombustionScalars sc,
        unsigned long long* __restrict__ cnt) {
    constexpr int NSLOT = 2 * R * (R + 1);         // 4, 12, 24
    constexpr int NBALL = 2 * (R - 1) * R + 1;     // 1, 5, 13
    const int n = h * w;
    const int32_t FUEL_FLOOR = CombustionSolver::FUEL_FLOOR;
    const bool e_on = (gas_energy != nullptr);
    const o2_pressure::Factor pf{sc.p_ext_q, sc.p_full_q, sc.p_inv_span_q};
    const long long total = (long long)n_env * n;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int e = (int)(idx / n);
        const int j = (int)(idx - (long long)e * n);
        const q16 burn_cap_q = burn_cap_env[e];
        if (burn_cap_q <= 0) continue;   // the CPU's early return, per env
        // ---- this env's planes (§A: env = idx / n, cell = idx % n) ----------
        const size_t c0 = (size_t)e * n;
        int32_t* O2 = gas + ((size_t)e * n_gases + o2_idx) * n;
        const int32_t* N2 = gas + ((size_t)e * n_gases + inert_n2_idx) * n;
        const int32_t* Tsnap = tsnap + c0;
        int32_t* temp_e = temperature + c0;
        const int32_t* whp_e = wall_hp + c0;
        const int32_t* fire_e = fire + c0;
        const bool* flam_e = flammable + c0;
        const bool* sol_e = solid + c0;
        const bool* vac_e = is_vacuum + c0;
        const bool* tsol_e = thermal_solid ? thermal_solid + c0 : nullptr;
        const bool* amb_e = is_ambient ? is_ambient + c0 : nullptr;
        const int32_t* ign_e = ignition_temp_q16 + c0;
        int32_t* alloc_e = alloc_slot + (size_t)e * NSLOT * n;
        const int32_t* perm_e = perm_q ? perm_q + c0 : nullptr;
        int32_t* heat_e = heat ? heat + c0 : nullptr;
        int32_t* dacc_e = dem_acc ? dem_acc + (size_t)e * acc_depth * n : nullptr;
        const int32_t* tep_e = fire_T_ext_plane ? fire_T_ext_plane + c0 : nullptr;
        const int32_t* atm_e = atmosphere ? atmosphere + c0 : nullptr;
        int64_t* gE_e = e_on ? gas_energy + c0 : nullptr;
        int64_t* eslot_e = e_on ? e_slot + (size_t)e * NSLOT * n : nullptr;
        unsigned long long* cnt_e = cnt + (size_t)e * COMB_CNT_SLOTS;

        if (sol_e[j] || vac_e[j]) continue;

        const q16 o2j = O2[j];
        if (o2j <= sc.o2_thresh_q) continue;   // epsilon skip-floor (RETIRED gate)

        const int y = j / w, x = j % w;

        // o2f_j — the continuous-O2 factor at THIS air cell, LINEAR in its O2
        // MOLE FRACTION X_j = O2[j]/(O2[j]+N2[j]) (pass-entry). Same hoisted
        // x_ext_q/recip_x_span/x_degenerate/X_N_FLOOR as fire_logistic.
        const int64_t n_tot_j = (int64_t)o2j + (int64_t)N2[j];
        const q16 den_j = (n_tot_j < (int64_t)sc.x_n_floor) ? sc.x_n_floor : (q16)n_tot_j;
        const q16 Xj = mul_q16(o2j, reciprocal_q16_dev(den_j));
        const q16 o2f_x_j = sc.x_degenerate
            ? ((Xj < sc.x_ext_q) ? (q16)0 : (q16)FP_ONE)
            : clamp01_q_dev(recip_mul_dev(Xj - sc.x_ext_q, sc.recip_x_span));
        // THE PRESSURE FACTOR (issue #7) — VERBATIM the CPU pass: g of the
        // materialized pressure at THIS air cell, through the SAME FP_HD
        // o2_pressure::factor, folded into o2f_j by ONE mul_q16.
        const q16 g_j = (atm_e != nullptr)
            ? o2_pressure::factor(pf, atm_e[j]) : (q16)FP_ONE;
        const q16 o2f_j = mul_q16(o2f_x_j, g_j);

        // ---- P-O2b STEP 1: THE REVERSE RELAXATION — the CPU block verbatim.
        // Expand outward from j through OPEN CELLS ONLY over the baked BALL
        // offsets, in R-1 LEVELLED rounds; bd[b] is the BFS hop distance from j
        // and bw[b] the maximum permeability-multiplicative weight over j's
        // min-hop paths. Order-free at this site: a round reads only cells
        // stamped at the previous level and the combine is a MAX. At R == 1 the
        // whole block is NBALL == 1, zero rounds — free, and exactly the
        // shipped law.
        int    bcell[NBALL];
        int8_t bd[NBALL];
        q16    bw[NBALL];
        #pragma unroll
        for (int b = 0; b < NBALL; ++b) {
            bd[b] = -1; bw[b] = 0; bcell[b] = -1;
            const int cy = y + c_ball_dy[b], cx = x + c_ball_dx[b];
            if (!in_bounds(cy, cx, h, w)) continue;
            const int c = cy * w + cx;
            if (sol_e[c] || vac_e[c]) continue;   // open cells only; vacuum terminates
            bcell[b] = c;
        }
        bd[0] = 0; bw[0] = FP_ONE;    // the donor cell itself: hop 0, empty path
        #pragma unroll
        for (int r = 1; r < R; ++r) {
            #pragma unroll
            for (int b = 0; b < NBALL; ++b) {
                if (bd[b] >= 0 || bcell[b] < 0) continue;
                q16 best = 0;
                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int nb = c_ball_nbr[b * 4 + d];
                    if (nb < 0 || nb >= NBALL) continue;
                    if (bd[nb] != (int8_t)(r - 1)) continue;
                    q16 pfw = FP_ONE;
                    if (perm_e != nullptr) {
                        const q16 pa = perm_e[bcell[nb]], pb = perm_e[bcell[b]];
                        pfw = (pa < pb) ? pa : pb;
                    }
                    const q16 cand = mul_q16(bw[nb], pfw);
                    if (cand > best) best = cand;
                }
                if (best > 0) { bd[b] = (int8_t)r; bw[b] = best; }
            }
        }

        // ---- P-O2b STEP 2: reduce the walk onto the SOURCE SLOTS ------------
        // Lexicographic (minimum hop, then maximum path weight) per slot — an
        // order-free reduction, the CPU site verbatim.
        int8_t sd[NSLOT];
        q16    sw[NSLOT];
        #pragma unroll
        for (int s = 0; s < NSLOT; ++s) { sd[s] = -1; sw[s] = 0; }
        #pragma unroll
        for (int b = 0; b < NBALL; ++b) {
            if (bd[b] < 0) continue;
            const int hop = (int)bd[b] + 1;
            const q16 wp  = bw[b];
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                const int s = c_ball_slot[b * 4 + d];
                if (s < 0 || s >= NSLOT) continue;
                if (sd[s] < 0 || hop < (int)sd[s] ||
                    (hop == (int)sd[s] && wp > sw[s])) {
                    sd[s] = (int8_t)hop; sw[s] = wp;
                }
            }
        }

        // Gather the flammable claimant sources + each one's per-claimant
        // DEMAND (design §2.3 as generalized by v5.2, R3 hot-burns-faster):
        //   demand_k = burn_cap * I_k * o2f_j * hotf_k * W_hop[d_k] * w_path_k
        // (PINNED left-fold mul_q16, truncating). I_k = fire[i].
        int cl_slot[NSLOT];   // slot key of the claimant (alloc_slot key)
        int cl_src[NSLOT];    // global cell index of the claimant source
        int64_t dem[NSLOT];   // per-claimant O2 demand this tick (Q16.16 counts)
        int n_cl = 0;
        #pragma unroll
        for (int sl = 0; sl < NSLOT; ++sl) {
            const int iy = y + c_off_dy[sl], ix = x + c_off_dx[sl];
            // Out of bounds: never written at all — the CPU idiom verbatim.
            if (!in_bounds(iy, ix, h, w)) continue;
            const int i = iy * w + ix;
            // D1 RESET RULE — the CPU site verbatim: the slot's sub-count debt
            // survives only while the source is an ACTIVELY BURNING, REACHABLE
            // claimant; every rejecting gate zeroes it first.
            const size_t slot = (size_t)sl * n + j;
            // P-O2b's one new rejection: no longer reachable by the draw.
            if (sd[sl] < 0) { if (dacc_e) dacc_e[slot] = 0; continue; }
            if (!flam_e[i]) { if (dacc_e) dacc_e[slot] = 0; continue; }
            if (whp_e[i] <= FUEL_FLOOR) {            // no fuel (P5.1 ember out)
                if (dacc_e) dacc_e[slot] = 0; continue; }
            const q16 ign_i = ign_e[i];
            if (ign_i <= 0) { if (dacc_e) dacc_e[slot] = 0; continue; }
            // IGNITION vs SUSTAIN — the CPU site verbatim (combustion.cpp
            // carries the full finding): the ignition-temperature gate applies
            // only to a tile that is NOT already alight.
            const bool alight = (fire_e[i] > 0);
            if (!alight && Tsnap[i] < ign_i) {   // not alight -> the ign gate
                if (dacc_e) dacc_e[slot] = 0;
                continue;
            }
            if (!alight) {                        // I == 0 -> demands nothing
                if (dacc_e) dacc_e[slot] = 0;
                continue;
            }
            // R3 hot-burns-faster — VERBATIM the CPU per-claimant computation:
            // hotf_i is THIS claimant's own ramp, T source = Tsnap[i], folded
            // into o2f_j via ONE narrowing mul_q16 BEFORE the wide product.
            const q16 T_ext_i = tep_e ? tep_e[i] : sc.fire_T_ext_q;
            const q16 hotf_i = clamp0cap_q_dev(
                recip_mul_dev(Tsnap[i] - T_ext_i, sc.recip_T_span), sc.hotf_cap_q);
            const q16 o2f_hotf = mul_q16(o2f_j, hotf_i);
            // P-O2b: the draw weight wq = W_hop[d] * w_path for this pair.
            // FP_ONE at R == 1 (d == 1, empty path) -> every use is identity.
            const q16 wq = mul_q16(sc.w_hop_q[(int)sd[sl]], sw[sl]);
            q16 di;
            if (dacc_e != nullptr) {
                // D1 error-feedback demand — the CPU block verbatim (int64
                // wide product, scale-2^31 remainder, whole counts drawn), with
                // the draw weight folded into the WIDE product first.
                const long long P0 = (long long)burn_cap_q
                                   * (long long)fire_e[i] * (long long)o2f_hotf;
                const long long P = apply_draw_weight_dev(P0, wq);
                const long long wide = (long long)dacc_e[slot] + (P >> 1);
                const long long draw = wide >> 31;
                dacc_e[slot] = (int32_t)(wide - (draw << 31));
                di = (q16)draw;
            } else {
                di = mul_q16(mul_q16(mul_q16(burn_cap_q, fire_e[i]), o2f_hotf), wq);
            }
            cl_slot[n_cl] = sl;
            cl_src[n_cl] = i;
            dem[n_cl] = (int64_t)di;
            ++n_cl;
        }
        if (n_cl == 0) continue;

        // --- Allocate O2[j] across the claimants (design §3 step 2) ---------
        int64_t alloc[NSLOT];
        int64_t D = 0;
        for (int k = 0; k < n_cl; ++k) D += dem[k];
        if (D == 0) continue;   // all claimants choked/flameless -> draw nothing
        int64_t burn_j;
        if (D <= (int64_t)o2j) {
            // No contention: every claimant gets its full demand.
            for (int k = 0; k < n_cl; ++k) alloc[k] = dem[k];
            burn_j = D;
        } else {
            // Contention: EXACT INTEGER proportional split (plain int64 `/`,`%`
            // — bit-identical CPU<->GPU; keeps sum(alloc) == O2[j] exactly).
            int64_t keys[NSLOT];
            int64_t sum_alloc = 0;
            for (int k = 0; k < n_cl; ++k) {
                const int64_t num = (int64_t)o2j * dem[k];  // < 2^43
                alloc[k] = num / D;      // floor, exact integer divide
                keys[k]  = num % D;      // integer remainder = tiebreak key
                sum_alloc += alloc[k];
            }
            // The leftover LSBs (in [0, n_cl)) go to that many claimants with
            // the largest key; ties -> lowest source index. Order-free.
            int64_t rem_lsb = (int64_t)o2j - sum_alloc;
            bool chosen[NSLOT];
            for (int k = 0; k < n_cl; ++k) chosen[k] = false;
            for (int r = 0; r < (int)rem_lsb; ++r) {
                int best = -1;
                for (int k = 0; k < n_cl; ++k) {
                    if (chosen[k]) continue;
                    if (best < 0 ||
                        keys[k] > keys[best] ||
                        (keys[k] == keys[best] && cl_src[k] < cl_src[best])) {
                        best = k;
                    }
                }
                chosen[best] = true;
                alloc[best] += 1;
            }
            burn_j = (int64_t)o2j;   // contested cells fully drain (delta gamma)
        }

        // --- Single-writer O2 DEBIT at cell j (design §3 step 3) ------------
        O2[j] = (q16)((int64_t)o2j - burn_j);

        // ---- arc #54 P-G1b: THE DONOR'S ENERGY LEAVES WITH ITS O2 — the CPU
        // site verbatim (#71). MOVED mass carries its source's T_abs; a NON-
        // accountable donor's parcel is MINTED at ambient, counted. One common
        // factor, so the per-slot energies sum to the debit exactly.
        const int32_t t_amb_q = t_amb_env[e];
        const bool acct_j = e_on
            && gas_energy::accountable(sol_e, tsol_e, vac_e, amb_e, j);
        const int64_t t_abs_j = acct_j
            ? ((int64_t)Tsnap[j] + (int64_t)t_amb_q) : (int64_t)t_amb_q;
        if (e_on) {
            const int64_t e_out = burn_j * t_abs_j;
            if (acct_j) {
                gE_e[j] -= e_out;
                cadd(cnt_e, COMB_E_DRAW, e_out);
            } else {
                cadd(cnt_e, COMB_E_MINT, e_out);
            }
        }

        // File each claimant's allocation (and its energy) on the slot buffers
        // for Pass B, and pay each claimant its P-R4 FUEL-BED deposit — the CPU
        // site verbatim: H_bed proportional to the O2 the claimant ACTUALLY
        // got, int64 shift + INT32_MAX clamp, then a positive SATURATING atomic
        // add at the claimant's own cell (several air cells can feed one
        // source, hence the atomic).
        for (int k = 0; k < n_cl; ++k) {
            alloc_e[(size_t)cl_slot[k] * n + j] = (q16)alloc[k];
            if (e_on) eslot_e[(size_t)cl_slot[k] * n + j] = alloc[k] * t_abs_j;
            if (heat_e != nullptr && alloc[k] > 0 && sc.H_bed_m_q > 0) {
                long long bed = (long long)mul_q16((q16)alloc[k], sc.H_bed_m_q);
                bed <<= sc.H_bed_shift;
                if (bed > (long long)0x7fffffff) bed = (long long)0x7fffffff;
                heat_atomic_sat_add_dev(&heat_e[cl_src[k]], (int32_t)bed);
            }
        }
        // The donor's mirror follows its own stored energy (a seam write
        // refreshes the MIRROR, never the rails) — the CPU site verbatim.
        if (acct_j) {
            temp_e[j] = gas_energy::mirror_q(
                gE_e[j], (int64_t)O2[j] + (int64_t)N2[j], t_amb_q);
        }
    }
}

// ---- K2: Pass B — source cells (combustion.cpp Pass B) ---------------------
// One thread per (env, flammable source i). Single writer of wall_hp[i] AND of
// i's five deposit-site slots (mass and energy). Sums the incoming slot
// allocations and pays the stoichiometric fuel cost ONCE for the total,
// floored ONCE at FUEL_FLOOR ("total-then-floor-once"). Reads alloc_slot /
// e_slot (written by K1, barriered before this launch). Own-cell writes only.
//
// P-O2b's second job here — WHERE the products land — is the CPU split rule
// verbatim: what a face donated directly returns to that face; the hop >= 2
// remainder splits evenly across the fire's OPEN sites (own tile + open faces),
// exact integer divide, leftover LSBs to the lowest site index. The ENERGY
// follows the mass exactly (#71: the CPU's proportional remainder split, the
// leftover raw counts riding the first site that took remainder mass).
template <int R>
__global__ void combustion_pass_b(
        int n_env, int h, int w,
        int32_t* __restrict__ wall_hp,
        const bool* __restrict__ flammable,
        const bool* __restrict__ solid, const bool* __restrict__ is_vacuum,
        const int32_t* __restrict__ alloc_slot,
        int32_t* __restrict__ dep_site,
        const int64_t* __restrict__ e_slot,       // nullable (no energy)
        int64_t* __restrict__ e_dep_site,         // nullable (no energy)
        const int32_t* __restrict__ fuel_per_o2_plane,   // nullable (R14)
        CombustionScalars sc,
        unsigned long long* __restrict__ cnt) {
    constexpr int NSLOT = 2 * R * (R + 1);
    const int n = h * w;
    const int32_t FUEL_FLOOR = CombustionSolver::FUEL_FLOOR;
    const bool e_on = (e_slot != nullptr);
    const long long total = (long long)n_env * n;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int e = (int)(idx / n);
        const int i = (int)(idx - (long long)e * n);
        const size_t c0 = (size_t)e * n;
        if (!flammable[c0 + i]) continue;
        const bool* sol_e = solid + c0;
        const bool* vac_e = is_vacuum + c0;
        const int32_t* alloc_e = alloc_slot + (size_t)e * NSLOT * n;
        int32_t* dep_e = dep_site + (size_t)e * 5 * n;
        const int64_t* eslot_e = e_on ? e_slot + (size_t)e * NSLOT * n : nullptr;
        int64_t* edep_e = e_on ? e_dep_site + (size_t)e * 5 * n : nullptr;
        unsigned long long* cnt_e = cnt + (size_t)e * COMB_CNT_SLOTS;

        const int y = i / w, x = i % w;
        // Donor for slot s is j = i - OFF[s]; slot s < 4 has OFF[s] == D4[s],
        // so that donor sits on i's face in direction D4_OPP[s].
        int64_t burn_i = 0;
        int64_t direct[4] = {0, 0, 0, 0};
        int64_t e_i = 0;
        int64_t direct_e[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int s = 0; s < NSLOT; ++s) {
            const int jy = y - c_off_dy[s], jx = x - c_off_dx[s];
            if (!in_bounds(jy, jx, h, w)) continue;
            const int j = jy * w + jx;
            const int64_t a = (int64_t)alloc_e[(size_t)s * n + j];
            burn_i += a;
            if (s < 4) direct[D4_OPP[s]] = a;
            if (e_on) {
                const int64_t ea = eslot_e[(size_t)s * n + j];
                e_i += ea;
                if (s < 4) direct_e[D4_OPP[s]] = ea;
            }
        }
        if (burn_i == 0) continue;   // this source drew no O2 this tick
        const q16 fpo_i = fuel_per_o2_plane ? fuel_per_o2_plane[c0 + i]
                                            : sc.fuel_per_o2_q;
        const q16 fuel_cost = narrow_round(mul_wide(fpo_i, (q16)burn_i));
        int32_t* whp = wall_hp + c0;
        whp[i] -= fuel_cost;
        if (whp[i] < FUEL_FLOOR) whp[i] = FUEL_FLOOR;

        // ---- P-O2b: WHERE THE PRODUCTS LAND — the CPU site verbatim -------
        bool site_open[5];
        int  m = 0;
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            site_open[d] = false;
            const int sy = y + D4_dy[d], sx = x + D4_dx[d];
            if (!in_bounds(sy, sx, h, w)) continue;
            const int c = sy * w + sx;
            if (sol_e[c] || vac_e[c]) continue;
            site_open[d] = true; ++m;
        }
        site_open[4] = (!sol_e[i] && !vac_e[i]);   // the fire's OWN tile
        if (site_open[4]) ++m;

        const int64_t hop1 = direct[0] + direct[1] + direct[2] + direct[3];
        int64_t rem = burn_i - hop1;
        if (m == 0) rem = 0;                       // unreachable; guarded, not assumed
        const int64_t even  = (m > 0) ? rem / m : 0;
        const int64_t extra = (m > 0) ? rem - even * m : 0;
        int64_t rem_mass_share[5] = {0, 0, 0, 0, 0};
        int taken = 0;
        #pragma unroll
        for (int t = 0; t < 5; ++t) {
            if (!site_open[t]) continue;
            rem_mass_share[t] = even + ((taken < (int)extra) ? 1 : 0);
            ++taken;
            int64_t share = rem_mass_share[t];
            if (t < 4) share += direct[t];
            dep_e[(size_t)t * n + i] = (q16)share;
        }
        if (e_on) {
            // ---- arc #54 P-G1b: THE ENERGY FOLLOWS THE MASS, EXACTLY — the
            // CPU site verbatim (#71): hop-1 energy rides its own face home;
            // the hop >= 2 remainder energy splits in proportion to the
            // remainder MASS each site received (`q·share + (r·share)/rem`
            // from `rem_e = q·rem + r`, never leaving int64), the floor's
            // leftover riding the first site that took mass.
            const int64_t hop1_e = direct_e[0] + direct_e[1]
                                 + direct_e[2] + direct_e[3];
            int64_t rem_e = e_i - hop1_e;
            if (m == 0) { cadd(cnt_e, COMB_E_EXPORT, rem_e); rem_e = 0; }
            const int64_t q = (rem > 0) ? rem_e / rem : 0;
            const int64_t r = (rem > 0) ? rem_e - q * rem : 0;
            int64_t placed = 0;
            int first_massive = -1;
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                if (!site_open[t]) continue;
                int64_t share_e = 0;
                if (rem_mass_share[t] > 0 && rem > 0) {
                    share_e = q * rem_mass_share[t]
                            + (r * rem_mass_share[t]) / rem;
                    placed += share_e;
                    if (first_massive < 0) first_massive = t;
                }
                if (t < 4) share_e += direct_e[t];
                edep_e[(size_t)t * n + i] = share_e;
            }
            if (first_massive >= 0 && rem_e > placed) {
                edep_e[(size_t)first_massive * n + i] += rem_e - placed;
            } else if (first_massive < 0 && rem_e != 0) {
                cadd(cnt_e, COMB_E_EXPORT, rem_e);
            }
        }
    }
}

// ---- K3: Pass C — air cells, THE DEPOSIT (combustion.cpp Pass C) -----------
// One thread per (env, open cell s). Single writer of SOOT[s], N2[s],
// temperature[s] and gas_energy[s]. Gathers what the fires around it decided to
// deposit here (K2's dep_site / e_dep_site, barriered before this launch) —
// the SECOND GATHER v5.2 asks for. The arithmetic is the CPU deposit verbatim:
// the Dalton split, the soot mint, the parcel split in the SAME proportion as
// its mass, the parcel routed by the site's medium, then the fire heat landed
// in the conserved field through the RAILED seam on an accountable gas cell
// (#71) or added to `temperature` on an object / non-accountable site.
__global__ void combustion_pass_c(
        int n_env, int h, int w, int n_gases,
        int o2_idx, int inert_n2_idx, int black_smoke_idx,
        int32_t* __restrict__ gas, int32_t* __restrict__ temperature,
        const bool* __restrict__ solid, const bool* __restrict__ is_vacuum,
        const bool* __restrict__ thermal_solid,     // nullable
        const int32_t* __restrict__ heat_inv_shift, // nullable
        const bool* __restrict__ is_ambient,        // nullable
        const int32_t* __restrict__ dep_site,
        const int64_t* __restrict__ e_dep_site,     // nullable (no energy)
        int64_t* __restrict__ gas_energy,           // nullable (no energy)
        const int32_t* __restrict__ t_amb_env,
        CombustionScalars sc,
        unsigned long long* __restrict__ cnt) {
    const int n = h * w;
    const bool e_on = (gas_energy != nullptr);
    const long long total = (long long)n_env * n;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int e = (int)(idx / n);
        const int s = (int)(idx - (long long)e * n);
        const size_t c0 = (size_t)e * n;
        const bool* sol_e = solid + c0;
        const bool* vac_e = is_vacuum + c0;
        if (sol_e[s] || vac_e[s]) continue;
        const int32_t* O2 = gas + ((size_t)e * n_gases + o2_idx) * n;
        int32_t* N2 = gas + ((size_t)e * n_gases + inert_n2_idx) * n;
        int32_t* SOOT = gas + ((size_t)e * n_gases + black_smoke_idx) * n;
        int32_t* temp_e = temperature + c0;
        const bool* tsol_e = thermal_solid ? thermal_solid + c0 : nullptr;
        const int32_t* shift_e = heat_inv_shift ? heat_inv_shift + c0 : nullptr;
        const bool* amb_e = is_ambient ? is_ambient + c0 : nullptr;
        const int32_t* dep_e = dep_site + (size_t)e * 5 * n;
        const int64_t* edep_e = e_on ? e_dep_site + (size_t)e * 5 * n : nullptr;
        int64_t* gE_e = e_on ? gas_energy + c0 : nullptr;
        unsigned long long* cnt_e = cnt + (size_t)e * COMB_CNT_SLOTS;
        const int32_t t_amb_q = t_amb_env[e];

        const int y = s / w, x = s % w;
        int64_t burn_dep = (int64_t)dep_e[(size_t)4 * n + s];
        int64_t e_dep = e_on ? edep_e[(size_t)4 * n + s] : 0;
        for (int d = 0; d < 4; ++d) {
            const int iy = y + D4_dy[d], ix = x + D4_dx[d];
            if (!in_bounds(iy, ix, h, w)) continue;
            const int i = iy * w + ix;
            burn_dep += (int64_t)dep_e[(size_t)D4_OPP[d] * n + i];
            if (e_on) e_dep += edep_e[(size_t)D4_OPP[d] * n + i];
        }
        if (burn_dep == 0) continue;

        const q16 soot = narrow_round(mul_wide((q16)burn_dep, sc.soot_yield_q));
        // #88: a bulk-count mass entering the smoke plane in the trace currency
        // (the CPU twin's lines): the ATTEMPTED mint booked, then the counted
        // saturation, its excess and any negative hit booked.
        {
            const int64_t mint = fixedpoint::trace_from_bulk((int64_t)soot);
            int64_t c = 0, ng = 0;
            SOOT[s] = fixedpoint::trace_add_sat(SOOT[s], mint, &c, &ng);
            cadd(cnt_e, COMB_TRACE_MINT, mint);
            cadd(cnt_e, COMB_TRACE_CLIP, c);
            cadd(cnt_e, COMB_TRACE_NEG, ng);
        }
        N2[s]   += (q16)(burn_dep - (int64_t)soot);

        // ---- arc #54 P-G1b: THE PARCEL LANDS — the CPU site verbatim (#71):
        // split EXACTLY in the proportion of its mass (`soot/burn_dep` shed
        // with the black smoke, the rest delivered with the bulk N2), as
        // `q·soot + (r·soot)/burn_dep` from `e_dep = q·burn_dep + r`.
        const bool acct_s = e_on
            && gas_energy::accountable(sol_e, tsol_e, vac_e, amb_e, s);
        int64_t e_shed = 0, e_deliver = 0;
        if (e_on) {
            if (e_dep > 0 && soot > 0) {
                const int64_t q = e_dep / burn_dep;
                const int64_t r = e_dep - q * burn_dep;      // in [0, burn_dep)
                e_shed = q * (int64_t)soot + (r * (int64_t)soot) / burn_dep;
            }
            e_deliver = e_dep - e_shed;
        }

        // ONE aggregate heat deposit against the POST-burn N_total (delta delta).
        // burn*H_fuel with H_fuel = H_FUEL_M * 2^H_FUEL_SHIFT (T5a) — the CPU
        // twin's body, and the H_bed site's int64-shift-then-clamp idiom.
        long long dep_wide = (long long)mul_q16((q16)burn_dep, sc.H_fuel_m_q);
        dep_wide <<= sc.H_fuel_shift;
        if (dep_wide > (long long)INT32_MAX) dep_wide = (long long)INT32_MAX;
        const q16 deposit = (q16)dep_wide;
        q16 dT;
        // THERMAL-MASS AXIS, P-EOS (ruling §2 site 3) — the CPU branch verbatim:
        // an OBJECT burn site (furniture: open + gas-holding, but thermally
        // solid) converts the deposit through its own heat_inv_shift, not through
        // the thin pore gas's N. The n_floor counter is untouched on that path.
        const bool object_site = (tsol_e != nullptr) && (shift_e != nullptr)
                              && tsol_e[s];
        // Route the parcel now that the deposit site's MEDIUM is known — the
        // CPU site verbatim: a thermal solid or a ring cell takes the whole
        // parcel out of the gas books through a NAMED counter; only an
        // accountable gas cell receives it, and then only the bulk share.
        if (e_on) {
            if (object_site || (tsol_e != nullptr && tsol_e[s])) {
                cadd(cnt_e, COMB_E_TS_PRODUCTS, e_dep);
            } else if (!acct_s) {
                cadd(cnt_e, COMB_E_EXPORT, e_dep);
            } else {
                gE_e[s] += e_deliver;
                cadd(cnt_e, COMB_E_DELIVER, e_deliver);
                cadd(cnt_e, COMB_E_SOOT_SHED, e_shed);
            }
        }
        if (object_site) {
            // M1: WIDE and SIGNED — the CPU twin's line for line.
            const int shift = shift_e[s];   // log2(thermal_mass)
            int64_t dT_obj = shr_round0_signed_i64((int64_t)deposit, shift);
            if (dT_obj < 0) dT_obj = 0;
            if (dT_obj > (int64_t)INT32_MAX) dT_obj = (int64_t)INT32_MAX;
            dT = (q16)dT_obj;
        } else {
            const q16 n_real_s = (q16)((int64_t)O2[s] + (int64_t)N2[s]);
            q16 n_total_s = n_real_s;
            if (n_total_s < sc.n_floor_q) {
                n_total_s = sc.n_floor_q;
                cadd(cnt_e, COMB_HEAT_FLOOR_HITS, 1);
            }
            const q16 recip_n = reciprocal_q16_dev(n_total_s);
            if (n_total_s != n_real_s) {
                // P-E2b: the CPU e_deposit_drop_sum fold — WIDE throughout.
                const int64_t e_over_n_wide =
                    mul_wide(deposit, recip_n) >> FP_SHIFT;   // deposit/floor
                const int64_t drop = (int64_t)deposit
                    - ((e_over_n_wide * (int64_t)n_real_s) >> FP_SHIFT);
                cadd(cnt_e, COMB_DEPOSIT_DROP, drop);
            }
            // P-E2b: the WIDE deposit/(N*c_v) chain, clamped to a safe
            // non-negative int32 range BEFORE narrowing.
            const int64_t dT_wide =
                deposit_dT_wide_q16_dev(deposit, recip_n, sc.recip_cv);
            dT = (q16)(dT_wide < 0 ? 0 : (dT_wide > 0x7fffffffLL ? 0x7fffffffLL
                                                                  : dT_wide));
        }
        // ---- arc #54 P-G1b: the fire heat lands in the ENERGY field (#71) —
        // the CPU site verbatim: `N·dT` through the RAILED seam (combustion
        // runs after the once-per-tick recovery, so the rail lives here), the
        // seam called even when dT == 0 so the products credited above reach
        // this cell's mirror.
        if (acct_s && !object_site) {
            const int64_t nb = (int64_t)O2[s] + (int64_t)N2[s];
            const int64_t de = nb * (int64_t)dT;
            int64_t e_rail_local = 0, hits_local = 0;
            gas_energy::deposit_railed(gE_e, temp_e, s, de, nb, t_amb_q,
                                       sc.t_max_phys_q,
                                       &e_rail_local, &hits_local);
            cadd(cnt_e, COMB_E_RAIL, e_rail_local);
            cadd(cnt_e, COMB_T_MAX_PHYS_HITS, hits_local);
            cadd(cnt_e, COMB_E_HEAT, de);
        } else {
            // The object-site / non-accountable deposit into `temperature`,
            // and (arc #54 P-G5) the object site's ACTUAL applied ΔT priced at
            // its real capacity — the SAME conduction::cell_capacity_q kit.
            const int32_t t_before = temp_e[s];
            heat_saturating_add_dev(&temp_e[s], dT);
            if (temp_e[s] > sc.t_max_phys_q) {                // v2.4 rail
                temp_e[s] = sc.t_max_phys_q;
                cadd(cnt_e, COMB_T_MAX_PHYS_HITS, 1);
            }
            if (object_site) {
                int64_t cap_used_s = 0, cap_real_s = 0;
                conduction::cell_capacity_q(true, shift_e[s], 0, 0, 0,
                                            &cap_used_s, &cap_real_s);
                cadd(cnt_e, COMB_E_SOLID_HEAT,
                     ((int64_t)temp_e[s] - t_before) * cap_real_s);
            }
        }
    }
}

// Owns every device allocation of one per-call step; frees them on any exit
// (a cuda_check throw included), so a failed call leaks nothing.
struct DevAllocs {
    std::vector<void*> ptrs;
    template <typename T>
    T* alloc(size_t bytes, const char* what) {
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, bytes), what);
        ptrs.push_back(p);
        return static_cast<T*>(p);
    }
    ~DevAllocs() { for (void* p : ptrs) cudaFree(p); }
};

}  // namespace

CombustionScalars combustion_bake_scalars(
        float o2_thresh_burn, float o2_frac_ext, float o2_frac_full,
        float fire_T_ext, float fire_T_span, float hotf_cap,
        float H_BED_M, int H_BED_SHIFT, float fuel_per_o2, float soot_yield,
        float H_FUEL_M, int H_FUEL_SHIFT, float c_v, float n_floor_heat,
        float T_MAX_PHYS, int32_t p_ext_q, int32_t p_full_q) {
    // VERBATIM of combustion.cpp's load-time bake (in double, then quantize):
    // each line below is the CPU step's own.
    CombustionScalars sc;
    sc.o2_thresh_q   = quantize((double)o2_thresh_burn);
    sc.soot_yield_q  = quantize((double)soot_yield);
    sc.H_fuel_m_q    = quantize((double)H_FUEL_M);
    sc.H_fuel_shift  = (H_FUEL_SHIFT > 0) ? H_FUEL_SHIFT : 0;
    sc.fuel_per_o2_q = quantize((double)fuel_per_o2);
    sc.H_bed_m_q     = quantize((double)H_BED_M);
    sc.H_bed_shift   = (H_BED_SHIFT > 0) ? H_BED_SHIFT : 0;
    const double c_v_safe = (c_v > 0.0f) ? (double)c_v : 1.0;
    // T5b: ONE integer representation of `c_v` (`c_v_q`) and its exact inverse.
    const q16 c_v_q  = quantize(c_v_safe);
    sc.recip_cv      = make_recip((double)c_v_q / 65536.0);
    sc.n_floor_q     = quantize((double)n_floor_heat);
    sc.t_max_phys_q  = quantize((double)T_MAX_PHYS);
    sc.x_ext_q       = quantize((double)o2_frac_ext);
    const double x_span = (double)o2_frac_full - (double)o2_frac_ext;
    sc.x_degenerate  = (x_span <= 0.0) ? 1 : 0;
    sc.recip_x_span  = sc.x_degenerate ? 0 : make_recip(x_span);
    sc.x_n_floor     = quantize(0.01);   // 655 counts (see fire_simulation.cpp)
    const o2_pressure::Factor pf = o2_pressure::bake(p_ext_q, p_full_q);
    sc.p_ext_q       = pf.p_ext_q;
    sc.p_full_q      = pf.p_full_q;
    sc.p_inv_span_q  = pf.inv_span_q;
    sc.fire_T_ext_q  = quantize((double)fire_T_ext);
    sc.recip_T_span  = make_recip((double)fire_T_span);
    sc.hotf_cap_q    = quantize((double)hotf_cap);
    // W_hop: quantize(2/(1+d)), i.e. 1/(1+d) NORMALIZED so W_hop[1] == FP_ONE
    // — the R = 1 identity.
    sc.w_hop_q[0] = 0;
    for (int d = 1; d <= cd::R_MAX; ++d) sc.w_hop_q[d] = quantize(2.0 / (1.0 + (double)d));
    return sc;
}
static_assert(cd::R_MAX + 1 == 4, "CombustionScalars::w_hop_q is sized R_MAX + 1");

void combustion_upload_tables() {
    cuda_check(cudaMemcpyToSymbol(c_off_dy, cd::OFF_DY, sizeof(cd::OFF_DY)), "sym off_dy");
    cuda_check(cudaMemcpyToSymbol(c_off_dx, cd::OFF_DX, sizeof(cd::OFF_DX)), "sym off_dx");
    cuda_check(cudaMemcpyToSymbol(c_ball_dy, cd::BALL_DY, sizeof(cd::BALL_DY)), "sym ball_dy");
    cuda_check(cudaMemcpyToSymbol(c_ball_dx, cd::BALL_DX, sizeof(cd::BALL_DX)), "sym ball_dx");
    cuda_check(cudaMemcpyToSymbol(c_ball_nbr, cd::BALL_NBR.v, sizeof(cd::BALL_NBR.v)), "sym ball_nbr");
    cuda_check(cudaMemcpyToSymbol(c_ball_slot, cd::BALL_SLOT.v, sizeof(cd::BALL_SLOT.v)), "sym ball_slot");
}

int combustion_launch_resident(
        int n_env, int h, int w, int n_gases,
        int o2_idx, int inert_n2_idx, int black_smoke_idx, int draw_r,
        int32_t* d_gas, int32_t* d_temperature, int32_t* d_wall_hp,
        int64_t* d_gas_energy,
        const int32_t* d_fire, const int32_t* d_ignition_temp_q16,
        const bool* d_flammable, const bool* d_solid, const bool* d_is_vacuum,
        const bool* d_is_ambient, const bool* d_thermal_solid,
        const int32_t* d_heat_inv_shift,
        int32_t* d_heat, int32_t* d_dem_acc, int acc_depth,
        const int32_t* d_perm_q,
        const int32_t* d_fire_T_ext_plane, const int32_t* d_fuel_per_o2_plane,
        const int32_t* d_atmosphere,
        const int32_t* d_burn_cap_q, const int32_t* d_t_amb_q,
        const CombustionScalars& sc,
        int32_t* d_tsnap, int32_t* d_alloc, int32_t* d_dep,
        int64_t* d_e_slot, int64_t* d_e_dep,
        unsigned long long* d_cnt) {
    const int n = h * w;
    if (n_env <= 0 || n <= 0) return 0;
    if (draw_r < 1 || draw_r > cd::R_MAX) {
        throw std::invalid_argument(
            "combustion_launch_resident: draw_r out of range (1.."
            + std::to_string(cd::R_MAX) + ")");
    }
    const int n_slots = cd::slot_count(draw_r);
    if (d_dem_acc != nullptr && acc_depth < n_slots) {
        throw std::invalid_argument(
            "combustion_launch_resident: MAX_CLAIMANTS cap hit — dem_acc depth "
            + std::to_string(acc_depth) + " < slot_count(draw_r) "
            + std::to_string(n_slots));
    }
    if (d_gas_energy != nullptr && (d_e_slot == nullptr || d_e_dep == nullptr)) {
        throw std::invalid_argument(
            "combustion_launch_resident: gas_energy needs the e_slot / e_dep scratch");
    }
    // With no energy field the energy scratch is never touched.
    int64_t* e_slot = (d_gas_energy != nullptr) ? d_e_slot : nullptr;
    int64_t* e_dep  = (d_gas_energy != nullptr) ? d_e_dep : nullptr;

    const int block = 256;
    const long long total = (long long)n_env * n;
    const int grid = (int)((total + block - 1) / block);
    const int pgrid = (int)(((long long)n_env * COMB_CNT_SLOTS > total
                             ? (long long)n_env * COMB_CNT_SLOTS : total)
                            + block - 1) / block;

    // K0: the snapshot + the scratch / book zeroing (the explicit freeze).
    combustion_prep<<<pgrid, block, 0, tstream()>>>(d_temperature, d_tsnap, d_alloc, d_dep,
                                      e_slot, e_dep, d_cnt, n_env, n, n_slots);
    cuda_check(cudaGetLastError(), "prep launch");

    // K1 + K2 are TEMPLATED ON draw_r so every array bound and loop trip count
    // is a compile-time constant. One switch, three instantiations; the R the
    // config asks for is the only one that runs.
    //
    // ---- REGISTER / LOCAL-MEMORY COST (round-3 F6's named concern) ---------
    // F6 warned that the fixed-size per-thread arrays sized by MAX_CLAIMANTS
    // could spill, and pre-authorized restructuring to the offset-keyed plane
    // form if they did. MEASURED on sm_89 (ptxas -v, the shipping flags) before
    // #71 added the energy ledger (pass_a: +1 int64 debit, the e_slot store;
    // pass_b: +5 int64 energy shares; pass_c: the railed seam):
    //
    //   kernel      R    registers   stack frame   spill st/ld
    //   pass_a      1        40         176 B         0 / 0
    //   pass_a      2        70         512 B         0 / 0      <- SHIPPED
    //   pass_a      3       168        1040 B         0 / 0
    //   pass_b    1/2/3   34/40/40       32 B         0 / 0
    //   pass_c      -        22           0 B         0 / 0
    //
    // At the shipped R = 2 the pass is a few percent of the tick; the plane
    // form is not taken (it would trade registers for global traffic).
    // R = 3 (~17% occupancy) is the documented cost of the sweep's upper point.
    #define PO2B_LAUNCH(RVAL)                                                  \
        combustion_pass_a<RVAL><<<grid, block, 0, tstream()>>>(                              \
            n_env, h, w, n_gases, o2_idx, inert_n2_idx, acc_depth,             \
            d_gas, d_tsnap, d_temperature, d_wall_hp, d_fire,                  \
            d_flammable, d_solid, d_is_vacuum, d_thermal_solid, d_is_ambient,  \
            d_ignition_temp_q16, d_alloc, d_perm_q, d_heat, d_dem_acc,         \
            d_fire_T_ext_plane, d_atmosphere, d_gas_energy, e_slot,            \
            d_burn_cap_q, d_t_amb_q, sc, d_cnt);                               \
        cuda_check(cudaGetLastError(), "pass_a launch");                       \
        combustion_pass_b<RVAL><<<grid, block, 0, tstream()>>>(                              \
            n_env, h, w, d_wall_hp, d_flammable, d_solid, d_is_vacuum,         \
            d_alloc, d_dep, e_slot, e_dep, d_fuel_per_o2_plane, sc, d_cnt);    \
        cuda_check(cudaGetLastError(), "pass_b launch")

    // K1: Pass A (after K0 — d_tsnap is settled before any read).
    // K2: Pass B (separate launch = grid barrier: d_alloc / e_slot fully
    //     written by K1, and K1's wall_hp reads all complete before K2 writes).
    switch (draw_r) {
        case 1: PO2B_LAUNCH(1); break;
        case 2: PO2B_LAUNCH(2); break;
        case 3: PO2B_LAUNCH(3); break;
        default: throw std::invalid_argument("combustion_launch_resident: draw_r");
    }
    #undef PO2B_LAUNCH

    // K3: Pass C — the re-sited deposit (separate launch = grid barrier: d_dep
    // / e_dep fully written by K2 before any thread gathers them, and every K1
    // O2 debit and donor-energy debit is settled before Pass C reads them).
    combustion_pass_c<<<grid, block, 0, tstream()>>>(
        n_env, h, w, n_gases, o2_idx, inert_n2_idx, black_smoke_idx,
        d_gas, d_temperature, d_solid, d_is_vacuum, d_thermal_solid,
        d_heat_inv_shift, d_is_ambient, d_dep, e_dep, d_gas_energy,
        d_t_amb_q, sc, d_cnt);
    cuda_check(cudaGetLastError(), "pass_c launch");
    return 4;
}

void combustion_step(
        int32_t* gas, int n_gases,
        int o2_idx, int inert_n2_idx, int black_smoke_idx,
        int32_t* temperature, int32_t* wall_hp,
        const int32_t* fire,
        const bool* flammable, const bool* solid, const bool* is_vacuum,
        const int32_t* ignition_temp_q16,
        int h, int w, float dt, float c_v, float n_floor_heat,
        float burn_rate, float o2_thresh_burn,
        float H_FUEL_M, int H_FUEL_SHIFT, float soot_yield,
        float fuel_per_o2, float o2_frac_ext, float o2_frac_full, float T_MAX_PHYS,
        int64_t* heat_floor_hits, int64_t* t_max_phys_hits,
        int64_t* e_deposit_drop_sum,
        const bool* thermal_solid, const int32_t* heat_inv_shift,
        int32_t* heat, float H_BED_M, int H_BED_SHIFT,
        int32_t* dem_acc,
        int draw_r, const float* dyn_permeability, int max_claimants,
        float fire_T_ext, float fire_T_span, float hotf_cap,
        const int32_t* fire_T_ext_plane,
        const int32_t* fuel_per_o2_plane,
        const int32_t* atmosphere,              // #7: pressure factor input (nullable)
        int32_t p_ext_q, int32_t p_full_q,      // #7: its edges, Q16.16 atm
        int64_t* trace_mint, int64_t* trace_clip, int64_t* trace_neg,   // #88 P2
        int64_t* gas_energy, const bool* is_ambient, int32_t t_amb_q,   // #71
        int64_t* e_books) {

    // --- Guards (VERBATIM of combustion.cpp's head, in its order). A guarded
    //     early-return leaves ALL fields untouched (no launch, no D2H) —
    //     matching the CPU no-op exactly. ----------------------------------
    if (h <= 0 || w <= 0 || dt <= 0.0f) return;
    if (o2_idx < 0 || o2_idx >= n_gases) return;
    if (inert_n2_idx < 0 || inert_n2_idx >= n_gases) return;
    if (black_smoke_idx < 0 || black_smoke_idx >= n_gases) return;
    const int n = h * w;
    if (n <= 0) return;

    const q16 burn_cap_q = quantize((double)burn_rate * (double)dt);
    const CombustionScalars sc = combustion_bake_scalars(
        o2_thresh_burn, o2_frac_ext, o2_frac_full, fire_T_ext, fire_T_span,
        hotf_cap, H_BED_M, H_BED_SHIFT, fuel_per_o2, soot_yield, H_FUEL_M,
        H_FUEL_SHIFT, c_v, n_floor_heat, T_MAX_PHYS, p_ext_q, p_full_q);

    if (burn_cap_q <= 0) return;   // nothing burns this tick (fields untouched)

    // The SAME hard checks, after the burn-cap return exactly as on the CPU: a
    // radius past the baked tables, or a dem_acc plane too shallow for the
    // radius, would alias two sources' sub-count debts onto one slot.
    if (draw_r < 1 || draw_r > cd::R_MAX) {
        throw std::runtime_error("cuda combustion_step: draw_r out of range");
    }
    const int n_slots = cd::slot_count(draw_r);
    if (dem_acc != nullptr && max_claimants < n_slots) {
        throw std::runtime_error("cuda combustion_step: MAX_CLAIMANTS cap hit");
    }
    // The plane's DECLARED depth is max_claimants; only the first n_slots rows
    // are live (a deeper plane simply carries unused rows).
    const int acc_depth = (dem_acc != nullptr) ? max_claimants : n_slots;
    // The permeability plane, quantized ONCE on the host so no float ever
    // reaches the device and both backends read the identical integers.
    std::vector<int32_t> perm_q_h;
    if (draw_r > 1 && dyn_permeability != nullptr) {
        perm_q_h.resize(n);
        for (int i = 0; i < n; ++i) perm_q_h[i] = quantize((double)dyn_permeability[i]);
    }

    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    const size_t nb64  = (size_t)n * sizeof(int64_t);
    DevAllocs a;
    auto up = [](void* d, const void* hptr, size_t bytes, const char* what) {
        cuda_check(cudaMemcpy(d, hptr, bytes, cudaMemcpyHostToDevice), what);
    };

    // The gas block as (1, n_gases, h, w): only the three planes the pass
    // touches are uploaded (and downloaded) — the others are never read.
    int32_t* d_gas = a.alloc<int32_t>((size_t)n_gases * nb, "malloc gas");
    int32_t* const planes_h[3] = {gas + (size_t)o2_idx * n,
                                  gas + (size_t)inert_n2_idx * n,
                                  gas + (size_t)black_smoke_idx * n};
    int32_t* const planes_d[3] = {d_gas + (size_t)o2_idx * n,
                                  d_gas + (size_t)inert_n2_idx * n,
                                  d_gas + (size_t)black_smoke_idx * n};
    for (int p = 0; p < 3; ++p) up(planes_d[p], planes_h[p], nb, "H2D gas plane");

    int32_t* d_temp = a.alloc<int32_t>(nb, "malloc temperature");
    int32_t* d_whp  = a.alloc<int32_t>(nb, "malloc wall_hp");
    int32_t* d_fire = a.alloc<int32_t>(nb, "malloc fire");
    int32_t* d_ign  = a.alloc<int32_t>(nb, "malloc ignition_temp");
    bool* d_flam    = a.alloc<bool>(nbool, "malloc flammable");
    bool* d_solid   = a.alloc<bool>(nbool, "malloc solid");
    bool* d_vac     = a.alloc<bool>(nbool, "malloc is_vacuum");
    up(d_temp, temperature, nb, "H2D temp");
    up(d_whp, wall_hp, nb, "H2D wall_hp");
    up(d_fire, fire, nb, "H2D fire");
    up(d_ign, ignition_temp_q16, nb, "H2D ign");
    up(d_flam, flammable, nbool, "H2D flammable");
    up(d_solid, solid, nbool, "H2D solid");
    up(d_vac, is_vacuum, nbool, "H2D is_vacuum");

    // The nullable inputs — each allocated + uploaded only when supplied, so
    // the kernel takes the CPU's own null branch otherwise. (#71: thermal_solid
    // and heat_inv_shift ride independently now — the accountable set reads
    // the thermal mask on its own, the object deposit needs both, as on the
    // CPU.)
    bool* d_tsol = nullptr;
    if (thermal_solid) {
        d_tsol = a.alloc<bool>(nbool, "malloc thermal_solid");
        up(d_tsol, thermal_solid, nbool, "H2D thermal_solid");
    }
    int32_t* d_shift = nullptr;
    if (heat_inv_shift) {
        d_shift = a.alloc<int32_t>(nb, "malloc heat_inv_shift");
        up(d_shift, heat_inv_shift, nb, "H2D heat_inv_shift");
    }
    bool* d_amb = nullptr;
    if (is_ambient) {
        d_amb = a.alloc<bool>(nbool, "malloc is_ambient");
        up(d_amb, is_ambient, nbool, "H2D is_ambient");
    }
    int32_t* d_heat = nullptr;
    if (heat) {
        d_heat = a.alloc<int32_t>(nb, "malloc heat");
        up(d_heat, heat, nb, "H2D heat");
    }
    int32_t* d_T_ext_plane = nullptr;
    if (fire_T_ext_plane) {
        d_T_ext_plane = a.alloc<int32_t>(nb, "malloc fire_T_ext_plane");
        up(d_T_ext_plane, fire_T_ext_plane, nb, "H2D fire_T_ext_plane");
    }
    int32_t* d_fpo_plane = nullptr;
    if (fuel_per_o2_plane) {
        d_fpo_plane = a.alloc<int32_t>(nb, "malloc fuel_per_o2_plane");
        up(d_fpo_plane, fuel_per_o2_plane, nb, "H2D fuel_per_o2_plane");
    }
    int32_t* d_atm = nullptr;
    if (atmosphere) {
        d_atm = a.alloc<int32_t>(nb, "malloc atmosphere");
        up(d_atm, atmosphere, nb, "H2D atmosphere");
    }
    int32_t* d_dem_acc = nullptr;
    if (dem_acc) {
        d_dem_acc = a.alloc<int32_t>((size_t)acc_depth * nb, "malloc dem_acc");
        up(d_dem_acc, dem_acc, (size_t)acc_depth * nb, "H2D dem_acc");
    }
    int32_t* d_perm = nullptr;
    if (!perm_q_h.empty()) {
        d_perm = a.alloc<int32_t>(nb, "malloc perm_q");
        up(d_perm, perm_q_h.data(), nb, "H2D perm_q");
    }
    int64_t* d_gE = nullptr;
    int64_t* d_e_slot = nullptr;
    int64_t* d_e_dep = nullptr;
    if (gas_energy) {
        d_gE = a.alloc<int64_t>(nb64, "malloc gas_energy");
        up(d_gE, gas_energy, nb64, "H2D gas_energy");
        d_e_slot = a.alloc<int64_t>((size_t)n_slots * nb64, "malloc e_slot");
        d_e_dep  = a.alloc<int64_t>((size_t)5 * nb64, "malloc e_dep_site");
    }
    // Per-call scratch (the CPU's fresh per-tick vectors; the core zeroes them).
    int32_t* d_tsnap = a.alloc<int32_t>(nb, "malloc tsnap");
    int32_t* d_alloc = a.alloc<int32_t>((size_t)n_slots * nb, "malloc alloc_slot");
    int32_t* d_dep   = a.alloc<int32_t>((size_t)5 * nb, "malloc dep_site");
    unsigned long long* d_cnt = a.alloc<unsigned long long>(
        COMB_CNT_SLOTS * sizeof(unsigned long long), "malloc books");
    // The per-env scalars (N == 1).
    const int32_t env_h[2] = {burn_cap_q, t_amb_q};
    int32_t* d_env = a.alloc<int32_t>(sizeof(env_h), "malloc env scalars");
    up(d_env, env_h, sizeof(env_h), "H2D env scalars");

    // The draw geometry from combustion.h's single definition.
    combustion_upload_tables();

    combustion_launch_resident(
        1, h, w, n_gases, o2_idx, inert_n2_idx, black_smoke_idx, draw_r,
        d_gas, d_temp, d_whp, d_gE, d_fire, d_ign,
        d_flam, d_solid, d_vac, d_amb, d_tsol, d_shift,
        d_heat, d_dem_acc, acc_depth, d_perm,
        d_T_ext_plane, d_fpo_plane, d_atm,
        d_env + 0, d_env + 1, sc,
        d_tsnap, d_alloc, d_dep, d_e_slot, d_e_dep, d_cnt);

    cuda_check(cudaDeviceSynchronize(), "sync");

    // D2H the mutated fields (the three gas planes, temperature, wall_hp, and
    // the nullable in/out planes).
    auto down = [](void* hptr, const void* d, size_t bytes, const char* what) {
        cuda_check(cudaMemcpy(hptr, d, bytes, cudaMemcpyDeviceToHost), what);
    };
    for (int p = 0; p < 3; ++p) down(planes_h[p], planes_d[p], nb, "D2H gas plane");
    down(temperature, d_temp, nb, "D2H temp");
    down(wall_hp, d_whp, nb, "D2H wall_hp");
    if (heat) down(heat, d_heat, nb, "D2H heat");
    if (dem_acc) down(dem_acc, d_dem_acc, (size_t)acc_depth * nb, "D2H dem_acc");
    if (gas_energy) down(gas_energy, d_gE, nb64, "D2H gas_energy");

    unsigned long long books[COMB_CNT_SLOTS] = {};
    down(books, d_cnt, sizeof(books), "D2H books");
    auto b = [&](int slot) { return (int64_t)books[slot]; };
    if (heat_floor_hits)    *heat_floor_hits    += b(COMB_HEAT_FLOOR_HITS);
    if (t_max_phys_hits)    *t_max_phys_hits    += b(COMB_T_MAX_PHYS_HITS);
    if (e_deposit_drop_sum) *e_deposit_drop_sum += b(COMB_DEPOSIT_DROP);
    if (trace_mint) *trace_mint += b(COMB_TRACE_MINT);
    if (trace_clip) *trace_clip += b(COMB_TRACE_CLIP);
    if (trace_neg)  *trace_neg  += b(COMB_TRACE_NEG);
    if (e_books) {
        static_assert(COMB_E_SOLID_HEAT - COMB_E_DRAW + 1 == COMB_E_BOOKS,
                      "e_books is the COMB_E_* block in order");
        for (int k = 0; k < COMB_E_BOOKS; ++k) e_books[k] += b(COMB_E_DRAW + k);
    }
}

namespace {
bool g_combustion_backend_cuda = false;
}
bool combustion_backend_is_cuda() { return g_combustion_backend_cuda; }
void set_combustion_backend_cuda(bool on) { g_combustion_backend_cuda = on; }

}  // namespace breach_cuda
