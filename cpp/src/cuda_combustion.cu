// ============================================================================
// EOS P6.9b combustion solver implementation — see cuda_combustion.h.
// A bit-identical GPU port of CombustionSolver::step (combustion.cpp), the
// P6.9a two-gather reformulation (docs/eos_p6_9_combustion_design.md §3–§4).
// This is a FAITHFUL bit-identity port, NOT a redesign: the CPU pass was
// reformulated into two order-free gathers precisely so it runs identically on
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
// raycaster.h::heat_saturating_add). mul_wide / narrow_round / mul_q16 are the
// FP_HD host/device helpers used directly.
//
// S. Feldman, J.F. O'Brien, O. Arikan, "Animating Suspended Particle
// Explosions", SIGGRAPH 2003 — the heat + product-yield + ignition-threshold
// source-term structure this pass follows (constants game-tuned, not lit-derived).
// ============================================================================
#include "cuda_combustion.h"
#include "combustion.h"    // CombustionSolver::FUEL_FLOOR (the compile-time 1-LSB floor)
#include "fixed_point.h"   // q16, quantize, mul_q16, mul_wide, narrow_round, make_recip
#include "cuda_fixedpoint_device.cuh"  // reciprocal_q16_dev, recip_mul_dev, heat_saturating_add_dev

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <vector>

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
// These are NOT re-authored here. They are UPLOADED at every step from
// combustion.h's `combustion_draw` namespace — the single definition the CPU
// solver walks — via cudaMemcpyToSymbol below. A transcription drift between
// the backends is therefore structurally impossible, which matters more than
// the few hundred bytes of copy: the tables ARE the law's geometry, and a
// single swapped offset would be a silent, tol-0-invisible fork on one machine.
namespace cd = combustion_draw;
__device__ __constant__ int8_t c_off_dy[cd::SLOTS_MAX];
__device__ __constant__ int8_t c_off_dx[cd::SLOTS_MAX];
__device__ __constant__ int8_t c_ball_dy[cd::BALL_MAX];
__device__ __constant__ int8_t c_ball_dx[cd::BALL_MAX];
__device__ __constant__ int8_t c_ball_nbr[cd::BALL_MAX * 4];
__device__ __constant__ int8_t c_ball_slot[cd::BALL_MAX * 4];
__device__ __constant__ int32_t c_w_hop[cd::R_MAX + 1];

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

// P-R4: SATURATING integer atomic add — the device twin of raycaster.h's
// `heat_saturating_add` (and a verbatim copy of cuda_raycaster.cu's
// heat_atomic_sat_add). Needed here because several AIR cells can feed the same
// flammable claimant, so the H_bed deposits at one source cell race. Order-free
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

// ---- K1: Pass A — air cells (combustion.cpp Pass A) ------------------------
// One thread per cell j. Single writer of O2[j] and of ALL of the claimant-slot
// buffers at index j. Reads d_tsnap (the snapshot gate), the pass-entry O2 at
// the OWN cell (read-before-write => every claimant sees pass-entry O2 — deltas
// beta/gamma), and pre-payment wall_hp (K2 writes it, barriered after this
// kernel — so this read is pre-payment, matching the CPU).
//
// P-O2b: TEMPLATED ON THE DRAW RADIUS. R is a compile-time constant so the
// per-thread arrays below are sized exactly (NSLOT = 4 / 12 / 24) and every
// loop bound is known to ptxas — the difference between an unrolled,
// register-resident gather and a fully indirect local-memory one. The register
// budget was round-3 F6's named concern; see the cost note at the launch site.
template <int R>
__global__ void combustion_pass_a(
        int32_t* __restrict__ O2, const int32_t* __restrict__ N2,
        const int32_t* __restrict__ tsnap, const int32_t* __restrict__ wall_hp,
        const int32_t* __restrict__ fire,
        const bool* __restrict__ flammable, const bool* __restrict__ solid,
        const bool* __restrict__ is_vacuum,
        const int32_t* __restrict__ ignition_temp_q16,
        int32_t* __restrict__ alloc_slot,
        const int32_t* __restrict__ perm_q,
        int h, int w, int32_t burn_cap_q, int32_t o2_thresh_q,
        int32_t x_ext_q, int64_t recip_x_span, bool x_degenerate,
        int32_t X_N_FLOOR,
        // P-R4: the fuel-bed deposit's target plane + its split constant.
        int32_t* __restrict__ heat, int32_t H_bed_m_q, int H_bed_shift,
        // D1: the error-feedback demand accumulator, (max_claimants, h, w).
        // Single writer per air cell (this thread owns ALL of ITS slots), so no
        // atomics and no order dependence — see combustion.h.
        int32_t* __restrict__ dem_acc,
        // R3 hot-burns-faster (docs/fire_3c_design_2026-09-01.md "Ruling R3"):
        // the demand-side hotf ramp — VERBATIM the CPU combustion.cpp bake.
        const int32_t* __restrict__ fire_T_ext_plane,
        int32_t fire_T_ext_q, int64_t recip_T_span, int32_t hotf_cap_q) {
    constexpr int NSLOT = 2 * R * (R + 1);         // 4, 12, 24
    constexpr int NBALL = 2 * (R - 1) * R + 1;     // 1, 5, 13
    const int n = h * w;
    const int32_t FUEL_FLOOR = CombustionSolver::FUEL_FLOOR;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        if (solid[j] || is_vacuum[j]) continue;

        const q16 o2j = O2[j];
        if (o2j <= o2_thresh_q) continue;   // starved: no claimant can burn here

        const int y = j / w, x = j % w;

        // o2f_j — the continuous-O2 factor at THIS air cell, LINEAR in its O2
        // MOLE FRACTION X_j = O2[j]/(O2[j]+N2[j]) (pass-entry). Same hoisted
        // x_ext_q/recip_x_span/x_degenerate/X_N_FLOOR as fire_logistic.
        const int64_t n_tot_j = (int64_t)o2j + (int64_t)N2[j];
        const q16 den_j = (n_tot_j < (int64_t)X_N_FLOOR) ? X_N_FLOOR : (q16)n_tot_j;
        const q16 Xj = mul_q16(o2j, reciprocal_q16_dev(den_j));
        const q16 o2f_j = x_degenerate
            ? ((Xj < x_ext_q) ? (q16)0 : (q16)FP_ONE)
            : clamp01_q_dev(recip_mul_dev(Xj - x_ext_q, recip_x_span));

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
            if (solid[c] || is_vacuum[c]) continue;   // open cells only; vacuum terminates
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
                    q16 pf = FP_ONE;
                    if (perm_q != nullptr) {
                        const q16 pa = perm_q[bcell[nb]], pb = perm_q[bcell[b]];
                        pf = (pa < pb) ? pa : pb;
                    }
                    const q16 cand = mul_q16(bw[nb], pf);
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
        // DEMAND (design §2.3 as generalized by v5.2):
        //   demand_k = burn_cap * I_k * o2f_j * W_hop[d_k] * w_path_k
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
            if (sd[sl] < 0) { if (dem_acc) dem_acc[slot] = 0; continue; }
            if (!flammable[i]) { if (dem_acc) dem_acc[slot] = 0; continue; }
            if (wall_hp[i] <= FUEL_FLOOR) {            // no fuel (P5.1 ember out)
                if (dem_acc) dem_acc[slot] = 0; continue; }
            const q16 ign_i = ignition_temp_q16[i];
            if (ign_i <= 0) { if (dem_acc) dem_acc[slot] = 0; continue; }
            // IGNITION vs SUSTAIN — the CPU site verbatim (combustion.cpp
            // carries the full finding): the ignition-temperature gate applies
            // only to a tile that is NOT already alight. A burning tile drops
            // below its own ignition_temp within one cool_shift step, and
            // gating oxygen on it deadlocked the whole chain once the painter
            // stopped holding the tile up there. Death stays the fire
            // logistic's job (fire_T_ext + the I_min snap).
            const bool alight = (fire[i] > 0);
            if (!alight && tsnap[i] < ign_i) {   // not alight -> the ign gate
                if (dem_acc) dem_acc[slot] = 0;
                continue;
            }
            if (!alight) {                        // I == 0 -> demands nothing
                if (dem_acc) dem_acc[slot] = 0;
                continue;
            }
            // R3 hot-burns-faster (docs/fire_3c_design_2026-09-01.md "Ruling
            // R3") — VERBATIM the CPU combustion.cpp per-claimant computation:
            // hotf_i is THIS claimant's own ramp, T source = tsnap[i] (the
            // SAME snapshot the ignition gate above just read). Folded into
            // o2f_j via ONE narrowing mul_q16 BEFORE the wide product forms
            // below — see the CPU site's comment for the overflow bound this
            // keeps (o2f_hotf <= hotf_cap*FP_ONE, so P0 <= hotf_cap*2^48).
            const q16 T_ext_i = fire_T_ext_plane ? fire_T_ext_plane[i] : fire_T_ext_q;
            const q16 hotf_i = clamp0cap_q_dev(
                recip_mul_dev(tsnap[i] - T_ext_i, recip_T_span), hotf_cap_q);
            const q16 o2f_hotf = mul_q16(o2f_j, hotf_i);
            // P-O2b: the draw weight wq = W_hop[d] * w_path for this pair.
            // FP_ONE at R == 1 (d == 1, empty path) -> every use is identity.
            const q16 wq = mul_q16(c_w_hop[(int)sd[sl]], sw[sl]);
            q16 di;
            if (dem_acc != nullptr) {
                // D1 error-feedback demand — the CPU block verbatim (int64
                // wide product, scale-2^31 remainder, whole counts drawn), with
                // the draw weight folded into the WIDE product before the
                // accumulator sees it (so a weighted claimant still draws its
                // exact share in expectation instead of being truncated away).
                const long long P0 = (long long)burn_cap_q
                                   * (long long)fire[i] * (long long)o2f_hotf;
                const long long P = apply_draw_weight_dev(P0, wq);
                const long long wide = (long long)dem_acc[slot] + (P >> 1);
                const long long draw = wide >> 31;
                dem_acc[slot] = (int32_t)(wide - (draw << 31));
                di = (q16)draw;
            } else {
                di = mul_q16(mul_q16(mul_q16(burn_cap_q, fire[i]), o2f_hotf), wq);
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
        // P-O2b: the ONLY gas write Pass A makes. The O2 is booked where it was
        // taken (the donor); the products and the heat belong at the flame and
        // are placed there by K2/K3. The CPU site verbatim.
        O2[j] = (q16)((int64_t)o2j - burn_j);

        // File each claimant's allocation on the slot buffer for Pass B, and
        // pay each claimant its P-R4 FUEL-BED deposit — the CPU site verbatim
        // (combustion.cpp Pass A): H_bed proportional to the O2 the claimant
        // ACTUALLY got, int64 shift + INT32_MAX clamp, then a positive
        // SATURATING atomic add at the claimant's own cell (several air cells
        // can feed one source, hence the atomic).
        for (int k = 0; k < n_cl; ++k) {
            alloc_slot[(size_t)cl_slot[k] * n + j] = (q16)alloc[k];
            if (heat != nullptr && alloc[k] > 0 && H_bed_m_q > 0) {
                long long bed = (long long)mul_q16((q16)alloc[k], H_bed_m_q);
                bed <<= H_bed_shift;
                if (bed > (long long)0x7fffffff) bed = (long long)0x7fffffff;
                heat_atomic_sat_add_dev(&heat[cl_src[k]], (int32_t)bed);
            }
        }
    }
}

// ---- K2: Pass B — source cells (combustion.cpp Pass B) ---------------------
// One thread per flammable source i. Single writer of wall_hp[i] AND of i's
// five deposit-site slots. Sums the incoming slot allocations and pays the
// stoichiometric fuel cost ONCE for the total, floored ONCE at FUEL_FLOOR
// ("total-then-floor-once"). Reads alloc_slot (written by K1, barriered before
// this launch). Own-cell writes only.
//
// P-O2b's second job here — WHERE the products land — is the CPU split rule
// verbatim: what a face donated directly returns to that face; the hop >= 2
// remainder splits evenly across the fire's OPEN sites (own tile + open faces),
// exact integer divide, leftover LSBs to the lowest site index.
template <int R>
__global__ void combustion_pass_b(
        int32_t* __restrict__ wall_hp,
        const bool* __restrict__ flammable,
        const bool* __restrict__ solid, const bool* __restrict__ is_vacuum,
        const int32_t* __restrict__ alloc_slot,
        int32_t* __restrict__ dep_site,
        int h, int w, int32_t fuel_per_o2_q) {
    constexpr int NSLOT = 2 * R * (R + 1);
    const int n = h * w;
    const int32_t FUEL_FLOOR = CombustionSolver::FUEL_FLOOR;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!flammable[i]) continue;
        const int y = i / w, x = i % w;
        // Donor for slot s is j = i - OFF[s]; slot s < 4 has OFF[s] == D4[s],
        // so that donor sits on i's face in direction D4_OPP[s].
        int64_t burn_i = 0;
        int64_t direct[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int s = 0; s < NSLOT; ++s) {
            const int jy = y - c_off_dy[s], jx = x - c_off_dx[s];
            if (!in_bounds(jy, jx, h, w)) continue;
            const int j = jy * w + jx;
            const int64_t a = (int64_t)alloc_slot[(size_t)s * n + j];
            burn_i += a;
            if (s < 4) direct[D4_OPP[s]] = a;
        }
        if (burn_i == 0) continue;   // this source drew no O2 this tick
        const q16 fuel_cost = narrow_round(mul_wide(fuel_per_o2_q, (q16)burn_i));
        wall_hp[i] -= fuel_cost;
        if (wall_hp[i] < FUEL_FLOOR) wall_hp[i] = FUEL_FLOOR;

        // ---- P-O2b: WHERE THE PRODUCTS LAND — the CPU site verbatim -------
        bool site_open[5];
        int  m = 0;
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            site_open[d] = false;
            const int sy = y + D4_dy[d], sx = x + D4_dx[d];
            if (!in_bounds(sy, sx, h, w)) continue;
            const int c = sy * w + sx;
            if (solid[c] || is_vacuum[c]) continue;
            site_open[d] = true; ++m;
        }
        site_open[4] = (!solid[i] && !is_vacuum[i]);   // the fire's OWN tile
        if (site_open[4]) ++m;

        const int64_t hop1 = direct[0] + direct[1] + direct[2] + direct[3];
        int64_t rem = burn_i - hop1;
        if (m == 0) rem = 0;                       // unreachable; guarded, not assumed
        const int64_t even  = (m > 0) ? rem / m : 0;
        const int64_t extra = (m > 0) ? rem - even * m : 0;
        int taken = 0;
        #pragma unroll
        for (int t = 0; t < 5; ++t) {
            if (!site_open[t]) continue;
            int64_t share = even + ((taken < (int)extra) ? 1 : 0);
            ++taken;
            if (t < 4) share += direct[t];
            dep_site[(size_t)t * n + i] = (q16)share;
        }
    }
}

// ---- K3: Pass C — air cells, THE DEPOSIT (combustion.cpp Pass C) -----------
// One thread per open cell s. Single writer of SOOT[s], N2[s], temperature[s].
// Gathers what the fires around it decided to deposit here (K2's dep_site,
// barriered before this launch) — the SECOND GATHER v5.2 asks for. The
// arithmetic is the shipped deposit verbatim, with the gathered `burn_dep` in
// place of the old in-line `burn_j`; at R == 1 they are equal cell for cell.
__global__ void combustion_pass_c(
        const int32_t* __restrict__ O2, int32_t* __restrict__ N2,
        int32_t* __restrict__ SOOT, int32_t* __restrict__ temperature,
        const bool* __restrict__ solid, const bool* __restrict__ is_vacuum,
        const int32_t* __restrict__ dep_site,
        int* __restrict__ d_heat_floor_hits, int* __restrict__ d_t_max_phys_hits,
        unsigned long long* __restrict__ d_dep_drop,
        int h, int w, int32_t soot_yield_q, int32_t H_fuel_q, int64_t recip_cv,
        int32_t n_floor_q, int32_t t_max_phys_q,
        const bool* __restrict__ thermal_solid,
        const int32_t* __restrict__ heat_inv_shift) {
    const int n = h * w;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += gridDim.x * blockDim.x) {
        if (solid[s] || is_vacuum[s]) continue;
        const int y = s / w, x = s % w;
        int64_t burn_dep = (int64_t)dep_site[(size_t)4 * n + s];
        for (int d = 0; d < 4; ++d) {
            const int iy = y + D4_dy[d], ix = x + D4_dx[d];
            if (!in_bounds(iy, ix, h, w)) continue;
            const int i = iy * w + ix;
            burn_dep += (int64_t)dep_site[(size_t)D4_OPP[d] * n + i];
        }
        if (burn_dep == 0) continue;

        const q16 soot = narrow_round(mul_wide((q16)burn_dep, soot_yield_q));
        SOOT[s] += soot;
        N2[s]   += (q16)(burn_dep - (int64_t)soot);

        // ONE aggregate heat deposit against the POST-burn N_total (delta delta).
        const q16 deposit = mul_q16((q16)burn_dep, H_fuel_q);   // burn*H_fuel
        q16 dT;
        // THERMAL-MASS AXIS, P-EOS (ruling §2 site 3) — the CPU branch verbatim:
        // an OBJECT burn site (furniture: open + gas-holding, but thermally
        // solid) converts the deposit through its own heat_inv_shift, not through
        // the thin pore gas's N. The n_floor counter is untouched on that path
        // (no gas divisor to floor) — exactly as on the CPU.
        const bool object_site = (thermal_solid != nullptr)
                              && (heat_inv_shift != nullptr)
                              && thermal_solid[s];
        if (object_site) {
            const int shift = heat_inv_shift[s];   // log2(thermal_mass), >= 0
            dT = deposit >> shift;
        } else {
            const q16 n_real_s = (q16)((int64_t)O2[s] + (int64_t)N2[s]);
            q16 n_total_s = n_real_s;
            if (n_total_s < n_floor_q) { n_total_s = n_floor_q; atomicAdd(d_heat_floor_hits, 1); }
            const q16 recip_n = reciprocal_q16_dev(n_total_s);
            if (n_total_s != n_real_s) {
                // P-E2b: CUDA twin of the CPU e_deposit_drop_sum fold — WIDE
                // throughout (int64, no premature q16 narrow; see
                // fixed_point.h's deposit_dT_wide_q16 header comment for why
                // the narrowed form overflows at n_floor_heat as low as
                // 0.01-0.001) then int64 atomicAdd (order-free on two's
                // complement), NOT the 32-bit hit-count block, since this is
                // a value SUM not a count.
                const int64_t e_over_n_wide =
                    mul_wide(deposit, recip_n) >> FP_SHIFT;   // deposit/floor
                const int64_t drop = (int64_t)deposit
                    - ((e_over_n_wide * (int64_t)n_real_s) >> FP_SHIFT);
                if (drop != 0) atomicAdd(d_dep_drop, (unsigned long long)drop);
            }
            // P-E2b: the WIDE deposit/(N*c_v) chain (int64, no premature q16
            // narrow — cuda_fixedpoint_device.cuh's deposit_dT_wide_q16_dev,
            // the device twin of fixed_point.h's deposit_dT_wide_q16). Clamp
            // to a safe non-negative int32 range BEFORE narrowing; an
            // honestly-huge deposit still hits the T_MAX_PHYS rail right
            // below, through a value that was never corrupted on the way.
            const int64_t dT_wide =
                deposit_dT_wide_q16_dev(deposit, recip_n, recip_cv);
            dT = (q16)(dT_wide < 0 ? 0 : (dT_wide > 0x7fffffffLL ? 0x7fffffffLL
                                                                  : dT_wide));
        }
        heat_saturating_add_dev(&temperature[s], dT);
        if (temperature[s] > t_max_phys_q) {                   // v2.4 rail
            temperature[s] = t_max_phys_q; atomicAdd(d_t_max_phys_hits, 1);
        }
    }
}

}  // namespace

void combustion_step(
        int32_t* gas, int n_gases,
        int o2_idx, int inert_n2_idx, int black_smoke_idx,
        int32_t* temperature, int32_t* wall_hp,
        const int32_t* fire,
        const bool* flammable, const bool* solid, const bool* is_vacuum,
        const int32_t* ignition_temp_q16,
        int h, int w, float dt, float c_v, float n_floor_heat,
        float burn_rate, float o2_thresh_burn, float H_fuel, float soot_yield,
        float fuel_per_o2, float o2_frac_ext, float o2_frac_full, float T_MAX_PHYS,
        int64_t* heat_floor_hits, int64_t* t_max_phys_hits,
        int64_t* e_deposit_drop_sum,
        const bool* thermal_solid, const int32_t* heat_inv_shift,
        int32_t* heat, float H_BED_M, int H_BED_SHIFT,
        int32_t* dem_acc,
        int draw_r, const float* dyn_permeability, int max_claimants,
        float fire_T_ext, float fire_T_span, float hotf_cap,
        const int32_t* fire_T_ext_plane) {

    // --- Guards + load-time scalar precompute (VERBATIM of combustion.cpp:65-91,
    //     in double). A guarded early-return leaves ALL fields untouched (no
    //     launch, no D2H) — matching the CPU no-op exactly. -------------------
    if (h <= 0 || w <= 0 || dt <= 0.0f) return;
    if (o2_idx < 0 || o2_idx >= n_gases) return;
    if (inert_n2_idx < 0 || inert_n2_idx >= n_gases) return;
    if (black_smoke_idx < 0 || black_smoke_idx >= n_gases) return;
    const int n = h * w;
    if (n <= 0) return;

    const q16 burn_cap_q    = quantize((double)burn_rate * (double)dt);
    const q16 o2_thresh_q   = quantize((double)o2_thresh_burn);
    const q16 soot_yield_q  = quantize((double)soot_yield);
    const q16 H_fuel_q      = quantize((double)H_fuel);
    const q16 fuel_per_o2_q = quantize((double)fuel_per_o2);
    const double c_v_safe   = (c_v > 0.0f) ? (double)c_v : 1.0;
    const int64_t recip_cv  = make_recip(c_v_safe);
    const q16 n_floor_q     = quantize((double)n_floor_heat);
    const q16 t_max_phys_q  = quantize((double)T_MAX_PHYS);
    // Continuous-O2 law (design §2.3): o2f_j span, SAME hoisted constants as
    // fire_logistic / combustion.cpp (the two laws must read identically).
    // FULL-RESPONSE REFERENCE SPLIT (2026-07-30): the span's upper end is the
    // PURE-O2 reference o2_frac_full, NOT o2_frac_amb.
    const q16 x_ext_q          = quantize((double)o2_frac_ext);
    const double x_span        = (double)o2_frac_full - (double)o2_frac_ext;
    const bool   x_degenerate  = (x_span <= 0.0);
    const int64_t recip_x_span = x_degenerate ? 0 : make_recip(x_span);
    const q16 X_N_FLOOR        = quantize(0.01);   // 655 counts (see fire_simulation.cpp)
    // P-R4: the fuel-bed mantissa, quantized on the HOST with the identical
    // fixedpoint::quantize the CPU solver uses (the load-time boundary idiom).
    const q16 H_bed_m_q        = quantize((double)H_BED_M);
    const int H_bed_shift      = (H_BED_SHIFT > 0) ? H_BED_SHIFT : 0;
    // R3 hot-burns-faster (docs/fire_3c_design_2026-09-01.md "Ruling R3"):
    // the demand-side hotf load-time bake, VERBATIM the CPU combustion.cpp.
    const q16 fire_T_ext_q     = quantize((double)fire_T_ext);
    const int64_t recip_T_span = make_recip((double)fire_T_span);
    const q16 hotf_cap_q       = quantize((double)hotf_cap);

    if (burn_cap_q <= 0) return;   // nothing burns this tick (fields untouched)

    // ---- P-O2b load-time bake — the CPU site verbatim, in double ------------
    // The SAME hard checks: a radius past the baked tables, or a dem_acc plane
    // too shallow for the radius, would alias two sources' sub-count debts onto
    // one slot. v5.2: a hit cap is a VIOLATION, not a note.
    if (draw_r < 1 || draw_r > cd::R_MAX) {
        throw std::runtime_error("cuda combustion_step: draw_r out of range");
    }
    const int n_slots = cd::slot_count(draw_r);
    if (dem_acc != nullptr && max_claimants < n_slots) {
        throw std::runtime_error("cuda combustion_step: MAX_CLAIMANTS cap hit");
    }
    // W_hop, quantized on the HOST with the identical fixedpoint::quantize the
    // CPU solver uses (the load-time boundary idiom): quantize(2/(1+d)), i.e.
    // 1/(1+d) NORMALIZED so W_hop[1] == FP_ONE — the R = 1 identity.
    int32_t w_hop_q[cd::R_MAX + 1];
    w_hop_q[0] = 0;
    for (int d = 1; d <= cd::R_MAX; ++d) w_hop_q[d] = quantize(2.0 / (1.0 + (double)d));
    // The permeability plane, quantized ONCE on the host so no float ever
    // reaches the device and both backends read the identical integers.
    std::vector<int32_t> perm_q_h;
    if (draw_r > 1 && dyn_permeability != nullptr) {
        perm_q_h.resize(n);
        for (int i = 0; i < n; ++i) perm_q_h[i] = quantize((double)dyn_permeability[i]);
    }

    // The three mutated gas planes (all distinct ids by gases.py contract).
    int32_t* O2_h   = gas + (size_t)o2_idx * n;
    int32_t* N2_h   = gas + (size_t)inert_n2_idx * n;
    int32_t* SOOT_h = gas + (size_t)black_smoke_idx * n;

    const size_t nb    = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);

    int32_t *d_O2 = nullptr, *d_N2 = nullptr, *d_SOOT = nullptr,
            *d_temp = nullptr, *d_tsnap = nullptr, *d_whp = nullptr,
            *d_ign = nullptr, *d_alloc = nullptr, *d_fire = nullptr;
    bool *d_flam = nullptr, *d_solid = nullptr, *d_vac = nullptr;
    int *d_counters = nullptr;   // [0]=heat_floor_hits, [1]=t_max_phys_hits
    // P-E2b: the combustion-floor drop energy SUM — a separate int64 slot
    // (not the 32-bit d_counters pair above, which are occurrence COUNTS
    // safe in int32; this is a VALUE sum and needs the wider type, the
    // cuda_temperature.cu `d_cnt` idiom).
    unsigned long long* d_dep_drop = nullptr;

    cuda_check(cudaMalloc(&d_O2, nb), "malloc O2");
    cuda_check(cudaMalloc(&d_N2, nb), "malloc N2");
    cuda_check(cudaMalloc(&d_SOOT, nb), "malloc SOOT");
    cuda_check(cudaMalloc(&d_temp, nb), "malloc temperature");
    cuda_check(cudaMalloc(&d_tsnap, nb), "malloc tsnap");
    cuda_check(cudaMalloc(&d_whp, nb), "malloc wall_hp");
    cuda_check(cudaMalloc(&d_fire, nb), "malloc fire");
    cuda_check(cudaMalloc(&d_ign, nb), "malloc ignition_temp");
    cuda_check(cudaMalloc(&d_alloc, (size_t)n_slots * nb), "malloc alloc_slot");
    cuda_check(cudaMalloc(&d_flam, nbool), "malloc flammable");
    cuda_check(cudaMalloc(&d_solid, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_counters, 2 * sizeof(int)), "malloc counters");
    cuda_check(cudaMalloc(&d_dep_drop, sizeof(unsigned long long)), "malloc dep_drop");

    cuda_check(cudaMemcpy(d_O2, O2_h, nb, cudaMemcpyHostToDevice), "H2D O2");
    cuda_check(cudaMemcpy(d_N2, N2_h, nb, cudaMemcpyHostToDevice), "H2D N2");
    cuda_check(cudaMemcpy(d_SOOT, SOOT_h, nb, cudaMemcpyHostToDevice), "H2D SOOT");
    cuda_check(cudaMemcpy(d_temp, temperature, nb, cudaMemcpyHostToDevice), "H2D temp");
    cuda_check(cudaMemcpy(d_whp, wall_hp, nb, cudaMemcpyHostToDevice), "H2D wall_hp");
    cuda_check(cudaMemcpy(d_fire, fire, nb, cudaMemcpyHostToDevice), "H2D fire");
    cuda_check(cudaMemcpy(d_ign, ignition_temp_q16, nb, cudaMemcpyHostToDevice), "H2D ign");
    cuda_check(cudaMemcpy(d_flam, flammable, nbool, cudaMemcpyHostToDevice), "H2D flammable");
    cuda_check(cudaMemcpy(d_solid, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D is_vacuum");

    // THERMAL-MASS AXIS, P-EOS: the object-deposit inputs. Allocated + uploaded
    // ONLY when the caller supplies BOTH (the nullable idiom this file already
    // uses for the ambient statics elsewhere) — so the legacy path costs nothing
    // and the kernel takes the byte-identical gas branch.
    bool* d_tsol = nullptr;
    int32_t* d_shift = nullptr;
    if (thermal_solid && heat_inv_shift) {
        cuda_check(cudaMalloc(&d_tsol, nbool), "malloc thermal_solid");
        cuda_check(cudaMalloc(&d_shift, nb), "malloc heat_inv_shift");
        cuda_check(cudaMemcpy(d_tsol, thermal_solid, nbool,
                              cudaMemcpyHostToDevice), "H2D thermal_solid");
        cuda_check(cudaMemcpy(d_shift, heat_inv_shift, nb,
                              cudaMemcpyHostToDevice), "H2D heat_inv_shift");
    }

    // P-R4: the `heat[]` plane rides only when the caller supplies it (the same
    // nullable idiom as the thermal-mass pair above), so the legacy path costs
    // nothing and the kernel takes the byte-identical no-H_bed branch. It is
    // IN/OUT: uploaded so the saturating atomics start from the caller's
    // existing accumulation, downloaded after the launch.
    int32_t* d_heat = nullptr;
    if (heat) {
        cuda_check(cudaMalloc(&d_heat, nb), "malloc heat");
        cuda_check(cudaMemcpy(d_heat, heat, nb, cudaMemcpyHostToDevice), "H2D heat");
    }
    // R3 hot-burns-faster: the PER-MATERIAL T_ext plane, the same nullable
    // idiom as the thermal-mass pair above — allocated + uploaded only when
    // the caller supplies one; nullptr -> the kernel takes the fire_T_ext_q
    // scalar fallback (byte-identical to the pre-R3-plane law).
    int32_t* d_T_ext_plane = nullptr;
    if (fire_T_ext_plane) {
        cuda_check(cudaMalloc(&d_T_ext_plane, nb), "malloc fire_T_ext_plane");
        cuda_check(cudaMemcpy(d_T_ext_plane, fire_T_ext_plane, nb,
                              cudaMemcpyHostToDevice), "H2D fire_T_ext_plane");
    }
    // D1: the (max_claimants, h, w) demand accumulator — SYNCED state, IN/OUT.
    // P-O2b: the plane's DECLARED depth is max_claimants; only the first
    // n_slots rows are live (a deeper plane simply carries unused rows).
    const int acc_depth = (dem_acc != nullptr) ? max_claimants : n_slots;
    int32_t* d_dem_acc = nullptr;
    if (dem_acc) {
        cuda_check(cudaMalloc(&d_dem_acc, (size_t)acc_depth * nb), "malloc dem_acc");
        cuda_check(cudaMemcpy(d_dem_acc, dem_acc, (size_t)acc_depth * nb,
                              cudaMemcpyHostToDevice), "H2D dem_acc");
    }
    // P-O2b: the DEPOSIT-SITE plane (5 slots: 4 faces + the fire's own tile).
    // Per-tick scratch — never synced, never digested. COST, priced: one
    // 5*n int32 cudaMalloc + cudaMemset + cudaFree per tick, the twin of the
    // host vector in combustion.cpp.
    int32_t* d_dep = nullptr;
    cuda_check(cudaMalloc(&d_dep, (size_t)5 * nb), "malloc dep_site");
    // P-O2b: the quantized permeability plane (only when the draw is extended).
    int32_t* d_perm = nullptr;
    if (!perm_q_h.empty()) {
        cuda_check(cudaMalloc(&d_perm, nb), "malloc perm_q");
        cuda_check(cudaMemcpy(d_perm, perm_q_h.data(), nb,
                              cudaMemcpyHostToDevice), "H2D perm_q");
    }

    // P-O2b: upload the canonical draw tables from combustion.h's SINGLE
    // definition — the CPU solver walks these very arrays, so the two backends
    // cannot drift apart by a transcription slip.
    cuda_check(cudaMemcpyToSymbol(c_off_dy, cd::OFF_DY, sizeof(cd::OFF_DY)), "sym off_dy");
    cuda_check(cudaMemcpyToSymbol(c_off_dx, cd::OFF_DX, sizeof(cd::OFF_DX)), "sym off_dx");
    cuda_check(cudaMemcpyToSymbol(c_ball_dy, cd::BALL_DY, sizeof(cd::BALL_DY)), "sym ball_dy");
    cuda_check(cudaMemcpyToSymbol(c_ball_dx, cd::BALL_DX, sizeof(cd::BALL_DX)), "sym ball_dx");
    cuda_check(cudaMemcpyToSymbol(c_ball_nbr, cd::BALL_NBR.v, sizeof(cd::BALL_NBR.v)), "sym ball_nbr");
    cuda_check(cudaMemcpyToSymbol(c_ball_slot, cd::BALL_SLOT.v, sizeof(cd::BALL_SLOT.v)), "sym ball_slot");
    cuda_check(cudaMemcpyToSymbol(c_w_hop, w_hop_q, sizeof(w_hop_q)), "sym w_hop");

    // K0: snapshot Tsnap <- temperature (device-to-device; the explicit freeze).
    cuda_check(cudaMemcpy(d_tsnap, d_temp, nb, cudaMemcpyDeviceToDevice), "D2D tsnap");
    // Slot/deposit buffers + rail counters start at zero.
    cuda_check(cudaMemset(d_alloc, 0, (size_t)n_slots * nb), "memset alloc_slot");
    cuda_check(cudaMemset(d_dep, 0, (size_t)5 * nb), "memset dep_site");
    cuda_check(cudaMemset(d_counters, 0, 2 * sizeof(int)), "memset counters");
    cuda_check(cudaMemset(d_dep_drop, 0, sizeof(unsigned long long)), "memset dep_drop");

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // K1 + K2 are TEMPLATED ON draw_r so every array bound and loop trip count
    // is a compile-time constant. One switch, three instantiations; the R the
    // config asks for is the only one that runs.
    //
    // ---- REGISTER / LOCAL-MEMORY COST (round-3 F6's named concern) ---------
    // F6 warned that the fixed-size per-thread arrays sized by MAX_CLAIMANTS
    // could spill, and pre-authorized restructuring to the offset-keyed plane
    // form if they did. MEASURED on sm_89 (ptxas -v, the shipping flags):
    //
    //   kernel      R    registers   stack frame   spill st/ld
    //   pass_a      1        40         176 B         0 / 0
    //   pass_a      2        70         512 B         0 / 0      <- SHIPPED
    //   pass_a      3       168        1040 B         0 / 0
    //   pass_b    1/2/3   34/40/40       32 B         0 / 0
    //   pass_c      -        22           0 B         0 / 0
    //
    // ZERO spill stores and loads at every radius: the indexed arrays are
    // placed in local memory (that is the stack frame) but ptxas never has to
    // evict a live register to get there, which is the cost F6 was worried
    // about. At the shipped R = 2, 70 registers allows 3 resident blocks of
    // 256 threads per SM (~50% occupancy) — ample for a pass that is a few
    // percent of the tick. The plane-form restructure is therefore NOT taken:
    // it would trade this for extra global traffic and buy nothing.
    // R = 3 (168 registers, ~17% occupancy) is the documented cost of the
    // sweep's upper point; it is not the ship value, and if P-F4b's sweep ever
    // wants R = 3 in production THAT is when the plane form earns its keep.
    #define PO2B_LAUNCH(RVAL)                                                  \
        combustion_pass_a<RVAL><<<grid, block>>>(                              \
            d_O2, d_N2, d_tsnap, d_whp, d_fire,                                \
            d_flam, d_solid, d_vac, d_ign, d_alloc, d_perm,                    \
            h, w, burn_cap_q, o2_thresh_q,                                     \
            x_ext_q, recip_x_span, x_degenerate, X_N_FLOOR,                    \
            d_heat, H_bed_m_q, H_bed_shift, d_dem_acc,                         \
            d_T_ext_plane, fire_T_ext_q, recip_T_span, hotf_cap_q);            \
        cuda_check(cudaGetLastError(), "pass_a launch");                       \
        combustion_pass_b<RVAL><<<grid, block>>>(                              \
            d_whp, d_flam, d_solid, d_vac, d_alloc, d_dep, h, w, fuel_per_o2_q); \
        cuda_check(cudaGetLastError(), "pass_b launch")

    // K1: Pass A (barriers after K0's D2D — d_tsnap is settled before any read).
    // K2: Pass B (separate launch = grid barrier: d_alloc fully written by K1,
    //     and K1's d_whp reads all complete before K2 writes d_whp).
    switch (draw_r) {
        case 1: PO2B_LAUNCH(1); break;
        case 2: PO2B_LAUNCH(2); break;
        case 3: PO2B_LAUNCH(3); break;
        default: throw std::runtime_error("cuda combustion_step: draw_r out of range");
    }
    #undef PO2B_LAUNCH

    // K3: Pass C — the re-sited deposit (separate launch = grid barrier:
    // d_dep fully written by K2 before any thread gathers it, and every K1 O2
    // debit is settled before Pass C reads O2 for the gas divisor).
    combustion_pass_c<<<grid, block>>>(
        d_O2, d_N2, d_SOOT, d_temp, d_solid, d_vac, d_dep,
        d_counters + 0, d_counters + 1, d_dep_drop,
        h, w, soot_yield_q, H_fuel_q, recip_cv, n_floor_q, t_max_phys_q,
        d_tsol, d_shift);
    cuda_check(cudaGetLastError(), "pass_c launch");

    cuda_check(cudaDeviceSynchronize(), "sync");

    // D2H the mutated fields (the three gas planes + temperature + wall_hp).
    cuda_check(cudaMemcpy(O2_h, d_O2, nb, cudaMemcpyDeviceToHost), "D2H O2");
    cuda_check(cudaMemcpy(N2_h, d_N2, nb, cudaMemcpyDeviceToHost), "D2H N2");
    cuda_check(cudaMemcpy(SOOT_h, d_SOOT, nb, cudaMemcpyDeviceToHost), "D2H SOOT");
    cuda_check(cudaMemcpy(temperature, d_temp, nb, cudaMemcpyDeviceToHost), "D2H temp");
    cuda_check(cudaMemcpy(wall_hp, d_whp, nb, cudaMemcpyDeviceToHost), "D2H wall_hp");
    if (heat) cuda_check(cudaMemcpy(heat, d_heat, nb, cudaMemcpyDeviceToHost), "D2H heat");
    if (dem_acc) cuda_check(cudaMemcpy(dem_acc, d_dem_acc, (size_t)acc_depth * nb,
                                       cudaMemcpyDeviceToHost), "D2H dem_acc");

    int counters[2] = {0, 0};
    cuda_check(cudaMemcpy(counters, d_counters, 2 * sizeof(int), cudaMemcpyDeviceToHost),
               "D2H counters");
    if (heat_floor_hits)  *heat_floor_hits  += (int64_t)counters[0];
    if (t_max_phys_hits)  *t_max_phys_hits  += (int64_t)counters[1];
    unsigned long long dep_drop = 0;
    cuda_check(cudaMemcpy(&dep_drop, d_dep_drop, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost), "D2H dep_drop");
    if (e_deposit_drop_sum) *e_deposit_drop_sum += (int64_t)dep_drop;

    cudaFree(d_O2);
    cudaFree(d_N2);
    cudaFree(d_SOOT);
    cudaFree(d_temp);
    cudaFree(d_tsnap);
    cudaFree(d_whp);
    cudaFree(d_fire);
    cudaFree(d_ign);
    cudaFree(d_alloc);
    cudaFree(d_flam);
    cudaFree(d_solid);
    cudaFree(d_vac);
    cudaFree(d_counters);
    cudaFree(d_dep_drop);
    if (d_tsol)  cudaFree(d_tsol);
    if (d_shift) cudaFree(d_shift);
    if (d_heat)  cudaFree(d_heat);
    if (d_dem_acc) cudaFree(d_dem_acc);
    if (d_dep)   cudaFree(d_dep);
    if (d_perm)  cudaFree(d_perm);
    if (d_T_ext_plane) cudaFree(d_T_ext_plane);
}

namespace {
bool g_combustion_backend_cuda = false;
}
bool combustion_backend_is_cuda() { return g_combustion_backend_cuda; }
void set_combustion_backend_cuda(bool on) { g_combustion_backend_cuda = on; }

}  // namespace breach_cuda
