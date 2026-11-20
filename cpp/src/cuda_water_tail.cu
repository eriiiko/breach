// ============================================================================
// #29 P4a — THE WATER TAIL ON THE DEVICE (cuda_resident.h's
// water_tail_launch_resident). The device twin of PhysicsEngine::step_water_
// tail (physics_engine.cpp — THE oracle, read it line for line beside this):
//
//   W5  the flash-boil vacuum sink: where atmosphere < boil_p_thresh and there
//       is water, min(depth, boil_amount) leaves the depth and its steam puff
//       (steam_yield * the dequantized boil, a float multiply, then the trace
//       currency's quantize_trace) lands on the steam plane through the
//       counted saturation, booked in the mint / clip / negative-hit books.
//   W3  the occupancy-transition evacuation: a cell whose free air column
//       shrank (ratio = free_before / free_after > 1, not solid) sheds
//       (1 - 1/ratio) of every gas plane into its open 4-neighbours,
//       permeability-weighted, the last neighbour taking the integer
//       remainder; a bulk share carries the donor's T_abs into an accountable
//       receiver's gas_energy or is exported (e_water_evac_export_sum); a
//       cell whose air column is flooded (free_after <= flood_eps) has its
//       dyn_permeability sealed to 0 in its own iteration, BEFORE it would
//       evacuate; then before := water_depth.
//
// THE ORDER (the one hard part): the CPU's W3 loop is row-major and SERIAL. A
// cell reads (a) its own gas / gas_energy, which an earlier neighbour's
// evacuation may already have filled, and (b) its neighbours' permeability,
// which an earlier neighbour's own iteration may already have sealed. The
// device keeps that order exactly:
//   K_flags   per cell: the ratio, the seal flag and the evacuation fraction —
//             pure functions of before / water_depth (post-W5) / solid;
//   K_evac    ONE BLOCK PER ENV: an ordered compaction (block prefix sums,
//             chunk by chunk) of the cells that can evacuate, in row-major
//             order, then thread 0 walks them serially: the CPU loop
//             restricted to the cells that move gas. A neighbour j's
//             permeability as cell i sees it is `(j <= i && seal[j]) ? 0 :
//             perm[j]` — sealed iff its iteration came first (the cell's own
//             seal precedes its evacuation in the same iteration);
//   K_finish  per cell: the seals and the before := water_depth copy.
// A cell that is not a candidate moves no gas on the CPU either (ratio <= 1
// or solid skips the whole block), so the walk sees every write the CPU's
// loop makes, in the CPU's order.
//
// FLOAT BRIDGES: every float / double operation is the CPU's, in the CPU's
// precision, under this TU's --fmad=false, -prec-div=true and -ftz=false (the
// target-wide CUDA flags, CMakeLists.txt): the dequantized depths
// `(float)((double)q / 65536)`, the f32 differences / max / divide / clip,
// `quantize(1.0 - 1.0 / (double)ratio)` and `quantize((double)(w / wsum))`
// in double. std::max / std::min are spelled as their definitions
// ((a < b) ? b : a, (b < a) ? b : a), so NaN-free equal-value tie breaks
// match. The integer arithmetic is the shared kit (fixed_point.h, FP_HD):
// quantize, mul_q16, floordiv_q, quantize_trace, trace_add_sat.
// ============================================================================
#include "cuda_resident.h"
#include "cuda_batch.cuh"   // #29 P5: the tick stream, the per-env mask
#include "fixed_point.h"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace breach_cuda {

namespace {

using namespace fixedpoint;

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in water_tail/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

constexpr uint8_t F_CAND = 1;   // ratio > 1 && !solid (the CPU's evac block gate)
constexpr uint8_t F_SEAL = 2;   // free_after <= flood_eps (the flooded seal)

__device__ __forceinline__ float fmax_std(float a, float b) { return (a < b) ? b : a; }
__device__ __forceinline__ float fmin_std(float a, float b) { return (b < a) ? b : a; }

// The accountable set the evacuation's energy rides (the CPU's lambda).
__device__ __forceinline__ bool evac_accountable(
        int i, const bool* solid, const bool* ts, const bool* vac, const bool* amb) {
    return !solid[i]
        && !(ts != nullptr && ts[i])
        && !(vac != nullptr && vac[i])
        && !(amb != nullptr && amb[i]);
}

__global__ void wt_prep(unsigned long long* __restrict__ cnt, int total_slots) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < total_slots;
         k += gridDim.x * blockDim.x) {
        cnt[k] = 0ULL;
    }
}

// ---- W5: the flash-boil vacuum sink, per cell (each cell owns its own depth
// and its own steam cell; the books are order-free integer sums). The CPU's
// `boiling.any()` guard only skips a loop whose per-cell test is this one.
__global__ void wt_boil(int32_t* __restrict__ water_depth,
                        const int32_t* __restrict__ atmosphere,
                        int32_t* __restrict__ gas, int n_gases, int steam_idx,
                        WaterTailScalars sc,
                        unsigned long long* __restrict__ cnt_all,
                        int n, int n_env, const uint8_t* __restrict__ active) {
    const int total = n_env * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        if (active && !active[i / n]) continue;   // #29 P5: a dormant env does nothing
        if (!(atmosphere[i] < sc.boil_p_thresh_q && water_depth[i] > 0)) continue;
        const int env = i / n;
        const int li = i - env * n;
        unsigned long long* cnt = cnt_all + (size_t)env * WT_CNT_SLOTS;
        const q16 depth = water_depth[i];
        const q16 boiled_q = (depth < sc.boil_amount_q) ? depth : sc.boil_amount_q;
        water_depth[i] = depth - boiled_q;                         // exact (>= 0)
        const float boiled_m = (float)((double)boiled_q / (double)FP_ONE);
        const int64_t puff_q = quantize_trace((double)(sc.steam_yield_f * boiled_m));
        int32_t* steam = gas + ((size_t)env * n_gases + steam_idx) * n;
        int64_t clip = 0, neg = 0;
        steam[li] = trace_add_sat(steam[li], puff_q, &clip, &neg);
        if (puff_q != 0) atomicAdd(&cnt[WT_MINT], (unsigned long long)puff_q);
        if (clip != 0) atomicAdd(&cnt[WT_CLIP + steam_idx], (unsigned long long)clip);
        if (neg != 0) atomicAdd(&cnt[WT_NEG + steam_idx], (unsigned long long)neg);
        atomicAdd(&cnt[WT_N_BOIL], 1ULL);
    }
}

// ---- W3a: the per-cell ratio, seal flag and evacuation fraction. -----------
__global__ void wt_flags(const int32_t* __restrict__ water_depth,
                         const int32_t* __restrict__ before,
                         const bool* __restrict__ solid,
                         WaterTailScalars sc,
                         uint8_t* __restrict__ flag, int32_t* __restrict__ frac,
                         int total) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        const float before_m = (float)((double)before[i] / (double)FP_ONE);
        const float depth_m  = (float)((double)water_depth[i] / (double)FP_ONE);
        const float free_before = fmax_std(sc.ceiling_h_f - before_m, sc.flood_eps_f);
        const float free_after  = fmax_std(sc.ceiling_h_f - depth_m,  sc.flood_eps_f);
        float ratio = free_before / free_after;
        ratio = fmin_std(fmax_std(ratio, sc.clip_lo_f), sc.clip_hi_f);
        uint8_t f = 0;
        if (free_after <= sc.flood_eps_f) f |= F_SEAL;
        int32_t fq = 0;
        if (ratio > 1.0f && !solid[i]) {
            f |= F_CAND;
            fq = quantize(1.0 - 1.0 / (double)ratio);
        }
        flag[i] = f;
        frac[i] = fq;
    }
}

// ---- W3b: one block per env — the ordered compaction, then the serial walk.
constexpr int EVAC_BLOCK = 256;

__global__ void __launch_bounds__(EVAC_BLOCK)
wt_evac(int h, int w, int n_gases,
        int32_t* __restrict__ gas,
        const float* __restrict__ perm,
        const bool* __restrict__ solid,
        int64_t* __restrict__ gas_energy, const bool* __restrict__ cons,
        const bool* __restrict__ ts, const bool* __restrict__ vac,
        const bool* __restrict__ amb,
        const int32_t* __restrict__ d_t_amb_raw,
        const uint8_t* __restrict__ flag, const int32_t* __restrict__ frac,
        int32_t* __restrict__ list,
        unsigned long long* __restrict__ cnt_all,
        const uint8_t* __restrict__ active) {
    const int env = blockIdx.x;
    if (active && !active[env]) return;   // #29 P5: block-uniform, before any barrier
    const int n = h * w;
    const int base = env * n;
    const int tid = threadIdx.x;
    __shared__ int s_scan[2][EVAC_BLOCK];
    __shared__ int s_count;
    if (tid == 0) s_count = 0;
    __syncthreads();
    int32_t* lst = list + base;
    // Ordered compaction, chunk by chunk: an inclusive Hillis-Steele scan of
    // the candidate flags gives each candidate its slot; chunks are appended
    // in order, so the list is the candidates in ascending (row-major) index.
    for (int c0 = 0; c0 < n; c0 += EVAC_BLOCK) {
        const int li = c0 + tid;
        const int f = (li < n && (flag[base + li] & F_CAND)) ? 1 : 0;
        int cur = 0;
        s_scan[0][tid] = f;
        __syncthreads();
        for (int off = 1; off < EVAC_BLOCK; off <<= 1) {
            const int v = s_scan[cur][tid] + ((tid >= off) ? s_scan[cur][tid - off] : 0);
            s_scan[cur ^ 1][tid] = v;
            cur ^= 1;
            __syncthreads();
        }
        if (f) lst[s_count + s_scan[cur][tid] - 1] = li;
        __syncthreads();
        if (tid == 0) s_count += s_scan[cur][EVAC_BLOCK - 1];
        __syncthreads();
    }
    if (tid != 0) return;

    // ---- the serial walk (the CPU's W3 loop body, candidates only) --------
    unsigned long long* cnt = cnt_all + (size_t)env * WT_CNT_SLOTS;
    const bool energy_mode = (gas_energy != nullptr && cons != nullptr);
    const int32_t t_amb_raw = d_t_amb_raw[env];
    const int count = s_count;
    const int cand_dy[4] = {-1, 1, 0, 0};
    const int cand_dx[4] = {0, 0, -1, 1};
    for (int k = 0; k < count; ++k) {
        const int li = lst[k];
        const int gi = base + li;
        const int y = li / w, x = li % w;
        // The permeability as this iteration sees it: a cell whose own
        // iteration came first (index <= li) is sealed if it floods.
        const float perm_i = (flag[gi] & F_SEAL) ? 0.0f : perm[gi];
        int nbs[4]; float wts[4]; int cntn = 0; float wsum = 0.0f;
        for (int d = 0; d < 4; ++d) {
            const int ny = y + cand_dy[d], nx = x + cand_dx[d];
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
            const int nl = ny * w + nx;
            const int ng = base + nl;
            if (solid[ng]) continue;
            const float perm_n = (nl <= li && (flag[ng] & F_SEAL)) ? 0.0f : perm[ng];
            const float wgt = fmin_std(perm_i, perm_n);      // std::min(perm_i, perm_n)
            if (wgt <= 0.0f) continue;
            nbs[cntn] = nl; wts[cntn] = wgt; wsum += wgt; ++cntn;
        }
        if (cntn == 0) continue;
        atomicAdd(&cnt[WT_N_EVAC], 1ULL);
        const q16 frac_q = frac[gi];
        int64_t n_bulk_src = 0, t_abs_src = 0;
        const bool src_acct = energy_mode && evac_accountable(gi, solid, ts, vac, amb);
        if (src_acct) {
            for (int g = 0; g < n_gases; ++g)
                if (cons[g])
                    n_bulk_src += (int64_t)gas[((size_t)env * n_gases + g) * n + li];
            t_abs_src = (n_bulk_src >= 1)
                ? floordiv_q(gas_energy[gi], n_bulk_src)
                : (int64_t)t_amb_raw;
        }
        for (int g = 0; g < n_gases; ++g) {
            int32_t* plane = gas + ((size_t)env * n_gases + g) * n;
            const q16 evac = mul_q16(plane[li], frac_q);
            if (evac <= 0) continue;
            const bool carries = src_acct && cons[g];
            q16 distributed = 0;
            for (int kk = 0; kk < cntn; ++kk) {
                const q16 share = (kk == cntn - 1)
                    ? (q16)(evac - distributed)
                    : mul_q16(evac, quantize((double)(wts[kk] / wsum)));
                const int nl = nbs[kk];
                if (cons != nullptr && !cons[g]) {
                    int64_t clip = 0, neg = 0;
                    plane[nl] = trace_add_sat(plane[nl], (int64_t)share, &clip, &neg);
                    if (clip != 0) cnt[WT_CLIP + g] += (unsigned long long)clip;
                    if (neg != 0) cnt[WT_NEG + g] += (unsigned long long)neg;
                } else {
                    plane[nl] += share;
                }
                distributed += share;
                if (carries && share != 0) {
                    const int64_t de = (int64_t)share * t_abs_src;
                    gas_energy[gi] -= de;
                    if (evac_accountable(base + nl, solid, ts, vac, amb))
                        gas_energy[base + nl] += de;
                    else
                        cnt[WT_EXPORT] += (unsigned long long)de;
                }
            }
            plane[li] -= distributed;
        }
    }
}

// ---- W3c: the seals and the copy, per cell. --------------------------------
__global__ void wt_finish(const int32_t* __restrict__ water_depth,
                          int32_t* __restrict__ before, float* __restrict__ perm,
                          const uint8_t* __restrict__ flag, int total,
                          int n, const uint8_t* __restrict__ active) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        if (active && !active[i / n]) continue;
        if (flag[i] & F_SEAL) perm[i] = 0.0f;    // flooded -> seal
        before[i] = water_depth[i];              // the copyto (integer)
    }
}

}  // namespace

WaterTailScalars water_tail_bake_scalars(
        float sim_time, double ceiling_h, double flood_eps, double ratio_cap,
        double boil_rate, double boil_p_thresh, double steam_yield) {
    // The CPU tail's host folds, cast for cast (physics_engine.cpp).
    WaterTailScalars sc;
    sc.boil_p_thresh_q = quantize((double)boil_p_thresh);
    const float boil_amount_f = (float)((double)boil_rate * (double)sim_time);
    sc.boil_amount_q = quantize((double)boil_amount_f);
    sc.steam_yield_f = (float)steam_yield;
    sc.ceiling_h_f = (float)ceiling_h;
    sc.flood_eps_f = (float)flood_eps;
    sc.clip_lo_f   = (float)(1.0 / ratio_cap);
    sc.clip_hi_f   = (float)ratio_cap;
    return sc;
}

int water_tail_launch_resident(
        int n_env, int h, int w, int n_gases, int steam_idx,
        int32_t* d_water_depth, const int32_t* d_atmosphere, const bool* d_solid,
        int32_t* d_gas, int32_t* d_before, float* d_perm,
        int64_t* d_gas_energy, const bool* d_cons,
        const bool* d_thermal_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
        const int32_t* d_t_amb_raw, const WaterTailScalars& sc,
        uint8_t* d_flag, int32_t* d_frac, int32_t* d_list,
        unsigned long long* d_cnt, const uint8_t* d_active) {
    const int n = h * w;
    if (n_env <= 0 || n <= 0) return 0;
    if (n_gases < 1 || n_gases > WATER_TAIL_GAS_MAX) {
        throw std::invalid_argument(
            "water_tail_launch_resident: n_gases outside [1, "
            + std::to_string(WATER_TAIL_GAS_MAX) + "] (the books' per-gas slots)");
    }
    if (steam_idx < 0 || steam_idx >= n_gases) {
        throw std::invalid_argument("water_tail_launch_resident: steam_idx outside the gas block");
    }
    const int total = n_env * n;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    const int slots = n_env * WT_CNT_SLOTS;
    cudaStream_t st = tstream();
    wt_prep<<<(slots + block - 1) / block, block, 0, st>>>(d_cnt, slots);
    cuda_check(cudaGetLastError(), "prep launch");
    wt_boil<<<grid, block, 0, st>>>(d_water_depth, d_atmosphere, d_gas, n_gases, steam_idx,
                             sc, d_cnt, n, n_env, d_active);
    cuda_check(cudaGetLastError(), "boil launch");
    wt_flags<<<grid, block, 0, st>>>(d_water_depth, d_before, d_solid, sc, d_flag, d_frac, total);
    cuda_check(cudaGetLastError(), "flags launch");
    wt_evac<<<n_env, EVAC_BLOCK, 0, st>>>(h, w, n_gases, d_gas, d_perm, d_solid,
                                   d_gas_energy, d_cons, d_thermal_solid,
                                   d_is_vacuum, d_is_ambient, d_t_amb_raw,
                                   d_flag, d_frac, d_list, d_cnt, d_active);
    cuda_check(cudaGetLastError(), "evac launch");
    wt_finish<<<grid, block, 0, st>>>(d_water_depth, d_before, d_perm, d_flag, total,
                                      n, d_active);
    cuda_check(cudaGetLastError(), "finish launch");
    return 5;
}

}  // namespace breach_cuda
