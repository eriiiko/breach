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

__device__ __forceinline__ bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

// Integer clamp to [0, FP_ONE] (the [0,1] o2f saturation) — VERBATIM device
// port of combustion.cpp's file-local clamp01_q (mirrors cuda_fire.cu's
// clamp01_q_dev so the two O2 laws are bit-identical).
__device__ __forceinline__ q16 clamp01_q_dev(q16 v) {
    if (v < 0) return 0;
    if (v > FP_ONE) return FP_ONE;
    return v;
}

// ---- K1: Pass A — air cells (combustion.cpp:115-228) -----------------------
// One thread per cell j. Single writer of O2[j], SOOT[j], N2[j], temperature[j],
// and the four face buffers at index j. Reads d_tsnap (the snapshot gate), the
// pass-entry O2 at the OWN cell (read-before-write => every claimant sees
// pass-entry O2 — deltas beta/gamma), and pre-payment wall_hp (K2 writes it,
// barriered after this kernel — so this read is pre-payment, matching the CPU).
__global__ void combustion_pass_a(
        int32_t* __restrict__ O2, int32_t* __restrict__ N2,
        int32_t* __restrict__ SOOT, int32_t* __restrict__ temperature,
        const int32_t* __restrict__ tsnap, const int32_t* __restrict__ wall_hp,
        const int32_t* __restrict__ fire,
        const bool* __restrict__ flammable, const bool* __restrict__ solid,
        const bool* __restrict__ is_vacuum,
        const int32_t* __restrict__ ignition_temp_q16,
        int32_t* __restrict__ alloc_face,
        int* __restrict__ d_heat_floor_hits, int* __restrict__ d_t_max_phys_hits,
        int h, int w, int32_t burn_cap_q, int32_t o2_thresh_q,
        int32_t soot_yield_q, int32_t H_fuel_q, int64_t recip_cv,
        int32_t n_floor_q, int32_t t_max_phys_q,
        int32_t x_ext_q, int64_t recip_x_span, bool x_degenerate,
        int32_t X_N_FLOOR) {
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

        // Gather the <=4 flammable claimant sources + each one's per-claimant
        // DEMAND (design §2.3): demand_k = burn_cap * I_k * o2f_j (PINNED
        // left-fold mul_q16, truncating). I_k = fire[i].
        int cl_dir[4];   // D4 index of the claimant (face key for alloc_face)
        int cl_src[4];   // global cell index of the claimant source
        int64_t dem[4];  // per-claimant O2 demand this tick (Q16.16 counts)
        int n_cl = 0;
        for (int d = 0; d < 4; ++d) {
            const int iy = y + D4_dy[d], ix = x + D4_dx[d];
            if (!in_bounds(iy, ix, h, w)) continue;
            const int i = iy * w + ix;
            if (!flammable[i]) continue;
            if (wall_hp[i] <= FUEL_FLOOR) continue;   // no fuel (P5.1 ember out)
            const q16 ign_i = ignition_temp_q16[i];
            if (ign_i <= 0) continue;                 // material can't ignite
            if (tsnap[i] < ign_i) continue;           // below ignition (snapshot!)
            const q16 di = mul_q16(mul_q16(burn_cap_q, fire[i]), o2f_j);
            cl_dir[n_cl] = d;
            cl_src[n_cl] = i;
            dem[n_cl] = (int64_t)di;
            ++n_cl;
        }
        if (n_cl == 0) continue;

        // --- Allocate O2[j] across the claimants (design §3 step 2) ---------
        int64_t alloc[4];
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
            int64_t keys[4];
            int64_t sum_alloc = 0;
            for (int k = 0; k < n_cl; ++k) {
                const int64_t num = (int64_t)o2j * dem[k];  // < 2^43
                alloc[k] = num / D;      // floor, exact integer divide
                keys[k]  = num % D;      // integer remainder = tiebreak key
                sum_alloc += alloc[k];
            }
            // R leftover LSBs (in [0, n_cl) subset of [0,4)) go to the R
            // claimants with the largest key; ties -> lowest source index.
            int64_t R = (int64_t)o2j - sum_alloc;
            bool chosen[4] = {false, false, false, false};
            for (int r = 0; r < (int)R; ++r) {
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

        // --- Single-writer gas + heat writes at cell j (design §3 step 3) ---
        O2[j] = (q16)((int64_t)o2j - burn_j);
        const q16 soot = narrow_round(mul_wide((q16)burn_j, soot_yield_q));
        SOOT[j] += soot;
        N2[j]   += (q16)(burn_j - (int64_t)soot);

        // ONE aggregate heat deposit against the POST-burn N_total (delta delta).
        q16 n_total_j = (q16)((int64_t)O2[j] + (int64_t)N2[j]);
        if (n_total_j < n_floor_q) { n_total_j = n_floor_q; atomicAdd(d_heat_floor_hits, 1); }
        const q16 recip_n  = reciprocal_q16_dev(n_total_j);
        const q16 deposit  = mul_q16((q16)burn_j, H_fuel_q);   // burn*H_fuel
        const q16 e_over_n = mul_q16(deposit, recip_n);        // .../N
        const q16 dT       = recip_mul_dev(e_over_n, recip_cv);// .../c_v
        heat_saturating_add_dev(&temperature[j], dT);
        if (temperature[j] > t_max_phys_q) {                   // v2.4 rail
            temperature[j] = t_max_phys_q; atomicAdd(d_t_max_phys_hits, 1);
        }

        // File each claimant's allocation on the face buffer for Pass B.
        for (int k = 0; k < n_cl; ++k) {
            alloc_face[(size_t)cl_dir[k] * n + j] = (q16)alloc[k];
        }
    }
}

// ---- K2: Pass B — source cells (combustion.cpp:240-261) --------------------
// One thread per flammable source i. Single writer of wall_hp[i]. Sums the <=4
// incoming face allocations and pays the stoichiometric fuel cost ONCE for the
// total, floored ONCE at FUEL_FLOOR ("total-then-floor-once"). Reads alloc_face
// (written by K1, barriered before this launch). Own-cell wall_hp write only.
__global__ void combustion_pass_b(
        int32_t* __restrict__ wall_hp,
        const bool* __restrict__ flammable,
        const int32_t* __restrict__ alloc_face,
        int h, int w, int32_t fuel_per_o2_q) {
    const int n = h * w;
    const int32_t FUEL_FLOOR = CombustionSolver::FUEL_FLOOR;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!flammable[i]) continue;
        const int y = i / w, x = i % w;
        int64_t burn_i = 0;
        for (int d = 0; d < 4; ++d) {
            const int jy = y + D4_dy[d], jx = x + D4_dx[d];
            if (!in_bounds(jy, jx, h, w)) continue;
            const int j = jy * w + jx;
            // Air neighbour j filed its allocation toward THIS source under j's
            // outbound direction to i, which is D4_OPP[d].
            burn_i += (int64_t)alloc_face[(size_t)D4_OPP[d] * n + j];
        }
        if (burn_i == 0) continue;   // this source drew no O2 this tick
        const q16 fuel_cost = narrow_round(mul_wide(fuel_per_o2_q, (q16)burn_i));
        wall_hp[i] -= fuel_cost;
        if (wall_hp[i] < FUEL_FLOOR) wall_hp[i] = FUEL_FLOOR;
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
        float fuel_per_o2, float o2_frac_ext, float o2_frac_amb, float T_MAX_PHYS,
        int64_t* heat_floor_hits, int64_t* t_max_phys_hits) {

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
    const q16 x_ext_q          = quantize((double)o2_frac_ext);
    const double x_span        = (double)o2_frac_amb - (double)o2_frac_ext;
    const bool   x_degenerate  = (x_span <= 0.0);
    const int64_t recip_x_span = x_degenerate ? 0 : make_recip(x_span);
    const q16 X_N_FLOOR        = quantize(0.01);   // 655 counts (see fire_simulation.cpp)

    if (burn_cap_q <= 0) return;   // nothing burns this tick (fields untouched)

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

    cuda_check(cudaMalloc(&d_O2, nb), "malloc O2");
    cuda_check(cudaMalloc(&d_N2, nb), "malloc N2");
    cuda_check(cudaMalloc(&d_SOOT, nb), "malloc SOOT");
    cuda_check(cudaMalloc(&d_temp, nb), "malloc temperature");
    cuda_check(cudaMalloc(&d_tsnap, nb), "malloc tsnap");
    cuda_check(cudaMalloc(&d_whp, nb), "malloc wall_hp");
    cuda_check(cudaMalloc(&d_fire, nb), "malloc fire");
    cuda_check(cudaMalloc(&d_ign, nb), "malloc ignition_temp");
    cuda_check(cudaMalloc(&d_alloc, (size_t)4 * nb), "malloc alloc_face");
    cuda_check(cudaMalloc(&d_flam, nbool), "malloc flammable");
    cuda_check(cudaMalloc(&d_solid, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");
    cuda_check(cudaMalloc(&d_counters, 2 * sizeof(int)), "malloc counters");

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

    // K0: snapshot Tsnap <- temperature (device-to-device; the explicit freeze).
    cuda_check(cudaMemcpy(d_tsnap, d_temp, nb, cudaMemcpyDeviceToDevice), "D2D tsnap");
    // Face buffers + rail counters start at zero.
    cuda_check(cudaMemset(d_alloc, 0, (size_t)4 * nb), "memset alloc_face");
    cuda_check(cudaMemset(d_counters, 0, 2 * sizeof(int)), "memset counters");

    const int block = 256;
    const int grid = (n + block - 1) / block;

    // K1: Pass A (barriers after K0's D2D — d_tsnap is settled before any read).
    combustion_pass_a<<<grid, block>>>(
        d_O2, d_N2, d_SOOT, d_temp, d_tsnap, d_whp, d_fire,
        d_flam, d_solid, d_vac, d_ign, d_alloc,
        d_counters + 0, d_counters + 1,
        h, w, burn_cap_q, o2_thresh_q, soot_yield_q, H_fuel_q, recip_cv,
        n_floor_q, t_max_phys_q,
        x_ext_q, recip_x_span, x_degenerate, X_N_FLOOR);
    cuda_check(cudaGetLastError(), "pass_a launch");

    // K2: Pass B (separate launch = grid barrier: d_alloc fully written by K1,
    // d_whp reads in K1 all completed before this kernel writes d_whp).
    combustion_pass_b<<<grid, block>>>(
        d_whp, d_flam, d_alloc, h, w, fuel_per_o2_q);
    cuda_check(cudaGetLastError(), "pass_b launch");

    cuda_check(cudaDeviceSynchronize(), "sync");

    // D2H the mutated fields (the three gas planes + temperature + wall_hp).
    cuda_check(cudaMemcpy(O2_h, d_O2, nb, cudaMemcpyDeviceToHost), "D2H O2");
    cuda_check(cudaMemcpy(N2_h, d_N2, nb, cudaMemcpyDeviceToHost), "D2H N2");
    cuda_check(cudaMemcpy(SOOT_h, d_SOOT, nb, cudaMemcpyDeviceToHost), "D2H SOOT");
    cuda_check(cudaMemcpy(temperature, d_temp, nb, cudaMemcpyDeviceToHost), "D2H temp");
    cuda_check(cudaMemcpy(wall_hp, d_whp, nb, cudaMemcpyDeviceToHost), "D2H wall_hp");

    int counters[2] = {0, 0};
    cuda_check(cudaMemcpy(counters, d_counters, 2 * sizeof(int), cudaMemcpyDeviceToHost),
               "D2H counters");
    if (heat_floor_hits)  *heat_floor_hits  += (int64_t)counters[0];
    if (t_max_phys_hits)  *t_max_phys_hits  += (int64_t)counters[1];

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
}

namespace {
bool g_combustion_backend_cuda = false;
}
bool combustion_backend_is_cuda() { return g_combustion_backend_cuda; }
void set_combustion_backend_cuda(bool on) { g_combustion_backend_cuda = on; }

}  // namespace breach_cuda
