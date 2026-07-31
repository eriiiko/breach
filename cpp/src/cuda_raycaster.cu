// ============================================================================
// CUDA-S2 raycaster implementation — see cuda_raycaster.h.
// A faithful GPU port of Raycaster::march_ray_directional (raycaster.cpp).
// One thread per ray. Heat is bit-identical to the CPU /fp:strict march; render
// channels are deterministic-exempt.
// ============================================================================
#include "cuda_raycaster.h"
#include "raycaster.h"   // P-R4: e_bucket_of / rad_pair_budget / rad_quantize_signed
                         // / RAD_LIM_SHIFT — ONE definition of the exchange's
                         // boundaries, shared with the CPU march.

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in raycaster_cast_directional/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// MUST match raycaster.h::HEAT_SCALE / heat_quantize EXACTLY (the pinned
// truncation path, engine/08 §Determinism): double promote, *HEAT_SCALE, +0.5,
// truncate toward zero, clamp at INT32_MAX. NOT rintf / round-half-to-even.
__device__ __forceinline__ int32_t heat_quantize_dev(float energy) {
    if (energy <= 0.0f) return 0;
    double scaled = (double)energy * (double)65536;          // HEAT_SCALE = 2^16
    double max_i32 = (double)0x7fffffff;
    if (scaled >= max_i32) return 0x7fffffff;
    return (int32_t)(scaled + 0.5);
}

// SATURATING integer atomic add (mirrors heat_saturating_add). CAS loop: clamp at
// INT32_MAX, never wrap. Order-free for non-negative deltas (the property that
// makes scatter deterministic despite atomic ordering).
__device__ __forceinline__ void heat_atomic_sat_add(int32_t* addr, int32_t delta) {
    if (delta <= 0) return;
    int32_t old = *addr, assumed;
    do {
        assumed = old;
        int32_t sum = (assumed > 0x7fffffff - delta) ? 0x7fffffff : (assumed + delta);
        old = atomicCAS(addr, assumed, sum);
    } while (assumed != old);
}

// One thread per ray. Replicates march_ray_directional tile-for-tile.
//
// P-R4 radiation (ruling A1): the exchange is scattered with a PLAIN signed
// atomicAdd(int*) — integer addition is associative + commutative and CUDA's
// int atomicAdd wraps on overflow exactly as the CPU's `rad_signed_add` does,
// so the accumulation is ORDER-FREE and bit-identical to the CPU reference even
// in the (documented, out-of-band) overflow regime. A SATURATING signed atomic
// would NOT be order-free — that is why rad_net has its own plane and its own
// contract, separate from `heat[]`'s positive-saturating one (ruling A1.7).
__global__ void march_rays_kernel(
    const RayHD* __restrict__ rays, int n_rays,
    float* light_rgb, float* light_dx, float* light_dy,
    int32_t* heat, float* smoke_glow,
    const float* __restrict__ gas_field, const float* __restrict__ gas_absorption,
    const float* __restrict__ gas_scatter, int n_gases,
    const float* __restrict__ light_atten, const float* __restrict__ heat_atten,
    float smoke_absorb_scale, float light_cull, float heat_cull,
    int h, int w,
    const int32_t* __restrict__ e_table, const int32_t* __restrict__ temperature,
    const int32_t* __restrict__ heat_inv_shift, int32_t* rad_net,
    int32_t* rad_flux) {
    const int plane = h * w;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rays;
         r += gridDim.x * blockDim.x) {
        const RayHD ray = rays[r];
        const float dx = ray.dx, dy = ray.dy;
        const int step_x = (dx >= 0.0f) ? 1 : -1;
        const int step_y = (dy >= 0.0f) ? 1 : -1;
        const float dt_dx = (fabsf(dx) > 1e-8f) ? fabsf(1.0f / dx) : 1e8f;
        const float dt_dy = (fabsf(dy) > 1e-8f) ? fabsf(1.0f / dy) : 1e8f;
        float t_max_x = 0.5f * dt_dx;
        float t_max_y = 0.5f * dt_dy;
        int x = (int)ray.sx;
        int y = (int)ray.sy;

        const float e_r = ray.e_r, e_g = ray.e_g, e_b = ray.e_b;
        const bool emits_r = e_r > 0.0f, emits_g = e_g > 0.0f, emits_b = e_b > 0.0f;
        const bool emits_heat = (heat != nullptr) && (ray.heat_emit > 0.0f);
        const float heat_emit = ray.heat_emit;
        const float max_range = ray.max_range;
        // P-R4: the radiation channel — VERBATIM twin of the CPU `emits_rad`.
        const bool emits_rad = (rad_net != nullptr) && (e_table != nullptr) &&
                               (temperature != nullptr) && (heat_inv_shift != nullptr) &&
                               (heat_atten != nullptr) && (ray.rad_src_idx >= 0) &&
                               (ray.rad_coef != 0.0f);

        float sr = 1.0f, sg = 1.0f, sb = 1.0f;   // per-channel survival
        float heat_survival = 1.0f;
        float distance = 0.0f;

        for (;;) {
            const bool alive =
                (emits_r && sr > light_cull) || (emits_g && sg > light_cull) ||
                (emits_b && sb > light_cull) ||
                ((emits_heat || emits_rad) && heat_survival > heat_cull);
            if (!alive) break;
            if (x < 0 || x >= w || y < 0 || y >= h) break;
            const int idx = y * w + x;

            const float dep_r = e_r * sr;
            const float dep_g = e_g * sg;
            const float dep_b = e_b * sb;
            if (light_rgb != nullptr) {
                atomicAdd(&light_rgb[idx * 3 + 0], dep_r);
                atomicAdd(&light_rgb[idx * 3 + 1], dep_g);
                atomicAdd(&light_rgb[idx * 3 + 2], dep_b);
            }
            const float dep_agg = dep_r + dep_g + dep_b;
            if (light_dx != nullptr) atomicAdd(&light_dx[idx], dep_agg * (-dx));
            if (light_dy != nullptr) atomicAdd(&light_dy[idx], dep_agg * (-dy));

            // Heat: gated on heat_survival > heat_cull. Saturating integer atomic.
            if (emits_heat && heat_survival > heat_cull) {
                const float heat_dep = heat_emit * heat_survival;
                heat_atomic_sat_add(&heat[idx], heat_quantize_dev(heat_dep));
            }

            // ---- P-R4 NET-T⁴ EXCHANGE — the CPU block, line for line -------
            // Same gate (heat_survival > heat_cull), same absorber test
            // (a_r > 0), the SAME PINNED left fold
            //   f = rad_coef -> *= tau -> *= a_r -> (double)f * (double)diff
            // and the same single rad_quantize_signed boundary, the same
            // int64 limiter, the same antisymmetric ± apply. Nothing here may
            // be reordered "for the GPU": this is the tol-0 contract.
            if (emits_rad && heat_survival > heat_cull) {
                const float a_r = heat_atten[idx];
                // D3: the RADIANT-FLUX SENSOR at AIR cells — the CPU block
                // verbatim. NOT part of the energy ledger (no transport, no
                // temperature, nothing debited); unit heat damage is its only
                // consumer. Positive-only -> the SATURATING atomic, which is
                // order-free exactly as the retired painter's deposit was.
                if (rad_flux != nullptr && !(a_r > 0.0f)) {
                    float ff = ray.rad_coef;   // a_s · w
                    ff *= heat_survival;       // · τ
                    const int32_t q =
                        rad_quantize_signed((double)ff * (double)ray.rad_E_s);
                    if (q > 0) heat_atomic_sat_add(&rad_flux[idx], q);
                }
                if (a_r > 0.0f) {
                    const int32_t T_r = temperature[idx];
                    const int32_t diff = ray.rad_E_s - e_table[e_bucket_of(T_r)];
                    float f = ray.rad_coef;   // a_s · w
                    f *= heat_survival;       // · τ
                    f *= a_r;                 // · a_r
                    int32_t net = rad_quantize_signed((double)f * (double)diff);
                    const long long dT = (long long)ray.rad_T_q - (long long)T_r;
                    const long long adT = (dT < 0) ? -dT : dT;
                    const long long b_s = rad_pair_budget(adT, ray.rad_his_s);
                    const long long b_r = rad_pair_budget(adT, (int)heat_inv_shift[idx]);
                    const long long cap = (b_s < b_r) ? b_s : b_r;
                    if ((long long)net >  cap) net = (int32_t)cap;
                    if ((long long)net < -cap) net = (int32_t)(-cap);
                    atomicAdd(&rad_net[idx], net);                 // receiver gains
                    atomicAdd(&rad_net[ray.rad_src_idx], -net);    // emitter loses
                }
            }

            // Per-channel material occlusion decays survival.
            sr *= (1.0f - light_atten[idx * 3 + 0]);
            sg *= (1.0f - light_atten[idx * 3 + 1]);
            sb *= (1.0f - light_atten[idx * 3 + 2]);
            // Heat self-occlusion, with the source-tile skip (distance>0).
            if (heat_atten != nullptr && distance > 0.0f) {
                heat_survival *= (1.0f - heat_atten[idx]);
            }

            // Multi-gas optics (RGB survival only; heat never sees the exp).
            float tau_r = 0.0f, tau_g = 0.0f, tau_b = 0.0f;
            float sca_r = 0.0f, sca_g = 0.0f, sca_b = 0.0f;
            for (int g = 0; g < n_gases; ++g) {
                const float gd = gas_field[g * plane + idx];
                if (gd <= 0.001f) continue;
                const float* ab = &gas_absorption[g * 3];
                const float* sc = &gas_scatter[g * 3];
                tau_r += gd * ab[0]; tau_g += gd * ab[1]; tau_b += gd * ab[2];
                sca_r += gd * sc[0]; sca_g += gd * sc[1]; sca_b += gd * sc[2];
            }
            if (tau_r > 0.0f || tau_g > 0.0f || tau_b > 0.0f ||
                sca_r > 0.0f || sca_g > 0.0f || sca_b > 0.0f) {
                if (smoke_glow != nullptr) {
                    atomicAdd(&smoke_glow[idx * 3 + 0], dep_r * sca_r);
                    atomicAdd(&smoke_glow[idx * 3 + 1], dep_g * sca_g);
                    atomicAdd(&smoke_glow[idx * 3 + 2], dep_b * sca_b);
                }
                // expf is a transcendental (NOT bit-identical CPU↔GPU) but it lives
                // on RGB survival only — render-exempt. It can extend the march into
                // tiles where heat is already dead (deposits zero heat), so the
                // heat-touched set is unchanged (engine/08 §Determinism).
                sr *= expf(-smoke_absorb_scale * tau_r);
                sg *= expf(-smoke_absorb_scale * tau_g);
                sb *= expf(-smoke_absorb_scale * tau_b);
            }

            if (t_max_x < t_max_y) { x += step_x; distance = t_max_x; t_max_x += dt_dx; }
            else                   { y += step_y; distance = t_max_y; t_max_y += dt_dy; }
            if (distance > max_range) break;
        }
    }
}

// Small RAII-ish upload helper for an optional host array -> device pointer.
template <typename T>
T* upload_opt(const T* host, size_t count, const char* what) {
    if (host == nullptr) return nullptr;
    T* d = nullptr;
    cuda_check(cudaMalloc(&d, count * sizeof(T)), what);
    cuda_check(cudaMemcpy(d, host, count * sizeof(T), cudaMemcpyHostToDevice), what);
    return d;
}

}  // namespace

void raycaster_cast_directional(
    const RayHD* rays, int n_rays,
    float* light_rgb, float* light_dx, float* light_dy,
    int32_t* heat, float* smoke_glow,
    const float* gas_field, const float* gas_absorption, const float* gas_scatter,
    int n_gases,
    const float* light_atten, const float* heat_atten,
    float smoke_absorb_scale, float light_cull, float heat_cull,
    int h, int w,
    const int32_t* e_table, const int32_t* temperature,
    const int32_t* heat_inv_shift, int32_t* rad_net, int32_t* rad_flux) {
    const size_t n = (size_t)h * (size_t)w;
    if (n == 0 || n_rays <= 0) return;

    // --- upload inputs ---
    RayHD* d_rays = upload_opt(rays, (size_t)n_rays, "malloc rays");
    const float* d_gas = upload_opt(gas_field, n * (size_t)(n_gases > 0 ? n_gases : 0), "malloc gas");
    const float* d_gabs = upload_opt(gas_absorption, (size_t)(n_gases > 0 ? n_gases : 0) * 3, "malloc gabs");
    const float* d_gsca = upload_opt(gas_scatter, (size_t)(n_gases > 0 ? n_gases : 0) * 3, "malloc gsca");
    const float* d_atten = upload_opt(light_atten, n * 3, "malloc light_atten");
    const float* d_hatten = upload_opt(heat_atten, n, "malloc heat_atten");

    // --- output buffers: upload the caller's (pre-zeroed) contents so the atomic
    //     accumulation starts from the same baseline as the CPU cast ---
    float* d_lrgb = upload_opt(light_rgb, n * 3, "malloc light_rgb");
    float* d_ldx = upload_opt(light_dx, n, "malloc light_dx");
    float* d_ldy = upload_opt(light_dy, n, "malloc light_dy");
    int32_t* d_heat = upload_opt(heat, n, "malloc heat");
    float* d_glow = upload_opt(smoke_glow, n * 3, "malloc smoke_glow");

    // ---- P-R4 radiation inputs -------------------------------------------
    // The E° bake rides the per-call input set like every other table this
    // entry point uploads (16 KB — ~2 us next to the (h,w) planes already
    // moving). A one-shot __constant__ upload at bake time is a pure
    // optimisation and is deliberately NOT taken here: a cached device copy
    // would need a staleness protocol against `rad_scale`, and the measured
    // cost (gate g) leaves no reason to buy that risk.
    const int32_t* d_etab = upload_opt(e_table, (size_t)E_TABLE_SIZE, "malloc e_table");
    const int32_t* d_temp = upload_opt(temperature, n, "malloc temperature");
    const int32_t* d_his  = upload_opt(heat_inv_shift, n, "malloc heat_inv_shift");
    // rad_net is IN/OUT: uploaded (the caller's pre-existing accumulation) so
    // the atomics start from the same baseline the CPU cast would, exactly as
    // `heat` does above.
    int32_t* d_radnet = upload_opt(rad_net, n, "malloc rad_net");
    // D3: same IN/OUT treatment — uploaded so the saturating atomics start from
    // the caller's accumulation, downloaded after the launch.
    int32_t* d_radflux = upload_opt(rad_flux, n, "malloc rad_flux");

    const int block = 256;
    const int grid = (n_rays + block - 1) / block;
    march_rays_kernel<<<grid, block>>>(
        d_rays, n_rays, d_lrgb, d_ldx, d_ldy, d_heat, d_glow,
        d_gas, d_gabs, d_gsca, n_gases, d_atten, d_hatten,
        smoke_absorb_scale, light_cull, heat_cull, h, w,
        d_etab, d_temp, d_his, d_radnet, d_radflux);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    // --- download outputs ---
    if (light_rgb)  cuda_check(cudaMemcpy(light_rgb, d_lrgb, n * 3 * sizeof(float), cudaMemcpyDeviceToHost), "D2H light_rgb");
    if (light_dx)   cuda_check(cudaMemcpy(light_dx, d_ldx, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H light_dx");
    if (light_dy)   cuda_check(cudaMemcpy(light_dy, d_ldy, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H light_dy");
    if (heat)       cuda_check(cudaMemcpy(heat, d_heat, n * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H heat");
    if (smoke_glow) cuda_check(cudaMemcpy(smoke_glow, d_glow, n * 3 * sizeof(float), cudaMemcpyDeviceToHost), "D2H smoke_glow");
    if (rad_net)    cuda_check(cudaMemcpy(rad_net, d_radnet, n * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H rad_net");
    if (rad_flux)   cuda_check(cudaMemcpy(rad_flux, d_radflux, n * sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H rad_flux");

    cudaFree(d_rays);
    cudaFree((void*)d_gas); cudaFree((void*)d_gabs); cudaFree((void*)d_gsca);
    cudaFree((void*)d_atten); cudaFree((void*)d_hatten);
    cudaFree(d_lrgb); cudaFree(d_ldx); cudaFree(d_ldy);
    cudaFree(d_heat); cudaFree(d_glow);
    cudaFree((void*)d_etab); cudaFree((void*)d_temp); cudaFree((void*)d_his);
    cudaFree(d_radnet); cudaFree(d_radflux);
}

namespace {
bool g_ray_backend_cuda = false;
}
bool raycaster_backend_is_cuda() { return g_ray_backend_cuda; }
void set_raycaster_backend_cuda(bool on) { g_ray_backend_cuda = on; }

}  // namespace breach_cuda
