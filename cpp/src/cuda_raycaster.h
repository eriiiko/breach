#pragma once
// ============================================================================
// CUDA-S2 raycaster — GPU port of Raycaster::march_ray_directional.
// One thread per ray; the 1/r falloff is ray DENSITY (engine/08 §Falloff is
// density). HEAT is the gameplay output and must be bit-identical CPU↔GPU:
//   - ray DIRECTIONS are host-precomputed (no device cos/sin -> the DDA tile
//     path matches bit-for-bit);
//   - the deposit math is correctly-rounded IEEE +,-,*,/ (--fmad=false) with the
//     float->Q16.16 quantize in DOUBLE (promote, *HEAT_SCALE, +0.5, truncate),
//     replicating heat_quantize() exactly;
//   - heat scatters with a SATURATING integer atomic (order-free: saturating add
//     of non-negative deltas under a monotone clamp is associative+commutative).
// The RGB render channels (light_rgb/dir/smoke_glow) scatter with float atomics
// and use device expf for the gas optics — they are EXEMPT from the bit-identity
// gate (render-only). The gas exp can never perturb the heat-touched tile set:
// heat deposits are gated on heat_survival>heat_cull (material-only), and a ray
// kept alive by RGB past heat's death deposits zero heat (engine/08 §Determinism).
// ============================================================================
#include <cstdint>

namespace breach_cuda {

// One precomputed ray. The host (cast_source_directional analogue) folds the
// per-source angular_atten and the /N normalisation into these scalars, and the
// cos/sin into (dx,dy), so the device march is pure arithmetic. POD -> uploadable.
struct RayHD {
    float sx, sy;          // source tile coordinates
    float dx, dy;          // host-precomputed cos/sin(angle) (unit travel dir)
    float e_r, e_g, e_b;   // per-channel emitted energy = (P/N)*angular_atten*color[c]
    float heat_emit;       // per-ray heat budget = (heat/N)*angular_atten
    float max_range;
};

// GPU directional cast. Marches all `n_rays` rays into the output fields, which
// MUST be pre-zeroed by the caller (the deposits are additive, exactly as the CPU
// cast accumulates into zeroed frame buffers). All pointers are HOST pointers;
// this entry uploads, launches one-thread-per-ray, and downloads (per-call malloc,
// the S1 pattern). `light_dx`/`light_dy`/`smoke_glow` may be nullptr to skip;
// `heat`/`heat_atten` may be nullptr to skip the heat channel.
//
// Bit-identical HEAT to Raycaster::march_ray_directional (CPU /fp:strict). The
// render channels are deterministic-exempt (float-atomic order + expf).
void raycaster_cast_directional(
    const RayHD* rays, int n_rays,
    float* light_rgb, float* light_dx, float* light_dy,
    int32_t* heat, float* smoke_glow,
    const float* gas_field, const float* gas_absorption, const float* gas_scatter,
    int n_gases,
    const float* light_atten, const float* heat_atten,
    float smoke_absorb_scale, float light_cull, float heat_cull,
    int h, int w);

// Backend flag (mirrors the S1 temperature backend switch).
bool raycaster_backend_is_cuda();
void set_raycaster_backend_cuda(bool on);

}  // namespace breach_cuda
