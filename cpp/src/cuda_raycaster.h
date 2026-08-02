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
    // ---- P-R4: the net-T⁴ radiation payload (ruling A1) --------------------
    // The emitter's STATE, folded host-side exactly as `heat_emit` folded the
    // retired painter's payload: `rad_coef = a_s * angular_atten * (1/N)`.
    // `rad_src_idx < 0` == this ray carries no radiation (every non-fire
    // source), so the device march skips the exchange entirely.
    int     rad_src_idx = -1;   // emitter cell index (row*w+col) — the debit target
    int     rad_his_s   = 0;    // heat_inv_shift at the emitter (limiter budget)
    int     rad_T_q     = 0;    // emitter temperature, Q16.16
    // P-F1a (L2-B3): int64 — the E° table widened, because its int32 saturation
    // above T_game ≈ 1768 was the law's real ceiling.
    long long rad_E_s   = 0;    // E°[bucket(T_s)] (baked table lookup, host side)
    float   rad_coef    = 0.0f; // a_s * angular_atten * (1/ray_count)
    // P-F1a / v7.1 item 4 (M4): D3's rad_flux sensor keeps the LEGACY reach —
    // the emission ray marches to RADIATION_RANGE but the sensor write stays
    // behind this deterministic distance guard, so unit radiant damage is
    // unchanged and far-field bursts never ship.
    float   rad_damage_range = 0.0f;
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
//
// P-R4: the four trailing pointers carry the net-T⁴ EXCHANGE (ruling A1). All
// nullptr (the default) == exchange OFF, i.e. every legacy caller
// (cuda_raycaster_cast / _cast_batch) marches exactly as before. When supplied,
// each absorbing marched cell runs the antisymmetric pair update into
// `rad_net` — plain SIGNED int32 atomicAdd, order-free and exact, both ends
// getting the SAME integer. `e_table` is the E_TABLE_SIZE-entry black-body bake
// (uploaded with the rest of the per-call input set — see the .cu).
//
// P-F1a: the radiation arguments are GONE from this entry point. The exchange
// moved to `raycaster_cast_radiation` below (the mandatory pure-radiation fast
// path, v7 rule 4); this function is now purely the light/heat march, which is
// exactly what fire's SECOND, SHORT visible-light cast needs — unchanged, so
// the render output stays golden-neutral.
void raycaster_cast_directional(
    const RayHD* rays, int n_rays,
    float* light_rgb, float* light_dx, float* light_dy,
    int32_t* heat, float* smoke_glow,
    const float* gas_field, const float* gas_absorption, const float* gas_scatter,
    int n_gases,
    const float* light_atten, const float* heat_atten,
    float smoke_absorb_scale, float light_cull, float heat_cull,
    int h, int w);

// ---- P-F1a: THE EMISSION CAST (the pure-radiation fast path) ---------------
//
// The device twin of Raycaster::march_ray_radiation, line for line. One thread
// per emission ray; NO RGB, NO direction accumulation, NO gas optics, NO expf —
// only the integer exchange terms and the `heat_atten` survival. That is what
// makes rays >= the grid diagonal affordable.
//
// Everything here is bit-identical to the CPU march (tol 0, step AND resident):
// the same host-precomputed directions, the same PINNED float fold, the same
// single rad_quantize_signed64 boundary, the same int64 limiter with the
// halved mutual-branch cap, the same explicit distance-0 self-cell exclusion,
// the same rule-3 contact termination and the same rule-4 sky charge.
//
// `emit_mask` is the once-per-tick emitter plane (v7.1 item 13) — rule 2 keys
// on it, so both backends read the SAME bytes. `rad_amb` is the per-tile sky
// ledger: PLAIN int32 atomicAdd, deliberately not one global counter, so the
// accumulation is order-free and contention-free; the host reduces it.
//
// RETURNS: the number of rays that terminated on a CONTACT FACE (rule 3) —
// the same diagnostic Raycaster::cast_from_fire_plane returns, so the
// equivalence gate can derive its a = 0.5 tolerance identically on both
// backends instead of guessing one.
int64_t raycaster_cast_radiation(
    const RayHD* rays, int n_rays,
    const float* heat_atten,                 // a_x (h,w) — REQUIRED
    float heat_cull,
    int h, int w,
    const int64_t* e_table,                  // E_TABLE_SIZE black-body entries
    const int32_t* temperature,              // Q16.16 (h,w)
    const int32_t* heat_inv_shift,           // (h,w)
    const uint8_t* emit_mask,                // (h,w) 1 == emitter
    int32_t* rad_net,                        // Q16.16 (h,w) signed accumulator
    int32_t* rad_amb,                        // (h,w) sky ledger (rule 4)
    // D3: the RADIANT-FLUX SENSOR plane — positive-only, AIR cells only, NOT
    // part of the energy ledger (no temperature effect, nothing debited); its
    // only consumer is unit heat damage. Saturating atomic == the old heat
    // contract, so it is order-free. See raycaster.h RadCtx::rad_flux.
    int32_t* rad_flux = nullptr);

// Backend flag (mirrors the S1 temperature backend switch).
bool raycaster_backend_is_cuda();
void set_raycaster_backend_cuda(bool on);

}  // namespace breach_cuda
