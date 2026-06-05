#pragma once
// 2D raycaster — DDA ray marching for light and heat.
// Casts rays from light sources, deposits intensity into a light map.

#include <vector>
#include <cmath>
#include <cstdint>

// Falloff types for light sources
enum class Falloff : int { UNIFORM = 0, COSINE = 1, SHARP = 2 };

// ---- Fixed-point heat format (ch.04 §Fixed-point format) ----
//
// `heat` is the only sim-affecting ray output. It is stored as a Q16.16
// fixed-point int32: 16 integer bits, 16 fractional bits. One "unit" of heat
// energy == HEAT_SCALE raw int counts. The deposit is QUANTIZED into this
// domain (round-to-nearest) and added with a SATURATING add (clamp at int32
// max, never wrap) so a firestorm depositing many sources into few cells can
// never overflow past the ignition threshold (ch.04 review #6). Integer +=
// is order-independent -> deterministic / cross-machine-safe (the property
// that lets `heat` become an atomicAdd on CUDA later, ch.03 §CUDA contract).
//
// Nothing READS heat yet (this slice only DEPOSITS); the temperature pass
// (ch.04) will consume it non-destructively.
static constexpr int32_t HEAT_SCALE = 65536;   // 2^16 (Q16.16)

// Saturating quantize: float energy -> Q16.16 int32, rounded, clamped.
inline int32_t heat_quantize(float energy) {
    if (energy <= 0.0f) return 0;
    double scaled = static_cast<double>(energy) * static_cast<double>(HEAT_SCALE);
    double max_i32 = static_cast<double>(INT32_MAX);
    if (scaled >= max_i32) return INT32_MAX;
    return static_cast<int32_t>(scaled + 0.5);
}

// Saturating add into a Q16.16 accumulator: clamp at INT32_MAX, never wrap.
inline void heat_saturating_add(int32_t* cell, int32_t delta) {
    if (delta <= 0) return;
    // Overflow-safe: if adding delta would exceed INT32_MAX, clamp.
    if (*cell > INT32_MAX - delta) {
        *cell = INT32_MAX;
    } else {
        *cell += delta;
    }
}

struct LightSource {
    float x, y;              // tile coordinates
    float max_range  = 20;
    int   ray_count  = 0;    // 0 = auto-compute from range + spread
    float angle_center = 0;
    float angle_spread = 2.0f * 3.14159265f;
    float intensity  = 1.0f;
    float heat       = 0.0f;
    float jitter     = 0.0f;
    float color[3]   = {1.0f, 1.0f, 1.0f};   // RGB tint, default white
    Falloff falloff  = Falloff::UNIFORM;

    int get_ray_count() const {
        if (ray_count > 0) return ray_count;
        float full_circle = std::ceil(2.0f * 3.14159265f * max_range);
        float fraction = angle_spread / (2.0f * 3.14159265f);
        return std::max(8, static_cast<int>(std::ceil(full_circle * fraction)));
    }
};

class Raycaster {
public:
    float smoke_absorption = 0.8f;
    int   coarse_cluster   = 3;    // cluster fire sources on this grid

    // ---- Legacy API (intensity only) ----

    // Cast all fire sources and deposit into light_map.
    void update_from_fire(
        float* light_map,
        const float* fire,
        const float* smoke_field,
        const bool* is_wall,
        int h, int w
    ) const;

    // Cast a single source (for flashlights, muzzle flashes, etc.)
    void cast_source(
        const LightSource& src,
        float* light_map,
        const float* smoke_field,
        const bool* is_wall,
        int h, int w
    ) const;

    // ---- Directional API (RGB light + dominant light direction) ----
    //
    // Outputs three fields:
    //   light_rgb[i*3 + c] = accumulated RGB light arriving at tile i (c=0..2)
    //   light_dx[i]        = x component of the (unit) light direction at tile i
    //   light_dy[i]        = y component of the (unit) light direction at tile i
    //
    // light_rgb is interleaved (R,G,B per tile), shape (h, w, 3) in row-major.
    // The per-channel deposit is the scalar deposit (distance falloff) times
    // the source's color[c].
    //
    // Occlusion is PER-CHANNEL material attenuation (ch.03 §the march): the
    // per-tile `light_atten` input (h, w, 3, interleaved) is the material
    // table's static attenuation. After depositing into a tile the ray
    // attenuates each channel by (1 - mat_atten[c]) and (1 - smoke*absorb);
    // opaque tiles ([1,1,1]) drive every channel to 0 == the old wall
    // hard-stop, glass ([0.1,..]) transmits dimmed, an unequal triple tints.
    // There is NO binary wall stop and NO per-channel early-out — the ray
    // terminates on the AGGREGATE remaining energy (CUDA-divergence rule).
    //
    // light_dx/light_dy are unit-normalized after all rays are cast (vector
    // magnitude, not by intensity — see expert review notes in
    // docs/patch_level_pipeline_v1.md). At tiles where opposing rays cancel
    // (or no rays arrive), direction is (0,0) — the shader must handle that.

    // Cast a single source and accumulate RGB light + direction, plus the two
    // Slice-4 outputs:
    //   heat       : Q16.16 fixed-point int32, shape (h,w). Deposited where the
    //                source emits heat (src.heat > 0): the AGGREGATE per-tile
    //                deposit energy * src.heat, quantized, SATURATING-added.
    //                Nothing reads it this slice (ch.04). May be nullptr to skip
    //                (headless still wants it; render-only callers may pass it).
    //   smoke_glow : f32 RGB, shape (h,w,3), interleaved. God-ray glow (ch.03
    //                C16): the light each tile's SMOKE ABSORBS is deposited here
    //                per channel — energy-conserving by construction (the energy
    //                the smoke removed from the ray). May be nullptr to skip.
    //
    // Caller is responsible for zeroing the output buffers before casting the
    // frame's sources. Normalization of light_dx/dy is NOT performed here —
    // call normalize_directions() once after all sources have been cast.
    void cast_source_directional(
        const LightSource& src,
        float* light_rgb,
        float* light_dx,
        float* light_dy,
        int32_t* heat,              // Q16.16 fixed-point, (h,w) or nullptr
        float* smoke_glow,          // RGB god-ray glow, (h,w,3) or nullptr
        const float* smoke_field,
        const float* light_atten,   // per-tile static material atten (h,w,3)
        int h, int w
    ) const;

    // Normalize direction vectors in place: (dx, dy) /= length(dx, dy).
    // Tiles with zero-length direction stay (0, 0).
    static void normalize_directions(
        float* light_dx, float* light_dy,
        int h, int w
    );

private:
    void march_ray(
        float sx, float sy, float angle,
        float ray_intensity, float max_range,
        float* light_map,
        const float* smoke_field,
        const bool* is_wall,
        int h, int w
    ) const;

    void march_ray_directional(
        float sx, float sy, float angle,
        float ray_intensity, float max_range,
        const float color[3],
        float heat_emit,            // src.heat: 0 = no heat deposit on this ray
        float* light_rgb,
        float* light_dx, float* light_dy,
        int32_t* heat,              // Q16.16 fixed-point, (h,w) or nullptr
        float* smoke_glow,          // RGB god-ray glow, (h,w,3) or nullptr
        const float* smoke_field,
        const float* light_atten,   // per-tile static material atten (h,w,3)
        int h, int w
    ) const;
};
