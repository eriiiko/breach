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
    // ---- Smoke optics (ch.05 §6.1 §6 — decoupled per-channel absorption vs glow) ----
    //
    // Two INDEPENDENT per-channel budgets, NOT constrained to absorb + glow = 1:
    //
    //   1. Per-channel transmission (Beer-Lambert):
    //        tau_c   = smoke_absorption[c] * smoke_density * smoke_absorb_scale
    //        trans_c = exp(-tau_c)               // never reaches 0 -> beam survives deep smoke
    //        survival[c] *= trans_c              // (engine/08: occlusion decays SURVIVAL)
    //      smoke_absorb_scale is the global "beam reach" dial: LOW = long beam
    //      (flashlight travels far through smoke and still glows), HIGH = beam
    //      dies fast. Per-channel absorption is the (future) gas COLOUR.
    //
    //   2. Separate additive scatter/glow (god-rays, smoke_glow buffer):
    //        smoke_glow[c] += deposited_light[c] * smoke_scatter_albedo[c] * smoke_density
    //      This is the light the smoke SCATTERS BACK toward the viewer. It is
    //      independent of (and may exceed) absorption -> "barely absorbs, glows
    //      brightly" gases (steam) are expressible.
    //
    // Legacy scalar `smoke_absorption` (kept for the scalar march_ray /
    // update_from_fire path which has no per-channel notion).
    float smoke_absorption = 0.8f;

    // SUPERSEDED (engine/05 §6.2, M2): the single-field per-channel coefficients.
    // The directional march now reads the per-GAS `absorption`/`scatter_albedo`
    // tables passed per-cast (from GasTable), summed density-weighted across the
    // (N,h,w) gas array — these two struct members no longer drive the directional
    // look. They are kept INERT (still bound) so any non-gas caller compiles; the
    // active dial for the gas path is `smoke_absorb_scale` below.
    float smoke_absorption_rgb[3] = {1.0f, 1.0f, 1.0f};
    float smoke_scatter_albedo[3] = {1.0f, 1.0f, 1.0f};
    // Global beam-reach dial (STILL ACTIVE for the gas path): scales the summed
    // per-gas tau. LOW = long beam (flashlight travels far). Default 1.4.
    float smoke_absorb_scale = 1.4f;

    // ---- Propagation-model cull thresholds (engine/08 §The march, §Falloff is
    // density) — the per-channel SURVIVAL floors of the pure-density model ----
    //
    // A ray carries fixed per-channel energy and a survival ∈ [0,1] that decays
    // ONLY by occlusion (never by distance — the 1/r falloff is ray DENSITY, not
    // a per-ray multiplier). A channel keeps depositing while its own survival is
    // above its threshold; the ray terminates when EVERY emitting channel is
    // below its threshold (or max_range). Because survival decays only by
    // occlusion, in open air it stays 1.0 and the ray runs to max_range — the
    // cull only bites BEHIND occluders (≈99% absorbed at 0.01).
    //   light_cull : ε_rgb — the RGB render channels' floor.
    //   heat_cull  : ε_heat — the heat (gameplay/damage) channel's floor; its
    //                own dial so a heat-shield/low-E-glass material can diverge
    //                from light. Heat deposits gate on heat_survival > heat_cull,
    //                which is what DECOUPLES heat from the float light path
    //                (engine/08 §Determinism: heat is decoupled from light).
    float light_cull = 0.01f;
    float heat_cull  = 0.01f;

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
    // PURE-DENSITY model (engine/08 §Falloff is density): each ray carries fixed
    // energy = total_power / N (the cast divides by ray_count), and a per-channel
    // SURVIVAL ∈[0,1] that starts at 1 and decays ONLY by occlusion. The deposit
    // is energy·color[c]·survival[c] — there is NO per-ray distance falloff; the
    // 1/r intensity law emerges from ray density (N cancels).
    //
    // Occlusion is PER-CHANNEL material attenuation (ch.03 §the march): the
    // per-tile `light_atten` input (h, w, 3, interleaved) is the material
    // table's static attenuation. After depositing into a tile each channel's
    // survival is multiplied by (1 - mat_atten[c]) then the gas transmission;
    // opaque tiles ([1,1,1]) drive every channel to 0 == the old wall
    // hard-stop, glass ([0.1,..]) transmits dimmed, an unequal triple tints.
    // There is NO binary wall stop. Each channel STOPS DEPOSITING once its own
    // survival drops below its cull floor (light_cull for RGB, heat_cull for
    // heat); the ray marches to the AGGREGATE range — it continues while ANY
    // channel survives (no per-channel early-out, CUDA-divergence rule).
    //
    // light_dx/light_dy are unit-normalized after all rays are cast (vector
    // magnitude, not by intensity — see expert review notes in
    // docs/patch_level_pipeline_v1.md). At tiles where opposing rays cancel
    // (or no rays arrive), direction is (0,0) — the shader must handle that.

    // ---- Multi-gas optics (engine/05 §6.2 — coloured N-gas summation) ----
    //
    // The directional march generalises the single `smoke` scalar to N gas
    // density fields (gmap.gas, shape (N,h,w)), each with its OWN per-channel
    // `absorption` (N,3) and `scatter_albedo` (N,3) row from GasTable. Per tile,
    // per channel c, the two decoupled budgets above are SUMMED density-weighted
    // across all gases (engine/05 §6.2 — "mixing falls out of the sum"):
    //
    //   transmission:  tau_c = smoke_absorb_scale * Σ_g ( gas[g][tile] * absorption[g][c] )
    //                  trans_c = exp(-tau_c);  survival[c] *= trans_c
    //   scatter/glow:  smoke_glow[c] += dep_c * Σ_g ( gas[g][tile] * scatter_albedo[g][c] )
    //
    // `smoke_absorb_scale` stays the global beam-reach dial. A single populated
    // gas reproduces exactly what the old single-`smoke` path did for that gas's
    // coefficients (with absorption/scatter = that gas's row). Heat is untouched
    // (smoke/gas does not attenuate the heat channel).

    // Cast a single source and accumulate RGB light + direction, plus the two
    // Slice-4 outputs:
    //   heat       : Q16.16 fixed-point int32, shape (h,w). Deposited where the
    //                source emits heat (src.heat > 0). Heat is the INDEPENDENT
    //                4th ray channel (engine/06 §1): it carries its OWN scalar
    //                survival, attenuated per tile by `heat_atten` exactly as
    //                each RGB channel is attenuated by `light_atten[c]`. The
    //                deposit is (src.heat/N) * heat_survival (NO distance
    //                falloff), GATED on heat_survival > heat_cull, quantized +
    //                SATURATING-added — independent of the RGB survival, so a
    //                heat-shield (light-clear, heat-opaque) blocks heat while
    //                passing light, and smoked glass (light-opaque, heat-clear)
    //                does the converse. May be nullptr to skip.
    //   smoke_glow : f32 RGB, shape (h,w,3), interleaved. God-ray glow (ch.03
    //                C16): the light each tile's SMOKE ABSORBS is deposited here
    //                per channel — energy-conserving by construction (the energy
    //                the smoke removed from the ray). May be nullptr to skip.
    //
    // heat_atten : per-tile scalar heat-ray attenuation (h,w), the heat analogue
    //              of light_atten (air 0, walls 1.0, glass 0.3). May be nullptr,
    //              in which case heat is NOT attenuated (the pre-S6 behaviour:
    //              heat survival stays 1.0 the whole march).
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
        const float* gas_field,     // (n_gases, h, w) contiguous gas densities
        const float* gas_absorption,// (n_gases, 3) per-gas per-channel absorption
        const float* gas_scatter,   // (n_gases, 3) per-gas per-channel scatter
        int n_gases,
        const float* light_atten,   // per-tile static material atten (h,w,3)
        const float* heat_atten,    // per-tile heat atten (h,w) or nullptr
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
        const float* gas_field,     // (n_gases, h, w) contiguous gas densities
        const float* gas_absorption,// (n_gases, 3) per-gas per-channel absorption
        const float* gas_scatter,   // (n_gases, 3) per-gas per-channel scatter
        int n_gases,
        const float* light_atten,   // per-tile static material atten (h,w,3)
        const float* heat_atten,    // per-tile heat atten (h,w) or nullptr
        int h, int w
    ) const;
};
