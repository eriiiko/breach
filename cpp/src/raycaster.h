#pragma once
// 2D raycaster — DDA ray marching for light and heat.
// Casts rays from light sources, deposits intensity into a light map.

#include <vector>
#include <cmath>
#include <cstdint>

// Falloff types for light sources
enum class Falloff : int { UNIFORM = 0, COSINE = 1, SHARP = 2 };

// CUDA-S2 gate: the host-precomputed ray POD lives in cuda_raycaster.h (a plain
// header, no CUDA symbols). Forward-declare it so build_ray_list can return a
// vector of them without dragging the CUDA header into every CPU TU that
// includes raycaster.h.
namespace breach_cuda { struct RayHD; }

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
    // cast_source path which has no per-channel notion).
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

    // ---- Legacy API (intensity only) ----
    //
    // (P-R1, 2026-07-31: update_from_fire + its coarse_cluster dial were
    // deleted here — no production caller, legacy intensity-only signature,
    // an RNG jitter land-mine, and clustering is incompatible with the
    // planned radiation law (a merged pseudo-source has no well-defined T_s).
    // See docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.2.)

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

    // ---- P-R1: whole-fire-plane cast (source build moved into C++) ----
    //
    // docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.1-A4.2.
    // Replaces PhysicsRunner.cast_fire_heat's old per-tile Python loop
    // (one bp.LightSource() + ~10 pybind attribute writes PER BURNING TILE,
    // PER TICK): enumerates every burning tile (fire[i] > 0, Q16.16) in
    // ROW-MAJOR order via build_fire_sources() and casts each resulting
    // source immediately with cast_source_directional — i.e. the SAME
    // per-source CPU cast the old Python loop drove, just built natively.
    // `heat` is byte-identical to that old loop (same sources, same march,
    // same order-free saturating add) — this is a mechanical relocation,
    // no law/behavior change (P-R1's byte-identity gate).
    //
    // `fire` is the Q16.16 int32 fire plane (h, w); the dial parameters
    // mirror the old Python-side runner attributes EXACTLY (see
    // build_fire_sources for the float-parity contract on why they are
    // `double`, not `float`). `jitter` stays fixed at 0.0 by the caller
    // (fire heat is sim-affecting — no dither; S8c item 1's RNG-coupling
    // guard).
    void cast_from_fire_plane(
        const int32_t* fire, int h, int w,
        double k_fire_heat, int fire_ray_count,
        double range_base, double range_per_intensity,
        double intensity_base, double intensity_per_intensity,
        const float color[3],
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
        double jitter = 0.0
    ) const;

    // ---- CUDA-S2 gate: host ray-list builder (shared CPU/GPU angle math) ----
    //
    // Replicates cast_source_directional's per-ray loop EXACTLY — same
    // get_ray_count(), the same (i+0.5)/N angle sweep, the same jitter RNG
    // (mt19937 seeded (unsigned)(src.x*1000+src.y), uniform_real(-1,1)*jitter),
    // the same falloff angular_atten, the same inv_n normalisation — and folds
    // angle->(cos,sin), angular_atten, color and /N into each RayHD's
    // (dx,dy,e_r,e_g,e_b,heat_emit). Rays with angular_atten<=0 are SKIPPED, just
    // as the CPU cast skips them. Because this runs in THIS /fp:strict TU (the one
    // that already owns the identical angle math in cast_source_directional), the
    // GPU march's host-precomputed dx=cos(angle)/dy=sin(angle) are bit-identical
    // to what the CPU march_ray_directional computes internally from `angle` —
    // which is the contract that makes the DDA tile path (hence heat) match.
    std::vector<breach_cuda::RayHD> build_ray_list(const LightSource& src) const;

    // ---- P-R1: CUDA twin of cast_from_fire_plane ----
    //
    // The SAME enumeration + per-source parameter construction as
    // cast_from_fire_plane (build_fire_sources, shared — the float-parity-
    // critical code path runs exactly once for both backends), folded into
    // RayHD via build_ray_list and concatenated in row-major source order —
    // IDENTICAL to how cuda_raycaster_cast_batch (bindings.cpp, S8c item 1)
    // concatenates a Python-supplied source list, except the source list is
    // now built FROM THE FIRE PLANE here instead of supplied by Python. The
    // caller feeds the result straight into the existing
    // breach_cuda::raycaster_cast_directional batched march — no new device
    // code, no march/law change.
    std::vector<breach_cuda::RayHD> build_fire_ray_list(
        const int32_t* fire, int h, int w,
        double k_fire_heat, int fire_ray_count,
        double range_base, double range_per_intensity,
        double intensity_base, double intensity_per_intensity,
        const float color[3],
        double jitter = 0.0
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

    // ---- P-R1: the shared fire-plane source enumerator ----
    //
    // Enumerates fire[row*w+col] > 0 in ROW-MAJOR order (row outer, col
    // inner — the same order np.nonzero(fire > 0) yielded to the old Python
    // loop) and builds ONE LightSource per burning tile, reproducing
    // PhysicsRunner.cast_fire_heat's old per-tile Python math EXACTLY:
    //   x = col + 0.5, y = row + 0.5
    //   max_range  = range_base + range_per_intensity * I
    //   angle_center = ((col*7 + row*13) % ray_count) * (2*pi/ray_count)
    //   intensity  = intensity_base + intensity_per_intensity * I
    //   heat       = k_fire_heat * I
    //   jitter = 0, angle_spread = 2*pi (omni), ray_count = fire_ray_count
    // where I = float(fire_q) / 65536 (Q16.16 dequant, fire_fixed.FP_ONE).
    //
    // FLOAT-PARITY CONTRACT (the reason this function exists rather than
    // just inlining floats): the old Python loop computed every expression
    // above in DOUBLE (Python floats are C doubles; math.pi is a double)
    // and narrowed to float32 ONLY at the final `src.field = <python
    // float>` pybind attribute set. To land the identical float32 bits,
    // every expression here is DOUBLE arithmetic, with the same operator
    // shapes/order as the Python source, cast to `float` ONLY at the point
    // that mirrors that pybind narrowing. The dial parameters are `double`
    // in this signature for the same reason: the Python runner's attributes
    // (self.k_fire_heat etc.) stay double for their whole lifetime and are
    // never pre-narrowed to float32 before this multiply — accepting them
    // as `float` here would round a tick early and could flip the final
    // float32 bit. `color` has no arithmetic before it lands on the source
    // (a pure passthrough constant either way), so it is safely `float`.
    std::vector<LightSource> build_fire_sources(
        const int32_t* fire, int h, int w,
        double k_fire_heat, int fire_ray_count,
        double range_base, double range_per_intensity,
        double intensity_base, double intensity_per_intensity,
        const float color[3], double jitter
    ) const;
};
