#include "raycaster.h"
#include "cuda_raycaster.h"   // CUDA-S2 gate: RayHD POD (plain header, no CUDA symbols)
#include "fixed_point.h"      // Q2-LIFT: the deterministic trig kit (sin/cos_q16)
#include <algorithm>
#include <random>

static constexpr float PI = 3.14159265358979f;

// ---- Q2-LIFT: deterministic ray trig --------------------------------------
// std::cos/std::sin are CRT transcendentals: /fp:strict pins the float
// ARITHMETIC in this TU but NOT the libm library code, which may change at
// the last ULP across compiler/CRT versions — the last latent cross-machine
// hazard on the heat path (ray dirs decide which tiles the DDA visits, and
// heat is a SYNCED int32 field). Route every angle through the pure-integer
// kit instead: quantize the float angle to Q16.16 radians (round-to-nearest,
// the locked boundary idiom — double math, deterministic under /fp:strict),
// integer cos/sin, dequantize to float. The result is an exact n/65536
// float, bit-identical on every machine; direction shift vs libm <= ~1.5e-5
// (the kit's pinned 9e-6 bound + the input quantization) — pre-approved, no
// feel-check (docs/q2_lift_spec.md Patch 3). Angles here are within one wrap
// (|a| < 2pi + jitter), inside the kit's pinned |a| <= 4pi accuracy range.
// dx can now be EXACTLY 0.0f at the quantized axes — the existing
// `|dx| > 1e-8 ? 1/dx : 1e8` DDA guards already handle that.
// build_ray_list (the GPU's dir source) and the CPU march share these same
// helpers, so both backends walk identical DDA tiles (the S2/S2b contract).
static inline float det_cos(float angle) {
    return fixedpoint::dequantize_f(fixedpoint::cos_q16(fixedpoint::quantize(angle)));
}
static inline float det_sin(float angle) {
    return fixedpoint::dequantize_f(fixedpoint::sin_q16(fixedpoint::quantize(angle)));
}

// ---- P-R4: the E° bake (ruling A1.3) --------------------------------------
//
// Credit: J.R. Howell, M.P. Mengüç, R. Siegel, "Thermal Radiation Heat
// Transfer" — E°(T) = σT⁴ is the black-body emissive power this table holds;
// the emissivity ε is NOT in the bake (it lives in the per-material pair
// coefficient a_s·a_r, Kirchhoff), which is what keeps the pair coefficient
// symmetric and the exchange antisymmetric.
//
// Every entry is EXACT integer work up to ONE final boundary multiply:
//   K(t)  = 297 + 8t                       (the bucket midpoint's absolute K)
//   k2    = K*K;  k4 = k2*k2                (int64, exact — see raycaster.h)
//   E[t]  = clamp_int32( round( rad_scale * k4 ) )
// NEVER pow()/libm: a 1-ULP CRT difference here would desync a synced int32
// field across machines.
void Raycaster::bake_emissive_table() const {
    e_table_.resize(E_TABLE_SIZE);
    const double scale = rad_scale;
    for (int t = 0; t < E_TABLE_SIZE; ++t) {
        const int64_t K  = 297LL + 8LL * (int64_t)t;   // 293 + 2*(4t+2), exact
        const int64_t k2 = K * K;                       // repeated multiplication
        const int64_t k4 = k2 * k2;                     // <= 1.09e18, exact in int64
        const double  v  = (double)k4 * scale;          // the ONE rounding boundary
        e_table_[t] = (v >= 2147483647.0)
            ? INT32_MAX
            : (int32_t)((v > 0.0) ? (v + 0.5) : 0.0);   // rad_scale > 0 by contract
    }
    e_table_scale_ = scale;
}

const int32_t* Raycaster::emissive_table() const {
    if (e_table_.size() != (size_t)E_TABLE_SIZE || e_table_scale_ != rad_scale) {
        bake_emissive_table();
    }
    return e_table_.data();
}

void Raycaster::march_ray(
    float sx, float sy, float angle,
    float ray_intensity, float max_range,
    float* light_map,
    const float* smoke_field,
    const bool* is_wall,
    int h, int w
) const {
    float dx = det_cos(angle);   // Q2-LIFT: integer-kit trig (see det_cos above)
    float dy = det_sin(angle);

    int step_x = (dx >= 0) ? 1 : -1;
    int step_y = (dy >= 0) ? 1 : -1;
    float dt_dx = (std::abs(dx) > 1e-8f) ? std::abs(1.0f / dx) : 1e8f;
    float dt_dy = (std::abs(dy) > 1e-8f) ? std::abs(1.0f / dy) : 1e8f;

    float t_max_x = 0.5f * dt_dx;
    float t_max_y = 0.5f * dt_dy;

    int x = static_cast<int>(sx);
    int y = static_cast<int>(sy);
    float remaining = ray_intensity;
    float distance = 0.0f;

    while (remaining > 0.01f) {
        if (x < 0 || x >= w || y < 0 || y >= h) break;

        float dist_atten = (distance > 0.0f)
            ? 1.0f / (1.0f + distance * distance * 0.01f)
            : 1.0f;

        light_map[y * w + x] += remaining * dist_atten;

        // Wall stops ray (skip source tile)
        if (distance > 0.0f && is_wall[y * w + x]) break;

        // Smoke absorption
        float sd = smoke_field[y * w + x];
        if (sd > 0.001f) {
            remaining *= (1.0f - sd * smoke_absorption);
        }

        // DDA step
        if (t_max_x < t_max_y) {
            x += step_x;
            distance = t_max_x;
            t_max_x += dt_dx;
        } else {
            y += step_y;
            distance = t_max_y;
            t_max_y += dt_dy;
        }

        if (distance > max_range) break;
    }
}

void Raycaster::cast_source(
    const LightSource& src,
    float* light_map,
    const float* smoke_field,
    const bool* is_wall,
    int h, int w
) const {
    int ray_count = src.get_ray_count();
    float half_spread = src.angle_spread * 0.5f;
    bool is_cone = src.angle_spread < 2.0f * PI - 0.01f;

    // Simple deterministic jitter using source position as seed
    std::mt19937 rng(static_cast<unsigned>(src.x * 1000 + src.y));
    std::uniform_real_distribution<float> jitter_dist(-1.0f, 1.0f);

    for (int i = 0; i < ray_count; ++i) {
        float t = (i + 0.5f) / ray_count;
        float angle = src.angle_center - half_spread + t * src.angle_spread;

        if (src.jitter > 0.0f) {
            angle += jitter_dist(rng) * src.jitter;
        }

        float angular_atten = 1.0f;
        if (is_cone) {
            float offset = angle - src.angle_center;
            // Wrap to [-pi, pi]
            while (offset >  PI) offset -= 2.0f * PI;
            while (offset < -PI) offset += 2.0f * PI;
            float norm = std::abs(offset) / (half_spread + 1e-6f);

            switch (src.falloff) {
                case Falloff::COSINE:
                    // Q2-LIFT: integer-kit cos (arg in [0, pi/2] — in range).
                    angular_atten = det_cos(std::min(norm, 1.0f) * PI * 0.5f);
                    break;
                case Falloff::SHARP:
                    angular_atten = (norm < 0.9f) ? 1.0f : 0.0f;
                    break;
                default:
                    angular_atten = 1.0f;
                    break;
            }
        }

        float intensity = src.intensity * angular_atten;
        if (intensity > 0.01f) {
            march_ray(src.x, src.y, angle, intensity, src.max_range,
                      light_map, smoke_field, is_wall, h, w);
        }
    }
}

// ---- Directional ray march ----

void Raycaster::march_ray_directional(
    float sx, float sy, float angle,
    float ray_intensity, float max_range,
    const float color[3],
    float heat_emit,
    float* light_rgb,
    float* light_dx, float* light_dy,
    int32_t* heat,
    float* smoke_glow,
    const float* gas_field,
    const float* gas_absorption,
    const float* gas_scatter,
    int n_gases,
    const float* light_atten,
    const float* heat_atten,
    int h, int w,
    const RadCtx* rad,
    const RadRay* rr
) const {
    // Stride of one gas slice in the (n_gases, h, w) contiguous array: each
    // gas[g] starts at gas_field + g*plane and is itself a (h, w) plane.
    const int plane = h * w;
    float dx = det_cos(angle);   // Q2-LIFT: integer-kit trig (see det_cos above)
    float dy = det_sin(angle);

    int step_x = (dx >= 0) ? 1 : -1;
    int step_y = (dy >= 0) ? 1 : -1;
    float dt_dx = (std::abs(dx) > 1e-8f) ? std::abs(1.0f / dx) : 1e8f;
    float dt_dy = (std::abs(dy) > 1e-8f) ? std::abs(1.0f / dy) : 1e8f;

    float t_max_x = 0.5f * dt_dx;
    float t_max_y = 0.5f * dt_dy;

    int x = static_cast<int>(sx);
    int y = static_cast<int>(sy);
    // Per-channel EMITTED energy: the ray's fixed per-channel budget. The source
    // tint folds in here; `ray_intensity` is the per-ray energy P/N (the caller
    // divides the source's total power by the ray count — see
    // cast_source_directional). This is CONSTANT along the ray: occlusion acts
    // through `survival` below, NEVER through distance. The 1/r intensity falloff
    // emerges from ray DENSITY, not a per-ray multiplier (engine/08 §Falloff is
    // density: a cell at range r is crossed by ~N/(Δr) rays, the N cancels).
    const float e_r = ray_intensity * color[0];
    const float e_g = ray_intensity * color[1];
    const float e_b = ray_intensity * color[2];
    const bool emits_r = e_r > 0.0f;
    const bool emits_g = e_g > 0.0f;
    const bool emits_b = e_b > 0.0f;
    const bool emits_heat = (heat != nullptr) && (heat_emit > 0.0f);
    // P-R4: the RADIATION channel. It shares the heat channel's survival and
    // its cull gate (the material-only quantity — the determinism contract that
    // keeps the touched tile set independent of the float light path), but it
    // is an independent DEPOSIT: `emits_heat` is the retired painter's own
    // gate and no fire source sets it any more.
    const bool emits_rad = (rad != nullptr) && (rr != nullptr) &&
                           rad->active() && (heat_atten != nullptr) &&
                           (rr->coef != 0.0f);

    // Per-channel SURVIVAL ∈ [0,1]: starts at 1.0, decays ONLY by occlusion
    // (material atten + gas Beer-Lambert for RGB; `heat_atten` for heat). NEVER
    // by distance. The deposit is energy·survival; survival is the whole falloff
    // story besides density.
    float survival[3] = {1.0f, 1.0f, 1.0f};
    // Heat is the INDEPENDENT 4th ray channel (engine/06 §1). Its survival is
    // attenuated per tile by `heat_atten` ONLY (gases never block heat), so it
    // diverges from the RGB survivors — a heat-shield (light-clear, heat-opaque)
    // blocks heat while passing light, smoked glass does the converse. The heat
    // DEPOSIT gates on `heat_survival > heat_cull`, a material-only quantity with
    // NO gas-optics `exp` in it — that gating is what DECOUPLES the heat-touched
    // tile set from the float light path -> bit-identical CPU/GPU heat
    // (engine/08 §Determinism: heat is decoupled from light).
    float heat_survival = 1.0f;
    float distance = 0.0f;

    // Per-channel SURVIVAL termination (engine/08 §The march, step 6). The ray
    // marches while ANY emitting channel's own survival is above its own floor
    // (`light_cull` for RGB, `heat_cull` for heat); it dies when EVERY emitting
    // channel is below floor (fully absorbed) or max_range is exceeded. Survival
    // decays only by occlusion, so in OPEN AIR every survival stays 1.0 and the
    // ray runs to max_range (heat radiates across the whole room) — the cull only
    // bites BEHIND occluders. The march length is the AGGREGATE (any-channel)
    // range, uniform regardless of which channel dies first (no per-channel
    // early-out -> no GPU warp divergence, ch.03 §CUDA contract); the per-channel
    // DEPOSIT gating (heat below) is a branch on a value the thread already holds,
    // so it adds no divergence. A pure light source (emits_heat == false) is not
    // kept alive by heat_survival, and a pure heat source not by RGB.
    auto ray_alive = [&]() {
        if (emits_r && survival[0] > light_cull) return true;
        if (emits_g && survival[1] > light_cull) return true;
        if (emits_b && survival[2] > light_cull) return true;
        // P-R4: radiation keeps the ray alive on the SAME heat survival/cull
        // pair the painter used — so the marched (and therefore heat-touched)
        // tile set is governed by exactly the same material-only quantity as
        // before. Nothing about the determinism contract moves.
        if ((emits_heat || emits_rad) && heat_survival > heat_cull) return true;
        return false;
    };

    while (ray_alive()) {
        if (x < 0 || x >= w || y < 0 || y >= h) break;

        int idx = y * w + x;

        // Per-channel deposit: this channel's fixed energy times its survival.
        // There is NO distance falloff (engine/08 §Falloff is density) — the 1/r
        // law is carried by how many rays cross the cell, not by shrinking each
        // ray. The source-cell pile-up (all N rays start here) is the natural
        // peak; brightness is ray-count-independent because energy = P/N.
        float dep_r = e_r * survival[0];
        float dep_g = e_g * survival[1];
        float dep_b = e_b * survival[2];
        light_rgb[idx * 3 + 0] += dep_r;
        light_rgb[idx * 3 + 1] += dep_g;
        light_rgb[idx * 3 + 2] += dep_b;

        // Direction = where the light is COMING FROM (toward the source).
        // Ray travel direction is (dx, dy); light arrives FROM (-dx, -dy).
        // Weight by the AGGREGATE deposit (sum of channels) so the intensity-
        // weighting intent of the scalar path is preserved with RGB rays.
        float dep_agg = dep_r + dep_g + dep_b;
        light_dx[idx] += dep_agg * (-dx);
        light_dy[idx] += dep_agg * (-dy);

        // Heat deposit (ch.04 §Fixed-point format, engine/06 §1) — GATED on
        // `heat_survival > heat_cull`. The deposit is the source's per-ray heat
        // emission times THIS channel's own survivor — NO distance falloff, NOT
        // the RGB aggregate. The gate (material-only survival, no `exp`) is the
        // determinism contract: the heat-touched tile set never depends on the
        // float light path, so a heat-shield stops heat exactly at the shield and
        // GPU heat is bit-identical to CPU. Quantized + SATURATING add (clamp at
        // INT32_MAX, never wrap); the ordered scalar += stays deterministic
        // (becomes a CUDA integer atomicAdd later — order-free, exact).
        if (emits_heat && heat_survival > heat_cull) {
            float heat_dep = heat_emit * heat_survival;
            heat_saturating_add(&heat[idx], heat_quantize(heat_dep));
        }

        // ---- P-R4: the NET-T⁴ RADIATION EXCHANGE (ruling A1) --------------
        //
        // This REPLACES the painter for every fire source (`emits_heat` is
        // false on every source the fire builder makes — `k_fire_heat` no
        // longer exists). Per absorbing cell r, against the SAME
        // `heat_survival > heat_cull` gate the painter used:
        //
        //   diff = E°[T_s] − E°[T_r]                        (signed int32)
        //   net  = a_s · w · τ · a_r · diff                 (ONE quantize)
        //   net  = clamp(net, ± min(budget_s, budget_r))    (flux limiter)
        //   rad_net[r] += net;  rad_net[s] −= net           (the SAME integer)
        //
        // PINNED MULTIPLY ORDER (left fold; the CUDA twin in cuda_raycaster.cu
        // pins the IDENTICAL order — this is the tol-0 contract):
        //     f  = rr->coef            // == a_s * (1/ray_count), host-folded
        //     f *= heat_survival       // × τ   (running material transmittance)
        //     f *= a_r                 // × a_r (this cell's absorptivity)
        //     v  = (double)f * (double)diff        // ONE promotion, full
        //                                          // integer precision on diff
        //     net = rad_quantize_signed(v)         // round half away from zero
        // The float part follows the march's existing float discipline (the
        // TU is /fp:strict, the device is --fmad=false /fp:strict); the DEPOSIT
        // is quantized ONCE per cell, exactly the old heat_quantize boundary.
        //
        // SELF-CELL: the source is its own first marched cell, so r == s,
        // T_r == T_s, the two lookups hit the SAME bucket and diff == 0 —
        // "our radiation will not heat its own tile" falls out structurally
        // (ruling A1.2). The code runs; it produces exactly 0.
        if (emits_rad && heat_survival > heat_cull) {
            const float a_r = heat_atten[idx];
            // ---- D3: the RADIANT-FLUX SENSOR at AIR cells -----------------
            // *** NOT PART OF THE ENERGY LEDGER. ***  No energy moves, nothing
            // is debited, no temperature changes, no solver reads this. It is
            // the incident radiant flux a UNIT standing on this air tile would
            // feel, and its only consumer is unit heat damage
            // (apply_environmental_damage), which sampled the retired painter's
            // air deposit here. Radiation itself lands only on solids (air has
            // a == 0, so by Kirchhoff it neither absorbs nor emits), so without
            // this sensor a fire could not burn a marine standing beside it.
            // Positive-only, so it keeps `heat[]`'s ORDER-FREE saturating-add
            // contract (the CUDA twin uses the saturating atomic).
            // Pinned fold, same shape as the exchange's: f = a_s*w -> *= tau,
            // then ONE promotion against the integer E°.
            if (rad->rad_flux != nullptr && !(a_r > 0.0f)) {
                float ff = rr->coef;          // a_s · w
                ff *= heat_survival;          // · τ
                const int32_t q = rad_quantize_signed((double)ff * (double)rr->E_s);
                if (q > 0) heat_saturating_add(&rad->rad_flux[idx], q);
            }
            if (a_r > 0.0f) {                 // air (a == 0) neither absorbs nor receives
                const int32_t T_r = rad->temperature[idx];
                const int32_t diff = rr->E_s - rad->e_table[e_bucket_of(T_r)];
                float f = rr->coef;           // a_s · w
                f *= heat_survival;           // · τ
                f *= a_r;                     // · a_r
                int32_t net = rad_quantize_signed((double)f * (double)diff);
                // ---- flux limiter (Levermore & Pomraning 1981; ruling A1.6)
                // Each end may shed/absorb at most 1/2^RAD_LIM_SHIFT of the
                // pair's temperature GAP through its OWN thermal mass, per ray,
                // per tick. Inert in normal operation; a rail at extreme gaps.
                const int64_t dT = (int64_t)rr->T_q - (int64_t)T_r;
                const int64_t adT = (dT < 0) ? -dT : dT;
                const int64_t b_s = rad_pair_budget(adT, rr->his_s);
                const int64_t b_r = rad_pair_budget(adT, (int)rad->heat_inv_shift[idx]);
                const int64_t cap = (b_s < b_r) ? b_s : b_r;
                if ((int64_t)net >  cap) net = (int32_t)cap;
                if ((int64_t)net < -cap) net = (int32_t)(-cap);
                // Antisymmetric apply: the SAME integer, + to the receiver and
                // − to the emitter. Plain signed (wrapping) adds — order-free.
                rad_signed_add(&rad->rad_net[idx], net);
                rad_signed_add(&rad->rad_net[rr->src_idx], -net);
            }
        }

        // Occlusion via per-channel attenuation (ch.03 §the march). Static
        // material attenuation from the table decays each channel's survival.
        // (1 - atten): opaque 1.0 -> 0, glass 0.1 -> 0.9 survives, asymmetric
        // triples tint the survivor.
        float ma_r = light_atten[idx * 3 + 0];
        float ma_g = light_atten[idx * 3 + 1];
        float ma_b = light_atten[idx * 3 + 2];
        survival[0] *= (1.0f - ma_r);
        survival[1] *= (1.0f - ma_g);
        survival[2] *= (1.0f - ma_b);

        // Heat survival attenuates by the per-tile `heat_atten` EXACTLY as each
        // RGB channel attenuates by its `light_atten[c]`: multiplicative
        // transmission (1 - heat_atten). air 0.0 -> survives untouched,
        // walls 1.0 -> killed (== the old wall hard-stop for heat), glass 0.3 ->
        // 0.7 transmits. Independent of the RGB multiply above. nullptr field ==
        // no attenuation (pre-S6 behaviour: heat survival stays 1.0).
        //
        // SOURCE-TILE SKIP (K2 — fire as a sim-side heat source): a fire only
        // ever burns on a FLAMMABLE solid (wood/door), which is heat-opaque
        // (heat_atten 1.0). A radiating surface emits OUTWARD — it does not
        // absorb its own emission — so the burning tile's own heat_atten must
        // NOT kill the ray before it leaves the cell, or fire could never
        // radiate across the adjacent room (the canon model, engine/06 §1:
        // "a fire radiates heat across an open room; distant wood catches").
        // On the SOURCE tile (distance == 0, the very first marched cell) we
        // therefore deposit the heat (done above) but SKIP this self-occlusion,
        // so heat_survival stays 1.0 leaving the source and the ray radiates
        // into the air beyond. Every downrange tile attenuates normally, so a
        // wall still blocks the fire's heat beyond it (occlusion intact). This
        // is INERT for air-sourced lights/beams (their source heat_atten is 0,
        // so 1-0 == no-op) — it only matters for a heat source that sits inside
        // an opaque tile, i.e. fire. Light occlusion is untouched (a lamp never
        // sits in a wall; fire emits no meaningful light here anyway).
        if (heat_atten != nullptr && distance > 0.0f) {
            heat_survival *= (1.0f - heat_atten[idx]);
        }

        // ---- Multi-gas coloured optics (engine/05 §6.2 — density-weighted
        // per-channel sum over ALL gases) ----
        // Sum the two decoupled budgets across every gas sharing this tile, each
        // weighted by its local density and its OWN per-channel table row:
        //   tau_c     = absorb_scale * Σ_g ( gas[g][tile] * absorption[g][c] )
        //   scatter_c =                Σ_g ( gas[g][tile] * scatter_albedo[g][c] )
        // Mixing falls out of the sum (poison+black -> murky automatically); a
        // single populated gas reproduces the old single-`smoke` path for that
        // gas's coefficients. The 5-gas inner loop is a few FLOPs per tile-step.
        float tau_r = 0.0f, tau_g = 0.0f, tau_b = 0.0f;
        float sca_r = 0.0f, sca_g = 0.0f, sca_b = 0.0f;
        for (int g = 0; g < n_gases; ++g) {
            float gd = gas_field[g * plane + idx];
            if (gd <= 0.001f) continue;
            const float* ab = &gas_absorption[g * 3];
            const float* sc = &gas_scatter[g * 3];
            tau_r += gd * ab[0];
            tau_g += gd * ab[1];
            tau_b += gd * ab[2];
            sca_r += gd * sc[0];
            sca_g += gd * sc[1];
            sca_b += gd * sc[2];
        }
        if (tau_r > 0.0f || tau_g > 0.0f || tau_b > 0.0f ||
            sca_r > 0.0f || sca_g > 0.0f || sca_b > 0.0f) {
            // (1) God-rays / scatter (ADDITIVE deposit into smoke_glow): the
            // light the gases SCATTER BACK toward the viewer, per channel — the
            // density-weighted scatter sum times the LOCAL light (dep_c, the same
            // energy the light buffer saw). SEPARATE budget, decoupled from (and
            // may exceed) absorption -> "barely absorbs, glows brightly" (steam).
            // RGB-preserving (a red beam casts a red shaft).
            if (smoke_glow != nullptr) {
                smoke_glow[idx * 3 + 0] += dep_r * sca_r;
                smoke_glow[idx * 3 + 1] += dep_g * sca_g;
                smoke_glow[idx * 3 + 2] += dep_b * sca_b;
            }
            // (2) Per-channel transmission (Beer-Lambert, ch.05 §6.1 6a):
            //   trans_c = exp(-absorb_scale * tau_c)   // never reaches 0 -> beam survives
            //   survival[c] *= trans_c                 // multiplicative tint, long reach
            // absorb_scale is the global beam-reach dial (LOW = far). Green poison
            // absorbs R+B and passes G -> the surviving beam (and the light behind)
            // is greened; mixing falls out of the summed tau. This `exp` lives on
            // the RGB survival ONLY — heat survival never sees it, which is exactly
            // why the heat channel stays deterministic (engine/08 §Determinism).
            survival[0] *= std::exp(-smoke_absorb_scale * tau_r);
            survival[1] *= std::exp(-smoke_absorb_scale * tau_g);
            survival[2] *= std::exp(-smoke_absorb_scale * tau_b);
            // NOTE: gases do NOT attenuate heat — only material `heat_atten`
            // blocks the heat channel. (Heat radiates through smoke; the
            // god-ray/transmission optics above are a light-only model.)
        }

        if (t_max_x < t_max_y) {
            x += step_x;
            distance = t_max_x;
            t_max_x += dt_dx;
        } else {
            y += step_y;
            distance = t_max_y;
            t_max_y += dt_dy;
        }

        if (distance > max_range) break;
    }
}

void Raycaster::cast_source_directional(
    const LightSource& src,
    float* light_rgb,
    float* light_dx,
    float* light_dy,
    int32_t* heat,
    float* smoke_glow,
    const float* gas_field,
    const float* gas_absorption,
    const float* gas_scatter,
    int n_gases,
    const float* light_atten,
    const float* heat_atten,
    int h, int w,
    const RadCtx* rad,
    const RadSource* rs
) const {
    int ray_count = src.get_ray_count();
    // Pure-density falloff (engine/08 §Falloff is density): each ray carries the
    // source's TOTAL power / N, so a cell at range r — crossed by ~N/(Δr) rays —
    // accumulates P·survival/(Δr) and the N cancels (brightness is independent of
    // ray count; ray count is a QUALITY knob). `src.intensity` / `src.heat` now
    // mean total emitted power; this `inv_n` folds the /N into each ray's budget.
    float inv_n = 1.0f / static_cast<float>(ray_count);
    float half_spread = src.angle_spread * 0.5f;
    bool is_cone = src.angle_spread < 2.0f * PI - 0.01f;

    std::mt19937 rng(static_cast<unsigned>(src.x * 1000 + src.y));
    std::uniform_real_distribution<float> jitter_dist(-1.0f, 1.0f);

    for (int i = 0; i < ray_count; ++i) {
        float t = (i + 0.5f) / ray_count;
        float angle = src.angle_center - half_spread + t * src.angle_spread;

        if (src.jitter > 0.0f) {
            angle += jitter_dist(rng) * src.jitter;
        }

        float angular_atten = 1.0f;
        if (is_cone) {
            float offset = angle - src.angle_center;
            while (offset >  PI) offset -= 2.0f * PI;
            while (offset < -PI) offset += 2.0f * PI;
            float norm = std::abs(offset) / (half_spread + 1e-6f);

            switch (src.falloff) {
                case Falloff::COSINE:
                    // Q2-LIFT: integer-kit cos (arg in [0, pi/2] — in range).
                    angular_atten = det_cos(std::min(norm, 1.0f) * PI * 0.5f);
                    break;
                case Falloff::SHARP:
                    angular_atten = (norm < 0.9f) ? 1.0f : 0.0f;
                    break;
                default:
                    angular_atten = 1.0f;
                    break;
            }
        }

        // Per-ray budgets = (total power / N) · angular_atten, for BOTH light
        // and heat. The old `intensity > 0.01` guard was an absolute-energy cut
        // that, now that each ray only carries P/N, would wrongly drop every ray
        // of any many-ray source — gate on the angular weight instead: a SHARP
        // cone zeroes rays outside the beam (skip), UNIFORM/COSINE always cast.
        float ray_energy = src.intensity * angular_atten * inv_n;
        float ray_heat   = src.heat * angular_atten * inv_n;
        // P-R4: the radiation per-ray constant folds a_s and 1/N the SAME way
        // `ray_heat` folded the painter's payload — one PINNED product, host
        // side, so the marched cell sees a single float coefficient.
        RadRay rr;
        const bool has_rad = (rad != nullptr) && (rs != nullptr) && rad->active();
        if (has_rad) {
            rr.src_idx = rs->idx;
            rr.T_q     = rs->T_q;
            rr.E_s     = rs->E_s;
            rr.his_s   = rs->his_s;
            rr.coef    = rs->a_s * angular_atten * inv_n;
        }
        if (angular_atten > 0.0f) {
            march_ray_directional(src.x, src.y, angle, ray_energy, src.max_range,
                      src.color, ray_heat, light_rgb, light_dx, light_dy,
                      heat, smoke_glow, gas_field, gas_absorption, gas_scatter,
                      n_gases, light_atten, heat_atten, h, w,
                      has_rad ? rad : nullptr, has_rad ? &rr : nullptr);
        }
    }
}

void Raycaster::normalize_directions(float* light_dx, float* light_dy, int h, int w) {
    int n = h * w;
    for (int i = 0; i < n; ++i) {
        float lx = light_dx[i];
        float ly = light_dy[i];
        float len2 = lx * lx + ly * ly;
        if (len2 > 1e-12f) {
            float inv_len = 1.0f / std::sqrt(len2);
            light_dx[i] = lx * inv_len;
            light_dy[i] = ly * inv_len;
        } else {
            light_dx[i] = 0.0f;
            light_dy[i] = 0.0f;
        }
    }
}

// ---- P-R1: whole-fire-plane source build (moved from Python) --------------
//
// docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.1-A4.2. See the
// FLOAT-PARITY CONTRACT on the declaration (raycaster.h) for why every
// expression below is double arithmetic narrowed to float only at the very
// end — it must land the identical float32 bits PhysicsRunner.cast_fire_heat's
// old per-tile Python loop produced (pybind narrows a Python double to the
// LightSource's float field at the `src.x = ...`-style assignment; Python
// itself never narrows earlier, since Python floats are C doubles).
//
// P-R4 (ruling A1.8): the enumeration widens from `burning` to
// `burning ∪ (thermal_solid && T >= T_emit_gate)` and every source now also
// yields a RadSource (its T, its E°, its a_s, its heat_inv_shift, its index).
// `k_fire_heat` and the `heat` payload are GONE with the painter.
std::vector<LightSource> Raycaster::build_fire_sources(
    const int32_t* fire, int h, int w,
    int fire_ray_count,
    double range_base, double range_per_intensity,
    double intensity_base, double intensity_per_intensity,
    const float color[3], double jitter,
    const int32_t* temperature, const float* heat_atten,
    const int32_t* heat_inv_shift, const bool* thermal_solid,
    int tick,
    std::vector<RadSource>* rad_out
) const {
    // math.pi's bits: CPython's `math.pi` is the literal 3.14159265358979323846
    // (Py_MATH_PI, mathmodule.c) rounded to the nearest double. The SAME
    // literal here rounds to the identical double under any IEEE754 compiler
    // (it carries far more digits than needed to disambiguate the nearest
    // double), so `two_pi_d` below is bit-identical to Python's `2.0 * math.pi`.
    static constexpr double PI_D = 3.14159265358979323846;
    const double two_pi_d = 2.0 * PI_D;
    const float angle_spread_f = static_cast<float>(two_pi_d);
    const float jitter_f = static_cast<float>(jitter);
    // Hoisted invariant sub-expression: Python recomputes `two_pi / ray_count`
    // fresh on every loop iteration, but IEEE754 division is a deterministic
    // pure function of its two (here loop-invariant) operands, so evaluating
    // it once here is bit-identical to Python's per-iteration recompute.
    const double phase_step_d = two_pi_d / static_cast<double>(fire_ray_count);
    // D4 (amendment 5): the SUB-SPACING per-tick rotation. `phase_step_d` alone
    // is a NO-OP on a full-circle fan (rotating N evenly-spaced rays by a
    // multiple of their own spacing maps the set onto itself — which is why the
    // (+2,0) axis neighbour was NEVER connected at any intensity). Stepping by
    // one N-th of the spacing instead means N consecutive ticks sweep exactly
    // one full spacing, so every direction is sampled and the discrete view
    // factor time-averages to the continuous one. Pure function of the tick;
    // both backends build the fan from THIS function, so they stay identical.
    const double tick_step_d = phase_step_d / static_cast<double>(fire_ray_count);
    // Non-negative tick residue (a tick is never negative in the engine; the
    // guard keeps the modulo total and deterministic if one ever were).
    int tick_mod = tick % fire_ray_count;
    if (tick_mod < 0) tick_mod += fire_ray_count;
    // Q16.16 dequant: fire's fixed-point scale is the SAME 65536 as HEAT_SCALE
    // (== fire_fixed.FP_ONE, src/simulation/fire_fixed.py). Division by an
    // exact power of two is exact in IEEE754 (no rounding), matching Python's
    // `float(fire[y, x]) / fire_fixed.FP_ONE_F` bit-for-bit.
    const double fp_one_d = static_cast<double>(HEAT_SCALE);

    // P-R4: the warm-emitter gate, quantized ONCE (the load-time boundary cast)
    // so the per-tile test is a pure Q16.16 integer compare.
    const int32_t t_emit_q = fixedpoint::quantize(T_emit_gate);
    const int32_t* e_tab = emissive_table();   // bakes on first use / on a dial move

    std::vector<LightSource> sources;
    if (rad_out) rad_out->clear();
    // ROW-MAJOR enumeration (row outer, col inner) — the order np.nonzero
    // yields in C order, which the old Python loop
    // (`for yy, xx in zip(*np.nonzero(burning))`) walked. P-R4 keeps the order
    // and widens the predicate; the deposit is order-free either way, but the
    // order is part of the contract the CUDA concatenation mirrors.
    for (int row = 0; row < h; ++row) {
        const int32_t* row_ptr = fire + static_cast<size_t>(row) * w;
        for (int col = 0; col < w; ++col) {
            const size_t i = static_cast<size_t>(row) * w + col;
            const int32_t fq = row_ptr[col];
            const bool burning = (fq > 0);
            // WARM EMITTER (ruling A1.8): a thermal solid at or above the gate
            // also casts. The gate decides who can radiatively LOSE heat —
            // receivers are free, so a cold crate is still heated correctly.
            const bool warm = (!burning) && (temperature != nullptr)
                           && (thermal_solid != nullptr) && thermal_solid[i]
                           && (temperature[i] >= t_emit_q);
            if (!burning && !warm) continue;

            // A source with a_s == 0 can neither emit nor absorb (Kirchhoff),
            // so it would contribute exactly nothing — skip it rather than
            // spend 8 marches on a guaranteed zero. (Fire only ever burns on a
            // flammable solid, which is heat-opaque, so this never drops a real
            // flame; it drops a hypothetical fire painted onto air.)
            const float a_s = (heat_atten != nullptr) ? heat_atten[i] : 0.0f;
            if (rad_out != nullptr && !(a_s > 0.0f)) continue;

            // A WARM emitter has no flame, so I = 0 in the range/intensity
            // formulas: max_range = range_base (the documented interim choice —
            // short reach for a merely-hot surface).
            const double I = burning ? (static_cast<double>(fq) / fp_one_d) : 0.0;

            const double x_d = static_cast<double>(col) + 0.5;
            const double y_d = static_cast<double>(row) + 0.5;
            const double max_range_d = range_base + range_per_intensity * I;
            const double intensity_d = intensity_base + intensity_per_intensity * I;
            const int mod_result = (col * 7 + row * 13) % fire_ray_count;
            // D4: the spatial hash (decorrelates neighbouring fans) PLUS the
            // sub-spacing per-tick rotation (connects every pair within
            // ray_count ticks). Both terms in double, narrowed once, as the
            // float-parity contract requires.
            const double angle_center_d =
                static_cast<double>(mod_result) * phase_step_d
              + static_cast<double>(tick_mod) * tick_step_d;

            LightSource src;
            src.x = static_cast<float>(x_d);
            src.y = static_cast<float>(y_d);
            src.max_range = static_cast<float>(max_range_d);
            src.ray_count = fire_ray_count;
            src.angle_center = static_cast<float>(angle_center_d);
            src.angle_spread = angle_spread_f;
            src.intensity = static_cast<float>(intensity_d);
            // P-R4: the painter's payload is GONE. `heat` stays 0 so
            // march_ray_directional's `emits_heat` gate is false for every fire
            // source — the one-way deposit cannot run on this path at all.
            src.heat = 0.0f;
            src.jitter = jitter_f;
            src.color[0] = color[0];
            src.color[1] = color[1];
            src.color[2] = color[2];
            // falloff: default-initialized to Falloff::UNIFORM (LightSource's
            // in-class default) — the old Python loop never set it either.
            sources.push_back(src);

            if (rad_out != nullptr) {
                RadSource rs;
                rs.idx   = static_cast<int>(i);
                rs.T_q   = temperature[i];
                rs.E_s   = e_tab[e_bucket_of(rs.T_q)];
                rs.a_s   = a_s;
                rs.his_s = (heat_inv_shift != nullptr) ? (int)heat_inv_shift[i] : 0;
                rad_out->push_back(rs);
            }
        }
    }
    return sources;
}

void Raycaster::cast_from_fire_plane(
    const int32_t* fire, int h, int w,
    int fire_ray_count,
    double range_base, double range_per_intensity,
    double intensity_base, double intensity_per_intensity,
    const float color[3],
    float* light_rgb,
    float* light_dx,
    float* light_dy,
    float* smoke_glow,
    const float* gas_field,
    const float* gas_absorption,
    const float* gas_scatter,
    int n_gases,
    const float* light_atten,
    const float* heat_atten,
    const int32_t* temperature,
    const int32_t* heat_inv_shift,
    const bool* thermal_solid,
    int32_t* rad_net,
    int32_t* rad_flux,
    int tick,
    double jitter
) const {
    std::vector<RadSource> rads;
    std::vector<LightSource> sources = build_fire_sources(
        fire, h, w, fire_ray_count, range_base, range_per_intensity,
        intensity_base, intensity_per_intensity, color, jitter,
        temperature, heat_atten, heat_inv_shift, thermal_solid, tick, &rads);
    RadCtx ctx;
    ctx.e_table        = emissive_table();
    ctx.temperature    = temperature;
    ctx.heat_inv_shift = heat_inv_shift;
    ctx.rad_net        = rad_net;
    ctx.rad_flux       = rad_flux;
    for (size_t k = 0; k < sources.size(); ++k) {
        // `heat` is nullptr: the fire path has NO one-way deposit any more.
        cast_source_directional(sources[k], light_rgb, light_dx, light_dy,
                                 /*heat=*/nullptr, smoke_glow,
                                 gas_field, gas_absorption, gas_scatter, n_gases,
                                 light_atten, heat_atten, h, w,
                                 &ctx, &rads[k]);
    }
}

// ---- CUDA-S2 gate: host ray-list builder ----------------------------------
//
// A line-for-line replica of cast_source_directional's per-ray loop (above),
// EXCEPT the body folds each ray into a RayHD instead of marching it on the CPU.
// Same ray_count, the same t=(i+0.5)/N angle sweep, the SAME jitter RNG drawn in
// the SAME order, the same falloff angular_atten, the same inv_n. The per-channel
// emitted energy folds the source tint exactly as march_ray_directional does
// internally (e_c = ray_energy * color[c]; heat_emit = ray_heat). Rays with
// angular_atten<=0 are SKIPPED (the CPU cast guards `if (angular_atten > 0)`),
// so the RNG draw still advances for every i — preserving the jitter sequence.
//
// dx=cos(angle)/dy=sin(angle) are computed HERE in this /fp:strict TU, the same
// place cast_source_directional computes `angle`; march_ray_directional recomputes
// dx/dy from the identical `angle` with the same det_cos/det_sin (Q2-LIFT: the
// pure-integer kit — identical bits in ANY TU on ANY machine, stronger than the
// old shared-libm argument) — so the GPU (which reads these precomputed dx/dy)
// walks bit-identical DDA tiles -> heat matches CPU byte-for-byte.
std::vector<breach_cuda::RayHD> Raycaster::build_ray_list(
        const LightSource& src, const RadSource* rs) const {
    std::vector<breach_cuda::RayHD> rays;

    int ray_count = src.get_ray_count();
    float inv_n = 1.0f / static_cast<float>(ray_count);
    float half_spread = src.angle_spread * 0.5f;
    bool is_cone = src.angle_spread < 2.0f * PI - 0.01f;

    std::mt19937 rng(static_cast<unsigned>(src.x * 1000 + src.y));
    std::uniform_real_distribution<float> jitter_dist(-1.0f, 1.0f);

    rays.reserve(static_cast<size_t>(ray_count));

    for (int i = 0; i < ray_count; ++i) {
        float t = (i + 0.5f) / ray_count;
        float angle = src.angle_center - half_spread + t * src.angle_spread;

        if (src.jitter > 0.0f) {
            angle += jitter_dist(rng) * src.jitter;
        }

        float angular_atten = 1.0f;
        if (is_cone) {
            float offset = angle - src.angle_center;
            while (offset >  PI) offset -= 2.0f * PI;
            while (offset < -PI) offset += 2.0f * PI;
            float norm = std::abs(offset) / (half_spread + 1e-6f);

            switch (src.falloff) {
                case Falloff::COSINE:
                    // Q2-LIFT: integer-kit cos (arg in [0, pi/2] — in range).
                    angular_atten = det_cos(std::min(norm, 1.0f) * PI * 0.5f);
                    break;
                case Falloff::SHARP:
                    angular_atten = (norm < 0.9f) ? 1.0f : 0.0f;
                    break;
                default:
                    angular_atten = 1.0f;
                    break;
            }
        }

        float ray_energy = src.intensity * angular_atten * inv_n;
        float ray_heat   = src.heat * angular_atten * inv_n;
        if (angular_atten > 0.0f) {
            breach_cuda::RayHD ray;
            ray.sx = src.x;
            ray.sy = src.y;
            ray.dx = det_cos(angle);   // Q2-LIFT: same kit as the CPU march ->
            ray.dy = det_sin(angle);   // GPU and CPU consume IDENTICAL dirs
            ray.e_r = ray_energy * src.color[0];
            ray.e_g = ray_energy * src.color[1];
            ray.e_b = ray_energy * src.color[2];
            ray.heat_emit = ray_heat;
            ray.max_range = src.max_range;
            // P-R4: the emitter's radiation payload rides on the ray, folded
            // in the SAME pinned order the CPU cast folds it
            // (a_s * angular_atten * inv_n) and in the SAME /fp:strict TU — so
            // the device march starts from a bit-identical coefficient.
            if (rs != nullptr) {
                ray.rad_src_idx = rs->idx;
                ray.rad_T_q     = rs->T_q;
                ray.rad_E_s     = rs->E_s;
                ray.rad_his_s   = rs->his_s;
                ray.rad_coef    = rs->a_s * angular_atten * inv_n;
            }
            rays.push_back(ray);
        }
    }
    return rays;
}

// ---- P-R1: CUDA twin of cast_from_fire_plane -------------------------------
//
// Same shared enumerator (build_fire_sources) as the CPU entry point, folded
// into RayHD via build_ray_list and concatenated in row-major source order —
// IDENTICAL concatenation shape to cuda_raycaster_cast_batch's pybind lambda
// (bindings.cpp, S8c item 1), which does this same loop over a Python-supplied
// source list. No march/law change: the caller feeds the result straight into
// the existing breach_cuda::raycaster_cast_directional batched device march.
std::vector<breach_cuda::RayHD> Raycaster::build_fire_ray_list(
    const int32_t* fire, int h, int w,
    int fire_ray_count,
    double range_base, double range_per_intensity,
    double intensity_base, double intensity_per_intensity,
    const float color[3],
    const int32_t* temperature,
    const float* heat_atten,
    const int32_t* heat_inv_shift,
    const bool* thermal_solid,
    int tick,
    double jitter
) const {
    std::vector<RadSource> rads;
    std::vector<LightSource> sources = build_fire_sources(
        fire, h, w, fire_ray_count, range_base, range_per_intensity,
        intensity_base, intensity_per_intensity, color, jitter,
        temperature, heat_atten, heat_inv_shift, thermal_solid, tick, &rads);
    std::vector<breach_cuda::RayHD> rays;
    for (size_t k = 0; k < sources.size(); ++k) {
        std::vector<breach_cuda::RayHD> r = build_ray_list(sources[k], &rads[k]);
        rays.insert(rays.end(), r.begin(), r.end());
    }
    return rays;
}
