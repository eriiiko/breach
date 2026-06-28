#include "raycaster.h"
#include <algorithm>
#include <cstring>
#include <random>

static constexpr float PI = 3.14159265358979f;

void Raycaster::march_ray(
    float sx, float sy, float angle,
    float ray_intensity, float max_range,
    float* light_map,
    const float* smoke_field,
    const bool* is_wall,
    int h, int w
) const {
    float dx = std::cos(angle);
    float dy = std::sin(angle);

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
                    angular_atten = std::cos(std::min(norm, 1.0f) * PI * 0.5f);
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
    int h, int w
) const {
    // Stride of one gas slice in the (n_gases, h, w) contiguous array: each
    // gas[g] starts at gas_field + g*plane and is itself a (h, w) plane.
    const int plane = h * w;
    float dx = std::cos(angle);
    float dy = std::sin(angle);

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
        if (emits_heat && heat_survival > heat_cull) return true;
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
    int h, int w
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
                    angular_atten = std::cos(std::min(norm, 1.0f) * PI * 0.5f);
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
        if (angular_atten > 0.0f) {
            march_ray_directional(src.x, src.y, angle, ray_energy, src.max_range,
                      src.color, ray_heat, light_rgb, light_dx, light_dy,
                      heat, smoke_glow, gas_field, gas_absorption, gas_scatter,
                      n_gases, light_atten, heat_atten, h, w);
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

void Raycaster::update_from_fire(
    float* light_map,
    const float* fire,
    const float* smoke_field,
    const bool* is_wall,
    int h, int w
) const {
    // Zero light map
    std::memset(light_map, 0, h * w * sizeof(float));

    // Early exit if no fire
    float max_fire = 0.0f;
    for (int i = 0; i < h * w; ++i) max_fire = std::max(max_fire, fire[i]);
    if (max_fire < 0.01f) return;

    // Cluster fire tiles on coarse grid to avoid casting from every burning tile
    int co = std::max(coarse_cluster, 1);
    std::vector<LightSource> sources;

    for (int cy = 0; cy < h; cy += co) {
        for (int cx = 0; cx < w; cx += co) {
            int by2 = std::min(cy + co, h);
            int bx2 = std::min(cx + co, w);

            // Find max fire in block
            float block_max = 0.0f;
            int best_y = cy, best_x = cx;
            for (int y = cy; y < by2; ++y) {
                for (int x = cx; x < bx2; ++x) {
                    if (fire[y * w + x] > block_max) {
                        block_max = fire[y * w + x];
                        best_y = y;
                        best_x = x;
                    }
                }
            }

            if (block_max > 0.1f) {
                LightSource src;
                src.x = static_cast<float>(best_x);
                src.y = static_cast<float>(best_y);
                src.max_range = 15;
                src.intensity = 0.8f * block_max;
                src.heat = 1.0f;
                src.jitter = 0.05f;
                src.falloff = Falloff::UNIFORM;
                sources.push_back(src);
            }
        }
    }

    for (const auto& src : sources) {
        cast_source(src, light_map, smoke_field, is_wall, h, w);
    }
}
