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
    const float* smoke_field,
    const float* light_atten,
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
    // Per-channel remaining energy. The source tint is folded in here so the
    // deposit is the per-channel survivor (matches the old scalar*color deposit
    // on the source tile) and each colour attenuates independently downstream.
    float remaining[3] = {
        ray_intensity * color[0],
        ray_intensity * color[1],
        ray_intensity * color[2],
    };
    float distance = 0.0f;

    // Aggregate remaining-energy termination (max over channels). Opaque tiles
    // drive every channel to 0 -> aggregate < 0.01 -> ray ends next step,
    // identical to the old wall hard-stop. No per-channel early-out: all three
    // channels march in lockstep to the same aggregate range (CUDA-divergence
    // rule, ch.03 §CUDA contract).
    auto aggregate = [](const float r[3]) {
        return std::max(r[0], std::max(r[1], r[2]));
    };

    while (aggregate(remaining) > 0.01f) {
        if (x < 0 || x >= w || y < 0 || y >= h) break;

        float dist_atten = (distance > 0.0f)
            ? 1.0f / (1.0f + distance * distance * 0.01f)
            : 1.0f;
        int idx = y * w + x;

        // Per-channel deposit: this channel's survivor times distance falloff.
        float dep_r = remaining[0] * dist_atten;
        float dep_g = remaining[1] * dist_atten;
        float dep_b = remaining[2] * dist_atten;
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

        // Heat deposit (ch.04 §Fixed-point format). When the source emits heat
        // (heat_emit > 0), deposit the AGGREGATE per-tile light energy times the
        // source's heat multiplier into the Q16.16 `heat` buffer — quantized,
        // SATURATING add (clamp at INT32_MAX, never wrap). dep_agg is the energy
        // this tile received; src.heat scales light energy -> heat energy. The
        // ordered scalar += keeps it deterministic (CUDA atomicAdd later).
        if (heat != nullptr && heat_emit > 0.0f) {
            heat_saturating_add(&heat[idx], heat_quantize(dep_agg * heat_emit));
        }

        // Occlusion via per-channel attenuation (ch.03 §the march). Static
        // material attenuation from the table; then the live smoke attenuation,
        // both applied per channel. (1 - atten): opaque 1.0 -> 0, glass 0.1 ->
        // 0.9 survives, asymmetric triples tint the survivor.
        float ma_r = light_atten[idx * 3 + 0];
        float ma_g = light_atten[idx * 3 + 1];
        float ma_b = light_atten[idx * 3 + 2];
        remaining[0] *= (1.0f - ma_r);
        remaining[1] *= (1.0f - ma_g);
        remaining[2] *= (1.0f - ma_b);

        float sd = smoke_field[idx];
        if (sd > 0.001f) {
            // Fraction of the ray each smoke tile absorbs this step.
            float absorb_frac = sd * smoke_absorption;
            // God-rays (ch.03 C16): deposit the light the smoke ABSORBS into
            // smoke_glow, per channel. The absorbed energy is the survivor at
            // this tile (remaining[c] * dist_atten — the same energy `dep_*`
            // the light buffer saw) times the absorbed fraction. RGB-preserving
            // (a red beam casts a red shaft); energy-conserving by construction
            // (this is exactly the energy removed from `remaining` below).
            // Supersedes the old surface-tint light_modulation path (no
            // double-count) — see overlays.py / ch.05.
            if (smoke_glow != nullptr) {
                smoke_glow[idx * 3 + 0] += dep_r * absorb_frac;
                smoke_glow[idx * 3 + 1] += dep_g * absorb_frac;
                smoke_glow[idx * 3 + 2] += dep_b * absorb_frac;
            }
            float smoke_t = (1.0f - absorb_frac);
            remaining[0] *= smoke_t;
            remaining[1] *= smoke_t;
            remaining[2] *= smoke_t;
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
    const float* smoke_field,
    const float* light_atten,
    int h, int w
) const {
    int ray_count = src.get_ray_count();
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

        float intensity = src.intensity * angular_atten;
        if (intensity > 0.01f) {
            march_ray_directional(src.x, src.y, angle, intensity, src.max_range,
                      src.color, src.heat, light_rgb, light_dx, light_dy,
                      heat, smoke_glow, smoke_field, light_atten, h, w);
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
