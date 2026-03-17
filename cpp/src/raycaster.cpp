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
