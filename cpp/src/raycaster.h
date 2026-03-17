#pragma once
// 2D raycaster — DDA ray marching for light and heat.
// Casts rays from light sources, deposits intensity into a light map.

#include <vector>
#include <cmath>

// Falloff types for light sources
enum class Falloff : int { UNIFORM = 0, COSINE = 1, SHARP = 2 };

struct LightSource {
    float x, y;              // tile coordinates
    float max_range  = 20;
    int   ray_count  = 0;    // 0 = auto-compute from range + spread
    float angle_center = 0;
    float angle_spread = 2.0f * 3.14159265f;
    float intensity  = 1.0f;
    float heat       = 0.0f;
    float jitter     = 0.0f;
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

    // Cast all fire sources and deposit into light_map.
    // Collects fire tiles, clusters them, creates LightSources, casts rays.
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

private:
    void march_ray(
        float sx, float sy, float angle,
        float ray_intensity, float max_range,
        float* light_map,
        const float* smoke_field,
        const bool* is_wall,
        int h, int w
    ) const;
};
