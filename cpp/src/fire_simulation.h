#pragma once
// Fire simulation — spreading, O2 consumption, smoke emission, wall damage.
// Returns list of destroyed tile coordinates (Python handles destroy_wall callback).

#include <vector>
#include <utility>

struct FireParams {
    float spread_rate    = 0.3f;   // fire spread to neighbors
    float o2_threshold   = 0.60f;  // fire dies below this atmosphere
    float o2_consumption = 0.3f;   // atmosphere consumed by fire per second
    float smoke_emission = 0.8f;   // smoke produced by fire per second
    float wall_damage    = 0.4f;   // HP damage to walls per second
    float k_wind_thresh  = 0.5f;   // fire must exceed this * wind_speed to survive
    float k_wind_net     = 3.0f;   // rate of wind modulation
};

class FireSimulation {
public:
    FireParams params;

    // Returns vector of (y, x) coordinates where walls burned through.
    // Python must call destroy_wall() for each of these.
    std::vector<std::pair<int, int>> step(
        float* fire,
        float* atmosphere,
        float* smoke,
        float* wall_hp,
        const bool* is_wall,
        const bool* flammable,
        int h, int w,
        float dt
    ) const;
};
