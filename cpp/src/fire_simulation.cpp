#include "fire_simulation.h"
#include "fixed_point.h"
#include <cmath>
#include <algorithm>
#include <cstdint>

// Neighbor offsets: 4-connected (the open-neighbour pressure mean + smoke spread).
static constexpr int D4[][2] = {{-1,0},{1,0},{0,-1},{0,1}};

static inline bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

static inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Hermite smoothstep on [edge0, edge1] -> [0,1], clamped outside.
static inline float smoothstep(float edge0, float edge1, float x) {
    if (edge1 <= edge0) return x < edge0 ? 0.0f : 1.0f;
    float t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

std::vector<std::pair<int, int>> FireSimulation::step(
    float* fire,
    float* atmosphere,
    int32_t* smoke,            // S2b: Q16.16
    float* wall_hp,
    const int32_t* temperature,
    const float* wind_x,
    const float* wind_y,
    const bool* is_wall,
    const bool* is_vacuum,
    const bool* flammable,
    int h, int w,
    float dt
) const {
    const int n = h * w;
    const auto& p = params;

    // Early exit if no fire (mirrors the old behaviour; the heat-spread /
    // ignition path lives in apply_temperature_ignition, NOT here).
    float max_fire = 0.0f;
    for (int i = 0; i < n; ++i) max_fire = std::max(max_fire, fire[i]);
    if (max_fire < 0.001f) return {};

    const float inv_temp = 1.0f / p.temp_scale;

    // --- Per-tile signed-logistic FEEDBACK (fire_design_proposal §2 + §5) ---
    // Spread is gone (radiation -> heat -> temperature -> ignition handles it);
    // this is purely the life/death of an already-lit tile. Each tile reads its
    // own intensity I, temperature T, fuel F (from wall_hp), neighbour pressure P
    // (the O2 proxy) and wind magnitude W, then steps I by dt*(grow - die).
    for (int i = 0; i < n; ++i) {
        if (!flammable[i]) continue;        // only fuel burns
        float I = fire[i];
        if (I <= 0.0f) continue;            // unlit -> nothing to step (ignition is elsewhere)

        const int y = i / w, x = i % w;

        // T: temperature field (Q16.16 -> float game units, same scale as ignition_temp).
        const float T = (float)temperature[i] * inv_temp;

        // F: fuel from remaining wall HP, normalised. As wall_hp -> 0 the fire
        // starves (the burn-out brake; burn-through below is the consumption).
        const float F = clamp01(wall_hp[i] / p.fuel_ref);

        // P: mean atmosphere over OPEN (non-solid, non-vacuum) 4-neighbours — the
        // fire reads INCOMING fresh air (its own plume bump is on its own tile,
        // excluded here). No open neighbour -> P = 0 (sealed in solid -> starves).
        float sum_atm = 0.0f;
        float count = 0.0f;
        for (const auto& d : D4) {
            int ny = y + d[0], nx = x + d[1];
            int ni = ny * w + nx;
            if (in_bounds(ny, nx, h, w) && !is_wall[ni] && !is_vacuum[ni]) {
                sum_atm += atmosphere[ni];
                count += 1.0f;
            }
        }
        const float P = (count > 0.0f) ? (sum_atm / count) : 0.0f;

        // W: wind magnitude from the SHARED wind field (= -grad p incl. waves, so
        // a grenade shockwave is a transient spike here -> firestorm / blow-out).
        const float W = std::sqrt(wind_x[i] * wind_x[i] + wind_y[i] * wind_y[i]);

        // Gates.
        const float hot = clamp01((T - p.fire_T_ext) / p.fire_T_span);
        const float o2  = smoothstep(p.P_min, p.P_full, P);     // pressure AS oxygen
        const float avail = F * o2;

        // Signed logistic update, fanned + stripped by wind.
        const float grow = p.k_grow * avail * hot * I * (1.0f - I)
                           * (1.0f + p.k_wind_fan * W);
        const float die  = p.k_die * (1.0f - avail * hot) * I
                           + p.k_wind_strip * W * (1.0f - I) * I;

        float I_next = clamp01(I + dt * (grow - die));
        if (I_next < p.I_min) I_next = 0.0f;    // snap-extinguish
        fire[i] = I_next;
    }

    // --- Own-tile plume pressure DEPOSIT (fire_design_proposal §3) ---
    // Replaces the deleted O2-consumption subtraction. Each burning tile adds a
    // small SELF-LIMITING overpressure to its OWN index (order-independent write),
    // so wind = -grad p points OUTWARD -> the plume/smoke is pushed AWAY (not
    // sucked in). Sealed-room over-pressure feeds the existing burst-wall relief.
    for (int i = 0; i < n; ++i) {
        const float I = fire[i];
        if (I <= 0.0f) continue;
        float gain = p.fire_pressure_gain * I
                     * (1.0f - atmosphere[i] / p.p_expand_ref) * dt;
        if (gain > 0.0f) atmosphere[i] += gain;
    }

    // --- Fire produces smoke in neighbouring air tiles (KEPT) ---
    // S2b: smoke is Q16.16. The emission delta (smoke_emission*dt*I, a small
    // positive float) is QUANTIZED to Q16.16 (round-to-nearest) and integer-added
    // — order-free + deterministic. FLOAT BRIDGE: the rate stays float until the
    // fire system itself migrates (the brief leaves the fire bridge open).
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float I = fire[y * w + x];
            if (I <= 0.0f) continue;
            const int32_t delta_q = fixedpoint::quantize((double)p.smoke_emission
                                                         * (double)dt * (double)I);
            for (const auto& d : D4) {
                int ny = y + d[0], nx = x + d[1];
                if (in_bounds(ny, nx, h, w) && !is_wall[ny * w + nx]) {
                    smoke[ny * w + nx] += delta_q;     // Q16.16 integer add
                }
            }
        }
    }

    // --- Fire damages walls, collect destroyed tiles (KEPT — the fuel-consumption
    //     brake: as wall_hp -> 0, F -> 0 in the feedback, the fire starves) ---
    std::vector<std::pair<int, int>> destroyed;
    for (int i = 0; i < n; ++i) {
        if (fire[i] > 0.0f) {
            wall_hp[i] -= p.wall_damage * dt * fire[i];
            if (wall_hp[i] <= 0.0f && flammable[i] && is_wall[i]) {
                destroyed.push_back({i / w, i % w});
                fire[i] = 0.0f;
            }
        }
    }

    // --- Final clamp (fire and smoke only — atmosphere is unclamped) ---
    // S2b: smoke clamps to [0, FP_ONE] (Q16.16 [0,1]); fire stays float.
    for (int i = 0; i < n; ++i) {
        fire[i] = std::clamp(fire[i], 0.0f, 1.0f);
        if (smoke[i] < 0) smoke[i] = 0;
        else if (smoke[i] > fixedpoint::FP_ONE) smoke[i] = fixedpoint::FP_ONE;
    }

    return destroyed;
}
