#pragma once
// Fire simulation — signed-logistic intensity FEEDBACK, smoke emission, plume
// pressure deposit, wall burn-through (engine/06 §5 + fire_design_proposal §2/§3).
//
// Spread is NO LONGER cellular: it comes entirely from radiation -> heat ->
// temperature -> ignition (apply_temperature_ignition, wired in the sim). This
// step is purely the per-tile life/death of an ALREADY-lit fire:
//
//   T     = temperature[i]                         (Q16.16; temp_scale == FP_ONE)
//   F     = clamp01(wall_hp[i] / fuel_ref)         (fuel from remaining wall HP)
//   P     = mean atmosphere over OPEN (non-solid, non-vacuum) 4-neighbours
//   W     = sqrt(wind_x^2 + wind_y^2)              (the SHARED wind field)
//   hot   = clamp01((T - T_ext) / T_span)
//   o2    = smoothstep(P_min, P_full, P)           (pressure IS the O2 proxy)
//   avail = F * o2
//   grow  = k_grow * avail * hot * I * (1-I) * (1 + k_wind_fan * W)
//   die   = k_die * (1 - avail*hot) * I  +  k_wind_strip * W * (1-I) * I
//   I    += dt * (grow - die);  clamp01;  snap to 0 below I_min
//
// Pressure (replaces the old O2-consumption subtraction, which sucked smoke IN):
//   atmosphere[i] += max(fire_pressure_gain * I * (1 - atmosphere[i]/p_expand_ref) * dt, 0)
// An OWN-tile overpressure -> wind = -grad p points OUTWARD -> smoke pushed away.
// The sustain read P is the NEIGHBOUR mean, so the fire reads incoming fresh air,
// not its own bump.
//
// Determinism (S3b): the whole logistic is INTEGER Q16.16 (fire/wall_hp int32,
// atmosphere/wind/temperature int32). Cross-machine bit-identical: integer
// +/-/*/>> are exact + associative, and the per-cell sqrt is a fixed-iteration
// floor-isqrt (fixed_point.h::sqrt_q16, the arc's first per-cell transcendental).
// The multiply tree order is PINNED (left-fold mul_q16); the plume + smoke-emission
// + wall-burn deposits ROUND-TO-NEAREST (unbiased sources). The discrete outputs
// (the I_min extinguish flip, the wall_hp<=0 burn-through list) are integer compares
// -> bit-deterministic. Plume deposit is an own-index write -> order-independent.

#include <vector>
#include <utility>
#include <cstdint>

struct FireParams {
    // --- signed-logistic feedback (fire_design_proposal §2) ---
    float k_grow         = 4.0f;   // logistic growth gain (1/s)
    float k_die          = 2.0f;   // decay rate when starved/cold (1/s)
    float fire_T_ext     = 350.0f; // extinction temperature (~ignition_temp + 50)
    float fire_T_span    = 150.0f; // width of the `hot` ramp above T_ext
    float fuel_ref       = 60.0f;  // wall_hp normaliser: F = clamp01(wall_hp/fuel_ref)
    float P_min          = 0.60f;  // pressure below which the O2 proxy is 0
    float P_full         = 1.00f;  // pressure at which the O2 proxy is full
    float I_min          = 0.02f;  // snap-to-zero extinguish floor

    // --- wind coupling (fire_design_proposal §5; Erik's addition) ---
    // k_wind_fan / k_wind_strip are scaled against the shared wind field's
    // magnitude; both NEED TUNING vs the live wind scale (a shockwave is a large
    // transient spike). Defaults are deliberately gentle.
    float k_wind_fan     = 0.5f;   // (1 + k_wind_fan*W) fans growth (firestorm)
    float k_wind_strip   = 0.5f;   // W*(1-I)*I blows out small/marginal fires

    // --- plume pressure deposit (fire_design_proposal §3) ---
    float fire_pressure_gain = 0.15f; // own-tile overpressure gain (1/s)
    float p_expand_ref       = 1.30f; // self-limiting saturation ceiling

    // --- kept behaviours ---
    float smoke_emission = 0.8f;   // smoke produced per second per unit intensity
    float wall_damage    = 0.4f;   // wall HP lost per second per unit intensity
                                   //  (burn-through IS the fuel-consumption brake)

    // Q16.16 scale of the `temperature` field (== HEAT_SCALE / TEMP_SCALE). Fixed
    // at construction; exposed so Python/config and C++ never disagree.
    float temp_scale     = 65536.0f;
};

class FireSimulation {
public:
    FireParams params;

    // Returns vector of (y, x) coordinates where walls burned through.
    // Python must call destroy_wall() for each of these.
    //
    //   fire        : int32 (h, w) Q16.16 intensity in [0,1], mutated in place (S3b).
    //   atmosphere  : int32 (h, w) Q16.16 (S2c), read (neighbour mean) + own-tile
    //                 plume write (round-to-nearest deposit).
    //   smoke       : int32 (h, w) Q16.16 (S2b), fire ADDS to it (kept). The
    //                 emission delta smoke_emission*dt*I is round-to-nearest and
    //                 integer-added — order-free, deterministic.
    //   wall_hp     : int32 (h, w) Q16.16 (S3b), burn-through depletes it (the fuel
    //                 brake); fractional depletion needs the Q16.16 fraction.
    //   temperature : int32 (h, w) Q16.16, READ-ONLY (the conduction-pass field).
    //   wind_x/wind_y : int32 (h, w) Q16.16 (S2c), the SHARED wind field (= -grad p
    //                 incl. waves), READ-ONLY (the W = |wind| term, via sqrt_q16).
    //   is_wall     : bool (h, w) solid mask (a fire tile is itself solid).
    //   is_vacuum   : bool (h, w) vacuum mask (excluded from the P neighbour mean).
    //   flammable   : bool (h, w) fuel mask (fire only lives on fuel).
    std::vector<std::pair<int, int>> step(
        int32_t* fire,             // S3b: Q16.16 (was float)
        int32_t* atmosphere,       // S2c: Q16.16 (fire reads + plume-writes)
        int32_t* smoke,            // S2b: Q16.16 (fire emission round + added)
        int32_t* wall_hp,          // S3b: Q16.16 (was float)
        const int32_t* temperature,
        const int32_t* wind_x,     // S2c/S3b: Q16.16
        const int32_t* wind_y,     // S2c/S3b: Q16.16
        const bool* is_wall,
        const bool* is_vacuum,
        const bool* flammable,
        int h, int w,
        float dt
    ) const;
};
