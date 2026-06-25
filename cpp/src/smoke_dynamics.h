#pragma once
// Smoke dynamics — diffusion + advection by precomputed wind field.
// Wind is computed by the AtmosphereSolver (gradient of atmosphere + wave_p).
//
// S2b (fixed-point arc, docs/s2_fixed_point_plan.md §S2b): the smoke + 5 gas
// planes are now int32 Q16.16 (scale 2^16, shared with water/heat). The advection
// is the INTEGER semi-Lagrangian ("SLint") ported verbatim from the proven
// prototype tools/s2_advection_demo/advection_demo.py (commit ceb601b, branch
// s2-advection-demo): a sqrt-free DDA wall-clip march, an integer bilinear sample
// (int64 corner-weight accumulate -> narrow), and the renorm 1/wsum via the
// shared fixedpoint::reciprocal_q16 Newton reciprocal. NON-CONSERVATIVE by design
// (the >>16 truncation is a gentle built-in decay) — accepted (Q-S2-1): it is
// deterministic on every machine, so it is behaviour, not desync; smoke decay is
// the tuning knob. NO flux form, NO limiter, NO outflow clamp.
//
// FLOAT BRIDGE (until S2c): the `wind` is still produced by the float atmosphere
// (S2c not yet integer), so step() reads float wind_x/wind_y and quantizes the
// wind*dt_adv DISPLACEMENT to Q16.16 at the boundary. After S2c the wind is
// integer and this bridge closes (the displacement is computed integer-only).

#include <cstdint>
#include <vector>

class SmokeDynamics {
public:
    float d_smoke             = 0.4f;   // base smoke diffusion coefficient
    // Patch 2b: advection_rate's effective default is bumped ×dt_scale² (the old
    // 9× from the removed double-applied dt_scale) so the visible wind-ride is
    // preserved on the real dt. Old: 25.0 with dt_scale=3 -> effective 25·9.
    float advection_rate      = 225.0f; // advection strength by wind field (real dt now; was 25·dt_scale²)
    float wind_diffusion_scale = 0.0f;  // wind-dependent diffusion: D = d_smoke * (1 + scale * |wind|)
    float sink_strength       = 0.0f;   // smoke-side sink-pull toward nearest breach (0 = off)
    // Patch 2b: K = "vent hops/tick" — how many 1-cell sink_hop passes the engine
    // runs each tick, decoupled from the wave CFL n_wave. Default ≈ today's n_wave
    // so venting starts in a familiar place; Erik tunes by eye. (Read by
    // PhysicsEngine::run_substeps; a config dial: [physics] smoke_vent_hops.)
    int   vent_hops           = 16;     // 0 = no venting

    // Patch 2b: the smoke advection step is now WIND-ONLY (the breach sink-pull
    // is no longer fused into the back-trace — it is the standalone sink_hop()
    // below, run K× by the engine). Single step of size dt on the precomputed
    // wind field from the atmosphere solver.
    void step(
        int32_t* smoke,            // S2b: int32 Q16.16 density (was float)
        const float* wind_x,       // precomputed: -d/dx(atmosphere + wave_p) — FLOAT BRIDGE until S2c
        const float* wind_y,       // precomputed: -d/dy(atmosphere + wave_p) — FLOAT BRIDGE until S2c
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
        int h, int w,
        float dt
    ) const;

    // Patch 2b: ONE 1-cell BFS-gradient pull toward the nearest breach — the
    // exact sink mechanism formerly fused into step()'s back-trace, extracted as
    // a standalone pass the engine runs K times per tick (decoupled from the wave
    // CFL). The back-trace velocity is the sink direction ONLY, CAPPED at one
    // cell (LOAD-BEARING: sink_x/sink_y is a next-hop unit vector down the BFS
    // shortest path to the breach, so a multi-cell hop would cut a corner into a
    // wall and stall — one cell per hop walks the path). Permeability-aware
    // bilinear sample (a breach corner contributes 0, draining smoke to space);
    // zeroed on walls/vacuum. No dt: each call is exactly one hop. With no breach
    // the sink field is all-zero everywhere, so this is the identity (sealed
    // rooms are untouched — matches the old fused behaviour with sink off).
    void sink_hop(
        int32_t* smoke,            // S2b: int32 Q16.16 density (was float)
        const float* sink_x,       // smoke-side sink direction toward nearest breach (unit-ish, 0 if none)
        const float* sink_y,
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
        int h, int w
    ) const;

private:
    // Reused per-step scratch (GPU-prep: no per-step alloc). `mutable` so the
    // const step() can use them (temperature_solver idiom). S2b: int32 Q16.16.
    //   lap_ — diffusion Laplacian (the wind-coupled 4-neighbour stencil sum),
    //          carried as an int64 face-gather narrowed once per cell -> stored
    //          Q16.16; FULLY overwritten each step before read.
    //   src_ — pre-advection snapshot (a COPY of the int32 smoke). Read by the
    //          branchy back-trace loop that writes smoke; the SWAP idiom retains
    //          its storage across steps (no per-step alloc) and it is re-copied
    //          from smoke each step.
    mutable std::vector<int32_t> lap_;
    mutable std::vector<int32_t> src_;
};
