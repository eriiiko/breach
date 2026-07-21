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
// S2c: the `wind` is now Q16.16 int32 (= -grad(atm+wave_p), computed by the
// atmosphere solver) — the wind FLOAT BRIDGE is COLLAPSED. step() reads int32
// wind_x/wind_y: the advection displacement -wind·dt_adv is an integer multiply
// (dt_adv folded once in double, then quantized) and the wind-coupled diffusion
// |wind|² is an integer square (Q.32) dequantized once per cell to the positive
// d_eff·dt scalar coefficient. No per-cell float on the wind path.

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
    // (sink_strength / vent_hops / sink_hop() DELETED — EOS refactor P3,
    // decisions.md #3: venting is native to the compressible solver; smoke
    // rides the real venting wind out of a breach.)

    // The smoke advection step is WIND-ONLY. Single step of size dt on the
    // solver-owned velocity field (EOS P3: eos_solver's corrected u).
    void step(
        int32_t* smoke,            // S2b: int32 Q16.16 density (was float)
        const int32_t* wind_x,     // S2c: Q16.16 int32 = -d/dx(atmosphere + wave_p)
        const int32_t* wind_y,     // S2c: Q16.16 int32 = -d/dy(atmosphere + wave_p)
        const bool* obstacles,
        const bool* is_wall,
        const bool* is_vacuum,
        const float* permeability,
        int h, int w,
        float dt,
        // BC (boundary_conditions_spec_2026-07-19 §1): the planetside AMBIENT
        // ring is a trace SINK (traces reset to 0 there, absorbed — the
        // vacuum-breach idiom). nullptr on space maps -> byte-identical (the
        // widen is dormant by branch). Default keeps the direct-binding test
        // path + any other caller unchanged.
        const bool* is_ambient = nullptr
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
