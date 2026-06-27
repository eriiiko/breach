#pragma once
// ============================================================================
// CUDA-S1 — the temperature solver on the GPU (the first real physics kernel).
// ============================================================================
//
// A faithful, bit-identical port of TemperatureSolver::step (temperature_solver.
// cpp): the three-pass heat->temperature CONVERSION (§1), CONDUCTION relaxation
// (§2, double-buffered gather), and ambient COOLING (§3, vacuum-exposed 4x). All
// three passes are pure integer Q16.16 (two's-complement +,-,>>), so the GPU
// result is byte-for-byte identical to the CPU on every architecture — the whole
// point of S1.
//
// Plain C++ declaration header (no CUDA types) so the .cpp TUs (bindings.cpp,
// physics_engine.cpp, compiled by cl.exe even in the CUDA build) can include it;
// cuda_temperature.cu provides the definitions. Compiled only when BREACH_CUDA.
#include <cstdint>

namespace breach_cuda {

// One tick of thermal work on the GPU — IN-PLACE on `temperature`. Mirrors
// TemperatureSolver::step exactly (same args; the scalar dials are passed
// explicitly since this is a free function, and o2_vacuum_thresh is quantized
// ONCE on the host with fixedpoint::quantize — the identical boundary cast the
// CPU does). face_shift is (h,w,4) int32, dir order N,S,E,W.
void temperature_step(
    int32_t* temperature,           // Q16.16 (h,w) — in/out
    const int32_t* heat,            // Q16.16 (h,w) — per-tick deposit (read)
    const int32_t* heat_inv_shift,  // (h,w) per-tile log2(thermal_mass)
    const int32_t* face_shift,      // (h,w,4) per-tile face shifts (N,S,E,W)
    const bool* solid,              // (h,w) physics solid mask
    const bool* is_vacuum,          // (h,w) physics vacuum mask
    const int32_t* atmosphere,      // Q16.16 (h,w) — exposure test
    int no_face,                    // sentinel: face_shift==no_face -> skip
    int cool_shift,                 // interior cooling shift
    int cool_shift_vacuum,          // space-exposed cooling shift (faster)
    float o2_vacuum_thresh,         // config dial (quantized on host)
    int h, int w);

// Backend selection (S1 gate + integration). When true, PhysicsEngine::step_tail
// runs temperature on the GPU instead of the CPU solver. Defaults false so the
// game + suite run on the CPU path unchanged until explicitly switched.
bool temperature_backend_is_cuda();
void set_temperature_backend_cuda(bool on);

}  // namespace breach_cuda
