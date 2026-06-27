#pragma once
// ============================================================================
// CUDA-S0 hello-world — the toolchain + bit-identity de-risk (no GPU physics).
// ============================================================================
//
// This is the FIRST CUDA translation unit in Breach. It exists to prove, on the
// real hardware, three things before a single physics kernel is written:
//   1. nvcc compiles `fixed_point.h`'s integer toolkit for the DEVICE (the
//      __host__ __device__ annotation works; the kit is GPU-clean).
//   2. The host<->device round-trip plumbing works (cudaMalloc / cudaMemcpy /
//      launch / sync / copy-back) and links into the pybind module.
//   3. A trivial integer map computed ON THE GPU is BIT-IDENTICAL to the same
//      op on the CPU — the determinism contract, demonstrated end-to-end.
//
// It is compiled ONLY when CMake is configured with -DBREACH_CUDA=ON (which
// defines BREACH_HAS_CUDA); the default CPU build never sees it, so the running
// game is untouched. See docs/cuda_migration_plan.md §7.4 (S0).
#include <cstdint>
#include <string>
#include <vector>

namespace breach_cuda {

// True iff at least one CUDA device is present and queryable. Never throws.
bool available();

// Human-readable device line: "<name> | sm_<cc> | runtime <v> | driver <v>".
// Throws std::runtime_error if no device / query fails.
std::string device_info();

// THE S0 GATE: out[i] = fixedpoint::mul_q16(in[i], factor_q16), computed on the
// GPU through the shared __host__ __device__ toolkit, round-tripped host->
// device->host. Bit-identical to the CPU mul_q16 by construction (same two's-
// complement multiply, same >>16 arithmetic truncation). Throws on any CUDA
// error so pybind surfaces it as a Python exception.
std::vector<int32_t> map_mul_q16(const std::vector<int32_t>& in,
                                 int32_t factor_q16);

}  // namespace breach_cuda
