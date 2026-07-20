#pragma once
// ============================================================================
// CUDA-S8a spike — the residency foundation de-risk (raw device pointer in).
// ============================================================================
//
// S8a residency (docs/cuda_s8a_residency_spec_2026-07-19.md) keeps the synced
// physics fields GPU-resident across the whole tick. The fields are CuPy
// arrays, Python-owned; the breach `.pyd` receives their device address as
// `int(arr.data.ptr)` and launches kernels DIRECTLY on that CuPy-owned memory
// (no cudaMalloc / H2D / D2H). This TU proves the ONE primitive the whole
// scheme rests on, before any solver is refactored:
//
//   a kernel handed a CuPy array's device pointer (as uintptr_t through pybind)
//   mutates the CuPy-owned device memory IN PLACE — CuPy + the breach `.pyd`
//   share the one CUDA primary context, so the address is valid in both.
//
// `spike_add1(uintptr_t dev_ptr, int n)` reinterpret_casts the raw address to
// int32_t* and adds 1 to each of the n elements. The Python test allocates a
// CuPy int32 array, passes `int(arr.data.ptr)`, and asserts the in-place +1 is
// visible on the CuPy side — the exact contract STEP B–F launch cores use.
//
// Plain C++ declaration header (no CUDA types) so bindings.cpp (compiled by
// cl.exe even in the CUDA build) can include it; cuda_spike.cu provides the
// definition. Compiled only when BREACH_CUDA. See spec §"Proven foundation".
#include <cstdint>

namespace breach_cuda {

// Add 1 to each of the first `n` int32 elements at device address `dev_ptr`,
// IN PLACE, on the GPU. `dev_ptr` is a raw CUDA device address (typically a
// CuPy array's `int(arr.data.ptr)`) passed through pybind as uintptr_t. Throws
// std::runtime_error on any CUDA error so pybind surfaces it as a Python
// exception. No malloc, no transfer — the residency primitive in miniature.
void spike_add1(std::uintptr_t dev_ptr, int n);

}  // namespace breach_cuda
