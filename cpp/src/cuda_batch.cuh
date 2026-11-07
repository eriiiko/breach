#pragma once
// ============================================================================
// #29 P5 — THE BATCHED RESIDENT ENGINE's device-side launch shape
// (docs/residency_batching_design_2026-11-07.md §4.2).
// ============================================================================
//
// Every resident launch core runs over an (N, h, w) batch of homogeneous envs
// (one grid shape, one boundary type, one config). The launch shape is TWO-
// dimensional: blockIdx.y is the ENV, the x dimension is the kernel's own
// grid-stride loop over that env's cells — so a kernel body written for one
// (h, w) grid stays VERBATIM: it rebases its pointers by the env's stride at
// entry (bptr) and every stencil read stays inside the env's own plane. Never
// a tall grid (a stencil would read across an env boundary).
//
// N = 1 is the played single-instance game (`--cuda --resident`): blockIdx.y
// is 0, every rebase adds 0, and the kernels run byte for byte as before.
//
// Per-env scalars that differ across envs are only the ones the substep count
// selects (the FOLD TABLE row, EnvSub); a masked env (its substep index past
// its own n_sub) returns at the top of the kernel and does exactly nothing —
// the whole block belongs to one env, so the return is block-uniform and every
// __syncthreads() in a kernel is still reached by every thread that runs.
//
// Device-only header (CUDA types); included by the .cu TUs only.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "cuda_resident_tick.h"   // tick_stream_handle (plain C++)

namespace breach_cuda {

// THE ONE STREAM every resident launch core and every in-tick copy / memset
// runs on (design §4.2 "Streams"). Created once per process; a BLOCKING
// stream (cudaStreamCreate's default flags), so work the legacy default
// stream issues (CuPy's per-env from_host / to_host copies, the per-call
// twins) stays ordered with the tick without an explicit event.
inline cudaStream_t tstream() {
    return static_cast<cudaStream_t>(tick_stream_handle());
}

// EnvSub (one fold-table row per env) is declared in cuda_resident_tick.h.

// The env of this block.
__device__ __forceinline__ int benv() { return (int)blockIdx.y; }

// Rebase a per-env pointer by the env's stride (nullptr stays nullptr).
template <class T>
__device__ __forceinline__ T* bptr(T* p, size_t stride) {
    return p ? p + (size_t)blockIdx.y * stride : p;
}

// True iff this env sits out substep `s` (its own n_sub is smaller than the
// batch's maximum). es == nullptr: an unbatched / unmasked launch.
__device__ __forceinline__ bool sub_off(const EnvSub* es, int s) {
    return es != nullptr && s >= es[blockIdx.y].n_sub;
}

// True iff this env is switched off by a per-env byte mask (nullptr: all on).
__device__ __forceinline__ bool env_off(const uint8_t* active) {
    return active != nullptr && active[blockIdx.y] == 0;
}

// The (N, h, w) launch grid: x covers one env's cells, y the envs.
inline dim3 bgrid(int n_cells, int n_env, int block = 256) {
    return dim3((unsigned)((n_cells + block - 1) / block), (unsigned)(n_env > 0 ? n_env : 1));
}

}  // namespace breach_cuda
