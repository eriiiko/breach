#pragma once
// ============================================================================
// swarm_launch.cuh -- the swarm species laws' CUDA launch templates (arc #63
// P3, docs/swarm_P3_impl.md §3.6, §4.5): ONE kernel around a species' per-slot
// law functor, the per-call wrapper (host mirrors in and out, for `--cuda`
// without residency) and the resident launch core (the caller's device
// pointers, on the tick stream, no allocation, no sync).
// ============================================================================
//
// THE LAUNCH SHAPE (D19, the Batched resident engine row): (N, max_units),
// blockIdx.y is the env, the slot is blockIdx.x * blockDim.x + threadIdx.x,
// pointers rebased per env inside the law (flat index e * max_units + s,
// planes at e * h * w). Never a tall grid. A slot >= hw[e] returns at once
// (the per-env mask), so a short env's padding is never read or written.
// Both wrappers keep the zero-extent guard (n_env == 0 or max hw == 0: no
// launch -- a 0-block grid is cudaErrorInvalidConfiguration).
//
// The law writes only its own slot's 9 columns and 4 record words; T, solid,
// the launch block and the table rows are read-only; the device passes no
// branch tally (bit identity is unaffected by the CPU's counting). The
// resident kernel's two per-env counters (deaths, live slots stepped) are
// integer atomics into the books arena: an order-free sum, never a record.
//
// Device-only header (CUDA types); included by the .cu TUs only.
#include <cuda_runtime.h>
#include <cstdint>

#include "swarm.h"

namespace swarm {

template <class Law>
__global__ void swarm_kernel(Law law, SlotIO io, unsigned long long* cnt) {
    const int e = (int)blockIdx.y;
    const int s = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= (int)io.launch[e * LAUNCH_FIXED_WORDS + LAUNCH_HW]) return;
    const bool live = io.st.valid[e * io.max_units + s] != 0;
    const int died = law.step_slot(io, e, s);
    if (cnt != nullptr) {
        if (live) atomicAdd(&cnt[e * SWARM_CNT_SLOTS + SWARM_CNT_STEPPED], 1ULL);
        if (died) atomicAdd(&cnt[e * SWARM_CNT_SLOTS + SWARM_CNT_DEATHS], 1ULL);
    }
}

// The launch core: grid (ceil(launch_hw / 256), n_env), block 256.
template <class Law>
void launch_core(const Law& law, const SlotIO& io, int n_env, int launch_hw,
                 unsigned long long* cnt, cudaStream_t stream) {
    if (n_env <= 0 || launch_hw <= 0) return;
    const dim3 grid((unsigned)((launch_hw + 255) / 256), (unsigned)n_env);
    swarm_kernel<Law><<<grid, 256, 0, stream>>>(law, io, cnt);
}

// swarm.cu: the per-call staging (H2D the 9 columns whole, T, solid, the
// launch block, the table rows) and the copy back (the 9 columns whole --
// slots >= hw are never touched, so their staged copy equals the host -- and
// the records [0, hw[e]) per env only); returns the deaths counted from them.
SlotIO percall_upload(const LawIO& L);
int percall_download(const LawIO& L, const SlotIO& dev);
void percall_check(const char* what);

template <class Law>
int percall(const Law& law, const LawIO& L) {
    int launch_hw = 0;
    for (int e = 0; e < L.n_env; ++e) {
        const int hw = (int)L.launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        if (hw > launch_hw) launch_hw = hw;
    }
    if (L.n_env == 0 || launch_hw == 0) return 0;   // before any allocation or copy
    const SlotIO dev = percall_upload(L);
    launch_core(law, dev, L.n_env, launch_hw, nullptr, (cudaStream_t)0);
    percall_check("swarm per-call launch");
    return percall_download(L, dev);
}

// The resident launch core on the caller's device pointers.
template <class Law>
void resident(const Law& law, const ResidentLaunch& R) {
    launch_core(law, R.io, R.n_env, R.launch_hw, R.counters,
                static_cast<cudaStream_t>(R.stream));
}

}  // namespace swarm
