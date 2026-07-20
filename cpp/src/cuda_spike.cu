// ============================================================================
// CUDA-S8a spike implementation — see cuda_spike.h for the rationale.
// ============================================================================
#include "cuda_spike.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {
inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in spike_add1/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// The trivial residency kernel: in[i] += 1, in place, on memory OWNED by the
// caller (a CuPy array in the S8a scheme). Grid-stride loop -> correct for any
// <<<grid,block>>>. This is the same reinterpret_cast<int32_t*>(dev_ptr)
// pattern every STEP B launch core will use on the resident fields.
__global__ void spike_add1_kernel(int32_t* __restrict__ p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        p[i] += 1;
    }
}
}  // namespace

void spike_add1(std::uintptr_t dev_ptr, int n) {
    if (n <= 0) return;
    // Reinterpret the raw CuPy device address as an int32_t* and launch on it
    // DIRECTLY — no cudaMalloc, no cudaMemcpy. The whole S8a bet is that this
    // address (from int(arr.data.ptr)) is a valid device pointer in the breach
    // context because CuPy and the .pyd share the one CUDA primary context.
    int32_t* p = reinterpret_cast<int32_t*>(dev_ptr);
    const int block = 256;
    const int grid = (n + block - 1) / block;
    spike_add1_kernel<<<grid, block>>>(p, n);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

}  // namespace breach_cuda
