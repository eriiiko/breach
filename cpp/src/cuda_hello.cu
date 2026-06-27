// ============================================================================
// CUDA-S0 hello-world implementation — see cuda_hello.h for the rationale.
// ============================================================================
#include "cuda_hello.h"
#include "fixed_point.h"   // the __host__ __device__ integer toolkit (S0 annot.)

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {
// Minimal CUDA error check: throw on failure so pybind raises a Python
// exception (the S0 harness asserts the call does not throw). Kept local — the
// production kernels (S1+) will share a real checked-launch helper.
inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in " << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}
}  // namespace

// The trivial map kernel: out[i] = mul_q16(in[i], factor). One thread per
// element via a grid-stride loop (correct for ANY <<<grid,block>>>), calling
// fixedpoint::mul_q16 — the SAME integer op the CPU uses, now device-compiled.
// This is the proof the toolkit is __device__-clean and the math is identical.
__global__ void map_mul_q16_kernel(const int32_t* __restrict__ in,
                                   int32_t* __restrict__ out, int n,
                                   int32_t factor) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        out[i] = fixedpoint::mul_q16(in[i], factor);
    }
}

bool available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

std::string device_info() {
    int dev = 0;
    cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::ostringstream os;
    os << prop.name << " | sm_" << prop.major << prop.minor
       << " | runtime " << rt << " | driver " << drv;
    return os.str();
}

std::vector<int32_t> map_mul_q16(const std::vector<int32_t>& in,
                                 int32_t factor) {
    const int n = static_cast<int>(in.size());
    std::vector<int32_t> out(in.size());
    if (n == 0) return out;

    int32_t* d_in = nullptr;
    int32_t* d_out = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(int32_t);
    cuda_check(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    cuda_check(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
    cuda_check(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    const int block = 256;
    const int grid = (n + block - 1) / block;
    map_mul_q16_kernel<<<grid, block>>>(d_in, d_out, n, factor);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cuda_check(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");
    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

}  // namespace breach_cuda
