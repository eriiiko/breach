// ============================================================================
// #29 P6.2m -- TEST-ONLY: the walk's library functions (cuda_unit_lib.cuh's
// P6.2m block) on synthetic argument rows, so tests/cuda_unit_movement_check.py
// (part LIB) holds each to its Python over the whole proved range, apart from
// the scenes. One thread per argument row; see cuda_unit_kernels.h for the ops.
//
// CLASS T, test-only: nothing in the played tick calls this TU.
// ============================================================================
#include <string>
#include <vector>

#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

__global__ void k_movement_probe(int op, const int64_t* args, int n_rows, int64_t* out,
                                 const int8_t* material, int h, int w, const int64_t* mob,
                                 int n_mat) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_rows) return;
    const int64_t* a = args + (size_t)i * UNIT_MOVEMENT_PROBE_ARGS;
    int64_t* o = out + (size_t)i * UNIT_MOVEMENT_PROBE_OUT;
    bool bad = false;
    switch (op) {
    case 0:
        o[0] = default_speed_q(a[0], a[1], a[2], a[3], a[4], &bad);
        break;
    case 1:
        o[0] = move_speed_q16(a[0], a[1], a[2], a[3], a[4], a[5], &bad);
        break;
    case 2:
        o[0] = effort_scale_q16(a[0], a[1]);
        break;
    case 3:
        o[0] = facing_cos_q16(a[0], a[1], a[2], &bad);
        break;
    case 4: {
        const MobilityMap m{material, h, w, mob, n_mat};
        int64_t n = 0, total = 0;
        o[0] = passable_block(m, a[0], a[1], a[2], &bad) ? 1 : 0;
        footprint_mobility(m, a[0], a[1], a[2], &n, &total, &bad);
        o[1] = n;
        o[2] = total;
        break;
    }
    default:
        bad = true;
    }
    o[3] = bad ? 1 : 0;
}

template <class T>
T* dev_copy(const T* host, size_t n) {
    T* d = nullptr;
    if (cudaMalloc(&d, sizeof(T) * (n ? n : 1)) != cudaSuccess)
        throw std::runtime_error("unit_movement_probe: cudaMalloc failed");
    if (n && cudaMemcpy(d, host, sizeof(T) * n, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("unit_movement_probe: cudaMemcpy failed");
    return d;
}

}  // namespace

void unit_movement_probe(int op, const int64_t* args, int n_rows, const int8_t* material,
                         int h, int w, const int64_t* mobility, int n_mat, int64_t* out) {
    if (n_rows < 1) return;
    int64_t* d_args = dev_copy(args, (size_t)n_rows * UNIT_MOVEMENT_PROBE_ARGS);
    int8_t* d_mat = dev_copy(material, material ? (size_t)h * w : 0);
    int64_t* d_mob = dev_copy(mobility, mobility ? (size_t)n_mat : 0);
    int64_t* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int64_t) * (size_t)n_rows * UNIT_MOVEMENT_PROBE_OUT);
    k_movement_probe<<<(n_rows + 127) / 128, 128>>>(op, d_args, n_rows, d_out, d_mat, h, w,
                                                    d_mob, n_mat);
    const cudaError_t e = cudaGetLastError();
    cudaMemcpy(out, d_out, sizeof(int64_t) * (size_t)n_rows * UNIT_MOVEMENT_PROBE_OUT,
               cudaMemcpyDeviceToHost);
    cudaFree(d_args);
    cudaFree(d_mat);
    cudaFree(d_mob);
    cudaFree(d_out);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in unit_movement_probe: ") +
                                 cudaGetErrorString(e));
}

}  // namespace breach_cuda
