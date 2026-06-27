// ============================================================================
// CUDA-S1 temperature solver implementation — see cuda_temperature.h.
// A bit-identical GPU port of TemperatureSolver::step (temperature_solver.cpp).
// ============================================================================
#include "cuda_temperature.h"
#include "fixed_point.h"   // quantize() for the o2_vacuum_thresh boundary cast

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in temperature_step/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// Direction order MUST match temperature_solver.cpp: 0=N,1=S,2=E,3=W.
__device__ __forceinline__ int dy_of(int d) {
    // {-1, +1, 0, 0}
    return (d == 0) ? -1 : (d == 1) ? 1 : 0;
}
__device__ __forceinline__ int dx_of(int d) {
    // {0, 0, +1, -1}
    return (d == 2) ? 1 : (d == 3) ? -1 : 0;
}

// ---- Pass 1: heat -> temperature conversion (§1.2, solids only) ------------
// Mirrors the CPU loop + heat_saturating_add (raycaster.h) exactly. Each thread
// owns one cell -> writes only temperature[i], no race.
__global__ void temp_convert(int32_t* __restrict__ temperature,
                             const int32_t* __restrict__ heat,
                             const int32_t* __restrict__ heat_inv_shift,
                             const bool* __restrict__ solid, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!solid[i]) continue;
        const int32_t deposit = heat[i];
        if (deposit <= 0) continue;
        const int shift = heat_inv_shift[i];
        const int32_t gain = deposit >> shift;       // Q16.16 / 2^shift
        if (gain <= 0) continue;                      // heat_saturating_add: delta<=0
        const int32_t cell = temperature[i];
        // saturating add: pin at INT32_MAX rather than wrap.
        temperature[i] = (cell > (int32_t)0x7fffffff - gain)
                         ? (int32_t)0x7fffffff
                         : (cell + gain);
    }
}

// ---- Pass 2: conduction relaxation (§2.2, gather, double-buffered) ---------
// Reads the FROZEN temperature, writes temp_new[i]. The DIFFERENCE is shifted,
// not the neighbour (equal neighbours -> exactly 0). int64 accumulator, identical
// to the CPU. Every cell (incl. air -> all NO_FACE -> acc=0 -> temp_new=ti) is
// fully written, so temp_new has no uninitialised read (scratch hygiene).
__global__ void temp_conduct(const int32_t* __restrict__ temperature,
                             int32_t* __restrict__ temp_new,
                             const int32_t* __restrict__ face_shift,
                             int no_face, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / w;
        const int x = i % w;
        const int32_t* fs = &face_shift[i * 4];
        const int32_t ti = temperature[i];
        long long acc = 0;
        for (int d = 0; d < 4; ++d) {
            const int s = fs[d];
            if (s == no_face) continue;
            const int ny = y + dy_of(d);
            const int nx = x + dx_of(d);
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
            const int32_t tn = temperature[ny * w + nx];
            acc += (long long)(tn - ti) >> s;         // arithmetic shift
        }
        temp_new[i] = (int32_t)((long long)ti + acc);
    }
}

// ---- Pass 3: ambient cooling (§3, solids only, vacuum-exposed 4x) ----------
// In-place on temperature[i]; reads own cell + neighbours' is_vacuum/atmosphere
// (frozen -> safe). Symmetric round-toward-0 shift; the dead-band is preserved.
__global__ void temp_cool(int32_t* __restrict__ temperature,
                          const bool* __restrict__ solid,
                          const bool* __restrict__ is_vacuum,
                          const int32_t* __restrict__ atmosphere,
                          int cool_shift, int cool_shift_vacuum,
                          int32_t thresh_q, int h, int w) {
    const int n = h * w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (!solid[i]) continue;
        const int32_t t = temperature[i];
        if (t == 0) continue;
        const int y = i / w;
        const int x = i % w;
        bool exposed = false;
        for (int d = 0; d < 4; ++d) {
            const int ny = y + dy_of(d);
            const int nx = x + dx_of(d);
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
            const int ni = ny * w + nx;
            if (is_vacuum[ni] || atmosphere[ni] < thresh_q) {
                exposed = true;
                break;
            }
        }
        const int shift = exposed ? cool_shift_vacuum : cool_shift;
        const int32_t loss = (t < 0) ? -((-t) >> shift) : (t >> shift);
        temperature[i] = t - loss;
    }
}

}  // namespace

void temperature_step(
    int32_t* temperature, const int32_t* heat, const int32_t* heat_inv_shift,
    const int32_t* face_shift, const bool* solid, const bool* is_vacuum,
    const int32_t* atmosphere, int no_face, int cool_shift,
    int cool_shift_vacuum, float o2_vacuum_thresh, int h, int w) {
    const int n = h * w;
    if (n <= 0) return;

    // The SAME once-per-step boundary cast the CPU does (round-to-nearest).
    const int32_t thresh_q = fixedpoint::quantize((double)o2_vacuum_thresh);

    const size_t nb = (size_t)n * sizeof(int32_t);
    const size_t nbool = (size_t)n * sizeof(bool);
    int32_t *d_temp = nullptr, *d_temp_new = nullptr, *d_heat = nullptr,
            *d_his = nullptr, *d_fs = nullptr, *d_atm = nullptr;
    bool *d_solid = nullptr, *d_vac = nullptr;
    cuda_check(cudaMalloc(&d_temp, nb), "malloc temp");
    cuda_check(cudaMalloc(&d_temp_new, nb), "malloc temp_new");
    cuda_check(cudaMalloc(&d_heat, nb), "malloc heat");
    cuda_check(cudaMalloc(&d_his, nb), "malloc heat_inv_shift");
    cuda_check(cudaMalloc(&d_fs, nb * 4), "malloc face_shift");
    cuda_check(cudaMalloc(&d_atm, nb), "malloc atmosphere");
    cuda_check(cudaMalloc(&d_solid, nbool), "malloc solid");
    cuda_check(cudaMalloc(&d_vac, nbool), "malloc is_vacuum");

    cuda_check(cudaMemcpy(d_temp, temperature, nb, cudaMemcpyHostToDevice), "H2D temp");
    cuda_check(cudaMemcpy(d_heat, heat, nb, cudaMemcpyHostToDevice), "H2D heat");
    cuda_check(cudaMemcpy(d_his, heat_inv_shift, nb, cudaMemcpyHostToDevice), "H2D his");
    cuda_check(cudaMemcpy(d_fs, face_shift, nb * 4, cudaMemcpyHostToDevice), "H2D fs");
    cuda_check(cudaMemcpy(d_atm, atmosphere, nb, cudaMemcpyHostToDevice), "H2D atm");
    cuda_check(cudaMemcpy(d_solid, solid, nbool, cudaMemcpyHostToDevice), "H2D solid");
    cuda_check(cudaMemcpy(d_vac, is_vacuum, nbool, cudaMemcpyHostToDevice), "H2D vac");

    const int block = 256;
    const int grid = (n + block - 1) / block;
    // Pass 1: convert (in-place on d_temp).
    temp_convert<<<grid, block>>>(d_temp, d_heat, d_his, d_solid, n);
    cuda_check(cudaGetLastError(), "convert launch");
    // Pass 2: conduct (d_temp -> d_temp_new), then copy back (the CPU swap).
    temp_conduct<<<grid, block>>>(d_temp, d_temp_new, d_fs, no_face, h, w);
    cuda_check(cudaGetLastError(), "conduct launch");
    cuda_check(cudaMemcpy(d_temp, d_temp_new, nb, cudaMemcpyDeviceToDevice), "D2D swap");
    // Pass 3: cool (in-place on d_temp).
    temp_cool<<<grid, block>>>(d_temp, d_solid, d_vac, d_atm,
                               cool_shift, cool_shift_vacuum, thresh_q, h, w);
    cuda_check(cudaGetLastError(), "cool launch");
    cuda_check(cudaDeviceSynchronize(), "sync");

    cuda_check(cudaMemcpy(temperature, d_temp, nb, cudaMemcpyDeviceToHost), "D2H temp");

    cudaFree(d_temp);
    cudaFree(d_temp_new);
    cudaFree(d_heat);
    cudaFree(d_his);
    cudaFree(d_fs);
    cudaFree(d_atm);
    cudaFree(d_solid);
    cudaFree(d_vac);
}

namespace {
bool g_temp_backend_cuda = false;
}
bool temperature_backend_is_cuda() { return g_temp_backend_cuda; }
void set_temperature_backend_cuda(bool on) { g_temp_backend_cuda = on; }

}  // namespace breach_cuda
