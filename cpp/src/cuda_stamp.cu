// ============================================================================
// #29 P4b — THE STAMP KERNEL (cuda_resident.h's stamp_launch_resident). The
// device twin of PhysicsEngine::stamp_units (physics_engine.cpp — THE oracle;
// the Python reference is GameMap._stamp_units_python, held to it byte for
// byte by tests/test_stamp_units_cpp_ab.py): slot 6's FIELD REBUILD of the five
// stamp outputs from the living units' footprint rows.
//
//   a. RESET every plane to its static baseline:
//        obstacles          = (permeability <= 0)
//        dyn_permeability   = permeability
//        dyn_wave_absorb    = wave_absorb
//        dyn_heat_atten_q   = heat_atten_q
//        dyn_light_atten_q  = light_atten_q                    (x3 channels)
//   b. STAMP one row per (living unit, in-bounds footprint tile), in the
//      host's row order (unit order, then occupied_tiles order):
//        dyn_permeability[idx]  = min-vs-STATIC: perm < sp ? perm : sp
//                                 -- a WRITE, not a running min: where two
//                                 bodies overlap, the LAST row's value stands
//        dyn_wave_absorb[idx]   = max(running, wabsorb)
//        dyn_heat_atten_q[idx]  = max(running, heat_q)
//        dyn_light_atten_q[idx] = per-channel max(running, light_q[c])
//
// THE ORDER (the one subtle part): the three MAXes commute, so any thread
// order gives the host's bytes; the permeability WRITE does not. The device
// keeps "the last row wins" exactly: kernel 2 atomicMax-es the row index into
// a per-cell winner plane, kernel 3 lets only the winning row write.
//
// NO float ARITHMETIC: floats here are only compared and copied (the float
// maximum is a compare-and-swap loop on the bit pattern; the permeability
// min-vs-static is the host's `(up < sp) ? up : sp`). The device is bit
// identical by construction; the gate (tests/cuda_stamp_check.py and the
// every-tick plane check in tests/cuda_s8a_check.py) holds it at tol 0.
//
// ROWS: an (R, STAMP_ROW_COLS) int32 matrix, one H2D per tick:
//   [env, y, x, perm (float bits), wabsorb (float bits), heat_q, l0, l1, l2]
// The host already filtered the living units and the in-bounds tiles; the
// core still masks a row outside its env's grid (an illegal row is skipped,
// never an out-of-bounds write -- the sweep core's idiom).
// ============================================================================
#include "cuda_resident.h"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include "cuda_batch.cuh"   // #29 P5: the tick stream

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in stamp/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// The host's `cur if cur >= v else v` as a lock-free maximum on a float cell.
// (Equal values, including +0 / -0, leave the cell as it is, as the host does.)
__device__ inline void atomic_max_f32(float* addr, float v) {
    int* ai = reinterpret_cast<int*>(addr);
    int old = *ai;
    while (true) {
        if (__int_as_float(old) >= v) return;
        const int assumed = old;
        old = atomicCAS(ai, assumed, __float_as_int(v));
        if (old == assumed) return;
    }
}

// K1 -- the reset, one pass over N*h*w, and the winner plane cleared.
__global__ void stamp_reset(
        bool* __restrict__ obstacles, float* __restrict__ dyn_perm,
        float* __restrict__ dyn_wabs, int32_t* __restrict__ dyn_heat,
        int32_t* __restrict__ dyn_light, int32_t* __restrict__ winner,
        const float* __restrict__ perm, const float* __restrict__ wabs,
        const int32_t* __restrict__ heat, const int32_t* __restrict__ light,
        int total) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        obstacles[i] = (perm[i] <= 0.0f);
        dyn_perm[i] = perm[i];
        dyn_wabs[i] = wabs[i];
        dyn_heat[i] = heat[i];
        dyn_light[(size_t)i * 3 + 0] = light[(size_t)i * 3 + 0];
        dyn_light[(size_t)i * 3 + 1] = light[(size_t)i * 3 + 1];
        dyn_light[(size_t)i * 3 + 2] = light[(size_t)i * 3 + 2];
        winner[i] = -1;
    }
}

// K2 -- the commutative MAXes, and the last row to touch each cell.
__global__ void stamp_rows_max(
        float* __restrict__ dyn_wabs, int32_t* __restrict__ dyn_heat,
        int32_t* __restrict__ dyn_light, int32_t* __restrict__ winner,
        const int32_t* __restrict__ rows, int n_rows, int n_env, int h, int w) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rows;
         r += gridDim.x * blockDim.x) {
        const int32_t* row = rows + (size_t)r * STAMP_ROW_COLS;
        const int env = row[0], y = row[1], x = row[2];
        if (env < 0 || env >= n_env || y < 0 || y >= h || x < 0 || x >= w) continue;
        const size_t idx = ((size_t)env * h + y) * w + x;
        atomicMax(&winner[idx], r);
        atomic_max_f32(&dyn_wabs[idx], __int_as_float(row[4]));
        atomicMax(&dyn_heat[idx], row[5]);
        atomicMax(&dyn_light[idx * 3 + 0], row[6]);
        atomicMax(&dyn_light[idx * 3 + 1], row[7]);
        atomicMax(&dyn_light[idx * 3 + 2], row[8]);
    }
}

// K3 -- the permeability write, by the row that came last in the host's order.
__global__ void stamp_perm_write(
        float* __restrict__ dyn_perm, const int32_t* __restrict__ winner,
        const float* __restrict__ perm, const int32_t* __restrict__ rows,
        int n_rows, int n_env, int h, int w) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rows;
         r += gridDim.x * blockDim.x) {
        const int32_t* row = rows + (size_t)r * STAMP_ROW_COLS;
        const int env = row[0], y = row[1], x = row[2];
        if (env < 0 || env >= n_env || y < 0 || y >= h || x < 0 || x >= w) continue;
        const size_t idx = ((size_t)env * h + y) * w + x;
        if (winner[idx] != r) continue;
        const float sp = perm[idx];
        const float up = __int_as_float(row[3]);
        dyn_perm[idx] = (up < sp) ? up : sp;
    }
}

}  // namespace

int stamp_launch_resident(
        int n_env, int h, int w, int n_rows,
        const float* d_permeability, const float* d_wave_absorb,
        const int32_t* d_heat_atten_q, const int32_t* d_light_atten_q,
        bool* d_obstacles, float* d_dyn_perm, float* d_dyn_wabs,
        int32_t* d_dyn_heat, int32_t* d_dyn_light,
        const int32_t* d_rows, int32_t* d_winner) {
    if (n_env <= 0 || h <= 0 || w <= 0) return 0;
    const int total = n_env * h * w;
    const int block = 256;
    int launches = 0;
    int grid = (total + block - 1) / block;
    stamp_reset<<<grid, block, 0, tstream()>>>(d_obstacles, d_dyn_perm, d_dyn_wabs, d_dyn_heat,
                                 d_dyn_light, d_winner, d_permeability,
                                 d_wave_absorb, d_heat_atten_q, d_light_atten_q,
                                 total);
    cuda_check(cudaGetLastError(), "reset launch");
    ++launches;
    if (n_rows > 0) {
        grid = (n_rows + block - 1) / block;
        stamp_rows_max<<<grid, block, 0, tstream()>>>(d_dyn_wabs, d_dyn_heat, d_dyn_light,
                                        d_winner, d_rows, n_rows, n_env, h, w);
        cuda_check(cudaGetLastError(), "rows max launch");
        stamp_perm_write<<<grid, block, 0, tstream()>>>(d_dyn_perm, d_winner, d_permeability,
                                          d_rows, n_rows, n_env, h, w);
        cuda_check(cudaGetLastError(), "perm write launch");
        launches += 2;
    }
    return launches;
}

}  // namespace breach_cuda
