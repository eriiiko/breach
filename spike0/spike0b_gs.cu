// =============================================================================
// Spike-0b : the GS reciprocal (the other scary GPU op)
// -----------------------------------------------------------------------------
// A tiny fixed-point Red-Black Gauss-Seidel diffusion step on a small grid,
// in INTEGER Q16.16, using a PRECOMPUTED RECIPROCAL MULTIPLY (not a per-cell
// divide) -- the exact pattern the real solver will use.
//
// The diffusion update for an interior cell, with diffusion coefficient alpha
// (Q16.16), is the standard Jacobi/GS stencil:
//
//     x_new = (b + alpha * (xL + xR + xU + xD)) / (1 + 4*alpha)
//
// In fixed point we NEVER divide per cell. We precompute, ONCE on the host,
//     RECIP = round( 2^RECIP_SHIFT / (1 + 4*alpha) )           (integer)
// and replace the divide with a widened multiply + arithmetic shift:
//
//     num   = b + ((alpha * neighbour_sum) >> Q_SHIFT)         (Q16.16)
//     x_new = (int32)((num * RECIP) >> RECIP_SHIFT)            (Q16.16)
//
// All intermediate products are computed in int64 to avoid overflow, and the
// shift is an arithmetic right shift (floor toward -inf), which is identical
// on every CUDA architecture and on the CPU. That determinism is the whole
// point: the GPU integer result must equal the CPU integer result BIT-FOR-BIT.
//
// Red-Black ordering: update all "red" cells (i+j even) from current black
// values, then all "black" cells from the freshly updated red values, each
// sweep. Within a colour, every cell reads only the OTHER colour, so there is
// no read-write race and the result is independent of thread scheduling.
//
// Fixed initial condition, fixed sweep count, fixed boundary (Dirichlet).
// =============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef GRID_N
#define GRID_N 128            // GRID_N x GRID_N grid (incl. boundary ring)
#endif

#ifndef N_SWEEPS
#define N_SWEEPS 200
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const int Q_SHIFT     = 16;
static const int RECIP_SHIFT = 30;   // headroom for the reciprocal multiply
static const double Q_SCALE  = 65536.0;

// alpha = 0.25 in Q16.16 (a diffusion-ish coefficient). 0.25 * 65536 = 16384.
static const int32_t ALPHA_Q = 16384;

// Precompute RECIP = round(2^RECIP_SHIFT / (1 + 4*alpha)) as an integer.
// 1 + 4*alpha in Q16.16 = (1<<16) + 4*ALPHA_Q.
static inline int64_t compute_recip() {
    int64_t denom_q16 = ((int64_t)1 << Q_SHIFT) + 4LL * ALPHA_Q; // (1+4a) in Q16.16
    // We want RECIP such that  (num * RECIP) >> RECIP_SHIFT  ~=  num / (1+4a).
    // num is Q16.16, denom_q16 is Q16.16, so num/(denom_q16>>16-equivalent):
    // x_new[Q16.16] = num[Q16.16] * 2^16 / denom_q16  (since denom is Q16.16).
    // => RECIP = round( 2^RECIP_SHIFT * 2^16 / denom_q16 ), and we shift by
    //    RECIP_SHIFT. We fold the extra 2^16 into the shift bookkeeping below.
    // To keep it simple and exact, define:
    //    RECIP = round( (2^(RECIP_SHIFT) << 16) / denom_q16 )
    // and x_new = (num * RECIP) >> RECIP_SHIFT.
    // Check units: num[Q16.16] * (2^(RECIP_SHIFT+16)/denom_q16) >> RECIP_SHIFT
    //   = num * 2^16 / denom_q16 = num/(1+4a) in Q16.16. Correct.
    // numerator = 2^(RECIP_SHIFT+Q_SHIFT) = 2^46, fits comfortably in int64
    // (< 2^63). denom_q16 = 2^17 here, so r ~= 2^29 -- also well within int64.
    // No 128-bit arithmetic needed (MSVC host has no __int128).
    int64_t numerator = ((int64_t)1 << (RECIP_SHIFT + Q_SHIFT));
    int64_t r = (numerator + denom_q16 / 2) / denom_q16; // round to nearest
    return r;
}

// =============================================================================
// Device + host shared update (identical integer arithmetic)
// =============================================================================
__host__ __device__ static inline int32_t gs_update(
        int32_t b, int32_t xL, int32_t xR, int32_t xU, int32_t xD,
        int32_t alpha, int64_t recip, int recip_shift, int q_shift) {
    int64_t nsum = (int64_t)xL + xR + xU + xD;            // Q16.16
    int64_t adiff = ((int64_t)alpha * nsum) >> q_shift;  // alpha*sum, Q16.16
    int64_t num = (int64_t)b + adiff;                    // Q16.16
    int64_t prod = num * recip;                          // Q(16.16 + recip)
    int64_t xnew = prod >> recip_shift;                  // arithmetic shift
    return (int32_t)xnew;
}

// One colour of one sweep. colour = 0 (red, (i+j) even) or 1 (black).
__global__ void k_gs_sweep(int32_t* x, const int32_t* b, int nfull,
                          int32_t alpha, int64_t recip, int colour) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    // interior cells only: 1 .. nfull-2
    int i = gx + 1;
    int j = gy + 1;
    if (i >= nfull - 1 || j >= nfull - 1) return;
    if (((i + j) & 1) != colour) return;

    int idx = j * nfull + i;
    int32_t xL = x[idx - 1];
    int32_t xR = x[idx + 1];
    int32_t xU = x[idx - nfull];
    int32_t xD = x[idx + nfull];
    x[idx] = gs_update(b[idx], xL, xR, xU, xD, alpha, recip, RECIP_SHIFT, Q_SHIFT);
}

// =============================================================================
// CPU integer reference -- byte-identical arithmetic to the device path.
// =============================================================================
static void cpu_gs(int32_t* x, const int32_t* b, int nfull,
                   int32_t alpha, int64_t recip, int nsweeps) {
    for (int s = 0; s < nsweeps; ++s) {
        for (int colour = 0; colour < 2; ++colour) {
            for (int j = 1; j < nfull - 1; ++j) {
                for (int i = 1; i < nfull - 1; ++i) {
                    if (((i + j) & 1) != colour) continue;
                    int idx = j * nfull + i;
                    int32_t xL = x[idx - 1];
                    int32_t xR = x[idx + 1];
                    int32_t xU = x[idx - nfull];
                    int32_t xD = x[idx + nfull];
                    x[idx] = gs_update(b[idx], xL, xR, xU, xD,
                                       alpha, recip, RECIP_SHIFT, Q_SHIFT);
                }
            }
        }
    }
}

// FNV-1a 64-bit hash over the raw int32 grid bytes -- order-fixed checksum.
static uint64_t hash_grid(const int32_t* x, int count) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = (const unsigned char*)x;
    for (int i = 0; i < (int)(count * sizeof(int32_t)); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Deterministic fixed initial condition + source term b.
// A hot square in the middle (Q16.16), Dirichlet zero boundary.
static void init_grid(int32_t* x, int32_t* b, int nfull) {
    for (int j = 0; j < nfull; ++j) {
        for (int i = 0; i < nfull; ++i) {
            int idx = j * nfull + i;
            x[idx] = 0;
            int32_t bv = 0;
            int q = nfull / 4;
            if (i > q && i < nfull - q && j > q && j < nfull - q) {
                // source = 1.0 in Q16.16 inside the central square
                bv = (int32_t)(1 << Q_SHIFT);
            }
            b[idx] = bv;
        }
    }
}

int main(int argc, char** argv) {
    int nfull = GRID_N;
    int nsweeps = N_SWEEPS;
    int count = nfull * nfull;
    int64_t recip = compute_recip();

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# ===== SPIKE-0b : INTEGER RED-BLACK GAUSS-SEIDEL =====\n");
    printf("# GPU            : %s\n", prop.name);
    printf("# compute_cap    : %d.%d\n", prop.major, prop.minor);
    printf("# grid           : %d x %d (incl. boundary ring)\n", nfull, nfull);
    printf("# sweeps         : %d\n", nsweeps);
    printf("# alpha (Q16.16) : %d  (= %.4f)\n", ALPHA_Q, ALPHA_Q / Q_SCALE);
    printf("# RECIP_SHIFT    : %d\n", RECIP_SHIFT);
    printf("# RECIP (precomp): %lld\n", (long long)recip);

    int32_t* h_x = (int32_t*)malloc(count * sizeof(int32_t));
    int32_t* h_b = (int32_t*)malloc(count * sizeof(int32_t));
    int32_t* h_gpu_out = (int32_t*)malloc(count * sizeof(int32_t));
    init_grid(h_x, h_b, nfull);

    // ---- CPU integer reference --------------------------------------------
    int32_t* h_cpu = (int32_t*)malloc(count * sizeof(int32_t));
    memcpy(h_cpu, h_x, count * sizeof(int32_t));
    cpu_gs(h_cpu, h_b, nfull, ALPHA_Q, recip, nsweeps);
    uint64_t cpu_hash = hash_grid(h_cpu, count);

    // ---- GPU integer ------------------------------------------------------
    int32_t* d_x = nullptr; int32_t* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_b, count * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, count * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, count * sizeof(int32_t), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 gridDim((nfull - 2 + block.x - 1) / block.x,
                 (nfull - 2 + block.y - 1) / block.y);
    for (int s = 0; s < nsweeps; ++s) {
        k_gs_sweep<<<gridDim, block>>>(d_x, d_b, nfull, ALPHA_Q, recip, 0);
        CUDA_CHECK(cudaGetLastError());
        k_gs_sweep<<<gridDim, block>>>(d_x, d_b, nfull, ALPHA_Q, recip, 1);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_gpu_out, d_x, count * sizeof(int32_t), cudaMemcpyDeviceToHost));
    uint64_t gpu_hash = hash_grid(h_gpu_out, count);

    // ---- compare -----------------------------------------------------------
    // exact element-wise diff count, plus a sample value at the centre
    int diffs = 0; int first_i = -1, first_j = -1;
    for (int j = 0; j < nfull && diffs == 0; ++j)
        for (int i = 0; i < nfull; ++i)
            if (h_cpu[j*nfull+i] != h_gpu_out[j*nfull+i]) {
                diffs++; first_i = i; first_j = j; break;
            }
    int cidx = (nfull/2) * nfull + (nfull/2);

    printf("\nDIGEST 0b_integer cpu_hash = 0x%016llX\n", (unsigned long long)cpu_hash);
    printf("DIGEST 0b_integer gpu_hash = 0x%016llX\n", (unsigned long long)gpu_hash);
    printf("# centre cell (i=%d,j=%d): cpu_raw=%d gpu_raw=%d  (approx %.6f)\n",
           nfull/2, nfull/2, h_cpu[cidx], h_gpu_out[cidx], h_cpu[cidx]/Q_SCALE);

    bool match = (cpu_hash == gpu_hash);
    if (match && diffs == 0) {
        printf("RESULT 0b : GPU-integer == CPU-integer  BIT-FOR-BIT  (PASS)\n");
    } else {
        printf("RESULT 0b : MISMATCH  (FAIL) first diff at (i=%d,j=%d)\n",
               first_i, first_j);
    }

    free(h_x); free(h_b); free(h_cpu); free(h_gpu_out);
    cudaFree(d_x); cudaFree(d_b);
    printf("# ===== SPIKE-0b COMPLETE =====\n");

    // Hard assertion so the run script's exit code reflects correctness.
    if (!match || diffs != 0) {
        fprintf(stderr, "ASSERTION FAILED: GPU integer GS != CPU integer GS\n");
        return 2;
    }
    return 0;
}
