// ============================================================================
// #29 P7.0a -- TEST-ONLY: THE KIT ON SYNTHETIC INPUTS (residency design 4.4c
// decisions 5 and 6, the P7.0 row). See cuda_unit_kernels.h; the gate is
// tests/cuda_unit_kit_check.py (parts DRAWS, SIGHT, FALLOFF), which holds the
// device draws (cuda_unit_draws.cuh), the device sight function
// (cuda_unit_sight.cuh) and the falloffs (fixed_point.h) equal to
// simulation.draws / GameMap.has_los / unit_fixed over their whole range. No
// slot calls it.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_batch.cuh"          // tstream, bgrid, benv
#include "cuda_unit_draws.cuh"
#include "cuda_unit_sight.cuh"
#include "fixed_point.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace breach_cuda {

namespace {

void kit_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("unit kit probe/") + what + ": " +
                                 cudaGetErrorString(e));
}

// A device copy of a host int64 array, freed at scope exit.
struct DevI64 {
    int64_t* d = nullptr;
    size_t n = 0;
    DevI64(const int64_t* host, size_t count) : n(count) {
        kit_check(cudaMalloc(reinterpret_cast<void**>(&d), (count ? count : 1) * sizeof(int64_t)),
                  "malloc");
        if (host && count)
            kit_check(cudaMemcpyAsync(d, host, count * sizeof(int64_t), cudaMemcpyHostToDevice,
                                      tstream()),
                      "H2D");
        else
            kit_check(cudaMemsetAsync(d, 0, (count ? count : 1) * sizeof(int64_t), tstream()),
                      "memset");
    }
    void down(int64_t* host) const {
        kit_check(cudaMemcpyAsync(host, d, n * sizeof(int64_t), cudaMemcpyDeviceToHost, tstream()),
                  "D2H");
        kit_check(cudaStreamSynchronize(tstream()), "sync");
    }
    ~DevI64() { cudaFree(d); }
    DevI64(const DevI64&) = delete;
    DevI64& operator=(const DevI64&) = delete;
};

using namespace unit_draws_dev;

__global__ void k_draws(const int64_t* keys, const int64_t* args, int n_rows, int64_t* out) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_rows) return;
    const int e = benv();
    const int64_t* a = args + ((size_t)e * n_rows + r) * UNIT_KIT_DRAW_ARGS;
    int64_t* o = out + ((size_t)e * n_rows + r) * UNIT_KIT_DRAW_OUT;
    bool bad = false;
    const DrawKey k = draw_key(keys, e, &bad);
    const bool salt_ok = word_ok(a[0]) && word_ok(a[5]);
    if (!salt_ok) bad = true;
    const uint32_t u = draw_at(k, (uint32_t)a[0], a[1], a[2], &bad);
    o[0] = (int64_t)u;
    o[1] = cone_offset_q16(u, a[3], &bad);
    o[2] = roll_q16(u, a[4]) ? 1 : 0;
    DrawSite sa = draw_site((uint32_t)a[0], a[1], &bad);
    DrawSite sb = draw_site((uint32_t)a[5], a[1], &bad);
    for (int j = 0; j < 3; ++j) {
        o[4 + 2 * j] = (int64_t)draw_next(k, sa);
        o[5 + 2 * j] = (int64_t)draw_next(k, sb);
    }
    o[3] = bad ? 1 : 0;
}

__global__ void k_sight(const uint8_t* solid, const uint8_t* screen, int h, int w,
                        const int64_t* pairs, int n_rows, int64_t* out) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_rows) return;
    const int e = benv();
    const size_t plane = (size_t)h * w;
    const uint8_t* so = solid + (size_t)e * plane;
    const uint8_t* sc = screen + (size_t)e * plane;
    const int64_t* q = pairs + ((size_t)e * n_rows + r) * 4;
    int64_t* o = out + ((size_t)e * n_rows + r) * 3;
    bool bad = false;
    o[0] = unit_sight::has_los(so, sc, h, w, q[0], q[1], q[2], q[3], &bad) ? 1 : 0;
    o[1] = unit_sight::has_los(so, nullptr, h, w, q[0], q[1], q[2], q[3], &bad) ? 1 : 0;
    o[2] = bad ? 1 : 0;
}

__global__ void k_falloff(const int64_t* args, int n, int64_t* out) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const int64_t* a = args + (size_t)i * 3;
    out[(size_t)i * 2] = fixedpoint::falloff_floor_q(a[0], a[1], a[2]);
    out[(size_t)i * 2 + 1] = fixedpoint::falloff_round_q(a[0], a[1], a[2]);
}

}  // namespace

void unit_kit_draws_probe(const int64_t* keys, int n_env, const int64_t* args, int n_rows,
                          int64_t* out) {
    if (n_env < 1 || n_rows < 1 || !keys || !args || !out)
        throw std::invalid_argument("unit_kit_draws_probe: empty input");
    DevI64 dk(keys, (size_t)n_env * DRAW_KEY_W);
    DevI64 da(args, (size_t)n_env * n_rows * UNIT_KIT_DRAW_ARGS);
    DevI64 dout(nullptr, (size_t)n_env * n_rows * UNIT_KIT_DRAW_OUT);
    k_draws<<<bgrid(n_rows, n_env, 128), 128, 0, tstream()>>>(dk.d, da.d, n_rows, dout.d);
    kit_check(cudaGetLastError(), "k_draws");
    dout.down(out);
}

void unit_kit_sight_probe(std::uintptr_t solid, std::uintptr_t screen, int n_env, int h, int w,
                          const int64_t* pairs, int n_rows, int64_t* out) {
    if (n_env < 1 || n_rows < 1 || h < 1 || w < 1 || !solid || !screen || !pairs || !out)
        throw std::invalid_argument("unit_kit_sight_probe: empty input");
    DevI64 dp(pairs, (size_t)n_env * n_rows * 4);
    DevI64 dout(nullptr, (size_t)n_env * n_rows * 3);
    k_sight<<<bgrid(n_rows, n_env, 128), 128, 0, tstream()>>>(
        reinterpret_cast<const uint8_t*>(solid), reinterpret_cast<const uint8_t*>(screen), h, w,
        dp.d, n_rows, dout.d);
    kit_check(cudaGetLastError(), "k_sight");
    dout.down(out);
}

void unit_kit_falloff_probe(const int64_t* args, int n, int64_t* out) {
    if (n < 1 || !args || !out) throw std::invalid_argument("unit_kit_falloff_probe: empty input");
    DevI64 da(args, (size_t)n * 3);
    DevI64 dout(nullptr, (size_t)n * 2);
    k_falloff<<<bgrid(n, 1, 256), 256, 0, tstream()>>>(da.d, n, dout.d);
    kit_check(cudaGetLastError(), "k_falloff");
    dout.down(out);
}

const char* unit_draws_fingerprint() { return unit_draws::UNIT_DRAWS_FINGERPRINT; }

}  // namespace breach_cuda
