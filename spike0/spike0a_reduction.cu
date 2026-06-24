// =============================================================================
// Spike-0a : the reduction (mimics the `mean_wp` global sum)
// -----------------------------------------------------------------------------
// De-risks the premise of the fixed-point physics migration:
//   * float reductions whose summation ORDER is nondeterministic (atomicAdd)
//     can produce DIFFERENT bits run-to-run, even on ONE machine.
//   * a fixed-order float tree reduction is stable per-arch but may differ
//     ACROSS architectures (FMA contraction / rounding differences).
//   * an INTEGER (Q16.16 -> int64) reduction is associative, so it is
//     bit-identical regardless of thread order, run, or GPU architecture.
//
// We sum a large DETERMINISTIC array three ways, ~20 repeats each, and report
// whether the result BITS are STABLE or VARY across the repeats.
//
// Rigor notes:
//   * No wall-clock / nondeterministic seeding. The array is generated on the
//     GPU from a fixed closed-form formula (counter-based hash of the index),
//     so it is byte-reproducible on every machine.
//   * The value distribution is deliberately wide-dynamic-range with mixed
//     signs, which is the regime where float reordering actually changes the
//     rounded result. (A benign all-positive ~equal-magnitude array can sum
//     deterministically by luck and make the experiment vacuous.)
//   * Float results are printed as raw hex bits so a 1-ULP difference is
//     visible. Integer results are printed as exact decimal.
// =============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#ifndef N_ELEMS
#define N_ELEMS (32 * 1024 * 1024)   // 32M elements
#endif

#ifndef N_REPEATS
#define N_REPEATS 20
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

// ---- Q16.16 fixed point -----------------------------------------------------
// 16 integer bits, 16 fractional bits. One unit = 1/65536.
// #define so the constants are usable in both host and device code.
#define Q_SHIFT 16
#define Q_SCALE 65536.0   /* 2^16 */

// ---- Deterministic value generator -----------------------------------------
// A counter-based hash (SplitMix64-style) of the index, mapped to a float in a
// wide, signed range. Identical on host and device, identical on every machine.
__host__ __device__ static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x =  x ^ (x >> 31);
    return x;
}

// Map index -> deterministic float value.
// Wide dynamic range + mixed sign so that summation ORDER genuinely matters.
__host__ __device__ static inline float gen_value(uint32_t i) {
    uint64_t h = splitmix64((uint64_t)i * 2654435761ULL + 0x1234567ULL);
    // mantissa in [0,1)
    double frac = (double)(h >> 11) * (1.0 / 9007199254740992.0); // 2^-53 * (h>>11)
    // exponent in a wide-ish band so magnitudes span several orders
    int    expo = (int)((h >> 3) & 0x1F) - 16;        // -16 .. +15
    double mag  = ldexp(frac, expo);                   // frac * 2^expo
    double sign = (h & 1ULL) ? -1.0 : 1.0;
    return (float)(sign * mag);
}

// Quantize a float value to Q16.16 (int32). Deterministic rounding (round to
// nearest, ties away from zero via +-0.5 then truncation toward zero).
//
// LOAD-BEARING EXACTNESS -- do NOT "simplify" any step into something that
// rounds; the whole determinism claim rests on every operation here being
// exact (no implementation-defined / arch-dependent rounding):
//   * `(double)v`            : float -> double is EXACT (double's 52-bit mantissa
//                              strictly contains float's 23-bit mantissa).
//   * `* Q_SCALE` (= 2^16)   : multiplying a double by a power of two is an EXACT
//                              exponent bump (no mantissa change) -- it cannot
//                              round, on any arch. (This is why Q_SCALE MUST stay
//                              a power of two; do not change it to a "nicer"
//                              decimal.)
//   * `+ 0.5` / `- 0.5`      : 0.5 is exactly representable; the add is exact for
//                              every |scaled| < 2^52 (double has integer
//                              precision below 2^52). Our |scaled| <= ~|v|*2^16,
//                              and |v| < 2^15, so |scaled| < 2^31 << 2^52: exact.
//   * `(int32_t)r`           : truncation toward zero of a double that is already
//                              an exact integer-valued double -> EXACT, defined.
// The generator (gen_value) caps |v| at frac*2^15 with frac in [0,1), i.e. just
// UNDER 2^15 = 32768.0, which is the Q16.16 positive clip. It sits about
// 0.0002% (one part in ~2^19) under that clip boundary, so the clamp branches
// below are essentially never taken -- they exist only as a defensive guard.
// Because every step is exact, to_q16_16 returns the SAME int32 bit pattern on
// host and on every GPU arch; there is no FMA to contract and nothing to round.
__host__ __device__ static inline int32_t to_q16_16(float v) {
    double scaled = (double)v * Q_SCALE;
    // round to nearest integer, deterministic
    double r = (scaled >= 0.0) ? (scaled + 0.5) : (scaled - 0.5);
    // clamp into int32 range (values are small, but be safe)
    if (r >  2147483647.0) r =  2147483647.0;
    if (r < -2147483648.0) r = -2147483648.0;
    return (int32_t)r;
}

// =============================================================================
// Kernels
// =============================================================================

// (1) float atomicAdd into a single accumulator. Order is nondeterministic.
__global__ void k_float_atomic(const float* __restrict__ x, int n, float* acc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        atomicAdd(acc, x[i]);
    }
}

// (2) float fixed tree reduction. Each block reduces its slice with a
// fixed-order shared-memory tree; block partials are summed in a SECOND
// deterministic pass (single block, fixed order) so there is NO atomic and
// NO order ambiguity within a run.
__global__ void k_float_tree_blocksum(const float* __restrict__ x, int n,
                                      float* block_partials) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int base = blockIdx.x * (blockDim.x * 2);
    int stride = blockDim.x * 2 * gridDim.x;

    // grid-stride, but each thread accumulates a FIXED set of indices in a
    // fixed order -> deterministic per launch config.
    float sum = 0.0f;
    for (int i = base + tid; i < n; i += stride) {
        sum += x[i];
        int j = i + blockDim.x;
        if (j < n) sum += x[j];
    }
    s[tid] = sum;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) block_partials[blockIdx.x] = s[0];
}

// final fixed-order sum of block partials (single thread, ascending index)
__global__ void k_float_tree_final(const float* __restrict__ partials,
                                   int nblocks, float* out) {
    float sum = 0.0f;
    for (int i = 0; i < nblocks; ++i) sum += partials[i];
    *out = sum;
}

// (3) integer Q16.16 -> int64 accumulator via integer atomicAdd.
// Integer addition is associative, so order does not matter.
__global__ void k_int_atomic(const float* __restrict__ x, int n,
                             unsigned long long* acc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    long long local = 0;
    for (int i = idx; i < n; i += stride) {
        local += (long long)to_q16_16(x[i]);
    }
    // atomicAdd on unsigned long long; two's-complement bit pattern of the
    // signed sum is added in. Wrapping is associative & commutative, so the
    // final 64-bit pattern is independent of order. (Our true sum fits well
    // within int64, so no wrap actually occurs.)
    atomicAdd(acc, (unsigned long long)local);
}

// (4) FMA-contraction demo. The SAME fixed-order sum-of-products
//        acc = sum_k ( w_k * x_k )
// computed two ways:
//   (a) FUSED   : fmaf(w, x, acc)            -- one rounding (w*x+acc rounded once)
//   (b) SEPARATE: t = __fmul_rn(w, x);       -- round the product, THEN
//                 acc = __fadd_rn(acc, t)    -- round the add (two roundings)
// Same math, same order; the ONLY difference is fused vs separate rounding.
// On hardware with FMA (all our GPUs), the compiler/arch is free to contract
// `a*b+c` into an fmaf -- or not -- and different arches/compilers/flags choose
// differently. So the float result depends on a choice OUTSIDE the source's
// control: float is NOT portable. Integer arithmetic has no such freedom (there
// is exactly one int multiply and one int add, both exact-or-defined), so it
// has nothing to contract and nothing to diverge on.
//
// We write BOTH results out as raw bits and let the host show they differ.
__global__ void k_fma_demo(const float* __restrict__ x, int n,
                           float* out_fused, float* out_separate) {
    // single thread, fixed ascending order -> the order is identical for both;
    // only the rounding structure differs.
    float acc_fused = 0.0f;
    float acc_sep   = 0.0f;
    for (int i = 0; i < n; ++i) {
        // a deterministic "weight" derived from the same generator, so the
        // products have wide dynamic range and the rounding actually bites.
        float w = gen_value((uint32_t)(i ^ 0x5A5A5A5Au));
        float xi = x[i];
        // (a) fused: single rounding of (w*xi + acc)
        acc_fused = fmaf(w, xi, acc_fused);
        // (b) separate: round the product, then round the add. The volatile
        //     forces the intermediate to a real, separately-rounded float so the
        //     compiler cannot fold this back into an fmaf behind our back.
        volatile float prod = __fmul_rn(w, xi);
        acc_sep = __fadd_rn(acc_sep, prod);
    }
    *out_fused    = acc_fused;
    *out_separate = acc_sep;
}

// Fill the input array deterministically on the GPU.
__global__ void k_fill(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        x[i] = gen_value((uint32_t)i);
    }
}

// =============================================================================
// Host helpers
// =============================================================================
static void hexbits_f32(float f, char* out) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    sprintf(out, "0x%08X", u);
}

int main(int argc, char** argv) {
    int n = N_ELEMS;
    int repeats = N_REPEATS;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# ===== SPIKE-0a : REDUCTION =====\n");
    printf("# GPU            : %s\n", prop.name);
    printf("# compute_cap    : %d.%d\n", prop.major, prop.minor);
    printf("# N_ELEMS        : %d\n", n);
    printf("# N_REPEATS      : %d\n", repeats);
    printf("# value_gen      : splitmix64(index) -> signed wide-dynamic-range float\n");
    printf("# Q format       : Q16.16 (scale=%.0f) into int64 accumulator\n", Q_SCALE);

    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(float)));

    int threads = 256;
    int fillBlocks = 1024;
    k_fill<<<fillBlocks, threads>>>(d_x, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- method 1: float atomicAdd ----------------------------------------
    {
        float* d_acc = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc, sizeof(float)));
        int blocks = 4096;
        printf("\n# --- METHOD 1: float atomicAdd (order nondeterministic) ---\n");
        uint32_t first_bits = 0; bool have_first = false; bool varied = false;
        for (int r = 0; r < repeats; ++r) {
            float zero = 0.0f;
            CUDA_CHECK(cudaMemcpy(d_acc, &zero, sizeof(float), cudaMemcpyHostToDevice));
            k_float_atomic<<<blocks, threads>>>(d_x, n, d_acc);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            float h; CUDA_CHECK(cudaMemcpy(&h, d_acc, sizeof(float), cudaMemcpyDeviceToHost));
            char hb[16]; hexbits_f32(h, hb);
            uint32_t u; memcpy(&u, &h, sizeof(u));
            if (!have_first) { first_bits = u; have_first = true; }
            else if (u != first_bits) varied = true;
            printf("repeat %02d  float_atomic  bits=%s  approx=% .9e\n", r, hb, (double)h);
        }
        printf("RESULT method1 float_atomic : %s across %d repeats\n",
               varied ? "VARIES (jitter observed -- as expected)"
                      : "STABLE (no jitter seen -- UNEXPECTED, see README)",
               repeats);
        cudaFree(d_acc);
    }

    // ---- method 2: float fixed tree reduction -----------------------------
    {
        int blocks = 1024;  // fixed launch config -> fixed reduction order
        float* d_partials = nullptr;
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partials, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        printf("\n# --- METHOD 2: float fixed tree reduction (fixed order) ---\n");
        uint32_t first_bits = 0; bool have_first = false; bool varied = false;
        size_t shmem = threads * sizeof(float);
        for (int r = 0; r < repeats; ++r) {
            k_float_tree_blocksum<<<blocks, threads, shmem>>>(d_x, n, d_partials);
            CUDA_CHECK(cudaGetLastError());
            k_float_tree_final<<<1, 1>>>(d_partials, blocks, d_out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            float h; CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost));
            char hb[16]; hexbits_f32(h, hb);
            uint32_t u; memcpy(&u, &h, sizeof(u));
            if (!have_first) { first_bits = u; have_first = true; }
            else if (u != first_bits) varied = true;
            printf("repeat %02d  float_tree    bits=%s  approx=% .9e\n", r, hb, (double)h);
        }
        printf("RESULT method2 float_tree   : %s across %d repeats (this machine)\n",
               varied ? "VARIES" : "STABLE (per-arch; may still differ ACROSS arch)",
               repeats);
        cudaFree(d_partials); cudaFree(d_out);
    }

    // ---- method 3: integer Q16.16 -> int64 --------------------------------
    {
        unsigned long long* d_acc = nullptr;
        CUDA_CHECK(cudaMalloc(&d_acc, sizeof(unsigned long long)));
        int blocks = 4096;
        printf("\n# --- METHOD 3: integer Q16.16 -> int64 atomicAdd (associative) ---\n");
        unsigned long long first = 0; bool have_first = false; bool varied = false;
        for (int r = 0; r < repeats; ++r) {
            unsigned long long zero = 0ULL;
            CUDA_CHECK(cudaMemcpy(d_acc, &zero, sizeof(zero), cudaMemcpyHostToDevice));
            k_int_atomic<<<blocks, threads>>>(d_x, n, d_acc);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            unsigned long long h;
            CUDA_CHECK(cudaMemcpy(&h, d_acc, sizeof(h), cudaMemcpyDeviceToHost));
            long long s; memcpy(&s, &h, sizeof(s));
            double approx = (double)s / Q_SCALE;
            if (!have_first) { first = h; have_first = true; }
            else if (h != first) varied = true;
            printf("repeat %02d  int_q16_16    raw=%lld  approx=% .9e\n", r, s, approx);
        }
        long long s0; memcpy(&s0, &first, sizeof(s0));
        printf("RESULT method3 int_q16_16   : %s across %d repeats  (raw int64 sum = %lld)\n",
               varied ? "VARIES (UNEXPECTED -- integer should be exact!)"
                      : "IDENTICAL (bit-exact every repeat -- as expected)",
               repeats, s0);
        printf("DIGEST 0a_integer raw_int64 = %lld\n", s0);
        cudaFree(d_acc);
    }

    // ---- method 4: FMA contraction (fused vs separate) --------------------
    // The genuine cross-arch float hazard. SAME math, SAME fixed order; the only
    // difference is whether (w*x + acc) is fused (one rounding) or separate (two
    // roundings). The compiler/arch is free to pick either for `a*b+c`, so float
    // is not portable. We show the two bit patterns DIFFER. (Integer has no fused
    // multiply-add to choose -- there is nothing to contract -- so it cannot
    // diverge this way: that is precisely why the migration uses integers.)
    {
        // single-threaded fixed-order sum-of-products over a modest slice
        // (one thread, so keep it small enough to be instant).
        int m = 1 << 16;   // 65536 fixed-order terms
        float* d_fused = nullptr; float* d_sep = nullptr;
        CUDA_CHECK(cudaMalloc(&d_fused, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sep,   sizeof(float)));
        printf("\n# --- METHOD 4: FMA contraction (fused fmaf vs separate mul/add) ---\n");
        printf("# fixed-order sum of %d products  acc += w_k * x_k\n", m);
        k_fma_demo<<<1, 1>>>(d_x, m, d_fused, d_sep);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float hf, hs;
        CUDA_CHECK(cudaMemcpy(&hf, d_fused, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hs, d_sep,   sizeof(float), cudaMemcpyDeviceToHost));
        char bf[16], bs[16];
        hexbits_f32(hf, bf);
        hexbits_f32(hs, bs);
        uint32_t uf, us; memcpy(&uf, &hf, sizeof(uf)); memcpy(&us, &hs, sizeof(us));
        printf("  fused    fmaf(w,x,acc)        bits=%s  approx=% .9e\n", bf, (double)hf);
        printf("  separate __fmul_rn then __fadd_rn bits=%s  approx=% .9e\n", bs, (double)hs);
        bool differ = (uf != us);
        printf("RESULT method4 fma_contract : fused %s separate  (%s)\n",
               differ ? "!=" : "==",
               differ ? "DIFFER -- same math, fused-vs-separate -> different bits; "
                        "arches/compilers choose differently => float not portable; "
                        "integer has no such choice"
                      : "IDENTICAL on this build (no contraction happened here -- "
                        "still a hazard on other arches/flags)");
        printf("DIGEST 0a_fma fused=%s separate=%s\n", bf, bs);
        cudaFree(d_fused); cudaFree(d_sep);
    }

    cudaFree(d_x);
    printf("\n# ===== SPIKE-0a COMPLETE =====\n");
    return 0;
}
