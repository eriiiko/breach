// ============================================================================
// EOS P6.3 — multigrid Helmholtz pressure solve, device side — see
// cuda_mg_solve.h. A bit-identical GPU port of EOSSolver::mg_run_solve_cpu
// (eos_solver.cpp): the fixed-schedule V(nu1,nu2)xC cycles (or the flat RB-GS
// reference path), on the host-built per-tick hierarchy.
//
// KERNELS
//   mg_smooth_color   one launch per color per sweep on FINE levels — the
//                     RB-GS half-sweep (review §1.1: within a color every
//                     read is the cell's own pre-update P, an opposite-color
//                     neighbor, or a per-cell constant -> order-free; the
//                     launch boundary IS the CPU's color boundary)
//   mg_residual       r@F8 = b − A·P, per cell (pure gather)
//   mg_restrict       per-COARSE-cell SUM of ≤4 child residuals -> b;
//                     P := 0 (single writer — the PC transpose)
//   mg_prolong        per-FINE-cell += its ONE parent's correction (gather)
//   mg_fused_tail     THE COARSE-TAIL FUSION (review §2.2): every level with
//                     ≤ MG_TAIL_MAX_CELLS cells — at 160² that is levels 3..8
//                     — runs inside ONE kernel, ONE thread block, covering
//                     the whole sub-V (down-leg smooth/residual/restrict,
//                     the coarsest sweep block, up-leg prolong/smooth).
//   mg_zero_excl      the level-0 vacuum-Dirichlet/solid zero
//
// WHY THE FUSED TAIL IS BIT-IDENTICAL BY CONSTRUCTION (review §2.2): the CPU
// executes the tail as a strict sequence of PASSES (color half-sweep,
// residual, restrict, prolong), and the ONLY concurrency the fused kernel
// introduces is WITHIN one pass:
//   * within a color half-sweep, updates are order-free (§1.1 above — the
//     4-stencil flips parity, so no same-color operand is ever read; the
//     cell's own P is read once, pre-update, and written once);
//   * residual/restrict/prolong are single-writer gathers, order-free over
//     cells by construction;
// and EVERY cross-pass dependency (color→color, sweep→sweep, smooth→
// residual→restrict→(coarser)…→prolong→smooth) sits at a __syncthreads()
// placed exactly where the CPU has a loop boundary. One thread block makes
// that barrier global over every cell of every tail level (all tail levels
// have ≤ 1024 = one block's worth of cells). Same per-cell arithmetic
// (identical device helpers), same pass boundaries, order-free within each
// pass ⇒ the fused kernel computes the same bits as the launch-per-pass
// schedule, which computes the same bits as the CPU. It removes the ~238 of
// ~304 solve launches/tick that the ≤1024-cell levels would otherwise eat
// (incl. the 64 one-cell launches at the 1×1 coarsest) — measured counts are
// returned via launches_actual/launches_naive.
//
// DEVICE ARITHMETIC INVENTORY (review §1.8 — do NOT restructure): per
// cell-pass, 1× mul128_shr_signed(m, P, 8) + ≤4× mul128_shr_signed(g, ΔP, 8)
// + (smoother only) 1× mul128_shr_signed(r8, recip, 40); int64 adds;
// (int32_t) narrow at the P store. The 128-bit staging is LOAD-BEARING:
// deep-level g×ΔP products reach ~7e18 — at the int64 edge — which is
// exactly why the CPU stages them through 128 bits; the device mirrors that
// stage with __mul64hi (the proven MSVC-_mul128-equivalent hi:lo combine).
// No division, no reciprocal, no float anywhere on the device — all of
// those live in the host-side build (EOSSolver::mg_build_levels).
// ============================================================================
#include "cuda_mg_solve.h"
#include "cuda_fixedpoint_device.cuh"  // mul128_shr_signed (shared device kit)

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in eos_mg_solve/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// ---- digest: the cheap FNV-1a-style running hash over a Q16.16 buffer -------
// HOST-side (review §2.6). Byte-for-byte the anon-namespace digest_of in
// eos_solver.cpp (sequential, order-dependent, pure integer).
uint64_t digest_of_host(const int32_t* buf, int n, uint64_t seed) {
    uint64_t h = seed ^ 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint32_t)buf[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// The fused-tail entry threshold (review §2.2 "levels with ≤~1,024 cells" —
// at 160² that is levels 3..8; also exactly one thread block's max size, so
// a block-wide __syncthreads() is a barrier over EVERY cell of EVERY tail
// level). Level cell counts are non-increasing with depth ((d+1)>>1 ≤ d), so
// the first level under the threshold starts a contiguous tail.
constexpr int MG_TAIL_MAX_CELLS = 1024;
constexpr int MG_MAX_LEVELS = 9;         // eos_solver.cpp level-count cap

// ---- one device-resident MG level (device-pointer mirror of MGLevel) -------
// P/b/res are written on device (P by smooth/restrict/prolong/zero; b by
// restrict on coarse levels; res by residual). m/gE/gS/recip/excl are
// read-only operator data, host-built per tick.
struct MGLevelDev {
    int h, w;
    const uint8_t* excl;
    const int64_t* m;
    const int64_t* gE;
    const int64_t* gS;
    const int64_t* recip;
    int64_t* b;
    int32_t* P;
    int64_t* res;
};

// The fused tail's by-value kernel parameter (levels ts..n_levels-1 are the
// tail; passing the whole array keeps indexing identical to the host loop).
struct MGTailArgs {
    MGLevelDev L[MG_MAX_LEVELS];
    int ts;        // first tail level
    int n_levels;
    int nu1, nu2, coarsest_sweeps;
};

// ============================================================================
// Per-cell bodies — VERBATIM device transcriptions of mg_run_solve_cpu's
// smooth / residual / restrict_res / prolong_correct inner loops (same
// neighbor guards, same Dirichlet-reads-as-0, same 128-bit shifts 8/40, same
// (int32_t) narrows). Shared by the per-launch grid kernels AND the fused
// tail so there is exactly ONE transcription of each.
// ============================================================================

__device__ __forceinline__ void mg_smooth_cell(const MGLevelDev& L,
                                               int i, int x, int y) {
    if (L.excl[i] != 0) return;
    const int lh = L.h, lw = L.w;
    const int32_t pi = L.P[i];
    // (A·P)@F8 = m·P + Σ g·(P_i − P_nb), each product >>8.
    int64_t ap = mul128_shr_signed(L.m[i], (int64_t)pi, 8);
    if (x < lw - 1 && L.excl[i + 1] != 2) {
        const int32_t pn = (L.excl[i + 1] == 1) ? 0 : L.P[i + 1];
        ap += mul128_shr_signed(L.gE[i], (int64_t)(pi - pn), 8);
    }
    if (x > 0 && L.excl[i - 1] != 2) {
        const int32_t pn = (L.excl[i - 1] == 1) ? 0 : L.P[i - 1];
        ap += mul128_shr_signed(L.gE[i - 1], (int64_t)(pi - pn), 8);
    }
    if (y < lh - 1 && L.excl[i + lw] != 2) {
        const int32_t pn = (L.excl[i + lw] == 1) ? 0 : L.P[i + lw];
        ap += mul128_shr_signed(L.gS[i], (int64_t)(pi - pn), 8);
    }
    if (y > 0 && L.excl[i - lw] != 2) {
        const int32_t pn = (L.excl[i - lw] == 1) ? 0 : L.P[i - lw];
        ap += mul128_shr_signed(L.gS[i - lw], (int64_t)(pi - pn), 8);
    }
    const int64_t r8 = L.b[i] - ap;
    // inc(P counts) = r8·(2^48/d) >> 40  ==  (r8·2^8)/d.
    const int64_t inc = mul128_shr_signed(r8, L.recip[i], 40);
    L.P[i] = (int32_t)((int64_t)pi + inc);
}

__device__ __forceinline__ void mg_residual_cell(const MGLevelDev& L,
                                                 int i, int x, int y) {
    if (L.excl[i] != 0) { L.res[i] = 0; return; }
    const int lh = L.h, lw = L.w;
    const int32_t pi = L.P[i];
    int64_t ap = mul128_shr_signed(L.m[i], (int64_t)pi, 8);
    if (x < lw - 1 && L.excl[i + 1] != 2) {
        const int32_t pn = (L.excl[i + 1] == 1) ? 0 : L.P[i + 1];
        ap += mul128_shr_signed(L.gE[i], (int64_t)(pi - pn), 8);
    }
    if (x > 0 && L.excl[i - 1] != 2) {
        const int32_t pn = (L.excl[i - 1] == 1) ? 0 : L.P[i - 1];
        ap += mul128_shr_signed(L.gE[i - 1], (int64_t)(pi - pn), 8);
    }
    if (y < lh - 1 && L.excl[i + lw] != 2) {
        const int32_t pn = (L.excl[i + lw] == 1) ? 0 : L.P[i + lw];
        ap += mul128_shr_signed(L.gS[i], (int64_t)(pi - pn), 8);
    }
    if (y > 0 && L.excl[i - lw] != 2) {
        const int32_t pn = (L.excl[i - lw] == 1) ? 0 : L.P[i - lw];
        ap += mul128_shr_signed(L.gS[i - lw], (int64_t)(pi - pn), 8);
    }
    L.res[i] = L.b[i] - ap;
}

// restriction, one COARSE cell: P := 0 (corrections start at 0 — set for
// EVERY coarse cell, excluded included, exactly like the CPU), b := SUM of
// the ≤4 regular children's residuals (the PC transpose; child-read order is
// the CPU's dy-then-dx order — int64 addition is order-free anyway).
__device__ __forceinline__ void mg_restrict_cell(const MGLevelDev& F,
                                                 const MGLevelDev& Cl,
                                                 int A, int X, int Y) {
    Cl.P[A] = 0;
    if (Cl.excl[A] != 0) { Cl.b[A] = 0; return; }
    int64_t rsum = 0;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dxx = 0; dxx < 2; ++dxx) {
            const int fy = 2 * Y + dy, fx = 2 * X + dxx;
            if (fy >= F.h || fx >= F.w) continue;
            const int fi = fy * F.w + fx;
            if (F.excl[fi] == 0) rsum += F.res[fi];
        }
    }
    Cl.b[A] = rsum;
}

// prolongation, one FINE cell: += its ONE parent's correction (PC injection,
// the exact transpose), int64 add + (int32_t) narrow — the CPU's own store.
__device__ __forceinline__ void mg_prolong_cell(const MGLevelDev& F,
                                                const MGLevelDev& Cl,
                                                int fi, int fx, int fy) {
    if (F.excl[fi] != 0) return;
    const int A = (fy >> 1) * Cl.w + (fx >> 1);
    if (Cl.excl[A] != 0) return;
    F.P[fi] = (int32_t)((int64_t)F.P[fi] + (int64_t)Cl.P[A]);
}

// ============================================================================
// Grid kernels (fine levels — one launch per pass; the launch boundary is
// the CPU's pass boundary).
// ============================================================================

__global__ void mg_smooth_color(MGLevelDev L, int color) {
    const int n = L.h * L.w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / L.w;
        const int x = i - y * L.w;
        if (((x + y) & 1) != color) continue;
        mg_smooth_cell(L, i, x, y);
    }
}

__global__ void mg_residual(MGLevelDev L) {
    const int n = L.h * L.w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const int y = i / L.w;
        const int x = i - y * L.w;
        mg_residual_cell(L, i, x, y);
    }
}

__global__ void mg_restrict(MGLevelDev F, MGLevelDev Cl) {
    const int cn = Cl.h * Cl.w;
    for (int A = blockIdx.x * blockDim.x + threadIdx.x; A < cn;
         A += gridDim.x * blockDim.x) {
        const int Y = A / Cl.w;
        const int X = A - Y * Cl.w;
        mg_restrict_cell(F, Cl, A, X, Y);
    }
}

__global__ void mg_prolong(MGLevelDev F, MGLevelDev Cl) {
    const int fn = F.h * F.w;
    for (int fi = blockIdx.x * blockDim.x + threadIdx.x; fi < fn;
         fi += gridDim.x * blockDim.x) {
        const int fy = fi / F.w;
        const int fx = fi - fy * F.w;
        mg_prolong_cell(F, Cl, fi, fx, fy);
    }
}

__global__ void mg_zero_excl(MGLevelDev L) {
    const int n = L.h * L.w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        if (L.excl[i] != 0) L.P[i] = 0;   // vacuum Dirichlet + solid zero
    }
}

// ============================================================================
// The FUSED COARSE-TAIL kernel — ONE block executes the whole sub-V-cycle
// below level `ts` (down-leg smooth/residual/restrict per level, the
// coarsest sweep block, up-leg prolong/smooth per level), with
// __syncthreads() at every CPU pass boundary. Bit-identity argument in the
// file header. Every per-thread loop is a block-stride sweep over the
// level's cells; the barriers sit OUTSIDE those loops, so every thread
// reaches every barrier (no divergent sync).
// ============================================================================

// One block-wide smooth pass set: `sweeps` × (color 0, color 1), barrier
// after each color — exactly the CPU's sweep/color loop nesting.
__device__ __forceinline__ void tail_smooth(const MGLevelDev& L, int sweeps) {
    const int n = L.h * L.w;
    for (int it = 0; it < sweeps; ++it) {
        for (int color = 0; color < 2; ++color) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const int y = i / L.w;
                const int x = i - y * L.w;
                if (((x + y) & 1) != color) continue;
                mg_smooth_cell(L, i, x, y);
            }
            __syncthreads();   // color boundary == the CPU's color loop edge
        }
    }
}

__global__ void mg_fused_tail(MGTailArgs a) {
    // down-leg: smooth(nu1) -> residual -> restrict, levels ts..n_levels-2
    for (int lv = a.ts; lv < a.n_levels - 1; ++lv) {
        const MGLevelDev& L = a.L[lv];
        const MGLevelDev& C = a.L[lv + 1];
        tail_smooth(L, a.nu1);
        const int n = L.h * L.w;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int y = i / L.w;
            const int x = i - y * L.w;
            mg_residual_cell(L, i, x, y);
        }
        __syncthreads();       // residual complete before restriction reads it
        const int cn = C.h * C.w;
        for (int A = threadIdx.x; A < cn; A += blockDim.x) {
            const int Y = A / C.w;
            const int X = A - Y * C.w;
            mg_restrict_cell(L, C, A, X, Y);
        }
        __syncthreads();       // coarse (P, b) complete before its smooth
    }
    // coarsest sweep block
    tail_smooth(a.L[a.n_levels - 1], a.coarsest_sweeps);
    // up-leg: prolong <- coarser, then smooth(nu2), levels n_levels-2..ts
    for (int lv = a.n_levels - 2; lv >= a.ts; --lv) {
        const MGLevelDev& L = a.L[lv];
        const MGLevelDev& C = a.L[lv + 1];
        const int fn = L.h * L.w;
        for (int fi = threadIdx.x; fi < fn; fi += blockDim.x) {
            const int fy = fi / L.w;
            const int fx = fi - fy * L.w;
            mg_prolong_cell(L, C, fi, fx, fy);
        }
        __syncthreads();       // correction applied before the post-smooth
        tail_smooth(L, a.nu2);
    }
}

}  // namespace

// ============================================================================
// Host entry — see the header for the contract.
// ============================================================================
uint64_t eos_mg_vcycle(
        const MGLevelHostView* levels, int n_levels,
        bool use_multigrid, int mg_cycles, int mg_nu1, int mg_nu2,
        int mg_coarsest_sweeps, int flat_S,
        int32_t* p_out,
        int* launches_actual, int* launches_naive) {
    if (n_levels <= 0 || n_levels > MG_MAX_LEVELS) {
        throw std::runtime_error("eos_mg_vcycle: bad n_levels");
    }
    const int n0 = levels[0].h * levels[0].w;
    if (n0 <= 0) throw std::runtime_error("eos_mg_vcycle: empty level 0");

    // ---- device upload: every level's 8 arrays (per-call era; the S8
    //      endpoint keeps these resident + builds them on device) ------------
    std::vector<void*> allocs;
    auto dev_upload = [&](const void* src, size_t bytes) -> void* {
        void* d = nullptr;
        cuda_check(cudaMalloc(&d, bytes), "malloc level array");
        allocs.push_back(d);
        cuda_check(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice),
                   "H2D level array");
        return d;
    };
    auto free_all = [&]() {
        for (void* d : allocs) cudaFree(d);
        allocs.clear();
    };

    MGLevelDev dev[MG_MAX_LEVELS] = {};
    try {
        for (int lv = 0; lv < n_levels; ++lv) {
            const MGLevelHostView& H = levels[lv];
            const int n = H.h * H.w;
            if (n <= 0) throw std::runtime_error("eos_mg_vcycle: empty level");
            MGLevelDev& D = dev[lv];
            D.h = H.h; D.w = H.w;
            D.excl  = (const uint8_t*)dev_upload(H.excl, (size_t)n);
            D.m     = (const int64_t*)dev_upload(H.m, (size_t)n * 8);
            D.gE    = (const int64_t*)dev_upload(H.gE, (size_t)n * 8);
            D.gS    = (const int64_t*)dev_upload(H.gS, (size_t)n * 8);
            D.recip = (const int64_t*)dev_upload(H.recip, (size_t)n * 8);
            D.b     = (int64_t*)dev_upload(H.b, (size_t)n * 8);
            D.P     = (int32_t*)dev_upload(H.P, (size_t)n * 4);
            // res is device scratch — written by mg_residual before any read.
            void* dres = nullptr;
            cuda_check(cudaMalloc(&dres, (size_t)n * 8), "malloc res");
            allocs.push_back(dres);
            D.res = (int64_t*)dres;
        }

        const int block = 256;
        auto grid_for = [&](int n) { return (n + block - 1) / block; };
        int launches = 0;
        auto launch_check = [&](const char* what) {
            cuda_check(cudaGetLastError(), what);
            ++launches;
        };

        // smooth via per-color launches (fine levels + the flat path).
        auto smooth_launches = [&](int lv, int sweeps) {
            const MGLevelDev& L = dev[lv];
            const int g = grid_for(L.h * L.w);
            for (int it = 0; it < sweeps; ++it) {
                for (int color = 0; color < 2; ++color) {
                    mg_smooth_color<<<g, block>>>(L, color);
                    launch_check("smooth launch");
                }
            }
        };

        // The fused-tail entry level: first level with ≤ MG_TAIL_MAX_CELLS
        // cells (cell counts are non-increasing with depth, so the tail is
        // contiguous). ts == n_levels ⇒ no level qualifies (never happens at
        // game sizes — the coarsest is 1×1 — but handled: fully per-launch).
        int ts = n_levels;
        for (int lv = 0; lv < n_levels; ++lv) {
            if (dev[lv].h * dev[lv].w <= MG_TAIL_MAX_CELLS) { ts = lv; break; }
        }
        // The tail block: one block, threads ≥ min(1024, max tail cells) not
        // required — block-stride loops cover any tail size ≤ 1024; 256
        // threads keeps the one-cell coarsest from wasting a huge block.
        const int tail_block = 256;

        if (use_multigrid && n_levels > 1) {
            for (int cyc = 0; cyc < mg_cycles; ++cyc) {
                const int host_down_end =
                    (ts < n_levels - 1) ? ts : (n_levels - 1);
                for (int lv = 0; lv < host_down_end; ++lv) {
                    smooth_launches(lv, mg_nu1);
                    mg_residual<<<grid_for(dev[lv].h * dev[lv].w), block>>>(dev[lv]);
                    launch_check("residual launch");
                    mg_restrict<<<grid_for(dev[lv + 1].h * dev[lv + 1].w), block>>>(
                        dev[lv], dev[lv + 1]);
                    launch_check("restrict launch");
                }
                if (ts < n_levels) {
                    MGTailArgs a;
                    for (int lv = 0; lv < n_levels; ++lv) a.L[lv] = dev[lv];
                    a.ts = ts;
                    a.n_levels = n_levels;
                    a.nu1 = mg_nu1;
                    a.nu2 = mg_nu2;
                    a.coarsest_sweeps = mg_coarsest_sweeps;
                    mg_fused_tail<<<1, tail_block>>>(a);
                    launch_check("fused tail launch");
                } else {
                    smooth_launches(n_levels - 1, mg_coarsest_sweeps);
                }
                for (int lv = host_down_end - 1; lv >= 0; --lv) {
                    mg_prolong<<<grid_for(dev[lv].h * dev[lv].w), block>>>(
                        dev[lv], dev[lv + 1]);
                    launch_check("prolong launch");
                    smooth_launches(lv, mg_nu2);
                }
            }
        } else {
            smooth_launches(0, flat_S);   // flat A/B reference path
        }

        mg_zero_excl<<<grid_for(n0), block>>>(dev[0]);
        launch_check("zero-excl launch");

        cuda_check(cudaDeviceSynchronize(), "sync");
        cuda_check(cudaMemcpy(p_out, dev[0].P, (size_t)n0 * 4,
                              cudaMemcpyDeviceToHost), "D2H P");

        // Naive launch count: the SAME schedule with one kernel per color
        // per sweep + per pass everywhere (no tail fusion) — review §0's
        // tally, +1 for the zero-excl.
        int naive = 0;
        if (use_multigrid && n_levels > 1) {
            const int down = (n_levels - 1) * (mg_nu1 * 2 + 2);
            const int coarsest = mg_coarsest_sweeps * 2;
            const int up = (n_levels - 1) * (1 + mg_nu2 * 2);
            naive = mg_cycles * (down + coarsest + up);
        } else {
            naive = flat_S * 2;
        }
        naive += 1;
        if (launches_actual) *launches_actual = launches;
        if (launches_naive)  *launches_naive = naive;

        free_all();
    } catch (...) {
        free_all();
        throw;
    }

    // Host-side digest, byte-for-byte the CPU's digest_helmholtz expression.
    return digest_of_host(p_out, n0, 0);
}

namespace {
bool g_mg_solve_backend_cuda = false;
}
bool mg_solve_backend_is_cuda() { return g_mg_solve_backend_cuda; }
void set_mg_solve_backend_cuda(bool on) { g_mg_solve_backend_cuda = on; }

}  // namespace breach_cuda
