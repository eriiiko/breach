// ============================================================================
// cuda_radiation_sweep.cu — the CUDA twin of RadiationSweep (ray-engine-v2 P4).
// See cuda_radiation_sweep.h for the contract, the wavefront mechanics and the
// credits; radiation_sweep.cpp for the CPU body every kernel below transcribes.
// ============================================================================
//
// FOUR KERNELS, in launch order, all integer:
//
//   rs_init_env   one thread per env: initialise the env's counter block and
//                 check the per-env scalars (k_leak in [0, ONE]; vac_level <=
//                 E0 when the ambient is derived; since P5a every entry of the
//                 shared heat_absorb table in [0, HEAT_ABSORB_Q_MAX]) — run()'s
//                 scalar checks and derive_ambient()'s.
//   rs_prepass    one thread per cell: the extinction ingress check, the
//                 per-cell ambient (read, or derived from is_vacuum — the twin
//                 of derive_ambient), and run()'s pre-pass verbatim: amb_m,
//                 the emission excess, the EFFECTIVE extinction planes (the
//                 smoke term on a gas cell, P5a — radiation_sweep.h's FP_HD
//                 gas_* functions), the Fleck factor on both of its arms (the
//                 solid one, and since P5b the gas one, fleck_L_gas_q).
//   rs_zero       one thread per cell: OVERWRITE, not accumulate — run()
//                 zeroes its four outputs after its ingress check, and so does
//                 this, per env, only where the env passed it.
//   rs_wavefront  one launch per wavefront index, one thread per (ordinate,
//                 cell on that wavefront): run()'s `visit` lambda, line for
//                 line, with the four per-cell books as integer atomics.
//
// NOTHING HERE IS ALLOWED TO DIFFER FROM radiation_sweep.cpp IN VALUE. Every
// shift, split and product below is the CPU line it sits beside in design
// §2.3; the FP_HD helpers (fleck_L_solid_q, fleck_L_gas_q, fleck_f_q24,
// e_bucket_of) and fixedpoint::mul128_shr / reciprocal_q16 /
// deposit_dT_wide_i64 are the SAME function on both backends; the
// ordinate constants are copied on the host out of RadiationSweep's own
// table. The one freedom the GPU takes — the ORDER in which the per-cell books
// accumulate — is exactly the freedom integer addition grants.

#include "cuda_radiation_sweep.h"
#include "cuda_resident.h"            // radiation_sweep_launch_resident + slots
#include "radiation_sweep.h"          // OrdinateConst, ordinate_table (host);
                                      // fleck_L_solid_q, fleck_L_gas_q (P5b),
                                      // fleck_f_q24 (FP_HD)
#include "emissive_table.h"           // e_bucket_of (FP_HD), E_TABLE_SIZE
#include "fixed_point.h"              // FP_ONE, FP_SHIFT, mul128_shr (FP_HD)
#include "cuda_fixedpoint_device.cuh" // the shared device kit

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

using fixedpoint::FP_ONE;
using fixedpoint::FP_SHIFT;

// The Fleck factor's fixed point — RadiationSweep's own constants, never
// re-typed (row 32: Q24).
constexpr int     F_SHIFT = RadiationSweep::F_SHIFT;
constexpr int32_t F_ONE   = RadiationSweep::F_ONE;

constexpr int BLOCK = 256;            // a multiple of 32: the warp reduction
                                      // below needs whole warps

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in radiation_sweep/" << what << ": "
           << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// The per-ordinate constants, passed BY VALUE as a kernel argument (at most
// 16 x 8 bytes), filled on the host from RadiationSweep::ordinate_table — the
// one home of the checked-in literals (critique 3 §5b).
struct SweepOrdinates {
    OrdinateConst oc[16];
};

// The wavefront shape of one call: K wavefront INDICES (one launch each) and
// the longest wavefront any ordinate has at any index (the grid's x extent).
//   shear: an x-major ordinate's wavefront is a COLUMN (w of them, h cells
//          each), a y-major one's a ROW (h of them, w cells each); both kinds
//          advance on the same index, so K = the longest axis any ordinate
//          walks. The S16/S12 shear tables carry both kinds.
//   step:  the anti-diagonal x' + y' (x', y' measured from the ordinate's
//          upwind corner): K = h + w - 1 for every ordinate, at most
//          min(h, w) cells each (the Koch-Baker-Alcouffe skew).
struct WavefrontShape {
    int K;
    int L_max;
};

WavefrontShape wavefront_shape(const OrdinateConst* tbl, int n_ordinates,
                               bool shear, int h, int w) {
    WavefrontShape s{0, 0};
    if (shear) {
        for (int m = 0; m < n_ordinates; ++m) {
            const bool xm = (tbl[m].x_major != 0);
            const int k_axis = xm ? w : h;   // how many wavefronts
            const int l_axis = xm ? h : w;   // how long each one is
            if (k_axis > s.K) s.K = k_axis;
            if (l_axis > s.L_max) s.L_max = l_axis;
        }
    } else {
        s.K = h + w - 1;
        s.L_max = (h < w) ? h : w;
    }
    return s;
}

__device__ __forceinline__ bool env_rejected(const int64_t* c) {
    return (c[RS_SLOT_BAD_EXTINCTION] | c[RS_SLOT_BAD_AMBIENT] |
            c[RS_SLOT_BAD_SCALARS]) != 0;
}

// One book entry, the tree's int64 atomic idiom (design §8.2;
// cuda_temperature.cu's cadd): unsigned wrap IS two's-complement signed
// addition, and modular addition is order-free. A zero skips the atomic —
// uniform-ambient cells book exact zeros (gate 2a), so most calls are skipped.
__device__ __forceinline__ void book(int64_t* p, int64_t v) {
    if (v != 0) atomicAdd((unsigned long long*)p, (unsigned long long)v);
}

// ---- 1. per-env init: the counter block + the scalar checks ----------------
__global__ void rs_init_env(int n_env,
                            const int64_t* __restrict__ e_table,
                            const int32_t* __restrict__ k_leak_q,
                            const int64_t* __restrict__ vac_level,
                            bool derive,
                            const int32_t* __restrict__ heat_absorb_q16,  // P5a, nullable
                            int n_gases,
                            int64_t* __restrict__ cnt) {
    for (int env = blockIdx.x * blockDim.x + threadIdx.x; env < n_env;
         env += gridDim.x * blockDim.x) {
        int64_t* c = cnt + (size_t)env * RADIATION_SWEEP_CNT_SLOTS;
        int64_t bits = 0;
        const int32_t k = k_leak_q[env];
        if (k < 0 || k > FP_ONE) bits |= RS_BAD_K_LEAK;           // run()
        if (derive && vac_level[env] > e_table[0])
            bits |= RS_BAD_VAC_LEVEL;                               // derive_ambient()
        for (int g = 0; g < n_gases; ++g) {                         // run(), P5a
            const int32_t hq = heat_absorb_q16[g];
            if (hq < 0 || hq > RadiationSweep::HEAT_ABSORB_Q_MAX) bits |= RS_BAD_HEAT_ABSORB;
        }
        c[RS_SLOT_BAD_EXTINCTION] = 0;
        c[RS_SLOT_BAD_AMBIENT]    = 0;
        c[RS_SLOT_BAD_SCALARS]    = bits;
        c[RS_SLOT_MIN_STREAM]     = INT64_MAX;
        c[RS_SLOT_MAX_STREAM]     = INT64_MIN;
    }
}

// ---- 2. the pre-pass: ingress, ambient, excess, Fleck (design §2.8) --------
// run()'s first loop, per cell, indexed the RL-batch way (§A rule 1):
// env = i / (h*w), the per-env scalars read by env.
__global__ void rs_prepass(int n_env, int plane,
                           const int32_t* __restrict__ temperature,
                           const int32_t* __restrict__ heat_atten_q,
                           const int32_t* __restrict__ dyn_heat_atten_q,
                           const int32_t* __restrict__ heat_inv_shift,
                           const bool* __restrict__ thermal_solid,
                           const int64_t* __restrict__ amb_level,   // nullable
                           const bool* __restrict__ is_vacuum,      // nullable
                           const int64_t* __restrict__ e_table,
                           const int64_t* __restrict__ vac_level,
                           const int32_t* __restrict__ t_amb_q,
                           const int32_t* __restrict__ gas,         // P5a, nullable
                           int n_gases,
                           const int32_t* __restrict__ heat_absorb_q16,
                           const int32_t* __restrict__ n_bulk,
                           int32_t n_floor_q, int64_t recip_cv,   // P5b: the gas currency
                           int e_fine_bits,                       // #78: the table's currency
                           int64_t w_m, bool fleck_enabled,
                           int64_t* __restrict__ amb_m,
                           int64_t* __restrict__ ex_cell,
                           int32_t* __restrict__ f_q24,
                           int32_t* __restrict__ a_eff,
                           int32_t* __restrict__ d_eff,
                           int64_t* __restrict__ cnt) {
    const int64_t total = (int64_t)n_env * (int64_t)plane;
    for (int64_t gi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; gi < total;
         gi += (int64_t)gridDim.x * blockDim.x) {
        const int env = (int)(gi / plane);
        int64_t* c = cnt + (size_t)env * RADIATION_SWEEP_CNT_SLOTS;
        // The ingress re-check 0 <= a <= d <= ONE (design §2.3) — counted, not
        // thrown; the env is then skipped whole (see cuda_resident.h).
        const int64_t a = heat_atten_q[gi];
        const int64_t d = dyn_heat_atten_q[gi];
        if (a < 0 || a > d || d > FP_ONE)
            atomicAdd((unsigned long long*)&c[RS_SLOT_BAD_EXTINCTION], 1ULL);
        // The per-cell ambient LEVEL (thermal model v2 R3): run()'s plane when
        // given, else derive_ambient()'s select — a vacuum cell radiates against
        // vac_level (negative: E0, space at room temperature, R4), every other
        // cell against E0.
        const int64_t e0 = e_table[0];
        int64_t amb_i;
        if (amb_level != nullptr) {
            amb_i = amb_level[gi];
        } else {
            const int64_t vl = vac_level[env];
            const int64_t vac = (vl < 0) ? e0 : vl;
            amb_i = (is_vacuum != nullptr && is_vacuum[gi]) ? vac : e0;
        }
        if (amb_i < 0 || amb_i > e0)
            atomicAdd((unsigned long long*)&c[RS_SLOT_BAD_AMBIENT], 1ULL);
        amb_m[gi] = (amb_i * w_m) >> FP_SHIFT;
        int64_t ex = e_table[e_bucket_of(temperature[gi])] - amb_i;
        if (ex < 0) ex = 0;                 // dead under the invariant; run()'s belt and braces
        ex_cell[gi] = ex;
        // THE EFFECTIVE EXTINCTION (P5a) — run()'s, through the same FP_HD
        // functions: the density law over this env's gas planes on a GAS cell,
        // the N_EPS floor on its bulk count, both planes as MAXes. The sum over
        // at most N_GAS_PLANES_MAX planes is exact in int64 (radiation_sweep.h),
        // so the order the terms are added in is irrelevant, as on the CPU —
        // which skips the zero-coefficient gases this kernel may be handed.
        int32_t ae = (int32_t)a;
        int32_t de = (int32_t)d;
        int32_t ag = 0;                     // the smoke term's own extinction
        if (n_gases > 0 && !thermal_solid[gi]) {
            const size_t cell = (size_t)(gi - (int64_t)env * (int64_t)plane);
            const int32_t* genv = gas + (size_t)env * (size_t)n_gases * (size_t)plane;
            int64_t sum = 0;
            for (int g = 0; g < n_gases; ++g) {
                sum += gas_density_term(heat_absorb_q16[g], genv[(size_t)g * (size_t)plane + cell]);
            }
            ag = gas_extinction_finish(sum, n_bulk[gi]);
            ae = gas_effective_a(ae, ag, false);
            de = gas_effective_d(de, ae);
        }
        a_eff[gi] = ae;
        d_eff[gi] = de;
        int64_t L = 0;
        if (thermal_solid[gi]) {
            // #78: ex is in the table's currency; the arm converts once
            // (fixedpoint::fine_heat_shr at his), the same FP_HD function.
            L = fleck_L_solid_q(ex, heat_atten_q[gi], heat_inv_shift[gi], e_fine_bits);
        } else if (ag > 0) {
            // THE GAS ARM (P5b) — run()'s, through the same FP_HD function: the
            // smoke term's excess in the temperature fold's own gas currency
            // (the two scalars ride by value; the per-cell reciprocal_q16 and
            // the staged mul128_shr chain are the kit's, on both backends).
            L = fleck_L_gas_q(ex, ag, n_bulk[gi], n_floor_q, recip_cv, e_fine_bits);
        }
        int64_t T_abs = (int64_t)temperature[gi] + (int64_t)t_amb_q[env];
        if (T_abs < 1) T_abs = 1;
        f_q24[gi] = fleck_enabled ? fleck_f_q24(T_abs, L) : F_ONE;
    }
}

// ---- 3. OVERWRITE: zero the four books, per env, only if it passed ---------
__global__ void rs_zero(int n_env, int plane, const int64_t* __restrict__ cnt,
                        int64_t* __restrict__ rad_net,
                        int64_t* __restrict__ rad_flux,
                        int64_t* __restrict__ rad_amb,
                        int64_t* __restrict__ rad_fluence) {
    const int64_t total = (int64_t)n_env * (int64_t)plane;
    for (int64_t gi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; gi < total;
         gi += (int64_t)gridDim.x * blockDim.x) {
        const int env = (int)(gi / plane);
        if (env_rejected(cnt + (size_t)env * RADIATION_SWEEP_CNT_SLOTS)) continue;
        rad_net[gi] = 0;
        rad_flux[gi] = 0;
        rad_amb[gi] = 0;
        rad_fluence[gi] = 0;
    }
}

// ---- 4. one wavefront of every ordinate (design §2.3, §8.2) ----------------
// grid = (ceil(L_max / BLOCK), n_ordinates, n_env): blockIdx.y is the ordinate,
// blockIdx.z the env, so the env, the ordinate and the wavefront's geometry
// are BLOCK-UNIFORM (every early return below is taken by whole blocks, which
// is what lets the warp reduction at the end use a full mask).
__global__ void rs_wavefront(int wf, int h, int w, int n_ordinates, bool shear,
                             SweepOrdinates ords, int64_t w_m,
                             const int32_t* __restrict__ a_eff,       // P5a: the
                             const int32_t* __restrict__ d_eff,       // effective planes
                             const int32_t* __restrict__ k_leak_q,
                             const int64_t* __restrict__ amb_m,
                             const int64_t* __restrict__ ex_cell,
                             const int32_t* __restrict__ f_q24,
                             int64_t* outflow,   // read upwind, write own: not restrict
                             int64_t* __restrict__ rad_net,
                             int64_t* __restrict__ rad_flux,
                             int64_t* __restrict__ rad_amb,
                             int64_t* __restrict__ rad_fluence,
                             int64_t* __restrict__ cnt) {
    const int env = blockIdx.z;
    const int m = blockIdx.y;
    int64_t* c = cnt + (size_t)env * RADIATION_SWEEP_CNT_SLOTS;
    if (env_rejected(c)) return;

    const OrdinateConst oc = ords.oc[m];
    const int sx = oc.sx;
    const int sy = oc.sy;
    const bool xm = (oc.x_major != 0);
    const int64_t s_m = oc.s_m;
    // The two UPWIND reads as (dy, dx) offsets — a: the straight one (fa is
    // recomputed from it), b: the other (it carries the remainder fb). The
    // DOWNWIND targets are the same offsets negated. run()'s table, verbatim:
    //   shear, x-major:  a = (y, x-sx)   b = (y-sy, x-sx)
    //   shear, y-major:  a = (y-sy, x)   b = (y-sy, x-sx)
    //   step:            a = (y, x-sx)   b = (y-sy, x)
    int ady, adx, bdy, bdx;
    if (shear) {
        if (xm) { ady = 0;   adx = -sx; bdy = -sy; bdx = -sx; }
        else    { ady = -sy; adx = 0;   bdy = -sy; bdx = -sx; }
    } else {
        ady = 0; adx = -sx; bdy = -sy; bdx = 0;
    }

    // This ordinate's wavefront `wf`: its length L (0 when this ordinate has no
    // wavefront at this index — a short axis, or past the last anti-diagonal).
    int L = 0;
    int py_lo = 0;
    if (shear) {
        if (xm) L = (wf < w) ? h : 0;          // column wf, from the upwind side
        else    L = (wf < h) ? w : 0;          // row wf, from the upwind side
    } else {
        py_lo = (wf - (w - 1) > 0) ? (wf - (w - 1)) : 0;
        const int py_hi = (wf < h - 1) ? wf : (h - 1);
        L = (py_hi >= py_lo) ? (py_hi - py_lo + 1) : 0;
    }
    if (L == 0) return;

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t s_min = INT64_MAX;
    int64_t s_max = INT64_MIN;
    if (j < L) {
        int y, x;
        if (shear) {
            if (xm) { x = (sx > 0) ? wf : (w - 1 - wf); y = j; }
            else    { y = (sy > 0) ? wf : (h - 1 - wf); x = j; }
        } else {
            const int py = py_lo + j;          // distance from the upwind row
            const int px = wf - py;            // ... and column
            y = (sy > 0) ? py : (h - 1 - py);
            x = (sx > 0) ? px : (w - 1 - px);
        }
        const size_t plane = (size_t)h * (size_t)w;
        const size_t base = (size_t)env * plane;
        const int64_t* up = outflow + ((size_t)env * n_ordinates + m) * plane;
        int64_t* store = outflow + ((size_t)env * n_ordinates + m) * plane;
        const int i = y * w + x;
        const size_t gi = base + (size_t)i;

        const int uay = y + ady, uax = x + adx;
        const int uby = y + bdy, ubx = x + bdx;
        const bool in_a = (uay >= 0 && uay < h && uax >= 0 && uax < w);
        const bool in_b = (uby >= 0 && uby < h && ubx >= 0 && ubx < w);
        // THIS cell's ambient stream; the virtual ring returns the READING
        // cell's own value (report_t1.md §2.1), exactly as run() does.
        const int64_t amb = amb_m[gi];
        const int64_t io_a = in_a ? up[uay * w + uax] : amb;
        const int64_t io_b = in_b ? up[uby * w + ubx] : amb;
        // The split, recomputed from the stored outflow with the SAME shift;
        // fb is the remainder (critique 1 fix 3).
        const int64_t fa = (io_a * s_m) >> FP_SHIFT;
        const int64_t fb = io_b - ((io_b * s_m) >> FP_SHIFT);
        const int64_t i_in = fa + fb;
        // rad_amb collects up to five terms at a boundary cell; they are summed
        // here and booked once (integer addition: the same total).
        int64_t d_amb = 0;
        if (!in_a) d_amb -= fa;                // taken from the ring: booked
        if (!in_b) d_amb -= fb;

        const int64_t kq = k_leak_q[env];
        const int64_t leaked = (i_in * kq) >> FP_SHIFT;
        const int64_t ret    = (amb * kq) >> FP_SHIFT;   // ITS OWN ceiling
        const int64_t stream = i_in - leaked + ret;
        const int64_t a = a_eff[gi];                           // material + smoke (P5a)
        const int64_t b = (int64_t)d_eff[gi] - a;              // the body share
        const int64_t abs_mat  = (stream * a) >> FP_SHIFT;
        const int64_t abs_body = (stream * b) >> FP_SHIFT;
        const int64_t ex_m = (ex_cell[gi] * w_m) >> FP_SHIFT;
        const int64_t src  = amb + fixedpoint::mul128_shr(ex_m, (int64_t)f_q24[gi], F_SHIFT);
        const int64_t emitted   = (src * a) >> FP_SHIFT;   // the SAME arithmetic as abs_mat
        const int64_t emit_body = (amb * b) >> FP_SHIFT;   // a body re-emits AMBIENT (row 25)
        const int64_t i_out = stream - abs_mat - abs_body + emitted + emit_body;
        store[i] = i_out;

        d_amb += leaked - ret;
        // What this cell's own outflow sends into the virtual ring.
        const int64_t fa_o = (i_out * s_m) >> FP_SHIFT;
        const int64_t fb_o = i_out - fa_o;
        const int tay = y - ady, tax = x - adx;
        const int tby = y - bdy, tbx = x - bdx;
        if (!(tay >= 0 && tay < h && tax >= 0 && tax < w)) d_amb += fa_o;
        if (!(tby >= 0 && tby < h && tbx >= 0 && tbx < w)) d_amb += fb_o;

        book(&rad_net[gi],     abs_mat - emitted);     // the material ledger
        book(&rad_flux[gi],    abs_body - emit_body);  // the body sensor
        book(&rad_amb[gi],     d_amb);                 // ceiling + ring
        book(&rad_fluence[gi], stream);                // Φ, for the clamp
        s_min = stream;
        s_max = stream;
    }

    // The stream telemetry (run()'s min_stream / max_stream): a warp
    // reduction, then one 64-bit atomicMin / atomicMax per warp. min and max
    // are order-free, so the result is the CPU's.
    for (int off = 16; off > 0; off >>= 1) {
        const long long o_min = __shfl_down_sync(0xffffffffu, (long long)s_min, off);
        const long long o_max = __shfl_down_sync(0xffffffffu, (long long)s_max, off);
        if (o_min < s_min) s_min = o_min;
        if (o_max > s_max) s_max = o_max;
    }
    if ((threadIdx.x & 31) == 0 && s_min <= s_max) {
        atomicMin((long long*)&c[RS_SLOT_MIN_STREAM], (long long)s_min);
        atomicMax((long long*)&c[RS_SLOT_MAX_STREAM], (long long)s_max);
    }
}

int grid_for(int64_t total) {
    int64_t g = (total + BLOCK - 1) / BLOCK;
    if (g < 1) g = 1;
    if (g > 65535) g = 65535;          // the grid-stride loops cover the rest
    return (int)g;
}

// A single device arena for the per-call path: ONE cudaMalloc and ONE
// cudaFree per call instead of one per buffer (each sub-buffer 256-byte
// aligned). Freed by the destructor, so a throw cannot leak it.
struct DeviceArena {
    unsigned char* base = nullptr;
    size_t size = 0;
    size_t carve(size_t bytes) {
        const size_t off = size;
        size += (bytes + 255) & ~(size_t)255;
        return off;
    }
    template <typename T> T* at(size_t off) const {
        return reinterpret_cast<T*>(base + off);
    }
    ~DeviceArena() { if (base) cudaFree(base); }
};

bool g_radiation_backend_cuda = false;
long long g_radiation_sweep_calls = 0;

}  // namespace

// ---------------------------------------------------------------------------
int radiation_sweep_launch_count(int transport, int n_ordinates, int h, int w) {
    const OrdinateConst* tbl = RadiationSweep::ordinate_table(n_ordinates, transport);
    if (tbl == nullptr) return -1;
    if (h <= 0 || w <= 0) return 0;
    const bool shear = (transport == RadiationSweep::TRANSPORT_SHEAR);
    return 3 + wavefront_shape(tbl, n_ordinates, shear, h, w).K;
}

int radiation_sweep_launch_resident(
    int n_env, int h, int w,
    const int32_t* d_temperature,
    const int32_t* d_heat_atten_q, const int32_t* d_dyn_heat_atten_q,
    const int32_t* d_heat_inv_shift, const bool* d_thermal_solid,
    const int64_t* d_amb_level, const bool* d_is_vacuum,
    const int64_t* d_e_table, int e_fine_bits,
    const int64_t* d_vac_level, const int32_t* d_k_leak_q,
    const int32_t* d_t_amb_q,
    const int32_t* d_gas, int n_gases,
    const int32_t* d_heat_absorb_q16, const int32_t* d_n_bulk,
    int32_t n_floor_q, int64_t recip_cv,
    int transport, int n_ordinates, bool fleck_enabled,
    int64_t* d_outflow, int64_t* d_amb_m, int64_t* d_ex_cell, int32_t* d_f_q24,
    int32_t* d_a_eff, int32_t* d_d_eff,
    int64_t* d_rad_net, int64_t* d_rad_flux, int64_t* d_rad_amb,
    int64_t* d_rad_fluence,
    int64_t* d_cnt) {
    const OrdinateConst* tbl = RadiationSweep::ordinate_table(n_ordinates, transport);
    if (tbl == nullptr) {
        throw std::invalid_argument(
            "radiation_sweep_launch_resident: unsupported (n_ordinates, "
            "transport) - n_ordinates must be 16 or 12, transport "
            "TRANSPORT_STEP (0) or TRANSPORT_SHEAR (1)");
    }
    // P5a: the gas group's host-side shape checks (the table's VALUES are
    // counted on the device by rs_init_env, per env, the core's contract).
    const bool gas_given = (d_gas != nullptr);
    if (gas_given != (d_heat_absorb_q16 != nullptr) || gas_given != (d_n_bulk != nullptr) ||
        n_gases < 0 || n_gases > RadiationSweep::N_GAS_PLANES_MAX ||
        (!gas_given && n_gases != 0)) {
        throw std::invalid_argument(
            "radiation_sweep_launch_resident: the gas group (d_gas, "
            "d_heat_absorb_q16, d_n_bulk) is all or none, with 0 <= n_gases <= "
            "N_GAS_PLANES_MAX (design v3 section 6.3, P5a)");
    }
    // P5b: the gas arm's currency — two HOST scalars shared by every env (the
    // fold's config dials, like the heat_absorb table), so checked here as
    // host control flow, exactly as run() checks them.
    if (gas_given && (n_floor_q <= 0 || recip_cv <= 0)) {
        throw std::invalid_argument(
            "radiation_sweep_launch_resident: the gas group needs the "
            "temperature fold's gas currency - n_floor_q and recip_cv must both "
            "be > 0 (design v3 section 2.8's gas arm, P5b)");
    }
    if (d_a_eff == nullptr || d_d_eff == nullptr) {
        throw std::invalid_argument(
            "radiation_sweep_launch_resident: d_a_eff / d_d_eff scratch is "
            "required (the wavefronts read the effective extinction planes)");
    }
    // #78: the table's currency -- a host scalar shared by every env, checked
    // as host control flow exactly as run() checks it.
    if (e_fine_bits < 0 || e_fine_bits > E_FINE_BITS) {
        throw std::invalid_argument(
            "radiation_sweep_launch_resident: e_fine_bits outside [0, "
            "E_FINE_BITS] -- the table's currency (#78, emissive_table.h)");
    }
    if (n_env <= 0 || h <= 0 || w <= 0) return 0;
    if (n_env > 65535) {
        // The wavefront grid carries the env on blockIdx.z (hardware limit
        // 65535); a bigger batch needs the env folded into x, not a silent
        // launch failure.
        throw std::invalid_argument(
            "radiation_sweep_launch_resident: n_env exceeds 65535, the "
            "wavefront grid's z extent");
    }
    SweepOrdinates ords{};
    for (int m = 0; m < n_ordinates; ++m) ords.oc[m] = tbl[m];
    const bool shear = (transport == RadiationSweep::TRANSPORT_SHEAR);
    // w_m = ONE / n_ordinates — run()'s own integer (4096 at S16, 5461 at S12).
    const int64_t w_m = (int64_t)FP_ONE / n_ordinates;
    const int plane = h * w;
    const int64_t total = (int64_t)n_env * (int64_t)plane;
    int launches = 0;

    rs_init_env<<<(n_env + 63) / 64, 64>>>(n_env, d_e_table, d_k_leak_q,
                                           d_vac_level, d_amb_level == nullptr,
                                           d_heat_absorb_q16, n_gases,
                                           d_cnt);
    cuda_check(cudaGetLastError(), "init launch");
    ++launches;

    rs_prepass<<<grid_for(total), BLOCK>>>(
        n_env, plane, d_temperature, d_heat_atten_q, d_dyn_heat_atten_q,
        d_heat_inv_shift, d_thermal_solid, d_amb_level, d_is_vacuum,
        d_e_table, d_vac_level, d_t_amb_q,
        d_gas, n_gases, d_heat_absorb_q16, d_n_bulk,
        n_floor_q, recip_cv, e_fine_bits,
        w_m, fleck_enabled,
        d_amb_m, d_ex_cell, d_f_q24, d_a_eff, d_d_eff, d_cnt);
    cuda_check(cudaGetLastError(), "prepass launch");
    ++launches;

    rs_zero<<<grid_for(total), BLOCK>>>(n_env, plane, d_cnt, d_rad_net,
                                        d_rad_flux, d_rad_amb, d_rad_fluence);
    cuda_check(cudaGetLastError(), "zero launch");
    ++launches;

    const WavefrontShape ws = wavefront_shape(tbl, n_ordinates, shear, h, w);
    const dim3 grid((unsigned)((ws.L_max + BLOCK - 1) / BLOCK),
                    (unsigned)n_ordinates, (unsigned)n_env);
    for (int wf = 0; wf < ws.K; ++wf) {
        rs_wavefront<<<grid, BLOCK>>>(
            wf, h, w, n_ordinates, shear, ords, w_m,
            d_a_eff, d_d_eff, d_k_leak_q,
            d_amb_m, d_ex_cell, d_f_q24, d_outflow,
            d_rad_net, d_rad_flux, d_rad_amb, d_rad_fluence, d_cnt);
        cuda_check(cudaGetLastError(), "wavefront launch");
        ++launches;
    }
    return launches;
}

// ---------------------------------------------------------------------------
int radiation_sweep_step(
    const int32_t* temperature,
    const int32_t* heat_atten_q, const int32_t* dyn_heat_atten_q,
    const int32_t* heat_inv_shift, const bool* thermal_solid,
    const int64_t* e_table, int e_fine_bits,
    const int64_t* amb_level, const bool* is_vacuum, int64_t vac_level,
    int32_t t_amb_q, int32_t k_leak_q,
    int transport, int n_ordinates, int h, int w,
    int64_t* rad_net, int64_t* rad_flux, int64_t* rad_amb, int64_t* rad_fluence,
    bool fleck_enabled,
    int32_t* fleck_out, int64_t* min_stream_out, int64_t* max_stream_out,
    const int32_t* gas, int n_gases,
    const int32_t* heat_absorb_q16, const int32_t* n_bulk,
    int32_t n_floor_q, int64_t recip_cv) {
    if (RadiationSweep::ordinate_table(n_ordinates, transport) == nullptr) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): unsupported (n_ordinates, transport) "
            "- n_ordinates must be 16 or 12, transport TRANSPORT_STEP (0) or "
            "TRANSPORT_SHEAR (1)");
    }
    // #78: the table's currency, refused on the host exactly as run() refuses
    // it -- before anything crosses the bus.
    if (e_fine_bits < 0 || e_fine_bits > E_FINE_BITS) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): e_fine_bits outside [0, E_FINE_BITS] "
            "-- the table's currency (#78, emissive_table.h)");
    }
    // ---- P5a: the gas group, checked on the HOST exactly as run() checks it,
    // before anything crosses the bus (a rejected scene touches nothing) ----
    const bool gas_given = (gas != nullptr);
    if (gas_given != (heat_absorb_q16 != nullptr) || gas_given != (n_bulk != nullptr)) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): gas, heat_absorb_q16 and n_bulk are "
            "given together or not at all (the smoke term, design v3 §6.3)");
    }
    if (n_gases < 0 || n_gases > RadiationSweep::N_GAS_PLANES_MAX ||
        (!gas_given && n_gases != 0)) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): n_gases outside [0, "
            "N_GAS_PLANES_MAX] - the gas density sum's int64 headroom is "
            "argued for at most 16 planes");
    }
    // P5b: the gas arm's currency, refused on the host exactly as run()
    // refuses it — before anything crosses the bus.
    if (gas_given && (n_floor_q <= 0 || recip_cv <= 0)) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): the gas group needs the temperature "
            "fold's gas currency - n_floor_q and recip_cv must both be > 0 "
            "(design v3 section 2.8's gas arm, P5b)");
    }
    // The ACTIVE gases, as run() forms them: a zero coefficient adds exactly 0
    // to the density sum, so its plane never crosses the bus.
    int act_g[RadiationSweep::N_GAS_PLANES_MAX];
    int32_t act_hq[RadiationSweep::N_GAS_PLANES_MAX];
    int n_act = 0;
    for (int g = 0; g < n_gases; ++g) {
        const int32_t hq = heat_absorb_q16[g];
        if (hq < 0 || hq > RadiationSweep::HEAT_ABSORB_Q_MAX) {
            throw std::invalid_argument(
                "radiation_sweep_step (CUDA): a heat_absorb_q16 outside [0, "
                "2^28] (the gas door's [0, 4096]; negative would be a source, "
                "above it the density sum is no longer provably exact in int64)");
        }
        if (hq != 0) { act_g[n_act] = g; act_hq[n_act] = hq; ++n_act; }
    }
    const size_t n = (size_t)h * (size_t)w;
    if (h <= 0 || w <= 0) {
        if (min_stream_out) *min_stream_out = 0;
        if (max_stream_out) *max_stream_out = 0;
        return 0;
    }
    const size_t n64 = n * sizeof(int64_t);
    const size_t n32 = n * sizeof(int32_t);
    const size_t n8  = n * sizeof(bool);

    // ---- one arena for the whole call ----
    DeviceArena ar;
    const size_t o_T    = ar.carve(n32);
    const size_t o_aq   = ar.carve(n32);
    const size_t o_dq   = ar.carve(n32);
    const size_t o_his  = ar.carve(n32);
    const size_t o_ts   = ar.carve(n8);
    const size_t o_vac  = is_vacuum ? ar.carve(n8) : 0;
    const size_t o_amb  = amb_level ? ar.carve(n64) : 0;
    const size_t o_etab = ar.carve((size_t)E_TABLE_SIZE * sizeof(int64_t));
    const size_t o_vl   = ar.carve(sizeof(int64_t));
    const size_t o_kl   = ar.carve(sizeof(int32_t));
    const size_t o_ta   = ar.carve(sizeof(int32_t));
    const size_t o_out  = ar.carve((size_t)n_ordinates * n64);
    const size_t o_ambm = ar.carve(n64);
    const size_t o_ex   = ar.carve(n64);
    const size_t o_f    = ar.carve(n32);
    const size_t o_ae   = ar.carve(n32);                 // P5a: the effective
    const size_t o_de   = ar.carve(n32);                 // extinction planes
    const size_t o_gas  = n_act ? ar.carve((size_t)n_act * n32) : 0;   // active only
    const size_t o_hq   = n_act ? ar.carve((size_t)n_act * sizeof(int32_t)) : 0;
    const size_t o_nb   = n_act ? ar.carve(n32) : 0;
    const size_t o_rn   = ar.carve(n64);
    const size_t o_rf   = ar.carve(n64);
    const size_t o_ra   = ar.carve(n64);
    const size_t o_rl   = ar.carve(n64);
    const size_t o_cnt  = ar.carve(RADIATION_SWEEP_CNT_SLOTS * sizeof(int64_t));
    cuda_check(cudaMalloc((void**)&ar.base, ar.size), "malloc arena");

    // ---- H2D: the inputs and the three per-env scalars (N = 1) ----
    cuda_check(cudaMemcpy(ar.at<int32_t>(o_T), temperature, n32,
                          cudaMemcpyHostToDevice), "H2D temperature");
    cuda_check(cudaMemcpy(ar.at<int32_t>(o_aq), heat_atten_q, n32,
                          cudaMemcpyHostToDevice), "H2D heat_atten_q");
    cuda_check(cudaMemcpy(ar.at<int32_t>(o_dq), dyn_heat_atten_q, n32,
                          cudaMemcpyHostToDevice), "H2D dyn_heat_atten_q");
    cuda_check(cudaMemcpy(ar.at<int32_t>(o_his), heat_inv_shift, n32,
                          cudaMemcpyHostToDevice), "H2D heat_inv_shift");
    cuda_check(cudaMemcpy(ar.at<bool>(o_ts), thermal_solid, n8,
                          cudaMemcpyHostToDevice), "H2D thermal_solid");
    if (is_vacuum)
        cuda_check(cudaMemcpy(ar.at<bool>(o_vac), is_vacuum, n8,
                              cudaMemcpyHostToDevice), "H2D is_vacuum");
    if (amb_level)
        cuda_check(cudaMemcpy(ar.at<int64_t>(o_amb), amb_level, n64,
                              cudaMemcpyHostToDevice), "H2D amb_level");
    cuda_check(cudaMemcpy(ar.at<int64_t>(o_etab), e_table,
                          (size_t)E_TABLE_SIZE * sizeof(int64_t),
                          cudaMemcpyHostToDevice), "H2D e_table");
    cuda_check(cudaMemcpy(ar.at<int64_t>(o_vl), &vac_level, sizeof(int64_t),
                          cudaMemcpyHostToDevice), "H2D vac_level");
    cuda_check(cudaMemcpy(ar.at<int32_t>(o_kl), &k_leak_q, sizeof(int32_t),
                          cudaMemcpyHostToDevice), "H2D k_leak_q");
    cuda_check(cudaMemcpy(ar.at<int32_t>(o_ta), &t_amb_q, sizeof(int32_t),
                          cudaMemcpyHostToDevice), "H2D t_amb_q");
    // P5a: the ACTIVE gas planes, packed in order, their coefficients, and the
    // bulk count — nothing at all when every coefficient is zero (shipped).
    for (int j = 0; j < n_act; ++j) {
        cuda_check(cudaMemcpy(ar.at<int32_t>(o_gas) + (size_t)j * n,
                              gas + (size_t)act_g[j] * n, n32,
                              cudaMemcpyHostToDevice), "H2D gas plane");
    }
    if (n_act) {
        cuda_check(cudaMemcpy(ar.at<int32_t>(o_hq), act_hq,
                              (size_t)n_act * sizeof(int32_t),
                              cudaMemcpyHostToDevice), "H2D heat_absorb_q16");
        cuda_check(cudaMemcpy(ar.at<int32_t>(o_nb), n_bulk, n32,
                              cudaMemcpyHostToDevice), "H2D n_bulk");
    }

    const int launches = radiation_sweep_launch_resident(
        1, h, w,
        ar.at<int32_t>(o_T), ar.at<int32_t>(o_aq), ar.at<int32_t>(o_dq),
        ar.at<int32_t>(o_his), ar.at<bool>(o_ts),
        amb_level ? ar.at<int64_t>(o_amb) : nullptr,
        is_vacuum ? ar.at<bool>(o_vac) : nullptr,
        ar.at<int64_t>(o_etab), e_fine_bits,
        ar.at<int64_t>(o_vl), ar.at<int32_t>(o_kl), ar.at<int32_t>(o_ta),
        n_act ? ar.at<int32_t>(o_gas) : nullptr, n_act,
        n_act ? ar.at<int32_t>(o_hq) : nullptr,
        n_act ? ar.at<int32_t>(o_nb) : nullptr,
        n_floor_q, recip_cv,               // P5b: read only where a_gas > 0
        transport, n_ordinates, fleck_enabled,
        ar.at<int64_t>(o_out), ar.at<int64_t>(o_ambm), ar.at<int64_t>(o_ex),
        ar.at<int32_t>(o_f),
        ar.at<int32_t>(o_ae), ar.at<int32_t>(o_de),
        ar.at<int64_t>(o_rn), ar.at<int64_t>(o_rf), ar.at<int64_t>(o_ra),
        ar.at<int64_t>(o_rl),
        ar.at<int64_t>(o_cnt));
    cuda_check(cudaDeviceSynchronize(), "sync");

    // ---- the ingress verdict FIRST: a rejected scene copies nothing back ----
    int64_t cnt[RADIATION_SWEEP_CNT_SLOTS] = {0};
    cuda_check(cudaMemcpy(cnt, ar.at<int64_t>(o_cnt), sizeof(cnt),
                          cudaMemcpyDeviceToHost), "D2H counters");
    if (cnt[RS_SLOT_BAD_SCALARS] & RS_BAD_K_LEAK) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): k_leak_q outside [0, ONE] (design "
            "section 2.3 invariant)");
    }
    if (cnt[RS_SLOT_BAD_SCALARS] & RS_BAD_VAC_LEVEL) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): the vacuum ambient level exceeds E0; "
            "the per-cell ambient invariant is 0 <= amb <= E0, which is what "
            "keeps every cell's emission excess non-negative");
    }
    if (cnt[RS_SLOT_BAD_SCALARS] & RS_BAD_HEAT_ABSORB) {
        // Unreachable through this door (the host checked the table above);
        // kept so the device's verdict can never be silently ignored.
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): a heat_absorb_q16 outside [0, 2^28]");
    }
    if (cnt[RS_SLOT_BAD_EXTINCTION] != 0) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): extinction planes violate 0 <= a <= "
            "d <= ONE (heat_atten_q / dyn_heat_atten_q ingress invariant, "
            "design section 2.3)");
    }
    if (cnt[RS_SLOT_BAD_AMBIENT] != 0) {
        throw std::invalid_argument(
            "radiation_sweep_step (CUDA): amb_level outside [0, E0] (the "
            "per-cell ambient invariant, thermal model v2 R3)");
    }

    // ---- D2H: the four books, the Fleck plane, the stream telemetry ----
    cuda_check(cudaMemcpy(rad_net, ar.at<int64_t>(o_rn), n64,
                          cudaMemcpyDeviceToHost), "D2H rad_net");
    cuda_check(cudaMemcpy(rad_flux, ar.at<int64_t>(o_rf), n64,
                          cudaMemcpyDeviceToHost), "D2H rad_flux");
    cuda_check(cudaMemcpy(rad_amb, ar.at<int64_t>(o_ra), n64,
                          cudaMemcpyDeviceToHost), "D2H rad_amb");
    cuda_check(cudaMemcpy(rad_fluence, ar.at<int64_t>(o_rl), n64,
                          cudaMemcpyDeviceToHost), "D2H rad_fluence");
    if (fleck_out)
        cuda_check(cudaMemcpy(fleck_out, ar.at<int32_t>(o_f), n32,
                              cudaMemcpyDeviceToHost), "D2H fleck");
    if (min_stream_out) *min_stream_out = cnt[RS_SLOT_MIN_STREAM];
    if (max_stream_out) *max_stream_out = cnt[RS_SLOT_MAX_STREAM];
    ++g_radiation_sweep_calls;
    return launches;
}

bool radiation_backend_is_cuda() { return g_radiation_backend_cuda; }
void set_radiation_backend_cuda(bool on) { g_radiation_backend_cuda = on; }
long long radiation_sweep_cuda_calls() { return g_radiation_sweep_calls; }

}  // namespace breach_cuda
