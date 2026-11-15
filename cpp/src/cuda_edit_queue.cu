// ============================================================================
// #29 P4c — THE DEVICE EDIT QUEUE (cuda_resident.h's edit_queue_launch_resident;
// residency design §4.8). Slot 6b's FieldEdit FLUSH on the resident planes.
//
// THE ORACLE is the CPU flush, src/simulation/field_edit.py: EditQueue's
// stable sort by (field, source_id, region, seq), then apply_field_edit per
// edit, per tile of its region in row-major order. The host still enqueues
// and sorts; field_edit.pack_for_device writes that applied order as rows --
// one EDIT row per edit (the door's integers: amount_q in the field's
// currency, den, the noise's low bound, the edit key = its seq, the clamps)
// and one TILE row per tile its region yields (the region walked by the one
// walk, _iter_region, with its Q16 weight and the blast's scale) -- and this
// file applies them:
//
//   K1 edq_contrib  (one thread per tile, all envs): the contribution
//       amount_q x weight x mult x scale over den, rounded ONCE half away
//       from zero (unit_fixed.div_round), in 128-bit magnitude arithmetic;
//       mult = draws.noise_mult_q16 on the keyed draw (tick, key,
//       FIELD_NOISE, tile) under the env's seed, Philox-4x32-10 (philox32.h,
//       the vendored header the Python twin is held equal to). A pure
//       function of the row: the order does not matter here.
//   K2 edq_apply    (one thread per env, its tiles in row order): the skip
//       mask, the combine and the gas-energy seam -- SERIAL, because a tile's
//       result depends on the edits before it (a clamp or a MAX is order
//       dependent; the seam's mirror refresh reads the cell the previous
//       edit wrote), exactly as the host loop runs.
//
// Per field (FIELD_POLICY): gas / smoke (the trace currency, TRACE_CLAMP, the
// counted int32 saturation with its deposit / clip books), atmosphere (bulk N
// split 21 / 79 over O2 / inert_N2, minted at ambient or removed at the
// cell's T through the seam), wave_source (the temperature combine, capped at
// T_MAX_PHYS, landed as N * dT through the seam on an accountable cell),
// fire (clamped [0, ONE]), heat (the saturating additive branch), gas_energy
// (a raw deposit through the seam). water_depth is never packed.
//
// NO float anywhere. The host pack bounds every product (< 2^125 in
// magnitude), every denominator (< 2^62) and every contribution (< 2^55), so
// the device's 128 / 64-bit arithmetic is exact; a result the host flush
// would REFUSE (an int32 store out of range, an int64 energy overflow, a
// negative trace result) is counted in the env's failure slot and thrown at
// the books fold (cuda_resident_tick.cu), as the host raises.
// ============================================================================
#include "cuda_resident.h"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"   // the tick stream
#include "philox32.h"       // Salmon et al. SC'11 (vendored; host + device)

namespace breach_cuda {

namespace {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in edit_queue/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

constexpr int64_t ONE_Q16 = 65536;
constexpr int64_t I32_MAX = 2147483647LL;
constexpr int64_t I32_MIN = -2147483648LL;
// field_edit.T_MAX_PHYS (16000 game degrees) in Q16: the T write path's ceiling.
constexpr int64_t T_MAX_PHYS_Q = 16000LL * 65536LL;
// gas_fixed.TRACE_CELL_MAX: the counted int32 saturation's ceiling.
constexpr int64_t TRACE_CELL_MAX = 2147483647LL;
// field_edit.EditMode
constexpr int MODE_ADD = 0, MODE_REMOVE = 1, MODE_MAX = 2;

// ---- 128-bit magnitudes (hi, lo) -------------------------------------------
struct U128 { unsigned long long hi, lo; };

__device__ inline U128 mul_64_64(unsigned long long a, unsigned long long b) {
    U128 r;
    r.lo = a * b;
    r.hi = __umul64hi(a, b);
    return r;
}

// a * b, the product known (by the host's bound) to fit 128 bits.
__device__ inline U128 mul_128_64(U128 a, unsigned long long b) {
    U128 r = mul_64_64(a.lo, b);
    r.hi += a.hi * b;
    return r;
}

__device__ inline unsigned long long uabs64(int64_t v) {
    return v < 0 ? 0ULL - (unsigned long long)v : (unsigned long long)v;
}

// floor(n / d), n < 2^127, 0 < d < 2^63 (so the remainder's shift never
// overflows), the quotient known to fit 64 bits: restoring division.
__device__ inline unsigned long long div_128_64(U128 n, unsigned long long d) {
    unsigned long long r = 0, q = 0;
    for (int i = 127; i >= 0; --i) {
        const unsigned long long bit =
            i >= 64 ? (n.hi >> (i - 64)) & 1ULL : (n.lo >> i) & 1ULL;
        r = (r << 1) | bit;
        q <<= 1;
        if (r >= d) { r -= d; q |= 1ULL; }
    }
    return q;
}

// unit_fixed.div_round on int64 (d > 0): n / d rounded half AWAY from zero.
__device__ inline int64_t div_round_i64(int64_t n, int64_t d) {
    const unsigned long long a = uabs64(n);
    const unsigned long long q = (2ULL * a + (unsigned long long)d) /
                                 (2ULL * (unsigned long long)d);
    return n >= 0 ? (int64_t)q : -(int64_t)q;
}

// Python's floor division for a positive divisor.
__device__ inline int64_t floordiv_i64(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b) != 0 && a < 0) q -= 1;
    return q;
}

// Checked int64 arithmetic: false on overflow (numpy's int64 store raises).
__device__ inline bool add_ok(int64_t a, int64_t b, int64_t* out) {
    const unsigned long long r = (unsigned long long)a + (unsigned long long)b;
    const int64_t s = (int64_t)r;
    if (((a ^ s) & (b ^ s)) < 0) return false;
    *out = s;
    return true;
}
__device__ inline bool mul_ok(int64_t a, int64_t b, int64_t* out) {
    const int64_t lo = (int64_t)((unsigned long long)a * (unsigned long long)b);
    const int64_t hi = __mul64hi(a, b);
    if (hi != (lo >> 63)) return false;
    *out = lo;
    return true;
}

// field_edit._combine_q: the integer combine, then the clamp ('elif': a value
// below lo is never compared with hi).
__device__ inline int64_t combine_q(int64_t old, int64_t c, int mode, int flags,
                                    int64_t lo, int64_t hi, int k) {
    int64_t nw;
    if (mode == MODE_ADD) nw = old + c;
    else if (mode == MODE_REMOVE) nw = old - c;
    else nw = old >= c ? old : c;
    const bool has_lo = (flags >> (2 * k)) & 1;
    const bool has_hi = (flags >> (2 * k + 1)) & 1;
    if (has_lo && nw < lo) nw = lo;
    else if (has_hi && nw > hi) nw = hi;
    return nw;
}

inline __device__ bool fits_i32(int64_t v) { return v >= I32_MIN && v <= I32_MAX; }

// ---- K1: the contributions --------------------------------------------------
__global__ void edq_contrib(int64_t* __restrict__ contrib,
                            const int64_t* __restrict__ edits,
                            const int64_t* __restrict__ tiles,
                            const int64_t* __restrict__ env_sc,
                            uint32_t salt, int n_tiles) {
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n_tiles;
         t += gridDim.x * blockDim.x) {
        const int64_t* tr = tiles + (size_t)t * EDQ_TILE_COLS;
        const int64_t* er = edits + (size_t)tr[0] * EDQ_EDIT_COLS;
        const int64_t amount = er[EDQ_E_AMOUNT];
        const int64_t den = er[EDQ_E_DEN];
        const int64_t weight = tr[2];
        const int64_t scale = tr[3];
        bool neg = (amount < 0) != (weight < 0);
        U128 m = mul_64_64(uabs64(amount), uabs64(weight));
        if (er[EDQ_E_NOISY]) {
            const int64_t* es = env_sc + (size_t)er[EDQ_E_ENV] * EDQ_ENV_COLS;
            // Draw.counter(): (tick & M32, index = the edit key, salt, id = tile)
            const uint32_t ctr[4] = {(uint32_t)es[0], (uint32_t)er[EDQ_E_KEY], salt,
                                     (uint32_t)tr[1]};
            const uint32_t key[2] = {(uint32_t)es[1], (uint32_t)es[2]};
            uint32_t out[4];
            philox32_4x32_10(ctr, key, out);
            // draws.noise_mult_q16: lo + ((word * (ONE - lo + 1)) >> 32)
            const int64_t low = er[EDQ_E_LOW];
            const unsigned long long mult =
                (unsigned long long)low +
                (((unsigned long long)out[0] *
                  (unsigned long long)(ONE_Q16 - low + 1)) >> 32);
            m = mul_128_64(m, mult);
        }
        neg = neg != (scale < 0);
        m = mul_128_64(m, uabs64(scale));
        // div_round: (2|n| + d) // (2d)
        U128 n2;
        n2.hi = (m.hi << 1) | (m.lo >> 63);
        n2.lo = m.lo << 1;
        const unsigned long long ud = (unsigned long long)den;
        const unsigned long long lo_sum = n2.lo + ud;
        n2.hi += (lo_sum < n2.lo) ? 1ULL : 0ULL;
        n2.lo = lo_sum;
        const unsigned long long q = div_128_64(n2, 2ULL * ud);
        contrib[t] = neg ? -(int64_t)q : (int64_t)q;
    }
}

// ---- K2: the serial applier, one thread per env ----------------------------
struct EnvView {
    int n, n_gases, o2, n2;
    size_t cell0, gas0;
    int32_t* gas;
    int32_t* temperature;
    int64_t* gas_energy;
    int64_t t_amb;
    unsigned long long* books;

    __device__ int32_t& g(int slice, int i) const {
        return gas[gas0 + (size_t)slice * n + i];
    }
    __device__ int64_t bulk_n(int i) const {
        return (int64_t)g(o2, i) + (int64_t)g(n2, i);
    }
    // GameMap.gas_energy_refresh_cell on an accountable cell.
    __device__ void refresh(int i) const {
        const int64_t nn = bulk_n(i);
        int32_t* tp = temperature + cell0 + i;
        if (nn < 1) { *tp = 0; return; }
        int64_t t = floordiv_i64(gas_energy[cell0 + i], nn) - t_amb;
        if (t > I32_MAX) t = I32_MAX;
        else if (t < I32_MIN) t = I32_MIN;
        *tp = (int32_t)t;
    }
    // GameMap.gas_energy_deposit on an accountable cell: nothing on de == 0
    // (no book, no refresh); else E += de, the channel books it, the mirror
    // is refreshed.
    __device__ void deposit(int i, int64_t de, int slot) const {
        if (de == 0) return;
        int64_t e;
        if (!add_ok(gas_energy[cell0 + i], de, &e)) {
            books[EDQ_B_FAIL] += 1ULL;
            return;
        }
        gas_energy[cell0 + i] = e;
        books[slot] += (unsigned long long)de;
        books[slot + 1] += 1ULL;
        refresh(i);
    }
};

__global__ void edq_apply(const int64_t* __restrict__ edits,
                          const int64_t* __restrict__ tiles,
                          const int64_t* __restrict__ env_off,
                          const int64_t* __restrict__ env_sc,
                          const int64_t* __restrict__ contrib,
                          int n_env, int h, int w, int n_gases, int o2_idx, int n2_idx,
                          int32_t* gas, int32_t* temperature, int64_t* gas_energy,
                          int32_t* fire, int32_t* heat,
                          const bool* __restrict__ solid,
                          const bool* __restrict__ is_vacuum,
                          const bool* __restrict__ is_ambient,
                          const bool* __restrict__ thermal_solid,
                          const bool* __restrict__ flammable,
                          unsigned long long* books_all) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n_env) return;
    EnvView V;
    V.n = h * w;
    V.n_gases = n_gases;
    V.o2 = o2_idx;
    V.n2 = n2_idx;
    V.cell0 = (size_t)env * V.n;
    V.gas0 = (size_t)env * n_gases * V.n;
    V.gas = gas;
    V.temperature = temperature;
    V.gas_energy = gas_energy;
    V.t_amb = env_sc[(size_t)env * EDQ_ENV_COLS + 3];
    V.books = books_all + (size_t)env * EDQ_SLOTS;
    unsigned long long* B = V.books;

    for (int64_t t = env_off[env]; t < env_off[env + 1]; ++t) {
        const int64_t* tr = tiles + (size_t)t * EDQ_TILE_COLS;
        const int64_t* er = edits + (size_t)tr[0] * EDQ_EDIT_COLS;
        const int i = (int)tr[1];
        if (i < 0 || i >= V.n) { B[EDQ_B_FAIL] += 1ULL; continue; }
        const size_t ci = V.cell0 + i;
        const int field = (int)er[EDQ_E_FIELD];
        const int mode = (int)er[EDQ_E_MODE];
        const int flags = (int)er[EDQ_E_CLAMP_FLAGS];
        const bool accountable = !solid[ci] && !thermal_solid[ci] && !is_vacuum[ci] &&
                                 !(is_ambient != nullptr && is_ambient[ci]);
        // The policy's skip mask, FIRST (a skipped tile draws nothing).
        bool skip;
        switch (field) {
            case EDQ_F_ATMOSPHERE: case EDQ_F_GAS: case EDQ_F_SMOKE:
                skip = solid[ci]; break;
            case EDQ_F_WAVE_SOURCE: skip = solid[ci] || is_vacuum[ci]; break;
            case EDQ_F_FIRE: skip = !flammable[ci]; break;
            case EDQ_F_HEAT: skip = false; break;
            case EDQ_F_GAS_ENERGY: skip = !accountable; break;
            default: B[EDQ_B_FAIL] += 1ULL; continue;   // never packed
        }
        if (skip) continue;
        B[EDQ_B_TILES] += 1ULL;
        if (er[EDQ_E_NOISY]) B[EDQ_B_NOISE] += 1ULL;
        const int64_t c = contrib[t];

        if (field == EDQ_F_HEAT) {
            // field_edit._combine_heat_q
            const int64_t old = heat[ci];
            const int64_t raw = c > 0 ? c : 0;
            int64_t nw;
            if (mode == MODE_ADD) {
                nw = (raw <= 0) ? old : ((old > I32_MAX - raw) ? I32_MAX : old + raw);
            } else if (mode == MODE_REMOVE) {
                nw = old - raw;
                if (nw < 0) nw = 0;
            } else {
                nw = old >= raw ? old : raw;
            }
            if (!fits_i32(nw)) { B[EDQ_B_FAIL] += 1ULL; continue; }
            heat[ci] = (int32_t)nw;
        } else if (field == EDQ_F_FIRE) {
            const int64_t nw = combine_q(fire[ci], c, mode, flags, er[EDQ_E_LO],
                                         er[EDQ_E_HI], 0);
            if (!fits_i32(nw)) { B[EDQ_B_FAIL] += 1ULL; continue; }
            fire[ci] = (int32_t)nw;
        } else if (field == EDQ_F_GAS || field == EDQ_F_SMOKE) {
            const int g = (int)er[EDQ_E_SLICE];
            int32_t& cell = V.g(g, i);
            const int64_t old = cell;
            const int64_t nw = combine_q(old, c, mode, flags, er[EDQ_E_LO],
                                         er[EDQ_E_HI], 0);
            const int64_t d = nw - old;
            if (d > 0) {
                // booked at its ATTEMPTED amount, then the counted saturation
                B[EDQ_B_DEP + g] += (unsigned long long)d;
                const int64_t tt = old + d;
                if (tt < 0) { B[EDQ_B_FAIL] += 1ULL; continue; }
                if (tt > TRACE_CELL_MAX) {
                    cell = (int32_t)TRACE_CELL_MAX;
                    B[EDQ_B_CLIP + g] += (unsigned long long)(tt - TRACE_CELL_MAX);
                } else {
                    cell = (int32_t)tt;
                }
            } else {
                if (!fits_i32(nw)) { B[EDQ_B_FAIL] += 1ULL; continue; }
                cell = (int32_t)nw;
            }
        } else if (field == EDQ_F_ATMOSPHERE) {
            int32_t& o2c = V.g(V.o2, i);
            int32_t& n2c = V.g(V.n2, i);
            const int64_t old_o2 = o2c, old_n2 = n2c;
            const int64_t new_o2 = combine_q(old_o2, div_round_i64(c * 21, 100), mode,
                                             flags, er[EDQ_E_O2_LO], er[EDQ_E_O2_HI], 1);
            const int64_t new_n2 = combine_q(old_n2, div_round_i64(c * 79, 100), mode,
                                             flags, er[EDQ_E_N2_LO], er[EDQ_E_N2_HI], 2);
            if (!fits_i32(new_o2) || !fits_i32(new_n2)) { B[EDQ_B_FAIL] += 1ULL; continue; }
            o2c = (int32_t)new_o2;
            n2c = (int32_t)new_n2;
            const int64_t dn = (new_o2 + new_n2) - (old_o2 + old_n2);
            if (dn != 0 && accountable) {
                // MINTED mass is born at ambient; mass leaving carries the
                // cell's own absolute mirror temperature.
                const int64_t t_abs = dn > 0 ? V.t_amb
                                             : (int64_t)temperature[ci] + V.t_amb;
                int64_t de;
                if (!mul_ok(dn, t_abs, &de)) { B[EDQ_B_FAIL] += 1ULL; continue; }
                V.deposit(i, de, EDQ_B_GE_ATM);
            }
        } else if (field == EDQ_F_WAVE_SOURCE) {
            const int64_t t_old = temperature[ci];
            int64_t t_new = combine_q(t_old, c, mode, flags, er[EDQ_E_LO],
                                      er[EDQ_E_HI], 0);
            if (t_new > T_MAX_PHYS_Q) t_new = T_MAX_PHYS_Q;
            if (accountable) {
                int64_t de;
                if (!mul_ok(V.bulk_n(i), t_new - t_old, &de)) {
                    B[EDQ_B_FAIL] += 1ULL;
                    continue;
                }
                V.deposit(i, de, EDQ_B_GE_WAVE);
            } else {
                if (!fits_i32(t_new)) { B[EDQ_B_FAIL] += 1ULL; continue; }
                temperature[ci] = (int32_t)t_new;
            }
        } else {   // EDQ_F_GAS_ENERGY (its skip mask is the accountable set)
            V.deposit(i, c, EDQ_B_GE_FE);
        }
    }
}

}  // namespace

int edit_queue_launch_resident(
        int n_env, int h, int w, int n_gases, int o2_idx, int n2_idx,
        int n_tiles, uint32_t salt,
        const int64_t* d_edits, const int64_t* d_tiles,
        const int64_t* d_env_off, const int64_t* d_env,
        int32_t* d_gas, int32_t* d_temperature, int64_t* d_gas_energy,
        int32_t* d_fire, int32_t* d_heat,
        const bool* d_solid, const bool* d_is_vacuum, const bool* d_is_ambient,
        const bool* d_thermal_solid, const bool* d_flammable,
        int64_t* d_contrib, unsigned long long* d_books) {
    if (n_env <= 0 || h <= 0 || w <= 0 || n_tiles <= 0) return 0;
    const int block = 128;
    int grid = (n_tiles + block - 1) / block;
    edq_contrib<<<grid, block, 0, tstream()>>>(d_contrib, d_edits, d_tiles, d_env, salt,
                                               n_tiles);
    cuda_check(cudaGetLastError(), "contrib launch");
    grid = (n_env + 31) / 32;
    edq_apply<<<grid, 32, 0, tstream()>>>(
        d_edits, d_tiles, d_env_off, d_env, d_contrib, n_env, h, w, n_gases, o2_idx,
        n2_idx, d_gas, d_temperature, d_gas_energy, d_fire, d_heat, d_solid,
        d_is_vacuum, d_is_ambient, d_thermal_solid, d_flammable, d_books);
    cuda_check(cudaGetLastError(), "apply launch");
    return 2;
}

void edit_queue_contrib_probe(const int64_t* edits, int n_edits, const int64_t* tiles,
                              int n_tiles, const int64_t* env, int n_env, uint32_t salt,
                              int64_t* out) {
    if (n_tiles <= 0) return;
    int64_t *d_e = nullptr, *d_t = nullptr, *d_env = nullptr, *d_o = nullptr;
    const size_t be = (size_t)(n_edits > 0 ? n_edits : 1) * EDQ_EDIT_COLS * 8;
    const size_t bt = (size_t)n_tiles * EDQ_TILE_COLS * 8;
    const size_t bv = (size_t)(n_env > 0 ? n_env : 1) * EDQ_ENV_COLS * 8;
    cuda_check(cudaMalloc(&d_e, be), "probe malloc edits");
    cuda_check(cudaMalloc(&d_t, bt), "probe malloc tiles");
    cuda_check(cudaMalloc(&d_env, bv), "probe malloc env");
    cuda_check(cudaMalloc(&d_o, (size_t)n_tiles * 8), "probe malloc out");
    if (n_edits > 0)
        cuda_check(cudaMemcpy(d_e, edits, (size_t)n_edits * EDQ_EDIT_COLS * 8,
                              cudaMemcpyHostToDevice), "probe H2D edits");
    cuda_check(cudaMemcpy(d_t, tiles, bt, cudaMemcpyHostToDevice), "probe H2D tiles");
    if (n_env > 0)
        cuda_check(cudaMemcpy(d_env, env, (size_t)n_env * EDQ_ENV_COLS * 8,
                              cudaMemcpyHostToDevice), "probe H2D env");
    edq_contrib<<<(n_tiles + 127) / 128, 128>>>(d_o, d_e, d_t, d_env, salt, n_tiles);
    cuda_check(cudaGetLastError(), "probe launch");
    cuda_check(cudaMemcpy(out, d_o, (size_t)n_tiles * 8, cudaMemcpyDeviceToHost),
               "probe D2H");
    cudaFree(d_e); cudaFree(d_t); cudaFree(d_env); cudaFree(d_o);
}

}  // namespace breach_cuda
