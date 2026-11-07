// ============================================================================
// #29 P4a — the resident tick's drivers (see cuda_resident_tick.h).
//
// Each driver is the per-call twin's HOST side moved onto persistent state:
// the same host folds (the families' *_bake_scalars, the sweep's own host
// door and builders), the same launch core, the same argument choices as
// PhysicsEngine::step_tail / PhysicsRunner._run_combustion /
// PhysicsEngine::step_water_tail make on the host path — but on the resident
// device fields, with persistent scratch, tables uploaded only when their
// host value changes, and every stage's books in ONE arena read back ONCE
// per tick by resident_tail. Bit-identity gate: tests/cuda_s8a_check.py
// (every synced field, every book, the closure identity, tol 0).
// ============================================================================
#include "cuda_resident_tick.h"
#include "cuda_resident.h"
#include "physics_engine.h"
#include "fixed_point.h"
#include "emissive_table.h"
#include "radiation_sweep.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace breach_cuda {

namespace {

using namespace fixedpoint;

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in resident_tick/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// A persistent device buffer: grows on demand, never shrinks, never freed per
// tick (the zero-malloc steady state).
struct DevBuf {
    void* p = nullptr;
    size_t bytes = 0;
    template <class T> T* get(size_t count) {
        const size_t need = count * sizeof(T);
        if (need > bytes) {
            if (p) cudaFree(p);
            p = nullptr;
            cuda_check(cudaMalloc(&p, need > 0 ? need : 8), "malloc scratch");
            bytes = need;
        }
        return static_cast<T*>(p);
    }
};

// A host copy of what a device table holds, so an unchanged table is never
// re-uploaded (memcmp, a few microseconds).
template <class T>
bool upload_if_changed(std::vector<T>& cache, DevBuf& dev, const T* src, size_t count,
                       const char* what) {
    T* d = dev.get<T>(count);
    if (cache.size() == count && (count == 0 || std::memcmp(cache.data(), src,
                                                            count * sizeof(T)) == 0))
        return false;
    cache.assign(src, src + count);
    if (count > 0)
        cuda_check(cudaMemcpy(d, src, count * sizeof(T), cudaMemcpyHostToDevice), what);
    return true;
}

// The books arena, in unsigned long long units: every stage's counter block,
// then the fire's counter / gate pair, then the destroyed-wall list (n ints),
// whose first DESTROYED_HEAD entries ride the one D2H.
constexpr int OFF_WT     = 0;
constexpr int OFF_COMB   = OFF_WT + WT_CNT_SLOTS;
constexpr int OFF_TEMP   = OFF_COMB + COMB_CNT_SLOTS;
constexpr int OFF_SWEEP  = OFF_TEMP + TEMP_CNT_SLOTS;
constexpr int OFF_FIRE   = OFF_SWEEP + RADIATION_SWEEP_CNT_SLOTS;   // int counter, int gate
constexpr int OFF_DLIST  = OFF_FIRE + 1;                            // int list[n]
constexpr int DESTROYED_HEAD = 64;
constexpr size_t HEAD_BYTES = (size_t)OFF_DLIST * 8 + (size_t)DESTROYED_HEAD * 4;

// The per-env scalars (N == 1), int32: one block, uploaded when it changes.
enum : int {
    ENV_T_AMB = 0,       // T_AMB_K raw (temperature, the sweep, combustion)
    ENV_K_LEAK = 1,      // the sweep's k_leak_q
    ENV_BURN_CAP = 2,    // combustion's burn_cap_q (from the env's dt)
    ENV_FIRE_DT = 3,     // fire's dt_q
    ENV_WATER_T_AMB = 4, // the water tail's t_amb_raw
    ENV_I32 = 5,
};

struct ResidentTickState {
    int h = 0, w = 0;
    // per-env scalars
    DevBuf env32, env64;
    std::vector<int32_t> env32_cache;
    std::vector<int64_t> env64_cache;
    int32_t env32_host[ENV_I32] = {0, 0, 0, 0, 0};
    // tables
    DevBuf e_table, l_table, heat_absorb, light_absorb, light_glow, ign_table, cons;
    std::vector<int64_t> e_table_c, l_table_c;
    std::vector<int32_t> heat_absorb_c, light_absorb_c, light_glow_c, ign_table_c;
    std::vector<uint8_t> cons_c;
    // scratch
    DevBuf n_bulk, temp_new, cap_used, cap_real, de_gas;
    DevBuf outflow, amb_m, ex_cell, f_q24, a_eff, d_eff;
    DevBuf l_outflow, l_emit, l_d, l_g, cone_index, cone_inj, emit_mult, sky;
    DevBuf tsnap, alloc, dep, e_slot, e_dep, perm_q, ign_plane;
    DevBuf wt_flag, wt_frac, wt_list;
    DevBuf stamp_rows, stamp_winner;   // #29 P4b
    // the books arena + its pinned host mirror
    DevBuf arena;
    unsigned char* arena_h = nullptr;
    size_t arena_h_bytes = 0;
    // this tick's stage flags (set by the drivers, consumed by resident_tail)
    bool water_ran = false;
    int  water_steam_idx = 0;
    int  water_n_gases = 0;
    bool comb_ran = false;
    int  comb_smoke_idx = 0;
    // this tick's light inputs (resident_upload_light)
    bool light_uploaded = false;
    bool l_cones = false, l_sky = false, l_emult = false;
    bool tables_uploaded_once = false;

    void key(int hh, int ww) {
        if (hh == h && ww == w) return;
        h = hh; w = ww;
        // A new grid: every per-grid cache is stale (the buffers re-grow).
        light_uploaded = false;
    }
    unsigned long long* arena_dev() {
        const size_t n = (size_t)h * (size_t)w;
        return arena.get<unsigned long long>((size_t)OFF_DLIST + (n + 1) / 2 + 1);
    }
    unsigned char* arena_host() {
        if (arena_h_bytes < HEAD_BYTES) {
            if (arena_h) cudaFreeHost(arena_h);
            arena_h = nullptr;
            cuda_check(cudaMallocHost((void**)&arena_h, HEAD_BYTES), "malloc pinned books");
            arena_h_bytes = HEAD_BYTES;
        }
        return arena_h;
    }
    void push_env32() {
        upload_if_changed(env32_cache, env32, env32_host, (size_t)ENV_I32, "H2D env scalars");
    }
};

ResidentTickState g_rt;
long long g_counts[RESIDENT_STAGE_COUNTERS] = {0};
bool g_comb_tables_uploaded = false;

// ---- the small kernels this driver owns ------------------------------------
// The bulk sum (step_tail's n_bulk_: Σ conservative planes, the int32 add the
// host loop makes, written here as the defined uint32 wrap -- the K_ntot
// idiom of cuda_eos_resident.cu, byte-identical to the host's observed wrap).
__global__ void rt_nbulk(int32_t* __restrict__ n_bulk, const int32_t* __restrict__ gas,
                         const bool* __restrict__ cons, int n_gases, int n, int n_env) {
    const int total = n_env * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        const int env = i / n;
        const int li = i - env * n;
        uint32_t acc = 0;
        for (int g = 0; g < n_gases; ++g) {
            if (!cons[g]) continue;
            acc += (uint32_t)gas[((size_t)env * n_gases + g) * n + li];
        }
        n_bulk[i] = (int32_t)acc;
    }
}

// The ignition plane: ignition_temp_q16[material] with numpy's negative-index
// wrap (index -1 is the last row), the table already narrowed to int32.
__global__ void rt_ignition(int32_t* __restrict__ ign, const int8_t* __restrict__ material,
                            const int32_t* __restrict__ table, int n_mat, int total) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        int m = (int)material[i];
        if (m < 0) m += n_mat;
        ign[i] = (m >= 0 && m < n_mat) ? table[m] : 0;
    }
}

// The permeability quantized once (combustion's pass-entry quantize, the
// per-call wrapper's host loop): quantize((double)perm), the kit's FP_HD door.
__global__ void rt_perm_q(int32_t* __restrict__ perm_q, const float* __restrict__ perm,
                          int total) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        perm_q[i] = quantize((double)perm[i]);
    }
}

template <class T> T* P(std::uintptr_t a) { return reinterpret_cast<T*>(a); }

}  // namespace

void resident_stage_counters(long long out[RESIDENT_STAGE_COUNTERS]) {
    for (int k = 0; k < RESIDENT_STAGE_COUNTERS; ++k) out[k] = g_counts[k];
}

void resident_sync_fleck_plane(const PhysicsEngine& eng, int h, int w) {
    ResidentTickState& S = g_rt;
    const size_t n = (size_t)h * (size_t)w;
    if (S.h != h || S.w != w || S.f_q24.p == nullptr || S.f_q24.bytes < n * 4) {
        throw std::logic_error(
            "resident_sync_fleck_plane: no resident sweep has run on this grid");
    }
    int32_t* f = eng.radiation.fleck_plane_for_twin(h, w, 16);
    cuda_check(cudaMemcpy(f, S.f_q24.p, n * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "D2H fleck plane");
}

void resident_ignition_probe(const int8_t* material, int n,
                             const int32_t* table, int n_mat, int32_t* out) {
    if (n <= 0) return;
    int8_t* d_m = nullptr;
    int32_t* d_t = nullptr;
    int32_t* d_o = nullptr;
    cuda_check(cudaMalloc(&d_m, (size_t)n), "probe malloc material");
    cuda_check(cudaMalloc(&d_t, (size_t)(n_mat > 0 ? n_mat : 1) * 4), "probe malloc table");
    cuda_check(cudaMalloc(&d_o, (size_t)n * 4), "probe malloc out");
    cuda_check(cudaMemcpy(d_m, material, (size_t)n, cudaMemcpyHostToDevice), "probe H2D");
    if (n_mat > 0)
        cuda_check(cudaMemcpy(d_t, table, (size_t)n_mat * 4, cudaMemcpyHostToDevice),
                   "probe H2D table");
    rt_ignition<<<(n + 255) / 256, 256>>>(d_o, d_m, d_t, n_mat, n);
    cuda_check(cudaGetLastError(), "probe launch");
    cuda_check(cudaMemcpy(out, d_o, (size_t)n * 4, cudaMemcpyDeviceToHost), "probe D2H");
    cudaFree(d_m); cudaFree(d_t); cudaFree(d_o);
}

// ---------------------------------------------------------------------------
// W5 + W3 + the seal + the copy
// ---------------------------------------------------------------------------
void resident_water_tail(const PhysicsEngine& eng, const ResidentFields& f,
                         const bool* gas_conservative, int steam_idx,
                         float sim_time, double ceiling_h, double flood_eps,
                         double ratio_cap, double boil_rate,
                         double boil_p_thresh, double steam_yield,
                         bool ambient, int32_t t_amb_raw) {
    (void)eng;
    ResidentTickState& S = g_rt;
    S.key(f.h, f.w);
    const int n = f.h * f.w;
    if (n <= 0) return;
    // gas_conservative, uploaded when it changes (bool -> one byte each).
    std::vector<uint8_t> cons_h((size_t)f.n_gases);
    for (int g = 0; g < f.n_gases; ++g) cons_h[(size_t)g] = gas_conservative[g] ? 1 : 0;
    upload_if_changed(S.cons_c, S.cons, cons_h.data(), cons_h.size(), "H2D cons");
    S.env32_host[ENV_WATER_T_AMB] = t_amb_raw;
    S.push_env32();
    const WaterTailScalars sc = water_tail_bake_scalars(
        sim_time, ceiling_h, flood_eps, ratio_cap, boil_rate, boil_p_thresh, steam_yield);
    unsigned long long* arena = S.arena_dev();
    water_tail_launch_resident(
        1, f.h, f.w, f.n_gases, steam_idx,
        P<int32_t>(f.water_depth), P<const int32_t>(f.atmosphere), P<const bool>(f.solid),
        P<int32_t>(f.gas), P<int32_t>(f.water_before), P<float>(f.dyn_permeability),
        P<int64_t>(f.gas_energy), S.cons.get<bool>((size_t)f.n_gases),
        P<const bool>(f.thermal_solid), P<const bool>(f.is_vacuum),
        ambient ? P<const bool>(f.is_ambient) : nullptr,
        S.env32.get<int32_t>(ENV_I32) + ENV_WATER_T_AMB, sc,
        S.wt_flag.get<uint8_t>((size_t)n), S.wt_frac.get<int32_t>((size_t)n),
        S.wt_list.get<int32_t>((size_t)n), arena + OFF_WT);
    S.water_ran = true;
    S.water_steam_idx = steam_idx;
    S.water_n_gases = f.n_gases;
    ++g_counts[0];
}

// ---------------------------------------------------------------------------
// #29 P4b: the stamp (slot 6's five planes)
// ---------------------------------------------------------------------------
void resident_stamp(const PhysicsEngine& eng, const ResidentFields& f,
                    const int32_t* rows, int n_rows) {
    (void)eng;
    ResidentTickState& S = g_rt;
    S.key(f.h, f.w);
    const int n = f.h * f.w;
    if (n <= 0) return;
    if (n_rows < 0 || (n_rows > 0 && rows == nullptr))
        throw std::invalid_argument("resident_stamp: bad rows");
    if (!f.permeability || !f.wave_absorb || !f.heat_atten_q || !f.light_atten_q ||
        !f.obstacles || !f.dyn_permeability || !f.dyn_wave_absorb ||
        !f.dyn_heat_atten_q || !f.dyn_light_atten_q)
        throw std::invalid_argument("resident_stamp: a stamp plane is not resident");
    int32_t* d_rows = nullptr;
    if (n_rows > 0) {
        d_rows = S.stamp_rows.get<int32_t>((size_t)n_rows * STAMP_ROW_COLS);
        cuda_check(cudaMemcpy(d_rows, rows,
                              (size_t)n_rows * STAMP_ROW_COLS * sizeof(int32_t),
                              cudaMemcpyHostToDevice), "H2D stamp rows");
    }
    stamp_launch_resident(
        1, f.h, f.w, n_rows,
        P<const float>(f.permeability), P<const float>(f.wave_absorb),
        P<const int32_t>(f.heat_atten_q), P<const int32_t>(f.light_atten_q),
        P<bool>(f.obstacles), P<float>(f.dyn_permeability),
        P<float>(f.dyn_wave_absorb), P<int32_t>(f.dyn_heat_atten_q),
        P<int32_t>(f.dyn_light_atten_q), d_rows,
        S.stamp_winner.get<int32_t>((size_t)n));
    ++g_counts[11];
}

// ---------------------------------------------------------------------------
// combustion
// ---------------------------------------------------------------------------
void resident_combustion(const PhysicsEngine& eng, const ResidentFields& f,
                         int o2_idx, int inert_n2_idx, int black_smoke_idx,
                         int draw_r, float sim_time,
                         const int32_t* ignition_table, int n_materials,
                         bool ambient, int32_t t_amb_q) {
    (void)ambient;   // combustion reads the ring plane whole (the host call's
                     // `is_ambient=gmap.is_ambient`, all false on a space map)
    ResidentTickState& S = g_rt;
    S.key(f.h, f.w);
    S.comb_ran = false;
    const int n = f.h * f.w;
    const int n_gases = f.n_gases;
    // --- the guards, VERBATIM combustion_step's head (a guarded return
    //     launches nothing: the CPU no-op) ---------------------------------
    if (f.h <= 0 || f.w <= 0 || sim_time <= 0.0f) return;
    if (o2_idx < 0 || o2_idx >= n_gases) return;
    if (inert_n2_idx < 0 || inert_n2_idx >= n_gases) return;
    if (black_smoke_idx < 0 || black_smoke_idx >= n_gases) return;
    const CombustionSolver& cs = eng.combustion;
    const q16 burn_cap_q = quantize((double)cs.burn_rate * (double)sim_time);
    const CombustionScalars sc = combustion_bake_scalars(
        cs.o2_thresh_burn, cs.o2_frac_ext, cs.o2_frac_full, cs.fire_T_ext,
        cs.fire_T_span, cs.hotf_cap, cs.H_BED_M, cs.H_BED_SHIFT, cs.fuel_per_o2,
        cs.soot_yield, cs.H_FUEL_M, cs.H_FUEL_SHIFT, eng.temperature.c_v,
        eng.temperature.n_floor_heat, cs.T_MAX_PHYS, cs.p_ext_q, cs.p_full_q);
    if (burn_cap_q <= 0) return;   // nothing burns this tick (fields untouched)
    if (draw_r < 1 || draw_r > combustion_draw::R_MAX) {
        throw std::runtime_error("resident combustion: draw_r out of range");
    }
    const int n_slots = combustion_draw::slot_count(draw_r);
    if (f.acc_depth < n_slots) {
        throw std::runtime_error("resident combustion: MAX_CLAIMANTS cap hit");
    }
    if (!g_comb_tables_uploaded) {          // the draw geometry, once per process
        combustion_upload_tables();
        g_comb_tables_uploaded = true;
    }
    upload_if_changed(S.ign_table_c, S.ign_table, ignition_table, (size_t)n_materials,
                      "H2D ignition table");
    S.env32_host[ENV_BURN_CAP] = burn_cap_q;
    S.env32_host[ENV_T_AMB] = t_amb_q;
    S.push_env32();
    const int block = 256, grid = (n + block - 1) / block;
    int32_t* ign = S.ign_plane.get<int32_t>((size_t)n);
    rt_ignition<<<grid, block>>>(ign, P<const int8_t>(f.material),
                                 S.ign_table.get<int32_t>((size_t)n_materials),
                                 n_materials, n);
    cuda_check(cudaGetLastError(), "ignition gather");
    int32_t* perm_q = nullptr;
    if (draw_r > 1) {                       // the per-call: only with the extended draw
        perm_q = S.perm_q.get<int32_t>((size_t)n);
        rt_perm_q<<<grid, block>>>(perm_q, P<const float>(f.dyn_permeability), n);
        cuda_check(cudaGetLastError(), "perm quantize");
    }
    unsigned long long* arena = S.arena_dev();
    int32_t* env = S.env32.get<int32_t>(ENV_I32);
    combustion_launch_resident(
        1, f.h, f.w, n_gases, o2_idx, inert_n2_idx, black_smoke_idx, draw_r,
        P<int32_t>(f.gas), P<int32_t>(f.temperature), P<int32_t>(f.wall_hp),
        P<int64_t>(f.gas_energy), P<const int32_t>(f.fire), ign,
        P<const bool>(f.flammable), P<const bool>(f.solid), P<const bool>(f.is_vacuum),
        P<const bool>(f.is_ambient), P<const bool>(f.thermal_solid),
        P<const int32_t>(f.heat_inv_shift),
        P<int32_t>(f.heat), P<int32_t>(f.dem_acc), f.acc_depth,
        perm_q,
        P<const int32_t>(f.fire_T_ext_plane), P<const int32_t>(f.fuel_per_o2_plane),
        P<const int32_t>(f.atmosphere),
        env + ENV_BURN_CAP, env + ENV_T_AMB, sc,
        S.tsnap.get<int32_t>((size_t)n), S.alloc.get<int32_t>((size_t)n_slots * n),
        S.dep.get<int32_t>((size_t)5 * n), S.e_slot.get<int64_t>((size_t)n_slots * n),
        S.e_dep.get<int64_t>((size_t)5 * n), arena + OFF_COMB);
    S.comb_ran = true;
    S.comb_smoke_idx = black_smoke_idx;
    ++g_counts[2];
}

// ---------------------------------------------------------------------------
// the light inputs (entry phase)
// ---------------------------------------------------------------------------
void resident_upload_light(const PhysicsEngine& eng, const ResidentFields& f,
                           const int32_t* gas_light_absorb_q16,
                           const int32_t* gas_light_glow_q16,
                           const int64_t* light_cones, int n_light_cones,
                           const int64_t* light_sky,
                           const int64_t* light_emit_mult, int n_light_emit_mult) {
    ResidentTickState& S = g_rt;
    S.key(f.h, f.w);
    const int n = f.h * f.w;
    // The sweep's own host door, on the SAME LightChannels run_sweep_ builds
    // (the plane pointers are device addresses here: the door only checks
    // that they are given; the tables, rows and sky it reads are host).
    LightChannels lc;
    lc.light_atten_q     = P<const int32_t>(f.light_atten_q);
    lc.dyn_light_atten_q = P<const int32_t>(f.dyn_light_atten_q);
    lc.l_table           = eng.light_emission.table();
    lc.light_absorb_q16  = gas_light_absorb_q16;
    lc.light_glow_q16    = gas_light_glow_q16;
    lc.transport         = eng.light_transport;
    lc.light_q           = P<int64_t>(f.light_q);
    lc.light_flux_q      = P<int64_t>(f.light_flux_q);
    lc.light_glow        = P<int64_t>(f.light_glow);
    lc.cones             = light_cones;
    lc.n_cones           = (light_cones != nullptr) ? n_light_cones : 0;
    lc.sky               = light_sky;
    lc.emit_mult         = light_emit_mult;
    lc.n_emit_mult       = (light_emit_mult != nullptr) ? n_light_emit_mult : 0;
    if (lc.light_atten_q == nullptr || lc.dyn_light_atten_q == nullptr ||
        lc.light_q == nullptr || lc.light_flux_q == nullptr || lc.light_glow == nullptr) {
        throw std::invalid_argument(
            "PhysicsEngine: light is requested but the light planes (light_atten_q, "
            "dyn_light_atten_q, light_q, light_flux_q, light_glow) were not all "
            "handed in (P6a)");
    }
    RadiationSweep::validate_light_group(lc, 16, P<const int32_t>(f.gas),
                                         S.n_bulk.get<int32_t>((size_t)n), f.n_gases,
                                         f.h, f.w);
    upload_if_changed(S.l_table_c, S.l_table, lc.l_table,
                      (size_t)L_CHANNELS * E_TABLE_SIZE, "H2D l_table");
    S.l_cones = lc.n_cones > 0 && n > 0;
    if (S.l_cones) {
        ConeInjection cinj;
        RadiationSweep::build_cone_injection(lc.cones, lc.n_cones, 16, f.h, f.w, cinj);
        cuda_check(cudaMemcpy(S.cone_index.get<int32_t>((size_t)n), cinj.index.data(),
                              (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "H2D cone_index");
        cuda_check(cudaMemcpy(S.cone_inj.get<int64_t>(cinj.inj.size()), cinj.inj.data(),
                              cinj.inj.size() * sizeof(int64_t), cudaMemcpyHostToDevice),
                   "H2D cone_inj");
    }
    S.l_sky = (lc.sky != nullptr);
    if (S.l_sky) {
        cuda_check(cudaMemcpy(S.sky.get<int64_t>((size_t)16 * 3), lc.sky,
                              (size_t)16 * 3 * sizeof(int64_t), cudaMemcpyHostToDevice),
                   "H2D sky");
    }
    S.l_emult = lc.n_emit_mult > 0 && n > 0;
    if (S.l_emult) {
        std::vector<int32_t> emult;
        RadiationSweep::build_emit_mult(lc.emit_mult, lc.n_emit_mult, f.h, f.w, emult);
        cuda_check(cudaMemcpy(S.emit_mult.get<int32_t>((size_t)n), emult.data(),
                              (size_t)n * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "H2D emit_mult");
    }
    S.light_uploaded = true;
}

// ---------------------------------------------------------------------------
// the tail: the bulk sum, fire, the sweep, temperature; the books; destroyed
// ---------------------------------------------------------------------------
std::vector<std::pair<int, int>> resident_tail(
        const PhysicsEngine& eng, const ResidentFields& f,
        const bool* gas_conservative, int o2_idx, float sim_time,
        bool ambient, int32_t t_amb_q, int32_t k_leak_q, int64_t rad_amb_vacuum_q,
        const int32_t* gas_heat_absorb_q16,
        const int32_t* gas_light_absorb_q16, const int32_t* gas_light_glow_q16) {
    ResidentTickState& S = g_rt;
    S.key(f.h, f.w);
    const int h = f.h, w = f.w, n = h * w, n_gases = f.n_gases;
    std::vector<std::pair<int, int>> destroyed;
    if (n <= 0) return destroyed;
    const bool with_light = eng.light_requested;
    if (with_light && !S.light_uploaded) {
        throw std::logic_error(
            "resident_tail: light is requested but resident_upload_light did not "
            "run this tick (the entry phase hands the light inputs over)");
    }

    // ---- per-env scalars and tables (uploaded only when changed) ---------
    std::vector<uint8_t> cons_h((size_t)n_gases);
    for (int g = 0; g < n_gases; ++g) cons_h[(size_t)g] = gas_conservative[g] ? 1 : 0;
    upload_if_changed(S.cons_c, S.cons, cons_h.data(), cons_h.size(), "H2D cons");
    S.env32_host[ENV_T_AMB] = t_amb_q;
    S.env32_host[ENV_K_LEAK] = k_leak_q;
    S.env32_host[ENV_FIRE_DT] = fire_dt_q(sim_time);
    S.push_env32();
    upload_if_changed(S.env64_cache, S.env64, &rad_amb_vacuum_q, (size_t)1,
                      "H2D vac level");
    upload_if_changed(S.e_table_c, S.e_table, eng.emissive.table(),
                      (size_t)E_TABLE_SIZE, "H2D e_table");
    int32_t* env = S.env32.get<int32_t>(ENV_I32);
    unsigned long long* arena = S.arena_dev();
    const int block = 256, grid = (n + block - 1) / block;

    // ---- the bulk sum (step_tail's n_bulk_, built ONCE before the fire) ---
    int32_t* n_bulk = S.n_bulk.get<int32_t>((size_t)n);
    rt_nbulk<<<grid, block>>>(n_bulk, P<const int32_t>(f.gas),
                              S.cons.get<bool>((size_t)n_gases), n_gases, n, 1);
    cuda_check(cudaGetLastError(), "n_bulk");

    // ---- 2. fire (its P1 early exit as the device gate) ------------------
    const FireParams& fp = eng.fire.params;
    const FireScalars fsc = fire_bake_scalars(
        fp.k_grow, fp.k_die, fp.fire_T_ext, fp.fire_T_span, fp.fuel_ref,
        fp.o2_frac_ext, fp.I_min, fp.k_wind_fan, fp.k_wind_strip, fp.wall_damage,
        fp.temp_scale, fp.I_cap_per_avail, fp.o2_frac_amb, fp.o2f_cap, fp.hotf_cap,
        fp.p_ext_q, fp.p_full_q);
    int* fire_pair = reinterpret_cast<int*>(arena + OFF_FIRE);   // [counter, gate]
    int* dlist = reinterpret_cast<int*>(arena + OFF_DLIST);
    fire_launch_resident(
        1, h, w, P<int32_t>(f.fire), P<const int32_t>(f.atmosphere),
        P<const int32_t>(f.gas), n_gases, o2_idx, n_bulk, P<int32_t>(f.wall_hp),
        P<const int32_t>(f.temperature), P<const int32_t>(f.wind_x),
        P<const int32_t>(f.wind_y), P<const bool>(f.solid), P<const bool>(f.is_vacuum),
        P<const bool>(f.flammable), P<const int64_t>(f.fuel_recip),
        P<const int32_t>(f.fire_T_ext_plane), env + ENV_FIRE_DT, fsc,
        reinterpret_cast<int32_t*>(fire_pair + 1), fire_pair, dlist);
    ++g_counts[3];

    // ---- 2b. the radiation sweep (heat; light when requested) -------------
    // The ACTIVE gases decide the gas group, as the per-call wrapper decides
    // it: a zero coefficient adds exactly 0 to a density sum, so the full
    // table over the full gas block is the packed active set, integer for
    // integer -- and with no active gas at all the group is not given (the
    // pre-P5a sweep, the per-call's n_act == 0).
    const bool smoke = (gas_heat_absorb_q16 != nullptr);
    const bool lsmoke = with_light && smoke && gas_light_absorb_q16 != nullptr;
    bool any_active = false;
    for (int g = 0; g < n_gases && smoke; ++g) {
        if (gas_heat_absorb_q16[g] < 0 ||
            gas_heat_absorb_q16[g] > RadiationSweep::HEAT_ABSORB_Q_MAX) {
            throw std::invalid_argument(
                "resident sweep: a heat_absorb_q16 outside [0, 2^28] (the gas door's "
                "bound)");
        }
        any_active = any_active || gas_heat_absorb_q16[g] != 0;
        if (lsmoke) {
            for (int c = 0; c < 3; ++c) {
                any_active = any_active || gas_light_absorb_q16[g * 3 + c] != 0 ||
                             gas_light_glow_q16[g * 3 + c] != 0;
            }
        }
    }
    if (smoke && n_gases > RadiationSweep::N_GAS_PLANES_MAX) {
        throw std::invalid_argument("resident sweep: n_gases above N_GAS_PLANES_MAX");
    }
    const bool gas_on = smoke && any_active;
    if (gas_on) {
        upload_if_changed(S.heat_absorb_c, S.heat_absorb, gas_heat_absorb_q16,
                          (size_t)n_gases, "H2D heat_absorb");
    }
    const PhysicsEngine::GasCapacityQ cap = eng.gas_capacity_q();
    RadiationLightDev ldev{};
    if (with_light) {
        const bool l_gas = lsmoke && gas_on;
        if (l_gas) {
            upload_if_changed(S.light_absorb_c, S.light_absorb, gas_light_absorb_q16,
                              (size_t)n_gases * 3, "H2D light_absorb");
            upload_if_changed(S.light_glow_c, S.light_glow, gas_light_glow_q16,
                              (size_t)n_gases * 3, "H2D light_glow");
        }
        ldev.d_light_atten_q     = P<const int32_t>(f.light_atten_q);
        ldev.d_dyn_light_atten_q = P<const int32_t>(f.dyn_light_atten_q);
        ldev.d_l_table           = S.l_table.get<int64_t>((size_t)L_CHANNELS * E_TABLE_SIZE);
        ldev.d_light_absorb_q16  = l_gas ? S.light_absorb.get<int32_t>((size_t)n_gases * 3) : nullptr;
        ldev.d_light_glow_q16    = l_gas ? S.light_glow.get<int32_t>((size_t)n_gases * 3) : nullptr;
        ldev.transport           = eng.light_transport;
        ldev.d_l_outflow         = S.l_outflow.get<int64_t>((size_t)16 * n * 3);
        ldev.d_l_emit            = S.l_emit.get<int64_t>((size_t)n * 3);
        ldev.d_l_d               = S.l_d.get<int32_t>((size_t)n * 3);
        ldev.d_l_g               = S.l_g.get<int32_t>((size_t)n * 3);
        ldev.d_light_q           = P<int64_t>(f.light_q);
        ldev.d_light_flux_q      = P<int64_t>(f.light_flux_q);
        ldev.d_light_glow        = P<int64_t>(f.light_glow);
        if (S.l_sky)   ldev.d_sky = S.sky.get<int64_t>((size_t)16 * 3);
        if (S.l_cones) {
            ldev.d_cone_index = S.cone_index.get<int32_t>((size_t)n);
            ldev.d_cone_inj   = static_cast<const int64_t*>(S.cone_inj.p);
        }
        if (S.l_emult) ldev.d_emit_mult = S.emit_mult.get<int32_t>((size_t)n);
    }
    int64_t* sweep_cnt = reinterpret_cast<int64_t*>(arena + OFF_SWEEP);
    radiation_sweep_launch_resident(
        1, h, w,
        P<const int32_t>(f.temperature), P<const int32_t>(f.heat_atten_q),
        P<const int32_t>(f.dyn_heat_atten_q), P<const int32_t>(f.heat_inv_shift),
        P<const bool>(f.thermal_solid),
        /*d_amb_level=*/nullptr, P<const bool>(f.is_vacuum),
        S.e_table.get<int64_t>((size_t)E_TABLE_SIZE), eng.emissive.fine_bits,
        S.env64.get<int64_t>(1), env + ENV_K_LEAK, env + ENV_T_AMB,
        gas_on ? P<const int32_t>(f.gas) : nullptr, gas_on ? n_gases : 0,
        gas_on ? S.heat_absorb.get<int32_t>((size_t)n_gases) : nullptr,
        gas_on ? n_bulk : nullptr,
        cap.n_floor_q, cap.recip_cv,
        RadiationSweep::TRANSPORT_SHEAR, 16, /*fleck_enabled=*/true,
        S.outflow.get<int64_t>((size_t)16 * n), S.amb_m.get<int64_t>((size_t)n),
        S.ex_cell.get<int64_t>((size_t)n), S.f_q24.get<int32_t>((size_t)n),
        S.a_eff.get<int32_t>((size_t)n), S.d_eff.get<int32_t>((size_t)n),
        P<int64_t>(f.rad_net_sweep), P<int64_t>(f.rad_flux_sweep),
        P<int64_t>(f.rad_amb_sweep), P<int64_t>(f.rad_fluence),
        sweep_cnt, with_light ? &ldev : nullptr);
    ++g_counts[5];
    if (with_light) ++g_counts[6];
    S.light_uploaded = false;   // the next tick hands its own inputs over

    // ---- 3. temperature ---------------------------------------------------
    const TemperatureSolver& ts = eng.temperature;
    const TemperatureScalars tsc = temperature_bake_scalars(
        ts.c_v, ts.n_floor_heat, ts.T_MAX_PHYS, ts.no_face, eng.emissive.fine_bits);
    const bool* d_ts = P<const bool>(f.thermal_solid);
    temperature_launch_resident(
        1, h, w, P<int32_t>(f.temperature), S.temp_new.get<int32_t>((size_t)n),
        P<const int32_t>(f.heat), P<const int32_t>(f.heat_inv_shift),
        P<const int32_t>(f.face_shift),
        P<const bool>(f.solid), d_ts, P<const bool>(f.is_vacuum),
        ambient ? P<const bool>(f.is_ambient) : nullptr,
        n_bulk, P<const int64_t>(f.rad_net_sweep),
        P<const int64_t>(f.rad_fluence), S.e_table.get<int64_t>((size_t)E_TABLE_SIZE),
        P<int64_t>(f.gas_energy), S.de_gas.get<int64_t>((size_t)n), env + ENV_T_AMB,
        S.cap_used.get<int64_t>((size_t)n), S.cap_real.get<int64_t>((size_t)n),
        tsc, arena + OFF_TEMP, /*solid_books=*/true);
    ++g_counts[7];

    // ---- THE ONE BOOKS D2H (it waits for the whole chain) -----------------
    unsigned char* hb = S.arena_host();
    cuda_check(cudaMemcpy(hb, arena, HEAD_BYTES, cudaMemcpyDeviceToHost), "D2H books");
    ++g_counts[8];
    const unsigned long long* hu = reinterpret_cast<const unsigned long long*>(hb);
    auto b = [&](int off, int slot) { return (int64_t)hu[off + slot]; };

    // ---- the sweep's ingress verdict FIRST (the per-call order) ----------
    const int64_t* sc = reinterpret_cast<const int64_t*>(hu + OFF_SWEEP);
    if (sc[RS_SLOT_BAD_SCALARS] & RS_BAD_K_LEAK)
        throw std::invalid_argument("resident sweep: k_leak_q outside [0, ONE]");
    if (sc[RS_SLOT_BAD_SCALARS] & RS_BAD_VAC_LEVEL)
        throw std::invalid_argument("resident sweep: the vacuum ambient level exceeds E0");
    if (sc[RS_SLOT_BAD_SCALARS] & RS_BAD_HEAT_ABSORB)
        throw std::invalid_argument("resident sweep: a heat_absorb_q16 outside [0, 2^28]");
    if (sc[RS_SLOT_BAD_EXTINCTION] != 0)
        throw std::invalid_argument(
            "resident sweep: extinction planes violate 0 <= a <= d <= ONE");
    if (sc[RS_SLOT_BAD_AMBIENT] != 0)
        throw std::invalid_argument("resident sweep: amb_level outside [0, E0]");
    if (sc[RS_SLOT_BAD_LIGHT_EXTINCTION] != 0)
        throw std::invalid_argument(
            "resident sweep: the light extinction planes violate 0 <= a <= d <= ONE");
    eng.radiation.min_stream = sc[RS_SLOT_MIN_STREAM];
    eng.radiation.max_stream = sc[RS_SLOT_MAX_STREAM];
    if (with_light) {
        int64_t lbooks[12];
        for (int c = 0; c < 3; ++c) {
            lbooks[c]     = sc[RS_SLOT_LIGHT_EMIT + c];
            lbooks[3 + c] = sc[RS_SLOT_LIGHT_ABSORB + c];
            lbooks[6 + c] = sc[RS_SLOT_LIGHT_RING_IN + c];
            lbooks[9 + c] = sc[RS_SLOT_LIGHT_RING_OUT + c];
        }
        eng.radiation.set_light_telemetry_from_twin(
            lbooks, sc[RS_SLOT_LIGHT_MIN_STREAM], sc[RS_SLOT_LIGHT_MAX_STREAM]);
    }

    // ---- the water tail's books (the CPU tail's own writes) ---------------
    if (S.water_ran) {
        eng.e_water_evac_export_sum = b(OFF_WT, WT_EXPORT);     // per call: ASSIGNED
        eng.eos.ensure_trace_books(S.water_n_gases);
        eng.eos.trace_mint_sum_[S.water_steam_idx] += b(OFF_WT, WT_MINT);
        for (int g = 0; g < S.water_n_gases; ++g) {
            eng.eos.trace_clip_sum_[g] += b(OFF_WT, WT_CLIP + g);
            eng.eos.trace_neg_hits_[g] += b(OFF_WT, WT_NEG + g);
        }
        if (b(OFF_WT, WT_N_BOIL) > 0 || b(OFF_WT, WT_N_EVAC) > 0) ++g_counts[1];
        if (b(OFF_WT, WT_N_BOIL) > 0) ++g_counts[9];
        if (b(OFF_WT, WT_N_EVAC) > 0) ++g_counts[10];
        S.water_ran = false;
    }
    // ---- combustion's books (the binding's fold, the CPU step's +=) ------
    if (S.comb_ran) {
        const CombustionSolver& cs = eng.combustion;
        cs.heat_floor_hits    += b(OFF_COMB, COMB_HEAT_FLOOR_HITS);
        cs.t_max_phys_hits    += b(OFF_COMB, COMB_T_MAX_PHYS_HITS);
        cs.e_deposit_drop_sum += b(OFF_COMB, COMB_DEPOSIT_DROP);
        cs.e_comb_draw_sum       += b(OFF_COMB, COMB_E_DRAW);
        cs.e_comb_mint_sum       += b(OFF_COMB, COMB_E_MINT);
        cs.e_comb_deliver_sum    += b(OFF_COMB, COMB_E_DELIVER);
        cs.e_soot_shed_sum       += b(OFF_COMB, COMB_E_SOOT_SHED);
        cs.e_ts_products_sum     += b(OFF_COMB, COMB_E_TS_PRODUCTS);
        cs.e_comb_export_sum     += b(OFF_COMB, COMB_E_EXPORT);
        cs.e_comb_heat_sum       += b(OFF_COMB, COMB_E_HEAT);
        cs.e_comb_rail_sum       += b(OFF_COMB, COMB_E_RAIL);
        cs.e_comb_solid_heat_sum += b(OFF_COMB, COMB_E_SOLID_HEAT);
        // #88 P2: the soot mint's books, into the EOS's per-gas trace books.
        eng.eos.ensure_trace_books(n_gases);
        eng.eos.trace_mint_sum_[S.comb_smoke_idx] += b(OFF_COMB, COMB_TRACE_MINT);
        eng.eos.trace_clip_sum_[S.comb_smoke_idx] += b(OFF_COMB, COMB_TRACE_CLIP);
        eng.eos.trace_neg_hits_[S.comb_smoke_idx] += b(OFF_COMB, COMB_TRACE_NEG);
        S.comb_ran = false;
    }
    // ---- temperature's books (step_tail's CUDA branch, slot for slot) ----
    {
        const TemperatureSolver& t = eng.temperature;
        t.t_max_phys_hits    += b(OFF_TEMP, TEMP_CNT_HITS);
        t.t_low_rail_hits    += b(OFF_TEMP, TEMP_CNT_LOW_HITS);
        t.e_cond_trunc_sum   += b(OFF_TEMP, 0);
        t.e_cond_cap_sum     += b(OFF_TEMP, 1);
        t.cond_limit_hits    += b(OFF_TEMP, 2);
        t.e_vac_wipe_sum     += b(OFF_TEMP, 3);
        t.e_ring_pin_sum     += b(OFF_TEMP, 4);
        t.e_deposit_drop_sum += b(OFF_TEMP, 5);
        t.e_gas_deposit_sum  += b(OFF_TEMP, 6);
        t.e_gas_cond_sum     += b(OFF_TEMP, 7);
        t.e_gas_rail_sum     += b(OFF_TEMP, 8);
        t.e_solid_deposit_sum += b(OFF_TEMP, 9);
        t.e_solid_cond_sum    += b(OFF_TEMP, 10);
        t.solid_energy_books_sum = b(OFF_TEMP, TEMP_CNT_SOLID_BOOKS);   // =, not +=
        t.rad_clamp_hits      += b(OFF_TEMP, 11);
        t.e_rad_clamp_drop_sum += b(OFF_TEMP, 12);
        t.e_rad_boundary_export_sum += b(OFF_TEMP, 13);
        t.e_rad_floor_drop_sum      += b(OFF_TEMP, 14);
        t.e_solid_low_rail_sum      += b(OFF_TEMP, 15);
        t.e_dep_floor_drop_sum      += b(OFF_TEMP, 16);
    }
    // ---- the destroyed walls (row-major, the per-call's sort) ------------
    const int* fh = reinterpret_cast<const int*>(hu + OFF_FIRE);
    int counter = fh[0];
    if (fh[1] >= fsc.max_fire_thresh_q) ++g_counts[4];
    if (counter > 0) {
        if (counter > n) counter = n;   // defensive (cannot exceed n slots)
        std::vector<int> idx((size_t)counter);
        const int* head = reinterpret_cast<const int*>(hu + OFF_DLIST);
        const int k0 = std::min(counter, DESTROYED_HEAD);
        std::copy(head, head + k0, idx.begin());
        if (counter > DESTROYED_HEAD) {
            cuda_check(cudaMemcpy(idx.data() + DESTROYED_HEAD, dlist + DESTROYED_HEAD,
                                  (size_t)(counter - DESTROYED_HEAD) * sizeof(int),
                                  cudaMemcpyDeviceToHost), "D2H destroyed tail");
        }
        destroyed = fire_destroyed_row_major(std::move(idx), w);
    }
    return destroyed;
}

}  // namespace breach_cuda
