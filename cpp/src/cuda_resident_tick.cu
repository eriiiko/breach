// ============================================================================
// #29 P4a + P5 — the resident tick's drivers (see cuda_resident_tick.h).
//
// Each driver is the per-call twin's HOST side moved onto persistent state:
// the same host folds (the families' *_bake_scalars, the sweep's own host
// door and builders), the same launch core, the same argument choices as
// PhysicsEngine::step_tail / PhysicsRunner._run_combustion /
// PhysicsEngine::step_water_tail make on the host path — but on the resident
// device fields, with persistent scratch, tables uploaded only when their
// host value changes, and every stage's books in ONE arena read back ONCE
// per tick by resident_tail.
//
// #29 P5 (docs/residency_batching_design_2026-11-07.md §4.2): THE BATCH. The
// fields are (N, h, w) buffers of N homogeneous envs; every launch core runs
// over all N; the per-env scalars are (N,) device columns; the arena holds
// every stage's books PER ENV, region by region ((N, slots) each, the cores'
// own contract), and resident_tail folds env e's block into engs[e]. Every
// launch, copy and memset runs on THE tick stream (tstream); the only host
// syncs in a tick are the EOS stage's counted max(n_sub) read and the books
// D2H here. Bit-identity gates: tests/cuda_s8a_check.py (N = 1, every synced
// field, every book, the closure identity, tol 0) and
// tests/cuda_batch_invariance_check.py (env i of N == env i alone).
// ============================================================================
#include "cuda_resident_tick.h"
#include "cuda_resident.h"
#include "cuda_eos_resident.h"
#include "cuda_batch.cuh"
#include "physics_engine.h"
#include "fixed_point.h"
#include "emissive_table.h"
#include "radiation_sweep.h"
#include "bulk_transport.h"   // trace_diffusion_dd_q (the CPU tail's fold)
#include "trace_decay.h"      // trace_decay::frac_q

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace breach_cuda {

// THE ONE TICK STREAM (cuda_batch.cuh): created on first use, never destroyed
// (process lifetime); a blocking stream so the legacy stream's work (CuPy's
// copies, the per-call twins) stays ordered with it.
void* tick_stream_handle() {
    static cudaStream_t s = [] {
        cudaStream_t st = nullptr;
        if (cudaStreamCreate(&st) != cudaSuccess) {
            throw std::runtime_error("resident_tick: cudaStreamCreate failed");
        }
        return st;
    }();
    return static_cast<void*>(s);
}

namespace {
struct CaptureProbe {
    bool on = false;
    int segments = 0;
    long long nodes = 0;
    std::vector<std::string> points;
};
CaptureProbe g_cap;

void capture_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in resident capture/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

// End the open segment, run it, count it.
void capture_close_segment() {
    cudaStream_t st = static_cast<cudaStream_t>(tick_stream_handle());
    cudaGraph_t g = nullptr;
    capture_check(cudaStreamEndCapture(st, &g), "end capture (an op the capture refused?)");
    size_t nn = 0;
    capture_check(cudaGraphGetNodes(g, nullptr, &nn), "graph nodes");
    cudaGraphExec_t ex = nullptr;
    capture_check(cudaGraphInstantiate(&ex, g, 0), "graph instantiate");
    capture_check(cudaGraphLaunch(ex, st), "graph launch");
    capture_check(cudaStreamSynchronize(st), "graph sync");
    cudaGraphExecDestroy(ex);
    cudaGraphDestroy(g);
    g_cap.nodes += (long long)nn;
    ++g_cap.segments;
}
}  // namespace

void tick_sync_point(const char* name) {
    cudaStream_t st = static_cast<cudaStream_t>(tick_stream_handle());
    if (!g_cap.on) {
        capture_check(cudaStreamSynchronize(st), name);
        return;
    }
    capture_close_segment();
    g_cap.points.emplace_back(name);
    capture_check(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "begin capture");
}

void resident_capture_begin() {
    cudaStream_t st = static_cast<cudaStream_t>(tick_stream_handle());
    capture_check(cudaStreamSynchronize(st), "sync before capture");
    g_cap = CaptureProbe{};
    capture_check(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "begin capture");
    g_cap.on = true;
}

CaptureSummary resident_capture_end() {
    CaptureSummary out;
    if (!g_cap.on) throw std::logic_error("resident_capture_end: no capture is open");
    g_cap.on = false;
    capture_close_segment();
    out.segments = g_cap.segments;
    out.nodes = g_cap.nodes;
    out.sync_points = g_cap.points;
    return out;
}

void resident_capture_abort() {
    cudaStream_t st = static_cast<cudaStream_t>(tick_stream_handle());
    g_cap.on = false;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(st, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(st, &g);
        if (g) cudaGraphDestroy(g);
    }
    cudaGetLastError();   // clear a sticky capture error
}

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

// A pinned host staging buffer (async H2D / D2H on the tick stream).
struct PinBuf {
    void* p = nullptr;
    size_t bytes = 0;
    template <class T> T* get(size_t count) {
        const size_t need = count * sizeof(T);
        if (need > bytes) {
            if (p) cudaFreeHost(p);
            p = nullptr;
            cuda_check(cudaMallocHost(&p, need > 0 ? need : 8), "malloc pinned");
            bytes = need;
        }
        return static_cast<T*>(p);
    }
};

// A host copy of what a device table holds, so an unchanged table is never
// re-uploaded (memcmp, a few microseconds). The copy runs only on a change
// (the first tick, a config reload) — never in a steady-state tick.
template <class T>
bool upload_if_changed(std::vector<T>& cache, DevBuf& dev, const T* src, size_t count,
                       const char* what) {
    T* d = dev.get<T>(count);
    if (cache.size() == count && (count == 0 || std::memcmp(cache.data(), src,
                                                            count * sizeof(T)) == 0))
        return false;
    cache.assign(src, src + count);
    if (count > 0) {
        cuda_check(cudaStreamSynchronize(tstream()), what);
        cuda_check(cudaMemcpy(d, src, count * sizeof(T), cudaMemcpyHostToDevice), what);
    }
    return true;
}

// THE BOOKS ARENA, in unsigned long long units, REGION-MAJOR: each stage's
// (N, slots) block (the launch cores' own per-env contract), the EOS stage's
// blocks contiguous (one memset zeroes them), the pre-stage, max(n_sub), the
// fire's (N,) counter and gate. The destroyed-wall lists are their own
// (N, h, w) buffer; their first DESTROYED_HEAD entries per env ride the D2H.
constexpr int DESTROYED_HEAD = 64;
struct ArenaLayout {
    size_t wt, comb, temp, sweep, kick, flux, ecnt, rail, tcnt, ttail, pre, nsub,
           fire_cnt, fire_gate, end;
    explicit ArenaLayout(int N = 1) {
        const size_t n = (size_t)(N > 0 ? N : 1);
        wt    = 0;
        comb  = wt    + n * WT_CNT_SLOTS;
        temp  = comb  + n * COMB_CNT_SLOTS;
        sweep = temp  + n * TEMP_CNT_SLOTS;
        kick  = sweep + n * RADIATION_SWEEP_CNT_SLOTS;
        flux  = kick  + n * KICK_CNT_SLOTS;
        ecnt  = flux  + n * FLUX_CNT_SLOTS;
        rail  = ecnt  + n * EOS_ECNT_SLOTS;
        tcnt  = rail  + n * EOS_BOOK_GAS_MAX;
        ttail = tcnt  + n * EOS_TCNT_SLOTS;
        pre   = ttail + n * EOS_BOOK_GAS_MAX * TRACE_TAIL_CNT_SLOTS;
        nsub  = pre   + n * EOS_PRE_SLOTS;
        fire_cnt  = nsub + 1;
        fire_gate = fire_cnt + (n + 1) / 2;   // (N,) int
        end       = fire_gate + (n + 1) / 2;  // (N,) int32
    }
};

// The per-env scalars, int32, COLUMN-major (scalar k of env e at k·N + e),
// so each launch core reads its (N,) column. A batch is homogeneous: every
// env's value is the config's.
enum : int {
    ENV_T_AMB = 0,       // T_AMB_K raw (temperature, the sweep, combustion)
    ENV_K_LEAK = 1,      // the sweep's k_leak_q
    ENV_BURN_CAP = 2,    // combustion's burn_cap_q (from the env's dt)
    ENV_FIRE_DT = 3,     // fire's dt_q
    ENV_WATER_T_AMB = 4, // the water tail's t_amb_raw
    ENV_I32 = 5,
};

struct ResidentTickState {
    int n_env = 0, h = 0, w = 0;
    ArenaLayout L;
    // per-env scalars
    DevBuf env32, env64;
    std::vector<int32_t> env32_cache;
    std::vector<int64_t> env64_cache;
    std::vector<int32_t> env32_host;
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
    DevBuf wt_flag, wt_frac, wt_list, water_active;
    PinBuf water_active_h;
    DevBuf stamp_rows, stamp_winner;   // #29 P4b
    PinBuf stamp_rows_h;               // its pinned staging copy (async H2D)
    DevBuf tt_scal, tt_s0;             // #29 P5: the trace tail's folds + snapshot
    std::vector<int32_t> tt_scal_c;
    DevBuf dlist;                      // (N, h, w) destroyed-wall lists
    // the books arena + its pinned host mirror
    DevBuf arena;
    PinBuf arena_h;
    // this tick's stage flags (set by the drivers, consumed by resident_tail)
    bool water_ran = false;
    int  water_steam_idx = 0;
    int  water_n_gases = 0;
    std::vector<uint8_t> water_active_env;   // (N,) the envs the water stage ran on
    bool comb_ran = false;
    int  comb_smoke_idx = 0;
    bool eos_ran = false;
    std::vector<int> eos_cons, eos_trace_gi;
    bool eos_rail = false;
    bool tt_ran = false;
    std::vector<int> tt_trace_gi;
    // this tick's light inputs (resident_upload_light)
    bool light_uploaded = false;
    bool l_cones = false, l_sky = false, l_emult = false;

    void key(int nn, int hh, int ww) {
        if (nn == n_env && hh == h && ww == w) return;
        n_env = nn; h = hh; w = ww;
        L = ArenaLayout(nn);
        env32_host.assign((size_t)ENV_I32 * (size_t)nn, 0);
        env32_cache.clear();
        env64_cache.clear();
        // A new batch: every per-grid cache is stale (the buffers re-grow).
        light_uploaded = false;
    }
    unsigned long long* arena_dev() {
        return arena.get<unsigned long long>(L.end);
    }
    int* dlist_dev() {
        return dlist.get<int>((size_t)n_env * (size_t)h * (size_t)w);
    }
    size_t head_bytes() const {
        return L.end * 8 + (size_t)n_env * DESTROYED_HEAD * 4;
    }
    unsigned char* arena_host() { return arena_h.get<unsigned char>(head_bytes()); }
    void set_env32(int k, int32_t v) {
        for (int e = 0; e < n_env; ++e) env32_host[(size_t)k * n_env + e] = v;
    }
    int32_t* env32_col(int k) {
        return env32.get<int32_t>((size_t)ENV_I32 * n_env) + (size_t)k * n_env;
    }
    void push_env32() {
        upload_if_changed(env32_cache, env32, env32_host.data(), env32_host.size(),
                          "H2D env scalars");
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

void check_batch(const EngineList& engs, const ResidentFields& f, const char* who) {
    if (f.n_env < 1 || (int)engs.size() != f.n_env) {
        std::ostringstream os;
        os << who << ": the batch holds " << f.n_env << " envs but " << engs.size()
           << " engines were handed in (one per env)";
        throw std::invalid_argument(os.str());
    }
    for (const PhysicsEngine* e : engs) {
        if (e == nullptr) throw std::invalid_argument(std::string(who) + ": a null engine");
    }
}

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
    cuda_check(cudaStreamSynchronize(tstream()), "sync before fleck D2H");
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
// the water stage: the substeps, then W5 + W3 + the seal + the copy
// ---------------------------------------------------------------------------
void resident_water(const EngineList& engs, const ResidentFields& f,
                    const ResidentTickArgs& a) {
    check_batch(engs, f, "resident_water");
    const PhysicsEngine& eng = *engs[0];
    ResidentTickState& S = g_rt;
    S.key(f.n_env, f.h, f.w);
    const int n = f.h * f.w, N = f.n_env;
    if (n <= 0) return;
    cudaStream_t st = tstream();
    // the per-env live mask (nullptr when every env runs; skip when none does)
    S.water_active_env.assign((size_t)N, 1);
    const uint8_t* d_active = nullptr;
    if (a.water_active) {
        int live = 0;
        for (int e = 0; e < N; ++e) {
            S.water_active_env[(size_t)e] = a.water_active[e] ? 1 : 0;
            live += S.water_active_env[(size_t)e];
        }
        if (live == 0) return;
        if (live < N) {
            uint8_t* h = S.water_active_h.get<uint8_t>((size_t)N);
            std::memcpy(h, S.water_active_env.data(), (size_t)N);
            uint8_t* d = S.water_active.get<uint8_t>((size_t)N);
            cuda_check(cudaMemcpyAsync(d, h, (size_t)N, cudaMemcpyHostToDevice, st),
                       "H2D water mask");
            d_active = d;
        }
    }
    // the substep loop (the host path's step_water arguments)
    const WaterSolver& ws = eng.water;
    water_substeps_resident(
        P<int32_t>(f.water_depth), P<int32_t>(f.flow_vx), P<int32_t>(f.flow_vy),
        P<const int32_t>(f.floor_height), P<const int32_t>(f.atmosphere),
        P<const bool>(f.solid), f.h, f.w, a.water_n_sub, a.water_wdt,
        a.tilt_x, a.tilt_y, ws.g, ws.damping, ws.dx, ws.k_p, ws.v_max, ws.depth_eps,
        N, d_active);
    // gas_conservative, uploaded when it changes (bool -> one byte each).
    std::vector<uint8_t> cons_h((size_t)f.n_gases);
    for (int g = 0; g < f.n_gases; ++g) cons_h[(size_t)g] = a.gas_conservative[g] ? 1 : 0;
    upload_if_changed(S.cons_c, S.cons, cons_h.data(), cons_h.size(), "H2D cons");
    S.set_env32(ENV_WATER_T_AMB, a.t_amb_q);
    S.push_env32();
    const WaterTailScalars sc = water_tail_bake_scalars(
        a.sim_time, a.ceiling_h, a.flood_eps, a.ratio_cap, a.boil_rate,
        a.boil_p_thresh, a.steam_yield);
    unsigned long long* arena = S.arena_dev();
    const size_t nn = (size_t)N * n;
    water_tail_launch_resident(
        N, f.h, f.w, f.n_gases, a.steam_idx,
        P<int32_t>(f.water_depth), P<const int32_t>(f.atmosphere), P<const bool>(f.solid),
        P<int32_t>(f.gas), P<int32_t>(f.water_before), P<float>(f.dyn_permeability),
        P<int64_t>(f.gas_energy), S.cons.get<bool>((size_t)f.n_gases),
        P<const bool>(f.thermal_solid), P<const bool>(f.is_vacuum),
        a.ambient ? P<const bool>(f.is_ambient) : nullptr,
        S.env32_col(ENV_WATER_T_AMB), sc,
        S.wt_flag.get<uint8_t>(nn), S.wt_frac.get<int32_t>(nn),
        S.wt_list.get<int32_t>(nn), arena + S.L.wt, d_active);
    S.water_ran = true;
    S.water_steam_idx = a.steam_idx;
    S.water_n_gases = f.n_gases;
    ++g_counts[0];
}

// ---------------------------------------------------------------------------
// #29 P4b: the stamp (slot 6's five planes), every env's rows in one upload
// ---------------------------------------------------------------------------
void resident_stamp(const EngineList& engs, const ResidentFields& f,
                    const int32_t* rows, int n_rows) {
    check_batch(engs, f, "resident_stamp");
    ResidentTickState& S = g_rt;
    S.key(f.n_env, f.h, f.w);
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
        for (int r = 0; r < n_rows; ++r) {
            const int env = rows[(size_t)r * STAMP_ROW_COLS];
            if (env < 0 || env >= f.n_env)
                throw std::invalid_argument("resident_stamp: a row's env outside the batch");
        }
        const size_t count = (size_t)n_rows * STAMP_ROW_COLS;
        d_rows = S.stamp_rows.get<int32_t>(count);
        // Staged through PINNED memory so the copy is asynchronous. The
        // staging buffer is rewritten at the NEXT tick's entry, after this
        // tick's books D2H (a synchronizing read at the tail of the stream)
        // has completed the copy.
        int32_t* h = S.stamp_rows_h.get<int32_t>(count);
        std::memcpy(h, rows, count * sizeof(int32_t));
        cuda_check(cudaMemcpyAsync(d_rows, h, count * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, tstream()), "H2D stamp rows");
    }
    stamp_launch_resident(
        f.n_env, f.h, f.w, n_rows,
        P<const float>(f.permeability), P<const float>(f.wave_absorb),
        P<const int32_t>(f.heat_atten_q), P<const int32_t>(f.light_atten_q),
        P<bool>(f.obstacles), P<float>(f.dyn_permeability),
        P<float>(f.dyn_wave_absorb), P<int32_t>(f.dyn_heat_atten_q),
        P<int32_t>(f.dyn_light_atten_q), d_rows,
        S.stamp_winner.get<int32_t>((size_t)f.n_env * n));
    ++g_counts[11];
}

// ---------------------------------------------------------------------------
// the EOS stage (cuda_eos_resident.cu), its books into the arena
// ---------------------------------------------------------------------------
int resident_eos(const EngineList& engs, const ResidentFields& f,
                 const ResidentTickArgs& a) {
    check_batch(engs, f, "resident_eos");
    const PhysicsEngine& eng = *engs[0];
    ResidentTickState& S = g_rt;
    S.key(f.n_env, f.h, f.w);
    const int n = f.h * f.w, N = f.n_env;
    S.eos_ran = false;
    if (n <= 0 || a.sim_time <= 0.0f) return 0;
    unsigned long long* arena = S.arena_dev();
    EosBooksDev books;
    books.kick = arena + S.L.kick;
    books.flux = arena + S.L.flux;
    books.ecnt = arena + S.L.ecnt;
    books.rail = arena + S.L.rail;
    books.tcnt = arena + S.L.tcnt;
    books.pre  = arena + S.L.pre;
    books.nsub_max = arena + S.L.nsub;
    books.zero_begin = arena + S.L.kick;
    books.zero_bytes = (S.L.fire_cnt - S.L.kick) * 8;
    std::vector<const int32_t*> div_src((size_t)N, nullptr);
    for (int e = 0; e < N; ++e) {
        const std::vector<int32_t>& ds = engs[(size_t)e]->eos.div_source;
        if ((int)ds.size() == n) div_src[(size_t)e] = ds.data();
    }
    const int nsub = eos_step_resident(
        eng.eos, N, f.h, f.w, a.sim_time, a.gas_conservative, f.n_gases,
        a.n_amb, a.p_amb, a.ambient, div_src.data(),
        P<int32_t>(f.atmosphere), P<int32_t>(f.wave_p),
        P<int32_t>(f.wind_x), P<int32_t>(f.wind_y),
        P<int32_t>(f.temperature), P<int32_t>(f.gas),
        P<const bool>(f.solid), P<const bool>(f.is_vacuum),
        P<const float>(f.dyn_permeability), P<const float>(f.dyn_wave_absorb),
        a.ambient ? P<const bool>(f.is_ambient) : nullptr,
        a.ambient ? P<const int32_t>(f.sponge_sigma) : nullptr,
        a.ambient ? P<const int32_t>(f.sponge_udamp) : nullptr,
        P<const bool>(f.thermal_solid), P<int64_t>(f.gas_energy), books);
    S.eos_cons.clear();
    S.eos_trace_gi.clear();
    for (int g = 0; g < f.n_gases; ++g) {
        if (a.gas_conservative[g]) S.eos_cons.push_back(g);
        else S.eos_trace_gi.push_back(g);
    }
    S.eos_rail = a.ambient && !S.eos_cons.empty();
    S.eos_ran = true;
    ++g_counts[12];
    return nsub;
}

// ---------------------------------------------------------------------------
// the trace tail (cuda_bulk_transport.cu's launch core), books into the arena
// ---------------------------------------------------------------------------
void resident_trace_tail(const EngineList& engs, const ResidentFields& f,
                         const ResidentTickArgs& a) {
    check_batch(engs, f, "resident_trace_tail");
    ResidentTickState& S = g_rt;
    S.key(f.n_env, f.h, f.w);
    const int n = f.h * f.w, N = f.n_env, G = f.n_gases;
    S.tt_ran = false;
    if (n <= 0 || G <= 0) return;
    if (G > EOS_BOOK_GAS_MAX)
        throw std::invalid_argument("resident_trace_tail: n_gases above EOS_BOOK_GAS_MAX");
    // The per-env, per-gas folds — the CPU tail's own host folds, called
    // through the SAME out-of-line definitions (bulk_transport.cpp, on
    // /fp:strict): dd_q = trace_diffusion_dd_q(d, dt), frac_q =
    // trace_decay::frac_q(decay, dt). The stability door 4·dd_q <= ONE is
    // refused at the gases door and at PhysicsRunner's dt binding; a breach of
    // it here (a mis-wired caller) THROWS — the CPU asserts the same bound.
    std::vector<int32_t> scal((size_t)N * G * TRACE_TAIL_SCAL, 0);
    S.tt_trace_gi.clear();
    for (int gi = 0; gi < G; ++gi) {
        if (a.gas_conservative[gi]) continue;
        S.tt_trace_gi.push_back(gi);
        const int32_t dd_q = ::trace_diffusion_dd_q(a.gas_diffusion[gi], a.sim_time);
        if (4 * (int64_t)dd_q > (int64_t)FP_ONE) {
            std::ostringstream os;
            os << "trace_tail_resident: gas " << gi << " diffusion·dt fold "
               << dd_q << " breaks the Jacobi stability door 4·dd_q <= ONE";
            throw std::invalid_argument(os.str());
        }
        const int32_t fq = trace_decay::frac_q(a.gas_decay[gi], a.sim_time);
        for (int e = 0; e < N; ++e) {
            int32_t* r = scal.data() + ((size_t)e * G + gi) * TRACE_TAIL_SCAL;
            r[0] = 1; r[1] = dd_q; r[2] = fq;
        }
    }
    upload_if_changed(S.tt_scal_c, S.tt_scal, scal.data(), scal.size(), "H2D tail folds");
    unsigned long long* arena = S.arena_dev();
    trace_tail_launch_resident(N, f.h, f.w, G, P<int32_t>(f.gas),
                               P<const bool>(f.solid), P<const bool>(f.is_vacuum),
                               a.ambient ? P<const bool>(f.is_ambient) : nullptr,
                               P<const float>(f.dyn_permeability),
                               S.tt_scal.get<int32_t>(scal.size()),
                               S.tt_s0.get<int32_t>((size_t)N * G * n),
                               arena + S.L.ttail);
    S.tt_ran = true;
    ++g_counts[13];
}

// ---------------------------------------------------------------------------
// combustion
// ---------------------------------------------------------------------------
void resident_combustion(const EngineList& engs, const ResidentFields& f,
                         const ResidentTickArgs& a) {
    check_batch(engs, f, "resident_combustion");
    const PhysicsEngine& eng = *engs[0];
    // combustion reads the ring plane whole (the host call's
    // `is_ambient=gmap.is_ambient`, all false on a space map)
    ResidentTickState& S = g_rt;
    S.key(f.n_env, f.h, f.w);
    S.comb_ran = false;
    const int n = f.h * f.w, N = f.n_env;
    const int n_gases = f.n_gases;
    // --- the guards, VERBATIM combustion_step's head (a guarded return
    //     launches nothing: the CPU no-op) ---------------------------------
    if (f.h <= 0 || f.w <= 0 || a.sim_time <= 0.0f) return;
    if (a.o2_idx < 0 || a.o2_idx >= n_gases) return;
    if (a.inert_n2_idx < 0 || a.inert_n2_idx >= n_gases) return;
    if (a.black_smoke_idx < 0 || a.black_smoke_idx >= n_gases) return;
    const CombustionSolver& cs = eng.combustion;
    const q16 burn_cap_q = quantize((double)cs.burn_rate * (double)a.sim_time);
    const CombustionScalars sc = combustion_bake_scalars(
        cs.o2_thresh_burn, cs.o2_frac_ext, cs.o2_frac_full, cs.fire_T_ext,
        cs.fire_T_span, cs.hotf_cap, cs.H_BED_M, cs.H_BED_SHIFT, cs.fuel_per_o2,
        cs.soot_yield, cs.H_FUEL_M, cs.H_FUEL_SHIFT, eng.temperature.c_v,
        eng.temperature.n_floor_heat, cs.T_MAX_PHYS, cs.p_ext_q, cs.p_full_q);
    if (burn_cap_q <= 0) return;   // nothing burns this tick (fields untouched)
    const int draw_r = a.draw_r;
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
    upload_if_changed(S.ign_table_c, S.ign_table, a.ignition_table, (size_t)a.n_materials,
                      "H2D ignition table");
    S.set_env32(ENV_BURN_CAP, burn_cap_q);
    S.set_env32(ENV_T_AMB, a.t_amb_q);
    S.push_env32();
    cudaStream_t st = tstream();
    const size_t nn = (size_t)N * n;
    const int block = 256, grid = (int)((nn + block - 1) / block);
    int32_t* ign = S.ign_plane.get<int32_t>(nn);
    rt_ignition<<<grid, block, 0, st>>>(ign, P<const int8_t>(f.material),
                                        S.ign_table.get<int32_t>((size_t)a.n_materials),
                                        a.n_materials, (int)nn);
    cuda_check(cudaGetLastError(), "ignition gather");
    int32_t* perm_q = nullptr;
    if (draw_r > 1) {                       // the per-call: only with the extended draw
        perm_q = S.perm_q.get<int32_t>(nn);
        rt_perm_q<<<grid, block, 0, st>>>(perm_q, P<const float>(f.dyn_permeability), (int)nn);
        cuda_check(cudaGetLastError(), "perm quantize");
    }
    unsigned long long* arena = S.arena_dev();
    combustion_launch_resident(
        N, f.h, f.w, n_gases, a.o2_idx, a.inert_n2_idx, a.black_smoke_idx, draw_r,
        P<int32_t>(f.gas), P<int32_t>(f.temperature), P<int32_t>(f.wall_hp),
        P<int64_t>(f.gas_energy), P<const int32_t>(f.fire), ign,
        P<const bool>(f.flammable), P<const bool>(f.solid), P<const bool>(f.is_vacuum),
        P<const bool>(f.is_ambient), P<const bool>(f.thermal_solid),
        P<const int32_t>(f.heat_inv_shift),
        P<int32_t>(f.heat), P<int32_t>(f.dem_acc), f.acc_depth,
        perm_q,
        P<const int32_t>(f.fire_T_ext_plane), P<const int32_t>(f.fuel_per_o2_plane),
        P<const int32_t>(f.atmosphere),
        S.env32_col(ENV_BURN_CAP), S.env32_col(ENV_T_AMB), sc,
        S.tsnap.get<int32_t>(nn), S.alloc.get<int32_t>((size_t)n_slots * nn),
        S.dep.get<int32_t>((size_t)5 * nn), S.e_slot.get<int64_t>((size_t)n_slots * nn),
        S.e_dep.get<int64_t>((size_t)5 * nn), arena + S.L.comb);
    S.comb_ran = true;
    S.comb_smoke_idx = a.black_smoke_idx;
    ++g_counts[2];
}

// ---------------------------------------------------------------------------
// the light inputs (entry phase; N == 1 only)
// ---------------------------------------------------------------------------
void resident_upload_light(const PhysicsEngine& eng, const ResidentFields& f,
                           const ResidentTickArgs& a) {
    if (f.n_env != 1) {
        throw std::invalid_argument(
            "resident_upload_light: light is render state and is refused in a batch "
            "(N > 1) until #81 (Light as Game State)");
    }
    ResidentTickState& S = g_rt;
    S.key(f.n_env, f.h, f.w);
    const int n = f.h * f.w;
    // The sweep's own host door, on the SAME LightChannels run_sweep_ builds
    // (the plane pointers are device addresses here: the door only checks
    // that they are given; the tables, rows and sky it reads are host).
    LightChannels lc;
    lc.light_atten_q     = P<const int32_t>(f.light_atten_q);
    lc.dyn_light_atten_q = P<const int32_t>(f.dyn_light_atten_q);
    lc.l_table           = eng.light_emission.table();
    lc.light_absorb_q16  = a.gas_light_absorb_q16;
    lc.light_glow_q16    = a.gas_light_glow_q16;
    lc.transport         = eng.light_transport;
    lc.light_q           = P<int64_t>(f.light_q);
    lc.light_flux_q      = P<int64_t>(f.light_flux_q);
    lc.light_glow        = P<int64_t>(f.light_glow);
    lc.cones             = a.light_cones;
    lc.n_cones           = (a.light_cones != nullptr) ? a.n_light_cones : 0;
    lc.sky               = a.light_sky;
    lc.emit_mult         = a.light_emit_mult;
    lc.n_emit_mult       = (a.light_emit_mult != nullptr) ? a.n_light_emit_mult : 0;
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
    // The render-state inputs change every frame: synchronous uploads, light
    // only (never on a headless or batched tick).
    cuda_check(cudaStreamSynchronize(tstream()), "sync before light inputs");
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
std::vector<std::vector<std::pair<int, int>>> resident_tail(
        const EngineList& engs, const ResidentFields& f, const ResidentTickArgs& a) {
    check_batch(engs, f, "resident_tail");
    const PhysicsEngine& eng = *engs[0];
    ResidentTickState& S = g_rt;
    S.key(f.n_env, f.h, f.w);
    const int h = f.h, w = f.w, n = h * w, n_gases = f.n_gases, N = f.n_env;
    const size_t nn = (size_t)N * n;
    std::vector<std::vector<std::pair<int, int>>> destroyed((size_t)N);
    if (n <= 0) return destroyed;
    const bool with_light = eng.light_requested;
    if (with_light && N != 1) {
        throw std::invalid_argument(
            "resident_tail: light is refused in a batch (N > 1) until #81");
    }
    if (with_light && !S.light_uploaded) {
        throw std::logic_error(
            "resident_tail: light is requested but resident_upload_light did not "
            "run this tick (the entry phase hands the light inputs over)");
    }
    cudaStream_t st = tstream();

    // ---- per-env scalars and tables (uploaded only when changed) ---------
    std::vector<uint8_t> cons_h((size_t)n_gases);
    for (int g = 0; g < n_gases; ++g) cons_h[(size_t)g] = a.gas_conservative[g] ? 1 : 0;
    upload_if_changed(S.cons_c, S.cons, cons_h.data(), cons_h.size(), "H2D cons");
    S.set_env32(ENV_T_AMB, a.t_amb_q);
    S.set_env32(ENV_K_LEAK, a.k_leak_q);
    S.set_env32(ENV_FIRE_DT, fire_dt_q(a.sim_time));
    S.push_env32();
    {
        std::vector<int64_t> vac((size_t)N, a.rad_amb_vacuum_q);
        upload_if_changed(S.env64_cache, S.env64, vac.data(), vac.size(), "H2D vac level");
    }
    upload_if_changed(S.e_table_c, S.e_table, eng.emissive.table(),
                      (size_t)E_TABLE_SIZE, "H2D e_table");
    unsigned long long* arena = S.arena_dev();
    const int block = 256, grid = (int)((nn + block - 1) / block);

    // ---- the bulk sum (step_tail's n_bulk_, built ONCE before the fire) ---
    int32_t* n_bulk = S.n_bulk.get<int32_t>(nn);
    rt_nbulk<<<grid, block, 0, st>>>(n_bulk, P<const int32_t>(f.gas),
                                     S.cons.get<bool>((size_t)n_gases), n_gases, n, N);
    cuda_check(cudaGetLastError(), "n_bulk");

    // ---- 2. fire (its P1 early exit as the device gate) ------------------
    const FireParams& fp = eng.fire.params;
    const FireScalars fsc = fire_bake_scalars(
        fp.k_grow, fp.k_die, fp.fire_T_ext, fp.fire_T_span, fp.fuel_ref,
        fp.o2_frac_ext, fp.I_min, fp.k_wind_fan, fp.k_wind_strip, fp.wall_damage,
        fp.temp_scale, fp.I_cap_per_avail, fp.o2_frac_amb, fp.o2f_cap, fp.hotf_cap,
        fp.p_ext_q, fp.p_full_q);
    int* fire_cnt = reinterpret_cast<int*>(arena + S.L.fire_cnt);
    int32_t* fire_gate = reinterpret_cast<int32_t*>(arena + S.L.fire_gate);
    int* dlist = S.dlist_dev();
    fire_launch_resident(
        N, h, w, P<int32_t>(f.fire), P<const int32_t>(f.atmosphere),
        P<const int32_t>(f.gas), n_gases, a.o2_idx, n_bulk, P<int32_t>(f.wall_hp),
        P<const int32_t>(f.temperature), P<const int32_t>(f.wind_x),
        P<const int32_t>(f.wind_y), P<const bool>(f.solid), P<const bool>(f.is_vacuum),
        P<const bool>(f.flammable), P<const int64_t>(f.fuel_recip),
        P<const int32_t>(f.fire_T_ext_plane), S.env32_col(ENV_FIRE_DT), fsc,
        fire_gate, fire_cnt, dlist);
    ++g_counts[3];

    // ---- 2b. the radiation sweep (heat; light when requested) -------------
    // The ACTIVE gases decide the gas group, as the per-call wrapper decides
    // it: a zero coefficient adds exactly 0 to a density sum, so the full
    // table over the full gas block is the packed active set, integer for
    // integer -- and with no active gas at all the group is not given (the
    // pre-P5a sweep, the per-call's n_act == 0).
    const int32_t* gas_heat_absorb_q16 = a.gas_heat_absorb_q16;
    const int32_t* gas_light_absorb_q16 = a.gas_light_absorb_q16;
    const int32_t* gas_light_glow_q16 = a.gas_light_glow_q16;
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
    int64_t* sweep_cnt = reinterpret_cast<int64_t*>(arena + S.L.sweep);
    radiation_sweep_launch_resident(
        N, h, w,
        P<const int32_t>(f.temperature), P<const int32_t>(f.heat_atten_q),
        P<const int32_t>(f.dyn_heat_atten_q), P<const int32_t>(f.heat_inv_shift),
        P<const bool>(f.thermal_solid),
        /*d_amb_level=*/nullptr, P<const bool>(f.is_vacuum),
        S.e_table.get<int64_t>((size_t)E_TABLE_SIZE), eng.emissive.fine_bits,
        S.env64.get<int64_t>((size_t)N), S.env32_col(ENV_K_LEAK), S.env32_col(ENV_T_AMB),
        gas_on ? P<const int32_t>(f.gas) : nullptr, gas_on ? n_gases : 0,
        gas_on ? S.heat_absorb.get<int32_t>((size_t)n_gases) : nullptr,
        gas_on ? n_bulk : nullptr,
        cap.n_floor_q, cap.recip_cv,
        RadiationSweep::TRANSPORT_SHEAR, 16, /*fleck_enabled=*/true,
        S.outflow.get<int64_t>((size_t)16 * nn), S.amb_m.get<int64_t>(nn),
        S.ex_cell.get<int64_t>(nn), S.f_q24.get<int32_t>(nn),
        S.a_eff.get<int32_t>(nn), S.d_eff.get<int32_t>(nn),
        P<int64_t>(f.rad_net_sweep), P<int64_t>(f.rad_flux_sweep),
        P<int64_t>(f.rad_amb_sweep), P<int64_t>(f.rad_fluence),
        sweep_cnt, with_light ? &ldev : nullptr);
    ++g_counts[5];
    if (with_light) ++g_counts[6];
    S.light_uploaded = false;   // the next tick hands its own inputs over

    // ---- 3. temperature ---------------------------------------------------
    const TemperatureSolver& tsv = eng.temperature;
    const TemperatureScalars tsc = temperature_bake_scalars(
        tsv.c_v, tsv.n_floor_heat, tsv.T_MAX_PHYS, tsv.no_face, eng.emissive.fine_bits);
    const bool* d_ts = P<const bool>(f.thermal_solid);
    temperature_launch_resident(
        N, h, w, P<int32_t>(f.temperature), S.temp_new.get<int32_t>(nn),
        P<const int32_t>(f.heat), P<const int32_t>(f.heat_inv_shift),
        P<const int32_t>(f.face_shift),
        P<const bool>(f.solid), d_ts, P<const bool>(f.is_vacuum),
        a.ambient ? P<const bool>(f.is_ambient) : nullptr,
        n_bulk, P<const int64_t>(f.rad_net_sweep),
        P<const int64_t>(f.rad_fluence), S.e_table.get<int64_t>((size_t)E_TABLE_SIZE),
        P<int64_t>(f.gas_energy), S.de_gas.get<int64_t>(nn), S.env32_col(ENV_T_AMB),
        S.cap_used.get<int64_t>(nn), S.cap_real.get<int64_t>(nn),
        tsc, arena + S.L.temp, /*solid_books=*/true);
    ++g_counts[7];

    // ---- THE ONE BOOKS D2H (every stage, every env; it waits for the chain)
    unsigned char* hb = S.arena_host();
    cuda_check(cudaMemcpyAsync(hb, arena, S.L.end * 8, cudaMemcpyDeviceToHost, st),
               "D2H books");
    const int head = std::min(n, DESTROYED_HEAD);
    int* hheads = reinterpret_cast<int*>(hb + S.L.end * 8);
    cuda_check(cudaMemcpy2DAsync(hheads, (size_t)DESTROYED_HEAD * 4, dlist, (size_t)n * 4,
                                 (size_t)head * 4, (size_t)N, cudaMemcpyDeviceToHost, st),
               "D2H destroyed heads");
    tick_sync_point("books");   // the tick's one books read
    ++g_counts[8];
    const unsigned long long* hu = reinterpret_cast<const unsigned long long*>(hb);

    for (int e = 0; e < N; ++e) {
        const PhysicsEngine& E = *engs[(size_t)e];
        auto b = [&](size_t region, int slots, int slot) {
            return (int64_t)hu[region + (size_t)e * slots + slot];
        };

        // ---- the sweep's ingress verdict FIRST (the per-call order) ------
        const int64_t* sc = reinterpret_cast<const int64_t*>(
            hu + S.L.sweep + (size_t)e * RADIATION_SWEEP_CNT_SLOTS);
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
        E.radiation.min_stream = sc[RS_SLOT_MIN_STREAM];
        E.radiation.max_stream = sc[RS_SLOT_MAX_STREAM];
        if (with_light) {
            int64_t lbooks[12];
            for (int c = 0; c < 3; ++c) {
                lbooks[c]     = sc[RS_SLOT_LIGHT_EMIT + c];
                lbooks[3 + c] = sc[RS_SLOT_LIGHT_ABSORB + c];
                lbooks[6 + c] = sc[RS_SLOT_LIGHT_RING_IN + c];
                lbooks[9 + c] = sc[RS_SLOT_LIGHT_RING_OUT + c];
            }
            E.radiation.set_light_telemetry_from_twin(
                lbooks, sc[RS_SLOT_LIGHT_MIN_STREAM], sc[RS_SLOT_LIGHT_MAX_STREAM]);
        }

        // ---- the water tail's books (the CPU tail's own writes) -----------
        if (S.water_ran && S.water_active_env[(size_t)e]) {
            E.e_water_evac_export_sum = b(S.L.wt, WT_CNT_SLOTS, WT_EXPORT);   // ASSIGNED
            E.eos.ensure_trace_books(S.water_n_gases);
            E.eos.trace_mint_sum_[S.water_steam_idx] += b(S.L.wt, WT_CNT_SLOTS, WT_MINT);
            for (int g = 0; g < S.water_n_gases; ++g) {
                E.eos.trace_clip_sum_[g] += b(S.L.wt, WT_CNT_SLOTS, WT_CLIP + g);
                E.eos.trace_neg_hits_[g] += b(S.L.wt, WT_CNT_SLOTS, WT_NEG + g);
            }
            const int64_t nb_ = b(S.L.wt, WT_CNT_SLOTS, WT_N_BOIL);
            const int64_t ne_ = b(S.L.wt, WT_CNT_SLOTS, WT_N_EVAC);
            if (nb_ > 0 || ne_ > 0) ++g_counts[1];
            if (nb_ > 0) ++g_counts[9];
            if (ne_ > 0) ++g_counts[10];
        }

        // ---- the EOS stage's books (eos_step_resident's old fold, per env)
        if (S.eos_ran) {
            const EOSSolver& sv = E.eos;
            const unsigned long long* pre = hu + S.L.pre + (size_t)e * EOS_PRE_SLOTS;
            if (pre[EOS_PRE_FAIL] != 0) {
                throw std::runtime_error(
                    "gas-energy flux constant k_flux_q out of range — check dt / dx / "
                    "adiabatic_index / T_AMB_K (design §2.4 range guard)");
            }
            sv.dbg_last_n_sub = (int)pre[EOS_PRE_N_SUB];
            sv.dbg_last_c_local_q = (int32_t)(int64_t)pre[EOS_PRE_C_LOCAL];
            // The CPU twin's reset block (EOSSolver::step, the shared host
            // pre-stage's), then the stage-3b/3c trace books ASSIGNED into the
            // trace slots and the rail into the bulk slots.
            sv.ensure_trace_books(n_gases);
            std::fill(sv.boundary_flux_.begin(), sv.boundary_flux_.end(), (int64_t)0);
            std::fill(sv.trace_wipe_sum_.begin(), sv.trace_wipe_sum_.end(), (int64_t)0);
            std::fill(sv.trace_sink_sum_.begin(), sv.trace_sink_sum_.end(), (int64_t)0);
            std::fill(sv.trace_decay_sum_.begin(), sv.trace_decay_sum_.end(), (int64_t)0);
            for (size_t k = 0; k < S.eos_trace_gi.size(); ++k) {
                const int gi = S.eos_trace_gi[k];
                sv.boundary_flux_[gi]   = b(S.L.tcnt, EOS_TCNT_SLOTS, (int)k);
                sv.trace_wipe_sum_[gi]  = b(S.L.tcnt, EOS_TCNT_SLOTS, EOS_BOOK_GAS_MAX + (int)k);
                sv.trace_clip_sum_[gi] += b(S.L.tcnt, EOS_TCNT_SLOTS, 2 * EOS_BOOK_GAS_MAX + (int)k);
                sv.trace_neg_hits_[gi] += b(S.L.tcnt, EOS_TCNT_SLOTS, 3 * EOS_BOOK_GAS_MAX + (int)k);
            }
            if (S.eos_rail) {
                for (size_t k = 0; k < S.eos_cons.size(); ++k)
                    sv.boundary_flux_[S.eos_cons[k]] = b(S.L.rail, EOS_BOOK_GAS_MAX, (int)k);
            }
            auto kc = [&](int slot) { return b(S.L.kick, KICK_CNT_SLOTS, slot); };
            sv.u_clamp_hits      += kc(0);
            sv.u_max_hits        += kc(1);
            // slots 2-4 retired (work_clamp_hits / the T rails — moved to §2.6).
            // P-E3 (design §2.8): PER-TICK semantics (assigned).
            sv.ke_drag_removed       = kc(5);
            sv.e_drag_heat_sum       = kc(6);
            // arc #54 §2.3: the per-stage KE-bracket counters, PER-TICK.
            sv.e_kick_ke_sum         = kc(9);
            sv.e_absorb_export_sum   = kc(10);
            sv.e_sponge_export_sum   = kc(11);
            sv.e_clamp_destroyed_sum = kc(12);
            sv.rad_clip_hits         = kc(13);
            sv.e_ts_ke_sum           = kc(14);
            // arc #54 §2.4/§2.6: the flux step + recovery counters, PER-TICK.
            auto fc = [&](int slot) { return b(S.L.flux, FLUX_CNT_SLOTS, slot); };
            sv.p_face_floor_hits     = fc(0);
            sv.p_face_ceil_hits      = fc(1);
            sv.flux_sat_hits         = fc(2);
            sv.e_energy_floor_sum    = fc(3);
            sv.e_work_export_sum     = fc(4);
            sv.e_ts_work_sum         = fc(5);
            sv.e_wall_work_probe_sum = fc(6);
            sv.energy_floor_hits    += fc(7);
            sv.t_max_phys_hits      += fc(8);
            sv.e_rail_sum            = fc(9);
            // P-E1 energy counters — ASSIGNED (per-TICK semantics). e_wipe_sum:
            // bulk transport's share PLUS the §2.6 recovery's (fc(10)) — the
            // SAME member both sources book into on the CPU.
            auto ec = [&](int slot) { return b(S.L.ecnt, EOS_ECNT_SLOTS, slot); };
            sv.e_ts_residual       = ec(0);
            sv.e_wipe_sum          = ec(1);
            sv.e_wipe_sum         += fc(10);
            sv.e_floor_sum         = ec(2);
            sv.n_active_flux       = ec(3);
            sv.n_bulk_active_sum   = ec(4);
            sv.e_transport_net_sum = ec(5);
        }

        // ---- the trace tail's books, ADDED after the EOS's assigned slots
        //      (vent in boundary_flux_'s trace slots, then sink, decay) ---
        if (S.tt_ran) {
            const EOSSolver& sv = E.eos;
            sv.ensure_trace_books(n_gases);
            for (int gi : S.tt_trace_gi) {
                const int base = gi * TRACE_TAIL_CNT_SLOTS;
                const int st_ = n_gases * TRACE_TAIL_CNT_SLOTS;
                sv.boundary_flux_[gi]   += b(S.L.ttail, st_, base + TRACE_TAIL_VENT);
                sv.trace_sink_sum_[gi]  += b(S.L.ttail, st_, base + TRACE_TAIL_SINK);
                sv.trace_decay_sum_[gi] += b(S.L.ttail, st_, base + TRACE_TAIL_DECAY);
            }
        }

        // ---- combustion's books (the binding's fold, the CPU step's +=) --
        if (S.comb_ran) {
            const CombustionSolver& cs = E.combustion;
            auto cb = [&](int slot) { return b(S.L.comb, COMB_CNT_SLOTS, slot); };
            cs.heat_floor_hits    += cb(COMB_HEAT_FLOOR_HITS);
            cs.t_max_phys_hits    += cb(COMB_T_MAX_PHYS_HITS);
            cs.e_deposit_drop_sum += cb(COMB_DEPOSIT_DROP);
            cs.e_comb_draw_sum       += cb(COMB_E_DRAW);
            cs.e_comb_mint_sum       += cb(COMB_E_MINT);
            cs.e_comb_deliver_sum    += cb(COMB_E_DELIVER);
            cs.e_soot_shed_sum       += cb(COMB_E_SOOT_SHED);
            cs.e_ts_products_sum     += cb(COMB_E_TS_PRODUCTS);
            cs.e_comb_export_sum     += cb(COMB_E_EXPORT);
            cs.e_comb_heat_sum       += cb(COMB_E_HEAT);
            cs.e_comb_rail_sum       += cb(COMB_E_RAIL);
            cs.e_comb_solid_heat_sum += cb(COMB_E_SOLID_HEAT);
            // #88 P2: the soot mint's books, into the EOS's per-gas trace books.
            E.eos.ensure_trace_books(n_gases);
            E.eos.trace_mint_sum_[S.comb_smoke_idx] += cb(COMB_TRACE_MINT);
            E.eos.trace_clip_sum_[S.comb_smoke_idx] += cb(COMB_TRACE_CLIP);
            E.eos.trace_neg_hits_[S.comb_smoke_idx] += cb(COMB_TRACE_NEG);
        }

        // ---- temperature's books (step_tail's CUDA branch, slot for slot)
        {
            const TemperatureSolver& t = E.temperature;
            auto tb = [&](int slot) { return b(S.L.temp, TEMP_CNT_SLOTS, slot); };
            t.t_max_phys_hits    += tb(TEMP_CNT_HITS);
            t.t_low_rail_hits    += tb(TEMP_CNT_LOW_HITS);
            t.e_cond_trunc_sum   += tb(0);
            t.e_cond_cap_sum     += tb(1);
            t.cond_limit_hits    += tb(2);
            t.e_vac_wipe_sum     += tb(3);
            t.e_ring_pin_sum     += tb(4);
            t.e_deposit_drop_sum += tb(5);
            t.e_gas_deposit_sum  += tb(6);
            t.e_gas_cond_sum     += tb(7);
            t.e_gas_rail_sum     += tb(8);
            t.e_solid_deposit_sum += tb(9);
            t.e_solid_cond_sum    += tb(10);
            t.solid_energy_books_sum = tb(TEMP_CNT_SOLID_BOOKS);   // =, not +=
            t.rad_clamp_hits      += tb(11);
            t.e_rad_clamp_drop_sum += tb(12);
            t.e_rad_boundary_export_sum += tb(13);
            t.e_rad_floor_drop_sum      += tb(14);
            t.e_solid_low_rail_sum      += tb(15);
            t.e_dep_floor_drop_sum      += tb(16);
        }

        // ---- the destroyed walls (row-major, the per-call's sort) --------
        const int* fcnt = reinterpret_cast<const int*>(hu + S.L.fire_cnt);
        const int32_t* fgate = reinterpret_cast<const int32_t*>(hu + S.L.fire_gate);
        int counter = fcnt[e];
        if (fgate[e] >= fsc.max_fire_thresh_q) ++g_counts[4];
        if (counter > 0) {
            if (counter > n) counter = n;   // defensive (cannot exceed n slots)
            std::vector<int> idx((size_t)counter);
            const int* hh = hheads + (size_t)e * DESTROYED_HEAD;
            const int k0 = std::min(counter, DESTROYED_HEAD);
            std::copy(hh, hh + k0, idx.begin());
            if (counter > DESTROYED_HEAD) {
                cuda_check(cudaMemcpy(idx.data() + DESTROYED_HEAD,
                                      dlist + (size_t)e * n + DESTROYED_HEAD,
                                      (size_t)(counter - DESTROYED_HEAD) * sizeof(int),
                                      cudaMemcpyDeviceToHost), "D2H destroyed tail");
            }
            destroyed[(size_t)e] = fire_destroyed_row_major(std::move(idx), w);
        }
    }
    S.water_ran = false;
    S.comb_ran = false;
    S.eos_ran = false;
    S.tt_ran = false;
    return destroyed;
}

}  // namespace breach_cuda
