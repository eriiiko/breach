// ============================================================================
// swarm.cu -- the swarm step's device side (arc #63 P3, docs/swarm_P3_impl.md
// §4.4, §4.5): the per-call wrapper's staging scratch and transfers, and the
// resident stage's per-species staging (the launch block and the table rows
// up on the tick stream, the core launched, each env's [0, hw[e]) records
// down into a pinned host buffer before the tail's ONE books sync).
//
// Persistent buffers grow on demand and are never freed per tick: a
// steady-state resident tick allocates nothing and syncs nothing here.
// ============================================================================
#include "swarm.h"
#include "swarm_launch.cuh"
#include "unit_draws_layout.h"   // #29 P7e.5: the tick row's columns

#include <cuda_runtime.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace swarm {

namespace {

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in swarm/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

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

// The per-call scratch (one set; the per-call path is one species at a time).
struct PercallScratch {
    DevBuf col[N_COLUMNS];
    DevBuf T, solid, launch, tables, death;
};
PercallScratch g_pc;

// The resident stage's per-species staging.
struct ResidentScratch {
    DevBuf launch, tables, death;
    PinBuf launch_h, tables_h, death_h;
};
std::vector<ResidentScratch> g_res;

ResidentScratch& res_scratch(int law) {
    if ((int)g_res.size() < N_SPECIES_LAWS) g_res.resize((size_t)N_SPECIES_LAWS);
    return g_res[(size_t)law];
}

template <class T>
T* up(DevBuf& b, const T* src, size_t count, const char* what) {
    T* d = b.get<T>(count);
    if (count > 0) cuda_check(cudaMemcpy(d, src, count * sizeof(T), cudaMemcpyHostToDevice), what);
    return d;
}

template <class T>
void down(T* dst, const T* src, size_t count, const char* what) {
    if (count > 0) cuda_check(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost), what);
}

// arc #63 P5a / P5b: THE OUTPUT PLANES on the device -- the host builder's
// loop (output_planes_host) over (ceil(hw / 256), N): each active slot adds 1
// to its tile of the count plane and each biting slot its species' bite row
// to the bite plane, integer atomic adds, so neither plane depends on the
// thread order. Outside the law (the law has no atomics).
__global__ void k_planes(SlotIO io, int bite_row, int32_t* bite, int32_t* count) {
    const int e = (int)blockIdx.y;
    const int s = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= (int)io.launch[e * LAUNCH_FIXED_WORDS + LAUNCH_HW]) return;
    const size_t base = (size_t)e * io.h * io.w;
    int idx;
    if (count != nullptr && active_on(io, e, s, &idx)) atomicAdd(&count[base + idx], 1);
    if (bite == nullptr || bite_row < 0) return;
    const int32_t q = (int32_t)io.tables[(size_t)e * io.table_width + bite_row];
    if (q == 0) return;
    if (!bites_on(io, e, s, &idx)) return;
    atomicAdd(&bite[base + idx], q);
}

}  // namespace

void percall_check(const char* what) {
    cuda_check(cudaGetLastError(), what);
}

SlotIO percall_upload(const LawIO& L) {
    const SlotIO& h = L.io;
    const size_t nu = (size_t)L.n_env * (size_t)h.max_units;
    const size_t np = (size_t)L.n_env * (size_t)h.h * (size_t)h.w;
    SlotIO d = h;
    d.st.pos_x = up(g_pc.col[0], h.st.pos_x, nu, "H2D pos_x");
    d.st.pos_y = up(g_pc.col[1], h.st.pos_y, nu, "H2D pos_y");
    d.st.heading = up(g_pc.col[2], h.st.heading, nu, "H2D heading");
    d.st.hp = up(g_pc.col[3], h.st.hp, nu, "H2D hp");
    d.st.ai_state = up(g_pc.col[4], h.st.ai_state, nu, "H2D ai_state");
    d.st.dist_walked = up(g_pc.col[5], h.st.dist_walked, nu, "H2D dist_walked");
    d.st.unit_id = up(g_pc.col[6], h.st.unit_id, nu, "H2D unit_id");
    d.st.faction = up(g_pc.col[7], h.st.faction, nu, "H2D faction");
    d.st.valid = up(g_pc.col[8], h.st.valid, nu, "H2D valid");
    d.T = up(g_pc.T, h.T, np, "H2D T");
    d.solid = up(g_pc.solid, h.solid, np, "H2D solid");
    d.launch = up(g_pc.launch, h.launch, (size_t)L.n_env * LAUNCH_FIXED_WORDS, "H2D launch");
    d.tables = up(g_pc.tables, h.tables, (size_t)L.n_env * (size_t)h.table_width, "H2D tables");
    d.death = g_pc.death.get<int32_t>(nu * DEATH_WORDS);
    d.tally = nullptr;
    return d;
}

int percall_download(const LawIO& L, const SlotIO& d) {
    const SlotIO& h = L.io;
    const size_t nu = (size_t)L.n_env * (size_t)h.max_units;
    down(h.st.pos_x, d.st.pos_x, nu, "D2H pos_x");
    down(h.st.pos_y, d.st.pos_y, nu, "D2H pos_y");
    down(h.st.heading, d.st.heading, nu, "D2H heading");
    down(h.st.hp, d.st.hp, nu, "D2H hp");
    down(h.st.ai_state, d.st.ai_state, nu, "D2H ai_state");
    down(h.st.dist_walked, d.st.dist_walked, nu, "D2H dist_walked");
    down(h.st.unit_id, d.st.unit_id, nu, "D2H unit_id");
    down(h.st.faction, d.st.faction, nu, "D2H faction");
    down(h.st.valid, d.st.valid, nu, "D2H valid");
    int deaths = 0;
    for (int e = 0; e < L.n_env; ++e) {
        const int hw = (int)L.launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        if (hw <= 0) continue;
        const size_t off = (size_t)e * h.max_units * DEATH_WORDS;
        down(h.death + off, d.death + off, (size_t)hw * DEATH_WORDS, "D2H records");
        for (int s = 0; s < hw; ++s) {
            if (h.death[off + (size_t)s * DEATH_WORDS + DEATH_UNIT_ID] != 0) ++deaths;
        }
    }
    return deaths;
}

// #29 P7e.5: env e's launch words from the device tick row (the draws'
// tick word and key, as _env_words writes them from sim.draws on the host).
__global__ void k_launch_from_row(uint32_t* launch, const int64_t* row, int n_env) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_env) return;
    const int64_t* r = row + (size_t)e * unit_draws::TICK_ROW_W;
    uint32_t* o = launch + (size_t)e * LAUNCH_FIXED_WORDS;
    o[LAUNCH_TICK] = (uint32_t)((uint64_t)r[unit_draws::TICK_ROW_DRAW_TICK] & 0xffffffffull);
    o[LAUNCH_SEED_LO] = (uint32_t)((uint64_t)r[unit_draws::TICK_ROW_SEED_LO] & 0xffffffffull);
    o[LAUNCH_SEED_HI] = (uint32_t)((uint64_t)r[unit_draws::TICK_ROW_SEED_HI] & 0xffffffffull);
}

int resident_species(int law, const StoreView& st_dev, int max_units, int n_env, int h,
                     int w, const int32_t* d_T, const bool* d_solid,
                     const uint32_t* launch_host, const int64_t* tables_host,
                     int table_width, unsigned long long* d_counters, void* stream,
                     int32_t* d_bite, int32_t* d_count, const int64_t* d_tick_row) {
    int launch_hw = 0;
    for (int e = 0; e < n_env; ++e) {
        const int hw = (int)launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        if (hw > launch_hw) launch_hw = hw;
    }
    if (n_env <= 0 || launch_hw <= 0) return 0;
    if (SPECIES_LAWS[law].cuda_resident == nullptr) {
        throw std::runtime_error("swarm: a species without a resident CUDA entry");
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    ResidentScratch& S = res_scratch(law);
    const size_t nl = (size_t)n_env * LAUNCH_FIXED_WORDS;
    const size_t nt = (size_t)n_env * (size_t)table_width;
    const size_t nr = (size_t)n_env * (size_t)max_units * DEATH_WORDS;
    uint32_t* lh = S.launch_h.get<uint32_t>(nl);
    int64_t* th = S.tables_h.get<int64_t>(nt);
    std::memcpy(lh, launch_host, nl * sizeof(uint32_t));
    std::memcpy(th, tables_host, nt * sizeof(int64_t));
    uint32_t* ld = S.launch.get<uint32_t>(nl);
    int64_t* td = S.tables.get<int64_t>(nt);
    int32_t* rd = S.death.get<int32_t>(nr);
    int32_t* rh = S.death_h.get<int32_t>(nr);
    cuda_check(cudaMemcpyAsync(ld, lh, nl * sizeof(uint32_t), cudaMemcpyHostToDevice, st),
               "H2D resident launch");
    cuda_check(cudaMemcpyAsync(td, th, nt * sizeof(int64_t), cudaMemcpyHostToDevice, st),
               "H2D resident tables");
    if (d_tick_row != nullptr) {
        k_launch_from_row<<<(n_env + 127) / 128, 128, 0, st>>>(ld, d_tick_row, n_env);
        cuda_check(cudaGetLastError(), "launch words from the tick row");
    }
    ResidentLaunch R;
    R.io.st = st_dev;
    R.io.max_units = max_units;
    R.io.h = h;
    R.io.w = w;
    R.io.T = d_T;
    R.io.solid = d_solid;
    R.io.launch = ld;
    R.io.tables = td;
    R.io.table_width = table_width;
    R.io.death = rd;
    R.io.tally = nullptr;
    R.n_env = n_env;
    R.launch_hw = launch_hw;
    R.counters = d_counters;
    R.stream = stream;
    SPECIES_LAWS[law].cuda_resident(R);
    cuda_check(cudaGetLastError(), "resident swarm launch");
    // arc #63 P5a / P5b: the species' bites and counts onto the planes,
    // after its law.
    if (d_count != nullptr || (d_bite != nullptr && SPECIES_LAWS[law].bite_row >= 0)) {
        const dim3 grid((unsigned)((launch_hw + 255) / 256), (unsigned)n_env);
        k_planes<<<grid, 256, 0, st>>>(R.io, SPECIES_LAWS[law].bite_row, d_bite, d_count);
        cuda_check(cudaGetLastError(), "resident swarm output planes");
    }
    // Each env's [0, hw[e]) records, host-ward on the tick stream BEFORE the
    // tail's books sync, so they arrive with the books in the tick's one read.
    for (int e = 0; e < n_env; ++e) {
        const int hw = (int)launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        if (hw <= 0) continue;
        const size_t off = (size_t)e * max_units * DEATH_WORDS;
        cuda_check(cudaMemcpyAsync(rh + off, rd + off, (size_t)hw * DEATH_WORDS * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, st),
                   "D2H resident records");
    }
    return launch_hw;
}

const int32_t* resident_records_host(int law) {
    return static_cast<const int32_t*>(res_scratch(law).death_h.p);
}

}  // namespace swarm
