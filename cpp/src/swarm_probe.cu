// ============================================================================
// swarm_probe.cu -- TEST-ONLY (arc #63 P3, docs/swarm_P3_impl.md v4.7 G-b):
// the resident launch core driven directly on GATE-OWNED arrays, the
// cuda_unit_*_probe.cu pattern. The sim-path stores are zero beyond hw by
// the store's law, so the padding a gate owns (live-looking rows past a short
// env's hw, past hw up to capacity, garbage record slots) can reach the
// resident core only through here. Copies every caller array into fresh
// device memory (padding included), runs the species' resident entry -- the
// SAME launch core the tick's resident_swarm runs -- on the tick stream, and
// copies every array back WHOLE, so a gate sees any byte the core touched
// outside [0, hw[e]). Allocates and syncs: never on the hot path.
// ============================================================================
#include "swarm.h"
#include "cuda_resident_tick.h"   // tick_stream_handle

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace swarm {

namespace {
void pcheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in swarm_probe/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

struct Owned {
    std::vector<void*> ptrs;
    ~Owned() { for (void* p : ptrs) cudaFree(p); }
    template <class T> T* dup(const T* src, size_t count) {
        void* p = nullptr;
        pcheck(cudaMalloc(&p, count > 0 ? count * sizeof(T) : 8), "probe malloc");
        ptrs.push_back(p);
        if (count > 0)
            pcheck(cudaMemcpy(p, src, count * sizeof(T), cudaMemcpyHostToDevice), "probe H2D");
        return static_cast<T*>(p);
    }
};

template <class T>
void back(T* dst, const T* src, size_t count) {
    if (count > 0)
        pcheck(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost), "probe D2H");
}
}  // namespace

int resident_probe(int law, const LawIO& L) {
    if (SPECIES_LAWS[law].cuda_resident == nullptr)
        throw std::runtime_error("swarm_probe: no resident CUDA entry");
    const SlotIO& h = L.io;
    const size_t nu = (size_t)L.n_env * (size_t)h.max_units;
    const size_t np = (size_t)L.n_env * (size_t)h.h * (size_t)h.w;
    int launch_hw = 0;
    for (int e = 0; e < L.n_env; ++e) {
        const int hw = (int)L.launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        if (hw > launch_hw) launch_hw = hw;
    }
    Owned o;
    ResidentLaunch R;
    R.io = h;
    R.io.st.pos_x = o.dup(h.st.pos_x, nu);
    R.io.st.pos_y = o.dup(h.st.pos_y, nu);
    R.io.st.heading = o.dup(h.st.heading, nu);
    R.io.st.hp = o.dup(h.st.hp, nu);
    R.io.st.ai_state = o.dup(h.st.ai_state, nu);
    R.io.st.dist_walked = o.dup(h.st.dist_walked, nu);
    R.io.st.unit_id = o.dup(h.st.unit_id, nu);
    R.io.st.faction = o.dup(h.st.faction, nu);
    R.io.st.valid = o.dup(h.st.valid, nu);
    R.io.T = o.dup(h.T, np);
    R.io.solid = o.dup(h.solid, np);
    R.io.launch = o.dup(h.launch, (size_t)L.n_env * LAUNCH_FIXED_WORDS);
    R.io.tables = o.dup(h.tables, (size_t)L.n_env * (size_t)h.table_width);
    R.io.death = o.dup(h.death, nu * DEATH_WORDS);
    R.io.tally = nullptr;
    std::vector<unsigned long long> zero((size_t)(L.n_env > 0 ? L.n_env : 1) * SWARM_CNT_SLOTS, 0ull);
    unsigned long long* cnt = o.dup(zero.data(), zero.size());
    R.n_env = L.n_env;
    R.launch_hw = launch_hw;
    R.counters = cnt;
    R.stream = breach_cuda::tick_stream_handle();
    SPECIES_LAWS[law].cuda_resident(R);   // the zero-extent guard is the core's
    pcheck(cudaGetLastError(), "probe launch");
    pcheck(cudaStreamSynchronize(static_cast<cudaStream_t>(R.stream)), "probe sync");
    back(h.st.pos_x, R.io.st.pos_x, nu);
    back(h.st.pos_y, R.io.st.pos_y, nu);
    back(h.st.heading, R.io.st.heading, nu);
    back(h.st.hp, R.io.st.hp, nu);
    back(h.st.ai_state, R.io.st.ai_state, nu);
    back(h.st.dist_walked, R.io.st.dist_walked, nu);
    back(h.st.unit_id, R.io.st.unit_id, nu);
    back(h.st.faction, R.io.st.faction, nu);
    back(h.st.valid, R.io.st.valid, nu);
    back(h.death, R.io.death, nu * DEATH_WORDS);
    back(zero.data(), cnt, zero.size());
    int deaths = 0;
    for (int e = 0; e < L.n_env; ++e) {
        const int hw = (int)L.launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        int de = 0;
        for (int s = 0; s < hw; ++s) {
            if (h.death[((size_t)e * h.max_units + s) * DEATH_WORDS + DEATH_UNIT_ID] != 0) ++de;
        }
        if ((unsigned long long)de != zero[(size_t)e * SWARM_CNT_SLOTS + SWARM_CNT_DEATHS]) {
            std::ostringstream os;
            os << "swarm_probe: env " << e << ": the records hold " << de
               << " deaths but the core's counter says "
               << zero[(size_t)e * SWARM_CNT_SLOTS + SWARM_CNT_DEATHS];
            throw std::runtime_error(os.str());
        }
        deaths += de;
    }
    return deaths;
}

}  // namespace swarm
