// ============================================================================
// swarm.cpp -- the swarm step's host side (arc #63 P3, docs/swarm_P3_impl.md
// §3.6, §4.4): the species-law registry, PhysicsEngine::step_swarm's body
// (step_all: dormancy, the launch block, the CPU loop or the per-call CUDA
// wrapper, the host-dirty flag, the (species, env, slot) death compaction),
// the counted wrappers and the out-of-line entries the bindings reach the law
// through. A strict TU (/fp:strict, cpp/CMakeLists.txt); pure integer.
//
// Credit: the head sweep after Luo et al., J Neurosci 2010;30(12):4261-72,
// doi:10.1523/JNEUROSCI.4090-09.2010 (see swarm.h / swarm_larva.h).
// ============================================================================
#include "swarm.h"
#include "swarm_larva.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace swarm {

#ifdef BREACH_HAS_CUDA
#define SWARM_CUDA_ENTRY(f) (&(f))
#else
#define SWARM_CUDA_ENTRY(f) nullptr
#endif

// THE REGISTRY: one line per species, index == position in swarm.SWARM_SPECIES
// (identity-tested by tests/test_swarm_constants.py).
const SpeciesLaw SPECIES_LAWS[] = {
    {"larva", swarm_tables::SWARM_LARVA_WIDTH, "temperature", "center",
     SWARM_LARVA_BRANCH_NAMES, LARVA_BR_COUNT, g_larva_branch_tally,
     &swarm_larva_cpu, SWARM_CUDA_ENTRY(swarm_larva_cuda_percall),
     SWARM_CUDA_ENTRY(swarm_larva_cuda_resident), swarm_tables::SWARM_LARVA_BITE_Q},
};
const int N_SPECIES_LAWS = (int)(sizeof(SPECIES_LAWS) / sizeof(SPECIES_LAWS[0]));

int species_law_index(const char* name) {
    for (int i = 0; i < N_SPECIES_LAWS; ++i) {
        if (std::strcmp(SPECIES_LAWS[i].name, name) == 0) return i;
    }
    return -1;
}

namespace {
long long g_cpu_calls = 0;
long long g_cuda_calls = 0;
bool g_use_cuda = false;

int launch_hw_of(const LawIO& L) {
    int m = 0;
    for (int e = 0; e < L.n_env; ++e) {
        const int hw = (int)L.launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        if (hw > m) m = hw;
    }
    return m;
}
}  // namespace

void compose_launch(const uint32_t* env, const int32_t* hw, int n_env, uint32_t* out) {
    for (int e = 0; e < n_env; ++e) {
        uint32_t* o = out + e * LAUNCH_FIXED_WORDS;
        o[LAUNCH_TICK] = env[e * ENV_WORDS + ENV_TICK];
        o[LAUNCH_SEED_LO] = env[e * ENV_WORDS + ENV_SEED_LO];
        o[LAUNCH_SEED_HI] = env[e * ENV_WORDS + ENV_SEED_HI];
        o[LAUNCH_HW] = (uint32_t)hw[e];
    }
}

int run_cpu(int law, const LawIO& L) {
    if (launch_hw_of(L) <= 0) return 0;
    ++g_cpu_calls;
    LawIO Lc = L;
    Lc.io.tally = SPECIES_LAWS[law].tally;   // the process-global CPU tally
    return SPECIES_LAWS[law].cpu(Lc);
}

int run_cuda_percall(int law, const LawIO& L) {
    if (SPECIES_LAWS[law].cuda_percall == nullptr) {
        throw std::runtime_error(std::string("swarm: species '") + SPECIES_LAWS[law].name +
                                 "' has no per-call CUDA entry (the CPU build)");
    }
    if (launch_hw_of(L) <= 0) return 0;
    ++g_cuda_calls;
    LawIO Lc = L;
    Lc.io.tally = nullptr;                   // device code never touches the tally
    return SPECIES_LAWS[law].cuda_percall(Lc);
}

long long swarm_cpu_calls() { return g_cpu_calls; }
long long swarm_cuda_calls() { return g_cuda_calls; }
void set_swarm_backend(bool use_cuda) { g_use_cuda = use_cuda; }
bool get_swarm_backend() { return g_use_cuda; }

void output_planes_host(int law, const SlotIO& io, int n_env, const uint32_t* launch_host,
                        int32_t* bite, int32_t* count) {
    const int row = SPECIES_LAWS[law].bite_row;
    const size_t plane = (size_t)io.h * (size_t)io.w;
    for (int e = 0; e < n_env; ++e) {
        const int hw = (int)launch_host[e * LAUNCH_FIXED_WORDS + LAUNCH_HW];
        const int32_t q = (row >= 0 && bite != nullptr)
                              ? (int32_t)io.tables[(size_t)e * io.table_width + row]
                              : 0;
        int32_t* b = bite != nullptr ? bite + (size_t)e * plane : nullptr;
        int32_t* c = count != nullptr ? count + (size_t)e * plane : nullptr;
        for (int s = 0; s < hw; ++s) {
            int idx;
            if (c != nullptr && active_on(io, e, s, &idx)) c[idx] += 1;
            if (q != 0 && bites_on(io, e, s, &idx)) b[idx] += q;
        }
    }
}

void step_all(const SpeciesStep* packs, int n_packs, const int32_t* T, const bool* solid,
              int h, int w, const uint32_t* env, int n_env, StepScratch& scratch,
              std::vector<int32_t>& rows, int32_t* bite, int32_t* count) {
    rows.clear();
    // D14: dormancy is "nothing present in any env", decided here, before any
    // launch or scratch growth.
    bool any = false;
    for (int p = 0; p < n_packs && !any; ++p) {
        for (int e = 0; e < n_env; ++e) {
            if (packs[p].hw[e] > 0) { any = true; break; }
        }
    }
    if (!any) return;
    // arc #63 P5a / P5b: the output planes are rebuilt from zero every
    // non-dormant tick.
    if (bite != nullptr) std::memset(bite, 0, (size_t)n_env * h * w * sizeof(int32_t));
    if (count != nullptr) std::memset(count, 0, (size_t)n_env * h * w * sizeof(int32_t));

    for (int p = 0; p < n_packs; ++p) {
        const SpeciesStep& sp = packs[p];
        bool present = false;
        for (int e = 0; e < n_env; ++e) present = present || sp.hw[e] > 0;
        if (!present) continue;
        scratch.launch.assign((size_t)n_env * LAUNCH_FIXED_WORDS, 0u);
        compose_launch(env, sp.hw, n_env, scratch.launch.data());
        const size_t nrec = (size_t)n_env * (size_t)sp.max_units * DEATH_WORDS;
        if (scratch.records.size() < nrec) scratch.records.resize(nrec);
        LawIO L;
        L.io.st = sp.st;
        L.io.max_units = sp.max_units;
        L.io.h = h;
        L.io.w = w;
        L.io.T = T;
        L.io.solid = solid;
        L.io.launch = scratch.launch.data();
        L.io.tables = sp.tables;
        L.io.table_width = sp.table_width;
        L.io.death = scratch.records.data();
        L.n_env = n_env;
        L.launch_host = scratch.launch.data();
        if (g_use_cuda && SPECIES_LAWS[sp.law].cuda_percall != nullptr) {
            run_cuda_percall(sp.law, L);
        } else {
            run_cpu(sp.law, L);
        }
        // arc #63 P5a / P5b: this species' biting and active slots onto the
        // planes (the host store holds the post-step state on both host
        // branches).
        output_planes_host(sp.law, L.io, n_env, L.launch_host, bite, count);
        // R9 / V7: the host store moved, so any device copy is stale.
        if (sp.host_dirty) *sp.host_dirty = 1;
        // The (species, env, slot) compaction from the per-slot records.
        for (int e = 0; e < n_env; ++e) {
            for (int s = 0; s < sp.hw[e]; ++s) {
                const int32_t* r =
                    scratch.records.data() + ((size_t)e * sp.max_units + s) * DEATH_WORDS;
                if (r[DEATH_UNIT_ID] == 0) continue;
                rows.push_back(sp.law);
                rows.push_back(e);
                rows.push_back(s);
                rows.push_back(r[DEATH_UNIT_ID]);
                rows.push_back(r[DEATH_POS_X]);
                rows.push_back(r[DEATH_POS_Y]);
                rows.push_back(r[DEATH_HEADING]);
            }
        }
    }
}

int32_t wrap_heading_q16_host(int64_t h) { return wrap_heading_q16(h); }

int32_t felt_tile_host(const int32_t* T, int h, int w, int32_t x_q, int32_t y_q) {
    int idx;
    if (!tile_of(x_q, y_q, h, w, &idx)) {
        throw std::invalid_argument("swarm felt_T: the point is outside the grid");
    }
    const TileReader R{T, w};
    return R(x_q, y_q);
}

}  // namespace swarm
