// ============================================================================
// swarm_larva.cpp -- the larva law's CPU entry (arc #63 P3,
// docs/swarm_P3_impl.md §3.6): THE REFERENCE the per-call and resident CUDA
// entries are gated against at tol 0 (there is no Python copy of the law, D1).
// host_loop<LarvaLaw> over (env, slot); the process-global branch tally
// (v4.7 G-d) and its names. A strict TU (/fp:strict); pure integer.
//
// Credit: Luo L, Gershow M, Rosenzweig M, Kang K, Fang-Yen C, Garrity PA,
// Samuel ADT, "Navigational decision making in Drosophila thermotaxis",
// J Neurosci 2010;30(12):4261-72, doi:10.1523/JNEUROSCI.4090-09.2010; the
// thresholds' sources (MacMillan & Sinclair 2011; Monzon et al. 2022) are
// cited in swarm_larva.h.
// ============================================================================
#include "swarm_larva.h"

namespace swarm {

const char* const SWARM_LARVA_BRANCH_NAMES[LARVA_BR_COUNT] = {
    "slot_invalid",
    "coma_held",
    "coma_to_run",
    "run_to_coma",
    "alarm_on",
    "alarm_off",
    "pref_plus",
    "pref_minus",
    "pref_none_equal",
    "pref_none_blocked",
    "sweep_solid",
    "sweep_off_grid",
    "tumble_taken",
    "tumble_not_taken",
    "turn_by_coin",
    "turn_toward",
    "turn_away",
    "probe_solid",
    "probe_edge",
    "moved",
    "died",
    "survived",
    "wrap_neg_remainder",
};

uint64_t g_larva_branch_tally[LARVA_BR_COUNT] = {0};

int swarm_larva_cpu(const LawIO& L) {
    return host_loop(LarvaLaw{}, L);
}

}  // namespace swarm
