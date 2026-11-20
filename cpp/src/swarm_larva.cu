// ============================================================================
// swarm_larva.cu -- the larva law's CUDA entries (arc #63 P3,
// docs/swarm_P3_impl.md §3.6): the per-call wrapper and the resident launch
// core, both instantiations of swarm_launch.cuh's templates around the ONE
// FP_HD law in swarm_larva.h (never a copy). Pure integer; on the target-wide
// CUDA determinism floor (--fmad=false, the strict host floor).
//
// Credit: Luo L, Gershow M, Rosenzweig M, Kang K, Fang-Yen C, Garrity PA,
// Samuel ADT, "Navigational decision making in Drosophila thermotaxis",
// J Neurosci 2010;30(12):4261-72, doi:10.1523/JNEUROSCI.4090-09.2010; the
// thresholds' sources are cited in swarm_larva.h.
// ============================================================================
#include "swarm_larva.h"
#include "swarm_launch.cuh"

namespace swarm {

int swarm_larva_cuda_percall(const LawIO& L) {
    return percall(LarvaLaw{}, L);
}

void swarm_larva_cuda_resident(const ResidentLaunch& R) {
    resident(LarvaLaw{}, R);
}

}  // namespace swarm
