// ============================================================================
// #29 P7a.3 -- THE DEVICE EDIT DOOR's buffers (see cuda_edit_door.h). The
// device functions a producer writes through are cuda_edit_door.cuh's.
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_edit_door.h"
#include "cuda_batch.cuh"          // tstream

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {
inline void ed_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in edit_door/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}
}  // namespace

EditDoor::EditDoor(int n_env, int edit_cap, int tile_cap)
    : edits_(n_env, 1, edit_cap, EDIT_DOOR_COLS, 1), tiles_(n_env, 1, tile_cap, EDQ_TILE_COLS, 1) {
    ed_check(cudaMalloc(reinterpret_cast<void**>(&d_seq_), (size_t)n_env * sizeof(int64_t)),
             "malloc seq");
    begin();
}

EditDoor::~EditDoor() {
    cudaStreamSynchronize(tstream());
    if (d_seq_) cudaFree(d_seq_);
}

void EditDoor::begin() {
    edits_.begin();
    tiles_.begin();
    ed_check(cudaMemsetAsync(d_seq_, 0, (size_t)n_env() * sizeof(int64_t), tstream()), "zero seq");
}

void EditDoor::set_seq(const int64_t* seq) {
    ed_check(cudaMemcpyAsync(d_seq_, seq, (size_t)n_env() * sizeof(int64_t),
                             cudaMemcpyHostToDevice, tstream()), "H2D seq");
    ed_check(cudaStreamSynchronize(tstream()), "seq sync");
}

void EditDoor::download_seq(int64_t* seq) {
    ed_check(cudaMemcpyAsync(seq, d_seq_, (size_t)n_env() * sizeof(int64_t),
                             cudaMemcpyDeviceToHost, tstream()), "D2H seq");
    ed_check(cudaStreamSynchronize(tstream()), "seq sync");
}

}  // namespace breach_cuda
