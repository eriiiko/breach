#pragma once
// ============================================================================
// #29 P6.1 -- THE DEVICE UNIT STORE (docs/residency_batching_design_2026-11-07.md
// §4.4b, decisions 1 to 3).
// ============================================================================
//
// One (N, U, W) int64 device buffer: N envs, U unit rows each, a row the unit
// schema's group-S columns at their encoded widths then `valid` (the layout is
// cpp/src/unit_store_layout.h, GENERATED from src/simulation/unit_schema.py by
// tools/gen_unit_layout.py). Allocated ONCE when built, freed when destroyed;
// uploaded and downloaded on THE tick stream (cuda_batch.cuh's tstream)
// through pinned staging and async copies; the unit kernels run on it in
// place, each born (N, U) -- bgrid(U, N), blockIdx.y the env.
//
// CLASS T (training only, design §6): the played --cuda / --resident tick
// never touches it -- it is not in run_resident_batch's entry or exit sets.
// The gates drive it now (tests/cuda_unit_store_check.py: the round trip and
// the shadow lockstep); the device backend from P8b.
//
// Plain C++ declarations (no CUDA types) so bindings.cpp (cl.exe) can include
// it. Compiled only when BREACH_CUDA.
#include <cstddef>
#include <cstdint>

namespace breach_cuda {

class UnitStoreDevice {
public:
    // Refuses n_env < 1, n_units < 1, or a width that is not the compiled
    // layout's UNIT_STORE_W (a header regenerated without a rebuild).
    UnitStoreDevice(int n_env, int n_units, int width);
    ~UnitStoreDevice();
    UnitStoreDevice(const UnitStoreDevice&) = delete;
    UnitStoreDevice& operator=(const UnitStoreDevice&) = delete;

    int n_env() const { return n_env_; }
    int n_units() const { return n_units_; }
    int width() const { return width_; }
    size_t count() const { return (size_t)n_env_ * n_units_ * width_; }
    std::uintptr_t device_ptr() const { return reinterpret_cast<std::uintptr_t>(d_); }

    // The whole store from / to host memory (count() int64, row-major
    // (N, U, W)): staged through pinned memory, one async copy on the tick
    // stream. upload returns once the host buffer is staged (the copy may
    // still run); download returns once the copy has landed.
    void upload(const int64_t* host);
    void download(int64_t* host);

    // Slot 1c, exposure decay (src/simulation/exposure.py decay_units /
    // decay_step, integer for integer): every VALID row's exposure block,
    // each non-zero entry x -> max(0, x - max(1, (x * decay_q) >> 16)) (0
    // when x <= 0). decay_q in [0, 65536] (refused otherwise). Writes, per
    // env, the rows whose block changed into coverage_out (n_env entries).
    void exposure_decay(int64_t decay_q, int64_t* coverage_out);

private:
    void stage_wait();          // the pinned staging buffer is free again
    int n_env_ = 0, n_units_ = 0, width_ = 0;
    int64_t* d_ = nullptr;              // (N, U, W) device
    int64_t* h_ = nullptr;              // pinned staging, count()
    unsigned long long* d_cov_ = nullptr;   // (N,) device coverage
    unsigned long long* h_cov_ = nullptr;   // (N,) pinned
    void* staged_ = nullptr;            // cudaEvent_t: the last copy through h_
};

// The compiled layout (unit_store_layout.h): W, the fingerprint, the valid
// column and the exposure block -- what the Python driver checks against the
// schema before it builds a store.
int unit_store_width();
const char* unit_store_fingerprint();
int unit_store_valid_col();
int unit_store_exposure_col();
int unit_store_exposure_width();

}  // namespace breach_cuda
