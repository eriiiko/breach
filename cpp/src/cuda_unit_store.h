#pragma once
// ============================================================================
// #29 P6.1 / P6.2 -- THE DEVICE UNIT STORE and THE DEVICE TABLES
// (docs/residency_batching_design_2026-11-07.md §4.4b, §4.11).
// ============================================================================
//
// UnitStoreDevice: one (N, U, W) int64 device buffer: N envs, U unit rows
// each, a row the unit schema's group-S columns at their encoded widths then
// `valid` (the layout is cpp/src/unit_store_layout.h, GENERATED from
// src/simulation/unit_schema.py by tools/gen_unit_layout.py). Allocated ONCE
// when built, freed when destroyed; uploaded and downloaded on THE tick stream
// (cuda_batch.cuh's tstream) through pinned staging and async copies. It also
// owns what every slot kernel shares: the per-env COUNTS (coverage = rows the
// kernel changed, faults = rows whose operands left the range the kernel was
// proved for, refused = status applications refused at the stack cap), and
// the staging of a slot's per-tick HOST INPUTS (the tick, the order requests).
// It has NO method per slot: a slot's kernel is a free function in its slot
// family's TU (cuda_unit_kernels.h), taking the store.
//
// UnitTablesDevice: one (N, T) int64 device buffer, an env's vector the
// device tables of cpp/src/unit_tables_layout.h (GENERATED from
// src/simulation/unit_tables.py), uploaded once when the store is built.
//
// CLASS T (training only, design §6): the played --cuda / --resident tick
// never touches either -- they are not in run_resident_batch's entry or exit
// sets. The gates drive them now (tests/cuda_unit_store_check.py,
// tests/cuda_unit_kernels_check.py); the device backend from P8b.
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
    int64_t* rows() { return d_; }

    // The whole store from / to host memory (count() int64, row-major
    // (N, U, W)): staged through pinned memory, one async copy on the tick
    // stream. upload returns once the host buffer is staged (the copy may
    // still run); download returns once the copy has landed.
    void upload(const int64_t* host);
    void download(int64_t* host);

    // THE COUNTS every slot kernel writes, 3 x N unsigned 64-bit on the
    // device: [0, N) coverage, [N, 2N) faults, [2N, 3N) refused. A kernel's
    // driver calls begin_counts() before its launch (zeroes them, on the tick
    // stream) and end_counts() after (one D2H and a stream sync; each out
    // array holds N int64).
    unsigned long long* counts_dev() { return d_cnt_; }
    void begin_counts();
    void end_counts(int64_t* coverage, int64_t* faults, int64_t* refused);

    // A slot's per-tick host input (`count` int64) staged into a device
    // buffer of its own (`slot` 0 .. INPUT_SLOTS-1; grown, never shrunk, so no
    // allocation in a steady state): pinned copy + one async H2D on the tick
    // stream; returns the device pointer (valid until the slot is staged
    // again). count 0 returns nullptr.
    static constexpr int INPUT_SLOTS = 4;
    const int64_t* stage_input(int slot, const int64_t* host, size_t count);

private:
    void stage_wait();          // the pinned staging buffer is free again
    int n_env_ = 0, n_units_ = 0, width_ = 0;
    int64_t* d_ = nullptr;              // (N, U, W) device
    int64_t* h_ = nullptr;              // pinned staging, count()
    unsigned long long* d_cnt_ = nullptr;   // (3, N) device counts
    unsigned long long* h_cnt_ = nullptr;   // (3, N) pinned
    void* staged_ = nullptr;            // cudaEvent_t: the last copy through h_
    int64_t* d_in_[INPUT_SLOTS] = {};   // per-slot device input
    int64_t* h_in_[INPUT_SLOTS] = {};   // per-slot pinned staging
    size_t in_cap_[INPUT_SLOTS] = {};   // their capacity (int64 count)
};

class UnitTablesDevice {
public:
    // Refuses n_env < 1 or a width that is not the compiled UNIT_TABLES_W.
    UnitTablesDevice(int n_env, int width);
    ~UnitTablesDevice();
    UnitTablesDevice(const UnitTablesDevice&) = delete;
    UnitTablesDevice& operator=(const UnitTablesDevice&) = delete;

    int n_env() const { return n_env_; }
    int width() const { return width_; }
    const int64_t* values() const { return d_; }
    std::uintptr_t device_ptr() const { return reinterpret_cast<std::uintptr_t>(d_); }

    // (N, T) int64 host rows -> device (a blocking copy on the tick stream:
    // a build-time upload, never per tick); download for the gate.
    void upload(const int64_t* host);
    void download(int64_t* host);

private:
    int n_env_ = 0, width_ = 0;
    int64_t* d_ = nullptr;
};

// The compiled layouts (unit_store_layout.h, unit_tables_layout.h): what the
// Python driver checks against the schema before it builds a store.
int unit_store_width();
const char* unit_store_fingerprint();
int unit_store_valid_col();
int unit_store_exposure_col();
int unit_store_exposure_width();
int unit_tables_width();
const char* unit_tables_fingerprint();

}  // namespace breach_cuda
