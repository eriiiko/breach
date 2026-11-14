#pragma once
// ============================================================================
// #29 P6.3 -- THE EVENT RING of the device unit kernels (residency design
// §4.5, "the event ring"; §4.4b). See src/simulation/unit_events.py (the ONE
// field table; GENERATED cpp/src/unit_events_layout.h).
// ============================================================================
//
// UnitEventRing: per env a ring of E_max records of UNIT_EVENT_F int64 (the
// fields field_ab_harness._event_record hashes plus the render-only
// direction), kept PER TICK (begin() empties it), and the STAGING area the
// per-row slot kernels write into: (N, U, UNIT_EVENT_ROW_CAP) records with a
// count per row. gather() then appends, per env, every row's staged records
// to the ring IN ROW ORDER (= id order: the reference's emission order, per
// slot) and empties the staging -- so the ring's order never depends on the
// thread schedule. A serial per-env producer (P6.4's applier) appends to the
// ring directly (cuda_unit_damage.cuh's env_sink).
//
// CAPACITY: a record past E_max, or past a row's staging capacity, is never
// written; the env's DROPPED count rises (a counted per-env overflow, never a
// silent drop) and the Python face (simulation.unit_kernels) refuses the run.
//
// Every copy and launch on THE tick stream (cuda_batch.cuh's tstream). Plain
// C++ declarations (no CUDA types) so bindings.cpp (cl.exe) can include it.
// Compiled only when BREACH_CUDA. CLASS T: nothing in the played tick uses it.
#include <cstddef>
#include <cstdint>

namespace breach_cuda {

class UnitEventRing {
public:
    // Refuses n_env < 1, n_units < 1, e_max < 1.
    UnitEventRing(int n_env, int n_units, int e_max);
    ~UnitEventRing();
    UnitEventRing(const UnitEventRing&) = delete;
    UnitEventRing& operator=(const UnitEventRing&) = delete;

    int n_env() const { return n_env_; }
    int n_units() const { return n_units_; }
    int e_max() const { return e_max_; }

    // Device pointers (what a kernel writes through the sinks).
    int64_t* ring_dev() { return d_ring_; }              // (N, E_max, F)
    int64_t* count_dev() { return d_count_; }            // (N,)
    unsigned long long* dropped_dev() { return d_drop_; }    // (N,)
    int64_t* stage_dev() { return d_stage_; }            // (N, U, ROW_CAP, F)
    int64_t* stage_count_dev() { return d_stage_n_; }    // (N, U)

    // Empty every env's ring: counts, dropped and the staging counts to 0.
    void begin();
    // Append the staged records to the ring, per env in row order; the
    // staging counts back to 0. A record past E_max is dropped and counted.
    void gather();
    // The rings (N * E_max * F), the counts (N) and the dropped counts (N)
    // to host memory (any may be null); returns once they landed.
    void download(int64_t* recs, int64_t* counts, int64_t* dropped);

private:
    int n_env_ = 0, n_units_ = 0, e_max_ = 0;
    int64_t* d_ring_ = nullptr;
    int64_t* d_count_ = nullptr;
    unsigned long long* d_drop_ = nullptr;
    int64_t* d_stage_ = nullptr;
    int64_t* d_stage_n_ = nullptr;
};

// The compiled layout (unit_events_layout.h): what the Python driver checks.
int unit_events_width();
int unit_events_row_cap();
const char* unit_events_fingerprint();

}  // namespace breach_cuda
