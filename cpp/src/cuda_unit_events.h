#pragma once
// ============================================================================
// #29 P6.3 -- THE EVENT RING of the device unit kernels (residency design
// §4.5, "the event ring"; §4.4b). See src/simulation/unit_events.py (the ONE
// field table; GENERATED cpp/src/unit_events_layout.h).
// #29 P6.4a -- generalised: the ring is ONE INSTANCE of a per-env RECORD LIST
// (UnitRecordList) whose record width and row cap are the instance's; the
// request list of slot 1b (src/simulation/unit_requests.py, GENERATED
// cpp/src/unit_requests_layout.h) is the other. One staging, one gather.
// ============================================================================
//
// UnitRecordList: per env a list of `cap` records of `width` int64, kept PER
// TICK (begin() empties it), and the STAGING area the per-row slot kernels
// write into: (N, U, row_cap) records with a count per row. gather() then
// appends, per env, every row's staged records to the list IN ROW ORDER (= id
// order: the reference's emission order, per slot) and empties the staging --
// so the list's order never depends on the thread schedule. A serial per-env
// producer (the applier) appends to the list directly (cuda_unit_damage.cuh's
// env_sink).
//
// UnitEventRing: the record list of UNIT_EVENT_F-wide events with
// UNIT_EVENT_ROW_CAP staged per row (unit_events_layout.h).
//
// CAPACITY: a record past `cap`, or past a row's staging capacity, is never
// written; the env's DROPPED count rises (a counted per-env overflow, never a
// silent drop) and the Python face (simulation.unit_kernels) refuses the run.
//
// Every copy and launch on THE tick stream (cuda_batch.cuh's tstream). Plain
// C++ declarations (no CUDA types) so bindings.cpp (cl.exe) can include it.
// Compiled only when BREACH_CUDA. CLASS T: nothing in the played tick uses it.
#include <cstddef>
#include <cstdint>

namespace breach_cuda {

class UnitRecordList {
public:
    // Refuses n_env < 1, n_units < 1, cap < 1, width < 1, row_cap < 1.
    UnitRecordList(int n_env, int n_units, int cap, int width, int row_cap);
    ~UnitRecordList();
    UnitRecordList(const UnitRecordList&) = delete;
    UnitRecordList& operator=(const UnitRecordList&) = delete;

    int n_env() const { return n_env_; }
    int n_units() const { return n_units_; }
    int cap() const { return cap_; }
    int width() const { return width_; }
    int row_cap() const { return row_cap_; }

    // Device pointers (what a kernel writes through the sinks).
    int64_t* records_dev() { return d_ring_; }           // (N, cap, width)
    int64_t* count_dev() { return d_count_; }            // (N,)
    unsigned long long* dropped_dev() { return d_drop_; }    // (N,)
    int64_t* stage_dev() { return d_stage_; }            // (N, U, row_cap, width)
    int64_t* stage_count_dev() { return d_stage_n_; }    // (N, U)

    // Empty every env's list: counts, dropped and the staging counts to 0.
    void begin();
    // Append the staged records to the list, per env in row order; the
    // staging counts back to 0. A record past cap is dropped and counted.
    void gather();
    // The lists (N * cap * width), the counts (N) and the dropped counts (N)
    // to host memory (any may be null); returns once they landed.
    void download(int64_t* recs, int64_t* counts, int64_t* dropped);

private:
    int n_env_ = 0, n_units_ = 0, cap_ = 0, width_ = 0, row_cap_ = 0;
    int64_t* d_ring_ = nullptr;
    int64_t* d_count_ = nullptr;
    unsigned long long* d_drop_ = nullptr;
    int64_t* d_stage_ = nullptr;
    int64_t* d_stage_n_ = nullptr;
};

class UnitEventRing : public UnitRecordList {
public:
    // Refuses n_env < 1, n_units < 1, e_max < 1.
    UnitEventRing(int n_env, int n_units, int e_max);
    int e_max() const { return cap(); }
    int64_t* ring_dev() { return records_dev(); }        // (N, E_max, F)
};

// The compiled layouts (unit_events_layout.h, unit_requests_layout.h): what
// the Python drivers check.
int unit_events_width();
int unit_events_row_cap();
const char* unit_events_fingerprint();
int unit_requests_width();
int unit_requests_row_cap();
const char* unit_requests_fingerprint();

}  // namespace breach_cuda
