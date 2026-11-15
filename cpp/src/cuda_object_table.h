#pragma once
// ============================================================================
// #29 P7.0b -- THE OBJECT TABLES of the device (residency design 4.4c
// decision 3): what flies and what is placed (the planted charges today; the
// grenades, rounds, blast sources and entity classes later) as per-env
// (cap, W) int64 rows with a count. See src/simulation/object_tables.py (the
// Python face) and src/simulation/unit_schema.py (OBJECT_TABLES: every
// table's facts; GENERATED cpp/src/object_tables_layout.h).
// ============================================================================
//
// ObjectTable IS a record list (cuda_unit_events.h's UnitRecordList), never a
// parallel mechanism: the rows are the list's records, the count its count,
// the OVERFLOW count its dropped count; a serial per-env producer appends
// through the list's sink (cuda_unit_lib.cuh's emit_record) and a per-row
// kernel stages per row and the list's ONE gather appends per env in row
// order -- both AFTER what the table holds, so an append keeps the order it
// is called in. Unlike a per-tick list it is never begun by a tick: it is
// set by upload() and shrinks only by compact().
//
// compact(): per env, ONE thread walks rows 0 .. count-1 in order, keeps the
// rows whose live column is not 0 (stable: their order is kept), zeroes the
// freed rows and sets the count. Rows past the count are never read.
//
// CAPACITY: past `cap` nothing is written and the env's overflow count rises;
// the Python face refuses the run naming the table and the env.
//
// Every copy and launch on THE tick stream. Plain C++ declarations (no CUDA
// types) so bindings.cpp (cl.exe) can include it. Compiled only when
// BREACH_CUDA. CLASS T: nothing in the played tick uses it.
#include <cstdint>

#include "cuda_unit_events.h"

namespace breach_cuda {

class ObjectTable : public UnitRecordList {
public:
    // Refuses n_env < 1, cap < 1, width < 1, live outside 0 .. width-1,
    // n_units < 1, row_cap < 1.
    ObjectTable(int n_env, int cap, int width, int live, int n_units, int row_cap);
    ~ObjectTable();

    int live_col() const { return live_; }

    // Set every env's table from host memory: rows (N * cap * width) and
    // counts (N, each in 0 .. cap: the caller checks); overflow and staging
    // counts back to 0. Returns once the copies landed.
    void upload(const int64_t* rows, const int64_t* counts);
    // Append, per env e, its first n[e] of `recs` (N * k * width; n[e] <= k:
    // the caller checks) in order after what the table holds: ONE thread per
    // env through the record list's sink. Past cap: not written, counted.
    void append_host(const int64_t* recs, const int64_t* n, int k);
    // The stable compaction of every env (see above).
    void compact();

private:
    int live_ = 0;
    int src_k_ = 0;                 // the append's host staging (records per env)
    int64_t* d_src_ = nullptr;
    int64_t* d_n_ = nullptr;
};

// The compiled layout (object_tables_layout.h): what the Python face checks.
const char* object_tables_fingerprint();

}  // namespace breach_cuda
