#pragma once
// ============================================================================
// #29 P7e.4 -- THE CONDUCTOR'S ONE BOOKS READ, and the host-sync count.
// ============================================================================
//
// DEFERRAL. While the device conductor (src/simulation/conductor.py) runs a
// tick it opens the DEFERRED BOOKS (books_defer_begin): every unit-side site
// that would download its counts at its own launch and sync -- a slot's
// (coverage, faults, refused) counts (UnitStoreDevice::end_counts), a record
// list's counts and dropped counts (UnitRecordList::download with no records),
// the applier's books (FieldApplier::download_books), the device flush's books
// and fault words, the door / object slots' counts -- instead copies them
// DEVICE TO DEVICE, on the tick stream, into ONE arena, labelled with the
// conductor's current call line (books_set_label) and the site's kind, and
// hands the host zeros (so the Python face's checks pass). books_defer_end is
// THE ONE READ: one D2H of the arena, one sync; the Python conductor raises
// there by name what each face would have raised at its launch.
//
// THE COUNT. Every host sync of the unit side (and the resident tick's
// tick_sync_point) goes through counted_sync / counted_event_sync (or
// note_host_sync), which count it under the current label while a count is
// open (host_sync_log_begin / _end): the S4 gate's instrument.
//
// Plain C++ declarations (bindings.cpp includes it). CLASS T.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace breach_cuda {

struct DeferEntry {
    std::string label;   // the conductor's call line
    std::string kind;    // the site: counts, list_counts, list_dropped, books, ...
    size_t offset = 0;   // in the arena, int64 words
    size_t n = 0;        // words
};

// The current call line (a label every deferred entry and counted sync takes).
void books_set_label(const std::string& label);
const std::string& books_label();

// Open the deferral (the arena grown to cap_words, the cursor at 0).
void books_defer_begin(size_t cap_words);
bool books_deferring();
// Deferred: an async D2D copy of n 8-byte words from d_src into the arena,
// labelled; returns true (the caller hands the host zeros). Not deferring:
// returns false (the caller downloads as before). Past the arena's capacity:
// throws (a capacity, named).
bool books_defer(const char* kind, const void* d_src, size_t n_words);
// THE ONE READ: close the deferral, one D2H of the arena and one sync;
// `entries` gets the labels. Returns the words.
std::vector<int64_t> books_defer_end(std::vector<DeferEntry>& entries);

// The host-sync count (the S4 instrument).
void host_sync_log_begin();
std::vector<std::string> host_sync_log_end();
void note_host_sync(const char* what);

}  // namespace breach_cuda
