// ============================================================================
// #29 P7e.4 -- the conductor's ONE books read and the host-sync count (see
// cuda_unit_books.h). CLASS T: nothing in the played tick defers; the
// resident tick's tick_sync_point only notes its sync while a count is open.
// ============================================================================
#include "cuda_unit_books.h"

#include <cuda_runtime.h>

#include <chrono>
#include <sstream>
#include <stdexcept>

#include "cuda_batch.cuh"

namespace breach_cuda {

namespace {

struct Arena {
    int64_t* d = nullptr;
    int64_t* h = nullptr;      // pinned
    size_t cap = 0;
    size_t cursor = 0;
    bool on = false;
    std::vector<DeferEntry> entries;
};
Arena g_arena;
std::string g_label = "";
bool g_log_on = false;
std::vector<std::string> g_log;

void bk_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in the conductor's books/" << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

}  // namespace

struct Profile {
    bool on = false;
    std::vector<cudaEvent_t> pool;
    std::vector<std::string> labels;
    std::vector<std::chrono::steady_clock::time_point> host;
    size_t used = 0;
};
Profile g_prof;

void books_set_label(const std::string& label) {
    g_label = label;
    if (g_prof.on && !label.empty()) profile_mark(label.c_str());
}
const std::string& books_label() { return g_label; }

void books_defer_begin(size_t cap_words) {
    if (g_arena.on) throw std::logic_error("books_defer_begin: a deferral is already open");
    if (cap_words > g_arena.cap) {
        bk_check(cudaStreamSynchronize(tstream()), "grow sync");
        if (g_arena.d) cudaFree(g_arena.d);
        if (g_arena.h) cudaFreeHost(g_arena.h);
        g_arena.d = nullptr;
        g_arena.h = nullptr;
        bk_check(cudaMalloc(reinterpret_cast<void**>(&g_arena.d), cap_words * 8), "malloc arena");
        bk_check(cudaMallocHost(reinterpret_cast<void**>(&g_arena.h), cap_words * 8),
                 "malloc pinned arena");
        g_arena.cap = cap_words;
    }
    g_arena.cursor = 0;
    g_arena.entries.clear();
    g_arena.on = true;
}

bool books_deferring() { return g_arena.on; }

bool books_defer(const char* kind, const void* d_src, size_t n_words) {
    if (!g_arena.on) return false;
    if (g_arena.cursor + n_words > g_arena.cap) {
        std::ostringstream os;
        os << "the conductor's books arena: " << g_arena.cursor + n_words << " words past its "
           << "capacity " << g_arena.cap << " (at " << g_label << " / " << kind << ")";
        g_arena.on = false;
        throw std::runtime_error(os.str());
    }
    if (n_words) {
        bk_check(cudaMemcpyAsync(g_arena.d + g_arena.cursor, d_src, n_words * 8,
                                 cudaMemcpyDeviceToDevice, tstream()), "D2D defer");
    }
    DeferEntry e;
    e.label = g_label;
    e.kind = kind;
    e.offset = g_arena.cursor;
    e.n = n_words;
    g_arena.entries.push_back(e);
    g_arena.cursor += n_words;
    return true;
}

std::vector<int64_t> books_defer_end(std::vector<DeferEntry>& entries) {
    if (!g_arena.on) throw std::logic_error("books_defer_end: no deferral is open");
    g_arena.on = false;
    const size_t n = g_arena.cursor;
    std::vector<int64_t> out(n);
    if (n) bk_check(cudaMemcpyAsync(g_arena.h, g_arena.d, n * 8, cudaMemcpyDeviceToHost,
                                    tstream()), "D2H arena");
    note_host_sync("books_read");
    bk_check(cudaStreamSynchronize(tstream()), "books read sync");
    for (size_t i = 0; i < n; ++i) out[i] = g_arena.h[i];
    entries = g_arena.entries;
    g_arena.entries.clear();
    return out;
}

void host_sync_log_begin() {
    g_log.clear();
    g_log_on = true;
}

std::vector<std::string> host_sync_log_end() {
    g_log_on = false;
    std::vector<std::string> out;
    out.swap(g_log);
    return out;
}

void profile_begin() {
    g_prof.on = true;
    g_prof.used = 0;
    g_prof.labels.clear();
    g_prof.host.clear();
}

bool profiling() { return g_prof.on; }

void profile_mark(const char* label) {
    if (!g_prof.on) return;
    if (g_prof.used == g_prof.pool.size()) {
        cudaEvent_t ev = nullptr;
        bk_check(cudaEventCreate(&ev), "profile event");
        g_prof.pool.push_back(ev);
    }
    bk_check(cudaEventRecord(g_prof.pool[g_prof.used], tstream()), "profile record");
    g_prof.labels.emplace_back(label);
    g_prof.host.push_back(std::chrono::steady_clock::now());
    ++g_prof.used;
}

std::vector<ProfileSegment> profile_end() {
    g_prof.on = false;
    std::vector<ProfileSegment> out;
    for (size_t i = 0; i + 1 < g_prof.used; ++i) {
        ProfileSegment s;
        s.label = g_prof.labels[i];
        float ms = 0.0f;
        const cudaError_t e = cudaEventElapsedTime(&ms, g_prof.pool[i], g_prof.pool[i + 1]);
        if (e == cudaErrorNotReady)
            throw std::logic_error("profile_end: an event is not complete (call it after "
                                   "the tick's books read)");
        bk_check(e, "profile elapsed");
        s.gpu_ms = (double)ms;
        s.host_ms = std::chrono::duration<double, std::milli>(g_prof.host[i + 1] -
                                                              g_prof.host[i]).count();
        out.push_back(s);
    }
    g_prof.used = 0;
    g_prof.labels.clear();
    g_prof.host.clear();
    return out;
}

void note_host_sync(const char* what) {
    if (g_log_on) g_log.push_back(g_label + " | " + what);
}

cudaError_t counted_sync(cudaStream_t s, const char* what) {
    note_host_sync(what);
    return cudaStreamSynchronize(s);
}

cudaError_t counted_event_sync(cudaEvent_t ev, const char* what) {
    // an event wait on work already complete costs nothing, but it IS a host
    // wait on the device: counted
    note_host_sync(what);
    return cudaEventSynchronize(ev);
}

}  // namespace breach_cuda
