// ============================================================================
// #29 P6.4b -- TEST-ONLY: the topology seam's on_tile_changed on its own, on
// fixture maps (tests/cuda_unit_topology_check.py, part SEAM). destroy_wall
// only ever turns a tile into AIR, so its membership diff can never RETIRE
// (air is never less accountable than what it replaced); a seal can. This
// probe drives the device function the way every structural writer drives
// the Python one -- `material[y, x] = m; on_tile_changed(y, x)` -- for a list
// of (y, x, m), so the gate reaches both directions of the membership diff
// and every material row as a target. The staged FieldApplier list carries
// the records (kind ignored, `arg` the new material id); the books are the
// applier's. Never called by a slot.
//
// CLASS T, TEST-ONLY: nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_applier.h"
#include "cuda_batch.cuh"
#include "cuda_topology.cuh"
#include "cuda_unit_store.h"
#include "unit_tables_layout.h"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace breach_cuda {

namespace {

using namespace applier;

__global__ void k_probe_tile_changed(const int64_t* recs, const int64_t* counts, int cap,
                                     topo::PlanePtrs P, int h, int w, int n_gases,
                                     const int64_t* tables, int64_t* books) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = benv();
    const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
    int64_t* B = books + (size_t)e * APPLIER_N_BOOKS;
    const topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
    const int M = unit_tables::UNIT_TAB_MAT_FLAMMABLE_W;
    const int64_t k = counts[e] < (int64_t)cap ? counts[e] : (int64_t)cap;
    for (int64_t j = 0; j < k; ++j) {
        const int64_t* r = recs + ((size_t)e * cap + (size_t)j) * APPLIER_F;
        const int64_t y = r[APPLIER_FX_A0], x = r[APPLIER_FX_A1], m = r[APPLIER_FX_A2];
        if (y < 0 || y >= h || x < 0 || x >= w || m < 0 || m >= M) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        E.material[(int)y * w + (int)x] = (int8_t)m;
        topo::on_tile_changed(E, tab, (int)y, (int)x, B);
        B[APPLIER_BOOK_APPLIED] += 1;
    }
}

}  // namespace

void topology_probe_tile_changed(FieldApplier& a, const UnitTablesDevice& tables,
                                 const ApplierPlanes& planes, int64_t* books_out) {
    topo::PlanePtrs P;
    for (int k = 0; k < APPLIER_N_PLANES; ++k) P.p[k] = (unsigned long long)planes.ptr[k];
    cudaMemsetAsync(a.books_dev(), 0, (size_t)a.n_env() * APPLIER_N_BOOKS * sizeof(int64_t),
                    tstream());
    k_probe_tile_changed<<<dim3(1, (unsigned)a.n_env()), 32, 0, tstream()>>>(
        a.list().records_dev(), a.list().count_dev(), a.cap(), P, planes.h, planes.w, planes.n_gases,
        tables.values(), a.books_dev());
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in topology_probe: " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
    a.download_books(books_out);
}

// ============================================================================
// #29 P6b.1 -- TEST-ONLY: the seam's seal / unseal / can_seal, the gas-N
// primitives and _aperture_blocked on their own (tests/cuda_unit_doors_check
// .py, part PROBE): per env a list of OPS walked serially on thread 0, each
// the device function on the env's planes in place, its result row (code,
// value, then the per-slice withdrawal) and the books. Never called by a slot.
//   op 1 seal_tiles(span, mid)      a0 span start, a1 span length, a2 mid
//   op 2 unseal_tiles(span)         a0, a1
//   op 3 can_seal_tiles(span)       a0, a1                    value 1 = yes
//   op 4 inject_gas_n(y, x, dn)     a0 y, a1 x, a2 dn
//   op 5 extract_gas_n_vec(y,x,dn)  a0, a1, a2  value = the bulk taken
//   op 6 transfer_gas_n(s, d, dn)   a0 sy, a1 sx, a2 dy, a3 dx, a4 dn
//   op 7 _aperture_blocked(y, x)    a0, a1                    value 1 = blocked
// A span is rows of the env's (T, 2) tile list.
// ============================================================================
namespace {

constexpr int PROBE_OP_W = 8;
constexpr int PROBE_RES_W = 2 + topo::GAS_MAX;

__global__ void k_probe_seam(const int64_t* ops, const int64_t* n_ops, int k_cap,
                             const int64_t* tiles, int t_cap, topo::PlanePtrs P, int h, int w,
                             int n_gases, const int64_t* tables, int64_t* books, int64_t* res) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int e = benv();
    const int64_t* tab = bptr(tables, (size_t)unit_tables::UNIT_TABLES_W);
    int64_t* B = books + (size_t)e * APPLIER_N_BOOKS;
    const topo::TopoEnv E = topo::env_view(P, e, h, w, n_gases, tab);
    const int64_t* T = tiles + (size_t)e * t_cap * 2;
    for (int64_t j = 0; j < n_ops[e]; ++j) {
        const int64_t* o = ops + ((size_t)e * k_cap + (size_t)j) * PROBE_OP_W;
        int64_t* r = res + ((size_t)e * k_cap + (size_t)j) * PROBE_RES_W;
        for (int c = 0; c < PROBE_RES_W; ++c) r[c] = 0;
        const int64_t op = o[0];
        if (op >= 1 && op <= 3) {
            const int64_t a = o[1], n = o[2];
            if (a < 0 || n < 0 || a + n > t_cap) { r[0] = APPLIER_SEAM_FAULT; continue; }
            const int64_t* sp = T + (size_t)a * 2;
            if (op == 1) r[0] = topo::seal_tiles(E, tab, sp, 2, (int)n, o[3], B);
            else if (op == 2) r[0] = topo::unseal_tiles(E, tab, sp, 2, (int)n, B);
            else {
                r[0] = topo::can_seal(E, tab, sp, 2, (int)n);
                r[1] = r[0] == 0 ? 1 : 0;
            }
        } else if (op == 4) {
            r[0] = topo::inject_gas_n(E, tab, o[1], o[2], o[3], B);
        } else if (op == 5) {
            r[0] = topo::extract_gas_n_vec(E, tab, o[1], o[2], o[3], r + 2, r + 1, B);
        } else if (op == 6) {
            r[0] = topo::transfer_gas_n(E, tab, o[1], o[2], o[3], o[4], o[5], r + 1, B);
        } else if (op == 7) {
            if (o[1] < 0 || o[1] >= h || o[2] < 0 || o[2] >= w) { r[0] = APPLIER_SEAM_OUT_OF_BOUNDS; continue; }
            r[1] = topo::aperture_blocked(E, (int)o[1], (int)o[2]) ? 1 : 0;
        } else {
            r[0] = APPLIER_SEAM_FAULT;
        }
    }
}

}  // namespace

void topology_probe_seam(const UnitTablesDevice& tables, const ApplierPlanes& planes,
                         const int64_t* ops, const int64_t* n_ops, int k_cap,
                         const int64_t* tiles, int t_cap, int64_t* results_out,
                         int64_t* books_out) {
    const int n = tables.n_env();
    topo::PlanePtrs P;
    for (int k = 0; k < APPLIER_N_PLANES; ++k) P.p[k] = (unsigned long long)planes.ptr[k];
    const size_t no = (size_t)n * k_cap * PROBE_OP_W, nt = (size_t)n * t_cap * 2;
    const size_t nr = (size_t)n * k_cap * PROBE_RES_W, nb = (size_t)n * APPLIER_N_BOOKS;
    int64_t *d_ops = nullptr, *d_n = nullptr, *d_t = nullptr, *d_r = nullptr, *d_b = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_ops), no * sizeof(int64_t));
    cudaMalloc(reinterpret_cast<void**>(&d_n), (size_t)n * sizeof(int64_t));
    cudaMalloc(reinterpret_cast<void**>(&d_t), (nt ? nt : 1) * sizeof(int64_t));
    cudaMalloc(reinterpret_cast<void**>(&d_r), nr * sizeof(int64_t));
    cudaMalloc(reinterpret_cast<void**>(&d_b), nb * sizeof(int64_t));
    cudaMemcpyAsync(d_ops, ops, no * sizeof(int64_t), cudaMemcpyHostToDevice, tstream());
    cudaMemcpyAsync(d_n, n_ops, (size_t)n * sizeof(int64_t), cudaMemcpyHostToDevice, tstream());
    if (nt) cudaMemcpyAsync(d_t, tiles, nt * sizeof(int64_t), cudaMemcpyHostToDevice, tstream());
    cudaMemsetAsync(d_b, 0, nb * sizeof(int64_t), tstream());
    k_probe_seam<<<dim3(1, (unsigned)n), 32, 0, tstream()>>>(
        d_ops, d_n, k_cap, d_t, t_cap, P, planes.h, planes.w, planes.n_gases, tables.values(), d_b,
        d_r);
    cudaError_t err = cudaGetLastError();
    cudaMemcpyAsync(results_out, d_r, nr * sizeof(int64_t), cudaMemcpyDeviceToHost, tstream());
    cudaMemcpyAsync(books_out, d_b, nb * sizeof(int64_t), cudaMemcpyDeviceToHost, tstream());
    if (err == cudaSuccess) err = cudaStreamSynchronize(tstream());
    cudaFree(d_ops);
    cudaFree(d_n);
    cudaFree(d_t);
    cudaFree(d_r);
    cudaFree(d_b);
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << "CUDA error in topology_probe_seam: " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
}

}  // namespace breach_cuda
