// ============================================================================
// #29 P6.2 -- THE ORDER'S TWO SLOTS on the device unit store: slot 1b'
// (unit_orders.apply_requests) and slot 10' (unit_orders.expire). See
// cuda_unit_kernels.h; residency design §4.4b.
//
// Integer for integer src/simulation/unit_orders.py:
//   1b'  this tick's latched requests (the host's Simulation._order_requests,
//        captured when the slot is entered and staged as (N, R, 5) rows
//        (uid, what, where, until, target), uid -1 = none) written onto their
//        rows -- the four columns, or the four defaults for what = 0 -- for a
//        row still alive and not a zombie. Dropping the requests is the
//        host's side (the dict is cleared by the slot's owner).
//   10'  every valid row whose order has a limit and whose limit has come
//        (order_until <= the env's tick, read after the advance) is cleared;
//        like expire, it reads no life state.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

//: unit_orders.NO_ORDER (what, where, until, target) and its codes.
constexpr int64_t ORDER_NONE = 0;
constexpr int64_t NO_PLACE = -1;
constexpr int64_t NO_LIMIT = -1;
constexpr int64_t NO_TARGET = -1;
constexpr int REQ_W = 5;

__device__ __forceinline__ void put_order(int64_t* row, int64_t what, int64_t where,
                                          int64_t until, int64_t target, bool& changed) {
    put(row, UNIT_COL_ORDER_WHAT, what, changed);
    put(row, UNIT_COL_ORDER_WHERE, where, changed);
    put(row, UNIT_COL_ORDER_UNTIL, until, changed);
    put(row, UNIT_COL_ORDER_TARGET, target, changed);
}

// Slot 1b': one thread per unit row of one env.
__global__ void k_apply_orders(int64_t* store, int n_units, const int64_t* requests,
                               int n_req, unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    if (!row_alive(row) || row[UNIT_COL_IS_ZOMBIE]) return;
    const int64_t* req = bptr(requests, (size_t)n_req * REQ_W);
    const int64_t id = row[UNIT_COL_ID];
    for (int i = 0; i < n_req; ++i) {
        const int64_t* q = req + (size_t)i * REQ_W;
        if (q[0] < 0 || q[0] != id) continue;
        bool changed = false;
        if (q[1] == ORDER_NONE) {
            put_order(row, ORDER_NONE, NO_PLACE, NO_LIMIT, NO_TARGET, changed);
        } else {
            put_order(row, q[1], q[2], q[3], q[4], changed);
        }
        if (changed) count_changed(cnt);
        return;                              // one request per unit (the dict's key)
    }
}

// Slot 10': one thread per unit row of one env.
__global__ void k_expire_orders(int64_t* store, int n_units, const int64_t* ticks,
                                unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row)) return;
    const int64_t tick = ticks[benv()];
    const int64_t until = row[UNIT_COL_ORDER_UNTIL];
    if (row[UNIT_COL_ORDER_WHAT] != ORDER_NONE && until != NO_LIMIT && until <= tick) {
        bool changed = false;
        put_order(row, ORDER_NONE, NO_PLACE, NO_LIMIT, NO_TARGET, changed);
        if (changed) count_changed(cnt);
    }
}

}  // namespace

void unit_apply_orders(UnitStoreDevice& store, const UnitTablesDevice& tables,
                       const int64_t* requests, int n_req, const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_apply_orders");
    if (n_req < 1 || requests == nullptr)
        throw std::invalid_argument("unit_apply_orders: requests must be (N, n_req >= 1, 5)");
    const int64_t* d_req = store.stage_input(0, requests,
                                             (size_t)store.n_env() * n_req * REQ_W);
    store.begin_counts();
    k_apply_orders<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), d_req, n_req, store.counts_dev());
    unit_finish(store, out, "unit_apply_orders");
}

void unit_expire_orders(UnitStoreDevice& store, const UnitTablesDevice& tables,
                        const int64_t* ticks, const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_expire_orders");
    if (ticks == nullptr) throw std::invalid_argument("unit_expire_orders: null ticks");
    const int64_t* d_ticks = store.stage_input(0, ticks, (size_t)store.n_env());
    store.begin_counts();
    k_expire_orders<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), d_ticks, store.counts_dev());
    unit_finish(store, out, "unit_expire_orders");
}

}  // namespace breach_cuda
