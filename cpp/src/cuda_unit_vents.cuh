#pragma once
// ============================================================================
// #29 P6b.2 -- THE VENT CIRCULATION SWEEP as a device function (residency
// design 4.4c, the cut table's P6b.2 row; decision 1: slot 9e is a STAGE of
// the one applier kernel walking the entity tables in the reference's order,
// each entity applied AT ONCE). src/simulation/vent_system.sweep_vents /
// _duct_sweep integer for integer, on ONE env: its duct table
// (unit_schema.DUCT_TABLE: the plenum ledgers, ordinal order), its vent table
// (unit_schema.VENT_TABLE: each vent's duct as that duct's ENTITY ORDINAL,
// its role, aperture, Bresenham accrual and guard count, ordinal order) and
// its planes (cuda_topology.cuh). The vents are bus-independent: alive and
// every load constant are read as the host left them.
//
// Per duct, in row order (an unwired vent -- duct ordinal -1 or no such duct
// -- is inert):
//   0. THE RANGE CHECK (below): a duct outside it is REFUSED BY NAME (the
//      `ledger_refused` book; nothing of the duct written) -- before any write.
//   1. every RETURN vent of the duct, in vent-row order: dead skipped; a
//      blocked aperture (topo::aperture_blocked: solid / thermal_solid /
//      vacuum / ambient / flooded) counts a guard skip, its accumulator
//      frozen; else the Bresenham accrual, and a quantum > 0 extracts through
//      topo::extract_gas_n_vec (T read BEFORE it); the bulk pair credited
//      verbatim, each trace slot through the duct's filter row (scrub =
//      (r * eff) >> 16 to the sink, the rest to the plenum), the energy
//      credit BULK-ONLY at the tile's pre-edit T.
//   2. every SUPPLY vent: dead skipped; blocked: a guard skip; else the
//      accrual, a quantum > 0 its weight.
//   3. N_bulk < vent_n_eps: the energy wiped into e_wipe; nothing deposited.
//   4. total = min(N_bulk, sum of weights); t_dep = floordiv_q(e, N_bulk);
//      per ready vent its share (muldiv_floor_q), split over the REMAINING
//      bulk pair (gas_proportional_split: floor shares, the shortfall
//      cascaded O2 first), its trace at the plenum ratio against the ORIGINAL
//      holdings; the plenum debited; each deposit through
//      topo::inject_gas_n_vec at t_dep with the rails; e -= sum dN * t_dep;
//      the rail hits counted.
//
// THE PROVED RANGE (design 4.4c "As built (P6b.2)"; the T bound widened by
// #217 §8, 2026-11-21). With the tile planes >= 0 and int32 (a tile's bulk
// < 2^32, a trace slice < 2^31) and the tile's relative T an int32 plane
// (|T| <= 2^31: the STORE's bound, so every T the extraction accepts --
// its pre-check |T_abs| < 2^30, cuda_topology.cuh's one-term primitive,
// i.e. every legal gas temperature up to T_MAX_PHYS -- is inside it),
// ONE return vent of quantum q can add at most min(q, 2^32) to the bulk pair,
// < 2^31 to a trace slot and to its sink, and min(q, 2^32) * 2^31 in
// magnitude to e_plenum; a supply only drains (the deposit never takes more
// than the plenum holds: the floor split and floor subadditivity), keeps
// |e| <= max(|e|, N_bulk), and adds 1 to a rail count. So, with B_N = sum over
// the duct's returns of min(q_r, 2^32), B_E = B_N * 2^31, R / S the return /
// supply counts, a duct whose ledgers satisfy AT THE HEAD OF ITS SWEEP
//   0 <= o2_raw, n2_raw            <= 2^61 - B_N
//   0 <= trace_raw[i]              <= 2^54 - R * 2^31
//   0 <= sink[i]                   <= 2^62 - R * 2^31
//   |e_plenum|                     <= 2^61 - B_E
//   |e_wipe|                       <= 2^62
//   0 <= rail_lo_hits, rail_hi_hits <= 2^62 - S
//   0 <= guard_skips (each vent)   <= 2^62 - 1
// keeps every intermediate in int64: N_bulk <= 2^62; e + every credit, the
// wipe e_wipe + e (<= 2^62 + 2^61), the debit sum (|sum dN t_dep| <= |e| +
// N_bulk < 2^63); the three products (total * w, holding * share, share *
// trace) through muldiv_floor_q (exact; each quotient <= its first operand);
// a deposit's trace (<= trace_raw <= 2^54) inside inject_gas_n_vec's range.
// The vent rows' load constants are checked too (tps in [1, 2^16], accum in
// [0, tps), the rate in [0, 2^39], so a quantum < 2^40 is inside the
// extraction's range; a role 0 / 1; the aperture in the grid): a row outside
// them is a FAULT. Past the range the device REFUSES; Python's ints would
// carry on (the serializer packs the ledgers as int64: it fails loud past
// int64 itself).
//
// A FAULT (counted in `faults`, the Python face refuses the run): a vent row
// outside its constants, a filter index outside the table, a seam call
// refused (the reference would raise or the operands left the seam's proved
// range). The duct is abandoned there (its ledger row not written).
// `applied` counts extractions that took gas and deposits that landed.
//
// Device-only header; included by cuda_applier.cu only. CLASS T.
#include "cuda_topology.cuh"
#include "object_tables_layout.h"

namespace breach_cuda {
namespace vents {

using namespace object_tables;
using namespace unit_tables;
using namespace applier;

constexpr int N_TRACE = UNIT_TAB_VENT_TRACE_IDS_W;
static_assert(N_TRACE == OBJ_DUCT_COL_TRACE_RAW_W && N_TRACE == OBJ_DUCT_COL_SINK_W,
              "a duct's trace and sink slots are the trace ids' count");
constexpr int64_t L_BULK = (int64_t)1 << 61;
constexpr int64_t L_TRACE = (int64_t)1 << 54;
constexpr int64_t L_SINK = (int64_t)1 << 62;
constexpr int64_t L_E = (int64_t)1 << 61;
constexpr int64_t L_WIPE = (int64_t)1 << 62;
constexpr int64_t L_COUNT = (int64_t)1 << 62;
constexpr int64_t TILE_BULK = (int64_t)1 << 32;     // a tile's bulk bound (two int32 planes)
constexpr int64_t TILE_TRACE = (int64_t)1 << 31;    // a trace slice's bound
constexpr int TILE_T_SHIFT = 31;                    // |relative T| <= 2^31 (int32 plane)
constexpr int64_t TPS_MAX = (int64_t)1 << 16;
constexpr int64_t RATE_MAX = (int64_t)1 << 39;
constexpr int VENT_MAX = OBJ_VENT_CAP;              // one duct's supply scratch

// The duct's ledger, as locals.
struct Ledger {
    int64_t o2, n2, e, wipe, lo, hi;
    int64_t tr[N_TRACE], sk[N_TRACE];
};

__device__ inline Ledger read_ledger(const int64_t* d) {
    Ledger L;
    L.o2 = d[OBJ_DUCT_COL_O2_RAW];
    L.n2 = d[OBJ_DUCT_COL_N2_RAW];
    L.e = d[OBJ_DUCT_COL_E_PLENUM];
    L.wipe = d[OBJ_DUCT_COL_E_WIPE];
    L.lo = d[OBJ_DUCT_COL_RAIL_LO_HITS];
    L.hi = d[OBJ_DUCT_COL_RAIL_HI_HITS];
    for (int i = 0; i < N_TRACE; ++i) {
        L.tr[i] = d[OBJ_DUCT_COL_TRACE_RAW + i];
        L.sk[i] = d[OBJ_DUCT_COL_SINK + i];
    }
    return L;
}

__device__ inline void write_ledger(int64_t* d, const Ledger& L) {
    d[OBJ_DUCT_COL_O2_RAW] = L.o2;
    d[OBJ_DUCT_COL_N2_RAW] = L.n2;
    d[OBJ_DUCT_COL_E_PLENUM] = L.e;
    d[OBJ_DUCT_COL_E_WIPE] = L.wipe;
    d[OBJ_DUCT_COL_RAIL_LO_HITS] = L.lo;
    d[OBJ_DUCT_COL_RAIL_HI_HITS] = L.hi;
    for (int i = 0; i < N_TRACE; ++i) {
        d[OBJ_DUCT_COL_TRACE_RAW + i] = L.tr[i];
        d[OBJ_DUCT_COL_SINK + i] = L.sk[i];
    }
}

// The vent row's load constants (a FAULT outside them).
__device__ inline bool vent_row_ok(const topo::TopoEnv& E, const int64_t* v) {
    const int64_t tps = v[OBJ_VENT_COL_TPS], acc = v[OBJ_VENT_COL_ACCUM];
    const int64_t rate = v[OBJ_VENT_COL_RATE_RAW_PER_S], role = v[OBJ_VENT_COL_ROLE];
    return tps >= 1 && tps <= TPS_MAX && acc >= 0 && acc < tps && rate >= 0 &&
           rate <= RATE_MAX && (role == 0 || role == 1) &&
           topo::in_grid64(E, v[OBJ_VENT_COL_APERTURE_Y], v[OBJ_VENT_COL_APERTURE_X]);
}

// VentRuntime.accrue, in place: the whole quanta crossed (accum stays in [0, tps)).
__device__ __forceinline__ int64_t accrue(int64_t* v) {
    const int64_t tps = v[OBJ_VENT_COL_TPS];
    int64_t acc = v[OBJ_VENT_COL_ACCUM] + v[OBJ_VENT_COL_RATE_RAW_PER_S];
    const int64_t emit = acc / tps;                    // acc >= 0, tps >= 1
    acc -= emit * tps;
    v[OBJ_VENT_COL_ACCUM] = acc;
    return emit;
}

__device__ __forceinline__ int64_t abs64(int64_t x) { return x < 0 ? -x : x; }

// THE RANGE CHECK at the head of a duct's sweep (see the header): true when
// every ledger is inside its proved range for this duct's vents.
__device__ inline bool in_range(const Ledger& L, int64_t b_n, int64_t b_e, int64_t r, int64_t s) {
    if (L.o2 < 0 || L.o2 > L_BULK - b_n || L.n2 < 0 || L.n2 > L_BULK - b_n) return false;
    for (int i = 0; i < N_TRACE; ++i) {
        if (L.tr[i] < 0 || L.tr[i] > L_TRACE - r * TILE_TRACE) return false;
        if (L.sk[i] < 0 || L.sk[i] > L_SINK - r * TILE_TRACE) return false;
    }
    if (L.e < -(L_E - b_e) || L.e > L_E - b_e) return false;
    if (L.wipe < -L_WIPE || L.wipe > L_WIPE) return false;
    if (L.lo < 0 || L.lo > L_COUNT - s || L.hi < 0 || L.hi > L_COUNT - s) return false;
    return true;
}

// The whole sweep over this block env's duct table (`drows`, n_ducts rows of
// OBJ_DUCT_W) and vent table (`vrows`, n_vents rows of OBJ_VENT_W).
__device__ inline void sweep(const topo::TopoEnv& E, const int64_t* tab, int64_t* drows,
                             int64_t n_ducts, int64_t* vrows, int64_t n_vents, int64_t* B) {
    const int n_filters = UNIT_TAB_VENT_FILTER_EFF_Q_W / N_TRACE;
    const int64_t n_eps = tab[UNIT_TAB_VENT_N_EPS];
    const int64_t t_min = tab[UNIT_TAB_VENT_T_MIN_Q], t_max = tab[UNIT_TAB_VENT_T_MAX_Q];
    const int o2 = E.o2, n2 = E.n2;
    bool ids_ok = E.n_gases <= topo::GAS_MAX;
    for (int t = 0; t < N_TRACE; ++t) {
        const int64_t g = tab[UNIT_TAB_VENT_TRACE_IDS + t];
        if (g < 0 || g >= E.n_gases || g == o2 || g == n2) ids_ok = false;
    }
    if (n_vents > VENT_MAX || n_vents < 0 || n_ducts < 0 || !ids_ok) {
        B[APPLIER_BOOK_FAULTS] += 1;
        return;
    }
    for (int64_t k = 0; k < n_ducts; ++k) {
        int64_t* d = drows + (size_t)k * OBJ_DUCT_W;
        const int64_t ord = d[OBJ_DUCT_COL_ORDINAL];
        const int64_t fi = d[OBJ_DUCT_COL_FILTER];
        if (fi < 0 || fi >= n_filters) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        const int64_t* eff = tab + UNIT_TAB_VENT_FILTER_EFF_Q + fi * N_TRACE;
        // --- 0. the head: the vents' constants, the step bounds, the range
        bool bad = false;
        int64_t b_n = 0, r_cnt = 0, s_cnt = 0;
        bool guard_ok = true;
        for (int64_t j = 0; j < n_vents; ++j) {
            const int64_t* v = vrows + (size_t)j * OBJ_VENT_W;
            if (v[OBJ_VENT_COL_DUCT_ORDINAL] != ord) continue;
            if (!vent_row_ok(E, v)) { bad = true; break; }
            const int64_t g = v[OBJ_VENT_COL_GUARD_SKIPS];
            if (g < 0 || g > L_COUNT - 1) guard_ok = false;
            if (v[OBJ_VENT_COL_ROLE] == 1) {
                const int64_t q = (v[OBJ_VENT_COL_ACCUM] + v[OBJ_VENT_COL_RATE_RAW_PER_S]) /
                                  v[OBJ_VENT_COL_TPS];
                b_n += q < TILE_BULK ? q : TILE_BULK;      // <= 64 * 2^32: no overflow
                ++r_cnt;
            } else {
                ++s_cnt;
            }
        }
        if (bad) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        Ledger L = read_ledger(d);
        // B_E = B_N * 2^31, refused (as out of range) when it alone passes L_E
        const bool be_ok = b_n <= (L_E >> TILE_T_SHIFT);
        const int64_t b_e = be_ok ? (b_n << TILE_T_SHIFT) : 0;
        if (!be_ok || !guard_ok || !in_range(L, b_n, b_e, r_cnt, s_cnt)) {
            B[APPLIER_BOOK_LEDGER_REFUSED] += 1;
            continue;
        }
        // --- 1. the returns
        bool fault = false;
        for (int64_t j = 0; j < n_vents && !fault; ++j) {
            int64_t* v = vrows + (size_t)j * OBJ_VENT_W;
            if (v[OBJ_VENT_COL_DUCT_ORDINAL] != ord || v[OBJ_VENT_COL_ROLE] != 1) continue;
            if (v[OBJ_VENT_COL_ALIVE] == 0) continue;
            const int fy = (int)v[OBJ_VENT_COL_APERTURE_Y], fx = (int)v[OBJ_VENT_COL_APERTURE_X];
            if (topo::aperture_blocked(E, fy, fx)) {
                v[OBJ_VENT_COL_GUARD_SKIPS] += 1;          // counted no-op; accum frozen
                continue;
            }
            const int64_t emit = accrue(v);
            if (emit <= 0) continue;
            const int64_t t_tile = (int64_t)E.temperature[fy * E.w + fx];   // pre-edit
            int64_t rem[topo::GAS_MAX];
            int64_t taken = 0;
            if (topo::extract_gas_n_vec(E, tab, fy, fx, emit, rem, &taken, B) != 0) {
                fault = true;
                break;
            }
            int64_t total = 0;
            for (int g = 0; g < E.n_gases; ++g) total += rem[g];
            if (total == 0) continue;                       // the aperture was empty
            L.o2 += rem[o2];
            L.n2 += rem[n2];
            for (int t = 0; t < N_TRACE; ++t) {
                const int64_t r = rem[(int)tab[UNIT_TAB_VENT_TRACE_IDS + t]];
                if (r == 0) continue;
                const int64_t scrub = (r * eff[t]) >> 16;   // r >= 0, < 2^31; eff <= 2^16
                L.tr[t] += r - scrub;
                L.sk[t] += scrub;
            }
            L.e += (rem[o2] + rem[n2]) * t_tile;
            B[APPLIER_BOOK_APPLIED] += 1;
        }
        if (fault) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        // --- 2. the supplies' weights
        int ready_v[VENT_MAX];
        int64_t ready_w[VENT_MAX];
        int n_ready = 0;
        for (int64_t j = 0; j < n_vents; ++j) {
            int64_t* v = vrows + (size_t)j * OBJ_VENT_W;
            if (v[OBJ_VENT_COL_DUCT_ORDINAL] != ord || v[OBJ_VENT_COL_ROLE] != 0) continue;
            if (v[OBJ_VENT_COL_ALIVE] == 0) continue;
            if (topo::aperture_blocked(E, (int)v[OBJ_VENT_COL_APERTURE_Y],
                                       (int)v[OBJ_VENT_COL_APERTURE_X])) {
                v[OBJ_VENT_COL_GUARD_SKIPS] += 1;
                continue;
            }
            const int64_t w = accrue(v);
            if (w > 0) {
                ready_v[n_ready] = (int)j;
                ready_w[n_ready] = w;
                ++n_ready;
            }
        }
        const int64_t n_bulk = L.o2 + L.n2;
        // --- 3. the near-empty plenum
        if (n_bulk < n_eps) {
            if (L.e != 0) {
                L.wipe += L.e;
                L.e = 0;
            }
            write_ledger(d, L);
            continue;
        }
        int64_t sum_w = 0;
        for (int a = 0; a < n_ready; ++a) sum_w += ready_w[a];
        const int64_t total_avail = n_bulk < sum_w ? n_bulk : sum_w;
        if (total_avail <= 0) {
            write_ledger(d, L);
            continue;
        }
        // --- 4. the distribution
        const int64_t t_dep = fixedpoint::floordiv_q(L.e, n_bulk);
        int64_t o2_rem = L.o2, n2_rem = L.n2;
        int64_t out_o2 = 0, out_n2 = 0, out_tr[N_TRACE];
        for (int t = 0; t < N_TRACE; ++t) out_tr[t] = 0;
        int dep_v[VENT_MAX];
        int64_t dep_c[VENT_MAX][2 + N_TRACE];
        int n_dep = 0;
        for (int a = 0; a < n_ready && !fault; ++a) {
            const int64_t share = fixedpoint::muldiv_floor_q(total_avail, ready_w[a], sum_w);
            if (share < 0) { fault = true; break; }
            if (share <= 0) continue;
            // GameMap.gas_proportional_split([o2_rem, n2_rem], share)
            const int64_t n_tot = o2_rem + n2_rem;
            int64_t tk0 = 0, tk1 = 0;
            if (n_tot > 0) {
                tk0 = fixedpoint::muldiv_floor_q(o2_rem, share, n_tot);
                tk1 = fixedpoint::muldiv_floor_q(n2_rem, share, n_tot);
                if (tk0 < 0 || tk1 < 0) { fault = true; break; }
                int64_t short_ = share - tk0 - tk1;
                if (short_ > 0) {
                    const int64_t room = o2_rem - tk0;
                    if (room > 0) {
                        const int64_t grab = room < short_ ? room : short_;
                        tk0 += grab;
                        short_ -= grab;
                    }
                }
                if (short_ > 0) {
                    const int64_t room = n2_rem - tk1;
                    if (room > 0) {
                        const int64_t grab = room < short_ ? room : short_;
                        tk1 += grab;
                        short_ -= grab;
                    }
                }
            }
            o2_rem -= tk0;
            n2_rem -= tk1;
            dep_v[n_dep] = ready_v[a];
            dep_c[n_dep][0] = tk0;
            dep_c[n_dep][1] = tk1;
            for (int t = 0; t < N_TRACE; ++t) {
                const int64_t tr = fixedpoint::muldiv_floor_q(share, L.tr[t], n_bulk);
                if (tr < 0) { fault = true; break; }
                dep_c[n_dep][2 + t] = tr;
                out_tr[t] += tr;
            }
            out_o2 += tk0;
            out_n2 += tk1;
            ++n_dep;
        }
        if (fault) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        if (n_dep == 0) {                                   // every share floored to 0
            write_ledger(d, L);
            continue;
        }
        L.o2 -= out_o2;
        L.n2 -= out_n2;
        for (int t = 0; t < N_TRACE; ++t) L.tr[t] -= out_tr[t];
        int64_t debit = 0;
        for (int a = 0; a < n_dep; ++a) {
            const int64_t* v = vrows + (size_t)dep_v[a] * OBJ_VENT_W;
            int64_t comp[topo::GAS_MAX];
            for (int g = 0; g < topo::GAS_MAX; ++g) comp[g] = 0;
            for (int t = 0; t < N_TRACE; ++t)
                comp[(int)tab[UNIT_TAB_VENT_TRACE_IDS + t]] = dep_c[a][2 + t];
            comp[o2] = dep_c[a][0];
            comp[n2] = dep_c[a][1];
            int64_t tn = 0, rail = 0;
            if (topo::inject_gas_n_vec(E, tab, v[OBJ_VENT_COL_APERTURE_Y],
                                       v[OBJ_VENT_COL_APERTURE_X], comp, t_dep, t_min, t_max,
                                       &tn, &rail, B, topo::vent_inject_books()) != 0) {
                fault = true;
                break;
            }
            debit += (dep_c[a][0] + dep_c[a][1]) * t_dep;
            if (rail < 0) L.lo += 1;
            else if (rail > 0) L.hi += 1;
            B[APPLIER_BOOK_APPLIED] += 1;
        }
        if (fault) {
            B[APPLIER_BOOK_FAULTS] += 1;
            continue;
        }
        L.e -= debit;
        write_ledger(d, L);
    }
}

}  // namespace vents
}  // namespace breach_cuda
