// ==========================================================================
// #29 P8.7: THE DEVICE BOOKS (see cuda_obs.h and src/simulation/device_books.py),
// Class T. The twin of the env's per-tick reward books (rl_env.EnvRunner.step's
// loop body: rl_env.count_events, reward_v1.RewardBooks.end / finish, the
// MatchRules end, EnvRunner._reward), line for line, ONE THREAD PER ENV (the
// books are a serial walk in the reference's event order), on THE tick stream,
// after the device team picture (its bars) and the observation's blocks of the
// tick. The host books are the reference. Nothing here writes a synced buffer;
// the fault words and the per-env "ended now" flags ride the conductor's one
// books read (kind "reward_books").
//
// Every float64 +, -, *, / goes through the observation's float kit (never
// fused); the old counters are float32 (numpy 2's NEP 50: the damage cast to
// float32 once, a float32 add); the served arrays take ONE cast each.
//
// THE ORDER of one tick (env e, books not finished, ended latch clear):
//   the step's accumulators zeroed on a step's first tick (start_step);
//   begin(): the identity on the card (the snapshot IS the tick's start);
//   the old counters over the ring (rows ended before the tick frozen);
//   end(): the walk over the ring (hits, kills) through _settle; the turnings
//     and the mismatch check per store row; the snapshot; the team part (the
//     standing in store row order, the teams' first-appearance order); the
//     eliminations; the flush;
//   the end of the match (fewer than two marine teams alive, or the limit
//     tick): the result, finish(), the ended latch, the ending tick;
//   on a decision tick or the ending tick: the served arrays.
// ==========================================================================
#include "cuda_obs.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"          // tstream(): THE tick stream (blocking)
#include "cuda_obs_float.cuh"
#include "cuda_obs_picture.cuh"
#include "cuda_unit_books.h"       // books_defer: the words ride the one books read
#include "cuda_unit_events.h"
#include "obs_layout.h"
#include "unit_draws_layout.h"
#include "unit_events_layout.h"
#include "unit_store_layout.h"

namespace breach_cuda {

using namespace obs_layout;

namespace {

using namespace unit_store;

void need(bool ok, const char* who, const char* what) {
    if (!ok) throw std::invalid_argument(std::string(who) + ": " + what);
}

void launched(const char* who) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(who) + ": launch failed: " + cudaGetErrorString(err));
}

// The block's sections (device_books.layout, in order).
struct BkLayout {
    int64_t unit, h, a, hz, ord, ord_n, living, elim, ended, pend, rows_ended, scr, cnt, raw, old,
        rest, dbal, outc, work_h, work_al, sc, w;
};

__host__ __device__ inline BkLayout bk_layout(int T, int U, int R) {
    BkLayout L{};
    int64_t o = 0;
    L.unit = o; o += (int64_t)U * OBS_BK_UNIT_W;
    L.h = o; o += T;
    L.a = o; o += T;
    L.hz = o; o += 1;
    L.ord = o; o += T;
    L.ord_n = o; o += 1;
    L.living = o; o += 1;
    L.elim = o; o += 1;
    L.ended = o; o += R;
    L.pend = o; o += R;
    L.rows_ended = o; o += R;
    L.scr = o; o += (int64_t)R * OBS_BK_NB;
    L.cnt = o; o += (int64_t)R * OBS_BK_NB;
    L.raw = o; o += (int64_t)R * OBS_BK_NB;
    L.old = o; o += (int64_t)R * OBS_BK_K0;
    L.rest = o; o += T;
    L.dbal = o; o += T;
    L.outc = o; o += T;
    L.work_h = o; o += U;
    L.work_al = o; o += U;
    L.sc = o; o += OBS_BK_S_W;
    L.w = o;
    return L;
}

__device__ __forceinline__ double dv(int64_t b) { return __longlong_as_double(b); }
__device__ __forceinline__ int64_t db(double x) { return __double_as_longlong(x); }
// numpy's `a[i] += x` on a float64 array (one add, the result stored)
__device__ __forceinline__ void fadd(int64_t* p, double x) { *p = db(obs_f64::add(dv(*p), x)); }
// numpy 2's `a[i] += x` on a float32 array with a Python float x (NEP 50: x
// cast to float32 once, then a float32 add)
__device__ __forceinline__ void f32add(int64_t* p, double x) {
    const float a = __uint_as_float((unsigned)(*p & 0xffffffffLL));
    *p = (int64_t)__float_as_uint(__fadd_rn(a, __double2float_rn(x)));
}
__device__ __forceinline__ float f32of(int64_t w) {
    return __uint_as_float((unsigned)(w & 0xffffffffLL));
}

__device__ __forceinline__ bool alive(const int64_t* u) {
    return u[UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE;
}

// reward_v1.bar_ratio: min(max(hp_q, 0), bar_q) / float(bar_q), 0 without a bar
__device__ __forceinline__ double bar_ratio(int64_t hp, int64_t bar) {
    if (bar <= 0) return 0.0;
    int64_t m = hp > 0 ? hp : 0;
    m = m < bar ? m : bar;
    return obs_f64::div((double)m, (double)bar);
}

__device__ __forceinline__ double health(int64_t hp, bool al, int64_t bar) {
    return al ? bar_ratio(hp, bar) : 0.0;
}

// One env's view.
struct Env {
    int64_t* blk;
    BkLayout L;
    const int64_t* rows;     // the store rows (U, UNIT_STORE_W)
    const int32_t* rmap;     // the books rows' store rows (R,)
    const int64_t* ot;       // the obs table row
    const int64_t* pic;      // the picture block
    obs_pic::PicLayout PL;
    int U, R, S, T, n_match;
    double mb, ma, zb;
    __device__ int64_t* scr() const { return blk + L.scr; }
    __device__ int64_t* snap(int r) const { return blk + L.unit + (int64_t)r * OBS_BK_UNIT_W; }
    __device__ double term(int i) const { return dv(ot[OBS_TAB_BOOK_TERMS + i]); }
};

// The store row holding unit id `uid` (-1: none).
__device__ __forceinline__ int row_of(const Env& E, int64_t uid) {
    for (int r = 0; r < E.U; ++r) {
        const int64_t* u = E.rows + (size_t)r * UNIT_STORE_W;
        if (u[UNIT_COL_VALID] && u[UNIT_COL_ID] == uid) return r;
    }
    return -1;
}

// The books row of store row r (row_of / _id_row: the roster), -1 none.
__device__ __forceinline__ int book_row(const Env& E, int r) {
    if (r < 0) return -1;
    for (int k = 0; k < E.R; ++k)
        if (E.rmap[k] == r) return k;
    return -1;
}

// _member: the books row of store row r when its unit is on its row's team
// at the tick's start and not a zombie (a corpse included), else -1.
__device__ __forceinline__ int member(const Env& E, int r) {
    const int k = book_row(E, r);
    if (k < 0) return -1;
    const int64_t* s = E.snap(r);
    if (s[OBS_BK_UNIT_ZOMBIE] || s[OBS_BK_UNIT_TEAM] != (int64_t)(k / E.S)) return -1;
    return k;
}

// The picture's bar of store row r (bar_q_of(bars.get(uid, 0.0))).
__device__ __forceinline__ int64_t bar_now(const Env& E, int r) {
    const int64_t* m = obs_pic::unit_of(E.pic, E.PL, r);
    return m[OBS_PIC_UNIT_BAR_SET] ? m[OBS_PIC_UNIT_BAR_Q] : 0;
}

// RewardBooks._settle (cases 1-4).
__device__ void settle(const Env& E, int vr, double r, int dead, int64_t attacker, bool hit) {
    using namespace obs_f64;
    const int64_t* st = E.snap(vr);
    const int64_t vuid = E.rows[(size_t)vr * UNIT_STORE_W + UNIT_COL_ID];
    const bool zomb = st[OBS_BK_UNIT_ZOMBIE] != 0;
    const int mv = member(E, vr);
    int mav = -1;
    if (attacker >= 0 && attacker != vuid) {
        const int ar = row_of(E, attacker);
        if (ar >= 0) mav = member(E, ar);
    }
    int64_t* scr = E.scr();
    if (mv >= 0) {
        fadd(scr + (int64_t)mv * OBS_BK_NB + OBS_BK_C_BARS_LOST, r);
        fadd(scr + (int64_t)mv * OBS_BK_NB + OBS_BK_C_DIED, (double)dead);
    }
    bool enemy = false;
    const int mvt = mv >= 0 ? mv / E.S : -1;
    const int mat = mav >= 0 ? mav / E.S : -1;
    if (mav >= 0) {
        int64_t* a = scr + (int64_t)mav * OBS_BK_NB;
        if (zomb) {
            fadd(a + OBS_BK_C_BARS_ZOMBIE, r);
        } else if (mv >= 0 && mvt != mat) {
            enemy = true;
            fadd(a + OBS_BK_C_BARS_ENEMY, r);
            fadd(a + OBS_BK_C_KILLS_ENEMY, (double)dead);
        } else if (mv >= 0 && hit) {
            fadd(a + OBS_BK_C_BARS_TEAM, r);
        }
    }
    const double dd = (double)dead;
    const int64_t steam = st[OBS_BK_UNIT_TEAM];
    for (int t = 0; t < E.n_match; ++t) {
        double delta;
        if (zomb) delta = mul(E.zb, r);
        else if (steam == (int64_t)t) delta = sub(mul(-E.mb, r), mul(E.ma, dd));
        else delta = add(mul(E.mb, r), mul(E.ma, dd));
        double covered = 0.0;
        if (mv >= 0 && mvt == t) covered = add(covered, sub(mul(-E.mb, r), mul(E.ma, dd)));
        if (mav >= 0 && mat == t) {
            if (zomb) covered = add(covered, mul(E.zb, r));
            else if (enemy) covered = add(covered, add(mul(E.mb, r), mul(E.ma, dd)));
        }
        fadd(E.blk + E.L.rest + t, sub(delta, covered));
    }
}

// reward_v1._contrib
__device__ __forceinline__ double contrib(const Env& E, double hl, bool al, int64_t team, bool zomb,
                                          int t) {
    using namespace obs_f64;
    if (!al) return 0.0;
    if (zomb) return mul(-E.zb, hl);
    if (team == (int64_t)t) return add(mul(E.mb, hl), E.ma);
    return sub(mul(-E.mb, hl), E.ma);
}

// The standing (reward_v1.standing): per team H, A in the order a team first
// appears among the living non-zombies (the dict's insertion order), the
// zombies' H. Faults a non-zombie unit off the layout's marine teams.
struct Standing {
    double H[OBS_BK_TMAX];
    int64_t A[OBS_BK_TMAX];
    int ord[OBS_BK_TMAX];
    int n;
    double hz;
};

__device__ void standing_now(const Env& E, Standing* P, int64_t* fault) {
    for (int t = 0; t < OBS_BK_TMAX; ++t) { P->H[t] = 0.0; P->A[t] = 0; P->ord[t] = 0; }
    P->n = 0;
    P->hz = 0.0;
    for (int r = 0; r < E.U; ++r) {
        const int64_t* u = E.rows + (size_t)r * UNIT_STORE_W;
        if (!u[UNIT_COL_VALID] || !alive(u)) continue;
        const double x = bar_ratio(u[UNIT_COL_CURRENT_HP], bar_now(E, r));
        if (u[UNIT_COL_IS_ZOMBIE]) {
            P->hz = obs_f64::add(P->hz, x);
            continue;
        }
        const int64_t t = u[UNIT_COL_TEAM];
        if (t < 0 || t >= E.T) {
            *fault |= OBS_BK_FAULT_TEAM;
            continue;
        }
        bool seen = false;
        for (int i = 0; i < P->n; ++i) seen = seen || P->ord[i] == (int)t;
        if (!seen) P->ord[P->n++] = (int)t;
        P->H[t] = obs_f64::add(P->H[t], x);   // H.get(t, 0.0) + r
        P->A[t] += 1;
    }
}

__device__ void standing_load(const Env& E, Standing* P) {
    for (int t = 0; t < OBS_BK_TMAX; ++t) { P->H[t] = 0.0; P->A[t] = 0; P->ord[t] = 0; }
    for (int t = 0; t < E.T; ++t) {
        P->H[t] = dv(E.blk[E.L.h + t]);
        P->A[t] = E.blk[E.L.a + t];
        P->ord[t] = (int)E.blk[E.L.ord + t];
    }
    P->n = (int)E.blk[E.L.ord_n];
    P->hz = dv(E.blk[E.L.hz]);
}

__device__ void standing_store(const Env& E, const Standing& P) {
    for (int t = 0; t < E.T; ++t) {
        E.blk[E.L.h + t] = db(P.H[t]);
        E.blk[E.L.a + t] = P.A[t];
        E.blk[E.L.ord + t] = t < P.n ? P.ord[t] : 0;
    }
    E.blk[E.L.ord_n] = P.n;
    E.blk[E.L.hz] = db(P.hz);
}

__device__ __forceinline__ bool has(const Standing& P, int t) {
    for (int i = 0; i < P.n; ++i)
        if (P.ord[i] == t) return true;
    return false;
}
__device__ __forceinline__ double h_of(const Standing& P, int t) {
    return has(P, t) ? P.H[t] : 0.0;
}
__device__ __forceinline__ int64_t a_of(const Standing& P, int t) {
    return has(P, t) ? P.A[t] : 0;
}

// sum(H.values()) in dict order (Python's sum: 0 + v1 + v2 ...)
__device__ __forceinline__ double h_all(const Standing& P) {
    double s = 0.0;
    for (int i = 0; i < P.n; ++i) s = obs_f64::add(s, P.H[P.ord[i]]);
    return s;
}
__device__ __forceinline__ int64_t a_all(const Standing& P) {
    int64_t s = 0;
    for (int i = 0; i < P.n; ++i) s += P.A[P.ord[i]];
    return s;
}

// reward_v1.balance: mb (H_t - H_en) + ma (A_t - A_en) - zb H_zombie
__device__ double balance(const Env& E, const Standing& P, int t) {
    using namespace obs_f64;
    double h_en = 0.0;
    int64_t a_en = 0;
    for (int i = 0; i < P.n; ++i) {
        const int k = P.ord[i];
        if (k != t) {
            h_en = add(h_en, P.H[k]);
            a_en += P.A[k];
        }
    }
    return sub(add(mul(E.mb, sub(h_of(P, t), h_en)), mul(E.ma, (double)(a_of(P, t) - a_en))),
               mul(E.zb, P.hz));
}

// The living marine teams (MatchRules.living_teams) as a bit set.
__device__ int64_t living_now(const Env& E) {
    int64_t m = 0;
    for (int r = 0; r < E.U; ++r) {
        const int64_t* u = E.rows + (size_t)r * UNIT_STORE_W;
        if (!u[UNIT_COL_VALID] || !alive(u) || u[UNIT_COL_IS_ZOMBIE]) continue;
        const int64_t t = u[UNIT_COL_TEAM];
        if (t >= 0 && t < OBS_BK_TMAX) m |= (int64_t)1 << t;
    }
    return m;
}

__device__ __forceinline__ int popcount64(int64_t m) { return __popcll((unsigned long long)m); }

// Every roster row of team t (rows[t]).
template <class F>
__device__ __forceinline__ void for_rows(const Env& E, int t, F f) {
    for (int s = 0; s < E.S; ++s) {
        const int k = t * E.S + s;
        if (k < E.R && E.rmap[k] >= 0) f(k);
    }
}

__device__ __forceinline__ int n_roster(const Env& E, int t) {
    int n = 0;
    for_rows(E, t, [&](int) { ++n; });
    return n;
}

// RewardBooks._flush
__device__ void flush(const Env& E) {
    int64_t* scr = E.scr();
    int64_t* cnt = E.blk + E.L.cnt;
    int64_t* raw = E.blk + E.L.raw;
    for (int k = 0; k < E.R; ++k) {
        const bool live = E.blk[E.L.ended + k] == 0;
        for (int c = 0; c < OBS_BK_NB; ++c) {
            const int64_t i = (int64_t)k * OBS_BK_NB + c;
            fadd(raw + i, dv(scr[i]));
            if (live) fadd(cnt + i, dv(scr[i]));
        }
    }
    for (int k = 0; k < E.R; ++k) {
        if (E.blk[E.L.pend + k] && !E.blk[E.L.ended + k]) {
            E.blk[E.L.ended + k] = 1;
            E.blk[E.L.rows_ended + k] = 1;
        }
        E.blk[E.L.pend + k] = 0;
    }
    for (int64_t i = 0; i < (int64_t)E.R * OBS_BK_NB; ++i) scr[i] = 0;   // +0.0
}

// RewardBooks._eliminations
__device__ void eliminations(const Env& E, int64_t now) {
    const int64_t was = E.blk[E.L.living];
    int64_t elim = E.blk[E.L.elim];
    for (int t = 0; t < E.T; ++t) {
        const int64_t bit = (int64_t)1 << t;
        if (!(was & bit) || (now & bit)) continue;
        if ((elim & bit) || t >= E.n_match) continue;
        elim |= bit;
        for_rows(E, t, [&](int k) {
            fadd(E.scr() + (int64_t)k * OBS_BK_NB + OBS_BK_C_ELIMINATED, 1.0);
            E.blk[E.L.pend + k] = 1;
        });
        fadd(E.blk + E.L.outc + t, E.term(OBS_BK_TERM_ELIMINATED));
    }
    E.blk[E.L.elim] = elim;
    E.blk[E.L.living] = now;
}

// The snapshot of store row r from the store and the picture's bar now.
__device__ __forceinline__ void take_snapshot(const Env& E, int r) {
    const int64_t* u = E.rows + (size_t)r * UNIT_STORE_W;
    int64_t* s = E.snap(r);
    s[OBS_BK_UNIT_HP_Q] = u[UNIT_COL_CURRENT_HP];
    s[OBS_BK_UNIT_ALIVE] = alive(u) ? 1 : 0;
    s[OBS_BK_UNIT_TEAM] = u[UNIT_COL_TEAM];
    s[OBS_BK_UNIT_ZOMBIE] = u[UNIT_COL_IS_ZOMBIE] ? 1 : 0;
    s[OBS_BK_UNIT_BAR_Q] = bar_now(E, r);
}

__global__ void k_books(const int64_t* store, int U, const int64_t* ring, const int64_t* ring_count,
                        int ring_cap, const int32_t* rowmap, int R, int S, int T,
                        const int64_t* otab, const int64_t* trow, const int64_t* params,
                        uint8_t* ended, const int64_t* pic_state, int pic_w, int64_t* state, int W,
                        float* reward, float* counters, float* team_balance,
                        float* team_uncredited, float* team_outcome, uint8_t* rows_ended,
                        uint8_t* terminated, uint8_t* truncated, int64_t* words, int n_env,
                        int first, int brk) {
    using namespace obs_f64;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_env) return;
    Env E;
    E.blk = state + (size_t)e * W;
    E.L = bk_layout(T, U, R);
    E.rows = store + (size_t)e * U * UNIT_STORE_W;
    E.rmap = rowmap + (size_t)e * R;
    E.ot = otab + (size_t)e * OBS_TAB_W;
    E.pic = pic_state + (size_t)e * pic_w;
    E.PL = obs_pic::pic_layout(T, U);
    E.U = U; E.R = R; E.S = S; E.T = T;
    int64_t* sc = E.blk + E.L.sc;
    E.n_match = (int)sc[OBS_BK_S_N_MATCH];
    E.mb = E.term(OBS_BK_TERM_MARINE_BAR);
    E.ma = E.term(OBS_BK_TERM_MARINE_ALIVE);
    E.zb = E.term(OBS_BK_TERM_ZOMBIE_BAR);
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    int64_t fault = 0, mismatch = 0, ended_now = 0;
    if (first) {
        // the constructor: the snapshot, the standing, the living teams
        for (int r = 0; r < U; ++r)
            if (E.rows[(size_t)r * UNIT_STORE_W + UNIT_COL_VALID]) take_snapshot(E, r);
        Standing P;
        standing_now(E, &P, &fault);
        standing_store(E, P);
        E.blk[E.L.living] = living_now(E);
        sc[OBS_BK_S_TICK] = tick;
        words[e] = 0;
        words[n_env + e] = fault;
        words[2 * n_env + e] = 0;
        return;
    }
    // a finished env's books are frozen (THE BREAK bk_freeze: not frozen)
    if ((ended[e] || sc[OBS_BK_S_FINISHED]) && brk != OBS_BK_BREAK_BK_FREEZE) {
        words[e] = 0;
        words[n_env + e] = 0;
        words[2 * n_env + e] = 0;
        return;
    }
    if (sc[OBS_BK_S_TICK] >= 0 && tick != sc[OBS_BK_S_TICK] + 1) fault |= OBS_BK_FAULT_SKIPPED;
    const int64_t t0 = params[(size_t)e * OBS_PARAM_W + OBS_PARAM_EPISODE_TICK];
    const int64_t dec = params[(size_t)e * OBS_PARAM_W + OBS_PARAM_DECIMATION];
    // -- start_step: the first tick of a step zeroes the step's accumulators
    if (dec > 0 && (tick - 1 - t0) % dec == 0) {
        for (int64_t i = 0; i < (int64_t)R * OBS_BK_NB; ++i) {
            E.blk[E.L.cnt + i] = 0;
            E.blk[E.L.raw + i] = 0;
        }
        for (int64_t i = 0; i < (int64_t)R * OBS_BK_K0; ++i) E.blk[E.L.old + i] = 0;
        for (int t = 0; t < T; ++t) {
            E.blk[E.L.rest + t] = 0;
            E.blk[E.L.dbal + t] = 0;
            E.blk[E.L.outc + t] = 0;
        }
        for (int k = 0; k < R; ++k) E.blk[E.L.rows_ended + k] = 0;
    }
    using namespace unit_events;
    const int64_t n_ev = ring_count[e] < ring_cap ? ring_count[e] : ring_cap;
    const int64_t* evs = ring + (size_t)e * ring_cap * UNIT_EVENT_F;
    // -- the old counters (rl_env.count_events; the rows ended before the tick
    // keep their values: count_events' frozen)
    for (int64_t k = 0; k < n_ev; ++k) {
        const int64_t* ev = evs + k * UNIT_EVENT_F;
        const int64_t kind = ev[UNIT_EV_KIND];
        const int64_t v = ev[UNIT_EV_UNIT_ID], a = ev[UNIT_EV_ATTACKER_ID];
        const int vs = row_of(E, v);
        // row(uid): its roster row while on the row's team at the tick's start
        int vr = book_row(E, vs);
        if (vr >= 0 && E.snap(vs)[OBS_BK_UNIT_TEAM] != (int64_t)(vr / S)) vr = -1;
        int ar = -1, as = -1;
        if (a >= 0 && a != v) {
            as = row_of(E, a);
            ar = book_row(E, as);
            if (ar >= 0 && E.snap(as)[OBS_BK_UNIT_TEAM] != (int64_t)(ar / S)) ar = -1;
        }
        const bool same = vs >= 0 && as >= 0 &&
                          E.snap(vs)[OBS_BK_UNIT_TEAM] == E.snap(as)[OBS_BK_UNIT_TEAM];
        if (vr >= 0 && E.blk[E.L.ended + vr]) vr = -1;
        if (ar >= 0 && E.blk[E.L.ended + ar]) ar = -1;
        int64_t* old = E.blk + E.L.old;
        if (kind == UNIT_EVKIND_HIT) {
            const double dmg = (double)ev[UNIT_EV_DAMAGE_Q] / 65536.0;
            if (dmg <= 0.0) continue;
            if (vr >= 0) {
                f32add(old + (int64_t)vr * OBS_BK_K0 + OBS_BK_OLD_HITS_TAKEN, 1.0);
                f32add(old + (int64_t)vr * OBS_BK_K0 + OBS_BK_OLD_DAMAGE_TAKEN, dmg);
            }
            if (ar >= 0) {
                if (same) {
                    f32add(old + (int64_t)ar * OBS_BK_K0 + OBS_BK_OLD_TEAM_HITS, 1.0);
                } else {
                    f32add(old + (int64_t)ar * OBS_BK_K0 + OBS_BK_OLD_HITS_DEALT, 1.0);
                    f32add(old + (int64_t)ar * OBS_BK_K0 + OBS_BK_OLD_DAMAGE_DEALT, dmg);
                }
            }
        } else if (kind == UNIT_EVKIND_KILLED) {
            const int64_t c = ev[UNIT_EV_CAUSE];
            if (vr >= 0 && c >= 0 && c < OBS_BK_N_CAUSES)
                f32add(old + (int64_t)vr * OBS_BK_K0 + OBS_BK_OLD_DEATHS0 + c, 1.0);
            if (ar >= 0)
                f32add(old + (int64_t)ar * OBS_BK_K0 +
                           (same ? OBS_BK_OLD_TEAM_KILLS : OBS_BK_OLD_KILLS), 1.0);
        }
    }
    // -- end(): the walk over the tick's events (THE BREAK bk_order: reversed)
    int64_t* wh = E.blk + E.L.work_h;
    int64_t* wal = E.blk + E.L.work_al;
    for (int r = 0; r < U; ++r) {
        wh[r] = E.snap(r)[OBS_BK_UNIT_HP_Q];
        wal[r] = E.snap(r)[OBS_BK_UNIT_ALIVE];
    }
    for (int64_t k0 = 0; k0 < n_ev; ++k0) {
        const int64_t k = brk == OBS_BK_BREAK_BK_ORDER ? n_ev - 1 - k0 : k0;
        const int64_t* ev = evs + k * UNIT_EVENT_F;
        const int64_t kind = ev[UNIT_EV_KIND];
        const int vr = row_of(E, ev[UNIT_EV_UNIT_ID]);
        if (vr < 0) continue;
        const int64_t bar = E.snap(vr)[OBS_BK_UNIT_BAR_Q];
        if (kind == UNIT_EVKIND_HIT) {
            const int64_t dq = ev[UNIT_EV_DAMAGE_Q];
            if (dq == 0) continue;
            const double before = health(wh[vr], wal[vr] != 0, bar);
            wh[vr] -= dq;
            const double r = sub(before, health(wh[vr], wal[vr] != 0, bar));
            settle(E, vr, r, 0, dq > 0 ? ev[UNIT_EV_ATTACKER_ID] : -1, true);
        } else if (kind == UNIT_EVKIND_KILLED) {
            if (!wal[vr]) continue;
            const double r = health(wh[vr], true, bar);
            wal[vr] = 0;
            settle(E, vr, r, 1, ev[UNIT_EV_ATTACKER_ID], false);
        }
    }
    // -- the turnings and the check, per unit (store row order); the snapshot
    for (int r = 0; r < U; ++r) {
        const int64_t* u = E.rows + (size_t)r * UNIT_STORE_W;
        if (!u[UNIT_COL_VALID]) continue;
        int64_t* st = E.snap(r);
        const bool al_old = wal[r] != 0, al_new = alive(u);
        const int64_t team_new = u[UNIT_COL_TEAM];
        const bool z_new = u[UNIT_COL_IS_ZOMBIE] != 0;
        if (team_new != st[OBS_BK_UNIT_TEAM] || z_new != (st[OBS_BK_UNIT_ZOMBIE] != 0)) {
            const double old_h = health(wh[r], al_old, st[OBS_BK_UNIT_BAR_Q]);
            const double new_h = health(u[UNIT_COL_CURRENT_HP], al_new, bar_now(E, r));
            for (int t = 0; t < E.n_match; ++t)
                fadd(E.blk + E.L.rest + t,
                     sub(contrib(E, new_h, al_new, team_new, z_new, t),
                         contrib(E, old_h, al_old, st[OBS_BK_UNIT_TEAM],
                                 st[OBS_BK_UNIT_ZOMBIE] != 0, t)));
        } else if ((wh[r] != u[UNIT_COL_CURRENT_HP] && (al_old || al_new)) || al_old != al_new) {
            ++mismatch;
        }
    }
    for (int r = 0; r < U; ++r)
        if (E.rows[(size_t)r * UNIT_STORE_W + UNIT_COL_VALID]) take_snapshot(E, r);
    // -- the team part: the change itself, from state, over the roster
    {
        Standing P0, P1;
        standing_load(E, &P0);
        standing_now(E, &P1, &fault);
        const double ho = h_all(P0), hn = h_all(P1);
        const int64_t ao = a_all(P0), an = a_all(P1);
        const double dz = sub(P1.hz, P0.hz);
        for (int t = 0; t < E.n_match; ++t) {
            fadd(E.blk + E.L.dbal + t, sub(balance(E, P1, t), balance(E, P0, t)));
            const int n = n_roster(E, t);
            if (n == 0) continue;
            const double nd = (double)n;
            const double dh = sub(h_of(P1, t), h_of(P0, t));
            const int64_t da = a_of(P1, t) - a_of(P0, t);
            const double dhe = sub(sub(hn, h_of(P1, t)), sub(ho, h_of(P0, t)));
            const int64_t dae = (an - a_of(P1, t)) - (ao - a_of(P0, t));
            const double x1 = div(dh, nd), x2 = div((double)da, nd), x3 = div(dhe, nd),
                         x4 = div((double)dae, nd), x5 = div(dz, nd);
            for_rows(E, t, [&](int k) {
                int64_t* s = E.scr() + (int64_t)k * OBS_BK_NB;
                fadd(s + OBS_BK_C_SHARE_BARS, x1);
                fadd(s + OBS_BK_C_SHARE_ALIVE, x2);
                fadd(s + OBS_BK_C_SHARE_ENEMY_BARS, x3);
                fadd(s + OBS_BK_C_SHARE_ENEMY_ALIVE, x4);
                fadd(s + OBS_BK_C_SHARE_ZOMBIE_BARS, x5);
            });
        }
        standing_store(E, P1);
    }
    const int64_t living = living_now(E);
    eliminations(E, living);
    flush(E);
    // -- the end of the match (MatchRules.is_terminal) and finish()
    bool terminal = popcount64(living) < 2 || tick >= E.ot[OBS_TAB_LIMIT_TICKS];
    if (terminal && brk == OBS_BK_BREAK_BK_LATE && !sc[OBS_BK_S_TERM_SEEN]) {
        sc[OBS_BK_S_TERM_SEEN] = 1;                // THE BREAK: one tick late
        terminal = false;
    }
    if (terminal) {
        const bool win = popcount64(living) == 1;
        sc[OBS_BK_S_RES_OUTCOME] = win ? OBS_BK_OUTCOME_WIN : OBS_BK_OUTCOME_DRAW;
        sc[OBS_BK_S_RES_TEAM] = win ? (int64_t)(__ffsll((long long)living) - 1) : -1;
        sc[OBS_BK_S_RES_TICKS] = tick;
        if (win) {
            const int t = (int)sc[OBS_BK_S_RES_TEAM];
            if (t < E.n_match) {
                for_rows(E, t, [&](int k) {
                    fadd(E.scr() + (int64_t)k * OBS_BK_NB + OBS_BK_C_WON, 1.0);
                });
                fadd(E.blk + E.L.outc + t, E.term(OBS_BK_TERM_WIN));
            }
        } else if (popcount64(living) >= 2) {
            for (int t = 0; t < E.T; ++t) {
                if (!(living >> t & 1) || t >= E.n_match) continue;
                for_rows(E, t, [&](int k) {
                    fadd(E.scr() + (int64_t)k * OBS_BK_NB + OBS_BK_C_CLOCK, 1.0);
                });
                fadd(E.blk + E.L.outc + t, E.term(OBS_BK_TERM_CLOCK));
            }
        }
        for (int t = 0; t < E.n_match; ++t)
            for_rows(E, t, [&](int k) { E.blk[E.L.pend + k] = 1; });
        flush(E);
        sc[OBS_BK_S_FINISHED] = 1;
        sc[OBS_BK_S_ENDED_TICK] = tick;
        sc[OBS_BK_S_TERMINATED] = 1;               // a named profile: every end
        sc[OBS_BK_S_TRUNCATED] = 0;
        ended[e] = 1;                               // THE ENDED LATCH
        ended_now = 1;
    }
    // -- the served arrays, on a decision tick or the ending tick
    const bool decision = dec > 0 && (tick - t0) % dec == 0;
    if (decision || terminal) {
        const int64_t* G = E.ot + OBS_TAB_BOOK_G;
        for (int k = 0; k < R; ++k) {
            const size_t ek = (size_t)e * R + k;
            float* c = counters + ek * OBS_BK_K;
            double cv[OBS_BK_K];
            for (int j = 0; j < OBS_BK_K0; ++j) {
                const float f = f32of(E.blk[E.L.old + (int64_t)k * OBS_BK_K0 + j]);
                cv[j] = (double)f;
                c[j] = f;
            }
            for (int j = 0; j < OBS_BK_NB; ++j) {
                cv[OBS_BK_K0 + j] = dv(E.blk[E.L.cnt + (int64_t)k * OBS_BK_NB + j]);
                c[OBS_BK_K0 + j] = to_f32(cv[OBS_BK_K0 + j]);
            }
            // (counters[:, :, None] * G[None]).sum(axis=1): sequential from +0.0
            for (int g = 0; g < OBS_BK_GROUPS; ++g) {
                double acc = 0.0;
                for (int j = 0; j < OBS_BK_K; ++j)
                    acc = add(acc, mul(cv[j], dv(G[(int64_t)j * OBS_BK_GROUPS + g])));
                reward[ek * OBS_BK_GROUPS + g] = to_f32(acc);
            }
            rows_ended[ek] = E.blk[E.L.rows_ended + k] ? 1 : 0;
        }
        for (int t = 0; t < T; ++t) {
            team_balance[(size_t)e * T + t] = to_f32(dv(E.blk[E.L.dbal + t]));
            team_uncredited[(size_t)e * T + t] = to_f32(dv(E.blk[E.L.rest + t]));
            team_outcome[(size_t)e * T + t] = to_f32(dv(E.blk[E.L.outc + t]));
        }
        terminated[e] = terminal ? 1 : 0;
        truncated[e] = 0;
    }
    sc[OBS_BK_S_TICK] = tick;
    words[e] = mismatch;
    words[n_env + e] = fault;
    words[2 * n_env + e] = ended_now;
}

}  // namespace

bool obs_books(int n_env, std::uintptr_t store, int n_units, int store_w, UnitEventRing& ring,
               std::uintptr_t rowmap, int n_rows, int n_slots, int n_teams,
               std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
               std::uintptr_t params, std::uintptr_t ended, std::uintptr_t pic_state, int pic_w,
               std::uintptr_t state, int state_w, std::uintptr_t reward, std::uintptr_t counters,
               std::uintptr_t team_balance, std::uintptr_t team_uncredited,
               std::uintptr_t team_outcome, std::uintptr_t rows_ended, std::uintptr_t terminated,
               std::uintptr_t truncated, std::uintptr_t words, bool first, int brk) {
    const char* who = "obs_books";
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && n_teams >= 1, who,
         "a non-positive size");
    need(n_teams <= OBS_BK_TMAX, who, "more marine teams than the books hold (OBS_BK_TMAX)");
    need(n_rows == n_teams * n_slots, who, "n_rows is not n_teams x n_slots");
    need(store_w == UNIT_STORE_W, who, "the store's width is not UNIT_STORE_W");
    need(table_w == OBS_TAB_W, who, "the obs table's width is not OBS_TAB_W");
    need(pic_w == obs_pic::pic_layout(n_teams, n_units).w, who, "the picture block's width");
    need(state_w == bk_layout(n_teams, n_units, n_rows).w, who,
         "the books block's width is not the compiled layout's");
    need(store && rowmap && obs_table && tick_row && params && ended && pic_state && state &&
             reward && counters && team_balance && team_uncredited && team_outcome &&
             rows_ended && terminated && truncated && words, who, "a null device pointer");
    need(ring.n_env() == n_env && ring.width() == unit_events::UNIT_EVENT_F, who,
         "the event ring's shape");
    cudaStream_t s = tstream();
    int64_t* wd = reinterpret_cast<int64_t*>(words);
    k_books<<<(n_env + 63) / 64, 64, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, ring.records_dev(), ring.count_dev(),
        ring.cap(), reinterpret_cast<const int32_t*>(rowmap), n_rows, n_slots, n_teams,
        reinterpret_cast<const int64_t*>(obs_table), reinterpret_cast<const int64_t*>(tick_row),
        reinterpret_cast<const int64_t*>(params), reinterpret_cast<uint8_t*>(ended),
        reinterpret_cast<const int64_t*>(pic_state), pic_w, reinterpret_cast<int64_t*>(state),
        state_w, reinterpret_cast<float*>(reward), reinterpret_cast<float*>(counters),
        reinterpret_cast<float*>(team_balance), reinterpret_cast<float*>(team_uncredited),
        reinterpret_cast<float*>(team_outcome), reinterpret_cast<uint8_t*>(rows_ended),
        reinterpret_cast<uint8_t*>(terminated), reinterpret_cast<uint8_t*>(truncated), wd, n_env,
        first ? 1 : 0, brk);
    launched(who);
    // the words ride the conductor's one books read; outside a conductor tick
    // the caller reads them (the build)
    return books_defer("reward_books", wd, (size_t)3 * n_env);
}

std::vector<std::pair<std::string, int64_t>> obs_books_layout(int n_teams, int n_units,
                                                              int n_rows) {
    const BkLayout L = bk_layout(n_teams, n_units, n_rows);
    return {{"UNIT", L.unit}, {"H", L.h}, {"A", L.a}, {"HZ", L.hz}, {"ORD", L.ord},
            {"ORD_N", L.ord_n}, {"LIVING", L.living}, {"ELIM", L.elim}, {"ENDED", L.ended},
            {"PEND", L.pend}, {"ROWS_ENDED", L.rows_ended}, {"SCR", L.scr}, {"CNT", L.cnt},
            {"RAW", L.raw}, {"OLD", L.old}, {"REST", L.rest}, {"DBAL", L.dbal},
            {"OUTC", L.outc}, {"WORK_H", L.work_h}, {"WORK_AL", L.work_al}, {"SC", L.sc},
            {"W", L.w}};
}

}  // namespace breach_cuda
