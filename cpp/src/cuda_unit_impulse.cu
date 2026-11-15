// ============================================================================
// #29 P6.2 -- THE UNIT IMPULSE LAW'S TWO SLOTS on the device unit store:
// slot 1a (integrate + turn) and slot 9c2 (the wind's impulse). See
// cuda_unit_kernels.h; residency design §4.4b.
//
// Slot 1a is src/simulation/unit_impulse.py's integrate_all integer for
// integer: per valid LIVING row, unit_impulse.integrate (the accumulators into
// dv / dw with the remainders kept, the V_MAX clamp, the demand on the
// standing hold and the effort's low-pass, the prone branch or the sway and
// its overflow into pv, the resist, the knockdown checks through the status
// block's apply), then turning.step (the live aim the target while a
// non-zero one is held; omega += dw; the turning law). The hooks it reads:
// TURN (the turning dials, Simulation.unit_turn_rate_q16 / _accel_q16) and
// HOLD (the stability, unit_impulse.stability_q16) -- through the library's
// scaled().
//
// Slot 9c2 is exchange.apply_wind_force: per valid LIVING row,
// GameMap.footprint_wind over the gas / wind / solid planes (the max over the
// footprint's gas tiles of N_raw * ((vx^2 + vy^2) >> 16), the first such
// tile row-major on a tie) and unit_impulse.add_wind (the impulse
// ((W >> 16) * k) >> 16 along that tile's wind; prone: the prone area's, and
// the standing area's surplus into stand_x / stand_y).
//
// What has NO device form (reference-only, never read by a law): the
// momentum books (Unit.impulse_books, unit_impulse.books_identity's
// diagnostic) and the render-only StaggerEvent the sway's overflow appends.
//
// WIDTHS. Python's ints are unbounded; here every product is either proved to
// fit int64 from the tables' doors (unit_tables.ROWS: velocities <= 2^22,
// stability <= 2^20 Q16, hooks <= 8 ONE, n_rec <= 256) or taken in 128 bits
// (the wind load W = N * v2 < 2^80, its impulse ((W >> 16) * k) >> 16; the
// projection j * v / n by the exact split j = q n + r). A state operand that
// would enter a square past 2^31 (a velocity of 2^15 tiles a tick -- no game
// reaches it) is a FAULT: the row is left as it was and the driver refuses
// the kernel's result, naming the slot.
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

//: unit_impulse.V_MAX_Q / EFFORT_MAX.
constexpr int64_t V_MAX_Q = 4 * ONE;
constexpr int64_t EFFORT_MAX = 2 * ONE;
//: A demand component past this saturates the effort whatever the rest (the
//: hold is at most 2^22, so isqrt * ONE / cap >= 2^24 > EFFORT_MAX): the
//: component is clamped to it before its square, exactly.
constexpr int64_t DEMAND_COMPONENT_MAX = (int64_t)1 << 30;

// unit_impulse._clamp_vmax: (x, y) scaled into V_MAX_Q.
__device__ inline void clamp_vmax(int64_t& x, int64_t& y, bool* bad) {
    const int64_t m2 = sq2(x, y, bad);
    if (*bad || m2 <= V_MAX_Q * V_MAX_Q) return;
    const int64_t n = fixedpoint::isqrt_ceil_q(m2);
    x = (x * V_MAX_Q) / n;
    y = (y * V_MAX_Q) / n;
}

// unit_impulse._shrink: (x, y) shortened toward zero by b in magnitude;
// returns the magnitude used, min(|v|, b).
__device__ inline int64_t shrink(int64_t& x, int64_t& y, int64_t b, bool* bad) {
    const int64_t m2 = sq2(x, y, bad);
    if (*bad) return 0;
    const int64_t n = fixedpoint::isqrt_q(m2);
    if (n <= b) {
        x = 0;
        y = 0;
        return n;
    }
    x = (x * (n - b)) / n;
    y = (y * (n - b)) / n;
    return b;
}

// unit_impulse._blend / _limit: a direction limit from (fwd, side, back) at
// the cosine of (vx, vy) to the facing.
__device__ inline int64_t limit3(const int64_t* v3, int64_t vx, int64_t vy, int64_t facing_q,
                                 bool* bad) {
    const int64_t c = facing_cos_q16(vx, vy, facing_q, bad);
    const int64_t fwd = v3[0], side = v3[1], back = v3[2];
    if (c >= 0) return side + (((fwd - side) * c) >> 16);
    return side + (((side - back) * c) >> 16);
}

// ImpulseTable.hold_q: one tick of hold (standing, or prone friction), the
// zombie state's multiplier on is_zombie, at least 1.
__device__ inline int64_t hold_q(const int64_t* row, const int64_t* tab, bool prone) {
    int64_t cap = prone ? tab[UNIT_TAB_IMPULSE_CAP_PRONE_Q] : tab[UNIT_TAB_IMPULSE_CAP_Q];
    const int64_t zh = tab[UNIT_TAB_IMPULSE_ZOMBIE_HOLD_Q16];
    if (row[UNIT_COL_IS_ZOMBIE] && zh != ONE) cap = fixedpoint::shr_trunc_q(cap * zh, 16);
    return cap >= 1 ? cap : 1;
}

// unit_impulse.stability_q16: the class value (exchange._stability_for: the
// zombie state's, else the env profile's, else the species' environment)
// times the HOLD hook.
__device__ inline int64_t stability_q16(const int64_t* row, const int64_t* tab, bool* bad) {
    int64_t cls;
    if (row[UNIT_COL_IS_ZOMBIE]) {
        cls = tab[UNIT_TAB_STABILITY_ZOMBIE_Q16];
    } else {
        const int64_t p = row[UNIT_COL_ENV_PROFILE_NAME];
        if (p >= 0) {
            if (p >= UNIT_TAB_STABILITY_PROFILE_Q16_W) { *bad = true; return 0; }
            cls = tab[UNIT_TAB_STABILITY_PROFILE_Q16 + p];
        } else {
            const int64_t sp = row[UNIT_COL_SPECIES_ID];
            if (sp < 0 || sp >= UNIT_TAB_STABILITY_SPECIES_Q16_W) { *bad = true; return 0; }
            cls = tab[UNIT_TAB_STABILITY_SPECIES_Q16 + sp];
        }
    }
    return scaled(row, UNIT_HOOK_HOLD, cls);
}

// Slot 1a: one thread per unit row of one env.
__global__ void k_integrate_turn(int64_t* store, int n_units, const int64_t* tables,
                                 unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row) || !row_alive(row)) return;
    const int64_t* tab = env_tables(tables);
    bool bad = false;

    int64_t imp_x = row[UNIT_COL_IMP_X], imp_y = row[UNIT_COL_IMP_Y];
    int64_t imp_l = row[UNIT_COL_IMP_L];
    int64_t stand_x = row[UNIT_COL_STAND_X], stand_y = row[UNIT_COL_STAND_Y];
    int64_t swx = row[UNIT_COL_SWX], swy = row[UNIT_COL_SWY];
    int64_t pvx = row[UNIT_COL_PVX], pvy = row[UNIT_COL_PVY];
    int64_t effort = row[UNIT_COL_EFFORT], react = row[UNIT_COL_REACT_TICKS];
    const int64_t facing = row[UNIT_COL_FACING_Q];
    bool knock = false;
    int64_t dw = 0;

    // ---- unit_impulse.integrate -------------------------------------------
    const bool at_rest = !(imp_x || imp_y || imp_l || swx || swy || pvx || pvy || effort ||
                           react || stand_x || stand_y);
    if (!at_rest) {
        const int64_t m = row[UNIT_COL_MASS_G];
        const int64_t inertia = tab[UNIT_TAB_IMPULSE_INERTIA_G];
        if (m <= 0) bad = true;                    // Python: a division by zero
        int64_t dvx = 0, dvy = 0;
        if (!bad) {
            // 1. dv, dw -- the remainder stays in the accumulator.
            dvx = imp_x / m;
            dvy = imp_y / m;
            imp_x -= dvx * m;
            imp_y -= dvy * m;
            if (imp_l) {
                dw = imp_l / inertia;
                imp_l -= dw * inertia;
            }
            clamp_vmax(dvx, dvy, &bad);
        }
        // 2. the demand on the STANDING hold and the effort's low-pass.
        const int64_t cap = hold_q(row, tab, false);
        int64_t st = 0;
        bool prone = false;
        if (!bad) {
            int64_t ddx = dvx, ddy = dvy;
            if (stand_x || stand_y) {
                ddx += stand_x / m;
                ddy += stand_y / m;
                stand_x = 0;
                stand_y = 0;
            }
            int64_t demand = 0;
            if (ddx || ddy) {
                // Exact: past DEMAND_COMPONENT_MAX the demand saturates.
                const int64_t cx = ddx > DEMAND_COMPONENT_MAX ? DEMAND_COMPONENT_MAX
                                 : (ddx < -DEMAND_COMPONENT_MAX ? -DEMAND_COMPONENT_MAX : ddx);
                const int64_t cy = ddy > DEMAND_COMPONENT_MAX ? DEMAND_COMPONENT_MAX
                                 : (ddy < -DEMAND_COMPONENT_MAX ? -DEMAND_COMPONENT_MAX : ddy);
                demand = (fixedpoint::isqrt_q(cx * cx + cy * cy) * ONE) / cap;
                if (demand > EFFORT_MAX) demand = EFFORT_MAX;
            }
            effort += fixedpoint::effort_step_q(demand - effort);
            st = stability_q16(row, tab, &bad);
            prone = composed_flags(row, tab, false, &bad).is_prone;
        }
        if (!bad && prone) {
            // PRONE: no sway, no reaction delay -- every input meets the prone
            // friction at once and what is left moves the body.
            pvx += swx + dvx;
            pvy += swy + dvy;
            swx = 0;
            swy = 0;
            react = 0;
            clamp_vmax(pvx, pvy, &bad);
            if (!bad) {
                shrink(pvx, pvy, hold_q(row, tab, true), &bad);
                if (effort * ONE >= st * st || pvx || pvy) knock = true;
            }
        } else if (!bad) {
            // 3. the in-place sway; its overflow becomes the pushed velocity.
            swx += dvx;
            swy += dvy;
            if (swx || swy) {
                const int64_t lim = limit3(tab + UNIT_TAB_IMPULSE_V_INPLACE_Q, swx, swy,
                                           facing, &bad);
                const int64_t m2 = sq2(swx, swy, &bad);
                if (!bad && m2 > lim * lim) {
                    const int64_t n = fixedpoint::isqrt_ceil_q(m2);
                    const int64_t kx = (swx * lim) / n, ky = (swy * lim) / n;
                    pvx += swx - kx;
                    pvy += swy - ky;
                    swx = kx;
                    swy = ky;
                    react = tab[UNIT_TAB_IMPULSE_REACT_TICKS];
                }
            }
            if (!bad) clamp_vmax(pvx, pvy, &bad);
            // 4. resist with one tick of hold: the sway first, every tick; the
            //    rest of the budget works off pv once the delay has run.
            int64_t margin = 0;
            if (!bad) {
                const int64_t used = shrink(swx, swy, cap, &bad);
                margin = cap - used;
            }
            if (!bad) {
                if (react == 0) {
                    if ((pvx || pvy) && margin > 0) shrink(pvx, pvy, margin, &bad);
                } else {
                    react -= 1;
                }
            }
            // 6. the knockdown checks.
            if (!bad) {
                bool down = false;
                const int64_t vx = pvx + swx, vy = pvy + swy;
                if (vx || vy) {
                    const int64_t vf = (limit3(tab + UNIT_TAB_IMPULSE_V_FALL_Q, vx, vy, facing,
                                               &bad) * st) >> 16;
                    down = sq2(vx, vy, &bad) >= vf * vf;
                }
                if (!down) down = effort * ONE >= st * st;
                if (!down && react == 0 && (pvx || pvy)) {
                    const int64_t rr = tab[UNIT_TAB_IMPULSE_N_REC] * margin;
                    down = sq2(pvx, pvy, &bad) > rr * rr;
                }
                knock = down;
            }
        }
    }

    // ---- turning.step ---------------------------------------------------------
    int64_t target = row[UNIT_COL_FACING_TARGET_Q];
    int64_t facing_new = facing, omega_new = row[UNIT_COL_OMEGA_Q];
    if (!bad) {
        // turning.aim_target_q16: a living non-zombie's non-zero live aim.
        const int64_t* aim = row + UNIT_COL_LIVE_AIM;            // (present, dx, dy)
        if (!row[UNIT_COL_IS_ZOMBIE] && aim[0] && (aim[1] != 0 || aim[2] != 0)) {
            if (!fits_i32(-aim[2]) || !fits_i32(aim[1])) {
                bad = true;
            } else {
                target = normalize_q16(fixedpoint::atan2_q16((q16)(-aim[2]), (q16)aim[1]));
            }
        }
        const int64_t a = scaled(row, UNIT_HOOK_TURN, tab[UNIT_TAB_TURN_ACCEL_CLASS_Q16]);
        const int64_t w = scaled(row, UNIT_HOOK_TURN, tab[UNIT_TAB_TURN_RATE_CLASS_Q16]);
        if (!bad) turn_law_q16(facing, omega_new + dw, target, a, w, &facing_new, &omega_new,
                               &bad);
    }

    if (bad) {
        count_fault(cnt);
        return;                                   // the row as it was
    }
    bool changed = false;
    put(row, UNIT_COL_IMP_X, imp_x, changed);
    put(row, UNIT_COL_IMP_Y, imp_y, changed);
    put(row, UNIT_COL_IMP_L, imp_l, changed);
    put(row, UNIT_COL_STAND_X, stand_x, changed);
    put(row, UNIT_COL_STAND_Y, stand_y, changed);
    put(row, UNIT_COL_SWX, swx, changed);
    put(row, UNIT_COL_SWY, swy, changed);
    put(row, UNIT_COL_PVX, pvx, changed);
    put(row, UNIT_COL_PVY, pvy, changed);
    put(row, UNIT_COL_EFFORT, effort, changed);
    put(row, UNIT_COL_REACT_TICKS, react, changed);
    put(row, UNIT_COL_FACING_TARGET_Q, target, changed);
    put(row, UNIT_COL_FACING_Q, facing_new, changed);
    put(row, UNIT_COL_OMEGA_Q, omega_new, changed);
    if (knock) {
        // unit_impulse._knock_down: KNOCKED_DOWN, magnitude 0, the get-up
        // ticks, no source (a refresh kind: never refused).
        const int res = apply_status(row, tab, UNIT_STATUS_KNOCKED_DOWN, 0,
                                     tab[UNIT_TAB_IMPULSE_GETUP_TICKS], -1, changed);
        if (res == APPLY_REFUSED) count_refused(cnt);
        if (res == APPLY_BAD) count_fault(cnt);
    }
    if (changed) count_changed(cnt);
}

// ---- slot 9c2 ------------------------------------------------------------------

// The 128-bit wind load (hi, lo) as an unsigned pair.
struct U128 {
    uint64_t hi, lo;
};
__device__ __forceinline__ bool gt128(U128 a, U128 b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

// unit_impulse.wind_impulse_q: ((w >> 16) * k) >> 16 (0 for w <= 0 or k 0);
// w < 2^80, so w >> 16 fits a uint64; a result past int64 sets *bad.
__device__ inline int64_t wind_impulse_q(U128 w, int64_t k, bool* bad) {
    if ((w.hi == 0 && w.lo == 0) || k == 0) return 0;
    const uint64_t ws = (w.hi << 48) | (w.lo >> 16);
    if ((w.hi >> 16) != 0) { *bad = true; return 0; }
    uint64_t h, l;
    fixedpoint::umul64_wide(ws, (uint64_t)k, &h, &l);
    if ((h >> 15) != 0) { *bad = true; return 0; }   // the result >= 2^63
    return (int64_t)((h << 48) | (l >> 16));
}

// div_trunc(j * v, n) for j >= 0, |v| <= n, n > 0, exact without 128 bits:
// j = q n + r, so j v / n = q v + r v / n with r v < 2^63.
__device__ __forceinline__ int64_t project_trunc(int64_t j, int64_t v, int64_t n) {
    const int64_t q = j / n, rem = j % n;
    const int64_t av = v < 0 ? -v : v;
    const int64_t mag = q * av + (rem * av) / n;
    return v < 0 ? -mag : mag;
}

// Slot 9c2: one thread per unit row of one env.
__global__ void k_wind_impulse(int64_t* store, int n_units, const int64_t* tables,
                               const int32_t* gas, const int32_t* wind_x,
                               const int32_t* wind_y, const uint8_t* solid, int n_gases,
                               int h, int w, unsigned long long* cnt) {
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= n_units) return;
    int64_t* row = unit_row(store, n_units, r);
    if (!row_valid(row) || !row_alive(row)) return;
    const int64_t* tab = env_tables(tables);
    const int64_t k_stand = tab[UNIT_TAB_IMPULSE_WIND_K_Q];
    const int64_t k_prone = tab[UNIT_TAB_IMPULSE_WIND_K_PRONE_Q];
    if (k_stand == 0 && k_prone == 0) return;     // the input switched off
    const size_t plane = (size_t)h * w;
    const int32_t* g = bptr(gas, (size_t)n_gases * plane);
    const int32_t* n_o2 = g + (size_t)tab[UNIT_TAB_GAS_O2] * plane;
    const int32_t* n_n2 = g + (size_t)tab[UNIT_TAB_GAS_N2] * plane;
    const int32_t* vxp = bptr(wind_x, plane);
    const int32_t* vyp = bptr(wind_y, plane);
    const uint8_t* sol = bptr(solid, plane);

    // GameMap.footprint_wind at (tile_y, tile_x): int(y), int(x) truncated.
    const int64_t fy = row[UNIT_COL_Y_Q] / 65536, fx = row[UNIT_COL_X_Q] / 65536;
    const int64_t fp = row[UNIT_COL_FOOTPRINT];
    const int64_t y0 = fy > 0 ? fy : 0, x0 = fx > 0 ? fx : 0;
    const int64_t y1 = (fy + fp) < h ? (fy + fp) : h, x1 = (fx + fp) < w ? (fx + fp) : w;
    U128 best{0, 0};
    int64_t bvx = 0, bvy = 0;
    for (int64_t yy = y0; yy < y1; ++yy) {
        for (int64_t xx = x0; xx < x1; ++xx) {
            const size_t i = (size_t)yy * w + (size_t)xx;
            if (sol[i]) continue;
            const int64_t vx = vxp[i], vy = vyp[i];
            const uint64_t v2 = ((uint64_t)(vx * vx) + (uint64_t)(vy * vy)) >> 16;
            const int64_t nraw = (int64_t)n_o2[i] + (int64_t)n_n2[i];
            if (nraw <= 0 || v2 == 0) continue;        // w_i <= 0: never above best
            U128 wi;
            fixedpoint::umul64_wide((uint64_t)nraw, v2, &wi.hi, &wi.lo);
            if (gt128(wi, best)) {
                best = wi;
                bvx = vx;
                bvy = vy;
            }
        }
    }
    if (best.hi == 0 && best.lo == 0) return;     // w <= 0

    // unit_impulse.add_wind.
    bool bad = false;
    const bool prone = composed_flags(row, tab, false, &bad).is_prone;
    const int64_t n = (int64_t)fixedpoint::isqrt_u64_q((uint64_t)(bvx * bvx) +
                                                       (uint64_t)(bvy * bvy));
    int64_t stand_x = row[UNIT_COL_STAND_X], stand_y = row[UNIT_COL_STAND_Y];
    int64_t imp_x = row[UNIT_COL_IMP_X], imp_y = row[UNIT_COL_IMP_Y];
    if (n != 0 && !bad) {
        const int64_t j = wind_impulse_q(best, prone ? k_prone : k_stand, &bad);
        if (prone && !bad) {
            const int64_t js = wind_impulse_q(best, k_stand, &bad);
            if (!bad && js != j) {
                stand_x += project_trunc(js, bvx, n) - project_trunc(j, bvx, n);
                stand_y += project_trunc(js, bvy, n) - project_trunc(j, bvy, n);
            }
        }
        if (!bad && j != 0) {
            imp_x += project_trunc(j, bvx, n);
            imp_y += project_trunc(j, bvy, n);
        }
    }
    if (bad) {
        count_fault(cnt);
        return;
    }
    bool changed = false;
    put(row, UNIT_COL_STAND_X, stand_x, changed);
    put(row, UNIT_COL_STAND_Y, stand_y, changed);
    put(row, UNIT_COL_IMP_X, imp_x, changed);
    put(row, UNIT_COL_IMP_Y, imp_y, changed);
    if (changed) count_changed(cnt);
}

}  // namespace

void unit_integrate_turn(UnitStoreDevice& store, const UnitTablesDevice& tables,
                         const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_integrate_turn");
    store.begin_counts();
    k_integrate_turn<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(), store.counts_dev());
    unit_finish(store, out, "unit_integrate_turn");
}

void unit_wind_impulse(UnitStoreDevice& store, const UnitTablesDevice& tables,
                       const UnitFieldPlanes& planes, const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_wind_impulse");
    if (!planes.gas || !planes.wind_x || !planes.wind_y || !planes.solid ||
        planes.n_gases < 1 || planes.h < 1 || planes.w < 1)
        throw std::invalid_argument("unit_wind_impulse: the gas, wind_x, wind_y and solid "
                                    "planes and their shape are required");
    store.begin_counts();
    k_wind_impulse<<<bgrid(store.n_units(), store.n_env()), 128, 0, tstream()>>>(
        store.rows(), store.n_units(), tables.values(),
        reinterpret_cast<const int32_t*>(planes.gas),
        reinterpret_cast<const int32_t*>(planes.wind_x),
        reinterpret_cast<const int32_t*>(planes.wind_y),
        reinterpret_cast<const uint8_t*>(planes.solid), planes.n_gases, planes.h, planes.w,
        store.counts_dev());
    unit_finish(store, out, "unit_wind_impulse");
}

}  // namespace breach_cuda
