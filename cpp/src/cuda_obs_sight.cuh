#pragma once
// ==========================================================================
// #29 P8.2: THE OBSERVATION'S SIGHT ON THE CARD (Class T), the device
// functions every observation block that asks a line, who sees whom or a
// round's way calls. The observation's own header (never the unit library,
// cuda_unit_lib.cuh); it INCLUDES the one device sight function
// (cuda_unit_sight.cuh's unit_sight::has_los) and never walks a line itself.
//
//   line_clear(..., physical, y0, x0, y1, x1)   vision.rays_clear's question for
//                                               ONE pair: the sight form (solid or
//                                               the sight screen) or physical
//                                               (solid alone); the walk is
//                                               has_los's, not symmetric
//   window_cell(..., to_window, y, x, wy, wx,   vision.window_clear's cell k
//               ww, k)                          (row-major) of the window whose
//                                               top-left is (wy, wx): FROM (y, x)
//                                               to the cell, or (to_window) FROM
//                                               the cell to (y, x)
//   blocks_segment(cover, x0, y0, x1, y1)       CoverRuntime.blocks_segment: the
//                                               slab test in float64, the
//                                               reference's operation order,
//                                               never fused (the float kit)
//   solid_in_way(..., covers, n, ax, ay, bx,by) line_of_fire.solid_in_way: the
//                                               physical walk blocked, or a live
//                                               cover crosses the segment
//   can_see(observer row, target row, rule,     vision.can_see for two units of
//           cone cos, awareness radius, ...)    DIFFERENT teams, both rules
//
// Cover rows are the observation's cover table (device_obs.cover_values,
// columns OBS_COVER_* of the generated obs_layout.h): the floats carried as
// their own float64 bit patterns. Cover that blocks SIGHT (blocks_los) has no
// device form: device_obs refuses a level holding one at the build, so
// line_clear IS vision.rays_clear on every level the device serves.
//
// Bad operands (a coordinate past unit_sight::COORD_MAX) set *bad: the caller
// counts a fault, never an answer.
// ==========================================================================
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_obs_float.cuh"
#include "cuda_unit_sight.cuh"
#include "fixed_point.h"
#include "obs_layout.h"
#include "unit_store_layout.h"

namespace breach_cuda {
namespace obs_sight {

// Python's floor division by a positive divisor.
__device__ __forceinline__ int64_t floordiv(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// Unit.tile_x / tile_y: the Q16 position truncated toward zero.
__device__ __forceinline__ int64_t tile_of(int64_t q) {
    return q >= 0 ? (q >> 16) : -((-q) >> 16);
}

// Unit.center_tile_x / _y of a store row: tile + footprint // 2.
__device__ __forceinline__ int64_t centre_x(const int64_t* u) {
    return tile_of(u[unit_store::UNIT_COL_X_Q]) + floordiv(u[unit_store::UNIT_COL_FOOTPRINT], 2);
}
__device__ __forceinline__ int64_t centre_y(const int64_t* u) {
    return tile_of(u[unit_store::UNIT_COL_Y_Q]) + floordiv(u[unit_store::UNIT_COL_FOOTPRINT], 2);
}

// ONE line, one env's (h, w) planes (`screen` the sight screen; physical
// ignores it). The has_los walk of the device sight function.
__device__ __forceinline__ bool line_clear(const uint8_t* solid, const uint8_t* screen, int h,
                                           int w, bool physical, int64_t y0, int64_t x0,
                                           int64_t y1, int64_t x1, bool* bad) {
    return unit_sight::has_los(solid, physical ? nullptr : screen, h, w, y0, x0, y1, x1, bad);
}

// Cell k (row-major, ww wide) of the window whose top-left is (wy, wx).
__device__ __forceinline__ bool window_cell(const uint8_t* solid, const uint8_t* screen, int h,
                                            int w, bool physical, bool to_window, int64_t y,
                                            int64_t x, int64_t wy, int64_t wx, int64_t ww,
                                            int64_t k, bool* bad) {
    const int64_t ty = wy + k / ww, tx = wx + k % ww;
    return to_window ? line_clear(solid, screen, h, w, physical, ty, tx, y, x, bad)
                     : line_clear(solid, screen, h, w, physical, y, x, ty, tx, bad);
}

// CoverRuntime.blocks_segment on one cover row (x0, y0, x1, y1 the segment's
// end points as the reference hands them: the centre tiles, ints converted
// exactly). Plain IEEE: the per-axis d is the int difference, inv = 1.0 / d,
// the two products, the swaps and the clips, in the reference's order.
__device__ __forceinline__ bool blocks_segment(const int64_t* cov, int64_t sx0, int64_t sy0,
                                               int64_t sx1, int64_t sy1) {
    using namespace obs_layout;
    if (!cov[OBS_COVER_ALIVE]) return false;
    double t0 = 0.0, t1 = 1.0;
    for (int axis = 0; axis < 2; ++axis) {
        const double p = (double)(axis == 0 ? sx0 : sy0);
        const double d = (double)(axis == 0 ? sx1 - sx0 : sy1 - sy0);
        const double lo = obs_f64::of_bits(cov[axis == 0 ? OBS_COVER_X0_BITS : OBS_COVER_Y0_BITS]);
        const double hi = obs_f64::of_bits(cov[axis == 0 ? OBS_COVER_X1_BITS : OBS_COVER_Y1_BITS]);
        if (d == 0.0) {
            if (p < lo || p >= hi) return false;
            continue;
        }
        const double inv = obs_f64::div(1.0, d);
        double ta = obs_f64::mul(obs_f64::sub(lo, p), inv);
        double tb = obs_f64::mul(obs_f64::sub(hi, p), inv);
        if (ta > tb) {
            const double s = ta;
            ta = tb;
            tb = s;
        }
        if (ta > t0) t0 = ta;
        if (tb < t1) t1 = tb;
        if (t0 > t1) return false;
    }
    return true;
}

// line_of_fire.solid_in_way between the centre tiles a = (ax, ay) and
// b = (bx, by): the PHYSICAL walk (a grille does not stop a round) blocked,
// or EVERY live cover's slab test (a crate you can see over included), in
// the cover table's (ordinal) order.
__device__ __forceinline__ bool solid_in_way(const uint8_t* solid, int h, int w,
                                             const int64_t* covers, int n_cover, int64_t ax,
                                             int64_t ay, int64_t bx, int64_t by, bool* bad) {
    if (!unit_sight::has_los(solid, nullptr, h, w, ay, ax, by, bx, bad)) return true;
    for (int c = 0; c < n_cover; ++c)
        if (blocks_segment(covers + (size_t)c * obs_layout::OBS_COVER_W, ax, ay, bx, by))
            return true;
    return false;
}

// vision.within_cone with the facing as its Q16 (Unit.facing is facing_q /
// 65536.0 exactly, so unit_fixed.quantize_scalar gives facing_q back and
// facing_vector is (cos_q16(f) / 65536.0, -(sin_q16(f) / 65536.0))): IEEE
// basic operations only, never fused:
//   d2 = dx*dx + dy*dy (int); d2 <= 0 -> inside;
//   dot = (dx*fx + dy*fy) / sqrt(d2); inside iff dot >= thr.
__device__ __forceinline__ bool within_cone(int64_t facing_q, int64_t dx, int64_t dy,
                                            double thr) {
    using namespace obs_f64;
    const int64_t d2 = dx * dx + dy * dy;
    if (d2 <= 0) return true;
    const double fx = div((double)fixedpoint::cos_q16((int32_t)facing_q),
                          (double)obs_layout::OBS_FP_ONE);
    const double fy = -div((double)fixedpoint::sin_q16((int32_t)facing_q),
                           (double)obs_layout::OBS_FP_ONE);
    const double dot = div(add(mul((double)dx, fx), mul((double)dy, fy)), __dsqrt_rn((double)d2));
    return dot >= thr;
}

// vision.can_see(observer, target) for two living units of DIFFERENT teams
// (compute_pairwise / _compute_all_round skip the rest before asking):
//   all_round: a zombie observes nothing; else the line (ray_clear);
//   cone: inside the awareness radius ((dx^2 + dy^2) <= r*r, the int against
//   the float64 product) or inside the facing cone, then the line.
// The line is vision.ray_clear(observer centre -> target centre): the sight
// form (cover that blocks sight has no device form: refused at the build).
__device__ __forceinline__ bool can_see(const int64_t* o, const int64_t* t, int64_t rule,
                                        double cone_cos, double radius, const uint8_t* solid,
                                        const uint8_t* screen, int h, int w, bool* bad) {
    using namespace unit_store;
    const int64_t ox = centre_x(o), oy = centre_y(o);
    const int64_t tx = centre_x(t), ty = centre_y(t);
    if (rule == obs_layout::OBS_SIGHT_ALL_ROUND) {
        if (o[UNIT_COL_IS_ZOMBIE]) return false;
        return line_clear(solid, screen, h, w, false, oy, ox, ty, tx, bad);
    }
    const int64_t dx = tx - ox, dy = ty - oy;
    const bool in_awareness = (double)(dx * dx + dy * dy) <= obs_f64::mul(radius, radius);
    if (!in_awareness && !within_cone(o[UNIT_COL_FACING_Q], dx, dy, cone_cos)) return false;
    return line_clear(solid, screen, h, w, false, oy, ox, ty, tx, bad);
}

}  // namespace obs_sight
}  // namespace breach_cuda
