#pragma once
// ============================================================================
// #29 P7a.3 -- THE SPRAY LAW's device functions (combat.spray_cone_tiles and
// combat._enqueue_spray_cells), shared by slot 4b (cuda_unit_sprays.cu, with
// its TEST-ONLY cone probe) and the applier's SPRAY_CELLS effect
// (cuda_applier.cu). Integer for integer the Python:
//
// THE CONE (spray_cone_tiles; its apex the sprayer's centre tile, its bearing
// the burst's Q16 aim): cone_params() derives a sprayer's cone from its row
// and the tables (the weapon's spray_range_tiles and spray_cone_half_q
// rows; cos_q16 of the half-angle, as the law does per call); cone_tile()
// decides ONE tile of the clipped box -- its order-free part, so a block's
// threads split the box: the apex and the range (dx^2 + dy^2 in (0, r^2]), the
// NOZZLE rule (the sprayer's own square footprint excluded), the half-plane
// (dot_q >= 0) and the cone (dot_q^2 >= dist_sq * c_q^2, exact in int64 for a
// range <= 2^14), then the PHYSICAL line of sight (unit_sight::has_los with a
// null screen: solid alone), and the falloff divisor max(1, isqrt(dist_sq)).
// The tile ORDER is the caller's: row-major over the box (the Python's two
// loops), a serial pass over the decided box.
//
// THE CELLS AT THE RESOLVE (apply_spray_cells, the SPRAY_CELLS effect): per
// cell of the record's span of the env's cell buffer, in order -- heat_q > 0:
// a TILE ADD edit of div_round(heat_q, div) into `heat` (source
// spray_src_heat) at the edit door, then its heat claim; a gas slice and
// gas_q > 0: the gas claim, then a TILE ADD edit of div_round(gas_q, div)
// into that `gas` slice (source spray_src_gas). It reads NO map state (the
// cells were decided in phase A against the walls standing then): an earlier
// CHEW in the same walk changes nothing here. All or nothing: every operand
// (the span, the slice, every cell) is checked before the first write.
//
// Device-only header; CLASS T: nothing in the played tick includes it.
#include <cstdint>

#include "cuda_edit_door.cuh"
#include "cuda_unit_lib.cuh"
#include "cuda_unit_sight.cuh"
#include "applier_layout.h"       // GENERATED: SPRAY_CELLS's fields
#include "unit_events_layout.h"    // GENERATED: UNIT_CREDIT_HEAT

namespace breach_cuda {
namespace unit_spray {

using namespace unit_lib;

//: The proved range of a spray: a range of at most RANGE_MAX tiles keeps
//: dot_q^2 (|dot| <= 2 * 2^14 * 2^16 = 2^31) and dist_sq * c_q^2 (<= 2^29 *
//: 2^32) inside int64; an apex within +-APEX_MAX.
constexpr int64_t RANGE_MAX = (int64_t)1 << 14;
constexpr int64_t APEX_MAX = (int64_t)1 << 20;
//: field_edit.EditMode.ADD.
constexpr int64_t EDIT_MODE_ADD = 0;
//: A cell's width in the cell buffer: (y, x, div).
constexpr int CELL_W = 3;

struct ConeParams {
    int64_t ay, ax;          // the apex (the centre tile)
    int64_t r_sq;            // range^2
    int64_t cos_q, sin_q;    // the bearing's unit vector, Q16
    int64_t c_sq;            // cos(half-angle)^2, Q32
    int64_t ex_x, ex_y, fp;  // the nozzle rule: the anchor and the footprint
    int64_t y0, y1, x0, x1;  // the clipped box (empty when y0 > y1 or x0 > x1)
};

// The cone with apex (ay, ax), the nozzle square of side fp at (ex_x, ex_y)
// (fp 0: nothing excluded), the bearing `angle`, range r and half-angle
// `half` (Q16) on an (h, w) map. False with *bad set outside the proved range.
__device__ inline bool cone_from(int64_t ay, int64_t ax, int64_t ex_x, int64_t ex_y, int64_t fp,
                                 int64_t angle, int64_t r, int64_t half, int h, int w,
                                 ConeParams& P, bool* bad) {
    if (fp < 0 || fp > FOOTPRINT_MAX || angle < INT32_MIN || angle > INT32_MAX ||
        ax < -APEX_MAX || ax > APEX_MAX || ay < -APEX_MAX || ay > APEX_MAX ||
        ex_x < -APEX_MAX || ex_x > APEX_MAX || ex_y < -APEX_MAX || ex_y > APEX_MAX ||
        r < 0 || r > RANGE_MAX || half < INT32_MIN || half > INT32_MAX) {
        *bad = true;
        return false;
    }
    P.ax = ax;
    P.ay = ay;
    P.r_sq = r * r;
    P.cos_q = fixedpoint::cos_q16((q16)angle);
    P.sin_q = fixedpoint::sin_q16((q16)angle);
    const int64_t c_q = fixedpoint::cos_q16((q16)half);
    P.c_sq = c_q * c_q;
    P.ex_x = ex_x;
    P.ex_y = ex_y;
    P.fp = fp;
    P.y0 = ay - r > 0 ? ay - r : 0;
    P.y1 = ay + r < h - 1 ? ay + r : h - 1;
    P.x0 = ax - r > 0 ? ax - r : 0;
    P.x1 = ax + r < w - 1 ? ax + r : w - 1;
    return true;
}

// The cone of a sprayer whose row is `row` (its weapon m, its aim angle a
// Q16 int): apex the centre tile (tile + fp // 2), the nozzle its footprint.
__device__ inline bool cone_params(const int64_t* row, const int64_t* tab, int64_t m,
                                   int64_t angle, int h, int w, ConeParams& P, bool* bad) {
    const int64_t fp = row[UNIT_COL_FOOTPRINT];
    if (fp < 1 || fp > FOOTPRINT_MAX) {
        *bad = true;
        return false;
    }
    const int64_t tx = tile_of(row[UNIT_COL_X_Q]), ty = tile_of(row[UNIT_COL_Y_Q]);
    return cone_from(ty + fp / 2, tx + fp / 2, tx, ty, fp, angle,
                     tab[unit_tables::UNIT_TAB_SPRAY_RANGE_TILES + m],
                     tab[unit_tables::UNIT_TAB_SPRAY_CONE_HALF_Q + m], h, w, P, bad);
}

// spray_cone_tiles' decision for tile (y, x): its falloff divisor, or 0 when
// it is not yielded. `solid` the env's (h, w) plane.
__device__ inline int64_t cone_tile(const ConeParams& P, const uint8_t* solid, int h, int w,
                                    int64_t y, int64_t x, bool* bad) {
    const int64_t dy = y - P.ay, dx = x - P.ax;
    const int64_t dist_sq = dx * dx + dy * dy;
    if (dist_sq == 0 || dist_sq > P.r_sq) return 0;                     // apex / out of range
    if (x >= P.ex_x && x < P.ex_x + P.fp && y >= P.ex_y && y < P.ex_y + P.fp)
        return 0;                                                       // the nozzle rule
    const int64_t dot = dx * P.cos_q + dy * P.sin_q;
    if (dot < 0) return 0;                                              // behind the sprayer
    if (dot * dot < dist_sq * P.c_sq) return 0;                         // outside the cone
    if (!unit_sight::has_los(solid, nullptr, h, w, P.ay, P.ax, y, x, bad))
        return 0;                                                       // occluded: no pour
    const int64_t d = fixedpoint::isqrt_q(dist_sq);
    return d > 1 ? d : 1;
}

// The box's tile count (0 when empty).
__device__ __forceinline__ int64_t box_size(const ConeParams& P) {
    if (P.y0 > P.y1 || P.x0 > P.x1) return 0;
    return (P.y1 - P.y0 + 1) * (P.x1 - P.x0 + 1);
}

// BLOCK-COOPERATIVE: every tile of the box decided by the block's threads
// into `dec` (the divisor, 0 not yielded, -1 a fault). The caller syncs.
__device__ inline void decide_box(const ConeParams& P, const uint8_t* solid, int h, int w,
                                  int64_t* dec) {
    const int64_t n = box_size(P);
    const int64_t bw = P.x1 - P.x0 + 1;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        bool bad = false;
        const int64_t d = cone_tile(P, solid, h, w, P.y0 + i / bw, P.x0 + i % bw, &bad);
        dec[i] = bad ? -1 : d;
    }
}

// ONE thread: the decided box in ROW-MAJOR order (the Python's two loops) as
// cells (y, x, div) through `sink`. False (nothing appended) when a tile
// faulted.
__device__ inline bool emit_box(const ConeParams& P, const int64_t* dec, RecordSink& sink) {
    const int64_t n = box_size(P);
    const int64_t bw = P.x1 - P.x0 + 1;
    for (int64_t i = 0; i < n; ++i)
        if (dec[i] < 0) return false;
    for (int64_t i = 0; i < n; ++i) {
        if (dec[i] == 0) continue;
        const int64_t c[CELL_W] = {P.y0 + i / bw, P.x0 + i % bw, dec[i]};
        emit_record(sink, c);
    }
    return true;
}

// The SPRAY_CELLS effect (see the header). Returns false (nothing written)
// when an operand leaves the proved range: the caller FAULTS.
__device__ inline bool apply_spray_cells(const int64_t* rec, const RecordListView& cells, int e,
                                         const edit_door::DoorView& D, RecordSink& claims,
                                         const int64_t* tab, int h, int w, int n_gases) {
    using namespace applier;
    const int64_t owner = rec[APPLIER_SPRAY_CELLS_OWNER];
    const int64_t heat_q = rec[APPLIER_SPRAY_CELLS_HEAT_Q];
    const int64_t gas_id = rec[APPLIER_SPRAY_CELLS_GAS_ID];
    const int64_t gas_q = rec[APPLIER_SPRAY_CELLS_GAS_Q];
    const int64_t start = rec[APPLIER_SPRAY_CELLS_START];
    const int64_t count = rec[APPLIER_SPRAY_CELLS_COUNT];
    const int64_t have = cells.count[e] < (int64_t)cells.cap ? cells.count[e] : (int64_t)cells.cap;
    if (start < 0 || count < 0 || start > have || count > have - start) return false;
    if (gas_id < -1 || gas_id >= n_gases) return false;
    if (heat_q < 0 || heat_q > ((int64_t)1 << 48) || gas_q < 0 || gas_q > ((int64_t)1 << 48))
        return false;
    const int64_t* C = cells.recs + (size_t)e * cells.cap * cells.width;
    for (int64_t i = start; i < start + count; ++i) {
        const int64_t* c = C + (size_t)i * cells.width;
        if (c[0] < 0 || c[0] >= h || c[1] < 0 || c[1] >= w || c[2] < 1) return false;
    }
    const int64_t src_heat = tab[unit_tables::UNIT_TAB_SPRAY_SRC_HEAT];
    const int64_t src_gas = tab[unit_tables::UNIT_TAB_SPRAY_SRC_GAS];
    for (int64_t i = start; i < start + count; ++i) {
        const int64_t* c = C + (size_t)i * cells.width;
        const int64_t y = c[0], x = c[1], div = c[2];
        if (heat_q > 0) {
            edit_door::door_tile_edit(D, e, tab, EDQ_F_HEAT, -1, EDIT_MODE_ADD,
                                      fixedpoint::div_round_q(heat_q, div), src_heat, y, x, h, w);
            edit_door::claim(claims, unit_events::UNIT_CREDIT_HEAT, y, x, owner);
        }
        if (gas_id >= 0 && gas_q > 0) {
            edit_door::claim(claims, gas_id, y, x, owner);
            edit_door::door_tile_edit(D, e, tab, EDQ_F_GAS, gas_id, EDIT_MODE_ADD,
                                      fixedpoint::div_round_q(gas_q, div), src_gas, y, x, h, w);
        }
    }
    return true;
}

}  // namespace unit_spray
}  // namespace breach_cuda
