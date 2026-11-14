#pragma once
// ============================================================================
// Smoke transport v2 (#12, docs/smoke_transport_design_2026-10-04.md §2.3):
// THE TRACE DECAY ROUNDING — one per-cell rule, shared by both backends.
// ============================================================================
//
// The once-per-tick trace tail's third step removes, from a trace cell holding
// v > 0 counts,
//
//     lost = max(1, round_nearest(v · frac_q / 2^16)),   capped at v,
//
// (#88 amended this from ceil: at the fine trace currency, 2^8 finer, ceil
// over-removes up to a whole count per tick of a decay that is resolved to
// 1/256 of one; nearest keeps the configured e-fold within 3 % down to
// tau = 0.03 per tile, design docs/smoke_unit_rescale_design_2026-10-20.md
// sec 2.3 / D4 and sec 3.2. The floor of 1 keeps the bounded-time-to-zero
// property the D4 fix needed.)
//
// where frac_q = quantize(decay_g · dt) clamped to [0, ONE] (the host fold —
// trace_decay::frac_q below). Above the floor this keeps the configured
// e-fold; below it, it removes one count a tick, so thin gas reaches 0 in
// bounded time (the D4 fix: the floored `mul_q16(v, frac_q)` left a permanent
// haze, and a permanent lethal poison band, once SL's leak was gone).
//
// WHY A HEADER OF ITS OWN: the rounding is under review (a decay-rounding
// amendment may follow). Both backends call exactly these two functions — the
// CPU tail's decay step (bulk_transport.cpp `trace_decay_plane`) and the
// resident CUDA tail's isolated decay kernel (cuda_bulk_transport.cu
// `trace_tail_decay`) — so an amendment is ONE edit here, and the CPU == CUDA
// gates (tests/cuda_trace_smoke_check.py, chained and resident) keep holding
// the two call sites to the same law.
//
// Pure integer (int64), FP_HD: the same expression compiles for host and device.
#include <cstdint>

#include "fixed_point.h"   // q16, quantize, FP_ONE, FP_SHIFT, FP_HD

namespace trace_decay {

// The per-tick decay fraction of a trace gas, Q16.16: quantize(decay_g · dt),
// 0 for a non-decaying gas, clamped to [0, ONE] (decay·dt >= 1 removes it all).
// HOST fold — called once per gas per tick, never per cell. Defined OUT OF
// LINE in bulk_transport.cpp (on /fp:strict): a double fold inline in a header
// could be compiled by a fast-math TU and FMA-contract (the emissive-table
// bake lesson), so both backends call this one compiled definition.
int32_t frac_q(float decay_g, float dt);

// Counts removed from a cell holding `v` (callers skip v <= 0 and frac_q == 0).
// v < 2^31, frac_q <= 2^16 -> the product < 2^47: no overflow.
FP_HD inline int64_t lost(int64_t v, int32_t frac_q) {
    // #88 (design v2 sec 2.3 / D4, B6): NEAREST, with a floor of one count.
    // frac_q <= 0 -> 0 INSIDE the kit: the floor of 1 would otherwise remove a
    // count from a gas that does not decay (callers also skip it; the kit no
    // longer relies on them). v <= 0 -> 0.
    if (frac_q <= 0 || v <= 0) return 0;
    int64_t l = (v * (int64_t)frac_q + (int64_t)(fixedpoint::FP_ONE >> 1))
                >> fixedpoint::FP_SHIFT;
    if (l < 1) l = 1;
    if (l > v) l = v;
    return l;
}

}  // namespace trace_decay
