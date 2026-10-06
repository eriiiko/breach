#pragma once
// ============================================================================
// Smoke transport v2 (#12, docs/smoke_transport_design_2026-10-04.md §2.3):
// THE TRACE DECAY ROUNDING — one per-cell rule, shared by both backends.
// ============================================================================
//
// The once-per-tick trace tail's third step removes, from a trace cell holding
// v > 0 counts,
//
//     lost = ceil(v · frac_q / 2^16),   capped at v,
//
// where frac_q = quantize(decay_g · dt) clamped to [0, ONE] (the host fold —
// trace_decay::frac_q below). Above the old floor this keeps the configured
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
    int64_t l = (v * (int64_t)frac_q + (int64_t)(fixedpoint::FP_ONE - 1))
                >> fixedpoint::FP_SHIFT;
    if (l > v) l = v;
    return l;
}

}  // namespace trace_decay
