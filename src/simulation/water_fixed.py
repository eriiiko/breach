"""Q16.16 fixed-point helpers for the water core (S1).

The synced water state (water_depth, flow_vx, flow_vy, floor_height, and the
runner's `before` snapshot) is int32 Q16.16 metres / m/s — scale 2^16 == 65536 —
so the integer transport is bit-identical cross-machine (the determinism the
float path could not give). These helpers convert metres <-> Q16.16 at the
boundaries (level painting, field edits, the renderer, the float bridges).

Mirrors the C++ ``fixed_point.h`` convention exactly:
  * quantize: round-to-nearest (round-half-away-from-zero), matching
    ``fixedpoint::quantize`` so a value written Python-side and one written
    C++-side land on the same integer.
  * dequantize: exact /65536.

Keep this the SINGLE Python source of the scale (FP_ONE) so the renderer, the
field-edit clamp, and the tests never hardcode 65536.
"""
from __future__ import annotations

import numpy as np

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536
FP_ONE_F = float(FP_ONE)


def quantize(metres):
    """Real metres (scalar or array) -> Q16.16 int32, round-to-nearest.

    Round-half-away-from-zero (symmetric), matching ``fixedpoint::quantize``:
    a positive value adds 0.5 before truncation, a negative subtracts 0.5.
    Computed in float64 so the product is exact for in-range inputs.
    """
    arr = np.asarray(metres, dtype=np.float64) * FP_ONE_F
    # round-half-away-from-zero (np.round is banker's rounding; use the
    # +/-0.5-then-truncate form to match the C++ helper exactly).
    out = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return out.astype(np.int32)


def quantize_scalar(metres: float) -> int:
    """Scalar metres -> Q16.16 int (round-half-away-from-zero)."""
    v = float(metres) * FP_ONE_F
    return int(np.floor(v + 0.5) if v >= 0.0 else np.ceil(v - 0.5))


def dequantize(q):
    """Q16.16 int32 (scalar or array) -> float64 metres (exact /65536)."""
    return np.asarray(q, dtype=np.float64) / FP_ONE_F


def dequantize_f32(q):
    """Q16.16 int32 -> float32 metres (the renderer/overlay boundary)."""
    return (np.asarray(q, dtype=np.float64) / FP_ONE_F).astype(np.float32)
