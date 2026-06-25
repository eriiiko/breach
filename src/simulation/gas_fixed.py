"""Q16.16 fixed-point helpers for the smoke + 5 gas planes (S2b).

The synced smoke/gas state — ``gmap.gas`` (the (N, h, w) density planes) and its
``gmap.smoke`` view (BLACK_SMOKE slice) — is int32 Q16.16, scale 2^16 == 65536,
the SAME scale as water/heat/wave (so the whole fixed-point sim shares one
domain). The gases are [0,1]-clamped tracers: 0 == clear, FP_ONE (65536) == fully
opaque/saturated.

Unlike the conserved fields (water/atmosphere), smoke/gas are advected by an
INTEGER semi-Lagrangian whose >>16 truncation is a deliberate gentle decay — so
they are NON-conservative but DETERMINISTIC (Q-S2-1, docs/s2_fixed_point_plan.md
§S2b). Integer +/-/* are exact + associative, so the transport is bit-identical
cross-machine — that determinism, not conservation, is the contract.

These helpers convert real density <-> Q16.16 at the boundaries (field edits, the
recorder/render dequantize, the raycaster float bridge, tests). Mirrors C++
``fixed_point.h`` exactly:
  * quantize  — round-to-nearest (round-half-away-from-zero), matching
    ``fixedpoint::quantize`` so a value written Python-side and one written
    C++-side land on the same integer.
  * dequantize — exact /65536.

GPU note (Q-S2-6, frozen): the 5 [0,1]-clamped gas planes are recorded as
int16(Q1.15) in the format-version tag for the eventual CUDA bandwidth win; we
ship int32 on CPU now. The scale here (FP_ONE == 65536) is the int32 CPU form.

Same scale as ``water_fixed`` / ``wave_fixed``; a separate module so each system
names its own boundary helpers (no cross-import implying smoke "is" water/wave).
"""
from __future__ import annotations

import numpy as np

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536
FP_ONE_F = float(FP_ONE)
# [0,1] tracer saturation ceiling in Q16.16 (the integer clamp the solver applies).
SMOKE_MAX_Q = FP_ONE


def quantize(value):
    """Real density (scalar or array) -> Q16.16 int32, round-half-away-from-zero.

    Matches ``fixedpoint::quantize``: a positive value adds 0.5 before
    truncation, a negative subtracts 0.5. Computed in float64 (exact for the
    in-range [0,1] densities).
    """
    arr = np.asarray(value, dtype=np.float64) * FP_ONE_F
    out = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return out.astype(np.int32)


def quantize_scalar(value: float) -> int:
    """Scalar density -> Q16.16 int (round-half-away-from-zero)."""
    v = float(value) * FP_ONE_F
    return int(np.floor(v + 0.5) if v >= 0.0 else np.ceil(v - 0.5))


def dequantize(q):
    """Q16.16 int32 (scalar or array) -> float64 (exact /65536)."""
    return np.asarray(q, dtype=np.float64) / FP_ONE_F


def dequantize_f32(q):
    """Q16.16 int32 -> float32 (the renderer/raycaster/recorder float bridge)."""
    return (np.asarray(q, dtype=np.float64) / FP_ONE_F).astype(np.float32)
