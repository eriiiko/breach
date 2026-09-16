"""Q16.16 fixed-point helpers for the OPTICAL EXTINCTION planes (ray-engine-v2 P1).

THE BOUNDARY MODULE for every extinction coefficient the radiation sweep's
channels see (docs/ray_engine_v2_design_v3_2026-09-15.md §3, row 7): the
material heat extinction ``heat_atten`` -> ``GameMap.heat_atten_q``, the
per-unit body extinction ``unit.heat_atten`` -> the ``stamp_units`` MAX into
``GameMap.dyn_heat_atten_q``, and — at P6 — the light RGB planes. None of the
other seven ``*_fixed.py`` modules is about optics, hence a new one (reuse-or-
new, decided in the design).

An extinction coefficient is a FRACTION in [0, 1]: 0 = transparent (air), 1 =
opaque (a wall, a body). Q16 with ONE = 65536, the SAME scale as every other
Q16.16 field, so ``(stream * a) >> 16`` is the sweep's one shift. The design's
ingress invariant ``0 <= a <= d <= ONE`` (§2.3) rests on the coefficient never
leaving [0, 1]: ``quantize``/``quantize_scalar`` therefore RAISE outside that
range instead of clamping — an out-of-range coefficient is a config or code
error, never something to silently fix at the door.

Mirrors C++ ``fixed_point.h`` exactly:
  * quantize  — round-to-nearest (round-half-away-from-zero), matching
    ``fixedpoint::quantize`` so a coefficient written Python-side and one
    written C++-side land on the same integer.
  * dequantize — exact /65536.
"""
from __future__ import annotations

import numpy as np

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536 == a coefficient of exactly 1.0
FP_ONE_F = float(FP_ONE)


def _check_range(arr: np.ndarray, what: str) -> None:
    """RAISE (never clamp) if any coefficient is outside [0, 1] or non-finite."""
    if arr.size == 0:
        return
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: an extinction coefficient must be finite")
    lo = float(arr.min())
    hi = float(arr.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"{what}: an extinction coefficient must lie in [0, 1] "
            f"(got min {lo}, max {hi}); the sweep's positivity rests on "
            f"0 <= a <= d <= ONE (design v3 section 2.3)")


def quantize(value):
    """Real coefficient(s) in [0, 1] -> Q16 int32, round-half-away-from-zero.

    Raises ``ValueError`` outside [0, 1]. Matches ``fixedpoint::quantize``.
    """
    arr = np.asarray(value, dtype=np.float64)
    _check_range(arr, "optics_fixed.quantize")
    scaled = arr * FP_ONE_F
    out = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return out.astype(np.int32)


def quantize_scalar(value: float) -> int:
    """Scalar coefficient in [0, 1] -> Q16 int (round-half-away-from-zero).

    Raises ``ValueError`` outside [0, 1].
    """
    v = float(value)
    _check_range(np.asarray([v], dtype=np.float64), "optics_fixed.quantize_scalar")
    s = v * FP_ONE_F
    return int(np.floor(s + 0.5) if s >= 0.0 else np.ceil(s - 0.5))


def dequantize(q):
    """Q16 int32 (scalar or array) -> float64 coefficient (exact /65536)."""
    return np.asarray(q, dtype=np.float64) / FP_ONE_F


def dequantize_f32(q):
    """Q16 int32 -> float32 coefficient (any render/overlay/debug boundary)."""
    return (np.asarray(q, dtype=np.float64) / FP_ONE_F).astype(np.float32)
