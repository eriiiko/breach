"""Q16.16 fixed-point helpers for the explicit WAVE state (S2a).

The synced wave state — ``wave_p`` (acoustic anomaly), ``wave_v`` (wave
velocity), ``wave_source`` (injected energy) — is int32 Q16.16, scale 2^16 ==
65536, the SAME scale as water/heat (so the whole fixed-point sim shares one
domain). Integer +/-/* are exact + associative, so the wave transport is
bit-identical cross-machine (the determinism the float path could not give).

The Q-S2-2 measurement (tests/_s2a_wave_v_measure.py) confirmed wave_v stays
inside +/-32768 even under a maximal blast (peak ~2674), so wave_v keeps Q16.16
(no Q24.8 exception). wave_p (~710 peak) and wave_source likewise fit.

These helpers convert real units <-> Q16.16 at the boundaries (field edits, the
recorder/render dequantize, tests). Mirrors C++ ``fixed_point.h`` exactly:
  * quantize  — round-to-nearest (round-half-away-from-zero), matching
    ``fixedpoint::quantize`` so a value written Python-side and one written
    C++-side land on the same integer.
  * dequantize — exact /65536.

This is the SAME scale as ``water_fixed`` (FP_ONE == 65536); we keep a separate
module so each system names its own boundary helpers (no cross-import that would
imply the wave is "water").
"""
from __future__ import annotations

import numpy as np

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536
FP_ONE_F = float(FP_ONE)


def quantize(value):
    """Real value (scalar or array) -> Q16.16 int32, round-half-away-from-zero.

    Matches ``fixedpoint::quantize``: a positive value adds 0.5 before
    truncation, a negative subtracts 0.5. Computed in float64 (exact for the
    in-range wave magnitudes).
    """
    arr = np.asarray(value, dtype=np.float64) * FP_ONE_F
    out = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return out.astype(np.int32)


def quantize_scalar(value: float) -> int:
    """Scalar -> Q16.16 int (round-half-away-from-zero)."""
    v = float(value) * FP_ONE_F
    return int(np.floor(v + 0.5) if v >= 0.0 else np.ceil(v - 0.5))


def dequantize(q):
    """Q16.16 int32 (scalar or array) -> float64 (exact /65536)."""
    return np.asarray(q, dtype=np.float64) / FP_ONE_F


def dequantize_f32(q):
    """Q16.16 int32 -> float32 (the renderer/overlay/recorder boundary)."""
    return (np.asarray(q, dtype=np.float64) / FP_ONE_F).astype(np.float32)
