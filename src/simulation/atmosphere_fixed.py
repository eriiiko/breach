"""Q16.16 fixed-point helpers for the atmosphere (pressure) + wind (S2c).

The synced atmosphere state — ``gmap.atmosphere`` (bulk air pressure) and the
derived ``gmap.wind_x`` / ``gmap.wind_y`` (= -grad(atmosphere + wave_p)) — is
int32 Q16.16, scale 2^16 == 65536, the SAME scale as water/heat/wave/gas (so the
whole fixed-point sim shares one domain). S2c is the CLOSER of the S2 group:
with atmosphere + wind integer the entire atmosphere/wave/wind/smoke/gas group is
cross-machine bit-identical (the only float bridge left is the downstream FIRE
coupling, S3). Integer +/-/* are exact + associative, so the transport is
bit-identical cross-machine — that determinism is the contract.

atmosphere is the CONSERVED field (the wave->atmosphere transfer is a conservative
integer ±-pair, exactly mass-neutral to the LSB; the vacuum/sponge BC + the W3
P*V compression are the deliberate-sink exceptions, by design). wind is a derived
signed pressure-gradient (NOT conserved).

These helpers convert real units <-> Q16.16 at the boundaries (level painting,
field edits, the recorder/render dequantize, the fire bridge, tests). Mirrors C++
``fixed_point.h`` exactly:
  * quantize  — round-to-nearest (round-half-away-from-zero), matching
    ``fixedpoint::quantize`` so a value written Python-side and one written
    C++-side land on the same integer.
  * dequantize — exact /65536.

WIND shares this scale (FP_ONE == 65536) and these helpers — the renderer /
recorder / fire bridge dequantize wind through ``dequantize`` / ``dequantize_f32``
here too (no separate ``wind_fixed`` module: one boundary helper, two consumers).

Same scale as ``water_fixed`` / ``wave_fixed`` / ``gas_fixed``; a separate module
so each system names its own boundary helpers (no cross-import implying the
atmosphere "is" water/wave/gas).
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
    in-range atmosphere/wind magnitudes ~1-2 interior, signed wind).
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
    """Q16.16 int32 -> float32 (the renderer/overlay/recorder/fire-bridge boundary)."""
    return (np.asarray(q, dtype=np.float64) / FP_ONE_F).astype(np.float32)
