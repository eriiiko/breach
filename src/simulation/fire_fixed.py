"""Q16.16 fixed-point helpers for the fire intensity field (S3a).

The synced fire state — ``gmap.fire`` (the per-tile combustion intensity I) — is
int32 Q16.16, scale 2^16 == 65536, the SAME scale as water/heat/wave/atmosphere/
gas (so the whole fixed-point sim shares one domain). Fire intensity is a
[0,1]-clamped tracer: 0 == unlit, FP_ONE (65536) == fully ablaze. The fire field
is the THIRD and FINAL sim field migrated to integer (S3, the closer of the
fixed-point arc).

Unlike the conserved fields (water/atmosphere), fire is a NON-conserved logistic
source/sink (it grows/dies per cell, it is not transported). Its determinism
contract is the same as smoke/gas: integer +/-/* are exact + associative, so the
field is bit-identical cross-machine — that determinism, not conservation, is the
contract. (S3a flips ONLY the representation; the C++ logistic math stays float
behind a temporary internal bridge until S3b — so the FEEL is unchanged this
commit, only the storage dtype + the Python ignition twin go integer.)

These helpers convert real intensity <-> Q16.16 at the boundaries (level
painting / debug seeds, the renderer dequantize, the recorder, the C++ float
bridge in physics_engine.step_tail, the heat-ray ``range``/``intensity`` params,
tests). Mirrors C++ ``fixed_point.h`` exactly:
  * quantize  — round-to-nearest (round-half-away-from-zero), matching
    ``fixedpoint::quantize`` so a value written Python-side and one written
    C++-side land on the same integer.
  * dequantize — exact /65536.

GPU note (Q5, frozen): the [0,1]-clamped fire plane is recorded as int16(Q1.15)
in the format-version tag for the eventual CUDA bandwidth win; we ship int32 on
CPU now. The scale here (FP_ONE == 65536) is the int32 CPU form — same decide-once
rule as smoke/gas.

Same scale as ``water_fixed`` / ``wave_fixed`` / ``gas_fixed`` / ``atmosphere_fixed``;
a separate module so each system names its own boundary helpers (no cross-import
implying fire "is" water/wave/gas/atmosphere).
"""
from __future__ import annotations

import numpy as np

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536
FP_ONE_F = float(FP_ONE)
# [0,1] tracer saturation ceiling in Q16.16 (the integer clamp the solver applies).
FIRE_MAX_Q = FP_ONE


def quantize(value):
    """Real intensity (scalar or array) -> Q16.16 int32, round-half-away-from-zero.

    Matches ``fixedpoint::quantize``: a positive value adds 0.5 before
    truncation, a negative subtracts 0.5. Computed in float64 (exact for the
    in-range [0,1] intensities).
    """
    arr = np.asarray(value, dtype=np.float64) * FP_ONE_F
    out = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return out.astype(np.int32)


def quantize_scalar(value: float) -> int:
    """Scalar intensity -> Q16.16 int (round-half-away-from-zero)."""
    v = float(value) * FP_ONE_F
    return int(np.floor(v + 0.5) if v >= 0.0 else np.ceil(v - 0.5))


def dequantize(q):
    """Q16.16 int32 (scalar or array) -> float64 (exact /65536)."""
    return np.asarray(q, dtype=np.float64) / FP_ONE_F


def dequantize_f32(q):
    """Q16.16 int32 -> float32 (the renderer/raycaster/recorder/C++ float bridge)."""
    return (np.asarray(q, dtype=np.float64) / FP_ONE_F).astype(np.float32)
