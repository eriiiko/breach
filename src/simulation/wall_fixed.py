"""Q16.16 fixed-point helpers for the wall HP field (S3b).

The synced wall fuel/structure state — ``gmap.wall_hp`` (per-tile structural HP,
the fire's fuel source ``F = clamp01(wall_hp/fuel_ref)``) — is int32 Q16.16, scale
2^16 == 65536, the SAME scale as water/heat/wave/atmosphere/gas/fire (so the whole
fixed-point sim shares one domain). wall_hp is a PHYSICAL >1 quantity (wood 30,
door 40, hull 300, steel 200) — NOT a [0,1] tracer like fire — but its DEPLETION
``wall_damage*dt*I`` per tick is fractional (≪ 1 HP), so the Q16.16 fraction lets
the burn-through accumulate correctly rather than rounding every sub-HP tick to 0
(the ratified S3b decision: int32 Q16.16, the clean fractional-depletion form).

300 HP -> 300*65536 = 19,660,800 counts, well inside int32 (< 2^31). The fire C++
step reads + depletes wall_hp as int32; ``destroy_wall`` resets it from the table;
the explosion structural-damage path (physics.apply_blast_damage) depletes it with
a quantized decrement.

GPU note (ratified, frozen): wall_hp is recorded as int16 (Q8.8) in the
format-version tag for the eventual CUDA bandwidth win (Erik's cave/wall-dense
maps) — Q8.8 holds 0..255.996 HP, enough for the live HP range with the structural
walls the dense maps care about (a decide-once like smoke/gas/fire); we ship int32
on CPU now (the full HP range + the Q16.16 fraction).

Mirrors C++ ``fixed_point.h`` exactly:
  * quantize  — round-to-nearest (round-half-away-from-zero), matching
    ``fixedpoint::quantize`` so an HP written Python-side and one written C++-side
    land on the same integer.
  * dequantize — exact /65536.
"""
from __future__ import annotations

import numpy as np

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536
FP_ONE_F = float(FP_ONE)


def quantize(value):
    """Real HP (scalar or array) -> Q16.16 int32, round-half-away-from-zero.

    Matches ``fixedpoint::quantize``: a positive value adds 0.5 before truncation,
    a negative subtracts 0.5. Computed in float64 (exact for the in-range HP).
    """
    arr = np.asarray(value, dtype=np.float64) * FP_ONE_F
    out = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return out.astype(np.int32)


def quantize_scalar(value: float) -> int:
    """Scalar HP -> Q16.16 int (round-half-away-from-zero)."""
    v = float(value) * FP_ONE_F
    return int(np.floor(v + 0.5) if v >= 0.0 else np.ceil(v - 0.5))


def dequantize(q):
    """Q16.16 int32 (scalar or array) -> float64 HP (exact /65536)."""
    return np.asarray(q, dtype=np.float64) / FP_ONE_F


def dequantize_f32(q):
    """Q16.16 int32 -> float32 HP (any render/overlay/debug boundary)."""
    return (np.asarray(q, dtype=np.float64) / FP_ONE_F).astype(np.float32)
