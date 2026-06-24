"""S1 test helpers — Q16.16 water field quantize/dequantize for the water tests.

The water core is int32 Q16.16 (S1). The water tests author scenarios in metres;
these helpers convert at the solver boundary so the tests stay readable in SI
units. Mirrors src/simulation/water_fixed.py (and the C++ fixed_point.h).
"""
from __future__ import annotations

import numpy as np

FP_ONE = 65536
FP_ONE_F = float(FP_ONE)


def q(metres) -> np.ndarray:
    """Real metres -> Q16.16 int32 (round-half-away-from-zero)."""
    arr = np.asarray(metres, dtype=np.float64) * FP_ONE_F
    out = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return out.astype(np.int32)


def deq(qarr) -> np.ndarray:
    """Q16.16 int32 -> float64 metres (exact /65536)."""
    return np.asarray(qarr, dtype=np.float64) / FP_ONE_F


def zeros_q(h, w) -> np.ndarray:
    """A zeroed Q16.16 int32 field (depth / velocity)."""
    return np.zeros((h, w), dtype=np.int32)
