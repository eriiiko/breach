"""S1 test helpers — Q16.16 water field quantize/dequantize for the water tests.

Re-export shim (issue #15): this used to be a hand-maintained copy of the
Q16.16 rounding helpers, admittedly just "mirroring" src/simulation/water_fixed.py
(and the C++ fixed_point.h) — a test-only duplicate of a sim-path rounding
rule, which is drift risk (a rounding fix could land in water_fixed.py and
silently not apply here). It now imports the canonical helpers directly; the
public names (`q`, `deq`, `zeros_q`, `FP_ONE`, `FP_ONE_F`) are unchanged so
every existing importer keeps working.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Defensive path bootstrap (relative to this file, not cwd): every current
# importer already puts ROOT and ROOT/src on sys.path before importing this
# module (see e.g. tests/test_water_solver.py), but this module may also be
# imported standalone/outside pytest, so don't rely on that.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from simulation.water_fixed import (  # noqa: E402
    FP_ONE,
    FP_ONE_F,
    quantize as q,
    dequantize as deq,
)


def zeros_q(h, w) -> np.ndarray:
    """A zeroed Q16.16 int32 field (depth / velocity)."""
    return np.zeros((h, w), dtype=np.int32)
