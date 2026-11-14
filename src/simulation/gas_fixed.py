"""Fixed-point boundary helpers for the gas planes: THREE currencies, three names.

``gmap.gas`` is ONE ``(N_GASES, h, w)`` int32 array holding two currencies
(#88, docs/smoke_unit_rescale_design_2026-10-20.md §4):

* the five TRACE planes (steam, smoke, poison, teargas, fuel_gas; ids
  ``0 .. N_TRACE_GASES - 1``) are in the TRACE currency, ``TRACE_ONE =
  2^(16 + TRACE_FINE_BITS)`` counts per real unit of density;
* the two BULK planes (O2, inert_N2) stay Q16.16, ``BULK_ONE = 2^16`` counts
  per real unit (one ambient air cell's worth of molecules is ~1.0).

A third family, ``q16_*``, names the Q16 FRACTIONS and TEMPERATURES that used to
borrow this module's ambiguous names (filter efficiencies, the O2 fraction,
``t_min`` / ``t_max``, the sky lambda, the T_AMB fold): they are neither bulk
nor trace, and must never move with the trace currency (critique A12).

The old ambiguous names (``FP_ONE``, ``FP_ONE_F``, ``SMOKE_MAX_Q``,
``quantize``, ``quantize_scalar``, ``dequantize``, ``dequantize_f32``) are
DELETED, so a site that was not moved to its own currency fails loudly with an
AttributeError (V1's lint keeps them gone).

``TRACE_FINE_BITS`` is a LITERAL twin of ``cpp/src/fixed_point.h``'s, not #78's
read-through of the bound engine: this module is import-light and serves
GameMaps that have no engine (level load, most unit tests, the tools), which a
read-through would silently read at 0 bits (critique C5). The twins, the
integer reference's and the digest spec's are held equal by
``tests/test_trace_currency.py``.

Rounding matches ``fixedpoint::quantize`` exactly: round-half-away-from-zero, so
a value written Python-side and one written C++-side land on the same integer.
Coefficients (``heat_absorb_q16``, ``light_absorb_q16``, ...) stay Q16 per REAL
unit; a density law over the trace planes absorbs the extra bits in its one
shift, ``trace_density_shr``.
"""
from __future__ import annotations

import numpy as np

# ---- Q16 fractions and temperatures (A12) ---------------------------------
Q16_SHIFT = 16
Q16_ONE = 1 << Q16_SHIFT                 # 65536: a Q16 fraction's 1.0
_Q16_ONE_F = float(Q16_ONE)

# ---- the BULK planes (O2, inert_N2): Q16.16 --------------------------------
BULK_SHIFT = 16
BULK_ONE = 1 << BULK_SHIFT               # 65536 counts per real unit
_BULK_ONE_F = float(BULK_ONE)

# ---- the TRACE planes (#88) -----------------------------------------------
# The trace currency's extra fractional bits. THE twin of fixed_point.h
# TRACE_FINE_BITS (a literal, see the module docstring).
TRACE_FINE_BITS = 8
TRACE_SHIFT = 16 + TRACE_FINE_BITS
# One unit of trace density. NOT a ceiling: smoke transport v2 (#12) deleted
# every trace clamp outside a reader, so a compressed pocket can hold more.
TRACE_ONE = 1 << TRACE_SHIFT
_TRACE_ONE_F = float(TRACE_ONE)
# The int32 storage's representability limit (the counted saturation, §5.3).
TRACE_CELL_MAX = (1 << 31) - 1


def _round_half_away(scaled):
    """float64 array -> float64 array of integers, round-half-away-from-zero
    (``fixedpoint::quantize``'s +/-0.5 then truncate)."""
    return np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))


def _round_half_away_scalar(v: float) -> int:
    return int(np.floor(v + 0.5) if v >= 0.0 else np.ceil(v - 0.5))


# ---- Q16 fractions / temperatures ----------------------------------------
def quantize_q16_scalar(value: float) -> int:
    """A Q16 fraction or temperature -> its Q16 int (round-half-away)."""
    return _round_half_away_scalar(float(value) * _Q16_ONE_F)


def dequantize_q16(q):
    """Q16 int (scalar or array) -> float64, exact /2^16."""
    return np.asarray(q, dtype=np.float64) / _Q16_ONE_F


# ---- bulk -----------------------------------------------------------------
def quantize_bulk(value):
    """Real bulk density (scalar or array) -> Q16.16 int32, round-half-away."""
    arr = np.asarray(value, dtype=np.float64) * _BULK_ONE_F
    return _round_half_away(arr).astype(np.int32)


def quantize_bulk_scalar(value: float) -> int:
    """Scalar bulk density -> Q16.16 Python int (round-half-away)."""
    return _round_half_away_scalar(float(value) * _BULK_ONE_F)


def dequantize_bulk(q):
    """Bulk Q16.16 (scalar or array) -> float64, exact /2^16."""
    return np.asarray(q, dtype=np.float64) / _BULK_ONE_F


def dequantize_bulk_f32(q):
    """Bulk Q16.16 -> float32 (the render float bridge)."""
    return (np.asarray(q, dtype=np.float64) / _BULK_ONE_F).astype(np.float32)


# ---- trace ----------------------------------------------------------------
def quantize_trace(value):
    """Real trace density (scalar or array) -> the trace currency as an INT64
    array, round-half-away. It never narrows (critique A3): only the counted
    saturation narrows to the int32 plane, so an out-of-range seed is seen,
    never wrapped. Exact in float64 for any int32-representable result."""
    arr = np.asarray(value, dtype=np.float64) * _TRACE_ONE_F
    return _round_half_away(arr).astype(np.int64)


def quantize_trace_scalar(value: float) -> int:
    """Scalar trace density -> the trace currency, a Python int (round-half-
    away). A gate's threshold is quantized here, at the trace currency (A1)."""
    return _round_half_away_scalar(float(value) * _TRACE_ONE_F)


def dequantize_trace(q):
    """Trace counts (scalar or array) -> float64 real density, exact
    /2^TRACE_SHIFT (a power of two: float-identical on S_old * 2^k)."""
    return np.asarray(q, dtype=np.float64) / _TRACE_ONE_F


def dequantize_trace_f32(q):
    """Trace counts -> float32 real density (the renderer / recorder bridge)."""
    return (np.asarray(q, dtype=np.float64) / _TRACE_ONE_F).astype(np.float32)


def trace_density_shr(s: int) -> int:
    """sum_g coef_q16[g] * S_g (S in trace counts, coef Q16 per real unit) ->
    a Q16 fraction: ``s >> TRACE_SHIFT`` (fixed_point.h trace_density_shr).
    The sum is never negative (each term clamps a negative density to 0)."""
    s = int(s)
    if s < 0:
        raise ValueError(f"trace_density_shr: negative density sum {s}")
    return s >> TRACE_SHIFT


def trace_to_q16(s):
    """Trace counts (scalar or array) -> Q16 counts, floor (``>>
    TRACE_FINE_BITS``): the smoke sensor's signal currency (critique A7). An
    array stays int32 (the shift only narrows); a scalar is a Python int."""
    if np.isscalar(s) and not isinstance(s, np.ndarray):
        return int(s) >> TRACE_FINE_BITS
    return np.right_shift(np.asarray(s), TRACE_FINE_BITS)


def trace_from_bulk(q):
    """A bulk-count (Q16) mass entering a trace plane: ``q * 2^TRACE_FINE_BITS``
    (fixed_point.h trace_from_bulk: the soot mint). Non-negative by contract;
    an array comes back int64 (never narrowed), a scalar as a Python int."""
    if np.isscalar(q) and not isinstance(q, np.ndarray):
        q = int(q)
        if q < 0:
            raise ValueError(f"trace_from_bulk: negative mass {q}")
        return q * (1 << TRACE_FINE_BITS)
    arr = np.asarray(q, dtype=np.int64)
    if (arr < 0).any():
        raise ValueError("trace_from_bulk: negative mass")
    return arr * (1 << TRACE_FINE_BITS)


def trace_add_sat(s: int, add: int):
    """The counted int32 saturation (§5.3), the twin of fixed_point.h
    trace_add_sat: ``t = s + add`` exactly; above TRACE_CELL_MAX it stores
    TRACE_CELL_MAX and returns the excess as ``clip``. It saturates from ABOVE
    ONLY: a negative ``t`` is a contract violation and raises (clamping it would
    mint unbooked mass, critique A4). Returns ``(stored, clip)`` as Python ints."""
    t = int(s) + int(add)
    if t < 0:
        raise ValueError(f"trace_add_sat: negative result {t} (s={s}, add={add})")
    if t > TRACE_CELL_MAX:
        return TRACE_CELL_MAX, t - TRACE_CELL_MAX
    return t, 0
