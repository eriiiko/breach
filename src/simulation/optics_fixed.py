"""Q16.16 fixed-point helpers for the OPTICAL EXTINCTION planes (ray-engine-v2 P1).

THE BOUNDARY MODULE for every extinction coefficient the radiation sweep's
channels see (docs/ray_engine_v2_design_v3_2026-09-15.md §3, row 7): the
material heat extinction ``heat_atten`` -> ``GameMap.heat_atten_q``, the
per-unit body extinction ``unit.heat_atten`` -> the ``stamp_units`` MAX into
``GameMap.dyn_heat_atten_q``, and — at P6 — the light RGB planes. None of the
other seven ``*_fixed.py`` modules is about optics, hence a new one (reuse-or-
new, decided in the design).

Since #78 it is also the Python side's ONE door out of the sweep's FINE HEAT
CURRENCY (the section at the end: ``fine_heat_shr`` for the sim path,
``dequantize_heat`` for readouts, ``sweep_fine_bits`` for which currency a
GameMap's sweep planes are in).

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


# ---------------------------------------------------------------------------
# THE SWEEP'S FINE HEAT CURRENCY (#78, 2026-09-25; docs/sweep_fine_heat_
# currency_brief_78_2026-09-25.md). The radiation sweep's heat channel works in
# a currency 2^k finer than one heat count: k is the E° table's own
# `fine_bits` (cpp/src/emissive_table.h E_FINE_BITS on the engine's table, read
# through the binding -- this module holds no copy of the number). The four
# sweep planes (`rad_net_sweep`, `rad_flux_sweep`, `rad_amb_sweep`,
# `rad_fluence`) are in it. A Python reader converts ONCE, through one of the
# two functions below -- never an inline `>> 11`:
#   * fine_heat_shr -- the SIM path (a synced quantity is derived from it, e.g.
#     a unit's heat exposure in exchange.py): the exact integer twin of the
#     kit's fixedpoint::fine_heat_shr, symmetric round-toward-zero;
#   * dequantize_heat -- the RENDER / readout / report path: a float in heat
#     counts, never written back.
# ---------------------------------------------------------------------------
def sweep_fine_bits(gmap) -> int:
    """The currency the gmap's four sweep planes are in: the fine bits of the E°
    table the bound engine's sweep books them from. A GameMap with no engine
    bound has no sweep writing them (a test that fills a plane by hand fills it
    in whole heat counts): 0."""
    eng = getattr(gmap, "_physics_engine", None)
    if eng is None or not hasattr(eng, "emissive"):
        return 0
    return int(eng.emissive.fine_bits)


def fine_heat_shr(x, s, fine_bits: int):
    """The Python twin of fixed_point.h ``fine_heat_shr`` (#78), value for value
    (tests/test_sweep_fine_currency.py holds the two equal): ``x`` in a currency
    2^fine_bits finer than one heat count, divided by 2^(s + fine_bits),
    rounded toward zero, SYMMETRIC -- +x and -x lose the same magnitude; a
    negative total exponent is an exact multiply. ``s = 0`` gives whole heat
    counts, ``s = heat_inv_shift`` a thermal solid's temperature step. Accepts
    a Python int (returns an int) or an integer numpy array (returns int64);
    ``s`` may be an int or, with an array ``x``, an array of per-cell shifts."""
    if isinstance(x, (int, np.integer)) and isinstance(s, (int, np.integer)):
        v, e = int(x), int(s) + int(fine_bits)
        if e < 0:
            return v << (-e)
        return -((-v) >> e) if v < 0 else v >> e
    a = np.asarray(x, dtype=np.int64)
    e = np.asarray(s, dtype=np.int64) + np.int64(fine_bits)
    pos = e >= 0
    mag = np.abs(a)
    d = np.where(pos, mag >> np.where(pos, e, 0), mag << np.where(pos, 0, -e))
    return np.where(a < 0, -d, d)


def dequantize_heat(raw, fine_bits: int):
    """A sweep plane (or one value of it, or a sum of them) as a float in HEAT
    COUNTS: raw / 2^fine_bits, exact in float64 for every value a plane holds.
    RENDER / readout / report only (the dequantize convention: a fresh float
    copy, never written back); a sim-path reader uses fine_heat_shr."""
    return np.asarray(raw, dtype=np.float64) / float(1 << int(fine_bits))
