"""Q16.16 boundary helpers for the SYNCED UNIT STATE (Q2-lift).

The unit fields the lockstep digest watches (``facing``, ``current_hp``) were
the last synced state still produced by machine-dependent float paths:

* ``facing = math.atan2(...)`` — libm; differs at the last ULP across
  CRT/Python versions (py3.11 desktop vs py3.12 Lenovo), and the digest's
  quantization amplifies that ULP into a hash flip (the X-ARCH Ada finding,
  docs/xarch_ada_beatB_findings_2026-06-29.md).
* combat bullet trajectories (``math.cos/sin`` per bullet) — the same libm
  hazard, feeding hit/miss -> ``current_hp`` -> kills.
* HP damage deltas — plain float ``*``/``-`` (IEEE-exact, already stable);
  quantized anyway as belt-and-suspenders so every delta is an exact multiple
  of 1/65536 (docs/q2_lift_spec.md Patch 2).

This module is the ONE boundary: quantize float radians/deltas to Q16.16,
run the PURE-INTEGER kit (``fixed_point.h`` atan2_q16/sin_q16/cos_q16, pybind
via ``breach_physics``), and dequantize back to floats that are exact n/65536
doubles — bit-identical on every machine, compiler, and Python version.

There is deliberately NO Python fallback for the trig: a second
implementation could drift from the C++ kit and silently desync. If
``breach_physics`` is not importable when the first trig call happens, that
is a broken configuration and it fails LOUDLY. (The import is lazy so that
merely importing ``simulation.unit`` / ``simulation.combat`` — e.g. from
asset tools that never step physics — does not require the compiled module.)

The scalar quantize/dequantize twins below mirror ``fixedpoint::quantize`` /
``dequantize`` exactly (round-half-away-from-zero; /65536 is exact), the same
documented-twin pattern as ``fire_fixed`` / ``gas_fixed`` / ``wave_fixed``.

Accuracy of the trig kit (pinned in tests/test_fixed_trig.py): <= 9.0e-6 rad
for atan2, <= 9.0e-6 for sin/cos — the Q16.16 quantization floor. Facing
therefore moves by <= ~1.5e-5 rad vs libm (imperceptible; Erik pre-approved,
no feel-check — the Q2-lift spec).
"""
from __future__ import annotations

import math

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT          # 65536
FP_ONE_F = float(FP_ONE)

# Lazily-bound compiled module (see module docstring: lazy on purpose, loud on
# failure, never twinned).
_bp = None


def _kit():
    global _bp
    if _bp is None:
        import breach_physics as _m   # hard requirement at first trig call
        _bp = _m
    return _bp


def quantize_scalar(value: float) -> int:
    """Float -> Q16.16 int, round-half-away-from-zero.

    Twin of ``fixedpoint::quantize`` (C++: ``(q16)(v*65536 +/- 0.5)`` — the
    cast truncates toward zero, so +0.5-then-floor / -0.5-then-ceil). The
    *65536 scaling is a power of two -> EXACT in float64; the +/-0.5 add and
    the floor/ceil are IEEE-deterministic. Caller owns range safety (|value|
    < 32768): the trig wrappers below pass tile deltas and radians, far
    inside; an out-of-range int32 is rejected LOUDLY by pybind, not wrapped.
    """
    v = float(value) * FP_ONE_F
    return int(math.floor(v + 0.5) if v >= 0.0 else math.ceil(v - 0.5))


def dequantize_scalar(q: int) -> float:
    """Q16.16 int -> float (exact: n/65536 is a power-of-two divide)."""
    return q / FP_ONE_F


def atan2_rad(y: float, x: float) -> float:
    """Deterministic atan2: float in, float out, PURE INTEGER in between.

    Quantizes both args (only their ratio + signs matter — a common scale
    factor drops out of atan2), runs ``atan2_q16``, dequantizes. Result is an
    exact n/65536 double in [-3.14159, +3.14159] ([-quantize(pi), +] / 2^16).
    Deltas that BOTH quantize to 0 (|d| < 1/131072) return 0.0 — the kit's
    pinned atan2(0,0) — deterministic, and unreachable from real movement
    steps. Replaces ``math.atan2`` on every synced-state path.
    """
    b = _kit()
    return dequantize_scalar(b.atan2_q16(quantize_scalar(y), quantize_scalar(x)))


def atan2_q16(y_q: int, x_q: int) -> int:
    """The kit's ``atan2_q16`` on Q16.16 ints, Q16.16 radians out -- the
    integer form of :func:`atan2_rad` (``atan2_rad(dequantize(y_q),
    dequantize(x_q)) == dequantize_scalar(atan2_q16(y_q, x_q))`` exactly,
    since quantizing an exact n/65536 double gives n back). The turn-rate
    law (``simulation/turning.py``) works in this currency."""
    return int(_kit().atan2_q16(int(y_q), int(x_q)))


def atan2_q16_of(y: float, x: float) -> int:
    """:func:`atan2_rad` with the Q16.16 radians left as the integer (the
    turning law's currency, G5p-a1): ``quantize_scalar(atan2_rad(y, x))``
    exactly."""
    return atan2_q16(quantize_scalar(y), quantize_scalar(x))


def cos_q16(angle_q: int) -> int:
    """The kit's ``cos_q16`` on a Q16.16 radian int, Q16.16 out -- the
    integer form of :func:`cos_rad` (G1p-c: the speed law's direction
    factor, ``movement.facing_cos_q16``)."""
    return int(_kit().cos_q16(int(angle_q)))


def sin_q16(angle_q: int) -> int:
    """The kit's ``sin_q16`` on a Q16.16 radian int, Q16.16 out -- the
    integer form of :func:`sin_rad` (G3p: the direction a throw leaves the
    hand, the facing's unit vector)."""
    return int(_kit().sin_q16(int(angle_q)))


def sin_rad(angle: float) -> float:
    """Deterministic sin of ``angle`` (radians): quantize -> sin_q16 ->
    dequantize. Exact n/65536 double in [-1.0, 1.0]. Accuracy pinned for
    |angle| <= 4pi (every caller passes (-pi-cone, 2pi+eps))."""
    b = _kit()
    return dequantize_scalar(b.sin_q16(quantize_scalar(angle)))


def cos_rad(angle: float) -> float:
    """Deterministic cos of ``angle`` (radians) — see :func:`sin_rad`."""
    b = _kit()
    return dequantize_scalar(b.cos_q16(quantize_scalar(angle)))


def quantize_hp_delta(dmg: float) -> float:
    """Snap a damage delta to the Q16.16 grid: ``dequantize(quantize(dmg))``.

    Belt-and-suspenders for the ``current_hp -=`` sites (q2_lift_spec Patch 2):
    plain float +/-/* is IEEE-correctly-rounded and already cross-machine
    stable, but snapping every applied delta to an exact multiple of 1/65536
    future-proofs HP against any later float-path change. Behaviour change
    <= 1/131072 HP per application (~7.6e-6 — imperceptible; pre-approved).
    Integer damages (bullets, blasts, melee) pass through EXACTLY unchanged.
    Pure Python float64 (the quantize twin) — no compiled module needed.
    """
    return dequantize_scalar(quantize_scalar(dmg))
