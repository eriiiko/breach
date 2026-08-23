"""Drag-law v2 (docs/drag_law_v2_design_2026-08-23.md §2/§8 gate 6) — the
stage-Q LAW gate.

Bit-identity between the CPU reference and the GPU kernel (tests/
cuda_kick_check.py) proves the two mirrors AGREE, not that either
implements the right formula — both could transcribe the same wrong law and
every cross-mirror gate would still pass (this is the "both-mirrors-wrong"
hole neither the P1 v1 draft nor a plain lockstep gate can close). This gate
replays the EXACT integer chain independently, in Python, against
``eos_kick_compression_ref`` on a random-plus-corners field:

    denom = 65536 + ((kd2_q * isqrt(rad1)) >> 16)
    u'    = trunc(u * 65536 / denom)

``math.isqrt`` is Python's exact floor(sqrt) for a non-negative int —
bit-identical to ``fixedpoint::sqrt_q16`` over any range that does not hit
its INT32_MAX self-guard (nowhere near it at these velocities). Also
asserts per-component sign preservation, ``|u'| <= |u|``, and the one-tick
ceiling (design §1c/§4): from ``|u| = U_MAX = 1000``, one armed tick at
k_drag2=1 lands below ``u_ceil = 1/(k_drag2*dt) = 24``. CPU-only (no CUDA
build required) — this gate is independent of the GPU mirror by design.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402

FP_ONE = 65536

CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    n_floor_solver=1e-3, t_min=-289.0, t_work_clamp=0.5,
    t_max_phys=16000.0, u_max=1000.0,
    k_drag=0.0, k_drag_heat_frac=1.0, c_v=1.0,
)


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _kd2_q(k_drag2, dt):
    return int(_q(k_drag2 * dt))


def _trunc_div(a, b):
    """C++ trunc-toward-0 integer division (Python's // truncs toward -inf)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _replay_stage_q(ux, uy, kd2_q):
    """Design §2's exact stage-Q formula, replayed in Python bignums. Note
    this is the FULL formula with no explicit skip branch: when rad1 is
    below the dead zone, prod == 0 by construction, so denom == 65536 and
    the divide is an EXACT identity — the fast path (design §7) is a pure
    optimization of this same expression, so replaying it unconditionally
    is faithful to the real (skip-branched) implementation in ALL cases."""
    rad1 = ux * ux + uy * uy
    umag = math.isqrt(rad1)
    prod = (kd2_q * umag) >> 16
    denom = FP_ONE + prod
    return _trunc_div(ux * FP_ONE, denom), _trunc_div(uy * FP_ONE, denom)


def _run_ref(wind_x, wind_y, temperature, cap2, k_drag2, dt):
    h, w = wind_x.shape
    p_new = np.full((h, w), _q(1.0), dtype=np.int32)   # uniform -> zero grad(P)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0] = _q(0.21)
    gas[1] = _q(0.79)
    gas_conservative = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    wx, wy, t = wind_x.copy(), wind_y.copy(), temperature.copy()
    bp.eos_kick_compression_ref(
        wx, wy, t, p_new, gas, gas_conservative, solid, is_vacuum, absorb,
        dt, cap2, k_drag2=k_drag2, **CONSTS)
    return wx, wy


@pytest.mark.parametrize("k_drag2", [1.0, 0.01, 0.25])
def test_stage_q_law_matches_the_exact_integer_chain(k_drag2):
    dt = 1.0 / 24.0
    kd2_q = _kd2_q(k_drag2, dt)
    assert kd2_q > 0

    rng = np.random.default_rng(20260823)
    h = w = 12
    wind_x = _q((rng.random((h, w)) * 2 - 1) * 1200.0).astype(np.int32)
    wind_y = _q((rng.random((h, w)) * 2 - 1) * 1200.0).astype(np.int32)
    # Random-PLUS-corners: the four sign quadrants, the origin, and a
    # near-U_MAX axis-aligned case, deliberately placed (not left to fuzz).
    wind_x[0, 0], wind_y[0, 0] = _q(500.0), _q(500.0)      # ++
    wind_x[0, 1], wind_y[0, 1] = _q(500.0), _q(-500.0)     # +-
    wind_x[0, 2], wind_y[0, 2] = _q(-500.0), _q(500.0)     # -+
    wind_x[0, 3], wind_y[0, 3] = _q(-500.0), _q(-500.0)    # --
    wind_x[0, 4], wind_y[0, 4] = 0, 0                       # origin
    wind_x[0, 5], wind_y[0, 5] = _q(999.0), 0                # near U_MAX, axis-aligned
    temperature = np.full((h, w), _q(20.0), dtype=np.int32)
    cap2 = np.full((h, w), int(_q(2300.0)) ** 2, dtype=np.int64)   # clamp stays clear

    wx_ref, wy_ref = _run_ref(wind_x, wind_y, temperature, cap2, k_drag2, dt)

    for y in range(h):
        for x in range(w):
            ux0, uy0 = int(wind_x[y, x]), int(wind_y[y, x])
            ex, ey = _replay_stage_q(ux0, uy0, kd2_q)
            got = (int(wx_ref[y, x]), int(wy_ref[y, x]))
            assert (ex, ey) == got, (
                f"cell ({y},{x}) u0=({ux0},{uy0}): law replay ({ex},{ey}) "
                f"!= eos_kick_compression_ref {got}")
            assert (ex == 0) or (np.sign(ex) == np.sign(ux0)), (
                f"cell ({y},{x}): stage Q flipped ux's sign")
            assert (ey == 0) or (np.sign(ey) == np.sign(uy0)), (
                f"cell ({y},{x}): stage Q flipped uy's sign")
            assert ex * ex + ey * ey <= ux0 * ux0 + uy0 * uy0, (
                f"cell ({y},{x}): stage Q grew |u| (must shrink-only)")


def test_one_tick_ceiling_from_u_max():
    """Design §1c/§4: from |u|=U_MAX=1000 m/s, one armed tick at k_drag2=1
    lands below u_ceil = 1/(k_drag2*dt) = 24 m/s (dt=1/24)."""
    dt = 1.0 / 24.0
    k_drag2 = 1.0
    kd2_q = _kd2_q(k_drag2, dt)

    h = w = 4
    wind_x = np.full((h, w), _q(1000.0), dtype=np.int32)   # axis-aligned, |u| == U_MAX exactly
    wind_y = np.zeros((h, w), dtype=np.int32)
    temperature = np.full((h, w), _q(20.0), dtype=np.int32)
    # cap == U_MAX exactly: rad == cap2 (not >), so the |u| clamp does NOT
    # fire (strict >) -- u enters stage Q at exactly 1000 m/s, untouched.
    cap2 = np.full((h, w), int(_q(1000.0)) ** 2, dtype=np.int64)

    wx_ref, _wy_ref = _run_ref(wind_x, wind_y, temperature, cap2, k_drag2, dt)
    u_ceil_raw = int(_q(24.0))
    assert np.all(np.abs(wx_ref) < u_ceil_raw), (
        "one armed tick from U_MAX did not land below u_ceil=24 m/s "
        f"(max |u|={float(np.abs(wx_ref).max()) / FP_ONE:.4f} m/s)")

    ex, ey = _replay_stage_q(int(wind_x[0, 0]), 0, kd2_q)
    assert ex == int(wx_ref[0, 0])
    assert abs(ex) < u_ceil_raw
