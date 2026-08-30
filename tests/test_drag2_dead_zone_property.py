"""Drag-law v2 (docs/drag_law_v2_design_2026-08-23.md §4/§8 gate 4) — the
k_drag2 dead-zone property gate.

Stage Q's calm-cell fast path (design §7) skips the divide exactly when
``prod = trunc(kd2_q*umag/2^16) == 0``, i.e. when ``umag < U0 = ceil(2^16/
kd2_q)``. This gate LOCATES U0 from the LIVE INTEGER CHAIN (never from a
printed/hand-derived number — R5) by direct integer search on the exact
``mul128_shr(kd2_q, umag, 16)`` expression, then constructs components
straddling it — including negative components and the diagonal case (§4) —
and asserts: cells strictly below U0 are bit-unchanged by stage Q (the skip
is an EXACT no-op) while cells at/above U0 change (the divide actually
fired). Drives ``eos_kick_compression_ref`` directly (the CPU P6.4
reference — the same loop ``EOSSolver::step`` runs), CPU-only (no CUDA
build required).
"""
from __future__ import annotations

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

# k_drag/heat stay at their dormant defaults throughout this module so the
# ONLY mechanism ever live is stage Q itself — nothing here confounds with
# stage L or the heat deposit.
CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    # arc #54 P-G1a (MECHANICAL): `t_work_clamp` and `k_drag_heat_frac` left
    # the kick reference's signature (design D11 / D5). Stage Q's dead-zone
    # threshold -- the property this module measures -- is untouched.
    n_floor_solver=1e-3, t_min=-289.0,
    t_max_phys=16000.0, u_max=1000.0,
    k_drag=0.0, c_v=1.0,
)


def _q(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _kd2_q(k_drag2, dt):
    return int(_q(k_drag2 * dt))


def _find_u0_raw(kd2_q):
    """The live-chain floor: the smallest positive integer ``umag_raw`` with
    ``mul128_shr(kd2_q, umag_raw, 16) > 0`` — i.e. ``(kd2_q*umag_raw) >> 16
    > 0`` — located by direct integer search (kd2_q is small for this
    module's dials, so U0 is at most a few thousand: dozens of iterations),
    never from a printed/derived-by-hand constant (design R5)."""
    assert kd2_q > 0
    umag = 1
    while (kd2_q * umag) >> 16 == 0:
        umag += 1
    return umag


def _run_kick(wind_x, wind_y, temperature, cap2, k_drag2, dt=1.0 / 24.0):
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
    res = bp.eos_kick_compression_ref(
        wx, wy, t, p_new, gas, gas_conservative, solid, is_vacuum, absorb,
        dt, cap2, k_drag2=k_drag2, **CONSTS)
    return wx, wy, t, res


@pytest.mark.parametrize("k_drag2", [1.0, 0.01, 0.5])
def test_dead_zone_floor_matches_the_live_integer_chain(k_drag2):
    dt = 1.0 / 24.0
    kd2_q = _kd2_q(k_drag2, dt)
    assert kd2_q > 0, "test dial must quantize to a live kd2_q"
    u0 = _find_u0_raw(kd2_q)

    h = w = 8
    below = u0 - 1                      # the largest raw magnitude that skips
    above = u0 + 5                      # clear of the boundary (avoid isqrt-adjacent noise)
    diag_below = max(1, int(below / 1.41422))   # each component ~u0/sqrt(2)
    diag_above = int(above * 0.9)

    wind_x = np.zeros((h, w), dtype=np.int32)
    wind_y = np.zeros((h, w), dtype=np.int32)
    wind_x[0, :] = below                          # axis, below, positive
    wind_x[1, :] = -below                          # axis, below, negative
    wind_x[2, :], wind_y[2, :] = diag_below, -diag_below   # diagonal, below, mixed sign
    wind_x[3, :] = above                           # axis, above, positive
    wind_x[4, :] = -above                          # axis, above, negative
    wind_x[5, :], wind_y[5, :] = diag_above, -diag_above   # diagonal, above, mixed sign

    temperature = np.full((h, w), _q(20.0), dtype=np.int32)
    cap2 = np.full((h, w), int(_q(2300.0)) ** 2, dtype=np.int64)   # far above any of these speeds

    wx_on, wy_on, t_on, _ = _run_kick(wind_x, wind_y, temperature, cap2, k_drag2, dt)
    wx_off, wy_off, t_off, _ = _run_kick(wind_x, wind_y, temperature, cap2, 0.0, dt)

    for r in (0, 1, 2):
        assert np.array_equal(wx_on[r], wx_off[r]) and np.array_equal(wy_on[r], wy_off[r]), (
            f"k_drag2={k_drag2}: row {r} (|u| <= {below} < U0={u0}) was NOT "
            "bit-unchanged by stage Q -- the calm-cell fast path is not exact")

    for r in (3, 4, 5):
        changed = (not np.array_equal(wx_on[r], wx_off[r])) or \
                  (not np.array_equal(wy_on[r], wy_off[r]))
        assert changed, (
            f"k_drag2={k_drag2}: row {r} (|u| >= {above} > U0={u0}) was "
            "unchanged -- stage Q never engaged (gate is vacuous)")
        # Sign preservation (per component; 0 stays 0) + shrink-only.
        for arr_on, arr_in in ((wx_on[r], wind_x[r]), (wy_on[r], wind_y[r])):
            same_sign = np.all((arr_on == 0) | (np.sign(arr_on) == np.sign(arr_in)))
            assert same_sign, f"row {r}: stage Q flipped a component's sign"
        mag_on = wx_on[r].astype(np.int64) ** 2 + wy_on[r].astype(np.int64) ** 2
        mag_in = wind_x[r].astype(np.int64) ** 2 + wind_y[r].astype(np.int64) ** 2
        assert np.all(mag_on <= mag_in), f"row {r}: stage Q grew |u| (must shrink-only)"

    # Temperature must be untouched throughout (k_drag_heat_frac's deposit
    # needs a LIVE k_drag/k_drag2-driven du2 booking with heat_frac>0 to
    # write T at all here it would, since k_drag_heat_frac=1.0 is the
    # module default -- so assert the CALM rows specifically saw no T write
    # either, matching the fast path being an exact structural no-op).
    assert np.array_equal(t_on[0:3], t_off[0:3])


def test_u0_monotonicity_matches_design_worked_examples():
    """Cross-check the live-chain U0 search against the design doc's own
    worked anchors (§2/§8 gate 2): U0=24 at k_drag2=1.0, U0=2428 at
    k_drag2=0.01 (both at dt=1/24) -- pins that this test module's derivation
    matches the design's, so a future dt/tick-rate change is caught here
    too (R5: gates derive from the live chain, but the anchors are still
    worth a direct pin against the shipped values the design cites)."""
    dt = 1.0 / 24.0
    assert _find_u0_raw(_kd2_q(1.0, dt)) == 24
    assert _find_u0_raw(_kd2_q(0.01, dt)) == 2428
    # Monotonicity (design §4): cranking k2 SHRINKS the floor.
    u0_small_k2 = _find_u0_raw(_kd2_q(0.01, dt))
    u0_big_k2 = _find_u0_raw(_kd2_q(1.0, dt))
    assert u0_big_k2 < u0_small_k2
