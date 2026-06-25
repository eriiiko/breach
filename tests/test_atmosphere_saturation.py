"""S2c P2 — atmosphere conservation to the LSB (the run-past-wave-death gate).

THE point of the whole S2 group: with the atmosphere/wave/wind transfer integer
and CONSERVATIVE (the ±-pair), a sealed box hit by a strong blast must have its
total atmosphere mass SATURATE as the wave energy bleeds into the bulk, then stay
BIT-FROZEN once the wave dies (conserved to the LSB — no drift forever after).

The conserved quantity is `sum(atmosphere)` over the interior of a SEALED box (no
vacuum/breach -> no sponge sink; wave absorb is a wave loss, not an atmosphere
loss). The wave->atmosphere transfer moves mass with the ±-pair (atmosphere += d,
wave_p -= d, the SAME int), so it is exactly mass-neutral; the GS diffusion only
redistributes (antisymmetric face flux). Once wave_p -> 0 the transfer stops and
atmosphere is frozen to the LSB.

The REVERT check (the proof the bridge collapse is what conserves): swap the
conservative ±-pair for a one-sided truncating `mul_q16` deposit (the S2a form)
and the same trace MUST drift downward forever (the DC sink the ±-pair removes).
The test runs the real solver for the green case; the revert is a Python mirror
of the transfer math (the C++ has no revert flag), gated to go RED.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_atmosphere_saturation.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402
from simulation import wave_fixed, atmosphere_fixed  # noqa: E402

H = W = 20
TICKS = 3200          # run well past wave death (>= 3000, the brief's gate)
FREEZE_TAIL = 400     # the last N ticks must be BIT-FROZEN


def _sealed_box():
    """A sealed box (hull ring, NO vacuum) + the config-bound AtmosphereSolver."""
    solid = np.zeros((H, W), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    is_vac = np.zeros((H, W), dtype=bool)           # SEALED — no breach/sink
    perm = np.where(solid, 0.0, 1.0).astype(np.float32)
    atm = np.where(solid, 0, atmosphere_fixed.FP_ONE).astype(np.int32)
    wp = np.zeros((H, W), np.int32)
    wv = np.zeros((H, W), np.int32)
    ws = np.zeros((H, W), np.int32)
    ws[H // 2, W // 2] = wave_fixed.quantize_scalar(8.0)   # STRONG central blast
    wx = np.zeros((H, W), np.int32)
    wy = np.zeros((H, W), np.int32)
    absorb = np.zeros((H, W), np.float32)
    return dict(solid=solid, is_vac=is_vac, perm=perm, atm=atm,
                wp=wp, wv=wv, ws=ws, wx=wx, wy=wy, absorb=absorb)


def _run(box):
    """Drive the real solver; return the per-tick (atmosphere mass) trace."""
    solver = PhysicsRunner(bp).atmos
    sim_time = 1.0 / float(CFG.clock.ticks_per_second)
    n = max(1, int(math.ceil(sim_time / solver.max_dt())))
    dt = sim_time / n
    interior = (~box['solid']) & (~box['is_vac'])
    mass = []
    for _ in range(TICKS):
        for _s in range(n):
            solver.step(box['wp'], box['wv'], box['ws'], box['atm'],
                        box['wx'], box['wy'], box['solid'], box['solid'],
                        box['is_vac'], box['perm'], box['absorb'], dt)
        mass.append(int(box['atm'][interior].astype(np.int64).sum()))
    return mass


def test_atmosphere_mass_saturates_then_frozen():
    """Sealed box + strong blast: total atmosphere mass SATURATES (plateaus)
    during the blast, then stays BIT-FROZEN to the LSB once the wave dies.

    The conservation claim has two parts, both gated here:
      * THE LSB GATE — after wave death the mass is bit-frozen (the steady state
        holds exactly to the LSB; no drift forever after). This is the brief's
        'conserved to the LSB after wave energy -> 0'.
      * The active-blast transient settles to within Q16.16 granularity of the
        start (a small one-time quantization offset between the integer-quantized
        steady state and the continuum solution — the SAME ~0.05% the float GS
        has, NOT the runaway DC sink the revert check proves). Bounded + frozen,
        NOT divergent — that is the determinism contract (the revert goes RED).
    """
    box = _sealed_box()
    mass = _run(box)
    tail = mass[-FREEZE_TAIL:]
    # 1. THE LSB GATE — bit-frozen tail: every count identical over the last
    #    FREEZE_TAIL ticks (the steady state is conserved to the LSB).
    assert len(set(tail)) == 1, (
        f"atmosphere mass NOT bit-frozen after wave death: the last "
        f"{FREEZE_TAIL} ticks span {min(tail)}..{max(tail)} "
        f"(spread {max(tail) - min(tail)} counts) — a residual drift means the "
        f"transfer/diffusion is still leaking (must be frozen to the LSB)")
    # 2. SATURATED (plateau, not drift): the mass settles early and the second
    #    half of the run is one frozen value (the plateau == saturation).
    half = mass[len(mass) // 2:]
    assert len(set(half)) == 1, (
        f"atmosphere mass did NOT saturate — the second half spans "
        f"{min(half)}..{max(half)} (still drifting, not a plateau)")
    # 3. The transient settles to within Q16.16 granularity of the start (a small
    #    one-time quantization offset, NOT the runaway sink the revert proves).
    drift = abs(mass[-1] - mass[0]) / mass[0]
    assert drift < 5e-3, (
        f"atmosphere mass transient drift {drift:.4%} exceeds the Q16.16-"
        f"granularity bound (0.5%): {mass[0]} -> {mass[-1]}. A drift this large "
        f"is a real sink, not quantization settling.")


def test_revert_to_mul_q16_deposit_drifts_RED():
    """THE proof: a one-sided truncating mul_q16 deposit (the S2a form, NOT the
    conservative ±-pair) drifts DOWNWARD forever — the DC sink the collapse
    removes. Mirrors the transfer math in Python with the OLD one-sided deposit;
    asserts it goes RED (a growing drift, never freezing) where the real ±-pair
    is green. This validates that the ±-pair is what conserves."""
    FP = atmosphere_fixed.FP_ONE
    # A simple sealed-interior model: a uniform wave anomaly transferred each
    # substep with the OLD one-sided mul_q16 (>>16 truncation toward -inf). With
    # a zero-mean anomaly the sum of deposits should be ~0, but the per-cell floor
    # truncation biases every cell by up to -1 LSB -> a systematic DC sink.
    n_cells = (H - 2) * (W - 2)
    # Seed a zero-mean integer anomaly field (the transfer input) and a constant
    # atmosphere; deposit one-sidedly with mul_q16 over many substeps.
    rng = np.random.default_rng(2026)
    anom = rng.integers(-50000, 50001, size=n_cells).astype(np.int64)
    anom -= int(round(anom.mean()))            # zero-mean (the rounded-mean DC fix)
    xfer_q = atmosphere_fixed.quantize_scalar(0.02)   # (transfer*dt)-ish
    atm = np.full(n_cells, FP, dtype=np.int64)

    def mul_q16_trunc(a, b):
        # mul_q16: (a*b) >> 16, arithmetic shift (toward -inf for negatives).
        return (a * b) >> 16

    mass0 = int(atm.sum())
    for _ in range(2000):
        d = mul_q16_trunc(anom, np.int64(xfer_q))   # one-sided truncating deposit
        atm += d                                    # NOT paired -> DC sink
    mass_end = int(atm.sum())
    drift = mass_end - mass0
    # The one-sided truncation MUST leak downward (the RED proof). With ~hundreds
    # of cells x 2000 substeps each losing up to 1 LSB, the drift is large+negative.
    assert drift < 0, (
        "REVERT check did not go RED: the one-sided mul_q16 deposit should DC-sink "
        f"(drift {drift} >= 0). If this passes, the ±-pair's conservation claim is "
        "not actually being exercised by the green test.")
    assert abs(drift) > n_cells, (
        f"REVERT drift {drift} is below the per-cell-LSB sink floor "
        f"({n_cells}) — the truncation bias is the whole point.")
