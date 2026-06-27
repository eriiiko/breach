"""S3a headline gate — the ignition O2 predicate matches the C++ fire O2 gate.

The design invariant (combat.py docstring §"O2 check reuses the existing fire
semantics", combat.py:276-281): a tile must not ignite into a state the fire
step would immediately suffocate. So the Python ``apply_temperature_ignition``
O2 gate and the C++ ``FireSimulation::step`` O2 gate (``P`` neighbour-mean,
fire_simulation.cpp:72-82) MUST agree on the SAME predicate
(``mean(atmosphere over open neighbours) >= o2_threshold``, with
``o2_threshold == P_min == 0.60``).

S3a makes the Python O2 mean an INTEGER reduction on the int32 Q16.16 atmosphere
(int64 neighbour-sum + ``mean_round`` round-half-away-from-zero, over the
``~solid & ~is_vacuum`` open-neighbour mask — the EXACT C++ mask). This test
proves the two predicates agree across a sweep of atmosphere values, both for
homogeneous and heterogeneous neighbours.

THE ONE SUBTLETY (verified, documented, expected): the C++ ``P`` gate is STILL
FLOAT this commit (S3a flips the representation + the Python twin; the C++
logistic — including ``P`` — goes integer in S3b). The integer ``mean_round`` and
the C++ float32 division agree EVERYWHERE except at an EXACT threshold tie, where
the integer mean lands precisely on ``quantize(0.60)`` (so ``>=`` is True) while
the float32 sum/division rounds the exact-0.60 mean a hair BELOW 0.60 (so ``>=``
is False). At those ties the INTEGER gate is the correct, deterministic one — an
exact-0.60 mean genuinely meets a ``>= 0.60`` threshold — and S3b makes the C++
adopt the same integer ``mean_round``, eliminating the discrepancy bit-for-bit.
So the test asserts: (a) agreement on every NON-tie case, and (b) every
disagreement is an exact threshold tie (``integer mean == quantize(threshold)``)
— never a real divergence.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fire_o2_invariant.py -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from config import CFG  # noqa: E402
from simulation import atmosphere_fixed, fire_fixed  # noqa: E402
from simulation.combat import apply_temperature_ignition  # noqa: E402
from simulation.materials import MAT_AIR, MAT_WOOD, MaterialTable  # noqa: E402

_TBL = MaterialTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])
O2_THRESHOLD = float(getattr(CFG.physics.fire, "o2_threshold", 0.60))
IGN_SEED = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
THR_Q = fire_fixed.quantize_scalar(O2_THRESHOLD)


# ---------------------------------------------------------------------------
# Reference predicates: the C++ fire P gate (FLOAT, fire_simulation.cpp:72-82)
# and the Python integer gate (the mean_round reduction S3a installed).
# ---------------------------------------------------------------------------
def cpp_float_o2_gate(neigh_qs):
    """The C++ FireSimulation P gate, replicated faithfully in float32.

    The C++ reads each open neighbour's atmosphere as ``dequantize_f(atm_q)``
    (an int32 Q16.16 -> float32 bridge in step_tail), sums them in float, divides
    by the count, and the survival/ignition predicate is ``P >= o2_threshold``."""
    s = np.float32(0.0)
    for q in neigh_qs:
        s = np.float32(s + np.float32(float(q) / 65536.0))   # dequantize_f
    P = np.float32(s / np.float32(len(neigh_qs)))
    return bool(P >= np.float32(O2_THRESHOLD))


def py_int_o2_mean(neigh_qs):
    """The S3a Python integer O2 mean: int64 neighbour-sum + mean_round
    (round-half-away-from-zero), the EXACT reduction in apply_temperature_ignition
    (and the one the C++ fire adopts in S3b)."""
    s = np.int64(sum(int(q) for q in neigh_qs))
    count = np.int64(len(neigh_qs))
    half = count // 2
    return int((s + half) // count if s >= 0 else (s - half) // count)


def py_int_o2_gate(neigh_qs):
    return py_int_o2_mean(neigh_qs) >= THR_Q


# ---------------------------------------------------------------------------
# (1) The invariant: the two gates agree on every non-tie case; every
#     disagreement is an EXACT threshold tie (integer mean == quantize(thr)).
# ---------------------------------------------------------------------------
def test_o2_gate_homogeneous_sweep_bit_identical():
    """Homogeneous neighbours (all at the same atmosphere) across a fine sweep
    that straddles the 0.60 threshold — the gates must agree EXACTLY (a uniform
    mean is exact under both float division and mean_round, so there are no
    ties to worry about here)."""
    for atm in np.linspace(0.40, 0.80, 4001):
        q = atmosphere_fixed.quantize_scalar(float(atm))
        for count in (1, 2, 3, 4):
            combo = (q,) * count
            assert cpp_float_o2_gate(combo) == py_int_o2_gate(combo), (
                f"homogeneous O2 gate disagreement at atm={atm:.6f} count={count}")


def test_o2_gate_heterogeneous_disagreements_are_exact_ties_only():
    """Heterogeneous neighbours near the threshold — the HARD case where the
    integer mean_round and the C++ float32 division CAN differ. Assert that every
    disagreement is an EXACT threshold tie (integer mean == quantize(0.60)), never
    a real divergence — i.e. the integer gate is the correct/deterministic one and
    S3b's C++ mean_round will bit-match it."""
    vals = np.linspace(0.50, 0.70, 41)        # densely straddles 0.60
    qs = [atmosphere_fixed.quantize_scalar(float(v)) for v in vals]
    n_disagree = 0
    n_disagree_not_tie = 0
    for count in (1, 2, 3, 4):
        for combo in itertools.product(qs, repeat=count):
            if cpp_float_o2_gate(combo) != py_int_o2_gate(combo):
                n_disagree += 1
                # Every legitimate disagreement is an EXACT tie: the integer mean
                # equals the quantized threshold exactly (so the integer >= is
                # True, the float < is False due to float32 rounding of an
                # exact-0.60 mean). Anything else is a real divergence -> a bug.
                if py_int_o2_mean(combo) != THR_Q:
                    n_disagree_not_tie += 1
    assert n_disagree_not_tie == 0, (
        f"{n_disagree_not_tie} O2-gate disagreements were NOT exact threshold "
        f"ties — the integer O2 mean diverges from the C++ P gate by more than the "
        f"float32-rounding-at-the-tie epsilon (a real bug, not the expected "
        f"transitional float-vs-int boundary)")
    # Sanity: there ARE ties in this sweep (the sweep was built to hit 0.60), so
    # the test is actually exercising the boundary, not vacuously passing.
    assert n_disagree > 0, (
        "the heterogeneous sweep produced no threshold ties — it is not "
        "exercising the boundary where the gates could differ")


# ---------------------------------------------------------------------------
# (2) Drive the PRODUCTION function (apply_temperature_ignition) and confirm its
#     ignite decision matches the C++ float O2 gate on a hot flammable tile.
# ---------------------------------------------------------------------------
class _Cross3x3:
    """A flammable wood centre on a 3x3 with an air ring; set the ring atmosphere
    (the O2 source) and a hot centre, then run apply_temperature_ignition and see
    whether [1,1] ignited — the real production path."""

    def __init__(self, ring_atm):
        m = np.full((3, 3), MAT_AIR, dtype=np.int8)
        m[1, 1] = MAT_WOOD
        self.materials = _TBL
        self.material = m
        self.flammable = _TBL.flammable[m]
        self.solid = (_TBL.permeability[m] <= 0.0)
        self.is_vacuum = np.zeros((3, 3), dtype=bool)
        a = np.where(self.solid, 0,
                     atmosphere_fixed.quantize_scalar(float(ring_atm))).astype(np.int32)
        self.atmosphere = a
        self.temperature = np.full((3, 3), IGN_WOOD_Q16 * 2, dtype=np.int32)  # hot
        self.fire = np.zeros((3, 3), dtype=np.int32)


def test_production_ignition_matches_cpp_gate_off_tie():
    """Sweep the ring atmosphere AWAY from the exact tie (so the float and int
    gates agree by construction) and confirm the production ignition decision
    matches the C++ float O2 gate at each point. (The on-tie behaviour is the
    integer-canonical one tested above; here we exercise the real function.)"""
    for atm in np.linspace(0.40, 0.80, 401):
        # Skip a tiny band around the exact threshold (the integer-canonical tie
        # zone, covered by the dedicated test above).
        if abs(atm - O2_THRESHOLD) < 1e-3:
            continue
        g = _Cross3x3(ring_atm=float(atm))
        # The four air-ring neighbours of [1,1] are all at `atm`.
        q = atmosphere_fixed.quantize_scalar(float(atm))
        cpp_decision = cpp_float_o2_gate((q, q, q, q))
        apply_temperature_ignition(g, O2_THRESHOLD, IGN_SEED)
        py_ignited = bool(g.fire[1, 1] > 0)
        assert py_ignited == cpp_decision, (
            f"production ignition ({py_ignited}) != C++ O2 gate ({cpp_decision}) "
            f"at ring atm={atm:.5f}")
