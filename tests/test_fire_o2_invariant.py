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

S3b CLOSES THE GAP (the headline this file now gates): the C++ ``P`` gate is NOW
INTEGER too — fire_simulation.cpp adopted the same ``mean_round`` the Python twin
uses. So the two agree EXACTLY (ZERO disagreements), INCLUDING negative-atmosphere
neighbour configs — see ``test_py_twin_matches_cpp_integer_gate_exactly_incl_negative``.
That required the negative-branch fix: C++ integer ``/`` truncates toward ZERO,
Python ``//`` FLOORS (toward -inf), so a naive twin diverges on a transiently
NEGATIVE neighbour sum (atmosphere CAN dip negative — wave forcing subtracts, no
hard >=0 clamp); the twin now emulates trunc-toward-zero so both bit-match on ALL
inputs.

The HISTORICAL S3a tests below compare the Python integer gate to the OLD C++
FLOAT gate (``cpp_float_o2_gate``) and document the tie-only gap S3b just closed:
the integer ``mean_round`` and the old C++ float32 division agreed EVERYWHERE
except at an EXACT threshold tie (integer mean == ``quantize(0.60)``, so ``>=`` is
True; the float32 mean rounds a hair below). Those tests are kept as the record of
why S3b's integer gate is the correct/deterministic one.

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


def cpp_mean_round(s, count):
    """Faithful scalar replica of C++ fixed_point.h::mean_round — the gate the C++
    fire P now uses (S3b). round-half-away-from-zero, sign-symmetric, with C++
    integer `/` TRUNCATING TOWARD ZERO on both branches (NOT floor toward -inf)."""
    s = int(s)
    count = int(count)
    if count <= 0:
        return 0
    half = count // 2
    num = (s + half) if s >= 0 else (s - half)
    # C++ integer division truncates toward zero.
    return int(num / count) if num >= 0 else -((-num) // count)


def py_int_o2_mean(neigh_qs):
    """The Python integer O2 mean as apply_temperature_ignition computes it (S3b):
    int64 neighbour-sum + the mean_round emulation WITH the negative-branch
    trunc-toward-zero fix (review carry-forward #2) so it bit-matches the C++
    mean_round on ALL inputs, including a transiently-negative neighbour sum."""
    s = np.int64(sum(int(q) for q in neigh_qs))
    count = np.int64(len(neigh_qs))
    half = count // 2
    if s >= 0:
        return int((s + half) // count)
    neg_num = s - half
    return int(-((-neg_num) // count))   # trunc toward zero (== C++ `/`)


def py_int_o2_gate(neigh_qs):
    return py_int_o2_mean(neigh_qs) >= THR_Q


def cpp_int_o2_gate(neigh_qs):
    """The S3b C++ fire P gate: integer mean_round on the open-neighbour sum, then
    a Q16.16 `>=` threshold compare (the predicate fire_simulation.cpp now runs)."""
    s = sum(int(q) for q in neigh_qs)
    return cpp_mean_round(s, len(neigh_qs)) >= THR_Q


# ---------------------------------------------------------------------------
# (1) The invariant: the two gates agree on every non-tie case; every
#     disagreement is an EXACT threshold tie (integer mean == quantize(thr)).
# ---------------------------------------------------------------------------
def test_o2_gate_homogeneous_sweep_bit_identical():
    """Homogeneous neighbours (all at the same value) across a fine sweep
    that straddles the threshold — the gates must agree EXACTLY (a uniform
    mean is exact under both float division and mean_round, so there are no
    ties to worry about here). These reference functions (cpp_float_o2_gate /
    py_int_o2_gate) are field-agnostic — EOS refactor P4 re-pointed the
    PRODUCTION predicate from atmosphere to the real N_O2 plane (and moved
    O2_THRESHOLD from ~0.60 to ~0.12, config.toml [physics.fire]), so the
    sweep is rescaled to bracket the NEW threshold; the invariant proven is
    unchanged. UNLIKE the old 0.60 threshold, the new ~0.12 threshold's own
    quantize-then-dequantize happens to round DOWN (7864/65536 < 0.12), so a
    homogeneous mean can land EXACTLY on that boundary and hit the same
    float32-vs-integer tie the heterogeneous test documents below — skip that
    single point (a coincidence of THE THRESHOLD's own rounding direction,
    not a new class of disagreement)."""
    for atm in np.linspace(0.0, 0.40, 4001):
        if abs(atm - O2_THRESHOLD) < 1e-6:
            continue
        q = atmosphere_fixed.quantize_scalar(float(atm))
        for count in (1, 2, 3, 4):
            combo = (q,) * count
            assert cpp_float_o2_gate(combo) == py_int_o2_gate(combo), (
                f"homogeneous O2 gate disagreement at atm={atm:.6f} count={count}")


def test_o2_gate_heterogeneous_disagreements_are_exact_ties_only():
    """Heterogeneous neighbours near the threshold — the HARD case where the
    integer mean_round and the C++ float32 division CAN differ. Assert that every
    disagreement is an EXACT threshold tie (integer mean == quantize(threshold)),
    never a real divergence — i.e. the integer gate is the correct/deterministic
    one and S3b's C++ mean_round will bit-match it. EOS refactor P4: the sweep
    was rescaled to bracket the then-new ~0.12 threshold (was ~0.60).
    v2.4 re-pin (eos-p3fix-thermal-ceiling): o2_threshold moved 0.12 -> 0.01
    (the hot-zone-equilibrium rescale, config.toml [physics.fire]); the sweep
    rescales with it — same density, same 0.2x-to-2.2x threshold bracket,
    still containing the exact threshold value so ties occur."""
    vals = np.linspace(0.002, 0.022, 41)      # densely straddles the new threshold
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
# (1b) S3b HEADLINE — the Python ignition twin and the C++ fire P gate now BOTH
#      use the integer mean_round, so they agree EXACTLY (ZERO disagreements),
#      INCLUDING negative-atmosphere neighbour configs. S3a left a tie-only gap
#      (Python integer vs C++ FLOAT); S3b closes it to ZERO by making the C++ gate
#      integer too, AND fixes the Python twin's negative-branch divide
#      (trunc-toward-zero, NOT floor) so the two bit-match even on a transiently
#      negative neighbour sum (atmosphere CAN dip negative — wave forcing subtracts).
# ---------------------------------------------------------------------------
def test_py_twin_matches_cpp_integer_gate_exactly_incl_negative():
    """The S3b invariant the brief mandates: the Python twin's O2 mean and the C++
    fire P gate (both integer mean_round) agree on the IDENTICAL boolean across a
    sweep that includes NEGATIVE atmosphere — ZERO disagreements. This is the
    negative-branch fix: C++ `/` truncates toward zero, Python `//` floors, so a
    naive twin diverges on a negative neighbour sum; the fix makes both trunc-to-0."""
    # A COMPACT value set spanning negative through positive atmosphere, hitting the
    # sign boundary (where floor vs trunc diverge) and the 0.60 threshold. Kept small
    # (combinatorial: repeat up to 4) but chosen to land on odd sums over count so
    # the divide is NOT exact (the only regime where trunc != floor can bite), plus
    # raw odd/even count combinations. ~22 values -> 22^4 ~ 234k combos at most.
    raw_vals = [-0.30, -0.21, -0.13, -0.07, -0.03, -0.01, 0.0, 0.01, 0.03, 0.07,
                0.17, 0.31, 0.49, 0.55, 0.59, 0.60, 0.61, 0.63, 0.70, 0.83, 1.0]
    qs = [atmosphere_fixed.quantize_scalar(float(v)) for v in raw_vals]
    # Add a few RAW odd-count integers near 0 so a negative sum that is NOT a clean
    # multiple of count is exercised (the trunc-vs-floor divergence surface).
    qs += [-3, -1, 1, 3, -65535, 65535, -1, -2]

    # (a) the MEAN itself bit-matches (the reduction, before the threshold), so the
    # property holds independent of the specific threshold.
    n_mean_disagree = 0
    n_gate_disagree = 0
    n_neg_seen = 0
    n_neg_inexact = 0
    for count in (1, 2, 3, 4):
        for combo in itertools.product(qs, repeat=count):
            s = sum(int(q) for q in combo)
            if s < 0:
                n_neg_seen += 1
                if s % count != 0:
                    n_neg_inexact += 1   # the regime where trunc != floor
            if py_int_o2_mean(combo) != cpp_mean_round(s, count):
                n_mean_disagree += 1
            if py_int_o2_gate(combo) != cpp_int_o2_gate(combo):
                n_gate_disagree += 1
    assert n_mean_disagree == 0, (
        f"{n_mean_disagree} mean disagreements between the Python twin and the C++ "
        f"mean_round — the negative-branch trunc-toward-zero fix is incomplete")
    assert n_gate_disagree == 0, (
        f"{n_gate_disagree} O2-gate disagreements between the Python ignition twin "
        f"and the C++ fire P gate — must be ZERO after S3b (both integer mean_round)")
    # The sweep MUST actually exercise negative sums that are NOT exact multiples of
    # count (the regime where trunc-toward-zero and floor-toward-(-inf) DIVERGE — the
    # whole point of the fix). A vacuous sweep would silently pass.
    assert n_neg_seen > 0 and n_neg_inexact > 0, (
        f"the sweep did not exercise the negative inexact-divide regime "
        f"(neg_seen={n_neg_seen}, neg_inexact={n_neg_inexact}) — it is not testing "
        f"the trunc-vs-floor divergence the negative-branch fix closes")


# ---------------------------------------------------------------------------
# (2) Drive the PRODUCTION function (apply_temperature_ignition) and confirm its
#     ignite decision matches the C++ float O2 gate on a hot flammable tile.
# ---------------------------------------------------------------------------
class _GasTableStub:
    """Minimal stand-in for GameMap's real GasTable — apply_temperature_
    ignition only reads `.name_to_id["o2"]` (EOS refactor P4, design §6)."""

    def __init__(self):
        from simulation.gases import O2
        self.name_to_id = {"o2": O2}


class _Cross3x3:
    """A flammable wood centre on a 3x3 with an air ring; set the ring's REAL
    O2 (the O2 source, EOS refactor P4 — was `atmosphere`) and a hot centre,
    then run apply_temperature_ignition and see whether [1,1] ignited — the
    real production path."""

    def __init__(self, ring_atm):
        from simulation.gases import N_GASES, O2
        m = np.full((3, 3), MAT_AIR, dtype=np.int8)
        m[1, 1] = MAT_WOOD
        self.materials = _TBL
        self.material = m
        self.flammable = _TBL.flammable[m]
        self.solid = (_TBL.permeability[m] <= 0.0)
        self.is_vacuum = np.zeros((3, 3), dtype=bool)
        self.gases = _GasTableStub()
        # `ring_atm` names the test's SWEPT quantity (kept for call-site
        # readability / minimal diff) — it seeds the REAL O2 plane now, not
        # `atmosphere` (which apply_temperature_ignition no longer reads).
        o2_val = np.where(self.solid, 0,
                          atmosphere_fixed.quantize_scalar(float(ring_atm))).astype(np.int32)
        self.gas = np.zeros((N_GASES, 3, 3), dtype=np.int32)
        self.gas[O2] = o2_val
        self.temperature = np.full((3, 3), IGN_WOOD_Q16 * 2, dtype=np.int32)  # hot
        self.fire = np.zeros((3, 3), dtype=np.int32)


def test_production_ignition_matches_cpp_gate_off_tie():
    """Sweep the ring's REAL O2 AWAY from the exact tie (so the float and int
    gates agree by construction) and confirm the production ignition decision
    matches the C++ float O2 gate at each point. (The on-tie behaviour is the
    integer-canonical one tested above; here we exercise the real function.)"""
    # EOS refactor P4: O2_THRESHOLD now reads the REAL N_O2 scale (~0.12,
    # config.toml [physics.fire].o2_threshold) instead of the old
    # atmosphere/P scale (~0.60) — the sweep is rescaled to bracket it.
    for atm in np.linspace(0.0, 0.30, 301):
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
