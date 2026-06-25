"""Patch 0 — proves the field-level A/B determinism harness works AND is strictly
sharper than the legacy grid-mean check (which is blind to per-cell desync).

This is the gate the PhysicsEngine unification (Patch 1) leans on: it must (a) hold
the current code to per-cell bit-identity on a re-run, and (b) actually CATCH a
per-cell, mean-preserving perturbation — the exact failure mode the old
mean-signature determinism test (test_simulation.py) cannot see.

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_field_ab_harness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tests", ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import copy

import numpy as np

from field_ab_harness import (
    capture_trajectory, diff_trajectories, assert_trajectories_match,
    default_scenario_sim,
)

N = 30


def test_scenario_is_nontrivial():
    """The A/B scenario must actually exercise the solvers, or the gate is hollow."""
    traj = capture_trajectory(default_scenario_sim, N)
    first, last = traj[0], traj[-1]
    for k in ("atmosphere", "gas", "wave_p", "heat", "temperature", "water_depth"):
        assert k in last, f"field {k} not captured"
    # smoke moved/persisted, the wave fired, and SOMETHING evolved across the run.
    assert np.any(last["gas"] != 0.0), "smoke vanished entirely (no transport exercised)"
    assert any(np.any(traj[t]["wave_p"] != 0.0) for t in range(min(5, N))), \
        "wave never fired (wave solver not exercised)"
    assert not np.array_equal(first["atmosphere"], last["atmosphere"]), \
        "atmosphere never changed (physics not exercised)"


def test_field_trajectory_deterministic():
    """Same scenario + seed -> bit-identical field trajectory, per cell, every tick."""
    a = capture_trajectory(default_scenario_sim, N)
    b = capture_trajectory(default_scenario_sim, N)
    assert_trajectories_match(a, b, tol=0.0)


def test_harness_catches_mean_preserving_perturbation():
    """The whole point: a per-cell, MEAN-PRESERVING perturbation that the legacy
    grid-mean signature is blind to, this harness flags."""
    a = capture_trajectory(default_scenario_sim, N)
    b = copy.deepcopy(a)
    snap = b[N // 2]["atmosphere"]
    # S2c: atmosphere is int32 Q16.16 — perturb by integer COUNTS (a float 1e-3
    # would truncate to 0 in the int field). +1/-1 count is mean-preserving.
    snap[3, 3] += 1      # +d here ...
    snap[4, 4] -= 1      # ... -d there -> whole-grid mean is unchanged
    # the legacy mean signature CANNOT see this:
    assert abs(float(b[N // 2]["atmosphere"].mean())
               - float(a[N // 2]["atmosphere"].mean())) < 1e-12, \
        "perturbation was not mean-preserving — test is not making its point"
    # the per-cell harness DOES:
    diffs = diff_trajectories(a, b, tol=0.0)
    assert diffs, "harness missed a per-cell mean-preserving perturbation (it is blind!)"


def test_tolerance_mode():
    """tol>0 accepts sub-tolerance noise and rejects above-tolerance — the fallback
    (B) path if the /fp:precise 0-ULP gate is ever relaxed to a stated tolerance."""
    a = capture_trajectory(default_scenario_sim, N)
    b = copy.deepcopy(a)
    # S2c: atmosphere is int32 — use a `ripple` cell (still float) to exercise the
    # tolerance path on a sub-tolerance float delta (atmosphere counts are integer,
    # so a 1e-6 delta there is not representable).
    b[1]["ripple"][5, 5] += 1e-6
    assert not diff_trajectories(a, b, tol=1e-5), "tol=1e-5 should accept a 1e-6 delta"
    assert diff_trajectories(a, b, tol=1e-9), "tol=1e-9 should reject a 1e-6 delta"


if __name__ == "__main__":
    test_scenario_is_nontrivial()
    test_field_trajectory_deterministic()
    test_harness_catches_mean_preserving_perturbation()
    test_tolerance_mode()
    print("OK: field-level A/B harness — deterministic, per-cell-sharp, tolerance-aware")
