"""The larva law's sensing, statistically (arc #63 P3, docs/swarm_P3_impl.md
§11 T-P3-15..17; R8: fixed seeds, paired controls). The CPU twin through
``bp.swarm_species_step``; synthetic fields; explicit table values.

The pairing: two runs share every placement, heading, uid, seed and tick
and differ ONLY in the dials under test, so a larva's paired displacement
difference isolates the dials' effect (the wall drift of a random walk
started near a wall cancels in the pair).

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_swarm_larva_sensing.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _swarm_testkit as tk  # noqa: E402
from simulation import swarm  # noqa: E402

ONE = tk.FP_ONE
PI = tk.PI_Q16
N = 64
TICKS = 2400
P_BASE = 600                       # per tick: ~0.9 % (a run of ~110 ticks)
LAW = dict(turn_bias_q=int(0.8 * ONE), p_base_q=P_BASE, p_alarm_q=10 * P_BASE)
ABLATED = dict(turn_bias_q=ONE // 2, p_base_q=P_BASE, p_alarm_q=P_BASE)


def _gradient_field():
    # T(tx) = T_prefer + g * (tx - 32), inside (T_coma_exit, T_lethal_hot).
    g = ONE // 4
    tx = np.arange(N)
    row = (tk.DEFAULT_LARVA["T_prefer_q"] + g * (tx - 32)).astype(np.int32)
    return np.broadcast_to(row, (N, N)).copy()


def _border():
    solid = np.zeros((N, N), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    return solid


def _placements(rng, groups):
    out = []
    for (x0, x1, count) in groups:
        for _ in range(count):
            out.append((int(rng.integers(x0 << 16, x1 << 16)),
                        int(rng.integers(2 << 16, (N - 2) << 16)),
                        int(rng.integers(-PI + 1, PI + 1))))
    return out


def _run(T, placements, dials, ticks=TICKS):
    wd = tk.larva_world(N, N, len(placements), 1, T=T, solid=_border(), seed=(424242, 17),
                        **dials)
    for s, (x, y, h) in enumerate(placements):
        tk.place(wd, 0, s, x, y, h, uid=100 + s)
    x0 = wd.pos_x[0].astype(np.int64).copy()
    y0 = wd.pos_y[0].astype(np.int64).copy()
    for t in range(ticks):
        wd.env[:, swarm.ENV_TICK] = t
        tk.step_cpu(wd)
    assert (wd.valid[0] == 1).all(), "no larva may die or freeze in a sensing scene"
    return ((wd.pos_x[0].astype(np.int64) - x0) / ONE,
            (wd.pos_y[0].astype(np.int64) - y0) / ONE)


def _mean_se(d):
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


@pytest.fixture(scope="module")
def gradient_runs():
    rng = np.random.default_rng(15)
    pl = _placements(rng, ((4, 20, 400), (44, 60, 400)))
    T = _gradient_field()
    return {
        "law": _run(T, pl, LAW),
        "ablated": _run(T, pl, ABLATED),
        "ablated_flat": _run(np.full((N, N), tk.DEFAULT_LARVA["T_prefer_q"], np.int32), pl,
                             ABLATED),
    }


def test_larvae_climb_toward_T_prefer_from_both_sides(gradient_runs):
    """T-P3-15: on a linear gradient through T_prefer, the law (turn_bias
    0.8, alarm 10x base) moves the cold-side group up-gradient and the
    hot-side group down-gradient relative to the ablated law (bias 0.5, alarm
    = base) with identical placements, headings, uids, seeds and ticks: the
    left group's mean paired dx > 4 SE, the right group's < -4 SE.

    Must break if the alarm or bias sign flips, the comfort is T instead of
    |T - T_prefer| (up-gradient only: the right group's difference turns
    positive), or the sweep sides are swapped."""
    dx_a = gradient_runs["law"][0]
    dx_b = gradient_runs["ablated"][0]
    d = dx_a - dx_b
    left, right = d[:400], d[400:]
    ml, sl = _mean_se(left)
    mr, sr = _mean_se(right)
    assert ml > 4 * sl, f"cold side: paired mean dx {ml:.2f} tiles, SE {sl:.2f}"
    assert mr < -4 * sr, f"hot side: paired mean dx {mr:.2f} tiles, SE {sr:.2f}"


def test_the_drift_is_the_two_dials(gradient_runs):
    """T-P3-17: the ABLATED law on the gradient vs the ablated law on a flat
    field (paired): per group, the mean paired dx difference is within 4 SE
    of 0 -- with bias 0.5 and alarm = base the gradient has no lever left.

    Must break if the gradient reaches the motion through anything other
    than turn_bias / tumble_rate_alarm."""
    d = gradient_runs["ablated"][0] - gradient_runs["ablated_flat"][0]
    for grp in (d[:400], d[400:]):
        m, se = _mean_se(grp)
        assert abs(m) <= 4 * se, f"ablated gradient vs flat: {m:.2f} tiles, SE {se:.2f}"


def test_a_flat_field_is_unbiased():
    """T-P3-16: on a flat field, 800 larvae placed symmetrically (uniform over
    [2, 62)^2) have mean dx and mean dy within 4 SE of 0 after 2400 ticks;
    and as K x J independent single-tick trials (larvae >= 2 tiles from any
    wall, arrays restored per trial) the tumble frequency is p_base's
    (binomial 4 sigma).

    Must break on a floor-biased displacement (D12: mul_q16's floor instead
    of the sign-symmetric round), an alarm raised by geometry, or a
    directional draw bias."""
    rng = np.random.default_rng(16)
    pl = [(int(rng.integers(2 << 16, (N - 2) << 16)), int(rng.integers(2 << 16, (N - 2) << 16)),
           int(rng.integers(-PI + 1, PI + 1))) for _ in range(800)]
    flat = np.full((N, N), tk.DEFAULT_LARVA["T_prefer_q"], np.int32)
    dx, dy = _run(flat, pl, LAW)
    for d in (dx, dy):
        m, se = _mean_se(d)
        assert abs(m) <= 4 * se, f"flat field drift {m:.3f} tiles, SE {se:.3f}"

    K, J = 2000, 40
    wd = tk.larva_world(N, N, K, 1, T=flat, solid=_border(), seed=(5, 6), **LAW)
    for s in range(K):
        tk.place(wd, 0, s, int(rng.integers(4 << 16, (N - 4) << 16)),
                 int(rng.integers(4 << 16, (N - 4) << 16)), int(rng.integers(-PI + 1, PI + 1)))
    base = wd.copy()
    tumbles = 0
    for j in range(J):
        w = base.copy()
        w.env[:, swarm.ENV_TICK] = j
        tk.step_cpu(w)
        tumbles += int((w.heading[0] != base.heading[0]).sum())
    p = P_BASE / ONE
    sd = np.sqrt(p * (1 - p) / (K * J))
    assert abs(tumbles / (K * J) - p) <= 4 * sd
