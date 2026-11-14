"""Per-channel material extinction on the LIGHT SWEEP -- each channel reads its
own coefficient, and a partial column dims (never lights) what lies behind it.

Rewritten at ray-engine-v2 P6c (design v3 §11.2 row "test_rgb_light_atten.py:
rewrite as gate-3 scenes on the light extinction planes (opaque blocks, glass
~90 %, asymmetric tint); 'aggregate termination' dies (no rays)"). Until P6c
these were scenes of one pencil ray of the old C++ render march:

  * an opaque tile blocks everything beyond it -- held on the sweep by
    tests/test_radiation_sweep_light.py::test_an_opaque_wall_leaves_the_far_side_exactly_dark
    (P6a);
  * glass dims and an asymmetric triple tints -- HERE, as properties of the
    light extinction planes. (The march's "exactly 90 % behind one glass tile"
    was a property of one straight ray; the sweep's STEP transport lets a
    stream travel along a column, so a full-height glass column passes AT MOST
    90 % -- the bound is the property, and the tint is exact channel
    independence);
  * "aggregate termination" (a ray kept alive by its surviving channels) has no
    meaning for a sweep, which has no rays -- gone.

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_rgb_light_atten.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import ONE, R, cpp_sweep, live_table  # noqa: E402

Q = R.quant
N = 11
XW = 5          # the partial column
GLASS = (Q(0.1),) * 3              # the shipped glass light row
DARK = (Q(0.9),) * 3
TINT = (Q(0.9), Q(0.9), Q(0.1))    # kills red and green, passes blue


def _light(column):
    """light_q (3, h, w) of an N x N transparent room lit by a hot opaque
    column at x = 0, with a full-height column of per-channel extinction
    `column` (a = d, Q16) at x = XW (None = clear air)."""
    a = [[0] * N for _ in range(N)]
    d = [[0] * N for _ in range(N)]
    T = [[0] * N for _ in range(N)]
    his = [[3] * N for _ in range(N)]
    ts = [[0] * N for _ in range(N)]
    la = [[[0] * N for _ in range(N)] for _ in range(3)]
    ld = [[[0] * N for _ in range(N)] for _ in range(3)]
    for y in range(N):
        T[y][0] = 3000 << 16
        ts[y][0] = 1
        for c in range(3):
            la[c][y][0] = ld[c][y][0] = ONE
            if column is not None:
                la[c][y][XW] = ld[c][y][XW] = column[c]
    got = cpp_sweep(a, d, 0, T, his, ts, transport="shear", n_ord=16,
                    table=live_table(), light=dict(la=la, ld=ld))
    return np.asarray(got[6]["light_q"])


def test_each_channel_reads_only_its_own_extinction():
    """PROPERTY (design §5 "per-material light_atten RGB", on the sweep): the
    light channels are independent -- behind a (0.9, 0.9, 0.1) column the RED
    and GREEN planes are EXACTLY what a (0.9, 0.9, 0.9) column gives and the
    BLUE plane EXACTLY what a (0.1, 0.1, 0.1) glass column gives, everywhere;
    so the tinted column passes blue and kills red and green.

    BREAKS IF: a channel reads another channel's coefficient (or one scalar for
    all three), or the channels' streams are coupled anywhere in the traversal.
    """
    tint, dark, glass = _light(TINT), _light(DARK), _light(GLASS)
    assert np.array_equal(tint[0], dark[0]) and np.array_equal(tint[1], dark[1])
    assert np.array_equal(tint[2], glass[2])
    behind = (slice(2, N - 2), slice(XW + 1, None))
    assert np.all(tint[2][behind] > 5 * tint[0][behind]), "the tint did not pass blue"


def test_a_denser_column_passes_less_and_never_more_than_one_crossing():
    """PROPERTY (the absorb (stream·d) >> 16, on a light plane): every cell
    behind a full-height column receives, per channel, at most (ONE - d)/ONE of
    what it receives with the column clear (every path to it crosses the column
    at least once; the step transport's paths along the column only take more),
    and strictly less through the denser column; the column never lights the
    cells in front of it (no reflection, and at ambient it emits nothing).

    BREAKS IF: the absorb takes less than (stream·d) >> 16 of a crossing
    stream, the column emits or reflects at ambient, or a denser coefficient
    passes more light.
    """
    clear, glass, dark = _light(None), _light(GLASS), _light(DARK)
    behind = (slice(None), slice(2, N - 2), slice(XW + 1, None))
    assert np.all(clear[behind] > 0), "the scene must light the cells behind the column"
    for lit, col in ((glass, GLASS), (dark, DARK)):
        ratio = lit[behind] / clear[behind]
        assert np.all(ratio <= (ONE - col[0]) / ONE + 1e-4), (ratio.max(), col)
        assert np.all(ratio > 0)
        assert np.array_equal(lit[:, :, 1:XW], clear[:, :, 1:XW]), "the column lit its front"
    assert np.all(dark[behind] < glass[behind])
