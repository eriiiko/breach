"""#195 1g (b) -- ``GameMap.wind_load_plane``: the per-tile term of the one
wind read (``footprint_wind``), over the whole map, for the observation's
wind planes (obs spec 1.7, 3.2).

Property: on a random room (random bulk N, random winds, scattered solid
tiles) the plane's footprint max equals ``footprint_wind_load`` on random
footprints, its argmax tile carries ``footprint_wind``'s direction, and a
solid tile reads 0. Breaks if the plane re-derives the law (the shift, the
bulk sum, the solid exclusion) differently from the footprint read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation.gases import INERT_N2, O2  # noqa: E402
from test_wave_push import _room_map  # noqa: E402


def _random_air(g, rng):
    """Random bulk N (0.2 .. 3 atmospheres) and winds (up to ~30 m/s) on
    every open tile; a few open tiles made solid."""
    open_ = ~g.solid
    n = rng.integers(13_000, 200_000, size=g.solid.shape)
    g.gas[O2][open_] = (n // 5)[open_]
    g.gas[INERT_N2][open_] = (n - n // 5)[open_]
    g.wind_x[open_] = rng.integers(-30 << 16, 30 << 16, size=g.solid.shape)[open_]
    g.wind_y[open_] = rng.integers(-30 << 16, 30 << 16, size=g.solid.shape)[open_]
    ys, xs = np.nonzero(open_)
    pick = rng.choice(len(ys), size=max(1, len(ys) // 10), replace=False)
    g.solid[ys[pick], xs[pick]] = True


def test_the_planes_footprint_max_is_the_footprint_read():
    """Property: over 300 random footprints (sizes 1..4, partly off the map)
    on 5 random rooms, ``max(W over the footprint)`` equals
    ``footprint_wind_load`` exactly and the first row-major tile carrying
    that max has ``footprint_wind``'s ``(vx, vy)``; a solid tile's ``W`` is
    0. Breaks if the plane's law drifts from the footprint read."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(5):
        g = _room_map(20, 24)
        _random_air(g, rng)
        w, vx, vy = g.wind_load_plane()
        assert w.shape == g.solid.shape and w.dtype == np.float64
        assert not w[g.solid].any()
        assert vx.dtype == np.float64 and vy.dtype == np.float64
        h, wd = g.solid.shape
        for _ in range(60):
            fp = int(rng.integers(1, 5))
            fy = int(rng.integers(-2, h))
            fx = int(rng.integers(-2, wd))
            want_w, want_vx, want_vy = g.footprint_wind(fy, fx, fp)
            y0, x0 = max(0, fy), max(0, fx)
            y1, x1 = min(h, fy + fp), min(wd, fx + fp)
            if y1 <= y0 or x1 <= x0:
                assert want_w == 0
                continue
            block = w[y0:y1, x0:x1]
            got = float(block.max())
            assert got == float(want_w), (fy, fx, fp)
            if want_w > 0:
                i = int(np.argmax(block.reshape(-1)))     # first in row-major
                r, c = divmod(i, x1 - x0)
                assert (vx[y0 + r, x0 + c], vy[y0 + r, x0 + c]) == \
                    (float(want_vx), float(want_vy))
                checked += 1
    assert checked > 100
