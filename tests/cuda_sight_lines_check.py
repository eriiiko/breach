"""#195 1c gate (in the subprocess): sight for many tiles, the CUDA module's
answer.

``PhysicsEngine.sight_rays`` / ``sight_window`` (``cpp/src/sight_lines.cpp``)
is a host call in both modules -- the same translation unit, compiled into the
CUDA build (the horde field's shape). This script runs in a fresh interpreter
with the CUDA build first on ``sys.path`` and checks, on ship_demo, a generated
freighter and a screen-scattered box: the CUDA module's rays and windows (both
directions, sight and ``physical``) equal the pure-Python reference line for
line, and equal ``GameMap.has_los``. It prints a sha256 of every answer, which
the pytest wrapper compares with the CPU module's own (``digest``): the two
modules answer identically.

Prints ``SIGHT_LINES_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` (when this
# runs in the wrapper's own process the CPU build is the cached one).
import breach_physics as bp

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tests", ROOT / "src", ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from simulation import vision  # noqa: E402


def _worlds():
    import gen_ship as G
    import test_sight_lines as T
    plan = G.generate(1, "medium", recipe="freighter", profile="standard",
                      zombies=4)
    base = Path(tempfile.mkdtemp(prefix="sight_cuda_"))
    G.write(plan, base / plan.name)
    return [("ship_demo", level_loader.load("ship_demo", str(ROOT / "levels"))),
            ("freighter", level_loader.load(plan.name, levels_dir=str(base))),
            ("scatter", T._scatter_level())], T


def digest(check: bool = True) -> str:
    """sha256 over every answer of the routine on the three worlds; with
    ``check`` the answers are also held equal to the reference and ``has_los``
    (raising AssertionError). Uses whichever ``breach_physics`` is cached."""
    worlds, T = _worlds()
    h = hashlib.sha256()
    for name, level in worlds:
        sim = T._sim_of(level)
        g = sim.gmap
        pairs = T._rays(g, 2500, 21)
        for physical in (False, True):
            got = vision.lines_clear(g, pairs, physical)
            if check:
                solid, screen = vision._planes(g)
                ref = vision.sight_lines_reference(solid, screen, pairs,
                                                   physical)
                assert np.array_equal(got, ref), (name, physical, "ref")
                want = T._has_los(g, pairs, physical)
                assert np.array_equal(got.astype(bool), want), (name, physical)
            h.update(got.tobytes())
        hh, ww = g.solid.shape
        for (y, x, wy, wx, nh, nw) in [(hh // 2, ww // 2, hh // 2 - 12,
                                        ww // 2 - 12, 25, 25),
                                       (3, 4, -10, -9, 25, 25)]:
            for to_window in (False, True):
                for physical in (False, True):
                    got = vision.window_clear(sim, y, x, wy, wx, nh, nw,
                                              to_window, physical)
                    if check:
                        pr = vision._window_pairs(y, x, wy, wx, nh, nw,
                                                  to_window)
                        want = vision.rays_clear(sim, pr, physical).reshape(
                            nh, nw)
                        assert np.array_equal(got, want), (name, to_window)
                    h.update(got.tobytes())
    return h.hexdigest()


def main() -> int:
    print("MODULE:", bp.__file__)
    if "build_cuda" not in str(bp.__file__):
        print("SIGHT_LINES_RESULT: FAIL (not the CUDA module)")
        return 1
    try:
        d = digest()
    except AssertionError as exc:
        print("SIGHT_LINES_RESULT: FAIL", exc)
        return 1
    print("SIGHT_LINES_DIGEST:", d)
    print("SIGHT_LINES_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
