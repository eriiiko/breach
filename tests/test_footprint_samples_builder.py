"""#140 -- THE footprint-samples builder and the wind-load read.

``movement.footprint_samples(gmap, fy, fx, fp)`` is the ONE builder of the
``FootprintSamples`` every cadence caller hands ``movement.default_speed``;
it carries the terrain mobility AND the wind's already-reduced Q16 cadence
scale. ``GameMap.footprint_wind_load`` is the pure field read the wind law
thresholds (docs/wind_on_units_design_2026-10-31.md §2, §2.2, §7 tests 6/6b).

PROPERTIES PROTECTED
  1. Every ``default_speed`` call in ``src/`` takes samples from the builder
     (found by parsing ``src/``, never a pinned list). Breaks if a caller
     builds its own ``FootprintSamples`` and so escapes the wind scale.
  2. The scale composes after the mobility cost as
     ``half_up(cost * ONE, scale)`` and ONE is exactly the old cost. Breaks
     if the compose changes or the calm path stops being bit-identical.
  3. The wind load is the Python-int formula even where an int32 square
     would wrap (50 m/s, twice ambient density), max over GAS tiles only.
     Breaks if an int32 square returns or a solid tile is read.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation.atmosphere_fixed import FP_ONE  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gas_fixed import BULK_ONE  # noqa: E402
from simulation.gases import INERT_N2, N_GASES, O2  # noqa: E402
from simulation.movement import (  # noqa: E402
    MOBILITY_ONE, SCALE_ONE, FootprintSamples, default_speed, half_up,
)

SRC = ROOT / "src"


def _calls_named(node, name):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if (isinstance(f, ast.Name) and f.id == name) or (
                    isinstance(f, ast.Attribute) and f.attr == name):
                yield sub


def test_every_src_default_speed_caller_takes_samples_from_the_builder():
    """Property 1: parse every module under src/ (the defining module
    excepted); each function that calls ``default_speed`` also calls
    ``footprint_samples`` and constructs no ``FootprintSamples`` itself, and
    no other module constructs one. Breaks if a caller escapes the wind."""
    callers = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "simulation/movement.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not list(_calls_named(tree, "FootprintSamples")), (
            f"{rel} builds its own FootprintSamples -- use "
            f"movement.footprint_samples")
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # the innermost function owning each call
            own = [c for c in _calls_named(fn, "default_speed")
                   if not any(isinstance(inner, (ast.FunctionDef,
                                                 ast.AsyncFunctionDef))
                              and inner is not fn
                              and c in list(ast.walk(inner))
                              for inner in ast.walk(fn))]
            if own:
                callers.append(f"{rel}::{fn.name}")
                assert list(_calls_named(fn, "footprint_samples")), (
                    f"{rel}::{fn.name} calls default_speed without "
                    f"movement.footprint_samples")
    # Non-vacuous: the scan found the cadence callers it exists to guard.
    assert callers, "no default_speed caller found under src/ -- scan broken"


def test_the_scale_composes_after_mobility_and_one_is_exact():
    """Property 2: SCALE_ONE returns the mobility cost bit for bit; a half
    scale is the half-up compose of that cost; a smaller scale never makes
    a step cheaper (monotone)."""
    mob = [MOBILITY_ONE] * 6 + [400] * 3
    for base in (1, 3, 8, 12):
        plain = default_speed(FootprintSamples(mobility=mob), base)
        one = default_speed(FootprintSamples(mobility=mob,
                                             wind_scale_q16=SCALE_ONE), base)
        assert one == plain
        half = default_speed(FootprintSamples(mobility=mob,
                                              wind_scale_q16=SCALE_ONE // 2),
                             base)
        assert half == half_up(plain * SCALE_ONE, SCALE_ONE // 2)
        prev = plain
        for scale in range(SCALE_ONE, SCALE_ONE // 4, -997):
            c = default_speed(FootprintSamples(mobility=mob,
                                               wind_scale_q16=scale), base)
            assert c >= prev
            prev = c


def test_the_builder_carries_the_footprint_mobility():
    """The builder's mobility is the GameMap's own footprint projection."""
    from level_loader import load as load_level
    from simulation.movement import footprint_samples
    g = GameMap(load_level("ship_demo", levels_dir=str(ROOT / "levels")))
    s = footprint_samples(g, 20, 37, 3)
    assert list(s.mobility) == g.footprint_mobility(20, 37, 3)
    assert 0 < s.wind_scale_q16 <= SCALE_ONE


def _stub(h=6, w=6):
    return SimpleNamespace(
        _h=h, _w=w,
        solid=np.zeros((h, w), dtype=bool),
        wind_x=np.zeros((h, w), dtype=np.int32),
        wind_y=np.zeros((h, w), dtype=np.int32),
        gas=np.zeros((N_GASES, h, w), dtype=np.int32))


def test_wind_load_is_the_python_int_formula_past_int32():
    """Property 3: 50 m/s at twice ambient bulk N -- the int32 square of the
    raw wind (3,276,800^2) wraps; the read must equal the unbounded int."""
    g = _stub()
    vx, vy = 50 * FP_ONE, -30 * FP_ONE
    g.wind_x[2, 2], g.wind_y[2, 2] = vx, vy
    g.gas[O2][2, 2] = 2 * BULK_ONE // 5
    g.gas[INERT_N2][2, 2] = 2 * BULK_ONE - 2 * BULK_ONE // 5
    n = int(g.gas[O2][2, 2]) + int(g.gas[INERT_N2][2, 2])
    expect = n * ((vx * vx + vy * vy) >> 16)
    assert expect > 2 ** 31
    assert GameMap.footprint_wind_load(g, 1, 1, 3) == expect
    # max over the footprint: a calmer tile beside it does not lower it
    g.wind_x[1, 1] = FP_ONE
    g.gas[O2][1, 1] = BULK_ONE
    assert GameMap.footprint_wind_load(g, 1, 1, 3) == expect


def test_wind_load_skips_solid_tiles_and_reads_zero_off_grid():
    """A solid footprint tile is not read even when its planes hold a wind;
    a footprint wholly off the grid reads 0."""
    g = _stub()
    g.wind_x[2, 2] = 40 * FP_ONE
    g.gas[O2][2, 2] = BULK_ONE
    g.solid[2, 2] = True
    assert GameMap.footprint_wind_load(g, 1, 1, 3) == 0
    g.solid[2, 2] = False
    assert GameMap.footprint_wind_load(g, 1, 1, 3) > 0
    assert GameMap.footprint_wind_load(g, 10, 10, 3) == 0
