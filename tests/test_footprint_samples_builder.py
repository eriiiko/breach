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
  4. Every cadence caller -- MOVE_DIR, the onephase cadence, the zombie
     walk -- takes ``half_up(calm * ONE, s_min_q16)`` in an s_min wind.
     Breaks if one of them builds samples without the wind.
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
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
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


# ---------------------------------------------------------------------------
# Every cadence caller takes the wind's scale (design §7 test 6)
# ---------------------------------------------------------------------------
def _gale_room(h=24, w=24, gale=True):
    """A real GameMap: an open hull box at ambient bulk N, its wind planes
    set (a fixture write; nothing is stepped) to a gale far above q_fall, so
    the builder's scale is exactly s_min everywhere."""
    from level_loader import LevelData
    from simulation.materials import MAT_AIR, MAT_HULL
    tm = np.full((h, w), MAT_AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = MAT_HULL
    lvl = LevelData(name="gale", version="3", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    return lvl


def _blow(g, on=True):
    open_ = ~g.solid
    g.wind_x[open_] = 40 * FP_ONE if on else 0
    g.wind_y[open_] = 0


def _s_min_q16(g):
    from simulation.exchange import gmap_wind_settings
    return gmap_wind_settings(g).s_min_q16


def test_move_dir_walks_at_the_slowed_cadence_in_a_gale():
    """Property: a MOVE_DIR step in an s_min wind advances
    1 / half_up(calm * ONE, s_min_q16) tiles, against the field at
    execution. Breaks if MOVE_DIR builds samples without the wind."""
    bp = pytest.importorskip("breach_physics")
    from config import CFG
    from simulation import Simulation
    from simulation.orders import ORDER_MOVE_ATTACK
    from simulation.unit import Unit
    sim = Simulation(_gale_room(), seed=1, breach_physics=bp,
                     enable_recorder=False)
    u = Unit("w", x=6.0, y=10.0, team=0)
    sim.add_unit(u)
    md = SimpleNamespace(speed_mode=ORDER_MOVE_ATTACK, dx_q=FP_ONE, dy_q=0)
    calm = int(CFG.movement.marine_attack_ticks_per_tile)
    _blow(sim.gmap, on=False)
    x0 = u.x
    sim._step_move_dir(u, md)
    assert u.x == x0 + 1.0 / calm
    _blow(sim.gmap, on=True)
    x0 = u.x
    sim._step_move_dir(u, md)
    slowed = half_up(calm * SCALE_ONE, _s_min_q16(sim.gmap))
    assert slowed > calm
    assert u.x == x0 + 1.0 / slowed


def test_the_onephase_cadence_takes_the_slowed_scale():
    """Property: ``timeline.tile_cadence`` over a footprint in an s_min wind
    is the terrain cost composed with the scale, then the aim-relative
    fraction -- against the field when the plan is compiled. Breaks if the
    onephase cadence builds samples without the wind."""
    from simulation.orders import ORDER_MOVE_ATTACK
    from simulation.timeline import base_ticks_per_tile, speed_pct, tile_cadence
    from simulation.unit import Unit
    g = GameMap(_gale_room())
    u = Unit("o", x=6.0, y=10.0, team=0)
    base = base_ticks_per_tile(g)
    pct = speed_pct(ORDER_MOVE_ATTACK, False)

    def with_pct(terrain):
        return terrain if pct >= 1.0 else max(1, int(terrain / pct + 0.5))

    _blow(g, on=False)
    assert tile_cadence(g, u, 10, 7, ORDER_MOVE_ATTACK, False) == with_pct(base)
    _blow(g, on=True)
    slowed = half_up(base * SCALE_ONE, _s_min_q16(g))
    assert tile_cadence(g, u, 10, 7, ORDER_MOVE_ATTACK, False) == with_pct(slowed)


def _ticks_to_first_zombie_step(g):
    from simulation.ai_zombie import update_zombies_tick
    from simulation.factions import ZOMBIE_TEAM
    from simulation.unit import Unit
    z = Unit("z", x=4.0, y=10.0, team=ZOMBIE_TEAM)
    prey = Unit("m", x=18.0, y=10.0, team=0)
    z.id, prey.id = 1, 2
    z.zombie_activated = True
    start = (z.x, z.y)
    for t in range(1, 200):
        update_zombies_tick(g, [z, prey], t)
        if (z.x, z.y) != start:
            return t, z
    raise AssertionError("the zombie never stepped")


def test_the_zombie_walk_takes_the_slowed_scale():
    """Property: a zombie in an s_min wind takes its first step after
    half_up(calm * ONE, s_min_q16) ticks, calm after its own cadence --
    zombies lean in the wind like marines. Breaks if the zombie walk builds
    samples without the wind."""
    g = GameMap(_gale_room())
    _blow(g, on=False)
    t_calm, z = _ticks_to_first_zombie_step(g)
    assert t_calm == z.speed_ticks_per_tile
    _blow(g, on=True)
    t_gale, z = _ticks_to_first_zombie_step(g)
    assert t_gale == half_up(z.speed_ticks_per_tile * SCALE_ONE, _s_min_q16(g))
    assert t_gale > t_calm
