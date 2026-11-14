"""#140 -- THE footprint-samples builder and the wind-load read.

``movement.footprint_samples(gmap, fy, fx, fp, effort_scale_q16=)`` is the
ONE builder of the ``FootprintSamples`` every cadence caller hands
``movement.default_speed``; it carries the terrain mobility AND the walker's
already-reduced Q16 speed scale -- since G5p-b only the unit impulse law's
effort scale, which carries the wind's slowing too (the wind is a force on
that law; #140's ``wind_scale_q16`` member was deleted).
``GameMap.footprint_wind_load`` / ``footprint_wind`` are the pure field read
of what the air does to a body (docs/wind_on_units_design_2026-10-31.md §2,
§7 tests 6/6b; docs/unit_impulse_design_2026-11-10.md §4).

PROPERTIES PROTECTED
  1. Every ``default_speed`` call in ``src/`` takes samples from the builder
     (found by parsing ``src/``, never a pinned list). Breaks if a caller
     builds its own ``FootprintSamples`` and so escapes the walker's scale.
  2. The scale composes after the mobility cost as
     ``half_up(cost * ONE, scale)`` and ONE is exactly the old cost. Breaks
     if the compose changes or the unpushed path stops being bit-identical.
  4. Every cadence caller -- MOVE_DIR, the onephase cadence, the zombie
     walk -- takes ``half_up(calm * ONE, scale)`` from a walker's effort
     (the effort at the hold: half speed, the #140 s_min). Breaks if one of
     them builds samples without the walker's effort.
  3. The wind load is the Python-int formula even where an int32 square
     would wrap (50 m/s, twice ambient density), max over GAS tiles only,
     and ``footprint_wind`` hands back the wind of the tile the max was read
     on. Breaks if an int32 square returns, a solid tile is read, or the
     direction comes from another tile.
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
    no other module constructs one. Breaks if a caller escapes the
    walker's scale."""
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
    a step cheaper (monotone). (The member is the effort's since G5p-b.)"""
    mob = [MOBILITY_ONE] * 6 + [400] * 3
    for base in (1, 3, 8, 12):
        plain = default_speed(FootprintSamples(mobility=mob), base)
        one = default_speed(FootprintSamples(mobility=mob,
                                             effort_scale_q16=SCALE_ONE),
                            base)
        assert one == plain
        half = default_speed(FootprintSamples(
            mobility=mob, effort_scale_q16=SCALE_ONE // 2), base)
        assert half == half_up(plain * SCALE_ONE, SCALE_ONE // 2)
        prev = plain
        for scale in range(SCALE_ONE, SCALE_ONE // 4, -997):
            c = default_speed(FootprintSamples(mobility=mob,
                                               effort_scale_q16=scale), base)
            assert c >= prev
            prev = c


def test_the_builder_carries_the_footprint_mobility():
    """The builder's mobility is the GameMap's own footprint projection, and
    its scale is the effort scale it is handed (it reads no wind of its own
    since G5p-b: a gale on the planes leaves the samples unchanged)."""
    from level_loader import load as load_level
    from simulation.movement import footprint_samples
    g = GameMap(load_level("ship_demo", levels_dir=str(ROOT / "levels")))
    s = footprint_samples(g, 20, 37, 3, effort_scale_q16=40000)
    assert list(s.mobility) == g.footprint_mobility(20, 37, 3)
    assert s.effort_scale_q16 == 40000
    g.wind_x[20:23, 37:40] = 60 * FP_ONE
    assert footprint_samples(g, 20, 37, 3, effort_scale_q16=40000) == s


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
    # ... and the direction is that tile's own wind (G5p-b)
    assert GameMap.footprint_wind(g, 1, 1, 3) == (expect, vx, vy)


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
# Every cadence caller takes the walker's effort scale (design §7 test 6;
# G5p-b: the wind slows a walker through the effort -- the law's own read of
# what the wind does is tests/test_wave_push.py's)
# ---------------------------------------------------------------------------
def _room(h=24, w=24):
    """A real GameMap: an open hull box at ambient bulk N, still air."""
    from level_loader import LevelData
    from simulation.materials import MAT_AIR, MAT_HULL
    tm = np.full((h, w), MAT_AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = MAT_HULL
    lvl = LevelData(name="gale", version="3", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    return lvl


#: An effort at the hold: the walk at half speed (the #140 s_min band edge).
_AT_HOLD = FP_ONE
_HALF = SCALE_ONE // 2


def test_move_dir_walks_at_the_effort_slowed_cadence():
    """Property: a MOVE_DIR step by a walker whose effort is at its hold
    advances 1 / half_up(calm * ONE, ONE / 2) tiles; at effort 0 the calm
    cadence. Breaks if MOVE_DIR builds samples without the walker's effort.
    (G5p-a2: the step is that speed rounded half up ONCE onto the Q16
    grid, ``half_up(ONE^2, cost)`` counts -- critique B's M6.)"""
    bp = pytest.importorskip("breach_physics")
    from simulation import Simulation
    from simulation.unit import Unit
    sim = Simulation(_room(), seed=1, breach_physics=bp,
                     enable_recorder=False)
    u = Unit("w", x=6.0, y=10.0, team=0)
    sim.add_unit(u)
    # G5p-a1: a walker turns to its travel with inertia; this test is about
    # the cadence, so it already faces its walk (east): theta 0.
    from simulation.turning import place
    place(u, 0)
    md = SimpleNamespace(throttle_q=0, dx_q=FP_ONE, dy_q=0)  # the walk
    calm = sim.speed_table.walk_cost_q
    x0 = u.x
    sim._step_move_dir(u, md)
    assert u.x == x0 + half_up(FP_ONE * FP_ONE, calm) / FP_ONE
    u.effort = _AT_HOLD
    x0 = u.x
    sim._step_move_dir(u, md)
    slowed = half_up(calm * SCALE_ONE, _HALF)
    assert slowed > calm
    assert u.x == x0 + half_up(FP_ONE * FP_ONE, slowed) / FP_ONE


def test_the_onephase_cadence_takes_the_effort_scale():
    """Property: ``timeline.tile_cadence`` for a unit whose effort is at its
    hold is the terrain cost composed with the half scale, then the
    aim-relative fraction -- read when the plan is compiled. Breaks if the
    onephase cadence builds samples without the unit's effort."""
    from simulation.orders import ORDER_MOVE_ATTACK
    from simulation.timeline import base_ticks_per_tile, speed_pct, tile_cadence
    from simulation.unit import Unit
    g = GameMap(_room())
    u = Unit("o", x=6.0, y=10.0, team=0)
    base = base_ticks_per_tile(g)
    pct = speed_pct(ORDER_MOVE_ATTACK, False)

    def with_pct(terrain):
        return terrain if pct >= 1.0 else max(1, int(terrain / pct + 0.5))

    assert tile_cadence(g, u, 10, 7, ORDER_MOVE_ATTACK, False) == with_pct(base)
    u.effort = _AT_HOLD
    slowed = half_up(base * SCALE_ONE, _HALF)
    assert tile_cadence(g, u, 10, 7, ORDER_MOVE_ATTACK, False) == with_pct(slowed)


def _ticks_to_first_zombie_step(g, effort=0):
    from simulation.ai_zombie import update_zombies_tick
    from simulation.factions import ZOMBIE_TEAM
    from simulation.unit import Unit
    z = Unit("z", x=4.0, y=10.0, team=ZOMBIE_TEAM)
    prey = Unit("m", x=18.0, y=10.0, team=0)
    z.id, prey.id = 1, 2
    z.zombie_activated = True
    z.effort = effort
    start = (z.x, z.y)
    for t in range(1, 200):
        update_zombies_tick(g, [z, prey], t)
        if (z.x, z.y) != start:
            return t, z
    raise AssertionError("the zombie never stepped")


def test_the_zombie_walk_takes_the_effort_scale():
    """Property: a zombie whose effort is at its hold takes its first step
    after half_up(calm * ONE, ONE / 2) ticks, calm after its own cadence --
    zombies lean in the wind like marines. Breaks if the zombie walk builds
    samples without the zombie's effort."""
    from config import CFG
    from simulation.speeds import speed_table
    g = GameMap(_room())
    # G1p-c: the zombie's cadence is the [speeds] shamble at this map's tile
    shamble = speed_table(CFG, g.tile_size_m).zombie_shamble_ticks
    t_calm, z = _ticks_to_first_zombie_step(g)
    assert t_calm == shamble
    t_slow, z = _ticks_to_first_zombie_step(g, effort=_AT_HOLD)
    assert t_slow == half_up(shamble * SCALE_ONE, _HALF)
    assert t_slow > t_calm
