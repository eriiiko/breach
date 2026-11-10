"""G1p-c (gameplay pack v1, #183) -- the ``[speeds]`` table and its doors.

``simulation/speeds.py``: ``speed_settings`` (the config door: six keys in
m/s and fractions, refused by name; Erik's rules on them) and ``speed_table``
(quantized ONCE per map with its tile size and ``[clock] ticks_per_second``).
Design: ``docs/bots_zombies_speeds_design_2026-11-10.md`` §1.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG, ticks_from_seconds  # noqa: E402
from simulation.speeds import (SPEED_KEYS, speed_settings,  # noqa: E402
                               speed_table)
from simulation.unit_fixed import FP_ONE  # noqa: E402

SHIPPED = {k: getattr(CFG.speeds, k) for k in SPEED_KEYS}


def _cfg(**over):
    vals = dict(SHIPPED)
    for k, v in over.items():
        if v is _DROP:
            vals.pop(k)
        else:
            vals[k] = v
    return SimpleNamespace(speeds=SimpleNamespace(**vals),
                           clock=SimpleNamespace(ticks_per_second=24))


_DROP = object()


def _level_tiles():
    """Every shipped level's tile size (``level.toml`` ``tile_size_m``)."""
    out = set()
    for p in sorted((ROOT / "levels").glob("*/level.toml")):
        with open(p, "rb") as f:
            t = tomllib.load(f)
        ts = t.get("tile_size_m", t.get("level", {}).get("tile_size_m"))
        if ts is not None:
            out.add(float(ts))
    return sorted(out)


def test_the_shipped_table_passes_the_door_and_quantizes_on_every_level():
    """Property: the shipped ``[speeds]`` passes the door and quantizes on
    every shipped level's tile size with the marine's full speed under one
    tile a tick (a step never skips a tile), and the realised m/s are the
    config's to within one Q16 count of the cost. Breaks if a retune or a
    finer level makes a unit cover a whole tile in one tick, or the
    quantization drifts from the m/s."""
    s = speed_settings(CFG)
    tiles = _level_tiles()
    assert tiles, "no level declares a tile size"
    for tile in tiles:
        t = speed_table(CFG, tile)
        assert t.full_cost_q > FP_ONE and t.walk_cost_q >= t.full_cost_q
        for cost, mps in ((t.walk_cost_q, s.marine_walk_mps),
                          (t.full_cost_q, s.marine_full_mps),
                          (t.zombie_sprint_cost_q, s.zombie_sprint_mps)):
            assert abs(t.realised_mps(cost) - mps) <= mps / cost


@pytest.mark.parametrize("key", SPEED_KEYS)
def test_the_door_refuses_a_missing_key_by_name(key):
    """Property: every key of ``[speeds]`` is required; a missing one is
    refused naming it. Breaks if the door defaults a speed silently."""
    with pytest.raises(ValueError, match=key):
        speed_settings(_cfg(**{key: _DROP}))


@pytest.mark.parametrize("key", SPEED_KEYS)
@pytest.mark.parametrize("bad", ["1.4", True, None, [1.4]])
def test_the_door_refuses_a_mistyped_key_by_name(key, bad):
    """Property: a non-number (a string, a bool, a list, nothing) is refused
    naming the key. Breaks if the door coerces it."""
    with pytest.raises(ValueError, match=key):
        speed_settings(_cfg(**{key: bad}))


def test_the_door_refuses_a_table_without_a_speeds_section():
    """Property: a config with no ``[speeds]`` is refused. Breaks if the
    door falls back to the old cadence dials."""
    with pytest.raises(ValueError, match=r"\[speeds\]"):
        speed_settings(SimpleNamespace(clock=SimpleNamespace(
            ticks_per_second=24)))


@pytest.mark.parametrize("frac", [0.69, 0.5, 0.91, 1.2])
def test_the_door_refuses_a_zombie_sprint_outside_70_to_90_percent(frac):
    """Property (Erik's rule, 2026-11-10): the zombie's sprint is within
    70 % .. 90 % of the marine's walk; outside it is refused naming
    ``zombie_sprint_mps``, inside (both edges included) accepted. Breaks if
    a retune moves the sprint out of the band unnoticed."""
    walk = SHIPPED["marine_walk_mps"]
    with pytest.raises(ValueError, match="zombie_sprint_mps.*70% .. 90%"):
        speed_settings(_cfg(zombie_sprint_mps=frac * walk))
    for ok in (0.70, 0.80, 0.90):
        speed_settings(_cfg(zombie_sprint_mps=ok * walk))


def test_the_door_refuses_a_zombie_sprint_not_above_the_backward_walk():
    """Property (Erik's rule): a sprinting zombie catches a marine walking
    backwards, so the sprint must be ABOVE ``backward_frac x walk``; at or
    below it the door refuses naming ``zombie_sprint_mps`` and the backward
    walk, even inside the 70-90 % band. Breaks if the check is dropped or
    compares against the forward walk."""
    walk = SHIPPED["marine_walk_mps"]
    cfg = _cfg(sideways_frac=0.9, backward_frac=0.85,
               zombie_sprint_mps=0.85 * walk)            # == the backward walk
    with pytest.raises(ValueError, match="zombie_sprint_mps.*backward"):
        speed_settings(cfg)
    speed_settings(_cfg(sideways_frac=0.9, backward_frac=0.85,
                        zombie_sprint_mps=0.86 * walk))


def test_the_door_refuses_what_would_make_the_law_non_monotone():
    """Property: the law is monotone only if the full speed is at least the
    walk and ``0 < backward_frac <= sideways_frac <= 1``; the door refuses
    each breach by name. Breaks if a retune can make more throttle slower or
    a sideways step slower than a backward one."""
    walk = SHIPPED["marine_walk_mps"]
    with pytest.raises(ValueError, match="marine_full_mps"):
        speed_settings(_cfg(marine_full_mps=0.9 * walk))
    with pytest.raises(ValueError, match="backward_frac"):
        speed_settings(_cfg(backward_frac=0.85, sideways_frac=0.8))
    with pytest.raises(ValueError, match="sideways_frac"):
        speed_settings(_cfg(sideways_frac=1.1))
    for key in SPEED_KEYS:
        with pytest.raises(ValueError, match=key):
            speed_settings(_cfg(**{key: 0.0}))


def test_a_map_on_which_a_step_would_skip_a_tile_is_refused():
    """Property: a unit never covers a whole tile or more in one tick: a
    tile so small that the full speed would is refused at the map door,
    naming the key. Breaks if the table quantizes a cost at or below one
    tick per tile."""
    t = SHIPPED["marine_full_mps"] / CFG.clock.ticks_per_second
    with pytest.raises(ValueError, match="marine_full_mps.*whole tile"):
        speed_table(CFG, t)                       # exactly one tile a tick
    speed_table(CFG, 1.01 * t)


def test_the_zombie_shamble_is_its_integer_walk_cadence_at_the_maps_tile():
    """Property: the zombie walks at ``zombie_shamble_mps`` through its own
    cadence law, an integer number of ticks per tile, quantized as the old
    seconds-per-tile dial was (``ticks_from_seconds(tile / mps, tps)``) on
    the map's tile size -- a finer map takes more ticks per tile. Breaks if
    the cadence ignores the tile size or the m/s row."""
    s = speed_settings(CFG)
    tps = CFG.clock.ticks_per_second
    for tile in (0.333, 0.5, 1.0):
        t = speed_table(CFG, tile)
        assert t.zombie_shamble_ticks == ticks_from_seconds(
            tile / s.zombie_shamble_mps, tps)
        # whole ticks: within half a tick of the exact cadence
        assert abs(t.zombie_shamble_ticks - tile * tps / s.zombie_shamble_mps) <= 0.5
    assert (speed_table(CFG, 0.333).zombie_shamble_ticks
            < speed_table(CFG, 1.0).zombie_shamble_ticks)


def _zombie_step_ticks(tile, convert):
    """Ticks to a zombie's first step on an open map of ``tile`` metres; the
    zombie is born one or (``convert``) a marine that turned."""
    from level_loader import LevelData
    from simulation.ai_zombie import (convert_marines_to_zombies,
                                      update_zombies_tick)
    from simulation.factions import ZOMBIE_TEAM
    from simulation.gamemap import GameMap
    from simulation.unit import Unit
    tm = np.ones((20, 30), dtype=np.int32)
    tm[1:19, 1:29] = 4
    g = GameMap(LevelData(name="z", version="1", path=Path("."), tilemap=tm,
                          tile_size_m=tile, diffuse_path=Path(".")))
    if convert:
        z = Unit("z", x=4.0, y=8.0, team=0)
        z.alive = False
        z.killed_by_zombie = True
        convert_marines_to_zombies([z])
    else:
        z = Unit("z", x=4.0, y=8.0, team=ZOMBIE_TEAM)
    prey = Unit("m", x=24.0, y=8.0, team=0)
    z.id, prey.id = 1, 2
    z.zombie_activated = True
    for t in range(1, 400):
        update_zombies_tick(g, [z, prey], t)
        if (z.x, z.y) != (4.0, 8.0):
            return t
    raise AssertionError("the zombie never stepped")


@pytest.mark.parametrize("tile", [0.333, 1.0])
def test_a_born_and_a_converted_zombie_shamble_at_the_tables_cadence(tile):
    """Property: every zombie -- spawned as one or a marine that turned --
    takes its first step after exactly the table's shamble cadence on its
    map. Breaks if a converted marine keeps a marine cadence, or the walk
    reads anything but the ``[speeds]`` row."""
    want = speed_table(CFG, tile).zombie_shamble_ticks
    assert _zombie_step_ticks(tile, convert=False) == want
    assert _zombie_step_ticks(tile, convert=True) == want
