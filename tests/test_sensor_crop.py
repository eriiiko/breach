"""#29 P1a -- the observation crop, ``EntityFieldAccessor.crop``
(``src/simulation/sensor_accessor.py``, residency design §4.5).

Properties: every in-grid cell of a unit's window is the field at that tile
dequantized through the field's own Q16 boundary module (no raw Q16 int, no
offset or transposition); cells outside the grid and rows without a unit
are zero; the occupancy planes are relative to the observing unit's team;
``light`` and unknown planes are refused by name; reading a crop writes no
sim state.

Levels are written through ``level_lib``; the match spawns through
``Simulation.spawn_match``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay  # noqa: E402
from simulation import (atmosphere_fixed, fire_fixed, gas_fixed,  # noqa: E402
                        water_fixed)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gases import O2, SMOKE  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.sensor_accessor import (EntityFieldAccessor,  # noqa: E402
                                        crop_channel_names)

IDLE = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="idle")
ROOMS = [
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4),
]
SPEC = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=2, time_limit_s=5.0)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("crop")
    lvl = tmp / "cropbox"
    lvl.mkdir()
    level_lib.write_level_header(lvl, name="cropbox", tile_size_m=1.0)
    tm = np.full((24, 40), MAT_HULL, dtype=np.int32)
    tm[1:23, 1:39] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (40, 24)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS)
    level_lib.write_match(lvl, SPEC)
    s = MatchSession(level_loader.load("cropbox", str(tmp)),
                     breach_physics=bp)
    for _ in range(3):
        s.tick()
    return s


def _planes(g):
    """The reference: each plane through its own boundary module."""
    return {
        "pressure": atmosphere_fixed.dequantize(g.atmosphere),
        "smoke": gas_fixed.dequantize_trace(g.gas[SMOKE]),
        "water_depth": water_fixed.dequantize(g.water_depth),
        "temperature": gas_fixed.dequantize_q16(g.temperature),
        "fire": fire_fixed.dequantize(g.fire),
        "o2": gas_fixed.dequantize_bulk(g.gas[O2]),
        "solid": g.solid.astype(np.float64),
        "material": g.material.astype(np.float64),
    }


class _Stub:
    """A unit at a given top-left tile (3x3 footprint)."""

    def __init__(self, x, y, team=0, zombie=False):
        self.tile_x, self.tile_y, self.team = x, y, team
        self.is_zombie, self.alive = zombie, True
        self.offsets = [(dx, dy) for dy in range(3) for dx in range(3)]

    def center_tile_x(self):
        return self.tile_x + 1

    def center_tile_y(self):
        return self.tile_y + 1


def test_each_cell_is_the_field_through_its_boundary_module(session):
    """Property: for every unit and every frozen channel (+ material), each
    in-grid cell of the window equals the field at that tile dequantized by
    the field's own boundary module (float32 of the exact value), and every
    out-of-grid cell is zero. Breaks on a window offset, an axis swap, a
    raw Q16 int leaking into the observation, or a wrong currency (the
    smoke plane is trace, not Q16)."""
    g = session.sim.gmap
    names = [n for n in crop_channel_names() if not n.startswith("units_")]
    r = 6
    units = list(session.sim.units)
    crop = EntityFieldAccessor(g).crop(units, names, r)
    ref = _planes(g)
    h, w = g.material.shape
    for i, u in enumerate(units):
        cy, cx = u.center_tile_y(), u.center_tile_x()
        for c, name in enumerate(names):
            for wy in range(2 * r + 1):
                for wx in range(2 * r + 1):
                    y, x = cy - r + wy, cx - r + wx
                    want = (np.float32(ref[name][y, x])
                            if 0 <= y < h and 0 <= x < w else 0.0)
                    assert crop[i, c, wy, wx] == want, (name, i, y, x)


def test_off_grid_cells_and_missing_rows_are_zero(session):
    """Property: a window hanging off the map's corner is zero outside the
    grid and equal to the field inside; a ``None`` row is all zero. Breaks
    if a crop wraps around (negative indexing) or reads garbage for an
    empty roster slot."""
    g = session.sim.gmap
    r = 4
    crop = EntityFieldAccessor(g).crop([_Stub(-1, -1), None],
                                       ["material", "solid"], r)
    assert not crop[1].any()
    # centre (0, 0): rows/cols 0..r-1 of the window are off the grid
    assert not crop[0, :, :r, :].any() and not crop[0, :, :, :r].any()
    assert np.array_equal(crop[0, 0, r:, r:],
                          g.material[:r + 1, :r + 1].astype(np.float32))


def test_occupancy_is_relative_to_the_observer():
    """Property: ``units_own`` marks the footprints of the observer's own
    team (itself included), ``units_enemy`` those of other marine teams,
    ``units_zombie`` the horde's; swapping the observer's team swaps own and
    enemy; a dead unit marks nothing. Breaks if occupancy is encoded by
    absolute team id (a shared-weights policy could not tell friend from
    foe) or a footprint is misplaced."""
    class _G:
        material = np.zeros((20, 20), dtype=np.int8)
    a, b = _Stub(2, 2, team=0), _Stub(8, 2, team=1)
    z, dead = _Stub(2, 8, zombie=True, team=ZOMBIE_TEAM), _Stub(8, 8, team=1)
    dead.alive = False
    acc = EntityFieldAccessor(_G())
    names = ["units_own", "units_enemy", "units_zombie"]
    crop = acc.crop([a, b], names, 9, all_units=[a, b, z, dead])
    # observer a (centre 3,3): window origin (-6,-6)
    def at(i, c, x, y, cx, cy):
        return crop[i, c, y - cy + 9, x - cx + 9]
    assert at(0, 0, 2, 2, 3, 3) == 1 and at(0, 1, 8, 2, 3, 3) == 1
    assert at(0, 2, 2, 8, 3, 3) == 1 and at(0, 1, 8, 8, 3, 3) == 0
    # observer b (centre 9,3): a is the enemy now, b its own
    assert at(1, 0, 8, 2, 9, 3) == 1 and at(1, 1, 2, 2, 9, 3) == 1
    assert crop[0, 0].sum() == 9 and crop[0, 1].sum() == 9
    assert crop[0, 2].sum() == 9


def test_light_and_unknown_planes_are_refused_by_name(session):
    """Property: ``light`` (not sim-pure until #81) and an unknown name are
    refused, naming the plane. Breaks if an observation could silently read
    render state or a typo."""
    acc = EntityFieldAccessor(session.sim.gmap)
    with pytest.raises(ValueError, match="light.*#81"):
        acc.crop(session.sim.units, ["light"], 2)
    with pytest.raises(ValueError, match="presure"):
        acc.crop(session.sim.units, ["presure"], 2)


def test_reading_a_crop_writes_no_sim_state(session):
    """Property: the match's digests are the same before and after a crop of
    every plane. Breaks if the read-side dequantize writes back into a
    synced field (the sensor contract)."""
    before = match_replay.match_digests(session.sim)
    EntityFieldAccessor(session.sim.gmap).crop(
        session.sim.units, crop_channel_names(), 8,
        all_units=session.sim.units)
    assert match_replay.match_digests(session.sim) == before
