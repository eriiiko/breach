"""B3 — the v1 sensor catalog through the §5a accessor.

Arc B impl doc (docs/arc_b_impl_2026-07-21.md v2), patch B3. Gates:

- Each field sensor samples the RIGHT tile family (§4, D7/D8): temperature/fire
  read a hot/burning SOLID BODY tile while the faced air reads 0; pressure /
  smoke / water_depth / o2 read the AIR tile; o2 = gas[O2] density (D9).
- Area-mean over a disc, masked to live non-solid tiles (D6), floored (§6) — a
  destroyed wall re-enters the count.
- clock counter = tick // period.
- sensor_motion geometry: dist² threshold, corner anchor (N3), LOS asymmetry
  (sensor = Bresenham origin), shut-door blocks LOS, faction filter,
  min_footprint.
- Dead sensor → value 0 (fail-deadly, D13); combined with a decider
  `require_alive` → fail-passive (§2d).
- The accessor returns the raw mirror integer — NO dequantize (§5a).
- Dormancy preserved: a sensor builds a bus; a sensor/node/wire-free level does
  not (the B1 door-only byte-identical gate re-runs in tests/test_b1_signal_bus).

Fixtures are programmatic LevelData / EntityInstance + hand-built gmaps (the
B1/B2 idiom) — no repo level is mutated, no golden moves.

Run:
    conda run -n data python -m pytest tests/test_b3_sensors.py -q
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_loader import EntityInstance, LevelData, Wire  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities import sensors as sensors_mod  # noqa: E402
from simulation.gamemap import N_GASES  # noqa: E402
from simulation.gases import BLACK_SMOKE, O2  # noqa: E402
from simulation.sensor_accessor import (  # noqa: E402
    Channel, EntityFieldAccessor, SiteIndex,
)
from simulation.signal_bus import SignalBus, build_signal_bus  # noqa: E402
from simulation.sensor_system import (  # noqa: E402
    _ClockSensorRuntime, _FieldSensorRuntime, _MotionSensorRuntime,
    build_sensors, is_sensor, sample_sensors,
)
from simulation.unit import Unit  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeGmap:
    """Minimal mirror exposing only the fields the accessor gathers."""

    def __init__(self, h=6, w=6):
        self.atmosphere = np.zeros((h, w), dtype=np.int32)
        self.gas = np.zeros((N_GASES, h, w), dtype=np.int32)
        self.water_depth = np.zeros((h, w), dtype=np.int32)
        self.temperature = np.zeros((h, w), dtype=np.int32)
        self.fire = np.zeros((h, w), dtype=np.int32)
        self.solid = np.zeros((h, w), dtype=bool)


def _sensor_inst(class_name, eid, ordinal, **overrides):
    cls = REGISTRY[class_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name=class_name, ordinal=ordinal,
                          tags=(), fields=fields)


def _fake_sim(**kw):
    return types.SimpleNamespace(**kw)


def _split_box_tm(h=12, w=12, wall_x=6, gap_rows=()):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    tm[1:h - 1, wall_x] = 1
    for r in gap_rows:
        tm[r, wall_x] = 4
    return tm


def _level(tm, entities=(), wires=(), name="b3_fix", tile_size_m=1.0, **kw):
    return LevelData(name=name, version="1", path=Path("."), tilemap=tm,
                     tile_size_m=tile_size_m, diffuse_path=Path("."),
                     entities=list(entities), wires=list(wires), **kw)


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


# ---------------------------------------------------------------------------
# resolve_sample_tile — the per-channel sample family (D7/D8)
# ---------------------------------------------------------------------------

def test_air_family_samples_faced_tile():
    f = {"x": 2, "y": 1, "sample_dx": 1, "sample_dy": 0, "area_m": 0.0}
    assert REGISTRY["pressure"].resolve_sample_tile(f) == (1, 3)   # (y, x+dx)
    assert REGISTRY["smoke"].resolve_sample_tile(f) == (1, 3)


def test_body_family_samples_anchor_tile():
    # temperature/fire sample the MOUNT (anchor), NOT the faced air (D7/D8).
    f = {"x": 2, "y": 1, "sample_dx": 1, "sample_dy": 0, "area_m": 0.0}
    assert REGISTRY["temperature"].resolve_sample_tile(f) == (1, 2)  # (y, x)
    assert REGISTRY["fire"].resolve_sample_tile(f) == (1, 2)


# ---------------------------------------------------------------------------
# Field sensor sampling — the tile family + no dequantize (D7/D8/D9, §5a)
# ---------------------------------------------------------------------------

def _field_rt(channel, tile, disc=None, alive=True):
    return _FieldSensorRuntime(
        inst=types.SimpleNamespace(alive=alive), value_slot=0,
        channel=channel, sample_tile=tile, disc=disc)


def test_body_channels_read_solid_while_faced_air_reads_zero():
    # The D7/D8 regression: a hot/burning SOLID body at the anchor; the faced
    # air tile carries neither temperature nor fire.
    g = _FakeGmap()
    g.solid[1, 2] = True                      # the mount body tile
    g.temperature[1, 2] = 900                 # hot solid
    g.fire[1, 2] = 250                         # burning solid
    # faced air (1, 3): temperature/fire stay 0 (air carries none).
    sim = _fake_sim(_sensor_accessor=EntityFieldAccessor(g))
    assert _field_rt(Channel.TEMPERATURE, (1, 2)).evaluate(sim) == 900
    assert _field_rt(Channel.FIRE, (1, 2)).evaluate(sim) == 250
    # Had the body sensor sampled the faced air (1, 3) it would read 0:
    assert _field_rt(Channel.TEMPERATURE, (1, 3)).evaluate(sim) == 0


def test_air_channels_read_air_tile_no_dequantize():
    g = _FakeGmap()
    g.atmosphere[1, 3] = 65536                # 1.0 atm Q16.16 — returned RAW
    g.gas[BLACK_SMOKE][1, 3] = 4242
    g.gas[O2][1, 3] = 13107
    g.water_depth[1, 3] = 999
    sim = _fake_sim(_sensor_accessor=EntityFieldAccessor(g))
    assert _field_rt(Channel.PRESSURE, (1, 3)).evaluate(sim) == 65536
    assert _field_rt(Channel.SMOKE, (1, 3)).evaluate(sim) == 4242
    assert _field_rt(Channel.O2, (1, 3)).evaluate(sim) == 13107   # gas[O2] density
    assert _field_rt(Channel.WATER_DEPTH, (1, 3)).evaluate(sim) == 999


# ---------------------------------------------------------------------------
# Area-mean — non-solid mask (D6), floor (§6), destroyed wall re-enters
# ---------------------------------------------------------------------------

def test_area_mean_masks_solid_floors_and_tracks_destruction():
    g = _FakeGmap()
    disc = [(0, 0), (0, 1), (1, 0), (1, 1)]
    g.atmosphere[0, 0] = 10
    g.atmosphere[0, 1] = 10
    g.atmosphere[1, 0] = 11
    g.atmosphere[1, 1] = 100                  # this tile will be a solid wall
    g.solid[1, 1] = True
    rt = _field_rt(Channel.PRESSURE, (0, 0), disc=disc)
    sim = _fake_sim(_sensor_accessor=EntityFieldAccessor(g))
    # (10 + 10 + 11) // 3 = 10 — solid tile excluded, floored.
    assert rt.evaluate(sim) == 10
    # Destroy the wall: the tile re-enters the live non-solid count.
    g.solid[1, 1] = False
    assert rt.evaluate(sim) == (10 + 10 + 11 + 100) // 4   # 32


# ---------------------------------------------------------------------------
# clock — the tick counter
# ---------------------------------------------------------------------------

def test_clock_counter_floors_tick_over_period():
    rt = _ClockSensorRuntime(inst=types.SimpleNamespace(alive=True),
                             value_slot=0, period=2)
    for t, expect in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]:
        assert rt.evaluate(_fake_sim(tick=t)) == expect


# ---------------------------------------------------------------------------
# sensor_motion geometry — dist², corner anchor, LOS, filters
# ---------------------------------------------------------------------------

def _unit(uid, x, y, team=0, footprint=3, alive=True):
    u = Unit(name=f"u{uid}", x=float(x), y=float(y), team=team,
             footprint=footprint)
    u.id = uid
    u.alive = alive
    return u


def _motion(sy, sx, r_tiles, needs_los=False, faction=-1, min_fp=0):
    return _MotionSensorRuntime(
        inst=types.SimpleNamespace(alive=True), value_slot=0, sy=sy, sx=sx,
        r2=r_tiles * r_tiles, needs_los=needs_los, faction=faction,
        min_footprint=min_fp)


class _AlwaysLosGmap:
    def __init__(self):
        self.calls = []

    def has_los(self, y1, x1, y2, x2):
        self.calls.append((y1, x1, y2, x2))
        return True


def test_motion_dist2_threshold_inclusive():
    g = _AlwaysLosGmap()
    # sensor at (5,5), r_tiles=2 (r²=4). Anchor (5,7): d²=4 ≤ 4 counts;
    # (5,8): d²=9 does not.
    inside = _fake_sim(units=[_unit(0, x=7, y=5)], gmap=g)
    assert _motion(5, 5, 2).evaluate(inside) == 1
    outside = _fake_sim(units=[_unit(0, x=8, y=5)], gmap=g)
    assert _motion(5, 5, 2).evaluate(outside) == 0


def test_motion_uses_corner_anchor_not_center():
    # A footprint-3 unit at anchor (5,7): corner d²=4 (in for r=2); its CENTER
    # tile would be (8,6)-ish, d² far larger — proving the corner anchor (N3).
    g = _AlwaysLosGmap()
    sim = _fake_sim(units=[_unit(0, x=7, y=5, footprint=3)], gmap=g)
    assert _motion(5, 5, 2).evaluate(sim) == 1          # corner in range
    # centre-tile (tile + fp//2 = 8) would be d²=9 > 4 → would NOT count.


def test_motion_dead_units_excluded():
    g = _AlwaysLosGmap()
    sim = _fake_sim(units=[_unit(0, x=7, y=5, alive=False)], gmap=g)
    assert _motion(5, 5, 2).evaluate(sim) == 0


def test_motion_faction_filter():
    g = _AlwaysLosGmap()
    units = [_unit(0, x=6, y=5, team=0), _unit(1, x=6, y=5, team=1)]
    assert _motion(5, 5, 3, faction=-1).evaluate(
        _fake_sim(units=units, gmap=g)) == 2            # any
    assert _motion(5, 5, 3, faction=1).evaluate(
        _fake_sim(units=units, gmap=g)) == 1            # zombies only
    assert _motion(5, 5, 3, faction=0).evaluate(
        _fake_sim(units=units, gmap=g)) == 1            # marines only


def test_motion_min_footprint():
    g = _AlwaysLosGmap()
    units = [_unit(0, x=6, y=5, footprint=1), _unit(1, x=6, y=5, footprint=3)]
    assert _motion(5, 5, 3, min_fp=2).evaluate(
        _fake_sim(units=units, gmap=g)) == 1            # fp<2 dropped


def test_motion_los_uses_sensor_as_origin():
    # The asymmetry pin (§4): has_los is called with the SENSOR tile FIRST.
    g = _AlwaysLosGmap()
    _motion(5, 5, 4, needs_los=True).evaluate(
        _fake_sim(units=[_unit(0, x=8, y=5)], gmap=g))
    assert g.calls == [(5, 5, 5, 8)]                    # (sy, sx, uy, ux)


def test_motion_shut_wall_blocks_los():
    # A real gmap: a solid wall between the sensor and the unit blocks sight.
    sim = Simulation(_level(_split_box_tm()), seed=1, breach_physics=None,
                     enable_recorder=False)
    gmap = sim.gmap
    fake = _fake_sim(units=[_unit(0, x=9, y=5)], gmap=gmap)  # unit right of wall
    # sensor left of the wall (col 6 is solid) — LOS blocked → 0.
    assert _motion(5, 3, 10, needs_los=True).evaluate(fake) == 0
    # Without LOS the same unit is in range → counted.
    assert _motion(5, 3, 10, needs_los=False).evaluate(fake) == 1


def test_motion_open_gap_allows_los():
    sim = Simulation(_level(_split_box_tm(gap_rows=(5,))), seed=1,
                     breach_physics=None, enable_recorder=False)
    fake = _fake_sim(units=[_unit(0, x=9, y=5)], gmap=sim.gmap)
    assert _motion(5, 3, 10, needs_los=True).evaluate(fake) == 1   # gap at row 5


# ---------------------------------------------------------------------------
# Dead sensor → value 0 at 9e(a) (fail-deadly, D13)
# ---------------------------------------------------------------------------

def test_sample_sensors_dead_sensor_publishes_zero():
    g = _FakeGmap()
    g.atmosphere[1, 2] = 777
    bus = SignalBus([(0, "value")])
    rt = _FieldSensorRuntime(inst=types.SimpleNamespace(alive=True),
                             value_slot=bus.slot(0, "value"),
                             channel=Channel.PRESSURE, sample_tile=(1, 2),
                             disc=None)
    sim = _fake_sim(_signal_bus=bus, _sensors=[rt],
                    _sensor_accessor=EntityFieldAccessor(g))
    sample_sensors(sim)
    assert bus.read(0, "value") == 777        # alive → live reading
    rt.inst.alive = False
    sample_sensors(sim)
    assert bus.read(0, "value") == 0          # dead → 0, never stale (D13)


# ---------------------------------------------------------------------------
# Bus gating / dormancy — a sensor is logic; a bare fixture is not
# ---------------------------------------------------------------------------

def test_sensor_marker_and_bus_built_for_sensor_only_level():
    assert is_sensor("pressure") and is_sensor("clock")
    assert is_sensor("sensor_motion") and not is_sensor("door")
    # A lone sensor (no wires, no nodes) still builds a bus (sensors ∪ … ≠ ∅).
    lvl = _level(_split_box_tm(),
                 entities=[_sensor_inst("clock", "clk", 0, period=1)])
    bus = build_signal_bus(lvl)
    assert isinstance(bus, SignalBus)
    assert bus.slots == ((0, "value"),)       # the sensor's value slot


def test_button_only_level_stays_dormant():
    # A placeable-but-inert fixture (button) is NOT a sensor/node → no bus.
    btn = EntityInstance(id="b0", class_name="button", ordinal=0, tags=(),
                         fields={"x": 1, "y": 1})
    assert build_signal_bus(_level(_split_box_tm(), entities=[btn])) is None


# ---------------------------------------------------------------------------
# E2E through a live sim — build_sensors, 9e(a) publish, site index, and the
# dead-sensor + require_alive fail-passive composition
# ---------------------------------------------------------------------------

def test_e2e_field_sensor_publishes_into_signals():
    # A pressure sensor over a poked air tile; physics off (breach_physics=None)
    # so the mirror value is stable across the step.
    tm = _split_box_tm()
    ps = _sensor_inst("pressure", "ps", 0, x=3, y=5, sample_dx=0, sample_dy=0)
    sim = Simulation(_level(tm, [ps]), seed=1, breach_physics=None,
                     enable_recorder=False)
    assert sim._signal_bus is not None
    # The §5a accessor + its frozen site index were built at load.
    idx = sim._sensor_accessor.site_index
    assert isinstance(idx, SiteIndex) and len(idx) == 1
    assert idx.sites == ((5, 3),) and idx.channels == (Channel.PRESSURE,)

    sim.gmap.atmosphere[5, 3] = 42424
    _step(sim)
    assert sim._signal_bus.read(0, "value") == 42424
    assert sim._digest_signals() == ((0, "value", 42424),)


def test_e2e_dead_sensor_zeros_value_and_require_alive_gates_decider():
    # clock(value) → decider(require_alive, gt 0). Alive: value climbs, decider
    # fires. Dead: value 0 (fail-deadly) AND require_alive zeroes the decider
    # (fail-passive, §2d).
    clk = _sensor_inst("clock", "clk", 0, period=1)
    dec = _sensor_inst("decider", "dec", 1)   # helper builds any class' fields
    dec.fields.update(comparator="gt", threshold=0, require_alive=True)
    wire = Wire(0, "value", 1, "in", "single")   # clk.value → dec.in
    sim = Simulation(_level(_split_box_tm(), [clk, dec], [wire]), seed=1,
                     breach_physics=None, enable_recorder=False)

    _step(sim, 3)                              # let the clock climb + propagate
    assert sim._signal_bus.read(0, "value") >= 1      # clock alive & counting
    assert sim._signal_bus.read(1, "out") == 1        # decider fired

    sim.entities[0].alive = False              # kill the clock sensor
    _step(sim, 2)
    assert sim._signal_bus.read(0, "value") == 0       # fail-deadly (D13)
    assert sim._signal_bus.read(1, "out") == 0         # require_alive → passive


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
