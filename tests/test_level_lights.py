"""[[light]] entities — loader, beacon helpers, lamp port, editor writeback
(engine/15 §2.2, patch P4 — docs/patch_levels_p4_lights.md).

All headless, no ``breach_physics``. Pins the design-gate hard requirements:

  - loader: parse + defaults + validation (kind whitelist, period_s > 0,
    beam_deg (0, 360], finite floats, 0-255 int colors) and the REJECTION
    of ``heat``/``jitter`` keys (critique M2 — render-only lights must
    never touch the synced heat channel or C++ RNG jitter);
  - ``beacon_angle``: pinned values, freeze semantics (pure function of the
    sim tick — same tick, same angle), phase-pair 0/0.5 opposition, period
    scaling; params built twice at a fixed tick with production-shaped
    arguments are identical (critique B1: sim clock, never wall dt);
  - ``monotonic_total_tick``: advances by exactly 1 through every round
    boundary (critique M1 — ``sim.tick`` rewinds, ``turn_number``
    increments, the combined tick never snaps back);
  - ``light_source_params``: heat == 0.0 and jitter == 0.0 for EVERY kind,
    no ``falloff`` key (not in the Python bindings), only bound
    LightSource attribute names (the setattr-loop contract);
  - lamp port: 5 static lamps in both vessels + 3 in playground, positions
    verbatim from the deleted main.py block, colors within +/-1/255 of the
    old (1.0, 0.1, 0.05) floats (exactness is impossible under the 0-255
    int schema — integration critique minor 5);
  - editor: [[light]] managed-block writeback preserving unrelated bytes
    (spawn tables included), loader round-trip, the shared
    once-per-session .bak contract, and marker hit-testing.

Run:
    python -m pytest tests/test_level_lights.py -q
"""
from __future__ import annotations

import math
import struct
import sys
import tomllib
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from config import CFG  # noqa: E402
from level_loader import LIGHT_KINDS, LightEntry, load  # noqa: E402
from level_lights import (STATIC_SPREAD, beacon_angle,  # noqa: E402
                          light_source_params, monotonic_total_tick,
                          partition_lights)
from map_editor import (LIGHT_COLOR_PRESETS, LIGHT_PICK_RADIUS,  # noqa: E402
                        color_255, format_light_lines, light_at,
                        light_color_name, next_light_color, write_lights,
                        write_spawns)

# The Python-bound LightSource attribute surface (cpp/src/bindings.cpp:
# 847-865). light_source_params dicts are applied with a setattr loop, so
# any key OUTSIDE this set would raise AttributeError on the pybind class.
BOUND_LIGHTSOURCE_ATTRS = frozenset({
    "x", "y", "max_range", "ray_count", "angle_center", "angle_spread",
    "intensity", "heat", "jitter", "color",
})

# The five emergency lamps as main.py hardcoded them (positions verbatim).
LAMP_POSITIONS = [(25.0, 10.0), (25.0, 30.0), (25.0, 55.0),
                  (25.0, 88.0), (25.0, 110.0)]
LAMP_COLOR = (1.0, 0.1, 0.05)
COLOR_TOL = 1.0 / 255.0


# ---------------------------------------------------------------------------
# Fixture — a minimal loadable level folder (loader wants a real PNG header)
# ---------------------------------------------------------------------------

def _write_png(path: Path, w: int = 8, h: int = 6) -> None:
    """Smallest valid RGB PNG (pure stdlib) — the loader reads its IHDR."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * w for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", zlib.compress(raw))
                     + chunk(b"IEND", b""))


BASE_TOML = ('version = "2"\nname = "Lit"\ntilemap = "tilemap.csv"\n'
             'tile_size_m = 0.333\ndiffuse = "diffuse.png"\n\n')


def _mini_level(tmp_path: Path, light_toml: str = "") -> Path:
    d = tmp_path / "lit"
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(8)) for _ in range(6)) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(BASE_TOML + light_toml,
                                  encoding="utf-8", newline="\n")
    return d


# ---------------------------------------------------------------------------
# Loader — parse, defaults, validation, heat/jitter rejection
# ---------------------------------------------------------------------------

def test_light_parse_full_and_defaults(tmp_path):
    d = _mini_level(tmp_path, (
        "[[light]]\npos = [12.5, 40.0]\ncolor = [255, 200, 150]\n\n"
        "[[light]]\npos = [30.5, 8.5]\ncolor = [255, 40, 40]\n"
        "intensity = 1.5\nrange = 16.0\nkind = \"beacon\"\n"
        "period_s = 1.5\nbeam_deg = 45.0\nphase = 0.5\n"))
    lvl = load(str(d))
    assert len(lvl.lights) == 2
    a, b = lvl.lights
    # Minimal static entry: color normalized 0-1, every default applied.
    assert (a.x, a.y) == (12.5, 40.0)
    assert a.color == (1.0, 200 / 255.0, 150 / 255.0)
    assert (a.intensity, a.range, a.kind) == (1.0, 12.0, "static")
    assert (a.period_s, a.beam_deg, a.phase) == (2.0, 30.0, 0.0)
    # Fully-specified beacon.
    assert (b.x, b.y) == (30.5, 8.5)
    assert b.kind == "beacon"
    assert (b.intensity, b.range) == (1.5, 16.0)
    assert (b.period_s, b.beam_deg, b.phase) == (1.5, 45.0, 0.5)


def test_lights_absent_default_empty(tmp_path):
    lvl = load(str(_mini_level(tmp_path)))
    assert lvl.lights == []


@pytest.mark.parametrize("body,fragment", [
    ("pos = [1.0]\ncolor = [255, 0, 0]\n", "pos"),               # bad pair
    ("color = [255, 0, 0]\n", "pos"),                            # missing
    ("pos = [1.0, 2.0]\n", "color"),                             # missing
    ("pos = [1.0, 2.0]\ncolor = [256, 0, 0]\n", "0-255"),        # range
    ("pos = [1.0, 2.0]\ncolor = [0.5, 0.1, 0.9]\n", "color"),    # non-int
    ("pos = [1.0, 2.0]\ncolor = [255, 0, 0]\nkind = \"strobe\"\n",
     "kind"),                                                    # whitelist
    ("pos = [1.0, 2.0]\ncolor = [255, 0, 0]\nperiod_s = 0.0\n",
     "period_s"),                                                # > 0
    ("pos = [1.0, 2.0]\ncolor = [255, 0, 0]\nbeam_deg = 0.0\n",
     "beam_deg"),                                                # > 0
    ("pos = [1.0, 2.0]\ncolor = [255, 0, 0]\nbeam_deg = 361.0\n",
     "beam_deg"),                                                # <= 360
    ("pos = [1.0, 2.0]\ncolor = [255, 0, 0]\nrange = 0.0\n",
     "range"),                                                   # > 0
    ("pos = [1.0, 2.0]\ncolor = [255, 0, 0]\nintensity = nan\n",
     "intensity"),                                               # finite
])
def test_light_validation_errors(tmp_path, body, fragment):
    d = _mini_level(tmp_path, "[[light]]\n" + body)
    with pytest.raises(ValueError) as ei:
        load(str(d))
    msg = str(ei.value)
    # Entry index + required-fields hint, always (design §2.1).
    assert "[[light]] entry #0" in msg
    assert "Required fields" in msg
    assert fragment in msg


@pytest.mark.parametrize("key,value", [("heat", "0.0"), ("jitter", "0.25")])
def test_light_rejects_heat_jitter_keys(tmp_path, key, value):
    """Key PRESENCE is the offence (even heat = 0.0): the schema must never
    carry the synced-heat / C++-RNG knobs (critique M2)."""
    d = _mini_level(tmp_path, (
        f"[[light]]\npos = [1.0, 2.0]\ncolor = [255, 0, 0]\n"
        f"{key} = {value}\n"))
    with pytest.raises(ValueError) as ei:
        load(str(d))
    msg = str(ei.value)
    assert key in msg and "render-only" in msg


# ---------------------------------------------------------------------------
# beacon_angle — pinned values, freeze, phase pair, period scaling
# ---------------------------------------------------------------------------

def test_beacon_angle_pinned_values():
    dt = 1.0 / 24.0
    assert beacon_angle(0, dt, 2.0, 0.0) == 0.0
    # Half a 2 s rotation after 24 ticks at 24 Hz.
    assert beacon_angle(24, dt, 2.0, 0.0) == pytest.approx(math.pi)
    assert beacon_angle(48, dt, 2.0, 0.0) == pytest.approx(math.tau)
    # Phase is a fraction of a turn.
    assert beacon_angle(0, dt, 2.0, 0.25) == pytest.approx(math.pi / 2.0)
    # Tick granularity pin: 24 Hz, period 2 s -> 7.5 deg per tick (accepted
    # stepping, on record — no wall-clock smoothing, critique N2).
    step = beacon_angle(1, dt, 2.0, 0.0) - beacon_angle(0, dt, 2.0, 0.0)
    assert math.degrees(step) == pytest.approx(7.5)


def test_beacon_angle_freeze_semantics():
    """Pure function of the sim tick: while the sim is paused the tick does
    not advance, so the angle CANNOT move — same inputs, identical output
    (bit-exact, not approx)."""
    for tick in (0, 7, 123, 99999):
        a1 = beacon_angle(tick, 1.0 / 24.0, 1.5, 0.125)
        a2 = beacon_angle(tick, 1.0 / 24.0, 1.5, 0.125)
        assert a1 == a2


def test_beacon_phase_pair_opposite():
    """The chapter's cop-car pair: phases 0.0 / 0.5 stay exactly half a
    turn apart at every tick."""
    dt = 1.0 / 24.0
    for tick in (0, 1, 17, 240, 12345):
        a = beacon_angle(tick, dt, 1.5, 0.0)
        b = beacon_angle(tick, dt, 1.5, 0.5)
        assert (b - a) == pytest.approx(math.pi)


def test_beacon_period_scaling():
    dt = 1.0 / 24.0
    for tick in (1, 10, 100):
        assert (beacon_angle(tick, dt, 4.0, 0.0)
                == pytest.approx(beacon_angle(tick, dt, 2.0, 0.0) / 2.0))


# ---------------------------------------------------------------------------
# monotonic_total_tick — never rewinds across round boundaries
# ---------------------------------------------------------------------------

def test_monotonic_total_tick_across_rounds():
    """sim.tick rewinds to 0 exactly when turn_number (1-based) increments
    (simulation.py _end_round): the combined tick must advance by exactly 1
    through every boundary and start at 0."""
    tpr = 240
    seq = [monotonic_total_tick(turn, tpr, tick)
           for turn in (1, 2, 3) for tick in range(tpr)]
    assert seq == list(range(3 * tpr))


def test_beacon_no_round_boundary_snap():
    """Critique M1 end-to-end: angles computed from the monotonic tick step
    smoothly through the tick-rewind, instead of snapping to phase 0."""
    dt, tpr = 1.0 / 24.0, 240
    before = beacon_angle(monotonic_total_tick(1, tpr, 239), dt, 2.0, 0.0)
    after = beacon_angle(monotonic_total_tick(2, tpr, 0), dt, 2.0, 0.0)
    one_step = math.tau * dt / 2.0
    assert (after - before) == pytest.approx(one_step)


# ---------------------------------------------------------------------------
# light_source_params — structural zeroes, cone math, setattr surface
# ---------------------------------------------------------------------------

def _entry(kind: str) -> LightEntry:
    return LightEntry(x=3.0, y=4.0, color=(1.0, 0.5, 0.25), intensity=1.5,
                      range=16.0, kind=kind, period_s=2.0, beam_deg=30.0,
                      phase=0.25)


def test_params_pin_heat_and_jitter_zero_for_every_kind():
    for kind in LIGHT_KINDS:
        p = light_source_params(_entry(kind), 17, 1.0 / 24.0)
        assert p["heat"] == 0.0, kind      # the ONE synced ray output
        assert p["jitter"] == 0.0, kind    # no C++ RNG pull
        assert "falloff" not in p          # not in the Python bindings (M2)
        # Every key must be a bound LightSource attribute (setattr loop).
        assert set(p) <= BOUND_LIGHTSOURCE_ATTRS, kind


def test_params_static_uniform_and_beacon_cone():
    dt = 1.0 / 24.0
    ps = light_source_params(_entry("static"), 999, dt)
    assert ps["angle_spread"] == STATIC_SPREAD == math.tau
    assert ps["angle_center"] == 0.0
    assert (ps["x"], ps["y"]) == (3.0, 4.0)
    assert ps["max_range"] == 16.0 and ps["intensity"] == 1.5
    assert ps["color"] == (1.0, 0.5, 0.25)
    pb = light_source_params(_entry("beacon"), 0, dt)
    assert pb["angle_spread"] == pytest.approx(math.radians(30.0))
    assert pb["angle_center"] == pytest.approx(math.pi / 2.0)  # phase 0.25
    # angle_center is always reduced into [0, 2*pi).
    pb2 = light_source_params(_entry("beacon"), 100_000, dt)
    assert 0.0 <= pb2["angle_center"] < math.tau


def test_params_identical_when_rebuilt_at_fixed_tick():
    """Integration-shaped freeze test (design §2.2): building the params
    twice with the PRODUCTION arguments — sim_time_per_tick from config,
    the monotonic tick from turn/tick — yields identical dicts."""
    sim_time_per_tick = 1.0 / float(CFG.clock.ticks_per_second)
    total_tick = monotonic_total_tick(3, int(CFG.clock.ticks_per_round), 17)
    p1 = light_source_params(_entry("beacon"), total_tick, sim_time_per_tick)
    p2 = light_source_params(_entry("beacon"), total_tick, sim_time_per_tick)
    assert p1 == p2


def test_partition_lights_bounds_rule():
    """Same rule as the retired main.py block: 0 <= x < w and 0 <= y < h
    (the vessel's y=88/110 lamps are off-grid on the 100x70 playground)."""
    lamps = [LightEntry(x=x, y=y, color=(1.0, 0.1, 0.05))
             for x, y in LAMP_POSITIONS]
    in_b, off = partition_lights(lamps, 100, 70)
    assert [(l.x, l.y) for l in in_b] == LAMP_POSITIONS[:3]
    assert [(l.x, l.y) for l in off] == LAMP_POSITIONS[3:]
    in_b2, off2 = partition_lights(lamps, 50, 120)
    assert len(in_b2) == 5 and off2 == []


# ---------------------------------------------------------------------------
# Lamp port — the shipped levels carry the old main.py lamps
# ---------------------------------------------------------------------------

def _assert_lamp(l: LightEntry) -> None:
    assert l.kind == "static"
    assert l.range == 18.0
    assert l.intensity == 0.9
    for got, want in zip(l.color, LAMP_COLOR):
        # +/-1/255: 0.1 -> toml int 26 -> 0.10196 (int schema, minor 5).
        assert abs(got - want) <= COLOR_TOL, (l.color, LAMP_COLOR)


@pytest.mark.parametrize("level_name", ["unhcr_vessel", "unhcr_vessel_2"])
def test_vessel_lamp_port(level_name):
    # unhcr_vessel is the kept legacy fixture (levels/); unhcr_vessel_2 stays
    # retired (prototypes/).
    levels_dir = "prototypes" if level_name == "unhcr_vessel_2" else "levels"
    lvl = load(level_name, levels_dir=levels_dir)
    assert [(l.x, l.y) for l in lvl.lights] == LAMP_POSITIONS
    for l in lvl.lights:
        _assert_lamp(l)
    # All five sit in-grid on the 50x120 vessel — no level got darker.
    in_b, off = partition_lights(lvl.lights, lvl.width, lvl.height)
    assert len(in_b) == 5 and off == []


def test_playground_lamp_port():
    lvl = load("playground")
    assert [(l.x, l.y) for l in lvl.lights] == LAMP_POSITIONS[:3]
    for l in lvl.lights:
        _assert_lamp(l)
    in_b, off = partition_lights(lvl.lights, lvl.width, lvl.height)
    assert len(in_b) == 3 and off == []


# ---------------------------------------------------------------------------
# Editor — hit-testing, color presets, [[light]] writeback
# ---------------------------------------------------------------------------

def test_light_at_hit_testing():
    lights = [LightEntry(x=2.5, y=3.5, color=(1.0, 0.0, 0.0)),
              LightEntry(x=3.0, y=3.5, color=(0.0, 1.0, 0.0))]
    assert light_at(lights, 3.0, 3.5) == 1
    assert light_at(lights, 2.7, 3.5) == 1          # overlap -> topmost
    assert light_at(lights, 1.8, 3.5) == 0
    assert light_at(lights, 9.0, 9.0) is None
    just_out = 3.5 + LIGHT_PICK_RADIUS + 0.05
    assert light_at(lights[:1], 2.5, just_out) is None
    assert light_at([], 2.5, 3.5) is None


def test_color_helpers_round_trip_presets():
    assert color_255(LAMP_COLOR) == (255, 26, 13)
    first_name, first_ints = LIGHT_COLOR_PRESETS[0]
    first = tuple(v / 255.0 for v in first_ints)
    assert light_color_name(first) == first_name
    # Cycle forward/backward; unknown colors restart at preset 0.
    second = next_light_color(first, 1)
    assert color_255(second) == LIGHT_COLOR_PRESETS[1][1]
    assert color_255(next_light_color(second, -1)) == first_ints
    assert color_255(next_light_color((0.123, 0.456, 0.789))) == first_ints
    # Every preset survives the toml int round trip exactly.
    for _, ints in LIGHT_COLOR_PRESETS:
        assert color_255(tuple(v / 255.0 for v in ints)) == ints


def test_write_lights_managed_block_preserves_other_bytes(tmp_path):
    prefix = ("# hand comment stays\n"
              'version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n'
              "tile_size_m = 0.333\n\n")
    spawn_tbl = '[[spawn]]\nname = "Alpha"\nteam = 0\nx = 3\ny = 4\n\n'
    suffix = ('[bake]\ntileset = "x"\npx_per_tile = 8\nseed = 0\n')
    body = (prefix
            + '[[light]]\npos = [9.5, 9.5]\ncolor = [1, 2, 3]\n'
              '# doomed comment inside a managed table\n\n'
            + spawn_tbl + suffix)
    toml = tmp_path / "level.toml"
    toml.write_text(body, encoding="utf-8", newline="\n")
    original = toml.read_bytes()

    lights = [LightEntry(x=2.5, y=3.5, color=(255 / 255, 26 / 255, 13 / 255),
                         intensity=0.9, range=18.0),
              LightEntry(x=4.5, y=1.5, color=(255 / 255, 40 / 255, 40 / 255),
                         kind="beacon", period_s=1.5, beam_deg=45.0,
                         phase=0.5)]
    bak = write_lights(toml, lights, write_bak=True)
    assert bak.read_bytes() == original
    text = toml.read_text(encoding="utf-8")
    # Everything OUTSIDE the light tables is byte-preserved — the spawn
    # table and [bake] included.
    assert text.startswith(prefix)
    assert spawn_tbl in text
    assert text.endswith(suffix)
    raw = tomllib.loads(text)
    assert raw["spawn"] == [{"name": "Alpha", "team": 0, "x": 3, "y": 4}]
    assert raw["light"] == [
        {"pos": [2.5, 3.5], "color": [255, 26, 13], "intensity": 0.9,
         "range": 18.0, "kind": "static"},
        {"pos": [4.5, 1.5], "color": [255, 40, 40], "intensity": 1.0,
         "range": 12.0, "kind": "beacon", "period_s": 1.5,
         "beam_deg": 45.0, "phase": 0.5}]
    # Static entries carry no beacon keys (loader defaults own them).
    assert "period_s" not in raw["light"][0]


def test_write_lights_appends_deletes_and_keeps_crlf(tmp_path):
    toml = tmp_path / "level.toml"
    toml.write_bytes(b'version = "2"\r\nname = "T"\r\n')
    write_lights(toml, [LightEntry(x=1.5, y=2.5, color=(1.0, 0.0, 0.0))],
                 write_bak=False)
    data = toml.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n")   # newline style kept
    assert tomllib.loads(data.decode())["light"][0]["pos"] == [1.5, 2.5]
    # Deleting every light removes the whole managed block again.
    write_lights(toml, [], write_bak=False)
    assert "light" not in tomllib.loads(
        toml.read_text(encoding="utf-8"))


def test_light_round_trip_through_loader_shared_bak(tmp_path):
    """load -> edit -> save -> reload, LightEntry-exact; the light
    writeback runs AFTER write_spawns with write_bak=False and SHARES the
    session's one .bak (pre-session bytes) — the Ctrl+S sequence."""
    d = _mini_level(tmp_path)
    lvl = load(str(d))
    assert lvl.lights == [] and lvl.spawns == []
    pre = (d / "level.toml").read_bytes()

    lights = [LightEntry(x=2.5, y=3.5, color=(255 / 255, 26 / 255, 13 / 255),
                         intensity=0.9, range=18.0),
              LightEntry(x=4.5, y=1.5, color=(64 / 255, 96 / 255, 255 / 255),
                         kind="beacon", period_s=1.5, beam_deg=45.0,
                         phase=0.5)]
    # Ctrl+S order: spawn writeback FIRST (owns the .bak), lights second.
    write_spawns(d / "level.toml", [], write_bak=True)
    write_lights(d / "level.toml", lights, write_bak=False)
    lvl2 = load(str(d))
    assert lvl2.lights == lights                      # dataclass-exact
    assert (d / "level.toml.bak").read_bytes() == pre

    # Second save session step: edit + rewrite, .bak untouched.
    edited = [lights[1]]
    write_lights(d / "level.toml", edited, write_bak=False)
    lvl3 = load(str(d))
    assert lvl3.lights == edited
    assert (d / "level.toml.bak").read_bytes() == pre
    text = (d / "level.toml").read_text(encoding="utf-8")
    assert text.startswith(BASE_TOML)                 # unrelated bytes kept


def test_format_light_lines_schema():
    static = LightEntry(x=25.0, y=10.0, color=(1.0, 26 / 255, 13 / 255),
                        intensity=0.9, range=18.0)
    text = "".join(format_light_lines([static]))
    raw = tomllib.loads(text)
    assert raw["light"] == [{"pos": [25.0, 10.0], "color": [255, 26, 13],
                             "intensity": 0.9, "range": 18.0,
                             "kind": "static"}]
