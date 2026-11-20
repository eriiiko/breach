"""The water studio (#97 W0): tools/gen_water_studio.py -> levels/water_studio,
THE level for judging water (docs/water_look_design_2026-11-19.md, W0).

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_water_studio as gen  # noqa: E402

DATA_FILES = ("level.toml", "tilemap.csv", "water_init.npy", "match.toml")
IMAGE_FILES = ("diffuse.png", "normal.png")
# A flood that still runs is metres per second; a settled room's slosh and
# the solver's rest noise are hundredths (measured 2026-11-20: ~1.95 m/s at
# the shot's frame, ~0.25 m/s a minute in, when the front has long reached the
# basin). Generous both ways.
FLOWING_MPS = 0.5


def test_the_generator_is_byte_deterministic_and_is_the_checked_in_level(tmp_path):
    """Property: two runs of the generator write byte-identical folders, and
    the checked-in levels/water_studio is what it writes (the data files byte
    for byte; the two images pixel for pixel, since a PNG encoder may differ
    between machines). Breaks if the generator draws, iterates something
    unordered, or the level is edited without re-running it."""
    from PIL import Image
    gen.main(tmp_path / "a")
    gen.main(tmp_path / "b")
    checked_in = ROOT / "levels" / gen.LEVEL_NAME
    for name in DATA_FILES + IMAGE_FILES:
        a = (tmp_path / "a" / name).read_bytes()
        assert a == (tmp_path / "b" / name).read_bytes(), name
    for name in DATA_FILES:
        assert (tmp_path / "a" / name).read_bytes() == \
            (checked_in / name).read_bytes(), name
    for name in IMAGE_FILES:
        assert np.array_equal(np.asarray(Image.open(tmp_path / "a" / name)),
                              np.asarray(Image.open(checked_in / name))), name


def test_the_level_loads_airtight_with_its_water_and_its_match():
    """Property: the studio loads through the level door with standing water
    in exactly its three flooded rooms (none on a solid tile), a parsed match
    with a zombie and two marine teams, and passes the airtight lint, and the
    octopus corner holds nothing (ruling of 2026-11-20). Breaks if a room
    leaks, the [water] block or the match file stops being written, water is
    seeded on a solid tile or outside the flooded rooms, or something is
    placed in the octopus's corner."""
    import level_airtight
    import level_loader
    from simulation import water_fixed
    from simulation.materials import MAT_AIR
    assert level_airtight.check(gen.LEVEL_NAME, str(ROOT / "levels")) is True
    lv = level_loader.load(gen.LEVEL_NAME, str(ROOT / "levels"))
    tm = gen.build_tilemap()
    depth = water_fixed.dequantize_f32(lv.water_depth_q)
    assert depth.shape == tm.shape
    assert not depth[tm != MAT_AIR].any()
    flooded = np.zeros(tm.shape, dtype=bool)
    for box, key in ((gen.POOL, "pool"), (gen.WADE, "wade"),
                     (gen.TANK, "tank")):
        m = gen._box_mask(tm.shape, box) & (tm == MAT_AIR)
        assert np.allclose(depth[m], gen.DEPTH_M[key], atol=1e-4), key
        flooded |= m
    assert not depth[~flooded].any()
    assert lv.match is not None and lv.match.zombies >= 1
    assert len(lv.match.teams) >= 2
    x0, y0, x1, y1 = gen.OCTOPUS_CORNER
    assert (tm[y0:y1 + 1, x0:x1 + 1] == MAT_AIR).all()
    for e in lv.entities:
        x, y = e.fields.get("x"), e.fields.get("y")
        if x is not None and y is not None:
            assert not (x0 <= x <= x1 and y0 <= y <= y1), e.id
    for s in lv.spawns:
        assert s.x + s.footprint <= x0 or s.x > x1 or \
            s.y + s.footprint <= y0 or s.y > y1, s.name


def test_the_corridor_still_floods_at_the_flooding_corridor_shot():
    """Property: at the `flooding_corridor` look shot's frame, simulated the
    way `tools/lighting_demo.py` runs a shot (seed 42, the default ruleset,
    one tick per frame), the dam's gate is open, water has entered the
    corridor and is still running there faster than FLOWING_MPS (read
    through `water_fixed.dequantize_f32`) -- a shot of still water would
    prove nothing about the flow detail of W1. Breaks if the dam charge no
    longer opens the gate (payload, material or place), fires after the
    frame, or the shot's frame moves past the flood."""
    bp = pytest.importorskip("breach_physics")
    import level_loader
    from look_shots import load_shots
    from simulation import water_fixed
    from simulation.simulation import Simulation
    shot = load_shots().by_name("flooding_corridor")
    assert shot.level == gen.LEVEL_NAME
    lv = level_loader.load(gen.LEVEL_NAME, str(ROOT / "levels"))
    sim = Simulation(lv, seed=42, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    for _ in range(shot.frames):
        sim.step()
    g = sim.gmap
    gx0, gy0, gx1, gy1 = gen.GATE
    assert not np.asarray(g.solid)[gy0:gy1 + 1, gx0:gx1 + 1].any(), \
        "the dam's gate still stands"
    x0, y0, x1, y1 = gen.CORRIDOR
    cor = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    depth = water_fixed.dequantize_f32(np.asarray(g.water_depth))[cor]
    vx = water_fixed.dequantize_f32(np.asarray(g.flow_vx))[cor]
    vy = water_fixed.dequantize_f32(np.asarray(g.flow_vy))[cor]
    assert (depth > 0.01).any(), "no water has entered the corridor"
    speed = np.hypot(vx, vy)
    assert float(speed.max()) > FLOWING_MPS, float(speed.max())


def test_the_match_is_fought_headless_to_a_result():
    """Property: the studio's match (a bot marine and a zombie in the wading
    room, an idle marine team in the sealed booth) runs through
    `tools/run_match.py`'s `run_match` to a result within its clock, and the
    two in the wading room actually fight: a hit is credited from the bot
    marine to the zombie. Breaks if the match file or a spawn room is
    refused, the wading room's two cannot reach or see each other, or the
    match cannot end (a missing second marine team ends it at once with no
    fight)."""
    pytest.importorskip("breach_physics")
    import level_loader
    from run_match import run_match
    from simulation.events import UnitHitEvent
    lv = level_loader.load(gen.LEVEL_NAME, str(ROOT / "levels"))
    hits = []

    def on_tick(sim):
        for e in sim.tick_events:
            if not isinstance(e, UnitHitEvent):
                continue
            a, v = sim.get_unit(e.attacker_id), sim.get_unit(e.unit_id)
            if a is not None and v is not None and a.team == 0                     and v.is_zombie:
                hits.append(sim.tick)

    out = run_match(lv, on_tick=on_tick, out=lambda *_a, **_k: None)
    assert out.result is not None
    assert out.result.outcome in ("win", "draw")
    assert 0 < out.result.ticks <= int(round(gen.MATCH_TIME_LIMIT_S * 24))
    assert hits, "the bot marine never hit the zombie"
