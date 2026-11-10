"""The wind studio (#140 HUMAN-TEST level; docs/wind_on_units_design_2026-10-31.md
§6.1 scene 1): a sealed 1.44 atm room whose door opens onto a marine.

Only the level's PURPOSE is asserted -- the committed folder is the tool's
output, and opening the sealed room's door knocks the doorway marine's air
load over the fall threshold. Room sizes, pressures and spawn positions are
Erik's to retune; no test pins them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "cpp" / "build" / "Release", ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load  # noqa: E402
from simulation import Simulation, exchange  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402

LEVEL_DIR = ROOT / "levels" / "wind_studio"
GEN_PATH = ROOT / "tools" / "gen_wind_studio.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("_gen_wind_studio", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_generated_level_is_the_shipped_one(tmp_path):
    """PROPERTY: the committed levels/wind_studio is exactly what the
    generator writes -- data files byte for byte, the diffuse art pixel for
    pixel (PNG encoders differ across machines) -- regenerated into a tmp dir,
    never into the working tree.

    BREAKS IF: someone hand-edits the level (or the generator) without
    re-running the other, or the generator stops being deterministic.
    """
    gen = _load_generator()
    out = tmp_path / "wind_studio"
    gen.main(out_dir=out)
    for name in ("level.toml", "tilemap.csv", "air_init.npy"):
        assert (out / name).read_bytes() == (LEVEL_DIR / name).read_bytes(), (
            f"committed {name} does not match the tool's output")
    from PIL import Image
    assert np.array_equal(
        np.asarray(Image.open(out / "diffuse.png").convert("RGBA")),
        np.asarray(Image.open(LEVEL_DIR / "diffuse.png").convert("RGBA"))), (
        "committed diffuse.png pixels do not match the tool's output")


def test_opening_the_sealed_door_floors_the_doorway_marine():
    """PROPERTY: the studio's sealed room holds its overpressure while its
    door is shut, and opening that door (the door latch the O key flips)
    puts a wind on the marine spawned in the doorway whose force exceeds
    its hold (``unit_impulse.wind_impulse_q`` of the read against one tick
    of hold: #140's q_fall for the reference marine) and floors it, within
    2 s (G5p-b: the wind is a force on the unit impulse law).

    The doorway marine is the spawn whose footprint touches the door span;
    the sealed room is the door's high-pressure side, found from the field.

    BREAKS IF: the room leaks or its air_init seed is lost (the hold check),
    the door no longer opens from ``want_open``, the doorway spawn is moved
    off the door, the wind read (``GameMap.footprint_wind_load`` priced by
    the impulse table) stops seeing the vent, or the law stops flooring a
    unit in an over-hold wind.
    """
    lvl = load("wind_studio", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    for s in lvl.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    g = sim.gmap
    sim.set_paused(False)
    for _ in range(200):
        sim.step()

    # The highest-pressure door: the room behind it, the marine before it.
    best = None
    for d in sim._doors:
        for (fy, fx) in d.span:
            for (dy, dx) in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                p = int(g.atmosphere[fy + dy, fx + dx])
                if best is None or p > best[0]:
                    best = (p, d)
    p_room, door = best
    span = list(door.span)
    assert p_room / 65536 > 1.3, (
        f"the sealed room does not hold its overpressure: {p_room / 65536:.3f} atm")
    assert door.state == 0, "the sealed room's door is not closed"

    def touches(u):
        tiles = {(ty, tx) for (tx, ty) in u.occupied_tiles()}
        return any((fy + dy, fx + dx) in tiles for (fy, fx) in span
                   for (dy, dx) in ((0, 1), (0, -1), (1, 0), (-1, 0)))
    doorway = [u for u in sim.units if touches(u)]
    assert doorway, "no marine is spawned in the sealed room's doorway"
    marine = doorway[0]

    from simulation import unit_impulse
    from simulation.status import composed_flags
    t = sim.impulse_table
    door.want_open = True
    tps = int(CFG.clock.ticks_per_second)
    peak = 0
    floored = False
    for _ in range(2 * tps):
        sim.step()
        peak = max(peak, g.footprint_wind_load(marine.tile_y, marine.tile_x,
                                               marine.footprint))
        floored = floored or composed_flags(marine).is_prone
    q_peak = exchange.wind_q_pa(g, peak)
    assert unit_impulse.wind_impulse_q(t, peak) >         t.hold_q(marine) * marine.mass_g, (
        f"opening the door gave {marine.name} a peak q of {q_peak:.1f} Pa, "
        f"under its hold")
    assert floored, f"{marine.name} stood through a {q_peak:.0f} Pa gale"
