"""The playground level (P5) — load / construct / run smoke + determinism.

Deliberately NOT part of any golden/digest machinery (the canonical A/B
scenario and its aggregate golden stay untouched): this is a plain gate that
the standard-values sandbox stays launchable —

  - the level folder parses (v2 codes) and carries every showcase material
    (wood / glass / steel / furniture / door / hull / SPACE);
  - the spawn layout invariants hold: every spawn footprint stands on open
    walkable floor, and the penned zombies are far enough from the squad
    that the brawl cannot start at tick 0 (trigger_radius guard — a layout
    tweak in tools/gen_playground_level.py that breaks this fails HERE, not
    silently in play);
  - a headless sim constructs and runs 30 ticks without error, everyone
    alive (the map is quiet until the player pokes it);
  - the same seed run twice is bitwise identical (fields + synced unit
    state) — the playground obeys the same determinism contract as the ship.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_playground_level.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import SPACE_CODE, load, materials_from_tilemap  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_DOOR, MAT_FURNITURE, MAT_GLASS, MAT_HULL, MAT_STEEL, MAT_WOOD,
)
from simulation.status import serialize_statuses  # noqa: E402
from simulation.unit import Unit  # noqa: E402

N_TICKS = 30

# The fields a divergence would corrupt (the field_ab_harness SIM_FIELDS
# core, minus the static topology arrays already covered by wall_hp).
_FIELDS = ("atmosphere", "wave_p", "wave_v", "wind_x", "wind_y", "gas",
           "fire", "water_depth", "heat", "temperature", "wall_hp")


def _fresh_sim(seed: int) -> Simulation:
    level = load("playground")
    sim = Simulation(level, seed=seed, breach_physics=bp,
                     enable_recorder=False)
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    sim.set_paused(False)
    return sim


def _run(sim: Simulation, n: int) -> None:
    for _ in range(n):
        sim.step()


# ---------------------------------------------------------------------------
# Level structure
# ---------------------------------------------------------------------------
def test_playground_loads_with_all_showcase_materials():
    lvl = load("playground")
    assert lvl.version == "2"
    assert lvl.tilemap.shape == (70, 100), f"shape={lvl.tilemap.shape}"
    assert lvl.diffuse_path.exists()
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    for mid, label in ((MAT_HULL, "hull"), (MAT_WOOD, "wood"),
                       (MAT_DOOR, "door"), (MAT_STEEL, "steel"),
                       (MAT_GLASS, "glass"), (MAT_FURNITURE, "furniture")):
        assert (mat == mid).any(), f"playground must contain {label}"
    assert vac.any(), "playground must have outer SPACE (breach play)"
    assert int(lvl.tilemap[67, 10]) != SPACE_CODE  # breach wall is hull...
    assert bool(vac[68, 10]), "...with vacuum directly beyond it"


def test_playground_spawns_stand_on_open_floor():
    lvl = load("playground")
    marines = [s for s in lvl.spawns if s.team == 0]
    zombies = [s for s in lvl.spawns if s.team == 1]
    assert len(marines) >= 4, "a squad to play with"
    assert len(zombies) >= 4, "a horde to release"
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    for s in lvl.spawns:
        for dy in range(s.footprint):
            for dx in range(s.footprint):
                tx, ty = int(s.x) + dx, int(s.y) + dy
                code = int(lvl.tilemap[ty, tx])
                assert code == 0 and not vac[ty, tx], (
                    f"spawn {s.name} footprint tile ({tx},{ty}) is not open "
                    f"interior floor (code {code})")


def test_playground_zombies_outside_trigger_radius():
    """The pen sits beyond CFG.zombie.trigger_radius from every marine (and
    glass blocks LOS besides) — tick 0 must be QUIET. Guards layout tweaks."""
    lvl = load("playground")
    trig = float(CFG.zombie.trigger_radius)
    marines = [s for s in lvl.spawns if s.team == 0]
    zombies = [s for s in lvl.spawns if s.team == 1]
    for z in zombies:
        zc = (z.x + z.footprint / 2.0, z.y + z.footprint / 2.0)
        for m in marines:
            mc = (m.x + m.footprint / 2.0, m.y + m.footprint / 2.0)
            d = ((zc[0] - mc[0]) ** 2 + (zc[1] - mc[1]) ** 2) ** 0.5
            assert d > trig + 5.0, (
                f"{z.name} at {d:.1f} tiles from {m.name} — inside/near the "
                f"trigger radius {trig}; move the pen or the squad")


# ---------------------------------------------------------------------------
# Headless run
# ---------------------------------------------------------------------------
def test_playground_constructs_and_runs_30_ticks():
    sim = _fresh_sim(seed=42)
    _run(sim, N_TICKS)
    assert sim.tick == N_TICKS
    assert all(u.alive for u in sim.units), "the quiet map must hurt nobody"
    assert len(sim.marines()) == 4 and len(sim.zombies()) == 5
    g = sim.gmap
    assert int(g.fire.max()) == 0 and int(g.water_depth.max()) == 0, \
        "nothing burns or floods until the player acts"


def test_playground_deterministic_run_twice():
    a = _fresh_sim(seed=1234)
    b = _fresh_sim(seed=1234)
    _run(a, N_TICKS)
    _run(b, N_TICKS)
    for name in _FIELDS:
        fa, fb = getattr(a.gmap, name), getattr(b.gmap, name)
        assert np.array_equal(fa, fb), f"field {name!r} diverged run-to-run"
    for ua, ub in zip(a.units, b.units):
        assert (ua.x, ua.y) == (ub.x, ub.y), f"{ua.name} position diverged"
        assert ua.current_hp == ub.current_hp, f"{ua.name} hp diverged"
        assert ua.alive == ub.alive
        assert serialize_statuses(ua) == serialize_statuses(ub)


if __name__ == "__main__":
    test_playground_loads_with_all_showcase_materials()
    test_playground_spawns_stand_on_open_floor()
    test_playground_zombies_outside_trigger_radius()
    test_playground_constructs_and_runs_30_ticks()
    test_playground_deterministic_run_twice()
    print("OK — playground level: load / spawns / quiet-run / determinism")
