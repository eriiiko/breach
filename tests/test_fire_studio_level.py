"""The fire-studio level (B2 P1) — generator determinism + headless load/run.

Deliberately NOT part of any golden/digest machinery (a new showcase level, never
migrated): a plain gate that

  - the generator is DETERMINISTIC (no RNG in the source; re-running yields
    byte-identical tilemap.csv / water_init.npy / diffuse.png / level.toml — so
    the committed folder always matches the tool);
  - the level LOADS headless (v2 codes, the beacon + door + water + spawns);
  - a headless sim constructs + runs, and the sealed side-room door reconciles
    (closed -> stamped MAT_DOOR_CLOSED; toggled open -> unsealed).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fire_studio_level.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from level_loader import load, materials_from_tilemap  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.materials import MAT_DOOR_CLOSED, MAT_HULL, MAT_WOOD  # noqa: E402
from simulation.unit import Unit  # noqa: E402

LEVEL_DIR = ROOT / "levels" / "fire_studio"
GEN_PATH = ROOT / "tools" / "gen_fire_studio.py"
_OUTPUTS = ("tilemap.csv", "water_init.npy", "diffuse.png", "level.toml")


def _load_generator():
    spec = importlib.util.spec_from_file_location("_gen_fire_studio", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_generator_has_no_rng():
    src = GEN_PATH.read_text(encoding="utf-8")
    assert "import random" not in src, "generator must not import random"
    assert "np.random" not in src, "generator must not use np.random"
    assert "random." not in src, "generator must be RNG-free (deterministic)"


def test_generator_is_byte_deterministic():
    """Re-running the generator reproduces every output file byte-for-byte
    (and confirms the committed folder already matches the tool)."""
    before = {name: (LEVEL_DIR / name).read_bytes() for name in _OUTPUTS}
    _load_generator().main()          # regenerate in place
    after = {name: (LEVEL_DIR / name).read_bytes() for name in _OUTPUTS}
    for name in _OUTPUTS:
        assert before[name] == after[name], (
            f"{name} changed on re-run — generator is not deterministic")


# ---------------------------------------------------------------------------
# Headless load
# ---------------------------------------------------------------------------
def test_studio_loads_headless():
    lvl = load("fire_studio")
    assert lvl.version == "2"
    assert lvl.tilemap.shape == (32, 48), f"shape={lvl.tilemap.shape}"
    assert lvl.diffuse_path.exists()
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert (mat == MAT_HULL).any(), "the hull-sealed box"
    assert (mat == MAT_WOOD).any(), "the crate/furniture fuel"
    assert vac.any(), "the SPACE ring (ship in vacuum)"


def test_studio_has_lamps_beacon_door_water_spawns():
    lvl = load("fire_studio")
    statics = [l for l in lvl.lights if l.kind == "static"]
    beacons = [l for l in lvl.lights if l.kind == "beacon"]
    assert len(statics) == 3, "two hall lamps + one corridor lamp"
    assert len(beacons) == 1, "exactly one rotating beacon (Erik's addition)"
    doors = [e for e in lvl.entities if e.class_name == "door"]
    assert len(doors) == 1 and doors[0].id == "side_door"
    assert lvl.water_depth_q is not None and int((lvl.water_depth_q > 0).sum()) > 0
    marines = [s for s in lvl.spawns if s.team == 0]
    assert len(marines) == 3, "three marine spawns in the hall"


def test_studio_spawns_stand_on_open_floor():
    lvl = load("fire_studio")
    _, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    for s in lvl.spawns:
        for dy in range(s.footprint):
            for dx in range(s.footprint):
                tx, ty = int(s.x) + dx, int(s.y) + dy
                code = int(lvl.tilemap[ty, tx])
                assert code == 0 and not vac[ty, tx], (
                    f"spawn {s.name} footprint tile ({tx},{ty}) is not open "
                    f"interior floor (code {code})")


# ---------------------------------------------------------------------------
# Headless construct + run + door reconcile
# ---------------------------------------------------------------------------
def _fresh_sim():
    lvl = load("fire_studio")
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False)
    for s in lvl.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    sim.set_paused(False)
    return sim


def test_studio_constructs_runs_and_door_seals():
    sim = _fresh_sim()
    for _ in range(20):
        sim.step()
    assert sim.tick == 20
    assert len(sim.marines()) == 3 and all(u.alive for u in sim.units)

    # The sealed side-room door: one entity, stamped closed at load.
    assert len(sim._doors) == 1
    door = sim.door_at(5, 37)                 # door_at takes (fy, fx)
    assert door is not None and door.id == "side_door"
    assert int(sim.gmap.material[5, 37]) == MAT_DOOR_CLOSED
    assert int(sim.gmap.material[6, 37]) == MAT_DOOR_CLOSED

    # Toggle open (the demo's C key) -> the 9e sweep unseals it.
    door.want_open = True
    for _ in range(3):
        sim.step()
    assert int(sim.gmap.material[5, 37]) == 0, "opened door tile is air"


if __name__ == "__main__":
    test_generator_has_no_rng()
    test_generator_is_byte_deterministic()
    test_studio_loads_headless()
    test_studio_has_lamps_beacon_door_water_spawns()
    test_studio_spawns_stand_on_open_floor()
    test_studio_constructs_runs_and_door_seals()
    print("OK — fire_studio: deterministic gen / headless load / door reconcile")
