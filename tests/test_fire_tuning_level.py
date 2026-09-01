"""The fire-tuning level (fire session #12 phase 2 part B,
docs/fire_phase2_hud_and_level_2026-09-01.md SS B) — generator determinism +
headless load/run + the sealed-chamber airtight gate.

Deliberately NOT part of any golden/digest machinery (a new tuning level,
never migrated): a plain gate that

  - the generator is DETERMINISTIC (no RNG in the source; re-running yields
    byte-identical tilemap.csv / diffuse.png / level.toml — so the committed
    folder always matches the tool);
  - the level LOADS headless (v2 codes, the 5 stations + door + spawns);
  - the whole level passes the airtight lint (tools/level_airtight.py) —
    the hull-sealed ring boundary has no vacuum/edge leak;
  - the sealed chamber (station 4) really has NO opening at all;
  - a headless sim constructs + runs 100 ticks without error, and the door
    room's door (station 5) reconciles open/closed.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_fire_tuning_level.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from level_loader import load, materials_from_tilemap  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.materials import (MAT_DOOR_CLOSED, MAT_FURNITURE,  # noqa: E402
                                  MAT_HULL, MAT_KINDLING, MAT_WOOD)
from simulation.unit import Unit  # noqa: E402
from level_airtight import check as airtight_check  # noqa: E402

LEVEL_DIR = ROOT / "levels" / "fire_tuning"
GEN_PATH = ROOT / "tools" / "make_fire_tuning_level.py"
_OUTPUTS = ("tilemap.csv", "level.toml", "diffuse.png")


def _load_generator():
    spec = importlib.util.spec_from_file_location("_gen_fire_tuning", GEN_PATH)
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


def test_generator_is_byte_deterministic(tmp_path):
    """Two independent generator runs are byte-identical, and the committed
    folder matches the tool — WITHOUT writing into the working tree (the
    fire_studio test's own rewritten form, issue #47)."""
    gen = _load_generator()
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    gen.main(out_dir=run_a)
    gen.main(out_dir=run_b)
    for name in _OUTPUTS:
        assert (run_a / name).read_bytes() == (run_b / name).read_bytes(), (
            f"{name} differs between two runs — generator is not deterministic")
        assert (run_a / name).read_bytes() == (LEVEL_DIR / name).read_bytes(), (
            f"committed {name} does not match the tool's output")


# ---------------------------------------------------------------------------
# Headless load
# ---------------------------------------------------------------------------
def test_tuning_loads_headless():
    lvl = load("fire_tuning")
    assert lvl.version == "2"
    assert lvl.tilemap.shape == (46, 72), f"shape={lvl.tilemap.shape}"
    assert lvl.diffuse_path.exists()
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert (mat == MAT_HULL).any(), "the hull-sealed shell + station boxes"
    assert (mat == MAT_WOOD).any(), "wood fuel (stations 1/2/3/4/5)"
    assert (mat == MAT_FURNITURE).any(), "furniture sample (station 3)"
    assert (mat == MAT_KINDLING).any(), "kindling (stations 2/3)"
    assert (mat == MAT_DOOR_CLOSED).any(), "station 5's closed door"
    assert vac.any(), "the SPACE ring (item 6)"


def test_tuning_has_door_and_spawns():
    lvl = load("fire_tuning")
    doors = [e for e in lvl.entities if e.class_name == "door"]
    assert len(doors) == 1 and doors[0].id == "station5_door"
    marines = [s for s in lvl.spawns if s.team == 0]
    assert 1 <= len(marines) <= 2, "a small marine roster (design SS B item 6)"


def test_tuning_spawns_stand_on_open_floor():
    lvl = load("fire_tuning")
    _, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    for s in lvl.spawns:
        for dy in range(s.footprint):
            for dx in range(s.footprint):
                tx, ty = int(s.x) + dx, int(s.y) + dy
                code = int(lvl.tilemap[ty, tx])
                assert code == 0 and not vac[ty, tx], (
                    f"spawn {s.name} footprint tile ({tx},{ty}) is not open "
                    f"interior floor (code {code})")


def test_tuning_stations_are_isolated():
    """Every station cluster sits >=6 tiles from the next non-hull-boxed
    cluster's nearest fuel tile — G-note: ~2-tile radiation reach can't
    couple two stations (design SS B preamble)."""
    lvl = load("fire_tuning")
    tm = lvl.tilemap
    fuel_codes = {MAT_WOOD, MAT_FURNITURE, MAT_KINDLING}
    ys, xs = np.nonzero(np.isin(tm, list(fuel_codes)))
    pts = list(zip(xs.tolist(), ys.tolist()))
    # Cluster by connected (8-neighbour) fuel tiles — a station's own
    # internal spread line/crate is ONE cluster; different stations must
    # never be 8-adjacent to each other.
    pts_set = set(pts)
    seen = set()
    clusters = []
    for p in pts:
        if p in seen:
            continue
        stack, cluster = [p], []
        seen.add(p)
        while stack:
            cx, cy = stack.pop()
            cluster.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    q = (cx + dx, cy + dy)
                    if q in pts_set and q not in seen:
                        seen.add(q)
                        stack.append(q)
        clusters.append(cluster)
    # 7 distinct fuel clusters: bonfire (2x2), spread-line (crate+kindling
    # line, ONE connected run), 3 isolated material-row samples, and the
    # sealed-chamber + door-room crates (stations 4/5).
    assert len(clusters) == 7, f"expected 7 fuel clusters, got {len(clusters)}"
    # Pairwise minimum Chebyshev distance between clusters >= 6 tiles.
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            d = min(max(abs(ax - bx), abs(ay - by))
                   for (ax, ay) in clusters[i] for (bx, by) in clusters[j])
            assert d >= 6, f"clusters {i} and {j} are only {d} tiles apart"


def test_tuning_sealed_chamber_has_no_opening():
    """Station 4's box perimeter is pure MAT_HULL — no door, no gap."""
    lvl = load("fire_tuning")
    tm = lvl.tilemap
    from make_fire_tuning_level import S4_X0, S4_Y0, S4_X1, S4_Y1  # noqa: E402
    perim = np.concatenate([
        tm[S4_Y0, S4_X0:S4_X1 + 1], tm[S4_Y1, S4_X0:S4_X1 + 1],
        tm[S4_Y0:S4_Y1 + 1, S4_X0], tm[S4_Y0:S4_Y1 + 1, S4_X1],
    ])
    assert (perim == MAT_HULL).all(), "sealed chamber perimeter has a gap"


def test_tuning_is_airtight():
    assert airtight_check("fire_tuning"), "fire_tuning has a hull leak"


# ---------------------------------------------------------------------------
# Headless construct + run + door reconcile
# ---------------------------------------------------------------------------
def _fresh_sim():
    lvl = load("fire_tuning")
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False)
    for s in lvl.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    sim.set_paused(False)
    return sim


def test_tuning_constructs_and_runs_100_ticks():
    sim = _fresh_sim()
    for _ in range(100):
        sim.step()
    assert sim.tick == 100
    assert len(sim.units) >= 1 and all(u.alive for u in sim.units)


def test_tuning_door_room_door_reconciles():
    sim = _fresh_sim()
    for _ in range(5):
        sim.step()
    assert len(sim._doors) == 1
    door = sim.door_at(28, 27)                # door_at takes (fy, fx)
    assert door is not None and door.id == "station5_door"
    assert int(sim.gmap.material[28, 27]) == MAT_DOOR_CLOSED
    assert int(sim.gmap.material[28, 28]) == MAT_DOOR_CLOSED

    door.want_open = True
    for _ in range(3):
        sim.step()
    assert int(sim.gmap.material[28, 27]) == 0, "opened door tile is air"


if __name__ == "__main__":
    test_generator_has_no_rng()
    test_tuning_loads_headless()
    test_tuning_has_door_and_spawns()
    test_tuning_spawns_stand_on_open_floor()
    test_tuning_stations_are_isolated()
    test_tuning_sealed_chamber_has_no_opening()
    test_tuning_is_airtight()
    test_tuning_constructs_and_runs_100_ticks()
    test_tuning_door_room_door_reconciles()
    print("OK — fire_tuning: deterministic gen / headless load / airtight / "
          "100-tick run / door reconcile")
