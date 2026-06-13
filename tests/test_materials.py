"""Tests for the material-property table + table-driven gamemap caches.

Covers ch.02 (Material System) foundation:
  - the table loads from config and exposes every column (scalars + RGB)
  - the unified MAT_* ids are shared (no duplication drift)
  - derived caches (solid/flammable/wall_hp/conductivity) match the table
  - on_tile_changed patches ALL caches after a destroy_wall, with no O(grid)
    rebuild and identical observable behaviour

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_materials.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from config import CFG
from level_loader import load as load_level
from simulation.gamemap import GameMap
from simulation.materials import (
    MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_STEEL, MAT_GLASS,
    MAT_FURNITURE, MATERIAL_NAMES, MaterialTable,
)


SCALAR_COLUMNS = (
    "hp", "flammable", "mobility", "conductivity",
    "ignition_temp", "heat_atten", "wave_reflect", "wave_absorb",
    "blast_resist",
)


def test_ids_unified():
    """level_loader and gamemap must share ONE set of ids (no duplication)."""
    import level_loader
    import simulation.gamemap as gm
    for name, expected in (
        ("MAT_AIR", 0), ("MAT_HULL", 1), ("MAT_WOOD", 2),
        ("MAT_DOOR", 3), ("MAT_STEEL", 4), ("MAT_GLASS", 5),
        ("MAT_FURNITURE", 6),
    ):
        assert getattr(gm, name) == expected, f"gamemap.{name}"
    # gamemap re-exports the canonical ids from simulation.materials.
    from simulation import materials as mats
    assert gm.MAT_HULL is mats.MAT_HULL
    print("OK: ids_unified")


def test_table_loads_all_columns():
    tbl = MaterialTable.from_config(CFG)
    assert tbl.n == len(MATERIAL_NAMES), "row count mismatch"
    for col in SCALAR_COLUMNS:
        arr = getattr(tbl, col)
        assert arr.shape == (tbl.n,), f"{col} shape {arr.shape}"
    # light_atten is per-channel RGB.
    assert tbl.light_atten.shape == (tbl.n, 3), "light_atten not (N,3)"
    # Spot-check known illustrative values from ch.02 / config.toml.
    assert tbl.hp[MAT_HULL] == 300
    assert tbl.hp[MAT_WOOD] == 60
    assert tbl.hp[MAT_GLASS] == 15
    assert bool(tbl.flammable[MAT_WOOD]) is True
    assert bool(tbl.flammable[MAT_HULL]) is False
    # mobility (fixed-point milli-units) replaces the old passable bool:
    # air/door normal-speed (1000), hull a wall (0).
    assert int(tbl.mobility[MAT_AIR]) == 1000
    assert int(tbl.mobility[MAT_DOOR]) == 1000
    assert int(tbl.mobility[MAT_HULL]) == 0
    assert tbl.conductivity[MAT_HULL] == 50.0
    # air fully transparent, hull/door fully opaque, glass partial.
    assert np.all(tbl.light_atten[MAT_AIR] == 0.0)
    assert np.all(tbl.light_atten[MAT_HULL] == 1.0)
    assert np.all(tbl.light_atten[MAT_DOOR] == 1.0)
    assert np.all(tbl.light_atten[MAT_GLASS] < 1.0)
    print("OK: table_loads_all_columns")


def test_table_missing_material_raises():
    bad = {name: {} for name in MATERIAL_NAMES.values()}
    del bad["glass"]
    try:
        MaterialTable(bad)
    except KeyError:
        print("OK: table_missing_material_raises")
        return
    raise AssertionError("expected KeyError for missing material row")


def test_caches_match_table():
    """Derived caches must equal the table projection (no hardcoded lists)."""
    g = GameMap(load_level("unhcr_vessel"))
    m = g.material
    tbl = g.materials
    # Solid mask == old {HULL, WOOD, DOOR} behaviour for current set.
    expected_wall = np.isin(m, [MAT_HULL, MAT_WOOD, MAT_DOOR])
    assert np.array_equal(g.solid, expected_wall), "solid regressed"
    assert np.array_equal(g.flammable, (m == MAT_WOOD)), "flammable regressed"
    assert np.array_equal(g.flammable, tbl.flammable[m]), "flammable != table"
    assert np.array_equal(g.wall_hp, tbl.hp[m]), "wall_hp != table"
    assert np.array_equal(g.conductivity, tbl.conductivity[m]), "conductivity != table"
    # conductivity allocated + populated (metal hull spreads heat).
    assert g.conductivity.shape == m.shape
    if (m == MAT_HULL).any():
        assert g.conductivity[m == MAT_HULL].max() == 50.0
    print("OK: caches_match_table")


def test_on_tile_changed_patches_all_caches_after_destroy():
    g = GameMap(load_level("unhcr_vessel"))
    # Use an interior (non-edge) hull tile so destroy_wall does NOT open a
    # vacuum breach — we are testing the cache patch, not breach behaviour.
    h, w = g.material.shape
    interior = np.zeros_like(g.solid)
    interior[2:h - 2, 2:w - 2] = True
    ys, xs = np.where((g.material == MAT_HULL) & interior)
    assert len(ys) > 0, "level has no interior hull to destroy"
    y, x = int(ys[0]), int(xs[0])

    # Pre-conditions: hull is solid, has hp + conductivity.
    assert g.solid[y, x]
    assert g.wall_hp[y, x] == g.materials.hp[MAT_HULL]
    assert g.conductivity[y, x] == g.materials.conductivity[MAT_HULL]

    # Snapshot the rest of the grid to prove no O(grid) rebuild happened.
    wall_before = g.solid.copy()
    cond_before = g.conductivity.copy()

    g.destroy_wall(y, x)

    # The one tile is fully patched to AIR semantics.
    assert g.material[y, x] == MAT_AIR
    assert not g.solid[y, x]
    assert not g.flammable[y, x]
    assert g.wall_hp[y, x] == 0
    assert g.conductivity[y, x] == 0.0

    # Every OTHER tile is untouched (incremental patch, not a rebuild).
    wall_before[y, x] = False
    cond_before[y, x] = 0.0
    assert np.array_equal(g.solid, wall_before), "solid touched other tiles"
    assert np.array_equal(g.conductivity, cond_before), "conductivity touched others"
    print("OK: on_tile_changed_patches_all_caches_after_destroy")


def test_permeability_defaults_to_solid_set():
    """permeability is sealed (0) exactly where a tile occludes today and open
    (1) elsewhere, so the physics `obstacles` boundary == the legacy solid
    set — behaviour-preserving while flow now sources from permeability."""
    tbl = MaterialTable.from_config(CFG)
    assert tbl.permeability[MAT_AIR] == 1.0, "air must be open"
    for mid in (MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_STEEL, MAT_GLASS):
        assert tbl.permeability[mid] == 0.0, f"material {mid} must be sealed"
    g = GameMap(load_level("unhcr_vessel"))
    # obstacles base (before any unit stamp) derives from permeability and must
    # equal the occlusion set for the current materials.
    assert np.array_equal(g.obstacles, g.solid), "obstacles != solid set"
    assert np.array_equal((g.permeability <= 0.0), g.solid)
    print("OK: permeability_defaults_to_solid_set")


def test_furniture_row_values():
    """FURNITURE (id 6, editor proposal Q2): dedicated material row — lower
    hp, flammable, PARTIAL permeability (smoke drifts past crates), partial
    light occlusion, climbable-at-a-penalty movement (mobility 400, not a
    wall)."""
    assert MAT_FURNITURE == 6
    assert MATERIAL_NAMES[MAT_FURNITURE] == "furniture"
    tbl = MaterialTable.from_config(CFG)
    assert tbl.hp[MAT_FURNITURE] == 30
    assert bool(tbl.flammable[MAT_FURNITURE]) is True
    assert tbl.ignition_temp[MAT_FURNITURE] == 280.0
    # Furniture is now climbable-at-a-penalty, not a wall: mobility 400
    # (40% speed = 2.5x step time), the boolean it replaced was passable=false.
    assert int(tbl.mobility[MAT_FURNITURE]) == 400
    # The approved partial permeability — explicitly 0.5, NOT the derived
    # sealed-because-it-occludes default.
    assert tbl.permeability[MAT_FURNITURE] == np.float32(0.5)
    assert np.all(tbl.light_atten[MAT_FURNITURE] == np.float32(0.55))
    assert tbl.heat_atten[MAT_FURNITURE] == np.float32(0.5)
    assert tbl.conductivity[MAT_FURNITURE] == 0.0
    assert tbl.heat_inv_shift[MAT_FURNITURE] == 3       # thermal_mass 8 = 2**3
    assert tbl.wave_reflect[MAT_FURNITURE] == np.float32(0.2)
    assert tbl.wave_absorb[MAT_FURNITURE] == np.float32(0.5)
    assert tbl.blast_resist[MAT_FURNITURE] == 0.0
    assert tbl.burst_threshold[MAT_FURNITURE] == np.float32(2.0)
    # Ignition threshold lands in the shared Q16.16 temperature domain.
    assert tbl.ignition_temp_q16[MAT_FURNITURE] == round(280.0 * 65536)
    # kappa == 0 -> structural no-conduction face, like air (engine/06 §2.6).
    assert tbl.self_shift[MAT_FURNITURE] == tbl.no_face
    assert np.all(tbl.face_shift_table[MAT_FURNITURE, :] == tbl.no_face)
    print("OK: furniture_row_values")


def test_furniture_projects_into_grid_caches():
    """A tile patched to FURNITURE lands in every derived grid: flammable
    mask True, permeability 0.5 (NOT solid — flow drifts past), wall_hp 30,
    movement climbable-at-a-penalty (mobility 400 -> is_passable True)."""
    g = GameMap(load_level("unhcr_vessel"))
    ys, xs = np.where((g.material == MAT_AIR) & ~g.is_vacuum)
    y, x = int(ys[len(ys) // 2]), int(xs[len(ys) // 2])
    g.material[y, x] = MAT_FURNITURE
    g.on_tile_changed(y, x)
    assert bool(g.flammable[y, x]) is True
    assert g.permeability[y, x] == np.float32(0.5)
    assert not g.solid[y, x], "partial permeability must NOT make it solid"
    assert g.wall_hp[y, x] == 30
    assert g.conductivity[y, x] == 0.0
    # Movement: furniture is now ENTERABLE (mobility 400 > 0) — climbable at a
    # penalty, no longer a wall. The is_passable view is mobility > 0.
    assert g.is_passable(y, x)
    # And the projection equals the table everywhere (no hardcoded list).
    assert np.array_equal(g.flammable, g.materials.flammable[g.material])
    assert np.array_equal(g.permeability, g.materials.permeability[g.material])
    print("OK: furniture_projects_into_grid_caches")


def test_on_tile_changed_direct_patch():
    """Directly editing material + calling on_tile_changed updates caches."""
    g = GameMap(load_level("unhcr_vessel"))
    ys, xs = np.where(g.material == MAT_AIR)
    y, x = int(ys[0]), int(xs[0])
    assert not g.solid[y, x]
    # Promote an air tile to steel and patch its caches.
    g.material[y, x] = MAT_STEEL
    g.on_tile_changed(y, x)
    assert g.solid[y, x], "steel must be solid"
    assert g.wall_hp[y, x] == g.materials.hp[MAT_STEEL]
    assert g.conductivity[y, x] == g.materials.conductivity[MAT_STEEL]
    assert not g.flammable[y, x]
    print("OK: on_tile_changed_direct_patch")


if __name__ == "__main__":
    test_ids_unified()
    test_table_loads_all_columns()
    test_table_missing_material_raises()
    test_caches_match_table()
    test_on_tile_changed_patches_all_caches_after_destroy()
    test_permeability_defaults_to_solid_set()
    test_furniture_row_values()
    test_furniture_projects_into_grid_caches()
    test_on_tile_changed_direct_patch()
    print("\nAll material tests passed.")
