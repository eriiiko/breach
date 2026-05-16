"""Tests for level_loader.

Run with:
    C:/Users/steen/anaconda3/python.exe tests/test_level_loader.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from level_loader import load, materials_from_tilemap


def assert_eq(actual, expected, name=""):
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def test_load_unhcr_vessel():
    lvl = load("unhcr_vessel")
    assert_eq(lvl.name, "UNHCR Vessel", "name")
    assert_eq(lvl.version, "1", "version")
    assert lvl.tilemap.shape == (120, 50), f"shape={lvl.tilemap.shape}"
    assert lvl.diffuse_path.exists()
    assert lvl.normal_path is not None and lvl.normal_path.exists()
    print("OK: load_unhcr_vessel")


def test_materials_from_tilemap():
    lvl = load("unhcr_vessel")
    mat, vac = materials_from_tilemap(lvl.tilemap)
    # CSV: 0=vacuum, 1=hull, 3=door, else=air
    assert mat.shape == lvl.tilemap.shape
    assert vac.shape == lvl.tilemap.shape
    n_hull   = int((mat == 1).sum())
    n_door   = int((mat == 3).sum())
    n_vacuum = int(vac.sum())
    n_air    = int((mat == 0).sum())
    assert n_hull > 0, "expected some hull walls"
    assert n_door > 0, "expected some doors"
    assert n_vacuum > 0, "expected vacuum cells (outer space)"
    assert n_air > 0, "expected air cells (interior)"
    print(f"OK: materials hull={n_hull} door={n_door} air={n_air} vacuum={n_vacuum}")


def test_csv_values_match_expected():
    """CSV must contain only known tile values [0..8]."""
    lvl = load("unhcr_vessel")
    unique = sorted(np.unique(lvl.tilemap).tolist())
    for v in unique:
        assert 0 <= v <= 8, f"unexpected tile value {v}"
    print(f"OK: csv_values_in_range: {unique}")


def test_missing_level_raises():
    try:
        load("nonexistent_level_xyz")
    except ValueError as e:
        print(f"OK: missing_level raises ValueError ({e})")
        return
    raise AssertionError("expected ValueError for missing level")


if __name__ == "__main__":
    test_load_unhcr_vessel()
    test_materials_from_tilemap()
    test_csv_values_match_expected()
    test_missing_level_raises()
    print("\nAll level_loader tests passed.")
