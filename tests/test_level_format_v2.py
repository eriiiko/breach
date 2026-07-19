"""Level format v2 — canon CSV codes + migration (proposal §1.1, step F1).

Pins the two CSV vocabularies side by side:

  v1 (generator): 0=space, 1=hull, 2=wood wall, 3=door, 4..8=interior air —
     the EXACT legacy mapping, frozen (old levels must decode bit-identically);
  v2 (canon): codes ARE material ids (simulation.materials) + the single
     reserved non-material code SPACE_CODE 9 = MAT_AIR + is_vacuum; anything
     else is a hard ValueError (no silent garbage).

Plus the v1->v2 migrator (tools/migrate_tilemap_v2.py) as an importable
function: code translation (incl. the code-2 landmine: generator *floor* ->
air, NOT wood), sim-equivalence of a migrated grid, file round-trip with .bak
safety + comment-preserving version bump, double-migration refusal — and the
shipped (already-migrated) unhcr_vessel still loading v2 and being airtight
through the real GameMap flood-fill.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_level_format_v2.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT / "tools"))

from level_loader import (  # noqa: E402
    SPACE_CODE, SUPPORTED_VERSIONS, load as load_level, materials_from_tilemap,
)
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_STEEL, MAT_GLASS, MAT_FURNITURE,
    MATERIAL_NAMES,
)
from migrate_tilemap_v2 import V1_TO_V2, migrate_grid, migrate_level  # noqa: E402


# ---------------------------------------------------------------------------
# Loader: version gate + the two vocabularies
# ---------------------------------------------------------------------------

def test_version_gate_and_space_code():
    assert SUPPORTED_VERSIONS == {"1", "2"}
    assert SPACE_CODE == 9                      # Erik, 2026-06-10
    assert SPACE_CODE not in MATERIAL_NAMES     # reserved NON-material code


def test_v1_mapping_unchanged():
    """The legacy generator vocabulary decodes exactly as before F1."""
    tm = np.array([[0, 1, 2, 3],
                   [4, 5, 7, 8]], dtype=np.int32)
    mat, vac = materials_from_tilemap(tm, "1")
    expected = np.array([[MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR],
                         [MAT_AIR, MAT_AIR, MAT_AIR, MAT_AIR]])
    np.testing.assert_array_equal(mat, expected)
    np.testing.assert_array_equal(
        vac, np.array([[True, False, False, False],
                       [False, False, False, False]]))
    assert mat.dtype == np.int8


def test_v2_literal_mapping_and_space():
    """v2 codes are material ids read literally; 9 = SPACE = air + vacuum."""
    tm = np.array([[0, 1, 2],
                   [3, 4, 5],
                   [9, 9, 0]], dtype=np.int32)
    mat, vac = materials_from_tilemap(tm, "2")
    expected = np.array([[MAT_AIR, MAT_HULL, MAT_WOOD],
                         [MAT_DOOR, MAT_STEEL, MAT_GLASS],
                         [MAT_AIR, MAT_AIR, MAT_AIR]])
    np.testing.assert_array_equal(mat, expected)
    np.testing.assert_array_equal(
        vac, np.array([[False, False, False],
                       [False, False, False],
                       [True, True, False]]))
    assert mat.dtype == np.int8
    # v2 finally gives steel + glass a CSV code (v1 had none).
    assert int((mat == MAT_STEEL).sum()) == 1
    assert int((mat == MAT_GLASS).sum()) == 1


def test_v2_accepts_furniture():
    """Code 6 = MAT_FURNITURE (dedicated material row, proposal Q2): the v2
    vocabulary legalised it the moment the material-table row landed."""
    tm = np.array([[0, 6],
                   [6, 9]], dtype=np.int32)
    mat, vac = materials_from_tilemap(tm, "2")
    np.testing.assert_array_equal(
        mat, np.array([[MAT_AIR, MAT_FURNITURE],
                       [MAT_FURNITURE, MAT_AIR]]))
    np.testing.assert_array_equal(
        vac, np.array([[False, False], [False, True]]))


@pytest.mark.parametrize("bad", [8, 42, -1])
def test_v2_unknown_code_raises(bad):
    """v2 tolerates ONLY material-table ids + SPACE_CODE — fail loud.
    (6 left this list when FURNITURE became a real material row; 7 left
    when DOOR_CLOSED became one — A6 doors v0.)"""
    tm = np.array([[0, bad]], dtype=np.int32)
    with pytest.raises(ValueError, match="unknown codes"):
        materials_from_tilemap(tm, "2")


def test_unknown_version_raises():
    tm = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="unsupported tilemap version"):
        materials_from_tilemap(tm, "3")


# ---------------------------------------------------------------------------
# Migrator: pure grid translation
# ---------------------------------------------------------------------------

def test_migrate_grid_mapping():
    """0->9, 1->1, 3->3, 2/4..8->0 — and the input grid is left untouched."""
    v1 = np.array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]], dtype=np.int32)
    keep = v1.copy()
    v2 = migrate_grid(v1)
    expected = np.array([[9, 1, 0],
                         [3, 0, 0],
                         [0, 0, 0]])
    np.testing.assert_array_equal(v2, expected)
    np.testing.assert_array_equal(v1, keep)     # pure function, no aliasing
    assert v2.size == v1.size
    # The landmine dies here: generator code 2 was *floor*, not MAT_WOOD.
    assert V1_TO_V2[2] == MAT_AIR


def test_migrated_grid_is_sim_equivalent():
    """A migrated grid decodes (v2) to the same materials + vacuum the v1
    grid decoded to (v1) — for the codes shipped levels actually use.

    Code 2 is deliberately EXCLUDED: v1 decoded it as MAT_WOOD (wall) while
    the migration retires it as generator floor -> air (the proposal's
    landmine decision). Neither vessel contains a 2, so shipped levels are
    bit-identical through the sim before and after migration.
    """
    rng = np.random.default_rng(7)
    v1 = rng.choice([0, 1, 3, 4, 5, 6, 7, 8], size=(20, 15)).astype(np.int32)
    mat1, vac1 = materials_from_tilemap(v1, "1")
    mat2, vac2 = materials_from_tilemap(migrate_grid(v1), "2")
    np.testing.assert_array_equal(mat1, mat2)
    np.testing.assert_array_equal(vac1, vac2)


def test_migrate_grid_rejects_non_v1_codes():
    """An already-migrated grid (contains 9) must be refused — running the
    migration twice would turn interior air into outer space."""
    with pytest.raises(ValueError, match="non-v1 codes"):
        migrate_grid(np.array([[9, 0], [1, 3]], dtype=np.int32))
    with pytest.raises(ValueError, match="non-v1 codes"):
        migrate_grid(np.array([[0, 17]], dtype=np.int32))


# ---------------------------------------------------------------------------
# Migrator: file round-trip on a synthetic level folder
# ---------------------------------------------------------------------------

_TOML_V1 = (
    "# synthetic level — comments must survive the migration byte-for-byte\n"
    'version = "1"   # format version\n'
    'name = "Tiny"\n'
    'tilemap = "tilemap.csv"\n'
    "tile_size_m = 0.5\n"
    "# trailing comment\n"
)


def test_migrate_level_round_trip(tmp_path):
    csv_v1 = b"0,1,2\r\n3,4,8\r\n"            # CRLF, like the shipped levels
    (tmp_path / "level.toml").write_text(_TOML_V1, encoding="utf-8")
    (tmp_path / "tilemap.csv").write_bytes(csv_v1)

    before, after = migrate_level(tmp_path)
    assert before == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 8: 1}
    assert after == {0: 3, 1: 1, 3: 1, 9: 1}    # 2, 4, 8 all retire to air

    # CSV rewritten in canon codes, CRLF + trailing newline preserved.
    assert (tmp_path / "tilemap.csv").read_bytes() == b"9,1,0\r\n3,0,0\r\n"
    # Version bumped line-targeted; every other byte (comments incl. the one
    # trailing the version itself) untouched.
    assert (tmp_path / "level.toml").read_text(encoding="utf-8") == \
        _TOML_V1.replace('version = "1"', 'version = "2"')
    # .bak safety net == the exact pre-migration bytes.
    assert (tmp_path / "tilemap.csv.bak").read_bytes() == csv_v1
    assert (tmp_path / "level.toml.bak").read_text(encoding="utf-8") == _TOML_V1
    # And the migrated folder decodes through the v2 loader path.
    grid = np.loadtxt(tmp_path / "tilemap.csv", delimiter=",", dtype=np.int32)
    mat, vac = materials_from_tilemap(grid, "2")
    assert bool(vac[0, 0]) and int(vac.sum()) == 1

    # Re-running refuses: the level is no longer version "1".
    with pytest.raises(ValueError, match="nothing to migrate"):
        migrate_level(tmp_path)


# ---------------------------------------------------------------------------
# The real, shipped level: v2 + airtight through the real GameMap
# ---------------------------------------------------------------------------

def test_real_level_is_v2_and_airtight():
    lvl = load_level("unhcr_vessel")
    assert lvl.version == "2"
    codes = set(int(c) for c in np.unique(lvl.tilemap))
    assert codes <= set(MATERIAL_NAMES) | {SPACE_CODE}, f"non-canon codes: {codes}"

    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert int((mat == MAT_HULL).sum()) > 0
    assert int((mat == MAT_DOOR).sum()) > 0
    assert int(vac.sum()) > 0                   # space outside the hull

    # The canonical airtight certification: GameMap decode + flood fill.
    from level_airtight import check
    assert check("unhcr_vessel") is True
