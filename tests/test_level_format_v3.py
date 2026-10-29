"""Level format v3 — SPACE outside the material ids (#27, Erik 2026-10-29).

Erik's ruling: "rows suffice; design for 127 materials". v2 reserved CSV code
9 for SPACE, inside the id range (== MAT_FOLIAGE), which capped the material
table at ten rows. v3 moves SPACE to ``level_loader.SPACE_CODE`` (-1) and lets
material ids run as rows up to the int8 material plane's 127. Each test below
names the property it protects and the change that would break it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_level_format_v3.py -q
"""
from __future__ import annotations

import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT / "tools"))

import level_lib  # noqa: E402
from level_loader import (  # noqa: E402
    LEVEL_FORMAT_VERSION, MATERIAL_ID_MAX, SPACE_CODE, V2_SPACE_CODE,
    load as load_level, materials_from_tilemap,
)
from simulation import materials as materials_mod  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_FOLIAGE, MAT_HULL, MATERIAL_NAMES,
)
from migrate_tilemap_v3 import migrate_level  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: a minimal loadable level folder, written through level_lib
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


def _level(tmp_path: Path, grid: np.ndarray, name: str = "v3") -> Path:
    """A level folder whose header and CSV come from THE writer (level_lib),
    so the fixture is exactly what every generator writes."""
    d = tmp_path / name
    d.mkdir()
    g = np.asarray(grid, dtype=np.int32)
    level_lib.write_level_header(d, name=name, tile_size_m=0.333)
    level_lib.write_tilemap_csv(d, g, csv_bak=False)
    _write_png(d / "diffuse.png", w=g.shape[1], h=g.shape[0])
    return d


def _ring_grid(h: int = 6, w: int = 8, interior: int = MAT_AIR) -> np.ndarray:
    """SPACE border, hull ring, ``interior`` inside."""
    g = np.full((h, w), interior, dtype=np.int32)
    g[0, :] = g[-1, :] = SPACE_CODE
    g[:, 0] = g[:, -1] = SPACE_CODE
    g[1, 1:-1] = g[-2, 1:-1] = MAT_HULL
    g[1:-1, 1] = g[1:-1, -2] = MAT_HULL
    return g


# ---------------------------------------------------------------------------
# (a) SPACE_CODE loads as vacuum + air
# ---------------------------------------------------------------------------

def test_space_code_loads_as_vacuum_air(tmp_path):
    """Property: a v3 CSV's SPACE_CODE tile decodes to is_vacuum + MAT_AIR,
    and -1 never reaches the int8 material plane. Breaks if the loader stops
    intercepting SPACE_CODE (the plane would hold -1, indexing every
    material column from the END) or if the writer stops stamping v3."""
    g = _ring_grid()
    lvl = load_level("v3", levels_dir=str(_level(tmp_path, g).parent))
    assert lvl.version == LEVEL_FORMAT_VERSION
    space = (g == SPACE_CODE)
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert mat.dtype == np.int8
    np.testing.assert_array_equal(vac, space)
    assert (mat[space] == MAT_AIR).all()
    assert int(mat.min()) >= 0                 # no -1 in the material plane


def test_space_code_is_outside_the_material_ids():
    """Property: SPACE's code is not any material id a row could take (0 ..
    MATERIAL_ID_MAX). Breaks if SPACE_CODE moves back into the id range,
    which re-creates the v2 clash with whichever row owns that id."""
    assert not 0 <= SPACE_CODE <= MATERIAL_ID_MAX
    assert SPACE_CODE not in MATERIAL_NAMES


# ---------------------------------------------------------------------------
# (b) 9 is a material again: the old clash is gone
# ---------------------------------------------------------------------------

def test_nine_loads_as_foliage_not_space(tmp_path):
    """Property: in v3 the value 9 is material 9 (foliage) on a NON-vacuum
    tile — the v2 clash between SPACE and MAT_FOLIAGE is gone. Breaks if the
    v3 decode still treats V2_SPACE_CODE as space."""
    assert MAT_FOLIAGE == V2_SPACE_CODE      # the clash this format removes
    g = _ring_grid()
    g[2, 3] = MAT_FOLIAGE
    lvl = load_level("v3", levels_dir=str(_level(tmp_path, g).parent))
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert int(mat[2, 3]) == MAT_FOLIAGE
    assert not bool(vac[2, 3])


# ---------------------------------------------------------------------------
# (c) a v2 level on disk is refused, naming the migration tool
# ---------------------------------------------------------------------------

def test_v2_level_is_refused_naming_the_tool(tmp_path):
    """Property: the loader never reads a v2 file, whose 9 means SPACE, as
    v3, where 9 is foliage — it refuses and names the tool that converts it.
    Breaks if v2 is re-added to the loader's accepted versions (a v2 ship
    would silently load with foliage where its space was)."""
    d = _level(tmp_path, _ring_grid())
    level_lib.write_version_field(d / "level.toml", "2")
    with pytest.raises(ValueError, match="migrate_tilemap_v3"):
        load_level(d.name, levels_dir=str(d.parent))


# ---------------------------------------------------------------------------
# (d) the migration keeps the loader's planes bit-identical
# ---------------------------------------------------------------------------

def _as_v2_copy(src: Path, dst: Path) -> np.ndarray:
    """Copy a shipped (v3) level and turn the copy back into the v2 file it
    was before #27 (SPACE_CODE -> 9, version "2"), through level_lib.
    Returns the v2 grid."""
    shutil.copytree(src, dst)
    v3 = np.loadtxt(dst / "tilemap.csv", delimiter=",", dtype=np.int32)
    v2 = v3.copy()
    v2[v3 == SPACE_CODE] = V2_SPACE_CODE
    level_lib.write_tilemap_csv(dst, v2, csv_bak=False)
    level_lib.write_version_field(dst / "level.toml", "2")
    return v2


@pytest.mark.parametrize("level_name", ["ship_demo", "fire_tuning"])
def test_migration_keeps_material_and_vacuum_planes(tmp_path, level_name):
    """Property: migrating a v2 level gives a level that LOADS with the same
    material and is_vacuum planes the v2 decode produced, bit for bit (so the
    field digest cannot move). Breaks if the migration maps any code other
    than SPACE, or the v3 decode diverges from v2's on a material id."""
    dst = tmp_path / level_name
    v2 = _as_v2_copy(ROOT / "levels" / level_name, dst)
    assert (v2 == V2_SPACE_CODE).any()        # the level really has space
    mat2, vac2 = materials_from_tilemap(v2, "2")

    migrate_level(dst)
    lvl = load_level(level_name, levels_dir=str(tmp_path))
    assert lvl.version == LEVEL_FORMAT_VERSION
    mat3, vac3 = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert mat3.dtype == mat2.dtype
    np.testing.assert_array_equal(mat3, mat2)
    np.testing.assert_array_equal(vac3, vac2)
    # And the .bak files hold the v2 bytes the tool started from.
    assert (dst / "level.toml.bak").read_text(encoding="utf-8").count(
        'version = "2"') == 1


def test_migration_refuses_a_v3_level(tmp_path):
    """Property: the migration runs at most once — on a v3 level it refuses
    and writes nothing. Breaks if the version gate is dropped (nothing would
    happen to SPACE, but a future 9 -> x mapping would hit foliage)."""
    d = _level(tmp_path, _ring_grid())
    before = (d / "tilemap.csv").read_bytes()
    with pytest.raises(ValueError, match="nothing to migrate"):
        migrate_level(d)
    assert (d / "tilemap.csv").read_bytes() == before


# ---------------------------------------------------------------------------
# (e) ids past the old budget round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("new_id", [10, MATERIAL_ID_MAX])
def test_material_id_past_ten_round_trips(tmp_path, monkeypatch, new_id):
    """Property: a material id >= 10 (up to the int8 plane's 127) round-trips
    through level_lib's writer and the loader as itself, beside SPACE.
    Breaks if the loader caps ids below 127 or the CSV writer cannot carry
    them. There is no 11th config row yet, so the test registers a test-only
    row in ``MATERIAL_NAMES`` — the id door the loader reads — and asserts on
    the loader's planes only (no MaterialTable / GameMap is built)."""
    monkeypatch.setitem(materials_mod.MATERIAL_NAMES, new_id, "test_only")
    g = _ring_grid()
    g[2, 2:5] = new_id
    d = _level(tmp_path, g)
    lvl = load_level(d.name, levels_dir=str(d.parent))
    np.testing.assert_array_equal(lvl.tilemap, g)
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert (mat[2, 2:5] == new_id).all()
    assert not vac[2, 2:5].any()
    np.testing.assert_array_equal(vac, g == SPACE_CODE)


def test_unregistered_id_is_refused(tmp_path):
    """Property: an id with no material row is a hard error, never silent
    garbage. Breaks if the v3 decode stops validating codes."""
    g = _ring_grid()
    unused = max(MATERIAL_NAMES) + 1
    g[2, 2] = unused
    with pytest.raises(ValueError, match="unknown codes"):
        materials_from_tilemap(g, LEVEL_FORMAT_VERSION)


def test_id_past_the_int8_plane_is_refused(monkeypatch):
    """Property: a material table that outgrows the int8 plane is refused
    instead of wrapping (id 128 would land as -128). Breaks if the guard is
    removed before the plane is widened to int16 (Erik: "bump int8 to int16
    only if we ever fill it")."""
    monkeypatch.setitem(materials_mod.MATERIAL_NAMES, MATERIAL_ID_MAX + 1,
                        "too_far")
    with pytest.raises(ValueError, match="int8"):
        materials_from_tilemap(np.zeros((2, 2), dtype=np.int32),
                               LEVEL_FORMAT_VERSION)
