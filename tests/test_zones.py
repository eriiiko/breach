"""Zones (Arc A patch A8 — level editor v3 design §5, entity design §3e/§5).

Pins the zone binding: two zone classes only (breach_site / extraction_zone,
intangible, no runtime behavior), the zones.npy uint8 paint-id grid
(presence-discovered, shape == tilemap, absent file = dormancy), the §5
validators (duplicate zone_id claim = hard load error; orphaned paint id and
zero-tile instance = warnings), breach-site rosters ([[unit_type, count]]
pairs, unit_type never registry-validated — units are NOT entities, §3e),
level_lib's zones.npy carrier (byte-stable, all-zero deletes), and
_upscale_level replication (--res must not shape-mismatch or drop zones).

All fixtures are synthetic tmp levels — the repo's levels/ (and every
digest-suite level) is NEVER given zones (zone entities flip the A4
entity-presence digest as designed; existing goldens stay untouched).

Run:
    conda run -n data python -m pytest tests/test_zones.py -q
"""
from __future__ import annotations

import struct
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_lib import (ZONES_FILENAME, format_entity_lines,  # noqa: E402
                       open_level, write_zones_npy)
from simulation.entities import (  # noqa: E402
    KIND_INT, KIND_ROSTER, REGISTRY, serialize_entity_state,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic level folders (never the repo's levels/)
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


PREFIX = ("# hand comment stays\n"
          'version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n'
          'tile_size_m = 0.333\ndiffuse = "diffuse.png"\n\n')
SUFFIX = ('[art.bare]\n# baked by the P2 baker\n'
          'diffuse = "diffuse.png"\n\n'
          '[bake]\ntileset = "x"\npx_per_tile = 8\nseed = 0\n')

BREACH = ('[[entity]]\n'
          'id = "bs_1"\n'
          'class = "breach_site"\n'
          'zone_id = 1\n'
          'roster = [["marine", 3], ["heavy", 1]]\n'
          '\n')
EXTRACT = ('[[entity]]\n'
           'id = "ex_1"\n'
           'class = "extraction_zone"\n'
           'zone_id = 2\n'
           'faction = 1\n'
           '\n')

GRID_SHAPE = (6, 8)                 # rows x cols — matches the tilemap csv


def _mini_level(tmp_path: Path, body: str = "", name: str = "mini") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(GRID_SHAPE[1]))
                  for _ in range(GRID_SHAPE[0])) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(PREFIX + body + SUFFIX,
                                  encoding="utf-8", newline="\n")
    return d


def _grid(*paints) -> np.ndarray:
    """A GRID_SHAPE uint8 grid with ``(row, col, id)`` tiles painted."""
    g = np.zeros(GRID_SHAPE, dtype=np.uint8)
    for r, c, zid in paints:
        g[r, c] = zid
    return g


def _paint(d: Path, grid: np.ndarray) -> None:
    np.save(d / ZONES_FILENAME, grid)


def _load(d: Path):
    return level_loader.load(str(d))


def _load_silent(d: Path):
    """Load asserting NO warnings fire (the §5 happy path)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _load(d)


# ---------------------------------------------------------------------------
# (a) The two zone classes — registered, intangible, schema per §5
# ---------------------------------------------------------------------------

def test_zone_classes_registered_and_intangible():
    for name in ("breach_site", "extraction_zone"):
        cls = REGISTRY[name]
        assert cls.INTANGIBLE is True           # design §5: never on the grid
        assert cls.SIGNALS == () and cls.INPUTS == ()   # no runtime behavior
        assert cls.runtime_digest_rows(None) == ()      # no synced runtime rows
        fields = {f.name: f for f in cls.FIELDS}
        zid = fields["zone_id"]
        assert zid.kind == KIND_INT
        assert zid.default is None              # REQUIRED at authoring time
        assert (zid.minimum, zid.maximum) == (1, 255)   # uint8, 0 = unpainted
        assert fields["faction"].kind == KIND_INT
    # roster is breach_site's alone (§5), the narrow roster kind.
    bs = {f.name: f for f in REGISTRY["breach_site"].FIELDS}
    assert bs["roster"].kind == KIND_ROSTER
    assert bs["roster"].default == ()
    assert "roster" not in {f.name for f in REGISTRY["extraction_zone"].FIELDS}


def test_zone_id_required_and_bounded(tmp_path):
    body = '[[entity]]\nid = "bs_1"\nclass = "breach_site"\n\n'
    with pytest.raises(ValueError, match="missing required.*zone_id"):
        _load(_mini_level(tmp_path, body))
    body = ('[[entity]]\nid = "bs_1"\nclass = "breach_site"\n'
            'zone_id = 0\n\n')
    with pytest.raises(ValueError, match="below minimum"):
        _load(_mini_level(tmp_path, body, name="mini2"))
    body = ('[[entity]]\nid = "bs_1"\nclass = "breach_site"\n'
            'zone_id = 256\n\n')
    with pytest.raises(ValueError, match="above maximum"):
        _load(_mini_level(tmp_path, body, name="mini3"))


# ---------------------------------------------------------------------------
# (b) Roster parsing (§5) — pairs validated, unit_type NEVER registry-checked
# ---------------------------------------------------------------------------

def test_roster_parsed_as_authored(tmp_path):
    d = _mini_level(tmp_path, BREACH + EXTRACT)
    _paint(d, _grid((1, 1, 1), (4, 6, 2)))
    lvl = _load_silent(d)
    bs = lvl.entities[0]
    assert bs.class_name == "breach_site"
    assert bs.fields["roster"] == [["marine", 3], ["heavy", 1]]
    assert bs.fields["faction"] == 0            # registry default
    assert lvl.entities[1].fields["faction"] == 1


def test_roster_unit_type_is_never_registry_validated(tmp_path):
    # Units are NOT entities (entity design §3e): any non-empty string is a
    # legal unit_type — the UNIT system owns the vocabulary at stack-2.
    body = ('[[entity]]\nid = "bs_1"\nclass = "breach_site"\nzone_id = 1\n'
            'roster = [["absolutely_not_a_registered_anything", 7]]\n\n')
    lvl = _load(_mini_level(tmp_path, body))    # zero-tile warning is fine
    assert lvl.entities[0].fields["roster"] == [
        ["absolutely_not_a_registered_anything", 7]]


@pytest.mark.parametrize("roster", [
    'roster = ["marine"]',                      # not a pair
    'roster = [["marine", 3, 9]]',              # triple, not a pair
    'roster = [["marine", "three"]]',           # count not an int
    'roster = [[3, 1]]',                        # unit_type not a string
    'roster = [["", 2]]',                       # empty unit_type
    'roster = [["marine", 0]]',                 # count < 1
])
def test_roster_bad_forms_hard_error(tmp_path, roster):
    body = ('[[entity]]\nid = "bs_1"\nclass = "breach_site"\nzone_id = 1\n'
            + roster + '\n\n')
    with pytest.raises(ValueError, match="must be a roster"):
        _load(_mini_level(tmp_path, body))


# ---------------------------------------------------------------------------
# (c) zones.npy load path — presence-discovered, uint8, shape == tilemap
# ---------------------------------------------------------------------------

def test_zone_free_level_is_dormant(tmp_path):
    lvl = _load_silent(_mini_level(tmp_path))
    assert lvl.zone_grid is None                # absent file = no zones
    assert lvl.entities == []


def test_zones_happy_path_binds_silently(tmp_path):
    d = _mini_level(tmp_path, BREACH + EXTRACT)
    g = _grid((0, 0, 1), (1, 0, 1), (5, 7, 2))
    _paint(d, g)
    lvl = _load_silent(d)                       # every §5 validator satisfied
    assert lvl.zone_grid is not None
    assert lvl.zone_grid.dtype == np.uint8
    assert np.array_equal(lvl.zone_grid, g)


def test_wrong_dtype_hard_error(tmp_path):
    d = _mini_level(tmp_path, BREACH)
    np.save(d / ZONES_FILENAME, np.zeros(GRID_SHAPE, dtype=np.int32))
    with pytest.raises(ValueError, match="must be dtype uint8"):
        _load(d)


def test_wrong_shape_hard_error(tmp_path):
    d = _mini_level(tmp_path, BREACH)
    np.save(d / ZONES_FILENAME, np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        _load(d)


# ---------------------------------------------------------------------------
# (d) The §5 validators — dup id errors; orphan paint / zero tiles warn
# ---------------------------------------------------------------------------

def test_duplicate_zone_id_hard_error(tmp_path):
    # One paint-id namespace ACROSS both classes — a breach_site and an
    # extraction_zone may not share a zone_id either.
    body = (BREACH
            + '[[entity]]\nid = "ex_1"\nclass = "extraction_zone"\n'
              'zone_id = 1\n\n')
    with pytest.raises(ValueError, match="already claimed by 'bs_1'"):
        _load(_mini_level(tmp_path, body))


def test_orphan_paint_warns_not_fatal(tmp_path):
    d = _mini_level(tmp_path, BREACH)
    _paint(d, _grid((0, 0, 1), (3, 3, 9)))      # id 9 painted, never claimed
    with pytest.warns(UserWarning, match="paints zone id 9"):
        lvl = _load(d)
    assert np.array_equal(np.unique(lvl.zone_grid), [0, 1, 9])  # load kept it


def test_zero_tile_instance_warns(tmp_path):
    d = _mini_level(tmp_path, BREACH + EXTRACT)
    _paint(d, _grid((0, 0, 1)))                 # zone 2 claimed, never painted
    with pytest.warns(UserWarning, match="'ex_1'.*zone_id 2.*0 painted"):
        _load(d)


def test_zone_instance_without_npy_warns_zero_tiles(tmp_path):
    # Absent zones.npy + a zone instance = that instance has 0 painted tiles.
    with pytest.warns(UserWarning, match="'bs_1'.*zone_id 1.*0 painted"):
        _load(_mini_level(tmp_path, BREACH))


# ---------------------------------------------------------------------------
# (e) level_lib — the zones.npy carrier (byte-stable; all-zero deletes)
# ---------------------------------------------------------------------------

def test_write_zones_npy_round_trip_byte_stable(tmp_path):
    d = _mini_level(tmp_path, BREACH + EXTRACT)
    g = _grid((0, 0, 1), (2, 5, 2))
    _, has = write_zones_npy(d, g, npy_bak=False)
    assert has is True
    first = (d / ZONES_FILENAME).read_bytes()
    assert np.array_equal(_load_silent(d).zone_grid, g)     # loader accepts
    write_zones_npy(d, g, npy_bak=False)        # write -> write is stable
    assert (d / ZONES_FILENAME).read_bytes() == first
    # Load -> write is stable too (the grid IS the file, by identity).
    write_zones_npy(d, _load_silent(d).zone_grid, npy_bak=False)
    assert (d / ZONES_FILENAME).read_bytes() == first


def test_write_zones_npy_all_zero_deletes_and_bak(tmp_path):
    d = _mini_level(tmp_path)
    g = _grid((0, 0, 1))
    write_zones_npy(d, g, npy_bak=False)
    # Pre-session .bak mirrors the water carrier's contract.
    nbak, has = write_zones_npy(d, _grid((1, 1, 2)), npy_bak=True)
    assert has is True and nbak is not None and nbak.is_file()
    # All-zero = no zones: the file goes away, matching absent-file dormancy.
    _, has = write_zones_npy(d, np.zeros(GRID_SHAPE, np.uint8), npy_bak=False)
    assert has is False
    assert not (d / ZONES_FILENAME).is_file()
    assert _load_silent(d).zone_grid is None


def test_write_zones_npy_rejects_bad_grids(tmp_path):
    d = _mini_level(tmp_path)
    with pytest.raises(ValueError, match="integer paint ids"):
        write_zones_npy(d, np.zeros(GRID_SHAPE, np.float64), npy_bak=False)
    bad = np.zeros(GRID_SHAPE, np.int32)
    bad[0, 0] = 300
    with pytest.raises(ValueError, match="fit uint8"):
        write_zones_npy(d, bad, npy_bak=False)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="fit uint8"):
        write_zones_npy(d, bad, npy_bak=False)


def test_zone_entity_toml_round_trip_byte_stable(tmp_path):
    # The [[entity]] side round-trips byte-stably through level_lib —
    # including the nested roster array (authored keys, authored order).
    d = _mini_level(tmp_path, BREACH + EXTRACT)
    _paint(d, _grid((0, 0, 1), (1, 1, 2)))
    toml = d / "level.toml"
    before = toml.read_bytes()
    handle = open_level(str(d))
    handle.save({"entity":
                 lambda nl: format_entity_lines(handle.data.entities, nl)})
    assert toml.read_bytes() == before


# ---------------------------------------------------------------------------
# (f) _upscale_level — --res must not shape-mismatch or drop zones
# ---------------------------------------------------------------------------

def test_upscale_replicates_zone_grid(tmp_path):
    from main import _upscale_level             # heavy import: test-local
    d = _mini_level(tmp_path, BREACH + EXTRACT)
    g = _grid((0, 0, 1), (2, 3, 1), (5, 7, 2))
    _paint(d, g)
    lvl = _load_silent(d)
    n_painted = int((g > 0).sum())
    _upscale_level(lvl, 3)
    assert lvl.tilemap.shape == (GRID_SHAPE[0] * 3, GRID_SHAPE[1] * 3)
    assert lvl.zone_grid.shape == lvl.tilemap.shape     # never a mismatch
    expected = np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)
    assert np.array_equal(lvl.zone_grid, expected)      # same zones, denser
    assert int((lvl.zone_grid > 0).sum()) == n_painted * 9  # nothing dropped
    assert sorted(int(v) for v in np.unique(lvl.zone_grid)) == [0, 1, 2]
    # The binding side is untouched: same instances, same zone_ids/rosters.
    assert [e.id for e in lvl.entities] == ["bs_1", "ex_1"]
    assert lvl.entities[0].fields["roster"] == [["marine", 3], ["heavy", 1]]


def test_upscale_without_zones_stays_dormant(tmp_path):
    from main import _upscale_level             # heavy import: test-local
    lvl = _load_silent(_mini_level(tmp_path))
    _upscale_level(lvl, 2)
    assert lvl.zone_grid is None


# ---------------------------------------------------------------------------
# (g) A4 interplay — zone instances serialize; roster is NOT a synced row
# ---------------------------------------------------------------------------

def test_zone_entities_serialize_with_roster_excluded(tmp_path):
    d = _mini_level(tmp_path, BREACH + EXTRACT)
    _paint(d, _grid((0, 0, 1), (1, 1, 2)))
    lvl = _load_silent(d)
    payload = serialize_entity_state(lvl.entities)
    assert b"zone_id|" in payload               # KIND_INT: hashed
    assert b"faction|" in payload               # KIND_INT: hashed
    assert b"roster" not in payload             # authoring-bound: never hashed
