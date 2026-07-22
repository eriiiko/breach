"""Autotile baker — tools/bake_level_art.py (engine/15 §4, P2).

Pins the P2 contract the P3 editor builds on:

  - edge16 piece selection: the 4-bit N/E/S/W mask (straights, corners, T,
    cross, isolated; off-map/SPACE/non-group neighbours clear) selects the
    strip piece, connectivity via the manifest [groups] (glass counts hull).
  - compose rule: every non-SPACE tile = AIR deck under-layer + material
    piece on top; glass (alpha 150) integer-alpha-composites over the deck;
    SPACE is alpha-0 in BOTH outputs.
  - floor-variant hash: (tx*73856093 + ty*19349663 + seed) % pieces —
    position-based, golden values pinned (changing a prime re-bakes every
    tiled level).
  - downscale: integer box filter; normal maps are RENORMALIZED after the
    resize (a naive box average denormalizes every bevel diagonal).
  - determinism: two bakes are byte-identical; a bare re-bake from the
    recorded [bake] block reproduces the explicit bake.
  - region re-bake == the corresponding crop of a full bake (masks read
    neighbours OUTSIDE the rect; clipping pinned).
  - level.toml writeback: byte-preserving, line-targeted, .bak (save_align
    house style); loader round-trip on the committed levels/bake_demo.
  - golden bake: a tiny map through a 16 px tileset at 8 px/tile matches the
    committed pixels in tests/data/ exactly.

Run:
    python -m pytest tests/test_bake_level_art.py -q
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import level_loader  # noqa: E402
from level_loader import SPACE_CODE, materials_from_tilemap  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED,  # noqa: E402
                                  MAT_FURNITURE, MAT_GLASS, MAT_HULL,
                                  MAT_WOOD)
from make_tileset import build_tileset  # noqa: E402
from bake_level_art import (BIT_E, BIT_N, BIT_S, BIT_W,  # noqa: E402
                            DEFAULT_PX_PER_TILE, DEFAULT_TILESET,
                            DIFFUSE_FILENAME, FLOOR_HASH_P1, FLOOR_HASH_P2,
                            NORMAL_FILENAME, alpha_composite_over,
                            bake_full, bake_level, bake_region,
                            downscale_normal_strip, edge16_mask,
                            floor_variant, load_tileset, write_bake_blocks)
from bake_level_art import main as bake_main  # noqa: E402

TEST_PX = 16          # tmp tileset resolution — fast; CLI default stays 128
DATA_DIR = Path(__file__).resolve().parent / "data"

# Short canon-code aliases for hand-written maps (tilemap.csv literals).
AI, HU, WO, DO, GL, FU, SP = (MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR,
                              MAT_GLASS, MAT_FURNITURE, SPACE_CODE)

# The pinned golden map (also the determinism/region workhorse): space ring,
# hull ring with a 2-tile glass run, furniture, air, wood + door.
GOLDEN_MAP = np.array([
    [SP, SP, SP, SP, SP, SP],
    [SP, HU, GL, GL, HU, SP],
    [SP, HU, FU, AI, HU, SP],
    [SP, HU, WO, DO, HU, SP],
    [SP, SP, SP, SP, SP, SP],
], dtype=np.int32)


@pytest.fixture(scope="session")
def tileset16_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("greybox16")
    build_tileset(out, px=TEST_PX, seed=0)
    return out


@pytest.fixture(scope="session")
def ts16(tileset16_dir):
    return load_tileset(tileset16_dir)


def _piece(strip: np.ndarray, index: int, px: int = TEST_PX) -> np.ndarray:
    return strip[:, index * px:(index + 1) * px]


def _tile(img: np.ndarray, tx: int, ty: int, px: int) -> np.ndarray:
    return img[ty * px:(ty + 1) * px, tx * px:(tx + 1) * px]


def _write_min_level(level_dir: Path, tilemap: np.ndarray) -> Path:
    """A minimal pre-bake v2 level folder (the baker adds the art blocks)."""
    level_dir.mkdir(parents=True, exist_ok=True)
    csv = "\n".join(",".join(str(int(v)) for v in row)
                    for row in tilemap.tolist()) + "\n"
    (level_dir / "tilemap.csv").write_text(csv, encoding="ascii", newline="\n")
    (level_dir / "level.toml").write_text(
        '# tmp fixture level\nversion = "2"\nname = "Tmp"\n'
        'tilemap = "tilemap.csv"\ntile_size_m = 0.333\n',
        encoding="utf-8", newline="\n")
    return level_dir


# ---------------------------------------------------------------------------
# edge16 piece selection — straights / corners / T / cross / isolated
# ---------------------------------------------------------------------------

WALL_CODES = frozenset({MAT_HULL, MAT_WOOD, MAT_DOOR, 4, MAT_GLASS})


def _blank(n: int = 5) -> np.ndarray:
    return np.full((n, n), MAT_AIR, dtype=np.int32)


def test_edge16_mask_cross_and_arms():
    m = _blank()
    for x, y in ((2, 1), (1, 2), (2, 2), (3, 2), (2, 3)):
        m[y, x] = MAT_HULL
    assert edge16_mask(m, 2, 2, WALL_CODES) == 15          # cross center
    assert edge16_mask(m, 2, 1, WALL_CODES) == BIT_S       # N arm
    assert edge16_mask(m, 3, 2, WALL_CODES) == BIT_W       # E arm
    assert edge16_mask(m, 2, 3, WALL_CODES) == BIT_N       # S arm
    assert edge16_mask(m, 1, 2, WALL_CODES) == BIT_E       # W arm


def test_edge16_mask_straight_runs():
    m = _blank()
    m[2, 1:4] = MAT_HULL                                   # horizontal run
    assert edge16_mask(m, 1, 2, WALL_CODES) == BIT_E
    assert edge16_mask(m, 2, 2, WALL_CODES) == BIT_E | BIT_W
    assert edge16_mask(m, 3, 2, WALL_CODES) == BIT_W
    m = _blank()
    m[1:4, 2] = MAT_HULL                                   # vertical run
    assert edge16_mask(m, 2, 1, WALL_CODES) == BIT_S
    assert edge16_mask(m, 2, 2, WALL_CODES) == BIT_N | BIT_S
    assert edge16_mask(m, 2, 3, WALL_CODES) == BIT_N


def test_edge16_mask_corner_t_isolated():
    m = _blank()
    m[1, 1] = m[1, 2] = m[2, 1] = MAT_HULL                 # L corner at (1,1)
    assert edge16_mask(m, 1, 1, WALL_CODES) == BIT_E | BIT_S
    m = _blank()
    m[2, 1:4] = MAT_HULL
    m[3, 2] = MAT_HULL                                     # T junction
    assert edge16_mask(m, 2, 2, WALL_CODES) == BIT_E | BIT_S | BIT_W
    m = _blank()
    m[2, 2] = MAT_HULL                                     # isolated pillar
    assert edge16_mask(m, 2, 2, WALL_CODES) == 0


def test_edge16_mask_offmap_space_and_nongroup_are_clear():
    m = np.full((3, 3), MAT_HULL, dtype=np.int32)          # wall to the borders
    assert edge16_mask(m, 0, 0, WALL_CODES) == BIT_E | BIT_S   # off-map clear
    assert edge16_mask(m, 1, 1, WALL_CODES) == 15
    m = _blank()
    m[1, 2] = SPACE_CODE                                   # N neighbour SPACE
    m[2, 1] = MAT_FURNITURE                                # W neighbour non-group
    m[2, 2] = MAT_HULL
    assert edge16_mask(m, 2, 2, WALL_CODES) == 0


def test_group_connectivity_glass_continues_the_hull_run(ts16):
    """A glass window run inside a hull wall: every bit counts any wall-FAMILY
    neighbour (manifest [groups], not per-material islands)."""
    wall = ts16.group_codes["wall"]
    m = _blank()
    m[2, 0:5] = MAT_HULL
    m[2, 2] = MAT_GLASS
    assert edge16_mask(m, 2, 2, wall) == BIT_E | BIT_W     # glass mid-run
    assert edge16_mask(m, 1, 2, wall) == BIT_E | BIT_W     # hull sees glass
    m[2, 2] = MAT_DOOR                                     # door frames likewise
    assert edge16_mask(m, 2, 2, wall) == BIT_E | BIT_W


# ---------------------------------------------------------------------------
# Compose — piece selection observable in baked pixels
# ---------------------------------------------------------------------------

def test_baked_wall_tiles_are_the_masked_pieces(ts16):
    """Opaque hull composites reduce to an exact strip-piece copy, so the
    baked tile pins piece selection pixel-for-pixel (cross: 15/4/8/1/2)."""
    m = _blank()
    for x, y in ((2, 1), (1, 2), (2, 2), (3, 2), (2, 3)):
        m[y, x] = MAT_HULL
    patch = bake_region(m, ts16, (0, 0, 5, 5), px_per_tile=TEST_PX, seed=0)
    strip_d = ts16.materials["hull"].diffuse
    strip_n = ts16.materials["hull"].normal
    for tx, ty, mask in ((2, 2, 15), (2, 1, BIT_S), (3, 2, BIT_W),
                         (2, 3, BIT_N), (1, 2, BIT_E)):
        assert np.array_equal(_tile(patch.diffuse, tx, ty, TEST_PX),
                              _piece(strip_d, mask)), (tx, ty)
        tile_n = _tile(patch.normal, tx, ty, TEST_PX)
        assert np.array_equal(tile_n[..., :3], _piece(strip_n, mask)), (tx, ty)
        assert (tile_n[..., 3] == 255).all()


def test_baked_air_tiles_use_position_hash_variants(ts16):
    m = _blank(4)
    patch = bake_region(m, ts16, (0, 0, 4, 4), px_per_tile=TEST_PX, seed=0)
    strip = ts16.materials["air"].diffuse
    n_air = ts16.materials["air"].pieces
    for tx, ty in ((0, 0), (1, 0), (0, 1), (3, 2)):
        fv = floor_variant(tx, ty, 0, n_air)
        assert np.array_equal(_tile(patch.diffuse, tx, ty, TEST_PX),
                              _piece(strip, fv)), (tx, ty)
    # The hash actually varies across positions (not one variant everywhere).
    assert not np.array_equal(_tile(patch.diffuse, 0, 0, TEST_PX),
                              _tile(patch.diffuse, 1, 0, TEST_PX))


def test_glass_alpha_composites_over_the_deck(ts16):
    """Glass (alpha 150) shows deck plating through the pane: the baked tile
    is the exact integer 'over' of the glass piece on the tile's deck variant
    — hand-checked formula, opaque result, not a bare piece copy."""
    m = _blank(3)
    m[1, 1] = MAT_GLASS                                    # isolated -> piece 0
    patch = bake_region(m, ts16, (0, 0, 3, 3), px_per_tile=TEST_PX, seed=0)
    glass = _piece(ts16.materials["glass"].diffuse, 0).astype(np.uint32)
    assert (glass[..., 3] == 150).all()                    # P1 pane alpha
    fv = floor_variant(1, 1, 0, ts16.materials["air"].pieces)
    deck = _piece(ts16.materials["air"].diffuse, fv).astype(np.uint32)
    a = glass[..., 3:4]
    expected_rgb = (glass[..., :3] * a + deck[..., :3] * (255 - a) + 127) // 255
    tile = _tile(patch.diffuse, 1, 1, TEST_PX)
    assert np.array_equal(tile[..., :3], expected_rgb.astype(np.uint8))
    assert (tile[..., 3] == 255).all()                     # over opaque deck
    assert not np.array_equal(tile[..., :3], glass[..., :3].astype(np.uint8))
    # Normals: the pane owns the tile surface (no alpha blend on normals).
    tile_n = _tile(patch.normal, 1, 1, TEST_PX)
    assert np.array_equal(tile_n[..., :3],
                          _piece(ts16.materials["glass"].normal, 0))


def test_space_is_transparent_in_both_outputs(ts16):
    patch = bake_region(GOLDEN_MAP, ts16, (0, 0, 6, 5),
                        px_per_tile=TEST_PX, seed=0)
    space = GOLDEN_MAP == SPACE_CODE
    for img in (patch.diffuse, patch.normal):
        per_tile = img.reshape(5, TEST_PX, 6, TEST_PX, 4)
        for ty, tx in zip(*np.nonzero(space)):
            assert (per_tile[ty, :, tx, :, :] == 0).all(), (tx, ty)
        for ty, tx in zip(*np.nonzero(~space)):
            assert (per_tile[ty, :, tx, :, 3] == 255).all(), (tx, ty)


# ---------------------------------------------------------------------------
# Floor-variant hash — pinned golden values (position-based, region-safe)
# ---------------------------------------------------------------------------

def test_floor_variant_hash_pinned():
    """Golden values of (tx*P1 + ty*P2 + seed) % pieces with the fixed primes
    73856093 / 19349663. Changing either prime (or the formula) re-bakes
    every tiled level — this test is the tripwire."""
    assert (FLOOR_HASH_P1, FLOOR_HASH_P2) == (73856093, 19349663)
    assert floor_variant(0, 0, 0, 4) == 0
    assert floor_variant(1, 0, 0, 4) == 1        # 73856093 % 4
    assert floor_variant(0, 1, 0, 4) == 3        # 19349663 % 4
    assert floor_variant(0, 0, 1, 4) == 1        # seed shifts the pattern
    assert floor_variant(2, 3, 7, 4) == 2        # 205761182 % 4
    assert floor_variant(5, 9, 1234, 4) == 2     # 543428666 % 4
    assert floor_variant(7, 3, 0, 1) == 0        # furniture: single piece


# ---------------------------------------------------------------------------
# Downscale — integer box filter; normals renormalized
# ---------------------------------------------------------------------------

def test_downscaled_normals_are_renormalized(ts16):
    """After a 2x box downscale, decoded normals are unit length again — and
    differ from the naive (denormalized) average on the bevel diagonals."""
    strip = ts16.materials["hull"].normal
    k = 2
    down = downscale_normal_strip(strip, k)
    n = down.astype(np.float32) / 255.0 * 2.0 - 1.0
    norms = np.sqrt((n * n).sum(axis=-1))
    assert float(np.abs(norms - 1.0).max()) < 0.03          # uint8 quantum
    # Naive box average (no renormalize) must NOT equal the renormalized
    # result — bevel-boundary blocks average two unit vectors to length < 1.
    h, w, _ = strip.shape
    s = strip.reshape(h // k, k, w // k, k, 3).astype(np.int64).sum(axis=(1, 3))
    v = s.astype(np.float64) / (k * k * 127.5) - 1.0
    naive = ((v * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)
    assert not np.array_equal(naive, down)


def test_bake_at_half_resolution_keeps_unit_normals(ts16):
    patch = bake_region(GOLDEN_MAP, ts16, (0, 0, 6, 5), px_per_tile=8, seed=0)
    solid = patch.normal[..., 3] == 255
    n = patch.normal[..., :3].astype(np.float32) / 255.0 * 2.0 - 1.0
    norms = np.sqrt((n * n).sum(axis=-1))
    assert float(np.abs(norms[solid] - 1.0).max()) < 0.03


def test_non_integer_downscale_is_rejected(ts16):
    with pytest.raises(ValueError, match="divide"):
        bake_region(GOLDEN_MAP, ts16, (0, 0, 6, 5), px_per_tile=6, seed=0)


# ---------------------------------------------------------------------------
# Determinism + region re-bake == full-bake crop
# ---------------------------------------------------------------------------

def test_two_bakes_are_byte_identical(tileset16_dir, tmp_path):
    """bake_level twice -> byte-identical PNGs; and a bare re-bake driven by
    the recorded [bake] block reproduces the explicit bake exactly."""
    level = _write_min_level(tmp_path / "lvl", GOLDEN_MAP)
    bake_level(level, tileset=tileset16_dir, px_per_tile=8, seed=3)
    first = {p.name: p.read_bytes() for p in level.glob("baked_*.png")}
    assert set(first) == {DIFFUSE_FILENAME, NORMAL_FILENAME}
    bake_level(level, tileset=tileset16_dir, px_per_tile=8, seed=3)
    for name, data in first.items():
        assert (level / name).read_bytes() == data, name
    summary = bake_level(level)          # all-None: [bake] block drives it
    assert (summary["px_per_tile"], summary["seed"]) == (8, 3)
    for name, data in first.items():
        assert (level / name).read_bytes() == data, name


def test_region_rebake_equals_full_bake_crop(ts16):
    """The P3 live-preview property: any rect's patch equals the crop of a
    full bake — including rects whose edge masks depend on neighbours OUTSIDE
    the rect (the wood/door/glass tiles at the rect border)."""
    full = bake_full(GOLDEN_MAP, ts16, px_per_tile=8, seed=0)
    assert full.rect == (0, 0, 6, 5)
    for rect in ((1, 1, 3, 2), (2, 1, 2, 3), (3, 3, 1, 1), (0, 0, 6, 5)):
        patch = bake_region(GOLDEN_MAP, ts16, rect, px_per_tile=8, seed=0)
        x0, y0, tw, th = patch.rect
        assert (x0, y0, tw, th) == rect
        crop = slice(y0 * 8, (y0 + th) * 8), slice(x0 * 8, (x0 + tw) * 8)
        assert np.array_equal(patch.diffuse, full.diffuse[crop]), rect
        assert np.array_equal(patch.normal, full.normal[crop]), rect


def test_region_rect_is_clipped_to_the_grid(ts16):
    full = bake_full(GOLDEN_MAP, ts16, px_per_tile=8, seed=0)
    patch = bake_region(GOLDEN_MAP, ts16, (-2, -1, 4, 3),
                        px_per_tile=8, seed=0)
    assert patch.rect == (0, 0, 2, 2)
    assert np.array_equal(patch.diffuse, full.diffuse[0:16, 0:16])
    with pytest.raises(ValueError, match="does not intersect"):
        bake_region(GOLDEN_MAP, ts16, (10, 10, 2, 2), px_per_tile=8, seed=0)


def test_unknown_tilemap_code_is_a_hard_error(ts16):
    m = _blank(3)
    m[1, 1] = 8                # not a v2 code (7 became door_closed in A6)
    with pytest.raises(ValueError, match="unknown code 8"):
        bake_region(m, ts16, (0, 0, 3, 3), px_per_tile=TEST_PX, seed=0)


# ---------------------------------------------------------------------------
# Golden bake — committed pixels in tests/data/
# ---------------------------------------------------------------------------

def test_golden_bake_pinned(ts16):
    """GOLDEN_MAP through the 16 px tileset at 8 px/tile (the downscale +
    renormalize path) matches the committed goldens pixel-for-pixel. A diff
    here means the bake output changed for every tiled level — regenerate
    the goldens only for a DELIBERATE compose/hash/filter change."""
    patch = bake_region(GOLDEN_MAP, ts16, (0, 0, 6, 5), px_per_tile=8, seed=0)
    gold_d = np.asarray(Image.open(DATA_DIR / "bake16_golden_diffuse.png"))
    gold_n = np.asarray(Image.open(DATA_DIR / "bake16_golden_normal.png"))
    assert np.array_equal(patch.diffuse, gold_d)
    assert np.array_equal(patch.normal, gold_n)


# ---------------------------------------------------------------------------
# level.toml writeback — save_align house style
# ---------------------------------------------------------------------------

def test_writeback_appends_blocks_and_writes_bak(tmp_path):
    toml = tmp_path / "level.toml"
    body = ('# hand comment stays\nversion = "2"\nname = "T"\n'
            'tilemap = "tilemap.csv"\ntile_size_m = 0.333\n\n'
            '[[spawn]]\nname = "Alpha"\nteam = 0\nx = 3\ny = 4\n')
    toml.write_text(body, encoding="utf-8", newline="\n")
    original = toml.read_bytes()
    bak = write_bake_blocks(toml, tileset_rel="art/tilesets/greybox",
                            px_per_tile=64, seed=5)
    assert bak.read_bytes() == original
    text = toml.read_text(encoding="utf-8")
    assert text.startswith(body)                # append-only for a fresh file
    with open(toml, "rb") as f:
        raw = tomllib.load(f)
    assert raw["art"]["bare"] == {"diffuse": DIFFUSE_FILENAME,
                                  "normal": NORMAL_FILENAME}
    assert raw["art"]["align"] == {"offset_px": [0.0, 0.0],
                                   "px_per_tile": [64.0, 64.0]}
    assert raw["bake"] == {"tileset": "art/tilesets/greybox",
                           "px_per_tile": 64, "seed": 5}
    assert raw["spawn"][0]["name"] == "Alpha"   # untouched


def test_writeback_is_line_targeted_on_existing_blocks(tmp_path):
    toml = tmp_path / "level.toml"
    toml.write_text(
        'version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n\n'
        "[art.bare]\n# baked by the P2 baker\ndiffuse = \"old.png\"\n"
        'normal = "old_n.png"\n\n[bake]\ntileset = "art/tilesets/greybox"\n'
        "px_per_tile = 64\nseed = 1\n",
        encoding="utf-8", newline="\n")
    write_bake_blocks(toml, tileset_rel="art/tilesets/greybox",
                      px_per_tile=64, seed=1)
    first = toml.read_text(encoding="utf-8").splitlines()
    write_bake_blocks(toml, tileset_rel="art/tilesets/greybox",
                      px_per_tile=64, seed=9)
    second = toml.read_text(encoding="utf-8").splitlines()
    assert len(first) == len(second)
    diff = [(a, b) for a, b in zip(first, second) if a != b]
    assert diff == [("seed = 1", "seed = 9")]   # ONLY the seed line moved
    assert "# baked by the P2 baker" in second  # comments preserved


def test_writeback_preserves_crlf(tmp_path):
    toml = tmp_path / "level.toml"
    toml.write_bytes(b'version = "2"\r\nname = "T"\r\ntilemap = "t.csv"\r\n')
    write_bake_blocks(toml, tileset_rel="x", px_per_tile=32, seed=0)
    data = toml.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n")   # every newline is CRLF


# ---------------------------------------------------------------------------
# CLI + loader round-trip on the committed demo level
# ---------------------------------------------------------------------------

def test_cli_bakes_a_level_folder(tileset16_dir, tmp_path, capsys):
    level = _write_min_level(tmp_path / "clilvl", GOLDEN_MAP)
    bake_main([str(level), "--tileset", str(tileset16_dir),
               "--px-per-tile", "8", "--seed", "0"])
    out = capsys.readouterr().out
    assert "6x5 tiles @ 8 px/tile" in out
    assert (level / DIFFUSE_FILENAME).is_file()
    assert (level / NORMAL_FILENAME).is_file()
    assert (level / "level.toml.bak").is_file()
    # No emissive output: the greybox manifest declares no emissive pieces.
    assert not (level / "emissive_mask.png").exists()
    # The baked PNGs match the library bake exactly.
    patch = bake_region(GOLDEN_MAP, load_tileset(tileset16_dir),
                        (0, 0, 6, 5), px_per_tile=8, seed=0)
    assert np.array_equal(
        np.asarray(Image.open(level / DIFFUSE_FILENAME)), patch.diffuse)


def test_bake_demo_loader_round_trip():
    """The committed levels/bake_demo (baked at 64 with the repo greybox
    tileset) loads through level_loader without error — the ordinary-v2 gate:
    loader and renderer never know the art was baked."""
    lvl = level_loader.load("bake_demo")
    assert lvl.version == "2"
    materials_from_tilemap(lvl.tilemap, lvl.version)       # validates codes
    assert lvl.diffuse_path.name == DIFFUSE_FILENAME
    assert lvl.normal_path is not None
    assert lvl.normal_path.name == NORMAL_FILENAME
    assert lvl.art_align_explicit
    assert lvl.art_offset_px == (0.0, 0.0)
    assert lvl.art_px_per_tile == (float(DEFAULT_PX_PER_TILE),) * 2
    bake = lvl.raw_toml["bake"]
    assert bake["tileset"] == DEFAULT_TILESET
    assert bake["px_per_tile"] == DEFAULT_PX_PER_TILE
    art = Image.open(lvl.diffuse_path)
    assert art.size == (lvl.width * DEFAULT_PX_PER_TILE,
                        lvl.height * DEFAULT_PX_PER_TILE)
    # The demo exercises the whole compose: space, hull, glass run, wood
    # dividing wall, a door, furniture.
    codes = set(np.unique(lvl.tilemap).tolist())
    assert {MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_GLASS,
            MAT_FURNITURE, SPACE_CODE} <= codes


# ---------------------------------------------------------------------------
# MAT_DOOR_CLOSED preview bake (Arc C3 mandatory sub-fix)
# ---------------------------------------------------------------------------

def test_real_committed_tileset_bakes_mat_door_closed():
    """Regression pin for the C3 mandatory sub-fix: the DOOR tool stamps
    MAT_DOOR_CLOSED (id 7) immediately on placement, so the editor's live-
    preview re-bake must not crash on it. This loads the REAL COMMITTED
    art/tilesets/greybox (not a freshly generated one, unlike test_make_
    tileset.py's `tileset_dir` fixture) — the bug was the committed
    manifest/PNGs going stale after MAT_DOOR_CLOSED was added to
    materials.py; make_tileset.py's code already covered it generically
    (derived from MATERIAL_NAMES at build time), so the fix was
    regenerating the committed artifact + a curated colour/group, not a
    code change to the baker."""
    ts = load_tileset(ROOT / DEFAULT_TILESET)
    assert "door_closed" in ts.materials
    grid = np.array([
        [SP, SP, SP, SP, SP],
        [SP, HU, HU, HU, SP],
        [SP, HU, MAT_DOOR_CLOSED, HU, SP],
        [SP, HU, HU, HU, SP],
        [SP, SP, SP, SP, SP],
    ], dtype=np.int32)
    patch = bake_full(grid, ts, px_per_tile=DEFAULT_PX_PER_TILE, seed=0)
    n = grid.shape[0] * DEFAULT_PX_PER_TILE
    assert patch.diffuse.shape[:2] == (n, n)
    assert patch.normal.shape[:2] == (n, n)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
