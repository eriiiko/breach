"""Level format v2 [art] block + align transform (F2 + F2.1).

Pins level_editor_and_format_v2_proposal §1.2/§1.3 as implemented by
level_loader.py:

  - FREEZE: flat (v1-spelling) art keys parse to exactly the same LevelData
    fields as before F2 — for the real unhcr_vessel level AND for a synthetic
    version="1" level. New fields sit at backward-compatible defaults.
  - The v2 [art] block parses: [art.bare]/[art.furniture]/[art.destroyed]
    paths resolve, [art] background/emissive_mask resolve, declared-but-
    missing files still raise, [art.bare] wins over a flat duplicate.
  - [art.align]: offset_px + px_per_tile parse; px_per_tile is a scalar OR a
    per-axis [x, y] pair (F2.1) — both normalize to a pair on LevelData;
    defaults to (art_w / grid_w, art_h / grid_h) (read from the PNG header)
    when absent; malformed values raise.
  - unhcr_vessel_2 loads end-to-end (the level F2 exists for). Skipped where
    the untracked WIP folder is absent.
  - Align-transform math: tile -> art-pixel mapping (tile_to_art_px).
  - F2.1 renderer plumbing: set_art_align normalizes scalar/pair, and the
    per-axis src/UV rect math (renderer.lighting.art_src_and_uv_rect) is
    pinned for the legacy and the explicit-align path.
  - F2.1 ALIGN tool save: tools/align_level_art.py save_align rewrites ONLY
    the [art.align] lines (rest byte-identical, .bak written) and the result
    round-trips through the loader.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_level_art_v2.py -q
"""
from __future__ import annotations

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

from level_loader import load, materials_from_tilemap, tile_to_art_px  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic-level helpers
# ---------------------------------------------------------------------------

def _write_png(path: Path, w: int, h: int, rgb=(120, 120, 120)) -> None:
    """Write a minimal valid 8-bit RGB PNG (pure Python, no PIL)."""
    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = (b"\x00" + bytes(rgb) * w) * h          # filter byte 0 per row
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _make_level(tmp_path: Path, name: str, toml_text: str,
                art_files=(), grid=(6, 4)) -> Path:
    """Build a level folder under tmp_path; returns the folder path.

    ``art_files`` is an iterable of (relative_path, (w, h)) PNGs to create.
    ``grid`` is (rows, cols) for a v2-coded tilemap (SPACE border, air core).
    """
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    rows, cols = grid
    tm = np.full((rows, cols), 9, dtype=int)
    tm[1:-1, 1:-1] = 0
    np.savetxt(folder / "tilemap.csv", tm, fmt="%d", delimiter=",")
    for rel, (w, h) in art_files:
        _write_png(folder / rel, w, h)
    (folder / "level.toml").write_text(toml_text, encoding="utf-8")
    return folder


def _load_from(tmp_path: Path, name: str):
    # levels_dir joins with the loader's own directory, but an absolute path
    # replaces it (pathlib semantics) — so synthetic levels load from tmp_path.
    return load(name, levels_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# FREEZE: flat v1-spelling keys parse exactly as before F2
# ---------------------------------------------------------------------------

def test_freeze_unhcr_vessel_flat_keys():
    """The shipped level (flat art keys) — art fields identical to pre-F2."""
    lvl = load("unhcr_vessel", levels_dir="prototypes")
    base = lvl.path
    assert lvl.diffuse_path == base / "diffuse.png"
    assert lvl.normal_path == base / "normal.png"
    assert lvl.background_path == base / "background.png"
    assert lvl.emissive_mask_path is None
    assert lvl.emissive_bloom_path is None
    assert lvl.wall_mask_path is None
    # New v2 fields at backward-compatible defaults:
    assert lvl.specular_path is None
    assert lvl.furniture_diffuse_path is None
    assert lvl.furniture_normal_path is None
    assert lvl.furniture_specular_path is None
    assert lvl.destroyed_diffuse_path is None
    assert lvl.destroyed_normal_path is None
    assert lvl.destroyed_specular_path is None
    assert lvl.art_align_explicit is False
    assert lvl.art_offset_px == (0.0, 0.0)
    # Default px_per_tile = (art_w / grid_w, art_h / grid_h) — exactly the
    # implicit per-axis transform of the legacy stretch-to-grid draw
    # (diffuse.png is 1000x2400 over a 50x120 grid: 20 px/tile both axes).
    assert lvl.art_px_per_tile == pytest.approx((20.0, 20.0))


def test_freeze_version_1_level(tmp_path):
    """A version="1" level (legacy CSV vocabulary + flat art keys) parses
    with identical art fields and untouched v1 material translation."""
    folder = tmp_path / "old_one"
    folder.mkdir()
    tm = np.zeros((6, 4), dtype=int)
    tm[1:-1, 1:-1] = 4          # v1 vocabulary: 4..8 = interior air
    tm[0, :] = 1                # v1: hull
    np.savetxt(folder / "tilemap.csv", tm, fmt="%d", delimiter=",")
    _write_png(folder / "diffuse.png", 40, 60)
    _write_png(folder / "normal.png", 40, 60)
    (folder / "level.toml").write_text(
        'version = "1"\n'
        'name = "Old One"\n'
        'tilemap = "tilemap.csv"\n'
        'diffuse = "diffuse.png"\n'
        'normal = "normal.png"\n',
        encoding="utf-8")
    lvl = _load_from(tmp_path, "old_one")
    assert lvl.version == "1"
    assert lvl.diffuse_path == folder / "diffuse.png"
    assert lvl.normal_path == folder / "normal.png"
    assert lvl.background_path is None
    assert lvl.art_align_explicit is False
    assert lvl.art_offset_px == (0.0, 0.0)
    # Per-axis default pair (art_w / grid_w, art_h / grid_h) — 40x60 art over
    # a 4x6 grid is 10 px/tile both axes.
    assert lvl.art_px_per_tile == pytest.approx((10.0, 10.0))
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert vac.sum() > 0 and (mat == 1).sum() > 0   # v1 gate still applies


# ---------------------------------------------------------------------------
# v2 [art] block
# ---------------------------------------------------------------------------

_V2_FULL_TOML = """
version = "2"
name = "Art Block"
tilemap = "tilemap.csv"

[art]
background = "bg.png"
emissive_mask = "emissive.png"

[art.bare]
diffuse = "up/bare_d.png"
normal = "up/bare_n.png"
specular = "up/bare_s.png"

[art.furniture]
diffuse = "up/furn_d.png"
normal = "up/furn_n.png"

[art.destroyed]
diffuse = "up/dest_d.png"
normal = "up/dest_n.png"

[art.align]
offset_px = [4.0, -6.0]
px_per_tile = 12.5
"""

_V2_ART_FILES = [
    ("bg.png", (16, 16)), ("emissive.png", (16, 16)),
    ("up/bare_d.png", (64, 96)), ("up/bare_n.png", (64, 96)),
    ("up/bare_s.png", (64, 96)),
    ("up/furn_d.png", (64, 96)), ("up/furn_n.png", (64, 96)),
    ("up/dest_d.png", (64, 96)), ("up/dest_n.png", (64, 96)),
]


def test_v2_art_block_parses(tmp_path):
    folder = _make_level(tmp_path, "lvl", _V2_FULL_TOML, _V2_ART_FILES)
    lvl = _load_from(tmp_path, "lvl")
    assert lvl.diffuse_path == folder / "up/bare_d.png"
    assert lvl.normal_path == folder / "up/bare_n.png"
    assert lvl.specular_path == folder / "up/bare_s.png"
    assert lvl.background_path == folder / "bg.png"
    assert lvl.emissive_mask_path == folder / "emissive.png"
    assert lvl.furniture_diffuse_path == folder / "up/furn_d.png"
    assert lvl.furniture_normal_path == folder / "up/furn_n.png"
    assert lvl.furniture_specular_path is None
    assert lvl.destroyed_diffuse_path == folder / "up/dest_d.png"
    assert lvl.destroyed_normal_path == folder / "up/dest_n.png"
    assert lvl.art_align_explicit is True
    assert lvl.art_offset_px == (4.0, -6.0)
    # Scalar px_per_tile normalizes to a pair (same scale both axes).
    assert lvl.art_px_per_tile == pytest.approx((12.5, 12.5))


def test_v2_bare_wins_over_flat_duplicate(tmp_path):
    """[art.bare] is the new spelling; a leftover flat key loses to it."""
    toml = (
        'version = "2"\n'
        'tilemap = "tilemap.csv"\n'
        'diffuse = "old_d.png"\n'
        'normal = "old_n.png"\n'
        '[art.bare]\n'
        'diffuse = "new_d.png"\n'
    )
    files = [("old_d.png", (8, 8)), ("old_n.png", (8, 8)), ("new_d.png", (8, 8))]
    folder = _make_level(tmp_path, "lvl", toml, files)
    lvl = _load_from(tmp_path, "lvl")
    assert lvl.diffuse_path == folder / "new_d.png"
    # No [art.bare] normal -> the flat normal still applies (old keys keep
    # working inside a v2-with-art level).
    assert lvl.normal_path == folder / "old_n.png"


def test_v2_missing_files_still_raise(tmp_path):
    # Bare diffuse declared but absent.
    _make_level(tmp_path, "a", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\ndiffuse = "nope.png"\n'))
    with pytest.raises(ValueError, match="Diffuse texture not found"):
        _load_from(tmp_path, "a")
    # Furniture diffuse declared but absent — error names the toml location.
    _make_level(tmp_path, "b", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\ndiffuse = "d.png"\n'
        '[art.furniture]\ndiffuse = "nope.png"\n'),
        [("d.png", (8, 8))])
    with pytest.raises(ValueError, match=r"\[art.furniture\] diffuse"):
        _load_from(tmp_path, "b")
    # [art] present but no diffuse in either spelling.
    _make_level(tmp_path, "c", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\nnormal = "n.png"\n'),
        [("n.png", (8, 8))])
    with pytest.raises(ValueError, match=r"\[art.bare\] 'diffuse'"):
        _load_from(tmp_path, "c")


def test_align_defaults_and_validation(tmp_path):
    # No [art.align]: offset (0,0), px_per_tile = art_w / grid_w, not explicit.
    _make_level(tmp_path, "a", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\ndiffuse = "d.png"\n'),
        [("d.png", (64, 96))], grid=(6, 4))
    lvl = _load_from(tmp_path, "a")
    assert lvl.art_align_explicit is False
    assert lvl.art_offset_px == (0.0, 0.0)
    assert lvl.art_px_per_tile == pytest.approx((64 / 4, 96 / 6))
    # [art.align] with offset only: px_per_tile still defaults, but explicit.
    _make_level(tmp_path, "b", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\ndiffuse = "d.png"\n'
        '[art.align]\noffset_px = [10, 20]\n'),
        [("d.png", (64, 96))], grid=(6, 4))
    lvl = _load_from(tmp_path, "b")
    assert lvl.art_align_explicit is True
    assert lvl.art_offset_px == (10.0, 20.0)
    assert lvl.art_px_per_tile == pytest.approx((16.0, 16.0))
    # Malformed offset_px / px_per_tile raise.
    _make_level(tmp_path, "c", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\ndiffuse = "d.png"\n'
        '[art.align]\noffset_px = [1, 2, 3]\n'),
        [("d.png", (8, 8))])
    with pytest.raises(ValueError, match="offset_px"):
        _load_from(tmp_path, "c")
    _make_level(tmp_path, "d", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\ndiffuse = "d.png"\n'
        '[art.align]\npx_per_tile = 0\n'),
        [("d.png", (8, 8))])
    with pytest.raises(ValueError, match="px_per_tile"):
        _load_from(tmp_path, "d")


def test_align_px_per_tile_pair_parses(tmp_path):
    """F2.1: px_per_tile accepts a per-axis [x, y] pair (the v2 art's
    proportions differ from the tilemap per axis)."""
    _make_level(tmp_path, "p", (
        'version = "2"\ntilemap = "tilemap.csv"\n'
        '[art.bare]\ndiffuse = "d.png"\n'
        '[art.align]\noffset_px = [1.0, -38.4]\n'
        'px_per_tile = [77.40, 54.44]\n'),
        [("d.png", (64, 96))])
    lvl = _load_from(tmp_path, "p")
    assert lvl.art_align_explicit is True
    assert lvl.art_offset_px == (1.0, -38.4)
    assert lvl.art_px_per_tile == pytest.approx((77.40, 54.44))


def test_align_px_per_tile_pair_validation(tmp_path):
    """Malformed pairs raise: wrong arity, non-numbers, non-positive axis."""
    for bad in ("[1.0, 2.0, 3.0]", '["a", 2.0]', "[12.5, 0]", "[-1.0, 2.0]"):
        _make_level(tmp_path, "bad", (
            'version = "2"\ntilemap = "tilemap.csv"\n'
            '[art.bare]\ndiffuse = "d.png"\n'
            f'[art.align]\npx_per_tile = {bad}\n'),
            [("d.png", (8, 8))])
        with pytest.raises(ValueError, match="px_per_tile"):
            _load_from(tmp_path, "bad")


# ---------------------------------------------------------------------------
# The level F2 exists for
# ---------------------------------------------------------------------------

_VESSEL2 = ROOT / "prototypes" / "unhcr_vessel_2"


@pytest.mark.skipif(not _VESSEL2.is_dir(),
                    reason="unhcr_vessel_2 is untracked WIP art "
                           "(present only on the authoring machine)")
def test_unhcr_vessel_2_loads_end_to_end():
    lvl = load("unhcr_vessel_2", levels_dir="prototypes")
    assert lvl.version == "2"
    # Which layer fills the displayed (bare) slot is an authoring choice on
    # the untracked toml (currently the furnished interim until F3 composes
    # per tile) — pin structure, not the filename.
    assert lvl.diffuse_path.suffix == ".png"
    assert lvl.diffuse_path.exists()
    assert lvl.normal_path is not None and lvl.normal_path.exists()
    assert (lvl.furniture_diffuse_path is not None
            and lvl.furniture_diffuse_path.exists())
    assert (lvl.destroyed_diffuse_path is not None
            and lvl.destroyed_diffuse_path.exists())
    assert lvl.background_path is not None and lvl.background_path.exists()
    assert lvl.art_align_explicit is True
    # The align VALUES are hand-tuned on the authoring machine (seeded by the
    # analytic candidate, refined with tools/align_level_art.py) — pin the
    # structure, not the numbers, so tuning never breaks the suite.
    assert len(lvl.art_offset_px) == 2
    assert all(isinstance(v, float) for v in lvl.art_offset_px)
    assert len(lvl.art_px_per_tile) == 2
    assert all(v > 0.0 for v in lvl.art_px_per_tile)
    assert lvl.tilemap.shape == (120, 50)
    # CSV speaks canon v2 codes; the material translation must accept it.
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    assert vac.sum() > 0 and (mat == 1).sum() > 0
    assert len(lvl.spawns) == 7


# ---------------------------------------------------------------------------
# Align-transform math (§1.3)
# ---------------------------------------------------------------------------

def test_tile_to_art_px_math():
    # Art pixel offset_px lands on grid (0, 0).
    assert tile_to_art_px(0, 0, (7.0, 9.0), 78.0) == (7.0, 9.0)
    # px_per_tile art pixels span one tile.
    assert tile_to_art_px(1, 0, (0.0, 0.0), 24.0) == (24.0, 0.0)
    assert tile_to_art_px(0, 1, (0.0, 0.0), 24.0) == (0.0, 24.0)
    # Offsets (including negative) + scale compose linearly.
    assert tile_to_art_px(3, 5, (10.0, -8.0), 12.5) == (47.5, 54.5)
    # Fractional tiles map continuously (sub-tile precision).
    assert tile_to_art_px(0.5, 2.25, (0.0, 0.0), 78.0) == (39.0, 175.5)
    # The implicit v1 transform: 1000x2400 art over a 50x120 grid at
    # 20 px/tile spans the art exactly — the legacy full-stretch draw.
    assert tile_to_art_px(50, 120, (0.0, 0.0), 20.0) == (1000.0, 2400.0)
    # The vessel-2 first guess: 78 px/tile spans 3900 px over 50 tiles.
    assert tile_to_art_px(50, 0, (0.0, 0.0), 78.0) == (3900.0, 0.0)
    # F2.1 — per-axis pair: each axis scales independently.
    assert tile_to_art_px(1, 1, (0.0, 0.0), (77.4, 54.44)) == (77.4, 54.44)
    assert (tile_to_art_px(50, 120, (1.0, -38.4), (77.4, 54.44))
            == pytest.approx((1.0 + 50 * 77.4, -38.4 + 120 * 54.44)))
    # A scalar and the equal pair are the same transform.
    assert (tile_to_art_px(3, 5, (10.0, -8.0), (12.5, 12.5))
            == tile_to_art_px(3, 5, (10.0, -8.0), 12.5))


# ---------------------------------------------------------------------------
# F2.1 renderer plumbing — per-axis align rect (unit-level, no GL context)
# ---------------------------------------------------------------------------

def test_set_art_align_normalizes_scalar_and_pair():
    """Whatever set_art_align stores is the 4-tuple draw_lit_world consumes:
    (off_x, off_y, ppt_x, ppt_y) — scalar px_per_tile fills both axes."""
    from renderer.lighting import LightingPass

    class _Stub:        # set_art_align only touches self.art_align
        pass

    stub = _Stub()
    LightingPass.set_art_align(stub, (1.0, -38.4), (77.4, 54.44))
    assert stub.art_align == (1.0, -38.4, 77.4, 54.44)
    LightingPass.set_art_align(stub, [2, 3], 24)
    assert stub.art_align == (2.0, 3.0, 24.0, 24.0)


def test_art_src_and_uv_rect_per_axis():
    """The src/UV rect math behind draw_lit_world, from a known pair."""
    from renderer.lighting import art_src_and_uv_rect

    # Explicit per-axis align over the vessel-2 geometry (grid 50x120,
    # art 3900x6900): src spans offset + ppt*grid per axis; the UV rect is
    # the same rect normalized by the art dimensions per axis (this is what
    # u_art_uv_rect receives — the shader divides per-component, so per-axis
    # needs no shader change).
    src, uv = art_src_and_uv_rect((1.0, -38.4, 77.4, 54.44),
                                  grid_w=50, grid_h=120,
                                  art_w=3900.0, art_h=6900.0)
    assert src == pytest.approx((1.0, -38.4, 77.4 * 50, 54.44 * 120))
    assert uv == pytest.approx((1.0 / 3900.0, -38.4 / 6900.0,
                                (77.4 * 50) / 3900.0, (54.44 * 120) / 6900.0))
    # Legacy path (no [art.align]): FULL art src + identity UV, bit-exact —
    # the pre-F2 stretch draw.
    src, uv = art_src_and_uv_rect(None, grid_w=50, grid_h=120,
                                  art_w=1000.0, art_h=2400.0)
    assert src == (0.0, 0.0, 1000.0, 2400.0)
    assert uv == (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# F2.1 ALIGN tool — [art.align] in-place save round-trip
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "tools"))

_ALIGN_TOML = """\
version = "2"
name = "Save Roundtrip"   # stays byte-identical
tilemap = "tilemap.csv"

[art.bare]
diffuse = "d.png"

[art.align]
# a comment INSIDE the block — must survive the save untouched
offset_px = [0.0, 0.0]
px_per_tile = 78.0

[[spawn]]
name = "Alpha"
team = 0
x = 1
y = 1
"""


def test_save_align_roundtrip(tmp_path):
    """save_align: write -> load -> values match; every other line of the
    toml byte-identical; .bak carries the original bytes."""
    from align_level_art import save_align

    folder = _make_level(tmp_path, "rt", _ALIGN_TOML, [("d.png", (64, 96))])
    toml_path = folder / "level.toml"
    original = toml_path.read_bytes()

    bak = save_align(toml_path, (1.0, -38.4), (77.4, 54.44))
    assert bak.read_bytes() == original

    # Round-trip through the real loader: saved values come back (offset at
    # 1 decimal, px_per_tile at 2 decimals).
    lvl = _load_from(tmp_path, "rt")
    assert lvl.art_align_explicit is True
    assert lvl.art_offset_px == (1.0, -38.4)
    assert lvl.art_px_per_tile == pytest.approx((77.40, 54.44))

    # Byte-level: ONLY the two align assignment lines changed.
    old_lines = original.decode("utf-8").splitlines()
    new_lines = toml_path.read_bytes().decode("utf-8").splitlines()
    assert len(old_lines) == len(new_lines)
    changed = [(o, n) for o, n in zip(old_lines, new_lines) if o != n]
    assert changed == [
        ("offset_px = [0.0, 0.0]", "offset_px = [1.0, -38.4]"),
        ("px_per_tile = 78.0", "px_per_tile = [77.40, 54.44]"),
    ]


def test_save_align_appends_missing_block(tmp_path):
    """A level.toml without [art.align] gains a fresh block at EOF and still
    loads (valid TOML; values round-trip)."""
    from align_level_art import save_align

    toml = ('version = "2"\ntilemap = "tilemap.csv"\n'
            '[art.bare]\ndiffuse = "d.png"\n')
    folder = _make_level(tmp_path, "nb", toml, [("d.png", (64, 96))])
    toml_path = folder / "level.toml"
    original = toml_path.read_bytes()

    save_align(toml_path, (2.5, 3.25), (10.0, 20.0))
    assert toml_path.read_bytes().startswith(original)   # purely appended
    lvl = _load_from(tmp_path, "nb")
    assert lvl.art_align_explicit is True
    # offset saves at 1 decimal: 3.25 -> 3.2/3.3 (banker's rounding aside,
    # exactly what format_align_lines wrote is what comes back).
    assert lvl.art_offset_px == (2.5, float(f"{3.25:.1f}"))
    assert lvl.art_px_per_tile == pytest.approx((10.0, 20.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
