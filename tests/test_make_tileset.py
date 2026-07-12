"""Greybox tileset generator — tools/make_tileset.py (engine/15 §3, P1).

Pins the P1 contract the P2 baker builds on:

  - manifest coverage: every canon material (MATERIAL_NAMES — the tool owns
    no vocabulary) has a [materials.<name>] table with existing, correctly
    sized strips; SPACE is declared transparent with no strip.
  - edge16: every wall-family strip has exactly 16 pieces (index = 4-bit
    N/E/S/W mask) and the [groups] table declares the wall family.
  - determinism: two runs with the same args are byte-identical; the seed
    reaches the floor noise (and only the floor noise).
  - normals: unit length after uint8 decode, flat interior == (128,128,255)
    exactly, and the channel signs match tools/depth_to_normal.py (the
    convention lighting.fs consumes at u_normal_y_sign = +1): N bevel
    G < 128, S bevel G > 128, W bevel R < 128, E bevel R > 128.
  - px_per_tile is a real parameter (everything here generates at 32 px —
    fast — while the CLI default stays the chapter's 128).

Run:
    python -m pytest tests/test_make_tileset.py -q
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

from simulation.materials import MATERIAL_NAMES  # noqa: E402
from make_tileset import (BIT_E, BIT_N, BIT_S, BIT_W, DEFAULT_PX,  # noqa: E402
                          WALL_GROUP, WALL_PIECES, build_arg_parser,
                          build_tileset, main)

TEST_PX = 32          # keep the suite fast; the CLI default stays 128


@pytest.fixture(scope="session")
def tileset_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("greybox32")
    build_tileset(out, px=TEST_PX, seed=0)
    return out


@pytest.fixture(scope="session")
def manifest(tileset_dir) -> dict:
    with open(tileset_dir / "tileset.toml", "rb") as f:
        return tomllib.load(f)


def _strip(tileset_dir, filename) -> np.ndarray:
    return np.asarray(Image.open(tileset_dir / filename))


def _piece(strip: np.ndarray, index: int, px: int = TEST_PX) -> np.ndarray:
    return strip[:, index * px:(index + 1) * px]


# ---------------------------------------------------------------------------
# Manifest — full material coverage, derived from MATERIAL_NAMES
# ---------------------------------------------------------------------------

def test_manifest_covers_every_canon_material(tileset_dir, manifest):
    """Every material in MATERIAL_NAMES has a manifest table with existing,
    correctly sized strips — the §1 no-own-vocabulary rule, observable."""
    mats = manifest["materials"]
    assert set(mats) == set(MATERIAL_NAMES.values())
    assert manifest["tileset"]["px_per_tile"] == TEST_PX
    assert manifest["tileset"]["autotile"] == "edge16"
    for name, entry in mats.items():
        assert entry["mode"] in {"wall", "floor"}, name
        assert entry["pieces"] >= 1, name
        for key in ("diffuse", "normal"):
            img = Image.open(tileset_dir / entry[key])   # must exist + parse
            assert img.size == (entry["pieces"] * TEST_PX, TEST_PX), \
                f"{name}.{key}: strip is pieces*px wide, px tall"


def test_manifest_declares_space_transparent(manifest):
    """SPACE (tilemap code 9) is not a material: no strip, bakes transparent —
    the background starfield shows through (engine/15 §3)."""
    space = manifest["special"]["space"]
    assert space["mode"] == "transparent"
    assert "diffuse" not in space and "normal" not in space
    assert "space" not in manifest["materials"]


def test_manifest_wall_family_group(manifest):
    """The wall family is declared in [groups] (not code) and every wall-mode
    material points at a declared group (engine/15 §3 connectivity groups)."""
    groups = manifest["groups"]
    assert set(groups[WALL_GROUP]) == {"hull", "steel", "wood", "door", "glass"}
    for members in groups.values():
        for member in members:
            assert member in set(MATERIAL_NAMES.values())
    for name, entry in manifest["materials"].items():
        if entry["mode"] == "wall":
            assert entry["group"] in groups, name
            assert name in groups[entry["group"]]


# ---------------------------------------------------------------------------
# edge16 — 16 pieces per wall strip
# ---------------------------------------------------------------------------

def test_wall_strips_have_exactly_16_pieces(tileset_dir, manifest):
    wall_names = [n for n, e in manifest["materials"].items()
                  if e["mode"] == "wall"]
    assert wall_names, "at least the wall family must be wall-mode"
    for name in wall_names:
        entry = manifest["materials"][name]
        assert entry["pieces"] == WALL_PIECES == 16
        for key in ("diffuse", "normal"):
            strip = _strip(tileset_dir, entry[key])
            assert strip.shape[1] == WALL_PIECES * TEST_PX, f"{name}.{key}"
            assert strip.shape[0] == TEST_PX


def test_wall_pieces_differ_by_mask(tileset_dir, manifest):
    """The 16 pieces are genuinely distinct art (piece index = edge mask):
    isolated (0), N|S run (5), and interior (15) must not be identical."""
    strip = _strip(tileset_dir, manifest["materials"]["hull"]["diffuse"])
    isolated = _piece(strip, 0)
    ns_run = _piece(strip, BIT_N | BIT_S)
    interior = _piece(strip, 15)
    assert not np.array_equal(isolated, interior)
    assert not np.array_equal(ns_run, interior)
    # Interior is flat: one solid colour across the whole piece.
    assert (interior == interior[0, 0]).all()


# ---------------------------------------------------------------------------
# Determinism — same args, byte-identical files
# ---------------------------------------------------------------------------

def _files(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_two_runs_are_byte_identical(tileset_dir, tmp_path):
    again = tmp_path / "again"
    build_tileset(again, px=TEST_PX, seed=0)
    first, second = _files(tileset_dir), _files(again)
    assert first.keys() == second.keys()
    for filename in first:
        assert first[filename] == second[filename], filename


def test_cli_run_matches_library_run(tileset_dir, tmp_path):
    """main() with explicit argv produces the same bytes as build_tileset —
    the CLI adds nothing nondeterministic."""
    out = tmp_path / "cli"
    main(["--out", str(out), "--px", str(TEST_PX), "--seed", "0"])
    assert _files(out) == _files(tileset_dir)


def test_seed_reaches_floor_noise_only(tileset_dir, tmp_path, manifest):
    """A different seed changes the deck plating (noise is seeded) but not
    the walls (pure functions of px alone)."""
    other = tmp_path / "seed1"
    build_tileset(other, px=TEST_PX, seed=1)
    air = manifest["materials"]["air"]["diffuse"]
    hull = manifest["materials"]["hull"]["diffuse"]
    assert (tileset_dir / air).read_bytes() != (other / air).read_bytes()
    assert (tileset_dir / hull).read_bytes() == (other / hull).read_bytes()


# ---------------------------------------------------------------------------
# Normal maps — unit length, flat interior, depth_to_normal.py channel signs
# ---------------------------------------------------------------------------

def _decode(strip_rgb: np.ndarray) -> np.ndarray:
    return strip_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0


def test_normals_are_unit_length(tileset_dir, manifest):
    for name, entry in manifest["materials"].items():
        n = _decode(_strip(tileset_dir, entry["normal"]))
        norms = np.sqrt((n * n).sum(axis=-1))
        assert float(np.abs(norms - 1.0).max()) < 0.03, name   # uint8 quantum
        assert (n[..., 2] > 0.0).all(), f"{name}: z must point out of the map"


def test_normal_flat_interior_is_neutral(tileset_dir, manifest):
    """Piece 15 (all edges connected) is flat wall top: exactly (128,128,255).
    Floors declare flat normals: the whole strip is (128,128,255)."""
    hull_n = _strip(tileset_dir, manifest["materials"]["hull"]["normal"])
    assert (_piece(hull_n, 15) == (128, 128, 255)).all()
    air_n = _strip(tileset_dir, manifest["materials"]["air"]["normal"])
    assert (air_n == (128, 128, 255)).all()


def test_normal_green_convention_matches_depth_to_normal(tileset_dir, manifest):
    """Channel signs follow tools/depth_to_normal.py — what lighting.fs
    consumes at its default u_normal_y_sign = +1: a bevel facing image-north
    encodes G < 128 (and south > 128); west R < 128 (east > 128)."""
    strip = _strip(tileset_dir, manifest["materials"]["hull"]["normal"])
    mid = TEST_PX // 2
    n_bevel = _piece(strip, BIT_E | BIT_S | BIT_W)   # only N unconnected
    assert n_bevel[1, mid, 1] < 120                  # G dark on the N bevel
    assert n_bevel[1, mid, 0] == 128                 # no E-W slope there
    s_bevel = _piece(strip, BIT_N | BIT_E | BIT_W)   # only S unconnected
    assert s_bevel[TEST_PX - 2, mid, 1] > 136
    w_bevel = _piece(strip, BIT_N | BIT_E | BIT_S)   # only W unconnected
    assert w_bevel[mid, 1, 0] < 120
    assert w_bevel[mid, 1, 1] == 128
    e_bevel = _piece(strip, BIT_N | BIT_S | BIT_W)   # only E unconnected
    assert e_bevel[mid, TEST_PX - 2, 0] > 136


# ---------------------------------------------------------------------------
# Material identity — glass translucency, floor variants, crate
# ---------------------------------------------------------------------------

def test_glass_is_translucent_walls_are_opaque(tileset_dir, manifest):
    glass = _strip(tileset_dir, manifest["materials"]["glass"]["diffuse"])
    assert glass.shape[-1] == 4 and (glass[..., 3] < 255).all()
    hull = _strip(tileset_dir, manifest["materials"]["hull"]["diffuse"])
    assert (hull[..., 3] == 255).all()


def test_air_floor_variants_differ(tileset_dir, manifest):
    entry = manifest["materials"]["air"]
    assert entry["mode"] == "floor" and entry["pieces"] == 4
    strip = _strip(tileset_dir, entry["diffuse"])
    variants = [_piece(strip, v) for v in range(entry["pieces"])]
    assert any(not np.array_equal(variants[0], v) for v in variants[1:]), \
        "the seeded noise must actually vary the deck variants"


def test_furniture_is_a_single_distinct_tile(tileset_dir, manifest):
    entry = manifest["materials"]["furniture"]
    assert entry["mode"] == "floor" and entry["pieces"] == 1
    crate = _strip(tileset_dir, entry["diffuse"])
    crate_n = _strip(tileset_dir, entry["normal"])
    assert crate.shape == (TEST_PX, TEST_PX, 4)
    # The crate carries real relief (frame + brace), not a flat normal tile.
    assert not (crate_n == (128, 128, 255)).all()


# ---------------------------------------------------------------------------
# CLI defaults — the chapter's tunables live in one place
# ---------------------------------------------------------------------------

def test_cli_defaults_pin_the_chapter_values():
    defaults = {a.dest: a.default for a in build_arg_parser()._actions}
    assert defaults["px"] == DEFAULT_PX == 128    # source res, engine/15 §3
    assert defaults["seed"] == 0
    assert Path(defaults["out"]).parts[-3:] == ("art", "tilesets", "greybox")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
