"""[water] initial state — loader, GameMap seed, editor WATER mode
(engine/15 §2.3, patch P5 — docs/patch_levels_p5_water.md).

All headless, no ``breach_physics`` (GameMap and the editor helpers are pure
numpy). The physics-bound half of P5's test spine — Σ-conservation and the
at-rest byte-identity of the glass-bounded aquarium seed over 100 runner
ticks — lives in ``tests/test_level_water_physics.py``, a separate module
because the ONLY correct skip pattern (module-level try/except pytest.skip,
test_bedrock_cliff_counts.py) skips a whole module at collection and must
not take these headless tests down with it.

Pins the design-gate hard requirements:

  - loader: ``[water] depth_map`` .npy parse (np.load, allow_pickle=False)
    with hard validation — shape == tilemap, dtype int32, min >= 0,
    readable file — every error a ValueError carrying the offending path
    (design §2.2); no [water] key -> ``water_depth_q is None`` (dormancy);
  - identity round-trip: editor-written file -> loader -> GameMap
    ``water_depth``, int-exact (the .npy carrier's whole point vs the
    dropped auto-scaling PNG — critique B1);
  - GameMap seed: masked to ``(~solid) & (~is_vacuum)``, one counted
    RuntimeWarning on masked-out cells (hand-authoring backstop), and the
    seed lives in ``__init__`` ONLY — a config hot-reload
    (``reload_material_table`` -> ``_update_caches``) must NOT re-flood a
    drained tank (design §2.3 anti-pattern);
  - editor: solid-for-water = sim-exact ``permeability <= 0`` (critique M1
    — glass bounds, furniture does NOT); fills bounded by glass AND by
    SPACE (critique M2), 4-connected; refusal on SPACE/solid starts;
    save-time wall-over-pool masking (critique M3); ``[water]`` managed
    block + water_init.npy writeback sharing the P3/P4 .bak contracts;
    all-dry save removes the block and the file (dormancy pin); the water
    undo ring stores independent snapshots (critique M4).

Run:
    python -m pytest tests/test_level_water.py -q
"""
from __future__ import annotations

import struct
import sys
import tomllib
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from level_loader import SPACE_CODE, LevelData, load  # noqa: E402
from simulation import water_fixed  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_FURNITURE,  # noqa: E402
                                  MAT_GLASS, MAT_HULL)
from level_edit_common import UndoRing  # noqa: E402
from map_editor import (WATER_FILENAME, format_water_lines,  # noqa: E402
                        mask_water_to_open, scaffold_grid, water_fill_region,
                        water_open_mask, water_solid_codes, write_lights,
                        write_spawns, write_water)


# ---------------------------------------------------------------------------
# Fixture — a minimal loadable level folder (the P4 test_level_lights pattern)
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


BASE_TOML = ('version = "2"\nname = "Wet"\ntilemap = "tilemap.csv"\n'
             'tile_size_m = 0.333\ndiffuse = "diffuse.png"\n\n')

# 8x6 v2 grid: SPACE border, hull ring, 4x2 air interior — every mask class
# (vacuum / solid / open) present for the seed tests.
GRID_H, GRID_W = 6, 8


def _grid() -> np.ndarray:
    return scaffold_grid(GRID_W, GRID_H)     # SPACE border + hull ring + air


def _mini_level(tmp_path: Path, grid: np.ndarray | None = None,
                water_toml: str = "", name: str = "wet") -> Path:
    d = tmp_path / name
    d.mkdir()
    g = _grid() if grid is None else np.asarray(grid)
    (d / "tilemap.csv").write_text(
        "\n".join(",".join(str(int(v)) for v in row)
                  for row in g.tolist()) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(BASE_TOML + water_toml,
                                  encoding="utf-8", newline="\n")
    return d


def _interior_depth(g: np.ndarray, metres: float) -> np.ndarray:
    """Q16.16 depth grid: ``metres`` on every open (air) tile, 0 elsewhere."""
    depth = np.zeros(g.shape, dtype=np.int32)
    depth[g == MAT_AIR] = int(water_fixed.quantize(metres))
    return depth


# ---------------------------------------------------------------------------
# Loader — dormancy, parse, validation
# ---------------------------------------------------------------------------

def test_no_water_key_gives_none_and_dry_gamemap(tmp_path):
    """Dormancy at the loader level (design §2.5): no [water] -> None, and
    the GameMap built from it is bit-exactly dry — the existing runtime
    dormancy trio (test_water_integration) covers the tick side."""
    lvl = load(str(_mini_level(tmp_path)))
    assert lvl.water_depth_q is None
    g = GameMap(lvl)
    assert not g.water_depth.any()
    assert not g.flow_vx.any() and not g.flow_vy.any()


def test_water_parse_valid_int32_identity(tmp_path):
    d = _mini_level(tmp_path, water_toml=(
        f'[water]\ndepth_map = "{WATER_FILENAME}"\n'))
    depth = _interior_depth(_grid(), 1.2)
    np.save(d / WATER_FILENAME, depth)
    lvl = load(str(d))
    assert lvl.water_depth_q is not None
    assert lvl.water_depth_q.dtype == np.int32
    # Identity — the file IS the field (design §2.1): int-exact, no
    # re-quantization anywhere in the load path.
    assert np.array_equal(lvl.water_depth_q, depth)


def test_water_table_missing_depth_map_key(tmp_path):
    d = _mini_level(tmp_path, water_toml="[water]\n")
    with pytest.raises(ValueError) as ei:
        load(str(d))
    assert "depth_map" in str(ei.value)
    assert "level.toml" in str(ei.value)          # path-bearing


def test_water_array_of_tables_rejected(tmp_path):
    d = _mini_level(tmp_path, water_toml=(
        '[[water]]\ndepth_map = "water_init.npy"\n'))
    with pytest.raises(ValueError) as ei:
        load(str(d))
    assert "must be a table" in str(ei.value)


def test_water_file_missing(tmp_path):
    d = _mini_level(tmp_path, water_toml=(
        '[water]\ndepth_map = "nope.npy"\n'))
    with pytest.raises(ValueError) as ei:
        load(str(d))
    msg = str(ei.value)
    assert "missing" in msg and "nope.npy" in msg  # path-bearing


def test_water_garbage_file_rejected(tmp_path):
    d = _mini_level(tmp_path, water_toml=(
        f'[water]\ndepth_map = "{WATER_FILENAME}"\n'))
    (d / WATER_FILENAME).write_bytes(b"this is not an .npy file")
    with pytest.raises(ValueError) as ei:
        load(str(d))
    assert WATER_FILENAME in str(ei.value)


@pytest.mark.parametrize("depth_builder,fragment", [
    # wrong dtype: float32 metres instead of int32 Q16.16
    (lambda g: (np.zeros(g.shape, dtype=np.float32) + 0.5), "int32"),
    # wrong dtype: int64
    (lambda g: np.zeros(g.shape, dtype=np.int64), "int32"),
    # wrong shape
    (lambda g: np.zeros((g.shape[0] + 1, g.shape[1]), dtype=np.int32),
     "shape"),
    # negative depth
    (lambda g: np.full(g.shape, -1, dtype=np.int32), "negative"),
    # wrong ndim (flat) — caught by the shape check
    (lambda g: np.zeros(g.size, dtype=np.int32), "shape"),
])
def test_water_validation_errors(tmp_path, depth_builder, fragment):
    d = _mini_level(tmp_path, water_toml=(
        f'[water]\ndepth_map = "{WATER_FILENAME}"\n'))
    np.save(d / WATER_FILENAME, depth_builder(_grid()))
    with pytest.raises(ValueError) as ei:
        load(str(d))
    msg = str(ei.value)
    assert fragment in msg
    assert WATER_FILENAME in msg                  # path-bearing


def test_water_pickle_payload_rejected(tmp_path):
    """allow_pickle=False is load-bearing: an object-array .npy must be a
    clean ValueError, never a pickle execution."""
    d = _mini_level(tmp_path, water_toml=(
        f'[water]\ndepth_map = "{WATER_FILENAME}"\n'))
    obj = np.empty(( GRID_H, GRID_W), dtype=object)
    obj[...] = None
    np.save(d / WATER_FILENAME, obj, allow_pickle=True)
    with pytest.raises(ValueError):
        load(str(d))


# ---------------------------------------------------------------------------
# GameMap seed — masking, warn, hot-reload anti-pattern, round trip
# ---------------------------------------------------------------------------

def test_editor_to_loader_to_gamemap_round_trip(tmp_path):
    """The full identity chain (design §2.5): editor write path
    (mask + write_water) -> loader -> GameMap.water_depth, int-exact."""
    g = _grid()
    g[2:4, 3] = MAT_GLASS                       # a glass divider
    d = _mini_level(tmp_path, grid=g)
    solid = water_solid_codes()
    depth = _interior_depth(g, 1.2)
    masked, cleared = mask_water_to_open(depth, g, solid)
    assert cleared == 0                          # interior-only paint
    write_spawns(d / "level.toml", [], write_bak=True)   # Ctrl+S order
    write_lights(d / "level.toml", [], write_bak=False)
    write_water(d, masked, toml_bak=False, npy_bak=True)

    lvl = load(str(d))
    assert np.array_equal(lvl.water_depth_q, masked)
    gmap = GameMap(lvl)
    # Editor mask == GameMap mask (both derive from permeability <= 0), so
    # the seed lands verbatim — byte-identical field.
    assert np.array_equal(gmap.water_depth, masked)
    assert int(gmap.water_depth.sum()) == int(masked.sum())


def test_seed_masks_solid_and_vacuum_with_counted_warning():
    """Glass-box (design §2.3): a hand-broken file with depth EVERYWHERE
    seeds only ``(~solid) & (~is_vacuum)`` and warns once with the count."""
    g = _grid()
    depth = np.full(g.shape, int(water_fixed.quantize(0.5)), dtype=np.int32)
    lvl = LevelData(
        name="broken_water", version="2", path=Path("."), tilemap=g,
        tile_size_m=0.333, diffuse_path=Path("."), water_depth_q=depth)
    with pytest.warns(RuntimeWarning) as rec:
        gmap = GameMap(lvl)
    open_ = (~gmap.solid) & (~gmap.is_vacuum)
    n_bad = int(np.count_nonzero(~open_))
    assert n_bad > 0                              # non-vacuous fixture
    # Water strictly inside the mask, exact value on every open tile.
    assert not gmap.water_depth[~open_].any()
    assert (gmap.water_depth[open_]
            == int(water_fixed.quantize(0.5))).all()
    # One warning, carrying the masked-out cell count.
    msgs = [str(w.message) for w in rec]
    assert len(msgs) == 1
    assert str(n_bad) in msgs[0]


def test_clean_seed_does_not_warn(recwarn):
    g = _grid()
    lvl = LevelData(
        name="clean_water", version="2", path=Path("."), tilemap=g,
        tile_size_m=0.333, diffuse_path=Path("."),
        water_depth_q=_interior_depth(g, 1.0))
    GameMap(lvl)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_hot_reload_does_not_reflood_a_drained_tank():
    """THE anti-pattern pin (design §2.3): the seed lives in __init__ ONLY.
    reload_material_table() re-runs _update_caches (the atmosphere
    precedent) — it must NOT re-apply the water seed, or Ctrl+R would
    re-flood a tank the player drained."""
    g = _grid()
    lvl = LevelData(
        name="reflood_guard", version="2", path=Path("."), tilemap=g,
        tile_size_m=0.333, diffuse_path=Path("."),
        water_depth_q=_interior_depth(g, 1.0))
    gmap = GameMap(lvl)
    assert gmap.water_depth.any()                 # the seed applied at init
    gmap.water_depth[:] = 0                       # "the tank drained"
    gmap.reload_material_table()                  # config hot-reload path
    assert not gmap.water_depth.any(), (
        "config hot-reload re-flooded a drained tank — the water seed "
        "leaked into _update_caches")


def test_synthetic_leveldata_defaulted_tail_still_constructs():
    """Existing tests build LevelData without the new field (physics critic
    m3): the defaulted tail must keep that working, defaulting to None."""
    lvl = LevelData(name="t", version="1", path=Path("."),
                    tilemap=np.ones((4, 4), dtype=np.int32),
                    tile_size_m=0.333, diffuse_path=Path("."))
    assert lvl.water_depth_q is None


# ---------------------------------------------------------------------------
# Editor — solidity seam, fill, refusals, masking, writeback, undo ring
# ---------------------------------------------------------------------------

def test_water_solid_codes_are_sim_exact():
    """Critique M1: solid-for-water == permeability <= 0 from the CONFIG
    material table. Glass/hull are solid (no permeability key + occlude ->
    derived 0); furniture is 0.5 -> water flows past crates; air is open;
    SPACE_CODE is not a material id and never in the set."""
    codes = water_solid_codes()
    assert MAT_HULL in codes
    assert MAT_GLASS in codes                     # the aquarium wall
    assert MAT_AIR not in codes
    assert MAT_FURNITURE not in codes             # water stands on crates
    assert SPACE_CODE not in codes


def test_fill_bounded_by_glass_and_space_not_furniture():
    # 12x9: SPACE border, hull ring, interior air; a glass tank
    # (4x3 interior) and a furniture crate in the open room.
    g = scaffold_grid(12, 9)
    g[2:7, 5] = MAT_GLASS                        # tank west wall
    g[2:7, 10] = MAT_HULL                        # (hull ring is col 10)
    g[2, 5:10] = MAT_GLASS                       # tank north wall
    g[6, 5:10] = MAT_GLASS                       # tank south wall
    # tank interior: rows 3..5, cols 6..9 — bounded by glass + hull ring
    g[4, 3] = MAT_FURNITURE                      # a crate in the room
    solid = water_solid_codes()

    tank, why = water_fill_region(g, 7, 4, solid)
    assert why == "ok"
    assert tank == {(tx, ty) for ty in (3, 4, 5) for tx in (6, 7, 8, 9)}

    room, why = water_fill_region(g, 2, 2, solid)
    assert why == "ok"
    # The room fill never enters the tank, never crosses hull, never
    # reaches SPACE — but flows PAST (and onto) the furniture crate.
    assert tank.isdisjoint(room)
    assert (3, 4) in room                        # crate does not bound
    open_ = water_open_mask(g, solid)
    for tx, ty in room:
        assert open_[ty, tx]
    assert not any(g[ty, tx] == SPACE_CODE for tx, ty in room)


def test_fill_is_4_connected_no_diagonal_leak():
    """A diagonal gap must NOT leak the fill (matches the pipe model's
    4-face fluxes)."""
    g = scaffold_grid(8, 8)
    # A diagonal glass wall with a corner-touch pinch at (3,3)/(4,4).
    for i in range(2, 6):
        g[i, i] = MAT_GLASS
    solid = water_solid_codes()
    region, why = water_fill_region(g, 2, 3, solid)
    assert why == "ok"
    # Interior air is split by the diagonal: the north-east half is only
    # corner-adjacent, so it must be excluded... verify a cell clearly on
    # the far side of the diagonal is not in the region.
    assert (5, 2) not in region


def test_fill_refusals():
    g = _grid()
    solid = water_solid_codes()
    r, why = water_fill_region(g, 0, 0, solid)          # SPACE corner
    assert r is None and "SPACE" in why
    r, why = water_fill_region(g, 1, 1, solid)          # hull ring
    assert r is None and "solid" in why
    r, why = water_fill_region(g, -1, 3, solid)         # off-grid
    assert r is None and "outside" in why
    r, why = water_fill_region(g, GRID_W, 0, solid)     # off-grid east
    assert r is None and "outside" in why


def test_mask_water_to_open_wall_over_pool():
    """Critique M3: painting a wall over a pool then saving zeroes the
    covered cells and reports the count."""
    g = _grid()
    solid = water_solid_codes()
    depth = _interior_depth(g, 1.0)
    wet_before = int(np.count_nonzero(depth))
    # Paint a glass wall straight through the pool + one SPACE tile.
    g[2, 2:5] = MAT_GLASS
    g[3, 2] = SPACE_CODE
    masked, cleared = mask_water_to_open(depth, g, solid)
    covered = 4                                   # 3 glass + 1 space, all wet
    assert cleared == covered
    assert int(np.count_nonzero(masked)) == wet_before - covered
    assert not masked[2, 2:5].any() and masked[3, 2] == 0
    # Original grid untouched (masked is a copy).
    assert int(np.count_nonzero(depth)) == wet_before


def test_format_water_lines_schema():
    raw = tomllib.loads("".join(format_water_lines()))
    assert raw == {"water": {"depth_map": WATER_FILENAME}}


def test_write_water_managed_block_preserves_other_bytes(tmp_path):
    prefix = ("# hand comment stays\n"
              'version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n'
              "tile_size_m = 0.333\n\n")
    spawn_tbl = '[[spawn]]\nname = "Alpha"\nteam = 0\nx = 3\ny = 4\n\n'
    suffix = ('[bake]\ntileset = "x"\npx_per_tile = 8\nseed = 0\n')
    stale = ('[water]\ndepth_map = "old_name.npy"\n'
             '# doomed comment inside the managed table\n\n')
    d = tmp_path / "lvl"
    d.mkdir()
    toml = d / "level.toml"
    toml.write_text(prefix + stale + spawn_tbl + suffix,
                    encoding="utf-8", newline="\n")

    depth = np.zeros((4, 4), dtype=np.int32)
    depth[1:3, 1:3] = int(water_fixed.quantize(1.2))
    nbak, tbak, has_water = write_water(d, depth, toml_bak=False,
                                        npy_bak=True)
    assert has_water
    assert nbak is None                       # no pre-session npy to back up
    text = toml.read_text(encoding="utf-8")
    assert text.startswith(prefix)            # bytes outside preserved
    assert spawn_tbl in text
    assert text.endswith(suffix)
    assert "old_name" not in text             # stale block managed away
    raw = tomllib.loads(text)
    assert raw["water"] == {"depth_map": WATER_FILENAME}
    assert np.array_equal(np.load(d / WATER_FILENAME, allow_pickle=False),
                          depth)


def test_write_water_all_dry_removes_block_and_file(tmp_path):
    """Dormancy pin (design §2.5): a level saved dry carries NO [water]
    key and no stale .npy — the loader then returns None."""
    d = tmp_path / "lvl"
    d.mkdir()
    (d / "level.toml").write_text(
        'version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n',
        encoding="utf-8", newline="\n")
    depth = np.zeros((4, 4), dtype=np.int32)
    depth[2, 2] = int(water_fixed.quantize(1.0))
    write_water(d, depth, toml_bak=False, npy_bak=True)
    assert (d / WATER_FILENAME).is_file()
    assert "water" in tomllib.loads(
        (d / "level.toml").read_text(encoding="utf-8"))

    # Drain everything, save again: block gone, file gone, .bak kept the
    # pre-session bytes of the npy that existed before this save.
    pre_npy = (d / WATER_FILENAME).read_bytes()
    nbak, _, has_water = write_water(
        d, np.zeros((4, 4), dtype=np.int32), toml_bak=False, npy_bak=True)
    assert not has_water
    assert not (d / WATER_FILENAME).exists()
    assert nbak is not None and nbak.read_bytes() == pre_npy
    assert "water" not in tomllib.loads(
        (d / "level.toml").read_text(encoding="utf-8"))


def test_write_water_npy_bak_once_per_session(tmp_path):
    """The npy .bak carries PRE-SESSION bytes: first save of the session
    backs up the on-disk file; later saves (npy_bak=False) never touch it
    — the same contract as the toml .bak, on water_init.npy's own file."""
    d = tmp_path / "lvl"
    d.mkdir()
    (d / "level.toml").write_text('version = "2"\n', encoding="utf-8",
                                  newline="\n")
    pre = np.full((3, 3), int(water_fixed.quantize(0.4)), dtype=np.int32)
    np.save(d / WATER_FILENAME, pre)              # pre-session state
    pre_bytes = (d / WATER_FILENAME).read_bytes()

    first = np.full((3, 3), int(water_fixed.quantize(0.7)), dtype=np.int32)
    nbak, _, _ = write_water(d, first, toml_bak=False, npy_bak=True)
    assert nbak is not None and nbak.read_bytes() == pre_bytes

    second = np.full((3, 3), int(water_fixed.quantize(0.9)), dtype=np.int32)
    nbak2, _, _ = write_water(d, second, toml_bak=False, npy_bak=False)
    assert nbak2 is None
    assert (Path(str(d / WATER_FILENAME) + ".bak").read_bytes()
            == pre_bytes)                         # still pre-session


def test_water_undo_ring_snapshots_are_independent():
    """Critique M4 (the ring itself): UndoRing carries the water grid as
    independent numpy snapshots — mutating the live grid after push never
    corrupts the popped state. (Mode-scoping of Ctrl+Z is key-dispatch
    structure in run_editor, smoke-covered by --auto like SPAWN/LIGHT.)"""
    ring = UndoRing()
    water = np.zeros((5, 5), dtype=np.int32)
    water[2, 2] = int(water_fixed.quantize(1.0))
    ring.push(water)
    water[...] = 0                                # live grid drains
    snap = ring.pop()
    assert snap is not None
    assert int(snap[2, 2]) == int(water_fixed.quantize(1.0))
    assert ring.pop() is None                     # LIFO exhausted


# ---------------------------------------------------------------------------
# The committed aquarium demo — loadable, wet, in the safe depth band
# ---------------------------------------------------------------------------

def test_aquarium_demo_level_loads_wet_and_safe():
    """The P5 HUMAN-TEST fixture: levels/aquarium_demo carries a
    glass-bounded 4x3 tank at 1.2 m (SAFE: < 1.7 m, doc §3 drain
    asymmetry), spawns present, and the seed lands strictly inside the
    GameMap mask (glass-bounded => at-rest, tested bit-exactly in
    test_level_water_physics.py)."""
    lvl = load("aquarium_demo")
    assert lvl.version == "2"
    assert lvl.water_depth_q is not None
    depth_q = lvl.water_depth_q
    wet = depth_q > 0
    assert int(wet.sum()) == 12                   # 4x3 tank interior
    q12 = int(water_fixed.quantize(1.2))
    assert (depth_q[wet] == q12).all()            # flat seed, 1.2 m
    assert float(water_fixed.dequantize(depth_q.max())) < 1.7
    assert len(lvl.spawns) >= 1                   # spawns present
    gmap = GameMap(lvl)
    assert np.array_equal(gmap.water_depth, depth_q)   # clean seed, no mask
    # Glass-bounded: every 4-neighbour of a wet tile is wet or solid —
    # the precondition of the at-rest byte-identity property (never assert
    # UNCHANGED for open-edge pools).
    ys, xs = np.where(wet)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            assert wet[ny, nx] or gmap.solid[ny, nx], (
                f"wet tile ({y},{x}) has an open dry neighbour ({ny},{nx})"
                f" — the demo tank is not solid-bounded")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
