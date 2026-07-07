"""Map editor — tools/map_editor.py (engine/15 §5, P3).

Pins the pure helpers the interactive tool is built from (raylib has no
input-injection API, so the pyray loop is exercised by ``--auto`` and these
units carry the behaviour):

  - palette generation from MATERIAL_NAMES + SPACE_CODE at call time (§1
    rule: a monkeypatched new material appears with no tool change).
  - NEW-level scaffold (space border / hull ring / AIR interior), loadable
    by level_loader AND bakeable through the P2 baker's public API.
  - ROOM geometry: perimeter + AIR interior, shared-wall overlap (existing
    wall-family tiles are kept, never doubled; doors in a shared run
    survive).
  - CORRIDOR: swath of width w + wall lining on every 8-neighbour bordering
    non-AIR — airtight against SPACE even on diagonal-ish drags; open where
    it joins existing AIR; existing walls shared.
  - DOOR: straight-run acceptance, corner/T/end/isolated/non-wall refusals
    with status-line reasons.
  - dirty-rect +1 expansion (the edge16 neighbour contract): the expanded
    re-bake patch composited over the pre-stroke image equals a full
    re-bake; an unexpanded patch does NOT.
  - spawn writeback: managed [[spawn]] block, bytes outside it preserved,
    loader round-trip (load -> edit -> save -> reload), CRLF kept, .bak =
    pre-first-write bytes.
  - undo ring integration for ROOM/CORRIDOR ops (LIFO restore).

Run:
    python -m pytest tests/test_map_editor_tool.py -q
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import level_loader  # noqa: E402
from level_loader import (SPACE_CODE, SpawnEntry,  # noqa: E402
                          materials_from_tilemap)
from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_FURNITURE,  # noqa: E402
                                  MAT_GLASS, MAT_HULL, MAT_STEEL, MAT_WOOD,
                                  MATERIAL_NAMES)
from make_tileset import build_tileset  # noqa: E402
from bake_level_art import (bake_full, bake_region,  # noqa: E402
                            load_tileset)
from level_edit_common import UndoRing, build_palette  # noqa: E402
from map_editor import (REBAKE_MARGIN, SCAFFOLD_MIN,  # noqa: E402
                        apply_corridor, apply_room, choose_preview_ppt,
                        corridor_cells, create_level, diff_rect, door_check,
                        expand_dirty_rect, normalize_rect, parse_size,
                        scaffold_grid, spawn_at, unique_spawn_name,
                        wall_family_codes, write_spawns)

TEST_PX = 16          # tmp tileset resolution — fast (P2 test convention)

# The wall family as greybox declares it ([groups] wall) — tests pass it
# explicitly so the geometry pins don't depend on tileset IO.
WALL_CODES = frozenset({MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_STEEL, MAT_GLASS})

_NEIGH8 = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if dx or dy)


@pytest.fixture(scope="session")
def tileset16_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("greybox16")
    build_tileset(out, px=TEST_PX, seed=0)
    return out


@pytest.fixture(scope="session")
def ts16(tileset16_dir):
    return load_tileset(tileset16_dir)


# ---------------------------------------------------------------------------
# Palette — generated from the material table (§1 rule)
# ---------------------------------------------------------------------------

def test_build_palette_matches_material_table():
    pal = build_palette()
    assert set(pal) == set(MATERIAL_NAMES) | {SPACE_CODE}
    for mid, name in MATERIAL_NAMES.items():
        assert pal[mid][0] == name.upper()
    assert pal[SPACE_CODE][0] == "SPACE"
    # AIR is the absence of an overlay; everything else has an RGB fill.
    assert pal[MAT_AIR][1] is None
    for mid in (set(MATERIAL_NAMES) - {MAT_AIR}) | {SPACE_CODE}:
        rgb = pal[mid][1]
        assert len(rgb) == 3 and all(0 <= v <= 255 for v in rgb)


def test_build_palette_picks_up_new_materials(monkeypatch):
    """A material added to MATERIAL_NAMES (one config row) appears in the
    palette on the next call — no tool change (engine/15 §1)."""
    free_id = max(MATERIAL_NAMES) + 1
    assert free_id != SPACE_CODE                    # 9 stays reserved
    monkeypatch.setitem(MATERIAL_NAMES, free_id, "carpet")
    pal = build_palette()
    assert set(pal) == set(MATERIAL_NAMES) | {SPACE_CODE}
    assert pal[free_id][0] == "CARPET"
    rgb = pal[free_id][1]
    assert rgb is not None and len(rgb) == 3
    # The generated fallback colour reads apart from every curated colour.
    others = [c for pid, (_, c) in pal.items()
              if pid != free_id and c is not None]
    assert rgb not in others


# ---------------------------------------------------------------------------
# NEW — size parsing, scaffold geometry, loadable + bakeable folder
# ---------------------------------------------------------------------------

def test_parse_size():
    assert parse_size("48x32") == (48, 32)
    assert parse_size(" 12X9 ") == (12, 9)
    for bad in ("48", "x32", "48x", "48*32", "12x9x4", "0x0", "4x9"):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_scaffold_grid_geometry():
    g = scaffold_grid(8, 6)
    assert g.shape == (6, 8)
    # 1-tile SPACE border...
    assert (g[0, :] == SPACE_CODE).all() and (g[-1, :] == SPACE_CODE).all()
    assert (g[:, 0] == SPACE_CODE).all() and (g[:, -1] == SPACE_CODE).all()
    # ...MAT_HULL ring inside it...
    assert (g[1, 1:-1] == MAT_HULL).all() and (g[-2, 1:-1] == MAT_HULL).all()
    assert (g[1:-1, 1] == MAT_HULL).all() and (g[1:-1, -2] == MAT_HULL).all()
    # ...AIR interior; the whole grid speaks canon v2 codes.
    assert (g[2:-2, 2:-2] == MAT_AIR).all()
    materials_from_tilemap(g, "2")
    with pytest.raises(ValueError):
        scaffold_grid(SCAFFOLD_MIN - 1, 8)


def test_create_level_scaffold_loads_and_bakes(tileset16_dir, ts16,
                                               tmp_path):
    d = tmp_path / "fresh"
    summary = create_level(d, 12, 9, name="Fresh", tileset=tileset16_dir,
                           px_per_tile=8, seed=2)
    assert summary["size_tiles"] == (12, 9)
    # Loadable: level_loader requires art — create_level bakes immediately.
    lvl = level_loader.load(str(d))
    assert lvl.version == "2" and lvl.name == "Fresh"
    assert np.array_equal(lvl.tilemap, scaffold_grid(12, 9))
    materials_from_tilemap(lvl.tilemap, lvl.version)
    assert lvl.diffuse_path.is_file() and lvl.normal_path.is_file()
    assert lvl.art_px_per_tile == (8.0, 8.0)
    bake = lvl.raw_toml["bake"]
    assert (bake["px_per_tile"], bake["seed"]) == (8, 2)
    # Bakeable through the P2 public API (the editor's preview seam).
    patch = bake_full(lvl.tilemap, ts16, px_per_tile=8, seed=2)
    assert patch.diffuse.shape == (9 * 8, 12 * 8, 4)
    # Refuses to clobber an existing level; fresh folders carry no .bak.
    with pytest.raises(ValueError, match="already contains"):
        create_level(d, 12, 9, tileset=tileset16_dir)
    assert not (d / "level.toml.bak").exists()


# ---------------------------------------------------------------------------
# ROOM — perimeter + interior, shared walls
# ---------------------------------------------------------------------------

def test_normalize_rect_orders_and_clamps():
    assert normalize_rect(5, 7, 2, 3, 10, 10) == (2, 3, 5, 7)
    assert normalize_rect(-3, -2, 4, 5, 10, 8) == (0, 0, 4, 5)
    assert normalize_rect(8, 2, 99, 99, 10, 8) == (8, 2, 9, 7)
    assert normalize_rect(-5, -5, -1, -1, 10, 8) is None


def test_apply_room_perimeter_and_interior():
    g = np.full((10, 12), MAT_AIR, dtype=np.int32)
    g[4, 5] = MAT_FURNITURE                    # crate inside -> cleared
    changed = apply_room(g, (2, 2, 8, 7), MAT_WOOD, WALL_CODES)
    # Perimeter is wood, interior is AIR, outside untouched.
    for ty in range(2, 8):
        for tx in range(2, 9):
            want = (MAT_WOOD if tx in (2, 8) or ty in (2, 7) else MAT_AIR)
            assert int(g[ty, tx]) == want, (tx, ty)
    assert int(np.count_nonzero(g == MAT_WOOD)) == 22   # 2*7 + 2*6 - 4
    assert changed == 22 + 1                            # walls + the crate
    outside = g.copy()
    outside[2:8, 2:9] = MAT_AIR
    assert int(np.count_nonzero(outside)) == 0


def test_apply_room_shares_existing_walls():
    """A room dragged so its perimeter lands ON an existing wall reuses that
    wall (kept, material unchanged) — no doubling, and a door in the shared
    run survives."""
    g = np.full((10, 12), MAT_AIR, dtype=np.int32)
    g[2:8, 5] = MAT_HULL                       # existing bulkhead at x=5
    g[4, 5] = MAT_DOOR                         # with a door in it
    apply_room(g, (1, 2, 5, 7), MAT_WOOD, WALL_CODES)
    assert (g[2:8, 5][np.arange(6) != 2] == MAT_HULL).all()  # kept hull
    assert int(g[4, 5]) == MAT_DOOR                          # door survives
    assert (g[2:8, 1] == MAT_WOOD).all()       # the room's own wall is wood
    assert (g[2:8, 6] == MAT_AIR).all()        # nothing doubled next door
    # Interior strictly inside got carved to AIR.
    assert (g[3:7, 2:5] == MAT_AIR).all()


def test_apply_room_thin_rect_is_all_wall():
    g = np.full((8, 10), MAT_AIR, dtype=np.int32)
    changed = apply_room(g, (2, 3, 6, 3), MAT_STEEL, WALL_CODES)
    assert changed == 5
    assert (g[3, 2:7] == MAT_STEEL).all()
    assert int(np.count_nonzero(g)) == 5


# ---------------------------------------------------------------------------
# CORRIDOR — swath + lining, diagonal-ish airtightness, open joins
# ---------------------------------------------------------------------------

def test_corridor_cells_swath_width():
    cells = corridor_cells(2, 5, 8, 5, 3, 20, 12)
    assert cells == {(x, y) for x in range(1, 10) for y in range(4, 7)}
    # Width 1 degenerates to the drag line itself.
    assert corridor_cells(2, 5, 4, 5, 1, 20, 12) == {(2, 5), (3, 5), (4, 5)}


def test_apply_corridor_through_space_is_lined_and_airtight():
    g = np.full((12, 20), SPACE_CODE, dtype=np.int32)
    changed = apply_corridor(g, 3, 5, 15, 5, width=3,
                             wall_id=MAT_HULL, wall_codes=WALL_CODES)
    floor = corridor_cells(3, 5, 15, 5, 3, 20, 12)
    assert changed == len(floor) + int(np.count_nonzero(g == MAT_HULL))
    for cx, cy in floor:
        assert int(g[cy, cx]) == MAT_AIR
    # Airtight: every 8-neighbour of every floor cell is AIR or the lining.
    for cx, cy in floor:
        for dx, dy in _NEIGH8:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 20 and 0 <= ny < 12:
                assert int(g[ny, nx]) in (MAT_AIR, MAT_HULL), (nx, ny)


def test_apply_corridor_diagonalish_no_pinholes():
    """The diagonal drag is the case 4-neighbour lining would leak: a floor
    cell touching SPACE corner-to-corner. 8-neighbour lining closes it."""
    g = np.full((16, 16), SPACE_CODE, dtype=np.int32)
    apply_corridor(g, 2, 2, 12, 9, width=3,
                   wall_id=MAT_STEEL, wall_codes=WALL_CODES)
    air = np.argwhere(g == MAT_AIR)
    assert len(air) > 0
    for ty, tx in air.tolist():
        for dx, dy in _NEIGH8:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < 16 and 0 <= ny < 16:
                assert int(g[ny, nx]) != SPACE_CODE, (tx, ty, nx, ny)


def test_apply_corridor_opens_into_rooms_and_shares_walls():
    g = np.full((12, 20), SPACE_CODE, dtype=np.int32)
    g[2:10, 12:19] = MAT_HULL                  # a hull room to the east
    g[3:9, 13:18] = MAT_AIR
    apply_corridor(g, 4, 5, 14, 5, width=3,
                   wall_id=MAT_WOOD, wall_codes=WALL_CODES)
    # The swath punched straight through the room's west wall.
    assert (g[4:7, 12] == MAT_AIR).all()
    # Where the swath borders the room's own hull, the hull is SHARED.
    assert int(g[3, 12]) == MAT_HULL and int(g[7, 12]) == MAT_HULL
    # Out in space the lining is the selected wood.
    assert int(g[3, 6]) == MAT_WOOD and int(g[7, 6]) == MAT_WOOD
    # Inside the room the swath meets AIR: the join stays OPEN.
    assert (g[3, 13:18] == MAT_AIR).all()
    assert (g[7, 13:18] == MAT_AIR).all()


# ---------------------------------------------------------------------------
# DOOR — straight-run acceptance, everything else refused with a reason
# ---------------------------------------------------------------------------

def test_door_check_accepts_straight_runs():
    g = np.full((7, 7), MAT_AIR, dtype=np.int32)
    g[3, 1:6] = MAT_HULL                       # horizontal run
    ok, why = door_check(g, 3, 3, WALL_CODES)
    assert ok and "horizontal" in why
    g = np.full((7, 7), MAT_AIR, dtype=np.int32)
    g[1:6, 3] = MAT_STEEL                      # vertical run
    ok, why = door_check(g, 3, 3, WALL_CODES)
    assert ok and "vertical" in why
    # Group connectivity: a glass pane mid-run counts its hull neighbours.
    g = np.full((7, 7), MAT_AIR, dtype=np.int32)
    g[3, 1:6] = MAT_HULL
    g[3, 3] = MAT_GLASS
    assert door_check(g, 3, 3, WALL_CODES)[0]


def test_door_check_refusals():
    g = np.full((7, 7), MAT_AIR, dtype=np.int32)
    assert door_check(g, -1, 0, WALL_CODES) == (False, "outside the grid")
    assert door_check(g, 3, 3, WALL_CODES) == (False, "not a wall tile")
    g[3, 3] = SPACE_CODE
    assert door_check(g, 3, 3, WALL_CODES) == (False, "not a wall tile")
    g[3, 3] = MAT_HULL                         # isolated pillar
    ok, why = door_check(g, 3, 3, WALL_CODES)
    assert not ok and "isolated" in why
    g[3, 4] = MAT_HULL                         # (4, 3): E neighbour -> end
    ok, why = door_check(g, 3, 3, WALL_CODES)
    assert not ok and "end" in why
    g[4, 3] = MAT_HULL                         # + S neighbour -> corner
    ok, why = door_check(g, 3, 3, WALL_CODES)
    assert not ok and "corner" in why
    g[2, 3] = MAT_HULL                         # + N neighbour -> T junction
    ok, why = door_check(g, 3, 3, WALL_CODES)
    assert not ok and "corner" in why
    # An existing door is refused (nothing to do).
    g = np.full((7, 7), MAT_AIR, dtype=np.int32)
    g[3, 1:6] = MAT_HULL
    g[3, 3] = MAT_DOOR
    ok, why = door_check(g, 3, 3, WALL_CODES)
    assert not ok and "already" in why


# ---------------------------------------------------------------------------
# Dirty rect — the +1 expansion (edge16 neighbour contract)
# ---------------------------------------------------------------------------

def test_expand_dirty_rect_plus_one_all_sides():
    assert REBAKE_MARGIN == 1                  # the P2 contract — tripwire
    assert expand_dirty_rect((3, 4, 2, 2), 20, 20) == (2, 3, 4, 4)
    assert expand_dirty_rect((0, 0, 1, 1), 20, 20) == (0, 0, 2, 2)
    assert expand_dirty_rect((19, 19, 1, 1), 20, 20) == (18, 18, 2, 2)
    assert expand_dirty_rect((0, 0, 20, 20), 20, 20) == (0, 0, 20, 20)


def test_dirty_rect_rebake_equals_full_rebake(ts16):
    """The reason for the +1: breaking a wall run changes the NEIGHBOURS'
    edge16 pieces. Compositing the +1-expanded re-bake patch over the
    pre-stroke image reproduces a full re-bake exactly; the unexpanded
    (changed-cells-only) patch does not."""
    g = np.full((7, 7), MAT_AIR, dtype=np.int32)
    g[3, 1:6] = MAT_HULL
    before = bake_full(g, ts16, px_per_tile=8, seed=0)
    g2 = g.copy()
    g2[3, 3] = MAT_AIR                         # the stroke: break the run
    after = bake_full(g2, ts16, px_per_tile=8, seed=0)

    rect = expand_dirty_rect((3, 3, 1, 1), 7, 7)
    assert rect == (2, 2, 3, 3)
    patch = bake_region(g2, ts16, rect, px_per_tile=8, seed=0)
    img = before.diffuse.copy()
    x0, y0, tw, th = patch.rect
    img[y0 * 8:(y0 + th) * 8, x0 * 8:(x0 + tw) * 8] = patch.diffuse
    assert np.array_equal(img, after.diffuse)

    naive = bake_region(g2, ts16, (3, 3, 1, 1), px_per_tile=8, seed=0)
    img0 = before.diffuse.copy()
    img0[3 * 8:4 * 8, 3 * 8:4 * 8] = naive.diffuse
    assert not np.array_equal(img0, after.diffuse)


def test_diff_rect_bounds_undo_rebakes():
    a = np.zeros((6, 8), dtype=np.int32)
    b = a.copy()
    assert diff_rect(a, b) is None
    b[2, 3] = 1
    b[4, 6] = 2
    assert diff_rect(a, b) == (3, 2, 4, 3)


def test_choose_preview_ppt_divisor_and_texture_cap():
    assert choose_preview_ppt(128, 64, 100, 100) == 64
    assert choose_preview_ppt(16, 8, 10, 10) == 8
    # A 1000-tile-wide map cannot preview at 32 px (32000 > 16384) -> 16.
    assert choose_preview_ppt(128, 64, 1000, 200) == 16
    # Nothing fits: fall back to the coarsest divisor instead of crashing.
    assert choose_preview_ppt(128, 64, 20000, 10) == 1


# ---------------------------------------------------------------------------
# Tileset seam — the wall family comes from the manifest, not code
# ---------------------------------------------------------------------------

def test_wall_family_codes_from_manifest(ts16):
    assert wall_family_codes(ts16) == WALL_CODES


# ---------------------------------------------------------------------------
# SPAWN — hit test, auto-names, managed [[spawn]] writeback
# ---------------------------------------------------------------------------

def test_spawn_hit_test_and_auto_names():
    spawns = [SpawnEntry("marine_1", 0, 2.0, 3.0),          # x 2..5, y 3..6
              SpawnEntry("zombie_1", 1, 4.0, 4.0)]
    assert spawn_at(spawns, 2.5, 3.5) == 0
    assert spawn_at(spawns, 4.5, 4.5) == 1     # overlap -> topmost (last)
    assert spawn_at(spawns, 0.5, 0.5) is None
    assert unique_spawn_name(spawns, 0) == "marine_2"
    assert unique_spawn_name(spawns, 1) == "zombie_2"
    assert unique_spawn_name([], 1) == "zombie_1"


def test_write_spawns_managed_block_preserves_other_bytes(tmp_path):
    prefix = ("# hand comment stays\n"
              'version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n'
              "tile_size_m = 0.333\n\n")
    suffix = ('[art.bare]\n# baked by the P2 baker\n'
              'diffuse = "baked_diffuse.png"\n\n'
              '[bake]\ntileset = "x"\npx_per_tile = 8\nseed = 0\n')
    body = (prefix
            + '[[spawn]]\nname = "Alpha"\nteam = 0\nx = 3\ny = 4\n\n'
            + '[[spawn]]\n# doomed comment inside a managed table\n'
              'name = "Zed"\nteam = 1\nx = 8.5\ny = 2\nfootprint = 5\n\n'
            + suffix)
    toml = tmp_path / "level.toml"
    toml.write_text(body, encoding="utf-8", newline="\n")
    original = toml.read_bytes()

    spawns = [SpawnEntry("Alpha", 0, 3.0, 4.0),
              SpawnEntry("Bravo", 1, 10.5, 2.0, footprint=4)]
    bak = write_spawns(toml, spawns, write_bak=True)
    assert bak.read_bytes() == original
    text = toml.read_text(encoding="utf-8")
    # Everything OUTSIDE the spawn tables is byte-preserved.
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    raw = tomllib.loads(text)
    assert raw["spawn"] == [
        {"name": "Alpha", "team": 0, "x": 3.0, "y": 4.0, "footprint": 3},
        {"name": "Bravo", "team": 1, "x": 10.5, "y": 2.0, "footprint": 4}]
    assert raw["bake"]["px_per_tile"] == 8     # untouched


def test_write_spawns_appends_when_file_has_none_and_deletes_all(tmp_path):
    body = 'version = "2"\nname = "T"\ntilemap = "t.csv"\n'
    toml = tmp_path / "level.toml"
    toml.write_text(body, encoding="utf-8", newline="\n")
    write_spawns(toml, [SpawnEntry("A", 0, 1.0, 2.0)], write_bak=False)
    text = toml.read_text(encoding="utf-8")
    assert text.startswith(body)
    assert tomllib.loads(text)["spawn"][0]["name"] == "A"
    # Deleting every spawn removes the whole managed block again.
    write_spawns(toml, [], write_bak=False)
    raw = tomllib.loads(toml.read_text(encoding="utf-8"))
    assert "spawn" not in raw


def test_write_spawns_preserves_crlf(tmp_path):
    toml = tmp_path / "level.toml"
    toml.write_bytes(b'version = "2"\r\nname = "T"\r\n')
    write_spawns(toml, [SpawnEntry("A", 0, 1.0, 2.0)], write_bak=False)
    data = toml.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n")
    assert tomllib.loads(data.decode())["spawn"][0]["name"] == "A"


def test_spawn_round_trip_through_the_loader(tileset16_dir, tmp_path):
    """load -> edit -> save -> reload: SpawnEntry-exact, unrelated toml
    content untouched, .bak = pre-FIRST-write bytes (once per session)."""
    d = tmp_path / "rt"
    create_level(d, 10, 8, tileset=tileset16_dir, px_per_tile=8, seed=0)
    lvl = level_loader.load(str(d))
    assert lvl.spawns == []
    pre = (d / "level.toml").read_bytes()

    spawns = [SpawnEntry("marine_1", 0, 2.0, 3.0),
              SpawnEntry("zombie_1", 1, 6.0, 4.0, footprint=5)]
    write_spawns(d / "level.toml", spawns)               # first save: .bak
    lvl2 = level_loader.load(str(d))
    assert lvl2.spawns == spawns

    edited = [replace(spawns[1], x=7.0, team=0)]         # move + reteam
    write_spawns(d / "level.toml", edited, write_bak=False)
    lvl3 = level_loader.load(str(d))
    assert lvl3.spawns == edited
    assert (d / "level.toml.bak").read_bytes() == pre
    assert lvl3.raw_toml["bake"] == lvl.raw_toml["bake"]
    assert lvl3.raw_toml["name"] == lvl.raw_toml["name"]
    assert lvl3.raw_toml["tile_size_m"] == lvl.raw_toml["tile_size_m"]


# ---------------------------------------------------------------------------
# Undo ring integration — ROOM / CORRIDOR ops rewind LIFO
# ---------------------------------------------------------------------------

def test_undo_ring_restores_room_and_corridor_ops():
    g = scaffold_grid(18, 14)
    base = g.copy()
    ring = UndoRing()

    snap = g.copy()
    assert apply_room(g, (4, 4, 10, 9), MAT_WOOD, WALL_CODES) > 0
    ring.push(snap)
    after_room = g.copy()

    snap = g.copy()
    assert apply_corridor(g, 6, 6, 14, 11, width=3,
                          wall_id=MAT_STEEL, wall_codes=WALL_CODES) > 0
    ring.push(snap)
    assert not np.array_equal(g, after_room)

    g[...] = ring.pop()                        # corridor rewinds first
    assert np.array_equal(g, after_room)
    g[...] = ring.pop()                        # then the room
    assert np.array_equal(g, base)
    assert ring.pop() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
