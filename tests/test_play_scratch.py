"""tools/play_scratch.py — play-from-editor (F5), editor doc §8 (Arc C7).

Pins:
  - write_scratch_level populates a COMPLETE, loadable scratch level:
    tilemap + spawns + entities + wires + zones + air + water + boundary +
    [bake] blocks all present and round-trip through level_loader.load;
  - unsaved edits (in-memory state that differs from the source level_dir's
    own committed files) are what lands in the scratch dir, not whatever is
    on disk;
  - bake reuse: a CLEAN grid copies the source's existing baked PNGs
    byte-for-byte (no re-bake); a DIRTY grid always re-bakes from the
    freshly-written scratch tilemap — a stale bake never ships;
  - build_launch_argv defaults to sys.executable (never bare python);
  - cleanup_scratch_dir is idempotent and best-effort;
  - the loader accepts the real "_editor_scratch/<name>" path form and
    main.py's --level parsing rejects '..'/absolute escapes.

Run:
    python -m pytest tests/test_play_scratch.py -q
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_loader import EntityInstance, SpawnEntry, WireSpec  # noqa: E402
from level_loader import SPACE_CODE  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL  # noqa: E402
from make_tileset import build_tileset  # noqa: E402
from bake_level_art import bake_full, bake_level, load_tileset  # noqa: E402
import door_entity_port as dep  # noqa: E402

import play_scratch  # noqa: E402

TEST_PX = 16              # tiny/fast test tileset resolution
TILE_SIZE_M = 0.333       # the shipped default -> tiles_per_m == 3

SP, AI, HU, FU = SPACE_CODE, MAT_AIR, MAT_HULL, MAT_FURNITURE

SMALL_MAP = np.array([
    [SP, SP, SP, SP, SP, SP],
    [SP, HU, HU, HU, HU, SP],
    [SP, HU, AI, AI, HU, SP],
    [SP, HU, AI, AI, HU, SP],
    [SP, HU, HU, HU, HU, SP],
    [SP, SP, SP, SP, SP, SP],
], dtype=np.int32)

# One interior AIR tile swapped to furniture — a genuinely different bake
# (used to prove the "dirty" path re-bakes from the NEW grid, not a stale
# copy of the source's PNGs).
DIRTY_MAP = SMALL_MAP.copy()
DIRTY_MAP[2, 2] = FU


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tileset_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("play_scratch_tileset")
    build_tileset(out, px=TEST_PX, seed=0)
    return out


def _write_src_level(tmp_path: Path, tileset_dir: Path, *, name: str = "src",
                     tilemap: np.ndarray = SMALL_MAP) -> Path:
    """A minimal, ALREADY-BAKED source level folder (stands in for the
    currently-open real level_dir) — never under the repo's levels/."""
    d = tmp_path / name
    d.mkdir()
    np.savetxt(d / "tilemap.csv", tilemap, fmt="%d", delimiter=",")
    (d / "level.toml").write_text(
        f'version = "2"\nname = "{name}"\ntilemap = "tilemap.csv"\n'
        f'tile_size_m = {TILE_SIZE_M}\n', encoding="utf-8")
    bake_level(d, tileset=str(tileset_dir), px_per_tile=TEST_PX, seed=0,
              write_bak=False)
    return d


def _cleanup(name: str) -> None:
    play_scratch.cleanup_scratch_dir(play_scratch.scratch_dir_for(name))


# ---------------------------------------------------------------------------
# write_scratch_level — full population + round-trip
# ---------------------------------------------------------------------------

def test_write_scratch_level_full_round_trip(tmp_path, tileset_dir):
    src = _write_src_level(tmp_path, tileset_dir)
    name = "c7_full_rt"
    door = dep.build_door_instance(
        1, 2, "h", dep.length_m_for_tiles(1, TILE_SIZE_M), "closed", "door_1")
    sensor = EntityInstance(id="sensor_1", class_name="pressure", ordinal=0,
                            tags=(), fields={"x": 4, "y": 2},
                            authored_keys=("x", "y"))
    # Claims zone_id 1 (below) so the paint isn't orphaned (level_loader's
    # own §5 binding validator would otherwise WARN — not an error, but this
    # keeps the round-trip clean).
    zone = EntityInstance(id="site_1", class_name="breach_site", ordinal=0,
                          tags=(), fields={"zone_id": 1}, authored_keys=(
                              "zone_id",))
    entities = [door, sensor, zone]
    wires = [WireSpec(from_="sensor_1.value", to="door_1.close")]
    spawns = [SpawnEntry(name="m1", team=0, x=2.5, y=2.5, footprint=1)]
    zones = np.zeros(SMALL_MAP.shape, dtype=np.uint8)
    zones[2, 2] = 1
    air = np.full(SMALL_MAP.shape, 65536, dtype=np.int32)  # FP_ONE = 1.0 atm
    water = np.zeros(SMALL_MAP.shape, dtype=np.int32)
    water[2, 3] = 6553  # ~0.1 m, an OPEN interior tile

    try:
        dest = play_scratch.write_scratch_level(
            name, level_dir=src, grid=SMALL_MAP, water_masked=water,
            zones=zones, air=air, level_boundary="space", spawns=spawns,
            lights=[], entities=entities, wires=wires, light_form="entity",
            bake_clean=True, tileset_arg=str(tileset_dir), bake_ppt=TEST_PX,
            bake_seed=0)

        assert dest == play_scratch.scratch_dir_for(name)
        for fname in ("tilemap.csv", "level.toml", "baked_diffuse.png",
                     "baked_normal.png", "water_init.npy", "zones.npy",
                     "air_init.npy"):
            assert (dest / fname).is_file(), f"missing {fname}"

        # The REAL acceptance path: level_loader.load resolves
        # "_editor_scratch/<name>" under the repo's own levels/ (default
        # levels_dir) — not a mocked path.
        lvl = level_loader.load(play_scratch.scratch_level_arg(name))
        assert np.array_equal(lvl.tilemap, SMALL_MAP)
        assert lvl.boundary == "space"
        assert [s.name for s in lvl.spawns] == ["m1"]
        assert {e.id for e in lvl.entities} == {"door_1", "sensor_1", "site_1"}
        assert [(w.from_, w.to) for w in lvl.wire_specs] == [
            ("sensor_1.value", "door_1.close")]
        assert lvl.zone_grid is not None and lvl.zone_grid[2, 2] == 1
        assert lvl.air_init_q is not None
        assert lvl.water_depth_q is not None and lvl.water_depth_q[2, 3] > 0
    finally:
        _cleanup(name)
    assert not play_scratch.scratch_dir_for(name).exists()


def test_write_scratch_level_includes_unsaved_edits(tmp_path, tileset_dir):
    """The scratch dir reflects the LIVE in-memory state, not whatever the
    source level_dir has committed to disk (which here has none of this)."""
    src = _write_src_level(tmp_path, tileset_dir)
    name = "c7_unsaved"
    spawns = [SpawnEntry(name="fresh", team=1, x=3.0, y=3.0, footprint=1)]
    try:
        play_scratch.write_scratch_level(
            name, level_dir=src, grid=SMALL_MAP,
            water_masked=np.zeros(SMALL_MAP.shape, dtype=np.int32),
            zones=None, air=None, level_boundary="space", spawns=spawns,
            lights=[], entities=[], wires=[], light_form="entity",
            bake_clean=True, tileset_arg=str(tileset_dir), bake_ppt=TEST_PX,
            bake_seed=0)
        lvl = level_loader.load(play_scratch.scratch_level_arg(name))
        assert [s.name for s in lvl.spawns] == ["fresh"]
        # dormant grids never allocated this session stay absent, not
        # materialized as empty files.
        assert lvl.zone_grid is None
        assert lvl.air_init_q is None
    finally:
        _cleanup(name)


def test_write_scratch_level_writes_the_live_boundary(tmp_path, tileset_dir):
    """boundary is always written to the CURRENT live value (never the
    source level_dir's on-disk value) — level_lib.write_boundary_field's own
    unit tests (test_air_boundary.py) cover the writer itself; this pins
    play_scratch actually calling it with the live value."""
    import warnings
    src = _write_src_level(tmp_path, tileset_dir)
    name = "c7_boundary"
    try:
        play_scratch.write_scratch_level(
            name, level_dir=src, grid=SMALL_MAP,
            water_masked=np.zeros(SMALL_MAP.shape, dtype=np.int32),
            zones=None, air=None, level_boundary="ambient", spawns=[],
            lights=[], entities=[], wires=[], light_form="entity",
            bake_clean=True, tileset_arg=str(tileset_dir), bake_ppt=TEST_PX,
            bake_seed=0)
        with warnings.catch_warnings():
            # A tiny test map trips the (unrelated) ambient sponge-width
            # authoring warning — expected on a map this small, not a defect.
            warnings.simplefilter("ignore")
            lvl = level_loader.load(play_scratch.scratch_level_arg(name))
        assert lvl.boundary == "ambient"
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Bake reuse — clean copies, dirty re-bakes
# ---------------------------------------------------------------------------

def test_bake_clean_copies_existing_pngs_byte_for_byte(tmp_path, tileset_dir):
    src = _write_src_level(tmp_path, tileset_dir)
    name = "c7_clean_bake"
    try:
        dest = play_scratch.write_scratch_level(
            name, level_dir=src, grid=SMALL_MAP,
            water_masked=np.zeros(SMALL_MAP.shape, dtype=np.int32),
            zones=None, air=None, level_boundary="space", spawns=[],
            lights=[], entities=[], wires=[], light_form="entity",
            bake_clean=True, tileset_arg=str(tileset_dir), bake_ppt=TEST_PX,
            bake_seed=0)
        assert ((dest / "baked_diffuse.png").read_bytes()
               == (src / "baked_diffuse.png").read_bytes())
        assert ((dest / "baked_normal.png").read_bytes()
               == (src / "baked_normal.png").read_bytes())
    finally:
        _cleanup(name)


def test_bake_dirty_rebakes_from_the_live_grid_not_a_stale_copy(
        tmp_path, tileset_dir):
    # The source level_dir was baked from SMALL_MAP; the LIVE (unsaved) grid
    # is DIRTY_MAP (one interior tile repainted) — the scratch bake must
    # reflect DIRTY_MAP, never the source's stale SMALL_MAP PNGs.
    src = _write_src_level(tmp_path, tileset_dir, tilemap=SMALL_MAP)
    name = "c7_dirty_bake"
    try:
        dest = play_scratch.write_scratch_level(
            name, level_dir=src, grid=DIRTY_MAP,
            water_masked=np.zeros(SMALL_MAP.shape, dtype=np.int32),
            zones=None, air=None, level_boundary="space", spawns=[],
            lights=[], entities=[], wires=[], light_form="entity",
            bake_clean=False, tileset_arg=str(tileset_dir), bake_ppt=TEST_PX,
            bake_seed=0)

        got = np.asarray(Image.open(dest / "baked_diffuse.png"))
        ts = load_tileset(tileset_dir)
        want = bake_full(DIRTY_MAP, ts, px_per_tile=TEST_PX, seed=0).diffuse
        assert np.array_equal(got, want)

        stale = (src / "baked_diffuse.png").read_bytes()
        assert (dest / "baked_diffuse.png").read_bytes() != stale
    finally:
        _cleanup(name)


def test_write_scratch_level_replaces_a_stale_scratch_dir(tmp_path,
                                                          tileset_dir):
    """A second F5 press (dest already exists from a prior one) starts
    fresh — nothing from the old scratch level leaks into the new one."""
    src = _write_src_level(tmp_path, tileset_dir)
    name = "c7_replace"
    try:
        play_scratch.write_scratch_level(
            name, level_dir=src, grid=SMALL_MAP,
            water_masked=np.zeros(SMALL_MAP.shape, dtype=np.int32),
            zones=None, air=None, level_boundary="space",
            spawns=[SpawnEntry(name="old", team=0, x=1.0, y=1.0)],
            lights=[], entities=[], wires=[], light_form="entity",
            bake_clean=True, tileset_arg=str(tileset_dir), bake_ppt=TEST_PX,
            bake_seed=0)
        play_scratch.write_scratch_level(
            name, level_dir=src, grid=SMALL_MAP,
            water_masked=np.zeros(SMALL_MAP.shape, dtype=np.int32),
            zones=None, air=None, level_boundary="space", spawns=[],
            lights=[], entities=[], wires=[], light_form="entity",
            bake_clean=True, tileset_arg=str(tileset_dir), bake_ppt=TEST_PX,
            bake_seed=0)
        lvl = level_loader.load(play_scratch.scratch_level_arg(name))
        assert lvl.spawns == []
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Launch argv — sys.executable, never bare python
# ---------------------------------------------------------------------------

def test_build_launch_argv_uses_sys_executable_never_bare_python():
    argv = play_scratch.build_launch_argv("some_level")
    assert argv == [sys.executable, str(ROOT / "main.py"), "--level",
                    "_editor_scratch/some_level"]
    assert argv[0] != "python"


def test_build_launch_argv_python_override_seam_is_test_only():
    argv = play_scratch.build_launch_argv("x", python="C:/fake/python.exe")
    assert argv[0] == "C:/fake/python.exe"


# ---------------------------------------------------------------------------
# Cleanup — idempotent, best-effort
# ---------------------------------------------------------------------------

def test_cleanup_scratch_dir_removes_it(tmp_path, tileset_dir):
    src = _write_src_level(tmp_path, tileset_dir)
    name = "c7_cleanup"
    play_scratch.write_scratch_level(
        name, level_dir=src, grid=SMALL_MAP,
        water_masked=np.zeros(SMALL_MAP.shape, dtype=np.int32),
        zones=None, air=None, level_boundary="space", spawns=[], lights=[],
        entities=[], wires=[], light_form="entity", bake_clean=True,
        tileset_arg=str(tileset_dir), bake_ppt=TEST_PX, bake_seed=0)
    dest = play_scratch.scratch_dir_for(name)
    assert dest.exists()
    play_scratch.cleanup_scratch_dir(dest)
    assert not dest.exists()


def test_cleanup_scratch_dir_is_idempotent_on_a_missing_dir():
    play_scratch.cleanup_scratch_dir(
        play_scratch.scratch_dir_for("never_existed_c7"))  # must not raise


# ---------------------------------------------------------------------------
# Loader path acceptance — main.py's --level escape guard
# ---------------------------------------------------------------------------

def test_parse_level_override_accepts_scratch_form(monkeypatch):
    from main import _parse_level_override  # heavy import: test-local
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--level", "_editor_scratch/foo"])
    assert _parse_level_override() == "_editor_scratch/foo"


def test_parse_level_override_rejects_dotdot_escape(monkeypatch):
    from main import _parse_level_override  # heavy import: test-local
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--level", "../../etc/whatever"])
    with pytest.raises(SystemExit):
        _parse_level_override()


def test_parse_level_override_rejects_absolute_path(monkeypatch):
    from main import _parse_level_override  # heavy import: test-local
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--level", str(Path("C:/") / "evil")])
    with pytest.raises(SystemExit):
        _parse_level_override()


def test_parse_level_override_plain_name_unaffected(monkeypatch):
    from main import _parse_level_override  # heavy import: test-local
    monkeypatch.setattr(sys, "argv", ["main.py", "--level", "playground"])
    assert _parse_level_override() == "playground"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
