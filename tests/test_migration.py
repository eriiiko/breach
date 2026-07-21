"""A7 — the one-shot legacy -> [[entity]] migration tool
(tools/migrate_level_entities.py; level editor v3 design §6, arc plan A7).

Covers, on tmp copies of a synthetic fixture (never a repo level):

  - painted MAT_DOOR run grouping: maximal h runs first, then v runs, then
    singletons — L-shaped adjacencies stay SEPARATE h/v runs; a 2xN block
    is N.. two parallel h runs; ids door_1..n in anchor scan order.
  - length_m synthesis round-trips through THE canonical quantization.
  - [[light]] -> entity lights (field-for-field LightEntry equivalence),
    [[spawn]] untouched, tilemap 3 -> 7, .bak of every touched file.
  - idempotence (second run is a no-op), mixed-form refusal
    (half-migrated = corrupt), --dry-run writes nothing.
  - the migrated level round-trips byte-stably through level_lib.
  - THE DOOR-STAMP-LEAK GUARD, PORTED to MAT_DOOR_CLOSED semantics: the
    legacy guard (test_atmosphere_conservation.py) pins the walkable
    hybrid — a unit standing on a flow-solid MAT_DOOR tile must not raise
    its permeability. A migrated (closed entity) door is FULLY solid: its
    tiles must be flow-solid (solid mask, permeability 0, and the unit
    stamp still clamped to 0) AND movement-solid (is_passable /
    is_passable_block False) — the sanctioned A7 behavior change, pinned.
  - the repo's migrated test_level (post-A7 commit 2): entity-form,
    no painted MAT_DOOR, GameMap constructs. Skips on a pre-migration
    tree so commit 1 (tool + tests) stays green standalone.

Run:
    conda run -n data python -m pytest tests/test_migration.py -q
"""
from __future__ import annotations

import shutil
import struct
import sys
import tomllib
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_lib  # noqa: E402
import level_loader  # noqa: E402
from config import CFG  # noqa: E402
from migrate_level_entities import (  # noqa: E402
    MigrationPlan, door_runs, length_m_for_tiles, migrate_level,
)
from simulation.entities import door as door_schema  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_DOOR, MAT_DOOR_CLOSED  # noqa: E402
from simulation.unit import Unit  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture — a synthetic v2 level folder with every legacy form
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


# 16x16, v2 vocabulary: hull ring (1), interior air (0), a hull wall at
# col 8 sealing left from right with a painted 2-tile v door at (6,8),(7,8);
# an L-shape (h 3 + v 2) sharing the corner COLUMN at (3,3..5)/(4..5,3); a
# singleton at (10,10); a 2x3 block at (12..13, 3..5) = two parallel h runs.
_DOOR_TILES = sorted(
    [(3, 3), (3, 4), (3, 5), (4, 3), (5, 3),          # L-shape
     (6, 8), (7, 8),                                   # sealing v door
     (10, 10),                                         # singleton
     (12, 3), (12, 4), (12, 5), (13, 3), (13, 4), (13, 5)])  # 2x3 block

# (id, x, y, orientation, length_m, n_tiles) in anchor scan order.
_EXPECTED_DOORS = [
    ("door_1", 3, 3, "h", 1.0, 3),
    ("door_2", 3, 4, "v", 0.667, 2),
    ("door_3", 8, 6, "v", 0.667, 2),
    ("door_4", 10, 10, "h", 0.333, 1),
    ("door_5", 3, 12, "h", 1.0, 3),
    ("door_6", 3, 13, "h", 1.0, 3),
]

_TOML = '''# hand comment survives migration
version = "2"
name = "migfix"
tilemap = "tilemap.csv"
tile_size_m = 0.333
diffuse = "diffuse.png"

[[spawn]]
name = "M1"
team = 0
x = 2.0
y = 9.0
footprint = 3

[[light]]
pos = [2.5, 3.5]
color = [255, 0, 0]

[[light]]
pos = [10.0, 4.0]
color = [10, 20, 30]
intensity = 2.0
range = 8.0
kind = "beacon"
period_s = 1.5
beam_deg = 45.0
phase = 0.5
'''


def _fixture_tilemap() -> np.ndarray:
    tm = np.zeros((16, 16), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1     # hull ring
    tm[1:15, 8] = 1                                      # interior wall
    for r, c in _DOOR_TILES:
        tm[r, c] = MAT_DOOR
    return tm


def _make_fixture(tmp_path: Path, name: str = "migfix") -> Path:
    d = tmp_path / name
    d.mkdir()
    tm = _fixture_tilemap()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join(str(int(v)) for v in row) for row in tm.tolist())
        + "\n", encoding="ascii", newline="\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(_TOML, encoding="utf-8", newline="\n")
    return d


def _bytes_of(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())
            if p.is_file()}


# ---------------------------------------------------------------------------
# Run grouping + length synthesis (pure functions)
# ---------------------------------------------------------------------------

def test_door_runs_l_shape_stays_separate_h_and_v():
    runs = door_runs(_fixture_tilemap())
    got = [(o, tiles) for o, tiles in runs]
    assert got == [
        ("h", [(3, 3), (3, 4), (3, 5)]),
        ("v", [(4, 3), (5, 3)]),          # L: corner claimed by the h run
        ("v", [(6, 8), (7, 8)]),
        ("h", [(10, 10)]),                # singleton -> 1-tile h
        ("h", [(12, 3), (12, 4), (12, 5)]),
        ("h", [(13, 3), (13, 4), (13, 5)]),  # 2x3 block = two parallel h
    ]
    # Every painted tile lands in exactly one run.
    all_tiles = sorted(t for _, tiles in runs for t in tiles)
    assert all_tiles == _DOOR_TILES


def test_door_runs_cross_shape_is_deterministic():
    """A painted cross: the h bar claims the shared tile; the v remainder
    splits into two singletons (degenerate authoring, still deterministic
    and exhaustive)."""
    tm = np.zeros((7, 7), dtype=np.int32)
    for t in [(2, 2), (2, 3), (2, 4), (1, 3), (3, 3)]:
        tm[t] = MAT_DOOR
    runs = door_runs(tm)
    assert runs == [
        ("h", [(1, 3)]),
        ("h", [(2, 2), (2, 3), (2, 4)]),
        ("h", [(3, 3)]),
    ]


def test_length_m_round_trips_through_canonical_quantization():
    # tile_size_m 0.333 -> exactly 3 tiles/m (the shipped mapping).
    for n, expect in [(1, 0.333), (2, 0.667), (3, 1.0), (4, 1.333),
                     (5, 1.667), (6, 2.0), (7, 2.333)]:
        val = length_m_for_tiles(n, 0.333)
        assert val == expect
        assert door_schema.quantize_span_tiles(val, 0.333) == n
    # 1 tile/m level: n tiles == n meters.
    assert length_m_for_tiles(2, 1.0) == 2.0
    assert door_schema.quantize_span_tiles(2.0, 1.0) == 2


# ---------------------------------------------------------------------------
# The migration, end to end on the fixture
# ---------------------------------------------------------------------------

def test_migrate_fixture_end_to_end(tmp_path):
    d = _make_fixture(tmp_path)
    pre = _bytes_of(d)
    old_lvl = level_loader.load(str(d))
    assert old_lvl.entities == [] and len(old_lvl.lights) == 2

    plan = migrate_level(d, verbose=False)
    assert isinstance(plan, MigrationPlan) and not plan.noop

    # Doors: ids/anchors/orientations/lengths in anchor scan order.
    got = [(e.id, e.fields["x"], e.fields["y"], e.fields["orientation"],
            e.fields["length_m"],
            len(door_schema.base_span(e.fields, 0.333)))
           for e in plan.doors]
    assert got == _EXPECTED_DOORS
    assert all(e.fields["initial_state"] == "closed" for e in plan.doors)

    # The rewritten level parses as pure entity form.
    lvl = level_loader.load(str(d))
    assert [e.id for e in lvl.entities] == \
        [f"door_{i}" for i in range(1, 7)] + ["light_1", "light_2"]
    assert "light" not in lvl.raw_toml            # legacy blocks gone
    # Lights: field-for-field equivalent downstream LightEntry values.
    assert lvl.lights == old_lvl.lights
    # [[spawn]] untouched — parsed AND textual.
    assert lvl.spawns == old_lvl.spawns
    assert lvl.raw_toml["spawn"] == old_lvl.raw_toml["spawn"]
    # Tilemap: every painted 3 became 7, nothing else moved.
    assert int((lvl.tilemap == MAT_DOOR).sum()) == 0
    assert sorted((int(r), int(c)) for r, c in
                  np.argwhere(lvl.tilemap == MAT_DOOR_CLOSED)) == _DOOR_TILES
    changed = lvl.tilemap != _fixture_tilemap()
    assert sorted((int(r), int(c)) for r, c in np.argwhere(changed)) \
        == _DOOR_TILES
    # Untouched bytes outside the managed families (the hand comment).
    text = (d / "level.toml").read_text(encoding="utf-8")
    assert text.startswith("# hand comment survives migration\n")

    # .bak of every touched file carries the pre-migration bytes.
    assert (d / "level.toml.bak").read_bytes() == pre["level.toml"]
    assert (d / "tilemap.csv.bak").read_bytes() == pre["tilemap.csv"]


def test_migrated_level_round_trips_byte_stably_through_level_lib(tmp_path):
    d = _make_fixture(tmp_path)
    migrate_level(d, verbose=False)
    original = (d / "level.toml").read_bytes()
    handle = level_lib.open_level(str(d))
    handle.save()                                  # no-change save == identity
    assert (d / "level.toml").read_bytes() == original
    # And a full managed rewrite of the entity family is stable too.
    handle.save({"entity": lambda nl: level_lib.format_entity_lines(
        handle.data.entities, nl)})
    assert (d / "level.toml").read_bytes() == original


def test_second_run_is_a_noop(tmp_path):
    d = _make_fixture(tmp_path)
    migrate_level(d, verbose=False)
    after = _bytes_of(d)
    plan = migrate_level(d, verbose=False)
    assert plan.noop and plan.doors == [] and plan.lights == []
    assert _bytes_of(d) == after                   # nothing rewritten, no
    #                                                new/overwritten .bak


def test_lights_only_level_migrates_without_touching_csv(tmp_path):
    d = _make_fixture(tmp_path, name="lightsonly")
    tm = np.zeros((16, 16), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    (d / "tilemap.csv").write_text(
        "\n".join(",".join(str(int(v)) for v in row) for row in tm.tolist())
        + "\n", encoding="ascii", newline="\n")
    pre_csv = (d / "tilemap.csv").read_bytes()
    plan = migrate_level(d, verbose=False)
    assert plan.doors == [] and len(plan.lights) == 2
    assert (d / "tilemap.csv").read_bytes() == pre_csv
    assert not (d / "tilemap.csv.bak").exists()    # only touched files .bak
    assert (d / "level.toml.bak").exists()


def test_mixed_form_level_is_refused_untouched(tmp_path):
    # Half-migrated: entity doors landed but the CSV kept a painted 3
    # (the crash-between-writes state, or a hand edit).
    d = _make_fixture(tmp_path)
    migrate_level(d, verbose=False)
    tm = np.loadtxt(d / "tilemap.csv", delimiter=",", dtype=np.int32)
    tm[6, 8] = MAT_DOOR                            # repaint one legacy tile
    level_lib.write_tilemap_csv(d, tm, csv_bak=False)
    before = _bytes_of(d)
    with pytest.raises(ValueError, match="HALF-MIGRATED"):
        migrate_level(d, verbose=False)
    assert _bytes_of(d) == before                  # refusal writes nothing

    # Entity lights + legacy [[light]] in one file: the LOADER refuses
    # (naming this tool), so the tool refuses too, before any write.
    d2 = _make_fixture(tmp_path, name="mixlight")
    body = ('\n[[entity]]\nid = "lamp_x"\nclass = "light"\n'
            'x = 2.0\ny = 2.0\ncolor = [1, 2, 3]\n')
    with open(d2 / "level.toml", "a", encoding="utf-8", newline="\n") as f:
        f.write(body)
    before2 = _bytes_of(d2)
    with pytest.raises(ValueError, match="migrate_level_entities"):
        migrate_level(d2, verbose=False)
    assert _bytes_of(d2) == before2


def test_dry_run_plans_everything_writes_nothing(tmp_path):
    d = _make_fixture(tmp_path)
    before = _bytes_of(d)
    plan = migrate_level(d, dry_run=True, verbose=False)
    assert [e.id for e in plan.doors] == [f"door_{i}" for i in range(1, 7)]
    assert [e.id for e in plan.lights] == ["light_1", "light_2"]
    assert plan.door_tiles == _DOOR_TILES
    assert _bytes_of(d) == before
    assert not (d / "level.toml.bak").exists()
    assert not (d / "tilemap.csv.bak").exists()


def test_v1_level_is_refused(tmp_path):
    d = _make_fixture(tmp_path, name="v1fix")
    toml = (d / "level.toml").read_text(encoding="utf-8")
    (d / "level.toml").write_text(toml.replace('version = "2"',
                                               'version = "1"'),
                                  encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="migrate_tilemap_v2"):
        migrate_level(d, verbose=False)


# ---------------------------------------------------------------------------
# THE DOOR-STAMP-LEAK GUARD, ported to MAT_DOOR_CLOSED semantics
# ---------------------------------------------------------------------------

def test_leak_guard_ported_closed_entity_door_is_fully_solid(tmp_path):
    """The legacy guard (test_atmosphere_conservation.py, still in force
    for unmigrated levels) pins the HYBRID: painted MAT_DOOR is walkable
    yet flow-solid, and the unit stamp must not raise its permeability.
    Post-migration the hybrid is GONE: a closed entity door's tiles are
    flow-solid AND movement-solid — and the stamp clamp still holds, so
    the original leak configuration (a unit footprint over a sealed door
    tile) can never come back through the new semantics."""
    d = _make_fixture(tmp_path)

    # Pre-migration: the hybrid, for contrast (the sanctioned change's
    # "before" — walkable, flow-solid).
    g0 = GameMap(level_loader.load(str(d)))
    assert bool(g0.solid[6, 8]) and g0.permeability[6, 8] == 0.0
    assert g0.is_passable(6, 8)                    # the walkable hybrid

    migrate_level(d, verbose=False)
    g = GameMap(level_loader.load(str(d)))

    for (r, c) in [(6, 8), (7, 8)]:                # the sealing v door
        assert int(g.material[r, c]) == MAT_DOOR_CLOSED
        # Flow-solid: sealed exactly like a wall.
        assert bool(g.solid[r, c]) and g.permeability[r, c] == 0.0
        # Movement-solid: the A7 behavior change, pinned.
        assert not g.is_passable(r, c)
    assert not g.is_passable_block(5, 7, 3)        # block overlapping span

    # The stamp clamp (the two-layer fix's Python half): a unit footprint
    # covering the closed door tiles leaves them flow-sealed while its air
    # tiles carry the partial body permeability.
    u = Unit("DoorStander", x=7, y=5, team=0)      # 3x3 over (5..7, 7..9)
    g.stamp_units([u])
    assert g.dyn_permeability[6, 8] == 0.0
    assert g.dyn_permeability[7, 8] == 0.0
    expected = np.float32(getattr(CFG.physics, "unit_permeability", 0.5))
    for (tx, ty) in u.occupied_tiles():
        if not g.solid[ty, tx]:
            assert g.dyn_permeability[ty, tx] == expected


# ---------------------------------------------------------------------------
# The repo's migrated test_level (lands with A7 commit 2)
# ---------------------------------------------------------------------------

def test_repo_test_level_is_entity_form():
    lvl = level_loader.load("test_level")
    if int((lvl.tilemap == MAT_DOOR).sum()) > 0:
        pytest.skip("pre-A7 tree: levels/test_level not yet migrated")
    doors = [e for e in lvl.entities if e.class_name == "door"]
    lights = [e for e in lvl.entities if e.class_name == "light"]
    assert len(doors) == 5 and len(lights) == 8
    assert "light" not in lvl.raw_toml
    span_tiles = sorted(
        t for e in doors
        for t in door_schema.base_span(e.fields, float(lvl.tile_size_m)))
    assert sorted((int(r), int(c)) for r, c in
                  np.argwhere(lvl.tilemap == MAT_DOOR_CLOSED)) == span_tiles
    assert len(span_tiles) == 15
    GameMap(lvl)                                   # spans validate + seed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
