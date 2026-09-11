"""Props & vegetation arc #60, P3 — the prop entity + foliage material +
stamp_prop_tiles + level_lib append + renderer hand-off.

Design: docs/architecture/graphics/props_and_vegetation.md §4.1/§4.2/§4.3/§5
(P3 row)/§7. Mirrors the a6-doors test shape (tests/test_a6_doors.py):
programmatic ``LevelData`` fixtures, ``GameMap(level)`` directly for
load-order validation, a real ``Simulation`` only where fire/combat behavior
must be exercised.

Covers:
  - the appended ``foliage`` material row (id + columns, §6.1.2 numbers);
  - the prop entity's field kinds (synced footprint vs non-synced look —
    F10) and the digest consequence (art edits don't move it, x does);
  - ``stamp_prop_tiles``: landing (1x1 and 2x2), and every validation error
    (OOB, on-door, overlap, vacuum ring, wrong base material);
  - level_lib round-trip through the 'entity' managed family, and ordinal
    stability (D11): writing a prop into an existing level leaves every
    prior entity's ordinal + serialized record unchanged;
  - the foliage tile burns (fire consumes its wall_hp) via a real Simulation.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_prop_entity.py -q
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import level_loader  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from level_lib import format_entity_lines, open_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import prop as prop_mod  # noqa: E402
from simulation.entities.serialize import serialize_entity_state  # noqa: E402
from simulation.gamemap import GameMap, MAT_AIR  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_FOLIAGE, MAT_FURNITURE, MaterialTable,
)
from simulation.prop_system import prop_footprint, stamp_prop_tiles  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — programmatic LevelData (the a6-doors idiom; no repo level touched)
# ---------------------------------------------------------------------------

def _prop_inst(eid, ordinal, x, y, **overrides):
    fields = {f.name: f.default for f in prop_mod.prop.FIELDS}
    fields.update(x=x, y=y)
    fields.update(overrides)
    # authored_keys matters only for the level_lib round-trip tests (it is
    # what format_entity_lines actually writes) — x/y are always authored
    # (required fields), plus whatever this call explicitly overrode.
    authored = ("x", "y") + tuple(overrides.keys())
    return EntityInstance(id=eid, class_name="prop", ordinal=ordinal,
                          fields=fields, authored_keys=authored)


def _door_inst(eid, ordinal, x, y, orientation="h", length_m=1.0,
               initial_state="closed"):
    from simulation.entities import door as door_mod
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation=orientation, length_m=length_m,
                  initial_state=initial_state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal,
                          fields=fields)


def _level(tm, entities=(), name="prop_fix", version="1", tile_size_m=1.0,
          **kw):
    return LevelData(name=name, version=version, path=Path("."), tilemap=tm,
                     tile_size_m=tile_size_m, diffuse_path=Path("."),
                     entities=list(entities), **kw)


def _box_tm(h=12, w=12):
    """v1 vocabulary: hull ring (1), interior air (4). Matches the a6-doors
    fixture shape exactly (test_a6_doors.py:_box_tm)."""
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return tm


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


PREFIX = ('version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n'
          'tile_size_m = 0.333\ndiffuse = "diffuse.png"\n\n')


def _mini_level(tmp_path: Path, body: str = "", name: str = "mini") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(8)) for _ in range(6)) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(PREFIX + body, encoding="utf-8",
                                  newline="\n")
    return d


def _load(d: Path):
    return level_loader.load(str(d))


# ---------------------------------------------------------------------------
# (a) The appended `foliage` material row
# ---------------------------------------------------------------------------

def test_foliage_material_id_is_appended_last():
    # Ids are positional/contiguous — appending only (design §4.1). This
    # pins the id so a future material insertion before it is caught loudly.
    assert MAT_FOLIAGE == 9


def test_foliage_row_values_match_the_ruling():
    tbl = MaterialTable.from_config()
    # Erik's ruling 2026-09-07 §6.1.2: fuel/hp ~= 2x furniture's ACTUAL hp.
    furniture_hp = float(tbl.hp[MAT_FURNITURE])
    assert furniture_hp == 30.0, "furniture's hp moved — re-derive foliage's 2x"
    assert float(tbl.hp[MAT_FOLIAGE]) == 2.0 * furniture_hp == 60.0
    assert bool(tbl.flammable[MAT_FOLIAGE])
    # Fully walkable: mobility 1000 (== air/door's no-penalty convention),
    # NOT furniture's 400 (a movement penalty).
    assert int(tbl.mobility[MAT_FOLIAGE]) == 1000
    # No wind/vision interaction: permeability 1.0, zero light/heat atten.
    assert float(tbl.permeability[MAT_FOLIAGE]) == 1.0
    assert tuple(tbl.light_atten[MAT_FOLIAGE].tolist()) == (0.0, 0.0, 0.0)
    assert float(tbl.heat_atten[MAT_FOLIAGE]) == 0.0
    assert float(tbl.ignition_temp[MAT_FOLIAGE]) > 0.0
    # No vision interaction -> no concealment (never a cover roll).
    assert float(tbl.cover_exposure[MAT_FOLIAGE]) == 1.0


# ---------------------------------------------------------------------------
# (b) The entity's field kinds — F10 digest hygiene
# ---------------------------------------------------------------------------

def test_footprint_fields_are_synced_look_fields_are_not():
    from simulation.entities.serialize import SYNCED_FIELD_KINDS
    fields = {f.name: f for f in prop_mod.prop.FIELDS}
    for name in ("x", "y", "material", "stamp_tiles"):
        assert fields[name].kind in SYNCED_FIELD_KINDS, name
    for name in ("kind", "generator", "seed", "palette", "style", "decor",
                "height_m", "model"):
        assert fields[name].kind not in SYNCED_FIELD_KINDS, name


def test_art_field_edit_does_not_move_digest_moving_x_does():
    a = _prop_inst("p1", 0, x=5, y=5, seed="1", palette="green")
    b = _prop_inst("p1", 0, x=5, y=5, seed="99", palette="autumn",
                   decor="fruit", style="faceted", height_m=5.0,
                   generator="palm", kind="model", model="whatever.obj")
    assert serialize_entity_state([a]) == serialize_entity_state([b]), (
        "an art-only edit moved the entity digest (F10 violation)")
    c = _prop_inst("p1", 0, x=6, y=5)
    assert serialize_entity_state([a]) != serialize_entity_state([c]), (
        "moving x did not move the entity digest")
    d = _prop_inst("p1", 0, x=5, y=5, stamp_tiles=2)
    assert serialize_entity_state([a]) != serialize_entity_state([d]), (
        "changing stamp_tiles did not move the entity digest")


# ---------------------------------------------------------------------------
# (c) stamp_prop_tiles — landing
# ---------------------------------------------------------------------------

def test_stamp_lands_material_on_the_anchor_tile():
    tm = _box_tm()
    lvl = _level(tm, [_prop_inst("p1", 0, x=5, y=5)])
    g = GameMap(lvl)
    assert int(g.material[5, 5]) == MAT_FOLIAGE
    assert int(g.wall_hp[5, 5]) > 0, "foliage carries no fuel/HP"


def test_stamp_lands_a_2x2_footprint():
    tm = _box_tm()
    lvl = _level(tm, [_prop_inst("p1", 0, x=5, y=5, stamp_tiles=2)])
    g = GameMap(lvl)
    for fy, fx in prop_footprint({"x": 5, "y": 5, "stamp_tiles": 2}):
        assert int(g.material[fy, fx]) == MAT_FOLIAGE, (fy, fx)


# ---------------------------------------------------------------------------
# (d) stamp_prop_tiles — validation errors
# ---------------------------------------------------------------------------

def test_stamp_validation_out_of_bounds():
    tm = _box_tm()
    with pytest.raises(ValueError, match="out of bounds"):
        GameMap(_level(tm, [_prop_inst("p1", 0, x=50, y=5)]))


def test_stamp_validation_wrong_base_material():
    tm = _box_tm()
    with pytest.raises(ValueError, match="CSV material"):
        GameMap(_level(tm, [_prop_inst("p1", 0, x=0, y=5)]))  # on the hull ring


def test_stamp_validation_vacuum_ring():
    tm = _box_tm()
    tm[5, 5] = 0                                     # v1 code 0 = vacuum
    with pytest.raises(ValueError, match="vacuum"):
        GameMap(_level(tm, [_prop_inst("p1", 0, x=5, y=5)]))


def test_stamp_validation_overlap_with_another_prop():
    tm = _box_tm()
    with pytest.raises(ValueError, match="overlaps prop entity"):
        GameMap(_level(tm, [_prop_inst("a", 0, x=5, y=5, stamp_tiles=2),
                            _prop_inst("b", 1, x=6, y=6)]))


def test_stamp_validation_overlap_with_a_door_span():
    tm = _box_tm()
    door = _door_inst("d", 0, x=5, y=5, orientation="h", length_m=1.0)
    prop = _prop_inst("p1", 1, x=5, y=5)
    with pytest.raises(ValueError, match="door"):
        GameMap(_level(tm, [door, prop]))


def test_stamp_validation_height_m_cap():
    tm = _box_tm()
    # ~20 tiles at tile_size_m=1.0 -> cap 20 m.
    with pytest.raises(ValueError, match="ortho-camera budget"):
        GameMap(_level(tm, [_prop_inst("p1", 0, x=5, y=5, height_m=21.0)]))


def test_stamp_validation_model_kind_requires_existing_file():
    tm = _box_tm()
    with pytest.raises(ValueError, match="does not exist"):
        GameMap(_level(tm, [_prop_inst("p1", 0, x=5, y=5, kind="model",
                                       model="nope/nothing.obj")]))
    with pytest.raises(ValueError, match="unsupported extension"):
        GameMap(_level(tm, [_prop_inst("p2", 0, x=5, y=5, kind="model",
                                       model="nope/nothing.png")]))


# ---------------------------------------------------------------------------
# (e) level_lib — round-trip + ordinal stability
# ---------------------------------------------------------------------------

LAMPS = ('[[entity]]\nid = "lamp_1"\nclass = "light"\nx = 2.5\ny = 3.5\n\n'
        '[[entity]]\nid = "lamp_2"\nclass = "light"\nx = 1.0\ny = 4.0\n\n')


def test_prop_round_trips_through_level_lib(tmp_path):
    d = _mini_level(tmp_path, LAMPS)
    toml = d / "level.toml"
    handle = open_level(str(d))
    handle.data.entities.append(
        _prop_inst("tree_1", len(handle.data.entities), x=3, y=3,
                  seed="7", palette="autumn"))
    handle.save({"entity":
                lambda nl: format_entity_lines(handle.data.entities, nl)})
    reloaded = _load(d)
    assert [e.id for e in reloaded.entities] == ["lamp_1", "lamp_2", "tree_1"]
    prop_e = reloaded.entities[-1]
    assert prop_e.class_name == "prop"
    assert prop_e.fields["x"] == 3 and prop_e.fields["y"] == 3
    assert prop_e.fields["seed"] == "7"
    assert prop_e.fields["palette"] == "autumn"
    assert prop_e.fields["material"] == "foliage"
    # Byte-stable for what it didn't touch: writing back unmodified entities
    # (no prop appended) reproduces the file exactly.
    before = toml.read_bytes()
    handle2 = open_level(str(d))
    handle2.save({"entity":
                 lambda nl: format_entity_lines(handle2.data.entities, nl)})
    assert toml.read_bytes() == before


def test_ordinal_stability_writing_a_prop_leaves_prior_entities_unchanged(
        tmp_path):
    d = _mini_level(tmp_path, LAMPS)
    before = _load(d)
    before_records = serialize_entity_state(before.entities)
    assert [e.ordinal for e in before.entities] == [0, 1]

    handle = open_level(str(d))
    handle.data.entities.append(
        _prop_inst("tree_1", len(handle.data.entities), x=3, y=3))
    handle.save({"entity":
                lambda nl: format_entity_lines(handle.data.entities, nl)})

    after = _load(d)
    assert [e.id for e in after.entities] == ["lamp_1", "lamp_2", "tree_1"]
    assert [e.ordinal for e in after.entities] == [0, 1, 2]
    # The prior two entities' serialized records are byte-identical to
    # before the prop was appended (D11 — ordinal + record stability).
    prior_two = [e for e in after.entities if e.class_name == "light"]
    assert serialize_entity_state(prior_two) == before_records
    assert before.entities[0].fields == after.entities[0].fields
    assert before.entities[1].fields == after.entities[1].fields


# ---------------------------------------------------------------------------
# (f) The foliage tile burns
# ---------------------------------------------------------------------------

def test_foliage_tile_burns():
    tm = _box_tm()
    x, y = 5, 5
    lvl = _level(tm, [_prop_inst("tree_1", 0, x=x, y=y)])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    gmap = sim.gmap
    assert int(gmap.material[y, x]) == MAT_FOLIAGE
    hp0 = int(gmap.wall_hp[y, x])
    assert hp0 > 0, "foliage stamped with no fuel/HP"

    gmap.fire[y, x] = 40000            # ignite directly (raw Q16.16 ~ 0.61)
    sim.set_paused(False)
    for _ in range(400):
        sim.step()

    hp1 = int(gmap.wall_hp[y, x])
    assert hp1 < hp0, "fire did not consume the foliage tile's wall_hp"
