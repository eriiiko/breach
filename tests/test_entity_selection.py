"""tools/entity_selection.py — unified entity selection: hit-test, box/
shift/by-class select, move, delete, tag-assign, inspector field edits, and
clump copy/paste (Arc C4).

C3 left doors/sensors/generic placements with no select/move/delete/edit
UX; this module gives them the same affordances LIGHT/SPAWN already had,
uniformly, over the SAME `lights`/`entities` collections the C2 transaction
log tracks. Pins:

  - hit-testing: a door's hitbox is its whole SPAN (recomputed via
    `door_entity_port.instance_span`, never a parallel calculation), other
    positioned classes are their one tile, a positionless class has none;
  - box_select / select_by_class reach every hitbox / every class member
    (incl. positionless, which only select_by_class can reach);
  - move: ONE compound transaction, a door re-stamps its span (old cleared,
    new stamped), refuses (no mutation) when a door's new span would leave
    the grid, undo restores exactly;
  - delete: ONE compound transaction, drops wires whose non-tag endpoint
    named a deleted id (the C2 §7 referential-integrity rule), keeps
    tag-target wires;
  - tag-assign: ONE transaction, idempotent, touches only the collections
    that actually changed;
  - inspector field edits: ONE transaction, a door re-stamp on
    length_m/orientation/initial_state, id-uniqueness enforced BEFORE any
    mutation, a not-yet-authored field joins authored_keys so the edit
    survives a save;
  - clump copy/paste: internal (non-tag) wires re-pointed to freshly minted
    ids, external + tag-target wires dropped at copy time, a pasted door's
    grid stamp is part of the SAME transaction undo reverts atomically, and
    a copy -> paste -> save -> reload round-trip is loadable with unique
    ids and both the wire and entity families intact.

Run:
    python -m pytest tests/test_entity_selection.py -q
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import entity_selection as esel  # noqa: E402
import door_entity_port as dep  # noqa: E402
import light_entity_port as lep  # noqa: E402
import undo_log  # noqa: E402
from level_loader import EntityInstance, WireSpec  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR_CLOSED, MAT_HULL  # noqa: E402

TILE_SIZE_M = 0.333          # the shipped level default -> tiles_per_m == 3
WALL_CODES = frozenset({MAT_HULL})


def _wall_row(width=24, height=8, y=3, x0=1, x1=22):
    grid = np.full((height, width), MAT_AIR, dtype=np.int32)
    grid[y, x0:x1] = MAT_HULL
    return grid


def _door(id_, x, y, orientation="h", n_tiles=3, initial_state="closed"):
    length_m = dep.length_m_for_tiles(n_tiles, TILE_SIZE_M)
    return dep.build_door_instance(x, y, orientation, length_m,
                                   initial_state, id_)


def _stamp_door(grid, d):
    for fy, fx in dep.instance_span(d.fields, TILE_SIZE_M):
        grid[fy, fx] = dep.stamp_value_for(d.fields["initial_state"])


def _light(id_, x, y, tags=()):
    return lep.EditableLight(x=x, y=y, color=(1.0, 0.0, 0.0), id=id_,
                             tags=tuple(tags))


def _generic(id_, cls_name, x, y, tags=()):
    return EntityInstance(id=id_, class_name=cls_name, ordinal=0,
                          tags=tuple(tags), fields={"x": x, "y": y},
                          authored_keys=("x", "y"))


def _positionless(id_, cls_name="decider"):
    return EntityInstance(id=id_, class_name=cls_name, ordinal=0,
                          fields={}, authored_keys=())


# ---------------------------------------------------------------------------
# instance_tiles / hit_test / box_select / select_by_class
# ---------------------------------------------------------------------------

def test_instance_tiles_door_is_its_whole_span():
    d = _door("door_1", 2, 3, n_tiles=3)
    assert esel.instance_tiles("entities", d, TILE_SIZE_M) == frozenset(
        {(2, 3), (3, 3), (4, 3)})


def test_instance_tiles_light_is_its_floored_tile():
    l = _light("light_1", 2.9, 3.1)
    assert esel.instance_tiles("lights", l, TILE_SIZE_M) == frozenset({(2, 3)})


def test_instance_tiles_generic_entity_is_its_one_tile():
    e = _generic("sensor_1", "pressure", 5, 6)
    assert esel.instance_tiles("entities", e, TILE_SIZE_M) == frozenset({(5, 6)})


def test_instance_tiles_positionless_entity_is_empty():
    e = _positionless("dec_1")
    assert esel.instance_tiles("entities", e, TILE_SIZE_M) == frozenset()


def test_hit_test_entities_beat_lights_on_the_same_tile_topmost_wins():
    lights = [_light("light_1", 5.5, 3.5)]
    entities = [_generic("sensor_1", "pressure", 5, 3)]
    assert esel.hit_test(5, 3, lights, entities, TILE_SIZE_M) == "sensor_1"
    assert esel.hit_test(9, 9, lights, entities, TILE_SIZE_M) is None
    assert esel.hit_test(5, 3, lights, [], TILE_SIZE_M) == "light_1"


def test_box_select_finds_every_hitbox_tile_including_a_door_span():
    d = _door("door_1", 2, 3, n_tiles=3)
    lights = [_light("light_1", 8.5, 8.5)]
    entities = [d]
    assert esel.box_select(2, 3, 4, 3, lights, entities, TILE_SIZE_M) == [
        "door_1"]
    ids = esel.box_select(20, 20, 0, 0, lights, entities, TILE_SIZE_M)
    assert set(ids) == {"door_1", "light_1"}
    assert esel.box_select(0, 0, 1, 1, lights, entities, TILE_SIZE_M) == []


def test_select_by_class_reaches_positionless_instances():
    entities = [_positionless("dec_1"), _positionless("dec_2"),
               _generic("sensor_1", "pressure", 1, 1)]
    assert set(esel.select_by_class("decider", [], entities)) == {
        "dec_1", "dec_2"}
    assert esel.select_by_class(
        "light", [_light("light_1", 1, 1)], entities) == ["light_1"]
    assert esel.select_by_class("pressure", [], entities) == ["sensor_1"]


def test_find_instance_and_all_ids():
    lights = [_light("light_1", 1, 1)]
    entities = [_generic("sensor_1", "pressure", 2, 2)]
    assert esel.find_instance("light_1", lights, entities) == ("lights", 0)
    assert esel.find_instance("sensor_1", lights, entities) == ("entities", 0)
    assert esel.find_instance("ghost", lights, entities) == (None, None)
    assert set(esel.all_ids(lights, entities)) == {"light_1", "sensor_1"}


# ---------------------------------------------------------------------------
# Move — one compound transaction, door re-stamp, undo
# ---------------------------------------------------------------------------

def test_commit_move_selection_moves_light_and_entity_one_txn_each_coll():
    grid = _wall_row()
    lights = [_light("light_1", 2.5, 3.5)]
    entities = [_generic("sensor_1", "pressure", 5, 3)]
    ctx = undo_log.UndoContext(
        grids={"material": grid},
        collections={"lights": lights, "entities": entities})
    log = undo_log.TransactionLog(ctx)

    txn = esel.commit_move_selection(log, grid, lights, entities,
                                     ["light_1", "sensor_1"], 2, 1,
                                     TILE_SIZE_M)
    assert txn is not None
    assert len(txn.ops) == 2                  # lights + entities, no grid op
    assert (lights[0].x, lights[0].y) == (4.5, 4.5)
    assert entities[0].fields == {"x": 7, "y": 4}

    log.undo()
    assert (lights[0].x, lights[0].y) == (2.5, 3.5)
    assert entities[0].fields == {"x": 5, "y": 3}
    log.redo()
    assert entities[0].fields == {"x": 7, "y": 4}


def test_commit_move_selection_door_restamps_grid_and_undo_restores_span():
    grid = _wall_row()
    d = _door("door_1", 2, 3, n_tiles=3)
    _stamp_door(grid, d)
    entities = [d]
    ctx = undo_log.UndoContext(grids={"material": grid},
                               collections={"lights": [], "entities": entities})
    log = undo_log.TransactionLog(ctx)
    grid_before = grid.copy()

    txn = esel.commit_move_selection(log, grid, [], entities, ["door_1"],
                                     3, 0, TILE_SIZE_M)
    assert txn is not None
    assert len(txn.ops) == 2                  # material + entities
    assert (grid[3, 2:5] == MAT_AIR).all()          # old span cleared
    assert (grid[3, 5:8] == MAT_DOOR_CLOSED).all()  # new span stamped
    assert entities[0].fields["x"] == 5

    log.undo()
    assert np.array_equal(grid, grid_before)
    assert entities[0].fields["x"] == 2


def test_commit_move_selection_refuses_when_door_span_would_leave_grid():
    grid = _wall_row()
    d = _door("door_1", 2, 3, n_tiles=1)
    _stamp_door(grid, d)
    entities = [d]
    ctx = undo_log.UndoContext(grids={"material": grid},
                               collections={"lights": [], "entities": entities})
    log = undo_log.TransactionLog(ctx)
    grid_before = grid.copy()

    huge = grid.shape[1] + 100
    txn = esel.commit_move_selection(log, grid, [], entities, ["door_1"],
                                     huge, 0, TILE_SIZE_M)
    assert txn is None
    assert np.array_equal(grid, grid_before)
    assert entities[0].fields["x"] == 2
    assert log.undo_count == 0


def test_commit_move_selection_zero_delta_or_empty_ids_is_a_noop():
    grid = _wall_row()
    entities = [_generic("sensor_1", "pressure", 5, 3)]
    ctx = undo_log.UndoContext(grids={"material": grid},
                               collections={"lights": [], "entities": entities})
    log = undo_log.TransactionLog(ctx)
    assert esel.commit_move_selection(log, grid, [], entities, ["sensor_1"],
                                      0, 0, TILE_SIZE_M) is None
    assert esel.commit_move_selection(log, grid, [], entities, [],
                                      1, 1, TILE_SIZE_M) is None
    assert log.undo_count == 0


# ---------------------------------------------------------------------------
# Delete — one compound transaction, referential-integrity wire cleanup
# ---------------------------------------------------------------------------

def test_commit_delete_selection_clears_door_span_and_drops_touching_wires():
    grid = _wall_row()
    d = _door("door_1", 2, 3, n_tiles=3)
    _stamp_door(grid, d)
    other = _generic("sensor_1", "pressure", 8, 3)
    entities = [d, other]
    wires = [
        WireSpec("door_1.is_open", "sensor_1.threshold"),   # from touches
        WireSpec("sensor_1.value", "door_1.close"),          # to touches
        WireSpec("sensor_1.value", "tag:group.input"),       # tag -> kept
    ]
    ctx = undo_log.UndoContext(
        grids={"material": grid},
        collections={"lights": [], "entities": entities, "wires": wires})
    log = undo_log.TransactionLog(ctx)
    grid_before = grid.copy()

    txn = esel.commit_delete_selection(log, grid, [], entities, wires,
                                       ["door_1"], TILE_SIZE_M)
    assert txn is not None
    assert [e.id for e in entities] == ["sensor_1"]
    assert (grid[3, 2:5] == MAT_AIR).all()
    assert [w.to for w in wires] == ["tag:group.input"]

    log.undo()
    assert np.array_equal(grid, grid_before)
    assert [e.id for e in entities] == ["door_1", "sensor_1"]
    assert len(wires) == 3


def test_commit_delete_selection_keeps_unrelated_wires_and_lights():
    lights = [_light("light_1", 1, 1)]
    entities = [_generic("sensor_1", "pressure", 2, 2),
               _generic("sensor_2", "pressure", 3, 3)]
    wires = [WireSpec("sensor_2.value", "sensor_1.threshold")]
    grid = _wall_row()
    ctx = undo_log.UndoContext(
        grids={"material": grid},
        collections={"lights": lights, "entities": entities, "wires": wires})
    log = undo_log.TransactionLog(ctx)
    txn = esel.commit_delete_selection(log, grid, lights, entities, wires,
                                       ["light_1"], TILE_SIZE_M)
    assert txn is not None
    assert lights == []
    assert len(entities) == 2 and len(wires) == 1


def test_commit_delete_selection_empty_ids_is_noop():
    entities = [_generic("sensor_1", "pressure", 1, 1)]
    wires = []
    ctx = undo_log.UndoContext(
        grids={}, collections={"lights": [], "entities": entities,
                               "wires": wires})
    log = undo_log.TransactionLog(ctx)
    assert esel.commit_delete_selection(log, None, [], entities, wires,
                                        [], TILE_SIZE_M) is None
    assert log.undo_count == 0


# ---------------------------------------------------------------------------
# Assign-tag-to-selection
# ---------------------------------------------------------------------------

def test_commit_assign_tag_adds_to_every_selected_one_txn_and_undo():
    lights = [_light("light_1", 1.5, 1.5)]
    entities = [_generic("sensor_1", "pressure", 2, 2)]
    ctx = undo_log.UndoContext(
        grids={}, collections={"lights": lights, "entities": entities})
    log = undo_log.TransactionLog(ctx)

    txn = esel.commit_assign_tag(log, lights, entities,
                                 ["light_1", "sensor_1"], "wing_a")
    assert txn is not None
    assert lights[0].tags == ("wing_a",)
    assert entities[0].tags == ("wing_a",)

    log.undo()
    assert lights[0].tags == () and entities[0].tags == ()


def test_commit_assign_tag_idempotent_no_op_when_all_already_tagged():
    lights = [_light("light_1", 1, 1, tags=("wing_a",))]
    ctx = undo_log.UndoContext(grids={}, collections={"lights": lights,
                                                      "entities": []})
    log = undo_log.TransactionLog(ctx)
    assert esel.commit_assign_tag(log, lights, [], ["light_1"],
                                  "wing_a") is None
    assert log.undo_count == 0


def test_commit_assign_tag_empty_selection_or_tag_is_noop():
    lights = [_light("light_1", 1, 1)]
    ctx = undo_log.UndoContext(grids={}, collections={"lights": lights,
                                                      "entities": []})
    log = undo_log.TransactionLog(ctx)
    assert esel.commit_assign_tag(log, lights, [], [], "x") is None
    assert esel.commit_assign_tag(log, lights, [], ["light_1"], "") is None


# ---------------------------------------------------------------------------
# Inspector field edits — one transaction, door re-stamp, id uniqueness
# ---------------------------------------------------------------------------

def test_commit_field_edit_generic_int_field_one_txn_and_undo():
    entities = [_generic("sensor_1", "sensor_motion", 5, 3)]
    ctx = undo_log.UndoContext(grids={}, collections={"lights": [],
                                                      "entities": entities})
    log = undo_log.TransactionLog(ctx)
    ok, txn = esel.commit_field_edit(log, None, [], entities, "sensor_1",
                                     "x", 9, TILE_SIZE_M)
    assert ok and txn is not None and len(txn.ops) == 1
    assert entities[0].fields["x"] == 9
    log.undo()
    assert entities[0].fields["x"] == 5


def test_commit_field_edit_adds_field_to_authored_keys_when_missing():
    e = EntityInstance(id="sensor_1", class_name="sensor_motion", ordinal=0,
                       fields={"x": 5, "y": 3, "sample_dx": 0},
                       authored_keys=("x", "y"))
    entities = [e]
    ctx = undo_log.UndoContext(grids={}, collections={"lights": [],
                                                      "entities": entities})
    log = undo_log.TransactionLog(ctx)
    ok, _txn = esel.commit_field_edit(log, None, [], entities, "sensor_1",
                                      "sample_dx", 2, TILE_SIZE_M)
    assert ok
    assert "sample_dx" in entities[0].authored_keys
    assert entities[0].fields["sample_dx"] == 2


def test_commit_field_edit_light_field():
    lights = [_light("light_1", 1.0, 1.0)]
    ctx = undo_log.UndoContext(grids={}, collections={"lights": lights,
                                                      "entities": []})
    log = undo_log.TransactionLog(ctx)
    ok, txn = esel.commit_field_edit(log, None, lights, [], "light_1",
                                     "intensity", 2.5, TILE_SIZE_M)
    assert ok and txn is not None
    assert lights[0].intensity == 2.5
    log.undo()
    assert lights[0].intensity == 1.0


def test_commit_field_edit_door_length_restamps_grid_and_undo_restores():
    grid = _wall_row()
    d = _door("door_1", 2, 3, n_tiles=3)
    _stamp_door(grid, d)
    entities = [d]
    ctx = undo_log.UndoContext(grids={"material": grid},
                               collections={"lights": [], "entities": entities})
    log = undo_log.TransactionLog(ctx)
    grid_before = grid.copy()

    new_len = dep.length_m_for_tiles(5, TILE_SIZE_M)
    ok, txn = esel.commit_field_edit(log, grid, [], entities, "door_1",
                                     "length_m", new_len, TILE_SIZE_M)
    assert ok and txn is not None and len(txn.ops) == 2
    assert (grid[3, 2:7] == MAT_DOOR_CLOSED).all()

    log.undo()
    assert np.array_equal(grid, grid_before)
    assert entities[0].fields["length_m"] != new_len


def test_commit_field_edit_id_refuses_duplicate_across_lights_and_entities():
    lights = [_light("light_1", 1, 1)]
    entities = [_generic("sensor_1", "pressure", 2, 2)]
    ctx = undo_log.UndoContext(
        grids={}, collections={"lights": lights, "entities": entities})
    log = undo_log.TransactionLog(ctx)

    ok, reason = esel.commit_field_edit(log, None, lights, entities,
                                        "sensor_1", "id", "light_1",
                                        TILE_SIZE_M)
    assert not ok and "already in use" in reason
    assert entities[0].id == "sensor_1"
    assert log.undo_count == 0

    ok2, txn = esel.commit_field_edit(log, None, lights, entities,
                                      "sensor_1", "id", "sensor_renamed",
                                      TILE_SIZE_M)
    assert ok2 and txn is not None
    assert entities[0].id == "sensor_renamed"
    log.undo()
    assert entities[0].id == "sensor_1"


def test_commit_field_edit_no_such_instance_refuses():
    log = undo_log.TransactionLog(undo_log.UndoContext())
    ok, reason = esel.commit_field_edit(log, None, [], [], "ghost", "x", 1,
                                        TILE_SIZE_M)
    assert not ok and "ghost" in reason
    assert log.undo_count == 0


# ---------------------------------------------------------------------------
# Clump copy/paste
# ---------------------------------------------------------------------------

def test_compute_clump_copy_keeps_only_internal_non_tag_wires():
    lights = [_light("light_1", 1.5, 1.5)]
    entities = [_generic("sensor_1", "pressure", 2, 2),
               _generic("sensor_2", "pressure", 3, 3)]
    wires = [
        WireSpec("sensor_1.value", "sensor_2.threshold"),   # internal
        WireSpec("sensor_1.value", "sensor_3.threshold"),   # external (id)
        WireSpec("sensor_1.value", "tag:group.input"),      # tag -> external
    ]
    clump = esel.compute_clump_copy(["light_1", "sensor_1", "sensor_2"],
                                    lights, entities, wires)
    assert [l.id for l in clump["lights"]] == ["light_1"]
    assert {e.id for e in clump["entities"]} == {"sensor_1", "sensor_2"}
    assert len(clump["wires"]) == 1
    assert clump["wires"][0].from_ == "sensor_1.value"
    assert clump["wires"][0].to == "sensor_2.threshold"


def test_clump_anchor_tile_is_bbox_top_left_or_none_when_positionless():
    lights = [_light("light_1", 5.9, 2.5)]
    entities = [_generic("sensor_1", "pressure", 2, 8)]
    clump = {"lights": lights, "entities": entities, "wires": []}
    assert esel.clump_anchor_tile(clump) == (2, 2)
    empty = {"lights": [], "entities": [_positionless("dec_1")], "wires": []}
    assert esel.clump_anchor_tile(empty) is None


def test_commit_clump_paste_reids_reroutes_wires_and_stamps_door_atomically():
    grid = _wall_row()
    d = _door("door_1", 2, 3, n_tiles=3)
    _stamp_door(grid, d)
    sens = _generic("sensor_1", "pressure", 6, 3)
    lights = [_light("light_1", 6.5, 3.5)]
    entities = [d, sens]
    wires = [WireSpec("sensor_1.value", "door_1.close")]   # internal
    clump = esel.compute_clump_copy(["door_1", "sensor_1", "light_1"],
                                    lights, entities, wires)
    assert len(clump["wires"]) == 1

    ctx = undo_log.UndoContext(
        grids={"material": grid},
        collections={"lights": lights, "entities": entities, "wires": wires})
    log = undo_log.TransactionLog(ctx)
    grid_before = grid.copy()

    txn, new_ids = esel.commit_clump_paste(log, grid, lights, entities,
                                           wires, clump, 10, 0, TILE_SIZE_M)
    assert txn is not None and len(new_ids) == 3
    assert len(lights) == 2 and len(entities) == 4 and len(wires) == 2

    all_ids = [l.id for l in lights] + [e.id for e in entities]
    assert len(all_ids) == len(set(all_ids))          # no duplicate ids
    assert set(new_ids) <= set(all_ids)
    assert "door_1" in all_ids and "sensor_1" in all_ids and "light_1" in all_ids

    new_door = next(e for e in entities
                    if e.class_name == "door" and e.id != "door_1")
    new_sensor = next(e for e in entities
                      if e.class_name == "pressure" and e.id != "sensor_1")
    new_wire = wires[1]
    assert new_wire.from_ == f"{new_sensor.id}.value"
    assert new_wire.to == f"{new_door.id}.close"
    # the pasted door's span (anchor x=2 -> 12, 3 tiles) is stamped
    assert (grid[3, 12:15] == MAT_DOOR_CLOSED).all()

    log.undo()
    assert np.array_equal(grid, grid_before)
    assert len(lights) == 1 and len(entities) == 2 and len(wires) == 1
    log.redo()
    assert len(lights) == 2 and len(entities) == 4 and len(wires) == 2


def test_commit_clump_paste_drops_tag_target_wire():
    sens = _generic("sensor_1", "pressure", 6, 3)
    entities = [sens]
    wires = [WireSpec("sensor_1.value", "tag:group.input")]
    clump = esel.compute_clump_copy(["sensor_1"], [], entities, wires)
    assert clump["wires"] == []            # dropped at COPY time already

    grid = _wall_row()
    ctx = undo_log.UndoContext(
        grids={"material": grid},
        collections={"lights": [], "entities": entities, "wires": wires})
    log = undo_log.TransactionLog(ctx)
    txn, new_ids = esel.commit_clump_paste(log, grid, [], entities, wires,
                                           clump, 5, 0, TILE_SIZE_M)
    assert txn is not None and len(new_ids) == 1
    assert len(wires) == 1                 # only the ORIGINAL tag wire


def test_commit_clump_paste_empty_clump_is_noop():
    log = undo_log.TransactionLog(undo_log.UndoContext(
        grids={"material": _wall_row()},
        collections={"lights": [], "entities": [], "wires": []}))
    txn, new_ids = esel.commit_clump_paste(
        log, _wall_row(), [], [], [], {"lights": [], "entities": [],
                                       "wires": []}, 1, 1, TILE_SIZE_M)
    assert txn is None and new_ids == []
    assert log.undo_count == 0


# ---------------------------------------------------------------------------
# Integration: level_lib round-trip — copy -> paste -> save -> reload, and
# the wire-data load+save round-trip the C4 gate calls for.
# ---------------------------------------------------------------------------

def _write_png(path: Path, w: int = 8, h: int = 6) -> None:
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
SUFFIX = ('[art.bare]\ndiffuse = "diffuse.png"\n\n'
          '[bake]\ntileset = "x"\npx_per_tile = 8\nseed = 0\n')
DOOR_A = ('[[entity]]\nid = "door_1"\nclass = "door"\n'
         'x = 0\ny = 3\norientation = "h"\nlength_m = 1.0\n'
         'initial_state = "closed"\n\n')
SENSOR_A = ('[[entity]]\nid = "sensor_1"\nclass = "pressure"\n'
           'x = 6\ny = 3\n\n')
WIRE_A = ('[[wire]]\nfrom = "sensor_1.value"\nto = "door_1.close"\n\n')


def _mini_level(tmp_path: Path, body: str = "", name: str = "mini") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(10)) for _ in range(8)) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(PREFIX + body + SUFFIX,
                                  encoding="utf-8", newline="\n")
    return d


def test_wire_data_load_and_save_round_trip(tmp_path):
    """The editor's `wires` plumbing: `lvl.wire_specs` loads into a plain
    list, `level_lib.format_wire_lines` writes it back unconditionally on
    save (C4's own save-orchestration rule, mirroring C3's entity-family
    fix) — an untouched wire list round-trips byte-identically."""
    import level_lib
    d = _mini_level(tmp_path, DOOR_A + SENSOR_A + WIRE_A)
    handle = level_lib.open_level(str(d))
    wires = list(handle.data.wire_specs)
    assert [(w.from_, w.to) for w in wires] == [
        ("sensor_1.value", "door_1.close")]

    handle.save({"wire": lambda nl: level_lib.format_wire_lines(wires, nl)})
    reloaded = level_lib.open_level(str(d))
    assert [(w.from_, w.to) for w in reloaded.data.wire_specs] == [
        ("sensor_1.value", "door_1.close")]


def test_clump_copy_paste_save_reload_round_trip(tmp_path):
    """Copy the door+sensor+wire clump, paste it offset, save through
    level_lib (entity + wire families), and reload: the pasted clump's ids
    are unique, its internal wire is re-pointed, and the level loads
    without error (the gate's own explicit round-trip requirement)."""
    import level_lib
    d = _mini_level(tmp_path, DOOR_A + SENSOR_A + WIRE_A)
    handle = level_lib.open_level(str(d))
    lvl = handle.data

    entities = [e for e in lvl.entities
               if e.class_name != lep.LIGHT_CLASS]
    lights = lep.initial_editable_lights(lvl)
    wires = list(lvl.wire_specs)
    grid = np.array(lvl.tilemap, dtype=np.int32, copy=True)
    _stamp_door(grid, next(e for e in entities if e.class_name == "door"))

    clump = esel.compute_clump_copy(["door_1", "sensor_1"], lights, entities,
                                    wires)
    ctx = undo_log.UndoContext(
        grids={"material": grid},
        collections={"lights": lights, "entities": entities, "wires": wires})
    log = undo_log.TransactionLog(ctx)
    txn, new_ids = esel.commit_clump_paste(log, grid, lights, entities,
                                           wires, clump, 3, 2, TILE_SIZE_M)
    assert txn is not None and len(new_ids) == 2
    assert len(entities) == 4 and len(wires) == 2

    all_ids = [l.id for l in lights] + [e.id for e in entities]
    assert len(all_ids) == len(set(all_ids))

    replacements = {"entity": lambda nl: level_lib.format_entity_lines(
        entities, nl),
        "wire": lambda nl: level_lib.format_wire_lines(wires, nl)}
    handle.save(replacements)

    reloaded = level_lib.open_level(str(d))
    assert len(reloaded.data.entities) == 4
    assert len(reloaded.data.wire_specs) == 2
    reloaded_ids = [e.id for e in reloaded.data.entities]
    assert len(reloaded_ids) == len(set(reloaded_ids))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
