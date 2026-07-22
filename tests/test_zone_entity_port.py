"""tools/zone_entity_port.py — the ZONE paint tool's entity-authoring
bridge (Arc C5, editor doc §5).

Pins:
  - zone_ids_in_use / next_zone_id / find_zone_claim: the id bookkeeping a
    ZONE-mode paint target needs, over a plain `entities` list;
  - build_zone_instance: every OTHER field at its registry default (via
    entity_editor_ui.default_instance_fields — the SAME generic template
    C3's place-one uses), `zone_id` overridden, `authored_keys` = just
    `zone_id`;
  - commit_zone_paint: ONE `GridCellsOp("zones")` transaction, PLUS a
    `CollectionOp("entities")` in the SAME transaction when a new instance
    is minted — the C3 DOOR compound archetype; painting an existing id
    over another id's tiles just shrinks the other id's count (no special
    code — a plain overwrite); undo reverts both atomically;
  - commit_zone_clear: ONE `GridCellsOp("zones")` transaction (the
    same-code-select "erase this zone's paint" gesture);
  - zone_binding_summary: runs level_loader's OWN §5 validators live
    (duplicate zone_id -> error text; zero-tile / orphaned-paint ->
    warning text; a clean level -> "zones ok") — never re-derives the
    rules.

Run:
    python -m pytest tests/test_zone_entity_port.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import entity_editor_ui  # noqa: E402
import undo_log  # noqa: E402
import zone_entity_port as zep  # noqa: E402
from level_loader import EntityInstance  # noqa: E402

GRID_SHAPE = (8, 8)


def _zone_entity(id_, zone_id, cls_name="breach_site", faction=0):
    fields = {"zone_id": zone_id, "faction": faction}
    if cls_name == "breach_site":
        fields["roster"] = []
    return EntityInstance(id=id_, class_name=cls_name, ordinal=0, tags=(),
                          fields=fields, authored_keys=("zone_id",))


@pytest.fixture(scope="module")
def registry_payload():
    result = entity_editor_ui.load_registry()
    assert result.ok, result.error
    return result.payload


# ---------------------------------------------------------------------------
# zone_ids_in_use / next_zone_id / find_zone_claim
# ---------------------------------------------------------------------------

def test_zone_ids_in_use_scans_both_zone_classes():
    entities = [_zone_entity("bs_1", 3),
               _zone_entity("ex_1", 7, "extraction_zone"),
               EntityInstance(id="sensor_1", class_name="pressure",
                              ordinal=0, tags=(), fields={"x": 1, "y": 1},
                              authored_keys=("x", "y"))]
    assert zep.zone_ids_in_use(entities) == {3, 7}


def test_next_zone_id_is_the_smallest_unclaimed():
    assert zep.next_zone_id([]) == 1
    entities = [_zone_entity("bs_1", 1), _zone_entity("bs_2", 2)]
    assert zep.next_zone_id(entities) == 3
    # a gap is reused before extending past the max in use
    entities = [_zone_entity("bs_1", 1), _zone_entity("bs_3", 3)]
    assert zep.next_zone_id(entities) == 2


def test_find_zone_claim_returns_the_claiming_instance_or_none():
    bs1 = _zone_entity("bs_1", 5)
    entities = [bs1, _zone_entity("bs_2", 6)]
    assert zep.find_zone_claim(entities, 5) is bs1
    assert zep.find_zone_claim(entities, 99) is None


# ---------------------------------------------------------------------------
# build_zone_instance
# ---------------------------------------------------------------------------

def test_build_zone_instance_breach_site_defaults_plus_zone_id(
        registry_payload):
    cls_payload = registry_payload["classes"]["breach_site"]
    inst = zep.build_zone_instance(cls_payload, "breach_site", 42, "bs_9")
    assert inst.id == "bs_9"
    assert inst.class_name == "breach_site"
    assert inst.fields["zone_id"] == 42
    assert inst.fields["faction"] == 0          # registry default
    assert inst.fields["roster"] == []           # registry default
    assert tuple(inst.authored_keys) == ("zone_id",)   # only the required one


def test_build_zone_instance_extraction_zone_has_no_roster_field(
        registry_payload):
    cls_payload = registry_payload["classes"]["extraction_zone"]
    inst = zep.build_zone_instance(cls_payload, "extraction_zone", 5, "ex_1")
    assert "roster" not in inst.fields
    assert inst.fields["zone_id"] == 5


# ---------------------------------------------------------------------------
# commit_zone_paint — new id (compound), existing id (grid-only), shrink-B
# ---------------------------------------------------------------------------

def _ctx_log(zones, entities):
    ctx = undo_log.UndoContext(grids={"zones": zones},
                               collections={"entities": entities})
    return undo_log.TransactionLog(ctx)


def test_commit_zone_paint_new_id_is_compound_grid_plus_entity():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    entities = []
    log = _ctx_log(zones, entities)
    region = {(1, 1), (2, 1), (1, 2), (2, 2)}
    new_inst = _zone_entity("bs_1", 1)

    txn = zep.commit_zone_paint(log, zones, entities, region, 1, new_inst)
    assert txn is not None
    assert len(txn.ops) == 2                     # GridCellsOp + CollectionOp
    assert (zones[1, 1], zones[2, 1], zones[1, 2], zones[2, 2]) == (1,) * 4
    assert entities == [new_inst]

    log.undo()
    assert not zones.any()
    assert entities == []
    log.redo()
    assert entities == [new_inst]
    assert (zones[1, 1], zones[2, 1], zones[1, 2], zones[2, 2]) == (1,) * 4


def test_commit_zone_paint_existing_id_is_grid_only():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    entities = [_zone_entity("bs_1", 1)]
    log = _ctx_log(zones, entities)
    region = {(3, 3), (4, 3)}

    txn = zep.commit_zone_paint(log, zones, entities, region, 1, None)
    assert txn is not None
    assert len(txn.ops) == 1                     # GridCellsOp only
    assert (zones[3, 3], zones[3, 4]) == (1, 1)
    assert entities == [entities[0]]              # untouched, same object


def test_commit_zone_paint_a_over_b_shrinks_b():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    zones[1:4, 1:4] = 2                          # zone 2 covers a 3x3 block
    entities = [_zone_entity("bs_1", 1), _zone_entity("bs_2", 2)]
    log = _ctx_log(zones, entities)
    # paint zone 1 over the LEFT column of zone 2's block
    region = {(1, y) for y in range(1, 4)}

    zep.commit_zone_paint(log, zones, entities, region, 1, None)
    assert int((zones == 1).sum()) == 3
    assert int((zones == 2).sum()) == 6           # shrank from 9 to 6
    assert [e.id for e in entities] == ["bs_1", "bs_2"]   # neither dropped


def test_commit_zone_paint_empty_region_is_noop():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    entities = []
    log = _ctx_log(zones, entities)
    assert zep.commit_zone_paint(log, zones, entities, set(), 1, None) is None
    assert log.undo_count == 0


# ---------------------------------------------------------------------------
# commit_zone_clear
# ---------------------------------------------------------------------------

def test_commit_zone_clear_zeroes_the_region_and_undoes():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    zones[1:3, 1:3] = 5
    log = undo_log.TransactionLog(
        undo_log.UndoContext(grids={"zones": zones}))
    zones_before = zones.copy()
    region = {(x, y) for y in range(1, 3) for x in range(1, 3)}

    txn = zep.commit_zone_clear(log, zones, region)
    assert txn is not None
    assert not zones.any()
    log.undo()
    assert np.array_equal(zones, zones_before)


def test_commit_zone_clear_empty_region_is_noop():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    log = undo_log.TransactionLog(
        undo_log.UndoContext(grids={"zones": zones}))
    assert zep.commit_zone_clear(log, zones, set()) is None


# ---------------------------------------------------------------------------
# zone_binding_summary — runs level_loader's OWN §5 validators live
# ---------------------------------------------------------------------------

def test_zone_binding_summary_ok_when_clean():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    zones[1, 1] = 1
    entities = [_zone_entity("bs_1", 1)]
    assert zep.zone_binding_summary(zones, entities) == "zones ok"


def test_zone_binding_summary_warns_on_zero_tile_instance():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)   # nothing painted
    entities = [_zone_entity("bs_1", 1)]
    summary = zep.zone_binding_summary(zones, entities)
    assert "0 painted" in summary or "zone_id 1" in summary


def test_zone_binding_summary_warns_on_orphaned_paint():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    zones[1, 1] = 9                                # painted, no claiming inst
    assert "orphaned" in zep.zone_binding_summary(zones, [])


def test_zone_binding_summary_reports_duplicate_zone_id_as_error():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    zones[1, 1] = 1
    entities = [_zone_entity("bs_1", 1), _zone_entity("bs_2", 1)]
    summary = zep.zone_binding_summary(zones, entities)
    assert summary.startswith("zone error:")
    assert "zone_id 1" in summary


def test_zone_binding_summary_none_zone_grid_is_the_absent_file_case():
    entities = [_zone_entity("bs_1", 1)]
    # zone_grid=None mirrors "no zones.npy at all" — still a zero-tile warn.
    summary = zep.zone_binding_summary(None, entities)
    assert "0 painted" in summary or "zone_id 1" in summary
