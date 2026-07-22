"""tools/sensor_entity_port.py — sensor placement bridge (Arc C3).

Pins:
  - is_field_sensor / sample_family: the six field-sensor classes split AIR
    (pressure/smoke/water_depth/o2) vs BODY (temperature/fire) exactly as
    simulation.entities.sensors declares (D7/D8) — clock/sensor_motion are
    NOT field sensors (no resolve_sample_tile, no arrow/refusal concept);
  - resolve_sample_tile delegates to the class's OWN classmethod (AIR faces
    the mount + offset, BODY always samples the mount, ignoring the offset);
  - tile_is_open: solid-for-flow (permeability<=0, D6) AND SPACE both read
    "not open" — a placement must refuse both;
  - commit_sensor_placement is ONE CollectionOp("entities") transaction
    (no grid op — sensors don't stamp material) whose undo reverts it;
  - the "validate BEFORE mutate" contract: a refused placement (solid
    sample tile) never calls commit_sensor_placement, so the log stays
    untouched (mirrors map_editor.py's ENTITY-mode dispatch order).

Run:
    python -m pytest tests/test_sensor_entity_port.py -q
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

import sensor_entity_port as sep  # noqa: E402
import undo_log  # noqa: E402
from level_loader import SPACE_CODE  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

SOLID_CODES = frozenset({MAT_HULL})


def _grid(width=8, height=8):
    return np.full((height, width), MAT_AIR, dtype=np.int32)


# ---------------------------------------------------------------------------
# is_field_sensor / sample_family — the AIR vs BODY split (D7/D8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["pressure", "smoke", "water_depth", "o2",
                                  "temperature", "fire"])
def test_is_field_sensor_true_for_the_six_field_sensors(name):
    assert sep.is_field_sensor(name)


@pytest.mark.parametrize("name", ["clock", "sensor_motion", "door", "light",
                                  "decider", "not_a_class"])
def test_is_field_sensor_false_for_everything_else(name):
    assert not sep.is_field_sensor(name)


@pytest.mark.parametrize("name", ["pressure", "smoke", "water_depth", "o2"])
def test_sample_family_air_for_the_air_family(name):
    assert sep.sample_family(name) == sep.SAMPLE_AIR


@pytest.mark.parametrize("name", ["temperature", "fire"])
def test_sample_family_body_for_the_body_family(name):
    assert sep.sample_family(name) == sep.SAMPLE_BODY


def test_sample_family_none_for_a_non_field_sensor():
    assert sep.sample_family("clock") is None
    assert sep.sample_family("sensor_motion") is None


# ---------------------------------------------------------------------------
# resolve_sample_tile — AIR faces the offset, BODY samples the mount
# ---------------------------------------------------------------------------

def test_resolve_sample_tile_air_family_faces_the_offset():
    fields = {"x": 5, "y": 4, "sample_dx": 1, "sample_dy": -1}
    assert sep.resolve_sample_tile("pressure", fields) == (3, 6)   # (y+dy, x+dx)


def test_resolve_sample_tile_body_family_ignores_the_offset():
    fields = {"x": 5, "y": 4, "sample_dx": 3, "sample_dy": -2}
    assert sep.resolve_sample_tile("temperature", fields) == (4, 5)   # (y, x)
    assert sep.resolve_sample_tile("fire", fields) == (4, 5)


def test_resolve_sample_tile_none_for_a_non_field_sensor():
    assert sep.resolve_sample_tile("clock", {"x": 0, "y": 0}) is None


# ---------------------------------------------------------------------------
# tile_is_open — the D6 frozen "solid" channel + SPACE
# ---------------------------------------------------------------------------

def test_tile_is_open_true_on_plain_air():
    grid = _grid()
    assert sep.tile_is_open(grid, (3, 3), SOLID_CODES, SPACE_CODE)


def test_tile_is_open_false_on_solid():
    grid = _grid()
    grid[3, 3] = MAT_HULL
    assert not sep.tile_is_open(grid, (3, 3), SOLID_CODES, SPACE_CODE)


def test_tile_is_open_false_on_space():
    grid = _grid()
    grid[3, 3] = SPACE_CODE
    assert not sep.tile_is_open(grid, (3, 3), SOLID_CODES, SPACE_CODE)


def test_tile_is_open_false_out_of_bounds():
    grid = _grid()
    assert not sep.tile_is_open(grid, (-1, 0), SOLID_CODES, SPACE_CODE)
    assert not sep.tile_is_open(grid, (0, 99), SOLID_CODES, SPACE_CODE)


def test_tile_is_open_false_for_an_unresolvable_tile():
    grid = _grid()
    assert not sep.tile_is_open(grid, None, SOLID_CODES, SPACE_CODE)


# ---------------------------------------------------------------------------
# build_sensor_instance — authored_keys omits zero offsets (never a default)
# ---------------------------------------------------------------------------

def test_build_sensor_instance_omits_zero_offset():
    inst = sep.build_sensor_instance("temperature", 3, 4, 0, 0, "temp_1")
    assert inst.fields == {"x": 3, "y": 4, "sample_dx": 0, "sample_dy": 0}
    assert inst.authored_keys == ("x", "y")


def test_build_sensor_instance_authors_a_nonzero_offset():
    inst = sep.build_sensor_instance("pressure", 3, 4, 1, 0, "pressure_1")
    assert set(inst.authored_keys) == {"x", "y", "sample_dx"}
    inst2 = sep.build_sensor_instance("pressure", 3, 4, 0, -1, "pressure_2")
    assert set(inst2.authored_keys) == {"x", "y", "sample_dy"}


# ---------------------------------------------------------------------------
# commit_sensor_placement — ONE CollectionOp, atomic undo, no grid touched
# ---------------------------------------------------------------------------

def test_commit_sensor_placement_is_entities_only_and_reverts_on_undo():
    entities = []
    ctx = undo_log.UndoContext(collections={"entities": entities})
    log = undo_log.TransactionLog(ctx)
    inst = sep.build_sensor_instance("pressure", 3, 4, 1, 0, "pressure_1")

    txn = sep.commit_sensor_placement(log, entities, inst)
    assert len(txn.ops) == 1                       # entities only, no grid op
    assert entities == [inst]

    log.undo()
    assert entities == []
    log.redo()
    assert entities == [inst]


def test_refused_placement_never_opens_a_transaction():
    """The C3 contract (§6.2): validate BEFORE the first mutation. This
    mirrors map_editor.py's ENTITY-mode dispatch — resolve + tile_is_open
    FIRST, only call commit_sensor_placement when it passes."""
    grid = _grid()
    grid[3, 3] = MAT_HULL                          # mount tile — solid
    entities = []
    ctx = undo_log.UndoContext(collections={"entities": entities})
    log = undo_log.TransactionLog(ctx)

    fields = {"x": 3, "y": 3, "sample_dx": 0, "sample_dy": 0}
    sample_tile = sep.resolve_sample_tile("pressure", fields)   # == mount
    assert sample_tile == (3, 3)
    if sep.tile_is_open(grid, sample_tile, SOLID_CODES, SPACE_CODE):
        sep.commit_sensor_placement(
            log, entities, sep.build_sensor_instance(
                "pressure", 3, 3, 0, 0, "pressure_1"))

    assert entities == []
    assert log.undo_count == 0
    assert log.undo() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
