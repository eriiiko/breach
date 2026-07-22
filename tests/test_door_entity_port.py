"""tools/door_entity_port.py — the DOOR tool's entity-authoring bridge
(Arc C3: span drag + immediate MAT_DOOR_CLOSED/MAT_AIR stamp).

Pins:
  - span-quantization PARITY with simulation.entities.door — the editor
    never re-derives the span rule, it calls door.base_span /
    door.quantize_span_tiles directly, so length_m_for_tiles round-trips
    through the CANONICAL quantizer for several lengths, including the
    editor doc §4 exact-tie / re-derive-at-scaled-resolution trap;
  - door_anchor_check: straight-run acceptance/refusal + orientation,
    refusing an existing MAT_DOOR_CLOSED span (never grow into another
    door);
  - plan_door_span: default length on a plain click, forward drag-resize,
    clipped at the wall run's end / an existing door / the grid edge;
  - commit_door_placement is ONE compound transaction (GridCellsOp +
    CollectionOp) whose undo reverts BOTH atomically.

Run:
    python -m pytest tests/test_door_entity_port.py -q
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

import door_entity_port as dep  # noqa: E402
import undo_log  # noqa: E402
from simulation.entities import door as door_entity  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR_CLOSED, MAT_HULL  # noqa: E402

TILE_SIZE_M = 0.333          # the shipped level default -> tiles_per_m == 3
WALL_CODES = frozenset({MAT_HULL})


def _wall_row(width=10, height=8, y=3, x0=1, x1=8):
    """An AIR field with one horizontal MAT_HULL wall run at row y."""
    grid = np.full((height, width), MAT_AIR, dtype=np.int32)
    grid[y, x0:x1] = MAT_HULL
    return grid


def _wall_col(width=8, height=10, x=3, y0=1, y1=8):
    grid = np.full((height, width), MAT_AIR, dtype=np.int32)
    grid[y0:y1, x] = MAT_HULL
    return grid


# ---------------------------------------------------------------------------
# Span quantization parity — never re-derive, always door.py's own function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 9])
def test_length_m_for_tiles_round_trips_through_door_quantizer(n):
    lm = dep.length_m_for_tiles(n, TILE_SIZE_M)
    assert door_entity.quantize_span_tiles(lm, TILE_SIZE_M) == n


def test_default_tile_count_is_the_1m_default():
    # editor doc §4/§6: 1.0 m at tiles_per_m=3 -> 3 tiles (a marine footprint
    # doorway); matches door.quantize_span_tiles(1.0, 0.333) directly.
    assert dep.default_tile_count(TILE_SIZE_M) == 3
    assert (dep.default_tile_count(TILE_SIZE_M)
            == door_entity.quantize_span_tiles(1.0, TILE_SIZE_M))


def test_editor_doc_4_quantize_once_then_replicate_not_rederive_trap():
    """0.5 m: quantize ONCE at base res (2 tiles), then REPLICATE by the
    integer --res factor (2 tiles * 2 = 4) — never re-derive from meters at
    the scaled resolution (round(0.5*6)=3 != round(0.5*3)*2=4). Re-deriving
    at the scaled tile_size_m for a --res 2 level hard-errors instead (a6
    doors design §3 S1: tiles_per_m must be an exact integer)."""
    base_n = door_entity.quantize_span_tiles(0.5, TILE_SIZE_M)
    assert base_n == 2
    replicated_n = base_n * 2
    assert replicated_n == 4
    with pytest.raises(ValueError):
        door_entity.quantize_span_tiles(0.5, TILE_SIZE_M / 2)


def test_door_tool_availability_ok_for_the_shipped_tile_size():
    assert dep.door_tool_availability(TILE_SIZE_M) is None


def test_door_tool_availability_reports_non_integral_tiles_per_m():
    err = dep.door_tool_availability(0.15)     # 1/0.15 = 6.667 — non-integral
    assert err is not None and "tiles-per-meter" in err


# ---------------------------------------------------------------------------
# door_anchor_check — straight-run acceptance / refusal + orientation
# ---------------------------------------------------------------------------

def test_anchor_check_accepts_a_horizontal_run_with_h_orientation():
    grid = _wall_row()
    ok, orientation, why = dep.door_anchor_check(grid, 4, 3, WALL_CODES)
    assert ok and orientation == "h" and "horizontal" in why


def test_anchor_check_accepts_a_vertical_run_with_v_orientation():
    grid = _wall_col()
    ok, orientation, why = dep.door_anchor_check(grid, 3, 4, WALL_CODES)
    assert ok and orientation == "v" and "vertical" in why


def test_anchor_check_refuses_corner_end_isolated_and_non_wall():
    grid = _wall_row()
    assert dep.door_anchor_check(grid, 4, 4, WALL_CODES) == (
        False, None, "not a wall tile")
    assert dep.door_anchor_check(grid, 1, 3, WALL_CODES)[0] is False  # end
    assert dep.door_anchor_check(grid, -1, 3, WALL_CODES) == (
        False, None, "outside the grid")
    isolated = np.full((6, 6), MAT_AIR, dtype=np.int32)
    isolated[2, 2] = MAT_HULL
    assert dep.door_anchor_check(isolated, 2, 2, WALL_CODES) == (
        False, None, "isolated wall tile")


def test_anchor_check_refuses_an_existing_door_span():
    grid = _wall_row()
    grid[3, 4] = MAT_DOOR_CLOSED
    assert dep.door_anchor_check(grid, 4, 3, WALL_CODES) == (
        False, None, "already a door")


# ---------------------------------------------------------------------------
# plan_door_span — default length / drag-resize / clipping
# ---------------------------------------------------------------------------

def test_plan_door_span_plain_click_uses_default_3_tiles():
    grid = _wall_row()
    span, length_m = dep.plan_door_span(grid, (2, 3), "h", 2, TILE_SIZE_M,
                                        WALL_CODES)
    assert span == [(3, 2), (3, 3), (3, 4)]
    assert door_entity.quantize_span_tiles(length_m, TILE_SIZE_M) == 3


def test_plan_door_span_drag_extends_forward():
    grid = _wall_row()
    span, length_m = dep.plan_door_span(grid, (2, 3), "h", 5, TILE_SIZE_M,
                                        WALL_CODES)
    assert span == [(3, 2), (3, 3), (3, 4), (3, 5)]
    assert len(span) == 4


def test_plan_door_span_clips_at_the_wall_runs_end():
    """A 7-tile wall run (x0=1..7): default 3 tiles from x=6 would want
    x=6,7,8 but 8 is AIR — the walk stops at the run's own end."""
    grid = _wall_row(x0=1, x1=8)   # wall tiles at columns 1..7
    span, length_m = dep.plan_door_span(grid, (6, 3), "h", 6, TILE_SIZE_M,
                                        WALL_CODES)
    assert span == [(3, 6), (3, 7)]
    assert door_entity.quantize_span_tiles(length_m, TILE_SIZE_M) == 2


def test_plan_door_span_never_absorbs_an_existing_door():
    grid = _wall_row()
    grid[3, 5] = MAT_DOOR_CLOSED
    span, _lm = dep.plan_door_span(grid, (2, 3), "h", 6, TILE_SIZE_M,
                                   WALL_CODES)
    assert (3, 5) not in span
    assert span == [(3, 2), (3, 3), (3, 4)]


def test_plan_door_span_matches_door_base_span_exactly():
    """The parity pin: whatever plan_door_span stamps IS door.base_span's
    own output for the same fields — not a parallel calculation."""
    grid = _wall_row()
    span, length_m = dep.plan_door_span(grid, (2, 3), "h", 4, TILE_SIZE_M,
                                        WALL_CODES)
    canonical = door_entity.base_span(
        {"x": 2, "y": 3, "orientation": "h", "length_m": length_m},
        TILE_SIZE_M)
    assert span == canonical


# ---------------------------------------------------------------------------
# build_door_instance / stamp_value_for
# ---------------------------------------------------------------------------

def test_build_door_instance_authors_all_five_fields():
    inst = dep.build_door_instance(2, 3, "h", 1.0, "closed", "door_1")
    assert inst.id == "door_1" and inst.class_name == "door"
    assert inst.fields == {"x": 2, "y": 3, "orientation": "h",
                           "length_m": 1.0, "initial_state": "closed"}
    assert set(inst.authored_keys) == set(inst.fields)


def test_stamp_value_for_closed_and_open():
    assert dep.stamp_value_for("closed") == MAT_DOOR_CLOSED
    assert dep.stamp_value_for("open") == MAT_AIR


# ---------------------------------------------------------------------------
# commit_door_placement — ONE compound transaction, atomic undo
# ---------------------------------------------------------------------------

def test_commit_door_placement_is_one_atomic_transaction():
    grid = _wall_row()
    entities = []
    ctx = undo_log.UndoContext(grids={"material": grid},
                               collections={"entities": entities})
    log = undo_log.TransactionLog(ctx)
    grid_before = grid.copy()

    span, length_m = dep.plan_door_span(grid, (2, 3), "h", 2, TILE_SIZE_M,
                                        WALL_CODES)
    instance = dep.build_door_instance(2, 3, "h", length_m, "closed",
                                       "door_1")
    txn = dep.commit_door_placement(log, grid, entities,
                                    span, MAT_DOOR_CLOSED, instance)

    assert len(txn.ops) == 2                        # grid + entity, ONE txn
    assert (grid[3, 2:5] == MAT_DOOR_CLOSED).all()
    assert entities == [instance]
    assert log.undo_count == 1

    log.undo()
    assert np.array_equal(grid, grid_before)          # grid REVERTED
    assert entities == []                             # entity REVERTED too

    log.redo()
    assert (grid[3, 2:5] == MAT_DOOR_CLOSED).all()
    assert entities == [instance]


def test_commit_door_placement_open_door_stamps_mat_air():
    grid = _wall_row()
    entities = []
    ctx = undo_log.UndoContext(grids={"material": grid},
                               collections={"entities": entities})
    log = undo_log.TransactionLog(ctx)
    span, length_m = dep.plan_door_span(grid, (2, 3), "h", 2, TILE_SIZE_M,
                                        WALL_CODES)
    instance = dep.build_door_instance(2, 3, "h", length_m, "open", "door_1")
    dep.commit_door_placement(log, grid, entities, span,
                              dep.stamp_value_for("open"), instance)
    assert (grid[3, 2:5] == MAT_AIR).all()


# ---------------------------------------------------------------------------
# MAT_DOOR_CLOSED-outside-a-span validator (Arc C9 rider, canon §9)
# ---------------------------------------------------------------------------

def _door_at(x, y, orientation, n_tiles, id_="door_1"):
    length_m = dep.length_m_for_tiles(n_tiles, TILE_SIZE_M)
    return dep.build_door_instance(x, y, orientation, length_m, "closed", id_)


def test_door_span_tiles_unions_every_door_entity():
    d1 = _door_at(2, 3, "h", 3, "door_1")           # (3,2) (3,3) (3,4)
    d2 = _door_at(5, 0, "v", 2, "door_2")            # (0,5) (1,5)
    tiles = dep.door_span_tiles([d1, d2], TILE_SIZE_M)
    assert tiles == {(2, 3), (3, 3), (4, 3), (5, 0), (5, 1)}


def test_door_span_tiles_ignores_non_door_entities():
    from level_loader import EntityInstance
    other = EntityInstance(id="s1", class_name="pressure", ordinal=0,
                           tags=(), fields={"x": 1, "y": 1},
                           authored_keys=("x", "y"))
    assert dep.door_span_tiles([other], TILE_SIZE_M) == set()


def test_orphaned_door_closed_tiles_empty_on_a_clean_level():
    grid = _wall_row()
    door = _door_at(2, 3, "h", 3)
    for (fy, fx) in dep.instance_span(door.fields, TILE_SIZE_M):
        grid[fy, fx] = MAT_DOOR_CLOSED
    assert dep.orphaned_door_closed_tiles(grid, [door], TILE_SIZE_M) == set()


def test_orphaned_door_closed_tiles_flags_a_stray_tile():
    """A MAT_DOOR_CLOSED tile with no owning door entity at all (e.g. the
    door was deleted without clearing its stamped span) is flagged."""
    grid = _wall_row()
    grid[3, 5] = MAT_DOOR_CLOSED
    assert dep.orphaned_door_closed_tiles(grid, [], TILE_SIZE_M) == {(5, 3)}


def test_orphaned_door_closed_tiles_flags_only_the_uncovered_part():
    """A door's span shrunk (move/inspector-edit) without re-stamping the
    vacated tile: the tile just outside the NEW (shorter) span still reads
    MAT_DOOR_CLOSED on the grid and must be flagged, while the tiles still
    inside the span must not be."""
    grid = _wall_row()
    grid[3, 2:5] = MAT_DOOR_CLOSED    # originally a 3-tile door...
    door = _door_at(2, 3, "h", 2)     # ...now authored as only 2 tiles
    assert dep.orphaned_door_closed_tiles(
        grid, [door], TILE_SIZE_M) == {(4, 3)}


def test_door_span_validator_summary_ok_and_warning():
    grid = _wall_row()
    door = _door_at(2, 3, "h", 3)
    for (fy, fx) in dep.instance_span(door.fields, TILE_SIZE_M):
        grid[fy, fx] = MAT_DOOR_CLOSED
    assert dep.door_span_validator_summary(grid, [door], TILE_SIZE_M) \
        == "doors ok"

    grid2 = _wall_row()
    grid2[3, 5] = MAT_DOOR_CLOSED
    msg = dep.door_span_validator_summary(grid2, [], TILE_SIZE_M)
    assert msg.startswith("WARNING")
    assert "(5,3)" in msg


def test_door_span_validator_summary_never_raises_on_an_empty_level():
    grid = np.zeros((4, 4), dtype=np.int32)
    assert dep.door_span_validator_summary(grid, [], TILE_SIZE_M) \
        == "doors ok"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
