"""Transaction-log undo — tools/undo_log.py (Arc C2).

The oracle for the C2 design (docs/arc_c_c2_undo_design_2026-07-22.md §8):
pure, headless tests of the two op primitives + the Transaction + the
TransactionLog builder seam / linear cursor / saved-marker dirty tracking /
injectable-bound eviction, against plain numpy grids and dataclass lists (no
raylib). Mirrors the style of tests/test_editor_layout.py.

Coverage map to design §8: 1 op round-trip, 1b deep-copy IDENTITY (B1), 2
compound-op atomicity, 3 stroke coalescing, 3b whole-grid diff (B2), 4 redo
truncation, 5 saved-marker/dirty coherence, 6 global-history ordering, 7
memory-bound eviction + cursor renumbering (C3/C7) INCL. the NEW-1
second-eviction-after-None guard, 7b save-mask coherence (C1), 7c union id
allocator (B3), 8 fuzz round-trip.

Run:
    python -m pytest tests/test_editor_undo.py -q
"""
from __future__ import annotations

import copy
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from level_loader import EntityInstance, SPACE_CODE, SpawnEntry  # noqa: E402
import undo_log  # noqa: E402
from undo_log import (CollectionOp, GridCellsOp, Transaction,  # noqa: E402
                      TransactionLog, UndoContext)
from light_entity_port import EditableLight, unique_entity_id  # noqa: E402
from map_editor import (apply_corridor, water_fill_region,  # noqa: E402
                        water_solid_codes)
from simulation.materials import MAT_AIR, MAT_HULL, MAT_STEEL  # noqa: E402
from simulation import water_fixed  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def make_ctx():
    """A fresh ctx over two grids + three collections, all independent."""
    grids = {
        "material": np.full((12, 16), MAT_AIR, dtype=np.int32),
        "water": np.zeros((12, 16), dtype=np.int32),
    }
    colls = {
        "spawns": [SpawnEntry("marine_1", 0, 2.0, 3.0)],
        "lights": [EditableLight(x=1.5, y=1.5, color=(1.0, 0.0, 0.0),
                                 id="light_1")],
        "entities": [EntityInstance(id="door_1", class_name="door", ordinal=0,
                                    tags=("north",), fields={"span": 1})],
    }
    return UndoContext(grids=grids, collections=colls)


def edit_cell(log, ys, xs, vals, grid="material", label="paint"):
    """One committed grid gesture: snapshot, set cells, commit."""
    log.begin(label)
    log.snapshot_grid(grid)
    log.ctx.grids[grid][ys, xs] = vals
    return log.commit()


# ---------------------------------------------------------------------------
# 1 — op round-trip, each class
# ---------------------------------------------------------------------------

def test_gridcellsop_roundtrip_material_and_water():
    for gname, val in (("material", MAT_HULL),
                       ("water", int(water_fixed.quantize(1.0)))):
        ctx = make_ctx()
        before = ctx.grids[gname].copy()
        ys = np.array([1, 5, 8], np.int32)
        xs = np.array([2, 7, 3], np.int32)
        after_vals = np.array([val, val, val], ctx.grids[gname].dtype)
        op = GridCellsOp(gname, ys, xs, ctx.grids[gname][ys, xs].copy(),
                         after_vals)
        op.redo(ctx)
        assert np.array_equal(ctx.grids[gname][ys, xs], after_vals)
        op.undo(ctx)
        assert np.array_equal(ctx.grids[gname], before)


def test_collectionop_roundtrip_preserves_list_identity():
    ctx = make_ctx()
    for cname in ("spawns", "lights", "entities"):
        live = ctx.collections[cname]
        orig = copy.deepcopy(live)
        ident = id(live)
        after = copy.deepcopy(live) + copy.deepcopy(live)   # doubled list
        op = CollectionOp(cname, copy.deepcopy(orig), copy.deepcopy(after))
        op.redo(ctx)
        assert ctx.collections[cname] == after
        assert id(ctx.collections[cname]) == ident          # slice-assign
        op.undo(ctx)
        assert ctx.collections[cname] == orig
        assert id(ctx.collections[cname]) == ident


def test_gridcellsop_bounding_rect_and_nbytes():
    ys = np.array([2, 4], np.int32)
    xs = np.array([3, 9], np.int32)
    op = GridCellsOp("material", ys, xs, np.array([0, 0], np.int32),
                     np.array([1, 1], np.int32))
    assert op.bounding_rect() == (3, 2, 7, 3)               # x0,y0,tw,th
    assert op.nbytes == 4 * 2 * 4                            # 4 arrays x 2 x 4B
    empty = GridCellsOp("material", np.array([], np.int32),
                        np.array([], np.int32), np.array([], np.int32),
                        np.array([], np.int32))
    assert empty.bounding_rect() is None


# ---------------------------------------------------------------------------
# 1b — deep-copy IDENTITY (B1)
# ---------------------------------------------------------------------------

def test_collection_deepcopy_identity_independence():
    """A restored element's mutable substructure (`fields`/`tags`) must be
    an independent object from the op's retained snapshot — else undo returns
    an instance aliasing the snapshot's dict."""
    ctx = make_ctx()
    log = TransactionLog(ctx)
    log.begin("edit")
    log.snapshot_coll("entities")
    # inspector-style in-place field mutation (the aliasing-drop bug case)
    ctx.collections["entities"][0].fields["span"] = 3
    txn = log.commit()
    assert txn is not None and len(txn.ops) == 1             # deep compare saw it
    op = txn.ops[0]

    log.undo()
    restored = ctx.collections["entities"][0]
    assert restored.fields["span"] == 1                     # before-state
    # Mutating the restored element must NOT touch the op's snapshots.
    restored.fields["span"] = 999
    restored.fields["injected"] = True
    assert op.before[0].fields == {"span": 1}
    assert op.after[0].fields == {"span": 3}


def test_inplace_fields_edit_commits_nonempty_op():
    """Regression pin for B1: a change to a key INSIDE `.fields` is a real
    edit; a shallow copy would alias it (before==after -> dropped)."""
    ctx = make_ctx()
    log = TransactionLog(ctx)
    log.begin("edit")
    log.snapshot_coll("entities")
    ctx.collections["entities"][0].tags = ("north", "south")   # tuple swap
    txn = log.commit()
    assert txn is not None                                    # not dropped


def test_abort_reverts_live_state_and_drops_pending():
    """abort() (§2.1/§2.4) reverts every snapshotted grid + collection from
    the retained before-copy and drops the pending transaction — a true no-op
    regardless of how far the gesture had mutated."""
    ctx = make_ctx()
    log = TransactionLog(ctx)
    grid0 = ctx.grids["material"].copy()
    ents0 = copy.deepcopy(ctx.collections["entities"])
    log.begin("half-gesture")
    log.snapshot_grid("material")
    log.snapshot_coll("entities")
    ctx.grids["material"][4, 4] = MAT_HULL
    ctx.collections["entities"].append(
        EntityInstance(id="x", class_name="door", ordinal=9))
    log.abort()
    assert np.array_equal(ctx.grids["material"], grid0)
    assert ctx.collections["entities"] == ents0
    assert log.has_pending is False
    assert len(log.txns) == 0                                # nothing committed


# ---------------------------------------------------------------------------
# 2 — compound-op atomicity (the C3 door archetype)
# ---------------------------------------------------------------------------

def test_compound_transaction_atomic_grid_plus_entity():
    ctx = make_ctx()
    log = TransactionLog(ctx)
    grid0 = ctx.grids["material"].copy()
    ents0 = copy.deepcopy(ctx.collections["entities"])

    # A "door": stamp a material cell AND add a door entity, one action.
    log.begin("door")
    log.snapshot_grid("material")
    log.snapshot_coll("entities")
    ctx.grids["material"][5, 6] = MAT_HULL
    ctx.collections["entities"].append(
        EntityInstance(id="door_2", class_name="door", ordinal=1, tags=(),
                       fields={"span": 2}))
    txn = log.commit()
    assert len(txn.ops) == 2
    grid1 = ctx.grids["material"].copy()
    ents1 = copy.deepcopy(ctx.collections["entities"])

    log.undo()
    assert np.array_equal(ctx.grids["material"], grid0)      # BOTH reverted
    assert ctx.collections["entities"] == ents0
    log.redo()
    assert np.array_equal(ctx.grids["material"], grid1)      # BOTH re-applied
    assert ctx.collections["entities"] == ents1


# ---------------------------------------------------------------------------
# 3 — stroke coalescing
# ---------------------------------------------------------------------------

def test_stroke_coalesces_intermediate_values():
    """A stroke that paints one cell A->B->C stores before=A, after=C in ONE
    op — intermediate B never enters the transaction."""
    ctx = make_ctx()
    log = TransactionLog(ctx)
    ctx.grids["material"][3, 3] = MAT_AIR                    # A
    log.begin("paint")
    log.snapshot_grid("material")
    ctx.grids["material"][3, 3] = MAT_HULL                   # B (intermediate)
    ctx.grids["material"][3, 3] = MAT_STEEL                  # C (final)
    txn = log.commit()
    assert len(txn.ops) == 1
    op = txn.ops[0]
    assert op.before.tolist() == [MAT_AIR] and op.after.tolist() == [MAT_STEEL]


def test_noop_stroke_commits_nothing():
    ctx = make_ctx()
    log = TransactionLog(ctx)
    log.begin("paint")
    log.snapshot_grid("material")
    # paint X over the existing X — no change
    ctx.grids["material"][3, 3] = MAT_AIR
    assert log.commit() is None
    assert len(log.txns) == 0


# ---------------------------------------------------------------------------
# 3b — whole-grid diff captures out-of-bbox cells (B2)
# ---------------------------------------------------------------------------

def test_corridor_wholegrid_diff_roundtrips():
    """CORRIDOR lines walls a tile beyond the drag-line bbox — a rect-bounded
    diff would drop them. The whole-grid diff captures every changed cell."""
    ctx = make_ctx()
    ctx.grids["material"][:] = SPACE_CODE          # cut through vacuum -> walls
    log = TransactionLog(ctx)
    grid0 = ctx.grids["material"].copy()
    wall_codes = frozenset({MAT_HULL, MAT_STEEL})
    log.begin("corridor")
    log.snapshot_grid("material")
    changed = apply_corridor(ctx.grids["material"], 3, 5, 12, 5, width=3,
                             wall_id=MAT_STEEL, wall_codes=wall_codes)
    assert changed > 0
    txn = log.commit()
    grid1 = ctx.grids["material"].copy()
    # The op captured EVERY changed cell (whole-grid diff, not the bbox).
    assert int(txn.ops[0].ys.size) == int(np.count_nonzero(grid0 != grid1))
    log.undo()
    assert np.array_equal(ctx.grids["material"], grid0)
    log.redo()
    assert np.array_equal(ctx.grids["material"], grid1)


def test_water_flood_wholegrid_diff_roundtrips():
    ctx = make_ctx()
    # a glass-free open box: everything AIR, so a fill floods the whole grid
    solid = water_solid_codes()
    log = TransactionLog(ctx)
    region, why = water_fill_region(ctx.grids["material"], 6, 6, solid)
    assert region is not None, why
    water0 = ctx.grids["water"].copy()
    target = int(water_fixed.quantize(1.0))
    log.begin("water")
    log.snapshot_grid("water")
    for tx_, ty_ in region:
        ctx.grids["water"][ty_, tx_] = target
    txn = log.commit()
    water1 = ctx.grids["water"].copy()
    assert int(txn.ops[0].ys.size) == len(region)
    log.undo()
    assert np.array_equal(ctx.grids["water"], water0)
    log.redo()
    assert np.array_equal(ctx.grids["water"], water1)


# ---------------------------------------------------------------------------
# 4 — redo truncation
# ---------------------------------------------------------------------------

def test_redo_tail_truncated_by_new_commit():
    ctx = make_ctx()
    log = TransactionLog(ctx)
    for v in (MAT_HULL, MAT_STEEL, MAT_AIR):
        edit_cell(log, 0, 0, v)
    assert len(log.txns) == 3 and log.cursor == 3
    log.undo(); log.undo()
    assert log.cursor == 1
    edit_cell(log, 1, 1, MAT_HULL)                           # new action
    assert len(log.txns) == 2 and log.cursor == 2
    assert log.redo() is None                                # tail is gone


# ---------------------------------------------------------------------------
# 5 — saved-marker / dirty coherence
# ---------------------------------------------------------------------------

def test_saved_marker_dirty_coherence():
    ctx = make_ctx()
    log = TransactionLog(ctx)
    assert log.dirty is False                                # open == saved
    edit_cell(log, 0, 0, MAT_HULL)
    assert log.dirty is True
    log.undo()
    assert log.dirty is False                                # back at saved pos
    log.redo()
    assert log.dirty is True
    log.mark_saved()
    assert log.dirty is False
    edit_cell(log, 1, 1, MAT_STEEL)
    assert log.dirty is True


def test_saved_marker_none_when_evicted_below_it_stays_dirty():
    ctx = make_ctx()
    log = TransactionLog(ctx, depth=2, max_bytes=10 ** 9)
    edit_cell(log, 0, 0, 1)
    log.mark_saved()                                         # saved at cursor 1
    edit_cell(log, 0, 0, 2)
    edit_cell(log, 0, 0, 3)                                  # evicts txn 1 (saved)
    assert log.saved_cursor is None
    # undo all the way; dirty stays True since saved is unreachable
    while log.undo() is not None:
        pass
    assert log.cursor == 0 and log.dirty is True


# ---------------------------------------------------------------------------
# 6 — global-history ordering (different "modes", one history)
# ---------------------------------------------------------------------------

def test_global_history_orders_across_domains():
    ctx = make_ctx()
    log = TransactionLog(ctx)
    # a "material" grid op, then a "lights" collection op
    edit_cell(log, 2, 2, MAT_HULL, label="paint")
    log.begin("place light")
    log.snapshot_coll("lights")
    ctx.collections["lights"].append(
        EditableLight(x=4.5, y=4.5, color=(0.0, 0.0, 1.0), id="light_2"))
    log.commit()
    assert len(ctx.collections["lights"]) == 2
    # a single undo reverses the LAST action (the light), regardless of mode
    t = log.undo()
    assert t.label == "place light"
    assert len(ctx.collections["lights"]) == 1
    # the second undo reverses the earlier grid op
    t = log.undo()
    assert t.label == "paint"
    assert int(ctx.grids["material"][2, 2]) == MAT_AIR


# ---------------------------------------------------------------------------
# 7 — memory-bound eviction + cursor renumbering (C3/C7) + NEW-1
# ---------------------------------------------------------------------------

def test_depth_eviction_renumbers_cursor_and_keeps_correct_inverse():
    ctx = make_ctx()
    log = TransactionLog(ctx, depth=4, max_bytes=10 ** 9)
    # cell (0,0): 0->1->2->3->4->5->6, one txn each
    for v in range(1, 7):
        edit_cell(log, 0, 0, v)
    assert len(log.txns) == 4                                # oldest 2 evicted
    assert log.cursor == 4
    assert int(ctx.grids["material"][0, 0]) == 6
    # The inverse is CORRECT (not off-by-K): undo walks 6->5->4->3->2.
    for expect in (5, 4, 3, 2):
        log.undo()
        assert int(ctx.grids["material"][0, 0]) == expect
    assert log.cursor == 0
    assert log.undo() is None                                # only 4 retained


def test_byte_ceiling_governs_and_keeps_last_undoable():
    ctx = make_ctx()
    # tiny ceiling: a single 50-cell delta already exceeds it
    log = TransactionLog(ctx, depth=1000, max_bytes=10)
    ys = np.arange(50) % 12
    xs = np.arange(50) % 16
    edit_cell(log, ys, xs, np.ones(50, np.int32))
    assert len(log.txns) == 1                                # last is retained
    edit_cell(log, ys, xs, np.full(50, 2, np.int32))
    assert len(log.txns) == 1                                # evicted the old
    assert log._total_bytes() > log.max_bytes                # over ceiling, kept


def test_new1_second_eviction_after_saved_cursor_none():
    """NEW-1 (v2 verification): once saved_cursor is None, a SECOND front
    eviction must NOT do `None - K` — it stays None, no crash. The spec's own
    test only evicts below saved once; this drives the second eviction."""
    ctx = make_ctx()
    log = TransactionLog(ctx, depth=2, max_bytes=10 ** 9)
    edit_cell(log, 0, 0, 1)
    log.mark_saved()                                         # saved at 1
    edit_cell(log, 0, 0, 2)
    edit_cell(log, 0, 0, 3)                                  # 1st eviction -> saved None
    assert log.saved_cursor is None
    # A SECOND eviction while saved_cursor is already None:
    edit_cell(log, 0, 0, 4)                                  # would crash pre-fix
    assert log.saved_cursor is None
    assert len(log.txns) == 2
    # and the log is still coherent
    log.undo()
    assert int(ctx.grids["material"][0, 0]) == 3


# ---------------------------------------------------------------------------
# 7b — save-mask coherence (C1)
# ---------------------------------------------------------------------------

def test_save_mask_logged_keeps_live_vs_disk_coherent():
    """Fill water, wall over part, run the save-mask as a committed
    GridCellsOp('water'); the masked cells stay coherent across undo/redo and
    the dot reads clean after mark_saved."""
    ctx = make_ctx()
    solid = water_solid_codes()
    log = TransactionLog(ctx)
    # fill a region with water
    region, _ = water_fill_region(ctx.grids["material"], 6, 6, solid)
    target = int(water_fixed.quantize(1.0))
    log.begin("water")
    log.snapshot_grid("water")
    for tx_, ty_ in region:
        ctx.grids["water"][ty_, tx_] = target
    log.commit()
    # wall over part of the pool (a solid tile the mask must zero)
    log.begin("paint")
    log.snapshot_grid("material")
    ctx.grids["material"][6, 6] = MAT_HULL
    log.commit()

    # save-mask as a logged op
    from map_editor import mask_water_to_open
    log.begin("save-mask")
    log.snapshot_grid("water")
    masked, cleared = mask_water_to_open(ctx.grids["water"],
                                         ctx.grids["material"], solid)
    ctx.grids["water"][...] = masked
    log.commit()
    log.mark_saved()
    assert cleared >= 1
    assert log.dirty is False
    on_disk = ctx.grids["water"].copy()
    assert int(on_disk[6, 6]) == 0                           # masked under wall
    # undo/redo across the mask keeps live water == what the mask wrote
    log.undo()                                               # undo the mask
    log.redo()                                               # redo the mask
    assert np.array_equal(ctx.grids["water"], on_disk)
    assert log.dirty is False                                # back at saved pos


# ---------------------------------------------------------------------------
# 7c — union id allocator (B3)
# ---------------------------------------------------------------------------

def test_union_id_allocator_avoids_collision_across_families():
    lights = [EditableLight(x=0, y=0, color=(1, 1, 1), id="light_1")]
    entities = [EntityInstance(id="light_2", class_name="light", ordinal=0)]
    new_id = unique_entity_id("light", lights, entities)
    union = {l.id for l in lights} | {e.id for e in entities}
    assert new_id not in union                               # no [[entity]] clash
    assert new_id == "light_3"


# ---------------------------------------------------------------------------
# 8 — fuzz round-trip vs a reference model
# ---------------------------------------------------------------------------

def _snapshot_state(ctx):
    return ({k: v.copy() for k, v in ctx.grids.items()},
            {k: copy.deepcopy(v) for k, v in ctx.collections.items()})


def _states_equal(ctx, snap):
    grids, colls = snap
    return (all(np.array_equal(ctx.grids[k], grids[k]) for k in grids)
            and all(ctx.collections[k] == colls[k] for k in colls))


def _random_gesture(log, rng):
    """Apply one random, guaranteed-changing gesture through the builder."""
    ctx = log.ctx
    kind = rng.choice(("grid", "coll_add", "coll_del", "coll_edit"))
    if kind == "grid":
        gname = rng.choice(("material", "water"))
        g = ctx.grids[gname]
        log.begin(gname)
        log.snapshot_grid(gname)
        n = rng.randint(1, 6)
        for _ in range(n):
            y, x = rng.randrange(g.shape[0]), rng.randrange(g.shape[1])
            g[y, x] = int(g[y, x]) + rng.randint(1, 5)      # always changes
        return log.commit()
    if kind == "coll_add":
        log.begin("add")
        log.snapshot_coll("spawns")
        ctx.collections["spawns"].append(
            SpawnEntry(f"m_{rng.randrange(10**6)}", rng.randint(0, 1),
                       float(rng.randrange(16)), float(rng.randrange(12))))
        return log.commit()
    if kind == "coll_del" and ctx.collections["lights"]:
        log.begin("del")
        log.snapshot_coll("lights")
        ctx.collections["lights"].pop(
            rng.randrange(len(ctx.collections["lights"])))
        return log.commit()
    # coll_edit (entities .fields, exercises deep-copy)
    log.begin("edit")
    log.snapshot_coll("entities")
    e = ctx.collections["entities"][0]
    e.fields["span"] = int(e.fields.get("span", 0)) + 1
    return log.commit()


@pytest.mark.parametrize("seed", range(8))
def test_fuzz_undo_all_returns_to_start_then_redo_to_end(seed):
    rng = random.Random(seed)
    ctx = make_ctx()
    log = TransactionLog(ctx, depth=10 ** 6, max_bytes=10 ** 12)  # no eviction
    states = [_snapshot_state(ctx)]                          # states[cursor]
    for _ in range(40):
        if _random_gesture(log, rng) is not None:
            states.append(_snapshot_state(ctx))
    end = _snapshot_state(ctx)
    assert log.cursor == len(states) - 1

    # undo all the way -> start; compare at every cursor position
    while log.cursor > 0:
        log.undo()
        assert _states_equal(ctx, states[log.cursor])
    assert _states_equal(ctx, states[0])
    # redo all the way -> end
    while log.cursor < len(log.txns):
        log.redo()
        assert _states_equal(ctx, states[log.cursor])
    assert _states_equal(ctx, end)


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_random_undo_redo_walk_tracks_reference(seed):
    rng = random.Random(seed)
    ctx = make_ctx()
    log = TransactionLog(ctx, depth=10 ** 6, max_bytes=10 ** 12)
    states = [_snapshot_state(ctx)]
    for _ in range(30):
        if _random_gesture(log, rng) is not None:
            states.append(_snapshot_state(ctx))
    # random walk of undo/redo; the live state must match states[cursor] always
    for _ in range(80):
        if rng.random() < 0.5 and log.cursor > 0:
            log.undo()
        elif log.cursor < len(log.txns):
            log.redo()
        assert _states_equal(ctx, states[log.cursor])
