r"""tools/door_entity_port.py — the DOOR tool's entity-authoring bridge (Arc C3).

The DOOR tool (editor doc §6, canon engine/16 §6) places a `door`
`[[entity]]` instance — NOT the legacy painted MAT_DOOR — and stamps its
span onto the material grid IMMEDIATELY (`MAT_DOOR_CLOSED` for a closed
door, plain `MAT_AIR` for an open one).

Span math is NOT reimplemented here: every function below either calls
straight into :mod:`simulation.entities.door` (read-only import, entity
design §3b import-light — stdlib only) or hands its output straight to a
consumer, so the editor-stamped tile set is IDENTICAL, by construction, to
what the loader/runtime derive from the same `length_m` — the parity the C3
kickoff demands. `door.base_span`/`door.quantize_span_tiles`/
`door.tiles_per_m` are THE canonical functions; a parallel span calculation
here would be exactly the duplication the kickoff forbids.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from simulation.entities import door as door_entity  # noqa: E402
from level_loader import EntityInstance  # noqa: E402
from bake_level_art import BIT_E, BIT_N, BIT_S, BIT_W, edge16_mask  # noqa: E402
# Read-only reference (materials.py is explicitly allowed): the two stamp
# values a door's initial_state picks between (canon §6).
from simulation.materials import MAT_AIR, MAT_DOOR_CLOSED  # noqa: E402

DOOR_CLASS = "door"
DEFAULT_LENGTH_M = 1.0            # editor doc §6: the default span
_AUTHORED_KEYS = ("x", "y", "orientation", "length_m", "initial_state")


def door_tool_availability(tile_size_m) -> Optional[str]:
    """None if the DOOR tool can quantize spans at this level's
    `tile_size_m`; otherwise `door.tiles_per_m`'s own error string.

    Checked ONCE at editor launch (a non-integer tiles-per-meter level has
    no defined door quantization — a6 doors design §3 S1) so a placement
    click flashes a clear refusal instead of the tool crashing mid-drag.
    """
    try:
        door_entity.tiles_per_m(tile_size_m)
        return None
    except ValueError as e:
        return str(e)


def default_tile_count(tile_size_m) -> int:
    """The default door's tile count at this level's base resolution — 3
    tiles at the shipped `tile_size_m=0.333` (editor doc §4/§6: 1.0 m -> 3
    tiles). Delegates to `door.quantize_span_tiles`, the ONE quantizer."""
    return door_entity.quantize_span_tiles(DEFAULT_LENGTH_M, tile_size_m)


def length_m_for_tiles(n: int, tile_size_m) -> float:
    """The `length_m` value that reproduces EXACTLY `n` tiles through
    `door.quantize_span_tiles(length_m, tile_size_m)`.

    `n / tiles_per_m` computed once in float: with `tiles_per_m` a small
    integer (3 for every level in the tree today), the true rational value
    sits exactly AT the integer tile count when multiplied back — never
    near a round-half-up TIE (those fall at k + 1/2 tiles) — so the ~1e-16
    relative error of one float division, and the further ~1e-16 the toml
    round-trip introduces (`level_lib._fmt_value` writes floats via
    `repr()`, the shortest EXACT round-trip form), can never cross a tie
    boundary and flip the floor. Verified directly against
    `door.quantize_span_tiles` in tests, including the editor doc §4 trap
    (quantizing at a re-derived scaled resolution instead of replicating).
    """
    tpm = door_entity.tiles_per_m(tile_size_m)
    return float(n) / float(tpm)


def door_anchor_check(grid, tx, ty, wall_codes):
    """May a NEW door START at tile `(tx, ty)`? Returns `(ok, orientation,
    why)` — mirrors `map_editor.door_check`'s straight-run acceptance
    (corners/T/ends/isolated tiles refused, editor doc §6), but refuses an
    existing `MAT_DOOR_CLOSED` span instead of the legacy `MAT_DOOR` check
    (a new door must never start by growing INTO another door's tiles).
    `orientation` is `"v"` for a N|S wall run (the door widens along the
    column the wall already runs) / `"h"` for an E|W run, matching
    `door.py`'s x/y convention (leftmost tile for "h", topmost for "v") —
    the anchor clicked IS that tile, by construction, so no reassignment is
    ever needed downstream.
    """
    h, w = grid.shape
    tx, ty = int(tx), int(ty)
    if not (0 <= tx < w and 0 <= ty < h):
        return False, None, "outside the grid"
    v = int(grid[ty, tx])
    if v == MAT_DOOR_CLOSED:
        return False, None, "already a door"
    if v not in wall_codes:
        return False, None, "not a wall tile"
    mask = edge16_mask(grid, tx, ty, wall_codes)
    if mask == BIT_N | BIT_S:
        return True, "v", "vertical wall run"
    if mask == BIT_E | BIT_W:
        return True, "h", "horizontal wall run"
    if mask == 0:
        return False, None, "isolated wall tile"
    if mask in (BIT_N, BIT_E, BIT_S, BIT_W):
        return False, None, "end of a wall run"
    return False, None, "corner/junction (a door needs a straight run)"


def plan_door_span(grid, anchor, orientation: str, far_coord: int,
                   tile_size_m, wall_codes) -> tuple:
    """The forward-walked entity-door span from `anchor` (editor doc §6: a
    plain click uses the schema default length; a drag extends toward
    `far_coord`, snapped to the wall run under it).

    `anchor` is always the span's leftmost/topmost tile (door.py's own
    convention) — the walk only ever extends FORWARD (increasing column for
    "h", increasing row for "v"), so the anchor never needs reassigning. A
    release with no forward movement (`far_coord <= anchor`'s own
    coordinate) falls back to the DEFAULT tile count. Either way the walk
    STOPS at the first tile that is not a stampable wall tile (not in
    `wall_codes`, or already `MAT_DOOR_CLOSED` — a door never absorbs
    another door's span) or the grid edge — "snap to a wall run" (no hard
    minimum: a run shorter than the target simply yields a shorter door,
    the width-warning territory, never an error).

    Returns `(span, length_m)`: `span` is `door.base_span`'s OWN output
    (THE canonical function — never a parallel tile list) for the anchor +
    the derived `length_m`, so the stamped tiles are identical BY
    CONSTRUCTION to whatever a fresh load of the same fields would derive.
    """
    ax, ay = int(anchor[0]), int(anchor[1])
    is_h = orientation == "h"
    anchor_c = ax if is_h else ay
    grid_extent = grid.shape[1] if is_h else grid.shape[0]
    far_coord = int(far_coord)
    if far_coord > anchor_c:
        target_c = far_coord
    else:
        target_c = anchor_c + default_tile_count(tile_size_m) - 1

    def _tile(c: int) -> tuple:
        return (ay, c) if is_h else (c, ax)

    hi = anchor_c
    while hi < target_c and hi + 1 < grid_extent:
        v = int(grid[_tile(hi + 1)])
        if v not in wall_codes or v == MAT_DOOR_CLOSED:
            break
        hi += 1
    n = hi - anchor_c + 1
    length_m = length_m_for_tiles(n, tile_size_m)
    span = door_entity.base_span(
        {"x": ax, "y": ay, "orientation": orientation, "length_m": length_m},
        tile_size_m)
    return span, length_m


def instance_span(fields: dict, tile_size_m) -> list:
    """The tile span (`door.base_span`, THE canonical function — never a
    parallel calculation) for an ALREADY-AUTHORED door instance's fields
    (`x`/`y`/`orientation`/`length_m`). Arc C4's hit-testing/highlight/move/
    edit/paste all recompute the span through this one seam so a selected
    door's hitbox and re-stamp always match what a fresh load would derive —
    the same parity guarantee `plan_door_span` gives the placement tool."""
    return door_entity.base_span(fields, tile_size_m)


def door_span_tiles(entities, tile_size_m) -> set:
    """Every `(tx, ty)` covered by SOME `door` entity's span — the union
    over every `door` instance in `entities`, each via `instance_span`/
    `door.base_span` (THE canonical span, never a parallel derivation). The
    "legit door tile" set the MAT_DOOR_CLOSED-outside-a-span validator
    (canon §9) checks against."""
    legit = set()
    for e in entities:
        if e.class_name != DOOR_CLASS:
            continue
        for (fy, fx) in instance_span(e.fields, tile_size_m):
            legit.add((int(fx), int(fy)))
    return legit


def orphaned_door_closed_tiles(grid, entities, tile_size_m) -> set:
    """Every `(tx, ty)` carrying `MAT_DOOR_CLOSED` on `grid` that is NOT
    covered by any door entity's span (canon §9's forward pointer:
    "MAT_DOOR_CLOSED-outside-a-span validator warning") — door-material
    left behind by an edit: a door entity deleted without clearing its
    stamped span, a hand-edited tilemap, a span shrunk (C4 move/inspector-
    edit) without re-stamping the vacated tiles. A pure query — never
    mutates `grid`, never raises; the caller (the editor's status bar)
    decides how to surface it."""
    legit = door_span_tiles(entities, tile_size_m)
    g = np.asarray(grid)
    ys, xs = np.nonzero(g == MAT_DOOR_CLOSED)
    return {(int(tx), int(ty)) for ty, tx in zip(ys.tolist(), xs.tolist())
           if (int(tx), int(ty)) not in legit}


def door_span_validator_summary(grid, entities, tile_size_m) -> str:
    """The status-bar validator string for the MAT_DOOR_CLOSED-outside-a-
    span check (canon §9) — the C5 zone-binding-summary pattern applied to
    doors: `"doors ok"` when every `MAT_DOOR_CLOSED` tile sits inside some
    door's span, else a WARNING naming the orphaned tiles (sorted, so the
    message is stable frame to frame — never blocks save; the caller wires
    this into the SAME reserved slot C5's `zone_entity_port.
    zone_binding_summary` already uses)."""
    orphaned = orphaned_door_closed_tiles(grid, entities, tile_size_m)
    if not orphaned:
        return "doors ok"
    tiles = ", ".join(f"({tx},{ty})" for tx, ty in sorted(orphaned))
    n = len(orphaned)
    return (f"WARNING: {n} orphaned MAT_DOOR_CLOSED tile"
           f"{'s' if n != 1 else ''} outside any door span: {tiles}")


def build_door_instance(x: int, y: int, orientation: str, length_m: float,
                        initial_state: str, id_: str) -> EntityInstance:
    """The `[[entity]]` instance a placed door writes — all five schema
    fields authored explicitly (door.py FIELDS, §2a): every field but
    `initial_state` is REQUIRED-shaped (anchor/orientation/length_m define
    the span; nothing here has a sensible "leave at default" reading)."""
    fields = {"x": int(x), "y": int(y), "orientation": str(orientation),
              "length_m": float(length_m), "initial_state": str(initial_state)}
    return EntityInstance(id=id_, class_name=DOOR_CLASS, ordinal=0, tags=(),
                          fields=fields, authored_keys=_AUTHORED_KEYS)


def stamp_value_for(initial_state: str) -> int:
    """The material id a door's span stamps to at placement (canon §6): a
    closed door is fully solid `MAT_DOOR_CLOSED`; an open door's span is
    plain `MAT_AIR` (the "authored-open == authored-air" load-order pin)."""
    return MAT_DOOR_CLOSED if initial_state == "closed" else MAT_AIR


def commit_door_placement(log, grid, entities, span, stamp_value,
                          instance: EntityInstance):
    """The ONE compound transaction a door placement commits — the C2
    archetype (`GridCellsOp("material")` + `CollectionOp("entities")`, one
    `begin`/`commit` pair, editor doc §6 / C2 as-built §"the extension
    pattern"). `log` is an `undo_log.TransactionLog` whose ctx already
    registers `"material"` and `"entities"`; `grid`/`entities` are the SAME
    live objects the ctx holds (mutated in place, matching every other
    gesture in the editor). Returns the committed `Transaction` — never
    `None`, since a fresh instance always changes the `entities` list even
    when `span` is somehow empty."""
    log.begin("door")
    log.snapshot_grid("material")
    log.snapshot_coll("entities")
    for (fy, fx) in span:
        grid[fy, fx] = stamp_value
    entities.append(instance)
    return log.commit()
