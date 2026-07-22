r"""tools/entity_selection.py — unified selection over ALL `[[entity]]`
instances (doors, sensors, generic placements, lights) plus move/delete/
tag-assign/inspector-field-edit/clump-copy-paste (Arc C4).

C3 left doors/sensors/generic placements with NO select/move/delete/edit UX
(only SPAWN and LIGHT have their own bespoke hover+drag+RMB-delete). This
module is the pure, headless, raylib-free core that gives every OTHER
placed instance the same affordances, uniformly, over the SAME `lights`/
`entities` collections the transaction log already tracks (Arc C2) —
`tools/map_editor.py`'s new SELECT mode is a thin interactive shell over it
(smoke-tested only via `--auto`, per this codebase's established split:
raylib has no input-injection API, so gesture-driving logic lives here where
it IS unit-testable, and the loop just calls it).

SPAWN stays out of scope by construction: every function below takes only
`lights`/`entities` (never `spawns`) — units are not entities (canon
engine/16 §0/§3e), so folding SPAWN into this selection model would be
exactly the collapse Erik's A1 ruling forbade.

Identity model: every selectable instance carries a globally-unique `id`
(canon §2 — unique across `lights` u `entities`, `light_entity_port.
unique_entity_id`'s own invariant). A "selection" is therefore simply a set
of ids; :func:`find_instance` resolves an id back to its (collection name,
index) each time it's needed rather than caching a stale index, since any
mutation (delete, paste) can shift indices in the SAME frame.

Hitboxes (editor doc §8 / C4 kickoff): a door's hitbox is its WHOLE SPAN,
recomputed via :func:`door_entity_port.instance_span` (never a parallel
calculation — the same parity rule C3's placement tools follow) so a
selected door's highlight always matches its stamped material exactly; a
light's hitbox is its tile-floored center point; every other positioned
class is its one `(x, y)` tile; a POSITIONLESS class (no `x`/`y` — the logic
nodes, C3's own note) has no hitbox at all and is only reachable through
select-by-class, never box/click select.

Referential integrity (C2-mandated, canon engine/16 §7): delete and clump
paste are the two ops that change the entity id-set, so both carry the
`wires` `CollectionOp` in the SAME transaction as the `lights`/`entities`
one — an id-changing op that didn't would strand wires on dead ids (undo
would un-delete the entity but leave its wires gone) or point a pasted
wire's remapped endpoint nowhere.

Clump paste's tag-target wire simplification (documented, not a bug): a wire
whose `to` is `tag:name.input` is treated as EXTERNAL on both copy and
paste — its membership is resolved from LIVE `tags` at LOAD time, never
stored on the wire itself, so a tag-target wire already "reaches" a pasted
member for free (once the member's tags are copied verbatim, editor doc §8)
without this module doing anything wire-side. C6 (wire tool + LOGIC overlay)
owns any richer tag-aware clump behavior; this module deliberately does not
build tag-expansion machinery.
"""
from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from level_loader import WireSpec  # noqa: E402

import door_entity_port  # noqa: E402
import light_entity_port  # noqa: E402

LIGHTS = "lights"
ENTITIES = "entities"
WIRES = "wires"


# ---------------------------------------------------------------------------
# Identity, class/tag access, hit-testing
# ---------------------------------------------------------------------------

def find_instance(id_, lights, entities):
    """``(collection_name, index)`` for the given id — searches `lights`
    then `entities` (ids are unique over their union, canon §2). ``(None,
    None)`` when no instance carries it (e.g. it was already deleted this
    frame)."""
    for i, l in enumerate(lights):
        if l.id == id_:
            return LIGHTS, i
    for i, e in enumerate(entities):
        if e.id == id_:
            return ENTITIES, i
    return None, None


def instance_class_name(coll_name: str, obj) -> str:
    """The registry class name of a selectable instance — lights have no
    `class_name` attribute of their own (they are `EditableLight`, a
    `LightEntry` subclass), so a light's class is always the constant
    `"light"`."""
    return light_entity_port.LIGHT_CLASS if coll_name == LIGHTS else obj.class_name


def instance_tags(obj) -> tuple:
    return tuple(obj.tags)


def instance_tiles(coll_name: str, obj, tile_size_m) -> frozenset:
    """``{(tx, ty), ...}`` an instance's hitbox covers (col, row order — the
    editor's own cursor-tile convention, NOT `door.base_span`'s (row, col)
    `gamemap` convention, which this function converts). A door's WHOLE
    SPAN; a light's tile-floored center point; any other positioned class's
    one `(x, y)` tile; an empty set for a positionless class (no hitbox —
    select-by-class is the only way to reach it)."""
    if coll_name == LIGHTS:
        return frozenset({(int(math.floor(obj.x)), int(math.floor(obj.y)))})
    if obj.class_name == door_entity_port.DOOR_CLASS:
        span = door_entity_port.instance_span(obj.fields, tile_size_m)
        return frozenset((fx, fy) for (fy, fx) in span)
    x, y = obj.fields.get("x"), obj.fields.get("y")
    if x is None or y is None:
        return frozenset()
    return frozenset({(int(x), int(y))})


def all_ids(lights, entities) -> list:
    return [l.id for l in lights] + [e.id for e in entities]


def hit_test(tx, ty, lights, entities, tile_size_m):
    """The topmost instance whose hitbox contains tile `(tx, ty)`, or
    `None`. Entities are checked before lights (Arc C3 placements are the
    newer content, drawn on top), each in reverse list order (the
    `spawn_at`/`light_at` "most-recently-placed wins" convention)."""
    for e in reversed(entities):
        if (tx, ty) in instance_tiles(ENTITIES, e, tile_size_m):
            return e.id
    for l in reversed(lights):
        if (tx, ty) in instance_tiles(LIGHTS, l, tile_size_m):
            return l.id
    return None


def box_select(tx0, ty0, tx1, ty1, lights, entities, tile_size_m) -> list:
    """Ids of every instance any of whose hitbox tiles falls inside the
    inclusive tile rect (corners in any order)."""
    lo_x, hi_x = sorted((int(tx0), int(tx1)))
    lo_y, hi_y = sorted((int(ty0), int(ty1)))
    out = []
    for coll_name, obj in ([(LIGHTS, l) for l in lights]
                          + [(ENTITIES, e) for e in entities]):
        tiles = instance_tiles(coll_name, obj, tile_size_m)
        if any(lo_x <= tx <= hi_x and lo_y <= ty <= hi_y for tx, ty in tiles):
            out.append(obj.id)
    return out


def select_by_class(class_name: str, lights, entities) -> list:
    """Every instance of `class_name`, in list order — the ONLY way to
    reach a positionless class (no hitbox, so click/box select never find
    it)."""
    if class_name == light_entity_port.LIGHT_CLASS:
        return [l.id for l in lights]
    return [e.id for e in entities if e.class_name == class_name]


# ---------------------------------------------------------------------------
# Wire endpoint helpers (mirrors level_loader._split_endpoint's own "split on
# the FIRST dot" rule — ids/tag names cannot contain '.', the loader's slug
# charset, so this is unambiguous; not importing that private function).
# ---------------------------------------------------------------------------

_WIRE_TAG_PREFIX = "tag:"


def _endpoint_head(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _wire_touches(wire, ids_set: set) -> bool:
    """True iff a NON-TAG endpoint of `wire` names an id in `ids_set` — the
    delete-time referential check (a tag-target `to` is never "touched":
    its membership resolves from live tags at load, nothing to strand)."""
    if _endpoint_head(wire.from_) in ids_set:
        return True
    to_head = _endpoint_head(wire.to)
    return not to_head.startswith(_WIRE_TAG_PREFIX) and to_head in ids_set


def _wire_internal(wire, ids_set: set) -> bool:
    """True iff BOTH endpoints of `wire` are members of `ids_set` and the
    target is not a tag (a tag-target wire is always EXTERNAL on copy/paste
    — the module docstring's documented simplification)."""
    to_head = _endpoint_head(wire.to)
    if to_head.startswith(_WIRE_TAG_PREFIX):
        return False
    return _endpoint_head(wire.from_) in ids_set and to_head in ids_set


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

def plan_move_selection(lights, entities, ids, delta_tx: int, delta_ty: int,
                        tile_size_m, grid_shape) -> tuple:
    """Pre-mutation validation (§6.2 "validate before mutate"): refuses a
    move whose new door span would leave the grid, BEFORE anything is
    touched. Returns ``(True, None)`` or ``(False, reason)``."""
    h, w = grid_shape
    for id_ in ids:
        coll_name, idx = find_instance(id_, lights, entities)
        if coll_name != ENTITIES:
            continue
        e = entities[idx]
        if e.class_name != door_entity_port.DOOR_CLASS:
            continue
        fields = dict(e.fields)
        fields["x"] = int(fields["x"]) + delta_tx
        fields["y"] = int(fields["y"]) + delta_ty
        span = door_entity_port.instance_span(fields, tile_size_m)
        for fy, fx in span:
            if not (0 <= fx < w and 0 <= fy < h):
                return False, f"moving {id_} would push its span off the grid"
    return True, None


def commit_move_selection(log, grid, lights, entities, ids,
                          delta_tx: int, delta_ty: int, tile_size_m):
    """Move every selected instance by `(delta_tx, delta_ty)` whole tiles as
    ONE compound transaction: `GridCellsOp("material")` (a selected door's
    OLD span clears to open air, its NEW span re-stamps per its authored
    `initial_state` — canon engine/16 §6) + `CollectionOp("lights")` /
    `CollectionOp("entities")` (undo_log's `commit()` only keeps the ops that
    actually changed, so a selection with no lights never touches
    `"lights"`). Returns the committed Transaction, or `None` when the delta
    is `(0, 0)`, `ids` is empty, or :func:`plan_move_selection` refuses the
    move (nothing is mutated in the refusal case either)."""
    if (delta_tx == 0 and delta_ty == 0) or not ids:
        return None
    ok, _reason = plan_move_selection(lights, entities, ids, delta_tx,
                                      delta_ty, tile_size_m, grid.shape)
    if not ok:
        return None
    log.begin("move selection")
    log.snapshot_grid("material")
    log.snapshot_coll("lights")
    log.snapshot_coll("entities")
    for id_ in ids:
        coll_name, idx = find_instance(id_, lights, entities)
        if coll_name is None:
            continue
        if coll_name == LIGHTS:
            l = lights[idx]
            lights[idx] = replace(l, x=l.x + delta_tx, y=l.y + delta_ty)
            continue
        e = entities[idx]
        is_door = e.class_name == door_entity_port.DOOR_CLASS
        if is_door:
            for fy, fx in door_entity_port.instance_span(e.fields, tile_size_m):
                grid[fy, fx] = door_entity_port.stamp_value_for("open")
        fields = dict(e.fields)
        if "x" in fields:
            fields["x"] = fields["x"] + delta_tx
        if "y" in fields:
            fields["y"] = fields["y"] + delta_ty
        entities[idx] = replace(e, fields=fields)
        if is_door:
            new_span = door_entity_port.instance_span(fields, tile_size_m)
            stamp = door_entity_port.stamp_value_for(
                fields.get("initial_state", "closed"))
            for fy, fx in new_span:
                grid[fy, fx] = stamp
    return log.commit()


# ---------------------------------------------------------------------------
# Delete (+ wire referential-integrity cleanup, C2 §7 mandate)
# ---------------------------------------------------------------------------

def commit_delete_selection(log, grid, lights, entities, wires, ids,
                            tile_size_m, *, zones=None, clear_zone_ids=None):
    """Delete every selected instance as ONE compound transaction:
    `GridCellsOp("material")` (a selected door's span clears to open air —
    the same "clear" a move's old span gets) + `CollectionOp("lights")` /
    `CollectionOp("entities")` + `CollectionOp("wires")` — any wire whose
    `from` or non-tag `to` names a deleted id is dropped in the SAME
    transaction (the referential-integrity rule: undo must restore the
    wires along with the entity, never strand them on a dead id).

    Arc C5: `clear_zone_ids` (a set of `zone_id` ints, or `None`/empty) also
    clears every `zones.npy` tile carrying one of those ids to 0 in the SAME
    transaction — the "deleting a zone instance PROMPTS to clear its paint"
    gesture (editor design §5); the caller (`map_editor`'s SELECT mode)
    decides which ids to clear from the user's confirm keypress. A zone
    instance deleted WITHOUT its id in `clear_zone_ids` simply leaves its
    paint orphaned — the loader's own validator warning covers that case,
    never a crash. `zones` is ignored unless `clear_zone_ids` is non-empty
    (no `GridCellsOp("zones")` opens at all when nothing needs clearing, so
    a delete with no zone instances is byte-for-byte the pre-C5 behavior).

    Returns the committed Transaction, or `None` when `ids` is empty."""
    if not ids:
        return None
    ids_set = set(ids)
    clear_zone_ids = set(clear_zone_ids) if clear_zone_ids else set()
    log.begin("delete selection")
    log.snapshot_grid("material")
    clear_zones = zones is not None and clear_zone_ids
    if clear_zones:
        log.snapshot_grid("zones")
    log.snapshot_coll("lights")
    log.snapshot_coll("entities")
    log.snapshot_coll("wires")
    # Clear door spans BEFORE removing the instances — fields must still be
    # readable to recompute the span.
    for e in entities:
        if e.id in ids_set and e.class_name == door_entity_port.DOOR_CLASS:
            for fy, fx in door_entity_port.instance_span(e.fields, tile_size_m):
                grid[fy, fx] = door_entity_port.stamp_value_for("open")
    if clear_zones:
        mask = np.isin(zones, np.asarray(sorted(clear_zone_ids),
                                         dtype=zones.dtype))
        zones[mask] = 0
    lights[:] = [l for l in lights if l.id not in ids_set]
    entities[:] = [e for e in entities if e.id not in ids_set]
    wires[:] = [w for w in wires if not _wire_touches(w, ids_set)]
    return log.commit()


# ---------------------------------------------------------------------------
# Assign-tag-to-selection
# ---------------------------------------------------------------------------

def commit_assign_tag(log, lights, entities, ids, tag: str):
    """Add `tag` to every selected instance's `tags` (idempotent — already
    carrying it is a per-instance no-op) as ONE transaction (`CollectionOp`
    on whichever of `"lights"`/`"entities"` the selection actually touches;
    a selection with no lights never touches `"lights"`). Returns the
    committed Transaction, or `None` when `ids` or `tag` is empty, or the
    tag was already present on every selected instance (a genuine no-op —
    `commit()` drops an unchanged collection)."""
    if not ids or not tag:
        return None
    ids_set = set(ids)
    log.begin("assign tag")
    log.snapshot_coll("lights")
    log.snapshot_coll("entities")
    for i, l in enumerate(lights):
        if l.id in ids_set and tag not in l.tags:
            lights[i] = replace(l, tags=tuple(l.tags) + (tag,))
    for i, e in enumerate(entities):
        if e.id in ids_set and tag not in e.tags:
            entities[i] = replace(e, tags=tuple(e.tags) + (tag,))
    return log.commit()


# ---------------------------------------------------------------------------
# Inspector field edits
# ---------------------------------------------------------------------------

_DOOR_RESTAMP_FIELDS = frozenset({"length_m", "orientation", "initial_state"})


def commit_field_edit(log, grid, lights, entities, id_, field_name,
                      new_value, tile_size_m):
    """Commit ONE inspector field edit on the selected instance as ONE
    transaction (editor design §3 pillar 6 "inspector edits"). Editing a
    door's `length_m`/`orientation`/`initial_state` re-stamps its grid span
    (old span cleared, new span stamped per the NEW `initial_state`) in the
    SAME compound transaction. Editing `"id"` is refused BEFORE any
    mutation if the new id collides with any OTHER instance across
    `lights` u `entities` (canon §2 — mandatory unique ids); editing any
    OTHER field that the instance did not originally author adds it to
    `authored_keys` so the edit actually survives a save (an edit to a
    field already at its schema default would otherwise be silently
    dropped by `format_entity_lines`' "defaults are never materialized"
    contract).

    Returns ``(True, Transaction | None)`` on success (`None` only when the
    edit was a genuine no-op — `commit()` dropped an unchanged collection),
    or ``(False, reason)`` when refused before any mutation."""
    coll_name, idx = find_instance(id_, lights, entities)
    if coll_name is None:
        return False, f"no instance with id {id_!r}"
    if field_name == "id":
        new_id = str(new_value)
        if new_id in (x for x in all_ids(lights, entities) if x != id_):
            return False, f"id {new_id!r} is already in use"

    log.begin("edit field")
    if coll_name == LIGHTS:
        log.snapshot_coll("lights")
        l = lights[idx]
        if field_name == "id":
            lights[idx] = replace(l, id=str(new_value))
        else:
            lights[idx] = replace(l, **{field_name: new_value})
        return True, log.commit()

    e = entities[idx]
    is_door = e.class_name == door_entity_port.DOOR_CLASS
    restamp = is_door and field_name in _DOOR_RESTAMP_FIELDS
    if restamp:
        log.snapshot_grid("material")
    log.snapshot_coll("entities")

    if field_name == "id":
        new_e = replace(e, id=str(new_value))
    else:
        old_span = (door_entity_port.instance_span(e.fields, tile_size_m)
                   if restamp else None)
        fields = dict(e.fields)
        fields[field_name] = new_value
        authored = tuple(e.authored_keys)
        if field_name not in authored:
            authored = authored + (field_name,)
        new_e = replace(e, fields=fields, authored_keys=authored)
        if restamp:
            for fy, fx in old_span:
                grid[fy, fx] = door_entity_port.stamp_value_for("open")
            new_span = door_entity_port.instance_span(fields, tile_size_m)
            new_stamp = door_entity_port.stamp_value_for(
                fields.get("initial_state", "closed"))
            for fy, fx in new_span:
                grid[fy, fx] = new_stamp
    entities[idx] = new_e
    return True, log.commit()


# ---------------------------------------------------------------------------
# Clump copy/paste — the poor man's prefab (editor doc §8)
# ---------------------------------------------------------------------------

def compute_clump_copy(ids, lights, entities, wires) -> dict:
    """Snapshot the copied clump: every selected instance (deep-copied) +
    every INTERNAL wire among them (both endpoints in `ids`, non-tag
    target — see the module docstring's tag-target simplification).
    External and tag-target wires are never captured — dropping them at
    COPY time is equivalent to dropping them at paste time (the decision
    only depends on the copied id set, which never changes across repeated
    pastes of the same clipboard), and keeps the clipboard itself small."""
    ids_set = set(ids)
    return {
        LIGHTS: [deepcopy(l) for l in lights if l.id in ids_set],
        ENTITIES: [deepcopy(e) for e in entities if e.id in ids_set],
        WIRES: [deepcopy(w) for w in wires if _wire_internal(w, ids_set)],
    }


def clump_anchor_tile(clump: dict):
    """``(floor(min x), floor(min y))`` over every POSITIONED clump member
    (a light's continuous `x`/`y`, or an entity's int `x`/`y` fields) — the
    clump's paste anchor (top-left of its bounding box), so a paste-at-
    cursor offset preserves every member's position RELATIVE to this point.
    `None` for an all-positionless clump (e.g. logic nodes only) — the
    caller then pastes at a `(0, 0)` delta, landing members exactly where
    they were copied."""
    xs, ys = [], []
    for l in clump[LIGHTS]:
        xs.append(l.x)
        ys.append(l.y)
    for e in clump[ENTITIES]:
        x, y = e.fields.get("x"), e.fields.get("y")
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return int(math.floor(min(xs))), int(math.floor(min(ys)))


def commit_clump_paste(log, grid, lights, entities, wires, clump: dict,
                       delta_tx: int, delta_ty: int, tile_size_m):
    """Paste a copied clump offset by `(delta_tx, delta_ty)` whole tiles as
    ONE compound transaction — `GridCellsOp("material")` (pasted doors
    re-stamp their NEW span) + `CollectionOp("lights")` +
    `CollectionOp("entities")` + `CollectionOp("wires")`, atomic: undo
    reverts entities, wires, AND grid together (the C2 §7 referential-
    integrity rule, symmetric with delete).

    Algorithm (editor doc §8):
      1. Mint a NEW unique id for every member via
         `light_entity_port.unique_entity_id`, keyed by CLASS NAME (the
         same convention every placement tool already mints under) —
         scanning the LIVE `lights`/`entities`, which this function appends
         to progressively, so two same-class members never collide with
         each other either.
      2. Duplicate each member with its id remapped + position offset by
         the delta; a door re-stamps its NEW span (`MAT_DOOR_CLOSED`/
         `MAT_AIR` per its authored `initial_state`) — out-of-grid cells
         are silently skipped (defensive; the entity is still pasted).
         `tags` copy VERBATIM (editor doc §8).
      3. Duplicate each INTERNAL wire (both endpoints were in the copied
         set, by `compute_clump_copy`'s own filter) with BOTH endpoints
         remapped through the id map built in step 1.

    Returns ``(Transaction, new_ids)`` — `new_ids` is every freshly-minted
    id, lights first then entities, in clump order (a natural "select what
    was just pasted" list for the caller). ``(None, [])`` for an empty
    clump (no lights and no entities — a caller error, never hit from a
    real selection, which always has >= 1 member)."""
    if not clump[LIGHTS] and not clump[ENTITIES]:
        return None, []
    log.begin("paste")
    log.snapshot_grid("material")
    log.snapshot_coll("lights")
    log.snapshot_coll("entities")
    log.snapshot_coll("wires")

    h, w = grid.shape
    id_map = {}
    for l in clump[LIGHTS]:
        new_id = light_entity_port.unique_entity_id(
            light_entity_port.LIGHT_CLASS, lights, entities)
        id_map[l.id] = new_id
        lights.append(replace(l, id=new_id, x=l.x + delta_tx,
                              y=l.y + delta_ty))
    for e in clump[ENTITIES]:
        new_id = light_entity_port.unique_entity_id(
            e.class_name, lights, entities)
        id_map[e.id] = new_id
        fields = dict(e.fields)
        if "x" in fields:
            fields["x"] = fields["x"] + delta_tx
        if "y" in fields:
            fields["y"] = fields["y"] + delta_ty
        new_e = replace(e, id=new_id, fields=fields)
        entities.append(new_e)
        if e.class_name == door_entity_port.DOOR_CLASS:
            span = door_entity_port.instance_span(fields, tile_size_m)
            stamp = door_entity_port.stamp_value_for(
                fields.get("initial_state", "closed"))
            for fy, fx in span:
                if 0 <= fy < h and 0 <= fx < w:
                    grid[fy, fx] = stamp
    for wspec in clump[WIRES]:
        from_id, signal = wspec.from_.split(".", 1)
        to_id, input_name = wspec.to.split(".", 1)
        wires.append(WireSpec(from_=f"{id_map[from_id]}.{signal}",
                              to=f"{id_map[to_id]}.{input_name}"))
    new_ids = ([id_map[l.id] for l in clump[LIGHTS]]
              + [id_map[e.id] for e in clump[ENTITIES]])
    return log.commit(), new_ids
