r"""tools/wire_tool.py — the two-click wire tool + LOGIC overlay + tag badges
(Arc C6, editor doc §8, canon engine/16 §8).

Three things land here, pure and headless (no raylib import — `tools/
map_editor.py`'s new WIRE mode is a thin interactive shell over this module,
matching every prior Arc C patch's split between tested logic and raylib
plumbing):

  1. **Positionless-entity canvas layout** (the gap C3/C4/C5 each flagged and
     punted): logic nodes (`decider`/`gate_and`/`gate_or`/`gate_not`/
     `filter`/`clock`/`airlock_controller` — no `x`/`y` in their schema at
     all) and zone instances (`breach_site`/`extraction_zone` — likewise no
     `x`/`y`, defined entirely by their `zones.npy` paint) have nothing a
     wire tool could click. :func:`positionless_layout` gives every one of
     them a deterministic, session-computed clickable tile: a zone instance
     anchors at the CENTROID of its painted tiles (it has a real spatial
     home); anything else (true logic nodes, or a zone instance with no
     paint yet) gets a stable auto-layout slot in a strip just past the map
     grid's right edge, ordered by id. Recomputed fresh every frame from live
     state — never a new persisted field, never a new file (escalation
     trigger 1 stays clear: no format change).
  2. **The two-click wire gesture** (never a drag — editor doc §8): click a
     source -> resolve its output signal (multiple signals: cycle, default
     the primary non-`alive` one) -> pan/zoom/scroll stay live -> click a
     target -> resolve its input (multiple inputs: cycle, confirm) -> COMMIT.
     :func:`begin_pending_wire` / :func:`cycle_pending_signal` /
     :func:`set_pending_target_id` / :func:`set_pending_target_tag` /
     :func:`cycle_pending_input` walk a plain-dict pending-wire state machine
     (the same "gesture state as a dict" convention `entity_drag`/
     `select_box` already use in map_editor.py); :func:`commit_add_wire`
     validates then commits ONE `CollectionOp("wires")` transaction.
  3. **Validation reuse**: :func:`validate_wire` calls STRAIGHT into
     `level_loader._parse_wires` (the loader's own frozen §1b rules — source
     signal must exist, target input must exist, a unit reference hard-
     errors) rather than re-deriving them — the exact precedent
     `zone_entity_port.zone_binding_summary` set for
     `level_loader._validate_zone_binding` (a private loader function called
     via module-qualified reference, not re-implemented). Lights are folded
     in via `light_entity_port.light_to_entity_instance` so a wire can
     legally touch a light's free `alive` signal.
  4. **LOGIC overlay support**: :func:`wires_touching_selection` (the
     default filter — reuses `entity_selection`'s own referential-touch
     predicate, the same one C4's delete built) and
     :func:`wire_endpoints_for_draw` (splits wires into plain line segments
     vs tag-target BADGES — a tag wire renders as one compact badge near its
     source, never fanned to every member, editor doc §8).

Nothing here imports raylib; `tests/test_wire_tool.py` drives it directly.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import level_loader  # noqa: E402
from level_loader import WireSpec, ZONE_CLASSES  # noqa: E402

import entity_selection  # noqa: E402
import light_entity_port  # noqa: E402

_WIRE_TAG_PREFIX = "tag:"

# ---------------------------------------------------------------------------
# Positionless-entity canvas layout
# ---------------------------------------------------------------------------

LAYOUT_COLS = 4          # the auto-layout strip's column count
LAYOUT_SPACING = 2       # tiles between strip slots (both axes)
LAYOUT_MARGIN = 2         # tiles of gap between the map grid's right edge
                          # and the strip's first column


def is_positionless(fields: dict) -> bool:
    """True iff this entity's class declares no `x`/`y` (schema.py — a class
    with genuinely no `x` field has ``fields.get("x") is None`` by
    construction, the SAME convention `entity_selection.instance_tiles`
    already relies on for its own positionless-hitbox check)."""
    return fields.get("x") is None or fields.get("y") is None


def positionless_layout(entities, zones=None, grid_shape=None) -> dict:
    """``{id: (tx, ty)}`` for every positionless entity (the C3/C4/C5
    forward-pointer this patch closes).

    A zone-class instance (`breach_site`/`extraction_zone`) anchors at the
    ROUNDED CENTROID of its `zones.npy`-painted tiles when `zones` is given
    and any tile currently carries its `zone_id` — it has a genuine spatial
    home, painted by the ZONE tool (Arc C5). Everything else — the true
    logic nodes, AND a zone instance with no paint yet (freshly authored,
    or its paint was cleared) — falls back to a deterministic auto-layout
    strip starting `LAYOUT_MARGIN` tiles past the map grid's right edge
    (`grid_shape[1]`, or column 0 if the grid shape isn't known), filling
    `LAYOUT_COLS` columns top-down then left-to-right, ordered by id (a
    stable sort key independent of list/insertion order, so the layout is
    identical across an undo/redo or a save/reload even if `entities`' own
    order shifts). Recomputed fresh every call — never persisted, never a
    new level field (this is an EDITOR-session convenience, not level
    content)."""
    positions: dict = {}
    strip_ids: list = []
    for e in entities:
        if not is_positionless(e.fields):
            continue
        if e.class_name in ZONE_CLASSES and zones is not None:
            zid = e.fields.get("zone_id")
            if zid is not None:
                mask = (zones == int(zid))
                if mask.any():
                    ys, xs = mask.nonzero()
                    positions[e.id] = (int(round(float(xs.mean()))),
                                       int(round(float(ys.mean()))))
                    continue
        strip_ids.append(e.id)
    strip_ids.sort()
    base_x = int(grid_shape[1]) + LAYOUT_MARGIN if grid_shape else 0
    for i, id_ in enumerate(strip_ids):
        row, col = divmod(i, LAYOUT_COLS)
        positions[id_] = (base_x + col * LAYOUT_SPACING, row * LAYOUT_SPACING)
    return positions


def anchor_tile(id_, lights, entities, tile_size_m, layout: dict):
    """The ONE clickable/drawable tile for `id_` — a positioned instance's
    hitbox centroid (a door's whole span collapses to its center tile, a
    light/generic entity's hitbox is already one tile), or its
    `positionless_layout` slot. `None` when `id_` no longer resolves (an
    entity deleted elsewhere this frame) or a positionless id absent from
    `layout` (stale — `layout` should be recomputed every frame)."""
    coll_name, idx = entity_selection.find_instance(id_, lights, entities)
    if coll_name is None:
        return None
    obj = (lights[idx] if coll_name == entity_selection.LIGHTS
           else entities[idx])
    tiles = entity_selection.instance_tiles(coll_name, obj, tile_size_m)
    if tiles:
        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]
        return (int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys))))
    return layout.get(id_)


def hit_test_with_layout(tx, ty, lights, entities, tile_size_m,
                         layout: dict):
    """`entity_selection.hit_test`, extended to also hit a positionless
    entity at its `positionless_layout` slot (entity_selection's own
    hit_test never does — SELECT mode's box/click select intentionally
    stays "positionless is select-by-class only", per C4; the wire tool
    needs the extra reach since positionless nodes are exactly what it
    exists to wire together)."""
    hit = entity_selection.hit_test(tx, ty, lights, entities, tile_size_m)
    if hit is not None:
        return hit
    tile = (int(tx), int(ty))
    for id_, pos in layout.items():
        if pos == tile:
            return id_
    return None


def class_name_of(id_, lights, entities):
    coll_name, idx = entity_selection.find_instance(id_, lights, entities)
    if coll_name is None:
        return None
    obj = (lights[idx] if coll_name == entity_selection.LIGHTS
           else entities[idx])
    return entity_selection.instance_class_name(coll_name, obj)


def class_payload_of(id_, lights, entities, registry_payload):
    """``(class_name, cls_payload)`` for `id_` — `cls_payload` is `None`
    when the id no longer resolves, or its class fell out of the registry
    payload (a stale selection after an import fallback, mirroring
    `map_editor.py`'s ENTITY-mode check of the same shape)."""
    cls_name = class_name_of(id_, lights, entities)
    if cls_name is None:
        return None, None
    return cls_name, registry_payload.get("classes", {}).get(cls_name)


# ---------------------------------------------------------------------------
# Signal / input vocabulary (registry_payload's per-class shape, Arc C1)
# ---------------------------------------------------------------------------

def signal_names(cls_payload: dict) -> tuple:
    return tuple(s["name"] for s in cls_payload.get("signals", []))


def input_names(cls_payload: dict) -> tuple:
    return tuple(i["name"] for i in cls_payload.get("inputs", []))


def primary_signal(cls_payload: dict):
    """The wire tool's default source signal: the first NON-`alive` signal
    (`all_signals()`'s own order is always `(alive,) + cls.SIGNALS` — canon
    §8 — so index 1 is the class's own first declared signal) when one
    exists; else the free `alive` signal itself (every class has at least
    that one — button/terminal/light are `alive`-only, "inert but
    format-reserved", canon §8)."""
    names = signal_names(cls_payload)
    if not names:
        return None
    return names[1] if len(names) > 1 else names[0]


def tag_input_names(tag_name: str, lights, entities, registry_payload) -> tuple:
    """Candidate inputs for a `tag:name` target: the INTERSECTION of every
    CURRENT member's declared inputs (Arc B §1b — "a tag member lacking the
    input hard-errors", so only an input common to every member could ever
    commit). Empty when the tag currently has no members, or no input name
    is common to all of them (both legitimate "nothing to offer" outcomes,
    not errors — tag membership is dynamic, canon §8)."""
    classes_used = set()
    for l in lights:
        if tag_name in l.tags:
            classes_used.add(light_entity_port.LIGHT_CLASS)
    for e in entities:
        if tag_name in e.tags:
            classes_used.add(e.class_name)
    if not classes_used:
        return ()
    classes = registry_payload.get("classes", {})
    sets = [set(input_names(classes[c])) for c in classes_used if c in classes]
    if not sets:
        return ()
    common = set.intersection(*sets)
    return tuple(sorted(common))


def all_tags(lights, entities) -> tuple:
    """Every tag currently in use, sorted — the wire tool's tag-target
    picker cycles this list (map_editor's WIRE mode `G` key)."""
    tags = set()
    for l in lights:
        tags.update(l.tags)
    for e in entities:
        tags.update(e.tags)
    return tuple(sorted(tags))


# ---------------------------------------------------------------------------
# Validation — REUSED from level_loader, never re-derived (canon §8 / Arc B
# §1b). Mirrors zone_entity_port.zone_binding_summary's own precedent of
# calling a private loader validator via module-qualified reference.
# ---------------------------------------------------------------------------

def _combined_entities(lights, entities) -> list:
    """`lights` lifted to `EntityInstance` (`light_to_entity_instance`, the
    SAME bridge C1's save path uses) + `entities`, verbatim — the full
    id-addressable universe `level_loader._parse_wires` needs to resolve a
    candidate wire's endpoints (a light carries the free `alive` signal and
    can be a legal wire source)."""
    return ([light_entity_port.light_to_entity_instance(l, i)
            for i, l in enumerate(lights)] + list(entities))


def validate_wire(from_id: str, signal: str, to_spec: str, input_name: str,
                  lights, entities, spawns, existing_wires=()) -> tuple:
    """Validate a candidate wire ``from_id.signal -> to_spec.input_name``
    (`to_spec` a plain live id, or ``"tag:name"``) by calling straight into
    `level_loader._parse_wires` — the loader's own frozen §1b rules (source
    signal must exist on the source class; target input must exist on the
    target class, or on EVERY tag member; a unit reference hard-errors) —
    never re-implemented here.

    Liveness (source/non-tag-target id must currently exist) is checked
    directly first: the interactive tool can only ever click a LIVE id, so
    this is a cheap defensive refusal rather than the loader's own
    authoring-file "dangling id warns + drops" allowance (that allowance
    exists for a hand-edited file referencing something deleted elsewhere;
    an editor gesture pointed at a selected id should never silently author
    a wire that gets dropped).

    Returns ``(True, None)`` for a legal wire, or ``(False, reason)``.
    """
    combined = _combined_entities(lights, entities)
    ids = {e.id for e in combined}
    if from_id not in ids:
        return False, f"'{from_id}' is not a live entity id"
    is_tag = to_spec.startswith(_WIRE_TAG_PREFIX)
    if not is_tag and to_spec not in ids:
        return False, f"'{to_spec}' is not a live entity id"
    from_str = f"{from_id}.{signal}"
    to_str = f"{to_spec}.{input_name}"
    raw = {"wire": [{"from": w.from_, "to": w.to} for w in existing_wires]
                  + [{"from": from_str, "to": to_str}]}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # pre-existing dangling wires
                                               # elsewhere are not this call's
                                               # concern; liveness above
                                               # already covers OUR candidate
            level_loader._parse_wires(raw, "<editor wire tool>", combined,
                                      spawns)
    except ValueError as e:
        return False, str(e)
    return True, None


def commit_add_wire(log, wires, lights, entities, spawns, from_id: str,
                    signal: str, to_spec: str, input_name: str) -> tuple:
    """Validate (:func:`validate_wire`) then, ONLY if valid, commit ONE
    `CollectionOp("wires")` transaction — no transaction opens on a refusal
    (§6.2 validate-before-mutate, the same contract DOOR/sensor placement
    follow). Returns ``(True, Transaction)`` or ``(False, reason)``."""
    ok, reason = validate_wire(from_id, signal, to_spec, input_name,
                               lights, entities, spawns,
                               existing_wires=wires)
    if not ok:
        return False, reason
    log.begin("add wire")
    log.snapshot_coll("wires")
    wires.append(WireSpec(from_=f"{from_id}.{signal}",
                          to=f"{to_spec}.{input_name}"))
    return True, log.commit()


def commit_remove_wire(log, wires, wire):
    """Remove ONE wire (the LOGIC overlay's delete gesture) as a single
    `CollectionOp("wires")` transaction. Returns the committed Transaction,
    or `None` when `wire` is no longer present (already removed elsewhere
    this frame — a genuine no-op)."""
    if wire not in wires:
        return None
    log.begin("remove wire")
    log.snapshot_coll("wires")
    wires.remove(wire)
    return log.commit()


# ---------------------------------------------------------------------------
# Pending-wire gesture state (a plain dict, matching the `entity_drag`/
# `select_box` convention already used for in-progress gestures elsewhere in
# map_editor.py — never opens a log transaction until commit, so a mode
# switch can simply drop it, matching door_drag/entity_drag's own §6.2 note).
# ---------------------------------------------------------------------------

def begin_pending_wire(from_id: str, cls_payload: dict) -> dict:
    """Click 1: a source is picked. Defaults to its primary signal (the
    first non-`alive` signal, else `alive`); `signal_idx` is user-cyclable
    via :func:`cycle_pending_signal` before a target is chosen."""
    signals = signal_names(cls_payload)
    primary = primary_signal(cls_payload)
    idx = signals.index(primary) if primary in signals else 0
    return {"from_id": from_id, "signals": signals, "signal_idx": idx,
           "to_spec": None, "to_is_tag": False, "tag_name": None,
           "inputs": (), "input_idx": 0}


def cycle_pending_signal(pending: dict, step: int) -> dict:
    """No-op once a target is locked (a signal choice is meaningless after
    that point) or when there is only one signal to choose from."""
    if pending["to_spec"] is not None or len(pending["signals"]) <= 1:
        return pending
    n = len(pending["signals"])
    p = dict(pending)
    p["signal_idx"] = (pending["signal_idx"] + step) % n
    return p


def set_pending_target_id(pending: dict, to_id: str, cls_payload: dict) -> tuple:
    """Click 2 (the id form): lock the target onto a live entity id.
    Returns ``(pending, None)``, or ``(None, reason)`` when the target
    class declares no inputs at all (structurally un-wireable — e.g. a
    field sensor) — the caller should flash `reason` and keep the ORIGINAL
    `pending` alive so the user can pick a different target."""
    inputs = input_names(cls_payload)
    if not inputs:
        return None, "target class declares no inputs"
    p = dict(pending)
    p.update(to_spec=to_id, to_is_tag=False, tag_name=None,
            inputs=inputs, input_idx=0)
    return p, None


def set_pending_target_tag(pending: dict, tag_name: str, lights, entities,
                           registry_payload) -> tuple:
    """The tag-target form (editor doc §8: "offer this when wiring to a
    tagged group") — candidate inputs are :func:`tag_input_names`'s
    intersection. Returns ``(pending, None)`` or ``(None, reason)`` exactly
    like :func:`set_pending_target_id`."""
    inputs = tag_input_names(tag_name, lights, entities, registry_payload)
    if not inputs:
        return None, f"tag '{tag_name}' has no members with a common input"
    p = dict(pending)
    p.update(to_spec=f"{_WIRE_TAG_PREFIX}{tag_name}", to_is_tag=True,
            tag_name=tag_name, inputs=inputs, input_idx=0)
    return p, None


def cycle_pending_input(pending: dict, step: int) -> dict:
    if pending["to_spec"] is None or len(pending["inputs"]) <= 1:
        return pending
    n = len(pending["inputs"])
    p = dict(pending)
    p["input_idx"] = (pending["input_idx"] + step) % n
    return p


def pending_ready_to_commit(pending: dict) -> bool:
    return pending["to_spec"] is not None


def pending_wire_strings(pending: dict) -> tuple:
    """``(from_str, to_str)`` — the dotted strings for the pending wire's
    CURRENTLY selected signal/input, exactly what
    :func:`commit_add_wire`/:func:`validate_wire` want split back apart."""
    signal = pending["signals"][pending["signal_idx"]]
    input_name = pending["inputs"][pending["input_idx"]]
    return (f"{pending['from_id']}.{signal}",
           f"{pending['to_spec']}.{input_name}")


# ---------------------------------------------------------------------------
# LOGIC overlay: selection filter, line/badge split, wire hit-testing
# ---------------------------------------------------------------------------

def wires_touching_selection(wires, selection_ids) -> list:
    """Every wire whose `from` or non-tag `to` names an id in
    `selection_ids` — the LOGIC overlay's DEFAULT filter (editor doc §8).
    Reuses `entity_selection`'s own referential-touch predicate (the exact
    rule C4's delete built for "which wires does an id-changing op touch"),
    rather than re-deriving the tag-aware endpoint-head split."""
    ids_set = set(selection_ids)
    return [w for w in wires if entity_selection._wire_touches(w, ids_set)]


def wire_endpoints_for_draw(wires, lights, entities, tile_size_m,
                            layout: dict) -> tuple:
    """Split `wires` into ``(lines, badges)`` for the LOGIC overlay draw:

    - `lines`: ``[(wire, from_pos, to_pos), ...]`` for every NON-tag-target
      wire whose both endpoints currently resolve to a tile (via
      :func:`anchor_tile` — positionless entities included, through
      `layout`). A dangling endpoint (deleted elsewhere this frame, before
      the next referential-integrity cleanup runs) is silently skipped —
      draw-time defensive, never a crash.
    - `badges`: ``[(wire, from_pos, tag_name, input_name), ...]`` for every
      tag-target wire — ONE badge per wire near its source, naming the tag
      + input, NEVER fanned out to every current member (editor doc §8 —
      the whole point of the badge is to stay readable when a tag has many
      members)."""
    lines, badges = [], []
    for w in wires:
        from_id, signal = w.from_.split(".", 1)
        to_spec, input_name = w.to.split(".", 1)
        from_pos = anchor_tile(from_id, lights, entities, tile_size_m, layout)
        if from_pos is None:
            continue
        if to_spec.startswith(_WIRE_TAG_PREFIX):
            badges.append((w, from_pos, to_spec[len(_WIRE_TAG_PREFIX):],
                          input_name))
        else:
            to_pos = anchor_tile(to_spec, lights, entities, tile_size_m,
                                 layout)
            if to_pos is None:
                continue
            lines.append((w, from_pos, to_pos))
    return lines, badges


def _point_segment_distance(px, py, x0, y0, x1, y1) -> float:
    dx, dy = x1 - x0, y1 - y0
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy)
                    / (dx * dx + dy * dy)))
    projx, projy = x0 + t * dx, y0 + t * dy
    return math.hypot(px - projx, py - projy)


WIRE_HIT_RADIUS_TILES = 0.4


def hit_test_wire(tx, ty, lines, badges, radius: float = WIRE_HIT_RADIUS_TILES):
    """The nearest wire whose LINE segment (or, for a tag wire, whose
    BADGE point) passes within `radius` tiles of tile-center `(tx, ty)`, or
    `None`. `lines`/`badges` are `wire_endpoints_for_draw`'s own output —
    tile centers (`+0.5`) so a click anywhere inside the hovered tile is
    "close enough", matching every other mode's cursor-tile granularity."""
    best, best_d = None, radius
    cx, cy = tx + 0.5, ty + 0.5
    for w, (x0, y0), (x1, y1) in lines:
        d = _point_segment_distance(cx, cy, x0 + 0.5, y0 + 0.5,
                                    x1 + 0.5, y1 + 0.5)
        if d < best_d:
            best_d, best = d, w
    for w, (bx, by), _tag, _inp in badges:
        d = math.hypot(cx - (bx + 0.5), cy - (by + 0.5))
        if d < best_d:
            best_d, best = d, w
    return best
