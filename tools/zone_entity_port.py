r"""tools/zone_entity_port.py — the ZONE paint tool's entity-authoring
bridge (Arc C5, editor doc §5, canon engine/16 §7).

Each painted `zones.npy` id binds to exactly one `breach_site`/
`extraction_zone` `[[entity]]` instance carrying `zone_id` = that paint id
— the ONLY schema-required field either zone class declares (both are
`intangible`, no `x`/`y`: a zone is defined entirely by its paint, never a
point). Painting a BRAND NEW id therefore authors a fresh instance in the
SAME compound transaction as the grid stamp (the C3 DOOR archetype, C2's
own forward note for C5: "GridCellsOp('zones') — register those grids in
ctx.grids only once allocated"); repainting an id that ALREADY has a
claiming instance is a plain grid-only paint — nothing else to author.
Painting id A over id B's tiles needs no special code either: the cells
simply take A's value, so B's own painted-tile count shrinks as a natural
side effect of the same stamp (editor design §5 — "painting id A over id B
just shrinks B").

Binding validators are NOT reimplemented here: `level_loader.
_validate_zone_binding` already encodes the §5 rules (duplicate zone_id =
load error; zero-tile instance / orphaned paint = warnings) and is the
loader's own load-time gate. :func:`zone_binding_summary` runs that SAME
function live (capturing its `warnings.warn` calls instead of letting them
print to stderr) so the editor's status-bar validator slot shows the
identical verdict a save+reload would produce, reused, not re-derived.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import level_loader  # noqa: E402
from level_loader import ZONE_CLASSES, EntityInstance  # noqa: E402

import entity_editor_ui  # noqa: E402

ZONE_ID_MIN, ZONE_ID_MAX = 1, 255


def zone_ids_in_use(entities) -> set:
    """Every `zone_id` currently claimed by a zone-class instance."""
    return {int(e.fields["zone_id"]) for e in entities
           if e.class_name in ZONE_CLASSES}


def next_zone_id(entities) -> int:
    """The smallest unclaimed id in `[1, 255]` — the "new zone" default
    (editor design §5's id namespace, one space across both zone classes).
    Returns 256 (out of range) in the pathological all-255-ids-claimed case;
    callers that mint from this MUST still be prepared for that — the
    schema's own `maximum: 255` bound will refuse it, loudly, rather than
    silently wrapping to a colliding id."""
    used = zone_ids_in_use(entities)
    n = ZONE_ID_MIN
    while n in used and n <= ZONE_ID_MAX:
        n += 1
    return n


def find_zone_claim(entities, zone_id: int):
    """The instance claiming `zone_id`, or `None` (mirrors the loader's own
    `claims` scan — first match in file order; §5 guarantees at most one
    non-error claim on a level that already loaded)."""
    zid = int(zone_id)
    for e in entities:
        if e.class_name in ZONE_CLASSES and int(e.fields["zone_id"]) == zid:
            return e
    return None


def build_zone_instance(cls_payload: dict, zone_class: str, zone_id: int,
                        id_: str) -> EntityInstance:
    """A freshly-painted zone's `[[entity]]` instance: every OTHER field at
    its registry default (`entity_editor_ui.default_instance_fields` — the
    SAME generic template the C3 place-one path already builds from; a zone
    class declares no `x`/`y` so the override is a no-op there), `zone_id`
    overridden to the newly painted id, `authored_keys` = just `zone_id`
    (the class's one required field — `roster`/`faction` stay at their
    schema defaults until an inspector edit authors them, C4's
    `entity_selection.commit_field_edit`, which already generalizes to any
    field with no C5 changes needed)."""
    fields = entity_editor_ui.default_instance_fields(cls_payload)
    fields["zone_id"] = int(zone_id)
    required = entity_editor_ui.required_field_names(cls_payload)
    return EntityInstance(id=id_, class_name=zone_class, ordinal=0, tags=(),
                          fields=fields, authored_keys=required)


def commit_zone_paint(log, zones, entities, region, zone_id: int,
                      new_instance=None):
    """Paint `zone_id` into `zones` over `region` (a set of `(tx, ty)`) as
    ONE transaction: `GridCellsOp("zones")` always, plus `CollectionOp
    ("entities")` when `new_instance` is given — a brand-new id being
    painted for the first time (the C3/DOOR compound-transaction pattern:
    ONE `begin`/`commit` pair covers both ops). `zones` must already be
    registered in the log's `UndoContext` (`ctx.grids["zones"]` — the
    caller allocates + registers it on the level's FIRST zone paint, per
    C2's own forward note; this function never allocates). Returns the
    committed Transaction, or `None` when `region` is empty."""
    if not region:
        return None
    log.begin("paint zone")
    log.snapshot_grid("zones")
    if new_instance is not None:
        log.snapshot_coll("entities")
    for tx, ty in region:
        zones[ty, tx] = zone_id
    if new_instance is not None:
        entities.append(new_instance)
    return log.commit()


def commit_zone_clear(log, zones, region):
    """Clear `region`'s `zones.npy` paint to 0 (the same-code-select
    "erase this zone's paint" gesture) as ONE `GridCellsOp("zones")`
    transaction. Returns the committed Transaction, or `None` when `region`
    is empty or already all-zero there (a genuine no-op — `commit()` drops
    an unchanged grid)."""
    if not region:
        return None
    log.begin("clear zone paint")
    log.snapshot_grid("zones")
    for tx, ty in region:
        zones[ty, tx] = 0
    return log.commit()


def zone_binding_summary(zone_grid, entities, toml_path="<editor>") -> str:
    """Run the loader's OWN §5 validators live (capturing `warnings.warn`
    instead of letting them print to stderr) so the editor's status-bar
    validator slot shows the IDENTICAL verdict a save+reload would produce
    — reused, not re-derived. Returns ``"zones ok"`` when clean, else the
    joined warning text; a duplicate-id `ValueError` (the one hard-error
    case) is caught and reported the same way — the editor never crashes on
    a validator failure, it just surfaces it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            level_loader._validate_zone_binding(zone_grid, entities, toml_path)
        except ValueError as e:
            return f"zone error: {e}"
    if not caught:
        return "zones ok"
    return "; ".join(str(w.message) for w in caught)
