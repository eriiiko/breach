"""tools/light_entity_port.py — LIGHT ported onto the `light` entity class,
bespoke path deleted (Arc C1, Amendment A1).

Amendment A1 (docs/arc_c_impl_2026-07-22.md) draws a hard line: LIGHT is a
registry entity, so it ports onto ``[[entity]]`` and the bespoke path is
DELETED with parity; SPAWN is not (units are not entities) and stays on its
own bespoke-but-rehosted path untouched by this module.

The `light` registry class mirrors ``level_loader.LightEntry`` field for
field BY DESIGN (see ``src/simulation/entities/light.py``'s docstring) — so
NONE of ``map_editor.py``'s existing LIGHT-mode interactive editing code
(the B/C/R/E/P/X/H key handlers, drag, hover hit-test) needs to change. Only
the LOAD and SAVE boundaries move: instead of the deleted
``level_lib.write_lights`` bespoke writer, the editor now chooses between
``level_lib``'s two family-generic writers, "light" (legacy) and "entity"
(A3), preserving whichever family the level ALREADY used at open time
(editor doc §6 / Erik ruling 2 / canon §2: no forced migration on save).

:class:`EditableLight` is the one new piece of state: a `LightEntry`
subclass that adds the entity `id` a `[[entity]]` instance is mandatory to
carry (canon §2). Subclassing (not touching `level_loader.LightEntry`
itself — out of this arc's allowed surface) means every existing
``dataclasses.replace(lights[i], ...)`` call in map_editor.py keeps working
unchanged: ``replace()`` preserves the subclass and every field, `id`
included, across every edit.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from level_lib import color_255, format_entity_lines  # noqa: E402
from level_loader import EntityInstance, LightEntry  # noqa: E402

LIGHT_CLASS = "light"

# The light schema's field order (src/simulation/entities/light.py FIELDS):
# used so a freshly-authored entity-form light writes in a stable, readable
# order (format_entity_lines emits exactly `authored_keys`, in the order
# given). Beacon-only params are appended only for beacon lights — mirrors
# level_lib.format_light_lines' own conditional, so the AUTHORED FIELD SET a
# light gets is the same regardless of which family carries it.
_BASE_KEYS = ("x", "y", "color", "intensity", "range", "kind")
_BEACON_KEYS = ("period_s", "beam_deg", "phase")


@dataclass
class EditableLight(LightEntry):
    """A :class:`level_loader.LightEntry` plus the `[[entity]]` instance id
    AND tags (editor-only bookkeeping — NOT the schema; ``tags`` mirrors
    ``EntityInstance.tags``, Arc C4's unified-selection tag-assign target).
    ``id`` defaults to "" only so the dataclass stays constructible without
    it; every light the editor actually holds carries a real, unique id from
    the moment it enters the `lights` list (see
    :func:`initial_editable_lights` / :func:`unique_light_id`)."""
    id: str = ""
    tags: tuple = ()


def to_editable(l: LightEntry, id_: str, tags: tuple = ()) -> EditableLight:
    """Lift a plain ``LightEntry`` (as ``LevelData.lights`` carries it) into
    an :class:`EditableLight` carrying the given entity id + tags."""
    return EditableLight(x=l.x, y=l.y, color=l.color, intensity=l.intensity,
                         range=l.range, kind=l.kind, period_s=l.period_s,
                         beam_deg=l.beam_deg, phase=l.phase, id=id_,
                         tags=tuple(tags))


def unique_light_id(existing_ids) -> str:
    """``light_1`` / ``light_2`` / ... — the first id not already in
    ``existing_ids`` (the same auto-naming pattern
    ``map_editor.unique_spawn_name`` already uses for spawns)."""
    n = 1
    while f"light_{n}" in existing_ids:
        n += 1
    return f"light_{n}"


def unique_entity_id(prefix: str, *collections) -> str:
    """``{prefix}_1`` / ``{prefix}_2`` / ... — the first id free over the
    UNION of every collection's `.id` (Arc C2 invariant B3, design §7).

    ``lights`` and ``entities`` are separate LOGGED collections but both
    serialize to the ONE ``[[entity]]`` family, where ids are
    unique-or-hard-error at load. Minting a `light` via generic place-one
    while LIGHT mode already holds ``light_1`` would otherwise mint a second
    ``light_1`` -> save -> unloadable level. Every collection that serializes
    to ``[[entity]]`` MUST mint from this one union-scanning allocator
    (``lights`` u ``entities`` u future); C3 place-one/doors/sensors reuse it.
    """
    taken = set()
    for coll in collections:
        for e in coll:
            eid = getattr(e, "id", None)
            if eid:
                taken.add(eid)
    n = 1
    while f"{prefix}_{n}" in taken:
        n += 1
    return f"{prefix}_{n}"


def light_form(raw_toml: dict) -> str:
    """Which family a level's lights live in, decided ONCE at load time and
    held for the whole editor session (editor doc §6 / canon §2: preserve
    the file's existing form, never a save side effect).

    ``"legacy"`` when the file carries at least one raw ``[[light]]`` block
    (mixed legacy+entity light forms already hard-error at load —
    level_loader._parse_entities — so this is unambiguous). ``"entity"``
    otherwise: a level with ZERO lights today ALSO reports ``"entity"``,
    because the bespoke ``[[light]]`` WRITER is deleted in C1 — the first
    light placed in a light-free level is authored the modern way, which is
    not a migration (nothing legacy gets converted; there was nothing to
    convert)."""
    return "legacy" if raw_toml.get("light") else "entity"


def initial_editable_lights(lvl) -> list:
    """The editor's LIGHT-mode session state, seeded from a loaded
    ``LevelData``: preserves each light's authored `[[entity]]` id when the
    level already used entity form, else mints fresh ``light_N`` ids
    (legacy levels' `[[light]]` blocks carry no id to preserve — nothing
    references a legacy light by id, so a fresh mint is exactly as good).

    Relies on ``level_loader``'s own ordering contract: ``lvl.lights`` is
    ``[parsed legacy blocks...] + [entity-derived lights, in entity file
    order]``, and mixed forms hard-error at load — so legacy levels have
    ZERO entity-derived lights and entity-form levels have ZERO legacy ones;
    the two cases never interleave."""
    entity_light_tags = {e.id: tuple(e.tags) for e in lvl.entities
                        if e.class_name == LIGHT_CLASS}
    entity_light_ids = [e.id for e in lvl.entities if e.class_name == LIGHT_CLASS]
    ids_seen = set(entity_light_ids)
    out = []
    for i, l in enumerate(lvl.lights):
        if i < len(entity_light_ids):
            lid = entity_light_ids[i]
        else:
            lid = unique_light_id(ids_seen)
            ids_seen.add(lid)
        out.append(to_editable(l, lid, entity_light_tags.get(lid, ())))
    return out


def light_entry_to_fields(l: LightEntry) -> dict:
    """A light's EFFECTIVE `[[entity]]` field dict (colour back to 0-255
    ints — the toml schema's units, ``level_lib.color_255``, the same
    conversion ``format_light_lines`` applies)."""
    r, g, b = color_255(l.color)
    return {
        "x": float(l.x), "y": float(l.y), "color": [r, g, b],
        "intensity": float(l.intensity), "range": float(l.range),
        "kind": str(l.kind), "period_s": float(l.period_s),
        "beam_deg": float(l.beam_deg), "phase": float(l.phase),
    }


def light_authored_keys(kind: str) -> tuple:
    """Which fields get WRITTEN for a light of this kind — beacon params
    only for beacon lights, mirroring ``format_light_lines``'s own
    conditional (P4 §2.2) so the authored field SET a light gets is
    unchanged by which family carries it."""
    return _BASE_KEYS + (_BEACON_KEYS if kind == "beacon" else ())


def light_to_entity_instance(l: EditableLight, ordinal: int) -> EntityInstance:
    """One :class:`EditableLight` -> the `[[entity]]` instance
    ``level_lib.format_entity_lines`` serializes. ``ordinal`` is cosmetic
    here (format_entity_lines never reads it — file POSITION, not this
    field, determines the runtime ordinal on next load); callers pass the
    instance's position in the list being written."""
    return EntityInstance(
        id=l.id, class_name=LIGHT_CLASS, ordinal=int(ordinal),
        tags=tuple(l.tags),
        fields=light_entry_to_fields(l),
        authored_keys=light_authored_keys(l.kind))


def merge_light_entities(original_entities, lights: list) -> list:
    """Rebuild the full `[[entity]]` array for a save.

    Every ORIGINAL non-light entity keeps its exact file position
    (untouched — C1 has no tool that edits doors/zones yet). Every
    ORIGINAL light instance's slot is replaced by its CURRENT (possibly
    edited) :class:`EditableLight` if a light with that id still exists in
    ``lights``, or dropped if the user deleted it. Any BRAND NEW light (an
    id the original file never had) is appended at the end, in the order it
    was created.

    This keeps a save's file-order churn to "what actually changed" instead
    of shuffling every unrelated entity. That matters only for the
    ENTITY_SECT_V1 digest's row order (engine/16 §4, ordinal-keyed) — no
    wire/tag semantics can depend on array position (canon §2: "all
    references address ids, never array positions"), so a full reshuffle
    would not be a correctness bug, just unnecessary churn this avoids.
    """
    by_id = {l.id: l for l in lights}
    seen = set()
    out = []
    for e in original_entities:
        if e.class_name == LIGHT_CLASS:
            cur = by_id.get(e.id)
            if cur is not None:
                out.append(cur)
                seen.add(e.id)
            # else: this light was deleted this session — drop its slot.
        else:
            out.append(e)
    for l in lights:
        if l.id not in seen:
            out.append(l)
    return [light_to_entity_instance(o, i) if isinstance(o, EditableLight)
            else o for i, o in enumerate(out)]


def format_lights_for_save(light_form_: str, lights: list,
                           other_entities: list) -> dict:
    """The save-time managed-block replacement(s) for a level's lights —
    the ONE place that decides which level_lib family carries them,
    preserving whichever the level was already using (see
    :func:`light_form`). Returns a dict suitable for merging into the
    caller's ``level_lib`` ``replacements`` (either ``{"light": ...}`` or
    ``{"entity": ...}`` — never both, and never the family the level did
    NOT already use, so an untouched family is left for level_lib's own
    byte-preservation to handle)."""
    if light_form_ == "legacy":
        from level_lib import format_light_lines
        return {"light": lambda nl: format_light_lines(lights, nl)}
    merged = merge_light_entities(other_entities, lights)
    return {"entity": lambda nl: format_entity_lines(merged, nl)}


def light_and_entity_replacements(light_form_: str, lights: list,
                                  entities: list) -> dict:
    """The COMPLETE lights-AND-entities save replacement set (Arc C3).

    :func:`format_lights_for_save` decides ONLY the light family and
    omits ``"entity"`` entirely on a legacy-light level (C1's contract,
    unchanged above) — that omission meant "leave it byte-preserved" back
    when nothing but lights ever mutated ``entities``. Arc C3's placement
    tools (DOOR/sensor/generic place-one) now append to this SAME
    ``entities`` list regardless of which family lights use, so a legacy-
    light level's ``"entity"`` family must be written from the live
    ``entities`` on EVERY save, or anything placed this session would
    silently vanish. When ``light_form_ == "entity"`` the merge already
    folds ``entities`` in (:func:`merge_light_entities`), so adding it here
    would be a no-op — the ``"entity" not in repl`` guard skips exactly
    that case."""
    repl = format_lights_for_save(light_form_, lights, entities)
    if "entity" not in repl:
        repl["entity"] = lambda nl: format_entity_lines(entities, nl)
    return repl
