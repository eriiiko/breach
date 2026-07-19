"""Canonical entity/signal digest serialization — THE one serializer (A4).

The A4 impl note (docs/a4_digest_impl_note_2026-07-18.md, v2) pins one
module-level :func:`serialize_entity_state` consumed by BOTH the tick digest
(tests/field_digest.py) and the recorder (simulation/recorder.py) — never two
serializers (critique 9). It lives in this package because it is pure schema
machinery: stdlib only (the import-light rule, design §3b — no numpy, no
``simulation.simulation``), so the digest code in tests/ and the recorder in
src/ import the SAME bytes recipe.

Format (section-local version ``ENTITY_SECT_V1`` — a future change bumps the
preamble loudly without touching entity-free digests; ``DIGEST_SPEC_VERSION``
stays 1 globally, critique 3):

- preamble ``ENTITY_SECT_V1\\n``, then per entity in ORDINAL (file/id) order
  (§3a single ordering rule):
  - ASCII header ``"{ordinal}|{id}|{class_name}\\n"``
  - each SYNCED-KIND declared field in schema DECLARATION order:
    ``"{field_name}|"`` + signed little-endian int64 (``struct.pack('<q')``,
    out-of-range raises loudly) + ``"\\n"``. Synced kinds: int, q16, bool
    (0/1), enum (declared-choice index), entity_ref (target ordinal; -1 for
    unwired "" AND dangling — both resolve to nothing at runtime).
    EXCLUDED: float_render, str, color_rgb, str_list, roster, AND length_m
    (critique blocker 1) — length_m is authoring-bound, stored unquantized;
    its synced consequence is quantized tile state already hashed via
    material/obstacles/wall_hp. No digest-time quantization, ever.
  - ``"alive|"`` + int64 0/1 (always 1 in Arc A — no destruction path).
  - the per-class RUNTIME-STATE row block (same ``name|int64\\n`` encoding),
    empty in Arc A — :meth:`Entity.runtime_digest_rows` fills it per class
    (A6 door state, Arc B accumulators) with zero mechanism surgery.
  - record terminator ``"\\n"``.

Injectivity does not rest on the registry being closed: headers and rows are
newline-delimited with fixed 8-byte values, and the registry rejects
field/class/signal names outside ``[A-Za-z0-9_]+`` at registration
(critique 10); instance ids are loader-guarded slugs, re-checked here.

``__signals__`` (``SIGNAL_SECT_V1``) is defined now, empty until Arc B's
SignalBus: ``(emitter_ordinal, signal_name, int64 value)`` tuples sorted by
(ordinal, name). The free ``alive`` signal is EXCLUDED — hashed ONLY as the
``__entity__`` row (critique 7), so the bus's introduction cannot flip
digests before behavior changes.

Presence (the strict carrier, critique blocker 2): capture paths ALWAYS
write snapshot key :data:`ENTITY_DIGEST_KEY` via :func:`entity_carrier`
(``n_entities == 0`` for an entity-free level); the fold is gated on
``n_entities > 0`` so dormancy is untouched. :func:`require_entity_carrier`
raises when a sim with entities meets a snapshot without the key — an
entity-present run can never silently hash entity-free. Pre-A4 snapshots
(no key at all) are entity-free by construction.

Registry provenance (critique 4+5): entity-present digests are only
comparable at equal :func:`~simulation.entities.registry.
registry_content_hash` — match-setup material, like the seed. The carrier
records it; the xarch artifact line and recorder metadata surface it.
"""
from __future__ import annotations

import re
import struct

from simulation.entities.registry import registry_content_hash
from simulation.entities.schema import (
    KIND_BOOL, KIND_ENTITY_REF, KIND_ENUM, KIND_INT, KIND_Q16, REGISTRY,
)

# Snapshot key for the presence carrier — the __unit_state__ idiom: cannot
# collide with a gmap field name, special-cased by diff_trajectories.
ENTITY_DIGEST_KEY = "__entity__"

# Hashed section preambles — the SECTION-LOCAL versions (critique 3).
ENTITY_SECT_PREAMBLE = b"ENTITY_SECT_V1\n"
SIGNAL_SECT_PREAMBLE = b"SIGNAL_SECT_V1\n"

# The synced-kind partition (A4 impl note, critique blocker 1): exactly these
# declared kinds enter the hashed byte stream.
SYNCED_FIELD_KINDS = (KIND_INT, KIND_Q16, KIND_BOOL, KIND_ENUM,
                      KIND_ENTITY_REF)

# Names embedded as ASCII tokens between '|' delimiters. Declared names are
# registration-guarded (schema.py); runtime-row names come from class code,
# so they are re-checked at serialization. Ids follow the loader's slug rule
# (level_loader._ENTITY_ID_RE — '-' allowed, '|'/newlines impossible).
_NAME_RE = re.compile(r"[A-Za-z0-9_]+\Z")
_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]*\Z")


def _pack_i64(context: str, value) -> bytes:
    """``struct.pack('<q')`` with a loud, attributable overflow/type error."""
    try:
        return struct.pack("<q", value)
    except (struct.error, OverflowError, TypeError) as e:
        raise OverflowError(
            f"{context}: value {value!r} does not fit a signed little-endian "
            f"int64 (ENTITY_SECT_V1 row encoding)") from e


def _synced_row_value(f, value, ordinals: dict, context: str) -> int:
    """One declared field's int64 row value per its kind (see module doc)."""
    if f.kind in (KIND_INT, KIND_Q16):
        return value                     # already integer-domain (validated)
    if f.kind == KIND_BOOL:
        return 1 if value else 0
    if f.kind == KIND_ENUM:
        try:
            return f.choices.index(value)
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"{context}: enum value {value!r} not in declared choices "
                f"{f.choices!r}") from e
    if f.kind == KIND_ENTITY_REF:
        # Unwired ("") AND dangling both encode -1 — they resolve to
        # nothing at runtime, so they must hash alike.
        if not value:
            return -1
        return ordinals.get(value, -1)
    raise ValueError(f"{context}: kind {f.kind!r} is not a synced kind "
                     f"({SYNCED_FIELD_KINDS})")


def entity_records(entities, registry: dict = None) -> tuple:
    """One serialized ``ENTITY_SECT_V1`` record (bytes) per entity, in
    ordinal order — the per-instance granularity that lets the A/B harness
    LOCATE an entity divergence the way it locates a field one per-cell.

    Reads the RUNTIME entity object; Arc A's ``EntityInstance`` is its
    degenerate load-constant form (critique 6) — duck-typed on ``ordinal``,
    ``id``, ``class_name``, ``fields`` (and an optional ``alive``).
    """
    reg = REGISTRY if registry is None else registry
    ordered = sorted(entities, key=lambda e: int(e.ordinal))
    ordinals = {e.id: int(e.ordinal) for e in ordered}
    records = []
    for e in ordered:
        if not _ID_RE.fullmatch(e.id):
            raise ValueError(
                f"entity id {e.id!r} outside the loader slug charset — "
                f"cannot serialize an unambiguous ENTITY_SECT_V1 header")
        cls = reg[e.class_name]
        parts = [f"{int(e.ordinal)}|{e.id}|{e.class_name}\n".encode("ascii")]
        for f in cls.FIELDS:             # schema DECLARATION order, always
            if f.kind not in SYNCED_FIELD_KINDS:
                continue                 # render/authoring-bound: never hashed
            ctx = f"entity '{e.id}' field '{f.name}'"
            v = _synced_row_value(f, e.fields[f.name], ordinals, ctx)
            parts.append(f.name.encode("ascii") + b"|"
                         + _pack_i64(ctx, v) + b"\n")
        alive = 1 if getattr(e, "alive", True) else 0
        parts.append(b"alive|"
                     + _pack_i64(f"entity '{e.id}' alive", alive) + b"\n")
        for name, v in cls.runtime_digest_rows(e):
            if not _NAME_RE.fullmatch(name):
                raise ValueError(
                    f"entity '{e.id}' runtime row name {name!r} outside "
                    f"[A-Za-z0-9_]+ (ENTITY_SECT_V1 token charset)")
            parts.append(name.encode("ascii") + b"|"
                         + _pack_i64(f"entity '{e.id}' runtime row "
                                     f"'{name}'", v) + b"\n")
        parts.append(b"\n")              # record terminator
        records.append(b"".join(parts))
    return tuple(records)


def serialize_entity_state(entities, registry: dict = None) -> bytes:
    """THE canonical ``__entity__`` section bytes: preamble + all records.

    The single serializer both the digest and the recorder consume
    (critique 9) — the recorder persists exactly these bytes; the digest
    hashes exactly these bytes.
    """
    return ENTITY_SECT_PREAMBLE + b"".join(entity_records(entities, registry))


def serialize_signal_state(signals=()) -> bytes:
    """The ``__signals__`` section bytes — empty-but-defined until Arc B.

    ``signals`` is an iterable of ``(emitter_ordinal, signal_name, value)``
    tuples, hashed sorted by (ordinal, name). An empty bus hashes as the
    bare preamble (stable). The free ``alive`` signal is REFUSED here —
    hashed ONLY as the ``__entity__`` row (critique 7).
    """
    parts = [SIGNAL_SECT_PREAMBLE]
    for ordinal, name, value in sorted(signals,
                                       key=lambda s: (int(s[0]), s[1])):
        if name == "alive":
            raise ValueError(
                "the free 'alive' signal is hashed ONLY as the __entity__ "
                "alive row (A4 impl note, critique 7) — never via "
                "__signals__")
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                f"signal name {name!r} outside [A-Za-z0-9_]+ "
                f"(SIGNAL_SECT_V1 token charset)")
        ctx = f"signal ({ordinal}, '{name}')"
        parts.append(f"{int(ordinal)}|{name}|".encode("ascii")
                     + _pack_i64(ctx, value) + b"\n")
    return b"".join(parts)


def entity_carrier(entities, registry: dict = None, signals=()) -> dict:
    """The ``__entity__`` snapshot value — the strict presence carrier.

    ALWAYS written by capture paths (``n_entities == 0`` for an entity-free
    level; the fold is gated on ``n_entities > 0``, so dormancy holds).
    Carries the serialized payload (per-record, for per-instance locating),
    not just a hash, plus the registry content-hash provenance.
    """
    ents = list(entities) if entities else []
    if not ents:
        return {"n_entities": 0, "records": (), "signals": (),
                "registry_hash": ""}
    return {
        "n_entities": len(ents),
        "records": entity_records(ents, registry),
        "signals": tuple(signals),
        "registry_hash": registry_content_hash(registry),
    }


def entity_section_bytes(carrier: dict) -> bytes:
    """The hashed ``__entity__`` bytes for a carrier — identical to
    :func:`serialize_entity_state` on the same entities (one serializer)."""
    return ENTITY_SECT_PREAMBLE + b"".join(carrier["records"])


def signal_section_bytes(carrier: dict) -> bytes:
    """The hashed ``__signals__`` bytes for a carrier."""
    return serialize_signal_state(carrier.get("signals", ()))


def require_entity_carrier(entities, snapshot: dict) -> None:
    """Strictness (loud, like a missing field — critique blocker 2): a sim
    with entities loaded may never meet a snapshot lacking the carrier, or
    a capture path could silently compute the entity-free digest for an
    entity-present run. Entity-free sims accept pre-A4 snapshots (no key)."""
    if entities and ENTITY_DIGEST_KEY not in snapshot:
        raise KeyError(
            f"snapshot is missing the '{ENTITY_DIGEST_KEY}' presence "
            f"carrier but the sim has {len(list(entities))} entities loaded "
            f"— an entity-present run must never hash entity-free (A4 "
            f"strict presence rule)")
