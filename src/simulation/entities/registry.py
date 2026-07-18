"""entities.toml tuning overlay + registry hash/export (entity design §2/§3b).

Mirrors the materials.py pattern: schema and ids live in code, tuning NUMBERS
live in a TOML overlay. `entities.toml` (repo root, beside config.toml) may
override numeric field DEFAULTS only — a key naming a class or field that
does not exist in the registry is a hard error ("No schema in TOML", §3b),
and non-number values are rejected outright.

Hot-reload constraint (design §2): the overlay is a DEV-ONLY affordance. In
any lockstep session or ML rollout the registry content-hash
(:func:`registry_content_hash`) is part of match setup, like the seed;
mid-run reload is disabled, and changing a number that alters behavior is a
deliberate golden re-baseline event.

`entity_registry.json` (repo root, gitignored — a committed copy would go
stale) is the editor's last-good fallback (design §3b): rewritten on every
successful game launch, read by the editor only when its direct
``import simulation.entities`` fails on a half-written class.
"""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path

from simulation.entities.schema import (
    ALIVE_SIGNAL, INSTANCE_FIELDS, KIND_INT, KIND_Q16, NUMERIC_KINDS,
    REGISTRY, all_signals,
)

ROOT = Path(__file__).resolve().parents[3]
ENTITIES_TOML = ROOT / "entities.toml"
REGISTRY_JSON = ROOT / "entity_registry.json"

SCHEMA_VERSION = 1


class EntityTomlError(ValueError):
    """entities.toml violates the overlay rules (unknown key / non-number)."""


# Applied overrides: {class_name: {field_name: number}}. Module-level like
# REGISTRY; :func:`clear_tuning_overlay` resets it (tests, dev reload).
_TUNING: dict[str, dict] = {}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def apply_tuning_overlay(path=None, registry: dict = None) -> dict:
    """Load entities.toml and apply its numeric overrides onto the registry
    defaults (design §2 L1). Returns the ``{class: {field: value}}`` dict.

    A missing file is a no-op (the overlay is optional, dev-only). Everything
    else hard-errors: unknown class, unknown field, non-numeric field kind,
    non-number value, int-kind override with a float, out-of-bounds value.
    Declared :class:`Field` defaults are never mutated — overrides live in a
    side table consulted by :func:`effective_defaults`.
    """
    path = ENTITIES_TOML if path is None else Path(path)
    reg = REGISTRY if registry is None else registry
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    for cls_name, table in raw.items():
        if cls_name not in reg:
            raise EntityTomlError(
                f"{path.name}: unknown entity class '{cls_name}' — no schema "
                f"in TOML (design §3b); registered classes: "
                f"{sorted(reg)}")
        if not isinstance(table, dict):
            raise EntityTomlError(
                f"{path.name}: '{cls_name}' must be a table of "
                f"field = number overrides, got {table!r}")
        fields = {f.name: f for f in reg[cls_name].FIELDS}
        for key, value in table.items():
            if key not in fields:
                raise EntityTomlError(
                    f"{path.name}: [{cls_name}] names unknown field '{key}' "
                    f"— no schema in TOML (design §3b); declared fields: "
                    f"{sorted(fields)}")
            fld = fields[key]
            if fld.kind not in NUMERIC_KINDS:
                raise EntityTomlError(
                    f"{path.name}: [{cls_name}] '{key}' has kind "
                    f"'{fld.kind}' — the overlay carries tuning NUMBERS "
                    f"only (design §3b)")
            if not _is_number(value):
                raise EntityTomlError(
                    f"{path.name}: [{cls_name}] '{key}' = {value!r} — the "
                    f"overlay carries tuning NUMBERS only (design §3b)")
            if fld.kind in (KIND_INT, KIND_Q16) and not isinstance(value, int):
                raise EntityTomlError(
                    f"{path.name}: [{cls_name}] '{key}' = {value!r} — kind "
                    f"'{fld.kind}' is integer-domain (Q16.16 / int); floats "
                    f"never enter synced state")
            if fld.minimum is not None and value < fld.minimum:
                raise EntityTomlError(
                    f"{path.name}: [{cls_name}] '{key}' = {value!r} below "
                    f"minimum {fld.minimum!r}")
            if fld.maximum is not None and value > fld.maximum:
                raise EntityTomlError(
                    f"{path.name}: [{cls_name}] '{key}' = {value!r} above "
                    f"maximum {fld.maximum!r}")
            _TUNING.setdefault(cls_name, {})[key] = value
    return {k: dict(v) for k, v in _TUNING.items()}


def clear_tuning_overlay() -> None:
    """Drop all applied overrides (tests / dev reload)."""
    _TUNING.clear()


def effective_defaults(cls_name: str, registry: dict = None) -> dict:
    """``{field: default}`` for one class with the overlay applied on top."""
    reg = REGISTRY if registry is None else registry
    if cls_name not in reg:
        raise KeyError(f"unknown entity class '{cls_name}'")
    over = _TUNING.get(cls_name, {})
    return {f.name: over.get(f.name, f.default) for f in reg[cls_name].FIELDS}


# ---------------------------------------------------------------------------
# Serialization: one payload shape feeds BOTH the content hash and the JSON
# export, so what the editor falls back on is exactly what got hashed.
# ---------------------------------------------------------------------------
def _field_payload(f, default) -> dict:
    return {
        "name": f.name,
        "kind": f.kind,
        "default": list(default) if isinstance(default, tuple) else default,
        "minimum": f.minimum,
        "maximum": f.maximum,
        "choices": list(f.choices) if f.choices else None,
        "doc": f.doc,
    }


def registry_payload(registry: dict = None) -> dict:
    """The full registry as plain JSON-able data: classes, fields (with
    overlay-effective defaults), signal/input vocabulary, instance facts."""
    reg = REGISTRY if registry is None else registry
    classes = {}
    for cls_name in sorted(reg):
        cls = reg[cls_name]
        eff = effective_defaults(cls_name, registry=reg)
        classes[cls_name] = {
            "intangible": cls.INTANGIBLE,
            "fields": [_field_payload(f, eff[f.name]) for f in cls.FIELDS],
            "signals": [{"name": s.name, "doc": s.doc}
                        for s in all_signals(cls)],
            "inputs": [{"name": i.name, "mode": i.mode, "doc": i.doc}
                       for i in cls.INPUTS],
            "input_priority": [list(g) for g in cls.INPUT_PRIORITY],
            "interactions": list(cls.INTERACTIONS),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_fields": [_field_payload(f, f.default)
                            for f in INSTANCE_FIELDS],
        "free_signals": [{"name": ALIVE_SIGNAL.name, "doc": ALIVE_SIGNAL.doc}],
        "classes": classes,
    }


def registry_content_hash(registry: dict = None) -> str:
    """Deterministic sha256 over the canonical registry serialization.

    Match-setup material (design §2): sorted keys + compact separators +
    ascii-only, so the same schema + overlay hashes identically on every
    machine regardless of dict insertion order or platform.
    """
    canonical = json.dumps(registry_payload(registry), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def export_registry_json(path=None, registry: dict = None) -> Path:
    """Write entity_registry.json — the editor's last-good fallback (§3b).

    Called on every successful game launch (main.py, post-init) so the
    fallback can never lag a class the game itself accepted. Temp + rename
    so a crash mid-write can't leave a truncated fallback.
    """
    path = REGISTRY_JSON if path is None else Path(path)
    payload = registry_payload(registry)
    payload["content_hash"] = registry_content_hash(registry)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path
