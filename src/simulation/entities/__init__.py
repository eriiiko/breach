"""Entity registry package (entity design v2, LOCKED 2026-07-18 — §2/§3b).

One model for everything placeable that isn't paintable matter. This package
is the L1 vocabulary layer: :class:`Entity` subclasses declare their schema
(fields, signals, inputs) as class-level constants; :func:`register` adds
them to :data:`REGISTRY`; `entities.toml` overrides tuning numbers;
`entity_registry.json` is the editor's last-good fallback.

IMPORT-LIGHT (design §3b, CI-tested in tests/test_entities_import_light.py):
importing this package must succeed with no compiled ``breach_physics``
present and must never pull in ``simulation.simulation`` — the editor
imports it directly, like ``simulation.materials``. Keep it stdlib-only.

Arc A patch 1 ships schema machinery ONLY: no runtime behavior executes
anywhere; the sim loop does not know this package exists yet.
"""
from __future__ import annotations

from simulation.entities.schema import (  # noqa: F401
    ALIVE_SIGNAL, ALL_KINDS, Entity, EntitySchemaError, Field, INPUT_EDGE,
    INPUT_HELD, INSTANCE_FIELDS, InputDecl, KIND_BOOL, KIND_COLOR_RGB,
    KIND_ENTITY_REF, KIND_ENUM, KIND_FLOAT_RENDER, KIND_INT, KIND_LENGTH_M,
    KIND_Q16, KIND_STR, KIND_STR_LIST, NUMERIC_KINDS, REGISTRY, Signal,
    all_signals, field_value_error, register,
)
from simulation.entities.registry import (  # noqa: F401
    ENTITIES_TOML, EntityTomlError, REGISTRY_JSON, apply_tuning_overlay,
    clear_tuning_overlay, effective_defaults, export_registry_json,
    registry_content_hash, registry_payload,
)
from simulation.entities import light as _light  # noqa: F401  (registers the exemplar)
