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
    ALIVE_SIGNAL, ALL_INPUT_MODES, ALL_KINDS, Entity, EntitySchemaError, Field,
    INPUT_AND, INPUT_EDGE, INPUT_HELD, INPUT_SINGLE, INSTANCE_FIELDS, InputDecl,
    KIND_BOOL, KIND_COLOR_RGB, KIND_ENTITY_REF, KIND_ENUM, KIND_FLOAT_RENDER,
    KIND_INT, KIND_LENGTH_M, KIND_Q16, KIND_ROSTER, KIND_STR, KIND_STR_LIST,
    NUMERIC_KINDS, REGISTRY, Signal, all_signals, field_value_error, register,
)
from simulation.entities.registry import (  # noqa: F401
    ENTITIES_TOML, EntityTomlError, REGISTRY_JSON, apply_tuning_overlay,
    clear_tuning_overlay, effective_defaults, export_registry_json,
    registry_content_hash, registry_payload,
)
from simulation.entities.serialize import (  # noqa: F401
    ENTITY_DIGEST_KEY, ENTITY_SECT_PREAMBLE, SIGNAL_SECT_PREAMBLE,
    SYNCED_FIELD_KINDS, entity_carrier, entity_records, entity_section_bytes,
    require_entity_carrier, serialize_entity_state, serialize_signal_state,
    signal_section_bytes,
)
from simulation.entities import light as _light  # noqa: F401  (registers the exemplar)
from simulation.entities import zones as _zones  # noqa: F401  (registers breach_site/extraction_zone, A8)
from simulation.entities import door as _door    # noqa: F401  (registers the door class, A6)
from simulation.entities import controls as _controls  # noqa: F401  (registers button/terminal, B1 — inert)
from simulation.entities import nodes as _nodes  # noqa: F401  (registers decider/gate_*/filter, B2)
from simulation.entities import sensors as _sensors  # noqa: F401  (registers the v1 sensor catalog, B3)
from simulation.entities import actuators as _actuators  # noqa: F401  (registers the airlock_controller, B5)
from simulation.entities import cover as _cover  # noqa: F401  (registers the cover class — onephase_wego §7)
