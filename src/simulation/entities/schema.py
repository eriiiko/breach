"""Entity schema vocabulary + the class registry (entity design §2/§3b).

The schema LIVES IN CODE (design §3b, DECIDED): each entity kind is a Python
subclass of :class:`Entity` declaring its fields / signals / inputs as
class-level constants beside its (future, Arc B) L0 behavior; the
:func:`register` decorator adds it to the module-level :data:`REGISTRY` keyed
by class name. `entities.toml` (see :mod:`simulation.entities.registry`) may
override tuning NUMBERS only — never shape ("No schema in TOML").

Import-light rule (design §3b, CI-tested): this module and the whole
``simulation.entities`` package import stdlib only — no ``breach_physics``,
no ``simulation.simulation`` — so the editor can import the registry directly
even when the compiled physics is absent.

Field kinds carry the determinism story at the declaration level:

- ``KIND_Q16`` values are Q16.16 fixed-point INTEGERS — anything destined for
  synced sim state declares this kind and stores an int default, never a
  float (the ingress rule, docs/lenovo_dev_setup.md §8b).
- ``KIND_LENGTH_M`` values are meters-first authoring numbers (editor design
  §4): stored as declared here, quantized ONCE at load by the canonical
  rule when a loader consumes them. Never used raw in the sim path.
- ``KIND_FLOAT_RENDER`` values are render-local floats that NEVER enter
  synced state (same class as ``light_rgb`` — see LightEntry's contract in
  level_loader.py).
- ``KIND_ENTITY_REF`` values are strings naming another ``[[entity]]``
  instance id (design §3a: references address ids, never positions). The
  A3 loader WARNS on a ref naming a missing id (authoring error, not fatal
  — a destroyed entity at runtime is not an error either) and HARD-ERRORS
  on a ref naming a ``[[spawn]]`` unit: units are not entities until the
  stack-2 convergence (design §3e). Empty string = unwired.

Instance-level facts (design §3a/§3c) are schema too: every ``[[entity]]``
instance carries a mandatory unique ``id`` and a ``tags`` list. They are
declared once here (:data:`INSTANCE_FIELDS`), not per class — the A3 loader
patch enforces them; this module only states them.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


class EntitySchemaError(ValueError):
    """A class declaration violates the schema rules (raised at register)."""


# ---------------------------------------------------------------------------
# Field kinds — the closed vocabulary of declared value types.
# ---------------------------------------------------------------------------
KIND_INT = "int"                    # plain integer (counts, ids, team ints)
KIND_Q16 = "q16"                    # Q16.16 fixed-point integer (synced domain)
KIND_LENGTH_M = "length_m"          # meters-first length; quantized at load
KIND_BOOL = "bool"
KIND_STR = "str"
KIND_ENUM = "enum"                  # str constrained to `choices`
KIND_FLOAT_RENDER = "float_render"  # render-local float; never synced
KIND_COLOR_RGB = "color_rgb"        # (r, g, b) 0-255 ints; render-local
KIND_STR_LIST = "str_list"          # list of strings (tags)
KIND_ENTITY_REF = "entity_ref"      # str naming another [[entity]] instance id

ALL_KINDS = (KIND_INT, KIND_Q16, KIND_LENGTH_M, KIND_BOOL, KIND_STR,
             KIND_ENUM, KIND_FLOAT_RENDER, KIND_COLOR_RGB, KIND_STR_LIST,
             KIND_ENTITY_REF)

# Kinds the entities.toml tuning overlay may override — NUMBERS only.
NUMERIC_KINDS = (KIND_INT, KIND_Q16, KIND_LENGTH_M, KIND_FLOAT_RENDER)

# Input modes (design §4): *while-held* inputs OR across all driving wires;
# *edge* inputs fire once per tick regardless of how many wires pulse them.
INPUT_HELD = "held"
INPUT_EDGE = "edge"


@dataclass(frozen=True)
class Field:
    """One declared schema field: name, kind, default, constraints.

    ``default=None`` means the field is REQUIRED at authoring time (no
    default exists — e.g. the instance ``id``). ``minimum``/``maximum`` are
    inclusive bounds for numeric kinds; ``choices`` is the closed value set
    for :data:`KIND_ENUM`.
    """
    name: str
    kind: str
    default: object = None
    minimum: object = None
    maximum: object = None
    choices: tuple = None
    doc: str = ""


@dataclass(frozen=True)
class Signal:
    """One emitted signal (design §4): integer-valued, Q16.16 where physical."""
    name: str
    doc: str = ""


@dataclass(frozen=True)
class InputDecl:
    """One accepted input, marked while-held vs edge (design §4)."""
    name: str
    mode: str  # INPUT_HELD | INPUT_EDGE
    doc: str = ""


# The free `alive` signal (design §4): every entity emits it, 1 while
# functional. A destroyed entity's signals read 0 and its inputs go dead —
# fail-deadly by default; gating a wire on `alive` (or the Arc B decider's
# `require_alive` sugar) makes it fail-safe. Classes never declare it
# themselves; :func:`all_signals` prepends it.
ALIVE_SIGNAL = Signal(
    "alive", "free on every entity: 1 while functional, 0 once destroyed")


# Instance-level schema facts (design §3a/§3c) — carried by EVERY [[entity]]
# instance, declared once here rather than per class. The A3 loader enforces
# id uniqueness (hard error on duplicates) and file-order assignment.
INSTANCE_FIELDS = (
    Field("id", KIND_STR, default=None,
          doc="mandatory unique instance id (design §3a); all references "
              "address ids, never array positions"),
    Field("tags", KIND_STR_LIST, default=(),
          doc="wire targets may address tag:name; resolved at runtime in "
              "member-id order (design §3c)"),
)


class Entity(abc.ABC):
    """Abstract base for every registry class (design §3b).

    Pure schema in Arc A patch 1: class-level declarations only, NO runtime
    behavior — the sim loop never sees this package yet. L0 behavior methods
    arrive with Arc B (SignalBus-only I/O, design §4).
    """

    # Declared value fields (tuning numbers, enums, render params).
    FIELDS: tuple = ()
    # Emitted signals beyond the free `alive` (see ALIVE_SIGNAL).
    SIGNALS: tuple = ()
    # Accepted inputs, each marked while-held vs edge (design §4).
    INPUTS: tuple = ()
    # Complementary-input conflict resolution (design §4): groups of input
    # names, HIGHEST priority first within a group — a door declares
    # (("close", "open"),) so close beats open (the safe state).
    INPUT_PRIORITY: tuple = ()
    # Format-reserved, INERT in v1 (design §3d): declared so the future
    # control-scheme arc adds its policy layer without touching entities or
    # levels. Names only; no semantics execute anywhere.
    INTERACTIONS: tuple = ()
    # Physical by default (design §5); intangible classes never occupy the
    # grid. Per-instance override is the A3 loader's business.
    INTANGIBLE: bool = False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
REGISTRY: dict[str, type[Entity]] = {}

_NUMBER_TYPES = (int, float)


def _is_number(v) -> bool:
    return isinstance(v, _NUMBER_TYPES) and not isinstance(v, bool)


def field_value_error(f: Field, value) -> str | None:
    """Why ``value`` is invalid for ``f``'s declared kind and bounds — or
    None when it is valid.

    ONE rule for both ends of the pipeline: class defaults at register time
    (:func:`_check_default`) and authored ``[[entity]]`` values at load (the
    A3 loader in level_loader.py) are judged by exactly this function, so
    the registry stays THE validator (design §3b).
    """
    err = None
    if f.kind in (KIND_INT, KIND_Q16):
        if not (isinstance(value, int) and not isinstance(value, bool)):
            err = ("must be a plain int (Q16.16 / integer domain — floats "
                   "never enter synced state)")
    elif f.kind in (KIND_LENGTH_M, KIND_FLOAT_RENDER):
        if not _is_number(value):
            err = "must be a number"
    elif f.kind == KIND_BOOL:
        if not isinstance(value, bool):
            err = "must be a bool"
    elif f.kind == KIND_STR:
        if not isinstance(value, str):
            err = "must be a string"
    elif f.kind == KIND_ENTITY_REF:
        if not isinstance(value, str):
            err = "must be a string naming an [[entity]] instance id"
    elif f.kind == KIND_ENUM:
        if not f.choices:
            err = "enum field needs a non-empty `choices` tuple"
        elif value not in f.choices:
            err = f"must be one of {f.choices!r}"
    elif f.kind == KIND_COLOR_RGB:
        ok = (isinstance(value, (tuple, list)) and len(value) == 3
              and all(isinstance(c, int) and not isinstance(c, bool)
                      and 0 <= c <= 255 for c in value))
        if not ok:
            err = "must be an (r, g, b) triple of 0-255 ints"
    elif f.kind == KIND_STR_LIST:
        ok = (isinstance(value, (tuple, list))
              and all(isinstance(s, str) for s in value))
        if not ok:
            err = "must be a list of strings"
    else:
        err = f"unknown kind {f.kind!r} (valid: {ALL_KINDS})"
    if err:
        return err
    if f.minimum is not None and _is_number(value) and value < f.minimum:
        return f"below minimum {f.minimum!r}"
    if f.maximum is not None and _is_number(value) and value > f.maximum:
        return f"above maximum {f.maximum!r}"
    return None


def _check_default(cls_name: str, f: Field) -> None:
    """Type-check one field's default against its declared kind."""
    if f.default is None:
        return  # required field — no default to check
    err = field_value_error(f, f.default)
    if err:
        raise EntitySchemaError(
            f"entity class '{cls_name}' field '{f.name}': default "
            f"{f.default!r} {err}")


def _validate_class(cls: type) -> None:
    name = cls.__name__
    if not (isinstance(cls, type) and issubclass(cls, Entity)):
        raise EntitySchemaError(
            f"@register target '{name}' must subclass Entity")

    reserved = {f.name for f in INSTANCE_FIELDS}
    seen: set[str] = set()
    for f in cls.FIELDS:
        if not isinstance(f, Field):
            raise EntitySchemaError(
                f"entity class '{name}': FIELDS entries must be Field, "
                f"got {f!r}")
        if f.kind not in ALL_KINDS:
            raise EntitySchemaError(
                f"entity class '{name}' field '{f.name}': unknown kind "
                f"{f.kind!r} (valid: {ALL_KINDS})")
        if f.name in reserved:
            raise EntitySchemaError(
                f"entity class '{name}': field '{f.name}' shadows the "
                f"instance-level schema fact (INSTANCE_FIELDS)")
        if f.name in seen:
            raise EntitySchemaError(
                f"entity class '{name}': duplicate field '{f.name}'")
        seen.add(f.name)
        _check_default(name, f)

    sig_seen: set[str] = set()
    for s in cls.SIGNALS:
        if not isinstance(s, Signal):
            raise EntitySchemaError(
                f"entity class '{name}': SIGNALS entries must be Signal, "
                f"got {s!r}")
        if s.name == ALIVE_SIGNAL.name:
            raise EntitySchemaError(
                f"entity class '{name}': 'alive' is the free machinery "
                f"signal (design §4) — never declared per class")
        if s.name in sig_seen:
            raise EntitySchemaError(
                f"entity class '{name}': duplicate signal '{s.name}'")
        sig_seen.add(s.name)

    input_names: set[str] = set()
    for i in cls.INPUTS:
        if not isinstance(i, InputDecl):
            raise EntitySchemaError(
                f"entity class '{name}': INPUTS entries must be InputDecl, "
                f"got {i!r}")
        if i.mode not in (INPUT_HELD, INPUT_EDGE):
            raise EntitySchemaError(
                f"entity class '{name}' input '{i.name}': mode must be "
                f"'{INPUT_HELD}' or '{INPUT_EDGE}', got {i.mode!r}")
        if i.name in input_names:
            raise EntitySchemaError(
                f"entity class '{name}': duplicate input '{i.name}'")
        input_names.add(i.name)

    grouped: set[str] = set()
    for group in cls.INPUT_PRIORITY:
        if not (isinstance(group, (tuple, list)) and len(group) >= 2):
            raise EntitySchemaError(
                f"entity class '{name}': INPUT_PRIORITY groups need >= 2 "
                f"input names, got {group!r}")
        for iname in group:
            if iname not in input_names:
                raise EntitySchemaError(
                    f"entity class '{name}': INPUT_PRIORITY names unknown "
                    f"input '{iname}'")
            if iname in grouped:
                raise EntitySchemaError(
                    f"entity class '{name}': input '{iname}' appears in two "
                    f"INPUT_PRIORITY groups")
            grouped.add(iname)

    for it in cls.INTERACTIONS:
        if not isinstance(it, str):
            raise EntitySchemaError(
                f"entity class '{name}': INTERACTIONS entries are reserved "
                f"NAMES (strings, design §3d), got {it!r}")

    if not isinstance(cls.INTANGIBLE, bool):
        raise EntitySchemaError(
            f"entity class '{name}': INTANGIBLE must be a bool")


def register(cls=None, *, registry: dict = None):
    """Class decorator: validate the schema and add the class to the registry.

    Keyed by class name (design §3b); duplicate names hard-error. The
    ``registry`` kwarg lets tests register into a private dict instead of
    the module-level :data:`REGISTRY`.
    """
    target = REGISTRY if registry is None else registry

    def _do(c):
        _validate_class(c)
        if c.__name__ in target:
            raise EntitySchemaError(
                f"duplicate entity class name '{c.__name__}' — the registry "
                f"is keyed by class name; rename one of them")
        target[c.__name__] = c
        return c

    return _do if cls is None else _do(cls)


def all_signals(cls: type[Entity]) -> tuple:
    """The class's full emitted-signal tuple: free `alive` first (design §4)."""
    return (ALIVE_SIGNAL,) + tuple(cls.SIGNALS)
