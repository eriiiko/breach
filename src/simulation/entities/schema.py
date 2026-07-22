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
import re
from dataclasses import dataclass


class EntitySchemaError(ValueError):
    """A class declaration violates the schema rules (raised at register)."""


# Digest-token charset (A4): class / field / signal names become ASCII tokens
# between '|' delimiters in the hashed ENTITY_SECT_V1 / SIGNAL_SECT_V1 byte
# streams (simulation.entities.serialize). Guarding them to [A-Za-z0-9_]+ at
# REGISTRATION keeps that serialization injective with no escaping (A4 impl
# note, critique 10).
DIGEST_NAME_RE = re.compile(r"[A-Za-z0-9_]+\Z")


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
# Breach-site roster (editor design §5, A8): [[unit_type, count], ...] pairs.
# unit_type is UNIT-system vocabulary — units are NOT entities (design §3e),
# so it is never registry-validated beyond being a non-empty string. Spawn
# realization is stack-2's; in Arc A a roster is authored data only. Like
# str_list it is NOT a synced kind (serialize.SYNCED_FIELD_KINDS): its synced
# consequence is the spawned units, hashed as unit state when they exist.
KIND_ROSTER = "roster"              # list of [unit_type(str), count(int>=1)]

ALL_KINDS = (KIND_INT, KIND_Q16, KIND_LENGTH_M, KIND_BOOL, KIND_STR,
             KIND_ENUM, KIND_FLOAT_RENDER, KIND_COLOR_RGB, KIND_STR_LIST,
             KIND_ENTITY_REF, KIND_ROSTER)

# Kinds the entities.toml tuning overlay may override — NUMBERS only.
NUMERIC_KINDS = (KIND_INT, KIND_Q16, KIND_LENGTH_M, KIND_FLOAT_RENDER)

# Input modes (design §4, extended by Arc B §2d). The first two are Arc A's
# door vocabulary — their meaning is FROZEN (trigger-2 safe); B2 only ADDS two:
# - INPUT_HELD  (=OR):     1 if ANY driving wire != 0 (while-held).
# - INPUT_EDGE  (=fire-once): fires once on a 0->!=0 rise vs its own synced
#                            prev row (reserved; no B2 node uses it).
# - INPUT_AND   (new):     1 iff EVERY driving wire != 0 (empty => 0). Many-wire.
# - INPUT_SINGLE (new):    the ONE driving wire's integer value verbatim;
#                          arity == 1 (the §1b value-input arity check keys on
#                          this being single-arity). Used by decider/filter/
#                          gate_not `in`.
INPUT_HELD = "held"
INPUT_EDGE = "edge"
INPUT_AND = "and"
INPUT_SINGLE = "single"

# The closed set of accepted input modes (the _validate_class mode guard).
ALL_INPUT_MODES = (INPUT_HELD, INPUT_EDGE, INPUT_AND, INPUT_SINGLE)


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
    """One accepted input, marked with its aggregation mode (design §4/§2d)."""
    name: str
    mode: str  # one of ALL_INPUT_MODES
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
    # Arc B §2a/§5: True for L0 LOGIC-NODE classes (decider / gate_* / filter)
    # whose emitted signals are the prev-read/next-write NODE outputs — the
    # SignalBus stages them at 9e(b) and SWAPS them into pub at 9e(e) (one tick
    # per hop, §2c). A door's `is_open` / a sensor's `value` are NOT node
    # outputs (refreshed every tick at 9e(a), never swapped) so those classes
    # leave this False. build_signal_bus reads it to (i) count node instances
    # toward the "logic exists" union and (ii) mark the swapped slots.
    LOGIC_NODE: bool = False
    # Arc B §4 (B3): True for SENSOR classes (the field sensors, `clock`,
    # `sensor_motion`) whose free-standing `value` signal is SAMPLED from the
    # world at 9e(a) and published to `pub` — NOT a node output (never swapped;
    # refreshed every tick, and a DEAD sensor writes 0, fail-deadly, D13).
    # build_signal_bus reads it to (i) count sensor instances toward the "logic
    # exists" union (sensors ∪ nodes ∪ wires ≠ ∅, D1) and (ii) give each
    # sensor's `value` a bus slot so 9e(a) can write it.
    SENSOR: bool = False

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """Per-class RUNTIME-STATE digest rows: ``(name, int)`` pairs.

        The ENTITY_SECT_V1 runtime-row block (A4 impl note): empty in
        Arc A. A6 door state and Arc B EMA accumulators / controller phase /
        edge-detector prevs land here as runtime rows defined per class —
        NOT as schema FIELDS (they are not authorable) — under the section
        version, with zero mechanism surgery (critique 6). Names must be
        ``[A-Za-z0-9_]+`` and must not collide with the free ``alive`` row.
        """
        return ()


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
    elif f.kind == KIND_ROSTER:
        def _roster_pair_ok(p) -> bool:
            return (isinstance(p, (tuple, list)) and len(p) == 2
                    and isinstance(p[0], str) and p[0] != ""
                    and isinstance(p[1], int) and not isinstance(p[1], bool)
                    and p[1] >= 1)
        ok = (isinstance(value, (tuple, list))
              and all(_roster_pair_ok(p) for p in value))
        if not ok:
            err = ("must be a roster: [[unit_type, count], ...] — unit_type "
                   "a non-empty string (UNIT-system vocabulary, never "
                   "registry-validated: units are not entities, design §3e), "
                   "count an int >= 1")
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
    if not DIGEST_NAME_RE.fullmatch(name):
        raise EntitySchemaError(
            f"entity class name {name!r} outside [A-Za-z0-9_]+ — class "
            f"names are digest-section tokens (ENTITY_SECT_V1, A4)")

    reserved = {f.name for f in INSTANCE_FIELDS}
    seen: set[str] = set()
    for f in cls.FIELDS:
        if not isinstance(f, Field):
            raise EntitySchemaError(
                f"entity class '{name}': FIELDS entries must be Field, "
                f"got {f!r}")
        if not (isinstance(f.name, str) and DIGEST_NAME_RE.fullmatch(f.name)):
            raise EntitySchemaError(
                f"entity class '{name}': field name {f.name!r} outside "
                f"[A-Za-z0-9_]+ — field names are digest-section tokens "
                f"(ENTITY_SECT_V1, A4)")
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
        if not (isinstance(s.name, str) and DIGEST_NAME_RE.fullmatch(s.name)):
            raise EntitySchemaError(
                f"entity class '{name}': signal name {s.name!r} outside "
                f"[A-Za-z0-9_]+ — signal names are digest-section tokens "
                f"(SIGNAL_SECT_V1, A4)")
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
        if i.mode not in ALL_INPUT_MODES:
            raise EntitySchemaError(
                f"entity class '{name}' input '{i.name}': mode must be one of "
                f"{ALL_INPUT_MODES}, got {i.mode!r}")
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

    if not isinstance(cls.LOGIC_NODE, bool):
        raise EntitySchemaError(
            f"entity class '{name}': LOGIC_NODE must be a bool")

    if not isinstance(cls.SENSOR, bool):
        raise EntitySchemaError(
            f"entity class '{name}': SENSOR must be a bool")


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
