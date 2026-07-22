"""tools/entity_editor_ui.py — registry-driven palette + inspector (Arc C1).

Editor doc §3 pillar 1 (LOCKED): "the registry is the editor — and the
registry is CODE": palettes, panes, inspectors generate from the imported
entity module, with a graceful fallback to the last-good
``entity_registry.json`` + a red banner when the import breaks (canon
engine/16 §1, entity design §3b). This module is the pure, headless-testable
half of that pillar for `tools/map_editor.py`:

  - :func:`load_registry` — the import-then-fallback contract itself (canon
    §1 / editor design §3b). The ONE place that decides "is the registry
    fresh, or did we fall back, and why".
  - :func:`palette_entries` / :func:`inspector_rows` / :func:`quantize_length_m`
    / :func:`format_field_value` — pure functions over an ALREADY-LOADED
    registry payload (the same JSON shape
    ``simulation.entities.registry.registry_payload()`` returns, whether it
    came from a fresh import or the last-good fallback file) — so tests
    drive them with a hand-built payload and never need a live import, and
    the raylib loop never needs to know where the payload came from.

Deliberately does NOT import ``simulation.entities`` at module scope: the
whole point of the fallback contract is that this module — and everything
downstream of it (the palette, the inspector) — stays usable even when that
import is broken. The KIND_* string constants below duplicate
``simulation.entities.schema``'s closed vocabulary (frozen ASCII tokens,
digest-section material, A1) rather than importing them, for exactly that
reason; the JSON fallback payload already carries ``kind`` as a plain string,
so there is nothing to import.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

# Make project modules importable regardless of cwd (the level_edit_common /
# level_lib bootstrap pattern).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# entity_registry.json (repo root, gitignored) — the editor's last-good
# fallback (canon §1); rewritten on every successful load_registry() call
# with no explicit export_path override.
REGISTRY_JSON_FALLBACK = ROOT / "entity_registry.json"

# Editor design §4: the base authoring resolution — integer tiles_per_m,
# never the float tile_size_m. Callers at a non-default resolution pass
# their own tiles_per_m (simulation.entities.door.tiles_per_m derives it from
# the level's tile_size_m); this is just the common-case default for a
# stand-alone class template with no level context yet.
BASE_TILES_PER_M = 3

# ---------------------------------------------------------------------------
# Field kind vocabulary — mirrors simulation.entities.schema (see module
# docstring for why this is a deliberate duplication, not an import).
# ---------------------------------------------------------------------------
KIND_INT = "int"
KIND_Q16 = "q16"
KIND_LENGTH_M = "length_m"
KIND_BOOL = "bool"
KIND_STR = "str"
KIND_ENUM = "enum"
KIND_FLOAT_RENDER = "float_render"
KIND_COLOR_RGB = "color_rgb"
KIND_STR_LIST = "str_list"
KIND_ENTITY_REF = "entity_ref"
KIND_ROSTER = "roster"

# Kinds the inspector can nudge/cycle in place (numeric-ish + enum + bool).
# str / str_list / entity_ref / roster stay DISPLAY-ONLY in C1 — the editor
# has no in-UI text input in v1 (map_editor.py's own standing rule: names
# come from auto-generated ids, never retyped) and tag/roster assignment is
# C4's job (multi-select + tags) / a placement tool's job (roster editing
# rides zone placement, C5).
EDITABLE_KINDS = frozenset({KIND_INT, KIND_Q16, KIND_LENGTH_M, KIND_BOOL,
                           KIND_ENUM, KIND_FLOAT_RENDER, KIND_COLOR_RGB})


# ---------------------------------------------------------------------------
# Registry load + fallback (canon §1 — escalation trigger 4 territory: the
# FALLBACK SEMANTICS below are frozen; do not change them without Erik/Fable).
# ---------------------------------------------------------------------------

@dataclass
class RegistryLoadResult:
    """One :func:`load_registry` outcome. ``ok=True`` means a fresh import
    succeeded (and entity_registry.json was just rewritten as the new
    last-good, canon §1); ``ok=False`` means the import failed and
    ``payload`` came from the last-good fallback file instead — the caller
    shows a red banner with ``error`` and the palette stays usable minus
    whatever class broke."""
    payload: dict
    ok: bool
    error: Optional[str] = None


def _import_and_export(export_path=None) -> dict:
    """The live path: ``import simulation.entities`` (import-light — no
    compiled breach_physics, no ``simulation.simulation``) then rewrite
    entity_registry.json as the new last-good snapshot (canon §1:
    "rewritten on every successful launch")."""
    import simulation.entities as ents
    ents.export_registry_json(export_path)
    return ents.registry_payload()


def load_registry(*, importer: Optional[Callable[[], dict]] = None,
                  fallback_path=None, export_path=None) -> RegistryLoadResult:
    """Import the registry; on failure, fall back to reading the last-good
    ``entity_registry.json`` (canon §1 / entity design §3b editor failure
    mode).

    ``importer`` lets a caller (test) replace the live import with a stand-in
    that raises — simulating a half-written entity class — WITHOUT actually
    breaking a real class; the fallback SEMANTICS this exercises are frozen
    (escalation trigger 4), only the injection seam is test-only.
    ``fallback_path``/``export_path`` default to the real repo-root
    ``entity_registry.json``; tests override both to a tmp_path so a test
    run never touches the developer's real fallback file.

    Raises ``RuntimeError`` only when BOTH the import fails AND no last-good
    fallback can be read — the one case the editor truly cannot start from
    (nothing to show, nothing to fall back to).
    """
    fallback_path = (Path(fallback_path) if fallback_path is not None
                     else REGISTRY_JSON_FALLBACK)
    do_import = importer if importer is not None else (
        lambda: _import_and_export(export_path))
    try:
        return RegistryLoadResult(payload=do_import(), ok=True)
    except Exception as e:  # noqa: BLE001 — ANY import-time failure falls back
        try:
            payload = json.loads(fallback_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as fallback_err:
            raise RuntimeError(
                f"entity registry import failed ({e!r}) and no last-good "
                f"fallback could be read from {fallback_path} "
                f"({fallback_err!r})") from e
        return RegistryLoadResult(payload=payload, ok=False, error=str(e))


# ---------------------------------------------------------------------------
# Palette — one entry per registered class (editor design §3 pillar 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaletteEntry:
    class_name: str
    initial: str          # class-initial placeholder glyph (§8: permanent
                          # icon-less fallback, never an error)
    chip_rgb: tuple        # deterministic generated colour chip


def _auto_class_rgb(class_name: str) -> tuple:
    """Deterministic, well-spread chip colour for a registry class — the
    same golden-angle hue-walk recipe as
    ``level_edit_common._auto_rgb`` (material palette), keyed by a stable
    hash of the NAME since entity classes have no small integer id."""
    h = int(hashlib.sha256(class_name.encode("ascii")).hexdigest(), 16)
    hue = (h * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def palette_entries(payload: dict) -> list:
    """One :class:`PaletteEntry` per class in the loaded registry payload's
    ``classes`` table, sorted by name.

    The schema has no "placeable" flag distinguishing some registered
    classes from others (``INTANGIBLE`` means "no tile", not "not an
    instance") — so EVERY registered class gets a palette entry here; C3+
    placement tools decide which classes get a dedicated gesture (DOOR span
    drag, sensor sample-arrow) vs the generic place-one. No icons in C1
    (C8): colour chip + class initial only, the PERMANENT fallback per
    editor doc §8, not a placeholder to be removed later.
    """
    classes = payload.get("classes", {})
    return [PaletteEntry(class_name=name, initial=name[0].upper(),
                         chip_rgb=_auto_class_rgb(name))
            for name in sorted(classes)]


# ---------------------------------------------------------------------------
# The exact length_m -> tiles snap (editor doc §4, critique-hardened)
# ---------------------------------------------------------------------------

def quantize_length_m(length_m, tiles_per_m: int = BASE_TILES_PER_M) -> int:
    """``tiles = floor(length_m * tiles_per_m + 1/2)`` in EXACT arithmetic
    (Fraction) — round-half-up, never Python's banker's ``round()``, never a
    float division whose 15th digit decides a tie (editor doc §4).

    ``length_m`` ingresses as ``Fraction(str(length_m))`` — the decimal the
    author typed, not the binary float's expansion (the same N10 pin
    :func:`simulation.entities.door.quantize_span_tiles` uses; this is the
    same rule made class-agnostic, for ANY ``KIND_LENGTH_M`` field the
    inspector renders, not just a door span). Quantize ONCE at base
    resolution — a ``--res`` factor replicates the already-quantized tile
    count afterward, never re-derives from meters at the scaled resolution.
    """
    tpm = int(tiles_per_m)
    if tpm <= 0:
        raise ValueError(f"tiles_per_m must be > 0, got {tiles_per_m!r}")
    lm = Fraction(str(float(length_m)))
    n = (lm * tpm + Fraction(1, 2)).__floor__()
    return max(0, int(n))


# ---------------------------------------------------------------------------
# Inspector — kind-aware field rendering
# ---------------------------------------------------------------------------

def format_field_value(kind: str, value, *,
                       tiles_per_m: int = BASE_TILES_PER_M) -> str:
    """Kind-aware display text for one field value (editor doc §3/§4)."""
    if kind == KIND_LENGTH_M:
        tiles = quantize_length_m(value, tiles_per_m)
        return f"{float(value):g} m -> {tiles} tile{'s' if tiles != 1 else ''}"
    if kind == KIND_COLOR_RGB:
        r, g, b = (int(c) for c in value)
        return f"({r}, {g}, {b})"
    if kind == KIND_STR_LIST:
        return ", ".join(value) if value else "(none)"
    if kind == KIND_ROSTER:
        return ("; ".join(f"{u} x{c}" for u, c in value) if value
                else "(empty)")
    if kind == KIND_BOOL:
        return "yes" if value else "no"
    if kind == KIND_FLOAT_RENDER:
        return f"{float(value):g}"
    if kind == KIND_ENTITY_REF:
        return value if value else "(unwired)"
    return str(value)


@dataclass(frozen=True)
class InspectorField:
    """One rendered field row: display text is ALREADY kind-aware-formatted
    (:func:`format_field_value`); ``editable`` + bounds/choices are what a
    future nudge/cycle widget needs (C1 wires this only for the fields that
    already have a live instance to edit — LIGHT; other classes show the
    template read-only until their placement tool lands)."""
    name: str
    kind: str
    value: object
    display: str
    editable: bool
    minimum: object = None
    maximum: object = None
    choices: Optional[tuple] = None


def inspector_rows(cls_payload: dict, values: dict, *,
                   tiles_per_m: int = BASE_TILES_PER_M) -> list:
    """One :class:`InspectorField` per declared FIELD of a class (registry
    payload's ``classes[name]`` shape — NOT the instance-level ``id``/
    ``tags`` facts, which the caller renders separately since they are not
    per-class schema). ``values`` supplies authored overrides; any field
    absent from it falls back to the payload's own (overlay-effective)
    default — so passing ``{}`` renders the class's default TEMPLATE.
    Field order follows the payload's ``fields`` list (schema declaration
    order — stable, not sorted)."""
    rows = []
    for f in cls_payload["fields"]:
        name = f["name"]
        kind = f["kind"]
        value = values[name] if name in values else f["default"]
        choices = tuple(f["choices"]) if f.get("choices") else None
        rows.append(InspectorField(
            name=name, kind=kind, value=value,
            display=format_field_value(kind, value, tiles_per_m=tiles_per_m),
            editable=kind in EDITABLE_KINDS,
            minimum=f.get("minimum"), maximum=f.get("maximum"),
            choices=choices))
    return rows


# ---------------------------------------------------------------------------
# Generic place-one (Arc C3) — the catch-all placement gesture for a
# registered class with no bespoke tool (DOOR and the field sensors keep
# their own; see tools/door_entity_port.py / tools/sensor_entity_port.py).
# ---------------------------------------------------------------------------

def required_field_names(cls_payload: dict) -> tuple:
    """Field names with no default (``default is None`` in the registry
    payload) — REQUIRED at authoring time, schema.py's own convention (a
    kind's value domain never legitimately includes ``None``, so
    ``default is None`` is unambiguous). These MUST land in an instance's
    ``authored_keys`` or the loader rejects it as missing."""
    return tuple(f["name"] for f in cls_payload["fields"]
                if f["default"] is None)


def default_instance_fields(cls_payload: dict, *, x=None, y=None) -> dict:
    """``{field name: value}`` for a freshly placed instance of this class:
    every field at its registry-effective default, with ``x``/``y``
    overridden to the placement tile when the class declares them (the C3
    generic place-one gesture). A class that declares some OTHER required
    field (e.g. a zone's ``zone_id`` — no placement tool fills that yet)
    still gets that field's ``None`` here; callers refuse those classes
    (see :func:`required_field_names`) rather than author an invalid
    instance from this template."""
    names = {f["name"] for f in cls_payload["fields"]}
    fields = {f["name"]: f["default"] for f in cls_payload["fields"]}
    if x is not None and "x" in names:
        fields["x"] = x
    if y is not None and "y" in names:
        fields["y"] = y
    return fields
