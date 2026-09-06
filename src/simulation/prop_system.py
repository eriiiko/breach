"""Prop load-order stamp — props & vegetation arc #60, P3.

Design: docs/architecture/graphics/props_and_vegetation.md §4.1/§4.2. The
sim-side sibling of :mod:`simulation.door_system` (design §4.2: "the stamp
primitive is named: stamp_prop_tiles() beside stamp_door_tiles()"). Props
never tick (zero per-tick logic) — this module is LOAD-TIME ONLY:

- :func:`prop_instances` — the level's prop ``EntityInstance``\\ s in ordinal
  order (mirrors :func:`simulation.door_system.door_instances`).
- :func:`prop_footprint` — the stamp square: an n x n block of (row, col)
  tiles anchored at (x, y); v1 ``stamp_tiles`` is always 1 (design §4.1 F17
  — the trunk tile only; the ~3x3 crown is purely visual).
- :func:`stamp_prop_tiles` — the load-order stamp (design §4.2), called by
  ``GameMap.__init__`` in the SAME slot as ``stamp_door_tiles``, AFTER doors
  (props may not overlap a door span). Validation mirrors the door stamp:
  OOB, overlap (with another prop OR a door span), no vacuum/ambient ring,
  only a permitted (floor-like) base material underneath. Also validates the
  render-only look fields that the schema itself cannot bound (design §4.1
  F24/F18): ``height_m``'s ortho-camera meters cap (level-dependent — needs
  ``tile_size_m``) and, for ``kind == "model"``, the model path's existence
  + extension (F18).

Props carry no runtime object (unlike doors' ``DoorRuntime``): the stamped
``EntityInstance`` IS the runtime entity (props never tick, so there is
nothing to wrap — ``simulation.simulation`` leaves prop instances in
``self.entities`` untouched, exactly like every other non-door class).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from simulation.entities import prop as prop_schema
from simulation.materials import MAT_AIR, MATERIAL_NAMES

# Name -> id, restricted to whatever a prop's schema permits (append-only,
# prop_schema.PROP_MATERIAL_CHOICES); built from the canon MATERIAL_NAMES
# table so a new permitted row is one config append + one schema choices
# entry, never a hardcoded id here.
_NAME_TO_MAT_ID = {name: mat_id for mat_id, name in MATERIAL_NAMES.items()}

# CSV/authored materials a prop's footprint may sit on (design §4.2: "only
# permitted base materials underneath" — floor-like, the door-precedent
# shape). A prop always fully replaces the tile, so (unlike a door, which may
# re-stamp itself) the only legal base is plain open air.
_STAMP_OK_BASE_MATERIALS = (MAT_AIR,)

# The ortho-camera height budget (design §4.1 F24): "~20 tiles' worth of
# meters" at the level's own tile_size_m.
_HEIGHT_CAP_TILES = 20.0

# Model path validation (design §4.1/§7/F18): OBJ preferred (raylib 5.5
# cgltf rejects 2020-era GLBs — props_and_vegetation.md §2), but a glTF/GLB
# built cleanly for this engine is not itself forbidden.
_MODEL_EXTS = (".obj", ".gltf", ".glb")
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_ROOT = _REPO_ROOT / "assets" / "models" / "props"


def prop_instances(level_data) -> list:
    """The level's prop ``EntityInstance``\\ s in ordinal order (mirrors
    ``door_system.door_instances``)."""
    ents = getattr(level_data, "entities", None) or []
    props = [e for e in ents if e.class_name == "prop"]
    props.sort(key=lambda e: int(e.ordinal))
    return props


def prop_footprint(fields: dict) -> list:
    """The stamp footprint: an n x n square of ``(row, col)`` tiles anchored
    at ``(x, y)`` — v1 ``n == 1`` (design §4.1 F17)."""
    n = int(fields["stamp_tiles"])
    x, y = int(fields["x"]), int(fields["y"])
    return [(y + dy, x + dx) for dy in range(n) for dx in range(n)]


def _validate_prop_look(inst, tile_size_m: float, ctx: str) -> None:
    """Load-time checks the schema's per-field bounds cannot express
    (design §4.1 F24/F18): ``height_m``'s level-dependent meters cap, and
    ``model``'s existence + extension when ``kind == 'model'``."""
    height_m = float(inst.fields["height_m"])
    if not (height_m > 0.0):
        raise ValueError(f"{ctx}: height_m must be > 0, got {height_m!r}")
    cap = _HEIGHT_CAP_TILES * float(tile_size_m)
    if height_m > cap:
        raise ValueError(
            f"{ctx}: height_m {height_m!r} exceeds the ortho-camera budget "
            f"cap of {cap!r} m (~{_HEIGHT_CAP_TILES:.0f} tiles at this "
            f"level's tile_size_m={tile_size_m!r} — props & vegetation "
            f"design §4.1 F24)")

    kind = inst.fields["kind"]
    if kind == "model":
        model = inst.fields["model"]
        if not model:
            raise ValueError(
                f"{ctx}: kind='model' requires a non-empty 'model' path")
        ext = Path(model).suffix.lower()
        if ext not in _MODEL_EXTS:
            raise ValueError(
                f"{ctx}: model path {model!r} has unsupported extension "
                f"{ext!r} (permitted: {_MODEL_EXTS}; OBJ preferred — "
                f"raylib 5.5 cgltf rejects 2020-era GLBs, design §2)")
        full = _MODELS_ROOT / model
        if not full.is_file():
            raise ValueError(
                f"{ctx}: model path {model!r} does not exist under "
                f"{_MODELS_ROOT} (design §4.1 F18)")
    elif kind != "generated":
        raise ValueError(
            f"{ctx}: kind must be 'generated' or 'model', got {kind!r}")


def stamp_prop_tiles(material, is_vacuum, level_data, is_ambient=None) -> None:
    """Load-order stamp (design §4.2) — mutates ``material`` IN PLACE.

    Called by ``GameMap.__init__`` between the tilemap fill and
    ``_update_caches()``, AFTER ``stamp_door_tiles`` (props may not land on
    a door span). Per prop in ordinal order: validate (bounds / overlap with
    another prop or a door span / vacuum-ambient ring / permitted base
    material / look-field bounds), then stamp the resolved material id over
    the whole footprint.
    """
    h, w = material.shape
    if is_ambient is None:
        is_ambient = np.zeros_like(is_vacuum)

    # Props may not land on a door span (design §4.2) — doors stamp first,
    # so their spans are computed the same way stamp_door_tiles derives
    # them, independent of the CURRENT (already door-stamped) material grid.
    door_tiles: set = set()
    if any(e.class_name == "door"
           for e in (getattr(level_data, "entities", None) or [])):
        from simulation.door_system import door_spans
        for _inst, span in door_spans(level_data):
            door_tiles.update(span)

    tile_size_m = float(getattr(level_data, "tile_size_m", 1.0))
    seen: dict = {}                       # tile -> prop id (overlap check)
    lvl = getattr(level_data, "name", "?")
    for inst in prop_instances(level_data):
        ctx = f"prop entity '{inst.id}' (level '{lvl}')"

        mat_name = inst.fields["material"]
        if mat_name not in _NAME_TO_MAT_ID:
            raise ValueError(
                f"{ctx}: unknown material {mat_name!r} — permitted: "
                f"{prop_schema.PROP_MATERIAL_CHOICES}")
        mat_id = _NAME_TO_MAT_ID[mat_name]

        _validate_prop_look(inst, tile_size_m, ctx)

        footprint = prop_footprint(inst.fields)
        for (fy, fx) in footprint:
            if not (0 <= fy < h and 0 <= fx < w):
                raise ValueError(
                    f"{ctx}: stamp tile ({fy}, {fx}) out of bounds for the "
                    f"{h}x{w} grid")
            if (fy, fx) in door_tiles:
                raise ValueError(
                    f"{ctx}: stamp tile ({fy}, {fx}) overlaps a door span "
                    f"— a prop may not land on a door (design §4.2)")
            if (fy, fx) in seen:
                raise ValueError(
                    f"{ctx}: stamp tile ({fy}, {fx}) overlaps prop entity "
                    f"'{seen[(fy, fx)]}' — prop footprints must be disjoint "
                    f"(design §4.2)")
            if is_vacuum[fy, fx] or is_ambient[fy, fx]:
                ring = "vacuum" if is_vacuum[fy, fx] else "ambient"
                raise ValueError(
                    f"{ctx}: stamp tile ({fy}, {fx}) is {ring} — a prop on "
                    f"the boundary ring is an authoring error (design §4.2)")
            if int(material[fy, fx]) not in _STAMP_OK_BASE_MATERIALS:
                raise ValueError(
                    f"{ctx}: stamp tile ({fy}, {fx}) has CSV material "
                    f"{int(material[fy, fx])} — a prop may only be placed "
                    f"on open floor "
                    f"{tuple(int(m) for m in _STAMP_OK_BASE_MATERIALS)} "
                    f"(design §4.2)")
            seen[(fy, fx)] = inst.id
        for (fy, fx) in footprint:
            material[fy, fx] = mat_id
