"""The `prop` entity class — props & vegetation arc #60, P3.

Design: docs/architecture/graphics/props_and_vegetation.md §4.1/§4.2 (v2.1,
Erik's 2026-09-07 rulings locked). Mirrors the door.py precedent (design §4.1:
"x, y ... like door.py"): this module carries the SCHEMA only (fields), stays
import-light (stdlib only), and never touches the sim. The load-time stamp
(the sim-side consequence of a prop's footprint fields) lives in
:mod:`simulation.prop_system`, exactly the door/door_system split.

Field-kind choice is F10's digest-hygiene rule made concrete: **a prop's LOOK
is not digest material; its FOOTPRINT is.** ``x``/``y``/``material``/
``stamp_tiles`` use SYNCED kinds (int/enum — serialize.SYNCED_FIELD_KINDS) so
moving a prop or changing what tile it stamps moves the entity digest; every
"how does it look" field (``kind``/``generator``/``seed``/``palette``/
``style``/``decor``/``model``) is deliberately :data:`KIND_STR` rather than
:data:`KIND_ENUM` — enum IS a synced kind, so an enum here would (wrongly)
enter the digest the moment art fields existed at all. ``height_m`` is
:data:`KIND_FLOAT_RENDER`, the same "never synced" class as light.py's
render floats.

No SIGNALS/INPUTS in v1 (props never tick — design §4.2: "zero per-tick
logic"); no per-class runtime digest rows either — a prop's only digest
footprint is its declared fields, ``runtime_digest_rows`` stays the base
empty tuple.
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, KIND_ENUM, KIND_FLOAT_RENDER, KIND_INT, KIND_STR, register,
)

# v1 permitted prop base materials (design §4.1: "v1 choices: (foliage,)").
# APPEND-ONLY (design §4.1: "Registry choices tuples are append-only — enum
# digests hash the index").
PROP_MATERIAL_CHOICES = ("foliage",)


@register
class prop(Entity):
    """One placed 3D prop: a load-time material stamp (footprint) + a
    render-only look (generated mesh or a model file)."""

    INTANGIBLE = False   # a placed prop occupies its stamp tile(s) (design §5)

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="anchor tile COL (trunk tile) — REQUIRED [synced]"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="anchor tile ROW (trunk tile) — REQUIRED [synced]"),
        Field("material", KIND_ENUM, default="foliage",
              choices=PROP_MATERIAL_CHOICES,
              doc="the ONE [materials.*] row this prop stamps — replaces a "
                  "separate blocking+fuel pair (design §4.1 F7) [synced]"),
        Field("stamp_tiles", KIND_INT, default=1, minimum=1,
              doc="square stamp side in tiles; v1 default 1 (trunk tile "
                  "only) — the ~3x3 crown is purely visual (design §4.1 "
                  "F17) [synced]"),
        Field("kind", KIND_STR, default="generated",
              doc="'generated' | 'model' — validated at load "
                  "(prop_system._validate_prop_look), not by the schema "
                  "(F10: never a synced enum) [not synced]"),
        Field("generator", KIND_STR, default="tree",
              doc="'tree' | 'palm' — renderer/propgen.py's GENERATORS key; "
                  "an unknown name is a render-time WARN + skip, never a "
                  "load error (static_props.get_model) [not synced]"),
        Field("seed", KIND_STR, default="0",
              doc="art-only; parsed as an int by the renderer "
                  "[not synced]"),
        Field("palette", KIND_STR, default="green",
              doc="renderer/propgen.py PALETTES key [not synced]"),
        Field("style", KIND_STR, default="smooth",
              doc="'smooth' | 'faceted' [not synced]"),
        Field("decor", KIND_STR, default="",
              doc="'' | 'flowers' | 'fruit' [not synced]"),
        Field("height_m", KIND_FLOAT_RENDER, default=2.2, minimum=0.0,
              doc="render-only meters (F16: meters-first, never tiles); "
                  "must be > 0 and loader-capped to ~20 tiles' worth of "
                  "meters at this level's tile_size_m (F24, ortho-camera "
                  "budget) — both enforced at load "
                  "(prop_system._validate_prop_look) [not synced]"),
        Field("model", KIND_STR, default="",
              doc="relative path under assets/models/props/ when "
                  "kind=='model'; existence + extension validated at load "
                  "(F18) [not synced]"),
    )
    # No inputs, no class signals in v1 — props never tick (design §4.2).
