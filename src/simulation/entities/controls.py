"""Format-reserved control classes — `button` / `terminal` (canon engine/16 §8).

These are PLACEABLE, INERT entities: they exist in the schema (so the format,
the editor palette, and levels can carry them) but nothing drives them and they
drive nothing. Their behavior waits on the control-scheme decision (the
interaction / cost-policy split — Erik's standing note: never bake AP/phase
assumptions into entity code), so v1 declares NO inputs, NO class signals, and
NO INTERACTIONS: a button emits only the free ``alive`` signal and consumes
nothing. They occupy a tile like any placed fixture (physical by default).

Kept deliberately minimal (an anchor tile + the instance id/tags every entity
carries): adding real fields/inputs/signals is the control-scheme arc's job,
not Arc B's. Registering them here means an editor or level can reference them
today without a schema change later flipping their digest shape.
"""
from __future__ import annotations

from simulation.entities.schema import Entity, Field, KIND_INT, register


@register
class button(Entity):
    """A wall panel a marine can press — format-reserved, inert in v1."""

    INTANGIBLE = False   # a placed fixture occupies its tile (design §5)

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="anchor tile COL at base resolution — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="anchor tile ROW at base resolution — REQUIRED"),
    )
    # No inputs, no class signals, no interactions in v1 — the control-scheme
    # arc adds the interaction/cost-policy layer without touching this file's
    # digest shape (only the free `alive` signal exists today).


@register
class terminal(Entity):
    """A console a marine can operate — format-reserved, inert in v1."""

    INTANGIBLE = False

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="anchor tile COL at base resolution — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="anchor tile ROW at base resolution — REQUIRED"),
    )
