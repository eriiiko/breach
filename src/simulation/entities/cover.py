"""The ``cover`` entity class — physical, destructible cover (design §7).

``docs/onephase_wego_design_2026-07-28.md`` §7 is blunt about what this
replaces: "There is no statistical to-hit model and no XCOM-style modifier
stack… **Cover is physical.** Bullets ray-march in continuous space; a cover
object physically eats the rays that clip it. A marine hugging a crate is
protected exactly as much as geometry says — no cover bonus stat anywhere."

So a cover object is not a tile property and not a number on an attack roll.
It is an **entity with a rectangle**, living in continuous Q16.16-friendly tile
coordinates like units and bullets do (§4), with HP so it can be shot apart.
Two consequences the rest of the design leans on:

- "easiest to hit" becomes computable — the target with the largest exposed
  profile (§9's overwatch priority), which
  :func:`simulation.vision.exposed_profile` measures directly against these
  rectangles;
- directional cover is geometric by construction: a flanking approach simply
  does not have the crate between it and the target, with nothing to encode.

**v1 fence (§7, agreed 2026-07-28):** cover is **static-but-destructible**. The
full dynamics — cover and items pushed by shockwaves (``wave_p`` impulses),
carried or floated by water — is a separate continuous-space-dynamics arc with
its own design session. This module fixes only the interface v1 needs, so that
arc slots in behind it without touching the order system.

``blocks_los`` is the one authored choice that matters at the table: a crate
you can see over but not shoot through (the default, ``False``) versus a
full-height barricade that also breaks line of sight. Vision consults it;
the bullet march never does — every cover shape stops rounds.
"""
from __future__ import annotations

from fractions import Fraction

from simulation.entities.schema import (
    Entity, Field, KIND_BOOL, KIND_INT, KIND_LENGTH_M, register,
)


@register
class cover(Entity):
    """A destructible rectangle that eats bullets (design §7)."""

    # Cover does NOT occupy the material grid: it is a continuous-space shape,
    # not a wall. Marking it intangible keeps the loader from stamping it into
    # `material` — a crate must not become a tile that blocks pathfinding and
    # airflow, or it would stop being cover and start being architecture.
    INTANGIBLE = True

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="anchor tile COL at base resolution (the rectangle's "
                  "left edge) — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="anchor tile ROW at base resolution (its top edge) — "
                  "REQUIRED"),
        Field("width_m", KIND_LENGTH_M, default=1.0, minimum=0.0,
              doc="rectangle width in meters, quantized once at load (the "
                  "door-span rule, a6 §3); must be > 0"),
        Field("height_m", KIND_LENGTH_M, default=1.0, minimum=0.0,
              doc="rectangle height in meters, quantized once at load; "
                  "must be > 0"),
        Field("hp", KIND_INT, default=60, minimum=1,
              doc="structural hit points — cover is DESTRUCTIBLE (§7); "
                  "rounds that clip it chew it down and it stops being "
                  "cover when it breaks"),
        Field("blocks_los", KIND_BOOL, default=False,
              doc="does this shape break line of sight as well as stopping "
                  "rounds? False (default) = a crate you can see over but "
                  "not shoot through — the §7/§8 split, where vision is "
                  "limited by walls and cover is a bullet-stopper. True = a "
                  "full-height barricade."),
    )

    # Format-reserved, inert in v1 — the design notes cover may emit "a signal
    # on destruction later" (§7). Declaring nothing keeps the digest clean; the
    # free `alive` signal already carries destruction for any wire that wants it.
    SIGNALS = ()
    INPUTS = ()
    INTERACTIONS = ()

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """Synced runtime state: alive + remaining HP.

        Read plainly off the runtime object — a bare ``EntityInstance`` raises
        ``AttributeError``, which is correct and loud (digests are only ever
        captured from constructed sims).
        """
        return (("alive", 1 if entity.alive else 0),
                ("hp_now", int(entity.hp_now)))


def quantize_extent_tiles(length_m, tile_size_m, *, context: str = "cover") -> int:
    """Meters -> whole tiles, by the canonical door-span rule (a6 §3).

    ``floor(length_m * tiles_per_m + 1/2)`` in exact ``Fraction`` arithmetic —
    never ``round()`` on a float (Python 3's ``round`` is banker's rounding,
    the exact trap that rule exists to avoid) — clamped to at least 1. Reusing
    the door quantizer's shape rather than inventing a second one keeps every
    meters-first extent in the project on ONE rule.
    """
    from simulation.entities.door import tiles_per_m
    lm = Fraction(str(float(length_m)))
    if lm <= 0:
        raise ValueError(
            f"{context}: length must be > 0, got {length_m!r}")
    n = (lm * tiles_per_m(tile_size_m) + Fraction(1, 2)).__floor__()
    return max(1, int(n))
