"""Shared meters -> whole-tiles quantizer (a6 doors design §3).

Both :func:`simulation.entities.door.quantize_span_tiles` and
:func:`simulation.entities.cover.quantize_extent_tiles` round a meters-first
authoring length to whole tiles by the SAME rule — round-half-up in exact
``Fraction`` arithmetic, clamped >= 1. The two call sites differed only in
their default ``context`` prefix and the noun in the error text ("length_m"
for a door span, "length" for a cover extent); this module holds the one
rule they both delegate to so a future third meters->tiles quantizer has
somewhere to go instead of a third hand-copy (`docs/canonical_systems_survey_2026-08-22.md` #11).

Import-light (design §3b, CI-tested): stdlib only.
"""
from __future__ import annotations

from fractions import Fraction


def quantize_meters_to_tiles(length_m, tile_size_m, *, context: str,
                              field_name: str, tiles_per_m_fn) -> int:
    """``n = floor(length_m * tiles_per_m + 1/2)`` — round-half-up in exact
    ``Fraction`` arithmetic (never banker's ``round``), clamped >= 1.

    ``length_m`` ingresses as ``Fraction(str(length_m))`` (N10 pin — the
    decimal the author typed, not the binary float's expansion) and must be
    strictly positive (explicit check: schema minimums are inclusive). The
    positivity check runs BEFORE ``tiles_per_m_fn`` is called — preserves
    the original call sites' error precedence when both ``length_m`` and
    ``tile_size_m`` are invalid at once. ``tiles_per_m_fn`` resolves
    ``tile_size_m`` to the level's integer tiles-per-meter (door's
    ``tiles_per_m``, the one rule every meters-first extent uses).
    ``field_name`` names the quantity in the error text ("length_m" /
    "length") so each caller's message stays byte-identical to before this
    helper existed.
    """
    lm = Fraction(str(float(length_m)))
    if lm <= 0:
        raise ValueError(
            f"{context}: {field_name} must be > 0, got {length_m!r}")
    n = (lm * tiles_per_m_fn(tile_size_m) + Fraction(1, 2)).__floor__()
    return max(1, int(n))
