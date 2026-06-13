"""Movement cadence — the footprint -> speed reduction seam (mobility design §4.1).

The engine owns the per-material ``mobility`` *field* (a terrain coefficient,
universal). How a creature reduces its footprint's mobilities to an effective
step time is **unit policy** — a single named function behind a fixed contract,
so the catalogue can grow (heavy/bulldoze, chokepoint-crawler, …) without
touching the engine. This lives in the game/unit layer for exactly that reason.

Contract (§4.1)::

    speed_fn(footprint_samples, speed_class) -> tick_cost: int   # pure, integer, deterministic

* ``footprint_samples`` — a :class:`FootprintSamples` struct carrying the
  per-tile field values under the unit's footprint. v1 carries ``mobility``
  (milli-units) only; it is a *struct*, not a bare array, so future
  field-composition (water depth, pressure, …) is a non-breaking addition
  (§4.1 forward-compat). The function never receives the unit object — a float
  field on the unit (e.g. ``base_speed``) must not leak in and break lockstep.
* ``speed_class`` — a baked integer carrying the unit's existing order/species
  **base cadence** in ticks-per-tile (e.g. ``marine_attack_ticks_per_tile`` or
  ``zombie.ticks_per_tile``). The terrain reduction is composed *onto* this base
  as a multiplier; it does not replace it.

Determinism (§3): pure integer arithmetic, fixed-point milli-units, a single
documented rounding rule (half-up ``(num + den//2) // den`` — never ``round()``
on a float). Integer floor-division is bit-identical cross-machine.

v1 ships exactly one function — :func:`default_speed` (the §4 area-weighted
average). The seam is load-bearing; the catalogue grows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# Fixed-point scale for mobility milli-units (mobility design §3). air == 1000.
MOBILITY_ONE = 1000


@dataclass(frozen=True)
class FootprintSamples:
    """Per-tile field samples under a unit's footprint, for the speed reduction.

    Forward-compatible container (§4.1): v1 carries ``mobility`` only, so future
    dynamic fields (water depth, pressure, …) are added as extra members without
    a breaking signature change. ``mobility`` is the static per-material terrain
    floor in fixed-point milli-units; the speed function composes it with any
    dynamic factors.
    """

    mobility: Sequence[int]


def half_up(num: int, den: int) -> int:
    """Half-up integer division ``(num + den//2) // den`` (mobility design §3).

    Pure integer arithmetic — NEVER ``round()`` on a float (Python 3 ``round``
    is banker's rounding, the very rule to avoid here). Integer floor-division
    is bit-identical across architectures, so this is cross-machine
    deterministic with no float-ULP edge. ``den`` must be > 0 (a footprint with
    any ``mobility <= 0`` tile is not enterable, so the caller never reaches
    here with a zero denominator).
    """
    return (num + den // 2) // den


def default_speed(footprint_samples: FootprintSamples, speed_class: int) -> int:
    """v1 default footprint->speed reduction: the §4 area-weighted average.

    Composes a terrain MULTIPLIER onto the unit's existing base cadence
    ``speed_class`` (ticks-per-tile). The effective tick cost over a footprint
    of ``n`` tiles with mobilities ``m_i`` (milli-units) is::

        tick_cost = half_up(base_ticks * n * 1000, sum(m_i))

    which is ``base_ticks / avg_mobility_fraction`` — i.e. the base cadence
    divided by the area-averaged mobility. All-air (every ``m_i == 1000``)
    leaves the base untouched; all-furniture (``m_i == 400``) yields
    ``base * 1000/400 == base * 2.5``. A single obstacle is diluted by body
    area (the intended "size is not a movement liability" feature, §4).

    Pure / integer-in / integer-out / deterministic — takes a baked int
    ``speed_class``, never the unit object (§4.1 lockstep contract). Always
    returns at least 1 tick (a step can never be instantaneous; the §3
    ``mobility > 1`` speed-boost range quantises coarsely against this floor).
    """
    mob = footprint_samples.mobility
    n = len(mob)
    total = 0
    for m in mob:
        total += int(m)
    # Enterability (every m > 0) is the caller's gate; total > 0 holds whenever
    # the step was allowed. Guard defensively so a misuse fails loud, not silent.
    if total <= 0:
        raise ValueError("default_speed: footprint has non-positive total mobility "
                         "(should have been gated by is_passable_block)")
    cost = half_up(int(speed_class) * n * MOBILITY_ONE, total)
    return cost if cost >= 1 else 1
