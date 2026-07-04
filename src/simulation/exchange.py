"""The physics↔unit exchange layer — the coupling table (mechanics/05).

Erik's founding principle (mechanics/05, verbatim): "there simply shouldn't
be any barrier between gameplay and the physics" — operationally, **the unit
must be able to read every field**. This module is that principle's ONE home:
every physics→unit coupling — shockwave damage, heat damage, water slowing,
gas poisoning, O2, pushes — is a **row in a table**::

    (field, reduction over footprint, response(sample, unit.profile) -> outputs)

not a plumbing project. Adding a coupling is O(one row).

Contents:

- **The reduction vocabulary** (mechanics/05 §1): ``center | max | mean |
  sum | grad`` — small pure functions over a unit's footprint tiles on an
  int32 Q16.16 field. All integer-exact (ingress door 1): Python ints carry
  the sums (no overflow), and the single ``mean`` divide is the
  round-half-away-from-zero twin of ``fixed_point.h::mean_round``
  (sign-symmetric, no DC bias).
- **The coupling-table structure**: :class:`CouplingRow` + the ordered
  ``COUPLING_TABLE`` registering the shipped rows. Plain data + functions —
  no framework.
- **The shipped response implementations** (moved verbatim from
  ``combat.py``): ``apply_environmental_damage`` (the ``heat | max`` row) and
  ``apply_blast_damage`` (the ``wave_p`` blast row).

P1 scope note (behaviour-preserving refactor, 2026-07-05): the table is the
formal registry; **execution still happens at the rows' legacy tick
positions** (heat damage post-physics in ``Simulation.step`` 9c; blast damage
at detonation sites — grenade fuse-out and door explosives). The consolidated
named EXCHANGE-READ slot that iterates this table in table order (mechanics/05
§4, pipeline phase 2) is a later patch; nothing here reorders or merges the
shipped call sites.

Conventions (shared by every reduction):

- ``field`` is a 2-D numpy int array indexed ``field[ty, tx]`` (row-major,
  y-down) — the GameMap layout.
- ``tiles`` is a sequence of ``(tx, ty)`` tile coordinates — exactly what
  :meth:`Unit.occupied_tiles` returns.
- Off-grid tiles are skipped by an in-bounds guard (mirroring
  ``apply_environmental_damage``'s footprint loop); a footprint with **no**
  in-bounds tile reduces to the zero element (0, or ``(0, 0)`` for ``grad``).
- Every result is a plain Python int (exact, unbounded) in the field's own
  Q16.16 domain — determinism ingress door 1 (engine/14 §3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


# ---------------------------------------------------------------------------
# The reduction vocabulary (mechanics/05 §1) — v1: center | max | mean | sum
# | grad. Small pure functions, integer-exact, over footprint tiles.
# ---------------------------------------------------------------------------

def _mean_round(total: int, count: int) -> int:
    """``total / count`` rounded half away from zero — the Python twin of
    ``cpp/src/fixed_point.h::mean_round`` (sign-symmetric, so a mean never
    picks up a ``-sign(total)`` DC bias the way a plain truncating divide
    would). ``count <= 0`` returns 0 (the caller's empty-footprint fallback),
    matching the C++ ``count <= 0 -> 0`` guard.
    """
    if count <= 0:
        return 0
    half = count // 2
    if total >= 0:
        return (total + half) // count
    # C++ does (total - half) / count with TRUNC-toward-zero division; Python
    # // floors, so emulate trunc on the negative branch: trunc(a/b) == -((-a)//b).
    return -((-total + half) // count)


def reduce_center(field, tiles: Sequence[tuple[int, int]]) -> int:
    """Sample the field at the footprint's centre tile.

    The centre is the bounding-box middle of ALL tiles (geometry — including
    any off-grid ones), per axis ``(lo + hi + 1) // 2``: for a square
    ``footprint × footprint`` body anchored at ``a`` this is exactly
    ``a + footprint // 2``, i.e. it agrees with ``Unit.center_tile_x/y()``
    for every footprint size (odd or even). Order-independent in ``tiles``.
    Returns 0 if ``tiles`` is empty or the centre tile itself is off-grid.
    """
    if not tiles:
        return 0
    x_lo = min(tx for (tx, _ty) in tiles)
    x_hi = max(tx for (tx, _ty) in tiles)
    y_lo = min(ty for (_tx, ty) in tiles)
    y_hi = max(ty for (_tx, ty) in tiles)
    cx = (x_lo + x_hi + 1) // 2
    cy = (y_lo + y_hi + 1) // 2
    h, w = field.shape
    if 0 <= cy < h and 0 <= cx < w:
        return int(field[cy, cx])
    return 0


def reduce_max(field, tiles: Sequence[tuple[int, int]]) -> int:
    """True maximum over the in-bounds footprint tiles (may be negative on a
    signed field). Returns 0 when no tile is in bounds — the same "off-grid
    footprint reads cold" fallback the shipped heat row uses.
    """
    h, w = field.shape
    best: Optional[int] = None
    for (tx, ty) in tiles:
        if 0 <= ty < h and 0 <= tx < w:
            v = int(field[ty, tx])
            if best is None or v > best:
                best = v
    return 0 if best is None else best


def reduce_mean(field, tiles: Sequence[tuple[int, int]]) -> int:
    """Integer mean over the in-bounds footprint tiles: one exact integer sum
    + ONE round-half-away-from-zero divide (mechanics/05 §1; the
    ``mean_round`` convention). Off-grid tiles are excluded from both the sum
    AND the count. Returns 0 when no tile is in bounds.
    """
    h, w = field.shape
    total = 0
    count = 0
    for (tx, ty) in tiles:
        if 0 <= ty < h and 0 <= tx < w:
            total += int(field[ty, tx])   # Python int: exact, order-free
            count += 1
    return _mean_round(total, count)


def reduce_sum(field, tiles: Sequence[tuple[int, int]]) -> int:
    """Exact integer sum over the in-bounds footprint tiles (Python int —
    no int32 overflow; integer addition commutes, so the result is
    order-free). Returns 0 when no tile is in bounds.
    """
    h, w = field.shape
    total = 0
    for (tx, ty) in tiles:
        if 0 <= ty < h and 0 <= tx < w:
            total += int(field[ty, tx])
    return total


def reduce_grad(field, tiles: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Footprint gradient ``(gx, gy)`` — the v1 "footprint differences" form
    (mechanics/05 §1): per axis, the difference of the two extreme edge-line
    integer means over the in-bounds tiles::

        gx = mean(field on tiles with tx == x_hi) - mean(... tx == x_lo)
        gy = mean(field on tiles with ty == y_hi) - mean(... ty == y_lo)

    (each mean = one :func:`_mean_round` divide). Positive toward increasing
    tx / ty (y-down), i.e. it points UPHILL like ∇p — the future impulse-push
    row (mechanics/05 §1) consumes ``-grad``. The result is the raw field
    difference ACROSS the footprint extremes (span ``x_hi - x_lo`` tiles),
    deliberately NOT normalised per tile — v1 keeps the divides to one per
    axis and lets the consuming response own its scale constant.

    An axis with fewer than two distinct in-bounds lines (single tile, fully
    clipped, or empty footprint) contributes 0.
    """
    h, w = field.shape
    in_bounds: list[tuple[int, int, int]] = []
    for (tx, ty) in tiles:
        if 0 <= ty < h and 0 <= tx < w:
            in_bounds.append((tx, ty, int(field[ty, tx])))
    if not in_bounds:
        return (0, 0)

    def _edge_mean(axis_value: int, axis_index: int) -> int:
        total = 0
        count = 0
        for entry in in_bounds:
            if entry[axis_index] == axis_value:
                total += entry[2]
                count += 1
        return _mean_round(total, count)

    x_lo = min(e[0] for e in in_bounds)
    x_hi = max(e[0] for e in in_bounds)
    y_lo = min(e[1] for e in in_bounds)
    y_hi = max(e[1] for e in in_bounds)

    gx = _edge_mean(x_hi, 0) - _edge_mean(x_lo, 0) if x_hi > x_lo else 0
    gy = _edge_mean(y_hi, 1) - _edge_mean(y_lo, 1) if y_hi > y_lo else 0
    return (gx, gy)


#: The v1 vocabulary by design name (mechanics/05 §1). A CouplingRow's
#: ``reduction`` column names an entry here (or None — see the row notes).
REDUCTIONS: dict[str, Callable] = {
    "center": reduce_center,
    "max":    reduce_max,
    "mean":   reduce_mean,
    "sum":    reduce_sum,
    "grad":   reduce_grad,
}


# ---------------------------------------------------------------------------
# The coupling-table structure (mechanics/05 §1) — plain data, no framework.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CouplingRow:
    """One physics→unit coupling: a row in the mechanics/05 table.

    Attributes
    ----------
    field : str
        GameMap field name the row reads (``heat``, ``wave_p``, ...).
    reduction : str or None
        Name into :data:`REDUCTIONS` — the footprint reduction the row's
        physical read uses. ``None`` marks a shipped response that predates
        the field read and does its own sampling (see the row's ``note``).
    response : callable
        The response implementation. P1: the shipped functions, invoked at
        their legacy tick positions with their legacy signatures; the
        uniform ``response(sample, unit.profile)`` shape arrives with the
        named EXCHANGE-READ slot (a later patch).
    note : str
        Honest wiring status — what the row does TODAY vs the chapter row.
    """
    field: str
    reduction: Optional[str]
    response: Callable
    note: str = ""
