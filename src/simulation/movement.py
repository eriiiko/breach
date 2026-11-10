"""Movement cadence — the footprint -> speed reduction seam (mobility design §4.1).

The engine owns the per-material ``mobility`` *field* (a terrain coefficient,
universal). How a creature reduces its footprint's mobilities to an effective
step time is **unit policy** — a single named function behind a fixed contract,
so the catalogue can grow (heavy/bulldoze, chokepoint-crawler, …) without
touching the engine. This lives in the game/unit layer for exactly that reason.

Contract (§4.1)::

    speed_fn(footprint_samples, speed_class) -> tick_cost: int   # pure, integer, deterministic

* ``footprint_samples`` — a :class:`FootprintSamples` struct carrying the
  per-tile field values under the unit's footprint (``mobility``, milli-units)
  AND, since #140, a coupling row's already-reduced Q16 speed SCALE
  (``wind_scale_q16``; the planned ``water_depth`` speed row is the next
  member). It is a *struct*, not a bare array, so composition is a
  non-breaking addition (§4.1 forward-compat). The function never receives
  the unit object — a float field on the unit (e.g. ``base_speed``) must not
  leak in and break lockstep. :func:`footprint_samples` is THE builder: every
  ``default_speed`` caller in ``src/`` takes its samples from it.
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

from simulation.atmosphere_fixed import FP_ONE
from simulation import turning, unit_fixed


# Fixed-point scale for mobility milli-units (mobility design §3). air == 1000.
MOBILITY_ONE = 1000

#: The Q16 one a coupling row's speed SCALE is written in (#140). It is the
#: wind's own one (``atmosphere_fixed.FP_ONE``), never :data:`MOBILITY_ONE`:
#: a scale of ``SCALE_ONE`` is "no slowdown" and composes bit-identically.
SCALE_ONE = FP_ONE


@dataclass(frozen=True)
class FootprintSamples:
    """Per-tile field samples under a unit's footprint, for the speed reduction.

    Forward-compatible container (§4.1). It carries two kinds of member:

    * per-tile FIELD values the speed function reduces itself — ``mobility``,
      the static per-material terrain floor in fixed-point milli-units;
    * a coupling row's ALREADY-REDUCED Q16 speed scale — ``wind_scale_q16``
      (#140, ``exchange.wind_scale_q16`` over the footprint's wind load;
      :data:`SCALE_ONE` = no slowdown, the default, so every pre-#140
      construction is bit-identical). The planned ``water_depth`` speed row
      is the next member of this kind.

    :func:`default_speed` composes both and stays pure. Built in ``src/`` only
    by :func:`footprint_samples` (tests may build their own).
    """

    mobility: Sequence[int]
    wind_scale_q16: int = SCALE_ONE
    #: G5p-a2: the unit impulse law's walking scale from the unit's effort
    #: (``unit_impulse.effort_scale_q16``; ``SCALE_ONE`` = the feet are not
    #: working against anything), composed AFTER the wind's.
    effort_scale_q16: int = SCALE_ONE


def footprint_samples(gmap, fy, fx, fp,
                      effort_scale_q16: int = SCALE_ONE) -> FootprintSamples:
    """THE :class:`FootprintSamples` builder (#140, design §2.2): the terrain
    mobility under the ``fp`` x ``fp`` footprint anchored at ``(fy, fx)``
    (``GameMap.footprint_mobility``) and the wind's cadence scale there.

    Every ``default_speed`` caller in ``src/`` takes its samples from here, so
    every walker -- MOVE_DIR, the zombie walk, the onephase cadence, the WEGO
    order path -- is slowed by the same law (``tests/test_footprint_samples_
    builder.py`` greps for a caller that builds its own). Lives in the unit
    layer, not on GameMap: the field store must not import the coupling layer.

    The wind scale is ``exchange.wind_scale_q16`` over
    ``gmap.footprint_wind_load`` -- the field at the moment of the call, so
    MOVE_DIR and the zombie walk see the wind at execution, the WEGO order
    path and the onephase cadence when their path is compiled (design §2.2,
    an accepted gap).

    ``effort_scale_q16`` (G5p-a2) is the walker's own walking scale from its
    effort (``unit_impulse.effort_scale_q16(table, unit)``), passed by every
    caller in ``src/`` (the unit is not this builder's to read: the samples
    stay a struct of numbers).
    """
    from simulation import exchange
    settings = exchange.gmap_wind_settings(gmap)
    w = gmap.footprint_wind_load(fy, fx, fp)
    return FootprintSamples(
        mobility=gmap.footprint_mobility(fy, fx, fp),
        wind_scale_q16=exchange.wind_scale_q16(w, settings),
        effort_scale_q16=int(effort_scale_q16))


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

    #140: the samples' ``wind_scale_q16`` (Q16, ``<= SCALE_ONE``) composes
    AFTER the mobility cost as ``cost' = half_up(cost * SCALE_ONE, scale)`` --
    a scale of one half doubles the ticks per tile; ``SCALE_ONE`` returns the
    mobility cost exactly (the calm path is bit-identical). G5p-a2: the
    ``effort_scale_q16`` member composes the same way AFTER the wind (in
    patch a the two multiply; G5p-b deletes the wind's member when wind
    moves onto the impulse law).

    The CURRENCY of ``speed_class`` is the caller's, and the result comes
    back in it (both reductions are linear): whole ticks per tile for the
    zombie walk and the WEGO / onephase cadences, Q16.16 ticks per tile for
    the ``[speeds]`` table's marine costs (G1p-c, ``speeds.SpeedTable``) --
    one law at two resolutions, never a second copy of it.
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
    for name in ("wind_scale_q16", "effort_scale_q16"):
        # #140's wind first, then G5p-a2's effort: each half_up, each skipped
        # at SCALE_ONE so the unscaled path is bit-identical.
        scale = int(getattr(footprint_samples, name, SCALE_ONE))
        if scale != SCALE_ONE:
            if scale <= 0:
                raise ValueError(f"default_speed: {name} must be > 0, "
                                 f"got {scale}")
            cost = half_up(cost * SCALE_ONE, scale)
    return cost if cost >= 1 else 1


#: The throttle's Q16 one (gameplay pack v1, G1p): ``set_move_dir``'s
#: ``throttle_q`` lives in ``[0, THROTTLE_ONE]``.
THROTTLE_ONE = FP_ONE


def facing_cos_q16(dx_q: int, dy_q: int, facing_q: int) -> int:
    """``cos theta`` in Q16, theta the angle between a Q16 move direction
    ``(dx_q, dy_q)`` (world axes, y down) and a facing ``facing_q`` (Q16.16
    radians, math-style: the ``Unit.facing`` convention) -- through the kit:
    the move's own ``atan2_q16`` (``-dy``: world y runs down, as
    ``Unit.face_towards``), the shorter angle difference
    (``turning.wrap_q16``) and ``cos_q16``. Exact at the cardinal angles
    (``ONE`` when the facing IS the move's own atan2, ``-ONE`` reversed,
    ``0`` at a right angle) and in ``[-ONE, ONE]``. Integer only, no draw."""
    move_q = unit_fixed.atan2_q16(-int(dy_q), int(dx_q))
    theta = turning.wrap_q16(move_q - int(facing_q))
    c = unit_fixed.cos_q16(theta)
    return -FP_ONE if c < -FP_ONE else (FP_ONE if c > FP_ONE else c)


def move_speed_tiles_per_tick(cost_walk_q: int, cost_full_q: int,
                              side_q: int, back_q: int, throttle_q: int,
                              cos_q: int) -> float:
    """THE commanded speed of a unit on the continuous path (G1p-c, Erik's
    direction rule of 2026-11-10): tiles per tick at throttle ``throttle_q``
    (Q16, ``[0, THROTTLE_ONE]``) moving at ``cos theta = cos_q`` (Q16) to its
    facing::

        v = f(theta) * v_walk + t * (v_full - v_walk) * max(0, cos theta)
        f = 1 forward, side at a right angle, back reversed, linear in cos theta

    -- the walk slows by Erik's factors as the move turns from the facing,
    and the sprint's extra speed exists only forward: nobody sprints
    backwards. ``cost_walk_q`` / ``cost_full_q`` are the walk and the full
    speed as ALREADY-REDUCED Q16 tick costs (``default_speed`` over the
    ``[speeds]`` table's costs, so terrain and wind scale both ends by the
    one law); ``v = ONE / cost``. Formed as ONE exact rational in integers
    (``F`` = ``f`` in Q32, ``c+ = max(0, cos_q)``)::

        v = (F * Cf + t * c+ * (Cw - Cf)) / (ONE * Cw * Cf)

    and divided ONCE (Python's int / int is correctly rounded) -- the one
    division the walk step always made. So forward at throttle 0 it IS
    ``ONE / Cw`` (the walk), at ``THROTTLE_ONE`` ``ONE / Cf`` (the full
    speed), straight backward ``back * walk`` and at a right angle ``side *
    walk`` at EVERY throttle, each exactly the rounded rational; monotone in
    the throttle and in ``cos theta`` (``0 < back <= side <= ONE``,
    ``Cf <= Cw``: the speed door's rules). Refuses a throttle outside
    ``[0, THROTTLE_ONE]`` (the door clamps; this is the law's own guard), a
    ``cos_q`` outside ``[-ONE, ONE]`` and a cost below 1."""
    t = int(throttle_q)
    if not 0 <= t <= THROTTLE_ONE:
        raise ValueError(f"throttle_q must be in [0, {THROTTLE_ONE}], got {t}")
    c = int(cos_q)
    if not -FP_ONE <= c <= FP_ONE:
        raise ValueError(f"cos_q must be in [-{FP_ONE}, {FP_ONE}], got {c}")
    cw, cf = int(cost_walk_q), int(cost_full_q)
    if cw < 1 or cf < 1:
        raise ValueError(f"tick costs must be >= 1, got walk {cw}, full {cf}")
    side, back = int(side_q), int(back_q)
    if c >= 0:
        f32 = side * FP_ONE + (FP_ONE - side) * c
        c_plus = c
    else:
        f32 = side * FP_ONE + (side - back) * c
        c_plus = 0
    return (f32 * cf + t * c_plus * (cw - cf)) / (FP_ONE * cw * cf)


def move_speed_q16(cost_walk_q: int, cost_full_q: int, side_q: int,
                   back_q: int, throttle_q: int, cos_q: int) -> int:
    """:func:`move_speed_tiles_per_tick` on the Q16.16 grid (G5p-a2,
    critique B M6): the SAME exact rational, rounded half up ONCE to Q16
    tiles per tick -- ``half_up(F * Cf + t * c+ * (Cw - Cf), Cw * Cf)``, no
    float -- so the walk step and every position it writes lie on the grid
    (``unit_impulse.move``). Refuses what the float form refuses."""
    t = int(throttle_q)
    if not 0 <= t <= THROTTLE_ONE:
        raise ValueError(f"throttle_q must be in [0, {THROTTLE_ONE}], got {t}")
    c = int(cos_q)
    if not -FP_ONE <= c <= FP_ONE:
        raise ValueError(f"cos_q must be in [-{FP_ONE}, {FP_ONE}], got {c}")
    cw, cf = int(cost_walk_q), int(cost_full_q)
    if cw < 1 or cf < 1:
        raise ValueError(f"tick costs must be >= 1, got walk {cw}, full {cf}")
    side, back = int(side_q), int(back_q)
    if c >= 0:
        f32 = side * FP_ONE + (FP_ONE - side) * c
        c_plus = c
    else:
        f32 = side * FP_ONE + (side - back) * c
        c_plus = 0
    return half_up(f32 * cf + t * c_plus * (cw - cf), cw * cf)
