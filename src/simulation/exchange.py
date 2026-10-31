"""The physics↔unit exchange layer — the coupling table (mechanics/05).

Erik's founding principle (mechanics/05, verbatim): "there simply shouldn't
be any barrier between gameplay and the physics" — operationally, **the unit
must be able to read every field**. This module is that principle's ONE home:
every physics→unit coupling — shockwave damage, heat damage, water slowing,
gas poisoning, O2, wind — is a **row in a table**::

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

P4 (2026-07-05) added the first NEW row — ``apply_wave_push``, the
``wave_p | grad`` impulse push + KNOCKED_DOWN trigger at ``Simulation.step``
9c2. **#140 (2026-10-31) RETIRED it** (history in git and mechanics/05 §1):
the 9c2 row is now :func:`apply_wind_knockdown`, the ``wind | max`` row of
the momentum-flux threshold law (docs/wind_on_units_design_2026-10-31.md) --
the dynamic pressure ``q = 1/2 rho v^2`` at the footprint, nothing below
``wind_q_slow_pa``, a cadence scale (:func:`wind_scale_q16`, composed by
``movement.default_speed``) up to ``wind_q_fall_pa``, KNOCKED_DOWN at or
above it. Nothing in the law writes a position. Still directly after the
heat row (the documented within-tick exchange order: heat, then wind).

P2 note (also behaviour-preserving, 2026-07-05): both responses now hand
their pre-mitigation amounts to the mechanics/06 DamagePacket pipeline
(:mod:`simulation.damage`), which owns mitigation -> Q2-lift quantize -> hp
-> hit/kill events. With the shipped tables every path is bit-identical to
the inline chains this replaced (the zombie heat ×4 moved into
``species.ZOMBIE_MITIGATION``); call positions, signatures, and the
bare-name import pattern (liveheat instrumentation) are unchanged.

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

import math

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from config import CFG
from simulation import atmosphere_fixed as _atm_fx
from simulation import gas_fixed
from simulation import unit_fixed
from simulation.damage import (
    BLAST, CREDIT_HEAT, HEAT, POISON, DamagePacket, apply_packet,
    mitigation_for, tick_credit,
)
from simulation.events import NO_ATTACKER
# Gas SLICE ids (engine/05 §6.2) — aliased so they can't be confused with the
# damage-type constants above (gases.POISON = slice 2; damage.POISON = dtype 4).
from simulation.gases import POISON as GAS_POISON, TEARGAS as GAS_TEARGAS
from simulation.species import ZOMBIE_STABILITY
from simulation.status import BLINDED, KNOCKED_DOWN, apply_status


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
    tx / ty (y-down), i.e. it points UPHILL like ∇p. No row uses it since
    #140 retired the impulse push (it stays in the vocabulary). The result is the raw field
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
# Environmental (radiant heat) damage to units — engine/06 §4, proposal §4.2
# — the `heat | max` coupling row. Moved VERBATIM from combat.py (P1).
# ---------------------------------------------------------------------------
# Q16.16 scale shared with the `heat`/`temperature` fields (cpp/src/fixed_point.h
# HEAT_SCALE, the heat-domain kit; raycaster.h until P6c). One unit of heat
# energy == HEAT_SCALE raw int counts in the buffer; Phi divides back out to the energy-unit domain the [combat] consts and
# the felt-temp model are authored in.
HEAT_SCALE = 65536


def apply_environmental_damage(units, gmap, ticks_per_second, events=None):
    """Apply per-tick radiant heat damage to every LIVING unit (proposal §4.2).

    A unit is a full ray-blocker (stamped before the ray pass), so rays
    terminate on its leading tiles and ``gmap.heat`` already holds the
    correctly occluded, distance-attenuated **incident radiant flux** at the
    unit's footprint. We therefore sample the buffer directly — no new
    occlusion — and never write back into it (the kernel never writes the unit;
    the unit only reads). ``Phi_rad``-only: the optional contact term is
    deferred (Erik #6).

    Per living unit, in stored order (deterministic serial apply, mirroring
    :func:`apply_blast_damage`):

        Phi     = max(heat over occupied_tiles) / HEAT_SCALE      # incident flux
        Phi_abs = Phi * unit_absorption * (1 - unit_reflectivity)
        T_felt  = heat_ambient_ref + heat_flux_to_temp * Phi_abs
        over    = T_felt - temperature_max                        # damage band
        if over <= 0: no heat damage this tick
        dmg     = environmental_damage_rate * (1 + heat_overtemp_scale*over) * dt_tick
        apply_packet(u, DamagePacket(dmg, HEAT), ...)             # mechanics/06 §2

    The packet pipeline mitigates with the unit's resist table at the exact
    pre-quantize position the old ``if u.is_zombie: dmg *= zombie.fire_
    damage_multiplier`` branch occupied — that special case is DISSOLVED into
    ``resist_mult[HEAT] = 4.0`` on ``species.ZOMBIE_MITIGATION`` (an exact
    binary scale at the same float-chain position → bit-identical), and a
    marine's neutral table is an IEEE-exact no-op. The tick path no longer
    reads ``CFG.zombie.fire_damage_multiplier``; since P5 the species table
    sources its value from that key ONCE at import (restart-bound standard
    value — see species.py / config.toml).

    ``dt_tick = 1 / ticks_per_second`` makes the real DPS tick-rate independent.
    ``temperature_max`` and ``environmental_damage_rate`` come from the unit's
    :class:`EnvironmentProfile` when present, falling back to the global
    ``[combat]`` config values otherwise.

    Heat deaths set ``source="heat"`` on the hit / killed events and do **NOT**
    set ``killed_by_zombie`` — like blast and bullet deaths, only melee
    converts (a burned corpse converting would be wrong).

    Must run AFTER the ray pass fills ``heat`` and BEFORE the end-of-tick heat
    clear (its existence is precisely what makes clearing ``heat`` correct).
    """
    # D3 (P-R4, ruling amendment 5) — WHERE THE FLUX NOW COMES FROM.
    # This sampler used to read ``gmap.heat``, because the retired PAINTER
    # deposited its per-ray energy into every marched cell including the AIR
    # tiles a unit stands on. P-R4 replaced that with an antisymmetric
    # net-T^4 EXCHANGE, and by Kirchhoff air (``heat_atten == 0``) neither
    # absorbs nor emits — so nothing radiative lands on air any more and
    # ``heat`` now carries only combustion's own deposits at burn sites. Read
    # unchanged, a fire could no longer burn a marine standing next to it
    # (measured: unit HP never dropped).
    # ``gmap.rad_flux`` is the replacement: a positive-only SENSOR plane the
    # ray march fills at air cells with ``tau * w * a_s * E[T_s]`` — the same
    # occlusion and the same 1/r ray-density falloff the painter had, so the
    # "already correctly occluded incident flux" contract in the docstring
    # above still holds exactly. It is deliberately OUTSIDE the energy ledger:
    # nothing is debited to produce it and it changes no temperature.
    # ``heat`` is still read alongside it (max of the two) so the OTHER
    # writers of that plane — weapons/payload deposits, combustion at a burn
    # site — keep burning units exactly as before.
    #
    # T5b step 6, THE FLIP: the sensor is now `rad_flux_sweep`, the radiation
    # SWEEP's own body channel, not the retired cast's `rad_flux`. Three things
    # change with it, and all three are improvements:
    #   * it is a LEDGER EXIT, debited from the stream (the sweep's identity
    #     `sum rad_net + sum rad_flux + sum rad_amb == 0` holds in int64), where
    #     the old plane was an undebited incident ESTIMATE;
    #   * it is the absorbed energy per tick in the FOLD'S OWN CURRENCY, the
    #     same currency as `heat` -- so `max(heat, rad_flux)` below is now two
    #     exposures in one unit rather than two different kinds of number;
    #   * it is SIGNED (a body cooler than its surroundings books a negative
    #     net), which the `max` handles for free: `heat` is non-negative.
    # The body's share is `d - a` on `dyn_heat_atten_q` -- a unit MAX-stamps its
    # own `heat_atten` (default 1.0, an opaque body), so a marine on an air tile
    # absorbs the whole stream crossing its cell and re-emits ambient onward
    # (design v3 §6.2, row 25).
    #
    # #78: THE SWEEP'S PLANES ARE IN ITS FINE HEAT CURRENCY (2^k per heat
    # count, k = the engine table's `fine_bits`), `heat` in whole heat counts.
    # The sensor is converted ONCE, through the kit's twin
    # (optics_fixed.fine_heat_shr at s = 0: whole heat counts, symmetric), so
    # `max(heat, rad_flux)` compares two exposures in ONE unit and a marine burns
    # exactly as before in real units -- the fine bits refine the sweep's
    # exchange, they are not heat (tests/test_sweep_fine_currency.py: the same
    # HP per second within 1 %).
    from simulation import optics_fixed as _optics_fx
    h, w = gmap.heat.shape
    heat = gmap.heat
    rad_flux = getattr(gmap, "rad_flux_sweep", None)
    rad_fine_bits = _optics_fx.sweep_fine_bits(gmap)
    cmb = CFG.combat

    absorption   = float(cmb.unit_absorption)
    reflectivity = float(cmb.unit_reflectivity)
    flux_to_temp = float(cmb.heat_flux_to_temp)
    ambient_ref  = float(cmb.heat_ambient_ref)
    overtemp_k   = float(cmb.heat_overtemp_scale)
    temp_max_cfg = float(cmb.temperature_max)
    env_rate_cfg = float(cmb.environmental_damage_rate)

    dt_tick = 1.0 / float(ticks_per_second)

    for u in units:
        if not u.alive:
            continue

        # max-over-footprint incident flux (the hottest tile on the body is the
        # exposure that matters; shadowed tiles read ~0, max picks the burning
        # side). In-bounds guard for safety against off-grid footprints.
        peak_raw = 0
        # #123 credit: the hottest footprint tile and whether its exposure
        # is the `heat` deposit buffer (a claimable deposit: a jet, a splash,
        # an igniter's heat) or the radiation sensor (fire light: nobody's).
        peak_tile = None
        peak_from_heat = False
        for (tx, ty) in u.occupied_tiles():
            if 0 <= ty < h and 0 <= tx < w:
                v = int(heat[ty, tx])
                from_heat = True
                if rad_flux is not None:
                    # D3: the radiant sensor and the (non-radiative) heat
                    # writers are two independent exposures at the same tile —
                    # take the larger, exactly as the footprint loop takes the
                    # hottest tile. Both are Q16.16 in the same domain --
                    # once the sensor leaves the sweep's fine currency (#78).
                    fv = _optics_fx.fine_heat_shr(int(rad_flux[ty, tx]), 0,
                                                  rad_fine_bits)
                    if fv > v:
                        v = fv
                        from_heat = False
                if v > peak_raw:
                    peak_raw = v
                    peak_tile = (ty, tx)
                    peak_from_heat = from_heat
        if peak_raw <= 0:
            continue  # cold tile: no radiant flux, no heat damage

        phi = peak_raw / HEAT_SCALE
        phi_abs = phi * absorption * (1.0 - reflectivity)

        # Per-unit EnvironmentProfile band / rate, else the global fallback.
        env = getattr(u, "environment", None)
        temp_max = float(getattr(env, "temperature_max", temp_max_cfg))
        env_rate = float(getattr(env, "environmental_damage_rate", env_rate_cfg))

        t_felt = ambient_ref + flux_to_temp * phi_abs
        over = t_felt - temp_max
        if over <= 0.0:
            continue  # within the tolerance band: survivable, no damage

        dmg = env_rate * (1.0 + overtemp_k * over) * dt_tick

        # The DamagePacket pipeline (mechanics/06 §2) owns everything after
        # the pre-mitigation amount: mitigation applies the unit's resist
        # table at the EXACT pre-quantize position the old zombie ×4 branch
        # sat (the zombie's resist_mult[HEAT] = 4.0 — an exact binary scale —
        # reproduces it bit-for-bit; a marine's neutral table is an
        # IEEE-exact no-op), then Q2-lift quantize -> hp -> the same
        # hit/kill events as before (source "heat"; heat deaths never set
        # killed_by_zombie — only melee converts).
        #
        # CREDIT (#123): the claimant of the hurting tile's THIS-TICK heat
        # deposit (damage.TickCredit — the sprayer, a payload's owner), only
        # when the exposure there is the `heat` buffer; radiation, a burning
        # tile's own heat and an unclaimed deposit are the environment's.
        attacker = NO_ATTACKER
        if peak_from_heat and peak_tile is not None:
            attacker = tick_credit(gmap).attacker_at(CREDIT_HEAT, *peak_tile)
        apply_packet(u, DamagePacket(amount=dmg, dtype=HEAT,
                                     attacker_id=attacker,
                                     indirect=True),
                     events, source="heat")


# ---------------------------------------------------------------------------
# Blast damage to units — the `wave_p` blast coupling row. Moved VERBATIM
# from combat.py (P1).
# ---------------------------------------------------------------------------
def apply_blast_damage(units, fx, fy, radius, max_damage, events=None,
                       attacker_id=NO_ATTACKER):
    """Damage every unit within ``radius`` of (fx, fy), with linear falloff.

    ``attacker_id`` (#123) is the unit credited with the blast — the payload
    executor passes its owner (thrower, firer, planter); -1 = nobody (a
    timed charge, a scripted blast). A self-blast is credited to the
    thrower too (``attacker_id == unit_id``).

    Units below ``CFG.combat.blast_damage_threshold`` damage take none
    (prevents chip damage at the edge of distant blasts). Marks the
    unit dead if HP <= 0. Does NOT set ``killed_by_zombie`` — explosion
    deaths don't convert.

    If ``events`` is a list, append a :class:`UnitHitEvent` per hit and a
    :class:`UnitKilledEvent` per kill so the renderer can spawn matching
    visual effects.
    """
    for u in units:
        if not u.alive:
            continue
        uc_fx = u.center_tile_x()
        uc_fy = u.center_tile_y()
        dist = math.sqrt((uc_fx - fx) ** 2 + (uc_fy - fy) ** 2)
        if dist <= radius:
            falloff = 1.0 - (dist / radius)
            damage = int(max_damage * falloff)
            if damage >= CFG.combat.blast_damage_threshold:
                # BLAST packet through the pipeline (mechanics/06 §2). The
                # geometric pre-mitigation amount + the chip-damage threshold
                # stay site-side; neutral mitigation passes the int amount
                # through exactly, then the same Q2-lift quantize -> hp ->
                # hit/kill events as before (source "explosion"; blast
                # deaths never set killed_by_zombie — only melee converts).
                apply_packet(u,
                             DamagePacket(amount=damage, dtype=BLAST,
                                          attacker_id=attacker_id,
                                          indirect=True),
                             events, source="explosion")


# ---------------------------------------------------------------------------
# Wind on units -- the `wind | max` coupling row (#140; mechanics/05 §1, the
# knockdown spec mechanics/06 §4). Replaces the retired `wave_p | grad` push.
# docs/wind_on_units_design_2026-10-31.md; research basis
# docs/research/wind_on_a_person_2026-10-31.md.
# ---------------------------------------------------------------------------
def _stability_for(unit) -> float:
    """Resolve the unit's knockdown ``stability`` (mechanics/06 §4).

    Zombie-ness is runtime STATE on the human species, so the zombie's value
    is a state overlay selected here (``species.ZOMBIE_STABILITY``) — the
    exact ``mitigation_for`` pattern. Everyone else reads the species
    :class:`EnvironmentProfile` pointer stamped on the unit at construction
    (``unit.environment.stability``, default 1.0); bare stub units fall back
    to the human baseline 1.0. Since #140 it scales the knockdown WIND SPEED
    (the compare is against ``W_fall * stability^2``).
    """
    if getattr(unit, "is_zombie", False):
        return ZOMBIE_STABILITY
    env = getattr(unit, "environment", None)
    return float(getattr(env, "stability", 1.0))


#: The Q16 one the wind law's scale and stability are written in (the
#: cadence compose's one -- never ``movement.MOBILITY_ONE``).
WIND_ONE = _atm_fx.FP_ONE
#: ``W = (N / N_amb) * v^2 * 2^32``, so ``q = 1/2 rho_ref W / 2^32`` and a
#: threshold ``W_thr = q_thr * 2^33 / rho_ref``; 2^33 = 2 * BULK_ONE * FP_ONE.
_W_PER_Q_RHO = float(2 * gas_fixed.BULK_ONE * _atm_fx.FP_ONE)


@dataclass(frozen=True)
class WindSettings:
    """The wind law's dials, refused at the door and quantized ONCE per pass
    (door 2): ``w_slow`` / ``w_fall`` are the int wind loads at ``q_slow_pa``
    / ``q_fall_pa``; ``s_min_q16`` is the cadence scale at ``q_fall``."""
    q_slow_pa: float
    q_fall_pa: float
    slow_min_scale: float
    rho_ref: float
    getup_ticks: int
    w_slow: int
    w_fall: int
    s_min_q16: int


def _number(cfg, key):
    if not hasattr(cfg, key):
        raise KeyError(f"[exchange] {key} is missing (wind on units, #140)")
    v = getattr(cfg, key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"[exchange] {key} must be a number, got {v!r}")
    return float(v)


def wind_settings(cfg, rho_ref) -> WindSettings:
    """THE door of the wind law (#140, design §5). ``cfg`` is
    ``CFG.exchange``; ``rho_ref`` is the ``[materials.air]`` density
    (``gmap.materials.density[MAT_AIR]`` -- the one owner; there is no
    density dial here, :func:`gmap_wind_settings` passes it). Refuses, by
    name, a missing or mistyped key, ``q_slow >= q_fall``, two thresholds
    that quantize to the same load, ``wind_slow_min_scale`` outside (0, 1]
    and ``rho_ref <= 0``. Called every pass, so a Ctrl+R lands live."""
    q_slow = _number(cfg, "wind_q_slow_pa")
    q_fall = _number(cfg, "wind_q_fall_pa")
    s_min = _number(cfg, "wind_slow_min_scale")
    getup = int(_number(cfg, "knockdown_getup_ticks"))
    rho = float(rho_ref)
    if not rho > 0.0:
        raise ValueError(f"wind law: rho_ref (the [materials.air] density) "
                         f"must be > 0, got {rho_ref!r}")
    if not 0.0 <= q_slow < q_fall:
        raise ValueError(f"[exchange] wind_q_slow_pa ({q_slow}) must be >= 0 "
                         f"and below wind_q_fall_pa ({q_fall})")
    if not 0.0 < s_min <= 1.0:
        raise ValueError(f"[exchange] wind_slow_min_scale must be in (0, 1], "
                         f"got {s_min}")
    w_slow = int(q_slow * _W_PER_Q_RHO / rho + 0.5)
    w_fall = int(q_fall * _W_PER_Q_RHO / rho + 0.5)
    if w_slow >= w_fall:
        raise ValueError(f"[exchange] wind_q_slow_pa ({q_slow}) and "
                         f"wind_q_fall_pa ({q_fall}) quantize to the same "
                         f"wind load ({w_slow})")
    s_min_q16 = unit_fixed.quantize_scalar(s_min)
    return WindSettings(q_slow, q_fall, s_min, rho, getup,
                        w_slow, w_fall, s_min_q16)


def gmap_wind_settings(gmap) -> WindSettings:
    """:func:`wind_settings` for ``gmap``: the live ``[exchange]`` dials and
    the map's own ``[materials.air]`` density row."""
    from simulation.materials import MAT_AIR
    return wind_settings(CFG.exchange, gmap.materials.density[MAT_AIR])


def wind_scale_q16(w: int, settings: WindSettings) -> int:
    """The slowed band (design §2.1): the cadence scale (Q16, ``WIND_ONE`` =
    no slowdown) at wind load ``w``. Exactly ``WIND_ONE`` below ``w_slow``
    (the calm band is the dormancy guarantee, not ``w == 0``); linear in q
    (q is proportional to the drag force) down to ``s_min_q16`` at
    ``w_fall``; clamped there above. Pure int; ``w`` is clamped BEFORE the
    product so the slope term stays bounded."""
    w = int(w)
    lo, hi, s_min = settings.w_slow, settings.w_fall, settings.s_min_q16
    if w < lo:
        return WIND_ONE
    w_c = w if w < hi else hi
    scale = WIND_ONE - ((WIND_ONE - s_min) * (w_c - lo)) // (hi - lo)
    return scale if scale > s_min else s_min


def wind_knocks_down(w: int, stability_q16: int,
                     settings: WindSettings) -> bool:
    """The topple edge, THE pinned integer form (design §2.1):
    ``W * ONE^2 >= W_fall * stab_q16^2`` -- stability scales the knockdown
    wind SPEED (q goes as v^2)."""
    return (int(w) * WIND_ONE * WIND_ONE
            >= settings.w_fall * int(stability_q16) * int(stability_q16))


def apply_wind_knockdown(units, gmap):
    """The ``wind | max`` row's 9c2 output (#140, design §2.3): per LIVING
    unit, ``W = gmap.footprint_wind_load(...)`` (the max over the
    footprint's gas tiles of ``N_raw * (v^2 >> 16)``); at or above
    ``W_fall * stability^2`` the unit is laid KNOCKED_DOWN for
    ``knockdown_getup_ticks``, refresh-stacked while the wind holds (P3).
    It writes NO position -- nothing in the wind law displaces a unit; the
    slowed band's effect is the cadence scale the movement builder reads
    (``movement.footprint_samples``).

    Runs post-physics at ``Simulation.step`` 9c2, after the heat row (a unit
    the heat row killed this tick is a corpse). It reads the end-of-tick
    wind, which is the field MOVE_DIR and the zombie walk sample next tick,
    so a unit is either prone or slowed against the same field, never both.
    A status applied here starts suppressing NEXT tick (step-2b semantics).

    Determinism: every input is int (door 1) or a door-2 constant quantized
    once per pass; the compare is the pinned integer form. No RNG.
    """
    settings = gmap_wind_settings(gmap)
    for u in units:
        if not u.alive:
            continue
        w = gmap.footprint_wind_load(u.tile_y, u.tile_x, u.footprint)
        if w <= 0:
            continue
        stab_q16 = unit_fixed.quantize_scalar(_stability_for(u))
        if wind_knocks_down(w, stab_q16, settings):
            apply_status(u, KNOCKED_DOWN, magnitude=0,
                         duration_ticks=settings.getup_ticks, source_id=None)


# ---------------------------------------------------------------------------
# Gas coupling rows (mechanics/05 §1) — the owed `gas[teargas]` / `gas[poison]`
# rows, born INTO the table by weapons W3. Both mirror the heat row's shape:
# footprint reduction (max — the densest gas tile on the body is the exposure
# that matters, exactly the heat row's hottest-tile argument) -> a Q16.16
# integer threshold (config [exchange], quantized once per pass — door 2) ->
# the response. LAZY BY CONSTRUCTION: an all-zero plane early-outs on one
# integer .any(); a unit below threshold costs one integer reduction; the
# rows draw NO randomness ever — so every gas-free trajectory (the canonical
# golden included) is bit-identical to pre-W3.
# ---------------------------------------------------------------------------
def apply_teargas_blind(units, gmap):
    """The ``gas[teargas]`` coupling row (mechanics/05 §1, W3): footprint-max
    teargas density above ``teargas_blind_density`` applies BLINDED
    (``can_aim`` suppressed -> aimed fire collapses to the snap cone — the
    process_shooting cone-selection gate) for ``teargas_blind_ticks``,
    REFRESH-stacked: every qualifying tick re-ups the timer, so the blind
    outlasts leaving the cloud by the configured duration.

    Runs post-physics at ``Simulation.step`` 9c3 (after heat + wind — the
    documented within-tick exchange order), per LIVING unit in stored order
    (order-free anyway: each response reads only the field + its own state).
    Pure door-1 integer reduction + compare on a door-2 threshold; no RNG.
    A status applied here starts suppressing NEXT tick (the step-2b P3
    trigger-position semantics — this tick's shooting already ran at step 4).
    """
    slice_ = gmap.gas[GAS_TEARGAS]
    if not bool(slice_.any()):
        return                              # dormant plane: no teargas anywhere
    cfg = CFG.exchange
    # The threshold is quantized at the TRACE currency (#88 A1): the gate is
    # density >= the real-unit value, to within one fine count.
    thresh_q = gas_fixed.quantize_trace_scalar(float(cfg.teargas_blind_density))
    duration = int(cfg.teargas_blind_ticks)
    for u in units:
        if not u.alive:
            continue
        density_q = reduce_max(slice_, u.occupied_tiles())
        if density_q < thresh_q:
            continue
        apply_status(u, BLINDED, magnitude=0,
                     duration_ticks=duration, source_id=None)


def apply_poison_dose(units, gmap, ticks_per_second, events=None):
    """The ``gas[poison]`` coupling row (mechanics/05 §1, W3): footprint-max
    poison density at or above ``poison_min_density`` emits ONE POISON
    :class:`DamagePacket` per tick through the mechanics/06 pipeline.

    The per-tick amount KEEPS THE HEAT ROW'S IDIOM exactly — a door-3
    float64 chain on the door-1 field read and door-2 constants, quantized
    once at the HP boundary::

        density = dequantize_trace(density_q)          # exact 2^-TRACE_SHIFT
        amount  = poison_dps * density * (1 / tps)     # pure IEEE * (door 3)
        applied = dequantize(quantize(mitigated))      # apply_packet's
                                                       # quantize_hp_delta:
                                                       # round-half-away onto
                                                       # the Q16.16 grid

    so the applied delta is the integer-rounded (half away from zero)
    Q16.16 count of ``resist_mult[POISON] × poison_dps × density / tps`` —
    e.g. a marine (resist 1.0) in full density at 24 tps takes exactly
    0.25 HP/tick (16384 counts), 6 HP/s.

    LAZY EMISSION for the immune: a unit whose ``resist_mult[POISON]`` is
    exactly 0 (the zombie — they don't breathe; the future robot) draws NO
    packet at all, mirroring the lazy-roll rule: a packet that cannot matter
    is not emitted, so a horde standing in gas emits no 0-damage event spam
    (and takes 0 damage — poison is not the anti-horde answer; fire is).
    Resistances BETWEEN 0 and 1 still flow through the pipeline normally.
    Source string ``"poison_gas"``; gas deaths never convert (only melee).
    """
    slice_ = gmap.gas[GAS_POISON]
    if not bool(slice_.any()):
        return                              # dormant plane: no poison anywhere
    cfg = CFG.exchange
    thresh_q = gas_fixed.quantize_trace_scalar(float(cfg.poison_min_density))   # #88 A1
    dps = float(cfg.poison_dps)
    dt_tick = 1.0 / float(ticks_per_second)
    for u in units:
        if not u.alive:
            continue
        tiles = u.occupied_tiles()
        density_q = reduce_max(slice_, tiles)
        if density_q < thresh_q:
            continue
        if mitigation_for(u).resist_mult[POISON] == 0.0:
            continue                        # immune: no packet, no event (lazy)
        amount = dps * float(gas_fixed.dequantize_trace(density_q)) * dt_tick
        # CREDIT (#123): the claimant of THIS tick's poison deposit on the
        # densest footprint tile (the first one, in footprint order — the
        # tile reduce_max read): a miasma jet's sprayer, a gas payload's
        # owner. A cloud that only lingers is the environment's.
        attacker = NO_ATTACKER
        sh, sw = slice_.shape
        for (tx, ty) in tiles:
            if (0 <= ty < sh and 0 <= tx < sw
                    and int(slice_[ty, tx]) == density_q):
                attacker = tick_credit(gmap).attacker_at(GAS_POISON, ty, tx)
                break
        apply_packet(u,
                     DamagePacket(amount=amount, dtype=POISON,
                                  attacker_id=attacker, indirect=True),
                     events, source="poison_gas")


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


#: THE COUPLING TABLE (mechanics/05 §1) — ordered; it GROWS, one row per new
#: coupling (wind, water, gas, O2, fire, ... — see the chapter).
#: Table order is the chapter's row order and becomes the P0 execution order
#: ("couplings in table order") once the named EXCHANGE-READ slot lands
#: (P4-era); in P1 the registered rows still run at their legacy tick
#: positions (module docstring above).
COUPLING_TABLE: tuple[CouplingRow, ...] = (
    CouplingRow(
        field="heat",
        reduction="max",
        response=apply_environmental_damage,
        note=(
            "Radiant flux -> T_felt band -> damage (engine/06 §4). The shipped "
            "response computes its max-over-footprint inline (identical to "
            "REDUCTIONS['max'] on the non-negative heat field) and runs "
            "post-physics at Simulation.step 9c, before the end-of-tick heat "
            "clear."
        ),
    ),
    CouplingRow(
        field="wave_p",
        reduction=None,
        response=apply_blast_damage,
        note=(
            "Blast overpressure -> damage. The shipped response PREDATES the "
            "field read: it derives overpressure geometrically (linear radius "
            "falloff from the detonation point) instead of sampling wave_p, "
            "and runs at DETONATION SITES (grenade fuse-out in "
            "Simulation._update_projectiles; combat.process_door_explosives), "
            "not at a fixed tick slot."
        ),
    ),
    CouplingRow(
        field="wind",
        reduction="max",
        response=apply_wind_knockdown,
        note=(
            "Wind on units (#140; mechanics/05 §1 / 06 §4): the dynamic "
            "pressure q = 1/2 rho v^2 at the footprint as the int W = N_raw * "
            "(v^2 >> 16). The max is over the DERIVED per-tile W "
            "(GameMap.footprint_wind_load), not REDUCTIONS['max'] over a "
            "plane. Below wind_q_slow_pa nothing; up to wind_q_fall_pa a "
            "cadence scale (wind_scale_q16, composed in movement.default_"
            "speed through movement.footprint_samples); at W_fall * "
            "stability^2 KNOCKED_DOWN. Never a displacement. Replaces the "
            "retired wave_p|grad push row. Runs post-physics at "
            "Simulation.step 9c2, directly after the heat row."
        ),
    ),
    CouplingRow(
        field="gas[teargas]",
        reduction="max",
        response=apply_teargas_blind,
        note=(
            "Teargas -> BLINDED (mechanics/05 §1, W3): footprint-max density "
            "vs the Q16.16 teargas_blind_density threshold -> refresh-stacked "
            "BLINDED (can_aim off -> snap-cone fire, the owed P3 consumer). "
            "Runs post-physics at Simulation.step 9c3, after the wind row "
            "(within-tick exchange order: heat, wind, teargas, poison). "
            "Lazy: all-zero plane early-outs; no RNG."
        ),
    ),
    CouplingRow(
        field="gas[poison]",
        reduction="max",
        response=apply_poison_dose,
        note=(
            "Poison -> POISON DoT packets (mechanics/05 §1, W3): footprint-"
            "max density vs poison_min_density -> one packet/tick of "
            "poison_dps x density / tps through the mechanics/06 pipeline "
            "(zombie resist_mult[POISON]=0 -> immune, lazily skipped). Runs "
            "at Simulation.step 9c3 directly after the teargas row. Lazy: "
            "all-zero plane early-outs; no RNG."
        ),
    ),
)


__all__ = [
    "HEAT_SCALE",
    "REDUCTIONS",
    "reduce_center", "reduce_grad", "reduce_max", "reduce_mean", "reduce_sum",
    "CouplingRow", "COUPLING_TABLE",
    "apply_blast_damage", "apply_environmental_damage",
    "WindSettings", "wind_settings", "gmap_wind_settings", "wind_scale_q16",
    "wind_knocks_down", "apply_wind_knockdown",
    "apply_teargas_blind", "apply_poison_dose",
]
