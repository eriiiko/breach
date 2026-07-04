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

P4 (2026-07-05) adds the first NEW row — :func:`apply_wave_push`, the
``wave_p | grad`` impulse push + KNOCKED_DOWN trigger — invoked post-physics
at ``Simulation.step`` 9c2, directly after the heat row (the documented
within-tick exchange order: heat damage, then push).

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
from simulation.damage import BLAST, HEAT, DamagePacket, apply_packet
from simulation.species import ZOMBIE_STABILITY
from simulation.status import KNOCKED_DOWN, apply_status


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
# Environmental (radiant heat) damage to units — engine/06 §4, proposal §4.2
# — the `heat | max` coupling row. Moved VERBATIM from combat.py (P1).
# ---------------------------------------------------------------------------
# Q16.16 scale shared with the `heat`/`temperature` fields (cpp/src/raycaster.h
# HEAT_SCALE). One unit of heat energy == HEAT_SCALE raw int counts in the
# buffer; Phi divides back out to the energy-unit domain the [combat] consts and
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
    marine's neutral table is an IEEE-exact no-op. The sim no longer reads
    ``CFG.zombie.fire_damage_multiplier``.

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
    h, w = gmap.heat.shape
    heat = gmap.heat
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
        for (tx, ty) in u.occupied_tiles():
            if 0 <= ty < h and 0 <= tx < w:
                v = int(heat[ty, tx])
                if v > peak_raw:
                    peak_raw = v
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
        apply_packet(u, DamagePacket(amount=dmg, dtype=HEAT, source_id=None),
                     events, source="heat")


# ---------------------------------------------------------------------------
# Blast damage to units — the `wave_p` blast coupling row. Moved VERBATIM
# from combat.py (P1).
# ---------------------------------------------------------------------------
def apply_blast_damage(units, fx, fy, radius, max_damage, events=None):
    """Damage every unit within ``radius`` of (fx, fy), with linear falloff.

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
                                          source_id=None),
                             events, source="explosion")


# ---------------------------------------------------------------------------
# Wave impulse push + KNOCKED_DOWN trigger — the `wave_p | grad` coupling row
# (mechanics/05 §1; the knockdown spec is mechanics/06 §4). NEW in P4.
# ---------------------------------------------------------------------------
def _stability_for(unit) -> float:
    """Resolve the unit's knockdown ``stability`` (mechanics/06 §4).

    Zombie-ness is runtime STATE on the human species, so the zombie's value
    is a state overlay selected here (``species.ZOMBIE_STABILITY``) — the
    exact ``mitigation_for`` pattern. Everyone else reads the species
    :class:`EnvironmentProfile` pointer stamped on the unit at construction
    (``unit.environment.stability``, default 1.0); bare stub units fall back
    to the human baseline 1.0.
    """
    if getattr(unit, "is_zombie", False):
        return ZOMBIE_STABILITY
    env = getattr(unit, "environment", None)
    return float(getattr(env, "stability", 1.0))


def apply_wave_push(units, gmap, ticks_per_second):
    """The wave_p impulse-push row, BOTH outputs (mechanics/05 §1): the
    per-tick displacement nudge AND the KNOCKED_DOWN trigger (mechanics/06
    §4). Runs post-physics at ``Simulation.step`` 9c2, immediately AFTER the
    heat row (the documented within-tick exchange order: heat damage first,
    then push — a unit the heat row kills this tick is a corpse and is not
    displaced).

    Per LIVING unit, in stored order (deterministic; each unit's response
    reads only the field + its own state, so the result is order-free — the
    serial loop mirrors :func:`apply_environmental_damage`):

        (gx, gy) = reduce_grad(wave_p, occupied_tiles)     # Q16.16, uphill
        dvx = k_push * (-gx / 65536) / mass                # tiles/s
        dvy = k_push * (-gy / 65536) / mass
        KNOCKED_DOWN if dvx^2 + dvy^2 >= (threshold * stability)^2
        dx = clamp(dvx * dt, +-push_max_tile_per_tick)     # dt = 1/tps
        dy = clamp(dvy * dt, +-push_max_tile_per_tick)
        unit.x += dx ; unit.y += dy                        # wall-clamped

    Design notes (the P4 decisions, in one place):

    - **Reduction = the vocabulary's ``reduce_grad``** (footprint edge-line
      mean difference), NOT an inline per-tile Σ∇p. Why: it samples ONLY the
      unit's own footprint tiles, so a unit hugging a wall picks up no
      spurious force from solid cells (wave_p is 0 inside walls — an
      out-of-body stencil would fabricate wall-ward suction); and it is the
      shipped, integer-exact vocabulary entry the chapter row names. With
      the one shipped footprint (3x3, every unit) the chapter's
      area-scaling Σ form differs from this by a constant that ``k_push``
      absorbs; when footprint sizes diversify, the big-light-units-fly
      area scaling returns as an explicit footprint-area factor here.
    - **v1 is stateless**: no persistent velocity on the unit — an
      instantaneous per-tick nudge while the wave overlaps the footprint.
      Measured consequence (calibration 2026-07-05, config.toml [exchange]):
      a passing acoustic pulse pushes on its front and pulls on its tail
      (linear acoustics — the net impulse at a fixed point largely
      cancels), so the visible motion is a sharp BUFFET (peak hop 0.3-0.5
      tile, partial spring-back, net drift +-0.2 tile), not a sustained
      1-tile carry. The knockdown is the star output; a sustained blast-wind
      throw would read the atmosphere dome — a separate row, Erik's call.
    - **Knockdown compares SQUARES** — ``dvx*dvx + dvy*dvy`` against
      ``(threshold*stability)**2`` computed as a product — no sqrt needed
      (cheaper, and one fewer op to audit; the chain stays pure ``+ - x /``
      door 3 on door-2 constants and the door-1 integer gradient).
    - **Knockdown uses the UNCAPPED, UNCLAMPED dv** — the physical velocity
      change. The displacement cap and the wall clamp are motion sanity
      rails; a unit slammed against a wall by a blast still goes down.
    - **Trigger timing** (status.py P3 semantics): 9c2 sits AFTER this
      tick's status pass (step 2b), so a knockdown applied here starts
      suppressing movement/actions NEXT tick and lasts the full
      ``knockdown_getup_ticks`` from there — the wave visibly bowls the
      unit over on the spot, and re-knocks REFRESH the timer (P3 stacking).
    - **Wall clamp, per axis, x then y (fixed order → slide along walls)**:
      each axis's move is accepted only if the DESTINATION footprint stays
      in-bounds and enters no solid tile it does not already stand on (the
      already-standing carve-out covers walk-through door tiles, which are
      solid to gas but legally occupied). A blocked axis drops its
      displacement (the unit pins against the wall); the other axis still
      slides. The y test runs at the already-accepted x, so the final
      combined footprint is always explicitly checked. Units do NOT block
      each other (v1: bodies are not walls — overlap is already possible
      via pathfinding; ``gmap.solid`` is the only barrier).
    - **Determinism**: gradient is door-1 integer; ``/65536`` is an exact
      power-of-two scale; k_push / mass / dt / threshold / stability are
      door-2 constants; the chain is pure ``+ - x /`` float64 (door 3) into
      today's float positions (the Q16.16 position migration is a later
      arc). ``gx == gy == 0`` (quiet field / off-grid footprint) is an
      integer-exact early-out: the pass is a bit-identical no-op when no
      wave is up — which keeps every wave-free trajectory's digest
      untouched.
    """
    cfg = CFG.exchange
    k_push = float(cfg.k_push)
    cap = float(cfg.push_max_tile_per_tick)
    threshold = float(cfg.knockdown_dv_threshold)
    getup_ticks = int(cfg.knockdown_getup_ticks)
    dt_tick = 1.0 / float(ticks_per_second)

    wave_p = gmap.wave_p
    solid = gmap.solid
    h, w = solid.shape

    for u in units:
        if not u.alive:
            continue
        tiles = u.occupied_tiles()
        gx, gy = reduce_grad(wave_p, tiles)
        if gx == 0 and gy == 0:
            continue    # integer-exact no-op: quiet field or clipped footprint

        mass = float(u.mass)
        dvx = k_push * (-gx / 65536.0) / mass    # tiles/s (door 3)
        dvy = k_push * (-gy / 65536.0) / mass

        # Output 2 first in text order, but independent of the nudge: the
        # KNOCKED_DOWN trigger on the raw physical dv (squares — no sqrt).
        dv2 = dvx * dvx + dvy * dvy
        thr = threshold * _stability_for(u)
        if dv2 >= thr * thr:
            apply_status(u, KNOCKED_DOWN, magnitude=0,
                         duration_ticks=getup_ticks, source_id=None)

        # Output 1: the displacement nudge — capped, then wall-clamped.
        dx = dvx * dt_tick
        dy = dvy * dt_tick
        if dx > cap:
            dx = cap
        elif dx < -cap:
            dx = -cap
        if dy > cap:
            dy = cap
        elif dy < -cap:
            dy = -cap

        cur = set(tiles)

        def _free(nx: float, ny: float) -> bool:
            ax, ay = int(nx), int(ny)
            for (dxo, dyo) in u.offsets:
                tx, ty = ax + dxo, ay + dyo
                if not (0 <= tx < w and 0 <= ty < h):
                    return False
                if solid[ty, tx] and (tx, ty) not in cur:
                    return False
            return True

        # x first, then y at the accepted x — fixed order, slides along walls.
        nx = u.x + dx
        if dx != 0.0 and _free(nx, u.y):
            u.x = nx
        ny = u.y + dy
        if dy != 0.0 and _free(u.x, ny):
            u.y = ny


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
#: coupling (impulse push, water, gas, O2, fire, ... — see the chapter).
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
            "not at a fixed tick slot. The impulse-push row below is the "
            "chapter's first true wave_p footprint read."
        ),
    ),
    CouplingRow(
        field="wave_p",
        reduction="grad",
        response=apply_wave_push,
        note=(
            "Impulse push + KNOCKED_DOWN trigger (mechanics/05 §1 / 06 §4) — "
            "the first row born INTO the table (P4): dv = k_push*(-grad)/mass "
            "per tick, displacement wall-clamped, knockdown on dv^2 vs "
            "(threshold*stability)^2. Runs post-physics at Simulation.step "
            "9c2, directly after the heat row (documented within-tick order: "
            "heat damage, then push)."
        ),
    ),
)


__all__ = [
    "HEAT_SCALE",
    "REDUCTIONS",
    "reduce_center", "reduce_grad", "reduce_max", "reduce_mean", "reduce_sum",
    "CouplingRow", "COUPLING_TABLE",
    "apply_blast_damage", "apply_environmental_damage", "apply_wave_push",
]
