"""Zombie AI: activation, chain propagation, the horde field's descent, the
bite, conversion.

THE LAW (gameplay pack v1 G9p, the residency design's P3d; Erik's ruling of
2026-11-10, ``docs/bots_zombies_speeds_design_2026-11-10.md`` §3): the horde
walks the integer distance field of :mod:`simulation.horde` -- no per-zombie
search, no A* (A* stays the host pathfinder of the bots and the tools).
Activation (sight radius + line of sight, then the wake-neighbours chain)
and the bite are the pre-G9p law's; its ``math.sqrt`` gates are integer
compares of squared tile distances now.

- :func:`update_zombies_tick` -- slot 5's law (the conductor's, or a match's
  zombie controller's through ``Simulation.drive_zombies``);
- :func:`convert_marines_to_zombies` -- the conversion (lifted from
  ``game.py``'s ``_end_execution``).

The walk cadence: ``z.speed_ticks_per_tile`` ``None`` (every zombie unless a
test set an int) is the ``[speeds]`` shamble (G1p-c). Integer throughout; no
random number.
"""
from __future__ import annotations

from config import CFG
from simulation import exposure as _exposure
from simulation import horde as _horde
from simulation import turning as _turning
from simulation import unit_impulse as _unit_impulse
from simulation.damage import KINETIC, DamagePacket, apply_packet
from simulation.factions import ZOMBIE_TEAM
from simulation.movement import default_speed, footprint_samples
from simulation.status import composed_flags


def _shamble_ticks(gmap) -> int:
    """The ``[speeds]`` shamble cadence on ``gmap``'s tile (a direct caller's
    fallback; the Simulation hands its own quantization in)."""
    from simulation.speeds import speed_table
    return speed_table(CFG, gmap.tile_size_m).zombie_shamble_ticks


def _d2(ax, ay, bx, by) -> int:
    """Squared tile distance (the integer form of the law's old sqrt)."""
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


# ---------------------------------------------------------------------------
# Per-tick AI step
# ---------------------------------------------------------------------------
def update_zombies_tick(gmap, units, tick, events=None, shamble_ticks=None,
                        settings=None, state=None):
    """Run one tick of the zombie law for every zombie in ``units``.

    Three passes:
      1. Trigger detection -- an inactive zombie with line of sight to a
         living non-zombie within ``[zombie] trigger_radius`` tiles wakes
         (squared distance, strictly inside).
      2. Chain propagation -- an awake zombie wakes every inactive one within
         ``propagation_radius`` tiles (until the set stabilises).
      3. Per awake zombie, in list order: bite the (Euclidean-)nearest prey
         when within ``footprint + 1`` tiles (its cooldown), else DESCEND
         the prey field one step at its cadence (:mod:`simulation.horde`).

    ``settings`` (:class:`~simulation.horde.HordeSettings`) and ``state``
    (the field cache, :class:`~simulation.horde.HordeState`) are the
    Simulation's, built once per map; a direct caller that passes none gets
    the config's settings on ``gmap``'s tile and a fresh cache. ``events``
    (the sim's ``tick_events``; #122) receives each bite's
    :class:`UnitHitEvent` and, on a kill, :class:`UnitKilledEvent`, both
    with source ``"melee"``; ``None`` emits nothing.

    The walk cadence (G1p-c): a zombie whose ``speed_ticks_per_tile`` is
    ``None`` walks at ``shamble_ticks`` -- the ``[speeds] zombie_shamble_mps``
    cadence the Simulation quantized once with its map's tile size; a direct
    caller that passes none gets the same quantization of
    ``gmap.tile_size_m``, the first time a zombie of the call walks. A
    zombie steps when its stride clock reaches its terrain-scaled cadence
    and a step is open; one that cannot step keeps its clock and steps the
    tick one opens.
    """
    # Prey = every living non-zombie, whatever its team (#119); the horde is
    # is_zombie -- never a team literal.
    players = [u for u in units if not u.is_zombie and u.alive]
    if not players:
        return

    zombies = [u for u in units if u.is_zombie and u.alive]
    if settings is None:
        settings = _horde.horde_settings(
            CFG, float(getattr(gmap, "tile_size_m", 1.0)))
    if state is None:
        state = _horde.HordeState()
    state.begin_tick(gmap, tick, settings)

    # ---- 1. Trigger detection (LOS + radius) ----
    trigger_r2 = settings.trigger_r2
    for z in zombies:
        if z.zombie_activated:
            continue
        zc_fx = z.center_tile_x()
        zc_fy = z.center_tile_y()
        for p in players:
            pc_fx = p.center_tile_x()
            pc_fy = p.center_tile_y()
            if _d2(zc_fx, zc_fy, pc_fx, pc_fy) < trigger_r2:
                if gmap.has_los(zc_fy, zc_fx, pc_fy, pc_fx):
                    z.zombie_activated = True
                    break

    # ---- 2. Chain activation: BFS until no new activations ----
    prop_r2 = settings.propagation_r2
    changed = True
    while changed:
        changed = False
        for z in zombies:
            if not z.zombie_activated:
                continue
            for z2 in zombies:
                if z2.zombie_activated or z2 is z:
                    continue
                if _d2(z.center_tile_x(), z.center_tile_y(),
                       z2.center_tile_x(), z2.center_tile_y()) < prop_r2:
                    z2.zombie_activated = True
                    changed = True

    # ---- 3. Bite or descend, per awake zombie ----
    for z in zombies:
        if not z.zombie_activated:
            continue

        # Status gate (mechanics/06 §4): the composed flags decide what this
        # zombie may DO this tick -- can_act gates the bite, can_move the
        # walk below. Activation (passes 1-2) is deliberately ungated: a
        # paralyzed zombie still notices you.
        flags = composed_flags(z)
        if not flags.can_move and not flags.can_act:
            continue

        # Nearest living prey by (squared) Euclidean distance: the bite's.
        nearest = None
        nearest_d2 = None
        zc_fx = z.center_tile_x()
        zc_fy = z.center_tile_y()
        for p in players:
            d2 = _d2(zc_fx, zc_fy, p.center_tile_x(), p.center_tile_y())
            if nearest_d2 is None or d2 < nearest_d2:
                nearest_d2 = d2
                nearest = p

        # Bite if adjacent (footprint + 1 tiles). An adjacent zombie that
        # cannot act (stunned/paralyzed/knocked down) stands and waits -- its
        # cooldown does NOT reset (nothing is spent by suppression).
        reach = z.footprint + 1
        if nearest_d2 <= reach * reach:
            cooldown = CFG.zombie.attack_cooldown_ticks
            if flags.can_act and tick - z.last_melee_tick >= cooldown:
                z.last_melee_tick = tick
                # KINETIC packet through the pipeline (mechanics/06 §2); the
                # hit and the kill ride the tick's events with source
                # "melee" (#122), credited to the biter (#123), and the kill
                # keeps melee's conversion semantics (killed_by_zombie, the
                # ONLY converting death, mechanics/06 §6).
                apply_packet(nearest,
                             DamagePacket(amount=CFG.zombie.melee_damage,
                                          dtype=KINETIC,
                                          attacker_id=getattr(z, "id", -1)),
                             events=events, source="melee",
                             mark_killed_by_zombie=True)
            continue

        # The walk: one tile every (terrain-scaled) cadence ticks. Status
        # gate: an immobilized zombie's stride clock PAUSES; so does one
        # catching itself after a push (G5p-a2, unit_impulse.walk_allowed).
        if not flags.can_move or not _unit_impulse.walk_allowed(z):
            continue
        # Terrain cadence (mobility design §4.1): the area-average mobility
        # over the CURRENT footprint scales the base cadence into the ticks
        # this step costs; the field stays speed-blind.
        samples = footprint_samples(
            gmap, z.tile_y, z.tile_x, z.footprint,
            effort_scale_q16=_unit_impulse.effort_scale_q16(
                _unit_impulse.table_for(gmap), z))
        cadence = z.speed_ticks_per_tile
        if cadence is None:
            if shamble_ticks is None:
                shamble_ticks = _shamble_ticks(gmap)
            cadence = shamble_ticks
        step_ticks = default_speed(samples, cadence)
        z.zombie_move_accumulator += 1
        if z.zombie_move_accumulator < step_ticks:
            continue
        fp = z.footprint
        field = state.prey_field(gmap, fp, players)
        step = _horde.descend(gmap, field, z.tile_y, z.tile_x, fp)
        if step is not None:
            ny, nx = step
            z.zombie_move_accumulator = 0
            z.face_towards(float(nx), float(ny))
            z.x = float(nx)
            z.y = float(ny)


# ---------------------------------------------------------------------------
# End-of-round conversion
# ---------------------------------------------------------------------------
def convert_marines_to_zombies(units):
    """Turn every marine that was killed by a zombie this round into a zombie.

    Lifted from ``game.py:_end_execution`` (lines 2019-2028). The marine
    keeps its inventory (per locked decision G — inventory is base, all
    units carry); a converted zombie that still has a grenade can have
    it cook off later through emergent fire interaction.

    Resets the per-unit conversion flag after.
    """
    for u in units:
        if not u.is_zombie and not u.alive and u.killed_by_zombie:
            # Joins the horde's team, so it turns on its old squad (#119);
            # its roster slot stays in the old team's roster.
            u.team = ZOMBIE_TEAM
            u.is_zombie = True
            u.alive = True
            u.current_hp = float(CFG.zombie.hp)
            # G4p: a zombie starts with no exposure (its death zeroed it
            # already; this is the conversion's own rule).
            _exposure.clear(u)
            # G5p-a1: the turn comes to rest where the body faces (a revived
            # corpse inherits no stale target or angular velocity).
            _turning.settle(u)
            # G5p-a2: nor a stale push (zeroed at death too; a push that
            # landed on the corpse after its killing hit is dropped here).
            _unit_impulse.clear(u)
            u.zombie_activated = True
            u.killed_by_zombie = False
            u.name = f"Z-{u.name}"
