"""Zombie AI: activation, hearing, the horde field's descent (with the
wander, the fear of fire, the door bash and the sprint), the bite,
conversion.

THE LAW (gameplay pack v1 G9p, the residency design's P3d; Erik's rulings
of 2026-11-10, ``docs/bots_zombies_speeds_design_2026-11-10.md`` §3): the
horde walks the integer distance fields of :mod:`simulation.horde` -- no
per-zombie search, no A* (A* stays the host pathfinder of the bots and the
tools). Activation (sight radius + line of sight, then the wake-neighbours
chain) and the bite are the pre-G9p law's; its ``math.sqrt`` gates are
integer compares of squared tile distances now. A zombie is in one of three
states:

- AWAKE (``zombie_activated``; by sight or the chain, for good): bite the
  nearest prey in reach, else descend the PREY field -- at the ``[speeds]``
  SPRINT while it sees a living prey within ``[zombie] sprint_radius_m``,
  else the shamble;
- HEARING (a noise target, ``zombie_noise_*``; ``[zombie] hearing``): descend
  that noise's field at the shamble; on arrival (within
  ``noise_arrive_m``), on its timeout or with no path, it SETTLES -- dormant
  where it stands, its wander anchor moved there;
- DORMANT: with ``[zombie] wander``, one step every ``wander_period_seconds``
  in a direction hashed from (its id, the tick), within ``wander_leash_m`` of
  its anchor; without it, it stands.

A noise (``simulation.noise``, drained at this slot) wakes into HEARING every
non-awake zombie within its radius, walls ignored. With ``fear_fire`` no
step lands a body on a burning or hot tile; with ``bash_doors`` a descent
blocked by a closed (non-hull) door strikes it at the bite's cadence through
the wall-chew seam, each strike a noise. Every switch off is the stage-1
field law exactly.

- :func:`update_zombies_tick` -- slot 5's law (the conductor's, or a match's
  zombie controller's through ``Simulation.drive_zombies``);
- :func:`convert_marines_to_zombies` -- the conversion (lifted from
  ``game.py``'s ``_end_execution``).

Integer throughout; no random number.
"""
from __future__ import annotations

from config import CFG
from simulation import exposure as _exposure
from simulation import horde as _horde
from simulation import noise as _noise
from simulation import turning as _turning
from simulation import unit_impulse as _unit_impulse
from simulation.damage import KINETIC, DamagePacket, apply_packet
from simulation.factions import ZOMBIE_TEAM
from simulation.movement import default_speed, footprint_samples
from simulation.status import composed_flags


def _speed_table(gmap):
    """The ``[speeds]`` table on ``gmap``'s tile (a direct caller's
    fallback; the Simulation hands its own quantization in)."""
    from simulation.speeds import speed_table
    return speed_table(CFG, gmap.tile_size_m)


def _d2(ax, ay, bx, by) -> int:
    """Squared tile distance (the integer form of the law's old sqrt)."""
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _sees_prey(gmap, z, players, r2) -> bool:
    """Whether ``z`` has a line of sight to a living prey within ``r2``
    (squared tiles, inclusive): the sprint's condition."""
    zx, zy = z.center_tile_x(), z.center_tile_y()
    for p in players:
        px, py = p.center_tile_x(), p.center_tile_y()
        if _d2(zx, zy, px, py) <= r2 and gmap.has_los(zy, zx, py, px):
            return True
    return False


def _hear(zombies, noises, settings) -> None:
    """Every non-awake zombie within a noise's radius (squared tiles,
    inclusive; walls ignored) takes the NEAREST such noise as its target
    (ties to the earlier noise) with a fresh timeout; then at most
    ``noise_fields_max`` distinct targets stay live -- the newest (most
    timeout left, ties by tile) -- and a zombie whose target was dropped
    settles where it stands."""
    for z in zombies:
        if z.zombie_activated:
            continue
        zx, zy = z.center_tile_x(), z.center_tile_y()
        best = None
        for ty, tx, r in noises:
            d2 = _d2(zx, zy, tx, ty)
            if d2 <= r * r and (best is None or d2 < best[0]):
                best = (d2, ty, tx)
        if best is not None:
            z.zombie_noise_y = best[1]
            z.zombie_noise_x = best[2]
            z.zombie_noise_ticks = settings.noise_timeout_ticks
    live = {}
    for z in zombies:
        if not z.zombie_activated and z.zombie_noise_ticks > 0:
            t = (z.zombie_noise_y, z.zombie_noise_x)
            live[t] = max(live.get(t, 0), z.zombie_noise_ticks)
    if len(live) > settings.noise_fields_max:
        keep = set(sorted(live, key=lambda t: (-live[t], t))
                   [:settings.noise_fields_max])
        for z in zombies:
            if (not z.zombie_activated and z.zombie_noise_ticks > 0
                    and (z.zombie_noise_y, z.zombie_noise_x) not in keep):
                _horde.settle(z, settings)


def _bash(gmap, z, ny, nx, tile, tick, flags, settings, events) -> None:
    """A strike on the closed door blocking ``z``'s step: at the bite's
    cadence (``last_melee_tick``), only while it can act; through the
    wall-chew seam (``combat.chew_wall``: the door tile's own hit points,
    ``destroy_wall`` at 0, which the slot-9e door sweep completes for the
    whole door); each strike a noise at the struck tile."""
    if not flags.can_act:
        return
    if tick - z.last_melee_tick < CFG.zombie.attack_cooldown_ticks:
        return
    from simulation.combat import chew_wall
    from simulation.events import DoorDestroyedEvent
    from simulation.materials import MAT_DOOR_CLOSED
    z.last_melee_tick = tick
    z.face_towards(float(nx), float(ny))
    ty, tx = tile
    chew_wall(gmap, ty, tx, settings.door_bash_damage)
    if events is not None and int(gmap.material[ty, tx]) != MAT_DOOR_CLOSED:
        events.append(DoorDestroyedEvent(pos=(ty, tx)))
    _noise.emit(gmap, ty, tx, settings.bash_loudness_tiles)


def _walk(gmap, z, get_field, state, tick, flags, settings, events,
          base_ticks) -> str:
    """One tick of a zombie's walk at ``base_ticks`` per tile (terrain- and
    effort-scaled) on the field ``get_field()`` returns -- asked for only
    when the stride clock is due, so a field is flooded on the tick a step
    first needs it: ``"step"``, ``"bash"``, ``"wait"`` (no move open) or
    ``"clock"`` (its stride clock still running)."""
    samples = footprint_samples(
        gmap, z.tile_y, z.tile_x, z.footprint,
        effort_scale_q16=_unit_impulse.effort_scale_q16(
            _unit_impulse.table_for(gmap), z))
    step_ticks = default_speed(samples, base_ticks)
    z.zombie_move_accumulator += 1
    if z.zombie_move_accumulator < step_ticks:
        return "clock"
    move = _horde.descend(gmap, get_field(), z.tile_y, z.tile_x, z.footprint,
                          state)
    if move is None:
        return "wait"
    if move[0] == "step":
        _, ny, nx = move
        z.zombie_move_accumulator = 0
        z.face_towards(float(nx), float(ny))
        z.x = float(nx)
        z.y = float(ny)
        return "step"
    _, ny, nx, tile = move
    _bash(gmap, z, ny, nx, tile, tick, flags, settings, events)
    return "bash"


# ---------------------------------------------------------------------------
# Per-tick AI step
# ---------------------------------------------------------------------------
def update_zombies_tick(gmap, units, tick, events=None, shamble_ticks=None,
                        settings=None, state=None, noises=None, doors=(),
                        sprint_ticks=None):
    """Run one tick of the zombie law for every zombie in ``units``.

    Passes:
      1. Trigger detection -- a non-awake zombie with line of sight to a
         living non-zombie within ``[zombie] trigger_radius`` tiles wakes
         (squared distance, strictly inside).
      2. Chain propagation -- an awake zombie wakes every non-awake one
         within ``propagation_radius`` tiles (until the set stabilises). A
         zombie that wakes drops its noise and its anchor.
      2b. Hearing (``[zombie] hearing``) -- ``noises`` (the slot's drained
         ``GameMap.noise_log``) give every non-awake zombie within a radius
         a target (:func:`_hear`).
      3. Per zombie, in list order: AWAKE -- bite the (Euclidean-)nearest
         prey when within ``footprint + 1`` tiles (its cooldown), else walk
         the prey field (sprinting while it sees prey within the sprint
         radius); HEARING -- walk the noise's field, settling on arrival,
         timeout or no path; DORMANT -- the wander.

    ``settings`` (:class:`~simulation.horde.HordeSettings`), ``state`` (the
    field cache, :class:`~simulation.horde.HordeState`) and ``doors`` (the
    sim's door runtimes, for the bash) are the Simulation's; a direct caller
    that passes none gets the config's settings on ``gmap``'s tile, a fresh
    cache and no doors. ``noises`` ``None`` drains ``gmap``'s log here.
    ``events`` (the sim's ``tick_events``; #122) receives each bite's
    :class:`UnitHitEvent` and, on a kill, :class:`UnitKilledEvent`, both
    with source ``"melee"``, and a broken door tile's
    :class:`DoorDestroyedEvent`; ``None`` emits nothing.

    The walk cadence (G1p-c): a zombie whose ``speed_ticks_per_tile`` is
    ``None`` walks at ``shamble_ticks`` (or, sprinting, ``sprint_ticks``) --
    the ``[speeds]`` rows the Simulation quantized once for its map; a
    direct caller that passes none gets the same quantization of
    ``gmap.tile_size_m`` the first time a zombie of the call walks. A zombie
    steps when its stride clock reaches its terrain-scaled cadence and a
    step is open; one that cannot step keeps its clock and steps the tick
    one opens.
    """
    if noises is None:
        noises = _noise.take(gmap)
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
    state.begin_tick(gmap, tick, settings, doors)

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
    for z in zombies:
        if z.zombie_activated and (z.zombie_noise_ticks
                                   or z.zombie_anchor_y >= 0):
            _horde.clear(z)            # awake for good: no noise, no anchor

    # ---- 2b. Hearing ----
    if settings.hearing and noises:
        _hear(zombies, noises, settings)

    # ---- 3. Per zombie ----
    cooldown = CFG.zombie.attack_cooldown_ticks
    for z in zombies:
        # Status gate (mechanics/06 §4): the composed flags decide what this
        # zombie may DO this tick -- can_act gates the bite and the bash,
        # can_move the walk. Activation and hearing are deliberately
        # ungated: a paralyzed zombie still notices you.
        flags = composed_flags(z)

        if z.zombie_activated:
            if not flags.can_move and not flags.can_act:
                continue
            # Nearest living prey by (squared) Euclidean distance: the
            # bite's.
            nearest = None
            nearest_d2 = None
            zc_fx = z.center_tile_x()
            zc_fy = z.center_tile_y()
            for p in players:
                d2 = _d2(zc_fx, zc_fy, p.center_tile_x(), p.center_tile_y())
                if nearest_d2 is None or d2 < nearest_d2:
                    nearest_d2 = d2
                    nearest = p
            # Bite if adjacent (footprint + 1 tiles). An adjacent zombie
            # that cannot act stands and waits -- its cooldown does NOT
            # reset (nothing is spent by suppression).
            reach = z.footprint + 1
            if nearest_d2 <= reach * reach:
                if flags.can_act and tick - z.last_melee_tick >= cooldown:
                    z.last_melee_tick = tick
                    # KINETIC packet through the pipeline (mechanics/06 §2);
                    # the hit and the kill ride the tick's events with
                    # source "melee" (#122), credited to the biter (#123),
                    # and the kill keeps melee's conversion semantics
                    # (killed_by_zombie, the ONLY converting death).
                    # #185: the bite travels biter centre -> prey centre.
                    zx_q, zy_q = _unit_impulse.centre_q(z)
                    px_q, py_q = _unit_impulse.centre_q(nearest)
                    apply_packet(nearest,
                                 DamagePacket(amount=CFG.zombie.melee_damage,
                                              dtype=KINETIC,
                                              attacker_id=getattr(z, "id",
                                                                  -1),
                                              dir_x_q=px_q - zx_q,
                                              dir_y_q=py_q - zy_q),
                                 events=events, source="melee",
                                 mark_killed_by_zombie=True)
                continue
            # The walk. Status gate: an immobilized zombie's stride clock
            # PAUSES; so does one catching itself after a push (G5p-a2).
            if not flags.can_move or not _unit_impulse.walk_allowed(z):
                continue
            base = z.speed_ticks_per_tile
            if base is None:
                if settings.sprint and _sees_prey(gmap, z, players,
                                                  settings.sprint_r2):
                    if sprint_ticks is None:
                        sprint_ticks = _speed_table(gmap).zombie_sprint_ticks
                    base = sprint_ticks
                else:
                    if shamble_ticks is None:
                        shamble_ticks = _speed_table(
                            gmap).zombie_shamble_ticks
                    base = shamble_ticks
            fp = z.footprint

            def prey_field(_fp=fp, _z=z):
                f = state.prey_field(gmap, _fp, players)
                if (settings.fear_fire and _horde.field_at(
                        f, _z.tile_y, _z.tile_x) == _horde.UNREACHED):
                    # Fire closes every way: walk the goal flooded without
                    # the fear, up to the fire's edge (its steps still
                    # refuse every feared block), and wait there.
                    f = state.prey_field(gmap, _fp, players, fearless=True)
                return f
            _walk(gmap, z, prey_field, state, tick, flags, settings, events,
                  base)
            continue

        if z.zombie_noise_ticks > 0:
            # HEARING: walk the noise's field at the shamble.
            z.zombie_noise_ticks -= 1
            if z.zombie_noise_ticks <= 0:
                _horde.settle(z, settings)     # timed out
                continue
            if not flags.can_move or not _unit_impulse.walk_allowed(z):
                continue
            field = state.noise_field(gmap, z.footprint, z.zombie_noise_y,
                                      z.zombie_noise_x)
            here = _horde.field_at(field, z.tile_y, z.tile_x)
            if here <= settings.noise_arrive_tiles:
                _horde.settle(z, settings)     # arrived, saw nobody
                continue
            base = z.speed_ticks_per_tile
            if base is None:
                if shamble_ticks is None:
                    shamble_ticks = _speed_table(gmap).zombie_shamble_ticks
                base = shamble_ticks
            got = _walk(gmap, z, lambda: field, state, tick, flags, settings,
                        events, base)
            if got == "wait" and here == _horde.UNREACHED:
                _horde.settle(z, settings)     # no path to the noise
            continue

        if settings.wander:
            # DORMANT: one hashed step every period, within the leash.
            if z.zombie_anchor_y < 0:
                z.zombie_anchor_y = int(z.tile_y)
                z.zombie_anchor_x = int(z.tile_x)
            if not _horde.wander_due(getattr(z, "id", 0), tick,
                                     settings.wander_period_ticks):
                continue
            if not flags.can_move or not _unit_impulse.walk_allowed(z):
                continue
            to = _horde.wander_step(gmap, z.tile_y, z.tile_x,
                                    z.zombie_anchor_y, z.zombie_anchor_x,
                                    settings.wander_leash_r2,
                                    getattr(z, "id", 0), tick)
            if to is None:
                continue
            ny, nx = to
            fp = z.footprint
            if not gmap.is_passable_block(ny, nx, fp):
                continue
            if settings.fear_fire and _horde.block_feared(
                    gmap, ny, nx, fp, settings.fear_temp_q16):
                continue
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
            # G9p: nor a noise or a wander anchor (cleared at death too).
            _horde.clear(u)
            u.zombie_activated = True
            u.killed_by_zombie = False
            u.name = f"Z-{u.name}"
