"""Combat: projectiles, shots, shooting bursts, line-of-sight, blast damage.

Lifted from ``game.py``:

- :class:`Projectile`        (game.py:1195-1233)
- :class:`Shot`              (game.py:1239-1244)
- :func:`apply_blast_damage` (game.py:1981-1996 — was ``Game._apply_blast_damage``)
- :func:`process_shooting`   (game.py:1749-1791 — was ``Game._process_shooting``)
- :func:`auto_fire`          (game.py:1793-1818 — was ``Game._auto_fire``)
- :func:`fire_burst`         (game.py:1820-1865 — was ``Game._fire_burst``)
- :func:`process_door_explosives`  (game.py:1641-1658)

All shooting / burst / LOS code mutates the unit list and emits ``Shot``
tracer events into a caller-owned list. The pure physics-event entry
points (``apply_explosion``, ``add_explosion_smoke``) live in
:mod:`simulation.physics`; combat just calls into them at detonation
sites.

Determinism — Phase 2:
    ``fire_burst`` and the explosion-smoke helper accept a
    :class:`numpy.random.Generator` (``rng`` parameter) for the bullet
    cone offsets and smoke noise. The Simulation facade owns one
    :class:`numpy.random.Generator` and plumbs it through these calls
    on every tick so AI rollouts can be reproduced bit-for-bit. If you
    invoke these helpers ad-hoc (e.g. from a test), pass an RNG with a
    known seed — the legacy fallback to process-global ``random`` has
    been removed.
"""
from __future__ import annotations

import math

import numpy as np

from config import CFG
from simulation.events import (
    ShotFiredEvent, ExplosionEvent, UnitHitEvent, UnitKilledEvent,
)
from simulation.orders import (
    ORDER_GRENADE, ORDER_EXPLOSIVE, ORDER_FIRE, ORDER_MOVE_ATTACK,
)
from simulation.physics import apply_explosion, add_explosion_smoke


# ---------------------------------------------------------------------------
# Projectile: in-flight grenade
# ---------------------------------------------------------------------------
class Projectile:
    """In-flight grenade. Travels in a straight line from start to target;
    detonates when ``current_tick >= get_detonate_tick()``."""

    def __init__(self, proj_type, start_fx, start_fy, target_fx, target_fy,
                 fuse_seconds, thrown_tick):
        self.proj_type = proj_type
        self.fx = float(start_fx)
        self.fy = float(start_fy)
        self.start_fx = float(start_fx)
        self.start_fy = float(start_fy)
        self.target_fx = float(target_fx)
        self.target_fy = float(target_fy)
        self.fuse_seconds = fuse_seconds
        self.thrown_tick = thrown_tick
        self.detonated = False
        self.travel_speed = CFG.weapons.grenade.travel_speed

    def get_detonate_tick(self):
        """Tick at which this projectile detonates."""
        tps = CFG.clock.ticks_per_second
        return self.thrown_tick + int(self.fuse_seconds * tps)

    def update_position(self, current_tick):
        """Move the projectile toward the target based on travel time so far."""
        tps = CFG.clock.ticks_per_second
        elapsed_sec = (current_tick - self.thrown_tick) / tps
        dx = self.target_fx - self.start_fx
        dy = self.target_fy - self.start_fy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            self.fx = self.target_fx
            self.fy = self.target_fy
            return
        travel_time = dist / self.travel_speed
        if elapsed_sec >= travel_time:
            self.fx = self.target_fx
            self.fy = self.target_fy
        else:
            frac = elapsed_sec / travel_time
            self.fx = self.start_fx + dx * frac
            self.fy = self.start_fy + dy * frac


# ---------------------------------------------------------------------------
# Shot: visual tracer event
# ---------------------------------------------------------------------------
class Shot:
    """A single rifle tracer for visual feedback. Lives on a short-lived
    effects list owned by the caller; the renderer expires entries past
    ``self.duration``. Not gameplay state — purely visual."""

    def __init__(self, fx1, fy1, fx2, fy2, time):
        self.fx1, self.fy1 = fx1, fy1
        self.fx2, self.fy2 = fx2, fy2
        self.time = time
        self.duration = CFG.combat.shot_tracer_duration


# ---------------------------------------------------------------------------
# Blast damage to units
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
                u.current_hp -= damage
                if events is not None:
                    uid = getattr(u, "id", -1)
                    events.append(UnitHitEvent(unit_id=uid, damage=damage,
                                                source="explosion"))
                if u.current_hp <= 0:
                    u.alive = False
                    if events is not None:
                        uid = getattr(u, "id", -1)
                        events.append(UnitKilledEvent(unit_id=uid,
                                                       killed_by="explosion"))


# ---------------------------------------------------------------------------
# Environmental (radiant heat) damage to units — engine/06 §4, proposal §4.2
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
        if u.is_zombie: dmg *= zombie.fire_damage_multiplier      # the shipped 4.0
        u.current_hp -= dmg

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
    zombie_mult  = float(CFG.zombie.fire_damage_multiplier)

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
        if u.is_zombie:
            dmg *= zombie_mult

        u.current_hp -= dmg
        if events is not None:
            uid = getattr(u, "id", -1)
            events.append(UnitHitEvent(unit_id=uid, damage=dmg,
                                       source="heat"))
        if u.current_hp <= 0:
            u.alive = False
            if events is not None:
                uid = getattr(u, "id", -1)
                events.append(UnitKilledEvent(unit_id=uid, killed_by="heat"))


# ---------------------------------------------------------------------------
# Ignition from temperature — engine/06 §4 ("Ignition"), proposal §6 step 4b
# ---------------------------------------------------------------------------
def apply_temperature_ignition(gmap, o2_threshold, ignition_seed):
    """Ignite flammable tiles whose `temperature` has crossed `ignition_temp`
    AND that have oxygen (engine/06 §4, proposal §6 step 4b).

    This is the READ side of the temperature substrate: the C++
    ``TemperatureSolver`` (convert -> conduction -> cooling, inside
    ``PhysicsRunner.step``) has already filled ``gmap.temperature`` for this
    tick; here we gather it and start fires. For each FLAMMABLE tile::

        if temperature[y,x] >= ignition_temp_q16[material]   # Q16.16 threshold
           and mean(atmosphere of air-side 4-neighbours) >= o2_threshold:   # O2
               fire[y,x] = max(fire[y,x], ignition_seed)      # never lowers a fire

    Determinism / coexistence:

    - **Q16.16 threshold compare.** ``temperature`` is Q16.16 fixed-point;
      ``ignition_temp_q16`` is the per-material threshold quantized into the
      SAME domain ONCE at load (``MaterialTable.ignition_temp_q16``). The test
      is a plain integer ``>=`` — no per-tick rescale, no float on the threshold
      path, bit-identical cross-machine.
    - **Flammable gate.** Non-flammable materials carry ``ignition_temp = 0`` (so
      ``q16 == 0``); gating on ``gmap.flammable`` is what stops a red-hot hull or
      glass tile (also threshold 0) from ever catching. Only fuel ignites.
    - **O2 check reuses the existing fire semantics.** The C++ ``FireSimulation``
      kills a burning tile when the mean ``atmosphere`` of its 4-connected
      AIR-SIDE (non-solid) neighbours drops below ``o2_threshold``; ignition uses
      the SAME predicate (same threshold, same neighbourhood) so a tile cannot be
      ignited into a state the fire step would immediately suffocate. A flammable
      tile is itself solid (wood/door), so its O2 comes from the adjacent air.
    - **``max``, not assign.** ``fire = max(fire, ignition_seed)`` never lowers an
      existing, possibly larger, fire (e.g. one an explosion already set).
    - **Second, additive ignition path.** This runs ALONGSIDE the existing
      cellular fire spread/ignition — it does not replace it. With no sim heat
      sources wired yet, ``temperature`` is ~0, so this path is DORMANT in-game
      (no behaviour change) until fire/beams emit heat — the intended safe seam.
    - **Deterministic.** Fixed row-major traversal (vectorised, order-independent
      anyway — a pure gather that writes each tile from frozen inputs), no RNG.

    Parameters
    ----------
    gmap
        The :class:`GameMap`. Reads ``temperature`` (Q16.16), ``material``,
        ``flammable``, ``solid``, ``atmosphere`` and the material table's
        ``ignition_temp_q16``; writes ``fire`` in place.
    o2_threshold : float
        Minimum air-side-neighbour atmosphere for ignition (reuse the fire's
        ``o2_threshold``, 0.60).
    ignition_seed : float
        Seed intensity a freshly-ignited tile gets (``I_seed`` ~ 0.1).

    Must run AFTER physics fills ``temperature`` (so the threshold sees this
    tick's heat) and ALONGSIDE the other temperature consumers (unit damage).
    """
    flammable = gmap.flammable
    if not bool(flammable.any()):
        return  # no fuel on this map -> nothing can ever ignite

    temperature = gmap.temperature
    material = gmap.material
    # Per-material Q16.16 ignition threshold (quantized once at load). Project
    # onto the grid so the compare is element-wise against `temperature`.
    thresh_q16 = gmap.materials.ignition_temp_q16[material]   # (h, w) int64

    # Hot enough? (Q16.16 integer compare.) Restrict to flammable tiles only.
    hot = flammable & (temperature.astype(np.int64) >= thresh_q16)
    if not bool(hot.any()):
        return  # nothing crossed its threshold this tick (the dormant case)

    # --- O2 proxy: mean `atmosphere` over the air-side (non-solid) 4-neighbours,
    # exactly the predicate the C++ fire O2 check uses. Vectorised: accumulate
    # neighbour atmosphere and an air-side count per tile, then average. A
    # flammable wall tile's own cell is solid; its oxygen comes from adjacent air.
    h, w = temperature.shape
    air_side = ~gmap.solid                      # True on tiles that count toward O2
    # S2c: atmosphere is int32 Q16.16 — dequantize to REAL pressure so the O2
    # proxy (mean atmosphere >= o2_threshold) matches the C++ fire O2 check.
    from simulation import atmosphere_fixed as _atm_fx
    atm = _atm_fx.dequantize_f32(gmap.atmosphere)
    sum_atm = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    # N, S, E, W (fixed order; sum is order-independent regardless).
    for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        ys0, ys1 = max(0, -dy), h - max(0, dy)      # this-tile row span
        xs0, xs1 = max(0, -dx), w - max(0, dx)
        nbr_atm = atm[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        nbr_air = air_side[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        sum_atm[ys0:ys1, xs0:xs1] += np.where(nbr_air, nbr_atm, 0.0)
        count[ys0:ys1, xs0:xs1] += nbr_air
    # Match the C++ guard `if (count < 1) count = 1` so a fully walled-in tile
    # (no air neighbour) averages to 0 -> below threshold -> never ignites.
    safe_count = np.where(count < 1.0, 1.0, count)
    has_o2 = (sum_atm / safe_count) >= o2_threshold

    ignite = hot & has_o2
    if not bool(ignite.any()):
        return
    # max(fire, ignition_seed) on the igniting tiles only — never lowers a bigger
    # existing fire.
    np.maximum(gmap.fire, np.where(ignite, np.float32(ignition_seed), gmap.fire),
               out=gmap.fire)


# ---------------------------------------------------------------------------
# Shooting (fire orders + auto-fire)
# ---------------------------------------------------------------------------
def process_shooting(gmap, units, tick, shots, real_time, rng, events=None):
    """Run one tick of shooting for every player unit with a fire order
    (or auto-fire during Move & Attack).

    Lifted from ``game.py:_process_shooting``. ``shots`` is a list the
    caller owns; new :class:`Shot` tracer events are appended to it.
    ``real_time`` is the wall-clock seconds since the round started
    (used as the Shot's spawn time for tracer fade-out). ``rng`` is the
    simulation's :class:`numpy.random.Generator` (used in
    :func:`fire_burst` for the per-bullet cone). If ``events`` is a list,
    :class:`ShotFiredEvent` / :class:`UnitHitEvent` / :class:`UnitKilledEvent`
    are appended for the renderer to consume.
    """
    tpp = CFG.clock.ticks_per_phase
    phase = tick // tpp
    burst_interval = CFG.weapons.rifle.burst_interval_ticks

    for u in units:
        if u.team != 0 or not u.alive:
            continue

        fire_order = u.get_fire_order_in_phase(phase)
        if not fire_order:
            # Move & Attack: auto-fire at nearest visible enemy.
            for o in u.orders:
                if o.order_type == ORDER_MOVE_ATTACK and o.phase == phase:
                    auto_fire(gmap, units, u, tick, shots, real_time, rng,
                              events=events)
                    break
            continue

        # Burst cadence gate.
        if tick - u.last_fire_tick < burst_interval:
            continue

        target_fx = fire_order.target_fx
        target_fy = fire_order.target_fy
        uc_fx = u.center_tile_x()
        uc_fy = u.center_tile_y()

        # Range check.
        dist = math.sqrt((uc_fx - target_fx) ** 2 + (uc_fy - target_fy) ** 2)
        if dist > CFG.weapons.rifle.range_tiles:
            continue

        # LOS check.
        if not gmap.has_los(uc_fy, uc_fx, target_fy, target_fx):
            continue

        fire_burst(gmap, units, u, uc_fx, uc_fy, target_fx, target_fy,
                   tick, shots, real_time, rng, events=events)
        u.last_fire_tick = tick


def auto_fire(gmap, units, u, tick, shots, real_time, rng, events=None):
    """Find the nearest visible enemy within rifle range and fire a burst.

    Lifted from ``game.py:_auto_fire``. Skipped if still within the burst
    cooldown.
    """
    burst_interval = CFG.weapons.rifle.burst_interval_ticks
    if tick - u.last_fire_tick < burst_interval:
        return

    uc_fx = u.center_tile_x()
    uc_fy = u.center_tile_y()
    best_dist = float('inf')
    best_enemy = None

    for e in units:
        if e.team == u.team or not e.alive:
            continue
        ec_fx = e.center_tile_x()
        ec_fy = e.center_tile_y()
        dist = math.sqrt((uc_fx - ec_fx) ** 2 + (uc_fy - ec_fy) ** 2)
        if dist <= CFG.weapons.rifle.range_tiles and dist < best_dist:
            if gmap.has_los(uc_fy, uc_fx, ec_fy, ec_fx):
                best_dist = dist
                best_enemy = e

    if best_enemy:
        fire_burst(gmap, units, u, uc_fx, uc_fy,
                   best_enemy.center_tile_x(), best_enemy.center_tile_y(),
                   tick, shots, real_time, rng, events=events)
        u.last_fire_tick = tick


def fire_burst(gmap, units, shooter, fx1, fy1, fx2, fy2,
               tick, shots, real_time, rng, events=None):
    """Fire a burst of bullets from (fx1, fy1) toward (fx2, fy2).

    Lifted from ``game.py:_fire_burst``. Each bullet picks a random
    angle within the rifle's cone (sampled from ``rng`` — a
    :class:`numpy.random.Generator`), marches tile-by-tile, stops on
    wall hit or unit hit. Zombies take damage scaled by
    ``CFG.zombie.bullet_damage_multiplier``. Every bullet appends one
    :class:`Shot` tracer to ``shots`` regardless of hit/miss; if
    ``events`` is a list, a matching :class:`ShotFiredEvent` is also
    appended, plus :class:`UnitHitEvent` / :class:`UnitKilledEvent` on
    a hit.
    """
    cone = math.radians(CFG.weapons.rifle.cone_half_angle_degrees)
    n_bullets = CFG.weapons.rifle.bullets_per_burst
    dmg = CFG.weapons.rifle.damage_per_bullet
    base_angle = math.atan2(fy2 - fy1, fx2 - fx1)
    h, w = gmap.material.shape

    shooter_id = getattr(shooter, "id", -1)

    for _ in range(n_bullets):
        angle = base_angle + float(rng.uniform(-cone, cone))
        rx, ry = float(fx1), float(fy1)
        hit_unit = None
        for _step in range(int(CFG.weapons.rifle.range_tiles)):
            rx += math.cos(angle)
            ry += math.sin(angle)
            ix, iy = int(rx), int(ry)

            if 0 <= iy < h and 0 <= ix < w:
                if gmap.solid[iy, ix]:
                    break
            else:
                break

            for e in units:
                if e is shooter or not e.alive:
                    continue
                if (e.tile_x <= ix < e.tile_x + e.footprint
                        and e.tile_y <= iy < e.tile_y + e.footprint):
                    hit_unit = e
                    break
            if hit_unit:
                break

        if hit_unit:
            actual_dmg = dmg
            if hit_unit.team == 1:  # zombie
                actual_dmg = int(dmg * CFG.zombie.bullet_damage_multiplier)
            hit_unit.current_hp -= actual_dmg
            if events is not None:
                hit_id = getattr(hit_unit, "id", -1)
                events.append(UnitHitEvent(unit_id=hit_id, damage=actual_dmg,
                                            source="bullet"))
            if hit_unit.current_hp <= 0:
                hit_unit.alive = False
                if events is not None:
                    hit_id = getattr(hit_unit, "id", -1)
                    events.append(UnitKilledEvent(unit_id=hit_id,
                                                   killed_by="bullet"))

        shots.append(Shot(fx1, fy1, rx, ry, real_time))
        if events is not None:
            hit_id = getattr(hit_unit, "id", -1) if hit_unit else None
            events.append(ShotFiredEvent(
                unit_id=shooter_id,
                from_tile=(fx1, fy1),
                to_tile=(rx, ry),
                hit_target_id=hit_id,
            ))


# ---------------------------------------------------------------------------
# Door explosives (scheduled detonations at phase boundaries)
# ---------------------------------------------------------------------------
def process_door_explosives(gmap, queue, units, slot, rng, events=None):
    """Detonate every door-explosive order scheduled for ``slot``.

    Lifted from ``game.py:_process_door_explosives``. Called three times
    per round (start P1, between phases, end P2). Skips zombies — only
    player-issued orders detonate. ``rng`` flows into
    :func:`simulation.physics.add_explosion_smoke` for the per-tile
    noise; ``events`` (optional) collects :class:`ExplosionEvent` and
    unit hit / kill events for the renderer.
    """
    radius   = CFG.weapons.door_explosive.blast_radius
    pressure = CFG.weapons.door_explosive.pressure
    wall_dmg = CFG.weapons.door_explosive.wall_damage
    unit_dmg = CFG.weapons.door_explosive.unit_damage

    for u in units:
        if u.team != 0:
            continue
        for o in u.orders:
            if o.order_type == ORDER_EXPLOSIVE and o.det_slot == slot:
                fy, fx = o.target_fy, o.target_fx
                apply_explosion(gmap, queue, fy, fx, radius, pressure, wall_dmg)
                apply_blast_damage(units, fx, fy, radius, unit_dmg,
                                   events=events)
                add_explosion_smoke(gmap, queue, fy, fx, radius)
                if events is not None:
                    events.append(ExplosionEvent(
                        pos=(fx, fy), radius=radius, kind="door_explosive"))
