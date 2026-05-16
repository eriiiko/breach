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

Note: ``random`` is process-global today. The Phase 2 Simulation facade
plumbs a dedicated :class:`numpy.random.Generator` through these
functions so AI rollouts are reproducible. Flagged in the migration
plan as one of three nondeterminism sites.
"""
from __future__ import annotations

import math
import random

from config import CFG
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
def apply_blast_damage(units, fx, fy, radius, max_damage):
    """Damage every unit within ``radius`` of (fx, fy), with linear falloff.

    Units below ``CFG.combat.blast_damage_threshold`` damage take none
    (prevents chip damage at the edge of distant blasts). Marks the
    unit dead if HP <= 0. Does NOT set ``killed_by_zombie`` — explosion
    deaths don't convert.
    """
    for u in units:
        if not u.alive:
            continue
        uc_fx = u.center_fx()
        uc_fy = u.center_fy()
        dist = math.sqrt((uc_fx - fx) ** 2 + (uc_fy - fy) ** 2)
        if dist <= radius:
            falloff = 1.0 - (dist / radius)
            damage = int(max_damage * falloff)
            if damage >= CFG.combat.blast_damage_threshold:
                u.hp -= damage
                if u.hp <= 0:
                    u.alive = False


# ---------------------------------------------------------------------------
# Shooting (fire orders + auto-fire)
# ---------------------------------------------------------------------------
def process_shooting(gmap, units, tick, shots, real_time):
    """Run one tick of shooting for every player unit with a fire order
    (or auto-fire during Move & Attack).

    Lifted from ``game.py:_process_shooting``. ``shots`` is a list the
    caller owns; new :class:`Shot` tracer events are appended to it.
    ``real_time`` is the wall-clock seconds since the round started
    (used as the Shot's spawn time for tracer fade-out).
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
                    auto_fire(gmap, units, u, tick, shots, real_time)
                    break
            continue

        # Burst cadence gate.
        if tick - u.last_fire_tick < burst_interval:
            continue

        target_fx = fire_order.target_fx
        target_fy = fire_order.target_fy
        uc_fx = u.center_fx()
        uc_fy = u.center_fy()

        # Range check.
        dist = math.sqrt((uc_fx - target_fx) ** 2 + (uc_fy - target_fy) ** 2)
        if dist > CFG.weapons.rifle.range_tiles:
            continue

        # LOS check.
        if not gmap.has_los(uc_fy, uc_fx, target_fy, target_fx):
            continue

        fire_burst(gmap, units, u, uc_fx, uc_fy, target_fx, target_fy,
                   tick, shots, real_time)
        u.last_fire_tick = tick


def auto_fire(gmap, units, u, tick, shots, real_time):
    """Find the nearest visible enemy within rifle range and fire a burst.

    Lifted from ``game.py:_auto_fire``. Skipped if still within the burst
    cooldown.
    """
    burst_interval = CFG.weapons.rifle.burst_interval_ticks
    if tick - u.last_fire_tick < burst_interval:
        return

    uc_fx = u.center_fx()
    uc_fy = u.center_fy()
    best_dist = float('inf')
    best_enemy = None

    for e in units:
        if e.team == u.team or not e.alive:
            continue
        ec_fx = e.center_fx()
        ec_fy = e.center_fy()
        dist = math.sqrt((uc_fx - ec_fx) ** 2 + (uc_fy - ec_fy) ** 2)
        if dist <= CFG.weapons.rifle.range_tiles and dist < best_dist:
            if gmap.has_los(uc_fy, uc_fx, ec_fy, ec_fx):
                best_dist = dist
                best_enemy = e

    if best_enemy:
        fire_burst(gmap, units, u, uc_fx, uc_fy,
                   best_enemy.center_fx(), best_enemy.center_fy(),
                   tick, shots, real_time)
        u.last_fire_tick = tick


def fire_burst(gmap, units, shooter, fx1, fy1, fx2, fy2, tick, shots, real_time):
    """Fire a burst of bullets from (fx1, fy1) toward (fx2, fy2).

    Lifted from ``game.py:_fire_burst``. Each bullet picks a random
    angle within the rifle's cone, marches tile-by-tile, stops on
    wall hit or unit hit. Zombies take damage scaled by
    ``CFG.zombie.bullet_damage_multiplier``. Every bullet appends one
    :class:`Shot` tracer to ``shots`` regardless of hit/miss.
    """
    co = CFG.display.coarse
    cone = math.radians(CFG.weapons.rifle.cone_half_angle_degrees)
    n_bullets = CFG.weapons.rifle.bullets_per_burst
    dmg = CFG.weapons.rifle.damage_per_bullet
    base_angle = math.atan2(fy2 - fy1, fx2 - fx1)
    h, w = gmap.material.shape

    for _ in range(n_bullets):
        angle = base_angle + random.uniform(-cone, cone)
        rx, ry = float(fx1), float(fy1)
        hit_unit = None
        for _step in range(int(CFG.weapons.rifle.range_tiles)):
            rx += math.cos(angle)
            ry += math.sin(angle)
            ix, iy = int(rx), int(ry)

            if 0 <= iy < h and 0 <= ix < w:
                if gmap.is_wall[iy, ix]:
                    break
            else:
                break

            for e in units:
                if e is shooter or not e.alive:
                    continue
                if e.fx <= ix < e.fx + co and e.fy <= iy < e.fy + co:
                    hit_unit = e
                    break
            if hit_unit:
                break

        if hit_unit:
            actual_dmg = dmg
            if hit_unit.team == 1:  # zombie
                actual_dmg = int(dmg * CFG.zombie.bullet_damage_multiplier)
            hit_unit.hp -= actual_dmg
            if hit_unit.hp <= 0:
                hit_unit.alive = False

        shots.append(Shot(fx1, fy1, rx, ry, real_time))


# ---------------------------------------------------------------------------
# Door explosives (scheduled detonations at phase boundaries)
# ---------------------------------------------------------------------------
def process_door_explosives(gmap, units, slot):
    """Detonate every door-explosive order scheduled for ``slot``.

    Lifted from ``game.py:_process_door_explosives``. Called three times
    per round (start P1, between phases, end P2). Skips zombies — only
    player-issued orders detonate.
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
                apply_explosion(gmap, fy, fx, radius, pressure, wall_dmg)
                apply_blast_damage(units, fx, fy, radius, unit_dmg)
                add_explosion_smoke(gmap, fy, fx, radius)
