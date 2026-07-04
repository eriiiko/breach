"""Zombie AI: activation, chain propagation, target selection, melee, A* movement, conversion.

Lifted from ``game.py``:

- :func:`update_zombies_tick`  (game.py:1867-1979 — ``Game._update_zombies_tick``)
- :func:`convert_marines_to_zombies`  (extracted from ``Game._end_execution`` lines 2019-2028)

The legacy code monkey-patched ``zombie_speed_override`` after construction;
the lifted version reads ``z.speed_ticks_per_tile`` directly (the field
was promoted onto :class:`simulation.unit.Unit` in Phase 1 step 2).

Pathfinding is provided by :mod:`pathfinding.astar`. If pathfinding fails
to import, the helper falls back to "stand still" (matches legacy
``HAS_PATHFINDING=False`` branch). Temporal A* exists in pathfinding.py
but is explicitly NOT wired up here — see migration plan anti-goals.

The chain-activation loop is ``while changed:`` over a nested O(N^2)
sweep. Correct but O(N^3) in the worst case. Fine for current squad
sizes; flagged in docs/architecture.md §9 as a refactoring target.
"""
from __future__ import annotations

import math

from config import CFG
from simulation import unit_fixed
from simulation.movement import FootprintSamples, default_speed

try:
    from pathfinding import astar
    HAS_PATHFINDING = True
except ImportError:
    HAS_PATHFINDING = False


# ---------------------------------------------------------------------------
# Per-tick AI step
# ---------------------------------------------------------------------------
def update_zombies_tick(gmap, units, tick):
    """Run one tick of zombie AI for every active zombie on the map.

    Three sequential passes:
      1. Trigger detection — inactive zombies with LOS to any living
         player within ``CFG.zombie.trigger_radius`` activate.
      2. Chain propagation — activated zombies wake up nearby inactive
         zombies within ``CFG.zombie.propagation_radius`` (BFS until
         the set stabilises).
      3. Per-zombie movement + combat — pick nearest living player,
         melee if adjacent (with cooldown), otherwise A* toward target
         at ``z.speed_ticks_per_tile`` cadence. Repath every 5 steps.

    Mutates unit HP / alive / killed_by_zombie / zombie_* fields.
    """
    players = [u for u in units if u.team == 0 and u.alive]
    if not players:
        return

    zombies = [u for u in units if u.team == 1 and u.alive]

    # ---- 1. Trigger detection (LOS + radius) ----
    for z in zombies:
        if z.zombie_activated:
            continue
        zc_fx = z.center_tile_x()
        zc_fy = z.center_tile_y()
        for p in players:
            pc_fx = p.center_tile_x()
            pc_fy = p.center_tile_y()
            dist = math.sqrt((zc_fx - pc_fx) ** 2 + (zc_fy - pc_fy) ** 2)
            if dist < CFG.zombie.trigger_radius:
                if gmap.has_los(zc_fy, zc_fx, pc_fy, pc_fx):
                    z.zombie_activated = True
                    break

    # ---- 2. Chain activation: BFS until no new activations ----
    changed = True
    while changed:
        changed = False
        for z in zombies:
            if not z.zombie_activated:
                continue
            for z2 in zombies:
                if z2.zombie_activated or z2 is z:
                    continue
                dist = math.sqrt(
                    (z.center_tile_x() - z2.center_tile_x()) ** 2 +
                    (z.center_tile_y() - z2.center_tile_y()) ** 2)
                if dist < CFG.zombie.propagation_radius:
                    z2.zombie_activated = True
                    changed = True

    # ---- 3. Movement + combat for activated zombies ----
    h, w = gmap.material.shape
    for z in zombies:
        if not z.zombie_activated:
            continue

        # Nearest living player by Euclidean distance.
        nearest = None
        nearest_dist = float('inf')
        zc_fx = z.center_tile_x()
        zc_fy = z.center_tile_y()
        for p in players:
            pc_fx = p.center_tile_x()
            pc_fy = p.center_tile_y()
            dist = math.sqrt((zc_fx - pc_fx) ** 2 + (zc_fy - pc_fy) ** 2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = p

        if not nearest:
            continue

        # Melee attack if adjacent (footprint + 1 tiles threshold).
        if nearest_dist <= z.footprint + 1:
            cooldown = CFG.zombie.attack_cooldown_ticks
            if tick - z.last_melee_tick >= cooldown:
                z.last_melee_tick = tick
                # Q2-lift: snap the applied delta to the Q16.16 grid (exact
                # pass-through for an integer melee_damage config value;
                # belt-and-suspenders like the combat.py damage sites).
                nearest.current_hp -= unit_fixed.quantize_hp_delta(
                    CFG.zombie.melee_damage)
                if nearest.current_hp <= 0:
                    nearest.alive = False
                    nearest.killed_by_zombie = True
            continue

        # Movement: one tile every (terrain-scaled) speed_ticks_per_tile ticks.
        # Terrain cadence (mobility design §4.1): the §4 area-average mobility
        # over the zombie's CURRENT footprint scales its base cadence into the
        # ticks this step costs — a zombie clambering through furniture is 2.5x
        # slower. A* stays speed-blind; the multiplier composes here only.
        samples = FootprintSamples(
            mobility=gmap.footprint_mobility(z.tile_y, z.tile_x, z.footprint))
        step_ticks = default_speed(samples, z.speed_ticks_per_tile)
        z.zombie_move_accumulator += 1
        if z.zombie_move_accumulator >= step_ticks:
            z.zombie_move_accumulator = 0

            # Repath if no path, finished, or every 5 steps to track movement.
            needs_repath = (not z.zombie_path
                            or z.zombie_path_idx >= len(z.zombie_path)
                            or z.zombie_path_idx % 5 == 0)
            if needs_repath:
                if HAS_PATHFINDING:
                    def is_blocked(x, y, _gmap=gmap, _fp=z.footprint):
                        return not _gmap.is_passable_block(y, x, _fp)
                    z.zombie_path = astar(z.tile_x, z.tile_y,
                                          nearest.tile_x, nearest.tile_y,
                                          is_blocked, w, h)
                    z.zombie_path_idx = 1  # skip current position
                else:
                    z.zombie_path = []
                    z.zombie_path_idx = 0

            # Step along the path if any.
            if z.zombie_path and z.zombie_path_idx < len(z.zombie_path):
                next_x, next_y = z.zombie_path[z.zombie_path_idx]
                # Re-check passability (walls may have changed).
                if gmap.is_passable_block(next_y, next_x, z.footprint):
                    z.face_towards(float(next_x), float(next_y))
                    z.x = float(next_x)
                    z.y = float(next_y)
                    z.zombie_path_idx += 1
                else:
                    # Blocked: drop the stale path and try again next tick.
                    z.zombie_path = []
                    z.zombie_path_idx = 0


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
        if u.team == 0 and not u.alive and u.killed_by_zombie:
            u.team = 1
            u.is_zombie = True
            u.alive = True
            u.current_hp = float(CFG.zombie.hp)
            u.zombie_activated = True
            u.killed_by_zombie = False
            u.name = f"Z-{u.name}"
