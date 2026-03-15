"""
Breach — Tactical Squad Game Prototype (v2)
Two-phase simultaneous turns, tick-based execution, A* pathfinding.

Controls:
  PLANNING PHASE:
    Click unit to select it
    1 = Move & Attack mode
    2 = Move with Cover mode
    3 = Sprint mode
    G = Grenade mode (click target, scroll to set fuse timer)
    B = Plant Explosive mode (click adjacent door)
    F = Fire order (click target tile/enemy)
    Right-click = place waypoint for selected unit
    Backspace = undo last order
    Escape = deselect unit
    Space / Enter = Execute round

  EXECUTION PHASE:
    Plays out automatically (2 phases per round), then returns to planning.
    +/- = adjust playback speed

  F5 = Hot-reload config.toml
"""

import pygame
import numpy as np
import math
import os
import sys

from config import CFG

# Try to import pathfinding (may not exist yet during development)
try:
    from pathfinding import astar, temporal_astar, ReservationTable
    HAS_PATHFINDING = True
except ImportError:
    HAS_PATHFINDING = False

# ---------------------------------------------------------------------------
# Convenience aliases from config (updated on reload)
# ---------------------------------------------------------------------------
def _cfg():
    """Return frequently used config values. Call after any reload."""
    c = CFG
    return {
        'FINE_TILE': c.display.fine_tile_px,
        'COARSE': c.display.coarse,
        'COARSE_PX': c.display.coarse_px,
        'MAP_W': c.display.map_w,
        'MAP_H': c.display.map_h,
        'FINE_W': c.display.fine_w,
        'FINE_H': c.display.fine_h,
        'TICKS_PER_PHASE': c.clock.ticks_per_phase,
        'TICKS_PER_ROUND': c.clock.ticks_per_round,
        'TICKS_PER_SEC': c.clock.ticks_per_second,
        'PHASES_PER_ROUND': c.clock.phases_per_round,
        'AP_PER_PHASE': c.clock.ap_per_phase,
    }


# ---------------------------------------------------------------------------
# Constants that don't belong in config (internal IDs, colors)
# ---------------------------------------------------------------------------
# Game states
STATE_PLANNING = 0
STATE_EXECUTING = 1

# Order types
ORDER_MOVE_ATTACK = 0
ORDER_MOVE_COVER = 1
ORDER_SPRINT = 2
ORDER_GRENADE = 3
ORDER_EXPLOSIVE = 4
ORDER_FIRE = 5

ORDER_NAMES = {
    ORDER_MOVE_ATTACK: "Move & Attack",
    ORDER_MOVE_COVER: "Move w/ Cover",
    ORDER_SPRINT: "Sprint",
    ORDER_GRENADE: "Grenade",
    ORDER_EXPLOSIVE: "Explosive",
    ORDER_FIRE: "Fire",
}

# Colors
COL_BG = (10, 10, 15)
COL_FLOOR = (30, 35, 45)
COL_WALL_HULL = (80, 80, 90)
COL_WALL_WOOD = (70, 50, 30)
COL_GRID = (25, 30, 40)
COL_SELECT = (0, 200, 255)
COL_MOVE_ATTACK = (255, 100, 100)
COL_MOVE_COVER = (100, 200, 100)
COL_SPRINT = (100, 150, 255)
COL_GRENADE_TARGET = (255, 50, 0)
COL_EXPLOSIVE_TARGET = (255, 150, 0)
COL_FIRE = (255, 255, 100)
COL_UI_BG = (20, 20, 30)
COL_UI_TEXT = (200, 200, 210)
COL_UI_HIGHLIGHT = (0, 200, 255)
COL_TIMELINE_BG = (40, 40, 55)
COL_TIMELINE_TICK = (80, 80, 100)
COL_ZOMBIE = (180, 40, 40)
COL_PHASE_DIVIDER = (255, 200, 0)

ORDER_COLORS = {
    ORDER_MOVE_ATTACK: COL_MOVE_ATTACK,
    ORDER_MOVE_COVER: COL_MOVE_COVER,
    ORDER_SPRINT: COL_SPRINT,
    ORDER_GRENADE: COL_GRENADE_TARGET,
    ORDER_EXPLOSIVE: COL_EXPLOSIVE_TARGET,
    ORDER_FIRE: COL_FIRE,
}

# Detonation slots for door explosives
DET_START_PHASE1 = 0
DET_BETWEEN_PHASES = 1
DET_END_PHASE2 = 2
DET_SLOT_NAMES = {
    DET_START_PHASE1: "Start P1",
    DET_BETWEEN_PHASES: "Between P1/P2",
    DET_END_PHASE2: "End P2",
}

# Material IDs
MAT_AIR = 0
MAT_HULL = 1
MAT_WOOD = 2
MAT_DOOR = 3

MATERIAL_COLORS = {
    MAT_AIR: COL_FLOOR,
    MAT_HULL: COL_WALL_HULL,
    MAT_WOOD: COL_WALL_WOOD,
    MAT_DOOR: (50, 70, 50),
}


def ticks_per_tile(order_type):
    """Get movement speed (ticks per fine tile) for a movement order type."""
    if order_type == ORDER_MOVE_ATTACK:
        return CFG.movement.marine_attack_ticks_per_tile
    elif order_type == ORDER_MOVE_COVER:
        return CFG.movement.marine_cover_ticks_per_tile
    elif order_type == ORDER_SPRINT:
        return CFG.movement.marine_sprint_ticks_per_tile
    return CFG.movement.marine_attack_ticks_per_tile


def diagonal_distance(dx, dy):
    """Compute distance using alternating 1-2 diagonal cost."""
    dx, dy = abs(dx), abs(dy)
    diag = min(dx, dy)
    straight = max(dx, dy) - diag
    full_pairs = diag // 2
    remainder = diag % 2
    return straight + full_pairs * 3 + remainder


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
class GameMap:
    """2D grid map with fine-tile resolution."""

    def __init__(self):
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        co = CFG.display.coarse

        self.material = np.zeros((fh, fw), dtype=np.int8)
        self.wall_hp = np.zeros((fh, fw), dtype=np.float32)
        self.is_wall = np.zeros((fh, fw), dtype=bool)
        self.is_vacuum = np.zeros((fh, fw), dtype=bool)
        self.flammable = np.zeros((fh, fw), dtype=bool)
        self.atmosphere = np.ones((fh, fw), dtype=np.float32)
        self.wave_p = np.zeros((fh, fw), dtype=np.float32)  # wave pressure deviation
        self.wave_v = np.zeros((fh, fw), dtype=np.float32)  # wave velocity (dp/dt)
        self.wave_source = np.zeros((fh, fw), dtype=np.float32)  # pressure source (fed over time)
        self.smoke = np.zeros((fh, fw), dtype=np.float32)
        self.unit_absorb = np.zeros((fh, fw), dtype=np.float32)

        self._build_ship()
        self._update_caches()

    def _build_ship(self):
        """Build a simple ship layout for testing."""
        co = CFG.display.coarse
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        m = self.material
        m[:] = MAT_HULL

        hull_t = co
        m[hull_t:fh - hull_t, hull_t:fw - hull_t] = MAT_AIR

        # Vertical wall at coarse x=15
        wall_x = 15 * co
        for y in range(hull_t, fh - hull_t):
            m[y, wall_x] = MAT_WOOD

        # Doors in vertical wall
        for door_cy in [10, 18]:
            door_y = door_cy * co
            for dy in range(co):
                m[door_y + dy, wall_x] = MAT_DOOR

        # Horizontal wall in left section at coarse y=8
        wall_y = 8 * co
        for x in range(hull_t, wall_x):
            m[wall_y, x] = MAT_WOOD
        door_x = 8 * co
        for dx in range(co):
            m[wall_y, door_x + dx] = MAT_DOOR

        # Horizontal wall in right section at coarse y=12
        wall_y2 = 12 * co
        for x in range(wall_x + 1, fw - hull_t):
            m[wall_y2, x] = MAT_WOOD
        door_x2 = 25 * co
        for dx in range(co):
            m[wall_y2, door_x2 + dx] = MAT_DOOR

        # Room in top-right
        wall_y3 = 6 * co
        for x in range(wall_x + 1, 30 * co):
            m[wall_y3, x] = MAT_WOOD
        door_x3 = 22 * co
        for dx in range(co):
            m[wall_y3, door_x3 + dx] = MAT_DOOR

        # Vertical wall at coarse x=30
        wall_x2 = 30 * co
        for y in range(hull_t, wall_y2):
            m[y, wall_x2] = MAT_WOOD
        door_y3 = 4 * co
        for dy in range(co):
            m[door_y3 + dy, wall_x2] = MAT_DOOR

        # Ensure hull borders
        m[0:hull_t, :] = MAT_HULL
        m[fh - hull_t:, :] = MAT_HULL
        m[:, 0:hull_t] = MAT_HULL
        m[:, fw - hull_t:] = MAT_HULL

    def _update_caches(self):
        """Rebuild cached arrays from material grid."""
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        co = CFG.display.coarse
        m = self.material

        self.is_wall = np.isin(m, [MAT_HULL, MAT_WOOD])
        self.flammable = (m == MAT_WOOD)
        self.wall_hp = np.zeros((fh, fw), dtype=np.float32)

        # Set HP from config materials
        mat_props = {
            MAT_AIR: CFG.materials.air,
            MAT_HULL: CFG.materials.hull,
            MAT_WOOD: CFG.materials.wood,
            MAT_DOOR: CFG.materials.door,
        }
        for mat_id, props in mat_props.items():
            hp = props[0]
            if hp > 0:
                self.wall_hp[m == mat_id] = hp

        self.is_vacuum = np.zeros_like(self.is_wall)
        self.is_vacuum[0:co, :] = True
        self.is_vacuum[fh - co:, :] = True
        self.is_vacuum[:, 0:co] = True
        self.is_vacuum[:, fw - co:] = True

        self.atmosphere = np.where(
            self.is_wall | self.is_vacuum, 0.0, 1.0
        ).astype(np.float32)

    def stamp_units(self, units):
        """Stamp living unit positions onto the unit_absorb array."""
        co = CFG.display.coarse
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        self.unit_absorb[:] = 0.0
        for u in units:
            if not u.alive:
                continue
            uy, ux = u.fy, u.fx
            y1 = max(0, uy)
            y2 = min(fh, uy + co)
            x1 = max(0, ux)
            x2 = min(fw, ux + co)
            self.unit_absorb[y1:y2, x1:x2] = CFG.combat.unit_absorption

    def is_passable(self, fy, fx):
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        if fy < 0 or fy >= fh or fx < 0 or fx >= fw:
            return False
        return self.material[fy, fx] in (MAT_AIR, MAT_DOOR)

    def is_passable_block(self, fy, fx):
        """Check if a 3x3 fine-tile block (unit footprint) is fully passable."""
        co = CFG.display.coarse
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        if fy < 0 or fx < 0 or fy + co > fh or fx + co > fw:
            return False
        block = self.material[fy:fy + co, fx:fx + co]
        return np.all((block == MAT_AIR) | (block == MAT_DOOR))

    def has_los(self, fy1, fx1, fy2, fx2):
        """Bresenham line-of-sight check."""
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        dx = abs(fx2 - fx1)
        dy = abs(fy2 - fy1)
        sx = 1 if fx1 < fx2 else -1
        sy = 1 if fy1 < fy2 else -1
        err = dx - dy
        x, y = fx1, fy1
        while True:
            if x == fx2 and y == fy2:
                return True
            if 0 <= y < fh and 0 <= x < fw and self.is_wall[y, x]:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def destroy_wall(self, fy, fx):
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        co = CFG.display.coarse
        if 0 <= fy < fh and 0 <= fx < fw:
            was_hull = (self.material[fy, fx] == MAT_HULL)
            if self.material[fy, fx] in (MAT_HULL, MAT_WOOD, MAT_DOOR):
                self.material[fy, fx] = MAT_AIR
                self.wall_hp[fy, fx] = 0
                self.is_wall[fy, fx] = False
                self.flammable[fy, fx] = False
                # Hull breach: tile becomes vacuum-adjacent, starts at 0 atm
                # Interior wall: gets some atmosphere from neighbors
                if was_hull:
                    # Check if this tile is on the map edge (true hull breach)
                    if fy < co or fy >= fh - co or fx < co or fx >= fw - co:
                        self.is_vacuum[fy, fx] = True
                        self.atmosphere[fy, fx] = 0.0
                    else:
                        self.atmosphere[fy, fx] = 0.3
                else:
                    self.atmosphere[fy, fx] = 0.5


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
class Physics:

    @staticmethod
    def compute_laplacian(p, wall, unit_absorb=None):
        up = np.roll(p, 1, axis=0)
        down = np.roll(p, -1, axis=0)
        left = np.roll(p, 1, axis=1)
        right = np.roll(p, -1, axis=1)
        wall_up = np.roll(wall, 1, axis=0)
        wall_down = np.roll(wall, -1, axis=0)
        wall_left = np.roll(wall, 1, axis=1)
        wall_right = np.roll(wall, -1, axis=1)
        up = np.where(wall_up, p, up)
        down = np.where(wall_down, p, down)
        left = np.where(wall_left, p, left)
        right = np.where(wall_right, p, right)
        if unit_absorb is not None:
            refl = CFG.combat.unit_reflectivity
            abs_up = np.roll(unit_absorb, 1, axis=0)
            abs_down = np.roll(unit_absorb, -1, axis=0)
            abs_left = np.roll(unit_absorb, 1, axis=1)
            abs_right = np.roll(unit_absorb, -1, axis=1)
            up = np.where(abs_up > 0, 1.0 + (up - 1.0) * refl, up)
            down = np.where(abs_down > 0, 1.0 + (down - 1.0) * refl, down)
            left = np.where(abs_left > 0, 1.0 + (left - 1.0) * refl, left)
            right = np.where(abs_right > 0, 1.0 + (right - 1.0) * refl, right)
        return up + down + left + right - 4.0 * p

    @staticmethod
    def step_atmosphere(gmap, dt):
        lap = Physics.compute_laplacian(gmap.atmosphere, gmap.is_wall,
                                        gmap.unit_absorb)
        gmap.atmosphere += CFG.physics.d_atm * dt * lap
        gmap.atmosphere[gmap.is_wall] = 0.0
        gmap.atmosphere[gmap.is_vacuum] = 0.0
        absorb_mask = gmap.unit_absorb > 0
        excess = gmap.atmosphere[absorb_mask] - 1.0
        excess[excess > 0] *= (1.0 - gmap.unit_absorb[absorb_mask][excess > 0])
        gmap.atmosphere[absorb_mask] = 1.0 + excess
        np.clip(gmap.atmosphere, 0.0, 20.0, out=gmap.atmosphere)

    @staticmethod
    def step_smoke(gmap, dt):
        lap = Physics.compute_laplacian(gmap.smoke, gmap.is_wall)
        gmap.smoke += CFG.physics.d_smoke * dt * lap

        # Smoke gradient (shared by both advection sources)
        ds_dy = (np.roll(gmap.smoke, -1, axis=0) -
                 np.roll(gmap.smoke, 1, axis=0)) / 2.0
        ds_dx = (np.roll(gmap.smoke, -1, axis=1) -
                 np.roll(gmap.smoke, 1, axis=1)) / 2.0

        # Advection by atmosphere gradient (sustained wind, e.g. hull breach)
        a_grad_y = (np.roll(gmap.atmosphere, -1, axis=0) -
                    np.roll(gmap.atmosphere, 1, axis=0)) / 2.0
        a_grad_x = (np.roll(gmap.atmosphere, -1, axis=1) -
                    np.roll(gmap.atmosphere, 1, axis=1)) / 2.0
        gmap.smoke += CFG.physics.advection_rate * dt * (
            a_grad_x * ds_dx + a_grad_y * ds_dy)

        # Advection by wave pressure gradient (shockwave pushes smoke hard)
        w_grad_y = (np.roll(gmap.wave_p, -1, axis=0) -
                    np.roll(gmap.wave_p, 1, axis=0)) / 2.0
        w_grad_x = (np.roll(gmap.wave_p, -1, axis=1) -
                    np.roll(gmap.wave_p, 1, axis=1)) / 2.0
        gmap.smoke += 80.0 * dt * (w_grad_x * ds_dx + w_grad_y * ds_dy)

        gmap.smoke[gmap.is_wall] = 0.0
        gmap.smoke[gmap.is_vacuum] = 0.0
        np.clip(gmap.smoke, 0.0, 1.0, out=gmap.smoke)

    @staticmethod
    def apply_explosion(gmap, fy, fx, radius, pressure, wall_damage):
        """Apply explosion: damage walls, deposit wave + atmosphere pressure."""
        fh = CFG.display.fine_h
        fw = CFG.display.fine_w
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < fh and 0 <= nx < fw:
                    dist = math.sqrt(dy * dy + dx * dx)
                    if dist <= radius:
                        falloff = 1.0 - (dist / radius)
                        # Damage ALL wall types (including hull)
                        if gmap.material[ny, nx] in (MAT_HULL, MAT_WOOD, MAT_DOOR):
                            gmap.wall_hp[ny, nx] -= wall_damage * falloff
                            if gmap.wall_hp[ny, nx] <= 0:
                                gmap.destroy_wall(ny, nx)
                        if not gmap.is_wall[ny, nx] and not gmap.is_vacuum[ny, nx]:
                            # Feed pressure source (delivered over multiple substeps)
                            gmap.wave_source[ny, nx] += pressure * falloff
                            # Also deposit into atmosphere (creates sustained wind)
                            gmap.atmosphere[ny, nx] += pressure * falloff * 0.3
                        if dist <= radius * 0.4:
                            gmap.smoke[ny, nx] = 0.0

    # Wave parameters — all in physical units (per second)
    WAVE_C = 300.0          # wave speed in tiles/s (100 m/s, tiles are 1/3 m)
    WAVE_DAMPING = 3.0      # velocity damping rate (1/s)
    WAVE_TRANSFER = 0.5     # wave->atmosphere transfer rate (1/s)
    SOURCE_FEED_RATE = 200.0  # how fast source deposits into wave_p (1/s)

    @staticmethod
    def step(gmap, sim_time=None):
        """Advance all physics by sim_time seconds.
        Each system auto-computes its substep count from its own stability dt.
        If sim_time is None, advances one game tick (1/ticks_per_second)."""
        if sim_time is None:
            sim_time = 1.0 / CFG.clock.ticks_per_second  # 83.3ms per game tick

        c = Physics.WAVE_C
        c_squared = c * c
        damping = Physics.WAVE_DAMPING
        transfer = Physics.WAVE_TRANSFER
        feed_rate = Physics.SOURCE_FEED_RATE

        # Each system's stable dt
        dt_wave = 0.65 / c
        dt_diff = 0.24 / max(CFG.physics.d_atm, 0.01)

        # How many substeps to cover sim_time
        n_wave = max(1, int(math.ceil(sim_time / dt_wave)))
        n_diff = max(1, int(math.ceil(sim_time / dt_diff)))
        # Smoke: one step per call, using full sim_time (no CFL issue)
        dt_smoke = sim_time

        # Stability checks (once at startup)
        if not hasattr(Physics, '_checked'):
            Physics._checked = True
            print(f"[physics] sim_time={sim_time*1000:.1f}ms per game tick")
            print(f"[physics] Wave: c={c:.1f} tiles/s, dt={dt_wave*1000:.2f}ms, {n_wave} substeps")
            print(f"[physics] Diffusion: D={CFG.physics.d_atm}, dt={dt_diff*1000:.2f}ms, {n_diff} substeps")

        # --- Wave substeps ---
        for _ in range(n_wave):
            if np.any(gmap.wave_source > 0.001):
                feed = gmap.wave_source * feed_rate * dt_wave
                feed = np.minimum(feed, gmap.wave_source)
                gmap.wave_p += feed
                gmap.wave_source -= feed

            up = np.roll(gmap.wave_p, 1, axis=0)
            down = np.roll(gmap.wave_p, -1, axis=0)
            left = np.roll(gmap.wave_p, 1, axis=1)
            right = np.roll(gmap.wave_p, -1, axis=1)
            wall_up = np.roll(gmap.is_wall, 1, axis=0)
            wall_down = np.roll(gmap.is_wall, -1, axis=0)
            wall_left = np.roll(gmap.is_wall, 1, axis=1)
            wall_right = np.roll(gmap.is_wall, -1, axis=1)
            up = np.where(wall_up, gmap.wave_p, up)
            down = np.where(wall_down, gmap.wave_p, down)
            left = np.where(wall_left, gmap.wave_p, left)
            right = np.where(wall_right, gmap.wave_p, right)
            lap = up + down + left + right - 4.0 * gmap.wave_p

            gmap.wave_v += (c_squared * lap - damping * gmap.wave_v) * dt_wave
            gmap.wave_p += gmap.wave_v * dt_wave
            gmap.wave_p[gmap.is_wall] = 0.0
            gmap.wave_p[gmap.is_vacuum] = 0.0

            gmap.atmosphere += gmap.wave_p * transfer * dt_wave
            gmap.atmosphere[gmap.is_wall] = 0.0
            gmap.atmosphere[gmap.is_vacuum] = 0.0
            np.clip(gmap.atmosphere, 0.0, 20.0, out=gmap.atmosphere)

        # --- Diffusion substeps ---
        for _ in range(n_diff):
            Physics.step_atmosphere(gmap, dt_diff)

        # --- Smoke (advection + diffusion, one step) ---
        Physics.step_smoke(gmap, dt_smoke)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
class Order:
    def __init__(self, order_type, target_fx, target_fy, phase,
                 grenade_fuse=None, det_slot=None):
        self.order_type = order_type
        self.target_fx = target_fx
        self.target_fy = target_fy
        self.phase = phase          # 0 = Phase 1, 1 = Phase 2
        self.grenade_fuse = grenade_fuse  # seconds (for grenades)
        self.det_slot = det_slot    # detonation slot (for door explosives)
        self.ap_cost = 1            # most actions cost 1 AP


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
class Unit:
    def __init__(self, name, cx, cy, team=0):
        co = CFG.display.coarse
        self.name = name
        self.team = team
        self.fx = cx * co
        self.fy = cy * co
        self.fxf = float(self.fx)
        self.fyf = float(self.fy)
        self.orders = []  # list of Order objects
        self.alive = True
        self.hp = CFG.marine.hp if team == 0 else CFG.zombie.hp
        self.max_hp = self.hp
        self.facing = "S"
        self.current_order_type = ORDER_MOVE_ATTACK
        self.has_grenade = CFG.marine.grenades if team == 0 else 0
        self.has_explosive = CFG.marine.explosives if team == 0 else 0

        # AP tracking per phase
        self.ap = [CFG.clock.ap_per_phase, CFG.clock.ap_per_phase]

        # Combat state
        self.last_fire_tick = -999
        self.fire_target = None  # (fx, fy) for active fire orders

        # Zombie state
        self.zombie_activated = False
        self.zombie_path = []  # A* path for zombie movement
        self.zombie_path_idx = 0
        self.zombie_move_accumulator = 0  # ticks since last tile move
        self.last_melee_tick = -999
        self.killed_by_zombie = False  # for conversion tracking

        # Movement path (computed by temporal A* or simple pathfinding)
        self.move_path = []  # list of (fx, fy) positions per tick
        self.path_tick_offset = 0  # tick at which path starts

    @property
    def cx(self):
        return self.fx // CFG.display.coarse

    @property
    def cy(self):
        return self.fy // CFG.display.coarse

    def center_fx(self):
        return self.fx + CFG.display.coarse // 2

    def center_fy(self):
        return self.fy + CFG.display.coarse // 2

    def get_center_px(self):
        co_px = CFG.display.coarse_px
        ft = CFG.display.fine_tile_px
        return (int(self.fxf * ft + co_px / 2),
                int(self.fyf * ft + co_px / 2))

    def clear_orders(self):
        self.orders = []
        self.ap = [CFG.clock.ap_per_phase, CFG.clock.ap_per_phase]

    def get_ap(self, phase):
        return self.ap[phase]

    def spend_ap(self, phase, cost=1):
        self.ap[phase] -= cost

    def get_planned_end_pos(self):
        """Get fine tile position after all movement orders."""
        for o in reversed(self.orders):
            if o.order_type in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
                return o.target_fx, o.target_fy
        return self.fx, self.fy

    def get_orders_for_phase(self, phase):
        return [o for o in self.orders if o.phase == phase]

    def has_move_order_in_phase(self, phase):
        return any(o.order_type in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT)
                   and o.phase == phase for o in self.orders)

    def get_fire_order_in_phase(self, phase):
        for o in self.orders:
            if o.order_type == ORDER_FIRE and o.phase == phase:
                return o
        return None


# ---------------------------------------------------------------------------
# Projectiles
# ---------------------------------------------------------------------------
class Projectile:
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
        """Calculate the tick at which this projectile detonates."""
        tps = CFG.clock.ticks_per_second
        return self.thrown_tick + int(self.fuse_seconds * tps)

    def update_position(self, current_tick):
        """Update position based on travel toward target."""
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
# Shot tracers (visual feedback)
# ---------------------------------------------------------------------------
class Shot:
    def __init__(self, fx1, fy1, fx2, fy2, time):
        self.fx1, self.fy1 = fx1, fy1
        self.fx2, self.fy2 = fx2, fy2
        self.time = time
        self.duration = CFG.combat.shot_tracer_duration


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.init()
        ft = CFG.display.fine_tile_px
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        self.panel_w = CFG.display.panel_width

        self.screen_w = fw * ft + self.panel_w
        self.screen_h = fh * ft
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("BREACH — Tactical Prototype v2")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 11)

        # Map
        self.gmap = GameMap()

        # Units
        self.units = [
            Unit("Alpha", 4, 4, team=0),
            Unit("Bravo", 6, 4, team=0),
            Unit("Charlie", 5, 6, team=0),
        ]

        # Zombies — mixed tiers for tension
        # Regular zombies (config HP)
        zombie_regular = [(20, 9), (22, 10), (24, 8), (26, 15)]
        for i, (zx, zy) in enumerate(zombie_regular):
            z = Unit(f"Z{i+1}", zx, zy, team=1)
            self.units.append(z)

        # Fast weak zombies (runners) — low HP, faster
        zombie_runners = [(21, 14), (28, 10), (34, 8)]
        for i, (zx, zy) in enumerate(zombie_runners):
            z = Unit(f"Zr{i+1}", zx, zy, team=1)
            z.hp = CFG.zombie.hp // 4  # 100 HP — fragile
            z.max_hp = z.hp
            z.zombie_speed_override = max(1, CFG.zombie.ticks_per_tile - 3)  # faster
            self.units.append(z)

        # Tank zombie (brute) — very high HP, slower
        z = Unit("BRUTE", 32, 5, team=1)
        z.hp = CFG.zombie.hp * 3  # 1200 HP — bullet sponge
        z.max_hp = z.hp
        z.zombie_speed_override = CFG.zombie.ticks_per_tile + 3  # slower
        self.units.append(z)

        self.selected_unit = None
        self.current_mode = ORDER_MOVE_ATTACK
        self.grenade_fuse = CFG.weapons.grenade.fuse_default_seconds
        self.det_slot = DET_START_PHASE1  # default explosive detonation slot

        # Game state
        self.state = STATE_PLANNING
        self.turn_number = 1
        self.planning_phase = 0  # which phase the player is currently planning (0 or 1)

        # Execution state
        self.exec_tick = 0          # current tick within the round (0 to ticks_per_round)
        self.exec_phase = 0         # current phase (0 or 1)
        self.exec_speed = 1.0       # playback speed multiplier
        self.exec_accumulator = 0.0 # sub-tick accumulator
        self.projectiles = []
        self.shots = []
        self.real_time = 0.0
        self.frame_times = []   # last N frame times for FPS display
        self.physics_ms = 0.0   # physics compute time in ms

        # Sprites
        self.sprites = {}
        sprite_dir = os.path.join(os.path.dirname(__file__), "art", "sprites", "marine")
        for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            path = os.path.join(sprite_dir, f"marine_{direction}.png")
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                self.sprites[direction] = img

    # ===================================================================
    # Main loop
    # ===================================================================
    def run(self):
        import time as _time
        running = True
        while running:
            frame_start = _time.perf_counter()
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F5:
                    CFG.reload()
                elif self.state == STATE_PLANNING:
                    self._handle_planning_event(event)
                elif self.state == STATE_EXECUTING:
                    self._handle_execution_event(event)
            if self.state == STATE_EXECUTING:
                self._update_execution(dt)
            self._draw()
            # Frame timing
            frame_ms = (_time.perf_counter() - frame_start) * 1000.0
            self.frame_times.append(frame_ms)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
            pygame.display.flip()
        pygame.quit()

    # ===================================================================
    # Planning
    # ===================================================================
    def _handle_planning_event(self, event):
        ft = CFG.display.fine_tile_px
        fw = CFG.display.fine_w

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.current_mode = ORDER_MOVE_ATTACK
            elif event.key == pygame.K_2:
                self.current_mode = ORDER_MOVE_COVER
            elif event.key == pygame.K_3:
                self.current_mode = ORDER_SPRINT
            elif event.key == pygame.K_g:
                self.current_mode = ORDER_GRENADE
            elif event.key == pygame.K_b:
                self.current_mode = ORDER_EXPLOSIVE
            elif event.key == pygame.K_f:
                self.current_mode = ORDER_FIRE
            elif event.key == pygame.K_TAB:
                # Toggle planning phase
                self.planning_phase = 1 - self.planning_phase
            elif event.key == pygame.K_BACKSPACE:
                if self.selected_unit and self.selected_unit.orders:
                    removed = self.selected_unit.orders.pop()
                    if removed.ap_cost > 0:
                        self.selected_unit.ap[removed.phase] += removed.ap_cost
                    # Refund inventory
                    if removed.order_type == ORDER_GRENADE:
                        self.selected_unit.has_grenade += 1
                    elif removed.order_type == ORDER_EXPLOSIVE:
                        self.selected_unit.has_explosive += 1
            elif event.key == pygame.K_ESCAPE:
                if self.selected_unit:
                    self.selected_unit = None
                else:
                    self.current_mode = ORDER_MOVE_ATTACK
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                self._start_execution()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if mx < fw * ft:
                if event.button == 1:
                    self._handle_map_left_click(mx, my)
                elif event.button == 3:
                    self._handle_map_right_click(mx, my)

        elif event.type == pygame.MOUSEWHEEL:
            if self.current_mode == ORDER_GRENADE:
                self.grenade_fuse = max(
                    CFG.weapons.grenade.fuse_min_seconds,
                    min(CFG.weapons.grenade.fuse_max_seconds,
                        self.grenade_fuse + event.y * 0.5))
            elif self.current_mode == ORDER_EXPLOSIVE:
                # Cycle through detonation slots
                self.det_slot = (self.det_slot + (1 if event.y > 0 else -1)) % 3

    def _handle_map_left_click(self, mx, my):
        ft = CFG.display.fine_tile_px
        co = CFG.display.coarse
        fx = mx // ft
        fy = my // ft

        # Try to select a player unit
        for u in self.units:
            if (u.alive and u.team == 0 and
                    u.fx <= fx < u.fx + co and u.fy <= fy < u.fy + co):
                self.selected_unit = u
                return

        if self.selected_unit:
            self._place_order(mx, my)

    def _handle_map_right_click(self, mx, my):
        if self.selected_unit:
            self._place_order(mx, my)

    def _place_order(self, mx, my):
        u = self.selected_unit
        if not u or not u.alive:
            return
        ft = CFG.display.fine_tile_px
        co = CFG.display.coarse
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        phase = self.planning_phase

        if self.current_mode in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
            # Movement doesn't cost AP. Multiple waypoints per phase allowed.
            fx = mx // ft - co // 2
            fy = my // ft - co // 2
            fx = max(0, min(fw - co, fx))
            fy = max(0, min(fh - co, fy))
            if self.gmap.is_passable_block(fy, fx):
                order = Order(self.current_mode, fx, fy, phase)
                order.ap_cost = 0  # movement is free
                u.orders.append(order)

        elif self.current_mode == ORDER_GRENADE:
            if u.get_ap(phase) < CFG.weapons.grenade.ap_cost:
                return
            if u.has_grenade <= 0:
                return
            fx = mx // ft
            fy = my // ft
            order = Order(ORDER_GRENADE, fx, fy, phase,
                          grenade_fuse=self.grenade_fuse)
            order.ap_cost = CFG.weapons.grenade.ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)
            u.has_grenade -= 1

        elif self.current_mode == ORDER_EXPLOSIVE:
            if u.get_ap(phase) < CFG.weapons.door_explosive.ap_cost:
                return
            if u.has_explosive <= 0:
                return
            fx = mx // ft
            fy = my // ft
            order = Order(ORDER_EXPLOSIVE, fx, fy, phase,
                          det_slot=self.det_slot)
            order.ap_cost = CFG.weapons.door_explosive.ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)
            u.has_explosive -= 1

        elif self.current_mode == ORDER_FIRE:
            if u.get_ap(phase) < CFG.weapons.rifle.ap_cost:
                return
            fx = mx // ft
            fy = my // ft
            order = Order(ORDER_FIRE, fx, fy, phase)
            order.ap_cost = CFG.weapons.rifle.ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)

    # ===================================================================
    # Execution
    # ===================================================================
    def _start_execution(self):
        self.state = STATE_EXECUTING
        self.exec_tick = 0
        self.exec_phase = 0
        self.exec_accumulator = 0.0
        self.real_time = 0.0
        self.projectiles = []
        self.shots = []

        # Prepare projectiles from grenade orders
        tps = CFG.clock.ticks_per_second
        tpp = CFG.clock.ticks_per_phase
        for u in self.units:
            if u.team != 0:
                continue
            for o in u.orders:
                if o.order_type == ORDER_GRENADE:
                    phase_start_tick = o.phase * tpp
                    co = CFG.display.coarse
                    proj = Projectile(
                        ORDER_GRENADE,
                        u.get_planned_end_pos()[0] + co // 2,
                        u.get_planned_end_pos()[1] + co // 2,
                        o.target_fx + 0.5,
                        o.target_fy + 0.5,
                        fuse_seconds=o.grenade_fuse,
                        thrown_tick=phase_start_tick,
                    )
                    self.projectiles.append(proj)

        # Build movement paths for player units
        self._compute_player_paths()

        # Stamp initial unit positions
        self.gmap.stamp_units(self.units)

        # Process door explosives that detonate at start of phase 1
        self._process_door_explosives(DET_START_PHASE1)

    def _compute_player_paths(self):
        """Compute movement paths for all player units using A* pathfinding.
        Paths follow walls, support multiple waypoints per phase, and avoid
        friendly unit collisions via the reservation table."""
        co = CFG.display.coarse
        tpp = CFG.clock.ticks_per_phase
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        gmap = self.gmap

        def is_blocked(x, y):
            return not gmap.is_passable_block(y, x)

        for u in self.units:
            if u.team != 0 or not u.alive:
                continue
            u.move_path = []
            u.path_tick_offset = 0
            current_x, current_y = u.fx, u.fy

            for phase in range(CFG.clock.phases_per_round):
                # Collect all move orders for this phase (waypoint chain)
                move_orders = [o for o in u.orders
                               if o.phase == phase and o.order_type in
                               (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT)]

                if move_orders:
                    # Build a tile-by-tile path through all waypoints using A*
                    tile_path = []
                    cx, cy = current_x, current_y
                    speed = ticks_per_tile(move_orders[0].order_type)

                    for mo in move_orders:
                        speed = ticks_per_tile(mo.order_type)
                        if HAS_PATHFINDING:
                            segment = astar(cx, cy, mo.target_fx, mo.target_fy,
                                            is_blocked, fw, fh)
                            if segment and len(segment) > 1:
                                tile_path.extend(segment[1:])  # skip start (already there)
                            elif not segment:
                                # No path found — stay put
                                pass
                        else:
                            # Fallback: direct move (old behavior)
                            tile_path.append((mo.target_fx, mo.target_fy))
                        cx, cy = mo.target_fx, mo.target_fy

                    # Convert tile path to per-tick positions
                    # Each tile takes 'speed' ticks to traverse
                    if tile_path:
                        tick_positions = []
                        prev_x, prev_y = float(current_x), float(current_y)
                        for tile_x, tile_y in tile_path:
                            # Interpolate over 'speed' ticks from prev to this tile
                            for st in range(speed):
                                frac = (st + 1) / speed
                                ix = prev_x + (tile_x - prev_x) * frac
                                iy = prev_y + (tile_y - prev_y) * frac
                                tick_positions.append((ix, iy))
                            prev_x, prev_y = float(tile_x), float(tile_y)

                        # Fill remaining phase ticks with final position
                        for _ in range(tpp - len(tick_positions)):
                            tick_positions.append((prev_x, prev_y))
                        # Truncate if path is longer than phase
                        u.move_path.extend(tick_positions[:tpp])
                        current_x = int(round(tick_positions[min(len(tick_positions), tpp) - 1][0]))
                        current_y = int(round(tick_positions[min(len(tick_positions), tpp) - 1][1]))
                    else:
                        for _ in range(tpp):
                            u.move_path.append((float(current_x), float(current_y)))
                else:
                    for _ in range(tpp):
                        u.move_path.append((float(current_x), float(current_y)))

    def _process_door_explosives(self, slot):
        """Detonate all door explosives scheduled for the given detonation slot."""
        for u in self.units:
            if u.team != 0:
                continue
            for o in u.orders:
                if o.order_type == ORDER_EXPLOSIVE and o.det_slot == slot:
                    fy, fx = o.target_fy, o.target_fx
                    radius = CFG.weapons.door_explosive.blast_radius
                    pressure = CFG.weapons.door_explosive.pressure
                    wall_dmg = CFG.weapons.door_explosive.wall_damage
                    Physics.apply_explosion(self.gmap, fy, fx, radius,
                                            pressure, wall_dmg)
                    # Damage units near the blast
                    self._apply_blast_damage(fx, fy, radius,
                                             CFG.weapons.door_explosive.unit_damage)
                    # Add smoke
                    self._add_explosion_smoke(fy, fx, radius)

    def _update_execution(self, dt):
        tpp = CFG.clock.ticks_per_phase
        tpr = CFG.clock.ticks_per_round
        tps = CFG.clock.ticks_per_second

        # Accumulate time and advance ticks
        self.real_time += dt
        self.exec_accumulator += dt * self.exec_speed * tps

        ticks_to_process = int(self.exec_accumulator)
        self.exec_accumulator -= ticks_to_process

        for _ in range(ticks_to_process):
            if self.exec_tick >= tpr:
                break
            self._process_tick()
            self.exec_tick += 1

            # Check for phase transition
            new_phase = self.exec_tick // tpp
            if new_phase != self.exec_phase and new_phase < CFG.clock.phases_per_round:
                # Phase boundary reached — process between-phase explosives
                if self.exec_phase == 0 and new_phase == 1:
                    self._process_door_explosives(DET_BETWEEN_PHASES)
                self.exec_phase = new_phase

        # Expire old shot tracers
        self.shots = [s for s in self.shots if
                      self.real_time - s.time < s.duration]

        # Check if execution is complete
        if self.exec_tick >= tpr:
            # Process end-of-phase-2 explosives
            self._process_door_explosives(DET_END_PHASE2)
            self._end_execution()

    def _process_tick(self):
        """Process a single game tick. This is the core simulation step."""
        tick = self.exec_tick
        tpp = CFG.clock.ticks_per_phase

        # 1. Update projectile positions and check detonations
        for proj in self.projectiles:
            if proj.detonated:
                continue
            proj.update_position(tick)
            if tick >= proj.get_detonate_tick():
                proj.detonated = True
                fx = int(proj.target_fx)
                fy = int(proj.target_fy)
                if proj.proj_type == ORDER_GRENADE:
                    radius = CFG.weapons.grenade.blast_radius
                    Physics.apply_explosion(
                        self.gmap, fy, fx, radius,
                        CFG.weapons.grenade.pressure,
                        CFG.weapons.grenade.wall_damage)
                    self._apply_blast_damage(fx, fy, radius,
                                             CFG.weapons.grenade.unit_damage)
                    self._add_explosion_smoke(fy, fx, radius)

        # 2. Update player unit positions from precomputed paths
        for u in self.units:
            if not u.alive or u.team != 0:
                continue
            path_idx = tick - u.path_tick_offset
            if 0 <= path_idx < len(u.move_path):
                px, py = u.move_path[path_idx]
                u.fxf = px
                u.fyf = py
                u.fx = int(round(px))
                u.fy = int(round(py))

        # 3. Process shooting (fire orders)
        self._process_shooting(tick)

        # 4. Update zombie AI
        self._update_zombies_tick(tick)

        # 5. Re-stamp unit positions
        self.gmap.stamp_units(self.units)

        # 6. Physics substep (timed)
        import time as _time
        t0 = _time.perf_counter()
        Physics.step(self.gmap)
        self.physics_ms = (_time.perf_counter() - t0) * 1000.0

    def _process_shooting(self, tick):
        """Handle fire orders for the current tick."""
        tpp = CFG.clock.ticks_per_phase
        phase = tick // tpp
        co = CFG.display.coarse
        burst_interval = CFG.weapons.rifle.burst_interval_ticks

        for u in self.units:
            if u.team != 0 or not u.alive:
                continue

            # Check for fire order in current phase
            fire_order = u.get_fire_order_in_phase(phase)
            if not fire_order:
                # Also check move & attack auto-fire
                for o in u.orders:
                    if (o.order_type == ORDER_MOVE_ATTACK and o.phase == phase):
                        # Auto-fire at nearest visible enemy
                        self._auto_fire(u, tick)
                        break
                continue

            # Burst fire at target
            if tick - u.last_fire_tick < burst_interval:
                continue

            target_fx = fire_order.target_fx
            target_fy = fire_order.target_fy
            uc_fx = u.center_fx()
            uc_fy = u.center_fy()

            # Check range
            dist = math.sqrt((uc_fx - target_fx)**2 + (uc_fy - target_fy)**2)
            if dist > CFG.weapons.rifle.range_tiles:
                continue

            # Check LOS
            if not self.gmap.has_los(uc_fy, uc_fx, target_fy, target_fx):
                continue

            # Fire burst
            self._fire_burst(u, uc_fx, uc_fy, target_fx, target_fy, tick)
            u.last_fire_tick = tick

    def _auto_fire(self, u, tick):
        """Auto-fire at nearest visible enemy during move & attack."""
        burst_interval = CFG.weapons.rifle.burst_interval_ticks
        if tick - u.last_fire_tick < burst_interval:
            return

        uc_fx = u.center_fx()
        uc_fy = u.center_fy()
        best_dist = float('inf')
        best_enemy = None

        for e in self.units:
            if e.team == u.team or not e.alive:
                continue
            ec_fx = e.center_fx()
            ec_fy = e.center_fy()
            dist = math.sqrt((uc_fx - ec_fx)**2 + (uc_fy - ec_fy)**2)
            if dist <= CFG.weapons.rifle.range_tiles and dist < best_dist:
                if self.gmap.has_los(uc_fy, uc_fx, ec_fy, ec_fx):
                    best_dist = dist
                    best_enemy = e

        if best_enemy:
            self._fire_burst(u, uc_fx, uc_fy,
                             best_enemy.center_fx(), best_enemy.center_fy(), tick)
            u.last_fire_tick = tick

    def _fire_burst(self, shooter, fx1, fy1, fx2, fy2, tick):
        """Fire a burst of bullets from (fx1,fy1) toward (fx2,fy2)."""
        import random
        co = CFG.display.coarse
        cone = math.radians(CFG.weapons.rifle.cone_half_angle_degrees)
        n_bullets = CFG.weapons.rifle.bullets_per_burst
        dmg = CFG.weapons.rifle.damage_per_bullet
        base_angle = math.atan2(fy2 - fy1, fx2 - fx1)

        for _ in range(n_bullets):
            angle = base_angle + random.uniform(-cone, cone)
            # Ray march along angle
            rx, ry = float(fx1), float(fy1)
            hit_unit = None
            for step in range(int(CFG.weapons.rifle.range_tiles)):
                rx += math.cos(angle)
                ry += math.sin(angle)
                ix, iy = int(rx), int(ry)

                # Check wall hit
                if (0 <= iy < CFG.display.fine_h and 0 <= ix < CFG.display.fine_w):
                    if self.gmap.is_wall[iy, ix]:
                        break
                else:
                    break

                # Check unit hit
                for e in self.units:
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

            # Visual tracer (from shooter to hit point or max range)
            self.shots.append(Shot(fx1, fy1, rx, ry, self.real_time))

    def _update_zombies_tick(self, tick):
        """Update zombie AI for a single tick."""
        co = CFG.display.coarse
        tpp = CFG.clock.ticks_per_phase
        phase = tick // tpp
        players = [u for u in self.units if u.team == 0 and u.alive]
        if not players:
            return

        zombies = [u for u in self.units if u.team == 1 and u.alive]

        # Trigger detection and chain activation
        for z in zombies:
            if z.zombie_activated:
                continue
            zc_fx = z.center_fx()
            zc_fy = z.center_fy()
            for p in players:
                pc_fx = p.center_fx()
                pc_fy = p.center_fy()
                dist = math.sqrt((zc_fx - pc_fx)**2 + (zc_fy - pc_fy)**2)
                if dist < CFG.zombie.trigger_radius:
                    if self.gmap.has_los(zc_fy, zc_fx, pc_fy, pc_fx):
                        z.zombie_activated = True
                        break

        # Chain activation: BFS from triggered zombies
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
                        (z.center_fx() - z2.center_fx())**2 +
                        (z.center_fy() - z2.center_fy())**2)
                    if dist < CFG.zombie.propagation_radius:
                        z2.zombie_activated = True
                        changed = True

        # Movement and combat for activated zombies
        for z in zombies:
            if not z.zombie_activated:
                continue

            # Find nearest player
            nearest = None
            nearest_dist = float('inf')
            zc_fx = z.center_fx()
            zc_fy = z.center_fy()
            for p in players:
                pc_fx = p.center_fx()
                pc_fy = p.center_fy()
                dist = math.sqrt((zc_fx - pc_fx)**2 + (zc_fy - pc_fy)**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = p

            if not nearest:
                continue

            # Melee attack if adjacent
            if nearest_dist <= co + 1:
                cooldown = CFG.zombie.attack_cooldown_ticks
                if tick - z.last_melee_tick >= cooldown:
                    z.last_melee_tick = tick
                    nearest.hp -= CFG.zombie.melee_damage
                    if nearest.hp <= 0:
                        nearest.alive = False
                        nearest.killed_by_zombie = True
                continue

            # Move toward nearest player using A* pathfinding
            z.zombie_move_accumulator += 1
            speed = getattr(z, 'zombie_speed_override', CFG.zombie.ticks_per_tile)
            if z.zombie_move_accumulator >= speed:
                z.zombie_move_accumulator = 0

                # Recompute path if: no path, finished path, or every 5 steps
                # (to track moving players)
                needs_repath = (not z.zombie_path or
                                z.zombie_path_idx >= len(z.zombie_path) or
                                z.zombie_path_idx % 5 == 0)
                if needs_repath:
                    if HAS_PATHFINDING:
                        def is_blocked(x, y):
                            return not self.gmap.is_passable_block(y, x)
                        fw = CFG.display.fine_w
                        fh = CFG.display.fine_h
                        z.zombie_path = astar(z.fx, z.fy, nearest.fx, nearest.fy,
                                              is_blocked, fw, fh)
                        z.zombie_path_idx = 1  # skip start position
                    else:
                        z.zombie_path = []
                        z.zombie_path_idx = 0

                # Follow path
                if z.zombie_path and z.zombie_path_idx < len(z.zombie_path):
                    next_x, next_y = z.zombie_path[z.zombie_path_idx]
                    # Check if tile is still passable (walls may have changed)
                    if self.gmap.is_passable_block(next_y, next_x):
                        z.fx = next_x
                        z.fy = next_y
                        z.fxf = float(next_x)
                        z.fyf = float(next_y)
                        z.zombie_path_idx += 1
                    else:
                        # Path blocked, recompute next tick
                        z.zombie_path = []
                        z.zombie_path_idx = 0

    def _apply_blast_damage(self, fx, fy, radius, max_damage):
        """Damage all units within blast radius."""
        co = CFG.display.coarse
        for u in self.units:
            if not u.alive:
                continue
            uc_fx = u.center_fx()
            uc_fy = u.center_fy()
            dist = math.sqrt((uc_fx - fx)**2 + (uc_fy - fy)**2)
            if dist <= radius:
                falloff = 1.0 - (dist / radius)
                damage = int(max_damage * falloff)
                if damage >= CFG.combat.blast_damage_threshold:
                    u.hp -= damage
                    if u.hp <= 0:
                        u.alive = False

    def _add_explosion_smoke(self, fy, fx, radius):
        """Add smoke from an explosion."""
        fh = CFG.display.fine_h
        fw = CFG.display.fine_w
        for ddy in range(-radius, radius + 1):
            for ddx in range(-radius, radius + 1):
                ny, nx = fy + ddy, fx + ddx
                if (0 <= ny < fh and 0 <= nx < fw and
                        not self.gmap.is_wall[ny, nx]):
                    dist = math.sqrt(ddy**2 + ddx**2)
                    if dist < radius:
                        self.gmap.smoke[ny, nx] = min(
                            1.0, self.gmap.smoke[ny, nx] + 0.8 * (1 - dist/radius))

    def _end_execution(self):
        """End execution, handle zombie conversion, return to planning."""
        self.state = STATE_PLANNING
        self.turn_number += 1

        # Zombie conversion: dead marines killed by zombies become zombies
        for u in self.units:
            if u.team == 0 and not u.alive and u.killed_by_zombie:
                u.team = 1
                u.alive = True
                u.hp = CFG.zombie.hp
                u.max_hp = CFG.zombie.hp
                u.zombie_activated = True
                u.killed_by_zombie = False
                u.name = f"Z-{u.name}"

        # Snap units and reset orders
        for u in self.units:
            u.fx = round(u.fxf)
            u.fy = round(u.fyf)
            u.fxf = float(u.fx)
            u.fyf = float(u.fy)
            u.clear_orders()
            u.move_path = []
            u.last_fire_tick = -999

        self.gmap.unit_absorb[:] = 0.0
        # Keep unexploded projectiles (grenades with long fuses)
        self.projectiles = [p for p in self.projectiles if not p.detonated]
        self.planning_phase = 0

    def _handle_execution_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.exec_speed = min(10.0, self.exec_speed + 0.5)
            elif event.key == pygame.K_MINUS:
                self.exec_speed = max(0.25, self.exec_speed - 0.25)

    # ===================================================================
    # Drawing
    # ===================================================================
    def _draw(self):
        self.screen.fill(COL_BG)
        self._draw_map()
        self._draw_atmosphere()
        self._draw_smoke()
        self._draw_orders()
        self._draw_projectiles()
        self._draw_units()
        self._draw_shots()
        self._draw_ui_panel()
        if self.state == STATE_PLANNING:
            self._draw_cursor_info()

    def _draw_map(self):
        ft = CFG.display.fine_tile_px
        co = CFG.display.coarse
        co_px = CFG.display.coarse_px
        mw = CFG.display.map_w
        mh = CFG.display.map_h
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h

        for cy in range(mh):
            for cx in range(mw):
                px = cx * co_px
                py = cy * co_px
                fy, fx = cy * co, cx * co
                mat = self.gmap.material[fy:fy+co, fx:fx+co]
                if np.any(mat == MAT_HULL):
                    color = COL_WALL_HULL
                elif np.any(mat == MAT_WOOD):
                    color = COL_WALL_WOOD
                elif np.any(mat == MAT_DOOR):
                    color = MATERIAL_COLORS[MAT_DOOR]
                else:
                    color = COL_FLOOR
                pygame.draw.rect(self.screen, color, (px, py, co_px, co_px))

        for x in range(0, fw * ft, co_px):
            pygame.draw.line(self.screen, COL_GRID, (x, 0), (x, fh * ft), 1)
        for y in range(0, fh * ft, co_px):
            pygame.draw.line(self.screen, COL_GRID, (0, y), (fw * ft, y), 1)

    def _draw_atmosphere(self):
        """Draw pressure field (atmosphere + wave) using numpy for speed.
        Fire color scheme: black -> red -> orange -> yellow -> white."""
        total = self.gmap.atmosphere + self.gmap.wave_p
        if total.max() - total.min() < 0.01:
            return
        ft = CFG.display.fine_tile_px
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        mask = ~(self.gmap.is_wall | self.gmap.is_vacuum)

        # RGBA array for the overlay
        rgba = np.zeros((fh, fw, 4), dtype=np.uint8)

        # Overpressure (fire color ramp)
        excess = np.clip(total - 1.0, 0.0, None)
        t = np.clip(excess / 5.0, 0.0, 1.0)
        over = mask & (excess > 0.02)

        if np.any(over):
            # Piecewise color: 0-0.25 red, 0.25-0.5 orange, 0.5-0.75 yellow, 0.75-1 white
            t_o = t[over]
            r = np.where(t_o < 0.25, (t_o / 0.25) * 255,
                np.where(t_o < 0.75, 255, 255)).astype(np.uint8)
            g = np.where(t_o < 0.25, 0,
                np.where(t_o < 0.5, ((t_o - 0.25) / 0.25) * 140,
                np.where(t_o < 0.75, 140 + ((t_o - 0.5) / 0.25) * 115, 255))).astype(np.uint8)
            b = np.where(t_o < 0.5, 0,
                np.where(t_o < 0.75, ((t_o - 0.5) / 0.25) * 80,
                80 + ((t_o - 0.75) / 0.25) * 175)).astype(np.uint8)
            a = np.clip(excess[over] * 80, 0, 255).astype(np.uint8)
            rgba[over, 0] = r
            rgba[over, 1] = g
            rgba[over, 2] = b
            rgba[over, 3] = a

        # Underpressure (blue-purple)
        under = mask & (total < 0.9)
        if np.any(under):
            deficit = 1.0 - total[under]
            rgba[under, 0] = np.clip(60 + 40 * deficit, 0, 255).astype(np.uint8)
            rgba[under, 1] = 40
            rgba[under, 2] = 255
            rgba[under, 3] = np.clip(deficit * 400, 0, 220).astype(np.uint8)

        # Convert to pygame surface
        overlay = pygame.image.frombuffer(rgba.tobytes(), (fw, fh), "RGBA").convert_alpha()
        scaled = pygame.transform.scale(overlay, (fw * ft, fh * ft))
        self.screen.blit(scaled, (0, 0))

    def _draw_smoke(self):
        smoke = self.gmap.smoke
        if smoke.max() < 0.01:
            return
        ft = CFG.display.fine_tile_px
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h

        # Numpy RGBA overlay at fine tile resolution
        rgba = np.zeros((fh, fw, 4), dtype=np.uint8)
        has_smoke = smoke > 0.01
        if np.any(has_smoke):
            rgba[has_smoke, 0] = 180
            rgba[has_smoke, 1] = 160
            rgba[has_smoke, 2] = 140
            rgba[has_smoke, 3] = np.clip(smoke[has_smoke] * 220, 0, 200).astype(np.uint8)

        overlay = pygame.image.frombuffer(rgba.tobytes(), (fw, fh), "RGBA").convert_alpha()
        scaled = pygame.transform.scale(overlay, (fw * ft, fh * ft))
        self.screen.blit(scaled, (0, 0))

    def _draw_units(self):
        ft = CFG.display.fine_tile_px
        co = CFG.display.coarse
        co_px = CFG.display.coarse_px

        for u in self.units:
            if not u.alive:
                continue
            px = int(u.fxf * ft)
            py = int(u.fyf * ft)

            if u == self.selected_unit and self.state == STATE_PLANNING:
                sel_rect = (px - 2, py - 2, co_px + 4, co_px + 4)
                pygame.draw.rect(self.screen, COL_SELECT, sel_rect, 2)

            if u.team == 1:
                # Zombie
                zombie_col = COL_ZOMBIE
                if u.name.startswith("Z-"):
                    # Converted marine — use a different shade
                    zombie_col = (200, 80, 80)
                pygame.draw.rect(self.screen, zombie_col,
                                 (px + 2, py + 2, co_px - 4, co_px - 4))
                # Activation indicator
                if u.zombie_activated:
                    pygame.draw.rect(self.screen, (255, 0, 0),
                                     (px, py, co_px, co_px), 1)
                # HP bar
                max_hp = u.max_hp
                if u.hp < max_hp:
                    bar_w = int((u.hp / max_hp) * (co_px - 4))
                    pygame.draw.rect(self.screen, (255, 0, 0),
                                     (px + 2, py + co_px - 4, co_px - 4, 3))
                    pygame.draw.rect(self.screen, (0, 255, 0),
                                     (px + 2, py + co_px - 4, max(0, bar_w), 3))
            else:
                # Marine
                sprite = self.sprites.get(u.facing, self.sprites.get("S"))
                if sprite:
                    scaled = pygame.transform.scale(sprite, (co_px, co_px))
                    self.screen.blit(scaled, (px, py))
                else:
                    pygame.draw.rect(self.screen, (0, 180, 0),
                                     (px + 2, py + 2, co_px - 4, co_px - 4))
                if u.hp < CFG.marine.hp:
                    bar_w = int((u.hp / CFG.marine.hp) * (co_px - 4))
                    pygame.draw.rect(self.screen, (255, 0, 0),
                                     (px + 2, py + co_px - 4, co_px - 4, 3))
                    pygame.draw.rect(self.screen, (0, 255, 0),
                                     (px + 2, py + co_px - 4, max(0, bar_w), 3))

            # Name label
            if u.team == 0:
                label = self.font_small.render(u.name, True, (200, 200, 200))
                self.screen.blit(label, (px, py - 12))

    def _draw_orders(self):
        if self.state != STATE_PLANNING:
            return
        ft = CFG.display.fine_tile_px
        co_px = CFG.display.coarse_px

        for u in self.units:
            if u.team != 0 or not u.orders:
                continue
            alpha = 255 if u == self.selected_unit else 80
            prev_px = int(u.fx * ft + co_px // 2)
            prev_py = int(u.fy * ft + co_px // 2)

            for o in u.orders:
                color = ORDER_COLORS.get(o.order_type, (200, 200, 200))
                if alpha < 255:
                    color = tuple(c // 3 for c in color)

                phase_label = f"P{o.phase + 1}"

                if o.order_type in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
                    target_px = int(o.target_fx * ft + co_px // 2)
                    target_py = int(o.target_fy * ft + co_px // 2)
                    pygame.draw.line(self.screen, color,
                                     (prev_px, prev_py),
                                     (target_px, target_py), 2)
                    pygame.draw.circle(self.screen, color,
                                       (target_px, target_py), 5, 2)
                    t_label = self.font_small.render(
                        f"{ORDER_NAMES[o.order_type]} {phase_label}", True, color)
                    self.screen.blit(t_label, (target_px + 6, target_py - 6))
                    prev_px, prev_py = target_px, target_py

                elif o.order_type in (ORDER_GRENADE, ORDER_EXPLOSIVE):
                    target_px = int(o.target_fx * ft + ft // 2)
                    target_py = int(o.target_fy * ft + ft // 2)
                    pygame.draw.line(self.screen, color,
                                     (prev_px, prev_py),
                                     (target_px, target_py), 1)
                    r = 8
                    pygame.draw.circle(self.screen, color,
                                       (target_px, target_py), r, 2)
                    pygame.draw.line(self.screen, color,
                                     (target_px - r, target_py),
                                     (target_px + r, target_py), 1)
                    pygame.draw.line(self.screen, color,
                                     (target_px, target_py - r),
                                     (target_px, target_py + r), 1)

                    if o.order_type == ORDER_GRENADE:
                        lbl = f"GRN {phase_label} fuse:{o.grenade_fuse:.1f}s"
                        blast_r = CFG.weapons.grenade.blast_radius
                    else:
                        slot_name = DET_SLOT_NAMES.get(o.det_slot, "?")
                        lbl = f"EXP {phase_label} @{slot_name}"
                        blast_r = CFG.weapons.door_explosive.blast_radius
                    t_label = self.font_small.render(lbl, True, color)
                    self.screen.blit(t_label, (target_px + 10, target_py - 6))
                    pygame.draw.circle(self.screen, color,
                                       (target_px, target_py),
                                       blast_r * ft, 1)

                elif o.order_type == ORDER_FIRE:
                    target_px = int(o.target_fx * ft + ft // 2)
                    target_py = int(o.target_fy * ft + ft // 2)
                    pygame.draw.line(self.screen, color,
                                     (prev_px, prev_py),
                                     (target_px, target_py), 1)
                    # Crosshair
                    pygame.draw.circle(self.screen, color,
                                       (target_px, target_py), 6, 1)
                    lbl = f"FIRE {phase_label}"
                    t_label = self.font_small.render(lbl, True, color)
                    self.screen.blit(t_label, (target_px + 8, target_py - 6))

    def _draw_projectiles(self):
        ft = CFG.display.fine_tile_px
        for proj in self.projectiles:
            if proj.detonated:
                continue
            px = int(proj.fx * ft)
            py = int(proj.fy * ft)
            color = (255, 50, 0) if proj.proj_type == ORDER_GRENADE else (255, 150, 0)
            pygame.draw.circle(self.screen, color, (px, py), 4)
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 4, 1)

    def _draw_shots(self):
        ft = CFG.display.fine_tile_px
        for s in self.shots:
            age = self.real_time - s.time
            alpha = max(0, 1.0 - age / s.duration)
            brightness = int(255 * alpha)
            color = (brightness, brightness, int(brightness * 0.6))
            p1 = (int(s.fx1 * ft), int(s.fy1 * ft))
            p2 = (int(s.fx2 * ft), int(s.fy2 * ft))
            pygame.draw.line(self.screen, color, p1, p2, 2)
            if age < 0.05:
                pygame.draw.circle(self.screen, (255, 255, 200), p1, 5)

    def _draw_cursor_info(self):
        ft = CFG.display.fine_tile_px
        co = CFG.display.coarse
        fw = CFG.display.fine_w
        fh = CFG.display.fine_h
        mx, my = pygame.mouse.get_pos()
        if mx >= fw * ft:
            return

        color = ORDER_COLORS.get(self.current_mode, (200, 200, 200))

        if self.current_mode in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
            fx = mx // ft - co // 2
            fy = my // ft - co // 2
            fx = max(0, min(fw - co, fx))
            fy = max(0, min(fh - co, fy))
            ghost_rect = (fx * ft, fy * ft, co * ft, co * ft)
            passable = self.gmap.is_passable_block(fy, fx)
            c = color if passable else (255, 0, 0)
            pygame.draw.rect(self.screen, c, ghost_rect, 2)

        elif self.current_mode in (ORDER_GRENADE, ORDER_EXPLOSIVE):
            fx = mx // ft
            fy = my // ft
            cpx = fx * ft + ft // 2
            cpy = fy * ft + ft // 2
            pygame.draw.circle(self.screen, color, (cpx, cpy), 8, 1)
            pygame.draw.line(self.screen, color, (cpx - 10, cpy), (cpx + 10, cpy), 1)
            pygame.draw.line(self.screen, color, (cpx, cpy - 10), (cpx, cpy + 10), 1)
            if self.current_mode == ORDER_GRENADE:
                info = f"Fuse: {self.grenade_fuse:.1f}s (scroll)"
            else:
                slot_name = DET_SLOT_NAMES.get(self.det_slot, "?")
                info = f"Det: {slot_name} (scroll)"
            info_text = self.font_small.render(info, True, color)
            self.screen.blit(info_text, (mx + 15, my - 5))

        elif self.current_mode == ORDER_FIRE:
            fx = mx // ft
            fy = my // ft
            cpx = fx * ft + ft // 2
            cpy = fy * ft + ft // 2
            pygame.draw.circle(self.screen, color, (cpx, cpy), 6, 1)
            pygame.draw.line(self.screen, color, (cpx - 8, cpy), (cpx + 8, cpy), 1)
            pygame.draw.line(self.screen, color, (cpx, cpy - 8), (cpx, cpy + 8), 1)
            info_text = self.font_small.render("Click to set fire target", True, color)
            self.screen.blit(info_text, (mx + 15, my - 5))

    def _draw_ui_panel(self):
        ft = CFG.display.fine_tile_px
        fw = CFG.display.fine_w
        co_px = CFG.display.coarse_px
        tpp = CFG.clock.ticks_per_phase
        tpr = CFG.clock.ticks_per_round
        panel_x = fw * ft

        pygame.draw.rect(self.screen, COL_UI_BG,
                         (panel_x, 0, self.panel_w, self.screen_h))
        pygame.draw.line(self.screen, (50, 50, 60),
                         (panel_x, 0), (panel_x, self.screen_h), 2)

        x = panel_x + 10
        y = 10

        # Title
        title = self.font_big.render("BREACH v2", True, COL_UI_HIGHLIGHT)
        self.screen.blit(title, (x, y))
        y += 25

        # Turn info
        state_name = "PLANNING" if self.state == STATE_PLANNING else "EXECUTING"
        turn_text = self.font.render(
            f"Turn {self.turn_number} — {state_name}", True, COL_UI_TEXT)
        self.screen.blit(turn_text, (x, y))
        y += 18

        if self.state == STATE_EXECUTING:
            phase_text = self.font.render(
                f"Phase {self.exec_phase + 1}/{CFG.clock.phases_per_round}  "
                f"Tick {self.exec_tick}/{tpr}",
                True, COL_UI_HIGHLIGHT)
            self.screen.blit(phase_text, (x, y))
            y += 18
            speed_text = self.font_small.render(
                f"Speed: {self.exec_speed:.1f}x (+/- to adjust)", True, COL_UI_TEXT)
            self.screen.blit(speed_text, (x, y))
            y += 25
        else:
            phase_text = self.font.render(
                f"Planning: Phase {self.planning_phase + 1} (Tab to switch)",
                True, COL_PHASE_DIVIDER)
            self.screen.blit(phase_text, (x, y))
            y += 25

        # Mode selector
        pygame.draw.line(self.screen, (50, 50, 60), (x, y), (x + 230, y), 1)
        y += 8
        mode_label = self.font.render("MODE:", True, COL_UI_TEXT)
        self.screen.blit(mode_label, (x, y))
        y += 18

        modes = [
            (ORDER_MOVE_ATTACK, "1: Move & Attack"),
            (ORDER_MOVE_COVER, "2: Move w/ Cover"),
            (ORDER_SPRINT, "3: Sprint"),
            (ORDER_GRENADE, "G: Grenade"),
            (ORDER_EXPLOSIVE, "B: Explosive"),
            (ORDER_FIRE, "F: Fire"),
        ]
        for mode_id, label in modes:
            color = ORDER_COLORS[mode_id] if self.current_mode == mode_id else (120, 120, 130)
            prefix = "> " if self.current_mode == mode_id else "  "
            text = self.font.render(f"{prefix}{label}", True, color)
            self.screen.blit(text, (x, y))
            y += 16
        y += 10

        # Selected unit info
        pygame.draw.line(self.screen, (50, 50, 60), (x, y), (x + 230, y), 1)
        y += 8
        if self.selected_unit:
            u = self.selected_unit
            name_text = self.font_big.render(u.name, True, COL_SELECT)
            self.screen.blit(name_text, (x, y))
            y += 22
            hp_text = self.font.render(f"HP: {u.hp}/{CFG.marine.hp}", True, COL_UI_TEXT)
            self.screen.blit(hp_text, (x, y))
            y += 16
            pos_text = self.font.render(f"Pos: ({u.cx}, {u.cy})", True, COL_UI_TEXT)
            self.screen.blit(pos_text, (x, y))
            y += 16
            inv_text = self.font.render(
                f"Grenades: {u.has_grenade}  Charges: {u.has_explosive}",
                True, COL_UI_TEXT)
            self.screen.blit(inv_text, (x, y))
            y += 18

            # AP display
            for ph in range(CFG.clock.phases_per_round):
                ap_color = COL_UI_HIGHLIGHT if ph == self.planning_phase else COL_UI_TEXT
                ap_text = self.font.render(
                    f"P{ph+1} AP: {u.get_ap(ph)}/{CFG.clock.ap_per_phase}",
                    True, ap_color)
                self.screen.blit(ap_text, (x, y))
                y += 16
            y += 8

            # Timeline bar (two phases)
            bar_w = 220
            bar_h = 20
            pygame.draw.rect(self.screen, COL_TIMELINE_BG, (x, y, bar_w, bar_h))

            # Phase divider
            mid_x = x + bar_w // 2
            pygame.draw.line(self.screen, COL_PHASE_DIVIDER,
                             (mid_x, y), (mid_x, y + bar_h), 2)
            # Phase labels
            self.screen.blit(
                self.font_small.render("P1", True, COL_PHASE_DIVIDER),
                (x + 2, y + bar_h + 2))
            self.screen.blit(
                self.font_small.render("P2", True, COL_PHASE_DIVIDER),
                (mid_x + 2, y + bar_h + 2))

            # Draw order blocks
            for o in u.orders:
                color = ORDER_COLORS.get(o.order_type, (200, 200, 200))
                phase_offset = o.phase * (bar_w // 2)
                # Each order gets a small block in its phase section
                ox = x + phase_offset + 2
                # Stack orders vertically if multiple in same phase
                phase_orders = [oo for oo in u.orders if oo.phase == o.phase]
                idx = phase_orders.index(o)
                block_h = max(4, (bar_h - 4) // max(1, len(phase_orders)))
                oy = y + 2 + idx * block_h
                block_w = bar_w // 2 - 4
                pygame.draw.rect(self.screen, color,
                                 (ox, oy, block_w, min(block_h - 1, bar_h - 4)))

            # Execution progress
            if self.state == STATE_EXECUTING:
                ex = x + int(self.exec_tick / tpr * bar_w)
                pygame.draw.line(self.screen, (255, 255, 255),
                                 (ex, y - 2), (ex, y + bar_h + 2), 2)

            y += bar_h + 20

            # Orders list
            orders_title = self.font.render("Orders:", True, COL_UI_TEXT)
            self.screen.blit(orders_title, (x, y))
            y += 16
            for o in u.orders:
                color = ORDER_COLORS.get(o.order_type, COL_UI_TEXT)
                name = ORDER_NAMES.get(o.order_type, "?")
                phase_lbl = f"P{o.phase + 1}"
                if o.order_type == ORDER_GRENADE:
                    text = f"  {name} {phase_lbl} fuse:{o.grenade_fuse:.1f}s"
                elif o.order_type == ORDER_EXPLOSIVE:
                    slot = DET_SLOT_NAMES.get(o.det_slot, "?")
                    text = f"  {name} {phase_lbl} @{slot}"
                elif o.order_type == ORDER_FIRE:
                    text = f"  {name} {phase_lbl}"
                else:
                    text = f"  {name} {phase_lbl}"
                order_text = self.font_small.render(text, True, color)
                self.screen.blit(order_text, (x, y))
                y += 14
        else:
            no_sel = self.font.render("No unit selected", True, (100, 100, 110))
            self.screen.blit(no_sel, (x, y))
            y += 20

        # Performance stats
        y = self.screen_h - 180
        pygame.draw.line(self.screen, (50, 50, 60), (x, y), (x + 230, y), 1)
        y += 5
        if self.frame_times:
            avg_frame = sum(self.frame_times) / len(self.frame_times)
            fps = 1000.0 / avg_frame if avg_frame > 0 else 0
            perf_color = (0, 255, 0) if avg_frame < 16.7 else (255, 255, 0) if avg_frame < 33.3 else (255, 0, 0)
            ft_text = self.font_small.render(
                f"FPS: {fps:.0f}  Frame: {avg_frame:.1f}ms", True, perf_color)
            self.screen.blit(ft_text, (x, y))
            y += 14
            phys_text = self.font_small.render(
                f"Physics: {self.physics_ms:.1f}ms / tick", True, perf_color)
            self.screen.blit(phys_text, (x, y))
            y += 14
            wave_max = abs(self.gmap.wave_p).max()
            atm_max = self.gmap.atmosphere.max()
            pf_text = self.font_small.render(
                f"Wave: {wave_max:.1f}  Atm: {atm_max:.2f}", True, COL_UI_TEXT)
            self.screen.blit(pf_text, (x, y))
        y += 18

        # Controls help
        y = self.screen_h - 120
        pygame.draw.line(self.screen, (50, 50, 60), (x, y), (x + 230, y), 1)
        y += 5
        help_lines = [
            "Click unit to select",
            "Click/Right-click: place order",
            "Tab: switch planning phase",
            "Backspace: undo last order",
            "Scroll: fuse time / det slot",
            "Esc: deselect unit",
            "Space/Enter: execute round",
            "F5: reload config",
        ]
        for line in help_lines:
            text = self.font_small.render(line, True, (100, 100, 120))
            self.screen.blit(text, (x, y))
            y += 14


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    game = Game()
    game.run()
