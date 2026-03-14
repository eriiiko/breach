"""
Breach — Tactical Squad Game Prototype
Pygame prototype for testing gameplay mechanics.

Controls:
  PLANNING PHASE:
    Click unit to select it
    1 = Move & Attack mode
    2 = Move with Cover mode
    3 = Sprint mode
    G = Grenade mode (click target, scroll to set detonation delay)
    B = Plant Explosive mode (click adjacent wall/door)
    Right-click = place waypoint for selected unit
    Escape = clear selected unit's orders
    Space / Enter = End turn (start execution)

  EXECUTION PHASE:
    Plays out automatically, then returns to planning.
"""

import pygame
import numpy as np
import math
import os
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FINE_TILE = 10          # pixels per fine tile
COARSE = 3              # fine tiles per coarse tile (1 marine = 3x3 fine)
COARSE_PX = FINE_TILE * COARSE  # 30px per coarse tile

# Map dimensions in coarse tiles
MAP_W = 40              # coarse tiles wide
MAP_H = 25              # coarse tiles tall
FINE_W = MAP_W * COARSE # fine tiles wide
FINE_H = MAP_H * COARSE # fine tiles tall

# Turn system
T_STEPS = 5             # timesteps per turn
T_DURATION = 1.0        # seconds per timestep
TICKS_PER_T = 10        # simulation sub-ticks per timestep
EXEC_FPS_SCALE = 1.0    # execution playback speed multiplier

# Physics
D_ATM = 200.0
D_SMOKE = 0.4
ADVECTION_RATE = 25.0
PHYSICS_DT = 0.001

# Material properties: {material_id: {name, hp, reflectivity, absorption, flammable, passable, color}}
# reflectivity: how much blast energy bounces back (1.0 = perfect mirror, 0.0 = none)
# absorption:   how much blast energy is absorbed/lost (1.0 = absorbs all, 0.0 = none)
# What passes through = 1.0 - reflectivity - absorption
MATERIALS = {
    0: {"name": "air",    "hp": 0,    "reflectivity": 0.0,  "absorption": 0.0,  "flammable": False, "passable": True},
    1: {"name": "hull",   "hp": 300,  "reflectivity": 0.9,  "absorption": 0.1,  "flammable": False, "passable": False},
    2: {"name": "wood",   "hp": 60,   "reflectivity": 0.4,  "absorption": 0.5,  "flammable": True,  "passable": False},
    3: {"name": "door",   "hp": 40,   "reflectivity": 0.3,  "absorption": 0.3,  "flammable": True,  "passable": True},
    # Special: unit bodies (not a map material, but used for physics)
    # High absorption, low reflectivity — bodies soak up blast, don't bounce it
    99: {"name": "unit",  "hp": 100,  "reflectivity": 0.1,  "absorption": 0.85, "flammable": False, "passable": False},
}

# Colors
COL_BG = (10, 10, 15)
COL_FLOOR = (30, 35, 45)
COL_WALL_HULL = (80, 80, 90)
COL_WALL_WOOD = (70, 50, 30)
COL_VACUUM = (0, 20, 30)
COL_GRID = (25, 30, 40)
COL_SELECT = (0, 200, 255)
COL_WAYPOINT = (255, 200, 0)
COL_MOVE_ATTACK = (255, 100, 100)
COL_MOVE_COVER = (100, 200, 100)
COL_SPRINT = (100, 150, 255)
COL_GRENADE_TARGET = (255, 50, 0)
COL_EXPLOSIVE_TARGET = (255, 150, 0)
COL_UI_BG = (20, 20, 30)
COL_UI_TEXT = (200, 200, 210)
COL_UI_HIGHLIGHT = (0, 200, 255)
COL_TIMELINE_BG = (40, 40, 55)
COL_TIMELINE_TICK = (80, 80, 100)

# Movement speeds (fine tiles per timestep)
SPEED_MOVE_ATTACK = 3
SPEED_MOVE_COVER = 2
SPEED_SPRINT = 5

# Grenade
GRENADE_RANGE = 30       # fine tiles max throw distance
GRENADE_MIN_DELAY = 1.0  # minimum detonation delay in T
GRENADE_MAX_DELAY = 10.0
GRENADE_BLAST_RADIUS = 6  # fine tiles
GRENADE_PRESSURE = 10.0

# Explosive (breaching charge)
EXPLOSIVE_BLAST_RADIUS = 3  # fine tiles — small, focused
EXPLOSIVE_PRESSURE = 5.0
EXPLOSIVE_WALL_DAMAGE = 500  # enough to destroy any wall

# Zombies
ZOMBIE_SPEED = 6.0          # fine tiles per second during execution
ZOMBIE_DAMAGE = 15          # damage per melee hit
ZOMBIE_HP = 40
ZOMBIE_ATTACK_RANGE = 4     # fine tiles — adjacent
ZOMBIE_DETECT_RANGE = 30    # fine tiles — LOS activation range
ZOMBIE_ATTACK_COOLDOWN = 1.0  # seconds between attacks
COL_ZOMBIE = (180, 40, 40)

# Combat
OVERWATCH_RANGE = 40        # fine tiles
OVERWATCH_DAMAGE = 35
OVERWATCH_COOLDOWN = 0.8    # seconds between overwatch shots
SHOT_DURATION = 0.4         # how long tracer line shows
MOVE_ATTACK_RANGE = 30      # fine tiles — auto-fire range during move & attack
MOVE_ATTACK_DAMAGE = 25
MOVE_ATTACK_COOLDOWN = 1.0  # seconds between auto-fire shots
GRENADE_UNIT_DAMAGE = 60    # damage to units in grenade blast
BLAST_DAMAGE_THRESHOLD = 5  # minimum blast damage to register
UNIT_ABSORPTION = 0.85      # units absorb 85% of blast energy passing through
UNIT_REFLECTIVITY = 0.10    # units reflect only 10% of blast energy back

# Game states
STATE_PLANNING = 0
STATE_EXECUTING = 1

# Order types
ORDER_MOVE_ATTACK = 0
ORDER_MOVE_COVER = 1
ORDER_SPRINT = 2
ORDER_GRENADE = 3
ORDER_EXPLOSIVE = 4
ORDER_OVERWATCH = 5

ORDER_NAMES = {
    ORDER_MOVE_ATTACK: "Move & Attack",
    ORDER_MOVE_COVER: "Move w/ Cover",
    ORDER_SPRINT: "Sprint",
    ORDER_GRENADE: "Grenade",
    ORDER_EXPLOSIVE: "Explosive",
    ORDER_OVERWATCH: "Overwatch",
}

COL_OVERWATCH = (255, 255, 100)

ORDER_COLORS = {
    ORDER_MOVE_ATTACK: COL_MOVE_ATTACK,
    ORDER_MOVE_COVER: COL_MOVE_COVER,
    ORDER_SPRINT: COL_SPRINT,
    ORDER_GRENADE: COL_GRENADE_TARGET,
    ORDER_EXPLOSIVE: COL_EXPLOSIVE_TARGET,
    ORDER_OVERWATCH: COL_OVERWATCH,
}

ORDER_SPEEDS = {
    ORDER_MOVE_ATTACK: SPEED_MOVE_ATTACK,
    ORDER_MOVE_COVER: SPEED_MOVE_COVER,
    ORDER_SPRINT: SPEED_SPRINT,
}


# ---------------------------------------------------------------------------
# Map generation — simple ship layout
# ---------------------------------------------------------------------------
class GameMap:
    """2D grid map with fine-tile resolution."""

    def __init__(self):
        # Material types: 0=air, 1=hull, 2=wood wall, 3=door
        self.material = np.zeros((FINE_H, FINE_W), dtype=np.int8)
        self.wall_hp = np.zeros((FINE_H, FINE_W), dtype=np.float32)
        self.is_wall = np.zeros((FINE_H, FINE_W), dtype=bool)
        self.is_vacuum = np.zeros((FINE_H, FINE_W), dtype=bool)
        self.flammable = np.zeros((FINE_H, FINE_W), dtype=bool)

        # Physics fields
        self.atmosphere = np.ones((FINE_H, FINE_W), dtype=np.float32)
        self.smoke = np.zeros((FINE_H, FINE_W), dtype=np.float32)
        self.unit_absorb = np.zeros((FINE_H, FINE_W), dtype=np.float32)

        self._build_ship()
        self._update_caches()

    def _build_ship(self):
        """Build a simple ship layout for testing."""
        m = self.material

        # Fill everything with hull first
        m[:] = 1  # hull everywhere

        # Carve out interior (leave 1-tile hull border)
        hull_t = COARSE  # hull thickness in fine tiles (1 coarse tile)
        interior = np.s_[hull_t:FINE_H - hull_t, hull_t:FINE_W - hull_t]
        m[interior] = 0  # air

        # Add some internal wood walls to create rooms
        # Vertical wall dividing ship into left and right sections
        wall_x = 15 * COARSE  # at coarse tile 15
        for y in range(hull_t, FINE_H - hull_t):
            for dx in range(1):  # 1-tile-thick wall
                m[y, wall_x + dx] = 2

        # Door in the vertical wall (3 fine tiles wide = 1 coarse)
        door_y = 10 * COARSE
        for dy in range(COARSE):
            m[door_y + dy, wall_x] = 3  # door

        # Another door lower
        door_y2 = 18 * COARSE
        for dy in range(COARSE):
            m[door_y2 + dy, wall_x] = 3

        # Horizontal wall in left section
        wall_y = 8 * COARSE
        for x in range(hull_t, wall_x):
            m[wall_y, x] = 2
        # Door
        door_x = 8 * COARSE
        for dx in range(COARSE):
            m[wall_y, door_x + dx] = 3

        # Horizontal wall in right section
        wall_y2 = 12 * COARSE
        for x in range(wall_x + 1, FINE_W - hull_t):
            m[wall_y2, x] = 2
        # Door
        door_x2 = 25 * COARSE
        for dx in range(COARSE):
            m[wall_y2, door_x2 + dx] = 3

        # Another room in top-right
        wall_y3 = 6 * COARSE
        for x in range(wall_x + 1, 30 * COARSE):
            m[wall_y3, x] = 2
        door_x3 = 22 * COARSE
        for dx in range(COARSE):
            m[wall_y3, door_x3 + dx] = 3

        # Vertical wall creating a corridor on the right
        wall_x2 = 30 * COARSE
        for y in range(hull_t, wall_y2):
            m[y, wall_x2] = 2
        door_y3 = 4 * COARSE
        for dy in range(COARSE):
            m[door_y3 + dy, wall_x2] = 3

        # Set vacuum outside hull
        m[0:hull_t, :] = 1
        m[FINE_H - hull_t:, :] = 1
        m[:, 0:hull_t] = 1
        m[:, FINE_W - hull_t:] = 1

    def _update_caches(self):
        """Rebuild cached arrays from material grid using MATERIALS table."""
        m = self.material
        # Build caches from material properties
        self.is_wall = np.zeros((FINE_H, FINE_W), dtype=bool)
        self.flammable = np.zeros((FINE_H, FINE_W), dtype=bool)
        self.wall_hp = np.zeros((FINE_H, FINE_W), dtype=np.float32)
        self.mat_reflectivity = np.zeros((FINE_H, FINE_W), dtype=np.float32)
        self.mat_absorption = np.zeros((FINE_H, FINE_W), dtype=np.float32)

        for mat_id, props in MATERIALS.items():
            if mat_id == 99:  # skip unit pseudo-material
                continue
            mask = (m == mat_id)
            if not props["passable"]:
                self.is_wall[mask] = True
            if props["flammable"]:
                self.flammable[mask] = True
            if props["hp"] > 0:
                self.wall_hp[mask] = props["hp"]
            self.mat_reflectivity[mask] = props["reflectivity"]
            self.mat_absorption[mask] = props["absorption"]

        self.is_vacuum = np.zeros_like(self.is_wall)
        # Outside the hull = vacuum
        self.is_vacuum[0:COARSE, :] = True
        self.is_vacuum[FINE_H - COARSE:, :] = True
        self.is_vacuum[:, 0:COARSE] = True
        self.is_vacuum[:, FINE_W - COARSE:] = True

        # Atmosphere
        self.atmosphere = np.where(self.is_wall | self.is_vacuum, 0.0, 1.0).astype(np.float32)

    def stamp_units(self, units):
        """Stamp living unit positions onto the unit_absorb array."""
        self.unit_absorb[:] = 0.0
        for u in units:
            if not u.alive:
                continue
            uy, ux = int(u.fy), int(u.fx)
            for dy in range(COARSE):
                for dx in range(COARSE):
                    ny, nx = uy + dy, ux + dx
                    if 0 <= ny < FINE_H and 0 <= nx < FINE_W:
                        self.unit_absorb[ny, nx] = UNIT_ABSORPTION

    def is_passable(self, fy, fx):
        """Check if a fine tile is passable (air or door)."""
        if fy < 0 or fy >= FINE_H or fx < 0 or fx >= FINE_W:
            return False
        return self.material[fy, fx] == 0 or self.material[fy, fx] == 3

    def is_passable_coarse(self, cy, cx):
        """Check if a coarse tile is fully passable (all 3x3 fine tiles)."""
        fy, fx = cy * COARSE, cx * COARSE
        for dy in range(COARSE):
            for dx in range(COARSE):
                if not self.is_passable(fy + dy, fx + dx):
                    return False
        return True

    def is_passable_block(self, fy, fx):
        """Check if a 3x3 fine-tile block (unit footprint) is fully passable."""
        if fy < 0 or fx < 0 or fy + COARSE > FINE_H or fx + COARSE > FINE_W:
            return False
        for dy in range(COARSE):
            for dx in range(COARSE):
                if not self.is_passable(fy + dy, fx + dx):
                    return False
        return True

    def has_los(self, fy1, fx1, fy2, fx2):
        """Bresenham line-of-sight check between two fine tile positions."""
        dx = abs(fx2 - fx1)
        dy = abs(fy2 - fy1)
        sx = 1 if fx1 < fx2 else -1
        sy = 1 if fy1 < fy2 else -1
        err = dx - dy
        x, y = fx1, fy1
        while True:
            if x == fx2 and y == fy2:
                return True
            if 0 <= y < FINE_H and 0 <= x < FINE_W and self.is_wall[y, x]:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def destroy_wall(self, fy, fx):
        """Destroy a wall/door tile — turn it into air."""
        if 0 <= fy < FINE_H and 0 <= fx < FINE_W:
            if self.material[fy, fx] in (2, 3):  # wood or door
                self.material[fy, fx] = 0
                self.wall_hp[fy, fx] = 0
                self._update_wall_at(fy, fx)

    def _update_wall_at(self, fy, fx):
        """Update caches for a single tile."""
        m = self.material[fy, fx]
        self.is_wall[fy, fx] = (m == 1) or (m == 2)
        self.flammable[fy, fx] = (m == 2)
        if m == 0:
            # Newly opened tile gets atmosphere from neighbors
            self.atmosphere[fy, fx] = 0.5


# ---------------------------------------------------------------------------
# Physics engine (simplified from prototypes)
# ---------------------------------------------------------------------------
class Physics:
    """Handles atmosphere and smoke simulation."""

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
        # Walls: full reflection (neighbor pressure = own pressure)
        up = np.where(wall_up, p, up)
        down = np.where(wall_down, p, down)
        left = np.where(wall_left, p, left)
        right = np.where(wall_right, p, right)
        # Units: high absorption, low reflectivity
        # At rest (p=1.0, neighbor=1.0) there should be no effect.
        # Only excess pressure (shockwaves) gets reflected/absorbed.
        if unit_absorb is not None:
            abs_up = np.roll(unit_absorb, 1, axis=0)
            abs_down = np.roll(unit_absorb, -1, axis=0)
            abs_left = np.roll(unit_absorb, 1, axis=1)
            abs_right = np.roll(unit_absorb, -1, axis=1)
            # Blend neighbor toward: baseline + reflected portion of excess
            # neighbor_eff = 1.0 + (neighbor - 1.0) * REFLECTIVITY
            up = np.where(abs_up > 0, 1.0 + (up - 1.0) * UNIT_REFLECTIVITY, up)
            down = np.where(abs_down > 0, 1.0 + (down - 1.0) * UNIT_REFLECTIVITY, down)
            left = np.where(abs_left > 0, 1.0 + (left - 1.0) * UNIT_REFLECTIVITY, left)
            right = np.where(abs_right > 0, 1.0 + (right - 1.0) * UNIT_REFLECTIVITY, right)
        return up + down + left + right - 4.0 * p

    @staticmethod
    def step_atmosphere(gmap, dt):
        lap = Physics.compute_laplacian(gmap.atmosphere, gmap.is_wall,
                                         gmap.unit_absorb)
        gmap.atmosphere += D_ATM * dt * lap
        gmap.atmosphere[gmap.is_wall] = 0.0
        gmap.atmosphere[gmap.is_vacuum] = 0.0
        # Units absorb excess pressure (shockwaves) but not baseline atmosphere
        absorb_mask = gmap.unit_absorb > 0
        excess = gmap.atmosphere[absorb_mask] - 1.0
        excess[excess > 0] *= (1.0 - gmap.unit_absorb[absorb_mask][excess > 0])
        gmap.atmosphere[absorb_mask] = 1.0 + excess
        np.clip(gmap.atmosphere, 0.0, 20.0, out=gmap.atmosphere)

    @staticmethod
    def step_smoke(gmap, dt):
        lap = Physics.compute_laplacian(gmap.smoke, gmap.is_wall)
        gmap.smoke += D_SMOKE * dt * lap

        # Advection
        grad_y = (np.roll(gmap.atmosphere, -1, axis=0) -
                  np.roll(gmap.atmosphere, 1, axis=0)) / 2.0
        grad_x = (np.roll(gmap.atmosphere, -1, axis=1) -
                  np.roll(gmap.atmosphere, 1, axis=1)) / 2.0
        ds_dy = (np.roll(gmap.smoke, -1, axis=0) -
                 np.roll(gmap.smoke, 1, axis=0)) / 2.0
        ds_dx = (np.roll(gmap.smoke, -1, axis=1) -
                 np.roll(gmap.smoke, 1, axis=1)) / 2.0
        gmap.smoke += ADVECTION_RATE * dt * (grad_x * ds_dx + grad_y * ds_dy)

        gmap.smoke[gmap.is_wall] = 0.0
        gmap.smoke[gmap.is_vacuum] = 0.0
        np.clip(gmap.smoke, 0.0, 1.0, out=gmap.smoke)

    @staticmethod
    def apply_explosion(gmap, fy, fx, radius, pressure, wall_damage):
        """Apply explosion: damage walls, spike atmosphere, clear smoke.
        Units act as blast absorbers — pressure is reduced behind them."""
        # Precompute which tiles are blocked by units using radial raycasts
        blocked = np.zeros((FINE_H, FINE_W), dtype=np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < FINE_H and 0 <= nx < FINE_W:
                    dist = math.sqrt(dy * dy + dx * dx)
                    if dist <= radius:
                        # Walk ray from blast center to this tile,
                        # accumulate absorption from unit tiles along the way
                        steps = max(abs(dy), abs(dx))
                        absorbed = 0.0
                        if steps > 0:
                            for s in range(1, steps):
                                sy = fy + int(round(dy * s / steps))
                                sx = fx + int(round(dx * s / steps))
                                if 0 <= sy < FINE_H and 0 <= sx < FINE_W:
                                    absorbed += gmap.unit_absorb[sy, sx]
                        blocked[ny, nx] = min(absorbed, 1.0)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = fy + dy, fx + dx
                if 0 <= ny < FINE_H and 0 <= nx < FINE_W:
                    dist = math.sqrt(dy * dy + dx * dx)
                    if dist <= radius:
                        falloff = 1.0 - (dist / radius)
                        # Reduce by unit absorption along the ray
                        shield = 1.0 - blocked[ny, nx]
                        effective = falloff * shield
                        # Damage walls
                        if gmap.material[ny, nx] in (2, 3):
                            gmap.wall_hp[ny, nx] -= wall_damage * effective
                            if gmap.wall_hp[ny, nx] <= 0:
                                gmap.destroy_wall(ny, nx)
                        # Spike atmosphere (only in air tiles)
                        if not gmap.is_wall[ny, nx] and not gmap.is_vacuum[ny, nx]:
                            gmap.atmosphere[ny, nx] += pressure * effective
                        # Clear smoke near center
                        if dist <= radius * 0.4 and shield > 0.5:
                            gmap.smoke[ny, nx] = 0.0

    @staticmethod
    def step(gmap, n_substeps=5):
        """Run n substeps of physics."""
        for _ in range(n_substeps):
            Physics.step_atmosphere(gmap, PHYSICS_DT)
            Physics.step_smoke(gmap, PHYSICS_DT)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
class Order:
    def __init__(self, order_type, target_fx, target_fy, t_start, t_end=None,
                 grenade_delay=None):
        self.order_type = order_type
        self.target_fx = target_fx  # fine tile target
        self.target_fy = target_fy
        self.t_start = t_start      # timestep when order begins
        self.t_end = t_end          # timestep when order ends (for movement)
        self.grenade_delay = grenade_delay  # detonation time (for grenades)


class Unit:
    def __init__(self, name, cx, cy, team=0):
        self.name = name
        self.team = team
        # Position in fine tiles (center of 3x3 block = top-left of block)
        self.fx = cx * COARSE
        self.fy = cy * COARSE
        # Fractional position for smooth movement
        self.fxf = float(self.fx)
        self.fyf = float(self.fy)
        self.orders = []
        self.alive = True
        self.hp = 100
        self.facing = "S"  # direction for sprite
        self.current_order_type = ORDER_MOVE_ATTACK
        self.has_grenade = 2
        self.has_explosive = 2

    @property
    def cx(self):
        return int(self.fxf) // COARSE

    @property
    def cy(self):
        return int(self.fyf) // COARSE

    def get_center_px(self):
        """Get pixel position of unit center."""
        return (int(self.fxf * FINE_TILE + COARSE_PX / 2),
                int(self.fyf * FINE_TILE + COARSE_PX / 2))

    def clear_orders(self):
        self.orders = []

    def add_move_order(self, target_fx, target_fy, order_type, t_start):
        """Add a movement order to a fine tile position."""
        speed = ORDER_SPEEDS.get(order_type, SPEED_MOVE_ATTACK)

        # Calculate how many timesteps this move takes
        dist = math.sqrt((target_fx - self.get_planned_end_fx()) ** 2 +
                         (target_fy - self.get_planned_end_fy()) ** 2)
        t_needed = max(1, math.ceil(dist / speed))
        t_end = min(t_start + t_needed, T_STEPS)

        order = Order(order_type, target_fx, target_fy, t_start, t_end)
        self.orders.append(order)
        return t_end

    def add_grenade_order(self, target_fx, target_fy, t_start, delay):
        """Add a grenade throw order."""
        if self.has_grenade <= 0:
            return t_start
        order = Order(ORDER_GRENADE, target_fx, target_fy, t_start,
                      grenade_delay=delay)
        self.orders.append(order)
        return t_start  # throwing is free action

    def add_explosive_order(self, target_fx, target_fy, t_start, delay):
        """Add an explosive plant order."""
        if self.has_explosive <= 0:
            return t_start
        order = Order(ORDER_EXPLOSIVE, target_fx, target_fy, t_start,
                      grenade_delay=delay)
        self.orders.append(order)
        return t_start

    def get_planned_end_pos(self):
        """Get the coarse tile where the unit will be after all current orders."""
        fx, fy = self.get_planned_end_fx(), self.get_planned_end_fy()
        return fx // COARSE, fy // COARSE

    def get_planned_end_fx(self):
        if not self.orders:
            return self.fx
        # Find last movement order
        for o in reversed(self.orders):
            if o.order_type in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
                return o.target_fx
        return self.fx

    def get_planned_end_fy(self):
        if not self.orders:
            return self.fy
        for o in reversed(self.orders):
            if o.order_type in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
                return o.target_fy
        return self.fy

    def get_next_t(self):
        """Get the next available timestep for orders."""
        if not self.orders:
            return 0
        max_t = 0
        for o in self.orders:
            if o.t_end is not None:
                max_t = max(max_t, o.t_end)
            else:
                max_t = max(max_t, o.t_start)
        return min(max_t, T_STEPS)


# ---------------------------------------------------------------------------
# Projectiles (grenades, explosives in flight / planted)
# ---------------------------------------------------------------------------
class Projectile:
    def __init__(self, proj_type, fx, fy, target_fx, target_fy,
                 detonate_time, thrown_time):
        self.proj_type = proj_type  # ORDER_GRENADE or ORDER_EXPLOSIVE
        self.fx = float(fx)
        self.fy = float(fy)
        self.target_fx = float(target_fx)
        self.target_fy = float(target_fy)
        self.detonate_time = detonate_time  # absolute time in T
        self.thrown_time = thrown_time
        self.detonated = False
        # Travel time: ~0.5T for grenades to reach target
        self.travel_time = 0.3

    def update(self, current_t):
        """Update position. Returns True if detonated this frame."""
        # Lerp to target during travel
        if current_t < self.thrown_time + self.travel_time:
            frac = (current_t - self.thrown_time) / self.travel_time
            frac = max(0.0, min(1.0, frac))
            start_fx, start_fy = self.fx, self.fy
            self.fx = start_fx + (self.target_fx - start_fx) * frac
            self.fy = start_fy + (self.target_fy - start_fy) * frac
        else:
            self.fx = self.target_fx
            self.fy = self.target_fy

        if current_t >= self.detonate_time and not self.detonated:
            self.detonated = True
            return True
        return False


# ---------------------------------------------------------------------------
# Shot tracers (visual feedback for gunfire)
# ---------------------------------------------------------------------------
class Shot:
    def __init__(self, fx1, fy1, fx2, fy2, time):
        self.fx1, self.fy1 = fx1, fy1
        self.fx2, self.fy2 = fx2, fy2
        self.time = time
        self.duration = SHOT_DURATION


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game:
    def __init__(self):
        pygame.init()

        # Window
        self.panel_w = 250  # right side UI panel width
        self.screen_w = FINE_W * FINE_TILE + self.panel_w
        self.screen_h = FINE_H * FINE_TILE
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("BREACH — Tactical Prototype")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 11)

        # Map
        self.gmap = GameMap()

        # Units — 3 marines in the left room
        self.units = [
            Unit("Alpha", 4, 4, team=0),
            Unit("Bravo", 6, 4, team=0),
            Unit("Charlie", 5, 6, team=0),
        ]

        # Zombies — in the right section rooms
        zombie_positions = [(20, 9), (22, 10), (24, 8), (21, 14), (26, 15), (28, 10)]
        for i, (zx, zy) in enumerate(zombie_positions):
            z = Unit(f"Z{i+1}", zx, zy, team=1)
            z.hp = ZOMBIE_HP
            z.zombie_activated = False
            z.last_attack_time = -999.0
            self.units.append(z)

        self.selected_unit = None
        self.current_mode = ORDER_MOVE_ATTACK
        self.grenade_delay = 2.0  # default detonation delay in T

        # Game state
        self.state = STATE_PLANNING
        self.turn_number = 1

        # Execution state
        self.exec_time = 0.0  # current time in T during execution
        self.exec_speed = 1.0  # T per real second during playback
        self.projectiles = []
        self.shots = []  # tracer visuals
        self.real_time = 0.0  # total real seconds elapsed during execution

        # Camera / scroll (for future use)
        self.cam_x = 0
        self.cam_y = 0

        # Load sprites
        self.sprites = {}
        sprite_dir = os.path.join(os.path.dirname(__file__), "art", "sprites", "marine")
        for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            path = os.path.join(sprite_dir, f"marine_{direction}.png")
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                self.sprites[direction] = img

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif self.state == STATE_PLANNING:
                    self._handle_planning_event(event)
                elif self.state == STATE_EXECUTING:
                    self._handle_execution_event(event)

            if self.state == STATE_EXECUTING:
                self._update_execution(dt)

            self._draw()
            pygame.display.flip()

        pygame.quit()

    # --- Planning phase input ---
    def _handle_planning_event(self, event):
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
            elif event.key == pygame.K_o:
                self.current_mode = ORDER_OVERWATCH
            elif event.key == pygame.K_BACKSPACE:
                if self.selected_unit and self.selected_unit.orders:
                    self.selected_unit.orders.pop()
            elif event.key == pygame.K_ESCAPE:
                self.selected_unit = None
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                self._start_execution()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            # Check if click is on the map
            if mx < FINE_W * FINE_TILE:
                if event.button == 1:  # left click
                    self._handle_map_left_click(mx, my)
                elif event.button == 3:  # right click
                    self._handle_map_right_click(mx, my)

        elif event.type == pygame.MOUSEWHEEL:
            if self.current_mode in (ORDER_GRENADE, ORDER_EXPLOSIVE):
                self.grenade_delay = max(GRENADE_MIN_DELAY,
                                         min(GRENADE_MAX_DELAY,
                                             self.grenade_delay + event.y * 0.5))

    def _handle_map_left_click(self, mx, my):
        # Convert to fine tile
        fx = mx // FINE_TILE
        fy = my // FINE_TILE

        # Try to select a player unit (check if click is inside unit's 3x3 block)
        for u in self.units:
            if u.alive and u.team == 0 and u.fx <= fx < u.fx + COARSE and u.fy <= fy < u.fy + COARSE:
                self.selected_unit = u
                return

        # If we have a selected unit and clicked elsewhere, place order
        if self.selected_unit:
            self._place_order(mx, my)

    def _handle_map_right_click(self, mx, my):
        if self.selected_unit:
            self._place_order(mx, my)

    def _place_order(self, mx, my):
        u = self.selected_unit
        if not u or not u.alive:
            return

        next_t = u.get_next_t()
        if next_t >= T_STEPS:
            return  # no more timesteps available

        if self.current_mode in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
            # Fine-tile targeting: center the unit's 3x3 block on the clicked tile
            fx = mx // FINE_TILE
            fy = my // FINE_TILE
            target_fx = fx - COARSE // 2
            target_fy = fy - COARSE // 2
            # Clamp to map bounds
            target_fx = max(0, min(FINE_W - COARSE, target_fx))
            target_fy = max(0, min(FINE_H - COARSE, target_fy))
            if self.gmap.is_passable_block(target_fy, target_fx):
                u.add_move_order(target_fx, target_fy, self.current_mode, next_t)

        elif self.current_mode == ORDER_GRENADE:
            # Fine tile target
            fx = mx // FINE_TILE
            fy = my // FINE_TILE
            if u.has_grenade > 0:
                u.add_grenade_order(fx, fy, next_t, self.grenade_delay)

        elif self.current_mode == ORDER_EXPLOSIVE:
            fx = mx // FINE_TILE
            fy = my // FINE_TILE
            if u.has_explosive > 0:
                u.add_explosive_order(fx, fy, next_t, self.grenade_delay)

        elif self.current_mode == ORDER_OVERWATCH:
            # Overwatch fills remaining timesteps — unit holds and watches
            order = Order(ORDER_OVERWATCH, u.get_planned_end_fx(),
                          u.get_planned_end_fy(), next_t, t_end=T_STEPS)
            u.orders.append(order)
            u.overwatch_last_fire = -999.0

    # --- Execution ---
    def _start_execution(self):
        self.state = STATE_EXECUTING
        self.exec_time = 0.0
        self.real_time = 0.0
        self.projectiles = []
        self.shots = []

        # Init combat tracking
        for u in self.units:
            if u.team == 0:
                u.overwatch_last_fire = -999.0
                u.move_attack_last_fire = -999.0

        # Prepare projectiles from orders
        for u in self.units:
            for o in u.orders:
                if o.order_type == ORDER_GRENADE:
                    px, py = u.get_center_px()
                    proj = Projectile(
                        ORDER_GRENADE,
                        u.fx + COARSE / 2, u.fy + COARSE / 2,
                        o.target_fx + 0.5, o.target_fy + 0.5,
                        detonate_time=o.t_start + o.grenade_delay,
                        thrown_time=o.t_start,
                    )
                    self.projectiles.append(proj)
                    u.has_grenade -= 1
                elif o.order_type == ORDER_EXPLOSIVE:
                    proj = Projectile(
                        ORDER_EXPLOSIVE,
                        u.fx + COARSE / 2, u.fy + COARSE / 2,
                        o.target_fx + 0.5, o.target_fy + 0.5,
                        detonate_time=o.t_start + o.grenade_delay,
                        thrown_time=o.t_start,
                    )
                    self.projectiles.append(proj)
                    u.has_explosive -= 1

        # Save start positions for interpolation
        for u in self.units:
            u._exec_start_fx = float(u.fx)
            u._exec_start_fy = float(u.fy)

        # Stamp initial unit positions for physics
        self.gmap.stamp_units(self.units)

    def _update_execution(self, dt):
        prev_time = self.exec_time
        self.exec_time += dt * self.exec_speed
        self.real_time += dt

        # Update player unit positions based on orders
        for u in self.units:
            if not u.alive or u.team != 0:
                continue
            self._update_unit_position(u)

        # Update zombie AI
        self._update_zombies(dt)

        # Re-stamp unit positions for physics (units move each tick)
        self.gmap.stamp_units(self.units)

        # Update overwatch
        self._update_overwatch()

        # Update move & attack auto-fire
        self._update_move_attack()

        # Expire old shot tracers
        self.shots = [s for s in self.shots if
                      self.real_time - s.time < s.duration]

        # Update and detonate projectiles
        for proj in self.projectiles:
            if not proj.detonated:
                if proj.update(self.exec_time):
                    # Detonate!
                    fx = int(proj.target_fx)
                    fy = int(proj.target_fy)
                    if proj.proj_type == ORDER_GRENADE:
                        Physics.apply_explosion(
                            self.gmap, fy, fx,
                            GRENADE_BLAST_RADIUS, GRENADE_PRESSURE,
                            wall_damage=200)
                        # Damage units in blast radius
                        self._apply_blast_damage(fx, fy, GRENADE_BLAST_RADIUS,
                                                 GRENADE_UNIT_DAMAGE)
                        # Add smoke
                        for ddy in range(-4, 5):
                            for ddx in range(-4, 5):
                                ny, nx = fy + ddy, fx + ddx
                                if (0 <= ny < FINE_H and 0 <= nx < FINE_W and
                                        not self.gmap.is_wall[ny, nx]):
                                    dist = math.sqrt(ddy**2 + ddx**2)
                                    if dist < 5:
                                        self.gmap.smoke[ny, nx] = min(
                                            1.0, self.gmap.smoke[ny, nx] + 0.8 * (1 - dist/5))
                    elif proj.proj_type == ORDER_EXPLOSIVE:
                        Physics.apply_explosion(
                            self.gmap, fy, fx,
                            EXPLOSIVE_BLAST_RADIUS, EXPLOSIVE_PRESSURE,
                            wall_damage=EXPLOSIVE_WALL_DAMAGE)
                        # Damage units in blast radius
                        self._apply_blast_damage(fx, fy, EXPLOSIVE_BLAST_RADIUS,
                                                 GRENADE_UNIT_DAMAGE)
                        self.gmap._update_caches()

        # Run physics
        Physics.step(self.gmap, n_substeps=3)

        # Check if execution is complete
        if self.exec_time >= T_STEPS:
            self._end_execution()

    def _update_unit_position(self, u):
        """Interpolate unit position based on current execution time and orders."""
        # Find which movement order applies at current time
        current_fx = float(u._exec_start_fx)
        current_fy = float(u._exec_start_fy)

        for o in u.orders:
            if o.order_type not in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
                continue
            if o.t_end is None:
                continue

            if self.exec_time < o.t_start:
                break  # haven't reached this order yet

            # Previous position (start of this order)
            prev_fx = current_fx
            prev_fy = current_fy

            if self.exec_time >= o.t_end:
                # Order complete
                current_fx = float(o.target_fx)
                current_fy = float(o.target_fy)
            else:
                # Interpolate
                frac = (self.exec_time - o.t_start) / max(0.01, o.t_end - o.t_start)
                frac = max(0.0, min(1.0, frac))
                current_fx = prev_fx + (o.target_fx - prev_fx) * frac
                current_fy = prev_fy + (o.target_fy - prev_fy) * frac

        u.fxf = current_fx
        u.fyf = current_fy
        u.fx = int(round(current_fx))
        u.fy = int(round(current_fy))

        # Update facing based on movement direction
        dx = current_fx - u.fxf if hasattr(u, '_prev_fxf') else 0
        dy = current_fy - u.fyf if hasattr(u, '_prev_fyf') else 0
        u._prev_fxf = current_fx
        u._prev_fyf = current_fy

    def _end_execution(self):
        """End execution phase, return to planning."""
        self.state = STATE_PLANNING
        self.turn_number += 1

        # Snap units to nearest fine tile
        for u in self.units:
            u.fx = round(u.fxf)
            u.fy = round(u.fyf)
            u.fxf = float(u.fx)
            u.fyf = float(u.fy)
            u.clear_orders()

        # Clear unit physics mask
        self.gmap.unit_absorb[:] = 0.0

        # Clean up projectiles
        self.projectiles = [p for p in self.projectiles if not p.detonated
                            and p.detonate_time > T_STEPS]

    def _update_zombies(self, dt):
        """AI for zombie units during execution."""
        players = [u for u in self.units if u.team == 0 and u.alive]
        if not players:
            return

        for z in self.units:
            if z.team != 1 or not z.alive:
                continue

            # Check if any player is visible
            nearest = None
            nearest_dist = float('inf')
            zc_fy, zc_fx = z.fy + COARSE // 2, z.fx + COARSE // 2

            for p in players:
                pc_fy, pc_fx = p.fy + COARSE // 2, p.fx + COARSE // 2
                dist = math.sqrt((zc_fx - pc_fx) ** 2 + (zc_fy - pc_fy) ** 2)
                if dist < ZOMBIE_DETECT_RANGE:
                    if self.gmap.has_los(zc_fy, zc_fx, pc_fy, pc_fx):
                        z.zombie_activated = True
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest = p

            if not z.zombie_activated or nearest is None:
                continue

            # Attack if close enough
            if nearest_dist <= ZOMBIE_ATTACK_RANGE:
                if self.real_time - z.last_attack_time >= ZOMBIE_ATTACK_COOLDOWN:
                    z.last_attack_time = self.real_time
                    nearest.hp -= ZOMBIE_DAMAGE
                    if nearest.hp <= 0:
                        nearest.alive = False
                continue

            # Move toward nearest player
            dx = nearest.fx - z.fx
            dy = nearest.fy - z.fy
            dist = max(1.0, math.sqrt(dx * dx + dy * dy))
            move = ZOMBIE_SPEED * dt * self.exec_speed
            ndx = (dx / dist) * move
            ndy = (dy / dist) * move

            # Try full movement
            nx = z.fxf + ndx
            ny = z.fyf + ndy
            nfx, nfy = int(round(nx)), int(round(ny))

            if self.gmap.is_passable_block(nfy, nfx):
                z.fxf, z.fyf = nx, ny
                z.fx, z.fy = nfx, nfy
            else:
                # Try horizontal only
                nx2 = z.fxf + ndx
                nfx2 = int(round(nx2))
                if self.gmap.is_passable_block(z.fy, nfx2):
                    z.fxf = nx2
                    z.fx = nfx2
                else:
                    # Try vertical only
                    ny2 = z.fyf + ndy
                    nfy2 = int(round(ny2))
                    if self.gmap.is_passable_block(nfy2, z.fx):
                        z.fyf = ny2
                        z.fy = nfy2

    def _update_overwatch(self):
        """Check overwatch units and fire at visible enemies."""
        for u in self.units:
            if u.team != 0 or not u.alive:
                continue
            # Check if unit has an active overwatch order at current time
            has_ow = False
            for o in u.orders:
                if (o.order_type == ORDER_OVERWATCH and
                        o.t_start <= self.exec_time <= (o.t_end or T_STEPS)):
                    has_ow = True
                    break
            if not has_ow:
                continue

            if self.real_time - getattr(u, 'overwatch_last_fire', -999) < OVERWATCH_COOLDOWN:
                continue

            uc_fy, uc_fx = u.fy + COARSE // 2, u.fx + COARSE // 2

            for z in self.units:
                if z.team != 1 or not z.alive:
                    continue
                zc_fy, zc_fx = z.fy + COARSE // 2, z.fx + COARSE // 2
                dist = math.sqrt((uc_fx - zc_fx) ** 2 + (uc_fy - zc_fy) ** 2)
                if dist <= OVERWATCH_RANGE:
                    if self.gmap.has_los(uc_fy, uc_fx, zc_fy, zc_fx):
                        # Fire!
                        z.hp -= OVERWATCH_DAMAGE
                        if z.hp <= 0:
                            z.alive = False
                        u.overwatch_last_fire = self.real_time
                        self.shots.append(Shot(
                            uc_fx, uc_fy, zc_fx, zc_fy, self.real_time))
                        break  # one target per cooldown

    def _apply_blast_damage(self, fx, fy, radius, max_damage):
        """Damage all units within blast radius (both teams — friendly fire!).
        Units between blast and target absorb damage (body shielding)."""
        for u in self.units:
            if not u.alive:
                continue
            # Distance from blast center to unit center
            uc_fx = u.fx + COARSE // 2
            uc_fy = u.fy + COARSE // 2
            dist = math.sqrt((uc_fx - fx) ** 2 + (uc_fy - fy) ** 2)
            if dist <= radius:
                falloff = 1.0 - (dist / radius)
                # Check for shielding by other units along blast ray
                shield = self._calc_unit_shielding(fx, fy, uc_fx, uc_fy, u)
                damage = int(max_damage * falloff * shield)
                if damage >= BLAST_DAMAGE_THRESHOLD:
                    u.hp -= damage
                    if u.hp <= 0:
                        u.alive = False

    def _calc_unit_shielding(self, bx, by, tx, ty, target_unit):
        """Calculate how much blast reaches target, accounting for unit bodies
        blocking the path. Returns multiplier 0.0 (fully shielded) to 1.0."""
        dx = tx - bx
        dy = ty - by
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return 1.0
        absorbed = 0.0
        for s in range(1, steps):
            sx = bx + int(round(dx * s / steps))
            sy = by + int(round(dy * s / steps))
            # Check if any other alive unit occupies this fine tile
            for u in self.units:
                if u is target_unit or not u.alive:
                    continue
                if u.fx <= sx < u.fx + COARSE and u.fy <= sy < u.fy + COARSE:
                    absorbed += UNIT_ABSORPTION
                    break
        return max(0.0, 1.0 - min(absorbed, 1.0))

    def _update_move_attack(self):
        """Units with Move & Attack orders auto-fire at visible enemies."""
        for u in self.units:
            if u.team != 0 or not u.alive:
                continue
            # Check if unit has an active move & attack order
            has_ma = False
            for o in u.orders:
                if (o.order_type == ORDER_MOVE_ATTACK and
                        o.t_start <= self.exec_time <= (o.t_end or T_STEPS)):
                    has_ma = True
                    break
            if not has_ma:
                continue

            if self.real_time - getattr(u, 'move_attack_last_fire', -999) < MOVE_ATTACK_COOLDOWN:
                continue

            uc_fy, uc_fx = u.fy + COARSE // 2, u.fx + COARSE // 2

            for z in self.units:
                if z.team != 1 or not z.alive:
                    continue
                zc_fy, zc_fx = z.fy + COARSE // 2, z.fx + COARSE // 2
                dist = math.sqrt((uc_fx - zc_fx) ** 2 + (uc_fy - zc_fy) ** 2)
                if dist <= MOVE_ATTACK_RANGE:
                    if self.gmap.has_los(uc_fy, uc_fx, zc_fy, zc_fx):
                        z.hp -= MOVE_ATTACK_DAMAGE
                        if z.hp <= 0:
                            z.alive = False
                        u.move_attack_last_fire = self.real_time
                        self.shots.append(Shot(
                            uc_fx, uc_fy, zc_fx, zc_fy, self.real_time))
                        break

    def _handle_execution_event(self, event):
        # Allow speed adjustment during execution
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.exec_speed = min(10.0, self.exec_speed + 0.5)
            elif event.key == pygame.K_MINUS:
                self.exec_speed = max(0.5, self.exec_speed - 0.5)

    # --- Drawing ---
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
        """Draw the tile grid."""
        surf = self.screen
        for cy in range(MAP_H):
            for cx in range(MAP_W):
                px = cx * COARSE_PX
                py = cy * COARSE_PX
                fy, fx = cy * COARSE, cx * COARSE

                # Determine dominant material in this coarse tile
                mat = self.gmap.material[fy:fy+COARSE, fx:fx+COARSE]

                if np.any(mat == 1):  # hull
                    color = COL_WALL_HULL
                elif np.any(mat == 2):  # wood
                    color = COL_WALL_WOOD
                elif np.any(mat == 3):  # door
                    color = (50, 70, 50)
                else:
                    color = COL_FLOOR

                pygame.draw.rect(surf, color, (px, py, COARSE_PX, COARSE_PX))

        # Draw fine grid lines (subtle)
        for x in range(0, FINE_W * FINE_TILE, COARSE_PX):
            pygame.draw.line(surf, COL_GRID, (x, 0), (x, FINE_H * FINE_TILE), 1)
        for y in range(0, FINE_H * FINE_TILE, COARSE_PX):
            pygame.draw.line(surf, COL_GRID, (0, y), (FINE_W * FINE_TILE, y), 1)

    def _draw_atmosphere(self):
        """Draw atmosphere as a subtle blue overlay."""
        # Only draw if there's interesting atmosphere variation
        atm = self.gmap.atmosphere
        if atm.max() - atm.min() < 0.01:
            return

        # Downsample to coarse for performance
        overlay = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        for cy in range(MAP_H):
            for cx in range(MAP_W):
                fy, fx = cy * COARSE, cx * COARSE
                val = atm[fy:fy+COARSE, fx:fx+COARSE].mean()
                if val > 1.05:
                    # Overpressure shockwave: bright orange-white
                    intensity = min(255, int((val - 1.0) * 200))
                    r = min(255, 200 + int((val - 1.0) * 55))
                    g = min(255, 120 + int((val - 1.0) * 80))
                    b = min(255, int((val - 1.0) * 40))
                    overlay.set_at((cx, cy), (r, g, b, intensity))
                elif val < 0.9 and not self.gmap.is_wall[fy, fx]:
                    # Low pressure / vacuum pull: blue-purple tint
                    intensity = min(220, int((1.0 - val) * 300))
                    overlay.set_at((cx, cy), (80, 50, 255, intensity))

        scaled = pygame.transform.scale(overlay,
                                         (FINE_W * FINE_TILE, FINE_H * FINE_TILE))
        self.screen.blit(scaled, (0, 0))

    def _draw_smoke(self):
        """Draw smoke as a gray overlay."""
        smoke = self.gmap.smoke
        if smoke.max() < 0.01:
            return

        overlay = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        for cy in range(MAP_H):
            for cx in range(MAP_W):
                fy, fx = cy * COARSE, cx * COARSE
                val = smoke[fy:fy+COARSE, fx:fx+COARSE].mean()
                if val > 0.01:
                    intensity = min(200, int(val * 220))
                    overlay.set_at((cx, cy), (180, 160, 140, intensity))

        scaled = pygame.transform.scale(overlay,
                                         (FINE_W * FINE_TILE, FINE_H * FINE_TILE))
        self.screen.blit(scaled, (0, 0))

    def _draw_units(self):
        """Draw all units."""
        for u in self.units:
            if not u.alive:
                continue

            px = int(u.fxf * FINE_TILE)
            py = int(u.fyf * FINE_TILE)

            # Draw selection highlight
            if u == self.selected_unit and self.state == STATE_PLANNING:
                sel_rect = (px - 2, py - 2, COARSE_PX + 4, COARSE_PX + 4)
                pygame.draw.rect(self.screen, COL_SELECT, sel_rect, 2)

            if u.team == 1:
                # Zombie: red rectangle
                pygame.draw.rect(self.screen, COL_ZOMBIE,
                                 (px + 2, py + 2, COARSE_PX - 4, COARSE_PX - 4))
                # HP bar
                if u.hp < ZOMBIE_HP:
                    bar_w = int((u.hp / ZOMBIE_HP) * (COARSE_PX - 4))
                    pygame.draw.rect(self.screen, (255, 0, 0),
                                     (px + 2, py + COARSE_PX - 4, COARSE_PX - 4, 3))
                    pygame.draw.rect(self.screen, (0, 255, 0),
                                     (px + 2, py + COARSE_PX - 4, bar_w, 3))
            else:
                # Marine: sprite or green fallback
                sprite = self.sprites.get(u.facing, self.sprites.get("S"))
                if sprite:
                    scaled = pygame.transform.scale(sprite, (COARSE_PX, COARSE_PX))
                    self.screen.blit(scaled, (px, py))
                else:
                    pygame.draw.rect(self.screen, (0, 180, 0),
                                     (px + 2, py + 2, COARSE_PX - 4, COARSE_PX - 4))
                # Player HP bar
                if u.hp < 100:
                    bar_w = int((u.hp / 100) * (COARSE_PX - 4))
                    pygame.draw.rect(self.screen, (255, 0, 0),
                                     (px + 2, py + COARSE_PX - 4, COARSE_PX - 4, 3))
                    pygame.draw.rect(self.screen, (0, 255, 0),
                                     (px + 2, py + COARSE_PX - 4, bar_w, 3))

            # Unit name label (only for marines)
            if u.team == 0:
                label = self.font_small.render(u.name, True, (200, 200, 200))
                self.screen.blit(label, (px, py - 12))

    def _draw_orders(self):
        """Draw planned orders for selected unit."""
        if self.state != STATE_PLANNING:
            return

        for u in self.units:
            if not u.orders:
                continue

            alpha = 255 if u == self.selected_unit else 80
            prev_px = int(u.fx * FINE_TILE + COARSE_PX // 2)
            prev_py = int(u.fy * FINE_TILE + COARSE_PX // 2)

            for o in u.orders:
                color = ORDER_COLORS.get(o.order_type, (200, 200, 200))
                if alpha < 255:
                    color = tuple(c // 3 for c in color)

                if o.order_type in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
                    target_px = int(o.target_fx * FINE_TILE + COARSE_PX // 2)
                    target_py = int(o.target_fy * FINE_TILE + COARSE_PX // 2)
                    # Draw line
                    pygame.draw.line(self.screen, color,
                                     (prev_px, prev_py),
                                     (target_px, target_py), 2)
                    # Draw waypoint marker
                    pygame.draw.circle(self.screen, color,
                                       (target_px, target_py), 5, 2)
                    # Draw time label showing arrival time
                    t_label = self.font_small.render(
                        f"T{o.t_start}-{o.t_end}", True, color)
                    self.screen.blit(t_label, (target_px + 6, target_py - 6))
                    prev_px, prev_py = target_px, target_py

                elif o.order_type in (ORDER_GRENADE, ORDER_EXPLOSIVE):
                    target_px = int(o.target_fx * FINE_TILE + FINE_TILE // 2)
                    target_py = int(o.target_fy * FINE_TILE + FINE_TILE // 2)
                    # Draw throw arc (dashed line)
                    pygame.draw.line(self.screen, color,
                                     (prev_px, prev_py),
                                     (target_px, target_py), 1)
                    # Draw target crosshair
                    r = 8
                    pygame.draw.circle(self.screen, color,
                                       (target_px, target_py), r, 2)
                    pygame.draw.line(self.screen, color,
                                     (target_px - r, target_py),
                                     (target_px + r, target_py), 1)
                    pygame.draw.line(self.screen, color,
                                     (target_px, target_py - r),
                                     (target_px, target_py + r), 1)
                    # Detonation time label
                    det_t = o.t_start + (o.grenade_delay or 0)
                    name = "GRN" if o.order_type == ORDER_GRENADE else "EXP"
                    t_label = self.font_small.render(
                        f"{name} T{det_t:.1f}", True, color)
                    self.screen.blit(t_label, (target_px + 10, target_py - 6))

                    # Draw blast radius preview
                    radius_px = (GRENADE_BLAST_RADIUS if o.order_type == ORDER_GRENADE
                                 else EXPLOSIVE_BLAST_RADIUS) * FINE_TILE
                    pygame.draw.circle(self.screen, (*color[:3], 40),
                                       (target_px, target_py), radius_px, 1)

                elif o.order_type == ORDER_OVERWATCH:
                    # Draw overwatch indicator: crosshair + range circle at unit position
                    ow_px = int(o.target_fx * FINE_TILE + COARSE_PX // 2)
                    ow_py = int(o.target_fy * FINE_TILE + COARSE_PX // 2)
                    pygame.draw.circle(self.screen, color, (ow_px, ow_py),
                                       OVERWATCH_RANGE * FINE_TILE, 1)
                    # Eye icon (simple crosshair)
                    pygame.draw.circle(self.screen, color, (ow_px, ow_py), 6, 1)
                    pygame.draw.line(self.screen, color, (ow_px - 3, ow_py),
                                     (ow_px + 3, ow_py), 1)
                    pygame.draw.line(self.screen, color, (ow_px, ow_py - 3),
                                     (ow_px, ow_py + 3), 1)
                    t_label = self.font_small.render(
                        f"OW T{o.t_start}-{o.t_end}", True, color)
                    self.screen.blit(t_label, (ow_px + 8, ow_py - 6))

    def _draw_projectiles(self):
        """Draw in-flight projectiles during execution."""
        for proj in self.projectiles:
            if proj.detonated:
                continue
            px = int(proj.fx * FINE_TILE)
            py = int(proj.fy * FINE_TILE)
            color = (255, 50, 0) if proj.proj_type == ORDER_GRENADE else (255, 150, 0)
            pygame.draw.circle(self.screen, color, (px, py), 4)
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 4, 1)

    def _draw_shots(self):
        """Draw shot tracer lines."""
        for s in self.shots:
            age = self.real_time - s.time
            alpha = max(0, 1.0 - age / s.duration)
            brightness = int(255 * alpha)
            color = (brightness, brightness, int(brightness * 0.6))
            p1 = (int(s.fx1 * FINE_TILE), int(s.fy1 * FINE_TILE))
            p2 = (int(s.fx2 * FINE_TILE), int(s.fy2 * FINE_TILE))
            pygame.draw.line(self.screen, color, p1, p2, 2)
            # Muzzle flash
            if age < 0.05:
                pygame.draw.circle(self.screen, (255, 255, 200), p1, 5)

    def _draw_cursor_info(self):
        """Draw info at cursor position during planning."""
        mx, my = pygame.mouse.get_pos()
        if mx >= FINE_W * FINE_TILE:
            return

        color = ORDER_COLORS.get(self.current_mode, (200, 200, 200))

        if self.current_mode in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
            # Show 3x3 unit footprint at fine-tile resolution
            fx = mx // FINE_TILE - COARSE // 2
            fy = my // FINE_TILE - COARSE // 2
            fx = max(0, min(FINE_W - COARSE, fx))
            fy = max(0, min(FINE_H - COARSE, fy))
            ghost_rect = (fx * FINE_TILE, fy * FINE_TILE,
                          COARSE * FINE_TILE, COARSE * FINE_TILE)
            passable = self.gmap.is_passable_block(fy, fx)
            c = color if passable else (255, 0, 0)
            pygame.draw.rect(self.screen, c, ghost_rect, 2)

        elif self.current_mode in (ORDER_GRENADE, ORDER_EXPLOSIVE):
            # Grenade/explosive crosshair at fine tile resolution
            fx = mx // FINE_TILE
            fy = my // FINE_TILE
            cpx = fx * FINE_TILE + FINE_TILE // 2
            cpy = fy * FINE_TILE + FINE_TILE // 2
            pygame.draw.circle(self.screen, color, (cpx, cpy), 8, 1)
            pygame.draw.line(self.screen, color, (cpx - 10, cpy), (cpx + 10, cpy), 1)
            pygame.draw.line(self.screen, color, (cpx, cpy - 10), (cpx, cpy + 10), 1)
            # Show delay
            delay_text = self.font_small.render(
                f"Delay: {self.grenade_delay:.1f}T (scroll to change)", True, color)
            self.screen.blit(delay_text, (mx + 15, my - 5))

        elif self.current_mode == ORDER_OVERWATCH:
            # Show overwatch range circle centered on unit's planned end position
            if self.selected_unit:
                u = self.selected_unit
                efx = u.get_planned_end_fx()
                efy = u.get_planned_end_fy()
                cx_px = int(efx * FINE_TILE + COARSE_PX // 2)
                cy_px = int(efy * FINE_TILE + COARSE_PX // 2)
                pygame.draw.circle(self.screen, COL_OVERWATCH,
                                   (cx_px, cy_px), OVERWATCH_RANGE * FINE_TILE, 1)
                ow_text = self.font_small.render("Click to set Overwatch", True, COL_OVERWATCH)
                self.screen.blit(ow_text, (mx + 15, my - 5))

    def _draw_ui_panel(self):
        """Draw the right-side UI panel."""
        panel_x = FINE_W * FINE_TILE
        panel_rect = (panel_x, 0, self.panel_w, self.screen_h)
        pygame.draw.rect(self.screen, COL_UI_BG, panel_rect)
        pygame.draw.line(self.screen, (50, 50, 60),
                         (panel_x, 0), (panel_x, self.screen_h), 2)

        x = panel_x + 10
        y = 10

        # Title
        title = self.font_big.render("BREACH", True, COL_UI_HIGHLIGHT)
        self.screen.blit(title, (x, y))
        y += 25

        # Turn info
        state_name = "PLANNING" if self.state == STATE_PLANNING else "EXECUTING"
        turn_text = self.font.render(f"Turn {self.turn_number} — {state_name}", True,
                                      COL_UI_TEXT)
        self.screen.blit(turn_text, (x, y))
        y += 20

        if self.state == STATE_EXECUTING:
            t_text = self.font.render(f"T = {self.exec_time:.2f} / {T_STEPS}", True,
                                       COL_UI_HIGHLIGHT)
            self.screen.blit(t_text, (x, y))
            y += 20
            speed_text = self.font_small.render(
                f"Speed: {self.exec_speed:.1f}x (+/- to adjust)", True, COL_UI_TEXT)
            self.screen.blit(speed_text, (x, y))
            y += 30
        else:
            y += 10

        # Mode
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
            (ORDER_OVERWATCH, "O: Overwatch"),
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
            hp_text = self.font.render(f"HP: {u.hp}", True, COL_UI_TEXT)
            self.screen.blit(hp_text, (x, y))
            y += 16
            pos_text = self.font.render(f"Pos: ({u.cx}, {u.cy})", True, COL_UI_TEXT)
            self.screen.blit(pos_text, (x, y))
            y += 16
            inv_text = self.font.render(
                f"Grenades: {u.has_grenade}  Charges: {u.has_explosive}", True, COL_UI_TEXT)
            self.screen.blit(inv_text, (x, y))
            y += 20

            # Timeline
            next_t = u.get_next_t()
            timeline_text = self.font.render(
                f"Next T: {next_t}/{T_STEPS}", True, COL_UI_TEXT)
            self.screen.blit(timeline_text, (x, y))
            y += 20

            # Draw timeline bar
            bar_w = 220
            bar_h = 20
            pygame.draw.rect(self.screen, COL_TIMELINE_BG,
                             (x, y, bar_w, bar_h))
            # Tick marks
            for t in range(T_STEPS + 1):
                tx = x + int(t / T_STEPS * bar_w)
                pygame.draw.line(self.screen, COL_TIMELINE_TICK,
                                 (tx, y), (tx, y + bar_h), 1)
                label = self.font_small.render(str(t), True, COL_TIMELINE_TICK)
                self.screen.blit(label, (tx - 3, y + bar_h + 2))

            # Draw order blocks on timeline
            for o in u.orders:
                t_s = o.t_start
                t_e = o.t_end if o.t_end else t_s + 0.3
                ox = x + int(t_s / T_STEPS * bar_w)
                ow = max(4, int((t_e - t_s) / T_STEPS * bar_w))
                color = ORDER_COLORS.get(o.order_type, (200, 200, 200))
                pygame.draw.rect(self.screen, color, (ox, y + 2, ow, bar_h - 4))

            # Execution progress
            if self.state == STATE_EXECUTING:
                ex = x + int(self.exec_time / T_STEPS * bar_w)
                pygame.draw.line(self.screen, (255, 255, 255),
                                 (ex, y - 2), (ex, y + bar_h + 2), 2)

            y += bar_h + 20

            # Orders list
            orders_title = self.font.render("Orders:", True, COL_UI_TEXT)
            self.screen.blit(orders_title, (x, y))
            y += 16
            for i, o in enumerate(u.orders):
                color = ORDER_COLORS.get(o.order_type, COL_UI_TEXT)
                name = ORDER_NAMES.get(o.order_type, "?")
                if o.order_type in (ORDER_GRENADE, ORDER_EXPLOSIVE):
                    det = o.t_start + (o.grenade_delay or 0)
                    text = f"  {name} @T{o.t_start} det:{det:.1f}"
                else:
                    text = f"  {name} T{o.t_start}-{o.t_end}"
                order_text = self.font_small.render(text, True, color)
                self.screen.blit(order_text, (x, y))
                y += 14
        else:
            no_sel = self.font.render("No unit selected", True, (100, 100, 110))
            self.screen.blit(no_sel, (x, y))
            y += 20

        # Controls help at bottom
        y = self.screen_h - 120
        pygame.draw.line(self.screen, (50, 50, 60), (x, y), (x + 230, y), 1)
        y += 5
        help_lines = [
            "Click unit to select",
            "Click/Right-click: place order",
            "Backspace: undo last order",
            "Scroll: grenade delay",
            "Esc: deselect unit",
            "Space/Enter: execute turn",
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
