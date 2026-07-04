"""Unit — single class for marines, zombies, and future entity types.

Lifted from ``game.py:Unit`` (lines 1104-1189 in the legacy file). Per the
locked design decisions (``docs/patch_game_logic_migration.md``):

- One class for everyone. Zombie-ness is a STATE (``is_zombie``), not a
  subclass. Marine-only fields (``orders``, ``ap``) and zombie-only
  fields (``zombie_path``, ``last_melee_tick``) coexist on the same
  object; ``apply_action`` (Phase 2) is the chokepoint that ignores
  orders for zombified units.
- Inventory (``has_grenade``, ``has_explosive``) is a BASE field, always
  present. A converted zombie keeps its grenades — a grenade can still
  cook off when the unit walks through fire even though the zombie
  can't issue a "use grenade" order.
- ``speed_ticks_per_tile`` replaces the legacy
  ``zombie_speed_override`` monkey-patch (game.py:1294, 1301, 1944).
  It always exists; default comes from CFG.zombie.ticks_per_tile for
  zombies, CFG.marine attack speed for marines.
- Position lives in physics-tile units as float ``x`` / ``y``.
  Integer matrix indexing via ``tile_x`` / ``tile_y`` properties.
  The old fxf/fyf float fields and fx/fy integer fields are gone —
  ``x`` and ``y`` are the sole source of truth (coord system cleanup
  2026-05-20).

Unit class foundation additions (2026-05-21):
- ``species_id``, ``base_stats``, ``mass``, ``base_speed`` — sampled from
  the species distribution at construction (spec §11).
- ``life_state`` (LifeState enum) — the authoritative life status;
  ``alive`` is a @property derived from it.
- ``faction_id`` — alias for ``team`` (spec §10.1); full relationship
  table deferred.
- ``environment`` — pointer to species EnvironmentProfile (data only).
- ``inventory`` — empty Inventory stub (spec §9).
- ``offsets`` — per-unit copy of species footprint tile offsets.
- ``awakened`` / hidden stat fields — data only; behaviour deferred (spec §13).
- ``facing`` changed from str to float radians (0=East CCW, π/2=North).
  ``facing_compass()`` converts to "N"/"NE"/"E"/... for sprite lookup.
- ``hp`` renamed to ``current_hp``. ``max_hp`` removed; use
  ``effective_vitality(unit)`` from simulation.stats instead.
- ``occupied_tiles()`` / ``occupies()`` — spec §6 interface for collision,
  LOS, hit-detection, stamp_units.
"""
from __future__ import annotations

import math
from enum import Enum


from config import CFG
from simulation import unit_fixed
from simulation.generation import predefined_unit_attributes
from simulation.inventory import Inventory
from simulation.orders import (
    ORDER_MOVE_ATTACK, ORDER_FIRE, MOVE_ORDER_TYPES,
)
from simulation.species import get_species


# ---------------------------------------------------------------------------
# LifeState enum
# ---------------------------------------------------------------------------

class LifeState(Enum):
    """Authoritative life status — the LIFE axis of the two-axis model
    (mechanics/06 §1): is the body functional? ``ALIVE | DEAD``, minimal on
    purpose. Everything else temporarily true of a unit (knockdown, stun,
    burning, ...) is a CONDITION — a :mod:`simulation.status` StatusEffect —
    never a life state. The draft ``DOWNED`` value is retired by that design
    (P3, 2026-07-05): knockdown was never a *life* state; a ``DYING``/
    bleedout life-state remains expressible later if a ruleset wants it.

    Placed in unit.py (not a separate module) because it is used
    exclusively by Unit and its close consumers (combat, conversion).
    """
    ALIVE  = "alive"
    DEAD   = "dead"


# ---------------------------------------------------------------------------
# Facing convention (agent decision, 2026-05-21)
# ---------------------------------------------------------------------------
# Standard math convention: 0 = East, angles increase CCW.
#   π/2  = North  (default spawn — marines face north)
#   π    = West
#   3π/2 = South
#
# This matches Python's math.atan2 / trigonometry convention and makes
# angle arithmetic (e.g. angular difference between two positions) natural.

_NORTH = math.pi / 2   # 1.5707963267948966 radians

# 8-compass snap table (sector width = π/4 = 45°).
# Each entry: (low_bound, high_bound, label) where angles wrap at ±π.
# Sectors are centred on the 8 cardinal + intercardinal directions.
_COMPASS_LABELS = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
_SECTOR_HALF = math.pi / 8   # 22.5°; each direction owns a 45° wedge


class Unit:
    """Game entity: marine, zombie, or future creature.

    Position lives in physics-tile units. ``x`` and ``y`` are float;
    integer tile indices for matrix access come from ``tile_x`` /
    ``tile_y`` properties (which simply floor the float).

    ``team`` semantics: 0 = marine (player), 1 = zombie (enemy). The
    ``is_zombie`` flag mirrors this and is the source-of-truth check for
    "should AI take over this unit". Conversion (marine killed by
    zombie -> zombified at round end) flips both fields.

    Facing convention: float radians, 0=East, CCW positive.
    Default spawn: π/2 (North).
    """

    def __init__(self, name: str, x: float, y: float, team: int = 0,
                 footprint: int = 3, species_id: str = "human"):
        self.name = name
        self.team = team
        self.is_zombie = (team == 1)

        # Stable integer id, assigned by Simulation.add_unit. -1 means the
        # unit has not yet been added (constructor usage / tests).
        self.id = -1

        # Position on the physics-tile grid. Float so renderer can
        # interpolate; integer indexing via tile_x / tile_y properties.
        self.x = float(x)
        self.y = float(y)

        # Side length of the unit's square footprint, in physics tiles.
        # Kept as a plain int field for backward compatibility with callers
        # that read unit.footprint directly (input_handler, AI pathfinding).
        self.footprint = int(footprint)

        # ---- Foundation additions: species + stat sampling ---------------

        self.species_id = species_id
        species = get_species(species_id)

        # Deterministic predefined attributes (ingress door 2 — quantized
        # species means; see generation.py). No RNG at construction: the old
        # unseeded default_rng + MVN draw here was OS-entropy on top of a
        # BLAS/LAPACK transform — both ingress violations. Simulation.add_unit
        # assigns the same predefined values again (harmless, identical).
        self.base_stats, self.mass, self.base_speed = \
            predefined_unit_attributes(species)

        # Per-unit footprint offset list copied from the species default.
        # A 3×3 human uses the species default offsets; any other footprint
        # size gets a square grid of (footprint × footprint) offsets.
        if footprint == 3:
            self.offsets: list[tuple[int, int]] = list(species.default_offsets)
        else:
            self.offsets = [
                (dx, dy)
                for dy in range(footprint)
                for dx in range(footprint)
            ]

        # Life state — authoritative. ``alive`` is a @property derived from it.
        self.life_state = LifeState.ALIVE

        # Faction id — alias for team in this pass (spec §10.1).
        self.faction_id: int = int(team)

        # Environment profile pointer (species baseline; modifiers deferred).
        self.environment = species.environment

        # Mitigation tables pointer (mechanics/06 §3 — species baseline;
        # equipment/status composition deferred). Zombie-state units resolve
        # to species.ZOMBIE_MITIGATION at damage time instead
        # (damage.mitigation_for) — zombie-ness is state, not a species.
        self.mitigation = species.mitigation

        # Inventory stub (real item system deferred; has_grenade/explosive stay).
        self.inventory = Inventory()

        # Status/condition list (mechanics/06 §4) — the CONDITIONS axis of the
        # two-axis model: StatusEffect instances (simulation.status), SYNCED
        # state hashed by the lockstep digest (__unit_status__). Ticked in
        # unit-id / list order (P0) at the top of the unit-simulation section
        # of Simulation.step(); unit logic consults status.composed_flags(),
        # never this list directly.
        self.statuses: list = []

        # Hidden Hartmann fields — data only; behaviour deferred (spec §13).
        self.awakened: bool = False

        # ---- Facing (float radians, 0=East CCW, π/2=North) ---------------
        # Default spawn facing = North (π/2), matching legacy "N" default.
        self.facing: float = _NORTH

        # ---- HP: current_hp from sampled vitality ------------------------
        # max_hp is removed; use effective_vitality(unit) from stats module.
        self.current_hp: float = float(self.base_stats.vitality)

        # ---- Legacy fields: kept unchanged --------------------------------

        # Movement speed in ticks per fine tile. Replaces the legacy
        # zombie_speed_override monkey-patch. Marines override per-order
        # (movement.marine_attack/cover/sprint_ticks_per_tile); zombies
        # use this field directly.
        if self.is_zombie:
            self.speed_ticks_per_tile = CFG.zombie.ticks_per_tile
        else:
            self.speed_ticks_per_tile = CFG.movement.marine_attack_ticks_per_tile

        # Order / planning state.
        self.orders = []
        self.current_order_type = ORDER_MOVE_ATTACK
        self.ap = [CFG.clock.ap_per_phase, CFG.clock.ap_per_phase]

        # Inventory booleans — BASE fields. Wiring into self.inventory deferred.
        self.has_grenade   = 0 if self.is_zombie else CFG.marine.grenades
        self.has_explosive = 0 if self.is_zombie else CFG.marine.explosives

        # Combat state (used by marines mainly; harmless on zombies).
        self.last_fire_tick = -999
        self.fire_target = None

        # Zombie AI state.
        self.zombie_activated       = False
        self.zombie_path            = []
        self.zombie_path_idx        = 0
        self.zombie_move_accumulator = 0
        self.last_melee_tick        = -999
        self.killed_by_zombie       = False

        # Precomputed per-tick movement path.
        self.move_path        = []
        self.path_tick_offset = 0

    # ------------------------------------------------------------------
    # Life state
    # ------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        """True if the unit is in the ALIVE state.

        Kept as a @property (was a plain bool field) so all existing callers
        of ``unit.alive`` continue to work without change. The authoritative
        field is ``unit.life_state``.
        """
        return self.life_state is LifeState.ALIVE

    @alive.setter
    def alive(self, value: bool) -> None:
        """Support legacy assignments: ``unit.alive = False``."""
        self.life_state = LifeState.ALIVE if value else LifeState.DEAD

    # ------------------------------------------------------------------
    # Occupancy interface (spec §5, §6)
    # ------------------------------------------------------------------

    def occupied_tiles(self) -> list[tuple[int, int]]:
        """Tiles this unit currently occupies (spec §6 interface).

        Returns [(anchor_x + dx, anchor_y + dy) for (dx, dy) in self.offsets].

        No rotation applied — the 3×3 symmetric default doesn't need it.
        TODO: apply facing rotation for non-symmetric rigid shapes (spec §15
        item 3) when those footprints are introduced.
        """
        ax, ay = self.tile_x, self.tile_y
        return [(ax + dx, ay + dy) for (dx, dy) in self.offsets]

    def occupies(self, tile: tuple[int, int]) -> bool:
        """True if this unit occupies the given tile (spec §6 interface)."""
        ax, ay = self.tile_x, self.tile_y
        tx, ty = tile
        for dx, dy in self.offsets:
            if ax + dx == tx and ay + dy == ty:
                return True
        return False

    # ------------------------------------------------------------------
    # Facing helpers
    # ------------------------------------------------------------------

    def face_towards(self, target_x: float, target_y: float) -> None:
        """Update ``self.facing`` to point from the unit's current position
        toward ``(target_x, target_y)``. Called by the movement code each
        tick a unit takes a step, so sprites visibly track direction of
        travel. No-op if the target is the current position.

        World Y increases downward; the facing convention is math-style
        (Y-up). The negation on dy converts between them.
        """
        dx = target_x - self.x
        dy = target_y - self.y
        if dx == 0 and dy == 0:
            return
        # Q2-lift: facing is SYNCED state, and math.atan2 is libm — CRT/Python
        # versions differ at the last ULP, which the lockstep digest amplifies
        # into a cross-machine hash flip (the X-ARCH Ada finding). Route
        # through the pure-integer kit: quantize the deltas at the boundary,
        # integer atan2, dequantize back — an exact n/65536 double, identical
        # everywhere. Shift vs libm <= ~1.5e-5 rad (imperceptible;
        # pre-approved, no feel-check). Downstream consumers (facing_compass,
        # renderer, flashlight cone) are unchanged — they still see a float
        # radian. Deltas that BOTH quantize to zero (|d| < 1/131072 tiles)
        # yield facing 0.0 — deterministic, and unreachable from real
        # movement steps (>= ~1e-2 tiles).
        self.facing = unit_fixed.atan2_rad(-dy, dx)

    def facing_compass(self) -> str:
        """Convert self.facing (radians) to 8-compass string.

        Convention: 0=East, increasing CCW. Snaps to the nearest 45°
        sector. Returns one of: "N", "NE", "E", "SE", "S", "SW", "W", "NW".

        Used by the sprite system for directional art selection.
        """
        # Normalise to [-π, π].
        angle = self.facing
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi

        # East=0, NE=π/4, N=π/2, NW=3π/4, W=±π
        # SE=-π/4, S=-π/2, SW=-3π/4
        # Sector index = round(angle / (π/4)) mod 8.
        sector = round(angle / (math.pi / 4)) % 8
        # sector 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
        labels = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
        return labels[sector]

    # ------------------------------------------------------------------
    # Tile-coordinate helpers
    # ------------------------------------------------------------------

    @property
    def tile_x(self) -> int:
        """Integer tile index (col) — for matrix access."""
        return int(self.x)

    @property
    def tile_y(self) -> int:
        """Integer tile index (row) — for matrix access."""
        return int(self.y)

    def center_tile_x(self) -> int:
        """Col index of the unit's center tile."""
        return self.tile_x + self.footprint // 2

    def center_tile_y(self) -> int:
        """Row index of the unit's center tile."""
        return self.tile_y + self.footprint // 2

    # ------------------------------------------------------------------
    # Orders / AP helpers (used by the planning phase + UI)
    # ------------------------------------------------------------------

    def clear_orders(self):
        """Drop all queued orders and refill AP. Called at round teardown."""
        self.orders = []
        self.ap = [CFG.clock.ap_per_phase, CFG.clock.ap_per_phase]

    def get_ap(self, phase):
        return self.ap[phase]

    def spend_ap(self, phase, cost=1):
        self.ap[phase] -= cost

    def get_planned_end_pos(self):
        """Return the tile position the unit will reach after all queued
        movement orders. Used by grenade projectile spawn."""
        for o in reversed(self.orders):
            if o.order_type in MOVE_ORDER_TYPES:
                return o.target_fx, o.target_fy
        return self.tile_x, self.tile_y

    def get_planned_pos_after_phase(self, phase: int):
        """Position after executing all queued move orders up through and
        including ``phase``. Used by the order overlay so a Phase 2 line
        starts where Phase 1 leaves the unit, not where it currently is."""
        for o in reversed(self.orders):
            if o.order_type in MOVE_ORDER_TYPES and o.phase <= phase:
                return o.target_fx, o.target_fy
        return self.tile_x, self.tile_y

    def get_orders_for_phase(self, phase):
        return [o for o in self.orders if o.phase == phase]

    def has_move_order_in_phase(self, phase):
        return any(o.order_type in MOVE_ORDER_TYPES and o.phase == phase
                   for o in self.orders)

    def get_fire_order_in_phase(self, phase):
        for o in self.orders:
            if o.order_type == ORDER_FIRE and o.phase == phase:
                return o
        return None
