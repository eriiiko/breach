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
"""
from __future__ import annotations

from config import CFG
from simulation.orders import (
    ORDER_MOVE_ATTACK, ORDER_FIRE, MOVE_ORDER_TYPES,
)


class Unit:
    """Game entity: marine, zombie, or future creature.

    Position lives in physics-tile units. ``x`` and ``y`` are float;
    integer tile indices for matrix access come from ``tile_x`` /
    ``tile_y`` properties (which simply floor the float).

    ``team`` semantics: 0 = marine (player), 1 = zombie (enemy). The
    ``is_zombie`` flag mirrors this and is the source-of-truth check for
    "should AI take over this unit". Conversion (marine killed by
    zombie -> zombified at round end) flips both fields.
    """

    def __init__(self, name: str, x: float, y: float, team: int = 0,
                 footprint: int = 3):
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
        # Default 3 (size-1 human). Will become a function of the variant
        # system later — see unit_variants_design_brainstorm.md.
        self.footprint = int(footprint)

        self.alive = True
        self.facing = "N"   # default spawn pose: facing north (marines spawn south, look north)

        # HP from config based on type.
        self.hp = CFG.zombie.hp if self.is_zombie else CFG.marine.hp
        self.max_hp = self.hp

        # Movement speed in ticks per fine tile. Replaces the legacy
        # zombie_speed_override monkey-patch. Marines override per-order
        # (movement.marine_attack/cover/sprint_ticks_per_tile); zombies
        # use this field directly. Caller (e.g. spawn helper) can adjust
        # for runners (faster) and brutes (slower).
        if self.is_zombie:
            self.speed_ticks_per_tile = CFG.zombie.ticks_per_tile
        else:
            self.speed_ticks_per_tile = CFG.movement.marine_attack_ticks_per_tile

        # Order / planning state. Always present (zombie may still carry
        # orders pre-conversion; AI just doesn't read them once converted).
        self.orders = []
        self.current_order_type = ORDER_MOVE_ATTACK
        self.ap = [CFG.clock.ap_per_phase, CFG.clock.ap_per_phase]

        # Inventory — BASE field. Marines get a starting loadout from
        # config; zombies start with none. Stays with the unit through
        # conversion (the grenade is still in the zombie's pocket).
        self.has_grenade   = 0 if self.is_zombie else CFG.marine.grenades
        self.has_explosive = 0 if self.is_zombie else CFG.marine.explosives

        # Combat state (used by marines mainly; harmless on zombies).
        self.last_fire_tick = -999
        self.fire_target = None

        # Zombie AI state.
        self.zombie_activated = False
        self.zombie_path = []
        self.zombie_path_idx = 0
        self.zombie_move_accumulator = 0
        self.last_melee_tick = -999
        self.killed_by_zombie = False  # tracked for end-of-round conversion

        # Precomputed per-tick movement path (filled by the planning step
        # at start of execution; consumed tick-by-tick during execution).
        self.move_path = []
        self.path_tick_offset = 0

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
        movement orders. Used by grenade projectile spawn (the marine
        throws from where they'll be at the start of the phase)."""
        for o in reversed(self.orders):
            if o.order_type in MOVE_ORDER_TYPES:
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
