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
- Float positions ``fxf`` / ``fyf`` are KEPT — the renderer interpolates
  between them for smooth animation. The architecture doc flags them
  as a future cleanup; out of scope for this migration.
"""
from __future__ import annotations

from config import CFG
from simulation.orders import (
    ORDER_MOVE_ATTACK, ORDER_FIRE, MOVE_ORDER_TYPES,
)


class Unit:
    """Game entity: marine, zombie, or future creature.

    ``team`` semantics: 0 = marine (player), 1 = zombie (enemy). The
    ``is_zombie`` flag mirrors this and is the source-of-truth check for
    "should AI take over this unit". Conversion (marine killed by
    zombie -> zombified at round end) flips both fields.
    """

    def __init__(self, name, cx, cy, team=0):
        co = CFG.display.coarse
        self.name = name
        self.team = team
        self.is_zombie = (team == 1)

        # Stable integer id, assigned by Simulation.add_unit. -1 means the
        # unit has not yet been added (constructor usage / tests).
        self.id = -1

        # Position: integer fine-tile (top-left of 3x3 unit block).
        # fxf/fyf are float positions used by the renderer for smooth
        # interpolation between ticks. They stay in sync with fx/fy.
        self.fx = cx * co
        self.fy = cy * co
        self.fxf = float(self.fx)
        self.fyf = float(self.fy)

        self.alive = True
        self.facing = "S"

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
    def cx(self):
        """Coarse tile X (3-tile blocks)."""
        return self.fx // CFG.display.coarse

    @property
    def cy(self):
        """Coarse tile Y."""
        return self.fy // CFG.display.coarse

    def center_fx(self):
        """Approximate fine-tile X at the unit's center (top-left + co/2)."""
        return self.fx + CFG.display.coarse // 2

    def center_fy(self):
        """Approximate fine-tile Y at the unit's center."""
        return self.fy + CFG.display.coarse // 2

    def get_center_px(self):
        """Pixel-space center using float position. Used by renderer/UI."""
        co_px = CFG.display.coarse_px
        ft = CFG.display.fine_tile_px
        return (int(self.fxf * ft + co_px / 2),
                int(self.fyf * ft + co_px / 2))

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
        """Return the fine-tile position the unit will reach after all
        queued movement orders. Used by grenade projectile spawn (the
        marine throws from where they'll be at the start of the phase)."""
        for o in reversed(self.orders):
            if o.order_type in MOVE_ORDER_TYPES:
                return o.target_fx, o.target_fy
        return self.fx, self.fy

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
