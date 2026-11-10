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
  It always exists; its default ``None`` is the ``[speeds]`` shamble
  cadence at the map's tile size (G1p-c), resolved by the zombie walk.
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
from simulation import turning as _turning
from simulation import unit_fixed
from simulation.generation import predefined_unit_attributes
from simulation.inventory import Inventory
from simulation.orders import (
    ORDER_MOVE_ATTACK, ORDER_FIRE, MOVE_ORDER_TYPES,
)
from simulation.factions import ZOMBIE_TEAM
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

#   (The spawn facing is turning.FACING_NORTH_Q16 = quantize(pi/2), in the
#   facing's own Q16.16 radian currency -- G5p-a1.)

# (_COMPASS_LABELS / _SECTOR_HALF DELETED - audit Patch A / A9, 2026-08-04:
# an 8-compass snap table with no reader anywhere in the repo.)


#: ``Unit.throw_reach_q`` when the throw named no reach (the v0.1 fixed one).
THROW_NO_REACH = -(2 ** 31)


class Unit:
    """Game entity: marine, zombie, or future creature.

    Position lives in physics-tile units. ``x`` and ``y`` are float;
    integer tile indices for matrix access come from ``tile_x`` /
    ``tile_y`` properties (which simply floor the float).

    ``team`` semantics (#119): any int; hostility = a different team. Marine
    teams count from 0 (the human controls ``Simulation.player_team``);
    zombies live on :data:`ZOMBIE_TEAM`. ``is_zombie`` is set from it here and
    is the ONLY zombie check anywhere — never a team literal. Conversion
    (marine killed by zombie) moves the unit to ``ZOMBIE_TEAM`` and sets
    ``is_zombie``, so it turns on its old squad. (A level file's legacy
    ``team = 1`` is mapped to ``ZOMBIE_TEAM`` by the level loader, the one
    door.)

    ``slot`` is the unit's fixed index within the team it was spawned into,
    assigned by ``Simulation.add_unit`` in spawn order, never reused and kept
    at death and conversion (``Simulation.roster``). -1 = not yet added.

    ``loadout_name`` names a ``[loadouts.*]`` row (``None`` = ``[marine]
    loadout``); ``env_profile`` an ``[env_profiles.*]`` row (``None`` = the
    species baseline). Zombies carry no loadout.

    Facing convention: Q16.16 radians (``facing_q``; ``facing`` the float
    view), 0=East, CCW positive. Default spawn: North. Written only by the
    turning law (G5p-a1, ``simulation.turning``).
    """

    def __init__(self, name: str, x: float, y: float, team: int = 0,
                 footprint: int = 3, species_id: str = "human", *,
                 loadout_name: str | None = None,
                 env_profile: str | None = None):
        # Deferred import: loadouts -> weapons -> config only, but keep the
        # table build lazy (it validates the whole weapon table).
        from simulation.loadouts import get_tables as _loadout_tables

        self.name = name
        self.team = int(team)
        self.is_zombie = (self.team == ZOMBIE_TEAM)
        # Roster slot within the spawn team (Simulation.add_unit assigns it).
        self.slot = -1
        self.loadout_name = None if self.is_zombie else (
            loadout_name if loadout_name is not None
            else _loadout_tables().default_loadout)
        self.env_profile_name = env_profile

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

        # Environment profile pointer: the named [env_profiles.*] row (a
        # suit, #119), else the species baseline (modifiers deferred).
        self.environment = (species.environment if env_profile is None
                            else _loadout_tables().env_profile(env_profile))

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

        # EXPOSURE (gameplay pack G4p, simulation.exposure): the focus-fire
        # level, one Q16 int per damage type (a fixed block, all 0), raised by
        # the hits of a round with an `expose`, decaying every tick, read at
        # the damage door. Zeroed at death and at conversion. SYNCED: hashed
        # in the unit digest while non-zero (field_ab_harness._unit_record).
        from simulation.exposure import new_block as _new_exposure_block
        self.exposure: list = _new_exposure_block()

        # Hidden Hartmann fields — data only; behaviour deferred (spec §13).
        self.awakened: bool = False

        # ---- Facing (G5p-a1: the turning law, simulation.turning) ---------
        # ``_facing_q`` (Q16.16 radians, 0=East CCW) is the SYNCED facing;
        # ``facing`` / ``facing_q`` are read-only views. ``facing_target_q``
        # is where the unit is turning to (persists until replaced) and
        # ``omega_q`` its angular velocity per tick. Only simulation.turning
        # writes them: the spawn here through ``turning.place`` (the one
        # constructor-time door), every later reason to face somewhere through
        # ``turning.set_target``, the turn itself in slot 1a. Default spawn
        # facing = North (quantize(pi/2)), matching the legacy "N" default.
        _turning.place(self, _turning.FACING_NORTH_Q16)

        # ---- The unit impulse law (G5p-a2, simulation.unit_impulse) -------
        # The accumulators ``imp_x`` / ``imp_y`` (Q16 gram-tiles per tick,
        # the remainder of J / m kept) and ``imp_l`` (angular), the in-place
        # sway ``swx`` / ``swy`` and the pushed velocity ``pvx`` / ``pvy``
        # (Q16 tiles per tick), the ``effort`` (Q16) and ``react_ticks``; the
        # shove's ``last_shove_tick`` cooldown and ``pending_shove`` latch;
        # ``mass_g``, body + kit in integer grams; ``impulse_books``, the
        # momentum books (diagnostic, never digested). All integers, all at
        # rest here. Written only through unit_impulse (``add`` is the one
        # entry). SYNCED: hashed in the unit digest while non-zero.
        from simulation import unit_impulse as _unit_impulse
        _unit_impulse.init_state(self)

        # ---- HP: current_hp from sampled vitality ------------------------
        # max_hp is removed; use effective_vitality(unit) from stats module.
        self.current_hp: float = float(self.base_stats.vitality)

        # ---- Legacy fields: kept unchanged --------------------------------

        # The zombie walk's cadence in ticks per fine tile. Replaces the
        # legacy zombie_speed_override monkey-patch. None (G1p-c) = the
        # [speeds] zombie_shamble_mps cadence at the map's tile size
        # (ai_zombie.update_zombies_tick resolves it), so a marine that turns
        # zombie shambles like every other; an int overrides it (tests, a
        # later per-unit stat). Marines never read it: the continuous path
        # walks by the [speeds] table, WEGO by its per-order dials.
        self.speed_ticks_per_tile = None

        # Order / planning state.
        self.orders = []
        self.current_order_type = ORDER_MOVE_ATTACK
        self.ap = [CFG.clock.ap_per_phase, CFG.clock.ap_per_phase]

        # Inventory booleans — BASE fields. Wiring into self.inventory deferred.
        # A non-zombie is armed by its [loadouts.*] row (#119; the default
        # row carries the pre-#119 [marine] numbers, byte-identical).
        _kit = None if self.is_zombie else _loadout_tables().loadout(
            self.loadout_name)
        # Hand grenades (G3p): ONE count per hand-grenade [ammo.*] row, a
        # fixed block (the rows of the family are the capacity), indexed by
        # the throw verb's ``kind`` (``WeaponsTables.hand_grenade_rows``).
        # ``has_grenade`` below is the FRAG slot of it (the legacy single
        # count: the old order path, the bots, the pickup hooks, the tests).
        from simulation.weapons import FRAG_ROW as _FRAG_ROW
        from simulation.weapons import get_tables as _weapon_tables
        _rows = _weapon_tables().hand_grenade_rows
        self._frag_slot = _rows.index(_FRAG_ROW) if _FRAG_ROW in _rows else -1
        self.grenade_counts = [
            0 if _kit is None else int(_kit.grenades.get(r, 0)) for r in _rows]
        # The throw's wind-up (G3p, Simulation._start_throw): the tick the
        # grenade leaves the hand (-1 = not winding up) and what was asked
        # for. Integers only; deliberately NOT in the synced digest (the
        # magazine state's precedent): the grenade's flight and the blast
        # are, and the hashed events catch a divergence.
        self.throw_release_tick = -1
        self.throw_kind = 0
        self.throw_reach_q = THROW_NO_REACH
        self.throw_fuse_ticks = 0
        self.has_explosive = 0 if _kit is None else _kit.explosives
        # Magazines carried, TOTAL including the one loaded (G2p; the loadout
        # row's `magazines`, 0 = spares untracked = infinite auto-reload).
        # Config-static like the two counts above; read only by
        # combat.mag_bind, and only for a weapon with mag_size > 0.
        self.magazines = 0 if _kit is None else _kit.magazines
        # ... counted for the loadout's OWN weapon only (a swapped-in weapon
        # keeps untracked spares): combat.magazines_carried.
        self.magazines_weapon = "" if _kit is None else str(_kit.weapon)

        # Weapon binding (mechanics/03 §4, W1; data-driven W4): which
        # [weapons.*] row this unit fires — its loadout row's `weapon` (#119;
        # the marine default is `[marine] loadout`'s row — Erik's feel-check
        # dial: set that row's weapon to "dragon_7", RESTART the game, hose a
        # room; construction-bound like the weapon tables, Ctrl+R alone does
        # not re-arm existing units).
        # Zombie melee stays on its ai_zombie path (NPC weapons migrate onto
        # rows later, mechanics/03 §7). Config-static, so deliberately NOT
        # in the synced digest (field_ab_harness.SYNCED_UNIT_FIELDS) — it
        # joins when loadouts can change at runtime.
        self.weapon_id = "" if _kit is None else str(_kit.weapon)

        # Combat state (used by marines mainly; harmless on zombies).
        self.last_fire_tick = -999
        self.fire_target = None

        # W3 ammo economy (mechanics/03 §4 mag_size/reload_seconds) — only
        # weapons with mag_size > 0 ever touch these (the GL-6 tier; the k5
        # keeps mag_size 0 = untracked = exactly pre-W3 behavior, so shipped
        # scenarios never move this state). current_mag: None = full mag,
        # lazily bound to the weapon row's mag_size at the first trigger
        # (combat.mag_bind via mag_gate — survives weapon_id swaps in tests);
        # spare_mags: the spare magazines left (G2p; bound with current_mag
        # from `magazines`, combat.SPARES_UNTRACKED = infinite);
        # reload_done_tick: the reload stall's end tick (-1 = none pending;
        # started when the mag empties or by Simulation.reload_intent,
        # combat.reload_begin). Deterministic derivations of
        # synced inputs (intents + tick), deliberately NOT in the synced
        # digest surface — the last_fire_tick precedent
        # (field_ab_harness.SYNCED_UNIT_FIELDS).
        self.current_mag = None
        self.spare_mags = -1
        self.reload_done_tick = -1

        # The MELEE verb (gameplay pack G6p, ``Simulation.melee_intent``):
        # the loadout's melee weapon row ("" = none: the verb does nothing),
        # the verb's latch (consumed in slot 4's phase A,
        # ``combat.process_melee_intents``) and the tick of the last blow (the
        # row's cadence; -1 = none yet; a zombie's bite keeps its own
        # ``last_melee_tick``). Intent state: deliberately NOT in the synced
        # digest (the shove's and the magazine's precedent); the blow itself
        # is (hp, status, push, events).
        self.melee_id = "" if _kit is None else str(_kit.melee)
        self.pending_melee = False
        self.last_blow_tick = -1
        # The flashlight bit (gameplay pack G7p, ``Simulation.set_flashlight``):
        # ON by default; SYNCED, hashed in the unit digest only while OFF
        # (absence-transparent). Nothing in the sim reads it (light as game
        # state is #81): the frame-light assembly draws the cone from it.
        self.flashlight_on = True

        # W4 SPRAY burst state (mechanics/03 §5): armed by
        # combat.start_spray_burst on a spray trigger; counted down by
        # combat.process_sprays (one cone-deposit tick per count).
        # spray_order holds the originating fire order so an interruption
        # (composed can_act False) can CONSUME it. Like the mag state above:
        # a deterministic derivation of synced inputs, deliberately NOT in
        # the synced digest surface; cleared at the round boundary.
        self.spray_ticks_left = 0
        self.spray_target = None
        self.spray_order = None

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

        # ---- OnePhaseWEGO timeline state (design §3) ----------------------
        # Time is the only currency, so every one of these is an ABSOLUTE tick
        # deadline on the ruleset's free-running clock — which is exactly what
        # lets them survive a round boundary untouched (§13, "invisible
        # seams"). Read ONLY by simulation.timeline and the OnePhaseWEGO
        # ruleset; inert under TwoPhaseWEGO / ContinuousRealtime, and outside
        # the synced digest surface for the same reason last_fire_tick is
        # (deterministic derivations of synced inputs — orders + tick).
        self.plan                = None   # the compiled Plan (simulation.timeline)
        self.plan_round          = -1     # round_index whose orders are queued
        self.busy_until_tick     = 0      # current action's duration runs to here
        self.gcd_until_tick      = 0      # global cooldown (§3)
        self.swap_cd_until_tick  = 0      # weapon swap's own non-global CD (§3)
        self.action_cd           = {}     # {action_name: until_tick}

        # Loadout (§15): primary + secondary, swapped freely for 0 cost except
        # the swap CD. `active_slot` indexes it; `weapon_id` above mirrors the
        # active entry so every shipped combat consumer keeps working unchanged
        # — that mirroring is deliberate: it means the whole combat path never
        # learns that loadouts exist. A slot may hold a weapon OR an item (§15);
        # an item is simply a LOBBED/PLACED weapon row.
        #
        # The quick-item BELT is the same idea one layer out: its entries are
        # action-registry item rows, and belt slots are hotbar slots (§15 — "one
        # system wearing two skins"), which is why nothing here is a separate
        # item type.
        if self.is_zombie:
            self.loadout = []
            self.belt = []
        else:
            secondary = str(getattr(CFG.marine, "secondary", "") or "")
            self.loadout = [self.weapon_id]
            if secondary:
                self.loadout.append(secondary)
            self.belt = list(getattr(CFG.marine, "belt", ()) or ())
        self.active_slot = 0

        # Overwatch (§9) — a STATE, not an order: it persists across rounds
        # until replaced. None = not on overwatch.
        self.overwatch_facing  = None
        self.overwatch_half_deg = None

        # Idle stance (§13): who has shot at this unit THIS ROUND
        # ({attacker_id: hit_count}, insertion-ordered), so an order-less unit
        # can return fire at its attackers without free-firing at everything
        # it sees. Written by damage.apply_packet (the one place every damage
        # source converges); cleared at the round boundary, which is what
        # makes "recent" mean something without threading a clock through
        # every damage call site.
        self.recent_attackers: dict = {}

    # Faction id — alias for team (spec §10.1). A property so it can never go
    # stale when conversion moves the unit to ZOMBIE_TEAM (#119).
    @property
    def faction_id(self) -> int:
        return int(self.team)

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

    @property
    def facing(self) -> float:
        """The facing in float radians (0=East, CCW positive): a READ-ONLY
        view of the synced ``facing_q`` (an exact n/65536 double) for the
        renderer, the pose layer and every float reader. No setter: a unit's
        facing is written only by the turning law (``simulation.turning``) --
        hand it a target (:meth:`face_towards`, ``turning.set_target``)."""
        return self._facing_q / 65536.0

    @property
    def facing_q(self) -> int:
        """The synced facing, Q16.16 radians in ``(-PI_Q16, PI_Q16]``."""
        return self._facing_q

    def face_towards(self, target_x: float, target_y: float) -> None:
        """Make the direction from the unit's current position toward
        ``(target_x, target_y)`` its facing TARGET (``turning.set_target``):
        the turning law turns it there in slot 1a. Called by the movement
        code each tick a unit takes a step (the zombie walk, the legacy WEGO
        and onephase plans), so the sprite tracks the direction of travel.
        No-op if the target is the current position.

        World Y increases downward; the facing convention is math-style
        (Y-up). The negation on dy converts between them. The angle is the
        pure-integer kit's (Q2-lift: math.atan2 is libm and differs at the
        last ULP across CRT/Python versions): the deltas quantized at the
        boundary, integer atan2. Deltas that BOTH quantize to zero
        (|d| < 1/131072 tiles) give 0 -- deterministic, and unreachable from
        real movement steps (>= ~1e-2 tiles).
        """
        dx = target_x - self.x
        dy = target_y - self.y
        if dx == 0 and dy == 0:
            return
        _turning.set_target(self, unit_fixed.atan2_q16(
            unit_fixed.quantize_scalar(-dy), unit_fixed.quantize_scalar(dx)))

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

    @property
    def has_grenade(self) -> int:
        """The FRAG count: the legacy single grenade count, the slot of
        ``grenade_counts`` that holds ``grenade_frag`` (G3p). Settable, so
        the old order path, the pickup hooks and the tests keep their
        arithmetic; a table with no frag row reads 0."""
        return self.grenade_counts[self._frag_slot] if self._frag_slot >= 0 else 0

    @has_grenade.setter
    def has_grenade(self, n: int) -> None:
        if self._frag_slot < 0:
            raise ValueError("this weapon table has no grenade_frag row to "
                             "hold has_grenade")
        self.grenade_counts[self._frag_slot] = int(n)

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
