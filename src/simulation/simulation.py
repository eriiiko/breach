"""Simulation — the central facade for game logic.

Owns everything the world contains (map, units, projectiles, physics,
recorder, RNG) and exposes the small, well-defined API that both
``main.py`` (human play) and a future ``train.py`` (AI rollouts) talk to:

    sim = Simulation(level_data, seed=42)
    sim.add_unit(unit, position)
    sim.apply_action(unit_id, order)
    sim.step()                 # advance one tick
    state = sim.get_state()    # snapshot for renderer / AI

Architectural reasoning (see ``docs/architecture.md`` §2): the simulation
is the **only** place game state is mutated. Renderer reads
``sim.get_state()`` and never writes back. AI training rollouts call
``sim.step()`` in a tight loop, never touch pause, read ``get_reward``
and ``is_terminal`` for the Gymnasium contract.

Determinism / RNG plumbing
--------------------------
A single :class:`numpy.random.Generator` lives on ``sim.rng``. It is
plumbed through the three known nondeterminism sites:

- :func:`simulation.combat.fire_burst` — bullet cone offsets
- :func:`simulation.physics.add_explosion_smoke` — per-tile noise
- (deferred) ``Raycaster.cast_source`` — fire jitter. For now fire
  light sources default to ``jitter = 0.0`` (the natural smoke
  advection creates the flicker we want); the C++ raycaster keeps its
  internal seeding. Revisit when training begins.

The same seed gives the same trajectory for any sequence of
``apply_action / step`` calls — assuming the C++ physics is
deterministic, which the existing test_simulation determinism check
exercises.

Pause semantics
---------------
``set_paused(True)`` makes :meth:`step` a no-op (tick does not advance).
Orders may still be added / modified during pause — they take effect on
resume. The simulation auto-pauses at the end of each phase (tick
``ticks_per_phase``) and at the end of each round (tick
``ticks_per_round``). AI training simply never calls ``set_paused`` and
never sees a halt.

Tick events
-----------
Each ``step()`` clears ``sim.tick_events`` first, then any per-tick
visual signals (a shot fired, a unit hit, an explosion) are appended for
the caller (renderer) to read after the tick returns. The simulation
does NOT track decay/fade — those live in the renderer's effect queue.
See :mod:`simulation.events` for the event dataclasses.

Save / load
-----------
``get_state()`` returns enough to support future save/load via pickle —
not implemented yet (out of scope). The API will not change when it is
added; the snapshot contract is forward-compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import CFG
from simulation.ai_zombie import update_zombies_tick, convert_marines_to_zombies
from simulation.generation import predefined_unit_attributes
from simulation.species import get_species
from simulation.combat import (
    apply_temperature_ignition,
    process_door_explosives, process_shooting,
    Projectile, Shot,
)
# The two shipped physics->unit couplings — rows in the mechanics/05 coupling
# table — are invoked through the exchange module (P1 refactor). Imported as
# BARE NAMES (not module-qualified calls) deliberately: the call sites read
# this module's own binding, which keeps instrumentation that rebinds
# simulation.simulation.apply_environmental_damage working
# (tests/_xarch_liveheat_dump.py — the case-2 divergence instrument).
# Execution positions are UNCHANGED (heat at step 9c, blast at the grenade
# fuse-out below); the consolidated named EXCHANGE-READ slot is a later patch.
from simulation.exchange import (  # noqa: F401 (apply_blast_damage: legacy re-export — the executor calls it now)
    apply_blast_damage, apply_environmental_damage, apply_poison_dose,
    apply_teargas_blind, apply_wave_push,
)
from simulation.events import (  # noqa: F401 (ExplosionEvent: legacy re-export — emitted by the executor now)
    DoorDestroyedEvent, ExplosionEvent, WallDestroyedEvent,
)
from simulation.gamemap import GameMap, MAT_DOOR
from simulation.movement import FootprintSamples, default_speed
from simulation.orders import (
    DET_START_PHASE1, DET_BETWEEN_PHASES, DET_END_PHASE2,
    ORDER_GRENADE, ORDER_EXPLOSIVE, ORDER_FIRE, ORDER_MOVE_ATTACK,
    ORDER_MOVE_COVER, ORDER_SPRINT, MOVE_ORDER_TYPES,
)
from simulation.physics import apply_explosion, add_explosion_smoke  # noqa: F401 (legacy re-export)
# The payload EXECUTOR (mechanics/03 §4, W3): the grenade fuse-out below
# executes its round's payload row through this. Imported as a BARE NAME
# deliberately (the apply_environmental_damage pattern above): the call site
# reads this module's own binding, so replica tests can rebind
# simulation.simulation.execute_payload to the pre-W3 inline triple.
from simulation.payloads import execute_payload
from simulation.physics_runner import PhysicsRunner
from simulation.field_edit import EditQueue, FieldEdit
from simulation.recorder import PhysicsRecorder
from simulation.status import composed_flags, tick_statuses
# Weapon/ammo/payload data tables (mechanics/03 §4, W1): the facade owns a
# bundle, rebuilt fresh at every construction/reset (the GameMap material/gas
# table pattern) so it always reflects the current CFG.
from simulation.weapons import rebuild_tables as rebuild_weapon_tables

try:
    from pathfinding import astar
    HAS_PATHFINDING = True
except ImportError:
    HAS_PATHFINDING = False


# ---------------------------------------------------------------------------
# Snapshot returned by get_state(). Kept lightweight — references the live
# arrays for cheap read access. AI training serializes via pickle when needed.
# ---------------------------------------------------------------------------
@dataclass
class SimState:
    """Read-only snapshot of the simulation, returned by :meth:`Simulation.get_state`.

    Holds references (not deep copies) into the live simulation. Safe for
    the renderer to read each frame. AI training that needs a permanent
    snapshot must pickle / deepcopy explicitly.
    """
    gmap: GameMap
    units: list
    projectiles: list
    tick: int
    phase: int
    paused: bool


def _ticks_per_tile(order_type):
    """Movement-order speed lookup (lifted from game.py:155-163)."""
    if order_type == ORDER_MOVE_ATTACK:
        return CFG.movement.marine_attack_ticks_per_tile
    elif order_type == ORDER_MOVE_COVER:
        return CFG.movement.marine_cover_ticks_per_tile
    elif order_type == ORDER_SPRINT:
        return CFG.movement.marine_sprint_ticks_per_tile
    return CFG.movement.marine_attack_ticks_per_tile


class Simulation:
    """Headless, deterministic gameplay engine.

    A ``Simulation`` owns its world state for its lifetime. Use
    :meth:`reset` to start a new rollout from the same level on the same
    instance — avoids re-allocating the numpy grids the physics solvers
    bind to.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self, level_data, seed: Optional[int] = None,
                 breach_physics=None, enable_recorder: bool = True):
        """Construct a fresh simulation from a level.

        Parameters
        ----------
        level_data
            A :class:`level_loader.LevelData` (the CSV-loaded ship).
        seed
            Seed for the per-simulation RNG (combat / explosion noise).
            Pass an integer for deterministic rollouts; ``None`` picks
            a fresh seed from the OS each construction.
        breach_physics
            The compiled C++ physics module. Pass ``None`` to skip
            the physics-runner — useful for unit tests that exercise
            only the order / combat path. Most callers pass the
            ``breach_physics`` import.
        enable_recorder
            Whether to allocate the ring-buffer recorder (large memory
            cost). Default on for human play / debugging; AI training
            should pass False.
        """
        self.level = level_data
        self._enable_recorder = enable_recorder
        self._bp = breach_physics

        # Constants (cached so reset() doesn't re-read CFG every time)
        self._ticks_per_phase = CFG.clock.ticks_per_phase
        self._ticks_per_round = CFG.clock.ticks_per_round
        self._phases_per_round = CFG.clock.phases_per_round
        self._tps = CFG.clock.ticks_per_second

        # Build the world & RNG.
        self._reset_internal(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-initialise the simulation from the same level.

        Drops all units, projectiles, recorder history, and gameplay
        state. Re-seeds the RNG (so a fresh ``seed`` deterministically
        re-runs); equivalent to constructing a new ``Simulation`` but
        cheaper because we reuse the C++ solvers and avoid GC churn on
        the big grids.
        """
        self._reset_internal(seed)

    def _reset_internal(self, seed: Optional[int]) -> None:
        # Fresh map (allocates grids), fresh RNG, fresh entity lists.
        self.gmap = GameMap(self.level)
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        # Weapon/ammo/payload tables (mechanics/03 §4, W1) — rebuilt from the
        # live CFG at every reset, exactly like GameMap rebuilds the material/
        # gas tables above. Config-static data; Ctrl+R alone does NOT rebuild
        # (engine/12 §5 — the construction-bound precedent).
        self.weapons_tables = rebuild_weapon_tables()

        self.units: List = []
        self.projectiles: List = []
        # In-flight kinetic rounds (W2 unified march, mechanics/03 §2):
        # BulletInFlight entities for rounds whose range outruns their
        # per-tick speed. Every shipped small-arm resolves same-tick (speed
        # authored >= range), so this list is EMPTY in-game until slow
        # archetypes (plasma) ship — the machinery is exercised by tests.
        # Advanced in tick slot 2 (after grenades, before movement).
        self.bullets: List = []
        self.shots: List = []          # legacy tracer list; renderer reads it
        self.tick_events: List = []    # cleared each step()

        # Field-edit queue (engine/13): the canonical WRITE path. Weapon / fire /
        # explosion phases enqueue FieldEdits during the tick via :meth:`edit`;
        # :meth:`step` flushes the whole queue once — in a deterministic
        # stable-sorted order — just before the physics solvers run, so the
        # solvers see the settled net deposit. Recreated each reset so a fresh
        # rollout starts with an empty queue.
        self.edit_queue = EditQueue()

        self.tick = 0                  # tick within the round (0 .. ticks_per_round - 1)
        self.phase = 0                 # 0 = phase 1, 1 = phase 2
        self.paused = True             # start paused — the player plans first
        self.turn_number = 1
        self.real_time = 0.0           # wall-clock seconds since round start

        # Internal: have the start-of-round door-explosive callbacks fired?
        # Set on first step() after a reset; we don't fire on construction.
        self._fired_start_p1 = False
        self._fired_between = False
        self._fired_end_p2 = False

        # Physics runner (created once, re-bound on reset only if bp present).
        if self._bp is not None:
            # Always build a fresh runner so per-session params are clean.
            self.physics_runner = PhysicsRunner(self._bp)
            # Wire the C++ PhysicsEngine into the gmap for the C++ stamp_units
            # path (the runner owns the engine). A bare GameMap with no engine
            # bound falls back to the Python reference path automatically.
            self.gmap.bind_physics_engine(self.physics_runner.engine)
        else:
            self.physics_runner = None

        # Recorder ring buffer (owns ~80 MB at default 1200-frame capacity).
        if self._enable_recorder:
            self.recorder = PhysicsRecorder(
                fh=self.gmap.material.shape[0],
                fw=self.gmap.material.shape[1],
                capacity=CFG.recorder.capacity,
            )
        else:
            self.recorder = None

        # Next id to assign in add_unit.
        self._next_unit_id = 0

    # ------------------------------------------------------------------
    # Unit management
    # ------------------------------------------------------------------
    def add_unit(self, unit, position=None) -> int:
        """Register ``unit`` with the sim, assign it a stable id.

        ``position`` is optional — if provided as ``(x, y)`` in
        physics-tile coords, the unit is teleported there before its
        first tick. Otherwise the unit keeps its constructor-set position.

        Re-samples the unit's BaseStats using ``self.rng`` so that spawns
        are deterministic when the Simulation was constructed with a seed.
        This overwrites the random-seeded stats drawn at Unit construction.
        """
        if position is not None:
            x, y = position
            unit.x = float(x)
            unit.y = float(y)

        # Deterministic predefined attributes (ingress door 2 — quantized
        # species means; see generation.py). The sim RNG no longer touches
        # spawn stats: the draft MVN sampler was BLAS/LAPACK-backed and
        # cross-machine nondeterministic (docs/lenovo_dev_setup.md §8b).
        species = get_species(getattr(unit, "species_id", "human"))
        base_stats, mass, base_speed = predefined_unit_attributes(species)
        unit.base_stats = base_stats
        unit.mass = mass
        unit.base_speed = base_speed
        unit.current_hp = float(base_stats.vitality)

        unit.id = self._next_unit_id
        self._next_unit_id += 1
        self.units.append(unit)
        return unit.id

    def get_unit(self, unit_id: int):
        """Linear lookup by id. ``None`` if not found."""
        for u in self.units:
            if u.id == unit_id:
                return u
        return None

    def marines(self):
        """All living marines (team 0)."""
        return [u for u in self.units if u.team == 0 and u.alive]

    def zombies(self):
        """All living zombies (team 1)."""
        return [u for u in self.units if u.team == 1 and u.alive]

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def apply_action(self, unit_id: int, order) -> bool:
        """Append an order to ``unit_id``'s queue.

        Returns ``True`` on success, ``False`` if the order failed
        validation (no AP, no inventory, blocked tile). Lifted-and-
        simplified version of ``game.py:_place_order`` (lines 1465-1522).
        The caller (input layer) can surface failures as a UI toast.

        Movement orders also trigger an immediate
        :meth:`_compute_player_paths` recompute for the affected unit so
        the planning overlay and execution can both read the same path.
        """
        u = self.get_unit(unit_id)
        if u is None or not u.alive:
            return False
        if u.is_zombie:
            # Zombies don't take player orders. (Inventory cook-off is
            # handled by physics, not order placement.)
            return False

        phase = order.phase
        ot = order.order_type

        if ot in MOVE_ORDER_TYPES:
            # Validate target tile is passable for a 3x3 unit footprint.
            if not self.gmap.is_passable_block(order.target_fy, order.target_fx):
                return False
            order.ap_cost = 0  # movement is free
            u.orders.append(order)
            # Pre-compute the path so the overlay can show it. Per the
            # plan's decision D (precomputed + per-tick): we lay down the
            # tick-by-tick trajectory here. It's read at execute time.
            self._compute_player_paths()
            return True

        # Order AP costs come off the weapon rows (mechanics/03 §4, W1
        # re-home — same literal costs the old CFG.weapons.* keys held).
        weapon_rows = self.weapons_tables.weapons.by_name

        if ot == ORDER_GRENADE:
            ap_cost = weapon_rows["hand_grenade"].ap_cost
            if u.get_ap(phase) < ap_cost:
                return False
            if u.has_grenade <= 0:
                return False
            order.ap_cost = ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)
            u.has_grenade -= 1
            return True

        elif ot == ORDER_EXPLOSIVE:
            ap_cost = weapon_rows["breach_charge"].ap_cost
            if u.get_ap(phase) < ap_cost:
                return False
            if u.has_explosive <= 0:
                return False
            order.ap_cost = ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)
            u.has_explosive -= 1
            return True

        elif ot == ORDER_FIRE:
            # W2: the cost comes off the UNIT'S weapon row (the k5 for every
            # shipped marine — same literal 1); a unit with no ranged weapon
            # ("" — zombies never reach here anyway) can't take fire orders.
            weapon_id = getattr(u, "weapon_id", "")
            if not weapon_id:
                return False
            ap_cost = weapon_rows[weapon_id].ap_cost
            if u.get_ap(phase) < ap_cost:
                return False
            order.ap_cost = ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)
            return True

        return False

    def undo_last_order(self, unit_id: int) -> bool:
        """Pop the most recent order off ``unit_id``'s queue and refund.

        Lifted from ``game.py:1409-1418`` (Backspace handler). Refunds
        AP and inventory; returns ``True`` if an order was popped,
        ``False`` if there was nothing to undo or the unit doesn't exist.
        """
        u = self.get_unit(unit_id)
        if u is None or not u.orders:
            return False
        removed = u.orders.pop()
        if removed.ap_cost > 0:
            u.ap[removed.phase] += removed.ap_cost
        if removed.order_type == ORDER_GRENADE:
            u.has_grenade += 1
        elif removed.order_type == ORDER_EXPLOSIVE:
            u.has_explosive += 1
        if removed.order_type in MOVE_ORDER_TYPES:
            # Movement path may have changed; recompute.
            self._compute_player_paths()
        return True

    def get_legal_actions(self, unit_id: int) -> list:
        """Stub for the AI training contract — minimal v1.

        Returns ``[]`` for now: the AI is not training against this
        environment yet, and a complete enumeration of legal targets
        per (mode x phase x tile) is large and not yet needed. When
        train.py is wired up, replace this with the proper enumeration
        (movement → reachable tiles, fire → visible enemies in range,
        grenade → tiles within throw range, etc.).

        Callers can still call ``apply_action`` and check the return
        value for "was this order accepted?" — that doubles as a
        legality check in v1.
        """
        return []

    # ------------------------------------------------------------------
    # Path computation (precomputed at order-time per locked decision D)
    # ------------------------------------------------------------------
    def _compute_player_paths(self) -> None:
        """Fill each marine's ``move_path`` with per-tick positions.

        Lifted from ``game.py:_compute_player_paths`` (lines 1566-1639).
        Each marine's path is the per-tick trajectory through all their
        movement waypoints across both phases of the round. Filled with
        the static end position once they've reached the last waypoint.

        Called from :meth:`apply_action` after every movement order is
        placed or undone — so the renderer's overlay can always show a
        live preview, and execution reads the same data.
        """
        tpp = self._ticks_per_phase
        h = self.gmap.material.shape[0]
        w = self.gmap.material.shape[1]
        gmap = self.gmap

        for u in self.units:
            if u.team != 0 or not u.alive:
                continue
            u.move_path = []
            u.path_tick_offset = 0
            current_x, current_y = u.tile_x, u.tile_y
            fp = u.footprint

            def is_blocked(x, y, _gmap=gmap, _fp=fp):
                return not _gmap.is_passable_block(y, x, _fp)

            for phase in range(self._phases_per_round):
                move_orders = [o for o in u.orders
                               if o.phase == phase and
                               o.order_type in MOVE_ORDER_TYPES]

                if move_orders:
                    tile_path = []
                    cx, cy = current_x, current_y
                    speed = _ticks_per_tile(move_orders[0].order_type)

                    for mo in move_orders:
                        speed = _ticks_per_tile(mo.order_type)
                        if HAS_PATHFINDING:
                            segment = astar(cx, cy, mo.target_fx, mo.target_fy,
                                            is_blocked, w, h)
                            if segment and len(segment) > 1:
                                tile_path.extend(segment[1:])
                        else:
                            tile_path.append((mo.target_fx, mo.target_fy))
                        cx, cy = mo.target_fx, mo.target_fy

                    if tile_path:
                        tick_positions = []
                        prev_x, prev_y = float(current_x), float(current_y)
                        for tile_x, tile_y in tile_path:
                            # Terrain cadence (mobility design §4.1): the §4
                            # area-average mobility over the destination
                            # footprint scales the base order cadence ``speed``
                            # into the ticks this tile-step actually costs (a
                            # furniture tile is 2.5x slower). A* is speed-blind;
                            # this composes the multiplier at execution only.
                            samples = FootprintSamples(
                                mobility=gmap.footprint_mobility(tile_y, tile_x, fp))
                            step_ticks = default_speed(samples, speed)
                            for st in range(step_ticks):
                                frac = (st + 1) / step_ticks
                                ix = prev_x + (tile_x - prev_x) * frac
                                iy = prev_y + (tile_y - prev_y) * frac
                                tick_positions.append((ix, iy))
                            prev_x, prev_y = float(tile_x), float(tile_y)
                        for _ in range(tpp - len(tick_positions)):
                            tick_positions.append((prev_x, prev_y))
                        u.move_path.extend(tick_positions[:tpp])
                        last_idx = min(len(tick_positions), tpp) - 1
                        current_x = int(round(tick_positions[last_idx][0]))
                        current_y = int(round(tick_positions[last_idx][1]))
                    else:
                        for _ in range(tpp):
                            u.move_path.append((float(current_x), float(current_y)))
                else:
                    for _ in range(tpp):
                        u.move_path.append((float(current_x), float(current_y)))

    # ------------------------------------------------------------------
    # Pause + phase queries
    # ------------------------------------------------------------------
    def is_paused(self) -> bool:
        return self.paused

    def set_paused(self, value: bool) -> None:
        self.paused = bool(value)

    def get_tick(self) -> int:
        return self.tick

    def get_phase(self) -> int:
        return self.phase

    def get_state(self) -> SimState:
        return SimState(
            gmap=self.gmap,
            units=self.units,
            projectiles=self.projectiles,
            tick=self.tick,
            phase=self.phase,
            paused=self.paused,
        )

    def orders_for_phase(self, phase) -> dict:
        """Per-unit waypoint lists for the renderer's overlay.

        Returns ``{unit_id: [(x, y), (x2, y2), ...]}``. The first waypoint
        is where the unit will be at the *start* of ``phase`` — that's
        the current position for Phase 1, and the planned end of Phase 1
        for Phase 2. Subsequent entries are the targets of each move
        order in that phase. Empty for units with no movement.
        """
        out = {}
        for u in self.units:
            if u.team != 0 or not u.alive:
                continue
            start = (u.tile_x, u.tile_y) if phase == 0 else \
                    u.get_planned_pos_after_phase(phase - 1)
            waypoints = [start]
            for o in u.orders:
                if o.phase == phase and o.order_type in MOVE_ORDER_TYPES:
                    waypoints.append((o.target_fx, o.target_fy))
            if len(waypoints) >= 2:
                out[u.id] = waypoints
        return out

    # ------------------------------------------------------------------
    # AI training hooks
    # ------------------------------------------------------------------
    def get_reward(self, unit_id: int) -> float:
        """Stub — returns 0.0. Subclass / wrap to compute per-agent reward."""
        return 0.0

    def is_terminal(self) -> bool:
        """End-of-round or one side wiped out.

        Returns True if:
        - the round is complete (``tick == ticks_per_round``), OR
        - all marines are dead, OR
        - all zombies are dead.

        AI training treats this as an episode boundary.
        """
        if self.tick >= self._ticks_per_round:
            return True
        any_marine = any(u.team == 0 and u.alive for u in self.units)
        any_zombie = any(u.team == 1 and u.alive for u in self.units)
        # If we have neither, the simulation is degenerate; treat as terminal.
        if not any_marine and not any_zombie:
            return True
        # If the player had marines and they're all dead → terminal.
        if not any_marine and self.units:
            return True
        return False

    # ------------------------------------------------------------------
    # Field-edit enqueue API (engine/13 — the canonical WRITE primitive)
    # ------------------------------------------------------------------
    def edit(self, field_edit: FieldEdit) -> None:
        """Enqueue one :class:`FieldEdit` for this tick's flush.

        The single entry point any system uses to write a continuous field
        (smoke / atmosphere / wave_source / fire / heat / future gases). Nothing
        is applied here — :meth:`step` flushes the queue in deterministic
        stable-sorted order before the physics solvers run. Topology-changing
        edits (wall destruction) are NOT FieldEdits; they stay structural.
        """
        self.edit_queue.enqueue(field_edit)

    # ------------------------------------------------------------------
    # Tick loop — the core simulation step
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance the simulation by one tick.

        No-op while paused. Auto-pauses at phase boundary (tick ==
        ticks_per_phase) and at end of round (tick == ticks_per_round)
        for the human player's planning UI. AI training rollouts ignore
        pause (they never call set_paused).

        Tick ordering — load-bearing! Matches the legacy
        ``Game._process_tick`` (game.py:1696-1747), with the status pass
        (2b) added at the top of the unit-simulation section (P3):
          1. Clear tick_events (this tick is a fresh slate)
          2. Update projectiles (advance position, detonate)
          2b. Tick unit statuses (durations count down; DoT/HoT emit)
          3. Update player movement (read precomputed path)
          4. Process shooting (auto-fire on move, fire orders)
          5. Zombie AI
          6. Re-stamp obstacles (so physics sees the new unit positions)
          7. Physics step (atmosphere, smoke, fire)
          8. Door explosives at phase boundary or end of round
          9. Process burn-through walls (fire destroyed)
         10. Increment tick + auto-pause check
        """
        if self.paused:
            return

        # 1. Fresh event buffer.
        self.tick_events.clear()

        # On the very first tick of a round, fire the start-of-P1 explosives.
        if self.tick == 0 and not self._fired_start_p1:
            process_door_explosives(
                self.gmap, self.edit_queue, self.units, DET_START_PHASE1, self.rng,
                events=self.tick_events,
            )
            self._fired_start_p1 = True
            # Initialise per-unit path offsets for this round.
            for u in self.units:
                if u.team == 0:
                    u.path_tick_offset = 0
            # Stamp initial unit positions (legacy: done in _start_execution).
            self.gmap.stamp_units(self.units)

        # 2. Update projectiles.
        self._update_projectiles()

        # 2b. Tick unit statuses/conditions (mechanics/06 §4) — the TOP of
        # the unit-simulation section, per the ch. 05 §4 pipeline (phase 3:
        # "statuses tick; AI/orders; attacks resolve; movement"). Anchored
        # AFTER projectiles — an in-flight grenade is a world object whose
        # fuse-out is an ARRIVAL (ch. 05 P3 travel time), not a unit action —
        # and BEFORE every unit decision this tick, so movement / shooting /
        # zombie AI all see post-status composed flags and post-DoT hp (a
        # unit killed by its burning at the tick top neither moves nor acts
        # this tick). Consequence for triggers (P4): a status applied at or
        # before this point (projectile blast, exchange-read couplings)
        # suppresses for duration_ticks INCLUDING this tick; one applied
        # later in the tick (shooting, melee) starts next tick. DoT/HoT
        # packets flow through damage.apply_packet into tick_events —
        # synced, digest-hashed, in emission order.
        tick_statuses(self.units, events=self.tick_events)

        # 3. Update player movement.
        self._update_player_movement()

        # 4. Process shooting. W2: dispatches each shooter's weapon row by
        # archetype (projectile burst / hitscan beam); rounds that outrange
        # their per-tick speed land on self.bullets (advanced in slot 2).
        # W3: the edit queue rides along so payload rounds (GL-6) can
        # deposit their detonation; the mag/reload economy gates triggers.
        process_shooting(self.gmap, self.units, self.tick,
                         self.shots, self.real_time, self.rng,
                         events=self.tick_events, bullets=self.bullets,
                         queue=self.edit_queue)

        # 5. Zombie AI.
        update_zombies_tick(self.gmap, self.units, self.tick)

        # 6. Re-stamp obstacles.
        self.gmap.stamp_units(self.units)

        # 6b. Flush the field-edit queue (engine/13). The weapon / fire /
        # explosion phases above enqueued their FieldEdits (smoke / atmosphere /
        # wave_source / fire / heat deposits) via :meth:`edit`; we apply them ALL
        # here, in one deterministic stable-sorted pass, BEFORE the physics
        # solvers run — so the solvers advect / propagate the settled NET deposit
        # for this tick (a laser burn-off and a grenade cloud issued the same
        # tick both land before smoke advection). This is the single RNG consumer
        # for noise>0 edits, drawing from the seeded sim.rng in sorted order.
        self.edit_queue.flush(self.gmap, self.rng)

        # 7. Physics.
        sim_time_per_tick = 1.0 / float(self._tps)
        destroyed = []
        if self.physics_runner is not None:
            destroyed = self.physics_runner.step(self.gmap, sim_time_per_tick)

        # 9. Process fire burn-through walls.
        for (yy, xx) in destroyed:
            mat = int(self.gmap.material[yy, xx]) if (0 <= yy < self.gmap.material.shape[0]
                                                      and 0 <= xx < self.gmap.material.shape[1]) else -1
            self.gmap.destroy_wall(yy, xx)
            if mat == MAT_DOOR:
                self.tick_events.append(DoorDestroyedEvent(pos=(yy, xx)))
            else:
                self.tick_events.append(WallDestroyedEvent(pos=(yy, xx)))

        # 9b. Over-pressure wall failure (ch.04 §5) — the emergent pressure
        # relief valve. After physics settles, any wall holding a differential
        # above its material's burst_threshold fails and vents; over-pressured
        # clusters self-breach in a chain across ticks. Mirrors fire
        # burn-through: scan returns tiles, destroy_wall does the topology edit.
        # Capped per tick so a mistuned threshold can't nuke the whole ship.
        if getattr(CFG.physics, "burst_enabled", True):
            cap = int(getattr(CFG.physics, "burst_max_per_tick", 16))
            for (yy, xx) in self.gmap.find_burst_walls(max_pops=cap):
                mat = int(self.gmap.material[yy, xx])
                self.gmap.destroy_wall(yy, xx)
                if mat == MAT_DOOR:
                    self.tick_events.append(DoorDestroyedEvent(pos=(yy, xx)))
                else:
                    self.tick_events.append(WallDestroyedEvent(pos=(yy, xx)))

        # 9c. Unit heat damage (engine/06 §4, proposal §4.2/§6 step 4a). The
        # SECOND consumer of the per-tick `heat` deposit (the C++ heat ->
        # temperature conversion inside physics_runner.step() is the first):
        # each living unit samples the already-occluded `heat` buffer at its
        # footprint and takes radiant damage if the felt temperature pushes
        # past its tolerance band. Runs AFTER physics fills `heat`, BEFORE the
        # recorder snapshot (so the snapshot captures post-damage HP) and BEFORE
        # the end-of-tick heat clear below — its existence is precisely what
        # makes wiping `heat` correct (a reader finally consumes it).
        # This is the `heat | max` coupling row (mechanics/05 §1;
        # exchange.COUPLING_TABLE[0]) at its legacy pipeline position.
        if self.physics_runner is not None:
            apply_environmental_damage(
                self.units, self.gmap, self._tps,
                events=self.tick_events,
            )

        # 9c2. Wave impulse push + KNOCKED_DOWN (mechanics/05 §1 wave_p|grad
        # row; mechanics/06 §4 trigger — exchange.COUPLING_TABLE[2]). The
        # first coupling row born INTO the table (P4). Reads the post-physics
        # wave_p at each living unit's footprint: dv = k_push*(-grad)/mass —
        # the nudge displaces unit.x/y (wall-clamped) and dv above
        # threshold*stability lays the unit KNOCKED_DOWN (prone, no move/act,
        # refresh-stacked get-up timer). WITHIN-TICK EXCHANGE ORDER
        # (documented contract): heat damage (9c) FIRST, then the push — a
        # unit the heat row kills this tick is a corpse and is not displaced.
        # Runs BEFORE the recorder snapshot (it captures post-push positions)
        # and before the end-of-tick heat clear (irrelevant to wave_p, kept
        # for the fixed order). The moved position reaches the physics next
        # tick through the step-6 re-stamp. A status applied here starts
        # suppressing next tick (the step-2b P3 trigger-position semantics).
        if self.physics_runner is not None:
            apply_wave_push(self.units, self.gmap, self._tps)

        # 9c3. Gas coupling rows (mechanics/05 §1 gas[teargas] / gas[poison]
        # — exchange.COUPLING_TABLE[3..4], W3). Read the post-physics gas
        # planes at each living unit's footprint: teargas above threshold
        # applies BLINDED (snap-cone fire from next tick — the step-2b
        # trigger-position semantics); poison above threshold emits one
        # POISON packet per tick through the mechanics/06 pipeline (zombies
        # immune). WITHIN-TICK EXCHANGE ORDER (documented contract): heat
        # (9c), push (9c2), teargas, poison. Lazy: an all-zero plane is one
        # integer .any() and out — every gas-free trajectory (the canonical
        # golden included) is bit-identical to pre-W3. No RNG.
        if self.physics_runner is not None:
            apply_teargas_blind(self.units, self.gmap)
            apply_poison_dose(self.units, self.gmap, self._tps,
                              events=self.tick_events)

        # 9d. Ignition from temperature (engine/06 §4, proposal §6 step 4b). The
        # READ side of the temperature substrate: the C++ TemperatureSolver
        # (convert -> conduction -> cooling) ran inside physics_runner.step()
        # above and filled `temperature`; here each FLAMMABLE tile whose
        # `temperature` has crossed its (Q16.16-quantized) `ignition_temp` AND
        # has O2 (the same air-side-neighbour atmosphere check the fire uses) is
        # ignited via `fire = max(fire, ignition_seed)`. With the cellular spread
        # DELETED (fire_design_proposal §1), this radiation -> heat -> temperature
        # -> ignition path is now the SOLE way fire spreads tile-to-tile. Reads
        # `temperature` (gather) + `atmosphere`, writes `fire`; deterministic, no
        # RNG. With no sim heat sources active `temperature` is ~0, so it is
        # dormant. Slotted alongside the unit-heat-damage consumer (§6 step 4),
        # after the temperature passes, before the end-of-tick heat clear.
        if self.physics_runner is not None:
            fire_cfg = getattr(CFG.physics, "fire", None)
            ignition_seed = float(getattr(fire_cfg, "ignition_seed", 0.1))
            # Reuse the fire's O2 (pressure) survival threshold so a tile cannot be
            # ignited into a state the next fire step would immediately suffocate.
            # config [physics.fire].o2_threshold (0.60) mirrors the feedback P_min.
            o2_threshold = float(getattr(fire_cfg, "o2_threshold", 0.60))
            apply_temperature_ignition(self.gmap, o2_threshold, ignition_seed)

        # Recorder snapshot.
        if self.recorder is not None:
            self.recorder.record(self.gmap, self.tick, self.real_time, self.units)

        # Clear the per-tick `heat` deposit — END OF TICK, AFTER every heat
        # consumer (engine/06 §1.3/§6 step 7). `heat` is a per-tick deposit
        # buffer, not a cross-tick accumulator; it is wiped once both readers
        # have run — the C++ heat->temperature conversion inside
        # physics_runner.step() AND the unit heat damage above (plus the render
        # glow sample) — so the next frame's ray pass deposits into a clean
        # buffer. The clear moved here from PhysicsRunner.step() (STEP A) so the
        # downstream unit-damage consumer sees the pre-clear value. In-place
        # (never reassigned) so any C++ view of the buffer stays valid.
        if self.physics_runner is not None:
            self.gmap.heat.fill(0)

        # Expire visual shot tracers (legacy fade-out behaviour).
        if self.shots:
            self.shots = [s for s in self.shots if
                          self.real_time - s.time < s.duration]

        # 10. Advance tick + check for auto-pause boundaries.
        self.tick += 1
        self.real_time += sim_time_per_tick

        # Phase 1 → Phase 2 boundary: fire between-phase explosives and
        # advance the phase counter. NO auto-pause — the round plays
        # through both phases smoothly in one execution. The split is a
        # mental planning aid for the player, not a sim interruption.
        if (self.tick == self._ticks_per_phase
                and not self._fired_between):
            process_door_explosives(
                self.gmap, self.edit_queue, self.units, DET_BETWEEN_PHASES, self.rng,
                events=self.tick_events,
            )
            self._fired_between = True
            self.phase = 1

        # End of round: fire end-of-phase-2 explosives, convert zombies, reset.
        if self.tick >= self._ticks_per_round:
            if not self._fired_end_p2:
                process_door_explosives(
                    self.gmap, self.edit_queue, self.units, DET_END_PHASE2, self.rng,
                    events=self.tick_events,
                )
                self._fired_end_p2 = True
            self._end_round()
            self.paused = True

    # ------------------------------------------------------------------
    # Tick-step helpers
    # ------------------------------------------------------------------
    def _update_projectiles(self) -> None:
        """Tick all projectiles: grenades first (the shipped slot semantics,
        unchanged), then in-flight kinetic rounds (W2 unified march — tick
        slot 2, BEFORE movement, preserving the causal pipeline ordering of
        mechanics/03 §2). Fixed traversal order = spawn order, both lists."""
        for proj in self.projectiles:
            if proj.detonated:
                continue
            proj.update_position(self.tick)
            if self.tick >= proj.get_detonate_tick():
                proj.detonated = True
                if proj.proj_type == ORDER_GRENADE:
                    fx = int(proj.target_fx)
                    fy = int(proj.target_fy)
                    # W3: the LOBBED detonation goes through the payload
                    # EXECUTOR via the projectile's round (hand_grenade ->
                    # proj.ammo_name -> its payload row). The default
                    # grenade_frag -> payloads.frag_standard sequence is
                    # byte-identical to the pre-W3 inline triple (the
                    # replica gate); gas grenades ride the same call with
                    # their rows. Blast damage (the wave_p coupling row,
                    # exchange.COUPLING_TABLE[1]) stays a detonation-site
                    # invocation inside the executor.
                    payload = self.weapons_tables.payload_for_ammo(
                        proj.ammo_name)
                    execute_payload(
                        self.gmap, self.edit_queue, self.units, fy, fx,
                        payload, self.rng, events=self.tick_events,
                        kind="grenade")

        # W2: advance in-flight kinetic rounds (the unified march). Each
        # marches this tick's integer step budget; hits/chew/exposure resolve
        # inside advance() (same rng, fixed spawn order). Survivors stay.
        # W3: the edit queue rides along — an in-flight payload round (the
        # GL-6 40 mm) detonates at its stop through the payload executor.
        if self.bullets:
            survivors = []
            for b in self.bullets:
                if b.advance(self.gmap, self.units, self.shots,
                             self.real_time, self.rng,
                             events=self.tick_events,
                             queue=self.edit_queue):
                    survivors.append(b)
            self.bullets = survivors

    def _update_player_movement(self) -> None:
        """Step each player unit along its precomputed path.

        Status gate (mechanics/06 §4): a unit whose composed ``can_move`` is
        suppressed (knocked down / immobilized / paralyzed) holds position
        and its precomputed path PAUSES — ``path_tick_offset`` shifts by one
        so the path is not consumed while down; on release the unit resumes
        at the next un-walked path index (no catch-up teleport; the round
        may end before the tail is walked — being knocked down costs the
        distance). With no statuses this is a dead path, bit-identical to
        pre-P3 behavior.
        """
        for u in self.units:
            if not u.alive or u.team != 0:
                continue
            if not composed_flags(u).can_move:
                u.path_tick_offset += 1
                continue
            path_idx = self.tick - u.path_tick_offset
            if 0 <= path_idx < len(u.move_path):
                px, py = u.move_path[path_idx]
                u.face_towards(px, py)
                u.x = px
                u.y = py

    def _end_round(self) -> None:
        """End-of-round teardown — convert dead-by-zombie marines, reset state.

        Lifted from ``game.py:_end_execution`` (lines 2014-2043). Bumps
        the turn counter, converts marines, snaps floats, clears orders,
        refills AP, and rewinds the tick counter for the next round.
        Auto-pause happens in :meth:`step` (the caller).
        """
        convert_marines_to_zombies(self.units)

        for u in self.units:
            # Snap float position to nearest integer tile boundary.
            u.x = float(round(u.x))
            u.y = float(round(u.y))
            u.clear_orders()
            u.move_path = []
            u.last_fire_tick = -999
            # W3 ammo economy: the round boundary tops everyone off (the
            # WEGO planning pause is a between-rounds breather — v1 rule,
            # Erik's dial later). Also REQUIRED for correctness: the tick
            # counter rewinds to 0 below, so a carried reload_done_tick
            # from late in the round would stall the unit deep into the
            # next one (the exact hazard last_fire_tick = -999 solves).
            # None = full mag (the lazy first-trigger bind, combat.mag_gate);
            # untracked (mag_size 0) units never read either field.
            u.current_mag = None
            u.reload_done_tick = -1

        # Reset obstacles so dead bodies don't keep blocking physics. IN-PLACE
        # (not reassignment) so any bound view of the buffer stays valid — the
        # in-place-write discipline the PhysicsEngine will rely on (engine/02;
        # unification plan v2 §3a, the panel-flagged dangling-pointer trap).
        self.gmap.obstacles[:] = self.gmap.solid
        # Keep un-detonated projectiles (long-fuse grenades carry over).
        self.projectiles = [p for p in self.projectiles if not p.detonated]
        # In-flight kinetic rounds carry over too (W2): they are physical
        # objects mid-air, tick-counter-free (per-call budget), so the round
        # rewind cannot skew them. Empty for every shipped weapon.
        # Rewind for the next round.
        self.tick = 0
        self.phase = 0
        self.real_time = 0.0
        self.turn_number += 1
        self._fired_start_p1 = False
        self._fired_between = False
        self._fired_end_p2 = False

    # ------------------------------------------------------------------
    # Helpers used by main.py to populate projectiles when execution begins
    # ------------------------------------------------------------------
    def spawn_projectiles_from_grenade_orders(self) -> None:
        """Materialise queued grenade orders as in-flight projectiles.

        Called once before resuming a round (typically just before the
        player hits play / spacebar). Mirrors what
        ``game.py:_start_execution`` does at lines 1538-1556.
        """
        tpp = self._ticks_per_phase
        for u in self.units:
            if u.team != 0:
                continue
            for o in u.orders:
                if o.order_type == ORDER_GRENADE:
                    phase_start_tick = o.phase * tpp
                    end_x, end_y = u.get_planned_end_pos()
                    proj = Projectile(
                        ORDER_GRENADE,
                        end_x + u.footprint // 2,
                        end_y + u.footprint // 2,
                        o.target_fx + 0.5,
                        o.target_fy + 0.5,
                        fuse_seconds=o.grenade_fuse,
                        thrown_tick=phase_start_tick,
                        # W3: the order's round (None = the shipped
                        # grenade_frag — the UI path unchanged; smoke/tear/
                        # poison grenades name theirs; loadout UI = W6).
                        ammo_name=(getattr(o, "ammo_name", None)
                                   or "grenade_frag"),
                    )
                    self.projectiles.append(proj)


__all__ = ["Simulation", "SimState"]
