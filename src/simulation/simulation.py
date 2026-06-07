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
from simulation.generation import sample_unit_attributes
from simulation.species import get_species
from simulation.combat import (
    apply_blast_damage, process_door_explosives, process_shooting,
    Projectile, Shot,
)
from simulation.events import (
    DoorDestroyedEvent, ExplosionEvent, WallDestroyedEvent,
)
from simulation.gamemap import GameMap, MAT_DOOR
from simulation.orders import (
    DET_START_PHASE1, DET_BETWEEN_PHASES, DET_END_PHASE2,
    ORDER_GRENADE, ORDER_EXPLOSIVE, ORDER_FIRE, ORDER_MOVE_ATTACK,
    ORDER_MOVE_COVER, ORDER_SPRINT, MOVE_ORDER_TYPES,
)
from simulation.physics import apply_explosion, add_explosion_smoke
from simulation.physics_runner import PhysicsRunner
from simulation.recorder import PhysicsRecorder

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

        self.units: List = []
        self.projectiles: List = []
        self.shots: List = []          # legacy tracer list; renderer reads it
        self.tick_events: List = []    # cleared each step()

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
        else:
            self.physics_runner = None

        # Recorder ring buffer (owns ~80 MB at default 1200-frame capacity).
        if self._enable_recorder:
            self.recorder = PhysicsRecorder(
                fh=self.gmap.material.shape[0],
                fw=self.gmap.material.shape[1],
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

        # Re-sample with the sim's seeded RNG for deterministic rollouts.
        species = get_species(getattr(unit, "species_id", "human"))
        base_stats, mass, base_speed = sample_unit_attributes(species, self.rng)
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

        elif ot == ORDER_GRENADE:
            if u.get_ap(phase) < CFG.weapons.grenade.ap_cost:
                return False
            if u.has_grenade <= 0:
                return False
            order.ap_cost = CFG.weapons.grenade.ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)
            u.has_grenade -= 1
            return True

        elif ot == ORDER_EXPLOSIVE:
            if u.get_ap(phase) < CFG.weapons.door_explosive.ap_cost:
                return False
            if u.has_explosive <= 0:
                return False
            order.ap_cost = CFG.weapons.door_explosive.ap_cost
            u.orders.append(order)
            u.spend_ap(phase, order.ap_cost)
            u.has_explosive -= 1
            return True

        elif ot == ORDER_FIRE:
            if u.get_ap(phase) < CFG.weapons.rifle.ap_cost:
                return False
            order.ap_cost = CFG.weapons.rifle.ap_cost
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
                            for st in range(speed):
                                frac = (st + 1) / speed
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
    # Tick loop — the core simulation step
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance the simulation by one tick.

        No-op while paused. Auto-pauses at phase boundary (tick ==
        ticks_per_phase) and at end of round (tick == ticks_per_round)
        for the human player's planning UI. AI training rollouts ignore
        pause (they never call set_paused).

        Tick ordering — load-bearing! Matches the legacy
        ``Game._process_tick`` (game.py:1696-1747):
          1. Clear tick_events (this tick is a fresh slate)
          2. Update projectiles (advance position, detonate)
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
                self.gmap, self.units, DET_START_PHASE1, self.rng,
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

        # 3. Update player movement.
        self._update_player_movement()

        # 4. Process shooting.
        process_shooting(self.gmap, self.units, self.tick,
                         self.shots, self.real_time, self.rng,
                         events=self.tick_events)

        # 5. Zombie AI.
        update_zombies_tick(self.gmap, self.units, self.tick)

        # 6. Re-stamp obstacles.
        self.gmap.stamp_units(self.units)

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

        # Recorder snapshot.
        if self.recorder is not None:
            self.recorder.record(self.gmap, self.tick, self.real_time, self.units)

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
                self.gmap, self.units, DET_BETWEEN_PHASES, self.rng,
                events=self.tick_events,
            )
            self._fired_between = True
            self.phase = 1

        # End of round: fire end-of-phase-2 explosives, convert zombies, reset.
        if self.tick >= self._ticks_per_round:
            if not self._fired_end_p2:
                process_door_explosives(
                    self.gmap, self.units, DET_END_PHASE2, self.rng,
                    events=self.tick_events,
                )
                self._fired_end_p2 = True
            self._end_round()
            self.paused = True

    # ------------------------------------------------------------------
    # Tick-step helpers
    # ------------------------------------------------------------------
    def _update_projectiles(self) -> None:
        """Tick all projectiles. Detonate any whose fuse has run out."""
        for proj in self.projectiles:
            if proj.detonated:
                continue
            proj.update_position(self.tick)
            if self.tick >= proj.get_detonate_tick():
                proj.detonated = True
                if proj.proj_type == ORDER_GRENADE:
                    fx = int(proj.target_fx)
                    fy = int(proj.target_fy)
                    radius = CFG.weapons.grenade.blast_radius
                    apply_explosion(
                        self.gmap, fy, fx, radius,
                        CFG.weapons.grenade.pressure,
                        CFG.weapons.grenade.wall_damage,
                    )
                    apply_blast_damage(
                        self.units, fx, fy, radius,
                        CFG.weapons.grenade.unit_damage,
                        events=self.tick_events,
                    )
                    add_explosion_smoke(self.gmap, fy, fx, radius, self.rng)
                    self.tick_events.append(ExplosionEvent(
                        pos=(fx, fy), radius=radius, kind="grenade"))

    def _update_player_movement(self) -> None:
        """Step each player unit along its precomputed path."""
        for u in self.units:
            if not u.alive or u.team != 0:
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

        # Reset obstacles so dead bodies don't keep blocking physics.
        self.gmap.obstacles = self.gmap.is_wall.copy()
        # Keep un-detonated projectiles (long-fuse grenades carry over).
        self.projectiles = [p for p in self.projectiles if not p.detonated]
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
                    )
                    self.projectiles.append(proj)


__all__ = ["Simulation", "SimState"]
