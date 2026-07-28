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
    process_shooting, process_sprays,
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
from simulation.door_system import build_runtime_entities, sweep_doors
from simulation.entities.door import DOOR_OPEN
from simulation.entities.serialize import entity_carrier
from simulation.signal_bus import build_signal_bus
from simulation.logic_nodes import (
    aggregate_input, build_logic_nodes, sweep_logic_nodes,
)
from simulation.sensor_system import build_sensors, sample_sensors
from simulation.pump_system import build_pumps, sweep_pumps
from simulation.entities.schema import INPUT_HELD
from simulation.gamemap import GameMap, MAT_DOOR, MAT_DOOR_CLOSED
from simulation.movement import FootprintSamples, default_speed
from simulation.orders import (
    ORDER_GRENADE, ORDER_EXPLOSIVE, ORDER_FIRE, ORDER_MOVE_ATTACK,
    ORDER_MOVE_COVER, ORDER_SPRINT, MOVE_ORDER_TYPES,
    ONEPHASE_MOVE_ORDER_TYPES,
)
# Per-tick continuous intents (control-modularity P3, §3c) — the direct-control
# order vocabulary consumed under a ContinuousRealtime ruleset. Dormant under
# TwoPhaseWEGO: WEGO control sources never call the intent facade, so no unit
# ever carries a ``live_*``/``pending_*`` field and every consumer short-circuits
# on a ``getattr(..., default)`` (the dormancy guarantee).
from simulation import intents as _intents
from simulation import unit_fixed
# The Ruleset seam (control-modularity P1, docs/control_modularity_design_
# 2026-07-22.md §3a): TwoPhaseWEGO is the current round-clock/AP policy,
# extracted verbatim. Simulation owns one Ruleset instance, chosen at
# construction; DET_* slot constants now live behind ruleset.py's own
# import of orders (Simulation no longer references them directly).
from simulation.ruleset import Ruleset, TwoPhaseWEGO
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
from simulation.weapons import (
    FIRE_ORDER_ARCHETYPES, rebuild_tables as rebuild_weapon_tables,
)
# The action registry (onephase_wego design §5) — OnePhaseWEGO's verb table,
# rebuilt beside the weapon tables at every construction/reset.
from simulation.action_registry import rebuild_table as rebuild_action_table

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
    # A4: the strict entity presence carrier (simulation.entities.serialize.
    # entity_carrier) — ALWAYS written by get_state, n_entities == 0 for an
    # entity-free level. Carries the serialized ENTITY_SECT_V1 payload (not
    # just a hash) + the registry content-hash, so a state consumer can
    # locate an entity divergence per instance. Defaulted for any legacy
    # direct SimState(...) construction.
    entity_state: dict = None


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
                 breach_physics=None, enable_recorder: bool = True,
                 ruleset: Optional[Ruleset] = None):
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
        ruleset
            The turn-structure + cost-policy strategy (control-modularity
            P1, ``simulation.ruleset``). Chosen at construction, held for
            the simulation's lifetime — ``reset()`` does NOT replace it.
            Defaults to :class:`~simulation.ruleset.TwoPhaseWEGO`, the
            shipped two-phase WEGO round (byte-identical to the pre-P1
            hard-coded behavior).
        """
        self.level = level_data
        self._enable_recorder = enable_recorder
        self._bp = breach_physics
        self.ruleset = ruleset if ruleset is not None else TwoPhaseWEGO()

        # Constants (cached so reset() doesn't re-read CFG every time)
        self._ticks_per_phase = CFG.clock.ticks_per_phase
        self._ticks_per_round = CFG.clock.ticks_per_round
        self._phases_per_round = CFG.clock.phases_per_round
        self._tps = CFG.clock.ticks_per_second
        # OnePhaseWEGO's round length (onephase_wego design §2), cached beside
        # the two-phase numbers rather than replacing them — both rulesets are
        # constructible in the same process for the whole arc. Read only by
        # simulation.ruleset.OnePhaseWEGO.ticks_per_round.
        self._onephase_ticks_per_round = CFG.clock.onephase_ticks_per_round

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

        # A6 (a6 doors design §6.1): the sim's RUNTIME entity list — the
        # level's parsed instances with door entries replaced by fresh
        # DoorRuntime wrappers (ordinal order preserved), rebuilt on every
        # construct/reset so the shared LevelData never carries runtime
        # state. BOTH capture sites (get_state's carrier + the recorder
        # snapshot) read THIS list, so door state/want_open/hp rows ride
        # the one serializer automatically. `_doors` is the ordinal-order
        # doors sublist; door-free levels build an identical list to
        # `level.entities` and the 9e sweep is a single attribute check.
        self.entities, self._doors = build_runtime_entities(
            self.level, self.gmap)

        # Arc B (impl doc §2): the SignalBus + the resolved wire drive tables.
        # Built ONLY when the level declares wires (D1) — in B1 the union
        # sensors∪nodes∪wires reduces to wires; a wire-free level carries NO
        # bus, so `__signals__` stays empty and its digest is byte-identical to
        # Arc A (the dormancy guarantee, §8). All logic in slot 9e gates on
        # `self._signal_bus is not None`; the door STRUCTURAL sweep stays gated
        # on `self._doors` independently (the split gate, D2).
        self._signal_bus = build_signal_bus(self.level)
        self._entity_by_ordinal = {}
        self._door_drives = {}
        self._logic_nodes = []
        self._sensors = []
        self._sensor_accessor = None
        self._pumps = []
        if self._signal_bus is not None:
            self._build_logic_tables()

        # Weapon/ammo/payload tables (mechanics/03 §4, W1) — rebuilt from the
        # live CFG at every reset, exactly like GameMap rebuilds the material/
        # gas tables above. Config-static data; Ctrl+R alone does NOT rebuild
        # (engine/12 §5 — the construction-bound precedent).
        # W6 (meter-based ranges): the build binds THIS level's tile size, so
        # every row's authored range_m derives its integer range_tiles once,
        # here (quantize-once, engine/14 door 2). Pinned test worlds are
        # 1.0 m/tile (meters == tiles, the pre-W6 numbers exactly); the
        # playground's 0.333 m/tile derives 3x the tiles for the same reach.
        self.weapons_tables = rebuild_weapon_tables(
            tile_size_m=self.gmap.tile_size_m)
        # The ACTION REGISTRY (onephase_wego design §5) — the same
        # construction-bound, config-static contract as the weapon tables
        # above, and built AFTER them because its item rows are generated from
        # the LOBBED/PLACED weapon rows. Read only by the OnePhaseWEGO
        # executor and the hotbar; inert under every other ruleset.
        self.actions_table = rebuild_action_table(
            weapons_tables=self.weapons_tables)

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

        # Direct-control possession set (control-modularity P3, §3b/§3c): the
        # unit ids a ControlSource has issued at least one per-tick intent for.
        # Empty under WEGO (no control source ever calls the intent facade), so
        # ``_consume_direct_intents`` short-circuits and the whole continuous
        # path stays dormant — the dormancy guarantee for the digest gate.
        self._possessed_ids: set = set()

        # Target marking (onephase_wego design §11): ``{team: {unit_id, ...}}``
        # — a TEAM fact, not a unit's, consulted by every targeting function
        # (overwatch priority, idle return-fire preference). v1 is a single
        # "focus" level; graded 1-5 priorities are a future refinement. Empty
        # (and never touched) under every other ruleset.
        self.marks: dict = {}

        # Cover entities (onephase_wego design §7) — continuous-space
        # destructible collision shapes the bullet march can hit. Populated at
        # load from the level's [[entity]] rows in P5; the empty list here is
        # what every vision/march consumer iterates, so a cover-free level (and
        # every other ruleset) pays one empty loop.
        self.cover: list = []

        # Per-tick vision cache (design §8) — invalidated by tick number, so
        # every consumer in a tick sees ONE consistent answer.
        self._vision_cache = None

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

        # OnePhaseWEGO owns its own order vocabulary and interrupt semantics
        # (onephase_wego design §5/§13) — a whole different set of order types
        # and no AP gate at all, so it branches before the legacy body rather
        # than threading conditionals through it.
        if self.ruleset.drives_units:
            return self._onephase_apply_action(u, order)

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
            order.ap_cost = ap_cost
            if u.has_grenade <= 0:
                return False
            # AP check + spend (control-modularity P1): the cost-policy
            # chokepoint, verbatim behavior via the ruleset (§3a). Called
            # after the pure inventory guard above — both guards are
            # side-effect-free until this point, so their relative order
            # is unobservable; no mutation happens unless every guard
            # (inventory here, AP inside the ruleset) passes.
            if not self.ruleset.validate_and_cost(self, u, order):
                return False
            u.orders.append(order)
            u.has_grenade -= 1
            return True

        elif ot == ORDER_EXPLOSIVE:
            ap_cost = weapon_rows["breach_charge"].ap_cost
            order.ap_cost = ap_cost
            if u.has_explosive <= 0:
                return False
            if not self.ruleset.validate_and_cost(self, u, order):
                return False
            u.orders.append(order)
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
            order.ap_cost = ap_cost
            if not self.ruleset.validate_and_cost(self, u, order):
                return False
            u.orders.append(order)
            return True

        return False

    # ------------------------------------------------------------------
    # OnePhaseWEGO order placement (design §5/§13)
    # ------------------------------------------------------------------
    def _onephase_apply_action(self, u, order) -> bool:
        """Queue an order under OnePhaseWEGO and recompile the unit's timeline.

        Differences from the legacy body, all of them design consequences:

        - **No AP gate.** Time is the only currency (§3); an order's cost is
          the ticks it occupies on the compiled timeline, charged there.
        - **New orders interrupt by default** (§13). The FIRST order issued
          for a unit in a given round replaces whatever remains of its queue —
          which is what "issuing orders at the planning pause replaces the
          unit's remaining queue immediately" means operationally — and
          subsequent orders that round append to build the plan up. A step
          flagged ``interruptible = False`` (channeled: planting a charge,
          operating a terminal, objective interactions) must complete first
          and therefore survives the replacement.
        - **Every action is a registry row** (§5), so validation is a row
          lookup plus that row's own preconditions, not a per-order-type
          branch that has to be extended for every new verb.
        """
        action = self._onephase_action_for(order)
        if action is None:
            return False
        if not action.allows_class(getattr(u, "unit_class", "")):
            return False

        # Item-backed rows consume inventory (§15's loadout model; ground
        # pickup/drop is explicitly out of v1 scope).
        if action.item and not self._onephase_has_item(u, action.item):
            return False

        # Movement targets must be enterable for the unit's footprint — the
        # same geometric gate A* itself uses, so a plan can never be compiled
        # against a destination the unit could not stand on.
        if order.order_type in ONEPHASE_MOVE_ORDER_TYPES:
            if not self.gmap.is_passable_block(int(order.target_fy),
                                               int(order.target_fx),
                                               u.footprint):
                return False

        if u.plan_round != self.round_index:
            self._onephase_replace_queue(u)
            u.plan_round = self.round_index

        if action.item:
            self._onephase_take_item(u, action.item)
        if order.action_name is None:
            order.action_name = action.name
        u.orders.append(order)
        self.ruleset.on_orders_changed(self, u)
        return True

    def _onephase_action_for(self, order):
        """The registry row an order runs through, or ``None`` if this order
        type has no row (a legacy TwoPhaseWEGO order handed to the wrong
        ruleset)."""
        table = self.actions_table
        if getattr(order, "action_name", None):
            try:
                return table.get(order.action_name)
            except KeyError:
                return None
        try:
            return table.for_order_type(order.order_type)
        except KeyError:
            return None

    def _onephase_replace_queue(self, u) -> None:
        """Drop the unit's remaining orders, refunding their items — except a
        channeled action already in progress (§13), which must finish."""
        keep = []
        plan = getattr(u, "plan", None)
        in_progress = plan.step_at(self.tick) if plan is not None else None
        protected = (in_progress.order
                     if in_progress is not None
                     and not in_progress.action.interruptible else None)
        for o in u.orders:
            if o is protected:
                keep.append(o)
                continue
            action = self._onephase_action_for(o)
            if action is not None and action.item:
                self._onephase_return_item(u, action.item)
        u.orders = keep

    # Inventory hooks — v1 keeps the shipped two counters (§15: loadout-based
    # inventory, no ground pickup/drop). The item NAME is the weapon-row name
    # its generated action carries, so a real item system slots in behind
    # these three methods without touching the order path.
    _ITEM_COUNTERS = {"hand_grenade": "has_grenade",
                      "breach_charge": "has_explosive"}

    def _onephase_has_item(self, u, item) -> bool:
        attr = self._ITEM_COUNTERS.get(item)
        return True if attr is None else getattr(u, attr, 0) > 0

    def _onephase_take_item(self, u, item) -> None:
        attr = self._ITEM_COUNTERS.get(item)
        if attr is not None:
            setattr(u, attr, getattr(u, attr, 0) - 1)

    def _onephase_return_item(self, u, item) -> None:
        attr = self._ITEM_COUNTERS.get(item)
        if attr is not None:
            setattr(u, attr, getattr(u, attr, 0) + 1)

    def undo_last_order(self, unit_id: int) -> bool:
        """Pop the most recent order off ``unit_id``'s queue and refund.

        Lifted from ``game.py:1409-1418`` (Backspace handler). Refunds
        AP and inventory; returns ``True`` if an order was popped,
        ``False`` if there was nothing to undo or the unit doesn't exist.
        """
        u = self.get_unit(unit_id)
        if u is None or not u.orders:
            return False
        if self.ruleset.drives_units:
            # OnePhaseWEGO: nothing was charged at placement (§3), so undo is
            # "pop, hand the item back, recompile" — the timeline simply gets
            # shorter. A channeled step already in progress cannot be undone.
            plan = getattr(u, "plan", None)
            in_progress = plan.step_at(self.tick) if plan is not None else None
            if (in_progress is not None
                    and in_progress.order is u.orders[-1]
                    and not in_progress.action.interruptible):
                return False
            removed = u.orders.pop()
            action = self._onephase_action_for(removed)
            if action is not None and action.item:
                self._onephase_return_item(u, action.item)
            self.ruleset.on_orders_changed(self, u)
            return True
        removed = u.orders.pop()
        self.ruleset.refund(self, u, removed)
        if removed.order_type == ORDER_GRENADE:
            u.has_grenade += 1
        elif removed.order_type == ORDER_EXPLOSIVE:
            u.has_explosive += 1
        if removed.order_type in MOVE_ORDER_TYPES:
            # Movement path may have changed; recompute.
            self._compute_player_paths()
        return True

    # ------------------------------------------------------------------
    # Direct-control intent facade (control-modularity P3, §3c)
    # ------------------------------------------------------------------
    # The per-tick continuous verbs (MOVE_DIR / AIM / TRIGGER / THROW / USE).
    # A ControlSource (GamepadDirect, or an AgentPolicy) calls these each
    # FRAME; the sim samples the current value at each TICK boundary (§5 —
    # aim/move are frame-rate-continuous in the control layer, physics at 24
    # Hz). All directions/angles arrive ALREADY quantized to Q16.16 by the
    # control seam (``control_source.quantize_stick_direction``); these methods
    # never see a float axis. Continuous verbs (move/aim/trigger) OVERWRITE the
    # unit's live slot every call (last write before a tick wins); edge verbs
    # (throw/use) LATCH until a tick consumes them, so a button tap between two
    # ticks is neither dropped nor doubled. Setting any intent marks the unit
    # possessed, which arms ``_consume_direct_intents`` (dormant while empty).

    def set_move_dir(self, unit_id: int, dx_q: int, dy_q: int,
                     speed_mode: int) -> None:
        """MOVE_DIR: hold a Q16.16 unit-vector move direction + speed mode for
        ``unit_id`` (§3c). Overwrites each call; :meth:`clear_move_dir` (stick
        inside the deadzone) stops the unit."""
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.live_move_dir = _intents.MoveDirIntent(int(dx_q), int(dy_q),
                                                 int(speed_mode))
        self._possessed_ids.add(unit_id)

    def clear_move_dir(self, unit_id: int) -> None:
        """Drop ``unit_id``'s live MOVE_DIR (stick centered) — it holds position
        and replays no WEGO path (direct control owns it)."""
        u = self.get_unit(unit_id)
        if u is not None:
            u.live_move_dir = None

    def set_aim(self, unit_id: int, dx_q: int, dy_q: int) -> None:
        """AIM: hold a Q16.16 unit-vector facing for ``unit_id`` (§3c). The
        sim turns it into ``Unit.facing`` via the deterministic integer atan2
        kit at the next tick."""
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.live_aim = _intents.AimIntent(int(dx_q), int(dy_q))
        self._possessed_ids.add(unit_id)

    def set_trigger(self, unit_id: int, held: bool) -> None:
        """TRIGGER: hold/release ``unit_id``'s weapon trigger (§3c). While held,
        the unit auto-fires along its facing at its weapon cadence."""
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.live_trigger = bool(held)
        self._possessed_ids.add(unit_id)

    def throw_grenade_intent(self, unit_id: int, dx_q: int, dy_q: int,
                             fuse_seconds: float) -> None:
        """THROW: latch a grenade lob along a Q16.16 unit vector (§3c). Consumed
        (and cleared) at the next tick; ignored if the unit is out of
        grenades then."""
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_throw = _intents.ThrowIntent(int(dx_q), int(dy_q),
                                              float(fuse_seconds))
        self._possessed_ids.add(unit_id)

    def use_intent(self, unit_id: int) -> None:
        """USE: latch a context interaction (toggle an adjacent door's latch,
        §3c). Consumed (and cleared) at the next tick."""
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_use = True
        self._possessed_ids.add(unit_id)

    def _consume_direct_intents(self) -> None:
        """Apply the possessed units' latched/held per-tick intents (§3c),
        called once at the top of :meth:`step` (after the round-clock head).

        Dormant guarantee: returns immediately when no unit is possessed —
        under TwoPhaseWEGO ``_possessed_ids`` is always empty, so this is a
        single set-empty check and the digest is byte-identical.

        AIM sets facing (deterministic integer atan2); THROW spawns a grenade
        BEFORE the projectile-advance slot so a freshly-lobbed grenade travels
        this tick like a WEGO one; USE toggles a door latch. TRIGGER is not
        consumed here — a held ``live_trigger`` is read directly by the shooting
        slot (slot 4), where ``combat._directional_fire`` marches a free-aim
        shot along ``u.facing`` (free_aim_shooting_design §4b). MOVE_DIR is
        consumed later, in :meth:`_update_player_movement` (slot 3), where the
        WEGO branch it replaces already lives.
        """
        if not self._possessed_ids:
            return
        for u in self.units:
            if u.id not in self._possessed_ids or not u.alive or u.team != 0:
                # A now-invalid possessor (dead / unpossessed / wrong team)
                # releases a held trigger, so a later rebind starts clean.
                if getattr(u, "live_trigger", False):
                    u.live_trigger = False
                continue

            # AIM -> facing (synced state via the deterministic atan2 kit; a
            # zero vector leaves facing untouched).
            aim = getattr(u, "live_aim", None)
            if aim is not None and (aim.dx_q != 0 or aim.dy_q != 0):
                dx = _intents.dequantize(aim.dx_q)
                dy = _intents.dequantize(aim.dy_q)
                # World Y increases downward; facing is math-style (Y-up), so
                # negate dy — the same convention as Unit.face_towards.
                u.facing = unit_fixed.atan2_rad(-dy, dx)

            # THROW -> spawn a grenade now (edge; latched until consumed).
            throw = getattr(u, "pending_throw", None)
            if throw is not None:
                u.pending_throw = None
                self._spawn_direct_grenade(u, throw)

            # USE -> toggle an adjacent door latch (edge; latched).
            if getattr(u, "pending_use", False):
                u.pending_use = False
                self._direct_use(u)

            # TRIGGER is NOT consumed here: a held ``live_trigger`` (set by
            # set_trigger) is read directly by the shooting slot (slot 4),
            # where ``combat._directional_fire`` marches a FREE-AIM shot along
            # ``u.facing``, bypassing the range+LOS pre-gate (the march resolves
            # range/hit — free_aim_shooting_design §4b). No fabricated Order:
            # the tile-target band-aid (_aim_fire_order / live_fire_order) is
            # gone, and with it the unbounded-phase landmine (design §8).

    def _spawn_direct_grenade(self, u, throw) -> None:
        """THROW consumption: lob a grenade from ``u`` along the intent's
        Q16.16 unit vector, fusing after ``throw.fuse_seconds``. Reuses the
        shipped :class:`~simulation.combat.Projectile` (the WEGO grenade path);
        the target tile is derived deterministically from the fixed-point
        direction. No-op if the unit has no grenade left."""
        if getattr(u, "has_grenade", 0) <= 0:
            return
        cx = u.center_tile_x()
        cy = u.center_tile_y()
        dirx = _intents.dequantize(throw.dx_q)
        diry = _intents.dequantize(throw.dy_q)
        # v0.1 throw reach: a fixed tile distance along the aim (feel dial for
        # Erik; the grenade's own fuse/blast are the shipped numbers).
        reach = 6.0  # tiles
        tgt_fx = cx + dirx * reach
        tgt_fy = cy + diry * reach
        proj = Projectile(
            ORDER_GRENADE,
            cx, cy,
            tgt_fx + 0.5, tgt_fy + 0.5,
            fuse_seconds=throw.fuse_seconds,
            thrown_tick=self.tick,
            ammo_name="grenade_frag",
        )
        self.projectiles.append(proj)
        u.has_grenade -= 1

    def _direct_use(self, u) -> None:
        """USE consumption: flip the ``want_open`` latch of the first live door
        whose span touches the unit's footprint or its immediate ring (§3c).
        Reuses :meth:`door_at`; the slot-9e door sweep applies/retries the
        latch, exactly like the dev O-key."""
        x0, y0 = u.tile_x, u.tile_y
        fp = u.footprint
        # Footprint tiles plus a one-tile ring so a marine standing beside a
        # door can open it without overlapping it.
        for ty in range(y0 - 1, y0 + fp + 1):
            for tx in range(x0 - 1, x0 + fp + 1):
                door = self.door_at(ty, tx)
                if door is not None and door.alive:
                    door.want_open = not door.want_open
                    return

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

    def debug_cycle_weapon(self, unit_id: int):
        """DEBUG (W6, the playground tuning key): re-arm ``unit_id`` with the
        NEXT triggerable ``[weapons.*]`` row, in config (table) order.

        The armory-as-data payoff made playable: Erik selects a marine,
        taps the cycle key, and the unit's ``weapon_id`` walks every row
        whose archetype can take a FIRE order (``FIRE_ORDER_ARCHETYPES`` —
        projectile / hitscan / spray / melee; LOBBED and PLACED rows ride
        their own order modes, G and B, and have no trigger path, so
        cycling onto them would arm a weapon that cannot fire).

        This is the FACADE seam for the input layer (the renderer never
        mutates sim state; input debug actions land through sim methods —
        the I/J/K/U precedent, but through a proper facade call rather
        than a direct field write, because a weapon swap has coupled
        cadence/mag/burst state to reset):

        - ``current_mag`` / ``reload_done_tick`` reset (a swapped weapon
          arrives with a fresh magazine — the round-boundary rule);
        - spray burst state clears (a half-finished Dragon-7 burst does
          not continue out of an LR-50);
        - ``last_fire_tick`` is left alone (the cadence gate is per-unit
          wall-clock, and -999 resets at every round boundary anyway).

        Deterministic and RNG-free; ``weapon_id`` is ordinary synced input
        state (like an order), so replay determinism is unaffected: same
        key presses, same trajectory. Returns the new weapon name, or
        ``None`` for a dead/unknown/zombie unit (zombies carry no weapon
        rows — their melee is the ai_zombie path)."""
        u = self.get_unit(unit_id)
        if u is None or not u.alive or u.is_zombie:
            return None
        rows = [name for name, w in self.weapons_tables.weapons.by_name.items()
                if w.archetype in FIRE_ORDER_ARCHETYPES]
        if not rows:
            return None
        cur = getattr(u, "weapon_id", "")
        idx = rows.index(cur) if cur in rows else -1
        u.weapon_id = rows[(idx + 1) % len(rows)]
        # Fresh-magazine + dead-burst state (see docstring).
        u.current_mag = None
        u.reload_done_tick = -1
        u.spray_ticks_left = 0
        u.spray_order = None
        u.spray_target = None
        return u.weapon_id

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

    # Round-clock geometry, routed through the ruleset (onephase_wego design
    # §2). Under TwoPhaseWEGO / ContinuousRealtime these return exactly what
    # the pre-existing fields always meant (the base Ruleset defaults);
    # OnePhaseWEGO's free-running tick makes within-round position a modulo.
    # Read by the planning UI (arrival timestamps, the round progress bar) and
    # by the timeline executor.

    @property
    def ticks_per_round(self) -> int:
        return self.ruleset.ticks_per_round(self)

    @property
    def round_tick(self) -> int:
        return self.ruleset.round_tick(self)

    @property
    def round_index(self) -> int:
        return self.ruleset.round_index(self)

    def visible_enemy_ids(self, team: int = 0) -> tuple:
        """Enemies ``team`` can currently see — TEAM VISION, the union of its
        members' cones (onephase_wego design §8).

        THE fog-of-war gate: fog in v1 is visibility gating only, so the
        renderer draws no enemy whose id is absent here. Consult it only when
        ``ruleset.fog_of_war`` is set — the shipped rulesets have no vision
        model and show everything, exactly as they always have.
        """
        from simulation import vision
        return vision.visible_enemy_ids(self, team)

    def round_start_tick(self) -> int:
        """Absolute tick at which the CURRENT round began. The anchor every
        scheduled action resolves against (§12: "detonate 1.8 s into the
        round" is ``round_start_tick() + round(1.8 * tps)``)."""
        return self.tick - self.round_tick

    def get_state(self) -> SimState:
        return SimState(
            gmap=self.gmap,
            units=self.units,
            projectiles=self.projectiles,
            tick=self.tick,
            phase=self.phase,
            paused=self.paused,
            entity_state=entity_carrier(self.entities,
                                        signals=self._digest_signals()),
        )

    def door_at(self, fy: int, fx: int):
        """The DoorRuntime whose runtime span contains tile ``(fy, fx)``,
        or None. Unique by the load-time disjoint-span rule (a6 doors
        design §4.2); ordinal-order scan; matches OPEN doors' spans too —
        the span is geometry, not material. Used by the dev O-key (§10)."""
        for d in self._doors:
            if d.contains(fy, fx):
                return d
        return None

    # ------------------------------------------------------------------
    # Arc B logic layer (impl doc §2) — built only when the bus exists
    # ------------------------------------------------------------------
    def _build_logic_tables(self) -> None:
        """Precompute the ordinal→entity map, the node evaluator list, and the
        per-door wire drive table (impl doc §2b/§2d). Called from
        ``_reset_internal`` iff a bus exists.

        ``_logic_nodes`` is the ordinal-ordered node evaluator list for the
        9e(b) sweep; building it also REPLACES each ``filter`` instance in
        ``self.entities`` with its runtime wrapper (the EMA row, §5), so
        ``_entity_by_ordinal`` is rebuilt AFTER. ``_door_drives`` maps a WIRED
        door's ordinal to its open/close driving slot-index lists (from the
        resolved ``level.wires``, D3): only doors with an incoming open/close
        wire are wire-driven; every other door keeps its Arc-A ``want_open``
        latch + the dev O-key."""
        # Node evaluators (may patch self.entities with FilterRuntime wrappers).
        self._logic_nodes = build_logic_nodes(self)
        # Sensor runtimes + the §5a accessor (B3): ordinal-order samplers read
        # at 9e(a). Built AFTER the node build (sensors are plain
        # EntityInstances, never replaced) — the accessor's site index is
        # frozen from the field sensors' resolved tiles.
        self._sensors = build_sensors(self)
        # Pump actuators (B4, §6): the 9e(d) N-feed sweep. Building it REPLACES
        # each pump instance in self.entities with its PumpRuntime (the at_target
        # latch row, §8), so _entity_by_ordinal is rebuilt AFTER, like the nodes.
        self._pumps = build_pumps(self)
        self._entity_by_ordinal = {int(e.ordinal): e for e in self.entities}
        bus = self._signal_bus
        door_ordinals = {int(d.ordinal) for d in self._doors}
        drives: dict = {}
        for w in (getattr(self.level, "wires", None) or []):
            if w.target_ordinal not in door_ordinals:
                continue                  # non-door targets: later patches
            if w.input not in ("open", "close"):
                continue
            slot = bus.slot(w.source_ordinal, w.signal)
            drives.setdefault(int(w.target_ordinal),
                              {"open": [], "close": []})[w.input].append(slot)
        self._door_drives = drives

    def _signal_emit(self) -> None:
        """9e(a): write every wired door's ``is_open`` (and every wired
        entity's free ``alive``) into ``pub`` BEFORE the logic sweep, so
        current-tick reads (require_alive / door.is_open→node, later) are not
        a tick stale (D4). Sensor sampling joins here in B3."""
        bus = self._signal_bus
        for idx, (ordinal, name) in enumerate(bus.slots):
            ent = self._entity_by_ordinal.get(ordinal)
            if ent is None:
                continue                  # dangling/destroyed emitter → leave 0
            if name == "is_open":
                bus.set_pub(idx, 1 if getattr(ent, "state", None) == DOOR_OPEN
                            else 0)
            elif name == "alive":
                bus.set_pub(idx, 1 if getattr(ent, "alive", True) else 0)
            # other names (sensor value / node out): later patches (B2/B3)

    def _resolve_door_inputs(self) -> None:
        """9e(c): for each door WITH incoming open/close wires, aggregate them
        from ``pub`` (OR/held) and drive ``want_open`` — open active→1, close
        active→0 with close priority (INPUT_PRIORITY, §2d), neither→retain the
        latch. A door with NO open/close wire is skipped (keeps its Arc-A latch
        + dev O-key, D3). Ordinal order (``self._doors`` is pre-sorted)."""
        bus = self._signal_bus
        for d in self._doors:
            drv = self._door_drives.get(int(d.ordinal))
            if drv is None:
                continue                  # unwired door: Arc-A latch (D3)
            # OR/held aggregation via the shared helper (§2d) — the same rule
            # the node sweep uses, so the door input resolve is not a bespoke
            # second implementation.
            close_active = aggregate_input(bus, drv["close"], INPUT_HELD) != 0
            open_active = aggregate_input(bus, drv["open"], INPUT_HELD) != 0
            if close_active:              # close beats open (safe state)
                d.want_open = False
            elif open_active:
                d.want_open = True
            # else: neither driving → retain the current want_open latch

    def _digest_signals(self) -> tuple:
        """The ``__signals__`` payload for the digest/recorder: the bus's
        non-``alive`` slot values, or ``()`` when no bus exists (the dormant
        case — a wire-free level hashes exactly as Arc A, §8)."""
        if self._signal_bus is None:
            return ()
        return self._signal_bus.digest_rows()

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

        AI training treats this as an episode boundary. Routed through the
        ruleset (control-modularity P1, §3a) — ``TwoPhaseWEGO.is_terminal``
        carries the verbatim body this docstring describes.
        """
        return self.ruleset.is_terminal(self)

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

        # Round-clock head (control-modularity P1, §3a): tick-0-only work
        # (DET_START_PHASE1, path-offset reset, initial unit stamp under
        # TwoPhaseWEGO) now lives behind the ruleset; verbatim body in
        # simulation.ruleset.TwoPhaseWEGO.on_round_start.
        self.ruleset.on_round_start(self)

        # 1b. Direct-control intents (control-modularity P3, §3c): consume the
        # possessed units' held/latched per-tick verbs (AIM/THROW/USE/TRIGGER)
        # BEFORE projectiles advance, so a THROW lobbed this tick travels this
        # tick like a WEGO grenade and a TRIGGER arms slot 4's fire order.
        # Dormant under WEGO (no possessed units) — a single set-empty check.
        self._consume_direct_intents()

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

        # 3 + 4. Unit simulation. A ruleset that owns these slots
        # (OnePhaseWEGO — ``drives_units``) replaces BOTH with one call: its
        # compiled timeline decides movement and shooting together, because
        # under that ruleset they are two readings of the same schedule
        # (onephase_wego design §3). The legacy path below is untouched and
        # still runs verbatim for every other ruleset.
        if self.ruleset.drives_units:
            self.ruleset.drive_units(self)
        else:
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

        # 4b. SPRAY deposits (mechanics/03 §5, W4) — the shooting slot's
        # second half: every active spray burst enqueues its aimed heat/gas
        # cone into the edit queue (flushed at 6b with everything else, so
        # this tick's flame heat converts to temperature THIS tick). Writes
        # fields only (two-terminals invariant); draws no RNG; dormant (one
        # attribute read per unit) when no spray weapon is in play.
        # W6: tick_events rides along for the RENDER-ONLY SprayJetEvent
        # (the flame-jet visual) — not digest-hashed, pure synced-state fn.
        process_sprays(self.gmap, self.units, self.edit_queue,
                       events=self.tick_events)

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

        # 9. Process fire burn-through walls. A6: the door-event split keys
        # on BOTH door materials — legacy painted MAT_DOOR and the entity
        # door's MAT_DOOR_CLOSED (a6 doors design §1).
        for (yy, xx) in destroyed:
            mat = int(self.gmap.material[yy, xx]) if (0 <= yy < self.gmap.material.shape[0]
                                                      and 0 <= xx < self.gmap.material.shape[1]) else -1
            self.gmap.destroy_wall(yy, xx)
            if mat in (MAT_DOOR, MAT_DOOR_CLOSED):
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
                if mat in (MAT_DOOR, MAT_DOOR_CLOSED):
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

        # 9e. Entities — the SPLIT GATE (Arc B impl doc §2b, D2). Two
        # independently-gated pieces:
        #  · the door STRUCTURAL sweep runs on ANY door level (gated on
        #    `self._doors`, unchanged from Arc A) — external-destruction
        #    reconciliation + the synced want_open latch, ordinal order;
        #  · the LOGIC block runs iff the SignalBus exists (wires present),
        #    so a door-only wire-free level is byte-identical to Arc A (the
        #    dormancy guarantee, §8).
        # Sub-order per tick when the bus exists (§2b): (a) sample + emit —
        # write every wired door's is_open (and wired entities' free alive)
        # into pub BEFORE the logic sweep; (b) logic sweep — each node in
        # ordinal order reads pub, writes stg[out] (B2); (c) input resolve —
        # drive wired doors' want_open (OR/held, close-beats-open); (d) actuator
        # sweep — pumps (N-feed edit, B4) then the door structural sweep; (e) swap
        # pub[node-signals] ← stg (node outputs become readable next tick — one
        # tick per hop, §2c). BEFORE the recorder snapshot so recorder/digest
        # see state consistent with this tick's flips (a6 doors §5.1). NOT gated
        # on physics_runner — flips are pure gamemap edits; effects reach the
        # solvers next tick via the step-6 restamp.
        if self._signal_bus is not None:
            self._signal_emit()             # (a) is_open/alive → pub
            if self._sensors:
                sample_sensors(self)        # (a) sensors sample world → pub
            if self._logic_nodes:
                sweep_logic_nodes(self)     # (b) node sweep (ordinal order)
            self._resolve_door_inputs()     # (c) drive wired doors' want_open
            if self._pumps:
                sweep_pumps(self)           # (d) pump N-feed edit — BEFORE doors
        if self._doors:
            sweep_doors(self)               # (d) door structural sweep
        if self._signal_bus is not None:
            self._signal_bus.swap_node_signals()   # (e) pub[node-slots] ← stg

        # Recorder snapshot. A4: the entity list rides along (presence-gated
        # inside the recorder — an entity-free level's .npz is byte-identical).
        # A6: the SIM's runtime list (door rows live), not level.entities.
        if self.recorder is not None:
            self.recorder.record(self.gmap, self.tick, self.real_time,
                                 self.units, entities=self.entities,
                                 signals=self._digest_signals())

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

        # 10. Advance tick (the clock itself — not WEGO policy, stays here
        # unconditionally so a future ContinuousRealtime ruleset still
        # ticks) + route phase-boundary/round-end/auto-pause through the
        # ruleset (control-modularity P1, §3a). Verbatim body in
        # simulation.ruleset.TwoPhaseWEGO.on_tick_end.
        self.tick += 1
        self.real_time += sim_time_per_tick

        self.ruleset.on_tick_end(self)

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

            # Direct-control branch (control-modularity P3, §3c): a unit with a
            # live MOVE_DIR intent this tick moves by velocity, footprint-
            # collision-checked with the SAME predicate A* uses, and NEVER
            # replays a WEGO path. Dormant under WEGO (no unit carries
            # ``live_move_dir``), so the branch below is untouched and
            # byte-identical.
            move_dir = getattr(u, "live_move_dir", None)
            if move_dir is not None:
                if composed_flags(u).can_move:
                    self._step_move_dir(u, move_dir)
                continue

            if not composed_flags(u).can_move:
                u.path_tick_offset += 1
                continue
            path_idx = self.tick - u.path_tick_offset
            if 0 <= path_idx < len(u.move_path):
                px, py = u.move_path[path_idx]
                # A6 path-hold (a6 doors design §9): precomputed WEGO paths
                # are the one consumer that does not re-query the mobility
                # table, so re-check the next position's footprint block
                # against the LIVE grid — a door that closed across the
                # plan holds the unit here, burning the tick exactly like
                # the status gate (no catch-up teleport; the round may end
                # before the tail is walked). int() truncation anchors the
                # block the same way tile_x/tile_y anchor the unit's own
                # footprint; plan-time A* used this same predicate, and
                # nothing else ever makes a tile LESS passable mid-round,
                # so door-free trajectories are bit-identical.
                if not self.gmap.is_passable_block(int(py), int(px),
                                                   u.footprint):
                    u.path_tick_offset += 1
                    continue
                u.face_towards(px, py)
                u.x = px
                u.y = py

    def _step_move_dir(self, u, move_dir) -> None:
        """MOVE_DIR consumption (control-modularity P3, §3c) — the real new
        mechanic: advance ``u`` one tick along its live direction by a
        mobility-table-scaled velocity, footprint-collision-checked with the
        SAME predicate A* uses (:meth:`GameMap.is_passable_block`), so direct
        move and A* agree on both speed and what blocks.

        Speed: the unit's per-tile cadence is the WEGO base ticks-per-tile for
        the intent's ``speed_mode`` (:func:`_ticks_per_tile`), scaled by the
        footprint's area-average mobility exactly as the WEGO A* replay scales
        it (:func:`movement.default_speed`) — a furniture tile is 2.5x slower
        for both. The per-tick advance is ``1 / tick_cost`` tiles.

        Determinism: the direction arrives as a Q16.16 unit vector; the only
        floats are the exact power-of-two dequantize and the position math on
        ``u.x``/``u.y`` (plain IEEE +/-/*/ — no libm, no transcendentals), so
        the trajectory is bit-reproducible cross-machine (``u.x``/``y`` are
        synced digest fields — this stays on the same float discipline as the
        WEGO path).

        Collision: try the full move; if the destination footprint is blocked,
        slide along each axis independently (still :meth:`is_passable_block`),
        so pushing diagonally into a wall glides rather than dead-stops. A
        fully boxed-in unit holds. Facing follows the move direction unless a
        live AIM overrides it (aim already set facing this tick).
        """
        base_ticks = _ticks_per_tile(move_dir.speed_mode)
        fp = u.footprint
        samples = FootprintSamples(
            mobility=self.gmap.footprint_mobility(u.tile_y, u.tile_x, fp))
        tick_cost = default_speed(samples, base_ticks)   # ticks per tile
        tiles_per_tick = 1.0 / tick_cost
        dx = _intents.dequantize(move_dir.dx_q)
        dy = _intents.dequantize(move_dir.dy_q)
        step_x = dx * tiles_per_tick
        step_y = dy * tiles_per_tick
        nx = u.x + step_x
        ny = u.y + step_y

        gmap = self.gmap
        if gmap.is_passable_block(int(ny), int(nx), fp):
            u.x, u.y = nx, ny
        elif gmap.is_passable_block(int(u.y), int(nx), fp):
            u.x = nx                          # slide along X
        elif gmap.is_passable_block(int(ny), int(u.x), fp):
            u.y = ny                          # slide along Y
        # else fully blocked -> hold position this tick.

        # Facing: aim wins (already applied in _consume_direct_intents this
        # tick); otherwise face the direction of travel.
        if getattr(u, "live_aim", None) is None and (dx != 0.0 or dy != 0.0):
            u.face_towards(u.x + dx, u.y + dy)

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
            # W4: a spray burst does not survive the round boundary — the
            # orders it would ride are cleared above, and burst state is a
            # within-round derivation (the mag-state rule). Cleared here so
            # a rewound tick counter can never replay a stale burst.
            u.spray_ticks_left = 0
            u.spray_target = None
            u.spray_order = None

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
