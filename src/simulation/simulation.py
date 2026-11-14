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
Every random number the sim draws is a KEYED draw on ``sim.draws``
(:class:`simulation.draws.MatchDraws`, #29 P3b): Philox-4x32-10 keyed by the
seed, counter ``(tick, index, salt, id)`` -- the shot and beam cones, the
exposure and crit rolls, the FieldEdit noise at the flush. No draw depends on
a stream position. ``sim.rng`` (a numpy Generator) is drawn from by nothing in
the sim; a match replaces it with a stand-in that refuses any use
(:class:`simulation.draws.RefusingRng`, installed by ``spawn_match``).

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
from dataclasses import replace as _dc_replace
from numbers import Integral
from typing import List, Optional

import numpy as np

from config import CFG
from simulation.ai_zombie import update_zombies_tick, convert_marines_to_zombies
from simulation.generation import predefined_unit_attributes
from simulation.species import get_species
from simulation.combat import (
    apply_temperature_ignition,
    process_shooting, process_sprays, process_melee_intents,
    Projectile,
    mag_settle, reload_begin,
)
# Simultaneous resolution of an attack slot (#169): phase A records, phase B
# (``resolve``) applies. Bare name: the lockstep test rebinds it.
from simulation.simultaneous import SimultaneousEffects
# The focus-fire law (G4p): its dials (built at construction) and its decay.
from simulation import exposure
from simulation import horde
from simulation import noise
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
    apply_teargas_blind, apply_wind_force,
)
from simulation.events import (  # noqa: F401 (ExplosionEvent: legacy re-export — emitted by the executor now)
    DoorDestroyedEvent, ExplosionEvent, PlantEvent, ThrowEvent,
    WallDestroyedEvent,
)
from simulation.unit import THROW_NO_REACH
from simulation.unit_schema import MAG_UNBOUND
from simulation.cover_system import attach_cover
from simulation.door_system import build_runtime_entities, sweep_doors
from simulation.entities.door import DOOR_OPEN
from simulation.entities.serialize import entity_carrier
from simulation.signal_bus import build_signal_bus
from simulation.logic_nodes import (
    aggregate_input, build_logic_nodes, sweep_logic_nodes,
)
from simulation.sensor_system import build_sensors, sample_sensors
from simulation.pump_system import build_pumps, sweep_pumps
from simulation.vent_system import build_vents, sweep_vents
from simulation.timed_charge_system import (build_timed_charges,
                                            sweep_timed_charges)
from simulation.heat_source_system import (build_heat_sources,
                                           sweep_heat_sources)
from simulation.entities.schema import INPUT_HELD
from simulation.gamemap import GameMap, MAT_DOOR, MAT_DOOR_CLOSED
from simulation.movement import (default_speed, facing_cos_q16,
                                 footprint_samples, move_speed_q16,
                                 THROTTLE_ONE)
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
from simulation import charges
from simulation import draws as _draws
from simulation import intents as _intents
from simulation import speeds
from simulation import turning
from simulation import unit_fixed
from simulation import unit_impulse
from simulation import unit_orders
from simulation import unit_traits
from simulation import verb_gates
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
from simulation.damage import tick_credit
# Weapon/ammo/payload data tables (mechanics/03 §4, W1): the facade owns a
# bundle, rebuilt fresh at every construction/reset (the GameMap material/gas
# table pattern) so it always reflects the current CFG.
from simulation.weapons import (
    FIRE_ORDER_ARCHETYPES, PLANT_WEAPON, fuse_ticks_from_seconds,
    rebuild_tables as rebuild_weapon_tables,
)
# The action registry (onephase_wego design §5) — OnePhaseWEGO's verb table,
# rebuilt beside the weapon tables at every construction/reset.
from simulation.action_registry import rebuild_table as rebuild_action_table
from simulation.loadouts import rebuild_tables as rebuild_loadout_tables

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
        # The team the HUMAN controls (#119): what the HUD, selection, the
        # flashlights and the possession default mean by "the squad". Control
        # / UI state, never synced, never digested; the sim itself treats
        # every team alike.
        self.player_team: int = 0

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

        # ray-engine-v2 P6b: the renderer's light request and its render-
        # state inputs (cone emitters, sky; G3 F4b: the fire flicker), held
        # HERE so a reset() -- which builds a fresh PhysicsRunner -- re-applies
        # them (set_light).
        self._light = (False, None, None, None)

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
        # #29 P3b (design §4.7): THE draws -- keyed Philox, key = the seed,
        # counter (tick, index, salt, id) on this sim's clock. An unseeded
        # sim takes its key once from its (entropy-seeded) generator, so it
        # stays unseeded; a seeded one is keyed by its seed itself.
        self.draws = _draws.MatchDraws(
            seed if seed is not None
            else int(self.rng.integers(0, 2 ** 63)),
            clock=lambda: self.tick)

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
        # Cover rows become their runtime objects IN self.entities, the same
        # way door rows do — the entity list is what the serializer walks, and
        # it asks each row for its runtime digest fields (onephase_wego §7).
        self.entities, self.cover = attach_cover(self.entities, self.level)

        # Vent system PATCH 1 (issue #48, docs/vent_system_design_2026-08-23.
        # md §2/§7): duct + vent runtime, built UNCONDITIONALLY (unlike pumps
        # — vents carry no wired inputs in patch 1, so there is no SignalBus
        # gate). A duct/vent-free level builds two empty lists; the 9e(d)
        # sweep call is a single `if self._ducts` check (dormancy: byte-
        # identical to a vent-free build, ENTITY_SECT absence transparency).
        self._ducts, self._vents = build_vents(self)

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
        # Loadouts + env profiles (#119): validated against the weapon table
        # just built (an unknown weapon / profile is a load error here).
        self.loadout_tables = rebuild_loadout_tables()
        # The focus-fire law's dials (G4p, [damage.exposure] + [zombie]
        # expose_resist), quantized once through their refusing door.
        self.exposure_settings = exposure.rebuild_settings()
        # The ACTION REGISTRY (onephase_wego design §5) — the same
        # construction-bound, config-static contract as the weapon tables
        # above, and built AFTER them because its item rows are generated from
        # the LOBBED/PLACED weapon rows. Read only by the OnePhaseWEGO
        # executor and the hotbar; inert under every other ruleset.
        self.actions_table = rebuild_action_table(
            weapons_tables=self.weapons_tables)
        # Timed charges (#31 P5): built AFTER the weapon tables because each
        # charge's `payload` is validated against the PayloadTable here (an
        # unknown row name is a load error). Replaces each timed_charge row in
        # self.entities with its runtime (the countdown row serializes). A
        # charge-free level builds an empty list (dormancy).
        self._timed_charges = build_timed_charges(self)
        # Heat sources (levelgen vocabulary E.2 row 3): built after the
        # weapon tables for the same reason (death_payload is validated
        # here). Replaces each heat_source row with its runtime (the `hp`
        # row serializes). A source-free level builds an empty list.
        self._heat_sources = build_heat_sources(self)

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
        # G1p / G5p-a1: the marine turning law's dials, quantized ONCE here
        # through the refusing doors (simulation/turning.py): the top speed
        # (Q16.16 radians per tick, [marine] turn_rate_deg_per_s) and the
        # acceleration (per tick^2, [marine] turn_accel_rad_per_s2).
        self._marine_turn_rate_q16 = turning.turn_rate_q16(CFG)
        self._marine_turn_accel_q16 = turning.turn_accel_q16(CFG)
        # G1p-c: the [speeds] table (m/s) quantized ONCE here with this map's
        # tile size (the refusing door, simulation/speeds.py): the marine's
        # walk / full costs in Q16 ticks per tile, the direction factors, the
        # zombie's integer shamble cadence.
        self.speed_table = speeds.speed_table(CFG, self.gmap.tile_size_m)
        # G9p: the horde field's dials ([zombie], the refusing door
        # simulation/horde.py), quantized ONCE for this map, and the map's
        # field cache (derived, never synced: re-flooded every
        # field_refresh_ticks and whenever the topology moves).
        self.horde_settings = horde.horde_settings(CFG,
                                                   self.gmap.tile_size_m)
        self.horde_state = horde.HordeState()
        # G5p-a2: the unit impulse law's dials ([impulse], the ammo rows'
        # momentum) quantized ONCE here with this map's tile size (the
        # refusing door, simulation/unit_impulse.py), and handed to the map
        # so the combat helpers that take only the map read the same table.
        self.impulse_table = unit_impulse.impulse_table(
            CFG, self.gmap.tile_size_m,
            rho_air=unit_impulse.map_air_density(self.gmap))
        self.gmap._impulse_table = self.impulse_table

        # Who drives the horde (#122, rl-ready §5 item 1): ``None`` = the
        # conductor's zombie slot runs the shipped law over every zombie (every
        # game path: TwoPhaseWEGO / OnePhase / gamepad); a set = a MATCH whose
        # zombie controller owns the horde (:meth:`hand_zombies_to_controller`)
        # — the slot then runs the SAME law for exactly the zombie ids that
        # controller requested this tick (:meth:`drive_zombies`), so no unit is
        # ever driven twice, and an idle zombie controller leaves it standing.
        self._zombie_requests: set | None = None

        # The commander's orders latched this tick (#195 1f, the
        # ``drive_zombies`` precedent): unit id -> (what, where, until,
        # target), the last ``order_intent`` of the tick per unit; written
        # onto the units at slot 1b' (``unit_orders.apply_requests``) and
        # emptied there. Never synced, never digested; empty between ticks.
        self._order_requests: dict = {}

        # The intent tap (#126, replay by re-simulation): ``None`` = dormant
        # (every path but a recorded match; each intent entry point pays one
        # ``is None`` check). Attached by :meth:`attach_intent_log`, it is
        # handed EVERY call of the ten intent entry points, with the tick it
        # was issued at, BEFORE the call acts -- the one recording seam; no
        # controller logs anything itself. Never synced, never digested;
        # dropped by reset().
        self._intent_log = None

        # Target marking (onephase_wego design §11): ``{team: {unit_id, ...}}``
        # — a TEAM fact, not a unit's, consulted by every targeting function
        # (overwatch priority, idle return-fire preference). v1 is a single
        # "focus" level; graded 1-5 priorities are a future refinement. Empty
        # (and never touched) under every other ruleset.
        self.marks: dict = {}

        # (self.cover — the continuous-space destructible rectangles the
        # bullet march stops on, §7 — was built with the entity list above,
        # because each one IS an entry in it.)

        # Per-tick vision cache (design §8) — invalidated by tick number, so
        # every consumer in a tick sees ONE consistent answer.
        self._vision_cache = None

        # The sight rule (#195 1a): "cone" on every path that is not a match;
        # ``spawn_match`` sets it from the match's spec. Read by
        # ``vision.can_see`` from here, never from a caller. Constant for a
        # sim, so the per-tick cache above needs no key for it.
        self.sight = "cone"

        # The team picture (#195 1d, ``simulation/team_picture.py``): what
        # each marine team knows of the enemy. None off a match;
        # ``start_team_picture`` (called by ``spawn_match``) switches it on
        # and the conductor's slot 11 updates it once per tick. Not synced,
        # not hashed: rebuilt by re-simulation only.
        self.team_picture = None

        # Ambush barriers (design §10): ``{target_unit_id: release_tick}``.
        # A per-group readiness counter in the sim — NO SignalBus is needed in
        # v1, though the action registry's ``start_condition`` field keeps that
        # slot open for when level logic should trigger squad actions.
        self.ambush_released: dict = {}

        # Planted charges awaiting their moment (design §12) — the three
        # det SLOTS are replaced by a schedulable TICK, which only works
        # because the ruleset's clock is monotonic: "1.8 s into the round" and
        # "t=0 of the next round" are the same kind of number.
        self.planted_charges: list = []
        # Plants refused because the table held unit_schema.CHARGE_CAPACITY
        # live charges (#29 P3a, charges.place); a diagnostic counter.
        self.charges_refused = 0

        # Physics runner (created once, re-bound on reset only if bp present).
        if self._bp is not None:
            # Always build a fresh runner so per-session params are clean.
            self.physics_runner = PhysicsRunner(self._bp)
            # Wire the C++ PhysicsEngine into the gmap for the C++ stamp_units
            # path (the runner owns the engine). A bare GameMap with no engine
            # bound falls back to the Python reference path automatically.
            self.gmap.bind_physics_engine(self.physics_runner.engine)
            self._apply_light()             # P6b: the renderer's light survives a reset
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
        # Roster slots (#119, rl-ready §5.2): per SPAWN team, the units in
        # slot order. A slot is assigned once and kept at death and
        # conversion, so a team's size is a mask over a fixed layout.
        self._rosters: dict[int, list] = {}

    # ------------------------------------------------------------------
    # Unit management
    # ------------------------------------------------------------------
    def add_unit(self, unit, position=None, *, hooks=None) -> int:
        """Register ``unit`` with the sim, assign it a stable id.

        ``position`` is optional — if provided as ``(x, y)`` in
        physics-tile coords, the unit is teleported there before its
        first tick. Otherwise the unit keeps its constructor-set position.

        Re-assigns the unit's predefined BaseStats (the species means), then
        COMPILES it (#29 U1, ``unit_traits.compile_unit``: the hook block --
        all ONE bar the sheet overrides ``hooks``, ``{hook name: Q16 int}`` --
        and ``mass_g``) and sets its hp to its full health
        (``unit_traits.max_hp_q``).
        """
        if position is not None:
            x, y = position
            unit.set_pos(x, y)            # #29 P3c: the float door

        # Deterministic predefined attributes (ingress door 2 — quantized
        # species means; see generation.py). The sim RNG no longer touches
        # spawn stats: the draft MVN sampler was BLAS/LAPACK-backed and
        # cross-machine nondeterministic (docs/lenovo_dev_setup.md §8b).
        species = get_species(getattr(unit, "species_id", "human"))
        base_stats, mass, base_speed = predefined_unit_attributes(species)
        unit.base_stats = base_stats
        unit.mass = mass
        unit.base_speed = base_speed
        # #29 U1: THE compile step -- the hooks (unit_traits.compile_unit,
        # their only writer) and mass_g (body + kit, grams, G5p-a2) -- then
        # the full health of the class times the HP hook.
        unit_traits.compile_unit(self, unit, hooks)
        unit.hp_q = unit_traits.max_hp_q(unit)

        unit.id = self._next_unit_id
        self._next_unit_id += 1
        # Roster slot: the next index in the unit's spawn team, never reused.
        # Derived from the (deterministic) spawn order, so it is not separately
        # digested: the unit list is hashed in id order already.
        roster = self._rosters.setdefault(int(unit.team), [])
        unit.slot = len(roster)
        roster.append(unit)
        self.units.append(unit)
        return unit.id

    def roster(self, team: int, n_slots: int | None = None) -> list:
        """``team``'s units indexed by roster slot (#119): every unit ever
        spawned into ``team``, dead and converted ones included (a slot is
        never reused), padded with ``None`` up to ``n_slots`` (a level's
        ``max_per_team``). The living mask is ``u is not None and u.alive and
        u.team == team`` — a converted marine keeps its slot here but has left
        the team."""
        out = list(self._rosters.get(int(team), ()))
        if n_slots is not None:
            if n_slots < len(out):
                raise ValueError(f"roster({team}): {len(out)} slots used, "
                                 f"n_slots={n_slots} is too small")
            out.extend([None] * (n_slots - len(out)))
        return out

    def spawn_match(self, spec=None) -> list:
        """Spawn a match's units (#119 P2) — THE one entry: ``spec`` (default
        ``self.level.match``, the level's match.toml) placed by the level's
        spawn rooms (``simulation.match.spawn_match``). Needs a fresh sim."""
        from simulation.match import spawn_match
        spec = self.level.match if spec is None else spec
        if spec is None:
            raise ValueError(f"level {self.level.name!r} has no match.toml")
        return spawn_match(self, spec)

    def start_team_picture(self, teams, memory_ticks: int | None = None):
        """Switch on the team picture (#195 1d) for the marine ``teams`` and
        take its first update now; from then on the conductor's slot 11
        updates it every tick. ``spawn_match`` calls it for a match; a
        hand-built sim (a test) calls it itself. ``memory_ticks`` defaults
        to ``[team_picture] memory_s`` quantized once (tests may pass a
        short one). Returns the picture."""
        from simulation import team_picture as tp
        if memory_ticks is None:
            memory_ticks = tp.memory_ticks(tp.picture_settings(CFG),
                                           float(self._tps))
        self.team_picture = tp.TeamPicture(self, teams, int(memory_ticks))
        self.team_picture.update(self)
        return self.team_picture

    def hand_zombies_to_controller(self) -> None:
        """Give the horde to a match's zombie controller (#122): from now on
        the conductor's zombie slot no longer runs the law on its own; it
        runs it only for the zombies a controller names through
        :meth:`drive_zombies`, once per tick. Called by the match runners
        (``tools/run_match.py``, ``control_match.MatchObserver``) when they
        build the match's controllers; never on a game path. Idempotent."""
        if self._zombie_requests is None:
            self._zombie_requests = set()

    def attach_intent_log(self, log) -> None:
        """Attach the intent tap's sink (#126): from now on every call of
        ``set_move_dir`` / ``clear_move_dir`` / ``set_aim`` / ``set_trigger``
        / ``throw_grenade_intent`` / ``use_intent`` / ``reload_intent`` /
        ``shove_intent`` / ``melee_intent`` / ``set_flashlight`` /
        ``plant_charge_intent`` / ``order_intent`` / ``drive_zombies`` /
        ``apply_action`` is handed to ``log.record(tick, kind, args)`` (``kind``
        the method name, ``args`` its arguments normalised exactly as the
        method itself normalises them) before it acts. ``None`` detaches.
        The sink may refuse a call by raising (``match_replay`` refuses
        ``apply_action`` orders). Changes no sim behaviour."""
        self._intent_log = log

    def drive_zombies(self, unit_ids) -> None:
        """ZOMBIE intent (#122): the zombie law (``ai_zombie.
        update_zombies_tick`` — trigger, chain activation, the horde field's
        descent (G9p), the bite) drives these units at this tick's zombie
        slot. Requests in
        one tick UNION (a unit named twice is still driven once); the request
        is consumed by the slot. Refused unless the horde was handed to a
        controller (:meth:`hand_zombies_to_controller`) — on a game path the
        conductor drives every zombie already, and a second driver is the bug."""
        if self._zombie_requests is None:
            raise RuntimeError("drive_zombies: the horde is driven by the "
                               "conductor (no zombie controller attached; "
                               "see hand_zombies_to_controller)")
        ids = [int(i) for i in unit_ids]
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "drive_zombies", (ids,))
        self._zombie_requests.update(ids)

    def _run_zombie_slot(self) -> None:
        """Tick slot 5: the zombie law, run ONCE per tick by its one driver —
        the conductor over every zombie (``_zombie_requests is None``, every
        game path, byte-identical to before #122), or, in a match, the zombie
        controller's requested ids (filtering preserves ``self.units`` order,
        so a controller naming the whole horde reproduces the conductor's
        call exactly). A bite's hit / kill events ride ``tick_events``
        with source ``"melee"`` (#122), like a bullet's."""
        # G9p: the noises made since the last slot are drained HERE, every
        # tick, whether or not any zombie is driven (simulation/noise.py).
        noises = noise.take(self.gmap)
        reqs = self._zombie_requests
        if reqs is None:
            units = self.units
        else:
            if not reqs:
                return
            units = [u for u in self.units
                     if not u.is_zombie or u.id in reqs]
            reqs.clear()
        update_zombies_tick(self.gmap, units, self.tick,
                            events=self.tick_events,
                            # the class cadences, handed to the zombie walk's
                            # default_speed with each zombie's own WALK / SPRINT
                            # hook (ai_zombie._walk, #29 U1)
                            shamble_ticks=self.speed_table.zombie_shamble_ticks,  # traits-class: WALK
                            settings=self.horde_settings,
                            state=self.horde_state, noises=noises,
                            doors=self._doors,
                            sprint_ticks=self.speed_table.zombie_sprint_ticks)  # traits-class: SPRINT

    def controlled_units(self) -> list:
        """The living units of :attr:`player_team` — the human's squad."""
        return [u for u in self.units
                if u.team == self.player_team and u.alive]

    def get_unit(self, unit_id: int):
        """Linear lookup by id. ``None`` if not found."""
        for u in self.units:
            if u.id == unit_id:
                return u
        return None

    def marines(self):
        """All living non-zombies, every team (the renderer's marine look)."""
        return [u for u in self.units if not u.is_zombie and u.alive]

    def zombies(self):
        """All living zombies."""
        return [u for u in self.units if u.is_zombie and u.alive]

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
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "apply_action",
                                    (int(unit_id), order))
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

    def begin_new_plan(self, unit_id: int) -> bool:
        """Start a fresh plan for ``unit_id`` — the facade seam behind "a plain
        click replaces the plan, a shift-click appends to it" (§13/§16).

        Drops the unit's remaining orders (refunding their items) except a
        channeled action already in progress, and recompiles. Idempotent, so a
        control source may call it before every replacing order without having
        to track whether it already did.
        """
        u = self.get_unit(unit_id)
        if u is None or not self.ruleset.drives_units:
            return False
        self._onephase_replace_queue(u)
        u.plan_round = self.round_index
        self.ruleset.on_orders_changed(self, u)
        return True

    def _onephase_channeled_in_progress(self, u):
        """The unit's currently-running uninterruptible step, or ``None``.

        The predicate is ``started``, NOT "the plan says this tick belongs to
        it": at the planning pause a channeled action scheduled to begin this
        very tick has not begun, so it is freely replaceable and undoable.
        Only an action the unit is actually part-way through — a charge half
        planted, a terminal half operated — resists interruption (§13).
        """
        plan = getattr(u, "plan", None)
        if plan is None:
            return None
        step = plan.step_at(self.tick)
        if (step is not None and step.started and not step.retired
                and not step.action.interruptible):
            return step
        return None

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
        # Standing OVERWATCH ends here (design §9: it "persists across rounds
        # until REPLACED"). Replacing the plan IS the replacement — you have
        # told this marine to do something else. Clearing it at the one
        # chokepoint where a fresh plan begins is what makes the posture, and
        # the cone drawn for it, honest: submit nothing and the watch stands
        # into the next round; give an order and it is over.
        u.overwatch_facing = None
        u.overwatch_half_deg = None

        keep = []
        in_progress = self._onephase_channeled_in_progress(u)
        protected = in_progress.order if in_progress is not None else None
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
            in_progress = self._onephase_channeled_in_progress(u)
            if in_progress is not None and in_progress.order is u.orders[-1]:
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
                     throttle_q: int) -> None:
        """MOVE_DIR: hold a Q16.16 unit-vector move direction + a THROTTLE
        for ``unit_id`` (§3c; gameplay pack G1p). ``throttle_q`` is Q16 in
        ``[0, ONE]`` -- 0 the walk, ONE the full speed, continuous between,
        the extra speed forward only (``movement.move_speed_tiles_per_tick``,
        the ``[speeds]`` table in m/s) -- CLAMPED here, as a throw's
        reach is clamped (the neighbouring intents refuse nothing), and the
        clamped value is what the tap records. Overwrites each call;
        :meth:`clear_move_dir` (stick inside the deadzone) stops the unit."""
        throttle_q = min(max(int(throttle_q), 0), THROTTLE_ONE)
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "set_move_dir",
                                    (int(unit_id), int(dx_q), int(dy_q),
                                     throttle_q))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.live_move_dir = _intents.MoveDirIntent(int(dx_q), int(dy_q),
                                                 throttle_q)
        self._possessed_ids.add(unit_id)

    def clear_move_dir(self, unit_id: int) -> None:
        """Drop ``unit_id``'s live MOVE_DIR (stick centered) — it holds position
        and replays no WEGO path (direct control owns it)."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "clear_move_dir",
                                    (int(unit_id),))
        u = self.get_unit(unit_id)
        if u is not None:
            u.live_move_dir = None

    def set_aim(self, unit_id: int, dx_q: int, dy_q: int) -> None:
        """AIM: hold a Q16.16 unit-vector facing ``unit_id`` WANTS (§3c). Each
        tick the sim turns ``Unit.facing`` toward it (the deterministic
        integer atan2 kit) by at most the unit's turn rate, the shorter way,
        reaching it exactly (G1p, ``simulation/turning.py``); a shot leaves
        along the CURRENT facing."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "set_aim",
                                    (int(unit_id), int(dx_q), int(dy_q)))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.live_aim = _intents.AimIntent(int(dx_q), int(dy_q))
        self._possessed_ids.add(unit_id)

    def unit_turn_rate_q16(self, u) -> int:
        """The top turning speed of unit ``u`` in Q16.16 radians per tick
        (G1p's cap; the turning law's ``w``): THE accessor of the TURN hook
        (#29 U1) -- the class value, the marine row's ``[marine]
        turn_rate_deg_per_s`` (every unit's: a zombie too, a copy of the
        unit it infected), times ``u``'s TURN hook, rounded once. The law,
        the predictor, the bots and the observation read it; never the
        cached class value. int64: w < 2^18 x 2^19."""
        return unit_traits.scaled(u, unit_traits.TURN,
                                  self._marine_turn_rate_q16)

    def unit_turn_accel_q16(self, u) -> int:
        """The own angular acceleration of unit ``u`` in Q16.16 radians per
        tick^2 (G5p-a1, the turning law's ``a``; ``[marine]
        turn_accel_rad_per_s2``) times its TURN hook, read per unit as the
        top speed is (#29 U1; the compile door refuses a hook that takes the
        law past its trial bound). int64: a < 2^18 x 2^19."""
        return unit_traits.scaled(u, unit_traits.TURN,
                                  self._marine_turn_accel_q16)

    def facing_after_turn_q16(self, u, dx_q: int, dy_q: int) -> int:
        """The facing (Q16.16 radians) ``u`` has after this tick's turn
        toward the Q16.16 aim vector ``(dx_q, dy_q)``: the pure evaluation of
        exactly slot 1a's step (``turning.predict``: the law at the unit's
        facing, angular velocity and dials), also what a controller reads to
        know where a shot it fires this tick will go (the bots' trigger
        discipline). Exact when read after the aim is set and before
        :meth:`step`. World Y runs down and facing is math-style, hence
        ``-dy`` -- the convention of ``Unit.face_towards``."""
        target = unit_fixed.atan2_q16(-int(dy_q), int(dx_q))
        # G5p-a2: slot 1a adds the outside spin integrated from the unit's
        # angular accumulator before it turns (every producer of it -- slot
        # 2, 4c, the previous tick's shoves -- has run when a controller's
        # handle_frame reads this).
        return turning.predict(u, target, self.unit_turn_accel_q16(u),
                               self.unit_turn_rate_q16(u),
                               unit_impulse.pending_dw(self.impulse_table, u))

    def set_trigger(self, unit_id: int, held: bool) -> None:
        """TRIGGER: hold/release ``unit_id``'s weapon trigger (§3c). While held,
        the unit auto-fires along its facing at its weapon cadence."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "set_trigger",
                                    (int(unit_id), bool(held)))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.live_trigger = bool(held)
        self._possessed_ids.add(unit_id)

    def throw_grenade_intent(self, unit_id: int, dx_q: int, dy_q: int,
                             fuse_ticks: int,
                             reach_q: int | None = None,
                             kind: int = 0) -> None:
        """THROW: latch a grenade lob along a Q16.16 unit vector (§3c).
        Consumed (and cleared) at the next tick by :meth:`_start_throw`.

        ``kind`` (G3p) is the index of the hand-grenade ``[ammo.*]`` row
        thrown (``self.weapons_tables.hand_grenade_rows``; 0 = the frag),
        refused by name here when it names none. ``fuse_ticks`` is the fuse
        in INTEGER ticks (was float seconds: every caller converts once at
        its own door, ``weapons.fuse_ticks_from_seconds``), counted from the
        tick the grenade LEAVES the hand and clamped at release to the hand
        grenade row's fuse range. ``reach_q`` (#121) is the lob distance in
        Q16.16 tiles, clamped to the hand grenade's ``max_throw_range`` at
        release; ``None`` keeps the fixed v0.1 reach (the gamepad).

        The verb starts a WIND-UP (``[weapons.hand_grenade]
        throw_windup_seconds``): ignored if the unit has none of that kind,
        is already winding up, or cannot act; otherwise a render-only
        :class:`~simulation.events.ThrowEvent` is emitted and the grenade
        leaves the hand that many ticks later (the count is spent then)."""
        self.weapons_tables.hand_grenade_kind(kind)         # refuses by name
        if isinstance(fuse_ticks, bool) or not isinstance(fuse_ticks, Integral):
            raise TypeError(
                f"throw_grenade_intent: fuse_ticks must be an integer number "
                f"of ticks (G3p: it was float seconds), got {fuse_ticks!r}")
        if self._intent_log is not None:
            self._intent_log.record(
                self.tick, "throw_grenade_intent",
                (int(unit_id), int(dx_q), int(dy_q), int(fuse_ticks),
                 None if reach_q is None else int(reach_q), int(kind)))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_throw = _intents.ThrowIntent(
            int(dx_q), int(dy_q), int(fuse_ticks),
            None if reach_q is None else int(reach_q), int(kind))
        self._possessed_ids.add(unit_id)

    def plant_charge_intent(self, unit_id: int, fuse_ticks: int) -> None:
        """PLANT (#193): latch planting the unit's charge on the floor at its
        feet with a timer of ``fuse_ticks`` INTEGER ticks (every caller
        converts once at its own door, ``weapons.fuse_ticks_from_seconds``),
        CLAMPED here to the plant row's fuse range (``[weapons.c4_satchel]
        fuse_min_seconds`` .. ``fuse_max_seconds``, 5 .. 60 s), as a
        throttle is clamped; the clamped value is what the tap records.
        Consumed (and cleared) at the next tick's slot 1b by
        :meth:`_start_plant`.

        The plant takes ``[weapons.c4_satchel] plant_seconds``: ignored,
        with no event, if the unit carries no charge, is already planting
        or winding up a throw, or cannot act; otherwise a render-only
        :class:`~simulation.events.PlantEvent` is emitted and, that many
        ticks later, the charge lies on the unit's centre tile as it stands
        THEN, owned by the unit (the blast's credit), detonating ``fuse``
        ticks after that (the conductor's slot 9e, ``charges.tick``); the
        count is spent then. A unit that dies or is stunned meanwhile plants
        nothing and keeps its count. Integer, no random number. Bots never
        call it (yet); an RL agent and the human do."""
        if isinstance(fuse_ticks, bool) or not isinstance(fuse_ticks, Integral):
            raise TypeError(
                f"plant_charge_intent: fuse_ticks must be an integer number "
                f"of ticks, got {fuse_ticks!r}")
        row = self.weapons_tables.weapons.by_name[PLANT_WEAPON]
        lo = fuse_ticks_from_seconds(row.fuse_min_seconds, self._tps)
        hi = fuse_ticks_from_seconds(row.fuse_max_seconds, self._tps)
        fuse_ticks = min(max(int(fuse_ticks), lo), hi)
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "plant_charge_intent",
                                    (int(unit_id), fuse_ticks))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_plant = fuse_ticks
        self._possessed_ids.add(unit_id)

    def use_intent(self, unit_id: int) -> None:
        """USE: latch a context interaction (toggle an adjacent door's latch,
        §3c). Consumed (and cleared) at the next tick."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "use_intent", (int(unit_id),))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_use = True
        self._possessed_ids.add(unit_id)

    def reload_intent(self, unit_id: int) -> None:
        """RELOAD (gameplay pack G2p): latch a manual reload. Consumed (and
        cleared) at the next tick by :meth:`_reload`: if the unit's weapon
        tracks a magazine, a spare magazine exists, the unit is not already
        reloading and the magazine in the gun is not full, that magazine is
        DISCARDED with every round in it, one spare is used and the stall
        runs ``reload_seconds``; otherwise nothing happens. Hand-written
        bots never call it (they rely on the automatic reload when the
        magazine empties); an RL agent reloads when it chooses."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "reload_intent",
                                    (int(unit_id),))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_reload = True
        self._possessed_ids.add(unit_id)

    def shove_intent(self, unit_id: int) -> None:
        """SHOVE (gameplay pack G5p-a2, ``docs/unit_impulse_design_
        2026-11-10.md`` §4, §13): latch a shove. Consumed (and cleared) at
        the next tick's slot 1b by ``unit_impulse.shove``: if the unit can
        act and its cooldown (``[impulse.shove] cooldown_seconds``) has run,
        the adjacent living unit of ANY team nearest its facing (within the
        shove's cone) takes the shove's impulse (``[impulse.shove]
        impulse_ns``) along that facing through the unit impulse law -- it
        is pushed by its mass, staggers, or falls -- and a render-only
        ``ShoveEvent`` is emitted; with no such unit nothing happens and the
        cooldown is not spent. A zombie does not shove. Hand-written bots
        never call it yet (G8p); an RL agent and the human do."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "shove_intent",
                                    (int(unit_id),))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_shove = True
        self._possessed_ids.add(unit_id)

    def melee_intent(self, unit_id: int) -> None:
        """MELEE (gameplay pack G6p, ``docs/gameplay_pack_v1_design_
        2026-11-10.md`` §3): latch a blow. Consumed (and cleared) at the next
        slot 4 by ``combat.process_melee_intents``, in phase A against the
        slot's start state: if the unit is a living marine with a melee row
        (its loadout's ``melee``), can act, is not winding up a throw, its
        row's cadence has run and an ENEMY (another team, zombies included,
        never a teammate) is melee-adjacent within the row's facing arc
        (``melee_arc_half_deg``), the one nearest the facing (ties to the
        lower id) takes the row's blow -- a ``DamagePacket`` credited to the
        striker with the ``melee`` cause, the row's short STUNNED and its
        push along striker -> target through the unit impulse law, all
        landing at the slot's phase B -- and a render-only ``MeleeEvent`` is
        emitted. With no such target nothing happens and no cadence is
        spent. A zombie does not use the verb (its bite is its own law).
        Hand-written bots never call it yet (G8p); an RL agent and the human
        do."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "melee_intent",
                                    (int(unit_id),))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.pending_melee = True

    def set_flashlight(self, unit_id: int, off: bool) -> None:
        """FLASHLIGHT (gameplay pack G7p): hold ``unit_id``'s flashlight bit
        ``off`` (True) or on (False; every unit starts on). A HOLD verb,
        like :meth:`set_trigger`: the sim overwrites the bit on every call.
        The argument is the OFF sense so that a zero action row (a policy's
        do-nothing stance) leaves the light as it was and every pinned
        digest as it is. SYNCED (hashed in the unit digest only while off),
        read by nothing in the sim (light as game state is #81): the frame
        lights draw the unit's cone from it
        (``ui.model.flashlight_cones``)."""
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "set_flashlight",
                                    (int(unit_id), bool(off)))
        u = self.get_unit(unit_id)
        if u is None:
            return
        u.flashlight_on = not bool(off)

    def order_intent(self, unit_id: int, what: int, where: int, ticks: int,
                     target_id: int) -> None:
        """ORDER (#195 1f, ``simulation.unit_orders``): the COMMANDER's
        verb, not a soldier's -- give ``unit_id`` the order ``what``
        (``unit_orders.ORDER_WHAT``'s code: 1 go to, 2 hold, 3 clear, 4 fall
        back to; 0 = none, which CLEARS the unit's order and takes no other
        field) at the place ``where`` on the level's room graph (a room id,
        or ``ORDER_OPENING_BASE`` + an opening id), for ``ticks`` INTEGER
        ticks (0 = until told otherwise; every caller converts seconds once
        at its own door), with the priority target ``target_id`` (an enemy's
        unit id, or -1).

        Refused by name at :func:`unit_orders.check` BEFORE the tap (so a
        recording never holds a refused call); then recorded and LATCHED on
        the sim (a second call for the unit in one tick replaces the first)
        and written onto the unit at the next slot 1b'. Cleared at death, at
        conversion and when its tick comes (``order_until``, after the tick
        advance). Does not possess the unit. INFORMATION ONLY until rung 4:
        nothing in the game reads an order for a decision."""
        args = unit_orders.check(self, unit_id, what, where, ticks,
                                 target_id)
        if self._intent_log is not None:
            self._intent_log.record(self.tick, "order_intent", args)
        uid, what, where, ticks, target = args
        until = unit_orders.NO_LIMIT if ticks == 0 else self.tick + ticks
        self._order_requests[uid] = (what, where, until, target)

    def _reload(self, u) -> None:
        """RELOAD consumption (the law of :meth:`reload_intent`): integer
        state only, no random number. A unit whose weapon has no magazine
        (``mag_size`` 0), and any reload the gate refuses, leave the unit
        UNTOUCHED: the decision is the gate's (``verb_gates.can_reload``, a
        pure read of the settled magazine, ``mag_status``), and the lazy
        settle (binding a never-bound magazine, completing a stall that has
        run out) is bookkeeping the next trigger pull or ``mag_status`` does
        anyway -- so a refused reload writes nothing, and a forced one leaves
        the digests where a do-nothing tick leaves them (#195 1e, the 1h
        gate)."""
        if not verb_gates.can_reload(self, u):
            return
        weapon = self.weapons_tables.weapons.by_name[u.weapon_id]
        mag_settle(u, weapon, self.tick)          # a no-op when ready; the
        reload_begin(u, weapon, self.tick,         # begin's own precondition
                     events=self.tick_events)

    def _consume_direct_intents(self) -> None:
        """Apply the possessed units' latched/held per-tick intents (§3c),
        called once at the top of :meth:`step` (after the round-clock head).

        Dormant guarantee: returns immediately when no unit is possessed —
        under TwoPhaseWEGO ``_possessed_ids`` is always empty, so this is a
        single set-empty check and the digest is byte-identical.

        AIM is NOT consumed here: slot 1a turns toward a held aim
        (``unit_impulse.integrate_all``, G5p-a1); THROW spawns a grenade
        BEFORE the projectile-advance slot so a freshly-lobbed grenade travels
        this tick like a WEGO one; USE toggles a door latch; RELOAD discards
        the magazine and starts the stall (G2p). TRIGGER is not
        consumed here — a held ``live_trigger`` is read directly by the shooting
        slot (slot 4), where ``combat._directional_fire`` marches a free-aim
        shot along ``u.facing`` (free_aim_shooting_design §4b). MOVE_DIR is
        consumed later, in :meth:`_update_player_movement` (slot 3), where the
        WEGO branch it replaces already lives.
        """
        if not self._possessed_ids:
            return
        for u in self.units:
            if u.id not in self._possessed_ids or not u.alive or u.is_zombie:
                # A now-invalid possessor (dead / unpossessed / wrong team)
                # releases a held trigger, so a later rebind starts clean,
                # and drops a throw wind-up (G3p: nothing leaves the hand,
                # the count was never spent).
                if getattr(u, "live_trigger", False):
                    u.live_trigger = False
                if getattr(u, "throw_release_tick", -1) >= 0:
                    u.throw_release_tick = -1
                # ... and a plant under way (#193): nothing is planted, the
                # count was never spent.
                if getattr(u, "plant_done_tick", -1) >= 0:
                    u.plant_done_tick = -1
                continue

            # AIM: already turned toward in slot 1a (the turning law, G5p-a1;
            # a zero vector holds the last target).

            # THROW -> start the wind-up (edge; latched until consumed); a
            # wind-up under way is carried to its release tick (G3p).
            throw = getattr(u, "pending_throw", None)
            if throw is not None:
                u.pending_throw = None
                self._start_throw(u, throw)
            self._advance_throw(u)

            # PLANT -> start planting the charge (edge; latched); a plant
            # under way is carried to its last tick (#193).
            plant = getattr(u, "pending_plant", None)
            if plant is not None:
                u.pending_plant = None
                self._start_plant(u, plant)
            self._advance_plant(u)

            # USE -> toggle an adjacent door latch (edge; latched).
            if getattr(u, "pending_use", False):
                u.pending_use = False
                self._direct_use(u)

            # RELOAD -> discard the magazine, run the stall (edge; latched).
            if getattr(u, "pending_reload", False):
                u.pending_reload = False
                self._reload(u)

            # SHOVE -> push the adjacent unit nearest the facing (edge;
            # latched; G5p-a2). It ADDS to the target's accumulators only
            # (integrated next tick's 1a), reading the slot-start positions
            # and the shover's 1a-turned facing: every shove of a tick reads
            # the same state and integer sums commute, so shovers' ids
            # change nothing.
            if getattr(u, "pending_shove", False):
                u.pending_shove = False
                if verb_gates.can_shove(self, u):
                    unit_impulse.shove(self.impulse_table, self.units, u,
                                       self.tick, events=self.tick_events)

            # TRIGGER is NOT consumed here: a held ``live_trigger`` (set by
            # set_trigger) is read directly by the shooting slot (slot 4),
            # where ``combat._directional_fire`` marches a FREE-AIM shot along
            # ``u.facing``, bypassing the range+LOS pre-gate (the march resolves
            # range/hit — free_aim_shooting_design §4b). No fabricated Order:
            # the tile-target band-aid (_aim_fire_order / live_fire_order) is
            # gone, and with it the unbounded-phase landmine (design §8).

    def _start_throw(self, u, throw) -> None:
        """THROW consumption (the law of :meth:`throw_grenade_intent`, G3p).
        Integer state only, no random number. A throw of a kind the unit
        does not carry, one arriving while the unit is already winding up,
        and one from a unit whose statuses forbid acting, do nothing and
        emit nothing. Otherwise a :class:`~simulation.events.ThrowEvent` is
        emitted and the wind-up starts; with a wind-up of 0 ticks the
        grenade leaves the hand NOW, along the intent's own vector -- the
        pre-G3p same-tick throw, bit for bit."""
        # The decision is the gate's (verb_gates.can_throw, #195 1e): already
        # winding up, planting (#193), none of that kind, stunned / paralysed.
        if not verb_gates.can_throw(self, u, throw.kind):
            return
        # #29 U1: the unit's own wind-up, the HANDLING hook's accessor
        ticks = unit_traits.throw_windup_ticks(
            u, self.weapons_tables.weapons.by_name)
        self.tick_events.append(ThrowEvent(unit_id=int(u.id),
                                           kind=int(throw.kind),
                                           ticks=int(ticks)))
        if ticks == 0:
            self._release_grenade(u, throw.kind, throw.fuse_ticks,
                                  throw.reach_q, throw.dx_q, throw.dy_q)
            return
        u.throw_release_tick = self.tick + ticks
        u.throw_kind = int(throw.kind)
        u.throw_fuse_ticks = int(throw.fuse_ticks)
        u.throw_reach_q = (THROW_NO_REACH if throw.reach_q is None
                           else int(throw.reach_q))

    def _advance_throw(self, u) -> None:
        """One tick of a wind-up under way: a unit that can no longer act
        throws NOTHING and keeps its count (the wind-up is dropped); at the
        release tick the grenade leaves the hand along the facing the unit
        has THEN (Q16 ``cos`` / ``sin`` of its facing, world y down), at the
        reach the intent named. (A dead unit never gets here: the consume
        loop drops its wind-up.)"""
        if u.throw_release_tick < 0:
            return
        if not composed_flags(u).can_act:
            u.throw_release_tick = -1
            return
        if self.tick < u.throw_release_tick:
            return
        u.throw_release_tick = -1
        facing_q = int(u.facing_q)          # #29 P3e: the int, no round trip
        self._release_grenade(
            u, u.throw_kind, u.throw_fuse_ticks,
            None if u.throw_reach_q == THROW_NO_REACH else u.throw_reach_q,
            unit_fixed.cos_q16(facing_q), -unit_fixed.sin_q16(facing_q))

    def _start_plant(self, u, fuse_ticks) -> None:
        """PLANT consumption (the law of :meth:`plant_charge_intent`, #193).
        Integer state only, no random number. A unit carrying no charge,
        already planting or winding up a throw, or whose statuses forbid
        acting, does nothing and emits nothing; otherwise a
        :class:`~simulation.events.PlantEvent` is emitted and the plant runs
        until ``plant_done_tick`` (with a 0-tick plant the charge lies down
        this very tick, :meth:`_advance_plant` being called next)."""
        # The decision is the gate's (verb_gates.can_plant, #195 1e): hands
        # busy, no charge carried, stunned / paralysed.
        if not verb_gates.can_plant(self, u):
            return
        # #29 U1: the unit's own plant, the HANDLING hook's accessor
        ticks = unit_traits.plant_ticks(u, self.weapons_tables.weapons.by_name)
        self.tick_events.append(PlantEvent(unit_id=int(u.id),
                                           ticks=int(ticks)))
        u.plant_done_tick = self.tick + ticks
        u.plant_fuse_ticks = int(fuse_ticks)

    def _advance_plant(self, u) -> None:
        """One tick of a plant under way: a unit that can no longer act
        plants NOTHING and keeps its count; on the plant's last tick the
        charge lies on the unit's centre tile as it stands now, owned by the
        unit, detonating ``plant_fuse_ticks`` after this tick, and one
        charge is spent. (A dead unit never gets here: the consume loop
        drops its plant.)"""
        if u.plant_done_tick < 0:
            return
        if not composed_flags(u).can_act:
            u.plant_done_tick = -1
            return
        if self.tick < u.plant_done_tick:
            return
        u.plant_done_tick = -1
        ty, tx = u.center_tile_y(), u.center_tile_x()
        if not charges.place(self, charges.PlantedCharge(
                tx, ty, u.charge_ammo, u.id,
                det_tick=self.tick + int(u.plant_fuse_ticks),
                approach=(ty, tx))):
            return        # the charge table is full (#29 P3a): count kept
        u.has_explosive -= 1

    def _release_grenade(self, u, kind, fuse_ticks, reach_q, dx_q,
                         dy_q) -> None:
        """The grenade of throw kind ``kind`` leaves ``u``'s hand: lob it
        along the Q16.16 vector ``(dx_q, dy_q)``, fusing ``fuse_ticks`` ticks
        (clamped to the hand grenade row's fuse range) later, and spend one
        of that kind. Reuses the shipped
        :class:`~simulation.combat.Projectile` (the WEGO grenade path); the
        target tile is derived deterministically from the fixed-point
        direction."""
        if u.grenade_counts[kind] <= 0:
            return
        hand = self.weapons_tables.weapons.by_name["hand_grenade"]
        cx = u.center_tile_x()
        cy = u.center_tile_y()
        # v0.1 throw reach: a fixed tile distance along the aim (feel dial for
        # Erik; the grenade's own fuse/blast are the shipped numbers). A
        # controller throwing at a point (#121, the bots) names its reach in
        # Q16.16 tiles, clamped to [0, the hand grenade's max_throw_range].
        if reach_q is None:
            reach_q = 6 << _intents.FP_SHIFT  # tiles
        else:
            max_q = int(hand.max_throw_range) << _intents.FP_SHIFT
            reach_q = min(max(0, int(reach_q)), max_q)
        # #29 P3e: the target in Q16 integers -- the centre plus the Q16
        # direction times the Q16 reach (the float product of two n/65536
        # values was exact; this is its Q16 rounding, half away), plus the
        # half tile; the Projectile reads it exactly.
        half = 1 << (_intents.FP_SHIFT - 1)
        tgt_x_q = ((cx << _intents.FP_SHIFT) + half
                   + unit_fixed.mul_q16_round(int(dx_q), reach_q))
        tgt_y_q = ((cy << _intents.FP_SHIFT) + half
                   + unit_fixed.mul_q16_round(int(dy_q), reach_q))
        lo = fuse_ticks_from_seconds(hand.fuse_min_seconds, self._tps)
        hi = fuse_ticks_from_seconds(hand.fuse_max_seconds, self._tps)
        proj = Projectile(
            ORDER_GRENADE,
            cx, cy,
            unit_fixed.dequantize_scalar(tgt_x_q),
            unit_fixed.dequantize_scalar(tgt_y_q),
            fuse_seconds=0.0,
            thrown_tick=self.tick,
            ammo_name=self.weapons_tables.hand_grenade_kind(kind),
            owner_id=getattr(u, "id", -1),
            fuse_ticks=min(max(int(fuse_ticks), lo), hi),
        )
        self.projectiles.append(proj)
        u.grenade_counts[kind] -= 1

    def _direct_use(self, u) -> None:
        """USE consumption: flip the ``want_open`` latch of the first live door
        whose span touches the unit's footprint or its immediate ring (§3c).
        Reuses :meth:`door_at`; the slot-9e door sweep applies/retries the
        latch, exactly like the dev O-key."""
        door = verb_gates.use_door(self, u)         # the scan is the gate's
        if door is not None:
            door.want_open = not door.want_open

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
        u.current_mag = MAG_UNBOUND
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
            if u.is_zombie or not u.alive:
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
                    mo_type = move_orders[0].order_type     # #29 U1

                    for mo in move_orders:
                        speed = _ticks_per_tile(mo.order_type)
                        mo_type = mo.order_type
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
                            # #140: the builder adds the wind's cadence
                            # scale, sampled here at ORDER time (WEGO's
                            # plan-time precompute -- an accepted gap).
                            # G5p-a2: the unit's effort scale, read at
                            # order time like the wind.
                            samples = footprint_samples(
                                gmap, tile_y, tile_x, fp,
                                effort_scale_q16=unit_impulse.effort_scale_q16(
                                    unit_impulse.table_for(gmap), u),
                                # #29 U1: a sprint order is the full speed,
                                # every other order the walk
                                speed_hook_q16=unit_traits.hook(
                                    u, unit_traits.SPRINT
                                    if mo_type == ORDER_SPRINT
                                    else unit_traits.WALK))
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

    # ------------------------------------------------------------------
    # ray-engine-v2 P6b: THE RENDERER'S LIGHT (design v3 §7.1; the P6b brief
    # §1.5). Light is REQUESTED by the game whenever it renders and never by a
    # headless run, a bench or RL (P6a decision 2). The two inputs are RENDER
    # STATE -- the cursor lamp, the selected unit's flashlight mode, the
    # renderer's W6 effect queue -- so `light_q` is not sim-pure until P7
    # separates them (CLAUDE.md); none of this touches a synced field.
    # ------------------------------------------------------------------
    def set_light(self, requested: bool, cones=None, sky=None,
                  emit_mult=None) -> None:
        """Request (or stop requesting) the light channels, and hand the
        cone-emitter rows ((n, 7) int64, renderer/frame_lights.py), the sky
        ((16, 3) int64) and (graphics G3 F4b) the fire's light-flicker rows
        ((n, 3) int64 (y, x, mult_q), frame_lights.emit_mult_rows; None = no
        flicker) the next light computation reads. All three are RENDER STATE:
        none touches a synced field, and the flicker scales light only. No-op
        without a physics runner."""
        self._light = (bool(requested), cones, sky, emit_mult)
        self._apply_light()

    def _apply_light(self) -> None:
        if self.physics_runner is None:
            return
        requested, cones, sky, emit_mult = self._light
        self.physics_runner.engine.light_requested = requested
        self.physics_runner.set_light_cones(cones)
        self.physics_runner.set_light_sky(sky)
        self.physics_runner.set_light_emit_mult(emit_mult)

    def relight(self) -> None:
        """Recompute the light field on the current state WITHOUT a tick (the
        paused renderer's path; PhysicsRunner.relight). Writes only the three
        light planes. No-op without a runner or without a light request."""
        if self.physics_runner is not None and self._light[0]:
            self.physics_runner.relight(self.gmap)

    @property
    def light_serial(self) -> int:
        """Bumped by every light-carrying tick and every relight: the renderer
        re-uploads its light textures when this changes (once per tick)."""
        return 0 if self.physics_runner is None else int(self.physics_runner.light_serial)

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

    def visible_enemy_ids(self, team: int | None = None) -> tuple:
        """Enemies ``team`` can currently see — TEAM VISION, the union of its
        members' cones (onephase_wego design §8).

        THE fog-of-war gate: fog in v1 is visibility gating only, so the
        renderer draws no enemy whose id is absent here. Consult it only when
        ``ruleset.fog_of_war`` is set — the shipped rulesets have no vision
        model and show everything, exactly as they always have.
        """
        from simulation import vision
        if team is None:                    # the human's squad (#119)
            team = self.player_team
        return vision.visible_enemy_ids(self, team)

    def round_start_tick(self) -> int:
        """Absolute tick at which the CURRENT round began. The anchor every
        scheduled action resolves against (§12: "detonate 1.8 s into the
        round" is ``round_start_tick() + round(1.8 * tps)``)."""
        return self.tick - self.round_tick

    def round_end_tick(self) -> int:
        """Absolute tick at which the CURRENT round ends — i.e. the next
        planning pause.

        The sibling of :meth:`round_start_tick`, and the moment the planning UI
        cares most about after "now": everything a plan does past this tick
        happens in a round the player will get to re-plan first.
        """
        return self.round_start_tick() + self.ticks_per_round

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
            if u.is_zombie or not u.alive:
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
          1c. Exposure decays (G4p)
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
        if not self._step_pre_physics():
            return
        self._step_post_physics(self._step_physics())

    # #29 P5: the tick in three parts (pure code motion), so a ResidentBatch
    # (simulation/resident_batch.py) runs every env's host slots around ONE
    # batched slot 7. step() is the three in order; nothing else calls them.
    def _step_pre_physics(self) -> bool:
        """Slots 1 .. 6b. False (and nothing done) while paused."""
        if self.paused:
            return False

        # 1. Fresh event buffer.
        self.tick_events.clear()
        # #88 P2 (design §5.1, critique B7): the Python trace books (FieldEdit
        # deposits, the saturation's clips at the Python sites) are per tick,
        # over the window this step brackets -- reset BEFORE the 6b flush that
        # writes them. (The engine's twins reset at the head of physics.)
        self.gmap.reset_trace_books()

        # Round-clock head (control-modularity P1, §3a): tick-0-only work
        # (DET_START_PHASE1, path-offset reset, initial unit stamp under
        # TwoPhaseWEGO) now lives behind the ruleset; verbatim body in
        # simulation.ruleset.TwoPhaseWEGO.on_round_start.
        self.ruleset.on_round_start(self)

        # 1a. Integrate + turn (G5p-a2 / G5p-a1, simulation.unit_impulse and
        # simulation.turning): every living unit, in id order, integrates the
        # impulses it received since last tick (its sway, pushed velocity,
        # effort, reaction delay and the knockdown checks; the displacement
        # itself is slot 3's), then takes one tick of the turning law toward
        # its target (the live aim while one is held) with the outside spin
        # added -- the ONLY writer of a facing. BEFORE 1b (a throw leaves
        # along the turned facing; a shove reads it), the movement slot (the
        # Speed table's direction rule reads THIS tick's facing) and the
        # shooting slot (a shot leaves along it; the bots' predictor
        # evaluates exactly this step before the tick).
        unit_impulse.integrate_all(self)

        # 1b. Direct-control intents (control-modularity P3, §3c): consume the
        # possessed units' held/latched per-tick verbs (AIM/THROW/USE/TRIGGER)
        # BEFORE projectiles advance, so a THROW lobbed this tick travels this
        # tick like a WEGO grenade and a TRIGGER arms slot 4's fire order.
        # Dormant under WEGO (no possessed units) — a single set-empty check.
        self._consume_direct_intents()

        # 1b'. The commander's orders (#195 1f): this tick's latched
        # ``order_intent`` requests are written onto their units (the ONE
        # writer of the order columns, unit_orders). Information only: no
        # slot reads an order. Dormant on an empty request dict.
        unit_orders.apply_requests(self)

        # 1c. Exposure decays one tick (G4p, simulation.exposure) -- before
        # every attack of this tick, so each hit reads the level left by the
        # hits up to last tick, decayed exactly once since.
        exposure.decay_units(self.units)

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

        # 4's phase-B record (#169): every attack of slot 4 -- shots, strikes,
        # beams and 4b's spray deposits -- is decided against the slot-start
        # state and recorded here; ``resolve()`` after 4b applies them in the
        # walk's order, so no unit's attack this tick depends on its id.
        attack_effects = SimultaneousEffects()

        # 3 + 4. Unit simulation. A ruleset that owns these slots
        # (OnePhaseWEGO — ``drives_units``) replaces BOTH with one call: its
        # compiled timeline decides movement and shooting together, because
        # under that ruleset they are two readings of the same schedule
        # (onephase_wego design §3). The legacy path below is untouched and
        # still runs verbatim for every other ruleset.
        if self.ruleset.drives_units:
            # G5p-a2: the pushed velocity moves every unit under every
            # ruleset (slot 3's displacement), then the timeline plays.
            self._displace_pushed()
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
                             self.shots, self.real_time, self.draws,
                             events=self.tick_events, bullets=self.bullets,
                             queue=self.edit_queue, effects=attack_effects)

        # 4a. The MELEE verb (gameplay pack G6p): the units' latched
        # ``melee_intent`` blows, decided in the same phase A as the shots
        # (against the slot-start state) and recorded on the same effects, so
        # a blow and a shot of one tick land together at 4c. Under every
        # ruleset (the latch is consumed even where slot 4 is the timeline's).
        process_melee_intents(self.units, self.tick, self.weapons_tables,
                              self.impulse_table, events=self.tick_events,
                              effects=attack_effects)

        # 4b. SPRAY deposits (mechanics/03 §5, W4) — the shooting slot's
        # second half: every active spray burst enqueues its aimed heat/gas
        # cone into the edit queue (flushed at 6b with everything else, so
        # this tick's flame heat converts to temperature THIS tick). Writes
        # fields only (two-terminals invariant); draws no RNG; dormant (one
        # attribute read per unit) when no spray weapon is in play.
        # W6: tick_events rides along for the RENDER-ONLY SprayJetEvent
        # (the flame-jet visual) — not digest-hashed, pure synced-state fn.
        process_sprays(self.gmap, self.units, self.edit_queue,
                       events=self.tick_events, effects=attack_effects)
        # 4c. Phase B of the attack slot (#169): apply what 4 and 4b recorded.
        attack_effects.resolve()

        # 5. Zombie AI — its ONE driver: the conductor, or a match's zombie
        # controller through the ZOMBIE intent (#122, :meth:`drive_zombies`).
        self._run_zombie_slot()

        # 6. Re-stamp obstacles.
        self.gmap.stamp_units(self.units)

        # 6b. Flush the field-edit queue (engine/13). The weapon / fire /
        # explosion phases above enqueued their FieldEdits (smoke / atmosphere /
        # wave_source / fire / heat deposits) via :meth:`edit`; we apply them ALL
        # here, in one deterministic stable-sorted pass, BEFORE the physics
        # solvers run — so the solvers advect / propagate the settled NET deposit
        # for this tick (a laser burn-off and a grenade cloud issued the same
        # tick both land before smoke advection). This is the single RNG consumer
        # for noise>0 edits, keyed draws (seed, tick, edit key, tile; P3b).
        # #29 P4c: on the resident path the DEVICE applies this flush (the
        # device edit queue, applied by slot 7 right after its entry upload):
        # the runner takes the queue in the same order and packs it, or
        # flushes here on the host when the device cannot serve the tick.
        runner = self.physics_runner
        if runner is not None and runner.serves_edit_queue():
            runner.serve_edit_queue(self.gmap, self.edit_queue, self.draws)
        else:
            self.edit_queue.flush(self.gmap, self.draws)
        return True

    def _step_physics(self):
        """Slot 7: the physics step; returns the burn-through walls."""
        # 7. Physics.
        sim_time_per_tick = 1.0 / float(self._tps)
        destroyed = []
        if self.physics_runner is not None:
            # D4 (ruling amendment 5): the SIM TICK reaches the fire cast so the
            # 8-ray fan's phase rotates with it — without that rotation the fan
            # casts the same 8 directions from every source forever and some
            # tile pairs are NEVER connected (measured: the (+2,0) axis
            # neighbour). `self.tick` is synced state, so this is save/load- and
            # lockstep-safe (never a wall-clock or call-count counter).
            destroyed = self.physics_runner.step(
                self.gmap, sim_time_per_tick, tick=self.tick)
        return destroyed

    def _step_post_physics(self, destroyed) -> None:
        """Slots 9 .. 10, given slot 7's burn-through walls."""
        sim_time_per_tick = 1.0 / float(self._tps)
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

        # 9c2. Wind on units (#140's read, the `wind | max` row, mechanics/05
        # §1 -- exchange.apply_wind_force; G5p-b): the dynamic pressure
        # q = 1/2 rho v^2 at each living unit's footprint
        # (GameMap.footprint_wind) is a FORCE on the unit impulse law: one
        # tick of impulse q * drag_area along the wind, through
        # unit_impulse.add_wind, integrated in the next tick's slot 1a (whose
        # outcomes -- slowed, pushed by the excess, knocked down -- are the
        # wind's; a blast's wind included). It writes no position, status
        # or speed. WITHIN-TICK EXCHANGE ORDER (documented contract): heat
        # damage (9c) FIRST, then wind -- a unit the heat row kills this
        # tick is a corpse.
        if self.physics_runner is not None:
            apply_wind_force(self.units, self.gmap, self.impulse_table)

        # 9c3. Gas coupling rows (mechanics/05 §1 gas[teargas] / gas[poison]
        # — exchange.COUPLING_TABLE[3..4], W3). Read the post-physics gas
        # planes at each living unit's footprint: teargas above threshold
        # applies BLINDED (snap-cone fire from next tick — the step-2b
        # trigger-position semantics); poison above threshold emits one
        # POISON packet per tick through the mechanics/06 pipeline (zombies
        # immune). WITHIN-TICK EXCHANGE ORDER (documented contract): heat
        # (9c), wind (9c2), teargas, poison. Lazy: an all-zero plane is one
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
            # Continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md §2.4):
            # ignition and sustain read ONE law — a tile ignites only where the
            # local O2 MOLE FRACTION X exceeds the extinction limit X_ext, the
            # same o2_frac_ext the fire logistic uses. (Retires the old absolute
            # o2_threshold gate so ignition cannot seed a tile the next fire step
            # would immediately suffocate.)
            o2_frac_ext = float(getattr(fire_cfg, "o2_frac_ext", 0.13))
            apply_temperature_ignition(self.gmap, o2_frac_ext, ignition_seed)

        # 9e. Entities — the SPLIT GATE (Arc B impl doc §2b, D2). Two
        # independently-gated pieces:
        #  · the door STRUCTURAL sweep runs on ANY door level (gated on
        #    `self._doors`, unchanged from Arc A) — external-destruction
        #    reconciliation + the synced want_open latch, ordinal order;
        #  · the LOGIC block runs iff the SignalBus exists (wires present),
        #    so a door-only wire-free level is byte-identical to Arc A (the
        #    dormancy guarantee, §8);
        #  · the VENT circulation sweep (issue #48) is a THIRD independently-
        #    gated piece, gated on `self._ducts` alone — vents carry no wired
        #    inputs in patch 1, so it runs even on a bus-free level.
        # Sub-order per tick when the bus exists (§2b): (a) sample + emit —
        # write every wired door's is_open (and wired entities' free alive)
        # into pub BEFORE the logic sweep; (b) logic sweep — each node in
        # ordinal order reads pub, writes stg[out] (B2); (c) input resolve —
        # drive wired doors' want_open (OR/held, close-beats-open); (d) actuator
        # sweep — pumps (N-feed edit, B4), then vents (circulation N-feed edit,
        # #48, bus-independent), then the door structural sweep; (e) swap
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
        # Vent circulation N-feed edit (vent system PATCH 1, issue #48) — a
        # sibling of the pump sweep at the SAME 9e(d) actuator slot, but
        # OUTSIDE the SignalBus gate: vents carry no wired inputs in patch 1
        # (design §2), so a wire-free duct/vent level still circulates.
        # Ordered after pumps (both are gas-mass actuators; pumps are the
        # explicit player-controlled case) and BEFORE doors, like pumps.
        if self._ducts:
            sweep_vents(self)               # (d) vent circulation edit — BEFORE doors
        if self._doors:
            sweep_doors(self)               # (d) door structural sweep
        # Timed charges (#31 P5): wire-free like vents, so OUTSIDE the bus
        # gate; AFTER the door sweep so a blast this tick meets this tick's
        # door state, and its wall damage lands before the recorder snapshot.
        # Its queued FieldEdits land at the next step's 6b flush (the door
        # charge's slot-8 latency).
        if self._timed_charges:
            sweep_timed_charges(self)       # (d') timed-charge detonations
        # Heat sources (vocabulary E.2 row 3): LAST of the actuator
        # sub-slot, after the timed charges, so every blast of this tick
        # (slots 2, 8 and the charges just above) is in gmap.blast_log when
        # the sources read it for damage; then each living source enqueues
        # its per-tick heat (lands at the next 6b flush). A source-free level
        # only clears the log, which is empty at every tick boundary.
        # Planted charges (#193): the continuous path's ONE driver of
        # ``charges.tick`` -- a ruleset that owns the unit slots
        # (OnePhaseWEGO, ``drives_units``) ticks them at the head of its own
        # timeline instead, so no charge is ever ticked twice. Beside the
        # timed charges and for the same reasons (this tick's door state;
        # the wall damage before the recorder snapshot; the edits at the
        # next 6b), and BEFORE the heat sources so its blast is in
        # gmap.blast_log when they read it.
        if self.planted_charges and not self.ruleset.drives_units:
            charges.tick(self)              # (d') planted-charge detonations
        if self._heat_sources:
            sweep_heat_sources(self)        # (d'') heat-source blasts + deposits
        elif self.gmap.blast_log:
            self.gmap.blast_log.clear()
        if self._signal_bus is not None:
            self._signal_bus.swap_node_signals()   # (e) pub[node-slots] ← stg

        # 9f. Gas energy mirror refresh -- RETIRED at arc #54 P-G1b.
        #
        # P-G0 needed a second whole-grid re-derive here because 9e's
        # pump / vent / door sweeps run AFTER physics_runner.step() returns and
        # move gas N, which left the mirror-derived `gas_energy` one tick stale
        # on exactly the levels that exercise those seams. P-G1b makes every
        # one of those sweeps an ENERGY writer -- the pump primitives carry
        # `dN*T_abs` through the gas-energy seam, seal/unseal move and retire,
        # `on_tile_changed` books the membership flips -- so the field is never
        # stale, and re-deriving it from the mirror here would now DESTROY the
        # very state those seams just wrote (design §2.6's `N*floordiv(E,N)`
        # drip, plus any sub-count deposit the mirror cannot represent).

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
            # P-R4 (ruling A1.7): `rad_net` is the SIGNED radiation accumulator
            # and shares `heat`'s per-tick lifetime exactly — filled by the
            # fire-plane cast at the top of the physics step, consumed by the
            # temperature solver's Pass-1 fold in the same step, wiped here so
            # the next tick's cast starts clean. In-place (never reassigned) so
            # any C++ view of the buffer stays valid.
            self.gmap.rad_net.fill(0)
            # P-F1a (rule 4): the SKY LEDGER shares that lifetime exactly — it
            # is the counterparty half of the same per-tick booking, so the two
            # planes MUST be cleared together or the identity
            # `rad_net.sum() + rad_amb.sum() == 0` would be evaluated across
            # mismatched ticks. Filled by the emission cast at the top of the
            # physics step, read (pre-fold) by the ledger gate, wiped here.
            self.gmap.rad_amb.fill(0)
            # D3: the radiant-flux SENSOR shares the same per-tick lifetime —
            # filled by the cast at the top of the physics step, consumed by
            # unit heat damage above, wiped here. (`dem_acc` does NOT belong
            # in this clear: it is PERSISTENT synced state whose whole job is
            # to carry a sub-count oxygen debt ACROSS ticks.)
            self.gmap.rad_flux.fill(0)
            # Ray-engine-v2 P2a (design v3 row 38): the shadow sweep's FOUR
            # planes are NOT wiped here. `RadiationSweep::run` zeroes them
            # itself before its first ordinate (it accumulates over ordinates,
            # so it had to clear them anyway), which means they still hold this
            # tick's values at render time — the tile inspector's Φ, `f` and
            # E°⁻¹(Φ) rows were reading a wiped zero while the wipe lived here.
            # Nothing digested changes: none of the four is in DIGEST_FIELDS or
            # SIM_FIELDS. The three LIVE planes above keep their conductor wipe
            # because their writer (the old cast) does not clear them.

        # #123: the per-tick deposit credit (damage.TickCredit) shares the
        # `heat` buffer's lifetime -- it names who put THIS tick's heat / gas
        # deposit on a tile, read by 9c / 9c3 above. Cleared here, after both
        # readers and before the round-clock tail, so a door charge fired in
        # on_tick_end (its edits flush at the next 6b) keeps its claim into
        # the next tick's readers. Outside the physics gate: claims are made
        # with or without a runner.
        tick_credit(self.gmap).clear()

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

        # 10'. An order whose tick has come is cleared (#195 1f) -- right
        # after the advance, so between ticks a held order has time left.
        unit_orders.expire(self)

        self.ruleset.on_tick_end(self)

        # 11. The team picture (#195 1d; a match only): each marine team's
        # records of what it sees at the tick the next controllers run at --
        # after the tick advance and on_tick_end, so a conversion this step
        # made is seen. Once per tick, never on demand.
        if self.team_picture is not None:
            self.team_picture.update(self)

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
                    fy, fx = proj.target_tile()           # #29 P3e: integers
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
                        payload, self.draws, events=self.tick_events,
                        kind="grenade", owner_id=proj.owner_id,
                        approach=(unit_fixed.shr_trunc(proj.start_y_q),
                                  unit_fixed.shr_trunc(proj.start_x_q)))

        # W2: advance in-flight kinetic rounds (the unified march). Each
        # marches this tick's integer step budget; hits/chew/exposure resolve
        # inside advance() (keyed draws, #29 P3b). Survivors stay.
        # W3: the edit queue rides along — an in-flight payload round (the
        # GL-6 40 mm) detonates at its stop through the payload executor.
        # #169: two rounds landing on one tick land TOGETHER — every round
        # marches against the slot-start state and records its hit / chew /
        # detonation; resolve() applies them in list (spawn) order.
        if self.bullets:
            survivors = []
            landing = SimultaneousEffects()
            for b in self.bullets:
                if b.advance(self.gmap, self.units, self.shots,
                             self.real_time, self.draws,
                             events=self.tick_events,
                             queue=self.edit_queue, effects=landing):
                    survivors.append(b)
            landing.resolve()
            self.bullets = survivors

    def _update_player_movement(self) -> None:
        """Slot 3: THE displacement loop (G5p-a2, design §13) over every
        living unit in id order, then each player unit's legacy WEGO path.

        - A unit holding a live MOVE_DIR walks (:meth:`_step_move_dir`): its
          walk and its pushed velocity move it as ONE displacement
          (``unit_impulse.move``).
        - Every other unit with a pushed velocity -- standing, prone, a
          zombie, a WEGO unit -- moves by it alone; the pushed velocity moves
          a unit whatever its statuses say about moving on its own. A zombie
          so displaced reads the horde field at its floored tile in slot 5
          (G9p: no path to drop; its step snaps to the tile grid).
        - Then the WEGO path step (LEGACY), which overwrites a push (an
          accepted gap, the Control seams row).

        Each move reads only the map and the unit's own state, so the loop
        is order-free (units pass through each other, #142).

        Status gate (mechanics/06 §4): a unit whose composed ``can_move`` is
        suppressed (knocked down / immobilized / paralyzed) holds position
        and its precomputed path PAUSES — ``path_tick_offset`` shifts by one
        so the path is not consumed while down; on release the unit resumes
        at the next un-walked path index (no catch-up teleport; the round
        may end before the tail is walked — being knocked down costs the
        distance). With no statuses this is a dead path, bit-identical to
        pre-P3 behavior. A unit in its reaction delay
        (``unit_impulse.walk_allowed``) is held the same way.
        """
        gmap = self.gmap
        for u in self.units:
            if not u.alive:
                continue

            # Direct-control branch (control-modularity P3, §3c): a unit with a
            # live MOVE_DIR intent this tick moves by velocity, footprint-
            # collision-checked with the SAME predicate A* uses, and NEVER
            # replays a WEGO path. Dormant under WEGO (no unit carries
            # ``live_move_dir``), so the branch below is untouched and
            # byte-identical.
            move_dir = None if u.is_zombie else getattr(u, "live_move_dir",
                                                        None)
            if move_dir is not None:
                self._step_move_dir(u, move_dir)
                continue

            # No walk of its own this slot: the pushed velocity alone.
            if u.pvx or u.pvy:
                unit_impulse.move(gmap, u)
            if u.is_zombie:
                continue

            if (not composed_flags(u).can_move
                    or not unit_impulse.walk_allowed(u)):
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
                # #29 P3c: the precomputed path's interpolated position is
                # quantized ONCE at the float door (legacy WEGO: its
                # (st + 1) / step_ticks fractions are off the Q16 grid).
                u.set_pos(px, py)

    def _displace_pushed(self) -> None:
        """Slot 3 under a ruleset that drives the units itself (OnePhaseWEGO,
        ``drives_units``): every living unit with a pushed velocity moves by
        it (``unit_impulse.move``), before the ruleset's compiled timeline
        (which then overwrites it: an accepted legacy gap). The same rule as
        the pushed branch of :meth:`_update_player_movement`."""
        for u in self.units:
            if u.alive and (u.pvx or u.pvy):
                unit_impulse.move(self.gmap, u)

    def _step_move_dir(self, u, move_dir) -> None:
        """MOVE_DIR consumption (control-modularity P3, §3c) — the real new
        mechanic: advance ``u`` one tick along its live direction by a
        mobility-table-scaled velocity, footprint-collision-checked with the
        SAME predicate A* uses (:meth:`GameMap.is_passable_block`), so direct
        move and A* agree on both speed and what blocks.

        Speed (G1p-c): the ``[speeds]`` table's marine walk and full speed
        (``self.speed_table``, Q16 ticks per tile, quantized once from m/s),
        each scaled by the footprint's area-average mobility, the wind and
        (G5p-a2) the unit's effort exactly as every walker is
        (:func:`movement.default_speed`) — a furniture tile is 2.5x slower
        for both — then the direction rule and the intent's throttle
        (:func:`movement.move_speed_q16`, the rational rounded once to Q16
        tiles per tick): the walk slows sideways and backward, the throttle's
        extra speed is forward only. ``theta`` is the angle between the move
        and the facing THIS tick (slot 1a already turned it, G5p-a1); a unit
        holding no aim makes its travel direction its facing TARGET and turns
        there with inertia like everyone else, so for a few ticks after a
        change of direction it walks at the sideways / backward factor.

        The walk runs only while the unit's statuses let it move
        (``can_move``) and it is not catching itself after a push
        (``unit_impulse.walk_allowed``); either way its pushed velocity is
        applied. Walk and push move it as ONE displacement
        (``unit_impulse.move``): on the Q16 grid, in sub-tile substeps (one,
        today's single test, at a walk alone), each the slide rule -- the
        full move; else along X alone; else along Y alone; else hold -- so
        pushing diagonally into a wall glides rather than dead-stops.

        Determinism: integers only; ``u.x`` / ``u.y`` leave as exact n/65536
        doubles (synced digest fields).
        """
        walking = (composed_flags(u).can_move
                   and unit_impulse.walk_allowed(u))
        wdx = wdy = 0
        if walking:
            fp = u.footprint
            samples = footprint_samples(
                self.gmap, u.tile_y, u.tile_x, fp,
                effort_scale_q16=unit_impulse.effort_scale_q16(
                    self.impulse_table, u),
                speed_hook_q16=unit_traits.hook(u, unit_traits.WALK))
            # G1p-c: the walk and the full speed (Q16 ticks per tile), each
            # reduced by terrain, wind and effort -- and (#29 U1) the unit's
            # own WALK / SPRINT hook -- then the direction rule and the
            # throttle as ONE exact rational, rounded once to Q16.
            table = self.speed_table
            cos_q = facing_cos_q16(move_dir.dx_q, move_dir.dy_q, u.facing_q)
            v_q = move_speed_q16(
                default_speed(samples, table.walk_cost_q),  # traits-class: WALK
                default_speed(_dc_replace(samples, speed_hook_q16=(
                    unit_traits.hook(u, unit_traits.SPRINT))),
                    table.full_cost_q),  # traits-class: SPRINT, the samples' hook
                table.side_q, table.back_q, move_dir.throttle_q, cos_q)
            wdx, wdy = unit_impulse.walk_step_q(move_dir, v_q)
        unit_impulse.move(self.gmap, u, wdx, wdy)

        # Facing: a held aim wins (slot 1a's target); otherwise the
        # direction of travel becomes the facing TARGET (G5p-a1: turned
        # toward with inertia from the next tick's 1a, never written here).
        if walking and getattr(u, "live_aim", None) is None and (
                move_dir.dx_q != 0 or move_dir.dy_q != 0):
            turning.set_target(u, unit_fixed.atan2_q16(-int(move_dir.dy_q),
                                                       int(move_dir.dx_q)))

    def _end_round(self) -> None:
        """End-of-round teardown — convert dead-by-zombie marines, reset state.

        Lifted from ``game.py:_end_execution`` (lines 2014-2043). Bumps
        the turn counter, converts marines, snaps floats, clears orders,
        refills AP, and rewinds the tick counter for the next round.
        Auto-pause happens in :meth:`step` (the caller).
        """
        convert_marines_to_zombies(self.units)

        for u in self.units:
            # Snap the position to the nearest integer tile boundary (the
            # same round() as before #29 P3c, on the Q16 position's view).
            u.set_pos_q(round(u.x) << 16, round(u.y) << 16)
            u.clear_orders()
            u.move_path = []
            u.last_fire_tick = -999
            # G5p-a2: the shove's cooldown rewinds with the tick counter (the
            # last_fire_tick precedent), and a push does not survive the
            # round boundary's snap to the tile grid.
            u.last_shove_tick = -1
            u.last_blow_tick = -1             # G6p: the melee cadence, same
            unit_impulse.clear(u)
            # W3 ammo economy: the round boundary tops everyone off (the
            # WEGO planning pause is a between-rounds breather — v1 rule,
            # Erik's dial later). Also REQUIRED for correctness: the tick
            # counter rewinds to 0 below, so a carried reload_done_tick
            # from late in the round would stall the unit deep into the
            # next one (the exact hazard last_fire_tick = -999 solves).
            # None = full mag (the lazy first-trigger bind, combat.mag_gate);
            # untracked (mag_size 0) units never read either field.
            u.current_mag = MAG_UNBOUND
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
        # Rewind for the next round. #29 P3b: the draws' clock carries on
        # past the rewind (legacy TwoPhaseWEGO), so a round never repeats
        # the previous one's draws.
        self.draws.tick_base += self.tick
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
            if u.is_zombie:
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
                        owner_id=getattr(u, "id", -1),
                    )
                    self.projectiles.append(proj)


__all__ = ["Simulation", "SimState"]
