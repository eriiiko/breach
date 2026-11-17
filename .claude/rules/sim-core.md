---
paths:
  - "src/simulation/simulation.py"
  - "src/simulation/physics_runner.py"
  - "cpp/src/physics_engine.cpp"
  - "cpp/src/physics_engine.h"
  - "config.py"
  - "src/simulation/recorder.py"
  - "src/simulation/entities/**"
  - "src/simulation/door_system.py"
  - "src/simulation/sensor_accessor.py"
  - "src/simulation/signal_bus.py"
  - "src/temperature_scale.py"
  - "src/debug_keys.py"
---

# Sim core: the facade, the conductor, the runner

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Tick conductor
- A new system is ONE call line plus an ordering comment in `Simulation.step()`'s numbered slots, never a logic block (god-file policy).
- Slot 1a is `unit_impulse.integrate_all`: every living unit's impulse integration, then its turn, before the intents (1b), movement (3) and shooting (4). Slot 3 is the ONE displacement loop (Unit impulse).
- Slot 2 is two calls: `_update_grenades` (every grenade flies, a fuse-out detonates at once, list order) THEN `_update_rounds` (slot 2': `_march_rounds` marches every round against the state the grenades left, recording on the landing it is handed, then the landing resolves together). Their order is load-bearing; each has its own profiler label (slot 2''s is the march; its landing is timed as "4c resolve"), so the device port takes them one at a time.
- Slot 5 (`_run_zombie_slot`) is the zombie law's ONE driver per tick: over every zombie on a game path, or in a match only the ids the zombie controller named through `drive_zombies` (#122). It drains the noise log every tick first, driven or not.
- Slot 9e(d') fires the planted charges (`charges.tick`): the continuous path's ONE driver, skipped when the ruleset `drives_units` (OnePhaseWEGO's timeline ticks them).
- Slot 11, the very last (after the tick advance and `on_tick_end`), updates the team picture, in a match only.

## PhysicsRunner / PhysicsEngine
- New C++ orchestration lands in `PhysicsEngine`, not Python glue; solver params are bound in PhysicsRunner only.

## Config reload seam
- After Ctrl+R `src/debug_keys.py` calls `CFG.reload()` and then `Simulation.on_config_reload()`, and that is THE place a live table re-binds, applying from the next tick. A new hot-reloadable table adds one call there: never a lazy identity check inside a reader (a debug tool must not decide which table is in force).

## Recorder
- Frozen `.npz` contract: extend `DEFAULT_FIELDS` additively. A new DTYPE CLASS (int64) is a contract extension with its own ring branch, never a cast: a float64 ring is exact only to 2^53 and would drop the LSBs the energy gates assert on (arc #54 D13).
- It is not the replay path (Match recording / replay is re-simulation).

## Door table
- `door_system.sweep_doors` (slot 9e(d)) has a device twin, the applier's DOORS stage over `cpp/src/cuda_unit_doors.cuh` (the Door table, `.claude/rules/device-units.md`): a change to the sweep, `_occupancy_clear`, the hp copy / restamp or a `DoorRuntime` attribute that crosses a tick lands there and in `unit_schema.DOOR_TABLE` in the SAME patch. The corpus has no door entity, so the twin ledger traces none of it; `tests/cuda_unit_doors_check.py` is the guard.

## Entity system
- `Simulation._signal_emit`, `_resolve_door_inputs`, `sensor_system.sample_sensors`, `logic_nodes.sweep_logic_nodes` and `SignalBus.swap_node_signals` (slot 9e (a), (b), (c), (e)) have device twins, the applier's EMIT, LOGIC, DOOR_INPUTS and SWAP stages over the signal tables (`.claude/rules/device-units.md`): a change to one of them, to `aggregate_input`, an evaluator, `EntityFieldAccessor.sample` / `.area` or to a runtime attribute that crosses a tick lands there in the SAME patch. Each sub-step has its own profiler label (`tools/tick_stage_profile.py`). The corpus has no bus; `tests/cuda_unit_signals_check.py` is the guard.
- `entities/serialize.entity_records` is `record_head` + `record_tail` per entity: the device export (`unit_kernels.EntityExport`) composes the same two, so a change to the record format lands in those two functions, never in a copy.
