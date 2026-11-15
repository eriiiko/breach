---
paths:
  - "src/simulation/simulation.py"
  - "src/simulation/physics_runner.py"
  - "cpp/src/physics_engine.cpp"
  - "cpp/src/physics_engine.h"
  - "config.py"
  - "src/simulation/recorder.py"
  - "src/simulation/entities/**"
  - "src/simulation/sensor_accessor.py"
  - "src/simulation/signal_bus.py"
  - "src/temperature_scale.py"
---

# Sim core: the facade, the conductor, the runner

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Tick conductor
- A new system is ONE call line plus an ordering comment in `Simulation.step()`'s numbered slots, never a logic block (god-file policy).
- Slot 1a is `unit_impulse.integrate_all`: every living unit's impulse integration, then its turn, before the intents (1b), movement (3) and shooting (4). Slot 3 is the ONE displacement loop (Unit impulse).
- Slot 2 is two calls: `_update_grenades` (every grenade flies, a fuse-out detonates at once, list order) THEN `_update_rounds` (slot 2': the rounds march against the state the grenades left; the landing resolves together). Their order is load-bearing; each has its own profiler label, so the device port takes them one at a time.
- Slot 5 (`_run_zombie_slot`) is the zombie law's ONE driver per tick: over every zombie on a game path, or in a match only the ids the zombie controller named through `drive_zombies` (#122). It drains the noise log every tick first, driven or not.
- Slot 9e(d') fires the planted charges (`charges.tick`): the continuous path's ONE driver, skipped when the ruleset `drives_units` (OnePhaseWEGO's timeline ticks them).
- Slot 11, the very last (after the tick advance and `on_tick_end`), updates the team picture, in a match only.

## PhysicsRunner / PhysicsEngine
- New C++ orchestration lands in `PhysicsEngine`, not Python glue; solver params are bound in PhysicsRunner only.

## Recorder
- Frozen `.npz` contract: extend `DEFAULT_FIELDS` additively. A new DTYPE CLASS (int64) is a contract extension with its own ring branch, never a cast: a float64 ring is exact only to 2^53 and would drop the LSBs the energy gates assert on (arc #54 D13).
- It is not the replay path (Match recording / replay is re-simulation).
