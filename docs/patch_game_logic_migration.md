# Patch Plan: Game Logic Migration from `game.py` to `main.py`

**Date:** 2026-05-16
**Status:** decisions locked, ready for review and then implementation
**Prerequisite reading:**
- `docs/game_py_inventory_and_migration_plan.md` (inventory of what exists today)
- `docs/architecture.md` §2 (the simulation/presentation boundary we're enforcing)

---

## Decisions locked this session

| # | Decision | Choice |
|---|----------|--------|
| A | Strict simulation / presentation boundary | **Now**, while we're moving things anyway |
| B | Unit class structure | **One `Unit` class** for now; `zombie` is a STATE on it, not a subtype. Inventory always on base. |
| C | Order type variety (MOVE_ATTACK / MOVE_COVER / SPRINT) | **Keep all** — they encode different speeds |
| D | Path: pre-computed vs per-tick | **Both** — pre-compute at order time (visualization), execute per-tick (movement) |
| E | PhysicsRecorder | **Keep verbatim** — active debugging workflow |
| F | Combat tempo | **Real-time with scheduled pauses** (the existing two-sub-rounds-per-round model). Real-time is canonical for AI training. |
| G | Inventory on units | **Yes** — base field, all units carry. Enables emergent zombie-grenade-carry stuff. |
| H | Order undo (Backspace) | **Yes** |
| I | "Phase" concept location | **Simulation layer** knows ticks + rounds; main.py owns the pause toggle. |
| J | Slow-mo / cinematics | **Defer** — but don't block it |

**Out of scope** (deferred design decisions):
- Single Unit class vs. split — chose single for v1. Revisit when adding worms / robots / animals.
- "Should all entities be units?" — yes for now. Doors, projectiles, decals stay as their own thing.
- Multiplayer / multi-player local — deferred.
- AI inference scheduling (how often the AI samples the world) — experiment when training begins.

---

## Architectural target

```
src/
  simulation/                  # NEW — gameplay logic, headless, deterministic
    gamemap.py                 # The world state container (tile + physics fields)
    unit.py                    # Unit class with marine/zombie state, inventory
    orders.py                  # Order types and the OrderQueue
    combat.py                  # Shooting, projectiles, apply_explosion
    ai_zombie.py               # Zombie activation + pathfinding orchestration
    physics_runner.py          # Wraps the C++ physics module per-tick
    simulation.py              # Facade: Simulation class — the AI training entry point
    recorder.py                # PhysicsRecorder (moved verbatim from game.py)
  renderer/                    # EXISTS — pyray rendering
  main.py                      # ENTRY — input + main loop, orchestrates sim + renderer
  game.py                      # LEGACY — deleted at end of migration
```

`Simulation` is the central API. Both `main.py` (human play) and a future
`train.py` (AI rollouts) talk to `Simulation` the same way:

```python
sim = Simulation(level_data)
sim.apply_action(unit_id, order)
sim.step()                # advance one tick
state = sim.get_state()    # what AI / renderer reads
```

`get_state()` returns the full `GameMap` + unit list. The renderer reads it
read-only; AI training serializes it.

---

## Migration order (13 steps, each independently runnable)

### Phase 1 — Lift simulation pieces out of game.py

Each step here is a refactor: copy code, adjust imports, keep behavior identical, run the smoke test.

**Step 1: `src/simulation/gamemap.py`**
Lift the `GameMap` class. Drop dead code (`_build_ship`). Levels load via `level_loader`. The shim in `main.py:GameMap` gets replaced with this.

**Step 2: `src/simulation/unit.py`**
Lift `Unit`. Inventory becomes a base field. Make zombie a state (`is_zombie: bool`). Marine-only fields stay on Unit but conditional on state.

After Step 2: `main.py` can instantiate marines and zombies. They can be rendered (already supported by `renderer.compose_world`). No orders yet.

**Step 3: `src/simulation/orders.py`**
Lift order classes (Order, MoveOrder with the three move modes, FireOrder, GrenadeOrder, ExplosiveOrder). Pure data — no execution logic here.

**Step 4: `src/simulation/combat.py`**
Lift `Projectile`, `Shot`, the shooting / line-of-sight functions. Functions take `gmap` + `units` as inputs, mutate them.

`apply_explosion` stays in the physics namespace (it is a physical event that has gameplay consequences — pressure waves, fire ignition, wall damage, unit blast). Splitting it would scatter event-effect logic and make calling code error-prone (easy to forget the damage step). The same pattern is already present in C++ (fire damages walls, atmosphere drains through breaches). Combat calls into `physics.apply_explosion(gmap, ...)` from grenades / explosives / weapon impacts — physics owns the "this is a physical event" entry point.

**Step 5: `src/simulation/physics_runner.py`**
Lift `PhysicsRunner` from main.py (already extracted, just move to its proper home).

**Step 6: `src/simulation/ai_zombie.py`**
Lift zombie activation, pathfinding orchestration, target selection, conversion-to-zombie logic.

**Step 7: `src/simulation/recorder.py`**
Move `PhysicsRecorder` verbatim. Hook into Simulation's tick step.

### Phase 2 — The Simulation facade

**Step 8: `src/simulation/simulation.py`**
Create the `Simulation` class. It owns the GameMap, the unit list, the order queue, the physics_runner, the recorder. Public API:
```python
class Simulation:
    def __init__(self, level_data, seed: int | None = None): ...
    def add_unit(self, unit: Unit, position: tuple) -> int: ...
    def apply_action(self, unit_id: int, order: Order) -> None: ...
    def undo_last_order(self, unit_id: int) -> None: ...
    def step(self) -> None: ...                  # advance one tick
    def get_state(self) -> SimState: ...          # frozen snapshot
    def get_tick(self) -> int: ...
    def get_phase(self) -> int: ...               # round phase for pause-points
    def is_paused(self) -> bool: ...
    def set_paused(self, pause: bool) -> None: ...
```

### Phase 3 — Wire into main.py

**Step 9: main.py uses Simulation**
Replace the GameMap shim with `Simulation`. Call `sim.step()` from the tick loop. Renderer reads `sim.get_state()`.

**Step 10: Input + order placement**
Click handlers in main.py translate mouse clicks into orders. Calls `sim.apply_action(unit_id, order)`. Backspace undoes via `sim.undo_last_order`. Phase pause: spacebar toggles `sim.set_paused()`.

**Step 11: Render units + projectiles + shot tracers**
Use the existing `renderer.compose_world(units_marines=..., units_zombies=...)` API. Add projectile rendering (bullets, grenades in flight, explosive markers).

**Step 12: HUD / phase indicator / order placement UI**
The right panel grows: turn timer, AP indicators per unit, current order being placed, phase indicator.

**Step 13: Delete `game.py`**
When main.py reaches feature parity. Big symbolic moment.

---

## Re-runnable verifications

- After **step 2**: smoke test still passes; main.py renders ship + a hardcoded marine.
- After **step 5**: physics tick happens through `PhysicsRunner` — atmosphere/smoke/fire visible.
- After **step 7**: PhysicsRecorder writes .npz each tick (no behavior change).
- After **step 8**: `from simulation import Simulation; sim.step()` works in isolation (a unit test should call `step()` 100 times with no exceptions).
- After **step 10**: a marine can move and fire.
- After **step 12**: full feature parity with `game.py` — gameplay loop intact.
- After **step 13**: `python main.py` is the only entry point.

---

## Anti-goals during the migration

These are explicitly NOT part of this patch — don't get sucked into them:
- Real-time vs turn-based final decision (we ship with the current pause model)
- New unit types (worms, robots, animals)
- New weapons (flamethrower, shotgun)
- The "scorch marks on walls from explosions" idea (see TODO)
- AI training scaffolding (`train.py`) — separate patch
- Network multiplayer
- Splitting Unit into Marine/Zombie subclasses

Each of these is worth doing — but each is its own patch. This one is a pure refactor + boundary cleanup. We need to land on the new entry point first.

---

## After-migration follow-ups (track in TODO.md)

- Scorch marks on tiles from grenades / fire (permanent visual scarring)
- AI training pipeline scaffolding
- Per-creature AI sample rate (humans slow, robots fast)
- Flamethrower + shotgun + teargas weapons
- Ammunition as inventory items
- "Find a flamethrower on the ship" first mission idea
- Smoke-at-vacuum-tiles bug (zero gmap.smoke at vacuum before upload) — 30 second fix
- Consider replacing per-frame physics state copy with a double-buffered swap (perf, not behavior)

---

## How to execute this patch

Given the conversation context budget remaining is tight, the recommended approach:

1. **Spawn a reviewer agent** on this plan first. Catch design problems before code touches the disk. (~5 min)
2. **Spawn an implementation agent** with this plan + full read access to game.py. The implementation agent does steps 1-7 (the lift). They have fresh context to read game.py thoroughly. (~30-60 min of agent work)
3. **Erik + this conversation reviews the lift result.** Smoke-test, commit incrementally per step.
4. **Then spawn a second implementation agent** for steps 8-12 (Simulation facade + wiring + UI). (~30-60 min)
5. **Step 13 (delete game.py)** happens last when Erik is confident.

The "this conversation" stays in orchestration and review mode. We don't run out of context reading game.py ourselves.
