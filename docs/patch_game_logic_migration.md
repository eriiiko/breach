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

## Migration order

### Pre-Step 0 — Reconcile two GameMap constructors

Legacy `game.py:GameMap` takes no args (calls `_build_ship()` internally),
the shim in `main.py:GameMap(level)` takes a LevelData and populates from
CSV. Step 1 cannot be a verbatim lift — it must merge these. Resolution:
the canonical `GameMap.__init__(level_data)` always takes a level. Drop
`_build_ship` entirely (already dead — CSV loading covers it). Caller
(`Simulation`) holds the level reference.

### Set up the Python package layout BEFORE step 1

Create `src/simulation/__init__.py`. Add `src` to `sys.path` (or use a
proper `setup.py` / `pyproject.toml`). Pick ONE qualifier and use it
everywhere — recommend `from simulation import Simulation` with
`sys.path.insert(0, str(ROOT / "src"))` in main.py. Document in the
module docstring.

### Phase 1 — Lift simulation pieces out of game.py

Each step is a refactor: copy code, adjust imports, keep behavior
identical, run the smoke test. **Note:** "independently runnable" means
the renderer still launches and shows the ship — gameplay features
accumulate across steps. Game is not fully playable until Phase 3.

**Step 1: `src/simulation/gamemap.py`**
Lift the `GameMap` class. Drop dead code (`_build_ship`). Levels load via `level_loader`. The shim in `main.py:GameMap` gets replaced with this.

**Step 2: `src/simulation/unit.py`**
Lift `Unit`. Inventory becomes a base field. Make zombie a state
(`is_zombie: bool`). Marine-only fields stay on Unit but conditional on
state. **Fix the `zombie_speed_override` monkey-patch** at this point —
replace with a proper `speed_ticks_per_tile: int` field. Bonus: easier
to balance later.

After Step 2: `main.py` can instantiate marines and zombies. They can be
rendered (already supported by `renderer.compose_world`). No orders yet.

**Step 3: `src/simulation/orders.py`**
Lift order classes (Order, MoveOrder with the three move modes, FireOrder, GrenadeOrder, ExplosiveOrder). Pure data — no execution logic here.

**Step 4: `src/simulation/combat.py`**
Lift `Projectile`, `Shot`, the shooting / line-of-sight functions. Functions take `gmap` + `units` as inputs, mutate them.

`apply_explosion` stays in the physics namespace (it is a physical event that has gameplay consequences — pressure waves, fire ignition, wall damage, unit blast). Splitting it would scatter event-effect logic and make calling code error-prone (easy to forget the damage step). The same pattern is already present in C++ (fire damages walls, atmosphere drains through breaches). Combat calls into `physics.apply_explosion(gmap, ...)` from grenades / explosives / weapon impacts — physics owns the "this is a physical event" entry point.

**Step 5: `src/simulation/physics_runner.py`**
Lift `PhysicsRunner` from main.py (already extracted, just move to its
proper home). **Important:** legacy `game.py` binds Fire parameters via
`_init_solvers` (FIRE_D, FIRE_O2_THRESHOLD, etc.). The current main.py
`PhysicsRunner` silently SKIPS this. Bring the fire binding over too —
otherwise fire behavior subtly diverges.

**Step 6: `src/simulation/ai_zombie.py`**
Lift zombie activation, pathfinding orchestration, target selection, conversion-to-zombie logic.

**Step 7: `src/simulation/recorder.py`**
Move `PhysicsRecorder` verbatim. Hook into Simulation's tick step.

### Phase 2 — The Simulation facade

**Step 8: `src/simulation/simulation.py`**
Create the `Simulation` class. It owns the GameMap, the unit list, the
order queue, the physics_runner, the recorder, and an `np.random.Generator`
for all nondeterminism. Public API designed for **both** human play AND
AI training rollouts:

```python
class Simulation:
    # -- construction / lifecycle --
    def __init__(self, level_data, seed: int | None = None): ...
    def reset(self, seed: int | None = None) -> None: ...
        # Re-init from level_data with new seed. For AI training rollouts.

    # -- units --
    def add_unit(self, unit: Unit, position: tuple) -> int: ...

    # -- actions --
    def apply_action(self, unit_id: int, order: Order) -> None: ...
    def undo_last_order(self, unit_id: int) -> None: ...
    def get_legal_actions(self, unit_id: int) -> list[Order]: ...
        # For AI: enumerate valid orders this unit can issue NOW.

    # -- tick loop --
    def step(self) -> None: ...                 # advance one tick
    def get_tick(self) -> int: ...
    def get_phase(self) -> int: ...

    # -- pause (human convenience; AI ignores) --
    def is_paused(self) -> bool: ...
    def set_paused(self, pause: bool) -> None: ...

    # -- state access --
    def get_state(self) -> SimState: ...        # snapshot for renderer + AI

    # -- AI training --
    def get_reward(self, unit_id: int) -> float: ...
        # Per-agent reward signal. Default is 0; subclasses override
        # for specific training environments.
    def is_terminal(self) -> bool: ...          # round ended / all units dead?

    # -- determinism plumbing --
    @property
    def rng(self) -> np.random.Generator: ...
        # ALL nondeterminism in combat (rifle cone), explosion smoke
        # noise, and Raycaster jitter MUST pull from this. Otherwise
        # AI replays / rollouts diverge.
```

**Three nondeterminism call sites** the implementation must plumb the
RNG through (caught by reviewer): `Physics._add_explosion_smoke` (smoke
noise), `Raycaster.cast_source` (jitter for fire flicker — pass seed),
`_fire_burst` (bullet cone offsets).

### Phase 3 — Wire into main.py

**Step 9: main.py uses Simulation**
Replace the GameMap shim with `Simulation`. Call `sim.step()` from the tick loop. Renderer reads `sim.get_state()`.

**Step 10: Input + order placement**
Click handlers in main.py translate mouse clicks into orders. Calls
`sim.apply_action(unit_id, order)`. Backspace undoes via
`sim.undo_last_order`. Phase pause: spacebar toggles `sim.set_paused()`.

**Real-time + pause walkthrough** (caught by reviewer — be explicit):
- Game starts paused, in planning mode for Phase 1.
- Player places orders for all marines for Phase 1 + Phase 2 (Tab switches).
- Spacebar starts execution. `sim.set_paused(False)`. Time flows.
- At any moment during execution, spacebar pauses. `sim.set_paused(True)`.
  Orders can be MODIFIED during pause (replace / add waypoints).
- At end of Phase 1 (tick 60), sim auto-pauses (`sim.set_paused(True)`),
  player tweaks Phase 2 orders, spacebar resumes.
- At end of round (tick 120), auto-pause, planning UI for next round.
- AI training rollouts ignore pause entirely — they call `sim.step()`
  in a tight loop and never set_paused.

**Sprite loading**: legacy game.py loads `art/sprites/zombies/*.png` on
init. Lift to a `renderer.sprites` module or have main.py load them
once and pass to renderer. Do NOT scatter image loading across
gameplay code.

**F5 key collision**: renderer currently binds F5 to "normal map
toggle." Legacy game.py binds F5 to "reload config." Decide: rebind
config reload to F12 or Ctrl+R; keep F5 for renderer toggle. Document.

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
- **Sprite rework** — keep the existing zombie civilian sprites as-is
- **Temporal A*** — the `temporal_astar` + `ReservationTable` in
  pathfinding.py are unused scaffolding. Leave them alone. Do NOT wire
  them up "while I'm here."
- **Removing the float position fields** (`fxf`, `fyf` on Unit) — they
  are still used by the renderer for interpolation. The architecture
  doc says they shouldn't be in game state, but that's a separate
  patch. Keep them for now.
- **Order subclassing** — keep Order as a single class with a
  type discriminator + payload. Subclasses are a different patch.
- **Fixing apply_explosion's mixed concerns** — already discussed,
  decided to keep cross-cutting. Don't refactor it here.

Each of these is worth doing — but each is its own patch. This one is a pure refactor + boundary cleanup. We need to land on the new entry point first.

---

## Testing strategy

- After **each step in Phase 1**: run `tests/test_renderer_smoke.py
  --auto` and `tests/test_level_loader.py`. Both must still pass.
- **Add** `tests/test_simulation.py` during step 8: instantiate
  `Simulation(level_data, seed=42)`, call `step()` 100 times,
  assert no exceptions. Then `reset(seed=42)`, step 100 times again,
  assert state matches the first run (determinism check).
- **Defer** end-to-end gameplay tests until after Phase 3 (full
  feature parity). Not worth writing tests against logic that's
  still moving.
- **Error handling convention**: gameplay code raises clear
  `ValueError` / `RuntimeError` with context. No silent failures.
  Renderer never raises during normal operation — it logs and
  continues.
- **Save / load**: out of scope for this patch. `Simulation.get_state()`
  returns enough info that a future patch can implement save/load on
  top via `pickle`. Note in docstring: "future save/load reads from
  this snapshot."

## Playtest findings (2026-05-17) — before deleting game.py

Erik playtested a round after Phase 2-3 landed. Game is broadly working,
but several behaviors regressed or differ from legacy game.py. Investigate
each by running both side by side before step 13.

### To investigate / restore

1. **Explosions visuals are weak.** No flame burst, no flash. Legacy
   `game.py` had a pressure-to-color mapping that made explosions look
   dramatic. Verify it was actually there (might be misremembered) and
   either port it or design something better.

2. **Explosions should emit a light source** at the center for a few
   ticks. Currently the world goes dark again immediately. New idea —
   could be implemented as a transient LightSource added to the
   `ExplosionEvent` consumer in the renderer (high intensity, short
   life), OR a `Simulation`-side temporary entity.

3. **Spawn sites are hardcoded.** Marines should spawn at level-defined
   spawn points (likely a new field in `level.toml` — `marine_spawns`,
   `zombie_spawns`). Currently main.py drops a few hardcoded marines for
   the demo.

4. **Sprite models** — Phase 2-3 left units as circles per anti-goal.
   Need to wire `art/sprites/*` back. Legacy game.py used round-robin
   assignment (game.py:1349-1354).

5. **Strong / weak zombie variants** — legacy had multiple zombie kinds
   (runner: low HP fast, brute: high HP slow). Check `unit.py` carries
   the `speed_ticks_per_tile` and `max_hp` fields (it should), then
   reintroduce variant spawning.

6. **Turn / phase flow is not working properly.** Investigate against
   game.py — possible candidates: phase transition trigger, pause
   release timing, order resolution at tick boundaries. Specific
   symptom not captured at playtest time; reproduce + diagnose.

### New design ideas (might do, might not)

7. **Level editor mode** — a tool/mode that loads a level and lets the
   designer place monsters, spawn points, and triggers interactively.
   Could be its own entry point (`editor.py`) reusing `renderer` +
   `level_loader`, writing to a `level_extras.toml` per level.

### Anti-goal still standing

Do not delete `game.py` until items 1-6 are reconciled against the
legacy reference. The playtest pass + this list is exactly why we held
back on step 13.

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

1. **Reviewer agent ran** (see `docs/review_game_logic_migration.md`).
   3 critical + 3 high + 4 medium findings — addressed inline in this
   plan above.

2. **Implementation agent #1** does Phase 1 (steps 0, set-up, 1-7 —
   the lift). Read access to:
   - This plan: `docs/patch_game_logic_migration.md`
   - Reviewer notes: `docs/review_game_logic_migration.md`
   - Inventory: `docs/game_py_inventory_and_migration_plan.md`
   - Architecture: `docs/architecture.md`
   - Files to lift FROM: `game.py`, `main.py` (GameMap shim)
   - Files to lift INTO: create `src/simulation/` package
   - Reference: `config.py`, `level_loader.py`, `pathfinding.py`,
     `cpp/src/fire_simulation.h` (for FireParams field names)
   - Renderer API not to break: `renderer/game_renderer.py`,
     `tests/test_renderer_smoke.py`
   Commit per step.

3. **Erik reviews** the lift result. Smoke test + ship + a hardcoded
   marine should render.

4. **Implementation agent #2** does Phase 2-3 (steps 8-12 — Simulation
   facade + wiring + UI). Needs everything agent #1 had, plus the new
   `src/simulation/` modules agent #1 produced.

5. **Erik does Step 13** (delete `game.py`) when confident gameplay
   parity is reached.

The orchestration conversation does not read `game.py` itself — that
work is delegated to the implementation agents, who get fresh context.
