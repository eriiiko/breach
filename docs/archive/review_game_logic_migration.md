# Review: `patch_game_logic_migration.md`

*Reviewer pass on the migration plan from `game.py` (pygame) to `src/simulation/`
+ existing pyray `main.py`. Cross-checked against `game.py` (2639 lines),
`main.py` (273 lines), `renderer/game_renderer.py` (396 lines), and
`docs/architecture.md`.*

---

## Verdict

The plan is **fundamentally sound** — the boundary it draws between simulation
and presentation matches `architecture.md:60-77`, the 13-step decomposition is
sensible, and the locked decisions (single Unit class, real-time-canonical with
pause, inventory on base) are pragmatic for v1. Two structural problems and one
process problem need addressing before an implementation agent is spawned: the
"each step independently runnable" claim is **not true** for steps 1-6 (the
game does not launch until step 9 at the earliest, and main.py's existing
GameMap shim has a different signature than the legacy `GameMap()`), the
**Simulation facade API is too thin for AI training** (no seed, no reset, no
legal-action enumeration, no terminal/reward — the architecture explicitly
commits to headless self-play, §17), and the **plan elides several concrete
artifacts** from game.py that an implementation agent will trip over (sprite
loading, F5 reload, the `_init_solvers` parameter-binding side channel, and
the import-path mechanics for `src/simulation/`). Net: it is a competent
plan that needs ~6 specific edits before handing off, not a redo.

---

## Critical concerns (must address before implementation)

### 1. "Each step independently runnable" is wrong for steps 1-6

The plan claims at `patch_game_logic_migration.md:67` and explicitly at
`:80` ("After Step 2: `main.py` can instantiate marines and zombies...") that
each step leaves a runnable game. This is **not** how `main.py` is currently
wired. The `Unit` class at `game.py:1104-1189` constructs with
`Unit(name, cx, cy, team=0)` and reads `CFG.display.coarse` to compute
`fx, fy` — but `main.py` never imports `Unit` at all and the demo at
`main.py:261` passes `units_marines=[]`. So Step 2's "instantiate marines"
isn't a no-op edit to `main.py` — it's a new feature: import path setup,
the spawn list, and pushing them through `renderer.compose_world`. That's
fine, but it needs to be called out as work, not framed as a free byproduct.

More seriously, the legacy `GameMap()` constructor at `game.py:295-321`
takes **no arguments** and reads `CFG.display.fine_w/fine_h` (set globally
at startup), while the shim at `main.py:36-73` takes a `level` parameter
and sizes from `level.height/width`. Step 1 says "lift the `GameMap` class
... the shim in `main.py:GameMap` gets replaced with this." Lifting the
legacy class verbatim **breaks** `main.py` because the signatures don't
match. The actual work in step 1 is a *merge* (take legacy fields, take
the shim's `(level)` constructor pattern). The plan's inventory at
`game_py_inventory_and_migration_plan.md:25-29` got this right; the patch
plan glossed over it.

**Fix:** Restate step 1 as "merge legacy fields into the shim's
constructor signature" and step 2 as "Unit class + main.py spawn-list edit
+ render wiring (new work, not just a move)". Add a sentence per
intermediate step naming exactly what still works and what doesn't. After
step 6, for example, there's no turn loop, no input, and physics doesn't
tick — the game is "running" but with nothing happening.

### 2. Import path for `src/simulation/` is undefined

`src/simulation/` is currently an empty directory (no `__init__.py`,
verified). The plan writes `from simulation import Simulation`
(`patch_game_logic_migration.md:55, 140`) but `main.py` does
`sys.path.insert(0, str(ROOT))` (`main.py:20`) which would make the
import `from src.simulation import ...`, not `from simulation`. Either
`src/` needs a `sys.path.insert(0, ROOT/"src")` in main.py, or the package
needs to be at the project root, or `src/` needs to become a top-level
package itself.

This is a 1-line fix but the implementation agent will hit it
immediately and a 2-minute discussion will save 10 minutes of "why
doesn't this import." Be explicit: which directory ends up on sys.path,
which module qualifier does the import use.

### 3. Simulation facade API missing AI-training essentials

`architecture.md:73-76` calls out "headless self-play for neural network
training at scale" as a **first-class** goal of the two-layer split, and
§17 commits to AlphaStar-style stacked feature planes with a Gymnasium
loop. The facade at `patch_game_logic_migration.md:102-114` exposes
`apply_action / step / get_state / set_paused / get_phase / get_tick`
but is missing:

- **`reset(seed) -> SimState`** — every Gymnasium env requires this. With
  the plan as-written there's no way to start a new rollout without
  reconstructing the `Simulation`, which means re-allocating the 15+ numpy
  grids (`game.py:300-314`) — wasteful and breaks vec-env determinism.
- **`get_legal_actions(unit_id)`** — the existing `_place_order` at
  `game.py:1465-1522` silently `return`s on failures (no AP, no
  inventory, blocked tile). The inventory called this out as a problem
  (decision F). The facade still doesn't expose legality up front, so
  the agent has to do `apply_action`-with-no-feedback or duplicate
  the validation rules.
- **`get_reward() / is_terminal()`** — for training, even a placeholder
  is fine, but the API must commit to where these will hook in.

`set_seed` is implied by `__init__(..., seed)` but `numpy.random` and
`random` (used at `game.py:1822, 2010, 1029`) are process-global. Without
an explicit `rng = np.random.default_rng(seed)` plumbed through
`apply_explosion / _add_explosion_smoke / Raycaster.cast_source`,
training rollouts will not be deterministic.

**Snapshot / restore** can wait — they're convenient but not blocking
for a first training pass.

**Fix:** Add `reset`, `get_legal_actions`, `get_reward`, `is_terminal`
to the v1 facade. Add an `rng` parameter that flows down to the three
non-deterministic sites. Save snapshot/restore for v2 — flag it
explicitly so it isn't forgotten.

---

## High concerns (should address)

### 4. "Single Unit class with `is_zombie` flag" is fine for v1, but the inventory pain point will hit at the marine→zombie conversion site

The plan at `:14` chose "one Unit class; zombie is a state." The legacy
already does this and the marine→zombie conversion at `game.py:2019-2028`
just rewrites `team / hp / max_hp / zombie_activated` in-place and prefixes
the name with `Z-`. Architecture's intent at `architecture.md:769`
("zombie as state, not type") explicitly supports this. The pain point
isn't AI dispatch (a single `is_zombie` branch handles it), it's
**inventory ownership**: when a zombie carries a grenade through fire and
it detonates, *who* enforces "zombies can't issue use-grenade orders but
the grenade can still cook off"? With one class, this is a few
`if not is_zombie:` guards in `apply_action`. With subclasses it would be
clean polymorphism. Both work. The locked decision is defensible; just
make sure `apply_action` is the single chokepoint that enforces the
"zombies can't pick orders" rule, not scattered throughout the order types.

The harder issue is `zombie_speed_override` at `game.py:1294, 1301, 1944`
— this is monkey-patched after construction. The plan doesn't mention it.
Per `architecture.md:781`, this should become a proper `speed` field on
`Unit`. Add to step 2: "promote `zombie_speed_override` to a `speed`
field, defaulting to `CFG.zombie.ticks_per_tile`."

### 5. Real-time + pause: walk-through is missing for AI training

The plan at `:24` locks in "real-time with scheduled pauses" and at `:25`
puts the pause toggle in `main.py`. Spelling out the actual semantics:

- Spacebar pressed at tick 47 of phase 1: `sim.set_paused(True)`. The
  next `sim.step()` call from `main.py`'s loop becomes a no-op. The
  renderer keeps rendering the frozen state. **Can orders be modified
  during pause?** The locked decisions don't say. The legacy is strictly
  PLANNING → EXECUTING with no mid-execution editing (`game.py:1371-1376`).
  If we keep that, pause is purely a display pause — no `apply_action`
  during EXECUTING. If we allow editing during pause, we need to define
  what happens to in-flight projectiles, ongoing fire bursts, and the
  precomputed `move_path`.
- **Phase-end checkpoint forcing a pause:** the existing model
  (`game.py:1660-1694`) just transitions phases inside `_update_execution`
  and ends back at PLANNING at tick 120. There's no concept of "force
  the human to confirm." If the plan wants this, the facade needs
  `is_awaiting_input() -> bool` or an event queue with phase-boundary
  events. The architecture's "every in-game second is fully simulated"
  (`architecture.md:48`) is silent on whether the simulation can request
  a pause.
- **AI training:** `train.py` would simply never call `set_paused(True)`.
  This works as long as pause is purely "skip `sim.step()` calls" — but
  if pause can also gate phase transitions (e.g. "force planning input
  here"), then AI training needs an explicit "auto-resolve" mode.

**Fix:** Add one paragraph to the plan: "pause = main.py skips step()
calls. No mid-execution edits in v1. End-of-round auto-returns to
PLANNING and AI sees that as a state field, not a halt."

### 6. Sprite loading + F5 reload + the `_init_solvers` side channel are unaddressed

Three things in the legacy that the plan never names:

**Sprite loading** (`game.py:1325-1354`): the pygame `Surface` objects
die with pygame, but the **deterministic round-robin name → sprite_index
assignment** at `:1349-1354` is gameplay-adjacent (the inventory called
this out as decision G but the patch plan didn't pick a side). My
recommendation matches the inventory: store `sprite_id: int` on Unit
during creation, let the renderer resolve to a `pyray.Texture`. The
implementation agent will *invent* a solution if not told — and they may
invent badly.

The `art/sprites/marine/` and `art/sprites/zombies/` folders exist
(verified) but `renderer/game_renderer.py` doesn't currently load
sprites — `_draw_units_world` at `:206-216` draws colored
rectangles. So the migration introduces sprite loading on the renderer
side too, not just relocation. **Anti-goals at `:155` doesn't cover this**,
so an implementation agent might decide "while we're here, let me build
sprite loading." That's net-positive but blows the scope estimate.

**F5 hot-reload** (`game.py:1367-1368`): `CFG.reload()` is one line.
Currently main.py's F5 is bound to *toggle normal map*
(`renderer/game_renderer.py:314-315`). After migration, F5 needs to mean
config reload (per `architecture.md:14` and the legacy semantics) AND
the renderer toggles need to move to another key. This is a UX collision
the plan doesn't flag.

**The `_init_solvers` side channel** (`game.py:754-800`): the legacy
binds C++ solver parameters once at first physics tick, including the
fire params at `:778-784`. `main.py`'s `PhysicsRunner` at `:98` calls
`bp.FireSimulation()` but **never sets any fire params**. Either the
C++ defaults are good (then the legacy code is dead) or `main.py` already
has a silent bug. Either way, the migration needs to either pull the
parameter-binding into `PhysicsRunner.__init__` or document explicitly
that we accept C++ defaults. Step 5's "just move PhysicsRunner to its
proper home" understates this.

---

## Medium concerns (worth noting)

### 7. `apply_explosion` placement is defensible, but flag the future extensibility tension

The decision at `patch_game_logic_migration.md:87` to keep `apply_explosion`
in `simulation/combat.py` calling into `physics.apply_explosion(gmap, ...)`
is sound. The reasoning matches `game.py:704-741` (this isn't a physics
*solver* step; it's a gameplay event that *uses* physics fields).

For future weapon types (energy explosions, lightning at
`architecture.md:714-727`, flamethrowers per `:165`), the pattern
generalizes cleanly: each weapon-effect function lives in `combat.py`
and writes into appropriate `gmap` fields. The temptation is to build
a generic `Effect` system. **Defer** — three more weapons is when the
abstraction pays off, not now.

One thing: `_add_explosion_smoke` at `game.py:1998-2012` uses
`random.uniform(0.4, 1.0)` — non-deterministic per seed concern from
critical issue #3.

### 8. PhysicsRecorder schema needs a rename-watch

The plan at `:96` says "Move PhysicsRecorder verbatim." Checked the
field list against current `GameMap`:

The recorder reads `wave_p, wave_v, atmosphere, smoke, fire, obstacles`
by name (`game.py:187`) via `getattr(gmap, name)`. All six are present
on both the legacy GameMap and main.py's shim. **Good.**

Unit snapshot at `game.py:232-236` reads `name, team, fx, fy, hp, alive`
— all base fields. If step 2 promotes `zombie_speed_override` to a `speed`
field, recorder still works (it doesn't snapshot speed). If
`fxf/fyf` are dropped (per `architecture.md:1107`), no recorder change
needed. The `.npz` files in the repo
(`debug_blowup_20260318_205408.npz`, etc.) will remain loadable.

**One risk:** if the migration also restructures `obstacles` semantics
(e.g. splits walls vs units into separate arrays), the recorder's
`obstacles` snapshot changes meaning silently. Flag a "schema freeze"
comment in `recorder.py` after porting.

### 9. Anti-goals don't fence off the seductive side-quests

The anti-goals list at `:147-157` is good but missing:

- **Sprite loading rework** (see #6 — implementation agent will be
  tempted)
- **Hot-reload extension** ("while I'm wiring F5, let me also reload
  the level CSV...")
- **Move-path replacement with temporal A*** (the architecture explicitly
  flags this at `architecture.md:1093, 308` and the unused
  `temporal_astar` import at `game.py:40` is a temptation)
- **Float-position removal** (`architecture.md:1107` calls this out as a
  cleanup; the plan keeps `fxf/fyf` implicitly by lifting Unit verbatim;
  the implementation agent may "tidy" them out and break the renderer's
  interpolation hooks)
- **Splitting `Order` into `MoveOrder/FireOrder/...`** (the inventory
  recommends it at `game_py_inventory_and_migration_plan.md:46`; the
  patch plan at `:82` says "lift order classes" implying one class,
  but the inventory recommendation is sitting right there waiting to
  be picked up)

Add a clarifying line per item: "explicitly not part of this patch — own
tracked TODO."

### 10. Hand-off context for the implementation agent

The plan at `:177-186` says "spawn an implementation agent with this plan
+ full read access to game.py." Reality check: that agent will *also*
need:

- `docs/architecture.md` — for the simulation/presentation principle and
  the AI training context
- `config.py` (and `config.toml`) — to understand `CFG.display.fine_w`
  vs the level dims, the materials block, the `CFG.reload()` contract
- `level_loader.py` — for `materials_from_tilemap` signature
- `pathfinding.py` — to know what `astar()` returns (list of tuples)
  and that `temporal_astar` is the intentionally-unused branch
- A glance at `cpp/src/fire_simulation.h` and `atmosphere_solver.h` to
  see the public C++ API — currently the agent has to infer it from
  game.py's `_init_solvers` and main.py's `PhysicsRunner.__init__`.
  Confusing because the two callsites disagree on which params are set.
- The existing `renderer/game_renderer.py` API (already in the read list,
  good)

This is ~6 short files. Total context is comfortable. No need for
explainers — the code reads cleanly. But list the files explicitly in
the hand-off so the agent doesn't grep wide.

---

## Nice-to-have suggestions

### 11. Testing strategy is entirely missing

The plan has zero mention of tests. The inventory says nothing either.
`tests/test_renderer_smoke.py` exists (verified). At minimum:

- A `test_simulation_smoke.py` after step 8 that does
  `sim = Simulation(level); for _ in range(100): sim.step()` — the
  plan mentions this verbally at `:140` but doesn't make it a
  deliverable
- A determinism test: same seed, two `Simulation` instances, after 100
  ticks the `get_state()` arrays match elementwise
- An `apply_action` validation test (legal placements succeed, illegal
  fail with a clear return code)

These are 50-100 lines total. Add as step 8.5 and 9.5.

### 12. Error handling philosophy

`game.py` silently `return`s on order placement failures
(`game.py:1465-1522`). The architecture's `apply_action` contract should
return a result (`OK / NO_AP / NO_INVENTORY / BLOCKED`). The plan at
critical concern #3 picks this up via `get_legal_actions`; equally fine
to make `apply_action` return a `Result` enum. Either way, **commit to
a convention** before implementation so the renderer can show toasts
on failure (UI step 12).

### 13. Save/load and level transitions

Not in the plan. `architecture.md:237` calls them out as "essentially
free given the state model." For v1, fine to skip. For the AI training
work, **`Simulation.serialize() / Simulation.deserialize(blob)`** is
nearly free if `get_state()` returns a flat dict of arrays. Worth
mentioning as a follow-up so it doesn't get forgotten when train.py
needs replay buffers.

---

## What was done well

- The locked-decisions table at `patch_game_logic_migration.md:13-25`
  is a model of patch-plan hygiene — each decision named, choice
  written, rationale linked. Future-you will thank present-you.
- The "out of scope" block at `:26-30` correctly fences off the
  worms/robots/animals question.
- The reasoning for `apply_explosion`'s placement at `:87` is the
  best-argued single passage in the plan — it cites the C++ pattern
  (fire damages walls, atmosphere drains through breaches) as
  precedent. Good.
- The handoff strategy at `:177-186` is realistic about context budget
  and correctly separates "reviewer" (this pass), "lift agent",
  "wire agent", and "Erik in the driver's seat for the symbolic
  deletion."
- Decision E (pre-computed *and* per-tick path) at `:18` correctly
  picks both: precomputed for visualization (the order-line overlay
  in `renderer/game_renderer.py:218-224` already consumes waypoints),
  executed per-tick (since `_compute_player_paths` at
  `game.py:1566-1639` builds per-tick lists). The two are not in
  tension — they're the same data viewed at different granularities.
- The `architecture.md:60-77` boundary is honored, not paid lip
  service: the proposed `Simulation` API actually has the
  `apply_action / step / get_state` triad the doc commits to.
- The 13-step decomposition is correctly ordered (data classes →
  GameMap → Unit → orders → combat → physics_runner → AI → recorder →
  facade → wire → input → render → UI → delete). This is the right
  topological sort.

---

*Reviewer notes: cross-checked all line citations in the inventory
against `game.py` at the current revision; key spot-checks at
`game.py:292, 1104, 1195, 1239, 1250, 1466, 1527, 1660, 1696, 1867,
2014` all matched. Inventory accuracy: high. Patch plan accuracy on
"each step independently runnable": low (see critical #1).*
