# game.py Inventory & Migration Plan

*Audit of the legacy pygame entry point (`game.py`, 2639 lines) and a phased plan for moving the gameplay systems into clean modules consumable by the new pyray-based `main.py`.*

*Created: 2026-05-16. Source files audited at this revision are `game.py`, `main.py`, `pathfinding.py`, `level_loader.py`, `config.py`.*

---

## 0. Context

- `game.py` is the **legacy entry**: it bundles every game system into one file using pygame for both input and rendering. Imports: `pygame`, `numpy`, `config.CFG`, optional `pathfinding`, optional `breach_physics` (C++).
- `main.py` (273 lines) is the **new entry**: pyray window, world-space render target, full `renderer/` package, demo scene only. It has a `GameMap` shim (`main.py:36-73`) and a `PhysicsRunner` adapter (`main.py:80-123`), but **no units, orders, turns, AI, combat, or UI**.
- `docs/architecture.md:60-77` already commits to a strict two-layer split: a headless **Simulation Layer** with `get_state() / apply_action() / step()`, and a **Presentation Layer** that never mutates state. The migration should land on that boundary — not just move code, but separate it.
- An empty `src/simulation/` directory exists — the obvious destination for the simulation package.

---

## 1. State & Entities

### 1.1 GameMap (gameplay-relevant fields)
- **Location**: class `GameMap` at `game.py:292-542`. Constructor `__init__` (295-321), level loader `_load_level` (323-338), hard-coded fallback `_build_ship` (340-406), cache rebuild `_update_caches` (408-439), `stamp_units` (441-466), `is_passable`/`is_passable_block` (468-483), `has_los` (485-506), `_neighbor_mean` (508-518), `destroy_wall` (520-542).
- **External deps**: `CFG.display.*`, `CFG.materials.*`, `level_loader.load`, `level_loader.materials_from_tilemap`.
- **State owned**: 14 grid arrays at `(fh, fw)` resolution — `material` (int8), `wall_hp`, `is_wall`, `is_vacuum`, `flammable`, `atmosphere`, `wave_p`, `wave_v`, `wave_source`, `wind_x`, `wind_y`, `smoke`, `fire`, `obstacles`, `light_map`. Plus `self.level` (when CSV-loaded).
- **Complexity**: Medium. The grid allocation is mechanical; `stamp_units` and `destroy_wall` carry the only nontrivial logic (atmosphere refill from neighbor mean on freed tiles, hull-edge vacuum promotion).
- **Migration notes**:
  - `main.py:36-73` already has a slimmer `GameMap`. The two need to merge. The new home should be `src/simulation/world.py` (or `gamemap.py`).
  - `_build_ship` (340-406) is dead code now that CSV levels work — **drop it** rather than port it.
  - `stamp_units` couples world to `Unit` objects; the simulation module should depend on a Unit type defined in the same package (no pygame import).
  - `has_los` (485-506) and `is_passable_block` (475-483) are pure utilities used by AI/combat — they may belong on a `World` helper module rather than the GameMap class.

### 1.2 Unit
- **Location**: class `Unit` at `game.py:1104-1189`.
- **External deps**: `CFG.marine`, `CFG.zombie`, `CFG.clock.ap_per_phase`, `CFG.display.coarse`.
- **State owned**: position (`fx`, `fy`, `fxf`, `fyf`), `team`, `hp`/`max_hp`, `alive`, `facing`, `orders` list, `current_order_type`, inventory (`has_grenade`, `has_explosive`), AP per phase (`ap[2]`), combat (`last_fire_tick`, `fire_target`), zombie AI (`zombie_activated`, `zombie_path`, `zombie_path_idx`, `zombie_move_accumulator`, `last_melee_tick`, `killed_by_zombie`), movement (`move_path`, `path_tick_offset`). Plus optional `zombie_speed_override` monkey-patched at construction (`game.py:1294`, `1301`).
- **Complexity**: Medium (the class is small but the state surface is wide and conflates marine/zombie data).
- **Migration notes**:
  - This is the single most-coupled class in the codebase. Worth splitting into a base `Unit` and `Marine` / `Zombie` subclasses (or a component bag), so the marine never carries `zombie_path` and vice versa. **Open design question — discuss before migrating.**
  - `move_path` is a per-tick list of `(fxf, fyf)` floats produced by `_compute_player_paths`. Replacing this with on-the-fly path execution is a possible v2 simplification.
  - The two computed properties `cx`/`cy` (1141-1147) and helpers `center_fx`/`center_fy` (1149-1153) should be kept exactly — they're used throughout combat and AI.

### 1.3 Order
- **Location**: class `Order` at `game.py:1089-1098`. Order type constants at `game.py:85-99`. Per-order colors at `game.py:122-129`. Detonation slot constants at `game.py:132-139`.
- **External deps**: none (data class).
- **State owned**: `order_type`, `target_fx/fy`, `phase`, `grenade_fuse`, `det_slot`, `ap_cost`.
- **Complexity**: Simple.
- **Migration notes**:
  - Replace with a dataclass or per-order-type dataclasses (`MoveOrder`, `FireOrder`, `GrenadeOrder`, `ExplosiveOrder`). The current "one class, optional fields" pattern is fragile.
  - Material IDs (`MAT_AIR/HULL/WOOD/DOOR` at `game.py:142-145`) live in the same constants block but conceptually belong with the world module, not orders.

### 1.4 Projectile (in-flight grenade)
- **Location**: class `Projectile` at `game.py:1195-1233`.
- **External deps**: `CFG.weapons.grenade.travel_speed`, `CFG.clock.ticks_per_second`.
- **State owned**: type, current/start/target floats, `fuse_seconds`, `thrown_tick`, `detonated`, `travel_speed`.
- **Complexity**: Simple.
- **Migration notes**: Single use (only grenades). Position interpolation is linear from `start -> target` over computed `travel_time`. Keep as-is, but place beside `Order` in the simulation package.

### 1.5 Shot (visual tracer)
- **Location**: class `Shot` at `game.py:1239-1244`.
- **External deps**: `CFG.combat.shot_tracer_duration`.
- **State owned**: endpoints, spawn `time`, `duration`.
- **Complexity**: Simple.
- **Migration notes**: This is presentation/effect data, **not gameplay**. Either expose it as a per-tick event the renderer consumes ("shot fired from A to B at tick T") or keep an effects buffer. Don't store it on the simulation.

---

## 2. Turn / Phase System

### 2.1 State machine & phase transitions
- **Location**: State constants `STATE_PLANNING / STATE_EXECUTING` at `game.py:81-82`. Initial state in `Game.__init__` at `game.py:1309-1323`. Tick loop in `_update_execution` at `game.py:1660-1694`. Per-tick step `_process_tick` at `game.py:1696-1747`. Round teardown `_end_execution` at `game.py:2014-2043`.
- **External deps**: `CFG.clock.ticks_per_phase`, `ticks_per_round`, `ticks_per_second`, `phases_per_round`.
- **State owned**: `state`, `turn_number`, `planning_phase`, `exec_tick`, `exec_phase`, `exec_speed`, `exec_accumulator`, `real_time`, `projectiles` list, `shots` list.
- **Complexity**: Medium. The accumulator → integer ticks pattern (1666-1670) is standard. Phase-boundary detection (1678-1684) is where between-phase explosives detonate.
- **Migration notes**:
  - The tick loop is the heart of the simulation and the natural site of the `World.step()` interface called out in `architecture.md:69`.
  - `exec_speed` (1317) is presentation-layer playback control; it should NOT live in the simulation. The renderer/main loop decides how many simulation ticks to consume per frame.
  - `_process_tick` (1696-1747) prescribes the order: projectiles → player movement → shooting → zombie AI → re-stamp units → physics → recorder. **This ordering is load-bearing** — units must be stamped before physics so explosions push the right obstacles. Document it explicitly in the new module.

### 2.2 Round transition / zombie conversion
- **Location**: `_end_execution` at `game.py:2014-2043`.
- **External deps**: `CFG.zombie.hp`.
- **State owned**: writes through unit list (resets paths, snaps floats to ints, clears orders, refills AP).
- **Complexity**: Simple.
- **Migration notes**: Conversion (`killed_by_zombie -> team=1, hp=CFG.zombie.hp`) is the only non-trivial design choice baked in here. Easy to port.

---

## 3. Input / Orders

### 3.1 Planning input dispatch
- **Location**: `_handle_planning_event` at `game.py:1389-1443`. Mouse routing `_handle_map_left_click` / `_handle_map_right_click` at `game.py:1445-1463`.
- **External deps**: pygame events, `CFG.weapons.grenade.fuse_*`.
- **State owned**: `selected_unit`, `current_mode`, `grenade_fuse`, `det_slot`, `planning_phase`.
- **Complexity**: Medium.
- **Migration notes**:
  - Pygame events must become **pyray polling**. The dispatcher logic (key → mode change, click → place order) is pygame-agnostic and easy to lift if input is abstracted (e.g. an `InputState` struct passed in each frame).
  - The mode selector + the order placement are coupled by `self.current_mode`. Keep the same UX initially, but consider per-mode handlers (a `ToolController` per mode).

### 3.2 Order placement
- **Location**: `_place_order` at `game.py:1465-1522`.
- **External deps**: `GameMap.is_passable_block`, `Unit.spend_ap/has_grenade/has_explosive`, `CFG.weapons.*`.
- **State owned**: appends to `Unit.orders`.
- **Complexity**: Medium. Each order type has its own AP/inventory/passability rule.
- **Migration notes**: Clean candidate for the `apply_action()` simulation interface. The simulation should validate (passable? has AP? has grenade?) and return success/failure rather than silently dropping like the current code does (`return` on failure with no signal).

### 3.3 Undo
- **Location**: Backspace handler at `game.py:1409-1418`.
- **External deps**: Unit state.
- **State owned**: pops from `Unit.orders`, refunds AP, refunds inventory.
- **Complexity**: Simple.
- **Migration notes**: Refund logic is duplicated with the spend logic in `_place_order`. Better to have `Order.cancel(unit)` or maintain a transaction log so refunds are symmetric by construction.

### 3.4 Mouse wheel: grenade fuse + detonation slot
- **Location**: `game.py:1435-1443`. Clamps to `CFG.weapons.grenade.fuse_min_seconds / fuse_max_seconds`; det slot cycles `(0, 1, 2)`.
- **Complexity**: Simple.

---

## 4. Combat

### 4.1 Shooting / fire orders
- **Location**: `_process_shooting` at `game.py:1749-1791` (dispatches by phase, picks fire-order target, range + LOS check, burst gating). `_auto_fire` at `game.py:1793-1818` (Move & Attack auto-target nearest visible enemy). `_fire_burst` at `game.py:1820-1865` (per-bullet ray-march, wall stop, unit hit, zombie multiplier, tracer record).
- **External deps**: `Unit.get_fire_order_in_phase`, `GameMap.has_los`/`is_wall`, `CFG.weapons.rifle.*`, `CFG.zombie.bullet_damage_multiplier`, `CFG.combat.*`, `random`.
- **State owned**: `Unit.last_fire_tick`, `Game.shots` (tracer effects).
- **Complexity**: Medium. The per-bullet integer ray-march at 1834-1854 is the load-bearing combat algorithm.
- **Migration notes**:
  - **The burst per-bullet logic is the core combat simulation** — port verbatim into a `combat.py` module first, with no behavior changes.
  - `Shot` tracer creation should produce an event the renderer consumes, not be added to a list the simulation owns.
  - Local `import random` (1822) is inside the hot loop — remove on port.

### 4.2 Line of sight
- **Location**: `GameMap.has_los` at `game.py:485-506` (Bresenham, stops on `is_wall`).
- **Complexity**: Simple. Note: only walls block — smoke doesn't (yet). Discussion candidate.

### 4.3 Explosion / blast
- **Location**: `Physics.apply_explosion` at `game.py:704-741` (deposits to `wave_source`, atmosphere, fire ignition, smoke clear). `_apply_blast_damage` at `game.py:1981-1996` (radial unit damage with falloff and `blast_damage_threshold`). `_add_explosion_smoke` at `game.py:1998-2012` (random-noise smoke deposition). `_process_door_explosives` at `game.py:1641-1658` (triggered at three detonation slots).
- **External deps**: `GameMap.material`, `destroy_wall`, `is_wall`, `flammable`, `wave_source`, `atmosphere`, `smoke`, `fire`; `CFG.weapons.{grenade,door_explosive}.*`.
- **Complexity**: Medium. Three different smoke/pressure mechanisms operate during a single explosion (wave source 3x3 kernel, direct atmosphere add, noisy smoke deposition).
- **Migration notes**:
  - `Physics.apply_explosion` is currently in the `Physics` class but is fundamentally a **gameplay action**, not a physics solver step. It belongs in the simulation/combat module, not in `PhysicsRunner`.
  - The comment at 743-745 hints at a previous refactoring round — pay attention before changing.

### 4.4 Projectile in-flight update
- **Location**: per-tick block in `_process_tick` at `game.py:1701-1718`.
- **External deps**: `Projectile.update_position`, `Physics.apply_explosion`, `_apply_blast_damage`, `_add_explosion_smoke`.
- **Complexity**: Simple.

---

## 5. AI

### 5.1 Zombie activation + chain propagation + movement + melee
- **Location**: `_update_zombies_tick` at `game.py:1867-1979`. Three sequential passes:
  - Trigger detection (player within `CFG.zombie.trigger_radius` + LOS), `game.py:1879-1891`.
  - Chain activation BFS (`propagation_radius`), `game.py:1894-1908`.
  - Per-zombie nearest-player selection + melee or A* move, `game.py:1910-1979`.
- **External deps**: `GameMap.has_los`, `is_passable_block`, `pathfinding.astar`, `CFG.zombie.*` (HP, ranges, cooldowns, melee_damage, ticks_per_tile), `Unit.zombie_*` fields.
- **State owned**: writes through Unit (`zombie_activated`, `zombie_path*`, `last_melee_tick`, `hp`, `alive`, `killed_by_zombie`).
- **Complexity**: Complex. Two nested O(N^2) loops over zombies for chain activation, A* repath every 5 ticks, monkey-patched `zombie_speed_override` for fast/slow variants.
- **Migration notes**:
  - The chain-activation `while changed:` loop (1894-1908) is correct but O(N^3); fine for current N but flag for later.
  - `is_blocked` closure (1955-1956) is created per-zombie per-tick — hoist.
  - A* fallback when `HAS_PATHFINDING` is false (1962-1964) silently freezes zombies. The new simulation should hard-require pathfinding.

### 5.2 Player pathfinding (planning-time)
- **Location**: `_compute_player_paths` at `game.py:1566-1639`. Called in `_start_execution` (1558).
- **External deps**: `pathfinding.astar`, `Unit.orders`, `GameMap.is_passable_block`.
- **State owned**: Writes `Unit.move_path` (list of `(fxf, fyf)` per tick) and `Unit.path_tick_offset`.
- **Complexity**: Complex. Precomputes the full move trajectory per tick, interpolates `speed` ticks between tiles, fills remaining phase ticks with the end pose.
- **Migration notes**:
  - **Design question**: precompute-once vs. step-by-step. Precompute keeps execution dumb but means waypoints can't react to runtime events (zombies moving into the path). The architecture doc's emphasis on "systems, not scripts" leans toward per-tick stepping.
  - `temporal_astar` and `ReservationTable` (from `pathfinding.py:184-446`) are imported but **never used** in `game.py`. They're scaffolding for a planned per-tick scheme.

---

## 6. Rendering (REPLACE — do not migrate)

The pygame draw methods are all to be discarded. The `renderer/` package + `main.py` already replaces them. Inventory included only so nothing is missed.

| Method | Location | Replacement (new) |
|---|---|---|
| `_draw` (compositor) | `game.py:2055-2068` | `GameRenderer.compose_world` / `end_frame` in `renderer/game_renderer.py` |
| `_draw_map` | `game.py:2070-2095` | World render target with material texture upload |
| `_draw_atmosphere` | `game.py:2097-2149` | pressure_stops colormap → shader uniform; the colormap definition is gameplay-adjacent and should be ported (`config.toml`) |
| `_draw_smoke` | `game.py:2151-2170` | smoke field upload in renderer |
| `_draw_fire` | `game.py:2172-2192` | fire field upload in renderer |
| `_draw_light` | `game.py:2194-2216` | `renderer/lighting.py` (already exists) |
| `_draw_units` | `game.py:2218-2276` | new sprite layer (TBD — sprite loading at 1325-1354 is also pygame and dies with it) |
| `_draw_orders` | `game.py:2278-2351` | new overlay layer in `renderer/overlays.py` |
| `_draw_projectiles` | `game.py:2353-2362` | new effect layer |
| `_draw_shots` | `game.py:2364-2375` | new effect layer (consumes Shot events) |
| `_draw_cursor_info` | `game.py:2377-2423` | new tool ghost overlay |

**Sprite loading** (`game.py:1325-1354`): zombie/marine sprite assignment is gameplay-adjacent (the round-robin assignment at 1349-1354 is deterministic per-name). Worth preserving the assignment logic in the simulation/Unit, even though the actual surface objects move to the renderer.

---

## 7. UI

### 7.1 Side panel
- **Location**: `_draw_ui_panel` at `game.py:2425-2631`.
- **State read**: `state`, `turn_number`, `exec_phase`, `exec_tick`, `exec_speed`, `selected_unit` (name, hp, cx/cy, inventory, ap, orders), `frame_times`, `physics_ms`, `gmap.wave_p`, `gmap.atmosphere`.
- **Complexity**: Medium. Long but linear.
- **Migration notes**: A panel renderer is already stubbed in `renderer/game_renderer.py` (`renderer.draw_panel(None)` at `main.py:266`). Reproduce the same information; the data sources are simulation-owned and should be exposed via getters, not direct attribute reads.

### 7.2 Timeline bar + order blocks
- **Location**: `game.py:2524-2562` (inside `_draw_ui_panel`).
- **Complexity**: Medium (per-phase order stacking).

### 7.3 Mode selector
- **Location**: `game.py:2478-2491`.
- **Complexity**: Simple.

### 7.4 Performance / debug HUD
- **Location**: `game.py:2589-2612` (FPS, frame time, physics time, wave/atm extents).
- **Complexity**: Simple.

### 7.5 Controls help text
- **Location**: `game.py:2614-2631`.
- **Complexity**: Simple. Mostly static text — update once for the pyray rebind set.

---

## 8. Debug / Instrumentation

### 8.1 PhysicsRecorder (ring buffer)
- **Location**: class `PhysicsRecorder` at `game.py:179-286`. Wired in `Game.__init__` (1272), called per tick from `_process_tick` (1747), manual dump on F8 (1369-1370), auto-dump on blowup threshold inside `record()` (242-247).
- **External deps**: `numpy.savez_compressed`, GameMap field names, Unit fields.
- **State owned**: ring buffers for 6 grids by default + tick metadata + unit snapshots; `index`, `count`, `dumped`.
- **Complexity**: Medium.
- **Migration notes**:
  - **Keep this verbatim.** The `.npz` dumps already in the repo (`debug_blowup_20260318_*.npz`) are part of the active physics debugging workflow per `MEMORY.md`. Don't change field names or schema unless coordinated.
  - Move to `src/simulation/recorder.py` (or `debug/recorder.py`). Should hook a simulation observer, not be called inline.

### 8.2 Hot-reload (F5)
- **Location**: `game.py:1367-1368`. Calls `CFG.reload()` and that's it.
- **Migration notes**: `CFG` (in `config.py`) is process-global, so any module reading `CFG.foo` picks up the reload. Re-bind in renderer too. Keep the F5 binding.

### 8.3 Blowup snapshot dump (F8)
- **Location**: `game.py:1369-1370`. See 8.1.

### 8.4 Per-frame timing
- **Location**: `Game.run` at `game.py:1359-1384`. `frame_times` list capped at 60 frames; `physics_ms` measured in `_process_tick` at 1742-1744.
- **Migration notes**: pyray has its own frame timer; re-wire the panel to read from the new source.

---

## 9. Migration Order

Each step should leave a runnable game on `main.py`.

1. **Lift simple data classes into `src/simulation/`** (no behavior change):
   - `material.py` (MAT_* constants, MATERIAL_COLORS, ticks_per_tile)
   - `order.py` (Order, ORDER_* constants, DET_* constants — consider dataclasses)
   - `projectile.py`, `shot.py` (Shot becomes an event type)
   - **Test**: imports work, `main.py` unchanged.

2. **Move `GameMap` into `src/simulation/world.py`**, merging with the shim in `main.py`. Drop `_build_ship`. Wire `main.py` to use the full version. Verify physics + render unchanged.

3. **Lift `Unit` into `src/simulation/unit.py`**. Spawn the same demo squad + zombies in `main.py` and confirm they appear (using a placeholder marker until the sprite layer is rebuilt).

4. **Add `World.step(actions, dt)` (new file `src/simulation/world.py` or `simulation.py`)** wrapping the tick loop body from `_process_tick` (game.py:1696-1747). Initially no input — just projectiles, paths, shooting, AI, physics, recorder.

5. **Port pathfinding-driven movement** (`_compute_player_paths` at 1566-1639). Add a debug command in `main.py` to set a fake order and watch a marine walk.

6. **Port shooting + LOS** (`_process_shooting`, `_auto_fire`, `_fire_burst`, 1749-1865). Emit `Shot` events for the renderer to draw tracers.

7. **Port zombie AI** (`_update_zombies_tick`, 1867-1979). At this point a runnable scripted scenario exists.

8. **Port explosions** (`Physics.apply_explosion` 704-741 + `_apply_blast_damage` 1981-1996 + `_add_explosion_smoke` 1998-2012 + `_process_door_explosives` 1641-1658).

9. **Port turn machine** (planning ↔ execution, exec_tick loop, end-of-round zombie conversion 2014-2043). Replace `main.py`'s "run forever" with the two-state machine.

10. **Port input / order placement** (`_handle_planning_event`, `_place_order`, undo, mouse wheel). Translate pygame events to pyray polling.

11. **Port UI panel** (mode selector, selected-unit panel, timeline, perf HUD). Re-implement in `renderer/`.

12. **Re-wire debug tools** (PhysicsRecorder hookup, F5 reload, F8 dump, frame timing).

13. **Retire `game.py`** — move it to `legacy/` or delete after a final diff to make sure nothing was missed.

**First runnable checkpoint** is after step 2 (full `GameMap` in `main.py`). **First "playable"** is after step 10. Steps 1–4 should be doable in a single session.

---

## 10. Decisions Erik Should Make Before Migration

### A. Simulation/presentation boundary — strictness?
The architecture doc commits to a hard split with `get_state() / apply_action() / step()`. `game.py` does **not** honor this — `Shot` tracers, sprite loading, frame timing, and physics_ms all live on `Game`. **Decision**: do we enforce the boundary now (every effect becomes an event, the simulation has no pygame/pyray imports), or keep practical shortcuts for v1? Recommend enforcing; the cost is small and self-play AI training needs it.

### B. Unit class structure
Marine vs. zombie share the `Unit` class with ~6 zombie-only fields and `zombie_speed_override` monkey-patched after construction. Options:
1. Keep as-is, accept the wide state surface.
2. Subclass: `Marine(Unit)`, `Zombie(Unit)`.
3. Components: `Unit` carries an optional `ZombieBrain`, `MarineLoadout`, etc.
Recommend **(2)** for now — simple and matches the team split.

### C. Order types — keep or redesign?
Six order types currently (`MOVE_ATTACK`, `MOVE_COVER`, `SPRINT`, `GRENADE`, `EXPLOSIVE`, `FIRE`). Movement variants differ only in `ticks_per_tile`. `MOVE_COVER` and `MOVE_ATTACK` have no other gameplay distinction in code (no actual cover mechanic, no auto-fire on `MOVE_COVER`). **Decision**: prune to `MOVE / SPRINT / FIRE / GRENADE / EXPLOSIVE` and reintroduce cover as a posture flag later?

### D. Per-tick movement vs. precomputed move_path
`_compute_player_paths` builds the entire phase's per-tick positions upfront. This is fast and deterministic but blocks reactive behavior. `pathfinding.py:289` has a `temporal_astar` + reservation table already (unused). **Decision**: keep precompute for v1 and switch later, or bite the bullet on temporal A* now?

### E. Real-time vs. turn-based
Currently turn-based with `planning_phase` UI plus per-tick execution playback. `main.py`'s demo loop is real-time (no planning phase). **Decision**: keep the planning/execution structure, or move toward continuous play with pausable orders? The phase structure (2 phases per round, 3 detonation slots) is the load-bearing UX choice — don't change it without intent.

### F. Order validation feedback
`_place_order` silently `return`s on failure (no AP, no inventory, blocked tile). **Decision**: surface failures as toasts/log messages now or later? `apply_action()` should return a result either way.

### G. Sprite handling
Pygame sprite Surfaces (`game.py:1326-1354`) die with the pygame import. The `name -> sprite_index` round-robin assignment is deterministic gameplay data (which sprite each zombie gets is stable across reloads). **Decision**: store `sprite_id` on Unit and let the renderer resolve it, or push the assignment into the renderer?

### H. Shot/projectile as event vs. state
`shots` and `projectiles` are stored on `Game` and rendered directly. **Decision**: simulation emits `ShotFired(t, fx1, fy1, fx2, fy2)` events; renderer maintains its own short-lived effect queue. Cleaner, matches A.

### I. PhysicsRecorder coupling
Currently inline in `_process_tick`. **Decision**: keep as a step inside `World.step()`, or as an external observer registered with the simulation? The latter is cleaner but adds a hook system.

### J. Material constants — where?
`MAT_AIR/HULL/WOOD/DOOR` are used by both world (passability, destruction) and rendering (colors). They should live with the simulation; renderer reads them. `MATERIAL_COLORS` should live in `renderer/` or be moved entirely to `config.toml`.

---

*End of inventory. Total line ranges audited: `game.py:1-2639`. Cross-references verified against `main.py`, `pathfinding.py`, `level_loader.py`, `docs/architecture.md`.*
