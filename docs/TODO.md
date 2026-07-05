# Breach — TODO

> What needs to be done. Not what's done — git has that.

---

## Pending — small (background, queue up next session)

- **Fire never destroys furniture (audit rider, weapons W2, 2026-07-05)** —
  the C++ fire's burn-through list is `is_wall`-gated, so a burning crate
  depletes its fuel (`wall_hp`) but the tile itself survives as a husk;
  meanwhile bullet chew (W2 widened `destroy_wall` to `material != MAT_AIR`)
  CAN break crates. Inconsistent on purpose for now — fix belongs to the
  fire system (engine/06): let burn-out destroy/convert furniture-class
  tiles too (burn-to-wreck material conversion is the nicer answer).

- **Scorch marks** — grenades and fire should leave permanent visual
  marks on the floor/walls where they hit. Persistent darkening, soot,
  burn patterns. **Design now in `graphics_lighting_design.md` §7
  (Destruction Painting Layer)** — single edit-texture approach, with
  normal-map dot product giving directional grenade burns. Ready to
  implement.

- **Blood splats** — reuse the destruction-painting tech from scorch
  marks for blood. Triggered by ranged/melee damage and unit death;
  brush is a dark-red feathered blob, no normal-map relief. Design
  added to `graphics_lighting_design.md` §7.5.

- **Fire as a short-range light source** — wire burning tiles into
  the raycaster as `LightSource` instances with small `max_range`
  (2–4 tiles). Design in `docs/fire_design_notes.md`. Lands with the
  upcoming physics-engine pass.

- **1-bounce raycaster with surface tint** — light rays bounce once
  off walls, tinted by the surface colour. Cheap caustics-lite for
  metal corridors and coloured rooms. Note in `architecture.md` §7
  flags this as "secondary priority".

---

## Gameplay / graphics — before returning to physics (Erik, 2026-05-23)

Three small items to land before the next deep physics-engine pass.

1. **Ambient lighting + two-kinds-of-lights discussion** — picks up the
   prior thread (see [[project-lighting-vision]] in memory and
   `graphics_lighting_design.md`). Goal: an ambient floor of light that
   reveals room geometry plus the existing directional/raycast lights
   for flashlights, fires, emergency. Two roles, distinct rendering paths.

2. **Line-of-sight for AI + players** — don't draw zombies (or any
   enemy unit) that the player has no LOS to (e.g. behind closed doors,
   around corners). Representation of "areas we don't see" is undecided
   — fog-of-war shroud, dimmed render, simple "don't draw"; start with
   the simplest "don't draw units there" and iterate. LOS check exists
   in `gamemap.has_los` (Bresenham); the question is how to integrate
   it into both the renderer (visibility filter) and the AI (already
   uses it for trigger detection).

3. **Wall collision** — units can currently walk through walls during
   execution (no per-tick collision check; only `is_passable_block`
   at order placement). Need real collision in the movement loop.
   Also: **grenades should bounce** off walls (currently they just
   stop / detonate). Grenade *explosions* still destroy walls as today.

---

## Resolution audit & consolidation

Tile size / pixel resolution decisions are sprinkled across multiple docs and
sometimes contradict each other (e.g. `graphics_lighting_design.md` still says
"32px tile — exact size TBD"). The actual decisions are presumably resolved in
code now.

Task:
1. Find every doc and code location that touches on resolution / tile size /
   sprite size / normal-map dimensions / physics-vs-render resolution.
2. Consolidate the canonical decisions into a single doc
   (e.g. `docs/resolution.md`).
3. Audit: for each claim, is it (a) still the design intent and
   (b) actually what the implementation does?
4. Update or remove stale resolution mentions in the other docs; have them
   point to the canonical doc instead.

---

## NEXT SESSION: Build One Complete Level

_Priority #1 — everything else is blocked on having a real testbed._

> Status note: the current `unhcr_vessel` level is **render-complete** (walls
> cast shadows, normal map present). The goal below — a fully layered level
> running every physics system end to end — remains OPEN.

**Goal:** One fully layered level (textures, heightmaps, normalmaps) that can run
all physics systems (atmosphere, smoke, fire, water, water-air coupling). Pick a
reference ship section and build it end to end.

**Plan:**
1. **Research tools** — search the internet for the best tools to create layered
   ship levels. We need tools that can produce: diffuse textures, height maps,
   normal maps — ideally with layer-based workflows matching the ship build-up
   described in the graphics docs.
2. **Choose a workflow** — evaluate candidates, pick a pipeline. Consider:
   what creates the layers, what exports the maps, how do we scale/align to our
   pixel grid (may need manual or heterogeneous scaling).
3. **Choose materials** — decide on surface types (metal hull, deck plating,
   interior walls, grating, etc.) so the heightmap has real physical detail for
   the water sim to interact with.
4. **Build the level** — execute the workflow. Produce all required layers for
   one complete ship section.
5. **Integration test** — load it into the engine and run every physics system
   on it: atmosphere, smoke, fire, fluid sim, water-air coupling. This becomes
   the permanent testbed.

**Open question:** Pick a specific reference ship/deck type before starting, so
tool research is grounded in something concrete.

---

## Migrate Rendering: pygame → pyray

Replace all pygame rendering with pyray (Python bindings for raylib). Game logic
stays in Python, only the draw calls change. This unlocks GPU-accelerated
rendering, normal map shaders, and aligns with the C++ physics / CUDA pipeline.

- [ ] Set up pyray in the main game loop (window, input, frame cycle)
- [ ] Port tile/map rendering
- [ ] Port unit sprite rendering
- [ ] Port UI (orders, phase indicator, debug overlays)
- [ ] Port physics debug visualization (pressure, smoke, fire colormaps)
- [ ] Remove pygame dependency entirely

**Prototype exists:** `prototypes/raylib_test.py` already uses pyray.

---

## Blocking: One Perfect Level (original notes)

1. **Art assets** — 4 congruent textures: ship hull, interior, skeleton, + normal/height maps. Erik's job, requires graphic design work.
2. **Normal map shader** — integrate into raylib rendering pipeline. Course notes in `breach_graphics_course.md`. Huge visual upgrade once textures exist.

## Physics — recently landed (coefficient model)

- ~~**Pressure-driven wall failure**~~ **DONE** (commit 3fbdc12) —
  `MaterialTable.burst_threshold` + `GameMap.find_burst_walls(max_pops)`;
  `Simulation.step` (after fire burn-through) destroys walls holding a
  pressure differential above their per-material `burst_threshold`, capped
  by `[physics] burst_max_per_tick`, gated by `[physics] burst_enabled`.
  The emergent pressure-relief valve. (Architecture: engine/04 §2.7.)
- ~~**Permeability column + permeability boundary**~~ **DONE** (5220148,
  e005c9a) — `MaterialTable.permeability` (default sealed iff occludes
  light); `GameMap.permeability`/`dyn_permeability`; the C++ atmosphere +
  smoke solvers gather flux via `face = min(perm[self], perm[neighbor])`.
  `obstacles` now sourced from `permeability == 0`. Behaviour-identical for
  the current materials.
- ~~**Soft units (units as partial gas)**~~ **DONE** (4f26f0c) — a living
  unit writes a partial `dyn_permeability` (default 0.5, per-unit hook +
  `[physics] unit_permeability`) so smoke/air seep past a body; still casts
  light shadows + impassable to movement.
- ~~**Units absorb blasts (4a)**~~ **DONE** (89026ca) — `GameMap.wave_absorb`
  / `dyn_wave_absorb` (material `wave_absorb` + units via
  `[physics] unit_wave_absorb`); the C++ wave update damps per cell by it.
  Energy-out only; open air bit-identical.
- ~~**Retire `is_wall`**~~ **DONE** (3c99b1c) — `GameMap.solid`
  (= `permeability <= 0`) replaces `is_wall` everywhere in Python.
  *Follow-up still open:* remove the now-vestigial C++ `is_wall` parameter
  (fed `gmap.solid`) + rebuild.

## Physics — Open Items

4. **Breach decompression / lingering-smoke venting fix** — sponge + vacuum
   relaxation work but aren't physical, and leave a stubborn haze in a
   vented room. Face-flux *as a pressure sink* was attempted and reverted:
   with `d_atm = 200` it cannot clear the room (interior gradient flattens
   → wind → 0). The real fix needs a *sustained continuity wind toward the
   breach* — an open design decision. (Architecture: engine/04 §4; smoke
   ch.05.) See `atmosphere_solver_analysis_and_patch_plan_20260319.md`.
4. **Shallow water / fluid simulation** — prototype exists (`prototypes/fluid_test.py`: pipe model + shallow water equations, ship tilting). Needs integration into game engine. Use cases: water flooding, coolant leaks, blood pooling.
5. **Fire ignition model** — ignition as O₂ + temperature function. Explosions deposit heat, temperature diffuses, spontaneous ignition above threshold. Pieces exist but integration glue is missing.

## CUDA Migration

6. **Raycaster → CUDA** — first target, embarrassingly parallel. See `cuda_integration_plan.md`.
7. **Diffusion solver → CUDA** — 2D stencil, textbook GPU kernel.
8. **Wave equation → CUDA** — same pattern as diffusion.
9. **Smoke advection → CUDA** — semi-Lagrangian with GPU texture interpolation.

## Code Cleanup

10. **Remove deprecated solvers** — `wave_solver.cpp` and `atmo_diffusion.cpp` are superseded by `atmosphere_solver.cpp`.
11. **Fix debug_physics.py** — references `WaveSolver` which doesn't exist anymore (should be `AtmosphereSolver`).
12. **Flashlight prototype: take CLI args** — `prototypes/raylib_cpp/flashlight.cpp` hardcodes `art/ships/chatgptSpaceShip1.png` and the matching wall mask / normal map / emissive files. Should accept `--ship`, `--walls`, `--normals` arguments so we can test arbitrary levels without overwriting files. Recompile with cmake after changes.

## Gameplay

12. **Mission 1 implementation** — "Silent Cargo" is fully designed in `missions/missions.md`. Needs the art assets first.
13. **Creature AI** — genetic soldiers and hybrids not yet designed. Zombies work.
14. **Weapons** — need 1-2 more weapon types (at minimum) for Mission 1.

## Future (not blocking anything)

15. **Faction campaign system** — see `missions/campaign_meta_design.md`. Depends on tactical layer being solid first.
16. **Narrative systems** — news cycle, phone notifications, Chase Hughes dialogue. See `narrative_media_systems_update_2026-03-08.md`.

---

## Swept from consolidated docs (2026-06-06)

_Still-open items pulled out of landed review/patch docs + architecture §16
before those docs are archived. Source doc noted in parens._

### Rendering — renderer correctness

- **Drop `tobytes()` in `update_rgba_texture`** — pass the numpy array
  directly to `ffi.from_buffer` (zero-copy) and drop the explicit `cast`;
  removes a per-frame 36 KB×3 copy and a fragile FFI-lifetime pattern.
  (code_review_renderer_v1.md C1)
- **sRGB/gamma handling in `lighting.fs`** — diffuse PNGs are sRGB but the
  shader does lighting math in gamma space; decode on sample and re-encode
  before write (or load diffuse as sRGB texture). (code_review_renderer_v1.md
  C2; patch_level_pipeline_v1.md expert review)
- **Audit normal-map orientation/encoding** — confirm Laigter exports OpenGL
  (Y-up) convention and linear; add a sign-flip toggle so inverted lighting
  is a config change, not an afternoon of debugging. (code_review_renderer_v1.md
  C3; patch_level_pipeline_v1.md expert review)
- **Light-field texture is RGBA8, not float** — quantizes light direction to
  256 angles → visible banding; switch to `R32G32B32A32` float format.
  (code_review_renderer_v1.md M4; patch_level_pipeline_v1.md expert review C++)
- **`load_shader_with_fallback` can't detect compile failures** — use
  `IsShaderReady`/raylib 4.x compile check; warn or fail loud instead of a
  silent black screen. (code_review_renderer_v1.md M5)
- **Validate shader uniform locations** — log a warning when any
  `get_shader_location` returns -1 (silent no-op today). (code_review_renderer_v1.md M3)
- **Move `set_shader_value_texture` calls to *after* `begin_shader_mode`** —
  current bind-before order risks stale sampler bindings once a second shader
  (smoke god-rays, post-FX) is added. (code_review_renderer_v1.md H5;
  architecture_review_camera_rt_patch.md H2; code_review_camera_rt_patch.md C3)
- **Centralize renderer resource cleanup** — give each subsystem
  (`LightingPass`, overlays, dynamic textures, `WorldComposite`) its own
  `unload()`; have `GameRenderer.shutdown()` iterate instead of reaching into
  private GPU handles. Also wrap `__init__` so partial construction cleans up.
  (code_review_renderer_v1.md H3; architecture_review_camera_rt_patch.md M7)
- **`TextureSet`/`WorldComposite` leak on reload** — `unload_all` doesn't reset
  slot attributes; `WorldComposite.unload` leaves `self.rt` dangling. Reset to
  None so a second `load_level_textures`/level switch doesn't leak or crash.
  (code_review_renderer_v1.md H3; code_review_camera_rt_patch.md N1)
- **`PhysicsRunner.step` returns `destroyed` walls but `main` ignores them** —
  consume so the renderer can spawn debris/smoke or invalidate light bakes.
  (code_review_renderer_v1.md M7)

### Rendering — camera / world RT

- **`coords.py` is dead code** — zero imports outside itself; either route
  `Camera2D`/`overlays.draw_unit/draw_waypoint_line/draw_grid` through it
  (rename the `ft` param to `world_px_per_tile`) or delete it. Consider a lint
  guard against naked `* ft` arithmetic. (architecture_review_camera_rt_patch.md
  H1; code_review_camera_rt_patch.md M5)
- **`compose_world` signature will balloon** — split into
  begin/draw_terrain/draw_entities/draw_overlays/draw_grid/end (or a `Scene`
  object) before projectiles, particles, decals, debug HUDs land.
  (architecture_review_camera_rt_patch.md H3)
- **Camera/RT smoke test doesn't test** — `tests/test_renderer_smoke.py` is an
  interactive demo: no assertions, no camera pan, no headless CI run. Add a
  pan-camera + place-light + read-back-pixel assertion to catch Y-flip /
  clamp / inverse-transform regressions. Drop unused `os`/`math` imports.
  (architecture_review_camera_rt_patch.md H4, C2 `__all__`/docstring;
  code_review_camera_rt_patch.md N4)
- **`mouse_to_tile` assumes viewport anchored at screen (0,0)** — breaks for
  the planned security-cam inset / offset blit; store viewport screen origin on
  `Camera2D` or route through the dst-rect. Use `math.floor` not `int()`.
  (code_review_camera_rt_patch.md C1; architecture_review_camera_rt_patch.md M2)
- **Camera viewport not updated on resize / zoom** — `FLAG_WINDOW_RESIZABLE`
  is set but `viewport_px_w/h` are baked at construction; wire `IsWindowResized`
  → update camera + RenderConfig + scissor, or drop the resize flag.
  (code_review_camera_rt_patch.md C2; code_review_renderer_v1.md H4;
  architecture_review_camera_rt_patch.md future-proofing)
- **Zoom-out past world bounds smears edges** — `clamp_to_world` lets the
  visible rect exceed world size at low zoom; clamp rect size to world (letterbox
  or stretch) and raise the `set_zoom` floor so `viewport/zoom <= world_size`.
  (code_review_camera_rt_patch.md H2, H3)
- **`clamp_to_world` not called on construction / manual pos set** — clamp in
  `__post_init__` so a level smaller than the initial camera pos doesn't start
  off-world. (code_review_camera_rt_patch.md M4)
- **Pixel-art filter decision is accidental** — bilinear is set on RT and on
  smoke/fire textures (double-blur); decide POINT-vs-BILINEAR per texture
  deliberately and document it. Add a `pixel_perfect` snap option.
  (code_review_camera_rt_patch.md H4; architecture_review_camera_rt_patch.md
  future-proofing)
- **`world_px_per_tile = 24` is unvalidated magic** — compute from diffuse
  dimensions or assert it matches the asset; add a guard that RT size won't blow
  GPU memory on large ships. (architecture_review_camera_rt_patch.md M6;
  code_review_camera_rt_patch.md M6)
- **Add camera helpers before they're written inline** — `zoom_at(focal)`
  (zoom-around-cursor), `add_shake`, `follow(target, lerp)`.
  (architecture_review_camera_rt_patch.md M3, M4)
- **Generalize `blit_world_to_screen` → `blit_view(camera, dst_rect)`** for the
  planned multi-camera / security-cam inset; drop the redundant scissor on the
  full-viewport blit. (architecture_review_camera_rt_patch.md future-proofing;
  code_review_camera_rt_patch.md H1)
- **Multiple opposing lights cancel direction to (0,0)** → "dead spots" where
  cones meet; switch to per-tile dominant-direction (running max) before
  shipping multiple emergency lights. (code_review_renderer_v1.md future-proofing)
- **Decide smoke-vs-unit draw order / occlusion** — units currently draw in
  front of smoke even when inside a cloud; open question whether smoke should
  occlude units. (architecture_review_camera_rt_patch.md M5)
- **Future-proofing render hooks** — `FieldOverlay` is the wrong abstraction for
  particles (need a `ParticleSystem` owning a RenderTexture/batched quads);
  `draw_unit` draws a circle and needs a sprite-atlas batcher for 30+ units;
  add HDR/multi-RT format param to `WorldComposite` before it has many callers.
  (code_review_renderer_v1.md future-proofing; architecture_review_camera_rt_patch.md
  future-proofing)
- **Fix `renderer/__init__.py` docstring + `__all__`** — documents a removed API
  (`draw_world`/`draw_units`/`draw_overlays`); export `RenderConfig`.
  (architecture_review_camera_rt_patch.md C2; code_review_renderer_v1.md N1)
- **Remove stale `math` imports** in `lighting.py`, `overlays.py`,
  `game_renderer.py`. (code_review_renderer_v1.md N3)
- **Cross-check level asset dimensions** — `level_loader.py` validates files
  exist but not that diffuse/normal/etc. share dimensions; a mismatched normal
  map samples wrong silently. (code_review_renderer_v1.md N6)

### Physics / simulation

- **Smoke-at-vacuum-tiles bug** — zero `gmap.smoke` at vacuum tiles before
  upload (~30s fix). (patch_game_logic_migration.md after-migration follow-ups)
- **Liquids system not implemented** — per-tile liquid type (none/blood/water/
  fuel) + depth as state fields, interacting with fire/creatures/electricity;
  needs its own design pass. (architecture.md §16 #11)
- **Double-buffered propagation not implemented** — introduce during C++ port;
  consider replacing the per-frame physics state copy with a buffer swap.
  (architecture.md §16 #12; patch_game_logic_migration.md follow-ups)
- **Fire wall-destruction double-loop** — replace `for fy/for fx if burned_out`
  with `np.argwhere(burned_out)`. (architecture.md §16 #5)
- **Smoke/fire have no substeps** — run once per tick at 83ms dt; evaluate
  stability under large parameters. (architecture.md §16 #6)
- **Config inconsistency** — wave/fire/explosion parameters hardcoded instead
  of in config.toml; unify all tunables. (architecture.md §16 #1)
- **Plumb the seeded RNG through all nondeterminism** — `_add_explosion_smoke`
  (smoke noise), `Raycaster.cast_source` (fire flicker jitter), `_fire_burst`
  (bullet cone) must pull from `Simulation.rng` or AI rollouts/replays diverge.
  (review_game_logic_migration.md C3; patch_game_logic_migration.md Step 8)

### Units / gameplay

- **Phase-transition detection is fragile** — `new_phase = exec_tick // tpp`
  works but consider explicit phase-boundary tracking. (architecture.md §16 #7)
- **Turn/phase flow regression** — flagged at migration playtest; symptom not
  captured (candidates: transition trigger, pause-release timing, order
  resolution at tick boundaries). Reproduce + diagnose against legacy behaviour.
  (patch_game_logic_migration.md playtest #6)
- **Explosion visuals are weak** — no flame burst/flash; port or redesign the
  legacy pressure-to-color drama. (patch_game_logic_migration.md playtest #1)
- **Explosions should emit a transient light source** at the blast center for a
  few ticks (renderer-side transient `LightSource` or sim-side temp entity).
  (patch_game_logic_migration.md playtest #2; cross-ref memory
  project_explosion_as_light_idea.md)
- **Strong/weak zombie variants** — runner (low HP, fast) vs brute (high HP,
  slow); `unit.py` carries `speed_ticks_per_tile`/vitality, wire up variant
  spawning. See `unit_variants_design_brainstorm.md` and the `level.toml` TODO
  for the ogryn-zombie spawn. (patch_game_logic_migration.md playtest #5)
- **Per-creature AI sample rate** — humans slow, robots fast.
  (patch_game_logic_migration.md follow-ups)
- **Promote inventory booleans into `Inventory`** — `has_grenade`/`has_explosive`
  still live on Unit alongside the stub `Inventory`; migrate them in.
  (patch_unit_class_foundation.md decision 8)
- **Non-symmetric footprint rotation** — `occupied_tiles()` applies no rotation;
  add rigid-body rotation for non-symmetric footprints (spec §15 item 3).
  (patch_unit_class_foundation.md §5)
- **Unit-system deferrals (data exists, behaviour missing)** — modifier system
  (`compute_effective_stats` returns base), environment damage
  (`EnvironmentProfile` is data-only), faction relationship table (combat still
  uses `team != team`), fear / Gray hook / awakening trigger. Per spec §13.
  (patch_unit_class_foundation.md decision 9)
- **`apply_action` result/legality convention** — commit to a `Result` enum
  (OK/NO_AP/NO_INVENTORY/BLOCKED) and/or `get_legal_actions` so the UI can show
  failure toasts instead of silent returns. (review_game_logic_migration.md
  #3, #12)
- **Order subclassing** — split single `Order` discriminator into
  `MoveOrder`/`FireOrder`/etc. (its own patch, deliberately deferred).
  (patch_game_logic_migration.md anti-goals; review_game_logic_migration.md #9)

### Pathfinding

- **Temporal A* unused** — `ReservationTable` + `temporal_astar` exist but are
  never called; player units can overlap during execution. Enable or remove.
  (architecture.md §16 #4)
- **Pathfinding constants hardcoded** — `FINE_W=120`, `FINE_H=75`,
  `UNIT_SIZE=3` in pathfinding.py don't reference config. (architecture.md §16 #8)

### Cleanup / resolution

- **Remove the coarse-tile concept** — ~66 references in legacy code, some dead,
  some needed for unit footprints; dedicated cleanup.
  (patch_level_pipeline_v1.md "What this patch does NOT cover")
- **Aspect-ratio / diffuse alignment** — 972×1619 diffuse vs 50×120 tilemap is
  stretched for v1; align the art to tilemap bounds (manual or automate).
  (patch_level_pipeline_v1.md open question #2)
- **`_draw_ui_panel()` is 207 lines** — consider a lightweight UI layout system.
  (architecture.md §16 #9)

### AI training infrastructure

- **AI training scaffolding (`train.py`)** — Gymnasium loop over `Simulation`;
  add `get_reward`/`is_terminal` hooks (facade stubs exist). Separate patch.
  (patch_game_logic_migration.md follow-ups; review_game_logic_migration.md #3)
- **Save/load + `serialize()/deserialize()`** — nearly free given `get_state()`
  returns flat arrays; needed for replay buffers. (review_game_logic_migration.md
  #13; patch_game_logic_migration.md testing strategy)
