# Breach — TODO

> What needs to be done. Not what's done — git has that.

---

## Waiting on Erik (human-gated)

- **Armory tuning session — AFTER mission 1 exists (re-scoped 2026-07-21)** —
  W6 merged with standard values (PR #3, `f482131`); Erik's call: balance is
  meaningless against an empty playground, so the grand tuning session waits
  until weapons → units → enemies → a first mission are all in place, then
  tune against real encounters. Tool: **N** cycles the selected unit's weapon
  (walkthrough `docs/playground_guide.md` §9). Standing dial list: the
  chain-stun pair (`rof_interval_seconds` vs `status_seconds` on
  `[weapons.arc_baton]`), the plasma-vs-zombie resist wash (bullet ×0.25 then
  HEAT ×4 ≈ no-op), flamethrower feel at the 10 m / 20 m meter-based ranges —
  plus whatever mission 1 teaches. Quick residual: 30 s look at the W6 jet
  fans + 3D marines rendered together (never exercised in one window).


## Open threads — cross-arc index (2026-07-22, "don't forget")

One place to see every loose end left by the recent burst of simultaneous
patches (Arc B, W6, Fire & Heat, S8c, animation). Each points to where the real
detail lives; this list is the map, not the territory.

**The big fork that gates other things:**
- **Control-scheme decision** — WEGO vs direct gamepad control (one marine).
  Gates: unit-initiated interactions (`button`/terminal are inert until then),
  the **manual airlock buttons** below, and any AP/phase assumptions. Nothing
  new should bake in AP/phase until this is decided. (Entity design §3d.)

**Arc B (entity logic layer — DONE + merged 2026-07-22), its two parked riders:**
- **Resident sensor-gather kernel** — the §5a `(n_sites × n_channels)` int32 GPU
  gather buffer. Arc B stubbed the accessor to the CPU mirror; interface is
  FROZEN, only the GPU impl is missing. Belongs to the **S8 / CUDA-residency
  line** (throughput for batched training — nothing broken without it). **Now
  unblocked** (S8c landed). ↓ "Arc B follow-ups".
- **Manual airlock buttons ("airlock v2")** — gated on the control-scheme fork
  above. ↓ "Arc B follow-ups".

**Weapons (W6 — merged):**
- **Armory tuning session** — deferred until weapons→units→enemies→mission 1
  exist, then tune against real encounters. ↑ "Waiting on Erik".

**Fire & Heat (active arc, NOT a loose end — living plan
`docs/plan-for-tuning-and-graphics.md`):**
- Fire B1 blackbody lights shipped; **Fire B2 smoke-honesty** in flight
  (`fire_b2_smoke_honesty_design_2026-07-21.md`). Fire *tuning* — "burns too
  easily" + the fire→pressure link — is on that plan. (Tracked there; listed
  here only so it isn't mistaken for forgotten.)

**S8c (fire-heat batch — item 1 merged):**
- **Items 2 & 3 DEFERRED** (accepted gaps): render CUDA-GL interop + recorder
  kernels. Part of the S8-optimize line. (`s8c_items_2_3_deferred_2026-07-21.md`.)

**Physics riders on the books (ledger stack #1):**
- **Blast-pressure-threshold material column** — direction decided 2026-07-19;
  impl + tuning is a chat-sized HUMAN-TEST rider (`physics.py:104` blast-tuple
  wart retires here). · **Dust-stirring shockwaves** — dusty-ground flag +
  wave_p threshold → smoke injection. · **Post-EOS doc consolidation.**

**Animation track:** the weapon/grenade + shockwave-push question (just folded
from TODO2.md) + the appearance/skin-pipeline items — all ↓ "Animation".

**Housekeeping:**
- The `.claude/worktrees/arc-b-logic` directory lingers on disk (Windows
  file-lock after the arc merge); git is clean — `Remove-Item -Recurse -Force`
  it after a reboot.


## Animation / character-render track (3D marines shipped 2026-07-21)

**Shipped (arc `anim-phase0-3d-marines`, merged 2026-07-21):** render-only 3D
marines/zombies over the 2D world (toggle **M**, default off), lit by the
raycast light field so they match the ship, 2× scale, blob shadows; a
tangent-free normal-map *capability* is present but default-off (the Quaternius
model is untextured, so nothing to reveal). Docs: `marine_shader_foundation_design_2026-07-20.md`,
`research/ml_animation_litsearch_2026-07-20.md`, `procedural_animation_brainstorm.md`.
All render-only — no sim/determinism surface, auto-skipped in headless training.

**Wanted next (Erik, 2026-07-21 — capture, later work):**
- **Marine appearance system** — a clean per-unit *visual profile*: one model +
  animation set + skin per unit type, with **variation** (later). Today it's a
  single shared model + a flat group tint (green marines / red zombies via
  `colDiffuse`). Generalize to `unit-type → {model, skin/material, clip set,
  gait params}` so new looks are data, not code.
- **Zombies look like their victim** — when a unit turns into a zombie
  *mid-match*, it **keeps its own skin/model** (appearance unchanged) but
  **swaps to the zombie animation** (shambling gait), optionally with a
  **"bloodied" overlay**. **Pre-placed** zombies (spawned as zombies) get a
  **dedicated zombie skin**. So: turning = animation swap (+ optional blood),
  NOT a skin swap; pre-placed = special skin. (The current green/red tint is
  fine for now.)
- **Weapon & grenade animations + shockwave-push behaviour** (Erik, folded from
  TODO2.md 2026-07-22) — marines should **carry at least one visible weapon**
  and have a **rifle carry + shoot/fire** animation, plus a **throw-grenade**
  animation. Open question: what should a marine do when **pushed by a
  shockwave** (they already get pushed today) — maybe
  *hunker down* a little on a small push; on a big push, unclear. **Meta-
  principle (Erik, load-bearing for this whole track):** the **ML path is the
  priority** — he has no ambition to be an old-school animator, and much
  hand-authored prep may be *thrown out* once ML drives motion. So weigh every
  animation task by "will ML redo this?" — use the right tool for the job, do
  the minimum that reads well now, and stay open to letting the ML track own
  reactive motion (push/stagger/limp) rather than scripting it.

- **Retire the M toggle + sprite path** (Erik, 2026-07-21) — make the 3D marines
  the **default and only** unit render; drop the old 2D sprite fallback (the
  `use_3d_units` toggle / `M` key / `UnitSprites` unit path) once confident. The
  old sprites won't be needed anymore.
- **Skin / appearance-asset pipeline** (Erik, 2026-07-21) — we need real
  **textured skins** (marine, zombie, variations) to unlock the visual-profile
  system above *and* the already-built normal-map capability (P2). The current
  Quaternius model is untextured. Evaluate **AI generation** — mesh-texturing
  tools that paint albedo + normal/PBR onto the existing rig's UVs (Meshy-style
  AI texturing), or text-to-3D for whole rigged+textured models — vs
  hand-authored. A focused tool eval (like the shader lit-search) is the right
  first step when we pursue it.
- **Body-part damage → animation/behaviour hook** — Erik's `01_units.md` note
  (commit `350179c`): body parts carry hp/damaged states that drive a different
  animation, speed, even behaviour (limping) via the ML animation system. Ties
  render ↔ mechanics; needs refining/planning.

**Deferred fixes/items from this arc:**
- **Move-order animation bug (OURS, not the command system)** — when one marine
  gets a move order, ALL marines' models play the WALK clip while staying put.
  `UnitModelRenderer`'s motion inference mis-selects "walk" for stationary units
  (likely `move_path` non-empty on all during planning, or the position-delta
  test). Fix in the clip/motion-inference; its own session.
- **P2 real asset drop-in** — the normal-map capability is inert until a
  textured/normal-mapped marine asset exists: drop it over
  `assets/models/marine/marine_normal_PLACEHOLDER.png`, flip
  `MARINE_USE_NORMAL_DEFAULT` (or `marine_shader.set_use_normal(True)`), tune
  `MARINE_NORMAL_STRENGTH` — no code change.
- **`fire`/`dead` clips dormant** — wired in `CLIP_MAP` but unused (firing not
  inferred from sim; dead units skipped for sprite parity). One-line extensions.
- **GPU skinning (perf lever, deferred)** — CPU-skinning soft ceiling ~20 units;
  when counts grow, rebuild the raylib binding with `-DSUPPORT_GPU_SKINNING=ON`
  and flip the `_draw_one` seam (the fragment/lighting half is unchanged). Not
  needed yet.
- **Ceiling-lamp z** — lights carry a *constant* vertical component
  (`u_light_z`); a true overhead shaft wants per-lamp/per-tile z. Lighting
  nicety, low prio.

## Pending — small (background, queue up next session)

- **Fire & Heat tuning session (Erik, 2026-07-21, after B1 merged) — DEDICATED
  SESSION.** B1 (black-body overlay + brightest-K fire lights) merged and the
  look is blessed ("much better"), but needs a tuning pass:
  - *Render mapping:* fires read too white at the default `k_temp_to_kelvin=2.0`
    (saturates to white by ~T_game 3000). Dial `k_temp_to_kelvin` DOWN / raise
    `kelvin_ref` so white is reserved for extremes (config `[render.blackbody]`).
  - *Sim — EVERYTHING BURNS TOO EASILY (Erik 2026-07-21, headline):* the whole
    room lit and went white. Primary suspect `k_fire_heat = 1600` (the radiation
    heat deposit — "ignites too fast"); then `ignition_temp = 300` (raise),
    `range_per_intensity = 3` (shorten reach), `o2_threshold = 0.01` (raise the
    ignition O2 gate). Sim-side, NOT render — B1 only reveals it. Verify with a
    headless probe (ticks-to-full-involvement).
  - *Fire↔pressure link:* Erik hit a `[recorder] BLOWUP DETECTED` during the B1
    session. NOTE blowups are PRE-EXISTING (dumps back to 2026-07-12; several
    on 07-21 from W6 explosion testing) — not B1. But over-ignition plausibly
    FEEDS it: a fully-involved room = many `fire_pressure_gain = 0.15` plume
    over-pressures summing → blowup trip. Fixing ignition upstream should ease
    the pressure symptom; confirm during the session.
  - *Diagnostics wanted (Erik's preference over more shortcut keys):* make ONE
    dedicated fire-tuning level and hardcode a per-tile value readout into the
    launch script — OR a small "all values of the hovered tile as a table"
    (T in game-units + pseudo-Kelvin, fire intensity, material, ignition_temp /
    ignited flag). Render-only (reads gmap fields). Deferred on purpose; build it
    as the session opener, not now.

- **Blast-tuple wart (Arc A rider, A6, 2026-07-19) — direction DECIDED
  2026-07-19 at physics close-out:** `apply_explosion`'s structural wall
  damage gates on the hardcoded tuple at `physics.py:104`
  (`MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_DOOR_CLOSED`) instead of the material
  table. Fix: NOT tuple-widening — a per-material **blast-pressure-threshold
  column** in the material table (damage only when local blast amplitude ≥
  threshold; Erik's intent: steel shrugs off many small waves, one big one
  can bite; also enables brittle vs space-rated glass as two rows). Defaults
  reproduce current behavior (excluded materials ≈ ∞ threshold —
  digest-safe). Implement + tune as a chat-sized HUMAN-TEST rider AFTER the
  residency patch (priority ledger stack #1).

- **Baker writeback onto level_lib (Arc A rider, A2 accepted gap,
  2026-07-19)** — `bake_level_art.write_bake_blocks` is still its own
  non-atomic `[art]`/`[bake]` writer; entity design §3c says level_lib is
  THE data layer, all clients. Fold it in at Arc C (editor arc). Ctrl+S
  re-records mtime+hash after baking, so staleness tracking stays honest
  meanwhile.

- **Lights → entities convergence (Arc C candidate; captured 2026-07-22
  during fire-B2 design)** — lamps/beacons still live in the pre-entity
  levels-w1 `[[light]]` schema (`src/level_lights.py`), parallel to the
  Arc A–B entity system (doors/sensors/nodes are entities; lights are
  not). Fold at Arc C: lamp/beacon as entity → on/off state, SignalBus
  wiring (lamp toggled by a sensor!), editor placement — replacing the
  `[[light]]` loader. Fire-B2's `renderer/frame_lights.py` assembly
  helper is input-agnostic and survives this migration (B2 design §2).
  Original vision: tuning-plan §1a "entity/prop system + the LAMP".

- **Legacy-level entity migration (Arc A ruling 2 remainder, 2026-07-19)** —
  only `test_level` was migrated (A7). `bake_demo` waits until its committed
  baked art rebakes (a 3→7 tilemap rewrite would desync tilemap ↔ baked
  PNGs); `unhcr_vessel`/`unhcr_vessel_2`/`playground` migrate at Erik's
  choosing, likely Arc C. Each later migration is a new digest event with
  its own rationale (`docs/archive/a7_rebaseline_rationale_2026-07-19.md`).

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

### Arc B follow-ups (entity logic layer)

- **Manual airlock / "airlock v2"** (Erik, 2026-07-21 HUMAN-TEST) — manual OPEN
  buttons on each side of both airlock doors, alongside the automatic
  `airlock_controller`. Deferred: `button`/terminal are format-reserved but
  INERT in v1 (entity design §3d) — they wake up only with the **control-scheme
  decision** (WEGO → possible direct gamepad control), so manual operation is
  gated on that arc, not Arc B. Slot after the control scheme lands.
- **Resident sensor-gather kernel** — Arc B stubbed the §5a accessor to the host
  mirror (no GPU gather kernel, per the S8c-concurrency constraint). Build the
  `(n_sites × n_channels)` int32 gather kernel on the resident path once S8c has
  landed; the accessor interface is already frozen (cuda_s8a spec §5a).
