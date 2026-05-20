# Breach — TODO

> What needs to be done. Not what's done — git has that.

---

## Pending — small (background, queue up next session)

- **Smoke at vacuum tiles** — smoke draws over vacuum where stars are
  meant to show. Zero `gmap.smoke` at vacuum positions before uploading
  to the overlay texture. ~30 second fix in renderer/game_renderer.py.

- **Scorch marks** — grenades and fire should leave permanent visual
  marks on the floor/walls where they hit. Persistent darkening, soot,
  burn patterns. **Design now in `graphics_lighting_design.md` §7
  (Destruction Painting Layer)** — single edit-texture approach, with
  normal-map dot product giving directional grenade burns. Ready to
  implement.

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

## Physics — Open Items

3. **Breach decompression fix** — sponge layer works but isn't physical. See `atmosphere_solver_analysis_and_patch_plan_20260319.md`. Not blocking but worth fixing.
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
