# Breach — TODO

> What needs to be done. Not what's done — git has that.

---

## Blocking: One Perfect Level

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

## Gameplay

12. **Mission 1 implementation** — "Silent Cargo" is fully designed in `missions/missions.md`. Needs the art assets first.
13. **Creature AI** — genetic soldiers and hybrids not yet designed. Zombies work.
14. **Weapons** — need 1-2 more weapon types (at minimum) for Mission 1.

## Future (not blocking anything)

15. **Faction campaign system** — see `missions/campaign_meta_design.md`. Depends on tactical layer being solid first.
16. **Narrative systems** — news cycle, phone notifications, Chase Hughes dialogue. See `narrative_media_systems_update_2026-03-08.md`.
