# Doc Consolidation — Reading List & Order

_Created: 2026-06-06 · **Temporary working doc** — delete once every design doc is folded into the
`docs/architecture/` chapters and the originals are removed._

**Purpose:** reorganize the scattered design docs into the new **one-file-per-system** canon
(`docs/architecture/`). Read + fold in **dependency order (foundation first)**, ticking each off.

**How to use:** `- [ ]` → `- [x]` as you read + fold a doc. Tags: **CANON** (already converted) ·
**FOLD** (design to consolidate) · **SUPERSEDED** (already in a chapter — delete, don't re-read) ·
**EXCLUDE** (process/archive, not design canon) · **CONTENT** (separate track).

**Definition of "done" per chapter** (how rays were finished): (1) a doc that matches the *code*,
not aspiration; (2) decisions locked + extension points named; (3) tests cover the contract;
(4) old/contradicting docs superseded or deleted.

**Principles** (now in `README.md`): *canon over prototype* · *canonize bottom-up*.

---

## Target chapter structure (`docs/architecture/`)

Existing: `01_state_and_ownership` · `02_material_system` · `03_ray_engine` · `04_temperature_and_heat`
· `05_lighting_and_render`.
Planned (rough): `06_atmosphere_pressure` · `07_smoke` · `08_fire` · `09_fluid_water` ·
`10_electricity` · `11_units_entities` · `12_ai_los` · `13_combat_weapons` · `14_turn_and_control`
· `15_destruction` (scorch/blood). `cuda_integration_plan.md` stays a *plan*, not a system chapter.

---

## ① Foundation — Materials / Map / Coords → ch.01, ch.02 (+ a grid/levels chapter)

- [x] **CANON** `architecture/01_state_and_ownership.md` — re-read + reconcile to code
- [x] **CANON** `architecture/02_material_system.md` — re-read + reconcile to code (the `is_wall` drift)
- [ ] **FOLD** `map_and_physics_design.md` — richest map+physics source (feeds 02 + seeds atmosphere)
- [ ] **FOLD** `patch_coord_system_cleanup.md` — coord system (work done; fold locked decisions)
- [ ] **FOLD** `patch_level_pipeline_v1.md` — level / CSV loading pipeline
- [ ] **FOLD** `design_camera_and_coordinate_systems.md` — coords (foundation) + camera (render); split
- [ ] **EXCLUDE** `design_camera_and_coordinate_systems_research.md` — background; skim only

## ② Physics fields — Atmosphere → ch.06 ; Smoke → ch.07

- [ ] **FOLD** `atmosphere_solver_analysis_and_patch_plan_20260319.md` — atmosphere/pressure solver (extract the *design*, drop the patch-process parts)
- [ ] (smoke design likely lives inside `map_and_physics_design.md` + `graphics_lighting_design.md §4` — confirm when here)

## ③ Fire / Temperature → ch.04 (exists) + ch.08

- [x] **CANON** `architecture/04_temperature_and_heat.md` — canon *design* (not built yet)
- [ ] **FOLD** `fire_design_notes.md` — fire ignition/spread → ch.04 / ch.08
- [ ] **SUPERSEDED** `implementation_plan_radiation_temperature.md` — already in ch.03/04; banner'd → delete (don't re-read)

## ④ Graphics / Render extras → ch.05 + ch.15 (destruction)

- [x] **CANON** `architecture/03_ray_engine.md`, `architecture/05_lighting_and_render.md`
- [ ] **FOLD** `graphics_lighting_design.md` — *partly* in ch.03/05; fold the **unique** parts (destruction/scorch/blood §7, smoke normal-maps, stealth) into ch.05 + ch.15
- [ ] **EXCLUDE** `breach_graphics_course.md` — learning notes; keep as reference

## ⑤ Units / Entities / AI / Combat → ch.11, ch.12, ch.13

- [ ] **FOLD** `breach_unit_class_design.md` — the unit spec (most mature; near-canon already)
- [ ] **FOLD** `patch_unit_class_foundation.md` — what got built (fold locked parts)
- [ ] **FOLD** `unit_variants_design_brainstorm.md` — brainstorm → fold decided parts, mark rest "ideas"
- [ ] **FOLD** `breach_metaphysics_design_notes.md` — deferred design → "future" chapter/appendix

## ⑥ CUDA

- [ ] **FOLD/KEEP** `cuda_integration_plan.md` — stays a plan; §3/4/7 banner'd, reword when convenient

---

## EXCLUDE — process artifacts (history, not design canon; leave or archive)

`code_review_renderer_v1.md` · `code_review_camera_rt_patch.md` · `architecture_review_camera_rt_patch.md`
· `review_game_logic_migration.md` · `patch_game_logic_migration.md` · `game_py_inventory_and_migration_plan.md`
· `patch_lighting_demo_tool.md` · `ai_tile_generation_research.md` · `TODO.md` (living — keep) ·
`dev_setup.md` (keep — reference) · everything in `archive/`

## CONTENT track — real design, but not architecture/system docs (own home)

`missions/missions.md` · `missions/campaign_meta_design.md` · `narrative_media_systems_update_2026-03-08.md`
· `lore/*` (historical_events_inspo, lore_the_femme_fatale, lore_the_grays, story_research_watergate_and_princes)

## DELETE after folding (superseded scaffolding)

`implementation_plan_radiation_temperature.md` (→ ch.03/04) · `ray_engine_reconciliation.md` (the
ray tracker) · this file (`_consolidation_plan.md`) when consolidation is complete
