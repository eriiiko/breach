# Priority ledger

**What we want to work on and complete, in order.** Coarser than a roadmap:
roadmaps (e.g. `roadmap_2026-07.md`) plan a window; this ledger holds the
standing stack so any session can orient in ten seconds. Update it whenever a
priority is decided, finished, or dropped — a stale ledger is worse than none.
Created 2026-07-17 from Erik's stated stack; Erik owns the ordering.

---

## The stack

### 1. Physics engine v1 — close it out
- **EOS residency patch** — the last rung-B step: stop streaming all fields
  GPU→CPU each tick (S8a). Finishes the EOS arc. ⚠ `cuda_s8a_residency_spec.md`
  is pre-EOS (lists retired fields incl. wave_p) — rewrite the spec first.
- **Boundary conditions** (space vs planetside, per-map) — next physics
  project, "pretty much a must," and surveyed SMALL (2026-07-17): an
  AMBIENT border-ring tile symmetric to the existing SPACE ring (MG pins
  P=P_amb instead of 0; species reservoir; optional sponge band). Existing
  space-map goldens untouched. Spec before/alongside the residency patch
  (same kernels). Details: `notes_2026-07-17_topics_backlog.md` Topic 4.
- Riders (chat-sized, slot when convenient):
  - ~~Wall-burst differential fix~~ **DONE 2026-07-18** — merged to main
    (true differential; only 1-deep membranes burst; Erik blessed).
  - **Dust-stirring shockwaves** — dusty-ground flag + wave_p threshold →
    smoke injection (notes 2026-07-17 Topic 1).
  - Post-EOS doc consolidation (roadmap §1.3 rider).

### 2. Weapons, units, classes, a small enemy roster
- Weapons wave finale: W6 armory tuning session (Erik, human-gated — see
  TODO.md) → merge → wave close.
- Unit classes per `breach_unit_class_design.md`; enemy marines; a few
  critters (see `beastiary/beastiary.md`); zombies already work.

### 3. The vertical slice — "Counter-Strike, but the map fights back"
Two opposing teams with objectives. The twist is the physics sandbox:
- destructible map, pressure/fire/water fully in play — rooms can be
  flooded, atmosphere drained, walls blown through;
- an **animal pen** that can open (by plan or by damage) and release
  critters for further chaos;
- **zombies as a third faction**: hostile to both teams, and infection
  creates more zombies mid-match.
This is the setting the ML end-goal trains in: chaotic initial conditions,
genuinely different rounds, agents that learn to *handle* fire/flood/poison
rather than memorize an optimal line (`missions/mission_ideas.md` ML note).

### 4. The end goal (standing, shapes everything above)
Self-play NN training on the finished physics — train once, on final
physics. Big training runs wait for the S8 optimize-hard pass.

## Side tracks (not blocking the stack)
- **Procedural skeletal animation** — marines first (render-only), then the
  menagerie; post physics-v1. `procedural_animation_brainstorm.md`.
- **Entity system + editor v3** — **DESIGN LOCKED 2026-07-18**
  (`entity_system_design_2026-07-18.md` canon model +
  `level_editor_v3_design_2026-07-18.md` view; 48-finding adversarial
  critique folded; Erik approved). Build order (Erik):
  **Arc A** (entity foundation: registry-in-code, [[entity]]+ids, doors v0
  + EOS evacuation prerequisite, zones/air/boundary format, level_lib) may
  start on Erik's word → **physics close-out (stack #1, S8a spec must
  include the sensor-gather contract)** → **Arc B** (SignalBus dataflow
  logic, sensors, pump, automatic airlock) → **Arc C** (editor UX panes,
  wand, wiring, play-from-editor, icons). AI tilesets (levels-w1 P6) still
  parked behind it.
- **Sound-ML** — parked (`sound_ml_research_brief.md`), junior to the EOS arc.
- Beauty tracks: black-body emitter, smoke visuals, scorch/blood painting.

## Next chat-sized sessions (each its own chat)
1. Boundary-conditions spec (against landed rung-B; decide mode set + level
   format field + kernel touch points).
2. Wall-burst differential fix (+ threshold re-tune + tests).
3. EOS residency patch (if not already in flight).
4. W6 armory tuning (Erik) → weapons wave close ritual.
5. Dust-stirring shockwaves spec/impl.
6. Animation P0 — marine prototype, 3D-model-vs-part-sprites by eye.
