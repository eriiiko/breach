# Priority ledger

**What we want to work on and complete, in order.** Coarser than a roadmap:
roadmaps (e.g. `roadmap_2026-07.md`) plan a window; this ledger holds the
standing stack so any session can orient in ten seconds. Update it whenever a
priority is decided, finished, or dropped — a stale ledger is worse than none.
Created 2026-07-17 from Erik's stated stack; Erik owns the ordering.

---

## The stack

### 1. Physics engine v1 — close it out  ← NEXT (Arc A done; this unblocks Arc B)
- **EOS residency patch** — the last rung-B step: stop streaming all fields
  GPU→CPU each tick (S8a). Finishes the EOS arc. Spec REWRITTEN 2026-07-19:
  `cuda_s8a_residency_spec_2026-07-19.md` (post-EOS; carries the
  sensor-gather contract §5a — Arc B gated on it — and the structural
  dirty-set rider §5b; two-rung H2D plan; old pre-EOS spec superseded with
  banner). Awaiting Erik's review, then build per §4.
- **Boundary conditions** (space vs planetside, per-map) — surveyed SMALL
  (2026-07-17): an AMBIENT border-ring tile symmetric to the existing SPACE
  ring (MG pins P=P_amb instead of 0; species reservoir; optional sponge
  band). Existing space-map goldens untouched. **Sequencing DECIDED
  (Erik, 2026-07-19): BC lands BEFORE the residency build** so residency
  freezes final kernel content (S8a spec §5c). Spec it next session.
  Details: `notes_2026-07-17_topics_backlog.md` Topic 4.
- Riders (chat-sized, slot when convenient):
  - ~~Wall-burst differential fix~~ **DONE 2026-07-18** — merged to main
    (true differential; only 1-deep membranes burst; Erik blessed).
  - **Dust-stirring shockwaves** — dusty-ground flag + wave_p threshold →
    smoke injection (notes 2026-07-17 Topic 1).
  - Post-EOS doc consolidation (roadmap §1.3 rider).
  - **`physics.py:104` blast-tuple wart — DECIDED 2026-07-19 (direction):**
    do NOT widen the tuple; replace it with a per-material
    **blast-pressure-threshold column in the material table** — damage only
    when local blast amplitude ≥ threshold (many small waves harmless, one
    big one bites; Erik's steel-resilience intent). Defaults reproduce
    today's behavior (excluded materials ≈ ∞ threshold → digest-safe);
    enables two glass types (brittle vs space-rated) as table rows.
    Implementation + tuning = chat-sized HUMAN-TEST rider AFTER residency.

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
- **Entity system + editor v3** — design LOCKED 2026-07-18
  (`entity_system_design_2026-07-18.md` canon model +
  `level_editor_v3_design_2026-07-18.md` view; both erratad as-built).
  **Arc A (entity foundation) DONE 2026-07-19** — A1–A9 merged to main,
  Erik blessed (doors v0 human-tested, A7 re-baseline blessed — and found
  EMPTY of committed artifacts). Canon:
  `architecture/engine/16_entity_system.md`; arc docs in `docs/archive/`.
  Build order (Erik): **next = physics close-out (stack #1 — S8a spec
  rewrite REQUIRED to carry the sensor-gather contract + the structural
  dirty-set rider, see stack #1)** → **Arc B** (SignalBus dataflow logic,
  sensors, pump, automatic airlock) → **Arc C** (editor UX panes, wand,
  wiring, play-from-editor, icons). AI tilesets (levels-w1 P6) still
  parked behind it. Arc riders on the books:
  - baker `[art]`/`[bake]` writeback → `level_lib` client (A2 accepted
    gap; fold in at Arc C).
  - `bake_demo` stays legacy-form until its committed baked art rebakes
    (migrating now would desync tilemap ↔ baked PNGs); shipped/showcase
    levels (`unhcr_vessel`, `playground`, …) migrate at Erik's choosing,
    likely Arc C.
  - `physics.py:104` blast-tuple wart → decide at physics close-out
    (listed under stack #1).
- **Sound-ML** — parked (`sound_ml_research_brief.md`), junior to the EOS arc.
- Beauty tracks: black-body emitter, smoke visuals, scorch/blood painting.

## Next chat-sized sessions (each its own chat)
1. Boundary-conditions spec (against landed rung-B; decide mode set + level
   format field + kernel touch points).
2. ~~Wall-burst differential fix~~ (DONE 2026-07-18) — the
   `burst_threshold` re-tune dial stays open.
3. EOS residency patch (if not already in flight).
4. W6 armory tuning (Erik) → weapons wave close ritual.
5. Dust-stirring shockwaves spec/impl.
6. Animation P0 — marine prototype, 3D-model-vs-part-sprites by eye.
