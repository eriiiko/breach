# Arc A — entity foundation: patch plan (2026-07-18)

**Status: AGREED with Erik 2026-07-18.** Executes the arc plan of
`entity_system_design_2026-07-18.md` §10 (LOCKED) under the autonomous patch
workflow: plan agreed once (this doc), patches run in fresh subagents,
orchestrator holds summaries only, memory checkpoint at every boundary.
Companion view doc: `level_editor_v3_design_2026-07-18.md` (Arc C).

**Branch:** `entity-arc-a`. Sequential execution on the dependency spine;
A8/A9 may run in parallel worktrees once A3 lands (one git-touching agent per
working tree).

## Erik's rulings (2026-07-18, at plan agreement)

1. **Doors v0 HUMAN-TEST driver:** a **debug toggle hotkey** — dev-only key
   toggles the door under the cursor. Render/dev layer, NOT synced state, so
   determinism/digests are untouched; it exists to feel close-over-occupancy,
   evacuation, and path-hold in play before Arc B gives doors real drivers.
2. **MAT_DOOR migration scope: test levels only.** Migrate the levels the
   golden/digest suites use; ONE deliberate re-baseline with written
   rationale. Shipped/showcase levels (unhcr_vessel, …) stay legacy until
   Erik chooses (likely Arc C, when the editor handles entities).
3. **Merge cadence: two tranches.** Mechanical patches (A1–A5, A8, A9)
   auto-merge to main as each goes green (standing authorization). Doors v0 +
   migration (A6–A7) land only after Erik's HUMAN-TEST and re-baseline
   blessing.
4. **(2026-07-19) §7-contradiction ruling: option (c)** — conservative
   seal/unseal pair for door flips (exact N), `destroy_wall` stays minting
   for destruction (Erik's rubble model: debris displaces volume, pressure
   constant — canon). Amendment folded into A5: on close, the door tile's
   temperature is set to the mean of its SOLID neighbors (panel belongs to
   the wall assembly), falling back to local air T if none — no instant
   hot-door from post-grenade air. HUMAN-TEST rider for A6: judge whether
   instant whole-span opening produces odd-looking pressure transients
   (k+1 rarefaction, one tick); if it reads badly, options are staged
   tile-per-tick opening (entity-level, primitive unchanged) or rebuild to
   (d) delete+mint (documented fallback in a5 doc). Canon errata (§7 L241 +
   eos §2.2) due at arc close.

## Patch list

| # | Patch | Mode | Gate |
|---|-------|------|------|
| A1 | **Entity core** — `src/simulation/entities/` package: `Entity` ABC, schema-in-code class declarations, registration decorator, `entities.toml` tuning overlay (dev-only, §2 hot-reload constraint), `entity_registry.json` export on successful launch + editor fallback data, import-light CI test. Ships one exemplar class (`light`) to prove the machinery. | subagent | `pytest tests -q` green → auto-merge |
| A2 | **`level_lib`** — THE single read+write data layer (entity doc §3c): `level_loader` read side and the editor's managed-block writeback become clients; atomic temp+rename writes; mtime+hash recorded at load/save; byte-stable round-trip tests over all existing levels. | subagent | round-trip + digest green → auto-merge |
| A3 | **`[[entity]]` format** — mandatory unique ids (file-order assignment, hard error on duplicates, warn on dangling refs), legacy `[[spawn]]`/`[[light]]` aliases, mixed-form hard error, `tags` field. | subagent | digest green → auto-merge |
| A4 | **Digest scaffolding** — `__entity__`/`__signals__` sections in `field_digest` + `get_state`/recorder; dormancy tests. **Impl-note first:** pin how sections fold in while entity-free digests stay byte-identical (no blanket `DIGEST_SPEC_VERSION` bump — existing goldens are NEVER re-baselined by this arc except A7). | subagent, impl-note gated | dormancy green → auto-merge |
| A5 | **EOS evacuation rule** — the seal/close half of `gamemap` (only the destroy direction exists today): closing evacuates tile gas to neighbors with exact N conservation; sealed-room door-cycle conservation test. **Design-gated:** short impl doc + adversarial pass (conservation + determinism lenses) — neighbor-distribution order, all-neighbors-solid case, GPU-resident-field interaction. | subagent, design-gated | N-conservation + dormancy green → auto-merge |
| A6 | **Doors v0** — door entity class; structural sweep (slot 9e, door half only — no logic until Arc B); flips applied in entity-id order via `on_tile_changed`; occupancy rule (living footprint blocks close) + water rule (`water_depth > 0` blocks close); path-hold (unit whose next tile turned impassable holds and burns the tick) + door-closes-across-path test; load order (entity tile state applied BEFORE field seeding, authored-open ≡ authored-air round-trip test); debug toggle hotkey (ruling 1). | subagent, design-gated | digest green + **HUMAN-TEST (Erik)** |
| A7 | **Migration tool + re-baseline** — one-shot in-place converter (`.bak`; groups painted MAT_DOOR runs into door entities); door-stamp-leak regression guard ported to full-solid CLOSED semantics; migrate test levels only (ruling 2); the arc's single deliberate golden re-baseline with written rationale. | subagent | **Erik blesses re-baseline** |
| A8 | **Zones** — `zones.npy` integer id grid + zone `[[entity]]` binding (`zone_id`, breach-site `roster`), validators (editor doc §5), `_upscale_level` replication. | subagent (parallel-ok) | digest green → auto-merge |
| A9 | **`air_init.npy` + `boundary` field** — format + loader seeding only; `boundary` semantics belong to the boundary-conditions physics project (ledger #1). | subagent (parallel-ok) | digest green → auto-merge |

Dependency spine: A1→A2→A3→A4→A5→A6→A7. A8/A9 need only A3.

## Standing constraints (bind every patch)

- Determinism iron rule: synced state is Q16.16 integer; no floats in the sim
  path. The debug door toggle is dev/render-layer and must stay out of synced
  state and digests.
- Dormancy guarantee: an entity-free level is bit-identical to today at every
  patch boundary; existing goldens re-baseline exactly once (A7).
- `pytest tests -q`, conda env `data`, never bare `python`.
- Stage explicit paths, never `git add -A`.
- Cross-arc contract: the S8a spec rewrite (between Arc A and Arc B) must
  include the sensor-gather contract (entity doc §7) — noted here so arc
  close-out reminds the ledger.

## Accepted gaps (documented per workflow, revisit at arc close)

- **A2:** the baker's `[art]`/`[bake]` writeback (`bake_level_art.
  write_bake_blocks`) is NOT yet a level_lib client — still its own
  non-atomic writer. Entity doc §3c says "all clients, period"; fold it in
  as a small rider in Arc C (editor arc) or at arc close. Ctrl+S re-records
  the handle's mtime+hash after baking, so staleness tracking stays honest.
