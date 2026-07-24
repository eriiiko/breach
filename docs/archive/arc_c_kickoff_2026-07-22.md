# Arc C kickoff — editor UX: panes, tools, undo, wiring, play-from-editor

> Ready-to-paste session prompt (written 2026-07-22, post-Arc-B merge
> `6909c01`). Arc C's soft gate — Arc B, for real wire endpoints — is MET.
> Paste the block below into a fresh session. Workflow:
> autonomous-patch-workflow; the design is ALREADY LOCKED (2026-07-18) —
> this arc is build-to-spec, with ONE in-arc design-gate (C3 undo).

---

You are building **Arc C — the editor UX layer** of the entity system on the
breach project. The design is LOCKED and errata'd as-built:
`docs/level_editor_v3_design_2026-07-18.md` (the authoring VIEW — §§4–8 are
your spec) + `docs/entity_system_design_2026-07-18.md` §10 (arc split) +
canon chapter `docs/architecture/engine/16_entity_system.md` (**read the
as-built §§1–8, not just the design docs** — Arc A/B errata live there:
`MAT_DOOR_CLOSED` id 7, the `[[wire]]` dotted format, tag pre-expansion,
`button`/terminal inert). Arcs A (foundation) and B (logic: SignalBus,
sensors, pump, airlock — B1–B7, merge `6909c01`) are merged and
Erik-blessed. The editor today is `tools/map_editor.py` (~1.7k lines) +
`tools/level_edit_common.py`; `src/level_lib.py` is THE data layer (canon
§3) — the editor is its client for every write.

**Scope (from the locked design; editor doc §8 + entity doc §10 + canon §9
forward pointers):**
- **Panes shell** — top bar / tool rail / canvas / tabbed palette /
  inspector / status bar (mode · cursor tile · validator summary · unsaved
  dot · registry-import banner slot). Keyboard-first survives the panes.
- **Registry-driven palette + inspector** — generated from the imported
  entity module; graceful fallback to last-good `entity_registry.json` +
  red banner on import failure (entity §3b). SPAWN/LIGHT modes honestly
  ported onto the entity model; the bespoke paths DELETED. `length_m`
  fields display meters, inspector shows the snapped tile result (§4
  quantization: integer `tiles_per_m`, exact Fraction, round-half-up,
  quantize-once-then-replicate).
- **Transaction-log undo** — a single log of compound operations (grid
  delta + entity delta per user action), REPLACING the per-domain rings.
  Every operation class joins it in the patch that introduces it:
  placement, moves, paints, zone paints, wires, tags, inspector edits.
- **Placement tools** (editor doc §6): DOOR tool (wall-run snap, default
  1.0 m, drag-resize, width warnings not errors; door-material tiles
  stamped to the grid IMMEDIATELY on placement — `MAT_DOOR_CLOSED`, per
  A6 errata — one undo transaction) · sensor placement (body tile +
  `sample_tile` offset rendered as a small arrow; refuse a solid sample
  tile at placement) · generic entity place-one from the palette.
- **Multi-select** (box + shift-click + select-by-class) + assign-tag-to-
  selection + **clump copy/paste** preserving internal wires (re-id on
  paste, external wires dropped) — the poor man's prefab.
- **Wand + zone/air paint** (editor doc §§5, 7): magic wand (enclosure
  fill + same-code select) for materials/AIR/vacuum/zones ·
  `air_init.npy` painting with the hull-leak validator (fill escaping to
  the border = warn, don't paint) · zone paint with the §5 binding
  validators (paint id ↔ exactly one instance; zero-tile warn; delete
  prompts to clear paint) · `boundary = "space" | "ambient"` in a
  level-properties pane.
- **Two-click wire tool** (click source → navigate freely → click target,
  Esc cancels; NEVER a drag) + LOGIC overlay defaulting to
  wires-touching-selection with show-all toggle; wires to `tag:` targets
  render to a tag badge, not fanned to members. Wires are the as-built
  `[[wire]]` format (canon §8): dotted `from = "id.signal"` /
  `to = "id.input" | "tag:name.input"`.
- **Play-from-editor (F5)** — save everything to
  `levels/_editor_scratch/<name>/` (gitignored), reuse baked PNGs when
  the grid is clean; launch `[sys.executable, "main.py", "--level",
  "_editor_scratch/<name>"]` (NEVER bare `python` — documented machine
  footgun); scratch deleted on subprocess exit and editor quit; loader
  accepts the `_editor_scratch/` path form.
- **Icons** — SVG sources `art/entities/icons/` + committed PNGs via
  `tools/rasterize_icons.py` (rasterizer dep is dev-only; a test asserts
  PNG freshness). No icon → generated color chip + class initial,
  permanent fallback, never an error.
- **Riders folded in (on the books for Arc C):** baker `[art]`/`[bake]`
  writeback ported onto `level_lib` (A2 accepted gap; atomic multi-family
  replace) · `MAT_DOOR_CLOSED`-outside-a-span validator warning (canon
  §9).

**Erik's rulings (2026-07-22, kickoff session):**
1. **HUMAN-TEST cadence: ONE gate, at the end.** Patches auto-proceed on
   green unit tests within the arc branch; NO per-patch Erik drives
   (relaxes entity §10's "per patch" — Erik's explicit call). The arc
   merges to main only after the end-of-arc acceptance drive.
2. **Legacy migrations are OUT of Arc C.** `unhcr_vessel`,
   `unhcr_vessel_2`, `playground` will NOT be migrated — Erik is
   retiring that art direction; the replacement is a NEW level authored
   in this editor (the acceptance drive). `bake_demo` stays legacy until
   its art rebakes. Do not touch shipped levels' files at all.
3. Machine: **Lenovo**.
4. Arc B merged; strictly sequential — no concurrency constraint from B.

**Patch plan (each patch = fresh subagent; auto-proceed on green):**
- **C0 — recon + shell skeleton:** map the current `map_editor.py` modes/
  input/draw loop; land the panes shell with existing tools rehosted,
  zero behavior change. (Recon writes a short as-is note into the arc
  impl doc first — the orchestrator never reads the 1.7k-line file.)
- **C1 — registry-driven palette + inspector;** SPAWN/LIGHT ported onto
  `[[entity]]`, bespoke paths deleted; registry-import banner + fallback.
- **C2 — ★ DESIGN-GATE: transaction-log undo.** Write the JIT impl doc
  (op/transaction model, capture points, memory bounds, redo, interaction
  with level_lib dirty-tracking), run an independent adversarial critique
  (lenses: data-loss/corruption, compound-op atomicity, scope/regression),
  resolve blockers on paper, THEN build. Existing op classes join here;
  later patches each register their ops.
- **C3 — placement tools:** DOOR (span drag + immediate stamp) · sensor
  (sample-tile arrow + solid refusal) · generic place-one.
- **C4 — multi-select + tags + clump copy/paste** (re-id on paste,
  internal wires kept, external dropped).
- **C5 — wand + zone/air/vacuum paint + hull-leak validator +
  level-properties pane (boundary field).**
- **C6 — wire tool + LOGIC overlay + tag badges.**
- **C7 — play-from-editor (F5)** + `_editor_scratch/` loader form +
  gitignore.
- **C8 — icons pipeline** (SVGs, rasterizer, freshness test, chip
  fallback).
- **C9 — riders:** baker writeback → level_lib client ·
  `MAT_DOOR_CLOSED`-outside-span validator warning.
- **C10 — ACCEPTANCE (HUMAN-TEST, the one gate):** Erik drives the
  editor end-to-end and authors a NEW level (a Contested Chip slice:
  rooms, doors, sensors, wires, an airlock clump, zones, air paint,
  F5-play). Merge to main only after Erik blesses.

**Environment (Lenovo):** python =
`C:/Users/steen/miniconda3/envs/data/python.exe` (NOT `conda run`, NOT
bare `python`); pytest = `pytest tests -q`; no C++/CUDA builds expected
(editor is pure Python) — if a merge from main brings cpp changes,
rebuild via `cpp/build_cpu_data.bat`. Work in a dedicated worktree +
branch (`arc-c-editor`); one git-touching agent per tree; commit the arc
impl doc to the branch BEFORE spawning worktree subagents. Never
`git add -A`.

**Gates (every patch):** unit tests for the new surface + the FULL suite
green (`pytest tests -q`) + level_lib byte-stable round-trip tests stay
green + **zero goldens/digests touched** — the editor is not sim path; if
a patch thinks it needs a re-baseline, that's escalation trigger 1, not a
gate to negotiate. Editor writes to real level files only through
level_lib atomic writeback.

**Concurrency constraint (fire/smoke beauty track may be in flight, same
repo):** Arc C must not touch `src/render*`/renderer modules,
`src/simulation/`, `physics_runner.py`, or `cpp/` — its surface is
`tools/`, `src/level_lib.py` (client-side additions only), the loader's
`_editor_scratch/` path acceptance, `art/entities/icons/`, `tests/`. If a
scope need crosses that line, stop and surface it.

**Escalation triggers — STOP and bring Erik (or a Fable design review) in
if the work finds it must do any of:**
1. Change any frozen format or contract: `[[entity]]`/`[[wire]]`/zone
   binding semantics, level_lib API meaning, digest surfaces, goldens.
2. Touch sim behavior (anything under `src/simulation/` beyond read-only
   imports) or the CUDA/resident path.
3. The C2 undo design fails its critique twice — the model has a hole,
   not the critique.
4. Registry import/fallback semantics (entity §3b) need to change.
5. Any DELETE of a bespoke path can't reach behavior-parity first
   (SPAWN/LIGHT ports are "honest" — same authored result, via entities).

---

## Model verdict (Fable, 2026-07-22, pre-handoff session)

**Run this with Opus** driving the full autonomous-patch-workflow.
Rationale: the design is LOCKED (2026-07-18, critique folded) and the
model/format layers it builds on are canon as-built (engine/16). This arc
is UI plumbing over frozen contracts — mechanical oracles everywhere
except C2 (which carries its own in-arc design-gate) and the single
end-of-arc HUMAN-TEST (Erik's cadence ruling above). Fable re-enters only
on the escalation triggers.
