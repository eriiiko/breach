---
name: level-design-lead
description: Claude's standing role as lead level designer of Breach -- everything that makes a level: generation (the graph grammar, departments, cycle types, embedding, gates), hand-authored levels, the level format and its writer, pieces, vents and cells as level data, caverns and other archetypes, and the LLM-authored-rules experiment. Load at the start of ANY session that touches levelgen, a level file, level_lib / level_loader, gen_ship, the map editor's level output or milestones 11 and 12, before reading the code.
---

# Level design lead -- continuity skill

**Status: PROVISIONAL** (created 2026-10-31, the grammar session). Every
session that uses it amends it before merging: what moved, what was decided,
what was learned. A step is a MUST only with its incident beside it.

## 1. The mandate (Erik, 2026-10-31)

Erik appointed Claude **lead level designer**: "the role that's responsible for
all things level building -- generation etc." It is the levelgen twin of the
`rl-ready-lead` role and runs on the same terms:

- **Claude decides**: the shape of the generator (grammar method, data
  formats, the embedding pipeline, gates), what is a move and what is data,
  the issue cuts and their order inside milestones 11 and 12, which
  department is built next, when a department is "playing", what goes in
  this skill.
- **Erik decides, always**: anything judged by eye (how a ship FEELS to play
  or looks; HUMAN-TEST unchanged), game-design content (which departments
  exist, the dark list, secrets policy, the mirror policy when it changes),
  physics-modelling calls the signatures reach into (cold rooms, dust, a new
  gas row), budget (effort, Fable vs Opus agents), and the word choices that
  land in code for good (`department` vs `zone`).
- **Default**: when unsure whose call it is, make it, say so in the report,
  let Erik veto. Record rulings scoped to their case.

Erik's goal for the department (2026-10-29): "the better this part is done,
the more fun the game will be for a human player too, almost for free" and
for RL "one ship = memorised policy; variety = generalisation". He wants
"something special" (2026-10-31) out of a 1/3 m square-tile world.

## 2. What is special here, so nobody optimises the wrong thing

The special thing is **not** the grammar engine. It is **what the grammar
emits**: pressure cells, ducts with supply and return, fuel loads, an air mix
and temperature per room, hazards that are payloads waiting. No other game's
generator outputs physics. A department that only looks different is
wallpaper; a department whose AIR is different plays different (vocabulary
doc §0.2). Effort goes into departments, cycle types and their physics
signatures, and into the gates that prove a ship is fair and airtight. The
engine stays small.

## 3. Decided (with dates)

| Date | Decision |
|---|---|
| 2026-08-10 | The nine LOCKED decisions L1-L9 of `docs/breach_levelgen_design_v0.1.md`: graph grammar is the spine, space-only with archetype profiles, axiom per archetype, four planarity-safe moves, pressure cells by construction, phased application, offline authorship + deterministic seeded execution, grammar/validator division, the L9 output contract. Reopening one is an explicit decision, not drift. |
| 2026-10-29 | Levelgen v1 (`tools/gen_ship.py`, #120) is the light generator: mirrored BSP stage 1, `embed` + `validate` stage 2. The grammar REPLACES stage 1 and REUSES stage 2 and the gates. |
| 2026-10-29 | The research pass (#132, `docs/research/levelgen_litsearch_2026-10-29.md`) is read before any grammar design: hull as input, subdivision not search for ships, corridors are rooms, the no-triangle door-edge rule, the pressure cell as master boundary, supply + return per room, white-box LevelSpec and seed banks for RL. |
| 2026-10-29 | **Departments** (the vocabulary doc, Part B): a department is a recipe (rooms, adjacency by edge type, cell, ducts, physics signature, secrets), never a code path; eight blessed; mirror pairs + axis departments; the edge-type table §0.4; the sealed-room vent rule; a duct never crosses a cell. |
| 2026-10-31 | **Grammar method: family B** -- a FIXED MOVE LIBRARY in code (the L4 moves, each with precondition, phase, provenance) driven by DATA: department recipes, named **cycle types** (retreat loop, flank pair, airlock bypass, maintenance ring, ...) and an archetype **recipe** of (phase, move, count-range). The grammar runs on the quotient graph (one wing + the axis) so mirror symmetry holds by construction. Not a general LHS/RHS rewriting engine (family A), not constraint solving. Erik: "B is the right starting point, and perhaps even the best solution overall." |
| 2026-10-31 | **L7 amended** (Claude's call, Erik agreed): moves in code, recipes and departments in data. The LLM authors departments and recipes offline; it does not author moves. A bad authoring can produce a dull ship, never a broken generator. |
| 2026-10-31 | **Family A is milestone 12** ("LLM-authored grammar rules (family A)"): a general rewriting engine bolted onto the family-B grammar as an EXPERIMENT once that grammar plays. Erik: "the one thing that harnesses the power of LLMs in a special way"; not started, he does not want to widen current work. |
| 2026-10-31 | **Caverns are archetype three** (#145, parked, milestone 11): after the ship, before station fragments; chain layout + a CA realizer inside regions, same engine, different axiom / move subset / profile. Erik: "must not be forgotten." |
| 2026-10-31 | **Airlock behaviour**: an interior airlock pair (two pressurised cells) only sequences its doors and equalises the vestibule to the side being opened; only a HULL airlock (one side is space) evacuates. The generator picks the mode at generation time (on #133). |
| 2026-10-31 | **Part F rulings** (Erik, the vocabulary chat, morning): the word is `department` (`zone` stays the painted entity); departments as mirror pairs + axis departments, engineering the first axis department; secrets have their OWN budget (one mirrored pair per ship), separate from #135's; the garden dome gets LAMPS ("it's in space, no sun guaranteed"), a corridor-side window wall, not hull glass; cold rooms WAIT for a physics session; dust LATER; new slot-9e tick logic ACCEPTED for the heat trigger and the heat-source entity. |
| 2026-10-31 | **Stage-2 extensions landed** (#133 P1 `a0367ce`, P2 `9e5e69f`; E.2 heat trigger `e0d3832`, bulk-gas payload this merge): `EDGE_KINDS`, `ROOM_TAGS`, compartments from the graph, secret reachability, multi-duct vents, `vent_only` secrets (gated headless), entity doors (emitted OPEN: no bot opens doors, #142), airlock pairs (the demo recipe whole -- the interior-airlock ruling above is a FOLLOW-UP patch: today every generated airlock evacuates), `trigger = "heat"` charges, `hazard` room tag, bulk-gas payloads. Debts in the CLAUDE.md Level generators row. |
| 2026-10-31 | **Floor height** is level-format work, not v1 (#144): `GameMap.floor_height` and tilt already exist and the water solver reads them; no writer sets them. Not touched until Erik pulls it forward. |

## 4. Deliberately open (the session list)

- ~~The first cycle types~~ DONE 2026-10-31 (the cycle-types doc). Next session: the archetype recipe (phases, budgets, spawn exits, secret budget).
- From the vocabulary doc's Part F, still open: sub-ambient temperature (a
  physics session: cold storage, the morgue), dust as a gas row (Erik: later,
  after the first departments play), the high-O2 demo (now runnable:
  `[payloads.o2_store_rupture]` on a heat-triggered charge).
- The recipe file format (toml under `levels/recipes/` or beside the
  departments) and the LevelSpec the RL track reads (digest strand c, Q10).
- Station fragments (archetype four): the other embedding family, no hull.

## 5. Procedure for a session

1. `git fetch origin && git status`; read §3 and §7 of this skill, then the
   vocabulary doc's Part F and the latest comments on #131 / #133, then
   memory (`project_levelgen_milestone.md`, `project_levelgen_zones.md`).
2. Say which issue the session serves. One system at a time; the simple
   version first.
3. Design before code: a grammar change is a dated section in the design doc
   (or the vocabulary doc) FIRST, with its Systems section; departments are
   data rows; a stage-2 extension is a patch on `gen_ship.embed` /
   `validate`, never a parallel generator. Every level is written through
   `level_lib`; `ship_demo` regenerating byte-identical is the standing gate
   for a stage-2 patch that does not mean to move it.
4. Build per `autonomous-patch-workflow` and `git-workflow`: a worktree per
   branch, one writer, property tests (never pin a count of something
   designed to grow), digest gates.
5. How a ship plays or looks -> HUMAN-TEST, Erik plays before merge.
   Mechanical (gates, byte-identical regeneration) -> auto-merge on green.
6. Close: post the result on the issue, update memory, amend this skill.

## 6. Where things live

- Canon: `docs/breach_levelgen_design_v0.1.md` (L1-L9 + the dated grammar
  session block); vocabulary + departments:
  `docs/levelgen_zone_vocabulary_2026-10-29.md` (+ its critique); research:
  `docs/research/levelgen_litsearch_2026-10-29.md`.
- Code: `tools/gen_ship.py` (stage 1 `plan_graph`, stage 2 `embed`,
  `validate`, `compartments`, `EDGE_KINDS`, `ROOM_TAGS`), `src/level_lib.py`
  (the one writer), `src/level_loader.py`, `tools/level_airtight.py`,
  `tools/map_editor.py`.
- Planning: milestone 11 (umbrella #131; #27 vocabulary, #133 pieces, #134
  vents, #135 interactivity, #136 adjacency, #137 cells, #145 caverns),
  milestone 12 (family A), #144 floor height (parked). Roadmap #46 has the
  section.
- Sibling role: `rl-ready-lead` (the match, bots, replay). Levelgen serves
  it: a seeded, fair, varied ship is RL's generalisation lever.
- Memory: `project_levelgen_milestone.md`, `project_levelgen_zones.md`.

## 7. Run log

- 2026-10-31 (evening) -- #136 cycle types rev 2: the parallel chat's rev 1 (merged by it, `76f3987`) was critiqued (Opus, 20 findings: 3 blockers -- no loop in the worked example, no-triangle on the wrong graph, L5 broken silently) and folded here; L5 amended by Erik (breakable wall as third crossing); the LAYOUT stage (1.5) named as the largest unbuilt piece; `role` -> `fixture`; the duct link lost its direction. Learned: the other chat's 136 worktree stays its own (one writer), so the revision ran on `136-cycle-types-rev2` off main; a doc written with Erik at the screen needs the critique BEFORE its worked example is trusted (rev 1's example was infeasible at its own ranges). Open for Erik: the galley stove vs kindling call.
- 2026-10-31 -- grammar session (this skill created). Decided family B,
  L7 amended, milestone 12 created for family A, #144 floor height and #145
  caverns filed, the interior-airlock ruling posted on #133. Learned: the
  worktree inventory (`git worktree list`) carries ~30 finished worktrees;
  the §7 audit of `git-workflow` is owed before the next patch run.
- 2026-10-31 -- heat-source entity patch (#133 E.2 row 3, branch
  `133-heat-source`, one implementation agent). Followed §5 steps 3-4 as
  written; design came from the vocabulary row, no new doc section (a
  mechanism patch, not a grammar change). Learned: entities have no hp path,
  so the death hook is `GameMap.blast_log` fed by `execute_payload` (reuse it
  for the next blast-sensitive entity); a fresh worktree needs
  `cpp/build_cpu_home.bat` (~3 min) before any sim test; and a `pressure > 0`
  blast beside a thermal solid breaks the energy closure on its own (base
  8da03d6) -- closure tests that blow things up near fuel will fail for that
  reason, not theirs.
- 2026-10-31 -- initial-temperature level field (#133 E.2 row 4, branch
  `133-initial-temp`, one implementation agent). Followed §5 steps 3-4 as
  written; a format patch on the air_init.npy pattern, no new doc section.
  Learned: GameMap.__init__'s `refresh_gas_energy` + `seed_gas_temperature`
  on the seeded cells IS the sanctioned non-ambient start for gas (no new
  path needed); every shipped wall material is a thermal solid today; and a
  hot thermal solid seeded in AMBIENT air drives its neighbours sub-ambient
  on tick 1 and walls into the low rail -- seed a hot room whole (gen_ship's
  `temp` tag seeds floor air only).
- 2026-10-31 -- interior airlock mode (#133, Erik's ruling §3, branch
  `133-airlock-mode`, one implementation agent). Followed §5 steps 3-4; the
  ruling was the design (a dated section in the entity-system canon chapter
  §8, no new doc). Learned: the honest place for the mode is the controller
  (its legs already gate on sensed inputs), but an interior leg also needs a
  pump that MOVES air -- the pump grew a `feed = "transfer"` row and GameMap a
  conservative `transfer_gas_n`, one per side; v1's graphs only join rooms,
  so no generated airlock is hull yet (the hull branch waits for an edge onto
  space, the cargo-door row); adding a synced enum to an entity class moves
  every level's `__entity__` bytes that carries it (airlock_demo is in no
  golden, so nothing re-baselined), and the per-TILE pump quantum (1/3 atm at
  rate 8) is too coarse a tolerance for "equal pressure" -- test against the
  controller's band.
- 2026-10-31 -- cargo door in the hull (#133 E.2 row 5, branch
  `133-hull-door`, one implementation agent). Followed §5 steps 3-4; the
  vocabulary row was the design. Chose C(i) (a `hull_door` room tag; the hull
  airlock stays unreached: an airlock edge onto space needs a far-room-less
  airlock, more than a tag). Learned: a door to space is a ROOM TAG, not an
  edge kind (it joins a room to the void); `find_leaks` needed nothing --
  `door_closed`'s derived permeability 0 makes it solid; an opened span next
  to vacuum JOINS vacuum (`unseal_tiles`), so decompression is free and the
  re-closed room conserves N to the LSB; mirror twins must REFLECT the lower
  id's site (floor-centring a 3-wide door on a 10-tile run lands the twins a
  tile apart).
- 2026-10-31 -- cycle types + recipe format session (afternoon, design only,
  branch `136-cycle-types-recipes`): `docs/levelgen_cycle_types_and_recipes_2026-10-31.md`.
  Erik's rulings: walk and air loops are different and both are named; every
  walkable path >= 6 tiles, the flank pair's service leg may be 3; all six
  first types in (retreat loop, flank pair, airlock bypass, maintenance ring,
  duct link, pocket); the galley stove is ON from tick 0 (to be looked over
  once ships play); spawn exits 1-3 default 2 as an archetype parameter.
  Followed §5 steps 1-3 (no code). Learned: opening with "where we are" in
  four scales of lego pieces (edge kind / room role / cycle type /
  department / archetype) is what let Erik follow a fast-moving track --
  reuse that framing at the start of every levelgen session; one English
  word ("galley") cost a round trip -- gloss ship vocabulary the first time
  it appears. Next: critique pass on the doc, then the archetype recipe.
  Implementation handed to another chat; this worktree left in place for it.
