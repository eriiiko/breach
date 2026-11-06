---
name: level-design-lead
description: Claude's standing role as lead level designer of Breach -- everything that makes a level: generation (the graph grammar, departments, cycle types, embedding, gates), hand-authored levels, the level format and its writer, pieces, vents and cells as level data, caverns and other archetypes, and the LLM-authored-rules experiment. Load at the start of ANY session that touches levelgen, a level file, level_lib / level_loader, gen_ship, the map editor's level output or milestones 11 and 12, before reading the code.
---

# Level design lead -- continuity skill

**Status: PROVISIONAL** (created 2026-10-31, the grammar session; last amended 2026-11-06, LevelSpec + the first ships). Every
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
| 2026-11-06 | **Build-session rulings** (Erik: "All OK"): the layout patch may run on a Fable agent; mechanical, gated generator patches merge `--no-ff` to main on green without asking; the content calls the build makes stand by default and Erik judges the first ships by eye (the vetoable lists are on #131); generated level folders ship only after he has played them. |
| 2026-11-06 | **Props and vents are rung 2** (Erik): he is building many room props (furniture etc.) that did not exist when family B was designed; placing them, and a "vent material" that blocks walking but not air (his fix for air not travelling between the important rooms), are the generator's second rung. He handed the vent material to the lead ("feel free to invent the material ... entirely up to you"); planned as its own patch after the first ships (a `[materials.*]` row + an `EDGE_KINDS` row), tracked on #134. Rooms must keep variable sizes so recipes can change later (they do: every room is a range). |
| 2026-11-06 | **The grille** (the lead, on Erik's hand-off; #134): `[materials.grille]` blocks walking and passes air -- mobility 0, permeability 1.0, the gas thermal regime, so to the gas it is an air tile bit for bit (PLACEHOLDER values, Erik tunes); edge kinds `grille` (air only, the window's width) and `vented_door` (a door with a grille in each jamb, the same opening as `door`); neither ever crosses a pressure cell; the freighter's spine (slab, passageway, lobby, main sections) uses `vented_door`, departments keep their own air (their entries stay `door`). Vetoable: revert = `"door"` on the four spine lines of `ship_layout.FREIGHTER.edges`. |
| 2026-10-31 | **Floor height** is level-format work, not v1 (#144): `GameMap.floor_height` and tilt already exist and the water solver reads them; no writer sets them. Not touched until Erik pulls it forward. |

## 4. Deliberately open (the session list)

- ~~The first cycle types~~ DONE 2026-10-31 (the cycle-types doc). Next session: the archetype recipe (phases, budgets, spawn exits, secret budget).
- From the vocabulary doc's Part F, still open: sub-ambient temperature (a
  physics session: cold storage, the morgue), dust as a gas row (Erik: later,
  after the first departments play), the high-O2 demo (now runnable:
  `[payloads.o2_store_rupture]` on a heat-triggered charge).
- Station fragments (archetype four): the other embedding family, no hull.
- Prop placement from department recipes (rung 2; waits for Erik's props).
- Centring / shaping the lobby (a large empty hall today).

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

- 2026-11-06 -- THE GRILLE (#134, branch `134-grille`, one implementation agent): `[materials.grille]` (id 10, the gas thermal regime, mobility 0), `EDGE_KINDS` `grille` + `vented_door` (`EdgeKind.jambs`), `vented_door` in `WALK_KINDS`, `grille` in the recipe loader's `GAS_PASSING`, the freighter's spine vented. Followed §5 step 3 (the vocabulary §0.4 addition was committed before any code). Learned: the energy closure identity alone does NOT prove the row is air-like -- a permeable THERMAL-SOLID grille closed the books just as well (the engine books that path); the test that tells them apart is BIT-IDENTITY of every gas plane between a grille opening and an air opening, which only the gas-regime row passes -- write the strongest equivalence the design claims, then sabotage the row; the first divergence of the seed-28 bot match (tick 1057, outcome flipped from team 1 to team 0) came from SIGHT through a jamb grille (`has_los` stops on `solid`, and a grille is not solid), not from air -- a see-through grille is a gameplay change on every spine door, found by diffing `vision.compute` with grilles forced solid inside a `run_match` `on_tick` probe; two pre-existing tests pinned "air is THE only gas row" and the skeleton's literal `"door"` and were rewritten to the property (declared gas rows; the skeleton table). The 600-ship sweep vented-vs-plain (same g, a and rooms, 600/600, 259 s) is the cheap proof that a kind "changes no layout arithmetic". Skipped: the sealed-room vent rule for rooms that breathe through grilles (should read the AIR volume -- rooms joined by open / grille / vented_door edges -- not the room; left as is), grille art (it draws as a flat teal tile; `make_tileset`'s fallback hue covers baked levels). Cost: one agent run; the new tests add ~4 s.
- 2026-11-06 -- LEVELSPEC + the first two ships (#136, branch `136-levelspec`, one implementation agent): the `[levelgen]` record (`level_loader.LevelgenRecord` / `parse_levelgen`, `level_lib.format_levelgen_lines`, `gen_ship.levelgen_record` / `generate_from_spec` / `--from-level`), the centred spawn box (`gen_ship.centred_spawn_box`, recipe path only), the lamp budget. Followed §5 step 3 (the archetype doc §5.1 was the spec; deviations as a dated as-built section). Learned: the first look at a recipe ship IN THE GAME found what 600 headless ships could not -- `main.py` refused the first large freighter on frame 3 (the lamps passed the frame's cone-light door; the layout's `lamp_span` on every room gave 161 lamps), and every medium ship would have tripped the same door in a full volley; run `tools/e2e_drive.py` on a medium AND a large ship in any generator patch that changes what a level carries, not only `run_match`. The spawn box keeps its host's `_centre` (parity matched), so it moved no gate verdict and no tile -- a geometry fix that preserves the A* endpoints is free; the obvious "centred within one tile" test stayed green under a parity sabotage, a hand-built sweep over room sizes caught it. A field that records the BUILD (`generator_version`) must be left out of every "regenerates identically" gate. Skipped: banks, `mutate`, the #126 header's `recipe_hash`; a placement value (the box needed none). Cost: one agent run; generator test files ~35 s, the full suite ~9 min once at the end; a headless bot match 37 s (medium) / 85 s (large).
- 2026-11-06 -- the LAYOUT STAGE (#136, branch `136-ship-layout`, one implementation agent, M1-M4 in one run): `tools/ship_layout.py` gains the skeleton's geometry (`layout`, `pack_department`, the band, the ring, the hull strip, the reflection), the feasibility arithmetic (`layout_feasibility` exact on a graph, the loader's demand test, `department_feasibility` replacing the perimeter bound), `check_request` and `acceptance`; `gen_ship.generate(recipe=, profile=)` runs grammar -> layout -> `embed` -> `validate` -> the archetype's gates. Followed §5 step 3 (layout doc rev 2 + §13 were the spec; fifteen deviations as dated as-built lines). Learned: build feasibility and the layout as ONE enumeration of exact sets (an arrangement's x-extents and depths, rows as sum sets, the skeleton as ~1200 table lookups) -- the layout then never rejects on packing, the repair pass has nothing to repair and A4's conditioned draws fall out; the docs' arithmetic was all OVERFLOW, and the failure mode that decided the recipes was UNDERFILL (a lone department cannot fill a large row: three wing departments are infeasible; two bunks do not line a 30-long stub) -- so ranges are wide and counts small, never the other way round; rev 2's `behind` (a deeper column) costs 7 tiles of row length a medium ship never has, the end-zone corner costs none; voids that are each <= 3 ADD UP across a boundary (10 tiles of steel in the first pictures) -- look at a picture after the first working ship, before tuning anything; the first written-and-loaded recipe level found that no recipe ship could load at all (the `dark` preset wrote a zero-intensity lamp) -- write AND load one level in any generator patch's first end-to-end test; tests that patch a shipped recipe by a literal line break at every tuning (13 anchors moved) -- anchor on a field's NAME, or copy and edit the parsed table. Skipped: the wall-slide repair, LevelSpec / `[levelgen]`, banks, `mutate`, a hull of several rectangles, centring the squad in its slab (it stands at the fore end: a placement value or a spawn box, stage 2). Also learned: sabotage each rule once (13 one-line breaks, a script) -- two of the new tests stayed green and were rewritten on hand-built inputs. Measured: 600 of 600 ships (100 seeds x 2 sizes x 3 profiles), mean g <= 0.9, mean a <= 0.11, 0.14-0.34 s per ship. Cost: one long agent run; the full suite is 9 minutes on the work desktop, run after M1/M2 and at the end, the generator test files (35 s) in between.
- 2026-11-06 -- the ship GRAMMAR (#136, branch `136-ship-grammar`, one implementation agent): `tools/ship_grammar.py` (archetype door with the closure walk and per-(size, profile) effective ranges, the executor with `G_MAX` = 16 attempts on `SeedSequence([seed, 0, g])`, the seven moves as pure functions on a `Draft`, the §1.5 invariants by name, the `QuotientGraph` the layout consumes, `recipe_hash`), `tools/ship_layout.py` (the freighter skeleton's graph side + the ONE `layout_feasibility` call site, accepting everything until the layout patch), `levels/recipes/archetypes/freighter.toml`; layout doc §13 A7 (the grammar deals rows). Followed §5 step 3 (the three docs + the lead's ten decisions were the spec; deviations as dated as-built lines in the archetype doc). Learned: the docs' skeleton declares a TRIANGLE (lobby / aft-most main section / hub: door + airlock + bulkhead) that §1.5 refuses as written -- exempted only when every pair is a skeleton contact; a profile shift must not raise a [0, 0] range (hard would force a two-stub onto every medium ship) and must cap an exclusive group's minima together (else hard large rejects every attempt: it now always has the ring, never the bypass -- a feel item); a test whose expected ranges come from the function under test proves nothing -- sabotage each rule once and keep only tests that go red (four did not at first). Skipped: every packing check (the layout patch), department `[[cycle]]` rows (refused at load).
- 2026-11-06 -- recipe format 2.1 + the four v1 departments (#152, branch `152-recipe-format-2.1`, one implementation agent): `second_leg` (one-leg flank pair / retreat loop), `attach` side / end / behind and fore / aft for axis departments, `exit`, `hull_door`; the resolver now writes #133's fuel class and vent list (no longer pending) and the hull door; `engineering`, `cargo_hold`, `medbay`, `armory` as data; crew quarters made depth-elastic; the #172 ship_demo test compares a PNG by its pixels. Followed §5 step 3 (the doc rows were the spec; deviations written as dated as-built lines). Learned: the docs never said which contacts a declared edge may be -- reading "a side room touches only its stub" as a load rule (an edge joins a room to what it attaches to) made the triangle and contact rules hold by construction and caught nothing shipped; the perimeter feasibility bound passes ranges the layout will still reroll, so the depth span (24 at the minima, 47 at the maxima) is guarded by a separate property test until the layout's twin replaces both; worktree files are CRLF on this machine -- a Python `write_text` rewrite of a TOML turns it CRLF where git's attribute says LF, so edit through bytes or the Edit tool. Skipped: the loader twin (layout patch), any check that a heat source next door to fuel cannot light it (the engine's `max_temp` sits below kindling's by preset value only).
- 2026-11-06 -- the four family-B stage-2 extensions (#133, branch `133-stage2-family-b`, one implementation agent): breakable as a cell crossing + the airlock bypass as the ONE allowed multi-edge (`_check_bypass`; every other multi-edge now refused -- v1 never made one), the fuel class (`{clusters, mix}`; `fuel_materials()` = the flammable rows as the one contents set), the vent list (`vent_specs`, rules per duct), opt-in `lamp_span`. Followed §5 step 3 (patches on embed / validate / the tag schema, no new doc section; the design rows were the spec). Learned: §5.2's "contents = mobility > 0" was wrong against the table (wood is 0, doors 1000) -- read a doc's derivation against the live table before coding it; `diffuse.png` of ship_demo is NOT byte-reproducible on the work desktop (Pillow 10.4 and 12.1 both differ from the committed file, pixels equal), so the byte-identity test fails there before any change -- check the base first and compare pixels + the other three files; checking 72 `generate()` plans against the base module (imported from `git show HEAD:`) is a cheap, stronger "moves nothing" gate than ship_demo alone. Skipped: the resolver still marks fuel classes and vent lists pending (`levelgen_recipes.py`, the next patch).
- 2026-10-31 (night, Erik on a break) -- the two remaining family-B design docs, each written, critiqued by one Opus critic (20 and 18 findings) and folded to rev 2 on branch `136-layout-stage`: the LAYOUT STAGE (`levelgen_layout_stage_2026-10-31.md`: skeleton = slab / passageway / LOBBY / sections as code, seven guillotine primitives, bounded steel voids, axis parity, two rng streams, worked numbers: medium = 2 departments per wing, large = 4, small not a recipe ship) and the ARCHETYPE RECIPE (`levelgen_archetype_recipe_2026-10-31.md`: executor pseudo-code with a grammar attempt loop, quotient counts, named profile operators, `[validation]`, LevelSpec with every input + generator version, the level's own seed with match.toml's `seed` unchanged, banks, mutate). Seventh move `second_leg` added to the recipe doc and v0.1 §9; the scope doc's seed line amended. Neither rev 2 was re-critiqued (budget: one critic per doc). Learned: a doc written at the screen with Erik needs the critique before its worked example is trusted (twice now); 'two rng streams' has the consequence that an unrealisable grammar draw kills the seed unless the grammar has its own attempt loop; the research digest's seed-split sentence was wrong and was copied before being checked. Next: Erik's veto pass on the three docs' feel lists (layout §10, archetype §7), then cut the implementation issues (ship_grammar.py, ship_layout.py, the data files) once the loader branch lands.
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
- 2026-10-31 -- recipe loader + resolver (#136, branch `136-recipe-loader`,
  one implementation agent). Re-scoped mid-run: the brief was written
  against revision 1 and included a stopgap placement (`plan_graph_v2`) and a
  generated level; revision 2 landed on main during the run, the lead
  dropped placement and the level (the layout stage, §5.1, is undesigned),
  and the agent merged main and reworked. Built: the loader, the resolver
  (pending-never-dropped for unbuilt §5.2 mechanisms), the recipe files, the
  §3.4 presets, property tests; gen_ship untouched. Learned: brief the agent
  AFTER the critique folds, not before -- the first draft of the loader
  coded revision 1's `roles.toml`, leash and direction; and the §3.2 "wishes
  name library departments" rule refuses the worked example until those
  departments exist (amended: blessed names allowed).
