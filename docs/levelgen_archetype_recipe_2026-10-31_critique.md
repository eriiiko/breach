# Critique: levelgen archetype recipe, revision 1 (2026-10-31)

> Adversarial critique of `docs/levelgen_archetype_recipe_2026-10-31.md`
> (worktree `136-layout-stage`, HEAD `933f9d4`), 2026-10-31, Opus, read-only.
> Checked against the recipe doc rev 2.1, the layout doc rev 2, both earlier
> critiques (their findings are not repeated, only flagged where this doc
> reintroduces one), design v0.1 (L2/L3/L5/L6/L7/L8, §9), the litsearch
> strands (a) and (c) and Q5/Q10, the vocabulary doc Part B/C/F,
> `tools/gen_ship.py`, `level_lib.py`, `level_loader.py`,
> `src/simulation/match.py`, `src/simulation/simulation.py`,
> `src/match_replay.py`, `tools/run_match.py`, the `rl-ready-lead` skill and
> `docs/rl_ready_scope_2026-10-29.md`. Numbers are recomputed from `SIZES`
> and the layout doc's §9 table; nothing was run.

Line references `A:n` are to the archetype doc, `R §x` the recipe doc rev 2.1,
`L §x` the layout doc rev 2.

## Findings

### 1. BLOCKER -- one rng per attempt contradicts the layout's two streams, and "reject the draw" has no meaning once the grammar is drawn once

> "One `rng` (`SeedSequence([level_seed, attempt])`, as `generate` does today). ... a count that cannot be met at its minimum rejects the DRAW (reroll under `MAX_ATTEMPTS`, L8)." (A:159-168, §2)

**Evidence.** L §11 (rev 2, folding layout critique 14): "the GRAMMAR draws once from `SeedSequence([seed, 0])` (so the resolved graph is the (recipe hash, seed) function the recipe doc promises); layout + `embed` draw from `SeedSequence([seed, 1, attempt])`". The archetype doc puts the grammar back on the per-attempt stream, which is exactly what layout critique 14 removed (and it contradicts R §3.4's "(recipe hash, seed) reproduces the resolved graph"). Under the layout's scheme, the second half of the sentence is undefined: the grammar never rerolls, so a count missing its minimum has no attempt to fall to. Worse, the two-stream design has an unstated consequence this document must own: **any grammar draw the layout cannot realise kills the seed on all `MAX_ATTEMPTS` (32, gen_ship.py:113)**, because every layout attempt sees the same graph. Findings 2 and 4 show such draws are common in the freighter as written. Nobody has estimated the per-seed failure rate, while the RL-ready scope's levelgen gate is "100 seeds, all airtight, all connected, all mirrored" (rl_ready_scope line 84).

**Fix.** Copy L §11's two streams into §2 verbatim. Add a GRAMMAR attempt loop as a named design choice: grammar attempt `g` draws from `SeedSequence([seed, 0, g])`; a count below its minimum, or a layout that fails all its attempts on that graph, moves to `g + 1` (bounded, e.g. 8), and layout attempts draw from `[seed, 1, g, a]`. Record `(g, a)` in the LevelSpec factors and `ShipPlan`. This keeps "never re-enter the grammar" in its real sense (no repair or edit of a drawn graph) and keeps (recipe hash, seed) reproducible. Add a costed gate: the recipe path passes N seeds per (size, profile) with the failure rate reported.

### 2. BLOCKER -- the freighter draft cannot be built on medium, and §5's check as written would not notice

> "Recipe ships are MEDIUM (two departments per wing) and LARGE (four)" (A:232); `cargo_hold medium = [1, 1]`, `medbay medium = [1, 1]`, armory / garden / pool `medium = [0, 1]` (A:60-83); `maintenance_ring ... the loader refuses it on medium` with `count = { medium = [0, 1], ... }` (A:104-108); `band = { sections = [3, 5] } # within the preset's range` (A:39); §5(1) "against the preset's smallest wing length" (A:214-216)

**Evidence.**
- Department count. Crew quarters (axiom), cargo, medbay, armory and pool are mirror pairs (vocabulary B.1, B.3, B.4, B.5, B.7); the garden is "mirror pair, or axis" (B.6). The medium minimum is therefore **3 wing departments per wing** (quarters + cargo + medbay) and the maximum 6, against a medium capacity of **2** (L §9: row length 16-21, one-sided x-demand 13-15, "one department per row"). This contradicts §6's own sentence.
- Retreat loop on medium. A two-stub needs `2·across + 2·depth + gap + 4` (L §4.1) >= `12 + 12 + 0 + 4 = 28` even with no gap, > the medium row length 16-21. Any medium grammar draw that applies it is unrealisable (and kills the seed, finding 1).
- Ring. The comment says the loader refuses the ring on medium; §5(3) refuses the WHOLE RECIPE on a size where the ring "may be applied" and cannot be afforded. With `medium = [0, 1]` it may be, so the freighter is refused on medium by its own loader.
- Band. `SIZES["medium"].sections = (3, 4)` (gen_ship.py:192); `[3, 5]` is outside it, and §1.1 says the loader checks "within the preset's range". (`passageway` and `lobby` have no preset range in `SIZES` at all, so for them the check is empty.)
- §5(1) compares against the "wing length" (38-39 on medium, L §9). Quarters + medbay in row A at minimum one-sided demand (13 + 13 + 1 void = 27) pass that test and fail the real one, row length 16-21 (L §6). The layout doc already specifies the right check, its "loader twin" (L §4: two-bin packing at the minima; L §11 "the loader imports the layout's arithmetic, never a copy"); §5 restates a weaker one.

**Fix.** Medium counts that fit two per wing (e.g. quarters + cargo `[1, 1]`, everything else `[0, 0]`, retreat loop and ring `[0, 0]`); which second department a medium freighter carries is Erik's content call (§6). Make the band sections per size or "intersected with the preset", and say which. Replace §5(1)-(3) with "the archetype loader calls the layout's loader twin per allowed size", including depth and the phase-2 entries at their maximum (the hull strip takes 4 rows of depth from row A, a two-stub widens a department), and run it per profile (finding 7).

### 3. MAJOR -- two `on` rows name an edge the move cannot rewrite: the flank pair and the retreat loop are a seventh move

> "`slab-lobby` | the wing's two hubs | passageway + hull strip (flank pair)"; "`department-entry` | the entry edge of a wing department ... | two-stub (retreat loop)" (A:177-178); R §0.1 `edge_to_cycle`: "A-B becomes A-C-B plus A-D-B; **A-B is removed**"

**Evidence.** The skeleton's declared edges are slab-passageway, passageway-lobby, lobby-section and lobby-engine room (L §2). There is **no slab-lobby edge** for `edge_to_cycle` to remove. The wide leg is the EXISTING passageway (L §4.2), so the application creates one leg, the strip, between two nodes already joined by a two-edge path. The retreat loop is the same: stub_1 is the authored `entry` and END the authored mess; only stub_2 is new, and the cycle's anchors are passageway and END, not passageway and stub (L §4.1). "Add a one-room leg between two nodes already at distance 2" is neither `edge_to_cycle` nor any other row of R §0.1, and the recipe doc's Systems rule says "never a seventh move without a design line". It also leaves the cycle files' wide-leg fields (`rooms = [1, 3]`, `edges = ["door", "open"]`, R §2; "two `wide` legs of 1-2 rooms each" for the retreat loop) with nothing to draw. Separately, §1.1 calls every `on` value an "edge class", but `any-room`, `<department>.<room>` and `same-cell-rooms` are room classes.

**Fix.** Name the operation as a move with a precondition ("`second_leg`: A-C-B exists; add A-D-B with D new; the existing leg is not redrawn"), add it to R §0.1 and v0.1 §9's count (a design line, as the rule demands), and point the two cycle files at it with ONE leg each. Rename the column "anchor class".

### 4. MAJOR -- `any-room` and the bypass/ring order hand the layout graphs it cannot realise

> "`any-room` | any room with a free side | satellite (pocket)" (A:181); phase-2 file order: `retreat_loop`, `flank_pair`, `airlock_bypass` on `lobby-engineering`, then `maintenance_ring` on `engineering.engine_room` (A:86-114)

**Evidence.**
- "Free side" is geometry; the grammar runs before the layout and has none. The candidate set as worded includes the axiom rooms (slab, passageway, lobby, band sections) and ring rooms, but the layout realises a pocket only as "a `behind` room or a packed side room" of a DEPARTMENT (L §4.5). A pocket on the lobby or a section is a `_Reject` on every layout attempt, so (finding 1) a dead seed. A pocket off a one-sided medium department that already fills its stub's `along` exactly (L §4 step 3) is the same.
- The bypass is applied first, as a `parallel` opening on the lobby-engine room airlock edge. The ring, applied after it, "rewrites the lobby-hub edge to lobby-r_side" (L §4.4). Nothing says what happens to the bypass's breakable: it is stranded on a removed edge, or it moves to the ring side room, whose height is about half the hub's when the side column is split in two, against the bypass's 13-tile wall demand (L §4.3). Two implementers will differ, and one of them produces unrealisable graphs.

**Fix.** Define `any-room` as "a department room that the layout can attach to (`side` or `behind`), never an axiom or ring room", with the precondition stated in graph terms (the department's loader-twin slack at the room's minimum). State the bypass/ring rule: either the ring precedes the bypass and the bypass's anchor becomes `lobby-r_side` with the 13 checked on the side room, or the two are mutually exclusive per ship. File order must not silently decide it.

### 5. MAJOR -- the executor is under-specified: two implementers will build different ships from the same file

> "the count first, then each application's choices (which edge of the class, which rooms, which leg kinds) in the layout's canonical order of candidates (rooms by id, edges by `(a, b)`) ... a failing application is REJECTED (skipped, logged in provenance) and the count continues with the next candidate" (A:161-168)

**Evidence.**
- How is the count drawn? Uniform over `[lo, hi]` is the obvious reading, but nothing says so.
- "Draw ... in canonical order" and "continue with the next candidate" are two different algorithms: a uniform draw from the sorted candidate list, or a deterministic walk down it. The second makes the first candidate win every time.
- With or without replacement: can a second `retreat_loop` (large `[1, 2]`) hit the department that already has one? Can a second pocket attach to the same room? In the draft, quarters is the only department with an `attach = "end"` room, so the candidate set holds one entry per wing.
- Does a rejected application consume one of the drawn count? If 2 are drawn and 1 succeeds with `lo = 1`, is that a silent success? (The factors record what was realised, so the answer matters to the RL track.)
- "Rooms by id": department room ids are local strings (`"bunk"`, R §3.1), repeated across `count` instances, across departments drawn twice (cargo large `[1, 2]`) and across wings. There is no global id before the grammar assigns one, and the doc does not say how it does.
- "After every application the invariants of the recipe doc §1.5 run": R §5.3 runs a per-phase SUBSET (reachability only from phase 3, the air invariants in phase 4). Reachability from both spawns cannot hold in phase 1, before the spawn exits of phase 2.

**Fix.** Write the executor as pseudo-code in §2, e.g. `k = rng.integers(lo, hi + 1)`; `cand = sorted(candidates, key=global_id)`; `while placed < k and cand: c = cand.pop(rng.integers(len(cand))); apply-or-reject(c)`; `if placed < lo: grammar reject` (finding 1). Fix the global id as a tuple `(department, instance, room id, wing)` and the sort key. Point at R §5.3's per-phase invariant table instead of "§1.5 all".

### 6. MAJOR -- per wing or per ship? Mirror counting is stated for one row of eleven

> `count = { medium = [1, 1], large = [1, 2] }   # per wing (mirrored)` (A:63, the cargo row only); "The mirror is never drawn: every wing application is applied to the twin." (A:163-164); `secrets = { budget = 1 }  # mirrored pairs per ship` (A:42)

**Evidence.** v0.1 §9 runs the grammar on the QUOTIENT (one wing + the axis band), so "the twin" does not exist during the grammar; it exists after the layout reflects the half (L §1). Only the cargo row says "per wing". Axis departments (engineering; the garden may be "mirror pair, or axis", B.6) exist once per ship. For `same-cell-rooms`, the main cell is "both wings + the spine band" (R §3.2 `cell`): a link between a left-wing room and a right-wing room has as its twin ANOTHER pair (two links, two ducts), and a link between a wing room and an axis section has the section as its own twin (one section on two links). The secret budget counts "mirrored pairs", but engineering carries a secret (the still, 0.2, B.2) on an axis host, which is one self-twin room, not a pair.

**Fix.** One sentence in §2: "every count is a QUOTIENT count: a wing application appears once per wing, an axis one once per ship". Restrict `same-cell-rooms` candidates to one quotient wing, or to the axis band (self-twin links). Say that an axis secret consumes one unit of the budget.

### 7. MAJOR -- profiles: four operators under the one word "shift", unrealisable minima under `hard`, feasibility never run per profile

> "difficulty profiles: shifts applied to every count range" (A:125); `cycles = -1 # ... floored at 0`; `secrets = 0`; `zombies = 0.5 # multiplier`; `own_cells = [...]`; field ref "`cycles` an int, `secrets` an int, `zombies` a float > 0, `own_cells` a superset of `cells.own`" (A:126-136, 153)

**Evidence.**
- `cycles` is additive, `zombies` multiplicative, `own_cells` a replacement. `secrets = 0` is ambiguous: as a SHIFT (which is what the heading says) easy keeps the one secret pair, against §6's "one mirrored pair per ship on every profile but easy". As an absolute, it is a fourth operator.
- `hard` (+1) gives `maintenance_ring` a minimum of 1 on medium, where the ring is impossible (L §9). It gives `flank_pair` and `airlock_bypass` maxima of 2, where the skeleton realises at most one of each per wing (one hull strip, on row A, L §4.2; one lobby-engineering wall, L §4.3). Whether the shift also hits the phase-1 pocket and the phase-4 `duct_link` ("every cycle-type range") is unclear. Department counts are not shifted, but only §6 prose says so.
- §5's feasibility runs on "the file as written", so the `hard` minima and the `own_cells` stub constraint (finding 8) are never checked at load.

**Fix.** Give each profile key its operator in §1.1 (`cycles_shift`, `secrets` absolute, `zombies_scale`, `own_cells` absolute set) and say which phases the shift covers. Cap every shifted range at a per-cycle realisable maximum the layout publishes (one ring per hub, one strip per wing, one bypass per lobby wall). Run the loader-twin feasibility per (size, profile).

### 8. MAJOR -- cells: two sources of truth, a check with nothing to read, and a room-granularity comment on a department list

> "`cells.own` | each a department whose recipe allows `cell = "own"`" (A:149); `own_cells = ["engineering", "armory"]   # the armory's magazine becomes its own cell` (A:135); §6 "The hard profile makes the armory its own cell (an airlock fight for the guns)"

**Evidence.**
- `department.cell` is ONE value, `main | own` (R §3.1, §3.2). It is a declaration, not a permission, so "allows" has nothing to read. Engineering is declared `own` in its file AND listed in `cells.own`: when the two disagree, which wins?
- The comment speaks of the magazine (a ROOM), the key lists departments. Vocabulary B.5: "a magazine of its own cell is an archetype for later; in v1 its doors are plain". The hard profile reopens that line without saying so.
- Promoting a wing department to `own` turns its entry (stub-passageway, an edge the grammar declares) into an airlock pair. That needs stub `across >= 7` (L §4), and nothing says who rewrites the edge kind. The bots' A* does not route through an airlock's closed outer door (#142, R §1.3 accepted gap), so on `hard` the armory is unreachable to bots. That is a feel consequence for Erik's veto, not listed in §6.
- Medbay's isolation room is "a cell of its own inside it" (B.3). The resolver numbers cells per DEPARTMENT ("`main` = 0, each `own` department the next int", R §3.4), so it has no slot for a room-level cell. The archetype's `[cells]` table claims cell authority it cannot express.
- L5's per-archetype vent leak policy has no field. R §1.3 names "the archetype flag `allow_cross_cell_leak`", and the format omits it.

**Fix.** The department file's `cell` is the default. The archetype's `cells.own` / a profile's `own_cells` PROMOTES a `main` department and is refused for a department whose stub cannot take an airlock. The grammar rewrites that department's entry edge to `airlock` in phase 1 and logs it in provenance. Drop the magazine comment, or make it a room-level cell (a new key) and mark it as reopening B.5. Add `allow_cross_cell_leak = false` (refused `true` until #49). List the bot-unreachable armory under §6.

### 9. MAJOR -- the zombie multiplier is wrong about the code: the level fixes the zombie count's ceiling

> `zombies = 0.5  # multiplier on the match's zombie count (the match config's number is the base)` (A:129)

**Evidence.** Today the zombie count is a GENERATOR input: `generate(..., zombies=4)` (gen_ship.py:1950-1951), checked for capacity by `check_request` (gen_ship.py:932-939), stored on `ShipGraph.zombies`, and written as the ZOMBIE_TEAM `[[spawn_room]]`'s `max_per_team = graph.zombies` (gen_ship.py:1596-1599). `match.toml`'s `zombies` is then written FROM the plan (`ShipPlan.match_spec`, gen_ship.py:1938-1943). `parse_match` refuses `zombies > zroom.max_per_team` (match.py:289-297), and `write_match` re-parses through that door before it writes (level_lib.py:449-466). So ×1.5 on "the match's number" is refused at the writer, and ×0.5 of an odd base needs a rounding rule. The profile is a LEVEL property (it is in the LevelSpec), while the multiplied number is a MATCH property: the doc does not say which file the result lands in.

**Fix.** The profile scales the GENERATOR's `zombies` argument (say round half up, at least 1 when the base is > 0). The recipe path's `check_request` (L §11) checks the scaled count against the zombie section, and the scaled count is written into both the spawn room and `match.toml`. Or take zombies out of the difficulty profile and leave them to the match config. Either way the LevelSpec records the base and the result.

### 10. BLOCKER (RL contract) -- the seed split cannot do what §4 says it is for, and it does not fit the code it names

> "the match seed spawns `level_seed` and `episode_seed` through `SeedSequence.spawn`; ... spawns, rosters and controllers draw from the episode seed. A training run holds `level_seed` and varies the episode ... Both are recorded in `match.toml`." (A:195-199); Systems (a): "`match.toml` gains the two derived seeds through `level_lib.write_match`" (A:265-267)

**Evidence.**
- **Self-contradiction.** If both seeds are functions of one match seed, a training run cannot hold `level_seed` while varying `episode_seed`: changing the episode means changing the match seed, which changes the level. (The research digest has the same slip, strand c / Q10; it was copied, not checked.)
- **How seeds flow today.** The level is generated OFFLINE into a folder (`gen_ship --seed`), and the folder's `match.toml` gets `seed = plan.seed` (gen_ship.py:1941). `run_match(level, spec, seed=)` loads a written folder and its `seed` OVERRIDES the config's (run_match.py:84-114) into `Simulation(seed=)` -> `np.random.default_rng(seed)` (simulation.py:205, 279). So "fixed level, varying episode" is already `run_match --seed N` on one folder. No match-time path generates a level.
- **The door refuses it.** `parse_match` rejects any key outside `_MATCH_KEYS = (objective, seed, time_limit_s, zombies, zombie_controller, team)` (match.py:57, 218-221). The two new seeds are a match-format change the doc does not cost.
- **Type and digests.** `SeedSequence.spawn` returns SeedSequences, not ints. `Simulation(seed: Optional[int])` and the #126 recording header store an int (`match_replay.py:209, 447`). Seeding the sim with a DERIVED episode seed instead of `match.toml`'s `seed` changes every pinned match digest (ship_demo seed 1 tick_digest `4d32170a`, #123) and invalidates every recording made so far (`check_header` compares the seed).
- **Wrong about what draws.** "Spawns, rosters ... draw from the episode seed": `spawn_match` "Draws no randomness" (match.py:313-317), and rosters are fixed slots (rl-ready-lead §5 item 2). Only `sim.rng` consumers (bots, combat, zombies) draw.
- **An unflagged amendment.** The RL-ready scope's decided line is "One seed drives levelgen, spawn, bots and zombies" (rl_ready_scope line 34). The doc changes it without saying so.

**Fix.** Write the contract the code already almost has: `level_seed` is the GENERATOR's seed, recorded in `level.toml [levelgen]`. `match.toml`'s existing `seed` IS the episode seed (no new key, no re-pin, recordings stay valid). A root seed spawning both (`SeedSequence(root).spawn(2)`, integerised with `generate_state(1)[0]`) is an optional convenience for a loop that generates a level per episode, and records both ints. Amend the scope-doc line explicitly ("one seed per match; the level has its own"), and say that #76's patch, not this arc, owns any `match.toml` change.

### 11. MAJOR -- "a ship is a pure function of (archetype, size, profile, level seed)" is false; LevelSpec and `recipe_hash` miss inputs

> "A ship is a pure function of (archetype file, size preset, profile, level seed)." (A:25-26); "`LevelSpec = (archetype, recipe_hash, size, profile, level_seed)` ... `recipe_hash` hashes the archetype file, every department and cycle file it names, `fixtures.toml` and the RESOLVED preset values" (A:188-194)

**Evidence.**
- Missing generator inputs: `max_per_team` sets the slab's x-extent through `_spawn_grid`/`_extent` (gen_ship.py:900-911; L §9's numbers assume 6), the row lengths and so feasibility; `zombies` sets the zombie section's tag and capacity (finding 9); `vents` is a `generate` argument today.
- Missing code version: moves, layout constants (`VOID_MAX`, `REPAIR_SLIDE`), `SIZES` and `MAX_ATTEMPTS` all change the ship for the same spec. The "generator version" appears only on banks (A:200-201), not in the spec. The #126 recording already carries `"build": "<git describe>"` (match_replay.py:36) as a precedent.
- `recipe_hash` is under-defined: bytes or parsed-canonical TOML (does a comment edit re-baseline every bank)? Is it transitive (department `[[cycle]]` files, fixtures, departments named only by a profile's `own_cells`)? Does "resolved preset values" include `SIZES`? Every profile lives in the archetype file, so tuning `hard` changes the hash of every `easy` ship. Say so, or hash the profile separately.
- Folder identity: `default_name(seed, size)` = `ship_s{seed}_{size}` (gen_ship.py:1946-1947) collides across archetypes and profiles. Recordings identify a level by FOLDER (`match_replay.level_id`, #126 run log), so a regenerated folder with another recipe hash replays against a different ship and is caught only as a late checkpoint divergence.

**Fix.** `LevelSpec = (archetype, recipe_hash, generator_version, size, profile, max_per_team, zombies_base, level_seed)`, plus the realised `(g, a)` attempt indices (finding 1). Define the hash over parsed, key-sorted TOML of the transitive file closure plus the resolved preset values; `SIZES` and the constants go under `generator_version`. Recipe folders get a name carrying archetype and profile, and the recording header carries `recipe_hash`.

### 12. MAJOR -- L2's third component, the validation profile, is missing; "difficulty profiles" took its place

> "v0.1 L2: **a map archetype = axiom + rule subset + validation profile.** In family B that is one file: the SKELETON ..., the AXIOM ..., an ORDERED LIST ..., the budgets ..., the cells, and the DIFFICULTY PROFILES" (A:19-25)

**Evidence.** L2 is locked (v0.1: "Reopening one requires an explicit decision, not drift"). L8 assigns post-validation metric checks per archetype: "rats-reach-food distances, retreat-room quality ..., chokepoint counts within bounds, connectivity sanity". Strand (a) adds pressure-cell levels as a validator metric; strand (d) the n-compartment standard. The layout adds an archetype-specific acceptance, "bot-walkable between spawns" (L §2). §4 RECORDS "spawn-to-spawn walk distance" as a factor, but nothing GATES it. The doc never mentions a validation profile and does not mark the omission as a departure.

**Fix.** Add a `[validation]` table: named gates from a closed registry in code (the `validate` gates are always on; `bot_walkable = true`; `spawn_distance = { medium = [lo, hi], ... }`; `cells = [lo, hi]`; `rooms = { size = [lo, hi] }`, which also answers R §5.4's open "target room counts"). A miss is a layout reject. If v1 deliberately ships with an empty profile, say so as a dated departure from L2.

### 13. MAJOR -- the draft rolls departments whose recipes need unbuilt mechanisms, and their refusal is a crash, not a reroll

> `pool_hall count = { medium = [0, 1], large = [0, 1] }`, `garden_dome ...`, `medbay ... [1, 1]` (A:65-83); Costs: "the freighter file + the first eight department files + six cycle files | data" (A:248)

**Evidence.** R §3.2: "A key whose mechanism is not built yet (`water`, a future `air`) is ACCEPTED by the schema and REFUSED at stamp time naming the missing extension". The pool hall's tank and the aquarium fixture are `water = "full"`, and the per-room water stamp does not exist (R §5.2). The garden's pond is a water group. Medbay (a FIXED `[1, 1]`) has an isolation room with its own airlock and cell, which finding 8 shows the cell numbering cannot express; its O2 store needs the bulk-gas payload plus the heat trigger (both now built, per the rl-ready run log). A stamp-time refusal is an exception, and `generate` catches only `_Reject` (gen_ship.py:1960-1965), so every seed that rolls the pool hall ends in an exception, not a reroll. Costing eight department files as "data" hides that at least two need stage-2 patches first.

**Fix.** The archetype loader walks its transitive closure at load and refuses (by name, naming the extension) any department or fixture key whose mechanism is unbuilt. The v1 freighter names only buildable departments (pool and garden `[0, 0]` or omitted until their stamps land). Cost each department against R §5.2's extension table.

### 14. MAJOR -- `mutate` is not well defined under phases, two streams and rerolls

> "`mutate(spec, entry)`: apply one more `[[apply]]` entry to a spec's ship and re-lay it out ... trivial once the executor exists" (A:205-208)

**Evidence.**
- Phase order (L6): a phase-2 entry applied to a finished ship runs after phases 3 and 4 (secrets drawn, ducts linked, tags written). Either the executor re-runs phases 3-4 (changing the secret draw), or the mutated ship violates L6.
- Purity: the result is no longer a function of the LevelSpec. The spec must carry the mutation list, and `recipe_hash` must not change, or every mutant is a new "archetype".
- Rng: where do the extra entry's draws come from: the end of the grammar stream, or a new child? The layout's draw list is "ordered ... per department" (L §11), so one more department or room shifts every later jitter draw, and the "edited" ship differs everywhere. ACCEL's premise is a SMALL edit of a level the agent already found hard.
- Failure: a mutant can fail every layout attempt (finding 1). ACCEL needs a defined "mutation rejected" result.

**Fix.** Define `mutate(spec, entry) -> spec'` with `spec'.mutations = spec.mutations + (entry,)`. Mutations are appended to their phase's entry list and the whole pipeline is re-run from the same seeds. Layout draws are keyed per department, e.g. `SeedSequence([seed, 1, g, a, dept_index])`, so unrelated departments keep their geometry. A rejected mutant returns `None`. Cost it as a real piece, not "trivial".

### 15. MINOR -- banks are a name, not a format

> "Banks (`train` / `test-iid` / `test-ood` / `probe`) are lists of LevelSpecs plus the generator version, kept as data under `levels/banks/`; a generator change that moves any bank's ships is a re-baseline" (A:200-204)

**Evidence.** "Moves any bank's ships" can only be detected if each bank row records an OUTPUT digest (tilemap hash or the `ShipGraph`'s), which the format lacks. `test-ood` is defined in the research by a held-out factor value (silhouette, recipe, loop count above the training range). With one archetype and one skeleton, the only axes are size, profile and counts; a loop count above training must lie inside the recipe's own ranges to be generable, and nothing says which axis a bank holds out. `run_match` takes a LOADED level (`run_match(level, ...)`, run_match.py:84), and recordings name a folder, so a bank row also needs a materialisation rule (generate to a folder, or in memory with a level id).

**Fix.** Bank row = `(LevelSpec, output_digest)`. Bank header = `(name, held_out = {factor: values}, generator_version)`. State how a row becomes a `LevelData` (folder path convention or in-memory id). Keep "not built in this arc", but fix these fields now, since the format is what §4 claims to fix.

### 16. MINOR -- spawn exits: range or fixed, when drawn, and who knows which exits exist

> `spawn = { exits = [1, 3], default = 2 }` (A:40)

**Evidence.** Is the exit count drawn from `[1, 3]` or fixed at `default`? If drawn, when (R §5.3 puts exits in phase 2; §2's draw order never mentions them)? Exits 2 and 3 exist only if the first department of row A / row B has an `exit` room (L §7), and which department is first in its row is decided by the LAYOUT's row assignment (L §6, gradient order + alternation + first-fit). So the grammar needs that arithmetic to know what it can declare. The hull-strip door "counts as an exit" (L §7), so a flank pair can push the realised count past the drawn one. No profile key touches exits, though they are the most direct difficulty knob for a spawn.

**Fix.** "The grammar draws the exit count uniformly in `exits` in phase 2, after the cycle entries, capped at what the layout's (deterministic, draw-free) row assignment makes realisable; `default` is used when ..." (or delete `default`). Say whether the strip door counts toward the drawn number. Consider an `exits` profile key.

### 17. MINOR -- the axiom roles live in code, but the doc says they live in the file

> "The skeleton's own rooms (slab, passageway, lobby, sections) and their edges ... are axiom nodes implied by `skeleton`" (A:48-50); Systems: "A new ship kind is a file, never a code path" (A:278-279); L §2: "The freighter archetype's axiom (archetype doc §1) declares these quotient rooms with fixed roles"

**Evidence.** The layout doc says this file DECLARES the roles; this file says the skeleton name IMPLIES them. The roles, their edges and their edge kinds (slab-passageway door, lobby-engine room airlock pair) are therefore code in `ship_layout.py`. The skeleton also presupposes an engineering axis department (the lobby airlocks land on its hub), so the `[[axiom]] department = "engineering"` row is not optional, and the doc does not say what happens if a recipe omits it. §8 admits "a second skeleton needs its own rows"; the Systems rule overstates.

**Fix.** Pick one home and state it in both docs: either the archetype file lists the axiom roles and skeleton edges as data and the layout maps them, or the skeleton is code and the rule reads "a new ship kind ON AN EXISTING SKELETON is a file; a new skeleton is layout code first". The loader refuses a freighter-skeleton recipe without an `axis` engineering axiom.

### 18. MINOR -- L6 budgets, target counts and the cost table

- L6 gives each phase "its own rule subset and budget". R §5.4 handed this session "budgets per phase ... target room counts", and v0.1 §8 lists "Target counts: rooms per map per archetype (affects budgets in L6)". Neither appears. Per-entry ranges bound termination, but there is no room-count target or ceiling, which the factors and the bank axes need (finding 12's `rooms` gate would close both).
- §1.1 allows no phase-3 entries at all, so L6's terminalise phase has no archetype-chosen rule subset (fixtures are department data and the secret draw is code). That is fine, but say it is a choice.
- §1.1 "phase ... 2 cycles with walk legs": `airlock_bypass` has a `walk = false` leg and `pocket` is phase `[1, 2]`. Say "cycles whose `phase` list contains 2", which is the next clause anyway.
- `on = "lobby-engineering"` (A:101) vs the layout's "lobby-engine room" (L §2, §4.3); `engineering.engine_room` presumes a room id in a department file that does not exist yet. Name the convention.
- The `maintenance_ring` cycle file allows `rooms = [4, 8]` (R §2); the layout refuses > 6 (L §4.4). The archetype loader is where this should be caught, since it knows the skeleton.
- R §3.1's prose still says the retreat loop is applied "between departments' entries" (R lines 316-317), against R §1.3 / L §4.1 (one department's entry) and this doc's `department-entry`. Fix it in R when this doc lands.
- Costs omit: the grammar attempt loop and its failure-rate gate (finding 1); the `match.toml` door change, re-pins and recording compatibility if any seed key is added (finding 10); a walk-distance metric for the recorded factor "spawn-to-spawn walk distance" (an A* after `embed`); provenance surviving the layout so "cycle applications per type" can be recorded; the recipe-path `check_request` with profile-scaled zombies (finding 9); a `[levelgen]` writer and reader (a new single-table `MANAGED_FAMILIES` row, as `[water]` / `[bake]` are, level_lib.py:80-99; `level_loader` has no `[levelgen]` field today, so the reader and the nested factor tables are new code).

## What the document gets right (must survive revision)

- One file per ship kind, holding the axiom, an ordered `(phase, department | cycle, on, count-per-size)` list and the budgets: the strand-(a) recipe idiom, authored as data an LLM can write (L7 as amended).
- Draw order fixed by file order within a phase, phases in order, no set iteration, and the mirror never drawn. The right skeleton for the executor once finding 5's pseudo-code fills it in.
- A failing application is REJECTED BY NAME and logged in provenance, not silently dropped (strand a, A6).
- Refusal at load, by name and size, including a feasibility check per allowed size, and a WARN (not a refusal) when the maxima overflow (A:217-218): the right severity split.
- `sizes` excludes small, which stays `plan_graph`'s arena with `ship_demo` byte-identical.
- The `on` vocabulary is closed and grows only as "a layout primitive first, then a row here" (A:184): the one-layouter rule kept from the archetype side.
- LevelSpec records the REALISED factors, not just the requested ones, and banks are versioned data with a re-baseline rule like a golden (strand c).
- `mutate` named once so it is not designed twice. Keep the name, fix the definition (finding 14).
- §6 separates Erik's content calls (which departments, at what odds) from the format, as vetoable statements, and "easy keeps every department (variety is not difficulty)" is a clean principle.
