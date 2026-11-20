# Swarm units P5d -- broods on generated ships (as built, 2026-11-19)

The design is the vocabulary doc's Part G
(`docs/levelgen_zone_vocabulary_2026-10-29.md`, committed first). This note
is the as-built: what was built, the numbers, the measurements, the breaks,
the findings.

## What was built

- **The department door** (`tools/levelgen_recipes.py`): an optional
  `[brood]` table -> `BroodRule` (`chance`, `rooms`, `density`, `max`,
  `with`), refused by name (unknown key; chance outside [0, 1]; a room that
  is not the department's, or is its entry or a corridor; density <= 0; max
  outside [1, `[swarm.larva] max_units`]; a `with` key that is no
  department, or the department itself; a multiplier < 0).
  `brood_eligible(dept)` is the room list (default: every non-entry,
  non-corridor room).
- **The archetype door** (`tools/ship_grammar.py`): an optional
  `[infestation]` table over `INFESTATION_STATES = (clean, nest, infested)`,
  weights >= 0, at least one > 0, normalised; unknown state refused by name.
  `Archetype.infestation` (empty: always clean).
- **`tools/ship_broods.py`** (new, the one writer): `draw(arch, seed)` on
  `SeedSequence([seed, 2])` (draw 0 the state, draw 1 + k department k's
  roll, k over the archetype's `[brood]` departments in archetype order);
  `hosts(arch, inf, present)`; `apply(graph, q, room_ids, arch, state,
  hosting)` tags the largest eligible room of every hosting department
  instance AND its mirror twin with `brood`, and on an infested ship
  rewrites every `door` edge between a hosting room and the passageway to
  `entity_door` (an existing row: same opening, width, compartment rule;
  the door entity is emitted open, the opening stamped air). Returns the
  graph untouched when nothing hosts.
- **Stage 2** (`tools/gen_ship.py`): the `brood` ROOM_TAGS row;
  `_build_broods` (called in `embed` after every other entity, so no
  existing ordinal moves) emits one `swarm_brood` zone entity per tagged
  room (`brood_r<room id>`, zone ids 1, 2, ... in room order) over the room
  rectangle less carved vestibules and spawn boxes, count
  `min(max, max(1, floor(density x free tiles)))`; refuses a brood on a
  spawn / zombie / secret / department-less room, a zone with no free tile,
  and a ship total past `max_units`. `Embedding.zones` -> `zones.npy`
  (`level_lib.write_zones_npy`), now an `OPTIONAL_DATA_FILES` row.
  `_generate_recipe` draws once before the grammar loop and applies per
  layout attempt; `ShipPlan.infestation` / `.broods`; the level header
  names the state.
- **The record** (`level_loader`): `[levelgen]` realised keys
  `infestation` (str) and `broods` (department names, a subset of
  `departments`). The decisive inputs (level seed, recipe tables) were
  already in the LevelSpec; the recipe hash reads the raw department and
  archetype files, so the new tables are in it.
- Data: `[brood]` rows on cargo_hold, crew_quarters, medbay;
  `[infestation]` on the freighter. Rules: one CLAUDE.md index line (Brood
  placement), a `levels.md` section, `tools/ship_broods.py` on its paths.

## The numbers (PLACEHOLDER; Erik tunes)

| Department | chance | rooms | density | max | with |
|---|---|---|---|---|---|
| cargo_hold | 0.6 | main_hold, bay | 0.5 / tile | 200 | engineering x 1.5 |
| crew_quarters | 0.15 | bunk, laundry | 0.5 | 200 | medbay x 0.5 |
| medbay | 0.1 | morgue | 0.5 | 200 | |
| engineering | none (moved, below) | | | | |
| armory | none | | | | |

Freighter: clean 0.25, nest 0.60, infested 0.15. Capacity: a ship's broods
total at most 690 larvae in the sweep, far under `max_units` 8192; `max`
200 is reached only by large main holds.

**Moved: engineering.** The lead's row (0.35, fuel bunker) could not hold:
every engineering room starts at `[temps.hot_engineering]` = +40 game
degrees (60 C), above the larva's `T_lethal_hot` 50 C, so the brood died on
its first tick (first run: a medium infested ship spawned 308, 252 alive
after tick 1 -- the fuel bunker's 60 gone). Nearest that holds: no row.

What the numbers mean on the freighter: engineering is the axiom, so the
cargo hold's `with engineering = 1.5` ALWAYS applies (effective 0.9); the
medbay is on large ships only, so the quarters' `medbay = 0.5` applies on
large only (0.075 there, 0.15 on medium).

## Measured: the standing sweep (100 seeds x medium/large x easy/standard/hard)

600 of 600 generate and validate (airtight, mirrored, reach, compartments,
the archetype's gates), 0 fairness failures (brood rooms mirror-closed with
equal twin counts, zone masks mirrored, never on a spawn box), 48 written
and loaded with an empty `brood_problems`; 128 s.

| | clean | nest | infested |
|---|---|---|---|
| states (100 seeds) | 0.27 | 0.57 | 0.16 |

The state is per seed (size and profile do not enter the draw).
Hosting share of ships: medium cargo 0.73, quarters 0.24; large cargo
0.72, quarters 0.20, medbay 0.20. Nest ships carry 2-4 broods (medium
90-316 larvae, mean 163; large 42-542, mean 305); infested 4 (medium,
128-294 larvae) or 6 (large, 310-690).

**Against the base.** The base generator (`63-swarm-units`, extracted with
`git archive`) and this one over the same 600 requests: every ship has the
same `(g, a)`; the 162 clean ones are identical (tiles, entities with
ordinals, lights, wires, spawn rooms, graph, temp seed); the 438 others
differ ONLY by the appended brood entities and, infested, the opened
corridor doors (`door` -> `entity_door`, tile closed-door -> air, entity
`initial_state` open). `levels/ship_demo` (v1) regenerates byte-identical
(its test, pixels for the PNG).

## Measured: bot matches (`docs/swarm_P5d_measure.py`)

CPU path, each ship's own match.toml (bots vs bots, 4 a side, the horde of
4, 300 s, eliminate), the same seed's ship WITHOUT broods as the control
(the archetype minus its `[infestation]` table: the same (g, a), tile for
tile). Written and LOADED through `level_loader`, played through
`tools/run_match.run_match`. Larvae alive at 0 / 30 / 60 / 120 / 240 s /
the end; the bite dose living marines took (Q16 poison units summed off the
bite plane under their footprints); bodies that rose.

| Ship (seed, state) | larvae (0 / 30 / 60 / 120 / end) | bites | risen | outcome, length | control | ms/tick (control) |
|---|---|---|---|---|---|---|
| medium s1 nest (cargo) | 116 / 82 / 9 / 5 / 1 | 0 | 0 | win 0, 218.8 s | win 0, 218.8 s | 20.4 (19.7) |
| medium s2 infested (quarters, cargo) | 248 / 205 / 25 / 9 / 6 | 0 | 0 | DRAW at 300 s | win 0, 114.8 s | 21.4 (17.9) |
| large s1 nest (cargo) | 400 / 343 / 343 / 343 / 343 | 0 | 0 | win 1, 133.6 s | win 1, 133.6 s | 52.5 (55.5) |
| large s2 infested (quarters, cargo, medbay) | 440 / 342 / 285 / - / 283 | 0 | 0 | win 0, 62.1 s | win 1, 102.8 s | 52.6 (52.3) |

Read: no marine was bitten and no body rose in any match. The bots never
walk into a department's rooms, and the larvae spread at 0.2 m/s from their
room, so a nest ship's match is the control's tick for tick (same outcome,
same tick). The INFESTED ships change the match (the outcome flips on both):
that is the opened corridor doors (sight and paths through the
departments), not the larvae. The tick cost of up to 440 larvae is in the
noise of this machine (another agent ran CUDA tests meanwhile): +0.7 to
+3.5 ms on medium, -3 to +0.3 ms on large.

Where the larvae die (the only death is the law's `T_lethal_hot`, 50 C):
first run with the engineering row, the fuel bunker's brood died on tick
1 (moved above). On the medium ships 90 % die within 60 s at tiles of +30
to +34 game degrees (50-54 C) inside their own rooms, idle as well as in a
match (s2 medium idle: 52 of 248 dead in 60 s; the match's gunfire adds
heat). The source: the generated ships' VENT tiles -- the tile beside a
cargo-hold return vent sits at +220 game degrees after 10 s on a ship with
nobody on it, others at +180 and +380 -- and the larvae, seeking 30 C,
climb the warm plume into it (finding 1).

## Measured: the game (`tools/e2e_drive.py --control direct --frames 400 --profile`)

All four ships (s1 / s2, medium / large) written by `tools/gen_ship.py
--recipe freighter` load and run 400 frames in `main.py`, exit 0. Medium
30.8 fps (s1 nest) / 25.5 fps (s2 infested). Large 3.0 fps with broods (s1
nest, s2 infested) and 3.1 fps for s1 WITHOUT broods (the control): the
large ships are that slow on the CPU path anyway; the broods cost nothing
measurable in the frame.

## Gates

- Full `pytest tests -q` at `109b329f` (+ docs): 6651 passed, 61 skipped,
  6 xfailed, 1 failed: `test_class_p_level.py::test_a_class_p_change_merges_only_with_a_green_stamp_at_head`
  -- the ARC's Class-P diff against main (swarm C++, gamemap, physics
  runner) with no `.cuda_green` in this fresh worktree; this patch changes
  no Class-P path (its diff: tools, level_loader, recipes, tests, docs,
  rules). Main's eight pinned-match files and GOLDEN_AGGREGATE pass; nothing
  re-baselined. 34 min on the Home Desktop with another agent's CUDA tests
  running.
- `tests/test_claude_rules_paths.py` 74 passed.

## Tests (`tests/test_ship_broods.py`, 29; each names its property and its break)

The draw moves no other draw (every ship == the same archetype without the
table, but for broods + opened doors; clean identical); no table = always
clean; state frequencies within 4.5 sigma of the weights; nest rolls at the
effective chance and `with` moving the frequency its declared way, a nest
always forced; rowless departments never host; mirror symmetry; forbidden
rooms never host (and no zone on a vestibule or spawn box); the stage-2
refusal on slab / passageway / lobby / section / secret; the vestibule mask
(hand-built input: no shipped row reaches a carved room); counts follow the
row; infested ships open exactly the hosts' corridor doors; the record
round trip through write -> load -> `generate_from_spec`; zones.npy cleared
on rewrite; both doors refuse by name (9 + 3 cases).
Amended: `test_gen_ship_recipe.py::test_a_recipe_ship_is_a_function_of_its_two_streams`
reproduces the third stream (the design change, not a bend).

## Breaks (each applied alone, `tests/test_ship_broods.py` run, restored)

| Sabotage | Red |
|---|---|
| S1 twin not tagged | mirror symmetry |
| S2 every department a candidate | nest rolls (a rowless department crashes the plans) |
| S3 weights paired with the wrong states | state frequencies |
| S4 `with` divides instead of multiplies | nest rolls |
| S5 the room filter ignores `rooms` | forbidden rooms |
| S6 stage-2 refusal off | the five refusal cases |
| S7 count cap dropped | counts follow the row |
| S8 any host door opens (not only corridor) | infested doors |
| S9 record drops `broods` | record round trip |
| S10 zones.npy not cleared on rewrite | zones cleared |
| S11 chance range unchecked | department door [chance] |
| S12 negative weight unchecked | archetype door [clean] |
| S13 the brood path takes a draw off the layout generator | the draw moves no other draw |
| S14 zone keeps carved / spawn tiles | the vestibule mask (green before that test was added) |
| S15 nest never forced | nest rolls |

## Findings (reported, not fixed)

1. **Vent hot spots kill the broods** (pre-existing, not this patch's):
   on recipe ships the tile at a vent reaches +180 to +380 game degrees
   within seconds with no fire and no one aboard (the cargo hold's return
   vent, ship s1 medium: (32, 60) at +220 after 10 s). The larvae's
   warmth-seeking climbs into it and dies at 50 C. Most medium broods are
   ~90 % dead in a minute; it is the reason a generated brood rarely meets a
   marine. The fix belongs to the vents / gas energy books (the room's air
   should not heat at an aperture), or a design call (broods only in rooms
   without a vent; a lethal line above the vent artefact). Not touched.
2. **Engineering cannot host** at today's presets (moved above): either the
   preset (+40) or `T_lethal_hot` (50 C) is Erik's call for a warm nest.
3. **Bots never meet the larvae**: no bite, no corpse raised in four
   matches. The broods sit in department rooms the bots do not enter; until
   the larvae spread (behaviour, not this patch) or the bots route through
   departments, a brood only matters to a player who goes in.
4. **The infested state changes the match through its doors** (both
   infested matches flipped their outcome, with zero bites): open corridor
   doors open sight lines and paths. Intended by the design, but it means
   `infested` is a much larger gameplay lever than `nest` today.
5. On the freighter the cargo hold's `with engineering = 1.5` is a constant
   (engineering is the axiom) and the quarters' `medbay = 0.5` exists on
   large ships only: the correlations do real work only when an archetype
   makes their partner optional.
6. The state is a function of the level seed alone (size and profile do not
   enter), so seed s is "infested" at every size and profile.

## Left out

Profiles shifting the weights (not one line); `rooms` by fixture; any
species but larva; mirrored scatter inside a room (the brood's own draws);
broods in the diffuse picture; any larva behaviour, sim or C++ change.
