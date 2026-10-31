# Levelgen: the archetype recipe -- the build order for a whole ship

> **Status**: design, revision 1 (2026-10-31 night, lead `level-design-lead`,
> Claude alone; Erik on a break). The FORMAT is the lead's call; the CONTENT
> of the first recipe (which departments a freighter carries, at what odds)
> is game design and is written here as Erik's to veto, §6. Critique pass
> pending; aligned with the layout doc rev 2 (lobby, `on` classes, sizes). Serves #136 inside milestone 11 (#131); the RL contract part
> serves #76. Design only: no code.
>
> This is the last of the three data formats of family B
> (`docs/breach_levelgen_design_v0.1.md` §9): cycle types and departments
> are `docs/levelgen_cycle_types_and_recipes_2026-10-31.md` (rev 2); the
> geometry they become is `docs/levelgen_layout_stage_2026-10-31.md`. Reads
> the research digest strand (a) (recipes as (phase, rule, count)) and
> strand (c) (LevelSpec, seed split, profiles, banks).

## 0. What an archetype recipe is

v0.1 L2: **a map archetype = axiom + rule subset + validation profile.** In
family B that is one file: the SKELETON it is laid out on (the layout
doc §2), the AXIOM (what every ship of the kind has before any draw), an
ORDERED LIST of applications per phase (which departments, which cycle
types, how many), the budgets (secrets, spawn exits), the cells, and the
DIFFICULTY PROFILES that shift the counts. Everything in it is a name or a
count range; no geometry, no physics number. A ship is a pure function of
(archetype file, size preset, profile, level seed).

## 1. The file

`levels/recipes/archetypes/<name>.toml`.

```toml
[archetype]
name = "freighter"
skeleton = "freighter"          # a skeleton the layout stage implements (v1: this one)
sizes = ["medium", "large"]     # gen_ship.SIZES presets this recipe fits (the loader checks §5); small stays plan_graph's arena (layout doc §9)
passageway = { across = [6, 8] }       # the wing passageway's width range (the skeleton's one corridor)
lobby = { x = [8, 10] }                # the wing lobby's x-extent (layout doc §2)
band = { sections = [3, 5] }           # axis-band sections along y (within the preset's range)
spawn = { exits = [1, 3], default = 2 }   # Erik's ruling: 1-3, default 2
zombies = "axis_section"        # where the horde spawns: the axis section nearest the centre (as today)
secrets = { budget = 1 }        # mirrored pairs per ship, drawn by department weight (vocabulary Part C, F4)

[cells]
own = ["engineering"]           # departments that are their own pressure cell (L5); everything else rides "main"

# ---- the axiom: present in every ship of this kind, before any draw ----
# The skeleton's own rooms (slab, passageway, lobby, sections) and their edges
# (slab-passageway, passageway-lobby, lobby-main section, lobby-engine room
# airlock pair) are axiom nodes implied by `skeleton` (layout doc §2).
[[axiom]]
department = "engineering"
placement = "axis"

[[axiom]]
department = "crew_quarters"
placement = "wing"              # a mirror pair

# ---- phase 1: grow (departments, their internal cycles) ----
[[apply]]
phase = 1
department = "cargo_hold"
count = { medium = [1, 1], large = [1, 2] }   # per wing (mirrored)

[[apply]]
phase = 1
department = "medbay"
count = { medium = [1, 1], large = [1, 1] }

[[apply]]
phase = 1
department = "armory"
count = { medium = [0, 1], large = [1, 1] }

[[apply]]
phase = 1
department = "garden_dome"
count = { medium = [0, 1], large = [0, 1] }

[[apply]]
phase = 1
department = "pool_hall"
count = { medium = [0, 1], large = [0, 1] }

# ---- phase 2: tactical patterns (ship-level cycle types) ----
[[apply]]
phase = 2
cycle = "retreat_loop"
on = "department-entry"         # the edge class the move rewrites (layout doc §4.1); needs an attach = "end" room
count = { medium = [0, 1], large = [1, 2] }

[[apply]]
phase = 2
cycle = "flank_pair"
on = "slab-lobby"               # the wide leg is the passageway; the narrow leg a hull strip (layout doc §4.2)
count = { medium = [0, 1], large = [0, 1] }

[[apply]]
phase = 2
cycle = "airlock_bypass"
on = "lobby-engineering"        # the cell boundary: the lobby's airlock wall (layout doc §4.3)
count = { medium = [0, 1], large = [0, 1] }

[[apply]]
phase = 2
cycle = "maintenance_ring"      # large only: needs band width >= 20 (layout doc §4.4); the loader refuses it on medium
on = "engineering.engine_room"  # the hub (layout doc §4.4)
count = { medium = [0, 1], large = [0, 1] }

[[apply]]
phase = 2
cycle = "pocket"
on = "any-room"
count = { medium = [0, 2], large = [1, 3] }

# ---- phase 3: terminalise (fixtures resolve; the secret draw) -- no entries: the budget above drives it ----

# ---- phase 4: annotate (ducts) ----
[[apply]]
phase = 4
cycle = "duct_link"
on = "same-cell-rooms"
count = { medium = [0, 1], large = [1, 2] }

# ---- difficulty profiles: shifts applied to every count range (strand c) ----
[profiles.easy]
cycles = -1                     # every cycle-type range shifted down by 1, floored at 0
secrets = 0
zombies = 0.5                   # multiplier on the match's zombie count (the match config's number is the base)

[profiles.standard]             # the file as written

[profiles.hard]
cycles = +1
own_cells = ["engineering", "armory"]   # the armory's magazine becomes its own cell
zombies = 1.5
```

### 1.1 Field reference

| Key | Meaning | Loader check |
|---|---|---|
| `archetype.skeleton` | the layout skeleton | one the layout stage implements |
| `archetype.sizes` | presets this recipe is allowed on | each in `SIZES`; the feasibility check (§5) passes on each |
| `archetype.passageway`, `lobby`, `band` | the skeleton's own ranges | within the preset's range; the engineering hub section's height >= 13 when `airlock_bypass` may be applied (layout doc §4.3); `wm >= 20` when the ring may (large only) |
| `archetype.spawn` | exits range + default | within 1-3 |
| `archetype.zombies` | `axis_section` (v1's only value) | known value |
| `archetype.secrets.budget` | mirrored pairs per ship | int >= 0 |
| `cells.own` | departments that are their own cell | each a department whose recipe allows `cell = "own"` |
| `[[axiom]]` | departments present in every ship | department exists; `placement` matches the recipe's |
| `[[apply]]` | one entry: `phase`, exactly one of `department` / `cycle`, `on` for a cycle, `count` per size | phase 1 holds departments only, 2 cycles with walk legs, 4 `duct_link`; the cycle file's `phase` list contains the entry's phase; `on` is one of the layout's edge classes (§3) |
| `count` | `{size = [lo, hi]}` per allowed size | `lo <= hi`, every allowed size present |
| `[profiles.*]` | shifts | `cycles` an int, `secrets` an int, `zombies` a float > 0, `own_cells` a superset of `cells.own` |

A phase with no entries is legal (phase 3 above). Refusal by name, at load.

## 2. Execution order (determinism)

One `rng` (`SeedSequence([level_seed, attempt])`, as `generate` does today).
Draw order is the FILE order within a phase, phases in order, and for each
entry: the count first, then each application's choices (which edge of the
class, which rooms, which leg kinds) in the layout's canonical order of
candidates (rooms by id, edges by `(a, b)`). The mirror is never drawn:
every wing application is applied to the twin. No set iteration anywhere.
After every application the invariants of the recipe doc §1.5 run; a
failing application is REJECTED (skipped, logged in provenance) and the
count continues with the next candidate; a count that cannot be met at its
minimum rejects the DRAW (reroll under `MAX_ATTEMPTS`, L8).

## 3. The `on` classes (what a ship-level cycle type may rewrite)

The layout doc's skeleton gives the archetype a small vocabulary of edge
classes a cycle type can name; the layout realises each (its §4):

| `on` | Candidates | Realised as |
|---|---|---|
| `department-entry` | the entry edge of a wing department with an `attach = "end"` room | two-stub (retreat loop) |
| `slab-lobby` | the wing's two hubs | passageway + hull strip (flank pair) |
| `lobby-engineering` | the lobby's airlock wall onto the engine room | airlock + breakable on one wall (bypass) |
| `<department>.<room>` | that room as a hub | ring |
| `any-room` | any room with a free side | satellite (pocket) |
| `same-cell-rooms` | two rooms of one cell, different departments | duct link |

A new class is a layout primitive first (layout doc §3), then a row here.

## 4. LevelSpec and seeds (the RL contract, strand c)

- **`LevelSpec = (archetype, recipe_hash, size, profile, level_seed)`**,
  written into the level folder (`level.toml`'s `[levelgen]` table through
  `level_lib`) with the FACTORS the ship actually got: departments present,
  cycle applications per type, cells, secrets, spawn exits, room count,
  spawn-to-spawn walk distance. `recipe_hash` hashes the archetype file,
  every department and cycle file it names, `fixtures.toml` and the
  RESOLVED preset values (recipe doc §3.4).
- **Two seeds from one**: the match seed spawns `level_seed` and
  `episode_seed` through `SeedSequence.spawn`; the level is a pure function
  of its spec; spawns, rosters and controllers draw from the episode seed.
  A training run holds `level_seed` and varies the episode; a
  generalisation test varies the level. Both are recorded in `match.toml`.
- **Banks** (`train` / `test-iid` / `test-ood` / `probe`) are lists of
  LevelSpecs plus the generator version, kept as data under
  `levels/banks/`; a generator change that moves any bank's ships is a
  re-baseline with rationale, like a golden. Not built in this arc; the
  format is fixed here so the RL track (#76 onward) finds it.
- **`mutate(spec, entry)`**: apply one more `[[apply]]` entry to a spec's
  ship and re-lay it out; the hook for ACCEL-style curricula. One function
  in the executor, trivial once the executor exists; named so it is not
  designed twice.

## 5. Feasibility at load

The archetype loader, per allowed size:

1. sums the MINIMUM x-demand of the axiom's wing departments plus every
   phase-1 entry at its minimum count, per row (two rows), against the
   preset's smallest wing length (the layout doc §6);
2. sums the MAXIMUM demand against the largest wing and WARNS (not refuses)
   when it exceeds it: the draw will reroll often;
3. checks the band's width range affords a ring (`wm >= 20`) if
   `maintenance_ring` may be applied on that size, and the hub section's
   height the bypass's 13.

A recipe failing (1) or (3) on a size it lists is refused at load, by name
and size.

## 6. The freighter's content: Erik's calls, written as the draft above

Each is a statement to veto, none is settled:

- Crew quarters and engineering are the AXIOM (every freighter has both);
  cargo is near-certain; medbay, armory, garden, pool are odds by size.
- Recipe ships are MEDIUM (two departments per wing) and LARGE (four);
  the small preset stays the v1 BSP arena (layout doc §9).
- The hard profile makes the armory its own cell (an airlock fight for the
  guns).
- Secrets: one mirrored pair per ship on every profile but easy.
- Zombies spawn in the centre axis section, as `ship_demo` does today.
- The profiles shift cycle counts by ±1 and zombies by ×0.5 / ×1.5; the
  easy profile keeps every department (variety is not difficulty).

## 7. Costs

| Piece | Where |
|---|---|
| archetype loader + the executor (phases, counts, candidate order, provenance, invariants after each application, reject-and-continue) | new `tools/ship_grammar.py` (stage 1), pure, seeded; ~400 lines + property tests (an invariant violated by a crafted application is rejected and named; the same spec gives the same graph; the mirror holds) |
| LevelSpec written through `level_lib` (`[levelgen]` table) and read by `level_loader` | small; the format v3 header grows a table |
| seed split in `run_match` / `Simulation(seed=)` | the RL-ready track's patch (#76), named here only |
| the freighter file + the first eight department files + six cycle files | data, with the implementing chat's loader |

## 8. Open

- Whether `count` ranges per size belong in the archetype (as here) or in a
  separate size table. Draft: here; one file per archetype is the unit an
  LLM authors (L7 as amended).
- The `on` classes are freighter-skeleton words; a second skeleton needs
  its own rows. Fine for v1.
- Which factors the LevelSpec records beyond the list in §4 (chokepoints,
  fuel density): the RL track's call when it reads them.

## Systems

**(a) Existing canonical systems this design uses**: Level generators
(`generate`'s `SeedSequence([seed, attempt])` and retry; `SIZES`; the
layout stage as its geometry producer); Level data layer (`[levelgen]`
written by `level_lib`, read by `level_loader`); Match config (the match
seed is `Simulation(seed=)`; `match.toml` gains the two derived seeds
through `level_lib.write_match`); Spawn rooms (unchanged); the Recipe
library, Resolver, Fixtures (recipe doc) and the Layout stage (layout doc)
as the stages this file drives.

**(b) New systems this design creates** (draft rules, into CLAUDE.md when
the code exists):

- **Archetype recipe** -- `A ship kind is one file under
  levels/recipes/archetypes/: skeleton, axiom, ordered (phase, department |
  cycle, on, count-per-size) entries, cells, budgets and difficulty
  profiles; the executor applies it in file order with one rng and rejects
  a failing application by name. A new ship kind is a file, never a code
  path; a new count knob is a profile shift.`
- **LevelSpec** -- `Every generated level records (archetype, recipe_hash,
  size, profile, level_seed) and its realised factors in level.toml's
  [levelgen] table; the level seed and the episode seed are spawned from
  the match seed and both recorded; a level is a pure function of its
  spec.`
