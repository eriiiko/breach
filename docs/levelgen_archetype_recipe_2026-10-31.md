# Levelgen: the archetype recipe -- the build order for a whole ship

> **Status**: design, **revision 2** (2026-10-31 night, lead
> `level-design-lead`, Claude alone; Erik on a break). Revision 1's
> adversarial critique (`levelgen_archetype_recipe_2026-10-31_critique.md`,
> Opus, 18 findings, 3 blockers, every code claim checked) is folded in.
> Revision 2 is NOT re-critiqued; Erik may call one. The FORMAT is the
> lead's call; the CONTENT of the first recipe (which departments a
> freighter carries, at what odds) is game design, written in §7 as Erik's
> to veto. Serves #136 inside milestone 11 (#131); §5 is the RL contract and
> serves #76. Design only: no code.
>
> **What revision 2 changed.** The executor is pseudo-code with a GRAMMAR
> attempt loop over its own rng stream (a draw the layout cannot realise no
> longer kills the seed). The freighter fits the layout's real numbers: a
> medium ship is quarters + cargo and nothing else. The flank pair and the
> retreat loop ride a named seventh move, `second_leg`. `any-room` became
> `department-room`; the ring and the bypass are mutually exclusive per
> ship. Counts are QUOTIENT counts. Profiles name their operators and are
> checked per (size, profile). Cells: the department file is the default,
> the archetype PROMOTES. Zombies scale the generator's argument. The seed
> contract is what the code almost has: the level has its own seed, the
> match keeps its one `seed`. LevelSpec carries every generator input and
> the generator version; `recipe_hash` is defined. A `[validation]` table
> restores L2's third component. Unbuilt departments are refused at load.
> `mutate` is defined. Banks are a format.
>
> **Amends** (dated lines land in the named documents in the same patch):
> recipe doc §0.1 (`second_leg`, the seventh move), §3.1 prose (the loop is
> on ONE department's entry); layout doc §2 (the skeleton's roles are CODE;
> the archetype names the skeleton); v0.1 §9 (seven moves); the RL-ready
> scope doc's line "one seed drives levelgen, spawn, bots and zombies"
> becomes "one seed per match; the level has its own" (§5).

## 0. What an archetype recipe is

v0.1 L2: **a map archetype = axiom + rule subset + validation profile.** In
family B that is one file: the SKELETON it is laid out on (code in the
layout stage; the file names it), the AXIOM (the departments every ship of
the kind carries; the skeleton's own rooms come with the skeleton), an
ORDERED LIST of applications per phase (which departments, which cycle
types, how many), the budgets (secrets, spawn exits), the cells it
promotes, the DIFFICULTY PROFILES that shift the counts, and the
VALIDATION profile (the metric gates, L8). Everything in it is a name or a
count range; no geometry, no physics number. A ship is a pure function of
its LevelSpec (§5.1).

## 1. The file

`levels/recipes/archetypes/<name>.toml`.

```toml
[archetype]
name = "freighter"
skeleton = "freighter"          # a skeleton the layout stage implements in code (layout doc §2); its rooms and edges come with it
sizes = ["medium", "large"]     # gen_ship.SIZES presets this recipe fits (checked per size and profile, §6); small stays plan_graph's arena
passageway = { across = [6, 8] }       # the wing passageway's width
lobby = { x = [8, 10] }                # the wing lobby's x-extent
band = { sections = [3, 5] }           # INTERSECTED with the preset's own range at load (medium: 3-4)
spawn = { exits = [1, 3] }      # drawn uniformly in phase 2 (§2); capped at what the skeleton can realise (§3)
zombies = "axis_section"        # the horde's room: the main-cell axis section nearest the centre
secrets = { budget = 1 }        # units per ship: a wing secret (a mirrored pair) or an axis secret each cost one
allow_cross_cell_leak = false   # L5's leak-path clause; true is REFUSED until the damper (#49) exists

[cells]
promote = []                    # main-cell departments this archetype makes their own cell (the department file's `cell` is the default)

# ---- the axiom: departments present in every ship of this kind ----
[[axiom]]
department = "engineering"      # REQUIRED on the freighter skeleton: the lobbies' airlocks land on its engine room
placement = "axis"

[[axiom]]
department = "crew_quarters"
placement = "wing"              # once per wing: every count in this file is a QUOTIENT count (§2)

# ---- phase 1: grow (departments, their internal cycles) ----
[[apply]]
phase = 1
department = "cargo_hold"
count = { medium = [1, 1], large = [1, 1] }

[[apply]]
phase = 1
department = "medbay"           # v1: without its isolation room (a room-level cell; §7)
count = { medium = [0, 0], large = [1, 1] }

[[apply]]
phase = 1
department = "armory"
count = { medium = [0, 0], large = [0, 1] }

# garden_dome and pool_hall: NOT in v1 -- their water stamps are unbuilt (recipe doc §5.2); the loader would refuse them (§6)

# ---- phase 2: tactical patterns (ship-level cycle types; then the spawn exits draw) ----
[[apply]]
phase = 2
cycle = "retreat_loop"
on = "department-entry"         # anchor class (§3): a wing department with an attach = "end" room
count = { medium = [0, 0], large = [0, 1] }

[[apply]]
phase = 2
cycle = "flank_pair"
on = "slab-lobby"
count = { medium = [0, 1], large = [0, 1] }

[[apply]]
phase = 2
cycle = "maintenance_ring"      # large only (band width >= 20); mutually exclusive with the bypass per ship (§3)
on = "engineering.engine_room"
count = { medium = [0, 0], large = [0, 1] }

[[apply]]
phase = 2
cycle = "airlock_bypass"
on = "lobby-engine_room"
count = { medium = [0, 1], large = [0, 1] }

[[apply]]
phase = 2
cycle = "pocket"
on = "department-room"
count = { medium = [0, 1], large = [0, 2] }

# ---- phase 3: terminalise -- no entries by choice: fixtures are department data, the secret draw is the budget ----

# ---- phase 4: annotate (ducts) ----
[[apply]]
phase = 4
cycle = "duct_link"
on = "same-cell-rooms"
count = { medium = [0, 1], large = [0, 2] }

# ---- validation profile (L2's third component; L8): metric gates from a closed registry in code ----
[validation]
bot_walkable = true             # A* between the two slabs with airlock outer doors CLOSED (layout doc §2)
spawn_distance = { medium = [40, 120], large = [80, 220] }   # walk-graph tiles between slab centres, after embed
cells = [2, 3]                  # pressure cells per ship
rooms = { medium = [14, 30], large = [28, 60] }              # rooms per ship (L6's target count)

# ---- difficulty profiles: each key names its operator ----
[profiles.easy]
cycles_shift = -1               # added to every phase-2 and phase-4 cycle range, floored at 0 (pockets included)
secrets = 0                     # ABSOLUTE budget
zombies_scale = 0.5             # multiplies the generator's zombies argument (§4)

[profiles.standard]             # the file as written

[profiles.hard]
cycles_shift = 1                # capped at the layout's realisable maximum per type (§3)
promote = ["armory"]            # ABSOLUTE set, replaces [cells].promote: the armory becomes its own cell
zombies_scale = 1.5
```

### 1.1 Field reference

| Key | Meaning | Loader check |
|---|---|---|
| `archetype.skeleton` | the layout skeleton (code) | one the layout stage implements; its required axiom departments present (freighter: an `axis` engineering) |
| `archetype.sizes` | presets this recipe is allowed on | each in `SIZES`; the feasibility twin (§6) passes on each (size, profile) |
| `archetype.passageway`, `lobby`, `band` | the skeleton's own ranges | `band.sections` intersected with the preset's; a ring entry on a size needs `band_w.hi >= 20`; a bypass entry needs the hub section's height >= 13 (the band packer pins it) |
| `archetype.spawn.exits` | range drawn in phase 2 | within 1-3 |
| `archetype.zombies` | `axis_section` (v1's only value) | known value |
| `archetype.secrets.budget` | units per ship | int >= 0 |
| `archetype.allow_cross_cell_leak` | L5's leak clause | `true` refused naming #49 |
| `cells.promote` | `main` departments made `own` | each a wing department whose stub `across` range allows >= 7 (the airlock entry; layout doc §4); the grammar rewrites its entry edge to `airlock` in phase 1, logged in provenance |
| `[[axiom]]` | departments in every ship | exists; `placement` matches the department's |
| `[[apply]]` | `phase`, exactly one of `department` / `cycle`, `on` for a cycle, `count` per allowed size | phase 1 departments only; phase 2 cycles whose `phase` list contains 2; phase 4 `duct_link`; `on` a row of §3; the cycle's `rooms` within what the skeleton realises (ring <= 6) |
| `count` | `{size = [lo, hi]}`, a QUOTIENT count | `lo <= hi`; every allowed size present |
| `[validation]` | gates from the registry | known keys; ranges per allowed size where sized |
| `[profiles.*]` | `cycles_shift` int; `secrets` int (absolute); `zombies_scale` float > 0; `promote` list (absolute) | per profile: the shifted ranges capped at the realisable maxima; the feasibility twin re-run; `promote` checked as `cells.promote` |

A phase with no entries is legal. Refusal by name, at load, before any
ship is planned. **The loader walks the recipe's transitive closure**
(departments, their fixtures, cycles) and refuses, naming the extension,
any key whose mechanism is unbuilt (recipe doc §5.2), so an unbuilt
department never reaches a stamp-time exception.

## 2. The executor (determinism)

Two rng streams (layout doc §11), with a GRAMMAR attempt loop:

```
for g in range(G_MAX):                       # G_MAX = 8
    rng_g = default_rng(SeedSequence([seed, 0, g]))
    graph = grammar(recipe, size, profile, rng_g)      # may raise GrammarReject
    if graph is None: continue
    for a in range(MAX_ATTEMPTS):            # 32, as today
        rng_a = default_rng(SeedSequence([seed, 1, g, a]))
        try: plan = layout(graph, rng_a); emb = embed(plan, rng_a)
        except _Reject: continue
        if validate(...) passes and validation_profile(...) passes: return plan, (g, a)
raise ShipGenError
```

`grammar` applies the axiom, then the phases in order, then within a
phase the `[[apply]]` entries in FILE order:

```
k = rng_g.integers(lo, hi + 1)                        # the count, uniform
cand = sorted(candidates(on), key=global_id)          # (side, department, instance, room)
placed = 0
while placed < k and cand:
    c = cand.pop(rng_g.integers(len(cand)))           # uniform, without replacement per entry
    if precondition(c) and apply(c) keeps the phase's invariants (recipe doc §5.3):
        placed += 1; provenance.append((entry, c, "ok"))
    else:
        provenance.append((entry, c, reason))         # a rejected candidate does not consume the count
if placed < lo: raise GrammarReject(entry)            # the grammar attempt g is over; g + 1 draws afresh
```

Then, still in phase 2 and after the cycle entries, the spawn exits:
`exits = rng_g.integers(lo, hi + 1)` capped at what the skeleton realises
(§3). `global_id` is `(side, department, instance, room_id)` with `side`
in `(axis, wing)`, `instance` the department's draw index, `room_id` the
recipe's id plus the `count` index; it is assigned when the department
expands and never changes. **Every count is a quotient count**: the grammar
runs on one wing plus the axis band; a wing application appears once per
wing after the layout reflects, an axis one once per ship. `(g, a)` are
recorded in the LevelSpec's realised factors. The mirror is never drawn.
Invariants run per phase as the recipe doc §5.3 says (reachability from
phase 3 on, the air invariants in phase 4), never "all after every
application".

## 3. The anchor classes (what a ship-level entry may act on)

| `on` | Candidates (graph terms) | Move | Realised as (layout doc §4) | Realisable maximum |
|---|---|---|---|---|
| `department-entry` | a wing department with an `attach = "end"` room and loader-twin slack for a second stub | `second_leg` on passageway-entry | two-stub | one per department |
| `slab-lobby` | the wing's two hubs | `second_leg` on slab-passageway-lobby | hull strip | one per wing |
| `engineering.engine_room` (the `<department>.<room id>` convention) | the hub | `ring` | ring in the band | one per hub |
| `lobby-engine_room` | the lobby's airlock edge into engineering | `parallel` | breakable on the airlock wall | one per wing; **mutually exclusive with the ring** (v1: a ship has one or the other; the second application is rejected by name) |
| `department-room` | a department room the layout can attach to (`side` or `behind`), never an axiom or ring room; precondition: the department's loader-twin slack at the room's minimum | `leaf` | packed room or behind room | bounded by slack |
| `same-cell-rooms` | two rooms of one cell in different departments, both in the quotient wing or both on the axis | `duct` | none | bounded by vent sites |

**`second_leg`** (the seventh move, amendment to the recipe doc §0.1 and
v0.1 §9): precondition, a path A-C-B exists; effect, add A-D-B with D new
(one room, the cycle file's single leg); the existing leg is not redrawn.
It is what `edge_to_cycle` becomes when one leg is already an axiom path.
The flank pair and the retreat loop files carry ONE `[[leg]]` each.

A new class is a layout primitive first, then a row here; the "realisable
maximum" column is what the profiles' shifts are capped at.

## 4. Zombies and cells under a profile

- **Zombies.** The profile scales the GENERATOR's `zombies` argument
  (round half up; at least 1 when the base is > 0). The recipe path's
  `check_request` checks the scaled count against the zombie section's
  capacity; the scaled count is written into the zombie `[[spawn_room]]`
  and `match.toml`'s `zombies`, which is how `parse_match`'s ceiling is met.
  The LevelSpec records the base and the result.
- **Cells.** The department file's `cell` is the default (engineering is
  `own` in its own file; listing it in `promote` is redundant and refused
  as such). `promote` makes a `main` wing department `own`: its entry edge
  becomes an airlock pair (the grammar rewrites it in phase 1), its stub
  must take `across >= 7`. On `hard` the armory is then unreachable to the
  bots until #142 (a feel consequence, §7). A ROOM-level cell (the
  medbay's isolation room, vocabulary B.3) cannot be expressed by the
  resolver's per-department numbering; v1's medbay file omits the room.
  The magazine-as-own-cell idea of revision 1 is dropped (vocabulary B.5
  stands: v1's magazine doors are plain).

## 5. LevelSpec, seeds, banks, mutate (the RL contract, strand c)

### 5.1 LevelSpec

`LevelSpec = (archetype, recipe_hash, generator_version, size, profile,
max_per_team, zombies_base, level_seed)`, every generator input. A ship is
a pure function of it. Written into the level folder (`level.toml`'s
`[levelgen]` table, a new managed family in `level_lib`, read by
`level_loader`) together with the REALISED factors: `(g, a)`, departments
present, cycle applications per type (from provenance, which survives the
layout), cells, secrets, spawn exits, zombies after scaling, room count,
spawn-to-spawn walk distance (an A* after `embed`, the same metric the
validation profile gates).

- `recipe_hash`: SHA-256 over the parsed, key-sorted TOML of the archetype
  file WITHOUT its `[profiles]` table, plus the chosen profile's table,
  plus the transitive closure of every department, cycle and fixture file
  it names, plus the resolved preset values (recipe doc §3.4). A comment
  edit does not move it; retuning `hard` does not move `easy` ships.
- `generator_version`: `git describe` of the generator (the precedent is
  #126's recording header `build`), which covers the moves, `SIZES`, the
  layout constants and `MAX_ATTEMPTS`.
- Folder name: `ship_{archetype}_{profile}_s{seed}_{size}` (today's
  `ship_s{seed}_{size}` collides across archetypes and profiles, and
  recordings identify a level by folder). The #126 recording header gains
  `recipe_hash` when #126's patch touches it.

### 5.2 Seeds

- `level_seed` is the GENERATOR's seed, recorded in `[levelgen]`.
- `match.toml`'s existing `seed` IS the episode seed: no new key, no
  `parse_match` change, no re-pinned digest, every #126 recording stays
  valid. "Fixed level, varying episode" is `run_match --seed N` on one
  folder, as today.
- A loop that generates a level per episode may derive both from one root
  (`SeedSequence(root).spawn(2)`, each integerised by
  `generate_state(1)[0]`) and records both ints; a convenience, not the
  contract.
- The RL-ready scope's line "one seed drives levelgen, spawn, bots and
  zombies" is amended to "one seed per match; the level has its own"
  (dated line in `docs/rl_ready_scope_2026-10-29.md`, this patch). Any
  later `match.toml` change belongs to #76's patches, not this arc.

### 5.3 Banks

A bank is a file under `levels/banks/<name>.toml`: header `(name,
generator_version, held_out = {factor: values})`, rows `(LevelSpec,
output_digest)` where `output_digest` is the tilemap's digest from
`field_digest`. A row becomes a level by generating it into
`levels/banks/<name>/<folder>` (the §5.1 name) or in memory with that
folder name as its id. A generator change that moves any row's digest is a
re-baseline with rationale, like a golden. `train` / `test-iid` /
`test-ood` / `probe` are bank names; with one archetype and one skeleton
the held-out axes are size, profile and a count above the training range
INSIDE the recipe's own range. Not built in this arc; the format is fixed
here.

### 5.4 mutate

`mutate(spec, entry) -> spec' | None`: `spec'.mutations = spec.mutations +
(entry,)`; a mutation is an extra `[[apply]]` entry appended to its
phase's list, and the whole pipeline is re-run from the same seeds (so L6's
phase order holds and the result is a pure function of `spec'`, whose
`mutations` are part of it). The layout's per-department draws are keyed
`SeedSequence([seed, 1, g, a, dept_index])` so departments the mutation does
not touch keep their geometry (the small-edit property ACCEL needs). A
mutant that fails every attempt returns `None`. A real piece, costed in §8.

## 6. Feasibility at load, per (size, profile)

The archetype loader calls the LAYOUT's loader twin (layout doc §4, §6;
imported, never copied) for every allowed size and every profile:

1. the row packing at the recipe's MINIMA (departments at minimum count,
   one-sided demands) against the preset's row LENGTH and DEPTH, including
   phase-2 entries at their minimum (a mandatory two-stub widens, a
   mandatory hull strip shallows row A); a failure refuses the recipe by
   name, size and profile;
2. the same at the MAXIMA: a WARNING naming the expected reroll;
3. the skeleton conditions: ring needs `band_w.hi >= 20`, bypass needs a
   pinnable hub height >= 13, every `promote`d stub >= 7;
4. the closure walk for unbuilt mechanisms (§1.1).

A per-seed failure-rate GATE is part of the recipe path's tests: N = 100
seeds per (size, profile) must each produce a ship within `G_MAX ×
MAX_ATTEMPTS`, and the mean `(g, a)` is reported, so a recipe that only
just fits is seen before the RL track samples it.

## 7. The freighter's content: Erik's calls, written as the draft above

Each is a statement to veto, none is settled:

- Crew quarters and engineering are the AXIOM; cargo is in every ship.
- A MEDIUM freighter is quarters + cargo per wing and nothing else, no
  retreat loop, no ring: the two-department ship the layout's numbers
  allow. A LARGE one adds medbay and, half the time, an armory.
- The medbay ships WITHOUT its isolation room until room-level cells
  exist; the garden and the pool wait for their water stamps.
- The hard profile makes the armory its own cell (an airlock fight for the
  guns), which the bots cannot enter until #142.
- Secrets: one unit per ship on standard and hard, none on easy.
- Zombies: ×0.5 / ×1.5 of the generator's count on easy / hard.
- The validation ranges in §1 are first guesses to tune on the failure-rate
  gate.

## 8. Costs

| Piece | Where |
|---|---|
| archetype loader (closure walk, intersections, per-(size, profile) feasibility through the layout's twin) + the executor (§2: attempt loops, two streams, counts, candidates, provenance, per-phase invariants, exits draw, `second_leg`) | new `tools/ship_grammar.py` (stage 1), pure, seeded; ~500 lines + property tests (a crafted violating application is rejected by name; the same spec gives the same graph; the mirror holds; the failure-rate gate) |
| `[validation]` registry (bot-walkable A*, spawn distance, cells, rooms) | `ship_layout.py` acceptance + `validate` call site |
| `[levelgen]` managed family in `level_lib` + `level_loader` field; folder naming; `ShipPlan` carrying the spec and `(g, a)` | `level_lib.py`, `level_loader.py`, `gen_ship.py` |
| the recipe-path `check_request` with scaled zombies | `ship_layout.py` |
| `mutate` + per-department layout streams | `ship_grammar.py` / `ship_layout.py` |
| banks format + the walk-distance metric | this arc: the format and the metric; the banks themselves: the RL track |
| the freighter file, engineering / crew quarters / cargo / medbay / armory department files, six cycle files, `fixtures.toml` | data, with the implementing chat's loader; each department costed against recipe doc §5.2 before it enters the file |
| the scope-doc amendment (§5.2) | `docs/rl_ready_scope_2026-10-29.md`, one dated line |

## 9. Open

- An `exits` profile key (the most direct spawn difficulty knob). Draft:
  not in v1.
- Whether the strip's door into the slab counts toward the drawn exits.
  Draft: no, it is extra (the flank pair's own property).
- The validation ranges' numbers.

## Systems

**(a) Existing canonical systems this design uses**: Level generators
(`generate`'s retry and `MAX_ATTEMPTS`; `SIZES`; `check_request`'s role;
`ShipPlan`; the layout stage as the geometry producer); Level data layer
(`[levelgen]` through `level_lib`'s managed families, read by
`level_loader`); Match config (`match.toml`'s one `seed` unchanged;
`zombies` written from the plan as today); Spawn rooms (unchanged);
Determinism gates (`field_digest` for bank rows); the Recipe library,
Resolver, Fixtures (recipe doc) and the Layout stage (layout doc).

**(b) New systems this design creates** (draft rules, into CLAUDE.md when
the code exists):

- **Archetype recipe** -- `A ship kind ON AN EXISTING SKELETON is one file
  under levels/recipes/archetypes/: axiom departments, ordered (phase,
  department | cycle, on, count-per-size) entries, budgets, promoted cells,
  difficulty profiles with named operators, a validation profile; a new
  skeleton is layout code first. The executor applies the file in order
  with two rng streams and a bounded grammar attempt loop, rejects a
  failing application by name with provenance, and the loader refuses, per
  size and profile, a recipe the layout's own arithmetic cannot pack or
  whose closure names an unbuilt mechanism.`
- **LevelSpec** -- `Every generated level records its LevelSpec (every
  generator input, the recipe hash and the generator version) and its
  realised factors in level.toml's [levelgen] table; the level's seed is
  its own, the match's seed stays the episode's; a level is a pure function
  of its spec.`
