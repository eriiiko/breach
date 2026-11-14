# The unit: character sheet, compile step, unit row -- and the swarm boundary

**Date:** 2026-11-14. **Issues:** #29 (residency, patch P6), #63 (swarm units),
#199 (drawn stats), #200 (stress physiology).
**Status:** v1, written after the design session with Erik of 2026-11-14. The
decisions in section 0 marked AGREED are Erik's ("I agree with everything you
just wrote"); those marked LEAD are Claude's technical calls, open to his veto.
**Replaces** `docs/architecture/mechanics/01_units.md` as the description of
what a unit is (Erik, 2026-11-14: that chapter is "extremely outdated -- only
thing we can take from it are the ideas and concepts"). The ideas kept from it,
from `docs/breach_unit_class_design.md`, `docs/unit_variants_design_brainstorm.md`
and `docs/breach_metaphysics_design_notes.md` are named where they are used.

This document fixes the SHAPE of a unit so that the game can grow stats,
stamina, stress, wounds and equipment for years without the shape changing. It
designs none of those systems in detail; each gets its own session.

---

## 0. The decisions

| # | decision | who |
|---|---|---|
| D1 | A unit is a ROW of integer columns plus rows of shared tables; behaviour is LAWS that run over all units at a fixed place in the tick. No per-unit code, no per-unit growing list. | AGREED |
| D2 | **The line.** The RPG layer (stats, equipment, rings, the random draw, names) lives on the HOST as a character sheet. A COMPILE STEP turns the sheet into a fixed list of per-unit multipliers, the HOOKS. The device unit holds pools, conditions, kit counts, control state and the hooks. No kernel ever reads a stat. | AGREED |
| D3 | Every stat acts through a law that already exists (speed, turning, the push law, the shot cone, handling times), never through a dice roll of its own. | AGREED |
| D4 | Prepare now, draw later: the hooks and the compile step go in before P6 at neutral values (every hook 1.0), so nothing in the game changes and no digest moves; the kernels read per-unit values from their first day. | AGREED (the timing was delegated) |
| D5 | Effective values are composed ONCE per tick, at the tick's head, for every unit, and are frozen for that tick. | LEAD |
| D6 | **Full units and swarm units stay two stores with two schemas and two sets of laws.** They share one plumbing (the batch axis, the residency path, reset, the draws, the fixed-point kits). They meet only through per-tile fields and through effect records applied at the receiver's own door. | LEAD |
| D7 | There is ONE batch axis N, supplied by `ResidentBatch` / `BatchedEnv`. A store declares per-game arrays and never an N of its own; the swarm store drops its `N_ENV` axis. | LEAD |
| D8 | P6 is re-planned against today's code and split into smaller patches (section 7). The hook patch (U1) runs first. | LEAD |

---

## 1. What exists (code at `12a38642`)

- **The unit is already a table row on the host.** `src/simulation/unit_schema.py`
  (#29 P3a-P3e) lists every attribute in one of three groups: S, the synced
  columns (about 75, all integers: Q16 positions, `hp_q`, the turning and push
  state, a fixed 16-row status block, the exposure block, grenade counts,
  cool-downs, the held verbs, the order); D, constants derived from S; R,
  host-only state. `Unit.__slots__` is generated from it and the unit digest
  hashes each S column only while it differs from its default.
- **The RPG stats exist as data and are almost unused.** `stats.BaseStats`
  holds strength, agility, endurance, vitality, intelligence, will_strength,
  imagination, will_orientation; `species.py` holds a distribution per stat.
  But `generation.predefined_unit_attributes` gives every unit the species
  MEAN (the multivariate-normal sampler was removed: it ran through BLAS and
  gave different numbers on different PCs). Only vitality is read, as the
  starting 100 HP of a marine; a zombie's HP is `[zombie] hp`. The stats sit
  in group D (a function of the species).
- **Class values live in tables:** `[marine]`, `[zombie]`, `[speeds]`,
  `[impulse]`, `[env_profiles.*]` (bands, `stability`, `expose_resist`),
  `[loadouts.*]` (weapon, grenades, magazines, `kit_kg`, melee, charge),
  `[weapons.*]`, `[ammo.*]`.
- **Per-unit physical state that already matters:** `mass_g` (body + kit, the
  Unit impulse row), the per-unit turning dials (`Simulation.unit_turn_rate_q16`,
  written for exactly this: "later a stat -- a change here, never at the
  callers"), `stability_q16(u)`.
- **Precedents for what is coming:** the Exposure law (a per-unit level raised
  by events that decays each tick in integers; dormant at zero; hashed while
  non-zero) is the shape of every future meter; the status block (kind,
  magnitude, ticks left, source) is the shape of every timed condition; the
  keyed Philox draws (`draws.MatchDraws`, `rng_salts.Salt`) are the only
  source of randomness.
- **The swarm store** (arc #63, branch `63-swarm-units`, not on main):
  `src/simulation/swarm.py`, section 6.

---

## 2. The unit in three layers

```
  HOST, at spawn and on rare events            DEVICE (and the Python reference), every tick
  ------------------------------------         ------------------------------------------------
  character sheet  --compile step-->  traits   --compose (tick head)-->  eff  --> every law
  (stats, kit, rings,                 (static                (x statuses,
   the draw, the name)                 hooks)                  x meters)
```

### 2.1 The character sheet (host only)

Everything a role-playing game knows about a person: the primary stats, what
they wear and carry as ITEMS, rings, a name, a history, the random draw that
made them, what the player may see of it. It is free-form host data. It may be
redesigned at any time, because nothing in the tick reads it.

Today's sheet is small: the species' mean stats, the loadout row, the env
profile row. It is a pure function of the match spec and the seed, and it
stays one after stats are drawn (the draw is keyed by the match seed and the
unit id), so the sheet itself is never synced state: a replay rebuilds it.

### 2.2 The compile step

ONE host function, `unit_traits.compile_unit(sim, u)`, turns the sheet into the
unit's HOOKS: a fixed block of Q16 multipliers, `Unit.traits`, each `ONE` when
neutral. It runs at spawn, at a conversion to zombie, and later at any event
that changes the sheet (an item picked up or dropped). It is the only writer of
`traits`.

A hook is THIS INDIVIDUAL'S FACTOR on a class value that a table already
holds. The table row stays the class value (`[speeds] marine_walk_mps`, the
weapon row's `reload_seconds`, the env profile's `stability`); the hook says
how this person differs from the class. So a row keeps its meaning, tuning a
class stays a config edit, and a unit with every hook at `ONE` is exactly
today's unit.

"Strength gives melee damage" is a line of `compile_unit`. Changing it to
"strength and dexterity" is a host edit; no law, no kernel and no trained
network changes shape.

### 2.3 The unit row

What the tick reads and writes. Section 4 lists the kinds of state it holds.

### 2.4 The compose step (D5)

At the head of every tick, before slot 1a, `unit_traits.compose_all(sim)`
writes each living unit's EFFECTIVE block, `Unit.eff`:

    eff[h] = traits[h]  x  (the terms of its statuses)  x  (the terms of its meters)

Every law reads `eff`, never `traits` and never a stat. Today there are no
status terms and no meter terms, so `eff == traits == ONE` everywhere.

Rules of the compose step:

- **Frozen for the tick.** A status applied in slot 4c changes a multiplier
  from the next tick's head. This is the #169 principle (decide against the
  start state) and it is what makes the device form one small kernel.
- **Status FLAGS stay live.** Whether a unit can act, move or aim
  (`status.composed_flags`, the verb gates) is today's law and is untouched.
  The compose step is for multipliers only.
- **A term is a table entry.** "Slowed by 30 %" is a status-kind row saying
  which hook it scales and by its magnitude; adrenaline's effect on sprinting
  is a row saying which hook the meter scales and by what curve. Never a
  branch per status in a law.
- `eff` is not hashed: it is a function of the synced state at the tick head.
  A unit that comes into being or is converted mid-tick is composed at once.

---

## 3. The hook list v1

The hooks are the contract between the RPG layer and the engine. v1 holds a
hook only where a law reads it today.

| hook | scales | read at | neutral today |
|---|---|---|---|
| `HP` | the class's hit points (the species' vitality; `[zombie] hp` for a zombie) | spawn (`Simulation.add_unit`), the conversion, `stats.effective_vitality` (the panel, `bots_v1`) | `ONE` |
| `SPEED` | the walk and full speed (and a zombie's cadence) | the movement law, slot 3 (`movement.move_speed_q16`, the zombie step) | `ONE` |
| `TURN` | the top turning speed and the angular acceleration | `Simulation.unit_turn_rate_q16` / `unit_turn_accel_q16` (so the predictor and `ticks_to_within` follow) | `ONE` |
| `HOLD` | the stability (holding against a push, the fall limit) | `unit_impulse.stability_q16` | `ONE` |
| `MELEE` | the damage and the impulse of a melee blow, a shove and a bite | `combat.melee_blow`, the shove, the zombie bite | `ONE` |
| `SPREAD` | the shot cone's half-angle | the cone draw in slot 4 | `ONE` |
| `HANDLING` | the duration of a reload, a throw's wind-up and a plant | `combat.reload_begin`, `_start_throw`, `_start_plant` | `ONE` |

Mass stays its own column (`mass_g`, integer grams): it is a quantity, not a
factor, and the compile step already writes it (`unit_impulse.bind_mass`).

Rules:

- A hook is a Q16 int in `[ONE / 8, 8 x ONE]`, refused outside at the compile
  step's door (the bound is the kernels' integer headroom, not game design).
- The hook enters its law INSIDE the law's existing single rounding. At `ONE`
  the law's result is bit-identical to today's; U1 proves it (section 7).
- A zombie keeps its body's hooks: the conversion re-runs the compile step on
  the same sheet with the zombie state, so a strong crewman is a strong
  zombie (the 2026-05-20 brainstorm's principle).
- A new hook is a row of this table: a column of the block, a term in ONE
  law, and that law's device twin, in one patch. Candidates left out because
  no law is ready for them: throw reach (the action spec's `arg` head is
  defined on the row's range), sight and hearing (perception has no per-unit
  law; the training sight rule is `all_round`), recoil, getting up, carry
  capacity, air use, the stress gains.

What the agents see: observation spec v1 and action spec v1 are defined
against TABLE values, and at neutral hooks nothing they read changes. Hooks
that differ from `ONE` are a rule change, switched on between agent
generations; whether a soldier perceives its own or another's hooks is a new
observation spec version (the game AI lead's, #192).

---

## 4. The kinds of state in a unit row, and how each grows

| kind | what | today | how it grows |
|---|---|---|---|
| identity | id, slot, team, `is_zombie`, life state | built | -- |
| kit | row numbers into the shared tables, and counts | species, loadout, env profile, weapon, melee, charge; magazines, grenades by kind | an INVENTORY is a fixed number of slots, each (item row, count), with a rule for a full pack; an item's effect is a row of numbers read by the compile step, never code |
| traits | the hook block (section 3) | U1 | a hook per law |
| pools and meters | a level with a cap or a decay | `hp_q`, the exposure block, the push law's `effort` | stamina, the air reserve, adrenaline, strain: each one Q16 column in the Exposure law's shape (raised by events, changed each tick in integers, dormant at zero, hashed while non-zero), each feeding the compose step through a table of terms |
| conditions | discrete, timed or lasting | the status block (9 kinds), `is_zombie`, `flashlight_on` | a new timed condition is a status-kind row (a drug is one); WOUNDS are a small fixed block, one level per body part, each level a term of the compose step |
| body state | position, facing, the turning and push state | built | -- |
| control state | the held verbs, the latches, the order, the zombie's horde state | built | a new verb per the Match recording and Action spec rows |

The rules that keep any future column cheap:

1. **Integers, fixed size.** A collection has a capacity in the schema and a
   counted refusal at its door (the status block and the charge table are the
   precedents).
2. **A unit refers to things by id or row number**, never by reference.
3. **An effect is data.** An item, a buff, a drug, a wound and a stat all act
   through the compile step or the compose step. If a new idea seems to need
   its own code on one unit, it needs a new hook or a new law for everyone.
4. **A new column is a schema row** (the Unit schema row of CLAUDE.md); it is
   free until a law reads it, and each law that reads it has a device twin.
5. **Not on the unit, ever:** a name, a look, animation state, the item list
   as items, experience and campaign progress, `awakened`'s display rule. They
   are the sheet's or the renderer's.

---

## 5. The stats (direction agreed; each part designed in its own session)

Not built by this document. Recorded so the sessions start from one page.

- **Each stat acts through an existing law** (D3):

  | stat | acts on |
  |---|---|
  | strength | `MELEE`; later shove, holding (`HOLD`), carry capacity, throw reach |
  | agility (big movements) | `TURN`, `SPEED` at a sprint, `HOLD` (balance), getting up |
  | dexterity (fine movements) | `SPREAD`, `HANDLING` |
  | endurance | the stamina pool and its recovery, air use, how fast strain builds |
  | vitality | `HP`, resistance to poison |
  | perception (new) | the sight cone and awareness radius, hearing |
  | intelligence | technical verbs (hacking, defusing) when they exist; no law yet |
  | will | how much a real event raises stress, and the recovery from it |
  | imagination (hidden) | how much the UNSEEN raises stress: darkness, a noise without a source, being alone, an enemy last seen and lost |
  | will orientation (hidden) | which events count: the unit's own danger, or its team-mates' |

- **Agility and dexterity are two stats** because arousal moves them in
  opposite directions (#200: big movements improve with adrenaline, fine ones
  get worse).
- **Imagination gets its law from the stress system.** With #200's two levels
  (a fast one and a slow one, performance best in the middle), an imaginative
  unit with a strong will reaches a fight already near its peak, and one with
  a weak will is past the peak before the first shot: the metaphysics notes'
  "same faculty, opposite outcome", with no special code. A Gray's seed image
  is a stressor that is not there. Cocaine is a status row that raises the
  fast level's rest value and hides the slow one.
- **No separate size stat.** Mass and the footprint carry it.
- **The draw** (#199): at spawn, keyed by the match seed and the unit id
  through `MatchDraws` with its own `Salt`; a bounded bell shape from a sum
  of uniform words (so the tails are clamped by construction); correlations
  (heavier is stronger) through a shared factor with authored weights, never
  a matrix decomposition; all integer. The sheet stays a pure function of the
  spec and the seed.
- **Order of the sessions** (each: the law on the Python reference, Erik
  plays, then the kernel catches up, then a new agent generation): drawn
  stats with `HP` and `MELEE` first (#199's bite band), stamina, stress with
  will and imagination (#200), wounds, inventory and equipment.

---

## 6. Full units and swarm units (D6, D7)

### 6.1 What each is

| | full unit | swarm unit (arc #63) |
|---|---|---|
| is | a marine, a zombie, a civilian, any large animal: something that decides, carries, is commanded, or is played | a larva, an insect: one of many, uniform, with no decisions and no inventory |
| count per game | tens | thousands |
| row | about 75 columns, about 200 integers with the blocks | 9 columns |
| store | `(N, U, columns)`, a fixed roster by (team, slot), P6 | `swarm_<species>_<col>` arrays per species on the `GameMap`, stable slots, a high-water mark |
| reference | Python `Unit` objects and Python laws; the device kernels are training-only twins (Class T) | a C++ kernel per species with a bit-identical CUDA twin; the played game runs it |
| laws | many, some order-dependent (id order, the serial applier) | one uniform kernel per species, order-free |
| digest | per column, absent at its default (`UNIT_SECT_V2`) | a presence-gated section over `[0, high_water)` (`SWARM_SECT_V1`) |
| traits | the hook block from a character sheet | the species table row; no per-unit sheet |

The criterion is the swarm chapter's own (engine/17 section 1, blessed):
"homogeneous populations whose full per-agent state fits a fixed schema and
whose per-tick behavior is uniform and branch-light are swarm units; agents
with orders, plans, inventory, or dialogue stay CPU units". P6 moving the full
unit's row to the device does not make it a swarm unit.

### 6.2 Why not one store with two schemas

- **The reference differs.** A full unit's law is Python first and a kernel
  later (the snapshot model); a swarm law is C++ and CUDA together. One store
  would have to serve both authorities.
- **The lifecycle differs.** A fixed roster with slots kept at death, against
  spawn and reclaim with a high-water mark.
- **The width differs by twenty times.** A larva in a full-unit row would
  spend about 200 integers on a thing that needs 9; at thousands of larvae
  times N games that is the memory and the bus.
- **Both digests are pinned.** Merging the stores moves both sets of goldens
  for no gain in the game.

### 6.3 What they share (one of each, never a second)

| shared | rule |
|---|---|
| the batch axis | ONE N, supplied by `ResidentBatch` (device) or by one `GameMap` per env (farm). A store declares per-game arrays; `ResidentBatch` stacks every `GameMap._RESIDENT_FIELD_NAMES` entry to `(N,) + shape` and binds slice e to env e. **The swarm store drops its leading `N_ENV` axis** (arrays `(max_units,)` per map, `(N, max_units)` in a batch); its kernels are born `(N, max_units)` with `blockIdx.y` the env, per the Batched resident engine row. `SWARM_SECT_V1`'s bytes need not change (it hashes one env's extent). |
| reset and the bank | the swarm arrays are resident fields, so a reset by mask and the bank's template copy (P8a) cover them with no swarm-specific path |
| randomness | `draws.MatchDraws` keyed by the match seed, counters `(tick, index, salt, id)`, salts in `rng_salts.Salt`. The swarm's planned `StreamSalt` enum, `match_key(sim.rng)` and `total_tick` (`docs/swarm_P3_impl.md` section 6) are replaced by `Salt` members, the `MatchDraws` key and its clock |
| arithmetic | `fixed_point.h`, the device kit, the `*_fixed.py` doors |
| events | `tick_events` on the host, the event ring on the device (P6) |
| tables on the device | the table upload (design 4.11) carries the swarm species table as it carries the weapon and status tables |

### 6.4 The boundary

**No law of one store reads or writes a column of the other.** They meet in
two ways only.

1. **Per-tile fields.** Each side stamps what the other needs onto the grid
   and the other reads it like any field: the units' occupancy (slot 6's
   stamp, existing), a corpse / food plane (engine/17 section 8), the swarm's
   per-tile count plane (engine/17 section 7, "the closure mechanism"), and
   every physics field (heat and blast reach larvae through the fields they
   already read).
2. **Effect records.** A law that wants to change the other side appends a
   small integer record to a per-game queue with a schema capacity; the
   RECEIVER applies the queue at one named place in its own tick, in record
   order, through its own door:

   | from | record | applied by | through |
   |---|---|---|---|
   | larva bites a marine | (tile, species, strength) | the unit side, in the attack resolve | `status.apply_status(POISONED)` on the unit standing on the tile, and a `DamagePacket` if the row says so |
   | larvae infest a corpse | (tile) | the unit side, at the tick end's conversion | the conversion door (`convert_marines_to_zombies`'s law), on the dead unit lying on the tile |
   | a unit hurts larvae directly (a stomp, a round) | (tile or disc, amount) | the swarm kernel's pre-step | the species' own damage law |

   The record order is defined by the sender's order (slot order for larvae,
   id order for units), so the result does not depend on thread timing.
   Each record type arrives with the patch that builds its behaviour; none
   is built here.

Credit: a swarm unit has no entry in the unit id space. A hit or a status
from a swarm unit is credited to the environment (`attacker_id` -1) with its
own `events.Cause` row. If per-larva credit is ever wanted it is a design
line then, not a shared id space now.

### 6.5 What this asks of each arc

- **Arc #63:** drop the `N_ENV` axis (6.3); revise P3's section 6 against
  `draws.py` and `rng_salts.py`; amend the Swarm store row's last sentence
  ("`N_ENV` is the codebase's first batch-axis constant ...") to "the batch
  axis is `ResidentBatch`'s; never a second N". Bite and infestation are
  effect records (6.4), designed with their behaviour patch.
- **P6:** nothing swarm-specific. The unit store is its own; the effect-record
  queue shape is the one the event ring and the serial applier already need.

---

## 7. Patch plan

### U1: the hooks at neutral (before P6; reference only; Opus)

Builds, on the Python reference:

1. `src/simulation/unit_traits.py`: the hook enum and the table of section 3;
   `compile_unit` (today: every hook `ONE`; `mass_g` through the existing
   `bind_mass`); `compose_all` and its one line in the conductor at the tick
   head; the read `eff(u, hook)`.
2. Schema rows: `traits` (group S, a block, default all `ONE`, so absent from
   every digest today) and `eff` (declared, not hashed).
3. Each law of section 3 reads its hook inside its own rounding.
4. `stats.effective_vitality` and the spawn / conversion HP go through the
   `HP` hook.

Gates:

- **Sole cause, the P3a way:** the unit records and the field digests of the
  pinned matches, tick by tick, on main and on the branch, are EQUAL (no pin
  and no golden moves; the recording format does not change).
- **Each hook is live:** for each hook, a scene where setting it to 2 x `ONE`
  on one unit changes exactly that law's outcome in the expected direction
  and by the expected ratio (a property per hook, each failing if the law
  ignores its hook).
- **A scan:** no law site of section 3 reads `traits` or a stat directly.
- **A zombie keeps its hooks:** a unit with non-neutral hooks, converted,
  has the same block.
- The full suite; Class P is untouched unless a traced file moves (then
  `cuda_green`).

No HUMAN-TEST: nothing a player can see changes.

### P6, re-planned (after U1)

The design's P6 row (store, bridge, eight slots as kernels, the applier, the
event ring, the entity export) was written on 2026-11-07 for a tick that has
since gained slot 1a (impulse and turning), 1b' (orders), 1c (exposure), 4a
(melee) and 11 (the team picture). As P3a was (design 4.4a), P6 is re-planned
against the code before any agent starts, and cut into patches that each end
on a gate:

| patch | builds | gate |
|---|---|---|
| P6a | the device unit store `(N, U, columns)` from the schema (the `traits` and `eff` blocks included), the Unit <-> store bridge both ways, the `valid` column, the compose kernel | the bridge is the identity every tick of the pinned matches (rows -> units -> rows, and the unit digest computed from the rows equals the reference's) at N = 1 and in a batch |
| P6b | the per-unit-row kernels with no order dependence: exposure decay, the status tick, the impulse integration and the turn, movement, the coupling rows; the event ring | live lockstep: each device slot against the host slot on the bridged units, every tick, tol 0 |
| P6c | the per-env serial applier, the topology seam as device functions, burn-through, the tick end (conversion, the `done` kernel), the entity export for the digest | live lockstep and the topology scenes |

The exact slot table is written at the P6 boundary, from `Simulation.step` as
it stands then. The design's P6b (entities on the device) keeps its place
after these.

---

## 8. What changing the unit costs later

So that "the last chance to change the unit" is read correctly.

| the change | the cost |
|---|---|
| a new column | a schema row; free until a law reads it |
| the stat list, a formula from stats to hooks, an item, a ring, a class | host only (the sheet and the compile step) |
| a new status kind or a new term on an existing hook | a table row |
| a new hook, or a hook on another law | one law on the reference and its device twin |
| a new meter (stamina, adrenaline) | its law on the reference, its table of terms, its device twin |
| what agents see or may do | a new observation / action spec version and a new agent generation |
| the shape itself (integers, fixed sizes, effects as data, compose once per tick, two stores) | everything; this is what is fixed today |

---

## 9. Accepted gaps

- ACCEPTED GAP: a multiplier changed by a status acts from the next tick's
  head, up to one tick (1/24 s) late.
- ACCEPTED GAP: v1 has no hook for sight, hearing, throw reach, recoil or
  carry; each arrives with the law that reads it.
- ACCEPTED GAP: the stats in `stats.py` / `species.py` keep today's list and
  values until #199's session; U1 does not touch them.
- ACCEPTED GAP: swarm hits are credited to the environment, not to a larva.
- ACCEPTED GAP: WEGO and onephase read the hooks where they share a law with
  the continuous path and are otherwise not considered (WEGO is legacy).

---

## 10. Systems

**(a) Existing canonical systems this design uses:** the Unit schema (a new
attribute is a schema row); Unit (`is_zombie` is a state, one class); the
Speed table, the Turning law, Unit impulse, Weapons, Magazines, Grenade
kinds, Planted charge, Melee verb (each keeps its table row as the class
value and gains one hook term); the status system (`status.py`, flags live);
Exposure (the shape of a meter); Loadouts / suits; Match draws (`MatchDraws`,
`Salt`); the Tick conductor (one call line for the compose step); the Batched
resident engine and `ResidentBatch` (the one N); the Coupling table; Credited
cause; the Observation and Action specs (unchanged at neutral hooks).

**(b) New systems, with draft rules** (written into CLAUDE.md by the patch
that builds them):

- **Unit traits** (`src/simulation/unit_traits.py`, U1). Draft rule: "How one
  unit differs from its class is a HOOK: a Q16 multiplier in `Unit.traits`,
  written only by `compile_unit` (spawn, conversion, a sheet change), on a
  class value a table row holds. Laws read the composed `Unit.eff`, written
  once per tick at the tick head by `compose_all`; never `traits`, never a
  stat. A stat, an item, a wound, a drug or a meter acts through the compile
  step or the compose step, as data. A new hook is a row of `HOOKS`, one term
  in one law inside its rounding, and that law's device twin; at `ONE` the
  law is bit-identical."
- **The store boundary** (this document, section 6.4). Draft rule: "Full
  units (the unit store) and swarm units (the swarm store) are two stores. No
  law of one reads or writes a column of the other; they meet through
  per-tile fields and through effect records applied by the receiver at one
  named place through its own door. The batch axis is `ResidentBatch`'s;
  no store declares an N."

---

## 11. Critique lenses (one critic at a time; verdict written here before the next)

| lens | verdict | findings and resolution |
|---|---|---|
| 1. determinism and residency fit | | |
| 2. systems reuse and scope | | |
| 3. durability: the game's planned systems and the agents | | |
