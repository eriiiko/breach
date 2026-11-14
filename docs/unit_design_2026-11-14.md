# The unit: character sheet, compile step, unit row -- and the swarm boundary

**Date:** 2026-11-14. **Issues:** #29 (residency, patch P6), #63 (swarm units),
#199 (drawn stats), #200 (stress physiology).
**Status:** v2. v1 was written after the design session with Erik of
2026-11-14 and read by three critics, one lens each (section 12; all three
SOUND WITH FIXES). v2 resolves their findings and adds Erik's poison and
infestation rulings of the same day (section 6). Decisions marked AGREED are
Erik's ("I agree with everything you just wrote"); RULED are his rulings; LEAD
are Claude's technical calls, open to his veto.
**Replaces** `docs/architecture/mechanics/01_units.md` as the description of
what a unit is (Erik, 2026-11-14: that chapter is "extremely outdated -- only
thing we can take from it are the ideas and concepts"). The ideas kept from it,
from `docs/breach_unit_class_design.md`, `docs/unit_variants_design_brainstorm.md`
and `docs/breach_metaphysics_design_notes.md` are named where they are used.

This document fixes the SHAPE of a unit so that the game can grow stats,
stamina, stress, poison, wounds and equipment for years without the shape
changing. It designs none of those systems in detail, bar poison's frame
(section 6); each gets its own session.

---

## 0. The decisions

| # | decision | who |
|---|---|---|
| D1 | A unit is a ROW of integer columns plus rows of shared tables; behaviour is LAWS that run over all units at a fixed place in the tick. No per-unit code, no per-unit growing list. | AGREED |
| D2 | **The line.** The RPG layer (stats, equipment, rings, the random draw, names) lives on the HOST as a character sheet. A COMPILE STEP, run at spawn, turns the sheet into per-unit values: the HOOKS (multipliers) and a few compiled quantities. The device unit holds pools, meters, conditions, kit counts, control state and those values. No law and no kernel ever reads a stat. | AGREED |
| D3 | Every stat acts through a law that already exists (speed, turning, the push law, the shot cone, handling times), never through a dice roll of its own. | AGREED |
| D4 | Prepare now, draw later: the hooks go in before P6 at neutral values (every hook 1.0), so nothing in the game changes and no digest moves; the kernels read per-unit values from their first day. | AGREED (the timing was delegated) |
| D5 | Every value a hook touches is read through ONE accessor, by every reader: the law, the bots, the observation, the panel. What the accessor returns is fixed for a whole tick: it is a function of the unit's state at the END of the previous tick. | LEAD (v2, from the critique) |
| D6 | **Full units and swarm units stay two stores with two schemas and two sets of laws.** They share one plumbing (the batch axis, the residency path, reset, the draws, the fixed-point kits). **They meet only through per-tile fields.** | LEAD |
| D7 | There is ONE batch axis N, supplied by `ResidentBatch` / `BatchedEnv`. A store declares per-game arrays and never an N of its own; the swarm store drops its `N_ENV` axis. | LEAD |
| D8 | **Poison is a law of its own, not a status row:** two levels per unit (depot, blood), mass-conserving, with per-unit rates. Gas poison and larva bites both feed it. It is built on the reference before P6 and ported with P6's per-unit kernels, so it is in before training. | RULED (Erik, 2026-11-14) |
| D9 | Order of work: U1 (the hooks at neutral), the poison law, then P6 in three patches (section 8). | LEAD |

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
  starting hit points; `stats.compute_effective_stats` returns the base
  unchanged and seven of its eight accessors have no caller.
- **Three notions of full health exist:** the species' vitality (every spawn,
  a spawned zombie included), `[zombie] hp` (a CONVERTED zombie only),
  `TeamPicture.bars` (the reward's bar). `[marine] hp = 100` is in config.toml
  and nothing reads it. All are 100 today.
- **Class values live in tables:** `[marine]`, `[zombie]`, `[speeds]`,
  `[impulse]`, `[env_profiles.*]`, `[loadouts.*]`, `[weapons.*]`, `[ammo.*]`,
  and the zombie state's overlays selected on `is_zombie` (`ZOMBIE_STABILITY`,
  `ZOMBIE_MITIGATION`, `[impulse] zombie_hold_mult`).
- **Per-unit seams that already exist:** `mass_g` (body + kit);
  `Simulation.unit_turn_rate_q16` / `unit_turn_accel_q16` (read by the law,
  the predictor and the observation: the pattern D5 generalises);
  `unit_impulse.stability_q16(u)`; and `FootprintSamples`' Q16 scale members,
  composed in `movement.default_speed` for every walker on every path (the
  Footprint samples builder row: "a speed output is a Q16 scale member").
- **Precedents for what is coming:** the Exposure law (a per-unit level raised
  by events, changed each tick in integers, dormant at zero, hashed while
  non-zero, "not a status row") is the shape of every meter; the status block
  is the shape of every timed condition; `draws.MatchDraws` / `rng_salts.Salt`
  are the only source of randomness; the Coupling table (`exchange.py`) is
  how a field acts on a unit.
- **The swarm store** (arc #63, branch `63-swarm-units`, not on main):
  `src/simulation/swarm.py`, section 7.

---

## 2. The unit in three layers

```
  HOST, at spawn                               EVERY TICK (the reference and the device)
  ---------------------------------            -----------------------------------------------
  character sheet --compile step-->  traits    accessor(u) = class value (a table row)
  (stats, kit, rings,                (hooks)                 x  the unit's hook
   the draw, the name)               + compiled              (+ the terms of its statuses, items
                                       quantities               and meters, when they exist)
```

### 2.1 The character sheet (host only)

Everything a role-playing game knows about a person: the primary stats, what
they wear and carry, rings, a name, a history, the random draw that made
them, what the player may see of it. It is free-form host data and may be
redesigned at any time, because nothing in the tick reads it.

Today's sheet is small: the species' mean stats, the loadout row, the env
profile row.

**The sheet's seed (critique 1.5).** The sheet is a pure function of the
match spec and of ONE seed that belongs to the spawn, not to the episode: a
`sheet_seed` in the spec (when absent, the match's seed). The device bank
(residency design 4.6) copies a template's tick-0 state at a reset and only
changes the in-match draw key, so the soldiers of a template are fixed; the
farm must draw the same soldiers for the same template. Variety across
episodes comes from templates. #199 builds the draw under this rule.

### 2.2 The compile step

ONE host function, `unit_traits.compile_unit(sim, u)`, runs when a unit is
spawned and writes:

- the unit's HOOKS, `Unit.traits`: a fixed block of Q16 multipliers, each
  `ONE` when neutral;
- its COMPILED QUANTITIES: per-unit values that are amounts, not factors.
  Today `mass_g` (through the existing `unit_impulse.bind_mass`); later flat
  armour per damage type and the caps of pools. Each is its own schema
  column.

It is the only writer of `traits`. **It never runs inside a tick** (critiques
1.2, 3.6): on the device path there is no host inside a tick.

- **A conversion COPIES the block.** The row stays the same row, so the body
  keeps its hooks: a strong crewman is a strong zombie (the 2026-05-20
  brainstorm's principle). What differs for a zombie is the CLASS value,
  selected on `is_zombie` as today.
- **Items.** A SHEET item (worn at spawn) is compiled. A MATCH item (picked
  up or dropped during a match) acts from the inventory slots through terms
  of the accessor (2.4), never by re-running the compile step.

A hook is THIS INDIVIDUAL'S FACTOR on a class value that a table holds. The
table row stays the class value (`[speeds]`, a weapon row's `reload_seconds`,
`[marine] hp`); the hook says how this person differs from the class. A unit
with every hook at `ONE` is exactly today's unit.

"Strength gives melee damage" is a line of `compile_unit`. Changing it to
"strength and dexterity" is a host edit; no law and no kernel changes.

### 2.3 The unit row

What the tick reads and writes. Section 4 lists the kinds of state it holds.

### 2.4 One accessor per hooked value (D5)

Each value a hook touches has ONE function that returns it for a unit, in
integers, and EVERY reader calls it: the law, the bots' predictors, the
observation, the action masks, the panel. `Simulation.unit_turn_rate_q16` is
the existing example. A reader of the raw table value outside the accessor
is a bug, held by a scan.

    accessor(u) = round( class value  x  traits[hook] )        (U1: this, and nothing more)

When statuses, match items and meters gain effects, their TERMS enter the
accessor. The rules for that day are fixed now:

- **Frozen for the tick, composed at the tick END.** A term reads the unit's
  state as it stood at the end of the previous tick. So the controllers, the
  observation and the masks, which run between ticks, read exactly what the
  coming tick's laws will use (critiques 1.1, 2.8, 3.3: the Turning law's
  predictor promises the exact slot-1a value). A status applied in slot 4c
  changes a hooked value from the next tick. While the terms are few the
  accessor computes them on the read; when that costs, a COMPOSE step at the
  tick end (and at spawn and reset) stores the result in a per-unit block
  that is not hashed, with its own declared schema group, and the accessors
  read the block. Readers never change either way.
- **Status FLAGS stay live.** Whether a unit can act, move or aim
  (`status.composed_flags`, the verb gates) is today's law and is untouched.
- **A term is a table entry, of one of three kinds:** a FACTOR, an OFFSET or
  a CAP, applied in that fixed order after the hook (critique 3.1: an armour
  ring adds, an empty stamina pool caps the sprint at the walk, a slow
  multiplies). A status kind's term is a column of its `StatusKindDef` row;
  an item's a column of its item row.
- **A term reads ONE level.** A law whose effect depends on several levels
  together (#200: strain moves where adrenaline's peak lies) publishes that
  combination as a level of its own, and the term reads it. The accessor
  never contains a law.
- **Effects between units** (a leader's calm, panic spreading) are laws that
  read OTHER units' tick-end levels and write their own unit's meter. Never a
  mid-tick value of another unit.

---

## 3. The hook list v1

The hooks are the contract between the RPG layer and the engine. v1 holds a
hook only where a law reads it today.

| hook | scales (the class value) | its accessor; the sites that read it today |
|---|---|---|
| `HP` | the class's full health: `[marine] hp`, `[zombie] hp` for a zombie | `unit_traits.max_hp_q(u)`: the spawn, the conversion, the panel, `bots_v1`'s fall-back; it replaces `stats.effective_vitality` |
| `WALK` | the walk speed (a zombie's shamble) | a `FootprintSamples` scale member composed in `movement.default_speed`, with the walker's effort scale |
| `SPRINT` | the full speed (a zombie's sprint) | the same seam, on the full-speed class |
| `TURN` | the top turning speed and the angular acceleration | `Simulation.unit_turn_rate_q16` / `unit_turn_accel_q16` (existing) |
| `HOLD` | the knockdown stability (the fall limit and the effort a body stands) | `unit_impulse.stability_q16` (existing, its one reader) |
| `STRIKE` | the damage of a melee blow and of a bite | one accessor over `weapon.melee_damage` (`combat.melee_blow`) and the zombie bite's damage |
| `PUSH` | the impulse of a melee blow and of a shove | one accessor over `impulse_table.melee_push_q[weapon]` (`combat.process_melee_intents`) and `table.shove_q` |
| `SPREAD` | the shot cone's half-angle: aimed, snap and beam cones alike | one accessor over the cone selection (today duplicated at two sites in `combat.py` and one in `timeline.py`, and predicted from the raw row by both bots and by the observation) |
| `HANDLING` | the ticks of a reload, a throw's wind-up and a plant | one accessor each over `weapon.reload_ticks`, `throw_windup_ticks`, `plant_ticks`; never below 1 tick (a 0-tick wind-up is another code path) |

`SPEED` and `MELEE` of v1 are each split in two while they are neutral
(critique 3.7): a leg wound or an empty stamina pool takes the sprint and
leaves the walk; damage and impulse are different physics.

Rules:

- A hook is a Q16 int. The compile step's door refuses a value outside
  `[ONE / 8, 8 x ONE]` (the integer headroom; each accessor carries a
  one-line int64 range argument) and, for `TURN`, any value that takes the
  turning law past its 32-trial bound. DESIGN bounds per class and hook (for
  example Erik's rule that a zombie sprints at 70 to 90 % of a marine's walk)
  are #199's to set, inside that door.
- **Rounding (critique 1.3).** The class values are quantized once at their
  doors and cached. The hook is applied to the cached integer with ONE more
  integer rounding (`unit_fixed.mul_q16_round` or the law's own), which is
  the identity at `ONE`; the speed hooks enter `default_speed`'s existing
  rounding. Never a float, never a second copy of a table.
- A new hook is a row of this table: a column of the block, ONE accessor,
  and every reader through it. Its device form follows the twin ledger
  (Class T: at catch-up; the training door refuses a law in debt).
- Left out because no law is ready for them: throw reach (the action spec's
  `arg` head is defined on the row's range), sight and hearing, recoil,
  getting up, carry capacity, air use, the stress gains. Poison's three
  rates arrive with its patch (section 6).

**What the agents see.** Because the observation and the bots read through
the accessors, a soldier's own `spread`, its "time left" fractions and its
turning aid stay correct when its hooks differ from `ONE`, with no layout
change, and at neutral hooks every observation byte is what it is today. Two
things remain the game AI lead's (#192) before non-neutral hooks reach a
training run: whether a soldier is shown its own and its team-mates' hooks as
inputs (a new observation spec version), and the bots' `focus_key`, which
reads an enemy's absolute hit points where a soldier sees only the fraction
of its bar. Until then only `HP` may differ from `ONE` on a training run.

---

## 4. The kinds of state in a unit row, and how each grows

| kind | what | today | how it grows |
|---|---|---|---|
| identity | id, slot, team, `is_zombie`, life state | built | -- |
| kit | row numbers into the shared tables, and counts | species, loadout, env profile, weapon, melee, charge; magazines, grenades by kind | an INVENTORY is a fixed number of slots, each a fixed tuple (item row, count, a few integers of the item's own data), with a rule for a full pack. It replaces the unread `inventory.Inventory` stub and the WEGO belt when it is designed |
| traits | the hook block (section 3) | U1 | a hook per hooked value |
| compiled quantities | per-unit amounts | `mass_g` | flat armour per damage type; a pool's cap. A cap that falls clamps its pool at the next tick end |
| pools and meters | a level with a cap or a decay | `hp_q`, the exposure block, the push law's `effort` | poison's two levels (section 6), stamina, the air reserve, adrenaline, strain: each a Q16 column in the Exposure law's shape, each acting through accessor terms |
| conditions | discrete, timed or lasting | the status block, `is_zombie`, `flashlight_on` | a timed condition is a status-kind row; WOUNDS are a small fixed block whose slots are the species' body parts, written through one door, each level an accessor term |
| perception conditions | what a unit is SHOWN, as against what it can do | none | reserved as a kind (critique 3.5): tunnel vision, a Gray's seed image, a false contact. Read only by the observation door and the bots, through the team picture; never by a physics law. "Who is hostile to whom" is one predicate, so that a perception condition can change its answer for one unit |
| body state | position, facing, the turning and push state | built | -- |
| control state | the held verbs, the latches, the order, the zombie's horde state | built | a new verb per the Match recording and Action spec rows. Loss of control (panic) is a condition under which the unit's controller is replaced and its action masks leave one legal option; a design line of #200 |

The rules that keep any future column cheap:

1. **Integers, fixed size.** A collection has a capacity in the schema and a
   counted refusal at its door.
2. **A unit refers to things by id or row number**, never by reference.
3. **An effect is data.** An item, a buff, a drug, a wound and a stat all act
   through the compile step or through accessor terms. If an idea seems to
   need its own code on one unit, it needs a new hook or a new law for
   everyone.
4. **A new column is a schema row**; it is free until a law reads it.
5. **Not on the unit, ever:** a name, a look, animation state, the item list
   as host objects, experience and progress BETWEEN matches, `awakened`'s
   display rule. (Habituation within a match is a meter.)

---

## 5. The stats (direction agreed; each part designed in its own session)

Not built by this document; recorded so the sessions start from one page.
The mapping is a starting point, and each line is the later session's to
change.

| stat | acts on |
|---|---|
| strength | `STRIKE`, `PUSH`; later holding, carry capacity, throw reach |
| agility (big movements) | `TURN`, `SPRINT`, `HOLD` (balance), getting up |
| dexterity (fine movements) | `SPREAD`, `HANDLING` |
| endurance | the stamina pool and its recovery, air use, how fast strain builds |
| vitality | `HP`, poison's rates (section 6) |
| perception (new) | sight and hearing, when they have a per-unit law |
| intelligence | technical verbs when they exist; no law yet |
| will | how much a real event raises stress, and the recovery from it |
| imagination (hidden) | how much the UNSEEN raises stress: darkness, a noise without a source, being alone, an enemy last seen and lost |
| will orientation (hidden) | which events count: the unit's own danger, or its team-mates' |

- **Agility and dexterity are two stats** because arousal moves them in
  opposite directions (#200).
- **Imagination gets a law from the stress system.** With #200's two levels,
  an imaginative unit with a strong will reaches a fight already near its
  peak, and one with a weak will is past it before the first shot: the
  metaphysics notes' "same faculty, opposite outcome". Its inputs depend on
  two things that are not ready: light as game state (#81) and the team
  picture, which is not synced state today. The Grays' seed image and what a
  drug does to the levels are examples of what the shape must carry (a
  perception condition; an offset on a level's rest value), not decisions.
- **The draw** (#199): keyed by the sheet's seed (2.1) and the unit id
  through `MatchDraws` with its own `Salt`; a bounded bell shape from a sum
  of uniform words; correlations through a shared factor with authored
  weights, never a matrix decomposition; all integer; per-class bounds per
  hook.
- **Order of the sessions** (each: the law on the Python reference, then the
  kernel catches up, then a new agent generation): drawn stats with `HP` and
  `STRIKE` first (#199's bite band), stamina, stress with will and
  imagination (#200), wounds, inventory and equipment.

---

## 6. Poison (D8; Erik's rulings of 2026-11-14)

**The law** (Erik's, a two-compartment absorption and elimination model; the
Bateman curve):

    dx0/dt = -k0 x0 + input        the depot (a bite site, the lungs)
    dx1/dt = +k0 x0 - k1 x1        in the blood
    damage per tick = k_dmg x1     a POISON DamagePacket through damage.apply_packet

One dose rises, peaks and decays; the total damage of a dose is
`k_dmg x dose / k1`, so the dose sets the harm and `k0`, `k1` set the timing.
**Integer form:** per tick `moved = (x0 x a0) >> 16` is computed once,
subtracted from `x0` and added to `x1`; the same for what leaves `x1`. No
poison is created or lost to rounding.

**Where it lives** (the frame; the patch's own short design fills it in):

- A module of its own, `src/simulation/poison.py`, in the Exposure law's
  shape: construction-bound dials from `[damage.poison]` (the base `k0`,
  `k1`, `k_dmg`, the gas input gain, a cap on `x0`), integer, no draw,
  dormant when nothing is poisoned.
- **Columns:** `poison_x0`, `poison_x1` (Q16; zero on a fresh unit; zeroed
  at death and at conversion; hashed only while non-zero, so a match without
  poison hashes as before), and the unit that last dosed it, for the credit
  of damage that arrives long after its dose.
- **Per-unit rates** are three hooks on the three base rates, with accessors
  (section 3's rules); `compile_unit` sets them from the sheet (vitality).
  All `ONE` until #199.
- **Inputs, both through one door (`poison.dose`), both read from a field
  under the unit's footprint in the coupling slot (9c3):**
  - poison GAS: the gas density at the footprint times the gain. It replaces
    `exchange.apply_poison_dose`'s direct damage, so the poison grenade, the
    40 mm poison round and the Miasma Vent change their behaviour: ONE
    sole-cause re-baseline with its rationale written down.
  - larva BITES (section 7.4): the swarm's per-tile bite-dose field.
  A zombie takes neither (it does not breathe, and it is immune to bites).
- **The `POISONED` status** (a fixed magnitude and duration; applied by no
  shipped row) does not fit the law. Its registry slot STAYS for now:
  observation spec v1's layout is built from the number of status kinds, so
  removing one is an observation spec change. It is retired with the next
  observation spec version, which is also where a soldier is shown its own
  blood level.

**Corpse infestation** (Erik: lower priority, not needed for training). The
same shape on a dead unit: one Q16 level per corpse, raised from the swarm's
count field under the body, decaying each tick, converting through the
existing conversion door at a threshold (a second converting path beside
`killed_by_zombie`); the law deliberately runs on the dead and is zeroed at
conversion. Its column arrives WITH its law, in the larva behaviour patch
after P6: a column needs no reservation (section 9). Whether a zombie's
corpse can rise again is Erik's call then.

---

## 7. Full units and swarm units (D6, D7)

### 7.1 What each is

| | full unit | swarm unit (arc #63) |
|---|---|---|
| is | a marine, a zombie, a civilian, any large animal: something that decides, carries, is commanded, or is played | a larva, an insect: one of many, uniform, with no decisions and no inventory |
| count per game | tens | thousands |
| row | about 75 columns, about 200 integers with the blocks | 9 columns |
| store | `(N, U, columns)`, a fixed roster by (team, slot), P6 | `swarm_<species>_<col>` arrays per species on the `GameMap`, stable slots, a high-water mark |
| reference | Python `Unit` objects and Python laws; the device kernels are training-only twins (Class T) | a C++ kernel per species with a bit-identical CUDA twin; the played game runs it |
| laws | many, some order-dependent | one uniform kernel per species, order-free |
| digest | per column, absent at its default (`UNIT_SECT_V2`) | a presence-gated section over `[0, high_water)` (`SWARM_SECT_V1`) |
| traits | the hook block from a character sheet | the species table row; no per-unit sheet |

The criterion is the swarm chapter's own (engine/17 section 1, blessed):
"homogeneous populations whose full per-agent state fits a fixed schema and
whose per-tick behavior is uniform and branch-light are swarm units; agents
with orders, plans, inventory, or dialogue stay CPU units". P6 moving the full
unit's row to the device does not make it a swarm unit.

### 7.2 Why not one store with two schemas

The reference differs (Python first against C++ and CUDA together); the
lifecycle differs (a fixed roster with slots kept at death against spawn and
reclaim); the width differs by twenty times, which at thousands of larvae
times N games is the memory and the bus; and both digests are pinned.

### 7.3 What they share (one of each, never a second)

| shared | rule |
|---|---|
| the batch axis | ONE N, supplied by `ResidentBatch` (device) or by one `GameMap` per env (farm). `ResidentBatch` stacks every `GameMap._RESIDENT_FIELD_NAMES` entry to `(N,) + shape`. **The swarm store drops its leading `N_ENV` axis** (arrays `(max_units,)` per map, `(N, max_units)` in a batch); its kernels are born `(N, max_units)` with `blockIdx.y` the env, per the Batched resident engine row. `SWARM_SECT_V1`'s bytes and the recorder's rings read one env's extent and do not move (checked by critique 1) |
| reset and the bank | the swarm arrays are resident fields, so a reset by mask and the bank's template copy (P8a) cover them |
| randomness | `draws.MatchDraws` and `rng_salts.Salt`. The swarm's planned `StreamSalt` enum, `match_key(sim.rng)` and `total_tick` (`docs/swarm_P3_impl.md` section 6) become `Salt` members, the `MatchDraws` key and its clock |
| arithmetic | `fixed_point.h`, the device kit, the `*_fixed.py` doors |
| events | `tick_events` on the host, the event ring on the device (P6) |
| the twin ledger | a swarm kernel is a row of `tests/twin_manifest.toml` (Class P: `tools/cuda_green.py` at HEAD before a merge); its books a region of the books arena |

### 7.4 The boundary: fields only

**No law of one store reads or writes a column of the other. The two stores
meet through per-tile fields, and through nothing else.** This is the swarm
chapter's own rule (engine/17 section 7: one device-written per-tile field is
"the closure mechanism" for everything the unit side asks of the swarm). v1
of this document added a queue of effect records; the critique showed it to
be a second channel beside systems that exist (critique 2.5) with unsettled
order, overflow and tick-crossing rules (critique 1.7), and it is DROPPED.

| interaction | the field | who reads it, through what |
|---|---|---|
| a larva bites | the swarm kernel adds each bite's dose to a per-tile BITE-DOSE field (an integer sum per tile, so order-free; a larva bites where the unit-occupancy field shows a body) | a coupling row in slot 9c3, beside the gas rows: the sum under a living non-zombie unit's footprint goes into `poison.dose`. The same tick, no capacity, no tie rule (a tile is under one living unit) |
| larvae on a corpse | the swarm's per-tile COUNT field | the infestation law reads it under a dead unit's footprint (section 6) |
| units as seen by larvae | the units' occupancy stamp (slot 6, existing); a corpse / food plane (engine/17 section 8) | the swarm kernel, as it reads any field |
| a unit hurts larvae | the physics fields: heat today (Erik: larvae die of temperature only; a flamer is planned). A blast later: `GameMap.blast_log`, THE seam a slot-9e system reads to react to blasts (the Heat source row) | the swarm kernel |

Credit: a swarm unit has no entry in the unit id space. A dose from a bite is
credited to the environment (`attacker_id` -1) with its own `events.Cause`
row. The swarm side's bite rate and dose are larva table rows, designed with
the bite patch after P6.

If an interaction ever cannot be carried by a field, a record channel is a
design line then.

### 7.5 What this asks of each arc

- **Arc #63, now:** drop the `N_ENV` axis; revise P3's plan against
  `draws.py`, `rng_salts.py`, the Batched resident engine row and the twin
  ledger; amend the Swarm store row's last sentence to "the batch axis is
  `ResidentBatch`'s; never a second N". **After P6:** the P3 kernel; then
  the bite-dose field with the bite patch.
- **P6:** nothing swarm-specific, and no queue.

---

## 8. Patch plan

One implementation agent at a time; each patch its own branch off main.

### U1: the hooks at neutral (reference only; Opus)

Builds:

1. `src/simulation/unit_traits.py`: the hook table of section 3,
   `compile_unit` (every hook `ONE`; `mass_g` through `bind_mass`), the door
   with its bounds, and one accessor per hooked value.
2. The schema row `traits` (group S, a block, default all `ONE`, so absent
   from every digest); a conversion copies it.
3. Every reader of a hooked value goes through its accessor: the laws, both
   bots, the observation, the masks, the panel, the WEGO / onephase sites
   that share a law.
4. `max_hp_q(u)` over `[marine] hp` / `[zombie] hp`; `stats.compute_effective_stats`
   and the unread `effective_*` accessors are deleted (`stats.py` keeps the
   sheet's types).

Gates:

- **Sole cause, the P3a way:** the unit records, the field digests and every
  observation array of the pinned matches, tick by tick, on main and on the
  branch, are EQUAL. No pin, no golden and no recording format moves.
- **Each hook is live and exact:** per hook, a scene where one unit's hook is
  2 x `ONE` changes exactly that value at EVERY site of its row (for `PUSH`:
  the melee impulse and the shove; for `SPREAD`: the aimed, the snap and the
  beam cone) by the expected ratio, and the bots' predictor and the
  observation agree with the law. Each fails if a site ignores its hook.
- **The scan:** no reader of a hooked class value outside its accessor.
- **A zombie keeps its hooks:** a unit with non-neutral hooks, converted,
  holds the same block, and its values are the zombie class's times them.
- The full suite. `unit.py` is in a Class-P twin's file set (the stamp), so
  the merge needs a CUDA build and `tools/cuda_green.py` at HEAD.

No HUMAN-TEST: nothing a player can see changes.

### PX: the poison law (reference; Opus; its short design first)

Section 6's frame, a just-in-time design for the dials, the credit column and
the gas gain, then the patch: `poison.py`, the columns, the three hooks, gas
moved onto the law, the re-baseline with a sole-cause proof for everything
but the poison scenes, a bench of dose against time-to-kill for Erik's
tuning. Erik decides whether he plays it before the merge.

### P6, re-planned (after PX)

The design's P6 row was written on 2026-11-07 for a tick that has since
gained slots 1a, 1b', 1c, 4a and 11. As P3a was (residency design 4.4a), P6
is re-planned against `Simulation.step` as it stands at that boundary, and
the re-plan is written INTO the residency design as an amendment. Its cut,
as intended today (the residency design's own "P6b", entities on the device,
keeps its name and follows):

| patch | builds | gate |
|---|---|---|
| P6.1 | the device unit store `(N, U, columns)` generated from the schema (the `traits` block included), how its per-game arrays join the resident list, the Unit <-> store bridge both ways, the `valid` column | the bridge is the identity on every tick of the pinned matches (rows -> units -> rows; the unit digest computed from the rows equals the reference's), at N = 1 and in a batch |
| P6.2 | the per-unit kernels with no order dependence: slot 1a (impulse, turn), 1b / 1b' (the latches' consumption where it is per-unit), 1c exposure, 2b statuses, 3 movement, the coupling rows 9c to 9d, the poison law; the device accessors; the event ring | live lockstep, each device slot against the host slot on bridged units, every tick, tol 0 -- on the pinned matches AND on the per-hook scenes of U1 with non-neutral hooks, with a coverage counter per hook (critique 1.6: at neutral hooks a kernel that ignores its hook passes) |
| P6.3 | the per-env serial applier, the topology seam as device functions, burn-through, the tick end (conversion, the `done` kernel), the entity export for the digest | live lockstep and the topology scenes |

---

## 9. What changing the unit costs later

The residency design's section 10 is the general table. For the unit:

| the change | the cost |
|---|---|
| a new column | a schema row; free until a law reads it; the device store is generated from the schema |
| the stat list, a formula from stats to hooks, a sheet item, a class | host only |
| a new term on an accessor (a status, an item, a meter acting on a hooked value) | a table column and the accessor; its device twin at catch-up |
| a new hook, or a hook on another value | an accessor on the reference and every reader through it; its device twin at catch-up |
| a new meter or law (stamina, stress, infestation) | the law on the reference, its columns, its device twin at catch-up |
| a new status KIND, grenade row, or weapon / loadout row | the table row -- AND a new observation (and for a grenade, action) spec version: spec v1's layouts are built from the number of rows (critique 3.4). A later spec version should show statuses, wounds and items through fixed slot blocks described by their table numbers, so that a row no longer changes a layout (the game AI lead's, #192) |
| what agents see or may do | a new spec version and a new agent generation |
| the shape itself (integers, fixed sizes, effects as data, one accessor per value frozen for the tick, two stores meeting through fields) | everything; this is what is fixed today |

---

## 10. Accepted gaps

- ACCEPTED GAP: a hooked value changed by a status acts from the next tick,
  up to one tick (1/24 s) late.
- ACCEPTED GAP: v1 has no hook for sight, hearing, throw reach, recoil or
  carry; each arrives with the law that reads it.
- ACCEPTED GAP: the stats in `stats.py` / `species.py` keep today's list and
  values until #199's session.
- ACCEPTED GAP: a bite's poison is credited to the environment, not to a
  larva.
- ACCEPTED GAP: the `POISONED` status kind stays registered and unapplied
  until the next observation spec version.
- ACCEPTED GAP: on a training run only `HP` may differ from `ONE` until an
  observation spec version shows hooks (section 3).

---

## 11. Systems

**(a) Existing canonical systems this design uses:** the Unit schema (a new
attribute is a schema row); Unit (`is_zombie` is a state; one class); the
Footprint samples builder (the speed hooks are scale members of
`FootprintSamples`, composed in `default_speed`); the Speed table; the
Turning law (`unit_turn_rate_q16` is the accessor pattern); Unit impulse
(`stability_q16`, `bind_mass`, the impulse table's cached pushes); Weapons,
Magazines, Grenade kinds, Planted charge, Melee verb (their rows stay the
class values); the status system (`StatusKindDef` gains term columns when a
status first scales a value; flags stay live); Exposure (the shape of a
meter, and poison's precedent); the Coupling table (the gas and bite inputs
are rows of it); the Payload executor's gas path; Credited cause (a new
`Cause` row for a bite); Loadouts / suits; Match draws; the Tick conductor;
the Batched resident engine and `ResidentBatch` (the one N); the Class P /
Class T ledger; `GameMap.blast_log`; the Observation spec, Action spec and
Reward profile (unchanged at neutral hooks; they read through the
accessors); `TeamPicture.bars` (the reward's bar, unchanged).

**(b) New systems, with draft rules** (written into CLAUDE.md by the patch
that builds them):

- **Unit traits** (`src/simulation/unit_traits.py`, U1). Draft rule: "How one
  unit differs from its class is a HOOK: a Q16 multiplier in `Unit.traits`,
  written only by `compile_unit` at spawn (a conversion copies the block),
  on a class value a table row holds. Each hooked value has ONE accessor
  and every reader calls it -- the law, the bots, the observation, the
  masks, the panel -- never the raw row (scanned). The accessor's result is
  fixed for a tick (a function of the unit's state at the end of the
  previous tick); at `ONE` it is the class value bit for bit. A stat, an
  item, a wound, a drug or a meter acts through the compile step or as a
  term of an accessor (a factor, an offset or a cap, a table entry), never
  as code on one unit. A new hook is a row of `HOOKS` with its accessor;
  its device twin follows the twin ledger."
- **Poison** (`src/simulation/poison.py`, PX). Draft rule: "Poison is two
  per-unit levels (depot, blood) moved by one integer law that conserves
  what it moves; every poisoning enters through `poison.dose` from a
  coupling row; nothing else damages by poison. Never a status row."
- **The store boundary** (section 7.4). Draft rule: "Full units and swarm
  units are two stores. No law of one reads or writes a column of the
  other; they meet through per-tile fields only. The batch axis is
  `ResidentBatch`'s; no store declares an N."

---

## 12. Critique round (on v1; one critic at a time; resolved together in v2)

| lens | verdict | file |
|---|---|---|
| 1. determinism and residency fit | SOUND WITH FIXES | `docs/unit_design_critique_1_2026-11-14.md` |
| 2. systems reuse and scope | SOUND WITH FIXES | `docs/unit_design_critique_2_2026-11-14.md` |
| 3. durability: the planned systems and the agents | SOUND WITH FIXES | `docs/unit_design_critique_3_2026-11-14.md` |

Resolutions (finding -> where v2 answers it):

| finding | resolution |
|---|---|
| 1.1, 2.8, 3.3 compose at the tick head leaves between-tick readers stale | values are a function of the END of the previous tick (D5, 2.4) |
| 1.2, 2.9, 3.6 the compile step cannot run inside a tick | it runs at spawn only; a conversion copies the block; the zombie's difference is a class value; match items act through terms (2.2) |
| 1.3 "inside the law's single rounding" cannot be done at cached sites; SPEED overflows in `move_speed_q16` | one more integer rounding on the cached value, identity at `ONE`; the speed hooks enter `default_speed` (section 3) |
| 1.4 MELEE's push is not in `melee_blow` | `STRIKE` and `PUSH`, every site named, the gate tests each (sections 3, 8) |
| 1.5 the draw's seed against the bank's reset | the sheet's seed belongs to the template (2.1) |
| 1.6 P6's gates cannot catch an ignored hook | non-neutral scenes and a coverage counter in P6.2's lockstep (section 8) |
| 1.7, 2.5 the effect-record queue: unsettled and a duplicate | dropped; fields only (7.4) |
| 1.8, 2.7 `eff` fits no schema group; `compose_all` composes nothing | U1 builds no stored block and no conductor line; the compose step and its schema group arrive with the first term that costs (2.4) |
| 1.9 P6a-c omit slots; "P6b" collides | P6.1 to P6.3; the slot table is written at the boundary into the residency design (section 8) |
| 1.10, 2.2 the HP hook's class value | `[marine] hp` / `[zombie] hp` through `max_hp_q` (section 3) |
| 1.11 U1 needs `cuda_green` | section 8 |
| 1.12 SPREAD's cones, HANDLING's floor, TURN's bound | section 3 |
| 2.1 a second "effective stats" seam | `compute_effective_stats` and the unread accessors are deleted (section 8) |
| 2.3 SPEED beside the existing speed scale | a `FootprintSamples` member (section 3) |
| 2.4, 3.2 the hook binds laws, not readers | one accessor per value, every reader, a scan (D5, 2.4, section 3) |
| 2.6 the draft rule against the twin ledger | reworded (section 11) |
| 2.10 the `Inventory` stub | named (section 4) |
| 2.11 systems missing from (a) | section 11 |
| 2.12, 2.13 the P6 re-plan's home; the resident list | section 8 |
| 3.1 multiply-only terms | three term kinds in a fixed order; compiled quantities; a term reads one level and a law publishes a combined one (2.2, 2.4) |
| 3.4 a new kind breaks a spec layout | section 9 |
| 3.5 no place for per-unit perception | the perception-conditions kind and the one hostility predicate (section 4); section 5 names #81 and the team picture |
| 3.7 split SPEED and MELEE while neutral | done (section 3) |
| 3.8 design bounds per class | #199's, inside the door (section 3) |
| 3.9, 3.10, 3.11, 3.12 pool caps; wound slots and effects between units; loss of control; within-match experience, per-item data, examples that read as decisions | sections 2.4, 4 and 5 |
