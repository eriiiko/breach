# Unit design v1 -- critique 3: durability (the game's planned systems and the agents)

**Date:** 2026-11-14. **Document:** `docs/unit_design_2026-11-14.md` (v1).
**Lens:** durability: can the shape of sections 2-4 carry the systems Erik has
asked for without a shape change? Critiques 1 and 2 not read (independent
lenses). Read: the document; issues #199, #200 (+ comment); 
`docs/breach_metaphysics_design_notes.md`; `docs/architecture/mechanics/01_units.md`
sections 2, 4, 6, 7 and "Eriks additions"; `docs/unit_variants_design_brainstorm.md`;
`docs/rl_agent_design_2026-11-11.md` 3.1-3.3, 6.1, 7.1, 9; CLAUDE.md rows named
in the brief; `unit_schema.py`, `status.py`, `damage.py` (`mitigate_q`,
`mitigation_for`), `species.py`, `obs_spec_v1.py` (tables, `ObsContext`,
`build_vector`, `_kit_values`), `action_spec_v1.py` (`ActionTables`),
`reward_v1.py` (`bar_ratio`, `standing`), `control_bots.py` (`focus_key`,
`_weapon_spread_q`).

## Verdict: SOUND WITH FIXES

The core of the shape -- a host sheet, one compile step, a fixed block of
per-unit factors read inside existing laws, meters in the Exposure law's
shape, fixed-capacity blocks, two stores -- carries most of what Erik wants
(drawn stats and bands, stamina, the hidden stats as stress gains, a zombie
that keeps its body, big units, civilians, animals). It fails in five places
that are cheap to fix on paper now and expensive after U1 and P6a bake them in:
the compose formula is product-only and per-meter separable (it cannot carry
#200's own inverted U, the document's own cocaine example, armour, or a leg
wound that removes the sprint); the hook contract binds laws but not READERS
(the observation, the bots, the predictor), so non-neutral hooks make v1
observation columns wrong, not merely blind; compose at the tick head makes
every between-tick reader one tick stale; section 8 misprices what a new
status kind or item row costs the agents; and the shape has no place for
per-unit perception, which the Gray seed image, tunnel vision and
imagination's "unseen" stressors all need.

---

## Findings

### 1. The compose formula cannot carry the effects it is meant for -- MAJOR

**Wanted.** (a) #200: "performance follows the classic inverted U ... A long
one pushes the slow level up until the SAME adrenaline lands the unit past the
peak". The peak's position in adrenaline depends on strain. (b) The document's
own section 5: "Cocaine is a status row that raises the fast level's rest value
and hides the slow one." (c) Armour as equipment ("different armors ... all
standard equipment of an rpg", 01_units "Eriks additions"). (d) A leg wound or
an empty stamina pool that takes away the sprint but keeps the walk.

**Where the shape fails.** Section 2.4: `eff[h] = traits[h] x (the terms of its
statuses) x (the terms of its meters)`, each meter a separate "curve".
- (a) A product of per-meter curves `f(a) x g(s)` has its maximum in `a` at the
  same place for every `s`: strain can lower the curve but never move its
  peak. The wanted effect is a function of the two levels JOINTLY
  (e.g. of `a + k s`).
- (b) A rest value is a level, not a factor (a rest of 0 times anything is 0),
  and "hides the slow one" acts on a meter's INPUT to the compose step, not on
  a hook; no constant factor cancels a curve of a varying strain.
- (c) `damage.mitigate_q` is `max(0, a - max(0, armor - ap)) x resist`, and the
  armour comes from the species profile: `HUMAN` is `NEUTRAL_MITIGATION`
  ("armor 0, resist 1.0", `species.py`). Section 2.2 defines a hook as "THIS
  INDIVIDUAL'S FACTOR on a class value that a table already holds"; a factor on
  a class armour of 0 is 0. Flat armour per damage type is a QUANTITY, as
  `mass_g` is -- and the document treats mass as a one-off exception ("Mass
  stays its own column ... it is a quantity, not a factor"), not as a category
  a status or a wound may then act on.
- (d) The `SPEED` hook "scales the walk and full speed" together (section 3),
  and the Speed table law is `f(theta) v_walk + throttle (v_full - v_walk)
  max(0, cos theta)`. Removing the sprint is a cap on the full part, or a
  factor on `v_full - v_walk` alone; neither exists. (Section 5 already maps
  agility to "`SPEED` at a sprint", which no v1 hook expresses; see 7.)

**Smallest change on paper.**
1. Term KINDS in the term table: `factor`, `offset`, `cap` (min / max), applied
   in a fixed order (offsets, then factors, then caps), each a table row. The
   product-only compose is the special case.
2. A term's INPUT may be a fixed linear combination of meter levels named in
   the table (`arousal = a + k s`), or a 2-D curve; and a status term may scale
   a meter's level as seen by the compose step (the cocaine "hides" case).
3. Name COMPILED QUANTITIES as a category beside hooks (today `mass_g`;
   armour per damage type, carry capacity, a pool's cap next): written only by
   `compile_unit`, read by their law, and reachable by offset / cap terms of
   the compose step. Armour then moves from the species profile into this
   block, the species profile its class value.
4. Split `SPEED` into `WALK` and `SPRINT` (the throttle's extra part) while
   both are neutral (finding 7).
The P6a compose kernel is built against this table, so the term kinds must be
decided before P6a, not after.

### 2. The hook contract binds laws, not readers: v1 observation columns become wrong -- MAJOR

**Wanted.** #199 draws per unit; Erik: "all they may see is the final stats of
the units they control" (01_units, Eriks additions); the obs spec's rule
"Everything the bots have" (rl design 3.1) and "Every 'time left' is a
fraction of its own table duration" (3.1).

**Where the shape fails.** Section 3: "at neutral hooks nothing they read
changes. Hooks that differ from `ONE` are a rule change, switched on between
agent generations." The gate list of U1 scans only "law site[s] of section 3".
But the observation, the bots and the HUD also read the class values the hooks
scale, straight from the tables:
- `obs_spec_v1._kit_values`: `reload_left = ratio(u.reload_done_tick - tick,
  weapon.reload_ticks)` -- the table duration; with `HANDLING` 1.5 the column
  starts at 1.5, and the column is declared unsquashed (`("reload_left",
  "reload_ticks", False)`, vector col 28, entry col 14). The same for
  `throw_left` over `ctx.throw_windup_ticks` (cols 42 / 16) and `plant_left`
  over `ctx.plant_ticks` (47 / 19). With hooks up to `8 x ONE` (section 3's
  bound) these columns reach 8, against the spec's own rule that a ratio above
  1 is squashed and capped.
- Vector col 56 `spread` is `weapon.spread_deg / ref_spread_deg`: a soldier
  with `SPREAD` 2 is told the class spread.
- The entry's `hp` is `hp / bar` (the bar = the hp at creation, so it already
  carries the `HP` hook) and reads 1.0 for a 100 HP and a 200 HP enemy alike,
  while the bots' focus fire reads ABSOLUTE hit points (`control_bots.focus_key`:
  `float(e.current_hp)`). Once HP is drawn, the bots have what a soldier does
  not -- the 3.1 rule broken by turning a hook on.
- The bots' trigger tolerance reads the table spread through a cache keyed by
  weapon (`_weapon_spread_q`), so it cannot follow a per-unit `SPREAD`.
- Meanwhile `omega / unit_turn_rate_q16(u)` and `ticks_to_on` DO follow `TURN`
  (they go through the per-unit dial). So under non-neutral hooks v1 is
  internally inconsistent: some ratios absorb the hook, some do not.
So switching hooks on is not "a rule change" alone: under obs v1 it corrupts
columns a trained network relies on.

**Smallest change on paper.**
1. One accessor per hooked class value, returning the EFFECTIVE quantity
   (`eff_reload_ticks(u, weapon)`, `eff_spread(u, weapon)`, ... -- the pattern
   `Simulation.unit_turn_rate_q16` already follows), and the rule: every
   reader -- law, observation, bot, HUD, mask -- reads a hooked value through
   it. U1's scan covers readers, not only law sites.
2. Then the v1 columns keep their stated meaning ("a fraction of its OWN
   duration") with no spec version, and `spread` reads the soldier's own.
3. Name, in section 3, the obs version that adds an own / team-mates' `eff`
   block (what Erik said ML teams see) and Erik's ruling owed on enemies (#199
   open question "does the player ever learn a unit's strength"), and that
   hooks other than `HP` stay neutral on the training path until it exists.
   Whatever is ruled for enemies, the bots get the same (e.g. `focus_key` on
   bars) so 3.1 holds.

### 3. Compose at the tick head makes every between-tick reader one tick stale -- MAJOR

**Wanted.** The bots' trigger and the agents' trigger aid both use the
turning law's predictor: "A controller that needs to know where this tick's
shot goes reads `facing_after_turn_q16`, the pure evaluation of exactly slot
1a's step ... never a copy of the law" (Turning law row); obs `ticks_to_on` is
"the law ITERATED". The observation and the masks are built at the end of a
step for the next decision.

**Where the shape fails.** Section 2.4: compose runs "At the head of every
tick, before slot 1a", and "A status applied in slot 4c changes a multiplier
from the next tick's head." Controllers, the observation and the masks run
BETWEEN ticks (the `MatchSession` tick is controllers' `handle_frame`, then
`step`). After any status or meter change at tick t, they read `eff` composed
at the head of t, while slot 1a of t+1 turns with `eff` composed at the head
of t+1. The predictor is then wrong for one tick, silently -- invisible at
neutral hooks, so U1's gates cannot catch it; it appears with the first status
term on `TURN`.

**Smallest change on paper.** Make the composed `eff` the one the NEXT tick
will use and the one between-tick readers see: compose at the tick's END
(after slot 11, skipping status entries the next sweep removes, per the status
system's lazy expiry) and at birth / conversion; or give readers a pure
`compose(u)` that evaluates what the next head will write. Either keeps "frozen
for the tick".

### 4. Section 8 misprices what a new kind costs the agents -- MAJOR

**Wanted.** Drugs as status rows, wounds, items and rings in quantity, new
grenades.

**Where the shape fails.** Section 4: "a new timed condition is a status-kind
row (a drug is one)"; section 8: "a new status kind or a new term on an
existing hook -> a table row". But obs v1 lays the soldier's vector out by
`N_STATUS_KINDS`: `_cols(15, [status_left[k] for k in range(N_STATUS_KINDS)])`
then `_cols(24, ...)`, with `assert V == 97` and the positions checked at
import. A tenth status kind shifts every later column: obs v1 fails at import,
i.e. a new spec version and a new agent generation. Likewise
`HAND_GRENADE_SLOTS = 6` ("a new spec version" in `ObsContext.build`) and the
action spec's hand head, which appends one entry per grenade row (head size
changes). And every kind the spec embeds by row index (`class` = the loadout
row, `weapon` = the weapon row) grows with the item tables: an RPG item list
of many rows does not fit "a learned vector per kind".

**Smallest change on paper.** Section 8 gets the true row: "a new status kind,
a grenade row, a loadout / weapon / item row the specs enumerate -> a new obs
(for grenades also action) spec version and agent generation." And a rule for
every FUTURE spec version: statuses, wounds, items and meters are shown through
fixed-capacity slot blocks described by their table NUMBERS (which hooks they
move and by how much), not by per-kind columns or per-row embeddings, so a new
row costs no spec version again.

### 5. No place for per-unit perception -- MAJOR

**Wanted.** The Gray seed image: "the unit manifests the false reality -- sees
allies as enemies, believes a safe room is breached"; imagination with a weak
will "fires at shadows, behaves as if a distant xeno is adjacent"; with a
strong will "accurate threat anticipation"; manifestation "sensing a hidden
Gray" (metaphysics 4.1, 4.2, 4.4). #200: tunnel vision ("width of the sight
cone"), "Perception. Stress narrows or corrupts the observation ... fewer list
entries, a narrower window, stale records". Section 5's own imagination
inputs: "darkness, a noise without a source, being alone, an enemy last seen
and lost". Will orientation: "the unit's own danger, or its team-mates'".

**Where the shape fails.**
- All of these change what ONE unit perceives. The observation is built per
  TEAM (`TeamKnowledge.build(sim, picture, team)`), the team picture is per
  team, and sight under the training rule is shared. A phantom or a
  misidentified ally has no home: it is not a multiplier, not a meter, not a
  status flag, and statuses are read only by laws.
- "Sees allies as enemies" also needs hostility to be per-unit; today it is
  `a.team != b.team` in many places (the melee arc: "never a teammate").
- Section 5 resolves this by narrowing: "A Gray's seed image is a stressor
  that is not there", "Imagination gets its law from the stress system". That
  quietly fixes the Gray hook as stress-only, a choice the metaphysics notes
  list as open ("The Gray hook mechanic", "deliberately deferred").
- Two of imagination's four inputs are currently not allowed as sim inputs:
  light is "NOT SIM-PURE UNTIL #81 ... no rule may read it before then" (Light
  accessor / Radiation sweep rows), and the team picture is "Not synced, not
  hashed, in no digest or recording". Section 5 does not name either
  dependency.
- Will orientation's split needs per-unit attribution of events (my danger
  vs a mate's): perception-gated raises ("a teammate dying in sight") need a
  per-unit sight relation that the shared-sight training rule does not
  produce.

**Smallest change on paper.**
1. Reserve a PERCEPTION kind of state in section 4's table: per-unit
   perception conditions (a fixed block: kind, magnitude, ticks, source, and
   for a phantom a small fixed record -- a believed position and class),
   consumed only by the observation builder and the bots' read of the picture,
   never by a physics law; plus one hostility predicate door (`hostile(a, b)`,
   today `a.team != b.team`) that every law and builder calls.
2. In section 5, mark the Gray hook and imagination's effects as "stressor
   now; perception distortion reserved (the metaphysics session's)", and name
   the inputs' dependencies: darkness after #81; "last seen and lost" either
   a per-unit column the meter law keeps itself, or the team picture becoming
   synced state.
3. State where a stress law's perception events come from (credited events on
   `tick_events`, a near-miss event from the round's march, the sight oracle
   per unit) -- a new law for everyone, as rule 3 of section 4 already allows.

### 6. Mid-match sheet changes are host-only on a path that will not have the host -- MAJOR (durability side only)

**Wanted.** "items picked up and dropped during a match", carry weight
(01_units 7, Eriks additions); a zombie that keeps its body (every training
match converts).

**Where the shape fails.** Section 2.2: `compile_unit` is "ONE host function"
running "at spawn, at a conversion to zombie, and later at any event that
changes the sheet (an item picked up or dropped)", and the sheet is
"free-form host data". Section 7's P6c moves "the tick end (conversion ...)"
to the device. Every mid-match sheet event is then a host round trip inside a
device tick; and stat-dependent items (a ring of +2 strength feeding a
nonlinear strength -> `MELEE` formula) can only be compiled on the host.

**Smallest change on paper.** (a) Compile at spawn every block the unit can
need in the match: the human block AND the zombie block (the conversion
selects, never compiles). (b) Split items in section 4: SHEET items (rings,
stat-changing gear) compiled between matches or on host-in-the-loop paths
only; MATCH items (picked up / dropped in play) carry a fixed row of hook
terms and quantity offsets and enter through the compose step from the
inventory slots, so the device composes them. Erik's note that ML teams are
"equipped by us" suggests most RPG gear is sheet-level, which this split
makes explicit.

### 7. Hooks bundled where the wanted systems pull them apart -- MINOR

`SPEED` scales walk and full together (section 3) but section 5 gives agility
"`SPEED` at a sprint", #200 improves "sprint speed" with arousal, and stamina
or a leg wound should remove the sprint (finding 1d). `MELEE` scales "the
damage and the impulse" together, while 01_units 4 makes physical output
`strength x sqrt(mass / 80)` and knockback a function of mass -- strength and
mass want separate factors. **Fix:** at U1, while neutral, `WALK` + `SPRINT`
and `MELEE_DAMAGE` + `MELEE_PUSH`; splitting later re-twins kernels and
changes a hook's meaning under a trained generation.

### 8. Design invariants are checked on class values only -- MINOR

The Speed table door "holds Erik's rules -- the zombie sprint within 70..90 %
of the marine walk and ABOVE `backward_frac x walk`". Hooks are bounded only by
"the kernels' integer headroom, not game design" (`[ONE/8, 8 x ONE]`). A drawn
zombie at `SPEED` 1.2 or a marine at 0.8 breaks an Erik ruling silently; #199's
bands are designed on the effective values too. **Fix:** per-hook, per-class
design bounds in config (the draw's clamp), checked at the compile door, and
each table door that holds a cross-class rule evaluates it at those bounds.

### 9. A pool's cap and mid-life changes are undefined -- MINOR

`HP` is "read at spawn ..., the conversion"; there is no stored max ("the cap
is `effective_vitality`", 01_units 9; "No max-HP clamp on heals in v1",
`status.py`). An HP ring picked up, a wound that lowers the cap, or a heal has
no rule for what happens to `hp_q`; stamina and the air reserve will need caps
too. The reward's bar stays well defined (the hp at creation, clipped at one
bar, `reward_v1.bar_ratio`). **Fix:** a pool's cap is a compiled quantity
(finding 1.3) with one stated rule for a cap change (current hp unchanged and
clipped, or scaled), and the bar remains the creation value.

### 10. Wounds and cross-unit effects need one line each -- MINOR

- "WOUNDS are a small fixed block, one level per body part": the body-part set
  differs by species (a dog, the octopus of milestone 8, a big unit); say the
  slots are indexed by the species row. A hit needs a location law in the
  attack resolve, and a wound needs one writer door, as `hp` has
  `apply_packet` (a medic healing a mate's leg is then a packet).
- Leadership calm, contagion, auras, intimidation (#200) affect OTHER units.
  They fit as "a law for everyone that raises a meter or applies a status on
  others through the receiver's door", but section 4 does not say so, and
  section 6.4's receiver-door rule is written only between stores. Add the
  same rule within the unit store, with neighbour reads against the tick-head
  values (the #169 principle the document already cites).

### 11. Panic as loss of control has no named home -- MINOR

#200's comment opens "freeze, flight, fight", refusal and dropping a weapon.
Freeze fits the live status flags (2.4). Flight and charge need a law that
writes a unit's held verbs over its controller's. **Fix:** a "control
override" condition in section 4's control-state row: when set, the override
law writes the held verbs, and the action masks collapse to one legal option
per head (the action spec's existing "a head with ONE legal option"), so the
trainer is not credited with actions that did not run.

### 12. Smaller wording points -- MINOR

- "Not on the unit, ever: ... experience and campaign progress": #200's
  habituation is WITHIN a match ("Repeated exposure within a match ... lowers
  the raise") and must be a unit meter; scope the ban to between-match
  experience.
- Inventory slots "(item row, count)" leave no room for per-instance data
  (randomised affixes, durability), standard in RPG loot; reserve an instance
  column (an id into a per-match item table of fixed capacity), or rule it out.
- Section 5 decides "No separate size stat" -- the variants brainstorm's
  deferred decisions 1-2, where size is "The original axis"; and the cocaine
  mechanism. Mark both as examples for their sessions, not decisions.

---

## Tested, and the shape carries it

- #199 drawn stats and hits-to-kill bands: the draw on the sheet, `HP` and
  `MELEE` hooks, the band designed on the joint distribution host-side;
  measured by the existing benches. (Bounds: finding 8.)
- A zombie that keeps its body: compile on the same sheet in the zombie
  state; the reward bar is re-taken at turning (the team picture's `bars`).
- #200 gross up / fine down: per-hook curves of opposite sign on `TURN`,
  `HOLD`, `MELEE`, `SPEED` vs `SPREAD`, `HANDLING` (given finding 1's joint
  input).
- #200 two levels, fast and slow: two meters in the Exposure law's shape.
- #200 stressors from events and from the environment (O2, heat, pressure):
  events on `tick_events`, fields through the coupling rows.
- #200 threshold breakdowns that stop acting (freeze): a meter law applies a
  status; flags stay live.
- #200 stress in the reward (option 2) and in action sampling (option 3):
  a meter is a column the reward books and the trainer can read; nothing in
  the tick changes.
- Hidden stats as gains: imagination, will, will orientation compiled into
  the stress law's gain hooks (self vs mates as two hooks); visibility rules
  stay on the sheet; will-orientation drift is between matches, on the sheet.
- Awakening: a sheet flag, or a condition if it acts mid-match.
- Stamina: a meter with endurance-compiled gains and cap (given findings 1
  and 9).
- Drugs as timed conditions: status rows with terms (given findings 1 and 4).
- Carry weight: compiled into `mass_g` and an encumbrance factor; read by the
  impulse law as mass already is.
- Equipment weapons: the weapon row is the class value, the person's hooks on
  top; a swap re-runs compile.
- Big units: `footprint` is already an integer column (`unit_schema`), mass a
  quantity; per-footprint horde fields and A* grids are masks / goal sets of
  existing systems, not unit shape.
- Civilians and animals: a loadout without a weapon, an env profile, a
  controller, a team; a stance table (if wanted) is mission data, not unit
  shape.
- The action spec v1 with v1 hooks: throttles are fractions, masks come from
  the verb gates (which read no hook); a future throw-reach hook needs an
  action spec version, as section 3 already says.
- The reward's bar under drawn HP: per bar, creation value, unchanged.
- The bots' turning predictor with `TURN`: it reads the per-unit dial (given
  finding 3).
- Frozen for the tick: no wanted system needs a factor to change inside a
  tick; the only cost is reader alignment (finding 3).
- Swarm boundary for effects on units (larva bites, infestation): effect
  records at the receiver's door.
