# Rung 1, patch 1e: the verbs' gates as read-only functions (the note)

2026-11-12. Written for the orchestrator of rung 1 (#195) from the plan
(`docs/rl_agent_rung1_plan_2026-11-11.md` §3, §5 "1e"), the design
(`docs/rl_agent_design_2026-11-11.md` 2.1, 2.4, 2.5, 2.7, 10 "Verb gates",
11) and the critique (lens 3 M1), checked line by line against the code at
main `5bff4c53` (P3a merged: `src/simulation/unit_schema.py`,
`Unit.__slots__`). Same shape as the plan's 4.1-4.3 and the 1d note.
**CHOICE:** marks where the design is silent; **STOP:** marks a design/code
contradiction this note does not resolve (one, at the end).

**Design:** 2.4 (the masks: one gate per hand verb, "called by the verb and
the mask", the reload's from the read-only magazine status, the one-sided
promise). **Why now:** 1h's masks call these functions; if the mask had its
own copy of a gate the two would drift (lens 3 M1: the first draft's
paraphrase was already wrong).

## What the code does today (the facts the build rests on)

**Issue and consumption.** A controller's verb at tick T runs before
`sim.step()`; the step consumes it at the same `sim.tick` (the counter
advances only at `simulation.py:2379`). The order inside `_step_pre_physics`:
round head `:1991`, **slot 1a** `unit_impulse.integrate_all` `:2004` (may
apply KNOCKED_DOWN, `unit_impulse.py:613`; never removes a status), **slot
1b** `_consume_direct_intents` `:2011`, 1c exposure `:2016`, 2 projectiles
`:2019`, **2b** `tick_statuses` `:2035` (sweeps the statuses the previous pass
zeroed, `status.py:334`, then counts down), 3 movement, 4 shooting, **4a**
`process_melee_intents` `:2073`, 4c resolve `:2088`.

**The 1b envelope** (`simulation.py:1291-1307`): the loop acts only for a unit
that is possessed, alive and not a zombie. Every hand verb but melee adds the
unit to `_possessed_ids` when issued (`:1172`, `:1190`, `:1211`, `:1124`,
`:1161`), and nothing ever removes one, so for those verbs "possessed" is
granted by issuing; `melee_intent` does not possess (`:1213-1236`) and needs
not to (4a reads every unit). A dead or zombie unit's latches are **not
cleared** there (the `continue` at `:1307` only drops trigger, wind-up and
plant).

**Per verb, the decision as the code makes it:**

| verb | latch / consumed | the decision, in code order | side effects on refusal |
|---|---|---|---|
| throw (kind k) | `throw_grenade_intent` `:1086` (kind refused by name `:1108`) / 1b `:1314-1317` -> `_start_throw` `:1358` | envelope; `throw_release_tick >= 0` -> no `:1367`; `plant_done_tick >= 0` -> no `:1369`; `grenade_counts[k] <= 0` -> no `:1371`; not `can_act` -> no `:1373` | none |
| plant | `plant_charge_intent` `:1126` / 1b `:1322-1325` -> `_start_plant` `:1411`, **after** `_advance_throw` `:1318` | envelope; `plant_done_tick >= 0 or throw_release_tick >= 0` -> no `:1419`; `has_explosive <= 0 or not charge_ammo` -> no `:1421`; not `can_act` -> no `:1423` | none |
| use | `use_intent` `:1163` / 1b `:1329-1331` -> `_direct_use` `:1496` | envelope; the first live door (`door_at`, `door.alive`) over the footprint and its one-tile ring `:1503-1510`. **No `can_act`** (oversight, design 11) | none |
| reload | `reload_intent` `:1174` / 1b `:1334-1336` -> `_reload` `:1256` | envelope; weapon row exists and `mag_size > 0` `:1259-1263`; **`mag_settle`** `:1264` (False while a stall runs); full `:1266`; `spare_mags == 0` `:1268`; then `reload_begin` `:1270`. **No `can_act`, no wind-up / plant check** (oversight, design 11) | **`mag_settle` writes the unit** (`combat.py:752`): binds a never-bound magazine, completes a stall that has run out -- see STOP |
| shove | `shove_intent` `:1192` / 1b `:1344-1348` | envelope; `can_act` `:1346`; in `unit_impulse.shove` `:889`: cooldown `:897-899`; target `shove_target` `:859` (none -> nothing, no cooldown spent) | none |
| melee | `melee_intent` `:1213` / **4a** `combat.process_melee_intents` `:1955` | latch consumed for every unit `:1978`; alive, not zombie, `melee_id` `:1979`; `can_act`, no wind-up, no plant `:1981-1983`; cadence `:1984-1987`; target `melee_arc_target` `:1923` (none -> nothing, no cadence spent) | none |

`combat.mag_status` (`combat.py:836`) reads the magazine "as `mag_settle`
would leave it" without writing: `(rounds, spares or None, reloading)`, or
`None` for an untracked weapon. It is exactly the settled state `_reload`
tests: `rounds >= mag_size` = full, `spares == 0` = no spare (`None` =
untracked spares, which reload: `reload_begin` `combat.py:808-815` starts a
stall without spending one), `reloading` = `mag_settle` returned False.

**Every one of these verbs announces its own act** (the property tests use
this): throw `ThrowEvent` `:1377`, plant `PlantEvent` `:1426`, shove
`ShoveEvent` `unit_impulse.py:912`, melee `MeleeEvent` `combat.py:2004`,
reload `ReloadEvent` from `reload_begin` (also emitted by the automatic
reload in slot 4, `mag_spend`: tests keep the trigger released), use: no
event, it flips the door's synced `want_open` (`entities/door.py:87`).

## Build

### 1. The module and the functions

New `src/simulation/verb_gates.py` (CHOICE: one module for the six, so 1h
and the `CLAUDE.md` row have one place; each function delegates to its
law's own module). Every function is a **pure read**: no write, no settle,
no event, no draw, no noise, no cool-down. Return **bool** (CHOICE: the
design's masks are legal / not legal; no reason code is asked for).

| function | returns True when (exactly the code's decision above) |
|---|---|
| `can_throw(sim, u, kind) -> bool` | `u.alive`, not `u.is_zombie`; `u.throw_release_tick < 0`; `u.plant_done_tick < 0`; `u.grenade_counts[kind] > 0`; `composed_flags(u).can_act`. Refuses an unknown `kind` by name through `sim.weapons_tables.hand_grenade_kind(kind)`, as the verb does |
| `can_plant(sim, u) -> bool` | alive, not zombie; `u.plant_done_tick < 0`; **`u.throw_release_tick <= sim.tick`** (see 2); `u.has_explosive > 0 and u.charge_ammo`; `can_act` |
| `use_door(sim, u)` -> the door or `None` | the scan of `_direct_use` `:1501-1510` moved here verbatim: the first `sim.door_at(ty, tx)` with `door.alive`, same loop order |
| `can_use(sim, u) -> bool` | alive, not zombie, `use_door(sim, u) is not None` |
| `can_reload(sim, u) -> bool` | alive, not zombie; `w = sim.weapons_tables.weapons.by_name.get(getattr(u, "weapon_id", ""))`; `w is not None and w.mag_size > 0`; `s = combat.mag_status(u, w, sim.tick)`; `not s[2]` (not reloading) and `s[0] < w.mag_size` (not full) and `s[1] != 0` (a counted spare, or untracked spares) |
| `can_shove(sim, u) -> bool` | alive, not zombie; `can_act`; `unit_impulse.shove_cooled(sim.impulse_table, u, sim.tick)` |
| `can_melee(u, tick, tables, *, at_tick_start) -> bool` | alive, not zombie, `u.melee_id`; can act; no wind-up; no plant; `last_blow_tick < 0 or tick - last_blow_tick >= rof_interval_ticks` of `tables.weapons.by_name[u.melee_id]` -- the two moments in 2 |

Supporting changes, each a pure factoring with no change of behaviour:

- `unit_impulse.shove_cooled(table, u, tick) -> bool` = the test at
  `unit_impulse.py:897-898`; `shove()` calls it in place of those lines.
- `status.survives_sweep(st) -> bool` (`st.remaining_ticks > 0`), used by the
  sweep at `status.py:334`, and a keyword `composed_flags(unit, *,
  after_sweep=False)` that skips statuses that do not survive the next sweep
  (default False: today's function, its no-status fast path unchanged).

The alive / not-zombie terms repeat the 1b envelope inside each gate so the
mask gets the whole answer; at the verb's call site they are always true, so
they change nothing there (CHOICE). Possession is not a term: issuing the
verb grants it (melee needs none). Signatures differ for melee only because
`process_melee_intents` holds `(units, tick, tables)`, not the sim; 1h calls
`can_melee(u, sim.tick, sim.weapons_tables, at_tick_start=True)`.

**No gate for the trigger or the flashlight** (CHOICE): the design masks
neither (2.4: the trigger's two options are always legal; 2.6: the
flashlight's entry is reserved and always masked).

### 2. When each gate is read: mask == effect

The mask (1h) is made from the state at the **start of tick T** (after tick
T-1's step, before slot 1a); the verb decides at its consumption point. The
gates are written so that one function answers both, exactly, except for the
events in between that can only **forbid** a verb (a slot-1a knockdown, a
slot-2 blast, a slot-2b damage-over-time death, a target that moves away).
So the design's one-sided promise holds: **gate False at tick start => the
verb issued at T does nothing.** Between start and 1b nothing else moves the
state these gates read (1a only adds or refreshes KNOCKED_DOWN; displacement
is slot 3; statuses sweep at 2b).

- **Slot-1b verbs (throw, plant, use, reload, shove):** the gate read at
  tick start equals the gate read at 1b, with one formulation choice:
  - **plant** is decided after `_advance_throw` (`:1318`) has released or
    dropped a wind-up whose release tick is T. At that point
    `throw_release_tick` is either -1 or > T (a wind-up kept, or one started
    this tick at T + 7), so the code's `>= 0` equals `> T` there. Written as
    `> sim.tick` ("busy" = `throw_release_tick > sim.tick`), the same
    function is also right at tick start, where a wind-up releasing at T
    still reads T (CHOICE; with `>= 0` the mask would forbid a plant the
    game performs on a throw's release tick, breaking the one-sided
    promise). A dropped wind-up (unit cannot act) needs no term: the plant's
    own `can_act` then fails too.
  - throw's own checks (`:1367`, `:1369`) and plant's own `plant_done_tick`
    run before their own advance: tick-start state, as written.
- **Melee (4a)** reads state after 1b, 2 and 2b. Two things there can
  **allow** what tick start forbids: a status at `remaining_ticks == 0`
  (still forbidding at tick start and at 1b, swept at 2b), and a wind-up or
  plant ending at T (released, planted or dropped at 1b). So `can_melee`
  takes the moment as a required keyword (CHOICE: one function, each term
  beside its twin, so a change of the law is made once):
  - `at_tick_start=False` (the verb, at 4a): `can_act =
    composed_flags(u).can_act`; wind-up busy = `throw_release_tick >= 0`;
    plant busy = `plant_done_tick >= 0` -- today's literal terms.
  - `at_tick_start=True` (the mask): `can_act =
    composed_flags(u, after_sweep=True).can_act`; wind-up busy =
    `throw_release_tick > tick and composed_flags(u).can_act` (kept by
    `_advance_throw` `:1390-1404` only if not past and able to act at 1b);
    plant busy = `plant_done_tick > tick and composed_flags(u).can_act`
    (`_advance_plant` `:1431-1445`).
  A unit holding a wind-up or a plant is always possessed (only the 1b loop
  sets them), so the advance has always run by 4a.
- **What stays one step early (accepted gap, design 2.4):** a gate True at
  tick start may still do nothing (knocked down at 1a, stunned by a slot-2
  blast; for shove and melee, no target: whether anyone is in reach is not a
  gate term -- a miss costs nothing, `unit_impulse.py:901`,
  `combat.py:1988-1990`). A latch is sent on a decision's first tick only
  (`control_policy.py` latch rule): a cool-down that runs out on tick 2 of a
  decision is 1h's business, not 1e's.

### 3. The verbs call the gates (refactor only)

One decision, two callers. Every check keeps its order; every side effect
(events, noise, cool-downs, the settle) stays where it is.

- `_start_throw` `:1358`: lines `:1367-1374` become `if not
  verb_gates.can_throw(self, u, throw.kind): return`; the rest unchanged.
- `_start_plant` `:1411`: lines `:1419-1424` become `if not
  verb_gates.can_plant(self, u): return`.
- `_direct_use` `:1496`: `door = verb_gates.use_door(self, u)`; `if door is
  not None: door.want_open = not door.want_open`.
- `_reload` `:1256`, keeping the settle exactly where it was:
  ```
  ready = verb_gates.can_reload(self, u)       # pure, before the settle
  weapon = ...by_name.get(getattr(u, "weapon_id", ""))
  if weapon is None or weapon.mag_size <= 0:
      return
  mag_settle(u, weapon, self.tick)             # unchanged write, same cases
  if ready:
      reload_begin(u, weapon, self.tick, events=self.tick_events)
  ```
  Equal to today because `mag_status` before the settle is by definition the
  state after it (`combat.py:836-856` against `:752-767`), and `mag_settle`
  runs in exactly the cases it ran before.
- shove, 1b `:1344-1348`: `if verb_gates.can_shove(self, u):
  unit_impulse.shove(...)`; `shove()` keeps its own cooldown test through
  `shove_cooled` (tests call it directly).
- melee, `combat.py:1979-1987`: `if not verb_gates.can_melee(u, tick, tables,
  at_tick_start=False): continue`, then `weapon = ...`, the target and the
  blow as today. `verb_gates` is imported inside the function (the
  `from simulation import unit_impulse` precedent at `combat.py:1973`; it
  imports `combat` for `mag_status`).

### 4. What stays out

- **The two oversights are kept exactly** (design 11, Erik's to rule on): a
  stunned (or knocked-down, paralysed) unit can reload and can use a door;
  reload also ignores a wind-up and a plant. `can_reload` and `can_use`
  carry no `can_act` term, and tests pin that, so a later ruling is a
  deliberate test change.
- **Design wording read as the code** (CHOICE): 2.4's reload row says "a
  counted spare exists"; the code also reloads with UNTRACKED spares (a
  loadout without `magazines`, `combat.py:808-815`). The gate follows the
  code (design 2.4: "as the game's code gates them today").
- **No new unit state** (P3a merged: `Unit.__slots__` would refuse it), no
  schema row, no digest change, no recording format change (no intent kind
  added or changed), no config key. The latches stay where they are.

## Gate

1. **Lockstep, before and after, tick for tick.** Commit A, before any
   source change: `tests/_1e_lockstep.py`, a dump / `--compare` driver on
   the model of `tests/_p3a_sole_cause.py` (it imports that file's four
   SCENARIOS and adds one). Its tape records per tick `(tick,
   tick_digest(capture_snapshot(sim)), field_digest(...), the schema's unit
   records, synced events)` -- `field_ab_harness.capture_snapshot` (fields,
   units, charges, entities: a door's `want_open` is hashed) and
   `field_digest.tick_digest`, so the door verb is covered. The new
   `gates` scenario drives every verb directly, on
   `test_horde_behaviours._door_level()` (an unwired interior door) plus
   room for the others, and must reach **every refusal edge** below at least
   once and every act at least once; it prints, per verb, issues and acts
   (by event, or `want_open` flips), so a vacuous script shows:
   - use beside the door (open, then close), from three tiles away (nothing),
     while stunned (acts: the oversight);
   - reload: never-bound full magazine (refused; the settle binds), partial
     magazine with spares (acts), during the stall, on the stall's last tick
     (refused; the settle completes it), dry with no spare, while stunned
     (acts);
   - throw: every kind once, a second throw during a wind-up, a throw on a
     wind-up's release tick, a throw during a plant, a kind at count 0;
   - plant: one that acts, one during a wind-up, one on a wind-up's release
     tick (acts), one during a plant, one with no charge left;
   - shove: on cool-down, one tick before and on the tick it runs out, while
     stunned (the rifle butt stuns its target), by a zombie;
   - melee: during a wind-up, during a plant, on the cadence boundary, a
     whiff, and a blow on the tick the target's stun reaches
     `remaining_ticks == 0` at tick start (acts);
   - every verb on a dead unit and on a zombie.
   Run commit A's driver on commit A (main's game) to a dump outside the
   repo (`C:/tmp/breach_1e/before.pkl`, never committed); after the
   refactor, the same driver to `after.pkl`; `--compare` must say EQUAL for
   all five scenarios.
2. `tools/match_pin_check.py` EQUAL (CPU; cuda / resident if built here).
3. `pytest tests -q`, whole, green (incl. `test_magazines.py`,
   `test_melee_verb.py`, `test_unit_impulse.py`, `test_grenade_kinds.py`,
   `test_plant_charge.py`, `test_bots_v1.py`, `test_zombie_control.py`).
4. `tools/twin_manifest_gen.py` regenerated with a `[not_twinned]` row for
   `src/simulation/verb_gates.py` ("Class T: the hand verbs' gates, read at
   slots 1b and 4a (#195 1e); the device's at the residency arc's P6"); then
   `tools/twin_debt.py` and `tests/test_class_p_level.py` decide whether
   `tools/cuda_green.py` is needed. Expected not: every file touched
   (`simulation.py`, `combat.py`, `status.py`, `unit_impulse.py`) is a
   `[not_twinned]` row today (`tests/twin_manifest.toml:85,113,118,124`).

**Recommended commit order (makes the property tests able to fail):**
A the lockstep driver + its dump; B `verb_gates.py`, `shove_cooled`,
`survives_sweep` / `after_sweep` and the property tests -- green while the
verbs are still the OLD code, which proves each gate equals the old
decision; C the verbs call the gates -- tests still green, lockstep EQUAL;
D `CLAUDE.md` and the twin manifest. Break one gate on purpose once
(`can_plant` with `>= 0`; `can_melee(at_tick_start=True)` without
`after_sweep`) and report that the named test fails.

## Property tests (`tests/test_verb_gates.py`)

The harness: a continuous arena (as `_p3a_sole_cause.scenario_scripted`),
the door level for use; for a state S set up at tick start, read the gate,
issue the verb alone (one hand verb per tick, as a decision issues), step
once, and read the act from the verb's own event (or the `want_open` flip);
for shove and melee a target is placed in reach and arc, so the target term
is held true. Each test's docstring states the property and what breaks it.

| property | breaks if |
|---|---|
| for every verb, over a sweep of states with no forbidding event in the step: gate at tick start == the verb acted | a gate term differs from the verb's decision |
| reload: False on an untracked weapon, a full magazine, a never-bound magazine (full), during the stall and on its last tick, dry with no spare; True on a partial magazine with counted spares and with untracked spares | the gate reads the raw fields instead of `mag_status`, or "a counted spare" is read as the design's wording |
| every gate is pure: the unit's schema record (`unit_schema.digest_record`), its statuses and `tick_events` are unchanged by calling any gate, on a never-bound magazine and on a finished stall included | a gate settles, binds, or emits |
| plant on a throw's release tick: gate True at tick start, the plant acts | the plant gate reads `throw_release_tick >= 0` |
| throw on a plant's last tick and on a wind-up's release tick: gate False, nothing thrown | the throw gate reads `> tick` |
| melee on the tick a STUNNED reaches `remaining_ticks == 0`: shove gate False and no shove; melee gate (`at_tick_start=True`) True and the blow lands | the mask reads the 4a moment wrong, or the shove gate projects the sweep |
| melee with a wind-up releasing at T, and a plant ending at T: gate True, the blow lands; a wind-up dropped at 1b by a status that the 2b sweep releases: gate True, the blow lands | the busy terms ignore the advance at 1b |
| one-sided: with a slot-1a knockdown (a pending impulse over the fall limit) or a blast in slot 2, gate False => nothing; gate True may still do nothing | a gate allows less than the game in some state |
| cool-downs at their boundary: shove at `tick - last == cooldown - 1` False, `== cooldown` True; melee the same at `rof_interval_ticks`; a whiff spends neither | an off-by-one, or a miss spends a cool-down |
| the oversights, kept: a stunned unit's `can_reload` and `can_use` are True and the verbs act | the oversights are "fixed" without Erik's ruling |
| a dead unit and a zombie: every gate False and no act | the envelope is missing from a gate |
| `can_throw` refuses an unknown kind by name | the gate accepts what the verb refuses |
| `composed_flags(u, after_sweep=True)` equals `composed_flags` of the same unit after one `tick_statuses`, for no-DoT statuses | the sweep predicate is stated twice |
| the lockstep gate above | |

## `CLAUDE.md` (in the patch)

- **New row "Verb gates"** (Gameplay, after "Planted charge"), from design
  10's draft: `src/simulation/verb_gates.py` (`can_throw`, `can_plant`,
  `can_use` / `use_door`, `can_reload`, `can_shove`, `can_melee`) -- whether
  a hand verb would act if issued now is ONE pure read per verb (no settle,
  no event, no draw), called by the verb at its consumption (slot 1b; melee
  at 4a) and by the policy's masks (1h) at the start of the tick; exact at
  both moments except for events that can only forbid (the one-sided
  promise); melee takes the moment (`at_tick_start`); the target is never a
  term (a miss costs nothing); the reload's reads `mag_status`; the two
  oversights (stunned reload, stunned use) are kept, Erik's to rule on; a new
  verb brings its gate in the same patch, never a mask's copy.
- **Amend** "Magazines" (the verb's decision is `verb_gates.can_reload` over
  `mag_status`; the verb settles, then starts iff it said yes), "Melee verb"
  (`can_melee`), "Unit impulse" (`can_shove`, `shove_cooled`), "Grenade
  kinds" (`can_throw`), "Planted charge" (`can_plant`); one clause each. The
  header line.

## STOP

**The forced reload moves the digest when it is refused.** The design's 1h
gate (plan §5: "a masked choice issued by force leaves the match's digests
where a do-nothing row leaves them") contradicts the reload's code in two
common states. `_reload` calls `mag_settle` before deciding
(`simulation.py:1264`), and `mag_settle` writes: on a never-bound magazine it
binds `current_mag` and `spare_mags` (`combat.py:739-750`; both hashed while
not at their defaults, `unit_schema.py:515-518`), and on a stall that has run
out it fills the magazine and clears `reload_done_tick` (`combat.py:762-766`).
Every unit is never-bound until its first trigger, so a reload forced at a
match's start on a full gun is masked (`can_reload` False), does nothing in
the game, and still changes the unit digest. The game is the same either way
(`mag_status` reads both states alike; the next trigger settles the same
values), but the digests are not. 1e as written keeps the settle where it is
(the brief: every side effect unchanged), so `can_reload` is exact in the
game's sense and not in the digest's. Two ways out, the orchestrator's call:

- (a) **Recommended.** A separate last commit of 1e (or its own patch):
  `_reload` settles only when `ready` (the gate first, then `mag_settle` +
  `reload_begin`). Lockstep then differs only in the magazine columns, only
  on ticks between a refused reload in those two states and the unit's next
  trigger, with every field digest and every outcome equal (a sole-cause
  proof as P3a's). No existing pin moves: the bots never reload, and the
  P3a arena's one reload is issued on a bound, partial magazine. Needs the
  orchestrator's yes, since it moves digests the brief says must not move.
- (b) Keep the settle; 1h's gate compares reload states through
  `mag_status` rather than raw digests (a narrower promise, written into
  1h's note).

**For 1h, not a STOP:** a verb issued for a dead or zombie unit leaves its
latch set forever (the envelope's `continue` at `simulation.py:1307` does not
clear `pending_*`, which P3a hashes while set, `unit_schema.py:556-563`); the
gates say False, the game does nothing, the digest moves. 1h's decoder must
not issue hand verbs for dead rows (design 6.3's dead rows), and its forced
test should run on living rows.
