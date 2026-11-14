# Unit design v1 -- critique 2: systems reuse and scope

**Date:** 2026-11-14. **Lens:** systems reuse and scope (the iron rule: use or
extend an existing canonical system, never build a parallel copy; and is the
design larger than its purpose). **Document:** `docs/unit_design_2026-11-14.md`
(v1). Critic 1's file was not read. Code read on main (`12a38642` tree in the
main checkout) and on `63-swarm-units` through `git show`.

**Verdict: SOUND WITH FIXES.** The shape (host sheet, compile step, a fixed
per-unit Q16 block, two stores, one N) is right and mostly built on the right
seams (TURN on `unit_turn_rate_q16`, mass on `bind_mass`, the swarm's N and
salts folded into `ResidentBatch` / `Salt`). But four of the seven hooks sit
BESIDE an existing per-unit value or seam instead of joining it, the
swarm boundary invents a record channel that the coupling table and the swarm
chapter's own occupancy bridge already cover, and the draft rule contradicts
the twin ledger. All fixes are on paper and make U1 smaller.

---

## Findings (most severe first)

### F1. A second "effective value" seam beside `stats.compute_effective_stats` -- MAJOR

**Evidence.** `src/simulation/stats.py:43-49`: `compute_effective_stats(unit)`
is the existing base -> effective compose point, documented "Modifier system
slots in here later -- wounds, zombification, encumbrance, fear, buffs all
compose here", with eight per-stat accessors (`stats.py:52-77`) "gameplay
code calls these ... so the modifier system can be slotted in transparently".
Only `effective_vitality` has callers in `src/` (`control_bots_v1.py:369`,
`ui/model.py:854`); the other seven are read only by
`tests/test_unit_class.py:288-301` (identity stubs). The design builds
`compose_all` / `Unit.eff` (doc 2.4, lines 116-139) as THE place statuses,
meters and wounds compose, and routes only `effective_vitality` through the
HP hook (doc 7, U1 item 4). Section 10 (a) does not name `stats.py` at all.

**If built as written.** Two documented homes for "wounds, encumbrance, fear,
buffs compose here", one on stats (the sheet, D2 says never read by the
tick) and one on hooks. The #199 / #200 sessions will find both promises in
the code. `StatId` (`stats.py:13-23`) still lists `MASS` and `BASE_SPEED`.

**Fix.** U1 deletes `compute_effective_stats` and the seven unread accessors
(and their identity tests, which pin a stub, not a property), keeps
`BaseStats` / `StatId` / `is_stat_player_visible` as the SHEET's types, and
replaces `effective_vitality` by the one max-HP read of F2. State in 2.1:
"`stats.py` is the sheet's vocabulary; nothing composes there".

### F2. The HP hook multiplies a STAT, not a table value; three max-HP notions -- MAJOR

**Evidence.** Doc line 150: HP scales "the class's hit points (the species'
vitality; `[zombie] hp` for a zombie)", read at `Simulation.add_unit`, the
conversion and `stats.effective_vitality`. In code the marine's HP is
`unit.current_hp = float(base_stats.vitality)` (`simulation.py:578`,
`unit.py:274`): `base_stats` is the species stat vector (group D,
`unit_schema.py:612`), which #199 makes individual. Meanwhile `config.toml
[marine] hp = 100` exists and is read by NOTHING in `src/` (only comments,
`species.py:8,98,111`, say vitality's mean "= CFG.marine.hp"). The zombie
reads its table value (`ai_zombie.py:417`, `CFG.zombie.hp`), but
`effective_vitality(zombie)` still returns the marine vitality, so the panel
and `bots_v1`'s fallback compare a zombie's hp against 100. A third max-HP
lives in `TeamPicture.bars` (`team_picture.py:261-265`, current hp at first
sight / at turning), which the observation (`obs_spec_v1.py:1229, 1625, 1651`)
and the reward read.

**If built as written.** HP = vitality x HP-hook. Once #199 draws vitality
and `compile_unit` writes HP from it (doc 5: "vitality | HP"), vitality is
counted twice; and the tick reads a stat, against D2 ("No kernel ever reads a
stat") and 2.2 ("the table row stays the class value").

**Fix.** The class value is `[marine] hp` / `[zombie] hp` (two table rows,
symmetric, one of them dead today); `base_stats.vitality` becomes sheet input
to `compile_unit` only. ONE read `unit_traits.max_hp_q(u)` = class hp by
`is_zombie` x HP, used by spawn, conversion, the panel and `bots_v1` (it
also fixes the zombie panel). Say that `TeamPicture.bars` stays the reward's
and the observation's bar (it reads the result, consistent at any HP).

### F3. SPEED sits beside the FootprintSamples scale path (and two unread per-unit speed values) -- MAJOR

**Evidence.** CLAUDE.md, Footprint samples builder row: "A speed output is a
Q16 scale member on `FootprintSamples` ... composed in `default_speed`".
`movement.py:54-76, 116-169`: `effort_scale_q16` is exactly an
already-reduced per-unit Q16 speed scale, passed by every walker's
`footprint_samples(...)` call and composed by `default_speed` with one
`half_up`, skipped at `SCALE_ONE`. Every walker goes through it: the marine
(`simulation.py:2630-2641`: both `walk_cost_q` and `full_cost_q` pass through
`default_speed`), the zombie (`ai_zombie.py:144-148`), WEGO
(`simulation.py:1682-1686`) and onephase (`timeline.py:158-162`). The design
reads SPEED at two other sites, `movement.move_speed_q16` and "the zombie
step" (doc line 151), and lists WEGO / onephase as an accepted gap (doc 9).
Two more per-unit speed values already exist: `Unit.base_speed` (group D,
species mean 1.0, sd 0.1, `species.py:117,130`; `unit_schema.py:614`) read by
nothing, and `speed_ticks_per_tile` (`unit_schema.py:484`, the zombie cadence
override, `ai_zombie.py:318,359`).

**If built as written.** A second speed-scale path with two read sites and
two device twins, beside the canonical member, while the builder's scan
(`tests/test_footprint_samples_builder.py`) keeps guarding the first.

**Fix.** SPEED is a `FootprintSamples` member (or folded into the
caller's scale argument) passed by `footprint_samples` callers as
`effort_scale_q16` is; `default_speed` stays pure (the caller reads
`eff(u, SPEED)`, the struct carries a number). One site, every walker,
WEGO / onephase included (strike that half of the accepted gap). Retire
`base_speed` (and `StatId.BASE_SPEED`) or name it as sheet input to SPEED;
state that `speed_ticks_per_tile` is a class override, not an individual one.

### F4. Hooks entered at the law site, not at a per-unit read the predictors share -- MAJOR

**Evidence.** The repo already has the right pattern: `Simulation.
unit_turn_rate_q16` (`simulation.py:1060-1066`, "later a stat -- a change
here, never at the callers"), read by the law, by `facing_after_turn_q16`
(`:1088`), by the bots, and by the observation (`obs_spec_v1.py:580, 1233,
1448`). TURN joins it; the other quantity hooks do not have one. SPREAD's
cone is SELECTED in two copies in `combat.py:1107` and `:1192`
(`weapon.spread_deg if flags.can_aim else weapon.spread_snap_deg`) and in
`timeline.py:181`, drawn at `combat.py:1323, 1442`, and PREDICTED from the
raw row by the bots' trigger discipline (`control_bots.py:1194`,
`control_bots_v1.py:286`) and the observation (`obs_spec_v1.py:740, 1279,
1445`). HANDLING's durations are predicted from table constants
(`obs_spec_v1.py:529-530`, `ctx.throw_windup_ticks` / `ctx.plant_ticks` as
denominators at `:1260, 1269`).

**If built as written.** With SPREAD at "the cone draw in slot 4" (doc line
155) a non-neutral hook moves the law and leaves the bots firing on the
table's cone, and the observation splits: TURN reaches it through the
accessor, SPREAD and HANDLING do not. "The law and its predictor are one"
(the Turning law row's "never a copy of the law") breaks per hook.

**Fix.** The hook table gains a column "the one read", and each hooked
quantity gets one accessor in the `unit_turn_rate_q16` mould that the law,
the bots and the predictors call: e.g. `combat.cone_q16_for(u, weapon,
can_aim)` replacing the two duplicated selection lines, a
`handling_ticks(u, base_ticks)`. The U1 scan gate (doc line 371) becomes
"no reader of `spread_deg` / `reload_seconds` / ... on the continuous path
outside its accessor" -- a checkable scan instead of "no law reads traits".
Whether the observation shows the per-unit value stays the game AI lead's,
but it then reads ONE number either way.

### F5. The effect-record channel duplicates the coupling table and the swarm chapter's one bridge -- MAJOR

**Evidence.** Doc 6.4 item 2 (lines 313-327) creates a per-game record queue
with three record types: larva bites (applied through
`status.apply_status(POISONED)` + a `DamagePacket`), infestation, units
hurting larvae. (a) The CLAUDE.md Coupling table row: "A physics->unit
coupling is one row, not plumbing"; `exchange.py:639-669`
(`apply_teargas_blind`) and `:672` (`apply_poison_dose`) are exactly "a
per-tile plane under the footprint -> a status / a packet, environment
credit". A larva bite is that row over the swarm's per-tile count plane,
which doc 6.4 item 1 already lists. (b) The swarm chapter on
`63-swarm-units`, `docs/architecture/engine/17_swarm_units.md` section 7:
"The closure mechanism for all CPU-side queries is one device-written
per-tile occupancy/count field ... no per-tick position round-trips ever",
and its Systems list's draft rule "Occupancy field -- the one lawful bridge
from swarm-unit positions to CPU-side queries". A per-bite (tile, species,
strength) record is a per-larva round-trip. (c) For unit -> swarm,
`GameMap.blast_log` is "THE seam a slot-9e entity reads to react to blasts --
never a second explosion hook" (CLAUDE.md Heat source row), heat reaches
larvae through `temperature` (engine/17 fork 7b), and events -> a field go
through FieldEdit (engine/17 section 8, the food writers). Being shot is
out of v1 scope (engine/17 section 7). (d) The doc builds none of the three
(line 326), and 6.5 already concedes the queue is "the one the event ring and
the serial applier already need" (`SimultaneousEffects`).

**If built as written.** A third cross-store mechanism (fields, records,
plus the coupling table the bite will be tempted to bypass), a queue type
beside `SimultaneousEffects` / `noise_log` / `blast_log`, and a draft rule
(10 (b)) that contradicts the swarm chapter's "one lawful bridge".

**Fix.** Boundary = per-tile fields only. Larva -> unit effects are
Coupling-table rows over the swarm count plane(s) (environment credit, a
`Cause` row: as written); infestation is the conversion law reading that
plane; unit -> larva effects go through fields, `blast_log` or a FieldEdit.
Replace the record table by one sentence: "if a behaviour ever needs what a
field cannot carry, it is a design line then, in `SimultaneousEffects`'
shape". Amend the draft rule in 10 (b) accordingly. This also deletes 6.4's
order-of-records paragraph.

### F6. The draft rule contradicts the twin ledger and the hook table itself -- MAJOR

**Evidence.** Doc lines 170-171 and 10 (b) (lines 451-453): "A new hook is
... one term in one law ..., and that law's device twin, in one patch". The
CLAUDE.md Class P / Class T row and the residency design section 10
(`residency_batching_design_2026-11-07.md:1127`): "a new per-unit field: a
schema row ...; the device kernel that USES it is a Class-T twin at
catch-up", debt computed by `twin_debt.py`. The doc itself says the unit
kernels are Class T (6.1) and orders the sessions "the law on the Python
reference, Erik plays, then the kernel catches up" (doc 5, line 250).
"One term in ONE law" is also false of v1's own table: MELEE enters three
laws (`melee_blow`, the shove, the bite), HANDLING three, TURN two dials,
SPEED two sites.

**If built as written.** The CLAUDE.md row written by U1 would require CUDA
in every stats patch, against another canonical row and the arc's snapshot
model; and the "one law" clause is broken on day one.

**Fix.** "A new hook is a row of `HOOKS` naming its one read (F4), a term
inside each named law's rounding, bit-identical at `ONE`; the kernels follow
as Class-T twins at catch-up, their debt computed by the ledger." Same
change to section 4 rule 4 and section 8's "a new hook" row.

### F7. `Unit.eff` fits no schema group, and `compose_all` composes nothing -- MINOR

**Evidence.** `unit_schema.py:5-28`: every attribute is in exactly one of S
(synced, hashed per rule), D (derived, never written after spawn) or R
(host-only, never read on the match path) -- "THE RULE". Doc U1 item 2 (line
357): `eff` "declared, not hashed", written every tick and read by every
law: none of the three. Doc 2.4 line 124: "Today there are no status terms
and no meter terms, so `eff == traits == ONE`".

**If built as written.** U1 bends the Unit schema's rule for a column that
is a copy of another, adds a conductor line and a per-unit pass that do
nothing, and P6a builds a compose kernel with no term to compose.

**Fix (scope).** U1 ships `unit_traits.eff(u, hook)` as a function
returning `traits[hook]`: the one place a term will be added, so every law
already reads "eff". The stored block, the conductor line and the compose
kernel arrive with the first term (stamina, or a stat-modifier status), with
D5 as the rule then and a schema group defined for it. P6a drops "the
compose kernel".

### F8. Composing at the tick head breaks the turning predictor's contract -- MINOR

**Evidence.** CLAUDE.md Turning law row: a controller "reads
`facing_after_turn_q16`, the pure evaluation of exactly slot 1a's step --
never a copy of the law". Controllers run `handle_frame` BEFORE
`Simulation.step` (MatchSession), and the predictor reads
`unit_turn_rate_q16` / `unit_turn_accel_q16` (`simulation.py:1088-1090`).
Doc 2.4: `compose_all` runs at the head of the tick, inside the step.

**If built as written.** Once a status or meter term exists, a status that
landed at last tick's 4c changes TURN at this head, after the controllers
predicted with the old value: the bots' trigger discipline and
`ticks_to_within` (the observation) read a stale law.

**Fix.** Compose at the END of the tick (after slot 11, where the team
picture already runs) and at spawn / conversion -- the same frozen value
for the coming tick, visible to the controllers. Moot until F7's first term.

### F9. HOLD's description and the zombie overlays -- MINOR

**Evidence.** Doc line 153: HOLD scales "the stability (holding against a
push, the fall limit)", read at `unit_impulse.stability_q16`. In code the
HOLD against a push is `ImpulseTable.hold_q(u)` (`unit_impulse.py:224-232`,
with its own per-state factor `zombie_hold_q16`); `stability_q16`
(`:597-601` -> `exchange._stability_for`, `exchange.py:554-570`) is the fall
limit and the effort knockdown, with its own per-state overlay
`ZOMBIE_STABILITY`. The zombie state's other class values are selected on
`is_zombie` at the law too (`ZOMBIE_MITIGATION`, `damage.py:344`;
`[zombie] bullet_damage_multiplier`, `combat.py:99-102`; `[zombie]
expose_resist`, `exposure.py:235-241`). Doc lines 167-169: "the conversion
re-runs the compile step on the same sheet with the zombie state"; U1 gate
(lines 372-373): "converted, has the same block".

**If built as written.** HOLD does not touch the hold its row promises; and
"re-run with the zombie state" invites folding the zombie overlays into the
hooks while the laws still apply them on `is_zombie` (counted twice), while
the gate says the block does not change.

**Fix.** Narrow HOLD to "stability (the fall limit, the effort knockdown)"
or add its term at `hold_q` too, named. State: "the zombie state's
differences stay class values selected on `is_zombie` at their law;
`compile_unit` does not read `is_zombie`, so a conversion keeps `traits`
without re-running it" (and drop the re-run from 2.2 / 3).

### F10. Section 4's inventory ignores the inventory that exists -- MINOR

**Evidence.** `src/simulation/inventory.py` (40 lines): an `Inventory` stub
with growing `equipped` / `carried` lists on every unit (`unit.py:225`,
group D `unit_schema.py:598-600`, read by nothing), and
`InventoryProfile.carry_capacity_base` on `SpeciesDef`; the WEGO `loadout` /
`belt` slots are REFERENCE_ONLY (`unit_schema.py:643-645`). Doc 4 (line
190): "an INVENTORY is a fixed number of slots, each (item row, count)".
Also: `[loadouts.*]` and `[env_profiles.*]` are already "items" read by
their laws (`kit_kg` -> `mass_g`, `stability`, `expose_resist`).

**Fix.** One line in section 4: the inventory session replaces
`inventory.Inventory` (whose lists break rule 1) and decides whether
loadout / env-profile rows stay class tables read by their laws or become
items read by `compile_unit` -- never both routes for one value.

### F11. Section 10 (a) misses systems the design touches -- MINOR

Missing from (a): `stats.py` (F1), `generation.py` / `species.py` (the sheet
builder and the home of #199's draw: its docstring promises the
deterministic sampler, and engine/17 section 3 says the spawn-stat work
adopts the Philox kit "not a second sampler"), the Footprint samples builder
(F3), `inventory.py` (F10), `unit_impulse.hold_q` and
`exchange._stability_for` (F9), `TeamPicture.bars` (F2), the Class P / T
twin ledger (F6), `SimultaneousEffects`, `blast_log` and the swarm rows of
the swarm branch's CLAUDE.md (Swarm store, Swarm species table) (F5).

### F12. Scope: P6's plan and the cost table have homes already -- MINOR

**Evidence.** Doc 7 "P6, re-planned" (lines 379-396) re-plans a patch of the
residency design; the precedent for that is an amendment section inside the
residency design (`residency_batching_design_2026-11-07.md` 4.4a, "P3a
re-planned against the code"). Doc 8 restates the residency design's
section 10 table for the unit.

**If built as written.** Two documents hold P6's plan; the next P6 session
must reconcile them (the master CLAUDE.md: update the existing home, never
fork a parallel copy).

**Fix.** Move the P6 re-plan into the residency design as 4.4b (leave a
pointer); cut section 8 to the rows section 10 lacks (stat formula, item,
status term, meter) and point at 10 for the rest. Section 5 is direction
for later sessions and can stay as written.

### F13. "One N" leaves the unit store's arrays unplaced -- MINOR

**Evidence.** `resident_batch.py:58-67` stacks exactly
`GameMap._RESIDENT_FIELD_NAMES` to `(N,) + shape` -- the mechanism the doc
6.3 relies on for the swarm (whose arrays join that list on the branch,
`gamemap.py:274` there). Doc P6a builds "the device unit store `(N, U,
columns)`" without saying where its per-game arrays live.

**Fix.** One sentence in 6.3 or P6a: the unit store's per-game arrays are
registered in the same resident list (or name the one stacking path they
use), so reset by mask and the bank cover them as they cover the swarm and
P6a builds no second stacking path.

---

## Checked and found correct

- TURN enters `Simulation.unit_turn_rate_q16` / `unit_turn_accel_q16`, the seam written for it; the predictor and the observation follow (`simulation.py:1060-1090`, `obs_spec_v1.py:580, 1448`).
- Mass stays its own column written by `unit_impulse.bind_mass` (`unit_impulse.py:395-405`); not a hook.
- D7: dropping the swarm's `N_ENV` (`swarm.py:42` on the branch) is what `ResidentBatch`'s generic stacking needs; no second N.
- 6.3's randomness row folds the swarm's planned salt enum into `rng_salts.Salt` / `MatchDraws`, as engine/17 section 3 wants ("ONE enum").
- Swarm credit to the environment with its own `Cause` row matches the Credited cause row; the swarm has its own `unit_id` space (`swarm.py:71`), so no shared id convention is needed.
- The full / swarm criterion quoted in 6.1 matches engine/17 section 1 verbatim; two stores agree with its fork 2a (arrays on GameMap) and 9b.
- Exposure as the shape of a meter and the status block as the shape of a timed condition reuse existing systems; status FLAGS stay with `composed_flags` (untouched).
- A status term as a status-kind row matches `StatusKindDef`'s own plan ("Speed/accuracy modifier columns arrive with the stat-modifier statuses", `status.py:110-111`): the term belongs in that registry, not a new table.
- `traits` as a group-S block absent at its default follows the Unit schema's digest rule and moves no golden.
- Section 4's growth rules (fixed size, ids not references, schema row per column) restate the Unit schema and residency section 10 rules without contradicting them.
