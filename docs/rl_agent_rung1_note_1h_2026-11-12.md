# Rung 1, patch 1h: action spec v1, its decoder and its masks (the note)

2026-11-12. Written for the orchestrator of rung 1 (#195) from the plan
(`docs/rl_agent_rung1_plan_2026-11-11.md` §1, §2 row 1h, §5 "1h", "1i",
"1j"), the design (`docs/rl_agent_design_2026-11-11.md` 2.1-2.8, 8.2, 8.6,
10, 12.2 L3 M2, 12.3 items 5 and 7), the critique (lens 3 M2, m2; lens 4
minors 2-6), the 1e and 1f notes, and the observation spec
(`docs/rl_agent_obs_spec_v1_2026-11-12.md` 1.5, 2.1, 2.2 cols 28 / 66, 3.5,
6), checked against the code at main `8e61f179` (1a-1g merged). Same shape
as the 1e and 1f notes. **CHOICE:** marks where the design is silent and
this note decides. **STOP:** would mark a design/code contradiction; there
is none. The design is AGREED ("your action space is blessed"): this note
chooses encodings, orders and seams, never what the game is.

**Design:** 2.1 (the choices), 2.2 (the decoder, inside `policy`, holding
game ids), 2.3 (what depends on what), 2.4 (masks from the real gates, made
one step early, the one-sided promise), 2.5 (verbs pointed at), 2.6 (left
out: the flashlight reserved and masked), 2.7 (the bots through the door,
1j), 2.8 (reserved places), 8.2 (what is left out of the learning signal).
**Why now:** 1j drives the layered bots through this spec, and rung 2's
trainer samples it; 1i needs nothing from 1h (its row flags are 1g's).

## What the code does today (the facts the build rests on)

- **The policy controller** (`src/control_policy.py`). `AgentPolicy` holds
  ONE raw action row per roster slot, handed over by `set_rows` (`:296`),
  and issues it every tick through `issue_row` (`:258`): the `hold` kinds
  every tick, the `latch` kinds only on the first tick after `set_rows`
  (`VERB_RULES` `:84`). Columns are generated from `INTENT_KINDS` minus
  `EXCLUDED_KINDS` (`drive_zombies`, `order_intent`, `:78`). `handle_frame`
  (`:310`) skips a row whose unit is dead or no longer on the team
  (`:322`): a dead or turned row issues NOTHING today.
- **The env** (`src/rl_env.py`). `Layout.n_actions` = the raw schema's
  width (`make_layout` `:559`); `actions` is `(N, U, A)` int32 in shared
  memory (`Layout.shapes` `:527`); `EnvRunner.step` (`:670`) hands each
  policy team its block through `set_rows` (`:680`) and then ticks
  `decimation` times. The observation is written at the END of a step,
  after the tick advance (`publish` `:860`, `EnvRunner.observe` `:792`):
  `sim.tick` is then T, the tick the next step's first tick will run, so
  anything built there is built **at the start of tick T** -- exactly the
  moment the 1e gates answer for. The env-owned buffers and their `final_`
  twins are `BatchedEnv._buf` (`:1289`); `team_rows` (`:1442`) is THE path
  a policy is fed by; an auto-reset publishes episode k+1's observation, so
  the `list_uid` a trainer's next row refers to is always the match that
  row will act in (lens 4 minor 4).
- **The verbs** (`src/simulation/simulation.py`). `set_move_dir` (`:1005`,
  throttle clamped to `[0, THROTTLE_ONE]`, `movement.py:174`),
  `clear_move_dir` (`:1027`), `set_aim` (`:1037`; a zero aim holds the last
  target, and a walker holding no aim faces its travel: the Turning law
  row), `set_trigger` (`:1084`), `throw_grenade_intent` (`:1096`, integer
  `fuse_ticks`, `reach_q` clamped at release to `max_throw_range`
  `:1508`, `config.toml:3416` = 30 tiles), `plant_charge_intent` (`:1136`,
  clamped to the c4 row's 5..60 s), `use_intent`, `reload_intent`,
  `shove_intent`, `melee_intent`. Every one records itself on the tap
  first. Every held verb possesses the unit; a policy unit is possessed
  from its first `set_aim`, so a hand verb adds nothing there.
- **The wind-up.** `_start_throw` (`:1397`) asks `verb_gates.can_throw`,
  then sets `throw_release_tick = tick + 7` (0.3 s); `_advance_throw`
  (`:1425`) releases at 1b of the release tick along the facing the body
  has THEN, dropping the wind-up if the unit cannot act. Nothing in
  `combat.py` reads `throw_release_tick`: **a unit can fire during its
  wind-up** (the bots release the trigger themselves,
  `control_bots._tick_throw`).
- **The gates** (`src/simulation/verb_gates.py`, 1e): `can_throw(sim, u,
  kind)`, `can_plant`, `can_use` / `use_door`, `can_reload`, `can_shove`,
  `can_melee(u, tick, tables, at_tick_start=True)` for the mask; pure
  reads, exact at the start of the tick except for events that can only
  forbid. 1e's STOP was resolved by way (a): a refused reload settles
  nothing, so a forced reload on a never-bound magazine leaves the digests.
  1e's note for 1h: a hand latch issued for a dead or zombie unit is never
  consumed and stays hashed (`pending_*`) -- the decoder must issue nothing
  for such a row.
- **The team picture** (1d): `records(team)[uid].centre` is written at slot
  11 from the tick's sight, after the advance, so for a unit the team sees
  it IS the live centre at the start of the next tick, and for one it does
  not it is the last-seen centre. The bots aim with `_direction_q`
  (`control_bots_v1.py:157`: `quantize_stick_direction` of the centre-tile
  offset, the one control-seam quantizer) and judge a shot with
  `facing_after_turn_q16` (`simulation.py:1066`, `control_bots._on_target`).
- **The observation** (1g) reserves for 1h: vector cols 71..76 `prev[h]`
  (-1 today), list col 28 `on_next_tick` and col 66 `prev_aim` (0 today,
  `obs_spec_v1.py:1614`), the near plane `sees_target` (0 today,
  `build_windows` `:1063`). The list is `[S-1, 16, 8, 6, 4]` entries of types
  0 mates, 1 enemies seen, 2 records, 3 objects, 4 doors (`E = S + 33`);
  `list_uid` is the game's unit id for types 0-2, an equality key only.

## Build

### 1. The five heads (`src/simulation/action_spec_v1.py`, NEW)

THE action spec as code: `ACTION_SPEC_NAME = "v1"`, the tables below as
data, the mask builder, the pure mask helpers for the trainer, and the
nearest-choice encoders for 1j. Pure host Python; imports `verb_gates`,
`weapons`, `unit_fixed`, nothing from a controller or `renderer/`.

**The action row a trainer sends is five int32 per row: `(feet, aim,
trigger, hand, arg)`**, in `actions` `(N, U, 5)`. Each is an index into its
head's option table:

| head | size | index -> option | reason |
|---|---|---|---|
| feet | 49 | 0 = stand; `1 + 3*d + s` = compass direction `d` (0..15) at speed `s` (0 walk, 1 half, 2 full) | design 2.1/8.2: one choice among 49; direction-major so a direction's three speeds sit together (CHOICE, the network does not care) |
| aim | `18 + E` | 0..15 compass direction `d`; 16 = "where I walk"; 17 = "keep the throw's aim" (see 3); `18 + i` = list entry `i` (the observation's list position, re-sorted every decision) | the tokens FIRST, so their indices do not move with `S`; the pointer covers the WHOLE list (types 0, 3, 4 masked in v1), so the entry types of 2.8 (a downed mate, a bag, a door) open by a mask change, never a shape change |
| trigger | 2 | 0 release, 1 hold | |
| hand | `7 + K` | 0 nothing, 1 reload, 2 melee, 3 shove, 4 use, 5 plant, 6 flashlight (RESERVED, always masked: design 2.6), `7 + k` = throw hand-grenade row `k` (`WeaponsTables.hand_grenade_rows`, config order: frag, smoke, tear, poison, flash, incendiary today, K = 6) | the throws LAST: a new grenade is a row APPENDED to the family (Grenade kinds row), so it appends here too and no index moves; each entry has a stable name (`HAND_NAMES`: `"reload"`, `"throw:grenade_frag"`, ...) so a network keys its verb vectors by name (design 2.5, pointing) |
| arg | 5 | for a throw: 0..3 = a quarter, half, three quarters, all of `max_throw_range`; 4 = onto my aim target. For a plant: 0..3 = the four timers; 4 not used. Any other verb: not used | one shared head told the verb (design 8.2; lens 4 minor 2 allows it); the four quantities share 0..3, the special one is 4 |

Sizes at S = 4 (E = 37): 49, 55, 2, 13, 5. Design 2.1's "about 17 + the
aimable entries" is the 18 tokens plus the 24 aimable places (16 seen, 8
records). **CHOICE: stance (design 2.8) is not a sixth integer in v1**; it
arrives with the game's stances as a new action spec, as does the
flashlight's opening at rung 8 (the reserved entry keeps the hand head's
shape until then).

**The numbers inside the tables, each from its table row, quantized once
per match (an `ActionTables` built by `action_tables(sim)` at the decoder's
first tick and by the mask builder; never a literal):**

- **Compass:** 16 Q16.16 unit vectors at `d * 22.5` degrees, counter-
  clockwise from east in the facing convention (`(cos, -sin)`, world y
  down), as a CHECKED-IN literal table `COMPASS_Q16` (the radiation sweep's
  ordinate precedent: no `math.cos` at load). A test holds it to
  `quantize_stick_direction(cos, -sin)` and to `atan2_q16` landing on each
  bin's centre. Feet and aim share it.
- **Throttle:** walk 0, half `THROTTLE_ONE // 2`, full `THROTTLE_ONE` (the
  Throttle row: 0 IS the walk, ONE the full speed, linear in speed
  between, so half is the speed halfway; design 2.1's "half is the normal
  pace").
- **Fuse:** `fuse_ticks_from_seconds([weapons.hand_grenade]
  fuse_default_seconds, tps)` (design 2.6; lens 3 m6: one fuse, not per
  kind).
- **Reach:** `(j + 1) * (max_throw_range << 16) // 4` for `j` = 0..3 (lens
  3 m6: a fraction of the one range); "onto my aim target" is the bots'
  `_lob` distance, `quantize_scalar(sqrt(d2))` of the centre-tile offset to
  the target's resolved centre on the issuing tick.
- **Plant timers:** `lo`, `hi` = `fuse_ticks_from_seconds` of
  `[weapons.c4_satchel] fuse_min_seconds` / `fuse_max_seconds` (the
  `PLANT_WEAPON` row, lens 3 m6); `t_j` = the integer cube root, rounded to
  nearest, of `lo^(3-j) * hi^j` (exact integers: `t` = the floor root, `+1`
  when `8 N >= (2t + 1)^3`). At 24 tps: 120, 275, 629, 1440 ticks = 5,
  11.5, 26.2, 60 s -- the design's "5, 11, 26, 60", spaced by one factor,
  with no number of seconds in the spec and no `pow` (CHOICE: integers
  rather than a float power, the iron rule's spirit for a value that becomes
  a verb argument).

**The run file chooses the spec** (CHOICE): a new optional table `[action]
spec = "raw" | "v1"`, default `"raw"` = today's generated columns, unchanged
and still the named spec a run without the table gets (the 1g farm readings
and every existing run file stay valid). `"v1"` requires `[obs] spec =
"v1"` (its pointer indexes the v1 list; refused by name otherwise).
`Layout` gains `action_spec`; `n_actions` is 5 under `"v1"`.
`BatchedEnv.step` refuses, naming the env, row and head, any index outside
`[0, size)` -- an out-of-range integer is a trainer bug, never decoded. A
masked IN-range index is decoded (section 2's fallbacks) and COUNTED:
`info["masked_choices"]` `(N, U)` uint8, the number of heads whose sent
option was masked, computed in the driver from the masks it served before
the step overwrites them (CHOICE: a trainer that samples a masked option
has a bug the counter shows at once; the forced test reads it too).

### 2. The decoder: a per-tick step inside `AgentPolicy`

`AgentPolicy(team)` stays the one controller (MatchSession row: a policy
drives units only as `policy`; lens 3 m2). It gains an action spec,
`configure("v1", tables)`, set by `EnvRunner.reset` right after the
session is built and refused once a row has been handed over; under `"raw"`
nothing changes, bit for bit.

- **`set_decisions(rows5, list_uid)`** (the `"v1"` twin of `set_rows`,
  each refusing the other spec): `rows5` is the team's `(S, 5)` block,
  `list_uid` the team's `(S, E)` block of the observation those rows were
  chosen on -- the EnvRunner passes its own last-published `list_uid` (the
  array in shared memory, written by its last `observe`). **The pointer is
  turned into a game id here, once, and never kept as a position**: aim
  `18 + i` becomes `("unit", list_uid[s, i])`; a token stays a token
  (design 2.2, lens 4 minor 4).
- **Per unit, the decoder keeps** (controller state, never on the `Unit`,
  never synced; design 9 "the decoder's held aim"): the decision's five
  decoded choices; for an aim at a unit, its uid and `last_point` (the last
  centre tile it resolved to); `throw_aim` (an aim choice held through a
  wind-up, below); `aimed_uid` (the uid it aimed at on the last tick it
  issued, -1 for a token: what the observation marks); `prev` (the six
  decoded choices of the last decision, for vector cols 71..76).
- **Every tick, per living row on the team** (the existing skip, `:322`,
  stays the first line: a dead or turned row issues nothing, ever), the
  decoder builds ONE raw row of today's schema and hands it to today's
  `issue_row(sim, uid, row, first, plan)` -- so `VERB_RULES` stays the one
  statement of hold vs latch and nothing new reaches the sim:
  - **feet:** stand -> the `clear_move_dir` flag; else `set_move_dir.dx_q /
    dy_q` = `COMPASS_Q16[d]`, `throttle_q` = the speed's throttle. Hold.
  - **aim:** a compass token -> its vector; "where I walk" -> `(0, 0)`; a
    unit -> the direction from the soldier's centre tile to
    `sim.team_picture.records(team)[uid].centre` **as it is on this tick**
    (the live centre while the team sees it, the last-seen one otherwise:
    design 2.2, read through the team's own picture, never a live enemy),
    by the bots' `_direction_q` (one quantizer); `last_point` remembers
    it. When the record is gone (expired, or a death the team saw) or the
    offset is zero, the decoder aims at `last_point` (or holds the last
    vector issued) -- "its last-seen position" (CHOICE). Hold.
  - **trigger:** `set_trigger.held`. Hold.
  - **hand:** on the decision's FIRST tick only (`first`, a latch): reload,
    melee, shove, use as their flag columns; a plant as
    `plant_charge_intent.fuse_ticks = t_j`; a throw of row `k` as
    `throw_grenade_intent` with `kind = k`, the one default fuse, the reach
    of `arg`, and **the body's facing as its direction**
    (`cos_q16(facing_q)`, `-sin_q16(facing_q)`, design 2.2 and lens 3 m2:
    the latch gate `vector` needs a non-zero pair, and a non-zero wind-up
    releases along the facing anyway). The decoder **does not consult the
    hand mask**: a masked verb is issued and the game's own gate refuses
    it -- which is what the gate of section 5 tests.
- **The aim held through a wind-up** (design 2.1, 12.3 item 5; lens 3 M2
  way (a)). On a decision's first tick, if its hand is a throw AND the unit
  holds no wind-up at that moment (`u.throw_release_tick < 0`, read before
  issuing), the decoder sets `throw_aim` = this decision's aim CHOICE (a
  token, or the unit with its `last_point`). At the start of every later
  tick: if `throw_aim` is set and `u.throw_release_tick >= sim.tick` (the
  game's own state: a wind-up is running and has not yet let go), the aim
  issued is `throw_aim`'s, resolved as above, whatever the current decision
  chose; otherwise `throw_aim` is cleared and the decision's aim is used.
  So: the aim of the decision that STARTED the throw steers it to its
  release (a moving target is followed, as any aim is; the reach was fixed
  on the issuing tick, as the verb fixes it); a throw the game refused
  never started a wind-up and holds nothing (the do-nothing row's aim on
  every tick); a second throw forced during a running wind-up does not
  overwrite the first's held aim. CHOICE: the CHOICE is held, not the
  vector of the issuing tick (the bots freeze the vector): the design's
  words are "keeps the aim of the decision that started it", and an entry
  followed to the release lands where the target is. The trigger is NOT
  overridden during a wind-up: the game lets a unit fire then (see the
  question at the end), and design 2.1 holds only the aim.
- **Fallbacks for a masked IN-range index** (the decoder's rules, needed
  because a masked option may name nothing; CHOICE): an aim at a list
  position whose `list_uid` is -1 or whose type is not aimable (a mate, an
  object, a door), or "keep" with no wind-up running, decodes as **"where I
  walk"**; a throw whose `arg` is masked (4 = onto target with a token aim)
  and a plant with `arg` 4 decode as **hand "nothing"**; the `arg` of any
  other verb is ignored. Feet and trigger have no masked option on a living
  row. A dead or turned row issues nothing whatever it sends.
- **`prev`** = the decoded choices actually issued (after the fallbacks),
  split for the observation's six columns (below).

### 3. The masks

Built **once per decision at the start of tick T** -- inside
`EnvRunner.observe`, right after the observation (whose list they read), by
`action_spec_v1.build_masks(sim, units, list_type, list_mask, list_uid,
tables, decimation, out)`; for every row of the layout, `units` the
`row_units()` (None for an invalid row). **Each hand bit is a call of the
1e gate, never a copy** (design 2.4, 12.3 item 7; a test swaps a gate and
watches the mask follow):

| mask key (`obs`) | shape | a valid row (alive, still on its team) |
|---|---|---|
| `mask_feet` | `(N, U, 49)` u8 | all 1 (design 2.4: "full" always legal; nothing else is masked) |
| `mask_aim` | `(N, U, 18 + E)` u8 | tokens 0..16 always 1; 17 ("keep") 0; entry `18 + i` = `list_mask[i] and type[i] in (1, 2)` -- an enemy seen now or a last-seen record, i.e. a unit in the team picture (design 2.4: never "is alive"). **While a wind-up will hold the aim on every tick of the coming decision** (`u.throw_release_tick >= T + decimation - 1`): ONLY 17 is 1 |
| `mask_trigger` | `(N, U, 2)` u8 | both 1 (design 2.4: no gate) |
| `mask_hand` | `(N, U, 7 + K)` u8 | 0 nothing: 1; 1 `can_reload(sim, u)`; 2 `can_melee(u, T, sim.weapons_tables, at_tick_start=True)`; 3 `can_shove(sim, u)`; 4 `can_use(sim, u)`; 5 `can_plant(sim, u)`; 6 flashlight: 0 (reserved); `7 + k` `can_throw(sim, u, k)` |

An **invalid row** (dead, turned, or an empty roster place) gets exactly
one legal option per head: feet 0 (stand), aim 16 ("where I walk"),
trigger 0, hand 0, arg 0 (lens 4 minor 5: the fixed default; and the
decoder issues nothing for it anyway). Each key has its `final_` twin and
lives in `BatchedEnv._buf` with the observation's keys (the env-owned
rule), and is served in `obs`, so `team_rows` hands a policy its own masks
and nothing else (CHOICE: the masks are part of what a policy is FED, not a
training-only `info` item; they are not observation-spec keys, so the
observation spec's name is untouched). Under `[action] spec = "raw"` none
of these keys exists.

**The argument's mask is realised after the aim and the hand are drawn**
(lens 4 minor 2): it is a pure function of the two choices and needs no
buffer. `action_spec_v1.arg_mask(hand, aim)` (numpy, vectorised, no sim):
a throw -> 0..3 legal, 4 legal iff `aim >= 18`; a plant -> 0..3; any other
hand -> only 0. The trainer calls it at sampling time and STORES its result
(design 8.6's "the masks as they were").

**"Left out of the learning signal" is the mask itself** (design 8.2,
lens 4 minors 5 and 6; CHOICE): every case 8.2 lists -- the argument of a
verb that takes none, the aim while a throw holds it, every head of a row
that cannot act -- is a head with EXACTLY ONE legal option. Its probability
is then 1, its log-probability 0, its gradient 0 and its entropy 0, so it
adds nothing to the ratio, the loss or the entropy bonus, with no separate
"used" flag to keep in step with the masks. `action_spec_v1.heads_used(
masks, arg_mask)` = "more than one legal option" is exported for the
trainer's bookkeeping and 8.6's tests. No head is ever all-masked
(a softmax over all -inf is NaN): "nothing", the trigger's two options and
the default of an invalid row are always legal, and "keep" is legal exactly
when it is the only one. (Design 2.4's "the compass directions are never
masked" exists to keep a head from being all-masked; masking them while
"keep" is the one legal option keeps that purpose and is lens 3 M2's way
(a), which design 12.2 adopted.)

**The one-sided promise** (design 2.4) is the gates' (1e): a mask bit 0 at
the start of tick T means the verb issued at T does nothing; a bit 1 may
still do nothing (a knockdown at 1a, a blast at 2, no one in reach). A
latch goes out on the decision's first tick only, which IS tick T: the
mask and the consumption see the same moment. A cool-down that runs out on
tick 2 of a decision is not offered until the next decision (accepted, the
decimation's cost, measured by 1j's 4/6/8-tick variants).

### 4. What 1h fills in the observation, and why spec "v1" keeps its name

Through the existing door: `EnvRunner.observe` passes the decoder's view,
`aims = policy.aim_view()` -- per unit id of a policy team, `(aimed_uid,
prev)` -- to `accessor.observe("v1", ..., aims=aims)`, which hands it to
`observe_rows` as plain values beside `TeamKnowledge` (the soldier's own
controller state; no sim read crosses the wall). A unit of a team not on a
v1 policy is absent and reads the "none" values.

- **Vector cols 71..76 `prev[h]`** (spec 1.5, kinds, -1 before the
  match's first decision): feet direction 0..15, 16 = stood; feet speed
  0..2, -1 when stood; aim: the token index 0..17, or 18 = "a list entry"
  (which one is marked by col 66 -- spec 1.5: "never a position"); trigger
  0/1; hand: its index; arg: its index, -1 when the hand took none. CHOICE
  for the encodings; the spec document's row 71..76 is amended with them.
- **List col 66 `prev_aim`**: 1 on the entry whose `list_uid ==
  aimed_uid` (at most one; none for a token). `aimed_uid` is what the body
  was actually turning toward on the last tick -- the previous decision's
  aim, or the throwing decision's while its aim was kept -- one definition
  for cols 28, 66 and `sees_target` (CHOICE).
- **List col 28 `on_next_tick`**: on that entry only, the bots' own test
  (`control_bots._on_target`): `abs(wrap(atan2_q16(-dy, dx) -
  facing_after_turn_q16(me, dx, dy))) <= spread_q`, with `(dx, dy)` the
  decoder's own aim vector to the entry's centre (one function,
  `action_spec_v1.aim_vector(me, centre)`, called by the decoder and by
  this column -- never a second quantizer) and `spread_q` the weapon's
  `spread_deg` quantized as the bots quantize it. `facing_after_turn_q16`
  goes through `ObsWorld` with a MATE argument (me), as 1g already allows.
- **Near plane `sees_target`**: one `vision.window_clear(sim, ty, tx,
  window, to_window=True)` FROM every window tile TO the aimed entry's
  centre (its record's centre, the list's own), all 0 without an aimed
  entry (spec 3.5's builder, its one line).

**No version bump** (the brief's question): the spec document already
DEFINES these four as the decoder's (1.5, 2.2 cols 28 / 66, 3.5); 1g built
them at their "nothing aimed / no previous decision" values because the
decoder did not exist, which is their meaning when there is none. Nothing
has been trained on spec v1 (rung 2 has not started; the only readers are
1g's tests and the farm reading), so no network's input changes meaning.
And the meaning stays one thing on both action specs: under `"raw"` there
is no decoder and the columns truthfully read "none". A version bump is
for a change of meaning under a network that exists; this is the first
writing of a reserved column. The spec document's "until 1h" lines become
"as built (1h)" in the patch.

### 5. The gate: a masked choice issued by force does nothing

**The property** (plan §2, §5 "1h"): for every masked option of every
head, on a living row, sending it leaves the match's `field_digest` and
`tick_digest` where the row with that option's DECLARED FALLBACK leaves
them: hand "nothing" for a masked hand verb or a masked argument, "where I
walk" for a masked aim. Plus: anything sent on an invalid row leaves them
where the all-default row does.

**The driver** (`tests/_1h_forced.py`, on the model of `tests/_1e_lockstep.py`
but in ONE process, two lockstep runners -- no before/after build is
involved, the comparison is between two choices): two `EnvRunner`s (in-
process, the CPU build, `record=False`) on the same run (`[obs] spec = "v1"`,
`[action] spec = "v1"`, every team on `policy`, all-round sight, decimation
4), the same level and seed. A seeded sampler draws, at every decision,
row A uniformly among A's LEGAL options of each head (the realised
`arg_mask` included), with the hand biased toward the non-"nothing" legal
verbs so throws, plants, wind-ups, cool-downs and reloads happen. Runner B
gets the same rows except that ONE living row (rotating) gets one MASKED
option of one head (rotating through the hand table, the aim's masked
places -- an empty slot, a mate, an object, a door, "keep" outside a
wind-up --, and the masked argument), and A gets that option's fallback in
the same place (equal to A's own draw for the hand: "nothing"). After every
TICK the two `(field_digest, tick_digest)` (`session.digests()`) must be
equal; a difference stops the run naming the decision, the tick, the row,
the head, the option and the unit's state (its gate inputs: `can_act`, the
wind-up and plant ticks, `mag_status`, the cool-downs, the counts). Because
the promise holds, the runners stay identical, so one pair tests every
forced choice of the run in sequence. The masks B is forced against are
B's own (equal to A's while the digests are). On ship_demo, two seeds, at
least 300 decisions each; the driver prints, per hand verb, how many forced
issues it made and in which states, and FAILS (vacuous) unless each masked
hand entry was forced at least 20 times, including at least once each:
throw during a wind-up, throw of a kind at 0, plant during a plant, plant
during a wind-up, melee during a wind-up and during a plant, melee and shove
on cool-down, shove and melee and throw and plant while stunned (the rifle
butt stuns), reload on a never-bound full magazine (match start), reload on
a full magazine and during a stall, use away from every door, the reserved
flashlight, and a forced hand verb on a dead row and on a turned row (the
latch must never be set: `pending_*` stays at its default in the unit
record). The pytest wrapper (`tests/test_action_spec_v1.py::
test_forced_masked_choices_do_nothing`) runs one seed at 120 decisions with
the coverage floor lowered to "each masked hand entry at least once"; the
full driver's printout goes in the report.

Not a lockstep refactor proof (1e's) but a property over the real game.
Since the mask and the verb call the SAME gate, breaking a gate breaks both
alike and cannot show the driver failing; what the driver guards is
everything around the gate: the MOMENT (the mask read at tick start, the
verb at its slot), effects outside the gate, the decoder's fallbacks and
its dead-row skip. The report shows it can fail with three deliberate
breaks, each reverted, naming the check that failed: (i) `build_masks`
calls `can_melee(..., at_tick_start=False)` -- a melee on the tick a stun
is swept is then masked yet lands; (ii) `_reload` settles the magazine
before asking its gate again (1e's way (a) undone) -- a forced reload at
match start moves the unit digest; (iii) the decoder's dead-row skip
removed -- a forced latch on a dead row stays set and hashed.

### 6. The recording

**No format change** (`FORMAT_VERSION` stays 8): the decoder issues the
existing verbs through the existing tap (design 2.2: "nothing new enters
the game"); `INTENT_KINDS`, `EXCLUDED_KINDS` and `VERB_RULES` are untouched.
The action rows, masks and the decoder's state are the trainer's and the
controller's, never the recording's. **A replay of a policy match** is the
ordinary re-simulation: `EnvRunner(..., record=True)` records every episode
through the tap (`MatchSession.record`), and `run_match(level, spec,
replay=rec)` (or `main.py --replay`) re-issues the recorded intents through
the `replay` controller, verifying every checkpoint. The decoder is not
needed to replay, and a replay cannot depend on it. The run file's action
spec is part of the run's provenance through the run file's sha256 in the
manifest.

### 7. Cost

Per env-decision (8 rows on ship_demo), against the 1g-c farm reading in
the Batched environment row (v1 + full view 358.2 env-ticks/s, p50 722 ms a
step at 64 envs / 16 workers, decimation 4; the observation 7.7 ms and the
full view 6.6 ms per env-decision):

- **Masks:** per living row 6 + K gate calls (13 today; `can_use` scans the
  live doors through the ring key, a few dozen tiles), the rest numpy over
  the list's columns: estimate 0.05-0.1 ms a row, under 1 ms per
  env-decision.
- **Decoder:** per living row per tick, one record lookup and one
  quantization beside the `issue_row` the raw spec already runs: about 10 us,
  0.3 ms per env-decision over its 4 ticks.
- **Observation fill:** one `window_clear` (25 x 25, the `watched`
  plane's own cost per known enemy) and one `predict` per row with an aimed
  entry: under 1 ms per env-decision.

Expected total under 2.5 ms per env-decision, ~1 % of a worker's ~180 ms
per env-decision at that farm size. **Gate:** the design's §9.1 protocol
(`tools/rl_smoke.py`, 64 envs / 16 workers, decimation 4, 750 measured
steps, one session) on `[action] spec = "v1"` beside `"raw"` with the same
v1 observation and full view: v1 must stay above spec 6's 70 % of crop
(the hard line); a loss of more than 5 % against `"raw"` is a finding
reported with a profile, not a stop.

### 8. What 1j needs, built here

1j translates each bot decision "to the nearest choice"; the translation
must use this spec's tables, never a second copy. So `action_spec_v1`
exports the inverse helpers, each tested as a round trip with the decoder:
`feet_index(dx_q, dy_q, throttle_q)` (nearest compass bin by angle,
`atan2_q16`; nearest of the three throttles; a cleared move -> 0),
`aim_index_for_unit(uid, list_uid_row, list_type_row)` (the entry's place,
or None), `aim_index_for_vector(dx_q, dy_q)` (nearest compass token; a zero
vector -> 16), `throw_choice(kind, reach_q, aimed)` (hand `7 + kind`, the
nearest of the four reach fractions, or 4 when the lob is onto the aimed
entry), `plant_choice(fuse_ticks)` (nearest timer by ticks). And the
decoder runs in-process (`EnvRunner` with `record=True`, the same masks
from `build_masks`), so 1j's saved (observation, choices, masks) records
come from the one builder.

## Property tests (`tests/test_action_spec_v1.py`, NEW)

Each test's docstring states its property and the change that must break
it (the master rule). No test pins a count or an order of something built
to grow (the hand table is checked AGAINST `hand_grenade_rows`, never as a
list of six).

| property | breaks if |
|---|---|
| the hand table is the fixed verbs then one throw per `hand_grenade_rows` row in config order; appending a grenade row (monkeypatched tables) appends one entry and moves no other index | throws are not last, or a kind is hard-coded |
| `COMPASS_Q16[d]` equals the stick quantizer of `(cos, -sin)(d * 22.5 deg)` and `atan2_q16` of it lies in bin `d`; every feet index decodes to its `(d, s)` and back | the table, the convention (y down) or the index formula drifts |
| walk / half / full send throttle 0, `THROTTLE_ONE // 2`, `THROTTLE_ONE` (read from the tap) | a speed is a literal of another scale |
| the four reaches are `(j+1)/4` of the table's `max_throw_range` (a monkeypatched range moves them); "onto target" equals the bots' `_lob` distance to the record's centre on the issuing tick | a tile literal, or the reach read on another tick |
| the plant timers are the row's `lo` and `hi` at the ends, strictly increasing, each the integer cube root of `lo^(3-j) hi^j` (a monkeypatched row moves them); the plant the game starts carries the timer (its `plant_fuse_ticks`) | seconds hard-coded, or a float `pow` that rounds differently |
| the decoder holds a UID: across a decision in which the list re-sorts (two enemies swap places), the issued `set_aim` keeps pointing at the same unit's record, every tick | the decoder keeps a list position (lens 4 minor 4) |
| the aim follows: against an enemy walking across the soldier's view, each tick's `set_aim` (from the tap) equals `aim_vector` to that tick's record centre; after the record expires it keeps the last point | the aim is computed once per decision, or reads the live unit |
| the decoder never reads a hidden unit: two sims differing only by an unseen move of a remembered enemy (1g's counterfactual (b)) issue the same intents tick for tick | the decoder reads `sim.units` / a live position |
| hold vs latch: over a decision, move / aim / trigger are issued on every tick and the hand verb exactly once, on the first tick (the tap) | the decoder re-sets rows every tick (lens 3 m2) or bypasses `VERB_RULES` |
| the throw's aim is held: a throw at decision k toward entry X, decision k+1 choosing a compass token: every `set_aim` through the release tick aims at X, the grenade leaves along the facing that aim produced, and the token applies from the tick after the release; `mask_aim` at k+1 is "keep" alone exactly when the release is at or after the decision's last tick | the next decision's aim steers the grenade (lens 3 M2) |
| a refused throw holds nothing: a throw forced with 0 of its kind issues on every tick the same aim as the "nothing" row | the hold keys on "I sent a throw" instead of the game's `throw_release_tick` |
| mask == gate: for every living row and hand entry, over a played match, the mask bit equals the 1e function called at the same moment; with a gate monkeypatched to answer False the bit follows | the mask copies a gate |
| aim mask: an entry is legal iff it is a type-1 or type-2 entry with `list_mask` 1; a record of an enemy that died unseen stays legal (design 2.4) | the mask reads liveness |
| an invalid row (dead, turned, empty place): one legal option per head (the defaults), and NOTHING is issued for its unit whatever is sent (the tap holds no call with its id after the death; its `pending_*` stay at default) | a dead row issues a latch (1e's note) |
| `arg_mask(hand, aim)`: "onto target" legal iff a throw and `aim >= 18`; a plant 0..3; any other verb 0 alone; `heads_used` is "more than one legal" and is False for exactly 8.2's three cases | the argument's legality ignores the drawn aim or verb (lens 4 minor 2) |
| no head is ever all-masked, over every row of a played match | a gate state leaves a head with nothing legal (NaN in the softmax) |
| the fallbacks: a masked aim decodes as "where I walk", a masked argument as hand "nothing" (the tap) | a masked index is decoded as something else |
| the observation fill: col 66 marks exactly the entry whose uid the body aimed at on the last tick (none for a token); col 28 equals `_on_target` for it; `sees_target` equals `has_los` from each window tile to the entry's centre; cols 71..76 equal the decoded previous choices, -1 before the first decision; under `[action] spec = "raw"` all of them read the none values | a column is filled from a list position, or from the raw rows |
| 1g's counterfactual family (a)-(e) still holds with the decoder on (the fill reads only the decoder and the team's knowledge) | the fill leaks a live unit |
| the run-file door: `[action] spec` unknown refused by name; `"v1"` with `[obs] spec = "crop"` refused; under `"v1"` the actions are `(N, U, 5)` and an out-of-range index is refused naming env, row and head; `info["masked_choices"]` counts a forced masked option | the door accepts what the decoder cannot read |
| the inverse helpers round-trip with the decoder (`feet_index` of a decoded feet index is that index, etc.) | 1j would need a second table |
| a recorded v1 policy match replays with every checkpoint equal (`run_match(replay=)`), and the raw spec's existing farm gates are green unchanged | the decoder issues outside the tap, or the format moved |
| the forced-mask gate (section 5) | |

Plus the existing suites, whole and green (`test_rl_env.py`,
`test_obs_spec_v1.py`, `test_verb_gates.py`, `test_bots_v1.py`,
`test_zombie_control.py`), and `tools/match_pin_check.py` EQUAL (no game
code moves: nothing outside the controller, the env and the observation is
touched).

## The patch cut

**One patch, one Opus implementer** (worktree `.claude/worktrees/192-1h-
actions-impl` or this one, cut from main), in five commits, each green:

| commit | content |
|---|---|
| A | `src/simulation/action_spec_v1.py`: the tables, `action_tables`, `COMPASS_Q16`, the timers, `arg_mask` / `heads_used`, the inverse helpers; their table tests |
| B | the decoder in `src/control_policy.py` (`configure`, `set_decisions`, the per-tick step through `issue_row`, the held throw aim, the fallbacks, `aim_view`); `src/rl_env.py` (`[action] spec`, `Layout.action_spec`, `n_actions`, `EnvRunner.reset` / `step` passing `list_uid`, the range check); the decoder's tests and the replay test |
| C | `build_masks`, the `mask_*` keys and their `final_` twins in `Layout.shapes`, `_buf`, `obs`; `info["masked_choices"]`; the mask tests |
| D | the observation fill (`obs_spec_v1.observe_rows` taking `aims`, cols 71..76, 28, 66, `sees_target`; `sensor_accessor.observe` passing it); its tests; the spec document's "until 1h" rows amended |
| E | `tests/_1h_forced.py` and its pytest wrapper; the cost reading; `CLAUDE.md`; `tests/twin_manifest.toml` regenerated (a `[not_twinned]` row for `action_spec_v1.py`: "Class T: the action spec's masks and tables, host-side (#195 1h)") |

Why one patch and not two (spec + decoder, then masks): the gate the plan
names needs all of it, and nothing consumes half -- a decoder without masks
is a spec a trainer cannot sample honestly, and the masks share the held-
aim rule ("keep") with the decoder. The commits give the stop points.

**Gates for the implementer:** the forced-mask driver EQUAL on ship_demo,
two seeds, with its coverage printout (and the two deliberate breaks of
section 5 reported); the property tests; `pytest tests -q` whole;
`tools/match_pin_check.py` EQUAL; `tools/rl_smoke.py` on a `"v1"` action
run (2 envs) and the §9.1 farm reading of section 7; `tools/twin_debt.py`
and `tests/test_class_p_level.py` decide whether `tools/cuda_green.py` is
needed (expected not: no Class-P file is touched; `simulation.py`,
`combat.py` and the gates are untouched).

**Files:** NEW `src/simulation/action_spec_v1.py`, `tests/test_action_spec_v1.py`,
`tests/_1h_forced.py`; `src/control_policy.py`, `src/rl_env.py`,
`src/simulation/obs_spec_v1.py`, `src/simulation/sensor_accessor.py`,
`docs/rl_agent_obs_spec_v1_2026-11-12.md` (the "until 1h" rows),
`CLAUDE.md`, `tests/twin_manifest.toml`.

## `CLAUDE.md` (in the patch)

- **New row "Action spec"** (Gameplay, after "Observation spec" / "Full
  view"), from the roadmap's draft ("the one layer from a network's choices
  to control verbs; masks come from the game's own gates, never a copy of
  them"): `src/simulation/action_spec_v1.py` + the decoder in
  `control_policy.AgentPolicy` + `[action] spec` -- an action spec is a
  named, versioned set of heads (`"raw"` the generated columns, `"v1"` five
  integers: feet 49, aim 18 + E, trigger 2, hand 7 + K, arg 5); a pointer
  becomes a game id at the decision and is never kept as a position; the
  decoder is a per-tick step of the `policy` controller that builds a raw
  row and issues it through `issue_row` (so `VERB_RULES` stays THE hold /
  latch rule and nothing new reaches the sim or the recording); the aim of
  the decision that started a throw is held while `throw_release_tick >=
  tick`; the masks are built at the start of the decision tick from the
  verb gates, one call per bit, served in `obs` with `final_` twins; a head
  left out of learning is a head with one legal option; a new verb is a
  hand-table row with its gate, a new grenade row appends a throw entry; a
  masked choice issued by force does nothing (`tests/_1h_forced.py`).
- **Amend:** "MatchSession + the policy controller" (the two action
  specs; `set_decisions`), "Batched environment" (`[action] spec`, the
  mask keys, `info["masked_choices"]`, the 1h farm reading), "Observation
  spec" (cols 28, 66, 71..76 and `sees_target` filled from the decoder's
  view, "read 0 until 1h" deleted), "Verb gates" (the masks are the second
  caller now), "Match recording / replay" (a v1 policy match replays; no
  format change). The header line.

## Questions for Erik (the game, not the encoding)

1. A unit can fire its weapon while winding up a throw (nothing in
   `combat.py` reads the wind-up; the bots release the trigger themselves).
   The note leaves the agent's trigger free, as the game is. Intended?
