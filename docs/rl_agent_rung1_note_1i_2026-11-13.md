# Rung 1, patch 1i: reward profile v1 (the note)

2026-11-13. Written for the orchestrator of rung 1 (#195) from the plan
(`docs/rl_agent_rung1_plan_2026-11-11.md` §1, §2 row 1i, §5 "1i"), the
design (`docs/rl_agent_design_2026-11-11.md` 6.1-6.9, 7.2, 10, 11), the
observation spec (`docs/rl_agent_obs_spec_v1_2026-11-12.md` 1.6, 4.1, 4.2,
5.4) and the 1g / 1h notes, checked against the code at main `cff35c83`
(1a-1h merged). Same shape as the 1f and 1h notes. **CHOICE:** marks where
the design is silent and this note decides an encoding. **STOP:** would
mark a design/code contradiction; there is none. The design is AGREED (6 is
the lead's, delegated, agreed by Erik 2026-11-11 and corrected by the
critique): this note chooses encodings, seams and tests, never what the game
rewards. The one place where it touches what is rewarded is a question at
the end.

**Design:** 6.1 (rules: winning is the objective, a helper is the change of
a score read off the state, no game number, a named spec), 6.2 (the terms
and sizes), 6.3 (own against team; the two row flags), 6.4 (not rewarded),
6.5 (sizes; the clock is an ending), 6.6 (three sides), 6.7 (the order
term, weight 0), 6.8 (what 1i builds and its tests), 7.2 (five value
outputs, one per number), 10 (the Reward profile row). **Why now:** rung 2's
trainer needs the five numbers; 1j's imitation records carry them.

**The pending zombie balance** (branch `183-zombie-bite-balance`, awaiting
Erik's play: `[zombie] hp` 400 -> 100, the bite 34 damage / 2.2 s): nothing
here reads either number. Health enters only as `hp / bar` with the bar the
unit's own (6.1), so every property below holds under both values; no test
may name a hit-point figure.

## What the code does today (the facts the build rests on)

- **The env's reward** (`src/rl_env.py`). `COUNTER_NAMES` (`:200-215`) are
  generated: `hits_dealt`, `damage_dealt`, `team_hits`, `kills`,
  `team_kills`, `hits_taken`, `damage_taken`, then `deaths_<cause>` per
  `events.CAUSE_NAMES`. `EnvRunner._count` (`:786-817`) fills them per tick
  from `sim.tick_events`: a hit with `damage > 0` counts on the victim's row
  and, when `attacker_id >= 0` and not the victim, on the attacker's row
  (`team_hits` if both were on one team at the tick's start, else
  `hits_dealt` / `damage_dealt` -- raw damage, unclipped, marine and zombie
  alike); kills likewise (`kills` counts zombie kills too). The row of a
  unit is credited only while its team at the tick's start is the row's
  team (`_row`, `:780`): a dead marine keeps its team and its row, a turned
  one does not. The worker writes `reward[e] = (counters * weights).sum(1)`
  (`:967-969`) with the run file's `[reward]` weights per counter
  (`RunConfig.reward_weights`, `:264`; the door `parse_run`, `:427-435`).
  `reward` is `(N, U)` float32, `counters` `(N, U, K)` float32
  (`Layout.shapes`, `:568-590`). `terminated` = fewer than two marine teams
  alive, `truncated` = its negation at an ending (`:771-772`): **the clock
  is reported as a cut** (design 6.5's "today").
- **The row flags** (1g). `EnvRunner.row_flags` (`:842-859`): `in_match` =
  the roster slot holds a unit (dead or turned included), `valid` = alive
  and `team == t`; served in `obs` under `[obs] spec = "v1"` only (the crop
  spec has `valid` alone). `roster_units` (`:830`) gives every row's roster
  unit.
- **The bar tracker exists** (1g, `src/simulation/team_picture.py`):
  `TeamPicture.bars[uid]` = `float(current_hp)` at the first slot-11 update
  at which the unit exists (`:261-262`) and again at the update that first
  sees `is_zombie` (`:263-265`, the conversion ran in `on_tick_end` before
  slot 11). That IS design 6.1's bar ("the health a unit had when it came
  into being on its side: at spawn, or at the turning"), on the slot-11
  driver, never on demand, not synced. The obs spec 1.6 table already says
  "patch 1i's bar tracker reads this one": **1i reuses it and builds no
  second one.** `death_tick`, `last_hit` and the damage window are not
  needed by the reward (the damage window is a team's SEEN damage, the
  reward's is credited damage).
- **Health and the events** (`src/simulation/damage.py:363-460`). Nothing
  writes `hp_q` but `apply_packet` (the Weapons row) and the conversion
  (`ai_zombie.convert_marines_to_zombies`, `:400-417`: `team =
  ZOMBIE_TEAM`, `alive = True`, `current_hp = [zombie] hp`). `apply_packet`
  subtracts the APPLIED damage (mitigated, exposure-scaled, Q16) from
  `hp_q` with no clamp (hp goes below 0), emits ONE `UnitHitEvent` whose
  `damage` is that applied amount's exact float view -- also on a corpse,
  and also negative for a HEAL packet (`status.py:355`, the `regen` row) --
  and on the alive -> dead transition only a `UnitKilledEvent` (it dies
  once). Every `apply_packet` call site on the match path passes the tick's
  events (`ai_zombie.py:304`, `exchange.py:483/539/733`, `simultaneous.py:
  107/167`, `status.py:357`), so within a tick every change of `hp_q` of a
  unit that does not turn is an event, in emission order (the attack slot's
  walk order, #169).
- **Endings** (`src/simulation/ruleset.py:309-352`). `MatchRules.
  living_teams(sim)` = the marine teams with a living non-zombie unit (THE
  statement of "a team is alive"); `is_terminal` = fewer than two of them or
  `sim.tick >= limit_ticks`; `result()` = `win` for the one team left (even
  on the limit tick), else `draw` -- **including a simultaneous wipe of the
  last two teams**, which design 6.5 pays as two eliminations (below: the
  eliminations are read from `living_teams`, the win and the clock from
  `result()`, so no second rule for who won exists; design 10).
- **The full view's balance column** (`obs_spec_v1.FullView.match`,
  `:1964-1978`): `balance[t]` = the sum over team `t`'s living marines of
  `current_hp / bars[uid]` (unclipped) over `n_t`, squashed. Obs spec 5.4:
  "1i replaces the builder, not the column."
- **The match**: ship_demo and playground are 2 marine teams of 4 and a
  horde (`levels/*/match.toml`); no shipped level has three marine teams,
  but the env's layout takes T from the levels, so 6.6 is built and tested
  on a box.

## Build

### 1. The reward module (`src/simulation/reward_v1.py`, NEW)

THE reward as code, beside `obs_spec_v1` and `action_spec_v1`: pure host
Python, reads the sim, the tick's events and the team picture's `bars`,
writes nothing to any of them. No BLAS (`@`, `dot`): the No-BLAS rule's
scan covers `src/simulation/` (sums are broadcasts).

**A named profile** (design 6.1 "a named spec"; 10 "a named table of
weights over counters"): `PROFILES = {"fireteam": {...}}`, its TERMS and
their sizes exactly design 6.2's:

| term | fireteam | applied to (counter, sign fixed in code) | group |
|---|---|---|---|
| `win` | +10 | `won` | outcome |
| `eliminated` | -10 | `eliminated` | outcome |
| `clock` | -3 | `clock` | outcome |
| `marine_bar` | 1 | `+bars_enemy`, `-bars_lost` / `+share_bars`, `-share_enemy_bars` | own / team |
| `marine_alive` | 0.5 | `+kills_enemy`, `-died` / `+share_alive`, `-share_enemy_alive` | own / team |
| `zombie_bar` | 0.25 | `+bars_zombie` / `-share_zombie_bars` | own / team |
| `ff_bar` | -2 | `bars_team` | friendly fire |
| `ff_kill` | -3 | `team_kills` (the existing counter) | friendly fire |

CHOICE: the three balance terms are ONE weight each, used with a code-fixed
sign on the own and the team counters alike -- so the mirror of 6.2 ("the
four balance terms are one thing seen four ways") and the identities of
the gate (below) hold for every profile a run file can write. "range" and "duel"
(6.9) are rung 2's rows of this table.

**The five groups** (6.8, 7.2), in this order everywhere:
`REWARD_GROUPS = ("outcome", "own", "team", "friendly_fire", "order")`.
`group_weights(terms, counter_names)` -> a `(K, 5)` float64 matrix, zero for
every counter in no group (the old counters, `kills`, `deaths_*`, ...);
`order` is an all-zero column (6.7, rung 4). A group's number for a row is
the sum of its counters times their term weights. **The env never sums
across groups** (design 10): the trainer weights the five and blends own
and team by its team-share schedule (6.3: 0.3 -> 0.8), which is a trainer
number and never in the env.

**The bar ratio, one function**: `bar_ratio(hp_q, bar)` = `min(max(hp_q,
0), bar_q) / bar_q` in float64, `bar_q` the bar quantized once to the Q16
grid it lives on (`bars` holds `float(current_hp)`, exact n/65536). Health
above one bar counts as one (6.1); a corpse's health is 0.

**The balance, read from state** (6.2): for marine team `t`,
`B_t = mb * (H_t - H_enemy) + ma * (A_t - A_enemy) - zb * H_zombie`, with
`H_t` = the sum of `bar_ratio` over `t`'s living non-zombie units, `A_t`
their count, `H_enemy` / `A_enemy` the same over every OTHER marine team
(6.6: same weight), `H_zombie` the sum of `bar_ratio` over the living
zombies (spawned and turned alike, each over its own bar). One function,
`standing(sim, bars) -> StandingParts` (per team `H_t`, `A_t`; `H_zombie`),
and `balance(parts, t, terms)`.

### 2. The books: `RewardBooks`, one per episode

`RewardBooks(sim, spec, n_slots, terms)` -- built by `EnvRunner.reset`
after the session (and usable on any `MatchSession`, for the replay test).
It holds, per unit, its LAST state `(hp_q, alive, team, is_zombie, bar_q)`
and per team the last `StandingParts` and `living_teams`. Two calls a tick:

- **`begin(sim)`**, right before `s.tick()` (where `team_before` is built
  today): any difference between the stored state and the state now
  happened OUTSIDE a tick (only a test writes it: a heal between steps).
  It is settled with no attacker: a member's change on its own row
  (`bars_lost`, `died`, either sign), everything else in `rest_t`, a
  team's `living_teams` exit paid as an elimination; then the per-unit
  state is stored. The team part's `standing` before is NOT refreshed here:
  it stays the last `end`'s (or the episode start's), so the next tick's
  team part includes the outside change too, and identities 1-3 telescope
  over it. In a farm
  episode it is always empty; the books count `begin` residuals
  (`books.outside_changes`) and the env never raises on them.
- **`end(sim)`**, right after `s.tick()`: walks `sim.tick_events` and
  settles the tick (below), then stores the new state.

**Membership.** A unit is a MEMBER of team `t` for a tick if at the tick's
start `team == t` and not `is_zombie` -- a corpse included (the `_row`
rule, `rl_env.py:780`). A member's row is `t * S + slot`.

**The walk** (per tick, events in emission order). Per unit the books
carry `h` = its stored `hp_q` and `alive`. For each `UnitHitEvent` on unit
`v` with damage `d != 0` (`d_q = round(d * 65536)`, exact): `r =
bar_ratio(h) - bar_ratio(h - d_q)` (the bars REALLY removed: 0 on a corpse
and above the bar, never more than was there; negative for a heal), then
`h -= d_q`. The attacker `a` is CREDITED when `d > 0`, `a >= 0`, `a != v`
and `a` is a member of some team `ta` (a dead member included: a grenade
or charge landing after its thrower's death is credited to the thrower, as
the game credits it). Each of the following that applies, independently:

1. **the victim's own** -- `v` a member of `tv`: `bars_lost[row(v)] += r`
   (its own health, whoever or whatever took it);
2. **the attacker's own** -- `a` credited and `v` a zombie:
   `bars_zombie[row(a)] += r`; `a` credited and `v` a member of a team
   other than `ta`: `bars_enemy[row(a)] += r`;
3. **friendly fire** (its own group, never balance) -- `a` credited and
   `v` a member of `ta`: `bars_team[row(a)] += r` (beside case 1 for the
   victim). A self-hit (`a == v`) is case 1 only (6.2: "ordinary lost
   health");
4. **the uncredited rest** -- for every marine team `t`, the change this
   hit makes to `B_t` (`-mb*r` if `v` is a member of `t`, `+mb*r` if `v` is
   a member of another team, `+zb*r` if `v` is a zombie) that cases 1-2 do
   not put on a row of `t` goes to `rest_t`: a zombie's bite on an enemy,
   the environment's `-1`, a third team's hit, any heal of a non-member.

For each `UnitKilledEvent` on `v` (the alive -> dead transition), likewise:
`alive = False`; `v` a member of `tv` -> `died[row(v)] += 1`; `a` credited
and `v` a member of another team -> `kills_enemy[row(a)] += 1`; `a`
credited and `v` a mate -> the existing `team_kills` (the old counter is
already exactly this: same team at the tick's start, `a != v`); every
team's change of its `A` terms not put on its own rows -> `rest_t`. A zombie
has no alive term (6.4: its death is not paid, only its health). Credit
inside one attack slot follows the emission order (the walk order of
#169): the first packet takes what was there, so of two shooters on one
tick the earlier-walked one is credited with the kill (CHOICE: the game's
own order, no splitting).

**The turning** (and only the turning) is settled after the walk: a unit
whose `(team, is_zombie)` changed during the tick was a dead member whose
contribution the walk already brought to 0 (its bite and death are
events); it now stands as a zombie at `bar_ratio(hp, new bar)`. That change
(`-zb * 1` today) is UNCREDITED: it goes to `rest_t` of every team (6.2:
"and then a quarter more"; 6.8 "a turning included"; 6.3 "what the zombie
it became does ... is the horde's doing, not friendly fire").

**The check, not a remainder.** For every unit whose `(team, is_zombie)`
did NOT change, the walk must end exactly at its new `hp_q` and `alive`.
A mismatch is an hp write outside `apply_packet` or an event with no
packet: the books raise `RewardBooksMismatch` naming the unit, the tick
and both values (a game finding, never patched over). So `rest_t` is
BUILT from its own sources (uncredited events, turnings, `begin`
residuals) and never computed as "the balance change minus the own parts"
-- which is what makes the gate's identity 2 a check that can fail.

**The team part** (6.3). Per tick, `ΔH_t`, `ΔA_t`, `ΔH_enemy(t)`,
`ΔA_enemy(t)`, `ΔH_zombie` are read from `standing()` before and after
(state, not events); each row of team `t` that is `in_match` gets
`share_* += Δ / n_t` with `n_t = len(sim.roster(t))` (the units spawned
into the team: dead and turned rows keep their share, 6.3 "a dead soldier
stays on the books"). So the team group of every row of `t` is
`ΔB_t / n_t`, whoever or whatever caused it -- the horde's work on the
enemy included (6.3's luring line).

**Outcomes** (6.2, 6.5, 6.6), per team, each paid ONCE:

- **eliminated**: a team in `living_teams` at `begin` (or at the episode's
  start) and not at `end` -> `eliminated += 1` on every `in_match` row of
  the team (dead rows included). Two last teams wiped in one tick are two
  eliminations (`result()` says `draw`; neither team appears in
  `living_teams`).
- **won / clock**: when the session is terminal, `books.finish(result)`:
  `result.outcome == "win"` -> `won += 1` on every row of `result.team`;
  `draw` with two or more teams in `living_teams` (the limit) -> `clock +=
  1` on every row, dead included, of each such team (6.2: "Why the clock's
  -3 goes to the dead as well"); `draw` with none alive pays nothing more.
- **records end** (6.6): a row's records end at its team's elimination or at
  the episode's end. After that, in a match that goes on (three marine
  teams), every counter of the row stays 0 (posthumous credit included).
  The books keep computing the team's numbers (the identity is checked on
  them); only the payment to the row is masked.

**The new counters** (6.8 "per soldier per decision"), APPENDED to the
generated `COUNTER_NAMES` after the old ones (CHOICE: one generated tuple,
so the run file's door, the shapes and `info["counters"]` know them with no
second list): `bars_enemy`, `kills_enemy`, `bars_zombie`, `bars_lost`,
`died`, `bars_team`, `share_bars`, `share_alive`, `share_enemy_bars`,
`share_enemy_alive`, `share_zombie_bars`, `won`, `eliminated`, `clock`.
Every old counter keeps its meaning and its column (6.8 "keeps every old
counter working"); the new ones are filled under every profile (CHOICE:
the books always run -- one code path -- and a `counters`-profile run may
weight them like any counter). Tests never pin the
tuple's length or order (it grows with `CAUSE_NAMES`).

### 3. The env (`src/rl_env.py`)

- **The run file.** `[reward]` stays THE table. A new optional key
  `profile` (string). Absent = `"counters"`, today's placeholder, unchanged
  bit for bit: weights per counter name, `reward` `(N, U)` = the weighted
  sum, the clock `truncated` (every existing run file and test stays
  valid). `profile = "fireteam"`: every other key of `[reward]` must be a
  TERM of the profile (an override: `zombie_bar = 0.1` for a ship with many
  zombies, 6.5; `clock = -10`, 6.9), refused by name otherwise -- a counter
  name included (CHOICE: under a profile the weights live on terms, so a
  counter weight would break the mirror); a term's value must be a number.
  `profile = "fireteam"` requires `[obs] spec = "v1"` (refused by name: the
  dead row's pay needs `in_match`, 6.3's table). `RunConfig` gains
  `reward_profile` and `reward_terms` (the profile's table with the
  overrides).
- **The buffers.** Under `"fireteam"`, `reward` is `(N, U, 5)` float32 --
  the five numbers in `REWARD_GROUPS` order, for every row with `in_match`
  1 (dead and turned rows included), 0 for an empty roster place and for a
  row whose records ended; never the weighted sum. `Layout` gains
  `reward_profile`; `Layout.shapes` gives `reward` its shape from it.
  `counters` keeps `(N, U, K)` float32 with the new columns. The worker
  writes `reward[e]` by broadcasting `counters[:, :, None] * G[None]` summed
  over the counter axis (no `@`; `G` = `group_weights`, built once per
  worker, in place of today's `weights`).
- **Internally float64.** `EnvRunner.step`'s counters array becomes
  float64 (cast once at the shared-memory write); the books are float64
  throughout. The identity tests read the in-process float64 values.
- **Per team per decision** (6.8): `info["team_balance"]` `(N, T)` float32,
  `ΔB_t` over the step; `info["team_uncredited"]` `(N, T)` float32, the
  step's `rest_t` (6.9's friendly-fire experiment reads team losses with
  no attacker); `info["team_outcome"]` `(N, T)` float32, the outcome group
  paid to the team's rows this step. For an env that ended inside the step
  these are the ENDED episode's (the step's numbers belong to the episode
  they happened in; the reset starts fresh books). Zero for a team the
  level lacks. All three are new small shared blocks (the env-owned rule
  is for the large observation buffers; these are fresh copies like
  `reward`).
- **`info["rows_ended"]`** `(N, U)` bool: 1 on the step where a row's
  records end (its team eliminated, or the episode over -- every `in_match`
  row of an ended env). CHOICE: the per-row ending a trainer needs in a
  three-team match, where a team's episode ends before the env's.
- **The clock as an ending** (6.5): under `"fireteam"` an episode's end is
  `terminated` whatever ended it and `truncated` is always False; under
  `"counters"` unchanged.
- **What a policy reads**: nothing of this. `team_rows` hands a policy the
  `obs` keys only (1g); no reward, counter or team key ever enters `obs`.

### 4. The full view's balance column

`FullView.match` (`obs_spec_v1.py:1964-1978`) stops computing its own sum:
`balance[t]` = `standing(...).H[t] / n_t` (squashed as today) through
`reward_v1.standing` -- ONE implementation of "a team's health in bars"
(obs spec 5.4: "1i replaces the builder, not the column"). CHOICE: the
column stays the team's own health term `H_t / n_t` (its meaning, its name,
its denominator); its only change is the clip at one bar, which no shipped
unit reaches (nothing heals above its spawn health). The value network has
the living counts beside it and every unit in its full list, so the
balance `B_t` itself is not a second column (a new column would be a new
version of the full view, its row). Obs spec 5.4's wording ("the same
quantity 1i's team part differences") is corrected in the patch: the team
part differences `B_t`, of which `H_t` is one term.

### 5. Cost

Per tick: one pass over `sim.units` (~12 on ship_demo) twice (`begin`,
`end`), one walk over the tick's events, two `standing()` reads.
Estimate: 20-40 us a tick, ~0.15 ms per env-decision at decimation 4,
against ~100 ms of an env-decision (the 1h profile): ~0.2 %.
**Gate:** the design's §9.1 protocol (`tools/rl_smoke.py`, 64 envs / 16
workers, ship_demo, decimation 4, 750 measured steps after 20 of warm-up,
one session), the 1h run (v1 observation + full view, `[action] spec =
"v1"`) with `[reward] profile = "fireteam"` beside the same run on
`"counters"`. **The line:** above spec 6's 70 % of crop (the hard line,
1h read 80 %); a loss of more than 2 % against `"counters"` is a finding
reported with an in-process profile, not a stop. `tools/rl_smoke.py`
learns to accept `(N, U, 5)` rewards (it ignores them today; a smoke run
asserts their shape and finiteness).

## The gate: the books close (design 6.8)

**The identities** (float64, absolute tolerance `1e-9` per comparison;
values are of order 10):

1. **The books close over a match**: for each team, the sum over the
   episode's steps of `ΔB_t` equals `B_t(end) - B_t(start)`, both read
   from state by the TEST's own second implementation of the balance (a
   ten-line loop in the test over `sim.units` and `team_picture.bars`,
   never `reward_v1.standing`).
2. **Own plus uncredited is the change, every decision**: for each team
   `t`, the sum over `t`'s rows of the own group + `rest_t` (by its terms)
   == `ΔB_t` -- before the records-ended mask.
3. **The team part is the change**: the sum over `t`'s `in_match` rows of
   the team group == `ΔB_t`, every decision (each row `ΔB_t / n_t`).

**The scenes** (in-process `EnvRunner`s; levels written through
`level_lib`, the test convention): ship_demo, two seeds, every team on
`policy` with the 1h legal-uniform sampler (`rl_smoke.legal_v1_rows`) and
both teams on `bots` (the bots kill; random rows rarely do), whole
matches; `test_rl_env`'s zombie box (the horde eats a team: bites,
turnings, a converted zombie's bites on its old team); a three-team box
(one team eliminated mid-match). The driver FAILS as vacuous unless the
run saw at least one of each: an overkill hit (damage > health left), a
hit on a corpse, a turning, a zombie's bite on a marine, an uncredited
(environment) hit, friendly fire, an elimination and a win; it prints the
counts (a posthumous credit is counted and printed, not required).

**The gate can fail -- three deliberate breaks, each reverted, naming the
check that failed** (the report shows them):

- (i) the walk credits the event's full `d` (no clip at `h`): identity 2
  fails on the first overkill hit;
- (ii) the turning is not settled into `rest`: identity 2 fails on the
  first turning decision (the zombie box);
- (iii) `begin` is skipped (the out-of-tick residual is lost): identity 1
  fails in the heal test of the property table (a heal written between
  steps).

## Property tests (`tests/test_reward_v1.py`, NEW)

Each test's docstring states its property and the change that must break
it. No test names a hit-point figure (the pending balance), and none pins
the counter tuple's length or order.

The design's 6.8 list, as written:

| property | breaks if |
|---|---|
| the books close: a team's balance changes over a whole match add up to its balance at the end minus its balance at the start (identity 1; the gate's scenes) | a change of state is missed (break iii), or `ΔB` is computed from events instead of state |
| own parts plus the uncredited rest (a turning included) equal the balance change, on every decision, also when a hit is far stronger than the health left (identity 2; plus a scripted overkill: a frag on a unit at a sliver of health pays `bar_ratio(h)` and nothing more) | the full force is credited (break i), or a turning is dropped (break ii) |
| a hit on a corpse pays nothing and blames nobody: a credited hit (and a friendly one) on a dead unit leaves every counter of every row unchanged | the corpse's negative `hp_q` is not clipped, or the alive term is paid twice |
| healing an enemy takes back, from the team part, exactly what hurting it paid: hurt an enemy by `x` bars (team part `+mb*x/n_t` per row), heal it back (`regen` status or a HEAL packet written between steps): the team part over the two is 0 to 1e-9, the hurter's own part is unchanged by the heal | a heal is ignored (`begin` skipped, or `d <= 0` dropped from the walk), or credited to the healer's own part |
| a win, an elimination or the clock is paid to every row of the team exactly once, the dead rows included; two last soldiers dying in one tick is paid as two eliminations (a box: one soldier a side and a timed charge between them, both killed by its blast on one tick) | the outcome is paid twice (e.g. the clock also on a win), skipped on a dead row, or a simultaneous wipe is read from `result()` as a draw |
| friendly-fire blame lands on the credited unit and on no other; a turned soldier's bites blame nobody (the zombie box: the bites of a converted marine on its old squad move no `bars_team` / `team_kills`) | blame is shared out, or a turned unit is still mapped to its old row |
| the same match replayed gives the same rewards: two runners on one seed and one row stream give equal counters and rewards every step; a recorded episode re-run through `MatchSession(level, replay=rec)` with fresh books gives the same per-tick numbers | the books read anything but state, events and bars (a clock, a dict order, a draw) |

And the build's own:

| property | breaks if |
|---|---|
| the five numbers: `reward` is `(N, U, 5)` in `REWARD_GROUPS` order under `"fireteam"`; a row with `in_match` 0 is all zeros; a dead and a turned row get the team and outcome groups (not zero) and nothing in own / friendly fire unless credited; `order` is 0 | a dead row is skipped (the share divided among the living), or the env sums the groups |
| the clock is an ending: under `"fireteam"` a time-limit end is `terminated` and not `truncated`; under `"counters"` `test_truncation_is_the_time_limit_and_not_a_termination` stays green unchanged | the clock is reported as a cut under the profile |
| three sides: in a three-team box the eliminated team gets `-10` at its elimination, `rows_ended` there, zeros after; the last team wins; the middle team's balance rises while the other two fight (`team_balance > 0` on a step it dealt nothing) | records do not end, or enemies of different teams are weighted differently |
| the zombie term is not a mirror: a zombie's lost health raises every marine team's balance | the zombie term is credited to one team only |
| sizes from the profile: overriding `zombie_bar` (or `clock`) in the run file moves exactly the counters' weights it names (the group matrix); the mirror holds for any override | a size is a literal in the walk |
| the run-file door: an unknown profile, a counter name or an unknown term under a profile, a non-number, `"fireteam"` with `[obs] spec = "crop"`: each refused by name; no `profile` key = today's behaviour bit for bit (`test_rl_env.py` green unchanged) | the door accepts what the books cannot pay |
| the books' check: an `hp_q` written inside a tick outside `apply_packet` (monkeypatched slot) raises `RewardBooksMismatch` naming the unit and tick | the books absorb an unexplained change into `rest` |
| a policy never reads a reward: `obs` holds no reward, counter or team key, and `team_rows(obs, t)`'s keys are `obs`'s (obs spec 4.1, "1i's trainer test") | a reward key is added to `obs` |
| the full view's `balance[t]` equals `squash(standing().H[t] / n_t)` (and the old unclipped sum wherever no unit is above its bar) | the full view keeps its own builder |
| `match_pin_check` EQUAL and the farm runs: the reward reads only (no synced state, no digest moves) | the books write a unit attribute or a field |

Plus the existing suites whole (`test_rl_env.py`, `test_obs_spec_v1.py`,
`test_action_spec_v1.py`, `test_team_picture.py`), `pytest tests -q`, and
`tools/match_pin_check.py` EQUAL.

## The patch cut

**One patch, one Opus implementer** (this worktree, or a fresh one cut
from main), four commits, each green:

| commit | content |
|---|---|
| A | `src/simulation/reward_v1.py`: `PROFILES`, `REWARD_GROUPS`, `bar_ratio`, `standing`, `balance`, `group_weights`, `RewardBooks` (`begin` / `end` / `finish`, the walk, the turning, the check, the outcomes, the records-ended mask); its unit tests on hand-built events |
| B | `src/rl_env.py`: the appended counters, `[reward] profile` and the term door, `Layout.reward_profile`, the `(N, U, 5)` reward, the team blocks, `rows_ended`, the clock as an ending, the books in `EnvRunner.reset` / `step`; `tools/rl_smoke.py` accepting the new reward |
| C | `obs_spec_v1.FullView.match` through `standing`; the obs spec document's 5.4 line |
| D | `tests/test_reward_v1.py` with the gate's driver and the breaks; the farm reading; `CLAUDE.md`; `tests/twin_manifest.toml` if `test_twin_debt` asks (a `[not_twinned]` row "Class T: the reward, host-side (#195 1i)" if the corpus runs the module; it should not) |

**Gates for the implementer:** the three identities on every scene with
the vacuity floor and the three breaks reported; the property tests;
`pytest tests -q` whole; `tools/match_pin_check.py` EQUAL; the farm
reading of section 5; `tools/twin_debt.py` / `tests/test_class_p_level.py`
decide whether `tools/cuda_green.py` is needed (expected not: no Class-P
file is touched).

**Files:** NEW `src/simulation/reward_v1.py`, `tests/test_reward_v1.py`;
`src/rl_env.py`, `src/simulation/obs_spec_v1.py`, `tools/rl_smoke.py`,
`docs/rl_agent_obs_spec_v1_2026-11-12.md` (5.4), `CLAUDE.md`, maybe
`tests/twin_manifest.toml`.

## Systems

**Existing systems used** (canonical-systems rows):

- **Team picture** -- its `bars` tracker IS the bar of design 6.1; 1i
  builds no second one and reads it after slot 11 of each tick.
- **Credited cause** -- the own and friendly-fire parts are built from
  `UnitHitEvent` / `UnitKilledEvent` with their `attacker_id`; the
  environment's `-1` is uncredited.
- **Weapons / `apply_packet`** -- the only hp writer: the walk replays its
  applied damage in Q16 and asserts it reproduces the state.
- **Unit** -- `is_zombie` the only zombie check, `team` a member's side;
  the conversion read as a change of both.
- **Match config / `MatchRules`** -- `living_teams` is the one statement of
  a living team (eliminations), `result()` the one statement of who won
  (win, clock); never a second rule.
- **Batched environment** -- the reward groups, the team blocks and
  `rows_ended` join its buffers; `team_rows` stays the policy's one path.
- **Observation spec / Full view** -- the row flags `in_match` / `valid`
  (1g) are the reward's row contract; the full view's balance column calls
  the reward's `standing`.
- **MatchSession** -- the books run on any session, the replay's included.

**New system, with its draft rule** (written into `CLAUDE.md` by the
patch, design 10's Reward profile row sharpened against the code):

| system | rule |
|---|---|
| Reward profile | `src/simulation/reward_v1.py` (`PROFILES`, `REWARD_GROUPS`, `RewardBooks`, `standing`, `bar_ratio`) + `[reward] profile` in the run file: a reward is a named table of term weights over counters, in the run file; the env returns one number per group (outcome, own, team, friendly fire, order) for every `in_match` row, dead rows included, and never the sum; a new term is a new counter in a group. Health is counted in bars (`TeamPicture.bars`), clipped at one. A change not credited to a unit is a change of a team's balance read from state; the uncredited rest is BUILT from its sources (uncredited events, turnings, out-of-tick changes), never computed as a remainder, and a tick whose events do not reproduce a unit's hp raises. The balance is one function, also the full view's |

**Amended rows:** Batched environment (`[reward] profile`, the `(N, U, 5)`
reward, `team_balance` / `team_uncredited` / `team_outcome`, `rows_ended`,
the clock as an ending under a profile, the 1i farm reading), Full view
(`balance[t]` through `standing`), Team picture (the bars' second reader),
Credited cause (the reward is a consumer), the header line.

## Question for Erik (the game, not the encoding)

1. **Who is paid for health given back?** Nothing heals in a match today,
   but the `regen` status row exists (a HEAL packet credited to its
   applier). This note pays a heal as the healed soldier's own health
   change (`bars_lost` goes down: "the health I lost myself", backwards)
   and through the team part, and NEVER to the healer's own part -- so
   healing an enemy costs the healer only its team share, and healing a
   mate earns the healer only its team share. Recommendation: keep it so
   until a healing verb exists; a medic's reward is that verb's design
   question.

## As built

(Filled by the implementer: the commits, every choice beyond this note,
the gate's printout with its vacuity counts and the three breaks, the farm
reading.)
