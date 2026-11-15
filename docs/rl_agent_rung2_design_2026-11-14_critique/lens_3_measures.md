# Rung 2 design critique, lens 3: can each measure fail, and does it mean what it says

Critic: Claude Opus 5.5, 2026-11-15. Object: `docs/rl_agent_rung2_design_2026-11-14.md`
at `5c61d1d1` (branch `192-rung2-cadence`). Read: the whole object; the roadmap
(rungs 2 and 3, section 5); design sections 6.2-6.9, 8.3, 8.6, 13.1-13.4; the 1j
note (the measurement, the as-built tables, the variants); read-only in the main
checkout: `tools/bot_tournament.py` (`play`, `summarise`, `jobs_for`),
`src/simulation/ruleset.py` (`MatchRules`, `MatchResult`),
`src/control_bots_v1.py` (the whole law), `src/simulation/reward_v1.py`
(the fireteam terms), `tools/rl_smoke.py` (what it reports). The statistics
below are my arithmetic (binomial tails, a normal Monte Carlo in a scratch
script that imports nothing of the game). Nothing was run in the game.

## Verdict

**Survives with fixes.** 1 BLOCKER, 12 MAJOR, 14 MINOR.

The machinery's own tests (section 8's toy tasks, "masked choices sent must
stay 0", the per-option agreement against the majority baseline) are real
measures that can fail and mean what they say. The rung's GATE and its
EXPERIMENT are not yet measures: the gate has no number of duels or rooms, no
decision rule, and is read on the same held-back rooms that the in-run curve,
the choice of "best" network, the range's mark and the experiment's readings
all use, so picking a checkpoint by that curve alone can lift a 0.50 soldier
to about 0.59; "decided" drops draws, while the duel's reward (the clock's -3
seen through a 17 s look-ahead in a 60-90 s match) and an opponent that never
searches make hiding nearly free; `bots_v1` in a 1-vs-1 never throws, always
falls back when hurt and walks one fixed path. The experiment's three runs per
arm can separate the arms only when they differ by two to three run-to-run
spreads, it names no primary outcome, and its stated decision ("if B is no
faster, rung 3 starts from scratch") is overridden by section 0, where rung 3
re-runs every candidate anyway. Nothing in the document says what happens on
a fail, which is how rung 1's missed mark reached a design session. Every fix
is a paragraph or a number in the document; none needs a new system.

## Findings

### B1. BLOCKER: one held-back set is the validation set and the test set

**The design.** 5.1: "A bank of a few hundred rooms per task plus a held-back
set". 5.2: the yardsticks are measured "on each task's held-back rooms". 5.3:
the range's mark "on held-back rooms". 7.3: "Every so many rounds, the farm
plays the held-back rooms with the current network against the fixed opponent
and the win rate is one more point on the curve." Section 0: the gate is on
"rooms it has never seen"; "the best duel network of rung 2" is a candidate
for rung 3. Section 6 reads "decisions needed to reach half the duels won"
off the same curve. Section 4.4 makes every trainer number "the start of a
sweep".

**How it misleads.** Every choice that is made by looking at the held-back
curve (which checkpoint is "the best duel network", when to stop, which sweep
setting wins, whether memory on beats memory off, the arm comparison) is a
selection on the set the gate is then read on. The gate's number for the
chosen network is the maximum of many noisy readings, not a reading. With K
checkpoints of equal true strength, each read with standard error s, the
expected maximum is about 1.5 s above the truth at K = 10, 1.9 s at K = 20 and
2.2 s at K = 50 (normal order statistics). At 100 decided duels per reading
(s = 0.05) and twenty checkpoints, a soldier whose true rate is 0.50 shows
about **0.59**, i.e. a clear pass. The same holds for the sweep settings of
4.4 chosen on that curve. Rung 1 kept a tuning seed set apart from its
held-out set for exactly this reason (1j: tuning 1-20, held-out 101-130); the
rung 2 design dropped the split.

**Smallest fix.** Three disjoint seed lists per task, all written into the
repository before any training run: TRAINING (the bank), VALIDATION (the
in-run curve of 7.3, the choice of checkpoint, stopping, every sweep of 4.4,
memory on/off, the experiment's curves, "best duel network"), and GATE
(touched once per candidate network, after the network to be judged is named
in writing; the yardsticks of 5.2 are measured on it too, which is harmless
because they are not chosen on it). Write the rule: "a network is judged on
the gate set once; a second gate reading of a re-trained network is reported
with the count of readings beside it".

### M1. MAJOR: the gate has no number of duels or rooms and no decision rule

**The design.** Section 0 and 5.4: "on rooms it has never seen, the soldier
wins more than half of its decided duels against `bots_v1`". No count of
rooms, sides, seeds or duels; no rule for what "more than half" means for an
estimate.

**How it misleads.** Read as a point estimate above 0.50, the gate is close
to a coin toss near the bar. P(estimate > 0.50):

| decided duels | true 0.45 | true 0.50 | true 0.55 | true 0.60 |
|---|---|---|---|---|
| 60 (1j's cell size) | 0.18 | 0.45 | 0.74 | 0.93 |
| 100 | 0.13 | 0.46 | 0.82 | 0.97 |
| 200 | 0.07 | 0.47 | 0.91 | 1.00 |
| 400 | 0.02 | 0.48 | 0.97 | 1.00 |

To tell a true 0.60 from 0.50 at 5 % one-sided with 80 % power needs about
155 decided duels; to tell 0.55 from 0.50, about 620. The 1j note shows the
size of the noise in our own instrument: the same pairing on two seed sets
gave 0.250 against 0.368 (ship_demo, horde in) and 0.179 against 0.271
(freighter, horde out) -- differences of 0.09 to 0.12, the gate's whole
margin.

Worse, duels are not independent. `bots_v1` draws nothing (its own docstring:
"the bots draw nothing from `sim.rng` at all"; only its bullet cones are
drawn), so on a given room and side it plays the same route every time; a
learner evaluated by its most likely choices is also nearly deterministic.
Repeats on one room differ only through the cone draws. The unit of
independence is the (room, side) pair, not the duel. With m duels per
room-side and an intra-room correlation rho, the effective count is n /
(1 + (m - 1) rho): 200 duels as 40 room-sides of 5 at rho = 0.7 are worth
about 53 independent duels, SE about 0.07.

The match settings that move the draw rate and the difficulty are also left
open: the duel's clock is "60 to 90 s" (5.4), the range's "30 to 60 s"; the
sight rule is never named (1j measured `all_round` by CHOICE; the farm and the
tournament take it from the level's match.toml unless `--sight` is given);
the mix of one-, two- and three-room levels in the gate set is not fixed (a
one-room level is a quick-draw from spawn, where rung 1 found the door's
reaction cost decides).

**Smallest fix.** Write into 5.4, before any run: the gate set is R rooms
(suggest R >= 100, so 200 room-sides with the swap of 5.1), one fixed seed per
room-side (or two), a fixed clock (one value), `all_round` sight, no zombies,
a fixed share of one-, two- and three-room levels; the decision rule is "the
lower end of the 90 % interval of the decided win rate, computed by resampling
whole rooms, is above 0.50" -- or, if the lead prefers the plain point
estimate, the false-pass rate at that n is written beside the mark (at 200,
a true 0.45 passes 7 % of the time). Report the per-room-size rates beside the
pooled one.

### M2. MAJOR: "decided" hides draws, and the duel makes hiding nearly free

**The design.** 5.4: "Beside the win rate, the share of matches that reach
the clock is printed: a soldier that hides has not won." Printed, no rule.
`summarise` counts `draws` as every outcome that is not a win and computes
`a_rate = wins_a / (wins_a + wins_b)`; draws vanish from the gate's number.

**How it misleads.** A soldier that engages only in the rooms and sides where
it is favoured and hides in the rest passes. Wins 30 %, draws 50 %, losses 20
% is a decided rate of 0.60.

Three things make this likely rather than possible:

1. **The reward pays for it.** In the fireteam terms (`reward_v1.PROFILES`,
   which the duel weights 1/1 per 4.3) losing a duel costs about -10 - 1 - 0.5
   = -11.5, winning pays about +11.5, the clock -3. Undiscounted, hiding beats
   engaging below an engagement win chance of 0.37. But 4.4 discounts at
   0.99 per decision, a 17 s look-ahead, and the clock is 60 to 90 s away at
   the start: 0.99^360 = 0.027, 0.99^540 = 0.004, so the -3 is worth -0.08 to
   -0.01 when the decision to hide is made. Seen from the start of the match,
   hiding costs almost nothing and the break-even is close to 0.50 -- i.e. a
   learner that is not yet better than the bot prefers a draw, throughout
   early training. Design 6.2's reason for the -3 ("so that hiding until the
   end is not a plan") does not survive this look-ahead in a 60-90 s match.
   (Whether the trainer should change is lens 2's; the measurement
   consequence is mine.)
2. **The opponent never finds it.** `BotTeamV1._advance` walks to the last
   known enemy position, else to the centre of the enemy's spawn room; on
   arrival with nothing in sight `_walk_to` returns None and the bot stands
   there. It never searches. In a two- or three-room level a soldier that
   leaves its spawn room for a corner the bot's route does not see draws
   every time.
3. **The observation shows the time left** (design 3.2), so a soldier behind
   on health late in a match can run the clock out on purpose.

Also inside "draws": a 1-vs-1 in which both die in one tick (the attack slot
resolves together, #169) is a `MatchResult("draw", None)` to the tournament
but two eliminations to the reward (design 6.5), so the instrument and the
reward disagree about trades.

**Smallest fix.** Make the draw share part of the gate: "decided win rate >
0.50 (by M1's rule) AND the share of matches reaching the clock is at most the
yardstick's (bots_door vs bots_v1 on the same gate set, 5.2) plus 0.10".
Print draws split as clock / both dead. Add to 7.1's curves the clock share on
the validation set, and to the risks the line "hiding is the reward's local
optimum while the soldier is weaker than the bot".

### M3. MAJOR: `bots_v1` in a 1-vs-1 is a reduced opponent, and the document does not say so

**The design.** 5.4: "One rifleman against one `bots_v1` rifleman"; section
0: the gate's opponent.

**What the code does in a duel** (`src/control_bots_v1.py`): rule 2 (grenade a
group) needs `len(seen) >= 2`, so in a 1-vs-1 it **never throws**; rule 3
(fall back) fires when health is below `fallback_hp_frac` "with no living
teammate within `fallback_r`", which in a 1-vs-1 is always true, so **every
time it is hurt below the fraction it walks home**; with a target it
**stands still** to shoot (`clear_move_dir`) and fires only when its facing is
within the spread; without one it walks **one deterministic A* path** at walk
throttle to the soldier's spawn-room centre and **stands there** if nobody is
seen. No draw in any decision.

**How it misleads.** The gate then measures "beats a walking target that
shoots back from a standstill, never throws, retreats when hurt and never
searches". Two learned policies pass it that tell different stories: an
ambusher that pre-aims at the door between its spawn and the bot's (the route
is a function of the room, so this generalises to unseen rooms and is not
memorisation), and a hider (M2). Both would look purposeful in a hand-picked
replay. Neither is evidence the soldier can fight anything that moves
differently, which is what rung 3 needs. Conversely the gate is not obviously
easy: in one-room levels both start in sight and the bot acts every tick
against the soldier's every fourth, the reaction cost 1j measured at about 14
points.

**Smallest fix.** One paragraph in 5.4 stating the bot's duel behaviour (as
above) so the gate is read for what it is; beside the gate (not part of it)
print the same soldier against the layered `bots` in the same 1-vs-1 rooms,
and against `bots_door`, as 13.1 item 4 already does for rung 3. "Never tune
`bots_v1`" stays.

### M4. MAJOR: the yardsticks are measured and then not used to set any mark

**The design.** 5.2: "Without these a pass mark is a guess, which is the
lesson of 1j." But section 0: the gate is "the roadmap's, unchanged", and 5.3:
"Mark (the lead's guess, to be reset once 5.2 is measured): ... at least nine
times in ten." No rule says how a yardstick turns into a mark.

**How it misleads.** The yardsticks can come back in three ways, and the
document has no sentence for any of them:

- `bots_door` (the competent player through the same door, the closest thing
  to a ceiling for a copy and a reference for a learner) wins **under 0.50**
  against `bots_v1` in the duel rooms. Plausible: rung 1 found the door costs
  about 14 points against bots that act every tick, and `bots_v1` acts every
  tick. Then the gate asks a learner to beat the reference by more than the
  door costs the reference -- possibly fine (a learner can anticipate), but
  the document should say before the run whether the gate stands, rather
  than meeting the question as rung 1 did, after the fact.
- `bots_door` wins **well over 0.50** (the layered bots beat `bots_v1` on the
  ships). Then the gate is passed by the copy before any reward learning (M5).
- the range: `bots_v1` and `bots_door` against idle targets will clear close to
  every room in time, the random player close to none, so the yardsticks
  bracket 0 and 1 and the "reset" is an unconstrained choice made after
  seeing data.

The `bots_v1` row of 5.2 ("in the soldier's seat") against `bots_v1` is not a
skill yardstick: by symmetry it measures the side bias of the room bank.
Note also that `summarise` computes `a_rate` only when the two controllers
differ, and `jobs_for` adds the side-swapped job only when `a != b` (line
354), so this row comes out as team-0 wins and draws, with the swap supplied
only by 5.1's swapped rooms.

**Smallest fix.** Write the mark rules before the yardsticks are run. For the
range: "the mark is min(0.9, the `bots_door` clear rate minus two standard
errors)". For the duel: "the roadmap's gate stands whatever the yardsticks
say; if `bots_door` wins under 0.40 of decided duels in these rooms, the
gate's failure is read as the door's reaction cost first (design 13.1 item 4
re-opened), not as the trainer's". Relabel the `bots_v1` row "side balance:
team-0 share of decided, must lie within 0.5 +- 2 SE".

### M5. MAJOR: the gate does not say which network must pass it

**The design.** Section 0: "the soldier wins more than half of its decided
duels"; the goal is "a learning curve that rises". Section 6: arm B starts
from the copied network, whose records include "one door bot against one
first-generation bot" in the duel rooms (3.1).

**How it misleads.** If the copy alone already wins half of the decided duels
against `bots_v1` (see M4: plausible if `bots_door` does), arm B's start
passes the gate with no learning from reward, and "the trainer works" is not
shown by the gate at all. Even a modest rise of B above its start is
uninformative unless its starting reading is printed.

**Smallest fix.** One line: "the gate is passed by a network trained by
reward, read beside its starting network on the same gate rooms; for arm B
the copy's own reading is printed and the rise must exceed M1's interval; at
least one arm-A run is read on the gate too, so that the trainer is proven
without the copy".

### M6. MAJOR: the copy's duel records may include the gate rooms

**The design.** 3.1: the records are made again with "the duel rooms of
section 5, with one door bot against one first-generation bot, so that the
copy has also seen the task it will be compared on". Which rooms is not said.

**How it misleads.** If the recording uses the held-back (gate) rooms, arm B
has seen them through the copy and arm A has not: the experiment is biased
toward B, and for B the gate's "rooms it has never seen" is false. With a
deterministic opponent (M3), the copy can carry a per-room ambush straight
into the gate.

**Smallest fix.** "The copy's duel records are made on the TRAINING bank of
5.1 only; the validation and gate lists of B1 are never recorded."

### M7. MAJOR: no stated bar for the copy before it becomes arm B's start

**The design.** 3.3: agreement per head and per option on held-back matches
is reported against the majority baseline; "Does it play?" is "reported, not
a gate"; DAgger is "reserved ... until the tournament says it is needed", but
no number says when the tournament says so.

**How it misleads.** Every measure in 3.3 can be passed by a copy that is
useless as a start: high agreement is mostly "hand: nothing", "trigger:
release", "feet: same as last time" (the per-option rows help, but have no
threshold). If B starts from such a copy and is no faster than A, section 6
concludes "copying did not help" about a copy that never played. The
experiment's negative answer is then uninterpretable; its positive answer is
fine.

**Smallest fix.** Before arm B starts: "the copied soldier, in the duel's
validation rooms against `bots_v1`, wins at least half as many decided duels
as `bots_door` does there, and beats the random player's rate by more than M1's
interval; otherwise DAgger (3.3) is run first, or the experiment is reported
as 'copy too weak to answer'". Read it through the same tournament counting as
the gate (m2).

### M8. MAJOR: three runs per arm, no primary outcome, and "overlap" read as "no faster"

**The design.** Section 6: "At least three repeats of each arm ... Read off:
decisions needed to reach half the duels won; the win rate at a fixed budget;
and, by eye, what B does that A does not. ... If B is no faster, rung 3 starts
from scratch."

**How it misleads.**

- **What three against three can separate.** With run-to-run spread sigma
  (the document itself says two runs of one setting can differ more than two
  arms), the chance that all three runs of the better arm beat all three of
  the other (complete separation, the only result 3 v 3 can call clear) is
  0.05 if the arms are equal, about **0.26** if they differ by one sigma,
  **0.62** at two sigma, **0.89** at three (normal Monte Carlo). A t-test
  at 3 v 3 needs about three sigma for 80 % power. Deep-RL seed spreads are
  large (Henderson et al. 2018 split ten runs of one algorithm into two
  groups of five with "significantly different" curves; Agarwal et al. 2021
  on few-run evaluation -- both recalled, not in our survey). So the likely
  result is overlapping ranges.
- **Absence read as evidence.** The decision line turns that likely overlap
  into "B is no faster, rung 3 starts from scratch". Overlap at n = 3 says
  "not shown", not "no difference".
- **No primary outcome.** Three readings are named and none is primary, so
  whichever favours an arm can be quoted. "Decisions to half": undefined for
  a run that never reaches half (censored: the likely case for some arm-A
  runs within budget); and the first crossing of a noisy curve comes early --
  on a curve rising linearly through 0.5, the first evaluation reading 0.50
  happens on average when the true rate is 0.46 at SE 0.05 per point, 0.43 at
  SE 0.07, 0.39 at SE 0.10. The "fixed budget" has no value.
- **Head start versus speed.** B begins at the copy's strength. If the copy
  is near 0.5 (M5), "decisions to half" is the copy's start, not learning.
  That is a legitimate meaning of "shortens the path" only if the copy's cost
  is on the bill (the records: about an hour of 12-worker tournament and 1.1
  million bot decisions; the copy's training: about 20 minutes, 9), and B's
  value-only warm-up rounds are counted in B's decisions. Neither is said.
- **Habits are the roadmap's half of the question** ("or leave habits to
  unlearn") and are read "by eye". They are countable from the stored
  decisions: share of feet choices at full throttle, reloads by choice
  (hand = reload with rounds left), plants, throws per match, per arm and per
  evaluation. Without A's counts beside B's, "B never sprints" cannot be told
  from "nobody sprints in these rooms".
- **Settings.** Section 4.4's sweeps are not said to be finished before the
  experiment, nor run on which arm; settings tuned on A (or on the range) may
  wreck the copy, settings tuned on B favour it.

**Smallest fix.** Pre-state in section 6, before any run: the PRIMARY
outcome is the validation win rate (decided, M2's draw cap beside it) at two
budgets, B (the duel's planned length, B's warm-up included) and 2B; the
secondary is decisions to first sustain 0.50 over three consecutive
evaluations, censored at 2B; the habit counters above per arm. All settings
are frozen before the experiment, chosen on the range or on arm A's validation
curve, and the same for both arms bar B's warm-up. Same seed lists, paired:
run i of A and run i of B share the env seed and episode draws. The reading
rule: "B helps" if all B runs beat all A runs at budget B; "B hurts" the
reverse; otherwise "not separated at three runs each", and then the decision
falls to section 0's rung-3 measurement (M9), not to "start from scratch". If
three PCs allow it, five runs per arm make one-sigma differences visible about
half the time; say which is chosen.

### M9. MAJOR: sections 0 and 6 decide rung 3's start twice, differently

**The design.** Section 6: "If B is no faster, rung 3 starts from scratch."
Section 0: "the first stretch of rung 3's training is run from each
candidate [nothing, the copied network, the best duel network], three times
each, and the duel network is kept only if its runs are clearly ahead of the
others' (their spread does not overlap) at the same number of decisions."

**How it misleads.** If section 0 holds, rung 3 re-runs "nothing" and "the
copy" anyway, so section 6's outcome changes no decision -- then either it is
an experiment run for its own sake (fine, but say so) or section 6's sentence
is dropped. If section 6 holds, the copy is struck before rung 3 on a
duel-task result, although the copy was made from fireteam records and is the
candidate most likely to help a fireteam. Section 0's own rule is also
under-specified as a measure:

- **what is compared** at rung 3 is not named (win rate against the layered
  bots on validation ships? the team reward?);
- **"the first stretch"** has no length. Its length decides both the cost
  (nine runs of rung 3 at 720 to 2 300 decisions a second; lens 4's to price)
  and the bias: a single early comparison favours whichever start has skills
  that pay at once (the duel network's aim) even if it plateaus lower;
- **only the duel network** gets a rule; nothing chooses between "nothing"
  and "the copy";
- **"spread does not overlap"** at three runs each is complete separation:
  a 5 % false keep in the named direction if the duel network is no better,
  and a 26 % keep if it is one sigma better. That is a deliberately
  conservative rule and matches the stated default (not used unless it
  earns it); write that it is chosen to be strict, so a "not kept" reads as
  "not shown better", not "worse".

**Smallest fix.** Pick one place for the decision (section 0, at rung 3, on
rung 3's own task), delete or reword section 6's last sentence to "the result
is carried to rung 3's comparison as a prior expectation", and give section
0's rule its measure (the validation win rate against the layered bots), its
stretch length (a fixed number of decisions), two reading points (the
stretch's middle and end; ahead at the end is required), and a rule for copy
versus nothing (the same non-overlap rule, default nothing).

### M10. MAJOR: no written rule for any fail, and no budget for re-trying

**The design.** No section says what happens when the range misses its mark,
the memorise test fails, the copy plays badly, the duel misses the gate, the
two halves of the gate disagree, or the experiment is inconclusive. Rung 1's
1j note had one ("Below the mark: the variants say which coarseness costs ...
the finding goes to a design session before rung 2. A failure is never a
patch to fix"); rung 2 has none, and design 13.1 shows the cost: the missed
mark was ruled on afterwards.

**How it misleads.** Reinforcement learning invites the quiet loop "fail,
adjust a setting, run again, read the held-back rooms again" until a pass,
which is the selection of B1 by another name, with no record of how many
tries were made. And a split gate (the win rate passes, Erik's eye rejects an
ambush-or-hide exploit; or the eye likes it and the rate is 0.48) has no
owner.

**Smallest fix.** One table in section 7 or 12, per measure: the mark; what a
fail means first (range: the trainer or the pointer -- back to the toy tasks;
memorise: a fault, stop; copy below M7's bar: DAgger; gate below: the
suspects in order -- the observation, the door's reaction cost (13.1 item 4),
the reward's draw incentive (M2), then the trainer's settings; experiment not
separated: M8's wording); a budget ("at most N training configurations are
read on the validation set before the gate reading; the count is reported
with the gate"); and "both halves of the gate are required; a split goes to a
design session with the replays and the numbers, and is never resolved by
re-running until one half moves".

### M11. MAJOR: Erik's eye has no protocol

**The design.** Section 0: "the replay shows it aiming, firing and moving on
purpose. The second half is Erik's eye." 7.2: watching is "a recorded
tournament match played back with `main.py --replay`". Nothing says which
replays, how many, chosen by whom, or when relative to the win rate.

**How it misleads.** Replays picked by the lead (the best ones, or ones that
show an intended behaviour) make the eye half pass by selection; seen after a
good win rate, the judgment is anchored. Exactly the exploits of M2 and M3
(hiding, ambushing a fixed route) are visible only in the right replays: a
hide shows in a draw, never in a win.

**Smallest fix.** "Erik watches k replays (suggest 8) from the gate set,
drawn by a fixed rule written before the gate run (e.g. the first k
room-sides of the gate list by seed), stratified to show wins, losses and
draws in their measured proportions, before he is told the win rate; his
verdict is written in the rung's note in his words before the number is
added." For section 6's "by eye, what B does that A does not": the replays
are shown with the arm hidden.

### M12. MAJOR (parked, fix before it is run): design 13.4's spot-versus-direction protocol is confounded

**The design** (13.4, referenced by rung 2's section 11): "A second action
spec beside v1 in which the feet head picks a cell of a small grid of offsets
around the soldier ... and the decoder walks the first step of the route there
each tick. ... The same network is trained on the range and the duel with each
spec, three runs each, and compared on decisions to reach the pass mark and
on what Erik sees. Records for the copy are made again with the bots'
destination as the choice."

**How it misleads.** The difference measured would be decided by things other
than "a spot instead of a direction":

- **Per-tick steering.** The spot spec's decoder re-routes every tick; v1
  holds one direction for four ticks. Rung 1 found reaction between decisions
  is the door's main cost (13.1). A win for spots may be a win for tick-level
  path-following -- a game-side reflex 13.1 item 5 deliberately did not take.
- **Head size and the entropy push.** 49 options against "hundreds": the
  same 0.01 entropy weight (4.4) acts on a maximum entropy of ln 49 = 3.9
  against ln 200 = 5.3, so exploration differs by construction; "the same
  network" is not the same network (the head differs).
- **Copy quality.** If either arm starts from a copy, the spot copy is likely
  better (the bots think in destinations, 13.4's own argument), so the arms
  differ in their start, not their form.
- **The range** barely asks for movement (idle targets, small rooms); the
  stated trigger for the experiment is "a soldier that ... cannot learn to
  cross a room", which the range does not measure.
- **Three runs each** (M8), and "the pass mark" -- which one, for which task --
  is not named.

**Smallest fix.** In 13.4: both arms from scratch; a third arm "v1 with the
decoder re-steering the held direction each tick" (or the spot spec with its
route frozen for the decision's four ticks) to separate steering from naming;
the entropy weight scaled by each head's maximum entropy, or swept for both;
the primary task the duel (plus a crossing task if one exists); M8's outcome
rule.

### m1. MINOR: "rooms it has never seen" from a small generator

5.1: one to three rooms, doors, a few crates. The space of distinct
SITUATIONS (where the bot comes from relative to my spawn, how many doors
between) is small; a few hundred training rooms cover it, and held-back rooms
are new tiles of familiar shapes. The gate then measures interpolation over
layouts, which is a fair rung-2 claim but not what "never seen" suggests for
rung 3's "ships never seen". Fix: report the same soldier's rate on its
training rooms beside the gate rooms (the gap is the generalisation finding);
fix the gate set's mix of one-, two- and three-room levels (M1); optionally
hold out one structural class (e.g. every three-room L) as a second, harder
reading.

### m2. MINOR: three instruments count one number

5.2 measures the random player "through the farm, as `tools/rl_smoke.py`
already plays it" and the bots "through the bot tournament"; 7.3 reads the
in-run strength through the farm; the gate (presumably) through the tournament
(7.2). `rl_smoke` reports throughput only (env-ticks/s, step ms), no outcome,
and drives every row with random choices. Fix: one counting function for
every reading of a win rate -- `bot_tournament.summarise`'s "decided" and
draws, applied to farm episodes too, or the random player as a tournament
controller beside `net` (7.2) -- and the same match.toml (clock, sight,
zombies 0) for the learner and every yardstick.

### m3. MINOR: sampled or most-likely choices, unstated

7.3 and the gate do not say whether the network is evaluated with its choices
drawn (as trained) or the most likely one taken. They differ, sometimes a
lot, and the most-likely form makes room repeats nearly identical (M1's
clustering). Fix: the gate and the curve read the drawn policy with fixed
seeds; the most-likely form printed beside.

### m4. MINOR: the range's mark has no n, no composition, and a silent role

"At least nine times in ten": on 20 rooms a true 0.85 clears the mark 40 % of
the time, a true 0.95 misses it 8 % (binomial). The one-to-four targets and
30-60 s clock change the difficulty per room; a self-kill by the soldier's
own frag, a self-blinding by its flash, or its charge, ends as a loss and
should be counted separately (`summarise` already has `self_per_match`).
Whether the duel waits for the range's mark is not said. Fix: n, fixed
composition, the derivation rule of M4, self-harm printed, and "the duel
starts when the range passes or its fail is ruled on (M10)".

### m5. MINOR: "memorise 1 000 decisions" has no threshold and cannot see mask faults

"It must reproduce nearly all of them": no number. Consecutive decisions of
one soldier standing still can carry near-identical observations and
different labels (the bots' hidden state), so "all" can fail without a fault
if the 1 000 are drawn consecutively. And a network that ignores the masks
memorises just as well: this test checks the data path and capacity, not the
choosing module (the toy tasks of section 8 do that, rightly). Fix: draw the
1 000 spread across matches; "at least 99 % most-likely agreement on every
head that counted"; say what it does not test.

### m6. MINOR: training-set error is not all hidden state

3.3: "what stays wrong even there is the bots' hidden state". Training error
is hidden state plus under-fitting (capacity, too few passes, the 2 % floor's
pull). Fix: read it at convergence and at double width (if it falls, it was
under-fitting); or count it directly as the disagreement among near-identical
observations.

### m7. MINOR: splitting the copy's records "by match" leaks

1j records up to six door-team records per (level, seed) (both sides with the
horde, both sides without, the horde pairings), and a mirrored pair shares
seed and layout. Split by (level, seed), not by record file.

### m8. MINOR: per-head agreement treats a near miss as a miss

The feet head's 16 directions are the nearest bins of the bot's bearing (1j:
39 % of decisions not on a bin, mean error 4.5 deg); one bin off scores as
wrong as the opposite way. Print the mean angular error of the feet and the
aim directions beside the agreement, and for the pointer a baseline of "aim at
the nearest enemy entry" beside "most common option".

### m9. MINOR: single runs where the document says single runs prove nothing

5.4: "With the memory; a second run without it". 3.2/3.3: memory on and off
for the copy. Section 12's risk row: "three repeats of anything that is
compared". Under all-round sight design 8.3 itself expects "little memory" to
be needed, so the likely result is a null that one run cannot show. Fix:
three runs each, or the comparison labelled indicative and not used for any
decision.

### m10. MINOR: draws mix the clock and mutual kills

See M2's last paragraph: print "clock" and "both dead" separately, and note
that the reward treats the second as two eliminations.

### m11. MINOR: the in-run curve has no n and no stall rule

7.3: "every so many rounds" with no count of matches per point; at 40 matches
a point carries SE ~0.08, so the curve cannot show a stall from a slow rise.
7.1 lists eleven curves and no single reading for "stalled" or "solved". Fix:
n per point (and the band drawn), and one rule: "stalled = the validation win
rate has not risen by two standard errors over the last K points while the
entropy of the heads that counted is falling"; "solved" = M1's rule met on
validation.

### m12. MINOR: arms on different PCs

4.6 / section 6: three PCs run the repeats. If arms are assigned by PC, wall
clock and any machine-specific difference confound the arms. Fix: compare at
equal decisions only (as section 0 already says for rung 3), and put one run
of each arm on each PC.

### m13. MINOR: clocks given as ranges

5.3 "30 to 60 s", 5.4 "60 to 90 s". The draw share and the clock's weight in
the reward (M2) depend on the value. Fix one value per task before the
yardsticks are measured.

### m14. MINOR: the copy's "Does it play?" row measures the fireteam, arm B uses the duel

3.3 measures the copied team in the four-a-side tournament. Arm B starts from
the copy in the 1-vs-1. Add the copy's duel row (M7) beside the tournament
rows.

## Holds

- **Section 8's toy tasks** each name the fault that makes them fail (the
  pointer following a place, a graded choice that did not count, the memory
  and dead rows); "no run with the game starts before all of these are
  green" is a real gate.
- **"The count of masked choices sent (must stay 0)"** (7.1) is a hard
  invariant with a zero mark, the same as 1j's gate 1.
- **Per-option agreement against the majority baseline** (3.3) is the right
  correction for "hand: nothing is right 99 % of the time".
- **Sides by construction** (5.1): each room also made with the spawn rooms
  swapped, so the soldier's seat is balanced without environment code; with
  M1's room-side pairing this is the right unit.
- **The instrument is reused, not rebuilt**: the learned team enters the bot
  tournament as a player (7.2), so wins, survivors, rounds per kill and
  friendly fire are counted by the code that counted the bots.
- **`bots_v1` is a fixed opponent**: deterministic, never tuned, drawing
  nothing, so it is the same opponent on every PC and in every run -- good for
  comparisons, once its reduced duel behaviour is written down (M3).
- **"Reported, not a gate"** for the copy's play (3.3) is honest, and DAgger
  has a stated trigger in principle (M7 asks for its number).
- **"Earns it" leans the right way**: the default is not to carry the duel
  network, and non-overlap at three runs is strict (5 % false keep), so the
  rule cannot carry a habit-laden network forward by luck (M9 asks for its
  measure and length).
- **The draw share is at least printed** beside the gate (5.4), and the
  hiding risk is named there; M2 asks for it to count.
- **The copy's "plays worse than it predicts" warning** (3.3) correctly sets
  expectations for the compounding-error effect.

## Not checked

- Whether `BatchedEnv` can put a bot controller on one team and a learner on
  the other, whether the random player can be a tournament controller, and
  whether the training-room generator can be built on `embed` (lens 1).
- PPO, grading, the floor of 3.2, the value warm-up of arm B as mathematics
  (lens 2); the discount's effect on the clock is noted only for its
  measurement consequence (M2).
- The cost of the experiment's runs, of "earns it" at rung 3, of in-run
  evaluation and of a 100-room gate set (lens 4).
- Whether `bots_door` beats `bots_v1` 1-vs-1 in small rooms; M4 and M5 turn on
  that number and it does not exist yet.
- The roadmap's "rating against a fixed pool" for rung 2 against the
  design's deferral of ratings to rung 3 (a scope question, not a measure).
