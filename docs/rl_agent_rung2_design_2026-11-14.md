# RL agent, rung 2 (first light): the trainer, the copying step, the first tasks

2026-11-14. **PROPOSED, version 1.** The lead's (the trainer and the network
are the lead's to decide, skill `game-ai-lead` section 1), written for Erik to
veto. Not yet critiqued. Nothing is built from a section that is not AGREED.
Documents only: written while the residency arc's P6 (#29) is changing the
unit code, so no number in it comes from a new run.

It continues `docs/rl_agent_design_2026-11-11.md` (cited as "design x.y"),
whose sections 6, 7, 8 and 13 it builds on and does not repeat, and
`docs/rl_agent_roadmap_2026-11-11.md` (rung 2). The method is explained in
`docs/rl_lesson_0_learning_from_reward_2026-11-11.md` ("lesson 0"); the
sources are in `docs/research/rl_agent_survey_2026-11-11_raw/05_trainer_state_of_the_art.md`
("digest 05"). A number is marked **measured** (with where) or **estimate**.

## 0. Rung 2 on one page

**The goal** (roadmap): a learning curve that rises, and a learned soldier
Erik can watch.

**What is built, in this order:**

| step | what happens | a game runs while it trains? | what Erik sees at the end |
|---|---|---|---|
| 1. The network | network v1 (design 8) as code, with its tests | no | a printed table of its parts and weights |
| 2. Copying the bots | the network learns to predict the door bots' recorded choices | no | the first network-driven fireteam, as a replay in the game window |
| 3. The trainer | learning from reward, proven on three toy tasks made for our own failure modes | no | the toy tasks' curves; the lesson document |
| 4. Shooting range | one soldier, standing targets, small random rooms, from scratch | yes | a curve, a replay |
| 5. Duel | one soldier against one first-generation bot, random rooms | yes | a curve, replays |
| 6. The experiment | the duel started from the copied network against the duel from scratch | yes | one table: did copying help or hurt |

**The gate** (roadmap, unchanged): on rooms it has never seen, the soldier
wins more than half of its decided duels against the first-generation bot
(`bots_v1`), and the replay shows it aiming, firing and moving on purpose.
The second half is Erik's eye.

**What waits for P6:** every run with the game in it (steps 4 to 6), the
final copy (step 2 is re-run on fresh records), and every patch that touches
a game file. Steps 1 and 3 touch no game file and use no farm.

**Rung 2's networks are throwaway.** They exist to prove the machinery and
to answer the copying question. Nothing trained here is carried to rung 3
unless it earns it.

## 1. Where the code lives and what it may touch (PROPOSED)

The roadmap's rule: the trainer is outside the game's source tree and meets
the game only through the batched environment.

- **A top-level folder `agents/` in this repository**, beside `src/`, not
  inside it. One repository, so that one tag fixes the game and the trainer
  of a training run together (the Training snapshot rule).
- **Nothing in `src/`, `renderer/`, `ui/` or `main.py` imports it.** A test
  scans for that. The game never depends on PyTorch.
- **It reaches the game through four existing seams and no other:**
  `rl_env.BatchedEnv` (training and measuring in the farm);
  `control_bots_door.read_record` (the copying data); the `policy`
  controller's decoder with `rl_env.observe_v1` (a learned team in a played
  match, section 7.2); the spec tables, for shapes (`BatchedEnv.layout`).
- **PyTorch from the shared ML stack** (`ml-stack` skill; 2.15 on the
  desktops). No training framework: digest 05 found that every framework
  wants to own the loop that `BatchedEnv` already is, and that the two the
  July survey named no longer fit (one takes only C environments, the other
  has no Windows support and no release since 2023).
- **Settings in two files, both outside the repository:** the run file (the
  environment's, as today) and a **training file** beside it (network spec,
  the numbers of section 4, the task, the reward group weights, how often to
  save and measure). Both files' sha256 go into the run manifest.

## 2. The network as code (PROPOSED)

Design section 8 is the specification. What this section adds is how it is
built so that the usual faults cannot hide.

- **One implementation of "choosing".** A single module turns the network's
  raw outputs and the masks into the chances of each option, draws a choice,
  and computes the probability of a given choice. It is used in three places:
  when a soldier acts, when the trainer learns from reward, and when the
  network copies the bots. So a fault in the masks or in "which choices
  counted" shows up in the cheapest of the three (copying) and is not written
  twice.
- **The inputs are the stored arrays, as served.** The network acts on
  exactly the arrays that are kept for learning (same number format), so the
  probability of a choice can be recomputed later and compared (design 8.6).
- **True sizes, now that rung 1 fixed them** (design 8.1's were estimates):
  the soldier's vector 97 numbers; the list 39 entries of 67 numbers on a
  ship with 6 roster places; the near window 13 planes, the far window 15;
  the choices 49 (feet), 57 (aim), 2 (trigger), 13 (hand), 5 (argument). The
  value network's list is 64 entries of 85. The weight count is recounted
  from these and printed by a test; if it leaves the 1 to 2 million asked
  for, that is reported before anything is trained.
- **Memory can be switched off** in the spec, for the range and for the
  first copy (design 8.6's order of work).
- **A network is a named spec** ("v1"); a checkpoint names the network spec,
  the observation spec and the action spec it was made for, and loading it
  into anything else is refused.

## 3. Copying the bots (PROPOSED; the order AGREED, design 13.3)

In the literature this is called behaviour cloning: ordinary supervised
learning, where the "label" is what the teacher did.

### 3.1 The data

- **The records** of patch 1j: per decision of a door-bot soldier, the
  observation it was served, the masks, the five integers that were sent,
  the argument's mask, the rewards. About 1.1 million decisions of living
  soldiers in the 600 recorded matches (estimate from the 663 000 measured
  in 360 of them).
- **They are recorded again after P6,** because the poison law changes how
  matches with poison gas play. Three changes at the same time:
  - more ships: about ten generated freighters with a room graph in place of
    two, two of them held back for measuring, so that "does the copy work on
    a ship it never saw" can be asked at all;
  - the duel rooms of section 5, with one door bot against one
    first-generation bot, so that the copy has also seen the task it will be
    compared on;
  - no change of format.
- **A correction to design 13.3:** recording again is "about ten minutes" there.
  The 1j note's own walls say 600 door matches at about 65 s each on 12
  workers: **about an hour**. Still cheap; the records are still not precious.
- **Unpacked once.** One decision is about 40 KB as served, so the whole set
  is about 45 GB (estimate), more than any of the PCs' memory. The records
  are unpacked once into one file per array on disk that is read in place,
  rows of dead soldiers and empty roster places left out.

### 3.2 What is learned, and how

- **The loss:** make the bot's recorded choice more likely, head by head,
  under the masks, counting only the heads that counted (design 8.2: an
  argument when the hand verb takes none does not count; a head with one
  legal option does not count). This is the same probability the trainer
  uses in section 4.
- **A floor under every legal option.** The bots never sprint, never reload
  by choice and never plant, so a pure copy would give those options a
  chance of about zero, and learning from reward afterwards could not find
  them again. The copy is therefore trained toward "the bot's choice with 98
  %, the other legal options sharing 2 %". The 2 % is a starting value.
- **Without memory first, then with.** The bots' choices depend on things the
  observation does not show (the path they planned, the position they
  committed to). A memory can recover part of that from what the soldier did
  a moment ago. Both are trained; the difference is one of the results.
- **Split by match and by ship:** some matches for training, others for
  measuring, and the two held-back ships never trained on.

### 3.3 What is measured (each can fail)

| question | measure | reading |
|---|---|---|
| Is the machinery sound? | the network trained on 1 000 decisions only | it must reproduce nearly all of them. A network that cannot memorise 1 000 decisions has a fault |
| Does it predict the bots? | agreement with the bot's choice on held-back matches, per head and per option | against "always the most common option". Per option, because "hand: nothing" is right 99 % of the time: the throws and the shoves are counted on their own |
| How much does the observation not explain? | the same on the training matches | what stays wrong even there is the bots' hidden state |
| Does the memory help? | both of the above, memory on and off | |
| **Does it play?** | the copied team through the door in the bot tournament: against `bots_door` (its teacher), `bots`, `bots_v1`, and the horde alone | **reported, not a gate.** At best it plays like `bots_door`. A copy usually plays worse than it predicts: one small mistake puts the soldier somewhere its teacher never was, and the mistakes add up |

**If the copy plays badly although it predicts well,** there is a known
remedy, and our door bots make it cheap: let the network drive, let the bots
run beside it on their capture proxy (the 1j construction) and record what
they would have done in the situations the network got itself into, then
train on those too. (The literature's name is DAgger.) It is reserved, not
built, until the tournament says it is needed.

## 4. Learning from reward: the trainer (PROPOSED)

PPO in the arrangement where every soldier shares one network and a
training-only value network sees the whole match (roadmap rule 8; lesson 0).
Synchronous: the farm already steps all its matches together.

### 4.1 One round

1. **Collect.** The farm plays T decisions in each of its N matches. At each
   step the network answers for every learning soldier of every match in one
   pass on the graphics card.
2. **Grade.** The value network estimates, for every stored decision, the
   reward still to come; from that and the rewards that did come, every
   choice gets its grade (lesson 0, sections 5 and 6).
3. **Learn.** A few passes over the batch, under the clip (lesson 0, 7).
4. **Throw the batch away, raise the weights' version by one, repeat.**

### 4.2 The stored decision

What is kept for each soldier at each decision (design 8.6): the observation
as served; the value network's full view; the five choices; the masks and
the argument's mask as they were; which heads counted; the probability of
each choice; the five reward numbers; the two row flags; and, once per
stored stretch, the memory vector where the stretch began.

- **About 63 KB per decision** (40 KB observation, 22 KB full view; my
  arithmetic from the spec's shapes). A batch of 8 000 decisions is 0.5 GB; a
  fireteam batch of 32 000 would be 2 GB. On the 16 GB desktops, beside 16
  farm workers, that sets the batch size as much as any theory does.
- **A dead soldier's row** keeps its rewards and its value estimate and has
  no choice: it teaches the value network what the death was worth and
  teaches the soldier's network nothing (design 6.3).
- **An empty roster place** is stored nowhere and counts in nothing.

### 4.3 Grading

- **Five estimates, one per reward group** (design 7.2). Grades are computed
  per group and added with the training file's weights. For rung 2's
  one-soldier tasks the weights are: outcome 1, own 1, team 0, friendly fire
  0, order 0. So the "range" and "duel" profiles of design 6.9 need no new
  environment code: they are a weighting of the five numbers the environment
  already returns under the "fireteam" profile.
- **Endings.** A win, an elimination and the clock are all real endings
  (design 6.5), so nothing is still to come after them. The only cut is the
  end of a batch in the middle of a match; there the value network's estimate
  stands in for the rest.
- **A match that ends inside a step:** its last rewards belong to the ended
  match; the next stored decision is the new match's first, with the memory
  set to zero.
- **The value targets are rescaled** by their own running mean and spread,
  per output (the MAPPO paper calls this a must; it is the one place where
  the trainer keeps running statistics, and it is of the targets, not of the
  network's inputs).

### 4.4 The update: starting numbers

Every one is the start of a sweep, with where it comes from.

| number | start | why |
|---|---|---|
| look-ahead (the discount) | 0.99: about 17 s | the range and the duel last 30 to 90 s. Lengthened at rung 3 |
| blend of short and long grading (lambda) | 0.95 | the usual value; 0.9 is the main alternative |
| clip | 0.2 | the MAPPO paper prefers smaller; 0.1 is the first sweep |
| passes over a batch | 4, stopped early when the policy has moved too far from where it was | the open question of digest 05: the squad benchmarks use 10 to 15, OpenAI Five found 8 ruinous. With a slow simulator, re-using data is the main lever, so this is the first sweep (2, 4, 8) |
| pieces per pass | 4 | |
| step size | 0.0003, Adam, falling to zero over the run | the common default |
| push toward variety (entropy) | 0.01, on the heads that counted only | design 8.2 |
| batch | 8 000 decisions for the range and the duel | memory (4.2) and a round of well under a minute |
| limit on one update's size (gradient norm) | 0.5 | the common default |

### 4.5 Memory

The part that most often hides a fault (lesson 0, section 10), so it comes
second: the range is trained without it.

- A stored batch is cut into **stretches of 16 decisions** (2.7 s) per
  soldier. Learning replays each stretch in order from the memory vector
  stored at its start, and takes whole stretches, never single decisions.
- The memory is zero at a match's first decision.
- Test: recomputing the memory along a stored stretch reproduces what the
  soldier had when it acted.

### 4.6 The boundary between collecting and learning (Erik's note, 2026-11-11)

- **Two parts with one hand-over each way.** The collector takes weights and
  returns a batch; the learner takes a batch and returns weights. A batch is
  plain arrays with a header: the weights' version it was played with, the
  hashes of the run file and the training file, the game's build id.
- **The learner refuses a batch played with other weights than its current
  ones.** (PPO may only learn from play by the current policy. Learning from
  slightly older play needs a correction; that is the later, asynchronous
  form, and the header is what makes it possible.)
- **In rung 2 both parts run in one process on one PC.** The boundary costs
  nothing now and is what a spread-out run needs later.
- **What a spread-out run would have to move, measured in advance:** the
  observations. 63 KB per decision is 0.5 GB per batch of 8 000 before
  packing. The PCs stand in different places, and at least the laptop
  reaches the work desktop only through Tailscale's relay (no direct path,
  per the environment notes of 2026-10-09), which will not carry that.
  The weights going the other way are about 6 MB. So one run over three PCs
  needs a direct link between them, and that is measured before anything is
  built for it.
- **Until then, as Erik said:** three PCs run three experiments and share
  saved networks, which are small. Section 6's experiment is built to use
  exactly that (several repeats of each arm).

### 4.7 Saving, and which code a run used

- **A saved network** holds: both networks' weights, the learner's state,
  the value targets' statistics, the counters (decisions, batches, weights'
  version), the three spec names, both files' hashes, the game's build id,
  and what it was started from (nothing; a copy of which records; or which
  saved network).
- **Outside the checkout,** in the run's own folder (`[run] out_dir`, which
  the environment already refuses inside the checkout).
- **A run can be stopped and resumed.** Not bit for bit: the graphics card's
  arithmetic is not exactly repeatable. The matches are (every one replays).
- **Tags.** A run whose network is kept (shown to Erik, used as an opponent,
  or counted for the gate) runs from an annotated tag in its own checkout, as
  the Training snapshot rule says; the tags are Erik's to place. A short
  exploratory run may run at any clean commit: the manifest records the
  commit, and its network is not kept.

## 5. The tasks (PROPOSED)

### 5.1 Training rooms

Observation spec v1 needs a room graph, and only ships have one. So rung 2
needs small levels of its own:

- **A small generator of training rooms:** one to three rooms, with doors
  and a few crates, a room graph, a spawn room per team. Written through the
  level writer like every generated level, and built on the ship generator's
  second stage (the part that turns a room graph into tiles and checks it)
  as far as that fits rooms this small; the patch's note settles it.
- **A bank of a few hundred rooms per task plus a held-back set,** made from
  a list of seeds. The generator and the seed lists are in the repository;
  the rooms are outside it and are made again from the seeds.
- **Sides.** Each room is also made with the two teams' spawn rooms swapped,
  so the soldier starts from either end without any change to the
  environment.

### 5.2 Three yardsticks before any training

On each task's held-back rooms, through the bot tournament (the random
player through the farm, as `tools/rl_smoke.py` already plays it):

| player | why |
|---|---|
| random legal choices | the floor: what no learning looks like |
| `bots_door` | what the control panel allows a competent player |
| `bots_v1` | the first-generation bot, the duel's opponent, in the soldier's seat |

Without these a pass mark is a guess, which is the lesson of 1j.

### 5.3 Shooting range

- One rifleman (the learner) and one to four standing marines of another
  team, driven by `idle`. No zombies. 30 to 60 s.
- The match's own rule ends it: the last marine team standing wins. The
  soldier is paid for the health it takes off the targets, for the win, and
  loses 3 if the clock runs out.
- From scratch, without memory.
- **Mark (the lead's guess, to be reset once 5.2 is measured):** on
  held-back rooms it clears the range in time at least nine times in ten.
- What it proves: the trainer, the pointer at a list entry, the trigger, the
  turning aid. Nothing about tactics.

### 5.4 Duel

- One rifleman against one `bots_v1` rifleman, rooms of one to three rooms.
  No zombies. 60 to 90 s.
- With the memory; a second run without it (design 8.3).
- **Gate:** the roadmap's (section 0). Beside the win rate, the share of
  matches that reach the clock is printed: a soldier that hides has not won.

## 6. The experiment: from the copy against from scratch (PROPOSED)

The roadmap's question: does copying shorten the path, or leave habits to
unlearn?

| arm | starts from | note |
|---|---|---|
| A | nothing | |
| B | the copied network | the value network is new, so for the first rounds only the value network learns and the soldier's network is held still. Without this, the first grades are noise and they wreck the copy (a known effect) |
| C (reserved) | the copied network, with a fading pull back toward the copy | only if B forgets the copy at once |

- **At least three repeats of each arm,** with different seeds. A single run
  of this method proves nothing: two runs of the same settings can differ
  more than two arms do. Three PCs make this cheap.
- **Read off:** decisions needed to reach half the duels won; the win rate
  at a fixed budget; and, by eye, what B does that A does not. Known habits
  to look for: B walks and never sprints, and never reloads before its
  magazine is empty, because its teacher did not.
- **Either answer is a result.** If B is no faster, rung 3 starts from
  scratch and the copying step has still paid for itself as the network's
  first test.

## 7. Measuring and watching (PROPOSED)

### 7.1 The curves

Written as one plain line of numbers per round, plotted by a small script
(no new dependency). Lesson 0's section 11 says how to read them: reward per
match and per group; win rate against the fixed opponent; how much of the
return each of the five value outputs explains; entropy per head; how far
each update moved the policy; plus the count of masked choices sent (must
stay 0), the seconds spent collecting and learning, and the share of matches
that reach the clock.

### 7.2 A learned team in a played match

- **A match controller `net`, built like the door bots** (1j): where
  `bots_door` puts the bots' brain behind the control panel, `net` puts a
  saved network there. Same observation door, same masks, same decoder. It
  lives in `agents/` and is registered by the tool that needs it; it joins
  the list of door drivers that get the bots' commander.
- **Network numbers stay outside the game** (roadmap rule 12): the
  controller hands the game five whole numbers per soldier. Two PCs may draw
  differently from the same network; the recorded match replays exactly on
  both.
- **The bot tournament gets it as a player,** so a learned team is measured
  with the same instrument and the same counts as the bots (wins, survivors,
  rounds per kill, friendly fire).
- **Watching** is a recorded tournament match played back with
  `main.py --replay`, which exists.

### 7.3 Strength during a run

Every so many rounds, the farm plays the held-back rooms with the current
network against the fixed opponent and the win rate is one more point on the
curve. At the milestones (the copy, the range's end, the duel's end, each
arm of the experiment) the tournament is run and replays are saved. A rating
on one scale across many saved networks (Elo or TrueSkill) is rung 3's: it
needs past selves to place.

## 8. Tests (PROPOSED)

Design 8.6's list stands. The "one small known task" there becomes three,
each built to fail on one of our own traps, each in a toy environment with
the real layout's shapes and solved in seconds without the game:

| toy task | fails if |
|---|---|
| point at the list entry that carries a flag; the list is shuffled at every decision | the pointer follows a place in the list and not the entry |
| the legal arguments depend on the hand verb and the aim just chosen; one head is unused on most decisions | a choice is graded that did not count, or the masks at learning differ from those at acting |
| a cue at the first decision, the answer asked for 20 decisions later; soldiers die mid-match and matches end mid-batch | the memory, the stretches, the dead rows or the endings are wrong |

Plus a grading case worked by hand (a death, a match ending, the clock), and
for the copy the 1 000-decision test of 3.3.

**No run with the game starts before all of these are green.**

## 9. Speed and cost

| what | number | status |
|---|---|---|
| the farm on `ship_demo` (the medium ship), spec v1, both teams learning | 362 match-ticks a second, so about **720 soldier decisions a second** | measured (2026-11-13, the 1i reading) |
| the same on the small ship | about 2 300 | estimate (the old small / medium ratio) |
| the roadmap's planning figures | 3 000 small, 1 000 medium | set on 2026-11-11, before observation spec v1 and before the gameplay pack made a tick heavier |
| the farm in a training room | unknown; the rooms are meant to be some twenty times smaller than the medium ship, with one or two soldiers | the first measurement of step 4 |
| learning time per batch | unknown; the design's critique expects it near the collecting time | measured on the toy tasks and the first range run |
| the copy | about 20 minutes per training run on one graphics card | estimate |

**A finding for rung 3, not for rung 2:** at 720 decisions a second, the
roadmap's "100 million to a billion decisions" for a fireteam on the medium
ship is **1.6 to 16 days** on this PC, against the roadmap's "half a day to
four days". Rung 3 will want the small ship first, and the graphics-card
backend (rung 5, #29) matters more than the roadmap says. No decision is
asked for now; the roadmap's cost table gets a dated note.

## 10. The patches

One implementation agent at a time, each patch with its own note written at
its boundary and its own gate, as in rung 1.

| patch | what | touches the game? | needs the farm? |
|---|---|---|---|
| 2a | the network, the choosing module, their tests (sections 2, 8) | no | no |
| 2b | the trainer's core: the stored decision, grading, the update, the two parts and their hand-over, saving, the curves, the toy tasks (4, 8). The lesson document "The trainer, line by line" is written from this code | no | no |
| 2c | the records unpacked; the copying trainer and its measures (3) | no (built on the existing records) | no |
| 2d | the `net` controller and the tournament's new player (7.2); the copied team measured and watched | yes: one registry line, the tournament | the tournament |
| 2e | the training rooms, their banks, the three yardsticks (5.1, 5.2) | a new generator tool | the tournament |
| 2f | the records made again (3.1); the final copy | no | the tournament |
| 2g | the range (5.3) | no | yes |
| 2h | the memory in real runs; the duel (5.4) | no | yes |
| 2i | the experiment (6); the gate; Erik watches | no | yes, on three PCs |

2a, 2b and 2c touch no game file and use no farm, so they do not collide
with P6's files or its timings. Whether they are built while P6 runs is
Erik's call (one implementation agent at a time across sessions). 2d to 2i
wait for P6.

## 11. Not in rung 2

Each has a rung or a trigger.

- Training against itself, pools of past selves, a rating scale: rung 3.
- The team reward blend, friendly fire, the horde: rung 3.
- Collecting and learning at the same time, and one run over several PCs:
  when a direct link is measured (4.6).
- Other optimisers, a classification-style value loss, a wider or deeper
  network: after the duel learns; half and double the width first (design
  8.3).
- **Observation spec v2.** The unit design (`docs/unit_design_2026-11-14.md`
  3 and 9) asks whether a soldier is shown its own and its team-mates' hooks,
  and for fixed slot blocks so that a new status or grenade row no longer
  changes a layout. Rung 2 trains on spec v1 with every hook neutral, where
  neither matters. Both are designed when the first of these is about to
  reach a training run: a hook that differs from neutral, a new status kind,
  a new grenade row. A new observation spec means a new generation of
  networks, which costs nothing while the networks are throwaway.

## 12. Risks

| risk | what is done |
|---|---|
| a silent fault in a self-written trainer that looks like "learning is hard" | one choosing module used three ways (2); the toy tasks and the hand-worked case before any game run (8); the copy as the network's first test |
| the copy's habits (walk only, no reload, no plant) | the floor of 3.2; the experiment reads them off (6) |
| the copy is wrecked by the first rounds of reward learning | the value network learns alone first (6, arm B); arm C in reserve |
| memory on the 16 GB desktops | the batch sized by 4.2; the copy's data read from disk in place |
| learning takes as long as collecting | measured in 2b and 2g before any size is fixed |
| one run's result taken for a fact | three repeats of anything that is compared (6) |

## 13. What Erik decides

- Whether 2a to 2c may be built while P6 runs (section 10).
- The tags for runs whose networks are kept (4.7).
- The second half of the gate, by eye (0).

Everything else here is the lead's, and his to veto.

## 14. Systems

**Existing systems this design must use** (rows of the project's
canonical-systems table):

- **Batched environment:** the only way a training run touches the game;
  `team_rows` for a team's observation, the masks from `obs`, the full view
  from `info`, the five reward groups, `rows_ended`, the two row flags; the
  run file and its manifest.
- **Observation spec, Full view, Action spec, Reward profile:** read as
  served; shapes from the layout; `heads_used` and `arg_mask` decide which
  heads counted, never a second rule.
- **Bots through the door:** the records and `read_record`; the capture
  construction for the reserved remedy of 3.3.
- **MatchSession + the `policy` controller, Match controller registry,
  Order:** the `net` controller is a registry row that drives its team only
  through the decoder, and gets the commander as a door driver.
- **Bot tournament, Headless match runner, Match recording / replay:** every
  measurement of a learned team and every replay Erik watches.
- **Training snapshot:** tags, the clean-tree checks, the run manifest.
- **Level data layer, Level generators, Room graph, Spawn rooms, Match
  config:** the training rooms are written through the level writer with a
  room graph, and their matches through the match writer.
- **ML stack:** PyTorch as installed by `ml_stack.py`; no package by hand.

**New systems**, each with its draft rule (written into the project
instructions only when the code exists):

| system | patch | draft rule |
|---|---|---|
| Network spec | 2a | A network is a named spec in `agents/`; it meets the game only through an observation spec and an action spec. A saved network names all three and is refused by any other. |
| Choosing | 2a | The chances of the choices under the masks, the draw, the probability of a given choice and which heads counted are ONE module, used when acting, when learning from reward and when copying. |
| Stored decision and batch | 2b | What is kept per decision is one layout; a batch carries the weights' version, both files' hashes and the build id. |
| Trainer | 2b | In `agents/`, outside the game's source tree; touches the game only through `BatchedEnv`. A collector and a learner with one hand-over each way; the learner refuses a batch from other weights. |
| Saved networks | 2b | One store, outside the checkout; every saved network says what it was started from. |
| Copying | 2c | Learning from records uses the trainer's own choosing module and the records' own masks; the records are data, re-made when the bots or the observation change. |
| Net controller | 2d | A learned team in a played match is a registered controller built like the door bots: the observation door, the masks, the decoder; five whole numbers reach the game. |
| Training rooms | 2e | Small levels for single skills, generated through the level writer with a room graph; the generator and seed lists are tracked, the rooms are not. |
