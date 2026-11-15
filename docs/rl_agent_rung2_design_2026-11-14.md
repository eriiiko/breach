# RL agent, rung 2 (first light): the trainer, the copying step, the first tasks

2026-11-14, **version 2 of 2026-11-15, after the critique.** The lead's (the
trainer and the network are the lead's to decide, skill `game-ai-lead`
section 1), written for Erik to veto. Documents only: written while the
residency arc's P6 (#29) is changing the unit code, so no number in it comes
from a new run.

**Read by Erik, 2026-11-15 (version 1):** "i read through rung2 document and
im very impressed"; no veto, one question (what "earns it" means, section
0), one decision (all building waits for P6, section 10) and one idea parked
as an experiment (a feet choice that names a spot, design 13.4).

**The critique, 2026-11-15:** four critics, one lens each, verdicts in
`docs/rl_agent_rung2_design_2026-11-14_critique/`. All four: survives with
fixes; 1 blocker, 33 major and 50 minor findings in all. Section 15 lists
every blocker and major with its fix. The method did not change. What
changed: how the goal is measured (sections 0, 5, 6, 7), the copy's floor
(3.2), the tests (8), the cost (9) and the patch list (10). **Section 13
lists what changed in what Erik read, and what is his to decide.**

It continues `docs/rl_agent_design_2026-11-11.md` (cited as "design x.y"),
whose sections 6, 7, 8 and 13 it builds on and does not repeat, and
`docs/rl_agent_roadmap_2026-11-11.md` (rung 2). The method is explained in
`docs/rl_lesson_0_learning_from_reward_2026-11-11.md` ("lesson 0"); the
sources are in `docs/research/rl_agent_survey_2026-11-11_raw/05_trainer_state_of_the_art.md`
("digest 05"). A number is marked **measured** (with where) or **estimate**.

## 0. Rung 2 on one page

**The goal** (roadmap): a learning curve that rises, and a learned soldier
Erik can watch.

**The steps**, in order:

| step | what | game running? | what Erik sees |
|---|---|---|---|
| 1. The network | network v1 (design 8) as code, with its tests | no | a printed table of its parts and weights |
| 2. The trainer's core | learning from reward, proven on small made-up tasks built for our own failure modes | no | their curves |
| 3. A first real run | the range on one to three hand-made rooms, a few hours | yes | does the curve rise; the true speed and memory |
| 4. Copying the bots | the network learns to predict the door bots' recorded choices | no | the first network-driven fireteam, as a replay in the game window |
| 5. Shooting range | one soldier, standing targets, generated rooms, from scratch, to its mark | yes | a curve, a replay |
| 6. Duel | one soldier against one first-generation bot | yes | a curve, replays |
| 7. The experiment | the duel started from the copied network against the duel from scratch | yes | one table: did copying help or hurt |

Step 3 is new (critique, lens 4): in version 1 the trainer first met the
real game six patches in, and every unknown about the machines hid until
then.

**The gate.** The roadmap's sentence stands: on rooms it has never seen, the
soldier wins more than half of its decided duels against the
first-generation bot (`bots_v1`), and the replay shows it aiming, firing and
moving on purpose. Version 1 gave it no numbers. They are (5.4, 7.4):

- **Which rooms:** the GATE list of 5.1, 100 rooms played from both ends,
  never used for anything else.
- **Which network:** one trained by reward, named in writing before the
  reading, read beside the network it started from. At least one soldier
  trained from nothing is read too, so that the trainer is proven without
  the copy.
- **"More than half":** the win rate of the decided duels, with its
  uncertainty worked out by resampling whole rooms; the lower end of the
  90 % range must be above one half. In practice that asks for about 57 %
  measured.
- **Hiding does not pass:** the share of duels that run out the clock may be
  at most what the bots behind the control panel (`bots_door`) show in the
  same rooms, plus 10 points.
- **Erik's eye:** eight replays picked by a rule written beforehand, watched
  before he is told the win rate (7.4).
- **Both halves are needed.** If they disagree, it goes to a design session.

**What waits for P6:** everything (Erik, 2026-11-15; section 10).

**Rung 2's networks are throwaway.** They exist to prove the machinery and
to answer the copying question. Nothing trained here is carried to rung 3
unless it earns it.

**What "earns it" means** (Erik asked, 2026-11-15; made exact after the
critique). Rung 3 needs a network to start from. The candidates are:
nothing, the copied network, and the best duel network of rung 2. The choice
is made in ONE place: at the start of rung 3, on rung 3's own task.

- The first 10 million decisions of rung 3's training are run from each
  candidate, three times each.
- The measure is the win rate against the layered bots on rung 3's
  validation ships, read at 5 and at 10 million decisions.
- A candidate is "ahead" of another only if all three of its runs are above
  all three of the other's at 10 million. This is strict on purpose: "not
  ahead" then means "not shown better", not "worse".
- The start is the candidate that is ahead of both others. If none is, the
  simplest wins: nothing, then the copy, then the duel network.
- The duel network must also come from a tagged run and fit the specs rung 3
  uses.

By default, then, the duel network is not used: a soldier that has only ever
fought alone against one bot has learned nothing about team-mates, fire
lanes or a team reward. If it is thrown away, nothing is lost, because
nothing was built on it. The cost of this comparison is nine runs, 12 to 42
hours of all three PCs (estimate, lens 4); rung 3's design confirms or
shortens it.

## 1. Where the code lives and what it may touch

The roadmap's rule: the trainer is outside the game's source tree and meets
the game only through the batched environment.

- **A top-level folder `agents/` in this repository**, beside `src/`, not
  inside it. One repository, so that one tag fixes the game and the trainer
  of a training run together.
- **The run check must be taught about it.** Today the Training snapshot
  check looks for stray files only under the game's folders and checks only
  where the game's modules were loaded from. Patch 2d adds `agents/` and
  `tools/` to its list and makes it check that the `agents` code in use is
  this checkout's (an edit of `src/rl_env.py`).
- **Nothing in `src/`, `renderer/`, `ui/` or `main.py` imports it.** A test
  scans for that. The game never depends on PyTorch.
- **What each part reaches the game through:**

  | part | through |
  |---|---|
  | the trainer's learning loop | `rl_env.BatchedEnv` only |
  | the copy | the records (`control_bots_door.read_record`, their own `meta` for sizes); `action_spec_v1.heads_used` / `arg_mask` |
  | a learned team in a played match | the shared door driver of 7.2, the controller registry, the tournament |
  | the training rooms | the level writer (`level_lib`) and the match writer |

- **PyTorch from the shared ML stack** (`ml-stack` skill; 2.15 on the
  desktops). No training framework: digest 05 found that every framework
  wants to own the loop that `BatchedEnv` already is, and that the two the
  July survey named no longer fit.
- **Settings in two files, both outside the repository:** the run file (the
  environment's, as today) and a **training file** beside it (network spec,
  the numbers of section 4, the task, the reward group weights, how often to
  save and measure).
- **A training manifest of its own.** The environment's run manifest has a
  fixed set of fields and a resume refuses any new one. So the trainer
  writes `training_manifest.json` beside it: the training file's sha256, the
  room generator's commit, the seed lists' sha256 and a hash over the level
  folders the run file names. A resume refuses a change in any of them.

## 2. The network as code

Design section 8 is the specification. What this section adds is how it is
built so that the usual faults cannot hide.

- **One implementation of "choosing".** A single module turns the network's
  raw outputs and the masks into the chances of each option, draws a choice,
  and computes the probability of a given choice. It is used when a soldier
  acts, when the trainer learns from reward, and when the network copies the
  bots. A fault in the masks or in "which choices counted" then shows up in
  copying, the cheapest of the three. (Copying never draws, so it does not
  test acting against learning; the live check of 4.8 does.)
- **Which choices counted is the game's own function** (`heads_used` with
  `arg_mask`), never a second rule. This replaces the sentence in design 8.2
  about "a soldier that could not act": a stunned soldier's choices are
  still graded. The cost is a little noise, not a bias.
- **The inputs are the stored arrays, as served.** The network acts on
  exactly the arrays that are kept for learning, so the probability of a
  choice can be recomputed later and compared.
- **One arithmetic for acting and learning:** the same number precision in
  both, layer normalisation (never a batch statistic), no dropout.
- **Sizes depend on the team size S** (the largest team in the run's
  levels). The list has S + 33 entries of 67 numbers and the aim choice
  18 + (S + 33) options:

  | task | S | list entries | aim options |
  |---|---|---|---|
  | ships, as the bots' records | 6 | 39 | 57 |
  | range, up to three targets | 3 | 36 | 54 |
  | duel | 1 | 34 | 52 |

  The rest is fixed: the vector 97 numbers; the near window 13 planes, the
  far window 15; feet 49, trigger 2, hand 13, argument 5; the value
  network's list 64 entries of 85.
- **Network v1 does not care about S.** Every list entry is read by the same
  weights and the aim points at entries. So one network plays the duel and
  the fireteam. "Fits the specs" means the same spec names, the same number
  of grenade kinds and planes; not the same list length. Data of mixed S is
  padded to the longest list with the padding masked off.
- **The weight count is recounted and printed by a test,** with the peak
  graphics-card memory of one learning step at 512 and 2 048 decisions, both
  networks, at 32-bit and 16-bit precision, on the machine at hand.
- **A reduced-width setting** of the same code exists for the tests, which
  must run on a processor in seconds.
- **Memory can be switched off** in the spec, for the range and for the
  first copy.
- **A network is a named spec** ("v1"); a saved network names the network
  spec, the observation spec and the action spec it was made for, and
  loading it into anything else is refused.

## 3. Copying the bots (the order AGREED, design 13.3)

In the literature this is called behaviour cloning: ordinary supervised
learning, where the "label" is what the teacher did.

### 3.1 The data

- **The records** of patch 1j: per decision of a door-bot soldier, the
  observation it was served, the masks, the five integers that were sent,
  the argument's mask, the rewards; one file per match and team, decisions
  in order. About 1.1 million decisions of living soldiers in the 600
  recorded matches (estimate from the 663 000 measured in 360 of them).
- **They are recorded again after P6,** because the poison law changes how
  matches with poison gas play. Changes at the same time:
  - more ships: about ten generated freighters in place of two, two of them
    held back for measuring;
  - duel rooms, with one door bot against one first-generation bot, so that
    the copy has also seen the duel. **Only rooms of the TRAINING list of
    5.1.** The validation and gate rooms are never recorded: otherwise the
    copy would carry them into the exam;
  - how many matches per ship and per room, and so the hours and the disk,
    is written in the patch's note before the run. The same 600 ship records
    are about 67 minutes on 12 workers (800 matches at about 60 s; measured
    walls of 1j), with the workers held to one thread each.
- **Read packed, not unpacked.** Version 1 unpacked everything to 45 GB of
  files read in place. The critique showed that reading single decisions
  from such files is slower than the graphics card (40 to 100 minutes a
  training run in place of 20). The records pack about ten times. So
  loader processes unpack whole matches into a shuffle buffer of some 30
  matches (about 2 GB) and deal decisions from there: about a minute a pass
  over all the data (estimate), no 45 GB on any disk, nothing to redo when
  the records are made again. The patch measures decisions read per second
  before its first long run.
- **Split by level and seed,** never by file: one match gives several files
  (both sides, with and without the horde), and they must fall on the same
  side of the split.

### 3.2 What is learned, and how

- **The loss:** make the bot's recorded choice more likely, head by head,
  under the masks, counting only the heads that counted. This is the same
  probability the trainer uses in section 4.
- **No floor on the hand.** Version 1 trained the copy toward "the bot's
  choice with 98 %, the other legal options sharing 2 %", so that sprinting,
  reloading by choice and planting would not be ruled out. The critique
  worked out what that does: about 0.3 % on each other hand choice at every
  decision, six times a second, is a C4 at the soldier's own feet in about
  two duels of three, plus grenades and reloads at random. So:
  - the hand, the argument, the aim and the trigger are copied as they are;
  - the feet keep the 2 % floor, where a random step harms nothing and the
    three speeds stay alive;
  - reloading by choice and planting are then things the copy does not do.
    That is the "habit" the experiment measures (section 6). If learning
    from reward needs help finding them, the help is a small chance of a
    random legal choice built into the choosing module itself (so acting and
    learning agree on the probability), off by default, and a number that is
    swept.
- **Without memory first, then with.** The bots' choices depend on things
  the observation does not show (the path they planned, the position they
  committed to). A memory can recover part of that.
- **The memory copy is trained the way it will play:** on whole matches of
  one soldier in order from the first decision (about 390 decisions), cut
  into the same stretches as section 4.5 with the memory carried from one
  stretch to the next. Never on windows started from an empty memory in the
  middle of a match: a network trained so has never seen a memory older
  than its window and either drifts or learns to ignore it.

### 3.3 What is measured (each can fail)

| question | measure | reading |
|---|---|---|
| Is the data path sound? | the network trained on 1 000 decisions spread across matches | at least 99 % agreement on every head that counted. It tests the data path and the network's capacity, not the masks |
| Does it predict the bots? | agreement with the bot's choice on held-back matches, per head and per option | against two baselines: "always the most common option" and **"repeat the previous choice"**. The second matters because the soldier is shown its own previous choices, and the bots change their mind rarely: a network that only repeats itself scores well |
| Does it predict the changes? | the same, only on decisions where the bot's choice differs from its previous one | a copycat scores near zero here |
| Does it lean on its own past? | one run with the previous-choice inputs blanked at the network's input | whether those inputs stay for the copy is then measured |
| How close are the misses? | mean angle error of feet and aim; for the pointer, against "aim at the nearest enemy" | one direction off is not the opposite way |
| All heads together? | every counted head right, with the network's own aim and hand chosen first | per-head agreement given the bot's aim overstates the later heads |
| Lossy labels? | agreement split by the records' loss marks | a label the bot decided a few ticks after the observation is a weaker label |
| How much does the observation not explain? | the same on the training matches, at the end of training and at double width | what stays wrong at both is the bots' hidden state |
| Does the memory help? | memory on and off, each held-back match run from its start | three runs each, or it is labelled "indicative" and decides nothing |
| **Does it play?** | the copied team in the bot tournament: against `bots_door` (its teacher), `bots`, `bots_v1`, the horde alone; and **in the duel rooms** against `bots_v1` | reported with plants, throws and self-inflicted damage per match. At best it plays like `bots_door` |

**The bar before the copy may start arm B of the experiment:** in the duel's
validation rooms against `bots_v1`, the copied soldier wins at least half as
many of its decided duels as `bots_door` does there, and more than the
random player by more than the reading's uncertainty. Below the bar, the
remedy below is run first, or the experiment is reported as "the copy was
too weak to answer".

**If the copy plays badly although it predicts well,** there is a known
remedy, and our door bots make it cheap: let the network drive, let the bots
run beside it on their capture proxy (the 1j construction) and record what
they would have done in the situations the network got itself into, then
train on those too. (The literature's name is DAgger.) One caveat found by
the critique: the bot beside the network keeps its own plan as if its
choices had been played, so its labels refer to a plan the soldier is not
following. It is reserved, not built, and that caveat is settled first.

## 4. Learning from reward: the trainer

PPO in the arrangement where every soldier shares one network and a
training-only value network sees the whole match (roadmap rule 8; lesson 0).
Synchronous: the farm already steps all its matches together.

### 4.1 One round

1. **Collect.** The farm plays T decisions in each of its N matches. At each
   step the soldier's network answers for every learning soldier of every
   match in one pass on the graphics card. The value network is not run
   here.
2. **Grade.** The value network estimates, for every stored decision, the
   reward still to come; from that and the rewards that did come, every
   choice gets its grade (lesson 0, sections 5 and 6).
3. **Learn.** A few passes over the batch, under the clip (lesson 0, 7).
4. **Throw the batch away, raise the weights' version by one, repeat.**

### 4.2 The stored decision, and whose decisions are stored

**Only the learning team's.** The environment observes, masks and pays every
soldier of every team, the opponent's and the standing targets' included.
The trainer stores a decision only for a soldier of a team that the run file
puts on the `policy` controller, read through `BatchedEnv.team_rows`, and
only while that row is in the match.

| row | stored | learns |
|---|---|---|
| a living soldier of a learning team | yes | the soldier's network and the value network |
| a dead soldier of a learning team | yes | the value network only (what the death was worth, design 6.3) |
| any other team's soldier; an empty roster place | never | nothing |

**What is sent for rows the trainer does not drive:** the always-legal
default (stand, aim where I walk, no trigger, hand nothing). The count of
masked choices sent, which must stay 0, is read on the driven living rows
only.

**What is kept per stored decision** (design 8.6): the observation as
served; the value network's full view; the five choices; the masks and the
argument's mask as they were; which heads counted; the probability of each
choice; the five reward numbers; the two row flags; and, once per stored
stretch, the memory vector where the stretch began. The environment's arrays
are its own and are overwritten at the next step, so everything kept is
copied.

- **About 63 KB per decision** (40 KB observation, 22.7 KB full view;
  recomputed by two critics). It is mostly padding in a duel. A batch of
  8 192 decisions is 0.5 GB.
- **The computer's memory does not set the batch size; the graphics card
  does** (4.4). Version 1 had this the wrong way round.

### 4.3 Grading

- **Five estimates, one per reward group** (design 7.2). Grades are computed
  per group and added with the training file's weights. Rung 2's weights:
  outcome 1, own 1, team 0, friendly fire 0, order 0.
- **One reward table for both tasks:** the run file names the "fireteam"
  profile with its terms as they are. So the soldier is paid for health
  taken off the enemy, +0.5 for a kill, charged for its own lost health and
  -0.5 for its death, +10 for the win, -10 for elimination, -3 when the
  clock runs out. (Design 6.9's range paid no win; one table for all of rung
  2 is chosen here.) No new environment code.
- **The learning team holds the bots' commander's orders** in every rung-2
  task (the builder adds the commander beside every `policy` team). So the
  soldier is shown where the objective is. The duel's opponent holds none.
- **Endings.** A win, an elimination and the clock are all real endings
  (design 6.5; the soldier is shown the time left), so nothing is still to
  come after them. The trainer refuses a run on the old "counters" profile,
  where the clock is not an ending.
- **The one cut is the end of a batch in the middle of a match.** There the
  value network's estimate of the batch's last full view stands in for the
  rest; it is 0 if that match ended on that very step (the view returned is
  then the next match's).
- **A match that ends inside a step:** its last rewards belong to the ended
  match; the next stored decision is the new match's first, with the memory
  set to zero. With three or more teams a row can end while its match goes
  on (`rows_ended`); its story ends there.
- **The value targets are rescaled** per output, in the form that leaves the
  network's real-unit outputs unchanged when the scale moves (PopArt). The
  order: grade with the scale as it was while collecting; then update the
  scale with this batch's targets; then learn. The scale's first move, on a
  fresh network, leaves the outputs to move with it (amended 2026-11-15 from
  patch 2b's "scale" task: there is nothing learnt to keep, and keeping it
  cost thousands of steps; the reason is in `agents/value_scale.py`). A floor
  under the spread, so
  that a group that is always zero (friendly fire and order, in rung 2) does
  not divide by nothing. All five outputs learn in rung 2, the weight-0 ones
  too: rung 3 needs them, and they cost little.
- **Grades are brought to a common size** (subtract the mean, divide by the
  spread) after the five are added, over the living learning rows only, per
  piece.

### 4.4 The update: starting numbers

Every one is the start of a sweep, with where it comes from.

| number | start | why |
|---|---|---|
| look-ahead (the discount) | range 0.99 (about 17 s); **duel 0.995 (about half a minute)** | design 6.5 says half a minute. At 0.99 the clock's penalty is invisible until the last 17 s of a duel, which makes hiding nearly free (lens 3). 0.99 against 0.995 is the duel's first sweep |
| blend of short and long grading (lambda) | 0.95 | the usual value; 0.9 is the main alternative |
| clip | 0.2, on the probability of the whole decision (all counted heads together) | the MAPPO paper prefers smaller; 0.1 is a sweep |
| passes over a batch | 4, stopped early when the policy has moved more than 0.02 (the usual distance measure between the old and new chances) | digest 05's open question: the squad benchmarks use 10 to 15, OpenAI Five found 8 ruinous. First sweep: 2, 4, 8 |
| a round | 64 matches x 128 decisions = 8 192 decisions | four matches per farm worker; 128 is a whole number of memory stretches |
| piece | 2 048 decisions, worked through in smaller lumps of 512 whose results are added up before the step | the lumps make the update the same on every card (below) |
| step size | 0.0003, Adam, falling to zero over the run | the common default |
| push toward variety (entropy) | 0.01 per counted head, each head's variety measured as a share of the most it could be | so the trigger (2 options) and the feet (49) are pushed alike |
| limit on one update's size | 0.5, separately for the two networks, each with its own optimiser | a spike in the value network must not shrink the soldier's step |

**The graphics card sets the piece.** One decision through both networks
keeps about 2.35 MB of intermediate results while learning (estimate, lens
4, within 50 %). A piece of 2 048 at once is about 6 GB: it fits the work
PC's card (12 GB) and not the home PC's (8 GB) or the laptop's (6 GB). Lumps
of 512 fit all three and give the same update. On Windows a card that runs
out does not fail; it silently spills into the computer's memory and runs
several times slower. That spill is switched off for the training Python on
every PC, so running out is an error.

### 4.5 Memory

The part that most often hides a fault (lesson 0, section 10), so it comes
second: the range is trained without it.

- A stored batch is cut into **stretches of 16 decisions** (2.7 s) per
  soldier. Learning replays each stretch in order from the memory vector
  stored at its start, and takes whole stretches, never single decisions.
- **What the memory can LEARN to hold is one stretch long.** The stored
  vector carries information across stretches when the soldier acts, but
  learning cannot reach back past the start of a stretch. So with 16, the
  soldier can learn to remember for 2.7 s. Version 1 did not say this. The
  stretch length is a number to sweep (16, 32, 64) before rung 8, where
  "where did the enemy go" may need longer.
- The memory is zero at a match's first decision, also when that falls
  inside a stretch.
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
- **In rung 2 both parts run in one process on one PC.** Learning is
  expected to take 0.1 to 0.7 of the collecting time (estimate, lens 4), so
  the farm is the bottleneck and the synchronous form loses little.
- **What a spread-out run would have to move, measured in advance:** the
  observations, 0.5 GB per round before packing, about ten times less
  packed. At least the laptop reaches the work desktop only through
  Tailscale's relay. So one run over three PCs needs a direct link, and the
  link and the packed size are measured before anything is built for it.
- **Until then, as Erik said:** three PCs run three experiments and share
  saved networks.
- **Using several PCs for ONE run (Erik, 2026-11-15: "we can gain a lot by
  running on more than one computer - we should really try to exploit
  that").** The hand-over above already allows the simple form: every PC
  collects with the same weights, sends its packed batch to the one PC that
  learns, and gets the new weights back. Nothing is learned from old play,
  so no correction is needed; the price is waiting for the slowest PC and
  for the link. A PC's share of a round is 20 to 50 MB packed (estimate) and
  the weights back are 6 MB. On a home network that is nothing; over the
  internet it depends on the upload speed, and through Tailscale's relay it
  does not work. The gain is at most the sum of the farms, about 2.5 times
  with all three. Where it pays: rung 3, whose runs are days. In rung 2 the
  work is many separate runs (repeats, sweeps), which three PCs already
  share with no new code. So: the links are measured early (a file of 50 MB
  sent each way between each pair of PCs), and the spread-out run is built
  for rung 3 if they carry it.
- **A PC given to training.** A running farm takes every thread of its PC,
  for hours to days. The lead's choice is the work desktop: the same
  processor as the home desktop, the larger card memory (12 GB, the only
  card that takes a full piece at once), and the home desktop stays free for
  building, the game's own checks and watching replays. Against it: it also
  serves the shared memory (egregore) and the art pipeline. Erik's to
  decide.
- **DECIDED (Erik, 2026-11-15):** the work desktop is given to training;
  egregore moves off it first, to the home desktop or the laptop, decided
  before training starts ("i suspect home desktop since it is also always
  on"); and the home desktop is set up for training as well ("we will train
  a lot in the months to come, i say we set up both"). The plan
  (`docs/rl_agent_rung2_plan_2026-11-15.md`, section 3) says what "set up"
  means.

### 4.7 Saving, and which code a run used

- **A saved network** holds: both networks' weights, the learner's state,
  the value targets' scale, the counters (decisions, batches, weights'
  version), the three spec names, both files' hashes, the game's build id,
  the training manifest's hashes, and what it was started from (nothing; a
  copy of which records; or which saved network). About 32 MB. The soldier's
  weights alone, as an opponent needs them, are about 6 MB.
- **What is kept of a long run:** the last, the best on the validation
  rooms, and every Nth.
- **Outside the checkout,** in the run's own folder.
- **A run can be stopped and resumed.** Not bit for bit: the graphics card's
  arithmetic is not exactly repeatable. The matches are (every one replays).
- **Tags.** A run whose network is kept (shown to Erik, used as an opponent,
  or counted for the gate) runs from an annotated tag in its own checkout;
  the tags are Erik's to place. A short exploratory run may run at any clean
  commit: the manifest records the commit, and its network is not kept.

### 4.8 A check that guards every real run

The tests of section 8 run on made-up tasks. One check runs in every round
of every real run, and stops the run when it fails:

- on the first piece of the first pass, the probability of every stored
  choice, worked out again, equals the stored one (to 0.0001), and the
  measured movement of the policy is 0;
- with memory on, the memory worked out again at each stretch's end equals
  the one the soldier had.

This one check catches a kept reference to an array the environment has
since overwritten, masks that differ between acting and learning, a
mismatch of number formats, and any difference between acting and learning
in the choosing module.

## 5. The tasks

### 5.1 Training rooms and the three room lists

Observation spec v1 needs a room graph. So rung 2 needs small levels of its
own.

- **A small generator, `tools/gen_training_rooms.py`,** writing straight
  through the level writer with a room graph, as the two-room test level of
  patch 1j does. Not built on the ship generator: that one insists on a
  hull, mirror symmetry and ship rules that a single room does not meet
  (critique). Reused from it: the airtightness check and the check that the
  two spawn rooms are joined by a walkable path.
- **Three shapes, in equal shares:** one open space; two rooms joined by a
  door; three rooms. Each team's spawn room is a room of the graph, so "one
  open space" is two graph rooms joined by an open edge. A soldier always
  starts on the same tile of its spawn room; variety comes from the rooms.
- **Crates are cover** (the game's cover objects), not the ship generator's
  crates, which are fuel and burn.
- **The match file says `sight = "all_round"`,** no zombies, and one fixed
  clock per task. Without the sight line a match defaults to the cone, and
  the farm, the tournament and the replays would disagree.
- **Sides.** Each room is also made with the two teams' spawn rooms swapped.
- **Three lists of seeds per task, written into the repository before any
  training run:**

  | list | rooms | used for |
  |---|---|---|
  | TRAINING | 300 | training; the copy's duel records |
  | VALIDATION | 50 (100 with sides) | the curve during a run, choosing a saved network, stopping, every sweep, memory on or off, the experiment's readings |
  | GATE | 100 (200 with sides) | the yardsticks, and each candidate network ONCE |

  Version 1 had one held-back set for all of this. Choosing the best of
  twenty saved networks on the exam's own rooms lifts a soldier that truly
  wins half its duels to a reading of about 59 % (lens 3, the blocker).
- **A gate room that comes out with the same layout as a training room is
  replaced.** The generator is small, so the gate measures new rooms of
  familiar shapes; the same soldier's rate on its training rooms is printed
  beside, and the gap is the finding.
- The generator and the seed lists are in the repository; the rooms are
  outside it and are made again from the seeds.

### 5.2 Yardsticks before any training, and the rules that turn them into marks

On each task's GATE rooms, all through the bot tournament with the same
match file and the same counting:

| player | why |
|---|---|
| random legal choices (a registered controller on the shared door driver, 7.2) | the floor: what no learning looks like |
| `bots_door` | what the control panel allows a competent player |
| `bots_v1` against `bots_v1` | not a skill: it measures whether one end of the rooms is favoured. Team 0's share of the decided duels must be within one half plus or minus two standard errors |

**The rules, written before the yardsticks are run** (the lesson of 1j,
where a mark set by guess was ruled on afterwards):

- **The range's mark** is the smaller of 0.9 and `bots_door`'s clear rate
  minus two standard errors.
- **The duel's gate stands whatever the yardsticks say.** If `bots_door`
  wins under 0.40 of its decided duels in these rooms, a failed gate is read
  first as the control panel's reaction cost (design 13.1, item 4,
  re-opened), not as the trainer's fault.
- **The duel's clock allowance** is `bots_door`'s clock share plus 0.10.

### 5.3 Shooting range

- One rifleman (the learner) and one to three standing marines of another
  team, driven by `idle`, in equal shares. No zombies. **45 s.** The first
  real run (step 3) uses one target.
- The match's own rule ends it: the last marine team standing wins.
- From scratch, without memory.
- **Mark:** by the rule of 5.2, read on the 200 gate room-sides. A soldier
  killed by its own grenade or charge has not cleared the room; self-harm is
  printed on its own.
- **The duel starts when the range passes, or when its fail is ruled on**
  (7.5).
- What it proves: the trainer, the pointer at a list entry, the trigger, the
  turning aid. Nothing about tactics.

### 5.4 Duel

- One rifleman against one `bots_v1` rifleman. No zombies. **75 s.**
- With the memory; without it as a comparison (three runs each, or labelled
  indicative).
- **What this opponent is, one against one** (read from its code by the
  critique): it never throws (its grenade rule needs two enemies in sight);
  it walks home every time it is hurt below its limit (its retreat rule
  looks for a team-mate nearby and it has none); it stands still to shoot;
  with no enemy in sight it walks one fixed path to the soldier's spawn room
  and stands there. It never searches. So the gate proves "beats a target
  that walks a known path, shoots back from a standstill and retreats when
  hurt". That is a fair first goal. It does not prove the soldier can fight
  something that moves differently, which rung 3 needs.
- **Beside the gate, not part of it:** the same soldier against the layered
  `bots` and against `bots_door` in the same rooms.
- **Gate:** section 0, by the reading of 7.4.
- **Hiding is the trap of this task.** While the soldier is weaker than the
  bot, a draw is worth more to it than a fight, and this opponent never
  comes looking. The longer look-ahead (4.4), the clock allowance in the
  gate and the clock share on every curve are the three guards.

## 6. The experiment: from the copy against from scratch

The roadmap's question: does copying shorten the path, or leave habits to
unlearn?

| arm | starts from | note |
|---|---|---|
| A | nothing | |
| B | the copied network, once it has cleared the bar of 3.3 | the value network is new, so first only the value network learns while the frozen copy plays |
| C (reserved) | the copied network, with a fading pull back toward the copy | only by the trigger below |

**Written before any run of it:**

- **Settings frozen first.** Every number of 4.4 is chosen on the range or
  on arm A's validation curve, then fixed, and is the same for both arms
  apart from B's start-up.
- **Arm B's start-up.** The value network learns alone until it explains
  more than half of the spread of the copy's returns, or for 20 rounds at
  most. Then the soldier's step size rises from zero to its value over 10
  rounds. These decisions count in B's budget. The critique showed that the
  value-only start removes one cause of a wrecked copy and leaves others
  (a fresh optimiser at full step, the push toward variety acting hardest on
  a network that is sure of itself); the ramp is for the first, and
  "forgetting" is measured, below, so that the second is seen.
- **The budget** is in decisions, not hours: B = the decisions the first
  duel run (step 6) needed to reach half on the validation rooms, rounded
  up, at most 15 million. Every run goes to 2B and stops.
- **Three runs per arm,** one of each arm on each PC, so a difference
  between machines cancels. Run i of A and run i of B share their seeds.
- **The one outcome that decides:** the win rate of the decided duels on the
  validation rooms at B and at 2B, with the clock share beside it.
- **Beside it:** the decisions until three readings in a row are at or above
  one half (or "not reached by 2B"); and the habits, counted per arm from
  the stored decisions and not judged by eye: the share of steps at full
  speed, reloads by choice, plants, throws per match. A's counts stand
  beside B's, or "B never sprints" cannot be told from "nobody sprints in
  these rooms".
- **Forgetting, every round of arm B:** how far the policy has moved from
  the copy on the batch's own situations, and the validation win rate.
- **The reading rule.** "The copy helps" if all three B runs are above all
  three A runs at B decisions. "It hurts" if the reverse. Otherwise **"not
  separated at three runs each"**. Three against three can only see large
  differences: it separates the arms about one time in four when they differ
  by one run-to-run spread, and about six in ten at two spreads (lens 3). So
  "not separated" is not "no help".
- **What the result decides:** nothing by itself. It is carried to rung 3's
  comparison (section 0) as what to expect. The copy's own cost goes on the
  bill beside it: about an hour of recording and the copy's training.
- **Arm C's trigger:** in at least two of the three B runs, the validation
  win rate falls below half of the copy's own within 20 rounds of the
  soldier's network starting to learn. **Arm C's pull:** a term added to the
  loss, outside the clip, that grows with how far the chances of the counted
  heads have moved away from the copy's on the batch's situations (measured
  from the copy's side, so no option the copy uses is dropped); weight 0.1,
  falling to zero over the first half of the run. With memory on, the copy's
  own memory is replayed along each stretch too.

## 7. Measuring and watching

### 7.1 The curves

Written as one plain line of numbers per round, plotted by a small script
(no new dependency). Lesson 0's section 11 says how to read them: reward per
match and per group; the training matches' own win rate against the fixed
opponent (free, every round); how much of the return each value output
explains (only for groups that vary); variety per head; how far each update
moved the policy; the count of masked choices sent on driven rows (must stay
0); the seconds spent collecting and learning; the clock share, split into
"the clock ran out" and "both died".

### 7.2 A learned team in a played match

- **One shared door driver.** Today the loop "observe through the
  observation door at each decision tick, fill the row flags, hand five
  integers to the decoder" exists only inside the door bots. Writing it
  again for a network would be a second copy outside the reach of 1j's
  tests. So patch 2f lifts it into one driver in `src/control_bots_door.py`
  whose subclass supplies only "choose five integers from this observation
  and these masks". The door bots become one subclass (their tests
  unchanged); `net` (a saved network) and `random` (a uniform choice among
  the legal options) are two more, living in `agents/`.
- **Registered as door drivers.** The list of door drivers that get the
  bots' commander is a fixed constant today; the registry gets a flag for
  it (`src/control_source.py`). A `net` team must get the commander, because
  it trained with its orders in view.
- **Which network** a `net` team loads is part of the tournament's job
  description, read inside each worker.
- **Where the network runs in a tournament:** on the processor, one thread
  per worker. On the card, twelve workers would each load their own copy of
  the card's software and exhaust the machine. The tournament pins its
  workers to one thread as the farm does (it does not today; 1j recorded
  the memory incident). Matches are 10 to 20 % slower with a `net` team
  (estimate).
- **Network numbers stay outside the game** (roadmap rule 12): the
  controller hands the game five whole numbers per soldier. Two PCs may draw
  differently from the same network; the recorded match replays exactly on
  both.
- **Watching** is a recorded tournament match played back with
  `main.py --replay`. Three things are added for that: the tournament
  records a replay for any match when asked (today only for door-bot
  teams); `main.py --replay` takes a `--levels-dir` (today it loads levels
  only from the repository, and the training rooms are outside it); the
  process that records must have `net` registered.
- **The roadmap's "export of a training episode as a replay" is dropped.**
  The farm cannot do it today, and the tournament path covers the need.

### 7.3 Strength during a run

- **Every round, free:** the training matches' win rate.
- **Every 50 rounds:** one pass of 64 matches on the VALIDATION rooms in a
  second, small environment (4 workers), created once. Under a tenth of the
  run's time (estimate). A separate environment is needed because one
  environment draws its levels from all the levels of its run file.
- **The soldier plays as trained,** with its choices drawn, on fixed seeds.
  The "always the most likely choice" form is printed beside.
- Each point is drawn with its uncertainty (64 matches: about 6 points).
- **Stalled** means: the validation win rate has not risen by two standard
  errors over the last five points while the variety of the counted heads
  is falling. **Solved** means: the gate's rule met on the validation rooms.
- At the milestones (the copy, the range's end, the duel's end, each arm)
  the tournament is run and replays are saved.
- A rating on one scale across many saved networks is rung 3's.

### 7.4 Reading the gate

- **Once per network.** The network is named in writing first. A second
  reading of a re-trained network is reported with the count of readings
  beside it.
- **200 duels:** the 100 gate rooms from both ends, one fixed seed each, the
  soldier playing as trained. The unit that counts is the room, not the
  duel: this opponent draws no random numbers, so repeats on one room are
  nearly the same duel.
- **The rule:** resample whole rooms; the lower end of the 90 % range of the
  decided win rate is above one half; and the clock share is within the
  allowance of 5.2. Printed beside: the rate per room shape, draws split
  into clock and both dead, the starting network's own reading on the same
  rooms.
- **Erik's eye.** Eight replays from the gate duels, picked by a rule
  written before the run (the first room-sides of the list in seed order
  that give wins, losses and draws in their measured shares), watched
  **before he is told the win rate**. His verdict is written into the rung's
  note in his words, then the number is added. For the experiment's "what
  does B do that A does not", replays are shown without saying which arm.
- **AGREED as a first try (Erik, 2026-11-15):** "Let's commit to watch those
  8 replays, and then evaluate how it was (also the watching part), what was
  useful, what to change to next evaluation." So after the first gate
  reading the procedure itself is reviewed with him, and this paragraph is
  amended.

### 7.5 When something fails

Version 1 had no rule for any fail, which is how rung 1's missed mark came
to be ruled on afterwards.

| measure | a fail means first | then |
|---|---|---|
| the made-up tasks (8), the live check (4.8) | a fault | stop; no game run |
| memorising 1 000 decisions | a fault in the data path | stop |
| the copy below its bar (3.3) | the copy is weak | the remedy of 3.3, or "too weak to answer" |
| the range below its mark | the trainer or the pointer | back to the made-up tasks; then the settings |
| the duel below the gate | in this order: the observation; the control panel's reaction cost (design 13.1 item 4); the reward's pull toward a draw (5.4); the trainer's settings | a design session, never a quiet re-run |
| the gate's two halves disagree | not resolved by re-running until one moves | a design session with the replays and the numbers |
| the experiment not separated | "not shown" | carried to rung 3 as such |

**A budget for trying:** at most ten training set-ups per task are read on
the validation rooms before the gate reading, and the count is reported with
the gate.

## 8. Tests

Design 8.6's list stands. The "one small known task" there becomes a set of
made-up tasks, each built to fail on one of our own traps.

- **They run against a stand-in environment that behaves as `BatchedEnv`
  does:** arrays that are overwritten at the next step, a match that ends
  and restarts inside one step with its last view under the `final` keys,
  those keys wiped the step after, five reward groups, rows that end. The
  trainer under test is then the one that will meet the farm.
- **They run the real network code at reduced width on the processor,** over
  a few fixed seeds, with a pass threshold. One full-width run on the card
  is a measurement for the patch's note, not a test.

| made-up task | fails if |
|---|---|
| point at the list entry that carries a flag; the list is shuffled at every decision | the pointer follows a place in the list and not the entry |
| the legal arguments depend on the hand verb and the aim just chosen; one head is unused on most decisions | a choice is graded that did not count, or the masks at learning differ from those at acting |
| the reward comes in the "own" group, with the opposite signal in a group whose weight is 0 | the weights are applied to the wrong group |
| rewards a hundred times larger than usual, beside a group that is always 0 | the rescaling of the value targets is wrong |
| a second team in the match that does not learn, whose rewards are the opposite of the learner's | the trainer stores or grades rows it does not drive |
| a start that is already the best possible; 20 rounds of learning | learning itself wears down a good start (the copy's danger) |
| a cue, and the answer asked for **12** decisions later; soldiers die mid-match and matches end mid-batch | the memory, the stretches, the dead rows or the endings are wrong |
| the same with **24** decisions between: must FAIL with stretches of 16 and pass with 32 | the memory's limit is not where 4.5 says |

Version 1's memory task asked for the answer 20 decisions later with
stretches of 16. No correct trainer can pass that (lens 2), and the easy
ways to turn it green are the wrong ones.

Plus a grading case worked by hand: an ending on the batch's last step; a
dead soldier of a two-soldier team that still receives team and outcome
rewards; five groups, two of them at weight 0; a last stretch that is not
full. And for the copy the 1 000-decision test of 3.3.

**No run with the game starts before all of these are green.**

## 9. Speed and cost

**The unit is decisions of LEARNING soldiers per second.**

| what | number | status |
|---|---|---|
| the farm on `ship_demo` (the medium ship), spec v1, both teams learning, eight learning soldiers per match | 362 match-ticks a second, so about **720 a second** | measured (2026-11-13, the 1i reading), with a random driver |
| the same on the small ship | about 2 000 | estimate (the measured small / medium ratio of 2.76) |
| the roadmap's planning figures | 3 000 small, 1 000 medium | set on 2026-11-11, before observation spec v1 and the gameplay pack |
| **the range and the duel** | **320 to 1 320 a second, most likely about 730** | estimate (lens 4, from measured costs per tick and per observed soldier) |
| learning time per round | 0.1 to 0.7 of the collecting time | estimate (lens 4) |
| the network's answer during collecting | 3 to 6 ms per step, 3 to 12 % | estimate |
| the copy | about 20 minutes per training run, if the reading keeps up (3.1) | estimate |

**The small rooms are not faster where it counts.** Version 1 expected
rooms twenty times smaller to train much faster. A training room's match
has ONE learning soldier where the ship's has eight, and much of a tick's
cost does not shrink with the room. Per learning decision the training rooms
cost about what the medium ship costs. The first real run (step 3) measures
it.

**What that does to the roadmap's times** (its decision counts are its own
estimates "from other people's tasks"):

| run | roadmap | now |
|---|---|---|
| range | minutes | 8 minutes to 8 hours |
| duel | about an hour | 1.3 to 90 hours; about 6 hours at 10 million decisions and the likely speed |
| the experiment, six runs on three PCs | "cheap" | an afternoon to 11 days of all three PCs |

**All of rung 2, machine time, no re-runs counted** (estimate, lens 4):
about one day of one PC plus an afternoon of three at the low end; about one
day of one PC plus one to two days of three most likely; about three weeks
at the high end. A first self-written trainer's re-runs multiply it by two
or three. The spread comes from two unknowns: how many decisions the duel
needs (a factor of ten, unknown until step 6) and the farm's speed in a
training room (a factor of four, known after step 3).

**Memory.** The computer's memory is not a limit on any PC (6 to 11.5 GB of
16 with the farm running). The card is, per piece (4.4).

**The three PCs are not equal.** The laptop's card is the smallest; its
processor mixes fast and slow cores, and the farm waits for its slowest
worker at every step (estimate: half to nine tenths of a desktop). It is not
yet on the shared ML stack and `rl_smoke.py` has never run there. Both are
done before the experiment.

**A running farm owns its PC.** Sixteen workers take every thread. No test
suite, build or tournament runs on a PC while its farm trains; a run
calendar says which PC is busy when.

**Paying for soldiers that do not learn.** The farm builds the observation
for every living soldier, so the range pays for its targets' (about a third
of its speed at three targets) and the duel for the bot's (about a tenth).
Patch 2i adds a run-file key so that only the learning team's rows are
built (an edit of `src/rl_env.py`).

**A finding for rung 3, not for rung 2:** the roadmap's "100 million to a
billion decisions" for a fireteam was for the SMALL ship: at about 2 000 a
second that is 14 hours to 6 days, close to the roadmap's "half a day to
four days". On the medium ship at 720 it would be 1.6 to 16 days. (Version 1
and the roadmap's note applied the medium ship's figure to the small ship's
row.) Rung 3 starts on the small ship. The roadmap's note is corrected.

## 10. The patches

One implementation agent at a time, each patch with its own note written at
its boundary and its own gate, as in rung 1.

| patch | what | touches the game? | needs the farm? |
|---|---|---|---|
| 2a | the network, the choosing module, their tests; the weight count and the card's peak memory printed (2) | no | no |
| 2b | the trainer's core: whose rows are stored, the stored decision, grading with the hand-worked case, the update, the collector against the stand-in environment, the live check, the made-up tasks without memory (4, 8) | no | no |
| 2c | **the first real run:** the range on one to three hand-made rooms written through the level writer, from scratch, no memory, 64 matches on 16 workers, one to three hours. Reports: learning decisions a second, where a step's time goes, the card's peak memory, whether the curve rises; the speed also at 128 matches | no | yes |
| 2d | the trainer's rest: saving and resuming, the hand-over headers and the refusal, the training manifest, the run check extended to `agents/` and `tools/`, the curves script, the memory path with its two made-up tasks, the "already good start" task | **yes: `src/rl_env.py` (the run check)** | no |
| (the lead) | the lesson document "The trainer, line by line", from 2b's and 2d's code; Erik reads it | | |
| 2e | copying: the packed-record loader, the copying trainer, its measures, on the existing records (3) | no | no |
| 2f | the shared door driver; the `net` and `random` players; the tournament (a network per job, a replay for any match, workers pinned); `main.py --replay --levels-dir`; the copied team measured and watched (7.2) | **yes: `src/control_bots_door.py`, `src/control_source.py`, `tools/bot_tournament.py`, `main.py`** | the tournament |
| 2g | the training-room generator, the three seed lists, the yardsticks, the marks set by their rules (5.1, 5.2) | a new tool | the tournament |
| 2h | the records made again on the training lists; the final copy; the copy's bar (3) | no | the tournament |
| 2i | the range to its mark; the run-file key for "only the learning team's rows"; the second, small environment for validation (5.3, 7.3) | **yes: `src/rl_env.py` (one key)** | yes |
| 2j | the memory in real runs; the duel (5.4) | no | yes |
| 2k | the experiment (6); the gate (7.4); Erik watches | no | yes, on three PCs |

Version 1's patch 2b was three patches in one (lens 4) and its table said
"touches the game: no" for four patches that must edit a game file (lens 1).

**DECIDED (Erik, 2026-11-15): every patch waits for P6.** "P6 is nearly
done btw. I will let you know when it finish, we might as well wait for it
to finnish i think, now that we've waited so long." So 2a starts on his word
that P6 is through, on the unit code as P6 leaves it. The time until then is
used for the critique and the handoff plan, which are documents.

## 11. Not in rung 2

Each has a rung or a trigger.

- Training against itself, pools of past selves, a rating scale: rung 3.
- The team reward blend, friendly fire, the horde: rung 3.
- Collecting and learning at the same time, and one run over several PCs:
  when a direct link is measured (4.6).
- Other optimisers, a classification-style value loss, a wider or deeper
  network: after the duel learns; half and double the width first (design
  8.3).
- **A feet choice that names a spot** in place of a direction (Erik's idea,
  2026-11-15): an experiment after the duel learns, as a new action spec
  beside v1. Design 13.4 holds the reasoning and the test.
- **Teams larger than four, up to 20 against 20** (Erik asked, 2026-11-15):
  the design is not sized by the team; design 13.5 says what is fixed and
  the road there.
- **A memory of stored moments** in place of, or beside, the running memory
  (Erik's idea, 2026-11-15): design 13.6; first tested on the made-up memory
  tasks, then at rung 8.
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
| a silent fault in a self-written trainer that looks like "learning is hard" | one choosing module (2); the made-up tasks on a stand-in that behaves like the real environment (8); the live check in every round (4.8); the first real run early (step 3) |
| the soldier learns to hide | the longer look-ahead (4.4); the clock allowance in the gate; the clock share on every curve (5.4) |
| the exam's rooms leak into training or into choosing | three lists (5.1); the copy recorded on training rooms only (3.1); one gate reading per network (7.4) |
| the copy looks good and is a copycat | the "repeat the previous choice" baseline and the change points (3.3); the bar before arm B |
| the copy misuses its kit | no floor on the hand (3.2); plants, throws and self-harm counted (3.3) |
| the copy's habits (walk only, no reload, no plant) | counted per arm (6); a small built-in chance of a random legal choice in reserve (3.2) |
| the copy is wrecked by the first rounds of reward learning | arm B's start-up; forgetting measured every round; arm C defined (6) |
| the card runs out and silently slows down | lumps of 512; the spill switched off; the peak printed in 2a (4.4) |
| the runs take weeks, not hours | budgets in decisions with a stop (6); the first real run measures the speed (step 3) |
| one run's result taken for a fact | three runs of anything compared, and "not separated" said plainly (6) |
| a fail is patched until it passes | the fail table and the budget for trying (7.5) |

## 13. What changed in what Erik read, and what he decides

**Changed by the critique, his to veto:**

1. **The goal has numbers now, and is harder to pass by luck or by hiding**
   (section 0, 7.4): 200 duels on rooms used for nothing else; "more than
   half" must hold with its uncertainty, which asks for about 57 % measured;
   running out the clock is capped.
2. **The copy no longer keeps a small chance for everything** (3.2): only
   for the feet. The 2 % he read would have made the copied soldier plant
   C4 at its own feet in about two duels of three.
3. **A first real run comes early** (step 3), and there are eleven patches
   in place of nine, four of them touching a game file.
4. **The times are longer** (section 9): the duel 1.3 to 90 hours, all of
   rung 2 between two days and three weeks of machine time.
5. **The copying experiment will often say "not separated"** with three runs
   per arm, and says so in those words (6).
6. **What the first-generation bot is as a duel opponent** is written down
   (5.4): beating it is a first goal, not proof of a fighter.

**His to decide:**

- **How his own half of the gate is run** (7.4): eight replays picked by a
  rule written beforehand, watched before he is told the win rate. DECIDED
  2026-11-15: yes, as a first try, reviewed afterwards.
- **Which PC is given to training** (4.6): DECIDED 2026-11-15, the work
  desktop, with the home desktop set up too. Still his: where egregore
  moves, before training starts.
- The tags for runs whose networks are kept (4.7).
- When P6 is through (10).

Everything else here is the lead's, and his to veto.

## 14. Systems

**Existing systems this design must use** (rows of the project's
canonical-systems table):

- **Batched environment:** the only way a training run touches the game;
  `team_rows` for a team's observation, the masks from `obs`, the full view
  from `info`, the five reward groups, `rows_ended`, the two row flags; the
  run file and its manifest. Extended by 2d (the run check covers `agents/`
  and `tools/`) and 2i (the "learning rows only" key).
- **Observation spec, Full view, Action spec, Reward profile:** read as
  served; sizes from S; `heads_used` and `arg_mask` decide which heads
  counted, never a second rule.
- **Bots through the door:** the records and `read_record`; its decision
  loop lifted into the shared door driver (2f); the capture construction for
  the reserved remedy of 3.3.
- **MatchSession + the `policy` controller, Match controller registry,
  Order:** `net` and `random` are registry rows that drive their team only
  through the decoder, and get the commander as door drivers.
- **Bot tournament, Headless match runner, Match recording / replay:** every
  measurement of a win rate (one instrument, one counting) and every replay
  Erik watches.
- **Training snapshot:** tags, the clean-tree checks, the run manifest.
- **Level data layer, Room graph, Spawn rooms, Match config, Cover:** the
  training rooms are written through the level writer with a room graph,
  and their matches through the match writer.
- **ML stack:** PyTorch as installed by `ml_stack.py`; no package by hand.

**New systems**, each with its draft rule (written into the project
instructions only when the code exists):

| system | patch | draft rule |
|---|---|---|
| Network spec | 2a | A network is a named spec in `agents/`; it meets the game only through an observation spec and an action spec, at any team size. A saved network names all three and is refused by any other. |
| Choosing | 2a | The chances of the choices under the masks, the draw, the probability of a given choice and which heads counted are ONE module, used when acting, when learning from reward and when copying. |
| Stored decision and batch | 2b | Only rows of a learning team are stored; what is kept per decision is one layout, copied from the environment's arrays; a batch carries the weights' version, both files' hashes and the build id. |
| Trainer | 2b | In `agents/`, outside the game's source tree; touches the game only through `BatchedEnv`. A collector and a learner with one hand-over each way; the learner refuses a batch from other weights; every round runs the live check. |
| Saved networks, training manifest | 2d | One store, outside the checkout; every saved network says what it was started from; the training manifest holds what the run manifest cannot (the training file, the room lists). |
| Copying | 2e | Learning from records uses the trainer's own choosing module and the records' own masks; the records are data, read packed, re-made when the bots or the observation change. |
| Door driver | 2f | The decision loop of a team behind the control panel is ONE driver in `src/`; the door bots, a saved network and the random player are subclasses that only choose. |
| Training rooms and the room lists | 2g | Small levels for single skills, generated through the level writer with a room graph; the generator and the three seed lists (training, validation, gate) are tracked, the rooms are not; a gate room is read once per network. |

## 15. Critique log (2026-11-15)

Four critics on Opus, one at a time, each blind to the others, each verdict
in `docs/rl_agent_rung2_design_2026-11-14_critique/`. All majors and the
blocker are adopted; the table says where. Minors are adopted in place
unless listed under it.

| lens | finding | fix | section |
|---|---|---|---|
| 1 systems | F1 the run check does not cover `agents/` | the check extended in 2d | 1, 10 |
| 1 | F2 the run manifest cannot hold the training file | a training manifest of its own | 1, 4.7 |
| 1 | F3 the `net` controller cannot be built as described | one shared door driver in `src/`; a registry flag; the network named per job | 7.2, 10 |
| 1 | F4 the sizes hold for six roster places only | sizes by S; the network indifferent to S; padding | 2 |
| 1 | F5 the watching path lacks three pieces | replay for any match; `--levels-dir`; registration | 7.2, 10 |
| 2 trainer | M1 the memory task cannot be passed; the learnable memory is one stretch | 12 and 24 decisions; the limit stated | 4.5, 8 |
| 2 | M2 the 2 % floor makes the copy misuse its kit | no floor on the hand | 3.2 |
| 2 | M3 a copycat passes the copy's measures | the repeat baseline, change points, a blanked-input run | 3.3 |
| 2 | M4 the value-only start does not protect the copy; arm C undefined | the ramp; forgetting measured; arm C written | 6 |
| 2 | M5 which rows learn is never said | the rule of 4.2 | 4.2 |
| 2 | M6 nothing guards the live run | the live check; the stand-in behaves as the environment | 4.8, 8 |
| 2 | M7 how the memory copy is trained is not said | whole matches in order | 3.2 |
| 3 measures | **B1 one held-back set is validation and exam** | three lists | 5.1 |
| 3 | M1 the gate has no numbers or rule | 200 duels, rooms resampled, the 90 % range | 0, 7.4 |
| 3 | M2 "decided" hides draws; hiding is nearly free | the clock allowance; the longer look-ahead | 0, 4.4, 5.4 |
| 3 | M3 `bots_v1` one against one is a reduced opponent | said; two more opponents printed beside | 5.4 |
| 3 | M4 the yardsticks set no mark | the rules of 5.2 | 5.2 |
| 3 | M5 which network must pass | one trained by reward, beside its start; one from nothing | 0, 7.4 |
| 3 | M6 the copy's records may hold the gate rooms | training rooms only | 3.1 |
| 3 | M7 no bar for the copy | the bar of 3.3 | 3.3 |
| 3 | M8 three runs, no primary outcome, overlap read as "no faster" | one deciding outcome; "not separated" | 6 |
| 3 | M9 rung 3's start decided twice | one place, section 0 | 0, 6 |
| 3 | M10 no rule for a fail | the fail table and the budget | 7.5 |
| 3 | M11 Erik's eye has no procedure | 7.4; his to decide | 7.4, 13 |
| 3 | M12 the parked spot-against-direction test is not fair | design 13.4 amended | design 13.4 |
| 4 cost | M1 small rooms are not faster per learning decision | the band; step 3 measures | 9 |
| 4 | M2 the card, not the computer's memory, limits the piece | lumps of 512; the spill off | 4.4 |
| 4 | M3 the three PCs are neither cheap nor equal | budgets in decisions; one of each arm per PC | 6, 9 |
| 4 | M4 the first real run comes six patches in | patch 2c | 0, 10 |
| 4 | M5 patch 2b is three patches | split into 2b, 2c, 2d and the lesson | 10 |
| 4 | M6 the farm observes soldiers that do not learn | a run-file key in 2i | 9, 10 |
| 4 | M7 PyTorch in every tournament worker | on the processor, one thread; workers pinned | 7.2 |
| 4 | M8 measuring during a run costs 10 to 40 % unless sized | a small second environment, every 50 rounds | 7.3 |
| 4 | M9 the copy's 20 minutes needs the data read in runs | read packed | 3.1 |

**Minors not adopted, with why:**

- Lens 3 m1's optional second, harder gate reading (one room shape held out
  entirely): not in rung 2; the training-rooms rate beside the gate rate
  shows the gap.
- Lens 4 minor 3 (store only the used list entries, 63 KB to about 35 KB in
  a duel): not needed while the computer's memory is not the limit; before a
  spread-out run.
