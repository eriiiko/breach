# Rung 2 design critique, lens 4: does it fit the machines and the time

Critic: Claude Opus 5.5 (subagent), 2026-11-15. Object: `docs/rl_agent_rung2_design_2026-11-14.md`
at commit d3fd2a5e (branch `192-rung2-cadence`). Read: the whole design; roadmap 3 (rungs 2, 3)
and 5; design 3.4, 7, 8, 13; the 1i note's farm reading; the 1j note's size, cost, measurement and
surprises sections; `residency_benchmark_2026-11-07.md` (sections 1, 2, P1a, G1, the lock-step
loss); the hardware lines of the environment notes; read-only from the main checkout:
`src/rl_env.py` (`BatchedEnv`, `EnvRunner`, the worker loop), `src/simulation/obs_spec_v1.py`
(`shapes`, `full_shapes`, `observe_rows`), `tools/rl_smoke.py`, `tools/bot_tournament.py`,
`level_loader.py` / `tools/gen_ship.py` (what a level costs to load). No game code was run; the
arithmetic is three small scripts that import nothing from the game.

Every input below is marked MEASURED (with where it is written) or ASSUMED.

## Verdict

**Survives with fixes.** BLOCKER 0, MAJOR 9, MINOR 11.

The design's memory arithmetic for the stored decision is right (62.7 KB, recomputed), and its
instinct that batch size is a memory question is right, but it points at the wrong memory: on the
16 GB desktops host RAM is comfortable at 8 000 and even at 32 000 decisions; what binds is the
graphics card's activation memory during learning, where "4 pieces" of 8 000 (2 000 decisions per
piece through both networks) needs roughly 6 GB of saved activations in 32-bit and does not fit an
8 GB 3070 beside its context, let alone the laptop's 6 GB, and on Windows an overflow does not
fail, it silently pages to system memory. The larger correction is throughput: the small training
rooms are not "much faster" in what PPO consumes, learning-soldier decisions per second, because
the medium-ship figure of 720 counts eight learning soldiers per match and the range and duel have
one, while a tick's fixed Python costs and the observation of non-learning rows do not shrink with
the room. A defensible band is 320 to 1 320 learning decisions a second (central about 730), so the
roadmap's "duel in about an hour" is 1 to 90 hours depending mostly on the unknown number of
decisions to the pass mark, and the experiment is between an afternoon and eleven days of three
PCs (rung 2 in all: about a day of one PC at the low end, three weeks at the high end, before
reruns). None of this stops the plan; each item has a small fix, mostly a number to state, a measurement
to take first, or an order to change: the first real-game run should come right after the trainer's
core, before the copy, the `net` controller and the room generator, so that these costs are
measured in the first week instead of the fourth.

## Findings

### MAJOR 1. The small rooms are not faster in learning decisions per second (sections 5, 9; roadmap 5)

**The design:** "the farm in a training room: unknown; the rooms are meant to be some twenty times
smaller than the medium ship, with one or two soldiers" (9), and by implication the roadmap's
"duel: about ten million, an hour" and "shooting range: minutes" stand.

**Arithmetic.** What PPO consumes is decisions of LEARNING soldiers. The 720 a second is
362 match-ticks / 4 x 8 learning rows (MEASURED, 1i: 64 envs / 16 workers, both teams on
`policy`). One medium env-decision costs about 85 ms of one core (MEASURED, 1i: "1.0 % of an 85 ms
env-decision") and yields 8 learning decisions: about 10.6 ms of core time per learning decision.
In the range and the duel an env-decision yields ONE.

What an env-decision costs in a training room, by parts:

| part | medium ship | training room (400 to 1 500 cells) | basis |
|---|---|---|---|
| one tick, physics | ~15 ms | 0.4 to 3.8 ms | MEASURED per cell: CPU 2.1 to 2.5 us/cell at the substep cap, ~1 us/cell at one substep (benchmark section 2; G1 "small ship settles ... 5.3 to 2.8 ms"); room sizes ASSUMED from "one to three rooms" and "twenty times smaller" |
| one tick, gameplay, controllers, books, session | ~3 ms | 0.4 to 1.5 ms | MEASURED on medium: gameplay slots 1.85 ms, controllers 1.5 ms, books 0.217 ms a tick (benchmark section 1; 1i); much of it per tick, not per cell; the small-room share ASSUMED |
| observation + full view + masks, per decision | 7.7 + 6.6 + ~0.8 ms (8 rows) | 0.6 to 1.2 ms per observed row | MEASURED on medium (1g via the full-view row of the project instructions: ~0.8 ms per row each); small-room per-row cost ASSUMED |
| rows observed | 8, all learning | duel 2 (one is `bots_v1`'s); range 2 to 5 (four may be idle targets) | `EnvRunner.observe` builds every living row and the full view for every in-match row (`rl_env.py` 948-983) |

So a duel env-decision is 5.6 to 24 ms of one core (central ~10), a range env-decision 5.6 to
21 ms. On the 16-worker farm (9.6 single-core equivalents, MEASURED G1; lock-step efficiency 0.84
at 4 envs per worker, MEASURED "the farm's lock-step loss") with 64 envs and 4 to 6 ms of serial
driver work per step (MEASURED 5.9 ms for the random driver at 64 envs; a GPU forward ASSUMED
similar, finding MINOR 2), the model reproduces the medium reading (752 against 724 measured) and
gives:

| task | learning decisions / s | one round of 8 000 collected |
|---|---|---|
| duel | 320 (high cost) / 730 (central) / 1 320 (low cost) | 25 / 11 / 6 s |
| range | 370 / 720 / 1 320 | 21 / 11 / 6 s |
| medium fireteam (for comparison) | 720 MEASURED | 11 s |

"Some twenty times smaller, so much faster" does not follow: per learning decision the training
rooms are between half and twice the medium fireteam, centrally the same. Run lengths at the
roadmap's sample counts (ASSUMED, as the roadmap's own: range 0.5 to 5 M, duel 5 to 50 M; the
roadmap's figures are "estimates from other people's tasks"):

| run | roadmap | this estimate (collect x learning x evaluation overheads) |
|---|---|---|
| range | minutes | 0.13 h (0.5 M at 1 320 / s) to 7.8 h (5 M at 370 / s) |
| duel | about an hour | 1.3 h (5 M at 1 320 / s) to 90 h (50 M at 320 / s); 5.7 h at 10 M, central throughput |

(Overheads: learning 0.15 to 0.6 of collecting, MINOR 1; evaluation 0.1 to 0.3, MAJOR 8.)

**What breaks:** nothing, but every schedule built on "minutes" and "an hour" (the roadmap's cost
table, the experiment's "three PCs make this cheap") is 2 to 90 times short, and the larger factor
is the sample count, which no one can know before step 4.

**Smallest fix:** in section 9, replace "unknown" by the band above with its derivation, state that
the unit is LEARNING decisions, and add one measurement to the first real-game run (finding MAJOR
4): `rl_smoke.py` on a hand-made training room at 64 envs / 16 workers, read in learning decisions,
with the per-step driver time split out. Give the roadmap's rung 2 rows the same dated note the
fireteam row got.

### MAJOR 2. The batch is limited by the card's activation memory, not by host RAM; "4 pieces" does not fit the 8 GB and 6 GB cards (4.2, 4.4, 12)

**The design:** "A batch of 8 000 decisions is 0.5 GB; a fireteam batch of 32 000 would be 2 GB. On
the 16 GB desktops, beside 16 farm workers, that sets the batch size as much as any theory does"
(4.2); "pieces per pass: 4" (4.4); risk "memory on the 16 GB desktops: the batch sized by 4.2" (12).

**Host RAM, recomputed** (duel or range, Home Desktop class, 16 GB):

| item | GB | basis |
|---|---|---|
| Windows and the usual desktop programs | 3 to 5 | ASSUMED |
| 16 farm workers | 0.75 to 2.4 | MEASURED 45 to 49 MB working set each with the crop spec (P1a, G1); spec v1, its context and a bot controller ASSUMED to raise it to 60 to 150 MB |
| the trainer process (Python, PyTorch, the CUDA runtime and libraries) | 1.5 to 3 (commit) | ASSUMED, typical for a CUDA PyTorch process on Windows |
| the stored batch, 8 000 x 62.7 KB, plus one staging copy | 0.5 to 1.0 | recomputed from `obs_spec_v1.shapes` / `full_shapes` |
| the env's shared and env-owned buffers (4 copies of every row) | 0.03 (duel, U = 2) to 0.13 (range, U = 8) | `BatchedEnv.__init__` |
| **total** | **6 to 11.5** | |

8 000 fits with room to spare, and 32 000 (+1.5 GB) fits too. Host RAM does not set the batch.
(The real host risk is the COMMIT charge when processes that do not pin their BLAS threads run
beside the farm: the 1j incident, finding MAJOR 7.)

**The card, estimated.** Saved activations for one decision through network v1 as design 8.1 and
8.4 give it (fp32, PyTorch eager; my count of the tensors a backward pass keeps: the converted
window inputs, each convolution's output, the attention's q, k, v, its probabilities, the
feed-forward's two activations, the normalisations): the soldier about 0.96 MB (the two map
readers 0.35 MB, two attention layers over 40 tokens 0.54 MB), the value network about 1.38 MB (two
attention layers over 65 tokens 0.96 MB). Together **about 2.35 MB per decision** (ASSUMED within
+- 50 %: implementation choices move it).

| minibatch (decisions per piece) | activations fp32, x1.3 backward peak | with fp16 autocast | + context, cuDNN workspace, batch on card, desktop | fits |
|---|---|---|---|---|
| 2 000 (8 000 / 4 pieces) | 6.1 GB | ~3.4 GB | +1.3 to 2.3 GB | fp32: 3060 (12 GB) only; fp16: 3070 yes, laptop barely |
| 1 000 (8 pieces) | 3.1 GB | ~1.7 GB | same | fp32: 3070 yes, laptop barely; fp16: all |
| 500 (16 pieces) | 1.5 GB | ~0.9 GB | same | all |
| 8 000 (32 000 / 4 pieces, the fireteam) | 24 GB | ~13 GB | | none |

**What breaks.** At the design's numbers the learner does not fit the Home Desktop's 8 GB card in
32-bit, nor the laptop's 6 GB. On Windows this will not show as an out-of-memory error: since the
2023 drivers the default "CUDA sysmem fallback" pages the overflow into system RAM over PCIe, and
the update runs several times slower, which a curve of "seconds spent learning" shows only as
"learning is slow". The 32 000-decision fireteam batch with 4 pieces fits no card we have.

**Smallest fix.** (a) In 4.4 state the piece size in DECISIONS (a minibatch of 2 000 is a fine
starting point), and let the learner split a piece into micro-batches with gradients accumulated,
so that the update is the same on every card (the normalising layers carry no batch statistics,
design 8.3, so this is exact up to rounding). (b) Make 2a's weight-count test also print the peak
card memory of one training step at minibatch 500 and 2 000, both networks, fp32 and fp16, on the
machine at hand. (c) Set the sysmem fallback to "prefer no fallback" for the training Python on
every PC, so an overflow fails loudly. (d) Rewrite 4.2's sentence and risk 12's row: the limit is
the card per piece; host RAM is not binding at 8 000 or 32 000.

### MAJOR 3. "Three PCs make this cheap": the experiment is an afternoon to eleven days of three PCs, and the three PCs are not equal (6, 4.6, 0)

**The design:** "At least three repeats of each arm ... Three PCs make this cheap" (6); "three PCs
run three experiments and share saved networks, which are small" (4.6); the "earns it" rule runs
"the first stretch of rung 3's training ... from each candidate, three times each" (0).

**PC-time.** Six duel runs (A x 3, B x 3), two per PC, each one duel run of MAJOR 1:

| | per run | per PC (2 runs) | the laptop (x 1.5, below) |
|---|---|---|---|
| low (5 M decisions at 1 320 / s) | 1.3 h | 2.7 h | 4 h |
| central (10 M at 730 / s) | 5.7 h | 11.4 h | 17 h |
| high (50 M at 320 / s) | 90 h | 180 h (7.5 days) | 270 h (11 days) |

Not yet counted: the memory-off duel of 5.4 (one more run), arm C if needed (three more), the
range runs and their first sweep (passes 2 / 4 / 8: three runs), and reruns: a first self-written
trainer usually needs two or three goes at each task before a curve can be trusted (ASSUMED, the
lesson-0 experience the design itself plans tests against). The "earns it" comparison adds nine
runs of rung 3's first stretch, whose length is not given; at 10 to 20 M decisions on the medium
ship at 400 to 720 / s that is 35 to 125 PC-hours, 12 to 42 hours of all three PCs.

**The PCs.**

- **The laptop's card** (6 GB) needs smaller pieces than the desktops (MAJOR 2). If it runs other
  update settings, its arm differs by more than the arm: with micro-batching the update is the
  same, without it the comparison is confounded by the machine.
- **The laptop's processor** has performance, efficiency and low-power cores. The farm is
  LOCK-STEP: every step waits for the slowest worker (`BatchedEnv.step` broadcasts and waits for
  all). Workers that the scheduler puts on efficiency cores set the pace of all; a laptop also
  throttles under hours of all-core load. ASSUMED 0.5 to 0.9 of a desktop's farm, hence the x 1.5.
  Not measured: `rl_smoke.py` has never run there.
- **The laptop is not yet on the ML stack** (the master instructions' rollout list leaves it
  unticked); its CPU build is per checkout.
- **The Work Desktop is not a dedicated machine either:** whatever else holds its card or cores
  during a multi-day run makes its runs slower and noisier than the Home Desktop's. Check before
  placing runs there.
- **The Home Desktop is also where the residency arc's CUDA gates run** (`cuda_green`) and where
  the copy's records live; a training run takes all its sixteen threads for hours to days.
- **Moving things.** One tagged checkout per PC with its own CPU build (and CUDA build only if a
  gate needs it), the room banks made again from the seed lists (deterministic, so the bank sha256
  in the run manifest checks them), the copied network (about 32 MB with the learner's state,
  MINOR 5) to two other PCs: small; the transfer between PCs (the `pc-transfer` procedure or Drive)
  is checked once on each PC before 2i.

Measured in decisions (as section 6 reads the result) the machines do not bias the comparison;
measured in wall clock they do, and the experiment ends when the slowest PC ends.

**Smallest fix.** (a) Section 6: read every arm at equal DECISIONS (it already does) and say the
budget per run in decisions, with a stop: e.g. each run to the pass mark or to a cap (say 30 M), so
the worst case is bounded. (b) Place one repeat of EACH arm on each PC (A and B on every machine),
not one arm per machine, so a machine effect cancels. (c) Before 2i: the laptop brought to the ML
stack, `rl_smoke.py` run there once on a training room, and its pieces set by the micro-batch rule.
(d) Section 0: give "the first stretch" a length in decisions, so the "earns it" cost is known.
(e) A run calendar: which PC is busy when, since a running farm takes the whole machine.

### MAJOR 4. The trainer meets the real game only at patch 2g, after six patches; there is a cheaper first end-to-end path (10, 9, 12)

**The design:** patches 2a to 2f, then "2g: the range (5.3) ... needs the farm: yes" (10); "learning
time per batch: unknown ... measured on the toy tasks and the first range run" (9); risk "learning
takes as long as collecting: measured in 2b and 2g before any size is fixed" (12).

**What hides until 2g.** Every unknown of section 9 that is about the machines: the farm's speed in
a training room (MAJOR 1), the driver's serial time per step with a GPU forward in it (MINOR 2),
the card's memory with the real shapes (MAJOR 2), the learn / collect ratio (MINOR 1), the cost of
observing rows that do not learn (MAJOR 6). Also every seam of `BatchedEnv` the toy tasks only
imitate: an env reset inside a step and its `final_obs` / `final_full_view`; `rows_ended`; the
full view under `info`, not `obs`; the rule that a v1 action step needs the `list_uid` of the
observation it answers (`EnvRunner.step` raises otherwise); `masked_choices`. A mismatch between
the toy environment and the real one is found after 2c, 2d, 2e and 2f are built on the trainer's
shape. At one implementation agent at a time (master rule) and roughly one patch every day or two
(ASSUMED from rung 1's pace), that is one to two weeks before the first real number.

**What a first real-game run needs, and what it does not.** It needs network v1 (2a) and the
trainer's core (choosing, the stored decision, grading, the update: 2b without saving-and-resume,
the hand-over headers and the lesson document). It needs ONE small level with a room graph, which
the level writer already makes: the rung-1 tests build boxes with a room graph and a door through
`level_lib` (the 1f box, the 1j door box), and a range is such a box with one rifleman and two
`idle` targets in the other spawn room. It does not need the room generator, the banks, the
yardsticks, the copy, the `net` controller or the new records.

**Smallest fix: one more step, "2b'" (or the first half of 2g), right after 2b's core:** the range
from scratch on one to three hand-made rooms written through the level writer, no memory, no
held-back set, 64 envs / 16 workers, one to three hours on the Home Desktop. Its report gives the
learning-decision rate, the per-step split (workers, driver copies, forward, learning), the card's
peak memory, and whether the curve rises at all. Then 2b's rest, 2c to 2f, and 2g proper on the
bank to the mark. It touches no game file (the levels are outside the repo, as 5.1 wants) and does
not change what waits for P6.

**Where else a week can go.**

- **2e's generator** is "built on the ship generator's second stage ... as far as that fits rooms
  this small; the patch's note settles it" (5.1). That stage's gates were written for ships (mirror
  symmetry, airtight via `find_leaks`, A* spawn reachability, formation cells, vent rules, lamp
  budgets). If they refuse one-room levels, 2e becomes a generator patch. The cheap path is the
  precedent of the project's generators table: a small `tools/gen_training_rooms.py` writing
  through `level_lib` directly with `level_lib.write_room_graph`, and only the airtight lint and
  reachability reused. Decide which before 2e starts.
- **2b's size** (MAJOR 5).
- **2i's three-PC set-up** (MAJOR 3).

**What can be gated without a graphics card or without the game.** Without the game: 2a, 2b, 2c
(the records exist). Without a card: 2a's tests; 2b's hand-worked grading case and the toy tasks at
a reduced width of the same network code (MINOR 4); 2c's 1 000-decision memorisation test (1 000
decisions x ~176 MFLOP per training pass is a few seconds per pass on the CPU). The full copy
(1.1 M decisions) and every game run need a card.

### MAJOR 5. Patch 2b is three patches in one agent run (10, 8)

**The design:** "2b: the trainer's core: the stored decision, grading, the update, the two parts and
their hand-over, saving, the curves, the toy tasks (4, 8). The lesson document 'The trainer, line by
line' is written from this code" (10).

**Size.** Against a rung-1 patch (1j's note sizes its own, one controller, one record format and
one tournament, at "~1 implementer session"): 2b holds (i) the stored-decision layout with its five-group
rewards, flags, masks, counted heads and the memory vector per stretch; (ii) grading, five value
outputs, rescaled targets with running statistics, the endings and the mid-match cut, the hand case;
(iii) the PPO update over five autoregressive heads with a pointer head, counted-head entropy, the
early stop on policy movement; (iv) the collector against `BatchedEnv`'s API and the learner with
versioned headers and refusal; (v) saving and resuming with the learner state and statistics; (vi)
the curves file and its plotting script; (vii) THREE toy environments shaped like the real layout,
one of them with memory, deaths and endings, each a stochastic test; (viii) a lesson document.
Items (iii) and (vii) are the ones the design itself names as where self-written trainers fail, and
an agent run that also has to write (v), (vi) and (viii) is the run most likely to be cut off or
to hand back a green suite over a subtle fault. Memory (4.5) "comes second", yet toy task 3 needs
it, so 2b also builds the memory path the range does not use.

**Smallest fix.** Split:

| patch | what | gate |
|---|---|---|
| 2b-1 | stored decision, grading (+ the hand case), the PPO update, the collector against a fake env with `BatchedEnv`'s exact API; toy tasks 1 and 2 (no memory) | the hand case; tasks 1 and 2 solved at reduced width on the CPU |
| 2b' | the first real run (MAJOR 4) | a measured rate, memory, split; a curve |
| 2b-2 | saving / resume, the hand-over headers and refusal, the curves script; the memory path and toy task 3 | resume reproduces counters and weights; task 3 solved |
| 2b-3 | the lesson document | Erik reads it |

The memory path then lands where the design wants it: after the range, before the duel (2h).

### MAJOR 6. The farm observes every living row, so the range pays for its targets' eyes, and the fix touches a game file the patch table says is untouched (5.3, 5.4, 10)

**The design:** the range is "one rifleman (the learner) and one to four standing marines of another
team, driven by `idle`" (5.3); the duel is against a `bots_v1` rifleman (5.4); the table marks 2g
and 2h "touches the game? no" (10).

**The code.** `EnvRunner.observe` builds observation spec v1 for every row whose unit is alive
(`row_units`, every team's roster), and the full view for every in-match row (`roster_units`),
whatever the row's controller (`rl_env.py` 906-983; `observe_rows` loops over all non-None rows).
The driver then copies every row of every key into its own buffers each step (`BatchedEnv._obs`,
`_full_view`). The layout's S is the largest `max_per_team` over the run's levels
(`make_layout`), so a range with up to four targets has U = 2 x 4 = 8 rows in EVERY env.

**Cost.** In the duel, two rows are observed for one that learns; in the range, two to five for
one. With MAJOR 1's central numbers: the duel loses about 9 % (731 against 796 learning decisions a
second if the opponent's row were skipped), the range about 32 % (718 against 945 at three targets).
The range's driver also copies 64 x 8 rows x 62.7 KB = 32 MB per step instead of 8 MB (about 3 ms
of the serial part; MEASURED rate for such copies: 9.8 MB in 0.65 ms into an existing array, the
lock-step section).

**What breaks:** only time, but the cure (a run-file key such as `[obs] rows = "policy"` so that
only rows of teams on the `policy` controller are built, and a full view only for those) is a change
to `src/rl_env.py`, a game file, in a patch the table says touches none.

**Smallest fix.** Either accept the cost and say so in section 9, or add the key to 2b' / 2g's
scope and mark 2g "touches the game: `rl_env.py`, one run-file key". Independently: give the range
ONE target per room for the first runs (U = 2): two rows observed instead of up to five.

### MAJOR 7. The `net` controller puts PyTorch in every tournament worker; nothing says where it runs or what it costs (7.2, 3.3, 10)

**The design:** "A match controller `net`, built like the door bots ... It lives in `agents/` and is
registered by the tool that needs it" (7.2); "the bot tournament gets it as a player" (7.2); the
copied team is measured in the tournament against four opponents (3.3); 2d, 2i use it.

**The code.** `bot_tournament.py --workers N` runs a spawn `Pool(N)` (lines 416-418): every worker
re-imports the game, and with a `net` team it would import PyTorch and load the network. It does not
pin BLAS or OpenMP threads (the 1j note's surprise: "~2 GB of commit each; 12 of them plus the suite
exhausted the 16 GB machine's commit charge"; still so at the main checkout's head).

**Cost per worker, by where the network runs** (ASSUMED, typical PyTorch on Windows):

| | host commit per worker | card memory per worker | 12 workers on a 16 GB desktop |
|---|---|---|---|
| on the card | 1.5 to 2.5 GB (CUDA libraries) + the unpinned BLAS ~2 GB | 0.3 to 0.5 GB context each | 40+ GB commit, 4 to 6 GB of card: refused or paging |
| on the CPU, threads not pinned | 0.5 to 1 GB + ~2 GB BLAS | 0 | ~30 GB commit; PyTorch's default of 8 threads per worker x 12 = 96 threads on 16 |
| on the CPU, one thread, BLAS pinned | 0.5 to 1 GB | 0 | 6 to 12 GB: fits, beside nothing else |

Time on the CPU, one thread: a soldier forward is ~59 MFLOP (my count from design 8.1), so a team
of 4 to 6 rows is 0.25 to 0.35 GFLOP per decision, ~10 to 20 ms with PyTorch's per-operation
overhead, once every 4 ticks: +2.5 to 5 ms on a door duel's ~25 ms tick (MEASURED, 1j's walls
under load), i.e. 10 to 20 % slower matches. Acceptable, if it is the choice made.

**Smallest fix.** Section 7.2 names it: the `net` controller runs on the CPU with one thread;
the tournament exports `rl_env.WORKER_ENV` to its pool (one line, the farm's own fix) before 2d;
2d measures one worker's private bytes with a `net` team and sets `--workers` from it.

### MAJOR 8. In-run evaluation on a lock-step farm costs 10 to 40 % of training unless it is sized, and it needs a second env (7.3)

**The design:** "Every so many rounds, the farm plays the held-back rooms with the current network
against the fixed opponent and the win rate is one more point on the curve" (7.3).

**Arithmetic.** One evaluation pass = one match in each of 64 envs. Lock-step, it lasts as long as
the LONGEST match: a 90 s duel is 540 decisions, so 540 steps at 48 to 200 ms (MAJOR 1) = 26 to
108 s, the farm time of about 4.3 training rounds of 8 000; it yields 64 matches, a win rate within
about +- 6 points (binomial at 0.5). Every 10 rounds that is +43 % on the run's wall clock; every 25
rounds +17 %; every 50 rounds +9 %. To tell 0.45 from 0.55 needs about 100 decided matches per point
(two passes).

**A hidden constraint.** A `BatchedEnv` draws each episode's level from ALL the run's levels
(`episode_draw(run_seed, e, k, len(levels))`); there is no per-env level set. So "the farm plays
the held-back rooms" means a second `BatchedEnv` on a second run file: either 16 more worker
processes (1 to 2.4 GB more host memory, MAJOR 2's table) that sit idle between passes, or the
training env closed and re-opened (each worker start re-imports the game and loads every level of
the run: seconds per pass).

**Smallest fix.** Section 7.3: (a) a second, small env for evaluation (4 workers x 4 envs, created
once, idle between passes), (b) its cadence set by wall-clock share, e.g. one pass of 64 matches
every 50 rounds (under 10 %), and (c) the free signal used between passes: the training matches are
themselves against the fixed opponent, so the training win rate is on the curve every round at no
cost; the held-back pass checks it does not generalise worse.

### MAJOR 9. The copy's "about 20 minutes" holds only if the unpacked data is read in contiguous stretches; the layout the design names reads it row by row (3.1, 9)

**The design:** "One decision is about 40 KB as served, so the whole set is about 45 GB ... unpacked
once into one file per array on disk that is read in place, rows of dead soldiers and empty roster
places left out" (3.1); "the copy: about 20 minutes per training run on one graphics card,
estimate" (9).

**Arithmetic.**

- Data: 1.1 M living decisions (the design's extrapolation of 662 766 in 360 matches, MEASURED 1j)
  x ~40 KB (observation 39.8 KB + masks and choices; recomputed from the shapes; the records carry
  no full view, so 45 GB is already the copy's whole need) = 44 GB.
- GPU side: the soldier network alone, forward and backward, ~176 MFLOP per decision (my count).
  At 2 to 4 TFLOP/s effective on the 3070 (ASSUMED, eager PyTorch on small convolutions and
  attention): 11 000 to 23 000 decisions a second, one pass over the data in 48 to 97 s. It needs
  0.45 to 0.9 GB/s of input.
- Page cache: with the trainer and Windows resident, 5 to 8 GB of a 16 GB desktop can cache files,
  11 to 18 % of the set: every pass reads from disk. The Home Desktop's disk is NVMe (the
  environment notes); the Work Desktop's is not written down.
- "One file per array": a decision is a row in about twelve files (`vector`, `list`, `list_mask`,
  `list_uid`, the four windows, four masks, the argument mask, the choices). A shuffled minibatch
  reads each decision's row from each: ~12 random reads, the small ones each faulting a whole 4 KB
  page. Through memory-mapped fancy indexing the faults are served one after another per thread
  (ASSUMED 20 to 100 us per fault from the disk): 0.25 to 1.2 ms per decision per reader, 3 000 to
  16 000 decisions a second with four reader processes, i.e. at or below the GPU's rate: a pass in
  70 to 370 s.
- A copy needs some 5 to 20 passes (ASSUMED; early-stopped on held-back loss). In 20 minutes:
  12 to 25 passes if the reading keeps up with the card, 3 to 17 if it does not. So "20 minutes"
  holds with contiguous reads and becomes 40 to 100 minutes with random rows.
- The memory variant (3.2, "without memory first, then with") trains on STRETCHES of consecutive
  decisions of one soldier. A record is (decisions, S rows, ...) per match; leaving dead rows out
  while keeping that order interleaves the six soldiers, and a stretch becomes six-way strided.
- The unpack itself: 600 records, 4.2 GB compressed (MEASURED, 1j: mean 7.0 MB), inflated at
  ~0.3 to 0.5 GB/s per core (ASSUMED zlib) and written at 1 to 2 GB/s: 5 to 15 minutes. Cheap.

**A smaller form?** The windows are already served as float16 / uint8 (they are 28.8 of the
40 KB), so storing "as served" changes nothing. The list (10.5 KB, float32, "mostly padding" per
1j) could keep only its used entries, and float16 would halve it; together perhaps 40 -> 32 KB, not
a change of regime. The regime-changing option is NOT to unpack: the records compress about
10 times (70 to 120 MB raw per match by the 1j note's estimate, 5 to 14 MB compressed,
MEASURED 1j), so loader processes
inflating whole matches into a shuffle buffer of some 30 matches (~2 GB of RAM) inflate about
90 GB per pass (a record also holds its dead and empty rows' zeros: ~625 decisions x 6 rows x
40 KB = 150 MB per match) at about 1.6 GB/s with four cores: about a minute a pass, the card's rate, with no 45 GB file on any disk and nothing
to redo when the records are made again (3.1 says they will be).

**Smallest fix.** Section 3.1: write the unpacked data soldier-major, (match, soldier, decision),
one record per decision in one file or one file per array but read in contiguous runs of whole
stretches, shuffled at the stretch level; or stream the compressed records as above. Either way
2c measures decisions read per second before the first long run.

### MINOR 1. Learning time per batch can be bounded now: 0.1 to 0.7 of collecting, not "near" it (9, 12)

Both networks, forward ~59 + ~75 MFLOP per decision (my count from design 8.1 / 8.4: the
convolutions on 25 x 25 dominate the readers, the attention over 40 and 65 tokens the rest);
training ~3x forward; 8 000 decisions x 4 passes = 12.8 TFLOP a round. At 3 to 6 TFLOP/s effective
on the 3070 (ASSUMED; the memory-traffic bound gives a similar 1.2 to 1.8 s) that is 2 to 4 s; the
3060 3 to 6 s; the laptop's card 4 to 11 s. Grading adds one value forward over the batch (0.6
TFLOP, under 0.5 s). Against 6 to 25 s of collecting (MAJOR 1) the ratio is 0.1 to 0.7 on the
desktops, up to ~1 on the laptop or at 8 passes. So the farm is the bottleneck, the card idles over
90 % of the collecting, and the synchronous form loses 10 to 40 % of wall clock to the alternation:
not worth the asynchronous form in rung 2. **Fix:** put this bound in section 9 beside "measured on
the first range run".

### MINOR 2. Acting on the card while the farm waits costs 3 to 6 ms a step, 3 to 12 % (4.1)

Per step, for 64 learning rows: the observation to the card, 64 x 40 KB = 2.6 MB (0.3 to 1 ms
pageable, a quarter of that pinned; the full view is not needed to act); one forward of the soldier
network, launch-bound at roughly 150 to 200 kernels (ASSUMED 5 to 15 us each on Windows): 1.5 to
3 ms; one synchronisation to read the five choices; the stored rows, 64 x 62.7 KB = 4 MB into the
batch (~0.5 ms); the env's own copies of every row (MAJOR 6). Total 3.5 to 6 ms against a 48 to
200 ms step; every worker idles meanwhile (lock-step), so it adds directly. Acceptable. **Fix:**
say in 4.1 that the value network is NOT run while collecting (grading runs it over the batch,
4.1 step 2; a per-step value forward would add another 1.5 to 3 ms), and use pinned staging.

### MINOR 3. The stored decision is right; the duel stores mostly padding (4.2)

Recomputed from `obs_spec_v1.shapes` and `full_shapes`: 62.7 KB at S = 6 (observation 39.8,
masks 0.13, full view 22.7, choices / probabilities / rewards ~0.1), 62.2 KB for the range (S = 4),
61.4 KB for the duel (S = 1). The duel saves almost nothing because the full list is fixed at
64 x 85 float32 = 21.8 KB, of which a duel uses some 4 to 12 entries (1.4 to 4.1 KB), and the
observation's list 3 to 8 of 34 entries. Storing only used entries would bring a duel decision to
about 35 KB. Not needed at 8 000 (host RAM is not binding, MAJOR 2); worth having before a
spread-out run (MINOR 6). **Fix:** none required; one line in 4.2 that the 63 KB is padded.

### MINOR 4. "Solved in seconds without the game" holds only for a reduced-width network (8)

A PPO round of 2 000 decisions through both full-width networks is about a second on the 3070
(MINOR 1); a pointer task needs tens to hundreds of rounds, a cue-recall task over 20 decisions
typically 10^5 to 10^6 decisions (ASSUMED from the memory-task literature): minutes to an hour on
the card, 10 to 50 times that on a CPU, and stochastic. As suite tests they would need a card, a
new convention for PyTorch on the card inside pytest (the project runs its CUDA checks in a
subprocess), and seeds. **Fix:** the toy tasks run the same network code at a reduced width (and
smaller windows) on the CPU in the suite, with a pass threshold over a few fixed seeds; one
full-width run on the card is the patch note's measurement, not a suite test.

### MINOR 5. A saved network is ~32 MB, not 6 MB, and a long run makes hundreds (4.6, 4.7)

Weights 1.55 M + 1.1 M x 4 B = 10.6 MB; with Adam's two moments 32 MB (the "about 6 MB" of 4.6 is
the soldier's weights alone, the right number for an opponent). A 40-hour run saving every 15
minutes: 160 saves, ~5 GB. **Fix:** 4.7 states a retention (the last, the best, every Nth) and that
an opponent or a shared network is the soldier's weights alone.

### MINOR 6. A spread-out run's traffic is compressible about ten times (4.6)

0.5 GB per round of 8 to 30 s is 17 to 60 MB/s sustained, which a relayed link will not carry
(as the design says). The records already show the observations compress about ten times
(MAJOR 9), i.e. 2 to 6 MB/s, which a direct path may. **Fix:** "measured in advance" includes the
compressed size of a batch, not only the link.

### MINOR 7. Recording again: "about an hour" holds for the same 600 records; the new set is not sized (3.1, 2f)

The 1j design runs were 800 matches for 600 door-team records at 56 to 87 s each (MEASURED 1j
walls), 800 x ~60 s / 12 workers = 67 min. The tournament must be run with
`OPENBLAS_NUM_THREADS=1` exported (MEASURED 1j: ~2 GB commit per worker otherwise; 12 workers plus
anything exhausted the commit). With about ten freighters and the duel rooms, how many matches are
recorded is not said; disk is ~7 MB a record compressed (MEASURED). **Fix:** 3.1 gives the match
count per ship and per duel room, hence the hours and GB.

### MINOR 8. Disk: only the PC that copies needs the copy's data (3.1)

45 GB unpacked on the copying PC (or none, MAJOR 9); the Home Desktop has ~200 GB free on C:
(1j note) and the records are there. The Work Desktop's free space is not in the environment notes:
check before a copy runs there. Room banks are small (a tiny level is a toml, a csv, a png of 16
pixels per tile, a match file, `rooms.npy`) and are made again from seeds. Saved networks: MINOR 5.

### MINOR 9. A running farm owns its machine; nothing else heavy may run beside it (6, 10)

Sixteen workers take all sixteen threads; the 1j measurement saw the test suite fail on memory
beside twelve tournament workers. The residency arc's gates, any implementation agent's suite and
Erik's own use of the PC collide with a multi-hour training run. **Fix:** with MAJOR 3's calendar,
one rule in section 10: no suite, build or tournament on a PC while its farm trains.

### MINOR 10. Section 9's small-ship figure: about 2 000 by the newer ratio, and the 720 is a random driver's (9)

The G1 reading's full-game farm at decimation 4 and 16 workers gives small / medium = 1 342 / 487 =
2.76 (MEASURED), so 720 x 2.76 is about 2 000, not 2 300. Both are with the random-legal driver;
a driver that acts on the card costs 3 to 6 ms a step more (MINOR 2): a few per cent at the
medium ship's 700 ms step. **Fix:** "about 2 000 (G1's ratio)"; note the driver.

### MINOR 11. The batch's shape (envs x steps) is not stated, and it sets the lock-step efficiency (4.4)

A batch of 8 000 learning decisions with one learner per env is N envs x T steps. Four envs per
worker gave +16 % over one (MEASURED, the lock-step section; eight per worker not measured), and the
driver's per-step costs amortise over N. **Fix:** 4.4 starts at N = 64 (4 per worker), T = 125, and
2b' also measures N = 128 (T = 62).

## A budget table

All estimates unless marked; the derivations are in the findings. "Minibatch" = decisions per
optimiser step (one piece).

**Memory per PC during a run.**

| PC | range / duel: host | range / duel: card at the learner's peak | copy: host | copy: card |
|---|---|---|---|---|
| Home Desktop (16 GB, RTX 3070 8 GB) | 6 to 11.5 GB of 16 (MAJOR 2's table): fits | minibatch 2 000 fp32: 7.4 to 8.4 GB, does NOT fit; 1 000 fp32: 4.4 to 5.4 GB fits; 2 000 fp16: 4.7 to 5.7 GB fits | 7 to 12 GB commit (trainer 2 to 3, four loader processes 2 to 4, Windows 3 to 5), the rest page cache | soldier only, minibatch 1 024 fp32: 2.3 to 2.8 GB |
| Work Desktop (16 GB, RTX 3060 12 GB) | the same, plus whatever else it runs | 2 000 fp32 fits (7.4 to 8.4 of 12), if nothing else holds its card | the same | the same |
| Laptop (32 GB, RTX 1000 Ada 6 GB) | 6 to 11.5 GB of 32: fits easily | 500 fp32: 2.5 to 3.3 GB fits; 1 000 fp32: 4.1 to 4.9 GB, barely; 2 000: no | fits easily | fits |

The range and the duel cost the same per decision (the network is the same; the lists differ by
three entries); the range's env buffers are four times the duel's but tens of MB either way.

**Machine time for the design's six steps** (one PC unless said; implementation sessions not
counted; no reruns counted).

| step | low | central | high | what sets it |
|---|---|---|---|---|
| 1. The network | minutes | minutes | < 1 h | tests only |
| (2b' the first real run, proposed) | 1 h | 2 h | 3 h | MAJOR 4 |
| 2. Copying the bots | 2.5 h | 6 h | 15 h | recording 1 to 2.5 h; unpack 0 to 0.3 h; 2 to 6 copy runs of 0.25 to 1.5 h (MAJOR 9); the copied team's tournament 0.7 to 3 h (MAJOR 7) |
| 3. The trainer's toy tasks | 0.3 h | 1 h | 4 h | MINOR 4 |
| 4. Shooting range | 1 h | 6 h | 33 h | bank and yardsticks 0.5 to 2 h; 3 to 4 runs of 0.13 to 7.8 h (MAJOR 1) |
| 5. Duel | 2.7 h | 11.4 h | 180 h | two runs (memory on / off) of 1.3 to 90 h |
| 6. The experiment | 2.7 h of each of three PCs (the laptop 4 h) | 11.4 h (the laptop 17 h) | 180 h (the laptop 270 h) | six runs, two per PC (MAJOR 3) |
| ("earns it", rung 3's first stretch x 9) | 12 h of three PCs | ~20 h of three PCs | 42 h of three PCs | its length is not given (MAJOR 3 d) |

**Bottom line.** Rung 2's machine time is about **one day of one PC plus an afternoon of three**
at the low end, about **one day of one PC plus one to two days of three** centrally, and about
**ten days of one PC plus eleven of three (three weeks)** at the high end; a first self-written
trainer's reruns multiply it by two or three. What sets the spread is the number of decisions to
the pass mark (a factor of ten, unknowable before step 4) times the farm's rate in a training room
(a factor of four, measurable in the first week if MAJOR 4's run is made). Host memory is not a
constraint anywhere; the card is, per piece.

## Holds

- The stored decision: 62.7 KB recomputed from the spec's shapes (S = 6), 61.4 KB for the duel;
  8 000 decisions 0.50 GB, 32 000 2.0 GB.
- The records do not fit in any PC's memory (44 GB of living decisions), and the copy needs no full
  view (the records carry none); unpacking is cheap (5 to 15 min).
- Re-recording 600 door-team records in about an hour (67 min by the 1j walls), with BLAS pinned.
- The soldier's weights are about 6 MB (1.55 M x 4 B): the right figure for sharing an opponent.
- Acting on the card is right: the farm uses every core.
- Collector and learner in one process, synchronous, in rung 2: right, since learning is 0.1 to
  0.7 of collecting (MINOR 1); the boundary costs nothing.
- Reading the experiment at equal decisions makes unequal PCs fair for the result (not for the
  wall clock).
- Steps 1 and 3 and patches 2a to 2c need no farm and no game run; 2a's tests, the grading case,
  reduced-width toy tasks and the copy's memorisation test need no card.
- The rung-3 note (1.6 to 16 days at 720 / s) holds; if anything it leaves out the driver's and the
  learner's share (10 to 40 %).
- A spread-out run over a relayed link will not carry the observations (4.6), even compressed.

## Not checked

- Nothing was run: the training-room tick, the per-row observation cost in a small room, a farm
  worker's memory with spec v1, every card figure and the loader rates are estimates from measured
  neighbours and stated assumptions. The 1h note's own per-worker memory readings, if any, were not
  read.
- The laptop's farm speed and thermals; the Work Desktop's free disk and what else uses its
  card; the desktops' PCIe generation.
- The number of decisions to each pass mark: only the roadmap's borrowed figures exist.
- Whether the ship generator's second stage accepts one-room levels (MAJOR 4).
- The PPO and copying mathematics, and whether each measurement can fail: other lenses.
