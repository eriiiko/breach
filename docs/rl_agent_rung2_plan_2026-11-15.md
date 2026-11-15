# RL agent, rung 2: the plan for the orchestrator

2026-11-15. The contract for whoever runs rung 2's patches (this time the
lead's own session; a later session picks up from here). The design is
`docs/rl_agent_rung2_design_2026-11-14.md`, version 2 ("the design"); this
plan adds only how the work is run. Where the two disagree, the design wins
and this plan is corrected.

**Erik's word to start (2026-11-15):** "i think we can start, if u are
ready", then: "sorry, we wait, pls set an alarm and start after it merges".
So patch 2a starts when P6.4b (the last patch of the residency arc's P6,
#29) is merged into main, without asking again.

## 1. How the orchestrator works

1. **Load** the skills `game-ai-lead`, `autonomous-patch-workflow` and
   `git-workflow`. Read this plan, then the design's sections named by the
   patch in hand. `git fetch origin && git status` first.
2. **One patch at a time, one implementer at a time across all sessions,**
   each a fresh agent in its own worktree `.claude/worktrees/192-2x-<slug>`
   on branch `192-2x-<slug>`, cut from a freshly pulled main. The
   orchestrator holds the plan and short summaries; it does not implement.
3. **Before a patch starts it has a note.** 2a's is section 4. For the
   others the orchestrator writes a one-page note from the design and from
   section 5, commits it on the patch's branch, and only then briefs the
   implementer.
4. **The brief** follows `reference_residency_patch_brief_template.md`
   (memory): role and scope; where to work; read first; what to build; the
   gates; the house rules; the report. The house rules that matter here:
   - this machine's breach Python (`docs/dev_setup.md`);
   - the game's suite is `pytest tests -q`; the agents' own tests are
     `pytest agents/tests -q` and are NOT part of the game's suite, so the
     game's suite never needs PyTorch;
   - never `git add -A`; stage explicit paths; commit at every coherent step;
   - tests assert properties, from the note's test list; a failing old test
     is a finding to report, never a target;
   - the canonical-systems table is checked before anything new is built; a
     new system's rule is written where main's `CLAUDE.md` says rules go at
     the time of the patch (one index line there; its constraints in a path
     rule if main has path rules by then), from the design's section 14;
   - a patch that touches a game file follows that file's own gates
     (`tools/twin_debt.py` and `tests/test_class_p_level.py` say whether a
     CUDA stamp is needed);
   - a game finding outside the patch is one line in the report, not a fix.
5. **Green gate, then merge** `--no-ff` from the main checkout, push, delete
   the branch and the worktree, without asking. Exceptions are marked in
   section 2.
6. **The lead reviews the first instance hard.** 2a and 2b are the code the
   whole rung stands on, and their faults are the kind a green suite does
   not show. Before each of them merges, the lead reads the choosing module
   and the grading and update code line by line against the design.
7. **After each merge:** a checkpoint in plain words for Erik; memory
   (`project_rl_agent_design_session.md`); one dated line in
   `game-ai-lead`'s run log if something was learned. The checkpoint is
   posted as a comment on #195 (Erik, 2026-11-15: "yes post the checkpoints
   on #195"; the first, for the design, is issuecomment-6889058559).
8. **When to stop and ask.** A question about what the game is, what a
   soldier may know, or anything by eye goes to Erik, one question at a
   time. A gate that fails twice, or a note that would contradict the
   design, stops the patch: commit, report, wait. The design's fail table
   (7.5) says what each failed measure means.

## 2. The patches

| patch | implementer | gate | Erik |
|---|---|---|---|
| 2a network, choosing | Opus | the note's tests green on the processor; the game's suite untouched and green; the lead's review | sees the printed table of parts and weights in the checkpoint |
| 2b trainer core | Opus | the hand-worked grading case; the made-up tasks without memory solved at reduced width over fixed seeds; the live check trips when broken on purpose; the lead's review | |
| 2c first real run | Opus (a small patch: the hand-made rooms, the run and training files, the report) | a report with measured numbers; not a pass / fail | sees the curve and the numbers. The design's cost table (9) is corrected from them |
| 2d trainer's rest | Opus | resume reproduces counters and weights; the refusal of a foreign batch; the run check refuses a stray file in `agents/`; the two memory tasks (12 passes; 24 fails at 16 and passes at 32); the "good start" task | |
| the lesson | the lead | | reads it |
| 2e copying | Opus | the 1 000-decision test; the measures of design 3.3 reported on the existing records | |
| 2f door driver, `net`, `random`, tournament, replay | Opus | 1j's door tests unchanged and green; the 1j pins equal; a `net` match records, replays and verifies its checkpoints | watches the copied team (a look, not a gate) |
| 2g rooms, lists, yardsticks | Opus | generator properties (airtight, reachable, both sides, three shapes, no gate room equal to a training room); the yardsticks measured; the marks written down by their rules | |
| 2h records again, final copy | Opus | the copy's bar (design 3.3) | |
| 2i range | Opus | the range's mark on the gate rooms | sees a replay |
| 2j memory, duel | Opus | the gate's rule on the VALIDATION rooms ("solved", design 7.3) | |
| 2k experiment, gate | Opus runs; the lead reads | the gate (design 7.4) | **watches eight replays before the number; his verdict is half the gate** |

Fable for a patch only on Erik's word about budget. If 2b's review finds a
fault of judgement that Opus repeats, that is the moment to ask.

## 3. Machines

Erik, 2026-11-15: "yes we can give it to training - we need to move the
egregore to either the home pc or the lenovo - we make the decision before
training starts. (i suspect home desktop since it is also always on). Home
desktop should probably also be set up for training - in case i need work
desktop - so, we will train a lot in the months to come, i say we set up
both."

- **Both desktops are set up for training.** The work desktop is the one
  given to it; the home desktop is the second.
- **Egregore moves off the work desktop before the first long run there.**
  Where to is Erik's decision, made before training starts. The move is its
  own session in the egregore project, not a patch of this arc. Until it is
  done, long runs go on the home desktop.
- **"Set up" means, per PC,** checked once and written into the patch's
  note: the ML stack at the shared version (`ml-stack` skill; both desktops
  are ticked); a checkout with its own processor build (a training checkout
  at a tag when the network is kept, git-workflow section 1); the card's
  spill into the computer's memory switched off for the training Python
  (design 4.4); `rl_smoke.py` on a training room, read in learning decisions
  a second; the links between the PCs measured (design 4.6). This is done
  with 2c for the PC that runs 2c, and for the other before 2i.
- **A running farm owns its PC** (design 9): no suite, build or tournament
  beside it. The orchestrator keeps a run calendar in memory: which PC is
  busy with what, since when.
- **The laptop** joins for the experiment (2k) at the earliest, after it is
  on the ML stack.
- A session on the home desktop cannot start a run on the work desktop.
  How runs are started there (a session on that PC, or a remote start) is
  settled with Erik before the first run there.

## 4. The note for patch 2a: the network and the choosing module

**Design sections:** rung 2 design 1, 2, 8 (the parts about the network);
design 2 (the five heads, their order, the masks), 7 (the value network and
the full view), 8 (the network: 8.1 sizes, 8.2 which heads count, 8.3 the
memory and the normalising layer, 8.6 the test list); the critique's lens 2
"Holds" list and its minors 4, 6, 7, 14.

**What is built**, all under a new top-level `agents/`:

1. **The package.** `agents/` with its own `tests/` folder. It may import
   the game's Python (`simulation.*` from `src/`, found the way the tools
   find it); the game never imports it.
2. **Sizes from the game's own tables,** for a given team size S: one small
   module that asks `obs_spec_v1` and `action_spec_v1` for every size the
   network needs (the vector, an entry, the planes, the heads, the full
   view). No size is a literal in the network.
3. **Network spec "v1": the soldier's network,** as design 8 describes it:
   the readers of the vector, the list (the same weights for every entry,
   attention, the padding masked), the two windows as served (16-bit floats
   and bytes in, converted inside); the five heads chosen in the design's
   order, each later head seeing the earlier choices; the aim as a pointer
   at list entries beside its fixed tokens; a running memory (GRU) that the
   spec can switch off. Layer normalisation only; no dropout; nothing that
   depends on the other rows of a batch.
4. **The value network,** reading the full view and giving five outputs in
   the environment's reward-group order. (The rescaling of its targets is
   patch 2b's.)
5. **A reduced-width setting** of the same code, for tests.
6. **The choosing module,** the one implementation of: the chances of every
   option under the masks; drawing the five choices in order (the
   argument's mask worked out from the drawn hand and aim); the probability
   of five GIVEN choices; which heads counted; each counted head's variety
   as a share of the most it could be (design 4.4). Its "which heads
   counted" and its argument mask are held EQUAL to the game's own
   functions (`action_spec_v1.heads_used`, `arg_mask`) by a test over drawn
   cases: on the card they must be computed for a whole batch, so the
   module carries a batched form of the game's rule, and the test is what
   keeps it from becoming a second rule. An optional small chance of a
   random legal choice per head, inside the module, default 0.
7. **The spec's name and what it fits:** a small record (network spec,
   observation spec, action spec, the number of grenade kinds and planes)
   and a check that refuses a mismatch. Files on disk are patch 2d's.
8. **A script that prints** the table of parts and weights, and (when run
   with a card) the peak card memory of one learning step at 512 and 2 048
   decisions, both networks, 32-bit and 16-bit.
9. **In the game's suite, one test:** a scan that nothing under `src/`,
   `renderer/`, `ui/` or `main.py` imports `agents`.

**The card is not used during this patch's run.** The tests run on the
processor. The card-memory numbers are taken by the orchestrator at the
patch's boundary, when no other session's checks hold the card.

**Tests, each a property** (the implementer may add; none is a snapshot):

| property | breaks when |
|---|---|
| the probability of a drawn decision, worked out again from the same inputs and the same choices, equals the one returned at drawing | acting and learning differ anywhere in the module |
| an option the masks forbid is never drawn, over many random masks; no head is ever without a legal option | a mask is ignored or misapplied |
| "which heads counted" and the argument's mask equal the game's own functions on drawn cases | the batched form drifts from the game's rule |
| a head that did not count adds nothing to the decision's probability or to the variety | an unused head is graded |
| shuffling the list's entries (their masks and the pointed-at entry moved with them) leaves the probability of pointing at that ENTRY unchanged | the pointer follows a place in the list |
| the same soldier's input padded to a longer list (padding masked) gives the same chances | the network depends on the list's length |
| one set of weights runs at S = 1, 3 and 6 | a size is baked in |
| a row's output does not change with what else is in the batch | a batch statistic crept in |
| with memory on, feeding a sequence step by step equals feeding it as one stretch from the same start; a zero start mid-stretch is honoured | the memory path differs between acting and replay |
| with memory off there is no state | the switch leaks |
| the windows are accepted in the served number formats | a silent conversion differs between paths |
| the weight count is printed and reported against the design's 1 to 2 million | (reported, not forced) |
| the spec check refuses a network made for another observation or action spec | a saved network could be loaded into the wrong layout |

**Not in 2a:** the trainer, the stored decision, grading, the update, the
value targets' rescaling, saving, the copy.

**Where the design is silent** (the layer sizes that give 1.55 and 1.1
million weights, the exact form of the attention, how earlier choices are
fed to later heads): design 8 first; then the simplest form that meets it;
each such choice one line in the report.

## 5. What each later note must settle

- **2b:** the stored decision's layout by name and number format; the exact
  bootstrap (design 4.3); PopArt's update order; the piece and lump loop;
  the stand-in environment's behaviour list checked against `rl_env.py` at
  main's head; the seeds and thresholds of each made-up task; how the live
  check is broken on purpose.
- **2c:** the PC; the three hand-made rooms; the run file and training file
  in full; what the report measures (design 10's row) and how the speed is
  split.
- **2d:** the saved file's fields; what a resume compares; the edit of
  `rl_env.py`'s run check and its own tests; the training manifest's fields.
- **2e:** the loader's processes and buffer; the split by level and seed on
  the existing records; every row of design 3.3 with its number of matches.
- **2f:** the driver's interface (what a subclass supplies); how 1j's tests
  prove the bots unchanged; the registry flag; the job element naming a
  network; the tournament's worker pinning; `--levels-dir`.
- **2g:** the generator's parameters; the three lists' seeds; the layout
  hash for "the same room"; the yardstick runs (matches, clock, sight).
- **2h:** matches per ship and per room, hours and disk; the bar's reading.
- **2i:** the run-file key's name and meaning; the validation environment;
  the sweep order (passes 2 / 4 / 8 first).
- **2j:** the stretch length; 0.99 against 0.995; what "solved on
  validation" triggers.
- **2k:** the budget B from 2j's run; the PCs' calendar; the replay rule for
  Erik's eight; the note's place for his verdict.

## 6. Systems

The design's section 14 holds the existing systems this rung must use and
the new ones with their draft rules. This plan adds none.
