# RL agent, rung 2, patch 2c: the first real run (the note)

2026-11-17, the lead's. The design is `docs/rl_agent_rung2_design_2026-11-14.md`
("the design": 0 step 3, 4.4, 5.1, 5.3, 7.1, 9, 10) over
`docs/rl_agent_design_2026-11-11.md` (6.2, 6.9); the plan is
`docs/rl_agent_rung2_plan_2026-11-15.md` (section 5's row for 2c, and what 2b
handed on). Where this note and the design disagree, the design wins and the
note is corrected.

**What 2c is.** One soldier, from scratch, without memory, learns to shoot a
standing marine in three hand-made rooms, for one to three hours on the
whole farm. It is a MEASUREMENT, not a pass or a fail: its product is a
report with numbers that replace the design's estimates (design 9), and a
curve that either rises or does not. Nothing in rung 2 has yet been run at
full width on the game; this is where a slow farm, a full card or a dead
curve shows, six patches earlier than it otherwise would.

**In 2c:** the three rooms and the tool that writes them; the "range" reward
row; the training file on disk; the run script and its per-round line; a
first small curve plot; the report.

**Not in 2c:** saving and resuming, the hand-over header, the manifest, the
memory path (2d); the seeded room generator and the three room lists (2g);
the run-file key that skips the targets' observations (2i); any sweep.

## 1. The PC and when the run may start

- **The Home Desktop** (RTX 3070, 8 GB; 16 GB; 16 threads). The plan sends
  long runs here until egregore has moved off the Work Desktop, and Erik gave
  this PC to the session for 2026-11-17.
- **Two stages, and the second waits for the lead's word.** Stage A (build):
  the code, the tests, and smoke runs on at most 4 workers. It may run beside
  one building agent. Stage B (the run): 16 workers take every thread, so it
  starts only when the lead says the PC is free, and nothing else builds,
  tests or trains until it ends (design 9: a running farm owns its PC).
- **Set up, checked once in stage A and written into the report:** the ML
  stack at the shared version (`ml-stack` skill: `ml_stack.py check` with
  this PC's `data` Python; never an install by hand); the processor build of
  this worktree; the card's spill into the computer's memory. The spill is a
  driver setting per program ("CUDA sysmem fallback policy"); say whether it
  is off for the training Python. If it cannot be read or set from a script,
  cap the process with `torch.cuda.set_per_process_memory_fraction` so that
  running out is an error, say that this is what was done, and leave the
  driver setting to Erik as one line in the report. The links between the
  PCs (design 4.6) are NOT measured here: one PC runs alone.

## 2. The three rooms

Hand-made, one of each shape of design 5.1, written through the level writer
with a room graph by a new tool, **`tools/gen_training_rooms.py`**. This is
the file patch 2g grows into the seeded generator: 2c gives it the writer
functions (a room, a door, a cover crate, the two spawn rooms, the match
file) and one table `HAND_ROOMS` of three fixed layouts; 2g adds the seeded
shapes on the same functions and never a second tool.

All at 0.333 m tiles, airtight, one learner (a rifleman, team 0) and one
standing marine (team 1), each always on the same tile of its spawn room.
The match file of each: `sight = "all_round"`, no zombies, a 45 s clock,
elimination ends it.

| name | shape | size | what it asks of the soldier |
|---|---|---|---|
| `range_hall` | one open space (two graph rooms joined by an open edge) | 12 m x 8 m | the target is in plain view about 9 m away: point, pull the trigger. Two cover crates off the line, so the windows are not empty |
| `range_door` | two rooms joined by a door | 7 m x 7 m each | the target is not in view from the spawn tile: walk through the door, then shoot |
| `range_corner` | three rooms in an L | 6 m x 6 m each | two doors and a turn; the target is in the far room, off the axis of the second door |

The lead's choices, with their reasons: the hall is first because it is the
shortest path from a fresh network to a paid kill (the curve should move
here first, and if it does not move here the fault is the trainer, not the
task); the door room is the smallest case of "the reward is behind a wall";
the corner is the same with a turn, and will most likely not be solved in
this run, which is itself a number worth having. Distances are far inside
the carbine's reach, so missing is the soldier's doing and not the
weapon's.

- Each room is also written with the two spawn rooms swapped (design 5.1,
  "sides"): six levels, in equal shares.
- The rooms are written OUTSIDE the repository (`--out`, default a folder
  beside the run's output) and made again from the tool; the run file's
  `levels_dir` points there. Deterministic: two calls write the same bytes.
- Reused, never rebuilt: `level_lib` (the one writer), the level lint, the
  airtightness check and the walkable-path check the ship generator uses,
  the game's cover objects (not the ship generator's crates, which burn).

## 3. The reward: the "range" row

`reward_v1.PROFILES` gains one row, **"range"** (design 6.9: "a target loses
health; the clock"), the same sizes as "fireteam" where it has a term:

| term | weight | why it is in |
|---|---|---|
| `win` | 10 | the room is cleared |
| `eliminated` | −10 | the soldier killed itself with a grenade or a charge |
| `clock` | −3 | the 45 s ran out |
| `marine_bar` | 1 | health taken off the target, and the soldier's own lost |
| `marine_alive` | 0.5 | the kill, and its own death |
| `zombie_bar`, `ff_bar`, `ff_kill` | 0 | no zombie and no team-mate exists; written as 0 so that a later change of the room cannot pay through them unnoticed |

In a room with one soldier, one target and no zombie this pays exactly what
"fireteam" pays, tick for tick; a test holds that. The lead's reasoning for
not adding anything (a pay for facing the target, for walking toward it):
the first run must show what the plain table does, and a helper added before
that hides the baseline it would be judged against. The trainer's group
weights stay rung 2's (outcome 1, own 1, the rest 0).

This edits one game file (`src/simulation/reward_v1.py`, one table row). It
is host Python outside the tick; `tools/twin_debt.py` and
`tests/test_class_p_level.py` say whether anything more is owed (expected:
nothing).

## 4. The run file and the training file

**The run file** (the environment's, read by `rl_env.load_run`), in full:

```toml
[run]
seed = 1
n_envs = 64
decimation = 4
levels = ["range_hall", "range_hall_swapped", "range_door",
          "range_door_swapped", "range_corner", "range_corner_swapped"]
policy = ["policy", "idle"]
levels_dir = '<the rooms folder>'
[obs]
spec = "v1"
full_view = true
[action]
spec = "v1"
[reward]
profile = "range"
```

(The level names are what the tool writes; if the env shares levels over
matches differently from "in equal shares", the report says how.)

**The training file** (the trainer's, new: `agents/training_file.py`), one
TOML read into `Settings` and the network's spec. Every key is a field that
exists; an unknown key, a missing `planned_rounds` or a wrong type is refused
by name; a key left out takes `Settings`' default, so the file below is the
design's table (4.4) and nothing else is typed anywhere:

```toml
[network]
spec = "v1"          # full width
memory = false
[train]
planned_rounds = 1200
decisions = 128
gamma = 0.99
lam = 0.95
clip = 0.2
passes = 4
max_distance = 0.02
piece = 2048
lump = 512
step_size = 3e-4
variety = 0.01
max_grad_norm = 0.5
[weights]
outcome = 1.0
own = 1.0
team = 0.0
friendly_fire = 0.0
order = 0.0
[stop]
wall_hours = 2.5     # the run ends at planned_rounds or here, whichever first
```

`planned_rounds` is set in stage A from the measured speed so that the plan
is about 2.5 hours (the step size falls to zero over the PLANNED rounds, so
the plan must be the run: plan lesson of 2b); the number above is a place
holder for that measurement.

Both files are copied into the run's output folder beside the curve.

## 5. The run script and the per-round line

`python -m agents.run --run <run.toml> --train <train.toml> --out <folder
outside the checkout> [--workers 16] [--device cuda]`, built on the existing
`Trainer` (the script joins, it does not re-implement; `agents/real_run.py`
stays the small smoke).

Each round appends ONE line of JSON to `curve.jsonl` (design 7.1), with at
least:

- round, learning decisions so far, wall seconds so far;
- matches ended this round; of them: won (room cleared), the clock ran out,
  the soldier died (by its own hand: the only way here), per level;
- reward per ended match, and per group; **the reward groups' sizes** (the
  mean absolute pay per decision of each group: 2b's hand-on);
- **the hand verbs' counts**, per verb, and the trigger's share pulled (2b's
  hand-on: a verb dying out is the crowding-out trait);
- rounds fired per match; the share of decisions with the aim on the target;
- per value output of a group that varies: the share of the return it
  explains;
- variety per head; the distance each pass moved the policy, and whether the
  early stop cut the round; the clipped share; both losses; the scale's
  spread;
- the live check's two numbers, and the count of masked choices sent on a
  driven row (must be 0; a non-zero stops the run with the line written);
- **seconds, split**: the environment's step (wait for the workers), the
  network's answers while collecting, the copy into the batch, grading, the
  learning passes; and the card's peak memory this round.

A small `agents/plot_curve.py` draws the report's figures from that file
(matplotlib, already in the stack; publication quality: labelled axes with
units, one legend, no title-case clutter). 2d's curves script extends this
one, never a second.

At the end the script writes both networks' weights to the output folder
with `torch.save`, marked in the file's name as NOT the saved-run format
(that is 2d's): so that a later patch can look at what this run learnt.

## 6. What the report measures

`docs/rl_agent_rung2_report_2c_2026-11-17.md`, plain words, with the figures
beside it. Before any number it says who played whom: ONE untrained soldier
against ONE marine that stands still and never shoots.

1. **Speed:** learning decisions a second over the run; the split of section
   5 as a share of a round; the same for a short run (five rounds) at 128
   matches; and at 8 workers (five rounds), to see what the 16 buy.
2. **The card:** peak memory per round, and whether the spill is off.
3. **The computer's memory** with the farm up.
4. **Does the curve rise:** rooms cleared per level over decisions; reward
   per match; the first round at which the hall is cleared in more than half
   its matches, if any.
5. **The hand verbs and the trigger over the run** (one figure); whether any
   verb died out; self-harm counted on its own.
6. **The reward groups' sizes**, and what share of the return each value
   output explains.
7. **The corrected rows of the design's cost table (9):** the estimate, the
   measurement, and what it does to the times of the range, the duel and the
   experiment. The design's table is amended in this patch, with the old
   figure kept beside the new and marked.
8. **What looked wrong,** even if unexplained. A curve that does not rise is
   reported as that, with the numbers that say where it stalls (does the aim
   reach the target; is the trigger pulled; do shots land); it is not tuned
   in this patch. Design 7.5's fail table says what each means.

## 7. Tests (properties; each names what it protects)

In `tests/` (the game's suite; no PyTorch):
- the rooms: every one loads through the level loader, passes the lint and
  the airtightness check, has both spawn rooms joined by a walkable path, a
  room graph with the shape its name says, the match file's sight, clock and
  no zombies; the swapped twin differs only in the spawn rooms; two writes
  are byte-equal. Breaks if a writer function drops the graph, the door or
  the match line.
- the "range" row: in a one-soldier, one-target, no-zombie room it pays
  what "fireteam" pays on every tick of a played match; its three zero terms
  are zero. Breaks if a weight drifts from the table it was cut from.

In `agents/tests` (the trainer's own):
- the training file: the design's file gives `Settings`' defaults; each
  unknown key, missing `planned_rounds` and wrong type is refused by name; a
  weights table that does not name every group is refused.
- the per-round line: on the stand-in environment, a round's line has every
  field of section 5, the seconds' parts sum to the round's wall within a
  tolerance, the verbs' counts sum to the decisions that counted a hand.
- a smoke of `agents.run` on the real rooms: 2 rounds, 4 matches, 2 workers,
  reduced width, on the processor: the line is written, the live check is
  silent, masked choices 0.

Every new test is broken once on purpose with a fault that changes the
output; the report lists each and whether it went red.

## 8. Systems

- **Uses:** the Batched environment (the trainer touches the game only
  through it); the Trainer, Choosing, Stored decision and Network spec rows;
  the Reward profile row (a reward is a named table); the level data layer
  (`level_lib`, one writer); the level lint; Match config (`match.toml`
  through `level_lib.write_match`); the controller registry (`idle`).
- **Creates:**
  - *Training rooms* (`tools/gen_training_rooms.py`): THE writer of rung 2's
    small levels, through `level_lib`; a new room shape is a function or a
    table row here, never a second tool. One index line; its constraints in
    `.claude/rules/levels.md`.
  - *Training file* (`agents/training_file.py`): the only way a run's
    trainer numbers reach `Settings` from disk; an unknown key is an error.
    Constraints in `.claude/rules/agents.md`, under the Trainer line (no new
    index line unless the file is used outside the trainer).
