# Rung 2, patch 2e: copying the bots (the lead's note)

2026-11-19. The note the plan (`docs/rl_agent_rung2_plan_2026-11-15.md`,
section 2 and "what each note must hold") asks for before patch 2e. The
design is `docs/rl_agent_rung2_design_2026-11-14.md` section 3; where this
note is more specific, the note decides; where it would contradict the
design, the patch stops and reports.

## 1. What 2e is, and is not

2e builds the path from the bots' recorded decisions to a soldier network
that imitates them: the packed-record loader, the copying trainer, and the
measures of design 3.3 that need no played match. It touches no game file.

Not in 2e: a learned team in a played match (the `net` controller and the
tournament are 2f, so the row "does it play?" of 3.3 is 2f's and 2h's);
recording at scale (2h); the remedy that lets the bots label the network's
own situations (reserved in the design).

## 2. The data

- **The existing records** are in `C:/tmp/breach_1j/records` (two ships,
  4.0 GB packed), written by 1j through `tools/bot_tournament.py --records`.
  They were made on 2026-11-13. Since then the game changed (poison law,
  doors closed at start, body temperature), and the observation may have
  changed shape.
- **First step of the patch: a census of the records** (in the as-built):
  the file format, what one file holds, the number of matches, files and
  living-soldier decisions per ship, and whether every array's shape and
  the spec names in the records fit today's `agents/sizes.py` and
  observation / action spec v1.
- **If they fit,** 2e runs on them as the plan says.
- **If they do not fit,** nothing is converted or padded. The patch records
  a small fresh set with the same tool into `C:/tmp/breach_2e/records`
  (outside the checkout): 200 matches on the two ships of the existing
  records, seeds 1 to 100 each, the door bots as they are recorded by 1j's
  procedure, workers held to one thread. Say which case applied. Either
  way the records of 2e are development data: 2h records again at scale
  after this, and its copy is the one that plays.
- **The loader refuses a record that does not fit by name** (spec names and
  shapes), the way `agents/saved.py` refuses a foreign saved run. This is
  the guard for the day the observation changes again.

## 3. The loader

- A module `agents/records.py` (reading) with the loader beside it. It
  reads PACKED files; nothing is unpacked to disk.
- Loader processes (start with 4) each take whole match files from a
  shuffled list, unpack them and put them into a shuffle buffer of about
  30 matches; the trainer is dealt batches of single decisions from the
  buffer. One pass = every training decision once. The order is a function
  of one seed, so a run can be repeated.
- For the memory copy the loader has a second mode: whole matches of one
  soldier in order from its first decision, cut into the stretches of the
  trainer (`Learner._stretches`' rule: 16 decisions, cut at a match start),
  the memory carried from one stretch to the next, never a window started
  in mid-match from an empty memory (design 3.2).
- Only decisions of living soldiers are dealt (the records' own flag).
- **Measured before any long run:** decisions per second dealt, with the
  network not attached, and the share of time the card waits for data in a
  short real run. The design's estimate is about a minute a pass over 1.1
  million decisions; report the real number.

## 4. The split

By (level, seed), never by file: every file of one match (both teams, with
and without the horde) falls on the same side. Held back: every (level,
seed) with `seed % 5 == 0`, 20 % of the matches of each ship. The split is
one function with a test (no (level, seed) on both sides; breaks when the
split is made by file).

## 5. What is learned, and how

- A module `agents/copying.py`. It uses the trainer's own choosing module
  (`agents/choosing.py`, `evaluate`) under the records' own masks and the
  records' argument mask, counting only the heads that counted. Never a
  second way of computing a probability.
- The loss: minus the log-probability of the bot's recorded choice, summed
  over the counted heads, averaged over the batch. The feet alone are
  trained toward "the bot's choice with 98 %, the other LEGAL options
  sharing 2 %" (design 3.2); hand, argument, aim and trigger as they are.
- Starting numbers: Adam, step size 3e-4 falling in a straight line to
  zero over the planned passes; batch 2 048 decisions worked in lumps of
  512 (the trainer's numbers, for the 8 GB card); gradient norm cut at 0.5;
  planned passes 10, with the held-back agreement measured after every
  pass and the best pass's weights kept. These are settings in the
  training file under their own table, refused when unknown.
- The network is the soldier's network of `agents/spec.py` at full width,
  without memory first, then with memory. The value network is not
  trained here.
- The result is saved through `agents/saved.py` only (a soldier's export
  that says what it was started from: the records' folder, their census
  hash, the split, the pass). No `torch.save` of its own.

## 6. The measures (design 3.3), with their numbers

Reported per head and, where it says so, per option, in the as-built and
as one JSON file per run:

| row of 3.3 | how, in 2e |
|---|---|
| Is the data path sound? | a network trained on 1 000 decisions drawn across at least 50 matches reaches at least 99 % agreement on every counted head. A TEST of the patch (reduced width allowed if it still passes), and the gate |
| Does it predict the bots? | agreement on ALL held-back matches, against "always the most common legal option" and "repeat the previous choice" |
| Does it predict the changes? | the same on the decisions where the bot's choice differs from its previous one |
| Does it lean on its own past? | one more run with the previous-choice inputs blanked at the network's input |
| How close are the misses? | mean angle error of feet and of aim; for the pointer, against "aim at the nearest enemy" |
| All heads together? | every counted head right at once, with the network's own aim and hand chosen first |
| Lossy labels? | agreement split by the records' loss marks |
| What does the observation not explain? | agreement on the TRAINING matches at the end, and once at double width |
| Does the memory help? | memory on and off, each held-back match run from its start; three runs each, or labelled "indicative" |
| Does it play? | NOT in 2e (2f, 2h) |

Every number carries how many matches and decisions it was measured on.

## 7. Tests (properties)

The split; the loader deals every living training decision exactly once a
pass and no held-back one; the same seed gives the same order; a record
that does not fit is refused by name; the copy's loss is the choosing
module's probability (a hand-worked case of two decisions); the feet's
floor is spread over the legal options only; an uncounted head adds
nothing; the memory mode never starts a stretch in mid-match from an empty
memory; the 1 000-decision test. Each new gate broken once on purpose.

## 8. The machine

Another agent builds and runs GPU gates on this PC. The patch's tests run
on the processor. The short real runs on the card (the speed measurement,
the 1 000-decision test) are a few minutes each. The LONG runs of section
6 (ten passes at full width, the double-width run, the memory runs) are
NOT started by the implementer: the patch ends with the commands ready and
a time estimate from the speed measurement, and the lead starts them when
the card is free.

## 9. Systems

Uses: `agents/choosing.py` (the one probability), `agents/network.py` /
`spec.py` / `sizes.py`, `agents/saved.py` (the one writer), the trainer's
stretch rule, `tools/bot_tournament.py --records` (the one recorder).

New, each with its one-line rule (index line in `CLAUDE.md`, constraints
in `.claude/rules/agents.md`):
- **Record reader and loader** (`agents/records.py`): THE one reader of
  the bots' records; packed in, decisions out; refuses a record that does
  not fit by name; the split is by level and seed.
- **Copying** (`agents/copying.py`): learning from records uses the
  choosing module and the records' own masks; the records are data, re-made
  when the bots or the observation change.

## As built (2e, 2026-11-19)

Branch `192-2e-copying` (off main `461d150f`). Not merged, not pushed.

### The census, and which data case applied: THE RECORDS FIT

`python -m agents.copying census --records C:/tmp/breach_1j/records --out
C:/tmp/breach_2e/census_1j.json` (69 s, 4 processes; census hash
`bb480f532cc3d1630d4a5b9885b87af7af81f5d078a4d7c7dca6b1624f0dde2e`).

- **Format.** One `.npz` (compressed) per match and door team, written by
  `control_bots_door._write_record` (record format 1), read through the
  game's `control_bots_door.read_record`. A file holds, per decision of the
  team (every 4 ticks), every roster slot's served observation (`vector`,
  `list`, `list_mask`, `list_uid`, the four window arrays), the row flags
  `valid` / `in_match`, the four masks, `arg_mask`, the five integers sent
  (`actions`, int16), the translation's loss bits (`loss`, uint16), the
  fireteam rewards and `rows_ended`; `meta` (JSON) names the level, seed,
  match spec, spec names, grenade rows, S / E / K, build ids and the
  replay file. One file is one match seen by one door team.
- **Counts.** 600 files = 600 matches (no match has two door teams), 4.2 GB.

  | ship | files (matches) | seeds | living decisions train / held | matches train / held |
  |---|---|---|---|---|
  | `ship_demo` | 300 | 1-20, 101-130 (50) | 371 720 / 81 764 | 240 / 60 |
  | `ship_freighter_standard_s28_medium` | 300 | same 50 | 531 248 / 122 526 | 240 / 60 |
  | **both** | 600 | | **902 968 / 204 290** (1 107 258) | 480 / 120 |

  Per (level, seed) six matches: the door team on either side against
  `bots`, with and without the horde (`__z0` / `__z4`), and against `idle`
  on either side with the horde (200 of the 600 files, 406 922 of the
  living decisions: the copy also learns from matches against a team that
  does nothing). Held back: seeds 5, 10, 15, 20, 105 ... 130 = 10 of 50 per
  ship, 20 % of the matches. Every soldier's living decisions are the first
  of its match (no gaps: `not_prefix` 0 on both ships).
- **Fit.** All 600 fit today's layout: observation and action spec `v1`,
  the six grenade rows by name, every array's trailing shape and number
  format for S = 6 (E = 39), and every kind value of a living decision
  inside the network's rows. So **case "fit"**: nothing was recorded again.
  Caution, not a refusal: the records were played on the game of
  2026-11-13 (builds `beatrice-b1g-892-gb055a61b-dirty` and `-893`),
  before the poison law, the doors closed at start and body temperature;
  the observation's shape and names did not change, its meaning in those
  matches did (development data, as the note says; 2h records again).

### What was built

- **`agents/records.py`, the record reader and loader.** `check_record`
  refuses by name (`RecordRefused`) a record whose spec names, grenade rows,
  array shapes or formats, or kind values do not fit (never converted);
  `census`; `side(level, seed)` (held iff `seed % 5 == 0`) and `split`;
  `unpack` (one file's living decisions, slot-major, refused when not a
  prefix); `DecisionDealer` (processes unpack whole files in the pass's
  seeded order, at most 2 x workers files ahead; a buffer of 30 files;
  batches drawn at random from it; near-equal batches of at most 2 048 so
  a pass deals every living decision once, never a short last batch);
  `StretchDealer` (128 lanes x 16: a lane runs one soldier's match from its
  first decision, stretches cut by `agents.learner.stretches`, the next
  soldier drawn at random from the buffer when a match ends); no torch in
  the module.
- **`agents/copying.py`, the copy.** `copy_loss` (choosing's `evaluate`
  under the records' masks, the stored argument mask held equal to the
  game's rule or `CopyRefused`; the feet's 2 % over the other LEGAL options);
  `Copier` (Adam, step size falling in a straight line to zero over the
  planned passes x decisions, gradient cut 0.5, lumps of 512 summed to the
  batch's mean; with memory each lane's memory carried detached from
  stretch to stretch, zeroed only at a soldier's first decision);
  `measure` / `Tally` (every row of the note's table but "does it play?",
  each with files and decisions); `run` (census, split checked, passes with
  the held-back agreement after each, the best pass exported); the command
  line `python -m agents.copying census | speed | train`.
- **Small additions on the way:** `choosing.mode` (the most likely option by
  the same walk, free or given the earlier choices: the measures'
  argmax, never a second walk); `settings.CopySettings` (the copy's numbers,
  the trainer's defaults read from `Settings`' fields, not retyped);
  `training_file.parse_copy` / `load_copy` (a copying run's file is
  `[network]` + `[copy]`, refused by name otherwise; width `"double"`
  added); `spec.double`; `saved.export_soldier(..., specs=, started_from=)`
  (a soldier made without an environment names the records' spec names, no
  reward profile, and what it was started from).
- **The stretch rule moved, unchanged:** `Learner._stretches`' body is now
  the module function `agents.learner.stretches(first, stretch)`, which
  `Learner._stretches` calls and the loader uses; `test_memory.py` and
  `test_saved.py` green after the move (45 passed).
- **The `_segs` gap closed:** `Learner.__init__` sets `self._segs = None`
  (the stretches of the batch last learnt; cut afresh every round), it is in
  `saved.NOT_SAVED["Learner"]` with why, and
  `test_every_attribute_is_saved_or_named_as_not_saved` now inspects a fresh
  trainer AND one that has run a round with the memory on. Nothing else of
  the trainer changed.

Where the note was silent: the held-back agreement that picks the best
pass is "every counted head right at once, teacher-forced"; "repeat the
previous choice" falls back to the most common legal option at a soldier's
first decision and counts a previous choice now illegal as a miss (on the
changes it scores 0 by construction); "most common" counts the aim's list
positions as options; per-option agreement lumps the aim's list entries
into one "entry"; the aim's angle is a compass token's or the bearing of
the pointed entry from its unsquashed offset, "walk" and "keep" left out;
the memory dealer's last steps of a pass hold fewer decisions as the lanes
run dry (their loss is the mean over what they hold).

### Speed (measured)

- **Loader, no network** (`python -m agents.copying speed`, 4 processes,
  90 s, training side): 7 466 decisions a second dealt singly, 6 066 in
  stretches; the first batch after 7.7 / 8.8 s (the processes' start and
  the buffer). A pass over the 902 968 training decisions: about 2 min (the
  design's estimate was about 1 min).
- **Short real runs on the card** (RTX 3070, full width, one pass over the
  first 60 training files = 89 103 decisions, 6 held files = 8 738
  decisions; card at 315 MiB before): without memory 44 steps, 12.3 s
  learning (about 7 200 decisions a second), 13.6 s waiting for data
  (52 % of the pass; about 8 s of it the start), measuring 5.0 s. With
  memory 120 steps (the lanes running dry over only 60 files), 33.7 s
  learning (about 2 640 a second), 18.1 s waiting (35 %), measuring 7.1 s.
  So without memory the card is about as fast as the 4 loader processes;
  more processes would shorten a pass when the PC is free.

### The 1 000-decision test (design 3.3, "is the data path sound?")

1 000 living decisions, 20 from each of 50 training files spread over both
ships (`agents/tests/_thousand.py`); heads counted on 1 000 / 994 / 1 000 /
991 / 3 of them (feet, aim, trigger, hand, argument). Trained whole each
step with the copy's numbers (constant step size 3e-4):

| where | width | steps | seconds | agreement feet / aim / trigger / hand / arg |
|---|---|---|---|---|
| card | full | 400 | 44 | 1.000 / 0.993 / 1.000 / 1.000 / 1.000 |
| card | reduced | 1 725 | 132 | 0.999 / 0.991 / 1.000 / 1.000 / 1.000 |
| processor | reduced | 1 500 (stopped by me) | 940 | 0.965 / 0.982 / 1.000 / 1.000 / 1.000 |

The test (`test_a_thousand_real_decisions_are_learnt_to_99_percent`) runs
the reduced width on the processor with a budget of 4 000 steps under the
`tasks` marker (about 20 min); it is skipped by name (`THOUSAND`) where the
records are not (set `BREACH_RECORDS`).

### Tests and their breaks

New: `agents/tests/test_records.py` (7), `agents/tests/test_copying.py`
(10, the 1 000-decision one under `tasks`), the strengthened
`test_saved.py` attribute test; helpers `_records.py` (record files in the
game's format, a mark per decision) and `_thousand.py`. Each new gate broken
once on purpose (edit, run, restore):

| break | caught by |
|---|---|
| the split made by file (the door team decides the side) | the split test |
| a dealt decision left in the pool | the deal-once test |
| the `valid` flag ignored | the deal-once test |
| an unseeded order | the same-seed test |
| a lane given another soldier's middle | the stretch-mode test |
| the fit check skipped | the refusal test |
| the feet's probability from an unmasked softmax of the copy's own | the hand-worked loss test |
| the feet's floor over every option | the floor test |
| the stored argument mask not compared | the argument-mask test |
| the lanes' memory reset at every stretch | the carried-memory test |
| `mode` ignoring the given choices | the mode test |
| the export dropping `started_from` | the export test |
| a `[copy]` key ignored | the training-file test |
| every head counted in the measures | the measures test (NOT caught at first: the test asked the function under test which heads counted; it now asks the game's rule) |
| the masks shifted against their decision | the 1 000-decision test (a masked choice refused at once) |
| `_segs` left out of `NOT_SAVED` | the attribute test (and the old fresh-only test could not see `_segs` at all) |

### The long runs: commands and cost (NOT started)

From the branch's checkout (`.claude/worktrees/192-2e-copying`, or a
training checkout at a tag of it), with the training files already in
`C:/tmp/breach_2e/files/`, the card free, one run at a time:

```
cd C:/Users/steen/projects/breach/.claude/worktrees/192-2e-copying
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=C:/Users/steen/anaconda3/envs/data/python.exe
R=C:/tmp/breach_1j/records; F=C:/tmp/breach_2e/files; O=C:/tmp/breach_2e/runs
for s in 0 1 2; do $PY -m agents.copying train --records $R --training-file $F/copy_full_nomem_s$s.toml --out $O/full_nomem_s$s --device cuda; done
$PY -m agents.copying train --records $R --training-file $F/copy_full_nomem_blankprev.toml --out $O/full_nomem_blankprev --device cuda
$PY -m agents.copying train --records $R --training-file $F/copy_double_nomem.toml --out $O/double_nomem --device cuda
for s in 0 1 2; do $PY -m agents.copying train --records $R --training-file $F/copy_full_mem_s$s.toml --out $O/full_mem_s$s --device cuda; done
```

Each writes `census.json`, `curve.jsonl` (a line a pass: steps, loss,
seconds learning / waiting / measuring, held-back agreement per head),
`copy.pt` (the best pass's soldier through `saved.export_soldier`) and
`measures.json` (every measure on the held-back and the training records,
with files and decisions, the option counts).

Estimates from the measurements above (10 passes each; census 70 s;
held-back measure after every pass):

| run | a pass (learning / waiting) | measure a pass | total |
|---|---|---|---|
| full width, no memory (x 3 seeds) | about 2.5 to 4 min | about 2 min | about 1 h each |
| previous-choice inputs blanked | same | same | about 1 h |
| double width, no memory (lump 256) | about 8 min (estimate: about 3 x the compute, not measured) | about 4 min | about 2 h |
| full width, memory (x 3 seeds) | about 8.5 min | about 3 min | about 2 h each |

About 12 hours for all eight, the card alone (the loader's 4 processes use
the processor). The double width's card memory was not measured; at lump
256 it should stay under the full width's 2 048-decision step (583 MiB was
the peak of a 1 000-decision full-width step).

### Findings (reported, not fixed)

1. The records predate three game changes (see the census); they fit by
   name and shape only. The copy measured on them is development data.
2. A third of the records (200 files, 37 % of the living decisions) are
   matches against `idle`; whether the copy should learn from them is the
   lead's call (they are in both sides of the split).
3. The loader is slower than the design's estimate (about 2 min a pass, not
   1) and, without memory, as fast as the card: about half of a pass waits
   for data at 4 processes. With the PC free, 8 processes would be the
   first thing to try (`[copy] loader_workers`).
4. On the processor the copy is slow to fit even 1 000 decisions at the
   reduced width (1 725 steps on the card's arithmetic): the 1 000-decision
   test costs about 20 minutes of the agents suite.
5. The memory dealer's pass ends with lanes running dry: steps with fewer
   decisions at a pass's end (120 steps for 89 103 decisions over 60
   files; over all 480 training files the tail is a small share).
