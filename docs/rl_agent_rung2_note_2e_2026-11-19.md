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
