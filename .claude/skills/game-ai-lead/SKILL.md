---
name: game-ai-lead
description: Claude's standing role as lead designer of Breach's game AI -- the learned agents and everything that trains, measures and commands them (reinforcement learning first; also imitation of the bots, a later commander, unit classes, Tsetlin machines). Holds the mandate and its limits, how a design session with Erik runs, the rules of the build kept from the 2026-11-11 survey, what is decided (skeleton, action space, observation, sight), the ladder and which rung is live, and where everything lives. Load at the start of ANY session that touches the agent's observation or action spaces, the reward, the trainer, a training run, the commander, unit classes for agents, or issue #192, before reading the code.
---

# Game AI lead -- continuity skill

**Status: PROVISIONAL** (created 2026-11-11, the first agent design session).
Every session that uses it amends it before merging: what moved, what was
decided, what was learned. A step is a MUST only with its incident beside it.

The name: reinforcement learning is the spine, but the role also covers
imitation of the bots, a commander that may be a language model, scripted
opponents and Tsetlin machines, so it is "game AI", not "RL". The sibling
skill `rl-ready-lead` is the game side (levelgen, match, bots, replay, the
residency arc's log); this one is the agents.

## 1. The mandate (Erik, 2026-11-11)

"You are the lead RL designer now, and we talk to each other to get all the
right features implemented." "We aim at making the highest quality possible:
we will check the latest literature and research, and we want to use state of
the art solutions." "We are starting from the ground and aim for the stars."

- **Claude decides:** the technical shape of everything learned: the specs'
  encodings, the network, the trainer, the experiments and their order inside
  a rung, how patches are cut, what goes in this skill.
- **Erik decides, always:** what the game is (which verbs exist, what a
  soldier may know, how obedient is right, anything by eye), the gameplay
  patches the design leans on, budget (effort level, agent models, parallel
  sessions), and every question put to him as "your call".
- **Any earlier decision may be overturned** where it is worth it (Erik: "none
  of the attempts I've done on designing this has been of this caliber").
  Record a ruling scoped to its case; a musing is not an order.

## 2. The goal, in Erik's words

- The game's architecture "is built with this training in mind, as the main
  goal even". Real time is the game now; the turn-based mode is legacy.
- Long term: many unit classes ("aliens, octopusses, animals, different kinds
  of marines with different armor, robots: think of different classes almost
  as in an RPG"), and many kinds of AI on one interface ("RL, Tsetlin machines,
  commander + agents"). Different architectures and action spaces will be
  tried side by side.
- "A great foundation now, then a quite quick path to first training."
- An ambitious roadmap in rungs spanning months, "so we always have a plan on
  what to do next". Other ideas may still be pursued along the way.
- He intends to work on Breach for many years.

## 3. How a design session with Erik runs (learned 2026-11-11)

1. **Consult egregore and the docs first.** July's survey and roadmap had
   already answered a question he raised as open, and neither of us remembered.
2. **One question per message; everything else as statements he can veto,**
   each with its reason. Put a recommendation beside the question.
3. **Plain words.** He has a physics and maths background and not the RL
   vocabulary: explain a term in a clause the first time. Do not coin words
   ("one rule per match, the same for every seat" needed a second message).
4. **He reads everything.** Keep a message compact; show a proposal as a small
   table; put the detail in the design document.
5. **Record his answer in the same turn:** the design document's section
   becomes AGREED with his words and the date, and memory gets the ruling.
6. **Gameplay he mentions is listed, not built** (design document section 11).
7. **The parity question to ask of any feature:** do the bots have it? "Let's
   give them everything that the bots have."
8. A GitHub write (an issue) needs his explicit word in the session.

## 4. The rules of the build

The fourteen rules in `docs/rl_agent_roadmap_2026-11-11.md` section 2, each
from the survey. The ones most often needed:

- Units are a list read by attention; the map is a small window.
- The soldier points at its target; the game turns the body; the soldier
  pulls the trigger. Several choices at once, masks on each.
- The policy sees only what its team may know; a training-only value network
  sees everything.
- Interfaces are named, versioned specs chosen by the run file.
- No game number in an interface: ratios of table values. Hit points, armour
  and weapon values are untested and will change.
- Network numbers never enter the synced game: integer verbs in, every match
  replays.
- Every claim becomes an experiment that can fail, above all where the survey
  found nobody to lean on (friendly fire, "whom is my team-mate aiming at", a
  cone, orders to a shooter soldier, classes with different verbs).
- Every rung opens with a short lesson document.

## 5. Decided (with dates)

| Date | Decision |
|---|---|
| 2026-11-11 | **The skeleton.** One network for all marines, same weights, about six decisions a second. A soldier sees itself, a list of things, a small map window and an order. A training-only value network sees the whole match. Four choices at once: feet, aim, trigger, one hand action. The order input exists from the first trained network; the bots' team layer is the first commander. Class is a type on every list entry plus a verb mask. |
| 2026-11-11 | **The soldier pulls its own trigger,** shown the bots' on-target prediction. |
| 2026-11-11 | **Action spec v1** ("your action space is blessed"): a decision every 4 ticks; feet = stand or 16 compass directions at walk / half / full; aim = an enemy the team sees, a last-seen record, 16 directions or "where I walk"; trigger; one hand verb, pointed at (reload, melee, shove, use, throw per grenade kind, plant); a throw goes where the soldier aims, with a reach choice. Flashlight left out until light is game state (#81). |
| 2026-11-11 | **Places reserved** for verbs Erik wants and has not built: stance (stand / crouch / prone) as a fifth choice; revive, drag, pick up / put down, a general use as hand verbs acting on the aim target. |
| 2026-11-11 | **Everything the bots have** is visible to a learned soldier, an enemy's health and focus-fire level included. |
| 2026-11-11 | **Observation spec v1** ("I bless all of it"): the soldier's own vector; the list (team-mates, enemies the team sees, last-seen records, grenades and charges, doors); a two-scale map window; physics planes = every field that can hurt, blind, push or drown a unit, as ratios of that unit's environment profile; two sight planes (tiles the known enemies watch, tiles that see my target); a compass to the order's place. |
| 2026-11-11 | **Sight: all-round for now,** walls block, shared across the team, the same rule for both teams of a match. Cones are wanted later ("just to have agents playing good is more important"; reducing information "needs much more computation"). No match played before 2026-11-11 has to be preserved. |
| 2026-11-11 | **The team display** ("I think we got it now"): read from the game's state, never a controller's memory, so a bot, a learned soldier and a human look alike. Per team-mate: its aim, feet, a throw being wound up, a reload, its order, the fire lanes between us. Per enemy: how many team-mates aim and fire at it, recent damage, shared target, team-mates a frag would catch. The bots' counts for the team. A fire-lanes plane. The radio is perfect for now. |
| 2026-11-11 | **The order** ("agree with all your suggestions"): a place on the ship's room graph, never a tile. What (none, go to, hold, clear, fall back to), where (a room or an opening), until; plus an optional priority target. Route, position, cover and fire are the soldier's. The bots' team layer is the first commander; orders are information until rung 4 makes them binding; the format is a named spec ("rooms"). Erik's aim: "not a copy of StarCraft, but distributed intelligence: the agents are their own AIs, the commander is its own". His commander idea: a graph network over the room graph (rung 6). He hopes the generation graph is used directly or as a seed, and sees a later project: level generation with AI input on what makes a strong defensive layout. |
| 2026-11-11 | **Three sides matter to Erik:** zombies bring the chaos that makes matches winnable by wit (ambush the team busy with zombies; make the horde work for you). Design document 4a (proposed): each enemy entry shows its side and whom it aims at; a zombie shows dormant / hunting / drawn by noise. |
| 2026-11-11 | **Erik delegated the rest of rung 0** (reward, the value network's input, network v1) to Claude, to be brought back as proposals he can veto. |
| 2026-11-11 | **The ladder** (`docs/rl_agent_roadmap_2026-11-11.md`): 0 the design, 1 the door, 2 first light, 3 a fireteam that fights, 4 orders, 5 scale, 6 the commander, 7 unit classes, 8 senses, 9 other learners. Network sizes: about 1 million weights first, 1 to 3 million for the fireteam, 5 to 20 million at scale. |
| 2026-11-11 | **The residency patch P3a restarts when rung 0 is agreed** and runs beside rungs 1 and 2. |
| 2026-11-11 | **C4 on the continuous path** is Erik's own patch; the agent design assumes the verb. It landed as #193 (`plant_charge_intent`); the plant is a hand verb whose one argument is a timer (four choices). |
| 2026-11-11 | **"yes i agree with it all"** (Erik, to the second session's summary, before the critique): 4a (sides); reward profile v1 (outcome +10 / −10 / clock −3; a "balance" of health in bars read off the state; team share 0.3 to 0.8; friendly fire the shooter's alone; an order term reserved at weight 0); the value network's full view; network v1 (about 1.55 M weights; list attention, two small map readers, a GRU, pointer heads); an enemy charge's timer visible when seen; four plant timers. |
| 2026-11-11 | **Erik's note for rung 2:** the trainer keeps collecting and learning behind a boundary, with a weights version on every batch, so one run can later be spread over his three PCs on Tailscale; until then the three PCs run three experiments at once and share saved networks as opponents. (Roadmap, rung 2.) |
| 2026-11-11 | **The critique** (four lenses, one Opus critic at a time; design document section 12): no blocker, 23 major points, all fixed on paper as version 2. Eight of the fixes changed sections Erik had agreed (section 12.3): sight as a match setting; records that outlive an unseen death; the physics planes' true scales; "whom it aims at" from the facing; a throw's aim held through the wind-up; "the play" dropped; masks from the real gates; the map shown through walls. **Erik: "yes i agree".** Rung 1 was re-cut into ten patches, 1a to 1j (roadmap); **"i dont need to look at 1d, i bless it all ready now"**. |

| 2026-11-14 | **The decision rate stays at 4 ticks; rung 1 is closed** (design document 13.1). 1j was bots against the same bots behind the door, no network; the door team won 0.27 to 0.37 against a 0.40 mark, and the cost is reaction, not the choices. Erik: "making a decision 6 times per second is quite a lot. I don't think we shall increase it"; "if we focus too much on acting every tick, we miss what we're trying to build here: intelligent agents that can learn how to do smart moves"; "I'd say we've already passed this test actually". The free bots are not slowed; from rung 3 a learned team is measured against the free bots (the gate) and `bots_door` (beside it). Sixteen directions stay (the lead's; 8 "looks a little not so nice"). |
| 2026-11-14 | **Rung 2 starts by copying the bots** (design document 13.3; Erik: "I lean on copying first too"), then the shooting range and the duel from reward, the duel run from the copy and from scratch. **No training run before the residency arc's P6 is through** (the unit code is moving). |

## 6. Open

- **Rung 1 is CLOSED (2026-11-14). Rung 2's design is at VERSION 2**
  (`docs/rl_agent_rung2_design_2026-11-14.md`): Erik read version 1 on
  2026-11-15 ("very impressed", no veto); four critics the same day (1
  blocker, 33 major, 50 minor; verdicts in
  `docs/rl_agent_rung2_design_2026-11-14_critique/`, the log in the
  design's section 15); every blocker and major folded in. Its section 13
  lists what changed in what he read (the gate has numbers and a cap on
  running out the clock; no floor on the copy's hand; a first real run
  early; eleven patches 2a to 2k, four touching a game file; the times are
  days, not hours); his by-eye half of the gate is agreed as a first try
  (design 7.4). **The patches are being run from
  `docs/rl_agent_rung2_plan_2026-11-15.md`** (the orchestrator's contract:
  one Opus implementer at a time, the lead reads 2a and 2b line by line,
  a checkpoint on #195 after every merge); which patch is live is in
  memory (`project_rl_agent_design_session.md`, top). His idea of a feet choice that names a spot is parked as an
  experiment after the duel learns (design 13.4, protocol amended by the
  critique). The lesson document ("the trainer, line by line") is the
  lead's, written from the code of patches 2b and 2d. The two questions the
  unit design (`docs/unit_design_2026-11-14.md` 3 and 9) hands this role
  (hooks shown; fixed slot blocks for statuses and items) are parked in
  that document's section 11 with their trigger.
- **The farm is slower than the roadmap planned with:** about 720 soldier
  decisions a second on the medium ship with spec v1 (measured 2026-11-13),
  against 1 000 planned; a fireteam run there is 1.6 to 16 days. Rung 3
  starts on the small ship; re-measure it first.
- **Rung 0 is CLOSED** (Erik, 2026-11-11: "yes i agree" to the eight changes of
  design document 12.3; merged the same evening). **Rung 1 is live:** ten
  patches, 1a to 1j (roadmap). Erik blessed 1d's step (ii) in advance: it
  merges on the tournament's numbers, without his look. The residency patch
  P3a may restart, with the design document's section 9 as its input.
- **Rung 1 is run by an orchestrator session** (Opus), from the handoff plan
  `docs/rl_agent_rung1_plan_2026-11-11.md`, tracked on **#195**. See section
  11 below. The plan holds the full notes for 1a, 1b and 1c; the orchestrator
  writes the others' notes at each boundary; 1g gets a spec document first
  and is tiered to Fable in small pieces (Erik, 2026-11-11).
- Game findings from the critique, Erik's to rule on (design document section
  11): spawned zombies have 100 hit points and turned ones 400; a stunned
  soldier can reload and open doors; the bots' adjacent-enemy check ignores
  walls.
- Gameplay the design leans on, Erik's to order (design document section 11):
  a reason to walk (footsteps and spread by speed), the reserved verbs.
- The trainer's own design is rung 2 (recommended by the survey: PPO in the
  MAPPO arrangement, written by us against `BatchedEnv`).

## 7. Where things live

- **Issues:** #192 (the design, and the ladder's later rungs); **#195 (rung 1,
  the orchestrator's checkpoints)**. The pinned roadmap issue #46 carries the order across arcs.
  #194 (milestone 12) holds Erik's idea of levels shaped by AI input on what
  makes a strong defensive layout; it needs the room graph saved with the
  level and, in one form, trained agents as the judge.
- **Documents:** `docs/rl_agent_survey_2026-11-11.md` (what the env gives
  today; how six other systems were built; section B.9 says what was checked);
  `docs/research/rl_agent_survey_2026-11-11_raw/` (seven digests with source
  ledgers); `docs/rl_agent_roadmap_2026-11-11.md` (the ladder);
  `docs/rl_agent_design_2026-11-11.md` (rung 0, section by section; version
  2, agreed; its critique in `docs/rl_agent_design_2026-11-11_critique/`);
  `docs/rl_agent_rung1_plan_2026-11-11.md` (rung 1's handoff plan);
  `docs/rl_lesson_0_learning_from_reward_2026-11-11.md` (rung 0's lesson:
  PPO in plain words; rung 2's lesson goes through the trainer);
  `docs/rl_agent_rung2_design_2026-11-14.md` (rung 2: the trainer, the
  copying step, the tasks, the patches). Earlier:
  `docs/research/rl_litsearch_2026-07-20.md`, `docs/roadmap_2026-07-30_rl_push.md`.
- **Code seams:** `src/rl_env.py` (`BatchedEnv`), `src/control_policy.py`
  (the `policy` controller, `VERB_RULES`), `src/match_replay.py`
  (`INTENT_KINDS`), `src/simulation/sensor_accessor.py` (the observation
  door), `src/simulation/vision.py` (sight), `src/simulation/turning.py`
  (`set_target`, `predict`), `src/control_bots.py` (`TeamBrain`, the position
  features), `tools/bot_tournament.py`, `tools/run_match.py`, `tools/rl_smoke.py`.
- **Memory:** `project_rl_agent_design_session.md`.
- **Related skills:** `rl-ready-lead` (the env and the residency log),
  `git-workflow`, `autonomous-patch-workflow`.

## 8. Procedure for a session

1. `git fetch origin && git status`; read sections 5 and 6 here, then the
   design document's status marks, then memory.
2. Say which rung and which section the session serves.
3. Design work is language first: a section of the design document, proposed,
   then agreed. Nothing is built from a section that is not AGREED.
4. Build per `autonomous-patch-workflow` on a branch with its worktree. A patch
   of rung 1 changes no behaviour of the game and merges on green; a patch
   that changes how a match plays merges on green too, unless Erik asked to
   play it first (CLAUDE.md iron rule, 2026-11-13).
5. Close: record rulings in the design document and memory, update #192, amend
   this skill.

## 9. Research method that worked (2026-11-11)

Seven research agents on Opus in parallel, one topic each, about ten minutes
and 1.5 million tokens in all. What made it work: each brief named the game,
the eight design questions, the earlier survey to build on and not repeat, and
one output file; each agent wrote its digest from its first source on (nothing
is lost at a usage wall), tagged every claim READ or RECALL, and closed with a
source ledger (FULL / PART / ABS / RECALL / NF). The lead then re-read two
load-bearing claims at the source before using them. This is research only:
implementation agents and critics stay one at a time.

## 10. Run log

- 2026-11-11 -- the opening session (branch `rl-agent-design`, documents only;
  issue #192): the survey, the roadmap, rung 0 started; skeleton, trigger,
  action spec v1, observation spec v1 and sight agreed. Learned: (1) egregore
  first (section 3.1); (2) the env's observation was a placeholder (a 17 x 17
  tile window against a 90-tile carbine, no line of sight, six numbers); a
  sight model existed and only the bots used it; (3) a plant verb with a timer
  existed for the turn-based mode only, so memory's "no plant verb ever
  existed" was wrong: read the code before repeating a memory; (4) in the game
  today the three speeds differ in speed only (no footstep noise, spread
  independent of speed), so a designed choice can be empty until the game
  gives it a meaning: check what each choice does in the code before putting
  it in a spec; (5) Erik blesses a compact table quickly and stalls on a
  coined sentence.
- 2026-11-11, evening -- the second session (branch `192-rl-agent-design`,
  documents only): sections 6, 7, 8 written, #193 folded in, Erik's yes, then
  the critique. What the critique taught, for the next design: (1) a design
  section that says "as the bots do it" or "the game's own gate" must be
  checked against the code line by line before it is shown to Erik: the
  masks, the fire lanes, the last-seen records and the meaning of the physics
  planes were all paraphrases that the code contradicted; (2) "the unit's
  full health" and "1.0 = it hurts" were claims about game numbers that the
  game does not yet make true: ask of every ratio which table value divides
  it and what reads that value; (3) section 9 must list the state a design
  READS, not only what it adds, or the device port cuts the observation off;
  (4) four lenses that worked for an interface design: how a learner cheats
  the reward; what a soldier may know; is every named thing real and is
  anything built twice; does it fit the trainer. About 5 minutes and 200 to
  270 thousand tokens per critic on Opus. Each critic wrote its own verdict
  file from its first finding, and was told not to read the others';
  (5) put Erik's summary and question BEFORE the critique, but expect to come
  back with the fixes that touch what he agreed, as one short list.
- 2026-11-13 -- rung 1, 1j (the bots through the door): the pass mark FAILED
  on the held-out seeds (win rate of the decided duels 0.27 to 0.37 in the
  four cells; friendly fire within the lead's 1.2 x + 0.2 rule), so it goes to
  a design session. Learned: (1) a proxy in place of the sim (capture, forward
  every other read and write) turned "the bots decide exactly as they do" into
  a digest test, with no edit of the bots; (2) a 40 s recorded match met no
  masked latch, so the gate-5 break on the mask passed the fixture: a gate that
  must fail needs a scene where its event happens -- scan seeds for it with the
  tournament's own counts, then pin that seed; (3) `bot_tournament` workers do
  not pin BLAS threads: 12 of them commit ~2 GB each and starve a 16 GB machine
  (an e2e test then fails on OpenBLAS); export OPENBLAS_NUM_THREADS=1 before a
  long run; (4) stopping a backgrounded bash loop does not stop it on Windows --
  kill the loop's own bash process tree, or each kill advances it to the next
  run.

- 2026-11-14 -- the design session after rung 1 (branch `192-rung2-cadence`,
  documents only; egregore did not connect). Learned: (1) Erik read "the bots
  through the door won 36 %" as "trained agents already beat the bots 36 % of
  the time": a result must say WHO played WHOM and that no network exists,
  before any number; "the door" is our word, not his; (2) recompute a handed-on
  number before building on it: the 1j note's "0.48 at one decision per tick"
  was the drop per 2 ticks taken as the drop per tick (the line gives 0.42
  +- 0.06), which changes how much a faster rate would buy; (3) a pass mark
  set before any measurement is a guess: say so when it is set, so that
  missing it by 4 points is read as a finding and not as a failure; (4) the
  question that settled it was a game question (how fast should a marine
  react), which Erik answered from what the game is for, without the options
  table.
- 2026-11-14, later -- rung 2's design written (same branch, documents only).
  Learned: (1) my own "about ten minutes to record again" of that morning
  was an unchecked number (the 1j note's walls give about an hour): the
  recompute rule holds for my own figures within one session; (2) size the
  data before designing around it: one decision is 63 KB with the full view,
  so memory, not theory, sets the batch on a 16 GB PC, the copy's records are
  45 GB unpacked, and a run spread over three PCs has to move observations
  that Tailscale's relay will not carry; (3) the planning figures in a
  roadmap go stale when the interface they were set before is built: read
  the latest measured farm number, not the roadmap's.

- 2026-11-15 -- rung 2's design read by Erik and critiqued (same branch,
  documents only). Learned: (1) the four lenses for a TRAINER design: is
  every named game thing real (code read line by line on main); would a
  trainer built exactly so learn correctly; can each measure and experiment
  fail and mean what it says; does it fit the cards, the PCs and the patch
  sizes. 280 to 420 thousand tokens and 10 to 17 minutes each on Opus, one
  at a time, each verdict committed before the next spawned; (2) the
  measures were the weakest part and I had not seen it: one held-back set
  used for the curve, for choosing and for the exam; a gate with no number
  of duels and no rule on a fail; "decided" hiding the draws. Write the
  three room lists, the reading rule and the fail table BEFORE the method;
  (3) a floor or a default that sounds harmless must be multiplied out per
  decision per match (2 % over seven hand options at six decisions a second
  is a C4 at the soldier's feet in two duels of three); (4) a test task
  must be checked against the trainer's own limits (a memory task of 20
  decisions against stretches of 16 cannot pass); (5) count LEARNING
  soldiers when estimating speed: a small room with one learner is not
  faster per learning decision than a ship with eight; (6) "touches no game
  file" was wrong for four patches: check each patch against what the door
  code allows before writing the table; (7) Erik's side question (name a
  spot to walk to) got a parked experiment with its reasoning in the design
  (13.4), not a design change: say what the literature does and does not
  settle, and what the design already has of the idea.
- **2026-11-15, patch 2a (network + choosing).** One Opus agent, about
  2 h 15 min (48 min of it the game's suite), 290 thousand tokens, green
  first time. What the next patch should know: (1) the agent's tests feed
  MADE-UP inputs, so the lead ran a few REAL decisions through the
  full-width networks at the boundary (a one-worker `BatchedEnv`, 30
  decisions): do this for every patch whose tests cannot see the game;
  (2) the line-by-line read found what no test could: count planes fed raw
  beside squashed inputs. Read the inputs' ranges, not only the shapes;
  (3) on Windows `CUDA_VISIBLE_DEVICES=""` does NOT hide the card; use
  `-1`; (4) two of the agent's own on-purpose faults were no-ops its first
  tests passed (a constant shift before a LayerNorm; a filter doubled by
  the maths): "break it once" must be a fault that changes the output;
  (5) small fixes after the report are cheaper done by the lead in the
  finished agent's worktree than by resuming the agent.
- **2026-11-15, patch 2b (trainer core).** One Opus agent, about 6.5 h and
  450 thousand tokens, most of the hours spent waiting on task runs on the
  processor; it built everything and stopped on three tasks that did not
  pass, changing nothing (right). The lead then spent about four hours of
  runs deciding them. What the next patch should know: (1) a made-up task
  written only as a sentence in a note gets a first guess at its size; SIZE
  IT IN THE NOTE (how many samples of each right answer a learning step
  sees: about 13 learnt in 40 to 90 rounds, 3.5 did not in 500) and say the
  farm; (2) a task must test its trap and nothing else: three forms of one
  task failed on a trait of the trainer (the sooner-paying answer crowds
  the other out) that its trap had nothing to do with; a mask or a single
  answer keeps that out; (3) when a task fails, COUNT ITS PARTS APART
  before changing anything (one run with the verb's and the argument's
  accuracy printed separately told in 12 minutes what three reruns had
  not); (4) "each trap must fail its task" needs a margin, not only a
  FAIL: one trap ended at 0.75 of the way under a 0.8 line; (5) measure a
  budget at the budget (the step size falls over the planned rounds, so a
  run planned for 400 rounds is not a run planned for 150); (6) run the
  task gate as three processes of four threads; the whole gate is about
  1.5 h, so launch it once, after one seed of every task has passed; (7)
  the brief's rule "never an easier task" made the permission system
  refuse the agent's own shrink of a task, which was the right outcome:
  such a change is the lead's, with the measurement beside it, and Erik is
  told.
- **2026-11-17, patch 2c stage A (the first real run, built and measured
  small).** One Opus agent. What the next patch should know: (1) the card
  SPILLS for the training Python on the Home Desktop (9 216 MiB allocated
  on the 8 GB card without an error); `agents/run.py` caps the process at
  the free memory, so every run on the card goes through it; (2) a scratch
  script that starts a farm needs `if __name__ == "__main__":` (spawn
  re-imports it; without the guard it hung silently, no workers, no
  output); (3) the env's `info["level"]` after a step is already the NEXT
  match's: a per-level count must take the level from the view before the
  step, and only a stand-in that varies the level can test that; (4) an
  untrained soldier kills itself (grenades, fire) in a large share of the
  range's matches: the outcome group is mostly the -10 of elimination at
  the start; (5) at 4 workers x 16 matches, full width on the card, a round
  of 2 048 decisions took 12 s (env step 67 %, answers 15 %, learning 16 %):
  the farm is the bottleneck, as the design said.

## 11. Running rung 1 (for the orchestrator session)

Erik handed rung 1's implementation to an orchestrator on 2026-11-11. If you
are that session:

1. You act inside this role's mandate (section 1), with one limit: you
   execute the agreed design, you do not change it. A design question that
   the design document does not answer, and that is more than an encoding
   detail, stops the patch and goes to Erik or to a design session.
2. Load `autonomous-patch-workflow` and `git-workflow` as well. Read
   `docs/rl_agent_rung1_plan_2026-11-11.md` in full: it is your contract
   (how you work, the ten patches with implementer and gate, the notes for
   1a to 1c, what each later note must settle, how to live beside the
   residency arc).
3. Checkpoints go on #195 after every merge, in plain words for Erik, and
   into memory (`project_rl_agent_design_session.md`).
4. Add one dated line to the run log below when a patch teaches something
   the next one should know.
- **2026-11-17, patch 2c stage B (the first real run; report
  `docs/rl_agent_rung2_report_2c_2026-11-17.md`).** The soldier cleared all
  six hand rooms after 81 minutes (1.9 M decisions) at 391 learning
  decisions a second, with the design's numbers untouched. What the next
  run should know: (1) WATCH a trained match before believing a curve: the
  corner room was "solved" by a frag thrown through two walls (a grenade's
  flight asks no wall; Erik's to rule on, #215), found only by driving the
  saved weights in-process and printing the events; (2) a task whose
  target always stands on the same tile is answered by a memorised walk,
  and its matches shrink to one or two seconds, after which the round
  measures nothing: the seeded rooms (2g) come before the range is a gate;
  (3) the range does not shape the trigger or the hand (a pull at a wall
  and a plant at the feet cost nothing in a two-second match); (4) 128
  matches buy no speed on 16 workers and fill the computer's memory (15.3
  of 15.9 GB); 8 workers give three quarters of 16; (5) start a long run
  detached (`Start-Process`), with a background waiter on its pid, so an
  editor restart cannot kill it; (6) keep a run's folder outside the scratch
  folder (`C:/Users/steen/breach_runs/`).
- 2026-11-17 -- 2d.1 (the memory path) built on `192-2d-trainer-rest`:
  stretches, the stored memory, the live check's second clause (catches all
  four memory faults; the first clause barely does on a fresh network), the
  value-first rounds, `good_start` (gate held; neither no value-first rounds
  nor ten times the variety wore the start down). FINDING for the lead
  before 2d.2: `recall12` is not learnt with the design's numbers (even a
  gap of 4 reaches 0.11 of the way in 300 rounds; the cue's mark on the
  fresh memory is small and fades 0.6 x a decision), so its three tests are
  skipped by name; decide the cell's start / step size / farm before any
  run with the memory on. Bit-identity without memory is a machine-keyed
  hash in `agents/tests/test_memory.py`.
- 2026-11-17 -- 2d.1b (why the memory did not learn): the heads read the
  cell's output raw, about 8 x smaller than the join; a LayerNorm between
  (`memory_norm`) makes a supervised gap-12 fit learn (0.29 -> 0.99) and
  the cue answered at once pass by reward in 87 rounds. Chrono start kept.
  Still open: the tasks' farm gives recall12 only 8.5 teaching answers a
  round (flat after 370 rounds); the farm is the lead's call. Measure a
  memory question SUPERVISED first (minutes) before a reward run (hours);
  and never `git stash pop` in a worktree: the stash is shared by all of
  them (it popped another branch's WIP here; undone).
- 2026-11-17 -- 2d.2 (saving and resuming) built on `192-2d2-saving`:
  `agents/saved.py`, `--resume`, the batch header, the training manifest,
  `SNAPSHOT_PATHS` + `agents/` `tools/`. Learned: (1) a PyTorch optimiser
  keeps the state tensors it is handed and steps them in place, so a
  restore must copy them (the determinism test caught two restored
  trainers sharing Adam moments); (2) a kill between a round's line and
  its save leaves stale curve lines, so a resume moves lines past the saved
  round to `curve_discarded.jsonl`; (3) the record says "v1" for both
  widths, so the saved file and the export also carry the widths.
