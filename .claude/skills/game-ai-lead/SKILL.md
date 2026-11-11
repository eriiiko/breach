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
| 2026-11-11 | **The ladder** (`docs/rl_agent_roadmap_2026-11-11.md`): 0 the design, 1 the door, 2 first light, 3 a fireteam that fights, 4 orders, 5 scale, 6 the commander, 7 unit classes, 8 senses, 9 other learners. Network sizes: about 1 million weights first, 1 to 3 million for the fireteam, 5 to 20 million at scale. |
| 2026-11-11 | **The residency patch P3a restarts when rung 0 is agreed** and runs beside rungs 1 and 2. |
| 2026-11-11 | **C4 on the continuous path** is Erik's own patch; the agent design assumes the verb. |

## 6. Open

- Rung 0, still to write and agree: the team display, the order, reward
  profile v1, what the value network sees, network v1. Then the critique
  passes, one critic at a time.
- Gameplay the design leans on, Erik's to order (design document section 11):
  a reason to walk (footsteps and spread by speed), the reserved verbs.
- The trainer's own design is rung 2 (recommended by the survey: PPO in the
  MAPPO arrangement, written by us against `BatchedEnv`).

## 7. Where things live

- **Issue:** #192. The pinned roadmap issue #46 carries the order across arcs.
- **Documents:** `docs/rl_agent_survey_2026-11-11.md` (what the env gives
  today; how six other systems were built; section B.9 says what was checked);
  `docs/research/rl_agent_survey_2026-11-11_raw/` (seven digests with source
  ledgers); `docs/rl_agent_roadmap_2026-11-11.md` (the ladder);
  `docs/rl_agent_design_2026-11-11.md` (rung 0, section by section). Earlier:
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
   of rung 1 changes no behaviour of the game and merges on green; anything
   that changes how a match plays is a HUMAN-TEST.
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
