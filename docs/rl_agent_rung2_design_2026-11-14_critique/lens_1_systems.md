# Rung 2 design, critique lens 1: systems (is it real, does the code do it, is anything built twice)

Critic: Claude Opus 5.5, 2026-11-15. Object: `docs/rl_agent_rung2_design_2026-11-14.md` at `4564afad` (branch `192-rung2-cadence`). Read beside it: `rl_agent_design_2026-11-11.md` (2, 6, 7, 8, 13), `rl_agent_roadmap_2026-11-11.md` (rung 2, section 5), `rl_agent_rung1_note_1j_2026-11-13.md`, `rl_agent_rung1_note_1i_2026-11-13.md` (the farm reading). Code read-only from the main checkout `c:\Users\steen\projects\breach` (on `main`): `src/rl_env.py`, `src/control_source.py`, `src/control_bots_door.py`, `src/simulation/action_spec_v1.py`, `src/simulation/obs_spec_v1.py`, `src/simulation/reward_v1.py`, `src/simulation/match.py`, `src/simulation/match_session.py`, `src/match_replay.py`, `level_loader.py`, `main.py`, `tools/bot_tournament.py`, `tools/rl_smoke.py`, `tools/run_match.py`, `tools/gen_ship.py`, `tests/test_bots_door.py`, `.gitignore`.

(Findings are appended as they are found; the verdict is written last.)

## Findings

### F1 (MAJOR). The training snapshot does not cover a new top-level `agents/` folder

**The design, 1 and 4.7:** "A top-level folder `agents/` in this repository ... so that one tag fixes the game and the trainer of a training run together (the Training snapshot rule)"; 14: "Training snapshot: tags, the clean-tree checks, the run manifest."

**The code.** `require_snapshot`'s clean-tree check refuses a modified TRACKED file anywhere, but an UNTRACKED file only under `SNAPSHOT_PATHS = ("src/", "levels/", "config.toml", "cpp/")` (`src/rl_env.py:1168`, `check_clean_tree` `:1237-1262`). An untracked module dropped into `agents/` (or `tools/`, where the room generator of 2e goes) passes the check. `check_module_paths` probes only where `simulation` and `breach_physics` were imported from (`:1273-1287`, `module_probe` `:1336-1343`); nothing checks that the `agents` package the trainer imported is this checkout's (a `sys.path` pointing at another worktree's `agents/` is invisible). The design's "one tag fixes the trainer" is therefore not what the door enforces.

**Why it matters.** The point of the tag rule is that a kept network can be traced to its code. With the trainer outside `SNAPSHOT_PATHS` and outside the probe, a kept network can come from trainer code that is in no commit.

**Smallest fix.** Say in 1 / 4.7 that 2b extends the door: `agents/` (and `tools/`, if the room generator runs inside a training run) joins `SNAPSHOT_PATHS`, and `require_snapshot` / the probe also checks the `agents` package's file is under the checkout. Both are edits of `src/rl_env.py`, so 2b's row in section 10 becomes "touches the game: yes, the snapshot door" (and it must then wait for P6 like the rest, which it already does).

### F2 (MAJOR). "Both files' sha256 go into the run manifest": the manifest cannot hold a second file, and adding a key breaks every resume

**The design, 1:** "the run file ... and a training file beside it ... Both files' sha256 go into the run manifest." 4.6: a batch header carries "the hashes of the run file and the training file".

**The code.** `BatchedEnv.__init__` writes the manifest itself with a fixed set of keywords (`run_sha256`, `build_id`, `backend`, `tag`; `src/rl_env.py:1571-1580`); `run_manifest_record` has no parameter for another file (`:1429-1443`, the only spare slot is `bank_sha256`, documented as P8a's). A trainer that writes its own key into `run_manifest.json` breaks the next resume: `check_run_manifest` compares the UNION of the old and new keys and refuses any difference (`:1467-1474`), and the env's new record will not carry the trainer's key.

**Why it matters.** Either the trainer cannot record the training file where the design says, or 2b edits `rl_env.py` (a game file; section 10 says 2b touches none).

**Smallest fix.** Either (a) a separate `training_manifest.json` in `out_dir`, written and checked by the trainer (a new system for section 14's "Saved networks" row), with the env's manifest untouched; or (b) a `training_sha256` keyword on `run_manifest_record` / `BatchedEnv` in 2b, and 2b's row says it touches `rl_env.py`. (a) keeps 2b off the game.

### F3 (MAJOR). The `net` controller: joining `DOOR_CONTROLLERS` is a game edit, and "built like the door bots" means copying a loop that exists only inside `BotsThroughDoor`

**The design, 7.2:** "A match controller `net`, built like the door bots ... Same observation door, same masks, same decoder. It lives in `agents/` and is registered by the tool that needs it; it joins the list of door drivers that get the bots' commander." Section 10, 2d: "touches the game? yes: one registry line, the tournament". Section 14: "the `net` controller is a registry row ... and gets the commander as a door driver."

**The code.**
- `DOOR_CONTROLLERS = frozenset({"policy"})` is a module constant of `src/control_source.py:316`, read by `create_match_controllers` (`:387-390`). Nothing registers into it at run time: `register_controller` takes only `nameable` (`:319-330`). "Joins the list" is therefore an edit of a game file, whichever tool registers the factory. The meaning is consistent (`:311-315`: "drive a team THROUGH THE DOOR ... WITH NO BRAIN OF THEIR OWN"; the door bots are kept out because they carry a brain), so `net` belongs there, but the patch must say it edits `control_source.py` (or makes the set extendable, e.g. `register_controller(..., door=True)`). It must: in the farm every `policy` team on a room-graph level already gets the commander (`control_source.py:384-390`), so a `net` team without it would be shown no orders in the tournament although it trained with them (spec v1's order columns, `obs_spec_v1.py:237-242`, the entry's `priority` flag).
- The thing `net` is to be "built like" is not a seam. The decision loop of the door bots -- observe through `rl_env.observe_v1` on `tick % D == 0`, fill `valid` / `in_match` (a copy of `EnvRunner.row_flags`), hold `list_uid`, `set_decisions`, issue through the inner `AgentPolicy` -- is the private body of `BotsThroughDoor._decide` / `handle_frame` (`src/control_bots_door.py:735-881`; the row flags re-derived at `:826-835`). A `net` in `agents/` would write that loop a second time, outside `src/`, where the 1j gate "the records are the env's" (`tests/test_bots_door.py`, note 1j's property table) does not reach it. That is the "parallel copy" the canonical-systems rule forbids.
- The registry's factory is `team -> ControlSource` (`create_controller` passes options only to `replay`, `control_source.py:343-351`). A `net` needs to know WHICH saved network; the design does not say how (one registered name per network, or a process-global set before `run_match` as `apply_sets` does, `tools/bot_tournament.py:319-333`). The tournament runs matches in `spawn` pool workers (`tools/bot_tournament.py:416-419`), so the registration (and the network choice) must happen at import of `bot_tournament` or inside `play`, never only in the parent.

**Why it matters.** 2d as written cannot be built without either a game edit it does not list or a copy of the door loop.

**Smallest fix.** In 7.2 / 10 / 14: 2d lifts the door bots' decision loop into one reusable driver in `src/control_bots_door.py` (a base whose subclass supplies only "choose five integers from this observation and these masks"; `BotsThroughDoor` becomes one subclass, `net` in `agents/` another, the existing door tests unchanged), adds `net` to `DOOR_CONTROLLERS` (or an extendable flag) in `control_source.py`, and says how the network is chosen per match (a job element read in `play`, like `--set`). 2d's "touches the game" cell then names `control_bots_door.py`, `control_source.py` and `tools/bot_tournament.py`.

### F4 (MAJOR). The "true sizes" hold for 6 roster places only; every rung-2 task has another list length and aim head, and the copy's data mixes them

**The design, 2:** "True sizes, now that rung 1 fixed them ... the list 39 entries of 67 numbers on a ship with 6 roster places ... the choices 49 (feet), 57 (aim), 2 (trigger), 13 (hand), 5 (argument)." "A checkpoint names the network spec, the observation spec and the action spec it was made for, and loading it into anything else is refused." 3.1: the re-recorded copy data adds "the duel rooms of section 5, with one door bot against one first-generation bot". 6: arm B starts the duel "from the copied network".

**The code.** The list length is `E = S + 33` (`obs_spec_v1.py:108-111`, `shapes` `:2290-2307`) and the aim head is `18 + E` (`action_spec_v1.head_sizes`, `:107-111`), where `S` is the run's largest `max_per_team` over its levels (`rl_env._team_slots` / `make_layout`, `:664-687`). The aimable blocks start at list position `S - 1` (`action_tables`, `action_spec_v1.py:200,211`; `CAPS_FIXED` `obs_spec_v1.py:109`). So: ships, S = 6: E = 39, aim 57 (the design's numbers, and 1j's records: note 1j `:410`). The duel, 1 v 1: S = 1, E = 34, aim 52. The range with up to four targets: S = 4, E = 37, aim 55. A gen_ship stage-2 room gives every spawn room the same `max_per_team` (`tools/gen_ship.py:2088-2092`). The records carry their own `S`, `E`, `K` in `meta` (`control_bots_door.py:613`), and a record of a duel room and one of a ship have different array shapes.

**Why it matters.** "Refused by anything else" plus fixed sizes would refuse arm B (copy on S = 6 plus S = 1 data, duel on S = 1), and 3.1's "one file per array" cannot stack records of two shapes without a padding rule. Neither is a method question: it is what the spec's shapes are.

**Smallest fix.** In 2: state that the observation and action specs are parametrised by S (E = S + 33, aim = 18 + E, the aimable blocks at S - 1), that network v1 is S-agnostic by construction (per-entry readers, attention, the pointer head), and that "fits the specs" means same spec names and same K and V/F/plane counts, not same E. In 3.1: the unpacked copy data pads each record's list and aim mask to the largest E (mask 0), or keeps one store per S. Quote the sizes per task (ships 39/57, duel 34/52, range S = number of targets' room budget).

### F5 (MAJOR). "Watching is a recorded tournament match played back with `main.py --replay`, which exists": three pieces of that path are missing for a learned soldier in a training room

**The design, 7.2:** "Watching is a recorded tournament match played back with `main.py --replay`, which exists." 5.1: "the rooms are outside [the repository] and are made again from the seeds." Section 0 and 5.4: the gate's second half is Erik's eye on the replay.

**The code.**
- The tournament records a replay only when a door team plays and `--records` is given: `record=sink is not None and sink.records`, and the sink is installed only for a `DOOR_VARIANTS` controller (`tools/bot_tournament.py:171-181`). A `net` vs `bots_v1` match writes no replay.
- `main.py --replay` loads the recording's level by folder name from the repo's `levels/` only: `load_level(level_name)` is `level_loader.load(name, levels_dir="levels")` relative to the checkout (`main.py:70,450,483`; `level_loader.py:1186-1191`); there is no `--levels-dir`, and `--level` refuses an absolute path or `..` (`main.py:124-128`). The recording names the level by its folder (`match_replay.level_id`, `src/match_replay.py:201-205`), and `replay_level_name` refuses a disagreeing `--level` (`src/control_match.py:134-138`). A training room living outside the repo cannot be watched until it is copied into `levels/<name>` (an untracked file under `levels/`, which `require_snapshot` refuses in a training checkout, `rl_env.py:1168`).
- `main.py --replay` itself needs no `net` (the replay is the match's one controller, `control_source.py:375-377`), but the process that RECORDS the match must have `net` registered; `tools/run_match.py --record` does not import `agents/`.

**Why it matters.** The rung's gate depends on Erik watching; as written, the path does not exist for the levels rung 2 trains on.

**Smallest fix.** In 7.2: 2d makes the tournament record a replay for any match when `--records` is given (or for any match with a `net` team), and the watching step says the room is first copied (or regenerated) into the viewing checkout's `levels/` -- or 2d adds a `--levels-dir` to `main.py --replay` (a game edit to list). Either way, name it in 2d's row.

### F6 (MINOR). The yardsticks and the in-run win rate use two instruments, and the random player has neither a tournament seat nor a win count

**The design, 5.2:** "On each task's held-back rooms, through the bot tournament (the random player through the farm, as `tools/rl_smoke.py` already plays it)". 7.3: "the farm plays the held-back rooms with the current network against the fixed opponent and the win rate is one more point on the curve."

**The code.** `rl_smoke.legal_v1_rows` does play uniformly among the legal options (`tools/rl_smoke.py:51-59`), but `drive` reports speed and memory only; it never reads `info["result"]` (`:62-118`). The tournament seats registered controllers by name (`tools/bot_tournament.py:156-158`); there is no random controller. So the floor yardstick needs a new win counter over the farm, beside the tournament's `summarise` (`:242-292`), with different side handling (the tournament mirrors teams; the farm's learner is always the run file's team). For 7.3: a `BatchedEnv` plays the levels its run file names, each episode's level drawn by `episode_draw` with replacement (`rl_env.py:249-263`), so "the held-back rooms" means a second `BatchedEnv` (own run file, own workers), and it cannot share the training env's `out_dir`: the manifest check refuses a different `run_sha256` (`rl_env.py:1575-1578`, `:1457-1474`).

**Smallest fix.** Seat the random player in the tournament as a registered controller (the 2d driver with a uniform chooser over the masks), so all three yardsticks and the gate come from one instrument; label 7.3's farm figure as an in-run proxy read from `info["result"]`, measured in a second env with no `out_dir` (or its own).

### F7 (MINOR). The reward groups: the claim holds, the paraphrase of what the soldier is paid does not

**The design, 4.3:** "the 'range' and 'duel' profiles of design 6.9 need no new environment code: they are a weighting of the five numbers the environment already returns under the 'fireteam' profile." 5.3: "The soldier is paid for the health it takes off the targets, for the win, and loses 3 if the clock runs out."

**The code.** True that the env returns `(N, U, 5)` unsummed under a named profile (`rl_env.py:116-123`, `_reward` `:889-900`) and that a run file may override the profile's terms by name (`reward_v1.profile_terms`, `:110-128`; `rl_env.parse_run` `:467-482`), so no env code is needed. But under "fireteam" with weights outcome 1 / own 1, the soldier is also paid +0.5 per kill (`kills_enemy` -> own `marine_alive`, `reward_v1.py:95`), charged its own lost bars and -0.5 for its death (`:94,96`), and the outcome group carries -10 for elimination (`:51,91`). Design 6.9's "range" is "a target loses health; the clock" -- no win. A group weight cannot remove one term inside a group; the run file's term overrides can (`marine_alive = 0`, `win = 0`, ...).

**Smallest fix.** In 4.3 / 5.3 write the run file's `[reward]` table for each task (profile "fireteam" plus the term overrides) and say the deviation from design 6.9 (the range pays the win) is chosen.

### F8 (MINOR). Training rooms on gen_ship's stage 2: feasible, with constraints the section does not name

**The design, 5.1:** "Observation spec v1 needs a room graph, and only ships have one ... one to three rooms, with doors and a few crates, a room graph, a spawn room per team ... built on the ship generator's second stage ... as far as that fits rooms this small."

**The code.**
- `embed` makes each team's spawn room a WHOLE graph room, one `spawn_team` tag per room, every spawn room at `graph.max_per_team` (`tools/gen_ship.py:2088-2092`, `ROOM_TAGS` `:640`). A "one-room" range is therefore two graph rooms (joined by an `open` edge, one compartment), never one.
- `formation` places slot k at a fixed, centred cell of its room (`simulation/match.py:130-143`): within one room the soldier always starts on the same tile; starting places vary only across the bank.
- `embed` needs a hull rectangle with every room strictly inside (`gen_ship.py:2034-2041`); `validate` requires left-right mirror symmetry unless called with `mirrored=False` (`:2526-2547`), every formation cell at `max_per_team` on floor (`:2549-2554`), and an A* path between the spawn rooms (`:2570-2575`).
- gen_ship stamps no cover crate: its "crates" are fuel clusters from the `fuel` tag, i.e. flammable contents (`gen_ship.py:2071-2074`, `ROOM_TAGS["fuel"]` `:643-646`); cover crates are entities of `cover_system.py`.
- "Only ships have one" is true of the repo's levels, but any level can carry a room graph through the Level data layer (`level_lib.write_room_graph`, used by `tests/test_obs_spec_v1.py:186`, `tests/test_match_replay.py:73`), and 1j's "door box" is a two-room 1 v 1 level built that way (`tests/test_bots_door.py:814-848`): the precedent nearest to a training room.

**Smallest fix.** Name these in 5.1 (and decide crates = fuel or cover entities, since a fuel crate can burn under a frag's heat), and point at the door box as the first precedent.

### F9 (MINOR). Section 9 applies the medium ship's rate to the roadmap's small-ship row

**The design, 9:** "at 720 decisions a second, the roadmap's '100 million to a billion decisions' for a fireteam on the medium ship is 1.6 to 16 days ... against the roadmap's 'half a day to four days'."

**The documents.** The roadmap's row is "fireteam on the SMALL ship" (`rl_agent_roadmap_2026-11-11.md:197`, under "small ship, at decimation 4" `:191`); half a day to four days is 1e8 to 1e9 decisions at its 3 000 a second. At the design's own small-ship estimate (2 300 a second, section 9's second row) the row is about 12 hours to 5 days, close to the roadmap's. The 1.6 to 16 days is a medium-ship figure for a row that was never about the medium ship. The roadmap's own note of 2026-11-14 repeats it (`:202-208`). The 720 itself checks: 361.7 env-ticks/s (1i note `:622`) / decimation 4 x 8 rows = 723, rows of both teams including the dead ones (not only living soldiers).

**Smallest fix.** Re-state the finding as "on the medium ship it would be 1.6 to 16 days; on the small ship, by the estimate, 12 hours to 5 days", and correct the roadmap's note the same way.

### F10 (MINOR). "Four existing seams and no other" is contradicted by the design's own later sections

**The design, 1:** "It reaches the game through four existing seams and no other: `rl_env.BatchedEnv` ...; `control_bots_door.read_record` ...; the `policy` controller's decoder with `rl_env.observe_v1` ...; the spec tables ... (`BatchedEnv.layout`)."

**The code / the design.** 7.2 also needs `ControlSource` and `register_controller` (`control_source.py:319`), `DOOR_CONTROLLERS`, the tournament and `run_match`; 3 needs `action_spec_v1.heads_used` / `arg_mask` (named in 14, `action_spec_v1.py:310-334`); the copy's shapes come from the records' `meta` (`S`, `E`, `K`, `control_bots_door.py:613`), not from `BatchedEnv.layout`, which needs levels and workers; 5.1's generator uses `level_lib` / gen_ship; the reserved DAgger remedy uses `IntentCapture` / `translate` (`control_bots_door.py:85,982-987`).

**Smallest fix.** List the seams per part (trainer, copy, `net`, rooms) or drop "and no other" for "the trainer's learning loop reaches the game only through `BatchedEnv`".

### F11 (MINOR). The sight rule is never named

**The code / the documents.** The 1j records were made under `--sight all_round` (note 1j `:481-483`), the rule every later rung trains under (design 3.6). A match file without a `sight` key is `cone` (`simulation/match.py:68,311`), the tournament uses the level's rule unless `--sight` is given (`tools/bot_tournament.py:163-164,394`), and a run file's `[run] sight` replaces the level's (`rl_env.py:442-445,776-777`). A training room written with the match writer's default plays under cone.

**Smallest fix.** 5.1: the room generator writes `sight = "all_round"` into every match file (the writer omits only the default), so the farm, the tournament and the replays agree without a flag.

### F12 (MINOR). "The count of masked choices sent (must stay 0)" counts the opponent's rows too

**The design, 7.1:** "plus the count of masked choices sent (must stay 0)".

**The code.** `BatchedEnv.step` checks and counts masked choices over EVERY row of `actions` (`rl_env.py:1742-1749`, `masked_choices` `action_spec_v1.py:351-363`), including the rows of a `bots_v1` or `idle` team, whose actions are ignored (`EnvRunner.step` feeds only `self.policies`, `:830-832`) but whose masks are built from their living units (`observe` -> `build_masks`, `:972-974`). Zeros sent on those rows are masked whenever an opponent winds up a throw (only "keep" is legal, `action_spec_v1.py:290-293`).

**Smallest fix.** 7.1: the counter is read on the learning team's rows only (or the trainer sends each other row its first legal option).

### F13 (MINOR). The roadmap's "export of a training episode as a replay" is dropped without a word, and the farm cannot do it today

**The documents / the code.** Roadmap rung 2 "Built": "... a checkpoint store, the rating against a fixed pool, the export of a training episode as a replay" (`rl_agent_roadmap_2026-11-11.md:95`). The design moves the rating to rung 3 (7.3) and says so; the replay export is not mentioned. `EnvRunner` can record (`record=True`, `rl_env.py:726-728,781-782`), but the farm worker always builds runners without it (`:1093`) and the run file has no key for it.

**Smallest fix.** Either say watching goes through tournament replays only (F5) and the roadmap item is dropped, or give 2b / 2g a run-file key that records every k-th episode of one env through `MatchSession.record` (a game edit).

### F14 (MINOR). Nothing records which rooms a run trained on

**The design, 5.1:** "the rooms are outside it and are made again from the seeds." 4.7: what a saved network holds.

**The code.** The run manifest holds the run file's sha256 (which names the level folders and the `levels_dir` path, not their content) and a `bank_sha256` that is reserved for P8a and written as None (`rl_env.py:1429-1441`, `:1571-1574`). A room bank regenerated by a changed generator would train under the same manifest.

**Smallest fix.** The training manifest of F2 (a) records a hash over the level folders the run file names (or the generator's commit plus the seed list), and 4.7 lists it among what a saved network holds.

### F15 (MINOR). The learning team gets the bots' commander in every rung-2 task, in training too; the design mentions it only for `net`

**The code.** `create_match_controllers` adds the `commander` (`control_bots.BotCommander`, the layered bots' team layer) beside every `policy` team on a level whose room graph is at the map's shape (`control_source.py:384-390`, `unit_orders.graph_of` `:57-66`). Its order names a place (the objective, e.g. the enemy's spawn room through `room_goal`, `control_bots.py:428-440,1519-1531`) and a priority target (`team_order`, `:1582-1600`), shown to the soldier in the vector's order block and the entry's `priority` flag (`obs_spec_v1.py:237-242,269`). So the "from scratch" range and the duel are trained with a scripted team layer pointing at the target. Consistent with the copy (the door bots write the same orders, note 1j section 2) and with design 5.4 (information only), but it is part of what the soldier sees, and design 6.7's order blanking is rung 4's.

**Smallest fix.** One sentence in 5.3 / 5.4 that the learning team holds the commander's orders (and that the duel's opponent, `bots_v1`, holds none), so a reader of a curve knows the soldier was shown where the target is.

### F16 (MINOR). Tournament workers are not thread-pinned; a `net` in each of them loads PyTorch

**The code.** The farm starts every worker with one BLAS / OpenMP thread (`WORKER_ENV`, `rl_env.py:199-206`, set around `p.start()` `:1549-1558`) because an unpinned worker commits ~1.9 GB on the 16-thread desktop. The tournament's `spawn` pool sets nothing (`tools/bot_tournament.py:416-419`); 1j recorded it as a surprise ("~2 GB of commit each; 12 of them plus the suite exhausted the 16 GB machine's commit charge", note 1j `:938-942`, "Not fixed here"). A `net` controller in each of 12 workers adds PyTorch's own intra-op thread pool and, if it uses the graphics card, 12 CUDA contexts.

**Smallest fix.** 2d pins the tournament workers as the farm does and states where `net` runs its network (CPU with one thread per worker, or one process holding the GPU).

## Verdict

**Survives with fixes.** BLOCKER 0, MAJOR 5, MINOR 11.

The design's picture of the environment is accurate where it matters most to the trainer: the shapes and dtypes it quotes are the code's (for six roster places), the 63 KB per decision recomputes exactly, the env returns the five reward groups unsummed with the clock as an ending, a run file may put a `policy` team against `bots_v1` or `idle`, and 1j's records hold everything the copying loss needs, in decision order per soldier. What does not survive as written is the plumbing around the trainer. The training snapshot and the run manifest do not cover a new `agents/` folder or a second settings file (F1, F2), so "one tag fixes the trainer" and "both hashes in the manifest" need edits to `rl_env.py` that 2b's row denies. The `net` controller cannot "join the door drivers" without editing `control_source.py`, and "built like the door bots" would copy a loop that lives privately inside `BotsThroughDoor` (F3). The spec's sizes depend on the roster size, which differs between the ships the copy learns from and the rooms the duel is played in (F4). And the path by which Erik watches a learned soldier, which the gate depends on, is missing three pieces (F5). Each fix is a sentence or a row of the patch table; none changes the method.

## Checked and true

- Vector 97 (`obs_spec_v1.py:246`), entry 67 (`:288`), E = S + 33 (`:109-111`) = 39 at S = 6; near window 8 + 5 = 13 planes and far 10 + 5 = 15 (`:149-153`), float16 / uint8 (`:2301-2304`); heads 49 / 18 + E / 2 / 7 + K / 5 (`action_spec_v1.py:107-111`), K = 6 hand-grenade slots (`obs_spec_v1.py:116`); full view 64 x 85 (`:296,318`), `full_match` 2T + 3 (`:2239`), `full_watched` 25 x 25 uint8 (`:2240`); reward `(N, U, 5)` (`rl_env.py:557-563,647`).
- The arithmetic: observation at S = 6 is 388 + 10 452 + 39 + 156 + 10 000 + 3 125 + 12 500 + 3 125 + 2 = 39 787 B (+ 126 B of masks); full view 21 760 + 64 + 256 + 28 + 625 = 22 733 B; ~63 KB per decision, 8 000 decisions ~0.5 GB. True.
- Records (`control_bots_door.py:582-643`): observation arrays, `valid` / `in_match`, the four masks, the realised `arg_mask`, the five integers (int16), loss bits, the five reward groups and `rows_ended`, per decision `(n_dec, S)` in tick order, dead and empty rows included as the env serves them; `heads_used` is a pure function of the masks and `arg_mask` (`action_spec_v1.py:326-334`), so nothing it needs is missing; a memory copy can be trained (one file per match and team, rows in decision order). The full view is not recorded (note 1j `:423-425`), which arm B does not need.
- 662 766 living-soldier decisions in 360 held-out records (note 1j `:844-845`), so ~1.1 M in 600; 600 records from the design runs, 4.0 GB (`:903-906`); walls 58-87 s per match (`:825-828`): "about an hour" at 12 workers holds.
- `[run] policy` per team: any nameable controller or `policy` (`rl_env.py:401-415`); `bots_v1`, `idle`, `bots_door` are nameable (`control_source.py:295-310`). A 1-soldier team against 1-4 `idle` marines with no horde is a legal match (`simulation/match.py:256-321`: two team rows, `n` within the room's `max_per_team`, no zombie room needed at `zombies = 0`); `MatchRules` ends it on fewer than two marine teams alive.
- Endings under a named profile: every end is `terminated`, `truncated` False (`rl_env.py:876-878`); the ended episode's terminal observation and full view are in `info["final_obs"]` / `info["final_full_view"]`, the env resets inside the step and `obs` is the next episode's tick 0 (`:1130-1133,1759-1762`); the step's reward belongs to the ended episode (`books.finish` `:861-862`).
- `load_levels` refuses a level without a room graph under spec v1 (`rl_env.py:703-711`), as does `bots_door`'s `bind_match` (`control_bots_door.py:698-706`).
- `[run] out_dir` is refused inside the checkout (`rl_env.py:437-440`); `require_snapshot` checks an annotated tag at HEAD, the clean tree, module paths, build id and the run file's sha256 (`:1346-1376`); the manifest records python / numpy / torch / CuPy / CUDA / GPU without importing torch (`:1409-1426`).
- `rl_smoke.legal_v1_rows` draws each head uniformly among the served legal options and the argument within `arg_mask` (`tools/rl_smoke.py:51-59`).
- The tournament seats any registered controller per team, mirrors sides, and accepts `--levels-dir`, `--zombies 0`, `--sight` (`tools/bot_tournament.py:145-165,344-365,375-413`); `bots_door` plays a 1 v 1 on a two-room level (`tests/test_bots_door.py:814-863`).
- A replay re-simulates from recorded intents with the `replay` controller as the match's only controller, so the drawing process need not be reproducible (`control_source.py:375-377`).
- Farm figure: 361.7 env-ticks/s on fireteam, 64 envs / 16 workers, decimation 4 (1i note `:613-622`; CLAUDE.md Batched environment row); x 8 rows / 4 = ~720.
- The bots never sprint (`travel_throttle = 0`), never reload by choice, never plant (note 1j `:50-65`).
- PyTorch 2.15 is the shared stack's pin (`ml-stack/ml_stack.txt:6`).
- `DOOR_CONTROLLERS` means "a door driver with no brain of its own" (`control_source.py:311-316`), so `net` belongs in it by meaning (F3 is about how, not whether).

## Not checked

- `ship_layout.py` and the recipe path of gen_ship for "about ten generated freighters with a room graph" (assumed from the CLAUDE.md Level generators row: every ship is written with its room graph since 1b).
- Whether `bots_v1` and the commander behave sensibly in a two-room level (they read spawn rooms, `control_bots_v1.py:492-510`, `control_bots.py:1517-1531`; not run).
- The environment notes' claim about the laptop's Tailscale relay (4.6).
- The 1h note and the residency benchmark document beyond the 1i reading.
- `level_lib`'s writers line by line (only their use in tests was read).

