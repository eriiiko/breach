# Rung 1, "the door": the handoff plan

2026-11-11. Written by the game AI lead for the session that runs rung 1 (the
orchestrator), and for Erik. Rung 1 builds everything a network needs from
the game, with no network yet. It ends when the scripted bots can be driven
through the new door and still fight.

- **The design** is agreed and critiqued: `docs/rl_agent_design_2026-11-11.md`
  (version 2; "the design" below, cited by section). **Do not re-derive it.**
  A line it marks as an accepted gap is a decision. Where it is silent, choose
  the simplest honest thing and say so in the report.
- **The ladder:** `docs/rl_agent_roadmap_2026-11-11.md`, rung 1.
- **The critics' verdicts**, with file and line for most seams named below:
  `docs/rl_agent_design_2026-11-11_critique/` (lens 3 is the systems one).
- **Tracker:** issue #195. One checkpoint comment
  after every merge.

## 1. How the orchestrator works

1. **Load** the skills `game-ai-lead`, `autonomous-patch-workflow` and
   `git-workflow`. Read this plan, then the design's sections named by the
   patch in hand. Run on Opus. `git fetch origin && git status` first.
2. **One patch at a time, one implementer at a time,** each a fresh agent in
   its own worktree `.claude/worktrees/192-1x-<slug>` on branch
   `192-1x-<slug>`, cut from a freshly pulled main. The orchestrator holds the
   plan and short summaries; it does not implement.
3. **Before a patch starts it has a note.** 1a, 1b and 1c have theirs below
   (section 4). For the others the orchestrator writes a one-page note from
   the design and from the questions listed in section 5, commits it on the
   patch's branch, and only then briefs the implementer. 1g is different
   (section 5).
4. **The brief** follows the shape that worked for the residency patches
   (memory: `reference_residency_patch_brief_template.md`): role and scope;
   where to work; read first; what to build; the gates; the house rules; the
   report. The house rules that matter most here:
   - this machine's breach Python (`docs/dev_setup.md`), `pytest tests -q`;
   - never `git add -A`; stage explicit paths; commit at every coherent step;
   - tests assert properties, from the note's test list; a failing old test
     is a finding to report, never a target;
   - the canonical-systems table in `CLAUDE.md` is checked before anything
     new is built; a new system's row is written into `CLAUDE.md` in the
     patch that builds it, from the draft rules in the design's section 10;
   - a patch that touches a Class-P file or the build (`tools/twin_debt.py`
     and `tests/test_class_p_level.py` say which) merges only with
     `tools/cuda_green.py` green on this checkout's own CUDA build;
   - a game finding outside the patch is one line in the report, not a fix.
5. **Green gate, then merge** `--no-ff` from the main checkout, push, delete
   the branch and the worktree, without asking. No patch of rung 1 waits for
   Erik: 1d's second step changes how the bots play, and he blessed it in
   advance (2026-11-11); its gate is the tournament's numbers.
6. **After each merge:** a checkpoint comment on the tracker issue, in plain
   words for Erik; memory (`project_rl_agent_design_session.md`); one dated
   line in `game-ai-lead`'s run log if something was learned.
7. **When to stop and ask.** A question about what the game is, what a
   soldier may know, or anything by eye goes to Erik, one question at a time.
   A gate that fails twice, or a note that would contradict the design, stops
   the patch: commit, report, wait.

## 2. The ten patches

| patch | what | implementer | gate | needs | note |
|---|---|---|---|---|---|
| 1a | sight as a setting of the match | Sonnet | every pinned match bit for bit | | section 4.1 |
| 1b | the room graph saved with the level | Opus | the saved graph equals the generator's; match digests unmoved | | section 4.2 |
| 1c | sight for many tiles, in C++ | Sonnet; Opus if the gate fails twice | equal to the sight test, ray for ray; the CUDA build green | 1a | section 4.3 |
| 1d | the team picture, in two steps | Opus | (i) the layered bots bit for bit; (ii) the bot tournament | 1a, 1c | to write |
| 1e | the verbs' gates as read-only functions | Sonnet | digests in lockstep before and after | see section 3 | to write |
| 1f | the order | Opus | a match of bots alone plays and hashes as before | 1b; see section 3 | to write |
| 1g | observation spec v1 and the value network's full view | **Fable** (Erik, 2026-11-11), in small pieces | equals an independent reconstruction; shows nothing the team cannot know; the farm's speed measured | 1b, 1c, 1d, 1f | its own spec document first |
| 1h | action spec v1, the decoder, the masks | Opus | a masked choice issued by force does nothing | 1d, 1e | to write |
| 1i | reward profile v1 | Opus | the books close (design 6.8) | 1g's row flags | to write |
| 1j | the bots through the door | Opus | section 5 | all | to write |

No patch has a HUMAN-TEST. Model names are families; spawn agents with the
family alias.

## 3. Beside the residency arc (#29)

The residency patch P3a gives a unit one definition of its synced state
(a schema, and `__slots__` on `Unit`, so an attribute nobody declared raises
where it is set) and re-pins the unit digest. It restarts now that rung 0 is
closed, in its own session; its input from this design is section 9.

- **1a, 1b, 1c and 1d do not touch unit state** and run beside P3a freely.
- **1e and 1f edit `unit.py` and `simulation.py`.** Before starting either,
  look at `git worktree list` and at #29's last comment. If P3a is in flight,
  take another patch first or wait for its merge. If P3a has merged, 1f's
  four order fields are four rows of the schema; if not, they are plain
  attributes with their own digest key, and P3a picks them up.
- One implementer at a time holds **per session**. Two sessions running an
  implementer each is Erik's budget call, not the orchestrator's.

## 4. The notes for 1a, 1b and 1c

### 4.1 Patch 1a: sight as a setting of the match

**Design:** 3.6. **Why first:** everything later is measured under all-round
sight, and nothing may move the existing matches.

**Build**

1. A key of the match file: `sight = "cone" | "all_round"`, default `"cone"`.
   `src/simulation/match.py`: `_MATCH_KEYS`, a `sight` field on `MatchSpec`,
   and `parse_match` refusing any other value by name.
   `level_lib.format_match_text` writes the line only when it is not the
   default (as it does for `zombie_controller`), so every existing
   `match.toml` stays byte for byte.
2. The simulation holds the rule: set by `Simulation.spawn_match(spec)`,
   `"cone"` on every path that is not a match. `vision.can_see` reads it from
   the sim it is handed, never from a caller. The per-tick cache
   (`vision.state_for`) needs no change: the rule is constant for a sim.
3. **What all-round means:** every living unit that is not a zombie sees
   every living unit of another team to which `vision.ray_clear` is clear. No
   cone, no radius. Zombies are not observers under this rule: nothing reads
   what a zombie sees, and their waking is its own law (`ai_zombie`), which
   this patch does not touch. Under `"cone"` the code path is today's,
   unchanged, zombies included.
4. The run file: an optional `[run] sight` key in `rl_env.load_run`, refused
   by name when it is not one of the two values; when present it replaces
   the level's value in the spec before the match spawns. It is part of the
   spec the recorder stores, so a recorded all-round match replays as one. An
   old recording (no such field) must still replay as a cone match:
   `match_replay.spec_to_dict` writes the field only when it is not the
   default and `spec_from_dict` reads it with the default, so no format
   version moves. If that cannot be done, say so and stop: a format version
   is the orchestrator's call.

**Do not** change either bot generation, the cone's config keys
(`[onephase] vision_cone_half_deg`, `awareness_radius_tiles`), or the zombie
law.

**Gates**

- `tests/test_bots_v1.py::test_bots_v1_play_the_pre_g8p_pinned_match`, the
  whole suite, and `tools/match_pin_check.py` equal on every backend built
  here.
- New property tests: under all-round a unit directly behind an observer,
  with a clear line, is seen, and with a wall between it is not; a zombie
  observes nothing under all-round; the match door refuses an unknown value
  by name; a spec at the default writes no `sight` line; a recorded
  all-round match replays with every checkpoint equal.

**`CLAUDE.md`:** amend the rows "Match config" and "Vision / engagements /
cover" (one clause each).

### 4.2 Patch 1b: the room graph saved with the level

**Design:** 5.1, 5.7; section 10's draft rule "Room graph". **Read first:**
the `CLAUDE.md` rows "Level data layer", "Level generators" and "LevelSpec";
`tools/gen_ship.py` (`Room`, `Edge`, `ShipGraph`, `EDGE_KINDS`, `write`);
`level_lib.py` (`MANAGED_FAMILIES`, `write_managed_blocks`); `level_loader.py`.
The level generator belongs to the level design lead's domain: if a choice
here would change what a generated ship *is*, stop and ask.

**Build**

1. **The fixed part, in the level folder, through `level_lib` only:**
   - `[[room]]` rows in `level.toml` (a new managed family): `id, x, y, w, h,
     kind`, the generator's own air rectangle and id;
   - `[[opening]]` rows: `id, a, b, kind, x, y, w, h` (the two room ids, the
     edge's kind, the opening's tiles). One row for every edge whose kind can
     be walked now or later (a door of any sort, an open edge, an airlock, a
     breakable wall); none for a window, a grille, a bulkhead or a vent-only
     edge. The id is the row's rank when the openings are sorted by (lower
     room id, higher room id, first tile in row-major order): the same ship
     gives the same ids, and nothing counts upward;
   - `rooms.npy`: for every tile the id of the room whose rectangle holds it,
     and −1 elsewhere (walls, voids, the tiles of an opening).
2. **The loader door:** `LevelData.rooms`, a frozen record (rooms, openings,
   the grid), `None` for a level that has none. It refuses by name: an
   unknown key, a room outside the map, an opening naming a room that does
   not exist, a grid whose shape is not the map's.
3. **The generator writes it** on both of its paths (the first generator and
   the recipe ships: both end in a `ShipGraph`). The checked-in generated
   levels are regenerated with their own seeds; everything they already hold
   must come out byte for byte, with only the new rows and the new file
   added.
4. **The changing part, worked out and never stored:** a small module (for
   example `src/simulation/room_graph.py`) with two read-only functions over
   a loaded graph and the map as it stands:
   - for each saved opening: passable now; and, if not, whether it is a door
     a unit could open (a door entity that is not a hull door and is not
     wired);
   - the pairs of rooms that are joined now, which also finds a hole blown
     through a wall: a connected patch of walkable tiles outside every room
     that touches two rooms joins them.
   Both are cached on the map's own "the walls changed" counter
   (`GameMap._tile_serial`) and the doors' states. Nothing here is synced or
   hashed.

**Do not** touch the bots (their room goals read the spawn rooms and stay
as they are; the draft rule says where a later reader goes), the order
(1f), or the simulation's tick.

**Accepted gap:** a hole blown through a wall joins two rooms in the worked-
out graph but has no id, so an order cannot name it. The later commander
(rung 6) reads the joined pairs.

**Gates**

- Property tests: every tile of a room's rectangle that is floor maps to
  that room; every saved opening's tiles touch both of its rooms; the same
  seed gives the same ids twice; closing a door entity turns its opening
  not passable and openable; destroying a wall tile between two rooms joins
  them and sealing it parts them again; the loader's refusals.
- The regenerated levels differ from the checked-in ones only by the new
  rows and file. The whole suite; `tools/match_pin_check.py` equal (no match
  digest moves: the simulation does not read the graph).

**`CLAUDE.md`:** a new row "Room graph"; amend "Level data layer" (the new
family and file) and "Level generators".

### 4.3 Patch 1c: sight for many tiles, in C++

**Design:** 3.4 (the sight planes), 3.7 (cost); section 10's draft rule
"Sight for many tiles". **Read first:** `GameMap.has_los`
(`src/simulation/gamemap.py`, the line walk), `vision.ray_clear` (the walk
plus cover that blocks sight), `GameMap.sight_screen`; as the model of a
host C++ call in both modules: `cpp/src/horde_field.{h,cpp}`, its binding in
`cpp/src/bindings.cpp` and its two entries in `cpp/CMakeLists.txt`, and the
Python pair `horde.flood` / `horde.flood_reference`.

**Build**

1. **One C++ routine with two entries,** integer only, in a new translation
   unit on the `/fp:strict` list, bound in the CPU and the CUDA module:
   - *many rays:* for an array of (from tile, to tile) pairs, is each line
     clear;
   - *a window:* from one tile to every tile of a rectangle, or from every
     tile of the rectangle to the one tile (the direction matters: the walk
     is not symmetric), as a yes / no plane.
   The walk is `has_los`'s own, step for step: the same error term, the same
   order of the two axis steps, the same treatment of the end tile and of
   tiles outside the map, stopping on `solid` or on the sight screen. A
   switch gives the other question `has_los` already answers
   (`physical=True`: stop on `solid` only), which the line-of-fire flags of
   1g need.
2. **Cover that blocks sight** (`vision.ray_clear`'s second half) stays in
   Python: the cover entities are few. The Python wrappers in `vision.py`
   apply it to the routine's answer, so that the wrapper equals `ray_clear`
   on every map. A pure-Python reference of the routine stands beside it for
   a bare map with no engine, as `horde.flood_reference` does.
3. **The first user:** `vision.compute` under the all-round rule of 1a asks
   for its unit-to-unit lines through the many-rays entry in one call. The
   cone rule's path is not touched.

**Why not a fill of the whole map:** on the small ship that is about 7 000
lines per viewpoint. The sight planes are needed only inside a soldier's
25 x 25 near window, so the window entry is the shape 1g will call.

**Gates**

- Equality, ray for ray, with `vision.ray_clear` on generated maps that
  include a grille (the sight screen) and a cover entity that blocks sight;
  the window entry equals the many-rays entry in both directions; the
  `physical` switch equals `has_los(physical=True)`. Break the routine once
  on purpose (swap the axis steps) to see the gate fail.
- Under all-round, `vision.compute` gives the same seen-by lists with the
  routine as with 1a's Python path, over a recorded match.
- Report rays per second and the per-tick cost of `vision.compute` under
  both rules, before and after, on the small and the medium ship, timed over
  a fixed number of steps.
- `tools/cuda_green.py` green (the patch touches the build), the twin ledger
  clean, the whole suite, `tools/match_pin_check.py` equal.

**`CLAUDE.md`:** a new row "Sight for many tiles"; amend "Vision /
engagements / cover".

## 5. What each later note must settle

The orchestrator writes these, one at the boundary before its patch, from
the design. Each ends with its own test list (properties) and the `CLAUDE.md`
rows it writes or amends.

**1d, the team picture** (design 3.3, 3.6, 4.5; roadmap system "Team
picture"; lens 2 M1 and M2, lens 3 m1).
- Where it lives and who drives it. It is memory that must be updated on
  every tick, so it cannot be built on demand like the sight cache; it is
  not synced and not hashed; it is switched on only for a match.
- What a record holds (place, tick, class and motion at the sighting; no
  live unit object).
- Step (i): the bots read it with today's exact meaning, records dropped at
  death included. Gate: the layered bots' matches bit for bit against final
  digests taken on main before the patch, over several seeds and both ships
  (`bots` has no frozen reference match today; take them first).
- Step (ii): records outlive an unseen death; lanes are read from state
  through the one drawing routine; the bots move onto both. Gate:
  `tools/bot_tournament.py`, tuning seeds and held-out seeds, before and
  after, in the checkpoint comment. The first-generation bots are not
  touched in either step.

**1e, the verbs' gates** (design 2.4; lens 3 M1).
- One read-only function per hand verb, called by the verb and later by the
  mask; the reload's from the read-only magazine status.
- The two oversights the critique found (a stunned unit can reload and open
  a door) are **not** fixed: they are Erik's to rule on (design section 11).
- Gate: the A/B lockstep harness, digests equal tick for tick.

**1f, the order** (design 5, 5.7; lens 3 M4).
- The four fields and their whole-number codes; the commander's verb, in
  `INTENT_KINDS` and in `control_policy.EXCLUDED_KINDS`; a new recording
  format version; applied inside the tick; cleared at death, at turning and
  when its time runs out; hashed only while set.
- The bots' team layer writes orders only for a team that is driven through
  the door: its objective tile becomes a room through 1b's grid, its play
  becomes the order's "what", its shared target the priority target.
- Section 3 above (P3a).

**1g, the observation and the full view. Fable, in small pieces.**
- **First a spec document,** `docs/rl_agent_obs_spec_v1_<date>.md`, written
  by one Fable agent with a narrow brief: every feature of design sections
  3, 4, 4a, 5.2 and 7 as a row (position, name, the seam it is read from,
  the table value that divides it, the squash, the stored type), the list's
  caps and order, the planes of both windows in order, the layout in the
  env's buffers, the two row flags, the cost budget per decision. One Opus
  critic then reads it through the information lens (lens 2's brief is the
  model), and the verdict is on disk before any code.
- **Then three implementation patches,** each its own branch, worktree,
  brief and merge, each small enough that a Fable agent finishes without a
  resume: the soldier's vector, the list and the row flags; the two windows
  with the sight planes and the lanes; the full view. A killed Fable agent
  is not resumed for more than its report: rebuild from its commits.
- Everything is served through the sensor accessor, the one observation
  door. A run whose observation spec reads the room graph refuses a level
  without one.
- Gates: the design's 3.7 and 7.4, and the farm's speed against today's by
  the benchmark protocol (a fixed number of steps per env).

**1h, the action spec and decoder** (design 2; lens 3 M2 and m2; lens 4
minor 2 to 5).
- The order of the options in every head; the action row a trainer sends
  (five small numbers) beside today's raw columns, which stay as their own
  spec; the decoder as a per-tick step inside `AgentPolicy`, holding game
  ids; the masks in the env's buffers; the aim held through a wind-up.
- Gate: a masked choice issued by force leaves the match's digests where a
  do-nothing row leaves them.

**1i, the reward** (design 6, above all 6.3 and 6.8).
- The bar tracker (health at spawn or at turning), credited health through
  the tick's events, the balance read from state, the five numbers, the
  clock as an ending. Tests: the design's 6.8 list, as written.

**1j, the bots through the door** (roadmap rung 1; design 2.7).
- A controller that lets the layered bots decide as they do and then sends
  each decision through the action spec at the agent's decision rate, each
  translated to the nearest choice.
- The measurement: door-driven bots against free bots, sides mirrored,
  tuning and held-out seeds, on both ships, with and without the horde,
  through `tools/bot_tournament.py`.
- **The pass mark** (the lead's call, 2026-11-11; Erik may move it): the
  door-driven bots win at least 40 % of the decided matches against the free
  ones, with no more friendly-fire hits per match than the free bots plus a
  fifth. Below that, the design's variants (8 against 16 directions; a
  decision every 4, 6, 8 ticks; the trigger per tick; throws by direction)
  say which coarseness costs, and the finding goes to a design session
  before rung 2.
- The same runs save the bots' play as (observation, choices, masks)
  records outside the repository, for rung 2's imitation experiment.

## 6. Systems

Existing systems each patch must use, and the new systems with their draft
rules, are the design's section 10 and the roadmap's section 7. This plan
creates none of its own.
