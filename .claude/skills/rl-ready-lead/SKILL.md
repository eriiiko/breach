---
name: rl-ready-lead
description: Claude's standing role as lead of Breach's RL-ready milestone (#76) — the mandate and its limits, what "RL ready" means (the game complete enough that agent design can start), the decided v1 match, the five things built early so both agent shapes fit later, the open items, the order of work and where everything lives. Load at the start of ANY session that touches levelgen, the match config, bots, zombies, the replay path or the RL-ready issue list, before reading the code.
---

# RL-ready lead — continuity skill

**Status: PROVISIONAL** (created 2026-10-29, the #76 scoping session).
Every session that uses it amends it before merging: what moved, what was
decided, what was learned. A step is a MUST only with its incident beside it.

## 1. The mandate (Erik, 2026-10-29)

Erik appointed Claude **lead of making the game RL ready**, in the shape of
the octopus lead: "steer this project to its goal, I'd like to complete this
quickly." Concretely:

- **Claude decides**: scope and order inside the milestone, how issues are
  cut, which prep patch runs first, the technical shape of levelgen, the
  match config, the bots and the replay path, what goes in this skill.
- **Erik decides, always**: anything feel-adjacent by eye (HUMAN-TEST is
  unchanged), the game-design calls listed in §3 when they change (objective,
  loadout, suits), physics-modelling calls that reach into the shared sim,
  budget (effort level, Fable vs Opus agents), and every agent-design
  question (§4) — those are deliberately AFTER this milestone.
- **Default**: when unsure whether a call is Claude's, make it, say so in the
  report, and let Erik veto. Record rulings scoped to their case.

## 2. The goal, in Erik's words

- "RL ready" = **the game is complete enough that it makes sense to start
  building the reinforcement learning.** Not a gym env, not a trained agent,
  not batched envs: those come after, and "once we get started we'll get new
  motivation for many things — moving all the physics to the GPU,
  parallelise etc."
- **Speed matters.** Erik's Berzelius (NSC supercomputer) access lasts while
  he is a PhD student, about 3.5 years from 2026-10-29. Finish the milestone
  quickly; prefer the simple version of everything.
- Once RL starts he wants to **try many kinds of agents and compare**: per-
  marine vs whole-team policies, suited vs unsuited marines, different
  rewards. So the game must not bake in one answer (§5).

## 3. Decided (with dates)

| Date | Decision |
|---|---|
| 2026-10-29 | **Levelgen v1 is simple**, written through `level_lib` like `gen_explosion_studio.py`: a seeded ship surrounded by vacuum with an airtight hull, inner compartments and doors, two mirrored spawn zones, a `max_per_team`, **vents and ducts auto-placed**, optional zombie and wildlife spawn points. The graph-grammar levelgen (`docs/breach_levelgen_design_v0.1.md`, #27) is the long game, not v1. |
| 2026-10-29 | **Objective v1: eliminate the other team**, time limit, draw at the limit. Bomb-site is v1.1 (it is what makes breach charges and decompression matter). |
| 2026-10-29 | **Loadout v1: everyone the same**: the K5 carbine + 2 frag grenades. One flamethrower per team is a roster VARIANT behind the match config (after #20). |
| 2026-10-29 | **Team size is a variable** of the match config, within the level's `max_per_team`. Slots are fixed per team (§5). |
| 2026-10-29 | **Wildlife behind a flag, off in v1.** The octopus (milestone 8) and the swarm (#63, a third done) are not on this path; levelgen leaves them spawn points. **Zombies ARE in v1** (Erik: "I'd really really like that"): `ai_zombie.py` attack-nearest, a levelgen count, and **killed marines convert to zombies** (Erik, 2026-10-29: "please, let's not remove this" — Claude had proposed OFF; the ruling is IN). |
| 2026-10-29 | **Suits are a per-unit environment-profile row chosen by the match config**: a team can be suited, unsuited or mixed. Unsuited marines die of vacuum, fire and gas, which Erik finds more interesting. |
| 2026-10-29 | **No respawn. Friendly fire ON.** |
| 2026-10-29 | **Rule-based team AI is built** (a control source): walk toward the enemy, shoot on sight, throw a grenade at a group. The first opponent and the baseline. |
| 2026-10-29 | **Headless match**: no window, no light requested, no renderer touched. |
| 2026-10-29 | **Replay / watching a match is its own milestone** (recorder grows units + events, a playback mode in the game, a camera that follows). It is the only way to judge an agent. |
| 2026-10-29 | **Time model**: matches run on the plain tick loop (`ContinuousRealtime`), never TwoPhaseWEGO's rewinding tick. |

## 4. Deliberately NOT decided here (agent design, after RL ready)

Observation (full map vs local crop, line of sight), action set, reward,
episode length, one policy per marine vs one per team, communication between
marines, trainer library, batching (#77), device unit store (#29 B3). Erik:
"RL ready is when we start with the agent design." Do not pre-empt these; do
build so both answers fit (§5).

## 5. The five things built early so both agent shapes fit

1. **One control seam.** Every controller — the rule-based team AI, the
   zombie AI, later a per-marine policy or a team policy — is a
   `ControlSource` emitting per-unit intents each tick. No controller outside
   the seam.
2. **Fixed roster slots.** Each team has `max_per_team` slots assigned at
   spawn and KEPT when a unit dies. Variable team size is a mask, never a
   shape change.
3. **Observations per unit and per team from one accessor.** Sensors read
   through `sensor_accessor`; a team view is the union of its units' views.
   No global observer gets baked into the engine.
4. **Credited events.** Every hit and kill carries attacker, victim and
   cause — bullets, a grenade's owner, fire, vacuum, gas. Audit that every
   environmental death is credited; reward design needs it per unit.
5. **Simultaneous actions on one tick.** All units act in the same tick.

## 6. Where things live

- Scope doc: `docs/rl_ready_scope_2026-10-29.md` (the design doc behind this
  skill; its §Systems lists what is reused and what is new).
- Planning: GitHub milestone **RL ready** (#76 the exit gate; the issues cut
  from the scope doc are listed on #76). The pinned Roadmap #46 carries the
  order. Post agreed plans as issue comments, not only in chat.
- Levels are written by `level_lib` only; the precedent generator is
  `tools/gen_explosion_studio.py`; the airtight lint is `tools/level_airtight.py`.
- Controllers: `src/simulation/ruleset.py` (`ContinuousRealtime`),
  `control_source.py`, `intents.py`, `ai_zombie.py`.
- Environment profiles: `src/simulation/environment.py`.
- Events: `src/simulation/events.py` (`UnitHitEvent.source` is a string today).
- Recorder: `src/simulation/recorder.py`.
- Memory: `project_rl_ready_milestone.md`.

## 7. Procedure for a session

1. `git fetch origin && git status`; read §3 and §8 of this skill, then #76's
   latest comment, then memory.
2. Say which issue the session serves. One system at a time. The simple
   version first; a nicer one is a follow-up issue, not a scope creep.
3. Build per `autonomous-patch-workflow`: a worktree per branch, one writer,
   property tests, digest gates. New levels through `level_lib`; new
   controllers through the control seam; anything that moves a digest is a
   deliberate re-baseline with rationale.
4. Feel-adjacent (how a match plays, how a bot behaves on screen) →
   HUMAN-TEST; Erik plays before merge. Headless and mechanical → auto-merge
   on green.
5. Close: post the result on the issue, update memory, amend this skill.

## 8. Run log

- 2026-10-29 — #125 tick budget (branch `125-tick-budget`, measurement only): `bench_bot_matches --timing` (ctrl vs step per tick, `--controller idle`, `--zombies-idle`, `--sim-seconds`, `--repeat`, `--cuda [--resident]` importing a build dir read-only); report `docs/tick_budget_2026-10-29.md`. v1 match, CPU: small 5.9x / medium 2.0x / large 0.71x real time; resident CUDA 4.8x / 3.0x / 1.3x. Learned: physics is ~85-90 % of a tick and linear in area; the bots cost 1 / 2 / 8.5 ms a tick but as pure-Python A* SPIKES (to ~280 ms on large); `find_burst_walls` is a pure-Python 3.5 ms every tick on medium; CUDA's second match in one process runs 10-20 % slower (not investigated); the tick_digest is identical on CPU, CUDA and resident for every run.
- 2026-10-29 — #123 credited events (branch `123-credited-events`): `events.Cause` (values = the old strings) refused at the packet door; `DamagePacket.attacker_id` + `indirect`; payloads executed with `owner_id`; same-tick heat/gas claims in `damage.TickCredit` for jets, frags' heat splash and gas payloads; attacker hashed in the unit digest (ship_demo seed 1 tick_digest re-pinned c69925fa -> 4d32170a, sole cause proven; GOLDEN_AGGREGATE unmoved — the golden has no hits); bench reports deaths by attacker team. Learned: the golden scenario has NO hit in its 30 ticks (a briefing assumed it did — check before planning a re-baseline); vacuum / O2 / pressure damage does not exist yet, so 'credit a breach's vacuum' had nothing to credit; on ship_demo seeds 1–3 more than half the marine deaths are TEAM-KILLS (11 of 21, the bots shoot through team-mates) — a finding for #121's bots; a marine's frag landing after it was bitten and turned is credited to its id, so the bench files indirect kills under the attacker's last marine team; the bench's watcher first sees a match after tick 1, so a death on tick 1 would be misfiled (no real match has one).
- 2026-10-29 — #122 zombies through the seam (branch `122-zombies-seam`): `control_zombies.ZombieControl` registered as `zombies`, the horde's controller named by match.toml `zombie_controller` (default `zombies`, not written at the default, so no match file changed); the law is WRAPPED — the conductor's slot 5 stays its one driver and, in a match, runs it only for the ids the controller names through `Simulation.drive_zombies` (`create_match_controllers` hands the horde over). Learned: wrapping by calling the law from `handle_frame` would have moved it ahead of movement and shooting in the tick and changed every match — the intent-consumed-at-the-slot shape is what keeps ship_demo seed 1 bit for bit (c69925fa..., pinned); bites now emit melee UnitHit/UnitKilled, and those events ARE in the per-tick unit digest (a briefing said otherwise), so a tick holding a bite hashes differently — the golden has no zombies. #139 on seed 1: the survivors stall ~28 s with MOVE_DIR held, a 5-tile path and no fall-back, beside the frag'd zombie room at 1.7 atm decaying to ~1.25 — candidate 1.
- 2026-10-29 — #121 bot team built (branch `121-bot-team`): `control_bots.BotTeam` registered as `bots`, every unit through the gamepad's intent facade (only seam change: THROW's optional `reach_q`), `[bots]` table, `main.py --control match` (MatchObserver + `ControlSource.on_tick`, so the window plays the runner's ticks). Learned: steer for a waypoint tile's MIDDLE (a footprint anchors at `int(x)`; steering for the edge creeps forever); on ship_demo NO unit under MOVE_DIR can cross a legacy `MAT_DOOR` — the exchange pressure push reads the door tile's atmosphere (0, it is flow-solid) as a pressure and shoved the unit back 0.2 tile/tick (FIXED on this branch, lead's call: the push now reads only a footprint's gas tiles; goldens unchanged), while an over-pressured compartment still pushes back across its door (~35 s stalls at the frag'd zombie room — Erik's call); with the fix ship_demo bots resolve in ~64 s, team 0 winning 9/10; the playground's spawns see each other, so its matches end in ~15 ticks of gunfire; `process_shooting` resolves in unit-id order, a first-shot edge for team 0 (§5 item 5).
- 2026-10-29 — #120 levelgen v1 built (branch `120-ship-levelgen`): `tools/gen_ship.py`, a room GRAPH (mirrored binary split) then an EMBEDDING, gates inside the generator, `levels/ship_demo` (medium, seed 1, vents). Learned: entity doors spawn CLOSED as `MAT_DOOR_CLOSED` (mobility 0), which A* cannot cross, so v1 cuts legacy `MAT_DOOR` tiles (walkable, flow-solid) — doors that open are for the bots (#121); count compartments by structure, not by air (a pocket behind a crate is its room's, not a lost compartment: that one rule took the rejection rate from ~25 % to ~1 %).
- 2026-10-29 — #119 P2 built (branch `119-match-p2-runner`): match.toml + `[[spawn_room]]` through `level_lib`, `MatchRules`, the match controller registry (`idle`), `tools/run_match.py`. Learned: with idle controllers nothing draws from `sim.rng`, so two seeds give ONE digest on the playground; the seed gate needs shots (a test controller through the registry) — the bots (#121) make the seed live in real matches. A playground tick is ~12 ms on the Home Desktop, so the playground match runs 60 s of sim (1440 ticks, ~18 s wall), not the 300 s default.
- 2026-10-29 — zombie conversion ruled IN by Erik (Claude's OFF proposal vetoed); #116 then #73 started as agent patches, sequentially per the one-implementation-agent rule.
- 2026-10-29 — skill created in the #76 scoping session; scope doc written;
  issues cut; Roadmap's RL-ready section reordered.
