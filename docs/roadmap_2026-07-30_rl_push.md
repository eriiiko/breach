# Roadmap 2026-07-30 — the RL push: from finished mechanics to trained bot teams

*Dated capture (append-only per doc culture). Agreed direction Erik + Fable,
2026-07-30; **BLESSED by Erik 2026-07-30** — the priority ledger carries the
planning-window pointer (the ledger stays the standing stack; Erik owns
ordering; `TODO.md` stays the item-level ledger — this doc is the map, not the
territory).
Successor planning window to `roadmap_2026-07.md`, which is fully discharged
(Phase 0 determinism tag + Phase 1 EOS done; its Phase 2 one-liner — "weapons →
units → game rules → self-play NN training" — is what this doc expands).*

*Committed to main on blessing, 2026-07-30, together with the TODO.md
staleness sweep (verified deletions enumerated in that commit message).*

---

## 0. Vision update (2026-07-30)

Erik, after refining OnePhaseWEGO: pure WEGO feels like re-inventing XCOM; the
controller session produced joy. New framing:

- **Real-time action is the primary experience.** WEGO stays — as an alternate
  control scheme AND as a training-data/annotation tool (§3.0).
- **Teams are fully autonomous end-to-end** (AI commander + AI units). Everything
  from the previous design discussions stands — the three-model architecture and
  the commander↔unit abstraction dial (`research/rl_litsearch_2026-07-20.md`)
  remain the research program.
- **The player is insertable at ANY seat**: commander of a squad, squad leader
  mid-hierarchy, or a rookie taking orders from an AI commander. Campaign
  fantasy: advance in rank across a single-player campaign. **Design invariant
  that enables this: one protocol** — every seat speaks the same order/intent
  channel, so a human or a policy can occupy any node. (Bots are just another
  `ControlSource` emitting intents; an order is the goal input conditioning a
  unit policy. NN floats never enter the synced sim — integer intents in.)
- **Sequencing ruling (Erik): keep full-AI teams and human-in-the-loop separate
  in our minds.** Implement 1–2 full-AI architectures *nicely* first; move to
  human-insertion after they work.
- **Commander research stays open** — e.g. timed orders ("breach location X at
  time t_breach") with units adapting to the reward that implies; probably
  combined with demonstrations; survey the published tooling when that phase
  opens (lit-search Thread B / FMH is the anchor).
- No hard global prioritization yet (Erik): tracks + dependencies now,
  natural prioritization when implementation starts.

## Track F — fire & atmosphere joint re-tune (Erik, ACTIVE — excluded here)

Listed only so the map is complete: o2-continuous-law + sky-exchange joint
re-tune, Step-3 heat / hot-gate escalation, ONE golden rebase + HUMAN-TEST,
then push/merge. First thing to complete; Erik is on it.

## Track 1 — combat completeness (LOCKED by Erik)

The rules the bots learn against. Order within the track is loose; the kit
descriptor rider is the one RL-critical constraint.

1. **Movement feel: momentum / earned-sprint arc** — design doc on main
   (`inertia_and_sprint_design_2026-07-30.md`); queued as next arc per Erik.
2. **Weapons & loadouts**: ammo, magazines, reload mechanics.
3. **Damage & unit HP consolidation** (damage types/resists exist from the
   W-wave; unify into one legible system).
4. **Armor**: flat defence + percentual reduction (Dark Souls model) — part of
   the unit-class design pass (`breach_unit_class_design.md`; own pass, per the
   2026-07-22 ruling that units aren't entities yet).
5. **Grenade energy-budget retune** — TODO.md physics rider (2026-07-30): heat
   as primary payload, less static over-pressure, evaluate a radial wind
   (velocity IC) to carry the shockwave. `wave_p` CONFIRMED alive post-EOS
   (rung B kept it a separate blast field) — check u-injection ↔ wave_p
   coupling before tuning.
6. **Enemy roster v1**: enemy marines, zombie variants (runner/brute — TODO),
   first critters per `beastiary/beastiary.md`.
7. **RL rider on all of the above**: every loadout/class exposes a clean
   machine-readable **kit descriptor** — heterogeneous kit breaks naive
   weight-sharing (lit-search caution), so the policy must condition on it.

Chat-sized engine riders that slot into this track (all already in TODO.md):
blast-pressure-threshold material column · door↔unit occupancy rules A+B ·
per-tick wall collision + grenade bounce · fire destroys/converts furniture ·
dust-stirring shockwaves · inventory booleans → `Inventory` · footprint
rotation · unit-system deferrals (modifiers, environment damage, faction table).

## Track 2 — RL substrate (parallel; can start now)

2.0 **Demonstration recorder — FIRST (agreed).** Determinism ⇒ a complete demo
    = seed + per-tick intent stream (kilobytes), replayable into any future
    observation encoding. Auto-on in **every** play mode (gamepad, WEGO, feel
    tests). Every hour Erik ever plays becomes training data. Gate:
    byte-identical session replay from the log.
2.1 **Headless throughput + vectorized envs + determinism canary in CI**
    (two seeded sims, N ticks, assert identical — specified in
    `architecture/ml/01_ml_and_training.md`, never built). Includes the seeded-
    RNG sweep: raycaster fire jitter + any process-global RNG (old TODO items,
    now load-bearing).
2.2 **Observation encoder**: v0 host/NumPy for semantics → CUDA port when
    stable (zero-copy DLPack hand-off; GPUDrive pattern). The frozen §5a
    sensor-gather kernel interface (Arc B rider) is the on-device sibling.
    Erik's "move more logic to CUDA?" → answered by 2.6: measure first.
2.3 **Reward SYSTEM — a real design pass (Erik's emphasis), not a hook.**
    Reward profiles as data/config; composable components; potential-based-
    shaping-only library (Ng/Harada/Russell — the theorem that shaping can't
    create degenerate optima); profile-vs-profile evaluation harness; ablation
    bookkeeping. Own design doc + lesson before code.
2.4 **Legal-action masking** at the order level (`get_legal_actions` is a stub;
    reachable tiles × order type × visible targets).
2.5 **Level generator v1 — entity-aware from day one**: operable doors inside
    generated levels (Erik), later water zones/aquariums. Builds on
    editor/bake/`level_lib`. SMACv2 lesson: procedural randomization is a
    REQUIREMENT before serious training, not polish.
2.6 **The optimize-hard gate before big runs** (already ledger #4): profile the
    whole training loop end-to-end, then attack the worst measured bottlenecks —
    S8b CUDA graphs (parked), batch A*, Python-rules hot spots, more residency,
    possibly more game logic to arrays/CUDA. Measurement-driven, not
    speculative.

## Track 3 — training ladder (R = full-AI research; H = human insertion)

Per Erik's separation ruling: R-phases to a *nice* working state before H.

- **R1** — one unit, goal-conditioned PPO, masked action head; BC bootstrap
  from recordings (GCSL + HER relabeling). Toy version starts on today's
  mechanics (§de-risk below). Success: reliably executes each v1 order across
  randomized rooms.
- **R2** — squad from one brain: parameter sharing + role/ID channel,
  team-spirit reward annealing (MAPPO conventions).
- **R3** — AI commander on top (FMH/RODE band): small discrete human-legible
  order vocabulary; timed orders (t_breach) as a flavor; the abstraction dial
  stays an experimental variable. THE design artifact: the **order
  vocabulary** (each order = semantics + termination predicate + reward).
- **R4** — league / opponent zoo / architecture tournament (Model 1 vs 2 vs 3)
  — the stated research goal, cheap under determinism.
- **H1** — human at a seat: human squad-leader co-op first (bots "useful while
  not getting in the way"; obedience-vs-judgment dial starts obedience-high;
  possibly the human leader still receives objectives from above — open design).
- **H2** — human as a *unit* under an AI commander (rookie fantasy); rank
  progression across a campaign (far; capture only).

**Teaching moments (standing rule):** each R/H milestone opens with a short
lesson doc at point of use (PPO, MAPPO, GCSL/HER, leagues…). Erik's learning is
a stated goal of the project; don't assume RL-acronym familiarity in docs.

## Track 4 — content & world richness (parallel; feeds the training distribution)

- **Premade entity assets + project skills** — the
  `000_read_with_fable_after_level_editor_v1_is_done.md` topic; own session.
  Airlock kit, sentry turrets, terminals (lock/unlock doors, ship lights, read
  logs — mission 1 needs), lamps→entities convergence (TODO).
- **Critters & cages**: the animal pen (vertical-slice element) generalized —
  caged xenomorphs / giant squid the players can release; **aquariums** =
  glass + water volume + squid (needs the water/fluid integration arc —
  prototype `prototypes/fluid_test.py`; brittle vs space-rated glass falls out
  of the blast-threshold material column). Lit-search Thread D (animals) is the
  training program; swarms (MAgent-scale cheap shared policies) are a later,
  different scaling regime (lit-search §6).
- **Mission 1 "Silent Cargo"** (designed in `missions/missions.md`) → unblocks
  the Erik-gated armory grand-tuning session (TODO "Waiting on Erik").
- Campaign/faction meta + narrative systems — far future; capture only.

## Track 5 — rendering & beauty (Erik: separate, motivation-driven, never blocking)

- **B2 smoke-honesty: BUILT, awaiting Erik's HUMAN-TEST** (branch
  `fire-b2-smoke-honesty`; launch cmd + feel items in its design doc §8).
- B-ladder continuation per the living plan (`plan-for-tuning-and-graphics.md`
  bottom; gas-chemistry wish list lives there; standing rule: **no bloom ever**).
- Scorch marks + blood splats (destruction painting — design ready,
  `graphics_lighting_design.md` §7/§7.5) · explosion visuals + transient blast
  light · fire as short-range light source · 1-bounce tinted raycaster ·
  ambient-lighting/two-kinds-of-lights discussion · smoke-vs-unit draw order.
- Animation/appearance: marine visual-profile system · zombies keep victim skin
  + swap to zombie gait · visible weapons + fire/throw anims · move-order
  walk-anim bug · skin/AI-texturing tool eval · GPU skinning past ~20 units ·
  retire the M-toggle/sprite path. Meta-principle (Erik): ML path is the
  priority — weigh every hand-authored animation task by "will ML redo this?".
- **Renderer correctness sweep** — the ~25-item tail stays in TODO.md ("Swept
  from consolidated docs"); one dedicated cleanup session.
- Resolution audit & consolidation (TODO section).

## De-risk plan — substrate milestones that run on TODAY's mechanics

Answers "what concretely can start before combat completeness lands." All
read-only w.r.t. sim behavior; dedicated worktree, parallel to fire tuning.
Weights are disposable; pipeline code is the durable asset.

- **M0 — recorder** (2.0). Gate: byte-identical replay of a real session.
- **M1 — N-env headless harness**: ticks/sec at 64²/128², determinism canary in
  CI. Gate: numbers in a table + green canary.
- **M2 — env v0**: NumPy obs planes (material/solid, self/team/enemy rasters,
  fire/smoke/O2/heat), Gymnasium-style wrapper on the `Simulation` facade,
  reward hook v0. Gate: random-policy rollouts at speed.
- **M3 — first learning run**: ONE marine, `move-to(tile)`, generated random
  empty rooms, PPO via an established single-file trainer. Gate: ≥90% order
  completion on held-out rooms — **plus the first lesson doc (what PPO is and
  why it wins here)**.
- **M4 — demo pipeline proof**: ~20 recorded demos of one order → BC pretrain →
  measurable behavior delta vs from-scratch. Proves the recorder→demos→policy
  loop end-to-end.

## Meta / process loose ends (gathered 2026-07-30)

- **Canon folds pending**: OnePhaseWEGO (+ TwoPhaseWEGO retirement decision) ·
  control-modularity fold (mechanics/04, fix stale 12 Hz table) · post-EOS doc
  consolidation (on the books since 07-05) · `architecture/ml/01` needs an
  intent-era update when the RL arc opens (it's WEGO/AP-era).
- **TODO.md staleness sweep**: whole sections predate reality — pygame→pyray
  migration (DONE), CUDA migration §6–9 (DONE), "One Perfect Level" / "Build One
  Complete Level" (superseded by Arc C + the art-direction retirement),
  playground re-convert (DONE 07-24). Prune with git-history pointers.
- **Branch/worktree hygiene**: delete merged branches/worktrees;
  `.claude/worktrees/arc-b-logic` dir after a reboot; `origin/main` push when
  Erik OKs; `o2-continuous-law` is LOCAL-ONLY pending Erik's word.
- **Repo-based project memory** (Erik's standing wish; design pending his go).
- Process rules that govern all of the above: Fable designs / Opus implements ·
  autonomous-patch-workflow for multi-patch arcs · HUMAN-TEST for anything
  feel-adjacent · every RL milestone opens with a lesson doc.

## Suggested critical path (a suggestion — the ledger ordering is Erik's)

Fire re-tune (F, Erik) → momentum/sprint arc → Track 1 items, with Track 2
M0–M4 running in parallel worktrees → level generator → R1 proper → R2 →
optimize-hard gate (2.6) → big runs → R3/R4 → H1. Rendering (Track 5) and
content (Track 4) interleave by motivation; content earns its keep early
because it widens the training distribution.

## Open design questions (parked, percolating)

Order vocabulary (the load-bearing artifact) · obedience-vs-judgment dial ·
observation honesty / per-species views (bots see what a marine could see) ·
commander timed-order semantics (t_breach) · human-seat UX (does the human
leader get objectives from above?) · reward-profile system shape · WEGO
annotation UX · aquarium/water arc scope.
