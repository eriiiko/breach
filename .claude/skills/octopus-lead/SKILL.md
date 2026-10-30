---
name: octopus-lead
description: Claude's standing role as lead designer and developer of Breach's octopus (milestone 8) — the mandate and its limits, what has been decided and why, the control architecture (goal → primitive → pose, coarse integer chain for gameplay), the capability tiers, the research rules, the soft values behind the work, and where everything lives. Load at the start of ANY session that touches the octopus, its animation, its arm brains or any later multi-limbed creature, before reading the code.
---

# Octopus lead — continuity skill

**Status: PROVISIONAL** (created 2026-10-29, session 1 of the animation study).
Every session that uses it amends it before merging: what moved, what was
decided, what was learned. A step is a MUST only with its incident beside it.

## 1. The mandate (Erik, 2026-10-29)

Erik appointed Claude **lead designer and developer of the octopus**: "you are
the lead so you decide all these things." Concretely:

- **Claude decides**: scope and order of the octopus work, the control
  architecture, what is prototyped and how, which option is recommended, how
  issues under milestone 8 are cut, what goes in this skill.
- **Erik decides, always**: the final verdict by eye on anything feel-adjacent
  (the HUMAN-TEST gate is unchanged — he plays it before merge), gameplay
  verbs that touch the rest of the game (eating a marine, console use are
  tier-3 verbs *he* ratifies), physics-modelling calls that reach into the
  shared sim (per the master CLAUDE.md split), and budget (effort level,
  Fable vs Opus agents).
- **Default**: when unsure whether a call is Claude's, make it, say so in the
  report, and let Erik veto. Record rulings scoped to their case.

## 2. The goal, in Erik's words

- "I want to put the bar so high that when a player sees the octopus in
  action, they'll be surprised how good it is. The goal is to make them
  amazed." Quality first, no rush; "everything is up for experimentation".
- The octopus is a top-quality monster for three uses at once: RL training,
  play, and development. It is intelligent (it will use consoles one day).
- It is the first non-humanoid animation skeleton in the game and the
  foundation for every multi-limbed creature after it (spider, worm, larvae):
  what carries over is the CONTROL ARCHITECTURE and the integer chain, not
  the bones (an octopus arm is a muscular hydrostat; a spider leg is jointed).
- Erik has never animated anything; explain the craft as we go, language
  first, code second.

## 3. Decided (with dates)

| Date | Decision |
|---|---|
| 2026-10-26 | Build on Codex's curve-driven arm proof; procedural Blender with its own skeleton; Krea only for concept/texture. |
| 2026-10-27 | Whole animal built (`octopus-v1`, `eca6758`); CONTROLS.md = 7 floats/arm + 6 body floats, independence-tested. Erik: "over all expectations". Refinement parked as #110 until seen moving. |
| 2026-10-27 | 11 skin patterns kept (#113); favourites zebra, red_whitespot, ghost; red = possible berserk flush. |
| 2026-10-28 | Size B = mid (1.4 m body), arm ratio 1.8 (~2.5 m arms, 3.3 m stretched); C kept as "big". Sizes eventually generated through `src/simulation/species.py` (an OCTOPUS SpeciesDef). Must squeeze through human doors (mid mantle 0.93 m → 72 % for 2-tile doors). |
| 2026-10-29 | **Architecture hypothesis: goal → primitive → pose.** Brain (later Tsetlin/NN, #108) issues per-arm goals; a primitive layer turns a goal into control curves over time (the real animal's stereotyped primitives: reach = bend wave base→tip, fetch = pseudo-joints, crawl = elongation pushes, jet = mantle + umbrella); the pose layer is CONTROLS.md, render-only. |
| 2026-10-29 | **Capability tiers confirmed.** T1 (the study): swim stroke that really travels; separate crawl gait; every arm independent and purposeful; reach → grab → hold → lift a marine and carry it while moving; idle "alive" (arms exploring, tip curls, mantle breathing). T2 (#107/#108): doors, pulling over obstacles, several marines, fighting (strike, throw, pin), berserk flush. T3: consoles/levers/doors as reach+grab plus a "use" verb; eating/killing a held marine. |
| 2026-10-29 | **Game-state split accepted.** Sim owns, in Q16, per octopus: body position/heading, mantle state, per arm a COARSE chain (root, 2–3 waypoints, tip) + goal + grab (unit, arm). Render owns the 17-joint curve, stretch thinning, breathing, sucker roll. Grab, blocking, reach range and hits are decided on the coarse chain; the float rig passes through its waypoints so the picture never lies. Governing rule (egregore, 2026-07-20): no float (IK/ragdoll/neural) ever becomes synced state. |
| 2026-10-29 | T1's held marine is a carried marine-shaped PROP for the study; the real unit coupling (wading, struggling, shootable while held) is the end goal and is #107's. |
| 2026-10-29 | **The octopus is an ASSET, not a class** (Erik: "I agree completely"). A versioned creature definition as DATA (`OctopusDef`, the first row of a creatures table at P6: model, sizes, skins, primitive vocabulary, chain parameters, brain), on the shared systems (chain solver, primitive executor, hub) that every later creature also uses; versions are rows + git tags, old versions REACHABLE in history, never playable side by side (no live code path per version); fidelity tiers are explicit, one live implementation each (sim coarse chain / render curve / numpy research prototypes, optionally an offline Cosserat calibration tier under `prototypes/`), talking through the game-state split. No `Octopus` subclass, ever (the Unit rule). |
| 2026-10-30 | **P6 design session rulings (Erik).** (1) COST IS NOT A DESIGN INPUT: build the best octopus first, measure, optimise only after the final result is seen ("rather one fantastic octopus than ten meh"); the 0.2 ms ceiling is suspended for P6. (2) z IN THE CHAIN: absolutely. (3) Integer trig is the kit's own `atan2_q16`/`sin_q16`/`cos_q16`/`sqrt_q16`; a LUT with a few percent error is blessed if atan2 ever binds. (4) **Claude is CREATURES LEAD** as well as octopus lead: how creature DATA is organised is Claude's call. Lead's decision: `[creatures.octopus]` in config.toml (where materials, gases, weapons, loadouts all live; splitting config.toml is not this arc's question), read by one door `src/simulation/creatures.py` -> `CreatureDef` (body, tissue, primitives, limits, solver), the first row. |
| 2026-10-30 | **The octopus is a UNIT** (Erik: "definitely"): `Unit(species_id="octopus")`, the one class; HP/team/events/digest unit hash for free; arms never block; its position is written by the creature body (the arms move it). P6 design doc drafted: `docs/octopus_p6_sim_port_design_2026-10-30.md` (four layers: row -> Unit -> CreatureBody [Python hub + C++ Q16 ArmChainSolver] -> render accessor; CREATURE_SECT_V1 digest record; slot 5b; patches P6a-e; CUDA twin deferred as debt; cost measured, never gated). |
| 2026-10-30 | **P6 design v2 after the Opus critique** (23 findings, every checked claim true -- the critic earned its budget): see `docs/octopus_p6_critique_2026-10-30.md` and the doc's section 10. Lessons: an entity-section record cannot carry a unit's state (carrier gates on n_entities); Q16 positions + PBD velocity rebuild erase buoyancy gravity -> fine chain currency Q24; never form alpha~ or call reciprocal_q16 in XPBD -> stiffness form with exact 1/dt^2; floor rounding breaks the no-slide guarantee; occupied_tiles() is the footprint seam for stamps/doors/exchange/melee -> a separate hit_tiles(); prototypes ran at 0.24 m and 2-D only -> a float game-scale 3-D step (P5z) before any integer work. OPEN: team/faction (Erik). |
| 2026-10-30 | **Team ruled (Erik)**: the octopus is its OWN team (`CREATURE_TEAM`), hostile to all, never the zombie team; hazard in a match (not counted for victory). IDEA, not order: a hostility matrix between teams for different animals. **Arms vs marines**: UNDESIGNED on purpose -- researched together with the marine interaction after P6; v1 arms pass through marines. Erik: press on through P6 without stopping for a session break; the viewer must be amazed again. |
| 2026-10-29 | **P1 PASSED (7/7)** *(lead)*: an 18-node XPBD chain + Yekutieli's stiffening wave + anisotropic quadratic drag reproduces the real reach (`prototypes/octopus_motion/`, `out/report.md`). The recommendation stands: option 2, the chain. Findings that bind P6: straightness behind the bend is set by SOLVER RESOLUTION, not muscle force (8 x 1 fails at any k_bend; 16 substeps x 2 passes passes; 32 x 1 whips the tip); drag is load-bearing (no drag = 300 deg tip overshoot); the weighted op count is ~0.6 M per octopus-tick, 3x over the 0.2 ms ceiling at 1 ns/op -- P6 measures before anything is promised. |

## 4. Research rules (Erik, 2026-10-29)

- **Research agents run SEQUENTIALLY, never in parallel** — parallel agents
  have hit the 5-hour token wall and lost all five results. One agent at a
  time, writing its digest to disk incrementally.
- **Full texts before building.** Every paper we rely on is read in full; a
  digest states per paper whether it was full text or abstract only. Paywalled
  papers: give Erik the DOIs/titles — he fetches them through Lund University.
- Reference footage of real octopuses is the judging target for every clip.
- Strands of the 2026-10-29 search: (a) octopus motor biology; (b) soft-arm
  simulation (Cosserat rods, mass-spring/verlet, real cost); (c) games & film
  procedural tentacles (what shipped); (d) CPGs for multi-limb locomotion.
  Digests land in `docs/research/octopus_*_litsearch_<date>.md`.

## 5. Where things live

- Branch/worktree `33-octopus-concept` (`.claude/worktrees/33-octopus-concept`).
- Model + scripts: `art/characters/octopus/` — `CONTROLS.md` (the control
  contract), `scripts/octopus_math.py` (pure pose math, Blender and runtime
  share it), `runtime_octopus.py` (live posing in the game), `verify_octopus.py`
  (independence gate), `pattern_math.py` (skins), `previews/`.
- Planning: GitHub milestone **8 - Octopus** (#104 umbrella, #105 animation
  study, #106 body plan, #107 grab, #108 arm brains, #109 arena, #110
  refinement, #111 aquarium, #112 sizes, #113 skins). Erik wants planning on
  GitHub — post agreed plans as issue comments, not only in chat.
- Memory: `project_octopus_creature.md` (the session log; this skill is the
  distilled version).
- Prior art in-repo: `docs/procedural_animation_brainstorm.md` (2026-07-17),
  `docs/research/ml_animation_litsearch_2026-07-20.md` (humanoid, not arms).
- Study brief v1: `docs/octopus_animation_study_2026-10-29.md` (synthesis of
  the four digests + nine full texts, the four-layer architecture, the
  prototype plan P0–P8). Reading notes: `docs/research/octopus_papers_
  reading_notes_2026-10-29.md`; PDFs `docs/papers/octopus/` (README index).

## 6. Procedure for a session

1. `git fetch origin && git status` in the worktree; read §3 of this skill,
   then #105's latest comment, then memory.
2. Say which tier/issue the session serves. One system at a time.
3. Research: one agent, digest to disk, full-text ledger, then into
   `docs/research/`. Build: pure core + thin shell, every clip rendered on the
   real mid-size model, cost per octopus measured (`runtime_octopus_bench`).
4. Feel-adjacent → HUMAN-TEST; Erik looks before merge.
5. Close: post the result on the issue, update memory, amend this skill.

## 7. Soft values heard along the way

- Erik is excited about this collaboration and considers the octopus "one of
  the most important aspects of the game"; he is willing to put lots of effort
  in. Match that: no shortcuts that would show on screen.
- Every arm must look like it has a mind of its own; "no arm just follows
  along."
- Forward travel comes from the stroke itself, never a slide added on top.
- Erik likes to understand before building; keep him in the loop on the
  animal (what it does) and take the computer-science calls yourself.

## 8. Run log

- 2026-10-29 — skill created in session 1 of #105. Four research strands run
  SEQUENTIALLY (Opus, high; ~250–325k tokens, ~18–20 min each), digests
  committed as they landed; Erik fetched nine papers the same day; the lead
  read all nine in full and wrote the study brief v1. Decided *(lead)*:
  XPBD 18-node chain per arm in sim (not a Cosserat rod, not kinematic
  cycles); crawl recruitment by seeded integer lottery, no oscillator; swim
  on one phase clock; per-arm brain interface = fixed integer observation →
  (request, score). Next: P0 footage sheet, then P1 the one-arm float
  prototype (the make-or-break: does a chain + stiffness wave reproduce
  Yekutieli's reach?). #118 (per-arm HP) filed. Cost: one session.
- 2026-10-29 (session 2) — P0 + P1 done by the lead inline (no agent: a
  ~600-line pure-numpy prototype is cheaper to write than to brief). P0:
  `docs/octopus_footage_sheet_2026-10-29.md` (watch-list, scoring sheet;
  nothing watched yet — Erik or a later session fills timestamps). P1: 7/7
  gates on the first honest run after two fixes found by LOOKING at the
  figures before tuning: (1) the Rayleigh damping was parametrised as a
  fraction of k and killed all motion (now a damping time, Yekutieli's muscle
  has ~7e-5 s); (2) the start pose must be the paper's (one bend near the
  base, distal arm lying back) -- a tip-weighted curl gives no travelling
  bend. Three gates failed at 8 x 1 and were NOT bent to pass: the sensitivity
  scan (k_bend x4, substeps, passes, c_tan, passive, damping) showed solver
  resolution was the cause; two gate DEFINITIONS were corrected with the
  reason written in the report (a3 measures the END overshoot, the paper's
  meaning, and reports the mid-reach S-flex beside it; a4 gates the natural
  t_act range 0.6-1.2 s the paper names). Lesson: run the sensitivity scan
  before touching a gate. Cost: one session, ~2 min per gate run. Next: P2
  crawl (eight chains + body + floor + anchors + lottery).
- 2026-10-29 (session 2, continued; Erik: "go ahead as far as you can
  manage") — P2, P3, P4, P5 built and gated the same day, all inline, on a
  batched arm bank (`arm_bank.py`, bit-identical to the P1 chain, the
  SIMD-over-arms the port will use). Results: P2 5/6 (Levy's per-push r
  0.88-0.94 vs 0.9, left as the miss), P3 3/3, P4 3/3, P5 5/5 measurable
  rows. What was learned: (1) a background Bash with `&` dies with the tool
  call -- use the tool's own run_in_background; (2) Python's default file
  encoding on Windows is cp1252: write prototype files with
  encoding="utf-8" or a `§` breaks the interpreter; (3) every gate that
  failed first was understood before anything moved: the rhythm test
  needed a CLOCKED CONTROL to be able to fail, the crawl needed closed-loop
  steering to survive a dead arm, the idle needed the seated arms ANCHORED
  (a reach's reaction moved the whole body), the swim's control needed
  linear AND isotropic drag on arms and body; (4) a gate definition is
  corrected only with its reason in the report (a3, a4, P2 gate 1, P4 gate
  2, P5 rows 1 and 7 -- each one named). Cost: ~1 min of wall per 25 s of
  eight-arm sim at 16 x 2; a full gate run 2-8 min. Next: P6 is the port
  into the shared sim -- a design gate with Erik first (cost risk 3x, z in
  the chain, the OctopusDef row), then P7 render, P8 clips.
- 2026-10-30 (P5z, worktree agent, Opus) -- the float game-scale 3-D step:
  `arm_chain3d.py` (Bishop frame transported without trig, two signed bends
  per node = the P1 constraint in each frame plane, floor with position-based
  Coulomb, g_eff, a 1025-sample activation table), `arm_bank3d.py` (bodies with
  z, views that let the 2-D hubs run INHERITED), the four 3-D sims, `game_scale.
  mid_row()` -> `out/row_mid.toml`, `gates3d.py`. 33/35: P1 at mid scale
  reproduces the 2-D numbers digit for digit (the law is exact similarity
  where gravity does not enter; 3-D == 2-D in a plane to 1e-14, tilted plane
  to 1e-13), table == tanh, lift works (116 kg prop at 0.2-0.35 m, 0 pops),
  dead arm sinks and rests, free-animal COM 2.5e-13 m. Failed: P2 r (the known
  0.87-0.94 miss) and NEW P2 heading drift (+59 deg on one seed; a 3-dial
  sweep -- mu x S, the law's body, yaw damping x10 -- showed seed-dominated
  wander, nothing adopted). No dial tuned, 16 x 2 kept. Learned: (1) the
  gates' wall was the risk (3-D ~2x the 2-D per tick): each scenario its own
  process (25 jobs, 14 workers) took the run to ~4.5 min -- batch independent
  arms in one bank where a run has many (P1's 100 arms in one), parallelise
  scenarios otherwise; (2) a "symmetric drag" for the no-slide gate must be
  uniform-RATE (mass-proportional): with the real quadratic drag an idle
  animal swims 3.8 m/min by drag nonlinearity; (3) the dead curled arm
  unfurls 2.5 m across the deck on its passive stiffness at game scale (mu g
  is 10x weaker relatively) -- #118 should give dead arms their own pose as
  the passive target; (4) SIZES draws the arm ~1.9x thicker than the scaled
  real arm and the body 510 vs 347 kg -- Erik's ruling before P6a.
- 2026-10-30 (P6a, worktree agent, Opus) -- the creature row and its door:
  `[creatures.octopus]` in config.toml (row_mid.toml's values, provenance
  condensed; `footprint = 3` added, `solver.ticks_per_second` dropped -- the
  clock has one owner), `creatures.py` (CreatureDef, the schema IS the
  dataclasses' field metadata, so an unknown key is refused for free),
  `creature_fixed.py` (Q24 chain currency, Q16 mass, Q32 inverse mass, Q16
  stiffness, Q0 rate -- 1/dt^2 = 147456 exact -- and ten named headroom
  bounds), the octopus species reading mass / footprint / rules from the row,
  `armed` and `can_become_zombie` honoured, `CREATURE_TEAM = 254`. 56 property
  tests; full suite 4083 passed / 33 skipped / 5 xfailed in 5 min 52 s,
  golden untouched. Learned: (1) under the law k W is SIZE-INVARIANT (k_dist
  L^3 x w L^-3, k_bend L^5 x W L^-5), so the stiffness bounds refuse typos,
  not sizes; sizes are bound by the tip mass (L^3), the bend's W (L^-5), the
  body's inverse-mass precision (L^-3) and the +/-128 m range -- the accepted
  window is arms 0.36-12.7 m, and the REAL 0.24 m animal is refused (W_fit;
  its tip is 0.75 Q16 counts); (2) the wave travel times want SUBSTEP
  quantization beside ticks (the activation clock runs in substeps); (3)
  `bend_damp_s` is a 1 ms time constant -- quantizing it to ticks would make it
  41 ms; it enters as gamma in Q16. Nothing failed that needed a second run.
- 2026-10-30 (P6b, worktree agent, Opus) -- the integer reference:
  `docs/octopus_integer_reference/chain_ref_q.py` (+ gates, README,
  report_p6b.md), the kit's Python twins `src/simulation/fixedpoint_twins.py`
  (+ six `fp_*` bindings, held to the C++ by `tests/test_fixedpoint_twins.py`),
  the checked-in `cpp/src/activation_table.inc` (`tools/
  gen_activation_table.py`). All seven gates pass: P1's a1-c on ints with the
  float's numbers (a2 2.45 deg, a4 1.03-1.28), dead arm rests at 0.02 mm/s,
  free-animal COM 0.04 mm / 60 s (its floor-rounding pair slides 6 mm in 10 s),
  reach RMS 0.57 mm vs P5z (0.01 mm once t_act sits on the substep grid),
  bytes deterministic, no int64 site past 59 bits at 1.80-3.39 m; cost
  REPORTED (~28 k weighted / arm-substep, +12 % over the float's bound). What
  failed first and what P6c must therefore copy exactly: (1) a PBD rebuild
  v = 384 (p - p_prev) on Q24 positions erased every sub-count correction --
  the passive tip, a 510 kg body's share of an attachment -- and the dead
  arm's slide ended 77 mm off the float: the reference carries the predict's
  rounding remainder AND a Q32 fine remainder of every correction into the
  velocity (`_rebuild`, `_move`; scratch, not state) -> 15 mm; (2) P6a's
  per-node distance factor f = k w / r rounds each node's share separately
  (1e-7) and slid the free animal 1.1 mm / 60 s: two-body corrections move
  each participant by w x ONE shared impulse -> 0.04 mm; (3) a body
  remainder never reset re-injected itself every substep (crawl body 215 mm
  off -> 5 mm): reset EVERY scratch remainder at the predict; (4) heading in
  Q16 rad is 118 chain counts at the crown rim: the reference carries Q24 rad
  (a design-5 state change P6d1 adopts); (5) the 3 mm reach deviation is the
  t_act quantization to whole substeps (346 for 0.9 s), not arithmetic.
  Lessons: measure a deviation against the float re-run at the int's
  quantized INPUTS before hunting arithmetic; a gate that is a property of
  exact arithmetic (mirror symmetry, ballistic identity) catches a biased
  rounding in one tick where the 60 s no-slide gate needs minutes; Python
  ints run ~2.4 ms per arm-substep, so parallelise scenarios.
