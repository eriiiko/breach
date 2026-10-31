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
- 2026-10-31 (P6c, worktree agent, Opus) -- the C++ `ArmChainSolver`
  (`cpp/src/arm_chain_solver.*`, `breach_physics.arm_chain_tick`, Python side
  `src/simulation/arm_chain.py`): stateless over caller-owned buffers born
  (N, A, n, 3), pure integer, 0/0/0 on the ratchet, held at TOL 0 to
  `chain_ref_q` after every tick -- reach / dead arm / four wall scenarios /
  a synthetic 8-arm animal at 1.80, 2.52, 3.39 m (10 s each behind the long
  flag) plus P5z's four traces replayed: all equal; it passed on the first
  build, because the reference had already fixed every rounding. Reference
  FIRST: static-topology contacts (nodes vs `GameMap.solid`, face of least
  penetration toward an open neighbour, tangential friction, normal velocity
  clamped at a face; the crown's disc vs blocked tiles and footprints),
  per-tick outputs (contacts, anchor / grab force), G8 (inert on an open map:
  bytes identical; flat throw: 0 counts in, 0 counts of slide). What failed
  first: (1) CHECK at 3.39 m on a tip folded into a corner put `bend.W` at
  63 bits -- an in-plane projection near 0; the bend now skips a plane below
  the door's l_min (the assumption W_fit always made); G3's COM moved 0.041
  -> 0.024 mm from it, nothing else; (2) the honest `dist_impulse_fit`
  bound (k C, L^4) caps arms at ~4.2 m (the P6a window said 12.7 m: it was
  bounding a factor no code forms) -- a pre-shift can widen it if #112 wants
  bigger; (3) scenario set-ups that start nodes inside walls explode -- size
  rooms by the row. Skipped: no CUDA twin (recorded debt); report writing
  by a subagent's Write is refused by the harness, so `report_p6c.md` is the
  generated gate table and the tol-0 table lives in the test's docstring.
  Learned: a lockstep harness that packs the reference World's INPUTS each
  tick (never its state) catches a 1-count dial change at tick 0. Cost: one
  session; full gates 7 min, long tol-0 ~2.5 min x 4 processes; the C++ is
  ~2 ms per octopus-tick (P6e measures properly).
- 2026-10-31 (P6d1, worktree agent, Opus) -- the LIMP octopus in the shared
  sim: `src/simulation/creature_system.py` (`CreatureBody` owns the solver's
  buffers + the hub's integer state, own xorshift, elapsed-tick counters;
  `CreatureSystem.tick` = slot 5b, ONE solver call per living creature in
  unit-id order, the pose written back through `creature_fixed`; the ONE
  serializer `serialize_record` / `decode_record` of the `__creature__`
  section and the recorder ring), `[[creature]]` rows (`level_lib.
  write_creatures`, the loader's door incl. a rest-pose-arms-on-open-floor
  check, `Simulation.spawn_creatures`, match.toml), writer routing (slot 3,
  OnePhaseWEGO, the AIM facing write and the round-end snap skip a creature;
  the wave push is a body impulse through the coupling row
  `apply_creature_wave_impulse`, vetoable), `GameMap.topology_serial` (the
  sim-side twin of `render_dirty`, so a body re-reads its walls). Gates
  (`tests/test_creature_body.py`, 18; `test_cuda_creature.py`): released
  0.2 m up, every node on the deck at 2.21 s, never below it, last-second
  motion 0.34 mm/s, free-node velocity 0.58 mm/s; CPU / `--cuda` /
  `--resident` identical 48 ticks (field + tick digest + section); a planted
  one-count flip located at its arm / node / axis; WEGO round boundary ==
  continuous; pickle round trip; humans byte-identical with the octopus in
  the room; golden + ship_demo pin unmoved. What failed first: (1) the
  "speed < 1 mm/s" bound on STORED velocity fails forever on the two root
  nodes -- they carry a standing ~5 mm/s velocity the attachment cancels
  every substep (their motion is 0.3 mm/s): the reference's attachment +
  rebuild behaviour, held at tol 0 by P6c, so the gate now bounds MOTION (all
  nodes) and stored velocity (free nodes) and reports the root ghost --
  worth a look in P6e or before P6d2's anchors read root velocities; (2)
  `obstacles` does not carry unit stamps (a unit is permeability 0.5), so
  "only the crown's stamp differs" is asserted on `dyn_permeability` /
  `dyn_heat_atten_q` / `dyn_wave_absorb`; (3) the worktree's copied
  `cpp/build` and `cpp/build_cuda` caches pointed at other checkouts and
  predated P6c: rebuild both first (`build_cpu_home.bat` uses --fresh; move
  the CUDA dir aside). Skipped: the Simulation as a whole does not pickle
  today (its module handle and the C++ PhysicsEngine), so the pickle gate
  round-trips the creature system mid-run; the `creature` controller (P6d2);
  `hit_tiles` (P6d3). Learned: two pre-existing tests pinned the coupling
  table's exact rows / index -- a table designed to grow; rewritten to the
  property. Cost: one session; ~2.1 ms per solver call, ~3.7 ms per tick on
  a 30 x 30 flooded room vs 1.1 ms without the octopus (rough, P6e measures).
- 2026-10-31 (P6d2, worktree agent, Opus) -- the PRIMITIVES in the shared
  sim: `src/simulation/creature_primitives.py` (the ONE module owning the
  state machines: idle -- now the default --, crawl, swim, reach, grab;
  `PrimDials` quantizes the row once; `hub_step` before the one solver call,
  `after_solver` after it), the goal table (none -> idle, MOVE_DIR -> crawl,
  at move-with-cover -> umbrella, at sprint -> jet train, AIM -> reach, USE
  -> grab a creature-owned PROP, solver body 1), `CREATURE_SECT_V2` (gait,
  swim clock / angle / rig, reach goal, per-arm aim + stretch, the steering
  ring, the props), `CreatureBody.set_limp` (the limp gate now asks for it),
  the `creature` match controller (crawl at the nearest unit seen, else
  idle; match.toml `creature_controller`), `tests/_creature_scenarios.py` +
  `tests/test_creature_primitives.py` (17; the full P2-P5 runs behind
  BREACH_OCTOPUS_LONG_GATES=1, ~4 min), the CUDA lockstep gained a 96-tick
  crawl (identical), `tools/octopus_clips.py` -> `out/sim_*.gif`. Sim vs
  P5z: crawl 0.55-0.60 m/s (0.58-0.61), 3-9 deg off (1-3), Levy r 0.79-0.87
  (0.87-0.94, the known miss, lower), heading drift -27/-25/+25/-69 deg
  (P5z +9/-1/+25/+59, both "fail" 30 -- reported); umbrella 0.63 m per
  stroke (0.57), linear control 4 % (7 %); one jet 7.6 mantle lengths, 4.0 s
  decay (7.8, 3.8); jet train 4.01 m/s = 7.3x the crawl (3.95, 6.5x); grab
  contact 0.42 s at 0.59 L (0.29 s, 0.53 L), 4 sealed 0.47-0.65 L (5), 0
  let go, prop lifted 0.21-0.51 m (0.20-0.35), loaded crawl 110 % of free
  (112 %); idle rows 1/3/7/13 pass on two seeds. What failed first, each
  understood before anything moved: (1) the umbrella swam 60 deg sideways --
  the deck-to-cone blend mapped the arms' places around the swimming axis
  asymmetrically (psi_a = phi_a), a net yaw torque; psi_a = phi_a - pi/2 is
  the same eight places, mirror-symmetric; (2) idle row 3 at 68 % -- the
  wave started in the same tick as the root's aim swing, stiffening the arm
  before it bent: the prototype started the wave a tick later; ported
  faithfully, 86-87 %; (3) the clocked rhythm control showed NO line --
  the 40 s crawl hit the room's wall; the prototypes crawled on an endless
  deck: size rooms so no run reaches a wall (rhythm in a 50 m room); (4) the
  single jet carried 3 mantle lengths -- it fired while the arms were still
  blending in from the deck pose; measured from the swimming pose, 7.6. Gate
  DEFINITIONS changed, reasons in the test docstrings: the rhythm gate is
  the median of three seeds (seed 7 12.2x at 0.85 Hz, one 0.2x over; seeds
  1/3 7.0/5.6x; the clocked control 24.2x, P5z 77.5x), idle row 8's counted
  parts over two seeds (one seed left R3 unused, p ~ 0.07; close starts 1/23
  and 3/25 vs P5z 2/24). Finding: the 2-tick comb (the design's 0.1 s) is
  1.5x the float prototype's effective comb (it decided every 3 ticks, its
  float accumulator overshooting 0.1 s), so p_recruit / p_start per decision
  land 1.5x more often per second -- the rhythm and de-synchrony spread is
  partly that; the dials are the row's and were NOT re-tuned (P8's eye).
  Choices (vetoable): swim by speed mode (no new intent), no swim steering
  (along the heading), a sucker grips only a node the contact report put on
  the FLOOR (a crawl never anchors to water; idle seats an arm when it
  lands), a seal lets go above `grab.grip_limit` (the prototype never
  popped grabs), RECOVER relaxes from the elongation at release (the
  prototype jumped to e_max after an early pop), the deck<->swim root blend
  over one closing time. New row dials (each a literal in the prototype):
  `swim.root_ring_frac`, `grab.wrap_s`, `grab.prop_c_drag`,
  `grab.prop_area_m2`. Root-velocity ghost: no primitive reads node
  velocities; measured, the root nodes' stored velocity is within 2-7 mm/s
  of the crown's rigid velocity while it idles / crawls at up to 0.55 m/s
  (0.02 mm/s limp) -- harmless, left to P6e. The GIFs show: a seated
  octopus with one or two arms reaching out and settling (idle), a
  crawl with one to three red arms pushing toward 70 deg, three umbrella
  strokes in a cone then a bundled jet train from mid water (side view),
  the reach / wrap / lift of a 116 kg prop held at the crown while the other
  arms crawl. Cost: one session; the sim runs ~3 ms per tick on a 30 x 30
  flooded room, ~10 ms on 150 x 40 (mostly the room's physics; P6e measures).
- 2026-10-31 (P6d3, worktree agent, Opus) -- the HIT SEAM: `Unit.hit_tiles()`
  (+ `in_hit_tiles`, `note_hit`), the footprint for every unit but a
  creature, whose `CreatureBody.write_unit` rebuilds a derived `HitShape`
  (footprint + every in-map tile an arm segment crosses, sorted, not
  digested) at spawn and every slot-5b pose write. The rasteriser is
  `creature_fixed.segment_tiles`: an exact-integer supercover walk over the
  Q24 chain in the solver's own tile law, crossings compared by
  cross-multiplication, a corner pass stepping diagonally. Wired into
  exactly the bullet march and the laser skewer; a connecting hit writes
  the struck (arm, node) to the hashed `last_hit` ((-1, -1) on the crown).
  Sites classified and LEFT on the footprint: melee target + adjacency, the
  zombie bite (centre distance), every exchange row incl. the blast
  (centre distance), stamps, doors, the spray's nozzle exclusion, the
  solver's unit rects, vision's silhouette, the UI's click-picking. Choices
  (vetoable): z ignored (an arm mid-water is hittable from the deck), a
  crown hit writes (-1, -1) so a renderer pairing `UnitHitEvent` with
  `last_hit` never shows a stale node, node k owns its tile and its
  segment's corner clips (first claim in (arm, node) order). Found: in the
  idle pose the eight seated arms ring the crown so tightly that no march
  path passes within 3 tiles of the footprint without crossing an arm --
  the "crown-side miss" gate is a near miss beside an arm inside the reach
  (4 tiles from the crown), found by a deterministic search. The melee gate
  cannot see a target-only regression (adjacency is a second footprint
  test). Cost: ~0.3 ms per creature per tick in Python (P6e measures).
- 2026-10-31 (P6e, worktree agent, Opus) -- the COST BENCH:
  `tools/bench_octopus.py` (+ `--chain-micro`), the report
  `docs/octopus_cost_2026-10-31.md`. Slot 5b got a part timer,
  `creature_system.set_tick_laps`, module-level and OFF by default; with
  the timer on and off the digests were identical in 10/10 scenarios on all
  three backends. Numbers, reported, no gate: one octopus = 2.5-2.7 ms per
  tick on every backend (6 % of 41.7 ms). The chain is 2.0 ms (80 %; 0.55
  ns per weighted op against P6c's 3.6 M, i.e. half the "1 ns/op" of the
  suspended 0.2 ms ceiling), the hit shape 0.34 ms (13 %, Python, rebuilt
  every tick), the hub 0.13-0.24 ms. The cost is exactly linear in animals
  (8 = 19.9 ms). Batching animals into one call gains 0-1 %; the chain is
  linear in substeps (12 = 0.75x, 8 = 0.5x). Learned: (1) a CUDA process
  slows from run to run (#125's repeat slowdown: per-call 10.7 -> 17.1 ms
  by the third run), so GPU A/B timings need separate processes
  (`--no-baseline` / `--baseline-only`); the first same-process layout
  charged up to +7 ms of drift to the octopus. (2) On CUDA, long host work
  slows the device share of the step (+1.3-2 ms per-call with one octopus,
  +9 ms resident with eight). (3) The v1 horde bites an octopus that shares
  its room dead by tick 43, and a bot shoots it at tick 590; the ship bench
  runs a standing horde. (4) gen_ship seed 1 medium fits the rest pose in
  one room only. Levers for Erik, none recommended: a lazy or C++ hit
  shape (-0.3 ms), substeps (a modelling call, gates re-run), threads over
  animals or arms. Cost: one session, ~1 h of machine time for the three
  backends.
