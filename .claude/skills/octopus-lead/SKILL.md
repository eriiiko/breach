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
