# Procedural skeletal animation — investigation brief

**Status: BRAINSTORM / investigation (2026-07-17).** Erik's ask: can we have
procedurally generated animations? What do we need? One humanoid skeleton
reused across many animation sets; exotic skeletons too (spider, if possible
an octopus — and the beastiary wants a gråsugga and an armored quadruped).
Big project, lots of effort budgeted, slotted somewhere after physics v1.
Captured from `notes_2026-07-17_topics_backlog.md` Topic 3.

---

## 0. Where we start from (surveyed 2026-07-17)

- Units are **static raster sprites**: 32×32 PNGs, marines with 8 facing
  directions (`renderer/sprites.py:41-123`), zombies one still each, drawn
  with `rl.draw_texture_pro` scaled to the unit footprint
  (`renderer/overlays.py:471-537`). Circle fallback if the texture is missing.
- **There is no animation system at all** — no frames, no tweening, no walk
  cycles. The only motion is the 8-direction sprite swap
  (`src/simulation/unit.py:344-365`), sub-tile position interpolation, and a
  static prone rotation.
- **No bones/IK anywhere** in code or docs. Closest relative:
  `breach_unit_class_design.md` §"articulated occupancy" designs a tile
  segment-chain for snakes/worms — a *gameplay footprint* model, explicitly
  not a visual rig, but a future consumer of the same spine.
- View is **true 90° top-down**; `unit.facing` is already continuous float
  radians — the 8-way snap exists only because the sprites are stills.
- Renderer is pyray/raylib; ~tens of units on screen, 48 world-px/tile,
  humanoid footprint 3×3 tiles (≈1 m²).

Two consequences fall out immediately:

1. **The blank slate is an advantage.** There is no legacy frame-animation
   system to fight; procedural + skeleton can be THE animation system, not a
   layer on top of one.
2. **Determinism: split by creature class, don't blanket-rule it** *(revised
   2026-07-17 after Erik's pushback — the first draft said "render-only,
   always"; that's wrong for part of the menagerie)*:
   - **Humanoids (start here — animate the marines first):** poses have no
     gameplay effect, so they are render-layer only. Floats fine, no integer
     treatment, sim state flows in and never back. This tier is
     determinism-free by construction and is the right first project.
   - **World-interacting bodies (octopus, long worms):** Erik wants their
     animation to *possibly interact with the world* (arms that reach/grab,
     bodies that occupy space). For these, do a **determinism cost
     evaluation** before designing, with two candidate routes:
     (a) *split spine*: gameplay interaction runs through the sim-side
     articulated-occupancy segment chain (`breach_unit_class_design.md`,
     integer, synced) and the visual rig only decorates it — animation stays
     cosmetic, interaction stays deterministic; or (b) *deterministic rig*:
     make the pose solver itself fixed-point — note the Q2-lift already
     shipped `atan2_q16`/`sin_q16`/`cos_q16` in `cpp/src/fixed_point.h`, and
     FABRIK/2-bone IK are mostly mul/add/sqrt, so integer IK is plausible,
     just not free (goldens, gates, per-arch verification). (a) is the cheap
     default; (b) only if a creature's *visual* pose must be authoritative.
   - Special solutions per creature are acceptable (Erik) — this is a
     per-skeleton decision, not a framework-wide one.

## 1. What "procedural animation" means here

A **skeleton** = a hierarchy of bones (rest pose, lengths, joint limits) that
exists purely in the render layer. A **procedural animator** = code that
computes the pose each frame from the unit's *state* (velocity, facing,
stance, events) instead of playing back authored keyframes.

Erik's intent (clarified 2026-07-17): **one skeleton per morphology, added
only when needed** — all humanoids share the humanoid skeleton, all
8-legged creatures share the 8-leg skeleton, and a new skeleton is created
only when a new body plan demands it. (Maybe a single über-skeleton could do
everything; that was *not* the intention — treat it as a curiosity to
revisit, not a goal.) Within a morphology, the procedural approach makes the
sharing strong: an "animation set" stops being a pile of hand-authored clips
and becomes **a parameter set** (gait frequency, stride length, posture
lean, arm-swing amplitude, weapon grip) plus a small library of event poses.
A zombie is the humanoid skeleton with a shuffling gait profile and a
slumped posture; a runner variant is the same skeleton with the frequency
knob turned up. New variant within a morphology ≈ new TOML block, not new
art.

## 2. The technique stack (all standard, all cheap)

Layered from must-have to optional:

**a) FK core.** Bone hierarchy + forward kinematics. Trivial (a few dozen
2D transforms per unit).

**b) Analytic 2-bone IK** for legs and arms — closed-form, exact, no
iteration. Covers foot placement and weapon aiming.

**c) FABRIK** (Aristidou & Lasenby 2011, *FABRIK: A fast, iterative solver
for the Inverse Kinematics problem*) for longer chains — tentacles, tails,
snake bodies. A page of code, no matrices, handles joint constraints.

**d) Gait controller** — the actual "procedural" heart. The canonical
stepping trick (popularized by David Rosen's GDC 2014 talk *An Indie
Approach to Procedural Animation*, Overgrowth): each foot has a *home*
anchor that moves with the body; when a foot's world-pinned position drifts
past a threshold from home, it steps — swings to a predicted landing point —
while gait-phase groups (alternating pairs for bipeds, tetrapod/wave gaits
for 8+ legs) keep steps coordinated. Body bobs and leans from the same
phase. This single mechanism gives walking that automatically adapts to
speed, turning, and strafing — no walk-cycle authoring, ever.

**e) Verlet / position-based secondary motion** (Müller et al. 2007,
*Position Based Dynamics*) — dangling straps, tentacle inertia, death
ragdoll-lite. Same family of tricks as (c), a natural extension.

**f) Pose blending** — a tiny library of authored *poses* (aim, reload,
melee wind-up, hit flinch, death) blended over the procedural base. Poses
are single keyframes, not clips — cheap to author even programmatically.

**g) (Far-future, optional) learned motion** — Phase-Functioned Neural
Networks (Holden, Komura & Saito, SIGGRAPH 2017) / Mode-Adaptive Neural
Networks for quadrupeds (Zhang, Starke, Komura & Saito, SIGGRAPH 2018), or
physics-trained control à la DeepMimic / AMP (Peng et al. 2018 / 2021).
Almost certainly unnecessary at 32-px top-down scale, but worth naming: it's
squarely inside this project's ML identity, and rung (d) produces exactly
the parameterized rigs such a network would drive. Park it.

Per the repo convention ([[credit-paper-authors]]): whichever of these land
in code get the paper credit header, and the papers go to `docs/papers/`.

## 3. The skeletons

- **Humanoid (marine, zombies, civilians) — the starting point (Erik,
  2026-07-17): animate just the marines first.** Render-only, zero
  determinism concerns, and the art question can be explored cheaply. Top-
  down caveat, stated honestly: from 90° overhead you see head, shoulders,
  arms, weapon — legs are mostly under the torso. The payoff is shoulder
  sway, arm swing, weapon recoil, continuous facing, flinches, prone
  crawling — *aliveness*, not leg detail. Real but subtler than the exotic
  wins.
- **Spider.** The procedural showpiece, and top-down is its *best* camera:
  all 8 legs fully visible. 8 × 2-bone analytic IK + stepping gait (d) is
  the entire implementation. This is the prototype to build first.
- **Gråsugga / woodlouse & armored dino (beastiary.md).** The spider module
  with more legs and a wave gait; the dino is a quadruped — same machinery,
  different leg count and phase offsets. Zero new tech.
- **Octopus.** 8 FABRIK chains (c) + verlet inertia (e), targets from a
  slow crawl-reach cycle. No gait logic at all — harder to make *good*
  (motion quality is all tuning) but technically simpler than the spider.
- **Snakes/worms.** The rig follows the articulated-occupancy segment chain
  from `breach_unit_class_design.md` — first case where gameplay footprint
  and visual skeleton share a spine (sim owns the segment positions; render
  rig only decorates them).

## 3.5 Sprites vs 3D models — keep the draw decision open

Erik is considering moving to **3D models** for the marines (free downloads
exist — e.g. CC0/CC-BY rigged humanoids on Sketchfab/Quaternius/Kenney, and
Mixamo auto-rigs any humanoid mesh for free, license permitting re-use in
games). Key insight, his: **the skeleton is the shared abstraction** — the
same procedurally-computed pose can drive either renderer, so the rig can be
developed now *without* forcing the sprites-vs-3D decision:

- **Part-sprites:** project each bone's 2D transform → rotated
  `draw_texture_pro` per part (§4.4).
- **3D model:** map the rig's bones onto the model's armature (top-down 2D
  gait solved in the ground plane, applied to the 3D skeleton's yaw/limb
  joints) and render the posed mesh from the overhead camera — raylib loads
  glTF with skeletal animation natively, so pyray can do this without a new
  engine dependency.

Design rule that follows: keep the pose solver **renderer-agnostic** (bones
in world-space, no sprite knowledge inside the solver); the renderer binding
(§4.4) is a swappable back end. If a downloaded 3D marine looks good from
90° overhead, the part-sprite pipeline (§4.5) may never need to be built for
humanoids at all — that's the cheapest possible resolution of the art-
pipeline risk in §5.

## 4. What we'd need to build (the actual answer to "what do we need?")

1. **Rig data model + loader** — skeleton definitions in TOML (bones,
   lengths, hierarchy, joint limits, part-sprite bindings, gait profile).
2. **Pose solver module** — FK, 2-bone IK, FABRIK, verlet. Pure Python
   first (NumPy at most); this is dozens-of-units × ~20 bones × simple
   trig per *render frame* — nowhere near needing C++/GPU.
3. **Gait/behavior controller** — maps sim state (velocity, facing, stance,
   events like fired/hit/died) to solver targets. The design-heavy part.
4. **Part-sprite renderer** — `draw_unit` becomes a short draw list of
   rotated `draw_texture_pro` calls (one per boned part, depth-sorted).
   Raylib does rotated blits natively; this also finally *uses* the
   continuous `facing` instead of the 8-way snap.
5. **Art pipeline change** — units become **segmented part sprites**
   (torso, head, upper/lower limbs, weapon) instead of whole-body stills.
   `art/export_sprites.py` / the HTML generators need a per-part export
   mode. Natural tie-in to the AI-sprite/tileset generation track
   (levels-w1 P6): AI-generating small *parts* is easier than generating
   coherent 8-direction character sheets.
6. **A tuning playground** — feel is everything in procedural animation;
   a `prototypes/` sandbox with live sliders (like the weapons armory
   pattern) before engine integration.

Explicit non-need: no third-party animation runtime (Spine, DragonBones),
no keyframe clip format, no GPU skinning, no determinism work.

## 5. Risks / open questions

- **Humanoid top-down readability** — will the effort show at this sprite
  scale? De-risked by prototyping the spider first (guaranteed payoff) and
  a humanoid mock second, *before* committing to the art-pipeline change.
- **Art direction** — segmented parts can look "paper-doll" if joints are
  naked; overlap margins and shadowing need iteration.
- **Scope discipline** — (g) ML motion is a rabbit hole; it stays parked
  until the plain gait controller visibly runs out of quality headroom.
- **Zombie identity** — zombies currently get charm from hand-drawn variant
  stills; a shared skeleton must not homogenize them (answer: per-variant
  part sets + gait profiles).

## 6. Suggested phasing (post physics-v1; order revised 2026-07-17)

- **P0 — marines first (Erik's call)**: prototype in `prototypes/` (raylib,
  sliders) animating the humanoid skeleton — walk/aim/fire, render-only,
  zero determinism stakes. In the same prototype, test a downloaded free
  3D marine from the 90° camera (§3.5) vs part-sprites, so the art-pipeline
  question gets answered by eye before anything is built for real.
- **P1 — rig core**: extract the data model, solvers (FK/2-bone IK/FABRIK),
  TOML skeleton defs from the prototype.
- **P2 — spider**: the 8-leg skeleton + stepping gait — the tech showcase,
  and the template for every multi-leg creature after it.
- **P3 — menagerie**: gråsugga, quadruped, octopus (FABRIK stress test) —
  each preceded by its determinism-cost call per §0.2 if it interacts with
  the world; snake-on-articulated-occupancy.
- **P4 (parked)** — learned motion, only if wanted.

Slotting per roadmap: this is a **beauty track** in Phase-2 terms — it
doesn't block weapons/units/game-rules/NN-training, and nothing in it
touches the physics engine or the determinism fence. P0 is small enough to
be a palate-cleanser side quest whenever a heavy physics arc needs a break.
