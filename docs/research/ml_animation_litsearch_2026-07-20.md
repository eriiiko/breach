# Breach ML-Animation Literature Review & First-Steps Recommendation

*Dated capture, 2026-07-20. Append-only per doc culture — this is a research synthesis, not canon. Six parallel research agents swept the literature on ML-powered and physics-based character animation for Breach; the full cited digests live in [`ml_animation_litsearch_2026-07-20_raw/`](ml_animation_litsearch_2026-07-20_raw/) (one file per theme) and are the durable reference. This doc distills them and ends with an opinionated "what to build first" and the cost verdict Erik asked for.*

**Feeds / relates to:** [`procedural_animation_brainstorm.md`](../procedural_animation_brainstorm.md) — the *classical* (non-ML) procedural-animation brief from 2026-07-17, which already decided the determinism split (marines render-only; world-interacting bodies get a per-creature cost eval) and §3.5 the sprites-vs-3D question. **This search is the deep dive into the branch that doc explicitly parked** (its rung (g) / Phase-4 "learned motion"), plus the 3D-render-cost and NVIDIA-physics-doll questions it only named in passing. Also relates to the RL search [`rl_litsearch_2026-07-20.md`](rl_litsearch_2026-07-20.md) (Breach's RL identity is why the physics-control line and Isaac Gym rhyme with the engine).

---

## 0. The one idea that unifies everything Erik asked about

The things you asked — 3D marines over the 2D world, "is NVIDIA-style ML animation expensive?", the deterministic monster path, the sprint-into-each-other-and-fall scenario, and (the high-prio addition) realistic reactions to **impacts** — bullets, grenade blowback, shockwaves — are **not** separate questions. They are **one decision made at two independent dials**, plus **one fact that collapses the cost fear.**

```
DIAL 1 — the determinism fence        DIAL 2 — the animation source
(you already drew this yourself)      (rising realism, rising substrate cost)

render-only ───────► gameplay-        hand-tuned ──► data-driven ──► physics-
  (marines)          authoritative     procedural     kinematic       based
                     (monsters)        gait+IK        (PFNN/LMM)      RL control
  no determinism     Q16.16, gated                    mocap needed   (the NVIDIA
  work at all        golden-tested                                    dolls)
```

- **Dial 1 is the determinism ladder** (Agent 5's spine): rung 0 render-only → rung 1 split-spine (integer footprint is authoritative, float rig only decorates) → rung 2 fixed-point IK → rung 3 fixed-point ragdoll. **Effort roughly decuples per rung, and the gameplay need almost never climbs past rung 1.** You already put marines at rung 0 and monsters "somewhere higher" — the literature's refinement is that "higher" almost always means **rung 1, not rung 3**.
- **Dial 2 is where the animation comes from.** Classical procedural (the sibling doc) → data-driven kinematic playback (PFNN, Motion Matching, Learned Motion Matching) → physics-based RL control (DeepMimic→AMP→ASE→MaskedMimic, the "boxing/falling dolls").

**The fact that collapses the cost fear (every one of the five agents reached it independently): the ML network is never the bottleneck.** A learned animation controller is a *tiny MLP* — microseconds per character, runs on a 5 MHz microcontroller in the robotics literature. So "is ML animation super expensive?" has a precise answer: **the network is nearly free; the cost is entirely in whatever you attach it to** —

| Animation source | Where the real cost hides |
|---|---|
| Data-driven kinematic (PFNN/LMM) | **acquiring + cleaning + retargeting mocap data** (weeks of pipeline) |
| Physics-based learned (DeepMimic/AMP/ASE) | **the rigid-body physics solver** the tiny policy drives (Breach doesn't have one yet) |
| Gameplay-authoritative anything | **determinism engineering** — goldens, gates, per-arch verification |

This is the through-line to carry forward: **pick the lowest dial-2 rung that looks good, and the lowest dial-1 rung that gameplay demands — because moving up either dial buys cost in data, physics, or gates, never in "the AI."**

---

## 1. What every theme agreed on (the convergent findings)

Five agents searched independently; these showed up in three or more digests.

1. **The network is free; the substrate is the cost.** (Above — every agent.) The MLP in PFNN, in an AMP policy, in DReCon is ~0.1–1M params, sub-millisecond, batchable. Erik's "is it super expensive?" fear is **unfounded on the compute axis** and correct only about *data* (kinematic) and *a physics engine* (physics-based).

2. **At true 90° top-down, the camera hides exactly what ML animation buys.** PFNN/MANN/LMM exist to perfect **foot placement, weight-shift, stride phase, contact** — all directly under the torso and occluded at a 3×3-tile overhead footprint. What reads top-down is **facing, translation, torso/arm silhouette, weapon**. `unit.facing` is *already* a continuous float, so a hand-tuned procedural gait + a little IK gets ~all the visible win with none of the mocap burden. **Neural kinematic motion is the wrong first rung for Breach.** (Agents 1, 4.)

3. **Live GPU-skinned 3D is cheap and is the right first step — not sprites, not ML.** raylib ≥5.5 (Nov 2024) added optional **GPU skinning** (`UpdateModelAnimationBoneMatrices` + a skinning shader, by Daniel Holden). On that path, "tens of marines" is a **low-single-digit-ms fraction** of a 60fps frame; hundreds before you're even draw-call-bound (GPU Gems 3 rendered ~10k animated characters at 30fps on *2007* hardware). The one trap: raylib's *default* CPU-skinning path caps at **~20 characters** (worse in CPython) — so use the GPU path from day one. (Agent 4.)

4. **The NVIDIA "boxing/falling dolls" are physics-based RL control, and you should REUSE, not train.** DeepMimic → AMP → ASE → PHC/PULSE → MaskedMimic is a lineage of small policies driving a ragdoll in a physics sim, trained (GPU-days on Isaac Gym) to imitate mocap while pursuing goals — so **falls, staggers, and get-ups are emergent physics, not authored**. Pretrained, released controllers exist (**PULSE** 32-dim latent covering 99.8% of AMASS; **MaskedMimic/ProtoMotions** public weights). The honest cost is **"add a rigid-body character solver," not "add an MLP."** (Agent 2.)

5. **The determinism fence goes exactly where the iron rule already puts it — and stays there.** Animation may *consume* synced integer state and *propose* integer events; **no float — IK, ragdoll, or neural — may ever become synced state.** A pose that draws pixels is determinism-free; a pose that says "this tile is blocked / this unit is prone" inherits the full float-desync problem (the same disease as the X-ARCH BLAS-in-RNG desync). (Agent 5.)

6. **The sprint-into-each-other-and-fall scenario is a solved, staged problem — and cheapest done render-only.** Menu from Agent 3: **Rung 0** = sprite/clip tumble + impulse-scaled knockback slide (buildable in *today's* renderer, zero physics — the honest first spike); **Rung 2** = blended partial ragdoll on high-relative-velocity capsule overlap + front/back get-up (the AAA default, only felled units simulate, ~12–20 bodies); up to **Rung 4** = a learned DReCon-style controller. The hard part is never triggering the fall — it's making the **get-up blend-out invisible.** (Agents 2, 3.)

7. **Breach's determinism is an asset here too, and fixed-point animation math is friendlier than it sounds.** 2-bone IK / FABRIK / Verlet are mul/add/sqrt with **no parallel reductions**, so fixed-point *kinematics* is bit-reproducible almost for free (Q16.16 already ships `atan2_q16/sin_q16/cos_q16`). The genuinely hard fixed-point case is *contact* (ragdoll) — overflow, world-size, constraint ordering. And any integer animation kernel that graduates into the GPU-resident sim inherits cross-GPU bit-identity **for free** (integer add is associative), provided no float atomics. (Agent 5.)

8. **Impact reactions follow force scale, and Breach already computes the force.** The field's most-repeated rule (Agents 3, 6): *small forces make physics look twitchy — reserve physics for big events.* So bullets → **authored additive flinch** (direction × body-location × strength); grenades → **full ragdoll + radial impulse launch** (no clip set covers every launch vector); shockwaves → the middle case (a radial impulse with an arrival delay, steeper-than-linear falloff, and decay-over-time, so a wall of marines folds outward *in sequence*). Crucially, **Breach's sim already produces the blast field** (EOS gas, wall-bursts) — so the cheapest convincing reaction is for the render layer to *consume the sim's impulse*, not compute anything new. There is **no character-animation shockwave-response paper** — the game treatment just *is* the radial-impulse recipe (Unity `AddExplosionForce` / Unreal `AddRadialImpulse`) with delay + decay. (Agent 6.)

---

## 2. Thread-by-thread landscape (compressed)

### Thread A — Data-driven kinematic motion (render-only playback)
The "learn from mocap, emit next pose, no physics" family. **PFNN** (Holden, Komura & Saito, SIGGRAPH 2017 — phase-function morphs the network weights through the gait cycle; ~0.5M params, ~10 MB, ~1 ms/frame) → **MANN** (Zhang et al. 2018 — mixture-of-experts, learns its own "phase," quadrupeds) → **Neural State Machine** / **Local Motion Phases** / **DeepPhase** (Starke et al. 2019–2022 — scene interaction, per-bone phases, auto-extracted phase manifold). The production-realistic branch is **Motion Matching** (Clavet, GDC 2016, For Honor — a KNN search over a mocap DB, *not* neural) and its neural compression **Learned Motion Matching** (Holden et al., SIGGRAPH 2020 — 590 MB → 17 MB, constant CPU cost, ships on CPU). **MotionVAE** (Ling et al. 2020) is the RL-relevant one: a learned latent action space steered by a deep-RL policy — the natural bridge to Breach's RL half. Motion **diffusion** (MDM arXiv:2209.14916; MotionDiffuse arXiv:2208.15001) is powerful but **offline** — an asset-authoring tool, never a render-loop component.
→ **Verdict: don't go neural-kinematic now.** Compute is a non-issue; the *mocap pipeline* is the cost, and the top-down camera throws away the detail it buys. Revisit **LMM** only if the camera tilts or you commit to a rich mocap combat set; **MotionVAE** only if you want RL-authored visible motion.
→ Full digest: [`01_kinematic_neural_motion.md`](ml_animation_litsearch_2026-07-20_raw/01_kinematic_neural_motion.md)

### Thread B — Physics-based learned control (the NVIDIA boxing/falling dolls)
A ~15-body / ~30-DOF ragdoll in a rigid-body sim, driven by a small MLP trained via RL to imitate mocap while pursuing goals — so falls/staggers/get-ups are *emergent physics*. Lineage: **DeepMimic** (Peng et al., SIGGRAPH 2018, arXiv:1804.02717 — imitation reward + reference-state-init + early-termination) → **AMP** (arXiv:2104.02180, 2021 — a learned style discriminator replaces hand-tuned rewards) → **ASE / CALM / PADL** (NVIDIA, 2022–23 — reusable, directable, language-conditioned skill latents) → **PHC / PULSE** (arXiv:2305.06456 / 2310.04582 — universal imitation, **fall-recovery without external forces**, a 32-dim reusable "foundation" latent) → **MaskedMimic / ProtoMotions** (NVIDIA, SIGGRAPH Asia 2024 — one controller from any partial spec, **pretrained weights released**). Training substrate: **Isaac Gym** (arXiv:2108.10470), a massively-parallel GPU physics-RL sim that **rhymes with Breach's own GPU-physics-as-RL-state-space identity**.
→ **Verdict: reuse a pretrained controller (PULSE/ProtoMotions), don't train.** Runtime cost = tiny-MLP inference (free) + a rigid-body physics step per character (the real budget). A handful-to-dozens of physics ragdolls is affordable *if Breach adds an articulated-body+contact solver* — which it lacks today. So this is a **render-only "looks cool" win** cheaply (float ragdoll on the render side), or a **real arc** if the monster physics must be gameplay-authoritative (see Thread E).
→ Full digest: [`02_physics_based_learned_control.md`](ml_animation_litsearch_2026-07-20_raw/02_physics_based_learned_control.md)

### Thread C — Ragdoll, hit-reactions & the bump-and-fall scenario
The bridge between keyframes and full physics: **dead** ragdoll (limp, cheapest) → **active/powered** ragdoll (PD-controlled joints torque toward an animation pose; strength is a dial you drop on impact) → **blended** ragdoll (keyframe animation, physics blended in 0→1 only on impact — the AAA default). Production landmarks: **Euphoria** (GTA/RDR — the gold standard *and* the cautionary tale: CPU-heavy, a-few-actors-only, never a crowd); EA/Frostbite and Unreal's **PhysicalAnimation / Physics Control** and Unity's **PuppetMaster** are the shipped, affordable driven-ragdoll paths. On the ML side, **DReCon** (Bergamin, Clavet, Holden, Forbes; Ubisoft, SIGGRAPH Asia 2019 — motion-matching + deep-RL PD control, ~2–4 ms/char, auto-recovers from pushes) and its GDC-2020 sibling **Ragdoll Motion Matching** are directly on-theme: **perturbation recovery *is* the bump-and-fall loop, learned.**
→ **Verdict: prototype Rung 0 (sprite tumble + knockback slide) in the current renderer immediately** to test whether the *idea* reads top-down. If it earns 3D marines, jump to **Rung 2** (blended partial ragdoll, felled-units-only, ~8–12 concurrent is comfortable). Because Breach is top-down, a **planar 2D rigid-body** knockdown may suffice and is far cheaper — worth a spike.
→ Full digest: [`03_ragdoll_reactions_fall.md`](ml_animation_litsearch_2026-07-20_raw/03_ragdoll_reactions_fall.md)

### Thread D — Rendering 3D over 2D & the compute budget (the direct cost answer)
**raylib loads rigged glTF/GLB natively** (`LoadModel`, `LoadModelAnimations`) and composites 3D into Breach's existing world-space RT cleanly (`BeginMode3D` with an orthographic top-down camera, nested inside `BeginTextureMode`; the single camera blit is unchanged; **continuous `facing` maps straight to model yaw** — a real win over sprites' 8-direction snap). The decisive fact: **use raylib ≥5.5 GPU skinning** (`UpdateModelAnimationBoneMatrices`) — the default CPU path caps at ~20 characters, the GPU path pushes hundreds. Free rigged assets: **Quaternius** (CC0, no strings, Universal Animation Library for shared-skeleton retargeting), **Kenney** (CC0), **Mixamo** (auto-rig + mocap, royalty-free for games — *but forbids ML-training use; fine since marines are render-only, keep render out of the training data path*). Crowd toolbox for "big battles later": GPU instancing + **animation/vertex textures** (~10k characters in ~20 draw calls), impostors/LOD. Middle-ground alternative: **pre-render 3D → sprite sheets** (Diablo/StarCraft), zero runtime 3D cost but throws away continuous facing.
→ **Verdict: no, it is not expensive — start with live GPU-skinned 3D marines.** Cheapest fixes for the only real cost centres: GPU-skinning path (biggest lever), blob-shadow sprites instead of shadow maps, share one `Model` across units. Dominant risk is **glTF-import friction**, not per-frame compute. Keep pre-render-to-sprites as the safety net.
→ Full digest: [`04_rendering_3d_on_2d_cost.md`](ml_animation_litsearch_2026-07-20_raw/04_rendering_3d_on_2d_cost.md)

### Thread E (cross-cutting) — Determinism, fixed-point & GPU-residency
Two desync families map onto Breach's two animation concerns: **(A)** float non-portability (x87/SSE, transcendentals, FMA, fast-math — hits IK/ragdoll math) and **(B)** parallel-reduction-order + atomics + vendor kernels (hits NN inference; the same class as the X-ARCH BLAS desync). Fixed-point Q16.16 is the proven RTS/fighting-game answer and Breach's proven path; **fixed-point IK/FABRIK/Verlet is nearly free** (no reductions), **fixed-point contact/ragdoll is the expensive case** (overflow, world-size, constraint ordering — Box2D's author's explicit warnings; note Box2D 3.1 achieved determinism *without* fixed-point on a CPU matrix, an honest counterpoint that doesn't change Breach's harsher multi-vendor-GPU calculus). For NN inference: either int-quantize **with pinned reduction order** (real work) or — Breach's native pattern, the recommendation — **keep the net outside the fence and let only an integer action/pose cross** (the DeepMimic controller/sim split). The **4-rung determinism ladder** (§0) is the deliverable; the decision rule is *"at what spatial resolution must the pose be mechanically authoritative?"* — tile-res → rung 1, joint-res → rung 2/3.
→ **Verdict: default every creature to rung 0 (marines) or rung 1 (split-spine cosmetic rig over an integer footprint — covers ~all world-interacting monsters and "falls as an integer prone/footprint event"). Reserve fixed-point rig work (2–3) for a specific, argued need.** On GPU residency: an integer animation kernel inherits cross-GPU bit-identity for free — one block per creature, Q16.16/Q32.32, no float atomics, fixed iteration count, index-ordered constraints.
→ Full digest: [`05_determinism_fixedpoint_residency.md`](ml_animation_litsearch_2026-07-20_raw/05_determinism_fixedpoint_residency.md)

### Thread F — Impact reactions: bullets, grenades & shockwave knockback (companion to Thread C)
Impacts sort by **force scale**, and the technique follows the scale. **Localized/small (bullets):** the workhorse is an **additive flinch layer** (stored as a pose *difference* added over whatever the base is playing, so one clip works over idle/walk/reload), decomposed **direction × body-location × strength** (asset packs ship ~127–168 clips on exactly that grid); craft is in the *feel* — kill the cross-fade, fast-in/slow-out, hit-pause/hit-stop, spark VFX. Physics enters only for heavier localized hits (a partial-body blend on the struck limb, weight 0→1→0). **Whole-body/large (grenade):** the **full-ragdoll launch** case — gather bodies in radius → distance falloff → **line-of-sight/cover test** (Unreal's `RadialForceComponent` notoriously blasts *through walls* without one) → upward bias ("pop up") → ragdoll + impulse → land → get-up. Copyable primitives: Unity `AddExplosionForce(force, pos, radius, upwardsModifier)`, Unreal `AddRadialImpulse(origin, radius, strength, RIF_Linear)`. **Shockwave/blast-wave:** the same radial impulse *plus* a propagation delay (`distance / wave_speed`), a steeper (cube-root/inverse-square) falloff, and decay-over-time — so an expanding ring hits nearer marines first and harder, folding a wall of bodies outward in sequence, with an optional negative-phase "suck-back" flourish. **ML angle:** a blast is just an *external perturbation* — DeepMimic policies recover from big pushes *without being trained on them* (emergent from exploration noise), so "hit and stay up vs go down" can be **emergent**, not an authored threshold (DReCon/SuperTrack + robotics push-recovery are the Rung-4 aspiration, render-layer only).
→ **Verdict: Breach already computes the blast — make the marine visually *answer* it.** Cheapest convincing path (Rung 0, buildable today): render layer reads the sim's per-unit impact event (direction + magnitude + body-location) and plays a magnitude-scaled sprite knockback slide + billboard spin, with an expanding-ring arrival delay for shockwaves. Reserve Rung 2 full-ragdoll launch for grenades if marines go 3D — **cap concurrent ragdolls** (a grenade into a squad = N simultaneous launches, the theme's worst-case spike; overflow to Rung 0), and spike a **planar 2D ragdoll** first (top-down may make it sufficient and far cheaper). Two honest gaps: no shockwave-response *paper* exists (it's just the radial-impulse recipe), and squad-grenade concurrency is the one place to budget the peak not the average.
→ Full digest: [`06_impact_shockwave_reactions.md`](ml_animation_litsearch_2026-07-20_raw/06_impact_shockwave_reactions.md)

---

## 3. What I'd actually build first (concrete, staged)

The sibling brainstorm already names the classical rig stack. This ladder sequences the *ML/3D/physics* work so each rung is a checkpoint you can stop at, and so the cheap-but-high-value steps come before any ML. **None of the first three rungs is ML** — that's the honest recommendation.

**Phase 0 — Live GPU-skinned 3D marines, render-only (the "learn a little" rung).** *This is the real first step, and it is not expensive.*
- One low-poly rigged humanoid (**Quaternius** CC0, or **Mixamo** if you want polished mocap), ≤40 bones, ≤~3k tris. Confirm raylib **≥5.5** under pyray and use the **GPU-skinning path** (`UpdateModelAnimationBoneMatrices` + the stock skinning shader) from the start — never the CPU path.
- Pick the clip from unit state (idle/move/fire/die), advance its frame each render tick, map `facing` → yaw. Composite inside the existing world-space RT with an orthographic top-down camera. **Blob-shadow sprites**, not shadow maps.
- **Render-only, read-only off sim state, gated OFF in headless ML training.** Zero determinism stakes. Safety net: the same asset pre-renders to sprite sheets if glTF import fights you.
- *Success signal:* tens of 3D marines turning/walking/firing smoothly over the 2D world at 60fps. This answers the whole "is it expensive?" question empirically, cheaply.

**Phase 1 — Reactions (bump-and-fall *and* impacts), render-only, cheapest-first.** *These share one mechanism — a sim-authored impact event (direction + magnitude + body-location) that the render layer answers — so build them together.*
- **Rung 0 first, for both:** on a high-relative-velocity capsule overlap *or* the sim's blast/impact event, trigger a scripted reaction — a tumble (bump/grenade) or a 1–2 frame directional jolt/flash (bullet) + an **impulse-scaled knockback slide** and billboard spin, debounced. For a shockwave, add an **expanding-ring arrival delay** so nearer marines fold outward first. Buildable *today* in the current sprite renderer; tests whether the idea reads top-down before any physics. This alone may satisfy "blown out from a grenade."
- If it earns more (once marines are 3D): **additive directional flinch** for bullets, **Rung 2 blended/full ragdoll** for falls and grenade launches — felled/launched units only, front/back get-up, and a **cap on concurrent ragdolls** with overflow falling back to Rung 0 (the squad-grenade worst case). Spike a **planar 2D rigid-body** knockdown first (top-down makes it plausible and far cheaper).
- Gameplay effect (prone flag, blocked tile, knockback distance, who-dies, stun ticks, blast falloff + cover) is computed in the **Q16.16 sim as integers**; the tumble/launch is render-side float decoration over that sim-authored start/end. **Never let the ragdoll's floats decide the outcome** (the X-ARCH lesson).

**Phase 2 (optional "wow") — Reuse a pretrained physics controller, render-only.**
- Drop a **ProtoMotions/PULSE** controller onto a float ragdoll on the render side for genuinely emergent stagger/topple/get-up (PHC-style recovery). No training. This is the "NVIDIA doll" look as a cosmetic treat, sidestepping determinism entirely.
- Cost gate: it needs a rigid-body character solver on the render side — evaluate whether that's worth it over Rung 2's blended ragdoll, which is far cheaper and usually reads the same top-down.

**Phase 3 — Deterministic monsters (only when a monster's physics must be gameplay-authoritative).**
- **Default to Rung 1 (split-spine):** the monster's integer articulated-occupancy chain (already the snake/worm design) owns tiles/grabs/prone; a render-side float rig decorates it. **≈zero new determinism work.** This covers essentially every "monster with real collisions" case Erik has named.
- **Rung 2 (fixed-point IK)** only if a creature's *reach/curl* must be authoritative at joint resolution — friendly math, the cost is the golden/per-arch gate (the S8a-style dance the team already knows).
- **Rung 3 (fixed-point ragdoll)** is a *project*, not a feature — a fixed-point contact solver with overflow discipline + a batched Q16.16 GPU-residency kernel. **Do not enter without a concrete gameplay reason rungs 0–2 cannot serve.**

**Parallel, independent:** the classical procedural rig from the sibling doc (spider/menagerie showcase) can proceed on its own track — it's the render-side pose engine that *any* of these rungs decorate with, and it needs no ML at all.

---

## 4. Reading queue (if you only read a handful)

Ranked for Breach's situation, mixing "read to decide" and "read before coding":

1. **raylib `models_gpu_skinning` example + PR #4321** (Daniel Holden) — the exact API Phase 0 builds on; turns "~20 characters" into "hundreds." The single most actionable link.
2. **NVIDIA GPU Gems 3, Ch. 2 "Animated Crowd Rendering"** — ~10k animated characters at 30fps on 2007 hardware; the definitive proof Breach's scale is trivially cheap, plus the instancing+vertex-texture technique for big battles later.
3. **DReCon** (Bergamin, Clavet, Holden, Forbes; SIGGRAPH Asia 2019) — the canonical low-runtime responsive physics controller; motion-matching + RL PD; **perturbation recovery = the bump-and-fall loop, learned.** Most on-theme for an RL game.
4. **MoCap Online, "Ragdoll Physics in Games"** + **Unity `AddExplosionForce` docs** + **Game Developer, "The Role of Animations in Hit Effects"** — the Phase-1 trio: the dead/active/blended taxonomy and cost rules (12–20 bodies, 8–12 active ragdolls, physics LOD); the copyable radial-impulse-with-falloff + "pop up" formula for grenades/shockwaves; and the *feel* of localized hits (kill the cross-fade, fast-in/slow-out, hit-pause) with an honest "we skipped ragdoll and authored was enough." Read before building Phase 1.
5. **DeepMimic** (Peng et al., SIGGRAPH 2018, arXiv:1804.02717) — the foundational physics-control recipe; understand it and the whole NVIDIA-doll field clicks.
6. **MaskedMimic + ProtoMotions** (Tessler et al., SIGGRAPH Asia 2024; `github.com/NVlabs/ProtoMotions`) — the current unified, **pretrained, released** controller; the practical reuse entry point if Phase 2 ever happens.
7. **Box2D 3.1 determinism post** (2024) + **Gaffer-on-Games "Floating Point Determinism"** — the "why the iron rule exists" pair, and exactly what deterministic float would cost if it ever tempted you.
8. **PFNN** (Holden, Komura & Saito, SIGGRAPH 2017) + **Learned Motion Matching** (Holden et al. 2020) — read only if/when the camera tilts or you commit to a mocap combat set; what "neural animation in a real game today" actually means.

---

## 5. Cautions carried across the whole search

- **The cost question has a precise answer, and it's not "the AI."** The network is nearly free; cost is mocap data (kinematic), a physics solver (physics-based), or determinism gates (gameplay). Don't let "ML animation" sound expensive — name the actual substrate cost each time.
- **The top-down camera is the quiet veto on most of this.** It hides gait/foot detail (kills the case for neural kinematic motion) and makes cheap tricks read well (a sprite tumble may be "cool" enough). Prototype top-down *before* committing to any expensive rung — the projection decides what's worth paying for.
- **CPU skinning is the one real performance cliff** (~20 units, worse in CPython). Use raylib ≥5.5 GPU skinning from the first commit; everything else is far below the ceiling.
- **Determinism is an asset you must actively guard for animation too.** *Animation may consume synced integer state and propose integer events; no float — IK, ragdoll, or neural — may ever become synced state.* A gameplay-authoritative fall is a stance+footprint integer event, not a physics simulation. Keep the float ragdoll strictly downstream (the X-ARCH lesson).
- **Reuse, don't train, the physics-doll controllers.** Training is GPU-days on Isaac Gym; PULSE/PHC/ProtoMotions ship pretrained. The build cost is a rigid-body character solver + skeleton retarget, not the policy.
- **Mixamo forbids ML-training use of its assets.** Fine for render-only marines, but keep the 3D-render layer out of any training-data pipeline; Quaternius/Kenney (CC0) have no such clause.
- **Match the impact technique to the force, and budget the concurrency *peak*.** Physics on a small hit reads as twitchy noise — bullets want authored additive flinch, not ragdoll; reserve physics for launches. And a **grenade into a dense squad = N simultaneous ragdoll launches** (more bursty than pairwise bumps) — the one concurrency spike this whole space has; cap concurrent ragdolls and overflow to the cheap Rung-0 slide, or you'll frame-drop exactly when the action peaks.
- **Citations are agent-surfaced and several are flagged.** Confident: PFNN/MANN/LMM/MotionVAE, DeepMimic/AMP/ASE/PHC/PULSE/MaskedMimic/Isaac Gym arXiv IDs, DReCon, the raylib PRs/issues, Box2D/Gaffer/NVIDIA-CCCL determinism sources. Spot-check before formal use: the DeepMimic arXiv ID (not re-fetched), PADL/SuperPADL/Isaac-Lab venue details, exact PFNN ms/frame + memory-mode numbers, DReCon's exact per-frame ms, the Euphoria "$100k+" figure (forum hearsay), and any 2025–2026 robotics get-up IDs (leads, not settled). Details flagged inline in each raw digest's final section.

---

## 6. Ideas raised while scoping (2026-07-20 — capture, act on later)

- **"Start with just the marines to learn a little."** Endorsed and made concrete: Phase 0 (live GPU-skinned 3D marine, render-only) *is* that learning rung, and it's cheap. It also builds the render-side pose plumbing that every later rung reuses.
- **Larger monsters with real collisions and gameplay mechanics.** The default answer is **rung 1 (split-spine)**, not a fixed-point ragdoll: the monster's integer footprint carries gameplay, a float rig decorates it. This gives "real collisions" deterministically with ~zero new determinism work. Full fixed-point ragdoll (rung 3) is reserved for a monster whose *collapse/pile-up geometry itself* must be mechanically exact — a real arc, entered only with a stated reason.
- **The bump-and-fall as a signature feature.** Cheapest convincing path is render-only (Rung 0 today → Rung 2 blended ragdoll with 3D marines). If it becomes a signature, DReCon-style learned recovery (Rung 4) is the on-theme end-state for an RL project — but it's an arc, and the gameplay effect must still resolve as integers in the sim.
- **Impacts (bullets, grenades, shockwaves) — high-prio, and cheaper than it sounds because the sim already does the hard part.** Breach already computes blasts (EOS gas, wall-bursts); the animation just *consumes* the impulse the sim emits. So "realistic reactions to shockwaves / getting blown out of a grenade" starts as a Rung-0 magnitude-scaled knockback slide + spin with an expanding-ring delay — buildable today, no 3D or physics. The richer rungs (additive flinch for bullets → full-ragdoll launch for grenades → learned perturbation-recovery) are all *render-only upgrades* over the same sim-authored impulse, so none of them ever threatens determinism. The one genuinely novel finding: **no character-animation shockwave paper exists** — the game answer is the radial-impulse recipe with a propagation delay and steeper falloff, which Breach can drive directly from its own pressure field.
- **The Breach ↔ Isaac Gym rhyme.** Breach *is* a GPU physics engine used as an RL state space; Isaac Gym is the (float, non-deterministic) one the NVIDIA dolls train in. If Breach ever gains a character-physics solver, it could in principle *train* animation controllers in its own engine — a research stretch, noted not planned.

---

*Next session: this is capture, not canon. When an animation arc actually opens, fold the chosen first-experiment (almost certainly Phase 0 + Phase 1) into a proper design doc and the render/architecture canon, and archive the exploratory bits. The two-dials framing (determinism fence × animation source) and "the network is never the bottleneck" are the through-line to carry forward — together with the sibling [`procedural_animation_brainstorm.md`](../procedural_animation_brainstorm.md), which owns the classical rig this search decorates.*
