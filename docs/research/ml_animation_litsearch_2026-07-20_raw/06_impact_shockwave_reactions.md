# Impact Reactions — Bullet Hits, Grenades & Shockwave Knockback

**Agent 6 of 6 — ML/physics animation lit search for Breach**
**Date:** 2026-07-20
**Scope:** how characters *react* to impacts — localized bullet/small-projectile hits, whole-body grenade/explosion knockback, and expanding blast-wave/shockwave blowback — spanning authored additive reactions, blended/active ragdoll, and learned perturbation-recovery policies. Focus on the **reaction** (Breach already simulates the blast; the question is how a marine/monster answers it) and on the render-vs-gameplay split.

**Capture, not canon.** Every claim is sourced or flagged. Breach today: pyray/raylib, true 90° top-down camera, static 32×32 sprites, ~tens of units, humanoid ≈3×3 tiles, continuous float `facing`, **no animation system yet**, deliberately cheap. Determinism iron rule: synced sim state is Q16.16 integer, no floats/libm in sim path; render layer exempt.

**Reads sibling 03 first, extends it.** Sibling `03_ragdoll_reactions_fall.md` owns the dead/active/blended taxonomy, Euphoria, DReCon/PuppetMaster, and the *agent-vs-agent bump-and-fall* 5-rung menu. I do **not** restate the taxonomy — I reference its rungs by number and build the *impact-specific* ladder (bullet flinch → grenade launch → shockwave blowback → learned recovery). Sibling `02` owns DeepMimic/AMP/ASE/SuperTrack in depth; sibling `05` owns determinism specifics. I hand off to all three rather than duplicate.

---

## 1. Framing: three impact scales, one render/gameplay fence

Impacts sort into a **force spectrum**, and the right technique is a function of scale — this is the field's single most repeated rule:

> *Small forces make physics look twitchy; reserve physics for big events.* (Sibling 03 §1c; Epic "Physics-Driven Animation"; MoCap Online.)

- **Localized / small (bullet, pellet, small-arms hit):** the character is *hit but stays up*. Physics on a settled/near-settled body at these force levels vibrates and reads as noise (angular drive springs fighting sleep thresholds — Unity forums). **Authored additive flinch wins here.**
- **Whole-body / large (grenade, rocket, close explosion):** the character is *launched*. Momentum dominates pose; a physics ragdoll with a radial impulse looks great and authored clips can't cover every launch angle/spin. **Physics wins here.**
- **Shockwave / blast-wave (pressure front, not a point impact):** a *directional wall of force* arriving over a short but non-zero time, decaying with distance and time — between the two: too big to flinch through, but not always a full launch. **Blended/scaled response.**

Cross-cutting the whole spectrum is the **render/gameplay fence** (owned by sibling 05): the numbers that must be identical on every machine (blast radius, falloff, who-dies, knockback vector, final position, prone/stun ticks) are **Q16.16 sim**; the *visual* flinch/tumble/launch is **float render decoration** stretched between sim-authored start and end states. Breach's blast **force field itself, if it already lives in the sim, is authoritative integer** — the animation is a strictly downstream consumer that never feeds back (§9).

---

## 2. Localized hit reactions (bullets, small impacts)

The production-standard answer to "get shot in the shoulder and keep fighting" is **not** ragdoll — it is an **additive animation layer** plus optional **partial-body physics blend**.

### 2a. Additive flinch layers (the workhorse)
- An **additive animation** is stored as the *difference* between a pose and a reference pose; at runtime that difference is **added on top of whatever the base layer is playing**, so one flinch clip works over idle, walk, run, or reload without authoring a combinatorial set. (MoCap Online, "Blend Trees in Game Engines"; Game Developer, "The Role of Animations in Hit Effects.")
- **Upper-body-only additive overlays** are the norm for hit reactions: the legs keep locomoting while the torso/arms register the shot. (MoCap Online third-person guide.)
- Craft details that sell the hit (Game Developer, "The Role of Animations in Hit Effects"):
  - **Kill the cross-fade** into a hit clip — snap in with ~zero blend time so the impact is not "softened."
  - **Fast-in, slow-out** timing (quick jolt, slow settle) conveys kinetic-energy transfer, not a floaty wobble.
  - **Additive shake** overlaid on an *uninterruptible* animation when you can't cancel the base state.
  - **Hit-pause / hit-stop** (brief time-freeze on connect, borrowed from fighting games) + camera shake + spark VFX + spine **IK post-processing** for non-repeating variation.
- **Asset-market baseline** (shows the expected shape of a full set): *HitReact Pro* ships **127 hit-reaction animations**, directional + body-part + additive, "zero scripting"; *Ultimate Hit Reaction System* ships **168 mocap clips** across directions Front/Back/Left/Right/Up/Down × strengths Light/Medium/Heavy/**Explosion**. These confirm the industry's canonical decomposition: **direction × body-part × strength**. (Fab listings — *flagged: vendor pages, counts as advertised.*)

### 2b. Where physics *does* enter a localized hit
- Unreal's exact recipe for a **projectile-struck limb**: `Set All Bodies Below Simulate Physics` at the hit bone (e.g. left clavicle → arm goes physical), then drive **`Physics Blend Weight` up to 1.0 and back to 0.0** over ~0.1–0.3 s so the reaction blends in then out, with the impulse applied to the struck bone. (Epic "Physics-Based Animation"; MoCap Online.) This is a *partial-body blended ragdoll* used surgically, not a full ragdoll.
- The honest tradeoff from a shipped hack-and-slash dev: they *tried* ragdoll-follows-animation-with-external-forces and **concluded it wasn't worth it** for their non-dynamic combat — authored additive reactions were enough. (Game Developer, "The Role of Animations in Hit Effects.") A useful counterweight to "physics everything."

**Localized-hit verdict:** additive flinch layer first; reserve the partial-body physics blend for heavier localized hits (shotgun at range, a wound stagger). At a **true top-down 90° camera with 32×32 sprites**, even the additive layer is over-engineered for round one — a 1–2 frame sprite "jolt"/tint + tiny positional kick reads as a hit (§7, Rung 0).

---

## 3. Directional & located reactions (which way, which limb)

The canonical decomposition is **direction × body-location × strength** (§2a).

- **Direction:** compute hit direction relative to the character's facing (attacker→victim vector projected into the character's frame) → bucket into Front/Back/Left/Right (optionally 8-way) → pick the matching additive/full clip. Top-down Breach already has a continuous float `facing`, so the relative-direction bucket is trivial to compute render-side; the *authoritative* direction (if it gates gameplay, e.g. back-shot bonus) is a Q16.16 `atan2_q16` in the sim. (MoCap Online; kolosdev shooter tutorial.)
- **Body location:** raycast/hit-capsule reports which bone/region was struck (head/torso/arm/leg); play the region's reaction and optionally apply damage multipliers / weak-spots. (Fab HitReact Pro; kolosdev.)
- **Procedural aim/look + flinch:** additive layers commonly combine with **aim-offset / look-at IK** so a hit briefly perturbs the aim pose before it re-settles.
- **Euphoria-style "wound reaching"** (the aspirational end of located reactions): NaturalMotion's Dynamic Motion Synthesis has characters **reach a hand toward the exact wound location** where they were shot, stagger to keep balance, and stay standing — synthesized fresh each time, never identical. (Sibling 03 §2; RDR2 W.E.R.O. mod notes describe "reach for the exact wound instead of always the stomach" — *flagged: mod-community description of Rockstar's RAGE Euphoria behavior, not a primary Rockstar source.*) This is the gold-standard *located* reaction and, per sibling 03, the cautionary tale on cost — you do not run it on a crowd.

---

## 4. Explosion / grenade knockback ("blown out from a grenade")

The whole-body-launch case. The shipped recipe everywhere is: **detect who's in radius → compute a radial impulse with distance falloff (+ optional upward bias, + line-of-sight/cover test) → hand the impulse to a ragdoll → launch, tumble, land, get up.**

### 4a. The two engine primitives (copyable formulas)
- **Unity `Rigidbody.AddExplosionForce(explosionForce, explosionPosition, explosionRadius, upwardsModifier = 0, mode = ForceMode.Force)`** — models the blast as a sphere; **force decreases in proportion to distance from the centre** (radius 0 ⇒ full force regardless of distance). `upwardsModifier` shifts the apparent application point **downward on Y**, which levers objects **up into the air** for a more dramatic launch (the "pop up" every explosion uses). "Plays nicely with ragdolls." Typical usage: `Physics.OverlapSphere` to gather bodies, then `AddExplosionForce` per rigidbody. (Unity Scripting API, `Rigidbody.AddExplosionForce`.)
- **Unreal `AddRadialImpulse(Origin, Radius, Strength, Falloff, bVelChange)`** — "adds an impulse to all rigid bodies in a component, radiating out from a position … best described as a bomb explosion." Falloff is **`RIF_Linear`** (linear weakening to the radius edge) or **`RIF_Constant`** (full strength everywhere inside). Companion **`RadialForceComponent` / `FireImpulse()`** packages it on an actor (e.g. a grenade), but **note it ignores walls** — you affect actors *through cover* unless you add a manual line-of-sight/visibility filter, which is exactly what a "grenade behind a wall shouldn't ragdoll you" needs. (Epic "Add Radial Impulse" docs; unrealcpp.com; Epic forums "About Radial Force Component.")

### 4b. The production launch pipeline
1. **Gather** bodies in radius (`OverlapSphere` / sphere sweep).
2. **Falloff** the impulse by distance (linear is the default; inverse-square is more physical — see §5).
3. **Line-of-sight / cover** test per body (raycast blast-center→body); scale or zero the impulse if blocked. (Epic forum note on the through-walls limitation is the standard motivation.)
4. **Upward bias** so bodies pop rather than skid (Unity `upwardsModifier`; Unreal usually a manual `+Z` term).
5. **Switch the character to ragdoll** and apply the impulse (Unreal: `Set … Simulate Physics` + `AddImpulse`, or blend weight 0→1 first for a softer onset; the correct impulse magnitude is a well-known tuning pain — Epic forum "what is the correct impulse value?").
6. **Land → detect front/back → get-up clip → blend physics weight back to 0** (sibling 03 §1c/§4, Rung 2 — the get-up transition is the craft cost, not the compute cost).

**Grenade verdict:** this is squarely the **full-ragdoll-launch** case where physics beats authored — no clip set covers every launch vector + spin + landing. It maps directly onto **sibling 03's Rung 2 (blended → full ragdoll)**, differing only in the *trigger* (radial impulse from a blast center vs a capsule-overlap collision) and in **concurrency** (a grenade in a squad launches several bodies at once — the worst case, §8).

---

## 5. Shockwaves / blast-waves specifically (the pressure front)

Breach already has a **shockwave/blast-propagation concept in-sim** (EOS gas, wall-bursts, electricity). The novel part for this theme is the **character's reaction to a shockwave the sim produces** — and a shockwave differs from the point-impulse of §4 in physically meaningful, animation-relevant ways:

- **It arrives over time, not instantaneously.** A blast wave is "an area of pressure expanding supersonically outward from an explosive core"; at any point there's an **instantaneous overpressure jump (the shock front)** followed by an **exponential decay**, then a **negative-pressure phase (blast wind / suction back toward the center)** before returning to baseline. (Wikipedia "Blast wave"; SAGE "Blast wave interaction with structures," Isaac et al. 2023; NASA NTRS "Propagation and Interaction of Spherical Blast Waves.")
- **Intensity falls with both distance and time** as energy spreads over a growing volume — "the intensity of the blast wave always decreases with time and distance." (SAGE; ScienceDirect "Blast Wave.") Physical scaling is stronger than linear: peak overpressure roughly follows a **cube-root (Hopkinson–Cranz) scaled-distance** law, i.e. far closer to inverse-square/inverse-cube than the linear falloff games default to. *Flagged: cube-root scaling is standard blast physics; I'm asserting the qualitative shape, not prescribing an exact formula for Breach.*

**What this buys the animation** — a shockwave reaction that is *richer than a point impulse* without new tech:
- **Timed radial force, not a single frame.** Instead of one `AddImpulse`, drive the character with a **short force ramp** whose arrival is delayed by `distance / wave_speed` and whose magnitude follows the sim's decaying pressure — an *expanding ring* hits nearer marines first and harder. This "expanding radial force over time" is the shockwave's signature and is what makes a wall of bodies fold outward in sequence rather than simultaneously.
- **Directional blowback + a suck-back beat.** The dominant motion is **directly away from the blast center** (align the launch/tumble to the radial direction); the physical **negative-phase suction** can be faked as a small pull-back settle at the tail for extra realism (optional flourish).
- **Same falloff/cover machinery as §4** — a shockwave is a §4 explosion with (a) a propagation delay, (b) a steeper (cube-root/inverse-square) falloff, and (c) a decaying-over-time magnitude. If Breach's sim already emits the pressure field per tile/tick, the animation just **samples that field** for arrival-time and magnitude rather than inventing a falloff.

**No dedicated "physically-based blast response for game characters" paper surfaced.** Academic blast-wave literature is injury/structural (blast-injury neurotrauma, piezoresistive pressure sensing, structure interaction), **not** character animation. The character-animation treatment of shockwaves in games is, in practice, **§4's radial impulse with a time-delayed, steeper-falloff, decaying profile** — there isn't a separate research lineage to cite. *Flagged as a genuine gap, not an omission.*

---

## 6. Learned / physics impact response — the ML angle

A shove, a blast, a bullet impulse are all **external perturbations**, and the physics-control literature's headline robustness result is exactly *recovering from perturbations*. This is on-theme for an RL project, not a detour. (Depth on these models: sibling 02.)

- **DeepMimic (Peng et al. 2018, arXiv:1804.02717):** imitation-trained PD policies are **robust to significant external pushes and produce plausible recoveries — despite never being trained on perturbations.** The paper attributes this emergent robustness to the **stochastic policy's exploration noise during training**. Translation for Breach: a policy trained just to walk/fight will *already* stagger-and-recover from a modest blast impulse for free; whether it goes down is emergent from impulse magnitude vs. learned balance. (DeepMimic, xbpeng.github.io / ar5iv 1804.02717.)
- **DReCon (Bergamin, Clavet, Holden, Forbes; SIGGRAPH Asia 2019) & Ubisoft "Ragdoll Motion Matching" (GDC 2020):** motion-matching target + deep-RL PD controller that **balances with its own strength and recovers from unplanned perturbations**, then blends straight back to responsive locomotion — this *is* the "hit, stagger, recover, keep going" loop, learned. Low runtime cost was an explicit design goal. (Sibling 03 §3; DReCon PDF.)
- **SuperTrack (Fussell, Bergamin, Clavet; Ubisoft La Forge, SIGGRAPH Asia 2021):** learns a **differentiable world model** and tracks motion via **supervised** learning (cheaper/more stable to train than PPO), producing the same physically-simulated-character-follows-animation-and-reacts family. Recovery from impulses comes from the tracking staying physical. (Semantic Scholar "SuperTrack"; sibling 02/03.)
- **Explicit push-recovery / adversarial-perturbation RL (robotics-adjacent, transfers by analogy):**
  - "On the Emergence of Whole-body Strategies from Humanoid Robot Push-recovery Learning" (arXiv:2104.14534) — model-free DRL trains a **general robust push-recovery policy**; whole-body strategies (ankle/hip/stepping) *emerge*.
  - "Keep on Going: Learning Robust Humanoid Motion Skills via Selective Adversarial Training" (arXiv:2507.08303) — a **learnable adversary** finds the policy's weak points and perturbs them with minimal budget → tougher recovery. This is literally "train against something that tries to knock you over."
  - "Stubborn: … Robust Motion Tracking and Fall Recovery for Humanoids" (arXiv:2606.12814) and "Learning Humanoid Standing-up Control across Diverse Postures" (arXiv:2502.08378) — the **get-up-after-you-do-go-down** half. (*Flagged: 2025–26 robotics arXiv, title/abstract level; transfer to game characters is by analogy, and they're float NN + float physics = render-layer only.*)

**The ML payoff for Breach specifically:** "getting hit and **staying up vs going down**" need not be an authored threshold — it can be **emergent** from a trained controller answering the sim's impulse. That is precisely the kind of behavior an RL-first project would *want* to be emergent. But it's a full arc (training pipeline + float physics), lives entirely render-side (§9), and is Rung 4.

---

## 7. The staged impact menu (cheapest → richest)

Mirrors sibling 03's ladder, specialized to impacts. Each rung is a complete shippable answer; stop where it looks good enough. **Trigger, common to all:** the sim emits an impact event with a **direction, a magnitude/impulse, and a body-location** (bullet: from damage source; grenade/shockwave: radial vector from blast center, magnitude from falloff). All rungs consume that sim-authored event.

- **Rung 0 — Sprite-only reaction (buildable in Breach *today*, no physics).**
  - *Localized hit:* 1–2 frame sprite "jolt"/hit-flash/tint + a tiny positional kick opposite the shot; directional bucket from `facing` picks which way to nudge.
  - *Grenade/shockwave:* swap to a "knocked/prone" sprite + a **scripted knockback slide of the unit's position, distance & direction scaled by the sim's impulse**, easing out, plus a billboard **tumble/spin**. An expanding shockwave nudges nearer units first (sample arrival time).
  - **Cost ≈ zero.** From a true 90° top-down camera a well-timed slide + spin already reads as "blown back." **Recommended first build** — it tests whether the *idea* reads before any 3D/physics investment.
- **Rung 1 — Render-3D clip + impulse-driven root motion (no ragdoll).** When marines become render-only 3D meshes: additive **directional flinch clips** for bullets; a **"launched + get-up" clip pair** for grenades, with **knockback direction/distance derived from the sim's Q16.16 impulse**. No solver. (Sibling 03 Rung 1.)
- **Rung 2 — Blended partial → full ragdoll on impact (the sweet spot).**
  - *Bullet:* partial-body physics blend on the struck limb (§2b), weight 0→1→0.
  - *Grenade/shockwave:* full ragdoll + radial impulse (§4), land, detect front/back, get-up clip, blend weight →0. Only felled/launched units simulate. **This is what most AAA ships for knockdowns/launches** and is the industry default. (Sibling 03 Rung 2.)
- **Rung 3 — Always-on active (PD) ragdoll.** Every character is a powered ragdoll tracking animation; **drop muscle strength on impact** so physics dominates, ramp back up so the controller drives its *own* get-up. Best-looking non-learned launch; priciest continuous cost (§8). (Sibling 03 Rung 3.)
- **Rung 4 — Learned perturbation-recovery controller.** DReCon/SuperTrack-style PD policy tracking motion-matched locomotion; the blast impulse is just a perturbation the policy **recovers from or falls to, emergently** (§6). On-theme for an RL project; a real arc, not a weekend. (Sibling 03 Rung 4; sibling 02.)

**Practical pick:** **Rung 0 now** (proves the top-down read at zero cost). If 3D marines earn it, jump to **Rung 2** for grenades/shockwaves (bounded cost — only launched units simulate) with additive flinch (Rung 1-style) for bullets. Reserve Rung 3/4 for when "impacts look incredible" becomes a signature feature — which, given Erik's stated priority, is plausible later.

---

## 8. Compute-cost reality check

- **The grenade-in-a-squad worst case.** A single blast launching **N marines at once = N simultaneous ragdoll starts**. This is *the* concurrency spike for this theme (more bursty than sibling 03's pairwise bump). Budget for the **peak**, not the average.
- **Concurrent budget.** MoCap Online's rule of thumb ≈ **8–12 simultaneously active ragdolls**; community reports frame-rate cliffs **past ~20 active units** on modest/mobile hardware. Breach's "tens of units" fits **only if just the currently-launched ones simulate** (Rung 2), not everyone always (Rung 3). A big grenade into a dense squad could exceed the comfortable count — cap concurrent ragdolls and let overflow units use Rung 0/1 knockback.
- **Physics LOD (standard mitigation).** Tiered: near = all bodies (+cloth); mid = reduced bone set (pelvis/spine/head/upper limbs, skip fingers/feet/secondary spine); far (~20–40 m) = **no active ragdoll at all**. **Sleep** bodies once at rest — "a sleeping physics body costs almost nothing." (Number Analytics "Advanced Ragdoll Physics Techniques"; MoCap Online.) Extra levers: drop physics tick 60→30 Hz, per-bone→single-capsule colliders, disable off-screen. GPU rigid-body solvers (PhysX 5 GPU, 2022) exist for crowds but are overkill here and non-deterministic.
- **Bone budget.** 12–20 physics bodies, not the full deform skeleton (sibling 03 §5).
- **Top-down 2D/planar simplification (big win for Breach).** Because the camera is a true 90° top-down, an impact reaction can plausibly be a **cheap 2D/planar rigid-body chain** (a few jointed capsules in the ground plane, or even a single capsule + spin) rather than full 3D — dramatically cheaper and possibly *sufficient* for the top-down read. Strongly worth a spike before committing to 3D ragdoll (echoes sibling 03 §5).

---

## 9. Determinism hand-off (defer to sibling 05 for specifics)

The fence, applied to impacts:

- **Sim (Q16.16, deterministic, authoritative):** the blast **radius, falloff, and — if it exists — the pressure/force field itself**; the **line-of-sight/cover** test; **who's in radius**; the **knockback vector and magnitude** (via `atan2_q16`/`sin_q16`/`cos_q16` from `fixed_point.h`); **who dies**; **final knocked-back position**; **prone/stun flags and durations**. Every number any machine must agree on. Breach's existing blast/shockwave sim is the source of truth — the animation **consumes** it.
- **Render (float, exempt):** the *visual* flinch / tumble / launch / get-up — Unity/Unreal-style radial impulse, PD ragdoll, or a learned policy — played out **between the sim-authored start state and the sim-authored end state**. Cosmetic interpolation over deterministic endpoints. Each client may simulate the ragdoll differently without desync.
- **The one hard rule:** **never let the float ragdoll's output feed back into synced state.** The launch is a downstream consumer; the final resting position that matters for gameplay is the *sim's* Q16.16 value, not wherever the float solver happened to stop. (This is exactly the class of nondeterminism — BLAS/LAPACK in spawn-RNG — that bit the X-ARCH investigation; keep the reaction strictly downstream.)
- **Bigger monsters with real collisions:** same split — collision *detection/resolution that affects gameplay* stays Q16.16; *animated flesh/secondary motion* is render-only. The test is always "does any machine need to agree on this number?"

**Canon-fold candidate (impacts):** *A blast emits a Q16.16 impulse field (vector + magnitude + prone/stun) from the sim; the render layer's flinch/launch/tumble is float decoration stretched between the sim's start and end states and never feeds back.*

---

## 10. Breach-fit recommendation

Given Breach **already simulates blasts/shockwaves**, the cheapest convincing reaction is to **let the animation *consume the impulse the sim already produces*** rather than compute anything new:

1. **Ship Rung 0 now.** Have the render layer read the sim's per-unit impact event (direction + magnitude + body-location) and play: for bullets a 1–2 frame directional sprite jolt/flash; for grenades/shockwaves a **magnitude-scaled knockback slide + billboard spin**, with an **expanding-ring arrival delay** so a shockwave folds the nearer marines outward first. Zero new tech, tests the top-down read, and is honest about Breach's "deliberately cheap" ethos. This alone may satisfy "blown out from a grenade."
2. **When (if) marines become render-only 3D:** add **additive directional flinch** for bullets (Rung 1/2b partial blend) and **Rung 2 full-ragdoll launch** for grenades/shockwaves — felled-units-only simulation, physics-LOD, and a **cap on concurrent ragdolls** (the squad-grenade worst case) with overflow falling back to the Rung 0 slide. Strongly consider the **planar 2D ragdoll spike** first (§8) — it may be enough and is far cheaper.
3. **Keep the fence (§9) from day one:** the impulse/falloff/cover/who-dies/final-position/prone-stun are Q16.16 sim; the launch/tumble is float render over those endpoints. This keeps every richer rung (2→4) a **render-only upgrade** that never threatens determinism.
4. **Rung 4 (learned recovery) is the on-theme aspiration**, not the starting point: a trained controller answering the sim's blast impulse makes "stay up vs go down" *emergent* — exactly what an RL-first project wants — but it's a full arc (sibling 02/03) and pure render-layer.

**One line:** *Breach already computes the blast — make the marine visually answer it, render-only, starting with a magnitude-scaled sprite knockback + spin, and reserve ragdoll for the grenade-launch case where authored clips can't win.*

---

## 11. If you read three things

1. **Unity `Rigidbody.AddExplosionForce` docs** — https://docs.unity3d.com/ScriptReference/Rigidbody.AddExplosionForce.html — the clearest, copyable radial-impulse-with-falloff + `upwardsModifier` "pop up" formula; the exact primitive the grenade/shockwave launch is built on.
2. **Game Developer, "The Role of Animations in Hit Effects"** — https://www.gamedeveloper.com/programming/the-role-of-animations-in-hit-effects — the craft of localized hit *feel* (kill the cross-fade, fast-in/slow-out, additive shake, hit-pause) and an honest "we skipped ragdoll and authored was enough" — the small-force rule in practice.
3. **DeepMimic (Peng et al. 2018, arXiv:1804.02717)** — https://xbpeng.github.io/projects/DeepMimic/ — the emergent-robustness result: policies recover from big external pushes *without being trained on them*; the ML answer to "getting hit and staying up," on-theme for an RL game.

*(Bonus, Breach-specific: sibling `03_ragdoll_reactions_fall.md` §4 (the rung ladder this extends) and Epic's "Add Radial Impulse" / `RadialForceComponent` docs — https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Physics/AddRadialImpulse — including the "affects actors through walls" gotcha that motivates a cover test.)*

---

## 12. Flagged / uncertain citations

- **Blast cube-root (Hopkinson–Cranz) scaling** — standard blast physics; I assert the *qualitative* shape (steeper-than-linear falloff, arrival-over-time, negative phase) from general blast-wave sources (Wikipedia; SAGE Isaac et al. 2023; NASA NTRS). I did **not** derive an exact Breach falloff formula — treat as directional.
- **No character-animation shockwave-response paper exists** — genuine gap. Academic blast literature is injury/structural, not animation. The game treatment is §4's radial impulse with delay + steeper falloff + decay; there is no separate research lineage to cite. Flagged as a finding, not a miss.
- **Euphoria "reach for the exact wound"** — described from RDR2/GTA **mod-community** pages (W.E.R.O., GTAForums), not a primary Rockstar/NaturalMotion source. Directionally accurate to Euphoria's known behavior; not authoritative on implementation.
- **Asset counts (HitReact Pro 127, Ultimate Hit Reaction 168)** — Fab vendor listings; counts as advertised, used only to show the direction×body×strength decomposition, not as technique authority.
- **Unity `AddExplosionForce` "proportion to distance"** — the doc says force "decreases in proportion to distance" and gives `upwardsModifier` behavior, but does **not** publish an exact falloff equation; whether it's strictly linear is unstated. Unreal `RIF_Linear` *is* explicitly linear.
- **"~20 active units" frame-rate cliff / physics-LOD tiers / 20–40 m cutoff** — from Number Analytics and Unity/Qt forum guidance; rules of thumb, hardware-dependent, not hard numbers.
- **Robotics push-recovery / adversarial papers (arXiv 2104.14534, 2507.08303, 2606.12814, 2502.08378)** — real arXiv results, title/abstract level here; robotics, transfer to game characters by analogy, and float-only (render-layer).
- **SuperTrack venue (SIGGRAPH Asia 2021)** — from memory of the abstract via Semantic Scholar; confirm year/venue before quoting. DReCon/DeepMimic IDs are solid (DeepMimic arXiv:1804.02717; DReCon SIGGRAPH Asia 2019).

### Source list (used)
- Unity `Rigidbody.AddExplosionForce`: https://docs.unity3d.com/ScriptReference/Rigidbody.AddExplosionForce.html
- Unreal Add Radial Impulse: https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Physics/AddRadialImpulse · unrealcpp: https://unrealcpp.com/add-radial-impulse-to-actor/ · RadialForceComponent (through-walls): https://forums.unrealengine.com/t/about-radial-force-component/475699
- Unreal Physics-Driven / Physics-Based Animation (blend weight 0→1→0): https://docs.unrealengine.com/4.26/en-US/AnimatingObjects/SkeletalMeshAnimation/PhysicallyDrivenAnimation · "correct impulse value?" https://forums.unrealengine.com/t/going-ragdoll-after-a-physics-hit-what-is-the-correct-impulse-value/455523
- Game Developer, "The Role of Animations in Hit Effects": https://www.gamedeveloper.com/programming/the-role-of-animations-in-hit-effects
- MoCap Online — Ragdoll blend guide: https://mocaponline.com/blogs/mocap-news/ragdoll-physics-animation-guide · Blend Trees: https://mocaponline.com/blogs/mocap-news/animation-blend-tree-guide · Third-person: https://mocaponline.com/blogs/mocap-news/third-person-animation
- HitReact Pro (Fab): https://www.fab.com/listings/dd1558d5-ce68-4e93-a931-954edd89d974 · Ultimate Hit Reaction System (Fab): https://www.fab.com/listings/0c79e08b-0440-4dfc-9321-79ba4a9dcbe3 · kolosdev shooter hit reactions: https://kolosdev.com/shooter-tutorial-base-enemy-hit-reactions-behavior-tree/
- Blast wave (Wikipedia): https://en.wikipedia.org/wiki/Blast_wave · SAGE, Isaac et al. 2023 "Blast wave interaction with structures": https://journals.sagepub.com/doi/10.1177/20414196221118595 · NASA NTRS "Spherical Blast Waves": https://ntrs.nasa.gov/api/citations/20130011523/downloads/20130011523.pdf · ScienceDirect "Blast Wave": https://www.sciencedirect.com/topics/materials-science/blast-wave
- DeepMimic (Peng et al. 2018): https://arxiv.org/abs/1804.02717 · project: https://xbpeng.github.io/projects/DeepMimic/
- DReCon (SIGGRAPH Asia 2019) PDF: https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf · SuperTrack (Semantic Scholar): https://www.semanticscholar.org/paper/SuperTrack-Fussell-Bergamin/aaac91a5ebef1ec06f15434fa1a3e15c4ed82d12
- Push-recovery / adversarial RL: arXiv:2104.14534 https://arxiv.org/pdf/2104.14534 · "Keep on Going" arXiv:2507.08303 https://arxiv.org/html/2507.08303v3 · "Stubborn" arXiv:2606.12814 https://arxiv.org/pdf/2606.12814 · Standing-up arXiv:2502.08378 https://arxiv.org/pdf/2502.08378
- Euphoria wound-reaching (mod-community): RDR2 W.E.R.O. https://www.rdr2mods.com/downloads/rdr2/other/85-rdr-2-wero-euphoria-ragdoll-overhaul/ · GTAForums Bullet Impact Euphoria https://gtaforums.com/topic/991318-bullet-impact-euphoria-an-authentic-ragdoll-on-damage-mod/
- Physics LOD / crowd cost: Number Analytics "Advanced Ragdoll Physics Techniques": https://www.numberanalytics.com/blog/advanced-ragdoll-physics-techniques
