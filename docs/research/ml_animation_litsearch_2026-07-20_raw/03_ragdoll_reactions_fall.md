# Ragdoll, Hit-Reactions & the "Sprint-Into-Each-Other-and-Fall" Problem

**Agent 3 of 5 — ML/physics animation lit search for Breach**
**Date:** 2026-07-20
**Scope:** the bridge between keyframe animation and full physics control — shipped-in-games techniques for collisions, staggers, knockdowns, falls, and recoveries. Focus on the *practical/production* angle and the specific agent-vs-agent bump-and-fall scenario Erik cares about. (The pure-research physics-control lineage — DeepMimic/AMP/SuperTrack — is owned by the physics-control agent; I touch it only where it lands in shipped reaction tech.)

**Capture, not canon.** Every claim is sourced or flagged. Breach today: pyray/raylib, true 90° top-down camera, static 32×32 sprites, continuous float `facing`, no animation system yet, deliberately cheap. Determinism iron rule: synced sim state is Q16.16 integer, no floats in sim path; render layer exempt.

---

## 1. Taxonomy: dead vs active vs blended ragdoll

Ragdoll physics is a **procedural animation technique that treats a character as a set of interconnected rigid bodies (bones/capsules) joined by constraints, driven by a physics solver** rather than by keyframes. Three points on a spectrum (industry treats it as a spectrum, not hard buckets — MoCap Online):

### 1a. Classic / "dead" ragdoll (passive)
- Physics takes over **completely** on death/knockback. No motors, no target pose — gravity + constraints + collision only. The character goes limp.
- Cheapest and oldest (Hitman: Codename 47, 2000 is the usual "first" cite). Trivial to trigger: swap the animated skeleton for a simulated one, apply an impulse.
- Looks *dead* — great for corpses, wrong for anything that should look alive/reacting. The "boneless flopping corpse" meme lives here.
- **Cost:** one rigid body per bone (typically 12–20), a constraint solver pass. Very cheap.

### 1b. Active / powered ragdoll (motor-driven, PD-controlled)
- The ragdoll bodies carry **joint motors** (PD / proportional-derivative controllers, or spring drives) that apply torques to drive each joint toward a **target pose from an animation**. The character "moves and balances entirely using its own strength" (DReCon's framing).
- Standard construction is a **dual rig**: (1) a hidden, non-physical **animated reference** playing normal animation; (2) the visible **physical ragdoll** that continuously tries to *mimic* the reference by applying per-joint torque proportional to the angular error (and damping proportional to angular velocity) — the textbook PD law `τ = kp·(θ_target − θ) − kd·θ̇`. (Sergio Abreu; EggyStudio Unity.Humanoid.ActiveRagdoll.)
- Can **weaken (drop `kp`) on collision and then recover**, giving natural reactive motion without ever leaving physics (EggyStudio). This is the key trick: strength is a dial, not a binary.
- Two sub-styles (Jan Schneider / Medium):
  - **Inside-out**: each bone torques to match its parent's relative orientation in the reference rig — most animation-faithful.
  - **Outside-in**: an external target drags a chain (Octodad's cursor-on-a-hand, TABS weapons) — wobbly, comedic marionette feel.
- **Balance is the hard part.** Naive active ragdolls fall over. Cheap fixes (Sergio Abreu): lock hip rotation (rigid, unstable), pin hips to a kinematic stabilizer body, or apply manual "upright" torque. Real balance needs a controller (see DReCon/SuperTrack/Ubisoft §3).
- **Cost:** same body count as dead ragdoll + PD evaluation per joint per substep + higher solver iteration counts (Unity guidance: solver iterations >8, max angular speed >20 for stability). Noticeably pricier than dead ragdoll, mostly from needing a stiff, stable solve.

### 1c. Blended ragdoll (kinematic animation ⟷ physics, weighted)
- The **professional default for hit reactions.** The character plays normal keyframed animation, and on impact you **blend in physics on a subset of bones** via a 0→1 weight, then blend back out. Nothing is fully limp; nothing is fully animated.
- MoCap Online's blunt summary: *"the hardest problem in ragdoll implementation is not activating the physics simulation — it is making the transition invisible to the player."* The activation is easy; the **blend-in and especially blend-out (get-up)** are where all the craft is.
- Unreal's concrete recipe (Epic docs, "Physics Driven Animation"): `Set All Bodies Below Simulate Physics` at a chosen bone (e.g. left clavicle → whole arm goes physical), then drive `Set All Bodies Below Physics Blend Weight` **up to 1.0 and back to 0.0** so the reaction blends in then out. Partial-body (upper-body-only) blends keep the legs animated and locomoting while the torso/arms react to a hit.
- **On impact**: blend weight ramps 0→1 over ~0.1–0.3 s, impulse applied, then held/decayed (MoCap Online).
- **Get-up**: sample the ragdoll's final pose, detect front vs back, pick the matching get-up animation, and blend physics weight back to 0 — the most technically demanding phase. Alternatives: pose-snapshot matching, IK-driven kinematic recovery.

**Rule of thumb from the field:** *light* reactions (flinches, staggers) look **better authored** than simulated — small forces make physics look chaotic/twitchy. Reserve physics for *big* events (knockdowns, deaths, launches, and — for Breach — full-body collisions). (Epic docs; MoCap Online.)

---

## 2. Euphoria & production reaction tech

### NaturalMotion Euphoria (the landmark)
- **Dynamic Motion Synthesis (DMS)**: procedural, real-time character behavior "based on a full simulation of the character including body, muscles and motor nervous system" — biomechanics + AI + physics, CPU-based. Ships in GTA IV/V, Red Dead, Star Wars: The Force Unleashed, Backbreaker. (Game Developer / GTA Wiki / HandWiki.)
- **What made it special:** reactions are *synthesized every time*, never identical on replay — characters reach for wounds, grab ledges, stagger to keep balance, brace before impact. It covers "nearly all animated behaviors": firearms, melee, jumps, climbs, recoveries, world interactions.
- **Cost/complexity reputation:** licensing reportedly **$100k+** (forum-level claim — *flagged, low confidence*). CPU-heavy — it "uses the CPUs of next-gen platforms to synthesize on the fly," integrated into Rockstar's RAGE engine on PS3/360. Widely described as **expensive, hard to author (tuning behavior trees of physical responses), and reserved for a handful of on-screen actors** — you do not run Euphoria on a crowd. It is the gold standard *and* the cautionary tale for "full procedural reaction is costly."
- **Breach relevance:** aspirational only. It's the proof that fully-simulated reactions look incredible, and the reason nobody ships them at scale on a budget.

### EA / Frostbite — "Physics-Driven Ragdolls and Animation: From Sports to Star Wars" (GDC 2018, Jalpesh Sachania)
- Ships **driven (active) ragdolls that follow animation and react to interactions** across FIFA/sports and Star Wars Battlefront. Emphasis on *emergent* reactions instead of growing the anim database.
- Practical themes (the useful part): **improving animation follow, reducing complexity to gain performance, reducing bad output poses, feeding physics results back into the animation system, and networking** driven ragdolls. Also used ragdolls to **generate procedural animations offline**. (GDC Vault abstract.) The "reduce complexity for perf" and "avoid bad poses" bullets are exactly Breach's concerns.

### Unreal — PhysicalAnimationComponent / Physics Control plugin
- **PhysicalAnimationComponent**: drives skeletal-mesh bodies toward the current animated pose using spring-like strength params (orientation + optionally position strength) — Unreal's built-in active/blended ragdoll driver. Used for hit reactions on top of locomotion. (Epic docs.)
- **Physics Control Component** (newer, UE5): a more flexible successor for authoring per-body/per-set control strengths and damping at runtime — the modern recommended path for physics-blended reactions and get-ups. (Epic docs; forum threads on Control Rig hit reactions.)
- Marketplace: **"Physics Get Up Blend / Ragdoll Replication"** productizes the exact get-up-from-ragdoll + network-replication problem.

### Unity — active-ragdoll ecosystem
- **PuppetMaster** (RootMotion, author of Final IK): the de-facto Unity active-ragdoll asset. **Dual rig** ("Target" animated + "Puppet" ragdoll); turns each `ConfigurableJoint` into a **"Muscle"** that computes joint target rotations, and adds **"Pins"** — `AddForce` springs pulling each ragdoll bone toward its animated target's world position. Muscle strength + pin weight are dials you drop on impact and restore for recovery. Pins are "unnatural god forces" that let the ragdoll hit poses real muscle couldn't — a deliberate cheat for game feel. One-click biped setup. (root-motion.com.)
- **Unity Physical Animation** (built-in, via ConfigurableJoint target rotation / ArticulationBody) + community assets (EggyStudio, Sergio Abreu's open-source repo) cover the DIY path.

---

## 3. Learned reactions (where ML enters the shipped pipeline)

These matter to Breach because it's an RL project — a learned reaction controller is *on-theme*, not a detour. But note: all of these are **float neural-net + float physics**, so they live in the **render layer**, not the deterministic sim (see §6).

### DReCon — Bergamin, Clavet, Holden, Forbes (Ubisoft La Forge, SIGGRAPH Asia 2019)
- **The most Breach-relevant paper.** Two-step responsive controller from **unstructured mocap**, explicitly targeting *"responsiveness to user input, visual quality, and low runtime cost for video games."*
- **Step 1 — Motion Matching** picks a target pose stream from the mocap DB given user input. **Step 2 — a deep-RL policy** drives a **PD-controlled physical character** to track that stream. The character **balances entirely with its own strength** and its real physical interactions (pushes, bumps, uneven ground) are the physics solver's, not scripted.
- **Cost:** roughly **2–4 ms/frame** (from fetch of the PDF — *treat exact number as approximate, verify against paper*), i.e. cheap enough to have been a real-time game target in 2019.
- **Why it matters for the bump-and-fall:** it recovers from **unplanned perturbations** (pushes) automatically and blends straight back to responsive locomotion — this IS the "bump, stagger/fall, get up, keep going" loop, learned rather than authored.

### Ubisoft "Ragdoll Motion Matching" (GDC 2020, Simon Clavet)
- The production/talk sibling of DReCon. A **virtual robot trained with deep RL to follow motion-matching output**, powered by **its own motors — no "god forces"** attaching limbs to global points (contrast PuppetMaster's pins). It **learns to balance** as it follows user-controlled mocap and **recovers from unplanned perturbations**. Trust the physics solver as ground truth for character↔environment interaction. (gameanim.com; GDC Vault.)

### SuperTrack (Ubisoft La Forge) — motion tracking via *supervised* learning
- A follow-on that learns a world model + policy to track motion with **supervised** learning (cheaper/more stable to train than pure RL). Same family: physically-simulated character faithfully following animation, reacting physically. (Ubisoft La Forge — *flagged, only skimmed via search.*)

### Learned get-up / fall recovery (research, robotics-adjacent)
- Deep body of RL work on **standing up from prone/supine and recovering mid-fall**: DeepMimic (Peng et al. 2018, example-guided), AMP (adversarial motion priors), "Learning to Get Up" (2022, discovers *distinct* strategies for face-down vs face-up starts *without* mocap), plus a 2025+ wave of real-humanoid-robot getting-up policies (FRASA, HiFAR, cross-morphology zero-shot). Directly usable as the **recovery half** of a bump-and-fall if Breach ever wants learned (vs authored) get-ups. (arXiv, multiple — see citations.)

---

## 4. THE bump-and-fall recipe (staged menu, cheapest → richest)

Erik's one emphatic scenario: **two marines sprint into each other and fall over, and it looks cool.** Below, a ladder. Each rung is a complete shippable answer; stop at whatever looks good enough.

**Trigger, common to all rungs:** on a frame where two units' capsules overlap **and** their **relative approach velocity exceeds a threshold**, fire a "collision knockdown" event for one or both. (Relative velocity gate is what separates a gentle nudge from a sprint-slam. Multiple ragdoll colliders re-fire collisions, so debounce to one event per pair per hit — Unity forum.)

- **Rung 0 — Sprite-only fake (no physics at all).** On the event, play a canned "stumble/fall" sprite sequence (or, in Breach's current stack, swap to a "prone" sprite + a short scripted knockback slide of the unit's position, easing out). Add a rotation/tumble on the billboard. **Cost: ~zero.** Fully authorable in the current top-down sprite renderer. This is the honest first thing to try — from a true 90° top-down camera, a well-timed sprite tumble + knockback slide may already read as "cool" and costs nothing. **Recommended first build.**

- **Rung 1 — Canned animation + impulse-driven position (render 3D, no ragdoll).** If/when marines become render-only 3D meshes: play a **motion-captured/handmade "knocked-down" + "get-up" animation pair**, but derive the **knockback direction and distance from the actual collision impulse** (momentum transfer: heavier/faster unit wins, `Δv ∝ relative momentum`). Root-motion or a scripted slide carries the body. **Cost: one skinned mesh + 2–3 clips + blend logic.** No physics solver. Deterministic-friendly if the impulse math is done in the sim in Q16.16 and the animation is pure render.

- **Rung 2 — Blended partial ragdoll on impact (the sweet spot).** Character keeps playing locomotion → on the event, **blend physics weight 0→1 on the upper body (and full body for a real fall)** over ~0.15–0.3 s, apply the collision impulse to the chest/hips, let the passive-ish ragdoll fall, then **detect front/back and blend into a get-up clip back to weight 0** (Unreal's exact recipe; MoCap Online's get-up flow). This is what most AAA games actually ship for knockdowns. **Cost: 12–20 rigid bodies + constraints per felled character, only while active.** Get-up transition is the craft cost, not the compute cost.

- **Rung 3 — Active (powered) ragdoll throughout.** Character is *always* a PD-driven active ragdoll following animation (PuppetMaster-style muscles + optional pins). On collision, **drop muscle strength** so the physics of the impact dominates, marine crumples naturally, then **ramp strength back up** and let the PD controller *itself* drive the get-up by tracking a stand-up pose. No discrete blend-in/out — strength is a continuous dial. Best-looking non-learned option; the fall and recovery are genuinely physical and never twice the same. **Cost: active ragdoll running continuously for every marine** (see §5) + tuning to stop the "wobbly drunk" look.

- **Rung 4 — Learned controller (DReCon / Ubisoft Ragdoll-MM).** PD-driven physical marine tracking motion-matched locomotion via a **deep-RL policy**; collisions are just perturbations the policy **recovers from automatically**, including staggering, catching itself, or falling and getting up — all emergent. This is the "physics solver is ground truth, ML keeps it looking like animation" dream. **Cost: NN inference (~2–4 ms/char, DReCon) + full physics + a training pipeline.** On-theme for an RL project but a real arc, not a weekend.

**Practical pick for Breach:** prototype **Rung 0** immediately in the current renderer (cheap, tests whether the *idea* reads well top-down). If it earns 3D marines, jump to **Rung 2** (blended partial ragdoll) — it's the industry default, the compute is bounded (only felled units simulate), and the get-up blend is well-documented. Reserve Rung 3/4 for if/when the fall becomes a signature feature.

---

## 5. Compute-cost reality check

- **Bone budget:** production ragdolls use **12–20 physics bodies**, not the full 60–100+ deform skeleton. Drop fingers and leaf bones; export deform bones only. (MoCap Online; Sergio Abreu.)
- **Concurrent budget:** MoCap Online's rule of thumb is **8–12 simultaneously *active* ragdolls per scene** as a comfortable target, with **physics LOD** (disable/simplify distant ones) and **sleeping bodies** (idle ragdolls "cost almost nothing"). Breach's "tens of units on screen" fits *if* only the currently-falling ones simulate (Rung 2) rather than all of them always (Rung 3).
- **Dead vs active vs blended, relative cost:** dead ≈ cheapest (constraints + collision only) < blended (same bodies, but needs a stiff/stable solve + PD only while active) < active-always (PD every substep for every character, higher solver iterations, stability tuning) < learned (add NN inference). 
- **CPU rigid-body vs full engine:** a knockdown ragdoll is a **small articulated rigid-body system** — 12–20 capsules, a handful of solver iterations. This is *cheap* on any modern CPU physics lib (Jolt, PhysX, Bullet, Box2D-if-2D). The expense in shipped games is almost never one ragdoll's solve — it's **stability** (needs many iterations / high stiffness → the reason Unity docs push solver-iteration and max-angular-speed tuning), **crowds** (N ragdolls at once), and **authoring** (making transitions invisible). For Breach's tens-of-units, **Rung 2 with a felled-only simulation set is comfortably affordable on CPU**; Rung 3 (everyone active always) is where you'd start counting.
- **Simpler still — 2D/planar physics:** because Breach is top-down, a knockdown could be simulated as a **cheap 2D/planar rigid-body chain** (or even a few jointed capsules in a plane) rather than full 3D — dramatically cheaper, and possibly *sufficient* for a top-down read. Worth a spike.

---

## 6. Breach-fit + determinism hand-off

The decisive question: **is the fall render-only, or does it change gameplay state?**

- **Render-only fall (recommended default).** The marine visually stumbles/tumbles/gets up, but the **authoritative sim** just sees a normal unit that was briefly knocked back (or a "stunned for N ticks" flag). Then the whole ragdoll/active-ragdoll/learned pipeline lives **entirely in the render layer**, which is **exempt from the determinism iron rule** — floats, PhysX/Jolt, PD controllers, and NN policies are all fine. Each client can even simulate the ragdoll differently; it doesn't desync the game. **This is the clean path and covers Erik's "make it look cool."**

- **Gameplay-authoritative fall (crosses into determinism territory — FLAG).** The moment the fall has *mechanical consequences* that must be identical on every machine — the unit is **prone → vulnerable/can't shoot for a synced duration**, or the **knockback distance/final position** affects who-hits-what, or a chain reaction (dominoes) — then the *decision and the numbers that matter* must be computed in the **Q16.16 integer sim**, not in a float ragdoll. **Hand-off rule for Breach:**
  1. **Sim (Q16.16, deterministic):** the trigger test (capsule overlap + relative-velocity threshold), momentum/knockback resolution, final knocked-back position, prone-state flag, and stun duration. All fixed-point, all synced. Use `fixed_point.h` (incl. `atan2_q16`/`sin_q16`/`cos_q16`) for any direction math.
  2. **Render (float, exempt):** the *visual* ragdoll/active-ragdoll/get-up that plays out between the deterministic start state and the deterministic end state. It's cosmetic interpolation over a sim-authored start/end.
  - **Never** let the float physics solver's output feed back into synced state — that's exactly the class of bug (BLAS/LAPACK non-determinism in the spawn-RNG) that bit the X-ARCH investigation. Keep the ragdoll a strictly downstream consumer.

- **Bigger monsters with real collisions:** same split. Collision *detection and resolution that affects gameplay* stays in the deterministic sim (capsules, Q16.16). A monster's *animated flesh/secondary motion* is render-only. The hand-off line is "does any machine need to agree on this number?"

**One-line canon fold candidate:** *Ragdolls and reactions are render-layer; the sim only ever sees knockback vectors, prone flags, and stun timers, all in Q16.16.*

---

## 7. If you read three things

1. **DReCon** (Bergamin, Clavet, Holden, Forbes; SIGGRAPH Asia 2019) — https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf — the canonical low-runtime-cost responsive physics controller; motion-matching + RL PD; perturbation recovery = the bump-and-fall loop, learned. Most on-theme for an RL game.
2. **MoCap Online, "Ragdoll Physics in Games: How to Blend Animation"** — https://mocaponline.com/blogs/mocap-news/ragdoll-physics-animation-guide — the clearest practical taxonomy (dead/active/blended), the "make the transition invisible" thesis, get-up flow, and the concrete cost rules of thumb (12–20 bodies, 8–12 active ragdolls, physics LOD).
3. **Unreal "Physics Driven Animation" docs** — https://dev.epicgames.com/documentation/unreal-engine/physics-driven-animation-in-unreal-engine — the exact, copyable blended-ragdoll recipe (`Set All Bodies Below Simulate Physics` + blend weight 0→1→0, partial-body reactions) that Rung 2 is built on.

*(Bonus if a fourth: Ubisoft "Ragdoll Motion Matching," GDC 2020, Simon Clavet — the production talk of DReCon's idea; "no god forces," learns to balance and recover.)*

---

## 8. Flagged / uncertain citations

- **Euphoria licensing "$100k+"** — forum/hearsay level (surfaced in a GTA-modding discussion). Directionally "expensive," but the exact figure is **not authoritative**. Do not cite as fact.
- **DReCon "2–4 ms/frame"** — from the WebFetch summary of the PDF, not a verbatim quote of the paper's benchmark table. **Verify the exact number and hardware** in the paper before quoting.
- **EA/Frostbite talk (GDC 2018)** — details are from the **GDC Vault abstract**, not the talk video/slides. Speaker attributed as Jalpesh Sachania; confirm before quoting specifics.
- **Ubisoft Ragdoll Motion Matching** — synthesized from search snippets + gameanim.com; the gameanim page fetch returned only the abstract-level text. Deeper claims (training detail, cost) would need the GDC Vault video.
- **SuperTrack** — only skimmed via search results; described from memory of the abstract. Verify before relying on specifics.
- **Learned get-up robotics papers (FRASA, HiFAR, cross-morphology 2025+)** — real arXiv results but only title/abstract level here; these are robotics, transfer to game characters is by analogy.
- **"First ragdoll = Hitman: Codename 47 (2000)"** — common industry lore, not verified in this pass.

### Source list (used)
- MoCap Online — Ragdoll Physics in Games: https://mocaponline.com/blogs/mocap-news/ragdoll-physics-animation-guide
- DReCon PDF (theorangeduck mirror): https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf
- DReCon (Ubisoft La Forge page): https://www.ubisoft.com/en-us/studio/laforge/news/VjEIwquaIyEZZSw5RZI0V/drecon-datadriven-responsive-control-of-physicsbased-characters
- Ubisoft Ragdoll Motion Matching (GDC 2020): https://www.gdcvault.com/play/1026712/Machine-Learning-Summit-Ragdoll-Motion · writeup https://www.gameanim.com/2020/03/24/ragdoll-motion-matching/
- EA/Frostbite Physics-Driven Ragdolls (GDC 2018): https://www.gdcvault.com/play/1025210/Physics-Driven-Ragdolls-and-Animation
- Euphoria / GTA IV (Game Developer): https://www.gamedeveloper.com/game-platforms/product-i-grand-theft-auto-iv-i-using-naturalmotion-s-euphoria · GTA Wiki https://gta.fandom.com/wiki/Euphoria · HandWiki https://handwiki.org/wiki/Software:Euphoria
- Unreal Physics Driven Animation: https://dev.epicgames.com/documentation/unreal-engine/physics-driven-animation-in-unreal-engine
- PuppetMaster (RootMotion): http://root-motion.com/puppetmasterdox/html/page3.html · Asset Store https://assetstore.unity.com/packages/tools/physics/puppetmaster-48977
- Active ragdolls in Unity (Sergio Abreu): https://sergioabreu-g.medium.com/how-to-make-active-ragdolls-in-unity-35347dcb952d
- EggyStudio Unity.Humanoid.ActiveRagdoll: https://github.com/EggyStudio/Unity.Humanoid.ActiveRagdoll
- Animation of Active Ragdolls (Jan Schneider): https://medium.com/@jacasch/animation-of-active-ragdolls-in-games-32ca9d98afc9
- Unity Ragdoll/Joint stability: https://docs.unity3d.com/Manual/RagdollStability.html
- SuperTrack (Ubisoft La Forge): https://www.ubisoft.com/en-us/studio/laforge/news/7fMzaMaDgnd0gqPsCaJZYb/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning
- DeepMimic (Peng et al. 2018): https://arxiv.org/abs/1804.02717 · Learning to Get Up (2022): https://arxiv.org/pdf/2205.00307
