# Physics-Based Learned Character Control — the "NVIDIA boxing/falling dolls"

**Agent 2 of 5 · ML-animation lit search for Breach · 2026-07-20**

Scope: the research lineage behind the NVIDIA-style demos Erik described ("dolls
that are boxing and running and falling etc") — physically-simulated humanoids
driven by RL policies that imitate motion capture. This is the one animation
family whose output is *real physics* (contacts, momentum, falling, getting up),
so it is the candidate for **gameplay-authoritative** monster/marine behavior,
not just a render treat. Where it touches Breach's determinism iron rule I flag a
hand-off to the determinism agent rather than resolving it here.

> **Capture, not canon.** This is an append-only research note. Numbers are
> quoted from papers/project pages; citations I am unsure of are flagged in the
> final section.

---

## 1. What the NVIDIA demos actually are

The videos Erik remembers are **physics-based character animation via
reinforcement learning**. The recipe is remarkably consistent across the whole
family:

1. A **ragdoll** — an articulated rigid-body humanoid (typically ~15 bodies,
   ~28–34 actuated degrees of freedom) — lives inside a **rigid-body physics
   simulator**. Nothing about its pose is keyframed; every frame is the result of
   joint torques applied to a physics body that then *falls, collides and
   balances* under gravity and contact forces.
2. A **neural network policy** (a small MLP) observes the character's state
   (joint angles/velocities, root orientation, maybe a goal) and outputs **target
   joint angles / torques** — i.e. it plays the character's "muscles" at
   30–60 Hz.
3. The policy is **trained with RL** to make the ragdoll's physical motion look
   like a library of **motion-capture clips** while also achieving goals (walk
   here, punch that, carry this).

So "the dolls that box and fall over" = a mocap-imitation policy (boxing style)
plus a task reward (hit the target) running on a ragdoll whose falls are *actual
simulated physics*, not an animation of a fall. When it gets knocked down and
stands back up, that recovery was **learned**, not authored. That emergent
robustness — not photoreal rendering — is the "cool" thing, and it is exactly the
property Breach would want for gameplay collisions.

Two distinct visual sources feed Erik's memory:
- **NVIDIA Research (Toronto AI Lab / PAR Lab)** graphics papers: **ASE**,
  **CALM**, **PADL**, **MaskedMimic**, **PhysDiff** — sword-and-shield warriors,
  boxers, parkour, all physically simulated. Much of this line is co-authored by
  **Xue Bin (Jason) Peng**.
- **NVIDIA Isaac Gym / Isaac Lab** robot-learning reels: thousands of humanoids/
  quadrupeds training in parallel in one scene, flailing then converging to a
  walk. Same substrate, robotics framing.

---

## 2. The research lineage (DeepMimic → AMP → ASE → CALM/PADL → PULSE/PHC → MaskedMimic)

### 2.1 DeepMimic — the foundation (SIGGRAPH 2018)
**Peng, Abbeel, Levine, van de Panne. "DeepMimic: Example-Guided Deep
Reinforcement Learning of Physics-Based Character Skills." ACM TOG (SIGGRAPH
2018).** arXiv **1804.02717**. Project + MIT-licensed code (C++/Python, Bullet +
TensorFlow).

- **What it produces:** one control policy per *clip* — the ragdoll reproduces a
  single mocap motion (walk, spin-kick, backflip, cartwheel) as robust physics,
  surviving pushes and recovering. Handles keyframes, highly-dynamic flips/spins,
  and retargeted motions.
- **Key ideas that the whole field inherited:** (1) an **imitation reward** =
  weighted pose + velocity + end-effector + center-of-mass error vs. the
  reference at the current phase; (2) **reference state initialization (RSI)** —
  start episodes at random points along the clip; (3) **early termination** on
  fall. These three tricks are why it worked where prior RL flailed.
- **Physics sim:** Bullet. **Cost:** on the order of tens of millions of samples
  per skill; ~a day on a multi-core CPU per clip (pre-GPU-sim era). This
  per-clip, hand-tuned-reward cost is precisely what the next papers attack.

### 2.2 AMP — Adversarial Motion Priors (SIGGRAPH 2021)
**Peng, Ma, Abbeel, Levine, Kanazawa. "AMP: Adversarial Motion Priors for
Stylized Physics-Based Character Control." ACM TOG (SIGGRAPH 2021).** arXiv
**2104.02180**. (Berkeley; not yet the NVIDIA line but the direct parent of it.)

- **The move:** replace DeepMimic's hand-designed per-phase imitation reward with
  a **learned discriminator** (a GAN-style "motion prior"). The policy gets a
  **task reward** (do the goal) plus a **style reward** = how well a discriminator
  believes the character's state-transitions came from the *unstructured* mocap
  dataset. No phase variable, no clip selection, no pose-matching by hand.
- **What it produces:** a character that pursues a goal (e.g. walk to target,
  strike) *in the style of* a whole motion dataset, blending clips automatically.
- **Exact specs (from the paper):** Bullet physics at **1.2 kHz** sim, policy
  queried at **30 Hz**. Policy = MLP with hidden layers **[1024, 512]** ReLU;
  discriminator/value share that architecture. Training **100–300 M samples**,
  **~30–140 h on 16 CPU cores** (still CPU-sim). Reward blend wG=0.5 task,
  wS=0.5 style; discriminator gradient penalty wgp=10.
- **Why it matters for Breach:** the "style vs. task" split is the clean knob —
  you specify *what* (walk into that marine) with a simple reward and *how it
  looks* (heavy marine gait) with a mocap set, no animation authoring.

### 2.3 ASE — Adversarial Skill Embeddings (SIGGRAPH 2022) — **NVIDIA**
**Peng, Guo, Halper, Levine, Fidler. "ASE: Large-Scale Reusable Adversarial Skill
Embeddings for Physically Simulated Characters." ACM TOG (SIGGRAPH 2022).** arXiv
**2205.01906**. NVIDIA Toronto AI Lab.

- **The move:** pre-train a **reusable low-dimensional skill latent space** from a
  large unstructured mocap set by combining adversarial imitation (AMP-style) with
  **unsupervised RL** (a skill-discovery objective). The result is a general
  motor "brain" you can reuse.
- **What it produces:** a sword-and-shield warrior with a *repertoire* — locomote,
  strike, shield-block, get up — and a latent that downstream tasks steer. New
  tasks train a small high-level policy that outputs latents into the frozen
  low-level controller, instead of training control from scratch. This is the
  "learn once, reuse forever" pivot.
- **Scale:** trained on a **massively-parallel GPU simulator** (Isaac Gym) with
  **"over a decade of simulated experience"** — i.e. GPU-days, thousands of
  parallel envs. The reuse is the payoff: downstream tasks converge fast because
  the motor skills already exist.
- **This is very close to what Erik saw.** The ASE warrior reels are canonical
  "NVIDIA physics doll" footage.

### 2.4 CALM and PADL — directability (SIGGRAPH 2023 / 2022) — **NVIDIA**
- **CALM: Conditional Adversarial Latent Models for Directable Virtual
  Characters.** Tessler, Kasten, Guo, Mannor, Chechik, Peng. SIGGRAPH 2023. NVIDIA.
  Jointly learns a control policy *and* a motion encoder, giving a **semantic**
  latent so a user can direct the character with specific motions and
  style-condition higher-level tasks. Think ASE but you can say "in *this*
  style."
- **PADL: Language-Directed Physics-Based Character Control.** Juravsky, Guo,
  Fidler, Peng. SIGGRAPH Asia 2022. arXiv **2301.13868**. Adds **natural-language**
  conditioning (text → skill embedding → physics motion). Followed by
  **SuperPADL** (arXiv **2407.10481**, SIGGRAPH 2024) which scales language-directed
  control via **progressive supervised distillation** — distilling many RL experts
  into one deployable model.

### 2.5 PHC and PULSE — universal imitation + a motion "foundation model"
- **PHC: Perpetual Humanoid Control for Real-time Simulated Avatars.** Luo,
  Cao, Kitani, Xu et al. ICCV 2023. arXiv **2305.06456**. A single controller
  imitates **~all of AMASS** (thousands→ten-thousand clips), **recovers from
  fall/fail-states without external forces**, and runs **real-time** for driving
  avatars from noisy video/language pose estimates. Reports **98.9% success on
  AMASS-train** without residual force control. Key trick: **PMCP (progressive
  multiplicative control policy)** allocates new network capacity for harder
  clips + fail-recovery without catastrophic forgetting. **This is the paper to
  cite for "gets knocked over and stands back up."**
- **PULSE: Universal Humanoid Motion Representations for Physics-Based Control.**
  Luo et al. ICLR 2024 (spotlight). arXiv **2310.04582**. Distills PHC's motor
  skills into a **32-dim latent "foundation model for control"** covering
  **99.8% of AMASS**; downstream locomotion/terrain/tracking tasks all reuse it,
  and it can be sampled generatively. This is the strongest evidence that
  **you do not have to train the motor system yourself** — reuse a released latent.

### 2.6 MaskedMimic — the current unifier (SIGGRAPH Asia 2024) — **NVIDIA**
**Tessler, Guo, Nabati, Chechik, Peng. "MaskedMimic: Unified Physics-Based
Character Control Through Masked Motion Inpainting." ACM TOG (SIGGRAPH Asia
2024).** NVIDIA (PAR Lab) + Bar-Ilan + SFU. Project + code:
`github.com/NVlabs/ProtoMotions`.

- **The move:** frame *all* control as **motion inpainting** — train one model to
  fill in full-body physical motion from **any partial spec** (a few keyframed
  joints, a text command, a joystick/pelvis target, a head target, an object to
  interact with). One controller, many input modalities, across **irregular
  terrain**.
- **What it produces:** the most general single controller to date — sparse
  keyframes → full body, "raise hands and spin" from text, path-following,
  object interaction, all as physics. Pretrained models are **publicly released**.
- **ProtoMotions** is NVIDIA's open framework ("primitive/fundamental movements")
  that packages MaskedMimic + prior work; it targets Isaac Gym / Isaac Lab (and
  has been ported to other sims). **This is the most practical single entry point
  for reuse today.**

### 2.7 Adjacent branches worth knowing
- **PhysDiff: Physics-Guided Human Motion Diffusion Model.** Yuan, Song, Iqbal,
  Vahdat, Kautz. ICCV 2023 (oral). arXiv **2212.02500**. NVIDIA. A **diffusion**
  motion generator with a **physics projection** step (a sim-based imitation
  step inside denoising) that removes floating/foot-sliding/ground-penetration
  (>78% physical-plausibility improvement). Relevant as the *diffusion* path to
  physical plausibility — heavier at inference than an MLP policy; likely not
  Breach-fit for runtime, but the "physics-corrected generation" idea is notable.
- **PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction.** Wang,
  Lin et al. 2023. arXiv **2312.04393**. Contact-graph reward for **contact-rich**
  whole-body object interaction (BallPlay basketball skills). The frontier for
  monsters *grabbing/smashing* objects with real contact. (Successors:
  **InterMimic** 2025, arXiv 2502.20390 — flag as newer, less-vetted.)

**Lineage in one line:** DeepMimic (imitate one clip, hand reward) → AMP (learned
style discriminator, whole dataset) → ASE/CALM/PADL (reusable + directable skill
latents, NVIDIA + Isaac Gym) → PHC/PULSE (universal imitation + fail-recovery +
reusable foundation latent) → MaskedMimic/ProtoMotions (one controller, any input
spec, released weights).

---

## 3. The training substrate: Isaac Gym / Isaac Lab

**Makoviychuk, Wawrzyniak, Guo, ... , State. "Isaac Gym: High Performance
GPU-Based Physics Simulation for Robot Learning." NeurIPS 2021 Datasets &
Benchmarks.** arXiv **2108.10470**. NVIDIA.

- **What it is:** a physics simulator that runs **thousands of environments in
  parallel entirely on one GPU**, keeping state in GPU tensors that flow straight
  into PyTorch — no CPU round-trip. This is the engine that made ASE/CALM/etc.
  practical, and it is the substrate the "thousands of flailing humanoids in one
  scene" reels come from.
- **Speed:** **2–3 orders of magnitude** faster than CPU sim. Reported anchors:
  a locomotion policy from **~39 M samples in ~6 minutes with 4096 envs** on one
  GPU (vs. ~30 h / 16 CPU cores in Bullet for a comparable count); **ANYmal walk
  in <30 min on a single A100**; **Shadow Hand in <2 h on an RTX 3090** vs. ~40 h
  on prior clusters.
- **Isaac Lab** (2024–2025, successor built on Isaac Sim/PhysX 5; arXiv paper
  "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot
  Learning", 2025) is the current maintained version; **ProtoMotions/MaskedMimic
  target this family.**

> **Overlap with Breach's own identity — worth Erik noticing.** Breach *is* a
> deterministic GPU physics engine used as an RL state space. Isaac Gym is a
> (non-deterministic-by-default, float) GPU physics engine used as an RL state
> space. The architectures rhyme. Two roads: (a) **train animation policies in
> Isaac Gym**, then run inference over Breach's own physics at runtime (needs a
> matching ragdoll in Breach — see §6); or (b) treat animation as **render-only**
> and never involve Breach's sim. The determinism agent should weigh (a).

---

## 4. Compute cost reality check — training vs. inference (the crux)

The single most important fact for a cheap game: **the network is tiny and cheap;
the physics simulation it drives is the cost.** Separate the two.

### (1) One-time training cost — large, but potentially skippable
- Order of magnitude: **GPU-days** on Isaac Gym for a rich reusable controller
  (ASE = "over a decade of simulated experience"; PHC scales to ~10k clips).
  Single-skill DeepMimic-style training is cheaper (tens of M samples), but the
  reusable/universal controllers are the valuable ones and they are GPU-days.
- **You almost certainly should NOT train this yourself.** Pretrained,
  release-quality artifacts already exist: **PULSE** (32-dim latent, 99.8% AMASS),
  **PHC** checkpoints, and **MaskedMimic/ProtoMotions** public weights. The
  reuse story (ASE→PULSE→ProtoMotions) is the field explicitly optimizing for
  "don't retrain the motor system." A custom marine *gait/style* would be a
  fine-tune or a small high-level policy over a frozen low-level controller, not a
  from-scratch train. **Caveat:** released humanoids use a **SMPL-like skeleton**;
  Breach's monsters/marines would need retarget or a matched rig — non-trivial but
  not research-grade.

### (2) Runtime cost — two parts, both cheap-ish
- **Policy inference = a tiny MLP forward pass.** Typical policies are **[1024,512]**
  (~0.7 M params) and shrink cleanly to **[128,32]** (~33 k params) with little
  quality loss (reported ~770k→33k). At **~30 Hz** control that is **microseconds**
  of matmul per character on CPU, negligible on GPU. **The network is free at the
  scale of tens of characters.** Robotics proves the floor: locomotion MLPs run at
  **50 Hz on ANYmal's onboard CPU**, ~**33 Hz on Cassie's onboard computer**, and
  a quadruped-microrobot MLP has run on a **5 MHz ARM Cortex-M0** (arXiv 2512.24740
  — *flagged, very recent*). If a microcontroller can do it, Breach can.
- **The real cost = the rigid-body physics step per animated character.** Each
  physics doll needs an articulated-body solve (~15 bodies / ~30 DOF) plus contact
  resolution, at a **sim rate well above the control rate** (papers use
  **600 Hz–1.2 kHz sim vs. 30 Hz control** — several physics substeps per policy
  query for contact stability). That substepping, times N characters, times
  collision pairs, is the budget.

### How many physics-driven characters can a cheap game afford?
- The papers routinely run **4096 characters in parallel on one GPU** *for
  training throughput*. A single game frame needs only the forward step for
  on-screen characters, which is far cheaper than a training batch.
- Realistic read for Breach (tens of units, 3×3-tile humanoids, top-down): **a
  handful to a few dozen fully physics-simulated ragdolls is affordable**,
  especially on GPU, *if* Breach already has an articulated-body + contact solver.
  Breach today is a 2D-ish top-down fixed-point engine — it does **not** obviously
  have a 3D articulated ragdoll solver, so the honest cost is **"add a rigid-body
  character solver," not "add an MLP."** The MLP is trivial; the physics is the
  build.
- **Cheapest viable posture:** simulate physics dolls only for the *few* units
  currently doing something physical (a bump-and-fall, a monster melee), and keep
  everyone else on cheap kinematic/sprite animation. Physics is a per-event luxury,
  not an always-on tax.

---

## 5. The bump-and-fall scenario ("sprint into each other → stagger → fall → get up")

This is the scenario physics-based control is *uniquely* good at, because the
stagger and fall are **emergent from momentum + contact**, not authored. Direct
mappings:

- **The collision + fall itself:** free. Two ragdolls with forward momentum
  colliding under any of the above controllers will stagger and topple by physics.
  No "fall animation" needed — the fall *is* the simulation. This is the core
  reason Erik's scenario is a natural fit rather than a hard authoring problem.
- **Perturbation recovery / staying up after a shove:** DeepMimic already showed
  push-recovery; **AMP/ASE** policies balance and recover under contact
  perturbation as a side effect of robust training (RSI + early termination +
  domain-randomized pushes).
- **Get-up after going down:** **PHC (arXiv 2305.06456)** is the headline — it
  learns **fail-state recovery without external forces** and runs perpetually
  (fall → get up → continue). This is the specific capability for "and then they
  get back up." **ASE** also includes a get-up skill in its repertoire.
- **Robotics corroboration (get-up is now solved enough to ship on hardware):**
  **HoST** ("Learning Humanoid Standing-up Control across Diverse Postures",
  arXiv 2502.08378), **"Learning Getting-Up Policies for Real-World Humanoid
  Robots"** (arXiv 2502.12152), and **FRASA** (fall-recovery + stand-up, arXiv
  2410.08655) — all RL get-up policies transferred to real robots. If a real
  humanoid can learn to stand up from arbitrary sprawls, a game ragdoll certainly
  can. (These are robotics, not graphics — *flagged as adjacent evidence*.)
- **Contact-rich body-on-body (monsters grabbing/shoving):** **PhysHOI** (contact
  graph) points at real object/agent contact if a monster needs to *grab* or
  *smash*, beyond just bumping.

**Bottom line:** the exact scenario Erik loves — marines sprint into each other,
stagger, topple, scramble up — is essentially the *demo* this research family
produces. It needs (a) a ragdoll in a rigid-body sim, (b) a locomotion/style
controller (reuse ASE/PULSE/MaskedMimic), and (c) PHC-style get-up. All three
exist as published, largely released components.

---

## 6. Breach-fit + determinism hand-off

**Two fundamentally different ways to use this in Breach:**

**A) Render-only physics dolls (low risk, no determinism concern).**
Run the ragdoll + policy in a **separate, float, non-synced** physics context on
the render side; the sim/gameplay stays as-is (a unit is still an authoritative
point/circle in Breach's fixed-point world). The doll is *cosmetic* — it makes
the marine visibly stagger and fall, but the fall doesn't decide anything. This
sidesteps the iron rule entirely (render layer is float-exempt) and is the cheap,
safe first step. Good enough for "it looks cool."

**B) Gameplay-authoritative monster physics (high value, determinism-critical).**
Erik's stated interest in **"larger monsters with real gameplay collisions where
a deterministic implementation matters."** Here the ragdoll's contacts/falls
*are* gameplay — a toppling monster blocks a corridor, a shove has tactical
effect — so the physics must run **inside Breach's synced state**: Q16.16
fixed-point, no floats, no libm, cross-GPU bit-identical, golden-gated.

> **HAND-OFF to the determinism agent — the hard questions for path B:**
> 1. **Articulated-body + contact solver in fixed-point.** Every published
>    controller runs on a **float** simulator (Bullet, PhysX/Isaac). A synced
>    ragdoll needs a **deterministic fixed-point rigid-body/contact solver** —
>    Breach doesn't obviously have one. This is the real engineering cost, far
>    more than the policy.
> 2. **Deterministic policy inference.** The MLP itself can be made deterministic:
>    quantize weights/activations to fixed-point integer matmul, avoid float
>    nonlinearities (or use fixed-point LUT activations like Breach's existing
>    `sin_q16`/`atan2_q16`). A [128,32] tanh/ReLU MLP in Q16.16 is plausible. But
>    it must be **bit-identical across GPUs** — the same class of problem Breach
>    already solved for its sim (cf. the X-ARCH Ada finding: the determinism break
>    was in CPU BLAS RNG, not the GPU). Policies trained in float must be
>    **quantization-aware or post-quantized then re-gated.**
> 3. **Training vs. inference split.** Training happens **offline, in float, in
>    Isaac Gym** — determinism does *not* matter for training. Only the **shipped
>    inference path** (policy + fixed-point sim step) must be deterministic. So the
>    workflow is: train float → quantize policy → run over Breach's fixed-point
>    ragdoll → golden-gate. This mirrors Breach's existing "float research,
>    fixed-point runtime" posture.

**Recommendation for the orchestrator:** treat **A (render-only)** as the cheap,
near-term "looks cool" win — reuse a pretrained ProtoMotions/PULSE controller,
float ragdoll on the render side, get bump-and-fall + get-up essentially for free.
Treat **B (authoritative monster physics)** as a *real arc* gated on the
determinism agent's verdict about a fixed-point articulated solver — high value
for the "monsters with real collisions" goal, but the cost is the deterministic
physics solver, not the animation network.

---

## 7. If you read three things
1. **DeepMimic** (Peng et al., SIGGRAPH 2018, arXiv **1804.02717**) — the
   foundational recipe (imitation reward + RSI + early termination). Understand
   this and the whole field clicks.
2. **PHC** (Luo et al., ICCV 2023, arXiv **2305.06456**) — universal imitation +
   **fall recovery / get-up**, real-time. This is Erik's bump-and-fall-and-get-up
   scenario, and it comes with released checkpoints.
3. **MaskedMimic + ProtoMotions** (Tessler et al., SIGGRAPH Asia 2024;
   `github.com/NVlabs/ProtoMotions`) — the current unified, **pretrained,
   released** controller and the most practical reuse entry point today.

(Runner-up if a fourth: **Isaac Gym**, arXiv **2108.10470**, for why any of this
is affordable to train, and for the Breach-vs-Isaac substrate parallel.)

---

## 8. Flagged citations (verify before quoting in canon)

**High confidence (title/authors/venue/arXiv cross-checked this session):**
DeepMimic (1804.02717, SIGGRAPH 2018) — *arXiv ID not re-fetched this session,
verify*; AMP (2104.02180, SIGGRAPH 2021, specs fetched from ar5iv); ASE
(2205.01906, SIGGRAPH 2022, NVIDIA); CALM (SIGGRAPH 2023, NVIDIA); PHC
(2305.06456, ICCV 2023); PULSE (2310.04582, ICLR 2024 spotlight); MaskedMimic
(SIGGRAPH Asia 2024, NVIDIA, ProtoMotions repo confirmed); PhysDiff (2212.02500,
ICCV 2023, NVIDIA); PhysHOI (2312.04393, 2023); Isaac Gym (2108.10470, NeurIPS
2021).

**Flagged — confirm details:**
- **DeepMimic arXiv 1804.02717** — ID from prior knowledge, **not re-verified via
  search this session**; confirm before citing.
- **PADL arXiv 2301.13868 / SuperPADL 2407.10481** — SuperPADL ID from search
  snippet; PADL venue (SIGGRAPH Asia 2022) from memory — **verify venue/year.**
- **Isaac Lab paper** ("A GPU-Accelerated Simulation Framework for Multi-Modal
  Robot Learning", 2025) — title paraphrased from a search snippet; **no arXiv ID
  captured.**
- **AMP author "Kanazawa" and "Ma"** — from search snippet; confirm full author
  list/ordering.
- **MaskedMimic author list** (Tessler, Guo, Nabati, Chechik, Peng) —
  **partially reconstructed; verify exact authors/ordering.**
- **Robotics get-up papers** — HoST (2502.08378), Getting-Up (2502.12152), FRASA
  (2410.08655) — IDs from search snippets, **not individually fetched**; these are
  adjacent (robotics, not graphics) corroboration, not core lineage.
- **Microcontroller MLP** (Cortex-M0, arXiv 2512.24740) and **InterMimic**
  (2502.20390) — **very recent, snippet-only, treat as illustrative.**
- **Network-size claim "770k→33k params"** and **"600 Hz sim / 30 Hz control"** —
  from an aggregated search snippet across multiple papers, **not tied to one
  source**; the AMP-specific figures (1.2 kHz sim, 30 Hz control, [1024,512]) are
  the reliable anchors (fetched from ar5iv).

*Note the several arXiv IDs above with 2502/2512/2602/2603/2604/2605/2606 prefixes
that appeared in search results are 2025–2026 papers — plausible given today is
2026-07-20, but I did not fetch them; treat as leads, not settled citations.*
