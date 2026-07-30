# Kinematic / Data-Driven Neural Motion Generation (Render-Only, Playback-Style)

**Agent 1 of 5 — ML animation literature search for Breach**
**Date:** 2026-07-20 · **Scope:** the "learn from mocap, generate poses at runtime, NO physics sim" family.

> Framing for Breach. This digest covers neural controllers that take a **pose + a control
> signal** and emit the **next pose** (or a full clip) by having learned a manifold of human
> motion from motion-capture data. There is no ragdoll, no forward dynamics, no torque — it is
> sophisticated **playback/blending**. That makes it a *render-only, non-deterministic* candidate,
> which is exactly the box Erik drew: 3D marines on top of the 2D world, running only at render
> time, never inside the headless Q16.16 sim. The question is never "can it run" (it can) but
> "is the **authoring + data burden** worth it at 32-px top-down scale, versus the classical gait
> controller already designed." Spoiler in the verdict.

---

## 1. The landscape, technique by technique

### 1.1 Phase-Functioned Neural Networks (PFNN) — the canonical one
- **Cite:** Holden, Komura & Saito, "Phase-Functioned Neural Networks for Character Control,"
  *ACM TOG* 36(4) (Proc. SIGGRAPH 2017). DOI 10.1145/3072959.3073663.
  ([Edinburgh PDF](https://www.pure.ed.ac.uk/ws/files/35467734/phasefunction.pdf),
  [SIGGRAPH history](https://history.siggraph.org/learning/phase-functioned-neural-networks-for-character-control-by-holden-komura-and-saito/))
- **What it does:** real-time locomotion controller over rough terrain. The core trick is that
  the network **weights are not fixed** — they are produced by a *phase function* Θ(p), a cyclic
  function of the gait phase p ∈ [0, 2π) (0 = left foot down, π = right foot down, etc.). One MLP
  whose weights *morph continuously through the gait cycle*. This is what kills the classic
  blend-tree problem of foot-sliding and mushy transitions: the network is a *different* network at
  every instant of the stride.
- **Input:** previous character pose (joint positions/velocities), user trajectory (past + future
  desired path, gait), and a **local terrain heightmap patch** sampled around the character
  (~342-dim input). **Output:** next-frame full-body pose + updated phase + trajectory
  (~311-dim). Autoregressive: its own output feeds the next frame.
- **Architecture:** 3-layer feed-forward MLP (two hidden layers of **512 units**, ELU). The phase
  function is a **cubic Catmull-Rom spline over 4 "control-point" networks** — i.e. you store 4
  sets of weights and blend them by phase to synthesize the actual runtime weights.
- **Training data:** ~**1.5–2 hours** of locomotion mocap, heavily hand-processed: gait/phase
  labelling and **terrain fitting** (registering flat-ground mocap onto synthetic height fields).
  This preprocessing is the real cost, not GPU time.
- **Inference cost:** small. ~0.5M weights per instantiated network. The paper reports
  **~1 ms/frame** single-character on a CPU (order-of-magnitude; see flag §5). Memory has three
  modes: (a) **precompute** the weights at ~50 phase slices → fast but ~**125 MB**; (b) store just
  the **4 control points and blend at runtime → ~10 MB** but a bit more compute; (c) a "constant"
  single-network baseline (no phase morph, lower quality). The famous headline: *"gigabytes of
  mocap compiled into a function that runs quickly in a few megabytes."*
- **Limitation for us:** needs an explicit **phase label**, which is clean for biped locomotion but
  awkward for aperiodic actions (reload, melee) — this is exactly what MANN and DeepPhase later fix.

### 1.2 Mode-Adaptive Neural Networks (MANN) — quadrupeds, no phase label
- **Cite:** Zhang, Starke, Komura & Saito, "Mode-Adaptive Neural Networks for Quadruped Motion
  Control," *ACM TOG* 37(4), Art. 145 (Proc. SIGGRAPH 2018).
  ([Edinburgh](https://www.research.ed.ac.uk/en/publications/mode-adaptive-neural-networks-for-quadruped-motion-control/),
  code [github.com/ShikamaruZhang/MANN](https://github.com/ShikamaruZhang/MANN))
- **What it does:** same job as PFNN but for dogs (walk/pace/trot/canter, sit, jump, idle), where a
  single clean global phase does **not** exist. Replaces the hand-authored phase function with a
  small **gating network** that looks at end-effector velocities and outputs blending coefficients
  over **K = 8 "expert" weight sets**; a larger **motion-prediction network** whose weights are the
  blended experts then produces the pose. This is a **mixture-of-experts** generalization of PFNN —
  the network learns its own "phase-like" mode selection end-to-end, no labels.
- **Cost/data:** comparable to PFNN (real-time, implemented in Unity + TensorFlow); the win is
  *removing the manual phase-labelling burden*, which matters a lot for messy action sets.
- **Relevance to Breach:** MANN's "learn the modes yourself" is the more practical lineage if you
  ever went neural, because Breach marines do aperiodic things (shoot, reload) that resist a single
  phase.

### 1.3 Neural State Machine (NSM) & Local Motion Phases — scene/object interaction
- **NSM cite:** Starke, Zhang, Komura & Saito, "Neural State Machine for Character-Scene
  Interactions," *ACM TOG* 38(6) (Proc. **SIGGRAPH Asia 2019**).
  ([Edinburgh](https://www.research.ed.ac.uk/en/publications/neural-state-machine-for-character-scene-interactions/),
  code in [AI4Animation](https://github.com/sebastianstarke/AI4Animation))
  Goal-driven, precise interactions: sit in *this* chair, open *this* door, pick up/carry an object,
  avoid obstacles — one model, real-time. Adds a goal/interaction encoding on top of the MoE idea.
- **Local Motion Phases cite:** Starke, Zhao, Komura & Zaman, "Local Motion Phases for Learning
  Multi-Contact Character Movements," *ACM TOG* 39(4) (Proc. SIGGRAPH 2020). Basketball dribbling,
  fakes, shooting. Key idea: **one global phase is too coarse** for asynchronous limbs (a hand
  dribbling while feet run). They attach a **separate phase per bone/contact**, unlocking rich
  multi-contact motion. This is the conceptual bridge to DeepPhase.
- **Follow-ons worth knowing:** *Neural Animation Layering for martial arts* (Starke, Zhao, Zinno &
  Komura, SIGGRAPH 2021, EA) layers additive neural motions; **DeepPhase** (Starke et al.,
  *ACM TOG*, SIGGRAPH 2022) uses a **periodic autoencoder** to *extract the phase manifold
  automatically* from raw mocap — the mature answer to PFNN's manual-phase problem. All of these are
  the same family: MoE pose predictor + some phase/goal conditioning, all real-time in Unity.

### 1.4 Motion Matching (MM) — the industry baseline, and it's *not* neural
- **Cite:** Simon Clavet (Ubisoft Montréal), "Motion Matching and the Road to Next-Gen Animation,"
  **GDC 2016**. ([GDC Vault](https://www.gdcvault.com/play/1023280/Motion-Matching-and-The-Road),
  [gameanim writeup](https://www.gameanim.com/2016/05/03/motion-matching-ubisofts-honor/)) Shipped in
  **For Honor**.
- **What it does:** no model at all — a **k-nearest-neighbour search**. Every frame, build a *query
  feature vector* (current pose: a few joint positions/velocities + desired future trajectory) and
  **search the whole mocap database** for the frame whose feature vector best matches; jump there and
  play forward; re-search continuously. With inertialized blending it's astonishingly responsive and
  natural, and it **deletes the state-machine/blend-tree authoring** that dominates game-anim work.
- **Cost:** the *engineering* baseline everyone compares against. **Memory-heavy** (tens to hundreds
  of MB of raw animation kept resident) and search cost grows with the database, but per-frame CPU is
  modest and **fully deterministic-ish** (it's just a search + lerp — no learned weights). It is
  **data-driven but not ML**; I include it because every neural method below is explicitly pitched as
  "MM but smaller/faster."

### 1.5 Learned Motion Matching (LMM) — neural *compression* of MM
- **Cite:** Holden, Kanoun, Perepichka & Popa, "Learned Motion Matching," *ACM TOG* 39(4) (Proc.
  SIGGRAPH 2020), Ubisoft La Forge.
  ([Ubisoft writeup](https://www.ubisoft.com/en-us/studio/laforge/news/6xXL85Q3bF2vEj76xmnmIu/introducing-learned-motion-matching),
  [paper PDF](https://theorangeduck.com/media/uploads/other_stuff/Learned_Motion_Matching.pdf))
- **What it does:** keeps MM's *behaviour and control feel* but replaces the giant database + search
  with **three small MLPs**:
  - **Decompressor** — feature vector → full pose (stores the pose detail neurally),
  - **Stepper** — current feature → next-frame feature (so you don't touch the DB every frame),
  - **Projector** — query → nearest matching feature (emulates the KNN search itself).
- **Concrete wins (from the Ubisoft article):** a **590 MB** complex database (47 joints, 30+
  locomotion styles) compresses to **~17 MB** of network weights (~**35×**), or **8.5 MB** with
  16-bit quantization (~**70×**); a simple example went ~**100 MB → ~10 MB**. Crucially it gives
  **constant CPU cost** *independent of dataset size* — the reason it's the current
  production-favourite direction. Runs on **CPU** in shipping engines; networks are tiny MLPs.
- **This is the single most Breach-relevant neural entry** if you ever wanted the "smooth mocap feel
  without a state machine" and were memory-constrained — but see verdict on whether top-down needs it.

### 1.6 MotionVAE (MVAE) — latent action space + RL controller
- **Cite:** Ling, Zinno, Cheng & van de Panne, "Character Controllers Using Motion VAEs," *ACM TOG*
  39(4) (Proc. SIGGRAPH 2020), EA / UBC.
  ([PDF](https://www.cs.ubc.ca/~van/papers/2020-TOG-MVAE/2020-TOG-MVAE.pdf),
  code [github.com/electronicarts/character-motion-vaes](https://github.com/electronicarts/character-motion-vaes))
- **What it does:** an **autoregressive conditional VAE** learns a compact **latent space of
  next-pose transitions**. Sampling a latent z (conditioned on current pose) → decoder → next pose;
  roll it forward to generate motion. Then — and this is the interesting part for an RL shop — they
  train a **deep-RL policy that acts in the VAE's latent space** to hit goals (reach a target, follow
  a path). So the VAE is a *learned, low-dimensional action space* and RL does the steering.
- **Cost:** small networks, real-time; the kinematic decoder is cheap. Note this is a *kinematic*
  generative model (no physics), so it can drift/foot-slide without the RL/goal loop keeping it
  honest.
- **Why it's on Erik's radar:** Breach *is* an RL project. If marines ever needed *learned* visible
  behaviour rather than scripted gait, "RL in a learned motion latent space" is the cleanest bridge
  between the two halves of the codebase. But it's a render/behaviour luxury, not a locomotion need.

### 1.7 Motion diffusion & text-to-motion — powerful, offline, situate honestly
- **MDM:** Tevet, Raab, Gordon, Shafir, Cohen-Or & Bermano, "Human Motion Diffusion Model,"
  **arXiv:2209.14916** (2022), ICLR 2023.
  ([arXiv](https://arxiv.org/abs/2209.14916), [code](https://github.com/GuyTevet/motion-diffusion-model))
  Transformer + classifier-free diffusion; predicts the *sample* (not the noise) so it can apply
  geometric/foot-contact losses. Text-to-motion and action-to-motion, SOTA on HumanML3D/KIT.
- **MotionDiffuse:** Zhang, Cai, Pan, Hong, Guo, Yang & Liu, **arXiv:2208.15001** (2022).
  ([arXiv](https://arxiv.org/abs/2208.15001)) Fine-grained, body-part-independent text control.
- **Also in this family:** PriorMDM / MDM-based composition (Tevet et al. 2023), T2M-GPT, MotionGPT,
  MLD (latent diffusion). All **text/label → clip generators**.
- **Honest situating:** these are **offline authoring tools**, not runtime controllers. Vanilla
  diffusion needs **tens to hundreds of denoising steps** → **~seconds per clip on a GPU**, and they
  generate a *whole clip*, not a *reactive next frame under live control*. Latent-diffusion and
  distilled/consistency variants are pushing this down, but as of now the right mental model is
  **"generate mocap-like clips in a content pipeline, then feed them to MM/PFNN/LMM,"** *not* "run
  per-frame for 30 marines." For Breach they are, at most, an **asset-generation** convenience, never
  a render-loop component.

---

## 2. Compute-cost reality check

**The FLOPs are trivial; the data is the cost.** Every real-time method here (PFNN, MANN, NSM, LMM,
MVAE) is a handful of small MLPs:

- **Params:** ~0.1M–1M per controller. **PFNN** ~0.5M weights; **LMM** ships in **~8.5–17 MB** total.
- **Per-character CPU:** order **~1 ms/frame** for a PFNN-class net (§5 flag on exact figure), less
  for the tiny LMM stepper/decompressor. **Tens of units → tens of ms on a single CPU core**, which
  *would* threaten a 60 fps budget if done naïvely on CPU one-at-a-time.
- **But Breach's constraints make this a non-issue:** it's **render-only and non-deterministic-OK**,
  so you **batch** all visible marines into one matmul. Tens of ~512-wide MLP evaluations is a
  sub-millisecond GPU workload even on a modest card; you already have a render GPU in hand. There is
  **no Q16.16 / determinism tax** because it never enters the sim. Memory (≤~20 MB of weights) is
  nothing next to textures.
- **The genuinely expensive part is authoring the data:**
  - You need **motion-capture** (or licensed mocap, or diffusion-generated mocap) covering every
    action, plus **cleaning, retargeting to the marine rig, phase/gait labelling** (PFNN) or DB
    curation (MM/LMM). This is *weeks of specialist animator/pipeline time*, and it's the reason
    these methods live at AAA studios. It dwarfs the compute cost and dwarfs the cost of the
    classical procedural controller Breach already designed.
  - Diffusion methods reduce *authoring* effort (text → motion) but add **seconds-scale offline GPU
    generation** and still need cleanup/retarget before they're game-ready.

**Bottom line on cost:** running a PFNN/LMM/MVAE for tens of marines at render time is **cheap on the
GPU and feasible even on CPU if batched** — Erik's "is it super expensive?" fear is unfounded *on the
inference axis*. The expense is entirely in **acquiring and grooming mocap data and a rigged 3D
marine**, i.e. content pipeline, not runtime.

---

## 3. Breach-fit verdict (opinionated)

**Recommendation: do not build a neural kinematic controller for the marines now. Ship the
classical analytic gait controller already designed; keep neural in the back pocket for a specific
future case.** Reasoning:

1. **The camera hides the exact thing these methods buy you.** PFNN/MANN/LMM exist to make
   **foot placement, weight shift, stride phase, and contact** look flawless. At a **true 90°
   top-down** view with a **~3×3-tile (~144-world-px) humanoid footprint**, the legs are **directly
   under the torso and mostly occluded**. The viewer perceives **facing, translation, and torso/arm
   silhouette**, not gait subtlety. You are paying a mocap pipeline for detail the projection throws
   away. This is the crux.

2. **`unit.facing` is already a continuous float in radians.** The single most valuable visual
   upgrade — smoothly oriented 3D bodies that lean into turns and translate believably — is reachable
   with a **hand-tuned procedural gait + a couple of tween/IK tricks** (the sibling doc's FK / 2-bone
   IK / gait-controller design), driven directly by facing and velocity. No data needed. No
   determinism concern (render-only). Deterministic authoring, instant iteration.

3. **The data/authoring burden is wildly out of proportion to a deliberately cheap game.** A neural
   controller means: rig a marine, capture/license/generate mocap for walk/run/strafe/idle/shoot/
   reload/melee/hit/die, clean and retarget it, and (for PFNN) phase-label it — before the first
   frame renders. The classical controller needs one rigged model and tuning. For "start with just
   the marines to learn a little," neural is the wrong first rung by a wide margin.

4. **Non-determinism is *fine* here but buys nothing.** Erik correctly notes the render layer is
   exempt from Q16.16. That removes an *objection* to neural methods, but it doesn't create a *reason*
   for them.

**Where a neural kinematic controller would genuinely beat hand-tuned gait — the honest exceptions:**
- **If the camera ever tilts** off true top-down (even a 45–60° tactical angle), legs and gait become
  visible and the procedural controller's seams start to show; PFNN/LMM's payoff reappears.
- **Combat richness with many smoothly-blended actions.** If marines need a *large vocabulary* of
  mocap-quality actions (contextual reloads, cover-lean, melee, staggers) with seamless transitions,
  **Motion Matching / Learned Motion Matching** deletes an enormous state-machine authoring effort —
  *but only once you already have the mocap*, which is the expensive precondition.
- **RL-authored visible behaviour.** Because Breach is an RL project, **MotionVAE** (learned latent
  action space + RL policy) is the one entry that could unify "trained agents" with "trained-looking
  motion." That's a research-y stretch goal, not a render upgrade.
- **Upper-body / weapon interaction** (aim, recoil, carry) is *visible* top-down (arms/weapon extend
  beyond the torso footprint). A small neural layer here is more defensible than a neural *leg*
  controller — though procedural aim-IK likely still wins on cost.

**Net:** at 32-px, 90°-overhead, cheap-game scale, hand-tuned procedural gait dominates on
cost-per-visible-quality. Revisit neural (LMM first, MVAE if RL-curious) only if (a) the camera
tilts, or (b) you commit to a mocap pipeline for a rich combat action set.

---

## 4. If you read three things
1. **Holden, Komura & Saito 2017 — PFNN** (the seminal idea; understand the phase function and why it
   fixes foot-sliding): the [Edinburgh PDF](https://www.pure.ed.ac.uk/ws/files/35467734/phasefunction.pdf).
2. **Holden et al. 2020 — Learned Motion Matching** + the
   [Ubisoft explainer](https://www.ubisoft.com/en-us/studio/laforge/news/6xXL85Q3bF2vEj76xmnmIu/introducing-learned-motion-matching)
   (the production-realistic version; concrete 590 MB → 17 MB numbers, constant CPU cost, ships on
   CPU). This is what "neural animation in a real game today" actually means.
3. **Clavet, GDC 2016 — Motion Matching** ([GDC Vault](https://www.gdcvault.com/play/1023280/Motion-Matching-and-The-Road)):
   the non-neural baseline every method here is measured against; the fastest way to understand what
   problem the neural nets are actually solving.

*(Bonus for the RL angle: Ling et al. 2020, MotionVAE — latent motion space steered by RL.)*

---

## 5. Flagged / uncertain citations
- **PFNN exact runtime & memory figures** — the **~1 ms/frame** and the **~10 MB (online 4-control-
  point) vs ~125 MB (precomputed ~50 slices)** split are from memory of the paper/its talk, not
  re-verified line-by-line (the PDF wouldn't parse in-session). Directionally correct and widely
  cited; **verify exact numbers against the paper's §Results before quoting in a spec.** The 3-layer
  / 512-unit / ELU architecture and ~0.5M-weight scale I'm confident on.
- **PFNN training-data duration (~1.5–2 h)** — right order of magnitude; confirm the exact hours in
  the paper.
- **MANN K = 8 experts** — I'm fairly confident it's 8 expert weight sets, but confirm against the
  paper/[MANN repo](https://github.com/ShikamaruZhang/MANN) if the exact K matters.
- **LMM 590 MB → 17 MB (35×) / 8.5 MB (70×) / simple ~100 MB → ~10 MB** — these come from the Ubisoft
  La Forge article (fetched this session), read partly off a chart for the "simple" case; treat the
  ~100 MB as approximate, the 590→17→8.5 as article-stated.
- **MDM / MotionDiffuse arXiv IDs** — **2209.14916** (MDM) and **2208.15001** (MotionDiffuse)
  confirmed via arXiv listings this session; author lists confirmed. Venue for MDM (ICLR 2023) is
  from memory — confirm if citing formally.
- **NSM = SIGGRAPH Asia 2019 (TOG 38(6)); Local Motion Phases = SIGGRAPH 2020 (TOG 39(4))** —
  confirmed via Edinburgh/SIGGRAPH-history pages. **DeepPhase (SIGGRAPH 2022)** and **Neural Animation
  Layering (SIGGRAPH 2021)** venues from memory — plausible, verify exact year/volume if citing.
- **Diffusion "seconds per clip / tens-hundreds of steps"** — general characterization of DDPM-style
  sampling, not a measured Breach-hardware benchmark; correct as an order-of-magnitude framing.
