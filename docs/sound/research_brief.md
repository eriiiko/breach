# Sound from the sim — physics-conditioned game audio (+ ML): research brief

> **Authored 2026-07-07** (Erik + Claude, last Fable day) · **Revised 2026-07-12** (Opus,
> post-EOS-arc). This doc is the **input** to a deep-research pass (deep-research workflow,
> passing this brief as the refined question). **The parking gate is LIFTED: the EOS arc
> shipped (P1–P6 closed on `main`, `cuda-breached`), so the acoustic field this work consumes
> now exists in final form.** The pass may run once Erik green-lights it. A **focused in-thread
> prior-art scout was run this session** (~16 targeted searches, 2026-07-12) — findings folded into
> §3.4, §3.6, §4, §6, §9; the full deep-research pass should *extend* it, not repeat it.
>
> **⚠️ 2026-07-12 revision — what changed and why.** The original brief was built on `wave_p`,
> the standalone explicit acoustic wave field, and assumed a ~600 Hz listener sampling rate.
> **`wave_p` no longer exists.** The EOS refactor replaced the `atmosphere`+`wave_p` two-field
> model with **one derived pressure `P = C·N_total·T`** advanced by a **Kwatra semi-implicit
> solver** (multigrid Helmholtz solve). Acoustic fronts still fall out of this single field —
> Erik's verdict is that it is a *physical improvement* — but the temporal resolution and the
> extraction mechanism are different (see §3). All engine facts below are re-grounded against
> `docs/architecture/engine/04_atmosphere_and_pressure.md` (as-built EOS banner), `config.toml`
> `[physics.eos]`, and `cpp/src/eos_solver.cpp`. The §2 ChatGPT distillation is preserved as
> origin history; §3 is where the reframing lives.
>
> Origin: an Erik ↔ ChatGPT discussion (2026-07-06/07). Its good first half is distilled in §2;
> its tail degraded into repetition and untrusted derivations and is discarded. **Every citation
> and DOI inherited from it is UNVERIFIED** — verification is part of the research pass (§6).
>
> Read-first siblings (paths from repo root): `docs/architecture/engine/04_atmosphere_and_pressure.md`
> (the unified `P` field + the retired-`wave_p`→`P_prev` note),
> `docs/architecture/mechanics/03_combat_and_weapons.md` (the reserved `loudness` stat),
> `docs/eos_refactor_design.md` / `docs/eos_refactor_decisions.md` (the shipped EOS).
> Companion docs in this folder: `README.md` (start here — status, plan, next steps),
> `papers.md` (annotated bibliography from the scout).

---

## 1. The idea in one paragraph

Breach runs a real compressible-gas simulation for gameplay physics: **`P = C·N_total·T`**, a
single pressure field derived from conserved species density `N` and temperature `T`, advanced
by a Kwatra semi-implicit solver on the tile grid, with per-material wave absorption, sealed/
breach boundaries, and live dynamic geometry (doors, breaches, destruction). **Sound is
pressure fluctuation** — so the physically honest microphone signal is already computed every
tick as **`|P − P_prev|`** (the per-tick pressure transient; the engine uses it today to drive
the water-ripple splash and the over-pressure blow-up trigger). The idea: **sample a small
neighbourhood around the listener over a few ticks**, feed it to a network (or an analytic map),
and drive **DSP parameters applied to prerecorded samples** — equalizer/EQ, gain, low-pass/muffle,
reverb send, stereo spread, sustain — rather than generating audio waveforms with a neural net, which is where much of the
current literature is heading. ML enters, if at all, only as the *mapping* from sim features to
DSP parameters. The bet: parameter-space is a radically easier, more explainable, more trainable
problem than waveform-space, and the expensive part (pressure propagation through real, dynamic
geometry) is something we already compute for free.

## 2. What the ChatGPT discussion established (the distilled good half — origin history)

- A four-part survey structure worth keeping: **(I) acoustics physics** (wave eq, FDTD,
  boundary conditions, diffusion vs wave models, attenuation, diffraction, impulse responses),
  **(II) how AAA games actually do audio** (occlusion, portals, reverb, indoor/outdoor
  transitions, dynamic environments, weapon audio), **(III) ML** (PINNs, neural operators,
  CNNs on acoustic fields, learned impulse responses, neural audio effects, differentiable
  DSP), **(IV) Breach's own formulation**.
- The core architectural contrast (Part IV): `explosion → pressure sim → listener neighborhood
  → feature vector → small NN → DSP parameters → explosion sample`, **instead of**
  `explosion → NN → generate audio`. **This thesis survives the EOS change intact — and is
  strengthened by it** (§3).
- A concrete input sketch: local pressure patch (e.g. 16×16) + velocity patch + time since
  event + distance → tiny CNN/ViT/MLP → {gain, EQ, bass, stereo spread, reverb send}.
  **Superseded** — see §3.1: at the shipped control rate the right input is a short listener
  *time-series*, not a spatial patch + vision transformer.
- One citation offered — and **VERIFIED 2026-07-12**: Raghuvanshi, Lauterbach, Chandak, Manocha
  & Lin, *"Real-time sound synthesis and propagation for games,"* Communications of the ACM 50(7),
  2007, pp. 66–73, DOI `10.1145/1272516.1272541`. A real, on-topic CACM overview — ChatGPT's one
  hard citation was accurate (the hallucination was confined to the derivation tail).
- ChatGPT's novelty claim ("nobody describes exactly this architecture") triggered the
  degradation and is **downgraded** here — see §3.4.

## 3. The reframing against the shipped engine (what the research must test, not assume)

1. **Sample a neighbourhood over a few ticks — Erik's original framing, kept.** The input is a
   small **spatiotemporal** tensor: a neighbourhood patch around the listener across the last
   K ticks, reading `P`, the transient `|P − P_prev|`, and `∇P`. The **spatial patch** encodes the
   listener's local acoustic environment (shadowed by a wall, channeled in a corridor, sealed in a
   pocket → reverberance & occlusion geometry); the **short history** encodes the event envelope
   (front arriving, dome lingering, decaying). Both matter — use both. Right-size it: a *small*
   neighbourhood (≈5×5–8×8) and a *short* history (K ≈ 8–24), fed to a small CNN or even an MLP —
   not a heavy 16×16 + vision transformer, because the field is smooth and the signal is
   low-dimensional. It is an **envelope-and-routing + local-environment** description, **not** a
   fine-structure impulse response.

2. **Sampling rate is a non-issue — the pressure field carries no high-frequency content.**
   (Correcting the original brief's ~600 Hz figure: *nothing was ever sampled at 600 Hz* — that
   was a hypothetical "if you read every wave substep" number, now moot.) The unified `P` is solved
   **once per tick (~12 Hz)** by the Kwatra semi-implicit scheme (the `N_SUB_MAX = 8` substeps are
   semi-Lagrangian advection accuracy, not acoustic resolution — they don't re-solve `P`). But the
   pressure front and its effect at the listener are inherently **slow and smooth**: a blast fills a
   room within a tick and then decays over the venting timescale (~1–2 s = 12–24 ticks), so the
   envelope the idea extracts lives well below 10 Hz and per-tick sampling captures it faithfully —
   **the front looks the same at 12 Hz as at any finer rate because there is no genuine activity up
   there.** Two real consequences remain, both properties of the *sim* (not of sampling): **reverb
   must be parametric** (send + decay derived from the transient's decay rate), **never
   convolution-from-a-sim-IR**; and **cross-wall muffling must be approximated** — the engine's
   through-wall transmission (4b) is deferred, so the acoustic front does not cross walls, only
   diffracts through openings. This *strengthens* the core thesis: parameter-space DSP is not just
   the easier target but the only honest one. The sim owns **structure** (how loud, how muffled,
   from where, decaying how); the samples own **all** timbre.

3. **Rung 0 needs no ML.** Analytic features from the listener transient → hand-tuned DSP mapping
   is fully interpretable, zero-training, and probably already good. ML earns its place only when
   there is a *ground truth to match* (§6 Q4). Do not skip rung 0.

4. **Prior-art proximity — and why NOT to frame this as game audio.** The game-audio cell is
   crowded and largely closed: **Planeverb** (Rosen, Cheng, Raghuvanshi et al., *Computer Graphics
   Forum* 2020) already does *live 2D wave simulation for dynamic scenes → perceptual acoustic
   parameters → sound rendering*, real-time on one CPU core, with **open-source C++** — most of
   what a "Breach game-audio tool" would claim. **Project Acoustics / Triton** does the
   precomputed-static version with a hand-designed perceptual encoding (early reflections + late
   reverb from IR energy decay). **Learning Acoustic Scattering Fields** (Tang et al. 2020) already
   *learns* dynamic propagation (geometric deep learning, spherical-harmonic scattering). Framed as
   a game-audio propagation system, Breach is **derivative** — so **don't frame it that way**
   (Erik's steer, 2026-07-12). The differentiators that remain — reuse of the **gameplay
   compressible-flow field** as the oracle (all these systems add a *dedicated* acoustic sim), a
   **coarse live control-rate transient** rather than a baked IR, and a **learned** field→DSP-params
   map onto **prerecorded samples** (not waveform synthesis) — are audio-ML / acoustics
   contributions, not game-audio ones. Where they are actually open: the scout below.

5. **Two consumers, one mechanism.** The same listener-transient features feed the **stealth
   layer**: the armory already reserves a `loudness` stat with no consumer
   (mechanics/03 §weapon-stats). AI hearing = a threshold/detector on the same transient, computed
   from the **deterministic, CUDA-bit-identical** `P` field. Gameplay hearing is authoritative →
   lives in the deterministic core (which `P` already satisfies). Audio DSP/NN is presentation →
   float, local, unconstrained (same split as rendering; the render-sim-decoupled budget rule
   applies).

### Scout findings — the prior-art landscape (2026-07-12, in-thread sweep, ~13 targeted searches)

Re-aimed per Erik's steer *away from game-dev* toward the ML / acoustics / signal-processing
literature. The idea sits at the junction of four bodies of work; none occupies Breach's exact
cell, and the non-game framing is where it reads as novel rather than derivative:

1. **Aeroacoustics / Computational Aeroacoustics (CAA).** Computes sound *from* a CFD pressure
   field — **Lighthill's acoustic analogy**: aerodynamic sound *is* oscillation of the pressure
   field. This is the physical licence for treating `|P − P_prev|` as the acoustic signal. But CAA
   is **offline, high-order-accurate, waveform-level**; Breach is its **coarse, real-time,
   perceptual-parameter** cousin. Anchor: *Sound Synthesis, Propagation, and Rendering: A Survey*
   (arXiv 2011.05538).
2. **Physically-based sound synthesis** (James / Langlois / Zheng lineage; **NeuralSound**, SIGGRAPH
   2022, adds learning). Synthesizes **waveforms** from physics sims (modal rigid-body, wave-based,
   precomputed acoustic transfer / FFAT maps). Breach **diverges deliberately**: params-on-samples,
   not synthesis; a fluid/pressure field, not modal resonators.
3. **Neural audio synthesis with physical conditioning** (DDSP lineage; closest single hit:
   *Learning Control of Neural Sound Effects Synthesis from Physically Inspired Models*, arXiv
   2503.08806, 2025). Conditions neural **synthesis** on physical/loudness/pitch features — rung-1's
   nearest neighbour, but it still *generates* audio; Breach *controls DSP on samples* instead.
4. **Parameter-Mapping Sonification** (*The Sonification Handbook* ch.15; ML variants learn the map
   with MLPs). Formally: data values drive audio-signal parameters (gain, pitch, pan, timbre…).
   **The cleanest non-game home for the idea:** rung-0 = parameter-mapping sonification of a live
   physics field; rung-1 = *learned* parameter-mapping sonification.
5. **Neural surrogates of acoustic simulation — the method Erik proposes, and it is PROVEN.**
   Distilling an expensive high-res acoustic solver into a cheap neural net is an established
   subfield: autoregressive CNN surrogates trained on high-fidelity LBM/FDTD (Alguacil et al.,
   *AIAA Journal* 2022), Fourier-neural-operator surrogates (67–210× faster than FDTD, trained on
   FDTD ground truth), CNN acoustic scattering (100–1000× speedup, ~50 ms eval), symmetry-informed
   surrogates (215× faster, >98 % R²), and 2024–25 runtime-neural-acoustics — **NAT** (Neural
   Acoustic Transfer, arXiv 2506.06190) and **SOAF** (Scene Occlusion-aware Neural Acoustic Field,
   arXiv 2407.02264, which explicitly models through-wall attenuation). **Implication:** the
   feasibility of "train a NN to match an expensive acoustic sim" is not in question — many systems
   do it. Breach's novelty is therefore **not the method** but the **configuration**: the student
   conditions on the *cheap live gameplay compressible-flow field* (not geometry/BCs) and outputs
   *DSP params on samples* (not a predicted field/waveform), in a *dynamic-geometry* game. Existing
   surrogates take geometry/initial-conditions as input; none found reuses the game's own coarse
   physics as the feature.

**Breach's open cell, in these terms:** a *learned parameter-mapping sonification of a live,
coarse, compressible-flow simulation field onto DSP-controlled prerecorded samples* — reusing the
**already-running gameplay physics** as the acoustic oracle, and deliberately choosing
**control-of-DSP over waveform-synthesis**. Each neighbour exists; this junction appears unclaimed.
That is a defensible **audio-ML / acoustics** contribution. The full pass should pressure-test each
of the five neighbours for a paper that already fills the cell before Breach claims it.

## 4. Ambition rungs

- **Rung 0 — analytic features → hand-tuned DSP** (no ML). Listener-neighbourhood patch × short
  history + `∇P` → {gain, EQ tilt, LP-filter cutoff (muffling), reverb send/decay, pan, sustain}.
  Ships game value immediately; doubles as the stealth-layer detector.
- **Rung 1 — learned mapping via teacher–student distillation** (*Erik's design, 2026-07-12*).
  **Teacher:** a high-resolution offline acoustic simulation (a proper FDTD/wave or aeroacoustic
  solve on the same geometry) — the ground truth for how sound truly attenuates, reverberates, and
  occludes. **Student:** a small CNN/MLP that reads only the **cheap live coarse features** the game
  already computes (the `|P−P_prev|` neighbourhood + `∇P`) and outputs DSP params (EQ, gain, muffle,
  reverb send, pan) that **reproduce the teacher's perceptual result** — it learns the *correction*
  from cheap-game-acoustics to true-acoustics ("adjust the sound to match the true simulation").
  **Training data is free and unlimited:** the deterministic engine emits (geometry, event)
  scenarios at will; the teacher labels each once, offline. **Crucial design choice** — the student
  conditions on the *cheap live field, not the geometry*: that is what makes it novel *and*
  runtime-cheap (geometry-conditioned acoustic surrogates already exist — §3.6.5). This is the
  portfolio-paper-shaped rung, and the scout shows the method itself is proven feasible.
- **Rung 2 — learned spatial/directional acoustics** (neural operator / learned IR field) —
  only if rungs 0–1 hit a wall the literature says this solves. **Default: out of scope**, and
  the ~12 Hz control rate makes the IR-flavored variants especially ill-fitting.

## 5. Fixed constraints (not up for re-litigation by the research)

1. **Presentation-layer only.** Audio never writes back into synced state. Gameplay hearing
   (rung 0's detector) reads only the already-deterministic `P` / `|P−P_prev|` field.
2. **The acoustic field is the shipped unified `P`.** `P = C·N_total·T`, Kwatra semi-implicit,
   materialized once per tick as a Q16.16 field; `|P−P_prev|` is the transient; walls fully
   block the acoustic front (through-wall transmission "4b" is deferred in the engine, so
   cross-wall muffling must be approximated, not read from the sim — see §6 Q3). This replaces
   the original brief's `wave_p` assumption wholesale.
3. **Realtime on consumer GPUs alongside game + NN inference**; the audio path gets the
   leftovers, and rung 0/1 inference is trivially small by design (a K≈24 scalar series).
4. **Prerecorded samples are the timbre source.** No waveform-generating nets (rung 2 caveat
   aside). Breach competes on qualitatively-different-things-happening; audio should make the
   *physics audible*, not showcase generative audio.

## 6. Questions the research pass MUST answer

- **Q1 — The open cell, in ML/acoustics terms (make-or-break; reframed 2026-07-12, see §3.6).**
  The game-audio cell is closed (Planeverb, Project Acoustics, Learning Acoustic Scattering Fields).
  The live question is in the **audio-ML / acoustics / sonification** venues: has anyone (a) used a
  **live, coarse physics/fluid simulation** as the control signal for audio, (b) *learned* a map
  from a simulation **field** to **DSP parameters on prerecorded samples** (vs. waveform synthesis),
  or (c) reused a **non-acoustic gameplay physics** solver as the propagation oracle? Name the
  closest hit in each §3.6 lineage and state the exact empty cell. (ChatGPT DOI already verified — §2.)
- **Q2 — What AAA actually ships** (survey Part II): occlusion/portal/reverb-zone practice in
  Wwise/FMOD/Unreal, weapon audio layering, indoor/outdoor transitions — the baseline rung 0
  must beat or match. Which of these does a live pressure field replace *for free*?
- **Q3 — Feature extraction & minimum control rate.** At **~12 Hz per-tick** sampling and 1/3 m
  tiles: which perceptual parameters are honestly recoverable (envelope: gain, muffle, sustain,
  pan) and which are fiction (fine reverb, discrete echo)? What is the **minimum control rate**
  for believable game-audio parameter automation (RTPC/parameter-smoothing precedent)? How is
  **cross-wall muffling** faked when the sim front does not cross walls (4b deferred) — an
  occlusion-ray/portal hybrid? Precedent for **parametric reverb** driven by a coarse energy-decay
  estimate rather than a convolution IR.
- **Q4 — The teacher–student design (target now CHOSEN; validate it).** Rung 1's target is a
  high-res offline acoustic sim as teacher (§4). The scout confirms distilling an expensive acoustic
  solver into a cheap NN is a **proven, common** method (§3.6.5) — feasibility is de-risked. Open
  questions for the pass: (a) **does the coarse gameplay field carry enough signal** to predict the
  high-res teacher's perceptual params — different physics (compressible-flow vs linear acoustics),
  so this is *the* risk — and how to test it cheaply *first*; (b) which teacher solver and which
  ground-truth parameterization (Project-Acoustics-style early-reflections + late-reverb encoding?
  energy-decay curve? T60/DRR?); (c) which fast-moving 2024–25 systems (NAT, SOAF, neural-operator
  surrogates) already cover this cell closely enough to borrow from or cede.
- **Q5 — ML architecture menu for the mapping** (survey Part III, scoped): differentiable DSP
  (DDSP lineage), neural audio effects, learned IRs (Neural Acoustic Fields etc.), operator
  learning on fields — each judged strictly as "does this beat a hand-tuned mapping for a small
  spatiotemporal input (≈8×8 neighbourhood × K≈24 ticks) → ~6 DSP params, at toy inference cost?"
- **Q6 — The stealth detector.** Precedent for propagation-based (not line-of-sight/radius)
  AI hearing in shipped games; how a per-tick `|P−P_prev|` threshold behaves as a hearing model;
  cost and determinism traps.
- **Q7 — Latency & scheduling.** Sim tick is 83 ms; audio onset is **event-triggered** (fires
  immediately), and the sim-derived parameters *modulate the already-playing sample*, arriving at
  ~12 Hz. Is an 83 ms (1-tick) parameter latency perceptually acceptable for the modulation, and
  how much parameter-smoothing/interpolation is needed to avoid zipper noise at 12 Hz?

## 7. Deliverables (the report)

1. **Prior-art map** (Q1/Q2): a table of the ~10 closest systems/papers × {live vs precomputed,
   static vs dynamic geometry, fine-IR vs coarse-control-rate, hand-coded vs learned mapping,
   DSP-params vs waveform}, with the empty cell Breach would occupy — or the citation that fills it.
2. **Rung 0 spec sketch**: the feature vector, extraction pseudocode against our field names
   (`atmosphere`/`P` alias, `wave_p`≡`P_prev` buffer, `wind_x/wind_y`≡`−∇P`, listener pos,
   neighbourhood-patch × tick-history sampler), and the DSP-parameter mapping table.
3. **Rung 1 verdict**: the recommended training target + architecture, or "rung 1 not justified."
4. **Verified reading list** — max ~10 items, DOIs checked, one line each on why.
5. **Risk register** — top 5, headed by: **"the coarse gameplay field doesn't carry enough signal
   to predict the high-res teacher's params"** (§6 Q4 — the make-or-break risk; validate cheaply
   first). Then: "12 Hz control rate → zipper noise / need smoothing"; "cross-wall muffling absent
   from sim → occlusion-ray hybrid"; "coarse-grid diffraction sounds wrong"; "a 2024–25 surrogate
   paper already fills the cell."

## 8. Placement recommendation (for Erik's sign-off — not yet in the roadmap)

- **Research pass:** runnable now (the EOS gate is lifted). It is independent of, and junior to,
  the rest of Phase 2. **Token cost flag:** the deep-research workflow is ~3M tokens and has blown
  the 5 h window before — partition it (survey Parts I–III as one pass, the Breach-specific
  Part IV synthesis + verify tail in-thread) rather than one monolithic run.
- **Rung 0 implementation:** attaches naturally to Phase 2 game rules (the stealth layer is where
  `loudness` finally gets its consumer); the audible-audio half is a beauty track alongside the
  black-body emitter, per taste.
- **Rung 1:** target is chosen (teacher–student distillation, §4); the gate is now the §6-Q4
  validation — *does the coarse field predict the teacher's params?* — best answered by a small
  standalone experiment before committing. Candidate ML-paper / portfolio piece, synergistic with
  the self-play training infra (same "engine as data generator" muscle).
- **Prerequisite nobody mentioned:** Breach currently has **no audio playback layer at all**.
  Rung 0 needs a mundane sample-playback + mixer foundation first (sounds load, play, pan,
  filter). Budget it. (Erik has not downloaded samples yet — that is this prerequisite, not a
  research-pass blocker.)

## 9. Reading list (scout-seeded & partly verified 2026-07-12; ✓ = surfaced this session)

**Physical licence (why `|P−P_prev|` is sound):**
- ✓ *Sound Synthesis, Propagation, and Rendering: A Survey* — arXiv 2011.05538. Umbrella survey; start here.
- Computational Aeroacoustics + **Lighthill's acoustic analogy** — sound as pressure-field oscillation.

**Neural surrogates of acoustic simulation (Erik's teacher–student method — PROVEN feasibility):**
- ✓ Alguacil et al. — *Deep Learning Surrogate for the Temporal Propagation and Scattering of
  Acoustic Waves*, AIAA J. 2022 — autoregressive CNN trained on high-fidelity LBM; the canonical
  "distil an expensive acoustic sim into a cheap net."
- ✓ *Fast Acoustic Scattering using CNNs* — arXiv 1911.01802 (100–1000× speedup, ~50 ms eval).
- ✓ FNO / neural-operator acoustic surrogates (67–210× vs FDTD; WaveBlender FDTD datasets; Stanford
  CS231N 2025) — also answers Q5's operator-learning branch.
- ✓ **NAT: Neural Acoustic Transfer for Interactive Scenes in Real Time** — arXiv 2506.06190 (2025).
- ✓ **SOAF: Scene Occlusion-aware Neural Acoustic Field** — arXiv 2407.02264 (2024); models
  through-wall attenuation (→ §6 Q3 cross-wall muffling).

**Sound-from-physics-simulation (the lineage Breach diverges from — synthesis, not params):**
- ✓ James / Langlois / Zheng — modal sound, wave-based synthesis, precomputed acoustic transfer /
  FFAT maps (Stanford "Physically Based Sound" course page collects these).
- ✓ **NeuralSound: Learning-based Modal Sound Synthesis with Acoustic Transfer** — SIGGRAPH 2022 (arXiv 2108.07425).

**Learned control / neural audio (rung-1's DSP-side neighbours):**
- ✓ **Learning Control of Neural Sound Effects Synthesis from Physically Inspired Models** — arXiv
  2503.08806 (2025); closest single paper — physical control of neural SFX (synthesis side).
- Engel et al. — **DDSP** (ICLR 2020) — params-not-waveform on the ML side.
- Steinmetz et al. — neural / differentiable audio effects; differentiable artificial reverb + FDN
  parameter estimation (for parametric reverb, §6 Q3).

**Parameter-mapping sonification (the cleanest non-game framing):**
- ✓ **The Sonification Handbook, Ch. 15 — Parameter Mapping Sonification** — the formal home for rung-0/1.

**Game-audio boundary (the crowded cell Breach is NOT competing in — cite to position, not to join):**
- ✓ **Planeverb** — Rosen et al., *Computer Graphics Forum* 2020; live 2D wave sim, dynamic scenes,
  perceptual params, open-source C++. Closest system overall — benchmark against it.
- ✓ Raghuvanshi & Snyder — **Project Acoustics / Triton** (precomputed/static parametric wave
  coding); ✓ Raghuvanshi et al., CACM 2007 (DOI `10.1145/1272516.1272541`).
- ✓ **Learning Acoustic Scattering Fields for Dynamic Interactive Sound Propagation** — Tang et al.
  2020 (arXiv 2010.04865); learned dynamic propagation.
- Valve **Steam Audio** / Google **Resonance Audio** docs — shipping baselines (Q2).
