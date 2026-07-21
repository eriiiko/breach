# Sound-from-the-sim — annotated bibliography

> Sources gathered in the **2026-07-12 in-thread prior-art scout** (~16 web searches). Grouped by
> lineage; each entry has a one-line *what* and a *→ Breach* relevance note. **✓** = surfaced &
> sanity-checked this session. **(mem)** = from memory/knowledge, not re-surfaced this session —
> **verify before citing**. Links are best-found URLs, not guaranteed canonical. A full deep-research
> pass should verify DOIs and add the ones it finds; this is the scout's starting set, not the last word.

Reading order for a cold start: the **Survey** (§1) → **Planeverb** (§6, the closest system) →
the **neural-surrogate** cluster (§2, the method) → **Sonification Handbook ch.15** (§5, the framing).

---

## 1. Physical licence — why `|P − P_prev|` is legitimately "sound"

- ✓ **Sound Synthesis, Propagation, and Rendering: A Survey** — Liu & Manocha, arXiv 2011.05538.
  <https://arxiv.org/abs/2011.05538> · The umbrella survey of the whole field; start here to place
  everything else. → Breach's one-paragraph map of who does what.
- ✓ **Computational Aeroacoustics (CAA) + Lighthill's acoustic analogy** — e.g. Colonius & Lele,
  *Computational aeroacoustics: progress on nonlinear problems of sound generation*
  <https://www.sciencedirect.com/science/article/pii/S0376042104000570> · Aerodynamic sound *is*
  oscillation of the pressure field. → The physics that licenses reading the pressure transient as
  the acoustic signal; Breach is the coarse, real-time, perceptual-parameter cousin of (offline,
  exact) CAA.

## 2. Neural surrogates of acoustic simulation — Erik's teacher–student method (PROVEN feasible)

- ✓ **Deep Learning Surrogate for the Temporal Propagation and Scattering of Acoustic Waves** —
  Alguacil et al., *AIAA Journal* 2022. <https://arc.aiaa.org/doi/10.2514/1.J061495> · PDF:
  <https://acoustique.ec-lyon.fr/publi/alguacil_aiaaj22.pdf> · Autoregressive spatiotemporal CNN
  trained on high-fidelity lattice-Boltzmann sims; extrapolates to unseen geometries. → The canonical
  "distil an expensive acoustic sim into a cheap net" — the closest existing proof that the rung-1
  method works.
- ✓ **Fast Acoustic Scattering using Convolutional Neural Networks** — arXiv 1911.01802.
  <https://arxiv.org/abs/1911.01802> · 108k training examples, ~50 ms eval, 100–1000× speedup over a
  4-min wave sim. → Same method, scattering flavour; shows the training-data scale involved.
- ✓ **Neural-operator acoustic surrogates (FNO / neural operators)** — e.g. *Modelling superposition
  in 2D linear acoustic wave problems using Fourier neural operators*, Acta Acustica 2025
  <https://acta-acustica.edpsciences.org/articles/aacus/full_html/2025/01/aacus240111/aacus240111.html>;
  Stanford CS231N 2025 "Fast Acoustic Wave Simulation with Neural Operators"
  <https://cs231n.stanford.edu/2025/papers/text_file_840592656-CS231N_Final_Paper.pdf> · 67–210× vs
  FDTD, trained on FDTD/WaveBlender ground truth. → Operator-learning baseline (also answers brief §6 Q5).
- ✓ **NAT: Neural Acoustic Transfer for Interactive Scenes in Real Time** — arXiv 2506.06190 (2025).
  <https://arxiv.org/abs/2506.06190> · Learned acoustic transfer for interactive scenes, real-time.
  → A 2024–25 system the full pass must check for cell-overlap with Breach.
- ✓ **SOAF: Scene Occlusion-aware Neural Acoustic Field** — arXiv 2407.02264 (2024).
  <https://arxiv.org/abs/2407.02264> · Explicitly models sound attenuation through walls / occlusion.
  → Directly relevant to Breach's cross-wall-muffling gap (brief §6 Q3); check for cell-overlap.
- ✓ **Symmetry-informed surrogate for real-time acoustic wave propagation** — Applied Acoustics 2023.
  <https://www.sciencedirect.com/science/article/pii/S0003682X2300484X> · 215× faster than FEM,
  >98% R². → Another feasibility data point; symmetry priors as an efficiency lever.
- ✓ **HergNet: a Fast Neural Surrogate Model for Sound Field Predictions via Superposition of Plane
  Waves** — arXiv 2510.24279 (2025). <https://arxiv.org/abs/2510.24279> · Physics-structured fast
  sound-field surrogate. → Recent architecture idea for the teacher-surrogate or the student.

## 3. Sound-from-physics-simulation — the lineage Breach deliberately DIVERGES from (synthesis, not params)

- ✓ **Doug L. James / Timothy Langlois / Changxi Zheng — physically based sound** (modal sound,
  wave-based synthesis, precomputed acoustic transfer / FFAT maps). Publications:
  <https://graphics.stanford.edu/~djames/publications/> · Stanford course "Physically Based Sound for
  Computer Animation and VE": <https://graphics.stanford.edu/courses/sound/> · → The gold-standard
  "sound from physics," but it **synthesizes waveforms** (modal resonators, rigid bodies). Breach
  chooses DSP-params-on-samples instead — cite to contrast, not to copy.
- ✓ **NeuralSound: Learning-based Modal Sound Synthesis with Acoustic Transfer** — Jin et al.,
  SIGGRAPH 2022, arXiv 2108.07425. <https://arxiv.org/abs/2108.07425> · The learned version of the
  above. → Precedent that learning + acoustic transfer works; still synthesis-side.
- (mem) **Rigid-Body Sound Synthesis with Differentiable Modal Resonators** — arXiv 2210.15306.
  <https://arxiv.org/abs/2210.15306> · Differentiable modal synthesis. → Differentiable-audio flavour
  on the synthesis side.

## 4. Learned control / neural audio — rung-1's DSP-side neighbours

- ✓ **Learning Control of Neural Sound Effects Synthesis from Physically Inspired Models** — arXiv
  2503.08806 (2025). <https://arxiv.org/abs/2503.08806> · Physical-model control of neural SFX
  synthesis, real-time. → The single closest paper to rung 1's *spirit* — but it drives *synthesis*;
  Breach drives *DSP on samples*. Read for the conditioning design.
- (mem) **DDSP: Differentiable Digital Signal Processing** — Engel et al., ICLR 2020, arXiv 2001.04643.
  <https://arxiv.org/abs/2001.04643> · Params-not-waveform: NN controls interpretable DSP modules.
  → The founding argument for Breach's whole "control DSP, don't synthesize audio" stance.
- ✓ **Neural black-box audio-effect modeling** (overview) — <https://www.emergentmind.com/topics/neural-black-box-modeling-of-audio-effect-graphs>;
  **Steerable discovery of neural audio effects** arXiv 2112.02926 <https://arxiv.org/abs/2112.02926>.
  · FiLM/hypernetwork conditioning of reverb/EQ/dynamics. → Architectures for conditioning the
  student on sim features.

## 5. Parameter-mapping sonification — the cleanest NON-GAME framing

- ✓ **The Sonification Handbook, Ch. 15 — Parameter Mapping Sonification** — Grond & Berger.
  <https://sonification.de/handbook/chapters/chapter15/> · PDF:
  <https://sonification.de/handbook/download/TheSonificationHandbook-chapter15.pdf> · "Data values
  drive the parameters of an audio signal (pitch, gain, pan, timbre…)." → The formal home for the
  whole idea: rung 0 = parameter-mapping sonification of a physics field; rung 1 = *learned* version.
  Framing Breach here (not in game audio) is what makes it read as novel.

## 6. Game-audio boundary — the crowded cell Breach is NOT competing in (cite to position, not to join)

- ✓ **Planeverb: Interactive sound propagation for dynamic scenes using 2D wave simulation** —
  Rosen, Godin, Raghuvanshi, *Computer Graphics Forum* 2020. Wiley:
  <https://onlinelibrary.wiley.com/doi/10.1111/cgf.14099> · MSR:
  <https://www.microsoft.com/en-us/research/publication/interactive-sound-propagation-for-dynamic-scenes-using-2d-wave-simulation/>
  · Live 2D wave sim, dynamic scenes, perceptual acoustic parameters, real-time on one CPU core,
  **open-source C++**. → **The closest system overall — the benchmark.** It is why "Breach as a game
  audio tool" is derivative; Breach's differences are reuse-the-gameplay-field + learned mapping.
- ✓ **Parametric Wave Field Coding for Precomputed Sound Propagation** (Project Acoustics / Triton) —
  Raghuvanshi & Snyder, SIGGRAPH 2014.
  <https://www.microsoft.com/en-us/research/publication/parametric-wave-field-coding-precomputed-sound-propagation/>
  · Precomputed-static wave sim → perceptual params (early reflections + late reverb from IR energy
  decay). → The perceptual-parameter *encoding* Breach's teacher can borrow; but it bakes on static
  geometry — Breach's live/dynamic angle is the departure.
- ✓ **Precomputed Wave Simulation for Real-Time Sound Propagation of Dynamic Sources in Complex
  Scenes** — Raghuvanshi et al., ACM ToG 2010. <https://dl.acm.org/doi/10.1145/1778765.1778805> ·
  The precompute-then-runtime pipeline. → Context for the Project Acoustics lineage.
- ✓ **Real-time sound synthesis and propagation for games** — Raghuvanshi, Lauterbach, Chandak,
  Manocha & Lin, *CACM* 50(7), 2007. DOI 10.1145/1272516.1272541
  <https://dl.acm.org/doi/10.1145/1272516.1272541> · **The ChatGPT-cited DOI — verified accurate.**
  A survey-level overview (2007). → Good Parts I–II entry point; predates the parametric-coding work.
- ✓ **Learning Acoustic Scattering Fields for Dynamic Interactive Sound Propagation** — Tang, Meng,
  Manocha et al., 2020, arXiv 2010.04865. <https://arxiv.org/abs/2010.04865> · GAMMA:
  <https://gamma.umd.edu/researchdirections/sound/asf/> · Geometric deep learning learns per-object
  acoustic scattering (spherical harmonics) vs a wave-solver ground truth, coupled to ray tracing.
  → The closest "learned + dynamic propagation" work; teacher-student against a wave solver already
  done — but for ray-traced propagation, not a coarse-gameplay-field-conditioned DSP controller.
- (mem) **Valve Steam Audio** / **Google Resonance Audio** docs — shipping real-time spatial-audio
  middleware (occlusion, reverb zones). → The engineering baseline rung 0 must match/beat (brief §6 Q2).

## 7. Room-acoustic parameter estimation & parametric reverb — for the teacher's ground-truth params + rung 0 reverb

- ✓ **Room Impulse Response Prediction with Neural Networks: From Energy Decay Curves to Perceptual
  Validation** — arXiv 2509.24834 (2025). <https://arxiv.org/abs/2509.24834> · Predict energy-decay
  curves from room dimensions / material absorption / source-receiver positions; reconstruct RIRs.
  → A candidate ground-truth parameterization for the teacher (EDC-based), and a data point that
  geometry→acoustic-params is learnable.
- ✓ **Data-driven room acoustic modeling via differentiable feedback delay networks** — J. Audio
  Speech Music Proc. 2024. <https://link.springer.com/article/10.1186/s13636-024-00371-5> ·
  Differentiable FDN with learnable delays fit to a target RIR. → The parametric-reverb engine rung 0
  needs (brief §6 Q3: reverb must be parametric, not convolution-from-a-sim-IR).
- ✓ **Differentiable Artificial Reverberation** — Lee et al., 2022.
  <https://www.researchgate.net/publication/362206894_Differentiable_Artificial_Reverberation> ·
  Backprop-through-reverb (FVN / delay-network models) with parameter-estimation networks.
  → Toolbox for learning reverb-send/decay params in rung 1.
- ✓ **Blind Room-Acoustic Parameter Estimation (T60, DRR) via DNNs** — e.g. IEEE 2020
  <https://ieeexplore.ieee.org/document/9052970/> · Estimate reverberation time / direct-to-reverberant
  ratio. → The compact perceptual targets (T60, DRR) the teacher could hand the student.
