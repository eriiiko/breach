# Octopus arm coordination — literature strand (d): CPGs, decentralised and learned coordination

**Date:** 2026-10-29 · **Issue:** #105 (octopus animation study), strand (d) of four · **Author:** Claude (Opus), for Erik

**Question.** How should the eight arms be coordinated in time — central pattern
generators, coupled oscillators, event-driven recruitment, or learned — so that
locomotion is rhythmic enough to travel and loose enough to look alive, degrades
gracefully with dead arms (7, 6), and stays deterministic in integer arithmetic?

**Context assumed.** Each arm = sim-side XPBD chain, 18 nodes. Primitive layer:
reach (travelling stiffening bend), crawl (plant mid-arm sucker foot, push by
proximal elongation; 1–3 arms pushing at once; no fixed gait in lab, loose
underside->top recruitment in the wild), jet (mantle contraction, arms bundled),
umbrella swim (slow open, fast close ~2.5:1). Per-arm brain later: Tsetlin machine
or small NN.

**Legend.** FULL = full text read; ABSTRACT = abstract/summary only (Erik fetches);
PARTIAL = parts of the full text (e.g. figures/sections via secondary pages).

## Intended reading list

1. CPG foundations
   - Ijspeert 2008, Neural Networks, CPG review
   - Ijspeert et al 2007, Science, salamander robot
   - Matsuoka 1985/1987 oscillators
   - Righetti & Ijspeert 2008, ICRA, pattern generators with sensory feedback (quadruped)
   - Hopf CPG with feedback (Righetti 2006 adaptive frequency; Buchli)
   - Owaki et al 2013 (J R Soc Interface / Bioinsp Biomim) and Owaki & Ishiguro 2017 Sci Rep — Tegotae
   - Kano et al 2017 (R Soc Open Sci) brittle star decentralised control; Kano et al 2019 (Front Neurorobot) adaptive to arm loss
   - Kano / Ishiguro centipede, snake decentralised control
2. Octopus coordination
   - Levy, Flash & Hochner 2015 Curr Biol "Arm coordination in octopus crawling..."
   - Levy & Hochner 2017 Front Physiol "Embodied organization..."
   - Katija et al 2025 Nature (wild arm use)
   - Calisti et al 2011 / 2015 Bioinsp Biomim
   - Sfakiotakis / Kazakidi FORTH eight-arm sculling (2013–2015)
3. Event-driven / rule-based
   - Cruse 1990/1998 WALKNET, Dürr Schmitz Cruse 2004 review, Schilling et al 2013
   - Hexapod free gait / follow-the-leader
   - Game procedural stepping (Rain World, Overgrowth)
4. Learned coordination
   - Huang et al 2020 ICML "One policy to control them all" (SMP)
   - Wang et al 2018 NerveNet
   - Naughton 2021 PyElastica RL, multi-arm results
5. Integer feasibility — folded into each source and the synthesis.

---

## Digest (written as read)

### A. Octopus-specific coordination

#### A1. Levy, Flash & Hochner 2015 — "Arm coordination in octopus crawling involves unique motor control strategies"
Curr Biol 25(9):1195–1200. DOI 10.1016/j.cub.2015.02.064 · PMID 25891406.
**ABSTRACT only** (Cell/ScienceDirect returned 403; Curr Biol open archive may still work through Erik's account). Strand (a) likely holds more.
- First kinematic study of whole-animal arm coordination in crawling (lab, glass/flat floor).
- (1) crawls in ANY direction relative to body orientation; (2) body orientation and crawl direction are controlled monotonically and INDEPENDENTLY; (3) NO apparent rhythmic pattern in limb coordination.
- Thrust = pushing-by-elongation of a proximal arm segment; each arm has one fixed pushing direction set by its position around the body (radial symmetry).
- Central controller "chooses in a moment-to-moment fashion which arms to recruit for pushing the body in an instantaneous direction". The model is a vector sum of equal-force pushes from the active arms.
- **For us:** this is the canonical source for *event-driven, direction-weighted recruitment* — the selection rule is "recruit arms whose fixed push direction has a positive projection on the desired velocity". No phase variable anywhere.

#### A2. Levy & Hochner 2017 — "Embodied organization of Octopus vulgaris morphology, vision, and locomotion"
Front Physiol 8:164. DOI 10.3389/fphys.2017.00164 · PMC5368235. **FULL** (Perspective article).
- Restates the 2015 model in more detail: recruitment is **probabilistic** — "the probability of recruiting each arm changes moment to moment, favoring arms likely to move the body in the desired direction". "There is no apparent order in the octopus arm stepping records."
- FFT of instantaneous crawling velocity shows **no characteristic frequency** (contrast Drosophila / stick insect: one prominent peak). A 0.3 Hz peak in their Fig 1 is an artefact of the ~6 s analysis window, not a rhythm.
- "The active arms at each moment in time apply virtually equal pushing forces"; body direction = vector sum of the active arms' pushing directions. The animal therefore steers by **which subset** is active, not by modulating force per arm.
- Walking (bipedal / stilt-like) needs ≥2 arms in contact for stability; the authors note that lack of CPG is "functionally more significant" for walking.
- Head kept horizontal during locomotion (statocyst); stiffness of interacting arms (~1 DOF per arm) suffices to set head orientation through contact feedback.
- Reach = 3–4 controlled DOFs (2 base direction, 1 stiffening-wave, ±1 elongation); fetch = pseudo-elbow where two waves collide. Peripheral program survives amputation.
- No numbers for step durations, speeds or arm counts. No quantitative coordination model beyond "probabilistic + vector sum".
- **For us:** gives the *architecture* (central picks a direction + subset; arms run their own primitive; no oscillator), plus the art requirement that body facing and travel direction are two independent knobs.

#### A3. Katija et al 2025 — "In situ light-field imaging of octopus locomotion reveals simplified control"
Nature (6 Aug 2025). DOI 10.1038/s41586-025-09379-z · PMC12545212 · correction DOI 10.1038/s41586-025-09512-y. **FULL** (CC BY 4.0; numbers in figure panels not in text).
- *Muusoctopus robustus*, 3,200 m, Davidson Seamount, ~1.6 °C, slow. 4 individuals' strides counted: n = 25, 73, 62, 67 strides; O15 followed ~12 min crawling forward over rough rock.
- **Arm pair 1 (dorsal) used mostly to probe the upcoming substrate**, pairs 2–4 for weight bearing. Arm numbering: pair 1 dorsal → pair 4 ventral.
- **Ventral arms: shorter cycle and larger contact duty cycle than dorsal arms**; minimal L–R differences for pairs 3–4, large variability for pairs 1–2.
- O15: arms on both sides used for support in 73% of frames; **3 ± 1 arms in contact at any moment**; this number itself **cycled between 2 and 6 over 2–3 min periods**.
- Order of contacts: "higher occurrence of cyclical contact and weight bearing **from ventral to dorsal arms**, with little to no occurrence of arms 1 after a consistent arm pair". The authors speculate this "might hint at a previously undescribed central pattern generator" — i.e. a loose rhythm emerges in the wild where Levy 2015's lab tanks allowed only a few strides.
- Stride mechanics: medial "foot" placed; high bend concentrated at the foot; strain concentrated in the third of the arm just proximal to the foot; foot holds until proximal portion reaches **maximal strain**, then suckers lose purchase and the foot is dragged off. Distal to the foot: little change. Typically 3 arms at a time → a support polygon + possibly a spring-loaded inverted pendulum.
- **For us:** this is the event that ends a stance: *release when proximal strain ≥ max*. It is a load/strain threshold — exactly a Tegotae / WALKNET-style local rule, not a timer. And it gives the target statistics: mean 3 contacts (range 2–6), ventral arms higher duty, dorsal pair mostly exploring.

#### A4. Sfakiotakis, Kazakidi & Tsakiris 2015 — "Octopus-inspired multi-arm robotic swimming"
Bioinspir Biomim 10(3):035005. DOI 10.1088/1748-3190/10/3/035005. **FULL** (accepted manuscript via a docksci mirror; Strathclyde AAM https://strathprints.strath.ac.uk/60917/ too large for my fetcher). Precursors (abstract only — FORTH host has a broken TLS chain): Sfakiotakis et al ICRA 2013 "Octopus-inspired eight-arm robotic swimming by sculling movements" (~0.2 BL/s, rigid arms); MED 2013 "Turning maneuvers of an octopus-inspired multi-arm robotic swimmer"; IROS 2014 "Multi-arm robotic swimming with octopus-inspired compliant web" (web → 0.5 BL/s, 10.5 N, CoT 0.62).
- Sculling = arm-base rotation, fast power stroke + slow recovery. Recovery speed ω, power speed βω. Real octopus recovery:power duration ratio **2.5 ± 0.5** (their reference to octopus arm swimming — matches our umbrella 2.5:1). Robot used **β = 5**, ω = 50°/s, amplitude A = 15–25°, offset ψ = 25–40°; period Ts = 2.5(β+1)A/(βω) ≈ 0.9 s at A = 15°. No forward motion at β = 1 — **the stroke asymmetry IS the thrust** (drag-based, quadratic in speed).
- Eight-arm gaits as phase fractions of Ts (arms L1–L4, R1–R4):
  - **G1** all eight synchronous (umbrella-like) — single large force peak per period, most discontinuous velocity, compliant arms twitch on recovery.
  - **G2** four diagonally-opposite pairs (L1/R4, L4/R1, L3/R2, L2/R3) staggered by **Ts/4** — **smoothest velocity** (ripple at 4× stride frequency), best matched by the model.
  - **G3** four adjacent pairs staggered Ts/4 — "slow, wobbling" (asymmetric about the axis → yaw/heave wobble) — dropped.
  - **G4** two alternating tetrads (L1,L3,R2,R4 vs L2,L4,R1,R3) at **Ts/2** — intermediate.
  - **Average speed ≈ the same for G1, G2, G4** (~70–100 mm/s; max 98.6 mm/s = 0.26 BL/s, 380 mm body); max axial force 3.5 N; CoT 1.5–2.0 across gaits. So phase pattern buys **smoothness**, not speed.
- **Arm count:** CoT 0.98 (2 arms) / 1.07 (4) / 1.63 (8) — fewer arms more efficient (quadratic drag on recovery). Two arms (L1, R4) grasped an object while six swam G1 — **swimming degrades gracefully when arms are diverted**.
- Turning: GR1 three adjacent arms on one side scull (spin in place, ~70° over 6 strides); GR2 three opposite lateral pairs scull sideways (fastest yaw, most variance); GR3 same with half amplitude on one set (turn + translate). Idle arms held at ψ+A. Backward = swap power/recovery.
- **For us:** for umbrella swim (all arms bundled into one stroke) G1 synchrony is right and biologically observed; but if we want a "cruise" variant, G2 (diagonal pairs at quarter-phase) is the known smooth one and G3 (adjacent pairs) is the known bad one. Asymmetric stroke β is the only parameter that matters for thrust. All of this is a phase accumulator + lookup — trivially integer.

#### A5. Calisti et al 2011 / Calisti, Arienti et al 2012 / Calisti, Corucci, Arienti & Laschi 2015 (SSSA Pisa, OCTOPUS IP)
- Calisti, Giorelli, Levy, Mazzolai, Hochner, Laschi & Dario 2011, "An octopus-bioinspired solution to movement and manipulation for soft robots", Bioinspir Biomim 6(3):036002, DOI 10.1088/1748-3182/6/3/036002. **ABSTRACT only** (scispace mirror empty).
- Calisti, Arienti, Giannaccini et al 2012 ICRA, "Design and development of a soft robot with crawling and grasping capabilities" — six soft limbs ("PoseiDRONE" lineage). **ABSTRACT only.**
- Calisti, Corucci, Arienti & Laschi 2015, "Dynamics of underwater legged locomotion: modeling and experiments on an octopus-inspired robot", Bioinspir Biomim 10(4):046012, DOI 10.1088/1748-3190/10/4/046012. **ABSTRACT only** (Lincoln repository 403).
- What they establish (abstract level): the 2011 paper builds a single cable-driven silicone arm that reproduces the Levy/Hochner crawl hypothesis (attach → proximal shortening → elongation push → detach) with the SAME limb used for grasping. The 2012 six-limb robot splits roles: "some limbs provide stability while others push and pull the robot in the desired direction" — a static role assignment, not a gait. The 2015 paper models underwater legged locomotion as a buoyancy-reduced SLIP ("U-SLIP"); forward and backward at several speeds. Their later bipedal-walk work (Living Machines 2014) uses two arms only.
- Coordination content is thin: these robots run open-loop fixed sequences (one or two pushers at a time). They are useful as a sanity check that **one or two simultaneous pushers + passive stabilisers** suffice to travel under water, consistent with Katija's 3 ± 1 contacts. Nothing here on graceful degradation or phase relations — not worth Erik's fetch budget unless strand (b)/(c) asked for the SLIP model.

### B. CPG foundations

#### B1. Ijspeert 2008 — "Central pattern generators for locomotion control in animals and robots: a review"
Neural Networks 21(4):642–653. DOI 10.1016/j.neunet.2008.03.014 · PMID 18555958. **ABSTRACT only this session** (EPFL preprint link redirects to the lab homepage; bohrium/core blocked). Points below are from my prior reading of the review, NOT re-verified today — Erik should fetch it if we lean on it.
- CPG = neural circuit producing a coordinated rhythmic output from a simple, non-rhythmic drive; robust because it is a **stable limit cycle** (perturbations decay back to the rhythm), **distributed**, and steered by **few parameters** (a tonic drive sets frequency/amplitude; left/right drive asymmetry turns; phase biases set the gait).
- Model families, in order of abstraction: biophysical (Hodgkin–Huxley), connectionist (half-centre, Matsuoka), **oscillator models** (Kuramoto-style phase oscillators, amplitude-controlled phase oscillators, Hopf). Robotics mostly uses the last family because phase lags and amplitudes are explicit parameters.
- Robotics claims for CPGs: smooth modulation of gait under abrupt parameter change; sensory feedback is easy to add as a coupling term (entrainment); fewer parameters for learning/optimisation; gait transitions through drive.
- Admitted weaknesses: no solid design methodology (parameters hand-tuned or optimised); weak theory for feedback integration; CPGs suit **rhythmic** movement and are a poor fit for **discrete, aperiodic, goal-directed** movement — exactly where the octopus's crawl sits per Levy 2015.
- **For us:** a central oscillator is the right tool only where the animal is genuinely periodic (umbrella swim, jet pulses). For crawl it would contradict the biology and impose a gait the lab data say does not exist.

#### B2. Ijspeert, Crespi, Ryczko & Cabelguen 2007 — "From swimming to walking with a salamander robot driven by a spinal cord model"
Science 315(5817):1416–1420. DOI 10.1126/science.1138353. **FULL** (main text; parameters in the SOM, not read).
- The canonical robot CPG. Each oscillatory centre = **amplitude-controlled phase oscillator**:
  `dθ_i/dt = 2π ν_i + Σ_j r_j w_ij sin(θ_j − θ_i − φ_ij)`
  `d²r_i/dt² = a_i ( (a_i/4)(R_i − r_i) − dr_i/dt )`   (critically damped approach to target amplitude)
  `x_i = r_i (1 + cos θ_i)`   (positive burst signal → joint set-point)
- Intrinsic frequency ν_i and amplitude R_i are **piecewise-linear saturating functions of one drive d**, active between `d_low` and `d_high`; outside that band the centre is silent. Limb centres have lower ν for the same d and a lower `d_high` → at high drive limbs fall silent and the body CPG's travelling wave is released: **the gait transition emerges from saturation**, not from a switch.
- Coupling: body chain gives a travelling wave (one body-length wavelength kept constant across frequency); limb→body couplings are **stronger** than body→body and unidirectional, so limbs impose an S-shaped standing wave in walking.
- Numbers (robot): walking 0.2–0.6 Hz, swimming 0.9–1.3 Hz (animal: 0.6–1.2 vs 1.6–2.9 Hz, with a gap). Walking 0.03–0.09 m/s (0.04–0.11 BL/s); swimming 0.07–0.12 m/s (0.08–0.14 BL/s). Turning = left/right drive asymmetry. Copes with abrupt drive steps.
- **For us:** two useful ideas even if we never run a body CPG: (1) **one scalar drive → gait selection through per-mode thresholds** (low drive = crawl, high = swim/jet) is an elegant, art-directable single knob; (2) the phase-oscillator equation is the template for a swim/jet rhythm. Integer cost: the phase equation needs `sin` of a phase difference — a 1024-entry Q16 LUT on a Q32 turn accumulator is exact and deterministic; the second-order amplitude ODE is a two-state integer filter (or replace by a first-order exponential approach, `r += (R − r) >> k`).

#### B3. Righetti & Ijspeert 2008 — "Pattern generators with sensory feedback for the control of quadruped locomotion"
ICRA 2008, pp. 819–824. DOI 10.1109/ROBOT.2008.4543306. **ABSTRACT only** (Infoscience PDF link serves 0 bytes to my fetcher; Erik can download https://infoscience.epfl.ch/record/130740). Equations below from my prior reading — verify.
- Hopf-type oscillator per leg with **separately settable stance and swing durations**: `ẋ = α(μ − r²)x − ωy`, `ẏ = β(μ − r²)y + ωx + Σ_j k_ij y_j + u`, with `ω = ω_stance/(e^{−by}+1) + ω_swing/(e^{by}+1)` (the sign of y selects which frequency is active → duty factor is a parameter, not a by-product).
- Gait = choice of the coupling matrix k_ij (walk / trot / pace / bound are four sign patterns).
- Touch-sensor feedback `u` **stops the oscillator** at the stance→swing transition while the foot is still loaded and **forces a fast transition** if contact is lost early — i.e. the oscillator becomes a phase-resetting state machine at the two event points. Same controller ran on three simulated quadrupeds (iCub crawling, Aibo, Ghostdog); "strong mutual coupling between the CPG and the robot allows more robust locomotion even under non-precise parameters and non-flat environment".
- **For us:** the takeaway is the hybrid: a continuous oscillator whose two event points (lift-off, touch-down) are gated by contact. The octopus crawl's stance end ("release at max proximal strain", Katija) is exactly such a gate. The Hopf ODE itself is unnecessary for us — the gating idea survives on a plain phase accumulator.

#### B4. Kano, Sato, Ono, Aonuma, Matsuzaka & Ishiguro 2017 — "A brittle star-like robot capable of immediately adapting to unexpected physical damage"
R Soc Open Sci 4:171200. DOI 10.1098/rsos.171200 · PMC5750017. **FULL** (HTML; a few equation renderings garbled, noted).
- **The most directly relevant paper for "eight arms, some dead".** Brittle star: 5 radially symmetric arms, no brain, nerve ring.
- Animal amputation series (A = 5 arms … G = 1 arm): one or two adjacent arms removed → same rowing gait, missing slot becomes a "hindlimb", **no loss of speed**; both neighbours of an arm removed → switches to **reverse rowing** (two remaining neighbours become forelimbs); one arm left → swings it side to side, still **> 50% of intact speed**. Coordination pattern is "not predetermined but changes with body configuration".
- Model (no central controller, no oscillator, only adjacent-arm interaction; equations "do not change under any physical damage"):
  - arm direction vectors `r_j = [−sin(2π(j−1)/5), cos(2π(j−1)/5)]`; **orientation signal `a_j = r_j · d`** (d = desired travel direction, given)
  - target-angle dynamics `dΦ_i/dt = c (Ξ_i + P_i + R_i − Φ_i)` (first-order lag toward the sum of noise, power and recovery terms)
  - **noise** Ξ_i = uniform random in [−1, 1] × α, **redrawn every 0.25 s** — exploration that seeds the pattern
  - **power stroke** `P_i = −β (U_R,i − U_L,i)`, `U_R,i = max(a_i − a_{i−1}, 0) · F_R,i`, `U_L,i = max(a_i − a_{i+1}, 0) · F_L,i` — an arm pushes on a side only if (a) it is MORE forward-pointing than that neighbour and (b) it feels a ground reaction on that side
  - **recovery stroke** `R_i = γ (S_R,i − S_L,i)`, S = sigmoid of yaw angle beyond a threshold θ_th = 0.94 rad — swing back only near the end of range
  - robot estimates F from target-vs-actual angle error: `F_R = max(−tanh(κ_f(φ − θ)), 0)` — i.e. **load = tracking error** (directly available in an XPBD sim as constraint lambda or position error)
  - params: c = 1.33 s⁻¹, α_yaw 0.175, β_yaw 5.24, γ_yaw 2.09, κ_f 2.86, κ_s 1.72.
- Robot (disc 0.18 m, arms 0.18 m, 1.06 kg): moved brittle-star-like **immediately after power-on in all configurations A–G**; adapted to destroyed arm tips "within a few seconds"; velocity unchanged except in F, G. With arms lifted off the ground the arms moved randomly and did **not** coordinate — **coordination is carried by the physics** (ground contact), not by neural coupling.
- **For us:** this is the template for graceful degradation: a dead arm simply produces no reaction force and no `a_j` comparisons, and the survivors' rule is unchanged. The `a_j = r_j · d` projection is Levy's "recruit arms whose push direction projects on the desired direction" made explicit. Integer cost: dot products, `max`, a threshold, a seeded PRNG, a first-order lag (`x += (target − x) >> k`). **Fully integer-friendly.**

#### B5. Kano, Kanauchi, Ono, Aonuma & Ishiguro 2019 — "Flexible coordination of flexible limbs: decentralized control scheme for inter- and intra-limb coordination in brittle stars' locomotion"
Front Neurorobot 13:104. DOI 10.3389/fnbot.2019.00104 · PMC6923253. **FULL.**
- Extends B4 to many-segment flexible arms (closest analogue of our 18-node chains). Central command = travel direction **D** only; per arm **`T_i = D · F_i`** with F_i the internal force at the arm's base: **T_i > 0 → the arm's force helps propulsion → segments "exploit" the contact; T_i < 0 → segments "avoid" it**.
- Segment rule: target yaw `τ dθ̄_y/dt = −θ̄_y + ξ_y + σ_y tanh(γ |T_i| S^y_{i,j})`, where `S^y` = Σ (right − left contact force) over the n_d = 2–3 distal segments minus the same over the n_p = 3–4 proximal ones. Local per-segment side forces `f^R = max(f·n, 0)`, `f^L = max(−f·n, 0)`.
- Works with 5 shortened arms ("breaststroke" gait emerges), 2 arms, 1 arm (sim + robot, 0.22 m disc, 0.35 m arms, 3.2 kg). **But**: parameters "had to be individually adjusted for each condition" (σ_y 0.8 vs 1.2 deg; τ 0.06 vs 0.15 s on the robot), no speeds reported, one-arm robot "extremely slow", rough terrain with several objects failed.
- Related: Kano et al 2019 Front Neurorobot 13:66 (DOI 10.3389/fnbot.2019.00066, PMC6716452, **ABSTRACT only**) — decentralised choice of travel direction through a nerve-ring model (the "which way" that B4/B5 take as given). Clark et al 2019 J Exp Biol 222:jeb192104 (DOI 10.1242/jeb.192104) — the nerve ring must connect the arms for coordination, signals route around a cut; more connected arms → more likely coordinated. Matsuzaka et al 2017 Biol Open 6:425 (DOI 10.1242/bio.019836) — mid-arm segments contact the floor to lift the disc; distal tips point up and do not touch (**cf. octopus: distal-to-foot portion "shows little change" — Katija**); anaesthetising all mid-segments kills coordination, but **one intact arm suffices to preserve it**. Watanabe et al 2012 J R Soc Interface 9:102 (DOI 10.1098/rsif.2011.0317) and Kano et al 2012 Bioinspir Biomim 7:034001 (DOI 10.1088/1748-3182/7/3/034001) — earlier version with an **active rotator** per arm (`dφ = ω − b sin φ + feedback`: oscillatory when ω > b, excitable when ω < b) giving spontaneous assignment of rhythmic vs non-rhythmic arm roles.
- **For us:** the honest caveat is that the purely physics-carried scheme needs per-configuration tuning once the limbs are long and flexible; the robust part is the **sign of `D · F_base`** as each arm's "am I helping?" bit, which is cheap, local, and degrades correctly.

#### B6. Owaki, Kano, Nagasawa, Tero & Ishiguro 2013 — "Simple robot suggests physical interlimb communication is essential for quadruped walking"
J R Soc Interface 10:20120669. DOI 10.1098/rsif.2012.0669 · PMC3565797. **ABSTRACT** (equation known from B7/B8, which cite it).
- Four **decoupled** phase oscillators, one per leg, NO neural coupling; each sees only its own foot's load: `dφ_i/dt = ω − σ N_i cos φ_i` (swing 0 ≤ φ < π, stance π ≤ φ < 2π). While the leg is loaded, the term pushes the phase toward 3π/2 (mid-stance) and **holds the leg in stance until the others take the weight**. Adapts to weight distribution and speed changes, reproduces quadruped walks.

#### B7. Owaki & Ishiguro 2017 — "A quadruped robot exhibiting spontaneous gait transitions from walking to trotting to galloping"
Sci Rep 7:277. DOI 10.1038/s41598-017-00348-9 · PMC5428244. **FULL** (equations stripped by the HTML export; the model is B6's).
- Same decoupled oscillators + foot pressure sensors (FSR). Changing **only ω** (one "MLR" scalar): ω = 6 rad/s lateral-sequence walk, 9 rad/s trot, 26 rad/s gallop, with a canter appearing transiently between. Gaits self-organise; CoT matches horses' energy-optimal transitions; hysteresis and critical slowing at transitions. Robot 2.0 kg, leg 0.20 m.
- Mechanism in words: LH loaded at end of stance → phase delayed in proportion to N → LF has time to land → CoM moves forward → LH unloads → LH swings. "N_i includes information regarding the physical situation of the other legs."
- Key argument against threshold-based load rules (Ekeberg, Maufroy): those switch stance↔swing at a load threshold and **depend on the threshold value**; a high threshold (< ~¼ body weight) is sensitive to body properties and perturbations. **Continuous** load-dependent phase modulation is more robust. The implicit threshold ω/σ splits **excitable** (low speed — leg waits for unloading, i.e. event-driven) from **oscillatory** (high speed) behaviour — a single equation covers "free gait" and "rhythmic gait".
- Authors explicitly name "automatic motion-creation algorithms for CG animations" as an application.

#### B8. Owaki, Goda, Miyazawa & Ishiguro 2017 — "A minimal model describing hexapedal interlimb coordination: the Tegotae-based approach"
Front Neurorobot 11:29. DOI 10.3389/fnbot.2017.00029 · PMC5465294. **FULL.**
- **Tegotae** ("how well the perceived reaction matches the intended action") as a design rule: write `T_i(φ_i, N) = C(φ_i)·S(N)` (intended action × sensed reaction) and set the feedback `f_i = ∂T_i/∂φ_i`, `dφ_i/dt = ω + f_i`. The rule makes sensory feedback **systematic** rather than hand-crafted.
- Hexapod instance: `dφ_i/dt = ω − σ1 N_i cos φ_i + σ2 (Σ_{j∈neighbours} k_j N_j) cos φ_i` — the first term: "stay in stance while I bear load" (late stance, cos φ > 0, phase slowed); the second has the opposite sign: "**when my neighbours bear load, I may lift off**" (late stance phase is sped up in proportion to the neighbours' load) — the pair is WALKNET's "do not lift while your neighbour is in swing" written as a continuous phase rule. Weights k_anterior 0.7, k_posterior 0.1, k_contralateral 0.2; σ1 = 0.2, σ2 = 1.2; ω = 2 rad/s.
- Results (10/10 trials each): converges to tetrapod from any initial phase **within about two periods**; ω 2→4 rad/s → tripod; +500 g on the hind body → stays tetrapod with longer hind duty; **both middle legs removed → keeps walking, converges to a trot/L-S walk without any change to the controller**. With σ2 = 0 (no neighbour term) insect-like gaits mostly fail (7/10 other patterns).
- **For us:** Tegotae gives a principled way to write each arm's local rule. For an octopus arm in crawl: action C = "I am pushing" (proximal elongation command), reaction S = foot load / proximal strain; release when the reaction no longer matches the intent (strain saturates, Katija's lift-off). Neighbour term: an arm should prefer to start a new step when its neighbours are loaded. With an XPBD sim, N_i is the constraint force at the foot attachment — available exactly, in integers.

#### B9. Sun, Xiong, Dai, Owaki & Manoonpong 2021 — "A comparative study of adaptive interlimb coordination mechanisms for self-organized robot locomotion"
Front Robot AI 8:638684. DOI 10.3389/frobt.2021.638684 · PMC8072274. **FULL.**
- Decoupled SO(2) neural oscillators per leg (discrete-time, 60 Hz: `a(n+1) = W o(n) + b + f(n)`, `o = tanh a`) + one of two feedback types:
  - **phase modulation (PM, Owaki)**: `f = −γ (F/mg) cos(o)` continuous, γ ≈ 0.36 (works in ~[0.12, 0.36]);
  - **phase resetting (PR, Aoi)**: on the touch-down EVENT (`F` crosses `F_t·mg/4` upward, F_t ≈ 0.64) jump the oscillator to a fixed reset point.
- Convergence to trot: PM ~6 s average (larger variance), PR ~1 s.
- Noisy feedback (20% GRF noise): **PR is worst** (spurious threshold crossings; worst case never resets) → PM wins. **Leg damage (one leg's hip + knee frozen)** and asymmetric load: **PR wins** (lower phase deviation, lower CoT, faster). Guidance: PM for symmetric load, PR for asymmetric; parameters outside the ranges stop convergence; suggest adaptive/learned gains.
- **For us:** dead arms make the octopus's load distribution asymmetric → favour the **event (reset) form** over continuous modulation; our sim is noise-free and deterministic, so PR's noise weakness does not bite (and we can add deterministic hysteresis anyway). Both are integer-trivial: PM is a LUT `cos` × load; PR is a comparison and an assignment.

#### B10. Matsuoka 1985/1987; Kuramoto-type chains (Cohen, Holmes & Rand 1982; Kopell & Ermentrout) — background only
- Matsuoka 1985, "Sustained oscillations generated by mutually inhibiting neurons with adaptation", Biol Cybern 52:367–376, DOI 10.1007/BF00449593; Matsuoka 1987, "Mechanisms of frequency and pattern control in the neural rhythm generators", Biol Cybern 56:345–353, DOI 10.1007/BF00319514. **Not fetched** (paywalled, Springer); from prior knowledge.
- Matsuoka unit: two mutually inhibiting leaky neurons with self-adaptation, `τ ẋ_i = −x_i − β v_i − w y_j + s`, `τ' v̇_i = −v_i + y_i`, `y_i = max(0, x_i)`. Oscillates for suitable (β, w, τ/τ'); frequency set mainly by τ; output is a half-centre burst. Popular in Taga's biped and many robots because it entrains to feedback. **Integer cost:** a 4-state linear ODE with a ReLU — fine in fixed point with explicit Euler and a Q16 state, but the frequency/amplitude are implicit functions of the parameters (hard to art-direct) and the oscillation condition is a parameter-space inequality — a bad fit for a designer-facing knob.
- Kuramoto / phase-oscillator chains: `θ̇_i = ω_i + Σ_j a_ij sin(θ_j − θ_i − φ_ij)` (Cohen–Holmes–Rand lamprey chain, J Math Biol 13:345–369, 1982, DOI 10.1007/BF00276069 (verified)). Phase lags are explicit parameters; the locked state is stable when couplings are strong enough relative to frequency spread. **Integer cost:** a phase accumulator in Q0.32 turns (wraps for free in uint32) + a sin LUT; coupling is a multiply-add. This is the cheapest CPG that exists and is what B2 uses.

### C. Rule-based / event-driven coordination

#### C1. Schilling, Hoinville, Schmitz & Cruse 2013 — "Walknet, a bio-inspired controller for hexapod walking"
Biol Cybern 107(4):397–419. DOI 10.1007/s00422-013-0563-5 · PMC3755227. **FULL** (review; some velocity-dependent formulas only as images).
- 18 DOF stick insect, **no central oscillator**: "the (quasi-) rhythmic movement is controlled by sensory-driven controllers". Each leg has a Swing-net and a Stance-net with **mutually inhibiting, low-pass-filtered motivation units** (winner-take-all) — a two-state machine with hysteresis.
- **Stance → swing** when the leg reaches its **PEP** (posterior extreme position; default AEP–PEP 0.3 m in the robot scale). **Swing → stance** on ground contact (later versions: an analog load signal crossing a threshold).
- **Coordination = neighbours shift each other's PEP** (Cruse's rules):
  - **Rule 1** (ipsilateral, back→front): "inhibits the beginning of swing of the receiver leg as long as the sender leg is in swing" (PEP pushed rearward) — never lift two adjacent legs at once.
  - **Rule 2** (ipsi back→front and contralateral): after the sender starts stance + **270 ms**, shift the receiver's PEP forward for **50 ms** — "your neighbour is down, you may go".
  - **Rule 3** (ipsi front→back, contralateral both ways): when the sender reaches a given position in stance, shift the receiver's PEP forward (velocity-dependent threshold) — produces the metachronal wave.
  - **Rule 4**: swing target = the anterior neighbour's current foot position (follow-the-leader foot placement).
  - **Rule 5**: load sharing — a loaded leg makes neighbours push harder; a heavily loaded leg's PEP moves rearward (keeps it down).
  - Rule 6 (treading-on-tarsus) not implemented.
- **Gaits emerge as a continuum** from the velocity command alone: wave (slow) → tetrapod → tripod (fast); "there exists a continuum of phase relations". Matches Graham 1972 within a 2× time scale.
- **Leg loss: "Walknet is able to overcome the loss of up to two legs"**; losing a middle leg needed one added load-dependent hind→front coupling. Recovers from stumbling, climbs obstacles, crosses body-wide gaps.
- Authors' view on CPGs: sensory-driven control suits unpredictable terrain at low speed; central oscillators are a "quick but dirty" approximation where feedback is too slow (fast rhythms). They suggest the two can coexist.
- **For us:** WALKNET is the closest engineering analogue of the octopus crawl as Levy/Katija describe it: no clock, a per-limb stance/swing state machine, release on a position/strain threshold, and a handful of neighbour rules that shift that threshold. It is **pure comparisons + timers** — trivially deterministic in integers. Rules 1 and 2 map directly onto "at most k arms in the air / a foot that just landed licenses its neighbour to release". Rule 4 is irrelevant (octopus foot placement is blind, Katija).

#### C2. Free gaits for walking machines (McGhee & Iswandhi 1979; Kugushev & Jaroshevskij 1975 follow-the-leader; Porta & Celaya 2004)
**Not fetched** — classic robotics, from prior knowledge; cite only as background.
- McGhee & Iswandhi 1979, "Adaptive locomotion of a multilegged robot over rough terrain", IEEE Trans SMC 9(4):176–182, DOI 10.1109/TSMC.1979.4310180: a "free gait" picks, at each step, the leg with the **least kinematic margin** (closest to its workspace limit) to lift next, subject to the **static stability margin** staying positive with that leg up. No periodic sequence.
- Porta & Celaya 2004, "Reactive free-gait generation to follow arbitrary trajectories with a hexapod robot", Robot Auton Syst 47(4):187–201, DOI 10.1016/j.robot.2004.03.008 — the reactive version: a leg steps when its reach error exceeds a threshold and its neighbours are planted (essentially Cruse rules 1–2 engineered).
- **For us:** "lift the arm with least remaining push stroke, if the rest still carry the body" is the one-line free-gait rule; for a radial animal under water (buoyant, no tipping), the stability margin reduces to "keep ≥ k_min arms planted", k_min ≈ 2 (Levy: ≥2 for walking; Katija: 2–6, mean 3).

#### C3. Game procedural stepping (Rain World, Overgrowth, common "spider leg" schedulers)
Strand (c) holds the games prior art; here only the coordination rule they share. Jakobsson, GDC 2016 "The Rain World Animation Process" (https://www.gdcvault.com/play/1023475/); Rosen, GDC 2014 "An Indie Approach to Procedural Animation" (https://www.gdcvault.com/play/1020583/). **Talk videos, not re-watched this session.**
- The shared game recipe: each limb has a planted target; it steps when its error (distance of the ideal foothold from the planted one) exceeds a threshold AND it is allowed to (alternating groups, or "no neighbour currently stepping", or a max-in-air count); the step is an eased arc of fixed duration. Rain World's creatures add a body that is physically pulled by whatever limbs are grabbing, so the gait is emergent from contact.
- **For us:** this is C1/C2 stripped to the bone; it is what makes procedural creatures look alive cheaply, and its known failure is looking "twitchy" when many limbs hit their threshold on the same frame — fixed in practice by a refractory time per limb and a global max-steps-per-tick (both integer).

### D. Learned coordination

#### D1. Shih, Naughton, Halder, Chang, Kim, Gillette, Mehta & Gazzola 2023 — "Hierarchical control and learning of a foraging CyberOctopus"
Adv Intell Syst 5(9):2300088. DOI 10.1002/aisy.202300088 · arXiv 2302.05811. **FULL** (arXiv HTML).
- The only multi-arm octopus RL I found. Three levels: **central** (PPO, 32×32×16 ReLU actor) picks a **primitive per arm** ("reach-all" or "crawl"); **arm** level executes the primitive by energy shaping (NN-ES, 200× faster than the optimiser it replaces); **local** level = contact reflex (on contact: relax tip-side muscles, raise longitudinal activation base-side; compliance does the rest).
- Central observation: positions of the F nearest uncollected food items relative to each arm base (F = 5 one arm, 10 two arms; zero-padded to a fixed size). Action: one discrete primitive choice per arm. Crawl = fixed step Δr along that arm's own direction (arms orthogonal) — **an abstracted, Levy-style "each arm has one push direction"**.
- Results: hierarchical PPO collects 66% (2 arms) / 88% (4 arms) of food vs greedy 54% / 38%; **end-to-end PPO on muscles was worse than every hierarchical variant with one arm and "too poor to provide any useful insight" with two**. 8 arms modelled in PyElastica, ≤ 4 active.
- **For us:** strong evidence for exactly our split — learn **which primitive each arm runs** (a small discrete action) over a hand-built primitive layer, never end-to-end on node forces. Crawl coordination itself was trivialised (orthogonal arms, fixed steps), so it does not tell us how to coordinate a real crawl.

#### D2. Huang, Mordatch & Pathak 2020 — "One policy to control them all: shared modular policies for agent-agnostic control" (SMP)
ICML 2020, PMLR 119. arXiv 2007.04976. **FULL** (arXiv HTML).
- One **shared** network module per actuator. Per-module observation = **only that limb's local state**: global position, linear and angular velocity, 3D rotation (exponential map), joint position normalised to [0, 1] of its range. Action = that limb's torque.
- Coordination comes from **learned 32-d messages** passed over the body tree: bottom-up (leaves → root, aggregated by sum/avg/max) then top-down (root → leaves; the top-down pass emits actions). Zero-padded child slots for variable child counts.
- Ablations: **no messages → limbs act independently and learn nothing meaningful**; one-way passing fails on 2–3-limb walkers; both-way matches a monolithic policy and learns several gaits at once. Zero-shot to **held-out morphologies made by removing leaf limbs** (figure-only numbers). TD3, planar MuJoCo walkers/humanoids/cheetahs, >20 morphologies.
- Limitation that matters: assumes a **tree**; cycles (a ring of arms around a body) untested — but an octopus is a star (body = root, arms = children), which IS a tree. Symmetry breaking between identical siblings (left/right legs) can fail with a common top-down message → each arm needs an **identity / position feature** (its angle around the body).
- **For us:** this is the plumbing shape for a learned replacement — per-arm module with local observation + a small message vector in and out through the body hub. A dead arm = a missing child; zero-padding handles it.

#### D3. Graph / attention successors (NerveNet, Amorpheus, MetaMorph) — background, not fetched
- Wang, Liao, Ba & Fidler 2018, "NerveNet: learning structured policy with graph neural networks", ICLR 2018 (https://openreview.net/forum?id=S1sqHMZCb): GNN policy over the body graph; zero-shot transfer between centipedes with different numbers of legs. **ABSTRACT level.**
- Kurin, Igl, Rocktäschel, Böhmer & Whiteson 2021, "My body is a cage: the role of morphology in graph-based incompatible control", ICLR 2021 (arXiv 2010.01856) — "Amorpheus": a **transformer over limb tokens** (no message passing along the graph) beats SMP/NerveNet; i.e. letting every limb attend to every other limb is better than respecting the body graph. **ABSTRACT level.**
- Gupta, Fan, Ganguli & Fei-Fei 2022, "MetaMorph: learning universal controllers with transformers", ICLR 2022 (arXiv 2203.11931) — same direction, scaled to 100 morphologies with morphology tokens. **ABSTRACT level.**
- **For us:** for eight identical arms around a hub, "every arm attends to every other arm" (transformer) and "arms talk through the hub" (SMP star tree) are the same topology. A Tsetlin machine cannot attend, so its observation must be a **fixed-size summary of the other arms** (counts and booleans, e.g. "number of arms planted", "left neighbour planted", "right neighbour planted") — which is precisely what the hand-written rules in C1/B4 read.

#### D4. Naughton, Sun, Tekinalp, Parthasarathy, Chowdhary & Gazzola 2021 — "Elastica: a compliant mechanics environment for soft robotic control"
IEEE RA-L 6(2):3389–3396. DOI 10.1109/LRA.2021.3063698 · arXiv 2009.08422. **ABSTRACT level** (strand (b) covers the rod model).
- Single Cosserat-rod arm benchmarks (3D tracking, reaching through obstacles) with TRPO/PPO/DDPG/TD3/SAC; no multi-arm coordination. Its relevance here is only as the precedent that the arm-level controller should be pre-built (cf. D1), not learned jointly with coordination.

### A (addendum). The one octopus gait that IS a fixed alternation

#### A6. Huffard, Boneka & Full 2005 — "Underwater bipedal locomotion by octopuses in disguise"
Science 307(5717):1927. DOI 10.1126/science.1109616 · PMID 15790846. **ABSTRACT only** (Brevia, 1 page + SOM).
- *Octopus marginatus* (coconut) and *Abdopus aculeatus* (algae clump) walk backwards on **two alternating arms with a rolling gait** (a propagating bend rolls the arm under the body — Katija's "rearward locomotion … propagating bend … rolling gait"), the other six arms held in a camouflage posture. "Each arm was on the sand for more than half of the stride" → duty factor > 0.5 → walking by definition.
- **For us:** when the octopus does move rhythmically on few arms, it is a strict **two-arm antiphase alternation**. That is the special case a phase rule handles best (one shared phase, arm B = arm A + ½). It is a different primitive from the crawl (rolling bend vs plant-and-push) and a candidate "sneak/walk" mode later; not part of the core crawl.

---

## Synthesis

### S1. Comparison of coordination schemes

Scores: ++ strong, + good, 0 neutral, − weak, −− contradicts / fails.

| Scheme (exemplars) | Realism vs octopus biology | Travel reliability | Dead arms (7, 6, …) | Integer feasibility | Art-directability | RL compatibility |
|---|---|---|---|---|---|---|
| **Central oscillator CPG** — coupled phase/Hopf/Matsuoka net with fixed phase biases (Ijspeert 2007, Righetti 2008 w/o feedback, FORTH G1–G4) | **−−** for crawl (Levy 2015/2017: no rhythm, no characteristic frequency, no order); **++** for umbrella swim and jet (genuinely periodic, synchronous) | ++ on flat ground; − on rough terrain without feedback | − a phase pattern designed for 8 leaves a hole; needs a re-tuned pattern per configuration (FORTH: removing arms changes CoT, not speed, in swim — swim is tolerant) | Phase oscillator ++ (uint32 phase, sin LUT); Hopf/Matsuoka 0 (multi-state ODE, Euler in Q16, oscillation conditions implicit) | ++ one drive knob, explicit phase lags, frequency, amplitude, stroke ratio | + drive/phase-bias as the action; but a learned arm brain fights a clock |
| **Decoupled phase oscillators + local load feedback (Tegotae)** (Owaki 2013/2017, Owaki 2017 hexapod; PM/PR, Sun 2021) | + the stance-end-by-load matches Katija's "release at max proximal strain"; − still imposes a per-arm period the lab octopus does not show; the excitable regime (low ω) is close to event-driven | ++ self-organises a gait from any start within ~2 periods; adapts to load and speed with one ω | + both middle legs removed → still walks, no change (Owaki 2017); phase resetting copes best with asymmetric load / frozen leg (Sun 2021) | + `φ += ω − σ N cos φ` is a LUT and a multiply; resetting is a comparison | + ω (speed), σ (how strongly load holds a stance), neighbour weights k | ++ per-arm local equation; σ, k learnable; observation = own + neighbour load |
| **Decentralised rule-based free gait** (Cruse WALKNET, McGhee free gait, brittle-star rules Kano 2017) | ++ no oscillator; per-limb stance/swing machine; neighbours shift thresholds; Kano's `a_j = r_j·d` IS Levy's direction-based recruitment | + emergent gait continuum; − can stall or dither if rules conflict (needs refractory/hysteresis) | **++** WALKNET survives loss of 2 legs; brittle star keeps speed with 1–2 arms lost, reroutes roles (rowing → reverse rowing), > 50% speed with one arm; robot coordinated "immediately after power-on" in every configuration | **++** comparisons, counters, dot products, `max`; no transcendentals | + thresholds (release strain, k_min/k_max planted), rule strengths, refractory times — each a readable dial | + rules become features/targets for a learned policy; a Tsetlin machine is literally a learned rule set |
| **Event-driven probabilistic recruitment** (Levy 2015/2017; game step schedulers) | **++** the published description of octopus crawl: moment-to-moment, probabilistic, direction-weighted subset selection, equal push forces, vector-sum steering | + travels as long as eligible arms exist; looks alive from the lottery; − with pure independence it can look jittery (needs refractory + neighbour bias) | ++ a dead arm leaves the lottery; others pick up the slack; steering coverage thins in the dead arm's direction (honest, like the brittle star) | ++ seeded integer PRNG + weighted draw (int64 sum) | ++ "how many arms down" target, ventral bias, randomness temperature, refractory | ++ the lottery weights ARE a policy's logits; replace hand weights by a learned scorer without changing plumbing |
| **Learned** (SMP, NerveNet/Amorpheus, CyberOctopus hierarchical PPO) | 0 whatever the reward makes it; CyberOctopus: hierarchical beats end-to-end decisively | 0–+ after training; end-to-end on muscles failed (Shih 2023) | + SMP zero-shot to removed limbs (shared module + zero-padded messages); needs training coverage | − NN inference in fixed point is doable (int8/int16 MLP) but training is float/off-line; Tsetlin machine ++ (bit operations, clause counts) | −− no dials; behaviour can only be shaped through reward/curriculum | ++ by definition — but needs a hand-written baseline behind the same interface to bootstrap and to compare |

**Reading of the table.** No single scheme fits all three modes. The crawl wants the event-driven/rule-based family (Levy + Kano + Cruse + Katija's strain-triggered release); the umbrella swim and jet want a central phase (FORTH, Ijspeert) because the animal genuinely synchronises its arms there. Tegotae is the bridge: its excitable regime is event-driven crawling and its oscillatory regime is rhythmic swimming, in one equation — but for the crawl we do not need its phase variable at all, only its principle (stance lasts as long as the arm's load matches its intent).

### S2. Recommended schemes

One **coordinator** per octopus (the "hub", sim-side, integer, part of synced state), three modes. Mode chosen by one scalar **drive** (Ijspeert 2007's MLR idea) with hysteresis: low → crawl, middle → umbrella swim, high or "escape" → jet. Body facing is a separate controller from travel direction (Levy 2015: independent).

#### S2.1 Crawl — event-driven, direction-weighted, probabilistic recruitment with neighbour rules (Levy + Kano + Cruse + Katija). NO oscillator.

Per-arm state machine (the primitive layer executes each state; the coordinator only issues transitions):
`IDLE → STEP (lift, place medial foot blind along own axis) → STANCE (proximal shorten, then elongate = push) → RECOVER (foot dragged off, arm swept under body; refractory) → IDLE`. Arms in a non-locomotor role (`HOLD`: reach, explore, grasp) and dead arms are outside the machine.

Fixed per-arm geometry: base angle α_i (L1..L4, R1..R4) → **push vector p_i** = direction the body moves when arm i elongates against its foot (pointing away from the foot; strand (a) to confirm the sign convention). Each tick: d = desired travel direction in the body frame (Q16 unit vector, rotated with the existing `sin_q16`/`cos_q16`); **alignment s_i = p_i · d** (Q16, in [−1, 1]).

Rules (every number is a dial; defaults from the sources):

- **R1 eligibility** — arm i may be recruited if alive, `IDLE`, refractory = 0, and `s_i ≥ s_min` (default cos 100° ≈ −0.17, so lateral arms pull-then-push too).
- **R2 target support count** — `K = clamp(K_base + wander, K_min, n_available)`, `K_base = 3`, `K_min = 2` (Katija: 3 ± 1 in contact; Levy: ≥ 2 for walking). `wander` = slow deterministic drift over ±1 with a period of minutes (Katija: the count cycled 2–6 over 2–3 min) — cheap life. The speed command raises K and push intensity and shortens the refractory time.
- **R3 recruitment lottery** (Levy's probabilistic recruitment; Kano's `a_j = r_j·d`) — at most ONE recruit per decision tick, only while `n_stance + n_step < K`. Weight
  `w_i = max(0, s_i − s_min) · B_pair(i) · (1 + β_wave · wave_i)` (Q16 products `>> 16`, sum in int64), drawn with the octopus's seeded integer PRNG (`r mod Σw`, cumulative pick; arm-index order breaks ties).
  - `B_pair` ventral bias (Katija: ventral arms have higher duty, pair 1 mostly probes): defaults pair 4 = 1.0, 3 = 0.9, 2 = 0.7, 1 = 0.3.
  - `wave_i = 1` if the next ALIVE arm on the same side, ventral of i, entered STANCE within the last `T_wave` ticks (Walknet rule 2; Katija's ventral→dorsal ordering). Looking past dead arms makes the wave skip holes instead of breaking.
- **R4 stance** — equal push intensity for all stance arms (Levy 2017: "virtually equal pushing forces"); steering is entirely by WHICH arms are down (vector sum of p_i).
- **R5 release** (STANCE → RECOVER), first to fire:
  (a) **strain saturation** `ε_prox ≥ ε_max` (Katija: the foot holds until the proximal third reaches maximal strain) — the normal end;
  (b) **Tegotae mismatch** — the arm commands push but its foot load `N_i < N_slip` for `T_slip` ticks (slipped, or doing no work);
  (c) **misalignment** — `s_i < s_min − h` after a direction change (hysteresis h).
  **Hold guard** (Walknet rule 1; Owaki's load-hold): (a) and (c) wait while `n_stance − 1 < K_min`, unless the primitive's hard limit is reached. At most one release per tick.
- **R6 recovery** — after release, refractory `T_ref` ticks before the arm is eligible again (gives the gait its breathing and prevents same-tick pile-ups — the game schedulers' fix for twitchiness).
- **R7 stuck fallback** (the brittle star's "reverse rowing", Kano 2017) — if Σw = 0 for `T_stuck` ticks (e.g. the arms behind the travel direction are dead), widen `s_min` toward −0.5 so leading arms crawl by pulling (proximal shortening). Speed drops, travel continues.

Dead arms need no special case: not alive → never eligible, never counted, skipped by the wave; K clamps to what is available; steering coverage thins in the dead arm's direction — the honest degradation the brittle star shows (Kano 2017: unchanged speed with 1–2 arms lost, > 50% with one arm left).
Arithmetic: dot products, `max`, compares, counters, an int64 sum, a uint32 PRNG — no transcendentals beyond the existing sin/cos LUT for the body-frame rotation. Deterministic by construction.

#### S2.2 Umbrella swim — one central phase accumulator, synchronous arms, asymmetric stroke.

- Hub phase `Φ` in uint32 turns, `Φ += Δ` per tick, `Δ = f · 2^32 / ticks_per_second`, f from a drive → frequency table (piecewise linear, Ijspeert's saturating map).
- Stroke time warp for slow open / fast close at 2.5 : 1 (FORTH: octopus recovery:power 2.5 ± 0.5): power fraction `ρ = 1/3.5 ≈ 0.286` turn; `u = Φ/ρ` in the close stroke, `(Φ − ρ)/(1 − ρ)` in the open stroke; pose from a LUT on `u`. One integer divide per octopus per tick, or a precomputed table.
- **All arms in phase** (FORTH G1; the biology). Optional cruise variant: diagonal pairs offset by Ts/4 (FORTH **G2**: same mean speed, smoothest velocity); never adjacent pairs (G3 wobbles).
- **Steer by amplitude, not phase**: `A_i = A · (1 + κ · (q_i · t))`, t = turn command, q_i = the arm's lateral axis (FORTH GR3: half amplitude on one side turns while translating; GR1: one side only spins in place).
- **Event gate** (Righetti 2008's contact gating, transposed): the hub holds Φ at the open→close boundary until ≥ M of the alive arms report their open pose reached (or a timeout) — keeps the synchrony honest when an arm is dragged, damaged or slow.
- Dead arms drop out; optionally scale down the diametrically opposite arm's amplitude to keep the thrust axis centred, else let the steering loop absorb the yaw. FORTH's arm-count data (CoT 0.98 / 1.07 / 1.63 for 2 / 4 / 8 arms) say the stroke, not the arm count, sets speed — a 7- or 6-arm umbrella degrades mildly.

#### S2.3 Jet — event-driven relaxation cycle; arms only bundle.

- Hub mantle machine `REFILL (slow) → READY → FIRE (fast) → GLIDE → REFILL`, repeating while the jet command holds; durations are dials (strand (a)'s numbers). A relaxation oscillator written as counters — no phase variable needed.
- At FIRE onset all alive arms switch to the `BUNDLE` primitive, staggered by a fixed deterministic per-arm delay of 0–2 ticks (arm-index hash) for life; arms trail through GLIDE; the hub returns to the previous mode when glide speed decays below a threshold.
- Dead arms: nothing to coordinate.

### S3. The per-arm brain interface (so a learned controller can replace the rules later)

Make the hand-written coordinator obey the SAME split a learned one will: a **per-arm policy** (one identical function for all eight arms: local observation in, request out) plus a **hub arbiter** (enforces hard constraints and the mode, owns the PRNG). SMP (Huang 2020) shows this shape works for learned control and survives removed limbs; CyberOctopus (Shih 2023) shows the action should be a primitive choice, not muscle or node forces.

**Observation per arm** (fixed size, all integer, thermometer-encodable for a Tsetlin machine):
- identity: arm index and base angle α_i (sin, cos in Q16) — needed to break symmetry between identical arms (SMP's failure mode with a shared top-down message);
- own state: state enum, ticks in state, refractory left, alive/health, role (locomotor / hold);
- own proprioception: proximal strain ε_prox (Q16), foot load N_i (the XPBD attachment constraint's λ, Q16), foot-attached bool, attached-sucker count;
- task: alignment s_i = p_i·d (Q16), speed command, mode;
- neighbours (left and right, skipping dead arms): state enum, foot load, ticks since entering STANCE;
- hub summary: n_stance, n_step, K, n_alive, body velocity in the body frame (2 × Q16), drive;
- (later) an incoming message vector m_hub→i (k integers, default 0).

**Action per arm:** a request `{none, recruit_me, release_me, hold_role(target), bundle}` + a score (Q16 — this IS the R3 weight today: the hand-written policy outputs `w_i`, the hub draws) + push intensity (Q16) + (later) an outgoing message m_i→hub.

**Hub arbiter** (stays hand-written): mode selection and the phase / mantle machines; the K_min hold guard; ≤ 1 recruit and ≤ 1 release per tick; the PRNG draw over the arms' scores; the stuck fallback. A learned arm brain therefore cannot make the animal collapse or twitch — it only reshapes *preferences*, which is what Levy says the octopus brain does.

Swap path: hand rules → (a) learn only the scorer `w_i = f(obs_i)` (a Tsetlin machine voting per arm, or a small integer MLP) with the hand scorer as the baseline it must beat; (b) learn release timing (`release_me`) next; (c) add messages only if needed (SMP's ablation: without messages limbs do not coordinate — here the hub summary plays that role from day one). Because the hub summary and neighbour fields are counts and booleans, a Tsetlin machine (which cannot attend over a variable set) sees exactly what the hand rules see.

### S4. Papers Erik should fetch (paywalled or not retrievable by me)

Priority 1 — directly shapes the crawl rules:
1. Levy, Flash & Hochner 2015, Curr Biol 25:1195–1200 — DOI 10.1016/j.cub.2015.02.064. The recruitment statistics (how many arms push, how arms are chosen vs direction, push durations). Cell open archive may serve it with a login.
2. Ijspeert 2008, Neural Networks 21:642–653 — DOI 10.1016/j.neunet.2008.03.014. The CPG review; I could only use it from memory.
3. Righetti & Ijspeert 2008, ICRA, pp. 819–824 — DOI 10.1109/ROBOT.2008.4543306 (free PDF on EPFL Infoscience record 130740 for a human browser). The stance/swing-gated Hopf oscillator; the equations in B3 are from memory.
4. Calisti, Corucci, Arienti & Laschi 2015, Bioinspir Biomim 10:046012 — DOI 10.1088/1748-3190/10/4/046012 (underwater SLIP for a legged octopus robot; only if strand (b) wants it).

Priority 2 — background / verification:
5. Ijspeert et al 2007 Science Supporting Online Material (coupling weights, drive thresholds) — DOI 10.1126/science.1138353.
6. Matsuoka 1985, Biol Cybern 52:367–376 — DOI 10.1007/BF00449593; Matsuoka 1987, Biol Cybern 56:345–353 — DOI 10.1007/BF00319514 (DOI from memory — verify).
7. Cruse 1990, "What mechanisms coordinate leg movement in walking arthropods?", Trends Neurosci 13:15–21 — DOI 10.1016/0166-2236(90)90057-H (from memory — verify); Dürr, Schmitz & Cruse 2004, Arthropod Struct Dev 33:237–250 — DOI 10.1016/j.asd.2004.05.004 (from memory — verify). Walknet's rules in their original form; C1's open review already covers most of it.
8. McGhee & Iswandhi 1979, IEEE Trans SMC 9:176–182 — DOI 10.1109/TSMC.1979.4310180 (free gait); Porta & Celaya 2004, Robot Auton Syst 47:187–201 — DOI 10.1016/j.robot.2004.03.008 (reactive free gait). Both DOIs from memory — verify.
9. Sfakiotakis, Kazakidi et al ICRA 2013 "Octopus-inspired eight-arm robotic swimming by sculling movements" and IROS 2014 "Multi-arm robotic swimming with octopus-inspired compliant web" — free PDFs on projects.ics.forth.gr/cvrl/octopus/publications/ (that host has a broken TLS chain for my fetcher; a browser should cope). The 2015 journal paper (A4) already contains the gait table.
10. Huffard, Boneka & Full 2005, Science 307:1927 + SOM — DOI 10.1126/science.1109616 (two-arm walking stride numbers, if a sneak/walk mode is ever wanted).

Open access, already read in full here (no fetch needed): Levy & Hochner 2017; Katija et al 2025; Sfakiotakis et al 2015 (AAM); Ijspeert et al 2007 (main text); Owaki & Ishiguro 2017; Owaki et al 2017 (Tegotae hexapod); Sun et al 2021; Kano et al 2017 (RSOS) and 2019 (Front Neurorobot); Schilling et al 2013 (Walknet); Shih et al 2023 (CyberOctopus); Huang et al 2020 (SMP).

### S5. Code worth reading

- **SMP** — https://github.com/huangwl18/modular-rl (PyTorch; `--td`/`--bu` flags toggle the message passes). Read the module and message-caching code for the per-limb interface shape.
- **Amorpheus** (transformer over limbs, Kurin et al 2021) — https://github.com/yobibyte/amorpheus (repo name from memory — verify).
- **NerveNet** — https://github.com/WilsonWangTHU/NerveNet (from memory — verify).
- **PyElastica / Gazzola lab** — https://github.com/GazzolaLab/PyElastica; look in the GazzolaLab organisation for the CyberOctopus / NN-ES code that goes with Shih et al 2023 (not verified that it is public).
- **FARMS** (EPFL BioRob's animal/robot simulation framework, incl. oscillator-network builders descended from the salamander CPG) — https://gitlab.com/farmsim (GitLab, not GitHub; read `farms_network` for a clean phase-oscillator network implementation).
- **Tsetlin machine** — https://github.com/cair/tmu and https://github.com/cair/pyTsetlinMachine (CAIR, Granmo's group; from memory — verify) — for the per-arm scorer once the interface in S3 exists.
- No public implementation found for the Tegotae controllers or the brittle-star rules; the equations in B4/B5/B8 are complete enough to write from.

### S6. Bottom line for #105

- **Crawl:** no oscillator. A hub that keeps about three arms in stance, recruits the next arm by a direction-weighted, ventral-biased lottery with a neighbour wave bonus, and releases an arm when its proximal strain saturates. Rhythm appears loosely as a ventral→dorsal wave; nothing in it assumes eight arms.
- **Umbrella swim:** one phase accumulator, all arms in phase, 2.5 : 1 time warp, steer by per-arm amplitude, gated on the arms actually reaching the open pose.
- **Jet:** a counter-based mantle cycle; arms bundle on the fire event.
- **Learning later:** keep the hub arbiter hand-written; let the learned per-arm brain output only the lottery score and release requests, from a fixed-size local + neighbour + hub-summary observation.
- **Numbers to calibrate against:** Katija 2025 (3 ± 1 arms in contact, range 2–6; ventral arms higher duty and shorter cycle; pair 1 mostly probing); FORTH 2015 (2.5 : 1 stroke, G1 vs G2 smoothness, arm-count CoT); Kano 2017 (speed unchanged with 1–2 arms lost, > 50% with one).

### Coverage notes

- Not covered from the intended list: Righetti/Buchli adaptive-frequency Hopf oscillators (not needed — our sim gives load directly, so frequency adaptation reduces to the event gate in S2.2); Kano/Ishiguro centipede and snake controllers (same TAKO-style local-feedback family as B4–B8; skipped for budget, nothing octopus-specific expected).
- Several equations are from memory and flagged in place (B1, B3, B10, C2); everything marked FULL was read this session. Full texts of open-access PubMed Central papers were retrieved via PubMed / PMC (DOIs listed per entry).
- Two scratch files, `part_s2.md` and `part_s3.md`, sit beside this digest in the scratchpad: they are the synthesis sections S2–S6 already appended here. Deleting them was not permitted; they can be removed.
