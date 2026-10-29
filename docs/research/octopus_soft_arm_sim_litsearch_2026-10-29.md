# Octopus soft-arm simulation: literature search, strand (b)

Date: 2026-10-29 · For: Breach octopus (#105 animation study) · Compiled by Claude (subagent)

Question: how to SIMULATE a soft octopus arm well enough that (1) a reach is a
travelling bend with a stiff arm behind it, (2) crawl pushes and jet strokes move
the body by reaction forces, (3) a limp arm drags and trails, (4) an arm can wrap
and hold a body — at well under 1 ms per arm per tick on CPU (later GPU, many
octopuses), in a game with a deterministic Q16.16 sim layer and a float render layer.
Existing rig: 17 bones per arm in game (32 segments in hero model), posed from 7
floats per arm (curl, bend dir, twist, tip curl, lift, swing, stretch).

Status legend per paper: FULL = full text read; ABSTRACT = abstract/summary only
(Erik to fetch via university account); PARTIAL = parts read (docs, preprint excerpts).

## Intended reading list (in order)

1. Cosserat rods / Gazzola lab
   - Gazzola, Dudte, McCormick, Mahadevan 2018, R Soc Open Sci, "Forward and inverse problems in the mechanics of soft filaments"
   - Zhang, Chan, Parthasarathy, Gazzola 2019, Nat Commun, "Modeling and simulation of complex dynamic musculoskeletal architectures"
   - Naughton et al. 2021, IEEE RA-L, "Elastica: a compliant mechanics environment for soft robotic control"
   - Chang et al. 2020 (CDC) "Energy shaping control of a CyberOctopus soft arm"; Chang et al. 2021/2023 "Controlling a CyberOctopus soft arm with muscle-like actuation"
   - Shih et al. 2023 "Hierarchical control and learning of a foraging CyberOctopus" (Adv Intell Syst)
   - Tekinalp et al. 2024 PNAS "Topology, dynamics, and control of a muscle-architected soft arm"
   - PyElastica docs + repo (licence, defaults, damping, dt)
2. Yekutieli, Sagiv-Zohar, Aharonov, Engel, Hochner, Flash 2005 J Neurophysiol I & II
3. Game/graphics cheap models
   - Jakobsen 2001 "Advanced character physics" (Verlet + constraint relaxation)
   - Müller et al. 2007 "Position based dynamics"; Macklin, Müller, Chentanez 2016 "XPBD"
   - Umetani, Schmidt, Stam 2014 "Position-based elastic rods"
   - Kugelstadt & Schömer 2016 "Position and orientation based Cosserat rods"
   - Bergou et al. 2008 "Discrete elastic rods"
   - Aristidou & Lasenby 2011 "FABRIK"; CCD for tentacles
   - (if found) Deul et al. 2018 "Direct position-based solver for stiff rods"; Müller "small steps" 2019
4. Drag / buoyancy / contact
   - Lighthill / Taylor 1952 resistive-force theory; Gray & Hancock 1955
   - PyElastica's drag/friction forcing (AnisotropicFrictionalPlane, slender-body theory forcing)
5. Locomotion controllers
   - Sfakiotakis, Kazakidi, Tsakiris et al. (FORTH) 2012–2015 octopus-like swimming robot
   - Kazakidi et al. CFD octopus arm
   - Cianchetti / Arienti et al. PoseiDRONE 2013
   - Calisti et al. 2011 "An octopus-bioinspired solution to movement and manipulation for soft robots" (crawling), Calisti 2015

---

## Digest (appended as read)

### 1. Cosserat rods (Gazzola lab)

#### 1.1 Gazzola, Dudte, McCormick & Mahadevan 2018 — "Forward and inverse problems in the mechanics of soft filaments"
R Soc Open Sci 5:171628. DOI 10.1098/rsos.171628. PMC6030325 (open access, CC BY). arXiv preprint 1607.00430 ("Computational mechanics of soft filaments").
**Status: FULL (arXiv/ar5iv version, first ~60 % of text in detail; publisher page 403'd to the fetcher but is OA via PMC).**
- The founding paper of "Elastica". Discretisation: n+1 NODES carry position r_i, velocity v_i, lumped mass m_i, external force; n ELEMENTS carry the material frame Q_i (directors d1,d2,d3), angular velocity, rotational inertia J_i, stretch e_i = l_i / l̂_i and shear strain σ_i = Q_i(e_i t_i − d3_i); n−1 interior VORONOI points carry curvature κ_i = log(Q_{i+1} Q_iᵀ)/D̂_i. Rotations via Rodrigues (sin, cos) and the matrix log (acos) — transcendentals in the inner loop.
- Integrator: symplectic 2nd-order position Verlet, ONE right-hand-side evaluation per step (half-drift, kick, half-drift).
- Time step is TINY relative to our tick: per test case dt = 1e-3·Δl to 1e-2·Δl (Δl in metres, dt in seconds). For a 0.3 m arm in 20 elements (Δl = 0.015 m) that is dt ≈ 1.5e-4 … 1.5e-5 s, i.e. ~300–3000 substeps per 50 ms game tick. This is the explicit stiff-stretch/shear CFL: shear/stretch stiffness is EA ~ E·πr², and wave speed sqrt(E/ρ) sets dt ~ Δl / c.
- Stiffness: B = diag(EI1, EI2, GI3); S = diag(4GA/3, 4GA/3, EA); G = 2E/3 in most cases (ν = 0.5, incompressible). Test materials E = 1e6–1e7 Pa, ρ = 1000 kg/m³, r = 0.025 m, L = 1 m.
- Damping: linear viscous f = −γ v, c = −γ ω, γ = 0.01 … 5 kg/(m·s) per case. Self-contact penalty k = 1e4, γ = 10.
- Hydrodynamics (Stokes regime, Re ≪ 1, resistive-force/slender-body): f_H = −(4πμ / ln(L/r)) · (I − ½ t tᵀ) · v, i.e. tangential drag coefficient is HALF the normal one (c∥ = 2πμ/ln(L/r), c⊥ = 4πμ/ln(L/r)). NOTE: this is the low-Re law; an octopus arm at 0.1–1 m/s in water is Re ~ 1e3–1e5, where drag is quadratic (see §4).
- Ground contact: penalty wall force + anisotropic Amontons–Coulomb friction with static/kinetic branches separately for axial (forward/backward) and lateral (rolling) directions; snake example μ_k forward : backward : lateral = 1 : 1.5 : 2. This anisotropy is what turns a travelling wave into net crawl.
- Muscle actuation: torque (or axial force) along a director with amplitude A(s,t) = β(s)·sin(2πt/T + 2πs/λ + φ); β(s) a cubic B-spline through N = 8 control points, ends pinned at 0. Gait parameters (β control points, λ, φ) optimised by CMA-ES. This IS a "travelling bend" generator — directly portable to an arm reach (bend wave with λ, speed, and an envelope).
- No wall-clock timings given (only "RHS is the expensive part").
- **For us:** the right reference physics; too expensive as-is in the sim (substep count), and the rotation log/exp + sqrt per element are hostile to Q16.16 integers. The actuation-as-travelling-wave formulation and the anisotropic friction model are directly reusable regardless of the solver.

#### 1.2 Zhang, Chan, Parthasarathy & Gazzola 2019 — "Modeling and simulation of complex dynamic musculoskeletal architectures"
Nat Commun 10:4825. DOI 10.1038/s41467-019-12759-5. PMC6811595. CC BY 4.0.
**Status: FULL (main text; Supplementary Notes with dt and drag coefficients not read).**
- Extends Elastica to ASSEMBLIES of rods: bones, tendons and muscles are all Cosserat rods, joined by soft penalty joints (spherical: force ∝ end separation with stiffness k; hinge and fixed joints: correcting torques). No Lagrange multipliers. Contact = penalty k + damping γ on overlap.
- Muscles = bundles of viscoelastic rods; force via contractile stress σ_m over cross-section, e.g. biceps = 2 heads × 18 rods. Snake: 1 m, r_max 25 mm, ρ 1000, E = 10 MPa, 100 vertebrae, 1–6 muscle pairs, peak force 0–3500 N, CMA-ES finds 4 muscles ≈ 0.58 m/s (≈ the continuous-actuation bound).
- Swimmer (biohybrid, Re ~ 1e-2) uses slender-body theory, μ = 1.2e-3 Pa·s; the pigeon wing (~3000 rods) uses a quadratic-in-velocity reduced aerodynamic drag — the authors state it cannot capture unsteady flapping. Same caveat will apply to a jetting octopus.
- Cost: linear in axial resolution (vs cubic for FEM) but NO timings, no hardware given. Code: github.com/mattialab/musculoskeletal-elastica (non-parallel version public).
- **For us:** shows the "muscle = actuator acting along/off-axis of the rod" idea scales to anatomy; for a game we only need the abstraction (a longitudinal-muscle pair = bend moment, transverse muscle = stiffening/elongation), not the rod-bundle anatomy.

#### 1.3 Naughton, Sun, Tekinalp, Parthasarathy, Chowdhary & Gazzola 2021 — "Elastica: a compliant mechanics environment for soft robotic control"
IEEE RA-L 6(2):3389–3396. DOI 10.1109/LRA.2021.3063698. arXiv 2009.08422.
**Status: FULL (ar5iv text; SI with dt / element count not included).**
- One Cosserat rod, base clamped upright, E = 10 MPa; RL (TRPO, PPO, DDPG, TD3, SAC) drives it through four tasks: 3D target tracking, reach-with-orientation, reach through a structured opening, navigate an obstacle nest.
- ACTION SPACE is the useful bit: internal bending TORQUES shaped by a spline with N control points and zero couple at both ends. Case 1: 6 control points per bending direction = 12 DOF; Case 2: + 6 twist points = 18 DOF; Cases 3–4: only 2 hand-placed points at 0.4 L and 0.9 L (2–4 DOF) suffice for obstacle reaching. Observation = 11 points along the arm + tip + target.
- Training needs 0.5–10 M policy steps; claims "near real-time" policy evaluation (no ms numbers). Code: github.com/GazzolaLab/PyElastica, MIT licence (docs cosseratrods.org).
- **For us:** our 7 floats/arm pose vector is in the same regime as their 2–4 DOF spline-torque cases; a sim arm driven by a handful of spline torque amplitudes is a sufficient control space for reach/avoid — strong precedent for "few actuation knobs → rod physics → pose".

#### 1.4 Elastica++ (Gazzola lab, 2026) — "Elastica++: a high-performance, multiphysics framework for large interacting assemblies of Cosserat rods"
arXiv 2605.13766 (May 2026). **Status: PARTIAL (HTML read for performance numbers).**
- C++ re-implementation (repo github.com/GazzolaLab/elasticapp; licence not named in paper). Up to 8.7× single-core speed-up vs their unoptimised baseline; SO(3) rotation kernel ~1 FLOP/byte, ~20 GFLOPS sustained, ~95 % of peak on an i9-11900K; strong scaling to 64 EPYC cores; scenarios up to 1,048,576 filaments. No GPU (named as future work). Does NOT state ns per element-step or dt.
- **For us:** even optimised, explicit Cosserat stays at dt ~ 1e-5 s for soft arms (see 1.5); its speed comes from throughput, not from larger steps. Our budget is the step count, not the FLOP rate.

#### 1.5 Chang, Halder, Shih, Tekinalp, Parthasarathy, Gribkova, Chowdhary, Gillette, Gazzola & Mehta — "Energy shaping control of a CyberOctopus soft arm" (IEEE CDC 2020, arXiv 2004.05747) and "Controlling a CyberOctopus soft arm with muscle-like actuation" (IEEE CDC 2021, arXiv 2010.03368)
DOIs (verified): 10.1109/CDC42340.2020.9304408 (CDC 2020, pp. 3913–3920); 10.1109/CDC45484.2021.9683318 (CDC 2021, pp. 1383–1390). **Status: FULL for 2021 (ar5iv); 2020 ABSTRACT only (same model, torque instead of muscle actuation).**
- THE best set of octopus-arm rod NUMBERS found: L = 20 cm, radius taper 12 mm base → 1.2 mm tip (linear), N = 100 elements, **dt = 1e-5 s**, E = 10 kPa (!), G = 40/9 kPa, ρ = 1050 kg/m³, linear damping γ = 0.02 kg/s. So a 50 ms game tick = 5000 Elastica steps per arm. Out of reach on CPU for 8 arms × many octopuses; this is why the sim needs a different solver.
- Muscle model (planar): two longitudinal muscles (LM top/bottom) at offset 2r(s)/3 from the centreline, each a force along the tangent whose couple ∓(2r/3)|n| bends the arm; one transverse muscle (TM) on the axis, contractile, which via incompressibility ELONGATES the arm (stretch ν_TM ≈ 2 − ν1). Force |n| = n_max · A_m · f_l(ν) · u, u ∈ [0,1]; f_l(z) = 3.06z³ − 13.64z² + 18.01z − 6.44 on 0.6 ≤ z ≤ 1.6 (squid tentacle, Kier & Curtin); n_max LM = 19.89 kPa over A/9, TM = 13.26 kPa over A/12. No force–velocity term.
- Control: static energy shaping — a bilevel optimisation (equilibrium solve + gradient steps on activations α(s), η = 1e-8) for reach (tip through three targets, "fetch-like") and grasp (distal 0.4L–L wraps a 2 cm sphere). No timings; no comparison to real bend propagation.
- **For us:** gives a 3-actuator-per-section abstraction that maps onto our pose floats: LM-top minus LM-bottom = curl/bend moment, LM co-contraction = stiffness, TM = stretch. The force-length cubic is a cheap polynomial — fixed-point friendly.

#### 1.6 Tekinalp, Naughton, Kim, Halder, Gillette, Mehta, Kier & Gazzola 2024 — "Topology, dynamics, and control of a muscle-architected soft arm"
PNAS 121(41):e2318769121. DOI 10.1073/pnas.2318769121. PMC11474103. CC BY-NC-ND 4.0.
**Status: FULL (main text; SI Appendix with dt, moduli, drag coefficients NOT read — in the PNAS SI PDF).**
- Anatomically resolved arm from histology/MRI of O. vulgaris: L = 20 cm, base diameter 24 mm; 197 Cosserat rods = 1 axial nerve cord (passive) + 8 longitudinal muscle rods + 8 oblique (4 right, 4 left, winding 74° at base) + 180 transverse-muscle-equivalent rings; neighbours bound by distributed displacement-force conditions; PyElastica v0.3.2, position Verlet; damping = Rayleigh + Laplacian filter; cross-sections rescaled with axial stretch (incompressibility).
- Drag: "simplified model from Yekutieli et al." (coefficients in SI §2.1.6). Contact friction: Haff & Werner on the outer oblique fibres (SI §2.1.5); too little friction → grasp fails.
- **Bend propagation REPRODUCED**: a stiffening wavefront of LM + TM CO-CONTRACTION travelling distally propagates the bend; peak bend velocity ≈ 30 cm/s, profile matching Gutfreund et al. Passive relaxation alone is "too slow and poorly localised". This is the mechanism to copy for our reach: the bend is the boundary between an activated (stiff, straight) proximal part and a passive distal part, and that boundary moves outward.
- Grasp = inject twist at the base via one oblique set, then an LM bending wave wraps the arm around the object; fails when the writhe |Wr| ≲ 1. Same activation sequence grasps cylinders up to a critical diameter and differently shaped objects. Reach / grasp / retrieve built from THREE primitives: bending wave, twisting, shortening.
- **For us:** confirms the primitive vocabulary (strand a) has a mechanical basis, and that a "stiffness front" controller produces the travelling bend without per-frame authoring. Cost: 197 rods × elements at dt ~1e-5 — a research model, not a runtime one.

#### 1.7 Shih, Naughton, Halder, Chang, Kim, Gillette, Mehta & Gazzola 2023 — "Hierarchical control and learning of a foraging CyberOctopus"
Adv Intell Syst 5(9):2300088. DOI 10.1002/aisy.202300088 (open access). arXiv 2302.05811.
**Status: FULL (arXiv HTML; SI tables with element count / dt not read).**
- Head + 8 arms in the scene, but only 1, 2 or 4 arms ACTIVE; each arm is a PLANAR tapered Cosserat rod (Elastica) with its base pose PRESCRIBED — the body has no dynamics. Muscles: LM1, LM2, TM, activations in [0,1], constant force-length, linear passive elasticity, viscous ζ; no obliques, no drag, no friction model.
- **CRAWL IS KINEMATIC**: TM extends the arm a fixed Δr, the suckers "grip" (only as an arm boundary condition), the muscles relax and the body is TRANSLATED by Δr. So even the flagship CyberOctopus does not get travel from reaction forces — the body displacement is scripted. Important negative result for our goal (2).
- Neural-network energy shaping (3 × 128 ReLU) replaces the iterative ES optimiser (≤10,000 iterations to 0.01 %): "200× faster", tip error < 1 % of arm length. High-level PPO picks per arm "reach" or "crawl Δr".
- Contact REFLEX worth stealing: on contact, a wave toward the tip relaxes all muscles, a wave toward the base raises LM activation on the contacting side; food collection in the obstacle arena rose from 21 % to 67 %, 9 of 11 obstacles passed. That is a cheap, local "wrap around what you touch" rule.

### 2. Yekutieli et al. 2005 — the classic 2D mass-spring octopus arm

#### 2.1 Yekutieli, Sagiv-Zohar, Aharonov, Engel, Hochner & Flash 2005 — "Dynamic model of the octopus arm. I. Biomechanics of the octopus reaching movement"
J Neurophysiol 94(2):1443–1458. DOI 10.1152/jn.00684.2004. PMID 15829594. **Status: ABSTRACT only (APS site 403 to the fetcher; not in PMC). Erik to fetch.**
- 2D multi-segment arm; each segment holds longitudinal (dorsal + ventral) and transverse muscles and keeps CONSTANT VOLUME (area in 2D) via internal pressure forces. Inputs = activation of each muscle. External forces: gravity, buoyancy, water DRAG (coefficients measured experimentally in this paper).
- Findings: (1) a single activation wave travelling at constant velocity reproduces natural reaching kinematics; (2) the mechanism is a STIFFENING WAVE of muscle contraction that pushes a bend forward along the arm; (3) **the perpendicular drag coefficient of an octopus arm is ~50× the tangential one**, and during a reach only a small part of the arm is perpendicular to its motion, so drag is minimised.
- **For us:** the single most useful number for the drag model (C⊥/C∥ ≈ 50 — much more anisotropic than the Stokes 2:1 of §1.1), and the mechanistic proof that "bend = moving stiffness front" works in a crude 2D mass-spring chain, not only in a Cosserat rod.

#### 2.2 Yekutieli, Sagiv-Zohar, Hochner & Flash 2005 — "Dynamic model of the octopus arm. II. Control of reaching movements"
J Neurophysiol 94(2):1459–1468. DOI 10.1152/jn.00685.2004. PMID 15829593. **Status: ABSTRACT only. Erik to fetch.**
- Only TWO control parameters specify a reach: activation AMPLITUDE and activation TRAVEL TIME (time for the wave to traverse the arm). The same kinematics arise for any amplitude above a minimum; larger amplitude = same motion, more stiffness against perturbation. Holds for constant and bell-shaped wave-velocity profiles.
- **For us:** a reach primitive is (amplitude, travel time) — and amplitude doubles as "how hard to push the arm off course", i.e. a natural knob for how a hit or a current deflects a reaching arm. Maps to our pose floats as: bend position s_b(t) advances at speed L / T_travel; stiffness proximal of s_b scales with amplitude.

#### 2.3 Engel, Szabo & Volkinshtein 2006 — "Learning to control an octopus arm with Gaussian process temporal difference methods"
NIPS 18, pp. 347–354. Open (proceedings.neurips.cc). **Status: FULL.**
- The Yekutieli model as an RL benchmark, described compactly: planar arm of N quadrilateral compartments, 2N+2 point masses at the corners (10 compartments → 22 masses, 88 state variables), dorsal and ventral longitudinal muscles and transverse muscles as damped springs, constant-AREA constraint per compartment enforced by internal pressure forces; gravity, buoyancy and drag on the point masses; arm "effectively 3D" for masses/volumes.
- Muscle law (linear, chosen for cheapness): **F(t) = (k0 + (kmax − k0)·A(t))·(ℓ(t) − ℓ_rest) + c·dℓ/dt**, A ∈ [0,1] — a spring whose stiffness the activation raises from passive k0 to kmax, plus a damper. Integration scheme and dt not given in this paper (refer to Yekutieli I).
- Later the "octopus arm" RL-competition benchmark (6–12 compartments, 50 state variables / 20 actions in some versions; OpenANN ships a 12-compartment one) — original Google-Code simulator URL is dead.
- **For us:** this activation-controlled stiffness spring is about the cheapest muscle that still gives a stiffening wave, and it is integer-friendly (multiply-adds only; the pressure term needs a quad area = cross products, no sqrt). The 2-strand (dorsal/ventral) quad chain is ≈ our 17-bone chain with a width.

### 3. Cheap models from games and graphics

#### 3.1 Jakobsen 2001 — "Advanced character physics"
Proc. Game Developers Conference 2001, San Jose (IO Interactive; used in Hitman: Codename 47). No DOI. Mirror PDF: edelkas.github.io/n/files/bib/jakobsen.pdf; also gamedeveloper.com. **Status: FULL.**
- Position (Störmer) Verlet with a FIXED step: store x and x_prev; x' = 2x − x_prev + a·dt². Velocity is implicit, so position and velocity "cannot drift apart" — fast, stable, dissipative. Global drag by replacing 2 with ~1.99 (i.e. x' = x + 0.99(x − x_prev) + a dt²).
- Constraints by RELAXATION (Gauss-Seidel projection): for a stick, delta = x2 − x1; diff = (|delta| − r)/|delta|; move each end half way. Repeat NUM_ITERATIONS times; "for most pieces of cloth only ONE iteration is necessary"; unconverged constraints catch up over the next frames thanks to Verlet. Plants in Hitman: one pass gave "exactly the right amount of bending".
- **SQRT-FREE stick** (1st-order Taylor of sqrt about the rest length r = one Newton step from guess r): `delta *= r²/(delta·delta + r²) − 0.5; x1 −= delta; x2 += delta;` — zero square roots, one division per constraint, r² precomputed. Exact fixed point (no change when |delta| = r). Mass-weighted variant given.
- Collisions by projection out of the obstacle; friction by damping the tangential displacement of a penetrating particle; rigid bodies and ragdolls = particles + stick constraints (+ angle constraints via sticks between second neighbours or min-distance sticks).
- **For us:** the whole sim layer for an arm could be this — 18 particles (17 bones), 17 sticks, 16 "bend" sticks or angle constraints. Integer-friendly: one division per constraint (Q16.16 has `floordiv`/`reciprocal_q16` kits), no sqrt, no transcendentals. Stiffness depends on iteration count and dt (the PBD flaw; fixed by XPBD) — irrelevant to us because both are FIXED constants in the sim.

#### 3.2 Müller, Heidelberger, Hennix & Ratcliff 2007 — "Position based dynamics" + Macklin, Müller & Chentanez 2016 — "XPBD: position-based simulation of compliant constrained dynamics"
PBD: J Vis Commun Image Represent 18(2):109–118, DOI 10.1016/j.jvcir.2007.01.005 — **ABSTRACT/known only (not fetched)**. XPBD: Proc. MIG '16, DOI 10.1145/2994258.2994272, PDF matthias-research.github.io/pages/publications/XPBD.pdf — **Status: FULL.**
- PBD step: predict x̃ = x + dt·v + dt²·M⁻¹f_ext; Gauss-Seidel project each constraint with Δx = k·s·M⁻¹∇C, s = −C/(∇C M⁻¹ ∇Cᵀ); v = (x_new − x)/dt. Flaw: effective stiffness rises with iteration count and with smaller dt.
- XPBD fix: carry a total Lagrange multiplier λ per constraint; compliance α (inverse stiffness), α̃ = α/dt²; **Δλ = (−C − α̃λ)/(∇C M⁻¹ ∇Cᵀ + α̃)**, Δx = M⁻¹∇Cᵀ Δλ. Stiffness becomes iteration- and timestep-independent and corresponds to the elastic potential U = ½ Cᵀα⁻¹C; λ/dt² gives the CONSTRAINT FORCE (useful: muscle tension readout, grip force, "is the arm holding?").
- Damping: Rayleigh potential with β̃ = dt²β; Δλ = (−C − α̃λ − γ∇C(x − x_n)) / ((1+γ)∇C M⁻¹∇Cᵀ + α̃), γ = α̃β̃/dt.
- Approximations: K ≈ M (drop geometric stiffness), g(x,λ) ≈ 0; validated "visually similar" to a Newton implicit solver.
- **For us:** XPBD is THE practical choice of solver family: unconditionally stable at large dt (it is an implicit-Euler-like projection), a muscle is a distance constraint with a TIME-VARYING REST LENGTH and/or compliance, and constraint forces fall out for reaction-force locomotion bookkeeping. One division per constraint, as Jakobsen.

#### 3.3 Angles, Rebain, Macklin, Wyvill, Barthe, Lewis, von der Pahlen, Izadi, Valentin, Bouaziz & Tagliasacchi 2019 — "VIPER: Volume Invariant Position-based Elastic Rods"
Proc. ACM Comput. Graph. Interact. Tech. 2(2) (SCA 2019), DOI 10.1145/3340260 (VERIFY). arXiv 1906.05260. Project/code: vcg-uvic.github.io/viper/. **Status: FULL (methods and engine sections).**
- PBD/XPBD Cosserat rod (centre-line positions + per-element frames, energies equal to Kugelstadt–Schömer's) PLUS one extra DOF per vertex: an isotropic SCALE (radius) and its velocity, with energies so that stretching the rod THINS it (volume invariance) — exactly the muscular-hydrostat behaviour (octopus arm elongates by thinning). Solved by Gauss-Newton prediction + XPBD-style correction (Eq. 28 "a generalisation of the XPBD update"), Jacobi on the GPU.
- **THEY RIGGED AN OCTOPUS**: "≈7 ms sim, ≈6 ms render for scenes containing up to 100 octopuses, each rigged with 37 pills, skinned to a ≈13k-face mesh", GTX 1080, non-optimised Thrust/CUDA prototype; overall "26,000 DOFs at 140 Hz". Collision: uniform-grid broad phase, pill–pill narrow phase by a 10-iteration dichotomous search, one PBD constraint per pair; floor = half-plane.
- Muscle activation demo: shorten the fibres (rest length) or insert internal forces → bulging at constant volume.
- **For us:** direct precedent for (a) many octopuses on GPU with rods-as-pills, (b) using the simulated rod as a SKINNING rig for a render mesh, (c) radius-as-DOF for "stretch thins the arm" — our `stretch` pose float. Its frames-as-quaternions part is what we would drop in the sim (float, normalisations).

#### 3.4 Kugelstadt & Schömer 2016 — "Position and orientation based Cosserat rods"
Proc. SCA 2016, pp. 169–178. DOI 10.2312/sca.20161234. Author PDF animation.rwth-aachen.de/media/papers/2016-SCA-Cosserat-Rods.pdf (connection refused from here; Eurographics DL 403). **Status: ABSTRACT only. Erik to fetch.**
- Frames as unit QUATERNIONS treated as PBD variables like positions; a stretch-shear constraint couples each segment's two end positions to its quaternion's d3; a bend-twist constraint on the Darboux vector between consecutive quaternions; unit length kept by renormalisation. Claimed > 10× faster than earlier position-based rod methods (Umetani 2014's ghost points). (Constraint detail confirmed indirectly via VIPER, which states its energies match Kugelstadt–Schömer's.)
- **For us:** gives TWIST (and therefore the oblique-muscle "twist then wrap" grasp of Tekinalp 2024) inside PBD — but quaternion normalisation needs a sqrt or rsqrt per element per iteration; in Q16.16 that is doable with a fixed-iteration integer Newton rsqrt, but it is the one non-trivial integer kernel. A 2.5D top-down game may not need twist in the sim at all.

#### 3.5 Macklin, Storey, Lu, Terdiman, Chentanez, Jeschke & Müller 2019 — "Small steps in physics simulation"
Proc. SCA '19. DOI 10.1145/3309486.3340247. PDF mmacklin.com/smallsteps.pdf. **Status: FULL.**
- Central finding: n SUBSTEPS × 1 XPBD iteration beats 1 step × n iterations at the same cost — error falls ~quadratically with substep size, and the artificial (implicit) damping of large steps disappears. Collision detection done ONCE per frame and reused across substeps.
- Numbers: hanging chain of 20 particles (1 kg, inextensible links of 0.01 m) with a 1:100,000 mass ratio: max error 322.1 m with 100 iterations vs 3.2 m with 100 substeps. Rigid-body chain + weight at 1:1000: 1 step × 100 iterations = 4 ms/frame with visible joint separation; 1 × 500 = 17 ms, still stretched; 100 substeps × 1 = 6 ms, stiff and stable. Cloth 150k particles / 896k springs: 1×30 = 12.4 ms (stretchy) vs 30×1 = 13.5 ms (stiff). Stays stable at k = 1e7 N/m where 40-substep semi-implicit Euler explodes.
- **For us:** the recipe for a sim arm: a 30 Hz tick split into a FIXED number of substeps (e.g. 4–8) with one Gauss-Seidel pass each. Deterministic by construction (fixed counts, fixed order). The arm–body mass ratio (arm tip ~1/100 of mantle) is exactly the stress case where substeps matter.

#### 3.6 Deul, Kugelstadt, Weiler & Bender 2018 — "Direct position-based solver for stiff rods"
Comput Graph Forum 37(6):313–324. DOI 10.1111/cgf.13326. **Status: ABSTRACT only (search summaries). Erik to fetch if stiff rods are wanted.**
- Rods/trees of rigid segments with Cosserat bend-twist constraints, but solved by Newton iterations with a DIRECT solver over the whole acyclic (tree) structure instead of Gauss-Seidel; ~10,000-segment rods, ~2 orders of magnitude faster than XPBD at equal stiffness (inextensible, stiff). A tree-structured system solves in linear time (block-tridiagonal elimination along the tree).
- **For us:** the right tool only if arms must be truly inextensible and very stiff; an octopus arm is the opposite (stretches 1.4×, soft), so Gauss-Seidel XPBD with substeps suffices. Noted for completeness and for the octopus BODY-PLUS-8-ARMS tree if we ever want it exact.

#### 3.7 Umetani, Schmidt & Stam 2014 — "Position-based elastic rods"; Bergou, Wardetzky, Robinson, Audoly & Grinspun 2008 — "Discrete elastic rods"
Umetani: Proc. SCA 2014 (Eurographics/ACM), DOI 10.2312/sca.20141119 (VERIFY). PDF research.autodesk.com (> 10 MB, fetch failed). **Status: ABSTRACT/known only.** Bergou: ACM TOG 27(3):63 (SIGGRAPH 2008), DOI 10.1145/1360612.1360662. **Status: ABSTRACT/known only.**
- Umetani: twist inside PBD via "ghost points" — one extra particle per edge defines the material frame; bend and twist as position constraints; superseded in speed by Kugelstadt–Schömer (3.4).
- Bergou DER: centre-line + per-edge twist angle, curvature binormal κb at vertices, Bishop (parallel-transport) frame, bending energy from κb, twist from the angle difference; the standard for hair/rods in film; explicit or implicit integration, generally needs Newton solves for stiff rods at large dt; parallel transport needs normalisations/rotations (sqrt, trig) — heavy for Q16.16.
- **For us:** neither is a runtime candidate for the sim; DER is a candidate for a high-quality RENDER-side secondary simulation only if XPBD rods look wrong, which is unlikely.

#### 3.8 Aristidou & Lasenby 2011 — "FABRIK: a fast, iterative solver for the inverse kinematics problem"; CCD (Wang & Chen 1991)
Graphical Models 73(5):243–260. DOI 10.1016/j.gmod.2011.05.003. Author page andreasaristidou.com/FABRIK. **Status: FULL.**
- FABRIK: forward pass from the end effector (set p_n = target, then each p_i on the line to p_{i+1} at distance d_i: λ = d_i/|p_{i+1} − p_i|, p_i = (1−λ)p_{i+1} + λp_i), then backward pass from the root; repeat to tolerance. No angles, no matrices — one sqrt + one division per joint per pass. Joint limits by projecting onto cones/orientation frames between passes; multiple end-effectors by sub-bases.
- Table 1 (10-joint chain, MATLAB on a 2.2 GHz Core 2, unoptimised, 20 runs): FABRIK 15.5 iterations / 13.3 ms (0.86 ms per iteration) to a reachable target; CCD 26.3 iterations / 124 ms (4.69 ms/it); Jacobian transpose 1311 iterations / 13 s; FABRIK ~10× faster than CCD, ~1000× faster than Jacobian methods. CCD "can roll and unroll itself" and over-emphasises joints near the end effector — wrong for an octopus reach (which bends at a travelling point, not at the tip).
- **For us:** FABRIK's forward/backward pass is literally a Jakobsen stick relaxation with the root and tip pinned — the same kernel as the PBD chain. It gives "an arm that reaches a point", but no dynamics, no reaction force, no trailing, no wave. Good for: the TARGET of a reach (where should the tip end?), a planted-sucker arm in a crawl (pin root and tip, solve the middle), and the render-side touch-up (make the rendered tip meet the grasped object exactly).

### 4. Drag, buoyancy, ground contact

#### 4.0 The regime question (synthesis from 1.1, 2.1, 4.2)
Octopus arms move at Re_D ≈ 1000 (base diameter; Gutfreund 1996 and Yekutieli 2005 via Kazakidi; Kazakidi simulates Re_D = 1000–3000). That is the INERTIAL regime: drag ∝ ½ρ·C·A·|v|v (quadratic), not the Stokes / resistive-force law of Elastica's slender-body module (1.1: linear in v, C⊥/C∥ = 2). Linear drag has a property fatal to us: a rigid paddle stroked out and back along the same path nets ZERO impulse whatever the speed profile (the impulse is ∝ displacement), so fast-power / slow-recovery sculling cannot work. Quadratic drag makes the fast power stroke push harder than the slow recovery takes back — exactly how octopus sculling and the FORTH robots make thrust (Kazakidi 2015: harmonic oscillation → ~zero net thrust; slow recovery + fast power at speed ratio β = 3 → net thrust).

#### 4.1 Taylor 1952 — "Analysis of the swimming of long and narrow animals"; Lighthill 1971 — "Large-amplitude elongated-body theory of fish locomotion"
Taylor: Proc R Soc Lond A 214:158–183, DOI 10.1098/rspa.1952.0159. Lighthill: Proc R Soc Lond B 179:125–138, DOI 10.1098/rspb.1971.0085. **Status: known / ABSTRACT only (not fetched).**
- Taylor's high-Re RESISTIVE model is what a game needs: each segment feels independent normal and tangential forces from its own velocity relative to the water; the normal force depends only on the normal velocity component (cross-flow principle), F_N = ½ρ·d·C_N·v_N|v_N|·ds; the tangential (skin friction) force is much smaller.
- Lighthill adds the REACTIVE (added-mass) force: the momentum of the water carried by each moving segment, m_a ≈ ρπ(d/2)² per unit length for a cylinder. Relevant to the jet stroke and to the arm "whip".

#### 4.2 Kazakidi, Vavourakis, Tsakiris & Ekaterinaris 2015 — "A numerical investigation of flow around octopus-like arms: near-wake vortex patterns and force development"
Comput Methods Biomech Biomed Engin 18(12):1321–1339 (DOI 10.1080/10255842.2014.900757 — VERIFY). Accepted manuscript: strathprints.strath.ac.uk/60914. **Status: FULL (accepted manuscript).**
- CFD on a tapered arm (base D, tip 0.1 D, length 10 D; with and without suckers 0.02–0.04 D high), Re_D = 1000 steady, 1000–3000 impulsive.
- A CLOSED-FORM force model for a tapered rough arm at incidence α (angle between arm axis and flow): normal **F_N = ½ρU²(R_b + R_t)L·(C_p1 sin²α + π·C_f1 sin α)**, axial **F_A = ½πρU²[R_b²·C_p2 cos²α + (R_b + R_t)L·C_f2 cos α]** (pressure + skin friction). Fitted: C_p1 = 0.27, C_p2 = −0.39, C_f1 = 0.12, C_f2 = −0.01 (RMS 1.1e-4); excluding the vortex-breakdown band α = 40–60°: C_p1 = 0.07, C_p2 = −0.15, C_f1 = 0.20, C_f2 = 0.03 (RMS 2.4e-5). C_D/C_L ≈ 15 at α = 0° and 90°; drag rises with α.
- Forces in IMPULSIVE rotations are independent of morphology and angle; slow-recovery / fast-power periodic motion gives considerable thrust for one arm, harmonic motion does not.
- **For us:** the per-segment form "normal ∝ v_N², axial small" is the anisotropic quadratic drag to use. Per segment of length ds, local radius R: F_N ≈ ½ρ(2R·ds)·C_N·v_N|v_N|, F_T ≈ ½ρ(2πR·ds)·C_T·v_T|v_T|, C_N ~ 1 (cylinder), C_T ~ 0.01–0.03.

#### 4.3 Yekutieli 2005 I — measured drag (supplement to 2.1)
**Status: ABSTRACT + search-index fragments of the full text (UNVERIFIED — Erik's copy must confirm).** The fragments indicate per-segment drag ½ρ·S·c·|v|² for perpendicular and tangential components separately, with Reynolds-dependent coefficients after Vogel (1981): c_per = 1 + 10·Re_per^(−2/3), c_tan = 0.64·Re_tan^(−1/2); measured tangential c_tan ≈ 0.0256 ± 0.0017; perpendicular ≈ 50 × tangential (this ratio is in the abstract, verified). At Re ~ 1000: c_per ≈ 1.1, c_tan ≈ 0.02 — consistent with 4.2.

#### 4.4 Buoyancy and gravity
- Octopus tissue is slightly denser than sea water (≈1040–1050 vs 1025 kg/m³; the Chang/Tekinalp rods use ρ = 1050): in water an octopus is NEARLY NEUTRAL, net weight ≈ 2–3 % of its weight in air. Calisti 2015 (5.3): the offset between centre of buoyancy and centre of mass, and the density ratio, set the STABILITY and SPEED of underwater walking — moving only the CoB with the same feed-forward activation gives forward or backward gaits at different speeds.
- In AIR (most Breach scenes are on a ship deck) an arm cannot hold itself up; real octopuses out of water drag their arms and pull with suckers. Model: per-node force m·g·(1 − ρ_fluid/ρ_body) along −z. In a top-down sim this mostly matters as the NORMAL LOAD for friction and for the `lift` pose float (an arm lifted in air sags; in water it floats).

#### 4.5 Ground contact and friction for crawling
- Elastica (1.1; gym-softrobot `build.py`): penalty plane contact (k = 1e2, ν = 1e1 for the 8-arm gym octopus) + ANISOTROPIC Coulomb friction, kinetic μ forward : backward : sideways = 1 : 1.5 : 2, static = 2 × kinetic, base μ = L/(T²·g·Fr) with T = 2 s, Fr = 0.1 → μ ≈ 0.089 for a 0.35 m arm. Anisotropy rectifies a symmetric wave into net crawl for a SNAKE; an octopus uses SUCKERS instead — a near-infinite static friction (an anchor) switched on and off by the controller.
- **Recommendation:** each arm node in ground contact has two states: SLIDING (Coulomb, μ_k ≈ 0.1–0.3 on a deck, anisotropy optional) and ANCHORED (a positional attachment constraint at the contact point, released by the controller or when the pull exceeds a grip limit F_grip). In XPBD an anchored node is a zero-compliance attachment whose λ/dt² IS the grip force; when it exceeds F_grip the sucker pops. One mechanism gives goal (2), crawl push-pull with travel from reaction forces, and goal (4), wrap and hold.

### 5. Locomotion controllers (crawl, sculling, jet)

#### 5.1 Calisti, Giorelli, Levy, Mazzolai, Hochner, Laschi & Dario 2011 — "An octopus-bioinspired solution to movement and manipulation for soft robots"
Bioinspir Biomim 6(3):036002. DOI 10.1088/1748-3182/6/3/036002. PMID 21670493. **Status: ABSTRACT only. Erik to fetch.**
- Tests a biological hypothesis of octopus crawling: PUSHING-based locomotion — the arm anchors distally and ELONGATES, pushing the body away from the anchor (rather than pulling toward it). A silicone arm with embedded cables mimicking longitudinal and transverse muscles both crawls and grasps with minimal control.
- **For us:** crawl = anchor + elongate (our `stretch` float, the "TM" actuator); the body moves because the anchored arm extends — reaction force, not scripted translation.

#### 5.2 Cianchetti, Calisti, Margheri, Kuba & Laschi 2015 — "Bioinspired locomotion and grasping in water: the soft eight-arm OCTOPUS robot"
Bioinspir Biomim 10(3):035003. DOI 10.1088/1748-3190/10/3/035003. PMID 25970014. **Status: ABSTRACT only.**
- Fully soft eight-arm robot, buoyancy near neutral; FRONT arms elongate and grasp, REAR arms walk; walks in water with the animal's strategy over different surfaces and through constraints; grasps varied objects thanks to the soft conical arms.
- **For us:** the functional split of arms (front manipulate, rear walk) is a cheap controller structure.

#### 5.3 Calisti, Corucci, Arienti & Laschi 2015 — "Dynamics of underwater legged locomotion: modeling and experiments on an octopus-inspired robot"
Bioinspir Biomim 10(4):046012. DOI 10.1088/1748-3190/10/4/046012. PMID 26226238. **Status: ABSTRACT only.**
- A reduced model with TWO propulsive actions: ground REACTION forces from leg contact + HYDRODYNAMIC forces from the legs' sculling; parameters fitted by evolutionary search. CoB–CoM separation and density set stability and speed; sculling propels the robot even with legs off the ground.
- **For us:** support for a lumped "body + legs + quadratic drag" model being sufficient for underwater walking; a per-octopus buoyancy offset is a cheap behavioural knob.

#### 5.4 FORTH swimming robots — Sfakiotakis, Kazakidi, Tsakiris et al.
- Sfakiotakis, Kazakidi & Tsakiris 2015, "Octopus-inspired multi-arm robotic swimming", Bioinspir Biomim 10(3):035005, DOI 10.1088/1748-3190/10/3/035005 (accepted MS, 24 MB, strathprints 60917 — too large to fetch here). **Status: ABSTRACT.** Arm swimming as SCULLING (fast power stroke, slow recovery); a lumped-element model with arm compliance and fluid interaction; gaits for forward, backward, turning; eight polyurethane arms; ≈ 0.26 body lengths/s (~100 mm/s), thrust up to 3.5 N, cost of transport 1.42 (8 arms) / 0.9 (2 arms).
- Sfakiotakis et al. IROS 2014, "Multi-arm robotic swimming with octopus-inspired compliant web": arms joined by a passively compliant WEB (the umbrella) → 0.5 BL/s, up to 10.5 N, CoT down to 0.62 — the web roughly DOUBLES speed and triples thrust. **Status: ABSTRACT (search summary).** (ICRA 2013 sculling-only eight-arm prototype: ≈ 0.2 BL/s.) FORTH PDFs at projects.ics.forth.gr/cvrl/octopus/publications/ failed TLS verification from here.
- Kazakidi, Tsakiris, Angelidis, Sotiropoulos & Ekaterinaris 2015, "CFD study of aquatic thrust generation by an octopus-like arm under intense prescribed deformations", Comput Fluids 115:54–65, DOI 10.1016/j.compfluid.2015.03.009. **Status: FULL (accepted MS, strathprints 60921).** Re_D = 1000; sinusoidal oscillation → effectively zero net thrust; sculling with recovery ω_r = 10°/s and power ω_p = β·ω_r, β = 3, amplitude 10°, C²-smooth polynomial transitions → net thrust; adding a travelling-wave UNDULATION (scaled amplitude 0.1, 0.05 wavelengths) synchronised with the strokes INCREASES thrust.
- **For us:** the jet-plus-umbrella swim needs no fluid solver: quadratic anisotropic drag per arm segment + the web as extra drag area between adjacent arms during the power stroke (a web panel's normal drag ½ρ·C_N·A·v_N|v_N| split onto the arm nodes bounding it) + a mantle jet impulse.

#### 5.5 Pulsed jet / mantle: Giorgio-Serchi & Weymouth 2016; Weymouth & Triantafyllou 2013; PoseiDRONE
- Giorgio-Serchi & Weymouth 2016, "Drag cancellation by added-mass pumping", J Fluid Mech 798:R3, DOI 10.1017/jfm.2016.353, arXiv 1604.02663. **Status: ABSTRACT.** A volume-changing body in still fluid: recovering added-mass kinetic energy can cancel viscous drag completely; sustained oscillation amplitude up to ~4 r₀ when pumping at 1 − a/(2r₀) < ω²/ω_n² < 1 + a/(2r₀).
- Weymouth & Triantafyllou 2013, "Ultra-fast escape of a deformable jet-propelled body", J Fluid Mech 721:367–385, DOI 10.1017/jfm.2013.65. **Status: known / ABSTRACT only.** A shrinking body sheds its added mass and gains thrust beyond the jet momentum.
- Giorgio-Serchi, Arienti & Laschi 2013, "Biomimetic vortex propulsion: toward the new paradigm of soft unmanned underwater vehicles", IEEE/ASME Trans Mechatron 18(2):484–493, DOI 10.1109/TMECH.2012.2220978; Arienti, Calisti, Giorgio-Serchi & Laschi 2013, "PoseiDRONE: design of a soft-bodied ROV with crawling, swimming and manipulation ability", MTS/IEEE OCEANS 2013. **Status: not fetched.** A cable-driven mantle collapses to expel a pulsed jet (vortex-ring formation).
- **For us:** a jet stroke = mantle volume V(t) drops over the power phase; thrust ≈ ρ·(dV/dt)²/A_funnel along the funnel axis, plus the added-mass term (shedding added mass helps); slow refill. The arms trailing behind the mantle in a jet are passive and dragged — the anisotropic quadratic drag produces the streamlined trailing-arm look automatically (goal 3).

#### 5.6 Open-source octopus environment: gym-softrobot (Kim et al., Gazzola lab)
github.com/skim0119/gym-softrobot, MIT licence, PyElastica-based. **Status: PARTIAL (source read: `envs/octopus/build.py`, `muscle_crawl_env.py`).**
- Default 8-arm octopus: 11 elements per arm, arm 0.35 (m), base radius 0.007, arm ρ 1000, head ρ 700, E = 1e6, damping 1e-2, head = rigid cylinder r 0.04 joined by `FixedJoint2Rigid` (k 1e6, kt 1), **dt = 7e-5 s** (≈ 700 steps per 50 ms tick, even at 11 elements), friction plane as in 4.5. In the base build the head is PINNED; `OctoMuscleCrawl-v0` has muscle channels (tm, lm0–lm3, om±) plus base and middle SUCTION channels and rewards the base sphere's forward progress — the closest open precedent to a reaction-force crawl, at research-code cost.

---

## Synthesis

What the literature says, in four sentences. (i) The travelling bend of a reach is a moving STIFFNESS FRONT — a wave of muscle co-contraction that straightens and stiffens the arm behind it and pushes a passive bend ahead of it; this holds in a crude 2D mass-spring chain (Yekutieli 2005) and in a 197-rod anatomical Cosserat model (Tekinalp 2024: peak bend velocity ≈ 30 cm/s), and two parameters specify a reach (activation amplitude, travel time). (ii) Research-grade octopus arms (Elastica) run at dt = 1e-5 to 7e-5 s, i.e. 700–5000 explicit steps per 50 ms game tick per arm — out of budget by two to three orders of magnitude, and their "crawl" is either scripted (CyberOctopus: the body is translated by Δr) or research code (gym-softrobot). (iii) Games and graphics get stable large-step rods from position-based dynamics: XPBD gives timestep- and iteration-independent stiffness with one division per constraint, substepping with one iteration per substep beats iterations (Macklin 2019), and a GPU XPBD rod engine ran 100 rod-rigged octopuses in ≈7 ms (VIPER 2019). (iv) Octopus arms live at Re ≈ 1000, where drag is QUADRATIC and strongly anisotropic (normal ≈ 50× tangential per Yekutieli; closed-form tapered-arm fits in Kazakidi 2015) — and only quadratic drag lets a fast-power / slow-recovery stroke make net thrust, which is how sculling and the FORTH robots swim.

### S1. Candidate arm models compared

Costs are my ESTIMATES for one 17-segment arm (18 nodes) per 50 ms game tick on one CPU core in C++, unless a paper is cited; they are order-of-magnitude, to be measured.

| Model | Travelling bend realism | Reaction forces for locomotion | Cost / arm / tick | Large-step stability | Fixed-point (Q16.16) feasibility | Implementation effort |
|---|---|---|---|---|---|---|
| **Cosserat rod, explicit (PyElastica / Elastica++)** | Best: emerges from muscle activations (Tekinalp 2024, Yekutieli-validated) | Exact (momentum-conserving internal forces + friction/drag) | 700–5000 steps × 11–100 elements (dt 7e-5 … 1e-5 s, gym-softrobot / Chang 2021): ≈ 1–50 ms in C++, far more in Python | Poor: explicit, dt ~ 1e-3·Δl … 1e-2·Δl | Poor: Rodrigues exp/log (sin, cos, acos), sqrt per element, tiny dt → huge integer range | High (or wrap PyElastica, MIT) — research tool, not runtime |
| **XPBD centre-line chain with bend (rest-curvature) constraints + substeps** | Good if the controller moves a stiffness/rest-curvature front (Yekutieli's 2-parameter reach); bend shape authored by the front, dynamics (lag, trailing, overshoot) emergent | Yes: internal constraint corrections are mass-weighted and sum to zero → momentum conserved; travel only from drag, friction, anchors; λ/dt² gives forces | ≈ 35 constraints × 4–8 substeps ≈ 150–300 projections ≈ 5–20 µs | Excellent: implicit-like, stable at any dt; accuracy from substeps | Good: precomputed reciprocals (constant denominators at fixed dt and masses), Jakobsen sqrt-free stick or squared-length constraint, int64 sums; |v| for drag needs one integer isqrt per segment | Low–medium; reference C++ in PositionBasedDynamics (MIT) |
| **XPBD Cosserat rod with orientations (Kugelstadt–Schömer / VIPER)** | As above + true twist (Tekinalp's twist-then-wrap grasp) and, with VIPER, stretch-thins-the-arm | Yes | ≈ 2–3× the chain (quaternion per segment, two constraint types) ≈ 15–50 µs | Excellent | Medium: quaternion renormalisation needs an rsqrt per segment per substep (integer Newton, fixed iterations) | Medium; code in PositionBasedDynamics (MIT), VIPER site |
| **Verlet chain + Jakobsen relaxation (PBD without λ)** | Fair: rest-angle animation gives a bend; stiffness depends on iteration count (fixed in our case, so tunable once) | Approximately (mass-weighted projections conserve momentum; no force readout) | Cheapest: ≈ 3–10 µs | Very good | Excellent: one division per constraint, no sqrt (Jakobsen's Taylor trick) | Lowest |
| **Kinematic FABRIK / CCD + authored bend wave** | Looks right only as authored; no lag, no trailing, no response to hits | None (body travel must be scripted, as in CyberOctopus) | ≈ 1–5 µs (FABRIK 15 iterations to a reachable target, 0.86 ms per iteration in MATLAB) | n/a (no dynamics) | Good (one sqrt + one division per joint per pass) | Lowest, but every behaviour (limp, drag, wrap) is extra authoring |
| **Yekutieli-style muscle-strand mass-spring (2 strands in 2D, 4 in 3D), explicit springs** | Very good: the stiffening wave is emergent from activation-controlled springs + constant area/volume | Yes | Explicit: stiff muscle springs force small dt → many substeps; ≈ 50–200 µs | Poor explicit; Good if the springs are XPBD compliant constraints | Good (multiply-adds; area via cross products, no sqrt) | Medium |
| **Muscle-strand topology solved with XPBD (hybrid)** | Very good, emergent (activation → compliance + rest length per strand; transverse/area constraints elongate) | Yes | 4 strands × 18 nodes ≈ 250 constraints × 4–8 substeps ≈ 30–100 µs | Excellent | Good | Medium–high; no direct published precedent (closest: VIPER bundles, Tekinalp 197 rods) |

### S2. Recommended split: cheap deterministic sim chain, richer float render

**Sim side (deterministic, Q16.16, in PhysicsEngine C++, one call line in the tick conductor):** an XPBD centre-line chain per arm, **17 segments / 18 nodes = one node per game bone** (no resampling, no fitting; 18 × 8 = 144 nodes per octopus; pos + prev pos = ~860 int32 of synced state), base node attached to a sim-side mantle body (position, heading; z for lift), solved with a FIXED number of substeps (start at 8 at 30 Hz ≈ 4 ms substeps; 16–20 at 10 Hz) × 1 Gauss-Seidel pass in a fixed constraint order. Constraints:
1. **Stretch** (distance) per segment, rest length = l0·(1 + stretch_activation·0.4) — the transverse-muscle "TM" (Chang 2021, Tekinalp 2024; our `stretch` float, 1.40 max), compliance low.
2. **Bend** per interior node as a vector target in the frame of the previous segment: C = (x_{i+1} − x_i) − R(κ_i, n_i)(x_i − x_{i−1}), with the rest rotation R from the joint's rest curvature κ_i and bend axis n_i (integer `cos_q16`/`sin_q16`, evaluated once per substep, not per iteration). Its gradients sum to zero over the three nodes, so internal muscle action can never translate the octopus — travel comes only from external forces (goal 2 holds by construction). Compliance α_i(t) is the "co-contraction" knob: low = stiff, high = limp.
3. **Base attachment** (arm root to mantle), two-way, so arm strokes push the body and the body drags the arms.
4. **Contact**: per node, SLIDING (Coulomb, μ_k ≈ 0.1–0.3; optional fwd/back/side anisotropy 1 : 1.5 : 2) or ANCHORED (attachment constraint = sucker; pops when λ/dt² > F_grip).
5. **Drag** (velocity-level, after the position solve): per segment anisotropic quadratic drag, F_N = ½ρ(2R·ds)C_N v_N|v_N|, F_T = ½ρ(2πR·ds)C_T v_T|v_T| with C_N ≈ 1.0–1.2, C_T ≈ 0.02 (Yekutieli/Vogel; Kazakidi), applied SEMI-IMPLICITLY (v ← v/(1 + dt·c·|v|/m) per component) so it never reverses velocity at large dt. ρ = 1025 in water; ρ_air = 1.2 on deck (drag ≈ nil, gravity dominates).

The **controller** turns the 7 pose floats into this chain's actuation: the floats stop being the pose and become the INTENT. A reach = (amplitude, travel time) per Yekutieli II: a front s_f(t) advances from base to tip at L/T_travel; joints proximal of s_f get rest curvature 0 and low compliance (stiff, straight), joints near s_f get the bend (rest curvature from `curl`/`tip curl`, axis from `bend direction`/`lift`/`swing`), joints distal stay compliant and are carried by inertia + drag — so the stiff-arm-behind-a-travelling-bend look (goal 1) is half authored (front position) and half emergent (lag, trailing, overshoot). LIMP/DEAD (goal 3) = all compliances high, rest curvature 0: the arm drags and trails from drag + friction alone. WRAP/HOLD (goal 4) = Shih 2023's contact reflex: on contact, a wave toward the tip raises the bend rest curvature toward the contact side and anchors touching nodes; a wave toward the base stiffens. CRAWL (goal 2) = anchor distal nodes, then elongate (TM) or pull (LM, shorten rest lengths) — Calisti 2011's push and the pull variant. SWIM = sculling strokes with fast power / slow recovery (β ≈ 3) and the web as extra normal-drag area between adjacent arms in the power stroke, plus a mantle jet impulse; the arms trail passively during the jet.

**Render side (float, non-synced):** read the 18 sim nodes per arm (dequantized at the render read), drive the game's 17 bones directly from them (bone i spans node i → i+1; roll from the bend axis + the `twist` float, which stays kinematic). Refine for the eye only: (a) upsample to the hero model's 32 segments by a centripetal Catmull-Rom through the nodes; (b) add render-only secondary detail that the sim need not carry — tip curl inside the last segment, sucker-side roll, a small high-frequency undulation, stretch-thinning of the radius (VIPER's insight, as a skinning scale from the node spacing). (c) OPTIONAL later: a render-only float XPBD chain at 32 segments softly attached to the sim nodes (attachment compliance > 0), for tip whip and jiggle — never read back. Interpolation between ticks (10–30 Hz sim, 60 Hz render) is a plain lerp of node positions, which is why the sim nodes must be the bones.

Why this split: the sim must own whatever moves the octopus (goal 2) and whatever other units can feel (a held marine, goal 4), so those are synced integers; everything that only looks (twist, tip curl, sucker roll, high-frequency detail) stays render-side float and costs no digest. XPBD is the only family here that is stable at a 30–100 ms step, integer-friendly (constant denominators, no transcendentals per iteration), gives constraint forces for grip and push bookkeeping, and has a GPU precedent at 100 octopuses (VIPER). Cosserat/PyElastica stays OFFLINE as the reference: run one arm in PyElastica with the same actuation (bend front, stretch) and compare tip trajectories and bend-point velocity against the XPBD chain — a property gate, not a runtime system.

Estimated cost: 8 arms × ≈ 10–20 µs ≈ 0.1–0.2 ms per octopus per tick on one core (well under 1 ms per ARM, the brief's ceiling); GPU later = one thread per arm, serial Gauss-Seidel inside the thread (no atomics → deterministic), arms of one octopus coupled through the mantle by a per-substep reduction.

Caveats to settle with Erik before design: (a) is the octopus in water rooms, on dry deck, or both (changes which forces dominate: drag vs gravity + friction); (b) does any sim rule (damage, hit boxes, vision, grip) need arm geometry, or only the mantle position — if only the mantle, the arm chain could be render-only and the mantle moved by a simpler stroke model (cheaper, but goal 2 then becomes a scripted impulse per stroke, as in CyberOctopus); (c) twist in the sim (needed only for Tekinalp-style twist-then-wrap grasps; a planar or rest-curvature wrap may be enough).

### S3. Drag, buoyancy and ground-contact model I would use

- **Water drag**: per segment, Taylor/Morison resistive form, quadratic, anisotropic: F_N = ½ρ·(2R·ds)·C_N·v_N|v_N|, F_T = ½ρ·(2πR·ds)·C_T·v_T|v_T|, C_N ≈ 1.1, C_T ≈ 0.02 (Re ≈ 1000; Yekutieli via Vogel: c_per = 1 + 10 Re^−2/3, c_tan = 0.64 Re^−1/2 — verify in Erik's copy), relative to the local water velocity (zero, or a room current); semi-implicit per component. Kazakidi's fit (F_N ∝ C_p1 sin²α + πC_f1 sin α) is the refinement if the simple form looks wrong. Web: extra normal-drag area between adjacent arms, scaled by the web-spread state. Added mass: skip on arms; on the mantle, include m_a ≈ ½ρV (sphere) in the mantle's inertia, and the jet thrust ρ(dV/dt)²/A_funnel.
- **Buoyancy/gravity**: per node m·g·(1 − ρ_fluid/ρ_body), ρ_body ≈ 1045; in water ≈ 2 % of weight (arms float, lift is cheap); in air the full weight (arms sag and drag, lifting costs stiffness). A CoB offset on the mantle (Calisti 2015) as a per-octopus tuning value.
- **Ground**: floor plane z = 0 in the sim (the deck), penalty-free XPBD contact (project z ≥ R(s)); Coulomb friction with μ_k ≈ 0.2 (deck) applied as a position correction limited by μ·λ_N (standard PBD friction); suckers as anchor constraints with a grip-force limit; walls via the GameMap's blocking tiles as half-plane contacts.

### S4. Papers Erik should fetch (university access), in priority order

1. Yekutieli, Sagiv-Zohar, Aharonov, Engel, Hochner & Flash 2005, J Neurophysiol 94:1443–1458, DOI 10.1152/jn.00684.2004 — model equations, drag coefficients, dt (verify the Vogel forms and c_tan ≈ 0.0256).
2. Yekutieli, Sagiv-Zohar, Hochner & Flash 2005, J Neurophysiol 94:1459–1468, DOI 10.1152/jn.00685.2004 — the 2-parameter reach controller.
3. Kugelstadt & Schömer 2016, SCA, DOI 10.2312/sca.20161234 — the PBD Cosserat constraints (if twist is wanted in the sim).
4. Calisti et al. 2011, Bioinspir Biomim 6:036002, DOI 10.1088/1748-3182/6/3/036002 — pushing-based crawl mechanics.
5. Calisti, Corucci, Arienti & Laschi 2015, Bioinspir Biomim 10:046012, DOI 10.1088/1748-3190/10/4/046012 — lumped underwater-walking model (ground + sculling + buoyancy).
6. Sfakiotakis, Kazakidi & Tsakiris 2015, Bioinspir Biomim 10:035005, DOI 10.1088/1748-3190/10/3/035005 — lumped sculling model with numbers (OA manuscript exists but is 24 MB).
7. Tekinalp et al. 2024 PNAS SI Appendix (OA, DOI 10.1073/pnas.2318769121) — moduli, dt, drag coefficients, activation tables (Table S6).
8. Müller et al. 2007, J Vis Commun Image Represent 18:109–118, DOI 10.1016/j.jvcir.2007.01.005 — PBD friction/collision details.
9. Giorgio-Serchi, Arienti & Laschi 2013, IEEE/ASME TMech 18:484–493, DOI 10.1109/TMECH.2012.2220978, and Weymouth & Triantafyllou 2013, JFM 721:367–385, DOI 10.1017/jfm.2013.65 — the jet stroke.
10. Optional: Deul et al. 2018 CGF DOI 10.1111/cgf.13326; Umetani et al. 2014 SCA; Bergou et al. 2008 TOG DOI 10.1145/1360612.1360662; Taylor 1952 DOI 10.1098/rspa.1952.0159; Calisti, Picardi & Laschi 2017 review, J R Soc Interface 14:20170101, DOI 10.1098/rsif.2017.0101 (OA, but blocked to the fetcher).

### S5. Open-source code worth reading

| Repo | What | Licence |
|---|---|---|
| github.com/InteractiveComputerGraphics/PositionBasedDynamics (Jan Bender) | C++ (Eigen, pybind11 `pyPBD`) PBD/XPBD library: XPBD constraints, Kugelstadt–Schömer Cosserat stretch-shear / bend-twist constraints, Deul 2018 direct stiff-rod solver with demos | MIT |
| github.com/GazzolaLab/PyElastica (docs cosseratrods.org) | Reference Cosserat solver (Python/numba): muscle torques, anisotropic friction plane, slender-body drag, joints | MIT |
| github.com/GazzolaLab/elasticapp | Elastica++ C++ (2026), throughput-optimised | not stated in the paper |
| github.com/skim0119/gym-softrobot | Gym envs on PyElastica incl. an 8-arm octopus (`envs/octopus/build.py`), `OctoMuscleCrawl-v0` with suction channels | MIT |
| github.com/hanson-hschang/COOMM | Chang et al.'s octopus muscle model (LM/TM/OM as forces and couples) on PyElastica | MIT |
| github.com/mattialab/elastica, github.com/mattialab/musculoskeletal-elastica | Original Elastica code (2018) and the musculoskeletal extension (2019) | not checked |
| vcg-uvic.github.io/viper | VIPER project page (code release announced in the paper) | not checked |

### Status summary

FULL text read: Gazzola 2018 (arXiv version), Zhang 2019, Naughton 2021, Chang 2021, Tekinalp 2024 (main text), Shih 2023, Engel 2006, Jakobsen 2001, XPBD 2016, VIPER 2019, Small steps 2019, FABRIK 2011, Kazakidi 2015 CMBBE, Kazakidi 2015 Comput Fluids — 14 papers. PARTIAL: Elastica++ 2026, gym-softrobot source. ABSTRACT only: Yekutieli I & II, Chang 2020, Kugelstadt–Schömer 2016, Deul 2018, Umetani 2014, Bergou 2008, Müller 2007, Taylor 1952, Lighthill 1971, Calisti 2011 / 2015 / 2017, Cianchetti 2015, Sfakiotakis 2015 / IROS 2014, Giorgio-Serchi & Weymouth 2016, Weymouth & Triantafyllou 2013, PoseiDRONE papers.
