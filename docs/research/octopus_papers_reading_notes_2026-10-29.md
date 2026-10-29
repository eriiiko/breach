# Octopus papers — Claude's reading notes (full texts, 2026-10-29)

Read in full by the lead (Claude), from `docs/papers/octopus/`, after the four
agent digests (`octopus_*_litsearch_2026-10-29.md`). These notes record what the
PAPERS say with numbers, and a line per paper on what it decides for the build.
Erik fetched the PDFs through Lund University.

## 1. Sumbre, Gutfreund, Fiorito, Flash, Hochner 2001 — Science 293:1845 (FULL)

Arm extension (reach) evoked in denervated / amputated arms by electrical
stimulation of the axial nerve cord (40–90 Hz, 200–500 ms) or by touching the
skin or suckers; the evoked movements are kinematically identical to natural
reaches, so the reach program lives in the arm.

- The bend (point of maximum curvature) travels in ONE PLANE (R² = 0.96 ± 0.06,
  SE 0.34 cm over 12.5 ± 4.7 cm travelled), along a lightly curved path.
- Bell-shaped tangential velocity of the bend point; peak 7–60 cm/s evoked,
  9–61 cm/s natural; the NORMALISED profile (by peak, duration, distance) is
  invariant; variance is lowest in the acceleration phase.
- The bend is dorsally oriented: the suckers point in the direction of travel.
  The arm proximal to the bend stays extended (straight).
- A PASSIVE bend (a dead arm whipped against a barrier) also travels, but its
  velocity decelerates monotonically — the bell shape is the signature of active
  muscle propagation. EMG: a burst just before the bend reaches the electrode,
  then tonic activity once that part is extended.
- Aiming = setting the initial posture before (or with) the extension command;
  same starting posture ⇒ same path.
- Only ~10 % of evoked movements were bend propagations; the rest were local
  contraction, stiffening, waving (a vocabulary of other outputs exists).
- The arm's own nervous system: ~5·10⁷ neurons per arm, ~4·10⁵ motor neurons,
  ~300 ganglia; the brain sends "global commands to activate and scale the
  program variables".

Build: the reach primitive = (plane, launch posture, amplitude, speed); one
canonical normalised velocity curve (accelerate early, long tail) scaled in
time and distance; arm straight behind the bend; suckers lead. A passive arm
gets the monotone-decelerating version for free from the chain's damping.

## 2. Sumbre, Fiorito, Flash, Hochner 2005 — Nature 433:595 (FULL)

Fetching (object grasped somewhere on the arm → mouth): the arm forms a
quasi-articulated structure of three bends — proximal (the arm base), medial,
distal (at the grasp site).

- Segments L1 (base→medial) and L2 (medial→distal) stay constant during the
  move; L3 (beyond the grasp) tends to shorten. L1 ≈ L2 (slope 0.8 ± 0.15,
  R² 0.78, n = 36).
- Segment lengths scale LINEARLY with where the object was grasped: the
  structure is re-formed per movement from the grasp site.
- The structure rotates mainly about the medial bend to bring the distal bend
  to the base, then rotation about the distal bend brings the object to the
  mouth. The figure's sequence spans 0 → 1.64 s.
- EMG: the segments are actively stiffened; activity correlates with segment
  length (stabilising against water drag).
- ~100 filmed trials; evoked by placing food anywhere on the arm.

Build: the fetch primitive = grasp site s ⇒ elbow at s/2 (two equal stiff
segments), rotate about the elbow (~1.5 s), then the wrist. A "hand" is a
sucker patch, not the tip.

## 3. Sumbre, Fiorito, Flash, Hochner 2006 — Curr Biol 16:767 (FULL)

Kinematics and EMG of 24 fetches (5 animals) + 58 EMG fetches (10 animals).

- Medial joint rotates 100.1° ± 8° (robust, always increasing); proximal
  11.6° ± 11° (first decreasing then increasing in 16/24); distal 37.9° ± 27°
  (variable, only in the last phase, bringing food to the mouth).
- Control is in JOINT space: q1–q2 either one straight line (8/24, slope ≈ −18.7
  deg/deg... reported −18.72 ± 17.5) or two straight lines with an abrupt slope
  change (16/24) = two submovements; joint-angle curves have far lower variance
  (0.028) than the Cartesian paths of the food (0.072) or the distal joint
  (0.052). Bell-shaped angular-velocity profiles, single- or double-peaked.
- 75 % of single-line moves were horizontal; 83 % of Cartesian-straight moves
  were vertical (> 45° to the floor).
- Two muscle-activation waves travel TOWARD each other: proximal→distal along
  L1 (+0.08 ± 0.08 s/cm), distal→proximal along L2 (−0.05 ± 0.05 s/cm); the
  medial joint forms ~0.1 s after EMG onset on either side, where the waves
  collide. So the elbow position is set by two wave speeds, not computed.
- Unlike the reach, fetching NEEDS the brain (lesions abolish it).

Build: the fetch is driven at the joint level (q2 sweeps ~100°, bell-shaped,
sometimes as two sub-sweeps), not by steering the object along a line. For the
animation: two stiffening fronts meet mid-arm (an authored visible beat), then
the elbow rotates.

## 4. Levy, Flash, Hochner 2015 — Curr Biol 25:1195 + SI (FULL, 19 pp)

Crawling kinematics of 9 adult O. vulgaris (120–550 g; mantle-tip→head 75–200 mm;
example arm 335 mm), filmed from below at 25 fps in a tank with water to half
the animal's height (crawl only).

- Crawl direction relative to body facing: any direction; |direction| is a
  gamma distribution with MODE ≈ 45° (mean ± SD 4.8° ± 88.3°). Body rotation
  (SD ≈ 25°/2.8 s) and crawl-direction change (SD ≈ 45°/2.8 s) are independent
  (r ≈ 0.12).
- Each arm has a FIXED pushing direction set by its position on the crown; the
  body moves along the UNWEIGHTED vector sum of the pushing arms' directions
  (r = 0.94 over 1636 frames). ⇒ all pushing arms apply equal thrust; the
  controlled variable is only WHICH arms push. Pushing directions (body frame,
  0° = forward, + = right): L1 157.5°, L2 112.5°, L3 67.5°, L4 22.5°, mirror
  for R (−). Arm i pushes at ±(202.5 − 45·i)°.
- Arm use: hind arms 1516 vs front arms 990 counts; 3rd > 4th (810 vs 706).
  Favoured pairs: L3+L4, L4+R4, L2+R4, L3+R3, L4+R2, R3+R4 (⇒ the ±45° mode);
  opposite-pushing pairs (L1+R4, L2+R3, L3+R2, L4+R1) NEVER together.
- NO RHYTHM: FFT of mouth velocity has no characteristic peak (insects do);
  step durations × mean speed pass a runs test for randomness (84 steps,
  p = 0.30; insects fail it). Multi-arm recruitment's SPATIAL pattern is not
  random (the animal does go where it wants).
- The push step (Fig S2, one arm, 3.6 s): (1) the proximal arm SHORTENS,
  drawing the suckers toward the mouth; (2) a group of ADJACENT suckers
  (22 & 24 of ~300, i.e. mid-arm) anchors; (3) the segment between mouth and
  anchor ELONGATES for 0.7 s (2.0–2.7 s), pushing the body; suckers distal to
  the anchor move passively with it (only the proximal part elongates); no
  consistent order within the elongating segment. Body speed within a step
  peaks ~100 mm/s; mouth velocities in Fig 4 range 0–140 mm/s.
- Body ROTATION, two strategies, both from a lateral bend with suckers distal
  to the bend anchored and the proximal part stiffened straight: (a) the bend
  angle increases, giving radial thrust that pushes and rotates together; or
  (b) the anchoring suckers release proximal→distal while the proximal segment
  straightens.
- Elongation is presumably a global transverse-muscle activation — a simple
  command — fitting the 170 M (brain + optic lobes) vs 330 M (arms) neuron
  split; the brain acts like the superior colliculus choosing saccades:
  moment-to-moment recruitment, "body saccades".

Build: crawl = a hub that picks the SET of pushing arms; the body velocity is
Σ unit pushing directions × one thrust; a step = shorten → anchor mid-arm →
elongate proximal (0.7 s push in a 3.6 s cycle) → release; recruit hind arms
more, never opposed pairs; no clock. Turning is a separate arm behaviour (bend
+ anchor + stiffen, then open the bend or peel the suckers).

## 5. Yekutieli, Sagiv-Zohar, Aharonov, Engel, Hochner, Flash 2005 — J Neurophysiol 94:1443, part I (FULL)

A 2D dynamic model of the arm: 2n point masses in n pairs (ventral/dorsal),
connected by longitudinal and transverse "muscles" (damped springs, linear or
nonlinear Hill-type); each quadrilateral compartment keeps CONSTANT AREA
(volume) via a per-compartment pressure found as a Lagrange multiplier;
forces: muscle + gravity/buoyancy + water drag + constraint. Integrated with
adaptive RK(2,3) in Matlab.

- Arm: 240 mm long, 15 mm wide at base, 1 mm at tip; 20 segments (40 made
  little difference; +0.05 s shift). Densities: arm 1042, seawater 1022 kg/m³
  (nearly neutral; a dead arm sinks slowly).
- DRAG, measured on a real amputated arm dropped in a tank: quadratic in
  velocity, c_per = 1.013 ± 0.055 (a cylinder in cross-flow at Re ~ 10³),
  c_tan = 0.0256 ± 0.0017 — perpendicular ≈ 50 × tangential.
  d_per = ½ρ·(2rl)·c_per·v_per², d_tan = ½ρ·(2πrl)·c_tan·v_tan². The model is
  2.5 × more sensitive to c_tan than to c_per; with both ≈ 0 the tip
  OVERSHOOTS; low c_tan makes the distal arm WIGGLE; high c_tan leaves the arm
  curved at the end (not enough force to straighten). Reach velocities 20–60
  cm/s, Re ≈ 3000.
- Muscle: peak active stress 8·10⁴ N/m² in the model (measured 4·10⁴ on
  octopus arm longitudinal muscle; squid mantle 2.6·10⁵); v_max = 10 lengths/s;
  relaxed length 0.8 l₀; linear model: T_total 1.34·10⁵ N/m², passive 2000
  N/m², damping 9 N·s/m²; muscle lengths stay in 0.7–1.1 l₀ so linear ≈
  nonlinear and 10 × faster to simulate.
- THE ACTIVATION: a uniform STIFFENING WAVE — all muscles of a segment
  activated equally (control parameters drop from 3n+2 to n+1, in practice
  to THREE: amplitude a_ca, per-segment delay τ, phase) — as a sigmoid in
  time ½(1 + tanh(t/τ − i − i₀)) travelling base→tip at CONSTANT velocity,
  activation travelling time t_act = τ·n. Constant signal velocity reproduced
  natural reaches (bend speed range, full straightening); a bell-shaped
  (minimum-jerk) signal velocity usually did NOT straighten the arm (its peak
  was too fast to propel the bend). Sustained activation behind the front can
  be lower than maximal and still stiffens the proximal arm.
- Reach from the natural start pose (arm curled dorsally near the body,
  ventral side up); t_act 0.6–1.2 s at a_ca ≈ 0.6 spans the natural velocity
  range; bend path nearly straight (natural paths curve slightly because the
  BASE rotates during the reach — not modelled).
- The bend-point tangential velocity has 3 phases: (1) small decrease / a
  local minimum, (2) monotonic rise, (3) plateau then decrease; terminating
  the activation earlier (fading over ~0.4 of the arm length) gives the larger
  natural phase-3 decrease. Over a whole reach, total perpendicular and
  tangential drag work are about equal — the reach is drag-efficient because
  only the bend region moves broadside.
- Equal dorsal/ventral activation straightens; a ratio off 1 leaves a curve;
  longitudinal:transverse force ratio 0.5–2 changes final length only ±10 %.
- Model limits: free movement only (no objects, no floor), planar.

Build: THIS is the arm model to copy in spirit — a chain with a per-segment
stiffness that is swept base→tip at constant speed is the reach; one
amplitude + one travel time are the whole command. Anisotropic quadratic drag
with c_per/c_tan ≈ 40–50 is load-bearing: without it the tip overshoots and
wiggles. Constant-volume ⇒ a stretched segment thins (CONTROLS.md's stretch
law already does 1/√length).

## 6. Yekutieli, Sagiv-Zohar, Hochner, Flash 2005 — J Neurophysiol 94:1459, part II (FULL)

Control of the reach with the part-I model. The command has TWO parameters:
activation amplitude a_ca and activation travelling time t_act (equivalently
wave speed V = L/t_act).

- Why equal activation STRAIGHTENS: with constant area, a segment whose four
  muscles are equally activated converges to its relaxed (square) shape — the
  unique minimum of potential energy (Appendix A, B). So the activated
  proximal arm is straight and stiff, and the bend is pushed ahead of the
  front by force transfer into the not-yet-activated segments.
- Fixing a_ca and varying t_act: big change in PEAK bend velocity, small
  change in time-to-peak; the peak bend velocity ≈ the wave's own velocity
  (the muscles accelerate the bend until it rides the command). E.g.
  t_act 0.9 s ⇒ V = 0.28 m/s.
- Fixing t_act and varying a_ca (0.375–1.0): velocity profiles nearly
  IDENTICAL — kinematics is independent of force above a threshold. The
  threshold is the MINIMUM-FORCE CURVE in (t_act, a_ca) space (arm straight to
  ≤ 2° mean local angle); below it the arm ends curved, and with a very short
  t_act no physiological force can push the bend: the whole arm opens
  uniformly ("open arm extension", also evoked in real arms by intense
  stimulation; the nerve cord conducts at ~3 m/s).
- Peak bend velocity vs amplitude along the minimum-force curve is a slightly
  curved positive relation — matching Gutfreund 1998's EMG-vs-peak-velocity.
- Extra force above the curve does not change kinematics but RESISTS
  PERTURBATION (a constant up/down force field): a cocontraction-like
  stiffness dial independent of the kinematics.
- In the real arm the EMG front leads the bend by ~100 ms (in the model the
  bend leads the front) — excitation-contraction delay and perhaps local
  feedback from mechanoreceptors.
- The four parameters the brain sets for a reach (their words): two base
  orientation angles, the wave speed, and the force level; termination when
  suckers touch the target (stop the wave = catch). Any part of the ventral
  side touching the target suffices — the grasp need not be precise.

Build: the reach primitive's interface is exactly (aim yaw, aim pitch, wave
speed, stiffness) + a stop event on contact. Stiffness is a separate dial
from speed: a "tense" octopus reaches with the same motion but shrugs off
water/wind/impact. The failure modes are authored looks too: too little force
= a lazy curved reach; too fast a wave = the whole arm opening at once (a
flinch / startle).

## 7. Schwartz 2015 — MSc thesis, Texas A&M, "Creating procedural animation for the terrestrial locomotion of tentacled digital creatures" (FULL, 52 pp)

A PhysX prototype: mantle = cylinder, each tentacle = a chain of capsule links
(tapered by `scale_i = 1 − i/(n−1)·(1 − end_scale)`), spherical joints with
angular motors driven by acceleration springs toward TARGET ORIENTATIONS that
are sine functions of time (amplitude, frequency, phase) — the Davy Jones
pattern. Suckers = distance constraints created/destroyed on contact.

- Locomotion: a goal (point + radius; a list = a path). One DRIVING tentacle at
  a time; push vs pull decided by the sign of (mantle→goal)·(mantle→tentacle
  base): tentacle between mantle and goal ⇒ PULL (extend, fix tip, compress),
  otherwise PUSH (compress, fix tip, extend). Each tentacle runs an ACTION
  QUEUE of (timer, target angle, strength); the timer sets speed and power.
  No goal ⇒ all tentacles idle-writhe (sine on motor strengths too).
- Cost: four creatures × six tentacles at ≥ 15 fps on a 2009 i7 + GTX 280.
- Its own verdict: locomotion "chaotic", a predetermined cyclical gait; future
  work = recruit tentacles by proximity/suitability to the goal, re-decide on
  the fly instead of a fixed queue, more intermediate actions. Not hero
  quality; a crowd tool.
- Case studies confirm: ILM Davy Jones = rigid chains + joint targeting +
  stiction springs; R&H Night at the Museum 2 = animator keys, a cloth strip
  per arm tracking the animated REST SHAPE in its bending term (so the sim
  "behaves actively"), suckers = springs released on energy or distance.

Build: the negative example we needed. Fixing the tip and driving joint angles
with sines gives exactly the "spider gait / tip-pulls / wiggle" failures the
biology digest's §4 lists. What it got right and we keep: goal lists with a
radius, push-vs-pull by geometry, an action queue per arm with a timer,
anchors as constraints, and the mantle as the coordinator. What we do
differently: mid-arm anchors + proximal elongation (Levy), recruitment by
alignment lottery (strand d), stiffness waves not joint-angle sines.

## 8. Criswell, Derlich, Hatch (ILM) 2006 — SIGGRAPH sketch "Davy Jones' Beard: Rigid Tentacle Simulation" (FULL, 1 p, read as an image)

46 tentacles, 492 rigid bodies, 446 point joints, Stanford PhysBAM articulated
rigid-body solver inside Zeno; 200+ shots. A per-tentacle controller (driven
mainly by head motion and collisions) plus one master multiplier controller
for the whole beard. Curl = joints motorized by a sine with amplitude,
frequency, time; parameters set ALONG the tentacle by a 2D curve (x = position
on the tentacle, y = value). Joint control = force-based targeting: torque
from the difference between target and current orientation. Keyframed shots:
rigid bodies follow the animation and act as collision geometry. "Stiction":
springs auto-created/destroyed between tentacle bodies and surfaces, from
area maps on the tentacle's underside, with create/destroy distances.

Build: the per-tentacle controller + master dial, parameters as a curve along
the arm, and stiction regions on the underside are all reusable ideas; the
sine curl is not.

## 9. Boucher, San Juan Pallares, Puertolas, Inagaki (DreamWorks) 2025 — DigiPro "Rubber Hose Revival: the Curvy-Limb rig" (FULL, 4 pp)

A rigging interface, not a simulator: a standard arm/leg skeleton is resampled
into a high-resolution joint SPLINE (50 joints per limb, 25 per digit) and
posed with few controls — "curviness" (U vs V elbow), "sharpness" (tangent
handle distance), start/end "aim" + "sharpness", secondary offset controls
that RIDE ALONG the curve with a location and a falloff (Variable FK), twist
along the curve, decoupled scale with volume preservation, slideable parent
joints, a post-process WAVE package (amplitude, frequency, phase travelling
down the appendage), and a SLIDE that grows/retracts the limb along its own
curve by moving the spline's end parameter. Used on the kraken family (Ruby
Gillman), Vontra the five-tentacled squid robot (each tentacle as an arm with
a reversible elbow, a colour-debug mode to tell tangled tentacles apart, a
primary jack moving all tentacles together).

Build: this is the RENDER-side vocabulary: spline through our solved nodes,
offset handles with location + falloff for authored shape, a slide for
"retract along own curve", twist along the curve, and a colour-debug mode for
eight arms (we will want it in the tile inspector). Also the DreamWorks
octopus-rig paper to fetch: Boucher & Dirksen 2015, DigiPro,
doi 10.1145/2791261.2791274.
