# Octopus animation & locomotion study — synthesis, architecture, prototype plan

**Status: STUDY BRIEF v1, 2026-10-29.** Issue #105, milestone 8. Author: Claude
(lead designer/developer of the octopus; the `octopus-lead` skill holds the
mandate). Erik picks at the end of the prototypes; the decisions marked
*(lead)* are mine and vetoable, the ones marked *(Erik)* are his.

Inputs: four literature digests (`docs/research/octopus_{motor_biology,
soft_arm_sim,games_film,cpg_coordination}_litsearch_2026-10-29.md`), nine
papers read in full by me (`docs/research/octopus_papers_reading_notes_
2026-10-29.md`, PDFs in `docs/papers/octopus/`), the existing model and its
control contract (`art/characters/octopus/CONTROLS.md`), and Erik's rulings on
#105 (tiers, game-state split, sequential research, carried prop).

---

## 1. What must be true (acceptance)

Tier 1, confirmed by Erik 2026-10-29:

1. **A swim stroke that really travels** — forward motion comes from the
   stroke (mantle jet + umbrella close) through drag and reaction, never a
   slide added on top.
2. **A separate crawl gait**, much slower than swimming, on the arms.
3. **Every arm independent and purposeful** — no arm just follows along;
   but only 2–3 arms *act* at any moment (biology AND Pixar agree).
4. **Reach → grab → hold → lift a marine-shaped prop and carry it** while
   moving. (The real unit coupling is #107.)
5. **Idle that reads as alive** — arms exploring surfaces, tip curls,
   mantle breathing, a visible hold before each decision.
6. **Graceful with fewer arms** (#118: a dead arm goes limp and the rest
   adapt) — no gait may assume eight.

Every clip is judged against (a) reference footage of real octopuses
(`docs/research/octopus_motor_biology_litsearch_2026-10-29.md` §6) and (b)
the 14-item craft checklist from the games/film digest (`octopus_games_film_
litsearch_2026-10-29.md`, synthesis §2), and (c) the ten "what a game gets
wrong" items (motor-biology digest §4). Erik's eye is the final gate.

---

## 2. What the literature settles

### 2.1 The animal's motor vocabulary (numbers from the full texts)

| primitive | what happens | numbers |
|---|---|---|
| **Reach** (Sumbre 2001; Yekutieli I/II) | A *stiffening wave* runs base→tip at constant speed; it pushes a pre-formed dorsal bend ahead of it (suckers lead); the arm behind the front is straight and stiff; ahead of it, limp. The brain sets aim (2 angles), wave speed, stiffness level; the arm does the rest. Stops on sucker contact. | Bend speed 7–60 cm/s on a 24–34 cm arm; wave travel time 0.6–1.2 s; bell-shaped bend velocity, normalised shape invariant (accelerates late, dissolves at the tip); proximal arm stretches ×1.5–1.8 and thins (constant volume). Kinematics independent of force above a threshold (the "minimum-force curve"); extra force = resistance to perturbation, not speed. Too little force = a lazy curved reach; a wave too fast for the force = the whole arm opens at once (a startle). A dead arm's passive bend decelerates monotonically. |
| **Fetch** (Sumbre 2005/2006) | Object grasped at s along the arm ⇒ three bends: base, an elbow at ≈ s/2 (two equal stiff segments, set by two activation waves colliding), a wrist at the grasp. Rotate about the elbow, then the wrist, to the mouth. Needs the brain. | Elbow sweeps ~100° ± 8°, base 12° ± 11°, wrist 38° ± 27° (last phase only); bell-shaped joint velocities, sometimes two sub-sweeps; ~1.6 s; control is in joint space, not end-point space. |
| **Push step** (Levy 2015) | Shorten the proximal arm → a patch of adjacent MID-ARM suckers anchors → the segment between mouth and anchor ELONGATES, pushing the body → release. Distal to the anchor nothing moves. Every arm pushes with the same force along its own fixed direction (arm i: ±(202.5 − 45 i)° from forward); body velocity = Σ of the pushing arms' unit directions. | One step 3.6 s, active push 0.7 s; body peaks ~0.1 m/s; lab 1–3 arms push at once, wild 3 ± 1 in contact; hind arms used 1.5× more than front; opposite-pushing pairs never together; favoured pairs give the ±45° crawl mode. **No rhythm** (FFT flat, step durations random). Release trigger = proximal strain saturates (Katija 2025). |
| **Turn** (Levy SI) | A lateral bend in one arm; suckers distal to the bend anchor; the proximal part stiffens straight; then either the bend angle opens (radial thrust: push + rotate) or the anchor suckers peel off proximal→distal while the proximal part straightens. Body heading and crawl direction are independent (r ≈ 0.12). | Body rotation SD ≈ 25° per 2.8 s. |
| **Jet** (Wells; Huffard; FORTH digests — stroke timing still unread) | Mantle fills, arms bundle straight behind, mantle contracts, siphon aims; the animal goes mantle-first. | 45–100 cm/s, a few metres (the heart stops while jetting). |
| **Umbrella / medusoid swim** (Huffard 2006; Sfakiotakis 2015) | Arms + web open slowly, close fast; the web's broadside drag on the fast stroke gives net thrust. | Open:close ≈ 2.5:1. All-in-phase, diagonal pairs ¼-period apart, and two groups of four all reach ~0.26 body lengths/s; diagonal smoothest; adjacent-pair phasing wobbles. Two arms swim more efficiently than eight (cost of transport 0.98 vs 1.63). |
| **Others** (Mather, Bidel, Alupay, Gendreau, Sivitilli) | grope/search with reel-in, arm slap, pounce, parachute (web over prey), snaptrap, sideways arm-walk, pull-in, two-arm rolling walk, bipedal walk. Front pairs explore (the 2nd arm leads strikes), back pairs support; a goal recruits the nearest arm plus its neighbours. | ~1 in 5 suckers sealed at a time; arms slide over each other freely. |

Two cross-cutting facts: (1) **the brain picks arm + primitive + direction +
speed; the arm's own circuits make the shape** (the whole Hochner programme,
confirmed by Sumbre 2001's denervated-arm reaches and by Shih 2023's finding
that learning works at the primitive-choice level and fails on muscles);
(2) **water drag is quadratic and anisotropic** — perpendicular drag
coefficient ≈ 1.0, tangential ≈ 0.026, a 40–50× ratio measured on a real arm
(Yekutieli I). Without it the tip overshoots and wiggles, and a fast-close /
slow-open stroke produces no net thrust at all.

### 2.2 How the arms are coordinated (strand d)

- **Crawl has no clock.** Arms are recruited moment to moment by how well
  each arm's fixed push direction aligns with the wanted travel direction,
  with a bias to the hind arms and to neighbours of an already-pushing arm;
  releases are load-triggered. The brittle star (Kano 2017/2019), Walknet and
  Tegotae all show that decentralised, contact-driven recruitment keeps
  working with limbs removed — no special case for a dead arm.
- **Swimming is rhythmic**: one phase clock for the animal, arms in phase or
  in two groups, steering by per-arm amplitude.
- **Learning plugs in at the top**: a per-arm module with local observations
  and a message to/from a hub (SMP) is the shape that survives lost limbs and
  can later be a Tsetlin machine or an NN (#108).

### 2.3 How to simulate an arm (strand b, Yekutieli)

- Research Cosserat rods (PyElastica, CyberOctopus) step at 10–70 µs: 700–5000
  steps per arm per 50 ms tick — out of budget, and their flagship crawl is
  scripted anyway.
- **A 20-segment mass-spring chain with a swept stiffness already produces
  the real reach** (Yekutieli I: 40 segments changed nothing). The stiffening
  wave is a property of "equal activation + constant volume", not of rod
  fidelity.
- **XPBD** (Macklin 2016/2019) is what games use: stable at large steps, one
  division per constraint, substeps × one pass beats passes; VIPER rigged 100
  octopuses in ~7 ms on a 2016 GPU. Bender's PositionBasedDynamics (C++, MIT)
  has the XPBD Cosserat rod to read.

### 2.4 What the best animation did (strand c + the three craft papers)

One pipeline, six layers, information flowing one way:
**intent → which arms act (2–3) → authored shape on a low-DOF curve →
soft chain solve with pinned contacts → bones from a spline through the
solved points → passive secondary motion that never feeds back.** Spore ran
it at 0.2 ms per 25-body creature in 2008; simulation moved Davy Jones' 46
tentacles 95 % of the time. Hank's animators led a reach from the "elbow",
never the tip (Octodad inverts this on purpose). DreamWorks' render-side
vocabulary: spline through the chain, offset handles with location + falloff,
twist along the curve, a slide to retract along the arm's own curve, a
colour-debug mode for eight arms.

Negative examples we now have in writing: Schwartz 2015 (tip-fixed push/pull
with sine-driven joints = chaotic spider gait), the Sea of Thieves kraken
(tentacles "connect to nothing"), Octodad.

---

## 3. The architecture *(lead; vetoable)*

Four layers. Sim layers are deterministic Q16.16 (`cpp/src/fixed_point.h`),
fixed substeps, integer RNG; the render layer is float and never writes back.

```
A  BRAIN / INTENT      (sim)   per arm: goal (type, target, priority, mood)
        │                      hub: drive scalar + mode (crawl | swim | hold)
B  PRIMITIVE EXECUTOR  (sim)   per arm: a state machine running ONE primitive,
        │                      emitting per-tick muscle targets per node
C  ARM CHAIN           (sim)   per arm: XPBD chain, 18 nodes (= the 17 game
        │                      bones + tip); anchors; drag; floor; the body
D  RENDER              (float) spline through C's nodes → 32 hero segments,
                               twist / tip curl / sucker roll / thinning /
                               breathing / secondary motion
```

### A. Brain and hub

- Per arm: `goal = (kind, target_xy(z), priority)`; kinds in tier 1: NONE,
  EXPLORE, PUSH(dir), REACH(target), GRAB(target), HOLD, SWIM_STROKE, BUNDLE,
  LIMP. A goal's priority drops once acted on (Trico's rule).
- Hub: `drive ∈ [0, 1]` and `mode` with hysteresis. Crawl mode keeps ~3 arms
  in stance and recruits ONE arm per decision tick by a **seeded integer
  lottery**, weight `max(0, p_i·d − s_min) × hind_bias × neighbour_bonus`
  (p_i = the arm's fixed push direction, d = wanted travel direction); never
  an opposed pair; never below 2 in stance; a refractory period per arm.
  Swim mode is one uint32 phase accumulator, 2.5:1 time-warped, all arms in
  phase (diagonal-pair phasing is a dial). Jet is a counter-driven mantle
  cycle triggered by `drive` above a threshold.
- **The per-arm brain interface** (so #108 can swap the hand-written
  preference for a Tsetlin machine / NN without re-plumbing): observation =
  (arm id, state, proximal strain, foot load, alignment, both neighbours'
  states, hub summary) as a fixed-size integer vector; action = (requested
  primitive, score). The hub arbitrates; a learned brain can change
  preferences but cannot make the animal collapse or twitch.

### B. Primitive executor

Per arm, one running primitive, each a small state machine with a timer,
emitting per node: `target_curvature`, `stiffness`, `rest_length` (the
stretch), and `anchor_request`. The catalogue, each in the paper's terms:

- REACH(aim, wave_speed, stiffness): form the dorsal bend at the base; sweep
  `stiffness` base→tip at `wave_speed` (0.6–1.2 s travel at rest scale);
  `rest_length` ×1.5 behind the front; stop on contact ⇒ GRAB.
- FETCH(s): two stiffness fronts meet at s/2; rotate the elbow ~100° as a
  bell-shaped sweep (~1.6 s); then the wrist.
- PUSH_STEP(dir): shorten proximal (rest_length ↓) → anchor nodes 7–9 (mid
  arm) → elongate proximal (rest_length ↑ over 0.7 s) → release on proximal
  strain saturation or lost foot load → refractory.
- TURN: lateral bend, anchor distal, stiffen proximal, then open the bend or
  peel the anchor base→tip.
- WRAP(target): a contact reflex — nodes touching bend toward the touched
  side and anchor to it; HOLD keeps stiffness high; LIFT raises the root
  (`lift`) with the wrap kept.
- UMBRELLA(phase): slow open / fast close of lift + curl across all arms;
  JET: bundle (curl 0, lift −1, swing to the bundle) while the mantle fires.
- EXPLORE: low-stiffness random-walk of the tip region with sucker touches
  (the idle "alive"); LIMP: stiffness 0, anchors off (#118).

The seven CONTROLS.md floats stop being *the pose* and become the
executor's intent where that is the natural handle (`lift`, `swing`,
`stretch`, `curl` as target curvature), as strand (b) proposed.

### C. Arm chain and body

- **XPBD chain, 18 nodes per arm** (one per game bone + tip), tapered mass,
  fixed `N_SUB` substeps × 1 Gauss-Seidel pass; constraints: distance
  (rest_length per segment — the muscle's stretch/shorten), bend with target
  curvature and per-node compliance (stiffness), anchor (node pinned to a
  world point, breaks above a grip limit), floor contact + friction, body
  attachment at node 0.
- **Drag**: quadratic, anisotropic, `c_per 1.0 / c_tan 0.026` on each
  segment's relative velocity, applied as a velocity-space damping that
  cannot overshoot; in air both drop to ~0 with gravity instead of near
  neutral buoyancy (per-node medium from the water field — the aquarium
  #111 has both).
- **Body**: position, heading, (z while swimming) as a rigid body moved ONLY
  by anchor reactions, drag on the web/mantle and the jet impulse; bend
  forces sum to zero inside the chain so the arms cannot push the body
  without an anchor or water — that is the "no slide" guarantee, structural.
- **Mantle**: the existing `mantle_squeeze` as a counter cycle; jet thrust
  = a law on the squeeze rate (Trueman & Packard / Wells timing once read).
- Determinism: Q16 kit only, integer substeps, kit `sqrt`/`reciprocal_q16`
  in the constraint projection, seeded `xorshift` for the lottery; the chain
  nodes, anchors, goals and HP are digested. Target cost ≤ 0.2 ms per octopus
  per tick on one core, measured, with the CUDA twin `(N, h, w)`-shaped per
  the RL-batch habits.

### D. Render

Spline through the 18 solved nodes → the hero's 32 segments
(`octopus_math.pose_octopus` keeps the skin contract); twist, tip curl,
sucker roll, thinning (1/√stretch — already there), breathing, and a verlet
secondary layer that reads C and never writes it. Contact and height cues
from above (the #114 projected shadow, contact darkening, suckers flattening);
a colour-debug mode per arm in the tile inspector.

### Game-state split (Erik: "very reasonable")

| sim (Q16, digested) | render (float) |
|---|---|
| body pose, mode, drive, mantle phase | spline, 32-segment skin, twist/sucker roll |
| per arm: 18 nodes, anchors, primitive state + timer, goal, grab (unit, arm), HP (#118) | breathing, secondary wiggle, contact darkening, debug colours |
| the lottery's RNG state | the camera's and the shadow's reads |

Richer than the "root + 2–3 waypoints + tip" first proposed, deliberately: the
chain IS the gameplay geometry, so the picture can never lie about a grab or
a hit, and the RL observation stays 8 × 18 × 3 integers.

---

## 4. The five options of #105, judged

| option | verdict |
|---|---|
| 1 Parametric cycles on the 7 floats | Not the main path (canned, no travel, "tentacle wiggle"). Kept as the A/B baseline clip and for authored flourishes on top of D. |
| 2 Physically simulated arms | **Chosen** — but an XPBD chain, not a Cosserat rod; the stiffening wave is the actuation. Travel comes from anchors and drag. |
| 3 Rhythm generators | Only for swimming (one phase clock); crawl is explicitly non-rhythmic. |
| 4 Learned controllers | Later (#108), at the primitive-choice level, on the brain interface above; never on muscles. |
| 5 Reference footage | The judging target; a watch-list with timestamps is deliverable P0. |

---

## 5. Prototype plan (each a clip on the real mid-size model, cost measured)

Designed so the make-or-break experiment comes first and can fail.

- **P0 — Footage sheet + checklist.** Watch-list with timestamps (paper
  supplements first: Katija 2025, Levy 2015, Huffard 2006, Sumbre 2005,
  Kennedy 2020, Bidel 2022; then My Octopus Teacher / Secrets of the Octopus
  / Blue Planet II), the 14-item checklist as a scoring sheet.
- **P1 — One arm, float, standalone (pure numpy).** XPBD chain + stiffening
  wave + anisotropic drag. Gates: (a) a reach reproduces Yekutieli: bell
  velocity profile, straight arm behind the bend, tip never overshoots,
  peak bend speed ≈ wave speed; (b) kinematics invariant to stiffness above
  the minimum-force curve; (c) a dead arm's bend decays monotonically;
  (d) ≤ 25 µs per arm-substep in numpy terms of operations (count, not time).
  If (a) fails on a chain, the study's recommendation changes — say so.
- **P2 — Crawl.** Eight chains + body on a floor; push step + lottery.
  Gates: travels at a wanted direction with r ≥ 0.9 between the predicted
  (Σ push directions) and actual direction; 1–3 arms pushing; FFT of body
  speed flat; still travels with 7 and 6 arms; body heading independent of
  travel direction.
- **P3 — Swim.** Umbrella + jet in water. Gates: net travel per stroke from
  drag alone (zero with linear drag — the control); ≥ 5× crawl speed; a jet
  carries a few body lengths then decays.
- **P4 — Reach, grab, hold, lift, carry** a marine-shaped prop. Gates: the
  grab closes mid-arm (not the tip), the prop stays attached through a crawl
  and a swim, the other arms re-plan around the loaded arm.
- **P5 — Idle.** Explore + breathing + decision holds; scored by the
  checklist only.
- **P6 — Sim port.** C++ Q16 chain in PhysicsEngine (one conductor slot),
  digest membership, CUDA twin at tol 0, cost measured.
- **P7 — Render.** Spline upsample into the existing skin, secondary layer,
  top-down cues, debug colours; the look shots.
- **P8 — Clips + verdict.** The five option clips (parametric baseline vs
  the chain) side by side; Erik picks.

P1–P5 can live in `prototypes/octopus_motion/` as pure numpy (the
map-editor pattern: pure core, thin shell), with P6 the port of the surviving
core. Cost ceiling to hold through P6: 0.2 ms / octopus / tick.

---

## 6. Open questions

For Erik, when they come up (one at a time):

1. **Gameplay speeds.** Biology gives *shapes and ratios* (swim ≈ 5–10×
   crawl; reach ~1 s; push cycle ~3.6 s with a 0.7 s push; crawl speed does
   not scale with body size). Our mid octopus is ~8× a real one; keeping the
   real durations and scaling distances gives crawl ≈ 0.5–1 m/s and reach
   bend ≈ 1–5 m/s. The final numbers are a feel dial *(Erik)*.
2. Whether the z axis of the chain is simulated (3D nodes) or height is
   render-only. *(Lead, provisional)*: 3D nodes in sim — lifting a marine and
   swimming need it, and the cost is the same order.

Still to fetch (not blocking): Hanassy 2015 (elongation, 10.1088/1748-3190/
10/3/035001), Trueman & Packard 1968 (10.1242/jeb.49.3.495) and Wells 1987
(10.1242/jeb.131.1.175) for the jet stroke timing, Kugelstadt & Schömer 2016
(position/orientation-based Cosserat rods), Boucher & Dirksen 2015 (DreamWorks
octopus rig, 10.1145/2791261.2791274).

---

## 7. Systems

**Existing canonical systems this design must use:** Simulation facade and
the tick conductor (one call line per new system); the fixed-point kits (CPU
+ device) for every sim arithmetic; the entity system + one serializer
(the octopus is an entity row; no pose state on `Unit`); field digest + the
A/B harness + the CUDA harness; the recorder (`DEFAULT_FIELDS` additive);
DamagePacket / status rows for #118; `species.py` for sizes (#112); the
light accessor pattern for the pose read; the look shots for every render
patch; RL-batch habits for the resident path.

**New systems it creates (draft rules, written into CLAUDE.md only when they
exist):**
- *Arm chain solver* — `cpp/src/arm_chain.*` (+ CUDA twin): THE soft-limb
  simulation (XPBD, Q16, fixed substeps). Any future limb that bends (worm,
  larva, tentacle) is a chain on it; never a second rod solver.
- *Primitive executor* — `src/simulation/octopus/primitives.py` (pure core):
  a creature's motor vocabulary is a table of primitives emitting chain
  targets; a new behaviour is a primitive row, never inline chain writes.
- *Limb hub / arbiter* — the recruitment lottery and mode machine; the ONLY
  thing that assigns primitives to arms; a learned brain only submits
  (request, score) through the per-arm brain interface.
- *Pose accessor* — the one read of the chain for render/RL (`read_pose`),
  dequantized at the read, never written back (the light accessor's twin).
